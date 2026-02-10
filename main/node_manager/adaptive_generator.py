"""
Adaptive node generator for MeshOptimiser.

Generates nodes with density that adapts to local geometry complexity:
  - Near boundaries / high-curvature regions  →  finer spacing (L_fine)
  - In smooth interior regions                →  coarser spacing (L_coarse)
  - Uniformity is maintained *within* each density zone

The key insight borrowed from Gmsh: regions with tight curvature, narrow
channels, or proximity to holes need more nodes to resolve the geometry,
while large flat interiors can use fewer nodes without losing quality.

The density field is computed from:
  1. Distance to nearest boundary  (closer → finer)
  2. Local boundary curvature      (higher curvature → finer)
  3. Medial-axis thickness          (thin regions → finer)
"""

import numpy as np
from shapely.geometry import Point, MultiPoint
from shapely.ops import nearest_points
from scipy.spatial import cKDTree
import math


# ─────────────────────────────────────────────────────────────
#  Curvature estimation
# ─────────────────────────────────────────────────────────────

def estimate_boundary_curvature(polygons, sample_spacing=0.02):
    """
    Sample boundary curves and estimate discrete curvature at each point.

    Returns:
        pts:       (M, 2) boundary sample coordinates
        curvature: (M,)   unsigned curvature at each sample
    """
    all_pts = []
    all_curv = []

    for poly in polygons:
        for ring in [poly.exterior] + list(poly.interiors):
            coords = np.array(ring.coords)[:, :2]
            # Resample at uniform spacing
            resampled = _resample_ring(coords, sample_spacing)
            if len(resampled) < 5:
                all_pts.extend(resampled.tolist())
                all_curv.extend([0.0] * len(resampled))
                continue

            # Discrete curvature via finite differences of tangent angle
            curv = _discrete_curvature(resampled)
            all_pts.extend(resampled.tolist())
            all_curv.extend(curv.tolist())

    return np.array(all_pts), np.array(all_curv)


def _resample_ring(coords, spacing):
    """Resample a polyline at approximately uniform arc-length spacing."""
    # Cumulative arc length
    diffs = np.diff(coords, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    cum_len = np.concatenate([[0], np.cumsum(seg_lens)])
    total_len = cum_len[-1]

    if total_len < 1e-12:
        return coords[:1]

    n_pts = max(4, int(total_len / spacing))
    target_s = np.linspace(0, total_len, n_pts, endpoint=False)

    resampled = np.zeros((n_pts, 2))
    for i, s in enumerate(target_s):
        idx = np.searchsorted(cum_len, s, side='right') - 1
        idx = np.clip(idx, 0, len(coords) - 2)
        t = (s - cum_len[idx]) / (seg_lens[idx] + 1e-30)
        resampled[i] = coords[idx] * (1 - t) + coords[idx + 1] * t

    return resampled


def _discrete_curvature(pts):
    """
    Unsigned discrete curvature from the circumradius of consecutive triplets.
    κ = 2 |sin(θ)| / |c| where θ is the angle at the middle vertex
    and |c| is the chord length opposite to it.
    """
    n = len(pts)
    curv = np.zeros(n)

    for i in range(n):
        p0 = pts[(i - 1) % n]
        p1 = pts[i]
        p2 = pts[(i + 1) % n]

        a = np.linalg.norm(p1 - p0)
        b = np.linalg.norm(p2 - p1)
        c = np.linalg.norm(p2 - p0)

        if a < 1e-12 or b < 1e-12 or c < 1e-12:
            continue

        # Area of triangle via cross product
        area2 = abs((p1[0] - p0[0]) * (p2[1] - p0[1]) -
                     (p2[0] - p0[0]) * (p1[1] - p0[1]))

        # Circumradius R = abc / (4 * area)  →  curvature = 1/R
        R = (a * b * c) / (2 * area2 + 1e-30)
        curv[i] = 1.0 / (R + 1e-30)

    return curv


# ─────────────────────────────────────────────────────────────
#  Density field
# ─────────────────────────────────────────────────────────────

def compute_density_field(query_pts, polygons, L_fine, L_coarse,
                          boundary_band, curvature_weight=0.5):
    """
    Compute a local desired spacing L(x) for each query point.

    The spacing varies smoothly from L_fine (near boundaries / high curvature)
    to L_coarse (deep interior / low curvature).

    Args:
        query_pts:       (N, 2) points to evaluate
        polygons:        list of Shapely polygons
        L_fine:          smallest spacing (boundary / high-curvature regions)
        L_coarse:        largest spacing (interior / low-curvature regions)
        boundary_band:   distance within which density transitions from fine→coarse
        curvature_weight: how much curvature affects density (0..1)

    Returns:
        L_field: (N,) desired spacing per point
    """
    # --- Distance to nearest boundary ---
    bdry_pts, bdry_curv = estimate_boundary_curvature(polygons, sample_spacing=L_fine * 0.5)

    if len(bdry_pts) == 0:
        return np.full(len(query_pts), L_coarse)

    tree = cKDTree(bdry_pts)
    dists, idx = tree.query(query_pts, k=1)

    # Distance-based blending: 0 at boundary → 1 at boundary_band
    t_dist = np.clip(dists / boundary_band, 0, 1)

    # Curvature-based refinement: high curvature → smaller t
    nearest_curv = bdry_curv[idx]
    # Normalise curvature to [0, 1]
    curv_max = np.percentile(bdry_curv, 95) if len(bdry_curv) > 0 else 1.0
    curv_norm = np.clip(nearest_curv / (curv_max + 1e-30), 0, 1)

    # Combined blending factor
    t = t_dist * (1 - curvature_weight * (1 - t_dist) * curv_norm)

    # Smooth interpolation (Hermite)
    t_smooth = t * t * (3 - 2 * t)

    L_field = L_fine + (L_coarse - L_fine) * t_smooth
    return L_field


# ─────────────────────────────────────────────────────────────
#  Adaptive grid generator
# ─────────────────────────────────────────────────────────────

def adaptive_grid_shapely(polygons, L_fine, L_coarse, boundary_band,
                          curvature_weight=0.5, jitter_factor=0.0, seed=42):
    """
    Generate interior nodes with spatially varying density.

    The algorithm:
      1. Create a *fine* candidate grid at spacing L_fine/2
      2. Compute the desired local spacing L(x) at every candidate
      3. Accept/reject each candidate with probability (L_fine/L(x))²
         This produces the correct point density everywhere.
      4. Apply optional jitter

    Within any region the accepted points form an approximately uniform
    distribution at the local density — critical regions get more nodes,
    ordinary regions get fewer, but both are locally regular.

    Args:
        polygons:        Shapely polygon list
        L_fine:          Finest spacing (near boundary / high curvature)
        L_coarse:        Coarsest spacing (deep interior)
        boundary_band:   Width of the refinement transition zone
        curvature_weight: How strongly curvature affects density (0..1)
        jitter_factor:   Random jitter fraction (0.0=grid-aligned, 1.0=full)
        seed:            Random seed

    Returns:
        pts: (N, 2) accepted node coordinates
    """
    rng = np.random.RandomState(seed)

    # --- Bounding box ---
    minx, miny, maxx, maxy = polygons[0].bounds
    for poly in polygons[1:]:
        bx = poly.bounds
        minx, miny = min(minx, bx[0]), min(miny, bx[1])
        maxx, maxy = max(maxx, bx[2]), max(maxy, bx[3])

    # Fine candidate grid
    h = L_fine / 2.0
    xs = np.arange(minx, maxx + h, h)
    ys = np.arange(miny, maxy + h, h)
    gx, gy = np.meshgrid(xs, ys)
    candidates = np.column_stack([gx.ravel(), gy.ravel()])

    # --- Filter: keep only points inside polygons ---
    from shapely import contains_xy, union_all
    combined = union_all(polygons)
    inside = contains_xy(combined, candidates[:, 0], candidates[:, 1])
    candidates = candidates[inside]

    if len(candidates) == 0:
        return np.empty((0, 2))

    # --- Compute local desired spacing ---
    L_local = compute_density_field(
        candidates, polygons, L_fine, L_coarse,
        boundary_band, curvature_weight
    )

    # --- Probabilistic thinning ---
    # Probability of keeping a point  ∝  (L_fine / L_local)²
    # so fine regions keep ~all candidates, coarse regions keep fewer
    accept_prob = (L_fine / L_local) ** 2
    accept_prob = np.clip(accept_prob, 0, 1)
    keep = rng.random(len(candidates)) < accept_prob
    pts = candidates[keep]

    # --- Optional jitter ---
    if jitter_factor > 0 and len(pts) > 0:
        L_kept = L_local[keep]
        dx = (rng.random(len(pts)) - 0.5) * jitter_factor * L_kept
        dy = (rng.random(len(pts)) - 0.5) * jitter_factor * L_kept
        pts = pts + np.column_stack([dx, dy])

    return pts


def generate_adaptive_nodes(path, L_fine=None, L_coarse=None, L=None,
                            boundary_band=None, curvature_weight=0.5,
                            jitter_factor=0.0, seed=42):
    """
    Full adaptive node generation from a DXF file.

    If L_fine / L_coarse are not given, they're derived from L:
        L_fine  = L * 0.4    (2.5× denser near boundaries)
        L_coarse = L * 1.2   (sparser in interior)

    Pipeline:
      1. Load DXF → segments → polygons
      2. Sample boundary nodes at fine spacing
      3. Generate offset boundary layers
      4. Generate adaptive interior grid
      5. Stack and return

    Args:
        path:             DXF file path
        L_fine:           Fine spacing (auto from L if None)
        L_coarse:         Coarse spacing (auto from L if None)
        L:                Base characteristic length (fallback)
        boundary_band:    Refinement transition width (auto if None)
        curvature_weight: Curvature influence (0..1)
        jitter_factor:    Grid jitter amount
        seed:             Random seed

    Returns:
        nodes:          (N, 2) combined node array
        interior_nodes: Adaptive interior nodes
        offset_nodes:   Offset boundary layer nodes
        boundary_nodes: Boundary curve samples
    """
    from node_manager.crude_generator import (
        load_dxf, extract_segments, segments_to_polygons,
        sample_boundaries_shapely, offset_boundary_layers,
    )

    # Default spacing from L
    if L is None:
        L = 0.4
    if L_fine is None:
        L_fine = L * 0.4
    if L_coarse is None:
        L_coarse = L * 1.2
    if boundary_band is None:
        boundary_band = L * 3.0

    msp = load_dxf(path)
    segments = extract_segments(msp)
    polygons = segments_to_polygons(segments)

    if not polygons:
        raise ValueError(f"No closed polygons found in {path}")

    # 1. Boundary nodes
    boundary_nodes = sample_boundaries_shapely(polygons, spacing=L_fine * 0.5)

    # 2. Offset layers (at fine spacing near boundary)
    offset_nodes = offset_boundary_layers(
        polygons,
        offsets=[L_fine, L_fine * 2.5, L_fine * 5.0],
        spacing=L_fine,
    )

    # 3. Adaptive interior grid
    interior_nodes = adaptive_grid_shapely(
        polygons,
        L_fine=L_fine,
        L_coarse=L_coarse,
        boundary_band=boundary_band,
        curvature_weight=curvature_weight,
        jitter_factor=jitter_factor,
        seed=seed,
    )

    # Stack
    parts = [p for p in [interior_nodes, offset_nodes, boundary_nodes]
             if len(p) > 0]
    if not parts:
        raise ValueError("No nodes generated")
    nodes = np.vstack(parts)

    # De-duplicate (keep one of each near-coincident pair)
    tree = cKDTree(nodes)
    pairs = tree.query_pairs(r=L_fine * 0.2)
    remove = set()
    for i, j in pairs:
        remove.add(max(i, j))
    if remove:
        keep = sorted(set(range(len(nodes))) - remove)
        nodes = nodes[keep]
        # Recompute partition sizes (approximate)
        n_int = len(interior_nodes)
        n_off = len(offset_nodes)
    else:
        n_int = len(interior_nodes)
        n_off = len(offset_nodes)

    # Re-split for the caller (approximate after dedup)
    interior_nodes = nodes[:n_int] if n_int <= len(nodes) else nodes
    offset_nodes_out = nodes[n_int:n_int + n_off] if n_int + n_off <= len(nodes) else np.empty((0, 2))
    boundary_nodes_out = nodes[n_int + n_off:] if n_int + n_off < len(nodes) else np.empty((0, 2))

    return nodes, interior_nodes, offset_nodes_out, boundary_nodes_out
