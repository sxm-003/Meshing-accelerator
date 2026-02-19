import math
import numpy as np
import ezdxf
from ezdxf import recover
from ezdxf.math import BSpline
from shapely.geometry import LineString, MultiLineString, Point
from shapely.ops import polygonize, snap, unary_union
import plotly.io as pio
import plotly.graph_objects as go

from node_manager.geometry_validator import GeometryValidator

def load_dxf(path):
    try:
        doc = ezdxf.readfile(path)
    except (ezdxf.DXFStructureError, IOError) as e:
        print(f"Warning: Standard read failed ({e}). Attempting recovery for: {path}")
        
        try:
            doc, auditor = recover.readfile(path)
            
            if auditor.has_errors:
                print(f"Errors found in {path}:")
                auditor.print_error_report()
            
            if doc is None:
                raise ValueError(f"Recovery failed: Could not create document for {path}")
                
        except Exception as recovery_error:
            print(f"Error: Critical failure loading or recovering DXF: {recovery_error}")
            return None  
    return doc.modelspace()

def extract_segments(msp, curve_samples=64):
    segments = []

    for e in msp:
        t = e.dxftype()

        if t == "LINE":
            p0 = np.array([e.dxf.start.x, e.dxf.start.y])
            p1 = np.array([e.dxf.end.x, e.dxf.end.y])
            segments.append((p0, p1))

        elif t == "LWPOLYLINE":
            pts = [np.array([p[0], p[1]]) for p in e]
            for i in range(len(pts) - 1):
                segments.append((pts[i], pts[i + 1]))
            if e.closed:
                segments.append((pts[-1], pts[0]))

        elif t == "POLYLINE":
            pts = [np.array([v.dxf.location.x, v.dxf.location.y])
                   for v in e.vertices]
            for i in range(len(pts) - 1):
                segments.append((pts[i], pts[i + 1]))
            if e.is_closed:
                segments.append((pts[-1], pts[0]))

        elif t == "CIRCLE":
            cx, cy = e.dxf.center.x, e.dxf.center.y
            r = e.dxf.radius
            pts = [
                np.array([
                    cx + r * math.cos(2 * math.pi * i / curve_samples),
                    cy + r * math.sin(2 * math.pi * i / curve_samples)
                ])
                for i in range(curve_samples)
            ]
            for i in range(len(pts)):
                segments.append((pts[i], pts[(i + 1) % len(pts)]))

        elif t == "ARC":
            cx, cy = e.dxf.center.x, e.dxf.center.y
            r = e.dxf.radius
            a0 = math.radians(e.dxf.start_angle)
            a1 = math.radians(e.dxf.end_angle)
            if a1 < a0:
                a1 += 2 * math.pi

            pts = [
                np.array([
                    cx + r * math.cos(a0 + (a1 - a0) * i / (curve_samples - 1)),
                    cy + r * math.sin(a0 + (a1 - a0) * i / (curve_samples - 1))
                ])
                for i in range(curve_samples)
            ]
            for i in range(len(pts) - 1):
                segments.append((pts[i], pts[i + 1]))

        elif t == "SPLINE":
            cps = []
            for p in e.control_points:
                if hasattr(p, "x"):
                    cps.append((p.x, p.y))
                else:
                    cps.append((p[0], p[1]))

            spline = BSpline(
                control_points=cps,
                order=e.dxf.degree + 1,
                knots=e.knots
            )

            pts = [
                np.array(spline.point(t))[:2]
                for t in np.linspace(0, 1, curve_samples)
            ]

            for i in range(len(pts) - 1):
                segments.append((pts[i], pts[i + 1]))

    return segments


def segments_to_polygons(segments):
    if not segments:
        return []

    lines = [LineString([p0, p1]) for p0, p1 in segments]
    multiline = MultiLineString(lines)

    candidates = []

    # Candidate 1: raw polygonization.
    raw_loops = list(polygonize(multiline))
    if raw_loops:
        try:
            raw_polys = GeometryValidator(raw_loops).polygons
            candidates.append(("raw", raw_polys))
        except ValueError:
            pass

    # Candidate 2: noded+simplified linework polygonization.
    # This recovers faces formed by mixed/open chains that only close after noding.
    try:
        noded = unary_union(snap(multiline, multiline, 1e-6))
        noded_loops = list(polygonize(noded))
        if noded_loops:
            noded_polys = GeometryValidator(noded_loops).polygons
            candidates.append(("noded", noded_polys))
    except Exception:
        pass

    if not candidates:
        return []

    def candidate_score(polys):
        holes = sum(len(p.interiors) for p in polys)
        largest_area = max((p.area for p in polys), default=0.0)
        total_area = sum(p.area for p in polys)
        # Prefer richer topology (holes), then dominant face recovery, then coverage.
        return holes, largest_area, total_area

    best_name, best_polys = max(candidates, key=lambda item: candidate_score(item[1]))
    return best_polys

def _resample_ring(ring_coords, spacing, min_samples_per_ring):
    coords = np.array(ring_coords, dtype=float)[:, :2]
    if len(coords) < 2:
        return np.empty((0, 2), dtype=float)

    diffs = np.diff(coords, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    total_len = float(seg_lens.sum())
    if total_len < 1e-12:
        return coords[:1]

    n = max(min_samples_per_ring, int(np.ceil(total_len / max(spacing, 1e-12))))
    target_s = np.linspace(0.0, total_len, n, endpoint=False)
    cum_len = np.concatenate([[0.0], np.cumsum(seg_lens)])

    sampled = np.zeros((n, 2), dtype=float)
    for i, s in enumerate(target_s):
        idx = np.searchsorted(cum_len, s, side="right") - 1
        idx = int(np.clip(idx, 0, len(seg_lens) - 1))
        t = (s - cum_len[idx]) / (seg_lens[idx] + 1e-30)
        sampled[i] = coords[idx] * (1.0 - t) + coords[idx + 1] * t
    return sampled


def jittered_grid_shapely(polygons, L, jitter_frac=0.3, seed=0, validator=None):
    return uniform_grid_shapely(
        polygons=polygons,
        L=L,
        jitter_factor=jitter_frac,
        seed=seed,
        validator=validator,
    )


def uniform_grid_shapely(polygons, L, jitter_factor=0.0, seed=0, validator=None):
    """
    Generate uniform grid nodes within polygons with optional jitter.
    
    Args:
        polygons: List of Shapely polygons
        L: Characteristic length scale
        jitter_factor: Amount of random jitter (0.0=no jitter, 1.0=full jitter)
                      Jitter is applied as ±jitter_factor * h/2
        seed: Random seed for reproducibility
    
    Returns:
        pts: Array of node coordinates
    """
    rng = np.random.RandomState(seed)
    if validator is None:
        validator = GeometryValidator(polygons)

    # Bounding box
    minx, miny, maxx, maxy = validator.bounds

    h = L / 2.0
    xs = np.arange(minx, maxx + h, h)
    ys = np.arange(miny, maxy + h, h)

    gx, gy = np.meshgrid(xs, ys)
    pts = np.column_stack([gx.ravel(), gy.ravel()])
    if jitter_factor > 0 and len(pts) > 0:
        pts[:, 0] += (rng.rand(len(pts)) - 0.5) * jitter_factor * h
        pts[:, 1] += (rng.rand(len(pts)) - 0.5) * jitter_factor * h

    inside = validator.mask_inside(pts, strict=False)
    pts = pts[inside]

    if len(pts) == 0:
        return np.empty((0, 2))
    return pts


def sample_boundaries_shapely(polygons, spacing, min_samples_per_ring=12):
    boundary_chunks = []

    for poly in polygons:
        ext = _resample_ring(
            poly.exterior.coords,
            spacing=spacing,
            min_samples_per_ring=min_samples_per_ring,
        )
        if len(ext) > 0:
            boundary_chunks.append(ext)

        for hole in poly.interiors:
            hpts = _resample_ring(
                hole.coords,
                spacing=spacing,
                min_samples_per_ring=min_samples_per_ring,
            )
            if len(hpts) > 0:
                boundary_chunks.append(hpts)

    if not boundary_chunks:
        return np.empty((0, 2))
    return np.vstack(boundary_chunks)

def offset_boundary_layers(polygons, offsets, spacing, min_samples_per_ring=8):
    chunks = []

    for poly in polygons:
        for d in offsets:
            inner = poly.buffer(-d)
            if inner.is_empty:
                continue

            if inner.geom_type == "Polygon":
                parts = [inner]
            else:
                parts = [g for g in inner.geoms if g.geom_type == "Polygon"]

            for part in parts:
                rings = [part.exterior, *part.interiors]
                for ring in rings:
                    sampled = _resample_ring(
                        ring.coords,
                        spacing=spacing,
                        min_samples_per_ring=min_samples_per_ring,
                    )
                    if len(sampled) > 0:
                        chunks.append(sampled)

    if not chunks:
        return np.empty((0, 2), dtype=float)
    return np.vstack(chunks)

def adaptive_jittered_grid_shapely(
    polygons,
    L_bulk,
    L_boundary,
    boundary_band,
    jitter_frac=0.3,
    seed=0,
    validator=None,
):
    np.random.seed(seed)
    if validator is None:
        validator = GeometryValidator(polygons)

    minx, miny, maxx, maxy = validator.bounds

    pts = []
    h = L_boundary / 2

    xs = np.arange(minx, maxx + h, h)
    ys = np.arange(miny, maxy + h, h)

    for x in xs:
        for y in ys:
            p = Point(x, y)

            for poly in polygons:
                if not poly.covers(p):
                    continue

                d = poly.exterior.distance(p)
                L = L_boundary if d < boundary_band else L_bulk

                if np.random.rand() > (L_boundary / L):
                    continue

                dx = (np.random.rand() - 0.5) * jitter_frac * L
                dy = (np.random.rand() - 0.5) * jitter_frac * L

                pts.append([x + dx, y + dy])
                break

    pts = np.array(pts, dtype=float)
    return validator.filter_points(pts, strict=False)

def generate_crude_nodes(path, jitter_factor=0.0):
    """
    Generate nodes for mesh from DXF file using uniform grid with optional jitter.
    
    Args:
        path: Path to DXF file
        jitter_factor: Random jitter amount (0.0=uniform grid, 1.0=full jitter)
                      Default 0.0 for consistent, reproducible meshes
    
    Returns:
        nodes: Combined node array
        interior_nodes: Interior grid nodes
        offset_nodes: Offset boundary layer nodes
        boundary_nodes: Boundary nodes
    """
    msp = load_dxf(path)
    segments = extract_segments(msp)
    polygons = segments_to_polygons(segments)
    if not polygons:
        raise ValueError(f"No closed polygons found in {path}")

    validator = GeometryValidator(polygons)
    polygons = validator.polygons

    boundary_nodes = sample_boundaries_shapely(
        polygons,
        spacing=0.05
    )

    offset_nodes = offset_boundary_layers(
        polygons,
        offsets=[0.05, 0.12, 0.25],
        spacing=0.1
    )

    # Use uniform grid with optional jitter
    interior_nodes = uniform_grid_shapely(
        polygons,
        L=0.4,  # Characteristic length scale
        jitter_factor=jitter_factor,  # 0.0 = uniform, >0 = jittered
        validator=validator,
    )

    interior_nodes = validator.filter_points(interior_nodes, strict=False)
    offset_nodes = validator.filter_points(offset_nodes, strict=False)
    boundary_nodes = validator.filter_points(boundary_nodes, strict=False)

    parts = [arr for arr in [interior_nodes, offset_nodes, boundary_nodes] if len(arr) > 0]
    if not parts:
        raise ValueError("No nodes generated")
    nodes = np.vstack(parts)

    return nodes, interior_nodes, offset_nodes, boundary_nodes

def interactive_view(polygons, interior, offset, boundary):
    fig = go.Figure()

    for poly in polygons:
        ext = np.array(poly.exterior.coords)[:, :2]
        fig.add_trace(go.Scattergl(
            x=ext[:, 0], y=ext[:, 1],
            mode="lines",
            line=dict(color="black", width=2),
            showlegend=False
        ))

        for hole in poly.interiors:
            h = np.array(hole.coords)[:, :2]
            fig.add_trace(go.Scattergl(
                x=h[:, 0], y=h[:, 1],
                mode="lines",
                line=dict(color="red", dash="dash"),
                showlegend=False
            ))

    fig.add_trace(go.Scattergl(
        x=interior[:, 0], y=interior[:, 1],
        mode="markers",
        marker=dict(size=3, color="blue"),
        name="Interior"
    ))

    fig.add_trace(go.Scattergl(
        x=offset[:, 0], y=offset[:, 1],
        mode="markers",
        marker=dict(size=4, color="green"),
        name="Offset layers"
    ))

    fig.add_trace(go.Scattergl(
        x=boundary[:, 0], y=boundary[:, 1],
        mode="markers",
        marker=dict(size=6, color="orange"),
        name="Boundary"
    ))

    fig.update_layout(
        width=900,
        height=900,
        xaxis=dict(scaleanchor="y"),
        title="Crude Node Generation (Geometry-Aware)"
    )

    fig.show()
