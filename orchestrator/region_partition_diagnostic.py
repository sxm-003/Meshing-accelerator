"""
Region partition diagnostics for pre-QAOA mesh planning.

This module visualizes:
  1. Adaptive nodes and critical-region partitioning
  2. Size-field-based outer/hole boundary bands and core region
  3. Critical patch footprints
  4. Gap-based AFM-vs-merge interface decisions using r = g / h_local

No QAOA execution or final mesh generation is performed here.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import plotly.graph_objects as go
from scipy.spatial import cKDTree
from shapely.geometry import (
    GeometryCollection,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
)
from shapely.ops import nearest_points, unary_union

from node_manager.adaptive_generator import generate_adaptive_nodes
from node_manager.crude_generator import extract_segments, load_dxf, segments_to_polygons
from node_manager.critical_region_manager import build_hybrid_region_patches
from node_manager.geometry_validator import GeometryValidator


def _to_polygons(geom) -> list[Polygon]:
    if geom is None or getattr(geom, "is_empty", True):
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return [g for g in geom.geoms if isinstance(g, Polygon) and not g.is_empty]
    if isinstance(geom, GeometryCollection):
        out = []
        for g in geom.geoms:
            if isinstance(g, Polygon) and not g.is_empty:
                out.append(g)
            elif isinstance(g, MultiPolygon):
                out.extend([p for p in g.geoms if isinstance(p, Polygon) and not p.is_empty])
        return out
    return []


def _resample_ring(coords: Iterable, spacing: float, min_samples: int = 8) -> np.ndarray:
    pts = np.asarray(coords, dtype=float)[:, :2]
    if len(pts) < 2:
        return np.empty((0, 2), dtype=float)

    diffs = np.diff(pts, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    total_len = float(seg_lens.sum())
    if total_len < 1e-12:
        return pts[:1]

    n = max(int(np.ceil(total_len / max(spacing, 1e-12))), min_samples)
    target_s = np.linspace(0.0, total_len, n, endpoint=False)
    cum_len = np.concatenate([[0.0], np.cumsum(seg_lens)])

    sampled = np.zeros((n, 2), dtype=float)
    for i, s in enumerate(target_s):
        idx = np.searchsorted(cum_len, s, side="right") - 1
        idx = int(np.clip(idx, 0, len(seg_lens) - 1))
        t = (s - cum_len[idx]) / (seg_lens[idx] + 1e-30)
        sampled[i] = pts[idx] * (1.0 - t) + pts[idx + 1] * t
    return sampled


def _extract_domain_and_boundaries(dxf_path: str):
    msp = load_dxf(dxf_path)
    if msp is None:
        raise ValueError(f"Failed to load DXF modelspace: {dxf_path}")

    segments = extract_segments(msp)
    loop_polygons = segments_to_polygons(segments)
    if not loop_polygons:
        raise ValueError(f"No closed polygon loops recovered from: {dxf_path}")

    validator = GeometryValidator(loop_polygons)
    polygons = validator.polygons
    domain = validator.domain
    return domain, polygons


def _compute_h_local(nodes: np.ndarray, k: int = 8) -> np.ndarray:
    pts = np.asarray(nodes, dtype=float)
    n = len(pts)
    if n == 0:
        return np.zeros(0, dtype=float)
    if n == 1:
        return np.ones(1, dtype=float)

    k_eff = int(np.clip(k + 1, 2, n))
    tree = cKDTree(pts)
    dists, _ = tree.query(pts, k=k_eff)
    dists = np.asarray(dists, dtype=float)
    if dists.ndim == 1:
        dists = dists[:, None]

    nn = dists[:, 1:]
    h_local = np.median(nn, axis=1)

    positive = h_local[h_local > 1e-12]
    fallback = float(np.median(positive)) if len(positive) > 0 else 1.0
    h_local = np.where(h_local > 1e-12, h_local, fallback)
    return h_local


def _extract_boundary_samples(polygons: list[Polygon], spacing: float):
    outer_pts = []
    hole_pts = []
    for poly in polygons:
        ext = _resample_ring(poly.exterior.coords, spacing=spacing, min_samples=12)
        if len(ext) > 0:
            outer_pts.append(ext)
        for hole in poly.interiors:
            hp = _resample_ring(hole.coords, spacing=spacing, min_samples=12)
            if len(hp) > 0:
                hole_pts.append(hp)

    outer = np.vstack(outer_pts) if outer_pts else np.empty((0, 2), dtype=float)
    holes = np.vstack(hole_pts) if hole_pts else np.empty((0, 2), dtype=float)
    return outer, holes


def _local_h_at_point(pt: np.ndarray, tree: cKDTree | None, h_local: np.ndarray) -> float:
    if tree is None or len(h_local) == 0:
        return 1.0
    k = min(8, len(h_local))
    _, idx = tree.query(np.asarray(pt, dtype=float), k=k)
    idx = np.asarray(idx, dtype=int).ravel()
    vals = h_local[idx]
    vals = vals[np.isfinite(vals) & (vals > 1e-12)]
    if len(vals) == 0:
        med = h_local[np.isfinite(h_local) & (h_local > 1e-12)]
        return float(np.median(med)) if len(med) > 0 else 1.0
    return float(np.median(vals))


def _build_size_based_band(
    sample_pts: np.ndarray,
    sample_h: np.ndarray,
    scale: float,
    domain,
    quantile_bins: int = 12,
):
    if len(sample_pts) == 0:
        return GeometryCollection()

    radii = np.asarray(sample_h, dtype=float) * float(scale)
    radii = np.where(np.isfinite(radii), radii, 0.0)
    radii = np.clip(radii, 1e-6, None)

    # Bin radii to keep union operations scalable on dense boundaries.
    q_count = max(1, min(int(quantile_bins), len(sample_pts)))
    q_edges = np.unique(np.quantile(radii, np.linspace(0.0, 1.0, q_count + 1)))

    parts = []
    if len(q_edges) <= 2:
        mp = MultiPoint(sample_pts)
        parts.append(mp.buffer(float(np.median(radii))))
    else:
        for lo, hi in zip(q_edges[:-1], q_edges[1:]):
            mask = (radii >= lo) & (radii <= hi)
            if np.count_nonzero(mask) == 0:
                continue
            group_pts = sample_pts[mask]
            rr = float(np.median(radii[mask]))
            parts.append(MultiPoint(group_pts).buffer(rr))

    if not parts:
        return GeometryCollection()
    return unary_union(parts).intersection(domain).buffer(0)


def _add_polygon_fill(
    fig: go.Figure,
    geom,
    name: str,
    fillcolor: str,
    linecolor: str,
    opacity: float,
):
    polys = _to_polygons(geom)
    for i, poly in enumerate(polys):
        x, y = poly.exterior.xy
        fig.add_trace(
            go.Scatter(
                x=np.asarray(x),
                y=np.asarray(y),
                mode="lines",
                line=dict(color=linecolor, width=1.2),
                fill="toself",
                fillcolor=fillcolor,
                opacity=opacity,
                name=name,
                showlegend=(i == 0),
                hoverinfo="skip",
            )
        )
        # Carve interiors for readability.
        for hole in poly.interiors:
            hx, hy = hole.xy
            fig.add_trace(
                go.Scatter(
                    x=np.asarray(hx),
                    y=np.asarray(hy),
                    mode="lines",
                    line=dict(color="rgba(255,255,255,0.6)", width=1.0),
                    fill="toself",
                    fillcolor="white",
                    opacity=1.0,
                    name=f"{name} hole",
                    showlegend=False,
                    hoverinfo="skip",
                )
            )


def _add_domain_boundaries(fig: go.Figure, polygons: list[Polygon]):
    first_outer = True
    first_hole = True
    for poly in polygons:
        x, y = poly.exterior.xy
        fig.add_trace(
            go.Scatter(
                x=np.asarray(x),
                y=np.asarray(y),
                mode="lines",
                line=dict(color="black", width=2),
                name="Domain boundary",
                showlegend=first_outer,
                hoverinfo="skip",
            )
        )
        first_outer = False

        for hole in poly.interiors:
            hx, hy = hole.xy
            fig.add_trace(
                go.Scatter(
                    x=np.asarray(hx),
                    y=np.asarray(hy),
                    mode="lines",
                    line=dict(color="black", width=1.5, dash="dot"),
                    name="Hole boundary",
                    showlegend=first_hole,
                    hoverinfo="skip",
                )
            )
            first_hole = False


def _sample_for_plot(indices: np.ndarray, max_count: int, seed: int = 42) -> np.ndarray:
    idx = np.asarray(indices, dtype=int)
    if len(idx) <= max_count:
        return idx
    rng = np.random.RandomState(seed)
    choose = rng.choice(len(idx), size=max_count, replace=False)
    return np.sort(idx[choose])


def _build_patch_footprint(patch: dict, nodes: np.ndarray, default_h: float):
    interior = np.asarray(patch.get("interior_idx", []), dtype=int)
    halo = np.asarray(patch.get("halo_idx", []), dtype=int)
    all_idx = np.concatenate([interior, halo]) if len(halo) > 0 else interior
    if len(all_idx) == 0:
        return None

    pts = np.asarray(nodes[all_idx], dtype=float)
    if len(pts) < 3:
        center = pts.mean(axis=0)
        return Point(center).buffer(max(default_h, 1e-4))

    hull_geom = MultiPoint(pts).convex_hull
    if isinstance(hull_geom, Polygon) and hull_geom.area > 1e-12:
        return hull_geom

    # Fallback ordering for degenerate hull edge-cases.
    center = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    ordered = pts[np.argsort(ang)]
    poly = Polygon(ordered).buffer(0)
    if isinstance(poly, Polygon) and not poly.is_empty and poly.area > 1e-12:
        return poly
    return Point(center).buffer(max(default_h, 1e-4))


def _classify_ratio(ratio: float, afm_ratio_ok: float, merge_ratio_cutoff: float) -> str:
    if ratio >= afm_ratio_ok:
        return "AFM"
    if ratio <= merge_ratio_cutoff:
        return "merge_or_repatch"
    return "refine_then_AFM"


def _compute_interfaces(
    patch_footprints: list[dict],
    outer_band,
    hole_band,
    nodes: np.ndarray,
    h_local: np.ndarray,
    afm_ratio_ok: float,
    merge_ratio_cutoff: float,
    neighbor_k: int = 6,
):
    tree = cKDTree(np.asarray(nodes, dtype=float)) if len(nodes) > 0 else None
    interfaces = []

    # Patch-to-patch interfaces, neighbor-pruned for scalability.
    n_patches = len(patch_footprints)
    if n_patches > 1:
        centers = np.array([pf["center"] for pf in patch_footprints], dtype=float)
        center_tree = cKDTree(centers)
        k_eff = min(max(2, int(neighbor_k) + 1), n_patches)
        candidate_pairs = set()
        for i in range(n_patches):
            _, nbr = center_tree.query(centers[i], k=k_eff)
            nbr = np.asarray(nbr, dtype=int).ravel()
            for j in nbr[1:]:
                if j <= i:
                    continue
                candidate_pairs.add((i, int(j)))

        for i, j in sorted(candidate_pairs):
            ga = patch_footprints[i]["geometry"]
            gb = patch_footprints[j]["geometry"]
            if ga.is_empty or gb.is_empty:
                continue

            # Skip very distant pairs before expensive boundary projections.
            center_dist = float(np.linalg.norm(centers[i] - centers[j]))
            ri = float(patch_footprints[i]["eq_radius"])
            rj = float(patch_footprints[j]["eq_radius"])
            h_ref = float(np.median([patch_footprints[i]["local_h"], patch_footprints[j]["local_h"]]))
            max_relevant = 3.0 * (ri + rj + afm_ratio_ok * h_ref)
            if center_dist > max_relevant:
                continue

            g = float(ga.boundary.distance(gb.boundary))
            pa, pb = nearest_points(ga.boundary, gb.boundary)
            p1 = np.array([pa.x, pa.y], dtype=float)
            p2 = np.array([pb.x, pb.y], dtype=float)
            mid = 0.5 * (p1 + p2)
            h_if = _local_h_at_point(mid, tree, h_local)
            ratio = float(g / max(h_if, 1e-12))
            decision = _classify_ratio(ratio, afm_ratio_ok, merge_ratio_cutoff)
            interfaces.append(
                {
                    "interface_id": f"pp_{i}_{j}",
                    "interface_type": "patch_to_patch",
                    "a": str(patch_footprints[i]["patch_id"]),
                    "b": str(patch_footprints[j]["patch_id"]),
                    "gap": g,
                    "h_local": h_if,
                    "ratio": ratio,
                    "decision": decision,
                    "p1": p1.tolist(),
                    "p2": p2.tolist(),
                    "mid": mid.tolist(),
                }
            )

    # Patch-to-band interfaces.
    bands = [("outer_band", outer_band), ("hole_band", hole_band)]
    for i, pf in enumerate(patch_footprints):
        gp = pf["geometry"]
        if gp.is_empty:
            continue
        for band_name, gb in bands:
            if gb is None or getattr(gb, "is_empty", True):
                continue
            g = float(gp.boundary.distance(gb.boundary))
            pa, pb = nearest_points(gp.boundary, gb.boundary)
            p1 = np.array([pa.x, pa.y], dtype=float)
            p2 = np.array([pb.x, pb.y], dtype=float)
            mid = 0.5 * (p1 + p2)
            h_if = _local_h_at_point(mid, tree, h_local)
            ratio = float(g / max(h_if, 1e-12))
            decision = _classify_ratio(ratio, afm_ratio_ok, merge_ratio_cutoff)
            interfaces.append(
                {
                    "interface_id": f"pb_{i}_{band_name}",
                    "interface_type": "patch_to_band",
                    "a": str(pf["patch_id"]),
                    "b": band_name,
                    "gap": g,
                    "h_local": h_if,
                    "ratio": ratio,
                    "decision": decision,
                    "p1": p1.tolist(),
                    "p2": p2.tolist(),
                    "mid": mid.tolist(),
                }
            )

    return interfaces


def _build_partition_figure(
    nodes: np.ndarray,
    critical_indices: np.ndarray,
    normal_indices: np.ndarray,
    polygons: list[Polygon],
    outer_band,
    hole_band,
    core_region,
    patch_footprints: list[dict],
) -> go.Figure:
    fig = go.Figure()

    _add_polygon_fill(
        fig,
        core_region,
        name="Core region",
        fillcolor="rgba(178, 223, 138, 0.45)",
        linecolor="rgba(67, 160, 71, 0.8)",
        opacity=1.0,
    )
    _add_polygon_fill(
        fig,
        outer_band,
        name="Outer band",
        fillcolor="rgba(255, 183, 77, 0.45)",
        linecolor="rgba(245, 124, 0, 0.9)",
        opacity=1.0,
    )
    _add_polygon_fill(
        fig,
        hole_band,
        name="Hole band",
        fillcolor="rgba(149, 117, 205, 0.40)",
        linecolor="rgba(94, 53, 177, 0.9)",
        opacity=1.0,
    )

    if len(normal_indices) > 0:
        n_show = _sample_for_plot(normal_indices, max_count=18000, seed=17)
        fig.add_trace(
            go.Scattergl(
                x=nodes[n_show, 0],
                y=nodes[n_show, 1],
                mode="markers",
                marker=dict(size=2, color="rgba(30,136,229,0.55)"),
                name=f"Normal nodes ({len(n_show)} shown)",
                hoverinfo="skip",
            )
        )
    if len(critical_indices) > 0:
        c_show = _sample_for_plot(critical_indices, max_count=18000, seed=29)
        fig.add_trace(
            go.Scattergl(
                x=nodes[c_show, 0],
                y=nodes[c_show, 1],
                mode="markers",
                marker=dict(size=2.2, color="rgba(216,27,96,0.85)"),
                name=f"Critical nodes ({len(c_show)} shown)",
                hoverinfo="skip",
            )
        )

    # Aggregate all patch outlines into one trace for scalability.
    px = []
    py = []
    for pf in patch_footprints:
        geom = pf["geometry"]
        if geom.is_empty:
            continue
        for poly in _to_polygons(geom):
            x, y = poly.exterior.xy
            px.extend(np.asarray(x).tolist() + [None])
            py.extend(np.asarray(y).tolist() + [None])
    if px:
        fig.add_trace(
            go.Scatter(
                x=px,
                y=py,
                mode="lines",
                line=dict(color="rgba(211,47,47,0.88)", width=1.2),
                name=f"Critical patch footprints ({len(patch_footprints)})",
                hoverinfo="skip",
            )
        )

    _add_domain_boundaries(fig, polygons)

    fig.update_layout(
        title="Region Partition Diagnostics",
        template="plotly_white",
        xaxis_title="x",
        yaxis_title="y",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.0),
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


def _build_gap_figure(polygons: list[Polygon], patch_footprints: list[dict], interfaces: list[dict]) -> go.Figure:
    fig = go.Figure()
    _add_domain_boundaries(fig, polygons)

    # Aggregate patch outlines into one trace.
    px = []
    py = []
    for pf in patch_footprints:
        for poly in _to_polygons(pf["geometry"]):
            x, y = poly.exterior.xy
            px.extend(np.asarray(x).tolist() + [None])
            py.extend(np.asarray(y).tolist() + [None])
    if px:
        fig.add_trace(
            go.Scatter(
                x=px,
                y=py,
                mode="lines",
                line=dict(color="rgba(60,60,60,0.55)", width=1.0),
                name=f"Patch footprints ({len(patch_footprints)})",
                hoverinfo="skip",
            )
        )

    decision_colors = {
        "AFM": "#2e7d32",
        "refine_then_AFM": "#f9a825",
        "merge_or_repatch": "#c62828",
    }

    for decision, color in decision_colors.items():
        group = [m for m in interfaces if m["decision"] == decision]
        if not group:
            continue

        xs = []
        ys = []
        mx = []
        my = []
        mtext = []
        for m in group:
            p1 = np.asarray(m["p1"], dtype=float)
            p2 = np.asarray(m["p2"], dtype=float)
            mid = np.asarray(m["mid"], dtype=float)
            xs.extend([p1[0], p2[0], None])
            ys.extend([p1[1], p2[1], None])
            mx.append(mid[0])
            my.append(mid[1])
            mtext.append(f"r={m['ratio']:.2f}")

        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                line=dict(color=color, width=2.3),
                name=f"{decision} ({len(group)})",
                hoverinfo="name",
            )
        )

        # Label a subset to keep plots responsive while preserving explicit r=g/h display.
        label_n = min(300, len(group))
        if label_n <= 0:
            continue
        step = max(1, len(group) // label_n)
        sel = np.arange(0, len(group), step)[:label_n]

        fig.add_trace(
            go.Scatter(
                x=np.asarray(mx)[sel],
                y=np.asarray(my)[sel],
                mode="markers+text",
                marker=dict(size=6, color=color, symbol="circle"),
                text=np.asarray(mtext)[sel],
                textposition="top center",
                textfont=dict(size=9, color=color),
                name=f"{decision} ratios (sampled)",
                showlegend=False,
                hoverinfo="skip",
            )
        )

    fig.update_layout(
        title="Gap Decision Diagnostics (r = g / h_local)",
        template="plotly_white",
        xaxis_title="x",
        yaxis_title="y",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.0),
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


def _jsonify(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, tuple):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


def _default_out_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("outputs") / "region_diag" / stamp


def run_region_partition_diagnostic(
    dxf_path: str,
    L: float = 0.1,
    Q_max: int = 14,
    overlap_factor: float = 1.0,
    curvature_weight: float = 0.5,
    critical_curvature_percentile: float = 80.0,
    critical_edge_ratio_threshold: float = 4.0,
    band_scale_outer: float = 4.0,
    band_scale_hole: float = 4.0,
    afm_ratio_ok: float = 3.5,
    merge_ratio_cutoff: float = 2.0,
    out_dir: str | None = None,
    return_figures: bool = False,
) -> dict:
    """
    Run pre-QAOA region partition diagnostics and save visual artifacts.

    Artifacts:
      - partition_map.html
      - gap_decision_map.html
      - diagnostic_summary.json

    Args:
      return_figures: If True, include Plotly Figure objects in the returned
                      dict under keys "partition_figure" and "gap_figure".
    """
    out_path = Path(out_dir) if out_dir else _default_out_dir()
    out_path.mkdir(parents=True, exist_ok=True)

    dxf_path = str(dxf_path)
    print("[diag] loading domain geometry...")
    domain, polygons = _extract_domain_and_boundaries(dxf_path)

    print("[diag] generating adaptive nodes...")
    nodes, interior_nodes, offset_nodes, boundary_nodes = generate_adaptive_nodes(
        dxf_path,
        L=L,
        curvature_weight=curvature_weight,
    )
    nodes = np.asarray(nodes, dtype=float)

    n_interior = len(interior_nodes)
    n_offset = len(offset_nodes)
    n_boundary = len(boundary_nodes)
    cad_boundary_idx = np.arange(n_interior + n_offset, n_interior + n_offset + n_boundary, dtype=int)

    print("[diag] computing local size field h_local...")
    h_local = _compute_h_local(nodes, k=8)
    node_tree = cKDTree(nodes) if len(nodes) > 0 else None

    # Boundary samples for size-field banding.
    print("[diag] building size-based outer/hole bands...")
    L_fine = max(L * 0.4, 1e-6)
    sample_spacing = max(L_fine * 0.5, 1e-6)
    outer_samples, hole_samples = _extract_boundary_samples(polygons, spacing=sample_spacing)

    if len(outer_samples) > 0 and node_tree is not None:
        _, oi = node_tree.query(outer_samples, k=1)
        outer_h = h_local[np.asarray(oi, dtype=int)]
    else:
        outer_h = np.full(len(outer_samples), float(np.median(h_local) if len(h_local) else 1.0))

    if len(hole_samples) > 0 and node_tree is not None:
        _, hi = node_tree.query(hole_samples, k=1)
        hole_h = h_local[np.asarray(hi, dtype=int)]
    else:
        hole_h = np.full(len(hole_samples), float(np.median(h_local) if len(h_local) else 1.0))

    outer_band = _build_size_based_band(
        outer_samples,
        outer_h,
        scale=band_scale_outer,
        domain=domain,
    )
    hole_band = _build_size_based_band(
        hole_samples,
        hole_h,
        scale=band_scale_hole,
        domain=domain,
    )

    bands_union = unary_union([outer_band, hole_band]).buffer(0)
    core_region = domain.difference(bands_union).buffer(0)

    print("[diag] detecting critical regions and generating patches...")
    region_data = build_hybrid_region_patches(
        nodes=nodes,
        L=L,
        Q_max=Q_max,
        overlap_factor=overlap_factor,
        cad_boundary_idx=cad_boundary_idx,
        use_critical_regions=True,
        curvature_threshold_percentile=critical_curvature_percentile,
        min_angle_threshold=15.0,
        edge_ratio_threshold=critical_edge_ratio_threshold,
    )

    critical_indices = np.asarray(region_data["critical_indices"], dtype=int)
    normal_indices = np.asarray(region_data["normal_indices"], dtype=int)
    critical_patches = region_data["critical_patches"]

    default_h = float(np.median(h_local[h_local > 0])) if len(h_local) > 0 else max(L, 1e-3)
    patch_footprints = []
    for p in critical_patches:
        geom = _build_patch_footprint(p, nodes=nodes, default_h=default_h)
        if geom is None or geom.is_empty:
            continue
        center = np.array(geom.representative_point().coords[0], dtype=float)
        area = float(max(geom.area, 1e-16))
        eq_radius = float(np.sqrt(area / np.pi))
        if node_tree is not None and len(h_local) > 0:
            _, ci = node_tree.query(center, k=1)
            local_h = float(h_local[int(ci)])
        else:
            local_h = default_h
        patch_footprints.append(
            {
                "patch_id": p.get("patch_id", "unknown"),
                "geometry": geom,
                "n_interior": int(len(p.get("interior_idx", []))),
                "n_halo": int(len(p.get("halo_idx", []))),
                "center": center,
                "eq_radius": eq_radius,
                "local_h": local_h,
            }
        )

    print("[diag] computing interface gap decisions...")
    interfaces = _compute_interfaces(
        patch_footprints=patch_footprints,
        outer_band=outer_band,
        hole_band=hole_band,
        nodes=nodes,
        h_local=h_local,
        afm_ratio_ok=afm_ratio_ok,
        merge_ratio_cutoff=merge_ratio_cutoff,
        neighbor_k=6,
    )

    print("[diag] rendering diagnostic figures...")
    partition_fig = _build_partition_figure(
        nodes=nodes,
        critical_indices=critical_indices,
        normal_indices=normal_indices,
        polygons=polygons,
        outer_band=outer_band,
        hole_band=hole_band,
        core_region=core_region,
        patch_footprints=patch_footprints,
    )
    gap_fig = _build_gap_figure(polygons=polygons, patch_footprints=patch_footprints, interfaces=interfaces)

    partition_path = out_path / "partition_map.html"
    gap_path = out_path / "gap_decision_map.html"
    summary_path = out_path / "diagnostic_summary.json"
    partition_fig.write_html(str(partition_path), include_plotlyjs="cdn")
    gap_fig.write_html(str(gap_path), include_plotlyjs="cdn")

    domain_area = float(domain.area)
    outer_area = float(outer_band.area)
    hole_area = float(hole_band.area)
    core_area = float(core_region.area)
    union_area = float(unary_union([core_region, bands_union]).intersection(domain).area)
    hole_union = unary_union([Polygon(h.coords) for p in polygons for h in p.interiors]) if any(
        len(p.interiors) > 0 for p in polygons
    ) else GeometryCollection()

    patch_sizes = [
        int(np.asarray(p.get("interior_idx", []), dtype=int).size + np.asarray(p.get("halo_idx", []), dtype=int).size)
        for p in critical_patches
    ]
    decision_counts = {"AFM": 0, "refine_then_AFM": 0, "merge_or_repatch": 0}
    type_counts = {"patch_to_patch": 0, "patch_to_band": 0}
    ratios = []
    finite_gap_metrics = True
    for m in interfaces:
        decision_counts[m["decision"]] = decision_counts.get(m["decision"], 0) + 1
        type_counts[m["interface_type"]] = type_counts.get(m["interface_type"], 0) + 1
        ratios.append(m["ratio"])
        if not (np.isfinite(m["gap"]) and np.isfinite(m["h_local"]) and np.isfinite(m["ratio"])):
            finite_gap_metrics = False

    invariants = {
        "core_band_intersection_area": float(core_region.intersection(bands_union).area),
        "domain_coverage_residual_area": float(abs(domain_area - union_area)),
        "band_outside_domain_area": float(bands_union.difference(domain).area),
        "band_overlap_hole_area": float(bands_union.intersection(hole_union).area) if not hole_union.is_empty else 0.0,
        "all_patch_footprints_nonempty": bool(all((not pf["geometry"].is_empty) for pf in patch_footprints)),
        "all_gap_metrics_finite": bool(finite_gap_metrics),
    }

    summary = {
        "inputs": {
            "dxf_path": dxf_path,
            "L": float(L),
            "Q_max": int(Q_max),
            "overlap_factor": float(overlap_factor),
            "curvature_weight": float(curvature_weight),
            "critical_curvature_percentile": float(critical_curvature_percentile),
            "critical_edge_ratio_threshold": float(critical_edge_ratio_threshold),
            "band_scale_outer": float(band_scale_outer),
            "band_scale_hole": float(band_scale_hole),
            "afm_ratio_ok": float(afm_ratio_ok),
            "merge_ratio_cutoff": float(merge_ratio_cutoff),
        },
        "node_counts": {
            "total_nodes": int(len(nodes)),
            "interior_nodes": int(len(interior_nodes)),
            "offset_nodes": int(len(offset_nodes)),
            "boundary_nodes": int(len(boundary_nodes)),
            "critical_nodes": int(len(critical_indices)),
            "normal_nodes": int(len(normal_indices)),
        },
        "areas": {
            "domain_area": domain_area,
            "outer_band_area": outer_area,
            "hole_band_area": hole_area,
            "core_area": core_area,
            "outer_band_fraction": float(outer_area / max(domain_area, 1e-12)),
            "hole_band_fraction": float(hole_area / max(domain_area, 1e-12)),
            "core_fraction": float(core_area / max(domain_area, 1e-12)),
        },
        "patch_stats": {
            "critical_patch_count": int(len(critical_patches)),
            "normal_patch_count": int(len(region_data["normal_patches"])),
            "critical_patch_size_min": int(min(patch_sizes)) if patch_sizes else 0,
            "critical_patch_size_max": int(max(patch_sizes)) if patch_sizes else 0,
            "critical_patch_size_mean": float(np.mean(patch_sizes)) if patch_sizes else 0.0,
            "critical_patch_footprints": int(len(patch_footprints)),
        },
        "interface_stats": {
            "interface_total": int(len(interfaces)),
            "interface_type_counts": type_counts,
            "decision_counts": decision_counts,
            "ratio_min": float(np.min(ratios)) if ratios else 0.0,
            "ratio_mean": float(np.mean(ratios)) if ratios else 0.0,
            "ratio_max": float(np.max(ratios)) if ratios else 0.0,
        },
        "invariants": invariants,
        "outputs": {
            "partition_map_html": str(partition_path),
            "gap_decision_map_html": str(gap_path),
            "diagnostic_summary_json": str(summary_path),
        },
        "interfaces_sample": _jsonify(
            sorted(interfaces, key=lambda m: float(m["ratio"]), reverse=True)[:500]
        ),
        "interfaces_sample_size": min(500, len(interfaces)),
        "diagnostics": _jsonify(region_data["diagnostics"]),
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(_jsonify(summary), f, indent=2)

    print("\n=== Region Partition Diagnostic ===")
    print(f"DXF: {dxf_path}")
    print(f"Nodes: total={len(nodes)}, critical={len(critical_indices)}, normal={len(normal_indices)}")
    print(
        "Areas: "
        f"outer={outer_area:.4f} ({100*outer_area/max(domain_area,1e-12):.1f}%), "
        f"hole={hole_area:.4f} ({100*hole_area/max(domain_area,1e-12):.1f}%), "
        f"core={core_area:.4f} ({100*core_area/max(domain_area,1e-12):.1f}%)"
    )
    print(
        "Interfaces: "
        f"total={len(interfaces)}, AFM={decision_counts.get('AFM', 0)}, "
        f"refine_then_AFM={decision_counts.get('refine_then_AFM', 0)}, "
        f"merge_or_repatch={decision_counts.get('merge_or_repatch', 0)}"
    )
    print(f"Saved: {partition_path}")
    print(f"Saved: {gap_path}")
    print(f"Saved: {summary_path}")
    if return_figures:
        summary["partition_figure"] = partition_fig
        summary["gap_figure"] = gap_fig
    return summary


if __name__ == "__main__":
    default_dxf = str((Path(__file__).resolve().parents[1] / "data" / "low_angle.dxf").resolve())
    run_region_partition_diagnostic(dxf_path=default_dxf)
