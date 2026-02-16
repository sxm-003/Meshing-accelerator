"""
Critical-region detection and hybrid patch generation utilities.

  1. Detect critical nodes using curvature + local edge-ratio indicators.
  2. Build separate patch sets for:
     - critical regions (QAOA)
     - normal regions (classical Delaunay)
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from node_manager.patch_generator import generate_patch


def compute_local_curvature(nodes: np.ndarray, k: int = 5) -> np.ndarray:
    """
    Compute local curvature via a Menger-curvature style estimate on nearest neighbors.
    """
    pts = np.asarray(nodes, dtype=float)
    n = len(pts)
    curvatures = np.zeros(n, dtype=float)

    if n < 3:
        return curvatures

    tree = cKDTree(pts)
    for i in range(n):
        _, indices = tree.query(pts[i], k=min(k + 1, n))
        neighbors = pts[np.atleast_1d(indices)[1:]]  # Exclude self

        if len(neighbors) < 2:
            continue

        p0 = pts[i]
        p1 = neighbors[0]
        p2 = neighbors[1]

        v1 = p1 - p0
        v2 = p2 - p0
        area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])

        a = np.linalg.norm(p1 - p0)
        b = np.linalg.norm(p2 - p1)
        c = np.linalg.norm(p0 - p2)
        denom = a * b * c
        if denom > 1e-10:
            curvatures[i] = 4.0 * area / denom

    return curvatures


def detect_critical_regions(
    nodes: np.ndarray,
    curvature_threshold_percentile: float = 90.0,
    min_angle_threshold: float = 15.0,  # Kept for notebook parity (unused).
    edge_ratio_threshold: float = 4.0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Detect critical regions where normal meshing is likely to degrade.

    Indicators:
      - high local curvature
      - high local edge-length ratio
      - poor_local_mesh is intentionally disabled (all False)
    """
    del min_angle_threshold

    pts = np.asarray(nodes, dtype=float)
    n = len(pts)
    if n == 0:
        empty = np.zeros(0, dtype=bool)
        diagnostics = {
            "curvatures": np.zeros(0, dtype=float),
            "curvature_threshold": 0.0,
            "edge_ratios": np.zeros(0, dtype=float),
            "n_high_curvature": 0,
            "n_high_edge_ratio": 0,
            "n_poor_local_mesh": 0,
            "n_critical": 0,
            "n_normal": 0,
            "n_uniform": 0,
        }
        return empty, ~empty, diagnostics

    curvatures = compute_local_curvature(pts, k=5)
    curvature_threshold = float(np.percentile(curvatures, curvature_threshold_percentile))
    high_curvature_mask = curvatures > curvature_threshold

    edge_lengths = np.linalg.norm(pts - np.roll(pts, 1, axis=0), axis=1)
    local_edge_ratio = np.zeros(n, dtype=float)
    for i in range(n):
        prev_edge = edge_lengths[i]
        next_edge = edge_lengths[(i + 1) % n]
        local_edge_ratio[i] = max(prev_edge, next_edge) / (min(prev_edge, next_edge) + 1e-10)
    high_edge_ratio_mask = local_edge_ratio > edge_ratio_threshold

    poor_local_mesh = np.zeros(n, dtype=bool)

    critical_mask = high_curvature_mask | high_edge_ratio_mask | poor_local_mesh

    # Minimal one-step expansion (same notebook behavior).
    if np.any(critical_mask):
        expanded = critical_mask.copy()
        for i in range(n):
            if critical_mask[i]:
                expanded[(i - 1) % n] = True
                expanded[(i + 1) % n] = True
        critical_mask = expanded

    normal_mask = ~critical_mask
    diagnostics = {
        "curvatures": curvatures,
        "curvature_threshold": curvature_threshold,
        "edge_ratios": local_edge_ratio,
        "n_high_curvature": int(np.sum(high_curvature_mask)),
        "n_high_edge_ratio": int(np.sum(high_edge_ratio_mask)),
        "n_poor_local_mesh": int(np.sum(poor_local_mesh)),
        "n_critical": int(np.sum(critical_mask)),
        "n_normal": int(np.sum(normal_mask)),
        "n_uniform": int(np.sum(normal_mask)),
    }
    return critical_mask, normal_mask, diagnostics


def _create_single_patch(
    center: np.ndarray,
    indices: np.ndarray,
    cad_boundary_idx: np.ndarray | None,
    region_type: str,
    patch_id: str,
) -> list[dict]:
    """Fallback patch when a region is too small for grid-based patching."""
    if len(indices) == 0:
        return []

    cad_local = None
    if cad_boundary_idx is not None and len(cad_boundary_idx) > 0:
        cad_local = np.where(np.isin(indices, cad_boundary_idx))[0]
        if len(cad_local) == 0:
            cad_local = None

    return [{
        "center": np.asarray(center, dtype=float),
        "interior_idx": np.asarray(indices, dtype=int),
        "halo_idx": np.empty(0, dtype=int),
        "patch_id": patch_id,
        "cad_boundary_idx_local": cad_local,
        "region_type": region_type,
    }]


def _patchify_region(
    nodes: np.ndarray,
    global_region_indices: np.ndarray,
    L: float,
    Q_max: int,
    overlap_factor: float,
    cad_boundary_idx: np.ndarray | None,
    region_type: str,
) -> list[dict]:
    """Generate patches on a region subset, remapping local indices back to global."""
    region_idx = np.asarray(global_region_indices, dtype=int)
    if len(region_idx) == 0:
        return []

    region_nodes = nodes[region_idx]

    # For tiny regions, use a single patch to avoid radius/density issues.
    # Respect qubit cap by only taking this path when region size <= Q_max.
    if len(region_nodes) <= max(1, Q_max):
        return _create_single_patch(
            center=region_nodes.mean(axis=0),
            indices=region_idx,
            cad_boundary_idx=cad_boundary_idx,
            region_type=region_type,
            patch_id=f"{region_type}_0",
        )

    region_cad_local = None
    if cad_boundary_idx is not None and len(cad_boundary_idx) > 0:
        region_cad_local = np.where(np.isin(region_idx, cad_boundary_idx))[0]
        if len(region_cad_local) == 0:
            region_cad_local = None

    local_patches = generate_patch(
        L=L,
        nodes=region_nodes,
        Q_max=Q_max,
        overlap_factor=overlap_factor,
        cad_boundary_idx=region_cad_local,
    )

    out = []
    for i, patch in enumerate(local_patches):
        local_interior = np.asarray(patch.get("interior_idx", []), dtype=int)
        local_halo = np.asarray(patch.get("halo_idx", []), dtype=int)

        global_interior = region_idx[local_interior] if len(local_interior) > 0 else np.empty(0, dtype=int)
        global_halo = region_idx[local_halo] if len(local_halo) > 0 else np.empty(0, dtype=int)

        # Skip empty patches (can happen on sparse masks).
        if len(global_interior) == 0 and len(global_halo) == 0:
            continue

        remapped = {
            "center": np.asarray(patch["center"], dtype=float),
            "interior_idx": global_interior,
            "halo_idx": global_halo,
            "patch_id": f"{region_type}_{i}",
            "cad_boundary_idx_local": patch.get("cad_boundary_idx_local", None),
            "region_type": region_type,
        }
        out.append(remapped)

    return out


def build_hybrid_region_patches(
    nodes: np.ndarray,
    L: float,
    Q_max: int,
    overlap_factor: float = 1.0,
    cad_boundary_idx: np.ndarray | None = None,
    use_critical_regions: bool = True,
    curvature_threshold_percentile: float = 90.0,
    min_angle_threshold: float = 15.0,
    edge_ratio_threshold: float = 4.0,
    normal_region_qmax: int | None = None,
) -> dict:
    """
    Split nodes into critical/normal regions and create patch sets for each.
    """
    pts = np.asarray(nodes, dtype=float)
    n = len(pts)
    if n == 0:
        return {
            "critical_mask": np.zeros(0, dtype=bool),
            "normal_mask": np.zeros(0, dtype=bool),
            "critical_indices": np.zeros(0, dtype=int),
            "normal_indices": np.zeros(0, dtype=int),
            "critical_patches": [],
            "normal_patches": [],
            "all_patches": [],
            "diagnostics": {
                "curvatures": np.zeros(0, dtype=float),
                "curvature_threshold": 0.0,
                "edge_ratios": np.zeros(0, dtype=float),
                "n_high_curvature": 0,
                "n_high_edge_ratio": 0,
                "n_poor_local_mesh": 0,
                "n_critical": 0,
                "n_normal": 0,
                "n_uniform": 0,
            },
        }

    if use_critical_regions:
        critical_mask, normal_mask, diagnostics = detect_critical_regions(
            pts,
            curvature_threshold_percentile=curvature_threshold_percentile,
            min_angle_threshold=min_angle_threshold,
            edge_ratio_threshold=edge_ratio_threshold,
        )
    else:
        critical_mask = np.ones(n, dtype=bool)
        normal_mask = np.zeros(n, dtype=bool)
        diagnostics = {
            "curvatures": np.zeros(n, dtype=float),
            "curvature_threshold": 0.0,
            "edge_ratios": np.zeros(n, dtype=float),
            "n_high_curvature": 0,
            "n_high_edge_ratio": 0,
            "n_poor_local_mesh": 0,
            "n_critical": int(n),
            "n_normal": 0,
            "n_uniform": 0,
        }

    critical_indices = np.where(critical_mask)[0]
    normal_indices = np.where(normal_mask)[0]

    # Normal-region patches are classical (not qubit-limited), so use a larger
    # cap by default to reduce patch count and generation time on large DXFs.
    if normal_region_qmax is None:
        normal_region_qmax = max(100, int(Q_max) * 10)

    critical_patches = _patchify_region(
        nodes=pts,
        global_region_indices=critical_indices,
        L=L,
        Q_max=Q_max,
        overlap_factor=overlap_factor,
        cad_boundary_idx=cad_boundary_idx,
        region_type="critical",
    )
    normal_patches = _patchify_region(
        nodes=pts,
        global_region_indices=normal_indices,
        L=L,
        Q_max=normal_region_qmax,
        overlap_factor=overlap_factor,
        cad_boundary_idx=cad_boundary_idx,
        region_type="normal",
    )

    return {
        "critical_mask": critical_mask,
        "normal_mask": normal_mask,
        "critical_indices": critical_indices,
        "normal_indices": normal_indices,
        "critical_patches": critical_patches,
        "normal_patches": normal_patches,
        "all_patches": [*critical_patches, *normal_patches],
        "diagnostics": diagnostics,
    }
