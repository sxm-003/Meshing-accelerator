"""
Mesh creation and export for MeshOptimiser.

After the QAOA pipeline selects optimal nodes and the Gaussian merger
produces the final set of global node indices, this module:
  1. Triangulates the selected nodes via constrained Delaunay
  2. Filters out-of-domain triangles
  3. Applies Laplacian smoothing (optional)
  4. Exports to common mesh formats (.msh, .vtk, .obj, .mesh)
"""

import numpy as np
from scipy.spatial import Delaunay, cKDTree
from pathlib import Path


# ─────────────────────────────────────────────────────────────
#  Triangulation + filtering
# ─────────────────────────────────────────────────────────────

def triangulate_selected_nodes(nodes, selected_indices, polygons=None):
    """
    Delaunay-triangulate the selected nodes and optionally clip to geometry.

    Args:
        nodes:           (N_total, 2) full node set
        selected_indices: 1-D array of global indices into `nodes`
        polygons:        list of Shapely polygons for filtering (optional)

    Returns:
        mesh_nodes:    (M, 2)  coordinates of the final mesh nodes
        triangles:     (T, 3)  triangle connectivity (0-indexed into mesh_nodes)
        global_idx:    (M,)    mapping from mesh_nodes back to original `nodes`
    """
    sel = np.asarray(selected_indices)
    mesh_nodes = nodes[sel]

    if len(mesh_nodes) < 3:
        raise ValueError(f"Need ≥3 nodes for triangulation, got {len(mesh_nodes)}")

    tri = Delaunay(mesh_nodes)
    triangles = tri.simplices

    # Filter triangles outside the geometry
    if polygons is not None:
        from shapely.geometry import Point
        centroids = mesh_nodes[triangles].mean(axis=1)
        inside = np.array([
            any(poly.contains(Point(c[0], c[1])) for poly in polygons)
            for c in centroids
        ])
        triangles = triangles[inside]

    return mesh_nodes, triangles, sel


# ─────────────────────────────────────────────────────────────
#  Laplacian smoothing
# ─────────────────────────────────────────────────────────────

def laplacian_smooth(nodes, triangles, iterations=5, weight=0.3,
                     fixed_mask=None):
    """
    Laplacian smoothing — moves each interior node toward the centroid
    of its neighbours.

    Args:
        nodes:      (N, 2) mutable copy of node coordinates
        triangles:  (T, 3) triangle connectivity
        iterations: number of smoothing passes
        weight:     relaxation factor (0..1), higher = more smoothing
        fixed_mask: (N,) bool — True for nodes that must not move
                    (e.g. boundary nodes)

    Returns:
        smoothed: (N, 2) smoothed coordinates
    """
    pts = nodes.copy()
    n = len(pts)

    # Build adjacency (set of neighbour indices per node)
    adj = [set() for _ in range(n)]
    for t in triangles:
        for i in range(3):
            adj[t[i]].add(t[(i + 1) % 3])
            adj[t[i]].add(t[(i + 2) % 3])

    if fixed_mask is None:
        fixed_mask = np.zeros(n, dtype=bool)

    for _ in range(iterations):
        new_pts = pts.copy()
        for i in range(n):
            if fixed_mask[i] or len(adj[i]) == 0:
                continue
            neighbours = np.array(list(adj[i]))
            centroid = pts[neighbours].mean(axis=0)
            new_pts[i] = pts[i] + weight * (centroid - pts[i])
        pts = new_pts

    return pts


# ─────────────────────────────────────────────────────────────
#  Quality metrics (lightweight, for post-processing report)
# ─────────────────────────────────────────────────────────────

def mesh_quality_summary(nodes, triangles):
    """
    Quick quality summary of a triangulated mesh.

    Returns:
        dict with keys: n_nodes, n_elements, min_angle, mean_min_angle,
                        max_aspect_ratio, mean_skewness
    """
    p = nodes[triangles]
    angles = np.zeros((len(triangles), 3))
    for i in range(3):
        v1 = p[:, (i + 1) % 3] - p[:, i]
        v2 = p[:, (i + 2) % 3] - p[:, i]
        cos_a = np.sum(v1 * v2, axis=1) / (
            np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + 1e-30
        )
        angles[:, i] = np.degrees(np.arccos(np.clip(cos_a, -1, 1)))

    edges = np.zeros((len(triangles), 3))
    for i in range(3):
        edges[:, i] = np.linalg.norm(p[:, (i + 1) % 3] - p[:, i], axis=1)
    ar = edges.max(axis=1) / (edges.min(axis=1) + 1e-30)

    theta_min = angles.min(axis=1)
    theta_max = angles.max(axis=1)
    sk = np.maximum((theta_max - 60) / 120, (60 - theta_min) / 60)

    return {
        "n_nodes": len(nodes),
        "n_elements": len(triangles),
        "min_angle": float(angles.min()),
        "mean_min_angle": float(theta_min.mean()),
        "max_aspect_ratio": float(ar.max()),
        "mean_aspect_ratio": float(ar.mean()),
        "mean_skewness": float(sk.mean()),
        "max_skewness": float(sk.max()),
    }


# ─────────────────────────────────────────────────────────────
#  Mesh export — multiple formats
# ─────────────────────────────────────────────────────────────

def save_mesh(nodes, triangles, filepath, fmt=None, quality=None):
    """
    Save a 2D triangle mesh to disk.

    Supported formats (auto-detected from extension if fmt=None):
        .msh   — Gmsh ASCII v2.2  (importable by Gmsh, FEniCS, Elmer)
        .vtk   — Legacy VTK ASCII  (ParaView, Mayavi)
        .obj   — Wavefront OBJ     (universal 3-D viewer)
        .mesh  — Medit / INRIA      (TetGen, Gmsh)

    Args:
        nodes:     (N, 2) or (N, 3) coordinates
        triangles: (T, 3) triangle connectivity (0-indexed)
        filepath:  output path (str or Path)
        fmt:       force format (optional, otherwise from extension)
        quality:   dict from mesh_quality_summary (optional, written as comment)
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if fmt is None:
        fmt = filepath.suffix.lstrip(".")

    # Ensure 3-D coords for formats that need z
    if nodes.shape[1] == 2:
        nodes3 = np.column_stack([nodes, np.zeros(len(nodes))])
    else:
        nodes3 = nodes

    writers = {
        "msh": _write_gmsh_msh,
        "vtk": _write_vtk,
        "obj": _write_obj,
        "mesh": _write_medit,
    }

    writer = writers.get(fmt)
    if writer is None:
        raise ValueError(f"Unsupported mesh format: '{fmt}'. "
                         f"Choose from {list(writers.keys())}")

    writer(nodes3, triangles, filepath, quality)
    return str(filepath)


# ── Gmsh MSH v2.2 ──────────────────────────────────────────

def _write_gmsh_msh(nodes, triangles, path, quality):
    with open(path, "w") as f:
        # Header
        f.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")

        if quality:
            f.write(f"// MeshOptimiser quality: {quality}\n")

        # Nodes
        f.write("$Nodes\n")
        f.write(f"{len(nodes)}\n")
        for i, (x, y, z) in enumerate(nodes, start=1):
            f.write(f"{i} {x:.10g} {y:.10g} {z:.10g}\n")
        f.write("$EndNodes\n")

        # Elements (type 2 = 3-node triangle)
        f.write("$Elements\n")
        f.write(f"{len(triangles)}\n")
        for i, tri in enumerate(triangles, start=1):
            # elm-number elm-type num-tags <tags> node-list  (1-indexed)
            f.write(f"{i} 2 0 {tri[0]+1} {tri[1]+1} {tri[2]+1}\n")
        f.write("$EndElements\n")


# ── Legacy VTK ──────────────────────────────────────────────

def _write_vtk(nodes, triangles, path, quality):
    n_pts = len(nodes)
    n_tri = len(triangles)

    with open(path, "w") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("MeshOptimiser output\n")
        f.write("ASCII\n")
        f.write("DATASET UNSTRUCTURED_GRID\n")

        f.write(f"POINTS {n_pts} double\n")
        for x, y, z in nodes:
            f.write(f"{x:.10g} {y:.10g} {z:.10g}\n")

        f.write(f"\nCELLS {n_tri} {n_tri * 4}\n")
        for tri in triangles:
            f.write(f"3 {tri[0]} {tri[1]} {tri[2]}\n")

        f.write(f"\nCELL_TYPES {n_tri}\n")
        for _ in range(n_tri):
            f.write("5\n")  # VTK_TRIANGLE


# ── Wavefront OBJ ──────────────────────────────────────────

def _write_obj(nodes, triangles, path, quality):
    with open(path, "w") as f:
        f.write("# MeshOptimiser output\n")
        if quality:
            f.write(f"# Quality: {quality}\n")

        for x, y, z in nodes:
            f.write(f"v {x:.10g} {y:.10g} {z:.10g}\n")

        for tri in triangles:
            # OBJ is 1-indexed
            f.write(f"f {tri[0]+1} {tri[1]+1} {tri[2]+1}\n")


# ── Medit / INRIA .mesh ────────────────────────────────────

def _write_medit(nodes, triangles, path, quality):
    with open(path, "w") as f:
        f.write("MeshVersionFormatted 1\n")
        f.write("Dimension 2\n\n")

        f.write(f"Vertices\n{len(nodes)}\n")
        for x, y, z in nodes:
            f.write(f"{x:.10g} {y:.10g} 0\n")  # reference = 0
        f.write("\n")

        f.write(f"Triangles\n{len(triangles)}\n")
        for tri in triangles:
            f.write(f"{tri[0]+1} {tri[1]+1} {tri[2]+1} 0\n")  # 1-indexed, ref=0
        f.write("\nEnd\n")


# ─────────────────────────────────────────────────────────────
#  Convenience: full pipeline from merged indices → saved mesh
# ─────────────────────────────────────────────────────────────

def build_and_save_mesh(nodes, selected_indices, output_dir,
                        polygons=None, smooth_iterations=5,
                        boundary_node_indices=None,
                        formats=("msh", "vtk", "obj")):
    """
    End-to-end: triangulate → smooth → export.

    Args:
        nodes:                Full node set
        selected_indices:     Global indices of QAOA-selected nodes
        output_dir:           Directory for output files
        polygons:             Shapely polygons for triangle clipping
        smooth_iterations:    Laplacian smoothing passes (0=disable)
        boundary_node_indices: Global indices of boundary nodes (kept fixed during smoothing)
        formats:              Tuple of export formats

    Returns:
        mesh_info: dict with keys: nodes, triangles, quality, files
    """
    output_dir = Path(output_dir)

    # 1. Triangulate
    mesh_nodes, triangles, global_idx = triangulate_selected_nodes(
        nodes, selected_indices, polygons=polygons
    )

    # 2. Smooth (keep boundary nodes fixed)
    if smooth_iterations > 0 and len(triangles) > 0:
        fixed = np.zeros(len(mesh_nodes), dtype=bool)
        if boundary_node_indices is not None:
            # Find which mesh nodes are boundary nodes
            boundary_set = set(np.asarray(boundary_node_indices).tolist())
            for i, gi in enumerate(global_idx):
                if gi in boundary_set:
                    fixed[i] = True
        mesh_nodes = laplacian_smooth(
            mesh_nodes, triangles,
            iterations=smooth_iterations,
            weight=0.3,
            fixed_mask=fixed,
        )

    # 3. Quality report
    quality = mesh_quality_summary(mesh_nodes, triangles) if len(triangles) > 0 else {}

    # 4. Export
    saved_files = []
    for fmt in formats:
        fpath = output_dir / f"optimised_mesh.{fmt}"
        save_mesh(mesh_nodes, triangles, fpath, fmt=fmt, quality=quality)
        saved_files.append(str(fpath))

    return {
        "nodes": mesh_nodes,
        "triangles": triangles,
        "global_idx": global_idx,
        "quality": quality,
        "files": saved_files,
    }
