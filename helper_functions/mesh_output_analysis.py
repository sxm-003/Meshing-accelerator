from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.graph_objects as go


DEFAULT_HEATMAP_COLORSCALE = [
    [0.0, "#1a9850"],
    [0.5, "#fee08b"],
    [1.0, "#d73027"],
]


@dataclass(frozen=True)
class MeshLoadResult:
    mesh_dir: Path
    mesh_file: Path
    mesh_format: str
    nodes: np.ndarray
    triangles: np.ndarray


def discover_mesh_runs(outputs_root: str | Path) -> pd.DataFrame:
    outputs_root = Path(outputs_root).expanduser().resolve()
    rows: list[dict[str, object]] = []

    for run_dir in sorted(outputs_root.iterdir()) if outputs_root.exists() else []:
        if not run_dir.is_dir():
            continue

        mesh_dir = run_dir / "mesh"
        if not mesh_dir.is_dir():
            continue

        vtk_file = mesh_dir / "optimised_mesh.vtk"
        msh_file = mesh_dir / "optimised_mesh.msh"
        obj_file = mesh_dir / "optimised_mesh.obj"

        if not (vtk_file.exists() or msh_file.exists() or obj_file.exists()):
            continue

        existing_files = [path for path in (vtk_file, msh_file, obj_file) if path.exists()]
        latest_mtime = max(path.stat().st_mtime for path in existing_files)

        rows.append(
            {
                "run_id": run_dir.name,
                "run_dir": str(run_dir),
                "mesh_dir": str(mesh_dir),
                "vtk_file": str(vtk_file) if vtk_file.exists() else "",
                "msh_file": str(msh_file) if msh_file.exists() else "",
                "obj_file": str(obj_file) if obj_file.exists() else "",
                "mesh_modified_at": pd.Timestamp.fromtimestamp(latest_mtime),
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "run_id",
                "run_dir",
                "mesh_dir",
                "vtk_file",
                "msh_file",
                "obj_file",
                "mesh_modified_at",
            ]
        )

    return pd.DataFrame(rows).sort_values("mesh_modified_at", ascending=False).reset_index(drop=True)


def resolve_mesh_target(path_like: str | Path, outputs_root: str | Path) -> Path:
    outputs_root = Path(outputs_root).expanduser().resolve()
    candidate = Path(path_like).expanduser()

    if not candidate.is_absolute():
        run_candidate = outputs_root / candidate
        if run_candidate.exists():
            candidate = run_candidate
        else:
            candidate = (Path.cwd() / candidate).resolve()
    else:
        candidate = candidate.resolve()

    if candidate.is_file():
        return candidate

    if candidate.is_dir() and candidate.name == "mesh":
        return candidate

    if candidate.is_dir() and (candidate / "mesh").is_dir():
        return candidate / "mesh"

    if candidate.is_dir() and any((candidate / name).exists() for name in ("optimised_mesh.vtk", "optimised_mesh.msh")):
        return candidate

    available = discover_mesh_runs(outputs_root)
    suggestion = ""
    if not available.empty:
        suggestion = "\nAvailable runs with mesh outputs:\n" + "\n".join(
            f"- {run_id}" for run_id in available["run_id"].head(10)
        )

    raise FileNotFoundError(f"Could not resolve a mesh directory or mesh file from: {candidate}{suggestion}")


def load_mesh_from_outputs(path_like: str | Path, outputs_root: str | Path) -> MeshLoadResult:
    target = resolve_mesh_target(path_like, outputs_root)

    if target.is_file():
        mesh_file = target
        mesh_dir = target.parent
    else:
        mesh_dir = target
        vtk_file = mesh_dir / "optimised_mesh.vtk"
        msh_file = mesh_dir / "optimised_mesh.msh"
        if vtk_file.exists():
            mesh_file = vtk_file
        elif msh_file.exists():
            mesh_file = msh_file
        else:
            raise FileNotFoundError(f"No supported mesh file found in {mesh_dir}")

    suffix = mesh_file.suffix.lower()
    if suffix == ".vtk":
        nodes, triangles = load_legacy_vtk_triangle_mesh(mesh_file)
        mesh_format = "vtk"
    elif suffix == ".msh":
        nodes, triangles = load_gmsh_msh_v22_triangle_mesh(mesh_file)
        mesh_format = "msh"
    else:
        raise ValueError(f"Unsupported mesh format: {mesh_file.suffix}")

    return MeshLoadResult(
        mesh_dir=mesh_dir,
        mesh_file=mesh_file,
        mesh_format=mesh_format,
        nodes=nodes,
        triangles=triangles,
    )


def load_legacy_vtk_triangle_mesh(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    path = Path(path)
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]

    try:
        points_idx = next(i for i, line in enumerate(lines) if line.startswith("POINTS "))
        cells_idx = next(i for i, line in enumerate(lines) if line.startswith("CELLS "))
    except StopIteration as exc:
        raise ValueError(f"{path} is not a supported legacy ASCII VTK mesh") from exc

    point_count = int(lines[points_idx].split()[1])
    cell_count = int(lines[cells_idx].split()[1])

    points = np.array(
        [[float(value) for value in lines[points_idx + 1 + i].split()[:3]] for i in range(point_count)],
        dtype=float,
    )

    triangles: list[list[int]] = []
    for line in lines[cells_idx + 1 : cells_idx + 1 + cell_count]:
        values = [int(value) for value in line.split()]
        if values and values[0] == 3:
            triangles.append(values[1:4])

    return points[:, :2], np.asarray(triangles, dtype=int)


def load_gmsh_msh_v22_triangle_mesh(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    path = Path(path)
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]

    node_index: dict[int, int] = {}
    node_rows: list[list[float]] = []
    triangles: list[list[int]] = []

    i = 0
    while i < len(lines):
        line = lines[i]

        if line == "$Nodes":
            node_count = int(lines[i + 1])
            for offset in range(node_count):
                parts = lines[i + 2 + offset].split()
                node_id = int(parts[0])
                node_index[node_id] = len(node_rows)
                node_rows.append([float(parts[1]), float(parts[2]), float(parts[3])])
            i += node_count + 2
            continue

        if line == "$Elements":
            element_count = int(lines[i + 1])
            for offset in range(element_count):
                parts = lines[i + 2 + offset].split()
                element_type = int(parts[1])
                num_tags = int(parts[2])
                node_ids = [int(value) for value in parts[3 + num_tags :]]
                if element_type == 2 and len(node_ids) >= 3:
                    triangles.append([node_index[node_id] for node_id in node_ids[:3]])
            i += element_count + 2
            continue

        i += 1

    if not node_rows or not triangles:
        raise ValueError(f"{path} does not contain node and triangle data")

    return np.asarray(node_rows, dtype=float)[:, :2], np.asarray(triangles, dtype=int)


def calculate_triangle_metrics(nodes: np.ndarray, triangles: np.ndarray) -> pd.DataFrame:
    if len(nodes) == 0 or len(triangles) == 0:
        return pd.DataFrame(
            columns=[
                "triangle_id",
                "node_0",
                "node_1",
                "node_2",
                "centroid_x",
                "centroid_y",
                "area",
                "edge_min",
                "edge_max",
                "min_angle_deg",
                "max_angle_deg",
                "aspect_ratio",
                "skewness",
            ]
        )

    tri_pts = np.asarray(nodes, dtype=float)[np.asarray(triangles, dtype=int)]

    p0 = tri_pts[:, 0]
    p1 = tri_pts[:, 1]
    p2 = tri_pts[:, 2]

    a = np.linalg.norm(p1 - p2, axis=1)
    b = np.linalg.norm(p2 - p0, axis=1)
    c = np.linalg.norm(p0 - p1, axis=1)

    edge_lengths = np.column_stack([a, b, c])
    edge_min = edge_lengths.min(axis=1)
    edge_max = edge_lengths.max(axis=1)

    double_area = np.abs(
        (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1])
        - (p1[:, 1] - p0[:, 1]) * (p2[:, 0] - p0[:, 0])
    )
    area = 0.5 * double_area

    eps = 1.0e-30
    alpha = np.degrees(np.arccos(np.clip((b**2 + c**2 - a**2) / (2.0 * b * c + eps), -1.0, 1.0)))
    beta = np.degrees(np.arccos(np.clip((a**2 + c**2 - b**2) / (2.0 * a * c + eps), -1.0, 1.0)))
    gamma = 180.0 - alpha - beta
    angles = np.column_stack([alpha, beta, gamma])

    min_angle_deg = angles.min(axis=1)
    max_angle_deg = angles.max(axis=1)
    aspect_ratio = edge_max / (edge_min + eps)
    skewness = np.maximum((max_angle_deg - 60.0) / 120.0, (60.0 - min_angle_deg) / 60.0)

    centroids = tri_pts.mean(axis=1)

    return pd.DataFrame(
        {
            "triangle_id": np.arange(len(triangles), dtype=int),
            "node_0": triangles[:, 0],
            "node_1": triangles[:, 1],
            "node_2": triangles[:, 2],
            "centroid_x": centroids[:, 0],
            "centroid_y": centroids[:, 1],
            "area": area,
            "edge_min": edge_min,
            "edge_max": edge_max,
            "min_angle_deg": min_angle_deg,
            "max_angle_deg": max_angle_deg,
            "aspect_ratio": aspect_ratio,
            "skewness": skewness,
        }
    )


def build_quality_summary(metrics: pd.DataFrame, n_nodes: int, mesh: MeshLoadResult) -> pd.DataFrame:
    if metrics.empty:
        rows = [
            ("mesh_file", str(mesh.mesh_file)),
            ("mesh_format", mesh.mesh_format),
            ("n_nodes", int(n_nodes)),
            ("n_triangles", 0),
        ]
    else:
        rows = [
            ("mesh_file", str(mesh.mesh_file)),
            ("mesh_format", mesh.mesh_format),
            ("n_nodes", int(n_nodes)),
            ("n_triangles", int(len(metrics))),
            ("area_min", float(metrics["area"].min())),
            ("area_median", float(metrics["area"].median())),
            ("area_max", float(metrics["area"].max())),
            ("min_angle_min_deg", float(metrics["min_angle_deg"].min())),
            ("min_angle_mean_deg", float(metrics["min_angle_deg"].mean())),
            ("min_angle_p05_deg", float(metrics["min_angle_deg"].quantile(0.05))),
            ("aspect_ratio_mean", float(metrics["aspect_ratio"].mean())),
            ("aspect_ratio_p95", float(metrics["aspect_ratio"].quantile(0.95))),
            ("aspect_ratio_max", float(metrics["aspect_ratio"].max())),
            ("skewness_mean", float(metrics["skewness"].mean())),
            ("skewness_p95", float(metrics["skewness"].quantile(0.95))),
            ("skewness_max", float(metrics["skewness"].max())),
        ]

    return pd.DataFrame(rows, columns=["metric", "value"])


def plot_mesh_wireframe(
    nodes: np.ndarray,
    triangles: np.ndarray,
    *,
    title: str = "Mesh wireframe",
    show_nodes: bool = True,
    max_node_markers: int = 10000,
) -> go.Figure:
    edge_x, edge_y = _build_edge_coordinate_lists(nodes, triangles)

    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=edge_x,
            y=edge_y,
            mode="lines",
            name="mesh edges",
            line={"color": "#1f2937", "width": 1},
            hoverinfo="skip",
        )
    )

    if show_nodes and len(nodes) <= max_node_markers:
        fig.add_trace(
            go.Scattergl(
                x=nodes[:, 0],
                y=nodes[:, 1],
                mode="markers",
                name="nodes",
                marker={"size": 4, "color": "#ef4444", "opacity": 0.8},
                customdata=np.arange(len(nodes), dtype=int),
                hovertemplate=(
                    "Node %{customdata}<br>"
                    "x=%{x:.6f}<br>"
                    "y=%{y:.6f}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=title,
        template="plotly_white",
        height=800,
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
        xaxis={"title": "x"},
        yaxis={"title": "y", "scaleanchor": "x", "scaleratio": 1},
    )
    return fig


def plot_triangle_metric_heatmap(
    nodes: np.ndarray,
    triangles: np.ndarray,
    values: Iterable[float],
    *,
    metric_name: str,
    title: str | None = None,
    cmin: float | None = None,
    cmax: float | None = None,
    colorscale: list[list[float | str]] | None = None,
) -> go.Figure:
    values = np.asarray(list(values), dtype=float)
    nodes = np.asarray(nodes, dtype=float)
    triangles = np.asarray(triangles, dtype=int)

    if len(values) != len(triangles):
        raise ValueError("Metric values must have one entry per triangle")

    if cmin is None:
        cmin = float(values.min()) if len(values) else 0.0
    if cmax is None:
        cmax = float(values.max()) if len(values) else 1.0
    if np.isclose(cmin, cmax):
        cmax = cmin + 1.0e-9

    edge_x, edge_y, edge_z = _build_edge_coordinate_lists_3d(nodes, triangles)

    fig = go.Figure()
    fig.add_trace(
        go.Mesh3d(
            x=nodes[:, 0],
            y=nodes[:, 1],
            z=np.zeros(len(nodes), dtype=float),
            i=triangles[:, 0],
            j=triangles[:, 1],
            k=triangles[:, 2],
            intensity=values,
            intensitymode="cell",
            flatshading=True,
            colorscale=colorscale or DEFAULT_HEATMAP_COLORSCALE,
            cmin=cmin,
            cmax=cmax,
            colorbar={"title": metric_name},
            hovertemplate=f"{metric_name}: %{{intensity:.6f}}<extra></extra>",
            showscale=True,
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=edge_x,
            y=edge_y,
            z=edge_z,
            mode="lines",
            name="mesh edges",
            line={"color": "rgba(0, 0, 0, 0.35)", "width": 2},
            hoverinfo="skip",
            showlegend=False,
        )
    )

    fig.update_layout(
        title=title or metric_name,
        template="plotly_white",
        height=850,
        margin={"l": 0, "r": 0, "t": 60, "b": 0},
        scene={
            "aspectmode": "data",
            "camera": {
                "eye": {"x": 0.0, "y": 0.0, "z": 2.5},
                "center": {"x": 0.0, "y": 0.0, "z": 0.0},
                "up": {"x": 0.0, "y": 1.0, "z": 0.0},
                "projection": {"type": "orthographic"},
            },
            "xaxis": {"visible": False},
            "yaxis": {"visible": False},
            "zaxis": {"visible": False},
        },
    )
    return fig


def _unique_edges(triangles: np.ndarray) -> np.ndarray:
    triangles = np.asarray(triangles, dtype=int)
    edges = np.vstack(
        [
            triangles[:, [0, 1]],
            triangles[:, [1, 2]],
            triangles[:, [2, 0]],
        ]
    )
    edges = np.sort(edges, axis=1)
    return np.unique(edges, axis=0)


def _build_edge_coordinate_lists(nodes: np.ndarray, triangles: np.ndarray) -> tuple[list[float | None], list[float | None]]:
    edges = _unique_edges(triangles)
    edge_x: list[float | None] = []
    edge_y: list[float | None] = []

    for start_idx, end_idx in edges:
        start = nodes[start_idx]
        end = nodes[end_idx]
        edge_x.extend([float(start[0]), float(end[0]), None])
        edge_y.extend([float(start[1]), float(end[1]), None])

    return edge_x, edge_y


def _build_edge_coordinate_lists_3d(
    nodes: np.ndarray,
    triangles: np.ndarray,
) -> tuple[list[float | None], list[float | None], list[float | None]]:
    edge_x, edge_y = _build_edge_coordinate_lists(nodes, triangles)
    edge_z = [0.0 if value is not None else None for value in edge_x]
    return edge_x, edge_y, edge_z


__all__ = [
    "DEFAULT_HEATMAP_COLORSCALE",
    "MeshLoadResult",
    "build_quality_summary",
    "calculate_triangle_metrics",
    "discover_mesh_runs",
    "load_gmsh_msh_v22_triangle_mesh",
    "load_legacy_vtk_triangle_mesh",
    "load_mesh_from_outputs",
    "plot_mesh_wireframe",
    "plot_triangle_metric_heatmap",
    "resolve_mesh_target",
]
