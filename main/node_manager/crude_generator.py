import ezdxf
from ezdxf import recover
import numpy as np
import math
from ezdxf.math import BSpline
from shapely.geometry import LineString, MultiLineString
from shapely.ops import polygonize
from shapely.geometry import Point
import plotly.io as pio
import plotly.graph_objects as go


import ezdxf
from ezdxf import recover
import sys

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
                np.array(spline.point(t))
                for t in np.linspace(0, 1, curve_samples)
            ]

            for i in range(len(pts) - 1):
                segments.append((pts[i], pts[i + 1]))

    return segments


def segments_to_polygons(segments):
    lines = [LineString([p0, p1]) for p0, p1 in segments]
    multiline = MultiLineString(lines)
    polygons = list(polygonize(multiline))
    return polygons

def jittered_grid_shapely(polygons, L, jitter_frac=0.3, seed=0):
    np.random.seed(seed)

    # Bounding box
    minx, miny, maxx, maxy = polygons[0].bounds
    for poly in polygons[1:]:
        bx = poly.bounds
        minx, miny = min(minx, bx[0]), min(miny, bx[1])
        maxx, maxy = max(maxx, bx[2]), max(maxy, bx[3])

    h = L / 2.0
    xs = np.arange(minx, maxx + h, h)
    ys = np.arange(miny, maxy + h, h)

    pts = []
    for x in xs:
        for y in ys:
            dx = (np.random.rand() - 0.5) * jitter_frac * h
            dy = (np.random.rand() - 0.5) * jitter_frac * h
            p = Point(x + dx, y + dy)

            # IMPORTANT: use covers, not contains
            if any(poly.covers(p) for poly in polygons):
                pts.append([p.x, p.y])

    pts = np.array(pts, dtype=float)
    if pts.size == 0:
        return np.empty((0, 2))
    return pts


def uniform_grid_shapely(polygons, L, seed=0):
    """
    Generate uniform (non-jittered) grid nodes within polygons.
    This is the preferred method as it provides more consistent and predictable node spacing.
    """
    np.random.seed(seed)

    # Bounding box
    minx, miny, maxx, maxy = polygons[0].bounds
    for poly in polygons[1:]:
        bx = poly.bounds
        minx, miny = min(minx, bx[0]), min(miny, bx[1])
        maxx, maxy = max(maxx, bx[2]), max(maxy, bx[3])

    h = L / 2.0
    xs = np.arange(minx, maxx + h, h)
    ys = np.arange(miny, maxy + h, h)

    pts = []
    for x in xs:
        for y in ys:
            # No jitter - use pure uniform grid
            p = Point(x, y)

            # Use covers for robust point-in-polygon test
            if any(poly.covers(p) for poly in polygons):
                pts.append([p.x, p.y])

    pts = np.array(pts, dtype=float)
    if pts.size == 0:
        return np.empty((0, 2))
    return pts


def sample_boundaries_shapely(polygons, spacing):
    boundary_pts = []

    for poly in polygons:
        ext = np.array(poly.exterior.coords)
        ext = ext[:, :2]  
        for i in range(len(ext) - 1):
            p0, p1 = ext[i], ext[i + 1]
            length = np.linalg.norm(p1 - p0)
            n = max(2, int(length / spacing))
            for t in np.linspace(0, 1, n):
                boundary_pts.append(p0 * (1 - t) + p1 * t)

        for hole in poly.interiors:
            hcoords = np.array(hole.coords)
            hcoords = hcoords[:, :2]  
            for i in range(len(hcoords) - 1):
                p0, p1 = hcoords[i], hcoords[i + 1]
                length = np.linalg.norm(p1 - p0)
                n = max(2, int(length / spacing))
                for t in np.linspace(0, 1, n):
                    boundary_pts.append(p0 * (1 - t) + p1 * t)

    boundary_pts = np.array(boundary_pts, dtype=float)

    if boundary_pts.size == 0:
        return np.empty((0, 2))

    return boundary_pts

def offset_boundary_layers(polygons, offsets, spacing):
    pts = []

    for poly in polygons:
        for d in offsets:
            inner = poly.buffer(-d)
            if inner.is_empty:
                continue

            if inner.geom_type == "Polygon":
                rings = [inner.exterior]
            else:
                rings = [g.exterior for g in inner.geoms]

            for ring in rings:
                coords = np.array(ring.coords)[:, :2]
                for i in range(len(coords) - 1):
                    p0, p1 = coords[i], coords[i + 1]
                    L = np.linalg.norm(p1 - p0)
                    n = max(2, int(L / spacing))
                    for t in np.linspace(0, 1, n):
                        pts.append(p0 * (1 - t) + p1 * t)

    return np.array(pts)

def adaptive_jittered_grid_shapely(
    polygons,
    L_bulk,
    L_boundary,
    boundary_band,
    jitter_frac=0.3,
    seed=0,
):
    np.random.seed(seed)

    minx, miny, maxx, maxy = polygons[0].bounds
    for poly in polygons[1:]:
        bx = poly.bounds
        minx, miny = min(minx, bx[0]), min(miny, bx[1])
        maxx, maxy = max(maxx, bx[2]), max(maxy, bx[3])

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

    return np.array(pts)

def generate_crude_nodes(path):
    msp = load_dxf(path)
    segments = extract_segments(msp)
    polygons = segments_to_polygons(segments)

    boundary_nodes = sample_boundaries_shapely(
        polygons,
        spacing=0.05
    )

    offset_nodes = offset_boundary_layers(
        polygons,
        offsets=[0.05, 0.12, 0.25],
        spacing=0.1
    )

    # Use UNIFORM GRID instead of jittered/adaptive
    interior_nodes = uniform_grid_shapely(
        polygons,
        L=0.4  # Characteristic length scale
    )

    nodes = np.vstack([
        interior_nodes,
        offset_nodes,
        boundary_nodes
    ])

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



