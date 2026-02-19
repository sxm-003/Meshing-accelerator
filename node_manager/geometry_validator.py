import numpy as np
from shapely import contains_xy, intersects_xy
from shapely.geometry import Polygon
from shapely.ops import unary_union


def _geometry_to_polygons(geom):
    if geom.is_empty:
        return []
    if geom.geom_type == "Polygon":
        return [geom]
    if geom.geom_type == "MultiPolygon":
        return list(geom.geoms)
    return [g for g in getattr(geom, "geoms", []) if g.geom_type == "Polygon"]


def _explode_to_simple_rings(polygons, area_tol=1e-12):
    simple = []
    for poly in polygons:
        ext_poly = Polygon(poly.exterior)
        if not ext_poly.is_empty and ext_poly.area > area_tol:
            simple.append(ext_poly)

        for hole in poly.interiors:
            hole_poly = Polygon(hole)
            if not hole_poly.is_empty and hole_poly.area > area_tol:
                simple.append(hole_poly)
    return simple


def _dedupe_overlapping_loops(
    polygons,
    area_tol=1e-12,
    overlap_ratio_tol=0.98,
    area_similarity_tol=0.95,
):
    cleaned = []
    ordered = sorted(polygons, key=lambda g: g.area, reverse=True)
    for poly in ordered:
        if poly.is_empty or poly.area <= area_tol:
            continue
        poly = poly.buffer(0)
        if poly.is_empty or poly.area <= area_tol:
            continue

        is_duplicate = False
        for kept in cleaned:
            inter = poly.intersection(kept).area
            min_area = min(poly.area, kept.area)
            max_area = max(poly.area, kept.area)
            if max_area <= area_tol:
                continue
            area_similarity = min_area / max_area
            overlap = inter / min_area if min_area > area_tol else 0.0
            # Only merge loops that are almost the same footprint and area.
            if area_similarity >= area_similarity_tol and overlap >= overlap_ratio_tol:
                is_duplicate = True
                break
        if not is_duplicate:
            cleaned.append(poly)
    return cleaned


def _containment_depth(idx, loops, contain_eps=1e-9):
    target = loops[idx]
    depth = 0
    for j, other in enumerate(loops):
        if j == idx:
            continue
        # Containment is evaluated on the whole loop polygon, not a sample point,
        # to avoid misclassifying outers whose representative point lands in a hole.
        if other.buffer(contain_eps).contains(target):
            depth += 1
    return depth


def build_domain_from_loops(loop_polygons, area_tol=1e-12):
    loops = _explode_to_simple_rings(loop_polygons, area_tol=area_tol)
    loops = _dedupe_overlapping_loops(loops, area_tol=area_tol)
    if not loops:
        raise ValueError("No valid polygon loops found")

    material_loops = []
    hole_loops = []
    for i, loop in enumerate(loops):
        depth = _containment_depth(i, loops)
        if depth % 2 == 0:
            material_loops.append(loop)
        else:
            hole_loops.append(loop)

    if not material_loops:
        raise ValueError("Could not identify material region from loops")

    domain = unary_union(material_loops).buffer(0)
    if hole_loops:
        holes = unary_union(hole_loops).buffer(0)
        domain = domain.difference(holes).buffer(0)

    polygons = _geometry_to_polygons(domain)
    if not polygons:
        raise ValueError("Material domain is empty after hole subtraction")
    return domain, polygons


class GeometryValidator:
    def __init__(self, loop_polygons, area_tol=1e-12):
        self.domain, self.polygons = build_domain_from_loops(
            loop_polygons,
            area_tol=area_tol,
        )
        self.bounds = self.domain.bounds

    def mask_inside(self, points, strict=True):
        pts = np.asarray(points, dtype=float)
        if pts.size == 0:
            return np.zeros((0,), dtype=bool)
        if pts.ndim != 2 or pts.shape[1] < 2:
            raise ValueError("points must have shape (N,2)")

        x = pts[:, 0]
        y = pts[:, 1]
        if strict:
            return contains_xy(self.domain, x, y)
        return intersects_xy(self.domain, x, y)

    def filter_points(self, points, strict=True):
        pts = np.asarray(points, dtype=float)
        if pts.size == 0:
            return np.empty((0, 2), dtype=float)
        mask = self.mask_inside(pts, strict=strict)
        return pts[mask]
