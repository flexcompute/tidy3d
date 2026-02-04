"""Polygon2D: polygon with optional curved edges and holes."""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pydantic.v1 as pydantic
import shapely

from tidy3d.components.base import skip_if_fields_missing
from tidy3d.components.geometry.base import Geometry
from tidy3d.components.geometry.geometry2d.base import Bound2D, Geometry2D
from tidy3d.components.types import ArrayFloat2D, Axis, Shapely, TYPE_TAG_STR, annotate_type
from tidy3d.exceptions import SetupError


def _extract_polygons(geom: Shapely) -> Optional[Shapely]:
    """Extract only polygon geometries from a shapely geometry.

    This is used after buffer(0) or make_valid() to filter out
    degenerate results like LineString or Point.

    Parameters
    ----------
    geom : Shapely
        Input geometry (may be Polygon, MultiPolygon, GeometryCollection, etc.)

    Returns
    -------
    Optional[Shapely]
        A Polygon or MultiPolygon containing only polygon parts,
        or None if no polygon parts exist.
    """
    if geom.is_empty:
        return None

    if geom.geom_type == "Polygon":
        return geom

    if geom.geom_type == "MultiPolygon":
        return geom

    if geom.geom_type == "GeometryCollection":
        polygons = []
        for g in geom.geoms:
            if g.geom_type == "Polygon":
                polygons.append(g)
            elif g.geom_type == "MultiPolygon":
                polygons.extend(g.geoms)
        if not polygons:
            return None
        if len(polygons) == 1:
            return polygons[0]
        return shapely.MultiPolygon(polygons)

    # For other types (Point, LineString, etc.), return None
    return None


def _bulge_to_arc_points(
    p1: tuple[float, float],
    p2: tuple[float, float],
    bulge: float,
    resolution: int,
) -> list[tuple[float, float]]:
    """Generate intermediate arc points for a bulged edge.

    The bulge parameter follows the DXF/CAD convention:
    bulge = tan(included_angle / 4)

    Parameters
    ----------
    p1 : tuple[float, float]
        Start point of the edge.
    p2 : tuple[float, float]
        End point of the edge.
    bulge : float
        Bulge parameter: tan(arc_angle / 4).
        - 0 = straight line
        - positive = counter-clockwise (CCW) arc, bulges to the left of edge direction
        - negative = clockwise (CW) arc, bulges to the right of edge direction
        - ±1 = semicircle
    resolution : int
        Number of segments for a full circle.

    Returns
    -------
    list[tuple[float, float]]
        Intermediate points along the arc (excludes p1 and p2).
    """
    if abs(bulge) < 1e-12:
        return []

    # Chord vector and length
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    chord_len = math.hypot(dx, dy)

    if chord_len < 1e-12:
        return []

    # Included angle from bulge: bulge = tan(theta/4)
    # theta is signed: positive for CCW, negative for CW
    theta = 4 * math.atan(bulge)
    abs_theta = abs(theta)

    # Radius: chord_len = 2 * radius * sin(|theta|/2)
    half_abs_theta = abs_theta / 2
    if abs(math.sin(half_abs_theta)) < 1e-12:
        return []
    radius = chord_len / (2 * math.sin(half_abs_theta))

    # Midpoint of chord
    mx = (p1[0] + p2[0]) / 2
    my = (p1[1] + p2[1]) / 2

    # Unit vector along chord
    ux = dx / chord_len
    uy = dy / chord_len

    # Unit normal to chord (perpendicular, 90° CCW rotation of chord direction)
    # Points to the LEFT of the chord direction
    nx = -uy
    ny = ux

    # Distance from chord midpoint to arc center along the normal
    d = radius * math.cos(half_abs_theta)

    # Center position: move from midpoint perpendicular to chord
    # For positive bulge (CCW arc): arc bulges LEFT, so center is to the RIGHT
    # For negative bulge (CW arc): arc bulges RIGHT, so center is to the LEFT
    # The center is on the OPPOSITE side of where the arc bulges
    sign = -1 if bulge > 0 else 1
    cx = mx + sign * d * nx
    cy = my + sign * d * ny

    # Start and end angles (from center to p1 and p2)
    start_angle = math.atan2(p1[1] - cy, p1[0] - cx)
    end_angle = math.atan2(p2[1] - cy, p2[0] - cx)

    # The angular span should equal theta (the included angle)
    # We need to traverse from start_angle to end_angle in the correct direction
    if bulge > 0:  # CCW arc: traverse counter-clockwise (increasing angle)
        # Normalize so that we go the SHORT way CCW
        angular_span = end_angle - start_angle
        if angular_span <= 0:
            angular_span += 2 * math.pi
        # But the span should be |theta|, not the complement
        # If we got the complement, flip it
        if angular_span > math.pi and abs_theta < math.pi:
            angular_span = angular_span - 2 * math.pi
    else:  # CW arc: traverse clockwise (decreasing angle)
        angular_span = end_angle - start_angle
        if angular_span >= 0:
            angular_span -= 2 * math.pi
        # If we got the complement, flip it
        if angular_span < -math.pi and abs_theta < math.pi:
            angular_span = angular_span + 2 * math.pi

    # Number of intermediate points based on arc length relative to full circle
    num_segments = max(2, int(abs(angular_span) / (2 * math.pi) * resolution))

    # Generate intermediate points (exclude endpoints)
    points = []
    for i in range(1, num_segments):
        t = i / num_segments
        angle = start_angle + t * angular_span
        x = cx + radius * math.cos(angle)
        y = cy + radius * math.sin(angle)
        points.append((x, y))

    return points


class Polygon2D(Geometry2D):
    """Polygon defined by vertices with optional curved edges and holes.

    The polygon is defined by an ordered list of vertices forming the exterior
    boundary. The polygon is auto-closed (first vertex should not be repeated
    at end). Each edge can optionally be curved using a bulge parameter.

    Vertex Winding Convention
    -------------------------
    - Exterior boundary: counter-clockwise (CCW) when viewed from +Z
    - Holes: clockwise (CW) when viewed from +Z

    Shapely will normalize winding automatically, but following this convention
    ensures predictable behavior.

    Bulge Convention (DXF/CAD standard)
    -----------------------------------
    bulge = tan(included_angle / 4)
    - 0 = straight edge
    - positive = CCW arc (bulges left of edge direction)
    - negative = CW arc (bulges right of edge direction)
    - ±1 = semicircle

    Example
    -------
    >>> # Simple triangle
    >>> tri = Polygon2D(vertices=[(0, 0), (1, 0), (0.5, 1)])
    >>>
    >>> # Rectangle with circular hole
    >>> rect_with_hole = Polygon2D(
    ...     vertices=[(-1, -1), (1, -1), (1, 1), (-1, 1)],
    ...     holes=(Circle2D(center=(0, 0), radius=0.3),),
    ... )
    >>>
    >>> # Square with rounded corners (bulged edges)
    >>> rounded = Polygon2D(
    ...     vertices=[(0, 0), (1, 0), (1, 1), (0, 1)],
    ...     bulges=(0.2, 0.2, 0.2, 0.2),
    ... )
    """

    vertices: ArrayFloat2D = pydantic.Field(
        ...,
        title="Vertices",
        description="Vertices as (N, 2) array defining the exterior boundary. "
        "Polygon auto-closes; do not repeat first vertex at end. "
        "Should use counter-clockwise (CCW) winding.",
        units="um",
    )

    bulges: Optional[tuple[float, ...]] = pydantic.Field(
        None,
        title="Bulges",
        description="Bulge value for each edge, where edge i connects vertex[i] to "
        "vertex[(i+1) % N]. bulge = tan(arc_angle/4): 0 = straight, "
        ">0 = CCW arc, <0 = CW arc, ±1 = semicircle. "
        "If provided, len(bulges) must equal len(vertices). "
        "None means all edges are straight.",
    )

    holes: tuple[annotate_type("Geometry2DType"), ...] = pydantic.Field(
        (),
        title="Holes",
        description="Geometries subtracted from the polygon interior. "
        "Can include Polygon2D with its own holes (nested). "
        "Hole boundaries should use clockwise (CW) winding.",
    )

    arc_resolution: pydantic.PositiveInt = pydantic.Field(
        32,
        title="Arc Resolution",
        description="Number of line segments used to approximate a full circle "
        "when tessellating bulged (curved) edges.",
    )

    @pydantic.validator("vertices", always=True, pre=True)
    def _coerce_and_validate_vertices(cls, val) -> ArrayFloat2D:
        """Ensure vertices is (N, 2) array with N >= 3, strip duplicates."""
        val = np.asarray(val, dtype=float)

        if val.ndim != 2 or val.shape[1] != 2:
            raise SetupError(
                f"Polygon2D.vertices must be shape (N, 2), got {val.shape}"
            )

        if val.shape[0] < 3:
            raise SetupError(
                f"Polygon2D requires at least 3 vertices, got {val.shape[0]}"
            )

        # Strip duplicate closing vertex if present (follow PolySlab convention)
        if np.allclose(val[0], val[-1], rtol=1e-10, atol=1e-14):
            val = val[:-1]

        # Remove consecutive duplicate vertices (can cause self-intersection errors)
        # Check distances between consecutive vertices, including wrap-around
        if val.shape[0] >= 2:
            # Compute squared distances between consecutive vertices
            diffs = np.diff(val, axis=0)  # (N-1, 2)
            dist_sq = np.sum(diffs**2, axis=1)  # (N-1,)
            
            # Also check wrap-around (last to first)
            wrap_diff = val[0] - val[-1]
            wrap_dist_sq = np.sum(wrap_diff**2)
            
            # Keep vertices where distance to previous is > tolerance
            # Use 1e-20 as threshold for squared distance (1e-10 linear)
            tol_sq = 1e-20
            keep_mask = np.concatenate([[True], dist_sq > tol_sq])  # First vertex always kept
            val = val[keep_mask]
            
            # Remove last vertex if it's duplicate of first (after filtering)
            if val.shape[0] >= 2:
                wrap_diff = val[0] - val[-1]
                if np.sum(wrap_diff**2) <= tol_sq:
                    val = val[:-1]

        if val.shape[0] < 3:
            raise SetupError(
                "Polygon2D requires at least 3 distinct vertices after "
                "removing duplicate vertices."
            )

        return val

    @pydantic.validator("bulges", always=True)
    @skip_if_fields_missing(["vertices"])
    def _validate_bulges_length(
        cls, val: Optional[tuple[float, ...]], values: dict
    ) -> Optional[tuple[float, ...]]:
        """Ensure bulges length matches vertices if provided."""
        if val is None:
            return None

        vertices = values.get("vertices")
        if vertices is not None and len(val) != len(vertices):
            raise SetupError(
                f"len(bulges)={len(val)} must equal len(vertices)={len(vertices)}"
            )

        return val

    def _tessellate_exterior(self) -> list[tuple[float, float]]:
        """Convert vertices + bulges to dense coordinate list.

        Returns
        -------
        list[tuple[float, float]]
            Dense list of coordinates representing the exterior boundary.
            Includes original vertices and interpolated arc points.
        """
        coords = []
        n = len(self.vertices)
        bulges = self.bulges if self.bulges is not None else (0.0,) * n

        for i in range(n):
            p1 = (float(self.vertices[i, 0]), float(self.vertices[i, 1]))
            next_idx = (i + 1) % n
            p2 = (float(self.vertices[next_idx, 0]), float(self.vertices[next_idx, 1]))
            bulge = bulges[i]

            coords.append(p1)

            if abs(bulge) > 1e-12:
                arc_points = _bulge_to_arc_points(p1, p2, bulge, self.arc_resolution)
                coords.extend(arc_points)

        return coords

    @property
    def bounds_2d(self) -> Bound2D:
        """Returns the 2D bounding box of the polygon.

        Accounts for bulged edges and does NOT include holes
        (holes are interior, don't affect bounding box).

        Returns
        -------
        Bound2D
            Tuple of ((min_x, min_y), (max_x, max_y)) coordinates.
        """
        # Use shapely for accurate bounds including arc bulges
        # Note: We only need exterior for bounds (holes don't extend bounds)
        exterior_coords = self._tessellate_exterior()
        exterior_ring = shapely.LinearRing(exterior_coords)
        minx, miny, maxx, maxy = exterior_ring.bounds
        return ((minx, miny), (maxx, maxy))

    def to_shapely(self) -> Shapely:
        """Convert to a shapely Polygon, including holes.

        Handles nested holes correctly: if a hole has its own holes,
        those become "islands" (solid regions) in the result.

        Uses shapely's difference operation to properly handle nested holes.
        When a hole itself has holes (interiors), those interior regions
        are NOT part of the hole's filled area, so they remain solid
        in the parent polygon (creating islands).

        Returns
        -------
        Shapely
            A shapely Polygon (or MultiPolygon if holes split the shape).
        """
        exterior_coords = self._tessellate_exterior()
        result = shapely.Polygon(exterior_coords)

        # Fix self-intersections using buffer(0)
        # This is equivalent to RF GUI's NonZero fill rule approach using Manifold3D
        # buffer(0) resolves self-intersections and always produces polygon output
        if not result.is_valid:
            result = result.buffer(0)
            # Extract only polygon parts if result is a collection
            result = _extract_polygons(result)
            if result is None or result.is_empty:
                raise SetupError(
                    "Polygon2D has self-intersecting geometry that collapsed to "
                    "non-polygon output. The source geometry may be degenerate."
                )

        if not self.holes:
            return result

        # Use difference operation to subtract each hole
        # This correctly handles nested holes: if hole A has interior B,
        # then A.to_shapely() has B already subtracted from its filled region.
        # So result.difference(A) leaves B as an island.
        for hole in self.holes:
            hole_shapely = hole.to_shapely()

            # Skip degenerate geometries
            if hole_shapely.is_empty:
                continue
            if hole_shapely.geom_type in ("Point", "LineString", "LinearRing"):
                continue

            result = result.difference(hole_shapely)

        # Final fix after hole subtraction
        if not result.is_valid:
            result = result.buffer(0)
            result = _extract_polygons(result)
            if result is None or result.is_empty:
                raise SetupError(
                    "Polygon2D became degenerate after hole subtraction. "
                    "Check that holes don't completely consume the polygon."
                )

        return result

    def to_3d_geometry(
        self,
        slab_bounds: tuple[float, float],
        axis: Axis = 2,
        sidewall_angle: float = 0.0,
    ) -> Geometry:
        """Convert to a 3D geometry by extrusion.

        Uses the existing from_shapely utility which handles polygons
        with holes via ClipOperation(difference).

        Parameters
        ----------
        slab_bounds : tuple[float, float]
            (z_min, z_max) bounds for the extruded geometry.
        axis : Axis
            Axis perpendicular to the 2D plane. Default is 2 (z-axis).
        sidewall_angle : float
            Angle of the sidewall in radians. Default is 0.0.

        Returns
        -------
        Geometry
            A 3D tidy3d Geometry (PolySlab, ClipOperation, or GeometryGroup).
        """
        from tidy3d.components.geometry.utils import from_shapely

        shapely_polygon = self.to_shapely()
        return from_shapely(
            shape=shapely_polygon,
            axis=axis,
            slab_bounds=slab_bounds,
            sidewall_angle=sidewall_angle,
        )

