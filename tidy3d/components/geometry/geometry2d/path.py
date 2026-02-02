"""Path2D: stroked path of connected line and arc segments."""

from __future__ import annotations

import math
from typing import Literal, Optional, Union

import numpy as np
import pydantic.v1 as pydantic
import shapely

from tidy3d.components.base import Tidy3dBaseModel, skip_if_fields_missing
from tidy3d.components.geometry.base import Geometry
from tidy3d.components.geometry.geometry2d.base import Bound2D, Geometry2D
from tidy3d.components.types import Axis, Shapely
from tidy3d.exceptions import SetupError


class ArcSegment(Tidy3dBaseModel):
    """Parameters defining an arc segment in a Path2D.

    An arc segment connects two consecutive vertices via a circular arc
    rather than a straight line. The arc is defined by its center point
    and direction (clockwise or counter-clockwise).

    Example
    -------
    >>> arc = ArcSegment(center=(5.0, 0.0), clockwise=False)
    """

    center: tuple[float, float] = pydantic.Field(
        ...,
        title="Center",
        description="Center point of the circular arc.",
        units="um",
    )

    clockwise: bool = pydantic.Field(
        False,
        title="Clockwise",
        description="If True, arc goes clockwise from start to end. "
        "If False (default), arc goes counter-clockwise.",
    )


class Path2D(Geometry2D):
    """Stroked path of connected line and arc segments.

    Represents PCB traces, waveguides, and routing paths. The path is defined
    by a sequence of vertices, with optional arc segments between them.
    Supports constant width or variable width (tapers).

    The path is rendered as a stroked shape with configurable end caps and
    corner styles. For variable width paths, the width can interpolate
    linearly between vertices or step (constant per segment).

    Parameters
    ----------
    vertices : tuple[tuple[float, float], ...]
        Ordered path vertices including start point. Must have at least 2 points.
    arcs : dict[int, ArcSegment]
        Segment index to arc parameters. Segment i connects vertex[i] to
        vertex[i+1]. Missing indices are straight line segments.
    width : float or tuple[float, ...]
        Constant width (float) or per-vertex widths (tuple). Per-vertex
        widths must have same length as vertices.
    width_interp : Literal["linear", "step"]
        Width interpolation mode. "linear" interpolates smoothly between
        vertex widths. "step" uses constant width per segment (from start vertex).
    end_cap : Literal["round", "square", "flat"]
        Style for path endpoints.
    corner_style : Literal["round", "bevel"]
        Style for corners between segments.
    arc_resolution : int
        Number of line segments per full circle when tessellating arcs.

    Example
    -------
    >>> # Simple polyline trace
    >>> trace = Path2D(vertices=[(0, 0), (10, 0), (10, 5)], width=0.15)
    >>>
    >>> # Path with arc segment
    >>> curved = Path2D(
    ...     vertices=[(0, 0), (10, 0), (10, 10)],
    ...     arcs={1: ArcSegment(center=(10, 5))},
    ...     width=0.2,
    ... )
    >>>
    >>> # Linear taper
    >>> taper = Path2D(vertices=[(0, 0), (50, 0)], width=(0.5, 2.0))
    >>>
    >>> # Step taper (width changes abruptly at vertices)
    >>> step_taper = Path2D(
    ...     vertices=[(0, 0), (10, 0), (20, 0)],
    ...     width=(0.5, 1.0, 1.0),
    ...     width_interp="step",
    ... )
    """

    vertices: tuple[tuple[float, float], ...] = pydantic.Field(
        ...,
        title="Vertices",
        description="Ordered path vertices including start point. "
        "Must have at least 2 points.",
        units="um",
    )

    arcs: dict[int, ArcSegment] = pydantic.Field(
        default_factory=dict,
        title="Arc Segments",
        description="Segment index → arc parameters. Segment i connects "
        "vertex[i] to vertex[i+1]. Missing indices are line segments.",
    )

    width: Union[pydantic.PositiveFloat, tuple[pydantic.PositiveFloat, ...]] = pydantic.Field(
        ...,
        title="Width",
        description="Constant width (float) or per-vertex widths (tuple). "
        "Per-vertex widths must match vertices length.",
        units="um",
    )

    width_interp: Literal["linear", "step"] = pydantic.Field(
        "linear",
        title="Width Interpolation",
        description="'linear': width interpolates smoothly between vertices. "
        "'step': width is constant per segment (uses start vertex width).",
    )

    end_cap: Literal["round", "square", "flat"] = pydantic.Field(
        "round",
        title="End Cap Style",
        description="Style for path endpoints: 'round' (semicircle), "
        "'square' (extends by half-width), 'flat' (no extension).",
    )

    corner_style: Literal["round", "bevel"] = pydantic.Field(
        "round",
        title="Corner Style",
        description="Style for corners between segments: 'round' (circular fill), "
        "'bevel' (flat triangular fill).",
    )

    arc_resolution: pydantic.PositiveInt = pydantic.Field(
        64,
        title="Arc Resolution",
        description="Number of line segments used to approximate a full circle "
        "when tessellating arc segments.",
    )

    @pydantic.validator("vertices", always=True, pre=True)
    def _validate_vertices(cls, val):
        """Ensure vertices is tuple of coordinate pairs with at least 2 points."""
        if hasattr(val, "tolist"):
            val = tuple(tuple(p) for p in val.tolist())
        else:
            val = tuple(tuple(p) for p in val)

        if len(val) < 2:
            raise SetupError("Path2D requires at least 2 vertices.")

        for i, pt in enumerate(val):
            if len(pt) != 2:
                raise SetupError(
                    f"Vertex {i} must have 2 coordinates, got {len(pt)}."
                )

        return val

    @pydantic.validator("width", always=True)
    @skip_if_fields_missing(["vertices"])
    def _validate_width(cls, val, values):
        """Ensure per-vertex widths match vertices length."""
        if isinstance(val, tuple):
            vertices = values.get("vertices", ())
            if len(val) != len(vertices):
                raise SetupError(
                    f"Per-vertex width length ({len(val)}) must match "
                    f"vertices length ({len(vertices)})."
                )
            for i, w in enumerate(val):
                if w <= 0:
                    raise SetupError(f"Width at index {i} must be positive, got {w}.")
        elif val <= 0:
            raise SetupError(f"Width must be positive, got {val}.")
        return val

    @pydantic.validator("arcs", always=True)
    @skip_if_fields_missing(["vertices"])
    def _validate_arcs(cls, val, values):
        """Ensure arc indices are within segment range."""
        vertices = values.get("vertices", ())
        n_segments = len(vertices) - 1

        for idx in val.keys():
            if not (0 <= idx < n_segments):
                raise SetupError(
                    f"Arc segment index {idx} out of range [0, {n_segments})."
                )
        return val

    @property
    def num_segments(self) -> int:
        """Number of segments in the path."""
        return len(self.vertices) - 1

    def width_at_vertex(self, index: int) -> float:
        """Get width at a specific vertex index.

        Parameters
        ----------
        index : int
            Vertex index (0 to len(vertices)-1).

        Returns
        -------
        float
            Width at the specified vertex.
        """
        if isinstance(self.width, tuple):
            return self.width[index]
        return self.width

    def width_at_segment(self, segment_index: int, t: float = 0.0) -> float:
        """Get width at position t along a segment.

        Parameters
        ----------
        segment_index : int
            Segment index (0 to num_segments-1).
        t : float
            Position along segment, 0.0 = start, 1.0 = end.

        Returns
        -------
        float
            Interpolated width at the specified position.
        """
        w_start = self.width_at_vertex(segment_index)

        if self.width_interp == "step" or t == 0.0:
            return w_start

        w_end = self.width_at_vertex(segment_index + 1)
        return w_start + t * (w_end - w_start)

    @property
    def bounds_2d(self) -> Bound2D:
        """Returns the 2D bounding box including stroke width.

        Returns
        -------
        Bound2D
            Tuple of ((min_x, min_y), (max_x, max_y)) coordinates.
        """
        shape = self.to_shapely()
        minx, miny, maxx, maxy = shape.bounds
        return ((minx, miny), (maxx, maxy))

    def to_shapely(self) -> Shapely:
        """Convert to a shapely Polygon representing the stroked path.

        Creates a union of all segment shapes plus corner shapes at junctions.
        Each segment is an independent polygon with appropriate end caps at
        path endpoints.

        Returns
        -------
        Shapely
            A shapely Polygon or MultiPolygon representing the stroked path.
        """
        all_shapes = []

        # Add segment shapes
        for i in range(self.num_segments):
            seg_shape = self._segment_to_shapely(i)
            if seg_shape is not None and not seg_shape.is_empty:
                all_shapes.append(seg_shape)

        # Add corner shapes at junction vertices (vertices 1 to n-2)
        for i in range(1, len(self.vertices) - 1):
            corner_shape = self._generate_corner_shape(i)
            if corner_shape is not None and not corner_shape.is_empty:
                all_shapes.append(corner_shape)

        if not all_shapes:
            return shapely.Polygon()

        if len(all_shapes) == 1:
            return all_shapes[0]

        # Union all shapes (segments + corners)
        result = shapely.union_all(all_shapes)
        return result

    def _generate_corner_shape(self, vertex_index: int) -> shapely.Polygon:
        """Generate corner shape at a junction vertex.

        Parameters
        ----------
        vertex_index : int
            Index of the junction vertex (1 to num_vertices-2).

        Returns
        -------
        shapely.Polygon
            Circle (for round) or triangle (for bevel) at the junction.
        """
        vertex = np.array(self.vertices[vertex_index])
        half_width = self.width_at_vertex(vertex_index) / 2

        if self.corner_style == "round":
            # Full circle at the junction - simple and robust
            return shapely.Point(vertex).buffer(half_width, resolution=self.arc_resolution // 4)

        elif self.corner_style == "bevel":
            # Bevel: create triangle to fill the corner gap
            # Get tangent directions of incoming and outgoing segments
            prev_idx = vertex_index - 1
            next_idx = vertex_index

            # Compute tangents at the junction
            tan_in = self._get_segment_tangent(prev_idx, is_start=False)
            tan_out = self._get_segment_tangent(next_idx, is_start=True)
            tan_out = -tan_out  # Flip because _get_segment_tangent returns outward direction

            # Cross product to determine turn direction
            cross = tan_in[0] * tan_out[1] - tan_in[1] * tan_out[0]

            if abs(cross) < 1e-10:
                # Nearly straight, no corner fill needed
                return shapely.Polygon()

            # Perpendicular offsets
            perp_in = np.array([-tan_in[1], tan_in[0]])
            perp_out = np.array([-tan_out[1], tan_out[0]])

            if cross > 0:
                # Left turn - gap is on the right side
                p1 = vertex + perp_in * half_width  # end of incoming segment right edge
                p2 = vertex + perp_out * half_width  # start of outgoing segment right edge
            else:
                # Right turn - gap is on the left side
                p1 = vertex - perp_in * half_width  # end of incoming segment left edge
                p2 = vertex - perp_out * half_width  # start of outgoing segment left edge

            return shapely.Polygon([tuple(p1), tuple(vertex), tuple(p2)])

        return shapely.Polygon()

    def _segment_to_shapely(self, segment_index: int) -> shapely.Polygon:
        """Convert a single segment to a shapely polygon.

        Parameters
        ----------
        segment_index : int
            Index of the segment (0 to num_segments-1).

        Returns
        -------
        shapely.Polygon
            Polygon representing this segment with appropriate end caps.
        """
        start = np.array(self.vertices[segment_index])
        end = np.array(self.vertices[segment_index + 1])
        w_start = self.width_at_vertex(segment_index)
        w_end = self.width_at_vertex(segment_index + 1)

        # Get left/right offset points for this segment
        if segment_index in self.arcs:
            left, right = self._tessellate_arc_segment(
                start, end, self.arcs[segment_index], w_start, w_end
            )
        else:
            left, right = self._tessellate_line_segment(start, end, w_start, w_end)

        if not left or not right:
            return shapely.Polygon()

        # Determine end caps for this segment
        is_first = segment_index == 0
        is_last = segment_index == self.num_segments - 1

        # Build polygon outline
        outline = list(left)

        # End cap at segment end
        if is_last:
            # Use user-specified end cap
            end_cap = self._generate_segment_end_cap(
                np.array(left[-1]), np.array(right[-1]),
                segment_index, is_start_cap=False
            )
            outline.extend(end_cap)
        # else: flat end (just close directly)

        # Right edge reversed
        outline.extend(right[::-1])

        # Start cap at segment start
        if is_first:
            # Use user-specified end cap
            start_cap = self._generate_segment_end_cap(
                np.array(right[0]), np.array(left[0]),
                segment_index, is_start_cap=True
            )
            outline.extend(start_cap)
        # else: flat end (polygon auto-closes)

        return shapely.Polygon(outline)

    def _generate_segment_end_cap(
        self,
        p1: np.ndarray,
        p2: np.ndarray,
        segment_index: int,
        is_start_cap: bool,
    ) -> list:
        """Generate end cap for a segment endpoint.

        Parameters
        ----------
        p1 : np.ndarray
            Start point of cap.
        p2 : np.ndarray
            End point of cap.
        segment_index : int
            Index of the segment.
        is_start_cap : bool
            True for start cap (at segment start), False for end cap.

        Returns
        -------
        list
            Intermediate points for the cap.
        """
        if self.end_cap == "flat":
            return []

        center = (p1 + p2) / 2
        radius = np.linalg.norm(p2 - p1) / 2

        if radius < 1e-12:
            return []

        # Get outward direction for this segment endpoint
        outward = self._get_segment_tangent(segment_index, is_start_cap)

        if self.end_cap == "round":
            angle1 = math.atan2(p1[1] - center[1], p1[0] - center[0])
            angle2 = math.atan2(p2[1] - center[1], p2[0] - center[0])

            # Two possible sweeps
            sweep_ccw = angle2 - angle1
            if sweep_ccw <= 0:
                sweep_ccw += 2 * math.pi
            sweep_cw = sweep_ccw - 2 * math.pi

            # Choose sweep that bulges outward
            mid_ccw = np.array([math.cos(angle1 + sweep_ccw / 2), math.sin(angle1 + sweep_ccw / 2)])
            mid_cw = np.array([math.cos(angle1 + sweep_cw / 2), math.sin(angle1 + sweep_cw / 2)])

            sweep = sweep_ccw if np.dot(mid_ccw, outward) > np.dot(mid_cw, outward) else sweep_cw

            n_points = max(int(abs(sweep) * self.arc_resolution / (2 * math.pi)), 4)
            points = []
            for i in range(1, n_points):
                t = i / n_points
                angle = angle1 + t * sweep
                points.append((center[0] + radius * math.cos(angle),
                               center[1] + radius * math.sin(angle)))
            return points

        elif self.end_cap == "square":
            p1_ext = p1 + outward * radius
            p2_ext = p2 + outward * radius
            return [tuple(p1_ext), tuple(p2_ext)]

        return []

    def _get_segment_tangent(self, segment_index: int, is_start: bool) -> np.ndarray:
        """Get tangent direction at a segment endpoint (pointing outward).

        Parameters
        ----------
        segment_index : int
            Segment index.
        is_start : bool
            True for segment start, False for segment end.

        Returns
        -------
        np.ndarray
            Unit tangent pointing away from the segment.
        """
        p0 = np.array(self.vertices[segment_index])
        p1 = np.array(self.vertices[segment_index + 1])

        if segment_index in self.arcs:
            arc_info = self.arcs[segment_index]
            center = np.array(arc_info.center)
            point = p0 if is_start else p1
            radial = point - center
            radial_len = np.linalg.norm(radial)
            if radial_len < 1e-12:
                return np.array([-1, 0]) if is_start else np.array([1, 0])
            radial = radial / radial_len

            # Tangent perpendicular to radial
            if arc_info.clockwise:
                tangent = np.array([radial[1], -radial[0]])
            else:
                tangent = np.array([-radial[1], radial[0]])

            # Negate for start (pointing backward)
            return -tangent if is_start else tangent
        else:
            direction = p1 - p0
            length = np.linalg.norm(direction)
            if length < 1e-12:
                return np.array([-1, 0]) if is_start else np.array([1, 0])
            tangent = direction / length
            return -tangent if is_start else tangent

    def to_3d_geometry(
        self,
        slab_bounds: tuple[float, float],
        axis: Axis = 2,
        sidewall_angle: float = 0.0,
    ) -> Geometry:
        """Convert to 3D geometry by creating one PolySlab per segment plus corners.

        Each segment is extruded independently, plus corner shapes at junctions.
        This approach is more robust than trying to create a single unified
        polygon with complex corner handling.

        For PCB traces (PEC/metal), overlapping material at corners is
        perfectly fine - it's all the same conductor.

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
            A GeometryGroup containing PolySlabs for segments and corners.
        """
        from tidy3d.components.geometry.base import GeometryGroup
        from tidy3d.components.geometry.utils import from_shapely

        geometries = []

        # Add segment geometries
        for i in range(self.num_segments):
            seg_shape = self._segment_to_shapely(i)
            if seg_shape is not None and not seg_shape.is_empty and seg_shape.is_valid:
                geom = from_shapely(
                    shape=seg_shape,
                    axis=axis,
                    slab_bounds=slab_bounds,
                    sidewall_angle=sidewall_angle,
                )
                # from_shapely may return a single geometry or a group
                if isinstance(geom, GeometryGroup):
                    geometries.extend(geom.geometries)
                else:
                    geometries.append(geom)

        # Add corner geometries at junction vertices
        for i in range(1, len(self.vertices) - 1):
            corner_shape = self._generate_corner_shape(i)
            if corner_shape is not None and not corner_shape.is_empty and corner_shape.is_valid:
                geom = from_shapely(
                    shape=corner_shape,
                    axis=axis,
                    slab_bounds=slab_bounds,
                    sidewall_angle=sidewall_angle,
                )
                if isinstance(geom, GeometryGroup):
                    geometries.extend(geom.geometries)
                else:
                    geometries.append(geom)

        if len(geometries) == 1:
            return geometries[0]

        return GeometryGroup(geometries=tuple(geometries))

    @classmethod
    def from_line(
        cls,
        start: tuple[float, float],
        end: tuple[float, float],
        width: float,
        **kwargs,
    ) -> "Path2D":
        """Create a single straight segment path.

        Parameters
        ----------
        start : tuple[float, float]
            Start point.
        end : tuple[float, float]
            End point.
        width : float
            Path width.
        **kwargs
            Additional Path2D parameters.

        Returns
        -------
        Path2D
            A single-segment straight path.
        """
        return cls(vertices=(start, end), width=width, **kwargs)

    @classmethod
    def from_arc(
        cls,
        start: tuple[float, float],
        end: tuple[float, float],
        center: tuple[float, float],
        width: float,
        clockwise: bool = False,
        **kwargs,
    ) -> "Path2D":
        """Create a single arc segment path.

        Parameters
        ----------
        start : tuple[float, float]
            Start point of the arc.
        end : tuple[float, float]
            End point of the arc.
        center : tuple[float, float]
            Center of the circular arc.
        width : float
            Path width.
        clockwise : bool
            If True, arc goes clockwise. Default is False.
        **kwargs
            Additional Path2D parameters.

        Returns
        -------
        Path2D
            A single-segment arc path.
        """
        return cls(
            vertices=(start, end),
            arcs={0: ArcSegment(center=center, clockwise=clockwise)},
            width=width,
            **kwargs,
        )

    def _tessellate_line_segment(
        self,
        start: np.ndarray,
        end: np.ndarray,
        w_start: float,
        w_end: float,
    ) -> tuple[list, list]:
        """Generate offset points for a line segment.

        Parameters
        ----------
        start : np.ndarray
            Start point.
        end : np.ndarray
            End point.
        w_start : float
            Width at start.
        w_end : float
            Width at end.

        Returns
        -------
        tuple[list, list]
            Left and right offset point lists.
        """
        direction = end - start
        length = np.linalg.norm(direction)
        if length < 1e-12:
            return [], []

        tangent = direction / length
        normal = np.array([-tangent[1], tangent[0]])

        # For step interpolation, use constant width
        if self.width_interp == "step":
            w_end = w_start

        left = [
            tuple(start + normal * w_start / 2),
            tuple(end + normal * w_end / 2),
        ]
        right = [
            tuple(start - normal * w_start / 2),
            tuple(end - normal * w_end / 2),
        ]
        return left, right

    def _tessellate_arc_segment(
        self,
        start: np.ndarray,
        end: np.ndarray,
        arc_info: ArcSegment,
        w_start: float,
        w_end: float,
    ) -> tuple[list, list]:
        """Generate offset points for an arc segment.

        The left/right offset is computed based on the tangent direction at each
        point along the arc. "Left" is perpendicular to the tangent, rotated 90°
        CCW from the tangent direction.

        For CCW arcs: tangent points in direction of increasing angle, so left
        points toward center (inner) and right points away from center (outer).

        For CW arcs: tangent points in direction of decreasing angle, so left
        points away from center (outer) and right points toward center (inner).

        Parameters
        ----------
        start : np.ndarray
            Start point.
        end : np.ndarray
            End point.
        arc_info : ArcSegment
            Arc parameters.
        w_start : float
            Width at start.
        w_end : float
            Width at end.

        Returns
        -------
        tuple[list, list]
            Left and right offset point lists.
        """
        center = np.array(arc_info.center)
        clockwise = arc_info.clockwise

        radius = np.linalg.norm(start - center)
        if radius < 1e-12:
            return [], []

        start_angle = math.atan2(start[1] - center[1], start[0] - center[0])
        end_angle = math.atan2(end[1] - center[1], end[0] - center[0])

        # Calculate sweep angle
        sweep = end_angle - start_angle
        if clockwise:
            if sweep > 0:
                sweep -= 2 * math.pi
        else:
            if sweep < 0:
                sweep += 2 * math.pi

        # Handle full circle case (start == end)
        if np.allclose(start, end):
            sweep = -2 * math.pi if clockwise else 2 * math.pi

        # Number of samples based on arc angle
        n_samples = max(int(abs(sweep) * self.arc_resolution / (2 * math.pi)), 4)

        left_points = []
        right_points = []

        # Determine offset direction based on arc direction
        # For CCW: left is inner (-radial), right is outer (+radial)
        # For CW: left is outer (+radial), right is inner (-radial)
        left_sign = 1.0 if clockwise else -1.0
        right_sign = -1.0 if clockwise else 1.0

        for j in range(n_samples + 1):
            t = j / n_samples
            angle = start_angle + t * sweep

            # Width at this position
            if self.width_interp == "step":
                width = w_start
            else:
                width = w_start + t * (w_end - w_start)
            half_w = width / 2

            # Point on centerline
            cx = center[0] + radius * math.cos(angle)
            cy = center[1] + radius * math.sin(angle)

            # Radial direction (outward from center)
            nx, ny = math.cos(angle), math.sin(angle)

            # Offset based on arc direction
            left_points.append((cx + left_sign * half_w * nx, cy + left_sign * half_w * ny))
            right_points.append((cx + right_sign * half_w * nx, cy + right_sign * half_w * ny))

        return left_points, right_points

