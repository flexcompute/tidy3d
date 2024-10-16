"""Geometry extruded from polygonal shapes."""

from __future__ import annotations

from copy import copy
from math import isclose
from typing import List, Tuple

import autograd.numpy as np
import pydantic.v1 as pydantic
import shapely
from autograd.tracer import isbox
from matplotlib import path

from ...constants import LARGE_NUMBER, MICROMETER, fp_eps
from ...exceptions import SetupError, ValidationError
from ...log import log
from ...packaging import verify_packages_import
from ..autograd import AutogradFieldMap, TracedVertices, get_static
from ..autograd.derivative_utils import DerivativeInfo
from ..base import cached_property, skip_if_fields_missing
from ..types import (
    ArrayFloat2D,
    ArrayLike,
    Axis,
    Bound,
    Coordinate,
    MatrixReal4x4,
    PlanePosition,
    Shapely,
)
from . import base, triangulation

# sampling polygon along dilation for validating polygon to be
# non self-intersecting during the entire dilation process
_N_SAMPLE_POLYGON_INTERSECT = 5

_IS_CLOSE_RTOL = np.finfo(float).eps

# Warn for too many divided polyslabs
_COMPLEX_POLYSLAB_DIVISIONS_WARN = 100

# Warn before triangulating large polyslabs due to inefficiency
_MAX_POLYSLAB_VERTICES_FOR_TRIANGULATION = 500


class PolySlab(base.Planar):
    """Polygon extruded with optional sidewall angle along axis direction.

    Example
    -------
    >>> vertices = np.array([(0,0), (1,0), (1,1)])
    >>> p = PolySlab(vertices=vertices, axis=2, slab_bounds=(-1, 1))
    """

    slab_bounds: Tuple[float, float] = pydantic.Field(
        ...,
        title="Slab Bounds",
        description="Minimum and maximum positions of the slab along axis dimension.",
        units=MICROMETER,
    )

    dilation: float = pydantic.Field(
        0.0,
        title="Dilation",
        description="Dilation of the supplied polygon by shifting each edge along its "
        "normal outwards direction by a distance; a negative value corresponds to erosion.",
        units=MICROMETER,
    )

    vertices: TracedVertices = pydantic.Field(
        ...,
        title="Vertices",
        description="List of (d1, d2) defining the 2 dimensional positions of the polygon "
        "face vertices at the ``reference_plane``. "
        "The index of dimension should be in the ascending order: e.g. if "
        "the slab normal axis is ``axis=y``, the coordinate of the vertices will be in (x, z)",
        units=MICROMETER,
    )

    @staticmethod
    def make_shapely_polygon(vertices: ArrayLike) -> shapely.Polygon:
        """Make a shapely polygon from some vertices, first ensures they are untraced."""
        vertices = get_static(vertices)
        return shapely.Polygon(vertices)

    @pydantic.validator("slab_bounds", always=True)
    def slab_bounds_order(cls, val):
        """Maximum position of the slab should be no smaller than its minimal position."""
        if val[1] < val[0]:
            raise SetupError(
                "Polyslab.slab_bounds must be specified in the order of "
                "minimum and maximum positions of the slab along the axis. "
                f"But now the maximum {val[1]} is smaller than the minimum {val[0]}."
            )
        return val

    @pydantic.validator("vertices", always=True)
    def correct_shape(cls, val):
        """Makes sure vertices size is correct.
        Make sure no intersecting edges.
        """
        # overall shape of vertices
        if val.shape[1] != 2:
            raise SetupError(
                "PolySlab.vertices must be a 2 dimensional array shaped (N, 2). "
                f"Given array with shape of {val.shape}."
            )

        # make sure no polygon splitting, islands, 0 area
        poly_heal = shapely.make_valid(cls.make_shapely_polygon(val))
        if poly_heal.area < fp_eps:
            raise SetupError("The polygon almost collapses to a 1D curve.")

        if not poly_heal.geom_type == "Polygon" or len(poly_heal.interiors) > 0:
            raise SetupError(
                "Polygon is self-intersecting, resulting in "
                "polygon splitting or generation of holes/islands. "
                "A general treatment to self-intersecting polygon will be available "
                "in future releases."
            )
        return val

    @pydantic.validator("vertices", always=True)
    @skip_if_fields_missing(["dilation"])
    def no_complex_self_intersecting_polygon_at_reference_plane(cls, val, values):
        """At the reference plane, check if the polygon is self-intersecting.

        There are two types of self-intersection that can occur during dilation:
        1) the one that creates holes/islands, or splits polygons, or removes everything;
        2) the one that does not.

        For 1), we issue an error since it is yet to be supported;
        For 2), we heal the polygon, and warn that the polygon has been cleaned up.
        """
        # no need to validate anything here
        if isclose(values["dilation"], 0):
            return val

        val_np = PolySlab._proper_vertices(val)
        dist = values["dilation"]

        # 0) fully eroded
        if dist < 0 and dist < -PolySlab._maximal_erosion(val_np):
            raise SetupError("Erosion value is too large. The polygon is fully eroded.")

        # no edge events
        if not PolySlab._edge_events_detection(val_np, dist, ignore_at_dist=False):
            return val

        poly_offset = PolySlab._shift_vertices(val_np, dist)[0]
        if PolySlab._area(poly_offset) < fp_eps**2:
            raise SetupError("Erosion value is too large. The polygon is fully eroded.")

        # edge events
        poly_offset = shapely.make_valid(cls.make_shapely_polygon(poly_offset))
        # 1) polygon split or create holes/islands
        if not poly_offset.geom_type == "Polygon" or len(poly_offset.interiors) > 0:
            raise SetupError(
                "Dilation/Erosion value is too large, resulting in "
                "polygon splitting or generation of holes/islands. "
                "A general treatment to self-intersecting polygon will be available "
                "in future releases."
            )

        # case 2
        log.warning(
            "The dilation/erosion value is too large. resulting in a "
            "self-intersecting polygon. "
            "The vertices have been modified to make a valid polygon."
        )
        return val

    @pydantic.validator("vertices", always=True)
    @skip_if_fields_missing(["sidewall_angle", "dilation", "slab_bounds", "reference_plane"])
    def no_self_intersecting_polygon_during_extrusion(cls, val, values):
        """In this simple polyslab, we don't support self-intersecting polygons yet, meaning that
        any normal cross section of the PolySlab cannot be self-intersecting. This part checks
        if any self-interction will occur during extrusion with non-zero sidewall angle.

        There are two types of self-intersection, known as edge events,
        that can occur during dilation:
        1) neighboring vertex-vertex crossing. This type of edge event can be treated with
        ``ComplexPolySlab`` which divides the polyslab into a list of simple polyslabs.

        2) other types of edge events that can create holes/islands or split polygons.
        To detect this, we sample _N_SAMPLE_POLYGON_INTERSECT cross sections to see if any creation
        of polygons/holes, and changes in vertices number.
        """

        # no need to validate anything here
        if isclose(values["sidewall_angle"], 0):
            return val

        # apply dilation
        poly_ref = PolySlab._proper_vertices(val)
        if not isclose(values["dilation"], 0):
            poly_ref = PolySlab._shift_vertices(poly_ref, values["dilation"])[0]
            poly_ref = PolySlab._heal_polygon(poly_ref)

        # Fist, check vertex-vertex crossing at any point during extrusion
        length = values["slab_bounds"][1] - values["slab_bounds"][0]
        dist = [-length * np.tan(values["sidewall_angle"])]
        # reverse the dilation value if it's defined on the top
        if values["reference_plane"] == "top":
            dist = [-dist[0]]
        # for middle, both direction needs to be examined
        elif values["reference_plane"] == "middle":
            dist = [dist[0] / 2, -dist[0] / 2]

        # capture vertex crossing events
        max_thick = []
        for dist_val in dist:
            max_dist = PolySlab._neighbor_vertices_crossing_detection(poly_ref, dist_val)

            if max_dist is not None:
                max_thick.append(max_dist / abs(dist_val) * length)

        if len(max_thick) > 0:
            max_thick = min(max_thick)
            raise SetupError(
                "Sidewall angle or structure thickness is so large that the polygon "
                "is self-intersecting during extrusion. "
                f"Please either reduce structure thickness to be < {max_thick:.3e}, "
                "or use our plugin 'ComplexPolySlab' to divide the complex polyslab "
                "into a list of simple polyslabs."
            )

        # vertex-edge crossing event.
        for dist_val in dist:
            if PolySlab._edge_events_detection(poly_ref, dist_val):
                raise SetupError(
                    "Sidewall angle or structure thickness is too large, "
                    "resulting in polygon splitting or generation of holes/islands. "
                    "A general treatment to self-intersecting polygon will be available "
                    "in future releases."
                )
        return val

    @classmethod
    def from_gds(
        cls,
        gds_cell,
        axis: Axis,
        slab_bounds: Tuple[float, float],
        gds_layer: int,
        gds_dtype: int = None,
        gds_scale: pydantic.PositiveFloat = 1.0,
        dilation: float = 0.0,
        sidewall_angle: float = 0,
        reference_plane: PlanePosition = "middle",
    ) -> List[PolySlab]:
        """Import :class:`PolySlab` from a ``gdstk.Cell`` or a ``gdspy.Cell``.

        Parameters
        ----------
        gds_cell : Union[gdstk.Cell, gdspy.Cell]
            ``gdstk.Cell`` or ``gdspy.Cell`` containing 2D geometric data.
        axis : int
            Integer index into the polygon's slab axis. (0,1,2) -> (x,y,z).
        slab_bounds: Tuple[float, float]
            Minimum and maximum positions of the slab along ``axis``.
        gds_layer : int
            Layer index in the ``gds_cell``.
        gds_dtype : int = None
            Data-type index in the ``gds_cell``.
            If ``None``, imports all data for this layer into the returned list.
        gds_scale : float = 1.0
            Length scale used in GDS file in units of MICROMETER.
            For example, if gds file uses nanometers, set ``gds_scale=1e-3``.
            Must be positive.
        dilation : float = 0.0
            Dilation of the polygon in the base by shifting each edge along its
            normal outwards direction by a distance;
            a negative value corresponds to erosion.
        sidewall_angle : float = 0
            Angle of the sidewall.
            ``sidewall_angle=0`` (default) specifies vertical wall,
            while ``0<sidewall_angle<np.pi/2`` for the base to be larger than the top.
        reference_plane : PlanePosition = "middle"
            The position of the GDS layer. It can be at the ``bottom``, ``middle``,
            or ``top`` of the PolySlab. E.g. if ``axis=1``, ``bottom`` refers to the
            negative side of y-axis, and ``top`` refers to the positive side of y-axis.

        Returns
        -------
        List[:class:`PolySlab`]
            List of :class:`PolySlab` objects sharing ``axis`` and  slab bound properties.
        """

        all_vertices = PolySlab._load_gds_vertices(gds_cell, gds_layer, gds_dtype, gds_scale)

        return [
            cls(
                vertices=verts,
                axis=axis,
                slab_bounds=slab_bounds,
                dilation=dilation,
                sidewall_angle=sidewall_angle,
                reference_plane=reference_plane,
            )
            for verts in all_vertices
        ]

    @staticmethod
    def _load_gds_vertices(
        gds_cell,
        gds_layer: int,
        gds_dtype: int = None,
        gds_scale: pydantic.PositiveFloat = 1.0,
    ) -> List[ArrayFloat2D]:
        """Import :class:`PolySlab` from a ``gdstk.Cell`` or a ``gdspy.Cell``.

        Parameters
        ----------
        gds_cell : Union[gdstk.Cell, gdspy.Cell]
            ``gdstk.Cell`` or ``gdspy.Cell`` containing 2D geometric data.
        gds_layer : int
            Layer index in the ``gds_cell``.
        gds_dtype : int = None
            Data-type index in the ``gds_cell``.
            If ``None``, imports all data for this layer into the returned list.
        gds_scale : float = 1.0
            Length scale used in GDS file in units of MICROMETER.
            For example, if gds file uses nanometers, set ``gds_scale=1e-3``.
            Must be positive.

        Returns
        -------
        List[ArrayFloat2D]
            List of :class:`.ArrayFloat2D`
        """

        # switch the GDS cell loader function based on the class name string
        # TODO: make this more robust in future releases
        gds_cell_class_name = str(gds_cell.__class__)

        if "gdstk" in gds_cell_class_name:
            gds_loader_fn = base.Geometry.load_gds_vertices_gdstk

        elif "gdspy" in gds_cell_class_name:
            gds_loader_fn = base.Geometry.load_gds_vertices_gdspy

        else:
            raise ValueError(
                f"validate 'gds_cell' of type '{gds_cell_class_name}' "
                "does not seem to be associated with 'gdstk' or 'gdspy' packages "
                "and therefore can't be loaded by Tidy3D."
            )

        all_vertices = gds_loader_fn(
            gds_cell=gds_cell, gds_layer=gds_layer, gds_dtype=gds_dtype, gds_scale=gds_scale
        )

        # convert vertices into polyslabs
        polygons = [PolySlab.make_shapely_polygon(vertices).buffer(0) for vertices in all_vertices]
        polys_union = shapely.unary_union(polygons, grid_size=base.POLY_GRID_SIZE)

        if polys_union.geom_type == "Polygon":
            all_vertices = [np.array(polys_union.exterior.coords)]
        elif polys_union.geom_type == "MultiPolygon":
            all_vertices = [np.array(polygon.exterior.coords) for polygon in polys_union.geoms]
        return all_vertices

    @property
    def center_axis(self) -> float:
        """Gets the position of the center of the geometry in the out of plane dimension."""
        zmin, zmax = self.slab_bounds
        if np.isneginf(zmin) and np.isposinf(zmax):
            return 0.0
        zmin = max(zmin, -LARGE_NUMBER)
        zmax = min(zmax, LARGE_NUMBER)
        return (zmax + zmin) / 2.0

    @property
    def length_axis(self) -> float:
        """Gets the length of the geometry along the out of plane dimension."""
        zmin, zmax = self.slab_bounds
        return zmax - zmin

    @cached_property
    def reference_polygon(self) -> np.ndarray:
        """The polygon at the reference plane.

        Returns
        -------
        ArrayLike[float, float]
            The vertices of the polygon at the reference plane.
        """
        vertices = self._proper_vertices(self.vertices)
        if isclose(self.dilation, 0):
            return vertices
        offset_vertices = self._shift_vertices(vertices, self.dilation)[0]
        return self._heal_polygon(offset_vertices)

    @cached_property
    def middle_polygon(self) -> np.ndarray:
        """The polygon at the middle.

        Returns
        -------
        ArrayLike[float, float]
            The vertices of the polygon at the middle.
        """

        dist = self._extrusion_length_to_offset_distance(self.finite_length_axis / 2)
        if self.reference_plane == "bottom":
            return self._shift_vertices(self.reference_polygon, dist)[0]
        if self.reference_plane == "top":
            return self._shift_vertices(self.reference_polygon, -dist)[0]
        # middle case
        return self.reference_polygon

    @cached_property
    def base_polygon(self) -> np.ndarray:
        """The polygon at the base, derived from the ``middle_polygon``.

        Returns
        -------
        ArrayLike[float, float]
            The vertices of the polygon at the base.
        """
        if self.reference_plane == "bottom":
            return self.reference_polygon
        dist = self._extrusion_length_to_offset_distance(-self.finite_length_axis / 2)
        return self._shift_vertices(self.middle_polygon, dist)[0]

    @cached_property
    def top_polygon(self) -> np.ndarray:
        """The polygon at the top, derived from the ``middle_polygon``.

        Returns
        -------
        ArrayLike[float, float]
            The vertices of the polygon at the top.
        """
        if self.reference_plane == "top":
            return self.reference_polygon
        dist = self._extrusion_length_to_offset_distance(self.finite_length_axis / 2)
        return self._shift_vertices(self.middle_polygon, dist)[0]

    @cached_property
    def _normal_2dmaterial(self) -> Axis:
        """Get the normal to the given geometry, checking that it is a 2D geometry."""
        if self.slab_bounds[0] != self.slab_bounds[1]:
            raise ValidationError("'Medium2D' requires the 'PolySlab' bounds to be equal.")
        return self.axis

    def _update_from_bounds(self, bounds: Tuple[float, float], axis: Axis) -> PolySlab:
        """Returns an updated geometry which has been transformed to fit within ``bounds``
        along the ``axis`` direction."""
        if axis != self.axis:
            raise ValueError(
                f"'_update_from_bounds' may only be applied along axis '{self.axis}', "
                f"but was given axis '{axis}'."
            )
        return self.updated_copy(slab_bounds=bounds)

    @cached_property
    def is_ccw(self) -> bool:
        """Is this ``PolySlab`` CCW-oriented?"""
        return PolySlab._area(self.vertices) > 0

    def inside(
        self, x: np.ndarray[float], y: np.ndarray[float], z: np.ndarray[float]
    ) -> np.ndarray[bool]:
        """For input arrays ``x``, ``y``, ``z`` of arbitrary but identical shape, return an array
        with the same shape which is ``True`` for every point in zip(x, y, z) that is inside the
        volume of the :class:`Geometry`, and ``False`` otherwise.

        Parameters
        ----------
        x : np.ndarray[float]
            Array of point positions in x direction.
        y : np.ndarray[float]
            Array of point positions in y direction.
        z : np.ndarray[float]
            Array of point positions in z direction.

        Returns
        -------
        np.ndarray[bool]
            ``True`` for every point that is inside the geometry.
        """
        self._ensure_equal_shape(x, y, z)

        z, (x, y) = self.pop_axis((x, y, z), axis=self.axis)

        z0 = self.center_axis
        dist_z = np.abs(z - z0)
        inside_height = dist_z <= (self.finite_length_axis / 2)

        # avoid going into face checking if no points are inside slab bounds
        if not np.any(inside_height):
            return inside_height

        # check what points are inside polygon cross section (face)
        z_local = z - z0  # distance to the middle
        dist = -z_local * self._tanq

        # # Leaving this function and commented out lines using it below in case we want to revert
        # # to it at some point, e.g. if we introduce a MATPLOTLIB_INSTALLED flag.
        # def contains_pointwise(face_polygon):
        #     def fun_contain(xy_point):
        #         point = shapely.Point(xy_point)
        #         return face_polygon.covers(point)
        #     return fun_contain

        if isinstance(x, np.ndarray):
            inside_polygon = np.zeros_like(inside_height)
            xs_slab = x[inside_height]
            ys_slab = y[inside_height]

            # vertical sidewall
            if isclose(self.sidewall_angle, 0):
                # face_polygon = self.make_shapely_polygon(self.reference_polygon)
                # fun_contain = contains_pointwise(face_polygon)
                # contains_vectorized = np.vectorize(fun_contain, signature="(n)->()")
                poly_path = path.Path(self.reference_polygon)
                contains_vectorized = poly_path.contains_points
                points_stacked = np.stack((xs_slab, ys_slab), axis=1)
                inside_polygon_slab = contains_vectorized(points_stacked)
                inside_polygon[inside_height] = inside_polygon_slab
            # slanted sidewall, offsetting vertices at each z
            else:
                # a helper function for moving axis
                def _move_axis(arr):
                    return np.moveaxis(arr, source=self.axis, destination=-1)

                def _move_axis_reverse(arr):
                    return np.moveaxis(arr, source=-1, destination=self.axis)

                inside_polygon_axis = _move_axis(inside_polygon)
                x_axis = _move_axis(x)
                y_axis = _move_axis(y)

                for z_i in range(z.shape[self.axis]):
                    if not _move_axis(inside_height)[0, 0, z_i]:
                        continue
                    vertices_z = self._shift_vertices(
                        self.middle_polygon, _move_axis(dist)[0, 0, z_i]
                    )[0]
                    # face_polygon = self.make_shapely_polygon(vertices_z)
                    # fun_contain = contains_pointwise(face_polygon)
                    # contains_vectorized = np.vectorize(fun_contain, signature="(n)->()")
                    poly_path = path.Path(vertices_z)
                    contains_vectorized = poly_path.contains_points
                    points_stacked = np.stack(
                        (x_axis[:, :, 0].flatten(), y_axis[:, :, 0].flatten()), axis=1
                    )
                    inside_polygon_slab = contains_vectorized(points_stacked)
                    inside_polygon_axis[:, :, z_i] = inside_polygon_slab.reshape(x_axis.shape[:2])
                inside_polygon = _move_axis_reverse(inside_polygon_axis)
        else:
            vertices_z = self._shift_vertices(self.middle_polygon, dist)[0]
            face_polygon = self.make_shapely_polygon(vertices_z)
            point = shapely.Point(x, y)
            inside_polygon = face_polygon.covers(point)
        return inside_height * inside_polygon

    @verify_packages_import(["trimesh"])
    def _do_intersections_tilted_plane(
        self, normal: Coordinate, origin: Coordinate, to_2D: MatrixReal4x4
    ) -> List[Shapely]:
        """Return a list of shapely geometries at the plane specified by normal and origin.

        Parameters
        ----------
        normal : Coordinate
            Vector defining the normal direction to the plane.
        origin : Coordinate
            Vector defining the plane origin.
        to_2D : MatrixReal4x4
            Transformation matrix to apply to resulting shapes.

        Returns
        -------
        List[shapely.geometry.base.BaseGeometry]
            List of 2D shapes that intersect plane.
            For more details refer to
            `Shapely's Documentation <https://shapely.readthedocs.io/en/stable/project.html>`_.
        """
        import trimesh

        if len(self.base_polygon) > _MAX_POLYSLAB_VERTICES_FOR_TRIANGULATION:
            log.warning(
                "Processing of PolySlabs with large numbers of vertices can be slow.", log_once=True
            )
        base_triangles = triangulation.triangulate(self.base_polygon)
        top_triangles = (
            base_triangles
            if isclose(self.sidewall_angle, 0)
            else triangulation.triangulate(self.top_polygon)
        )

        n = len(self.base_polygon)
        faces = (
            [[a, b, c] for c, b, a in base_triangles]
            + [[n + a, n + b, n + c] for a, b, c in top_triangles]
            + [(i, (i + 1) % n, n + i) for i in range(n)]
            + [((i + 1) % n, n + ((i + 1) % n), n + i) for i in range(n)]
        )

        x = np.hstack((self.base_polygon[:, 0], self.top_polygon[:, 0]))
        y = np.hstack((self.base_polygon[:, 1], self.top_polygon[:, 1]))
        z = np.hstack((np.full(n, self.slab_bounds[0]), np.full(n, self.slab_bounds[1])))
        vertices = np.vstack(self.unpop_axis(z, (x, y), self.axis)).T
        mesh = trimesh.Trimesh(vertices, faces)

        section = mesh.section(plane_origin=origin, plane_normal=normal)
        if section is None:
            return []
        path, _ = section.to_planar(to_2D=to_2D)
        return path.polygons_full

    def _intersections_normal(self, z: float):
        """Find shapely geometries intersecting planar geometry with axis normal to slab.

        Parameters
        ----------
        z : float
            Position along the axis normal to slab.

        Returns
        -------
        List[shapely.geometry.base.BaseGeometry]
            List of 2D shapes that intersect plane.
            For more details refer to
            `Shapely's Documentation <https://shapely.readthedocs.io/en/stable/project.html>`_.
        """
        if isclose(self.sidewall_angle, 0):
            return [self.make_shapely_polygon(self.reference_polygon)]

        z0 = self.center_axis
        z_local = z - z0  # distance to the middle
        dist = -z_local * self._tanq
        vertices_z = self._shift_vertices(self.middle_polygon, dist)[0]
        return [self.make_shapely_polygon(vertices_z)]

    def _intersections_side(self, position, axis) -> list:
        """Find shapely geometries intersecting planar geometry with axis orthogonal to slab.

        For slanted polyslab, the procedure is as follows,
        1) Find out all z-coordinates where the plane will intersect directly with a vertex.
        Denote the coordinates as (z_0, z_1, z_2, ... )
        2) Find out all polygons that can be formed between z_i and z_{i+1}. There are two
        types of polygons:
            a) formed by the plane intersecting the edges
            b) formed by the plane intersecting the vertices.
            For either type, one needs to compute:
                i) intersecting position
                ii) angle between the plane and the intersecting edge
            For a), both are straightforward to compute; while for b), one needs to compute
            which edge the plane will slide into.
        3) Looping through z_i, and merge all polygons. The partition by z_i is because once
        the plane intersects the vertex, it can intersect with other edges during
        the extrusion.

        Parameters
        ----------
        position : float
            Position along ``axis``.
        axis : int
            Integer index into 'xyz' (0,1,2).

        Returns
        -------
        List[shapely.geometry.base.BaseGeometry]
            List of 2D shapes that intersect plane.
            For more details refer to
            `Shapely's Documentation <https://shapely.readthedocs.io/en/stable/project.html>`_.
        """

        # find out all z_i where the plane will intersect the vertex
        z0 = self.center_axis
        z_base = z0 - self.finite_length_axis / 2

        axis_ordered = self._order_axis(axis)
        height_list = self._find_intersecting_height(position, axis_ordered)
        polys = []

        # looping through z_i to assemble the polygons
        height_list = np.append(height_list, self.finite_length_axis)
        h_base = 0.0
        for h_top in height_list:
            # length within between top and bottom
            h_length = h_top - h_base

            # coordinate of each subsection
            z_min = z_base + h_base
            z_max = np.inf if np.isposinf(h_top) else z_base + h_top

            # for vertical sidewall, no need for complications
            if isclose(self.sidewall_angle, 0):
                ints_y, ints_angle = self._find_intersecting_ys_angle_vertical(
                    self.reference_polygon, position, axis_ordered
                )
            else:
                # for slanted sidewall, move up by `fp_eps` in case vertices are degenerate at the base.
                dist = -(h_base - self.finite_length_axis / 2 + fp_eps) * self._tanq
                vertices = self._shift_vertices(self.middle_polygon, dist)[0]
                ints_y, ints_angle = self._find_intersecting_ys_angle_slant(
                    vertices, position, axis_ordered
                )

            # make polygon with intersections and z axis information
            for y_index in range(len(ints_y) // 2):
                y_min = ints_y[2 * y_index]
                y_max = ints_y[2 * y_index + 1]
                minx, miny = self._order_by_axis(plane_val=y_min, axis_val=z_min, axis=axis)
                maxx, maxy = self._order_by_axis(plane_val=y_max, axis_val=z_max, axis=axis)

                if isclose(self.sidewall_angle, 0):
                    polys.append(self.make_shapely_box(minx, miny, maxx, maxy))
                else:
                    angle_min = ints_angle[2 * y_index]
                    angle_max = ints_angle[2 * y_index + 1]

                    angle_min = np.arctan(np.tan(self.sidewall_angle) / np.sin(angle_min))
                    angle_max = np.arctan(np.tan(self.sidewall_angle) / np.sin(angle_max))

                    dy_min = h_length * np.tan(angle_min)
                    dy_max = h_length * np.tan(angle_max)

                    x1, y1 = self._order_by_axis(plane_val=y_min, axis_val=z_min, axis=axis)
                    x2, y2 = self._order_by_axis(plane_val=y_max, axis_val=z_min, axis=axis)
                    x3, y3 = self._order_by_axis(
                        plane_val=y_max - dy_max, axis_val=z_max, axis=axis
                    )
                    x4, y4 = self._order_by_axis(
                        plane_val=y_min + dy_min, axis_val=z_max, axis=axis
                    )
                    vertices = ((x1, y1), (x2, y2), (x3, y3), (x4, y4))
                    polys.append(self.make_shapely_polygon(vertices).buffer(0))
            # update the base coordinate for the next subsection
            h_base = h_top

        # merge touching polygons
        polys_union = shapely.unary_union(polys, grid_size=base.POLY_GRID_SIZE)
        if polys_union.geom_type == "Polygon":
            return [polys_union]
        if polys_union.geom_type == "MultiPolygon":
            return polys_union.geoms
        # in other cases, just return the original unmerged polygons
        return polys

    def _find_intersecting_height(self, position: float, axis: int) -> np.ndarray:
        """Found a list of height where the plane will intersect with the vertices;
        For vertical sidewall, just return np.array([]).
        Assumes axis is handles so this function works on xy plane.

        Parameters
        ----------
        position : float
            position along axis.
        axis : int
            Integer index into 'xyz' (0,1,2).

        Returns
        np.ndarray
            Height (relative to the base) where the plane will intersect with vertices.
        """
        if isclose(self.sidewall_angle, 0):
            return np.array([])

        # shift rate
        dist = 1.0
        shift_x, shift_y = PolySlab._shift_vertices(self.middle_polygon, dist)[2]
        shift_val = shift_x if axis == 0 else shift_y
        shift_val[np.isclose(shift_val, 0, rtol=_IS_CLOSE_RTOL)] = np.inf  # for static vertices

        # distance to the plane in the direction of vertex shifting
        distance = self.middle_polygon[:, axis] - position
        height = distance / self._tanq / shift_val + self.finite_length_axis / 2
        height = np.unique(height)
        # further filter very close ones
        is_not_too_close = np.insert((np.diff(height) > fp_eps), 0, True)
        height = height[is_not_too_close]

        height = height[height > fp_eps]
        height = height[height < self.finite_length_axis - fp_eps]
        return height

    def _find_intersecting_ys_angle_vertical(
        self, vertices: np.ndarray, position: float, axis: int, exclude_on_vertices: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Finds pairs of forward and backwards vertices where polygon intersects position at axis,
        Find intersection point (in y) assuming straight line,and intersecting angle between plane
        and edges. (For unslanted polyslab).
           Assumes axis is handles so this function works on xy plane.

        Parameters
        ----------
        vertices : np.ndarray
            Shape (N, 2) defining the polygon vertices in the xy-plane.
        position : float
            position along axis.
        axis : int
            Integer index into 'xyz' (0,1,2).
        exclude_on_vertices : bool = False
            Whether to exclude those intersecting directly with the vertices.

        Returns
        -------
        Union[np.ndarray, np.ndarray]
            List of intersection points along y direction.
            List of angles between plane and edges.
        """

        vertices_axis = vertices

        # flip vertices x,y for axis = y
        if axis == 1:
            vertices_axis = np.roll(vertices_axis, shift=1, axis=1)

        # get the forward vertices
        vertices_f = np.roll(vertices_axis, shift=-1, axis=0)

        # x coordinate of the two sets of vertices
        x_vertices_f, _ = vertices_f.T
        x_vertices_axis, _ = vertices_axis.T

        # find which segments intersect
        f_left_to_intersect = x_vertices_f <= position
        orig_right_to_intersect = x_vertices_axis > position
        intersects_b = np.logical_and(f_left_to_intersect, orig_right_to_intersect)

        f_right_to_intersect = x_vertices_f > position
        orig_left_to_intersect = x_vertices_axis <= position
        intersects_f = np.logical_and(f_right_to_intersect, orig_left_to_intersect)

        # exclude vertices at the position if exclude_on_vertices is True
        if exclude_on_vertices:
            intersects_on = np.isclose(x_vertices_axis, position, rtol=_IS_CLOSE_RTOL)
            intersects_f_on = np.isclose(x_vertices_f, position, rtol=_IS_CLOSE_RTOL)
            intersects_both_off = np.logical_not(np.logical_or(intersects_on, intersects_f_on))
            intersects_f &= intersects_both_off
            intersects_b &= intersects_both_off
        intersects_segment = np.logical_or(intersects_b, intersects_f)

        iverts_b = vertices_axis[intersects_segment]
        iverts_f = vertices_f[intersects_segment]

        # intersecting positions and angles
        ints_y = []
        ints_angle = []
        for vertices_f_local, vertices_b_local in zip(iverts_b, iverts_f):
            x1, y1 = vertices_f_local
            x2, y2 = vertices_b_local
            slope = (y2 - y1) / (x2 - x1)
            y = y1 + slope * (position - x1)
            ints_y.append(y)
            ints_angle.append(np.pi / 2 - np.arctan(np.abs(slope)))

        ints_y = np.array(ints_y)
        ints_angle = np.array(ints_angle)

        sort_index = np.argsort(ints_y)
        ints_y_sort = ints_y[sort_index]
        ints_angle_sort = ints_angle[sort_index]

        return ints_y_sort, ints_angle_sort

    def _find_intersecting_ys_angle_slant(
        self, vertices: np.ndarray, position: float, axis: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Finds pairs of forward and backwards vertices where polygon intersects position at axis,
        Find intersection point (in y) assuming straight line,and intersecting angle between plane
        and edges. (For slanted polyslab)
           Assumes axis is handles so this function works on xy plane.

        Parameters
        ----------
        vertices : np.ndarray
            Shape (N, 2) defining the polygon vertices in the xy-plane.
        position : float
            position along axis.
        axis : int
            Integer index into 'xyz' (0,1,2).

        Returns
        -------
        Union[np.ndarray, np.ndarray]
            List of intersection points along y direction.
            List of angles between plane and edges.
        """

        vertices_axis = vertices.copy()
        # flip vertices x,y for axis = y
        if axis == 1:
            vertices_axis = np.roll(vertices_axis, shift=1, axis=1)

        # get the forward vertices
        vertices_f = np.roll(vertices_axis, shift=-1, axis=0)
        # get the backward vertices
        vertices_b = np.roll(vertices_axis, shift=1, axis=0)

        ## First part, plane intersects with edges, same as vertical
        ints_y, ints_angle = self._find_intersecting_ys_angle_vertical(
            vertices, position, axis, exclude_on_vertices=True
        )
        ints_y = ints_y.tolist()
        ints_angle = ints_angle.tolist()

        ## Second part, plane intersects directly with vertices
        # vertices on the intersection
        intersects_on = np.isclose(vertices_axis[:, 0], position, rtol=_IS_CLOSE_RTOL)
        iverts_on = vertices_axis[intersects_on]
        # position of the neighbouring vertices
        iverts_b = vertices_b[intersects_on]
        iverts_f = vertices_f[intersects_on]
        # shift rate
        dist = -np.sign(self.sidewall_angle)
        shift_x, shift_y = self._shift_vertices(self.middle_polygon, dist)[2]
        shift_val = shift_x if axis == 0 else shift_y
        shift_val = shift_val[intersects_on]

        for vertices_f_local, vertices_b_local, vertices_on_local, shift_local in zip(
            iverts_f, iverts_b, iverts_on, shift_val
        ):
            x_on, y_on = vertices_on_local
            x_f, y_f = vertices_f_local
            x_b, y_b = vertices_b_local

            num_added = 0  # keep track the number of added vertices
            slope = []  # list of slopes for added vertices
            # case 1, shifting velocity is 0
            if np.isclose(shift_local, 0, rtol=_IS_CLOSE_RTOL):
                ints_y.append(y_on)
                # Slope w.r.t. forward and backward should equal,
                # just pick one of them.
                slope.append((y_on - y_b) / (x_on - x_b))
                ints_angle.append(np.pi / 2 - np.arctan(np.abs(slope[0])))
                continue

            # case 2, shifting towards backward direction
            if (x_b - position) * shift_local < 0:
                ints_y.append(y_on)
                slope.append((y_on - y_b) / (x_on - x_b))
                num_added += 1

            # case 3, shifting towards forward direction
            if (x_f - position) * shift_local < 0:
                ints_y.append(y_on)
                slope.append((y_on - y_f) / (x_on - x_f))
                num_added += 1

            # in case 2, and case 3, if just num_added = 1
            if num_added == 1:
                ints_angle.append(np.pi / 2 - np.arctan(np.abs(slope[0])))
            # if num_added = 2, the order of the two new vertices needs to handled correctly;
            # it should be sorted according to the -slope * moving direction
            elif num_added == 2:
                dressed_slope = [-s_i * shift_local for s_i in slope]
                sort_index = np.argsort(np.array(dressed_slope))
                sorted_slope = np.array(slope)[sort_index]

                ints_angle.append(np.pi / 2 - np.arctan(np.abs(sorted_slope[0])))
                ints_angle.append(np.pi / 2 - np.arctan(np.abs(sorted_slope[1])))

        ints_y = np.array(ints_y)
        ints_angle = np.array(ints_angle)

        sort_index = np.argsort(ints_y)
        ints_y_sort = ints_y[sort_index]
        ints_angle_sort = ints_angle[sort_index]

        return ints_y_sort, ints_angle_sort

    @cached_property
    def bounds(self) -> Bound:
        """Returns bounding box min and max coordinates. The dilation and slant angle are not
        taken into account exactly for speed. Instead, the polygon may be slightly smaller than
        the returned bounds, but it should always be fully contained.

        Returns
        -------
        Tuple[float, float, float], Tuple[float, float float]
            Min and max bounds packaged as ``(minx, miny, minz), (maxx, maxy, maxz)``.
        """

        # check for the maximum possible contribution from dilation/slant on each side
        max_offset = self.dilation
        if not isclose(self.sidewall_angle, 0):
            if self.reference_plane == "bottom":
                max_offset += max(0, -self._tanq * self.finite_length_axis)
            elif self.reference_plane == "top":
                max_offset += max(0, self._tanq * self.finite_length_axis)
            elif self.reference_plane == "middle":
                max_offset += max(0, abs(self._tanq) * self.finite_length_axis / 2)

        # special care when dilated
        if max_offset > 0:
            dilated_vertices = self._shift_vertices(
                self._proper_vertices(self.vertices), max_offset
            )[0]
            xmin, ymin = np.amin(dilated_vertices, axis=0)
            xmax, ymax = np.amax(dilated_vertices, axis=0)
        else:
            # otherwise, bounds are directly based on the supplied vertices
            xmin, ymin = np.amin(self.vertices, axis=0)
            xmax, ymax = np.amax(self.vertices, axis=0)

        # get bounds in (local) z
        zmin, zmax = self.slab_bounds

        # rearrange axes
        coords_min = self.unpop_axis(zmin, (xmin, ymin), axis=self.axis)
        coords_max = self.unpop_axis(zmax, (xmax, ymax), axis=self.axis)
        return (tuple(coords_min), tuple(coords_max))

    def _extrusion_length_to_offset_distance(self, extrusion: float) -> float:
        """Convert extrusion length to offset distance."""
        if isclose(self.sidewall_angle, 0):
            return 0
        return -extrusion * self._tanq

    @staticmethod
    def _area(vertices: np.ndarray) -> float:
        """Compute the signed polygon area (positive for CCW orientation).

        Parameters
        ----------
        vertices : np.ndarray
            Shape (N, 2) defining the polygon vertices in the xy-plane.

        Returns
        -------
        float
            Signed polygon area (positive for CCW orientation).
        """
        vert_shift = np.roll(vertices, axis=0, shift=-1)

        xs, ys = vertices.T
        xs_shift, ys_shift = vert_shift.T

        term1 = xs * ys_shift
        term2 = ys * xs_shift
        return np.sum(term1 - term2) * 0.5

    @staticmethod
    def _perimeter(vertices: np.ndarray) -> float:
        """Compute the polygon perimeter.

        Parameters
        ----------
        vertices : np.ndarray
            Shape (N, 2) defining the polygon vertices in the xy-plane.

        Returns
        -------
        float
            Polygon perimeter.
        """

        vert_shift = np.roll(vertices, axis=0, shift=-1)
        squared_diffs = (vertices - vert_shift) ** 2

        # distance along each edge
        dists = np.sqrt(squared_diffs.sum(axis=-1))

        # total distance along all edges
        return np.sum(dists)

    @staticmethod
    def _orient(vertices: np.ndarray) -> np.ndarray:
        """Return a CCW-oriented polygon.

        Parameters
        ----------
        vertices : np.ndarray
            Shape (N, 2) defining the polygon vertices in the xy-plane.

        Returns
        -------
        np.ndarray
            Vertices of a CCW-oriented polygon.
        """
        return vertices if PolySlab._area(vertices) > 0 else vertices[::-1, :]

    @staticmethod
    def _remove_duplicate_vertices(vertices: np.ndarray) -> np.ndarray:
        """Remove redundant/identical nearest neighbour vertices.

        Parameters
        ----------
        vertices : np.ndarray
            Shape (N, 2) defining the polygon vertices in the xy-plane.

        Returns
        -------
        np.ndarray
            Vertices of polygon.
        """

        vertices_f = np.roll(vertices, shift=-1, axis=0)
        vertices_diff = np.linalg.norm(vertices - vertices_f, axis=1)
        return vertices[~np.isclose(vertices_diff, 0, rtol=_IS_CLOSE_RTOL)]

    @staticmethod
    def _proper_vertices(vertices: ArrayFloat2D) -> np.ndarray:
        """convert vertices to np.array format,
        removing duplicate neighbouring vertices,
        and oriented in CCW direction.

        Returns
        -------
        ArrayLike[float, float]
           The vertices of the polygon for internal use.
        """

        vertices_np = PolySlab.vertices_to_array(vertices)
        return PolySlab._orient(PolySlab._remove_duplicate_vertices(vertices_np))

    @staticmethod
    def _edge_events_detection(
        proper_vertices: np.ndarray, dilation: float, ignore_at_dist: bool = True
    ) -> bool:
        """Detect any edge events within the offset distance ``dilation``.
        If ``ignore_at_dist=True``, the edge event at ``dist`` is ignored.
        """

        # ignore the event that occurs right at the offset distance
        if ignore_at_dist:
            dilation -= fp_eps * dilation / abs(dilation)
        # number of vertices before offsetting
        num_vertices = proper_vertices.shape[0]

        # 0) fully eroded?
        if dilation < 0 and dilation < -PolySlab._maximal_erosion(proper_vertices):
            return True

        # sample at a few dilation values
        dist_list = dilation * np.linspace(0, 1, 1 + _N_SAMPLE_POLYGON_INTERSECT)[1:]
        for dist in dist_list:
            # offset: we offset the vertices first, and then use shapely to make it proper
            # in principle, one can offset with shapely.buffer directly, but shapely somehow
            # automatically removes some vertices even though no change of topology.
            poly_offset = PolySlab._shift_vertices(proper_vertices, dist)[0]
            # flipped winding number
            if PolySlab._area(poly_offset) < fp_eps**2:
                return True

            poly_offset = shapely.make_valid(PolySlab.make_shapely_polygon(poly_offset))
            # 1) polygon split or create holes/islands
            if not poly_offset.geom_type == "Polygon" or len(poly_offset.interiors) > 0:
                return True

            # 2) reduction in vertex number
            offset_vertices = PolySlab._proper_vertices(list(poly_offset.exterior.coords))
            if offset_vertices.shape[0] != num_vertices:
                return True

            # 3) some split polygon might fully disappear after the offset, but they
            # can be detected if we offset back.
            poly_offset_back = shapely.make_valid(
                PolySlab.make_shapely_polygon(PolySlab._shift_vertices(offset_vertices, -dist)[0])
            )
            if poly_offset_back.geom_type == "MultiPolygon" or len(poly_offset_back.interiors) > 0:
                return True
            offset_back_vertices = list(poly_offset_back.exterior.coords)
            if PolySlab._proper_vertices(offset_back_vertices).shape[0] != num_vertices:
                return True

        return False

    @staticmethod
    def _neighbor_vertices_crossing_detection(
        vertices: np.ndarray, dist: float, ignore_at_dist: bool = True
    ) -> float:
        """Detect if neighboring vertices will cross after a dilation distance dist.

        Parameters
        ----------
        vertices : np.ndarray
            Shape (N, 2) defining the polygon vertices in the xy-plane.
        dist : float
            Distance to offset.
        ignore_at_dist : bool, optional
            whether to ignore the event right at ``dist`.

        Returns
        -------
        float
            the absolute value of the maximal allowed dilation
            if there are any crossing, otherwise return ``None``.
        """
        # ignore the event that occurs right at the offset distance
        if ignore_at_dist:
            dist -= fp_eps * dist / abs(dist)

        edge_length, edge_reduction = PolySlab._edge_length_and_reduction_rate(vertices)
        length_remaining = edge_length - edge_reduction * dist

        if np.any(length_remaining < 0):
            index_oversized = length_remaining < 0
            max_dist = np.min(
                np.abs(edge_length[index_oversized] / edge_reduction[index_oversized])
            )
            return max_dist
        return None

    @staticmethod
    def array_to_vertices(arr_vertices: np.ndarray) -> ArrayFloat2D:
        """Converts a numpy array of vertices to a list of tuples."""
        return list(arr_vertices)

    @staticmethod
    def vertices_to_array(vertices_tuple: ArrayFloat2D) -> np.ndarray:
        """Converts a list of tuples (vertices) to a numpy array."""
        return np.array(vertices_tuple)

    @staticmethod
    def _shift_vertices(
        vertices: np.ndarray, dist
    ) -> Tuple[np.ndarray, np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Shifts the vertices of a polygon outward uniformly by distances
        `dists`.

        Parameters
        ----------
        np.ndarray
            Shape (N, 2) defining the polygon vertices in the xy-plane.
        dist : float
            Distance to offset.

        Returns
        -------
        Tuple[np.ndarray, np.narray,Tuple[np.ndarray,np.ndarray]]
            New polygon vertices;
            and the shift of vertices in direction parallel to the edges.
            Shift along x and y direction.
        """

        if isclose(dist, 0):
            return vertices, np.zeros(vertices.shape[0], dtype=float), None

        def rot90(v):
            """90 degree rotation of 2d vector
            vx -> vy
            vy -> -vx
            """
            vxs, vys = v
            return np.stack((-vys, vxs), axis=0)

        def cross(u, v):
            return np.cross(u, v, axis=0)

        def normalize(v):
            return v / np.linalg.norm(v, axis=0)

        vs_orig = copy(vertices.T)
        vs_next = np.roll(copy(vs_orig), axis=-1, shift=-1)
        vs_previous = np.roll(copy(vs_orig), axis=-1, shift=+1)

        asp = normalize(vs_next - vs_orig)
        asm = normalize(vs_orig - vs_previous)

        # the vertex shift is decomposed into parallel and perpendicular directions
        perpendicular_shift = -dist
        det = cross(asm, asp)

        tan_half_angle = np.where(
            np.isclose(det, 0, rtol=_IS_CLOSE_RTOL),
            0.0,
            cross(asm, rot90(asm - asp)) / (det + np.isclose(det, 0, rtol=_IS_CLOSE_RTOL)),
        )
        parallel_shift = dist * tan_half_angle

        shift_total = perpendicular_shift * rot90(asm) + parallel_shift * asm
        shift_x = shift_total[0, :]
        shift_y = shift_total[1, :]

        return np.swapaxes(vs_orig + shift_total, -2, -1), parallel_shift, (shift_x, shift_y)

    @staticmethod
    def _edge_length_and_reduction_rate(vertices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Edge length of reduction rate of each edge with unit offset length.

        Parameters
        ----------
        vertices : np.ndarray
            Shape (N, 2) defining the polygon vertices in the xy-plane.

        Returns
        -------
        Tuple[np.ndarray, np.narray]
            edge length, and reduction rate
        """

        # edge length
        vs_orig = copy(vertices.T)
        vs_next = np.roll(copy(vs_orig), axis=-1, shift=-1)
        edge_length = np.linalg.norm(vs_next - vs_orig, axis=0)

        # edge length remaining
        dist = 1
        parallel_shift = PolySlab._shift_vertices(vertices, dist)[1]
        parallel_shift_p = np.roll(copy(parallel_shift), shift=-1)
        edge_reduction = -(parallel_shift + parallel_shift_p)
        return edge_length, edge_reduction

    @staticmethod
    def _maximal_erosion(vertices: np.ndarray) -> float:
        """The erosion value that reduces the length of
        all edges to be non-positive.
        """
        edge_length, edge_reduction = PolySlab._edge_length_and_reduction_rate(vertices)
        ind_nonzero = abs(edge_reduction) > fp_eps
        return -np.min(edge_length[ind_nonzero] / edge_reduction[ind_nonzero])

    @staticmethod
    def _heal_polygon(vertices: np.ndarray) -> np.ndarray:
        """heal a self-intersecting polygon."""
        shapely_poly = PolySlab.make_shapely_polygon(vertices)
        if shapely_poly.is_valid:
            return vertices
        elif isbox(vertices):
            raise NotImplementedError(
                "The dilation caused damage to the polygon. "
                "Automatically healing this is currently not supported when "
                "differentiating w.r.t. the vertices. Try increasing the spacing "
                "between vertices or reduce the amount of dilation."
            )
        # perform healing
        poly_heal = shapely.make_valid(shapely_poly)
        return PolySlab._proper_vertices(list(poly_heal.exterior.coords))

    def _volume(self, bounds: Bound) -> float:
        """Returns object's volume within given bounds."""

        z_min, z_max = self.slab_bounds

        z_min = max(z_min, bounds[0][self.axis])
        z_max = min(z_max, bounds[1][self.axis])

        length = z_max - z_min

        top_area = abs(self._area(self.top_polygon))
        base_area = abs(self._area(self.base_polygon))

        # https://mathworld.wolfram.com/PyramidalFrustum.html
        return 1.0 / 3.0 * length * (top_area + base_area + np.sqrt(top_area * base_area))

    def _surface_area(self, bounds: Bound) -> float:
        """Returns object's surface area within given bounds."""

        area = 0

        top_polygon = self.top_polygon
        base_polygon = self.base_polygon

        top_area = abs(self._area(top_polygon))
        base_area = abs(self._area(base_polygon))

        top_perim = self._perimeter(top_polygon)
        base_perim = self._perimeter(base_polygon)

        z_min, z_max = self.slab_bounds

        if z_min < bounds[0][self.axis]:
            z_min = bounds[0][self.axis]
        else:
            area += base_area

        if z_max > bounds[1][self.axis]:
            z_max = bounds[1][self.axis]
        else:
            area += top_area

        length = z_max - z_min

        area += 0.5 * (top_perim + base_perim) * length

        return area

    """ Autograd code """

    def compute_derivatives(self, derivative_info: DerivativeInfo) -> AutogradFieldMap:
        """Compute the adjoint derivatives for this object."""

        if derivative_info.paths != [("vertices",)]:
            raise ValueError("only support derivative wrt 'PolySlab.vertices'.")

        vjp_vertices = self.compute_derivative_vertices(derivative_info=derivative_info)

        return {("vertices",): vjp_vertices}

    def compute_derivative_vertices(self, derivative_info: DerivativeInfo) -> TracedVertices:
        # derivative w.r.t each edge

        vertices = np.array(self.vertices)
        num_vertices, _ = vertices.shape

        # compute edges between vertices

        vertices_next = np.roll(self.vertices, axis=0, shift=-1)
        edges = vertices_next - vertices

        # compute center positions between each edge
        edge_centers_plane = (vertices_next + vertices) / 2.0
        edge_centers_axis = self.center_axis * np.ones(num_vertices)
        edge_centers_xyz = self.unpop_axis_vect(edge_centers_axis, edge_centers_plane)

        if edge_centers_xyz.shape != (num_vertices, 3):
            raise AssertionError("something bad happened")

        # get basis vectors for every edge segment
        basis_vectors = self.edge_basis_vectors(edges=edges)

        grad_bases = derivative_info.grad_in_bases(
            spatial_coords=edge_centers_xyz, basis_vectors=basis_vectors
        )

        # unpack gradient contributions from different bases
        D_der_norm = grad_bases["D_norm"]
        E_der_edge = grad_bases["E_perp1"]
        E_der_slab = grad_bases["E_perp2"]

        # approximate permittivity in and out
        delta_eps_inv = 1.0 / derivative_info.eps_in - 1.0 / derivative_info.eps_out
        delta_eps = derivative_info.eps_in - derivative_info.eps_out

        # put together VJP using D_normal and E_perp integration
        vjps_edges = 0.0

        # perform D-normal integral
        contrib_D = -delta_eps_inv * D_der_norm
        vjps_edges += contrib_D

        # perform E-perpendicular integrals
        for E_der in (E_der_edge, E_der_slab):
            contrib_E = E_der * delta_eps
            vjps_edges += contrib_E

        # scale by edge area
        edge_lengths = np.linalg.norm(edges, axis=-1)
        edge_areas = edge_lengths

        # correction to edge area based on sidewall distance along slab axis
        slab_height = abs(float(np.squeeze(np.diff(self.slab_bounds))))
        if not np.isinf(slab_height):
            edge_areas *= slab_height

        vjps_edges *= edge_areas

        _, normal_vectors_in_plane = self.pop_axis_vect(basis_vectors["norm"])

        vjps_edges_in_plane = vjps_edges.values.reshape((num_vertices, 1)) * normal_vectors_in_plane

        vjps_vertices = vjps_edges_in_plane + np.roll(vjps_edges_in_plane, axis=0, shift=-1)
        vjps_vertices /= 2.0  # each vertex is effected only 1/2 by each edge

        # sign change if counter clockwise, because normal direction is flipped
        if self.is_ccw:
            vjps_vertices *= -1

        return vjps_vertices.real

    def edge_basis_vectors(
        self,
        edges: np.ndarray,  # (N, 2)
    ) -> dict[str, np.ndarray]:  # (N, 3)
        """Normalized basis vectors for 'normal' direction, 'slab' tangent direction and 'edge'."""

        num_vertices, _ = edges.shape
        zeros = np.zeros(num_vertices)
        ones = np.ones(num_vertices)

        # normalized vectors along edges
        edges_norm_in_plane = self.normalize_vect(edges)
        edges_norm_xyz = self.unpop_axis_vect(zeros, edges_norm_in_plane)

        # normalized vectors from base of edges to tops of edges
        slabs_axis_components = np.cos(self.sidewall_angle) * ones
        axis_norm = self.unpop_axis(1.0, (0.0, 0.0), axis=self.axis)
        slab_normal_xyz = -np.sin(self.sidewall_angle) * np.cross(edges_norm_xyz, axis_norm)
        _, slab_normal_in_plane = self.pop_axis_vect(slab_normal_xyz)
        slabs_norm_xyz = self.unpop_axis_vect(slabs_axis_components, slab_normal_in_plane)

        # normalized vectors pointing in normal direction of edge
        normals_norm_xyz = np.cross(edges_norm_xyz, slabs_norm_xyz)

        if self.axis != 1:
            normals_norm_xyz *= -1

        return dict(norm=normals_norm_xyz, perp1=edges_norm_xyz, perp2=slabs_norm_xyz)

    def unpop_axis_vect(self, ax_coords: np.ndarray, plane_coords: np.ndarray) -> np.ndarray:
        """Combine coordinate along axis with coordinates on the plane tangent to the axis.

        ax_coords.shape == [N]
        plane_coords.shape == [N, 2]
        return shape == [N, 3]

        """
        arr_xyz = self.unpop_axis(ax_coords, plane_coords.T, axis=self.axis)
        arr_xyz = np.stack(arr_xyz, axis=-1)
        return arr_xyz

    def pop_axis_vect(self, coord: np.ndarray) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Combine coordinate along axis with coordinates on the plane tangent to the axis.

        coord.shape == [N, 3]
        return shape == ([N], [N, 2]

        """

        arr_axis, arrs_plane = self.pop_axis(coord.T, axis=self.axis)
        arrs_plane = np.array(arrs_plane).T

        return arr_axis, arrs_plane

    @staticmethod
    def normalize_vect(arr: np.ndarray) -> np.ndarray:
        """normalize an array shaped (N, d) along the `d` axis and return (N, 1)."""
        norm = np.linalg.norm(arr, axis=-1, keepdims=True)
        norm = np.where(norm == 0, 1, norm)
        return arr / norm


class ComplexPolySlabBase(PolySlab):
    """Interface for dividing a complex polyslab where self-intersecting polygon can
    occur during extrusion. This class should not be used directly. Use instead
    :class:`plugins.polyslab.ComplexPolySlab`."""

    @pydantic.validator("vertices", always=True)
    def no_self_intersecting_polygon_during_extrusion(cls, val, values):
        """Turn off the validation for this class."""
        return val

    @classmethod
    def from_gds(
        cls,
        gds_cell,
        axis: Axis,
        slab_bounds: Tuple[float, float],
        gds_layer: int,
        gds_dtype: int = None,
        gds_scale: pydantic.PositiveFloat = 1.0,
        dilation: float = 0.0,
        sidewall_angle: float = 0,
        reference_plane: PlanePosition = "middle",
    ) -> List[PolySlab]:
        """Import :class:`.PolySlab` from a ``gdstk.Cell``.

        Parameters
        ----------
        gds_cell : gdstk.Cell
            ``gdstk.Cell`` containing 2D geometric data.
        axis : int
            Integer index into the polygon's slab axis. (0,1,2) -> (x,y,z).
        slab_bounds: Tuple[float, float]
            Minimum and maximum positions of the slab along ``axis``.
        gds_layer : int
            Layer index in the ``gds_cell``.
        gds_dtype : int = None
            Data-type index in the ``gds_cell``.
            If ``None``, imports all data for this layer into the returned list.
        gds_scale : float = 1.0
            Length scale used in GDS file in units of MICROMETER.
            For example, if gds file uses nanometers, set ``gds_scale=1e-3``.
            Must be positive.
        dilation : float = 0.0
            Dilation of the polygon in the base by shifting each edge along its
            normal outwards direction by a distance;
            a negative value corresponds to erosion.
        sidewall_angle : float = 0
            Angle of the sidewall.
            ``sidewall_angle=0`` (default) specifies vertical wall,
            while ``0<sidewall_angle<np.pi/2`` for the base to be larger than the top.
        reference_plane : PlanePosition = "middle"
            The position of the GDS layer. It can be at the ``bottom``, ``middle``,
            or ``top`` of the PolySlab. E.g. if ``axis=1``, ``bottom`` refers to the
            negative side of y-axis, and ``top`` refers to the positive side of y-axis.

        Returns
        -------
        List[:class:`.PolySlab`]
            List of :class:`.PolySlab` objects sharing ``axis`` and  slab bound properties.
        """

        # TODO: change for 2.0
        # handle reference plane kwarg
        all_vertices = PolySlab._load_gds_vertices(gds_cell, gds_layer, gds_dtype, gds_scale)
        polyslabs = [
            cls(
                vertices=verts,
                axis=axis,
                slab_bounds=slab_bounds,
                dilation=dilation,
                sidewall_angle=sidewall_angle,
                reference_plane=reference_plane,
            )
            for verts in all_vertices
        ]
        return [sub_poly for sub_polys in polyslabs for sub_poly in sub_polys.sub_polyslabs]

    @property
    def geometry_group(self) -> base.GeometryGroup:
        """Divide a complex polyslab into a list of simple polyslabs, which
        are assembled into a :class:`.GeometryGroup`.

        Returns
        -------
        :class:`.GeometryGroup`
            GeometryGroup for a list of simple polyslabs divided from the complex
            polyslab.
        """
        return base.GeometryGroup(geometries=self.sub_polyslabs)

    @property
    def sub_polyslabs(self) -> List[PolySlab]:
        """Divide a complex polyslab into a list of simple polyslabs.
        Only neighboring vertex-vertex crossing events are treated in this
        version.

        Returns
        -------
        List[PolySlab]
            A list of simple polyslabs.
        """
        sub_polyslab_list = []
        num_division_count = 0
        # initialize sub-polyslab parameters
        sub_polyslab_dict = self.dict(exclude={"type"}).copy()
        if isclose(self.sidewall_angle, 0):
            return [PolySlab.parse_obj(sub_polyslab_dict)]

        sub_polyslab_dict.update({"dilation": 0})  # dilation accounted in setup
        # initialize offset distance
        offset_distance = 0

        for dist_val in self._dilation_length:
            dist_now = 0.0
            vertices_now = self.reference_polygon

            # constructing sub-polyslabs until reaching the base/top
            while not isclose(dist_now, dist_val):
                # bounds for sub-polyslabs assuming no self-intersection
                slab_bounds = [
                    self._dilation_value_at_reference_to_coord(dist_now),
                    self._dilation_value_at_reference_to_coord(dist_val),
                ]
                # 1) find out any vertices touching events between the current
                # position to the base/top
                max_dist = PolySlab._neighbor_vertices_crossing_detection(
                    vertices_now, dist_val - dist_now
                )

                # vertices touching events captured, update bounds for sub-polyslab
                if max_dist is not None:
                    # max_dist doesn't have sign, so construct signed offset distance
                    offset_distance = max_dist * dist_val / abs(dist_val)
                    slab_bounds[1] = self._dilation_value_at_reference_to_coord(
                        dist_now + offset_distance
                    )

                # 2) construct sub-polyslab
                slab_bounds.sort()  # for reference_plane=top/bottom, bounds need to be ordered
                # direction of marching
                reference_plane = "bottom" if dist_val / self._tanq < 0 else "top"
                sub_polyslab_dict.update(
                    dict(
                        slab_bounds=tuple(slab_bounds),
                        vertices=vertices_now,
                        reference_plane=reference_plane,
                    )
                )
                sub_polyslab_list.append(PolySlab.parse_obj(sub_polyslab_dict))

                # Now Step 3
                if max_dist is None:
                    break
                dist_now += offset_distance
                # new polygon vertices where collapsing vertices are removed but keep one
                vertices_now = PolySlab._shift_vertices(vertices_now, offset_distance)[0]
                vertices_now = PolySlab._remove_duplicate_vertices(vertices_now)
                # all vertices collapse
                if len(vertices_now) < 3:
                    break
                # polygon collapse into 1D
                if self.make_shapely_polygon(vertices_now).buffer(0).area < fp_eps:
                    break
                vertices_now = PolySlab._orient(vertices_now)
                num_division_count += 1

        if num_division_count > _COMPLEX_POLYSLAB_DIVISIONS_WARN:
            log.warning(
                f"Too many self-intersecting events: the polyslab has been divided into "
                f"{num_division_count} polyslabs; more than {_COMPLEX_POLYSLAB_DIVISIONS_WARN} may "
                f"slow down the simulation."
            )

        return sub_polyslab_list

    @property
    def _dilation_length(self) -> List[float]:
        """dilation length from reference plane to the top/bottom of the polyslab."""

        # for "bottom", only needs to compute the offset length to the top
        dist = [self._extrusion_length_to_offset_distance(self.finite_length_axis)]
        # reverse the dilation value if the reference plane is on the top
        if self.reference_plane == "top":
            dist = [-dist[0]]
        # for middle, both directions
        elif self.reference_plane == "middle":
            dist = [dist[0] / 2, -dist[0] / 2]
        return dist

    def _dilation_value_at_reference_to_coord(self, dilation: float) -> float:
        """Compute the coordinate based on the dilation value to the reference plane."""

        z_coord = -dilation / self._tanq + self.slab_bounds[0]
        if self.reference_plane == "middle":
            return z_coord + self.finite_length_axis / 2
        if self.reference_plane == "top":
            return z_coord + self.finite_length_axis
        # bottom case
        return z_coord

    def intersections_tilted_plane(
        self, normal: Coordinate, origin: Coordinate, to_2D: MatrixReal4x4
    ) -> List[Shapely]:
        """Return a list of shapely geometries at the plane specified by normal and origin.

        Parameters
        ----------
        normal : Coordinate
            Vector defining the normal direction to the plane.
        origin : Coordinate
            Vector defining the plane origin.
        to_2D : MatrixReal4x4
            Transformation matrix to apply to resulting shapes.

        Returns
        -------
        List[shapely.geometry.base.BaseGeometry]
            List of 2D shapes that intersect plane.
            For more details refer to
            `Shapely's Documentation <https://shapely.readthedocs.io/en/stable/project.html>`_.
        """
        return [
            shapely.unary_union(
                [
                    base.Geometry.evaluate_inf_shape(shape)
                    for polyslab in self.sub_polyslabs
                    for shape in polyslab.intersections_tilted_plane(normal, origin, to_2D)
                ]
            )
        ]
