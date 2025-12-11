"""Geometry extruded from polygonal shapes."""

from __future__ import annotations

import math
from copy import copy
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Optional, Union

import autograd.numpy as np
import pydantic.v1 as pydantic
import shapely
from autograd.tracer import getval, isbox
from numpy._typing import NDArray
from numpy.polynomial.legendre import leggauss as _leggauss

from tidy3d.components.autograd import AutogradFieldMap, TracedVertices, get_static
from tidy3d.components.autograd.derivative_utils import DerivativeInfo
from tidy3d.components.autograd.types import TracedFloat
from tidy3d.components.base import cached_property, skip_if_fields_missing
from tidy3d.components.transformation import ReflectionFromPlane, RotationAroundAxis
from tidy3d.components.types import (
    ArrayFloat1D,
    ArrayFloat2D,
    ArrayLike,
    Axis,
    Bound,
    Coordinate,
    MatrixReal4x4,
    PlanePosition,
    Shapely,
)
from tidy3d.config import config
from tidy3d.constants import LARGE_NUMBER, MICROMETER, fp_eps
from tidy3d.exceptions import SetupError, Tidy3dImportError, ValidationError
from tidy3d.log import log
from tidy3d.packaging import verify_packages_import

from . import base, triangulation

if TYPE_CHECKING:
    from gdstk import Cell

# sampling polygon along dilation for validating polygon to be
# non self-intersecting during the entire dilation process
_N_SAMPLE_POLYGON_INTERSECT = 5

_IS_CLOSE_RTOL = np.finfo(float).eps

# Warn for too many divided polyslabs
_COMPLEX_POLYSLAB_DIVISIONS_WARN = 100

# Warn before triangulating large polyslabs due to inefficiency
_MAX_POLYSLAB_VERTICES_FOR_TRIANGULATION = 500

_MIN_POLYGON_AREA = fp_eps


@lru_cache(maxsize=128)
def leggauss(n: int) -> tuple[NDArray, NDArray]:
    """Cached version of leggauss with dtype conversions."""
    g, w = _leggauss(n)
    return g.astype(config.adjoint.gradient_dtype_float, copy=False), w.astype(
        config.adjoint.gradient_dtype_float, copy=False
    )


class PolySlab(base.Planar):
    """Polygon extruded with optional sidewall angle along axis direction.

    Example
    -------
    >>> vertices = np.array([(0,0), (1,0), (1,1)])
    >>> p = PolySlab(vertices=vertices, axis=2, slab_bounds=(-1, 1))
    """

    slab_bounds: tuple[TracedFloat, TracedFloat] = pydantic.Field(
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
    def slab_bounds_order(cls, val: tuple[float, float]) -> tuple[float, float]:
        """Maximum position of the slab should be no smaller than its minimal position."""
        if val[1] < val[0]:
            raise SetupError(
                "Polyslab.slab_bounds must be specified in the order of "
                "minimum and maximum positions of the slab along the axis. "
                f"But now the maximum {val[1]} is smaller than the minimum {val[0]}."
            )
        return val

    @pydantic.validator("vertices", always=True)
    def correct_shape(cls, val: ArrayFloat2D) -> ArrayFloat2D:
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
        if poly_heal.area < _MIN_POLYGON_AREA:
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
    def no_complex_self_intersecting_polygon_at_reference_plane(
        cls, val: ArrayFloat2D, values: dict[str, Any]
    ) -> ArrayFloat2D:
        """At the reference plane, check if the polygon is self-intersecting.

        There are two types of self-intersection that can occur during dilation:
        1) the one that creates holes/islands, or splits polygons, or removes everything;
        2) the one that does not.

        For 1), we issue an error since it is yet to be supported;
        For 2), we heal the polygon, and warn that the polygon has been cleaned up.
        """
        # no need to validate anything here
        if math.isclose(values["dilation"], 0):
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
    def no_self_intersecting_polygon_during_extrusion(
        cls, val: ArrayFloat2D, values: dict[str, Any]
    ) -> ArrayFloat2D:
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
        # sidewall_angle may be autograd-traced; use static value for this check only
        if math.isclose(getval(values["sidewall_angle"]), 0):
            return val

        # apply dilation
        poly_ref = PolySlab._proper_vertices(val)
        if not math.isclose(values["dilation"], 0):
            poly_ref = PolySlab._shift_vertices(poly_ref, values["dilation"])[0]
            poly_ref = PolySlab._heal_polygon(poly_ref)

        slab_min, slab_max = values["slab_bounds"]
        slab_bounds = [getval(slab_min), getval(slab_max)]

        # first, check vertex-vertex crossing at any point during extrusion
        length = slab_bounds[1] - slab_bounds[0]
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
        gds_cell: Cell,
        axis: Axis,
        slab_bounds: tuple[float, float],
        gds_layer: int,
        gds_dtype: Optional[int] = None,
        gds_scale: pydantic.PositiveFloat = 1.0,
        dilation: float = 0.0,
        sidewall_angle: float = 0,
        reference_plane: PlanePosition = "middle",
    ) -> list[PolySlab]:
        """Import :class:`PolySlab` from a ``gdstk.Cell``.

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
        gds_cell: Cell,
        gds_layer: int,
        gds_dtype: Optional[int] = None,
        gds_scale: pydantic.PositiveFloat = 1.0,
    ) -> list[ArrayFloat2D]:
        """Import :class:`PolySlab` from a ``gdstk.Cell``.

        Parameters
        ----------
        gds_cell : gdstk.Cell
            ``gdstk.Cell`` containing 2D geometric data.
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
        import gdstk

        gds_cell_class_name = str(gds_cell.__class__)
        if not isinstance(gds_cell, gdstk.Cell):
            if (
                "gdstk" in gds_cell_class_name
            ):  # Check if it might be a gdstk cell but gdstk is not found
                raise Tidy3dImportError(
                    "Module 'gdstk' not found. It is required to import gdstk cells."
                )
            raise ValueError(
                f"validate 'gds_cell' of type '{gds_cell_class_name}' "
                "does not seem to be associated with 'gdstk' package "
                "and therefore can't be loaded by Tidy3D."
            )

        all_vertices = base.Geometry.load_gds_vertices_gdstk(
            gds_cell=gds_cell,
            gds_layer=gds_layer,
            gds_dtype=gds_dtype,
            gds_scale=gds_scale,
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

    @property
    def finite_length_axis(self) -> float:
        """Gets the length of the PolySlab along the out of plane dimension.
        First clips the slab bounds to LARGE_NUMBER and then returns difference.
        """
        zmin, zmax = self.slab_bounds
        zmin = max(zmin, -LARGE_NUMBER)
        zmax = min(zmax, LARGE_NUMBER)
        return zmax - zmin

    @cached_property
    def reference_polygon(self) -> NDArray:
        """The polygon at the reference plane.

        Returns
        -------
        ArrayLike[float, float]
            The vertices of the polygon at the reference plane.
        """
        vertices = self._proper_vertices(self.vertices)
        if math.isclose(self.dilation, 0):
            return vertices
        offset_vertices = self._shift_vertices(vertices, self.dilation)[0]
        return self._heal_polygon(offset_vertices)

    @cached_property
    def middle_polygon(self) -> NDArray:
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
    def base_polygon(self) -> NDArray:
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
    def top_polygon(self) -> NDArray:
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

    def _update_from_bounds(self, bounds: tuple[float, float], axis: Axis) -> PolySlab:
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

    def inside(self, x: NDArray[float], y: NDArray[float], z: NDArray[float]) -> NDArray[bool]:
        """For input arrays ``x``, ``y``, ``z`` of arbitrary but identical shape, return an array
        with the same shape which is ``True`` for every point in zip(x, y, z) that is inside the
        volume of the :class:`Geometry`, and ``False`` otherwise.

        Note
        ----
        For slanted sidewalls, this function only works if x, y, and z are arrays produced by a
        ``meshgrid call``, i.e. 3D arrays and each is constant along one axis.

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

        if isinstance(x, np.ndarray):
            inside_polygon = np.zeros_like(inside_height)
            xs_slab = x[inside_height]
            ys_slab = y[inside_height]

            # vertical sidewall
            if math.isclose(self.sidewall_angle, 0):
                face_polygon = shapely.Polygon(self.reference_polygon).buffer(fp_eps)
                shapely.prepare(face_polygon)
                inside_polygon_slab = shapely.contains_xy(face_polygon, x=xs_slab, y=ys_slab)
                inside_polygon[inside_height] = inside_polygon_slab
            # slanted sidewall, offsetting vertices at each z
            else:
                # a helper function for moving axis
                def _move_axis(arr: NDArray) -> NDArray:
                    return np.moveaxis(arr, source=self.axis, destination=-1)

                def _move_axis_reverse(arr: NDArray) -> NDArray:
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
                    face_polygon = shapely.Polygon(vertices_z).buffer(fp_eps)
                    shapely.prepare(face_polygon)
                    xs = x_axis[:, :, 0].flatten()
                    ys = y_axis[:, :, 0].flatten()
                    inside_polygon_slab = shapely.contains_xy(face_polygon, x=xs, y=ys)
                    inside_polygon_axis[:, :, z_i] = inside_polygon_slab.reshape(x_axis.shape[:2])
                inside_polygon = _move_axis_reverse(inside_polygon_axis)
        else:
            vertices_z = self._shift_vertices(self.middle_polygon, dist)[0]
            face_polygon = self.make_shapely_polygon(vertices_z).buffer(fp_eps)
            point = shapely.Point(x, y)
            inside_polygon = face_polygon.covers(point)
        return inside_height * inside_polygon

    @verify_packages_import(["trimesh"])
    def _do_intersections_tilted_plane(
        self,
        normal: Coordinate,
        origin: Coordinate,
        to_2D: MatrixReal4x4,
        quad_segs: Optional[int] = None,
    ) -> list[Shapely]:
        """Return a list of shapely geometries at the plane specified by normal and origin.

        Parameters
        ----------
        normal : Coordinate
            Vector defining the normal direction to the plane.
        origin : Coordinate
            Vector defining the plane origin.
        to_2D : MatrixReal4x4
            Transformation matrix to apply to resulting shapes.
        quad_segs : Optional[int] = None
            Number of segments used to discretize circular shapes. Not used for PolySlab geometry.

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
                f"Processing PolySlabs with over {_MAX_POLYSLAB_VERTICES_FOR_TRIANGULATION} vertices can be slow.",
                log_once=True,
            )
        base_triangles = triangulation.triangulate(self.base_polygon)
        top_triangles = (
            base_triangles
            if math.isclose(self.sidewall_angle, 0)
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
        path, _ = section.to_2D(to_2D=to_2D)
        return path.polygons_full

    def _intersections_normal(self, z: float, quad_segs: Optional[int] = None) -> list[Shapely]:
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
        if math.isclose(self.sidewall_angle, 0):
            return [self.make_shapely_polygon(self.reference_polygon)]

        z0 = self.center_axis
        z_local = z - z0  # distance to the middle
        dist = -z_local * self._tanq
        vertices_z = self._shift_vertices(self.middle_polygon, dist)[0]
        return [self.make_shapely_polygon(vertices_z)]

    def _intersections_side(self, position: float, axis: int) -> list[Shapely]:
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
            if math.isclose(self.sidewall_angle, 0):
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

                if math.isclose(self.sidewall_angle, 0):
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

    def _find_intersecting_height(self, position: float, axis: int) -> NDArray:
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
        if math.isclose(self.sidewall_angle, 0):
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
        self,
        vertices: NDArray,
        position: float,
        axis: int,
        exclude_on_vertices: bool = False,
    ) -> tuple[NDArray, NDArray, NDArray]:
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

        # Find which segments intersect:
        # 1. Strictly crossing: one endpoint strictly left, one strictly right
        # 2. Touching: exactly one endpoint on the plane (xor), which excludes
        #    edges lying entirely on the plane (both endpoints at position).
        orig_on_plane = np.isclose(x_vertices_axis, position, rtol=_IS_CLOSE_RTOL)
        f_on_plane = np.roll(orig_on_plane, shift=-1)
        crosses_b = (x_vertices_axis > position) & (x_vertices_f < position)
        crosses_f = (x_vertices_axis < position) & (x_vertices_f > position)

        if exclude_on_vertices:
            # exclude vertices at the position
            not_touching = np.logical_not(orig_on_plane | f_on_plane)
            intersects_segment = (crosses_b | crosses_f) & not_touching
        else:
            single_touch = np.logical_xor(orig_on_plane, f_on_plane)
            intersects_segment = crosses_b | crosses_f | single_touch

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

        # Get rid of duplicate intersection points (vertices counted twice if directly on position)
        ints_y_sort, sort_index = np.unique(ints_y, return_index=True)
        ints_angle_sort = ints_angle[sort_index]

        # For tangent touches (vertex on plane, both neighbors on same side),
        # add y-value back to form a degenerate pair
        if not exclude_on_vertices:
            n = len(vertices_axis)
            for idx in np.where(orig_on_plane)[0]:
                prev_on = orig_on_plane[(idx - 1) % n]
                next_on = orig_on_plane[(idx + 1) % n]
                if not prev_on and not next_on:
                    prev_side = x_vertices_axis[(idx - 1) % n] > position
                    next_side = x_vertices_axis[(idx + 1) % n] > position
                    if prev_side == next_side:
                        ints_y_sort = np.append(ints_y_sort, vertices_axis[idx, 1])
                        ints_angle_sort = np.append(ints_angle_sort, 0)

            sort_index = np.argsort(ints_y_sort)
            ints_y_sort = ints_y_sort[sort_index]
            ints_angle_sort = ints_angle_sort[sort_index]
        return ints_y_sort, ints_angle_sort

    def _find_intersecting_ys_angle_slant(
        self, vertices: NDArray, position: float, axis: int
    ) -> tuple[NDArray, NDArray, NDArray]:
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
        # sidewall_angle may be autograd-traced; unbox for this check
        if not math.isclose(getval(self.sidewall_angle), 0):
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
        if math.isclose(self.sidewall_angle, 0):
            return 0
        return -extrusion * self._tanq

    @staticmethod
    def _area(vertices: NDArray) -> float:
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
    def _perimeter(vertices: NDArray) -> float:
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
    def _orient(vertices: NDArray) -> NDArray:
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
    def _remove_duplicate_vertices(vertices: NDArray) -> NDArray:
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
    def _proper_vertices(vertices: ArrayFloat2D) -> NDArray:
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
        proper_vertices: NDArray, dilation: float, ignore_at_dist: bool = True
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
        dist_list = (
            dilation
            * np.linspace(
                0, 1, 1 + _N_SAMPLE_POLYGON_INTERSECT, dtype=config.adjoint.gradient_dtype_float
            )[1:]
        )
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
        vertices: NDArray, dist: float, ignore_at_dist: bool = True
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
    def array_to_vertices(arr_vertices: NDArray) -> ArrayFloat2D:
        """Converts a numpy array of vertices to a list of tuples."""
        return list(arr_vertices)

    @staticmethod
    def vertices_to_array(vertices_tuple: ArrayFloat2D) -> NDArray:
        """Converts a list of tuples (vertices) to a numpy array."""
        return np.array(vertices_tuple)

    @cached_property
    def interior_angle(self) -> ArrayFloat1D:
        """Angle formed inside polygon by two adjacent edges."""

        def normalize(v: NDArray) -> NDArray:
            return v / np.linalg.norm(v, axis=0)

        vs_orig = self.reference_polygon.T
        vs_next = np.roll(vs_orig, axis=-1, shift=-1)
        vs_previous = np.roll(vs_orig, axis=-1, shift=+1)

        asp = normalize(vs_next - vs_orig)
        asm = normalize(vs_previous - vs_orig)

        cos_angle = asp[0] * asm[0] + asp[1] * asm[1]
        sin_angle = asp[0] * asm[1] - asp[1] * asm[0]

        angle = np.arccos(cos_angle)
        # concave angles
        angle[sin_angle < 0] = 2 * np.pi - angle[sin_angle < 0]
        return angle

    @staticmethod
    def _shift_vertices(
        vertices: NDArray, dist: float
    ) -> tuple[NDArray, NDArray, tuple[NDArray, NDArray]]:
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

        # 'dist' may be autograd-traced; unbox for the zero-check only
        if math.isclose(getval(dist), 0):
            return vertices, np.zeros(vertices.shape[0], dtype=float), None

        def rot90(v: tuple[NDArray, NDArray]) -> NDArray:
            """90 degree rotation of 2d vector
            vx -> vy
            vy -> -vx
            """
            vxs, vys = v
            return np.stack((-vys, vxs), axis=0)

        def cross(u: NDArray, v: NDArray) -> Any:
            return u[0] * v[1] - u[1] * v[0]

        def normalize(v: NDArray) -> NDArray:
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

        return (
            np.swapaxes(vs_orig + shift_total, -2, -1),
            parallel_shift,
            (shift_x, shift_y),
        )

    @staticmethod
    def _edge_length_and_reduction_rate(
        vertices: NDArray,
    ) -> tuple[NDArray, NDArray]:
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
    def _maximal_erosion(vertices: NDArray) -> float:
        """The erosion value that reduces the length of
        all edges to be non-positive.
        """
        edge_length, edge_reduction = PolySlab._edge_length_and_reduction_rate(vertices)
        ind_nonzero = abs(edge_reduction) > fp_eps
        return -np.min(edge_length[ind_nonzero] / edge_reduction[ind_nonzero])

    @staticmethod
    def _heal_polygon(vertices: NDArray) -> NDArray:
        """heal a self-intersecting polygon."""
        shapely_poly = PolySlab.make_shapely_polygon(vertices)
        if shapely_poly.is_valid:
            return vertices
        if isbox(vertices):
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

    def _compute_derivatives(self, derivative_info: DerivativeInfo) -> AutogradFieldMap:
        """
        Return VJPs while handling several edge-cases:

        - If the slab volume does not overlap the simulation, all grads are zero
          (one warning is issued).
        - Faces that lie completely outside the simulation give zero ``slab_bounds``
          gradients; this includes the +/- inf cases.
        - A 2d simulation collapses the surface integral to a line integral
        """
        vjps: AutogradFieldMap = {}

        intersect_min, intersect_max = map(np.asarray, derivative_info.bounds_intersect)
        sim_min, sim_max = map(np.asarray, derivative_info.simulation_bounds)

        extents = intersect_max - intersect_min
        is_2d = np.isclose(extents[self.axis], 0.0)

        # early return if polyslab is not in simulation domain
        slab_min, slab_max = self.slab_bounds
        if (slab_max < sim_min[self.axis]) or (slab_min > sim_max[self.axis]):
            log.warning(
                "'PolySlab' lies completely outside the simulation domain.",
                log_once=True,
            )
            for p in derivative_info.paths:
                vjps[p] = np.zeros_like(self.vertices) if p == ("vertices",) else 0.0
            return vjps

        # create interpolators once for ALL derivative computations
        # use provided interpolators if available to avoid redundant field data conversions
        interpolators = derivative_info.interpolators or derivative_info.create_interpolators(
            dtype=config.adjoint.gradient_dtype_float
        )

        for path in derivative_info.paths:
            if path == ("vertices",):
                vjps[path] = self._compute_derivative_vertices(
                    derivative_info, sim_min, sim_max, is_2d, interpolators
                )

            elif path == ("sidewall_angle",):
                vjps[path] = self._compute_derivative_sidewall_angle(
                    derivative_info, sim_min, sim_max, is_2d, interpolators
                )
            elif path[0] == "slab_bounds":
                idx = path[1]
                face_coord = self.slab_bounds[idx]

                # face entirely outside -> gradient 0
                if (
                    np.isinf(face_coord)
                    or face_coord < sim_min[self.axis]
                    or face_coord > sim_max[self.axis]
                    or is_2d
                ):
                    vjps[path] = 0.0
                    continue

                v = self._compute_derivative_slab_bounds(derivative_info, idx, interpolators)
                # outward-normal convention
                if idx == 0:
                    v *= -1
                vjps[path] = v
            else:
                raise ValueError(f"No derivative defined w.r.t. 'PolySlab' field '{path}'.")

        return vjps

    # ---- Shared helpers for VJP surface integrations ----
    def _z_slices(
        self, sim_min: NDArray, sim_max: NDArray, is_2d: bool, dx: float
    ) -> tuple[NDArray, float, float, float]:
        """Compute z-slice centers and spacing within bounds.

        Returns (z_centers, dz, z0, z1). For 2D, returns single center and dz=1.
        """
        if is_2d:
            midpoint_z = np.maximum(
                np.minimum(self.center_axis, sim_max[self.axis]),
                sim_min[self.axis],
            )
            zc = np.array([midpoint_z], dtype=config.adjoint.gradient_dtype_float)
            return zc, 1.0, self.center_axis, self.center_axis

        z0 = max(self.slab_bounds[0], sim_min[self.axis])
        z1 = min(self.slab_bounds[1], sim_max[self.axis])
        if z1 <= z0:
            return np.array([], dtype=config.adjoint.gradient_dtype_float), 0.0, z0, z1

        n_z = max(1, int(np.ceil((z1 - z0) / dx)))
        dz = (z1 - z0) / n_z
        z_centers = np.linspace(
            z0 + dz / 2, z1 - dz / 2, n_z, dtype=config.adjoint.gradient_dtype_float
        )
        return z_centers, dz, z0, z1

    @staticmethod
    def _clip_edges_to_bounds_batch(
        segment_starts: NDArray,
        segment_ends: NDArray,
        sim_min: NDArray,
        sim_max: NDArray,
        *,
        _edge_clip_tol: Optional[float] = None,
        _dtype: Optional[type] = None,
    ) -> tuple[NDArray, NDArray, NDArray]:
        """
        Compute parametric bounds for multiple segments clipped to simulation bounds.

        Parameters
        ----------
        segment_starts : NDArray
            (N, 3) array of segment start coordinates.
        segment_ends : NDArray
            (N, 3) array of segment end coordinates.
        sim_min : NDArray
            (3,) array of simulation minimum bounds.
        sim_max : NDArray
            (3,) array of simulation maximum bounds.

        Returns
        -------
        is_within_bounds : NDArray
            (N,) boolean array indicating if the segment intersects the bounds.
        t_starts : NDArray
            (N,) array of parametric start values (0.0 to 1.0).
        t_ends : NDArray
            (N,) array of parametric end values (0.0 to 1.0).
        """
        n = segment_starts.shape[0]
        if _edge_clip_tol is None:
            _edge_clip_tol = config.adjoint.edge_clip_tolerance
        if _dtype is None:
            _dtype = config.adjoint.gradient_dtype_float

        t_starts = np.zeros(n, dtype=_dtype)
        t_ends = np.ones(n, dtype=_dtype)
        is_within_bounds = np.ones(n, dtype=bool)

        for dim in range(3):
            start_coords = segment_starts[:, dim]
            end_coords = segment_ends[:, dim]
            bound_min = sim_min[dim]
            bound_max = sim_max[dim]

            # check for parallel edges (faster than isclose)
            parallel = np.abs(start_coords - end_coords) < 1e-12

            # parallel edges: check if outside bounds
            outside = parallel & (
                (start_coords < (bound_min - _edge_clip_tol))
                | (start_coords > (bound_max + _edge_clip_tol))
            )
            is_within_bounds &= ~outside

            # non-parallel edges: compute t_min, t_max
            not_parallel = ~parallel & is_within_bounds
            if np.any(not_parallel):
                denom = np.where(not_parallel, end_coords - start_coords, 1.0)  # avoid div by zero
                t_min = (bound_min - start_coords) / denom
                t_max = (bound_max - start_coords) / denom

                # swap if needed
                swap = t_min > t_max
                t_min_new = np.where(swap, t_max, t_min)
                t_max_new = np.where(swap, t_min, t_max)

                # update t_starts and t_ends for valid non-parallel edges
                t_starts = np.where(not_parallel, np.maximum(t_starts, t_min_new), t_starts)
                t_ends = np.where(not_parallel, np.minimum(t_ends, t_max_new), t_ends)

                # still valid?
                is_within_bounds &= ~not_parallel | (t_starts < t_ends)

        is_within_bounds &= t_ends > t_starts + _edge_clip_tol

        return is_within_bounds, t_starts, t_ends

    @staticmethod
    def _adaptive_edge_samples(
        L: float,
        dx: float,
        t_start: float = 0.0,
        t_end: float = 1.0,
        *,
        _sample_fraction: Optional[float] = None,
        _gauss_order: Optional[int] = None,
        _dtype: Optional[type] = None,
    ) -> tuple[NDArray, NDArray]:
        """
        Compute Gauss samples and weights along [t_start, t_end] with adaptive count.

        Parameters
        ----------
        L : float
            Physical length of the full edge.
        dx : float
            Target discretization step size.
        t_start : float, optional
            Start parameter, by default 0.0.
        t_end : float, optional
            End parameter, by default 1.0.

        Returns
        -------
        tuple[NDArray, NDArray]
            Tuple of (samples, weights) for the integration.
        """
        if _sample_fraction is None:
            _sample_fraction = config.adjoint.quadrature_sample_fraction
        if _gauss_order is None:
            _gauss_order = config.adjoint.gauss_quadrature_order
        if _dtype is None:
            _dtype = config.adjoint.gradient_dtype_float

        L_eff = L * max(0.0, t_end - t_start)
        n_uniform = max(1, int(np.ceil(L_eff / dx)))
        n_gauss = n_uniform if n_uniform <= 3 else max(2, int(n_uniform * _sample_fraction))
        if n_gauss <= _gauss_order:
            g, w = leggauss(n_gauss)
            half_range = 0.5 * (t_end - t_start)
            s = (half_range * g + 0.5 * (t_end + t_start)).astype(_dtype, copy=False)
            wt = (w * half_range).astype(_dtype, copy=False)
            return s, wt

        # composite Gauss with fixed local order
        g_loc, w_loc = leggauss(_gauss_order)
        segs = n_uniform
        edges_t = np.linspace(t_start, t_end, segs + 1, dtype=_dtype)

        # compute all segments at once
        a = edges_t[:-1]  # (segs,)
        b = edges_t[1:]  # (segs,)
        half_width = 0.5 * (b - a)  # (segs,)
        mid = 0.5 * (b + a)  # (segs,)

        # (segs, 1) * (order,) + (segs, 1) -> (segs, order)
        S = (half_width[:, None] * g_loc + mid[:, None]).astype(_dtype, copy=False)
        W = (half_width[:, None] * w_loc).astype(_dtype, copy=False)
        return S.ravel(), W.ravel()

    def _collect_sidewall_patches(
        self,
        vertices: NDArray,
        next_v: NDArray,
        edges: NDArray,
        basis: dict,
        sim_min: NDArray,
        sim_max: NDArray,
        is_2d: bool,
        dx: float,
    ) -> dict:
        """
        Collect sidewall patch geometry for batched VJP evaluation.

        Parameters
        ----------
        vertices : NDArray
            Array of polygon vertices.
        next_v : NDArray
            Array of next vertices (forming edges).
        edges : NDArray
            Edge vectors.
        basis : dict
            Basis vectors dictionary.
        sim_min : NDArray
            Simulation minimum bounds.
        sim_max : NDArray
            Simulation maximum bounds.
        is_2d : bool
            Whether the simulation is 2D.
        dx : float
            Discretization step.

        Returns
        -------
        dict
            Dictionary containing:
            - centers: (N, 3) array of patch centers.
            - normals: (N, 3) array of patch normals.
            - perps1: (N, 3) array of first tangent vectors.
            - perps2: (N, 3) array of second tangent vectors.
            - Ls: (N,) array of edge lengths.
            - s_vals: (N,) array of parametric coordinates along the edge.
            - s_weights: (N,) array of quadrature weights.
            - zc_vals: (N,) array of z-coordinates.
            - dz: float, slice thickness.
            - edge_indices: (N,) array of original edge indices.
        """
        # cache config values to avoid repeated lookups (overhead not insignificant here)
        _dtype = config.adjoint.gradient_dtype_float
        _edge_clip_tol = config.adjoint.edge_clip_tolerance
        _sample_fraction = config.adjoint.quadrature_sample_fraction
        _gauss_order = config.adjoint.gauss_quadrature_order

        theta = get_static(self.sidewall_angle)
        z_ref = self.reference_axis_pos

        cos_th = np.cos(theta)
        cos_th = np.clip(cos_th, 1e-12, 1.0)
        tan_th = np.tan(theta)
        dprime = -tan_th  # dd/dz

        # axis unit vector in 3D
        axis_vec = np.zeros(3, dtype=_dtype)
        axis_vec[self.axis] = 1.0

        # densify along axis as |theta| grows, dz scales with cos(theta)
        z_centers, dz, z0, z1 = self._z_slices(sim_min, sim_max, is_2d=is_2d, dx=dx * cos_th)

        # early exit: no slices
        if (not is_2d) and len(z_centers) == 0:
            return {
                "centers": np.empty((0, 3), dtype=_dtype),
                "normals": np.empty((0, 3), dtype=_dtype),
                "perps1": np.empty((0, 3), dtype=_dtype),
                "perps2": np.empty((0, 3), dtype=_dtype),
                "Ls": np.empty((0,), dtype=_dtype),
                "s_vals": np.empty((0,), dtype=_dtype),
                "s_weights": np.empty((0,), dtype=_dtype),
                "zc_vals": np.empty((0,), dtype=_dtype),
                "dz": dz,
                "edge_indices": np.empty((0,), dtype=int),
            }

        # estimate patches for pre-allocation
        n_edges = len(vertices)
        estimated_patches = 0
        denom_edge = max(dx * cos_th, 1e-12)
        for ei in range(n_edges):
            v0, v1 = vertices[ei], next_v[ei]
            L = np.linalg.norm(v1 - v0)
            if not np.isclose(L, 0.0):
                # prealloc guided by actual step; ds_phys scales with cos(theta)
                n_samples = max(1, int(np.ceil(L / denom_edge) * 0.6))
                estimated_patches += n_samples * max(1, len(z_centers))
        estimated_patches = int(max(1, estimated_patches) * 1.2)

        # pre-allocate arrays
        centers = np.empty((estimated_patches, 3), dtype=_dtype)
        normals = np.empty((estimated_patches, 3), dtype=_dtype)
        perps1 = np.empty((estimated_patches, 3), dtype=_dtype)
        perps2 = np.empty((estimated_patches, 3), dtype=_dtype)
        Ls = np.empty((estimated_patches,), dtype=_dtype)
        s_vals = np.empty((estimated_patches,), dtype=_dtype)
        s_weights = np.empty((estimated_patches,), dtype=_dtype)
        zc_vals = np.empty((estimated_patches,), dtype=_dtype)
        edge_indices = np.empty((estimated_patches,), dtype=int)

        patch_idx = 0

        # if the simulation is effectively 2D (one tangential dimension collapsed),
        # slightly expand degenerate bounds to enable finite-length clipping of edges.
        sim_min_eff = np.array(sim_min, dtype=_dtype)
        sim_max_eff = np.array(sim_max, dtype=_dtype)
        for dim in range(3):
            if dim == self.axis:
                continue
            if np.isclose(sim_max_eff[dim] - sim_min_eff[dim], 0.0):
                sim_min_eff[dim] -= 0.5 * dx
                sim_max_eff[dim] += 0.5 * dx

        # pre-compute values that are constant across z slices
        n_z = len(z_centers)
        z_centers_arr = np.asarray(z_centers, dtype=_dtype)

        # slanted local basis (constant across z for non-slanted case)
        # for slanted: rz = axis_vec + dprime * n2d, but dprime is constant
        for ei, (v0, v1) in enumerate(zip(vertices, next_v)):
            edge_vec = v1 - v0
            L = np.sqrt(np.dot(edge_vec, edge_vec))
            if L < 1e-12:
                continue

            # constant along edge: unit tangent in 3D (no axis component)
            t_edge = basis["perp1"][ei]

            # outward in-plane normal from canonical basis normal
            n2d = basis["norm"][ei].copy()
            n2d[self.axis] = 0.0
            nrm = np.linalg.norm(n2d)
            if not np.isclose(nrm, 0.0):
                n2d = n2d / nrm
            else:
                # fallback to right-handed construction if degenerate
                tmp = np.cross(axis_vec, t_edge)
                n2d = tmp / (np.linalg.norm(tmp) + 1e-20)

            # compute basis vectors once per edge
            rz = axis_vec + dprime * n2d
            T1_vec = t_edge
            N_vec = np.cross(T1_vec, rz)
            N_norm = np.linalg.norm(N_vec)
            if not np.isclose(N_norm, 0.0):
                N_vec = N_vec / N_norm

            # align N with outward edge normal
            if float(np.dot(N_vec, basis["norm"][ei])) < 0.0:
                N_vec = -N_vec

            T2_vec = np.cross(N_vec, T1_vec)
            T2_norm = np.linalg.norm(T2_vec)
            if not np.isclose(T2_norm, 0.0):
                T2_vec = T2_vec / T2_norm

            # batch compute offsets for all z slices at once
            d_all = -(z_centers_arr - z_ref) * tan_th  # (n_z,)
            offsets_3d = d_all[:, None] * n2d  # (n_z, 3) - faster than np.outer

            # batch compute segment starts and ends for all z slices
            segment_starts = np.empty((n_z, 3), dtype=_dtype)
            segment_ends = np.empty((n_z, 3), dtype=_dtype)
            plane_axes = [i for i in range(3) if i != self.axis]
            segment_starts[:, self.axis] = z_centers_arr
            segment_starts[:, plane_axes] = v0
            segment_starts += offsets_3d
            segment_ends[:, self.axis] = z_centers_arr
            segment_ends[:, plane_axes] = v1
            segment_ends += offsets_3d

            # batch clip all z slices at once
            is_within_bounds, t_starts, t_ends = self._clip_edges_to_bounds_batch(
                segment_starts,
                segment_ends,
                sim_min_eff,
                sim_max_eff,
                _edge_clip_tol=_edge_clip_tol,
                _dtype=_dtype,
            )

            # process only valid z slices (sampling has variable output sizes)
            valid_indices = np.nonzero(is_within_bounds)[0]
            if len(valid_indices) == 0:
                continue

            # group z slices by unique (t0, t1) pairs to avoid redundant quadrature calculations.
            # since most z-slices will have identical clipping bounds (0.0, 1.0),
            # we can compute the Gauss samples once and reuse them for almost all slices.
            # rounding ensures we get cache hits despite tiny floating point differences.
            t0_valid = np.round(t_starts[valid_indices], 10)
            t1_valid = np.round(t_ends[valid_indices], 10)

            # simple cache for sampling results: (t0, t1) -> (s_list, w_list)
            sample_cache = {}

            # process each z slice
            for zi, t0, t1 in zip(valid_indices, t0_valid, t1_valid):
                if (t0, t1) not in sample_cache:
                    sample_cache[(t0, t1)] = self._adaptive_edge_samples(
                        L,
                        denom_edge,
                        t0,
                        t1,
                        _sample_fraction=_sample_fraction,
                        _gauss_order=_gauss_order,
                        _dtype=_dtype,
                    )

                s_list, w_list = sample_cache[(t0, t1)]
                if len(s_list) == 0:
                    continue

                zc = z_centers_arr[zi]
                offset3d = offsets_3d[zi]

                pts2d = v0 + s_list[:, None] * edge_vec  # faster than np.outer

                # inline unpop_axis_vect for xyz computation
                n_pts = len(s_list)
                xyz = np.empty((n_pts, 3), dtype=_dtype)
                xyz[:, self.axis] = zc
                xyz[:, plane_axes] = pts2d
                xyz += offset3d

                n_patches = n_pts
                new_size_needed = patch_idx + n_patches
                if new_size_needed > centers.shape[0]:
                    # grow arrays by 1.5x to avoid frequent reallocations
                    new_size = int(new_size_needed * 1.5)
                    centers.resize((new_size, 3), refcheck=False)
                    normals.resize((new_size, 3), refcheck=False)
                    perps1.resize((new_size, 3), refcheck=False)
                    perps2.resize((new_size, 3), refcheck=False)
                    Ls.resize((new_size,), refcheck=False)
                    s_vals.resize((new_size,), refcheck=False)
                    s_weights.resize((new_size,), refcheck=False)
                    zc_vals.resize((new_size,), refcheck=False)
                    edge_indices.resize((new_size,), refcheck=False)

                sl = slice(patch_idx, patch_idx + n_patches)
                centers[sl] = xyz
                normals[sl] = N_vec
                perps1[sl] = T1_vec
                perps2[sl] = T2_vec
                Ls[sl] = L
                s_vals[sl] = s_list
                s_weights[sl] = w_list
                zc_vals[sl] = zc
                edge_indices[sl] = ei

                patch_idx += n_patches

        # trim arrays to final size
        centers = centers[:patch_idx]
        normals = normals[:patch_idx]
        perps1 = perps1[:patch_idx]
        perps2 = perps2[:patch_idx]
        Ls = Ls[:patch_idx]
        s_vals = s_vals[:patch_idx]
        s_weights = s_weights[:patch_idx]
        zc_vals = zc_vals[:patch_idx]
        edge_indices = edge_indices[:patch_idx]

        return {
            "centers": centers,
            "normals": normals,
            "perps1": perps1,
            "perps2": perps2,
            "Ls": Ls,
            "s_vals": s_vals,
            "s_weights": s_weights,
            "zc_vals": zc_vals,
            "dz": dz,
            "edge_indices": edge_indices,
        }

    def _compute_derivative_sidewall_angle(
        self,
        derivative_info: DerivativeInfo,
        sim_min: NDArray,
        sim_max: NDArray,
        is_2d: bool = False,
        interpolators: Optional[dict] = None,
    ) -> float:
        """VJP for dJ/dtheta where theta = sidewall_angle.

        Use dJ/dtheta = integral_S g(x) * V_n(x; theta) * dA, with g(x) from
        `evaluate_gradient_at_points`. For a ruled sidewall built by
        offsetting the mid-plane polygon by d(z) = -(z - z_ref) * tan(theta),
        the normal velocity is V_n = (dd/dtheta) * cos(theta) = -(z - z_ref)/cos(theta)
        and the area element is dA = (dz/cos(theta)) * d_ell.
        Therefore each patch weight is w = L * dz * (-(z - z_ref)) / cos(theta)^2.
        """
        if interpolators is None:
            interpolators = derivative_info.create_interpolators(
                dtype=config.adjoint.gradient_dtype_float
            )

        # 2D sim => no dependence on theta (z_local=0)
        if is_2d:
            return 0.0

        vertices, next_v, edges, basis = self._edge_geometry_arrays()

        dx = derivative_info.adaptive_vjp_spacing()

        # collect patches once
        patch = self._collect_sidewall_patches(
            vertices=vertices,
            next_v=next_v,
            edges=edges,
            basis=basis,
            sim_min=sim_min,
            sim_max=sim_max,
            is_2d=False,
            dx=dx,
        )
        if patch["centers"].shape[0] == 0:
            return 0.0

        # Shape-derivative factors:
        # - Offset: d(z) = -(z - z_ref) * tan(theta)
        # - Tangential rate: dd/dtheta = -(z - z_ref) * sec(theta)^2
        # - Normal velocity (project to surface normal): V_n = (dd/dtheta) * cos(theta) = -(z - z_ref)/cos(theta)
        # - Area element of slanted strip: dA = (dz/cos(theta)) * d_ell
        # => Patch weight scales as: V_n * dA = -(z - z_ref) * dz * d_ell / cos(theta)^2
        cos_theta = np.cos(get_static(self.sidewall_angle))
        inv_cos2 = 1.0 / (cos_theta * cos_theta)
        z_ref = self.reference_axis_pos

        g = derivative_info.evaluate_gradient_at_points(
            patch["centers"], patch["normals"], patch["perps1"], patch["perps2"], interpolators
        )
        z_local = patch["zc_vals"] - z_ref
        weights = patch["Ls"] * patch["s_weights"] * patch["dz"] * (-z_local) * inv_cos2
        return float(np.real(np.sum(g * weights)))

    def _compute_derivative_slab_bounds(
        self, derivative_info: DerivativeInfo, min_max_index: int, interpolators: dict
    ) -> float:
        """VJP for one of the two horizontal faces of a ``PolySlab``.

        The face is discretized into a Cartesian grid of small planar patches
        whose linear size does not exceed ``_VJP_SAMPLE_SPACING``. The adjoint surface
        integral is evaluated on every retained patch; the resulting derivative
        is split equally between the two vertices that bound the edge segment.
        """
        # rmin/rmax over the geometry and simulation box
        if np.isclose(self.slab_bounds[1] - self.slab_bounds[0], 0.0):
            log.warning(
                "Computing slab face derivatives for flat structures is not fully supported and "
                "may give zero for the derivative. Try using a structure with a small, but nonzero "
                "thickness for slab bound derivatives."
            )
        rmin, rmax = derivative_info.bounds_intersect
        _, (r1_min, r2_min) = self.pop_axis(rmin, axis=self.axis)
        _, (r1_max, r2_max) = self.pop_axis(rmax, axis=self.axis)
        ax_val = self.slab_bounds[min_max_index]

        # planar grid resolution, clipped to polygon bounding box
        face_verts = self.base_polygon if min_max_index == 0 else self.top_polygon
        face_poly = shapely.Polygon(face_verts).buffer(fp_eps)

        # limit the patch grid to the face that lives inside the simulation box
        poly_min_r1, poly_min_r2, poly_max_r1, poly_max_r2 = face_poly.bounds
        r1_min = max(r1_min, poly_min_r1)
        r1_max = min(r1_max, poly_max_r1)
        r2_min = max(r2_min, poly_min_r2)
        r2_max = min(r2_max, poly_max_r2)

        # intersect the polygon with the simulation bounds
        face_poly = face_poly.intersection(shapely.box(r1_min, r2_min, r1_max, r2_max))

        if (r1_max <= r1_min) and (r2_max <= r2_min):
            # the polygon does not intersect the current simulation slice
            return 0.0

        # re-compute the extents after clipping to the polygon bounds
        extents = np.array([r1_max - r1_min, r2_max - r2_min])

        # choose surface or line integral
        integral_fun = (
            self.compute_derivative_slab_bounds_line
            if np.isclose(extents, 0).any()
            else self.compute_derivative_slab_bounds_surface
        )
        return integral_fun(
            derivative_info,
            extents,
            r1_min,
            r1_max,
            r2_min,
            r2_max,
            ax_val,
            face_poly,
            min_max_index,
            interpolators,
        )

    def compute_derivative_slab_bounds_line(
        self,
        derivative_info: DerivativeInfo,
        extents: NDArray,
        r1_min: float,
        r1_max: float,
        r2_min: float,
        r2_max: float,
        ax_val: float,
        face_poly: shapely.Polygon,
        min_max_index: int,
        interpolators: dict,
    ) -> float:
        """Handle degenerate line cross-section case"""
        line_dim = 1 if np.isclose(extents[0], 0) else 0

        poly_min_r1, poly_min_r2, poly_max_r1, poly_max_r2 = face_poly.bounds
        if line_dim == 0:  # x varies, y is fixed
            l_min = max(r1_min, poly_min_r1)
            l_max = min(r1_max, poly_max_r1)
        else:  # y varies, x is fixed
            l_min = max(r2_min, poly_min_r2)
            l_max = min(r2_max, poly_max_r2)

        length = l_max - l_min
        if np.isclose(length, 0):
            return 0.0

        dx = derivative_info.adaptive_vjp_spacing()
        n_seg = max(1, int(np.ceil(length / dx)))
        coords = np.linspace(
            l_min, l_max, 2 * n_seg + 1, dtype=config.adjoint.gradient_dtype_float
        )[1::2]

        # build XY coordinates and in-plane direction vectors
        if line_dim == 0:
            xy = np.column_stack((coords, np.full_like(coords, r2_min)))
            dir_vec_plane = np.column_stack((np.ones_like(coords), np.zeros_like(coords)))
        else:
            xy = np.column_stack((np.full_like(coords, r1_min), coords))
            dir_vec_plane = np.column_stack((np.zeros_like(coords), np.ones_like(coords)))

        inside = shapely.contains_xy(face_poly, xy[:, 0], xy[:, 1])
        if not inside.any():
            return 0.0

        xy = xy[inside]
        dir_vec_plane = dir_vec_plane[inside]
        n_pts = len(xy)

        centers_xyz = self.unpop_axis_vect(np.full(n_pts, ax_val), xy)
        areas = np.full(n_pts, length / n_seg)  # patch length

        normals_xyz = self.unpop_axis_vect(
            np.full(
                n_pts, -1 if min_max_index == 0 else 1, dtype=config.adjoint.gradient_dtype_float
            ),
            np.zeros_like(xy, dtype=config.adjoint.gradient_dtype_float),
        )
        perps1_xyz = self.unpop_axis_vect(np.zeros(n_pts), dir_vec_plane)
        perps2_xyz = self.unpop_axis_vect(np.zeros(n_pts), np.zeros_like(dir_vec_plane))

        vjps = derivative_info.evaluate_gradient_at_points(
            centers_xyz, normals_xyz, perps1_xyz, perps2_xyz, interpolators
        )
        return np.real(np.sum(vjps * areas)).item()

    def compute_derivative_slab_bounds_surface(
        self,
        derivative_info: DerivativeInfo,
        extents: NDArray,
        r1_min: float,
        r1_max: float,
        r2_min: float,
        r2_max: float,
        ax_val: float,
        face_poly: shapely.Polygon,
        min_max_index: int,
        interpolators: dict,
    ) -> float:
        """2d surface integral on a Gauss quadrature grid"""
        dx = derivative_info.adaptive_vjp_spacing()

        # uniform grid would use n1 x n2 points
        n1_uniform, n2_uniform = np.maximum(1, np.ceil(extents / dx).astype(int))

        # use ~1/2 Gauss points in each direction for similar accuracy
        n1 = max(2, n1_uniform // 2)
        n2 = max(2, n2_uniform // 2)

        g1, w1 = leggauss(n1)
        g2, w2 = leggauss(n2)

        coords1 = (0.5 * (r1_max - r1_min) * g1 + 0.5 * (r1_max + r1_min)).astype(
            config.adjoint.gradient_dtype_float, copy=False
        )
        coords2 = (0.5 * (r2_max - r2_min) * g2 + 0.5 * (r2_max + r2_min)).astype(
            config.adjoint.gradient_dtype_float, copy=False
        )

        r1_grid, r2_grid = np.meshgrid(coords1, coords2, indexing="ij")
        r1_flat = r1_grid.flatten()
        r2_flat = r2_grid.flatten()
        pts = np.column_stack((r1_flat, r2_flat))

        in_face = shapely.contains_xy(face_poly, pts[:, 0], pts[:, 1])
        if not in_face.any():
            return 0.0

        xyz = self.unpop_axis_vect(
            np.full(in_face.sum(), ax_val, dtype=config.adjoint.gradient_dtype_float), pts[in_face]
        )
        n_patches = xyz.shape[0]

        normals_xyz = self.unpop_axis_vect(
            np.full(
                n_patches,
                -1 if min_max_index == 0 else 1,
                dtype=config.adjoint.gradient_dtype_float,
            ),
            np.zeros((n_patches, 2), dtype=config.adjoint.gradient_dtype_float),
        )
        perps1_xyz = self.unpop_axis_vect(
            np.zeros(n_patches, dtype=config.adjoint.gradient_dtype_float),
            np.column_stack(
                (
                    np.ones(n_patches, dtype=config.adjoint.gradient_dtype_float),
                    np.zeros(n_patches, dtype=config.adjoint.gradient_dtype_float),
                )
            ),
        )
        perps2_xyz = self.unpop_axis_vect(
            np.zeros(n_patches, dtype=config.adjoint.gradient_dtype_float),
            np.column_stack(
                (
                    np.zeros(n_patches, dtype=config.adjoint.gradient_dtype_float),
                    np.ones(n_patches, dtype=config.adjoint.gradient_dtype_float),
                )
            ),
        )

        w1_grid, w2_grid = np.meshgrid(w1, w2, indexing="ij")
        weights_flat = (w1_grid * w2_grid).flatten()[in_face]
        jacobian = 0.25 * (r1_max - r1_min) * (r2_max - r2_min)

        # area-based correction for non-rectangular domains (e.g. concave polygon)
        # for constant integrand, integral should equal polygon area
        sum_weights = np.sum(weights_flat)
        if sum_weights > 0:
            area_correction = face_poly.area / (sum_weights * jacobian)
            weights_flat = weights_flat * area_correction

        vjps = derivative_info.evaluate_gradient_at_points(
            xyz, normals_xyz, perps1_xyz, perps2_xyz, interpolators
        )
        return np.real(np.sum(vjps * weights_flat * jacobian)).item()

    def _compute_derivative_vertices(
        self,
        derivative_info: DerivativeInfo,
        sim_min: NDArray,
        sim_max: NDArray,
        is_2d: bool = False,
        interpolators: Optional[dict] = None,
    ) -> NDArray:
        """VJP for the vertices of a ``PolySlab``.

        Uses shared sidewall patch collection and batched field evaluation.
        """
        vertices, next_v, edges, basis = self._edge_geometry_arrays()
        dx = derivative_info.adaptive_vjp_spacing()

        # collect patches once
        patch = self._collect_sidewall_patches(
            vertices=vertices,
            next_v=next_v,
            edges=edges,
            basis=basis,
            sim_min=sim_min,
            sim_max=sim_max,
            is_2d=is_2d,
            dx=dx,
        )

        # early return if no patches
        if patch["centers"].shape[0] == 0:
            return np.zeros_like(vertices)

        dz = patch["dz"]
        dz_surf = 1.0 if is_2d else dz / np.cos(self.sidewall_angle)

        # use provided interpolators or create them if not provided
        if interpolators is None:
            interpolators = derivative_info.create_interpolators(
                dtype=config.adjoint.gradient_dtype_float
            )

        # evaluate integrand
        g = derivative_info.evaluate_gradient_at_points(
            patch["centers"], patch["normals"], patch["perps1"], patch["perps2"], interpolators
        )

        # compute area-based weights and weighted vjps
        areas = patch["Ls"] * patch["s_weights"] * dz_surf
        patch_vjps = (g * areas).real

        # distribute to vertices using vectorized accumulation
        normals_2d = np.delete(basis["norm"], self.axis, axis=1)
        edge_idx = patch["edge_indices"]
        s = patch["s_vals"]
        w0 = (1.0 - s) * patch_vjps
        w1 = s * patch_vjps
        edge_norms = normals_2d[edge_idx]

        # Accumulate per-vertex contributions using bincount (O(N_patches))
        num_vertices = vertices.shape[0]
        contrib0 = w0[:, None] * edge_norms  # (n_patches, 2)
        contrib1 = w1[:, None] * edge_norms  # (n_patches, 2)

        idx0 = edge_idx
        idx1 = (edge_idx + 1) % num_vertices

        v0x = np.bincount(idx0, weights=contrib0[:, 0], minlength=num_vertices)
        v0y = np.bincount(idx0, weights=contrib0[:, 1], minlength=num_vertices)
        v1x = np.bincount(idx1, weights=contrib1[:, 0], minlength=num_vertices)
        v1y = np.bincount(idx1, weights=contrib1[:, 1], minlength=num_vertices)

        vjp_per_vertex = np.stack((v0x + v1x, v0y + v1y), axis=1)
        return vjp_per_vertex

    def _edge_geometry_arrays(
        self, dtype: np.dtype = config.adjoint.gradient_dtype_float
    ) -> tuple[NDArray, NDArray, NDArray, dict[str, NDArray]]:
        """Return (vertices, next_v, edges, basis) arrays for sidewall edge geometry."""
        vertices = np.asarray(self.vertices, dtype=dtype)
        next_v = np.roll(vertices, -1, axis=0)
        edges = next_v - vertices
        basis = self.edge_basis_vectors(edges)
        return vertices, next_v, edges, basis

    def edge_basis_vectors(
        self,
        edges: NDArray,  # (N, 2)
    ) -> dict[str, NDArray]:  # (N, 3)
        """Normalized basis vectors for ``normal`` direction, ``slab`` tangent direction and ``edge``."""

        # ensure edges have consistent dtype
        edges = edges.astype(config.adjoint.gradient_dtype_float, copy=False)

        num_vertices, _ = edges.shape
        zeros = np.zeros(num_vertices, dtype=config.adjoint.gradient_dtype_float)
        ones = np.ones(num_vertices, dtype=config.adjoint.gradient_dtype_float)

        # normalized vectors along edges
        edges_norm_in_plane = self.normalize_vect(edges)
        edges_norm_xyz = self.unpop_axis_vect(zeros, edges_norm_in_plane)

        # normalized vectors from base of edges to tops of edges
        cos_angle = np.cos(self.sidewall_angle)
        sin_angle = np.sin(self.sidewall_angle)
        slabs_axis_components = cos_angle * ones

        # create axis_norm as array directly to avoid tuple->array conversion in np.cross
        axis_norm = np.zeros(3, dtype=config.adjoint.gradient_dtype_float)
        axis_norm[self.axis] = 1.0
        slab_normal_xyz = -sin_angle * np.cross(edges_norm_xyz, axis_norm)
        _, slab_normal_in_plane = self.pop_axis_vect(slab_normal_xyz)
        slabs_norm_xyz = self.unpop_axis_vect(slabs_axis_components, slab_normal_in_plane)

        # normalized vectors pointing in normal direction of edge
        # cross yields inward normal when the extrusion axis is y, so negate once for axis==1
        sign = (-1 if self.axis == 1 else 1) * (-1 if not self.is_ccw else 1)
        normals_norm_xyz = sign * np.cross(edges_norm_xyz, slabs_norm_xyz)

        return {
            "norm": normals_norm_xyz,
            "perp1": edges_norm_xyz,
            "perp2": slabs_norm_xyz,
        }

    def unpop_axis_vect(self, ax_coords: NDArray, plane_coords: NDArray) -> NDArray:
        """Combine coordinate along axis with coordinates on the plane tangent to the axis.

        ax_coords.shape == [N]
        plane_coords.shape == [N, 2]
        return shape == [N, 3]
        """
        n_pts = ax_coords.shape[0]
        arr_xyz = np.zeros((n_pts, 3), dtype=ax_coords.dtype)

        plane_axes = [i for i in range(3) if i != self.axis]

        arr_xyz[:, self.axis] = ax_coords
        arr_xyz[:, plane_axes] = plane_coords

        return arr_xyz

    def pop_axis_vect(self, coord: NDArray) -> tuple[NDArray, tuple[NDArray, NDArray]]:
        """Combine coordinate along axis with coordinates on the plane tangent to the axis.

        coord.shape == [N, 3]
        return shape == ([N], [N, 2]
        """

        arr_axis, arrs_plane = self.pop_axis(coord.T, axis=self.axis)
        arrs_plane = np.array(arrs_plane).T

        return arr_axis, arrs_plane

    @staticmethod
    def normalize_vect(arr: NDArray) -> NDArray:
        """normalize an array shaped (N, d) along the `d` axis and return (N, 1)."""
        norm = np.linalg.norm(arr, axis=-1, keepdims=True)
        norm = np.where(norm == 0, 1, norm)
        return arr / norm

    def translated(self, x: float, y: float, z: float) -> PolySlab:
        """Return a translated copy of this geometry.

        Parameters
        ----------
        x : float
            Translation along x.
        y : float
            Translation along y.
        z : float
            Translation along z.

        Returns
        -------
        :class:`PolySlab`
            Translated copy of this ``PolySlab``.
        """

        t_normal, t_plane = self.pop_axis((x, y, z), axis=self.axis)
        translated_vertices = np.array(self.vertices) + np.array(t_plane)[None, :]
        translated_slab_bounds = (self.slab_bounds[0] + t_normal, self.slab_bounds[1] + t_normal)
        return self.updated_copy(vertices=translated_vertices, slab_bounds=translated_slab_bounds)

    def scaled(self, x: float = 1.0, y: float = 1.0, z: float = 1.0) -> PolySlab:
        """Return a scaled copy of this geometry.

        Parameters
        ----------
        x : float = 1.0
            Scaling factor along x.
        y : float = 1.0
            Scaling factor along y.
        z : float = 1.0
            Scaling factor along z.

        Returns
        -------
        :class:`Geometry`
            Scaled copy of this geometry.
        """
        scale_normal, scale_in_plane = self.pop_axis((x, y, z), axis=self.axis)
        scaled_vertices = self.vertices * np.array(scale_in_plane)
        scaled_slab_bounds = tuple(scale_normal * bound for bound in self.slab_bounds)
        return self.updated_copy(vertices=scaled_vertices, slab_bounds=scaled_slab_bounds)

    def rotated(self, angle: float, axis: Union[Axis, Coordinate]) -> PolySlab:
        """Return a rotated copy of this geometry.

        Parameters
        ----------
        angle : float
            Rotation angle (in radians).
        axis : Union[int, Tuple[float, float, float]]
            Axis of rotation: 0, 1, or 2 for x, y, and z, respectively, or a 3D vector.

        Returns
        -------
        :class:`PolySlab`
            Rotated copy of this ``PolySlab``.
        """
        _, plane_axs = self.pop_axis([0, 1, 2], self.axis)
        if (isinstance(axis, int) and axis == self.axis) or (
            isinstance(axis, tuple) and all(axis[ax] == 0 for ax in plane_axs)
        ):
            verts_3d = np.zeros((3, self.vertices.shape[0]))
            verts_3d[plane_axs[0], :] = self.vertices[:, 0]
            verts_3d[plane_axs[1], :] = self.vertices[:, 1]
            rotation = RotationAroundAxis(angle=angle, axis=axis)
            rotated_vertices = rotation.rotate_vector(verts_3d)
            rotated_vertices = rotated_vertices[plane_axs, :].T
            return self.updated_copy(vertices=rotated_vertices)

        return super().rotated(angle=angle, axis=axis)

    def reflected(self, normal: Coordinate) -> PolySlab:
        """Return a reflected copy of this geometry.

            Parameters
            ----------
            normal : Tuple[float, float, float]
                The 3D normal vector of the plane of reflection. The plane is assumed
                    to pass through the origin (0,0,0).

            Returns
            -------
        -------
        :class:`PolySlab`
            Reflected copy of this ``PolySlab``.
        """
        if math.isclose(normal[self.axis], 0):
            _, plane_axs = self.pop_axis((0, 1, 2), self.axis)
            verts_3d = np.zeros((3, self.vertices.shape[0]))
            verts_3d[plane_axs[0], :] = self.vertices[:, 0]
            verts_3d[plane_axs[1], :] = self.vertices[:, 1]
            reflection = ReflectionFromPlane(normal=normal)
            reflected_vertices = reflection.reflect_vector(verts_3d)
            reflected_vertices = reflected_vertices[plane_axs, :].T
            return self.updated_copy(vertices=reflected_vertices)

        return super().reflected(normal=normal)


class ComplexPolySlabBase(PolySlab):
    """Interface for dividing a complex polyslab where self-intersecting polygon can
    occur during extrusion. This class should not be used directly. Use instead
    :class:`plugins.polyslab.ComplexPolySlab`."""

    @pydantic.validator("vertices", always=True)
    def no_self_intersecting_polygon_during_extrusion(
        cls, val: ArrayFloat2D, values: dict[str, Any]
    ) -> ArrayFloat2D:
        """Turn off the validation for this class."""
        return val

    @classmethod
    def from_gds(
        cls,
        gds_cell: Cell,
        axis: Axis,
        slab_bounds: tuple[float, float],
        gds_layer: int,
        gds_dtype: Optional[int] = None,
        gds_scale: pydantic.PositiveFloat = 1.0,
        dilation: float = 0.0,
        sidewall_angle: float = 0,
        reference_plane: PlanePosition = "middle",
    ) -> list[PolySlab]:
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
    def sub_polyslabs(self) -> list[PolySlab]:
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
        if math.isclose(self.sidewall_angle, 0):
            return [PolySlab.parse_obj(sub_polyslab_dict)]

        sub_polyslab_dict.update({"dilation": 0})  # dilation accounted in setup
        # initialize offset distance
        offset_distance = 0

        for dist_val in self._dilation_length:
            dist_now = 0.0
            vertices_now = self.reference_polygon

            # constructing sub-polyslabs until reaching the base/top
            while not math.isclose(dist_now, dist_val):
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
                    {
                        "slab_bounds": tuple(slab_bounds),
                        "vertices": vertices_now,
                        "reference_plane": reference_plane,
                    }
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
    def _dilation_length(self) -> list[float]:
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
        self,
        normal: Coordinate,
        origin: Coordinate,
        to_2D: MatrixReal4x4,
        cleanup: bool = True,
        quad_segs: Optional[int] = None,
    ) -> list[Shapely]:
        """Return a list of shapely geometries at the plane specified by normal and origin.

        Parameters
        ----------
        normal : Coordinate
            Vector defining the normal direction to the plane.
        origin : Coordinate
            Vector defining the plane origin.
        to_2D : MatrixReal4x4
            Transformation matrix to apply to resulting shapes.
        cleanup : bool = True
            If True, removes extremely small features from each polygon's boundary.
        quad_segs : Optional[int] = None
            Number of segments used to discretize circular shapes. Not used for PolySlab.

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
                    for shape in polyslab.intersections_tilted_plane(
                        normal, origin, to_2D, cleanup=cleanup, quad_segs=quad_segs
                    )
                ]
            )
        ]
