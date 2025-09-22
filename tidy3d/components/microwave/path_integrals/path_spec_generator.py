"""Module for automatic determination of voltage and current integration specifications."""

from __future__ import annotations

from itertools import chain
from math import isclose
from typing import Union, get_args

import shapely
from shapely.geometry import (
    LineString,
    Polygon,
)

from tidy3d.components.base import Tidy3dBaseModel
from tidy3d.components.geometry.base import Box, Geometry
from tidy3d.components.geometry.bound_ops import bounds_intersection
from tidy3d.components.geometry.utils import (
    SnapBehavior,
    SnapLocation,
    SnappingSpec,
    flatten_shapely_geometries,
    merging_geometries_on_plane,
    snap_box_to_grid,
)
from tidy3d.components.grid.grid import Grid
from tidy3d.components.medium import LossyMetalMedium, Medium, PECMedium
from tidy3d.components.structure import Structure
from tidy3d.components.types import Axis, Bound, Coordinate, Shapely, Symmetry
from tidy3d.exceptions import ValidationError

from .current_spec import CompositeCurrentIntegralSpec, CurrentIntegralAxisAlignedSpec

ConductorTypes = Union[PECMedium, LossyMetalMedium]


class PathSpecGenerator(Tidy3dBaseModel):
    """Automatically determines current integration paths based on structure geometry.

    This class analyzes the geometry of conductors in a simulation cross-section to determine
    appropriate paths for computing current line integrals. These paths are typically used
    for setting up a :class:`.MicrowaveModeSpec` automatically.

    The paths are chosen by:
    1. Finding and merging conductor surfaces that intersect with the plane
    2. Creating current paths that enclose isolated conductors
    """

    @staticmethod
    def _create_snap_spec(normal_axis: Axis, field_data_colocated: bool) -> SnappingSpec:
        """Creates snapping specification for grid alignment."""
        behavior = [SnapBehavior.StrictExpand] * 3
        location = [SnapLocation.Center] * 3
        behavior[normal_axis] = SnapBehavior.Off
        # To avoid interpolated H field near metal surface
        margin = (2, 2, 2) if field_data_colocated else (0, 0, 0)
        return SnappingSpec(location=location, behavior=behavior, margin=margin)

    @staticmethod
    def _create_port_boundary(mode_plane: Box, normal_axis: Axis) -> shapely.LineString:
        """Creates a Shapely LineString representing the port boundary."""
        _, min_b = Geometry.pop_axis(mode_plane.bounds[0], normal_axis)
        _, max_b = Geometry.pop_axis(mode_plane.bounds[1], normal_axis)
        port = Geometry.make_shapely_box(min_b[0], min_b[1], max_b[0], max_b[1])
        return shapely.LineString(port.exterior)

    @staticmethod
    def _get_mode_symmetry(
        mode_plane: Box, sim_box: Box, sym_symmetry: tuple[Symmetry, Symmetry, Symmetry]
    ) -> tuple[Symmetry, Symmetry, Symmetry]:
        """Get the mode symmetry, considering the mode plane, the simulation box and the simulation symmetry."""
        mode_symmetry = list(sym_symmetry)
        for dim in range(3):
            if sim_box.center[dim] != mode_plane.center[dim] or mode_plane.size[dim] == 0:
                mode_symmetry[dim] = 0

        return mode_symmetry

    @staticmethod
    def _get_mode_limits(
        mode_plane: Box, sim_box: Box, mode_symmetry: tuple[Symmetry, Symmetry, Symmetry]
    ) -> Bound:
        """Restrict mode plane bounds to the simulation bounds taking into account symmetry conditions.s"""
        # Restrict mode plane to the simulation size, if it is smaller
        min_b, max_b = bounds_intersection(mode_plane.bounds, sim_box.bounds)

        min_b_2d_list = list(min_b)
        for dim in range(3):
            if mode_symmetry[dim] != 0:
                min_b_2d_list[dim] = mode_plane.center[dim]

        return (tuple(min_b_2d_list), max_b)

    @staticmethod
    def _create_path_spec(box_snapped: Box) -> CurrentIntegralAxisAlignedSpec:
        """Creates a current integral specification from a snapped box."""
        return CurrentIntegralAxisAlignedSpec(
            center=box_snapped.center,
            size=box_snapped.size,
            sign="+",
            extrapolate_to_endpoints=False,
            snap_contour_to_grid=True,
        )

    @staticmethod
    def _get_isolated_conductors_as_shapely(
        plane: Box,
        structures: list[Structure],
    ) -> list[Shapely]:
        """Find and merge all conductor structures that intersect the given plane.

        Parameters
        ----------
        plane : Box
            The plane to check for conductor intersections
        structures : list[Structure]
            List of all simulation structures to analyze

        Returns
        -------
        list[Shapely]
            List of merged conductor geometries as Shapely Polygons and LineStrings
            that intersect with the given plane
        """

        def is_conductor(med: Medium) -> bool:
            union_types = get_args(ConductorTypes)
            return med.is_pec or isinstance(med, union_types)

        geometry_list = [structure.geometry for structure in structures]
        # For metal, we don't distinguish between LossyMetal and PEC,
        # so they'll be merged to PEC. Other materials are considered as dielectric.
        prop_list = [is_conductor(structure.medium) for structure in structures]
        # merge geometries
        geos = merging_geometries_on_plane(geometry_list, plane, prop_list)
        conductor_geos = [item[1] for item in geos if item[0]]
        shapely_list = flatten_shapely_geometries(conductor_geos, keep_types=(Polygon, LineString))
        return shapely_list

    @staticmethod
    def _filter_conductors_touching_sim_bounds(
        mode_limits: Bound,
        mode_symmetry_3d: tuple[Symmetry, Symmetry, Symmetry],
        normal_axis: Axis,
        conductor_polygons: list[Shapely],
    ) -> list[Shapely]:
        min_b_3d, max_b_3d = mode_limits[0], mode_limits[1]
        _, mode_symmetry = Geometry.pop_axis(mode_symmetry_3d, normal_axis)
        _, min_b = Geometry.pop_axis(min_b_3d, normal_axis)
        _, max_b = Geometry.pop_axis(max_b_3d, normal_axis)

        # Add top, right, left, bottom
        shapely_pec_bounds = [
            shapely.LineString([(min_b[0], max_b[1]), (max_b[0], max_b[1])]),
            shapely.LineString([(max_b[0], min_b[1]), (max_b[0], max_b[1])]),
            shapely.LineString([(min_b[0], min_b[1]), (min_b[0], max_b[1])]),
            shapely.LineString([(min_b[0], min_b[1]), (max_b[0], min_b[1])]),
        ]

        # If bottom bound is PMC remove
        if mode_symmetry[1] == 1:
            shapely_pec_bounds.pop(3)

        # If left bound is PMC remove
        if mode_symmetry[0] == 1:
            shapely_pec_bounds.pop(2)

        ml_pec_bounds = shapely.MultiLineString(shapely_pec_bounds)
        return [shape for shape in conductor_polygons if not ml_pec_bounds.intersects(shape)]

    @staticmethod
    def create_current_path_specs(
        mode_plane: Box,
        structures: list[Structure],
        grid: Grid,
        symmetry: tuple[Symmetry, Symmetry, Symmetry],
        sim_box: Box,
        field_data_colocated: bool = False,
    ) -> tuple[CompositeCurrentIntegralSpec, list[Shapely]]:
        """Creates path specifications for path integrals that encompass each isolated conductor
        in the mode plane.

        This method identifies isolated conductor geometries in the given plane and creates
        current paths that enclose each conductor. The paths are snapped to the simulation grid
        to ensure alignment with field data.

        Parameters
        ----------
        mode_plane : Box
            The cross-sectional plane where current paths are determined.
        structures : list[Structure]
            List of structures in the simulation.
        grid : Grid
            Simulation grid for snapping paths.
        symmetry : tuple[Symmetry, Symmetry, Symmetry]
            Symmetry conditions for the simulation in (x, y, z) directions.
        sim_box : Box
            Simulation domain box used for boundary conditions.
        field_data_colocated : bool
            Whether field data is colocated with grid points.

        Returns
        -------
            tuple[CompositeCurrentIntegralSpec, list[Shapely]]
                Composite path specification and list of merged conductor geometries.
        """

        normal_axis = mode_plane.size.index(0.0)

        def bounding_box_from_shapely(
            geom: Shapely, normal_axis: Axis, normal_center: float
        ) -> Box:
            """Helper to convert the shapely geometry bounds to a Box."""
            bounds = geom.bounds
            rmin = Geometry.unpop_axis(normal_center, (bounds[0], bounds[1]), normal_axis)
            rmax = Geometry.unpop_axis(normal_center, (bounds[2], bounds[3]), normal_axis)
            return Box.from_bounds(rmin, rmax)

        mode_symmetry_3d = PathSpecGenerator._get_mode_symmetry(mode_plane, sim_box, symmetry)
        min_b_3d, max_b_3d = PathSpecGenerator._get_mode_limits(
            mode_plane, sim_box, mode_symmetry_3d
        )

        intersection_plane = Box.from_bounds(min_b_3d, max_b_3d)
        conductor_polygons = PathSpecGenerator._get_isolated_conductors_as_shapely(
            intersection_plane, structures
        )

        conductor_polygons = PathSpecGenerator._filter_conductors_touching_sim_bounds(
            (min_b_3d, max_b_3d), mode_symmetry_3d, normal_axis, conductor_polygons
        )

        if len(conductor_polygons) < 1:
            expected_types = ", ".join(t.__name__ for t in get_args(ConductorTypes))
            raise ValidationError(
                "No valid isolated conductors were found in the mode plane. Please ensure that a 'Structure' "
                f"with a medium of type {expected_types} intersects the mode plane and is not touching "
                "the boundaries of the mode plane."
            )

        # Get desired snapping behavior of box enclosed conductors.
        # Ideally, just large enough to coincide with the H field positions outside of the conductor.
        # So a half grid cell, when the metal boundary is coincident with grid boundaries.
        snap_spec = PathSpecGenerator._create_snap_spec(normal_axis, field_data_colocated)

        # shapely_port_boundary = PathSpecGenerator._create_port_boundary(mode_plane, normal_axis)

        current_integral_specs = []
        for shape in conductor_polygons:
            box = bounding_box_from_shapely(shape, normal_axis, mode_plane.center[normal_axis])
            boxes = PathSpecGenerator._apply_symmetry(symmetry, sim_box.center, normal_axis, box)
            for box in boxes:
                box_snapped = snap_box_to_grid(grid, box, snap_spec)
                path_spec = PathSpecGenerator._create_path_spec(box_snapped)
                current_integral_specs.append(path_spec)

        for path_spec in current_integral_specs:
            if PathSpecGenerator._check_path_intersects_with_conductors(
                conductor_polygons, path_spec
            ):
                raise ValidationError(
                    "Failed to automatically generate path specification. "
                    "Please create a github issue so that the problem can be investigated. "
                    "In the meantime, please provide an explicit path specification for this structure."
                )

        path_spec = CompositeCurrentIntegralSpec(
            center=mode_plane.center,
            size=mode_plane.size,
            path_specs=current_integral_specs,
            sum_spec="split",
        )
        return path_spec, conductor_polygons

    @staticmethod
    def _check_path_intersects_with_conductors(shapely_list: Shapely, path: Box) -> bool:
        """Makes sure the path of the integral does not intersect with conductor shapes.

        Parameters
        ----------
        shapely_list : Shapely
            Merged conductor geometries, expected to be Polygons or LineStrings.
        path : Box
            Box corresponding with path specification.

        Returns
        -------
            bool: True if the path intersects with any conductor geometry, False otherwise.
        """
        normal_axis = path.size.index(0.0)
        min_b, max_b = path.bounds
        _, min_b = Geometry.pop_axis(min_b, normal_axis)
        _, max_b = Geometry.pop_axis(max_b, normal_axis)
        path_shapely = shapely.box(min_b[0], min_b[1], max_b[0], max_b[1])
        for shapely_geo in shapely_list:
            if path_shapely.intersects(shapely_geo) and not path_shapely.contains(shapely_geo):
                return True
        return False

    @staticmethod
    def _reflect_box(box: Box, axis: Axis, position: float) -> Box:
        """Reflects a box across a plane perpendicular to the given axis at the specified position.

        Parameters
        ----------
        box : Box
            The box to reflect
        axis : Axis
            The axis perpendicular to the reflection plane (0,1,2) -> (x,y,z)
        position : float
            Position along the axis where the reflection plane is located

        Returns
        -------
        Box
            The reflected box
        """
        new_center = list(box.center)
        new_center[axis] = 2 * position - box.center[axis]
        return box.updated_copy(center=new_center)

    @staticmethod
    def _apply_symmetry_to_box(box: Box, axis: Axis, position: float) -> list[Box]:
        """Applies symmetry operation to a box along specified axis.

        If the box touches the symmetry plane, merges the box with its reflection.
        Otherwise returns both the original and reflected box.

        Parameters
        ----------
        box : Box
            The box to apply symmetry to
        axis : Axis
            The axis along which to apply symmetry (0,1,2) -> (x,y,z)
        position : float
            Position of the symmetry plane along the axis

        Returns
        -------
        list[Box]
            List containing either merged box or original and reflected boxes
        """
        new_box = PathSpecGenerator._reflect_box(box, axis, position)
        if isclose(new_box.bounds[0][axis], box.bounds[1][axis]) or isclose(
            new_box.bounds[1][axis], box.bounds[0][axis]
        ):
            new_size = list(box.size)
            new_size[axis] = 2 * box.size[axis]
            new_center = list(box.center)
            new_center[axis] = position
            new_box = Box(size=new_size, center=new_center)
            return [new_box]
        return [box, new_box]

    @staticmethod
    def _apply_symmetry(
        symmetry: tuple[Symmetry, Symmetry, Symmetry],
        sim_center: Coordinate,
        normal_axis: Axis,
        box: Box,
    ) -> list[Box]:
        """Applies multiple symmetry operations to a box.

        Parameters
        ----------
        symmetry : tuple[Symmetry, Symmetry, Symmetry]
            Symmetry conditions for each axis
        sim_center : Coordinate
            Center coordinates where symmetry planes intersect
        normal_axis : Axis
            Axis normal to the plane of interest
        box : Box
            The box to apply symmetries to

        Returns
        -------
        list[Box]
            List of boxes after applying all symmetry operations
        """
        symmetry = list(symmetry)
        dims = [0, 1, 2]
        symmetry.pop(normal_axis)
        dims.pop(normal_axis)
        result = [box]
        for dim, sym in zip(dims, symmetry):
            if sym != 0:
                tmp_list = [
                    PathSpecGenerator._apply_symmetry_to_box(box, dim, sim_center[dim])
                    for box in result
                ]
                result = list(chain.from_iterable(tmp_list))
        return result
