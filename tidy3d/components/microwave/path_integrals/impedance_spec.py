"""Specification for impedance computation in transmission lines and waveguides."""

from __future__ import annotations

from itertools import chain
from math import isclose
from typing import Optional, Union

import pydantic.v1 as pd
import shapely
from shapely.geometry import LineString, Polygon

from tidy3d.components.base import Tidy3dBaseModel, cached_property
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
from tidy3d.components.medium import LossyMetalMedium, Medium
from tidy3d.components.microwave.path_integrals.current_spec import (
    CompositeCurrentIntegralSpec,
    CurrentIntegralAxisAlignedSpec,
)
from tidy3d.components.microwave.path_integrals.types import (
    CurrentPathSpecTypes,
    VoltagePathSpecTypes,
)
from tidy3d.components.types import Axis, Bound, Coordinate, Shapely, Symmetry
from tidy3d.components.validators import assert_plane
from tidy3d.exceptions import SetupError, ValidationError


class PathSpecGenerator(Box):
    """Generates current integration path specifications based on structure geometry.

    This class analyzes the geometry of conductors in a simulation cross-section to determine
    appropriate paths for computing current line integrals. These paths are typically used
    for setting up a :class:`.MicrowaveModeSpec` automatically.
    """

    _plane_validator = assert_plane()

    @cached_property
    def normal_axis(self) -> Axis:
        """Axis normal to the monitor's plane."""
        return self.size.index(0.0)

    field_data_colocated: bool = pd.Field(
        False,
        title="Field Data Colocated",
        description="Whether field data is colocated with grid points. When True, integration paths "
        "are placed with additional margin to avoid interpolated field values near conductor surfaces.",
    )

    @cached_property
    def _snap_spec(self) -> SnappingSpec:
        """Creates snapping specification for grid alignment."""
        behavior = [SnapBehavior.StrictExpand] * 3
        location = [SnapLocation.Center] * 3
        behavior[self.normal_axis] = SnapBehavior.Off
        # To avoid interpolated H field near metal surface
        margin = (2, 2, 2) if self.field_data_colocated else (0, 0, 0)
        return SnappingSpec(location=location, behavior=behavior, margin=margin)

    @cached_property
    def _create_port_boundary(self) -> shapely.LineString:
        """Creates a Shapely LineString representing the port boundary."""
        _, min_b = Geometry.pop_axis(self.bounds[0], self.normal_axis)
        _, max_b = Geometry.pop_axis(self.bounds[1], self.normal_axis)
        port = Geometry.make_shapely_box(min_b[0], min_b[1], max_b[0], max_b[1])
        return shapely.LineString(port.exterior)

    def _get_mode_symmetry(
        self, sim_box: Box, sym_symmetry: tuple[Symmetry, Symmetry, Symmetry]
    ) -> tuple[Symmetry, Symmetry, Symmetry]:
        """Get the mode symmetry, considering the mode plane, the simulation box and the simulation symmetry."""
        mode_symmetry = list(sym_symmetry)
        for dim in range(3):
            if sim_box.center[dim] != self.center[dim] or self.size[dim] == 0:
                mode_symmetry[dim] = 0

        return mode_symmetry

    def _get_mode_limits(
        self, sim_box: Box, mode_symmetry: tuple[Symmetry, Symmetry, Symmetry]
    ) -> Bound:
        """Restrict mode plane bounds to the simulation bounds taking into account symmetry conditions."""
        # Restrict mode plane to the simulation size, if it is smaller
        min_b, max_b = bounds_intersection(self.bounds, sim_box.bounds)

        min_b_2d_list = list(min_b)
        for dim in range(3):
            if mode_symmetry[dim] != 0:
                min_b_2d_list[dim] = self.center[dim]

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

    def _get_isolated_conductors_as_shapely(
        self,
        plane: Box,
        structures: list,  # list[Structure] - avoiding import for circular dependency
    ) -> list[Shapely]:
        """Find and merge all conductor structures that intersect the given plane.

        Parameters
        ----------
        plane : Box
            The plane to check for conductor intersections
        structures : list
            List of all simulation structures to analyze

        Returns
        -------
        list[Shapely]
            List of merged conductor geometries as Shapely Polygons and LineStrings
            that intersect with the given plane
        """

        def is_conductor(med: Medium) -> bool:
            return med.is_pec or isinstance(med, LossyMetalMedium)

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

    def create_current_path_specs(
        self,
        structures: list,  # list[Structure] - avoiding import for circular dependency
        grid: Grid,
        symmetry: tuple[Symmetry, Symmetry, Symmetry],
        sim_box: Box,
    ) -> tuple[CompositeCurrentIntegralSpec, list[Shapely]]:
        """Creates path specifications for path integrals that encompass each isolated conductor
        in the mode plane.

        This method identifies isolated conductor geometries in the given plane and creates
        current paths that enclose each conductor. The paths are snapped to the simulation grid
        to ensure alignment with field data.

        Parameters
        ----------
        structures : list
            List of structures in the simulation.
        grid : Grid
            Simulation grid for snapping paths.
        symmetry : tuple[Symmetry, Symmetry, Symmetry]
            Symmetry conditions for the simulation in (x, y, z) directions.
        sim_box : Box
            Simulation domain box used for boundary conditions.

        Returns
        -------
        tuple[CompositeCurrentIntegralSpec, list[Shapely]]
            Composite path specification and list of merged conductor geometries.
        """

        def bounding_box_from_shapely(geom: Shapely) -> Box:
            """Helper to convert the shapely geometry bounds to a Box."""
            bounds = geom.bounds
            normal_center = self.center[self.normal_axis]
            rmin = Geometry.unpop_axis(normal_center, (bounds[0], bounds[1]), self.normal_axis)
            rmax = Geometry.unpop_axis(normal_center, (bounds[2], bounds[3]), self.normal_axis)
            return Box.from_bounds(rmin, rmax)

        mode_symmetry_3d = self._get_mode_symmetry(sim_box, symmetry)
        min_b_3d, max_b_3d = self._get_mode_limits(sim_box, mode_symmetry_3d)

        intersection_plane = Box.from_bounds(min_b_3d, max_b_3d)
        conductor_polygons = self._get_isolated_conductors_as_shapely(
            intersection_plane, structures
        )

        conductor_polygons = self._filter_conductors_touching_sim_bounds(
            (min_b_3d, max_b_3d), mode_symmetry_3d, self.normal_axis, conductor_polygons
        )

        if len(conductor_polygons) < 1:
            raise ValidationError(
                "No valid isolated conductors were found in the mode plane. Please ensure that a 'Structure' "
                "with a medium of type 'PEC' or 'LossyMetalMedium' intersects the mode plane and is not touching "
                "the boundaries of the mode plane."
            )

        # Get desired snapping behavior of box enclosed conductors.
        # Ideally, just large enough to coincide with the H field positions outside of the conductor.
        # So a half grid cell, when the metal boundary is coincident with grid boundaries.
        snap_spec = self._snap_spec

        current_integral_specs = []
        for shape in conductor_polygons:
            box = bounding_box_from_shapely(shape)
            boxes = self._apply_symmetry(symmetry, sim_box.center, self.normal_axis, box)
            for box in boxes:
                box_snapped = snap_box_to_grid(grid, box, snap_spec)
                path_spec = self._create_path_spec(box_snapped)
                current_integral_specs.append(path_spec)

        for path_spec in current_integral_specs:
            if self._check_path_intersects_with_conductors(conductor_polygons, path_spec):
                raise ValidationError(
                    "Failed to automatically generate path specification. "
                    "Please create a github issue so that the problem can be investigated. "
                    "In the meantime, please provide an explicit path specification for this structure."
                )

        path_spec = CompositeCurrentIntegralSpec(
            center=self.center,
            size=self.size,
            path_specs=current_integral_specs,
            sum_spec="split",
        )

        result = (path_spec, conductor_polygons)
        return result

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


class AutoImpedanceSpec(Tidy3dBaseModel):
    """Specification for fully automatic transmission line impedance computation.

    This specification automatically calculates impedance by calculating the current associated
    with each conductor that intersects the mode plane.
    """


class CustomImpedanceSpec(Tidy3dBaseModel):
    """Specification for custom transmission line voltages and currents in mode solvers.

    The :class:`.CustomImpedanceSpec` class specifies how quantities related to transmission line
    modes are computed. For example, it defines the paths for line integrals, which are used to
    compute voltage, current, and characteristic impedance of the transmission line.

    Users may supply their own voltage and current path specifications to control where these integrals
    are evaluated. If neither voltage nor current specifications are provided, an automatic choice of
    paths will be made based on the simulation geometry and context.
    """

    voltage_spec: Optional[VoltagePathSpecTypes] = pd.Field(
        None,
        title="Voltage Integration Path",
        description="Path specification for computing the voltage associated with each mode. "
        "The number of path specifications should equal the 'num_modes' field "
        "in the 'ModeSpec'.",
    )

    current_spec: Optional[CurrentPathSpecTypes] = pd.Field(
        None,
        title="Current Integration Path",
        description="Path specification for computing the current associated with each mode. "
        "The number of path specifications should equal the 'num_modes' field "
        "in the 'ModeSpec'.",
    )

    @pd.validator("current_spec", always=True)
    def check_path_spec_combinations(cls, val, values):
        """In order to define voltage/current/impedance, either a voltage or current path specification
        must be provided.
        """

        voltage_spec = values["voltage_spec"]
        if val is None and voltage_spec is None:
            raise SetupError(
                "Not a valid 'CustomImpedanceSpec', the 'voltage_spec' and 'current_spec' cannot both be 'None'."
            )
        return val


ImpedanceSpecTypes = Union[AutoImpedanceSpec, CustomImpedanceSpec]
