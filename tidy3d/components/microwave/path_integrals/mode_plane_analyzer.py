"""Helper class for analyzing conductor geometry in a mode plane."""

from __future__ import annotations

from itertools import chain
from math import isclose

import pydantic.v1 as pd
import shapely
from shapely.geometry import LineString, Polygon

from tidy3d.components.base import cached_property
from tidy3d.components.geometry.base import Box, Geometry
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
from tidy3d.components.structure import Structure
from tidy3d.components.types import Axis, Bound, Coordinate, Shapely, Symmetry
from tidy3d.components.validators import assert_plane
from tidy3d.exceptions import SetupError


class ModePlaneAnalyzer(Box):
    """Analyzes conductor geometry intersecting a mode plane.

    Notes
    -----
        This class analyzes the geometry of conductors in a simulation cross-section and is for internal use.
    """

    _plane_validator = assert_plane()

    field_data_colocated: bool = pd.Field(
        False,
        title="Field Data Colocated",
        description="Whether field data is colocated with grid points. When 'True', bounding boxes "
        "are placed with additional margin to avoid interpolated field values near conductor surfaces.",
    )

    @cached_property
    def _snap_spec(self) -> SnappingSpec:
        """Creates snapping specification for bounding boxes."""
        behavior = [SnapBehavior.StrictExpand] * 3
        location = [SnapLocation.Center] * 3
        behavior[self._normal_axis] = SnapBehavior.Off
        # To avoid interpolated H field near metal surface
        margin = (2, 2, 2) if self.field_data_colocated else (0, 0, 0)
        return SnappingSpec(location=location, behavior=behavior, margin=margin)

    def _get_mode_symmetry(
        self, sim_box: Box, sym_symmetry: tuple[Symmetry, Symmetry, Symmetry]
    ) -> tuple[Symmetry, Symmetry, Symmetry]:
        """Get the mode symmetry, considering the simulation box and the simulation symmetry."""
        mode_symmetry = list(sym_symmetry)
        for dim in range(3):
            if sim_box.center[dim] != self.center[dim] or self.size[dim] == 0:
                mode_symmetry[dim] = 0
        return mode_symmetry

    def _get_mode_limits(
        self, sim_grid: Grid, mode_symmetry: tuple[Symmetry, Symmetry, Symmetry]
    ) -> Bound:
        """Restrict mode plane bounds to the final grid positions taking into account symmetry conditions.

        Mode profiles are calculated on a grid which is expanded from the monitor size to the closest grid boundaries.
        """
        behavior = [SnapBehavior.StrictExpand] * 3
        location = [SnapLocation.Boundary] * 3
        behavior[self._normal_axis] = SnapBehavior.Off
        margin = (1, 1, 1)
        snap_spec = SnappingSpec(location=location, behavior=behavior, margin=margin)
        mode_box = snap_box_to_grid(sim_grid, self.geometry, snap_spec=snap_spec)
        min_b, max_b = mode_box.bounds
        min_b_2d_list = list(min_b)
        for dim in range(3):
            if mode_symmetry[dim] != 0:
                min_b_2d_list[dim] = self.center[dim]

        return (tuple(min_b_2d_list), max_b)

    def _get_isolated_conductors_as_shapely(
        self,
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

    def _filter_conductors_touching_sim_bounds(
        self,
        mode_limits: Bound,
        mode_symmetry_3d: tuple[Symmetry, Symmetry, Symmetry],
        conductor_polygons: list[Shapely],
    ) -> list[Shapely]:
        """Filters a list of Shapely geometries representing conductors in the mode plane. PEC-type boundary
        conditions act like a short to ground, so any structures touching a PEC boundary can be ignored
        from the current calculation.

        Parameters
        ----------
        mode_limits : Bound
            The locations of the boundary conditions.
        mode_symmetry_3d : tuple[Symmetry, Symmetry, Symmetry]
            Symmetry settings for the mode solver plane.
        conductor_polygons : list[Shapely]
            List of shapely geometries (polygons/lines) representing the exterior of conducting
            structures in the mode plane.

        Returns
        -------
        list[Shapely]
            The filtered list of shapely geometries, where structures "shorted" to PEC boundaries have been removed.
        """
        min_b_3d, max_b_3d = mode_limits[0], mode_limits[1]
        _, mode_symmetry = Geometry.pop_axis(mode_symmetry_3d, self._normal_axis)
        _, min_b = Geometry.pop_axis(min_b_3d, self._normal_axis)
        _, max_b = Geometry.pop_axis(max_b_3d, self._normal_axis)

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

    def get_conductor_bounding_boxes(
        self,
        structures: list[Structure],
        grid: Grid,
        symmetry: tuple[Symmetry, Symmetry, Symmetry],
        sim_box: Box,
    ) -> tuple[list[Box], list[Shapely]]:
        """Returns bounding boxes that encompass each isolated conductor
        in the mode plane.

        This method identifies isolated conductor geometries in the given plane.
        The paths are snapped to the simulation grid
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
        tuple[list[Box], list[Shapely]]
            Bounding boxes and list of merged conductor geometries.
        """

        def bounding_box_from_shapely(geom: Shapely) -> Box:
            """Helper to convert the shapely geometry bounds to a Box."""
            bounds = geom.bounds
            normal_center = self.center[self._normal_axis]
            rmin = Geometry.unpop_axis(normal_center, (bounds[0], bounds[1]), self._normal_axis)
            rmax = Geometry.unpop_axis(normal_center, (bounds[2], bounds[3]), self._normal_axis)
            return Box.from_bounds(rmin, rmax)

        mode_symmetry_3d = self._get_mode_symmetry(sim_box, symmetry)
        min_b_3d, max_b_3d = self._get_mode_limits(grid, mode_symmetry_3d)

        intersection_plane = Box.from_bounds(min_b_3d, max_b_3d)
        isolated_conductor_shapely = self._get_isolated_conductors_as_shapely(
            intersection_plane, structures
        )

        filtered_conductor_shapely = self._filter_conductors_touching_sim_bounds(
            (min_b_3d, max_b_3d), mode_symmetry_3d, isolated_conductor_shapely
        )

        if len(filtered_conductor_shapely) < 1:
            raise SetupError(
                "No valid isolated conductors were found in the mode plane. Please ensure that a 'Structure' "
                "with a medium of type 'PEC' or 'LossyMetalMedium' intersects the mode plane and is not touching "
                "the boundaries of the mode plane."
            )

        # Get desired snapping behavior of box enclosed conductors.
        # Ideally, just large enough to coincide with the H field positions outside of the conductor.
        # So a half grid cell, when the metal boundary is coincident with grid boundaries.
        snap_spec = self._snap_spec

        bounding_boxes = []
        for shape in filtered_conductor_shapely:
            box = bounding_box_from_shapely(shape)
            boxes = self._apply_symmetries(symmetry, sim_box.center, box)
            for box in boxes:
                box_snapped = snap_box_to_grid(grid, box, snap_spec)
                bounding_boxes.append(box_snapped)

        # TODO Improve these checks once FXC-4112-PEC-boundary-position-not-respected-by-ModeSolver is merged
        for bounding_box in bounding_boxes:
            if self._check_box_intersects_with_conductors(isolated_conductor_shapely, bounding_box):
                raise SetupError(
                    "Failed to automatically generate path specification because a generated path "
                    "specification was found to intersect with a conductor. There is currently limited "
                    "support for complex conductor geometries, so please provide an explicit current "
                    "path specification through a 'CustomImpedanceSpec'. Alternatively, enforce a "
                    "smaller grid around the conductors in the mode plane, which may resolve the issue."
                )

        # Check that bounding boxes don't extend outside the original mode plane bounds
        mode_plane_min, mode_plane_max = self.bounds
        for bounding_box in bounding_boxes:
            box_min, box_max = bounding_box.bounds
            if any(box_min[i] < mode_plane_min[i] for i in range(3)) or any(
                box_max[i] > mode_plane_max[i] for i in range(3)
            ):
                raise SetupError(
                    "Failed to automatically generate path specification because a generated path "
                    "specification extends outside the mode solving plane bounds. This issue can be fixed "
                    "by enlarging the mode solving plane and ensuring that there is a buffer of at "
                    "least 2 grid cells between the mode solving plane bounds and the nearest conductors."
                    "Alternatively, enforce a smaller grid around the conductors in the mode plane, "
                    "which may resolve the issue."
                )

        return bounding_boxes, filtered_conductor_shapely

    def _check_box_intersects_with_conductors(
        self, shapely_list: list[Shapely], bounding_box: Box
    ) -> bool:
        """Makes sure that a box does not intersect with conductor shapes.

        Parameters
        ----------
        shapely_list : list[Shapely]
            Merged conductor geometries, expected to be polygons or lines for 2D structures.
        bounding_box : Box
            Box corresponding with a future path specification.

        Returns
        -------
        bool: ``True`` if the bounding box intersects with any conductor geometry, ``False`` otherwise.
        """
        min_b, max_b = bounding_box.bounds
        _, min_b = Geometry.pop_axis(min_b, self._normal_axis)
        _, max_b = Geometry.pop_axis(max_b, self._normal_axis)
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
            The box to reflect.
        axis : Axis
            The axis perpendicular to the reflection plane (0,1,2) -> (x,y,z).
        position : float
            Position along the axis where the reflection plane is located.

        Returns
        -------
        Box
            The reflected box.
        """
        new_center = list(box.center)
        new_center[axis] = 2 * position - box.center[axis]
        return box.updated_copy(center=new_center)

    @staticmethod
    def _apply_symmetry_to_box(box: Box, axis: Axis, position: float) -> list[Box]:
        """Applies a single symmetry condition to a box along a specified axis.

        If the box touches the symmetry plane, merges the box with its reflection.
        Otherwise returns both the original and reflected box.

        Parameters
        ----------
        box : Box
            The box that will be reflected.
        axis : Axis
            The axis along which to apply symmetry (0,1,2) -> (x,y,z).
        position : float
            Position of the symmetry plane along the axis.

        Returns
        -------
        list[Box]
            List containing either merged box or original and reflected boxes
        """
        new_box = ModePlaneAnalyzer._reflect_box(box, axis, position)
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

    def _apply_symmetries(
        self,
        symmetry: tuple[Symmetry, Symmetry, Symmetry],
        sim_center: Coordinate,
        box: Box,
    ) -> list[Box]:
        """Applies symmetry conditions to the location of a box. When a symmetry condition is present,
        the box will be reflected. If the reflection is touching the original box, they will be merged.

        Parameters
        ----------
        symmetry : tuple[Symmetry, Symmetry, Symmetry]
            Symmetry conditions for each axis.
        sim_center : Coordinate
            Center coordinates where symmetry planes intersect.
        box : Box
            The box that will be reflected.

        Returns
        -------
        list[Box]
            List of boxes after applying all symmetry operations, so a list with either 1, 2,
            or 4 Box elements
        """
        symmetry = list(symmetry)
        dims = [0, 1, 2]
        symmetry.pop(self._normal_axis)
        dims.pop(self._normal_axis)
        result = [box]
        for dim, sym in zip(dims, symmetry):
            if sym != 0:
                tmp_list = [
                    ModePlaneAnalyzer._apply_symmetry_to_box(box, dim, sim_center[dim])
                    for box in result
                ]
                result = list(chain.from_iterable(tmp_list))
        return result
