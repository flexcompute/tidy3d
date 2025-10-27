"""Helper class for analyzing conductor geometry in a mode plane."""

from __future__ import annotations

from itertools import chain
from math import isclose

import pydantic.v1 as pd
import shapely
from shapely.geometry import LineString, Point, Polygon

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
from tidy3d.components.microwave.mode_spec import TerminalSpec
from tidy3d.components.structure import Structure
from tidy3d.components.types import Axis, Bound, Coordinate, Shapely, Symmetry
from tidy3d.components.validators import assert_plane
from tidy3d.exceptions import SetupError

# Type for holding sets of indices associated with conductors,
# where the first set contains positive terminals and the second set contains negative terminals.
VoltageSets = tuple[set[int], set[int]]
# Conversion of user-supplied TerminalSpec into shapely geometries which are used to find intersecting conductors
TerminalSpecShapes = tuple[list[Shapely], list[Shapely]]


class ModePlaneAnalyzer(Box):
    """Analyzes conductor geometry intersecting a mode plane.

    This class analyzes the geometry of conductors in a simulation cross-section and is for internal use.
    """

    _plane_validator = assert_plane()

    field_data_colocated: bool = pd.Field(
        False,
        title="Field Data Colocated",
        description="Whether field data is colocated with grid points. When 'True', bounding boxes "
        "are placed with additional margin to avoid interpolated field values near conductor surfaces.",
    )

    structures: tuple[Structure, ...] = pd.Field(
        ...,
        title="Structures",
        description="Tuple of structures in the simulation to analyze for conductors.",
    )

    grid: Grid = pd.Field(
        ...,
        title="Grid",
        description="Simulation grid for snapping paths to field data positions.",
    )

    symmetry: tuple[Symmetry, Symmetry, Symmetry] = pd.Field(
        (0, 0, 0),
        title="Symmetry",
        description="Symmetry conditions for the simulation in (x, y, z) directions.",
    )

    sim_box: Box = pd.Field(
        ...,
        title="Simulation Box",
        description="Simulation domain box used for boundary condition analysis.",
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

    @cached_property
    def mode_symmetry(self) -> tuple[Symmetry, Symmetry, Symmetry]:
        """Mode symmetry considering simulation box and simulation symmetry."""
        return self._get_mode_symmetry(self.sim_box, self.symmetry)

    @cached_property
    def mode_limits(self) -> Bound:
        """Mode plane bounds restricted to final grid positions.

        Mode profiles are calculated on a grid which is expanded from the monitor size
        to the closest grid boundaries, taking into account symmetry conditions.
        """
        return self._get_mode_limits(self.grid, self.mode_symmetry)

    @cached_property
    def conductor_shapes(self) -> list[Shapely]:
        """Isolated conductor geometries in the mode plane.

        Finds all PEC/metal structures, merges touching conductors, and filters out
        grounded conductors (those touching PEC boundaries).

        Returns
        -------
        list[Shapely]
            List of conductor geometries after merging and filtering.

        Raises
        ------
        SetupError
            If no valid isolated conductors are found in the mode plane.
        """
        min_b_3d, max_b_3d = self.mode_limits

        intersection_plane = Box.from_bounds(min_b_3d, max_b_3d)
        conductor_shapely = self._get_isolated_conductors_as_shapely(
            intersection_plane, self.structures
        )

        conductor_shapely = self._filter_conductors_touching_sim_bounds(
            (min_b_3d, max_b_3d), self.mode_symmetry, conductor_shapely
        )

        if len(conductor_shapely) < 1:
            raise SetupError(
                "No valid isolated conductors were found in the mode plane. Please ensure that a 'Structure' "
                "with a medium of type 'PEC' or 'LossyMetalMedium' intersects the mode plane and is not touching "
                "the boundaries of the mode plane."
            )

        return conductor_shapely

    @cached_property
    def conductor_bounding_boxes(self) -> list[Box]:
        """Bounding boxes encompassing each isolated conductor.

        Each box is snapped to the grid and includes all symmetry-reflected regions.

        Returns
        -------
        list[Box]
            List of bounding boxes, one per isolated conductor.

        Raises
        ------
        SetupError
            If a generated bounding box intersects with a conductor.
        """

        def bounding_box_from_shapely(geom: Shapely) -> Box:
            """Helper to convert the shapely geometry bounds to a Box."""
            bounds = geom.bounds
            normal_center = self.center[self._normal_axis]
            rmin = Geometry.unpop_axis(normal_center, (bounds[0], bounds[1]), self._normal_axis)
            rmax = Geometry.unpop_axis(normal_center, (bounds[2], bounds[3]), self._normal_axis)
            return Box.from_bounds(rmin, rmax)

        # Get desired snapping behavior of box enclosed conductors.
        # Ideally, just large enough to coincide with the H field positions outside of the conductor.
        # So a half grid cell, when the metal boundary is coincident with grid boundaries.
        snap_spec = self._snap_spec

        bounding_boxes = []
        for shape in self.conductor_shapes:
            box = bounding_box_from_shapely(shape)
            boxes = self._apply_symmetries(self.symmetry, self.sim_box.center, box)
            for box in boxes:
                box_snapped = snap_box_to_grid(self.grid, box, snap_spec)
                bounding_boxes.append(box_snapped)

        for bounding_box in bounding_boxes:
            if self._check_box_intersects_with_conductors(self.conductor_shapes, bounding_box):
                raise SetupError(
                    "Failed to automatically generate path specification because a generated path "
                    "specification was found to intersect with a conductor. There is currently limited "
                    "support for complex conductor geometries, so please provide an explicit current "
                    "path specification through a 'CustomImpedanceSpec'. Alternatively, enforce a "
                    "smaller grid around the conductors in the mode plane, which may resolve the issue."
                )
        return bounding_boxes

    @cached_property
    def num_conductors(self) -> int:
        """Number of isolated conductors in the mode plane."""
        return len(self.conductor_shapes)

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

    def _convert_terminal_specifications_to_candidate_geometry(
        self, structures: list[Structure], terminal_specs: list[TerminalSpec]
    ) -> list[TerminalSpecShapes]:
        """Converts the different methods for specifying terminals in `TerminalSpec` into
        Shapely geometries. The intersection of these geometries with conductor polygons identifies
        the terminals in the mode plane.
        """

        def find_structure_by_name(target_name):
            """Find first element where name matches target_name."""
            return next(
                (item for item in structures if item.name is not None and item.name == target_name),
                None,
            )

        def convert_terminal(terminal_specifier) -> Shapely:
            if isinstance(terminal_specifier, tuple):
                return Point(*terminal_specifier)
            elif isinstance(terminal_specifier, str):
                structure = find_structure_by_name(terminal_specifier)
                if structure is None:
                    raise SetupError(
                        f"No structure found with name '{terminal_specifier}'. "
                        "Please ensure that a `Structure` with the same name has been added to the simulation."
                    )
                shapes_plane = self.intersections_with(structure.geometry)
                return shapes_plane
            elif terminal_specifier.shape[0] == 1:
                return Point(*terminal_specifier)
            elif terminal_specifier.shape[0] == 2:
                return LineString(terminal_specifier)
            else:
                return Polygon(terminal_specifier)

        terminal_spec_shapes = []
        for terminal_spec in terminal_specs:
            plus_spec_shapes = [
                convert_terminal(plus_terminal) for plus_terminal in terminal_spec.plus_terminals
            ]
            minus_spec_shapes = [
                convert_terminal(minus_terminal) for minus_terminal in terminal_spec.minus_terminals
            ]
            terminal_spec_shapes.append((plus_spec_shapes, minus_spec_shapes))
        return terminal_spec_shapes

    def _find_conductor_terminals(
        self,
        conductor_shapely: list[Shapely],
        terminal_specs: list[tuple[list[Shapely], list[Shapely]]],
    ) -> list[VoltageSets]:
        """Converts terminal specifications given as shapely geometry to conductor indices.

        For each terminal spec, identifies which conductors contain the specified terminal
        coordinates and returns sets of positive and negative conductor indices.

        Parameters
        ----------
        conductor_shapely : list[Shapely]
            List of conductor geometries in the mode plane.
        terminal_specs : list[Shapely]
            Terminal specifications with (x, y) coordinates for positive and negative terminals.

        Returns
        -------
        list[VoltageSets]
            List of (positive_conductor_indices, negative_conductor_indices) for each terminal spec.
        """

        def validate_conductor_intersection(indices: list[int], terminal_type: str) -> None:
            """Validate that exactly one conductor intersects with the terminal."""
            if len(indices) == 0:
                raise SetupError(
                    f"No conductor found intersecting with the {terminal_type}_terminal. "
                    "Please ensure that your terminal specification (coordinate, line, or polygon) "
                    "intersects with at least one conductive structure in the mode plane. "
                    "Check that the terminal coordinates are within the bounds of a conductor."
                )
            elif len(indices) > 1:
                raise SetupError(
                    f"Multiple conductors ({len(indices)}) found intersecting with the {terminal_type}_terminal. "
                    "Please ensure that your terminal specification intersects with exactly one conductor. "
                    "Consider making your terminal specification more precise (e.g., using a smaller region or point) "
                    "to uniquely identify a single conductor."
                )

        terminals = []
        for term_spec in terminal_specs:
            all_plus_indices = set()
            all_minus_indices = set()
            for plus_terminal in term_spec[0]:
                plus_indices = [
                    i for i, geom in enumerate(conductor_shapely) if geom.intersects(plus_terminal)
                ]
                validate_conductor_intersection(plus_indices, "plus")
                all_plus_indices.update(plus_indices)
            for minus_terminal in term_spec[1]:
                minus_indices = [
                    i for i, geom in enumerate(conductor_shapely) if geom.intersects(minus_terminal)
                ]
                validate_conductor_intersection(minus_indices, "minus")
                all_minus_indices.update(minus_indices)
            terminals.append((all_plus_indices, all_minus_indices))
        return terminals

    def _identify_conductor_voltage_sets(
        self, terminal_specs: tuple[TerminalSpec, ...]
    ) -> list[VoltageSets]:
        """Identifies the conductor polygons associated with the supplied `TerminalSpec`.
        The conductor polygons are identified through their index into `self.conductor_shapes`.

        Parameters
        ----------
        terminal_specs : tuple[TerminalSpec, ...]
            Terminal specifications with (x, y) coordinates for positive and negative terminals.

        Returns
        -------
        list[VoltageSets]
            List of (positive_conductor_indices, negative_conductor_indices) for each terminal spec."""

        conductor_shapes = self.conductor_shapes

        terminal_spec_shapes = self._convert_terminal_specifications_to_candidate_geometry(
            self.structures, terminal_specs
        )
        terminals = self._find_conductor_terminals(conductor_shapes, terminal_spec_shapes)
        return terminals

    def _validate_conductor_voltage_configurations(
        self, conductor_voltage_sets: list[VoltageSets]
    ) -> None:
        """Validates terminal specifications for conflicts and duplicates.

        Checks that no conductor appears in both positive and negative sets, and that
        no duplicate configurations exist (including polarity-reversed duplicates).

        Parameters
        ----------
        conductor_voltage_sets : list[VoltageSets]
            List of (positive_conductor_indices, negative_conductor_indices) to validate.

        Raises
        ------
        SetupError
            If a conductor appears in both positive and negative sets, or if duplicate
            configurations are detected.
        """
        # Check that a conductor index only belongs in either plus or minus sets
        for voltage_set in conductor_voltage_sets:
            if not voltage_set[0].isdisjoint(voltage_set[1]):
                raise SetupError(
                    "A conductor cannot be assigned to both a positive and negative voltage."
                )

        # Check that only unique polarity configurations exist
        # Two configurations are considered the same if one is the polarity-reversed version of the other
        unique_terminal_configuration = set()

        for voltage_set in conductor_voltage_sets:
            pos, neg = voltage_set

            # Create a normalized representation that treats (pos, neg) and (neg, pos) as equivalent
            # Using frozenset of frozensets ensures order-independence
            terminal_configuration = frozenset([frozenset(pos), frozenset(neg)])

            if terminal_configuration in unique_terminal_configuration:
                raise SetupError(
                    "Duplicate voltage configuration detected. "
                    "Each unique pair of conductor sets (including polarity-reversed pairs) can only appear once."
                )

            unique_terminal_configuration.add(terminal_configuration)

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
