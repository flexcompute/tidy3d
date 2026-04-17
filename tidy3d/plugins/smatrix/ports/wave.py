"""Class for representing a scattering matrix wave port."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Literal, Optional, Union

import numpy as np
import xarray as xr
from pydantic import Field, NonNegativeInt, field_validator, model_validator

from tidy3d.components.base import cached_property
from tidy3d.components.boundary import (
    ABCBoundary,
    InternalAbsorber,
    ModeABCBoundary,
)
from tidy3d.components.data.data_array import (
    FreqModeDataArray,
    ImpedanceFreqModeModeDataArray,
    ImpedanceFreqTerminalTerminalDataArray,
    ImpedanceModeDataArray,
    ImpedanceTerminalDataArray,
)
from tidy3d.components.data.sim_data import SimulationData
from tidy3d.components.geometry.base import Box, ClipOperation, GeometryGroup
from tidy3d.components.geometry.polyslab import PolySlab
from tidy3d.components.geometry.utils import (
    _shift_object,
    _shift_value_signed,
    filter_intersecting_geometries,
)
from tidy3d.components.grid.grid_spec import GridSpec
from tidy3d.components.microwave.mode_spec import (
    MONITOR_COLOCATE,
    MicrowaveModeSpec,
    MicrowaveTerminalModeSpec,
)
from tidy3d.components.microwave.monitor import MicrowaveModeMonitor
from tidy3d.components.microwave.path_integrals.mode_plane_analyzer import ModePlaneAnalyzer
from tidy3d.components.microwave.path_integrals.specs.impedance import (
    AutoImpedanceSpec,
    CustomImpedanceSpec,
)
from tidy3d.components.microwave.source import MicrowaveTerminalSource
from tidy3d.components.mode.simulation import ModeSimulation
from tidy3d.components.source.field import ModeSource
from tidy3d.components.source.frame import PECFrame
from tidy3d.components.structure import MeshOverrideStructure
from tidy3d.components.types import Complex, Direction
from tidy3d.components.validators import assert_plane
from tidy3d.constants import OHM, inf
from tidy3d.exceptions import SetupError, ValidationError
from tidy3d.log import log
from tidy3d.plugins.smatrix.ports.base_terminal import AbstractTerminalPort

if TYPE_CHECKING:
    from pydantic import NonNegativeFloat, ValidationInfo

    from tidy3d.compat import Self
    from tidy3d.components.data.data_array import (
        CurrentFreqTerminalDataArray,
        VoltageFreqTerminalDataArray,
    )
    from tidy3d.components.grid.grid import Grid
    from tidy3d.components.microwave.data.monitor_data import MicrowaveModeData
    from tidy3d.components.microwave.mode_spec import MicrowaveModeSpecType
    from tidy3d.components.simulation import Simulation
    from tidy3d.components.source.time import SourceTimeType
    from tidy3d.components.structure import Structure
    from tidy3d.components.types import Axis, FreqArray, Shapely, Symmetry
    from tidy3d.components.types.base import PriorityMode
    from tidy3d.plugins.mode import ModeSolver

DEFAULT_WAVE_PORT_NUM_CELLS = 5
MIN_WAVE_PORT_NUM_CELLS = 3
DEFAULT_WAVE_PORT_FRAME = PECFrame()
DEFAULT_REFERENCE_IMPEDANCE_VALUE = 50
DEFAULT_TERMINAL_LABEL_PREFIX = "T"
DEFAULT_DIFFERENTIAL_PAIR_LABEL_PREFIX = "Diff"
# Tolerance (in um) for finding structures that intersect the waveport plane.
# PCB modeler exports often have polygon vertices that don't exactly touch the
# port plane; 10 nm covers typical coordinate precision gaps.
EXTRUDE_STRUCTURES_TOL = 0.01
# Keep a thin buffer so near-edge meshing hints survive local port filtering.
REFINEMENT_BOX_SCALE = 1.25


class AbstractWavePort(AbstractTerminalPort, Box):
    """Class representing a single terminal-based port that requires 2D mode solving."""

    _plane_validator = assert_plane()

    direction: Direction = Field(
        title="Direction",
        description="'+' or '-', defining which direction is considered 'input'.",
    )

    num_grid_cells: Optional[int] = Field(
        DEFAULT_WAVE_PORT_NUM_CELLS,
        ge=MIN_WAVE_PORT_NUM_CELLS,
        title="Number of Grid Cells",
        description="Number of mesh grid cells in the transverse plane of the `WavePort`. "
        "Used in generating the suggested list of :class:`.MeshOverrideStructure` objects. "
        "Must be greater than or equal to 3. When set to `None`, no grid refinement is performed.",
    )

    conjugated_dot_product: bool = Field(
        False,
        title="Conjugated Dot Product",
        description="Use conjugated or non-conjugated dot product for mode decomposition.",
    )

    frame: Optional[PECFrame] = Field(
        DEFAULT_WAVE_PORT_FRAME,
        title="Source Frame",
        description="Add a thin frame around the source during FDTD run for an improved injection.",
    )

    absorber: Union[bool, ABCBoundary, ModeABCBoundary] = Field(
        True,
        title="Absorber",
        description="Place a mode absorber in the port. If ``True``, an automatically generated mode absorber is placed in the port. "
        "If :class:`.ABCBoundary` or :class:`.ModeABCBoundary`, a mode absorber is placed in the port with the specified boundary conditions.",
    )

    extrude_structures: bool = Field(
        False,
        title="Extrude Structures",
        description="Extrudes structures that intersect the wave port plane by a few grid cells when ``True``, improving mode injection accuracy.",
    )

    reference_impedance: Union[
        Literal["Z0"], Complex, ImpedanceModeDataArray, ImpedanceTerminalDataArray
    ] = Field(
        "Z0",
        title="Reference Impedance",
        description="User-specified reference impedance for S-parameter computation. "
        "If ``Z0`` (default), the characteristic impedance "
        "is used. Otherwise, it can be a single complex value applied to all modes, or a data array "
        f"specified for each. If the data array misses some modes, {DEFAULT_REFERENCE_IMPEDANCE_VALUE} "
        "Ohm is applied to the missing ones.",
        json_schema_extra={"units": OHM},
    )

    @field_validator("reference_impedance")
    @classmethod
    def _validate_reference_impedance_positive(
        cls, val: Union[Literal["Z0"], Complex, ImpedanceModeDataArray, ImpedanceTerminalDataArray]
    ) -> Union[Literal["Z0"], Complex, ImpedanceModeDataArray, ImpedanceTerminalDataArray]:
        """Validate that reference impedance has positive real part."""
        # Skip validation for "Z0"
        if val == "Z0":
            return val

        # Handle scalar complex values
        if isinstance(val, (int, float, complex)):
            if not np.isfinite(val):
                raise ValidationError(f"Reference impedance must be finite. Got {val}.")
            if np.real(val) <= 0:
                raise ValidationError(
                    f"Reference impedance must have positive real part. Got {val} with real part {np.real(val)}."
                )
            return val

        # Handle DataArray (ImpedanceModeDataArray or ImpedanceTerminalDataArray)
        values = val.values
        if not np.all(np.isfinite(values)):
            raise ValidationError("All reference impedance values must be finite (no NaN or inf).")
        # Check all values have positive real part
        real_parts = np.real(values)
        if np.any(real_parts <= 0):
            min_real = np.min(real_parts)
            raise ValidationError(
                f"All reference impedance values must have positive real part. "
                f"Found minimum real part: {min_real}."
            )
        return val

    def get_reference_impedance_matrix(
        self, sim_mode_data: Union[SimulationData, MicrowaveModeData]
    ) -> Union[ImpedanceFreqModeModeDataArray, ImpedanceFreqTerminalTerminalDataArray]:
        """Retrieve the reference impedance of the port. In general, it's a diagonal matrix; but
        it can be a full matrix when the terminals in the port are coupled.
        """
        computed_Z0 = self.get_characteristic_impedance_matrix(sim_mode_data)

        # If "Z0", return computed characteristic impedance matrix
        if self.reference_impedance == "Z0":
            return computed_Z0

        # User-specified reference impedance - build diagonal matrix
        # Detect whether computed_Z0 is mode-based or terminal-based
        is_mode_based = "mode_index_out" in computed_Z0.dims

        if is_mode_based:
            dim_prefix = "mode"
            result_cls = ImpedanceFreqModeModeDataArray
        else:
            dim_prefix = "terminal"
            result_cls = ImpedanceFreqTerminalTerminalDataArray

        suffix = "index" if is_mode_based else "label"
        dim_1d, dim_out, dim_in = (f"{dim_prefix}_{suffix}{s}" for s in ("", "_out", "_in"))

        indices = computed_Z0.coords[dim_out].values
        num_indices = len(indices)

        # Create identity matrix as xarray DataArray for broadcasting
        eye = xr.DataArray(
            np.eye(num_indices),
            coords={dim_out: indices, dim_in: indices},
        )

        # Handle scalar Complex vs DataArray reference impedance
        if isinstance(self.reference_impedance, (int, float, complex)):
            # Scalar case: create uniform reference values for all indices
            ref_values = xr.DataArray(
                np.full(num_indices, self.reference_impedance),
                coords={dim_1d: indices},
            )
        else:
            # Data array: use values for each mode/terminal, fallback to default for missing ones
            # Reindex to match indices, filling missing with default value
            ref_values = self.reference_impedance.reindex(
                {dim_1d: indices}, fill_value=DEFAULT_REFERENCE_IMPEDANCE_VALUE
            )

        # Rename and broadcast to diagonal matrix
        ref_diag = ref_values.rename({dim_1d: dim_out}) * eye

        return result_cls(ref_diag)

    @abstractmethod
    def get_characteristic_impedance_matrix(
        self, sim_mode_data: Union[SimulationData, MicrowaveModeData]
    ) -> Union[ImpedanceFreqModeModeDataArray, ImpedanceFreqTerminalTerminalDataArray]:
        """Retrieve the characteristic impedance matrix of the port."""

    @cached_property
    @abstractmethod
    def _mode_spec(self) -> Optional[MicrowaveModeSpecType]:
        """Internal mode specification for the port. Return ``None`` if
        it cannot be resolved in WavePort alone.
        """

    @abstractmethod
    def _mode_spec_from_isolated_floating_conductors(
        self, conductors: dict[str, tuple[Shapely, Box]]
    ) -> MicrowaveModeSpecType:
        """Generate a mode specification from isolated floating conductors."""

    @abstractmethod
    def _mode_indices(self, mode_spec: Optional[MicrowaveModeSpecType] = None) -> tuple[int, ...]:
        """Return the tuple of mode indices that will be excited and monitored by this port.

        Parameters
        ----------
        mode_spec : MicrowaveModeSpecType, optional
            Resolved mode specification whose ``num_modes`` is an integer.
            When ``None``, the implementation should fall back to ``self._mode_spec``.

        Returns
        -------
        tuple[int, ...]
            Ordered mode indices for this port.
        """

    @cached_property
    def injection_axis(self) -> Axis:
        """Injection axis of the port."""
        return self.size.index(0.0)

    @cached_property
    def transverse_axes(self) -> tuple[Axis, Axis]:
        """Transverse axes of the port."""
        _, trans_axes = Box.pop_axis([0, 1, 2], self.injection_axis)
        return trans_axes

    @cached_property
    def _mode_filter_box(self) -> Box:
        """Expanded bounding box used to filter structures and grid-spec entities
        when building a reduced ModeSimulation for this port.

        The transverse dimensions are scaled by ``REFINEMENT_BOX_SCALE`` so
        that nearby mesh overrides and snapping points that influence the grid
        around the port are captured (not just those strictly inside the port).

        The normal (injection) axis is set to infinity so that structures
        along this axis are always included when creating the final grid.
        This will give the closest possible match to the grid produced by
        the full simulation constructed in the TerminalComponentModeler.
        """
        center = list(self.center)
        size = list(self.size)
        for ax in self.transverse_axes:
            size[ax] *= REFINEMENT_BOX_SCALE
        size[self.injection_axis] = inf
        return Box(center=center, size=size)

    @cached_property
    def _mode_monitor_name(self) -> str:
        """Return the name of the :class:`.MicrowaveModeMonitor` associated with this port."""
        return f"{self.name}_mode"

    @cached_property
    def _mode_plane_analyzer(self) -> ModePlaneAnalyzer:
        """Mode plane analyzer for the port."""
        return ModePlaneAnalyzer(
            center=self.center,
            size=self.size,
            field_data_colocated=MONITOR_COLOCATE,
        )

    def _get_isolated_floating_conductors(
        self,
        structures: list[Structure],
        grid: Grid,
        symmetry: tuple[Symmetry, Symmetry, Symmetry],
        sim_box: Box,
        interior_disjoint_geometries: bool = True,
    ) -> dict[str, tuple[Shapely, Box]]:
        """Get isolated floating conductors (terminals) on the port plane.

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
        interior_disjoint_geometries : bool = True
            If ``True``, conductors on the plane will not be overridden by other materials,
            allowing a faster merging path that skips overlap removal.

        Returns
        -------
        dict[str, tuple[Shapely, Box]]:
            Mapping from terminal name to terminal shape and bounding box.
        """
        bounding_boxes, shapes = self._mode_plane_analyzer.get_conductor_bounding_boxes(
            structures,
            grid,
            symmetry,
            sim_box,
            interior_disjoint_geometries=interior_disjoint_geometries,
        )
        labels = [f"{DEFAULT_TERMINAL_LABEL_PREFIX}{i}" for i in range(len(shapes))]

        return {label: (shape, bbox) for label, shape, bbox in zip(labels, shapes, bounding_boxes)}

    def _isolated_floating_conductors_from_simulation(
        self, simulation: Simulation
    ) -> dict[str, tuple[Shapely, Box]]:
        """Get isolated floating conductors from a prepared simulation."""
        interior_disjoint_geometries = ModePlaneAnalyzer.apply_interior_disjoint_geometries(
            simulation.structure_priority_mode
        )
        return self._get_isolated_floating_conductors(
            simulation.volumetric_structures,
            simulation.grid,
            simulation.symmetry,
            simulation.simulation_geometry,
            interior_disjoint_geometries=interior_disjoint_geometries,
        )

    def _resolve_mode_spec_from_simulation(
        self,
        simulation: Simulation,
        mode_spec: Optional[MicrowaveModeSpecType] = None,
        conductors: Optional[dict[str, tuple[Shapely, Box]]] = None,
    ) -> MicrowaveModeSpecType:
        """Resolve the mode specification from a prepared simulation when needed."""
        if mode_spec is not None:
            return self._validate_resolved_mode_spec(mode_spec)

        if self._mode_spec is not None:
            return self._mode_spec

        if conductors is None:
            conductors = self._isolated_floating_conductors_from_simulation(simulation)
        return self._mode_spec_from_isolated_floating_conductors(conductors)

    def _validate_resolved_mode_spec(
        self, mode_spec: Optional[MicrowaveModeSpecType] = None
    ) -> MicrowaveModeSpecType:
        """If the resolved mode_spec is not provided, validate that self._mode_spec not None."""
        if mode_spec is not None:
            if mode_spec.num_modes == "auto":
                raise SetupError(
                    "The supplied mode specification has num_modes='auto'. "
                    "Please pass a mode_spec with an explicit number of modes."
                )
            return mode_spec

        if self._mode_spec is None:
            raise SetupError(
                "Mode specification cannot be resolved in WavePort alone. "
                "Please pass a mode_spec with an explicit number of modes."
            )
        return self._mode_spec

    def _validate_resolved_mode(self, mode_spec: MicrowaveModeSpecType) -> None:
        """Validate mode-field bounds against a resolved mode spec. Override in subclasses."""

    def _warn_resolved_multimode_absorber(
        self, mode_spec: MicrowaveModeSpecType, include_port_name: bool = False
    ) -> None:
        """Warn when absorber is enabled with multiple resolved modes."""
        if not self.absorber or mode_spec.num_modes <= 1:
            return

        warning_prefix = f"Port '{self.name}': " if include_port_name else ""
        log.warning(
            f"{warning_prefix}Absorber is enabled with {mode_spec.num_modes} modes. "
            "Absorption is not properly implemented for multimode cases yet and will be "
            "added in a future release. For now, please extend the transmission line into the PML region."
        )

    def to_monitors(
        self,
        freqs: FreqArray,
        snap_center: Optional[float] = None,
        grid: Optional[Grid] = None,
        mode_spec: Optional[MicrowaveModeSpecType] = None,
    ) -> list[MicrowaveModeMonitor]:
        """Create monitors from the wave port.

        The wave port uses a :class:`.MicrowaveModeMonitor` to compute the characteristic impedance
        and the port voltages and currents.

        Parameters
        ----------
        freqs : FreqArray
            Frequencies to monitor.
        snap_center : float, optional
            Position to snap the monitor center to along injection axis.
        grid : Grid, optional
            Simulation grid (unused but kept for API compatibility).
        mode_spec : MicrowaveModeSpecType, optional
            Resolved mode specification with integer num_modes. If None,
            uses self._mode_spec but raises SetupError if num_modes='auto'.
        """
        center = list(self.center)
        if snap_center:
            center[self.injection_axis] = snap_center
        # Use provided mode_spec if given, otherwise fall back to self._mode_spec
        mode_spec = self._validate_resolved_mode_spec(mode_spec)
        mode_mon = MicrowaveModeMonitor(
            center=self.center,
            size=self.size,
            freqs=freqs,
            name=self._mode_monitor_name,
            colocate=MONITOR_COLOCATE,
            use_colocated_integration=MONITOR_COLOCATE,
            mode_spec=mode_spec,
            store_fields_direction=self.direction,
            conjugated_dot_product=self.conjugated_dot_product,
        )
        return [mode_mon]

    def _extruded_structures(
        self,
        simulation: Simulation,
        tol: float = EXTRUDE_STRUCTURES_TOL,
    ) -> tuple[Structure, ...]:
        """Extrude local structures for this port using a simulation."""

        if not self.extrude_structures:
            return ()

        snap_center = self.center[self.injection_axis] + _shift_value_signed(
            self,
            simulation.grid,
            bounds=simulation.bounds,
            direction=self.direction,
            shift=-2,
            name=f"Port {self.name}",
        )
        absorber = self._to_absorber_geometry(snap_center=snap_center)
        shifted_absorber = _shift_object(
            obj=absorber,
            grid=simulation.grid,
            bounds=simulation.bounds,
            direction=absorber.direction,
            shift=absorber.grid_shift,
        )
        (box, inj_axis, direction) = simulation._pec_frame_box(shifted_absorber, expand=True)
        surfaces = box.surfaces(box.size, box.center)

        sign = 1 if direction == "+" else -1
        back_pec_plane = surfaces[2 * inj_axis + (1 if direction == "+" else 0)]
        extrude_to = back_pec_plane.center[inj_axis]

        # Build the cutting plane from the discretized monitor bounds so that
        # it covers the same region the mode solver sees (slightly larger
        # than the port box due to grid snapping).
        span_inds = simulation._discretize_inds_monitor(self, colocate=MONITOR_COLOCATE)
        bds = simulation._subgrid(span_inds=span_inds).boundaries.to_list
        rmin = [b[0] for b in bds]
        rmax = [b[-1] for b in bds]
        rmin[inj_axis] = self.center[inj_axis] - sign * tol
        rmax[inj_axis] = rmin[inj_axis]
        cutting_plane = Box.from_bounds(rmin=rmin, rmax=rmax)
        extrusion_bounds = [cutting_plane.center[inj_axis], extrude_to][::sign]

        new_structures = []
        for structure in simulation.structures:
            shapely_geom = cutting_plane.intersections_with(structure.geometry)

            polygon_list = []
            for geom in shapely_geom:
                polygon_list.extend(ClipOperation.to_polygon_list(geom))

            if not polygon_list:
                continue

            new_geoms = []
            for polygon in polygon_list:
                exterior_vertices = np.array(polygon.exterior.coords)
                outer_shell = PolySlab(
                    axis=inj_axis,
                    slab_bounds=extrusion_bounds,
                    vertices=exterior_vertices,
                )
                hole_polyslabs = [
                    PolySlab(
                        axis=inj_axis,
                        slab_bounds=extrusion_bounds,
                        vertices=np.array(hole.coords),
                    )
                    for hole in polygon.interiors
                ]

                if hole_polyslabs:
                    holes = GeometryGroup(geometries=hole_polyslabs)
                    extruded_slab = ClipOperation(
                        operation="difference",
                        geometry_a=outer_shell,
                        geometry_b=holes,
                    )
                else:
                    extruded_slab = outer_shell

                new_geoms.append(extruded_slab)

            extruded_name = f"{structure.name}_extruded_{self.name}" if structure.name else None
            new_structures.append(
                structure.updated_copy(
                    geometry=GeometryGroup(geometries=new_geoms),
                    name=extruded_name,
                )
            )

        if not new_structures:
            raise SetupError(
                f"The 'WavePort' '{self.name}' does not intersect any structures. "
                "Please ensure that it is located within or at the boundary of a structure."
            )

        return tuple(new_structures)

    def _prepare_simulation_for_mode(
        self,
        simulation: Simulation,
        structures: Optional[tuple[Structure, ...] | list[Structure]] = None,
        grid_spec: Optional[GridSpec] = None,
        reduce_simulation: bool = False,
    ) -> Simulation:
        """Return the local simulation prepared for this port's mode solve.

        By default the simulation is used as-is except for appending port mesh
        overrides and extruded structures.  When ``reduce_simulation`` is True,
        structures and grid spec are pruned to the port region for faster
        mode simulation/solver set up on large simulations.

        Parameters
        ----------
        reduce_simulation : bool = False
            When True, use ``simulation.subsection`` and
            ``filter_intersecting_geometries`` to prune structures and grid
            entities to the port region before assembling the mode simulation.
        """

        if reduce_simulation:
            # Extract a baseline localized simulation from the full one.
            localized_sim = simulation.subsection(
                region=self._mode_filter_box,
                sources=(),
                monitors=(),
                internal_absorbers=(),
                remove_outside_structures=True,
                remove_outside_grid_spec=True,
                warn_symmetry_expansion=False,
                deep_copy=False,
                validate_geometries=False,
            )

            # Use caller-supplied structures/grid_spec when provided,
            # filtering them to the port region; otherwise use the localized sim's.
            if structures is None:
                filtered_structures = localized_sim.structures
            else:
                input_structures = tuple(structures)
                pruned_geometries = filter_intersecting_geometries(
                    [structure.geometry for structure in input_structures], self._mode_filter_box
                )
                filtered_structures = tuple(
                    structure.updated_copy(geometry=geometry, deep=False)
                    for structure, geometry in zip(input_structures, pruned_geometries)
                    if geometry is not None
                )

            if grid_spec is None:
                filtered_grid_spec = localized_sim.grid_spec
            else:
                filtered_grid_spec = grid_spec._localized_copy(region=self._mode_filter_box)
        else:
            filtered_structures = (
                tuple(structures) if structures is not None else simulation.structures
            )
            filtered_grid_spec = grid_spec if grid_spec is not None else simulation.grid_spec

        # Append port-specific mesh overrides for the mode region.
        if self._is_using_mesh_refinement:
            overrides = list(filtered_grid_spec.override_structures)
            overrides.extend(self.to_mesh_overrides())
            filtered_grid_spec = filtered_grid_spec.updated_copy(override_structures=overrides)

        simulation = simulation.updated_copy(
            grid_spec=filtered_grid_spec,
            structures=filtered_structures,
            deep=False,
            validate=False,
        )

        # Step 4: Apply extrusion if used.  Freeze the grid first so the
        # extruded structures don't change the mesh.
        extra_structures = self._extruded_structures(
            simulation=simulation,
        )
        if not extra_structures:
            return simulation

        return simulation.updated_copy(
            grid_spec=GridSpec.from_grid(simulation.grid),
            structures=[*simulation.structures, *extra_structures],
            validate=False,
            deep=False,
        )

    def to_mode_solver(
        self,
        simulation: Simulation,
        freqs: FreqArray,
        mode_spec: Optional[MicrowaveModeSpecType] = None,
        reduce_simulation: bool = False,
        structures: Optional[tuple[Structure, ...] | list[Structure]] = None,
        grid_spec: Optional[GridSpec] = None,
        structure_priority_mode: Optional[PriorityMode] = "conductor",
    ) -> ModeSolver:
        """Helper to create a :class:`.ModeSolver` instance.

        Passing ``structures`` and/or ``grid_spec`` separately allows
        ``simulation`` to be a lightweight domain-only carrier that is fast
        to instantiate, while the heavy components are reduced to the port
        region independently.

        Parameters
        ----------
        simulation : Simulation
            Simulation to solve modes for.
        freqs : FreqArray
            Frequencies to solve at.
        mode_spec : MicrowaveModeSpecType, optional
            Resolved mode specification with integer num_modes. If None,
            the mode specification is resolved from the prepared local simulation.
            Explicitly passed specs must already have integer ``num_modes``.
        reduce_simulation : bool = False
            When True, prune structures and grid entities to the port region
            for faster mode simulation/solver set up on large simulations.
        structures : list[:class:`.Structure`] or tuple[:class:`.Structure`, ...], optional
            Structures to use in place of ``simulation.structures``. Passing
            structures separately allows ``simulation`` to be a lightweight
            domain-only carrier that is fast to instantiate, while the
            structures are reduced to the port region independently.
        grid_spec : :class:`.GridSpec`, optional
            Grid specification to use in place of ``simulation.grid_spec``.
            Like ``structures``, passing this separately avoids embedding
            a full grid spec in the simulation, speeding up instantiation.
        structure_priority_mode : :class:`.PriorityMode`, optional
            Optional structure-priority mode override. Defaults to
            ``"conductor"`` to match :class:`.TerminalComponentModeler`. Pass
            ``None`` to use ``simulation.structure_priority_mode`` as-is.

        Returns
        -------
        :class:`.ModeSolver`
            Standalone mode solver with port mesh overrides applied.
        """
        mode_sim = self.to_mode_simulation(
            simulation=simulation,
            freqs=freqs,
            mode_spec=mode_spec,
            structures=structures,
            grid_spec=grid_spec,
            structure_priority_mode=structure_priority_mode,
            reduce_simulation=reduce_simulation,
        )
        return mode_sim._mode_solver

    def to_mode_simulation(
        self,
        simulation: Simulation,
        freqs: FreqArray,
        mode_spec: Optional[MicrowaveModeSpecType] = None,
        reduce_simulation: bool = False,
        structures: Optional[tuple[Structure, ...] | list[Structure]] = None,
        grid_spec: Optional[GridSpec] = None,
        structure_priority_mode: Optional[PriorityMode] = "conductor",
    ) -> ModeSimulation:
        """Create a :class:`.ModeSimulation` with port mesh refinement applied.

        Passing ``structures`` and/or ``grid_spec`` separately allows
        ``simulation`` to be a lightweight domain-only carrier that is fast
        to instantiate, while the heavy components are reduced to the port
        region independently.

        Parameters
        ----------
        simulation : :class:`.Simulation`
            Base simulation defining the simulation domain and default structures,
            materials, and grid.
        freqs : :class:`.FreqArray`
            Frequencies at which to solve for modes.
        mode_spec : MicrowaveModeSpecType, optional
            Resolved mode specification with integer num_modes. If None,
            the mode specification is resolved from the prepared local simulation.
            Explicitly passed specs must already have integer ``num_modes``.
        reduce_simulation : bool = False
            When True, prune structures and grid entities to the port region
            for faster mode simulation/solver set up on large simulations.
        structures : list[:class:`.Structure`] or tuple[:class:`.Structure`, ...], optional
            Structures to use in place of ``simulation.structures``. Passing
            structures separately allows ``simulation`` to be a lightweight
            domain-only carrier that is fast to instantiate, while the
            structures are reduced to the port region independently.
        grid_spec : :class:`.GridSpec`, optional
            Grid specification to use in place of ``simulation.grid_spec``.
            Like ``structures``, passing this separately avoids embedding
            a full grid spec in the simulation, speeding up instantiation.
        structure_priority_mode : :class:`.PriorityMode`, optional
            Optional structure-priority mode override. Defaults to
            ``"conductor"`` to match :class:`.TerminalComponentModeler`. Pass
            ``None`` to use ``simulation.structure_priority_mode`` as-is.

        Returns
        -------
        :class:`.ModeSimulation`
            Standalone mode simulation with port mesh overrides applied.
        """

        if structure_priority_mode is not None:
            simulation = simulation.updated_copy(
                structure_priority_mode=structure_priority_mode,
                validate=False,
                deep=False,
            )

        # TODO: Propagate `WavePort.frame` when ModeSimulation support PEC frames.
        simulation = self._prepare_simulation_for_mode(
            simulation=simulation,
            structures=structures,
            grid_spec=grid_spec,
            reduce_simulation=reduce_simulation,
        )

        mode_spec = self._resolve_mode_spec_from_simulation(simulation, mode_spec=mode_spec)
        self._validate_resolved_mode(mode_spec)

        return ModeSimulation.from_simulation(
            simulation=simulation,
            plane=self.geometry,
            mode_spec=mode_spec,
            freqs=freqs,
            direction=self.direction,
            colocate=MONITOR_COLOCATE,
            use_colocated_integration=MONITOR_COLOCATE,
            conjugated_dot_product=self.conjugated_dot_product,
        )

    def to_absorber(
        self,
        snap_center: Optional[float] = None,
        freq_spec: Optional[NonNegativeFloat] = None,
        mode_spec: Optional[MicrowaveModeSpecType] = None,
    ) -> InternalAbsorber:
        """Create an internal absorber from the wave port.

        Parameters
        ----------
        snap_center : float, optional
            Position to snap the absorber center to along injection axis.
        freq_spec : float, optional
            Frequency specification for the mode ABC boundary.
        mode_spec : MicrowaveModeSpecType, optional
            Resolved mode specification with integer num_modes. If None,
            uses self._mode_spec but raises SetupError if num_modes='auto'.
        """
        absorber = self._to_absorber_geometry(snap_center=snap_center)
        if isinstance(self.absorber, (ABCBoundary, ModeABCBoundary)):
            return absorber.updated_copy(boundary_spec=self.absorber)

        mode_spec = self._validate_resolved_mode_spec(mode_spec)
        # TODO: ModeABCBoundary currently only accepts one mode, so
        # we choose the first mode for now until we have multimodal absorber support.
        mode_index = self._mode_indices(mode_spec)[0]
        boundary_spec = ModeABCBoundary(
            mode_spec=mode_spec,
            mode_index=mode_index,
            plane=self.geometry,
            freq_spec=freq_spec,
        )
        return absorber.updated_copy(boundary_spec=boundary_spec)

    def _to_absorber_geometry(self, snap_center: Optional[float] = None) -> InternalAbsorber:
        """Create an internal absorber with the correct geometry for geometry-only calculations.

        A dummy ABC boundary is used so geometry can be created before the real
        absorber boundary conditions are resolved.
        """
        center = list(self.center)
        if snap_center:
            center[self.injection_axis] = snap_center

        return InternalAbsorber(
            center=center,
            size=self.size,
            boundary_spec=ABCBoundary(permittivity=1.0),
            direction="-"
            if self.direction == "+"
            else "+",  # absorb in the opposite direction of source
            grid_shift=1,  # absorb in the next pixel
        )

    def to_mesh_overrides(self) -> list[MeshOverrideStructure]:
        """Creates a list of :class:`.MeshOverrideStructure` for mesh refinement in the transverse
        plane of the port. The mode source requires at least 3 grid cells in the transverse
        dimensions, so these mesh overrides will be added to the simulation to ensure that this
        requirement is satisfied.
        """
        dl = [None] * 3
        for trans_axis in self.transverse_axes:
            dl[trans_axis] = self.size[trans_axis] / self.num_grid_cells

        return [
            MeshOverrideStructure(
                geometry=Box(center=self.center, size=self.size),
                dl=dl,
                shadow=False,
                priority=-1,
            )
        ]

    def _get_mode_data(
        self, sim_mode_data: Union[SimulationData, MicrowaveModeData]
    ) -> MicrowaveModeData:
        """Get the mode data from the simulation data or mode data directly."""
        if isinstance(sim_mode_data, SimulationData):
            return sim_mode_data[self._mode_monitor_name]
        return sim_mode_data

    @property
    def _is_using_mesh_refinement(self) -> bool:
        """Check if this wave port is using mesh refinement options.

        Returns ``True`` if a custom grid cell count is specified.
        """
        return self.num_grid_cells is not None

    @model_validator(mode="after")
    def _check_absorber_if_extruding_structures(self) -> Self:
        """Raise validation error when ``extrude_structures`` is set to ``True``
        while ``absorber`` is set to ``False``."""

        if self.extrude_structures and not self.absorber:
            raise ValidationError(
                "Structure extrusion for a waveport requires an internal absorber. Set `absorber=True` to enable it."
            )

        return self


class WavePort(AbstractWavePort):
    """Class representing a single modal-driven wave port.

    Notes
    -----
    By default, the characteristic impedance of each mode is used as the reference impedance
    for S-parameter calculations.
    """

    mode_spec: MicrowaveModeSpec = Field(
        default_factory=MicrowaveModeSpec._default_without_license_warning,
        title="Mode Specification",
        description="Parameters to feed to mode solver which determine modes and how transmission line "
        "quantities, e.g., characteristic impedance, are computed.",
    )

    mode_index: Optional[NonNegativeInt] = Field(
        None,
        title="Mode Index (deprecated)",
        description="Index into the collection of modes returned by mode solver. "
        "Specifies which mode to inject using this port. "
        "Deprecated. Use the 'mode_selection' field instead.",
    )

    mode_selection: Optional[tuple[int, ...]] = Field(
        None,
        title="Mode Selection",
        description="Selects specific mode(s) to use from the mode solver. "
        "Can be a single integer for one mode, or a tuple of integers for multiple modes. "
        "If ``None`` (default), all modes from the ``mode_spec`` are used. "
        "Indices must be non-negative and less than ``mode_spec.num_modes``.",
    )

    def _mode_indices(self, mode_spec: Optional[MicrowaveModeSpecType] = None) -> tuple[int, ...]:
        """Return the tuple of mode indices that will be excited and monitored by this port.

        Resolution order:

        1. If ``mode_index`` is set (deprecated), return ``(mode_index,)``.
        2. If ``mode_selection`` is set, return that tuple directly.
        3. Otherwise, fall back to ``range(mode_spec.num_modes)``.

        Parameters
        ----------
        mode_spec : MicrowaveModeSpecType, optional
            Resolved mode specification whose ``num_modes`` is an integer.
            When ``None``, falls back to ``self._mode_spec``; raises
            ``SetupError`` if ``num_modes`` is still ``'auto'``.

        Returns
        -------
        tuple[int, ...]
            Ordered mode indices for this port.
        """
        if self.mode_index is not None and self.mode_selection is None:
            return (self.mode_index,)
        if self.mode_selection is not None:
            return self.mode_selection

        mode_spec = self._validate_resolved_mode_spec(mode_spec)
        return tuple(range(mode_spec.num_modes))

    @cached_property
    def _mode_spec(self) -> Optional[MicrowaveModeSpec]:
        """Mode specification for the port. Return None if it cannot be resolved in WavePort alone."""
        num_modes = self.mode_spec.num_modes
        # 1) num_modes already specified
        if num_modes != "auto":
            return self.mode_spec
        # 2) num_modes='auto', and can be refered from the size of impedance_specs
        impedance_specs = self.mode_spec.impedance_specs
        if isinstance(impedance_specs, (list, tuple)):
            return self.mode_spec.updated_copy(num_modes=len(impedance_specs))
        # 3) num_modes='auto', and cannot be refered from the size of impedance_specs
        return None

    def _mode_spec_from_isolated_floating_conductors(
        self, conductors: dict[str, tuple[Shapely, Box]]
    ) -> MicrowaveModeSpec:
        """Update num_modes from the number of isolated floating conductors."""
        return self.mode_spec.updated_copy(num_modes=len(conductors))

    @field_validator("mode_spec", mode="after")
    @classmethod
    def _validate_path_integrals_within_port(
        cls, val: MicrowaveModeSpec, info: ValidationInfo
    ) -> MicrowaveModeSpec:
        """Validate that the microwave mode spec contains path specs all within the port bounds."""
        center = info.data.get("center")
        size = info.data.get("size")
        self_plane = Box(size=size, center=center)
        try:
            val._check_path_integrals_within_box(self_plane)
        except SetupError as e:
            raise SetupError(
                f"Failed to setup '{cls.__name__}' with the suppled 'MicrowaveModeSpec'. {e!s}"
            ) from e
        return val

    @model_validator(mode="after")
    def _validate_mode_selection(self) -> Self:
        """Validate that mode_selection contains valid, unique indices within range.
        if mode_spec.num_modes is 'auto', it'll be validated in the component modeler.
        """
        if self.mode_spec is None:
            return self
        val = self.mode_selection
        if val is None:
            return self

        indices = val

        # Check for non-negative integers
        if any(idx < 0 for idx in indices):
            self._raise_validation_error_at_loc(
                f"'mode_selection' must contain non-negative integers. Got: {indices}",
                "mode_selection",
            )

        # Check for duplicates
        if len(indices) != len(set(indices)):
            duplicates = [idx for idx in set(indices) if list(indices).count(idx) > 1]
            self._raise_validation_error_at_loc(
                f"'mode_selection' contains duplicate entries: {duplicates}. "
                "Each index must appear only once.",
                "mode_selection",
            )

        # Check that indices are within range of num_modes
        mode_spec = self.mode_spec
        if mode_spec.num_modes == "auto":
            return self
        invalid_indices = [idx for idx in self.mode_selection if idx >= mode_spec.num_modes]
        if invalid_indices:
            self._raise_validation_error_at_loc(
                f"'mode_selection' contains indices {invalid_indices} that are >= "
                f"'mode_spec.num_modes' ({mode_spec.num_modes}). "
                f"Valid range is 0 to {mode_spec.num_modes - 1}.",
                "mode_selection",
            )

        return self

    @field_validator("mode_index", mode="after")
    @classmethod
    def _mode_index_deprecated(cls, val: Optional[NonNegativeInt]) -> Optional[NonNegativeInt]:
        """Warn that 'mode_index' is deprecated in favor of 'mode_selection'."""
        if val is not None:
            log.warning(
                "'mode_index' is deprecated and will be removed in future versions. "
                "Please use 'mode_selection' instead."
            )
        return val

    @model_validator(mode="after")
    def _validate_mode_index(self) -> Self:
        """Validate that mode_index is within the valid range.
        if mode_spec.num_modes is 'auto', it'll be validated in the component modeler.
        """
        val = self.mode_index
        if val is None:
            return self
        if self.mode_spec is None or self.mode_spec.num_modes == "auto":
            return self
        if val >= self.mode_spec.num_modes:
            self._raise_validation_error_at_loc(
                f"'mode_index' is >= "
                f"'mode_spec.num_modes' ({self.mode_spec.num_modes}). "
                f"Valid range is 0 to {self.mode_spec.num_modes - 1}.",
                "mode_index",
            )
        return self

    @model_validator(mode="after")
    def _warn_multimode_absorber(self) -> Self:
        """Warn when absorber is enabled with multiple modes.
        if mode_spec.num_modes is 'auto', it'll be validated in the component modeler.
        """
        if self.mode_spec.num_modes != "auto":
            self._warn_resolved_multimode_absorber(self.mode_spec)
        return self

    def _validate_resolved_mode(self, mode_spec: MicrowaveModeSpecType) -> None:
        """Validate mode-field bounds against a resolved mode spec."""
        self._validate_resolved_mode_selection_bounds(mode_spec)
        self._validate_resolved_mode_index_bounds(mode_spec)

    def _validate_resolved_mode_selection_bounds(
        self,
        mode_spec: MicrowaveModeSpecType,
        field_name: str = "mode_selection",
        include_port_name: bool = False,
    ) -> None:
        """Validate ``mode_selection`` against a resolved mode specification."""
        if self.mode_selection is None:
            return

        invalid_indices = [idx for idx in self.mode_selection if idx >= mode_spec.num_modes]
        if not invalid_indices:
            return

        port_suffix = f" for port '{self.name}'" if include_port_name else ""
        raise ValidationError(
            f"'{field_name}' contains indices {invalid_indices} that are >= "
            f"'mode_spec.num_modes' ({mode_spec.num_modes}){port_suffix}. "
            f"Valid range is 0 to {mode_spec.num_modes - 1}."
        )

    def _validate_resolved_mode_index_bounds(
        self, mode_spec: MicrowaveModeSpecType, include_port_name: bool = False
    ) -> None:
        """Validate ``mode_index`` against a resolved mode specification."""
        if self.mode_index is None or self.mode_index < mode_spec.num_modes:
            return

        port_suffix = f" for port '{self.name}'" if include_port_name else ""
        raise ValidationError(
            f"'mode_index' is >= "
            f"'mode_spec.num_modes' ({mode_spec.num_modes}){port_suffix}. "
            f"Valid range is 0 to {mode_spec.num_modes - 1}."
        )

    def to_source(
        self,
        source_time: SourceTimeType,
        snap_center: Optional[float] = None,
        mode_index: int = 0,
        mode_spec: Optional[MicrowaveModeSpecType] = None,
    ) -> ModeSource:
        """Create a mode source from the wave port.

        Parameters
        ----------
        source_time : SourceTimeType
            Source time specification.
        snap_center : float, optional
            Position to snap the source center to along injection axis.
        mode_index : int
            Mode index to inject.
        mode_spec : MicrowaveModeSpecType, optional
            Resolved mode specification with integer num_modes. If None,
            uses self._mode_spec but raises SetupError if num_modes='auto'.
        """
        center = list(self.center)
        if snap_center:
            center[self.injection_axis] = snap_center
        # Use provided mode_spec if given, otherwise fall back to self._mode_spec
        mode_spec = self._validate_resolved_mode_spec(mode_spec)
        return ModeSource(
            center=center,
            size=self.size,
            source_time=source_time,
            mode_spec=mode_spec,
            mode_index=mode_index,
            direction=self.direction,
            name=self.name,
            frame=self.frame,
            use_colocated_integration=False,
        )

    def get_characteristic_impedance_matrix(
        self, sim_mode_data: Union[SimulationData, MicrowaveModeData]
    ) -> ImpedanceFreqModeModeDataArray:
        """Retrieve the characteristic impedance matrix of the port."""
        mode_data = self._get_mode_data(sim_mode_data)
        return mode_data.transmission_line_data.Z0_matrix

    def compute_voltage(self, sim_data: SimulationData) -> FreqModeDataArray:
        """Helper to compute voltage across the port."""
        mode_data: MicrowaveModeData = sim_data[self._mode_monitor_name]
        voltage_coeffs = mode_data.transmission_line_data.voltage_coeffs
        amps = mode_data.amps
        fwd_amps = amps.sel(direction="+").squeeze()
        bwd_amps = amps.sel(direction="-").squeeze()
        return voltage_coeffs * (fwd_amps + bwd_amps)

    def compute_current(self, sim_data: SimulationData) -> FreqModeDataArray:
        """Helper to compute current flowing through the port."""
        mode_data: MicrowaveModeData = sim_data[self._mode_monitor_name]
        current_coeffs = mode_data.transmission_line_data.current_coeffs
        amps = mode_data.amps
        fwd_amps = amps.sel(direction="+").squeeze()
        bwd_amps = amps.sel(direction="-").squeeze()
        # In ModeData, fwd_amps and bwd_amps are not relative to
        # the direction fields are stored
        sign = 1.0
        if self.direction == "-":
            sign = -1.0
        return sign * current_coeffs * (fwd_amps - bwd_amps)

    def get_port_impedance(
        self, sim_mode_data: Union[SimulationData, MicrowaveModeData], mode_index: int
    ) -> FreqModeDataArray:
        """Retrieve the reference impedance of the port for a specific mode.

        Returns the diagonal element of the reference impedance matrix for the given
        mode. When ``reference_impedance`` is set to ``"Z0"``, this equals the
        characteristic impedance; otherwise it returns the user-specified reference
        impedance.

        Parameters
        ----------
        sim_mode_data : Union[:class:`.SimulationData`, :class:`.MicrowaveModeData`]
            Simulation data containing the mode monitor results, or the mode data directly.
            If :class:`.SimulationData` is provided, the mode data is extracted using the
            port's mode monitor name.
        mode_index : int
            Index of the mode for which to compute the impedance. This selects a specific
            mode from the mode spectrum computed by the mode solver.

        Returns
        -------
        :class:`.FreqModeDataArray`
            Frequency-dependent reference impedance for the specified mode.
            The impedance is complex-valued and varies with frequency.
        """
        reference_impedance_matrix = self.get_reference_impedance_matrix(sim_mode_data)
        # Select diagonal element and reshape to (f, mode_index) format
        Z0_selected = reference_impedance_matrix.sel(
            mode_index_out=mode_index, mode_index_in=mode_index
        )
        # Expand dims to add mode_index dimension and cast to FreqModeDataArray
        return FreqModeDataArray(Z0_selected.expand_dims(mode_index=[mode_index]))


class TerminalWavePort(AbstractWavePort):
    """Class representing a single terminal-driven wave port.

    Notes
    -----
    - By default, the terminals are single-ended, specified by ``terminal_specs`` parameters. They are
      labeled by ``T0``, ``T1``, ..., ``Tn``, where the order is defined by their order in ``terminal_specs``
      if it's a tuple/list, or their location from left to right and bottom to top if ``terminal_specs``
      is an ``AutoImpedanceSpec``.
    - Differential pairs are defined by selecting a pair of single-ended terminals based on their labels.
      The differential pair itself is labeled by "Diff0@comm", "Diff0@diff", "Diff1@comm", "Diff1@diff", ...,
      where the order is defined by their order in ``differential_pairs``.
    - The terminals are ordered so that the single-ended terminal labels come first, followed by differential pairs.
    - By default, a reference impedance of 50 Ohm is used for S-parameter calculations unless otherwise specified.
    """

    reference_impedance: Union[Literal["Z0"], Complex, ImpedanceTerminalDataArray] = Field(
        "Z0",
        title="Reference Impedance",
        description="User-specified reference impedance for S-parameter computation. "
        "If ``Z0``, the characteristic impedance "
        "is used. Otherwise, it can be a single complex value applied to all terminals, or a data array "
        f"specified for each terminal. If the data array misses some terminals, {DEFAULT_REFERENCE_IMPEDANCE_VALUE} "
        f"Ohm is applied to the missing terminals. By default, {DEFAULT_REFERENCE_IMPEDANCE_VALUE} Ohm is used.",
        json_schema_extra={"units": OHM},
    )

    absorber: Union[bool, ABCBoundary, ModeABCBoundary] = Field(
        False,
        title="Absorber",
        description="Place a mode absorber in the port. If ``True``, an automatically generated mode absorber is placed in the port. "
        "If :class:`.ABCBoundary` or :class:`.ModeABCBoundary`, a mode absorber is placed in the port with the specified boundary conditions.",
    )

    terminal_specs: Union[
        AutoImpedanceSpec,
        tuple[CustomImpedanceSpec, ...],
    ] = Field(
        default_factory=AutoImpedanceSpec._default_without_license_warning,
        title="Terminal Specification",
        description="Parameters to feed to terminal solver which determine single-ended terminals "
        "and how transmission line "
        "quantities for each single-ended terminal, e.g., charateristic impedance, are computed.",
    )

    differential_pairs: tuple[tuple[str, str], ...] = Field(
        (),
        title="Differential Pair",
        description="Differential pairs defined by a pair of single-ended terminals based "
        "on their labels, which can be "
        "found out through :class:`~tidy3d.plugins.smatrix.TerminalComponentModeler`.plot_port() method. "
        "In each pair, the first termial is positive, while the second is negative.",
    )

    @field_validator("differential_pairs", mode="after")
    @classmethod
    def _validate_differential_pairs(
        cls, val: tuple[tuple[str, str], ...]
    ) -> tuple[tuple[str, str], ...]:
        """Validate no duplicate terminals in differential pairs."""
        # Check for duplicates - flatten all terminal labels from all pairs
        terminals_present = set()
        terminals_duplicates = set()
        for pair in val:
            for terminal_label in pair:
                if terminal_label in terminals_present:
                    terminals_duplicates.add(terminal_label)
                terminals_present.add(terminal_label)

        if terminals_duplicates:
            raise ValidationError(
                f"Terminal labels {sorted(terminals_duplicates)} appear more than once in differential_pairs. "
                "Each terminal can only be used in one differential pair."
            )
        return val

    @model_validator(mode="after")
    def _validate_terminal_specs(self) -> Self:
        """Validate terminal_specs: if it's a list of CustomImpedanceSpec, validate that current and voltage specs
        are defined consistently: if one of them is None, it must be None for all CustomImpedanceSpec in the list.
        """
        val = self.terminal_specs
        # Skip validation for AutoImpedanceSpec
        if not isinstance(val, (tuple, list)):
            return self

        # Check for empty tuple/list
        if len(val) == 0:
            self._raise_validation_error_at_loc(
                ValidationError(
                    "Empty 'terminal_specs' tuple is not allowed. "
                    "Please provide at least one CustomImpedanceSpec, or use AutoImpedanceSpec "
                    "for automatic terminal detection."
                ),
                "terminal_specs",
            )

        # Check consistency of voltage_spec and current_spec across all CustomImpedanceSpec
        has_voltage_spec = []
        has_current_spec = []

        for spec in val:
            has_voltage_spec.append(spec.voltage_spec is not None)
            has_current_spec.append(spec.current_spec is not None)

        # Check if voltage_spec consistency: all None or all not None
        if not all(has_voltage_spec) and any(has_voltage_spec):
            self._raise_validation_error_at_loc(
                ValidationError(
                    "Inconsistent voltage specifications in terminal_specs: "
                    "If voltage_spec is defined for one terminal, it must be defined for all terminals."
                ),
                "terminal_specs",
            )

        # Check if current_spec consistency: all None or all not None
        if not all(has_current_spec) and any(has_current_spec):
            self._raise_validation_error_at_loc(
                ValidationError(
                    "Inconsistent current specifications in terminal_specs: "
                    "If current_spec is defined for one terminal, it must be defined for all terminals."
                ),
                "terminal_specs",
            )

        return self

    @cached_property
    def _mode_spec(self) -> Optional[MicrowaveTerminalModeSpec]:
        """Mode specification for the port if it can be resolved in this module;
        otherwise, return None.
        """
        # 1) in auto mode, needs more information to be determined
        if not isinstance(self.terminal_specs, (list, tuple)):
            return None
        # 2) manual definition
        num_modes = len(self.terminal_specs)
        terminal_labels = [f"{DEFAULT_TERMINAL_LABEL_PREFIX}{i}" for i in range(num_modes)]
        impedance_specs = dict(zip(terminal_labels, self.terminal_specs))
        terminals_mapping = self._get_terminals_mapping(terminal_labels)
        mode_spec = MicrowaveTerminalModeSpec(
            impedance_specs=impedance_specs,
            num_modes=num_modes,
            terminals_mapping=terminals_mapping,
        )
        return mode_spec

    def _mode_spec_from_isolated_floating_conductors(
        self, conductors: dict[str, tuple[Shapely, Box]]
    ) -> MicrowaveTerminalModeSpec:
        """Construct mode specification from isolated floating conductors
        when terminal_specs is an AutoImpedanceSpec.
        """
        terminal_labels = list(conductors)
        impedance_specs = {
            label: CustomImpedanceSpec.from_bounding_box(box, current_sign=self.direction)
            for label, (_, box) in conductors.items()
        }
        terminals_mapping = self._get_terminals_mapping(terminal_labels)
        return MicrowaveTerminalModeSpec(
            impedance_specs=impedance_specs,
            num_modes=len(terminal_labels),
            terminals_mapping=terminals_mapping,
        )

    def _mode_indices(self, mode_spec: Optional[MicrowaveModeSpecType] = None) -> tuple[int, ...]:
        """Return the tuple of mode indices that will be excited and monitored by this port.

        All terminal modes are always included, so this returns
        ``range(mode_spec.num_modes)`` (one index per terminal).

        Parameters
        ----------
        mode_spec : MicrowaveModeSpecType, optional
            Resolved mode specification whose ``num_modes`` is an integer.
            When ``None``, falls back to ``self._mode_spec``; raises
            ``SetupError`` if ``num_modes`` is still ``'auto'``.

        Returns
        -------
        tuple[int, ...]
            Ordered mode indices for this port.
        """
        mode_spec = self._validate_resolved_mode_spec(mode_spec)
        return tuple(range(mode_spec.num_modes))

    @cached_property
    def _differential_pair_mapping(self) -> dict[str, tuple[str, str]]:
        """Get a mapping from differential pairs to single-ended terminals.

        Returns
        -------
        dict[str, tuple[str, str]]: Mapping from differential pairs to single-ended terminals.
        """
        mapping = {}
        for pair_idx, (label1, label2) in enumerate(self.differential_pairs):
            # Create labels for common mode and differential mode
            comm_label = f"{DEFAULT_DIFFERENTIAL_PAIR_LABEL_PREFIX}{pair_idx}@comm"
            diff_label = f"{DEFAULT_DIFFERENTIAL_PAIR_LABEL_PREFIX}{pair_idx}@diff"

            # Map both modes to the same pair of single-ended terminals
            mapping[comm_label] = (label1, label2)
            mapping[diff_label] = (label1, label2)

        return mapping

    def _get_active_single_ended_terminals(self, single_ended_labels: list[str]) -> list[str]:
        """Get the single-ended terminals for S-parameter computation (excluding the
        ones used in differential pairs).

        Parameters
        ----------
        single_ended_labels : list[str]
            Labels of the single-ended terminals.

        Returns
        -------
        list[str]
            List of single-ended terminal labels not used in any differential pair.
        """
        active_terminals = single_ended_labels.copy()
        for pair_idx, pair in enumerate(self.differential_pairs):
            for label in pair:
                try:
                    active_terminals.remove(label)
                except ValueError:
                    raise SetupError(
                        f"Port '{self.name}': Differential pair {pair_idx} references terminal "
                        f"label '{label}' that is not present in the available single-ended "
                        f"terminals: {single_ended_labels}. "
                        f"Please ensure all differential pair labels match detected terminals."
                    ) from None
        return active_terminals

    def _get_terminals_mapping(
        self, single_ended_labels: list[str]
    ) -> dict[str, Union[str, tuple[str, str]]]:
        """Get a mapping from terminals (single-ended terminals or differential pairs) to single-ended terminals.

        The terminals are ordered such that single-ended terminal labels come first, followed by
        differential pairs. This ordering ensures consistent indexing in transformation matrices.

        Parameters
        ----------
        single_ended_labels : list[str]
            Labels of the single-ended terminals.

        Returns
        -------
        dict[str, Union[str, tuple[str, str]]]:
            Mapping from terminal (single-ended terminal or differential pair) to single-ended terminals.
            Keys are ordered with single-ended terminals first, followed by differential pairs.
        """
        # Start with single-ended terminals (these come first in the ordering)
        mapping = {
            label: label for label in self._get_active_single_ended_terminals(single_ended_labels)
        }
        # Then add differential pairs (these come after single-ended terminals)
        mapping.update(self._differential_pair_mapping)
        return mapping

    def to_source(
        self,
        source_time: SourceTimeType,
        snap_center: Optional[float] = None,
        terminal_label: Optional[str] = None,
        mode_spec: Optional[MicrowaveModeSpecType] = None,
    ) -> MicrowaveTerminalSource:
        """Create a microwave terminal source from the wave port.

        Parameters
        ----------
        source_time : SourceTimeType
            Source time specification.
        snap_center : float, optional
            Position to snap the source center to along injection axis.
        terminal_label : str, optional
            Terminal label to inject. If None, uses the first terminal in mode_spec.
        mode_spec : MicrowaveModeSpecType, optional
            Resolved mode specification with integer num_modes. If None,
            uses self._mode_spec but raises SetupError if num_modes='auto'.
        """
        center = list(self.center)
        if snap_center:
            center[self.injection_axis] = snap_center
        # Use provided mode_spec if given, otherwise fall back to self._mode_spec
        mode_spec = self._validate_resolved_mode_spec(mode_spec)
        if terminal_label is None:
            terminal_label = mode_spec._terminal_indices[0]
        return MicrowaveTerminalSource(
            center=center,
            size=self.size,
            source_time=source_time,
            mode_spec=mode_spec,
            terminal_label=terminal_label,
            direction=self.direction,
            name=self.name,
            frame=self.frame,
            use_colocated_integration=False,
        )

    def get_characteristic_impedance_matrix(
        self, sim_mode_data: Union[SimulationData, MicrowaveModeData]
    ) -> ImpedanceFreqTerminalTerminalDataArray:
        """Retrieve the characteristic impedance matrix of the port."""
        mode_data = self._get_mode_data(sim_mode_data)
        return mode_data.transmission_line_terminal_data.Z0_matrix

    def compute_voltage(self, sim_data: SimulationData) -> VoltageFreqTerminalDataArray:
        """Helper to compute voltage across the port."""
        mode_data: MicrowaveModeData = sim_data[self._mode_monitor_name]
        voltage_transform = mode_data.transmission_line_terminal_data.voltage_transform
        amps = mode_data.amps
        fwd_amps = amps.sel(direction="+").squeeze()
        bwd_amps = amps.sel(direction="-").squeeze()
        # Matrix multiply: voltage_transform[f, terminal_label, mode_index] @ amps[f, mode_index]
        # to get voltage[f, terminal_label]
        return voltage_transform.dot(fwd_amps + bwd_amps, dim="mode_index")

    def compute_current(self, sim_data: SimulationData) -> CurrentFreqTerminalDataArray:
        """Helper to compute current flowing through the port."""
        mode_data: MicrowaveModeData = sim_data[self._mode_monitor_name]
        current_transform = mode_data.transmission_line_terminal_data.current_transform
        amps = mode_data.amps
        fwd_amps = amps.sel(direction="+").squeeze()
        bwd_amps = amps.sel(direction="-").squeeze()
        # In ModeData, fwd_amps and bwd_amps are not relative to
        # the direction fields are stored
        sign = 1.0
        if self.direction == "-":
            sign = -1.0
        return sign * current_transform.dot(fwd_amps - bwd_amps, dim="mode_index")
