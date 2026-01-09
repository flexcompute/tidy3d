"""Class for representing a scattering matrix wave port."""

from __future__ import annotations

from typing import Optional, Union

import pydantic.v1 as pd

from tidy3d.components.base import cached_property, skip_if_fields_missing
from tidy3d.components.boundary import ABCBoundary, InternalAbsorber, ModeABCBoundary
from tidy3d.components.data.data_array import FreqDataArray, FreqModeDataArray
from tidy3d.components.data.sim_data import SimulationData
from tidy3d.components.geometry.base import Box
from tidy3d.components.grid.grid import Grid
from tidy3d.components.microwave.data.monitor_data import MicrowaveModeData
from tidy3d.components.microwave.mode_spec import MicrowaveModeSpec
from tidy3d.components.microwave.monitor import MicrowaveModeMonitor
from tidy3d.components.simulation import Simulation
from tidy3d.components.source.field import ModeSource
from tidy3d.components.source.frame import PECFrame
from tidy3d.components.source.time import GaussianPulse
from tidy3d.components.structure import MeshOverrideStructure
from tidy3d.components.types import Axis, Direction, FreqArray
from tidy3d.exceptions import SetupError, ValidationError
from tidy3d.log import log
from tidy3d.plugins.mode import ModeSolver
from tidy3d.plugins.smatrix.ports.base_terminal import AbstractTerminalPort

DEFAULT_WAVE_PORT_NUM_CELLS = 5
MIN_WAVE_PORT_NUM_CELLS = 3
DEFAULT_WAVE_PORT_FRAME = PECFrame()


class WavePort(AbstractTerminalPort, Box):
    """Class representing a single wave port"""

    direction: Direction = pd.Field(
        ...,
        title="Direction",
        description="'+' or '-', defining which direction is considered 'input'.",
    )

    mode_spec: MicrowaveModeSpec = pd.Field(
        default_factory=MicrowaveModeSpec._default_without_license_warning,
        title="Mode Specification",
        description="Parameters to feed to mode solver which determine modes and how transmission line "
        "quantities, e.g., charateristic impedance, are computed.",
    )

    num_grid_cells: Optional[int] = pd.Field(
        DEFAULT_WAVE_PORT_NUM_CELLS,
        ge=MIN_WAVE_PORT_NUM_CELLS,
        title="Number of Grid Cells",
        description="Number of mesh grid cells in the transverse plane of the `WavePort`. "
        "Used in generating the suggested list of :class:`.MeshOverrideStructure` objects. "
        "Must be greater than or equal to 3. When set to `None`, no grid refinement is performed.",
    )

    conjugated_dot_product: bool = pd.Field(
        False,
        title="Conjugated Dot Product",
        description="Use conjugated or non-conjugated dot product for mode decomposition.",
    )

    frame: Optional[PECFrame] = pd.Field(
        DEFAULT_WAVE_PORT_FRAME,
        title="Source Frame",
        description="Add a thin frame around the source during FDTD run for an improved injection.",
    )

    absorber: Union[bool, ABCBoundary, ModeABCBoundary] = pd.Field(
        True,
        title="Absorber",
        description="Place a mode absorber in the port. If ``True``, an automatically generated mode absorber is placed in the port. "
        "If :class:`.ABCBoundary` or :class:`.ModeABCBoundary`, a mode absorber is placed in the port with the specified boundary conditions.",
    )

    extrude_structures: bool = pd.Field(
        False,
        title="Extrude Structures",
        description="Extrudes structures that intersect the wave port plane by a few grid cells when ``True``, improving mode injection accuracy.",
    )

    mode_index: Optional[pd.NonNegativeInt] = pd.Field(
        None,
        title="Mode Index (deprecated)",
        description="Index into the collection of modes returned by mode solver. "
        "Specifies which mode to inject using this port. "
        "Deprecated. Use the 'mode_selection' field instead.",
    )

    mode_selection: Optional[tuple[int, ...]] = pd.Field(
        None,
        title="Mode Selection",
        description="Selects specific mode(s) to use from the mode solver. "
        "Can be a single integer for one mode, or a tuple of integers for multiple modes. "
        "If ``None`` (default), all modes from the ``mode_spec`` are used. "
        "Indices must be non-negative and less than ``mode_spec.num_modes``.",
    )

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
    def _mode_monitor_name(self) -> str:
        """Return the name of the :class:`.MicrowaveModeMonitor` associated with this port."""
        return f"{self.name}_mode"

    @cached_property
    def _mode_indices(self) -> tuple[int, ...]:
        """Mode indices that will be excited/monitored by this port."""
        if self.mode_index is not None and self.mode_selection is None:
            return (self.mode_index,)
        if self.mode_selection is not None:
            # User specified specific modes
            return self.mode_selection
        # Default: use all modes
        return tuple(range(self.mode_spec.num_modes))

    def to_source(
        self,
        source_time: GaussianPulse,
        snap_center: Optional[float] = None,
        mode_index: int = 0,
    ) -> ModeSource:
        """Create a mode source from the wave port."""
        center = list(self.center)
        if snap_center:
            center[self.injection_axis] = snap_center
        return ModeSource(
            center=center,
            size=self.size,
            source_time=source_time,
            mode_spec=self.mode_spec,
            mode_index=mode_index,
            direction=self.direction,
            name=self.name,
            frame=self.frame,
        )

    def to_monitors(
        self, freqs: FreqArray, snap_center: Optional[float] = None, grid: Grid = None
    ) -> list[MicrowaveModeMonitor]:
        """The wave port uses a :class:`.MicrowaveModeMonitor` to compute the characteristic impedance
        and the port voltages and currents."""
        center = list(self.center)
        if snap_center:
            center[self.injection_axis] = snap_center
        mode_mon = MicrowaveModeMonitor(
            center=self.center,
            size=self.size,
            freqs=freqs,
            name=self._mode_monitor_name,
            colocate=False,
            mode_spec=self.mode_spec,
            store_fields_direction=self.direction,
            conjugated_dot_product=self.conjugated_dot_product,
        )
        return [mode_mon]

    def to_mode_solver(self, simulation: Simulation, freqs: FreqArray) -> ModeSolver:
        """Helper to create a :class:`.ModeSolver` instance."""
        mode_solver = ModeSolver(
            simulation=simulation,
            plane=self.geometry,
            mode_spec=self.mode_spec,
            freqs=freqs,
            direction=self.direction,
            colocate=False,
        )
        return mode_solver

    def to_absorber(
        self, snap_center: Optional[float] = None, freq_spec: Optional[pd.NonNegativeFloat] = None
    ) -> InternalAbsorber:
        """Create an internal absorber from the wave port."""
        center = list(self.center)
        if snap_center:
            center[self.injection_axis] = snap_center
        if isinstance(self.absorber, (ABCBoundary, ModeABCBoundary)):
            boundary_spec = self.absorber
        else:
            # TODO: ModeABCBoundary currently only accepts one mode, so
            # we choose the first mode for now
            mode_index = self._mode_indices[0]
            boundary_spec = ModeABCBoundary(
                mode_spec=self.mode_spec,
                mode_index=mode_index,
                plane=self.geometry,
                freq_spec=freq_spec,
            )
        return InternalAbsorber(
            center=center,
            size=self.size,
            boundary_spec=boundary_spec,
            direction="-"
            if self.direction == "+"
            else "+",  # absorb in the opposite direction of source
            grid_shift=1,  # absorb in the next pixel
        )

    def compute_voltage(self, sim_data: SimulationData) -> FreqDataArray:
        """Helper to compute voltage across the port."""
        mode_data: MicrowaveModeData = sim_data[self._mode_monitor_name]
        voltage_coeffs = mode_data.transmission_line_data.voltage_coeffs
        amps = mode_data.amps
        fwd_amps = amps.sel(direction="+").squeeze()
        bwd_amps = amps.sel(direction="-").squeeze()
        return voltage_coeffs * (fwd_amps + bwd_amps)

    def compute_current(self, sim_data: SimulationData) -> FreqDataArray:
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
        """Retrieve the characteristic impedance of the port for a specific mode.

        The port impedance is computed from the transmission line mode characteristics,
        which should ideally be TEM (Transverse Electromagnetic) or at least quasi-TEM.
        The impedance is extracted from the transmission line data computed by the
        mode solver.

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
            Frequency-dependent characteristic impedance Z0 for the specified mode.
            The impedance is complex-valued and varies with frequency.
        """
        if isinstance(sim_mode_data, SimulationData):
            mode_data = sim_mode_data[self._mode_monitor_name]
        else:
            mode_data = sim_mode_data
        return mode_data.transmission_line_data.Z0.sel(mode_index=[mode_index])

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

    @pd.validator("mode_spec", always=True)
    def _validate_path_integrals_within_port(cls, val, values):
        """Validate that the microwave mode spec contains path specs all within the port bounds."""
        center = values["center"]
        size = values["size"]
        self_plane = Box(size=size, center=center)
        try:
            val._check_path_integrals_within_box(self_plane)
        except SetupError as e:
            raise SetupError(
                f"Failed to setup '{cls.__name__}' with the suppled 'MicrowaveModeSpec'. {e!s}"
            ) from e
        return val

    @pd.validator("mode_selection", always=True)
    @skip_if_fields_missing(["mode_spec"])
    def _validate_mode_selection(cls, val, values):
        """Validate that mode_selection contains valid, unique indices within range."""
        if val is None:
            return val

        indices = val

        # Check for non-negative integers
        if any(idx < 0 for idx in indices):
            raise ValidationError(
                f"'mode_selection' must contain non-negative integers. Got: {indices}"
            )

        # Check for duplicates
        if len(indices) != len(set(indices)):
            duplicates = [idx for idx in set(indices) if list(indices).count(idx) > 1]
            raise ValidationError(
                f"'mode_selection' contains duplicate entries: {duplicates}. "
                "Each index must appear only once."
            )

        # Check that indices are within range of num_modes
        mode_spec = values["mode_spec"]
        num_modes = mode_spec.num_modes
        invalid_indices = [idx for idx in indices if idx >= num_modes]
        if invalid_indices:
            raise ValidationError(
                f"'mode_selection' contains indices {invalid_indices} that are >= "
                f"'mode_spec.num_modes' ({num_modes}). Valid range is 0 to {num_modes - 1}."
            )

        return val

    @pd.root_validator(pre=False)
    def _check_absorber_if_extruding_structures(cls, values):
        """Raise validation error when ``extrude_structures`` is set to ``True``
        while ``absorber`` is set to ``False``."""

        if values.get("extrude_structures") and not values.get("absorber"):
            raise ValidationError(
                "Structure extrusion for a waveport requires an internal absorber. Set `absorber=True` to enable it."
            )

        return values

    @pd.validator("mode_index", always=True)
    def _mode_index_deprecated(cls, val):
        """Warn that 'mode_index' is deprecated in favor of 'mode_selection'."""
        if val is not None:
            log.warning(
                "'mode_index' is deprecated and will be removed in future versions. "
                "Please use 'mode_selection' instead."
            )
        return val

    @pd.validator("mode_index", always=True)
    @skip_if_fields_missing(["mode_spec"])
    def _validate_mode_index(cls, val, values):
        """Validate that mode_selection contains valid, unique indices within range."""
        if val is None:
            return val
        num_modes = values["mode_spec"].num_modes
        if val >= num_modes:
            raise ValidationError(
                f"'mode_index' is >= "
                f"'mode_spec.num_modes' ({num_modes}). Valid range is 0 to {num_modes - 1}."
            )
        return val

    @property
    def _is_using_mesh_refinement(self) -> bool:
        """Check if this wave port is using mesh refinement options.

        Returns ``True`` if a custom grid cell count is specified.
        """
        return self.num_grid_cells is not None
