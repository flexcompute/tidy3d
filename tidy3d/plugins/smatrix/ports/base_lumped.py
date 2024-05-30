"""Class and custom data array for representing a scattering matrix port based on lumped circuit elements."""

import pydantic.v1 as pd
from typing import Optional
from abc import abstractmethod

from ....constants import OHM
from ....components.types import Complex, FreqArray
from ....components.base import cached_property
from ....components.grid.grid import Grid, YeeGrid
from ....components.lumped_element import AbstractLumpedResistor
from ....components.monitor import FieldMonitor
from ....components.data.sim_data import SimulationData
from ....components.source import UniformCurrentSource, GaussianPulse
from ....components.data.data_array import FreqDataArray

from .base_terminal import AbstractTerminalPort

DEFAULT_PORT_NUM_CELLS = 3
DEFAULT_REFERENCE_IMPEDANCE = 50


class AbstractLumpedPort(AbstractTerminalPort):
    """Class representing a single lumped port"""

    impedance: Complex = pd.Field(
        DEFAULT_REFERENCE_IMPEDANCE,
        title="Reference impedance",
        description="Reference port impedance for scattering parameter computation.",
        units=OHM,
    )

    num_grid_cells: Optional[pd.PositiveInt] = pd.Field(
        DEFAULT_PORT_NUM_CELLS,
        title="Port grid cells",
        description="Number of mesh grid cells associated with the port along each direction, "
        "which are added through automatic mesh refinement. "
        "A value of ``None`` will turn off automatic mesh refinement.",
    )

    @cached_property
    def _voltage_monitor_name(self) -> str:
        return f"{self.name}_voltage"

    @cached_property
    def _current_monitor_name(self) -> str:
        return f"{self.name}_current"

    @cached_property
    @abstractmethod
    def injection_axis(self):
        """Injection axis of the port."""

    @abstractmethod
    def to_source(
        self, source_time: GaussianPulse, snap_center: float = None, grid: Grid = None
    ) -> UniformCurrentSource:
        """Create a current source from the lumped port."""

    @abstractmethod
    def to_load(self, snap_center: float = None) -> AbstractLumpedResistor:
        """Create a load resistor from the lumped port."""

    @abstractmethod
    def to_voltage_monitor(self, freqs: FreqArray, snap_center: float = None) -> FieldMonitor:
        """Field monitor to compute port voltage."""

    @abstractmethod
    def to_current_monitor(self, freqs: FreqArray, snap_center: float = None) -> FieldMonitor:
        """Field monitor to compute port current."""

    def to_field_monitors(self, freqs: FreqArray, snap_center: float = None) -> list[FieldMonitor]:
        """Field monitors to compute port voltage and current."""
        return [
            self.to_voltage_monitor(freqs, snap_center),
            self.to_current_monitor(freqs, snap_center),
        ]

    @abstractmethod
    def compute_voltage(self, sim_data: SimulationData) -> FreqDataArray:
        """Helper to compute voltage across the port."""

    @abstractmethod
    def compute_current(self, sim_data: SimulationData) -> FreqDataArray:
        """Helper to compute current flowing through the port."""

    @abstractmethod
    def _check_grid_size(yee_grid: YeeGrid):
        """Raises :class:``SetupError`` if the grid is too coarse at port locations"""
