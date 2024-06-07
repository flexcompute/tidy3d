"""Class and custom data array for representing a scattering matrix port based on lumped circuit elements."""

from abc import ABC, abstractmethod
from typing import Optional

import pydantic.v1 as pd

from ....components.base import Tidy3dBaseModel, cached_property
from ....components.data.data_array import DataArray, FreqDataArray
from ....components.data.sim_data import SimulationData
from ....components.grid.grid import Grid, YeeGrid
from ....components.lumped_element import AbstractLumpedResistor
from ....components.monitor import FieldMonitor
from ....components.source import GaussianPulse, UniformCurrentSource
from ....components.types import Complex, FreqArray
from ....constants import OHM

DEFAULT_PORT_NUM_CELLS = 3
DEFAULT_REFERENCE_IMPEDANCE = 50


class LumpedPortDataArray(DataArray):
    """Port parameter matrix elements for lumped ports.

    Example
    -------
    >>> import numpy as np
    >>> ports_in = ['port1', 'port2']
    >>> ports_out = ['port1', 'port2']
    >>> f = [2e14]
    >>> coords = dict(
    ...     port_in=ports_in,
    ...     port_out=ports_out,
    ...     f=f
    ... )
    >>> fd = LumpedPortDataArray((1 + 1j) * np.random.random((2, 2, 1)), coords=coords)
    """

    __slots__ = ()
    _dims = ("port_out", "port_in", "f")
    _data_attrs = {"long_name": "lumped port matrix element"}


class AbstractLumpedPort(Tidy3dBaseModel, ABC):
    """Class representing a single lumped port"""

    name: str = pd.Field(
        ...,
        title="Name",
        description="Unique name for the port.",
        min_length=1,
    )

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
        self, source_time: GaussianPulse, snap_center: float, grid=Grid
    ) -> UniformCurrentSource:
        """Create a current source from the lumped port."""

    @abstractmethod
    def to_load(self, snap_center: float) -> AbstractLumpedResistor:
        """Create a load resistor from the lumped port."""

    @abstractmethod
    def to_voltage_monitor(self, freqs: FreqArray, snap_center: float) -> FieldMonitor:
        """Field monitor to compute port voltage."""

    @abstractmethod
    def to_current_monitor(self, freqs: FreqArray, snap_center: float) -> FieldMonitor:
        """Field monitor to compute port current."""

    @abstractmethod
    def compute_voltage(self, sim_data: SimulationData) -> FreqDataArray:
        """Helper to compute voltage across the port."""

    @abstractmethod
    def compute_current(self, sim_data: SimulationData) -> FreqDataArray:
        """Helper to compute current flowing through the port."""

    @abstractmethod
    def _check_grid_size(yee_grid: YeeGrid):
        """Raises :class:``SetupError`` if the grid is too coarse at port locations"""
