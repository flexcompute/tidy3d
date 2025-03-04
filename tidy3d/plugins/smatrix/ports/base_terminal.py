"""Class and custom data array for representing a scattering-matrix port, which is defined by a pair of terminals."""

from abc import ABC, abstractmethod

import pydantic.v1 as pd

from ....components.base import Tidy3dBaseModel, cached_property
from ....components.data.data_array import FreqDataArray
from ....components.data.sim_data import SimulationData
from ....components.grid.grid import Grid
from ....components.monitor import FieldMonitor
from ....components.source.base import Source
from ....components.source.time import GaussianPulse
from ....components.types import FreqArray


class AbstractTerminalPort(Tidy3dBaseModel, ABC):
    """Class representing a single terminal-based port. All terminal ports must provide methods
    for computing voltage and current. These quantities represent the voltage between the
    terminals, and the current flowing from one terminal into the other.
    """

    name: str = pd.Field(
        ...,
        title="Name",
        description="Unique name for the port.",
        min_length=1,
    )

    @cached_property
    @abstractmethod
    def injection_axis(self):
        """Injection axis of the port."""

    @abstractmethod
    def to_source(
        self, source_time: GaussianPulse, snap_center: float = None, grid: Grid = None
    ) -> Source:
        """Create a current source from a terminal-based port."""

    @abstractmethod
    def to_field_monitors(
        self, freqs: FreqArray, snap_center: float = None, grid: Grid = None
    ) -> list[FieldMonitor]:
        """Field monitors to compute port voltage and current."""

    @abstractmethod
    def compute_voltage(self, sim_data: SimulationData) -> FreqDataArray:
        """Helper to compute voltage across the port."""

    @abstractmethod
    def compute_current(self, sim_data: SimulationData) -> FreqDataArray:
        """Helper to compute current flowing into the port."""
