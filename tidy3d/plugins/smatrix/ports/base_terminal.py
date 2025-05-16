"""Class and custom data array for representing a scattering-matrix port, which is defined by a pair of terminals."""

from abc import ABC, abstractmethod
from typing import Union

import pydantic.v1 as pd

from tidy3d.log import log

from ....components.base import Tidy3dBaseModel, cached_property
from ....components.data.data_array import FreqDataArray
from ....components.data.sim_data import SimulationData
from ....components.grid.grid import Grid
from ....components.monitor import FieldMonitor, ModeMonitor
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

    def to_field_monitors(
        self, freqs: FreqArray, snap_center: float = None, grid: Grid = None
    ) -> Union[list[FieldMonitor], list[ModeMonitor]]:
        """DEPRECATED: Monitors used to compute the port voltage and current."""
        log.warning(
            "'to_field_monitors' method name is deprecated and will be removed in the future. Please use "
            "'to_monitors' for the same effect."
        )
        return self.to_monitors(freqs=freqs, snap_center=snap_center, grid=grid)

    @abstractmethod
    def to_monitors(
        self, freqs: FreqArray, snap_center: float = None, grid: Grid = None
    ) -> Union[list[FieldMonitor], list[ModeMonitor]]:
        """Monitors used to compute the port voltage and current."""

    @abstractmethod
    def compute_voltage(self, sim_data: SimulationData) -> FreqDataArray:
        """Helper to compute voltage across the port."""

    @abstractmethod
    def compute_current(self, sim_data: SimulationData) -> FreqDataArray:
        """Helper to compute current flowing into the port."""

    @pd.root_validator(pre=False)
    def _warn_rf_license(cls, values):
        log.warning(
            "ℹ️ ⚠️ RF simulations are subject to new license requirements in the future. You have instantiated at least one RF-specific component.",
            log_once=True,
        )
        return values
