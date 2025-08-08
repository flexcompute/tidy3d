from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Union

import pydantic.v1 as pd

from tidy3d.components.base import Tidy3dBaseModel, cached_property
from tidy3d.components.data.data_array import FreqDataArray
from tidy3d.components.data.sim_data import SimulationData
from tidy3d.components.grid.grid import Grid
from tidy3d.components.monitor import FieldMonitor, ModeMonitor
from tidy3d.components.source.base import Source
from tidy3d.components.source.time import GaussianPulse
from tidy3d.components.types import FreqArray
from tidy3d.log import log


class AbstractTerminalPort(Tidy3dBaseModel, ABC):
    name: str = pd.Field(..., min_length=1)

    @cached_property
    @abstractmethod
    def injection_axis(self): ...

    @abstractmethod
    def to_source(
        self, source_time: GaussianPulse, snap_center: Optional[float] = None, grid: Grid = None
    ) -> Source: ...

    def to_field_monitors(
        self, freqs: FreqArray, snap_center: Optional[float] = None, grid: Grid = None
    ) -> Union[list[FieldMonitor], list[ModeMonitor]]:
        log.warning(
            "'to_field_monitors' method name is deprecated and will be removed in the future. Please use 'to_monitors' for the same effect."
        )
        return self.to_monitors(freqs=freqs, snap_center=snap_center, grid=grid)

    @abstractmethod
    def to_monitors(
        self, freqs: FreqArray, snap_center: Optional[float] = None, grid: Grid = None
    ) -> Union[list[FieldMonitor], list[ModeMonitor]]: ...

    @abstractmethod
    def compute_voltage(self, sim_data: SimulationData) -> FreqDataArray: ...

    @abstractmethod
    def compute_current(self, sim_data: SimulationData) -> FreqDataArray: ...

    @pd.root_validator(pre=False)
    def _warn_rf_license(cls, values):
        log.warning(
            "ℹ️ ⚠️ RF simulations are subject to new license requirements in the future. You have instantiated at least one RF-specific component.",
            log_once=True,
        )
        return values
