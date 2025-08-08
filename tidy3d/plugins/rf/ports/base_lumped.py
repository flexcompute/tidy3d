from __future__ import annotations

from abc import abstractmethod
from typing import Optional

import pydantic.v1 as pd

from tidy3d.components.base import cached_property
from tidy3d.components.geometry.utils_2d import snap_coordinate_to_grid
from tidy3d.components.grid.grid import Grid, YeeGrid
from tidy3d.components.lumped_element import LumpedElementType
from tidy3d.components.monitor import FieldMonitor
from tidy3d.components.types import Complex, Coordinate, FreqArray
from tidy3d.constants import OHM
from tidy3d.plugins.rf.ports.base_terminal import AbstractTerminalPort

DEFAULT_PORT_NUM_CELLS = 3
DEFAULT_REFERENCE_IMPEDANCE = 50


class AbstractLumpedPort(AbstractTerminalPort):
    impedance: Complex = pd.Field(
        DEFAULT_REFERENCE_IMPEDANCE,
        title="Reference impedance",
        description="Reference port impedance for scattering parameter computation.",
        units=OHM,
    )

    num_grid_cells: Optional[pd.PositiveInt] = pd.Field(
        DEFAULT_PORT_NUM_CELLS,
        title="Port grid cells",
        description="Number of mesh grid cells associated with the port along each direction, which are added through automatic mesh refinement. A value of `None` will turn off automatic mesh refinement.",
    )

    enable_snapping_points: bool = pd.Field(
        True,
        title="Snap Grid To Lumped Port",
        description="When enabled, snapping points are automatically generated to snap grids to key geometric features of the lumped port for more accurate modelling.",
    )

    @cached_property
    def _voltage_monitor_name(self) -> str:
        return f"{self.name}_voltage"

    @cached_property
    def _current_monitor_name(self) -> str:
        return f"{self.name}_current"

    def snapped_center(self, grid: Grid) -> Coordinate:
        center = list(self.center)
        normal_axis = self.injection_axis
        normal_port_center = center[normal_axis]
        center[normal_axis] = snap_coordinate_to_grid(grid, normal_port_center, normal_axis)
        return tuple(center)

    @cached_property
    @abstractmethod
    def to_load(self, snap_center: Optional[float] = None) -> LumpedElementType: ...

    @abstractmethod
    def to_voltage_monitor(
        self, freqs: FreqArray, snap_center: Optional[float] = None
    ) -> FieldMonitor: ...

    @abstractmethod
    def to_current_monitor(
        self, freqs: FreqArray, snap_center: Optional[float] = None
    ) -> FieldMonitor: ...

    def to_monitors(
        self, freqs: FreqArray, snap_center: Optional[float] = None, grid: Grid = None
    ) -> list[FieldMonitor]:
        return [
            self.to_voltage_monitor(freqs, snap_center, grid),
            self.to_current_monitor(freqs, snap_center, grid),
        ]

    @abstractmethod
    def _check_grid_size(self, yee_grid: YeeGrid): ...
