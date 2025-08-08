from __future__ import annotations

from typing import Optional

import numpy as np
import pydantic.v1 as pd

from tidy3d.components.base import cached_property
from tidy3d.components.data.data_array import FreqDataArray
from tidy3d.components.data.sim_data import SimulationData
from tidy3d.components.geometry.base import Box
from tidy3d.components.geometry.utils import (
    SnapBehavior,
    SnapLocation,
    SnappingSpec,
    snap_box_to_grid,
)
from tidy3d.components.geometry.utils_2d import increment_float
from tidy3d.components.grid.grid import Grid, YeeGrid
from tidy3d.components.lumped_element import LinearLumpedElement, LumpedResistor, RLCNetwork
from tidy3d.components.monitor import FieldMonitor
from tidy3d.components.source.current import UniformCurrentSource
from tidy3d.components.source.time import GaussianPulse
from tidy3d.components.types import Axis, FreqArray, LumpDistType
from tidy3d.components.validators import assert_line_or_plane
from tidy3d.exceptions import SetupError, ValidationError
from tidy3d.plugins.microwave import CurrentIntegralAxisAligned, VoltageIntegralAxisAligned
from tidy3d.plugins.rf.ports.base_lumped import AbstractLumpedPort


class LumpedPort(AbstractLumpedPort, Box):
    voltage_axis: Axis = pd.Field(...)
    snap_perimeter_to_grid: bool = pd.Field(True)
    dist_type: LumpDistType = pd.Field("on")

    _line_plane_validator = assert_line_or_plane()

    @cached_property
    def injection_axis(self):
        return self.size.index(0.0)

    @pd.validator("voltage_axis", always=True)
    def _voltage_axis_in_plane(cls, val, values):
        size = values.get("size")
        if val == size.index(0.0):
            raise ValidationError("'voltage_axis' must lie in the port's plane.")
        return val

    @cached_property
    def current_axis(self) -> Axis:
        return 3 - self.injection_axis - self.voltage_axis

    def to_source(
        self, source_time: GaussianPulse, snap_center: Optional[float] = None, grid: Grid = None
    ) -> UniformCurrentSource:
        if grid:
            load_box = self._to_load_box(grid=grid)
            center = load_box.center
            size = load_box.size
        else:
            center = list(self.center)
            if snap_center:
                center[self.injection_axis] = snap_center
            size = self.size
        component = "xyz"[self.voltage_axis]
        return UniformCurrentSource(
            center=center,
            size=size,
            source_time=source_time,
            polarization=f"E{component}",
            name=self.name,
            interpolate=True,
            confine_to_bounds=True,
        )

    def to_load(self, snap_center: Optional[float] = None) -> LumpedResistor:
        center = list(self.center)
        if snap_center:
            center[self.injection_axis] = snap_center
        network = RLCNetwork(resistance=np.real(self.impedance))
        return LinearLumpedElement(
            center=center,
            size=self.size,
            num_grid_cells=self.num_grid_cells,
            network=network,
            name=f"{self.name}_resistor",
            voltage_axis=self.voltage_axis,
            snap_perimeter_to_grid=self.snap_perimeter_to_grid,
            dist_type=self.dist_type,
            enable_snapping_points=self.enable_snapping_points,
        )

    def to_voltage_monitor(
        self, freqs: FreqArray, snap_center: Optional[float] = None, grid: Grid = None
    ) -> FieldMonitor:
        if grid:
            voltage_box = self._to_voltage_box(grid)
            center = voltage_box.center
            size = voltage_box.size
        else:
            center = list(self.center)
            if snap_center:
                center[self.injection_axis] = snap_center
            size = list(self.size)
            size[self.injection_axis] = 0.0
            size[self.current_axis] = 0.0
        e_component = "xyz"[self.voltage_axis]
        return FieldMonitor(
            center=center,
            size=size,
            freqs=freqs,
            fields=[f"E{e_component}"],
            name=self._voltage_monitor_name,
            colocate=False,
        )

    def to_current_monitor(
        self, freqs: FreqArray, snap_center: Optional[float] = None, grid: Grid = None
    ) -> FieldMonitor:
        if grid:
            current_box = self._to_current_box(grid)
            center = current_box.center
            size = current_box.size
        else:
            center = list(self.center)
            if snap_center:
                center[self.injection_axis] = snap_center
            dl = 2 * (
                increment_float(center[self.injection_axis], 1.0) - center[self.injection_axis]
            )
            size = list(self.size)
            size[self.injection_axis] = dl
            size[self.voltage_axis] = 0.0
        h_component = "xyz"[self.current_axis]
        h_cap_component = "xyz"[self.injection_axis]
        return FieldMonitor(
            center=center,
            size=size,
            freqs=freqs,
            fields=[f"H{h_component}", f"H{h_cap_component}"],
            name=self._current_monitor_name,
            colocate=False,
        )

    def compute_voltage(self, sim_data: SimulationData) -> FreqDataArray:
        voltage_box = self._to_voltage_box(sim_data.simulation.grid)
        field_data = sim_data[self._voltage_monitor_name]
        voltage_integral = VoltageIntegralAxisAligned(
            center=voltage_box.center,
            size=voltage_box.size,
            extrapolate_to_endpoints=True,
            snap_path_to_grid=True,
            sign="+",
        )
        voltage = voltage_integral.compute_voltage(field_data)
        return voltage

    def compute_current(self, sim_data: SimulationData) -> FreqDataArray:
        field_data = sim_data[self._current_monitor_name]
        current_box = self._to_current_box(sim_data.simulation.grid)
        I_integral = CurrentIntegralAxisAligned(
            center=current_box.center,
            size=current_box.size,
            sign="+",
            extrapolate_to_endpoints=True,
            snap_contour_to_grid=True,
        )
        return I_integral.compute_current(field_data)

    def _check_grid_size(self, yee_grid: YeeGrid):
        e_component = "xyz"[self.voltage_axis]
        e_yee_grid = yee_grid.grid_dict[f"E{e_component}"]
        coords = e_yee_grid.to_dict[e_component]
        min_bound = self.bounds[0][self.voltage_axis]
        max_bound = self.bounds[1][self.voltage_axis]
        coords_within_port = np.any(np.logical_and(coords > min_bound, coords < max_bound))
        if not coords_within_port:
            raise SetupError(
                f"Grid is too coarse along '{e_component}' direction for the lumped port "
                f"at location '{self.center}'. Either set the port's 'num_grid_cells' to "
                f"a nonzero integer or modify the 'GridSpec'."
            )

    def _to_load_box(self, grid: Grid) -> Box:
        load = self.to_load()
        load_box = load._create_box_for_network(grid=grid)
        return load_box

    def _to_voltage_box(self, grid: Grid) -> Box:
        load_box = self._to_load_box(grid=grid)
        size = list(load_box.size)
        size[self.current_axis] = 0
        size[self.injection_axis] = 0
        voltage_box = Box(center=load_box.center, size=size)
        return voltage_box

    def _to_current_box(self, grid: Grid) -> Box:
        load_box = self._to_load_box(grid=grid)
        size = list(load_box.size)
        size[self.voltage_axis] = 0
        current_box = Box(center=load_box.center, size=size)
        snap_location = [SnapLocation.Center] * 3
        snap_behavior = [SnapBehavior.Expand] * 3
        snap_behavior[self.voltage_axis] = SnapBehavior.Off
        snap_spec = SnappingSpec(location=snap_location, behavior=snap_behavior)
        current_box = snap_box_to_grid(grid, current_box, snap_spec)
        return current_box
