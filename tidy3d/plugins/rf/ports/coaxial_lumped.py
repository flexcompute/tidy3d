from __future__ import annotations

from typing import Optional

import numpy as np
import pydantic.v1 as pd

from tidy3d.components.base import cached_property
from tidy3d.components.data.data_array import FreqDataArray, ScalarFieldDataArray
from tidy3d.components.data.dataset import FieldDataset
from tidy3d.components.data.sim_data import SimulationData
from tidy3d.components.geometry.base import Box, Geometry
from tidy3d.components.geometry.utils_2d import increment_float
from tidy3d.components.grid.grid import Grid, YeeGrid
from tidy3d.components.lumped_element import CoaxialLumpedResistor
from tidy3d.components.monitor import FieldMonitor
from tidy3d.components.source.current import CustomCurrentSource
from tidy3d.components.source.time import GaussianPulse
from tidy3d.components.types import Axis, Coordinate, Direction, FreqArray, Size
from tidy3d.components.validators import skip_if_fields_missing
from tidy3d.constants import MICROMETER
from tidy3d.exceptions import SetupError, ValidationError
from tidy3d.plugins.microwave import CustomCurrentIntegral2D, VoltageIntegralAxisAligned
from tidy3d.plugins.microwave.path_integrals import AbstractAxesRH
from tidy3d.plugins.rf.ports.base_lumped import AbstractLumpedPort

DEFAULT_COAX_SOURCE_NUM_POINTS = 11


class CoaxialLumpedPort(AbstractLumpedPort, AbstractAxesRH):
    center: Coordinate = pd.Field((0.0, 0.0, 0.0), units=MICROMETER)
    outer_diameter: pd.PositiveFloat = pd.Field(..., units=MICROMETER)
    inner_diameter: pd.PositiveFloat = pd.Field(..., units=MICROMETER)
    normal_axis: Axis = pd.Field(...)
    direction: Direction = pd.Field(...)

    @cached_property
    def main_axis(self):
        return self.normal_axis

    @cached_property
    def injection_axis(self):
        return self.normal_axis

    @pd.validator("center", always=True)
    def _center_not_inf(cls, val):
        if any(np.isinf(v) for v in val):
            raise ValidationError("'center' can not contain 'td.inf' terms.")
        return val

    @pd.validator("inner_diameter", always=True)
    @skip_if_fields_missing(["outer_diameter"])
    def _ensure_inner_diameter_is_smaller(cls, val, values):
        outer_diameter = values.get("outer_diameter")
        if val >= outer_diameter:
            raise ValidationError(
                f"The 'inner_diameter' {val} of a coaxial lumped element must be less than its "
                f"'outer_diameter' {outer_diameter}."
            )
        return val

    def to_source(
        self, source_time: GaussianPulse, snap_center: Optional[float] = None, grid: Grid = None
    ) -> CustomCurrentSource:
        trans_axes = self.remaining_axes
        (coord1, coord2, coord3) = self.local_dims
        center = list(self.center)
        if snap_center:
            center[self.injection_axis] = snap_center

        size = [self.outer_diameter] * 3
        size[self.injection_axis] = 0
        bounding_box = Box(center=self.center, size=size)

        num1 = DEFAULT_COAX_SOURCE_NUM_POINTS
        num2 = DEFAULT_COAX_SOURCE_NUM_POINTS

        if grid:
            inds = grid.discretize_inds(box=bounding_box)
            num1 = inds[trans_axes[0]][1] - inds[trans_axes[0]][0]
            num2 = inds[trans_axes[1]][1] - inds[trans_axes[1]][0]

        def compute_coax_current(rin, rout, x, y):
            r = np.sqrt(x**2 + y**2)
            r_valid = np.where(r == 0.0, 1, r)
            denominator = 2 * np.pi * r_valid**2
            Jx = np.where(r <= rin, 0, (x / denominator))
            Jx = np.where(r >= rout, 0, Jx)
            Jy = np.where(r <= rin, 0, (y / denominator))
            Jy = np.where(r >= rout, 0, Jy)
            return (Jx, Jy)

        Router = self.outer_diameter / 2
        Rinner = self.inner_diameter / 2

        xs, ys = np.linspace(-Router, Router, 4 * num1), np.linspace(-Router, Router, 4 * num2)
        x_grid, y_grid = np.meshgrid(xs, ys, indexing="ij")
        Jx, Jy = compute_coax_current(Rinner, Router, x_grid, y_grid)

        E1 = "E" + coord1
        E2 = "E" + coord2

        coord_vals = {
            coord1: xs,
            coord2: ys,
            coord3: [center[self.injection_axis]],
            "f": [source_time.freq0],
        }

        kwargs = {
            E1: ScalarFieldDataArray(
                Jx[..., None, None],
                coords=coord_vals,
            ),
            E2: ScalarFieldDataArray(
                Jy[..., None, None],
                coords=coord_vals,
            ),
        }

        dataset_E = FieldDataset(**kwargs)

        return CustomCurrentSource(
            center=center,
            size=(self.outer_diameter, self.outer_diameter, 0),
            source_time=source_time,
            name=self.name,
            interpolate=True,
            confine_to_bounds=True,
            current_dataset=dataset_E,
        )

    def to_load(self, snap_center: Optional[float] = None) -> CoaxialLumpedResistor:
        center = list(self.center)
        if snap_center:
            center[self.injection_axis] = snap_center
        return CoaxialLumpedResistor(
            center=center,
            outer_diameter=self.outer_diameter,
            inner_diameter=self.inner_diameter,
            normal_axis=self.injection_axis,
            num_grid_cells=self.num_grid_cells,
            resistance=np.real(self.impedance),
            enable_snapping_points=self.enable_snapping_points,
            name=f"{self.name}_resistor",
        )

    def to_voltage_monitor(
        self, freqs: FreqArray, snap_center: Optional[float] = None, grid: Grid = None
    ) -> FieldMonitor:
        center = list(self.center)
        if snap_center:
            center[self.injection_axis] = snap_center
        (coord1, coord2) = self.remaining_dims
        E1 = "E" + coord1
        E2 = "E" + coord2
        return FieldMonitor(
            center=self._voltage_path_center(center),
            size=self._voltage_path_size,
            freqs=freqs,
            fields=[E1, E2],
            name=self._voltage_monitor_name,
            colocate=False,
        )

    def to_current_monitor(
        self, freqs: FreqArray, snap_center: Optional[float] = None, grid: Grid = None
    ) -> FieldMonitor:
        center = list(self.center)
        if snap_center:
            center[self.injection_axis] = snap_center
        (coord1, coord2) = self.remaining_dims
        H1 = "H" + coord1
        H2 = "H" + coord2
        dl = 2 * (increment_float(center[self.injection_axis], 1.0) - center[self.injection_axis])
        current_mon_size = [self.outer_diameter] * 3
        current_mon_size[self.injection_axis] = dl
        return FieldMonitor(
            center=center,
            size=current_mon_size,
            freqs=freqs,
            fields=[H1, H2],
            name=self._current_monitor_name,
            colocate=False,
        )

    def compute_voltage(self, sim_data: SimulationData) -> FreqDataArray:
        exact_port_center = self.snapped_center(sim_data.simulation.grid)
        field_data = sim_data[self._voltage_monitor_name]
        voltage_integral = VoltageIntegralAxisAligned(
            center=self._voltage_path_center(exact_port_center),
            size=self._voltage_path_size,
            extrapolate_to_endpoints=True,
            snap_path_to_grid=True,
            sign="+",
        )
        voltage = voltage_integral.compute_voltage(field_data)
        return voltage

    def compute_current(self, sim_data: SimulationData) -> FreqDataArray:
        exact_port_center = self.snapped_center(sim_data.simulation.grid)
        field_data = sim_data[self._current_monitor_name]
        (coord1, coord2, coord3) = self.local_dims
        field_coords = field_data.field_components["H" + coord1].coords
        num_coords_1 = len(field_coords[coord1].values)
        num_coords_2 = len(field_coords[coord2].values)
        num_coords = max(num_coords_1, num_coords_2)
        num_path_coords = round(np.pi * num_coords / 4) * 4 + 1
        normal_coords = field_coords[coord3].values
        radius = (self.outer_diameter + self.inner_diameter) / 4
        normal_port_position = exact_port_center[self.injection_axis]
        path_pos = CoaxialLumpedPort._determine_current_integral_pos(
            normal_port_position, normal_coords, self.direction
        )
        path_center = list(exact_port_center)
        path_center[self.injection_axis] = path_pos
        path_integral = CustomCurrentIntegral2D.from_circular_path(
            path_center, radius, num_path_coords, self.injection_axis, False
        )
        current = path_integral.compute_current(field_data)
        if self.direction == "+":
            current *= -1.0
        return current

    @staticmethod
    def _determine_current_integral_pos(
        snapped_center: float, normal_coords: np.array, direction: Direction
    ) -> float:
        upper_bound = np.searchsorted(normal_coords, snapped_center)
        lower_bound = upper_bound - 1
        if direction == "+":
            return normal_coords[upper_bound]
        return normal_coords[lower_bound]

    @cached_property
    def _voltage_axis(self) -> Axis:
        return self.remaining_axes[0]

    def _voltage_path_center(self, port_center: Coordinate) -> Coordinate:
        center = list(port_center)
        center[self._voltage_axis] += (self.outer_diameter + self.inner_diameter) / 4
        return tuple(center)

    @cached_property
    def _voltage_path_size(self) -> Size:
        axis_size = (self.outer_diameter - self.inner_diameter) / 2
        size = Geometry.unpop_axis(axis_size, (0, 0), self._voltage_axis)
        return size

    def _check_grid_size(self, yee_grid: YeeGrid):
        trans_axes = self.remaining_axes
        for axis in trans_axes:
            e_component = "xyz"[axis]
            e_grid = yee_grid.grid_dict[f"E{e_component}"]
            coords = e_grid.to_dict[e_component]
            min_bound = self.center[axis] - self.outer_diameter / 2
            max_bound = self.center[axis] - self.inner_diameter / 2
            coords_within_port = np.any(np.logical_and(coords > min_bound, coords < max_bound))
            min_bound = self.center[axis] + self.inner_diameter / 2
            max_bound = self.center[axis] + self.outer_diameter / 2
            coords_within_port2 = np.any(np.logical_and(coords > min_bound, coords < max_bound))
            if not coords_within_port or not coords_within_port2:
                raise SetupError(
                    f"Grid is too coarse along '{e_component}' direction for the lumped port "
                    f"at location '{self.center}'. Either set the port's 'num_grid_cells' to "
                    f"a nonzero integer or modify the 'GridSpec'. "
                )
