"""Class and custom data array for representing a scattering matrix wave port."""

from typing import Optional, Union

import numpy as np
import pydantic.v1 as pd

from ....components.base import cached_property, skip_if_fields_missing
from ....components.data.data_array import FreqDataArray, FreqModeDataArray
from ....components.data.monitor_data import ModeData
from ....components.data.sim_data import SimulationData
from ....components.geometry.base import Box
from ....components.grid.grid import Grid
from ....components.monitor import ModeMonitor
from ....components.simulation import Simulation
from ....components.source.field import ModeSource, ModeSpec
from ....components.source.time import GaussianPulse
from ....components.types import Bound, Direction, FreqArray
from ....exceptions import ValidationError
from ...microwave import (
    CurrentIntegralTypes,
    ImpedanceCalculator,
    VoltageIntegralTypes,
)
from ...mode import ModeSolver
from .base_terminal import AbstractTerminalPort


class WavePort(AbstractTerminalPort, Box):
    """Class representing a single wave port"""

    direction: Direction = pd.Field(
        ...,
        title="Direction",
        description="'+' or '-', defining which direction is considered 'input'.",
    )

    mode_spec: ModeSpec = pd.Field(
        ModeSpec(),
        title="Mode Specification",
        description="Parameters to feed to mode solver which determine modes measured by monitor.",
    )

    mode_index: pd.NonNegativeInt = pd.Field(
        0,
        title="Mode Index",
        description="Index into the collection of modes returned by mode solver. "
        " Specifies which mode to inject using this source. "
        "If larger than ``mode_spec.num_modes``, "
        "``num_modes`` in the solver will be set to ``mode_index + 1``.",
    )

    voltage_integral: Optional[VoltageIntegralTypes] = pd.Field(
        None,
        title="Voltage Integral",
        description="Definition of voltage integral used to compute voltage and the characteristic impedance.",
    )

    current_integral: Optional[CurrentIntegralTypes] = pd.Field(
        None,
        title="Current Integral",
        description="Definition of current integral used to compute current and the characteristic impedance.",
    )

    def _mode_voltage_coefficients(self, mode_data: ModeData) -> FreqModeDataArray:
        """Calculates scaling coefficients to convert mode amplitudes
        to the total port voltage.
        """
        mode_data = mode_data._isel(mode_index=[self.mode_index])
        if self.voltage_integral is None:
            current_coeffs = self.current_integral.compute_current(mode_data)
            voltage_coeffs = 2 * np.abs(mode_data.flux) / np.conj(current_coeffs)
        else:
            voltage_coeffs = self.voltage_integral.compute_voltage(mode_data)
        return voltage_coeffs.squeeze()

    def _mode_current_coefficients(self, mode_data: ModeData) -> FreqModeDataArray:
        """Calculates scaling coefficients to convert mode amplitudes
        to the total port current.
        """
        mode_data = mode_data._isel(mode_index=[self.mode_index])
        if self.current_integral is None:
            voltage_coeffs = self.voltage_integral.compute_voltage(mode_data)
            current_coeffs = (2 * np.abs(mode_data.flux) / voltage_coeffs).conj()
        else:
            current_coeffs = self.current_integral.compute_current(mode_data)
        return current_coeffs.squeeze()

    @cached_property
    def injection_axis(self):
        """Injection axis of the port."""
        return self.size.index(0.0)

    @cached_property
    def _mode_monitor_name(self) -> str:
        """Return the name of the :class:`.ModeMonitor` associated with this port."""
        return f"{self.name}_mode"

    def to_source(self, source_time: GaussianPulse, snap_center: float = None) -> ModeSource:
        """Create a mode source from the wave port."""
        center = list(self.center)
        if snap_center:
            center[self.injection_axis] = snap_center
        return ModeSource(
            center=center,
            size=self.size,
            source_time=source_time,
            mode_spec=self.mode_spec,
            mode_index=self.mode_index,
            direction=self.direction,
            name=self.name,
        )

    def to_monitors(
        self, freqs: FreqArray, snap_center: float = None, grid: Grid = None
    ) -> list[ModeMonitor]:
        """The wave port uses a :class:`.ModeMonitor` to compute the characteristic impedance
        and the port voltages and currents."""
        center = list(self.center)
        if snap_center:
            center[self.injection_axis] = snap_center
        mode_mon = ModeMonitor(
            center=self.center,
            size=self.size,
            freqs=freqs,
            name=self._mode_monitor_name,
            colocate=False,
            mode_spec=self.mode_spec,
            store_fields_direction=self.direction,
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

    def compute_voltage(self, sim_data: SimulationData) -> FreqDataArray:
        """Helper to compute voltage across the port."""
        mode_data = sim_data[self._mode_monitor_name]
        voltage_coeffs = self._mode_voltage_coefficients(mode_data)
        amps = mode_data.amps
        fwd_amps = amps.sel(direction="+").squeeze()
        bwd_amps = amps.sel(direction="-").squeeze()
        return voltage_coeffs * (fwd_amps + bwd_amps)

    def compute_current(self, sim_data: SimulationData) -> FreqDataArray:
        """Helper to compute current flowing through the port."""
        mode_data = sim_data[self._mode_monitor_name]
        current_coeffs = self._mode_current_coefficients(mode_data)
        amps = mode_data.amps
        fwd_amps = amps.sel(direction="+").squeeze()
        bwd_amps = amps.sel(direction="-").squeeze()
        # In ModeData, fwd_amps and bwd_amps are not relative to
        # the direction fields are stored
        sign = 1.0
        if self.direction == "-":
            sign = -1.0
        return sign * current_coeffs * (fwd_amps - bwd_amps)

    def compute_port_impedance(
        self, sim_mode_data: Union[SimulationData, ModeData]
    ) -> FreqModeDataArray:
        """Helper to compute impedance of port. The port impedance is computed from the
        transmission line mode, which should be TEM or at least quasi-TEM."""
        impedance_calc = ImpedanceCalculator(
            voltage_integral=self.voltage_integral, current_integral=self.current_integral
        )
        if isinstance(sim_mode_data, SimulationData):
            mode_data = sim_mode_data[self._mode_monitor_name]
        else:
            mode_data = sim_mode_data

        # Filter out unwanted modes to reduce impedance computation effort
        mode_data = mode_data._isel(mode_index=[self.mode_index])
        impedance_array = impedance_calc.compute_impedance(mode_data)
        return impedance_array

    @staticmethod
    def _within_port_bounds(path_bounds: Bound, port_bounds: Bound) -> bool:
        """Helper to check if one bounding box is completely within the other."""
        path_min = np.array(path_bounds[0])
        path_max = np.array(path_bounds[1])
        bound_min = np.array(port_bounds[0])
        bound_max = np.array(port_bounds[1])
        return (bound_min <= path_min).all() and (bound_max >= path_max).all()

    @pd.validator("voltage_integral", "current_integral")
    def _validate_path_integrals_within_port(cls, val, values):
        """Raise ``ValidationError`` when the supplied path integrals are not within the port bounds."""
        center = values["center"]
        size = values["size"]
        box = Box(center=center, size=size)
        if val and not WavePort._within_port_bounds(val.bounds, box.bounds):
            raise ValidationError(
                f"'{cls.__name__}' must be setup with all path integrals defined within the bounds "
                f"of the port. Path bounds are '{val.bounds}', but port bounds are '{box.bounds}'."
            )
        return val

    @pd.validator("current_integral", always=True)
    @skip_if_fields_missing(["voltage_integral"])
    def _check_voltage_or_current(cls, val, values):
        """Raise validation error if both ``voltage_integral`` and ``current_integral``
        were not provided."""
        if values.get("voltage_integral") is None and val is None:
            raise ValidationError(
                "At least one of 'voltage_integral' or 'current_integral' must be provided."
            )
        return val

    @pd.validator("current_integral", always=True)
    def validate_current_integral_sign(cls, val, values):
        """
        Validate that the sign of ``current_integral`` matches the port direction.
        """
        if val is None:
            return val

        direction = values.get("direction")
        name = values.get("name")
        if val.sign != direction:
            raise ValidationError(
                f"'current_integral' sign must match the '{name}' direction '{direction}'."
            )
        return val
