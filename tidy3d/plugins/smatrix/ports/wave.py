"""Class and custom data array for representing a scattering matrix wave port."""

import pydantic.v1 as pd
from typing import Optional

from ....components.base import cached_property

from ....components.types import FreqArray, Direction
from ....components.geometry.base import Box
from ....components.monitor import FieldMonitor, ModeSolverMonitor
from ....components.source import ModeSpec, ModeSource, GaussianPulse
from ....components.data.sim_data import SimulationData
from ....components.data.data_array import FreqDataArray
from ....exceptions import ValidationError

from ...microwave import ImpedanceCalculator, VoltageIntegralTypes, CurrentIntegralTypes
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

    @cached_property
    def injection_axis(self):
        """Injection axis of the port."""
        return self.size.index(0.0)

    @cached_property
    def _field_monitor_name(self) -> str:
        return f"{self.name}_field"

    @cached_property
    def _mode_monitor_name(self) -> str:
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

    def to_field_monitors(self, freqs: FreqArray, snap_center: float = None) -> list[FieldMonitor]:
        """Field monitor to compute port voltage and current."""
        center = list(self.center)
        if snap_center:
            center[self.injection_axis] = snap_center
        field_mon = FieldMonitor(
            center=center,
            size=self.size,
            freqs=freqs,
            name=self._field_monitor_name,
            colocate=False,
        )
        return [field_mon]

    def to_mode_solver_monitor(self, freqs: FreqArray) -> ModeSolverMonitor:
        """Mode solver monitor to compute modes that will be used to
        compute characteristic impedances."""
        mode_mon = ModeSolverMonitor(
            center=self.center,
            size=self.size,
            freqs=freqs,
            name=self._mode_monitor_name,
            colocate=False,
            mode_spec=self.mode_spec,
            direction=self.direction,
        )
        return mode_mon

    def compute_voltage(self, sim_data: SimulationData) -> FreqDataArray:
        """Helper to compute voltage across the port."""
        field_monitor = sim_data[self._field_monitor_name]
        return self.voltage_integral.compute_voltage(field_monitor)

    def compute_current(self, sim_data: SimulationData) -> FreqDataArray:
        """Helper to compute current flowing through the port."""
        field_monitor = sim_data[self._field_monitor_name]
        return self.current_integral.compute_current(field_monitor)

    def compute_port_impedance(self, sim_data: SimulationData) -> FreqDataArray:
        """Helper to compute impedance of port. The port impedance is computed from the
        transmission line mode, which should be TEM or atleast quasi-TEM."""
        impedance_calc = ImpedanceCalculator(
            voltage_integral=self.voltage_integral, current_integral=self.current_integral
        )
        mode_solver_data = sim_data[self._mode_monitor_name]
        impedance_array = impedance_calc.compute_impedance(mode_solver_data)
        impedance_array = impedance_array.sel(mode_index=self.mode_index)
        return FreqDataArray(impedance_array.values, coords=impedance_array.coords)

    @pd.validator("voltage_integral", "current_integral")
    def _validate_path_integrals_within_port(cls, val, values):
        """Raise ``ValidationError`` when the supplied path integrals are not within the port bounds."""
        center = values["center"]
        size = values["size"]
        box = Box(center=center, size=size)
        if val and not val.within_bounds(box.bounds):
            raise ValidationError(
                f"'{cls.__name__}' must be setup with all path integrals defined within the bounds "
                f"of the port. Path bounds are {val.bounds}, but port bounds are {box.bounds}."
            )
        return val

    @pd.validator("current_integral", always=True)
    def _check_voltage_or_current(cls, val, values):
        """Raise validation error if both ``voltage_integral`` and ``current_integral``
        were not provided."""
        if not values.get("voltage_integral") and not val:
            raise ValidationError(
                "Atleast one of 'voltage_integral' or 'current_integral' must be provided."
            )
        return val
