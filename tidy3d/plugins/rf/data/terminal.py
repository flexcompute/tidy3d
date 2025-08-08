"""RF terminal modeler data and S-matrix utilities."""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pydantic.v1 as pd

from tidy3d.components.base import Tidy3dBaseModel, cached_property
from tidy3d.components.data.data_array import FreqDataArray
from tidy3d.components.data.monitor_data import MonitorData
from tidy3d.components.data.sim_data import SimulationData
from tidy3d.components.microwave.data.monitor_data import AntennaMetricsData
from tidy3d.log import log
from tidy3d.plugins.rf.component_modelers.terminal import TerminalComponentModeler
from tidy3d.plugins.rf.data.data_array import PortDataArray, TerminalPortDataArray
from tidy3d.plugins.rf.data.modal import PortSimulationData
from tidy3d.plugins.rf.ports.types import TerminalPortType
from tidy3d.plugins.rf.utils import (
    ab_to_s,
    check_port_impedance_sign,
    compute_F,
    compute_port_VI,
    compute_power_delivered_by_port,
    compute_power_wave_amplitudes,
    s_to_z,
)


class MicrowaveSMatrixData(Tidy3dBaseModel):
    port_reference_impedances: Optional[PortDataArray] = pd.Field(
        None,
        title="Port Reference Impedances",
        description="Reference impedance for each port used in the S-parameter calculation.",
    )

    data: TerminalPortDataArray = pd.Field(
        ...,
        title="S-Matrix Data",
        description="Computed S-matrix over terminal ports.",
    )


class TerminalComponentModelerData(Tidy3dBaseModel):
    modeler: TerminalComponentModeler = pd.Field(...)
    data: PortSimulationData = pd.Field(...)
    log: str = pd.Field(None)

    @cached_property
    def smatrix(self) -> MicrowaveSMatrixData:
        from tidy3d.plugins.rf.analysis.terminal import terminal_construct_smatrix

        terminal_port_data = terminal_construct_smatrix(modeler_data=self)
        smatrix_data = MicrowaveSMatrixData(data=terminal_port_data)
        return smatrix_data

    @pd.root_validator(pre=False)
    def _warn_rf_license(cls, values):
        log.warning(
            "ℹ️ ⚠️ RF simulations are subject to new license requirements in the future. You have instantiated at least one RF-specific component.",
            log_once=True,
        )
        return values

    def _monitor_data_at_port_amplitude(
        self,
        port: TerminalPortType,
        sim_data: SimulationData,
        monitor_data: MonitorData,
        a_port: Union[FreqDataArray, complex],
    ) -> MonitorData:
        a_raw, _ = self.compute_power_wave_amplitudes_at_each_port(
            self.port_reference_impedances, sim_data=sim_data
        )
        a_raw_port = a_raw.sel(port=port.name)
        if not isinstance(a_port, FreqDataArray):
            freqs = list(monitor_data.monitor.freqs)
            array_vals = a_port * np.ones(len(freqs))
            a_port = FreqDataArray(array_vals, coords={"f": freqs})
        scale_array = a_port / a_raw_port
        return monitor_data.scale_fields_by_freq_array(scale_array, method="nearest")

    def get_antenna_metrics_data(
        self,
        port_amplitudes: Optional[dict[str, complex]] = None,
        monitor_name: Optional[str] = None,
    ) -> AntennaMetricsData:
        from tidy3d.plugins.rf.analysis.antenna import get_antenna_metrics_data

        antenna_metrics_data = get_antenna_metrics_data(
            terminal_component_modeler_data=self,
            port_amplitudes=port_amplitudes,
            monitor_name=monitor_name,
        )
        return antenna_metrics_data

    @cached_property
    def port_reference_impedances(self) -> PortDataArray:
        from tidy3d.plugins.rf.analysis.terminal import port_reference_impedances

        return port_reference_impedances(self)

    def compute_power_wave_amplitudes_at_each_port(
        self, port_reference_impedances: PortDataArray, sim_data: SimulationData
    ) -> tuple[PortDataArray, PortDataArray]:
        from tidy3d.plugins.rf.analysis.terminal import (
            compute_power_wave_amplitudes_at_each_port,
        )

        data = compute_power_wave_amplitudes_at_each_port(
            modeler=self.modeler,
            port_reference_impedances=port_reference_impedances,
            sim_data=sim_data,
        )
        return data

    # Mirror utils
    ab_to_s = ab_to_s
    compute_F = compute_F
    check_port_impedance_sign = check_port_impedance_sign
    compute_port_VI = compute_port_VI
    compute_power_wave_amplitudes = compute_power_wave_amplitudes
    compute_power_delivered_by_port = compute_power_delivered_by_port
    s_to_z = s_to_z
