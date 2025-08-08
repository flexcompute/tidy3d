from __future__ import annotations

from typing import Optional

import numpy as np

from tidy3d.components.microwave.data.monitor_data import AntennaMetricsData
from tidy3d.plugins.rf.analysis.terminal import (
    compute_power_wave_amplitudes_at_each_port,
)
from tidy3d.plugins.rf.data.data_array import PortDataArray
from tidy3d.plugins.rf.data.terminal import TerminalComponentModelerData


def get_antenna_metrics_data(
    terminal_component_modeler_data: TerminalComponentModelerData,
    port_amplitudes: Optional[dict[str, complex]] = None,
    monitor_name: Optional[str] = None,
) -> AntennaMetricsData:
    if port_amplitudes is None:
        port_amplitudes = {terminal_component_modeler_data.modeler.ports[0].name: None}
    port_names = [port.name for port in terminal_component_modeler_data.modeler.ports]
    port_dict = {}
    for key in port_amplitudes.keys():
        port = terminal_component_modeler_data.modeler.get_port_by_name(port_name=key)
        port_dict[port] = port_amplitudes[key]
    if monitor_name is None:
        rad_mon = terminal_component_modeler_data.modeler.radiation_monitors[0]
    else:
        rad_mon = terminal_component_modeler_data.modeler.get_radiation_monitor_by_name(
            monitor_name
        )
    f = list(rad_mon.freqs)
    coords = {"f": f, "port": port_names}
    a_sum = PortDataArray(np.zeros((len(f), len(port_names)), dtype=complex), coords=coords)
    b_sum = a_sum.copy()
    combined_directivity_data = None
    for port, amplitude in port_dict.items():
        sim_data_port = terminal_component_modeler_data.data[
            terminal_component_modeler_data.modeler.get_task_name(port)
        ]
        radiation_data = sim_data_port[rad_mon.name]
        a, b = compute_power_wave_amplitudes_at_each_port(
            modeler=terminal_component_modeler_data.modeler,
            port_reference_impedances=terminal_component_modeler_data.port_reference_impedances,
            sim_data=sim_data_port,
        )
        a = a.sel(f=f)
        b = b.sel(f=f)
        a_raw = a.sel(port=port.name)
        if amplitude is None:
            scaled_directivity_data = sim_data_port[rad_mon.name]
            scale_factor = 1.0
        else:
            scaled_directivity_data = (
                terminal_component_modeler_data._monitor_data_at_port_amplitude(
                    port, sim_data_port, radiation_data, amplitude
                )
            )
            scale_factor = amplitude / a_raw
        a = scale_factor * a
        b = scale_factor * b
        if combined_directivity_data is None:
            combined_directivity_data = scaled_directivity_data
        else:
            combined_directivity_data = combined_directivity_data + scaled_directivity_data
        a_sum += a
        b_sum += b
    power_incident = np.real(0.5 * a_sum * np.conj(a_sum)).sum(dim="port")
    power_reflected = np.real(0.5 * b_sum * np.conj(b_sum)).sum(dim="port")
    return AntennaMetricsData.from_directivity_data(
        combined_directivity_data, power_incident, power_reflected
    )
