from __future__ import annotations

from typing import Optional

import numpy as np

from tidy3d.components.microwave.data.monitor_data import AntennaMetricsData
from tidy3d.plugins.smatrix.analysis.terminal import (
    compute_wave_amplitudes_at_each_port,
)
from tidy3d.plugins.smatrix.data.data_array import PortDataArray
from tidy3d.plugins.smatrix.data.terminal import TerminalComponentModelerData


def get_antenna_metrics_data(
    terminal_component_modeler_data: TerminalComponentModelerData,
    port_amplitudes: Optional[dict[str, complex]] = None,
    monitor_name: Optional[str] = None,
) -> AntennaMetricsData:
    """Calculate antenna parameters using superposition of fields from multiple port excitations.

    The method computes the radiated far fields and port excitation power wave amplitudes
    for a superposition of port excitations, which can be used to analyze antenna radiation
    characteristics.

    Parameters
    ----------
    terminal_component_modeler_data: TerminalComponentModelerData
        Data associated with a :class:`.TerminalComponentModeler` simulation run.
    port_amplitudes : dict[str, complex] = None
        Dictionary mapping port names to their desired excitation amplitudes. For each port,
        :math:`\\frac{1}{2}|a|^2` represents the incident power from that port into the system.
        If None, uses only the first port without any scaling of the raw simulation data.
    monitor_name : str = None
        Name of the :class:`.DirectivityMonitor` to use for calculating far fields.
        If None, uses the first monitor in `radiation_monitors`.

    Returns
    -------
    :class:`.AntennaMetricsData`
        Container with antenna parameters including directivity, gain, and radiation efficiency,
        computed from the superposition of fields from all excited ports.
    """
    # Use the first port as default if none specified
    if port_amplitudes is None:
        port_amplitudes = {terminal_component_modeler_data.modeler.ports[0].name: None}
    # Check port names, and create map from port to amplitude
    port_dict = {}
    for key in port_amplitudes.keys():
        port, _ = terminal_component_modeler_data.modeler.network_dict[key]
        port_dict[port] = port_amplitudes[key]
    # Get the radiation monitor, use first as default
    # if none specified
    if monitor_name is None:
        rad_mon = terminal_component_modeler_data.modeler.radiation_monitors[0]
    else:
        rad_mon = terminal_component_modeler_data.modeler.get_radiation_monitor_by_name(
            monitor_name
        )

    # Create data arrays for holding the superposition of all port power wave amplitudes
    f = list(rad_mon.freqs)
    coords = {"f": f, "port": list(terminal_component_modeler_data.modeler.matrix_indices_monitor)}
    a_sum = PortDataArray(
        np.zeros(
            (len(f), len(terminal_component_modeler_data.modeler.matrix_indices_monitor)),
            dtype=complex,
        ),
        coords=coords,
    )
    b_sum = a_sum.copy()
    # Retrieve associated simulation data
    combined_directivity_data = None
    for port, amplitude in port_dict.items():
        if amplitude is not None:
            if np.isclose(amplitude, 0.0):
                continue
        sim_data_port = terminal_component_modeler_data.data[
            terminal_component_modeler_data.modeler.get_task_name(port)
        ]

        a, b = compute_wave_amplitudes_at_each_port(
            modeler=terminal_component_modeler_data.modeler,
            port_reference_impedances=terminal_component_modeler_data.port_reference_impedances,
            sim_data=sim_data_port,
            s_param_def="power",
        )
        # Select a possible subset of frequencies
        a = a.sel(f=f)
        b = b.sel(f=f)
        a_raw = a.sel(port=terminal_component_modeler_data.modeler.network_index(port))

        if amplitude is None:
            # No scaling performed when amplitude is None
            scaled_directivity_data = sim_data_port[rad_mon.name]
            scale_factor = 1.0
        else:
            scaled_directivity_data = (
                terminal_component_modeler_data._monitor_data_at_port_amplitude(
                    port, rad_mon.name, amplitude
                )
            )
            scale_factor = amplitude / a_raw
        a = scale_factor * a
        b = scale_factor * b

        # Combine the possibly scaled directivity data and the power wave amplitudes
        if combined_directivity_data is None:
            combined_directivity_data = scaled_directivity_data
        else:
            combined_directivity_data = combined_directivity_data + scaled_directivity_data
        a_sum += a
        b_sum += b

    # Compute and add power measures to results
    power_incident = np.real(0.5 * a_sum * np.conj(a_sum)).sum(dim="port")
    power_reflected = np.real(0.5 * b_sum * np.conj(b_sum)).sum(dim="port")
    return AntennaMetricsData.from_directivity_data(
        combined_directivity_data, power_incident, power_reflected
    )
