from __future__ import annotations

import numpy as np

from tidy3d.components.data.sim_data import SimulationData
from tidy3d.plugins.rf.component_modelers.terminal import TerminalComponentModeler
from tidy3d.plugins.rf.data.data_array import PortDataArray, TerminalPortDataArray
from tidy3d.plugins.rf.data.terminal import TerminalComponentModelerData
from tidy3d.plugins.rf.ports.wave import WavePort
from tidy3d.plugins.rf.utils import (
    ab_to_s,
    check_port_impedance_sign,
    compute_F,
    compute_port_VI,
)


def terminal_construct_smatrix(modeler_data: TerminalComponentModelerData) -> TerminalPortDataArray:
    port_names = [port.name for port in modeler_data.modeler.ports]
    values = np.zeros(
        (len(modeler_data.modeler.freqs), len(port_names), len(port_names)),
        dtype=complex,
    )
    coords = {
        "f": np.array(modeler_data.modeler.freqs),
        "port_out": port_names,
        "port_in": port_names,
    }
    a_matrix = TerminalPortDataArray(values, coords=coords)
    b_matrix = a_matrix.copy(deep=True)
    port_impedances = port_reference_impedances(modeler_data=modeler_data)
    for port_in in modeler_data.modeler.ports:
        sim_data = modeler_data.data[modeler_data.modeler.get_task_name(port=port_in)]
        a, b = compute_power_wave_amplitudes_at_each_port(
            modeler_data.modeler, port_impedances, sim_data
        )
        indexer = {"f": a.f, "port_in": port_in.name, "port_out": a.port}
        a_matrix.loc[indexer] = a
        b_matrix.loc[indexer] = b
    s_matrix = ab_to_s(a_matrix, b_matrix)
    return s_matrix


def port_reference_impedances(modeler_data: TerminalComponentModelerData) -> PortDataArray:
    port_names = [port.name for port in modeler_data.modeler.ports]
    values = np.zeros(
        (len(modeler_data.modeler.freqs), len(port_names)),
        dtype=complex,
    )
    coords = {"f": np.array(modeler_data.modeler.freqs), "port": port_names}
    port_impedances = PortDataArray(values, coords=coords)
    for port in modeler_data.modeler.ports:
        if isinstance(port, WavePort):
            sim_data_port = modeler_data.data[modeler_data.modeler.get_task_name(port=port)]
            impedances = port.compute_port_impedance(sim_data_port).values
            port_impedances.loc[{"port": port.name}] = impedances.squeeze()
        else:
            port_impedances.loc[{"port": port.name}] = np.full(
                len(modeler_data.modeler.freqs), port.impedance
            )
    port_impedances = modeler_data.modeler._set_port_data_array_attributes(port_impedances)
    return port_impedances


def compute_power_wave_amplitudes_at_each_port(
    modeler: TerminalComponentModeler,
    port_reference_impedances: PortDataArray,
    sim_data: SimulationData,
) -> tuple[PortDataArray, PortDataArray]:
    port_names = [port.name for port in modeler.ports]
    values = np.zeros(
        (len(modeler.freqs), len(port_names)),
        dtype=complex,
    )
    coords = {
        "f": np.array(modeler.freqs),
        "port": port_names,
    }
    V_matrix = PortDataArray(values, coords=coords)
    I_matrix = V_matrix.copy(deep=True)
    a = V_matrix.copy(deep=True)
    b = V_matrix.copy(deep=True)
    for port_out in modeler.ports:
        V_out, I_out = compute_port_VI(port_out, sim_data)
        indexer = {"port": port_out.name}
        V_matrix.loc[indexer] = V_out
        I_matrix.loc[indexer] = I_out
    V_numpy = V_matrix.values
    I_numpy = I_matrix.values
    Z_numpy = port_reference_impedances.values
    check_port_impedance_sign(Z_numpy)
    negative_real_Z = np.real(Z_numpy) < 0
    V_numpy = np.where(negative_real_Z, -V_numpy, V_numpy)
    Z_numpy = np.where(negative_real_Z, -Z_numpy, Z_numpy)
    F_numpy = compute_F(Z_numpy)
    a.values = F_numpy * (V_numpy + Z_numpy * I_numpy)
    b.values = F_numpy * (V_numpy - np.conj(Z_numpy) * I_numpy)
    return a, b
