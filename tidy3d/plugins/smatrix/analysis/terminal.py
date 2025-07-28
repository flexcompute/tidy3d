from __future__ import annotations

import numpy as np

from tidy3d.components.data.sim_data import SimulationData
from tidy3d.plugins.smatrix.component_modelers.terminal import TerminalComponentModeler
from tidy3d.plugins.smatrix.data.data_array import PortDataArray, TerminalPortDataArray
from tidy3d.plugins.smatrix.data.terminal import TerminalComponentModelerData
from tidy3d.plugins.smatrix.ports.wave import WavePort
from tidy3d.plugins.smatrix.utils import (
    ab_to_s,
    check_port_impedance_sign,
    compute_F,
    compute_port_VI,
)


def terminal_construct_smatrix(modeler_data: TerminalComponentModelerData) -> TerminalPortDataArray:
    """
    Constructs the scattering matrix (S-matrix) from raw simulation data stored in a :class:`TerminalComponentModelerData`

    This function iterates through each port excitation simulation. For each run,
    it calculates the resulting incident ('a') and reflected ('b') power wave
    amplitudes at all ports. These amplitudes are compiled into matrices,
    which are then used to compute the final S-matrix using the formula
    :math:`S = b a^{-1}`.

    Args:
        modeler_data: Data object containing the modeler definition and the raw
            results from each port simulation run.

    Returns:
        TerminalPortDataArray
            The computed S-matrix as a data array with dimensions for frequency,
            output port, and input port.
    """

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

    # Tabulate the reference impedances at each port and frequency
    port_impedances = port_reference_impedances(modeler_data=modeler_data)

    # loop through source ports
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
    """Calculates the reference impedance for each port across all frequencies.

    This function determines the characteristic impedance for every port defined
    in the modeler. It handles two types of ports differently: for a
    :class:`.WavePort`, the impedance is frequency-dependent and computed from
    modal properties, while for other types like :class:`.LumpedPort`, the
    impedance is a user-defined constant value.

    Args:
        modeler_data: Data object containing the modeler definition and the raw
            simulation data needed for :class:`.WavePort` impedance calculations.

    Returns:
        TerminalComponentModelerData
            A data array containing the complex impedance for each port at each
            frequency.
    """
    port_names = [port.name for port in modeler_data.modeler.ports]

    values = np.zeros(
        (len(modeler_data.modeler.freqs), len(port_names)),
        dtype=complex,
    )
    coords = {"f": np.array(modeler_data.modeler.freqs), "port": port_names}
    port_impedances = PortDataArray(values, coords=coords)
    for port in modeler_data.modeler.ports:
        if isinstance(port, WavePort):
            # Mode solver data for each wave port is stored in its associated SimulationData
            sim_data_port = modeler_data.data[modeler_data.modeler.get_task_name(port=port)]
            # WavePorts have a port impedance calculated from its associated modal field distribution
            # and is frequency dependent.
            impedances = port.compute_port_impedance(sim_data_port).values
            port_impedances.loc[{"port": port.name}] = impedances.squeeze()
        else:
            # LumpedPorts have a constant reference impedance
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
    """
    Computes the incident (a) and reflected (b) power wave amplitudes at all ports
    from a single simulation run where one port was excited.

    This function converts the raw voltage (V) and current (I) data into power wave
    amplitudes using standard microwave engineering formulas. It also performs a
    sanity check to ensure the real part of the reference impedance is positive,
    flipping signs of V and Z if necessary to maintain physical consistency.

    Parameters
    ----------
    modeler : TerminalComponentModeler
        The modeler setup defining the ports.
    port_reference_impedances : PortDataArray
        The characteristic impedance for each port at each frequency.
    sim_data : SimulationData
        The raw results (fields, currents, etc.) from a single simulation run.

    Returns
    -------
    tuple[PortDataArray, PortDataArray]
        A tuple containing two PortDataArrays:
        - a: The incident power wave amplitudes at each port.
        - b: The reflected power wave amplitudes at each port.
    """
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

    # Check to make sure sign is consistent for all impedance values
    check_port_impedance_sign(Z_numpy)

    # # Check for negative real part of port impedance and flip the V and Z signs accordingly
    negative_real_Z = np.real(Z_numpy) < 0
    V_numpy = np.where(negative_real_Z, -V_numpy, V_numpy)
    Z_numpy = np.where(negative_real_Z, -Z_numpy, Z_numpy)

    F_numpy = compute_F(Z_numpy)

    # Equation 4.67 - Pozar - Microwave Engineering 4ed
    a.values = F_numpy * (V_numpy + Z_numpy * I_numpy)
    b.values = F_numpy * (V_numpy - np.conj(Z_numpy) * I_numpy)

    return a, b
