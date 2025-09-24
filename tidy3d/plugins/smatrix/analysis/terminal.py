"""Terminal component modeler analysis functions.

This module contains functions for constructing S-matrices and computing wave amplitudes
for terminal-based component modeling in electromagnetic simulations.

References
----------
.. [1]  R. B. Marks and D. F. Williams, "A general waveguide circuit theory,"
        J. Res. Natl. Inst. Stand. Technol., vol. 97, pp. 533, 1992.

.. [2]  D. M. Pozar, Microwave Engineering, 4th ed. Hoboken, NJ, USA:
        John Wiley & Sons, 2012.
"""

from __future__ import annotations

import numpy as np

from tidy3d.components.data.sim_data import SimulationData
from tidy3d.plugins.smatrix.component_modelers.terminal import TerminalComponentModeler
from tidy3d.plugins.smatrix.data.data_array import PortDataArray, TerminalPortDataArray
from tidy3d.plugins.smatrix.data.terminal import TerminalComponentModelerData
from tidy3d.plugins.smatrix.ports.wave import WavePort
from tidy3d.plugins.smatrix.types import SParamDef
from tidy3d.plugins.smatrix.utils import (
    ab_to_s,
    check_port_impedance_sign,
    compute_F,
    compute_port_VI,
)


def terminal_construct_smatrix(
    modeler_data: TerminalComponentModelerData,
    assume_ideal_excitation: bool = False,
    s_param_def: SParamDef = "pseudo",
) -> TerminalPortDataArray:
    """Constructs the scattering matrix (S-matrix) from raw simulation data.

    This function iterates through each port excitation simulation. For each run,
    it calculates the resulting incident ('a') and reflected ('b') wave
    amplitudes at all ports. These amplitudes are compiled into matrices,
    which are then used to compute the final S-matrix.

    If all ports are excited and ``assume_ideal_excitation`` is ``False``, the
    S-matrix is computed using the formula :math:`S = b a^{-1}`. Otherwise,
    it is assumed that the incident wave matrix 'a' is diagonal, and the
    S-matrix is computed more efficiently by scaling the 'b' matrix. This
    is also necessary when only a subset of ports are excited.

    Parameters
    ----------
    modeler_data : TerminalComponentModelerData
        Data object containing the modeler definition and the raw
        results from each port simulation run.
    assume_ideal_excitation : bool, optional
        If ``True``, assumes that exciting one port does not produce incident
        waves at other ports. This simplifies the S-matrix calculation and is
        required if not all ports are excited. Default is ``False``.
    s_param_def : SParamDef, optional
        The definition of S-parameters to use depends whether "pseudo waves"
        or "power waves" are calculated. Default is "pseudo".

    Returns
    -------
    TerminalPortDataArray
        The computed S-matrix as a :class:`.TerminalPortDataArray` with dimensions
        for frequency, output port, and input port.
    """
    monitor_indices = list(modeler_data.modeler.matrix_indices_monitor)
    source_indices = list(modeler_data.modeler.matrix_indices_source)
    run_source_indices = list(modeler_data.modeler.matrix_indices_run_sim)

    if s_param_def == "pseudo":
        a_matrix, b_matrix = modeler_data.port_pseudo_wave_matrices
    else:
        a_matrix, b_matrix = modeler_data.port_power_wave_matrices

    # If excitation is assumed ideal, a_matrix is assumed to be diagonal
    # and the explicit inverse can be avoided. When only a subset of excitations
    # have been run, we cannot find the inverse anyways so must make this assumption.
    if len(monitor_indices) == len(run_source_indices) and not assume_ideal_excitation:
        s_matrix = ab_to_s(a_matrix, b_matrix)
    else:
        a_diag = np.diagonal(a_matrix, axis1=1, axis2=2)
        # Scale each column by the corresponding diagonal entry
        s_matrix = b_matrix / a_diag[:, np.newaxis, :]

    # Expand the smatrix using user defined mappings
    s_matrix_expanded = s_matrix.reindex(port_in=source_indices, fill_value=0.0)
    # element can be determined by user-defined mapping
    for (row_in, col_in), (row_out, col_out), mult_by in modeler_data.modeler.element_mappings:
        coords_from = {
            "port_in": col_in,
            "port_out": row_in,
        }
        coords_to = {
            "port_in": col_out,
            "port_out": row_out,
        }
        data = mult_by * s_matrix_expanded.loc[coords_from].data
        s_matrix_expanded = s_matrix_expanded._with_updated_data(data=data, coords=coords_to)
    return s_matrix_expanded


def port_reference_impedances(modeler_data: TerminalComponentModelerData) -> PortDataArray:
    """Calculates the reference impedance for each port across all frequencies.

    This function determines the characteristic impedance for every port defined
    in the modeler. It handles two types of ports differently: for a
    :class:`.WavePort`, the impedance is frequency-dependent and computed from
    modal properties, while for other types like :class:`.LumpedPort`, the
    impedance is a user-defined constant value.

    Parameters
    ----------
    modeler_data : TerminalComponentModelerData
        Data object containing the modeler definition and the raw
        simulation data needed for :class:`.WavePort` impedance calculations.

    Returns
    -------
    PortDataArray
        A ``PortDataArray`` containing the complex impedance for each port at each
        frequency.
    """
    values = np.zeros(
        (len(modeler_data.modeler.freqs), len(modeler_data.modeler.matrix_indices_monitor)),
        dtype=complex,
    )

    coords = {
        "f": np.array(modeler_data.modeler.freqs),
        "port": list(modeler_data.modeler.matrix_indices_monitor),
    }
    port_impedances = PortDataArray(values, coords=coords)
    # Each simulation will store the results from the ModeMonitors,
    # so here we just choose the first one.
    first_sim_index = modeler_data.modeler.matrix_indices_run_sim[0]
    port, mode_index = modeler_data.modeler.network_dict[first_sim_index]
    sim_data = modeler_data.data[
        modeler_data.modeler.get_task_name(port=port, mode_index=mode_index)
    ]
    for network_index in modeler_data.modeler.matrix_indices_monitor:
        port, mode_index = modeler_data.modeler.network_dict[network_index]
        indexer = {"port": network_index}
        if isinstance(port, WavePort):
            # WavePorts have a port impedance calculated from its associated modal field distribution
            # and is frequency dependent.
            data = port.get_port_impedance(sim_data, mode_index).data
            port_impedances = port_impedances._with_updated_data(data=data, coords=indexer)
        else:
            # LumpedPorts have a constant reference impedance
            data = np.full(len(modeler_data.modeler.freqs), port.impedance)
            port_impedances = port_impedances._with_updated_data(data=data, coords=indexer)

    port_impedances = modeler_data.modeler._set_port_data_array_attributes(port_impedances)
    return port_impedances


def _compute_port_voltages_currents(
    modeler: TerminalComponentModeler,
    sim_data: SimulationData,
) -> tuple[PortDataArray, PortDataArray]:
    """Compute voltage and current values at all ports for a single simulation.

    This function calculates the voltage and current at each monitor port from the
    electromagnetic field data in a single simulation result. The voltages and currents
    are computed according to the specific port type (e.g., lumped, wave) and are used
    as inputs for subsequent wave amplitude calculations.

    Parameters
    ----------
    modeler : :class:`.TerminalComponentModeler`
        The component modeler containing port definitions and network mapping.
    sim_data : :class:`.SimulationData`
        Simulation results containing the electromagnetic field data.

    Returns
    -------
    tuple[:class:`.PortDataArray`, :class:`.PortDataArray`]
        A tuple containing the voltage and current arrays with dimensions (f, port),
        where voltages and currents are computed for each frequency and monitor port.
    """
    network_indices = list(modeler.matrix_indices_monitor)
    values = np.zeros(
        (len(modeler.freqs), len(network_indices)),
        dtype=complex,
    )
    coords = {
        "f": np.array(modeler.freqs),
        "port": network_indices,
    }

    V_matrix = PortDataArray(values, coords=coords)
    I_matrix = V_matrix.copy(deep=True)

    waveport_cache_results = (None, None, None)
    for network_index in network_indices:
        port, mode_index = modeler.network_dict[network_index]
        if isinstance(port, WavePort):
            if waveport_cache_results[0] is not port:
                V_modes, I_modes = compute_port_VI(port, sim_data)
                waveport_cache_results = (port, V_modes, I_modes)
            V_out = waveport_cache_results[1].sel(mode_index=mode_index)
            I_out = waveport_cache_results[2].sel(mode_index=mode_index)
        else:
            V_out, I_out = compute_port_VI(port, sim_data)
        indexer = {"port": network_index}
        V_matrix = V_matrix._with_updated_data(data=V_out.data, coords=indexer)
        I_matrix = I_matrix._with_updated_data(data=I_out.data, coords=indexer)
    return (V_matrix, I_matrix)


def _compute_wave_amplitudes_from_VI(
    port_reference_impedances: PortDataArray,
    port_voltages: PortDataArray,
    port_currents: PortDataArray,
    s_param_def: SParamDef = "pseudo",
) -> tuple[PortDataArray, PortDataArray]:
    """Convert port voltages and currents to incident and reflected wave amplitudes.

    This function transforms voltage and current data at each port into forward-traveling
    (incident, 'a') and backward-traveling (reflected, 'b') wave amplitudes using the
    specified wave definition. The conversion handles impedance sign consistency and
    applies the appropriate normalization based on the chosen S-parameter definition.

    The wave amplitudes are computed using:
    - Pseudo waves: Equations 53-54 from Marks and Williams [1]
    - Power waves: Equation 4.67 from Pozar [2]

    Parameters
    ----------
    port_reference_impedances : :class:`.PortDataArray`
        Reference impedance values for each port with dimensions (f, port).
    port_voltages : :class:`.PortDataArray`
        Voltage values at each port with dimensions (f, port).
    port_currents : :class:`.PortDataArray`
        Current values at each port with dimensions (f, port).
    s_param_def : SParamDef, optional
        Wave definition type: "pseudo" for pseudo waves or "power" for power waves.
        Defaults to "pseudo".

    Returns
    -------
    tuple[:class:`.PortDataArray`, :class:`.PortDataArray`]
        A tuple containing the incident (a) and reflected (b) wave amplitude arrays,
        each with dimensions (f, port) representing the wave amplitudes at each
        frequency and port.
    """
    a = port_voltages.copy(deep=True)
    b = port_currents.copy(deep=True)
    V_numpy = port_voltages.values
    I_numpy = port_currents.values
    Z_numpy = port_reference_impedances.values

    # Check to make sure sign is consistent for all impedance values
    check_port_impedance_sign(Z_numpy)

    # Check for negative real part of port impedance and flip the V and Z signs accordingly
    negative_real_Z = np.real(Z_numpy) < 0
    V_numpy = np.where(negative_real_Z, -V_numpy, V_numpy)
    Z_numpy = np.where(negative_real_Z, -Z_numpy, Z_numpy)

    F_numpy = compute_F(Z_numpy, s_param_def)

    b_Zref = Z_numpy
    if s_param_def == "power":
        b_Zref = np.conj(Z_numpy)

    # Equations 53 and 54 from [1]
    # Equation 4.67 - Pozar - Microwave Engineering 4ed
    a.values = F_numpy * (V_numpy + Z_numpy * I_numpy)
    b.values = F_numpy * (V_numpy - b_Zref * I_numpy)

    return a, b


def compute_wave_amplitudes_at_each_port(
    modeler: TerminalComponentModeler,
    port_reference_impedances: PortDataArray,
    sim_data: SimulationData,
    s_param_def: SParamDef = "pseudo",
) -> tuple[PortDataArray, PortDataArray]:
    """Compute the incident and reflected amplitudes at each port.

    The computed amplitudes have not been normalized.

    Parameters
    ----------
    modeler : :class:`.TerminalComponentModeler`
        The component modeler defining the ports and simulation settings.
    port_reference_impedances : :class:`.PortDataArray`
        Reference impedance at each port.
    sim_data : :class:`.SimulationData`
        Results from a single simulation run.
    s_param_def : SParamDef
        The type of waves computed, either pseudo waves defined by Equation 53 and
        Equation 54 in [1], or power waves defined by Equation 4.67 in [2].

    Returns
    -------
    tuple[:class:`.PortDataArray`, :class:`.PortDataArray`]
        Incident (a) and reflected (b) wave amplitudes at each port.
    """

    port_voltages, port_currents = _compute_port_voltages_currents(modeler, sim_data)

    return _compute_wave_amplitudes_from_VI(
        port_reference_impedances, port_voltages, port_currents, s_param_def=s_param_def
    )


def compute_power_wave_amplitudes_at_each_port(
    modeler: TerminalComponentModeler,
    port_reference_impedances: PortDataArray,
    sim_data: SimulationData,
) -> tuple[PortDataArray, PortDataArray]:
    """Compute the incident and reflected power wave amplitudes at each port.

    This is a convenience function that calls
    :func:`.compute_wave_amplitudes_at_each_port` with ``s_param_def="power"``.
    The computed amplitudes have not been normalized.

    Parameters
    ----------
    modeler : :class:`.TerminalComponentModeler`
        The component modeler defining the ports and simulation settings.
    port_reference_impedances : :class:`.PortDataArray`
        Reference impedance at each port.
    sim_data : :class:`.SimulationData`
        Results from a single simulation run.

    Returns
    -------
    tuple[:class:`.PortDataArray`, :class:`.PortDataArray`]
        Incident (a) and reflected (b) power wave amplitudes at each port.
    """
    return compute_wave_amplitudes_at_each_port(
        modeler=modeler,
        port_reference_impedances=port_reference_impedances,
        sim_data=sim_data,
        s_param_def="power",
    )
