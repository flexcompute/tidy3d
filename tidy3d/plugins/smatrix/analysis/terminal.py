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

from typing import TYPE_CHECKING

import numpy as np

from tidy3d.plugins.smatrix.data.data_array import PortDataArray, TerminalPortDataArray
from tidy3d.plugins.smatrix.ports.base_lumped import AbstractLumpedPort
from tidy3d.plugins.smatrix.ports.wave import TerminalWavePort, WavePort
from tidy3d.plugins.smatrix.utils import (
    ab_to_s,
    check_port_impedance_sign,
    compute_F,
    compute_port_VI,
)

if TYPE_CHECKING:
    from tidy3d.components.data.sim_data import SimulationData
    from tidy3d.plugins.smatrix.component_modelers.terminal import TerminalComponentModeler
    from tidy3d.plugins.smatrix.data.data_array import TerminalPortDataArray
    from tidy3d.plugins.smatrix.data.terminal import TerminalComponentModelerData
    from tidy3d.plugins.smatrix.types import SParamDef


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
        Wave definition: "pseudo", "power", or "symmetric_pseudo". Default is "pseudo".
        See :class:`.TerminalComponentModeler` for details.

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
    elif s_param_def == "symmetric_pseudo":
        a_matrix, b_matrix = modeler_data.port_symmetric_pseudo_wave_matrices
    elif s_param_def == "power":
        a_matrix, b_matrix = modeler_data.port_power_wave_matrices
    else:
        raise ValueError(
            f"Unsupported S-parameter definition '{s_param_def}'. "
            "Supported values are 'pseudo', 'symmetric_pseudo', and 'power'."
        )

    # If excitation is assumed ideal, a_matrix is assumed to be diagonal
    # and the explicit inverse can be avoided. When only a subset of excitations
    # have been run, we cannot find the inverse anyways so must make this assumption.
    if len(monitor_indices) == len(run_source_indices) and not assume_ideal_excitation:
        s_matrix = ab_to_s(a_matrix, b_matrix)
    else:
        # Extract self-coupling terms a[j, j] for each run source port j.
        # np.diagonal won't work here because the matrix may not be square and
        # the row order (monitor_indices) may differ from column order (run_source_indices).
        diag_row_positions = [monitor_indices.index(src) for src in run_source_indices]
        diag_col_positions = list(range(len(run_source_indices)))
        a_diag = a_matrix.values[:, diag_row_positions, diag_col_positions]
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


def port_reference_impedances(modeler_data: TerminalComponentModelerData) -> TerminalPortDataArray:
    """Calculates the reference impedance matrix for each port across all frequencies.

    This function determines the characteristic impedance for every port defined
    in the modeler. It returns a matrix with dimensions (f, port_out, port_in)
    to support coupled impedances from :class:`.TerminalWavePort`. For
    :class:`.WavePort` and :class:`.LumpedPort`, the impedance matrix is diagonal.

    Parameters
    ----------
    modeler_data : TerminalComponentModelerData
        Data object containing the modeler definition and the raw
        simulation data needed for impedance calculations.

    Returns
    -------
    TerminalPortDataArray
        A ``TerminalPortDataArray`` containing the complex impedance matrix
        with dimensions (f, port_out, port_in) for each frequency.
    """
    network_indices = list(modeler_data.modeler.matrix_indices_monitor)
    num_ports = len(network_indices)
    num_freqs = len(modeler_data.modeler.freqs)

    # Initialize 3D array: (freq, port_out, port_in)
    values = np.zeros((num_freqs, num_ports, num_ports), dtype=complex)

    coords = {
        "f": np.array(modeler_data.modeler.freqs),
        "port_out": network_indices,
        "port_in": network_indices,
    }
    port_impedances = TerminalPortDataArray(values, coords=coords)

    # Each simulation will store the results from the ModeMonitors,
    # so here we just choose the first one.
    first_sim_index = modeler_data.modeler.matrix_indices_run_sim[0]
    port, selection_index = modeler_data.modeler.network_dict[first_sim_index]
    if isinstance(port, TerminalWavePort):
        task_name = modeler_data.modeler.get_task_name(port=port, terminal_label=selection_index)
    elif isinstance(port, WavePort):
        task_name = modeler_data.modeler.get_task_name(port=port, mode_index=selection_index)
    else:
        task_name = modeler_data.modeler.get_task_name(port=port)
    sim_data = modeler_data.data[task_name]

    # Track which ports have been processed to avoid redundant computation.
    # Each network index represents a single mode/terminal, but get_reference_impedance_matrix
    # returns the full matrix for all modes/terminals of that port. Process each port once
    # and extract all matrix elements to populate the impedance array.
    processed_ports = set()

    for network_index in network_indices:
        port, _ = modeler_data.modeler.network_dict[network_index]

        if isinstance(port, (WavePort, TerminalWavePort)):
            # get_reference_impedance_matrix returns the full impedance matrix.
            # Only process each port once to handle full matrix
            if port.name not in processed_ports:
                processed_ports.add(port.name)
                ref_Z0_matrix = port.get_reference_impedance_matrix(sim_data)

                # Extract dimension names (e.g., mode_index_out/in or terminal_label_out/in)
                is_mode_based = "mode_index_out" in ref_Z0_matrix.dims
                # Determine kwarg name for network_index based on dimension type
                idx_kwarg = "mode_index" if is_mode_based else "terminal_label"
                idx_out_name = f"{idx_kwarg}_out"
                idx_in_name = f"{idx_kwarg}_in"

                # Identify valid output and input indices and their corresponding network indices
                valid_out = [
                    (idx, modeler_data.modeler.network_index(port, **{idx_kwarg: idx}))
                    for idx in ref_Z0_matrix.coords[idx_out_name].values
                    if modeler_data.modeler.network_index(port, **{idx_kwarg: idx})
                    in network_indices
                ]

                valid_in = [
                    (idx, modeler_data.modeler.network_index(port, **{idx_kwarg: idx}))
                    for idx in ref_Z0_matrix.coords[idx_in_name].values
                    if modeler_data.modeler.network_index(port, **{idx_kwarg: idx})
                    in network_indices
                ]

                if valid_out and valid_in:
                    local_out, net_out = zip(*valid_out)
                    local_in, net_in = zip(*valid_in)

                    # Extract submatrix and transpose to ensure (freq, out, in) order
                    dims_other = [
                        d for d in ref_Z0_matrix.dims if d not in (idx_out_name, idx_in_name)
                    ]
                    sub_matrix = ref_Z0_matrix.sel(
                        {idx_out_name: list(local_out), idx_in_name: list(local_in)}
                    ).transpose(*dims_other, idx_out_name, idx_in_name)

                    # Assign to port_impedances using vectorized indexing
                    port_impedances.loc[{"port_out": list(net_out), "port_in": list(net_in)}] = (
                        sub_matrix.values
                    )

        elif isinstance(port, AbstractLumpedPort):
            # LumpedPorts have a constant diagonal reference impedance
            data = np.full(num_freqs, port.impedance)
            indexer = {"port_out": network_index, "port_in": network_index}
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
    are computed according to the specific port type (e.g., lumped, wave, terminal wave)
    and are used as inputs for subsequent wave amplitude calculations.

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

    wave_port_cache_results = (None, None, None)

    for network_index in network_indices:
        port, selection_index = modeler.network_dict[network_index]

        if isinstance(port, (WavePort, TerminalWavePort)):
            # Both WavePort and TerminalWavePort use similar logic with different index names
            if wave_port_cache_results[0] is not port:
                V_data, I_data = compute_port_VI(port, sim_data)
                wave_port_cache_results = (port, V_data, I_data)

            # Use mode_index for WavePort, terminal_label for TerminalWavePort
            index_dim = "mode_index" if isinstance(port, WavePort) else "terminal_label"
            V_out = wave_port_cache_results[1].sel({index_dim: selection_index})
            I_out = wave_port_cache_results[2].sel({index_dim: selection_index})

        else:
            # LumpedPort: direct voltage/current computation
            V_out, I_out = compute_port_VI(port, sim_data)

        indexer = {"port": network_index}
        V_matrix = V_matrix._with_updated_data(data=V_out.data, coords=indexer)
        I_matrix = I_matrix._with_updated_data(data=I_out.data, coords=indexer)

    return (V_matrix, I_matrix)


def _compute_wave_amplitudes_from_VI(
    port_reference_impedances: TerminalPortDataArray,
    port_voltages: PortDataArray,
    port_currents: PortDataArray,
    s_param_def: SParamDef = "pseudo",
) -> tuple[PortDataArray, PortDataArray]:
    """Convert port voltages and currents to incident and reflected wave amplitudes.

    This function transforms voltage and current data at each port into forward-traveling
    (incident, 'a') and backward-traveling (reflected, 'b') wave amplitudes using the
    specified wave definition. The conversion handles impedance sign consistency and
    applies the appropriate normalization based on the chosen S-parameter definition.

    For coupled systems (off-diagonal Z terms), the wave amplitudes are computed using
    matrix multiplication to account for inter-port coupling.

    The wave amplitudes are computed using:
    - Pseudo waves: Equations 53-54 from Marks and Williams [1]
    - Power waves: Equation 4.67 from Pozar [2]

    Parameters
    ----------
    port_reference_impedances : :class:`.TerminalPortDataArray`
        Reference impedance matrix with dimensions (f, port_out, port_in).
        Can include off-diagonal coupling terms for TerminalWavePort.
    port_voltages : :class:`.PortDataArray`
        Voltage values at each port with dimensions (f, port).
    port_currents : :class:`.PortDataArray`
        Current values at each port with dimensions (f, port).
    s_param_def : SParamDef, optional
        Wave definition: "pseudo", "power", or "symmetric_pseudo". Default is "pseudo".
        See :class:`.TerminalComponentModeler` for details.

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

    # Extract diagonal for F computation and sign handling
    Z_diag = np.diagonal(Z_numpy, axis1=1, axis2=2)

    # Check to make sure sign is consistent for all impedance values (checks diagonal)
    check_port_impedance_sign(Z_diag)

    # Check for negative real part of port impedance on diagonal and flip signs accordingly
    negative_real_Z_diag = np.real(Z_diag) < 0
    V_numpy = np.where(negative_real_Z_diag, -V_numpy, V_numpy)

    # Also apply sign changes in Z for negative diagonal
    sign_vec = np.where(negative_real_Z_diag, -1, 1)
    # Apply sign to Z: Z --> sign @ Z (flip rows for negative diagonal)
    if np.any(sign_vec == -1):
        # sign_vec has shape (f, port), Z_numpy has shape (f, port_out, port_in)
        # Apply: Z_new[f,i,j] = sign[f,i] * Z[f,i,j]
        Z_numpy = sign_vec[:, :, np.newaxis] * Z_numpy

    # Compute F
    F_numpy = compute_F(Z_numpy, s_param_def)

    # Matrix multiplication: Z @ I for full coupling support
    # Z_numpy has shape (f, port_out, port_in), I_numpy has shape (f, port)
    # Result ZI has shape (f, port_out)
    ZI = np.einsum("fij,fj->fi", Z_numpy, I_numpy)

    # For power waves, use conjugate of Z for reflected wave
    if s_param_def == "power":
        b_Zref_I = np.einsum("fij,fj->fi", np.conj(Z_numpy), I_numpy)
    else:
        b_Zref_I = ZI

    # Equations 53 and 54 from [1]
    # Equation 4.67 - Pozar - Microwave Engineering 4ed
    # Generalized for matrix Z: a = F @ (V + Z @ I), b = F @ (V - Z* @ I)
    a.values = np.einsum("fij,fj->fi", F_numpy, V_numpy + ZI)
    b.values = np.einsum("fij,fj->fi", F_numpy, V_numpy - b_Zref_I)

    return a, b


def compute_wave_amplitudes_at_each_port(
    modeler: TerminalComponentModeler,
    port_reference_impedances: TerminalPortDataArray,
    sim_data: SimulationData,
    s_param_def: SParamDef = "pseudo",
) -> tuple[PortDataArray, PortDataArray]:
    """Compute the incident and reflected amplitudes at each port.

    The computed amplitudes have not been normalized.

    Parameters
    ----------
    modeler : :class:`.TerminalComponentModeler`
        The component modeler defining the ports and simulation settings.
    port_reference_impedances : :class:`.TerminalPortDataArray`
        Reference impedance matrix with dimensions (f, port_out, port_in).
    sim_data : :class:`.SimulationData`
        Results from a single simulation run.
    s_param_def : SParamDef
        Wave definition: "pseudo", "power", or "symmetric_pseudo".
        See :class:`.TerminalComponentModeler` for details.

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
    port_reference_impedances: TerminalPortDataArray,
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
    port_reference_impedances : :class:`.TerminalPortDataArray`
        Reference impedance matrix with dimensions (f, port_out, port_in).
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
