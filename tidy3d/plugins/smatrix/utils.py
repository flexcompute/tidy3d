"""Utility functions for S-matrix calculations and conversions.

This module provides helper functions for scattering matrix computations,
impedance conversions, and wave amplitude calculations in electromagnetic
simulations.
"""

from __future__ import annotations

from typing import Union

import numpy as np

from tidy3d.components.data.data_array import (
    DataArray,
    FreqDataArray,
)
from tidy3d.components.data.sim_data import SimulationData
from tidy3d.components.types import ArrayFloat1D
from tidy3d.exceptions import Tidy3dError
from tidy3d.plugins.smatrix.data.data_array import PortDataArray, TerminalPortDataArray
from tidy3d.plugins.smatrix.ports.types import (
    LumpedPortType,
    PortCurrentType,
    PortVoltageType,
    TerminalPortType,
)
from tidy3d.plugins.smatrix.types import SParamDef


def port_array_inv(matrix: DataArray):
    """Helper to invert a port matrix.

    Parameters
    ----------
    matrix : DataArray
        The matrix to invert.

    Returns
    -------
    np.ndarray
        The inverted matrix.
    """
    return np.linalg.inv(matrix)


def ab_to_s(
    a_matrix: TerminalPortDataArray, b_matrix: TerminalPortDataArray
) -> TerminalPortDataArray:
    """Get the scattering matrix given the wave amplitude matrices.

    Parameters
    ----------
    a_matrix : TerminalPortDataArray
        Matrix of incident power wave amplitudes.
    b_matrix : TerminalPortDataArray
        Matrix of reflected power wave amplitudes.

    Returns
    -------
    TerminalPortDataArray
        The computed scattering (S) matrix.
    """
    validate_square_matrix(a_matrix, "ab_to_s")
    # Ensure dimensions are ordered properly
    a_matrix = a_matrix.transpose(*TerminalPortDataArray._dims)
    b_matrix = b_matrix.transpose(*TerminalPortDataArray._dims)

    s_matrix = a_matrix.copy(deep=True)
    a_vals = s_matrix.copy(deep=True).values
    b_vals = b_matrix.copy(deep=True).values

    s_vals = np.matmul(b_vals, port_array_inv(a_vals))

    s_matrix.data = s_vals
    return s_matrix


def check_port_impedance_sign(Z_numpy: np.ndarray) -> None:
    """Sanity check for consistent sign of real part of Z for each port.

    This check iterates through each port and ensures that the sign of the real
    part of its impedance does not change across all frequencies. A sign change
    can indicate an unphysical result or numerical instability.

    Parameters
    ----------
    Z_numpy : np.ndarray
        NumPy array of impedance values with shape (num_freqs, num_ports).

    Raises
    ------
    Tidy3dError
        If an inconsistent sign of the real part of the impedance
        is detected for any port.
    """
    for port_idx in range(Z_numpy.shape[1]):
        port_Z = Z_numpy[:, port_idx]
        signs = np.sign(np.real(port_Z))
        if not np.all(signs == signs[0]):
            raise Tidy3dError(
                f"Inconsistent sign of real part of Z detected for port {port_idx}. "
                "If you received this error, please create an issue in the Tidy3D "
                "github repository."
            )


def compute_F(Z_numpy: ArrayFloat1D, s_param_def: SParamDef = "pseudo"):
    r"""Helper to convert port impedance matrix to F, which is used for
    computing scattering parameters

    The matrix F is used when converting between S and Z parameters for circuits
    with differing port impedances. Its diagonal elements are defined as

    .. math::

        F_{kk} = 1 / (2 * \sqrt{Re(Z_k)})


    Parameters
    ----------
    Z_numpy : ArrayFloat1D
        NumPy array of complex port impedances.
    s_param_def : SParamDef, optional
        The type of wave amplitudes, by default "pseudo".

    Returns
    -------
    ArrayFloat1D
        NumPy array containing the computed F values.
    """
    # Defined in [2] after equation 4.67
    if s_param_def == "power":
        return 1.0 / (2.0 * np.sqrt(np.real(Z_numpy)))
    # Equation 75 from [1]
    return np.sqrt(np.real(Z_numpy)) / (2.0 * np.abs(Z_numpy))


def compute_port_VI(
    port_out: TerminalPortType, sim_data: SimulationData
) -> tuple[PortVoltageType, PortCurrentType]:
    """Compute the port voltages and currents.

    Parameters
    ----------
    port_out : ``TerminalPortType``
        Port for computing voltage and current.
    sim_data : :class:`.SimulationData`
        Results from simulation containing field data.

    Returns
    -------
    tuple[PortVoltageType, PortCurrentType]
        Voltage and current values at the port as frequency arrays.
    """
    voltage = port_out.compute_voltage(sim_data)
    current = port_out.compute_current(sim_data)
    return voltage, current


def compute_power_wave_amplitudes(
    port: LumpedPortType, sim_data: SimulationData
) -> tuple[FreqDataArray, FreqDataArray]:
    r"""Calculates the unnormalized power wave amplitudes from port voltage (V),
    current (I), and impedance (Z0) using:

    .. math::

        a = (V + Z0*I) / (2 * \sqrt(Re(Z0)))
        b = (V - Z0*I) / (2 * \sqrt(Re(Z0)))

    Parameters
    ----------
    port : :class:`.LumpedPortType`
        Port for computing voltage and current.
    sim_data : :class:`.SimulationData`
        Results from the simulation.

    Returns
    -------
    tuple[FreqDataArray, FreqDataArray]
        Incident (a) and reflected (b) power wave amplitude frequency arrays.
    """
    voltage, current = compute_port_VI(port, sim_data)
    # Amplitudes for the incident and reflected power waves
    a = (voltage + port.impedance * current) / 2 / np.sqrt(np.real(port.impedance))
    b = (voltage - port.impedance * current) / 2 / np.sqrt(np.real(port.impedance))
    return a, b


def compute_power_delivered_by_port(
    port: LumpedPortType, sim_data: SimulationData
) -> FreqDataArray:
    """Compute the power delivered to the network by a lumped port.
    The power is calculated as the incident power minus the reflected power:

    .. math::
        P = 0.5 * (|a|^2 - |b|^2)

    Parameters
    ----------
    port : :class:`.LumpedPortType`
        Port for computing voltage and current.
    sim_data : :class:`.SimulationData`
        Results from the simulation.

    Returns
    -------
    FreqDataArray
        Power in units of Watts as a frequency array.
    """
    a, b = compute_power_wave_amplitudes(port=port, sim_data=sim_data)
    # Power delivered is the incident power minus the reflected power
    return 0.5 * (np.abs(a) ** 2 - np.abs(b) ** 2)


def s_to_z(
    s_matrix: TerminalPortDataArray,
    reference: Union[complex, PortDataArray],
    s_param_def: SParamDef = "pseudo",
) -> DataArray:
    """Get the impedance matrix given the scattering matrix and a reference impedance.

    This function converts an S-matrix to a Z-matrix. It handles both a single
    uniform reference impedance and generalized per-port reference impedances.

    Parameters
    ----------
    s_matrix : :class:`.TerminalPortDataArray`
        Scattering matrix computed using either the pseudo or power wave formulation.
    reference : Union[complex, :class:`.PortDataArray`]
        The reference impedance used at each port.
    s_param_def : SParamDef, optional
        The type of wave amplitudes used for computing the scattering matrix, either pseudo waves
        defined by Equation 53 and Equation 54 in [1] or power waves defined by Equation 4.67 in [2].
        By default "pseudo".

    Returns
    -------
    DataArray
        The computed impedance (Z) matrix.

    Examples
    --------
    The `s_to_z` function is a standalone utility that requires an S-matrix as input.
    This is useful if you have S-matrix data generated
    externally from a :class:`.TerminalComponentModelerData` and want to compare them.

    >>> z_matrix = s_to_z(s_matrix=s_matrix, reference=50, s_param_def="power") # doctest: +SKIP
    >>> z_11 = z_matrix.sel(port_out="port_1", port_in="port_1") # doctest: +SKIP
    """
    validate_square_matrix(s_matrix, "s_to_z")
    # Ensure dimensions are ordered properly
    z_matrix = s_matrix.transpose(*TerminalPortDataArray._dims).copy(deep=True)
    s_vals = z_matrix.values
    eye = np.eye(len(s_matrix.port_out.values), len(s_matrix.port_in.values))[np.newaxis, :, :]
    # Ensure that Zport, F, and Finv act as diagonal matrices when multiplying by left or right
    shape_left = (len(s_matrix.f), len(s_matrix.port_out), 1)
    shape_right = (len(s_matrix.f), 1, len(s_matrix.port_in))
    # Setup the port reference impedance array (scalar)
    if isinstance(reference, PortDataArray):
        Zport = reference.values.reshape(shape_right)
        F = compute_F(Zport, s_param_def).reshape(shape_right)
        Finv = (1.0 / F).reshape(shape_left)
    else:
        Zport = reference
        F = compute_F(Zport, s_param_def)
        Finv = 1.0 / F
    # Use conjugate when S matrix is power-wave based
    if s_param_def == "power":
        Zport_mod = np.conj(Zport)
    else:
        Zport_mod = Zport

    # From equation 74 from [1] for pseudo waves
    # From Equation 4.68 - Pozar - Microwave Engineering 4ed for power waves
    FinvSF = Finv * s_vals * F
    RHS = eye * Zport_mod + FinvSF * Zport
    LHS = eye - FinvSF
    z_vals = np.linalg.solve(LHS, RHS)

    z_matrix.data = z_vals
    return z_matrix


def validate_square_matrix(matrix: TerminalPortDataArray, method_name: str) -> None:
    """Check if the matrix has equal input and output port dimensions.

    Parameters
    ----------
    matrix : TerminalPortDataArray
        Matrix to validate
    method_name : str
        Name of the calling method for error message

    Raises
    ------
    Tidy3dError
        If the matrix is not square (unequal input/output dimensions).
    """
    n_out = len(matrix.port_out)
    n_in = len(matrix.port_in)
    if n_out != n_in:
        raise Tidy3dError(
            f"Cannot compute {method_name}: number of input ports ({n_in}) "
            f"!= the number of output ports ({n_out}). This usually means the 'TerminalComponentModeler' "
            "was run with only a subset of port excitations. Please ensure that the `run_only` field in "
            "the 'TerminalComponentModeler' is not being used."
        )
