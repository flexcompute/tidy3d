from __future__ import annotations

from typing import Union

import numpy as np

from tidy3d.components.data.data_array import DataArray, FreqDataArray
from tidy3d.components.data.sim_data import SimulationData
from tidy3d.exceptions import Tidy3dError
from tidy3d.plugins.smatrix.component_modelers.base import (
    AbstractComponentModeler,
    TerminalPortType,
)
from tidy3d.plugins.smatrix.data.data_array import PortDataArray, TerminalPortDataArray
from tidy3d.plugins.smatrix.ports.coaxial_lumped import CoaxialLumpedPort
from tidy3d.plugins.smatrix.ports.rectangular_lumped import LumpedPort


def ab_to_s(
    a_matrix: TerminalPortDataArray, b_matrix: TerminalPortDataArray
) -> TerminalPortDataArray:
    """Get the scattering matrix given the power wave matrices.

    The scattering matrix S is computed from the incident (a) and reflected (b)
    power wave amplitude matrices using the formula :math:`S = b * a^-1`.

    Args:
        a_matrix: Matrix of incident power wave amplitudes.
        b_matrix: Matrix of reflected power wave amplitudes.

    Returns:
        The computed scattering (S) matrix.
    """
    # Ensure dimensions are ordered properly
    a_matrix = a_matrix.transpose(*TerminalPortDataArray._dims)
    b_matrix = b_matrix.transpose(*TerminalPortDataArray._dims)

    s_matrix = a_matrix.copy(deep=True)
    a_vals = s_matrix.copy(deep=True).values
    b_vals = b_matrix.copy(deep=True).values

    s_vals = np.matmul(b_vals, AbstractComponentModeler.inv(a_vals))

    s_matrix.data = s_vals
    return s_matrix


def check_port_impedance_sign(Z_numpy: np.ndarray):
    """Sanity check for consistent sign of real part of Z for each port.

    This check iterates through each port and ensures that the sign of the real
    part of its impedance does not change across all frequencies. A sign change
    can indicate an unphysical result or numerical instability.

    Args:
        Z_numpy: NumPy array of impedance values with shape (num_freqs, num_ports).

    Raises:
        Tidy3dError: If an inconsistent sign of the real part of the impedance
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


def compute_F(Z_numpy: np.array):
    r"""Helper to convert port impedance matrix to F for generalized S-parameters.

    The matrix F is used when converting between S and Z parameters for circuits
    with differing port impedances. Its diagonal elements are defined as
    :math:`F_{kk} = 1 / (2 * \sqrt{Re(Z_k)})`.

    Args:
        Z_numpy: NumPy array of complex port impedances.

    Returns:
        NumPy array containing the computed F values.
    """
    return 1.0 / (2.0 * np.sqrt(np.real(Z_numpy)))


def compute_port_VI(
    port_out: TerminalPortType, sim_data: SimulationData
) -> tuple[FreqDataArray, FreqDataArray]:
    """Compute the port voltages and currents.

    Parameters
    ----------
    port_out : ``TerminalPortType``
        Port for computing voltage and current.
    sim_data : :class:`.SimulationData`
        Results from simulation containing field data.

    Returns
    -------
    tuple[FreqDataArray, FreqDataArray]
        Voltage and current values at the port as frequency arrays.
    """
    voltage = port_out.compute_voltage(sim_data)
    current = port_out.compute_current(sim_data)
    return voltage, current


def compute_power_wave_amplitudes(
    port: Union[LumpedPort, CoaxialLumpedPort], sim_data: SimulationData
) -> tuple[FreqDataArray, FreqDataArray]:
    """Calculates the unnormalized power wave amplitudes from port voltage (V),
    current (I), and impedance (Z0) using:

    .. math::
        a = (V + Z0*I) / (2 * sqrt(Re(Z0)))
        b = (V - Z0*I) / (2 * sqrt(Re(Z0)))

    Parameters
    ----------
    port : Union[:class:`.LumpedPort`, :class:`.CoaxialLumpedPort`]
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
    port: Union[LumpedPort, CoaxialLumpedPort], sim_data: SimulationData
) -> FreqDataArray:
    """Compute the power delivered to the network by a lumped port.

    The power is calculated as the incident power minus the reflected power:
    P = 0.5 * (|a|^2 - |b|^2).

    Parameters
    ----------
    port : Union[:class:`.LumpedPort`, :class:`.CoaxialLumpedPort`]
        Port for computing voltage and current.
    sim_data : :class:`.SimulationData`
        Results from the simulation.

    Returns
    -------
    FreqDataArray
        Power in units of Watts as a frequency array.
    """
    a, b = compute_power_wave_amplitudes(sim_data=sim_data, port=port)
    # Power delivered is the incident power minus the reflected power
    return 0.5 * (np.abs(a) ** 2 - np.abs(b) ** 2)


def s_to_z(s_matrix: TerminalPortDataArray, reference: Union[complex, PortDataArray]) -> DataArray:
    """Get the impedance matrix given the scattering matrix and a reference impedance.

    This function converts an S-matrix to a Z-matrix. It handles both a single
    uniform reference impedance and generalized per-port reference impedances.

    Args:
        s_matrix: The scattering (S) matrix to convert.
        reference: The reference impedance. Can be a single complex value for all
            ports or a PortDataArray for per-port impedances.

    Returns:
        The computed impedance (Z) matrix as a DataArray.
    """

    # Ensure dimensions are ordered properly
    z_matrix = s_matrix.transpose(*TerminalPortDataArray._dims).copy(deep=True)
    s_vals = z_matrix.values
    eye = np.eye(len(s_matrix.port_out.values), len(s_matrix.port_in.values))
    if isinstance(reference, PortDataArray):
        # From Equation 4.68 - Pozar - Microwave Engineering 4ed
        # Ensure that Zport, F, and Finv act as diagonal matrices when multiplying by left or right
        shape_left = (len(s_matrix.f), len(s_matrix.port_out), 1)
        shape_right = (len(s_matrix.f), 1, len(s_matrix.port_in))
        Zport = reference.values.reshape(shape_right)
        F = compute_F(Zport).reshape(shape_right)
        Finv = (1.0 / F).reshape(shape_left)
        FinvSF = Finv * s_vals * F
        RHS = eye * np.conj(Zport) + FinvSF * Zport
        LHS = eye - FinvSF
        z_vals = np.matmul(AbstractComponentModeler.inv(LHS), RHS)
    else:
        # Simpler case when all port impedances are the same
        z_vals = np.matmul(AbstractComponentModeler.inv(eye - s_vals), (eye + s_vals)) * reference

    z_matrix.data = z_vals
    return z_matrix
