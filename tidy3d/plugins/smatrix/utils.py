"""Utility functions for S-matrix calculations and conversions.

This module provides helper functions for scattering matrix computations,
impedance conversions, and wave amplitude calculations in electromagnetic
simulations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy import linalg

from tidy3d.constants import dp_eps
from tidy3d.exceptions import Tidy3dError
from tidy3d.plugins.smatrix.data.data_array import TerminalPortDataArray

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from tidy3d.components.data.data_array import DataArray, FreqDataArray
    from tidy3d.components.data.sim_data import SimulationData
    from tidy3d.plugins.smatrix.ports.types import (
        LumpedPortType,
        PortCurrentType,
        PortVoltageType,
        TerminalPortType,
    )
    from tidy3d.plugins.smatrix.types import SParamDef


def port_array_inv(matrix: DataArray) -> NDArray:
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


def _is_diagonal_matrix(Z_numpy: np.ndarray, atol: float = dp_eps) -> bool:
    """Check if Z matrix is diagonal (off-diagonal elements are negligible).

    Parameters
    ----------
    Z_numpy : np.ndarray
        Impedance matrix with shape (f, port, port).
    atol : float
        Absolute tolerance for comparison. Default is double precision epsilon.

    Returns
    -------
    bool
        True if all off-diagonal elements are negligible.
    """
    _, num_ports, _ = Z_numpy.shape
    # Create a mask for off-diagonal elements
    mask = ~np.eye(num_ports, dtype=bool)
    # Extract off-diagonal elements for all frequencies
    off_diag = Z_numpy[:, mask]
    # Check if all off-diagonal elements are negligible
    return np.allclose(off_diag, 0, atol=atol)


def compute_F(
    Z_numpy: np.ndarray, s_param_def: SParamDef = "pseudo", compute_Finv: bool = False
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    r"""Helper to convert port impedance matrix to F (and optionally its inverse),
    which are used for computing scattering parameters. F represents the scaling
    factor applied to forward and backward waves.

    The matrix F is used when converting between S and Z parameters for circuits
    with differing port impedances. This function automatically detects whether
    Z is diagonal and uses optimized element-wise operations for diagonal matrices
    or full matrix operations for non-diagonal matrices.

    Parameters
    ----------
    Z_numpy : np.ndarray
        NumPy array of complex port impedance matrix with shape (f, port_out, port_in),
        where f is number of frequencies, port_out is number of output ports, and
        port_in is number of input ports. Z is assumed to be symmetric and positive
        semi-definite.
    s_param_def : SParamDef, optional
        Wave definition: "pseudo", "power", or "symmetric_pseudo". Default is "pseudo".
        See :class:`.TerminalComponentModeler` for details.
    compute_Finv : bool, optional
        If True, also compute and return F^{-1}. Default is False.

    Returns
    -------
    np.ndarray or tuple[np.ndarray, np.ndarray]
        F array with shape (f, port_out, port_in), or (F, Finv) tuple if
        ``compute_Finv`` is True.

    Notes
    -----
    The F matrix definitions for each wave formulation are:

    - ``"power"``: :math:`F = 0.5 \cdot (\mathrm{Re}[Z])^{-1/2}`
    - ``"pseudo"``: :math:`F = 0.5 \cdot (\mathrm{Re}(Z^{-1}))^{1/2}`
    - ``"symmetric_pseudo"``: :math:`F = 0.5 \cdot Z^{-1/2}`

    where :math:`A^{1/2}` denotes the matrix square root and :math:`A^{-1/2}` denotes
    the matrix inverse square root.
    """
    # Automatically detect if Z is diagonal and use appropriate method
    if _is_diagonal_matrix(Z_numpy):
        return _compute_F_diagonal(Z_numpy, s_param_def, compute_Finv)
    return _compute_F_full(Z_numpy, s_param_def, compute_Finv)


def _compute_F_diagonal(
    Z_numpy: np.ndarray, s_param_def: SParamDef, compute_Finv: bool = False
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Compute F matrix (and optionally Finv) assuming Z is diagonal.

    Parameters
    ----------
    Z_numpy : np.ndarray
        Impedance matrix with shape (f, port_out, port_in). Only diagonal elements are used.
    s_param_def : SParamDef
        Wave definition.
    compute_Finv : bool, optional
        If True, also compute and return Finv. Default is False.

    Returns
    -------
    np.ndarray or tuple[np.ndarray, np.ndarray]
        F matrix, or (F, Finv) tuple if ``compute_Finv`` is True.
    """
    num_freqs, num_ports, _ = Z_numpy.shape
    # Extract diagonal elements: shape (f, port)
    Z_diag = np.diagonal(Z_numpy, axis1=1, axis2=2)

    if s_param_def == "power":
        sqrt_Re_Z = np.sqrt(np.real(Z_diag))
        F_diag = 0.5 / sqrt_Re_Z
        Finv_diag = 2.0 * sqrt_Re_Z if compute_Finv else None
    elif s_param_def == "pseudo":
        sqrt_Re_Z = np.sqrt(np.real(Z_diag))
        abs_Z = np.abs(Z_diag)
        F_diag = sqrt_Re_Z / (2.0 * abs_Z)
        Finv_diag = 2.0 * abs_Z / sqrt_Re_Z if compute_Finv else None
    elif s_param_def == "symmetric_pseudo":
        sqrt_Z = np.sqrt(Z_diag)
        F_diag = 0.5 / sqrt_Z
        Finv_diag = 2.0 * sqrt_Z if compute_Finv else None
    else:
        raise ValueError(
            f"Unsupported S-parameter definition '{s_param_def}'. "
            "Supported values are 'pseudo', 'symmetric_pseudo', and 'power'."
        )

    # Convert back to diagonal matrices: shape (f, port, port)
    idx = range(num_ports)
    F = np.zeros((num_freqs, num_ports, num_ports), dtype=F_diag.dtype)
    F[:, idx, idx] = F_diag
    if not compute_Finv:
        return F
    Finv = np.zeros((num_freqs, num_ports, num_ports), dtype=Finv_diag.dtype)
    Finv[:, idx, idx] = Finv_diag
    return F, Finv


def _compute_F_full(
    Z_numpy: np.ndarray, s_param_def: SParamDef, compute_Finv: bool = False
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Compute F matrix (and optionally Finv) for full (non-diagonal) Z.

    Parameters
    ----------
    Z_numpy : np.ndarray
        Full impedance matrix with shape (f, port_out, port_in).
    s_param_def : SParamDef
        Wave definition.
    compute_Finv : bool, optional
        If True, also compute and return Finv. Default is False.

    Returns
    -------
    np.ndarray or tuple[np.ndarray, np.ndarray]
        F matrix, or (F, Finv) tuple if ``compute_Finv`` is True.
    """
    num_freqs = Z_numpy.shape[0]

    if s_param_def == "power":
        # F = 0.5 * Re(Z)^{-1/2}, Finv = 2 * Re(Z)^{1/2}
        Z_real = np.real(Z_numpy)
        sqrt_Re_Z = np.array([linalg.sqrtm(Z_real[f_idx]) for f_idx in range(num_freqs)])
        F = np.array([0.5 * linalg.inv(sqrt_Re_Z[f_idx]) for f_idx in range(num_freqs)])
        if not compute_Finv:
            return F
        return F, 2.0 * sqrt_Re_Z
    elif s_param_def == "pseudo":
        # F = 0.5 * Re(Z^{-1})^{1/2}, Finv = 2 * Re(Z^{-1})^{-1/2}
        sqrt_Re_Zinv = np.array(
            [linalg.sqrtm(np.real(np.linalg.inv(Z_numpy[f_idx]))) for f_idx in range(num_freqs)]
        )
        F = 0.5 * sqrt_Re_Zinv
        if not compute_Finv:
            return F
        # Finv still requires matrix inversion
        Finv = np.array([2.0 * linalg.inv(sqrt_Re_Zinv[f_idx]) for f_idx in range(num_freqs)])
        return F, Finv
    elif s_param_def == "symmetric_pseudo":
        # F = 0.5 * Z^{-1/2}, Finv = 2 * Z^{1/2}
        sqrt_Z = np.array([linalg.sqrtm(Z_numpy[f_idx]) for f_idx in range(num_freqs)])
        F = np.array([0.5 * linalg.inv(sqrt_Z[f_idx]) for f_idx in range(num_freqs)])
        if not compute_Finv:
            return F
        return F, 2.0 * sqrt_Z
    else:
        raise ValueError(
            f"Unsupported S-parameter definition '{s_param_def}'. "
            "Supported values are 'pseudo', 'symmetric_pseudo', and 'power'."
        )


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
    reference: complex | TerminalPortDataArray,
    s_param_def: SParamDef = "pseudo",
) -> DataArray:
    """Get the impedance matrix given the scattering matrix and a reference impedance.

    This function converts an S-matrix to a Z-matrix. It handles both a single
    uniform reference impedance and a generalized per-frequency reference impedance matrix.

    Parameters
    ----------
    s_matrix : :class:`.TerminalPortDataArray`
        Scattering matrix computed using either the pseudo or power wave formulation.
    reference : Union[complex, :class:`.TerminalPortDataArray`]
        The reference impedance. Either a scalar (uniform across all ports) or a
        full impedance matrix per frequency.
    s_param_def : SParamDef, optional
        Wave definition: "pseudo", "power", or "symmetric_pseudo". Default is "pseudo".
        See :class:`.TerminalComponentModeler` for details.

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

    if isinstance(reference, TerminalPortDataArray):
        Zport = reference.transpose(*TerminalPortDataArray._dims).values
        F, Finv = compute_F(Zport, s_param_def, compute_Finv=True)
        FinvSF = np.matmul(np.matmul(Finv, s_vals), F)
    else:
        Zport = reference
        # F^{-1} S F = S when F is the same scalar for all ports
        FinvSF = s_vals

    # Use conjugate when S matrix is power-wave based
    if s_param_def == "power":
        Zport_mod = np.conj(Zport)
    else:  # both pseudo and symmetric pseudo use this
        Zport_mod = Zport

    # From equation 74 from [1] for pseudo waves
    # From Equation 4.68 - Pozar - Microwave Engineering 4ed for power waves
    if isinstance(reference, TerminalPortDataArray):
        RHS = Zport_mod + np.matmul(FinvSF, Zport)
    else:
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
