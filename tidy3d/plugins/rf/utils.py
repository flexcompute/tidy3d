from __future__ import annotations

from typing import Union

import numpy as np

from tidy3d.components.data.data_array import DataArray, FreqDataArray
from tidy3d.components.data.sim_data import SimulationData
from tidy3d.exceptions import Tidy3dError
from tidy3d.plugins.rf.data.data_array import PortDataArray, TerminalPortDataArray
from tidy3d.plugins.rf.ports.coaxial_lumped import CoaxialLumpedPort
from tidy3d.plugins.rf.ports.rectangular_lumped import LumpedPort
from tidy3d.plugins.smatrix.component_modelers.base import (
    AbstractComponentModeler,
)


def ab_to_s(
    a_matrix: TerminalPortDataArray, b_matrix: TerminalPortDataArray
) -> TerminalPortDataArray:
    a_matrix = a_matrix.transpose(*TerminalPortDataArray._dims)
    b_matrix = b_matrix.transpose(*TerminalPortDataArray._dims)
    s_matrix = a_matrix.copy(deep=True)
    a_vals = s_matrix.copy(deep=True).values
    b_vals = b_matrix.copy(deep=True).values
    s_vals = np.matmul(b_vals, AbstractComponentModeler.inv(a_vals))
    s_matrix.data = s_vals
    return s_matrix


def check_port_impedance_sign(Z_numpy: np.ndarray):
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
    return 1.0 / (2.0 * np.sqrt(np.real(Z_numpy)))


def compute_port_VI(
    port_out: Union[LumpedPort, CoaxialLumpedPort], sim_data: SimulationData
) -> tuple[FreqDataArray, FreqDataArray]:
    voltage = port_out.compute_voltage(sim_data)
    current = port_out.compute_current(sim_data)
    return voltage, current


def compute_power_wave_amplitudes(
    port: Union[LumpedPort, CoaxialLumpedPort], sim_data: SimulationData
) -> tuple[FreqDataArray, FreqDataArray]:
    voltage, current = compute_port_VI(port, sim_data)
    a = (voltage + port.impedance * current) / 2 / np.sqrt(np.real(port.impedance))
    b = (voltage - port.impedance * current) / 2 / np.sqrt(np.real(port.impedance))
    return a, b


def compute_power_delivered_by_port(
    port: Union[LumpedPort, CoaxialLumpedPort], sim_data: SimulationData
) -> FreqDataArray:
    a, b = compute_power_wave_amplitudes(sim_data=sim_data, port=port)
    return 0.5 * (np.abs(a) ** 2 - np.abs(b) ** 2)


def s_to_z(s_matrix: TerminalPortDataArray, reference: Union[complex, PortDataArray]) -> DataArray:
    z_matrix = s_matrix.transpose(*TerminalPortDataArray._dims).copy(deep=True)
    s_vals = z_matrix.values
    eye = np.eye(len(s_matrix.port_out.values), len(s_matrix.port_in.values))
    if isinstance(reference, PortDataArray):
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
        z_vals = np.matmul(AbstractComponentModeler.inv(eye - s_vals), (eye + s_vals)) * reference
    z_matrix.data = z_vals
    return z_matrix
