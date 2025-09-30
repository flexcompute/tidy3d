from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pydantic.v1 as pd
import pytest
import skrf
import xarray as xr

import tidy3d as td
import tidy3d.plugins.smatrix.analysis.terminal
import tidy3d.plugins.smatrix.data.terminal
import tidy3d.plugins.smatrix.utils
from tidy3d import SimulationDataMap
from tidy3d.components.boundary import BroadbandModeABCSpec
from tidy3d.components.data.data_array import FreqDataArray
from tidy3d.exceptions import SetupError, Tidy3dError, Tidy3dKeyError
from tidy3d.plugins.smatrix import (
    CoaxialLumpedPort,
    LumpedPort,
    PortDataArray,
    TerminalComponentModeler,
    TerminalComponentModelerData,
    TerminalPortDataArray,
    WavePort,
)
from tidy3d.plugins.smatrix.data.data_array import PortNameDataArray
from tidy3d.plugins.smatrix.ports.base_lumped import AbstractLumpedPort
from tidy3d.plugins.smatrix.utils import s_to_z, validate_square_matrix

from ...utils import run_emulated
from .terminal_component_modeler_def import make_coaxial_component_modeler, make_component_modeler

mm = 1e3


def run_component_modeler(
    monkeypatch, modeler: TerminalComponentModeler
) -> TerminalComponentModelerData:
    sim_dict = modeler.sim_dict
    batch_data = {task_name: run_emulated(sim) for task_name, sim in sim_dict.items()}
    port_data = SimulationDataMap(
        keys=tuple(batch_data.keys()),
        values=tuple(batch_data.values()),
    )
    modeler_data = TerminalComponentModelerData(modeler=modeler, data=port_data)
    monkeypatch.setattr(
        td.plugins.smatrix.utils, "port_array_inv", lambda matrix: np.eye(len(modeler.ports))
    )
    monkeypatch.setattr(
        td.plugins.smatrix.utils,
        "compute_F",
        lambda Z_numpy, s_param_def: 1.0 / (2.0 * np.sqrt(np.abs(Z_numpy) + 1e-4)),
    )
    monkeypatch.setattr(
        td.plugins.smatrix.analysis.terminal,
        "check_port_impedance_sign",
        lambda Z_numpy: np.ndarray([]),
    )

    return modeler_data


def get_terminal_port_data_array(
    monkeypatch, modeler: TerminalComponentModeler
) -> TerminalPortDataArray:
    modeler_data = run_component_modeler(monkeypatch=monkeypatch, modeler=modeler)
    return modeler_data.smatrix().data


def check_lumped_port_components_snapped_correctly(modeler: TerminalComponentModeler):
    """Given an instance of a ``TerminalComponentModeler``, check that all simulation components
    have been snapped exactly to the position of the load resistor.
    """
    sim_dict = modeler.sim_dict
    num_ports = len(modeler.ports)
    # Check to make sure all components are exactly aligned along the normal axis
    for src_port, src_idx, src_sim in zip(modeler.ports, range(num_ports), sim_dict.values()):
        assert isinstance(src_port, AbstractLumpedPort)
        monitor_dict = {monitor.name: monitor for monitor in src_sim.monitors}
        normal_axis = src_port.injection_axis
        center_load = src_sim.lumped_elements[src_idx].center[normal_axis]
        assert len(src_sim.sources) == 1
        center_source = src_sim.sources[0].center[normal_axis]
        assert center_load == center_source
        for port, idx in zip(modeler.ports, range(num_ports)):
            assert isinstance(port, AbstractLumpedPort)
            normal_axis = port.injection_axis
            center_load = src_sim.lumped_elements[idx].center[normal_axis]
            center_voltage_monitor = monitor_dict[port._voltage_monitor_name].center[normal_axis]
            center_current_monitor = monitor_dict[port._current_monitor_name].center[normal_axis]
            assert center_load == center_voltage_monitor
            assert center_load == center_current_monitor


def make_t_network_impedance_matrix(
    series_a: complex, series_b: complex, shunt_c: complex
) -> np.ndarray:
    """Create impedance matrix for T-network with series elements A, B and shunt element C.

    Network topology:
        Port1 ----[A]----+----[B]---- Port2
                         |
                        [C]
                         |
                        GND
    """
    z11 = series_a + shunt_c
    z21 = shunt_c
    z12 = shunt_c
    z22 = series_b + shunt_c
    return np.array([[z11, z12], [z21, z22]])


def calc_transmission_line_S_matrix_pseudo(Z0, Zref1, Zref2, gamma, length):
    """
    Calculate complete 2x2 S-parameter matrix for a transmission line
    using pseudo wave definition

    [1] S. Amakawa, "Scattered reflections on scattering parameters—Demystifying complex-referenced
        S parameters—," IEICE Trans. Electron., vol. E99-C, no. 10, pp. 1100-1112, Oct. 2016.

    Parameters:
    -----------
    Z0 : complex or array-like
        Characteristic impedance (can be frequency-dependent)
    Zref1 : complex or array-like
        Reference impedance at port 1 (can be frequency-dependent)
    Zref2 : complex or array-like
        Reference impedance at port 2 (can be frequency-dependent)
    gamma : complex or array-like
        Propagation constant (can be frequency-dependent)
    length : float
        Length (scalar only)

    Returns:
    --------
    np.ndarray :
        S-parameter matrix of shape (nfreq, 2, 2)
    """

    # Calculate hyperbolic functions
    tanh_gamma_ell = np.tanh(gamma * length)
    cosh_gamma_ell = np.cosh(gamma * length)

    # Common denominator for all S-parameters
    denom = (Z0**2 + Zref1 * Zref2) * tanh_gamma_ell + Z0 * (Zref1 + Zref2)

    # Calculate S11
    numerator_S11 = (Z0**2 - Zref1 * Zref2) * tanh_gamma_ell + Z0 * (Zref2 - Zref1)
    S11 = numerator_S11 / denom

    # Calculate S22
    numerator_S22 = (Z0**2 - Zref1 * Zref2) * tanh_gamma_ell + Z0 * (Zref1 - Zref2)
    S22 = numerator_S22 / denom

    # Calculate S21 (transmission from port 1 to port 2)
    numerator_S21 = (
        np.sqrt(np.real(Zref1) / np.real(Zref2)) * (np.abs(Zref2) / np.abs(Zref1)) * 2 * Z0 * Zref1
    )
    S21 = numerator_S21 / (denom * cosh_gamma_ell)

    # Calculate S12 (transmission from port 2 to port 1)
    numerator_S12 = (
        np.sqrt(np.real(Zref2) / np.real(Zref1)) * (np.abs(Zref1) / np.abs(Zref2)) * 2 * Z0 * Zref2
    )
    S12 = numerator_S12 / (denom * cosh_gamma_ell)

    # Construct the S-parameter matrix (nfreq, 2, 2)
    nfreq = len(np.atleast_1d(S11))
    S_matrix = np.zeros((nfreq, 2, 2), dtype=complex)
    S_matrix[:, 0, 0] = S11
    S_matrix[:, 0, 1] = S12
    S_matrix[:, 1, 0] = S21
    S_matrix[:, 1, 1] = S22

    return S_matrix


def calc_transmission_line_S_matrix_power(Z0, Zref1, Zref2, gamma, length):
    """
    Calculate complete 2x2 S-parameter matrix for a transmission line
    using power wave definition

    [1] S. Amakawa, "Scattered reflections on scattering parameters—Demystifying complex-referenced
        S parameters—," IEICE Trans. Electron., vol. E99-C, no. 10, pp. 1100-1112, Oct. 2016.

    Parameters:
    -----------
    Z0 : complex or array-like
        Characteristic impedance (can be frequency-dependent)
    Zref1 : complex or array-like
        Reference impedance at port 1 (can be frequency-dependent)
    Zref2 : complex or array-like
        Reference impedance at port 2 (can be frequency-dependent)
    gamma : complex or array-like
        Propagation constant (can be frequency-dependent)
    length : float
        Length (scalar only)

    Returns:
    --------
    np.ndarray :
        S-parameter matrix of shape (nfreq, 2, 2)
    """

    # Calculate hyperbolic functions
    tanh_gamma_ell = np.tanh(gamma * length)
    cosh_gamma_ell = np.cosh(gamma * length)

    # Complex conjugates
    Zref1_conj = np.conj(Zref1)
    Zref2_conj = np.conj(Zref2)

    # Common denominator
    denom = (Z0**2 + Zref1 * Zref2) * tanh_gamma_ell + Z0 * (Zref1 + Zref2)

    # S11 with conjugate terms
    numerator_S11 = (Z0**2 - Zref1_conj * Zref2) * tanh_gamma_ell + Z0 * (Zref2 - Zref1_conj)
    S11 = numerator_S11 / denom

    # S22 with conjugate terms
    numerator_S22 = (Z0**2 - Zref1 * Zref2_conj) * tanh_gamma_ell + Z0 * (Zref1 - Zref2_conj)
    S22 = numerator_S22 / denom

    # S21 and S12 (transmission parameters)
    numerator_trans = 2 * Z0 * np.sqrt(np.real(Zref1) * np.real(Zref2))
    S21 = numerator_trans / (denom * cosh_gamma_ell)
    S12 = S21  # For reciprocal network

    # Construct the S-parameter matrix (nfreq, 2, 2)
    nfreq = len(np.atleast_1d(S11))
    S_matrix = np.zeros((nfreq, 2, 2), dtype=complex)
    S_matrix[:, 0, 0] = S11
    S_matrix[:, 0, 1] = S12
    S_matrix[:, 1, 0] = S21
    S_matrix[:, 1, 1] = S22

    return S_matrix


def test_validate_no_sources(tmp_path):
    modeler = make_component_modeler(planar_pec=True)
    source = td.PointDipole(
        source_time=td.GaussianPulse(freq0=2e14, fwidth=1e14), polarization="Ex"
    )
    sim_w_source = modeler.simulation.copy(update={"sources": (source,)})
    with pytest.raises(pd.ValidationError):
        _ = modeler.copy(update={"simulation": sim_w_source})


def test_validate_freqs():
    """Ensure the 'freqs' array does not contain negative values nor duplicate entries."""
    modeler = make_component_modeler(planar_pec=False)
    # It should be possible to provide a single frequency point
    freqs = np.array([1.0e6])
    modeler = modeler.updated_copy(freqs=freqs)
    _ = modeler._source_time
    # Negative frequencies are not allowed
    freqs = np.array([-1.0, 5]) * 1e9
    with pytest.raises(pd.ValidationError):
        _ = modeler.updated_copy(freqs=freqs)
    # Test case with non-unique value
    f_min, f_max = (0.5e9, 1.5e9)
    f0 = (f_min + f_max) / 2
    f_target = 1.35e9
    freqs = np.sort(np.append(np.linspace(f_min, f_max, 21), f_target))
    with pytest.raises(pd.ValidationError):
        _ = modeler.updated_copy(freqs=freqs)


def test_validate_3D_sim(tmp_path):
    modeler = make_component_modeler(planar_pec=False)
    sim = td.Simulation(
        size=(10e3, 10e3, 0),
        sources=[],
        monitors=[],
        grid_spec=td.GridSpec.uniform(dl=1e3),
        boundary_spec=td.BoundarySpec(
            x=td.Boundary.pml(),
            y=td.Boundary.pml(),
            z=td.Boundary.periodic(),
        ),
        run_time=1e-10,
    )
    with pytest.raises(pd.ValidationError):
        _ = modeler.updated_copy(simulation=sim)


def test_no_port(tmp_path):
    modeler = make_component_modeler(planar_pec=True)
    _ = modeler.ports
    with pytest.raises(Tidy3dKeyError):
        modeler.get_port_by_name(port_name="NOT_A_PORT")


def test_plot_sim(tmp_path):
    modeler = make_component_modeler(planar_pec=False)
    modeler.plot_sim(z=0)
    plt.close()


def test_plot_sim_eps(tmp_path):
    modeler = make_component_modeler(planar_pec=False)
    modeler.plot_sim_eps(z=0)
    plt.close()


@pytest.mark.parametrize("port_refinement", [False, True])
def test_make_component_modeler(tmp_path, port_refinement):
    modeler = make_component_modeler(planar_pec=False, port_refinement=port_refinement)
    if port_refinement:
        for sim in modeler.sim_dict.values():
            _ = sim.volumetric_structures


def test_run(monkeypatch, tmp_path):
    modeler = make_component_modeler(planar_pec=True)
    modeler_data = run_component_modeler(monkeypatch, modeler)


def test_run_component_modeler(monkeypatch, tmp_path):
    modeler = make_component_modeler(planar_pec=True)
    s_matrix = get_terminal_port_data_array(monkeypatch, modeler)

    for port_in in modeler.ports:
        for port_out in modeler.ports:
            coords_in = {"port_in": port_in.name}
            coords_out = {"port_out": port_out.name}

            assert np.all(s_matrix.sel(**coords_in) != 0), "source index not present in S matrix"
            assert np.all(s_matrix.sel(**coords_in).sel(**coords_out) != 0), (
                "monitor index not present in S matrix"
            )


def test_s_to_z_component_modeler():
    """Test conversion of S parameters to impedance matrix,
    for a simple test case of 2 port T network with reference impedance of 50 Ohm

    Network topology:
          Port1 ----[A]----+----[B]---- Port2
                           |
                          [C]
                           |
                          GND
    """
    A = 20 + 30j
    B = 50 - 15j
    C = 60

    Z = make_t_network_impedance_matrix(A, B, C)
    Z11 = Z[0, 0]
    Z21 = Z[1, 0]
    Z12 = Z[0, 1]
    Z22 = Z[1, 1]

    Z0 = 50.0
    # Manual creation of S parameters Pozar Table 4.2
    deltaZ = (Z11 + Z0) * (Z22 + Z0) - Z12 * Z21
    S11 = ((Z11 - Z0) * (Z22 + Z0) - Z12 * Z21) / deltaZ
    S12 = (2 * Z12 * Z0) / deltaZ
    S21 = (2 * Z21 * Z0) / deltaZ
    S22 = ((Z11 + Z0) * (Z22 - Z0) - Z12 * Z21) / deltaZ

    port_names = ["lumped_port_1", "lumped_port_2"]
    freqs = [1e8, 2e8, 3e8]

    values = np.array(
        3 * [[[S11, S12], [S21, S22]]],
        dtype=complex,
    )
    # Put coords in opposite order to check reordering
    coords = {
        "f": np.array(freqs),
        "port_out": port_names,
        "port_in": port_names,
    }

    s_matrix = TerminalPortDataArray(data=values, coords=coords)
    z_matrix = s_to_z(s_matrix, reference=Z0)
    z_matrix_at_f = z_matrix.sel(f=1e8)
    assert np.isclose(z_matrix_at_f[0, 0], Z11)
    assert np.isclose(z_matrix_at_f[0, 1], Z12)
    assert np.isclose(z_matrix_at_f[1, 0], Z21)
    assert np.isclose(z_matrix_at_f[1, 1], Z22)

    # test version with different port reference impedances
    values = np.full((len(freqs), len(port_names)), Z0)
    coords = {
        "f": np.array(freqs),
        "port": port_names,
    }
    z_port_matrix = PortDataArray(data=values, coords=coords)
    z_matrix = s_to_z(s_matrix, reference=z_port_matrix)
    z_matrix_at_f = z_matrix.sel(f=1e8)
    assert np.isclose(z_matrix_at_f[0, 0], Z11)
    assert np.isclose(z_matrix_at_f[0, 1], Z12)
    assert np.isclose(z_matrix_at_f[1, 0], Z21)
    assert np.isclose(z_matrix_at_f[1, 1], Z22)


def test_complex_reference_s_to_z_component_modeler():
    """Test conversion of S parameters to impedance matrix,
    for a test case of 2 port T network with complex reference impedances, which requires
    identifying the precise definition used for S parameters

    Network topology:
          Port1 ----[A]----+----[B]---- Port2
                           |
                          [C]
                           |
                          GND
    """
    A = np.array([0, 21 + 31j, 60])
    B = np.array([0, 51 - 16j, 40])
    C = np.array([50, 61, 0])

    freqs = np.array([1e9, 2e9, 3e9])
    # Build Z for each frequency with different A, B, C
    Z = np.stack([make_t_network_impedance_matrix(a, b, c) for a, b, c in zip(A, B, C)], axis=0)
    # Choose some port reference impedances
    z0 = np.stack(3 * [np.array([50 + 5j, 60 - 2j])], axis=0)

    skrf_S_50ohm = skrf.Network.from_z(z=Z, f=freqs)
    skrf_S_power = skrf.Network.from_z(z=Z, f=freqs, s_def="power", z0=z0)
    skrf_S_pseudo = skrf.Network.from_z(z=Z, f=freqs, s_def="pseudo", z0=z0)

    ports = ["port1", "port2"]
    smatrix = TerminalPortDataArray(
        skrf_S_50ohm.s, coords={"f": freqs, "port_out": ports, "port_in": ports}
    )
    # Test real reference impedance calculations
    z_tidy3d = s_to_z(smatrix, reference=50, s_param_def="power")
    assert np.all(np.isclose(z_tidy3d.values, Z))
    z_tidy3d = s_to_z(smatrix, reference=50, s_param_def="pseudo")
    assert np.all(np.isclose(z_tidy3d.values, Z))

    # Test complex reference impedance calculations
    z0_tidy3d = PortDataArray(data=z0, coords={"f": freqs, "port": ports})
    smatrix.values = skrf_S_power.s
    z_tidy3d = s_to_z(smatrix, reference=z0_tidy3d, s_param_def="power")
    assert np.all(np.isclose(z_tidy3d.values, Z))

    smatrix.values = skrf_S_pseudo.s
    z_tidy3d = s_to_z(smatrix, reference=z0_tidy3d, s_param_def="pseudo")
    assert np.all(np.isclose(z_tidy3d.values, Z))


def test_data_s_to_z(monkeypatch):
    """Test 's_to_z' method of 'TerminalComponentModelerData'."""
    modeler = make_component_modeler(planar_pec=True)
    modeler_data = run_component_modeler(monkeypatch, modeler)

    port_names = [p.name for p in modeler.ports]
    freqs = modeler.freqs

    s11 = 0.1
    s12 = 0.8
    s21 = 0.8
    s22 = 0.1

    values = np.array(
        len(freqs) * [[[s11, s12], [s21, s22]]],
        dtype=complex,
    )
    coords = {
        "f": freqs,
        "port_out": port_names,
        "port_in": port_names,
    }
    s_matrix_data = TerminalPortDataArray(data=values, coords=coords)

    from tidy3d.plugins.smatrix.data.terminal import MicrowaveSMatrixData

    s_matrix_container = MicrowaveSMatrixData(data=s_matrix_data)

    monkeypatch.setattr(
        TerminalComponentModelerData, "smatrix", lambda self, **kwargs: s_matrix_container
    )

    z0 = 50.0
    z_matrix = modeler_data.s_to_z(reference=z0)

    delta_s = (1 - s11) * (1 - s22) - s12 * s21
    z11 = z0 * ((1 + s11) * (1 - s22) + s12 * s21) / delta_s
    z12 = z0 * (2 * s12) / delta_s
    z21 = z0 * (2 * s21) / delta_s
    z22 = z0 * ((1 - s11) * (1 + s22) + s12 * s21) / delta_s

    assert np.allclose(z_matrix.sel(port_in=port_names[0], port_out=port_names[0]).values, z11)
    assert np.allclose(z_matrix.sel(port_in=port_names[1], port_out=port_names[0]).values, z12)
    assert np.allclose(z_matrix.sel(port_in=port_names[0], port_out=port_names[1]).values, z21)
    assert np.allclose(z_matrix.sel(port_in=port_names[1], port_out=port_names[1]).values, z22)


def test_ab_to_s_component_modeler():
    coords = {
        "f": np.array([1e8]),
        "port_out": ["lumped_port_1", "lumped_port_2"],
        "port_in": ["lumped_port_1", "lumped_port_2"],
    }
    # Common case is reference impedance matched to loads, which means ideally
    # the a matrix would be an identity matrix, and as a result the s matrix will be
    # given directly by the b_matrix
    a_values = np.eye(2, 2)
    a_values = np.reshape(a_values, (1, 2, 2))
    b_values = (1 + 1j) * np.random.random((1, 2, 2))
    a_matrix = TerminalPortDataArray(data=a_values, coords=coords)
    b_matrix = TerminalPortDataArray(data=b_values, coords=coords)
    S_matrix = TerminalComponentModelerData.ab_to_s(a_matrix, b_matrix)
    assert np.isclose(S_matrix, b_matrix).all()


def test_port_snapping(tmp_path):
    """Make sure that the snapping behavior of the load resistor is mirrored
    by all other components in the modeler simulations with rectangular ports.
    """
    y_z_grid = td.UniformGrid(dl=0.1 * 1e3)
    x_grid = td.UniformGrid(dl=11 * 1e3)
    grid_spec = td.GridSpec(grid_x=x_grid, grid_y=y_z_grid, grid_z=y_z_grid)
    modeler = make_component_modeler(planar_pec=True, port_refinement=False, grid_spec=grid_spec)
    check_lumped_port_components_snapped_correctly(modeler=modeler)


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_coaxial_port_source_size(axis):
    """Make sure source size is correct."""
    port = CoaxialLumpedPort(
        center=(0, 0, 0),
        inner_diameter=1,
        outer_diameter=2,
        normal_axis=axis,
        direction="+",
        name="port",
        impedance=50,
    )
    source = port.to_source(td.GaussianPulse(freq0=1e10, fwidth=1e9))
    assert np.isclose(source.size[axis], 0)


def test_coarse_grid_at_port(monkeypatch, tmp_path):
    modeler = make_component_modeler(planar_pec=True, port_refinement=False, port_snapping=False)
    # Without port refinement the grid is much too coarse for these port sizes
    with pytest.raises(SetupError):
        _ = run_component_modeler(monkeypatch, modeler)


def test_validate_port_voltage_axis():
    with pytest.raises(pd.ValidationError):
        LumpedPort(center=(0, 0, 0), size=(0, 1, 2), voltage_axis=0, impedance=50)


@pytest.mark.parametrize("snap_center", [None, 0.1])
def test_converting_port_to_simulation_objects(snap_center):
    """Test that the LumpedPort can be converted into monitors and source without the grid present."""
    port = LumpedPort(center=(0, 0, 0), size=(0, 1, 2), voltage_axis=2, impedance=50, name="Port1")
    freqs = np.linspace(1e9, 10e9, 11)
    source_time = td.GaussianPulse(freq0=5e9, fwidth=9e9)
    _ = port.to_monitors(freqs=freqs, snap_center=snap_center)
    _ = port.to_source(source_time=source_time, snap_center=snap_center)


@pytest.mark.parametrize("port_refinement", [False, True])
def test_make_coaxial_component_modeler(tmp_path, port_refinement):
    modeler = make_coaxial_component_modeler(port_refinement=port_refinement)
    if port_refinement:
        for sim in modeler.sim_dict.values():
            _ = sim.volumetric_structures


def test_run_coaxial_component_modeler(monkeypatch, tmp_path):
    modeler = make_coaxial_component_modeler()
    s_matrix = get_terminal_port_data_array(monkeypatch, modeler)

    for port_in in modeler.ports:
        for port_out in modeler.ports:
            coords_in = {"port_in": port_in.name}
            coords_out = {"port_out": port_out.name}

            assert np.all(s_matrix.sel(**coords_in) != 0), "source index not present in S matrix"
            assert np.all(s_matrix.sel(**coords_in).sel(**coords_out) != 0), (
                "monitor index not present in S matrix"
            )


@pytest.mark.parametrize(
    "grid_spec",
    [
        None,
        td.GridSpec(
            grid_x=td.UniformGrid(dl=0.1 * mm),
            grid_y=td.UniformGrid(dl=10 * mm),
            grid_z=td.UniformGrid(dl=0.1 * mm),
        ),
        td.GridSpec(
            grid_x=td.UniformGrid(dl=10 * mm),
            grid_y=td.UniformGrid(dl=0.1 * mm),
            grid_z=td.UniformGrid(dl=0.1 * mm),
        ),
    ],
)
def test_coarse_grid_at_coaxial_port(monkeypatch, tmp_path, grid_spec):
    """Ensure that the grid is fine enough at the coaxial ports along the transverse dimensions."""
    modeler = make_coaxial_component_modeler(port_refinement=False, grid_spec=grid_spec)
    # Without port refinement the grid is much too coarse for these port sizes
    with pytest.raises(SetupError):
        _ = run_component_modeler(monkeypatch, modeler)


def test_validate_coaxial_center_not_inf():
    with pytest.raises(pd.ValidationError):
        CoaxialLumpedPort(
            center=(td.inf, 0, 0),
            outer_diameter=8,
            inner_diameter=1,
            normal_axis=2,
            direction="+",
            name="coax_port_1",
            num_grid_cells=None,
            impedance=50,
        )


def test_validate_coaxial_port_diameters():
    with pytest.raises(pd.ValidationError):
        CoaxialLumpedPort(
            center=(0, 0, 0),
            outer_diameter=1,
            inner_diameter=2,
            normal_axis=2,
            direction="+",
            name="coax_port_1",
            num_grid_cells=None,
            impedance=50,
        )


@pytest.mark.parametrize("direction", ["+", "-"])
def test_current_integral_positioning_coaxial_port(direction):
    """Make sure the positioning of the current integral used by the CoaxialLumpedPort is correct,
    when the coordinates and port position do not exactly match. This requires that the port is
    snapped correctly to cell boundaries.
    """
    # Test coordinates from a failing case
    normal_coords = np.array(
        [
            -14069.999999999978,
            -14049.999999999978,
            -14029.999999999978,
            -14009.999999999978,
        ]
    )
    # The port center should be snapped to cell boundaries which is the midpoint of
    # adjacent transverse magnetic field locations
    normal_port_position = (normal_coords[2] + normal_coords[3]) / 2
    path_pos = CoaxialLumpedPort._determine_current_integral_pos(
        normal_port_position, normal_coords, direction
    )

    if direction == "+":
        assert path_pos == normal_coords[3]
    else:
        assert path_pos == normal_coords[2]


def test_coaxial_port_snapping(tmp_path):
    """Make sure that the snapping behavior of the load resistor is mirrored
    by all other components in the modeler simulations with coaxial ports.
    """
    x_y_grid = td.UniformGrid(dl=0.1 * 1e3)
    z_grid = td.UniformGrid(dl=11 * 1e3)
    grid_spec = td.GridSpec(grid_x=x_y_grid, grid_y=x_y_grid, grid_z=z_grid)
    modeler = make_coaxial_component_modeler(port_refinement=False, grid_spec=grid_spec)
    check_lumped_port_components_snapped_correctly(modeler=modeler)


def test_power_delivered_helper(monkeypatch, tmp_path):
    """Test computations involving power waves are correct by manually setting voltage and current
    at ports using monkeypatch.
    """
    modeler = make_coaxial_component_modeler()
    port1 = modeler.ports[0]
    port_impedance = port1.impedance
    freqs = np.linspace(1e9, 10e9, 11)
    # Emulate perfect power transmission
    voltage_amplitude = 1.0
    current_amplitude = voltage_amplitude / port_impedance
    # Average power assuming no reflections
    avg_power = 0.5 * voltage_amplitude * np.conj(current_amplitude)

    voltage = np.ones_like(freqs) * voltage_amplitude
    current = np.ones_like(freqs) * current_amplitude

    def compute_voltage_patch(self, sim_data):
        return FreqDataArray(voltage, coords={"f": freqs})

    def compute_current_patch(self, sim_data):
        return FreqDataArray(current, coords={"f": freqs})

    monkeypatch.setattr(CoaxialLumpedPort, "compute_voltage", compute_voltage_patch)
    monkeypatch.setattr(CoaxialLumpedPort, "compute_current", compute_current_patch)

    # First test should give complete power transfer into the network
    power = TerminalComponentModelerData.compute_power_delivered_by_port(sim_data=None, port=port1)
    assert np.allclose(power.values, avg_power)

    # Second test is complete reflecton
    current = np.ones_like(freqs) * 0
    power = TerminalComponentModelerData.compute_power_delivered_by_port(sim_data=None, port=port1)
    assert np.allclose(power.values, 0)

    # Third test is a custom test using equation 4.60 and 4.61 from
    # Microwave engineering/David M. Pozar.—4th ed.
    power_a = 2.0
    power_b = 1.0
    Zr = port_impedance
    Rr = np.sqrt(np.real(port_impedance))
    voltage_amplitude = (np.conj(Zr) * power_a + Zr * power_b) / Rr
    current_amplitude = (power_a - power_b) / Rr
    voltage = np.ones_like(freqs) * voltage_amplitude
    current = np.ones_like(freqs) * current_amplitude
    power = TerminalComponentModelerData.compute_power_delivered_by_port(sim_data=None, port=port1)
    assert np.allclose(power.values, 0.5 * (power_a**2 - power_b**2))


def test_make_coaxial_component_modeler_with_wave_ports(tmp_path):
    """Checks that the terminal component modeler is created successfully with wave ports."""
    z_grid = td.UniformGrid(dl=1 * 1e3)
    xy_grid = td.UniformGrid(dl=0.1 * 1e3)
    grid_spec = td.GridSpec(grid_x=xy_grid, grid_y=xy_grid, grid_z=z_grid)
    _ = make_coaxial_component_modeler(
        port_types=(WavePort, WavePort),
        grid_spec=grid_spec,
    )


@pytest.mark.parametrize("voltage_enabled", [False, True])
@pytest.mark.parametrize("current_enabled", [False, True])
def test_run_coaxial_component_modeler_with_wave_ports(
    monkeypatch, tmp_path, voltage_enabled, current_enabled
):
    """Checks that the terminal component modeler runs with wave ports."""
    z_grid = td.UniformGrid(dl=1 * 1e3)
    xy_grid = td.UniformGrid(dl=0.1 * 1e3)
    grid_spec = td.GridSpec(grid_x=xy_grid, grid_y=xy_grid, grid_z=z_grid)
    if not (voltage_enabled or current_enabled):
        with pytest.raises(pd.ValidationError):
            modeler = make_coaxial_component_modeler(
                port_types=(WavePort, WavePort),
                grid_spec=grid_spec,
                use_voltage=voltage_enabled,
                use_current=current_enabled,
            )
        return

    modeler = make_coaxial_component_modeler(
        port_types=(WavePort, WavePort),
        grid_spec=grid_spec,
        use_voltage=voltage_enabled,
        use_current=current_enabled,
    )
    s_matrix = get_terminal_port_data_array(monkeypatch, modeler)

    shape_one_port = (len(modeler.freqs), len(modeler.ports))
    shape_both_ports = (len(modeler.freqs),)
    for port_in in modeler.ports:
        for port_out in modeler.ports:
            coords_in = {"port_in": port_in.name}
            coords_out = {"port_out": port_out.name}

            assert np.all(s_matrix.sel(**coords_in).values.shape == shape_one_port), (
                "source index not present in S matrix"
            )
            assert np.all(
                s_matrix.sel(**coords_in).sel(**coords_out).values.shape == shape_both_ports
            ), "monitor index not present in S matrix"

    # Another run with more modes in the mode spec
    mode_spec = td.ModeSpec(num_modes=2)
    modeler = modeler.updated_copy(path="ports/0/", mode_spec=mode_spec)
    s_matrix = get_terminal_port_data_array(monkeypatch, modeler)

    shape_one_port = (len(modeler.freqs), len(modeler.ports))
    shape_both_ports = (len(modeler.freqs),)
    for port_in in modeler.ports:
        for port_out in modeler.ports:
            coords_in = {"port_in": port_in.name}
            coords_out = {"port_out": port_out.name}

            assert np.all(s_matrix.sel(**coords_in).values.shape == shape_one_port), (
                "source index not present in S matrix"
            )
            assert np.all(
                s_matrix.sel(**coords_in).sel(**coords_out).values.shape == shape_both_ports
            ), "monitor index not present in S matrix"


def test_run_mixed_component_modeler_with_wave_ports(monkeypatch, tmp_path):
    """Checks the terminal component modeler will allow mixed ports."""
    z_grid = td.UniformGrid(dl=1 * 1e3)
    xy_grid = td.UniformGrid(dl=0.1 * 1e3)
    grid_spec = td.GridSpec(grid_x=xy_grid, grid_y=xy_grid, grid_z=z_grid)
    modeler = make_coaxial_component_modeler(
        port_types=(CoaxialLumpedPort, WavePort), grid_spec=grid_spec
    )
    s_matrix = get_terminal_port_data_array(monkeypatch, modeler)

    shape_one_port = (len(modeler.freqs), len(modeler.ports))
    shape_both_ports = (len(modeler.freqs),)
    for port_in in modeler.ports:
        for port_out in modeler.ports:
            coords_in = {"port_in": port_in.name}
            coords_out = {"port_out": port_out.name}

            assert np.all(s_matrix.sel(**coords_in).values.shape == shape_one_port), (
                "source index not present in S matrix"
            )
            assert np.all(
                s_matrix.sel(**coords_in).sel(**coords_out).values.shape == shape_both_ports
            ), "monitor index not present in S matrix"


def test_wave_port_path_integral_validation():
    """Checks that wave port will ensure path integrals are within the bounds of the port."""
    size_port = [2, 2, 0]
    center_port = [0, 0, -10]

    voltage_path = td.AxisAlignedVoltageIntegral(
        center=(0.5, 0, -10),
        size=(1.0, 0, 0),
        extrapolate_to_endpoints=True,
        snap_path_to_grid=True,
        sign="+",
    )

    custom_current_path = td.Custom2DCurrentIntegral.from_circular_path(
        center=center_port, radius=0.5, num_points=21, normal_axis=2, clockwise=False
    )

    mode_spec = td.ModeSpec(num_modes=1, target_neff=1.8)

    _ = WavePort(
        center=center_port,
        size=size_port,
        name="wave_port_1",
        mode_spec=mode_spec,
        direction="+",
        voltage_integral=voltage_path,
        current_integral=None,
    )

    _ = WavePort(
        center=center_port,
        size=size_port,
        name="wave_port_1",
        mode_spec=mode_spec,
        direction="+",
        voltage_integral=None,
        current_integral=custom_current_path,
    )

    with pytest.raises(pd.ValidationError):
        _ = WavePort(
            center=center_port,
            size=size_port,
            name="wave_port_1",
            mode_spec=mode_spec,
            direction="+",
            voltage_integral=None,
            current_integral=None,
        )

    voltage_path = voltage_path.updated_copy(size=(4, 0, 0))
    with pytest.raises(pd.ValidationError):
        _ = WavePort(
            center=center_port,
            size=size_port,
            name="wave_port_1",
            mode_spec=mode_spec,
            direction="+",
            voltage_integral=voltage_path,
            current_integral=None,
        )

    custom_current_path = td.Custom2DCurrentIntegral.from_circular_path(
        center=center_port, radius=3, num_points=21, normal_axis=2, clockwise=False
    )
    with pytest.raises(pd.ValidationError):
        _ = WavePort(
            center=center_port,
            size=size_port,
            name="wave_port_1",
            mode_spec=mode_spec,
            direction="+",
            voltage_integral=None,
            current_integral=custom_current_path,
        )

    # Test integral path only slightly larger than port bounds
    wave_port = WavePort(
        center=(0, 10000, 115.00000022351743),
        size=(500, 0, 160.00000000000003),
        name="wave_port_1",
        mode_spec=mode_spec,
        direction="+",
        voltage_integral=voltage_path.updated_copy(
            size=(0, 0, 70.000000298023424), center=(0, 10000, 70.000000298023424)
        ),
        current_integral=None,
    )
    # Make sure validation would have failed if a strict comparison was used
    assert wave_port.bounds[0][2] > wave_port.voltage_integral.bounds[0][2]


def test_wave_port_grid_validation(tmp_path):
    """Ensure that 'num_grid_cells' is validated and works to ensure that the grid is refined around wave ports."""
    size_port = [2, 2, 0]
    center_port = [0, 0, -10]

    voltage_path = td.AxisAlignedVoltageIntegral(
        center=(0.5, 0, -10),
        size=(1.0, 0, 0),
        extrapolate_to_endpoints=True,
        snap_path_to_grid=True,
        sign="+",
    )

    current_path = td.AxisAlignedCurrentIntegral(
        center=(0.5, 0, -10),
        size=(0.25, 0.5, 0),
        snap_contour_to_grid=True,
        sign="+",
    )

    mode_spec = td.ModeSpec(num_modes=1, target_neff=1.8)

    _ = WavePort(
        center=center_port,
        size=size_port,
        name="wave_port_1",
        mode_spec=mode_spec,
        direction="+",
        voltage_integral=voltage_path,
        current_integral=current_path,
        num_grid_cells=None,
    )

    with pytest.raises(pd.ValidationError):
        _ = WavePort(
            center=center_port,
            size=size_port,
            name="wave_port_1",
            mode_spec=mode_spec,
            direction="+",
            voltage_integral=voltage_path,
            current_integral=current_path,
            num_grid_cells=2,
        )

    modeler = make_coaxial_component_modeler(
        grid_spec=td.GridSpec.auto(wavelength=10e3),
        port_refinement=True,
        port_types=(WavePort, WavePort),
    )
    _ = modeler.sim_dict

    modeler = make_coaxial_component_modeler(
        grid_spec=td.GridSpec.auto(wavelength=10e3),
        port_refinement=False,
        port_types=(WavePort, WavePort),
    )
    with pytest.raises(SetupError):
        _ = modeler.sim_dict


def test_wave_port_to_mode_solver(tmp_path):
    """Checks that wave port can be converted to a mode solver."""
    modeler = make_coaxial_component_modeler(port_types=(WavePort, WavePort))
    _ = modeler.ports[0].to_mode_solver(modeler.simulation, freqs=[1e9, 2e9, 3e9])


def test_port_source_snapped_to_PML(tmp_path):
    """Raise meaningful error message when source is snapped into PML because the port is too close
    to the boundary.
    """
    modeler = make_component_modeler(planar_pec=True)
    port_pos = 5e4
    voltage_path = td.AxisAlignedVoltageIntegral(
        center=(port_pos, 0, 0),
        size=(0, 1e3, 0),
        sign="+",
    )
    port = WavePort(
        center=(port_pos, 0, 0),
        size=(0, 1e3, 1e3),
        name="wave_port",
        mode_spec=td.ModeSpec(num_modes=1),
        direction="-",
        voltage_integral=voltage_path,
        current_integral=None,
    )
    modeler = modeler.updated_copy(ports=[port])

    # Error because port is snapped to PML layers; but the error message might not
    # be very informative, e.g. "simulation.sources[0]' is outside of the simulation domain".
    # So we also check where error should be raised immediately
    with pytest.raises(SetupError):
        modeler.sim_dict

    with pytest.raises(SetupError):
        modeler._shift_value_signed(port)

    # also validate the negative side
    voltage_path = voltage_path.updated_copy(center=(-port_pos, 0, 0))
    port = port.updated_copy(direction="+", center=(-port_pos, 0, 0), voltage_integral=voltage_path)
    modeler = modeler.updated_copy(ports=[port])
    with pytest.raises(SetupError):
        modeler.sim_dict

    with pytest.raises(SetupError):
        modeler._shift_value_signed(port)


def test_wave_port_validate_current_integral(tmp_path):
    """Checks that the current integral direction validator runs correctly."""
    modeler = make_coaxial_component_modeler(port_types=(WavePort, WavePort))
    with pytest.raises(pd.ValidationError):
        _ = modeler.updated_copy(direction="-", path="ports/0/")


def test_port_impedance_check():
    """Tests the impedance consistency check."""
    Z_numpy = np.ones((50, 3))
    Z_numpy[:, 1] = -1.0
    # All ok if same sign for every frequency
    TerminalComponentModelerData.check_port_impedance_sign(Z_numpy)
    Z_numpy[25, 1] = 1.0
    # Change of sign is unexpected
    with pytest.raises(Tidy3dError):
        TerminalComponentModelerData.check_port_impedance_sign(Z_numpy)


def test_antenna_helpers(monkeypatch, tmp_path):
    """Test monitor data normalization and combination helpers for antenna parameters."""
    # Setup basic modeler with radiation monitor
    modeler = make_component_modeler(False)
    sim = modeler.simulation
    theta = np.linspace(0, np.pi, 40)
    phi = np.linspace(0, 2 * np.pi, 80)
    radiation_monitor = td.DirectivityMonitor(
        size=sim.size,
        center=sim.center,
        freqs=modeler.freqs,
        name="antenna_monitor",
        far_field_approx=True,
        proj_distance=max(sim.size) * 100,
        theta=theta,
        phi=phi,
    )
    modeler = modeler.updated_copy(radiation_monitors=[radiation_monitor])

    # Run simulation to get data
    modeler_data = run_component_modeler(monkeypatch, modeler)
    sim_data = modeler_data.data[modeler_data.modeler.get_task_name(modeler.ports[0])]
    rad_mon_data = sim_data[radiation_monitor.name]

    # Test monitor helper
    found_mon = modeler.get_radiation_monitor_by_name(radiation_monitor.name)
    assert found_mon == radiation_monitor
    with pytest.raises(Tidy3dKeyError):
        modeler.get_radiation_monitor_by_name("invalid")

    # Test monitor data normalization with different amplitude types
    a_array = FreqDataArray(np.ones(len(modeler.freqs)), {"f": modeler.freqs})
    a_array_raw = 2.0 * a_array
    normalized_data_array = modeler_data._monitor_data_at_port_amplitude(
        modeler.ports[0], radiation_monitor.name, a_array, a_array_raw
    )
    normalized_data_const = modeler_data._monitor_data_at_port_amplitude(
        modeler.ports[0], radiation_monitor.name, 1.0, a_array_raw
    )
    assert isinstance(normalized_data_array, td.DirectivityData)
    assert isinstance(normalized_data_const, td.DirectivityData)

    # Test combining monitor data
    combined_data = normalized_data_array + normalized_data_const
    assert isinstance(combined_data, td.DirectivityData)

    # Test power wave amplitude computation
    a, b = modeler_data.compute_power_wave_amplitudes_at_each_port(sim_data=sim_data)
    assert isinstance(a, PortDataArray)
    assert isinstance(b, PortDataArray)


@pytest.mark.parametrize("port_type", ["lumped", "wave"])
def test_antenna_parameters(monkeypatch, port_type):
    """Test basic antenna parameters computation and validation."""
    # Setup modeler with radiation monitor
    if port_type == "lumped":
        modeler: TerminalComponentModeler = make_component_modeler(False)
    else:
        modeler: TerminalComponentModeler = make_coaxial_component_modeler(
            port_types=(WavePort, WavePort)
        )
    sim = modeler.simulation
    theta = np.linspace(0, np.pi, 101)
    phi = np.linspace(0, 2 * np.pi, 201)
    proj_distance = max(sim.size) * 100
    # First test validation of the radiation monitors
    # The frequencies should be a subset of the freqs set in the TerminalComponentModeler
    freqs = [3.14e13]
    radiation_monitor = td.DirectivityMonitor(
        size=sim.size,
        center=sim.center,
        freqs=freqs,
        name="antenna_monitor",
        far_field_approx=True,
        proj_distance=proj_distance,
        theta=theta,
        phi=phi,
    )
    with pytest.raises(pd.ValidationError):
        modeler = modeler.updated_copy(radiation_monitors=[radiation_monitor])

    radiation_monitor = radiation_monitor.updated_copy(freqs=modeler.freqs)
    modeler = modeler.updated_copy(radiation_monitors=[radiation_monitor])

    # Run simulation and get antenna parameters
    modeler_data = run_component_modeler(monkeypatch, modeler)

    # Make sure network index works for single mode / multimode cases
    port_1_network_index = modeler.network_index(modeler.ports[0])
    port_2_network_index = modeler.network_index(modeler.ports[1], 0)
    _ = modeler_data.get_antenna_metrics_data({port_1_network_index: 1.0})
    _ = modeler_data.get_antenna_metrics_data({port_2_network_index: None})
    antenna_params = modeler_data.get_antenna_metrics_data()

    # Test that all essential parameters exist and are correct type
    assert isinstance(antenna_params.radiation_efficiency, FreqDataArray)
    assert isinstance(antenna_params.reflection_efficiency, FreqDataArray)
    assert isinstance(antenna_params.gain, xr.DataArray)
    assert isinstance(antenna_params.realized_gain, xr.DataArray)

    # Test partial gain computations in linear basis
    partial_gain_linear = antenna_params.partial_gain(pol_basis="linear")
    assert isinstance(partial_gain_linear, xr.Dataset)
    assert "Gtheta" in partial_gain_linear
    assert "Gphi" in partial_gain_linear

    # Test partial gain computations in circular basis
    partial_gain_circular = antenna_params.partial_gain(pol_basis="circular")
    assert isinstance(partial_gain_circular, xr.Dataset)
    assert "Gright" in partial_gain_circular
    assert "Gleft" in partial_gain_circular

    # Test partial realized gain computations in both bases
    assert isinstance(antenna_params.partial_realized_gain("linear"), xr.Dataset)
    assert isinstance(antenna_params.partial_realized_gain("circular"), xr.Dataset)

    # Test validation of pol_basis parameter
    with pytest.raises(ValueError):
        antenna_params.partial_gain("invalid")
    with pytest.raises(ValueError):
        antenna_params.partial_realized_gain("invalid")


def test_get_combined_antenna_parameters_data(monkeypatch, tmp_path):
    """Test the computation of combined antenna parameters from multiple ports."""
    modeler = make_component_modeler(False)
    sim = modeler.simulation
    theta = np.linspace(0, np.pi, 101)
    phi = np.linspace(0, 2 * np.pi, 201)
    proj_distance = max(sim.size) * 100
    radiation_monitor = td.DirectivityMonitor(
        size=sim.size,
        center=sim.center,
        freqs=modeler.freqs,
        name="antenna_monitor",
        far_field_approx=True,
        proj_distance=proj_distance,
        theta=theta,
        phi=phi,
    )
    modeler = modeler.updated_copy(radiation_monitors=[radiation_monitor])
    modeler_data = run_component_modeler(monkeypatch=monkeypatch, modeler=modeler)

    # Define port amplitudes
    port_amplitudes = {modeler.ports[0].name: 1.0, modeler.ports[1].name: 1j}

    # Get combined antenna parameters
    antenna_params = modeler_data.get_antenna_metrics_data(
        port_amplitudes, monitor_name="antenna_monitor"
    )

    # Check that essential properties exist and are correct type
    assert isinstance(antenna_params.radiation_efficiency, FreqDataArray)
    assert isinstance(antenna_params.reflection_efficiency, FreqDataArray)
    assert isinstance(antenna_params.partial_gain(), xr.Dataset)
    assert isinstance(antenna_params.gain, xr.DataArray)
    assert isinstance(antenna_params.partial_realized_gain(), xr.Dataset)
    assert isinstance(antenna_params.realized_gain, xr.DataArray)

    # Test with single port for comparison
    single_port_params = modeler_data.get_antenna_metrics_data()
    # # Create mock batch data
    # Values should be different when combining ports vs single port
    assert not np.allclose(antenna_params.gain, single_port_params.gain)
    assert not np.allclose(
        antenna_params.radiation_efficiency, single_port_params.radiation_efficiency
    )

    # Define port amplitudes
    port_amplitudes = {modeler.ports[0].name: 1.0, modeler.ports[1].name: 0.0}
    port2_zero_params = modeler_data.get_antenna_metrics_data(port_amplitudes)
    # Should give idential results to only exciting one port
    assert np.allclose(port2_zero_params.gain, single_port_params.gain)
    assert np.allclose(
        port2_zero_params.radiation_efficiency, single_port_params.radiation_efficiency
    )


def test_run_only_and_element_mappings(monkeypatch, tmp_path):
    """Checks the terminal component modeler works when running with a subset of excitations."""
    z_grid = td.UniformGrid(dl=1 * 1e3)
    xy_grid = td.UniformGrid(dl=0.1 * 1e3)
    grid_spec = td.GridSpec(grid_x=xy_grid, grid_y=xy_grid, grid_z=z_grid)
    modeler = make_coaxial_component_modeler(
        port_types=(CoaxialLumpedPort, CoaxialLumpedPort), grid_spec=grid_spec
    )
    port0_idx = modeler.network_index(modeler.ports[0])
    port1_idx = modeler.network_index(modeler.ports[1])
    modeler_run1 = modeler.updated_copy(run_only=(port0_idx,))

    # Make sure the smatrix and impedance calculations work for reduced simulations
    modeler_data = run_component_modeler(monkeypatch, modeler_run1)
    s_matrix = modeler_data.smatrix()
    with pytest.raises(ValueError):
        validate_square_matrix(s_matrix.data, "test_method")
    _ = modeler_data.port_reference_impedances

    assert len(modeler_run1.sim_dict) == 1
    S11 = (port0_idx, port0_idx)
    S21 = (port1_idx, port0_idx)
    S12 = (port0_idx, port1_idx)
    S22 = (port1_idx, port1_idx)
    element_mappings = ((S11, S22, 1),)
    modeler_with_mappings = modeler.updated_copy(element_mappings=element_mappings)
    assert len(modeler_with_mappings.sim_dict) == 2

    # Column 1 is mapped to column 2, resulting in one simulation
    element_mappings = ((S11, S22, 1), (S21, S12, 1))
    modeler_with_mappings = modeler.updated_copy(element_mappings=element_mappings)
    tcm_data = run_component_modeler(monkeypatch, modeler_with_mappings)
    s_matrix = tcm_data.smatrix().data
    assert np.all(s_matrix.values[:, 0, 0] == s_matrix.values[:, 1, 1])
    assert np.all(s_matrix.values[:, 0, 1] == s_matrix.values[:, 1, 0])
    assert len(modeler_with_mappings.sim_dict) == 1

    # Mapping is incomplete, so two simulations are run
    element_mappings = ((S11, S22, 1), (S12, S21, 1))
    modeler_with_mappings = modeler.updated_copy(element_mappings=element_mappings)
    assert len(modeler_with_mappings.sim_dict) == 2


def test_internal_construct_smatrix_with_port_vi(monkeypatch):
    """Test _internal_construct_smatrix method by monkeypatching compute_port_VI
    with precomputed voltage and current values and comparing the final S-matrix to expected results.
    """
    # Create a simple 2-port modeler for testing
    modeler = make_component_modeler(planar_pec=False)
    freqs = np.array([1e9, 5e9, 10e9])
    modeler = modeler.updated_copy(freqs=freqs)

    # Some test data from a 20 mm lossy microstrip
    # Data is given using engineering convention exp(jwt)
    length = 0.02
    gamma = np.array(
        [
            35.845260386378 + 52.964956959149j,
            48.283102945750 + 208.91753284900j,
            50.594134809653 + 397.12168963974j,
        ]
    )
    Z0 = np.array(
        [
            18.725191534567 + 12.672421364213j,
            34.038884625562 + 7.8654410284980j,
            35.725175635077 + 4.5490999181327j,
        ]
    )
    # Break the reference impedance symmetry
    Zref = np.column_stack((0.5 * Z0, 2 * Z0))
    # Calculate analytical S matrices for power and pseudo wave formulations
    S_pseudo = calc_transmission_line_S_matrix_pseudo(Z0, Zref[:, 0], Zref[:, 1], gamma, length)
    S_power = calc_transmission_line_S_matrix_power(Z0, Zref[:, 0], Zref[:, 1], gamma, length)

    # Calculate A and B matrices where A is diagonal and B = S @ A
    A = np.tile(np.eye(2), (len(freqs), 1, 1))  # Identity matrix for each frequency
    B = S_pseudo @ A
    # Now get Voltages and Currents at each port due to excitations from each port
    Vscale = np.abs(Zref[:, :, np.newaxis]) / np.sqrt(np.real(Zref[:, :, np.newaxis]))
    Iscale = Vscale / Zref[:, :, np.newaxis]
    voltages = Vscale * (A + B)  # (f x port_out x port_in)
    currents = Iscale * (A - B)  # (f x port_out x port_in)

    port_names = [port.name for port in modeler.ports]

    # Build per-(excitation task, observed port) VI data and keep unique task indices
    sim_data_list = []
    port_name_list = []
    task_data_dict: dict[str, dict[str, dict[str, FreqDataArray]]] = {}
    sim_to_task: dict[int, str] = {}
    for j, port_in in enumerate(modeler.ports):
        task_name = modeler.get_task_name(port_in)
        # One simulation per excitation task
        sim_data = run_emulated(simulation=modeler.simulation)
        sim_data_list.append(sim_data)
        port_name_list.append(task_name)
        sim_to_task[id(sim_data)] = task_name
        # Store VI per observed port for this excitation
        task_data_dict[task_name] = {}
        for i, port_out in enumerate(modeler.ports):
            task_data_dict[task_name][port_out.name] = {
                "voltage": FreqDataArray(voltages[:, i, j], coords={"f": freqs}),
                "current": FreqDataArray(currents[:, i, j], coords={"f": freqs}),
            }

    index_data = SimulationDataMap(keys=tuple(port_name_list), values=tuple(sim_data_list))
    modeler_data = TerminalComponentModelerData(modeler=modeler, data=index_data)

    # Mock the compute_port_VI method
    def mock_compute_port_vi(port_out, sim_data):
        """Mock compute_port_VI to return voltage and current from dummy sim_data."""
        task_name = sim_to_task[id(sim_data)]
        voltage = task_data_dict[task_name][port_out.name]["voltage"]
        current = task_data_dict[task_name][port_out.name]["current"]
        return voltage, current

    # Mock port reference impedances to return frequency-dependent per-port Zref
    def mock_port_impedances(modeler_data):
        coords = {"f": np.array(freqs), "port": port_names}
        return PortDataArray(Zref, coords=coords)

    # Apply monkeypatches in all import locations
    monkeypatch.setattr(
        tidy3d.plugins.smatrix.analysis.terminal,
        "compute_port_VI",
        mock_compute_port_vi,
    )
    monkeypatch.setattr(
        tidy3d.plugins.smatrix.analysis.terminal, "port_reference_impedances", mock_port_impedances
    )

    # Test the _internal_construct_smatrix method
    S_computed = modeler_data.smatrix().data.values

    def check_S_matrix(S_computed, S_expected, tol=1e-12):
        # Check that S-matrix has correct shape
        assert S_computed.shape == (len(freqs), len(port_names), len(port_names))

        # Compare computed S-matrix with analytical values at each frequency
        for freq_idx in range(len(freqs)):
            S_computed_at_freq = S_computed[freq_idx, :, :]
            S_expected_at_freq = S_expected[freq_idx, :, :]
            max_rel_err = np.max(
                np.abs((S_computed_at_freq - S_expected_at_freq) / (S_expected_at_freq + 1e-14))
            )
            assert np.allclose(S_computed_at_freq, S_expected_at_freq, rtol=tol, atol=tol), (
                f"S-matrix mismatch at frequency index {freq_idx}\n"
                f"Expected:\n{S_expected_at_freq}\n"
                f"Computed:\n{S_computed_at_freq}\n"
                f"Difference:\n{S_computed_at_freq - S_expected_at_freq}\n"
                f"Max relative error: {max_rel_err:.2e}"
            )

    # Check pseudo wave S matrix
    check_S_matrix(S_computed, S_pseudo)

    # Check power wave S matrix
    S_computed = modeler_data.smatrix(s_param_def="power").data.values
    check_S_matrix(S_computed, S_power)


def test_wave_port_to_absorber(tmp_path):
    """Test that wave port absorber can be specified as a boolean, ABCBoundary, or ModeABCBoundary."""

    # test automatic absorber
    modeler = make_coaxial_component_modeler(port_types=(WavePort, WavePort))
    sim = list(modeler.sim_dict.values())[0]

    absorber = sim.internal_absorbers[0]

    assert absorber.boundary_spec.mode_spec == modeler.ports[0].mode_spec
    assert absorber.boundary_spec.mode_index == modeler.ports[0].mode_index
    assert absorber.boundary_spec.plane == modeler.ports[0].geometry
    assert absorber.boundary_spec.freq_spec == BroadbandModeABCSpec(
        frequency_range=(np.min(modeler.freqs), np.max(modeler.freqs))
    )

    # test to_absorber()
    absorber = modeler.ports[0].to_absorber(freq_spec=1e9)
    assert absorber.boundary_spec.freq_spec == 1e9

    absorber = modeler.ports[0].to_absorber(
        freq_spec=BroadbandModeABCSpec(frequency_range=(1e9, 2e9))
    )
    assert absorber.boundary_spec.freq_spec == BroadbandModeABCSpec(frequency_range=(1e9, 2e9))

    # test no automatic absorber
    modeler = modeler.updated_copy(ports=[modeler.ports[0].updated_copy(absorber=False)])
    sim = list(modeler.sim_dict.values())[0]
    assert len(sim.internal_absorbers) == 0

    # test custom boundary spec
    custom_boundary_spec = td.ModeABCBoundary(plane=td.Box(size=(0.1, 0.1, 0)), freq_spec=1e9)
    modeler = modeler.updated_copy(
        ports=[modeler.ports[0].updated_copy(absorber=custom_boundary_spec)]
    )
    sim = list(modeler.sim_dict.values())[0]
    absorber = sim.internal_absorbers[0]
    assert absorber.boundary_spec == custom_boundary_spec


def test_low_freq_smoothing_spec_initialization_default_values():
    """Test that LowFrequencySmoothingSpec initializes with correct default values."""
    from tidy3d.plugins.smatrix.component_modelers.terminal import ModelerLowFrequencySmoothingSpec

    spec = ModelerLowFrequencySmoothingSpec()
    assert spec.min_sampling_time == 1
    assert spec.max_sampling_time == 5
    assert spec.order == 1
    assert spec.max_deviation == 0.5


def test_low_freq_smoothing_spec_initialization_custom_values():
    """Test that LowFrequencySmoothingSpec initializes with custom values."""
    from tidy3d.plugins.smatrix.component_modelers.terminal import ModelerLowFrequencySmoothingSpec

    spec = ModelerLowFrequencySmoothingSpec(
        min_sampling_time=2, max_sampling_time=8, order=2, max_deviation=0.3
    )
    assert spec.min_sampling_time == 2
    assert spec.max_sampling_time == 8
    assert spec.order == 2
    assert spec.max_deviation == 0.3


def test_low_freq_smoothing_spec_edge_cases():
    """Test edge cases and boundary conditions."""
    from tidy3d.plugins.smatrix.component_modelers.terminal import ModelerLowFrequencySmoothingSpec

    # Test with order 0 (constant fit)
    spec = ModelerLowFrequencySmoothingSpec(order=0)
    assert spec.order == 0

    # Test with maximum order
    spec = ModelerLowFrequencySmoothingSpec(order=3)
    assert spec.order == 3

    # Test with zero max_deviation
    spec = ModelerLowFrequencySmoothingSpec(max_deviation=0.0)
    assert spec.max_deviation == 0.0

    # Test with maximum max_deviation
    spec = ModelerLowFrequencySmoothingSpec(max_deviation=1.0)
    assert spec.max_deviation == 1.0


def test_low_freq_smoothing_spec_validation_sampling_times_invalid():
    """Test validation of sampling time parameters."""
    from tidy3d.plugins.smatrix.component_modelers.terminal import ModelerLowFrequencySmoothingSpec

    # Test invalid range where min_sampling_time >= max_sampling_time
    with pytest.raises(
        ValueError, match="The minimum sampling time must be less than the maximum sampling time"
    ):
        ModelerLowFrequencySmoothingSpec(min_sampling_time=5, max_sampling_time=3)

    with pytest.raises(
        ValueError, match="The minimum sampling time must be less than the maximum sampling time"
    ):
        ModelerLowFrequencySmoothingSpec(min_sampling_time=3, max_sampling_time=3)


def test_low_freq_smoothing_spec_validation_order_bounds():
    """Test validation of order parameter bounds."""
    from tidy3d.plugins.smatrix.component_modelers.terminal import ModelerLowFrequencySmoothingSpec

    # Test valid orders
    ModelerLowFrequencySmoothingSpec(order=0)
    ModelerLowFrequencySmoothingSpec(order=3)

    # Test invalid orders
    with pytest.raises(pd.ValidationError):
        ModelerLowFrequencySmoothingSpec(order=-1)

    with pytest.raises(pd.ValidationError):
        ModelerLowFrequencySmoothingSpec(order=4)


def test_low_freq_smoothing_spec_validation_max_deviation_bounds():
    """Test validation of max_deviation parameter bounds."""
    from tidy3d.plugins.smatrix.component_modelers.terminal import ModelerLowFrequencySmoothingSpec

    # Test valid max_deviation
    ModelerLowFrequencySmoothingSpec(max_deviation=0.0)
    ModelerLowFrequencySmoothingSpec(max_deviation=1.0)

    # Test invalid max_deviation
    with pytest.raises(pd.ValidationError):
        ModelerLowFrequencySmoothingSpec(max_deviation=-0.1)


def test_low_freq_smoothing_spec_sim_dict():
    """Test that LowFrequencySmoothingSpec is correctly added to the sim_dict."""
    from tidy3d.plugins.smatrix.component_modelers.terminal import ModelerLowFrequencySmoothingSpec

    spec = ModelerLowFrequencySmoothingSpec(
        min_sampling_time=2, max_sampling_time=8, order=2, max_deviation=0.3
    )

    modeler = make_coaxial_component_modeler(port_types=(WavePort, WavePort))
    modeler = modeler.updated_copy(low_freq_smoothing=spec)
    for sim in modeler.sim_dict.values():
        assert spec.min_sampling_time == sim.low_freq_smoothing.min_sampling_time
        assert spec.max_sampling_time == sim.low_freq_smoothing.max_sampling_time
        assert spec.order == sim.low_freq_smoothing.order
        assert spec.max_deviation == sim.low_freq_smoothing.max_deviation
        assert sim.low_freq_smoothing.monitors == tuple(mnt.name for mnt in sim.monitors[-2:])

    modeler = modeler.updated_copy(low_freq_smoothing=None)
    for sim in modeler.sim_dict.values():
        assert sim.low_freq_smoothing is None


def test_S_parameter_deembedding(monkeypatch, tmp_path):
    """Test S-parameter de-embedding."""

    z_grid = td.UniformGrid(dl=1 * 1e3)
    xy_grid = td.UniformGrid(dl=0.1 * 1e3)
    grid_spec = td.GridSpec(grid_x=xy_grid, grid_y=xy_grid, grid_z=z_grid)
    modeler = make_coaxial_component_modeler(port_types=(WavePort, WavePort), grid_spec=grid_spec)

    # Make sure the smatrix and impedance calculations work for reduced simulations
    modeler_data = run_component_modeler(monkeypatch, modeler)
    s_matrix = modeler_data.smatrix()

    # set up port shifts
    port_names = [port.name for port in modeler.ports]
    coords = {"port": port_names}
    shift_vec = [0, 0]
    port_shifts = PortNameDataArray(data=shift_vec, coords=coords)

    # make sure that de-embedded S-matrices are identical to the original one if reference planes are not shifted
    S_dmb = modeler_data.change_port_reference_planes(smatrix=s_matrix, port_shifts=port_shifts)
    S_dmb_shortcut = modeler_data.smatrix_deembedded(port_shifts=port_shifts)
    assert np.allclose(S_dmb.data.values, s_matrix.data.values)
    assert np.allclose(S_dmb_shortcut.data.values, s_matrix.data.values)

    # make sure S-parameters are different if reference planes are moved
    port_shifts = PortNameDataArray(data=[-100, 200], coords=coords)
    S_dmb = modeler_data.change_port_reference_planes(smatrix=s_matrix, port_shifts=port_shifts)
    S_dmb_shortcut = modeler_data.smatrix_deembedded(port_shifts=port_shifts)
    assert not np.allclose(S_dmb.data.values, s_matrix.data.values)
    assert np.allclose(S_dmb.data.values, S_dmb_shortcut.data.values)

    # test if `.smatrix_deembedded()` raises a `ValueError` when at least one port to be shifted is not defined in TCM
    port_shifts_wrong = PortNameDataArray(data=[10, -10], coords={"port": ["wave_1", "LP_wave_2"]})

    with pytest.raises(ValueError):
        S_dmb = modeler_data.smatrix_deembedded(port_shifts=port_shifts_wrong)

    # set up a new TCM with a mixture of `WavePort` and `CoaxialLumpedPort`
    modeler_LP = make_coaxial_component_modeler(
        port_types=(WavePort, CoaxialLumpedPort), grid_spec=grid_spec
    )
    modeler_data_LP = run_component_modeler(monkeypatch, modeler_LP)

    # test if `.smatrix_deembedded()` raises a `ValueError` when one tries to de-embed a lumped port
    port_shifts_LP = PortNameDataArray(data=[10, -10], coords={"port": ["wave_1", "coax_2"]})
    with pytest.raises(ValueError):
        S_dmb = modeler_data_LP.smatrix_deembedded(port_shifts=port_shifts_LP)

    # update port shifts so that a reference plane is shifted only for `WavePort` port
    port_shifts_LP = PortNameDataArray(data=[100], coords={"port": ["wave_1"]})

    # get a new S-matrix
    s_matrix_LP = modeler_data_LP.smatrix()

    # de-embed S-matrix
    S_dmb = modeler_data_LP.change_port_reference_planes(
        smatrix=s_matrix_LP, port_shifts=port_shifts_LP
    )
    S_dmb_shortcut = modeler_data_LP.smatrix_deembedded(port_shifts=port_shifts_LP)
    assert not np.allclose(S_dmb.data.values, s_matrix_LP.data.values)
    assert np.allclose(S_dmb_shortcut.data.values, S_dmb.data.values)
