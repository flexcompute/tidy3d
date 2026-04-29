from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest
import skrf
import xarray as xr
from pydantic import ValidationError

import tidy3d as td
import tidy3d.plugins.smatrix.analysis.terminal
import tidy3d.plugins.smatrix.data.terminal
import tidy3d.plugins.smatrix.utils
from tidy3d import SimulationDataMap
from tidy3d.components.boundary import BroadbandModeABCSpec
from tidy3d.components.data.data_array import (
    CurrentFreqModeDataArray,
    FreqDataArray,
    ImpedanceFreqModeDataArray,
    ModeAmpsDataArray,
    ModeIndexDataArray,
    VoltageFreqModeDataArray,
)
from tidy3d.components.microwave.data.dataset import TransmissionLineDataset
from tidy3d.components.microwave.data.monitor_data import MicrowaveModeData
from tidy3d.components.mode.mode_solver import ModeSolver
from tidy3d.components.mode.simulation import ModeSimulation
from tidy3d.components.monitor import WARN_NUM_FREQS
from tidy3d.exceptions import (
    SetupError,
    Tidy3dError,
    Tidy3dKeyError,
)
from tidy3d.exceptions import (
    ValidationError as Tidy3dValidationError,
)
from tidy3d.plugins.smatrix import (
    CoaxialLumpedPort,
    DirectivityMonitorSpec,
    LumpedPort,
    PortDataArray,
    TerminalComponentModeler,
    TerminalComponentModelerData,
    TerminalPortDataArray,
    TerminalWavePort,
    WavePort,
)
from tidy3d.plugins.smatrix.component_modelers.terminal import (
    AUTO_RADIATION_MONITOR_NAME,
    ModelerLowFrequencySmoothingSpec,
)
from tidy3d.plugins.smatrix.data.data_array import PortNameDataArray
from tidy3d.plugins.smatrix.data.terminal import MicrowaveSMatrixData
from tidy3d.plugins.smatrix.ports.base_lumped import AbstractLumpedPort
from tidy3d.plugins.smatrix.utils import compute_F, s_to_z, validate_square_matrix

from ...utils import AssertLogLevel, AssertLogStr, run_emulated
from .terminal_component_modeler_def import (
    Rinner,
    Router,
    make_basic_filter_terminals,
    make_coaxial_component_modeler,
    make_component_modeler,
    make_differential_stripline_modeler,
    make_patch_antenna_modeler,
)

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
        td.plugins.smatrix.utils,
        "port_array_inv",
        lambda matrix: np.eye(len(modeler.matrix_indices_monitor)),
    )

    def _mock_compute_F(Z_numpy, s_param_def, compute_Finv=False):
        _num_freqs, num_ports, _ = Z_numpy.shape
        Z_diag = np.diagonal(Z_numpy, axis1=1, axis2=2)
        f_diag = 1.0 / (2.0 * np.sqrt(np.abs(Z_diag) + 1e-4))
        F = np.zeros_like(Z_numpy)
        for i in range(num_ports):
            F[:, i, i] = f_diag[:, i]
        if compute_Finv:
            finv_diag = 2.0 * np.sqrt(np.abs(Z_diag) + 1e-4)
            Finv = np.zeros_like(Z_numpy)
            for i in range(num_ports):
                Finv[:, i, i] = finv_diag[:, i]
            return F, Finv
        return F

    monkeypatch.setattr(
        td.plugins.smatrix.utils,
        "compute_F",
        _mock_compute_F,
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


def calc_transmission_line_S_matrix_pseudo(Z0, Zref1, Zref2, gamma, length, symmetric=False):
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
    symmetric : bool, optional
        If True, use symmetric_pseudo scaling (F = 1/(2*sqrt(Z))) which ensures
        S12 = S21 for reciprocal networks. Default is False.

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

    # Calculate S12 and S21 (off-diagonal transmission terms)
    if symmetric:
        # For symmetric_pseudo: F = 1/(2*sqrt(Z)), so F1/F2 = sqrt(Z2/Z1)
        # This gives S12 = S21 for reciprocal networks
        numerator_S12 = 2 * Z0 * np.sqrt(Zref1 * Zref2)
        numerator_S21 = numerator_S12
    else:
        # For pseudo: F = sqrt(Re(Z))/(2|Z|)
        numerator_S12 = (
            np.sqrt(np.real(Zref1) / np.real(Zref2))
            * (np.abs(Zref2) / np.abs(Zref1))
            * 2
            * Z0
            * Zref1
        )
        numerator_S21 = (
            np.sqrt(np.real(Zref2) / np.real(Zref1))
            * (np.abs(Zref1) / np.abs(Zref2))
            * 2
            * Z0
            * Zref2
        )

    S12 = numerator_S12 / (denom * cosh_gamma_ell)
    S21 = numerator_S21 / (denom * cosh_gamma_ell)

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
    with pytest.raises(ValidationError):
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
    with pytest.raises(ValidationError):
        _ = modeler.updated_copy(freqs=freqs)
    freqs = np.array([-1.0, 5])
    with pytest.raises(ValidationError):
        _ = modeler.updated_copy(freqs=freqs)
    freqs = np.array([1, 2, 1.9])
    with pytest.raises(ValidationError):
        _ = modeler.updated_copy(freqs=freqs)

    # Test case with non-unique value
    f_min, f_max = (0.5e9, 1.5e9)
    f0 = (f_min + f_max) / 2
    f_target = 1.35e9
    freqs = np.sort(np.append(np.linspace(f_min, f_max, 21), f_target))
    with pytest.raises(ValidationError):
        _ = modeler.updated_copy(freqs=freqs)


def test_warn_too_many_freqs():
    """Warn when the modeler's 'freqs' exceeds the FreqMonitor soft limit."""
    modeler = make_component_modeler(planar_pec=False)
    many_freqs = np.linspace(1e9, 2e9, WARN_NUM_FREQS + 1)
    with AssertLogStr("WARNING", contains_str="TerminalComponentModeler"):
        _ = modeler.updated_copy(freqs=many_freqs)

    # No "large number of frequencies" warning when under the threshold.
    few_freqs = np.linspace(1e9, 2e9, WARN_NUM_FREQS)
    with AssertLogStr("WARNING", excludes_str="large number"):
        _ = modeler.updated_copy(freqs=few_freqs)


def test_port_name_in_monitor_name():
    """Port-generated monitors embed the port name in the monitor's ``name``
    so the freq warning surfaces the port context.
    """
    modeler = make_component_modeler(planar_pec=False)
    port = modeler.ports[0]
    freqs = np.array([1e9])
    many_freqs = np.linspace(1e9, 2e9, WARN_NUM_FREQS + 1)

    voltage_mon = port.to_voltage_monitor(freqs=freqs)
    assert port.name in voltage_mon.name
    with AssertLogStr("WARNING", contains_str=voltage_mon.name):
        _ = port.to_voltage_monitor(freqs=many_freqs)

    current_mon = port.to_current_monitor(freqs=freqs)
    assert port.name in current_mon.name
    with AssertLogStr("WARNING", contains_str=current_mon.name):
        _ = port.to_current_monitor(freqs=many_freqs)

    # Also verify for AbstractWavePort (WavePort subclass).
    wave_modeler = make_coaxial_component_modeler(port_types=(WavePort, WavePort))
    wave_port = wave_modeler.ports[0]
    (mode_mon,) = wave_port.to_monitors(freqs=freqs)
    assert wave_port.name in mode_mon.name
    with AssertLogStr("WARNING", contains_str=mode_mon.name):
        _ = wave_port.to_monitors(freqs=many_freqs)


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
    with pytest.raises(ValidationError):
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


def test_sim_dict_serialization_with_custom_grid_boundaries(tmp_path):
    """TerminalComponentModeler sims should remain serializable with custom grid boundaries."""
    grid_spec = td.GridSpec(
        grid_x=td.CustomGridBoundaries(coords=np.linspace(-1000, 1000, 50)),
        grid_y=td.CustomGridBoundaries(coords=np.linspace(-1000, 1000, 50)),
        grid_z=td.CustomGridBoundaries(coords=np.linspace(-100, 100, 10)),
        wavelength=td.C_0 / 5e9,
    )
    sim = td.Simulation(
        size=(2000, 2000, 200),
        structures=[],
        grid_spec=grid_spec,
        monitors=[],
        run_time=1e-9,
    )
    port = LumpedPort(
        center=(0, 0, 0),
        size=(0.01, 100, 0),
        voltage_axis=1,
        name="port1",
        num_grid_cells=None,
        enable_snapping_points=False,
    )
    modeler = TerminalComponentModeler(
        simulation=sim,
        ports=[port],
        freqs=np.linspace(1e9, 10e9, 11),
    )

    sim_tcm = modeler.sim_dict["port1"]

    # Regression: this used to raise PydanticSerializationError because attrs contained ndarray.
    sim_tcm.model_dump_json()

    # Validate both user-facing export paths mentioned in the bug report.
    sim_tcm.to_file(tmp_path / "sim.json")
    sim_tcm.to_file(tmp_path / "sim.hdf5")


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

    # test version with per-port reference impedances (diagonal matrix)
    z_ref_values = np.array(
        [Z0 * np.eye(len(port_names)) for _ in freqs],
        dtype=complex,
    )
    z_ref = TerminalPortDataArray(
        data=z_ref_values,
        coords={"f": np.array(freqs), "port_out": port_names, "port_in": port_names},
    )
    z_matrix = s_to_z(s_matrix, reference=z_ref)
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
    skrf_S_traveling = skrf.Network.from_z(z=Z, f=freqs, s_def="traveling", z0=z0)

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
    # Convert per-port z0 (shape f, port) to diagonal TerminalPortDataArray (shape f, port, port)
    num_freqs, num_ports = z0.shape
    z0_diag = np.zeros((num_freqs, num_ports, num_ports), dtype=complex)
    idx = np.arange(num_ports)
    z0_diag[:, idx, idx] = z0
    z0_tidy3d = TerminalPortDataArray(
        data=z0_diag, coords={"f": freqs, "port_out": ports, "port_in": ports}
    )
    smatrix.values = skrf_S_power.s
    z_tidy3d = s_to_z(smatrix, reference=z0_tidy3d, s_param_def="power")
    assert np.all(np.isclose(z_tidy3d.values, Z))

    smatrix.values = skrf_S_pseudo.s
    z_tidy3d = s_to_z(smatrix, reference=z0_tidy3d, s_param_def="pseudo")
    assert np.all(np.isclose(z_tidy3d.values, Z))
    # Our symmetric_pseudo name is equivalent to "traveling" definition in scikit-rf
    smatrix.values = skrf_S_traveling.s
    z_tidy3d = s_to_z(smatrix, reference=z0_tidy3d, s_param_def="symmetric_pseudo")
    assert np.all(np.isclose(z_tidy3d.values, Z))

    # Check that invalid s_param_def raises ValueError
    with pytest.raises(ValueError, match=r"Unsupported S-parameter definition"):
        s_to_z(smatrix, reference=z0_tidy3d, s_param_def="invalid")


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

    s_matrix_container = MicrowaveSMatrixData(data=s_matrix_data)

    monkeypatch.setattr(
        TerminalComponentModelerData, "smatrix", lambda self, **kwargs: s_matrix_container
    )

    z0 = 50.0
    # Build a diagonal reference impedance matrix (z0 * I) per frequency
    z_ref_values = np.array(
        [z0 * np.eye(len(port_names)) for _ in freqs],
        dtype=complex,
    )
    z_ref = TerminalPortDataArray(
        data=z_ref_values,
        coords={"f": freqs, "port_out": port_names, "port_in": port_names},
    )
    monkeypatch.setattr(
        TerminalComponentModelerData,
        "port_reference_impedances",
        property(lambda self: z_ref),
    )

    z_matrix = modeler_data.s_to_z()

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


def test_port_snapping():
    """Make sure that the snapping behavior of the load resistor is mirrored
    by all other components in the modeler simulations with rectangular ports.
    """
    y_z_grid = td.UniformGrid(dl=0.1 * 1e3)
    x_grid = td.UniformGrid(dl=11 * 1e3)
    grid_spec = td.GridSpec(grid_x=x_grid, grid_y=y_z_grid, grid_z=y_z_grid)
    modeler = make_component_modeler(
        planar_pec=True, port_refinement=False, port_snapping=False, grid_spec=grid_spec
    )
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


def test_coarse_grid_at_port(monkeypatch):
    modeler = make_component_modeler(planar_pec=True, port_refinement=False, port_snapping=False)
    # Without port refinement the grid is much too coarse for these port sizes
    with pytest.raises(SetupError):
        _ = run_component_modeler(monkeypatch, modeler)


def test_validate_port_voltage_axis():
    with pytest.raises(ValidationError):
        LumpedPort(center=(0, 0, 0), size=(0, 1, 2), voltage_axis=0, impedance=50)


def test_validate_port_must_be_planar():
    """Test that 1D lumped ports (two zero-size dimensions) are not allowed.

    Users must provide a finite width along the lateral axis. This ensures the
    injection axis can be properly determined for the underlying lumped element.
    """
    # 1D port (two zeros) should fail validation
    with pytest.raises(ValidationError):
        LumpedPort(center=(0, 0, 0), size=(1, 0, 0), voltage_axis=0, impedance=50, name="1D_port")

    with pytest.raises(ValidationError):
        LumpedPort(center=(0, 0, 0), size=(0, 1, 0), voltage_axis=1, impedance=50, name="1D_port")

    with pytest.raises(ValidationError):
        LumpedPort(center=(0, 0, 0), size=(0, 0, 1), voltage_axis=2, impedance=50, name="1D_port")

    # Planar port (one zero) should work fine
    port = LumpedPort(
        center=(0, 0, 0), size=(0, 1, 2), voltage_axis=2, impedance=50, name="2D_port"
    )
    assert port.injection_axis == 0  # x is the injection axis (zero size)


def test_lumped_port_from_structures():
    """Test automatic lumped port setup between two terminal structures."""

    # set up terminals of a basic filter
    (str_gnd, str_resonator_basic, str_resonator_modified) = make_basic_filter_terminals()

    # Geometry and Structure
    mm = 1000  # Conversion mm to micron
    WL = 0.5 * mm
    LL1 = 5.8 * mm
    LL2 = 1.2 * mm

    # define basic parameters for automatic lumped port setup (except for signal terminal)
    lp_options = {
        "ground_terminal": str_gnd,
        "lateral_coord": -1450,
        "voltage_axis": 2,
        "impedance": 50,
    }

    # ensure that value error is triggered if signal and ground terminals are not specified
    with pytest.raises(ValueError):
        LP0 = LumpedPort.from_structures(x=-WL / 2 - LL1, name="LP1", **lp_options)

    # make sure the Lumped port is set correctly
    lp_options["signal_terminal"] = str_resonator_basic
    LP1 = LumpedPort.from_structures(x=-WL / 2 - LL1, name="LP1", **lp_options)
    assert LP1.voltage_axis == lp_options["voltage_axis"]
    assert np.isclose(LP1.impedance, lp_options["impedance"])

    # make sure that `port_width` does not cause port geometry exceed overlap of terminals
    lp_options["port_width"] = 1000
    with pytest.raises(ValueError):
        LP2 = LumpedPort.from_structures(x=-WL / 2 - LL1, name="LP1", **lp_options)

    lp_options["port_width"] = WL / 3
    LP2 = LumpedPort.from_structures(x=-WL / 2 - LL1, name="LP2", **lp_options)
    assert np.isclose(LP2.size[1], WL / 3)

    # specify lateral coordinate to select appropriate signal terminal to resolve ambiguity
    lp_options["signal_terminal"] = str_resonator_modified
    lp_options["lateral_coord"] = -2 * LL2 - WL / 2
    lp_options["port_width"] = None
    LP3 = LumpedPort.from_structures(x=-WL / 2 - LL1, name="LP3", **lp_options)
    assert np.isclose(LP3.center[1], -2 * LL2 - WL / 2)

    # test that an error is raised when port plane does not intersect any signal terminals
    with pytest.raises(ValueError):
        LP3 = LumpedPort.from_structures(y=8 * mm, name="LP3", **lp_options)

    # ensure that error is raised
    with pytest.raises(ValueError):
        lp_options["voltage_axis"] = 0
        LP3 = LumpedPort.from_structures(x=-WL / 2 - LL1, name="LP3", **lp_options)

    # test port width with lateral coords
    lp_options["port_width"] = WL / 3
    lp_options["voltage_axis"] = 2
    LP4 = LumpedPort.from_structures(x=-WL / 2 - LL1, name="LP4", **lp_options)
    assert np.isclose(LP4.size[1], lp_options["port_width"])

    # test port width with lateral coords
    lp_options["lateral_coord"] = None
    with pytest.raises(ValueError):
        LP5 = LumpedPort.from_structures(x=-WL / 2 - LL1, name="LP5", **lp_options)

    lp_options["lateral_coord"] = 10 * WL
    with pytest.raises(ValueError):
        LP6 = LumpedPort.from_structures(x=-WL / 2 - LL1, name="LP6", **lp_options)

    # ensure that validation error is raised when specified port width exceeds terminal overlap in lateral direction.
    lp_options["port_width"] = 4 * WL
    lp_options["lateral_coord"] = -2 * LL2 - WL / 2
    with pytest.raises(ValueError):
        LP6 = LumpedPort.from_structures(x=-WL / 2 - LL1, name="LP6", **lp_options)

    # ensure that validation error is raised when terminal medium is not PEC or lossy metal
    str_gnd_new = str_gnd.updated_copy(medium=td.Medium(conductivity=1e2))
    lp_options["lateral_coord"] = -2 * LL2 - WL / 2
    lp_options["ground_terminal"] = str_gnd_new
    with pytest.raises(ValueError):
        LP7 = LumpedPort.from_structures(x=-WL / 2 - LL1, name="LP7", **lp_options)


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
    with pytest.raises(ValidationError):
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
    with pytest.raises(ValidationError):
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
        port_types=(WavePort, WavePort), grid_spec=grid_spec, port_refinement=False
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
        with pytest.raises(ValidationError):
            modeler = make_coaxial_component_modeler(
                port_types=(WavePort, WavePort),
                grid_spec=grid_spec,
                use_voltage=voltage_enabled,
                use_current=current_enabled,
                port_refinement=False,
            )
        return

    modeler = make_coaxial_component_modeler(
        port_types=(WavePort, WavePort),
        grid_spec=grid_spec,
        use_voltage=voltage_enabled,
        use_current=current_enabled,
        port_refinement=False,
    )
    s_matrix = get_terminal_port_data_array(monkeypatch, modeler)

    shape_one_port = (len(modeler.freqs), len(modeler.ports))
    shape_both_ports = (len(modeler.freqs),)
    for port_in in modeler.network_dict.keys():
        for port_out in modeler.network_dict.keys():
            coords_in = {"port_in": port_in}
            coords_out = {"port_out": port_out}

            assert np.all(s_matrix.sel(**coords_in).values.shape == shape_one_port), (
                "source index not present in S matrix"
            )
            assert np.all(
                s_matrix.sel(**coords_in).sel(**coords_out).values.shape == shape_both_ports
            ), "monitor index not present in S matrix"

    # Another run with more modes in the mode spec
    mode_spec = td.MicrowaveModeSpec(num_modes=2)
    modeler: TerminalComponentModeler = modeler.updated_copy(path="ports/0/", mode_spec=mode_spec)
    s_matrix = get_terminal_port_data_array(monkeypatch, modeler)

    shape_one_port = (len(modeler.freqs), len(modeler.matrix_indices_monitor))
    shape_both_ports = (len(modeler.freqs),)
    for port_in in modeler.network_dict.keys():
        for port_out in modeler.network_dict.keys():
            coords_in = {"port_in": port_in}
            coords_out = {"port_out": port_out}

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
        port_types=(CoaxialLumpedPort, WavePort),
        grid_spec=grid_spec,
        port_refinement=False,
    )
    s_matrix = get_terminal_port_data_array(monkeypatch, modeler)

    shape_one_port = (len(modeler.freqs), len(modeler.ports))
    shape_both_ports = (len(modeler.freqs),)
    for port_in in modeler.network_dict.keys():
        for port_out in modeler.network_dict.keys():
            coords_in = {"port_in": port_in}
            coords_out = {"port_out": port_out}

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

    voltage_path = td.AxisAlignedVoltageIntegralSpec(
        center=(0.5, 0, -10),
        size=(1.0, 0, 0),
        extrapolate_to_endpoints=True,
        snap_path_to_grid=True,
        sign="+",
    )

    custom_current_path = td.Custom2DCurrentIntegralSpec.from_circular_path(
        center=center_port, radius=0.5, num_points=21, normal_axis=2, clockwise=False
    )

    mw_mode_spec = td.MicrowaveModeSpec(
        num_modes=1,
        target_neff=1.8,
        impedance_specs=(td.CustomImpedanceSpec(voltage_spec=voltage_path, current_spec=None),),
    )

    _ = WavePort(
        center=center_port,
        size=size_port,
        name="wave_port_1",
        mode_spec=mw_mode_spec,
        direction="+",
    )

    mw_mode_spec = td.MicrowaveModeSpec(
        num_modes=1,
        target_neff=1.8,
        impedance_specs=(
            td.CustomImpedanceSpec(voltage_spec=None, current_spec=custom_current_path),
        ),
    )

    _ = WavePort(
        center=center_port,
        size=size_port,
        name="wave_port_1",
        mode_spec=mw_mode_spec,
        direction="+",
    )

    with pytest.raises(ValidationError):
        mw_mode_spec = td.MicrowaveModeSpec(
            num_modes=1,
            target_neff=1.8,
            impedance_specs=(td.CustomImpedanceSpec(voltage_spec=None, current_spec=None),),
        )
        _ = WavePort(
            center=center_port,
            size=size_port,
            name="wave_port_1",
            mode_spec=mw_mode_spec,
            direction="+",
        )

    voltage_path = voltage_path.updated_copy(size=(4, 0, 0))
    with pytest.raises(ValidationError):
        mode_spec = td.MicrowaveModeSpec(
            num_modes=1,
            target_neff=1.8,
            impedance_specs=(td.CustomImpedanceSpec(voltage_spec=voltage_path, current_spec=None),),
        )
        _ = WavePort(
            center=center_port,
            size=size_port,
            name="wave_port_1",
            mode_spec=mode_spec,
            direction="+",
        )

    custom_current_path = td.Custom2DCurrentIntegralSpec.from_circular_path(
        center=center_port, radius=3, num_points=21, normal_axis=2, clockwise=False
    )

    with pytest.raises(ValidationError):
        mode_spec = td.MicrowaveModeSpec(
            num_modes=1,
            target_neff=1.8,
            impedance_specs=(
                td.CustomImpedanceSpec(voltage_spec=None, current_spec=custom_current_path),
            ),
        )
        _ = WavePort(
            center=center_port,
            size=size_port,
            name="wave_port_1",
            mode_spec=mode_spec,
            direction="+",
        )

    # Test integral path only slightly larger than port bounds
    voltage_path_large = td.AxisAlignedVoltageIntegralSpec(
        size=(0, 0, 70.000000298023424),
        center=(0, 10000, 70.000000298023424),
        extrapolate_to_endpoints=True,
        snap_path_to_grid=True,
        sign="+",
    )
    mw_mode_spec = td.MicrowaveModeSpec(
        num_modes=1,
        target_neff=1.8,
        impedance_specs=(
            td.CustomImpedanceSpec(voltage_spec=voltage_path_large, current_spec=None),
        ),
    )
    wave_port = WavePort(
        center=(0, 10000, 115.00000022351743),
        size=(500, 0, 160.00000000000003),
        name="wave_port_1",
        mode_spec=mw_mode_spec,
        direction="+",
    )
    # Make sure validation would have failed if a strict comparison was used
    # Note: Need to access the voltage spec from the mode spec now
    voltage_spec = wave_port.mode_spec.impedance_specs[0].voltage_spec
    assert wave_port.bounds[0][2] > voltage_spec.bounds[0][2]


def test_wave_port_grid_validation(tmp_path):
    """Ensure that 'num_grid_cells' is validated and works to ensure that the grid is refined around wave ports."""
    size_port = [2, 2, 0]
    center_port = [0, 0, -10]

    voltage_path = td.AxisAlignedVoltageIntegralSpec(
        center=(0.5, 0, -10),
        size=(1.0, 0, 0),
        extrapolate_to_endpoints=True,
        snap_path_to_grid=True,
        sign="+",
    )

    current_path = td.AxisAlignedCurrentIntegralSpec(
        center=(0.5, 0, -10),
        size=(0.25, 0.5, 0),
        snap_contour_to_grid=True,
        sign="+",
    )

    mw_mode_spec = td.MicrowaveModeSpec(
        num_modes=1,
        target_neff=1.8,
        impedance_specs=(
            td.CustomImpedanceSpec(voltage_spec=voltage_path, current_spec=current_path),
        ),
    )

    _ = WavePort(
        center=center_port,
        size=size_port,
        name="wave_port_1",
        mode_spec=mw_mode_spec,
        direction="+",
        num_grid_cells=None,
    )

    with pytest.raises(ValidationError):
        _ = WavePort(
            center=center_port,
            size=size_port,
            name="wave_port_1",
            mode_spec=mw_mode_spec,
            direction="+",
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
    voltage_path = td.AxisAlignedVoltageIntegralSpec(
        center=(port_pos, 0, 0),
        size=(0, 1e3, 0),
        sign="+",
    )
    mw_mode_spec = td.MicrowaveModeSpec(
        num_modes=1,
        impedance_specs=(td.CustomImpedanceSpec(voltage_spec=voltage_path, current_spec=None),),
    )
    port = WavePort(
        center=(port_pos, 0, 0),
        size=(0, 1e3, 1e3),
        name="wave_port",
        mode_spec=mw_mode_spec,
        direction="-",
    )
    modeler = modeler.updated_copy(ports=(port,))

    # Error because port is snapped to PML layers; but the error message might not
    # be very informative, e.g. "simulation.sources[0]' is outside of the simulation domain".
    # So we also check where error should be raised immediately
    with pytest.raises(SetupError):
        modeler.sim_dict

    with pytest.raises(SetupError):
        modeler._shift_value_signed(port, simulation=modeler.simulation)

    # also validate the negative side
    voltage_path = voltage_path.updated_copy(center=(-port_pos, 0, 0))
    mw_mode_spec = td.MicrowaveModeSpec(
        num_modes=1,
        impedance_specs=(td.CustomImpedanceSpec(voltage_spec=voltage_path, current_spec=None),),
    )
    port = WavePort(
        center=(-port_pos, 0, 0),
        size=(0, 1e3, 1e3),
        name="wave_port",
        mode_spec=mw_mode_spec,
        direction="+",
    )
    modeler = modeler.updated_copy(ports=(port,))
    with pytest.raises(SetupError):
        modeler.sim_dict

    with pytest.raises(SetupError):
        modeler._shift_value_signed(port, simulation=modeler.simulation)


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
    modeler: TerminalComponentModeler = modeler.updated_copy(
        radiation_monitors=(radiation_monitor,)
    )

    # Run simulation to get data
    modeler_data = run_component_modeler(monkeypatch, modeler)
    port_0_index = modeler.network_index(modeler.ports[0])
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
        port_0_index, radiation_monitor.name, a_array, a_array_raw
    )
    normalized_data_const = modeler_data._monitor_data_at_port_amplitude(
        port_0_index, radiation_monitor.name, 1.0, a_array_raw
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
    with pytest.raises(ValidationError):
        modeler = modeler.updated_copy(radiation_monitors=(radiation_monitor,))

    radiation_monitor = radiation_monitor.updated_copy(freqs=modeler.freqs)
    modeler = modeler.updated_copy(radiation_monitors=(radiation_monitor,))

    # Run simulation and get antenna parameters
    modeler_data = run_component_modeler(monkeypatch, modeler)

    # Make sure network index works for single mode / multimode cases
    if port_type == "lumped":
        port_1_network_index = modeler.network_index(modeler.ports[0])
        port_2_network_index = modeler.network_index(modeler.ports[1])
    else:
        port_1_network_index = modeler.network_index(modeler.ports[0], 0)
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
    modeler = modeler.updated_copy(radiation_monitors=(radiation_monitor,))
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
        port_types=(CoaxialLumpedPort, CoaxialLumpedPort),
        grid_spec=grid_spec,
        port_refinement=False,
    )
    port0_idx = modeler.network_index(modeler.ports[0])
    port1_idx = modeler.network_index(modeler.ports[1])
    modeler_run1 = modeler.updated_copy(run_only=(port1_idx,))

    # Make sure the smatrix and impedance calculations work for reduced simulations
    modeler_data = run_component_modeler(monkeypatch, modeler_run1)
    s_matrix = modeler_data.smatrix()
    with pytest.raises(ValueError):
        validate_square_matrix(s_matrix.data, "test_method")
    _ = modeler_data.port_reference_impedances

    # Verify the correct diagonal element (self-coupling a[j,j]) is used for normalization.
    # With run_only=(port1_idx,), port1 is at row index 1 in monitor_indices.
    # The bug would use a[port0, port1] (row 0) instead of a[port1, port1] (row 1).
    a_matrix, b_matrix = modeler_data.port_pseudo_wave_matrices
    monitor_indices = list(modeler_run1.matrix_indices_monitor)
    port1_row = monitor_indices.index(port1_idx)
    a_self = a_matrix.values[:, port1_row, 0]
    expected_col = b_matrix.values[:, :, 0] / a_self[:, np.newaxis]
    actual_col = s_matrix.data.sel(port_in=port1_idx).values
    np.testing.assert_allclose(actual_col, expected_col)

    assert len(modeler_run1.sim_dict) == 1
    S11 = (port0_idx, port0_idx)
    S21 = (port1_idx, port0_idx)
    S12 = (port0_idx, port1_idx)
    S22 = (port1_idx, port1_idx)
    element_mappings = ((S22, S11, 1 + 0j),)
    modeler_with_mappings = modeler.updated_copy(element_mappings=element_mappings)
    assert len(modeler_with_mappings.sim_dict) == 2

    # Column 0 is mapped from column 1, resulting in one simulation
    element_mappings = ((S22, S11, 1 + 0j), (S12, S21, 1 + 0j))
    modeler_with_mappings = modeler.updated_copy(element_mappings=element_mappings)
    tcm_data = run_component_modeler(monkeypatch, modeler_with_mappings)
    s_matrix = tcm_data.smatrix().data
    assert np.all(s_matrix.values[:, 0, 0] == s_matrix.values[:, 1, 1])
    assert np.all(s_matrix.values[:, 0, 1] == s_matrix.values[:, 1, 0])
    assert len(modeler_with_mappings.sim_dict) == 1

    # Mapping is incomplete, so two simulations are run
    element_mappings = ((S22, S11, 1 + 0j), (S21, S12, 1 + 0j))
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
            12.843105732941 + 15.394208173652j,
            28.567192048123 + 9.1023847562915j,
            31.209457618234 + 3.8475102934671j,
        ]
    )
    Z01 = np.array(
        [
            18.725191534567 + 12.672421364213j,
            34.038884625562 + 7.8654410284980j,
            35.725175635077 + 4.5490999181327j,
        ]
    )
    Z02 = np.array(
        [
            24.156839210485 + 10.234195827361j,
            41.892301567293 + 6.7812039451120j,
            29.451276384019 + 5.1298475620183j,
        ]
    )
    # Break the reference impedance symmetry
    Zref = np.column_stack((Z01, Z02))
    # Calculate analytical S matrices for power and pseudo wave formulations
    S_pseudo = calc_transmission_line_S_matrix_pseudo(Z0, Zref[:, 0], Zref[:, 1], gamma, length)
    S_symmetric_pseudo = calc_transmission_line_S_matrix_pseudo(
        Z0, Zref[:, 0], Zref[:, 1], gamma, length, symmetric=True
    )
    S_power = calc_transmission_line_S_matrix_power(Z0, Zref[:, 0], Zref[:, 1], gamma, length)
    Zref3 = Zref[:, :, np.newaxis]
    # Calculate A and B matrices where A is diagonal and B = S @ A

    A = np.tile(np.eye(2), (len(freqs), 1, 1))  # Identity matrix for each frequency
    B = S_pseudo @ A
    # Now get Voltages and Currents at each port due to excitations from each port
    Vscale = np.abs(Zref3) / np.sqrt(np.real(Zref3))
    Iscale = Vscale / Zref3
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
    # Build a diagonal 3D impedance matrix (f, port_out, port_in)
    def mock_port_impedances(modeler_data):
        num_ports = len(port_names)
        Zref_3d = Zref[:, :, None] * np.eye(num_ports)
        coords = {"f": np.array(freqs), "port_out": port_names, "port_in": port_names}
        return TerminalPortDataArray(Zref_3d, coords=coords)

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
    S_computed = modeler_data.smatrix().data.values
    check_S_matrix(S_computed, S_pseudo)

    # Check power wave S matrix
    S_computed = modeler_data.smatrix(s_param_def="power").data.values
    check_S_matrix(S_computed, S_power)

    # Check symmetric_pseudo wave S matrix
    S_computed = modeler_data.smatrix(s_param_def="symmetric_pseudo").data.values
    check_S_matrix(S_computed, S_symmetric_pseudo)

    # Check that invalid s_param_def raises ValueError
    with pytest.raises(ValueError, match=r"Unsupported S-parameter definition"):
        modeler_data.smatrix(s_param_def="invalid")


def test_wave_port_to_absorber(tmp_path):
    """Test that wave port absorber can be specified as a boolean, ABCBoundary, or ModeABCBoundary."""

    # test automatic absorber
    modeler = make_coaxial_component_modeler(port_types=(WavePort, WavePort))
    sim = list(modeler.sim_dict.values())[0]

    absorber = sim.internal_absorbers[0]

    assert absorber.boundary_spec.mode_spec == modeler.ports[0].mode_spec
    assert absorber.boundary_spec.mode_index == 0
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

    spec = ModelerLowFrequencySmoothingSpec()
    assert spec.min_sampling_time == 1
    assert spec.max_sampling_time == 5
    assert spec.order == 1
    assert spec.max_deviation == 0.5


def test_low_freq_smoothing_spec_initialization_custom_values():
    """Test that LowFrequencySmoothingSpec initializes with custom values."""

    spec = ModelerLowFrequencySmoothingSpec(
        min_sampling_time=2, max_sampling_time=8, order=2, max_deviation=0.3
    )
    assert spec.min_sampling_time == 2
    assert spec.max_sampling_time == 8
    assert spec.order == 2
    assert spec.max_deviation == 0.3


def test_low_freq_smoothing_spec_edge_cases():
    """Test edge cases and boundary conditions."""

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

    # Test invalid range where min_sampling_time >= max_sampling_time
    with pytest.raises(
        ValueError, match=r"The minimum sampling time must be less than the maximum sampling time"
    ):
        ModelerLowFrequencySmoothingSpec(min_sampling_time=5, max_sampling_time=3)

    with pytest.raises(
        ValueError, match=r"The minimum sampling time must be less than the maximum sampling time"
    ):
        ModelerLowFrequencySmoothingSpec(min_sampling_time=3, max_sampling_time=3)


def test_low_freq_smoothing_spec_validation_order_bounds():
    """Test validation of order parameter bounds."""

    # Test valid orders
    ModelerLowFrequencySmoothingSpec(order=0)
    ModelerLowFrequencySmoothingSpec(order=3)

    # Test invalid orders
    with pytest.raises(ValidationError):
        ModelerLowFrequencySmoothingSpec(order=-1)

    with pytest.raises(ValidationError):
        ModelerLowFrequencySmoothingSpec(order=4)


def test_low_freq_smoothing_spec_validation_max_deviation_bounds():
    """Test validation of max_deviation parameter bounds."""

    # Test valid max_deviation
    ModelerLowFrequencySmoothingSpec(max_deviation=0.0)
    ModelerLowFrequencySmoothingSpec(max_deviation=1.0)

    # Test invalid max_deviation
    with pytest.raises(ValidationError):
        ModelerLowFrequencySmoothingSpec(max_deviation=-0.1)


def test_low_freq_smoothing_spec_sim_dict():
    """Test that LowFrequencySmoothingSpec is correctly added to the sim_dict."""

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
    modeler = make_coaxial_component_modeler(
        port_types=(WavePort, WavePort),
        grid_spec=grid_spec,
        port_refinement=False,
    )

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

    # Test multimodal version of de-embedding
    old_custom_spec = modeler.ports[0].mode_spec.impedance_specs[0]
    new_mode_spec = modeler.ports[0].mode_spec.updated_copy(
        num_modes=2, impedance_specs=(old_custom_spec, old_custom_spec)
    )
    modeler = modeler.updated_copy(path="ports/0/", mode_spec=new_mode_spec)
    modeler_data = run_component_modeler(monkeypatch, modeler)
    s_matrix = modeler_data.smatrix()
    # reuse previous port shifts
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


def test_wave_port_extrusion_coaxial():
    """Test extrusion of structures wave port absorber."""

    # define a terminal component modeler
    tcm = make_coaxial_component_modeler(
        length=100000,
        port_types=(WavePort, WavePort),
    )

    # update ports and set flag to extrude structures
    ports = tcm.ports
    port_1 = ports[0]
    port_2 = ports[1]
    port_1 = port_1.updated_copy(center=(0, 0, -50000), extrude_structures=True)

    # test that structure extrusion requires an internal absorber (should raise ValidationError)
    with pytest.raises(ValidationError):
        _ = port_2.updated_copy(center=(0, 0, 50000), extrude_structures=True, absorber=False)

    # define a valid waveport
    port_2 = port_2.updated_copy(center=(0, 0, 50000), extrude_structures=True)

    # update component modeler
    tcm = tcm.updated_copy(ports=[port_1, port_2])

    # generate simulations from component modeler
    sim = tcm.base_sim

    # get injection axis that would be used to extrude structure
    inj_axis = sim.internal_absorbers[0].size.index(0.0)

    # get grid boundaries
    bnd_coords = sim.grid.boundaries.to_list[inj_axis]

    # get size of structures along injection axis directions
    str_bnds = [
        np.min(sim.structures[-4].geometry.geometries[0].slab_bounds),
        np.max(sim.structures[-2].geometry.geometries[0].slab_bounds),
    ]

    pec_bnds = []

    # infer placement of PEC plates beyond internal absorber
    for absorber in sim.internal_absorbers:
        absorber_cntr = absorber.center[inj_axis]
        right_ind = np.searchsorted(bnd_coords, absorber_cntr, side="right")
        left_ind = np.searchsorted(bnd_coords, absorber_cntr, side="left") - 1
        pec_bnds.append(bnd_coords[right_ind + 1])
        pec_bnds.append(bnd_coords[left_ind - 1])

    # get range of coordinates along injection axis for PEC plates
    pec_bnds = [np.min(pec_bnds), np.max(pec_bnds)]

    # ensure that structures were extruded up to PEC plates
    assert all(np.isclose(str_bnd, pec_bnd) for str_bnd, pec_bnd in zip(str_bnds, pec_bnds))

    # generate a new TCM simulation to test edge case when wave port plane does not intersect any structures
    tcm = make_coaxial_component_modeler(
        length=100000, port_types=(WavePort, WavePort), use_current=False
    )

    # update ports and set flag to extrude structures
    ports = tcm.ports
    port_1 = ports[0]
    port_2 = ports[1]
    port_1 = port_1.updated_copy(center=(0, 0, -50000), extrude_structures=True)
    port_2 = port_2.updated_copy(center=(0, 0, 50000), extrude_structures=True)

    # move wave port plane so that is does not intersect any structures
    port_1_center_new = (638.4, 0.0, -51000)

    # update WavePort
    port_1 = port_1.updated_copy(center=port_1_center_new)

    # update component modeler
    tcm = tcm.updated_copy(ports=[port_1, port_2])

    # make sure that SetupError is raised when port does not intersect any structure
    with pytest.raises(SetupError):
        sim = tcm.base_sim


def test_wave_port_extrusion_named_structures_no_duplicate_names():
    """With extrude_structures=True, extruded structures must have unique names when originals are named."""
    tcm = make_coaxial_component_modeler(
        length=100000,
        port_types=(WavePort, WavePort),
    )
    # Give structures explicit names so that extrusion would duplicate names without the fix
    sim = tcm.simulation
    structures_named = [
        sim.structures[0].updated_copy(name="inner_conductor"),
        sim.structures[1].updated_copy(name="outer_shell"),
    ]
    sim = sim.updated_copy(structures=structures_named)
    tcm = tcm.updated_copy(simulation=sim)

    port_1 = tcm.ports[0].updated_copy(center=(0, 0, -50000), extrude_structures=True)
    port_2 = tcm.ports[1].updated_copy(center=(0, 0, 50000), extrude_structures=True)
    tcm = tcm.updated_copy(ports=[port_1, port_2])

    base_sim = tcm.base_sim
    names = [s.name for s in base_sim.structures if s.name is not None]
    assert len(names) == len(set(names)), (
        f"Duplicate structure names found after waveport extrusion: {names}."
    )


def test_wave_port_extrusion_differential_stripline():
    """Test extrusion of structures wave port absorber for differential stripline."""

    tcm = make_differential_stripline_modeler()

    # update ports and set flag to extrude structures
    ports = tcm.ports
    port_1 = ports[0]
    port_2 = ports[1]
    port_1 = port_1.updated_copy(extrude_structures=True)

    # test that structure extrusion requires an internal absorber (should raise ValidationError)
    with pytest.raises(ValidationError):
        _ = port_2.updated_copy(extrude_structures=True, absorber=False)

    # define a valid waveport
    port_2 = port_2.updated_copy(extrude_structures=True)

    # update component modeler
    tcm = tcm.updated_copy(ports=[port_1, port_2])

    # generate simulations from component modeler
    sim = tcm.base_sim

    # get injection axis that would be used to extrude structure
    inj_axis = sim.internal_absorbers[0].size.index(0.0)

    # get grid boundaries
    bnd_coords = sim.grid.boundaries.to_list[inj_axis]

    # get size of structures along injection axis directions
    str_bnds = [
        np.min(sim.structures[-6].geometry.geometries[0].slab_bounds),
        np.max(sim.structures[-1].geometry.geometries[0].slab_bounds),
    ]

    pec_bnds = []

    # infer placement of PEC plates beyond internal absorber
    for absorber in sim._shifted_internal_absorbers:
        # get the PEC box with its face surfaces
        (box, inj_axis, direction) = sim._pec_frame_box(absorber)
        surfaces = box.surfaces(box.size, box.center)

        # get extrusion coordinates and a cutting plane for inference of intersecting structures.
        sign = 1 if direction == "+" else -1
        cutting_plane = surfaces[2 * inj_axis + (1 if direction == "+" else 0)]

        # get extrusion extent along injection axis
        pec_bnds.append(cutting_plane.center[inj_axis])

    # get range of coordinates along injection axis for PEC plates
    pec_bnds = [np.min(pec_bnds), np.max(pec_bnds)]

    # ensure that structures were extruded up to PEC plates
    assert all(np.isclose(str_bnd, pec_bnd) for str_bnd, pec_bnd in zip(str_bnds, pec_bnds))

    # test scenario when wave port extrusion is requested, but port plane does not intersect any structures
    mil = 25.4
    port_1_center_new = (0, 0, -2010 * mil)

    # re-assemble a new component modeler
    tcm = make_differential_stripline_modeler()

    # update ports and set flag to extrude structures
    ports = tcm.ports
    port_1 = ports[0]
    port_2 = ports[1]
    port_1 = port_1.updated_copy(extrude_structures=True)
    port_2 = port_2.updated_copy(extrude_structures=True)

    # update WavePort
    port_1 = port_1.updated_copy(center=port_1_center_new)

    # update component modeler
    tcm = tcm.updated_copy(ports=[port_1, port_2])

    # make sure that the error is triggered
    with pytest.raises(SetupError):
        sim = tcm.base_sim

    # update component modeler
    tcm = tcm.updated_copy(ports=[port_2, port_1])

    # make sure that the error is triggered even when ports are reshuffled
    with pytest.raises(SetupError):
        sim = tcm.base_sim


def test_wave_port_extrusion_vertex_tolerance():
    """Test that extrusion captures structures whose vertices are slightly offset from the port plane.

    Geometry vertices in PCB layouts may not land exactly on the waveport plane
    due to coordinate precision. The extrusion tolerance should bridge these
    small gaps (up to ~10 nm) for both WavePort and TerminalWavePort.
    """
    tcm = make_differential_stripline_modeler()
    sim = tcm.simulation
    num_original_structures = len(sim.structures)

    # Shorten structures by a small offset so they don't quite reach the port planes,
    # mimicking common coordinate precision gaps in PCB layouts.
    gap_offset = 0.001
    shortened_structures = []
    for structure in sim.structures:
        geom = structure.geometry
        if isinstance(geom, td.GeometryGroup):
            new_geoms = []
            for sub_geom in geom.geometries:
                size = list(sub_geom.size)
                size[2] -= 2 * gap_offset
                new_geoms.append(sub_geom.updated_copy(size=tuple(size)))
            new_geom = geom.updated_copy(geometries=new_geoms)
        else:
            size = list(geom.size)
            size[2] -= 2 * gap_offset
            new_geom = geom.updated_copy(size=tuple(size))
        shortened_structures.append(structure.updated_copy(geometry=new_geom))

    sim = sim.updated_copy(structures=shortened_structures)

    # Keep one WavePort and replace the other with a TerminalWavePort using
    # automatic terminal detection so the regression covers both port types.
    port_1 = tcm.ports[0].updated_copy(extrude_structures=True)
    port_2 = TerminalWavePort(
        center=tcm.ports[1].center,
        size=tcm.ports[1].size,
        direction=tcm.ports[1].direction,
        name="TWP2",
        absorber=True,
        extrude_structures=True,
    )

    tcm = tcm.updated_copy(simulation=sim, ports=[port_1, port_2])

    # Verify extrusion happens early in the pipeline (in _base_sim_with_grid_and_lumped_elements),
    # before conductor detection, so that downstream steps see the extruded structures.
    grid_sim = tcm._base_sim_with_grid_and_lumped_elements
    num_extruded_early = len(grid_sim.structures) - num_original_structures
    assert num_extruded_early > 0, (
        "Extrusion did not happen in _base_sim_with_grid_and_lumped_elements"
    )
    assert num_extruded_early == 2 * num_original_structures

    # Build the full base sim and verify the same extruded structures propagate through
    base_sim = tcm.base_sim
    num_extruded = len(base_sim.structures) - num_original_structures
    assert num_extruded == num_extruded_early
    assert len(base_sim.internal_absorbers) == 2

    # Verify plotting works with extrusion enabled for both WavePort and TerminalWavePort.
    ax = tcm.plot_port("WP1")
    assert ax is not None
    plt.close()

    ax = tcm.plot_port("TWP2")
    assert ax is not None
    plt.close()


def test_custom_source_time(monkeypatch, tmp_path):
    """Test that custom_source_time is properly used in the terminal component modeler."""
    # Create a custom source time
    custom_source = td.GaussianPulse(freq0=2e9, fwidth=1e8)

    # Create modeler with custom source time
    modeler = make_component_modeler(
        planar_pec=True, port_refinement=False, custom_source_time=custom_source
    )

    # Run the modeler and verify it works with custom source time
    modeler_data = run_component_modeler(monkeypatch, modeler=modeler)

    # Verify that simulations were created and run successfully
    s_matrix = modeler_data.smatrix()
    assert s_matrix is not None

    # Verify that the simulations in sim_dict use the custom source time
    for sim in modeler.sim_dict.values():
        # Each simulation should have sources with the custom source time
        assert len(sim.sources) > 0
        for source in sim.sources:
            assert source.source_time.freq0 == custom_source.freq0
            assert source.source_time.fwidth == custom_source.fwidth


def test_validate_run_only_uniqueness():
    """Test that run_only validator rejects duplicate entries for TerminalComponentModeler."""
    modeler = make_component_modeler(planar_pec=True)

    # Get valid network indices
    port0_idx = modeler.network_index(modeler.ports[0])
    port1_idx = modeler.network_index(modeler.ports[1])

    # Test with duplicate entries - should raise ValidationError
    with pytest.raises(ValidationError, match=r"duplicate entries"):
        modeler.updated_copy(run_only=(port0_idx, port0_idx, port1_idx))


def test_validate_run_only_membership():
    """Test that run_only validator rejects invalid indices for TerminalComponentModeler."""
    modeler = make_component_modeler(planar_pec=True)

    # Test with invalid index - should raise ValidationError
    with pytest.raises(ValidationError, match=r"not present in"):
        modeler.updated_copy(run_only=("invalid_port_name",))

    # Test with partially invalid indices
    port0_idx = modeler.network_index(modeler.ports[0])
    with pytest.raises(ValidationError, match=r"not present in"):
        modeler.updated_copy(run_only=(port0_idx, "invalid_port"))


def test_validate_run_only_with_wave_ports():
    """Test run_only validation with WavePorts in TerminalComponentModeler."""
    z_grid = td.UniformGrid(dl=1 * 1e3)
    xy_grid = td.UniformGrid(dl=0.1 * 1e3)
    grid_spec = td.GridSpec(grid_x=xy_grid, grid_y=xy_grid, grid_z=z_grid)
    modeler = make_coaxial_component_modeler(
        port_types=(WavePort, WavePort),
        grid_spec=grid_spec,
        port_refinement=False,
    )

    port0_idx = modeler.network_index(modeler.ports[0], 0)
    port1_idx = modeler.network_index(modeler.ports[1], 0)

    # Valid case
    modeler_updated = modeler.updated_copy(run_only=(port0_idx,))
    assert modeler_updated.run_only == (port0_idx,)

    # Invalid case
    with pytest.raises(ValidationError, match=r"not present in"):
        modeler.updated_copy(run_only=("nonexistent_wave_port",))


def test_radiation_monitors_auto():
    """Test DirectivityMonitorSpec with various configurations."""

    # Test 1: DirectivityMonitorSpec with default values
    auto_spec = DirectivityMonitorSpec()
    assert auto_spec.buffer == 2
    assert auto_spec.num_theta_points == 100
    assert auto_spec.num_phi_points == 200
    assert auto_spec.name is None
    assert auto_spec.freqs is None

    # Test 2: DirectivityMonitorSpec with custom values
    auto_spec_custom = DirectivityMonitorSpec(
        name="custom_auto",
        buffer=3,
        num_theta_points=50,
        num_phi_points=100,
    )
    assert auto_spec_custom.name == "custom_auto"
    assert auto_spec_custom.buffer == 3
    assert auto_spec_custom.num_theta_points == 50
    assert auto_spec_custom.num_phi_points == 100

    # Test 3: Default DirectivityMonitorSpec in modeler
    tcm = make_patch_antenna_modeler(padding=(0.5, 0.5, 0.5))
    tcm_with_auto = tcm.updated_copy(radiation_monitors=(auto_spec, auto_spec_custom))
    sim = tcm_with_auto.sim_dict["lumped_port"]

    assert len(sim.monitors) == 5

    assert isinstance(sim.monitors[-2], td.DirectivityMonitor)
    assert len(sim.monitors[-2].theta) == 100
    assert len(sim.monitors[-2].phi) == 200
    assert (sim.monitors[-2].freqs == tcm_with_auto.freqs).all()
    assert sim.monitors[-2].name == f"{AUTO_RADIATION_MONITOR_NAME}_0"

    assert isinstance(sim.monitors[-1], td.DirectivityMonitor)
    assert len(sim.monitors[-1].theta) == auto_spec_custom.num_theta_points
    assert len(sim.monitors[-1].phi) == auto_spec_custom.num_phi_points
    assert (sim.monitors[-1].freqs == tcm_with_auto.freqs).all()
    assert sim.monitors[-1].name == auto_spec_custom.name

    # Test 4: Mixed DirectivityMonitor and DirectivityMonitorSpec
    manual_monitor = td.DirectivityMonitor(
        size=tuple(0.8 * np.array(tcm.simulation.size)),
        center=tcm.simulation.center,
        freqs=tcm.freqs[1:3],
        name="manual_monitor",
        theta=np.linspace(0, np.pi, 50),
        phi=np.linspace(-np.pi, np.pi, 100),
    )
    tcm_mixed = tcm.updated_copy(
        radiation_monitors=(
            manual_monitor,
            DirectivityMonitorSpec(name="auto_1", buffer=2),
            DirectivityMonitorSpec(name="auto_2", buffer=3),
        )
    )
    sim_mixed = tcm_mixed.sim_dict["lumped_port"]

    # Check that all three monitors are present
    monitor_names = {m.name for m in sim_mixed.monitors}
    assert "manual_monitor" in monitor_names
    assert "auto_1" in monitor_names
    assert "auto_2" in monitor_names

    # Test 5: Buffer distance validation - should raise error with insufficient padding
    with pytest.raises(ValueError, match=r"Automatic construction of radiation monitors failed"):
        tcm_small = make_patch_antenna_modeler(padding=(0.01, 0.01, 0.5))
        tcm_small_with_auto = tcm_small.updated_copy(radiation_monitors=(DirectivityMonitorSpec(),))
        _ = tcm_small_with_auto.sim_dict["lumped_port"]

    # Test 6: Custom buffer parameter - smaller buffer should fail with tight padding
    with pytest.raises(ValueError, match=r"Automatic construction of radiation monitors failed"):
        tcm_tight = make_patch_antenna_modeler(padding=(0.05, 0.05, 0.5))
        tcm_tight_with_auto = tcm_tight.updated_copy(
            radiation_monitors=(DirectivityMonitorSpec(buffer=5),)  # Require 5 cells
        )
        _ = tcm_tight_with_auto.sim_dict["lumped_port"]

    # Test 7: custom_origin propagation
    # Test 7a: custom_origin with explicit coordinates
    custom_origin_coords = (1.0, 2.0, 3.0)
    spec_with_origin = DirectivityMonitorSpec(
        name="with_custom_origin", custom_origin=custom_origin_coords
    )
    tcm_with_origin = tcm.updated_copy(radiation_monitors=(spec_with_origin,))
    sim_with_origin = tcm_with_origin.sim_dict["lumped_port"]

    # Find the generated DirectivityMonitor
    origin_monitor = [m for m in sim_with_origin.monitors if m.name == "with_custom_origin"][0]
    assert isinstance(origin_monitor, td.DirectivityMonitor)
    assert origin_monitor.custom_origin == custom_origin_coords

    # Test 7b: custom_origin with None (should be accepted and propagated)
    spec_with_none_origin = DirectivityMonitorSpec(name="with_none_origin", custom_origin=None)
    tcm_with_none_origin = tcm.updated_copy(radiation_monitors=(spec_with_none_origin,))
    sim_with_none_origin = tcm_with_none_origin.sim_dict["lumped_port"]

    # Find the generated DirectivityMonitor
    none_origin_monitor = [
        m for m in sim_with_none_origin.monitors if m.name == "with_none_origin"
    ][0]
    assert isinstance(none_origin_monitor, td.DirectivityMonitor)
    assert none_origin_monitor.custom_origin is None

    # Test 7c: default custom_origin (should use the default value from field definition)
    spec_default_origin = DirectivityMonitorSpec(name="default_origin")
    tcm_default_origin = tcm.updated_copy(radiation_monitors=(spec_default_origin,))
    sim_default_origin = tcm_default_origin.sim_dict["lumped_port"]

    # Find the generated DirectivityMonitor
    default_origin_monitor = [m for m in sim_default_origin.monitors if m.name == "default_origin"][
        0
    ]
    assert isinstance(default_origin_monitor, td.DirectivityMonitor)
    # Default should be (0, 0, 0) as defined in the field
    assert default_origin_monitor.custom_origin == (0, 0, 0)


def test_wave_port_mode_index_validation():
    """Test that WavePort.mode_index is validated correctly."""
    # Create mode_spec with 3 modes - don't specify impedance_specs to use defaults
    mode_spec = td.MicrowaveModeSpec(num_modes=3)

    # Valid: single mode
    port = WavePort(
        center=(0, 0, -10),
        size=(0, 2, 2),
        name="port1",
        mode_spec=mode_spec,
        direction="+",
        mode_index=0,
    )
    assert port._mode_indices() == (0,)

    # Invalid: index greater than number of modes
    with pytest.raises(ValidationError):
        WavePort(
            center=(0, 0, -10),
            size=(0, 2, 2),
            name="port_neg",
            mode_spec=mode_spec,
            direction="+",
            mode_index=4,
        )

    # Valid: multiple modes as tuple
    port = WavePort(
        center=(0, 0, -10),
        size=(0, 2, 2),
        name="port2",
        mode_spec=mode_spec,
        direction="+",
        mode_selection=[0, 2],
    )
    assert port._mode_indices() == (0, 2)

    # Valid: None (use all modes)
    port = WavePort(
        center=(0, 0, -10),
        size=(0, 2, 2),
        name="port3",
        mode_spec=mode_spec,
        direction="+",
        mode_selection=None,
    )
    assert port._mode_indices() == (0, 1, 2)

    # Invalid: negative index
    with pytest.raises(ValidationError, match=r"non-negative"):
        WavePort(
            center=(0, 0, -10),
            size=(0, 2, 2),
            name="port_neg",
            mode_spec=mode_spec,
            direction="+",
            mode_selection=(-1,),
        )

    # Invalid: index >= num_modes
    with pytest.raises(ValidationError, match=r"mode_spec.num_modes"):
        WavePort(
            center=(0, 0, -10),
            size=(0, 2, 2),
            name="port_large",
            mode_spec=mode_spec,
            direction="+",
            mode_selection=(3,),  # num_modes is 3, so valid range is 0-2
        )

    # Invalid: duplicate indices
    with pytest.raises(ValidationError, match=r"duplicate"):
        WavePort(
            center=(0, 0, -10),
            size=(0, 2, 2),
            name="port_dup",
            mode_spec=mode_spec,
            direction="+",
            mode_selection=(0, 1, 0),
        )


def test_wave_port_multimode_absorption_warning():
    """Test that WavePort warns when absorber is enabled with multiple modes."""
    # Create mode_spec with 3 modes
    mode_spec = td.MicrowaveModeSpec(num_modes=3)

    # Test 1: Single mode with absorber=True - should NOT warn
    with AssertLogStr("WARNING", excludes_str="multimode"):
        port = WavePort(
            center=(0, 0, -10),
            size=(0, 2, 2),
            name="port_single_mode",
            mode_spec=td.MicrowaveModeSpec(num_modes=1),
            direction="+",
            absorber=True,
        )

    # Test 2: Multiple modes with absorber=True - SHOULD warn
    with AssertLogStr("WARNING", contains_str="multimode"):
        port = WavePort(
            center=(0, 0, -10),
            size=(0, 2, 2),
            name="port_multimode",
            mode_spec=mode_spec,
            direction="+",
            absorber=True,
        )

    # Test 3: Multiple modes with absorber=False - should NOT warn
    with AssertLogStr("WARNING", excludes_str="multimode"):
        port = WavePort(
            center=(0, 0, -10),
            size=(0, 2, 2),
            name="port_no_absorber",
            mode_spec=mode_spec,
            direction="+",
            absorber=False,
        )


def test_wave_port_mode_index_with_modeler():
    """Test that WavePort.mode_index works correctly with TerminalComponentModeler."""
    z_grid = td.UniformGrid(dl=1 * 1e3)
    xy_grid = td.UniformGrid(dl=0.1 * 1e3)
    grid_spec = td.GridSpec(grid_x=xy_grid, grid_y=xy_grid, grid_z=z_grid)

    # Create mode_spec with 4 modes - use defaults
    mode_spec = td.MicrowaveModeSpec(num_modes=4)

    # Create port with mode_index selecting subset
    port = WavePort(
        center=(0, 0, -10),
        size=(0, 2, 2),
        name="port1",
        mode_spec=mode_spec,
        direction="+",
        mode_selection=(0, 2),  # Only modes 0 and 2
        num_grid_cells=None,
    )

    # Verify the port has only the selected modes
    assert port._mode_indices() == (0, 2)

    # Create a simple simulation to test with
    sim = td.Simulation(
        size=(10, 10, 10),
        grid_spec=grid_spec,
        run_time=1e-12,
        structures=[],
        boundary_spec=td.BoundarySpec.all_sides(boundary=td.PML()),
    )

    # Create a modeler with this port
    lumped_port = LumpedPort(
        center=(0, 0, 10),
        size=(1, 1, 0),
        voltage_axis=0,  # x-direction since port is in x-y plane
        name="lumped_port",
        num_grid_cells=None,
        enable_snapping_points=False,
    )

    modeler = TerminalComponentModeler(
        simulation=sim,
        ports=[port, lumped_port],
        freqs=[2e9, 3e9],
        element_mappings=(),
    )

    # Verify network_dict only has entries for the selected modes
    network_indices = set(modeler.network_dict.keys())
    assert "port1@0" in network_indices
    assert "port1@2" in network_indices
    assert "port1@1" not in network_indices
    assert "port1@3" not in network_indices
    assert "lumped_port" in network_indices


def test_get_task_name():
    """Test get_task_name with RF ports."""

    # First make sure ports cannot have @ in their name
    with pytest.raises(ValidationError):
        lumped_port = LumpedPort(
            center=(0, 0, 0),
            size=(1, 0, 0.5),
            voltage_axis=0,
            name="lumped1@0",
        )

    lumped_port = LumpedPort(
        center=(0, 0, 0),
        size=(1, 0, 0.5),
        voltage_axis=0,
        name="lumped1",
    )

    # Test with lumped port (no mode_index) - should work
    task_name = TerminalComponentModeler.get_task_name(port=lumped_port)
    assert task_name == "lumped1"

    # Test with lumped port and mode_index - should raise ValueError
    with pytest.raises(ValueError, match=r"mode_index.*should not be specified.*lumped port"):
        TerminalComponentModeler.get_task_name(port=lumped_port, mode_index=0)

    # Test with CoaxialLumpedPort as well
    coax_port = CoaxialLumpedPort(
        center=(0, 0, 0),
        outer_diameter=2.0,
        inner_diameter=0.5,
        normal_axis=2,
        direction="+",
        name="coax1",
    )

    # Test with coaxial lumped port (no mode_index) - should work
    task_name = TerminalComponentModeler.get_task_name(port=coax_port)
    assert task_name == "coax1"

    # Test with coaxial lumped port and mode_index - should raise ValueError
    with pytest.raises(ValueError, match=r"mode_index.*should not be specified.*lumped port"):
        TerminalComponentModeler.get_task_name(port=coax_port, mode_index=1)

    wave_port = WavePort(
        center=(0, 0, 0),
        size=(4, 4, 0),
        direction="+",
        name="wave",
        mode_selection=(1, 2),
        mode_spec=td.MicrowaveModeSpec(num_modes=3),
    )
    # Test with wave port (no mode_index) - should work
    task_name = TerminalComponentModeler.get_task_name(port=wave_port)
    assert task_name == "wave@1"


def test_validate_port_refinement_with_uniform_grid():
    """Test that port refinement options raise error with uniform grid."""
    uniform_grid = td.GridSpec.uniform(dl=0.5 * mm)
    with AssertLogLevel("WARNING", contains_str="mesh refinement options enabled"):
        make_coaxial_component_modeler(
            port_types=(CoaxialLumpedPort, WavePort), grid_spec=uniform_grid
        )


def test_structure_priority_mode_default():
    """Test that TerminalComponentModeler defaults to conductor priority mode."""
    modeler = make_component_modeler(planar_pec=True)

    # Check that the modeler defaults to "conductor" mode
    assert modeler.structure_priority_mode == "conductor"

    # Check that base_sim uses this priority mode
    assert modeler.base_sim.structure_priority_mode == "conductor"


def test_structure_priority_mode_override():
    """Test that structure_priority_mode can be overridden."""
    # Create modeler with "equal" mode
    modeler = make_component_modeler(planar_pec=True, structure_priority_mode="equal")

    # Check that the override is applied
    assert modeler.structure_priority_mode == "equal"

    # Check that base_sim uses the overridden priority mode
    assert modeler.base_sim.structure_priority_mode == "equal"


# ---------------------------------------------------------------------------
# Renormalization tests
# ---------------------------------------------------------------------------


def _make_renormalize_fixture(monkeypatch):
    """Set up a 2-port modeler with monkeypatched V/I from an analytical transmission line.

    Monkeypatches ``port_voltage_current_matrices`` and ``port_reference_impedances``
    directly on the class so that the fixture survives ``updated_copy`` (which deep-copies
    data and invalidates object-id-based mocks).

    Returns ``(modeler_data, Zref, Z0, gamma, length, freqs, port_names)`` so that
    callers can compute the expected S-matrix for any reference impedance.
    """
    modeler = make_component_modeler(planar_pec=False)
    freqs = np.array([1e9, 5e9, 10e9])
    modeler = modeler.updated_copy(freqs=freqs)

    # Lossy microstrip parameters (engineering convention exp(jwt))
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
            12.843105732941 + 15.394208173652j,
            28.567192048123 + 9.1023847562915j,
            31.209457618234 + 3.8475102934671j,
        ]
    )
    Z01 = np.array(
        [
            18.725191534567 + 12.672421364213j,
            34.038884625562 + 7.8654410284980j,
            35.725175635077 + 4.5490999181327j,
        ]
    )
    Z02 = np.array(
        [
            24.156839210485 + 10.234195827361j,
            41.892301567293 + 6.7812039451120j,
            29.451276384019 + 5.1298475620183j,
        ]
    )
    Zref = np.column_stack((Z01, Z02))
    S_pseudo = calc_transmission_line_S_matrix_pseudo(Z0, Zref[:, 0], Zref[:, 1], gamma, length)
    Zref3 = Zref[:, :, np.newaxis]

    A = np.tile(np.eye(2), (len(freqs), 1, 1))
    B = S_pseudo @ A
    Vscale = np.abs(Zref3) / np.sqrt(np.real(Zref3))
    Iscale = Vscale / Zref3
    voltages = Vscale * (A + B)  # (f, port_out, port_in)
    currents = Iscale * (A - B)  # (f, port_out, port_in)

    port_names = [port.name for port in modeler.ports]

    # Build actual SimulationDataMap (contents don't matter — V/I are mocked)
    sim_data_list = []
    port_name_list = []
    for port_in in modeler.ports:
        task_name = modeler.get_task_name(port_in)
        sim_data_list.append(run_emulated(simulation=modeler.simulation))
        port_name_list.append(task_name)

    index_data = SimulationDataMap(keys=tuple(port_name_list), values=tuple(sim_data_list))
    modeler_data = TerminalComponentModelerData(modeler=modeler, data=index_data)

    # Pre-build the V/I matrices as TerminalPortDataArray
    vi_coords = {"f": np.array(freqs), "port_out": port_names, "port_in": port_names}
    voltage_matrix = TerminalPortDataArray(voltages, coords=vi_coords)
    current_matrix = TerminalPortDataArray(currents, coords=vi_coords)

    # Build the native Zref as TerminalPortDataArray
    num_ports = len(port_names)
    Zref_3d = Zref[:, :, None] * np.eye(num_ports)
    z_ref_native = TerminalPortDataArray(
        Zref_3d, coords={"f": np.array(freqs), "port_out": port_names, "port_in": port_names}
    )

    # Monkeypatch port_voltage_current_matrices as a regular property on the class.
    # This survives updated_copy because the property is on the class, not the instance.
    monkeypatch.setattr(
        TerminalComponentModelerData,
        "port_voltage_current_matrices",
        property(lambda self: (voltage_matrix, current_matrix)),
    )

    # Monkeypatch the module-level port_reference_impedances function used when
    # renormalized_reference_impedance is None or "Z0".
    monkeypatch.setattr(
        tidy3d.plugins.smatrix.analysis.terminal,
        "port_reference_impedances",
        lambda modeler_data_arg: z_ref_native,
    )

    return modeler_data, Zref, Z0, gamma, length, freqs, port_names


def test_renormalize_identity(monkeypatch):
    """Renormalizing to the same reference impedance gives the same S-matrix."""
    modeler_data, _Zref, *_ = _make_renormalize_fixture(monkeypatch)
    freqs, port_names = _[3], _[4]

    s_orig = modeler_data.smatrix().data.values
    z_ref_orig = modeler_data.port_reference_impedances

    s_renorm = modeler_data.renormalize(z_ref_orig).smatrix().data.values
    assert np.allclose(s_orig, s_renorm, rtol=1e-12, atol=1e-14), (
        f"Identity renormalization changed S-matrix.\n"
        f"Max diff: {np.max(np.abs(s_orig - s_renorm)):.2e}"
    )


def test_renormalize_z_matrix_invariance(monkeypatch):
    """Z-matrix should be the same regardless of reference impedance renormalization."""
    modeler_data, _Zref, _Z0, _gamma, _length, _freqs, _port_names = _make_renormalize_fixture(
        monkeypatch
    )

    z_orig = modeler_data.s_to_z().values
    z_renorm = modeler_data.renormalize(50).s_to_z().values

    assert np.allclose(z_orig, z_renorm, rtol=1e-10, atol=1e-12), (
        f"Z-matrix changed after renormalization.\n"
        f"Max diff: {np.max(np.abs(z_orig - z_renorm)):.2e}"
    )


def test_renormalize_scalar(monkeypatch):
    """Renormalize with a scalar impedance produces different S-matrix."""
    modeler_data, *_ = _make_renormalize_fixture(monkeypatch)

    s_orig = modeler_data.smatrix().data.values
    s_50 = modeler_data.renormalize(50).smatrix().data.values

    # S-matrix should change since original Zref is not 50
    assert not np.allclose(s_orig, s_50, atol=1e-6), "Renormalization to 50 did not change S-matrix"

    # Verify it computed the expected S for Z_ref=50
    Zref, Z0, gamma, length, freqs, port_names = _[0], _[1], _[2], _[3], _[4], _[5]
    S_expected = calc_transmission_line_S_matrix_pseudo(Z0, 50, 50, gamma, length)
    assert np.allclose(s_50, S_expected, rtol=1e-10, atol=1e-12), (
        f"Renormalized S-matrix does not match analytical expectation.\n"
        f"Max diff: {np.max(np.abs(s_50 - S_expected)):.2e}"
    )


def test_renormalize_port_data_array(monkeypatch):
    """Renormalize with a per-port PortDataArray impedance."""
    modeler_data, _Zref, Z0, gamma, length, freqs, port_names = _make_renormalize_fixture(
        monkeypatch
    )

    # Use 50 ohm for port 1, 75 ohm for port 2
    per_port_z = PortDataArray(
        np.array([[50, 75]] * len(freqs), dtype=complex),
        coords={"f": freqs, "port": port_names},
    )
    s_renorm = modeler_data.renormalize(per_port_z).smatrix().data.values

    S_expected = calc_transmission_line_S_matrix_pseudo(Z0, 50, 75, gamma, length)
    assert np.allclose(s_renorm, S_expected, rtol=1e-10, atol=1e-12), (
        f"Per-port renormalized S-matrix does not match analytical expectation.\n"
        f"Max diff: {np.max(np.abs(s_renorm - S_expected)):.2e}"
    )


def test_renormalize_terminal_port_data_array(monkeypatch):
    """Renormalize with a full TerminalPortDataArray impedance matrix."""
    modeler_data, _Zref, Z0, gamma, length, freqs, port_names = _make_renormalize_fixture(
        monkeypatch
    )

    # Build a diagonal TerminalPortDataArray with 50 ohm
    num_ports = len(port_names)
    z_values = np.zeros((len(freqs), num_ports, num_ports), dtype=complex)
    for i in range(num_ports):
        z_values[:, i, i] = 50
    z_ref_full = TerminalPortDataArray(
        z_values, coords={"f": freqs, "port_out": port_names, "port_in": port_names}
    )

    s_renorm = modeler_data.renormalize(z_ref_full).smatrix().data.values
    S_expected = calc_transmission_line_S_matrix_pseudo(Z0, 50, 50, gamma, length)
    assert np.allclose(s_renorm, S_expected, rtol=1e-10, atol=1e-12)


def test_renormalize_chaining_s_to_z(monkeypatch):
    """renormalize(50).s_to_z() works and equals original s_to_z()."""
    modeler_data, *_ = _make_renormalize_fixture(monkeypatch)

    z_orig = modeler_data.s_to_z().values
    z_50 = modeler_data.renormalize(50).s_to_z().values

    assert np.allclose(z_orig, z_50, rtol=1e-10, atol=1e-12)


def test_renormalize_invalid_type(monkeypatch):
    """Passing an unsupported type raises ValidationError from Pydantic."""
    modeler_data, *_ = _make_renormalize_fixture(monkeypatch)

    with pytest.raises(ValidationError):
        modeler_data.renormalize("invalid_string")


def test_renormalize_none_is_default(monkeypatch):
    """When renormalized_reference_impedance is None (default), behavior is unchanged."""
    modeler_data, *_ = _make_renormalize_fixture(monkeypatch)

    assert modeler_data.renormalized_reference_impedance is None
    # Should be able to call smatrix normally
    s = modeler_data.smatrix()
    assert s.data.values.shape[1] == 2


def test_wave_port_mode_spec_resolution():
    """Unit test WavePort._mode_spec for three resolution branches."""
    # Shared geometry for WavePort construction
    center = (0, 0, 0)
    size = (2 * Router, 2 * Router, 0)

    # Build a voltage spec that lies within the port bounds
    mean_radius = (Router + Rinner) / 2
    voltage_spec = td.AxisAlignedVoltageIntegralSpec(
        center=(mean_radius, 0, 0),
        size=(Router - Rinner, 0, 0),
        extrapolate_to_endpoints=True,
        snap_path_to_grid=True,
        sign="+",
    )
    custom_impedance = td.CustomImpedanceSpec(voltage_spec=voltage_spec)

    # Case 1: explicit integer num_modes — returns mode_spec as-is
    spec_explicit = td.MicrowaveModeSpec(
        num_modes=1,
        impedance_specs=(custom_impedance,),
    )
    port_explicit = WavePort(
        center=center, size=size, direction="+", name="explicit", mode_spec=spec_explicit
    )
    resolved = port_explicit._mode_spec
    assert resolved is not None
    assert resolved.num_modes == 1

    # Case 2: num_modes="auto" with a tuple of impedance_specs — infer from length
    spec_auto_tuple = td.MicrowaveModeSpec(
        num_modes="auto",
        impedance_specs=(custom_impedance, custom_impedance),
    )
    port_auto_tuple = WavePort(
        center=center, size=size, direction="+", name="auto_tuple", mode_spec=spec_auto_tuple
    )
    resolved_tuple = port_auto_tuple._mode_spec
    assert resolved_tuple is not None
    assert resolved_tuple.num_modes == 2

    # Case 3: num_modes="auto" with AutoImpedanceSpec — cannot resolve, returns None
    spec_auto_default = td.MicrowaveModeSpec(num_modes="auto")
    port_auto_default = WavePort(
        center=center, size=size, direction="+", name="auto_default", mode_spec=spec_auto_default
    )
    assert port_auto_default._mode_spec is None


def _make_wave_port_auto_num_modes_setup():
    """Build a stripline setup whose wave port resolves to a single floating conductor."""
    # Build a single-strip stripline: one signal trace between two ground planes.
    # PEC boundary on y ensures ground planes touching y-boundary are filtered out,
    # leaving only the signal strip as a floating conductor.
    mil = 25.4
    w = 3.2 * mil  # signal strip width
    t = 0.7 * mil  # conductor thickness
    h = 10.7 * mil  # substrate thickness
    L = 4000 * mil  # line length
    len_inf = 1e6
    f_max = 70e9

    med_sub = td.Medium(permittivity=4.4)
    med_metal = td.PEC

    str_sub = td.Structure(geometry=td.Box(center=(0, 0, 0), size=(len_inf, h, L)), medium=med_sub)
    str_signal = td.Structure(geometry=td.Box(center=(0, 0, 0), size=(w, t, L)), medium=med_metal)
    str_gnd_top = td.Structure(
        geometry=td.Box(center=(0, h / 2 + t / 2, 0), size=(len_inf, t, L)),
        medium=med_metal,
    )
    str_gnd_bot = td.Structure(
        geometry=td.Box(center=(0, -h / 2 - t / 2, 0), size=(len_inf, t, L)),
        medium=med_metal,
    )

    lr_spec = td.LayerRefinementSpec.from_structures(
        structures=[str_signal],
        axis=1,
        min_steps_along_axis=10,
        refinement_inside_sim_only=False,
        bounds_snapping="bounds",
        corner_refinement=td.GridRefinement(dl=t / 10, num_cells=2),
    )
    lr_spec_top = lr_spec.updated_copy(center=(0, h / 2 + t / 2, 0), size=(len_inf, t, L))
    lr_spec_bot = lr_spec.updated_copy(center=(0, -h / 2 - t / 2, 0), size=(len_inf, t, L))

    grid_spec = td.GridSpec.auto(
        wavelength=td.C_0 / f_max,
        min_steps_per_wvl=30,
        layer_refinement_specs=[lr_spec, lr_spec_top, lr_spec_bot],
    )

    sim = td.Simulation(
        size=(50 * mil, h + 2 * t, 1.05 * L),
        center=(0, 0, 0),
        grid_spec=grid_spec,
        boundary_spec=td.BoundarySpec(
            x=td.Boundary.pml(), y=td.Boundary.pec(), z=td.Boundary.pml()
        ),
        structures=[str_sub, str_signal, str_gnd_top, str_gnd_bot],
        monitors=[],
        run_time=2e-9,
        shutoff=1e-7,
    )

    port = WavePort(
        center=(0, 0, -L / 2),
        size=(len_inf, len_inf, 0),
        direction="+",
        name="wave_auto",
        mode_spec=td.MicrowaveModeSpec(num_modes="auto"),
    )

    freqs = np.array([1e9, 10e9])
    return sim, port, freqs


def test_wave_port_auto_num_modes_modeler():
    """Integration test: auto num_modes detection for WavePort via TerminalComponentModeler."""
    sim, port, freqs = _make_wave_port_auto_num_modes_setup()
    modeler = TerminalComponentModeler(simulation=sim, ports=[port], freqs=freqs)

    # Modeler should identify 1 wave port
    assert len(modeler._wave_ports) == 1

    # Should detect 1 floating isolated conductor (signal strip;
    # ground planes filtered because they touch the PEC boundary on y)
    conductors = modeler._floating_isolated_conductors_at_waveport["wave_auto"]
    assert len(conductors) == 1

    # Resolved mode spec should have num_modes == 1
    resolved = modeler._resolved_mode_specs["wave_auto"]
    assert isinstance(resolved, td.MicrowaveModeSpec)
    assert resolved.num_modes == 1

    # mode_solver_for_port should return a ModeSolver
    ms = modeler.mode_solver_for_port("wave_auto")
    assert isinstance(ms, ModeSolver)


def test_wave_port_auto_num_modes_modeler_validates_mode_selection_bounds():
    """Modeler construction should validate deferred mode-selection bounds for auto mode specs."""
    sim, port, freqs = _make_wave_port_auto_num_modes_setup()
    port = port.updated_copy(mode_selection=(1,))

    with pytest.raises(
        ValidationError,
        match=r"'mode_spec.mode_selection' contains indices \[1\].*for port 'wave_auto'",
    ):
        TerminalComponentModeler(simulation=sim, ports=[port], freqs=freqs)


def test_wave_port_auto_num_modes_modeler_validates_mode_index_bounds():
    """Modeler construction should validate deferred mode-index bounds for auto mode specs."""
    sim, port, freqs = _make_wave_port_auto_num_modes_setup()
    port = port.updated_copy(mode_index=1)

    with pytest.raises(ValidationError, match=r"'mode_index' is >= .*for port 'wave_auto'"):
        TerminalComponentModeler(simulation=sim, ports=[port], freqs=freqs)


def test_wave_port_auto_num_modes_to_mode_simulation():
    """Standalone mode-simulation creation should resolve ``num_modes='auto'``."""

    sim, port, freqs = _make_wave_port_auto_num_modes_setup()

    mode_sim = port.to_mode_simulation(sim, freqs=freqs, structure_priority_mode="conductor")

    assert isinstance(mode_sim, ModeSimulation)
    assert isinstance(mode_sim.mode_spec, td.MicrowaveModeSpec)
    assert mode_sim.mode_spec.num_modes == 1


def test_wave_port_auto_num_modes_to_mode_solver():
    """Standalone mode-solver creation should resolve ``num_modes='auto'``."""
    sim, port, freqs = _make_wave_port_auto_num_modes_setup()

    mode_solver = port.to_mode_solver(sim, freqs=freqs, structure_priority_mode="conductor")

    assert isinstance(mode_solver, ModeSolver)
    assert isinstance(mode_solver.mode_spec, td.MicrowaveModeSpec)
    assert mode_solver.mode_spec.num_modes == 1


def test_wave_port_auto_num_modes_to_mode_simulation_validates_mode_selection_bounds():
    """Deferred mode-selection bounds should be checked after auto mode-spec resolution."""
    sim, port, freqs = _make_wave_port_auto_num_modes_setup()
    port = port.updated_copy(mode_selection=(1,))

    with pytest.raises(Tidy3dValidationError, match=r"'mode_selection' contains indices \[1\]"):
        port.to_mode_simulation(sim, freqs=freqs)


def test_wave_port_auto_num_modes_to_mode_simulation_validates_mode_index_bounds():
    """Deferred mode-index bounds should be checked after auto mode-spec resolution."""
    sim, port, freqs = _make_wave_port_auto_num_modes_setup()
    port = port.updated_copy(mode_index=1)

    with pytest.raises(Tidy3dValidationError, match=r"'mode_index' is >="):
        port.to_mode_simulation(sim, freqs=freqs)


def test_compute_F_full_matrix():
    """Test compute_F with a non-diagonal (coupled) impedance matrix."""
    # 2-port coupled impedance: non-diagonal → exercises _compute_F_full
    Z = np.array([[[50 + 5j, 3 + 1j], [3 + 1j, 75 + 8j]]])  # shape (1, 2, 2)

    for s_def in ("power", "pseudo", "symmetric_pseudo"):
        F = compute_F(Z, s_def, compute_Finv=False)
        assert F.shape == (1, 2, 2)
        assert np.all(np.isfinite(F))

        F, Finv = compute_F(Z, s_def, compute_Finv=True)
        assert F.shape == Finv.shape == (1, 2, 2)
        # F @ Finv ≈ I (within numerical tolerance)
        product = F[0] @ Finv[0]
        assert np.allclose(product, np.eye(2), atol=1e-10)


def test_wave_port_validate_resolved_mode_spec_errors():
    """Test _validate_resolved_mode_spec error paths for WavePort."""
    # num_modes='auto' passed explicitly → SetupError
    auto_mode_spec = td.MicrowaveModeSpec(num_modes="auto")
    port = WavePort(
        center=(0, 0, -10),
        size=(2, 2, 0),
        direction="+",
        name="wp_auto",
        mode_spec=auto_mode_spec,
    )
    with pytest.raises(SetupError, match=r"num_modes='auto'"):
        port._validate_resolved_mode_spec(auto_mode_spec)

    # _mode_spec is None and no explicit mode_spec → SetupError
    # This happens when mode_spec.num_modes='auto' and impedance_specs is also 'auto'
    assert port._mode_spec is None
    with pytest.raises(SetupError, match=r"cannot be resolved"):
        port._validate_resolved_mode_spec()


def test_wave_port_get_port_impedance():
    """Test get_port_impedance returns correct impedance for each mode."""
    mode_spec = td.MicrowaveModeSpec(
        num_modes=2,
        impedance_specs=(
            td.CustomImpedanceSpec(
                voltage_spec=td.AxisAlignedVoltageIntegralSpec(
                    center=(0, 0, -10),
                    size=(1, 0, 0),
                    extrapolate_to_endpoints=True,
                    sign="+",
                )
            ),
            td.CustomImpedanceSpec(
                voltage_spec=td.AxisAlignedVoltageIntegralSpec(
                    center=(0.5, 0, -10),
                    size=(0.5, 0, 0),
                    extrapolate_to_endpoints=True,
                    sign="+",
                )
            ),
        ),
    )
    port = WavePort(
        center=(0, 0, -10),
        size=(2, 2, 0),
        direction="+",
        name="wp_impedance",
        mode_spec=mode_spec,
    )

    freqs = [10e9]
    mode_indices = [0, 1]
    index_coords = {"f": freqs, "mode_index": mode_indices}

    # Build TransmissionLineDataset with Z0=[50, 75]
    z0_data = ImpedanceFreqModeDataArray(np.array([[50.0, 75.0]]), coords=index_coords)
    v_data = VoltageFreqModeDataArray((1 + 0j) * np.ones((1, 2)), coords=index_coords)
    c_data = CurrentFreqModeDataArray((0.02 + 0j) * np.ones((1, 2)), coords=index_coords)
    tl_data = TransmissionLineDataset(Z0=z0_data, voltage_coeffs=v_data, current_coeffs=c_data)

    # Build MicrowaveModeData
    n_complex = ModeIndexDataArray((1.5 + 0.01j) * np.ones((1, 2)), coords=index_coords)
    amp_coords = {"direction": ["+", "-"], "f": freqs, "mode_index": mode_indices}
    amps = ModeAmpsDataArray((1 + 0.5j) * np.ones((2, 1, 2)), coords=amp_coords)
    monitor = td.MicrowaveModeMonitor(
        center=port.center,
        size=port.size,
        freqs=freqs,
        mode_spec=mode_spec,
        name=port._mode_monitor_name,
    )
    mode_data = MicrowaveModeData(
        monitor=monitor,
        amps=amps,
        n_complex=n_complex,
        transmission_line_data=tl_data,
    )

    z_mode0 = port.get_port_impedance(mode_data, mode_index=0)
    assert np.isclose(z_mode0.values.item(), 50.0)

    z_mode1 = port.get_port_impedance(mode_data, mode_index=1)
    assert np.isclose(z_mode1.values.item(), 75.0)


def test_wave_port_to_mode_simulation():
    """Test that to_mode_simulation creates a valid ModeSimulation."""

    modeler = make_coaxial_component_modeler(port_types=(WavePort, WavePort))
    port = modeler.ports[0]
    sim = modeler.simulation
    freqs = [1e9, 2e9, 3e9]

    mode_sim = port.to_mode_simulation(sim, freqs=freqs, reduce_simulation=True)

    assert isinstance(mode_sim, ModeSimulation)
    assert mode_sim.mode_spec == port.mode_spec
    assert mode_sim.direction == port.direction
    assert mode_sim.colocate is False
    assert mode_sim.conjugated_dot_product == port.conjugated_dot_product
    # Plane matches port geometry
    assert mode_sim.plane.center == port.center
    assert mode_sim.plane.size == port.size

    localized_sim = sim.subsection(
        region=port._mode_filter_box,
        sources=(),
        monitors=(),
        internal_absorbers=(),
        remove_outside_grid_spec=True,
        warn_symmetry_expansion=False,
        deep_copy=False,
        validate_geometries=False,
    )
    filtered_grid_spec = localized_sim.grid_spec
    filtered_grid_spec = filtered_grid_spec.updated_copy(
        override_structures=[*filtered_grid_spec.override_structures, *port.to_mesh_overrides()]
    )
    filtered_structures = localized_sim.structures

    assert len(mode_sim.grid_spec.override_structures) == len(
        filtered_grid_spec.override_structures
    )
    assert len(mode_sim.grid_spec.layer_refinement_specs) == len(
        filtered_grid_spec.layer_refinement_specs
    )
    assert len(mode_sim.grid_spec.snapping_points) == len(filtered_grid_spec.snapping_points)
    assert len(mode_sim.structures) == len(filtered_structures)
    assert not any(
        structure.name and structure.name.endswith(f"_extruded_{port.name}")
        for structure in mode_sim.structures
    )


def test_wave_port_to_mode_simulation_includes_single_port_extrusions():
    """WavePort mode simulations should include the same named extrusions for the active port."""

    modeler = make_coaxial_component_modeler(length=100000, port_types=(WavePort, WavePort))
    named_structures = [
        structure.updated_copy(name=f"structure_{index}")
        for index, structure in enumerate(modeler.simulation.structures)
    ]
    simulation = modeler.simulation.updated_copy(structures=named_structures)
    port = modeler.ports[0].updated_copy(center=(0, 0, -50000), extrude_structures=True)
    modeler = modeler.updated_copy(simulation=simulation, ports=(port, modeler.ports[1]))

    mode_sim = port.to_mode_simulation(
        simulation,
        freqs=modeler.freqs,
        mode_spec=modeler._resolved_mode_specs[port.name],
    )
    base_sim = modeler.base_sim

    mode_extruded = [
        structure
        for structure in mode_sim.structures
        if structure.name and structure.name.endswith(f"_extruded_{port.name}")
    ]
    base_extruded = [
        structure
        for structure in base_sim.structures
        if structure.name and structure.name.endswith(f"_extruded_{port.name}")
    ]

    assert len(mode_extruded) == len(base_extruded) > 0
    mode_bounds = np.array(
        [
            tuple(coord for endpoint in struct.geometry.bounds for coord in endpoint)
            for struct in mode_extruded
        ]
    )
    base_bounds = np.array(
        [
            tuple(coord for endpoint in struct.geometry.bounds for coord in endpoint)
            for struct in base_extruded
        ]
    )
    assert np.allclose(mode_bounds, base_bounds)


def _make_z_normal_wave_port():
    """Helper: z-normal WavePort at (0, 0, 5) with size (4, 6, 0)."""
    return WavePort(
        center=(0, 0, 5),
        size=(4, 6, 0),
        direction="+",
        name="test_port",
        num_grid_cells=10,
    )


def _localize_grid_spec_to_port(port, grid_spec):
    return grid_spec._localized_copy(region=port._mode_filter_box)


def test_wave_port_filter_override_keeps_axis_relevant_mesh_refinement():
    """Mesh overrides are kept when they affect a transverse axis inside the port region."""
    port = _make_z_normal_wave_port()
    override_y_only = td.MeshOverrideStructure(
        geometry=td.Box(center=(10, 0, 5), size=(1, 1, 1)),
        dl=(None, 0.1, 0.1),
    )
    result = _localize_grid_spec_to_port(
        port, td.GridSpec.auto(wavelength=1.0, override_structures=(override_y_only,))
    ).override_structures
    assert len(result) == 1
    assert result[0].dl == (None, 0.1, 0.1)

    override_y_far = td.MeshOverrideStructure(
        geometry=td.Box(center=(0, 10, 5), size=(1, 1, 1)),
        dl=(0.1, None, 0.1),
    )
    result = _localize_grid_spec_to_port(
        port, td.GridSpec.auto(wavelength=1.0, override_structures=(override_y_far,))
    ).override_structures
    assert len(result) == 1
    assert result[0].dl == (0.1, None, 0.1)

    override_far = td.MeshOverrideStructure(
        geometry=td.Box(center=(10, 10, 5), size=(1, 1, 1)),
        dl=(0.1, 0.1, 0.1),
    )
    result = _localize_grid_spec_to_port(
        port, td.GridSpec.auto(wavelength=1.0, override_structures=(override_far,))
    ).override_structures
    assert len(result) == 1
    assert result[0].dl == (None, None, 0.1)


def test_wave_port_filter_override_preserves_geometry():
    """Intersecting mesh overrides keep their geometry and only localize mesh hints."""
    port = _make_z_normal_wave_port()
    override = td.MeshOverrideStructure(
        geometry=td.Box(center=(2.75, 0, 5), size=(1.5, 1.0, 1.0)),
        dl=(0.1, 0.1, 0.1),
    )

    result = _localize_grid_spec_to_port(
        port, td.GridSpec.auto(wavelength=1.0, override_structures=(override,))
    ).override_structures
    assert len(result) == 1

    assert result[0].dl == (0.1, 0.1, 0.1)
    assert result[0].geometry == override.geometry


def test_wave_port_filter_override_keeps_near_edge_context():
    """The buffered refinement box retains overrides slightly outside the port footprint."""
    port = _make_z_normal_wave_port()
    override = td.MeshOverrideStructure(
        geometry=td.Box(center=(2.35, 0, 5), size=(0.4, 1.0, 1.0)),
        dl=(0.1, 0.1, 0.1),
    )

    result = _localize_grid_spec_to_port(
        port, td.GridSpec.auto(wavelength=1.0, override_structures=(override,))
    ).override_structures
    assert len(result) == 1

    preserved = result[0].geometry
    ref_bounds = port._mode_filter_box.bounds
    assert port.bounds[1][0] < preserved.bounds[0][0]
    assert np.allclose(preserved.bounds, override.geometry.bounds)
    assert preserved.bounds[1][0] > ref_bounds[1][0]


def test_wave_port_filter_structure_override_prunes_grouped_geometry():
    """Non-mesh structure overrides use recursive geometry pruning."""
    port = _make_z_normal_wave_port()
    geometry_in = td.Box(center=(0, 0, 5), size=(1, 1, 1))
    geometry_out = td.Box(center=(8, 0, 5), size=(1, 1, 1))
    override = td.Structure(
        geometry=td.GeometryGroup(geometries=(geometry_in, geometry_out)),
        medium=td.Medium(permittivity=2.0),
    )

    result = _localize_grid_spec_to_port(
        port, td.GridSpec.auto(wavelength=1.0, override_structures=(override,))
    ).override_structures
    assert len(result) == 1
    assert result[0].geometry == geometry_in


def test_wave_port_filter_layer_refinement_specs_clips_to_mode_filter_box():
    """Intersecting layer refinement specs are clipped to the refinement box."""
    port = _make_z_normal_wave_port()
    spec = td.LayerRefinementSpec.from_bounds(axis=2, rmin=(2.0, -1.0, 4.0), rmax=(4.0, 1.0, 6.0))

    result = _localize_grid_spec_to_port(
        port, td.GridSpec.auto(wavelength=1.0, layer_refinement_specs=(spec,))
    ).layer_refinement_specs
    assert len(result) == 1

    clipped = result[0]
    ref_bounds = port._mode_filter_box.bounds
    assert clipped.bounds[0][0] == pytest.approx(spec.bounds[0][0])
    assert clipped.bounds[1][0] == pytest.approx(ref_bounds[1][0])
    assert clipped.bounds[0][1:] == pytest.approx(spec.bounds[0][1:])
    assert clipped.bounds[1][1:] == pytest.approx(spec.bounds[1][1:])


def test_wave_port_filter_snapping_points_keeps_axis_relevant_coords():
    """Snapping points keep coordinates on axes that remain within the local mode region."""
    port = _make_z_normal_wave_port()
    points = (
        (0.0, 0.0, 5.0),
        (10.0, 0.0, 5.0),
        (0.0, 10.0, 5.0),
        (10.0, 10.0, 5.0),
        (None, None, 5.0),
    )

    result = _localize_grid_spec_to_port(
        port, td.GridSpec.auto(wavelength=1.0, snapping_points=points)
    ).snapping_points
    assert result == (
        (0.0, 0.0, 5.0),
        (None, 0.0, 5.0),
        (0.0, None, 5.0),
        (None, None, 5.0),
        (None, None, 5.0),
    )


def _shift_vertices(vertices, dx):
    return [(x + dx, y) for x, y in vertices]


def _boundaries_in_bounds(boundaries, lower, upper, atol=1e-12):
    boundaries = np.asarray(boundaries)
    return boundaries[(boundaries >= lower - atol) & (boundaries <= upper + atol)]


def _effective_override_count(simulation):
    return len(simulation.grid_spec.override_structures) + len(
        simulation.internal_override_structures
    )


def _effective_snapping_count(simulation):
    return len(simulation.grid_spec.snapping_points) + len(simulation.internal_snapping_points)


def _make_wave_port_filtering_cpw_simulation():
    trace_centers_x = (-4.0, 0.0, 4.0)
    trace_length_y = 12.0
    substrate_thickness = 0.6
    trace_thickness = 0.05
    ground_thickness = 0.06
    sim_size = (14.0, 14.0, 1.4)
    trace_zmin = substrate_thickness / 2
    trace_zmax = trace_zmin + trace_thickness
    trace_zmid = (trace_zmin + trace_zmax) / 2
    ground_zmid = -(substrate_thickness + ground_thickness) / 2

    trace_vertices = [
        (-0.35, -trace_length_y / 2),
        (0.35, -trace_length_y / 2),
        (0.35, 1.5),
        (0.5, 1.5),
        (0.5, 2.5),
        (0.35, 2.5),
        (0.35, trace_length_y / 2),
        (-0.35, trace_length_y / 2),
        (-0.35, 2.5),
        (-0.5, 2.5),
        (-0.5, 1.5),
        (-0.35, 1.5),
    ]

    signal_structures = [
        td.Structure(
            geometry=td.PolySlab(
                vertices=_shift_vertices(trace_vertices, dx=center_x),
                slab_bounds=(trace_zmin, trace_zmax),
                axis=2,
            ),
            medium=td.PECMedium(),
            name=f"trace_{ind}",
        )
        for ind, center_x in enumerate(trace_centers_x)
    ]

    substrate = td.Structure(
        geometry=td.Box(center=(0, 0, 0), size=(td.inf, sim_size[1], substrate_thickness)),
        medium=td.Medium(permittivity=4.1),
        name="substrate",
    )
    ground = td.Structure(
        geometry=td.Box(center=(0, 0, ground_zmid), size=(td.inf, sim_size[1], ground_thickness)),
        medium=td.PECMedium(),
        name="ground",
    )

    override_structures = [
        td.MeshOverrideStructure(
            geometry=td.Box(
                center=(center_x, 0, trace_zmid),
                size=(1.1, trace_length_y, trace_thickness),
            ),
            dl=(0.04, None, None),
        )
        for center_x in trace_centers_x
    ]
    snapping_points = [(center_x, 0.0, None) for center_x in trace_centers_x]

    layer_refinement = td.LayerRefinementSpec.from_structures(
        structures=signal_structures,
        axis=2,
        bounds_snapping="bounds",
        bounds_refinement=td.GridRefinement(dl=trace_thickness / 2, num_cells=2),
        corner_refinement=td.GridRefinement(dl=0.08, num_cells=2),
    )

    simulation = td.Simulation(
        center=(0, 0, 0.05),
        size=sim_size,
        structures=[substrate, ground, *signal_structures],
        grid_spec=td.GridSpec.auto(
            wavelength=4.0,
            min_steps_per_wvl=12,
            override_structures=override_structures,
            layer_refinement_specs=[layer_refinement],
            snapping_points=snapping_points,
        ),
        sources=[],
        monitors=[],
        run_time=1e-12,
        boundary_spec=td.BoundarySpec.all_sides(boundary=td.PML()),
    )

    port = WavePort(
        center=(0, 0, 0.02),
        size=(2.0, 0.0, 1.0),
        direction="+",
        name="cpw_port",
        num_grid_cells=10,
        mode_spec=td.MicrowaveModeSpec(num_modes=1, target_neff=np.sqrt(4.1)),
    )

    return simulation, port


def _simulation_with_port_mesh_overrides(simulation, port):
    grid_spec = simulation.grid_spec.updated_copy(
        override_structures=list(simulation.grid_spec.override_structures)
        + port.to_mesh_overrides()
    )
    return simulation.updated_copy(grid_spec=grid_spec)


def _make_wave_port_bare_simulation(simulation, *, structures=None, grid_spec=None):
    if structures is None:
        structures = simulation.structures
    if grid_spec is None:
        grid_spec = simulation.grid_spec
    return simulation.updated_copy(
        structures=structures,
        grid_spec=grid_spec,
        validate=False,
        deep=False,
    )


def _wave_port_localized_scene(simulation, port):
    return simulation.subsection(
        region=port._mode_filter_box,
        sources=(),
        monitors=(),
        internal_absorbers=(),
        remove_outside_grid_spec=True,
        warn_symmetry_expansion=False,
        deep_copy=False,
        validate_geometries=False,
    )


def test_wave_port_unfiltered_mode_simulation_matches_augmented_full_grid():
    """An unfiltered mode simulation should exactly reproduce the augmented full grid."""

    simulation, port = _make_wave_port_filtering_cpw_simulation()
    reference_sim = _simulation_with_port_mesh_overrides(simulation, port)
    mode_sim = ModeSimulation.from_simulation(
        simulation=reference_sim,
        plane=port.geometry,
        mode_spec=port.mode_spec,
        freqs=[20e9],
        direction=port.direction,
        colocate=False,
        conjugated_dot_product=port.conjugated_dot_product,
    )

    x_bounds = (port.bounds[0][0], port.bounds[1][0])
    z_bounds = (port.bounds[0][2], port.bounds[1][2])

    full_x = _boundaries_in_bounds(reference_sim.grid.boundaries.to_list[0], *x_bounds)
    full_z = _boundaries_in_bounds(reference_sim.grid.boundaries.to_list[2], *z_bounds)
    mode_x = _boundaries_in_bounds(mode_sim.grid.boundaries.to_list[0], *x_bounds)
    mode_z = _boundaries_in_bounds(mode_sim.grid.boundaries.to_list[2], *z_bounds)

    assert np.allclose(full_x, mode_x)
    assert np.allclose(full_z, mode_z)
    assert len(reference_sim.grid_spec.override_structures) == len(
        mode_sim.grid_spec.override_structures
    )
    assert len(reference_sim.grid_spec.layer_refinement_specs) == len(
        mode_sim.grid_spec.layer_refinement_specs
    )
    assert len(reference_sim.grid_spec.snapping_points) == len(mode_sim.grid_spec.snapping_points)


def test_wave_port_to_mode_simulation_freezes_grid_when_extrusions_are_added():
    """WavePort mode simulations should freeze meshing inputs once port extrusions are added."""
    simulation, port = _make_wave_port_filtering_cpw_simulation()
    port = port.updated_copy(extrude_structures=True)
    mode_sim = port.to_mode_simulation(simulation, freqs=[20e9], reduce_simulation=True)

    localized_sim = _wave_port_localized_scene(simulation, port)
    filtered_grid_spec = localized_sim.grid_spec
    filtered_grid_spec = filtered_grid_spec.updated_copy(
        override_structures=[*filtered_grid_spec.override_structures, *port.to_mesh_overrides()]
    )
    filtered_structures = localized_sim.structures
    filtered_sim = simulation.updated_copy(
        grid_spec=filtered_grid_spec,
        structures=filtered_structures,
        validate=False,
        deep=False,
    )

    extra_structures = port._extruded_structures(
        simulation=filtered_sim,
    )

    assert len(extra_structures) > 0
    assert len(mode_sim.grid_spec.override_structures) == 0
    assert len(mode_sim.grid_spec.layer_refinement_specs) == 0
    assert len(mode_sim.grid_spec.snapping_points) == 0
    assert _effective_override_count(mode_sim) == 0
    assert _effective_snapping_count(mode_sim) == 0

    assert len(mode_sim.structures) == len(filtered_structures) + len(extra_structures)

    # The frozen mode grid should match the filtered local grid in the port region.
    x_bounds = (port.bounds[0][0], port.bounds[1][0])
    z_bounds = (port.bounds[0][2], port.bounds[1][2])
    filtered_x = _boundaries_in_bounds(filtered_sim.grid.boundaries.to_list[0], *x_bounds)
    filtered_z = _boundaries_in_bounds(filtered_sim.grid.boundaries.to_list[2], *z_bounds)
    mode_x = _boundaries_in_bounds(mode_sim.grid.boundaries.to_list[0], *x_bounds)
    mode_z = _boundaries_in_bounds(mode_sim.grid.boundaries.to_list[2], *z_bounds)
    assert np.allclose(filtered_x, mode_x)
    assert np.allclose(filtered_z, mode_z)


def test_wave_port_to_mode_simulation_keeps_transverse_axis_grid_hints():
    """Filtered mode simulations keep overrides and snapping points that affect a local transverse axis."""
    simulation, port = _make_wave_port_filtering_cpw_simulation()
    zmin, zmax = port.bounds[0][2], port.bounds[1][2]
    override = td.MeshOverrideStructure(
        geometry=td.Box(center=(port.bounds[1][0] + 0.4, 0, port.center[2]), size=(0.2, 2.0, 0.3)),
        dl=(0.03, None, 0.03),
    )
    snap_z = zmin + 0.2 * (zmax - zmin)
    augmented_sim = simulation.updated_copy(
        grid_spec=simulation.grid_spec.updated_copy(
            override_structures=[*simulation.grid_spec.override_structures, override],
            snapping_points=[
                *simulation.grid_spec.snapping_points,
                (port.bounds[1][0] + 0.6, 0, snap_z),
            ],
        ),
        validate=False,
        deep=False,
    )

    mode_sim = port.to_mode_simulation(augmented_sim, freqs=[20e9], reduce_simulation=True)

    kept_override = next(
        mesh_override
        for mesh_override in mode_sim.grid_spec.override_structures
        if isinstance(mesh_override, td.MeshOverrideStructure)
        and mesh_override.dl == (None, None, 0.03)
    )
    assert kept_override.dl == (None, None, 0.03)
    assert kept_override.geometry.bounds[0][0] == pytest.approx(override.geometry.bounds[0][0])
    assert kept_override.geometry.bounds[1][0] == pytest.approx(override.geometry.bounds[1][0])
    assert mode_sim.grid_spec.snapping_points[-1] == (None, 0, snap_z)


def test_wave_port_to_mode_simulation_accepts_structure_and_grid_overrides():
    """Override structures and grid spec can be supplied separately from a bare simulation."""
    simulation, port = _make_wave_port_filtering_cpw_simulation()
    freqs = [20e9]
    bare_sim = _make_wave_port_bare_simulation(
        simulation,
        structures=(),
        grid_spec=td.GridSpec.auto(wavelength=simulation.grid_spec.wavelength),
    )

    mode_sim = port.to_mode_simulation(
        bare_sim,
        freqs=freqs,
        structures=simulation.structures,
        grid_spec=simulation.grid_spec,
        reduce_simulation=True,
    )

    assert mode_sim == port.to_mode_simulation(simulation, freqs=freqs, reduce_simulation=True)


def test_wave_port_to_mode_simulation_accepts_structure_override_only():
    """Structure overrides can reuse the passed simulation grid spec."""
    simulation, port = _make_wave_port_filtering_cpw_simulation()
    freqs = [20e9]
    bare_sim = _make_wave_port_bare_simulation(simulation, structures=())

    mode_sim = port.to_mode_simulation(
        bare_sim,
        freqs=freqs,
        structures=simulation.structures,
        reduce_simulation=True,
    )

    assert mode_sim == port.to_mode_simulation(simulation, freqs=freqs, reduce_simulation=True)


def test_wave_port_to_mode_simulation_accepts_grid_override_only():
    """Grid-spec overrides can reuse the passed simulation structures."""
    simulation, port = _make_wave_port_filtering_cpw_simulation()
    freqs = [20e9]
    bare_sim = _make_wave_port_bare_simulation(
        simulation,
        grid_spec=td.GridSpec.auto(wavelength=simulation.grid_spec.wavelength),
    )

    mode_sim = port.to_mode_simulation(
        bare_sim,
        freqs=freqs,
        grid_spec=simulation.grid_spec,
        reduce_simulation=True,
    )

    assert mode_sim == port.to_mode_simulation(simulation, freqs=freqs, reduce_simulation=True)


def test_wave_port_to_mode_simulation_override_path_preserves_extrusions():
    """The override path should preserve the existing extrusion and frozen-grid behavior."""
    simulation, port = _make_wave_port_filtering_cpw_simulation()
    port = port.updated_copy(extrude_structures=True)
    freqs = [20e9]
    bare_sim = _make_wave_port_bare_simulation(
        simulation,
        structures=(),
        grid_spec=td.GridSpec.auto(wavelength=simulation.grid_spec.wavelength),
    )

    mode_sim = port.to_mode_simulation(
        bare_sim,
        freqs=freqs,
        structures=simulation.structures,
        grid_spec=simulation.grid_spec,
        reduce_simulation=True,
    )

    assert mode_sim == port.to_mode_simulation(simulation, freqs=freqs, reduce_simulation=True)


def test_wave_port_to_mode_solver_uses_mode_simulation_override_path():
    """Mode-solver conversion should delegate through the mode-simulation helper."""
    simulation, port = _make_wave_port_filtering_cpw_simulation()
    freqs = [20e9]
    bare_sim = _make_wave_port_bare_simulation(
        simulation,
        structures=(),
        grid_spec=td.GridSpec.auto(wavelength=simulation.grid_spec.wavelength),
    )

    mode_sim = port.to_mode_simulation(
        bare_sim,
        freqs=freqs,
        structures=simulation.structures,
        grid_spec=simulation.grid_spec,
        reduce_simulation=True,
    )
    mode_solver = port.to_mode_solver(
        bare_sim,
        freqs=freqs,
        structures=simulation.structures,
        grid_spec=simulation.grid_spec,
        reduce_simulation=True,
    )

    assert mode_solver == mode_sim._mode_solver
