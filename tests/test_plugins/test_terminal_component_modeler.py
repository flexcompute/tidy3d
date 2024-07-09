import matplotlib.pyplot as plt
import numpy as np
import pydantic.v1 as pydantic
import pytest
import tidy3d as td
from tidy3d.components.data.data_array import FreqDataArray
from tidy3d.exceptions import SetupError, Tidy3dKeyError
from tidy3d.plugins.smatrix import (
    AbstractComponentModeler,
    CoaxialLumpedPort,
    LumpedPort,
    LumpedPortDataArray,
    TerminalComponentModeler,
)
from tidy3d.plugins.smatrix.ports.base_lumped import AbstractLumpedPort

from ..utils import run_emulated
from .terminal_component_modeler_def import make_coaxial_component_modeler, make_component_modeler


def run_component_modeler(monkeypatch, modeler: TerminalComponentModeler):
    sim_dict = modeler.sim_dict
    batch_data = {task_name: run_emulated(sim) for task_name, sim in sim_dict.items()}
    monkeypatch.setattr(TerminalComponentModeler, "_run_sims", lambda self, path_dir: batch_data)
    # for the random data, the power wave matrix might be singular, leading to an error
    # during inversion, so monkeypatch the inv method so that it operates on a dummy
    # identity matrix
    monkeypatch.setattr(AbstractComponentModeler, "inv", lambda matrix: np.eye(len(modeler.ports)))
    s_matrix = modeler.run(path_dir=modeler.path_dir)
    return s_matrix


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


def test_validate_no_sources(tmp_path):
    modeler = make_component_modeler(planar_pec=True, path_dir=str(tmp_path))
    source = td.PointDipole(
        source_time=td.GaussianPulse(freq0=2e14, fwidth=1e14), polarization="Ex"
    )
    sim_w_source = modeler.simulation.copy(update=dict(sources=(source,)))
    with pytest.raises(pydantic.ValidationError):
        _ = modeler.copy(update=dict(simulation=sim_w_source))


def test_no_port(tmp_path):
    modeler = make_component_modeler(planar_pec=True, path_dir=str(tmp_path))
    _ = modeler.ports
    with pytest.raises(Tidy3dKeyError):
        modeler.get_port_by_name(port_name="NOT_A_PORT")


def test_plot_sim(tmp_path):
    modeler = make_component_modeler(planar_pec=False, path_dir=str(tmp_path))
    modeler.plot_sim(z=0)
    plt.close()


def test_plot_sim_eps(tmp_path):
    modeler = make_component_modeler(planar_pec=False, path_dir=str(tmp_path))
    modeler.plot_sim_eps(z=0)
    plt.close()


@pytest.mark.parametrize("port_refinement", [False, True])
def test_make_component_modeler(tmp_path, port_refinement):
    _ = make_component_modeler(
        planar_pec=False, path_dir=str(tmp_path), port_refinement=port_refinement
    )


def test_run(monkeypatch, tmp_path):
    modeler = make_component_modeler(planar_pec=True, path_dir=str(tmp_path))
    monkeypatch.setattr(TerminalComponentModeler, "run", lambda self, path_dir: None)
    modeler.run(path_dir=str(tmp_path))


def test_run_component_modeler(monkeypatch, tmp_path):
    modeler = make_component_modeler(planar_pec=True, path_dir=str(tmp_path))
    s_matrix = run_component_modeler(monkeypatch, modeler)

    for port_in in modeler.ports:
        for port_out in modeler.ports:
            coords_in = dict(port_in=port_in.name)
            coords_out = dict(port_out=port_out.name)

            assert np.all(s_matrix.sel(**coords_in) != 0), "source index not present in S matrix"
            assert np.all(
                s_matrix.sel(**coords_in).sel(**coords_out) != 0
            ), "monitor index not present in S matrix"


def test_s_to_z_component_modeler():
    # Test case is 2 port T network with reference impedance of 50 Ohm
    A = 20 + 30j
    B = 50 - 15j
    C = 60

    Z11 = A + C
    Z21 = C
    Z12 = C
    Z22 = B + C

    Z0 = 50.0
    # Manual creation of S parameters Pozar Table 4.2
    deltaZ = (Z11 + Z0) * (Z22 + Z0) - Z12 * Z21
    S11 = ((Z11 - Z0) * (Z22 + Z0) - Z12 * Z21) / deltaZ
    S12 = (2 * Z12 * Z0) / deltaZ
    S21 = (2 * Z21 * Z0) / deltaZ
    S22 = ((Z11 + Z0) * (Z22 - Z0) - Z12 * Z21) / deltaZ

    port_names = ["lumped_port_1", "lumped_port_2"]
    freqs = [1e8]

    values = np.array(
        [[[S11, S12], [S21, S22]]],
        dtype=complex,
    )
    # Put coords in opposite order to check reordering
    coords = dict(
        f=np.array(freqs),
        port_out=port_names,
        port_in=port_names,
    )

    s_matrix = LumpedPortDataArray(data=values, coords=coords)
    z_matrix = AbstractComponentModeler.s_to_z(s_matrix, reference=Z0)
    z_matrix_at_f = z_matrix.sel(f=1e8)
    assert np.isclose(z_matrix_at_f[0, 0], Z11)
    assert np.isclose(z_matrix_at_f[0, 1], Z12)
    assert np.isclose(z_matrix_at_f[1, 0], Z21)
    assert np.isclose(z_matrix_at_f[1, 1], Z22)


def test_ab_to_s_component_modeler():
    coords = dict(
        f=np.array([1e8]),
        port_out=["lumped_port_1", "lumped_port_2"],
        port_in=["lumped_port_1", "lumped_port_2"],
    )
    # Common case is reference impedance matched to loads, which means ideally
    # the a matrix would be an identity matrix, and as a result the s matrix will be
    # given directly by the b_matrix
    a_values = np.eye(2, 2)
    a_values = np.reshape(a_values, (1, 2, 2))
    b_values = (1 + 1j) * np.random.random((1, 2, 2))
    a_matrix = LumpedPortDataArray(data=a_values, coords=coords)
    b_matrix = LumpedPortDataArray(data=b_values, coords=coords)
    S_matrix = AbstractComponentModeler.ab_to_s(a_matrix, b_matrix)
    assert np.isclose(S_matrix, b_matrix).all()


def test_port_snapping(tmp_path):
    """Make sure that the snapping behavior of the load resistor is mirrored
    by all other components in the modeler simulations with rectangular ports.
    """
    y_z_grid = td.UniformGrid(dl=0.1 * 1e3)
    x_grid = td.UniformGrid(dl=11 * 1e3)
    grid_spec = td.GridSpec(grid_x=x_grid, grid_y=y_z_grid, grid_z=y_z_grid)
    modeler = make_component_modeler(
        planar_pec=True, path_dir=str(tmp_path), port_refinement=False, grid_spec=grid_spec
    )
    check_lumped_port_components_snapped_correctly(modeler=modeler)


def test_coarse_grid_at_port(monkeypatch, tmp_path):
    modeler = make_component_modeler(planar_pec=True, path_dir=str(tmp_path), port_refinement=False)
    # Without port refinement the grid is much too coarse for these port sizes
    with pytest.raises(SetupError):
        _ = run_component_modeler(monkeypatch, modeler)


def test_validate_port_voltage_axis():
    with pytest.raises(pydantic.ValidationError):
        LumpedPort(center=(0, 0, 0), size=(0, 1, 2), voltage_axis=0, impedance=50)


@pytest.mark.parametrize("port_refinement", [False, True])
def test_make_coaxial_component_modeler(tmp_path, port_refinement):
    _ = make_coaxial_component_modeler(path_dir=str(tmp_path), port_refinement=port_refinement)


def test_run_coaxial_component_modeler(monkeypatch, tmp_path):
    modeler = make_coaxial_component_modeler(path_dir=str(tmp_path))
    s_matrix = run_component_modeler(monkeypatch, modeler)

    for port_in in modeler.ports:
        for port_out in modeler.ports:
            coords_in = dict(port_in=port_in.name)
            coords_out = dict(port_out=port_out.name)

            assert np.all(s_matrix.sel(**coords_in) != 0), "source index not present in S matrix"
            assert np.all(
                s_matrix.sel(**coords_in).sel(**coords_out) != 0
            ), "monitor index not present in S matrix"


def test_coarse_grid_at_coaxial_port(monkeypatch, tmp_path):
    modeler = make_coaxial_component_modeler(path_dir=str(tmp_path), port_refinement=False)
    # Without port refinement the grid is much too coarse for these port sizes
    with pytest.raises(SetupError):
        _ = run_component_modeler(monkeypatch, modeler)


def test_validate_coaxial_center_not_inf():
    with pytest.raises(pydantic.ValidationError):
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
    with pytest.raises(pydantic.ValidationError):
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
    modeler = make_coaxial_component_modeler(
        path_dir=str(tmp_path), port_refinement=False, grid_spec=grid_spec
    )
    check_lumped_port_components_snapped_correctly(modeler=modeler)


def test_power_delivered_helper(monkeypatch, tmp_path):
    """Test computations involving power waves are correct by manually setting voltage and current
    at ports using monkeypatch.
    """
    modeler = make_coaxial_component_modeler(path_dir=str(tmp_path))
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
        return FreqDataArray(voltage, coords=dict(f=freqs))

    def compute_current_patch(self, sim_data):
        return FreqDataArray(current, coords=dict(f=freqs))

    monkeypatch.setattr(CoaxialLumpedPort, "compute_voltage", compute_voltage_patch)
    monkeypatch.setattr(CoaxialLumpedPort, "compute_current", compute_current_patch)

    # First test should give complete power transfer into the network
    power = TerminalComponentModeler.compute_power_delivered_by_port(sim_data=None, port=port1)
    assert np.allclose(power.values, avg_power)

    # Second test is complete reflecton
    current = np.ones_like(freqs) * 0
    power = TerminalComponentModeler.compute_power_delivered_by_port(sim_data=None, port=port1)
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
    power = TerminalComponentModeler.compute_power_delivered_by_port(sim_data=None, port=port1)
    assert np.allclose(power.values, 0.5 * (power_a**2 - power_b**2))
