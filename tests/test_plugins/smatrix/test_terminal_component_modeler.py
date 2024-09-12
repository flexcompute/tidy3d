import matplotlib.pyplot as plt
import numpy as np
import pydantic.v1 as pydantic
import pytest
import tidy3d as td
from tidy3d.components.data.data_array import FreqDataArray
from tidy3d.exceptions import SetupError, Tidy3dKeyError
from tidy3d.plugins.microwave import CustomCurrentIntegral2D, VoltageIntegralAxisAligned
from tidy3d.plugins.smatrix import (
    AbstractComponentModeler,
    CoaxialLumpedPort,
    LumpedPort,
    PortDataArray,
    TerminalComponentModeler,
    TerminalPortDataArray,
    WavePort,
)
from tidy3d.plugins.smatrix.ports.base_lumped import AbstractLumpedPort

from ...utils import run_emulated
from .terminal_component_modeler_def import make_coaxial_component_modeler, make_component_modeler


def run_component_modeler(monkeypatch, modeler: TerminalComponentModeler):
    sim_dict = modeler.sim_dict
    batch_data = {task_name: run_emulated(sim) for task_name, sim in sim_dict.items()}
    monkeypatch.setattr(AbstractComponentModeler, "batch_data", property(lambda self: batch_data))
    monkeypatch.setattr(TerminalComponentModeler, "batch_data", property(lambda self: batch_data))
    monkeypatch.setattr(AbstractComponentModeler, "inv", lambda matrix: np.eye(len(modeler.ports)))
    monkeypatch.setattr(
        TerminalComponentModeler,
        "_compute_F",
        lambda matrix: 1.0 / (2.0 * np.sqrt(np.abs(matrix) + 1e-4)),
    )
    monkeypatch.setattr(
        TerminalComponentModeler,
        "_check_port_impedance_sign",
        lambda self, Z_numpy: (),
    )

    s_matrix = modeler._construct_smatrix()
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
    freqs = [1e8, 2e8, 3e8]

    values = np.array(
        3 * [[[S11, S12], [S21, S22]]],
        dtype=complex,
    )
    # Put coords in opposite order to check reordering
    coords = dict(
        f=np.array(freqs),
        port_out=port_names,
        port_in=port_names,
    )

    s_matrix = TerminalPortDataArray(data=values, coords=coords)
    z_matrix = TerminalComponentModeler.s_to_z(s_matrix, reference=Z0)
    z_matrix_at_f = z_matrix.sel(f=1e8)
    assert np.isclose(z_matrix_at_f[0, 0], Z11)
    assert np.isclose(z_matrix_at_f[0, 1], Z12)
    assert np.isclose(z_matrix_at_f[1, 0], Z21)
    assert np.isclose(z_matrix_at_f[1, 1], Z22)

    # test version with different port reference impedances
    values = np.full((len(freqs), len(port_names)), Z0)
    coords = dict(
        f=np.array(freqs),
        port=port_names,
    )
    z_port_matrix = PortDataArray(data=values, coords=coords)
    z_matrix = TerminalComponentModeler.s_to_z(s_matrix, reference=z_port_matrix)
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
    a_matrix = TerminalPortDataArray(data=a_values, coords=coords)
    b_matrix = TerminalPortDataArray(data=b_values, coords=coords)
    S_matrix = TerminalComponentModeler.ab_to_s(a_matrix, b_matrix)
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


@pytest.mark.parametrize("snap_center", [None, 0.1])
def test_converting_port_to_simulation_objects(snap_center):
    """Test that the LumpedPort can be converted into monitors and source without the grid present."""
    port = LumpedPort(center=(0, 0, 0), size=(0, 1, 2), voltage_axis=2, impedance=50, name="Port1")
    freqs = np.linspace(1e9, 10e9, 11)
    source_time = td.GaussianPulse(freq0=5e9, fwidth=9e9)
    _ = port.to_field_monitors(freqs=freqs, snap_center=snap_center)
    _ = port.to_source(source_time=source_time, snap_center=snap_center)


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


def test_make_coaxial_component_modeler_with_wave_ports(tmp_path):
    """Checks that the terminal component modeler is created successfully with wave ports."""
    z_grid = td.UniformGrid(dl=1 * 1e3)
    xy_grid = td.UniformGrid(dl=0.1 * 1e3)
    grid_spec = td.GridSpec(grid_x=xy_grid, grid_y=xy_grid, grid_z=z_grid)
    _ = make_coaxial_component_modeler(
        path_dir=str(tmp_path), port_types=(WavePort, WavePort), grid_spec=grid_spec
    )


def test_run_coaxial_component_modeler_with_wave_ports(monkeypatch, tmp_path):
    """Checks that the terminal component modeler runs with wave ports."""
    z_grid = td.UniformGrid(dl=1 * 1e3)
    xy_grid = td.UniformGrid(dl=0.1 * 1e3)
    grid_spec = td.GridSpec(grid_x=xy_grid, grid_y=xy_grid, grid_z=z_grid)
    modeler = make_coaxial_component_modeler(
        path_dir=str(tmp_path), port_types=(WavePort, WavePort), grid_spec=grid_spec
    )
    s_matrix = run_component_modeler(monkeypatch, modeler)

    shape_one_port = (len(modeler.freqs), len(modeler.ports))
    shape_both_ports = (len(modeler.freqs),)
    for port_in in modeler.ports:
        for port_out in modeler.ports:
            coords_in = dict(port_in=port_in.name)
            coords_out = dict(port_out=port_out.name)

            assert np.all(
                s_matrix.sel(**coords_in).values.shape == shape_one_port
            ), "source index not present in S matrix"
            assert np.all(
                s_matrix.sel(**coords_in).sel(**coords_out).values.shape == shape_both_ports
            ), "monitor index not present in S matrix"


def test_run_mixed_component_modeler_with_wave_ports(monkeypatch, tmp_path):
    """Checks the terminal component modeler will allow mixed ports."""
    z_grid = td.UniformGrid(dl=1 * 1e3)
    xy_grid = td.UniformGrid(dl=0.1 * 1e3)
    grid_spec = td.GridSpec(grid_x=xy_grid, grid_y=xy_grid, grid_z=z_grid)
    modeler = make_coaxial_component_modeler(
        path_dir=str(tmp_path), port_types=(CoaxialLumpedPort, WavePort), grid_spec=grid_spec
    )
    s_matrix = run_component_modeler(monkeypatch, modeler)

    shape_one_port = (len(modeler.freqs), len(modeler.ports))
    shape_both_ports = (len(modeler.freqs),)
    for port_in in modeler.ports:
        for port_out in modeler.ports:
            coords_in = dict(port_in=port_in.name)
            coords_out = dict(port_out=port_out.name)

            assert np.all(
                s_matrix.sel(**coords_in).values.shape == shape_one_port
            ), "source index not present in S matrix"
            assert np.all(
                s_matrix.sel(**coords_in).sel(**coords_out).values.shape == shape_both_ports
            ), "monitor index not present in S matrix"


def test_wave_port_path_integral_validation():
    """Checks that wave port will ensure path integrals are within the bounds of the port."""
    size_port = [2, 2, 0]
    center_port = [0, 0, -10]

    voltage_path = VoltageIntegralAxisAligned(
        center=(0.5, 0, -10),
        size=(1.0, 0, 0),
        extrapolate_to_endpoints=True,
        snap_path_to_grid=True,
        sign="+",
    )

    custom_current_path = CustomCurrentIntegral2D.from_circular_path(
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

    with pytest.raises(pydantic.ValidationError):
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
    with pytest.raises(pydantic.ValidationError):
        _ = WavePort(
            center=center_port,
            size=size_port,
            name="wave_port_1",
            mode_spec=mode_spec,
            direction="+",
            voltage_integral=voltage_path,
            current_integral=None,
        )

    custom_current_path = CustomCurrentIntegral2D.from_circular_path(
        center=center_port, radius=3, num_points=21, normal_axis=2, clockwise=False
    )
    with pytest.raises(pydantic.ValidationError):
        _ = WavePort(
            center=center_port,
            size=size_port,
            name="wave_port_1",
            mode_spec=mode_spec,
            direction="+",
            voltage_integral=None,
            current_integral=custom_current_path,
        )


def test_wave_port_to_mode_solver(tmp_path):
    """Checks that wave port can be converted to a mode solver."""
    modeler = make_coaxial_component_modeler(
        path_dir=str(tmp_path), port_types=(WavePort, WavePort)
    )
    _ = modeler.ports[0].to_mode_solver(modeler.simulation, freqs=[1e9, 2e9, 3e9])


def test_port_source_snapped_to_PML(tmp_path):
    """Raise meaningful error message when source is snapped into PML because the port is too close
    to the boundary.
    """
    modeler = make_component_modeler(planar_pec=True, path_dir=str(tmp_path))
    port_pos = 5e4
    voltage_path = VoltageIntegralAxisAligned(
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
    modeler = make_coaxial_component_modeler(
        path_dir=str(tmp_path), port_types=(WavePort, WavePort)
    )
    with pytest.raises(pydantic.ValidationError):
        _ = modeler.updated_copy(direction="-", path="ports/0/")
