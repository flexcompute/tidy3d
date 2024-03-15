import pytest
import numpy as np
import pydantic.v1 as pydantic
import matplotlib.pyplot as plt

import tidy3d as td
from tidy3d.plugins.smatrix import (
    AbstractComponentModeler,
    LumpedPort,
    LumpedPortDataArray,
    TerminalComponentModeler,
)
from tidy3d.exceptions import Tidy3dKeyError, SetupError
from ..utils import run_emulated
from .terminal_component_modeler_def import make_component_modeler


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


def test_port_snapping(monkeypatch, tmp_path):
    modeler = make_component_modeler(
        planar_pec=True, path_dir=str(tmp_path), port_refinement=False, auto_grid=False
    )
    # Without port refinement the grid is much too coarse for these port sizes
    with pytest.raises(SetupError):
        _ = run_component_modeler(monkeypatch, modeler)


def test_coarse_grid_at_port(monkeypatch, tmp_path):
    modeler = make_component_modeler(planar_pec=True, path_dir=str(tmp_path), port_refinement=False)
    # Without port refinement the grid is much too coarse for these port sizes
    with pytest.raises(SetupError):
        _ = run_component_modeler(monkeypatch, modeler)


def test_validate_port_voltage_axis():
    with pytest.raises(pydantic.ValidationError):
        LumpedPort(center=(0, 0, 0), size=(0, 1, 2), voltage_axis=0, impedance=50)
