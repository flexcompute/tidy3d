import pytest
import numpy as np
import pydantic.v1 as pydantic
import matplotlib.pyplot as plt

import tidy3d as td
from tidy3d.plugins.smatrix.smatrix import (
    TerminalComponentModeler,
    AbstractComponentModeler,
)
from tidy3d.exceptions import Tidy3dKeyError
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


def test_make_component_modeler(tmp_path):
    _ = make_component_modeler(planar_pec=False, path_dir=str(tmp_path))


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
