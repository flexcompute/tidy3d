from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import responses

import tidy3d as td
from tests.utils import AssertLogLevel
from tidy3d import config
from tidy3d.components.data.data_array import ScalarFieldDataArray
from tidy3d.components.data.monitor_data import FieldData
from tidy3d.components.data.sim_data import SimulationData
from tidy3d.components.grid.grid_spec import GridSpec
from tidy3d.components.monitor import FieldMonitor
from tidy3d.components.source.current import PointDipole
from tidy3d.components.source.time import GaussianPulse
from tidy3d.web.api.tidy3d_stub import Tidy3dStub, Tidy3dStubData
from tidy3d.web.core.types import TaskType

config.switch_profile("test")


def make_sim():
    """Makes a simulation."""
    pulse = td.GaussianPulse(freq0=200e12, fwidth=20e12)
    pt_dipole = td.PointDipole(source_time=pulse, polarization="Ex")
    return td.Simulation(
        size=(1, 1, 1),
        grid_spec=td.GridSpec.auto(wavelength=1.0),
        run_time=1e-12,
        sources=[pt_dipole],
    )


def is_lazy_object(data):
    assert set(data.__dict__.keys()) == {
        "_lazy_fname",
        "_lazy_group_path",
        "_lazy_parse_obj_kwargs",
    }
    return True


def make_sim_data(file_size_gb=0.001):
    """Makes a simulation data."""
    N = int(2.528e8 / 4 * file_size_gb)
    n = int(N ** (0.25))
    data = (1 + 1j) * np.random.random((n, n, n, n))
    x = np.linspace(-1, 1, n)
    y = np.linspace(-1, 1, n)
    z = np.linspace(-1, 1, n)
    f = np.linspace(2e14, 4e14, n)
    src = PointDipole(
        center=(0, 0, 0), source_time=GaussianPulse(freq0=3e14, fwidth=1e14), polarization="Ex"
    )
    coords = {"x": x, "y": y, "z": z, "f": f}
    Ex = ScalarFieldDataArray(data, coords=coords)
    monitor = FieldMonitor(size=(2, 2, 2), freqs=f, name="test", fields=["Ex"])
    field_data = FieldData(monitor=monitor, Ex=Ex)
    sim = td.Simulation(
        size=(2, 2, 2),
        grid_spec=GridSpec(wavelength=1),
        monitors=(monitor,),
        sources=(src,),
        run_time=1e-12,
    )
    return SimulationData(
        simulation=sim,
        data=(field_data,),
    )


@responses.activate
def test_stub_to_hdf5_gz(tmp_path):
    """Tests the to_hdf5_gz method of Tidy3dStub."""
    sim = make_sim()
    stub = Tidy3dStub(simulation=sim)
    file_path = os.path.join(tmp_path, "test.hdf5.gz")
    stub.to_hdf5_gz(file_path)
    assert os.path.exists(file_path)


@responses.activate
def test_stub_to_file(tmp_path):
    """Tests the to_file method of Tidy3dStub."""
    sim = make_sim()
    stub = Tidy3dStub(simulation=sim)
    file_path = os.path.join(tmp_path, "test.json")
    stub.to_file(file_path)
    assert os.path.exists(file_path)
    sim2 = Tidy3dStub.from_file(file_path)
    assert sim == sim2


@responses.activate
def test_stub_data_to_file(tmp_path):
    """Tests the to_file method of Tidy3dStubData."""
    sim_data = make_sim_data()
    stub_data = Tidy3dStubData(data=sim_data)
    file_path = os.path.join(tmp_path, "test.hdf5")
    stub_data.to_file(file_path)
    assert os.path.exists(file_path)
    sim_data2 = Tidy3dStubData.from_file(file_path)
    assert sim_data.simulation == sim_data2.simulation


@responses.activate
def test_stub_data_postprocess_logs(tmp_path):
    """Tests the postprocess method of Tidy3dStubData when simulation diverged."""
    td.log.set_capture(True)
    try:
        # test diverged
        sim_data = make_sim_data()
        sim_data = sim_data.updated_copy(diverged=True, log="The simulation has diverged!")
        file_path = os.path.join(tmp_path, "test_diverged.hdf5")
        sim_data.to_file(file_path)
        Tidy3dStubData.postprocess(file_path)

        # test warnings
        sim_data = make_sim_data()
        sim_data = sim_data.updated_copy(log="WARNING: messages were found in the solver log.")
        file_path = os.path.join(tmp_path, "test_warnings.hdf5")
        sim_data.to_file(file_path)
        Tidy3dStubData.postprocess(file_path)
    finally:
        td.log.set_capture(False)


@responses.activate
def test_stub_data_lazy_loading(tmp_path):
    """Tests the postprocess method with lazy loading of Tidy3dStubData when simulation diverged."""
    td.log.set_capture(True)
    sim_diverged_log = "The simulation has diverged!"
    try:
        # make sim data where test diverged
        sim_data = make_sim_data()
        sim_data = sim_data.updated_copy(diverged=True, log=sim_diverged_log)
        file_path = os.path.join(tmp_path, "test_diverged.hdf5")
        sim_data.to_file(file_path)

        # default case with lazy=False should output a warning
        with AssertLogLevel("WARNING", contains_str=sim_diverged_log):
            Tidy3dStubData.postprocess(file_path, lazy=False)

        # we expect no warning in lazy mode as object should not be loaded
        with AssertLogLevel(None):
            sim_data = Tidy3dStubData.postprocess(file_path, lazy=True)

        sim_data_copy = sim_data.copy()

        # variable dict should only contain metadata to load the data, not the data itself
        assert is_lazy_object(sim_data)

        # the type should be still SimulationData despite being lazy
        assert isinstance(sim_data, SimulationData)

        # metadata-only lazy access should not force materialization or warnings
        with AssertLogLevel(None):
            _ = sim_data_copy.monitor_data

        assert "_lazy_fname" in sim_data_copy.__dict__

        # warning is emitted once the full object is materialized
        with AssertLogLevel("WARNING", contains_str=sim_diverged_log):
            _ = sim_data_copy.data
    finally:
        td.log.set_capture(False)


@responses.activate
def test_sim_data_lazy_materialize_then_copy(tmp_path):
    """Regression test for copying lazily loaded SimulationData after materialization."""
    sim_data = make_sim_data()
    file_path = os.path.join(tmp_path, "test_lazy_copy.hdf5")
    sim_data.to_file(file_path)

    sim_data_lazy = SimulationData.from_file(file_path, lazy=True)
    assert is_lazy_object(sim_data_lazy)

    # Materialize the proxy first, then verify regular copy semantics work.
    _ = sim_data_lazy.data
    assert isinstance(sim_data_lazy, SimulationData)
    assert hasattr(sim_data_lazy, "__pydantic_extra__")

    sim_data_copy = sim_data_lazy.copy(deep=False)
    assert isinstance(sim_data_copy, SimulationData)
    assert sim_data_copy.simulation == sim_data_lazy.simulation
    assert len(sim_data_copy.data) == len(sim_data_lazy.data)


@pytest.mark.parametrize(
    "path_builder",
    (
        lambda tmp_path, name: Path(tmp_path) / name,
        lambda tmp_path, name: str(Path(tmp_path) / name),
    ),
)
def test_stub_pathlike_roundtrip(tmp_path, path_builder):
    """Ensure stub read/write helpers accept pathlib.Path and posixpath inputs."""

    # Simulation stub roundtrip
    sim = make_sim()
    stub = Tidy3dStub(simulation=sim)
    sim_path = path_builder(tmp_path, "pathlike_sim.json")
    stub.to_file(sim_path)
    assert os.path.exists(sim_path)
    sim_loaded = Tidy3dStub.from_file(sim_path)
    assert sim_loaded == sim

    # Simulation data stub roundtrip
    sim_data = make_sim_data()
    sim_data = sim_data.updated_copy(log="log")
    stub_data = Tidy3dStubData(data=sim_data)
    data_path = path_builder(tmp_path, "pathlike_data.hdf5")
    stub_data.to_file(data_path)
    assert os.path.exists(data_path)

    data_loaded = Tidy3dStubData.from_file(data_path)
    assert data_loaded.simulation == sim_data.simulation

    # Postprocess using the same PathLike ensures downstream helpers accept the type
    processed = Tidy3dStubData.postprocess(data_path, lazy=True)
    assert isinstance(processed, SimulationData)


def test_default_task_name():
    sim = make_sim()
    stub = Tidy3dStub(simulation=sim)
    default_task_name = stub.get_default_task_name()
    assert default_task_name.startswith(TaskType.FDTD.name.lower())
