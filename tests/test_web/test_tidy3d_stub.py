from __future__ import annotations

import os

import numpy as np
import responses

import tidy3d as td
from tidy3d.components.data.data_array import ScalarFieldDataArray
from tidy3d.components.data.monitor_data import FieldData
from tidy3d.components.data.sim_data import SimulationData
from tidy3d.components.grid.grid_spec import GridSpec
from tidy3d.components.monitor import FieldMonitor
from tidy3d.components.source.current import PointDipole
from tidy3d.components.source.time import GaussianPulse
from tidy3d.web.api.tidy3d_stub import Tidy3dStub, Tidy3dStubData
from tidy3d.web.core.environment import Env, EnvironmentConfig
from tidy3d.web.core.types import TaskType

test_env = EnvironmentConfig(
    name="test",
    s3_region="test",
    web_api_endpoint="https://test",
    website_endpoint="https://test",
)

Env.set_current(test_env)


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


def test_default_task_name():
    sim = make_sim()
    stub = Tidy3dStub(simulation=sim)
    default_task_name = stub.get_default_task_name()
    assert default_task_name.startswith(TaskType.FDTD.name.lower())
