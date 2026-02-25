# Tests webapi and things that depend on it
from __future__ import annotations

import concurrent.futures
import os
import posixpath
from concurrent.futures import Future
from os import PathLike
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import responses
from _pytest import monkeypatch
from pydantic import ValidationError
from responses import matchers

import tidy3d as td
from tests.test_web.test_tidy3d_stub import is_lazy_object
from tidy3d import Simulation, config
from tidy3d.__main__ import main
from tidy3d.components.data.data_array import ScalarFieldDataArray
from tidy3d.components.data.monitor_data import FieldData
from tidy3d.components.data.sim_data import SimulationData
from tidy3d.components.grid.grid_spec import GridSpec
from tidy3d.components.monitor import FieldMonitor
from tidy3d.components.source.current import PointDipole
from tidy3d.components.source.time import GaussianPulse
from tidy3d.exceptions import DataError, SetupError
from tidy3d.web import common
from tidy3d.web.api.asynchronous import run_async
from tidy3d.web.api.container import Batch, BatchData, Job, WebContainer
from tidy3d.web.api.run import _collect_by_hash, run
from tidy3d.web.api.tidy3d_stub import Tidy3dStubData, task_type_name_of
from tidy3d.web.api.webapi import (
    abort,
    delete,
    delete_old,
    download,
    download_json,
    download_log,
    estimate_cost,
    get_info,
    get_run_info,
    get_tasks,
    load,
    load_simulation,
    monitor,
    real_cost,
    start,
    upload,
)
from tidy3d.web.core.environment import Env
from tidy3d.web.core.exceptions import WebNotFoundError
from tidy3d.web.core.types import PayType, TaskType

TASK_NAME = "task_name_test"
TASK_ID = "1234"
FOLDER_ID = "1234"
CREATED_AT = "2022-01-01T00:00:00.000Z"
PROJECT_NAME = "default"
FLEX_UNIT = 1.0
EST_FLEX_UNIT = 11.11
FILE_SIZE_GB = 4.0
common.CONNECTION_RETRY_TIME = 0.1
INVALID_TASK_ID = "INVALID_TASK_ID"

task_core_path = "tidy3d.web.core.task_core"
api_path = "tidy3d.web.api.webapi"

config.switch_profile("dev")


class FakeJob:
    def __init__(self, task_id: str, statuses: list[str], events: list[str]):
        self.task_id = task_id
        self._statuses = statuses
        self._idx = 0
        self.events = events

    @property
    def status(self):
        status = self._statuses[self._idx]
        if self._idx < len(self._statuses) - 1:
            self._idx += 1
        self.events.append((self.task_id, "status", status))
        return status

    def get_info(self):
        return SimpleNamespace(status=self.status)

    def download(self, path: PathLike):
        self.events.append((self.task_id, "download", str(path)))

    @property
    def load_if_cached(self):
        return False


class FakeJobWithSimulation(FakeJob):
    def __init__(
        self, task_id: str, statuses: list[str], events: list[str], simulation: td.Simulation
    ):
        super().__init__(task_id=task_id, statuses=statuses, events=events)
        self.simulation = simulation


class UploadStartFakeJob:
    def __init__(self, task_id: str, events: list[tuple], cached: bool = False):
        self.task_id = task_id
        self.events = events
        self._cached = cached

    @property
    def load_if_cached(self):
        return self._cached

    def upload(self):
        self.events.append((self.task_id, "upload"))

    def start(self, priority=None):
        self.events.append((self.task_id, "start", priority))


class UploadEstimateFakeJob:
    def __init__(
        self,
        task_id: str,
        events: list[tuple],
        metadata_statuses: list[str],
    ):
        self.task_id = task_id
        self.events = events
        self._metadata_statuses = metadata_statuses
        self._metadata_idx = 0

    @property
    def load_if_cached(self):
        return False

    def upload(self, wait_for_estimate_cost=True):
        self.events.append((self.task_id, "upload", wait_for_estimate_cost))

    def get_info(self):
        status = self._metadata_statuses[self._metadata_idx]
        if self._metadata_idx < len(self._metadata_statuses) - 1:
            self._metadata_idx += 1
        self.events.append((self.task_id, "metadata", status))
        return SimpleNamespace(metadataStatus=status)

    def start(self, priority=None):
        self.events.append((self.task_id, "start", priority))


class LoadStatusFakeJob:
    def __init__(self, task_id: str, status: str, simulation: td.Simulation):
        self.task_id = task_id
        self._status = status
        self.simulation = simulation

    @property
    def status(self):
        return self._status

    @property
    def load_if_cached(self):
        return False


class ImmediateExecutor:
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.shutdown(wait=True)

    def submit(self, fn, *args, **kwargs):
        future = Future()
        try:
            result = fn(*args, **kwargs)
        except Exception as err:
            future.set_exception(err)
        else:
            future.set_result(result)
        return future

    def shutdown(self, wait=True):
        pass


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


def make_sim_data(file_size_gb=FILE_SIZE_GB):
    """Makes a simulation."""
    # approximate # of points in the scalar field data

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
    sim = Simulation(
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


@pytest.fixture
def set_api_key(monkeypatch):
    """Set the api key."""
    import tidy3d.web.core.http_util as http_module

    monkeypatch.setattr(http_module, "api_key", lambda: "apikey")
    monkeypatch.setattr(http_module, "get_version", lambda: td.version.__version__)


@pytest.fixture
def mock_upload(monkeypatch, set_api_key):
    """Mocks webapi.upload."""
    responses.add(
        responses.GET,
        f"{Env.current.web_api_endpoint}/tidy3d/project",
        match=[matchers.query_param_matcher({"projectName": PROJECT_NAME})],
        json={"data": {"projectId": FOLDER_ID, "projectName": PROJECT_NAME}},
        status=200,
    )

    responses.add(
        responses.POST,
        f"{Env.current.web_api_endpoint}/tidy3d/projects/{FOLDER_ID}/tasks",
        match=[
            matchers.json_params_matcher(
                {
                    "taskType": TaskType.FDTD.name,
                    "callbackUrl": None,
                    "simulationType": "tidy3d",
                    "parentTasks": None,
                    "fileType": "Gz",
                },
                strict_match=False,
            )
        ],
        json={
            "data": {
                "taskId": TASK_ID,
                "taskName": TASK_NAME,
                "createdAt": CREATED_AT,
            }
        },
        status=200,
    )

    def mock_upload_file(*args, **kwargs):
        pass

    monkeypatch.setattr("tidy3d.web.core.task_core.upload_file", mock_upload_file)


@pytest.fixture
def mock_get_info(monkeypatch, set_api_key):
    """Mocks webapi.get_info."""

    responses.add(
        responses.GET,
        f"{Env.current.web_api_endpoint}/tidy3d/tasks/{TASK_ID}/detail",
        json={
            "data": {
                "taskId": TASK_ID,
                "taskName": TASK_NAME,
                "createdAt": CREATED_AT,
                "realFlexUnit": FLEX_UNIT,
                "estFlexUnit": EST_FLEX_UNIT,
                "taskType": TaskType.FDTD.name,
                "metadataStatus": "processed",
                "status": "success",
                "s3Storage": 1.0,
            }
        },
        status=200,
    )


@pytest.fixture
def mock_start(monkeypatch, set_api_key, mock_get_info):
    """Mocks webapi.start."""

    def add_mock_response(priority=None):
        expected_body = {
            "solverVersion": None,
            "workerGroup": None,
            "protocolVersion": td.version.__version__,
            "enableCaching": Env.current.enable_caching,
            "payType": PayType.AUTO,
            "priority": priority,
        }

        responses.add(
            responses.POST,
            f"{Env.current.web_api_endpoint}/tidy3d/tasks/{TASK_ID}/submit",
            match=[matchers.json_params_matcher(expected_body)],
            json={
                "data": {
                    "taskId": TASK_ID,
                    "taskName": TASK_NAME,
                    "createdAt": CREATED_AT,
                }
            },
            status=200,
        )

    # Add response for calls without priority
    add_mock_response(None)

    # Add responses for calls with specific priority values
    for priority in [1, 5, 10]:
        add_mock_response(priority)


@pytest.fixture
def mock_monitor(monkeypatch):
    status_count = [0]
    statuses = ("upload", "running", "running", "running", "running", "running", "success")

    def mock_get_status(task_id):
        current_count = min(status_count[0], len(statuses) - 1)
        current_status = statuses[current_count]
        status_count[0] += 1
        return current_status

    run_count = [0]
    perc_dones = (1, 10, 20, 30, 100)

    def mock_get_run_info(task_id):
        current_count = min(run_count[0], len(perc_dones) - 1)
        perc_done = perc_dones[current_count]
        run_count[0] += 1
        return perc_done, 1

    monkeypatch.setattr("tidy3d.web.api.connect_util.REFRESH_TIME", 0.00001)
    monkeypatch.setattr(f"{api_path}.REFRESH_TIME", 0.00001)
    monkeypatch.setattr("tidy3d.web.api.container.web.REFRESH_TIME", 0.00001)
    monkeypatch.setattr(f"{api_path}.RUN_REFRESH_TIME", 0.00001)
    monkeypatch.setattr(f"{api_path}.get_status", mock_get_status)
    monkeypatch.setattr(f"{api_path}.get_run_info", mock_get_run_info)


@pytest.fixture
def mock_download(monkeypatch, set_api_key, mock_get_info, tmp_path):
    """Mocks webapi.download."""

    def _mock_download(*args, **kwargs):
        file_path = kwargs["to_file"]
        with open(file_path, "w") as f:
            f.write("0.3,5.7")

    monkeypatch.setattr(f"{task_core_path}.download_gz_file", _mock_download)
    monkeypatch.setattr(f"{task_core_path}.download_file", _mock_download)


@pytest.fixture
def mock_load(monkeypatch, set_api_key, mock_get_info):
    """Mocks webapi.load"""

    def _mock_download(*args, **kwargs):
        pass

    monkeypatch.setattr(f"{task_core_path}.download_file", _mock_download)


@pytest.fixture
def mock_metadata(monkeypatch, set_api_key):
    """Mocks call to metadata api"""
    responses.add(
        responses.POST,
        f"{Env.current.web_api_endpoint}/tidy3d/tasks/{TASK_ID}/metadata",
        json={
            "data": {
                "createdAt": CREATED_AT,
            }
        },
        status=200,
    )


@pytest.fixture
def mock_get_run_info(monkeypatch, set_api_key):
    """Mocks webapi.get_run_info"""
    responses.add(
        responses.GET,
        f"{Env.current.web_api_endpoint}/tidy3d/tasks/{TASK_ID}/progress",
        json={
            "data": {
                "perc_done": 100,
                "field_decay": 0,
            }
        },
        status=200,
    )


@pytest.fixture
def mock_webapi(
    mock_upload, mock_metadata, mock_get_info, mock_start, mock_monitor, mock_download, mock_load
):
    """Mocks all webapi operation."""


@responses.activate
def test_source_validation(monkeypatch, mock_upload, mock_get_info, mock_metadata):
    sim = make_sim().copy(update={"sources": ()})

    assert upload(sim, TASK_NAME, PROJECT_NAME, source_required=False)
    with pytest.raises(SetupError):
        upload(sim, TASK_NAME, PROJECT_NAME)


@responses.activate
def test_upload(monkeypatch, mock_upload, mock_get_info, mock_metadata):
    sim = make_sim()
    assert upload(sim, TASK_NAME, PROJECT_NAME)


@responses.activate
def test_get_info(mock_get_info):
    assert get_info(TASK_ID).taskId == TASK_ID


@responses.activate
def test_start(mock_start):
    start(TASK_ID)


@responses.activate
@pytest.mark.parametrize("priority", [1, 5, 10, None])
def test_start_with_valid_priority(mock_start, priority):
    """Test start with valid priority values."""
    start(TASK_ID, priority=priority)


@responses.activate
@pytest.mark.parametrize("priority", [0, -1, 11, 15])
def test_start_with_invalid_priority(mock_start, priority):
    """Test start with invalid priority values."""
    with pytest.raises(ValueError, match="Priority must be between '1' and '10' if specified."):
        start(TASK_ID, priority=priority)


@responses.activate
@pytest.mark.parametrize("priority", [5, None])
def test_run_with_valid_priority(mock_webapi, monkeypatch, priority):
    """Test run with valid priority parameter."""
    monkeypatch.setattr(f"{api_path}.load", lambda *args, **kwargs: True)
    sim = make_sim()
    run(sim, TASK_NAME, folder_name=PROJECT_NAME, priority=priority)


@responses.activate
@pytest.mark.parametrize("priority", [0, -1, 11, 15])
def test_run_with_invalid_priority(mock_webapi, priority):
    """Test run with invalid priority values."""
    sim = make_sim()
    with pytest.raises(ValueError, match="Priority must be between '1' and '10' if specified."):
        run(sim, TASK_NAME, folder_name=PROJECT_NAME, priority=priority)


@responses.activate
def test_get_run_info(mock_get_run_info, mock_get_info):
    assert get_run_info(TASK_ID) == (100, 0)


@responses.activate
def test_download(mock_download, tmp_path):
    download(TASK_ID, str(tmp_path / "web_test_tmp.json"))
    with open(str(tmp_path / "web_test_tmp.json")) as f:
        assert f.read() == "0.3,5.7"


@responses.activate
def _test_load(mock_load, mock_get_info, tmp_path):
    def mock_download(*args, **kwargs):
        pass

    monkeypatch.setattr(f"{task_core_path}.download_file", mock_download)
    load(TASK_ID, str(tmp_path / "monitor_data.hdf5"))


def test_batch_load_sim_data_skips_task_lookup(monkeypatch, tmp_path):
    data_path = tmp_path / "batch_results.hdf5"
    data_path.write_text("stub")
    batch_data = BatchData(
        task_paths={"task_1": str(data_path)},
        task_ids={"task_1": TASK_ID},
        cached_tasks={"task_1": False},
        is_downloaded=True,
    )

    def _raise(*args, **kwargs):
        raise AssertionError("Unexpected web lookup during batch load.")

    monkeypatch.setattr(f"{api_path}.get_info", _raise)
    monkeypatch.setattr(f"{task_core_path}.TaskFactory.get", _raise)
    monkeypatch.setattr(f"{task_core_path}.TaskFactory.get_kind", _raise)
    monkeypatch.setattr(f"{api_path}.resolve_local_cache", lambda: None)
    monkeypatch.setattr(
        f"{api_path}.Tidy3dStubData.postprocess", lambda *args, **kwargs: "stub_data"
    )

    assert batch_data.load_sim_data("task_1") == "stub_data"


@responses.activate
def test_delete(set_api_key, mock_get_info):
    responses.add(
        responses.GET,
        f"{Env.current.web_api_endpoint}/tidy3d/tasks/{TASK_ID}",
        json={
            "data": {
                "taskId": TASK_ID,
                "groupId": "group123",
                "version": "v1",
                "createdAt": CREATED_AT,
            }
        },
        status=200,
    )

    responses.add(
        responses.DELETE,
        f"{Env.current.web_api_endpoint}/tidy3d/group/group123/versions",
        match=[
            matchers.json_params_matcher(
                {
                    "versions": ["v1"],
                }
            )
        ],
        json={
            "data": {
                "taskId": TASK_ID,
                "createdAt": CREATED_AT,
            }
        },
        status=200,
    )

    responses.add(
        responses.DELETE,
        f"{Env.current.web_api_endpoint}/tidy3d/tasks/{TASK_ID}",
        json={
            "data": {
                "taskId": TASK_ID,
                "createdAt": CREATED_AT,
            }
        },
        status=200,
    )

    assert delete(TASK_ID).taskId == TASK_ID


@responses.activate
def test_estimate_cost(set_api_key, mock_get_info, mock_metadata):
    assert estimate_cost(TASK_ID) == EST_FLEX_UNIT


@responses.activate
def test_download_json(monkeypatch, mock_get_info, tmp_path):
    sim = make_sim()

    def mock_download(*args, **kwargs):
        pass

    def get_str(*args, **kwargs):
        return sim.model_dump_json().encode("utf-8")

    monkeypatch.setattr(f"{task_core_path}.download_gz_file", mock_download)
    monkeypatch.setattr(f"{task_core_path}.read_simulation_from_hdf5", get_str)

    fname_tmp = str(tmp_path / "web_test_tmp.json")
    download_json(TASK_ID, fname_tmp)
    assert Simulation.from_file(fname_tmp) == sim


@responses.activate
def test_load_simulation(monkeypatch, mock_get_info, tmp_path):
    def mock_download(*args, **kwargs):
        make_sim().to_file(args[1])

    monkeypatch.setattr(f"{task_core_path}.SimulationTask.get_simulation_json", mock_download)

    assert load_simulation(TASK_ID, str(tmp_path / "web_test_tmp.json"))


@responses.activate
def test_download_log(monkeypatch, mock_get_info, tmp_path):
    def mock(*args, **kwargs):
        file_path = kwargs["to_file"]
        with open(file_path, "w") as f:
            f.write("0.3,5.7")

    monkeypatch.setattr(f"{task_core_path}.download_file", mock)

    download_log(TASK_ID, str(tmp_path / "web_test_tmp.json"))
    with open(str(tmp_path / "web_test_tmp.json")) as f:
        assert f.read() == "0.3,5.7"


@responses.activate
def test_delete_old(set_api_key):
    responses.add(
        responses.GET,
        f"{Env.current.web_api_endpoint}/tidy3d/project",
        match=[matchers.query_param_matcher({"projectName": PROJECT_NAME})],
        json={"data": {"projectId": TASK_ID, "projectName": PROJECT_NAME}},
        status=200,
    )
    responses.add(
        responses.DELETE,
        f"{Env.current.web_api_endpoint}/tidy3d/tasks/{FOLDER_ID}/tasks",
        json={"data": 0, "warning": "string"},
        status=200,
    )

    delete_old(days_old=100)


@responses.activate
def test_get_tasks(set_api_key):
    responses.add(
        responses.GET,
        f"{Env.current.web_api_endpoint}/tidy3d/project",
        match=[matchers.query_param_matcher({"projectName": PROJECT_NAME})],
        json={"data": {"projectId": TASK_ID, "projectName": PROJECT_NAME}},
        status=200,
    )
    responses.add(
        responses.GET,
        f"{Env.current.web_api_endpoint}/tidy3d/projects/{TASK_ID}/tasks",
        json={"data": [{"taskId": TASK_ID, "createdAt": CREATED_AT}]},
        status=200,
    )

    assert get_tasks(1)[0]["task_id"] == TASK_ID


@responses.activate
@pytest.mark.parametrize("task_name", [TASK_NAME, None])
def test_run(mock_webapi, monkeypatch, tmp_path, task_name):
    sim = make_sim()
    monkeypatch.setattr(f"{api_path}.load", lambda *args, **kwargs: True)
    assert run(
        sim,
        task_name=task_name,
        folder_name=PROJECT_NAME,
        path=str(tmp_path / "web_test_tmp.json"),
    )


@responses.activate
def test_monitor(mock_get_info, mock_monitor):
    monitor(TASK_ID, verbose=True)
    monitor(TASK_ID, verbose=False)


@responses.activate
def test_real_cost(mock_get_info):
    assert real_cost(TASK_ID) == FLEX_UNIT


@responses.activate
def test_abort_task(set_api_key):
    responses.add(
        responses.PUT,
        f"{Env.current.web_api_endpoint}/tidy3d/tasks/abort",
        match=[
            matchers.json_params_matcher(
                {
                    "taskId": TASK_ID,
                    "taskType": TaskType.FDTD.name,
                }
            )
        ],
        json={"result": True},
        status=200,
    )
    responses.add(
        responses.GET,
        f"{Env.current.web_api_endpoint}/tidy3d/tasks/{TASK_ID}/detail",
        json={
            "taskId": TASK_ID,
            "taskName": TASK_NAME,
            "createdAt": "2022-01-01T00:00:00.000Z",
            "status": "running",
            "taskType": TaskType.FDTD.name,
            "taskBlockInfo": {
                "chargeType": "free",
                "maxFreeCount": 20,
                "maxGridPoints": 1000,
                "maxTimeSteps": 1000,
            },
        },
        status=200,
    )

    abort(TASK_ID)


""" Containers """


@responses.activate
@pytest.mark.parametrize("task_name", [TASK_NAME, None])
def test_job(mock_webapi, monkeypatch, tmp_path, task_name):
    monkeypatch.setattr("tidy3d.web.api.container.Job.load", lambda *args, **kwargs: True)
    sim = make_sim()
    j = Job(simulation=sim, task_name=task_name, folder_name=PROJECT_NAME)

    fname = str(tmp_path / "web_test_tmp.json")

    j.to_file(fname)

    j = j.from_file(fname)

    _ = j.run(path=fname)
    _ = j.status
    j.estimate_cost()
    # j.download
    _ = j.delete
    assert j.real_cost() == FLEX_UNIT


@pytest.fixture
def mock_job_status(monkeypatch):
    monkeypatch.setattr("tidy3d.web.api.container.Job.status", property(lambda self: "success"))
    monkeypatch.setattr("tidy3d.web.api.container.Job.load", lambda *args, **kwargs: True)


@responses.activate
@pytest.mark.parametrize("task_name", [TASK_NAME, None])
def test_batch(mock_webapi, mock_job_status, mock_load, tmp_path, task_name):
    # monkeypatch.setattr("tidy3d.web.api.container.Batch.monitor", lambda self: time.sleep(0.1))
    # monkeypatch.setattr("tidy3d.web.api.container.Job.status", property(lambda self: "success"))
    if task_name is None:
        sims = [make_sim()]
    else:
        sims = {TASK_NAME: make_sim()}

    b = Batch(simulations=sims, folder_name=PROJECT_NAME)

    fname = str(tmp_path / "batch.json")

    b.to_file(fname)
    b2 = b.from_file(fname)

    assert all(j.task_id == j2.task_id for j, j2 in zip(b.jobs.values(), b2.jobs.values()))

    b2.estimate_cost()
    b2.run(path_dir=str(tmp_path))
    _ = b2.get_info()
    assert b2.real_cost() == FLEX_UNIT * len(sims)


def test_batch_accepts_string_simulation_keys():
    sims = {"0": make_sim(), "1": make_sim()}

    batch = Batch(simulations=sims, folder_name=PROJECT_NAME)

    assert tuple(batch.simulations.keys()) == ("0", "1")


def test_batch_rejects_numeric_simulation_keys_with_clear_message():
    sims = {0: make_sim()}

    with pytest.raises(
        ValidationError,
        match="Batch simulations keys must be strings \\(task names\\)",
    ):
        Batch(simulations=sims, folder_name=PROJECT_NAME)


def test_batch_rejects_non_string_non_numeric_simulation_keys():
    sims = {("task",): make_sim()}

    with pytest.raises(ValidationError, match="Use explicit string keys"):
        Batch(simulations=sims, folder_name=PROJECT_NAME)


@responses.activate
def test_create_output_dirs(mock_webapi, tmp_path, monkeypatch):
    """Test that Job and Batch create output directories if they don't exist."""
    monkeypatch.setattr(f"{api_path}.load", lambda *args, **kwargs: True)
    non_existent_dirs_job = tmp_path / "new/nested/folders/job"
    output_file_job = non_existent_dirs_job / "output.hdf5"

    assert not non_existent_dirs_job.exists()

    sim = make_sim()
    job = Job(simulation=sim, task_name=TASK_NAME, folder_name=PROJECT_NAME)
    job.run(path=str(output_file_job))

    assert non_existent_dirs_job.exists()
    assert non_existent_dirs_job.is_dir()

    non_existent_dirs_batch = tmp_path / "new/nested/folders/batch"

    assert not non_existent_dirs_batch.exists()

    sims = {TASK_NAME: make_sim()}
    batch = Batch(simulations=sims, folder_name=PROJECT_NAME)
    batch.run(path_dir=str(non_existent_dirs_batch))

    assert non_existent_dirs_batch.exists()
    assert non_existent_dirs_batch.is_dir()


@responses.activate
def test_batch_run_saves_file_after_upload(mock_webapi, mock_job_status, tmp_path, monkeypatch):
    """Test that batch.run() saves batch file with task_ids immediately after upload."""
    sims = {TASK_NAME: make_sim()}
    batch = Batch(simulations=sims, folder_name=PROJECT_NAME)

    batch_file_saved = {"saved": False, "has_task_ids": False}
    original_to_file = Batch.to_file

    def track_to_file(self, fname):
        batch_file_saved["saved"] = True
        batch_file_saved["has_task_ids"] = self.jobs is not None and TASK_NAME in self.jobs
        return original_to_file(self, fname)

    # mock monitor to interrupt run() after upload/start and to_file
    def mock_monitor_interrupt(self, *args, **kwargs):
        # at this point, upload/start and to_file() should have been called
        assert batch_file_saved["saved"], "Batch file should be saved before monitor()"
        assert batch_file_saved["has_task_ids"], "Batch file should have task_ids"
        # verify file actually exists and can be loaded
        batch_path = self._batch_path(path_dir=str(tmp_path))
        assert os.path.exists(batch_path)
        recovered = Batch.from_file(batch_path)
        assert recovered.jobs[TASK_NAME].task_id == TASK_ID
        raise RuntimeError("Simulated interruption after upload")

    monkeypatch.setattr(Batch, "to_file", track_to_file)
    monkeypatch.setattr(Batch, "monitor", mock_monitor_interrupt)

    # run should save the batch file after upload, even if interrupted
    with pytest.raises(RuntimeError, match="Simulated interruption"):
        batch.run(path_dir=str(tmp_path))


def test_batch_run_saves_file_when_upload_and_start_fails(tmp_path, monkeypatch):
    sims = {TASK_NAME: make_sim()}
    batch = Batch(simulations=sims, folder_name=PROJECT_NAME)

    batch_file_saved = {"saved": False}
    original_to_file = Batch.to_file

    def track_to_file(self, fname):
        batch_file_saved["saved"] = True
        return original_to_file(self, fname)

    def mock_upload_and_start_fail(self, *args, **kwargs):
        raise RuntimeError("Simulated failure during upload/start")

    monkeypatch.setattr(Batch, "to_file", track_to_file)
    monkeypatch.setattr(Batch, "_upload_and_start", mock_upload_and_start_fail)

    with pytest.raises(RuntimeError, match="Simulated failure during upload/start"):
        batch.run(path_dir=str(tmp_path))

    assert batch_file_saved["saved"]
    assert os.path.exists(batch._batch_path(path_dir=str(tmp_path)))


def test_batch_run_surfaces_to_file_error_when_upload_and_start_succeeds(tmp_path, monkeypatch):
    sims = {TASK_NAME: make_sim()}
    batch = Batch(simulations=sims, folder_name=PROJECT_NAME)

    monkeypatch.setattr(Batch, "_upload_and_start", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        Batch,
        "to_file",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("to_file failed")),
    )
    monkeypatch.setattr(
        Batch,
        "monitor",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("monitor should not run")),
    )

    with pytest.raises(RuntimeError, match="to_file failed"):
        batch.run(path_dir=str(tmp_path))


def test_batch_monitor_downloads_on_success(monkeypatch, tmp_path):
    events = []

    monkeypatch.setattr("tidy3d.web.api.container.ThreadPoolExecutor", ImmediateExecutor)
    monkeypatch.setattr("tidy3d.web.api.container.time.sleep", lambda *_args, **_kwargs: None)

    sims = {"task_a": make_sim(), "task_b": make_sim()}
    batch = Batch(simulations=sims, folder_name=PROJECT_NAME, verbose=False)
    batch._cached_properties = {}
    fake_jobs = {
        "task_a": FakeJob("task_a_id", ["running", "success", "success"], events),
        "task_b": FakeJob("task_b_id", ["running", "running", "success"], events),
    }
    batch._cached_properties["jobs"] = fake_jobs

    batch.monitor(download_on_success=True, path_dir=str(tmp_path))

    downloads = [event for event in events if event[1] == "download"]
    assert len(downloads) == 2
    assert {event[0] for event in downloads} == {"task_a_id", "task_b_id"}

    expected_paths = {
        "task_a_id": os.path.join(str(tmp_path), "task_a_id.hdf5"),
        "task_b_id": os.path.join(str(tmp_path), "task_b_id.hdf5"),
    }

    for task_id, _, path in downloads:
        assert str(path) == expected_paths[task_id]

    job1_download_idx = next(
        i
        for i, event in enumerate(events)
        if event == ("task_a_id", "download", expected_paths["task_a_id"])
    )
    job2_success_idx = next(
        i for i, event in enumerate(events) if event == ("task_b_id", "status", "success")
    )

    assert job1_download_idx < job2_success_idx, "Download should start before other jobs finish"


def test_batch_monitor_does_not_repoll_completed_jobs(monkeypatch, tmp_path):
    events = []

    monkeypatch.setattr("tidy3d.web.api.container.ThreadPoolExecutor", ImmediateExecutor)
    monkeypatch.setattr("tidy3d.web.api.container.time.sleep", lambda *_args, **_kwargs: None)

    sims = {"done_task": make_sim(), "slow_task": make_sim()}
    batch = Batch(simulations=sims, folder_name=PROJECT_NAME, verbose=False)
    batch._cached_properties = {}
    batch._cached_properties["jobs"] = {
        "done_task": FakeJob("done_id", ["success", "success"], events),
        "slow_task": FakeJob("slow_id", ["running", "running", "success"], events),
    }

    batch.monitor(download_on_success=False, path_dir=str(tmp_path))

    done_status_calls = sum(1 for event in events if event == ("done_id", "status", "success"))
    slow_status_calls = sum(1 for event in events if event[0] == "slow_id" and event[1] == "status")
    assert done_status_calls == 1
    assert slow_status_calls >= 3


def test_batch_upload_and_start_starts_each_task_after_upload(monkeypatch):
    events = []

    monkeypatch.setattr("tidy3d.web.api.container.ThreadPoolExecutor", ImmediateExecutor)
    monkeypatch.setattr(Batch, "_check_folder", staticmethod(lambda *args, **kwargs: None))

    sims = {"task_a": make_sim(), "task_b": make_sim()}
    batch = Batch(simulations=sims, folder_name=PROJECT_NAME, verbose=False)
    batch._cached_properties = {}
    batch._cached_properties["jobs"] = {
        "task_a": UploadStartFakeJob("task_a_id", events),
        "task_b": UploadStartFakeJob("task_b_id", events),
    }

    batch._upload_and_start(priority=6)

    assert events[:2] == [
        ("task_a_id", "upload"),
        ("task_b_id", "upload"),
    ]
    assert sorted(events[2:]) == sorted(
        [
            ("task_a_id", "start", 6),
            ("task_b_id", "start", 6),
        ]
    )


def test_batch_upload_and_start_skips_cached_jobs(monkeypatch):
    events = []

    monkeypatch.setattr("tidy3d.web.api.container.ThreadPoolExecutor", ImmediateExecutor)
    monkeypatch.setattr(Batch, "_check_folder", staticmethod(lambda *args, **kwargs: None))

    sims = {"cached_task": make_sim(), "running_task": make_sim()}
    batch = Batch(simulations=sims, folder_name=PROJECT_NAME, verbose=False)
    batch._cached_properties = {}
    batch._cached_properties["jobs"] = {
        "cached_task": UploadStartFakeJob("cached_id", events, cached=True),
        "running_task": UploadStartFakeJob("running_id", events, cached=False),
    }

    batch._upload_and_start(priority=3)

    assert events == [
        ("running_id", "upload"),
        ("running_id", "start", 3),
    ]


def test_batch_upload_and_start_waits_for_metadata_and_disables_blocking_estimate(monkeypatch):
    events = []

    monkeypatch.setattr("tidy3d.web.api.container.ThreadPoolExecutor", ImmediateExecutor)
    monkeypatch.setattr("tidy3d.web.api.container.time.sleep", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(Batch, "_check_folder", staticmethod(lambda *args, **kwargs: None))

    sims = {"task_a": make_sim(), "task_b": make_sim()}
    batch = Batch(simulations=sims, folder_name=PROJECT_NAME, verbose=False)
    batch._cached_properties = {}
    batch._cached_properties["jobs"] = {
        "task_a": UploadEstimateFakeJob(
            "task_a_id",
            events,
            metadata_statuses=["validating", "processed"],
        ),
        "task_b": UploadEstimateFakeJob(
            "task_b_id",
            events,
            metadata_statuses=["validating", "validating", "processed"],
        ),
    }

    batch._upload_and_start(priority=5)

    upload_events = [event for event in events if event[1] == "upload"]
    start_events = [event for event in events if event[1] == "start"]
    metadata_events = [event for event in events if event[1] == "metadata"]

    assert upload_events == [
        ("task_a_id", "upload", False),
        ("task_b_id", "upload", False),
    ]
    assert sorted(start_events) == sorted(
        [
            ("task_a_id", "start", 5),
            ("task_b_id", "start", 5),
        ]
    )
    assert len(metadata_events) >= 3

    first_start_idx = min(i for i, event in enumerate(events) if event[1] == "start")
    last_upload_idx = max(i for i, event in enumerate(events) if event[1] == "upload")
    assert last_upload_idx < first_start_idx


def test_batch_upload_and_start_raises_when_metadata_errors(monkeypatch):
    events = []

    monkeypatch.setattr("tidy3d.web.api.container.ThreadPoolExecutor", ImmediateExecutor)
    monkeypatch.setattr("tidy3d.web.api.container.time.sleep", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(Batch, "_check_folder", staticmethod(lambda *args, **kwargs: None))

    sims = {"task_a": make_sim()}
    batch = Batch(simulations=sims, folder_name=PROJECT_NAME, verbose=False)
    batch._cached_properties = {}
    batch._cached_properties["jobs"] = {
        "task_a": UploadEstimateFakeJob(
            "task_a_id",
            events,
            metadata_statuses=["error"],
        ),
    }

    with pytest.raises(DataError, match="Failed cost estimation before start"):
        batch._upload_and_start(priority=4)

    assert ("task_a_id", "start", 4) not in events


def test_batch_upload_and_start_streams_ready_start_before_all_uploads(monkeypatch):
    events = []

    monkeypatch.setattr("tidy3d.web.api.container.ThreadPoolExecutor", ImmediateExecutor)
    monkeypatch.setattr("tidy3d.web.api.container.time.sleep", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(Batch, "_check_folder", staticmethod(lambda *args, **kwargs: None))

    sims = {"task_a": make_sim(), "task_b": make_sim()}
    batch = Batch(simulations=sims, folder_name=PROJECT_NAME, verbose=False, num_workers=1)
    batch._cached_properties = {}
    batch._cached_properties["jobs"] = {
        "task_a": UploadEstimateFakeJob(
            "task_a_id",
            events,
            metadata_statuses=["processed"],
        ),
        "task_b": UploadEstimateFakeJob(
            "task_b_id",
            events,
            metadata_statuses=["processed"],
        ),
    }

    batch._upload_and_start(priority=9)

    upload_indices = [i for i, event in enumerate(events) if event[1] == "upload"]
    start_indices = [i for i, event in enumerate(events) if event[1] == "start"]
    assert upload_indices
    assert start_indices
    assert min(start_indices) < max(upload_indices)


def test_batch_upload_and_start_respects_num_workers_bound(monkeypatch):
    events = []
    max_active_futures = [0]
    original_wait = concurrent.futures.wait

    def tracking_wait(fs, *args, **kwargs):
        max_active_futures[0] = max(max_active_futures[0], len(fs))
        return original_wait(fs, *args, **kwargs)

    monkeypatch.setattr("tidy3d.web.api.container.ThreadPoolExecutor", ImmediateExecutor)
    monkeypatch.setattr("tidy3d.web.api.container.concurrent.futures.wait", tracking_wait)
    monkeypatch.setattr("tidy3d.web.api.container.time.sleep", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(Batch, "_check_folder", staticmethod(lambda *args, **kwargs: None))

    sims = {f"task_{idx}": make_sim() for idx in range(6)}
    batch = Batch(simulations=sims, folder_name=PROJECT_NAME, verbose=False, num_workers=2)
    batch._cached_properties = {}
    batch._cached_properties["jobs"] = {
        task_name: UploadEstimateFakeJob(
            f"{task_name}_id",
            events,
            metadata_statuses=["processed"],
        )
        for task_name in sims
    }

    batch._upload_and_start(priority=3)

    assert max_active_futures[0] <= batch.num_workers


def test_batch_monitor_skips_existing_download(monkeypatch, tmp_path):
    events = []

    monkeypatch.setattr("tidy3d.web.api.container.ThreadPoolExecutor", ImmediateExecutor)
    monkeypatch.setattr("tidy3d.web.api.container.time.sleep", lambda *_args, **_kwargs: None)

    sims = {"task_a": make_sim(), "task_b": make_sim()}
    batch = Batch(simulations=sims, folder_name=PROJECT_NAME, verbose=False)
    batch._cached_properties = {}
    fake_jobs = {
        "task_a": FakeJob("task_a_id", ["success", "success"], events),
        "task_b": FakeJob("task_b_id", ["running", "success"], events),
    }
    batch._cached_properties["jobs"] = fake_jobs

    existing_path = os.path.join(str(tmp_path), "task_a_id.hdf5")
    with open(existing_path, "w", encoding="utf8") as handle:
        handle.write("cached")

    batch.monitor(download_on_success=True, path_dir=str(tmp_path))

    downloads = [event for event in events if event[1] == "download"]

    assert downloads == [("task_b_id", "download", os.path.join(str(tmp_path), "task_b_id.hdf5"))]


def test_batch_monitor_skips_get_info_for_cached(monkeypatch, tmp_path):
    """Cached jobs must not trigger get_info() remote calls during monitoring."""
    events = []

    monkeypatch.setattr("tidy3d.web.api.container.ThreadPoolExecutor", ImmediateExecutor)
    monkeypatch.setattr("tidy3d.web.api.container.time.sleep", lambda *_args, **_kwargs: None)

    class CachedFakeJob(FakeJob):
        @property
        def load_if_cached(self):
            return True

        def get_info(self):
            raise AssertionError("get_info() should not be called for cached jobs")

    sims = {"cached_task": make_sim(), "running_task": make_sim()}
    batch = Batch(simulations=sims, folder_name=PROJECT_NAME, verbose=False)
    batch._cached_properties = {}
    fake_jobs = {
        "cached_task": CachedFakeJob("cached_id", ["success"], events),
        "running_task": FakeJob("running_id", ["running", "success", "success"], events),
    }
    batch._cached_properties["jobs"] = fake_jobs

    batch.monitor(download_on_success=True, path_dir=str(tmp_path))

    # Cached job would have raised AssertionError if get_info() was called.
    # Verify the non-cached job still downloaded normally.
    downloads = [e for e in events if e[1] == "download"]
    assert any(e[0] == "running_id" for e in downloads)


def test_batch_monitor_quiet_mode_skips_status_poll_for_cached(monkeypatch, tmp_path):
    """Quiet monitor mode should not access ``status`` for cached jobs."""
    events = []

    monkeypatch.setattr("tidy3d.web.api.container.ThreadPoolExecutor", ImmediateExecutor)
    monkeypatch.setattr("tidy3d.web.api.container.time.sleep", lambda *_args, **_kwargs: None)

    class CachedNoStatusJob(FakeJob):
        @property
        def load_if_cached(self):
            return True

        @property
        def status(self):
            raise AssertionError("status should not be polled for cached jobs")

        def get_info(self):
            raise AssertionError("get_info() should not be called for cached jobs")

    sims = {"cached_task": make_sim(), "running_task": make_sim()}
    batch = Batch(simulations=sims, folder_name=PROJECT_NAME, verbose=False)
    batch._cached_properties = {}
    batch._cached_properties["jobs"] = {
        "cached_task": CachedNoStatusJob("cached_id", ["success"], events),
        "running_task": FakeJob("running_id", ["running", "success", "success"], events),
    }

    batch.monitor(download_on_success=True, path_dir=str(tmp_path))

    # Cached job would have raised AssertionError if status/get_info was touched.
    downloads = [e for e in events if e[1] == "download"]
    assert any(e[0] == "running_id" for e in downloads)


def test_batch_download_surfaces_download_errors(monkeypatch, tmp_path):
    monkeypatch.setattr("tidy3d.web.api.container.Job.status", property(lambda self: "success"))
    monkeypatch.setattr("tidy3d.web.api.container.Job.load_if_cached", property(lambda self: False))
    monkeypatch.setattr("tidy3d.web.api.container.Job.task_id", property(lambda self: "task_a_id"))

    def _raise_download(self, path):
        raise RuntimeError("gzip extraction failed")

    monkeypatch.setattr("tidy3d.web.api.container.Job.download", _raise_download)

    sims = {"task_a": make_sim()}
    batch = Batch(simulations=sims, folder_name=PROJECT_NAME, verbose=False)

    with pytest.raises(RuntimeError, match="gzip extraction failed"):
        batch.download(path_dir=str(tmp_path))


def test_batch_upload_surfaces_upload_errors(monkeypatch):
    monkeypatch.setattr("tidy3d.web.api.container.ThreadPoolExecutor", ImmediateExecutor)
    monkeypatch.setattr(Batch, "_check_folder", staticmethod(lambda *args, **kwargs: None))
    monkeypatch.setattr("tidy3d.web.api.container.Job.load_if_cached", property(lambda self: False))

    def _raise_upload(self):
        raise RuntimeError("upload failed")

    monkeypatch.setattr("tidy3d.web.api.container.Job.upload", _raise_upload)

    sims = {"task_a": make_sim()}
    batch = Batch(simulations=sims, folder_name=PROJECT_NAME, verbose=False)

    with pytest.raises(RuntimeError, match="upload failed"):
        batch.upload()


def test_batch_load_parallel_status_collection(monkeypatch, tmp_path):
    max_workers_used = []
    submit_calls = []
    warning_messages = []

    class CapturingExecutor(ImmediateExecutor):
        def __init__(self, *args, **kwargs):
            max_workers_used.append(kwargs.get("max_workers"))

        def submit(self, fn, *args, **kwargs):
            submit_calls.append((fn, args, kwargs))
            return super().submit(fn, *args, **kwargs)

    monkeypatch.setattr("tidy3d.web.api.container.ThreadPoolExecutor", CapturingExecutor)
    monkeypatch.setattr(
        "tidy3d.web.api.container.log.warning", lambda msg: warning_messages.append(msg)
    )

    sims = {"ok_task": make_sim(), "bad_task": make_sim()}
    batch = Batch(simulations=sims, folder_name=PROJECT_NAME, verbose=False)
    batch._cached_properties = {
        "jobs": {
            "ok_task": LoadStatusFakeJob("ok_task_id", "success", sims["ok_task"]),
            "bad_task": LoadStatusFakeJob("bad_task_id", "error", sims["bad_task"]),
        }
    }

    data = batch.load(path_dir=str(tmp_path), skip_download=True)

    assert max_workers_used == [batch.num_workers]
    assert len(submit_calls) == 2
    assert data.task_ids == {"ok_task": "ok_task_id"}
    assert set(data.task_paths.keys()) == {"ok_task"}
    assert warning_messages == ["Not loading 'bad_task' as the task errored."]


def test_batch_load_reuses_terminal_status_snapshot(monkeypatch, tmp_path):
    events = []

    monkeypatch.setattr("tidy3d.web.api.container.ThreadPoolExecutor", ImmediateExecutor)
    monkeypatch.setattr("tidy3d.web.api.container.time.sleep", lambda *_args, **_kwargs: None)

    sims = {"task_a": make_sim(), "task_b": make_sim()}
    batch = Batch(simulations=sims, folder_name=PROJECT_NAME, verbose=False)
    batch._cached_properties = {}
    batch._cached_properties["jobs"] = {
        "task_a": FakeJobWithSimulation(
            "task_a_id",
            ["running", "success", "success"],
            events,
            sims["task_a"],
        ),
        "task_b": FakeJobWithSimulation(
            "task_b_id",
            ["running", "running", "success", "success"],
            events,
            sims["task_b"],
        ),
    }

    batch.monitor(download_on_success=False, path_dir=str(tmp_path))
    status_calls_before_load = sum(1 for event in events if event[1] == "status")

    _ = batch.load(path_dir=str(tmp_path), skip_download=True)
    status_calls_after_load = sum(1 for event in events if event[1] == "status")

    assert status_calls_after_load == status_calls_before_load


def test_batch_load_does_not_upload_unknown_tasks(monkeypatch, tmp_path):
    monkeypatch.setattr("tidy3d.web.api.container.ThreadPoolExecutor", ImmediateExecutor)
    monkeypatch.setattr("tidy3d.web.api.container.Job.load_if_cached", property(lambda self: False))

    def _raise_upload(self, *args, **kwargs):
        raise AssertionError("Batch.load() should not upload tasks.")

    monkeypatch.setattr("tidy3d.web.api.container.Job._upload", _raise_upload)

    sims = {"task_a": make_sim()}
    batch = Batch(simulations=sims, folder_name=PROJECT_NAME, verbose=False)

    with pytest.raises(DataError, match="task hasn't been uploaded"):
        batch.load(path_dir=str(tmp_path), skip_download=True)


""" Async """


@responses.activate
@pytest.mark.parametrize("task_name", [TASK_NAME, None])
def test_async(mock_webapi, mock_job_status, tmp_path, task_name):
    # monkeypatch.setattr("tidy3d.web.api.container.Job.status", property(lambda self: "success"))
    sims = {TASK_NAME: make_sim()} if task_name else [make_sim()]
    _ = run_async(sims, folder_name=PROJECT_NAME, path_dir=str(tmp_path))


def test_async_forwards_num_workers(monkeypatch):
    captured_kwargs = {}
    captured_run_kwargs = {}

    class DummyBatch:
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)

        def run(self, **kwargs):
            captured_run_kwargs.update(kwargs)
            return {}

    monkeypatch.setattr("tidy3d.web.api.asynchronous.Batch", DummyBatch)

    sims = {TASK_NAME: make_sim()}
    _ = run_async(sims, folder_name=PROJECT_NAME, path_dir=".", num_workers=7, verbose=False)

    assert captured_kwargs["num_workers"] == 7
    assert captured_run_kwargs["path_dir"] == "."


def test_async_omits_num_workers_when_not_provided(monkeypatch):
    captured_kwargs = {}

    class DummyBatch:
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)

        def run(self, **kwargs):
            return {}

    monkeypatch.setattr("tidy3d.web.api.asynchronous.Batch", DummyBatch)

    sims = {TASK_NAME: make_sim()}
    _ = run_async(sims, folder_name=PROJECT_NAME, path_dir=".", verbose=False)

    assert "num_workers" not in captured_kwargs


def test_batch_run_uses_optimized_upload_and_start(monkeypatch, tmp_path):
    class RunFakeJob:
        @property
        def load_if_cached(self):
            return False

    upload_start_calls = {"count": 0, "priority": None}
    run_calls = []
    load_kwargs = {}

    def _raise_legacy_upload(*args, **kwargs):
        raise AssertionError("Batch.run should not call legacy Batch.upload().")

    def _raise_legacy_start(*args, **kwargs):
        raise AssertionError("Batch.run should not call legacy Batch.start().")

    def _track_upload_and_start(self, priority=None):
        upload_start_calls["count"] += 1
        upload_start_calls["priority"] = priority

    def _fake_load(self, **kwargs):
        load_kwargs.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(Batch, "upload", _raise_legacy_upload)
    monkeypatch.setattr(Batch, "start", _raise_legacy_start)
    monkeypatch.setattr(Batch, "_upload_and_start", _track_upload_and_start)
    monkeypatch.setattr(Batch, "to_file", lambda *args, **kwargs: run_calls.append("to_file"))
    monkeypatch.setattr(Batch, "monitor", lambda *args, **kwargs: run_calls.append("monitor"))
    monkeypatch.setattr(Batch, "load", _fake_load)

    batch = Batch(simulations={"task_a": make_sim()}, folder_name=PROJECT_NAME, verbose=False)
    batch._cached_properties = {"jobs": {"task_a": RunFakeJob()}}

    result = batch.run(path_dir=str(tmp_path), priority=8)

    assert result == {"ok": True}
    assert upload_start_calls["count"] == 1
    assert upload_start_calls["priority"] == 8
    assert run_calls == ["to_file", "monitor"]
    assert load_kwargs == {"path_dir": str(tmp_path), "skip_download": True}


""" Main """


@responses.activate
def test_main(mock_webapi, monkeypatch, mock_job_status, tmp_path):
    # sims = {TASK_NAME: make_sim()}
    # batch_data = run_async(sims, folder_name=PROJECT_NAME)

    def save_sim_to_path(path: str) -> None:
        sim = make_sim()
        sim.to_file(path)

    monkeypatch.setattr("builtins.input", lambda _: "Y")

    path = str(tmp_path / "sim.json")
    save_sim_to_path(path)
    main(
        [
            path,
            "--task_name",
            TASK_NAME,
            "--folder_name",
            PROJECT_NAME,
            "--inspect_credits",
            "--inspect_sim",
            "-o",
            str(tmp_path / "tmp.hdf5"),
        ]
    )

    monkeypatch.setattr("builtins.input", lambda _: "N")
    with pytest.raises(SystemExit):
        main(
            [
                path,
                "--task_name",
                TASK_NAME,
                "--folder_name",
                PROJECT_NAME,
                "--inspect_credits",
                "-o",
                str(tmp_path / "tmp.hdf5"),
            ]
        )

    with pytest.raises(SystemExit):
        main(
            [
                path,
                "--task_name",
                TASK_NAME,
                "--folder_name",
                PROJECT_NAME,
                "--inspect_sim",
                "-o",
                str(tmp_path / "tmp.hdf5"),
            ]
        )


@responses.activate
def test_load_invalid_task_raises(mock_webapi):
    """Ensure that load() raises TaskNotFoundError for a non-existent task ID."""

    responses.add(
        responses.GET,
        f"{Env.current.web_api_endpoint}/tidy3d/tasks/{INVALID_TASK_ID}/detail",
        json={"error": "Task not found"},
        status=404,
    )
    with pytest.raises(WebNotFoundError, match="Resource not found"):
        load(INVALID_TASK_ID, replace_existing=True)


def _fake_load_factory(tmp_root, taskid_to_sim: dict):
    def _fake_load(task_id, path="simulation_data.hdf5", lazy=False, **kwargs):
        abs_path = path if os.path.isabs(path) else os.path.join(tmp_root, path)
        abs_path = os.path.normpath(abs_path)
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)

        sim_for_this = taskid_to_sim.get(task_id)

        log = "- Time step    827 / time 4.13e-14s (  4 % done), field decay: 0.110e+00"
        sim_data = SimulationData(simulation=sim_for_this, data=[], diverged=False, log=log)

        sim_data.to_file(abs_path)
        return Tidy3dStubData.postprocess(abs_path, lazy=lazy)

    return _fake_load


def apply_common_patches(
    monkeypatch,
    tmp_root,
    *,
    api_path="tidy3d.web.api.webapi",
    taskid_to_sim=None,
):
    """Patch start/monitor/get_info/estimate_cost/upload/_check_folder/_modesolver_patch/load."""
    monkeypatch.setattr(f"{api_path}.start", lambda *a, **k: True)
    monkeypatch.setattr(f"{api_path}.monitor", lambda *a, **k: True)

    # --- make get_info return also task type ---
    def _fake_get_info(task_id: str, *_, **__):
        sim = taskid_to_sim.get(task_id) if taskid_to_sim else None
        task_type = task_type_name_of(sim) if sim is not None else None
        return SimpleNamespace(status="success", taskType=task_type)

    monkeypatch.setattr(f"{api_path}.get_info", _fake_get_info)

    # other patches
    def fake_estimate_cost(*args, **kwargs):
        verbose = kwargs.pop("verbose", True)
        if verbose:
            print("estimate cost")
        return 0.0

    monkeypatch.setattr(f"{api_path}.estimate_cost", fake_estimate_cost)

    def fake_upload(*args, **kwargs):
        verbose = kwargs.pop("verbose", True)
        verbose_estimate_cost = kwargs.pop("verbose_estimate_cost", None)
        verbose_estimate_cost = verbose if verbose_estimate_cost is None else verbose_estimate_cost
        fake_estimate_cost(verbose=verbose_estimate_cost)
        return kwargs["task_name"]

    monkeypatch.setattr(f"{api_path}.upload", fake_upload)
    monkeypatch.setattr(WebContainer, "_check_folder", lambda *a, **k: True)
    monkeypatch.setattr(f"{api_path}._modesolver_patch", lambda *_, **__: None, raising=False)
    monkeypatch.setattr(f"{api_path}.download", lambda *_, **__: None, raising=False)
    monkeypatch.setattr(
        f"{api_path}.load",
        _fake_load_factory(tmp_root=str(tmp_root), taskid_to_sim=taskid_to_sim),
        raising=False,
    )


@responses.activate
def test_run_with_flexible_containers_offline_lazy(monkeypatch, tmp_path):
    sim1 = make_sim()
    sim2 = sim1.updated_copy(run_time=sim1.run_time / 2)
    sim_container = [sim1, {"sim": sim1, "sim2": sim2}, (sim1, [sim2])]

    h2sim = _collect_by_hash(sim_container)
    task_name = "T"
    out_dir = tmp_path / "out"

    taskid_to_sim = {f"{task_name}_{h}": s for h, s in h2sim.items()}

    apply_common_patches(monkeypatch, tmp_path, taskid_to_sim=taskid_to_sim)

    data = run(sim_container, task_name=task_name, folder_name="PROJECT", path=str(out_dir))
    assert is_lazy_object(data[0])
    assert isinstance(data, list) and len(data) == 3

    assert isinstance(data[1], dict)
    assert "sim2" in data[1]
    assert is_lazy_object(data[1]["sim2"])
    assert isinstance(data[1]["sim2"], SimulationData)

    assert is_lazy_object(data[2][0])
    assert isinstance(data[2], tuple)
    assert is_lazy_object(data[2][1][0])
    assert isinstance(data[2][1], list)

    assert data[0].simulation == sim1
    assert data[1]["sim2"].simulation == sim2


@responses.activate
def test_run_single_offline_eager(monkeypatch, tmp_path):
    sim = make_sim()
    single_file = str(tmp_path / "sim.hdf5")
    task_name = "single"
    apply_common_patches(monkeypatch, tmp_path, taskid_to_sim={task_name: sim})

    sim_data = run(sim, task_name=task_name, path=single_file)

    assert isinstance(sim_data, SimulationData)
    assert sim_data.__class__.__name__ == "SimulationData"  # no proxy


class FauxPath:
    """Minimal PathLike to exercise __fspath__ support."""

    def __init__(self, path: PathLike | str):
        self._p = os.fspath(path)

    def __fspath__(self) -> str:
        return self._p


def _pathlib_builder(tmp_path, name: str):
    return Path(tmp_path) / name


def _posix_builder(tmp_path, name: str):
    return posixpath.join(tmp_path.as_posix(), name)


def _str_builder(tmp_path, name: str):
    return str(Path(tmp_path) / name)


def _fspath_builder(tmp_path, name: str):
    return FauxPath(Path(tmp_path) / name)


@pytest.mark.parametrize(
    "path_builder",
    [_pathlib_builder, _posix_builder, _str_builder, _fspath_builder],
    ids=["pathlib.Path", "posixpath_str", "str", "PathLike"],
)
def test_run_single_offline_eager_accepts_pathlikes(monkeypatch, tmp_path, path_builder):
    """run(sim, path=...) accepts any PathLike."""
    sim = make_sim()
    task_name = "pathlike_single"
    out_file = path_builder(tmp_path, "sim.hdf5")

    # Patch webapi for offline run and to write to the provided path
    apply_common_patches(monkeypatch, tmp_path, taskid_to_sim={task_name: sim})

    sim_data = run(sim, task_name=task_name, path=out_file)

    # File existed (written via patched load) and types are correct
    assert os.path.exists(os.fspath(out_file))
    assert isinstance(sim_data, SimulationData)
    assert sim_data.simulation == sim


@pytest.mark.parametrize(
    "path_builder",
    [_pathlib_builder, _posix_builder, _str_builder, _fspath_builder],
    ids=["pathlib.Path", "posixpath_str", "str", "PathLike"],
)
def test_job_run_accepts_pathlikes(monkeypatch, tmp_path, path_builder):
    """Job.run(path=...) accepts any PathLike."""
    sim = make_sim()
    task_name = "job_pathlike"
    out_file = path_builder(tmp_path, "job_out.hdf5")

    apply_common_patches(monkeypatch, tmp_path, taskid_to_sim={task_name: sim})

    j = Job(simulation=sim, task_name=task_name, folder_name=PROJECT_NAME)
    _ = j.run(path=out_file)

    assert os.path.exists(os.fspath(out_file))


@pytest.mark.parametrize(
    "dir_builder",
    [_pathlib_builder, _posix_builder, _str_builder, _fspath_builder],
    ids=["pathlib.Path", "posixpath_str", "str", "PathLike"],
)
@pytest.mark.slow
def test_batch_run_accepts_pathlike_dir(monkeypatch, tmp_path, dir_builder):
    """Batch.run(path_dir=...) accepts any PathLike directory location."""
    sims = {"A": make_sim()}
    out_dir = dir_builder(tmp_path, "batch_out")

    # Map task_ids to sims: upload() is patched to return task_name, which for dict input
    # corresponds to the dict keys ("A", "B"), so we map those.
    apply_common_patches(monkeypatch, tmp_path, taskid_to_sim={"A": sims["A"]})

    b = Batch(simulations=sims, folder_name=PROJECT_NAME)
    b.run(path_dir=out_dir)

    # Directory created and .hdf5 output produced
    out_dir_str = os.fspath(out_dir)
    assert os.path.isdir(out_dir_str)

    batch_file = Path(out_dir) / "batch.hdf5"
    assert batch_file.is_file()


def test_job_estimate_cost_logging(monkeypatch, tmp_path, capsys):
    def assert_estimate_cost_prints(count: int) -> None:
        out, err = capsys.readouterr()
        assert out.count("estimate cost") == count, (
            f"expected {count}, got {out.count('estimate cost')}\nout: {out}"
        )

    sim = make_sim()
    apply_common_patches(monkeypatch, tmp_path, taskid_to_sim={"task": sim})

    job = Job(simulation=sim, task_name=TASK_NAME)

    # accessing task_id should NOT print
    _ = job.task_id
    assert_estimate_cost_prints(0)

    # upload should print
    job.upload()
    assert_estimate_cost_prints(1)

    # test web estimate cost
    td.web.api.webapi.estimate_cost(job.task_id, verbose=True)
    assert_estimate_cost_prints(1)

    td.web.api.webapi.estimate_cost(job.task_id, verbose=False)
    assert_estimate_cost_prints(0)

    # test job estimate cost
    job.estimate_cost()
    assert_estimate_cost_prints(1)

    job.estimate_cost(verbose=False)
    assert_estimate_cost_prints(0)
