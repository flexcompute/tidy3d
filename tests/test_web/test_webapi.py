# Tests webapi and things that depend on it
from __future__ import annotations

import json
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
from tidy3d.exceptions import SetupError
from tidy3d.web import common
from tidy3d.web.api.asynchronous import run_async
from tidy3d.web.api.container import (
    UPLOAD_START_NUM_WORKERS,
    Batch,
    BatchData,
    Job,
    WebContainer,
)
from tidy3d.web.api.run import _collect_by_hash, run
from tidy3d.web.api.run_options import log_deprecated_run_args
from tidy3d.web.api.tidy3d_stub import Tidy3dStubData, task_type_name_of
from tidy3d.web.api.webapi import (
    abort,
    default_data_filename,
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
from tidy3d.web.core.constants import MODE_DATA_HDF5_GZ
from tidy3d.web.core.environment import Env
from tidy3d.web.core.exceptions import WebNotFoundError
from tidy3d.web.core.task_core import BatchTask
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

    @property
    def task_id_cached(self):
        return self.task_id

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

    @property
    def task_id_cached(self):
        return self.task_id


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

    def add_mock_response(priority=None, vgpu_allocation=None, ignore_memory_limit=None):
        expected_body = {
            "solverVersion": None,
            "workerGroup": None,
            "protocolVersion": td.version.__version__,
            "enableCaching": Env.current.enable_caching,
            "payType": PayType.AUTO,
            "priority": priority,
            "vgpuAllocation": vgpu_allocation,
            "ignoreMemoryLimit": ignore_memory_limit,
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

    # Add response for calls without priority, vgpu_allocation, or ignore_memory_limit
    add_mock_response(None)

    # Add responses for calls with specific priority values
    for priority in [1, 5, 10]:
        add_mock_response(priority=priority)

    # Add responses for calls with specific vgpu_allocation values
    for vgpu_alloc in [1, 2, 4, 8]:
        add_mock_response(vgpu_allocation=vgpu_alloc)

    # Add responses for calls with ignore_memory_limit
    add_mock_response(ignore_memory_limit=True)
    add_mock_response(ignore_memory_limit=False)


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
def reset_run_option_config():
    config.run.solver_version = None
    config.run.worker_group = None
    config.run.simulation_type = "tidy3d"
    config.run.additional_payload = None
    config.run.pay_type = PayType.AUTO
    config.vgpu.priority = None
    config.vgpu.vgpu_allocation = None
    config.vgpu.ignore_memory_limit = None
    yield
    config.run.solver_version = None
    config.run.worker_group = None
    config.run.simulation_type = "tidy3d"
    config.run.additional_payload = None
    config.run.pay_type = PayType.AUTO
    config.vgpu.priority = None
    config.vgpu.vgpu_allocation = None
    config.vgpu.ignore_memory_limit = None


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


def test_upload_logs_deprecated_run_args(monkeypatch):
    warning_calls = []

    monkeypatch.setattr(
        "tidy3d.web.api.webapi.log_deprecated_run_args",
        lambda **kwargs: warning_calls.append(kwargs),
    )
    monkeypatch.setattr(
        "tidy3d.web.api.webapi.resolve_upload_options",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("stop_after_warning")),
    )

    with pytest.raises(RuntimeError, match="stop_after_warning"):
        upload(
            make_sim(),
            TASK_NAME,
            PROJECT_NAME,
            simulation_type="special_type",
            solver_version="solver_x",
        )

    assert warning_calls == [
        {
            "solver_version": "solver_x",
        }
    ]


@responses.activate
def test_get_info(mock_get_info):
    assert get_info(TASK_ID).taskId == TASK_ID


@responses.activate
def test_start(mock_start):
    start(TASK_ID)


def test_start_logs_deprecated_run_args(monkeypatch):
    warning_calls = []

    monkeypatch.setattr(
        "tidy3d.web.api.webapi.log_deprecated_run_args",
        lambda **kwargs: warning_calls.append(kwargs),
    )
    monkeypatch.setattr(
        "tidy3d.web.api.webapi.TaskFactory.get",
        lambda task_id: td.web.core.task_core.SimulationTask(taskId=task_id),
    )
    monkeypatch.setattr(
        "tidy3d.web.api.webapi.resolve_run_start_options",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("stop_after_warning")),
    )

    with pytest.raises(RuntimeError, match="stop_after_warning"):
        start(
            TASK_ID,
            solver_version="solver_x",
            worker_group="worker_a",
            pay_type=PayType.CREDITS,
            priority=4,
            vgpu_allocation=2,
            ignore_memory_limit=True,
        )

    assert warning_calls == [
        {
            "solver_version": "solver_x",
            "worker_group": "worker_a",
            "pay_type": PayType.CREDITS,
            "priority": 4,
            "vgpu_allocation": 2,
            "ignore_memory_limit": True,
        }
    ]


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
@pytest.mark.parametrize("vgpu_allocation", [1, 2, 4, 8, None])
def test_start_with_valid_vgpu_allocation(mock_start, vgpu_allocation):
    """Test start with valid vgpu_allocation values."""
    start(TASK_ID, vgpu_allocation=vgpu_allocation)


@responses.activate
@pytest.mark.parametrize("vgpu_allocation", [0, -1, 3, 5, 9])
def test_start_with_invalid_vgpu_allocation(mock_start, vgpu_allocation):
    """Test start with invalid vgpu_allocation values."""
    with pytest.raises(ValueError, match="vgpu_allocation must be one of"):
        start(TASK_ID, vgpu_allocation=vgpu_allocation)


@responses.activate
@pytest.mark.parametrize("vgpu_allocation", [4, None])
def test_run_with_valid_vgpu_allocation(mock_webapi, monkeypatch, vgpu_allocation):
    """Test run with valid vgpu_allocation parameter."""
    monkeypatch.setattr(f"{api_path}.load", lambda *args, **kwargs: True)
    sim = make_sim()
    run(sim, TASK_NAME, folder_name=PROJECT_NAME, vgpu_allocation=vgpu_allocation)


@responses.activate
@pytest.mark.parametrize("vgpu_allocation", [0, -1, 3, 9])
def test_run_with_invalid_vgpu_allocation(mock_webapi, vgpu_allocation):
    """Test run with invalid vgpu_allocation values."""
    sim = make_sim()
    with pytest.raises(ValueError, match="vgpu_allocation must be one of"):
        run(sim, TASK_NAME, folder_name=PROJECT_NAME, vgpu_allocation=vgpu_allocation)


@responses.activate
@pytest.mark.parametrize("ignore_memory_limit", [True, False, None])
def test_start_with_ignore_memory_limit(mock_start, ignore_memory_limit):
    """Test start with ignore_memory_limit values."""
    start(TASK_ID, ignore_memory_limit=ignore_memory_limit)


@responses.activate
def test_upload_uses_run_config_defaults(
    monkeypatch, set_api_key, mock_get_info, reset_run_option_config
):
    sim = make_sim()
    config.run.simulation_type = "config_type"
    config.run.solver_version = "config_solver"

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
                    "simulationType": "config_type",
                    "parentTasks": None,
                    "fileType": "Gz",
                },
                strict_match=False,
            )
        ],
        json={"data": {"taskId": TASK_ID, "taskName": TASK_NAME, "createdAt": CREATED_AT}},
        status=200,
    )
    responses.add(
        responses.POST,
        f"{Env.current.web_api_endpoint}/tidy3d/tasks/{TASK_ID}/metadata",
        match=[
            matchers.json_params_matcher(
                {
                    "solverVersion": "config_solver",
                    "protocolVersion": None,
                }
            )
        ],
        json={"data": {"createdAt": CREATED_AT}},
        status=200,
    )

    monkeypatch.setattr("tidy3d.web.core.task_core.upload_file", lambda *args, **kwargs: None)

    assert upload(sim, TASK_NAME, PROJECT_NAME, verbose=False) == TASK_ID


@responses.activate
def test_start_uses_config_defaults(set_api_key, mock_get_info, reset_run_option_config):
    config.run.solver_version = "config_solver"
    config.run.worker_group = "config_group"
    config.run.additional_payload = {"routeHint": "special"}
    config.run.pay_type = PayType.CREDITS
    config.vgpu.priority = 5
    config.vgpu.vgpu_allocation = 4
    config.vgpu.ignore_memory_limit = True

    responses.add(
        responses.POST,
        f"{Env.current.web_api_endpoint}/tidy3d/tasks/{TASK_ID}/submit",
        match=[
            matchers.json_params_matcher(
                {
                    "solverVersion": "config_solver",
                    "workerGroup": "config_group",
                    "protocolVersion": None,
                    "enableCaching": Env.current.enable_caching,
                    "payType": PayType.CREDITS,
                    "priority": 5,
                    "vgpuAllocation": 4,
                    "ignoreMemoryLimit": True,
                    "additionalPayload": json.dumps({"routeHint": "special"}),
                }
            )
        ],
        json={"data": {"taskId": TASK_ID, "taskName": TASK_NAME, "createdAt": CREATED_AT}},
        status=200,
    )

    start(TASK_ID)


@responses.activate
def test_start_explicit_args_override_config_defaults(
    set_api_key, mock_get_info, reset_run_option_config
):
    config.run.worker_group = "config_group"
    config.run.pay_type = PayType.CREDITS
    config.vgpu.priority = 5
    config.vgpu.vgpu_allocation = 4
    config.vgpu.ignore_memory_limit = True

    responses.add(
        responses.POST,
        f"{Env.current.web_api_endpoint}/tidy3d/tasks/{TASK_ID}/submit",
        match=[
            matchers.json_params_matcher(
                {
                    "solverVersion": None,
                    "workerGroup": "explicit_group",
                    "protocolVersion": td.version.__version__,
                    "enableCaching": Env.current.enable_caching,
                    "payType": PayType.AUTO,
                    "priority": 1,
                    "vgpuAllocation": 2,
                    "ignoreMemoryLimit": False,
                }
            )
        ],
        json={"data": {"taskId": TASK_ID, "taskName": TASK_NAME, "createdAt": CREATED_AT}},
        status=200,
    )

    start(
        TASK_ID,
        worker_group="explicit_group",
        pay_type=PayType.AUTO,
        priority=1,
        vgpu_allocation=2,
        ignore_memory_limit=False,
    )


@responses.activate
def test_start_uses_scoped_config_without_leaking(
    set_api_key, mock_get_info, reset_run_option_config
):
    assert config.run.worker_group is None
    assert config.run.additional_payload is None
    assert config.vgpu.priority is None
    assert config.run.pay_type == "AUTO"

    with config as scoped_config:
        scoped_config.run.worker_group = "scoped_group"
        scoped_config.run.additional_payload = {"routeHint": "scoped"}
        scoped_config.run.pay_type = PayType.CREDITS
        scoped_config.vgpu.priority = 7

        responses.add(
            responses.POST,
            f"{Env.current.web_api_endpoint}/tidy3d/tasks/{TASK_ID}/submit",
            match=[
                matchers.json_params_matcher(
                    {
                        "solverVersion": None,
                        "workerGroup": "scoped_group",
                        "protocolVersion": td.version.__version__,
                        "enableCaching": Env.current.enable_caching,
                        "payType": PayType.CREDITS,
                        "priority": 7,
                        "vgpuAllocation": None,
                        "ignoreMemoryLimit": None,
                        "additionalPayload": json.dumps({"routeHint": "scoped"}),
                    }
                )
            ],
            json={"data": {"taskId": TASK_ID, "taskName": TASK_NAME, "createdAt": CREATED_AT}},
            status=200,
        )

        start(TASK_ID)

        assert config.run.worker_group == "scoped_group"
        assert config.run.additional_payload == {"routeHint": "scoped"}
        assert config.run.pay_type == PayType.CREDITS.value
        assert config.vgpu.priority == 7

    assert config.run.worker_group is None
    assert config.run.additional_payload is None
    assert config.run.pay_type == "AUTO"
    assert config.vgpu.priority is None


@responses.activate
def test_start_batch_ignores_config_run_defaults(monkeypatch, set_api_key, reset_run_option_config):
    batch_task_id = "batch-task-id"

    config.run.solver_version = "config_solver"
    config.run.worker_group = "config_group"
    config.run.pay_type = PayType.CREDITS
    config.vgpu.priority = 5
    config.vgpu.vgpu_allocation = 4
    config.vgpu.ignore_memory_limit = True

    monkeypatch.setattr(
        "tidy3d.web.api.webapi.TaskFactory.get",
        lambda task_id: td.web.core.task_core.BatchTask(taskId=task_id),
    )

    responses.add(
        responses.POST,
        f"{Env.current.web_api_endpoint}/rf/task/{batch_task_id}/submit",
        match=[
            matchers.json_params_matcher(
                {
                    "solverVersion": "config_solver",
                    "protocolVersion": td.version.__version__,
                    "workerGroup": "config_group",
                }
            )
        ],
        json={"data": {"taskId": batch_task_id}},
        status=200,
    )

    start(batch_task_id)


@responses.activate
def test_start_batch_uses_config_run_additional_payload(
    monkeypatch, set_api_key, reset_run_option_config
):
    batch_task_id = "batch-task-id"

    config.run.additional_payload = {"routeHint": "batch"}

    monkeypatch.setattr(
        "tidy3d.web.api.webapi.TaskFactory.get",
        lambda task_id: td.web.core.task_core.BatchTask(taskId=task_id),
    )

    responses.add(
        responses.POST,
        f"{Env.current.web_api_endpoint}/rf/task/{batch_task_id}/submit",
        match=[
            matchers.json_params_matcher(
                {
                    "solverVersion": None,
                    "protocolVersion": td.version.__version__,
                    "workerGroup": None,
                    "additionalPayload": json.dumps({"routeHint": "batch"}),
                }
            )
        ],
        json={"data": {"taskId": batch_task_id}},
        status=200,
    )

    start(batch_task_id)


@responses.activate
def test_start_batch_explicit_priority_preserves_legacy_error(
    monkeypatch, set_api_key, reset_run_option_config
):
    batch_task_id = "batch-task-id"

    monkeypatch.setattr(
        "tidy3d.web.api.webapi.TaskFactory.get",
        lambda task_id: td.web.core.task_core.BatchTask(taskId=task_id),
    )

    with pytest.raises(
        NotImplementedError,
        match="The 'priority' argument is not yet supported and will be ignored.",
    ):
        start(batch_task_id, priority=5)


@responses.activate
def test_start_batch_invalid_priority_preserves_legacy_error(
    monkeypatch, set_api_key, reset_run_option_config
):
    batch_task_id = "batch-task-id"

    monkeypatch.setattr(
        "tidy3d.web.api.webapi.TaskFactory.get",
        lambda task_id: td.web.core.task_core.BatchTask(taskId=task_id),
    )

    with pytest.raises(
        NotImplementedError,
        match="The 'priority' argument is not yet supported and will be ignored.",
    ):
        start(batch_task_id, priority=0)


@responses.activate
def test_start_batch_invalid_vgpu_allocation_preserves_legacy_error(
    monkeypatch, set_api_key, reset_run_option_config
):
    batch_task_id = "batch-task-id"

    monkeypatch.setattr(
        "tidy3d.web.api.webapi.TaskFactory.get",
        lambda task_id: td.web.core.task_core.BatchTask(taskId=task_id),
    )

    with pytest.raises(
        NotImplementedError,
        match="The 'vgpu_allocation' argument is not yet supported and will be ignored.",
    ):
        start(batch_task_id, vgpu_allocation=3)


def test_webapi_run_logs_deprecated_run_args_on_cache_hit(monkeypatch, tmp_path):
    warning_calls = []

    monkeypatch.setattr(
        "tidy3d.web.api.webapi.log_deprecated_run_args",
        lambda **kwargs: warning_calls.append(kwargs),
    )
    monkeypatch.setattr(
        "tidy3d.web.api.webapi.restore_simulation_if_cached",
        lambda **kwargs: (tmp_path / "cached.hdf5", None),
    )
    monkeypatch.setattr(
        "tidy3d.web.api.webapi.load",
        lambda **kwargs: make_sim_data(),
    )

    import tidy3d.web.api.webapi as webapi_module

    webapi_module.run(
        simulation=make_sim(),
        worker_group="worker_a",
        simulation_type="special_type",
        pay_type=PayType.CREDITS,
        priority=4,
        vgpu_allocation=2,
        ignore_memory_limit=True,
    )

    assert warning_calls == [
        {
            "solver_version": None,
            "worker_group": "worker_a",
            "pay_type": PayType.CREDITS,
            "priority": 4,
            "vgpu_allocation": 2,
            "ignore_memory_limit": True,
        }
    ]


@pytest.mark.parametrize(
    ("warning_target", "stop_target", "call_kind"),
    [
        (
            "tidy3d.web.api.run.log_deprecated_run_args",
            "tidy3d.web.api.run.run_autograd",
            "run",
        ),
        (
            "tidy3d.web.api.autograd.autograd.log_deprecated_run_args",
            "tidy3d.web.api.autograd.autograd.asynchronous_webapi.run_async",
            "run_async",
        ),
    ],
)
def test_public_run_interfaces_log_deprecated_simulation_type(
    monkeypatch, warning_target, stop_target, call_kind
):
    warning_calls = []

    monkeypatch.setattr(
        warning_target,
        lambda **kwargs: warning_calls.append(kwargs),
    )
    monkeypatch.setattr(
        stop_target,
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("stop_after_warning")),
    )

    with pytest.raises(RuntimeError, match="stop_after_warning"):
        if call_kind == "run":
            td.web.run(make_sim(), simulation_type="special_type")
        else:
            td.web.run_async({"sim": make_sim()}, simulation_type="special_type")

    assert len(warning_calls) == 1
    assert warning_calls[0]["simulation_type"] == "special_type"


@pytest.mark.parametrize("call_kind", ["run", "run_async"])
def test_regular_autograd_internal_simulation_type_does_not_warn(monkeypatch, call_kind):
    warning_calls = []
    upload_option_calls = []

    monkeypatch.setattr(
        "tidy3d.web.api.run_options.log.warning",
        lambda message, **kwargs: warning_calls.append((message, kwargs)),
    )
    monkeypatch.setattr(
        "tidy3d.web.api.autograd.autograd.is_valid_for_autograd",
        lambda simulation: True,
    )
    monkeypatch.setattr(
        "tidy3d.web.api.autograd.autograd.is_valid_for_autograd_async",
        lambda simulations: True,
    )
    monkeypatch.setattr(
        "tidy3d.web.api.autograd.autograd.setup_run",
        lambda simulation, numerical_structures=None: SimpleNamespace(
            simulation=simulation,
            sim_fields={("sources", 0, "center", 0): 0.0},
            numerical_structure_map={},
        ),
    )
    monkeypatch.setattr(
        "tidy3d.web.api.autograd.autograd.setup_fwd",
        lambda sim_fields, sim_original, local_gradient: sim_original,
    )
    monkeypatch.setattr("tidy3d.web.api.container.Folder.get", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "tidy3d.web.api.webapi.resolve_upload_options",
        lambda **kwargs: (
            upload_option_calls.append(kwargs),
            (_ for _ in ()).throw(RuntimeError("stop_after_warning_check")),
        )[1],
    )

    with pytest.raises(RuntimeError, match="stop_after_warning_check"):
        if call_kind == "run":
            td.web.run(make_sim(), verbose=False)
        else:
            td.web.run_async({"sim": make_sim()}, verbose=False)

    assert len(upload_option_calls) == 1
    assert upload_option_calls[0]["simulation_type"] == "autograd_fwd"
    assert warning_calls == []


def test_log_deprecated_run_args_uses_stable_message(monkeypatch):
    warning_calls = []

    monkeypatch.setattr(
        "tidy3d.web.api.run_options.log.warning",
        lambda message, **kwargs: warning_calls.append((message, kwargs)),
    )

    log_deprecated_run_args(simulation_type="special_type")
    log_deprecated_run_args(pay_type=PayType.CREDITS, priority=4)

    expected_message = (
        "Passing run options as direct arguments is deprecated. "
        "Set defaults via 'td.config.run' and 'td.config.vgpu' instead."
    )
    assert warning_calls == [
        (expected_message, {"log_once": True}),
        (expected_message, {"log_once": True}),
    ]


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


@pytest.mark.parametrize(
    ("task_type", "expected_name"),
    [
        (TaskType.FDTD, "simulation_data.hdf5"),
        (TaskType.MODE_SOLVER, "simulation_data.hdf5"),
        (TaskType.HEAT.name, "simulation_data.hdf5"),
        (TaskType.MODAL_CM, "cm_data.hdf5"),
        (TaskType.TERMINAL_CM, "cm_data.hdf5"),
        ("RF", "cm_data.hdf5"),
        ("UNKNOWN", "simulation_data.hdf5"),
        (None, "simulation_data.hdf5"),
    ],
)
def test_default_data_filename(task_type, expected_name):
    assert default_data_filename(task_type) == expected_name


def test_download_uses_mode_defaults_and_single_task_lookup(monkeypatch, tmp_path):
    captured = {}
    get_calls = 0

    class FakeTask:
        task_type = TaskType.MODE_SOLVER.name

        def get_data_hdf5(self, to_file, remote_data_file_gz, **kwargs):
            captured["to_file"] = Path(to_file)
            captured["remote_data_file_gz"] = remote_data_file_gz
            Path(to_file).write_text("mode-data")
            return Path(to_file)

    def _fake_get(*args, **kwargs):
        nonlocal get_calls
        get_calls += 1
        return FakeTask()

    monkeypatch.setattr(f"{task_core_path}.TaskFactory.get", _fake_get)
    monkeypatch.chdir(tmp_path)

    download(TASK_ID, path=None, verbose=False)

    assert get_calls == 1
    assert captured["to_file"] == Path("simulation_data.hdf5")
    assert captured["remote_data_file_gz"] == MODE_DATA_HDF5_GZ
    assert (tmp_path / "simulation_data.hdf5").exists()


def test_download_uses_cm_default_path_for_batch(monkeypatch, tmp_path):
    captured = {}
    batch_task = BatchTask(taskId=TASK_ID, taskType=TaskType.TERMINAL_CM.name)

    def _fake_get(*args, **kwargs):
        return batch_task

    def _fake_get_data_hdf5(self, to_file, remote_data_file_gz, **kwargs):
        captured["to_file"] = Path(to_file)
        captured["remote_data_file_gz"] = remote_data_file_gz
        Path(to_file).write_text("cm-data")
        return Path(to_file)

    monkeypatch.setattr(f"{task_core_path}.TaskFactory.get", _fake_get)
    monkeypatch.setattr(BatchTask, "get_data_hdf5", _fake_get_data_hdf5)
    monkeypatch.chdir(tmp_path)

    download(TASK_ID, path=None, verbose=False)

    assert captured["to_file"] == Path("cm_data.hdf5")
    assert (tmp_path / "cm_data.hdf5").exists()


def test_download_respects_explicit_batch_path(monkeypatch, tmp_path):
    captured = {}
    explicit_path = tmp_path / "simulation_data.hdf5"
    batch_task = BatchTask(taskId=TASK_ID, taskType=TaskType.TERMINAL_CM.name)

    def _fake_get(*args, **kwargs):
        return batch_task

    def _fake_get_data_hdf5(self, to_file, remote_data_file_gz, **kwargs):
        captured["to_file"] = Path(to_file)
        Path(to_file).write_text("cm-data")
        return Path(to_file)

    monkeypatch.setattr(f"{task_core_path}.TaskFactory.get", _fake_get)
    monkeypatch.setattr(BatchTask, "get_data_hdf5", _fake_get_data_hdf5)

    download(TASK_ID, path=explicit_path, verbose=False)

    assert captured["to_file"] == explicit_path


def test_load_uses_cm_default_path_for_batch(monkeypatch, tmp_path):
    captured = {"get_calls": 0}
    batch_task = BatchTask(taskId=TASK_ID, taskType=TaskType.TERMINAL_CM.name)

    def _fake_get(*args, **kwargs):
        captured["get_calls"] += 1
        return batch_task

    def _fake_get_data_hdf5(self, to_file, remote_data_file_gz, **kwargs):
        captured["download_path"] = Path(to_file)
        captured["remote_data_file_gz"] = remote_data_file_gz
        Path(to_file).write_text("cm-data")
        return Path(to_file)

    def _fake_postprocess(path, lazy=False):
        captured["postprocess_path"] = Path(path)
        return "cm-stub-data"

    monkeypatch.setattr(f"{task_core_path}.TaskFactory.get", _fake_get)
    monkeypatch.setattr(BatchTask, "get_data_hdf5", _fake_get_data_hdf5)
    monkeypatch.setattr(f"{api_path}.Tidy3dStubData.postprocess", _fake_postprocess)
    monkeypatch.setattr(f"{api_path}.resolve_local_cache", lambda: None)
    monkeypatch.chdir(tmp_path)

    data = load(TASK_ID, path=None, verbose=False)

    assert data == "cm-stub-data"
    assert captured["get_calls"] == 1
    assert captured["download_path"] == Path("cm_data.hdf5")
    assert captured["postprocess_path"] == Path("cm_data.hdf5")
    assert (tmp_path / "cm_data.hdf5").exists()


def test_load_respects_explicit_batch_path(monkeypatch, tmp_path):
    captured = {}
    explicit_path = tmp_path / "simulation_data.hdf5"
    batch_task = BatchTask(taskId=TASK_ID, taskType=TaskType.TERMINAL_CM.name)

    def _fake_get(*args, **kwargs):
        return batch_task

    def _fake_get_data_hdf5(self, to_file, remote_data_file_gz, **kwargs):
        captured["download_path"] = Path(to_file)
        Path(to_file).write_text("cm-data")
        return Path(to_file)

    def _fake_postprocess(path, lazy=False):
        captured["postprocess_path"] = Path(path)
        return "cm-stub-data"

    monkeypatch.setattr(f"{task_core_path}.TaskFactory.get", _fake_get)
    monkeypatch.setattr(BatchTask, "get_data_hdf5", _fake_get_data_hdf5)
    monkeypatch.setattr(f"{api_path}.Tidy3dStubData.postprocess", _fake_postprocess)
    monkeypatch.setattr(f"{api_path}.resolve_local_cache", lambda: None)

    data = load(TASK_ID, path=explicit_path, verbose=False)

    assert data == "cm-stub-data"
    assert captured["download_path"] == explicit_path
    assert captured["postprocess_path"] == explicit_path


@responses.activate
def _test_load(mock_load, mock_get_info, tmp_path):
    def mock_download(*args, **kwargs):
        pass

    monkeypatch.setattr(f"{task_core_path}.download_file", mock_download)
    load(TASK_ID, str(tmp_path / "monitor_data.hdf5"))


def test_load_existing_explicit_path_skips_task_lookup(monkeypatch, tmp_path):
    data_path = tmp_path / "existing_results.hdf5"
    data_path.write_text("stub")

    def _raise(*args, **kwargs):
        raise AssertionError("Unexpected task lookup during explicit-path load.")

    monkeypatch.setattr(f"{api_path}.get_info", _raise)
    monkeypatch.setattr(f"{task_core_path}.TaskFactory.get", _raise)
    monkeypatch.setattr(f"{task_core_path}.TaskFactory.get_kind", _raise)
    monkeypatch.setattr(f"{api_path}.resolve_local_cache", lambda: None)
    monkeypatch.setattr(
        f"{api_path}.Tidy3dStubData.postprocess", lambda *args, **kwargs: "stub_data"
    )

    assert load(TASK_ID, path=data_path, replace_existing=False, verbose=False) == "stub_data"


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


def test_batch_data_items_dict_loads_all_tasks(monkeypatch):
    task_names = ("task_a", "task_b", "task_c")
    batch_data = BatchData(
        task_paths={task_name: f"{task_name}.hdf5" for task_name in task_names},
        task_ids={task_name: f"id_{task_name}" for task_name in task_names},
        cached_tasks=dict.fromkeys(task_names, False),
        is_downloaded=True,
    )

    loaded_task_names = []

    def _fake_load(self, task_name):
        loaded_task_names.append(task_name)
        return f"loaded_{task_name}"

    monkeypatch.setattr(BatchData, "load_sim_data", _fake_load)

    loaded_data = dict(batch_data.items())

    assert loaded_data == {task_name: f"loaded_{task_name}" for task_name in task_names}
    assert loaded_task_names == list(task_names)


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


def test_batch_from_file_lazy_sets_batch_lazy(tmp_path):
    batch = Batch(simulations={TASK_NAME: make_sim()}, folder_name=PROJECT_NAME)
    batch_path = tmp_path / "batch.hdf5"
    batch.to_file(batch_path)

    eager_batch = Batch.from_file(batch_path)
    assert eager_batch.lazy is False

    callback_lazy_values = []

    def _on_load(loaded_batch):
        callback_lazy_values.append(loaded_batch.lazy)

    lazy_batch = Batch.from_file(batch_path, lazy=True, on_load=_on_load)
    assert type(lazy_batch).__name__.endswith("Proxy")
    assert "_lazy_fname" in lazy_batch.__dict__

    assert lazy_batch.lazy is True
    assert type(lazy_batch) is Batch
    assert callback_lazy_values == [True]


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
    """Test that batch.run() saves batch file with task_ids before monitor()."""
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


def test_batch_run_does_not_save_file_when_upload_fails_early(tmp_path, monkeypatch):
    sims = {TASK_NAME: make_sim()}
    batch = Batch(simulations=sims, folder_name=PROJECT_NAME)

    monitor_called = {"value": False}

    def mock_upload_fail(self, *args, **kwargs):
        raise RuntimeError("Simulated failure during upload")

    monkeypatch.setattr(Batch, "upload", mock_upload_fail)
    monkeypatch.setattr(Batch, "monitor", lambda *args, **kwargs: monitor_called.update(value=True))

    with pytest.raises(RuntimeError, match="Simulated failure during upload"):
        batch.run(path_dir=str(tmp_path))

    assert not os.path.exists(batch._batch_path(path_dir=str(tmp_path)))
    assert not monitor_called["value"]


def test_batch_run_surfaces_to_file_error_when_upload_succeeds(tmp_path, monkeypatch):
    sims = {TASK_NAME: make_sim()}
    batch = Batch(simulations=sims, folder_name=PROJECT_NAME)
    monitor_called = {"value": False}

    def _fake_upload(self):
        return

    monkeypatch.setattr(Batch, "upload", _fake_upload)
    monkeypatch.setattr(
        Batch,
        "to_file",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("to_file failed")),
    )

    def _track_monitor(*args, **kwargs):
        monitor_called["value"] = True

    monkeypatch.setattr(Batch, "monitor", _track_monitor)

    with pytest.raises(RuntimeError, match="to_file failed"):
        batch.run(path_dir=str(tmp_path))

    assert not monitor_called["value"]


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


def test_batch_monitor_verbose_refreshes_terminal_cache_for_cached_end_state(monkeypatch, tmp_path):
    events = []

    monkeypatch.setattr("tidy3d.web.api.container.ThreadPoolExecutor", ImmediateExecutor)
    monkeypatch.setattr("tidy3d.web.api.container.time.sleep", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(Batch, "estimate_cost", lambda *args, **kwargs: None)

    sims = {"done_task": make_sim()}
    batch = Batch(simulations=sims, folder_name=PROJECT_NAME, verbose=True)
    batch._cached_properties = {}
    batch._cached_properties["jobs"] = {
        "done_task": FakeJob("done_id", ["success"], events),
    }
    batch._terminal_status_by_task = {"done_task": "success"}
    batch._terminal_task_id_by_task = {}

    batch.monitor(download_on_success=False, path_dir=str(tmp_path))

    assert batch._terminal_status_by_task["done_task"] == "success"
    assert batch._terminal_task_id_by_task["done_task"] == "done_id"


def test_batch_monitor_quiet_refreshes_terminal_cache_for_cached_end_state(monkeypatch, tmp_path):
    events = []

    monkeypatch.setattr("tidy3d.web.api.container.ThreadPoolExecutor", ImmediateExecutor)
    monkeypatch.setattr("tidy3d.web.api.container.time.sleep", lambda *_args, **_kwargs: None)

    sims = {"done_task": make_sim()}
    batch = Batch(simulations=sims, folder_name=PROJECT_NAME, verbose=False)
    batch._cached_properties = {}
    batch._cached_properties["jobs"] = {
        "done_task": FakeJob("done_id", ["success"], events),
    }
    batch._terminal_status_by_task = {"done_task": "success"}
    batch._terminal_task_id_by_task = {}

    batch.monitor(download_on_success=False, path_dir=str(tmp_path))

    assert batch._terminal_status_by_task["done_task"] == "success"
    assert batch._terminal_task_id_by_task["done_task"] == "done_id"


def test_batch_monitor_does_not_cache_none_task_ids(monkeypatch, tmp_path):
    monkeypatch.setattr("tidy3d.web.api.container.ThreadPoolExecutor", ImmediateExecutor)
    monkeypatch.setattr("tidy3d.web.api.container.time.sleep", lambda *_args, **_kwargs: None)

    class CachedNoTaskIdJob:
        task_id = None
        task_id_cached = None

        @property
        def load_if_cached(self):
            return True

        @property
        def status(self):
            raise AssertionError("status should not be polled for cached jobs")

        def get_info(self):
            raise AssertionError("get_info() should not be called for cached jobs")

    sims = {"cached_task": make_sim()}
    batch = Batch(simulations=sims, folder_name=PROJECT_NAME, verbose=False)
    batch._cached_properties = {}
    batch._cached_properties["jobs"] = {
        "cached_task": CachedNoTaskIdJob(),
    }

    batch.monitor(download_on_success=False, path_dir=str(tmp_path))

    assert batch._terminal_status_by_task["cached_task"] == "success"
    assert "cached_task" not in batch._terminal_task_id_by_task


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


def test_batch_monitor_cached_missing_task_ids_use_unique_download_paths(monkeypatch, tmp_path):
    monkeypatch.setattr("tidy3d.web.api.container.ThreadPoolExecutor", ImmediateExecutor)
    monkeypatch.setattr("tidy3d.web.api.container.time.sleep", lambda *_args, **_kwargs: None)

    materialized_paths = []

    class CachedMissingTaskIdJob:
        _cached_task_id = None

        def __init__(self, task_name: str):
            self.task_name = task_name
            self.simulation = make_sim()

        @property
        def load_if_cached(self):
            return True

        @property
        def task_id_cached(self):
            return None

        @property
        def task_id(self):
            return None

        @property
        def status(self):
            raise AssertionError("status should not be polled for cached jobs")

        def get_info(self):
            raise AssertionError("get_info() should not be called for cached jobs")

        def _materialize_from_stash(self, path: PathLike):
            materialized_paths.append(str(path))

        def download(self, path: PathLike):
            self._materialize_from_stash(path)

    sims = {"cached_task_a": make_sim(), "cached_task_b": make_sim()}
    batch = Batch(simulations=sims, folder_name=PROJECT_NAME, verbose=False)
    batch._cached_properties = {
        "jobs": {
            "cached_task_a": CachedMissingTaskIdJob("cached_task_a"),
            "cached_task_b": CachedMissingTaskIdJob("cached_task_b"),
        }
    }

    batch.monitor(download_on_success=True, path_dir=str(tmp_path))

    expected_a = Batch._cached_fallback_task_id("cached_task_a", batch.jobs["cached_task_a"])
    expected_b = Batch._cached_fallback_task_id("cached_task_b", batch.jobs["cached_task_b"])
    assert len(materialized_paths) == 2
    assert any(path.endswith(f"{expected_a}.hdf5") for path in materialized_paths)
    assert any(path.endswith(f"{expected_b}.hdf5") for path in materialized_paths)


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


def test_batch_download_uses_terminal_status_cache(monkeypatch, tmp_path):
    downloaded_paths = []
    monkeypatch.setattr(Batch, "to_file", lambda *_args, **_kwargs: None)

    class CachedTerminalStatusJob:
        def __init__(self):
            self._cached_properties = {}
            self.simulation = make_sim()

        @property
        def load_if_cached(self):
            return False

        @property
        def task_id(self):
            return "task_cached_status"

        @property
        def status(self):
            raise AssertionError("status should not be polled when terminal status is cached")

        def download(self, path: PathLike):
            downloaded_paths.append(str(path))

    sims = {"task_a": make_sim()}
    batch = Batch(simulations=sims, folder_name=PROJECT_NAME, verbose=False)
    batch._cached_properties = {"jobs": {"task_a": CachedTerminalStatusJob()}}
    batch._terminal_status_by_task = {"task_a": "success"}

    batch.download(path_dir=str(tmp_path))

    assert len(downloaded_paths) == 1
    assert downloaded_paths[0].endswith("task_cached_status.hdf5")


def test_batch_download_cached_missing_task_ids_use_unique_paths(monkeypatch, tmp_path):
    materialized_paths = []
    monkeypatch.setattr(Batch, "to_file", lambda *_args, **_kwargs: None)

    class CachedMissingTaskIdJob:
        _cached_task_id = None
        task_id_cached = None

        def __init__(self, task_name: str):
            self.task_name = task_name
            self._cached_properties = {}
            self.simulation = make_sim()

        @property
        def load_if_cached(self):
            return True

        @property
        def task_id(self):
            return None

        @property
        def status(self):
            return "success"

        def _materialize_from_stash(self, path: PathLike):
            materialized_paths.append(str(path))

    sims = {"cached_task_a": make_sim(), "cached_task_b": make_sim()}
    batch = Batch(simulations=sims, folder_name=PROJECT_NAME, verbose=False)
    batch._cached_properties = {
        "jobs": {
            "cached_task_a": CachedMissingTaskIdJob("cached_task_a"),
            "cached_task_b": CachedMissingTaskIdJob("cached_task_b"),
        }
    }

    batch.download(path_dir=str(tmp_path))

    expected_a = Batch._cached_fallback_task_id("cached_task_a", batch.jobs["cached_task_a"])
    expected_b = Batch._cached_fallback_task_id("cached_task_b", batch.jobs["cached_task_b"])
    assert len(materialized_paths) == 2
    assert any(path.endswith(f"{expected_a}.hdf5") for path in materialized_paths)
    assert any(path.endswith(f"{expected_b}.hdf5") for path in materialized_paths)


def test_batch_upload_surfaces_upload_errors(monkeypatch):
    monkeypatch.setattr("tidy3d.web.api.container.ThreadPoolExecutor", ImmediateExecutor)
    monkeypatch.setattr(Batch, "_check_folder", staticmethod(lambda *args, **kwargs: None))
    monkeypatch.setattr("tidy3d.web.api.container.Job.load_if_cached", property(lambda self: False))
    error_messages = []
    monkeypatch.setattr(
        "tidy3d.web.api.container.log.error", lambda msg: error_messages.append(msg)
    )

    def _raise_upload(self):
        raise RuntimeError("upload failed")

    monkeypatch.setattr("tidy3d.web.api.container.Job.upload", _raise_upload)

    sims = {"task_a": make_sim()}
    batch = Batch(simulations=sims, folder_name=PROJECT_NAME, verbose=False)

    with pytest.raises(RuntimeError, match="upload failed"):
        batch.upload()
    assert any("task_a" in msg for msg in error_messages)


def test_batch_start_surfaces_start_errors(monkeypatch):
    monkeypatch.setattr("tidy3d.web.api.container.ThreadPoolExecutor", ImmediateExecutor)
    started = []
    error_messages = []
    monkeypatch.setattr(
        "tidy3d.web.api.container.log.error", lambda msg: error_messages.append(msg)
    )

    class StartErrorFakeJob:
        def __init__(self, task_name: str, should_fail: bool):
            self.task_name = task_name
            self._should_fail = should_fail

        def start(self, priority=None, vgpu_allocation=None, ignore_memory_limit=None):
            started.append((self.task_name, priority))
            if self._should_fail:
                raise RuntimeError("start failed")

    jobs_to_start = [StartErrorFakeJob("ok_task", False), StartErrorFakeJob("bad_task", True)]

    sims = {"ok_task": make_sim(), "bad_task": make_sim()}
    batch = Batch(simulations=sims, folder_name=PROJECT_NAME, verbose=False)
    monkeypatch.setattr(Batch, "_prepare_uncached_jobs", lambda self: jobs_to_start)

    with pytest.raises(RuntimeError, match="start failed"):
        batch.start(priority=6)

    assert started == [("ok_task", 6), ("bad_task", 6)]
    assert any("bad_task" in msg for msg in error_messages)


def test_batch_load_parallel_status_collection(monkeypatch, tmp_path):
    warning_messages = []
    monkeypatch.setattr(
        "tidy3d.web.api.container.log.warning", lambda msg: warning_messages.append(msg)
    )

    sims = {"ok_task": make_sim(), "bad_task": make_sim()}
    batch = Batch(simulations=sims, folder_name=PROJECT_NAME, verbose=False)
    batch._cached_properties = {
        "jobs": {
            "ok_task": LoadStatusFakeJob("ok_task_id", "success", sims["ok_task"]),
            "bad_task": LoadStatusFakeJob("bad_task_id", "blocked", sims["bad_task"]),
        }
    }

    data = batch.load(path_dir=str(tmp_path), skip_download=True)

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
    monkeypatch.setattr("tidy3d.web.api.container.Job.load_if_cached", property(lambda self: False))
    warning_messages = []

    def _raise_upload(self, *args, **kwargs):
        raise AssertionError("Batch.load() should not upload tasks.")

    monkeypatch.setattr("tidy3d.web.api.container.Job._upload", _raise_upload)
    monkeypatch.setattr(
        "tidy3d.web.api.container.log.warning", lambda msg: warning_messages.append(msg)
    )

    sims = {"task_a": make_sim()}
    batch = Batch(simulations=sims, folder_name=PROJECT_NAME, verbose=False)

    data = batch.load(path_dir=str(tmp_path), skip_download=True)

    assert data.task_ids == {}
    assert data.task_paths == {}
    assert warning_messages == ["Not loading 'task_a' as the task hasn't been uploaded."]


def test_batch_load_keeps_cached_task_with_missing_task_id(monkeypatch, tmp_path):
    warning_messages = []
    monkeypatch.setattr(
        "tidy3d.web.api.container.log.warning", lambda msg: warning_messages.append(msg)
    )

    class CachedMissingTaskIdJob:
        simulation = make_sim()
        _cached_task_id = None

        @property
        def load_if_cached(self):
            return True

        @property
        def task_id_cached(self):
            return None

    sims = {"cached_task_a": make_sim(), "cached_task_b": make_sim()}
    batch = Batch(simulations=sims, folder_name=PROJECT_NAME, verbose=False)
    batch._cached_properties = {
        "jobs": {
            "cached_task_a": CachedMissingTaskIdJob(),
            "cached_task_b": CachedMissingTaskIdJob(),
        }
    }

    data = batch.load(path_dir=str(tmp_path), skip_download=True)

    expected_a = Batch._cached_fallback_task_id("cached_task_a", batch.jobs["cached_task_a"])
    expected_b = Batch._cached_fallback_task_id("cached_task_b", batch.jobs["cached_task_b"])
    assert data.cached_tasks["cached_task_a"] is True
    assert data.cached_tasks["cached_task_b"] is True
    assert data.task_ids["cached_task_a"] == expected_a
    assert data.task_ids["cached_task_b"] == expected_b
    assert data.task_paths["cached_task_a"].endswith(f"{expected_a}.hdf5")
    assert data.task_paths["cached_task_b"].endswith(f"{expected_b}.hdf5")
    assert data.task_paths["cached_task_a"] != data.task_paths["cached_task_b"]
    assert warning_messages == []


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


def test_batch_run_uses_legacy_upload_then_start(monkeypatch, tmp_path):
    class RunFakeJob:
        @property
        def load_if_cached(self):
            return False

    upload_calls = {"count": 0}
    start_calls = {"count": 0, "priority": None}
    run_calls = []
    load_kwargs = {}

    def _track_upload(self):
        upload_calls["count"] += 1

    def _track_start(self, priority=None, vgpu_allocation=None, ignore_memory_limit=None):
        start_calls["count"] += 1
        start_calls["priority"] = priority

    def _fake_load(self, **kwargs):
        load_kwargs.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(Batch, "upload", _track_upload)
    monkeypatch.setattr(Batch, "start", _track_start)
    monkeypatch.setattr(Batch, "to_file", lambda *args, **kwargs: run_calls.append("to_file"))
    monkeypatch.setattr(Batch, "monitor", lambda *args, **kwargs: run_calls.append("monitor"))
    monkeypatch.setattr(Batch, "load", _fake_load)

    batch = Batch(simulations={"task_a": make_sim()}, folder_name=PROJECT_NAME, verbose=False)
    batch._cached_properties = {"jobs": {"task_a": RunFakeJob()}}

    result = batch.run(path_dir=str(tmp_path), priority=8)

    assert result == {"ok": True}
    assert upload_calls["count"] == 1
    assert start_calls["count"] == 1
    assert start_calls["priority"] == 8
    assert run_calls == ["to_file", "monitor"]
    assert load_kwargs == {"path_dir": str(tmp_path), "skip_download": True}


def test_batch_upload_and_start_use_fixed_worker_bound(monkeypatch):
    worker_counts = []

    class DummyJob:
        task_name = TASK_NAME

        @property
        def load_if_cached(self):
            return False

        def upload(self):
            return None

        def start(self, priority=None, vgpu_allocation=None, ignore_memory_limit=None):
            return None

    def _fake_prepare_uncached_jobs(self, **kwargs):
        return [DummyJob()]

    class CapturingExecutor(ImmediateExecutor):
        def __init__(self, *args, **kwargs):
            worker_counts.append(kwargs.get("max_workers"))
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(Batch, "_prepare_uncached_jobs", _fake_prepare_uncached_jobs)
    monkeypatch.setattr("tidy3d.web.api.container.ThreadPoolExecutor", CapturingExecutor)

    batch = Batch(simulations={TASK_NAME: make_sim()}, folder_name=PROJECT_NAME, verbose=False)
    batch.upload()
    batch.start(priority=5)

    assert worker_counts == [UPLOAD_START_NUM_WORKERS, UPLOAD_START_NUM_WORKERS]


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


@responses.activate
def test_run_single_offline_eager_uses_default_path(monkeypatch, tmp_path):
    sim = make_sim()
    task_name = "single_default"
    monkeypatch.chdir(tmp_path)
    apply_common_patches(monkeypatch, tmp_path, taskid_to_sim={task_name: sim})

    sim_data = run(sim, task_name=task_name)

    assert (tmp_path / "simulation_data.hdf5").exists()
    assert isinstance(sim_data, SimulationData)


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


def test_job_run_uses_default_path(monkeypatch, tmp_path):
    sim = make_sim()
    task_name = "job_default"
    monkeypatch.chdir(tmp_path)
    apply_common_patches(monkeypatch, tmp_path, taskid_to_sim={task_name: sim})

    job = Job(simulation=sim, task_name=task_name, folder_name=PROJECT_NAME)
    _ = job.run()

    assert (tmp_path / "simulation_data.hdf5").exists()


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


def test_job_upload_is_idempotent(monkeypatch):
    upload_calls = []

    def fake_upload(**kwargs):
        upload_calls.append(kwargs)
        return TASK_ID

    monkeypatch.setattr(f"{api_path}.upload", fake_upload)
    monkeypatch.setattr(WebContainer, "_check_folder", lambda *a, **k: True)

    job = Job(simulation=make_sim(), task_name=TASK_NAME, folder_name=PROJECT_NAME, verbose=False)

    job.upload()
    first_task_id = job.task_id
    job.upload()
    job.upload()
    second_task_id = job.task_id

    assert first_task_id == TASK_ID
    assert second_task_id == TASK_ID
    assert len(upload_calls) == 1
    assert "wait_for_estimate_cost" not in upload_calls[0]


def test_job_upload_checks_folder_once(monkeypatch):
    upload_calls = []
    check_folder_calls = []

    def fake_upload(**kwargs):
        upload_calls.append(kwargs)
        return TASK_ID

    def fake_check_folder(*args, **kwargs):
        check_folder_calls.append((args, kwargs))
        return True

    monkeypatch.setattr(f"{api_path}.upload", fake_upload)
    monkeypatch.setattr(WebContainer, "_check_folder", fake_check_folder)

    job = Job(simulation=make_sim(), task_name=TASK_NAME, folder_name=PROJECT_NAME, verbose=False)

    job.upload()
    job.upload()

    assert len(check_folder_calls) == 1
    assert len(upload_calls) == 1
    assert "wait_for_estimate_cost" not in upload_calls[0]


def test_job_estimate_cost_logging(monkeypatch, tmp_path, capsys):
    def assert_estimate_cost_prints(count: int) -> None:
        out, err = capsys.readouterr()
        assert out.count("estimate cost") == count, (
            f"expected {count}, got {out.count('estimate cost')}\nout: {out}"
        )

    sim = make_sim()
    apply_common_patches(monkeypatch, tmp_path, taskid_to_sim={"task": sim})

    # fresh upload in verbose mode should print estimate cost once
    job_fresh = Job(simulation=sim, task_name=f"{TASK_NAME}_fresh")
    job_fresh.upload()
    assert_estimate_cost_prints(1)

    job = Job(simulation=sim, task_name=TASK_NAME)

    # accessing task_id should NOT print
    _ = job.task_id
    assert_estimate_cost_prints(0)

    # upload should not print again once task_id is already cached
    job.upload()
    assert_estimate_cost_prints(0)

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
