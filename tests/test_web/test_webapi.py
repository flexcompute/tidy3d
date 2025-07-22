# Tests webapi and things that depend on it
from __future__ import annotations

import json

import numpy as np
import pytest
import responses
from _pytest import monkeypatch
from responses import matchers

import tidy3d as td
from tidy3d import Simulation
from tidy3d.__main__ import main
from tidy3d.components.data.data_array import ScalarFieldDataArray
from tidy3d.components.data.monitor_data import FieldData
from tidy3d.components.data.sim_data import SimulationData
from tidy3d.components.grid.grid_spec import GridSpec
from tidy3d.components.monitor import FieldMonitor
from tidy3d.components.source.current import PointDipole
from tidy3d.components.source.time import GaussianPulse
from tidy3d.exceptions import SetupError
from tidy3d.web.api.asynchronous import run_async
from tidy3d.web.api.container import Batch, Job
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
    run,
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

task_core_path = "tidy3d.web.core.task_core"
api_path = "tidy3d.web.api.webapi"

Env.dev.active()


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
                    "taskName": TASK_NAME,
                    "callbackUrl": None,
                    "simulationType": "tidy3d",
                    "parentTasks": None,
                    "fileType": "Gz",
                }
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
def mock_batch_upload_single(monkeypatch, set_api_key):
    """Mocks batch upload endpoint for single task."""
    # Mock folder retrieval
    responses.add(
        responses.GET,
        f"{Env.current.web_api_endpoint}/tidy3d/project",
        match=[matchers.query_param_matcher({"projectName": PROJECT_NAME})],
        json={"data": {"projectId": FOLDER_ID, "projectName": PROJECT_NAME}},
        status=200,
    )

    # mock batch endpoint - returns single task
    def batch_request_matcher(request):
        json_data = json.loads(request.body)
        assert "tasks" in json_data
        assert "batchType" in json_data
        assert "groupName" in json_data
        for task in json_data["tasks"]:
            assert "groupName" in task
            assert task["groupName"] == json_data["groupName"]
        return True, None

    responses.add(
        responses.POST,
        f"{Env.current.web_api_endpoint}/tidy3d/projects/{FOLDER_ID}/batch-tasks",
        match=[batch_request_matcher],
        json={"batchId": "batch_123", "tasks": [{"taskId": "task_id_0", "taskName": "task_0"}]},
        status=200,
    )

    # mock task detail endpoints for the single task
    responses.add(
        responses.GET,
        f"{Env.current.web_api_endpoint}/tidy3d/tasks/task_id_0",
        json={
            "data": {
                "taskId": "task_id_0",
                "taskName": "task_0",
                "createdAt": CREATED_AT,
                "fileType": "Gz",
                "resourcePath": "output/task_id_0.json",
                "solverVersion": None,
                "taskType": TaskType.FDTD.name,
            }
        },
        status=200,
    )

    responses.add(
        responses.GET,
        f"{Env.current.web_api_endpoint}/tidy3d/tasks/task_id_0/detail",
        json={
            "data": {
                "taskId": "task_id_0",
                "taskName": "task_0",
                "createdAt": CREATED_AT,
                "realFlexUnit": FLEX_UNIT,
                "estFlexUnit": EST_FLEX_UNIT,
                "taskType": TaskType.FDTD.name,
                "metadataStatus": "processed",
                "status": "draft",
                "s3Storage": 1.0,
            }
        },
        status=200,
    )

    def mock_upload_file(*args, **kwargs):
        pass

    monkeypatch.setattr("tidy3d.web.core.task_core.upload_file", mock_upload_file)


@pytest.fixture
def mock_batch_upload_triple(monkeypatch, set_api_key):
    """Mocks batch upload endpoint for three tasks."""
    # Mock folder retrieval
    responses.add(
        responses.GET,
        f"{Env.current.web_api_endpoint}/tidy3d/project",
        match=[matchers.query_param_matcher({"projectName": PROJECT_NAME})],
        json={"data": {"projectId": FOLDER_ID, "projectName": PROJECT_NAME}},
        status=200,
    )

    def batch_request_matcher(request):
        import json

        json_data = json.loads(request.body)
        assert "tasks" in json_data
        assert "batchType" in json_data
        assert "groupName" in json_data
        for task in json_data["tasks"]:
            assert "groupName" in task
            assert task["groupName"] == json_data["groupName"]
        return True, None

    responses.add(
        responses.POST,
        f"{Env.current.web_api_endpoint}/tidy3d/projects/{FOLDER_ID}/batch-tasks",
        match=[batch_request_matcher],
        json={
            "batchId": "batch_123",
            "tasks": [
                {"taskId": "task_id_0", "taskName": "task_0"},
                {"taskId": "task_id_1", "taskName": "task_1"},
                {"taskId": "task_id_2", "taskName": "task_2"},
            ],
        },
        status=200,
    )

    for i in range(3):
        task_name = f"task_{i}"
        task_id = f"task_id_{i}"

        responses.add(
            responses.GET,
            f"{Env.current.web_api_endpoint}/tidy3d/tasks/{task_id}",
            json={
                "data": {
                    "taskId": task_id,
                    "taskName": task_name,
                    "createdAt": CREATED_AT,
                    "fileType": "Gz",
                    "resourcePath": f"output/{task_id}.json",
                    "solverVersion": None,
                    "taskType": TaskType.FDTD.name,
                }
            },
            status=200,
        )

        responses.add(
            responses.GET,
            f"{Env.current.web_api_endpoint}/tidy3d/tasks/{task_id}/detail",
            json={
                "data": {
                    "taskId": task_id,
                    "taskName": task_name,
                    "createdAt": CREATED_AT,
                    "realFlexUnit": FLEX_UNIT,
                    "estFlexUnit": EST_FLEX_UNIT,
                    "taskType": TaskType.FDTD.name,
                    "metadataStatus": "processed",
                    "status": "draft",
                    "s3Storage": 1.0,
                }
            },
            status=200,
        )

    def mock_upload_file(*args, **kwargs):
        pass

    monkeypatch.setattr("tidy3d.web.core.task_core.upload_file", mock_upload_file)


@pytest.fixture
def mock_webapi(
    mock_upload, mock_metadata, mock_get_info, mock_start, mock_monitor, mock_download, mock_load
):
    """Mocks all webapi operation."""


@responses.activate
def test_source_validation(monkeypatch, mock_upload, mock_get_info, mock_metadata):
    sim = make_sim().copy(update={"sources": []})

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
def test_get_run_info(mock_get_run_info):
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
        return sim.json().encode("utf-8")

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
def test_run(mock_webapi, monkeypatch, tmp_path):
    sim = make_sim()
    monkeypatch.setattr(f"{api_path}.load", lambda *args, **kwargs: True)
    assert run(
        sim,
        task_name=TASK_NAME,
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
def test_job(mock_webapi, monkeypatch, tmp_path):
    monkeypatch.setattr("tidy3d.web.api.container.Job.load", lambda *args, **kwargs: True)
    sim = make_sim()
    j = Job(simulation=sim, task_name=TASK_NAME, folder_name=PROJECT_NAME)

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
def test_batch(mock_webapi, mock_job_status, mock_load, tmp_path):
    # monkeypatch.setattr("tidy3d.web.api.container.Batch.monitor", lambda self: time.sleep(0.1))
    # monkeypatch.setattr("tidy3d.web.api.container.Job.status", property(lambda self: "success"))

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


@responses.activate
def test_batch_with_endpoint(mock_batch_upload_triple, tmp_path):
    """Test batch with new batch endpoint."""

    sims = {f"task_{i}": make_sim() for i in range(3)}

    batch = Batch(simulations=sims, folder_name=PROJECT_NAME, use_batch_endpoint=True)

    assert batch.use_batch_endpoint is True

    # access jobs property to trigger batch submission
    jobs = batch.jobs

    # verify jobs were created with pre-assigned task_ids
    assert len(jobs) == 3
    for i, (task_name, job) in enumerate(jobs.items()):
        assert task_name == f"task_{i}"
        assert job.task_id == f"task_id_{i}"
        assert job._cached_properties.get("task_id") == f"task_id_{i}"

    # test serialization preserves the flag
    fname = str(tmp_path / "batch_endpoint.json")
    batch.to_file(fname)
    batch_loaded = Batch.from_file(fname)

    assert batch_loaded.use_batch_endpoint is True
    assert len(batch_loaded.jobs) == 3


@responses.activate
def test_batch_backward_compatibility(mock_webapi, mock_job_status, mock_load, tmp_path):
    """Test that default behavior remains unchanged (backward compatibility)."""
    sims = {TASK_NAME: make_sim()}

    # Create batch without specifying use_batch_endpoint (should default to False)
    batch = Batch(simulations=sims, folder_name=PROJECT_NAME)

    # Verify default is False
    assert batch.use_batch_endpoint is False

    # Access jobs to trigger normal flow
    jobs = batch.jobs
    assert len(jobs) == 1

    # Run and verify it works as before
    batch.run(path_dir=str(tmp_path))
    assert batch.real_cost() == FLEX_UNIT * len(sims)


@responses.activate
def test_batch_endpoint_integration(
    mock_batch_upload_single, mock_webapi, mock_job_status, tmp_path
):
    """Test both batch endpoint modes produce compatible results."""
    sim = make_sim()

    # old way
    batch_old = Batch(
        simulations={"task_0": sim}, folder_name=PROJECT_NAME, use_batch_endpoint=False
    )
    fname_old = str(tmp_path / "batch_old.json")
    batch_old.to_file(fname_old)

    # new way
    batch_new = Batch(
        simulations={"task_0": sim}, folder_name=PROJECT_NAME, use_batch_endpoint=True
    )
    fname_new = str(tmp_path / "batch_new.json")
    batch_new.to_file(fname_new)

    # load and verify both work
    batch_old_loaded = Batch.from_file(fname_old)
    batch_new_loaded = Batch.from_file(fname_new)

    assert batch_old_loaded.use_batch_endpoint is False
    assert batch_new_loaded.use_batch_endpoint is True
    assert len(batch_old_loaded.jobs) == 1
    assert len(batch_new_loaded.jobs) == 1


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


""" Async """


@responses.activate
def test_async(mock_webapi, mock_job_status):
    # monkeypatch.setattr("tidy3d.web.api.container.Job.status", property(lambda self: "success"))
    sims = {TASK_NAME: make_sim()}
    _ = run_async(sims, folder_name=PROJECT_NAME)


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
            ]
        )


@responses.activate
def test_load_invalid_task_raises(mock_webapi):
    """Ensure that load() raises TaskNotFoundError for a non-existent task ID."""
    fake_id = "INVALID_TASK_ID"

    responses.add(
        responses.GET,
        f"{Env.current.web_api_endpoint}/tidy3d/tasks/{fake_id}/detail",
        json={"error": "Task not found"},
        status=404,
    )
    with pytest.raises(WebNotFoundError, match="Resource not found"):
        load(fake_id)
