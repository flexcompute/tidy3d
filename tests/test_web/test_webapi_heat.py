# Tests webapi and things that depend on it

import pytest
import responses
from _pytest import monkeypatch

import tidy3d as td
from responses import matchers
from tidy3d import HeatSimulation
from tidy3d.web.core.environment import Env
from tidy3d.web.api.webapi import download, download_json, run, abort
from tidy3d.web.api.webapi import estimate_cost, get_info, get_run_info
from tidy3d.web.api.webapi import load_simulation, upload
from tidy3d.web.api.container import Job, Batch
from tidy3d.web.api.asynchronous import run_async

from tidy3d.web.core.types import TaskType
from ..test_components.test_heat import make_heat_sim

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
                    "taskType": TaskType.HEAT.name,
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

    def mock_download(*args, **kwargs):
        pass

    monkeypatch.setattr("tidy3d.web.core.task_core.upload_file", mock_download)


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
                "taskType": TaskType.HEAT.name,
                "createdAt": CREATED_AT,
                "realFlexUnit": FLEX_UNIT,
                "estFlexUnit": EST_FLEX_UNIT,
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

    responses.add(
        responses.POST,
        f"{Env.current.web_api_endpoint}/tidy3d/tasks/{TASK_ID}/submit",
        match=[
            matchers.json_params_matcher(
                {
                    "solverVersion": None,
                    "workerGroup": None,
                    "protocolVersion": td.version.__version__,
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


@pytest.fixture
def mock_monitor(monkeypatch):
    status_count = [0]
    statuses = ("upload", "running", "running", "running", "running", "running", "success")

    def mock_get_status(task_id):
        current_count = min(status_count[0], len(statuses) - 1)
        current_status = statuses[current_count]
        status_count[0] += 1
        return current_status
        # return TaskInfo(
        #     status=current_status, taskName=TASK_NAME, taskId=task_id, realFlexUnit=1.0
        #     )

    run_count = [0]
    perc_dones = (1, 10, 20, 30, 100)

    def mock_get_run_info(task_id):
        current_count = min(run_count[0], len(perc_dones) - 1)
        perc_done = perc_dones[current_count]
        run_count[0] += 1
        return perc_done, 1

    monkeypatch.setattr("tidy3d.web.api.connect_util.REFRESH_TIME", 0.00001)
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

    monkeypatch.setattr(f"{task_core_path}.download_file", _mock_download)
    download(TASK_ID, str(tmp_path / "web_test_tmp.json"))
    with open(str(tmp_path / "web_test_tmp.json"), "r") as f:
        assert f.read() == "0.3,5.7"


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
def test_upload(mock_upload):
    sim = make_heat_sim()
    assert upload(sim, TASK_NAME, PROJECT_NAME)


@responses.activate
def test_get_info(mock_get_info):
    assert get_info(TASK_ID).taskId == TASK_ID
    assert get_info(TASK_ID).taskType == "HEAT"


@responses.activate
def test_get_run_info(mock_get_run_info):
    assert get_run_info(TASK_ID) == (100, 0)


@responses.activate
def test_estimate_cost(set_api_key, mock_get_info, mock_metadata):
    assert estimate_cost(TASK_ID) == EST_FLEX_UNIT


@responses.activate
def test_download_json(monkeypatch, mock_get_info, tmp_path):
    sim = make_heat_sim()

    def mock_download(*args, **kwargs):
        pass

    def get_str(*args, **kwargs):
        return sim.json().encode("utf-8")

    monkeypatch.setattr(f"{task_core_path}.download_file", mock_download)
    monkeypatch.setattr(f"{task_core_path}._read_simulation_from_hdf5", get_str)

    fname_tmp = str(tmp_path / "web_test_tmp.json")
    download_json(TASK_ID, fname_tmp)
    assert HeatSimulation.from_file(fname_tmp) == sim


@responses.activate
def test_load_simulation(monkeypatch, mock_get_info, tmp_path):
    def mock_download(*args, **kwargs):
        make_heat_sim().to_file(args[1])

    monkeypatch.setattr(f"{task_core_path}.SimulationTask.get_simulation_json", mock_download)

    assert load_simulation(TASK_ID, str(tmp_path / "web_test_tmp.json"))


@responses.activate
def test_run(mock_webapi, monkeypatch, tmp_path):
    sim = make_heat_sim()
    monkeypatch.setattr(f"{api_path}.load", lambda *args, **kwargs: True)
    assert run(
        sim,
        task_name=TASK_NAME,
        folder_name=PROJECT_NAME,
        path=str(tmp_path / "web_test_tmp.json"),
    )


@responses.activate
def test_abort_task(set_api_key, mock_get_info):
    responses.add(
        responses.PUT,
        f"{Env.current.web_api_endpoint}/tidy3d/tasks/abort",
        match=[
            matchers.json_params_matcher(
                {
                    "taskId": TASK_ID,
                    "taskType": TaskType.HEAT.name,
                }
            )
        ],
        json={"result": True},
        status=200,
    )
    abort(TASK_ID)


""" Containers """


@responses.activate
def test_job(mock_webapi, monkeypatch, tmp_path):
    monkeypatch.setattr("tidy3d.web.api.container.Job.load", lambda *args, **kwargs: True)
    sim = make_heat_sim()
    j = Job(simulation=sim, task_name=TASK_NAME, folder_name=PROJECT_NAME)

    _ = j.run(path=str(tmp_path / "web_test_tmp.json"))
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
def test_batch(mock_webapi, mock_job_status, tmp_path):
    # monkeypatch.setattr("tidy3d.web.api.container.Batch.monitor", lambda self: time.sleep(0.1))
    # monkeypatch.setattr("tidy3d.web.api.container.Job.status", property(lambda self: "success"))

    sims = {TASK_NAME: make_heat_sim()}
    b = Batch(simulations=sims, folder_name=PROJECT_NAME)
    b.estimate_cost()
    _ = b.run(path_dir=str(tmp_path))
    assert b.real_cost() == FLEX_UNIT * len(sims)


""" Async """


@responses.activate
def test_async(mock_webapi, mock_job_status):
    # monkeypatch.setattr("tidy3d.web.api.container.Job.status", property(lambda self: "success"))
    sims = {TASK_NAME: make_heat_sim()}
    _ = run_async(sims, folder_name=PROJECT_NAME)
