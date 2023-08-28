import pytest
import tempfile

import responses
from responses import matchers

import tidy3d as td
from tidy3d.web.core import http_util
from tidy3d.web.core.environment import Env, EnvironmentConfig
from tidy3d.web.core.task_core import Folder, SimulationTask
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


@pytest.fixture
def set_api_key(monkeypatch):
    """Set the api key."""
    import tidy3d.web.core.http_util as httputil

    monkeypatch.setattr(httputil, "api_key", lambda: "apikey")
    monkeypatch.setattr(httputil, "get_version", lambda: td.version.__version__)


@responses.activate
def test_list_tasks(set_api_key):
    responses.add(
        responses.GET,
        f"{Env.current.web_api_endpoint}/tidy3d/projects",
        json={"data": [{"projectId": "1234", "projectName": "default"}]},
        status=200,
    )

    resp = Folder.list()
    assert resp is not None

    responses.add(
        responses.GET,
        f"{Env.current.web_api_endpoint}/tidy3d/projects/1234/tasks",
        json={"data": [{"taskId": "1234", "createdAt": "2022-01-01T00:00:00.000Z"}]},
        status=200,
    )
    tasks = resp[0].list_tasks()
    assert tasks is not None


@responses.activate
def test_query_task(set_api_key):
    responses.add(
        responses.GET,
        f"{Env.current.web_api_endpoint}/tidy3d/tasks/3eb06d16-208b-487b-864b-e9b1d3e010a7/detail",
        json={
            "data": {
                "taskId": "3eb06d16-208b-487b-864b-e9b1d3e010a7",
                "createdAt": "2022-01-01T00:00:00.000Z",
            }
        },
        status=200,
    )

    task = SimulationTask.get("3eb06d16-208b-487b-864b-e9b1d3e010a7")
    assert task


@responses.activate
def test_get_simulation_json(monkeypatch, set_api_key, tmp_path):
    sim = make_sim()

    def mock_download(*args, **kwargs):
        to_file = kwargs["to_file"]
        sim.to_file(to_file)

    monkeypatch.setattr("tidy3d.web.core.task_core.download_file", mock_download)

    responses.add(
        responses.GET,
        f"{Env.current.web_api_endpoint}/tidy3d/tasks/3eb06d16-208b-487b-864b-e9b1d3e010a7/detail",
        json={
            "data": {
                "taskId": "3eb06d16-208b-487b-864b-e9b1d3e010a7",
                "createdAt": "2022-01-01T00:00:00.000Z",
            }
        },
        status=200,
    )
    task = SimulationTask.get("3eb06d16-208b-487b-864b-e9b1d3e010a7")
    JSON_NAME = str(tmp_path / "task.json")
    task.get_simulation_json(JSON_NAME)
    assert td.Simulation.from_file(JSON_NAME) == sim


@responses.activate
def test_upload(monkeypatch, set_api_key):
    responses.add(
        responses.GET,
        f"{Env.current.web_api_endpoint}/tidy3d/tasks/3eb06d16-208b-487b-864b-e9b1d3e010a7/detail",
        json={
            "data": {
                "taskId": "3eb06d16-208b-487b-864b-e9b1d3e010a7",
                "createdAt": "2022-01-01T00:00:00.000Z",
            }
        },
        status=200,
    )

    def mock_download(*args, **kwargs):
        pass

    monkeypatch.setattr("tidy3d.web.core.task_core.upload_file", mock_download)
    task = SimulationTask.get("3eb06d16-208b-487b-864b-e9b1d3e010a7")
    with tempfile.NamedTemporaryFile() as temp:
        task.upload_file(temp.name, "temp.json")


@responses.activate
def test_create(set_api_key):
    task_id = "1234"
    responses.add(
        responses.GET,
        f"{Env.current.web_api_endpoint}/tidy3d/project",
        match=[matchers.query_param_matcher({"projectName": "test folder2"})],
        json={"data": {"projectId": "1234", "projectName": "test folder2"}},
        status=200,
    )
    responses.add(
        responses.POST,
        f"{Env.current.web_api_endpoint}/tidy3d/projects/{task_id}/tasks",
        match=[
            matchers.json_params_matcher(
                {
                    "taskType": TaskType.FDTD,
                    "taskName": "test task",
                    "callbackUrl": None,
                    "fileType": "Gz",
                    "simulationType": "tidy3d",
                    "parentTasks": None,
                }
            )
        ],
        json={
            "data": {
                "taskId": task_id,
                "taskName": "test task",
                "createdAt": "2022-01-01T00:00:00.000Z",
            }
        },
        status=200,
    )
    task = SimulationTask.create(TaskType.FDTD, "test task", "test folder2")
    assert task.task_id == task_id


@responses.activate
def test_submit(set_api_key):
    project_id = "1234"
    TASK_ID = "1234"
    task_name = "test task"
    responses.add(
        responses.GET,
        f"{Env.current.web_api_endpoint}/tidy3d/project",
        match=[matchers.query_param_matcher({"projectName": "test folder1"})],
        json={"data": {"projectId": project_id, "projectName": "test folder1"}},
        status=200,
    )
    responses.add(
        responses.POST,
        f"{Env.current.web_api_endpoint}/tidy3d/projects/{project_id}/tasks",
        match=[
            matchers.json_params_matcher(
                {
                    "taskType": TaskType.FDTD,
                    "taskName": task_name,
                    "callbackUrl": None,
                    "fileType": "Gz",
                    "simulationType": "tidy3d",
                    "parentTasks": None,
                }
            )
        ],
        json={
            "data": {
                "taskId": TASK_ID,
                "taskName": task_name,
                "createdAt": "2022-01-01T00:00:00.000Z",
            }
        },
        status=200,
    )
    responses.add(
        responses.POST,
        f"{Env.current.web_api_endpoint}/tidy3d/tasks/{TASK_ID}/submit",
        match=[
            matchers.json_params_matcher(
                {
                    "protocolVersion": http_util.get_version(),
                    "solverVersion": None,
                    "workerGroup": None,
                }
            )
        ],
        json={
            "data": {
                "taskId": TASK_ID,
                "taskName": task_name,
                "createdAt": "2022-01-01T00:00:00.000Z",
                "taskBlockInfo": {
                    "chargeType": "free",
                    "maxFreeCount": 20,
                    "maxGridPoints": 1000,
                    "maxTimeSteps": 1000,
                },
            }
        },
        status=200,
    )
    responses.add(
        responses.GET,
        f"{Env.current.web_api_endpoint}/tidy3d/tasks/{TASK_ID}/detail",
        json={
            "taskId": TASK_ID,
            "taskName": task_name,
            "createdAt": "2022-01-01T00:00:00.000Z",
            "status": "running",
            "taskBlockInfo": {
                "chargeType": "free",
                "maxFreeCount": 20,
                "maxGridPoints": 1000,
                "maxTimeSteps": 1000,
            },
        },
        status=200,
    )
    task = SimulationTask.create(TaskType.FDTD, task_name, "test folder1")
    task.submit()
    # test DE need to open the comment
    # monitor(TASK_ID, True)


@responses.activate
def test_estimate_cost(set_api_key):
    TASK_ID = "3eb06d16-208b-487b-864b-e9b1d3e010a7"
    responses.add(
        responses.GET,
        f"{Env.current.web_api_endpoint}/tidy3d/tasks/{TASK_ID}/detail",
        json={
            "data": {
                "taskId": "3eb06d16-208b-487b-864b-e9b1d3e010a7",
                "createdAt": "2022-01-01T00:00:00.000Z",
                "taskBlockInfo": {
                    "chargeType": "free",
                    "maxFreeCount": 20,
                    "maxGridPoints": 1000,
                    "maxTimeSteps": 1000,
                },
            }
        },
        status=200,
    )

    responses.add(
        responses.POST,
        f"{Env.current.web_api_endpoint}/tidy3d/tasks/{TASK_ID}/metadata",
        json={"data": {"flexUnit": 2.33}},
        status=200,
    )
    task = SimulationTask(taskId=TASK_ID)
    assert task.estimate_cost()["flexUnit"] == 2.33


@responses.activate
def test_get_log(monkeypatch, set_api_key, tmp_path):
    def mock(*args, **kwargs):
        file_path = kwargs["to_file"]
        with open(file_path, "w") as f:
            f.write("0.3,5.7")

    monkeypatch.setattr("tidy3d.web.core.task_core.download_file", mock)
    responses.add(
        responses.GET,
        f"{Env.current.web_api_endpoint}/tidy3d/tasks/3eb06d16-208b-487b-864b-e9b1d3e010a7/detail",
        json={
            "data": {
                "taskId": "3eb06d16-208b-487b-864b-e9b1d3e010a7",
                "createdAt": "2022-01-01T00:00:00.000Z",
            }
        },
        status=200,
    )
    task = SimulationTask.get("3eb06d16-208b-487b-864b-e9b1d3e010a7")
    LOG_FNAME = str(tmp_path / "test.log")
    task.get_log(LOG_FNAME)
    with open(LOG_FNAME, "r") as f:
        assert f.read() == "0.3,5.7"


@responses.activate
def test_get_running_tasks(set_api_key):
    responses.add(
        responses.GET,
        f"{Env.current.web_api_endpoint}/tidy3d/py/tasks",
        json={"data": [{"taskId": "1234", "status": "queued"}]},
        status=200,
    )

    tasks = SimulationTask.get_running_tasks()
    assert len(tasks) == 1
