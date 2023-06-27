import pytest
import os
import tempfile

import responses
from responses import matchers

from tidy3d.web import monitor
from tidy3d.web.environment import Env, EnvironmentConfig
from tidy3d.web.simulation_task import Folder, SimulationTask
from tidy3d.version import __version__

test_env = EnvironmentConfig(
    name="test",
    s3_region="test",
    web_api_endpoint="https://test",
    website_endpoint="https://test",
)

Env.set_current(test_env)


@pytest.fixture
def set_api_key(monkeypatch):
    """Set the api key."""
    import tidy3d.web.http_management as http_module

    monkeypatch.setattr(http_module, "api_key", lambda: "apikey")


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

    responses.add(
        responses.GET,
        f"{Env.current.web_api_endpoint}/tidy3d/tasks/xxx/detail",
        json={
            "data": {
                "taskId": "3eb06d16-208b-487b-864b-e9b1d3e010a7",
                "createdAt": "2022-01-01T00:00:00.000Z",
            }
        },
        status=404,
    )
    assert SimulationTask.get("xxx") is None


@responses.activate
def test_get_simulation_json(monkeypatch, set_api_key):
    def mock_download(*args, **kwargs):
        file_path = kwargs["to_file"]
        with open(file_path, "w") as f:
            f.write("test data")

    monkeypatch.setattr("tidy3d.web.simulation_task.download_file", mock_download)

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
    JSON_NAME = "tests/tmp/task.json"
    with open(JSON_NAME, "w") as f:
        task.get_simulation_json(JSON_NAME)
        assert os.path.getsize(JSON_NAME) > 0


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

    monkeypatch.setattr("tidy3d.web.simulation_task.upload_file", mock_download)
    task = SimulationTask.get("3eb06d16-208b-487b-864b-e9b1d3e010a7")
    with tempfile.NamedTemporaryFile() as temp:
        task.upload_file(temp.name, "temp.json")


@responses.activate
def test_create(set_api_key):
    responses.add(
        responses.GET,
        f"{Env.current.web_api_endpoint}/tidy3d/project",
        match=[matchers.query_param_matcher({"projectName": "test folder2"})],
        json={"data": {"projectId": "1234", "projectName": "test folder2"}},
        status=200,
    )
    responses.add(
        responses.POST,
        f"{Env.current.web_api_endpoint}/tidy3d/projects/1234/tasks",
        match=[
            matchers.json_params_matcher(
                {
                    "taskName": "test task",
                    "callbackUrl": None,
                    "simulationType": "tidy3d",
                    "parentTasks": None,
                }
            )
        ],
        json={
            "data": {
                "taskId": "1234",
                "taskName": "test task",
                "createdAt": "2022-01-01T00:00:00.000Z",
            }
        },
        status=200,
    )
    task = SimulationTask.create(None, "test task", "test folder2")
    assert task.task_id == "1234"


@responses.activate
def test_submit(set_api_key):
    project_id = "1234"
    task_id = "1234"
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
                    "taskName": task_name,
                    "callbackUrl": None,
                    "simulationType": "tidy3d",
                    "parentTasks": None,
                }
            )
        ],
        json={
            "data": {
                "taskId": task_id,
                "taskName": task_name,
                "createdAt": "2022-01-01T00:00:00.000Z",
            }
        },
        status=200,
    )
    responses.add(
        responses.POST,
        f"{Env.current.web_api_endpoint}/tidy3d/tasks/{task_id}/submit",
        match=[
            matchers.json_params_matcher(
                {"solverVersion": None, "workerGroup": None, "protocolVersion": __version__}
            )
        ],
        json={
            "data": {
                "taskId": task_id,
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
        f"{Env.current.web_api_endpoint}/tidy3d/tasks/{task_id}/detail",
        json={
            "taskId": task_id,
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
    task = SimulationTask.create(None, task_name, "test folder1")
    task.submit()
    # test DE need to open the comment
    # monitor(task_id, True)


@responses.activate
def test_estimate_cost(set_api_key):
    responses.add(
        responses.GET,
        f"{Env.current.web_api_endpoint}/tidy3d/tasks/3eb06d16-208b-487b-864b-e9b1d3e010a7/detail",
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
        f"{Env.current.web_api_endpoint}/tidy3d/tasks/3eb06d16-208b-487b-864b-e9b1d3e010a7/metadata",
        json={"data": {"flexUnit": 2.33}},
        status=200,
    )
    task = SimulationTask.get("3eb06d16-208b-487b-864b-e9b1d3e010a7")
    assert task.estimate_cost()["flexUnit"] == 2.33


@responses.activate
def test_get_log(monkeypatch, set_api_key):
    def mock(*args, **kwargs):
        file_path = kwargs["to_file"]
        with open(file_path, "w") as f:
            f.write("0.3,5.7")

    monkeypatch.setattr("tidy3d.web.simulation_task.download_file", mock)
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
    LOG_FNAME = "tests/tmp/test.log"
    with open(LOG_FNAME, "w") as f:
        task.get_log(LOG_FNAME)
        assert os.path.getsize(LOG_FNAME) > 0


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
