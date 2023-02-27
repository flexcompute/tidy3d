import os
import tempfile

import responses
from responses import matchers

from tidy3d.web.environment import Env, EnvironmentConfig
from tidy3d.web.simulation_task import Folder, SimulationTask

test_env = EnvironmentConfig(
    name="test",
    web_api_endpoint="https://test",
    website_endpoint="https://test",
)

Env.set_current(test_env)


@responses.activate
def test_list_tasks():
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
def test_query_task():
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
def test_get_simulation_json(monkeypatch):
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
    with tempfile.NamedTemporaryFile() as temp:
        task.get_simulation_json(temp.name)
        assert os.path.getsize(temp.name) > 0


@responses.activate
def test_upload(monkeypatch):
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
def test_create():
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
        match=[matchers.json_params_matcher({"task_name": "test task", "call_back_url": None})],
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
def test_submit():
    responses.add(
        responses.GET,
        f"{Env.current.web_api_endpoint}/tidy3d/project",
        match=[matchers.query_param_matcher({"projectName": "test folder1"})],
        json={"data": {"projectId": "1234", "projectName": "test folder1"}},
        status=200,
    )
    responses.add(
        responses.POST,
        f"{Env.current.web_api_endpoint}/tidy3d/projects/1234/tasks",
        match=[matchers.json_params_matcher({"task_name": "test task", "call_back_url": None})],
        json={
            "data": {
                "taskId": "1234",
                "taskName": "test task",
                "createdAt": "2022-01-01T00:00:00.000Z",
            }
        },
        status=200,
    )
    responses.add(
        responses.POST,
        f"{Env.current.web_api_endpoint}/tidy3d/tasks/1234/submit",
        match=[
            matchers.json_params_matcher(
                {"solverVersion": None, "workerGroup": None, "protocolVersion": "1.6.3"}
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
    task = SimulationTask.create(None, "test task", "test folder1")
    task.submit(protocol_version="1.6.3")


@responses.activate
def test_estimate_cost():
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

    responses.add(
        responses.POST,
        f"{Env.current.web_api_endpoint}/tidy3d/tasks/3eb06d16-208b-487b-864b-e9b1d3e010a7/metadata",
        json={"data": {"flexUnit": 2.33}},
        status=200,
    )
    task = SimulationTask.get("3eb06d16-208b-487b-864b-e9b1d3e010a7")
    assert task.estimate_cost()["flexUnit"] == 2.33


@responses.activate
def test_running_info(monkeypatch):
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
    assert task.get_running_info() == (0.3, 5.7)


@responses.activate
def test_get_log(monkeypatch):
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
    with tempfile.NamedTemporaryFile() as temp:
        task.get_log(temp.name)
        assert os.path.getsize(temp.name) > 0
