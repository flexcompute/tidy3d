import tempfile

import responses
import tidy3d as td
from responses import matchers

from tidy3d import Simulation
from tidy3d.web.environment import Env
from tidy3d.web.webapi import (
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
    start,
    upload,
)


def make_sim():
    """Makes a simulation."""
    return td.Simulation(size=(1, 1, 1), grid_spec=td.GridSpec.auto(wavelength=1.0), run_time=1e-12)


@responses.activate
def test_upload(monkeypatch):
    responses.add(
        responses.GET,
        f"{Env.current.web_api_endpoint}/tidy3d/project",
        match=[matchers.query_param_matcher({"projectName": "test webapi folder"})],
        json={"data": {"projectId": "1234", "projectName": "test webapi folder"}},
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

    def mock_download(*args, **kwargs):
        pass

    monkeypatch.setattr("tidy3d.web.simulation_task.upload_string", mock_download)

    sim = make_sim()
    assert upload(sim, "test task", "test webapi folder")


@responses.activate
def test_get_info():
    responses.add(
        responses.GET,
        f"{Env.current.web_api_endpoint}/tidy3d/tasks/abcd/detail",
        json={
            "data": {
                "taskId": "abcd",
                "createdAt": "2022-01-01T00:00:00.000Z",
            }
        },
        status=200,
    )
    assert get_info("abcd").taskId == "abcd"


@responses.activate
def test_start():
    responses.add(
        responses.GET,
        f"{Env.current.web_api_endpoint}/tidy3d/tasks/1234/detail",
        json={
            "data": {
                "taskId": "1234",
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
                {
                    "solverVersion": None,
                    "workerGroup": None,
                    "protocolVersion": td.version.__version__,
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
    start("1234")


@responses.activate
def test_get_run_info(monkeypatch):
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
    assert get_run_info("3eb06d16-208b-487b-864b-e9b1d3e010a7") == (0.3, 5.7)


@responses.activate
def test_download(monkeypatch):
    responses.add(
        responses.GET,
        f"{Env.current.web_api_endpoint}/tidy3d/tasks/abcd/detail",
        json={
            "data": {
                "taskId": "abcd",
                "createdAt": "2022-01-01T00:00:00.000Z",
            }
        },
        status=200,
    )

    def mock_download(*args, **kwargs):
        file_path = kwargs["to_file"]
        with open(file_path, "w") as f:
            f.write("0.3,5.7")

    monkeypatch.setattr("tidy3d.web.simulation_task.download_file", mock_download)
    with tempfile.NamedTemporaryFile() as f:
        download("abcd", f.name)
        assert f.read() == b"0.3,5.7"


@responses.activate
def _test_load(monkeypatch):
    responses.add(
        responses.GET,
        f"{Env.current.web_api_endpoint}/tidy3d/tasks/abcd/detail",
        json={
            "data": {
                "taskId": "abcd",
                "createdAt": "2022-01-01T00:00:00.000Z",
            }
        },
        status=200,
    )

    def mock_download(*args, **kwargs):
        pass

    monkeypatch.setattr("tidy3d.web.simulation_task.download_file", mock_download)
    load("abcd", "data/monitor_data.hdf5")


@responses.activate
def test_delete():
    responses.add(
        responses.GET,
        f"{Env.current.web_api_endpoint}/tidy3d/tasks/abcd/detail",
        json={
            "data": {
                "taskId": "abcd",
                "createdAt": "2022-01-01T00:00:00.000Z",
            }
        },
        status=200,
    )

    responses.add(
        responses.DELETE,
        f"{Env.current.web_api_endpoint}/tidy3d/tasks/abcd",
        json={
            "data": {
                "taskId": "abcd",
                "createdAt": "2022-01-01T00:00:00.000Z",
            }
        },
        status=200,
    )

    assert delete("abcd").taskId == "abcd"


@responses.activate
def test_estimate_cost():
    responses.add(
        responses.GET,
        f"{Env.current.web_api_endpoint}/tidy3d/tasks/abcd/detail",
        json={
            "data": {
                "taskId": "abcd",
                "createdAt": "2022-01-01T00:00:00.000Z",
            }
        },
        status=200,
    )
    responses.add(
        responses.POST,
        f"{Env.current.web_api_endpoint}/tidy3d/tasks/abcd/metadata",
        json={
            "data": {
                "flex_unit": 11.11,
                "createdAt": "2022-01-01T00:00:00.000Z",
            }
        },
        status=200,
    )
    assert estimate_cost("abcd") == 11.11


@responses.activate
def test_download_json(monkeypatch):
    responses.add(
        responses.GET,
        f"{Env.current.web_api_endpoint}/tidy3d/tasks/abcd/detail",
        json={
            "data": {
                "taskId": "abcd",
                "createdAt": "2022-01-01T00:00:00.000Z",
            }
        },
        status=200,
    )

    def mock_download(*args, **kwargs):
        file_path = kwargs["to_file"]
        with open(file_path, "w") as f:
            f.write("0.3,5.7")

    monkeypatch.setattr("tidy3d.web.simulation_task.download_file", mock_download)
    with tempfile.NamedTemporaryFile() as f:
        download_json("abcd", f.name)
        assert f.read() == b"0.3,5.7"


@responses.activate
def test_load_simulation(monkeypatch):
    responses.add(
        responses.GET,
        f"{Env.current.web_api_endpoint}/tidy3d/tasks/abcd/detail",
        json={
            "data": {
                "taskId": "abcd",
                "createdAt": "2022-01-01T00:00:00.000Z",
            }
        },
        status=200,
    )

    def mock_download(*args, **kwargs):
        make_sim().to_file(args[1])

    monkeypatch.setattr(
        "tidy3d.web.simulation_task.SimulationTask.get_simulation_json", mock_download
    )
    with tempfile.NamedTemporaryFile(suffix=".json") as f:
        assert load_simulation("abcd", f.name)


@responses.activate
def test_download_log(monkeypatch):
    responses.add(
        responses.GET,
        f"{Env.current.web_api_endpoint}/tidy3d/tasks/abcd/detail",
        json={
            "data": {
                "taskId": "abcd",
                "createdAt": "2022-01-01T00:00:00.000Z",
            }
        },
        status=200,
    )

    def mock(*args, **kwargs):
        file_path = kwargs["to_file"]
        with open(file_path, "w") as f:
            f.write("0.3,5.7")

    monkeypatch.setattr("tidy3d.web.simulation_task.download_file", mock)

    with tempfile.NamedTemporaryFile() as f:
        download_log("abcd", f.name)
        assert f.read() == b"0.3,5.7"


@responses.activate
def test_delete_old():
    responses.add(
        responses.GET,
        f"{Env.current.web_api_endpoint}/tidy3d/project",
        match=[matchers.query_param_matcher({"projectName": "default"})],
        json={"data": {"projectId": "abcd", "projectName": "default"}},
        status=200,
    )
    responses.add(
        responses.GET,
        f"{Env.current.web_api_endpoint}/tidy3d/projects/abcd/tasks",
        json={"data": [{"taskId": "abcd", "createdAt": "2022-01-01T00:00:00.000Z"}]},
        status=200,
    )

    responses.add(
        responses.DELETE,
        f"{Env.current.web_api_endpoint}/tidy3d/tasks/abcd",
        json={
            "data": {
                "taskId": "abcd",
                "createdAt": "2022-01-01T00:00:00.000Z",
            }
        },
        status=200,
    )

    delete_old(100)


@responses.activate
def test_get_tasks():
    responses.add(
        responses.GET,
        f"{Env.current.web_api_endpoint}/tidy3d/project",
        match=[matchers.query_param_matcher({"projectName": "default"})],
        json={"data": {"projectId": "abcd", "projectName": "default"}},
        status=200,
    )
    responses.add(
        responses.GET,
        f"{Env.current.web_api_endpoint}/tidy3d/projects/abcd/tasks",
        json={"data": [{"taskId": "abcd", "createdAt": "2022-01-01T00:00:00.000Z"}]},
        status=200,
    )

    assert get_tasks(1)[0]["task_id"] == "abcd"
