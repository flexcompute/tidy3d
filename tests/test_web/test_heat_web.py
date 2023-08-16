import pytest
import responses

import tidy3d as td

from tidy3d.web.heat import run as run_heat
from tidy3d.web.environment import Env
from ..test_components.test_heat import make_heat_sim, make_heat_sim_data


PROJECT_NAME = "Heat Solver"
HEATSIMULATION_NAME = "Unnamed Heat Simulation"
PROJECT_ID = "Project-ID"
TASK_ID = "Task-ID"


@pytest.fixture
def mock_remote_api(monkeypatch):
    def void(*args, **kwargs):
        return None

    def mock_download(task_id, remote_path, to_file, *args, **kwargs):
        heat_sim_data = make_heat_sim_data()
        heat_sim_data.to_file(to_file)

    monkeypatch.setattr(td.web.http_management, "api_key", lambda: "api_key")
    monkeypatch.setattr("tidy3d.web.heat.upload_file", void)
    monkeypatch.setattr("tidy3d.web.heat.download_file", mock_download)

    responses.add(
        responses.GET,
        f"{Env.current.web_api_endpoint}/tidy3d/project",
        match=[responses.matchers.query_param_matcher({"projectName": PROJECT_NAME})],
        json={"data": {"projectId": PROJECT_ID, "projectName": PROJECT_NAME}},
        status=200,
    )

    responses.add(
        responses.POST,
        f"{Env.current.web_api_endpoint}/tidy3d/heatsolver/py",
        match=[
            responses.matchers.json_params_matcher(
                {
                    "projectId": PROJECT_ID,
                    "heatSimulationName": HEATSIMULATION_NAME,
                    "fileType": "Hdf5",
                }
            )
        ],
        json={
            "data": {
                "id": TASK_ID,
                "status": "draft",
                "createdAt": "2023-05-19T16:47:57.190Z",
                "charge": 0,
                "fileType": "Hdf5",
            }
        },
        status=200,
    )

    responses.add(
        responses.POST,
        f"{Env.current.web_api_endpoint}/tidy3d/heatsolver/py",
        match=[
            responses.matchers.json_params_matcher(
                {
                    "projectId": PROJECT_ID,
                    "heatSimulationName": HEATSIMULATION_NAME,
                    "fileType": "Hdf5",
                }
            )
        ],
        json={
            "data": {
                "id": TASK_ID,
                "status": "draft",
                "createdAt": "2023-05-19T16:47:57.190Z",
                "charge": 0,
                "fileType": "Hdf5",
            }
        },
        status=200,
    )

    responses.add(
        responses.GET,
        f"{Env.current.web_api_endpoint}/tidy3d/heatsolver/py/{TASK_ID}",
        json={
            "data": {
                "id": TASK_ID,
                "status": "success",
                "createdAt": "2023-05-19T16:47:57.190Z",
                "charge": 0,
                "fileType": "Hdf5",
            }
        },
        status=200,
    )

    responses.add(
        responses.POST,
        f"{Env.current.web_api_endpoint}/tidy3d/heatsolver/py/{TASK_ID}/run",
        json={
            "data": {
                "id": TASK_ID,
                "status": "queued",
                "createdAt": "2023-05-19T16:47:57.190Z",
                "charge": 0,
                "fileType": "Hdf5",
            }
        },
        status=200,
    )


@responses.activate
def test_heat_solver_web(mock_remote_api):
    heat_sim = make_heat_sim()
    _ = run_heat(heat_simulation=heat_sim)


