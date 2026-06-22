from __future__ import annotations

import json
import tempfile
from types import SimpleNamespace

import pytest
import responses
from responses import matchers

import tidy3d as td
from tidy3d import config
from tidy3d.web.core import http_util
from tidy3d.web.core.exceptions import WebError
from tidy3d.web.core.task_core import BatchTask, Folder, SimulationTask
from tidy3d.web.core.types import PayType, TaskType

pytestmark = pytest.mark.usefixtures("use_test_profile")


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


def test_heat_parent_task_validation_rejects_non_mesh_parent(monkeypatch):
    task = SimulationTask(taskId="heat-task", taskType=TaskType.HEAT.name)

    def fake_get(task_id, **kwargs):
        if task_id == "parent-task":
            return SimpleNamespace(
                task_type=TaskType.FDTD.name,
                status="success",
                childFileMd5="mesh-md5",
            )
        return SimpleNamespace(fileMd5="heat-md5")

    monkeypatch.setattr(SimulationTask, "get", staticmethod(fake_get))

    with pytest.raises(WebError, match="VolumeMesher"):
        task.validate_post_upload(parent_tasks=["parent-task"])


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
        f"{config.web.api_endpoint}/tidy3d/projects",
        json={"data": [{"projectId": "1234", "projectName": "default"}]},
        status=200,
    )

    resp = Folder.list()
    assert resp is not None

    responses.add(
        responses.GET,
        f"{config.web.api_endpoint}/tidy3d/projects/1234/tasks",
        json={"data": [{"taskId": "1234", "createdAt": "2022-01-01T00:00:00.000Z"}]},
        status=200,
    )
    tasks = resp[0].list_tasks()
    assert tasks is not None


@responses.activate
def test_query_task(set_api_key):
    responses.add(
        responses.GET,
        f"{config.web.api_endpoint}/tidy3d/tasks/3eb06d16-208b-487b-864b-e9b1d3e010a7/detail",
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

    monkeypatch.setattr("tidy3d.web.core.task_core.download_gz_file", mock_download)

    responses.add(
        responses.GET,
        f"{config.web.api_endpoint}/tidy3d/tasks/3eb06d16-208b-487b-864b-e9b1d3e010a7/detail",
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
        f"{config.web.api_endpoint}/tidy3d/tasks/3eb06d16-208b-487b-864b-e9b1d3e010a7/detail",
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
        f"{config.web.api_endpoint}/tidy3d/project",
        match=[matchers.query_param_matcher({"projectName": "test folder2"})],
        json={"data": {"projectId": "1234", "projectName": "test folder2"}},
        status=200,
    )
    responses.add(
        responses.POST,
        f"{config.web.api_endpoint}/tidy3d/projects/{task_id}/tasks",
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
        f"{config.web.api_endpoint}/tidy3d/project",
        match=[matchers.query_param_matcher({"projectName": "test folder1"})],
        json={"data": {"projectId": project_id, "projectName": "test folder1"}},
        status=200,
    )
    responses.add(
        responses.POST,
        f"{config.web.api_endpoint}/tidy3d/projects/{project_id}/tasks",
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
        f"{config.web.api_endpoint}/tidy3d/tasks/{TASK_ID}/submit",
        match=[
            matchers.json_params_matcher(
                {
                    "protocolVersion": http_util.get_version(),
                    "solverVersion": None,
                    "workerGroup": None,
                    "enableCaching": config.web.enable_caching,
                    "payType": PayType.AUTO,
                    "priority": None,
                    "vgpuAllocation": None,
                    "ignoreMemoryLimit": None,
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
        f"{config.web.api_endpoint}/tidy3d/tasks/{TASK_ID}/detail",
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
def test_batch_get_preserves_status(set_api_key):
    responses.add(
        responses.GET,
        f"{config.web.api_endpoint}/rf/task/batch-task-id/statistics",
        json={
            "data": {
                "taskId": "batch-task-id",
                "taskType": TaskType.TERMINAL_CM.name,
                "status": "diverged",
            }
        },
        status=200,
    )

    task = BatchTask.get("batch-task-id")

    assert task.status == "diverged"
    assert task.task_type == TaskType.TERMINAL_CM.name


@responses.activate
def test_batch_submit_additional_payload(set_api_key):
    task = BatchTask(taskId="batch-task-id")
    responses.add(
        responses.POST,
        f"{config.web.api_endpoint}/rf/task/batch-task-id/submit",
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
        json={"data": {"taskId": "batch-task-id"}},
        status=200,
    )

    task.submit(additional_payload={"routeHint": "batch"})


@responses.activate
def test_batch_submit_pay_type(set_api_key):
    task = BatchTask(taskId="batch-task-id")
    responses.add(
        responses.POST,
        f"{config.web.api_endpoint}/rf/task/batch-task-id/submit",
        match=[
            matchers.json_params_matcher(
                {
                    "solverVersion": None,
                    "protocolVersion": td.version.__version__,
                    "workerGroup": None,
                    "payType": PayType.CREDITS.value,
                    "priority": 5,
                    "vgpuAllocation": 4,
                }
            )
        ],
        json={"data": {"taskId": "batch-task-id"}},
        status=200,
    )

    task.submit(pay_type="FLEX_CREDIT", priority=5, vgpu_allocation=4)


@pytest.mark.parametrize("priority", [0, -1, 11])
def test_batch_submit_invalid_priority(priority):
    task = BatchTask(taskId="batch-task-id")

    with pytest.raises(ValueError, match=r"Priority must be between"):
        task.submit(priority=priority)


@pytest.mark.parametrize("vgpu_allocation", [0, 3, 9])
def test_batch_submit_invalid_vgpu_allocation(vgpu_allocation):
    task = BatchTask(taskId="batch-task-id")

    with pytest.raises(ValueError, match=r"vgpu_allocation must be one of"):
        task.submit(vgpu_allocation=vgpu_allocation)


@responses.activate
def test_pay_type_case_insensitivity(set_api_key):
    """Test PayType enum's case-insensitive behavior with different string formats."""
    project_id = "1234"
    TASK_ID = "5678"
    task_name = "test pay type"

    responses.add(
        responses.GET,
        f"{config.web.api_endpoint}/tidy3d/project",
        match=[matchers.query_param_matcher({"projectName": "test pay type folder"})],
        json={"data": {"projectId": project_id, "projectName": "test pay type folder"}},
        status=200,
    )
    responses.add(
        responses.POST,
        f"{config.web.api_endpoint}/tidy3d/projects/{project_id}/tasks",
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
        f"{config.web.api_endpoint}/tidy3d/tasks/{TASK_ID}/submit",
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

    task = SimulationTask.create(TaskType.FDTD, task_name, "test pay type folder")

    valid_pay_types = [
        "auto",
        "AUTO",
        PayType.AUTO,
        "credits",
        "CREDITS",
        PayType.CREDITS,
        "vgpu",
        "VGPU",
        PayType.VGPU,
    ]

    for pay_type in valid_pay_types:
        task.submit(pay_type=pay_type)


@responses.activate
def test_estimate_cost(set_api_key):
    TASK_ID = "3eb06d16-208b-487b-864b-e9b1d3e010a7"
    responses.add(
        responses.GET,
        f"{config.web.api_endpoint}/tidy3d/tasks/{TASK_ID}/detail",
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
        f"{config.web.api_endpoint}/tidy3d/tasks/{TASK_ID}/metadata",
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
        f"{config.web.api_endpoint}/tidy3d/tasks/3eb06d16-208b-487b-864b-e9b1d3e010a7/detail",
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
    with open(LOG_FNAME) as f:
        assert f.read() == "0.3,5.7"


@responses.activate
def test_get_running_tasks(set_api_key):
    responses.add(
        responses.GET,
        f"{config.web.api_endpoint}/tidy3d/py/tasks",
        json={"data": [{"taskId": "1234", "status": "queued"}]},
        status=200,
    )

    tasks = SimulationTask.get_running_tasks()
    assert len(tasks) == 1
