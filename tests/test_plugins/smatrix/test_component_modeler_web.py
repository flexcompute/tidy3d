# Tests webapi and things that depend on it
from __future__ import annotations

from types import SimpleNamespace

import pytest
import responses
from responses import matchers

import tidy3d as td
from tests.test_plugins.smatrix.test_component_modeler import make_component_modeler
from tidy3d.web import common
from tidy3d.web.api.run import run
from tidy3d.web.api.webapi import (
    estimate_cost,
    start,
    upload,
)
from tidy3d.web.core.environment import Env
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

task_core_path = "tidy3d.web.core.task_core"
api_path = "tidy3d.web.api.webapi"

Env.dev.active()


@pytest.fixture
def set_api_key(monkeypatch):
    """Set the api key."""
    import tidy3d.web.core.http_util as http_module

    monkeypatch.setattr(http_module, "api_key", lambda: "apikey")
    monkeypatch.setattr(http_module, "get_version", lambda: td.version.__version__)


@pytest.fixture
def mock_modeler_estimate_cost(monkeypatch, set_api_key):
    """Mocks API calls for component modeler cost estimation."""
    batch_id = "bm-estimate-123"

    # From upload()
    responses.add(
        responses.GET,
        f"{Env.current.web_api_endpoint}/tidy3d/project",
        json={"data": {"projectId": FOLDER_ID, "projectName": PROJECT_NAME}},
        status=200,
    )
    responses.add(
        responses.POST,
        f"{Env.current.web_api_endpoint}/tidy3d/projects/{FOLDER_ID}/tasks",
        json={"data": {"taskId": TASK_ID, "batchId": batch_id}},
        status=200,
    )

    def mock_upload_sim(*args, **kwargs):
        pass

    monkeypatch.setattr(
        "tidy3d.web.core.task_core.SimulationTask.upload_simulation", mock_upload_sim
    )

    responses.add(
        responses.POST,
        f"{Env.current.web_api_endpoint}/tidy3d/projects/component-modeler-split",
        json={"data": {"taskIds": [TASK_ID]}},
        status=200,
    )
    responses.add(
        responses.POST, f"{Env.current.web_api_endpoint}/tidy3d/batch/{batch_id}/check", status=200
    )

    # from estimate_cost()
    responses.add(
        responses.GET,
        f"{Env.current.web_api_endpoint}/tidy3d/batch/{batch_id}/detail",
        json={"data": {"totalStatus": "validate_success", "estFlexUnit": 1.23}},
        status=200,
    )

    # from start()
    responses.add(
        responses.POST, f"{Env.current.web_api_endpoint}/tidy3d/batch/{batch_id}/submit", status=200
    )


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
def mock_webapi(mock_metadata, mock_get_info, mock_start, mock_monitor, mock_download, mock_load):
    """Mocks all webapi operation."""


@pytest.fixture
def mock_component_modeler_run(monkeypatch, set_api_key):
    """Mocks the component modeler run process."""

    batch_id = "bm-test-batch-id"

    # Mock SimulationTask.create to return a batchId starting with "bm-"
    responses.add(
        responses.POST,
        f"{Env.current.web_api_endpoint}/tidy3d/projects/{FOLDER_ID}/tasks",
        json={"data": {"taskId": TASK_ID, "batchId": batch_id}},
        status=200,
    )

    # Monkeypatch upload_simulation to do nothing
    def mock_upload_simulation(*args, **kwargs):
        pass

    monkeypatch.setattr(
        "tidy3d.web.core.task_core.SimulationTask.upload_simulation", mock_upload_simulation
    )

    # Mock for component-modeler-split
    responses.add(
        responses.POST,
        f"{Env.current.web_api_endpoint}/tidy3d/projects/component-modeler-split",
        match=[matchers.json_params_matcher({"batchId": batch_id}, strict_match=False)],
        json={"data": {"taskIds": [TASK_ID, "5678"]}},
        status=200,
    )

    # Mock for batch.check()
    responses.add(
        responses.POST,
        f"{Env.current.web_api_endpoint}/tidy3d/batch/{batch_id}/check",
        status=200,
    )

    # Mock for batch-detail (used by estimate_cost, start, monitor)
    responses.add(
        responses.GET,
        f"{Env.current.web_api_endpoint}/tidy3d/batch/{batch_id}/detail",
        json={"data": {"totalStatus": "validate_success", "estFlexUnit": 1.23}},
        status=200,
    )

    # Mock for batch-submit
    responses.add(
        responses.POST,
        f"{Env.current.web_api_endpoint}/tidy3d/batch/{batch_id}/submit",
        json={"data": {"batchId": batch_id}},
        status=200,
    )

    # Mock for monitor to show completion
    responses.add(
        responses.GET,
        f"{Env.current.web_api_endpoint}/tidy3d/batch/{batch_id}/detail",
        json={
            "data": {
                "status": "completed",
                "tasks": [{"taskId": TASK_ID}, {"taskId": "5678"}],
                "totalStatus": "success",
                "runSuccess": 2,
                "totalTask": 2,
            }
        },
        status=200,
    )

    # Mock webapi.load to return a mock object that satisfies the test's assertions
    mock_result = SimpleNamespace(
        batch_id=batch_id,
        status="completed",
        tasks=[{"taskId": TASK_ID}, {"taskId": "5678"}],
    )

    def mock_load(*args, **kwargs):
        return mock_result

    monkeypatch.setattr("tidy3d.web.api.webapi.load", mock_load)


@responses.activate
def test_modeler_estimate_cost(mock_modeler_estimate_cost):
    """Tests estimating cost for a component modeler."""
    comp_modeler = make_component_modeler()
    task_id = upload(comp_modeler, TASK_NAME, PROJECT_NAME)
    assert task_id.startswith("bm-")

    try:
        start(task_id)
        cost = estimate_cost(task_id)
        assert isinstance(cost, float)
        assert cost > 0
    finally:
        # web.delete is not implemented for batches, so we can't test it here yet.
        pass


@responses.activate
def test_component_modeler_run(mock_component_modeler_run):
    """Tests running a component modeler simulation."""
    comp_modeler = make_component_modeler()
    batch = run(simulation=comp_modeler, folder_name="test", task_name="test")
    assert batch.batch_id.startswith("bm-")
    assert batch.status == "completed"
    assert len(batch.tasks) == 2
