"""Tests webapi bits that dont require authentication."""
import pytest
import requests
import datetime

from requests import Session

import tidy3d as td
import tidy3d.web as web
from tidy3d.web import httputils, s3utils, webapi
from tidy3d.log import DataError, WebError

from ..utils import clear_tmp

TASK_NAME = "TASK_NAME"
TASK_ID = "TASK_ID"
PROJECT_ID = "PROJECT_ID"
PROJECT_NAME = "PROJECT_NAME"
FOLDER_NAME = "FOLDER_NAME"


class MockResponse:
    """generic response to a requests function."""

    status_code = 200

    @staticmethod
    def json():
        return {}

    def raise_for_status(self):
        pass


class MockResponseInfoOk(MockResponse):
    """response if web.getinfo(task_id) and task_id found"""

    @staticmethod
    def json():
        return {"taskId": TASK_ID}


class MockResponseInfoNotFound(MockResponse):
    """response if web.getinfo(task_id) and task_id not found"""

    @staticmethod
    def json():
        return {"data": None}


class MockResponseUpload(MockResponse):
    """response if web.upload()"""

    @staticmethod
    def json():
        return {"taskId": TASK_ID}


# class MockResponseUploadFailure(MockResponse):
#     """response if web.upload() faile"""

#     @staticmethod
#     def json():
#         raise requests.exceptions.HTTPError()


class MockResponseStart(MockResponse):
    """response if web.start()"""

    @staticmethod
    def json():
        return {"data": None}


# class MockResponseUploadString(MockResponse):
#     @staticmethod
#     def json():
#         return {
#             "cloudpath": "",
#             "userCredentials":
#                 dict(
#                     expiration=0.0,
#                     sessionToken='TOKEN',
#                     secretAccessKey='SECRET_KEY',
#                     accessKeyId='ACCESS_ID'
#                 )
#             }


class MockResponseFolder(MockResponse):
    @staticmethod
    def json():
        return {"projectId": PROJECT_ID, "projectName": PROJECT_NAME}


# map method path to the proper mocj response
RESPONSE_MAP = {
    # get responses
    f"tidy3d/tasks/{TASK_ID}/detail": MockResponseInfoOk(),
    f"tidy3d/tasks/None/detail": MockResponseInfoNotFound(),
    f"tidy3d/project?projectName={FOLDER_NAME}": MockResponseFolder(),
    # f'tidy3d/tasks/{TASK_ID}/file?filename=simulation.json': MockResponseUploadString()
    # post responses
    f"tidy3d/projects/{PROJECT_ID}/tasks": MockResponseUpload(),
    # f'tidy3d/projects/FAIL/tasks': MockResponseUploadFailure(),
    f"tidy3d/tasks/{TASK_ID}/submit": MockResponseStart(),
}


# monkeypatched requests.get moved to a fixture
@pytest.fixture
def mock_response(monkeypatch):
    """Requests.get() mocked to return {'mock_key':'mock_response'}."""

    def get_response(url: str) -> str:
        """Get the method path from a full url."""
        preamble = "https://tidy3d-api.simulation.cloud/"
        method = url.split(preamble)[-1]
        return RESPONSE_MAP[method]

    class MockRequests:
        def get(self, url, **kwargs):
            return get_response(url)

        def post(self, url, **kwargs):
            return get_response(url)

    monkeypatch.setattr(
        httputils, "get_headers", lambda: {"Authorization": None, "Application": "TIDY3D"}
    )
    monkeypatch.setattr(webapi, "upload_string", lambda *args, **kwargs: None)
    monkeypatch.setattr(httputils, "session", MockRequests())


def make_sim():
    """Makes a simulation."""
    return td.Simulation(size=(1, 1, 1), grid_spec=td.GridSpec.auto(wavelength=1.0), run_time=1e-12)


def test_get_info(mock_response):
    info = web.get_info(TASK_ID)
    assert info.taskId == TASK_ID


def test_get_info_none(mock_response):
    with pytest.raises(WebError):
        info = web.get_info("None")


def test_upload(mock_response):
    sim = make_sim()
    web.webapi.upload(sim, task_name=TASK_NAME, folder_name=FOLDER_NAME)


# def test_upload_fail(mock_response, monkeypatch):
#     sim = make_sim()
#     monkeypatch.setattr(webapi.DEFAULT_CONFIG, 'solver_version', 'test.test.test')
#     with pytest.raises(WebError):
#         web.webapi.upload(sim, task_name=TASK_NAME, folder_name=FOLDER_NAME)


def test_start(mock_response, monkeypatch):
    monkeypatch.setattr(webapi.DEFAULT_CONFIG, "solver_version", "test.test.test")
    web.webapi.start(TASK_ID)
    monkeypatch.setattr(webapi.DEFAULT_CONFIG, "solver_version", None)
    web.webapi.start(TASK_ID)
    monkeypatch.setattr(webapi.DEFAULT_CONFIG, "worker_group", "worker_group")
    web.webapi.start(TASK_ID)


def test_monitor(mock_response):
    web.webapi.start(TASK_ID)


def _test_job():
    """tests creation of a job."""
    sim = make_sim()
    j = web.Job(simulation=sim, task_name="test")


def _test_batch():
    """tests creation of a batch."""
    sim = make_sim()
    b = web.Batch(simulations={"test": sim})


@clear_tmp
def _test_batchdata_load():
    """Tests loading of a batch data from file."""
    sim = make_sim()
    b = web.Batch(simulations={"test": sim})
    b.to_file("tests/tmp/batch.json")
    with pytest.raises(DataError):
        web.BatchData.load(path_dir="tests/tmp")
