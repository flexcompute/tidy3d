import pytest
import responses
from responses import matchers
import tidy3d as td
from tidy3d.web.core.task_core import Folder
from tidy3d.web.core.environment import Env

Env.dev.active()


@pytest.fixture
def set_api_key(monkeypatch):
    """Set the api key."""
    import tidy3d.web.core.http_util as httputil

    monkeypatch.setattr(httputil, "api_key", lambda: "apikey")
    monkeypatch.setattr(httputil, "get_version", lambda: td.version.__version__)


@responses.activate
def test_list_folders(set_api_key):
    responses.add(
        responses.GET,
        f"{Env.current.web_api_endpoint}/tidy3d/projects",
        json={"data": [{"projectId": "1234", "projectName": "default"}]},
        status=200,
    )
    resp = Folder.list()
    assert resp is not None


@responses.activate
def test_get_folder(set_api_key):
    responses.add(
        responses.GET,
        f"{Env.current.web_api_endpoint}/tidy3d/project?projectName=default",
        json={"data": {"projectId": "1234", "projectName": "default"}},
        status=200,
    )
    resp = Folder.get("default")
    assert resp is not None


@responses.activate
def test_create_and_remove_folder(set_api_key):
    folder_name = "test folder2"
    responses.add(
        responses.GET,
        f"{Env.current.web_api_endpoint}/tidy3d/project?projectName={folder_name}",
        json={"data": {"projectId": "1234", "projectName": "default"}},
        status=200,
    )
    responses.add(
        responses.POST,
        f"{Env.current.web_api_endpoint}/tidy3d/projects",
        match=[matchers.json_params_matcher({"projectName": folder_name})],
        json={"data": {"projectId": "1234", "projectName": folder_name}},
        status=200,
    )
    responses.add(
        responses.DELETE,
        f"{Env.current.web_api_endpoint}/tidy3d/projects/1234",
        status=200,
    )

    resp = Folder.create(folder_name)

    assert resp is not None
    resp.delete()
