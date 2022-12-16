import responses
from responses import matchers

from tidy3d.web.simulation_task import Folder
from tidy3d.web.environment import Env

Env.dev.active()


@responses.activate
def test_list_folders():
    responses.add(
        responses.GET,
        f"{Env.current.web_api_endpoint}/tidy3d/projects",
        json={"data": [{"projectId": "1234", "projectName": "default"}]},
        status=200,
    )
    resp = Folder.list()
    assert resp is not None


@responses.activate
def test_get_folder():
    responses.add(
        responses.GET,
        f"{Env.current.web_api_endpoint}/tidy3d/project?projectName=default",
        json={"data": {"projectId": "1234", "projectName": "default"}},
        status=200,
    )
    resp = Folder.get("default")
    assert resp is not None


@responses.activate
def test_create_and_remove_folder():
    responses.add(
        responses.GET,
        f"{Env.current.web_api_endpoint}/tidy3d/project",
        match=[matchers.query_param_matcher({"projectName": "test folder2"})],
        status=404,
    )
    responses.add(
        responses.POST,
        f"{Env.current.web_api_endpoint}/tidy3d/projects",
        match=[matchers.json_params_matcher({"projectName": "test folder2"})],
        json={"data": {"projectId": "1234", "projectName": "test folder2"}},
        status=200,
    )
    responses.add(
        responses.DELETE,
        f"{Env.current.web_api_endpoint}/tidy3d/projects/1234",
        status=200,
    )
    resp = Folder.create("test folder2")

    assert resp is not None
    resp.delete()
