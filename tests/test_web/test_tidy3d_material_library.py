import pytest
import responses

from tidy3d.web.core.environment import Env
from tidy3d.web.api.material_libray import MaterialLibray
import tidy3d as td

Env.dev.active()


@pytest.fixture
def set_api_key(monkeypatch):
    """Set the api key."""
    import tidy3d.web.core.http_util as httputil

    monkeypatch.setattr(httputil, "api_key", lambda: "apikey")
    monkeypatch.setattr(httputil, "get_version", lambda: td.version.__version__)


@responses.activate
def test_lib(set_api_key):
    responses.add(
        responses.GET,
        f"{Env.current.web_api_endpoint}/tidy3d/libraries",
        json={"data": [{"id": "3eb06d16-208b-487b-864b-e9b1d3e010a7", "name": "medium1"}]},
        status=200,
    )
    libs = MaterialLibray.list()
    lib = libs[0]
    assert lib.name == "medium1"
