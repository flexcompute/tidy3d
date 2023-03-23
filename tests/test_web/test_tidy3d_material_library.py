import pytest
import responses

from tidy3d.web.environment import Env
from tidy3d.web.material_libray import MaterialLibray

Env.dev.active()


@pytest.fixture
def set_api_key(monkeypatch):
    """Set the api key."""
    import tidy3d.web.http_management as http_module

    monkeypatch.setattr(http_module, "api_key", lambda: "apikey")


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
