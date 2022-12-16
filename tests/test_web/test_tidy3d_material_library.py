import responses

from tidy3d.web.environment import Env
from tidy3d.web.material_libray import MaterialLibray

Env.dev.active()


@responses.activate
def test_lib():
    responses.add(
        responses.GET,
        f"{Env.current.web_api_endpoint}/tidy3d/libraries",
        json={"data": [{"id": "3eb06d16-208b-487b-864b-e9b1d3e010a7", "name": "medium1"}]},
        status=200,
    )
    libs = MaterialLibray.list()
    lib = libs[0]
    assert lib.name == "medium1"
