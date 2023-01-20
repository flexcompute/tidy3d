import re

import responses
from tidy3d.plugins import DispersionFitter

from tidy3d.web.environment import Env
from tidy3d.web.material_fitter import FitterOptions, MaterialFitterTask

Env.dev.active()


@responses.activate
def test_material_fitter(monkeypatch):
    fitter = DispersionFitter.from_file("tests/data/nk_data.csv", skiprows=1, delimiter=",")

    monkeypatch.setattr("tidy3d.web.material_fitter.uuid4", lambda: "fitter_id")

    responses.add(
        responses.GET,
        re.compile(f"{Env.current.web_api_endpoint}/tidy3d/fitter/.*"),
        json={"data": "https://example.com"},
        status=200,
    )
    responses.add(
        responses.PUT,
        "https://example.com",
        json={"data": "url"},
        status=200,
    )

    responses.add(
        responses.POST,
        f"{Env.current.web_api_endpoint}/tidy3d/fitter/fit",
        json={
            "data": {
                "id": "1234",
                "status": "RUNNING",
                "fileName": "filename",
                "resourcePath": "path",
            }
        },
        status=200,
    )
    task = MaterialFitterTask.submit(fitter, FitterOptions())

    responses.add(
        responses.GET,
        re.compile(f"{Env.current.web_api_endpoint}/tidy3d/fitter/1234"),
        json={"data": {"status": "running"}},
        status=200,
    )
    task.sync_status()
    task.status == "running"

    responses.add(
        responses.POST,
        f"{Env.current.web_api_endpoint}/tidy3d/fitter/save",
        json={"data": True},
        status=200,
    )
    task.status = "COMPLETED"
    assert task.save_to_library("test")
