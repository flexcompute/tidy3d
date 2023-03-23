import pytest
import re

import responses
from tidy3d.plugins.dispersion import DispersionFitter

from tidy3d.web.environment import Env
from tidy3d.web.material_fitter import FitterOptions, MaterialFitterTask

Env.dev.active()


@pytest.fixture
def set_api_key(monkeypatch):
    """Set the api key."""
    import tidy3d.web.http_management as http_module

    monkeypatch.setattr(http_module, "api_key", lambda: "apikey")


@responses.activate
def test_material_fitter(monkeypatch, set_api_key):
    fitter = DispersionFitter.from_file("tests/data/nk_data.csv", skiprows=1, delimiter=",")

    monkeypatch.setattr("tidy3d.web.material_fitter.uuid4", lambda: "fitter_id")

    responses.add(
        responses.GET,
        f"{Env.current.web_api_endpoint}/tidy3d/fitter/1234",
        json={"data": "https://example.com"},
        status=200,
    )
    responses.add(
        responses.PUT,
        "https://example.com",
        json={"data": "url"},
        status=200,
    )
    monkeypatch.setattr(
        "tidy3d.web.material_fitter.MaterialFitterTask.submit",
        lambda fitter, options: MaterialFitterTask(
            id="1234", status="ok", dispersion_fitter=fitter, resourcePath="", fileName=""
        ),
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
        f"{Env.current.web_api_endpoint}/tidy3d/fitter/1234",
        json={"data": {"status": "running"}},
        status=200,
    )
    monkeypatch.setattr(
        "tidy3d.web.material_fitter.MaterialFitterTask.sync_status", lambda self: None
    )
    task.sync_status()
    task.status == "running"

    responses.add(
        responses.POST,
        f"{Env.current.web_api_endpoint}/tidy3d/fitter/save",
        json={
            "data": {
                "id": "1234",
                "fitterName": "test",
            }
        },
        status=200,
    )
    task.status = "COMPLETED"
    # monkeypatch.setattr('tidy3d.web.material_fitter.MaterialFitterTask.save_to_library', lambda self: None)
    assert task.save_to_library("test")
