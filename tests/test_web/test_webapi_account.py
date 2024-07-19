# Tests webapi and things that depend on it

import pytest
import responses
import tidy3d as td
from tidy3d.web.api.webapi import (
    account,
)
from tidy3d.web.core.environment import Env

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
def mock_get_account(monkeypatch, set_api_key):
    """Mocks webapi.get_info."""

    responses.add(
        responses.GET,
        f"{Env.current.web_api_endpoint}/tidy3d/py/account",
        json={
            "data": {
                "credit": 10.0,
                "creditExpiration": "2024-10-01T00:00:00.000Z",
                "allowanceCycleType": "Daily",
                "allowanceCurrentCycleAmount": 10.0,
                "allowanceCurrentCycleEndDate": "2024-09-01T00:00:00.000Z",
                "dailyFreeSimulationCounts": 2,
            }
        },
        status=200,
    )


@responses.activate
def test_account(mock_get_account):
    info = account()
    assert info is not None
    assert info.credit == 10.0
