""" unit test authentication of python client """
import json

import pytest
import requests
import importlib

import tidy3d.web.auth as tidy3d_auth
import tidy3d.web.config as config
from tidy3d import log
from .mock_web import MockResponse
from ..utils import get_test_root_dir


@pytest.fixture
def mock_credential_path(monkeypatch):
    monkeypatch.setattr(
        tidy3d_auth, "CREDENTIAL_PATH", str(get_test_root_dir().joinpath("/tmp/auth.json"))
    )


def test__save_credential_to_stored_file(mock_credential_path):
    tidy3d_auth._save_credential_to_stored_file("user", "pwd")
    with open(tidy3d_auth.CREDENTIAL_PATH, "r", encoding="utf-8") as fp:
        auth_json = json.load(fp)
        log.info(auth_json)

    assert sorted(auth_json.items()) == sorted({"email": "user", "password": "pwd"}.items())


def test__get_credential_from_stored_file(mock_credential_path):
    with open(tidy3d_auth.CREDENTIAL_PATH, "w", encoding="utf-8") as fp:
        json.dump({"email": "user", "password": "pwd"}, fp)

    email, password = tidy3d_auth._get_credential_from_stored_file()
    assert email == "user" and password == "pwd"


def test__get_credential_from_env(monkeypatch):
    monkeypatch.setenv("TIDY3D_USER", "user")
    monkeypatch.setenv("TIDY3D_PASS", "pwd")
    email, password = tidy3d_auth._get_credential_from_env()
    assert email == "user" and password == tidy3d_auth.encode_password("pwd")


def test__get_credential_from_env_with_hash_pass(monkeypatch):
    monkeypatch.setenv("TIDY3D_USER", "user")
    monkeypatch.setenv("TIDY3D_PASS_HASH", "pwd_hash")
    email, password = tidy3d_auth._get_credential_from_env()
    assert email == "user" and password == "pwd_hash"


def test__get_credential_from_console_first_success(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "user")
    monkeypatch.setattr("getpass.getpass", lambda _: "pwd")
    email, password = tidy3d_auth._get_credential_from_console(None)
    assert email == "user" and password == tidy3d_auth.encode_password("pwd")


def test__get_credential_from_console_second_success(monkeypatch):
    inputs = iter(["", "user"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    monkeypatch.setattr("getpass.getpass", lambda _: "pwd")
    email, password = tidy3d_auth._get_credential_from_console(None)
    assert email == "user" and password == tidy3d_auth.encode_password("pwd")


def test_get_credentials_with_env(monkeypatch):
    monkeypatch.setenv("TIDY3D_USER", "user")
    monkeypatch.setenv("TIDY3D_PASS", "pwd")

    monkeypatch.setattr(
        requests,
        "get",
        lambda *args, **kwargs: MockResponse(
            200, {"data": {"user": {"name": "mock_user"}, "auth": "mock_auth"}}
        ),
    )
    tidy3d_auth.get_credentials()


def test_get_config_with_env(monkeypatch):

    for TIDY3D_ENV in ("prod", "dev", "uat"):
        monkeypatch.setenv("TIDY3D_ENV", TIDY3D_ENV)
        importlib.reload(config)
        assert config.DEFAULT_CONFIG == config.WEB_CONFIGS[TIDY3D_ENV]


def test_get_credentials_with_stored_file(monkeypatch, mock_credential_path):
    with open(tidy3d_auth.CREDENTIAL_PATH, "w", encoding="utf-8") as fp:
        json.dump({"email": "user", "password": "pwd"}, fp)

    monkeypatch.setattr(
        requests,
        "get",
        lambda *args, **kwargs: MockResponse(
            200, {"data": {"user": {"name": "mock_user"}, "auth": "mock_auth"}}
        ),
    )
    tidy3d_auth.get_credentials()


def test_get_credentials_with_console(monkeypatch, mock_credential_path):
    with open(tidy3d_auth.CREDENTIAL_PATH, "w", encoding="utf-8") as fp:
        json.dump({"email": "user", "password": "pwd"}, fp)
    auth_resps = iter(
        [
            MockResponse(401, {"data": {"user": {"name": "mock_user"}, "auth": "mock_auth"}}),
            MockResponse(200, {"data": {"user": {"name": "mock_user"}, "auth": "mock_auth"}}),
        ]
    )
    monkeypatch.setattr("builtins.input", lambda _: "user")
    monkeypatch.setattr("getpass.getpass", lambda _: "pwd")

    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: next(auth_resps))
    tidy3d_auth.get_credentials()
