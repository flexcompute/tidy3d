from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from pydantic import SecretStr

from tidy3d.config import config, reload_config
from tidy3d.web.cli import migrate as migrate_module


@pytest.fixture
def temp_config_dir(monkeypatch, tmp_path) -> Path:
    """Provide an isolated configuration directory for migration tests."""

    original_base = os.environ.get("TIDY3D_BASE_DIR")
    monkeypatch.setenv("TIDY3D_BASE_DIR", str(tmp_path))
    reload_config(profile="default")
    config_dir = Path(tmp_path) / ".tidy3d"
    config_dir.mkdir(parents=True, exist_ok=True)
    yield config_dir
    if original_base is None:
        monkeypatch.delenv("TIDY3D_BASE_DIR", raising=False)
    else:
        monkeypatch.setenv("TIDY3D_BASE_DIR", original_base)
    reload_config(profile="default")


def _normalize_secret(value):
    if isinstance(value, SecretStr):
        return value.get_secret_value()
    return value


def _write_auth_file(config_dir: Path) -> Path:
    credential_path = config_dir / "auth.json"
    credential_path.write_text(
        json.dumps({"email": "user@example.com", "password": "hunter2"}),
        encoding="utf-8",
    )
    return credential_path


def test_persist_api_key_creates_backup(temp_config_dir):
    credential_path = _write_auth_file(temp_config_dir)
    result = migrate_module._persist_api_key("new-api-key", credential_path)

    assert result is True
    assert not credential_path.exists()
    assert (temp_config_dir / "auth.json.bak").is_file()
    assert _normalize_secret(config.web.apikey) == "new-api-key"


def test_persist_api_key_rolls_back_when_save_fails(monkeypatch, temp_config_dir):
    credential_path = _write_auth_file(temp_config_dir)
    original_apikey = config.web.apikey

    def failing_save(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(migrate_module.config, "save", failing_save)

    result = migrate_module._persist_api_key("new-api-key", credential_path)

    assert result is False
    assert credential_path.is_file()
    assert not (temp_config_dir / "auth.json.bak").exists()
    assert _normalize_secret(config.web.apikey) == _normalize_secret(original_apikey)


def test_persist_api_key_rolls_back_when_backup_fails(monkeypatch, temp_config_dir):
    credential_path = _write_auth_file(temp_config_dir)
    original_apikey = config.web.apikey

    def failing_replace(*args, **kwargs):
        raise PermissionError("read-only filesystem")

    monkeypatch.setattr(migrate_module.os, "replace", failing_replace)

    result = migrate_module._persist_api_key("new-api-key", credential_path)

    assert result is False
    assert credential_path.is_file()
    assert not (temp_config_dir / "auth.json.bak").exists()
    assert _normalize_secret(config.web.apikey) == _normalize_secret(original_apikey)
