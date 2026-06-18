"""Tests for local license cache helpers."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tidy3d.web.license import (
    LICENSE_CACHE_GENERATION_FILE,
    refresh_license_cache,
    refresh_license_state,
)


def test_refresh_license_cache_removes_auth_files_and_updates_generation(tmp_path):
    """Test that license refresh clears auth files and updates the generation marker."""

    legacy_auth = tmp_path / "td.auth"
    hashed_auth = tmp_path / "td-abc123.auth"
    unrelated = tmp_path / "td-not-auth.txt"
    marker = tmp_path / LICENSE_CACHE_GENERATION_FILE

    legacy_auth.write_text("legacy", encoding="utf-8")
    hashed_auth.write_text("hashed", encoding="utf-8")
    unrelated.write_text("keep", encoding="utf-8")
    marker.write_text("old\n", encoding="utf-8")

    removed = refresh_license_cache(tmp_path)

    assert removed == 2
    assert not legacy_auth.exists()
    assert not hashed_auth.exists()
    assert unrelated.read_text(encoding="utf-8") == "keep"
    assert marker.read_text(encoding="utf-8").strip()
    assert marker.read_text(encoding="utf-8") != "old\n"


def test_refresh_license_cache_removes_auth_file_recreated_during_refresh(tmp_path, monkeypatch):
    """Test auth files recreated during refresh are swept after generation update."""

    auth_file = tmp_path / "td-stale.auth"
    concurrent_auth = tmp_path / "td-concurrent.auth"
    marker = tmp_path / LICENSE_CACHE_GENERATION_FILE
    auth_file.write_text("stale", encoding="utf-8")

    original_replace = Path.replace
    marker_swaps = 0
    wrote_concurrent_auth = False

    def write_auth_during_marker_swap(path: Path, target: Path):
        nonlocal marker_swaps, wrote_concurrent_auth

        result = original_replace(path, target)
        if Path(target) == marker:
            marker_swaps += 1
            if not wrote_concurrent_auth:
                concurrent_auth.write_text("concurrent", encoding="utf-8")
                wrote_concurrent_auth = True
        return result

    monkeypatch.setattr(Path, "replace", write_auth_during_marker_swap)

    removed = refresh_license_cache(tmp_path)

    assert removed == 2
    assert not auth_file.exists()
    assert not concurrent_auth.exists()
    assert marker.exists()
    assert marker_swaps == 2


def test_refresh_license_cache_keeps_auth_files_when_generation_write_fails(tmp_path, monkeypatch):
    """Test auth files are not removed if process cache invalidation cannot be written."""

    auth_file = tmp_path / "td-stale.auth"
    auth_file.write_text("stale", encoding="utf-8")

    original_write_text = Path.write_text

    def fail_marker_write(path: Path, *args, **kwargs):
        if path.name.startswith(f"{LICENSE_CACHE_GENERATION_FILE}.") and path.suffix == ".tmp":
            raise OSError("generation marker write failed")
        return original_write_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail_marker_write)

    with pytest.raises(OSError, match="generation marker write failed"):
        refresh_license_cache(tmp_path)

    assert auth_file.read_text(encoding="utf-8") == "stale"


def test_refresh_license_cache_preserves_fresh_auth_file_on_generation_write_failure(
    tmp_path, monkeypatch
):
    """Test rollback does not overwrite auth files recreated during a failed refresh."""

    auth_file = tmp_path / "td-stale.auth"
    auth_file.write_text("stale", encoding="utf-8")

    original_write_text = Path.write_text

    def fail_marker_write_after_recreated_auth(path: Path, *args, **kwargs):
        if path.name.startswith(f"{LICENSE_CACHE_GENERATION_FILE}.") and path.suffix == ".tmp":
            auth_file.write_text("fresh", encoding="utf-8")
            raise OSError("generation marker write failed")
        return original_write_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail_marker_write_after_recreated_auth)

    with pytest.raises(OSError, match="generation marker write failed"):
        refresh_license_cache(tmp_path)

    assert auth_file.read_text(encoding="utf-8") == "fresh"


def test_refresh_license_cache_post_marker_failure_is_partial_refresh(tmp_path, monkeypatch):
    """Test post-marker failures raise after clearing auth files."""

    auth_file = tmp_path / "td-stale.auth"
    concurrent_auth = tmp_path / "td-concurrent.auth"
    marker = tmp_path / LICENSE_CACHE_GENERATION_FILE
    auth_file.write_text("stale", encoding="utf-8")

    original_replace = Path.replace
    original_write_text = Path.write_text
    marker_swaps = 0
    marker_writes = 0

    def write_auth_during_first_marker_swap(path: Path, target: Path):
        nonlocal marker_swaps

        result = original_replace(path, target)
        if Path(target) == marker:
            marker_swaps += 1
            if marker_swaps == 1:
                concurrent_auth.write_text("concurrent", encoding="utf-8")
        return result

    def fail_second_marker_write(path: Path, *args, **kwargs):
        nonlocal marker_writes

        if path.name.startswith(f"{LICENSE_CACHE_GENERATION_FILE}.") and path.suffix == ".tmp":
            marker_writes += 1
            if marker_writes == 2:
                raise OSError("second generation marker write failed")
        return original_write_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "replace", write_auth_during_first_marker_swap)
    monkeypatch.setattr(Path, "write_text", fail_second_marker_write)

    with pytest.raises(OSError, match="second generation marker write failed"):
        refresh_license_cache(tmp_path)

    assert marker_swaps == 1
    assert marker_writes == 2
    assert marker.exists()
    assert not auth_file.exists()
    assert not concurrent_auth.exists()
    assert not list(tmp_path.glob(".td-license-refresh-*"))


def test_refresh_license_state_clears_failed_tidy3d_extras_import(tmp_path, monkeypatch):
    """Test license refresh lets a failed tidy3d-extras import retry later."""

    failed_module = SimpleNamespace(extension=None, __version__=None)
    loaded_module = SimpleNamespace(extension=object(), __version__="test")

    monkeypatch.setitem(sys.modules, "tidy3d_extras", failed_module)
    monkeypatch.setitem(sys.modules, "tidy3d_extras.utils", SimpleNamespace(_gencoeffs=None))
    refresh_license_state(tmp_path)
    assert "tidy3d_extras" not in sys.modules
    assert "tidy3d_extras.utils" not in sys.modules

    monkeypatch.setitem(sys.modules, "tidy3d_extras", loaded_module)
    monkeypatch.setitem(sys.modules, "tidy3d_extras.utils", SimpleNamespace(_gencoeffs=object()))
    refresh_license_state(tmp_path)
    assert sys.modules["tidy3d_extras"] is loaded_module
    assert "tidy3d_extras.utils" in sys.modules
