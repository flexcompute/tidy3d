"""Tests for tidy3d.plugins.klayout.util."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tidy3d.plugins.klayout import util

KLAYOUT_PLUGIN_PATH = "tidy3d.plugins.klayout"


def test_check_klayout_not_installed(monkeypatch):
    """check_installation raises when KLayout is not on PATH."""
    monkeypatch.setattr(f"{KLAYOUT_PLUGIN_PATH}.util.which", lambda _cmd: None)
    monkeypatch.setattr(f"{KLAYOUT_PLUGIN_PATH}.util._common_install_locations", lambda: ())
    with pytest.raises(RuntimeError):
        util.check_installation(raise_error=True)


def test_check_klayout_installed(monkeypatch):
    """check_installation returns a path and does not raise when present."""
    fake_path = "/usr/local/bin/klayout"
    monkeypatch.setattr(f"{KLAYOUT_PLUGIN_PATH}.util.which", lambda _cmd: fake_path)
    monkeypatch.setattr(f"{KLAYOUT_PLUGIN_PATH}.util._common_install_locations", lambda: ())
    assert util.check_installation(raise_error=True) == fake_path


def test_check_installation_finds_known_location(monkeypatch, tmp_path):
    """Return a discovered installation path even when PATH is empty."""

    fake_binary = tmp_path / "KLayout.app" / "Contents" / "MacOS" / "klayout"
    fake_binary.parent.mkdir(parents=True)
    fake_binary.write_text("#!/bin/sh\nexit 0\n")
    fake_binary.chmod(0o755)

    monkeypatch.setenv("PATH", "")
    monkeypatch.setattr(
        f"{KLAYOUT_PLUGIN_PATH}.util._common_install_locations",
        lambda: (fake_binary,),
    )
    monkeypatch.setattr(f"{KLAYOUT_PLUGIN_PATH}.util.which", lambda _cmd: None)

    resolved = util.check_installation(raise_error=True)
    assert resolved == str(fake_binary)
    assert os.environ["PATH"] == ""


def test_common_install_locations_include_homebrew_cask_app_suite(monkeypatch):
    """Darwin candidates include Homebrew cask app-suite install locations."""

    monkeypatch.setattr(f"{KLAYOUT_PLUGIN_PATH}.util.platform.system", lambda: "Darwin")
    paths = util._common_install_locations()
    assert Path("/Applications/KLayout/klayout.app/Contents/MacOS/klayout") in paths


def test_brew_cask_klayout_binaries_detect_versioned_paths(monkeypatch, tmp_path):
    """Discover binaries from versioned Homebrew cask directories."""

    cask_root = tmp_path / "Caskroom" / "klayout"
    binary = cask_root / "0.30.6" / "KLayout" / "klayout.app" / "Contents" / "MacOS" / "klayout"
    binary.parent.mkdir(parents=True)
    binary.write_text("#!/bin/sh\nexit 0\n")
    binary.chmod(0o755)

    monkeypatch.setattr(f"{KLAYOUT_PLUGIN_PATH}.util._MACOS_BREW_CASK_KLAYOUT_ROOTS", (cask_root,))
    candidates = util._brew_cask_klayout_binaries()
    assert binary in candidates
