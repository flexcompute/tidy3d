"""Shared setup helpers for script-style integration tests."""

from __future__ import annotations

import os

import tidy3d as td
import tidy3d.web as web


def configure_integration_environment() -> None:
    """Disable caches and configure web access for live integration runs."""
    td.config.web.enable_caching = False
    td.config.local_cache.enabled = False

    assert td.config.web.enable_caching is False
    assert td.config.local_cache.enabled is False

    web.configure(os.environ["TIDY3D_VGPU_API_KEY"])
