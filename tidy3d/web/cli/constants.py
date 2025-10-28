"""Constants for the CLI."""

from __future__ import annotations

from tidy3d.config.loader import resolve_config_directory

_CONFIG_ROOT = resolve_config_directory()

TIDY3D_DIR = str(_CONFIG_ROOT)
CONFIG_FILE = str(_CONFIG_ROOT / "config.toml")
