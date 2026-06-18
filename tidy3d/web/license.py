"""Helpers for clearing local license authentication state."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

if TYPE_CHECKING:
    from collections.abc import Iterator

LICENSE_CACHE_GENERATION_FILE = "td-license-cache-generation"


def _iter_license_auth_files(config_dir: Path) -> Iterator[Path]:
    """Yield persisted license authentication files in a config directory."""

    seen: set[Path] = set()
    candidates = [config_dir / "td.auth", *config_dir.glob("td-*.auth")]
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        if path.is_file():
            yield path


def refresh_license_cache(config_dir: str | Path) -> int:
    """Clear all persisted license auth state and bump the process cache generation."""

    config_path = Path(config_dir)
    config_path.mkdir(mode=0o700, parents=True, exist_ok=True)

    staged_dir = config_path / f".td-license-refresh-{uuid4().hex}"
    staged_files: list[tuple[Path, Path]] = []
    auth_files = list(_iter_license_auth_files(config_path))
    if auth_files:
        staged_dir.mkdir(mode=0o700)

    marker_tmp: Path | None = None
    marker_replaced = False
    removed_after_marker = 0
    try:
        for path in auth_files:
            try:
                staged_path = staged_dir / path.name
                path.replace(staged_path)
            except FileNotFoundError:
                continue
            staged_files.append((path, staged_path))

        marker = config_path / LICENSE_CACHE_GENERATION_FILE
        marker_tmp = marker.with_name(f"{marker.name}.{uuid4().hex}.tmp")
        marker_tmp.write_text(f"{uuid4().hex}\n", encoding="utf-8")
        marker_tmp.replace(marker)
        marker_replaced = True

        for path in _iter_license_auth_files(config_path):
            try:
                path.unlink()
            except FileNotFoundError:
                continue
            removed_after_marker += 1

        if removed_after_marker:
            marker_tmp = marker.with_name(f"{marker.name}.{uuid4().hex}.tmp")
            marker_tmp.write_text(f"{uuid4().hex}\n", encoding="utf-8")
            marker_tmp.replace(marker)
    except Exception:
        if not marker_replaced:
            for path, staged_path in reversed(staged_files):
                if staged_path.exists() and not path.exists():
                    staged_path.replace(path)
        if marker_tmp is not None:
            try:
                marker_tmp.unlink()
            except FileNotFoundError:
                pass
        raise
    finally:
        shutil.rmtree(staged_dir, ignore_errors=True)

    return len(staged_files) + removed_after_marker


def reset_failed_tidy3d_extras_import() -> None:
    """Clear a cached failed tidy3d-extras import so the next import retries."""

    module = sys.modules.get("tidy3d_extras")
    if module is None or getattr(module, "extension", None) is not None:
        return

    for name in list(sys.modules):
        if name == "tidy3d_extras" or name.startswith("tidy3d_extras."):
            sys.modules.pop(name, None)


def refresh_license_state(config_dir: str | Path) -> int:
    """Clear all persisted license auth state and failed local-license import state."""

    removed = refresh_license_cache(config_dir)
    reset_failed_tidy3d_extras_import()
    return removed


def refresh_licenses() -> None:
    """Clear cached local license entitlements from the active config directory.

    This does not contact the license server immediately. The next licensed
    local feature check fetches current entitlements and caches them again.
    The refresh is cache-wide: it clears all persisted license auth files in
    this config directory, including files for other API key or endpoint hashes.
    """

    from tidy3d.config import get_manager

    refresh_license_state(get_manager().config_dir)
