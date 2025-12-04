from __future__ import annotations

import os
import platform
from pathlib import Path
from shutil import which
from typing import Union

import tidy3d as td


def check_installation(raise_error: bool = False) -> Union[str, None]:
    """Return the path to the KLayout executable if it is installed.
    The executable is located by checking the system PATH and common platform-specific installation locations.

    Parameters
    ----------
    raise_error : bool
        Whether to raise an error if KLayout is not found. If ``False``, a warning is shown.

    Returns
    -------
    Union[str, None]
        The path to the KLayout executable. If KLayout is not found, returns ``None``.
    """

    path = _resolve_klayout_executable()
    msg = "KLayout was not found. Please ensure KLayout is installed and added to your system PATH before running KLayout."
    if path is None:
        if raise_error:
            raise RuntimeError(msg)
        td.log.warning(msg)
    return path


def _resolve_klayout_executable() -> Union[str, None]:
    """Return the path to the first platform-relevant KLayout executable we can find."""

    system = platform.system().lower()

    if system == "windows":
        names = ("klayout_app.exe", "klayout.exe")
    else:  # macOS ("darwin") and Linux
        names = ("klayout",)

    for name in names:
        resolved = which(name)
        if resolved:
            return resolved

    for binary in _common_install_locations():
        if binary.exists():
            return str(binary)
    return None


def _common_install_locations() -> tuple[Path, ...]:
    """Return possible platform-dependent installation paths for KLayout."""

    home = Path.home()
    system = platform.system().lower()
    paths: list[Path] = []

    if system == "darwin":
        apps = [
            Path("/Applications") / "KLayout.app" / "Contents" / "MacOS" / "klayout",
            Path("/Applications") / "klayout.app" / "Contents" / "MacOS" / "klayout",
            home / "Applications" / "KLayout.app" / "Contents" / "MacOS" / "klayout",
        ]
        brew_bins = [
            Path("/opt/homebrew/bin/klayout"),
            Path("/usr/local/bin/klayout"),
        ]
        paths.extend(apps + brew_bins)
    elif system == "windows":
        program_files = [
            Path(os.environ.get("ProgramFiles", r"C:\\Program Files")),
            Path(os.environ.get("ProgramFiles(x86)", r"C:\\Program Files (x86)")),
        ]
        for root in program_files:
            paths.extend(
                [
                    root / "KLayout" / "klayout_app.exe",
                    root / "KLayout" / "klayout.exe",
                ]
            )
        local_programs = home / "AppData" / "Local" / "Programs" / "KLayout"
        paths.extend(
            [
                local_programs / "klayout_app.exe",
                local_programs / "klayout.exe",
            ]
        )
    else:  # Linux and other Unix variants
        paths.extend(
            [
                Path("/usr/bin/klayout"),
                Path("/usr/local/bin/klayout"),
                Path("/snap/bin/klayout"),
                Path("/opt/klayout/klayout"),
                home / ".local" / "bin" / "klayout",
            ]
        )

    seen = set()
    unique_paths = []
    for path in paths:
        if path not in seen:
            unique_paths.append(path)
            seen.add(path)

    return tuple(unique_paths)
