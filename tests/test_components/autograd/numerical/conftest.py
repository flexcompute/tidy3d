from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

import pytest

ARTIFACT_ENV_VAR = "TIDY3D_NUMERICAL_ARTIFACT_DIR"
DEFAULT_RELATIVE_DIR = Path("tests/tmp/autograd_numerical")

# Optional extra cap for the per-test directory name length (in bytes, after fs encoding).
ARTIFACT_NAME_MAX_ENV_VAR = "TIDY3D_NUMERICAL_ARTIFACT_NAME_MAX"


def _sanitize_segment(value: str) -> str:
    sanitized = re.sub(r"[^\w.-]+", "_", value)
    sanitized = sanitized.strip("_")
    return sanitized or "case"


def _pathconf_limit(path: Path, key: str, fallback: int) -> int:
    """Best-effort os.pathconf lookup with a fallback."""
    try:
        return int(os.pathconf(str(path), key))
    except (AttributeError, ValueError, OSError):
        return fallback


def _artifact_name_max_override() -> int | None:
    """Optional user cap for artifact directory names (bytes)."""
    raw = os.environ.get(ARTIFACT_NAME_MAX_ENV_VAR)
    if not raw:
        return None
    try:
        return max(1, int(raw))
    except ValueError as e:
        raise ValueError(f"{ARTIFACT_NAME_MAX_ENV_VAR} must be an integer (got {raw!r}).") from e


def _resolve_artifact_root() -> Path:
    env_value = os.environ.get(ARTIFACT_ENV_VAR)
    if env_value:
        root = Path(env_value).expanduser()
    else:
        repo_root = Path(__file__).resolve().parents[4]
        root = repo_root / DEFAULT_RELATIVE_DIR
    root.mkdir(parents=True, exist_ok=True)
    return root


def _case_dir_name(request, artifact_root: Path) -> str:
    """Return a filesystem-friendly per-test artifact directory name.

    Uses ``request.node.name``. If truncation is needed to satisfy filesystem
    path/name limits (and optional ``TIDY3D_NUMERICAL_ARTIFACT_NAME_MAX``), append a
    short SHA1 digest of the full nodeid. Otherwise, no digest is added, so
    uniqueness across files is not guaranteed.
    """
    raw_nodeid = request.node.nodeid
    base_name = _sanitize_segment(request.node.name) or "case"

    # Use filesystem-encoded byte lengths to be conservative under multibyte encodings.
    def _fslen(s: str) -> int:
        return len(os.fsencode(s))

    root_abs = artifact_root.resolve()

    # Common Linux defaults: NAME_MAX ~255 bytes, PATH_MAX ~4096 bytes (may vary).
    name_max = _pathconf_limit(root_abs, "PC_NAME_MAX", 255)
    path_max = _pathconf_limit(root_abs, "PC_PATH_MAX", 4096)

    # Leave room for: <root>/<name> plus NUL (pathconf is in bytes).
    available = path_max - _fslen(str(root_abs)) - _fslen(os.sep) - 1

    max_len = min(name_max, max(1, available))
    override = _artifact_name_max_override()
    if override is not None:
        max_len = min(max_len, override)

    if _fslen(base_name) <= max_len:
        return base_name

    digest = hashlib.sha1(raw_nodeid.encode("utf-8")).hexdigest()[:8]
    suffix = f"-{digest}"
    max_base = max_len - _fslen(suffix)
    if max_base < 1:
        max_base = 1

    # Truncate by bytes (not chars): drop codepoints until it fits.
    while base_name and _fslen(base_name) > max_base:
        base_name = base_name[:-1]
    if not base_name:
        base_name = "c"

    return f"{base_name}{suffix}"


@pytest.fixture(scope="session")
def numerical_artifact_root() -> Path:
    return _resolve_artifact_root()


@pytest.fixture
def numerical_case_dir(request, numerical_artifact_root: Path) -> Path:
    case_dir = numerical_artifact_root / _case_dir_name(request, numerical_artifact_root)
    case_dir.mkdir(parents=True, exist_ok=True)
    return case_dir
