from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

ARTIFACT_ENV_VAR = "TIDY3D_NUMERICAL_ARTIFACT_DIR"
DEFAULT_RELATIVE_DIR = Path("tests/tmp/autograd_numerical")


def _sanitize_segment(value: str) -> str:
    sanitized = re.sub(r"[^\w.-]+", "_", value)
    sanitized = sanitized.strip("_")
    return sanitized or "case"


def _resolve_artifact_root() -> Path:
    env_value = os.environ.get(ARTIFACT_ENV_VAR)
    if env_value:
        root = Path(env_value).expanduser()
    else:
        repo_root = Path(__file__).resolve().parents[4]
        root = repo_root / DEFAULT_RELATIVE_DIR
    root.mkdir(parents=True, exist_ok=True)
    return root


@pytest.fixture(scope="session")
def numerical_artifact_root() -> Path:
    return _resolve_artifact_root()


@pytest.fixture
def numerical_case_dir(request, numerical_artifact_root: Path) -> Path:
    safe_nodeid = _sanitize_segment(request.node.nodeid.replace(os.sep, "_"))
    case_dir = numerical_artifact_root / safe_nodeid
    case_dir.mkdir(parents=True, exist_ok=True)
    return case_dir
