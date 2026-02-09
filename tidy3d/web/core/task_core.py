"""Compatibility shim for :mod:`tidy3d._common.web.core.task_core`."""

# ruff: noqa: F401 - ignore unused imports, imports ensure compatibility

# marked as migrated to _common
from __future__ import annotations

from tidy3d._common.web.core.task_core import (
    BatchTask,
    Folder,
    SimulationTask,
    TaskFactory,
    WebTask,
)
