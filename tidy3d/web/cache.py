"""Compatibility shim for :mod:`tidy3d._common.web.cache`."""

# ruff: noqa: F401 - ignore unused imports, imports ensure compatibility

# marked as migrated to _common
from __future__ import annotations

from typing import TYPE_CHECKING

from tidy3d._common.web.cache import (
    _CACHE,
    CACHE_ARTIFACT_NAME,
    CACHE_METADATA_NAME,
    CACHE_STATS_NAME,
    TMP_BATCH_PREFIX,
    TMP_PREFIX,
    CacheEntry,
    CacheEntryMetadata,
    CacheStats,
    LocalCache,
    _canonicalize,
    _copy_and_hash,
    _Hasher,
    _now,
    _read_metadata,
    _timestamp_suffix,
    _write_metadata,
    build_cache_key,
    build_entry_metadata,
    clear,
    get_cache_entry_dir,
    register_get_workflow_type,
    resolve_local_cache,
)
from tidy3d._common.web.core.types import TaskType
from tidy3d.web.api.tidy3d_stub import Tidy3dStub

if TYPE_CHECKING:
    import os

    from tidy3d.components.mode.mode_solver import ModeSolver
    from tidy3d.components.types.workflow import WorkflowType


def get_workflow_type(simulation: WorkflowType) -> str:
    """Resolve workflow type name for cache logging."""
    return Tidy3dStub(simulation=simulation).get_type()


register_get_workflow_type(get_workflow_type)


def _store_mode_solver_in_cache(task_id: str, simulation: ModeSolver, path: os.PathLike) -> bool:
    """
    Stores the results of a :class:`.ModeSolver` run in the local cache, if available.

    Parameters
    ----------
    task_id : str
        Unique identifier of the mode solver task.
    simulation : :class:`.ModeSolver`
        Mode solver simulation object whose results should be cached.
    path : PathLike
        Path to the result file on disk.

    Returns
    -------
    bool
        ``True`` if the result was successfully stored in the local cache, ``False`` otherwise.

    Notes
    -----
    This helper is used internally to persist completed mode solver results
    for reuse across repeated runs with identical configurations.
    """
    simulation_cache = resolve_local_cache()
    if simulation_cache is not None:
        stored = simulation_cache.store_result(
            task_id=task_id,
            path=path,
            workflow_type=TaskType.MODE_SOLVER.name,
            simulation=simulation,
        )
        return stored
    return False
