from __future__ import annotations

import os
import tempfile
from typing import TYPE_CHECKING

from flexcompute.core._migration.em.derivatives import read_results

import tidy3d as td
from tidy3d.web.api.tidy3d_stub import Tidy3dStub
from tidy3d.web.cache import resolve_local_cache
from tidy3d.web.core.s3utils import download_file

from .constants import SIM_DERIVATIVE_RESULTS_FILE, SIM_FWD_FLUX_DATA_FILE

if TYPE_CHECKING:
    from flexcompute.core._migration.em.derivatives import Results

    from tidy3d.components.types.workflow import WorkflowType
    from tidy3d.web.cache import LocalCache


RESULTS_CACHE_ARTIFACT_TYPE = "autograd_derivative_results"
FLUX_FORWARD_CACHE_ARTIFACT_TYPE = "autograd_flux_forward"
_CACHE_LOAD_ERRORS = (
    OSError,
    ValueError,
    KeyError,
)


def _artifact_cache_context(
    simulation: WorkflowType | None,
) -> tuple[LocalCache, str, str] | None:
    """Return cache, workflow type, and simulation hash for autograd artifact lookups."""
    simulation_cache = resolve_local_cache()
    if simulation_cache is None or simulation is None:
        return None
    return (
        simulation_cache,
        Tidy3dStub(simulation=simulation).get_type(),
        simulation._hash_self(),
    )


def get_cached_derivative_results(
    simulation: WorkflowType | None, verbose: bool = True
) -> Results | None:
    """Load an adjoint simulation's cached program results, outside the normal result cache.

    The adjoint simulation carries the program digest in its ``attrs``, so its hash keys the
    results by the program as well.
    """
    cache_context = _artifact_cache_context(simulation)
    if cache_context is None:
        return None
    simulation_cache, workflow_type, simulation_hash = cache_context

    entry = simulation_cache.try_fetch_with_hash(
        simulation_hash=simulation_hash,
        workflow_type=workflow_type,
        verbose=verbose,
        artifact_type=RESULTS_CACHE_ARTIFACT_TYPE,
    )
    if entry is None:
        return None

    try:
        return read_results(entry.artifact_path)
    except Exception as e:
        td.log.error(f"Could not load derivative results cache entry: {e}")
        simulation_cache.invalidate(entry.key)
        return None


def _store_results_cache_entry(
    task_id_adj: str, *, artifact_path: str, simulation: WorkflowType | None
) -> None:
    cache_context = _artifact_cache_context(simulation)
    if cache_context is None:
        return
    simulation_cache, workflow_type, simulation_hash = cache_context
    try:
        simulation_cache.store_result_with_hash(
            task_id=task_id_adj,
            path=artifact_path,
            workflow_type=workflow_type,
            simulation_hash=simulation_hash,
            artifact_type=RESULTS_CACHE_ARTIFACT_TYPE,
        )
    except Exception as e:
        td.log.error(f"Could not store derivative results cache entry: {e}")


def get_cached_flux_forward_data(
    task_id_fwd: str, simulation: WorkflowType | None, verbose: bool = True
) -> td.SimulationData | None:
    """Load cached hidden FluxMonitor forward data without using the normal result cache."""
    cache_context = _artifact_cache_context(simulation)
    if cache_context is None:
        return None
    simulation_cache, workflow_type, simulation_hash = cache_context

    entry = simulation_cache.try_fetch_with_hash(
        simulation_hash=simulation_hash,
        workflow_type=workflow_type,
        verbose=verbose,
        artifact_type=f"{FLUX_FORWARD_CACHE_ARTIFACT_TYPE}:{task_id_fwd}",
    )
    if entry is None:
        return None

    if not entry.artifact_path.is_file():
        td.log.error("Could not load FluxMonitor helper cache entry: artifact file is missing.")
        simulation_cache.invalidate(entry.key)
        return None

    try:
        return td.SimulationData.from_file(entry.artifact_path)
    except _CACHE_LOAD_ERRORS as e:
        td.log.error(f"Could not load FluxMonitor helper cache entry: {e}")
        simulation_cache.invalidate(entry.key)
        return None


def _store_flux_forward_cache_entry(
    task_id_fwd: str, *, artifact_path: str, simulation: WorkflowType | None
) -> None:
    cache_context = _artifact_cache_context(simulation)
    if cache_context is None:
        return
    simulation_cache, workflow_type, simulation_hash = cache_context
    stored = simulation_cache.store_result_with_hash(
        task_id=task_id_fwd,
        path=artifact_path,
        workflow_type=workflow_type,
        simulation_hash=simulation_hash,
        artifact_type=f"{FLUX_FORWARD_CACHE_ARTIFACT_TYPE}:{task_id_fwd}",
    )
    if not stored:
        td.log.error("Could not store FluxMonitor helper cache entry.")


def get_autograd_flux_forward_data(
    task_id_fwd: str,
    verbose: bool,
    *,
    cache_simulation: WorkflowType | None = None,
) -> td.SimulationData:
    """Download hidden FluxMonitor helper data for adjoint source construction."""
    cached_data = get_cached_flux_forward_data(task_id_fwd, cache_simulation, verbose=verbose)
    if cached_data is not None:
        return cached_data

    handle, fname = tempfile.mkstemp(suffix=".hdf5")
    os.close(handle)
    try:
        download_file(task_id_fwd, SIM_FWD_FLUX_DATA_FILE, to_file=fname, verbose=verbose)
        sim_data = td.SimulationData.from_file(fname)
        _store_flux_forward_cache_entry(
            task_id_fwd,
            artifact_path=fname,
            simulation=cache_simulation,
        )
        return sim_data
    except Exception as e:
        raise td.exceptions.AdjointError(
            f"Could not load hidden FluxMonitor forward data artifact '{SIM_FWD_FLUX_DATA_FILE}' "
            f"for forward task '{task_id_fwd}'. This artifact is required for FluxMonitor adjoint "
            "source construction. Rerun the autograd forward task. "
            f"Original error: {e}"
        ) from e
    finally:
        os.unlink(fname)


def get_derivative_results(
    task_id_adj: str,
    verbose: bool,
    *,
    cache_simulation: WorkflowType | None = None,
) -> Results:
    """Download the program results of a completed adjoint job."""
    handle, fname = tempfile.mkstemp(suffix=".hdf5")
    os.close(handle)
    try:
        download_file(task_id_adj, SIM_DERIVATIVE_RESULTS_FILE, to_file=fname, verbose=verbose)
        results = read_results(fname)
        if cache_simulation is not None:
            _store_results_cache_entry(
                task_id_adj, artifact_path=fname, simulation=cache_simulation
            )
    except Exception as e:
        td.log.error(f"Error occurred while getting derivative results: {e}")
        raise e
    finally:
        os.unlink(fname)
    return results
