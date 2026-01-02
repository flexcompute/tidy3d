# test autograd and compares to numerically computed finite difference gradients
from __future__ import annotations

import cProfile
import pstats
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pytest

from tidy3d.web.api.autograd.backward import postprocess_adj

from .postprocess_adj_utils import (
    DATASET_ROOT,
    PostprocessAdjInputs,
    dataset_exists,
    generate_postprocess_adj_inputs,
    load_postprocess_adj_inputs,
    persist_postprocess_adj_inputs,
)

MARK_GENERATE_NAME = "generate_profile"


@dataclass
class ProfileArtifacts:
    cpu_profile_path: Path
    stats_text_path: Path
    peak_memory_bytes: int


def _run_postprocess_adj(inputs: PostprocessAdjInputs) -> dict:
    return postprocess_adj(
        sim_data_adj=inputs.sim_data_adj,
        sim_data_orig=inputs.sim_data_orig,
        sim_data_fwd=inputs.sim_data_fwd,
        sim_fields_keys=inputs.sim_fields_keys,
    )


def _load_inputs_or_skip() -> PostprocessAdjInputs:
    if not dataset_exists(DATASET_ROOT):
        pytest.skip(
            f"Persisted postprocess_adj dataset missing at {DATASET_ROOT}. "
            f"Run with @{MARK_GENERATE_NAME} first."
        )
    return load_postprocess_adj_inputs(DATASET_ROOT)


def _profile_callable(func: Callable[[], dict], output_dir: Path) -> ProfileArtifacts:
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import tracemalloc
    except ImportError:
        tracemalloc = None

    if tracemalloc:
        tracemalloc.start()

    profile = cProfile.Profile()
    profile.enable()
    result = func()
    profile.disable()

    cpu_profile_path = output_dir / "postprocess_adj.prof"
    profile.dump_stats(cpu_profile_path)

    peak_memory_bytes = -1
    if tracemalloc:
        _, peak_memory_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

    stats_text_path = output_dir / "postprocess_adj_profile.txt"
    with stats_text_path.open("w", encoding="utf8") as stats_file:
        stats = pstats.Stats(profile, stream=stats_file)
        stats.sort_stats("cumulative")
        stats.print_stats(40)
        stats_file.write(f"\nResult keys: {sorted(result.keys())}\n")
        stats_file.write(f"\nPeak memory GB: {peak_memory_bytes / (1024.0 * 1024.0 * 1024.0)}\n")

    return ProfileArtifacts(
        cpu_profile_path=cpu_profile_path,
        stats_text_path=stats_text_path,
        peak_memory_bytes=peak_memory_bytes,
    )


@pytest.mark.generate_profile
def test_generate_postprocess_adj_dataset(tmp_path: Path, redirect_stdout_to_stderr):
    """Generate test data for running test_postprocess_adj_profile."""
    print(f"Profile data in {tmp_path}")

    inputs = generate_postprocess_adj_inputs(tmp_path)

    persist_postprocess_adj_inputs(inputs, DATASET_ROOT)
    assert dataset_exists(DATASET_ROOT)


@pytest.mark.performance_profile
def test_postprocess_adj_profile(tmp_path: Path, redirect_stdout_to_stderr):
    """Run profiling for postprocess_adj after having generated test data with test_generate_postprocess_adj_dataset"""
    print(f"Profile data in {tmp_path}")

    inputs = _load_inputs_or_skip()
    artifacts = _profile_callable(lambda: _run_postprocess_adj(inputs), tmp_path / "profile")
    assert artifacts.cpu_profile_path.exists()
    assert artifacts.stats_text_path.exists()
