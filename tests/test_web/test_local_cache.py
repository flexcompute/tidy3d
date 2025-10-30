from __future__ import annotations

import os
from pathlib import Path

import pytest

import tidy3d as td
from tests.test_components.autograd.test_autograd import ALL_KEY, get_functions, params0
from tests.test_web.test_webapi_mode import make_mode_sim
from tidy3d import config
from tidy3d.config import get_manager
from tidy3d.web import Job, common, run_async
from tidy3d.web.api import webapi as web
from tidy3d.web.api.container import Batch, WebContainer
from tidy3d.web.api.webapi import load_simulation_if_cached
from tidy3d.web.cache import CACHE_ARTIFACT_NAME, clear, resolve_local_cache

common.CONNECTION_RETRY_TIME = 0.1

MOCK_TASK_ID = "task-xyz"
# --- Fake pipeline global maps / queue ---
TASK_TO_SIM: dict[str, td.Simulation] = {}  # task_id -> Simulation
PATH_TO_SIM: dict[str, td.Simulation] = {}  # artifact path -> Simulation


def _reset_fake_maps():
    TASK_TO_SIM.clear()
    PATH_TO_SIM.clear()


class _FakeStubData:
    def __init__(self, simulation: td.Simulation):
        self.simulation = simulation


@pytest.fixture
def basic_simulation():
    pulse = td.GaussianPulse(freq0=200e12, fwidth=20e12)
    pt_dipole = td.PointDipole(source_time=pulse, polarization="Ex")
    return td.Simulation(
        size=(1, 1, 1),
        grid_spec=td.GridSpec.auto(wavelength=1.0),
        run_time=1e-12,
        sources=[pt_dipole],
    )


@pytest.fixture(autouse=True)
def fake_data(monkeypatch, basic_simulation):
    """Patch postprocess to return stub data bound to the correct simulation."""
    calls = {"postprocess": 0}

    def _fake_postprocess(path: str, lazy: bool = False):
        calls["postprocess"] += 1
        p = Path(path)
        sim = PATH_TO_SIM.get(str(p))
        if sim is None:
            # Try to recover task_id from file payload written by _fake_download
            try:
                txt = p.read_text()
                if "payload:" in txt:
                    task_id = txt.split("payload:", 1)[1].strip()
                    sim = TASK_TO_SIM.get(task_id)
            except Exception:
                pass
        if sim is None:
            # Last-resort fallback (keeps tests from crashing even if mapping failed)
            sim = basic_simulation
        return _FakeStubData(sim)

    monkeypatch.setattr(web.Tidy3dStubData, "postprocess", staticmethod(_fake_postprocess))
    return calls


def _patch_run_pipeline(monkeypatch):
    """Patch upload, start, monitor, and download to avoid network calls and map sims."""
    counters = {"upload": 0, "start": 0, "monitor": 0, "download": 0}
    _reset_fake_maps()  # isolate between tests

    def _extract_simulation(kwargs):
        """Extract the first td.Simulation object from upload kwargs."""
        if "simulation" in kwargs and isinstance(kwargs["simulation"], td.Simulation):
            return kwargs["simulation"]
        if "simulations" in kwargs:
            sims = kwargs["simulations"]
            if isinstance(sims, dict):
                for sim in sims.values():
                    if isinstance(sim, td.Simulation):
                        return sim
            elif isinstance(sims, (list, tuple)):
                for sim in sims:
                    if isinstance(sim, td.Simulation):
                        return sim
        return None

    def _fake_upload(**kwargs):
        counters["upload"] += 1
        task_id = f"{MOCK_TASK_ID}{kwargs['simulation']._hash_self()}"
        sim = _extract_simulation(kwargs)
        if sim is not None:
            TASK_TO_SIM[task_id] = sim
        return task_id

    def _fake_start(task_id, **kwargs):
        counters["start"] += 1

    def _fake_monitor(task_id, verbose=True):
        counters["monitor"] += 1

    def _fake_download(*, task_id, path, **kwargs):
        counters["download"] += 1
        # Ensure we have a simulation for this task id (even if upload wasn't called)
        sim = TASK_TO_SIM.get(task_id)
        Path(path).write_text(f"payload:{task_id}")
        if sim is not None:
            PATH_TO_SIM[str(Path(path))] = sim

    def _fake__check_folder(*args, **kwargs):
        pass

    def _fake_status(self):
        return "success"

    monkeypatch.setattr(WebContainer, "_check_folder", _fake__check_folder)
    monkeypatch.setattr(web, "upload", _fake_upload)
    monkeypatch.setattr(web, "start", _fake_start)
    monkeypatch.setattr(web, "monitor", _fake_monitor)
    monkeypatch.setattr(web, "download", _fake_download)
    monkeypatch.setattr(web, "estimate_cost", lambda *args, **kwargs: 0.0)
    monkeypatch.setattr(Job, "status", property(_fake_status))
    monkeypatch.setattr(
        web,
        "get_info",
        lambda task_id, verbose=True: type(
            "_Info", (), {"solverVersion": "solver-1", "taskType": "FDTD"}
        )(),
    )
    monkeypatch.setattr(
        web, "load_simulation", lambda task_id, *args, **kwargs: TASK_TO_SIM[task_id]
    )
    return counters


def _reset_counters(counters: dict[str, int]) -> None:
    for key in counters:
        counters[key] = 0


def _test_run_cache_hit(monkeypatch, tmp_path, basic_simulation, fake_data):
    counters = _patch_run_pipeline(monkeypatch)
    out_path = tmp_path / "result.hdf5"
    clear()

    data = web.run(basic_simulation, task_name="demo", path=str(out_path))
    assert isinstance(data, _FakeStubData)
    assert counters == {"upload": 1, "start": 1, "monitor": 1, "download": 1}

    _reset_counters(counters)
    data2 = web.run(basic_simulation, task_name="demo", path=str(out_path))
    assert isinstance(data2, _FakeStubData)
    assert counters == {"upload": 0, "start": 0, "monitor": 0, "download": 0}


def _test_load_simulation_if_cached(monkeypatch, tmp_path, basic_simulation):
    counters = _patch_run_pipeline(monkeypatch)
    out_path = tmp_path / "result_load_simulation_if_cached.hdf5"
    clear()

    data = web.run(basic_simulation, task_name="demo", path=str(out_path))
    assert isinstance(data, _FakeStubData)
    assert counters == {"upload": 1, "start": 1, "monitor": 1, "download": 1}

    sim_data_from_cache = load_simulation_if_cached(basic_simulation)
    assert sim_data_from_cache.simulation == basic_simulation

    out_path2 = tmp_path / "result_load_simulation_if_cached2.hdf5"
    sim_data_from_cache_with_path = load_simulation_if_cached(basic_simulation, path=out_path2)
    assert sim_data_from_cache_with_path.simulation == basic_simulation


def _test_mode_solver_caching(monkeypatch, tmp_path):
    counters = _patch_run_pipeline(monkeypatch)

    # store in cache
    mode_sim = make_mode_sim()
    mode_sim_data = web.run(mode_sim)

    # test basic loading from cache
    from_cache_data = load_simulation_if_cached(mode_sim)
    assert from_cache_data is not None
    assert isinstance(from_cache_data, _FakeStubData)
    assert mode_sim_data.simulation == from_cache_data.simulation

    # test loading from run
    _reset_counters(counters)
    mode_sim_data_run = web.run(mode_sim)
    assert counters["download"] == 0
    assert isinstance(mode_sim_data_run, _FakeStubData)
    assert mode_sim_data.simulation == mode_sim_data_run.simulation

    # test loading from job
    _reset_counters(counters)
    job = Job(simulation=mode_sim, task_name="test")
    job_data = job.run()
    assert counters["download"] == 0
    assert isinstance(job_data, _FakeStubData)
    assert mode_sim_data.simulation == job_data.simulation

    # test loading from batch
    _reset_counters(counters)
    mode_sim_batch = Batch(simulations={"sim1": mode_sim})
    batch_data = mode_sim_batch.run(path_dir=tmp_path)
    mode_sim_data_batch = batch_data["sim1"]
    assert counters["download"] == 0
    assert isinstance(mode_sim_data_batch, _FakeStubData)
    assert mode_sim_data.simulation == mode_sim_data_batch.simulation

    cache = resolve_local_cache(True)
    # test storing via job
    cache.clear()
    Job(simulation=mode_sim, task_name="test").run()
    assert load_simulation_if_cached(mode_sim) is not None

    # test storing via batch
    cache.clear()
    batch_mode_data = Batch(simulations={"sim1": mode_sim}).run(path_dir=tmp_path)
    _ = batch_mode_data["sim1"]  # access to store
    assert load_simulation_if_cached(mode_sim) is not None


def _test_run_cache_hit_async(monkeypatch, basic_simulation, tmp_path):
    counters = _patch_run_pipeline(monkeypatch)
    monkeypatch.setattr(config.local_cache, "max_entries", 128)
    monkeypatch.setattr(config.local_cache, "max_size_gb", 10)
    cache = resolve_local_cache(use_cache=True)
    cache.clear()
    _reset_fake_maps()

    _reset_counters(counters)
    sim2 = basic_simulation.updated_copy(shutoff=1e-4)
    sim3 = basic_simulation.updated_copy(shutoff=1e-3)

    data = run_async({"task1": basic_simulation, "task2": sim2}, path_dir=str(tmp_path))
    data_task1 = data["task1"]  # access to store in cache
    data_task2 = data["task2"]  # access to store in cache
    assert counters["download"] == 2
    assert isinstance(data_task1, _FakeStubData)
    assert isinstance(data_task2, _FakeStubData)
    assert len(cache) == 2

    _reset_counters(counters)
    run_async({"task1": basic_simulation, "task2": sim2}, path_dir=str(tmp_path))
    assert counters["download"] == 0
    assert isinstance(data_task1, _FakeStubData)
    assert len(cache) == 2

    _reset_counters(counters)
    data = run_async({"task1": basic_simulation, "task3": sim3}, path_dir=str(tmp_path))

    data_task1 = data["task1"]
    data_task2 = data["task3"]  # access to store in cache

    assert counters["download"] == 1  # sim3 is new
    assert isinstance(data_task1, _FakeStubData)
    assert isinstance(data_task2, _FakeStubData)
    assert len(cache) == 3


def _test_job_run_cache(monkeypatch, basic_simulation, tmp_path):
    counters = _patch_run_pipeline(monkeypatch)
    cache = resolve_local_cache(use_cache=True)
    cache.clear()
    job = Job(simulation=basic_simulation, task_name="test")
    job.run()

    assert len(cache) == 1

    _reset_counters(counters)

    job2 = Job(simulation=basic_simulation, task_name="test")
    out1_path = str(tmp_path / "result.hdf5")
    out2_path = str(tmp_path / "result2.hdf5")
    job2.run(path=out1_path)
    assert len(cache) == 1
    assert counters["download"] == 0

    job.load(path=out2_path)
    assert os.path.exists(out1_path)
    assert os.path.exists(out2_path)


def _test_autograd_cache(monkeypatch):
    counters = _patch_run_pipeline(monkeypatch)
    cache = resolve_local_cache(use_cache=True)
    cache.clear()

    functions = get_functions(ALL_KEY, "mode")
    make_sim = functions["sim"]
    sim = make_sim(params0)
    web.run(sim)
    assert counters["download"] == 1
    assert len(cache) == 1

    _reset_counters(counters)
    sim = make_sim(params0)
    web.run(sim)
    assert counters["download"] == 0
    assert len(cache) == 1


def _test_load_cache_hit(monkeypatch, tmp_path, basic_simulation, fake_data):
    clear()
    counters = _patch_run_pipeline(monkeypatch)
    out_path = tmp_path / "load.hdf5"

    cache = resolve_local_cache(use_cache=True)

    web.run(basic_simulation, task_name="demo", path=str(out_path))
    assert counters["download"] == 1
    assert len(cache) == 1

    _reset_counters(counters)
    data = web.load(None, path=str(out_path))
    assert isinstance(data, _FakeStubData)
    assert counters["download"] == 0  # served from cache
    assert len(cache) == 1  # still 1 item in cache


def _test_checksum_mismatch_triggers_refresh(monkeypatch, tmp_path, basic_simulation):
    out_path = tmp_path / "checksum.hdf5"
    clear()

    web.run(basic_simulation, task_name="demo", path=str(out_path))

    cache = resolve_local_cache(use_cache=True)
    metadata = cache.list()[0]
    corrupted_path = cache.root / metadata["cache_key"] / CACHE_ARTIFACT_NAME
    corrupted_path.write_text("corrupted")

    cache._fetch(metadata["cache_key"])
    assert len(cache) == 0


def _test_cache_eviction_by_entries(monkeypatch, tmp_path_factory, basic_simulation):
    monkeypatch.setattr(config.local_cache, "max_entries", 1)
    cache = resolve_local_cache(use_cache=True)
    cache.clear()

    file1 = tmp_path_factory.mktemp("art1") / CACHE_ARTIFACT_NAME
    file1.write_text("a")
    cache.store_result(_FakeStubData(basic_simulation), MOCK_TASK_ID, str(file1), "FDTD")
    assert len(cache) == 1

    sim2 = basic_simulation.updated_copy(shutoff=1e-4)
    file2 = tmp_path_factory.mktemp("art2") / CACHE_ARTIFACT_NAME
    file2.write_text("b")
    cache.store_result(_FakeStubData(sim2), MOCK_TASK_ID, str(file2), "FDTD")

    entries = cache.list()
    assert len(entries) == 1
    assert entries[0]["simulation_hash"] == sim2._hash_self()


def _test_cache_eviction_by_size(monkeypatch, tmp_path_factory, basic_simulation):
    monkeypatch.setattr(config.local_cache, "max_size_gb", float(10_000 * 1e-9))
    cache = resolve_local_cache(use_cache=True)
    cache.clear()

    file1 = tmp_path_factory.mktemp("art1") / CACHE_ARTIFACT_NAME
    file1.write_text("a" * 8_000)
    cache.store_result(_FakeStubData(basic_simulation), MOCK_TASK_ID, str(file1), "FDTD")
    assert len(cache) == 1

    sim2 = basic_simulation.updated_copy(shutoff=1e-4)
    file2 = tmp_path_factory.mktemp("art2") / CACHE_ARTIFACT_NAME
    file2.write_text("b" * 8_000)
    cache.store_result(_FakeStubData(sim2), MOCK_TASK_ID, str(file2), "FDTD")

    entries = cache.list()
    assert len(cache) == 1
    assert entries[0]["simulation_hash"] == sim2._hash_self()


def _test_configure_cache_roundtrip(monkeypatch, tmp_path):
    monkeypatch.setattr(config.local_cache, "enabled", True)
    monkeypatch.setattr(config.local_cache, "directory", tmp_path)
    monkeypatch.setattr(config.local_cache, "max_size_gb", 1.23)
    monkeypatch.setattr(config.local_cache, "max_entries", 5)

    local_cache = resolve_local_cache()
    assert local_cache is not None
    assert local_cache._root == tmp_path
    assert local_cache.max_size_gb == 1.23
    assert local_cache.max_entries == 5


def _test_env_var_overrides(monkeypatch, tmp_path):
    cache_dir = tmp_path / "cache"
    monkeypatch.setenv("TIDY3D_LOCAL_CACHE__ENABLED", "true")
    monkeypatch.setenv("TIDY3D_LOCAL_CACHE__DIRECTORY", str(cache_dir))
    monkeypatch.setenv("TIDY3D_LOCAL_CACHE__MAX_SIZE_GB", "0.5")
    monkeypatch.setenv("TIDY3D_LOCAL_CACHE__MAX_ENTRIES", "7")
    manager = get_manager()
    manager._reload()

    cache = resolve_local_cache()
    assert cache is not None
    assert cache._root == cache_dir
    assert cache.max_size_gb == 0.5
    assert cache.max_entries == 7

    monkeypatch.delenv("TIDY3D_LOCAL_CACHE__ENABLED", raising=False)
    monkeypatch.delenv("TIDY3D_LOCAL_CACHE__DIRECTORY", raising=False)
    monkeypatch.delenv("TIDY3D_LOCAL_CACHE__MAX_SIZE_GB", raising=False)
    monkeypatch.delenv("TIDY3D_LOCAL_CACHE__MAX_ENTRIES", raising=False)
    manager = get_manager()
    manager._reload()


def test_cache_sequential(monkeypatch, tmp_path, tmp_path_factory, basic_simulation, fake_data):
    """Run all critical cache tests in sequence to ensure stability."""
    monkeypatch.setattr(config.local_cache, "enabled", True)

    # this at first as runtime changes overrides env
    _test_env_var_overrides(monkeypatch, tmp_path)

    _test_load_simulation_if_cached(monkeypatch, tmp_path, basic_simulation)
    _test_run_cache_hit(monkeypatch, tmp_path, basic_simulation, fake_data)
    _test_load_cache_hit(monkeypatch, tmp_path, basic_simulation, fake_data)
    _test_checksum_mismatch_triggers_refresh(monkeypatch, tmp_path, basic_simulation)
    _test_cache_eviction_by_entries(monkeypatch, tmp_path_factory, basic_simulation)
    _test_cache_eviction_by_size(monkeypatch, tmp_path_factory, basic_simulation)
    _test_run_cache_hit_async(monkeypatch, basic_simulation, tmp_path)
    _test_job_run_cache(monkeypatch, basic_simulation, tmp_path)
    _test_autograd_cache(monkeypatch)
    _test_configure_cache_roundtrip(monkeypatch, tmp_path)
    _test_mode_solver_caching(monkeypatch, tmp_path)
