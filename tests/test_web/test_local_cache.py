from __future__ import annotations

import importlib
import io
import json
import multiprocessing as mp
import os
import queue
import re
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import autograd as ag
import pytest
from autograd.core import defvjp
from click.testing import CliRunner
from rich.console import Console

import tidy3d as td
import tidy3d.web.cache as cache_mod
from tests.test_components.autograd.test_autograd import ALL_KEY, get_functions, params0
from tests.test_web.test_webapi_mode import make_mode_sim
from tests.utils import run_emulated
from tidy3d import config
from tidy3d.components.autograd.field_map import FieldMap
from tidy3d.config import get_manager
from tidy3d.web import Job, common, run, run_async
from tidy3d.web.api import webapi as web
from tidy3d.web.api.autograd import autograd, engine, io_utils
from tidy3d.web.api.autograd.autograd import run as run_autograd
from tidy3d.web.api.autograd.constants import SIM_VJP_FILE
from tidy3d.web.api.container import Batch, WebContainer
from tidy3d.web.api.webapi import load_simulation_if_cached
from tidy3d.web.cache import (
    CACHE_ARTIFACT_NAME,
    CACHE_STATS_NAME,
    TMP_BATCH_PREFIX,
    TMP_PREFIX,
    clear,
    get_cache_entry_dir,
    resolve_local_cache,
)
from tidy3d.web.cli.app import tidy3d_cli
from tidy3d.web.core.task_core import BatchTask, SimulationTask

common.CONNECTION_RETRY_TIME = 0.1

MOCK_TASK_ID = "task-xyz"
MP_BARRIER_TIMEOUT_S = 30  # long timeout for slow test runners
MP_JOIN_TIMEOUT_S = 60  # long timeout for slow test runners
# --- Fake pipeline global maps / queue ---
TASK_TO_SIM: dict[str, td.Simulation] = {}  # task_id -> Simulation
PATH_TO_SIM: dict[str, td.Simulation] = {}  # artifact path -> Simulation


@pytest.fixture(autouse=True)
def _isolate_local_cache(tmp_path, monkeypatch):
    """Keep cache operations in a temp dir and avoid moving/deleting real cache."""
    import tidy3d.web.cache as cache_mod
    from tidy3d.config import get_manager

    real_remove_cache_dir = cache_mod._remove_cache_dir
    temp_cache_dir = (tmp_path / "cache").resolve()

    def _safe_remove_cache_dir(path, *, recreate):
        target = Path(path).resolve()
        allowed_root = tmp_path.resolve()
        try:
            target.relative_to(allowed_root)
        except ValueError:
            return
        real_remove_cache_dir(target, recreate=recreate)

    monkeypatch.setattr(cache_mod, "_CACHE", None)
    monkeypatch.setenv("TIDY3D_LOCAL_CACHE__ENABLED", "true")
    monkeypatch.setenv("TIDY3D_LOCAL_CACHE__DIRECTORY", str(temp_cache_dir))
    manager = get_manager()
    manager._runtime_overrides.clear()
    manager._reload()
    monkeypatch.setattr(cache_mod, "_remove_cache_dir", _safe_remove_cache_dir)
    yield
    # Avoid resolve_local_cache() here to prevent migration into production paths.
    cache_mod.LocalCache(
        directory=temp_cache_dir,
        max_entries=config.local_cache.max_entries,
        max_size_gb=config.local_cache.max_size_gb,
    ).clear(hard=True)
    cache_mod._CACHE = None


def _reset_fake_maps():
    TASK_TO_SIM.clear()
    PATH_TO_SIM.clear()


def _stats_update_worker(
    cache_dir: str, barrier, key: str, file_size: int, sleep_s: float, error_queue
) -> None:
    try:
        cache = cache_mod.LocalCache(directory=cache_dir, max_entries=10, max_size_gb=1.0)

        original_write_stats = cache_mod.LocalCache._write_stats

        def _sleeping_write_stats(self, stats):
            time.sleep(sleep_s)
            return original_write_stats(self, stats)

        cache_mod.LocalCache._write_stats = _sleeping_write_stats

        artifact = Path(cache_dir) / f"artifact-{key}.hdf5"
        artifact.write_text("x" * file_size)
        metadata = cache_mod.build_entry_metadata(
            simulation_hash=key,
            workflow_type="FDTD",
            task_id=key,
            version="test",
            path=artifact,
        )

        barrier.wait(timeout=MP_BARRIER_TIMEOUT_S)
        cache._store(key=key, source_path=artifact, metadata=metadata)
    except Exception:
        error_queue.put(traceback.format_exc())
        raise


def _store_loop_worker(
    cache_dir: str,
    barrier,
    key: str,
    iterations: int,
    payload_size: int,
    sleep_s: float,
    error_queue,
) -> None:
    try:
        cache = cache_mod.LocalCache(directory=cache_dir, max_entries=10, max_size_gb=1.0)

        original_write_stats = cache_mod.LocalCache._write_stats

        def _sleeping_write_stats(self, stats):
            time.sleep(sleep_s)
            return original_write_stats(self, stats)

        cache_mod.LocalCache._write_stats = _sleeping_write_stats

        barrier.wait(timeout=MP_BARRIER_TIMEOUT_S)
        for idx in range(iterations):
            artifact = Path(cache_dir) / f"artifact-{key}-{idx}.hdf5"
            artifact.write_text("x" * payload_size)
            metadata = cache_mod.build_entry_metadata(
                simulation_hash=f"{key}-{idx}",
                workflow_type="FDTD",
                task_id=key,
                version="test",
                path=artifact,
            )
            cache._store(key=key, source_path=artifact, metadata=metadata)
    except Exception:
        error_queue.put(traceback.format_exc())
        raise


def _invalidate_loop_worker(
    cache_dir: str,
    barrier,
    key: str,
    iterations: int,
    sleep_s: float,
    error_queue,
) -> None:
    try:
        cache = cache_mod.LocalCache(directory=cache_dir, max_entries=10, max_size_gb=1.0)

        original_write_stats = cache_mod.LocalCache._write_stats

        def _sleeping_write_stats(self, stats):
            time.sleep(sleep_s)
            return original_write_stats(self, stats)

        cache_mod.LocalCache._write_stats = _sleeping_write_stats

        barrier.wait(timeout=MP_BARRIER_TIMEOUT_S)
        for _ in range(iterations):
            cache.invalidate(key)
    except Exception:
        error_queue.put(traceback.format_exc())
        raise


class _FakeStubData:
    def __init__(self, simulation: td.Simulation):
        self.simulation = simulation
        self.sim_data = run_emulated(self.simulation)

    def __getitem__(self, key):
        return self.sim_data[key]

    def _strip_traced_fields(self, *args, **kwargs):
        """Fake _strip_traced_fields: return minimal valid autograd-style mapping."""
        return self.sim_data._strip_traced_fields(*args, **kwargs)

    def _insert_traced_fields(self, field_mapping, *args, **kwargs):
        return self.sim_data._insert_traced_fields(field_mapping, *args, **kwargs)

    def _make_adjoint_sims(self, **kwargs):
        return self.sim_data._make_adjoint_sims(**kwargs)


@pytest.fixture
def basic_simulation():
    pulse = td.GaussianPulse(freq0=200e12, fwidth=20e12)
    pt_dipole = td.PointDipole(source_time=pulse, polarization="Ex")
    return td.Simulation(
        size=(1, 1, 1),
        grid_spec=td.GridSpec.auto(wavelength=1.0),
        run_time=1e-12,
        sources=(pt_dipole,),
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


def _patch_run_pipeline(
    monkeypatch,
    task_type: str = "FDTD",
    postprocess=None,
    load_simulation_fn=None,
    patch_autograd: bool = True,
):
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
        sim = _extract_simulation(kwargs)
        if sim is not None:
            task_id = f"{MOCK_TASK_ID}{sim._hash_self()}"
        else:
            task_id = f"{MOCK_TASK_ID}-{counters['upload']}"
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

    def _fake_load_simulation(task_id, path="simulation.json", verbose=True):
        if load_simulation_fn is not None:
            return load_simulation_fn(task_id=task_id, path=path, verbose=verbose)
        sim = TASK_TO_SIM.get(task_id)
        if sim is None:
            sim = next(iter(PATH_TO_SIM.values()), None)
        if sim is None:
            raise RuntimeError(f"No simulation mapped for task_id {task_id}")
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        path_obj.write_text("{}")
        return sim

    def _fake__check_folder(*args, **kwargs):
        pass

    def _fake_status(self):
        return "success"

    def _fake_download_file(resource_id, remote_filename, to_file=None, **kwargs):
        # Only count this download if it's the adjoint/VJP file
        if str(remote_filename) == SIM_VJP_FILE:
            counters["download"] += 1

    def _fake_postprocess_fwd(*, sim_data_combined=None, sim_original=None, context=None, **_):
        """Mimic ``autograd.postprocess_fwd`` side effects for tests."""
        if sim_original is None:
            sim_original = next(iter(PATH_TO_SIM.values()), None)
        if sim_original is None:
            sim_original = td.Simulation(
                size=(1, 1, 1),
                grid_spec=td.GridSpec.auto(wavelength=1.0),
                run_time=1e-12,
            )
        stub_data = _FakeStubData(sim_original)
        if context is not None:
            context.simulation_data_original = stub_data
            context.simulation_data_forward = stub_data
        return stub_data._strip_traced_fields()

    def _fake_postprocess_adj(
        sim_data_adj=None, sim_data_orig=None, sim_data_fwd=None, sim_fields_keys=None, **_
    ):
        """Return zeros for every requested field key."""
        counters["download"] += 1  # mimic VJP file download per autograd run
        sim_fields_keys = sim_fields_keys or []
        return dict.fromkeys(sim_fields_keys, 0.0)

    def _fake_field_map_from_file(*args, **kwargs):
        return FieldMap(tracers=())

    if patch_autograd:
        monkeypatch.setattr(io_utils, "download_file", _fake_download_file)
        monkeypatch.setattr(autograd, "postprocess_fwd", _fake_postprocess_fwd)
        monkeypatch.setattr(autograd, "postprocess_adj", _fake_postprocess_adj)
        monkeypatch.setattr(FieldMap, "from_file", _fake_field_map_from_file)
    monkeypatch.setattr(WebContainer, "_check_folder", _fake__check_folder)
    monkeypatch.setattr(web, "upload", _fake_upload)
    monkeypatch.setattr(web, "start", _fake_start)
    monkeypatch.setattr(web, "monitor", _fake_monitor)
    monkeypatch.setattr(web, "download", _fake_download)
    monkeypatch.setattr(web, "load_simulation", _fake_load_simulation)
    monkeypatch.setattr(web, "estimate_cost", lambda *args, **kwargs: 0.0)
    monkeypatch.setattr(Job, "status", property(_fake_status))
    if patch_autograd:
        monkeypatch.setattr(engine, "upload_sim_fields_keys", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        web,
        "get_info",
        lambda task_id, verbose=True: type(
            "_Info", (), {"solverVersion": "solver-1", "taskType": task_type, "status": "success"}
        )(),
    )
    if patch_autograd:
        monkeypatch.setattr(
            io_utils,
            "get_info",
            lambda task_id, verbose=True: type(
                "_Info",
                (),
                {"solverVersion": "solver-1", "taskType": task_type, "status": "success"},
            )(),
        )
    if postprocess is not None:
        monkeypatch.setattr(web.Tidy3dStubData, "postprocess", staticmethod(postprocess))
    monkeypatch.setattr(
        SimulationTask, "get", lambda *args, **kwargs: SimpleNamespace(taskType=task_type)
    )
    monkeypatch.setattr(
        BatchTask, "detail", lambda *args, **kwargs: SimpleNamespace(status="success")
    )
    return counters


def _reset_counters(counters: dict[str, int]) -> None:
    for key in counters:
        counters[key] = 0


def test_run_cache_hit(monkeypatch, tmp_path, basic_simulation, fake_data):
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


def test_load_simulation_if_cached(monkeypatch, tmp_path, basic_simulation):
    counters = _patch_run_pipeline(monkeypatch)
    out_path = tmp_path / "result_load_simulation_if_cached.hdf5"
    cache = resolve_local_cache(True)
    cache.clear()

    data = web.run(basic_simulation, task_name="demo", path=str(out_path))
    assert isinstance(data, _FakeStubData)
    assert counters == {"upload": 1, "start": 1, "monitor": 1, "download": 1}
    assert len(cache) == 1

    sim_data_from_cache = load_simulation_if_cached(basic_simulation, path=tmp_path / "tmp.hdf5")
    assert sim_data_from_cache is not None
    assert sim_data_from_cache.simulation == basic_simulation

    out_path2 = tmp_path / "result_load_simulation_if_cached2.hdf5"
    sim_data_from_cache_with_path = load_simulation_if_cached(basic_simulation, path=out_path2)
    assert sim_data_from_cache_with_path.simulation == basic_simulation


def test_mode_solver_caching(monkeypatch, tmp_path):
    counters = _patch_run_pipeline(monkeypatch)
    tmp_file = tmp_path / "tmp.hdf5"
    # store in cache
    mode_sim = make_mode_sim()
    mode_sim_data = web.run(mode_sim, path=tmp_file)

    # test basic loading from cache
    from_cache_data = load_simulation_if_cached(mode_sim, path=tmp_file)
    assert from_cache_data is not None
    assert isinstance(from_cache_data, _FakeStubData)
    assert mode_sim_data.simulation == from_cache_data.simulation

    # test loading from run
    _reset_counters(counters)
    mode_sim_data_run = web.run(mode_sim, path=tmp_file)
    assert counters["download"] == 0
    assert isinstance(mode_sim_data_run, _FakeStubData)
    assert mode_sim_data.simulation == mode_sim_data_run.simulation

    # test loading from job
    _reset_counters(counters)
    job = Job(simulation=mode_sim, task_name="test")
    job_data = job.run(path=tmp_file)
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
    Job(simulation=mode_sim, task_name="test").run(path=tmp_file)
    assert load_simulation_if_cached(mode_sim, path=tmp_file) is not None

    # test storing via batch
    cache.clear()
    batch_mode_data = Batch(simulations={"sim1": mode_sim}).run(path_dir=tmp_path)
    _ = batch_mode_data["sim1"]  # access to store
    assert load_simulation_if_cached(mode_sim, path=tmp_file) is not None


def test_run_cache_hit_async(monkeypatch, basic_simulation, tmp_path):
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


def test_verbosity(monkeypatch, basic_simulation, tmp_path):
    _CSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")  # ANSI CSI
    _OSC8_RE = re.compile(r"\x1b\]8;.*?(?:\x1b\\|\x07)", re.DOTALL)  # OSC-8 hyperlinks

    def _normalize_console_text(s: str) -> str:
        s = _OSC8_RE.sub("", s)  # drop hyperlinks
        s = _CSI_RE.sub("", s)  # drop colors, etc.
        s = s.replace("\r", "")
        s = re.sub(r"[ \t]+", " ", s)  # collapse runs of spaces/tabs
        return s.strip()

    buf = io.StringIO()
    test_console = Console(file=buf, force_terminal=True)

    counters = _patch_run_pipeline(monkeypatch)
    cache = resolve_local_cache(use_cache=True)
    cache.clear()
    _reset_fake_maps()
    _reset_counters(counters)
    sim2 = basic_simulation.updated_copy(shutoff=1e-4)
    sim3 = basic_simulation.updated_copy(shutoff=1e-3)
    tmp_file = tmp_path / "tmp.hdf5"
    run(basic_simulation, verbose=True, path=tmp_file)  # seed cache

    log_mod = importlib.import_module("tidy3d.log")

    # --- swap handler console (and restore later) ---
    if "console" not in log_mod.log.handlers:
        log_mod.set_logging_console()
    orig_console = log_mod.log.handlers["console"].console
    log_mod.log.handlers["console"].console = test_console
    try:
        buf.truncate(0)
        buf.seek(0)

        log_mod.get_logging_console().log("test")
        assert "test" in buf.getvalue()

        buf.truncate(0)
        buf.seek(0)

        # test for load_simulation_if_cached
        sim_data = load_simulation_if_cached(basic_simulation, verbose=True, path=tmp_file)
        assert sim_data is not None
        assert "Loading simulation from" in buf.getvalue(), (
            f"Expected 'Loading simulation from' in log, got '{buf.getvalue()}'"
        )

        buf.truncate(0)
        buf.seek(0)
        load_simulation_if_cached(basic_simulation, verbose=False, path=tmp_file)
        assert sim_data is not None
        assert buf.getvalue().strip() == "", f"Expected empty log, got '{buf.getvalue()}'"

        # test for batched runs
        buf.truncate(0)
        buf.seek(0)
        run([basic_simulation, sim3], verbose=True, path=tmp_path / "batch-1")
        txt = _normalize_console_text(buf.getvalue())
        assert "Got 1 simulation from cache" in txt, (
            f"Expected 'Got 1 simulation from cache' in log, got '{buf.getvalue()}'"
        )

        # if some found
        buf.truncate(0)
        buf.seek(0)
        run([basic_simulation, sim2], verbose=False, path=tmp_path / "batch-2")
        assert buf.getvalue().strip() == "", f"Expected empty log, got '{buf.getvalue()}'"

        # if all found
        buf.truncate(0)
        buf.seek(0)
        run([basic_simulation, sim2], verbose=False, path=tmp_path / "batch-3")
        assert buf.getvalue().strip() == "", f"Expected empty log, got '{buf.getvalue()}'"

    finally:
        # explicit restore so nothing leaks
        log_mod.log.handlers["console"].console = orig_console


def test_job_run_cache(monkeypatch, basic_simulation, tmp_path):
    counters = _patch_run_pipeline(monkeypatch)
    cache = resolve_local_cache(use_cache=True)
    cache.clear()
    job = Job(simulation=basic_simulation, task_name="test")
    job.run(path=tmp_path / "tmp.hdf5")

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


def test_run_uses_default_path_on_cache_hit(monkeypatch, basic_simulation, tmp_path):
    counters = _patch_run_pipeline(monkeypatch)
    cache = resolve_local_cache(use_cache=True)
    cache.clear()
    monkeypatch.chdir(tmp_path)

    default_path = tmp_path / "simulation_data.hdf5"

    run(basic_simulation, task_name="demo")
    assert default_path.exists()
    assert counters["download"] == 1

    default_path.unlink()
    _reset_counters(counters)

    run(basic_simulation, task_name="demo")
    assert counters["download"] == 0
    assert default_path.exists()


def test_job_download_uses_default_path_on_cache_hit(monkeypatch, basic_simulation, tmp_path):
    counters = _patch_run_pipeline(monkeypatch)
    cache = resolve_local_cache(use_cache=True)
    cache.clear()
    monkeypatch.chdir(tmp_path)

    Job(simulation=basic_simulation, task_name="job-download-default").run(
        path=tmp_path / "seed.hdf5"
    )

    default_path = tmp_path / "simulation_data.hdf5"
    if default_path.exists():
        default_path.unlink()
    _reset_counters(counters)

    cached_job = Job(simulation=basic_simulation, task_name="job-download-default")
    cached_job.download()

    assert counters["download"] == 0
    assert default_path.exists()


def test_job_load_uses_default_path_on_cache_hit(monkeypatch, basic_simulation, tmp_path):
    counters = _patch_run_pipeline(monkeypatch)
    cache = resolve_local_cache(use_cache=True)
    cache.clear()
    monkeypatch.chdir(tmp_path)

    Job(simulation=basic_simulation, task_name="job-load-default").run(path=tmp_path / "seed.hdf5")

    default_path = tmp_path / "simulation_data.hdf5"
    if default_path.exists():
        default_path.unlink()
    _reset_counters(counters)

    cached_job = Job(simulation=basic_simulation, task_name="job-load-default")
    data = cached_job.load()

    assert isinstance(data, _FakeStubData)
    assert counters["download"] == 0
    assert default_path.exists()


def test_autograd_cache(monkeypatch, request, tmp_path):
    counters = _patch_run_pipeline(monkeypatch)

    # "Original" rule: the one autograd uses by default
    def _orig_make_dict_vjp(ans, keys, vals):
        return lambda g: [g[key] for key in keys]

    def _zero_make_dict_vjp(ans, keys, vals):
        def vjp(g):
            # One gradient per entry in `vals`, all zeros, matching shape/dtype
            return [ag.numpy.zeros_like(v) for v in vals]

        return vjp

    # Install our zero-VJP (this is the thing that affects global state)
    defvjp(
        ag.builtins._make_dict,
        _zero_make_dict_vjp,
        argnums=(1,),  # gradient w.r.t. `vals`
    )

    # Make sure we restore it after the test
    def _restore_make_dict_vjp():
        defvjp(
            ag.builtins._make_dict,
            _orig_make_dict_vjp,
            argnums=(1,),
        )

    request.addfinalizer(_restore_make_dict_vjp)

    cache = resolve_local_cache(use_cache=True)
    cache.clear()

    functions = get_functions(ALL_KEY, "mode")
    make_sim = functions["sim"]
    postprocess = functions["postprocess"]

    def objective(params):
        sim = make_sim(params)
        sim_data = run_autograd(sim, path=tmp_path / "tmp.hdf5")
        value = postprocess(sim_data)
        return value

    ag.value_and_grad(objective)(params0)
    assert counters["download"] == 2
    assert len(cache) == 2

    _reset_counters(counters)
    ag.value_and_grad(objective)(params0)
    assert counters["download"] == 1  # download field data
    assert len(cache) == 2


def test_load_cache_hit(monkeypatch, tmp_path, basic_simulation, fake_data):
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


def test_checksum_mismatch_triggers_refresh(monkeypatch, tmp_path, basic_simulation):
    _patch_run_pipeline(monkeypatch)
    out_path = tmp_path / "checksum.hdf5"
    clear()

    web.run(basic_simulation, task_name="demo", path=str(out_path))

    cache = resolve_local_cache(use_cache=True)
    metadata = cache.list()[0]
    corrupted_path = get_cache_entry_dir(cache.root, metadata["cache_key"]) / CACHE_ARTIFACT_NAME
    corrupted_path.write_text("corrupted")

    cache._fetch(metadata["cache_key"])
    assert len(cache) == 0


def test_cache_eviction_by_entries(monkeypatch, tmp_path_factory, basic_simulation):
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


def test_cache_eviction_by_size(monkeypatch, tmp_path_factory, basic_simulation):
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


def test_cache_stats_tracking(monkeypatch, tmp_path_factory, basic_simulation):
    monkeypatch.setattr(config.local_cache, "max_entries", 10)
    cache = resolve_local_cache(use_cache=True)
    cache.clear()

    artifact = tmp_path_factory.mktemp("artifact_stats") / CACHE_ARTIFACT_NAME
    payload = "stats-payload"
    artifact.write_text(payload)

    cache.store_result(_FakeStubData(basic_simulation), MOCK_TASK_ID, str(artifact), "FDTD")

    stats_path = cache.root / CACHE_STATS_NAME
    assert stats_path.exists()
    stats = json.loads(stats_path.read_text())
    assert stats["total_entries"] == 1
    assert stats["total_size"] == len(payload)
    key = next(iter(stats["last_used"]))
    entry_last_used = stats["last_used"][key]
    assert isinstance(entry_last_used, str)

    time.sleep(0.001)
    cache_entry = cache._fetch(key)
    assert cache_entry is not None

    updated_stats = json.loads(stats_path.read_text())
    assert updated_stats["total_size"] == len(payload)
    assert updated_stats["last_used"][key] != entry_last_used

    cache.invalidate(key)
    final_stats = json.loads(stats_path.read_text())
    assert final_stats["total_entries"] == 0
    assert final_stats["total_size"] == 0
    assert final_stats["last_used"] == {}

    cache.clear()


def test_cache_stats_sync(monkeypatch, tmp_path_factory, basic_simulation):
    monkeypatch.setattr(config.local_cache, "max_entries", 10)
    cache = resolve_local_cache(use_cache=True)
    cache.clear()

    sim1 = basic_simulation
    sim2 = basic_simulation.updated_copy(shutoff=2e-4)

    artifact1 = tmp_path_factory.mktemp("artifact_sync1") / CACHE_ARTIFACT_NAME
    payload1 = "sync-one"
    artifact1.write_text(payload1)
    cache.store_result(_FakeStubData(sim1), f"{MOCK_TASK_ID}-1", str(artifact1), "FDTD")

    artifact2 = tmp_path_factory.mktemp("artifact_sync2") / CACHE_ARTIFACT_NAME
    payload2 = "sync-two"
    artifact2.write_text(payload2)
    cache.store_result(_FakeStubData(sim2), f"{MOCK_TASK_ID}-2", str(artifact2), "FDTD")

    stats_path = cache.root / CACHE_STATS_NAME
    assert stats_path.exists()

    stats_path.unlink()
    assert not stats_path.exists()

    rebuilt = cache.sync_stats()
    assert stats_path.exists()
    assert rebuilt.total_entries == 2
    assert rebuilt.total_size == len(payload1) + len(payload2)

    keys = {meta["cache_key"] for meta in cache.list()}
    assert set(rebuilt.last_used.keys()) == keys
    for info in rebuilt.last_used.values():
        assert isinstance(info, str)

    cache.clear()


@pytest.mark.skipif(
    (os.getenv("GITHUB_ACTIONS") or "").lower() == "true",
    reason="Skip on CI due to resource problems. Activate temporarily for targeted testing.",
)
def test_cache_stats_concurrent_updates():
    cache = resolve_local_cache(use_cache=True)
    cache.clear()

    cache_dir = str(cache.root)
    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(2)
    error_queue = ctx.Queue()

    key1 = "concurrent-key-1"
    key2 = "concurrent-key-2"
    file_size1 = 3
    file_size2 = 5
    sleep_s = 0.3

    proc1 = ctx.Process(
        target=_stats_update_worker,
        args=(cache_dir, barrier, key1, file_size1, sleep_s, error_queue),
    )
    proc2 = ctx.Process(
        target=_stats_update_worker,
        args=(cache_dir, barrier, key2, file_size2, sleep_s, error_queue),
    )

    proc1.start()
    proc2.start()

    proc1.join(timeout=MP_JOIN_TIMEOUT_S)
    proc2.join(timeout=MP_JOIN_TIMEOUT_S)

    if proc1.is_alive():
        proc1.terminate()
        proc1.join()
    if proc2.is_alive():
        proc2.terminate()
        proc2.join()

    errors = []
    while True:
        try:
            errors.append(error_queue.get_nowait())
        except queue.Empty:
            break

    if proc1.exitcode != 0 or proc2.exitcode != 0:
        error_text = "\n".join(errors) if errors else "No worker traceback captured."
        raise AssertionError(f"Worker exit codes: {proc1.exitcode}, {proc2.exitcode}\n{error_text}")

    stats_path = Path(cache_dir) / CACHE_STATS_NAME
    assert stats_path.exists()
    stats = json.loads(stats_path.read_text())
    assert stats["total_entries"] == 2
    assert stats["total_size"] == file_size1 + file_size2
    assert set(stats["last_used"].keys()) == {key1, key2}


@pytest.mark.skipif(
    (os.getenv("GITHUB_ACTIONS") or "").lower() == "true",
    reason="Skip on CI due to resource problems. Activate temporarily for targeted testing.",
)
def test_cache_stats_store_invalidate_race():
    cache = resolve_local_cache(use_cache=True)
    cache.clear()

    cache_dir = str(cache.root)
    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(2)
    error_queue = ctx.Queue()

    key = "race-key"
    iterations = 5
    payload_size = 7
    sleep_s = 0.05

    proc_store = ctx.Process(
        target=_store_loop_worker,
        args=(cache_dir, barrier, key, iterations, payload_size, sleep_s, error_queue),
    )
    proc_invalidate = ctx.Process(
        target=_invalidate_loop_worker,
        args=(cache_dir, barrier, key, iterations, sleep_s, error_queue),
    )

    proc_store.start()
    proc_invalidate.start()

    proc_store.join(timeout=MP_JOIN_TIMEOUT_S)
    proc_invalidate.join(timeout=MP_JOIN_TIMEOUT_S)

    if proc_store.is_alive():
        proc_store.terminate()
        proc_store.join()
    if proc_invalidate.is_alive():
        proc_invalidate.terminate()
        proc_invalidate.join()

    errors = []
    while True:
        try:
            errors.append(error_queue.get_nowait())
        except queue.Empty:
            break

    if proc_store.exitcode != 0 or proc_invalidate.exitcode != 0:
        error_text = "\n".join(errors) if errors else "No worker traceback captured."
        raise AssertionError(
            f"Worker exit codes: {proc_store.exitcode}, {proc_invalidate.exitcode}\n{error_text}"
        )

    stats_path = Path(cache_dir) / CACHE_STATS_NAME
    assert stats_path.exists()
    stats = json.loads(stats_path.read_text())

    cache_check = cache_mod.LocalCache(directory=cache_dir, max_entries=10, max_size_gb=1.0)
    entries = cache_check.list()
    entry_keys = {entry["cache_key"] for entry in entries}
    total_size = sum(entry["file_size"] for entry in entries)

    assert stats["total_entries"] == len(entries)
    assert stats["total_size"] == total_size
    assert set(stats["last_used"].keys()) == entry_keys


def test_store_and_fetch_do_not_iterate(monkeypatch, tmp_path, basic_simulation):
    cache = resolve_local_cache(use_cache=True)
    cache.clear()

    original_iter = cache._iter_entries
    iter_calls = {"count": 0}

    def _counting_iter():
        iter_calls["count"] += 1
        yield from original_iter()

    monkeypatch.setattr(cache, "_iter_entries", _counting_iter)

    artifact = tmp_path / "iter_guard.hdf5"
    artifact.write_text("payload")

    cache.store_result(_FakeStubData(basic_simulation), MOCK_TASK_ID, str(artifact), "FDTD")
    assert iter_calls["count"] == 0

    entry_dirs = []
    for prefix_dir in cache.root.iterdir():
        if not prefix_dir.is_dir() or prefix_dir.name.startswith((TMP_PREFIX, TMP_BATCH_PREFIX)):
            continue
        # collect child entry dirs (support nested layout)
        for child in prefix_dir.iterdir():
            if not child.is_dir() or child.name.startswith((TMP_PREFIX, TMP_BATCH_PREFIX)):
                continue
            entry_dirs.append(child.name)

    assert entry_dirs, "Expected stored cache entry"
    key = entry_dirs[0]

    before_fetch = iter_calls["count"]
    fetched_entry = cache._fetch(key)
    assert fetched_entry is not None
    assert iter_calls["count"] == before_fetch

    cache.clear()


def test_configure_cache_roundtrip(monkeypatch, tmp_path):
    monkeypatch.setattr(config.local_cache, "enabled", True)
    monkeypatch.setattr(config.local_cache, "directory", tmp_path)
    monkeypatch.setattr(config.local_cache, "max_size_gb", 1.23)
    monkeypatch.setattr(config.local_cache, "max_entries", 5)

    local_cache = resolve_local_cache()
    assert local_cache is not None
    assert local_cache._root == tmp_path
    assert local_cache.max_size_gb == 1.23
    assert local_cache.max_entries == 5


def test_env_var_overrides(monkeypatch, tmp_path):
    cache_dir = tmp_path / "cache"
    monkeypatch.setenv("TIDY3D_LOCAL_CACHE__ENABLED", "true")
    monkeypatch.setenv("TIDY3D_LOCAL_CACHE__DIRECTORY", str(cache_dir))
    monkeypatch.setenv("TIDY3D_LOCAL_CACHE__MAX_SIZE_GB", "0.5")
    monkeypatch.setenv("TIDY3D_LOCAL_CACHE__MAX_ENTRIES", "7")
    manager = get_manager()
    manager._reload()

    cache = resolve_local_cache()
    assert cache is not None
    assert cache._root == cache_dir.resolve()
    assert cache.max_size_gb == 0.5
    assert cache.max_entries == 7

    monkeypatch.delenv("TIDY3D_LOCAL_CACHE__ENABLED", raising=False)
    monkeypatch.delenv("TIDY3D_LOCAL_CACHE__DIRECTORY", raising=False)
    monkeypatch.delenv("TIDY3D_LOCAL_CACHE__MAX_SIZE_GB", raising=False)
    monkeypatch.delenv("TIDY3D_LOCAL_CACHE__MAX_ENTRIES", raising=False)
    manager = get_manager()
    manager._reload()


def test_cache_cli_commands(monkeypatch, tmp_path_factory, basic_simulation, tmp_path):
    runner = CliRunner()
    cache_dir = tmp_path
    artifact_dir = tmp_path_factory.mktemp("cli_cache_artifact")

    monkeypatch.setattr(config.local_cache, "enabled", True)
    monkeypatch.setattr(config.local_cache, "directory", cache_dir)
    monkeypatch.setattr(config.local_cache, "max_entries", 3)
    monkeypatch.setattr(config.local_cache, "max_size_gb", 2.5)

    cache = resolve_local_cache(use_cache=True)
    cache.clear()

    artifact = artifact_dir / CACHE_ARTIFACT_NAME
    artifact.write_text("payload_cli")
    cache.store_result(
        _FakeStubData(basic_simulation), f"{MOCK_TASK_ID}-cli", str(artifact), "FDTD"
    )

    info_result = runner.invoke(tidy3d_cli, ["cache", "info"])
    assert info_result.exit_code == 0
    assert "Enabled: yes" in info_result.output
    assert "Entries: 1" in info_result.output
    assert "Max entries: 3" in info_result.output
    assert "Max size: 2.50 GB" in info_result.output

    list_result = runner.invoke(tidy3d_cli, ["cache", "list"])
    assert list_result.exit_code == 0
    out = list_result.output
    assert "Cache Entry #1" in out
    assert "Workflow type: FDTD" in out
    assert "File size:" in out

    clear_result = runner.invoke(tidy3d_cli, ["cache", "clear"])
    assert clear_result.exit_code == 0
    assert "Local cache cleared." in clear_result.output
    assert len(cache) == 0

    list_after = runner.invoke(tidy3d_cli, ["cache", "list"])
    assert list_after.exit_code == 0
    assert "Cache is empty." in list_after.output


def test_cache_key_includes_workflow_type():
    """build_cache_key must produce different keys for different workflow types
    even when the simulation hash and version are identical.

    Regression test for a bug where a VolumeMesher result could be returned
    when a HeatChargeSimulation was run because both shared the same inner
    simulation hash and workflow_type was not part of the cache key.
    """
    from tidy3d.web.cache import build_cache_key

    common_hash = "abc123"
    version = "1.0"

    key_heat = build_cache_key(
        simulation_hash=common_hash, version=version, workflow_type="HEAT_CHARGE"
    )
    key_mesh = build_cache_key(
        simulation_hash=common_hash, version=version, workflow_type="VOLUME_MESH"
    )
    key_fdtd = build_cache_key(simulation_hash=common_hash, version=version, workflow_type="FDTD")

    assert key_heat != key_mesh, "HEAT_CHARGE and VOLUME_MESH must produce different cache keys"
    assert key_heat != key_fdtd, "HEAT_CHARGE and FDTD must produce different cache keys"
    assert key_mesh != key_fdtd, "VOLUME_MESH and FDTD must produce different cache keys"

    # Same inputs must be deterministic
    key_heat2 = build_cache_key(
        simulation_hash=common_hash, version=version, workflow_type="HEAT_CHARGE"
    )
    assert key_heat == key_heat2


def test_volume_mesh_cache_no_collision_with_heat(tmp_path_factory):
    """Storing a VOLUME_MESH result must not be returned when fetching a
    HeatChargeSimulation that shares the same inner simulation.

    Reproduces the notebook bug where web.run(sim, parent_tasks=[mesh_job.task_id])
    returned VolumeMesherData instead of running the heat solver.
    """
    heat_sim = td.HeatChargeSimulation(
        size=(3.0, 3.0, 3.0),
        medium=td.Medium(permittivity=1.0, heat_spec=td.FluidSpec()),
        structures=[
            td.Structure(
                geometry=td.Box(size=(1, 1, 1), center=(0, 0, 0)),
                medium=td.Medium(
                    permittivity=2.0,
                    heat_spec=td.SolidSpec(conductivity=1, capacity=1),
                ),
                name="box",
            ),
        ],
        grid_spec=td.UniformUnstructuredGrid(dl=0.5),
        sources=[td.HeatSource(rate=1, structures=["box"])],
        boundary_spec=[
            td.HeatChargeBoundarySpec(
                placement=td.StructureBoundary(structure="box"),
                condition=td.TemperatureBC(temperature=300),
            )
        ],
        monitors=[
            td.TemperatureMonitor(center=(0, 0, 0), size=(1, 1, 0), name="temp", unstructured=True),
        ],
    )

    mesher = td.VolumeMesher(
        simulation=heat_sim,
        monitors=[
            td.VolumeMeshMonitor(center=(0, 0, 0), size=(1, 1, 0), name="mesh"),
        ],
    )

    cache = resolve_local_cache(use_cache=True)
    cache.clear()

    # Store a fake result keyed as if it came from a VOLUME_MESH task.
    # Use simulation=None so store_result extracts stub_data.simulation (the HeatChargeSimulation).
    artifact = tmp_path_factory.mktemp("mesh_art") / CACHE_ARTIFACT_NAME
    artifact.write_text("mesh-payload")

    class _FakeVolumeMesherData:
        """Minimal stand-in for VolumeMesherData with .simulation = HeatChargeSimulation."""

        def __init__(self, sim):
            self.simulation = sim

    cache.store_result(
        stub_data=_FakeVolumeMesherData(heat_sim),
        task_id="vom-fake-id",
        path=str(artifact),
        workflow_type="VOLUME_MESH",
    )
    assert len(cache) == 1

    # try_fetch for the HeatChargeSimulation must NOT return the VOLUME_MESH entry
    entry = cache.try_fetch(heat_sim)
    assert entry is None, (
        "Cache returned a VOLUME_MESH entry for a HeatChargeSimulation lookup; "
        "workflow_type must be part of the cache key to prevent this collision."
    )

    # try_fetch for the VolumeMesher itself must also NOT match (different sim hash)
    entry_mesher = cache.try_fetch(mesher)
    assert entry_mesher is None

    # Storing a HEAT_CHARGE result should then be fetchable
    heat_artifact = tmp_path_factory.mktemp("heat_art") / CACHE_ARTIFACT_NAME
    heat_artifact.write_text("heat-payload")
    cache.store_result(
        stub_data=_FakeVolumeMesherData(heat_sim),
        task_id="hcs-fake-id",
        path=str(heat_artifact),
        workflow_type="HEAT_CHARGE",
    )
    assert len(cache) == 2

    entry_heat = cache.try_fetch(heat_sim)
    assert entry_heat is not None, "HEAT_CHARGE entry should be fetchable for HeatChargeSimulation"
