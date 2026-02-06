"""Tests that local cache works correctly with ModalComponentModeler via web.run()."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import tidy3d as td
from tests.test_plugins.smatrix.test_component_modeler import make_component_modeler
from tests.utils import run_emulated
from tidy3d.config import get_manager
from tidy3d.web.api import webapi as web
from tidy3d.web.api.container import WebContainer
from tidy3d.web.api.tidy3d_stub import Tidy3dStubData
from tidy3d.web.cache import resolve_local_cache
from tidy3d.web.core.task_core import BatchTask, SimulationTask

MOCK_TASK_ID = "task-modeler-cache"

# --- Fake pipeline maps ---
TASK_TO_SIM: dict[str, td.Simulation] = {}
PATH_TO_SIM: dict[str, td.Simulation] = {}


class _FakeModelerStubData:
    """Fake stub data that mimics ModalComponentModelerData loaded from disk.

    The real ModalComponentModelerData has a ``modeler`` attribute but no
    ``simulation`` attribute, so ``getattr(stub_data, 'simulation', None)``
    returns None inside ``store_result``.  This class reproduces that
    behaviour faithfully.
    """

    def __init__(self, modeler):
        from tidy3d import SimulationDataMap
        from tidy3d.plugins.smatrix.data.modal import ModalComponentModelerData

        sim_dict = modeler.sim_dict
        batch_data = {task_name: run_emulated(sim) for task_name, sim in sim_dict.items()}
        port_data = SimulationDataMap(
            keys=tuple(batch_data.keys()),
            values=tuple(batch_data.values()),
        )
        self._modeler_data = ModalComponentModelerData(modeler=modeler, data=port_data)
        # Mirror real ModalComponentModelerData: has .modeler, NOT .simulation
        self.modeler = modeler

    def smatrix(self):
        return self._modeler_data.smatrix()

    def __getattr__(self, name):
        return getattr(self._modeler_data, name)


@pytest.fixture(autouse=True)
def _isolate_local_cache(tmp_path, monkeypatch):
    """Keep cache operations in a temp dir and avoid moving/deleting real cache."""
    import tidy3d.web.cache as cache_mod
    from tidy3d.config import get_manager

    real_remove_cache_dir = cache_mod._remove_cache_dir
    temp_cache_dir = (tmp_path / "cache").resolve()

    def _safe_remove_cache_dir(path, *, recreate):
        target = Path(path).resolve()
        allowed_root = tmp_path.resolve().parent
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
    cache_mod.LocalCache(
        directory=temp_cache_dir,
        max_entries=td.config.local_cache.max_entries,
        max_size_gb=td.config.local_cache.max_size_gb,
    ).clear(hard=True)
    cache_mod._CACHE = None


def _reset_fake_maps():
    TASK_TO_SIM.clear()
    PATH_TO_SIM.clear()


def _patch_run_pipeline(monkeypatch, modeler):
    """Patch upload, start, monitor, download, and postprocess for modeler runs."""
    counters = {"upload": 0, "start": 0, "monitor": 0, "download": 0}
    _reset_fake_maps()

    fake_stub = _FakeModelerStubData(modeler)

    def _fake_upload(**kwargs):
        counters["upload"] += 1
        task_id = f"{MOCK_TASK_ID}-{counters['upload']}"
        sim = kwargs.get("simulation")
        if sim is not None:
            TASK_TO_SIM[task_id] = sim
        return task_id

    def _fake_start(task_id, **kwargs):
        counters["start"] += 1

    def _fake_monitor(task_id, verbose=True):
        counters["monitor"] += 1

    def _fake_download(*, task_id, path, **kwargs):
        counters["download"] += 1
        Path(path).write_text(f"payload:{task_id}")

    def _fake_postprocess(path, lazy=False):
        return fake_stub

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
    monkeypatch.setattr(Tidy3dStubData, "postprocess", staticmethod(_fake_postprocess))
    monkeypatch.setattr(
        web,
        "get_info",
        lambda task_id, verbose=True: SimpleNamespace(
            solverVersion="solver-1", taskType="MODAL_CM"
        ),
    )
    monkeypatch.setattr(
        web,
        "load_simulation",
        lambda task_id, *args, **kwargs: modeler,
    )
    monkeypatch.setattr(
        SimulationTask, "get", lambda *args, **kwargs: SimpleNamespace(taskType="MODAL_CM")
    )
    monkeypatch.setattr(
        BatchTask, "detail", lambda *args, **kwargs: SimpleNamespace(status="success")
    )
    return counters


def _reset_counters(counters):
    for key in counters:
        counters[key] = 0


def test_modal_component_modeler_cache_hit(monkeypatch, tmp_path):
    """Test that running a ModalComponentModeler via web.run stores results in cache
    and that a second identical run gets a cache hit (no upload/start/monitor/download)."""
    modeler = make_component_modeler()
    counters = _patch_run_pipeline(monkeypatch, modeler)
    cache = resolve_local_cache(use_cache=True)
    cache.clear()

    out_path = tmp_path / "modeler_result.hdf5"

    # First run: should make web calls and store in cache
    data = web.run(modeler, task_name="modeler_cache_test", path=str(out_path))
    assert counters["upload"] == 1
    assert counters["start"] == 1
    assert counters["monitor"] == 1
    assert counters["download"] == 1

    # Verify smatrix works
    s_matrix = data.smatrix()
    assert s_matrix is not None

    # Verify cache has an entry
    assert len(cache) == 1, (
        f"Expected 1 cache entry after first run, got {len(cache)}. "
        "The ModalComponentModelerData was not stored in cache."
    )

    # Second run: should be served from cache (no web calls)
    _reset_counters(counters)
    data2 = web.run(modeler, task_name="modeler_cache_test", path=str(out_path))
    assert counters["upload"] == 0, "Expected no upload on cache hit"
    assert counters["start"] == 0, "Expected no start on cache hit"
    assert counters["monitor"] == 0, "Expected no monitor on cache hit"
    assert counters["download"] == 0, "Expected no download on cache hit"

    # Verify smatrix still works from cached data
    s_matrix2 = data2.smatrix()
    assert s_matrix2 is not None


def test_modal_component_modeler_cache_stores_entry(monkeypatch, tmp_path):
    """Test that cache.store_result works for ModalComponentModelerData.

    The core issue: ModalComponentModelerData has a ``modeler`` attribute,
    not ``simulation``.  ``store_result`` uses
    ``getattr(stub_data, 'simulation', None)`` to obtain the simulation
    object for hashing.  This returns None for modeler data, so the entry
    is never stored.
    """
    modeler = make_component_modeler()
    cache = resolve_local_cache(use_cache=True)
    cache.clear()

    # Create fake modeler data (mirrors real ModalComponentModelerData: no .simulation)
    fake_data = _FakeModelerStubData(modeler)

    # Write a dummy artifact file (store_result needs a file to copy)
    artifact = tmp_path / "dummy.hdf5"
    artifact.write_text("dummy-payload")

    # Try storing — this tests the core issue
    stored = cache.store_result(
        stub_data=fake_data,
        task_id="test-task-123",
        path=str(artifact),
        workflow_type="MODAL_CM",
    )

    assert stored, (
        "cache.store_result returned False for ModalComponentModelerData. "
        "This likely means getattr(stub_data, 'simulation', None) returned None "
        "because the attribute is 'modeler' not 'simulation'."
    )
    assert len(cache) == 1
