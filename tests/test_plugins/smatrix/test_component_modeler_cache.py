"""Tests that local cache works correctly with ModalComponentModeler via web.run()."""

from __future__ import annotations

from tests.test_plugins.smatrix.test_component_modeler import make_component_modeler
from tests.test_web.test_local_cache import (
    _isolate_local_cache,  # noqa: F401
    _patch_run_pipeline,
    _reset_counters,
)
from tests.utils import run_emulated
from tidy3d import SimulationDataMap
from tidy3d.plugins.smatrix.data.modal import ModalComponentModelerData
from tidy3d.web.api import webapi as web
from tidy3d.web.cache import resolve_local_cache


class _FakeModelerStubData:
    """Fake stub data that mimics ModalComponentModelerData loaded from disk.

    The real ModalComponentModelerData has a ``modeler`` attribute but no
    ``simulation`` attribute, so ``getattr(stub_data, 'simulation', None)``
    returns None inside ``store_result``.  This class reproduces that
    behaviour faithfully.
    """

    def __init__(self, modeler):
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


def test_modal_component_modeler_cache_hit(monkeypatch, tmp_path):
    """Test that running a ModalComponentModeler via web.run stores results in cache
    and that a second identical run gets a cache hit (no upload/start/monitor/download)."""
    modeler = make_component_modeler()
    fake_stub = _FakeModelerStubData(modeler)
    counters = _patch_run_pipeline(
        monkeypatch,
        task_type="MODAL_CM",
        postprocess=lambda path, lazy=False: fake_stub,
        load_simulation_fn=lambda task_id, path="simulation.json", verbose=True: modeler,
        patch_autograd=False,
    )
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
