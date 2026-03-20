"""Tests that local cache works correctly with component modelers via web.run()."""

from __future__ import annotations

import numpy as np
import pytest

import tidy3d.plugins.smatrix.analysis.terminal as terminal_analysis
import tidy3d.plugins.smatrix.utils as smatrix_utils
from tests.test_plugins.smatrix.terminal_component_modeler_def import (
    make_component_modeler as make_terminal_component_modeler,
)
from tests.test_plugins.smatrix.test_component_modeler import make_component_modeler
from tests.test_web.test_local_cache import (
    _isolate_local_cache,  # noqa: F401
    _patch_run_pipeline,
    _reset_counters,
)
from tests.utils import run_emulated
from tidy3d import SimulationDataMap
from tidy3d.plugins.smatrix.data.modal import ModalComponentModelerData
from tidy3d.plugins.smatrix.data.terminal import TerminalComponentModelerData
from tidy3d.web import run as public_run
from tidy3d.web.api import webapi as web
from tidy3d.web.cache import resolve_local_cache
from tidy3d.web.core.types import TaskType


class _FakeComponentModelerStubData:
    """Fake stub data that mimics ComponentModelerData loaded from disk.

    The real ComponentModelerData has a ``modeler`` attribute but no
    ``simulation`` attribute, so ``getattr(stub_data, 'simulation', None)``
    returns None inside ``store_result``.  This class reproduces that
    behaviour faithfully.
    """

    def __init__(self, modeler, data_cls):
        sim_dict = modeler.sim_dict
        batch_data = {task_name: run_emulated(sim) for task_name, sim in sim_dict.items()}
        port_data = SimulationDataMap(
            keys=tuple(batch_data.keys()),
            values=tuple(batch_data.values()),
        )
        self._modeler_data = data_cls(modeler=modeler, data=port_data)
        # Mirror real ComponentModelerData: has .modeler, NOT .simulation
        self.modeler = modeler

    def smatrix(self):
        return self._modeler_data.smatrix()

    def __getattr__(self, name):
        return getattr(self._modeler_data, name)


def _patch_terminal_smatrix(monkeypatch, modeler) -> None:
    monkeypatch.setattr(
        smatrix_utils,
        "port_array_inv",
        lambda matrix: np.eye(len(modeler.matrix_indices_monitor)),
    )

    def _mock_compute_F(Z_numpy, s_param_def, compute_Finv=False):
        num_freqs, num_ports, _ = Z_numpy.shape
        Z_diag = np.diagonal(Z_numpy, axis1=1, axis2=2)
        f_diag = 1.0 / (2.0 * np.sqrt(np.abs(Z_diag) + 1e-4))
        F = np.zeros_like(Z_numpy)
        for i in range(num_ports):
            F[:, i, i] = f_diag[:, i]
        if compute_Finv:
            finv_diag = 2.0 * np.sqrt(np.abs(Z_diag) + 1e-4)
            Finv = np.zeros_like(Z_numpy)
            for i in range(num_ports):
                Finv[:, i, i] = finv_diag[:, i]
            return F, Finv
        return F

    monkeypatch.setattr(
        smatrix_utils,
        "compute_F",
        _mock_compute_F,
    )
    monkeypatch.setattr(
        terminal_analysis,
        "check_port_impedance_sign",
        lambda Z_numpy: np.ndarray([]),
    )


@pytest.mark.parametrize(
    "make_modeler, data_cls, task_type, patch_smatrix",
    [
        (make_component_modeler, ModalComponentModelerData, TaskType.MODAL_CM.value, None),
        (
            lambda: make_terminal_component_modeler(planar_pec=False),
            TerminalComponentModelerData,
            TaskType.TERMINAL_CM.value,
            _patch_terminal_smatrix,
        ),
    ],
)
def test_component_modeler_cache_hit(
    monkeypatch, tmp_path, make_modeler, data_cls, task_type, patch_smatrix
):
    """Test that running a component modeler via web.run stores results in cache
    and that a second identical run gets a cache hit (no upload/start/monitor/download)."""
    modeler = make_modeler()
    fake_stub = _FakeComponentModelerStubData(modeler, data_cls)
    counters = _patch_run_pipeline(
        monkeypatch,
        task_type=task_type,
        postprocess=lambda path, lazy=False: fake_stub,
        load_simulation_fn=lambda task_id, path="simulation.json", verbose=True: modeler,
        patch_autograd=False,
    )
    if patch_smatrix is not None:
        patch_smatrix(monkeypatch, modeler)
    cache = resolve_local_cache(use_cache=True)
    cache.clear()

    out_path = tmp_path / f"{task_type.lower()}_modeler_result.hdf5"

    # First run: should make web calls and store in cache
    data = web.run(
        modeler,
        task_name=f"{task_type.lower()}_modeler_cache_test",
        path=str(out_path),
    )
    assert counters["upload"] == 1
    assert counters["start"] == 1
    assert counters["monitor"] == 1
    assert counters["download"] == 1

    s_matrix = data.smatrix()
    assert s_matrix is not None

    # Verify cache has an entry
    assert len(cache) == 1, (
        f"Expected 1 cache entry after first run, got {len(cache)}. "
        "The ComponentModelerData was not stored in cache."
    )

    # Second run: should be served from cache (no web calls)
    _reset_counters(counters)
    data2 = web.run(
        modeler,
        task_name=f"{task_type.lower()}_modeler_cache_test",
        path=str(out_path),
    )
    assert counters["upload"] == 0, "Expected no upload on cache hit"
    assert counters["start"] == 0, "Expected no start on cache hit"
    assert counters["monitor"] == 0, "Expected no monitor on cache hit"
    assert counters["download"] == 0, "Expected no download on cache hit"

    s_matrix2 = data2.smatrix()
    assert s_matrix2 is not None


@pytest.mark.parametrize(
    "make_modeler, data_cls, task_type, patch_smatrix",
    [
        (make_component_modeler, ModalComponentModelerData, TaskType.MODAL_CM.value, None),
        (
            lambda: make_terminal_component_modeler(planar_pec=False),
            TerminalComponentModelerData,
            TaskType.TERMINAL_CM.value,
            _patch_terminal_smatrix,
        ),
    ],
)
def test_component_modeler_cache_hit_uses_default_cm_path(
    monkeypatch, tmp_path, make_modeler, data_cls, task_type, patch_smatrix
):
    modeler = make_modeler()
    fake_stub = _FakeComponentModelerStubData(modeler, data_cls)
    counters = _patch_run_pipeline(
        monkeypatch,
        task_type=task_type,
        postprocess=lambda path, lazy=False: fake_stub,
        load_simulation_fn=lambda task_id, path="simulation.json", verbose=True: modeler,
        patch_autograd=False,
    )
    if patch_smatrix is not None:
        patch_smatrix(monkeypatch, modeler)
    cache = resolve_local_cache(use_cache=True)
    cache.clear()
    monkeypatch.chdir(tmp_path)

    default_path = tmp_path / "cm_data.hdf5"

    web.run(modeler, task_name=f"{task_type.lower()}_modeler_cache_default_path")
    assert default_path.exists()
    assert counters["download"] == 1

    default_path.unlink()
    _reset_counters(counters)

    web.run(modeler, task_name=f"{task_type.lower()}_modeler_cache_default_path")
    assert counters["download"] == 0
    assert default_path.exists()


@pytest.mark.parametrize(
    "make_modeler, data_cls, task_type, patch_smatrix",
    [
        (make_component_modeler, ModalComponentModelerData, TaskType.MODAL_CM.value, None),
        (
            lambda: make_terminal_component_modeler(planar_pec=False),
            TerminalComponentModelerData,
            TaskType.TERMINAL_CM.value,
            _patch_terminal_smatrix,
        ),
    ],
)
def test_public_run_component_modeler_uses_default_cm_path(
    monkeypatch, tmp_path, make_modeler, data_cls, task_type, patch_smatrix
):
    modeler = make_modeler()
    fake_stub = _FakeComponentModelerStubData(modeler, data_cls)
    counters = _patch_run_pipeline(
        monkeypatch,
        task_type=task_type,
        postprocess=lambda path, lazy=False: fake_stub,
        load_simulation_fn=lambda task_id, path="simulation.json", verbose=True: modeler,
        patch_autograd=False,
    )
    if patch_smatrix is not None:
        patch_smatrix(monkeypatch, modeler)
    cache = resolve_local_cache(use_cache=True)
    cache.clear()
    monkeypatch.chdir(tmp_path)

    default_path = tmp_path / "cm_data.hdf5"

    public_run(modeler, task_name=f"{task_type.lower()}_public_wrapper_default_path")
    assert default_path.exists()
    assert counters["download"] == 1

    default_path.unlink()
    _reset_counters(counters)

    public_run(modeler, task_name=f"{task_type.lower()}_public_wrapper_default_path")
    assert counters["download"] == 0
    assert default_path.exists()


@pytest.mark.parametrize(
    "make_modeler, data_cls, workflow_type",
    [
        (make_component_modeler, ModalComponentModelerData, TaskType.MODAL_CM.value),
        (
            lambda: make_terminal_component_modeler(planar_pec=False),
            TerminalComponentModelerData,
            TaskType.TERMINAL_CM.value,
        ),
    ],
)
def test_component_modeler_cache_stores_entry(
    monkeypatch, tmp_path, make_modeler, data_cls, workflow_type
):
    """Test that cache.store_result works for ComponentModelerData.

    The core issue: ComponentModelerData has a ``modeler`` attribute,
    not ``simulation``.  ``store_result`` uses
    ``getattr(stub_data, 'simulation', None)`` to obtain the simulation
    object for hashing.  This returns None for modeler data, so the entry
    is never stored.
    """
    modeler = make_modeler()
    cache = resolve_local_cache(use_cache=True)
    cache.clear()

    # Create fake modeler data (mirrors real ModalComponentModelerData: no .simulation)
    fake_data = _FakeComponentModelerStubData(modeler, data_cls)

    # Write a dummy artifact file (store_result needs a file to copy)
    artifact = tmp_path / f"{workflow_type.lower()}_dummy.hdf5"
    artifact.write_text("dummy-payload")

    # Try storing — this tests the core issue
    stored = cache.store_result(
        stub_data=fake_data,
        task_id="test-task-123",
        path=str(artifact),
        workflow_type=workflow_type,
    )

    assert stored, (
        "cache.store_result returned False for ComponentModelerData. "
        "This likely means getattr(stub_data, 'simulation', None) returned None "
        "because the attribute is 'modeler' not 'simulation'."
    )
    assert len(cache) == 1
