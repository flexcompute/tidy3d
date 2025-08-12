from __future__ import annotations

import pytest


def test_import_smatrix_then_registry_has_builtins_and_plugin():
    # Importing smatrix.web will import web transitively (via container) and should leave
    # both built-ins and plugin types registered in the registry.
    import tidy3d.plugins.smatrix.web  # noqa: F401
    from tidy3d.components.data.data_array import DataArray
    from tidy3d.plugins.smatrix.component_modelers.base import AbstractComponentModeler
    from tidy3d.web.api.container import BatchData
    from tidy3d.web.api.registry import (
        get_registered_sim_loader,
        get_task_type_for_instance,
    )

    class DummyModeler(AbstractComponentModeler):
        @property
        def matrix_indices_monitor(self):
            return ()

        def _construct_smatrix(self, batch_data: BatchData) -> DataArray:  # type: ignore[override]
            return DataArray()

        def _internal_construct_smatrix(self, batch_data: BatchData) -> DataArray:  # type: ignore[override]
            return DataArray()

    assert get_registered_sim_loader("Simulation") is not None
    assert get_task_type_for_instance(DummyModeler.construct()) is not None


def test_double_imports_do_not_break_registry():
    # Import web twice
    import tidy3d.web
    from tidy3d.web.api.registry import get_registered_sim_loader

    assert get_registered_sim_loader("Simulation") is not None

    # Import smatrix twice
    import tidy3d.plugins.smatrix.web  # noqa: F401  # noqa: F401
    from tidy3d.components.data.data_array import DataArray
    from tidy3d.plugins.smatrix.component_modelers.base import AbstractComponentModeler
    from tidy3d.web.api.container import BatchData
    from tidy3d.web.api.registry import get_task_type_for_instance

    class DummyModeler(AbstractComponentModeler):
        @property
        def matrix_indices_monitor(self):
            return ()

        def _construct_smatrix(self, batch_data: BatchData) -> DataArray:  # type: ignore[override]
            return DataArray()

        def _internal_construct_smatrix(self, batch_data: BatchData) -> DataArray:  # type: ignore[override]
            return DataArray()

    assert get_task_type_for_instance(DummyModeler.construct()) is not None


@pytest.mark.usefixtures("monkeypatch")
def test_registry_populated_after_normal_import(monkeypatch):
    # Normal import order within this process should populate built-ins and plugin mapping
    import tidy3d.plugins.smatrix.web as smatrix_web
    import tidy3d.web  # noqa: F401  (side effect: registers built-ins)
    from tidy3d.components.data.data_array import DataArray
    from tidy3d.plugins.smatrix.component_modelers.base import AbstractComponentModeler
    from tidy3d.web.api.container import BatchData
    from tidy3d.web.api.registry import get_task_type_for_instance

    class DummyModeler(AbstractComponentModeler):
        @property
        def matrix_indices_monitor(self):
            return ()

        # Provide concrete no-op implementations to satisfy ABC
        def _construct_smatrix(self, batch_data: BatchData) -> DataArray:  # type: ignore[override]
            return DataArray()

        def _internal_construct_smatrix(self, batch_data: BatchData) -> DataArray:  # type: ignore[override]
            return DataArray()

    assert get_task_type_for_instance(DummyModeler.construct()) is not None

    # Also ensure smatrix.web.run delegates to webapi.run without raising
    import tidy3d.web.api.webapi as webapi

    called = {"ok": False}

    def _fake_run(**kwargs):
        called["ok"] = True
        return "OK"

    monkeypatch.setattr(webapi, "run", _fake_run, raising=True)
    out = smatrix_web.run(DummyModeler.construct(), task_name="t")
    assert out == "OK" and called["ok"]
