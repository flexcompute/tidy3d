from __future__ import annotations

import responses

from tidy3d.web.core.environment import Env


@responses.activate
def test_smatrix_registry_and_run_smoke(monkeypatch):
    # Lazily import after environment is active
    Env.dev.active()

    # Import smatrix web to trigger registration
    from tidy3d.plugins.smatrix.component_modelers.base import AbstractComponentModeler
    from tidy3d.plugins.smatrix.web import run  # noqa: F401
    from tidy3d.web.api.registry import get_task_type_for_instance

    # Create a minimal fake modeler instance by subclassing (to avoid setup complexity)
    class FakeModeler(AbstractComponentModeler):
        @property
        def matrix_indices_monitor(self):
            return ()

        @property
        def sim_dict(self):
            return {}

        def _construct_smatrix(self, batch_data=None):
            return None

        def _internal_construct_smatrix(self, batch_data=None):
            return None

    # Instantiate without validation; we don't need valid internals for registry mapping
    modeler = FakeModeler.construct()

    # Ensure the registry is aware of this class mapping
    assert get_task_type_for_instance(modeler) is not None
