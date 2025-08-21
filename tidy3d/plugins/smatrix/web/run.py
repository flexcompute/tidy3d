from __future__ import annotations

import tidy3d.web as web
from tidy3d.plugins.smatrix.components.data.modeler_data import (
    ComponentModeler,
    ComponentModelerDataType,
)
from tidy3d.plugins.smatrix.components.modeler import ComponentModelerType


def run(modeler: ComponentModelerType) -> ComponentModelerDataType:
    """Run a component modeler directly through remote web API."""
    web.run(modeler)


def _run_local(modeler: ComponentModelerType) -> ComponentModelerDataType:
    """Run a component modeler locally, running batch with run_async through remote web API."""

    simulations = modeler.to_simulations()
    batch_data = web.run_async(simulations)
    return ComponentModeler.from_batch_data(batch_data)
