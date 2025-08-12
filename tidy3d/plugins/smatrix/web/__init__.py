from __future__ import annotations

from tidy3d.plugins.smatrix.component_modelers.base import AbstractComponentModeler
from tidy3d.web.api.registry import register_simulation_type
from tidy3d.web.core.types import TaskType

# Register the ComponentModeler type with a new task type key used by the server
register_simulation_type(AbstractComponentModeler, TaskType.MODE.name)  # placeholder mapping


def run(modeler: AbstractComponentModeler, task_name: str, folder_name: str = "default", **kwargs):
    """Submit a ComponentModeler by delegating to web.run.

    Note: this uses an existing TaskType mapping (MODE) as a placeholder. Replace
    with a dedicated task type once server-side support exists.
    """
    from tidy3d.web.api import webapi as web

    return web.run(
        simulation=modeler,
        task_name=task_name,
        folder_name=folder_name,
        simulation_type="smatrix",
        **kwargs,
    )
