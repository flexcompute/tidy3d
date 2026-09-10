"""Public Tidy3D autograd file preparation at the translator edge."""

from __future__ import annotations


def prepare_forward_task_file(task_file: str, tracer_keys_file: str) -> str:
    """Add public adjoint monitors to a public task file before schema conversion."""

    from tidy3d import Simulation
    from tidy3d.components.autograd.field_map import TracerKeys

    sim = Simulation.from_file(task_file)
    sim_fields_keys = TracerKeys.from_file(tracer_keys_file).keys
    sim = sim._with_adjoint_monitors(sim_fields_keys)
    sim.to_file(task_file)
    return task_file
