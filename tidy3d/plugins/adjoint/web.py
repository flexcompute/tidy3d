"""Adjoint-specific webapi."""
from typing import Tuple
from functools import partial

from jax import custom_vjp

from ...components.simulation import Simulation
from ...components.data.sim_data import SimulationData
from ...web.webapi import run as web_run

from .components.simulation import JaxSimulation
from .components.data.sim_data import JaxSimulationData


def _task_name_fwd(task_name: str) -> str:
    """task name for forward run as a function of the original task name."""
    return task_name + "_fwd"


def _task_name_adj(task_name: str) -> str:
    """task name for adjoint run as a function of the original task name."""
    return task_name + "_adj"


def tidy3d_run_fn(simulation: Simulation, task_name: str, **kwargs) -> SimulationData:
    """Run a regular :class:`.Simulation` after conversion from jax type."""
    return web_run(simulation=simulation, task_name=task_name, **kwargs)


@partial(custom_vjp, nondiff_argnums=tuple(range(1, 5)))
def run(
    simulation: JaxSimulation,
    task_name: str,
    folder_name: str = "default",
    path: str = "simulation_data.hdf5",
    callback_url: str = None,
) -> JaxSimulationData:
    """Mocking original web.run function, using regular tidy3d components."""

    # convert to regular tidy3d (and accounting info)
    sim_tidy3d, jax_info = simulation.to_simulation()

    # run using regular tidy3d simulation running fn
    sim_data_tidy3d = tidy3d_run_fn(
        simulation=sim_tidy3d,
        task_name=task_name,
        folder_name=folder_name,
        path=path,
        callback_url=callback_url,
    )

    # convert back to jax type and return
    return JaxSimulationData.from_sim_data(sim_data_tidy3d, jax_info=jax_info)


def run_fwd(
    simulation: JaxSimulation,
    task_name: str,
    folder_name: str,
    path: str,
    callback_url: str,
) -> Tuple[JaxSimulationData, tuple]:
    """Run forward pass and stash extra objects for the backwards pass."""

    # add the gradient monitors and run the forward simulation
    grad_mnts = simulation.get_grad_monitors()
    sim_fwd = simulation.updated_copy(**grad_mnts)
    sim_data_fwd = run(
        simulation=sim_fwd,
        task_name=_task_name_fwd(task_name),
        folder_name=folder_name,
        path=path,
        callback_url=callback_url,
    )

    # remove the gradient data from the returned version (not needed)
    sim_data_orig = sim_data_fwd.copy(update=dict(grad_data=(), simulation=simulation))
    return sim_data_orig, (sim_data_fwd,)


# pylint:disable=too-many-arguments
def run_bwd(
    task_name: str,
    folder_name: str,
    path: str,
    callback_url: str,
    res: tuple,
    sim_data_vjp: JaxSimulationData,
) -> Tuple[JaxSimulation]:
    """Run backward pass and return simulation storing vjp of the objective w.r.t. the sim."""

    # grab the forward simulation and its gradient monitor data
    (sim_data_fwd,) = res
    grad_data_fwd = sim_data_fwd.grad_data
    grad_eps_data_fwd = sim_data_fwd.grad_eps_data

    # make and run adjoint simulation
    fwidth_adj = sim_data_fwd.simulation._fwidth_adjoint  # pylint:disable=protected-access
    sim_adj = sim_data_vjp.make_adjoint_simulation(fwidth=fwidth_adj)
    sim_data_adj = run(
        simulation=sim_adj,
        task_name=_task_name_adj(task_name),
        folder_name=folder_name,
        path=path,
        callback_url=callback_url,
    )
    grad_data_adj = sim_data_adj.grad_data

    # get gradient and insert into the resulting simulation structure medium
    sim_vjp = sim_data_vjp.simulation.store_vjp(grad_data_fwd, grad_data_adj, grad_eps_data_fwd)

    return (sim_vjp,)


# register the custom forward and backward functions
run.defvjp(run_fwd, run_bwd)
