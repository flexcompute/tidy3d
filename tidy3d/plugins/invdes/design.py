# container for everything defining the inverse design

import pydantic.v1 as pd
import typing
import dill
from copy import deepcopy

import jax.numpy as jnp
import jax
import optax

import tidy3d as td
from tidy3d.plugins.adjoint.components.simulation import OutputMonitorType, JaxInfo

import tidy3d.plugins.adjoint as tda

from .design_region import DesignRegion
from .optimizer import Optimizer
from .result import OptimizeResult


class InverseDesign(td.components.base.Tidy3dBaseModel):
    """Container for an inverse design problem."""

    simulation: td.Simulation = pd.Field(
        ...,
        title="Base Simulation",
        description="Simulation without the design regions or monitors used in the objective fn.",
    )

    design_region: DesignRegion = pd.Field(
        ...,
        title="Design Region",
        description="Region within which we will optimize the simulation.",
    )

    output_monitors: OutputMonitorType = pd.Field(
        (),
        title="Output Monitors",
        description="Tuple of monitors whose data the differentiable output depends on.",
    )

    optimizer: Optimizer = pd.Field(
        ...,
        title="Optimizer",
        description="Specification used to define the gradient-based optimization procedure.",
    )

    params0: list = pd.Field(
        None,
        title="Initial Parameters",
        description="Nested list of Initial parameters. Only "
        "optional when running an ``InverseDesign.continue_run()`` "
        "from a previous ``OptimizeResult``. ",
    )

    history_save_fname: str = pd.Field(
        None,
        title="History Storage File",
        description="If specified, will save the optimization state to a local ``.pkl`` file "
        "using ``dill.dump()``.",
    )

    def display_fn_default(self, history: dict, num_steps: int, loop_index: int = None) -> None:
        """Default display function while optimizing."""
        step_index = len(history["params"])
        if loop_index:
            steps_done_previously = step_index - loop_index
            num_steps += steps_done_previously
        print(f"step ({step_index + 1}/{num_steps})")
        print(f"\tobjective_fn_val = {history['objective_fn_val'][-1]:.3e}")
        print(f"\tgrad_norm = {jnp.linalg.norm(history['grad'][-1]):.3e}")
        print(f"\tpost_process_val = {history['post_process_val'][-1]:.3e}")
        print(f"\tpenalty = {history['penalty'][-1]:.3e}")

    def make_objective_fn(
        self, post_process_fn: typing.Callable[[tda.JaxSimulationData], float], **run_kwargs
    ) -> typing.Callable[[jnp.ndarray], float]:
        """construct the objective function for this ``InverseDesign`` object."""

        # turn off verbosity by default unless otherwise specified
        if "verbose" not in run_kwargs:
            run_kwargs["verbose"] = False

        def objective_fn(params: jnp.ndarray, **post_proc_kwargs) -> float:
            """Full objective function."""

            # TODO: I dont think post_proc_kwargs is ever exposed to the user

            # construct the jax simulation from the simulation + design region
            design_region_structure = self.design_region.make_structure(params)

            # TODO: do we want to add mesh override structure if the pixels are large / low res?
            mesh_override_structure = td.MeshOverrideStructure(
                geometry=design_region_structure.geometry,
                dl=self.design_region.step_sizes,
                enforce=True,
            )
            grid_spec = self.simulation.grid_spec.updated_copy(
                override_structures=list(self.simulation.grid_spec.override_structures)
                + [mesh_override_structure]
            )
            jax_info = JaxInfo(
                num_input_structures=0,
                num_output_monitors=0,
                num_grad_monitors=0,
                num_grad_eps_monitors=0,
            )

            jax_sim = tda.JaxSimulation.from_simulation(
                self.simulation,
                jax_info=jax_info,
            )

            jax_sim = jax_sim.updated_copy(
                input_structures=[design_region_structure],
                output_monitors=self.output_monitors,
                grid_spec=grid_spec,
            )

            # run the jax simulation
            jax_sim_data = tda.web.run(jax_sim, **run_kwargs)

            # construct objective function values
            post_process_val = post_process_fn(jax_sim_data, **post_proc_kwargs)
            penalty_value = self.design_region.penalty_value(params)
            objective_fn_val = post_process_val - penalty_value

            # return objective value and auxiliary data
            aux_data = dict(
                penalty=penalty_value,
                post_process_val=post_process_val,
                simulation=jax_sim.to_simulation()[0],
            )
            return objective_fn_val, aux_data

        return objective_fn

    def run(
        self,
        post_process_fn: typing.Callable[[tda.JaxSimulationData], float],
        display_fn: typing.Callable[[typing.Dict, int], None] = None,
        **run_kwargs,
    ) -> OptimizeResult:
        """Run this inverse design problem."""

        # initialize optimizer
        params = jnp.array(self.params0)
        optax_optimizer = self.optimizer.optax_optimizer
        opt_state = optax_optimizer.init(params)

        # initialize history
        history = dict(
            params=[params],
            opt_state=[opt_state],
            objective_fn_val=[],
            grad=[],
            penalty=[],
            post_process_val=[],
            simulation=[],
        )

        result = OptimizeResult(history=history)

        return self._run_optimizer(
            post_process_fn=post_process_fn,
            result=result,
            display_fn=display_fn,
            **run_kwargs,
        )

    def continue_run(
        self,
        result: OptimizeResult,
        post_process_fn: typing.Callable[[tda.JaxSimulationData], float],
        display_fn: typing.Callable[[typing.Dict, int], None] = None,
        num_steps: int = None,
        suppress_params_warning: bool = False,
        **run_kwargs,
    ) -> OptimizeResult:
        """Continue running an ``OptimizeResult``."""

        if not suppress_params_warning and self.params0 is not None:
            td.log.warning(
                "'InverseDesign.params0' is defined, note that this initial condition will be "
                "ignored when continuing an optimization run. You can suppress this warning by "
                "updating 'params0=None' or by passing 'suppress_params_warning=True' to "
                "'InverseDesign.continue_run'. "
            )

        return self._run_optimizer(
            post_process_fn=post_process_fn,
            result=result,
            display_fn=display_fn,
            num_steps=num_steps,
            **run_kwargs,
        )

    def _run_optimizer(
        self,
        post_process_fn: typing.Callable[[tda.JaxSimulationData], float],
        result: OptimizeResult,
        display_fn: typing.Callable[[typing.Dict, int], None] = None,
        num_steps: int = None,
        **run_kwargs,
    ) -> OptimizeResult:
        """Run optimizer for a series of steps with an initialized state. Used internally."""

        num_steps = num_steps if num_steps else self.optimizer.num_steps

        opt_state = result.get_final("opt_state")
        params = result.get_final("params")
        history = deepcopy(result.history)

        # use jax to grad the objective function
        objective_fn = self.make_objective_fn(post_process_fn=post_process_fn, **run_kwargs)
        val_and_grad_fn = jax.value_and_grad(objective_fn, has_aux=True)

        optax_optimizer = self.optimizer.optax_optimizer

        # main optimization loop
        for loop_index in range(num_steps):
            # evaluate gradient
            (val, aux_data), grad = val_and_grad_fn(params)

            # strip out auxiliary data
            penalty = aux_data["penalty"]
            post_process_val = aux_data["post_process_val"]
            simulation = aux_data["simulation"]

            # save history
            history["objective_fn_val"].append(val)
            history["grad"].append(grad)
            history["penalty"].append(penalty)
            history["post_process_val"].append(post_process_val)
            history["simulation"].append(simulation)

            if self.history_save_fname:
                with open(self.history_save_fname, "wb") as f_handle:
                    dill.dump(history, f_handle)

            # display informations
            _display_fn = display_fn or self.display_fn_default
            _display_fn(history, loop_index=loop_index, num_steps=num_steps)

            # update optimizer and parameters
            updates, opt_state = optax_optimizer.update(-grad, opt_state, params)
            params = optax.apply_updates(params, updates)
            history["params"].append(params)
            history["opt_state"].append(opt_state)

        return OptimizeResult(history=history)
