# container for everything defining the inverse design

import pydantic.v1 as pd
import typing

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
        ..., title="Initial Parameters", description="Nested list of Initial parameters."
    )

    def display_fn_default(self, **display_kwargs) -> None:
        """Default display function while optimizing."""
        step_index = display_kwargs.pop("step_index")

        print(f"step ({step_index + 1}/{self.optimizer.num_steps})")
        print(f"\tval = {display_kwargs['val']:.3e}")
        print(f"\tgrad_norm = {jnp.linalg.norm(display_kwargs['grad']):.3e}")
        print(f"\tpost_process_val = {display_kwargs['post_process_val']:.3e}")
        print(f"\tpenalty = {display_kwargs['penalty']:.3e}")

    def run(
        self,
        post_process_fn: typing.Callable[[tda.JaxSimulationData], float],
        display_fn: typing.Callable[[typing.Any, ...], None] = None,
        callback_fn: typing.Callable[[typing.Any, ...], typing.Any] = lambda **kwargs: None,
        **run_kwargs,
    ) -> OptimizeResult:
        """Run this inverse design problem."""

        # turn off verbosity by default unless otherwise specified
        if "verbose" not in run_kwargs:
            run_kwargs["verbose"] = False

        def objective_fn(params: jnp.ndarray, **post_proc_kwargs) -> float:
            """Full objective function."""

            # construct the jax simulation from the simulation + design region
            design_region_structure = self.design_region.make_structure(params)
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
                simulation=jax_sim.to_simulation(),
            )
            return objective_fn_val, aux_data

        # use jax to grad the objective function
        val_and_grad_fn = jax.value_and_grad(objective_fn, has_aux=True)

        # initialize history
        history = dict(
            params=[],
            opt_state=[],
            val=[],
            grad=[],
            penalty=[],
            post_process_val=[],
            simulation=[],
        )

        # initialize optimizer
        params = jnp.array(self.params0)
        optimizer = optax.adam(learning_rate=self.optimizer.learning_rate)
        opt_state = optimizer.init(params)

        # main optimization loop
        for step_index in range(self.optimizer.num_steps):
            # evaluate gradient
            (val, aux_data), grad = val_and_grad_fn(params)

            # strip out auxiliary data
            penalty = aux_data["penalty"]
            post_process_val = aux_data["post_process_val"]
            simulation = aux_data["simulation"]

            # save history
            history["params"].append(params)
            history["opt_state"].append(opt_state)
            history["val"].append(val)
            history["grad"].append(grad)
            history["penalty"].append(penalty)
            history["post_process_val"].append(post_process_val)
            history["simulation"].append(simulation)

            # print stuff
            display_kwargs = {key: val[-1] for key, val in history.items()}
            display_kwargs["step_index"] = step_index
            _display_fn = display_fn or self.display_fn_default
            _display_fn(**display_kwargs)

            # update optimizer and parameters
            updates, opt_state = optimizer.update(-grad, opt_state, params)
            params = optax.apply_updates(params, updates)

        return OptimizeResult(history=history)
