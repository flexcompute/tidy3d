# specification for running the optimizer
import abc
import typing
from copy import deepcopy

import dill
import pydantic.v1 as pd
import optax
import jax.numpy as jnp
import jax

import tidy3d as td
import tidy3d.plugins.adjoint as tda

from .design import InverseDesign
from .result import OptimizeResult


class AbstractOptimizer(abc.ABC, td.components.base.Tidy3dBaseModel):
    """Specification for an optimization."""

    params0: list = pd.Field(
        ...,
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

    learning_rate: pd.NonNegativeFloat = pd.Field(
        ...,
        title="Learning Rate",
        description="Step size for the gradient descent optimizer.",
    )

    num_steps: pd.PositiveInt = pd.Field(
        ...,
        title="Number of Steps",
        description="Number of steps in the gradient descent optimizer.",
    )

    def display_fn_default(
        self, result: OptimizeResult, num_steps: int, loop_index: int = None
    ) -> None:
        """Default display function while optimizing."""
        step_index = len(result.params)
        if loop_index:
            steps_done_previously = step_index - loop_index
            num_steps += steps_done_previously
        print(f"step ({step_index + 1}/{num_steps})")
        print(f"\tobjective_fn_val = {result.objective_fn_val[-1]:.3e}")
        print(f"\tgrad_norm = {jnp.linalg.norm(result.grad[-1]):.3e}")
        print(f"\tpost_process_val = {result.post_process_val[-1]:.3e}")
        print(f"\tpenalty = {result.penalty[-1]:.3e}")

    def initialize_result(self, design: InverseDesign) -> OptimizeResult:
        """Create an initially empty ``OptimizeResult`` from the starting parameters."""

        # initialize optimizer
        params0 = jnp.array(self.params0)
        optax_optimizer = self.optax_optimizer
        opt_state = optax_optimizer.init(params0)

        # initialize empty result
        return OptimizeResult(design=design, opt_state=[opt_state], params=[params0])

    def run(
        self,
        design: InverseDesign,
        post_process_fn: typing.Callable[[tda.JaxSimulationData], float],
        display_fn: typing.Callable[[typing.Dict, int], None] = None,
        **run_kwargs,
    ) -> OptimizeResult:
        """Run this inverse design problem."""

        starting_result = self.initialize_result(design=design)

        return self._run_optimizer(
            post_process_fn=post_process_fn,
            result=starting_result,
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
        display_fn: typing.Callable[[OptimizeResult, int], None] = None,
        num_steps: int = None,
        **run_kwargs,
    ) -> OptimizeResult:
        """Run optimizer for a series of steps with an initialized state. Used internally."""

        num_steps = num_steps if num_steps else self.num_steps

        # get the last state of the optimizer and the last number of params
        opt_state = result.get_final("opt_state")
        params = result.get_final("params")
        history = deepcopy(result.history)

        # use jax to grad the objective function
        objective_fn = result.design.make_objective_fn(
            post_process_fn=post_process_fn, **run_kwargs
        )
        val_and_grad_fn = jax.value_and_grad(objective_fn, has_aux=True)

        optax_optimizer = self.optax_optimizer

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

            # TODO: need to be able to load this somehow
            if self.history_save_fname:
                with open(self.history_save_fname, "wb") as f_handle:
                    dill.dump(history, f_handle)

            # display informations
            _display_fn = display_fn or self.display_fn_default
            result = OptimizeResult(design=result.design, **history)
            _display_fn(result, loop_index=loop_index, num_steps=num_steps)

            # update optimizer and parameters
            updates, opt_state = optax_optimizer.update(-grad, opt_state, params)
            params = optax.apply_updates(params, updates)
            history["params"].append(params)
            history["opt_state"].append(opt_state)

        return OptimizeResult(design=result.design, **history)


class Optimizer(AbstractOptimizer):
    """Specification for an optimization."""

    # optional kwargs passed to ``optax.adam()``

    b1: float = pd.Field(
        0.9,
        title="Beta 1",
        description="Beta 1 parameter in the Adam optimization method.",
    )

    b2: float = pd.Field(
        0.999,
        title="Beta 2",
        description="Beta 2 parameter in the Adam optimization method.",
    )

    eps: float = pd.Field(
        1e-8,
        title="Epsilon",
        description="Epsilon parameter in the Adam optimization method.",
    )

    @property
    def optax_optimizer(self) -> optax.GradientTransformationExtraArgs:
        """The optimizer used by ``optax`` corresponding to this spec."""
        return optax.adam(
            learning_rate=self.learning_rate,
            b1=self.b1,
            b2=self.b2,
            eps=self.eps,
        )

    # TODO: beta schedule
    # TODO: penalty schedule
