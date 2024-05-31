# specification for running the optimizer

import abc
import typing
from copy import deepcopy

import autograd as ag
import autograd.numpy as anp
import numpy as np
import pydantic.v1 as pd

import tidy3d as td

from .base import InvdesBaseModel
from .design import InverseDesignType
from .result import InverseDesignResult


class AbstractOptimizer(InvdesBaseModel, abc.ABC):
    """Specification for an optimization."""

    design: InverseDesignType = pd.Field(
        ...,
        title="Inverse Design Specification",
        description="Specification describing the inverse design problem we wish to optimize.",
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

    results_cache_fname: str = pd.Field(
        None,
        title="History Storage File",
        description="If specified, will save the optimization state to a local ``.pkl`` file "
        "using ``dill.dump()``. This file stores an ``InverseDesignResult`` corresponding "
        "to the latest state of the optimization. To continue this run from the file using the same"
        " optimizer instance, call ``optimizer.complete_run_from_history()``. "
        "Alternatively, the latest results can then be loaded with "
        "``td.InverseDesignResult.from_file(fname)`` and then continued using "
        "``optimizer.continue_run(result)``. ",
    )

    store_full_results: bool = pd.Field(
        True,
        title="Store Full Results",
        description="If ``True``, stores the full history for the vector fields, specifically "
        "the gradient, params, and optimizer state. For large design regions and many iterations, "
        "storing the full history of these fields can lead to large file size and memory usage. "
        "In some cases, we recommend setting this field to ``False``, which will only store the "
        "last computed state of these variables.",
    )

    @abc.abstractmethod
    def initial_state(self, parameters: np.ndarray) -> dict:
        """The initial state of the optimizer."""

    def display_fn(self, result: InverseDesignResult, step_index: int) -> None:
        """Default display function while optimizing."""
        print(f"step ({step_index + 1}/{self.num_steps})")
        print(f"\tobjective_fn_val = {result.objective_fn_val[-1]:.3e}")
        print(f"\tgrad_norm = {anp.linalg.norm(result.grad[-1]):.3e}")
        print(f"\tpost_process_val = {result.post_process_val[-1]:.3e}")
        print(f"\tpenalty = {result.penalty[-1]:.3e}")

    def _initialize_result(self, params0: anp.ndarray = None) -> InverseDesignResult:
        """Create an initially empty ``InverseDesignResult`` from the starting parameters."""

        # initialize optimizer
        if params0 is None:
            params0 = self.design.design_region.params_half
        params0 = anp.array(params0)

        state = self.initial_state(params0)

        # initialize empty result
        return InverseDesignResult(design=self.design, opt_state=[state], params=[params0])

    def run(
        self, post_process_fn: typing.Callable, params0: anp.ndarray = None
    ) -> InverseDesignResult:
        """Run this inverse design problem from an optional initial set of parameters."""
        self.design.design_region._check_params(params0)
        starting_result = self._initialize_result(params0)
        return self.continue_run(result=starting_result, post_process_fn=post_process_fn)

    def continue_run(
        self, result: InverseDesignResult, post_process_fn: typing.Callable
    ) -> InverseDesignResult:
        """Run optimizer for a series of steps with an initialized state."""

        # get the last state of the optimizer and the last parameters
        opt_state = result.get_last("opt_state")
        params = result.get_last("params")
        history = deepcopy(result.history)

        # use autograd to take gradient the objective function
        objective_fn = self.design.make_objective_fn(post_process_fn)
        val_and_grad_fn = ag.value_and_grad(objective_fn)

        # main optimization loop
        for step_index in range(self.num_steps):
            aux_data = {}
            val, grad = val_and_grad_fn(params, aux_data=aux_data)

            if anp.allclose(grad, 0.0):
                td.log.warning(
                    "All elements of the gradient are almost zero. This likely indicates "
                    "a problem with the optimization set up. This can occur if the symmetry of the "
                    "simulation and design region are preventing any data to be recorded in the "
                    "'output_monitors'. In this case, we recommend creating the initial parameters "
                    " as 'params0 = DesignRegion.params_random' and passing this to "
                    "'Optimizer.run()' to break the symmetry in the design region. "
                    "This zero gradient can also occur if the objective function return value does "
                    "not have a contribution from the input arguments. We recommend carefully "
                    "inspecting your objective function to ensure that the variables passed to the "
                    "function are contributing to the return value."
                )

            # strip out auxiliary data
            penalty = aux_data["penalty"]
            post_process_val = aux_data["post_process_val"]

            # update optimizer and parameters
            params, opt_state = self.update(parameters=params, state=opt_state, gradient=-grad)

            # cap the parameters
            params = anp.clip(params, a_min=0.0, a_max=1.0)

            # save the history of scalar values
            history["objective_fn_val"].append(val)
            history["penalty"].append(penalty)
            history["post_process_val"].append(post_process_val)

            # save the state of vector values
            for key, value in zip(("params", "opt_state", "grad"), (params, opt_state, grad)):
                if self.store_full_results:
                    history[key].append(value)
                else:
                    history[key] = [value]

            # display information
            result = InverseDesignResult(design=result.design, **history)
            self.display_fn(result, step_index=step_index)

            # save current results to file
            if self.results_cache_fname:
                result.to_file(self.results_cache_fname)

        return InverseDesignResult(design=result.design, **history)

    def continue_run_from_file(
        self, fname: str, post_process_fn: typing.Callable
    ) -> InverseDesignResult:
        """Continue the optimization run from a ``.pkl`` file with an ``InverseDesignResult``."""
        result = InverseDesignResult.from_file(fname)
        return self.continue_run(result=result, post_process_fn=post_process_fn)

    def continue_run_from_history(self, post_process_fn: typing.Callable) -> InverseDesignResult:
        """Continue the optimization run from a ``.pkl`` file with an ``InverseDesignResult``."""
        return self.continue_run_from_file(
            fname=self.results_cache_fname, post_process_fn=post_process_fn
        )


class AdamOptimizer(AbstractOptimizer):
    """Specification for an optimization."""

    beta1: float = pd.Field(
        0.9,
        title="Beta 1",
        description="Beta 1 parameter in the Adam optimization method.",
    )

    beta2: float = pd.Field(
        0.999,
        title="Beta 2",
        description="Beta 2 parameter in the Adam optimization method.",
    )

    eps: float = pd.Field(
        1e-8,
        title="Epsilon",
        description="Epsilon parameter in the Adam optimization method.",
    )

    def initial_state(self, parameters: np.ndarray) -> dict:
        """initial state of the optimizer"""
        zeros = np.zeros_like(parameters)
        return dict(m=zeros, v=zeros, t=0)

    def update(
        self, parameters: np.ndarray, gradient: np.ndarray, state: dict = None
    ) -> tuple[np.ndarray, dict]:
        if state is None:
            state = self.initial_state(parameters)

        # get state
        m = np.array(state["m"])
        v = np.array(state["v"])
        t = int(state["t"])

        # update time step
        t = t + 1

        # update moment variables
        m = self.beta1 * m + (1 - self.beta1) * gradient
        v = self.beta2 * v + (1 - self.beta2) * (gradient**2)

        # compute bias-corrected moment variables
        m_ = m / (1 - self.beta1**t)
        v_ = v / (1 - self.beta2**t)

        # update parameters and state
        parameters -= self.learning_rate * m_ / (np.sqrt(v_) + self.eps)
        state = dict(m=m, v=v, t=t)
        return parameters, state
