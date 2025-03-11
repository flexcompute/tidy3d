# specification for running the optimizer

import abc

import numpy as np
import pydantic.v1 as pd

from .base import InvdesBaseModel

# from .result import InverseDesignResult


class AbstractOptimizationSpec(InvdesBaseModel, abc.ABC):
    """Specification for an optimization."""

    learning_rate: pd.PositiveFloat = pd.Field(
        ...,
        title="Learning Rate",
        description="Step size for the gradient descent optimizer.",
    )

    @abc.abstractmethod
    def initial_state(self, parameters: np.ndarray) -> dict:
        """The initial state of the optimizer."""

    # def display_fn(self, result: InverseDesignResult, step_index: int) -> None:
    #     """Default display function while optimizing."""
    #     print(f"step ({step_index + 1}/{self.num_steps})")
    #     print(f"\tobjective_fn_val = {result.objective_fn_val[-1]:.3e}")
    #     print(f"\tgrad_norm = {anp.linalg.norm(result.grad[-1]):.3e}")
    #     print(f"\tpost_process_val = {result.post_process_val[-1]:.3e}")
    #     print(f"\tpenalty = {result.penalty[-1]:.3e}")


class AdamOptimizationSpec(AbstractOptimizationSpec):
    """Specification for an optimization."""

    beta1: float = pd.Field(
        0.9,
        ge=0.0,
        le=1.0,
        title="Beta 1",
        description="Beta 1 parameter in the Adam optimization method.",
    )

    beta2: float = pd.Field(
        0.999,
        ge=0.0,
        le=1.0,
        title="Beta 2",
        description="Beta 2 parameter in the Adam optimization method.",
    )

    eps: pd.PositiveFloat = pd.Field(
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


OptimizationSpecType = AdamOptimizationSpec
