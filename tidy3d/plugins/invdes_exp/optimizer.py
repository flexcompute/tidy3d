import abc
import typing

import numpy as np
import pydantic.v1 as pd

from .base import InvdesBaseModel
from .parameter import AbstractParameter


class AbstractOptimizerState(pd.BaseModel, abc.ABC):
    """abstract optimizer state"""

    class Config:
        allow_mutation = True


class AbstractOptimizer(InvdesBaseModel, abc.ABC):
    parameters: AbstractParameter = pd.Field(
        None, title="parameters", description="Parameters we are optimizing"
    )

    state: AbstractOptimizerState = pd.Field(
        None, title="optimizer state", description="State for restarting optimizer"
    )

    learning_rate: float = pd.Field(
        1.0, title="learning rate", description="learning rate for optimizer"
    )

    bounds: typing.Tuple[float, float] = pd.Field(
        None, title="bounds", description="bounds for optimization variable"
    )

    @pd.validator("bounds")
    def validate_bounds(bounds, values):
        return sorted(bounds)

    def apply_bounds(self, updated_values):
        if self.bounds is not None:
            updated_values = np.maximum(np.minimum(self.bounds[1], updated_values), self.bounds[0])

        return updated_values

    def zero_grad(self):
        self.parameters.zero_grad()

    @abc.abstractmethod
    def step(self):
        """Step function for optiimzer"""


class GradientAscentOptimizer(AbstractOptimizer):
    def step(self):
        updated_values = self.parameters.values + self.learning_rate * self.parameters.grad
        self.parameters.update_values(self.apply_bounds(updated_values))


class AdamOptimizerState(AbstractOptimizerState):
    # m: NDArray = pd.Field(..., title="m")

    # t: NDArray = pd.Field(..., title="t")

    # v: NDArray = pd.Field(..., title="v")

    m: float = pd.Field(..., title="m")

    t: int = pd.Field(..., title="t")

    v: float = pd.Field(..., title="v")

    # @pd.validator("t")
    # def validate_t(t, values):
    #     if not (t.dtype == np.long):

    def create_initial_state(parameters):
        """initial state of the optimizer"""
        zeros = np.zeros_like(parameters)
        return AdamOptimizerState(m=zeros, v=zeros, t=0)


class AdamOptimizer(AbstractOptimizer):
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

    state: AdamOptimizerState = pd.Field(
        None, title="state", description="state for Adam optimizer"
    )

    bounds: typing.Tuple[float, float] = pd.Field(
        None, title="bounds", description="bounds for optimization variable"
    )

    @pd.validator("state")
    def validate_state(state, values):
        if state is None:
            return AdamOptimizerState.create_initial_state(values["parameters"].values)

        return state

    def step(self):
        # get state
        m = np.array(self.state.m)
        v = np.array(self.state.v)
        t = int(self.state.t)

        # update time step
        t = t + 1

        gradient = -self.parameters.grad
        # update moment variables
        m = self.beta1 * m + (1 - self.beta1) * gradient
        v = self.beta2 * v + (1 - self.beta2) * (gradient**2)

        # compute bias-corrected moment variables
        m_ = m / (1 - self.beta1**t)
        v_ = v / (1 - self.beta2**t)

        # update parameters and state
        updated_values = self.parameters.values - self.learning_rate * m_ / (np.sqrt(v_) + self.eps)

        self.parameters.update_values(self.apply_bounds(updated_values))

        self.state.__dict__.update(m=m, v=v, t=t)
