import abc

import pydantic.v1 as pd

from .base import InvdesBaseModel
from .parameter import AbstractParameter


class AbstractOptimizer(InvdesBaseModel, abc.ABC):
    parameters: AbstractParameter = pd.Field(
        None, title="parameters", description="Parameters we are optimizing"
    )

    def zero_grad(self):
        self.parameters.zero_grad()

    @abc.abstractmethod
    def step(self):
        """Step function for optiimzer"""


class AbstractOptimizerState:
    """abstract optimizer state"""


class GradientAscentOptimizerState(AbstractOptimizerState):
    last_step_size: float = pd.Field(
        0.0, title="last step size", description="the last amount the optimizer moved"
    )

    class Config:
        allow_mutation = True


class GradientAscentOptimizer(AbstractOptimizer):
    step_size: float = pd.Field(
        1.0, title="step size", description="How much to scale the gradient upon stepping"
    )

    state: AbstractOptimizerState = pd.Field(
        None, title="config", description="state for optimizer"
    )

    def step(self):
        if self.state:
            self.state.last_step_size = self.step_size

        print("step function")
        print(self.parameters.values + self.step_size * self.parameters.grad)
        self.parameters.update_values(
            self.parameters.values + self.step_size * self.parameters.grad
        )
