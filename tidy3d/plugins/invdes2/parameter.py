import abc
import typing

import autograd.numpy as np
import pydantic.v1 as pd

from tidy3d.exceptions import ValidationError

from .base import InvdesBaseModel


def accumulate(current_grad, new_grad):
    return current_grad + new_grad


class AbstractParameter(InvdesBaseModel, abc.ABC):
    """abstract parameter class"""


class Parameter(AbstractParameter):
    values: typing.Any = pd.Field(
        None,
        title="values",
        description="Parameter values.",
    )

    grad: typing.Any = pd.Field(
        None, title="gradient", description="Gradient of parameter values.", init=False
    )

    @pd.validator("values")
    def validate_values(cls, values):
        if not (len(np.squeeze(values).shape) == 1):
            raise ValidationError("Parameter values should be 1-dimensional array.")

        return values

    @pd.validator("grad")
    def validate_grad(grad, values):
        return np.zeros(len(values["values"]))

    def __len__(self):
        return len(self.values)

    def update_grad(self, grad, update_fn=accumulate):
        self.grad[:] = update_fn(self.grad, grad)

    def update_values(self, values):
        print("value update = " + str(values))
        self.values[:] = values

    def zero_grad(self):
        self.grad[:] = 0


class MultiParameter(AbstractParameter):
    parameters: typing.Tuple[Parameter, ...] = pd.Field(
        None, title="values", description="multiple parameter values"
    )

    def __len__(self):
        return np.sum([len(parameter) for parameter in self.parameters])

    @property
    def values(self):
        return np.array(sum([list(parameter.values) for parameter in self.parameters], []))

    @property
    def grad(self):
        return np.array(sum([list(parameter.grad) for parameter in self.parameters], []))

    def update_grad(self, grads, update_fn=accumulate):
        p_start = 0
        for parameter in self.parameters:
            p_len = len(parameter)
            parameter.update_grad(grads[p_start : p_start + p_len], update_fn)

            p_start += p_len

    def update_values(self, values):
        print("values to update = " + str(values))
        p_start = 0
        for parameter in self.parameters:
            p_len = len(parameter)
            parameter.update_values(values[p_start : p_start + p_len])

            p_start += p_len

    def zero_grad(self):
        for parameter in self.parameters:
            parameter.zero_grad()
