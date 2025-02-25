# container for specification fully defining the inverse design problem

import typing

import numpy as np
import pydantic.v1 as pd

from .base import InvdesBaseModel
from .parameter import Parameter


class DesignRegion(InvdesBaseModel):
    size: typing.Tuple = pd.Field(
        None, title="size", description="Size of the array for this design space"
    )

    parameter_initialization: typing.Callable = pd.Field(
        ..., title="parameter initialization", description="Function to initialize the parameters"
    )

    parameters: Parameter = pd.Field(
        None,
        title="parameters",
        description="Parameters to optimize for this design space.",
        init=False,
    )

    name: str = pd.Field(
        "", title="region name", description="name of region to help identify parameters"
    )

    @pd.validator("parameters")
    def validate_parameters(parameters, values):
        return Parameter(values=values["parameter_initialization"](values["size"]))

    def parameters_to_sim(self, transformations=()):
        return self.parameters_to_variables(
            intermediate_idxs=[len(transformations)], transformations=transformations
        )[0]

    def parameters_to_variables(self, intermediate_idxs=(0,), transformations=()):
        reshape_parameters = [np.reshape(self.parameters.values, self.size)]

        for idx, transformation in enumerate(transformations):
            reshape_parameters.append(transformation(reshape_parameters[idx]))

        filter_parameter_list = [reshape_parameters[idx] for idx in intermediate_idxs]
        return filter_parameter_list

    def create_region(self, value):
        return self.updated_copy(
            parameter_initialization=lambda p_size: value
        )  # parameters=Parameter(values=value))

    def insert(self, sim, value, transformations=()):
        p_size = len(self.parameters)
        new_region = self.create_region(value[0:p_size])
        new_sim = sim.insert_permittivity(
            new_region.parameters_to_sim(transformations=transformations), sim.permittivity
        )
        return new_sim

    def extract(self, grad):
        p_size = len(self.parameters)
        self.parameters.update_grad(grad[0:p_size])
        return p_size


class DesignRegion2(DesignRegion):
    def insert(self, sim, value, transformations=()):
        p_size = len(self.parameters)
        new_region = self.create_region(value[0:p_size])

        new_sim = sim.insert_permittivity(
            sim.permittivity2, new_region.parameters_to_sim(transformations=transformations)
        )
        return new_sim


DesignRegionType = typing.Union[DesignRegion, DesignRegion2]
