import abc
import typing

import numpy as np
import pydantic.v1 as pd

import tidy3d as td
from tidy3d.exceptions import Tidy3dError

from .base import InvdesBaseModel
from .initialization import InitializationSpecType
from .parameter import Parameter


class AbstractDesignRegion(InvdesBaseModel, abc.ABC):
    size: typing.Tuple = pd.Field(
        None, title="size", description="Size of the array for this design space"
    )

    parameter_initialization: InitializationSpecType = pd.Field(
        None, title="parameter initialization", description="Function to initialize the parameters"
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

    tracked: bool = pd.Field(
        True,
        title="tracked",
        description="indicator of this design region is tracked. Only"
        "tracked regions can be used in objectives.",
    )

    def untrack(self):
        return self.updated_copy(deep=True, tracked=False, parameter_initialization=None)

    def copy(self, deep: bool = True, validate: bool = True, **kwargs):
        tracked_in_update = False
        if "update" in kwargs:
            tracked_in_update = "tracked" in kwargs["update"]

        if (self.tracked and not tracked_in_update) or (
            tracked_in_update and kwargs["update"]["tracked"]
        ):
            raise Tidy3dError(
                "We can't copy a tracked region. Please call untrack() on the region first."
            )

        return super().copy(deep=deep, validate=validate, **kwargs)

    def updated_copy(self, path: str = None, deep: bool = True, validate: bool = True, **kwargs):
        tracked_in_kwargs = "tracked" in kwargs

        if (self.tracked and not tracked_in_kwargs) or (tracked_in_kwargs and kwargs["tracked"]):
            raise Tidy3dError(
                "We can't copy a tracked region. Please call untrack() on the region first."
            )

        return super().updated_copy(path=path, deep=deep, validate=validate, **kwargs)

    def __copy__(self):
        return self.copy(deep=False)

    def __deepcopy__(self, memo):
        return self.copy(deep=True)

    @pd.validator("parameters")
    def validate_parameters(parameters, values):
        if values["parameter_initialization"]:
            return Parameter(
                values=values["parameter_initialization"]
                .create_parameters(values["size"])
                .flatten()
            )

        return parameters

    def parameters_to_sim(self, transformations=()):
        return self.parameters_to_variables(
            parameters=self.parameters.values,
            intermediate_idxs=[len(transformations)],
            transformations=transformations,
        )[0]

    def parameters_to_variables(self, parameters, intermediate_idxs=(0,), transformations=()):
        reshape_parameters = [np.reshape(parameters[0 : len(self.parameters)], self.size)]

        for idx, transformation in enumerate(transformations):
            reshape_parameters.append(transformation(reshape_parameters[idx]))

        filter_parameter_list = [reshape_parameters[idx] for idx in intermediate_idxs]
        return filter_parameter_list

    def create_region(self, value):
        return self.updated_copy(
            parameters=Parameter(values=value), parameter_initialization=None, tracked=False
        )

    @abc.abstractmethod
    def insert(self, sim, value, transformations=()) -> typing.Tuple[td.Structure, ...]:
        """Create a list of structures from this region that can be imported into td.Simulation"""

    def extract(self, grad):
        p_size = len(self.parameters)
        self.parameters.update_grad(grad[0:p_size])
        return p_size


class TopologyDesignRegion(AbstractDesignRegion):
    region_center: typing.Tuple[float, ...] = pd.Field(
        ..., title="Region center", description="Center of region in simulation"
    )

    region_size: typing.Tuple[float, ...] = pd.Field(
        ..., title="Region size", description="Size of region in simulation"
    )

    def insert(self, value, transformations=()) -> typing.Tuple[td.Structure, ...]:
        p_size = len(self.parameters)
        new_region = self.create_region(value[0:p_size])

        geometry = td.Box(center=self.region_center, size=self.region_size)
        return [
            td.Structure.from_permittivity_array(
                eps_data=new_region.parameters_to_sim(transformations=transformations),
                geometry=geometry,
                name=self.name,
            )
        ]

    def extract(self, grad):
        p_size = len(self.parameters)
        self.parameters.update_grad(grad[0:p_size])
        return p_size


DesignRegionType = typing.Union[TopologyDesignRegion]
