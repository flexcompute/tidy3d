import abc
import typing

import autograd as ag
import autograd.numpy as np
import pydantic.v1 as pd

from tidy3d.exceptions import Tidy3dError, ValidationError

from .base import InvdesBaseModel
from .dummy_simulation import DummySimulation
from .region import DesignRegionType
from .utils import validate_unique_names


# is this the right place for these to be? or their own file?
class CombinationSpec(InvdesBaseModel, abc.ABC):
    placeholder: float = pd.Field(
        1.0,
        title="placeholder",
        description="placeholder so that pydantic validation does not fail",
    )

    @abc.abstractmethod
    def apply(self, x: typing.Tuple[float, ...]):
        """Apply combination"""


class SumSpec(CombinationSpec):
    def apply(self, x: typing.Tuple[float, ...]):
        return np.sum(x)


class MaxSpec(CombinationSpec):
    def apply(self, x: typing.Tuple[float, ...]):
        return np.max(x)


class AbstractObjective(InvdesBaseModel):
    regions: typing.Tuple[DesignRegionType, ...] = pd.Field(
        None, title="regions", description="Design region references involved in this simulation"
    )

    parameter_dict: dict = pd.Field(
        {},
        title="parameter dictionary",
        description="Viewpoint into parameters for optimizer",
        init=False,
    )

    name: str = pd.Field(
        ..., title="region name", description="name of region to help identify parameters"
    )

    transformations: typing.Dict[str, typing.Optional[typing.Tuple[typing.Callable, ...]]] = (
        pd.Field(
            {},
            title="transformations",
            description="dictionary of transformations to apply between the parameter and the region data for each region",
        )
    )

    region_name_validator = validate_unique_names("regions")

    @pd.validator("parameter_dict")
    def validate_parameter_dict(parameter_dict, values):
        # print(values)
        regions = values["regions"]
        region_names = (region.name for region in regions)

        return dict(zip(region_names, (region.parameters for region in regions)))

    @pd.validator("transformations")
    def validate_transformations(transformations, values):
        fill_in_default_transformations = {}
        regions = values["regions"]
        region_names = [region.name for region in regions]

        for key in transformations.keys():
            if key not in region_names:
                raise ValidationError(
                    "Unexpected transformation key not associated with any regions!"
                )

        for region in regions:
            if region.name in transformations:
                fill_in_default_transformations[region.name] = transformations[region.name]
            else:
                fill_in_default_transformations[region.name] = ()

        return fill_in_default_transformations

    def parameters(self):
        return self.parameter_dict

    def combine_parameters(self):
        k = [list(region.parameters.values) for region in self.regions]
        total = []
        for k_ in k:
            total += k_
        p = np.squeeze(np.array(total))

        return p

    def evaluate(self):
        return self.call_objective(self.combine_parameters())

    def evaluate_and_grad(self):
        val, grad = self.compute_grad()

        self.apply_grad(grad)

        return val

    def compute_grad(self):
        def f(p):
            return self.call_objective(p)

        p = self.combine_parameters()

        val_and_grad_f = ag.value_and_grad(f)
        val, grad = val_and_grad_f(p)

        return val, np.array(grad)

    def apply_grad(self, grad):
        p_start = 0
        for region in self.regions:
            p_start += region.extract(grad[p_start:])

    def accumulate(self, grad_g, grads):
        return grad_g * grads
        # grad_per_objective = []

        # for idx in range(0, len(grad_g)):
        #     grad_per_objective.append(grad_g[idx] * np.array(grads[idx]))

    @abc.abstractmethod
    def call_objective(self, parameters):
        """Evaluate objective"""

    # potential name conflicts.. maybe we have a reservered name for these and we don't allow certain
    # keywords in objective names to prevent conflict? or we can have the option to have unnamed objectives
    # but then you can't get a dictionary for them back? or we can keep the objective tracking separate and allow
    # the user to assign names to objectives they insert into the function and then we don't need names? because
    # they would have to know what the names are anyway to use this in their custom objective function
    def __add__(self, other):
        return MultiObjective(
            objectives=(self, other), combine=SumSpec(), name=f"{self.name}_plus_{other.name}"
        )


class MultiObjective(AbstractObjective):
    objectives: typing.Tuple[AbstractObjective, ...] = pd.Field(
        (), title="objectives", description="List of objectives to combine together"
    )

    combine: CombinationSpec = pd.Field(
        None, title="combine", description="A way to combine the objective values together"
    )

    def __init__(
        self,
        objectives: typing.Tuple[AbstractObjective, ...] = (),
        combine: CombinationSpec = None,
        **kwargs,
    ):
        regions = []
        for objective in objectives:
            regions += list(objective.regions)

        set_regions = []
        set_region_names = []
        for region in regions:
            if region.name not in set_region_names:
                set_regions.append(region)
                set_region_names.append(region.name)

        super().__init__(objectives=objectives, combine=combine, regions=set_regions, **kwargs)

    def evaluate(self):
        # return self.combine.apply(list(objective.evaluate() for objective in self.objectives))
        return self.combine.apply([objective.evaluate() for objective in self.objectives])

    # def evaluate_and_grad(self):
    #     def g(objs):
    #         return self.combine.apply(objs)

    #     val_and_grad_g = ag.value_and_grad(g)

    #     vals, grads = zip(*list(objective.compute_grad() for objective in self.objectives))

    #     val_g, grad_g = val_and_grad_g(vals)

    #     for idx, objective in enumerate(self.objectives):
    #         objective.apply_grad(grad_g[idx] * grads[idx])

    #
    # f( g( a, b ), h( c, e ) )
    #
    # df/dx = df/dg (dg/da da/dx + dg/db db/dx) + df/dh (dh/dc dc/dx + dh/de de/dx)
    #

    def accumulate(self, grad_g, grads):
        grad_per_objective = []

        for idx in range(0, len(grad_g)):
            grad_per_objective.append(grad_g[idx] * np.array(grads[idx]))

        return grad_per_objective

    def compute_grad(self):
        def g(objs):
            return self.combine.apply(objs)

        val_and_grad_g = ag.value_and_grad(g)

        # vals, grads = zip(*list(objective.compute_grad() for objective in self.objectives))
        vals, grads = zip(*[objective.compute_grad() for objective in self.objectives])

        val_g, grad_g = val_and_grad_g(vals)

        grad_per_objective = []
        for idx in range(0, len(grad_g)):
            grad_per_objective.append(self.objectives[idx].accumulate(grad_g[idx], grads[idx]))

        return vals, grad_per_objective

    def apply_grad(self, grads):
        for idx, grad in enumerate(grads):
            self.objectives[idx].apply_grad(grad)

    def call_objective(self, parameters):
        raise Tidy3dError("Unexpected call to call objective in multiobjective")


class EMObjective(AbstractObjective):
    objective: typing.Callable = pd.Field(
        None, title="objective", description="Computes objective function based on simulation data"
    )

    base_simulation: DummySimulation = pd.Field(
        None, title="base simulation", description="Underlying simulation to be run"
    )

    def call_objective(self, parameters):
        p_start = 0
        for idx, region in enumerate(self.regions):
            p_increment = len(region.parameters)
            if idx == 0:
                new_sim = region.insert(
                    self.base_simulation, parameters[p_start:], self.transformations[region.name]
                )
            else:
                new_sim = region.insert(
                    new_sim, parameters[p_start:], self.transformations[region.name]
                )
            p_start += p_increment

        sim_data = new_sim.run()

        return self.objective(sim_data)


class PenaltyObjective(AbstractObjective):
    objective: typing.Callable = pd.Field(
        None, title="objective", description="Computes objective function based on design region"
    )

    def call_objective(self, parameters):
        p_start = 0
        # new_regions = []
        parameter_dict = {}
        for region in self.regions:
            p_increment = len(region.parameters)

            parameter_dict[region.name] = region.parameters_to_variables(
                intermediate_idxs=np.arange(0, len(self.transformations[region.name]) + 1),
                transformations=self.transformations[region.name],
            )

            p_start += p_increment

        return self.objective(parameter_dict)
