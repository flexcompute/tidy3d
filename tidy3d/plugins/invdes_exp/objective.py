import abc
import numbers
import typing
import uuid

import autograd.numpy as np
import pydantic.v1 as pd
from autograd.tracer import getval

import tidy3d as td
import tidy3d.web as web
from tidy3d.components.types import TYPE_TAG_STR
from tidy3d.exceptions import ValidationError
from tidy3d.plugins.autograd import value_and_grad
from tidy3d.plugins.expressions.types import ExpressionType  # noqa

from .base import InvdesBaseModel
from .penalty import PenaltyType
from .region import DesignRegionType
from .transformation import TransformationType
from .utils import check_unique_list, validate_unique_strings


def create_multiobjective(objectives, combine):
    return MultiObjective(objectives=objectives, combine=combine)


def rename_objective(objective, name):
    get_type = type(objective)
    objective_dict = objective.__dict__
    objective_dict["name"] = name

    return get_type(**objective_dict)


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


class ProductSpec(CombinationSpec):
    def apply(self, x: typing.Tuple[float, ...]):
        return np.prod(x)


class MaxSpec(CombinationSpec):
    def apply(self, x: typing.Tuple[float, ...]):
        return np.max(x)


class MinSpec(CombinationSpec):
    def apply(self, x: typing.Tuple[float, ...]):
        return np.min(x)


class AbstractObjective(InvdesBaseModel):
    regions: typing.Tuple[DesignRegionType, ...] = pd.Field(
        None, title="regions", description="Design region references involved in this simulation"
    )

    parameters: dict = pd.Field(
        {},
        title="parameters",
        description="Viewpoint into parameters for optimizer",
        init=False,
    )

    name: typing.Optional[str] = pd.Field(
        "_default_objective_name_",
        title="objective name",
        description="name of objective to help identify this objective",
    )

    transformations: typing.Optional[typing.Dict[str, typing.Tuple[TransformationType, ...]]] = (
        pd.Field(
            {},
            title="transformations",
            description="dictionary of transformations to apply between the parameter and the region data for each region",
        )
    )

    scale: float = pd.Field(
        1.0, title="scale", description="scaling value to apply to objective value"
    )
    offset: float = pd.Field(
        0.0, title="offset", description="offset value to apply to objective value"
    )

    identifier: typing.Optional[str] = pd.Field(
        None,
        title="simulation identifier",
        description="string identifier for tracking objective being run",
    )

    @pd.validator("identifier")
    def validate_identifier(identifier, values):
        if not identifier:
            return str(uuid.uuid4())

        return identifier

    region_name_validator = validate_unique_strings("regions", lambda region: region.name)

    @pd.validator("parameters")
    def validate_parameters(parameters, values):
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

    @property
    def has_auxiliary_data(self):
        return False

    def combine_parameters(self):
        k = [list(region.parameters.values) for region in self.regions]
        total = []
        for k_ in k:
            total += k_
        p = np.squeeze(np.array(total))

        return p

    def evaluate(self, **kwargs):
        return self.call_objective(self.combine_parameters(**kwargs))

    def evaluate_and_grad(self, **kwargs):
        # val, grad = self.compute_grad(**kwargs)

        if self.has_auxiliary_data:
            val, grad, aux = self.compute_grad(**kwargs)
        else:
            val, grad = self.compute_grad(**kwargs)

        self.apply_grad(grad)

        if self.has_auxiliary_data:
            return val, aux
        else:
            return val

    def compute_grad(self, **kwargs):
        def f(p):
            if self.has_auxiliary_data:
                obj_val, aux_data = self.call_objective(p, **kwargs)

                return (obj_val, aux_data)

            obj_val = self.call_objective(p, **kwargs)

            return obj_val

        p = self.combine_parameters()

        val_and_grad_f = value_and_grad(f, has_aux=self.has_auxiliary_data)

        if self.has_auxiliary_data:
            (val, grad), aux = val_and_grad_f(p)
            return val, np.array(grad), aux
        else:
            val, grad = val_and_grad_f(p)
            return val, np.array(grad)

    def apply_grad(self, grad):
        p_start = 0
        for region in self.regions:
            p_start += region.extract(grad[p_start:])

    def apply_objective(self, batch_dict):
        return self.scale * self.metric(batch_dict[self.typed_identifier]) + self.offset

    @abc.abstractmethod
    def call_objective(self, parameters):
        """Evaluate objective"""

    def __add__(self, other):
        if isinstance(other, AbstractObjective):
            return create_multiobjective(objectives=(self, other), combine=SumSpec())
        elif isinstance(other, numbers.Number):
            return self.copy(update=dict(offset=self.offset + float(other)), deep=False)
        else:
            raise TypeError("Unsupported types for operation (+)")

    def __mul__(self, other):
        if isinstance(other, AbstractObjective):
            return create_multiobjective(objectives=(self, other), combine=ProductSpec())
        elif isinstance(other, numbers.Number):
            return self.copy(update=dict(scale=self.scale * float(other)), deep=False)
        else:
            raise TypeError("Unsupported types for operation (*)")

    __radd__ = __add__
    __rmul__ = __mul__

    def __sub__(self, other):
        return self + (-1.0 * other)

    def __rsub__(self, other):
        return (-1.0 * self) + other


class MultiObjective(AbstractObjective):
    objectives: typing.Tuple[AbstractObjective, ...] = pd.Field(
        (), title="objectives", description="List of objectives to combine together"
    )

    combine: CombinationSpec = pd.Field(
        None, title="combine", description="A way to combine the objective values together"
    )

    def compile_identifiers(self, identifier_list):
        for objective in self.objectives:
            objective.compile_identifiers(identifier_list)

    @pd.root_validator(pre=True)
    def validate_objectives(cls, values):
        regions = []
        for objective in values["objectives"]:
            regions += list(objective.regions)

        set_regions = []
        set_region_names = []
        for region in regions:
            if region.name not in set_region_names:
                set_regions.append(region)
                set_region_names.append(region.name)

        values["regions"] = set_regions

        return values

    @pd.validator("objectives")
    def validate_objective_identifiers(objectives, values):
        identifier_list = []

        for objective in objectives:
            objective.compile_identifiers(identifier_list)

        if not check_unique_list(identifier_list):
            raise ValidationError(
                "Objective identifiers in a MultiObjective should not conflict. If you "
                "don't specify objective identifiers, they will be made unique automatically."
            )

        return objectives

    @property
    def has_auxiliary_data(self):
        return True

    def parameter_start_location_by_region(self, parameters):
        parameter_start_location_by_region = []
        p_idx = 0
        for region in self.regions:
            parameter_start_location_by_region.append(p_idx)
            p_idx += len(region.parameters)

        parameter_start_location_by_region.append(len(parameters))

        return parameter_start_location_by_region

    def choices(self, objective, parameter_start_location_by_region, parameters):
        choices = np.zeros(len(parameters))
        for region in objective.regions:
            match_idx = 0
            for idx, match_region in enumerate(self.regions):
                if region.name == match_region.name:
                    match_idx = idx
                    break

            parameter_start = parameter_start_location_by_region[match_idx]
            parameter_end = parameter_start_location_by_region[match_idx + 1]

            choices[parameter_start:parameter_end] = 1

        return choices

    def apply_grad(self, grad):
        parameter_start_location_by_region = self.parameter_start_location_by_region(grad)

        for objective in self.objectives:
            choices = self.choices(objective, parameter_start_location_by_region, grad)
            extract_grad = grad[choices.nonzero()]

            objective.apply_grad(extract_grad)

    def compile(self, parameters, batch_dict):
        parameter_start_location_by_region = self.parameter_start_location_by_region(parameters)

        for objective in self.objectives:
            choices = self.choices(objective, parameter_start_location_by_region, parameters)

            extract_parameters = parameters[choices.nonzero()]

            objective.compile(extract_parameters, batch_dict)

    def apply_objective(self, batch_dict):
        value_by_objective = []
        aux_objective_data = []
        for objective in self.objectives:
            if objective.has_auxiliary_data:
                apply_objective, aux_data = objective.apply_objective(batch_dict)
                aux_objective_data.append(aux_data)
            else:
                apply_objective = objective.apply_objective(batch_dict)
                aux_objective_data.append(getval(apply_objective))

            value_by_objective.append(apply_objective)

        return self.scale * self.combine.apply(value_by_objective) + self.offset, aux_objective_data

    def call_objective(self, parameters, **kwargs):
        batch_dict = {}

        self.compile(parameters, batch_dict)

        simulation_dict = {}
        for key, val in batch_dict.items():
            if key[1] == "em":
                simulation_dict[key[0]] = val

        simulation_data_dict = web.run_async(simulation_dict, local_gradient=False, **kwargs)

        batch_data_dict = {}
        for key in batch_dict:
            if key[1] == "em":
                batch_data_dict[key] = simulation_data_dict[key[0]]
            else:
                batch_data_dict[key] = batch_dict[key]

        value_by_objective = []
        aux_objective_data = []
        for objective in self.objectives:
            if objective.has_auxiliary_data:
                apply_objective, aux_data = objective.apply_objective(batch_data_dict)
                aux_objective_data.append(aux_data)
            else:
                apply_objective = objective.apply_objective(batch_data_dict)
                aux_objective_data.append(getval(apply_objective))

            value_by_objective.append(apply_objective)

        return self.combine.apply(value_by_objective), aux_objective_data


class EMCustomMetric(InvdesBaseModel):
    eval_fn: typing.Callable = pd.Field(
        ..., title="eval_fn", description="custom evaluation function for penalty"
    )

    def evaluate(self, sim_data) -> float:
        """Evaluate this penalty."""
        return self.eval_fn(sim_data)

    def __call__(self, sim_data) -> float:
        return self.evaluate(sim_data)


EMMetricType = typing.Union[typing.ForwardRef("ExpressionType"), EMCustomMetric]


class EMObjective(AbstractObjective):
    metric: EMMetricType = pd.Field(
        None,
        title="objective",
        description="Computes objective function based on simulation data",
        discriminator=TYPE_TAG_STR,
    )

    base_simulation: td.Simulation = pd.Field(
        None, title="base simulation", description="Underlying simulation to be run"
    )

    @property
    def typed_identifier(self):
        return (self.identifier, "em")

    def call_objective(self, parameters, **kwargs):
        batch_dict = {}
        self.compile(parameters, batch_dict)

        if "task_name" not in kwargs:
            kwargs["task_name"] = f"{self.name}_sim"
        if "path" not in kwargs:
            kwargs["path"] = f"{self.name}_sim_data.hdf5"

        sim_data = web.run(
            batch_dict[self.typed_identifier],
            local_gradient=False,
            **kwargs,
        )

        return self.scale * self.metric(sim_data) + self.offset

    def compile_identifiers(self, identifier_list):
        return identifier_list.append(self.typed_identifier)

    def compile(self, parameters, batch_dict):
        p_start = 0
        region_structures = []
        for region in self.regions:
            p_increment = len(region.parameters)
            region_structures += region.insert(
                parameters[p_start:], self.transformations[region.name]
            )
            p_start += p_increment

        all_structures = list(self.base_simulation.structures) + region_structures
        new_sim = self.base_simulation.copy(update=dict(structures=all_structures))

        if self.typed_identifier not in batch_dict:
            batch_dict[self.typed_identifier] = new_sim

    def apply_objective(self, batch_dict):
        return self.scale * self.metric(batch_dict[self.typed_identifier]) + self.offset


class Penalty(AbstractObjective):
    metric: PenaltyType = pd.Field(
        None,
        title="metric",
        description="Computes metric function based on design regions.",
        discriminator=TYPE_TAG_STR,
    )

    @property
    def typed_identifier(self):
        return (self.identifier, "penalty")

    def call_objective(self, parameters, **kwargs):
        batch_dict = {}
        self.compile(parameters, batch_dict)

        return self.scale * self.metric(batch_dict[self.typed_identifier]) + self.offset

    def compile_identifiers(self, identifier_list):
        return identifier_list.append(self.typed_identifier)

    def compile(self, parameters, batch_dict):
        p_start = 0
        parameter_dict = {}
        for region in self.regions:
            p_increment = len(region.parameters)

            parameter_dict[region.name] = region.parameters_to_variables(
                parameters[p_start:],
                intermediate_idxs=np.arange(0, len(self.transformations[region.name]) + 1),
                transformations=self.transformations[region.name],
            )

            p_start += p_increment

        batch_dict[self.typed_identifier] = parameter_dict
