import abc
import copy
import typing

import pydantic.v1 as pd

from tidy3d.exceptions import Tidy3dError

from .base import InvdesBaseModel
from .objective import AbstractObjective, rename_objective
from .optimizer import AbstractOptimizer
from .region import AbstractDesignRegion
from .result import Result
from .utils import validate_unique_strings


class TerminationSpec(InvdesBaseModel):
    @abc.abstractmethod
    def condition(
        self, objective_history, region_history, figure_of_merit_history, metadata_history
    ):
        """Termination condition to be implemented based on optimization history and current state."""


class FixedIterationTerminationSpec(TerminationSpec):
    iterations: int = pd.Field(
        ..., title="iterations", description="fixed number of iterations to run for"
    )

    def condition(
        self, objective_history, region_history, figure_of_merit_history, metadata_history
    ):
        return metadata_history["iteration"][-1] >= self.iterations


class InverseDesign(InvdesBaseModel):
    optimizers: typing.Tuple[AbstractOptimizer, ...] = pd.Field(
        ...,
        title="optimizers",
        description="optimizers to step at each iteration of the optimization",
    )

    objective_names: typing.Optional[typing.Tuple[str, ...]] = pd.Field(
        None, title="objective names", desription="optional names for objectives"
    )

    objectives: typing.Tuple[AbstractObjective, ...] = pd.Field(
        ..., title="objectives", description="objective functions to evaluate"
    )

    termination_spec: TerminationSpec = pd.Field(
        ..., title="termination", description="how to determine the optimization is finished"
    )

    @pd.root_validator(pre=True)
    def validate_objectives_and_names(cls, values):
        if values["objective_names"]:
            for obj_idx, name in enumerate(values["objective_names"]):
                values["objectives"][obj_idx] = rename_objective(
                    values["objectives"][obj_idx], name
                )

        return values

    objective_name_validator = validate_unique_strings(
        "objectives", lambda objective: objective.name
    )

    def zero_grad(self):
        for objective in self.objectives:
            objective.zero_grad()

    def evaluate_and_grad(self):
        return {objective.name: objective.evaluate_and_grad() for objective in self.objectives}

    def evaluate(self):
        return {objective.name: objective.evaluate() for objective in self.objectives}

    def access_region_by_name(self, name):
        for objective in self.objectives:
            for region in objective.regions:
                if region.name == name:
                    return region

        raise Tidy3dError("Region does not exist in any of the objectives.")

    def retrive_unique_region_copies(self) -> typing.Dict[str, AbstractDesignRegion]:
        all_region_names = [
            [design_region.name for design_region in obj.regions] for obj in self.objectives
        ]
        unique_region_names = set(sum(all_region_names, []))

        region_dict = {}
        for name in unique_region_names:
            region_dict[name] = self.access_region_by_name(name).untrack().copy(deep=True)

        return region_dict

    # verbose option?
    def continue_run(
        self,
        checkpoint: Result,
        iteration: int,
    ) -> Result:
        history = checkpoint.history
        objective_history = copy.deepcopy(history["objective_history"])
        region_history = copy.deepcopy(history["region_history"])
        optimizer_history = copy.deepcopy(history["optimizer_history"])

        figure_of_merit_history = copy.deepcopy(history["figure_of_merit_history"])
        metadata_history = copy.deepcopy(history["metadata_history"])

        def do_checkpoint(get_grad=True):
            for objective in self.objectives:
                eval_fn = objective.evaluate_and_grad if get_grad else objective.evaluate

                if objective.has_auxiliary_data:
                    fom, aux = eval_fn()
                    obj_history = (fom, aux)
                else:
                    fom = eval_fn()
                    obj_history = fom

                figure_of_merit_history[objective.name].append(fom)
                objective_history[objective.name].append(obj_history)

            optimizer_history.append(
                [optimizer.state.copy(deep=True) for optimizer in self.optimizers]
            )

            region_dict = self.retrive_unique_region_copies()
            for region_name, region in region_dict.items():
                region_history[region_name].append(region)

            metadata_history["iteration"].append(iteration)

        def zero_opt(optimizer):
            optimizer.zero_grad()

        def step_opt(optimizer):
            optimizer.step()

        def apply_to_opt(func):
            for optimizer in self.optimizers:
                func(optimizer)

        apply_to_opt(zero_opt)
        do_checkpoint()

        while not self.termination_spec.condition(
            objective_history, region_history, figure_of_merit_history, metadata_history
        ):
            apply_to_opt(step_opt)

            iteration += 1

            apply_to_opt(zero_opt)
            do_checkpoint()

        return Result(
            objective_history=objective_history,
            region_history=region_history,
            optimizer_history=optimizer_history,
            figure_of_merit_history=figure_of_merit_history,
            metadata_history=metadata_history,
        )

    def run(self) -> Result:
        objective_history = {objective.name: [] for objective in self.objectives}
        region_dict = self.retrive_unique_region_copies()
        region_history = {name: [] for name in region_dict.keys()}
        optimizer_history = []
        figure_of_merit_history = {objective.name: [] for objective in self.objectives}
        metadata_history = {"iteration": []}

        return self.continue_run(
            checkpoint=Result(
                objective_history=objective_history,
                region_history=region_history,
                optimizer_history=optimizer_history,
                figure_of_merit_history=figure_of_merit_history,
                metadata_history=metadata_history,
            ),
            iteration=0,
        )
