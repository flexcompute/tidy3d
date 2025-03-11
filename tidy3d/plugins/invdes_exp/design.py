import abc
import copy
import typing

import pydantic.v1 as pd

from tidy3d.exceptions import Tidy3dError, ValidationError

from .base import InvdesBaseModel
from .objective import AbstractObjective
from .optimizer import AbstractOptimizer
from .region import AbstractDesignRegion
from .result import Result


class TerminationSpec(InvdesBaseModel):
    @abc.abstractmethod
    def condition(self, objective_history, region_history, metadata_history):
        """Termination condition to be implemented based on optimization history and current state."""


class FixedIterationTerminationSpec(TerminationSpec):
    iterations: pd.PositiveInt = pd.Field(
        ..., title="iterations", description="fixed number of iterations to run for"
    )

    def condition(self, objective_history, region_history, metadata_history):
        return metadata_history["iteration"][-1] >= self.iterations


class InverseDesign(InvdesBaseModel):
    optimizers: typing.Tuple[AbstractOptimizer, ...] = pd.Field(
        ...,
        title="optimizers",
        description="optimizers to step at each iteration of the optimization",
    )

    objective: AbstractObjective = pd.Field(
        ..., title="objective", description="objective to evaluate"
    )

    termination: typing.Union[pd.StrictInt, TerminationSpec] = pd.Field(
        ..., title="termination", description="how to determine the optimization is finished"
    )

    @pd.validator("termination")
    def validate_termination_spec(termination, values):
        print(type(termination))
        print(termination)
        # asdf
        if isinstance(termination, int):
            if termination <= 0:
                raise ValidationError(
                    "When specifying number of iterations, it should be a strictly positive integer."
                )
            return FixedIterationTerminationSpec(iterations=termination)

        return termination

    def zero_grad(self):
        for objective in self.objectives:
            objective.zero_grad()

    def evaluate_and_grad(self):
        return self.objective.evaluate_and_grad()

    def evaluate(self):
        return self.objective.evaluate()

    def access_region_by_name(self, name):
        for region in self.objective.regions:
            if region.name == name:
                return region

        raise Tidy3dError("Region does not exist in any of the objectives.")

    def retrive_unique_region_copies(self) -> typing.Dict[str, AbstractDesignRegion]:
        all_region_names = [design_region.name for design_region in self.objective.regions]

        region_dict = {}
        for name in all_region_names:
            region_dict[name] = self.access_region_by_name(name).copy(deep=True)

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

        metadata_history = copy.deepcopy(history["metadata_history"])

        def do_checkpoint(get_grad=True):
            eval_fn = self.objective.evaluate_and_grad if get_grad else self.objective.evaluate

            if self.objective.has_auxiliary_data:
                fom, aux = eval_fn()
                obj_history = (fom, aux)
            else:
                fom = eval_fn()
                obj_history = fom

            objective_history.append(obj_history)

            optimizer_history.append([optimizer.state.copy() for optimizer in self.optimizers])

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

        while not self.termination.condition(objective_history, region_history, metadata_history):
            apply_to_opt(step_opt)

            iteration += 1

            apply_to_opt(zero_opt)
            do_checkpoint()

        return Result(
            objective_history=objective_history,
            region_history=region_history,
            optimizer_history=optimizer_history,
            metadata_history=metadata_history,
        )

    def run(self) -> Result:
        objective_history = []
        region_dict = self.retrive_unique_region_copies()
        region_history = {name: [] for name in region_dict.keys()}
        optimizer_history = []
        metadata_history = {"iteration": []}

        return self.continue_run(
            checkpoint=Result(
                objective_history=objective_history,
                region_history=region_history,
                optimizer_history=optimizer_history,
                metadata_history=metadata_history,
            ),
            iteration=0,
        )
