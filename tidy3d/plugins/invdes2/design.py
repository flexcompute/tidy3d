import abc
import copy
import typing

import pydantic.v1 as pd

from .base import InvdesBaseModel
from .objective import AbstractObjective
from .optimizer import AbstractOptimizer
from .result import Result
from .utils import validate_unique_names


class TerminationSpec(InvdesBaseModel):
    @abc.abstractmethod
    def condition(self, objective_history, figure_of_merit_history, metadata_history):
        """Termination condition to be implemented based on optimization history and current state."""


class FixedIterationTerminationSpec(TerminationSpec):
    iterations: int = pd.Field(
        ..., title="iterations", description="fixed number of iterations to run for"
    )

    def condition(self, objective_history, figure_of_merit_history, metadata_history):
        return metadata_history["iteration"][-1] >= self.iterations


class InverseDesign(InvdesBaseModel):
    #
    # we may want to allow custom control flow at some point but this class is for standard design flows
    # also may have other design tools come in in custom control flow
    # we may want to warn if these optimizers are working on a bunch of the same underlying parameters (?)
    #
    optimizers: typing.Tuple[AbstractOptimizer, ...] = pd.Field(
        ...,
        title="optimizers",
        description="optimizers to step at each iteration of the optimization",
    )

    objectives: typing.Tuple[AbstractObjective, ...] = pd.Field(
        ..., title="objectives", description="objective functions to evaluate"
    )

    termination_spec: TerminationSpec = pd.Field(
        ..., title="termination", description="how to determine the optimization is finished"
    )

    objective_name_validator = validate_unique_names("objectives")

    def zero_grad(self):
        for objective in self.objectives:
            objective.zero_grad()

    def evaluate_and_grad(self):
        return {objective.name: objective.evaluate_and_grad() for objective in self.objectives}

    def evaluate(self):
        return {objective.name: objective.evaluate() for objective in self.objectives}

    def continue_run(
        self,
        checkpoint: Result,
        iteration: int,
    ) -> Result:
        # we may want a way to confirm that the optimization you are re-running belongs to this history trace
        # also we will want to make this all very easy for people to run optimizations and possibly many of them
        # and then easily analyze the data and results

        history = checkpoint.history
        objective_history = copy.deepcopy(history["objective_history"])

        figure_of_merit_history = copy.deepcopy(history["figure_of_merit_history"])
        metadata_history = copy.deepcopy(history["metadata_history"])

        # all_region_names = [
        #     list(design_region.name for design_region in obj.regions) for obj in self.objectives
        #     # [design_region.name for design_region in obj.regions]
        #     for obj in self.objectives
        # ]
        # unique_region_names = set(sum(all_region_names, []))

        #
        # the MultiParameter allows you to combine multiple parameters into the same optimizer
        #

        # def find(name, parameter_dicts):
        #     for parameter_dict in parameter_dicts:
        #         if name in parameter_dict.keys():
        #             return parameter_dict[name]

        # all_parameters = [obj.parameters() for obj in self.objectives]
        # extract_parameters = [find(name, all_parameters) for name in unique_region_names]

        # opt_parameters = MultiParameter(parameters=extract_parameters)

        # optimizer = OptimizerMethods[self.optimizer_spec.method](
        #     parameters=opt_parameters,
        #     state=GradientAscentOptimizerState(),
        #     **self.optimizer_spec.config,
        # )

        def do_checkpoint(get_grad=True):
            for objective in self.objectives:
                objective_history[objective.name].append(copy.deepcopy(objective))

            if get_grad:
                figures_of_merit = self.evaluate_and_grad()
            else:
                figures_of_merit = self.evaluate()

            for key, val in figures_of_merit.items():
                figure_of_merit_history[key].append(val)

            metadata_history["iteration"].append(iteration)

        def zero_opt(optimizer):
            optimizer.zero_grad()

        def step_opt(optimizer):
            optimizer.step()

        def apply_to_opt(func):
            for optimizer in self.optimizers:
                func(optimizer)

        # optimizer.zero_grad()
        apply_to_opt(zero_opt)
        do_checkpoint()

        while not self.termination_spec.condition(
            objective_history, figure_of_merit_history, metadata_history
        ):
            apply_to_opt(step_opt)

            iteration += 1

            apply_to_opt(zero_opt)
            do_checkpoint()

        return Result(
            objective_history=objective_history,
            figure_of_merit_history=figure_of_merit_history,
            metadata_history=metadata_history,
        )

    def run(self) -> Result:
        objective_history = {objective.name: [] for objective in self.objectives}
        figure_of_merit_history = {objective.name: [] for objective in self.objectives}
        metadata_history = {"iteration": []}

        return self.continue_run(
            checkpoint=Result(
                objective_history=objective_history,
                figure_of_merit_history=figure_of_merit_history,
                metadata_history=metadata_history,
            ),
            iteration=0,
        )
