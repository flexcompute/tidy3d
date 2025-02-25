import abc
import copy
import inspect
import typing

import autograd as ag
import autograd.numpy as np
import pydantic.v1 as pd

from tidy3d.exceptions import Tidy3dError, ValidationError

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
        # print(values)
        # asdf
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
        # return np.sum(list(len(parameter) for parameter in self.parameters))
        return np.sum([len(parameter) for parameter in self.parameters])

    @property
    def values(self):
        # return np.array(sum(list(list(parameter.values) for parameter in self.parameters), []))
        return np.array(sum([list(parameter.values) for parameter in self.parameters], []))

    @property
    def grad(self):
        # return np.array(sum(list(list(parameter.grad) for parameter in self.parameters), []))
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


class DesignRegion(InvdesBaseModel):
    #
    # maybe transformation spec is up here instead of in the region so we can re-use a region with a different set of transformations
    #

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
        # reshape_parameters = np.reshape(self.parameters.values, self.size)

        # for transformation in transformations:
        #     reshape_parameters = transformation(reshape_parameters)

        # return reshape_parameters  # np.reshape(self.parameters.values, self.size)

    def parameters_to_variables(self, intermediate_idxs=tuple(0), transformations=()):
        reshape_parameters = [np.reshape(self.parameters.values, self.size)]

        for idx, transformation in enumerate(transformations):
            # reshape_parameters = transformation(reshape_parameters)
            reshape_parameters.append(transformation(reshape_parameters[idx]))

        filter_parameter_list = [reshape_parameters[idx] for idx in intermediate_idxs]
        return filter_parameter_list

        # return reshape_parameters  # np.reshape(self.parameters.values, self.size)

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


class DummySimulation(InvdesBaseModel):
    permittivity: typing.Any = pd.Field(
        None, title="permittivity", description="Simulation permittivity"
    )

    permittivity2: typing.Any = pd.Field(
        None, title="permittivity", description="Simulation permittivity"
    )

    def insert_permittivity(self, permittivity, permittivity2):
        return DummySimulation(permittivity=permittivity, permittivity2=permittivity2)

    def run(self):
        if self.permittivity2 is not None:
            return {"E": self.permittivity**2, "H": self.permittivity2**2}
        else:
            return {"E": self.permittivity**2, "H": self.permittivity**1.5}


def validate_unique_names(prop_name):
    @pd.validator(prop_name, allow_reuse=True)
    def validate_names(named_objects, values):
        # names = list(named_object.name for named_object in named_objects)
        names = [named_object.name for named_object in named_objects]
        if not (len(names) == len(set(names))):
            raise ValidationError(f"{prop_name} name conflicts")

        return named_objects

    return validate_names


class AbstractObjective(InvdesBaseModel):
    regions: typing.Tuple[DesignRegion, ...] = pd.Field(
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

    #
    # maybe put transformations here for design regions? tricky because
    # is this where you should know about them? are they fundamental to the objective or
    # fundamental to the design region? maybe there is a set of transformation config? or
    # just directly transformations
    #

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
            objectives=(self, other), combine=SumSpec(), name=f"{self.name}_{other.name}"
        )


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

        # print(grad_g)
        # print(grads)
        # asdf

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

        # allg = []
        # for objective in self.objectives:
        #     print(objective.compute_grad())
        #     allg.append(objective.comp)

        # print('----')
        # print(list(zip(*list(objective.compute_grad() for objective in self.objectives)))[1])
        # print(all_vals_grads[1][0].dtype)

        # print('grad info...')
        # print(grad_g)
        # print(grads[0])
        # print(len(grads))

        grad_per_objective = []
        for idx in range(0, len(grad_g)):
            # grad_per_objective.append(grad_g[idx] * np.array(grads[idx]))

            grad_per_objective.append(self.objectives[idx].accumulate(grad_g[idx], grads[idx]))

        # grad_per_objective = self.accumulate(grad_g, grads)

        # asdf

        # print('grads')
        # print(grad_per_objective)
        # asdf
        # print(grad_per_objective)
        return vals, grad_per_objective
        # return val_g, grad_per_objective

    def apply_grad(self, grads):
        # print('grads.....')
        # print(grads)
        # asdf
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
            # new_region = region.create_region(parameters[p_start:])

            parameter_dict[region.name] = region.parameters_to_variables(
                intermediate_idxs=np.arange(0, len(self.transformations[region.name]) + 1),
                transformations=self.transformations[region.name],
            )

            # for transformation in self.transformations:

            # new_regions.append(new_region)
            p_start += p_increment

        # print('parameter dict = ' + str(parameter_dict))
        return self.objective(parameter_dict)


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


# class FakeFloat(InvdesBaseModel):

#     val : typing.Any = pd.Field(
#         None,
#         title="val",
#         description="value")

#     def __init__(self, val : float, **kwargs):
#         super().__init__(val=val * np.ones(1), **kwargs)

#     def __assign__(self, value):
#         self.update(value)

#     def update(self, val_):
#         self.val[:] = val_


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


class Result(InvdesBaseModel):
    # optimizer_state: = pd.Field(
    #     ...,
    #     title="optimizer state",
    #     description="state that can be used to reinitialize the optimizer"
    # )

    # this will have gradients in it as well via the parameters as long as you save
    # before zeroing it
    # do we need this because these are also in the objectives
    # design_region_history: typing.Dict[str, typing.Tuple[DesignRegion, ...]] = pd.Field(
    #     ...,
    #     title="history of design regions",
    #     description="design region history indicated by region name"
    # )

    # this will have simulations in it as well
    objective_history: typing.Dict[str, typing.Tuple[AbstractObjective, ...]] = pd.Field(
        ..., title="objective history", desription="history of objective values by objective name"
    )

    figure_of_merit_history: typing.Dict[str, typing.Tuple[float, ...]] = pd.Field(
        ..., title="figure of merit history", description="figure of merit for whole optimization"
    )

    metadata_history: typing.Dict[str, typing.Tuple[typing.Any, ...]] = pd.Field(
        {}, title="metadata history", description="can include iteration, epoch, other useful info"
    )

    def __init__(self, objective_history, figure_of_merit_history, metadata_history, **kwargs):
        super().__init__(
            objective_history={
                # list((key, tuple(val)) for key, val in objective_history.items())
                # [(key, tuple(val)) for key, val in objective_history.items()]
                key: tuple(val)
                for key, val in objective_history.items()
            },
            figure_of_merit_history={
                # list((key, tuple(val)) for key, val in figure_of_merit_history.items())
                # [(key, tuple(val)) for key, val in figure_of_merit_history.items()]
                key: tuple(val)
                for key, val in figure_of_merit_history.items()
            },
            # metadata_history=dict(list((key, tuple(val)) for key, val in metadata_history.items())),
            metadata_history={
                # [(key, tuple(val)) for key, val in metadata_history.items()]
                key: tuple(val)
                for key, val in metadata_history.items()
            },
        )

    @property
    def history(self) -> typing.Dict[str, typing.Dict[str, list]]:
        objective_history = {
            # list((key, list(val)) for key, val in self.objective_history.items())
            # [(key, list(val)) for key, val in self.objective_history.items()]
            key: list(val)
            for key, val in self.objective_history.items()
        }
        figure_of_merit_history = {
            # list((key, list(val)) for key, val in self.figure_of_merit_history.items())
            # [(key, list(val)) for key, val in self.figure_of_merit_history.items()]
            key: list(val)
            for key, val in self.figure_of_merit_history.items()
        }
        metadata_history = {
            # list((key, list(val)) for key, val in self.metadata_history.items())
            # [(key, list(val)) for key, val in self.metadata_history.items()]
            key: list(val)
            for key, val in self.metadata_history.items()
        }

        return dict(
            objective_history=objective_history,
            figure_of_merit_history=figure_of_merit_history,
            metadata_history=metadata_history,
        )


OptimizerType = str

OptimizerMethods = {"GradientAscent": GradientAscentOptimizer}


class OptimizerSpec(InvdesBaseModel):
    method: OptimizerType = pd.Field(
        ..., title="Optimizer type", description="type of optimizer to look up constructor"
    )

    # should we check if these are valid arguments to the optimizer?
    # also let these override the defaults
    config: typing.Dict[str, typing.Any] = pd.Field(
        {}, title="", description="arguments to initialize optimizer"
    )

    @pd.validator("method")
    def validate_method(method, values):
        if method not in OptimizerMethods:
            raise ValidationError("Unknown optimization method")

        return method

    @pd.validator("config")
    def validate_config(config, values):
        """Make sure there are no extra kwargs being specified in config"""
        full_arg_spec = inspect.getfullargspec(OptimizerMethods[values["method"]])
        for key in config.keys():
            if key not in full_arg_spec.kwonlyargs:
                raise ValidationError("Extra kwargs provided to optimizer!")

        return config


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
    # todo: pass optimizer directly to the inverse design
    #
    # optimizer_spec: OptimizerSpec = pd.Field(
    #     ..., title="Optimizer specification", description="Construction of optimizer"
    # )

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
