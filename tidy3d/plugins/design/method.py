"""Defines the methods used for parameter sweep."""

from abc import ABC, abstractmethod
from inspect import getsource
from typing import Any, Callable, Dict, Literal, Optional, Tuple, Union

import numpy as np
import pydantic.v1 as pd
import pygad
import scipy.stats.qmc as qmc
from bayes_opt import BayesianOptimization, UtilityFunction
from bayes_opt.event import Events
from bayes_opt.logger import JSONLogger
from pyswarms.single.global_best import GlobalBestPSO

import tidy3d.plugins.design as tdd
import tidy3d.web as web

# from ... import web # What's this for?
from ...components.base import Tidy3dBaseModel
from ...components.simulation import Simulation
from ...log import log
from .parameter import ParameterType

DEFAULT_MONTE_CARLO_SAMPLER_TYPE = qmc.LatinHypercube


class Method(Tidy3dBaseModel, ABC):
    """Spec for a sweep algorithm, with a method to run it."""

    name: str = pd.Field(None, title="Name", description="Optional name for the sweep method.")

    @abstractmethod
    def run(
        self, parameters: Tuple[ParameterType, ...], pre_fn: Callable, post_fn: Callable
    ) -> Tuple[Any]:
        """Defines the search algorithm (sequential)."""

    @staticmethod
    def assert_hashable(fn_args: dict) -> None:
        """Raise error if the function arguments aren't hashable (do before computation)."""
        fn_args_tuple = tuple(fn_args)
        try:
            hash(fn_args_tuple)
        except TypeError:
            raise ValueError(
                "Function arguments must be hashable. "
                "Parameter sweep tool won't work with sets of lists, dicts or numpy arrays. "
                "Convert these to 'tuple' for a workaround."
            )

    @staticmethod
    def assert_num_points(fn_args: Dict[str, tuple]) -> int:
        """Compute number of points from the function arguments and do error checking."""
        num_points_each_dim = [len(val) for val in fn_args.values()]
        if len(set(num_points_each_dim)) != 1:
            raise ValueError(
                f"Found different number of points: {num_points_each_dim} along each dimension. "
                "This suggests a bug in the parameter sweep tool. "
                "Please raise an issue on the front end GitHub repository."
            )

    def _force_int(self, next_point, parameters):
        """Convert a float asigned to an int parameter to be an int

        Update dict in place
        """
        # int_keys = [param.name for param in parameters if type(param) == tdd.ParameterInt]
        for param in parameters:
            if type(param) == tdd.ParameterInt:
                # Using int(round()) instead of just int as int always rounds down making upper bound value impossible
                new_int = int(round(next_point[param.name], 0))

                # Check that the new int is within the bounds - correct if not
                if new_int < param.span[0]:
                    new_int = param.span[0]
                elif new_int > param.span[1]:
                    new_int = param.span[1]

                next_point[param.name] = new_int

    def _tidy3d_run(self, sims, run_kwargs):
        """Generic logic for running a pre_fn on tidy3d servers"""

        # Add sim type right before running
        run_kwargs.update(simulation_type="tidy3d_design_testing")

        # If a single simulation is supplied
        if isinstance(sims, Simulation):
            return web.run(sims, **run_kwargs)

        # Uses batches if multiple simulations have been supplied
        else:
            batch = web.Batch(simulations=sims, **run_kwargs)
            return batch.run()

    @staticmethod
    def _can_batch(parameters: Tuple[ParameterType, ...], run_fn: Union[Callable, Tuple]):
        """See if pre produces an arrangement of td.Simulation or is unrelated"""

        if isinstance(run_fn, Tuple):
            # Check if tidy3d is called in func
            source = getsource(run_fn[0])
            if "web.Batch(" in source and "web.run(" in source:
                raise ValueError(
                    "Can't effectively control batches from a pre-function that calls the tidy3d API. "
                    "Please convert to a single function or change the function to return a Simulation object."
                )

            # Run fn_pre with the lower bound of each parameter
            lower_params = {param.name: param.span[0] for param in parameters}
            result = run_fn[0](**lower_params)

            # Check that pre_fn returns a Simulation object
            if isinstance(result, Simulation):
                return True
            else:
                # Not valid - batching can only be done on tidy3d Simulation generators
                raise ValueError(
                    "Can't create batches from a pre-function that does not output Simulation objects. "
                    "Please convert to a single function or change the function to return a Simulation object."
                )

        else:
            # For single functions that pass through sequentially
            return False


class MethodSample(Method, ABC):
    """A sweep method where all points are independently computed."""

    @abstractmethod
    def sample(self, parameters: Tuple[ParameterType, ...], **kwargs) -> Dict[str, Any]:
        """Defines how the design parameters are sampled."""

    def _assemble_args(
        self,
        parameters: Tuple[ParameterType, ...],
        run_fn: Callable,
    ) -> Tuple[dict, int]:
        """Sample design parameters, check the args are hashable and compute number of points."""

        fn_args = self.sample(parameters)
        for arg_dict in fn_args:
            self._force_int(arg_dict, parameters)
        is_batchable = self._can_batch(parameters, run_fn)
        # self.assert_hashable(fn_args)
        # self.assert_num_points(fn_args)
        return fn_args, is_batchable

    def _eval_run(self, fn_args: Dict[str, tuple], run_fn, is_batchable):
        """Handle batchable and non-batchable jobs for a given run_function"""

        processed_result = []
        if is_batchable:
            pre_fn = run_fn[0]
            post_fn = run_fn[1]
            run_kwargs = {}
            # Create dict of simulations
            if len(fn_args) == 1:
                sims = pre_fn(**fn_args[0])
            else:
                sims = {str(i): pre_fn(**arg_dict) for i, arg_dict in enumerate(fn_args)}

            pre_out = self._tidy3d_run(sims, run_kwargs)
            results = [sim_tuple[1] for sim_tuple in pre_out.items()]

            # Post process the data
            for res in results:
                processed_result.append(post_fn(res))

        else:
            for arg_dict in fn_args:
                fn_output = run_fn(**arg_dict)
                processed_result.append(fn_output)

        return processed_result

    def run(
        self, parameters: Tuple[ParameterType, ...], run_fn: Union[Callable, Tuple]
    ) -> Tuple[Any]:
        """Defines the search algorithm (sequential)."""

        # get all function inputs
        fn_args, is_batchable = self._assemble_args(parameters, run_fn)

        # Get min and max for each sample
        # args_as_params = [[arg_dict[key] for arg_dict in fn_args] for key in fn_args[0]]
        # min_max_params = [(min(param), max(param)) for param in args_as_params]

        # for each point, construct the function inputs, run it, record output
        results = self._eval_run(fn_args, run_fn, is_batchable)

        return fn_args, results


class MethodGrid(MethodSample):
    """Select parameters uniformly on a grid.

    Example
    -------
    >>> import tidy3d.plugins.design as tdd
    >>> method = tdd.MethodGrid()
    """

    def sample(self, parameters: Tuple[ParameterType, ...], **kwargs) -> Dict[str, Any]:
        """Defines how the design parameters are sampled on grid."""

        # sample each dimension individually
        vals_each_dim = {}
        for design_var in parameters:
            vals = design_var.sample_grid()
            vals_each_dim[design_var.name] = vals

        # meshgrid each dimension's results and combine them all
        vals_grid = np.meshgrid(*vals_each_dim.values())
        vals_grid = (np.ravel(x).tolist() for x in vals_grid)
        vals_dict = dict(zip(vals_each_dim.keys(), vals_grid))
        t_vals_dict = [dict(zip(vals_dict.keys(), values)) for values in zip(*vals_dict.values())]

        return t_vals_dict


class MethodOptimise(Method, ABC):
    """A method for handling design searches that optimise the design"""

    def create_boundary_dict(
        self,
        parameters: Tuple[ParameterType, ...],
    ):
        """Reshape parameter spans to dict of boundaries"""

        return {design_var.name: design_var.span for design_var in parameters}

    def manage_batch_run(self, run_fn, sol_dict, run_kwargs):
        sol_out = []
        if _is_batchable:
            pre_fn = run_fn[0]
            post_fn = run_fn[1]
            sim_dict = {}
            for sim_idx, arg_dict in enumerate(sol_dict):
                sim_dict[sim_idx] = pre_fn(**arg_dict)

            sim_result = self._tidy3d_run(sim_dict, run_kwargs)
            sim_data = [sim_tuple[1] for sim_tuple in sim_result.items()]
            for next_out in sim_data:
                sol_out.append(post_fn(next_out))

        else:
            for arg_dict in sol_dict:
                sol_out.append(run_fn(**arg_dict))

        return sol_out

    def sol_array_to_dict(self, solution, keys):
        return [dict(zip(keys, sol)) for sol in solution]


class MethodBayOpt(MethodOptimise, ABC):
    """A standard method for performing bayesian optimization search"""

    initial_iter: pd.PositiveInt = pd.Field(
        ...,
        title="Number of initial random search iterations",
        description="TBD",
    )

    n_iter: pd.PositiveInt = pd.Field(
        ...,
        title="Number of bayesian optimization iterations",
        description="TBD",
    )

    acq_func: Optional[Literal["ucb", "ei", "poi"]] = pd.Field(
        title="Type of acquisition function",
        description="TBD",
        default="ucb",
    )

    def run(
        self, parameters: Tuple[ParameterType, ...], pre_fn: Callable, post_fn: Callable
    ) -> Tuple[Any]:
        """Defines the search algorithm for BayOpt"""
        boundary_dict = self.create_boundary_dict(parameters)
        is_batchable = self._can_batch(self, parameters, pre_fn)

        # Fn can be defined here to be a combined func of pre, run_batch, post for BO to use
        utility = UtilityFunction(kind=self.acq_func, kappa=2.5, xi=0.0)
        opt = BayesianOptimization(
            f=pre_fn, pbounds=boundary_dict, random_state=1, allow_duplicate_points=True
        )

        # Create log and update
        logger = JSONLogger(path="./logs")
        opt.subscribe(Events.OPTIMIZATION_STEP, logger)

        # Run variables
        arg_list = []
        if is_batchable:
            # Handle the initial random samples as a batch to save time
            sim_dict = {}
            run_kwargs = {}
            for sim_idx in range(self.initial_iter):
                next_point = opt.suggest(utility)
                self._force_int(next_point, parameters)
                arg_list.append(next_point)
                sim_dict[f"init_{sim_idx}"] = pre_fn(**next_point)

            pre_out = self._tidy3d_run(sim_dict, run_kwargs)

            # Get the sim data out
            sim_data = [sim_tuple[1] for sim_tuple in pre_out.items()]

            for next_point, sim in zip(arg_list, sim_data):
                next_out = post_fn(sim)
                opt.register(params=next_point, target=next_out)

            # Handle subsequent iterations sequentially as BayOpt package does not allow for batched non-random predictions
            for idx in range(self.n_iter):
                next_point = opt.suggest(utility)
                self._force_int(next_point, parameters)

                # Determine task name
                task_name = str(idx)
                run_kwargs = {"task_name": task_name}

                # Data submitted as single Simulations not dict
                pre_out = self._tidy3d_run(pre_fn(**next_point), run_kwargs)

                next_out = post_fn(pre_out)
                opt.register(params=next_point, target=next_out)
        else:
            # Handle the initial random samples as a batch to save time
            init_output = []
            for _ in range(self.initial_iter):
                next_point = opt.suggest(utility)
                self._force_int(next_point, parameters)
                arg_list.append(next_point)
                init_output.append(post_fn(pre_fn(**next_point)))

            for next_point, next_out in zip(arg_list, init_output):
                opt.register(params=next_point, target=next_out)

            # Handle subsequent iterations sequentially as BayOpt package does not allow for batched non-random predictions
            for _ in range(self.n_iter):
                next_point = opt.suggest(utility)
                self._force_int(next_point, parameters)
                pre_out = pre_fn(**next_point)
                next_out = post_fn(pre_out)
                opt.register(params=next_point, target=next_out)

        # Output results from the BO.opt object
        result = []
        fn_args = []
        for output in opt.res:
            result.append(output["target"])
            fn_args.append(output["params"])

        return fn_args, result


class MethodGenAlg(MethodOptimise, ABC):
    """A standard method for performing genetic algorithm search"""

    # Args for the user
    # Solutions per pop
    # Num generations
    # Num parents mating
    # Parent selector, optional

    def run(
        self, parameters: Tuple[ParameterType, ...], run_fn: Union[Callable, Tuple]
    ) -> Tuple[Any]:
        """Defines the search algorithm for the GA"""

        # Create fitness function combining pre and post fn with the tidy3d call
        def fitness_function(ga_instance, solution, solution_idx):
            # Hard-coded kwargs
            run_kwargs = {}

            # Break solution down to dict
            sol_dict = self.sol_array_to_dict(solution, _param_keys)

            # Iterate through solutions for batched and non-batched data
            sol_out = self.manage_batch_run(run_fn, sol_dict, run_kwargs)

            return sol_out

        def on_generation(ga_instance):
            _store_parameters.append(ga_instance.population.copy())
            _store_fitness.append(ga_instance.last_generation_fitness)
            best_fitness = ga_instance.best_solution()[1]
            print(
                f"Generation {ga_instance.generations_completed}: Best Fitness = {best_fitness:.3f}"
            )

        # Make param names available to the fitness function
        global _param_keys
        _param_keys = [param.name for param in parameters]

        # Store parameters and fitness
        global _store_parameters
        global _store_fitness
        _store_parameters = []
        _store_fitness = []

        # Determine if batching is possible
        global _is_batchable
        _is_batchable = self._can_batch(parameters, run_fn)

        # Set gene_spaces to keep GA within ranges
        gene_spaces = []
        gene_types = []
        for param in parameters:
            if type(param) == tdd.ParameterFloat:
                gene_spaces.append({"low": param.span[0], "high": param.span[1]})
                gene_types.append(float)
            elif type(param) == tdd.ParameterInt:
                gene_spaces.append(
                    range(param.span[0], param.span[1] + 1)
                )  # +1 included so as to be inclusive of upper range value
                gene_types.append(int)
            else:
                print("Parameter type not supported by GA method.")

        # Determine initial array

        sol_per_pop = 10  # number of solutions in the population
        num_genes = len(parameters)
        num_generations = 3  # number of generation

        num_parents_mating = 4  # number of mating parents
        parent_selection_type = "rws"  # parent selection rule

        crossover_type = "scattered"  # crossover rule
        crossover_probability = 0.7  # cross over probability

        mutation_type = "random"  # mutation rule
        mutation_probability = 0.2

        # define the optimizer
        ga_instance = pygad.GA(
            num_generations=num_generations,
            num_parents_mating=num_parents_mating,
            fitness_func=fitness_function,
            parent_selection_type=parent_selection_type,
            mutation_type=mutation_type,
            mutation_probability=mutation_probability,
            crossover_type=crossover_type,
            crossover_probability=crossover_probability,
            sol_per_pop=sol_per_pop,
            num_genes=num_genes,
            fitness_batch_size=sol_per_pop,
            on_generation=on_generation,
            random_seed=1,
            gene_space=gene_spaces,
            gene_type=gene_types,
        )

        ga_instance.run()

        # Format output
        fn_args = [dict(zip(_param_keys, val)) for arr in _store_parameters for val in arr]
        results = [val for arr in _store_fitness for val in arr]

        return fn_args, results


class MethodParticleSwarm(MethodOptimise, ABC):
    """A standard method for performing particle swarm search"""

    def run(
        self, parameters: Tuple[ParameterType, ...], run_fn: Union[Callable, Tuple]
    ) -> Tuple[Any]:
        # Args for the user
        # n_particles
        # 3 PSO options
        # n_iter

        def fitness_function(solution):
            # Hard-coded kwargs
            run_kwargs = {}

            # Correct solutions that should be ints
            sol_dict = self.sol_array_to_dict(solution, _param_keys)
            for arg_dict in sol_dict:
                self._force_int(arg_dict, parameters)

            _store_parameters.append(sol_dict)

            sol_out = self.manage_batch_run(run_fn, sol_dict, run_kwargs)

            # Stored before minus to give true answers
            _store_fitness.append(sol_out)

            # Set as negative as PSO uses a minimising cost function
            return -np.array(sol_out)

        # Make param names available to the fitness function
        global _param_keys
        _param_keys = [param.name for param in parameters]

        global _store_parameters
        global _store_fitness
        _store_parameters = []
        _store_fitness = []

        # Determine if batching is possible
        global _is_batchable
        _is_batchable = self._can_batch(parameters, run_fn)

        # Determine bounds and format
        min_bound = [param.span[0] for param in parameters]
        max_bound = [param.span[1] for param in parameters]
        bounds = (min_bound, max_bound)

        options = {"c1": 1.5, "c2": 1.5, "w": 0.9}
        optimizer = GlobalBestPSO(
            n_particles=10,
            dimensions=len(parameters),
            options=options,
            bounds=bounds,
            oh_strategy={"w": "exp_decay"},
        )

        _ = optimizer.optimize(fitness_function, 100)

        # Collapse stores into fn_args and results lists
        fn_args = [val for sublist in _store_parameters for val in sublist]
        results = [val for sublist in _store_fitness for val in sublist]

        return fn_args, results


class AbstractMethodRandom(MethodSample, ABC):
    """Select parameters with an object with a ``random`` method."""

    num_points: pd.PositiveInt = pd.Field(
        ...,
        title="Number of points for sampling",
        description="TBD",
    )

    @abstractmethod
    def get_sampler(self, parameters: Tuple[ParameterType, ...]) -> qmc.QMCEngine:
        """Sampler for this ``Method`` class. If ``None``, sets a default."""

    def sample(self, parameters: Tuple[ParameterType, ...], **kwargs) -> Dict[str, Any]:
        """Defines how the design parameters are sampled on grid."""

        sampler = self.get_sampler(parameters)
        pts_01 = sampler.random(self.num_points)

        # Convert value from 0-1 to fit within the parameter spans
        args_by_param = []
        for i, design_var in enumerate(parameters):
            pts_i_01 = pts_01[..., i]
            args_by_param.append(design_var.select_from_01(pts_i_01))
        args_by_sample = [[row[i] for row in args_by_param] for i in range(len(args_by_param[0]))]

        # Get output list of kwargs for pre_fn
        keys = [param.name for param in parameters]
        result = [{keys[j]: row[j] for j in range(len(keys))} for row in args_by_sample]

        return result


class MethodMonteCarlo(AbstractMethodRandom):
    """Select sampling points using Monte Carlo sampling (Latin Hypercube method).

    Example
    -------
    >>> import tidy3d.plugins.design as tdd
    >>> method = tdd.MethodMonteCarlo(num_points=20)
    """

    def get_sampler(self, parameters: Tuple[ParameterType, ...]) -> qmc.QMCEngine:
        """Sampler for this ``Method`` class."""

        d = len(parameters)
        return DEFAULT_MONTE_CARLO_SAMPLER_TYPE(d=d)


class MethodRandom(AbstractMethodRandom):
    """Select sampling points uniformly at random.

    Example
    -------
    >>> import tidy3d.plugins.design as tdd
    >>> method = tdd.MethodRandom(num_points=20, monte_carlo_warning=False)
    """

    monte_carlo_warning: bool = pd.Field(
        True,
        title="Monte Carlo Suggestion",
        description="We recommend you use ``MethodMonteCarlo`` as it is more efficient at sampling."
        " Setting this field to ``False`` will disable the "
        "warning that occurs when this class is made.",
    )

    @pd.validator("monte_carlo_warning", always=True)
    def _suggest_monte_carlo(cls, val):
        """Suggest that the user use ``MethodMonteCarlo`` instead of this method."""
        if val:
            log.warning(
                "We recommend using the Monte Carlo method to sample your design space instead of "
                "this method, which samples uniformly at random. Monte Carlo is more efficient at "
                "sampling and generally needs fewer points than uniform random sampling. "
                "Please consider using 'sweep.MethodMonteCarlo'. "
                "If you are intentionally using uniform random sampling, "
                "you can disable this warning by setting 'monte_carlo_warning=False' "
                "in 'MethodRandom'."
            )
        return val

    def get_sampler(self, parameters: Tuple[ParameterType, ...]) -> qmc.QMCEngine:
        """Sampler for this ``Method`` class."""

        d = len(parameters)

        class UniformRandomSampler:
            """Has ``.random(n)`` returning ``(n, d)`` array sampled random uniformly in [0, 1]."""

            def random(self, n) -> np.ndarray:
                """Return ``(n, d)``-shaped array sampled uniformly at random in range [0, 1]."""
                return np.random.random((n, d))

        return UniformRandomSampler()


class MethodRandomCustom(AbstractMethodRandom):
    """Select parameters with an object with a user supplied sampler with a ``.random`` method.

    Example
    -------
    >>> import tidy3d.plugins.design as tdd
    >>> import scipy.stats.qmc as qmc
    >>> sampler = qmc.Halton(d=3)
    >>> method = tdd.MethodRandomCustom(num_points=20, sampler=sampler)
    """

    sampler: Any = pd.Field(
        None,
        title="Custom Sampler",
        description="An object with a ``.random(n)`` method, which returns a ``np.ndarray`` "
        "of shape ``(n, d)`` where d is the number of dimensions of the design space. "
        "Values must lie between [0, 1] and will be re-scaled depending on the design parameters. "
        " Compatible objects include instances of ``scipy.stats.qmc.QMCEngine``, but other objects "
        " can also be supplied.",
    )

    @pd.validator("sampler")
    def _check_sampler(cls, val):
        """make sure sampler has required methods."""
        if not hasattr(val, "random"):
            raise ValueError(
                "Sampler must have a 'random(n)' method, "
                "returning a numpy array of shape '(n, d)' where 'd' is the number of dimensions "
                "in the design space."
            )
        n = 30
        sample_values = val.random(n)
        if not isinstance(sample_values, np.ndarray):
            raise ValueError(
                f"'sampler.random(n)' must return a 'np.ndarray' object, got {type(sample_values)}."
            )
        sample_shape = sample_values.shape
        if len(sample_shape) != 2:
            raise ValueError(
                f"The 'sampler.random(n)' method must give a 'np.ndarray of shape (n, d)', "
                "where 'd' is the number of dimensions in the design parameters. "
                f"Supplied sampler gave an array with {len(sample_shape)} dimensions."
            )
        if sample_shape[0] != n:
            raise ValueError(
                f"The 'sampler.random(n)' method must give a 'np.ndarray of shape (n, d)', "
                "where 'd' is the number of dimensions in the design parameters. "
                f"Supplied sampler gave an array of shape ({sample_shape[0]}, d)."
            )
        if np.any(sample_values > 1) or np.any(sample_values < 0):
            raise ValueError(
                "The 'sampler.random(n)' method must give a 'np.ndarray of shape (n, d)' where all "
                "values lie between 0 and 1. After the points are generated, "
                "their values will be resampled appropriately depending "
                "on the 'parameters' used in the parameter sweep 'DesignSpace'."
            )
        return val

    def get_sampler(self, parameters: Tuple[ParameterType, ...]) -> qmc.QMCEngine:
        """Sampler for this ``Method`` class. If ``None``, sets a default."""

        num_dims_vars = len(parameters)
        num_dims_sampler = self.sampler.random(1).size

        if num_dims_sampler != num_dims_vars:
            raise ValueError(
                f"The sampler {self.sampler} has {num_dims_sampler} dimensions, "
                f"but the design space has {num_dims_vars} dimensions. These must be equivalent. "
            )

        return self.sampler


MethodType = Union[
    MethodMonteCarlo,
    MethodGrid,
    MethodRandom,
    MethodRandomCustom,
    MethodBayOpt,
    MethodGenAlg,
    MethodParticleSwarm,
]
