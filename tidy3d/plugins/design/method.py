"""Defines the methods used for parameter sweep."""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Literal, Optional, Tuple, Union

import numpy as np
import pydantic.v1 as pd
import pygad
import scipy.stats.qmc as qmc
from bayes_opt import BayesianOptimization, UtilityFunction
from bayes_opt.event import Events
from bayes_opt.logger import JSONLogger
from pyswarms.single.global_best import GlobalBestPSO

from ...components.base import Tidy3dBaseModel
from ...constants import inf
from ...log import log
from .parameter import ParameterAny, ParameterFloat, ParameterInt, ParameterType

DEFAULT_MONTE_CARLO_SAMPLER_TYPE = qmc.LatinHypercube


class Method(Tidy3dBaseModel, ABC):
    """Spec for a sweep algorithm, with a method to run it."""

    name: str = pd.Field(None, title="Name", description="Optional name for the sweep method.")

    @abstractmethod
    def run(self, parameters: Tuple[ParameterType, ...], run_fn: Callable) -> Tuple[Any]:
        """Defines the search algorithm (sequential)."""

    def _force_int(self, next_point: dict, parameters: list):
        """Convert a float asigned to an int parameter to be an int. Update dict in place."""

        for param in parameters:
            if isinstance(param, ParameterInt):
                # Using int(round()) instead of just int as int always rounds down making upper bound value impossible
                next_point[param.name] = int(round(next_point[param.name], 0))

    @staticmethod
    def _extract_output(output: list, sampler: bool = False) -> Tuple:
        """Format the user function output for further optimisation and result storage."""
        if sampler:
            return output

        if all(isinstance(val, (float, int)) for val in output):
            # No aux_out
            return (output, None)

        if all(isinstance(val, (list, Tuple)) for val in output):
            if all(isinstance(val[0], (float, int)) for val in output):
                float_out = []
                aux_out = []
                for val in output:
                    float_out.append(val[0])
                    aux_out.append(val[1])

                # Float with aux_out
                return (float_out, aux_out)

            else:
                raise ValueError(
                    "Unrecognised output from supplied run function. The first element in the iterable object should be a float."
                )

        else:
            raise ValueError(
                "Unrecognised output from supplied run function. Output should be a float or an iterable object."
            )

    @staticmethod
    def _flatten_and_append(list_of_lists: list[list], append_target: list):
        if list_of_lists is not None:
            for sub_list in list_of_lists:
                append_target.append(sub_list)


class MethodSample(Method, ABC):
    """A sweep method where all points are independently computed."""

    @abstractmethod
    def sample(self, parameters: Tuple[ParameterType, ...], **kwargs) -> Dict[str, Any]:
        """Defines how the design parameters are sampled."""

    def _assemble_args(
        self,
        parameters: Tuple[ParameterType, ...],
    ) -> Tuple[dict, int]:
        """Sample design parameters, check the args are hashable and compute number of points."""

        fn_args = self.sample(parameters)
        for arg_dict in fn_args:
            self._force_int(arg_dict, parameters)
        return fn_args

    def run(self, parameters: Tuple[ParameterType, ...], run_fn: Callable) -> Tuple[Any]:
        """Defines the search algorithm (sequential)."""

        # get all function inputs
        fn_args = self._assemble_args(parameters)

        # Run user function on sampled args
        results = self._extract_output(run_fn(fn_args), sampler=True)

        # None is aux_output and opt_output
        return fn_args, results, None, None


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
        for param in parameters:
            vals = param.sample_grid()
            vals_each_dim[param.name] = vals

        # meshgrid each dimension's results and combine them all
        vals_grid = np.meshgrid(*vals_each_dim.values())
        vals_grid = (np.ravel(x).tolist() for x in vals_grid)
        vals_dict = dict(zip(vals_each_dim.keys(), vals_grid))
        t_vals_dict = [dict(zip(vals_dict.keys(), values)) for values in zip(*vals_dict.values())]

        return t_vals_dict


class MethodOptimise(Method, ABC):
    """A method for handling design searches that optimise the design"""

    # NOTE: We could move this to the Method base class but it's not relevant to MethodGrid
    rng_seed: pd.PositiveInt = pd.Field(
        title="Seed for random number generation", description="TBD", default=None
    )

    def any_to_int_param(self, parameter):
        """Convert ParameterAny object to integers and provide a conversion dict to return"""

        return dict(enumerate(parameter.allowed_values))

    def sol_array_to_dict(
        self, solution: np.array, keys: list, param_converter: dict
    ) -> list[dict]:
        """Convert an array of solutions to a list of dicts for function input"""
        sol_dict_list = [dict(zip(keys, sol)) for sol in solution]

        self._handle_param_convert(param_converter, sol_dict_list)

        return sol_dict_list

    def _handle_param_convert(self, param_converter, sol_dict_list):
        for param, convert in param_converter.items():
            for sol in sol_dict_list:
                if isinstance(sol[param], float):
                    sol[param] = int(round(sol[param], 0))
                sol[param] = convert[sol[param]]


class MethodBayOpt(MethodOptimise, ABC):
    """A standard method for performing bayesian optimization search"""

    initial_iter: pd.PositiveInt = pd.Field(
        ...,
        title="Number of initial random search iterations.",
        description="TBD",
    )

    n_iter: pd.PositiveInt = pd.Field(
        ...,
        title="Number of bayesian optimization iterations.",
        description="TBD",
    )

    acq_func: Optional[Literal["ucb", "ei", "poi"]] = pd.Field(
        default="ucb",
        title="Type of acquisition function.",
        description="TBD",
    )

    kappa: Optional[pd.PositiveFloat] = pd.Field(
        default=2.5,
        title="Kappa parameter for the Gaussian processor.",
        description="TBD",
    )

    xi: Optional[pd.NonNegativeFloat] = pd.Field(
        default=0.0,
        title="Xi parameter for the Gaussian processor.",
        description="TBD",
    )

    def run(self, parameters: Tuple[ParameterType, ...], run_fn: Callable) -> Tuple[Any]:
        """Defines the search algorithm for BayOpt"""

        param_converter = {}
        boundary_dict = {}
        for param in parameters:
            if type(param) == ParameterAny:
                param_converter[param.name] = self.any_to_int_param(param)
                boundary_dict[param.name] = (
                    0,
                    len(param.allowed_values) - 1,
                )  # -1 as ints are starting from 0
            else:
                boundary_dict[param.name] = param.span

        # Params must be converted back to ints for BO to learn from them
        invert_param_converter = {
            param_name: {val: key for key, val in param_values.items()}
            for param_name, param_values in param_converter.items()
        }

        # Fn can be defined here to be a combined func of pre, run_batch, post for BO to use
        utility = UtilityFunction(kind=self.acq_func, kappa=self.kappa, xi=self.xi)
        opt = BayesianOptimization(
            f=run_fn, pbounds=boundary_dict, random_state=self.rng_seed, allow_duplicate_points=True
        )

        # Create log and update
        logger = JSONLogger(path="./logs")
        opt.subscribe(Events.OPTIMIZATION_STEP, logger)

        # Run variables
        arg_list = []
        total_aux_out = []
        result = []
        for _ in range(self.initial_iter):
            next_point = opt.suggest(utility)
            self._force_int(next_point, parameters)
            self._handle_param_convert(param_converter, [next_point])
            arg_list.append(next_point)

        init_output, aux_out = self._extract_output(run_fn(arg_list))
        self._flatten_and_append(aux_out, total_aux_out)

        for next_point, next_out in zip(arg_list, init_output):
            result.append(next_out)
            self._handle_param_convert(invert_param_converter, [next_point])
            opt.register(params=next_point, target=next_out)

        # Handle subsequent iterations sequentially as BayOpt package does not allow for batched non-random predictions
        for _ in range(self.n_iter):
            next_point = opt.suggest(utility)
            self._force_int(next_point, parameters)
            self._handle_param_convert(param_converter, [next_point])
            next_out, aux_out = self._extract_output(run_fn([next_point]))
            result.append(next_out[0])
            self._flatten_and_append(aux_out, total_aux_out)
            self._handle_param_convert(invert_param_converter, [next_point])
            opt.register(params=next_point, target=next_out[0])

        # Output fn_args from the BO.opt object - getting results in situ as opt changes type to float
        fn_args = []
        for output in opt.res:
            fn_args.append(output["params"])

        return fn_args, result, total_aux_out, opt


class MethodGenAlg(MethodOptimise, ABC):
    """A standard method for performing genetic algorithm search"""

    # Args for the user
    solutions_per_pop: pd.PositiveInt = pd.Field(
        ...,
        title="Number of solutions per population.",
        description="TBD",
    )

    n_generations: pd.PositiveInt = pd.Field(
        ...,
        title="Number of generations.",
        description="TBD",
    )

    n_parents_mating: pd.PositiveInt = pd.Field(
        ...,
        title="Number of parents mating.",
        description="TBD",
    )

    stop_criteria: Optional[pd.constr(regex=r"\b(?:reach|saturate)_\d+\b")] = pd.Field(
        default=None,
        title="Early stopping criteria.",
        description="Define the early stopping criteria. Supported words are 'reach_X' or 'saturate_X' where X is a number. See PyGAD docs for more details",
    )

    parent_selection_type: Optional[
        Literal["sss", "rws", "sus", "rank", "random", "tournament"]
    ] = pd.Field(
        default="sss",
        title="Parent selection type.",
        description="TBD",
    )

    crossover_type: Optional[Literal["single_point", "two_points", "uniform", "scattered"]] = (
        pd.Field(
            default="single_point",
            title="Crossover type.",
            description="TBD",
        )
    )

    crossover_prob: Optional[pd.confloat(ge=0, le=1)] = pd.Field(
        default=0.8,
        title="Crossover probability.",
        description="TBD",
    )

    mutation_type: Optional[Literal["random", "swap", "inversion", "scramble", "adaptive"]] = (
        pd.Field(
            default="random",
            title="Crossover type.",
            description="TBD",
        )
    )

    mutation_prob: Optional[pd.confloat(ge=0, le=1)] = pd.Field(
        default=0.2,
        title="Crossover probability.",
        description="TBD",
    )

    # TODO: See if anyone is interested in having the full suite of PyGAD options - there's a lot!

    def run(self, parameters: Tuple[ParameterType, ...], run_fn: Callable) -> Tuple[Any]:
        """Defines the search algorithm for the GA"""

        # Make param names available to the fitness function
        param_keys = [param.name for param in parameters]

        # Store parameters and fitness
        store_parameters = []
        store_fitness = []
        store_aux = []

        # Set gene_spaces to keep GA within ranges
        param_converter = {}
        gene_spaces = []
        gene_types = []
        for param in parameters:
            if type(param) == ParameterFloat:
                gene_spaces.append({"low": param.span[0], "high": param.span[1]})
                gene_types.append(float)
            elif type(param) == ParameterInt:
                gene_spaces.append(
                    range(param.span[0], param.span[1] + 1)
                )  # +1 included so as to be inclusive of upper range value
                gene_types.append(int)
            else:
                # Designed for str in ParameterAny but may work for anything
                param_converter[param.name] = self.any_to_int_param(param)

                gene_spaces.append(range(0, len(param.allowed_values)))
                gene_types.append(int)

        # Create fitness function combining pre and post fn with the tidy3d call
        def fitness_function(ga_instance, solution, solution_idx):
            # Break solution down to dict
            sol_dict = self.sol_array_to_dict(solution, param_keys, param_converter)

            # Iterate through solutions for batched and non-batched data
            sol_out, aux_out = self._extract_output(run_fn(sol_dict))

            self._flatten_and_append(aux_out, store_aux)

            return sol_out

        def on_generation(ga_instance):
            store_parameters.append(ga_instance.population.copy())
            store_fitness.append(ga_instance.last_generation_fitness)
            best_fitness = ga_instance.best_solution()[1]
            print(
                f"Generation {ga_instance.generations_completed}: Best Fitness = {best_fitness:.3f}"
            )

        # Determine initial array
        num_genes = len(parameters)

        # define the optimizer
        ga_instance = pygad.GA(
            num_generations=self.n_generations,
            num_parents_mating=self.n_parents_mating,
            fitness_func=fitness_function,
            parent_selection_type=self.parent_selection_type,
            mutation_type=self.mutation_type,
            mutation_probability=self.mutation_prob,
            crossover_type=self.crossover_type,
            crossover_probability=self.crossover_prob,
            sol_per_pop=self.solutions_per_pop,
            num_genes=num_genes,
            fitness_batch_size=self.solutions_per_pop,
            on_generation=on_generation,
            random_seed=self.rng_seed,
            gene_space=gene_spaces,
            gene_type=gene_types,
            stop_criteria=self.stop_criteria,
            save_solutions=True,
        )

        ga_instance.run()

        # Format output
        fn_args = [dict(zip(param_keys, val)) for arr in store_parameters for val in arr]
        results = [val for arr in store_fitness for val in arr]

        return fn_args, results, store_aux, ga_instance


class MethodParticleSwarm(MethodOptimise, ABC):
    """A standard method for performing particle swarm search"""

    n_particles: pd.PositiveInt = pd.Field(
        ...,
        title="Number of particles in the swarm.",
        description="TBD",
    )

    n_iter: pd.PositiveInt = pd.Field(
        ...,
        title="Number of generations.",
        description="TBD",
    )

    cognitive_coeff: Optional[pd.PositiveFloat] = pd.Field(
        default=1.5,
        title="Number of parents mating.",
        description="TBD",
    )

    social_coeff: Optional[pd.PositiveFloat] = pd.Field(
        default=1.5,
        title="Number of parents mating.",
        description="TBD",
    )

    weight: Optional[pd.PositiveFloat] = pd.Field(
        default=0.9,
        title="Number of parents mating.",
        description="TBD",
    )

    ftol: Optional[Union[pd.confloat(ge=0, le=1), Literal[-inf]]] = pd.Field(
        default=-inf,
        title="Relative error for convergence.",
        description="Relative error in objective_func(best_pos) acceptable for convergence. See https://pyswarms.readthedocs.io/en/latest/examples/tutorials/tolerance.html for details. Off by default.",
    )

    ftol_iter: Optional[pd.PositiveInt] = pd.Field(
        default=1,
        title="Number of iterations before acceptable convergence.",
        description="Number of iterations over which the relative error in objective_func is acceptable for convergence.",
    )

    def run(self, parameters: Tuple[ParameterType, ...], run_fn: Callable) -> Tuple[Any]:
        # Pyswarms doesn't have a seed set outside of numpy std method
        np.random.seed(self.rng_seed)

        # Variable assignment here so it is available to the fitness function
        param_keys = [param.name for param in parameters]

        store_parameters = []
        store_fitness = []
        store_aux = []

        # Build bounds and conversion dict for ParameterAny inputs
        param_converter = {}
        min_bound = []
        max_bound = []
        for param in parameters:
            if isinstance(param, ParameterAny):
                param_converter[param.name] = self.any_to_int_param(param)
                min_bound.append(0)
                max_bound.append(len(param.allowed_values) - 1)  # -1 as ints are starting from 0
            else:
                min_bound.append(param.span[0])
                max_bound.append(param.span[1])

        bounds = (min_bound, max_bound)

        def fitness_function(solution):
            # Correct solutions that should be ints
            sol_dict = self.sol_array_to_dict(solution, param_keys, param_converter)
            for arg_dict in sol_dict:
                self._force_int(arg_dict, parameters)

            store_parameters.append(sol_dict)

            sol_out, aux_out = self._extract_output(run_fn(sol_dict))

            self._flatten_and_append(aux_out, store_aux)

            # Stored before minus to give true answers
            store_fitness.append(sol_out)

            # Set as negative as PSO uses a minimising cost function
            return -np.array(sol_out)

        options = {"c1": self.cognitive_coeff, "c2": self.social_coeff, "w": self.weight}
        optimizer = GlobalBestPSO(
            n_particles=self.n_particles,
            dimensions=len(parameters),
            options=options,
            bounds=bounds,
            ftol=self.ftol,
            ftol_iter=self.ftol_iter,
            # TODO: including oh_strategy would be nice but complicated to specify with pydantic oh_strategy={"w": "exp_decay"},
        )

        _ = optimizer.optimize(fitness_function, self.n_iter)

        # Collapse stores into fn_args and results lists
        fn_args = [val for sublist in store_parameters for val in sublist]
        results = [val for sublist in store_fitness for val in sublist]

        return fn_args, results, store_aux, optimizer


class AbstractMethodRandom(MethodSample, ABC):
    """Select parameters with an object with a ``random`` method."""

    num_points: pd.PositiveInt = pd.Field(
        ...,
        title="Number of points for sampling",
        description="TBD",
    )

    rng_seed: pd.PositiveInt = pd.Field(
        title="Seed for random number generation", description="TBD", default=None
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
        return DEFAULT_MONTE_CARLO_SAMPLER_TYPE(d=d, seed=self.rng_seed)


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
        np.random.seed(self.rng_seed)

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

    sampler_kwargs: Any = pd.Field(
        {},
        title="Keyword arguments for sampler",
        description="tbd",
    )

    # @pd.validator("sampler")
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

        loaded_sampler = self.sampler(**self.sampler_kwargs)
        self._check_sampler(loaded_sampler)
        num_dims_vars = len(parameters)
        num_dims_sampler = loaded_sampler.random(1).size

        if num_dims_sampler != num_dims_vars:
            raise ValueError(
                f"The sampler {self.sampler} has {num_dims_sampler} dimensions, "
                f"but the design space has {num_dims_vars} dimensions. These must be equivalent. "
            )

        return loaded_sampler


MethodType = Union[
    MethodMonteCarlo,
    MethodGrid,
    MethodRandom,
    MethodRandomCustom,
    MethodBayOpt,
    MethodGenAlg,
    MethodParticleSwarm,
]
