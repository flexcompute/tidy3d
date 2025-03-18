"""Defines the methods used for parameter sweep."""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Literal, Tuple, Union

import numpy as np
import pydantic.v1 as pd
import scipy.stats.qmc as qmc

from ...components.base import Tidy3dBaseModel
from ...constants import inf
from .parameter import ParameterAny, ParameterFloat, ParameterInt, ParameterType

DEFAULT_MONTE_CARLO_SAMPLER_TYPE = qmc.LatinHypercube


class Method(Tidy3dBaseModel, ABC):
    """Spec for a sweep algorithm, with a method to run it."""

    name: str = pd.Field(None, title="Name", description="Optional name for the sweep method.")

    @abstractmethod
    def _run(self, parameters: Tuple[ParameterType, ...], run_fn: Callable) -> Tuple[Any]:
        """Defines the search algorithm."""

    @abstractmethod
    def _get_run_count(self, parameters: list = None) -> int:
        """Return the maximum number of runs for the method based on current method arguments."""

    def _force_int(self, next_point: dict, parameters: list) -> None:
        """Convert a float asigned to an int parameter to be an int. Update dict in place."""

        for param in parameters:
            if isinstance(param, ParameterInt):
                # Using int(round()) instead of just int as int always rounds down making upper bound value impossible
                next_point[param.name] = int(round(next_point[param.name], 0))

    @staticmethod
    def _extract_output(output: list, sampler: bool = False) -> Tuple:
        """Format the user function output for further optimization and result storage."""

        # Light check if all the outputs are the same type
        # If the user has supplied multiple returns this may catch it
        # If the user has multiple returns and it passes this then it's less likely to cause further problems
        if not all(isinstance(val, type(output[0])) for val in output):
            raise ValueError(
                "Unrecognized output from supplied post function. The type of output varies across the output."
                "Use of multiple return functions in 'fn_post' is discouraged."
                "If this is a problem please raise an issue on the Tidy3D Github page."
            )

        if sampler:
            return output

        if all(isinstance(val, (float, int)) for val in output):
            # No aux_out
            none_aux = [None for _ in range(len(output))]
            return (output, none_aux)

        if all(isinstance(val, (list, Tuple)) for val in output):
            if all(isinstance(val[0], (float, int)) for val in output):
                float_out = []
                aux_out = []
                for val in output:
                    if len(val) > 2:
                        raise ValueError(
                            "Unrecognized output from supplied post function. Too many elements in the return object, it should be a 'float' and a 'list'/'tuple'/'dict'."
                        )

                    float_out.append(val[0])
                    aux_out.append(val[1])

                # Float with aux_out
                return (float_out, aux_out)

            else:
                raise ValueError(
                    "Unrecognized output from supplied post function. The first element in the iterable object should be a 'float'."
                )

        else:
            raise ValueError(
                "Unrecognized output from supplied post function. Output should be a 'float' or an iterable object."
            )

    @staticmethod
    def _flatten_and_append(list_of_lists: list[list], append_target: list) -> None:
        """Flatten a list of lists and append the sublist to a new list."""
        if list_of_lists is not None:
            for sub_list in list_of_lists:
                append_target.append(sub_list)


class MethodSample(Method, ABC):
    """A sweep method where all points are independently computed in one iteration."""

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

    def _run(self, parameters: Tuple[ParameterType, ...], run_fn: Callable, console) -> Tuple[Any]:
        """Defines the search algorithm."""

        # get all function inputs
        fn_args = self._assemble_args(parameters)

        # Run user function on sampled args
        results = self._extract_output(run_fn(fn_args), sampler=True)

        # None is aux_output and opt_output
        return fn_args, results, None, None


class MethodGrid(MethodSample):
    """Select parameters uniformly on a grid.
    Size of the grid is specified by the parameter type,
    either as the number of unique discrete values (``ParameterInt``, ``ParameterAny``)
    or with the num_points argument (``ParameterFloat``).

    Example
    -------
    >>> import tidy3d.plugins.design as tdd
    >>> method = tdd.MethodGrid()
    """

    def _get_run_count(self, parameters: list) -> int:
        """Return the maximum number of runs for the method based on current method arguments."""
        return len(self.sample(parameters))

    @staticmethod
    def sample(parameters: Tuple[ParameterType, ...]) -> Dict[str, Any]:
        """Defines how the design parameters are sampled on the grid."""

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


class MethodOptimize(Method, ABC):
    """A method for handling design searches that optimize the design."""

    # NOTE: We could move this to the Method base class but it's not relevant to MethodGrid
    seed: pd.PositiveInt = pd.Field(
        default=None,
        title="Seed for random number generation",
        description="Set the seed used by the optimizers to ensure consistant random number generation.",
    )

    def any_to_int_param(self, parameter: ParameterAny) -> dict:
        """Convert ParameterAny object to integers and provide a conversion dict to return"""

        return dict(enumerate(parameter.allowed_values))

    def sol_array_to_dict(
        self, solution: np.array, keys: list, param_converter: dict
    ) -> list[dict]:
        """Convert an array of solutions to a list of dicts for function input"""
        sol_dict_list = [dict(zip(keys, sol)) for sol in solution]

        self._handle_param_convert(param_converter, sol_dict_list)

        return sol_dict_list

    def _handle_param_convert(self, param_converter: dict, sol_dict_list: list[dict]) -> None:
        for param, convert in param_converter.items():
            for sol in sol_dict_list:
                if isinstance(sol[param], float):
                    sol[param] = int(round(sol[param], 0))
                sol[param] = convert[sol[param]]


class MethodBayOpt(MethodOptimize, ABC):
    """A standard method for performing a Bayesian optimization search, built around the `Bayesian Optimization <https://bayesian-optimization.github.io/BayesianOptimization/basic-tour.html>`_ package.
    The fitness function is maximising by default.

    Example
    -------
    >>> import tidy3d.plugins.design as tdd
    >>> method = tdd.MethodBayOpt(initial_iter=4, n_iter=10)
    """

    initial_iter: pd.PositiveInt = pd.Field(
        ...,
        title="Number of Initial Random Search Iterations",
        description="The number of search runs to be done initialially with parameter values picked randomly. This provides a starting point for the Gaussian processor to optimize from. These solutions can be computed as a single ``Batch`` if the pre function generates ``Simulation`` objects.",
    )

    n_iter: pd.PositiveInt = pd.Field(
        ...,
        title="Number of Bayesian Optimization Iterations",
        description="Following the initial search, this is number of iterations the Gaussian processor should be sequentially called to suggest parameter values and register the results.",
    )

    acq_func: Literal["ucb", "ei", "poi"] = pd.Field(
        default="ucb",
        title="Type of Acquisition Function",
        description="The type of acquisition function that should be used to suggest parameter values. More detail available in the `package docs <https://bayesian-optimization.github.io/BayesianOptimization/exploitation_vs_exploration.html>`_.",
    )

    kappa: pd.PositiveFloat = pd.Field(
        default=2.5,
        title="Kappa",
        description="The kappa coefficient used by the ``ucb`` acquisition function. More detail available in the `package docs <https://bayesian-optimization.github.io/BayesianOptimization/exploitation_vs_exploration.html>`_.",
    )

    xi: pd.NonNegativeFloat = pd.Field(
        default=0.0,
        title="Xi",
        description="The Xi coefficient used by the ``ei`` and ``poi`` acquisition functions. More detail available in the `package docs <https://bayesian-optimization.github.io/BayesianOptimization/exploitation_vs_exploration.html>`_.",
    )

    def _get_run_count(self, parameters: list = None) -> int:
        """Return the maximum number of runs for the method based on current method arguments."""
        return self.initial_iter + self.n_iter

    def _run(self, parameters: Tuple[ParameterType, ...], run_fn: Callable, console) -> Tuple[Any]:
        """Defines the Bayesian optimization search algorithm for the method.

        Uses the ``bayes_opt`` package to carry out a Bayesian optimization. Utilizes the ``.suggest`` and ``.register`` methods instead of
        the ``BayesianOptimization`` helper class as this allows more control over batching and preprocessing.
        More details of the package can be found `here <https://bayesian-optimization.github.io/BayesianOptimization/basic-tour.html>`_.
        """
        try:
            from bayes_opt import BayesianOptimization, UtilityFunction
        except ImportError:
            raise ImportError(
                "Cannot run Bayesian optimization as 'bayes_opt' module not found. "
                "Please check installation or run 'pip install bayesian-optimization==1.5.1'."
            )

        # Identify non-numeric params and define boundaries for Bay-opt
        param_converter = {}
        boundary_dict = {}
        for param in parameters:
            if isinstance(param, ParameterAny):
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

        # Initialize optimizer and utility function. Carry out optimization manually instead of using helper for more control
        utility = UtilityFunction(kind=self.acq_func, kappa=self.kappa, xi=self.xi)
        opt = BayesianOptimization(
            f=run_fn, pbounds=boundary_dict, random_state=self.seed, allow_duplicate_points=True
        )

        # Build the initial starting set of solutions - randomly chosen and batched together
        arg_list = []
        total_aux_out = []
        result = []
        for _ in range(self.initial_iter):
            next_point = opt.suggest(utility)
            self._force_int(next_point, parameters)
            self._handle_param_convert(param_converter, [next_point])
            arg_list.append(next_point)

        # Compute batch
        init_output, aux_out = self._extract_output(run_fn(arg_list))
        self._flatten_and_append(aux_out, total_aux_out)

        # Update the sampler with results from random initial solutions
        for next_point, next_out in zip(arg_list, init_output):
            result.append(next_out)
            self._handle_param_convert(invert_param_converter, [next_point])
            opt.register(params=next_point, target=next_out)

        best_fit = max(init_output)

        if console is not None:
            console.log(f"Best Fit from Initial Solutions: {round(best_fit, 3)}\n")

        # Handle further iterations sequentially
        # BayOpt package does not allow for batched non-random predictions
        for iter_num in range(self.n_iter):
            next_point = opt.suggest(utility)
            self._force_int(next_point, parameters)
            self._handle_param_convert(param_converter, [next_point])

            next_out, aux_out = self._extract_output(run_fn([next_point]))
            result.append(next_out[0])
            self._flatten_and_append(aux_out, total_aux_out)

            # ParameterAny values in next_point need to be converted back for the optimizer
            self._handle_param_convert(invert_param_converter, [next_point])
            opt.register(params=next_point, target=next_out[0])

            if next_out[0] > best_fit:
                best_fit = next_out[0]
                if console is not None:
                    console.log(f"Latest Best Fit on Iter {iter_num}: {round(best_fit, 3)}\n")

        if console is not None:
            console.log(
                f"Best Result: {opt.max['target']}\n"
                f"Best Parameters: {' '.join([f'{param}: {val}' for param, val in opt.max['params'].items()])}\n"
            )

        # Output fn_args from the BO.opt object - getting results in situ as opt changes type to float
        fn_args = []
        for output in opt.res:
            self._handle_param_convert(param_converter, [output["params"]])
            fn_args.append(output["params"])

        return fn_args, result, total_aux_out, opt


class MethodGenAlg(MethodOptimize, ABC):
    """A standard method for performing genetic algorithm search, built around the `PyGAD <https://pygad.readthedocs.io/en/latest/index.html>`_ package.
    The fitness function is maximising by default.

    Example
    -------
    >>> import tidy3d.plugins.design as tdd
    >>> method = tdd.MethodGenAlg(solutions_per_pop=2, n_generations=1, n_parents_mating=2)
    """

    # Args for the user
    solutions_per_pop: pd.PositiveInt = pd.Field(
        ...,
        title="Solutions per Population",
        description="The number of solutions to be generated for each population.",
    )

    n_generations: pd.PositiveInt = pd.Field(
        ...,
        title="Number of Generations",
        description="The maximum number of generations to run the genetic algorithm.",
    )

    n_parents_mating: pd.PositiveInt = pd.Field(
        ...,
        title="Number of Parents Mating",
        description="The number of solutions to be selected as parents for the next generation. Crossovers of these parents will produce the next population.",
    )

    stop_criteria_type: Literal["reach", "saturate"] = pd.Field(
        default=None,
        title="Early Stopping Criteria Type",
        description="Define the early stopping criteria. Supported words are 'reach' or 'saturate'. 'reach' stops at a desired fitness, 'saturate' stops when the fitness stops improving. Must set ``stop_criteria_number``. See the `PyGAD docs <https://pygad.readthedocs.io/en/latest/pygad.html>`_ for more details.",
    )

    stop_criteria_number: pd.PositiveFloat = pd.Field(
        default=None,
        title="Early Stopping Criteria Number",
        description="Must set ``stop_criteria_type``. If type is 'reach' the number is acceptable fitness value to stop the optimization. If type is 'saturate' the number is the number generations where the fitness doesn't improve before optimization is stopped. See the `PyGAD docs <https://pygad.readthedocs.io/en/latest/pygad.html>`_ for more details.",
    )

    parent_selection_type: Literal["sss", "rws", "sus", "rank", "random", "tournament"] = pd.Field(
        default="sss",
        title="Parent Selection Type",
        description="The style of parent selector. See the `PyGAD docs <https://pygad.readthedocs.io/en/latest/pygad.html>`_ for more details.",
    )

    keep_parents: Union[pd.PositiveInt, Literal[-1, 0]] = pd.Field(
        default=-1,
        title="Keep Parents",
        description="The number of parents to keep unaltered in the population of the next generation. Default value of -1 keeps all current parents for the next generation. This value is overwritten if ``keep_parents`` is > 0. See the `PyGAD docs <https://pygad.readthedocs.io/en/latest/pygad.html>`_ for more details.",
    )

    keep_elitism: Union[pd.PositiveInt, Literal[0]] = pd.Field(
        default=1,
        title="Keep Elitism",
        description="The number of top solutions to be included in the population of the next generation. Overwrites ``keep_parents`` if value is > 0. See the `PyGAD docs <https://pygad.readthedocs.io/en/latest/pygad.html>`_ for more details.",
    )

    crossover_type: Union[None, Literal["single_point", "two_points", "uniform", "scattered"]] = (
        pd.Field(
            default="single_point",
            title="Crossover Type",
            description="The style of crossover operation. See the `PyGAD docs <https://pygad.readthedocs.io/en/latest/pygad.html>`_ for more details.",
        )
    )

    crossover_prob: pd.confloat(ge=0, le=1) = pd.Field(
        default=0.8,
        title="Crossover Probability",
        description="The probability of performing a crossover between two parents.",
    )

    mutation_type: Union[None, Literal["random", "swap", "inversion", "scramble", "adaptive"]] = (
        pd.Field(
            default="random",
            title="Mutation Type",
            description="The style of gene mutation. See the `PyGAD docs <https://pygad.readthedocs.io/en/latest/pygad.html>`_ for more details.",
        )
    )

    mutation_prob: Union[pd.confloat(ge=0, le=1), Literal[None]] = pd.Field(
        default=0.2,
        title="Mutation Probability",
        description="The probability of mutating a gene.",
    )

    save_solution: pd.StrictBool = pd.Field(
        default=False,
        title="Save Solutions",
        description="Save all solutions from all generations within a numpy array. Can be accessed from the optimizer object stored in the Result. May cause memory issues with large populations or many generations. See the `PyGAD docs <https://pygad.readthedocs.io/en/latest/pygad.html>_` for more details.",
    )

    # TODO: See if anyone is interested in having the full suite of PyGAD options - there's a lot!

    def _get_run_count(self, parameters: list = None) -> int:
        """Return the maximum number of runs for the method based on current method arguments."""
        # +1 to generations as pygad creates an initial population which is effectively "Generation 0"
        run_count = self.solutions_per_pop * (self.n_generations + 1)
        return run_count

    def _run(self, parameters: Tuple[ParameterType, ...], run_fn: Callable, console) -> Tuple[Any]:
        """Defines the genetic algorithm for the method.

        Uses the ``pygad`` package to carry out a particle search optimization. Additional development has ensured that
        previously suggested solutions are not repeatedly computed, and that all computed solutions are captured.
        More details of the package can be found `here <https://pygad.readthedocs.io/en/latest/index.html>`_.
        """
        try:
            import pygad
        except ImportError:
            raise ImportError(
                "Cannot run genetic algorithm optimization as 'pygad' module not found. Please check installation or run 'pip install pygad'."
            )

        # Make param names available to the fitness function
        param_keys = [param.name for param in parameters]

        # Store parameters and fitness
        store_parameters = []
        store_fitness = []
        store_aux = []
        previous_solutions = {}

        # Set gene_spaces to keep GA within ranges
        param_converter = {}
        gene_spaces = []
        gene_types = []
        for param in parameters:
            if isinstance(param, ParameterFloat):
                gene_spaces.append({"low": param.span[0], "high": param.span[1]})
                gene_types.append(float)
            elif isinstance(param, ParameterInt):
                gene_spaces.append(
                    range(param.span[0], param.span[1] + 1)
                )  # +1 included so as to be inclusive of upper range value
                gene_types.append(int)
            else:
                # Designed for str in ParameterAny but may work for anything
                param_converter[param.name] = self.any_to_int_param(param)

                gene_spaces.append(range(0, len(param.allowed_values)))
                gene_types.append(int)

        def capture_aux(sol_dict_list: list[dict]) -> None:
            """Store the aux data by pulling from previous_solutions."""
            aux_out = []
            for sol in sol_dict_list:
                composite_key = str(sol.keys()) + str(sol.values())
                _, aux_data = previous_solutions[composite_key]
                aux_out.append(aux_data)

            self._flatten_and_append(aux_out, store_aux)

        # Create fitness function combining pre and post fn with the tidy3d call
        def fitness_function(ga_instance: pygad.GA, solution: np.array, solution_idx) -> dict:
            """Fitness function for GA. Format of inputs cannot be changed."""
            # Break solution down to list of dict
            sol_dict_list = self.sol_array_to_dict(solution, param_keys, param_converter)

            # Check if solution already exists
            # Have to update the solutions as need to pass to run_fn together to be batched
            known_sol = {}
            unknown_sol = []
            unknown_keys = []
            for sol_idx, sol in enumerate(sol_dict_list):
                composite_key = str(sol.keys()) + str(sol.values())

                if composite_key in previous_solutions:
                    known_sol[sol_idx] = composite_key
                else:
                    # Catches when PyGAD proposes the same solution multiple times in the same generation
                    if composite_key in unknown_keys:
                        # This keeps known_sol ordered correctly so that it inserts at the correct idx
                        known_sol[sol_idx] = (
                            composite_key  # We'll fill the result once it's been run the first time
                        )

                    # For totally unknown solutions
                    else:
                        unknown_sol.append(sol)
                        unknown_keys.append(composite_key)

            # Run unknown solutions
            sol_out, aux_out = self._extract_output(run_fn(unknown_sol))

            # Add unknown solution results to previous_solutions
            for composite_key, sol_result, aux_result in zip(unknown_keys, sol_out, aux_out):
                previous_solutions[composite_key] = (sol_result, aux_result)

            # Add known_sol to sol_out - order must be preserved
            # Aux data is added later as PyGAD will add repeat solutions to the population
            for sol_idx, composite_key in known_sol.items():
                sol, _ = previous_solutions[composite_key]
                sol_out.insert(sol_idx, sol)

            return sol_out

        def on_generation(ga_instance: pygad.GA) -> None:
            """Additional code run after every generation. Format of input cannot be changed"""
            sol_dict_list = self.sol_array_to_dict(
                ga_instance.population.copy(), param_keys, param_converter
            )
            capture_aux(sol_dict_list)

            store_parameters.extend(sol_dict_list)
            store_fitness.append(ga_instance.last_generation_fitness)

            # Report generation progress
            if console is not None:
                best_fitness = ga_instance.best_solution()[1]
                console.log(
                    f"Generation {ga_instance.generations_completed} Best Fitness: {best_fitness:.3f}"
                )

        # Build stop criteria string - check if both stop criteria fields have been set
        if any(val is not None for val in (self.stop_criteria_type, self.stop_criteria_number)):
            if all(val is not None for val in (self.stop_criteria_type, self.stop_criteria_number)):
                stop_criteria = f"{self.stop_criteria_type}_{self.stop_criteria_number}"
            else:
                raise ValueError(
                    "Both 'stop_criteria_type' and 'stop_criteria_number' fields need to be set to define the GA stop criteria."
                )
        else:
            stop_criteria = None

        # Determine initial array
        num_genes = len(parameters)

        # PyGAD doesn't store the initial population fitness - this captures parameters, fitness and aux data
        init_state = []

        def capture_init_pop_fitness(ga_instance: pygad.GA, population_fitness) -> None:
            """Store the initial population fitness which PyGAD otherwise ignores

            Has to be run ``on_fitness`` but contains a check so that it only runs on the first pass
            """
            # Have to check len of list instead of bool as can't pass any args into this func, or capture a return
            if not len(init_state):
                sol_dict_list = self.sol_array_to_dict(
                    ga_instance.initial_population.copy(), param_keys, param_converter
                )
                store_parameters.extend(sol_dict_list)
                store_fitness.append(population_fitness)
                init_state.append("Stored Init")

                capture_aux(sol_dict_list)

        # Define the optimizer
        ga_instance = pygad.GA(
            on_fitness=capture_init_pop_fitness,
            num_generations=self.n_generations,
            num_parents_mating=self.n_parents_mating,
            fitness_func=fitness_function,
            parent_selection_type=self.parent_selection_type,
            keep_parents=self.keep_parents,
            mutation_type=self.mutation_type,
            mutation_probability=self.mutation_prob,
            crossover_type=self.crossover_type,
            crossover_probability=self.crossover_prob,
            sol_per_pop=self.solutions_per_pop,
            num_genes=num_genes,
            fitness_batch_size=self.solutions_per_pop,
            on_generation=on_generation,
            random_seed=self.seed,
            gene_space=gene_spaces,
            gene_type=gene_types,
            stop_criteria=stop_criteria,
            save_solutions=self.save_solution,
            suppress_warnings=True,  # Used to prevent delay_on_generation depreciation warning for PyGAD 3.3.0
        )

        ga_instance.run()

        if console is not None:
            solution, solution_fitness, _ = ga_instance.best_solution()
            console.log(
                f"Best Result: {solution_fitness}\n"
                f"Best Parameters: {' '.join([f'{param.name}: {value}' for param, value in zip(parameters, solution)])}\n"
            )

        # Format output
        fn_args = store_parameters
        results = [val for arr in store_fitness for val in arr]

        return fn_args, results, store_aux, ga_instance


class MethodParticleSwarm(MethodOptimize, ABC):
    """A standard method for performing particle swarm search, build around the `PySwarms <https://pyswarms.readthedocs.io/en/latest/index.html>`_ package.
    The fitness function is maximising by default.

    Example
    -------
    >>> import tidy3d.plugins.design as tdd
    >>> method = tdd.MethodParticleSwarm(n_particles=5, n_iter=3)
    """

    n_particles: pd.PositiveInt = pd.Field(
        ...,
        title="Number of Particles",
        description="The number of particles to be used in the swarm for the optimization.",
    )

    n_iter: pd.PositiveInt = pd.Field(
        ...,
        title="Number of Iterations",
        description="The maxmium number of iterations to run the optimization.",
    )

    cognitive_coeff: pd.PositiveFloat = pd.Field(
        default=1.5,
        title="Cognitive Coefficient",
        description="The cognitive parameter decides how attracted the particle is to its previous best position.",
    )

    social_coeff: pd.PositiveFloat = pd.Field(
        default=1.5,
        title="Social Coefficient",
        description="The social parameter decides how attracted the particle is to the global best position found by the swarm.",
    )

    weight: pd.PositiveFloat = pd.Field(
        default=0.9,
        title="Weight",
        description="The weight or inertia of particles in the optimization.",
    )

    ftol: Union[pd.confloat(ge=0, le=1), Literal[-inf]] = pd.Field(
        default=-inf,
        title="Relative Error for Convergence",
        description="Relative error in ``objective_func(best_solution)`` acceptable for convergence. See the `PySwarms docs <https://pyswarms.readthedocs.io/en/latest/examples/tutorials/tolerance.html>`_ for details. Off by default.",
    )

    ftol_iter: pd.PositiveInt = pd.Field(
        default=1,
        title="Number of Iterations Before Convergence",
        description="Number of iterations over which the relative error in the objective_func is acceptable for convergence.",
    )

    init_pos: np.ndarray = pd.Field(
        default=None,
        title="Initial Swarm Positions",
        description="Set the initial positions of the swarm using a numpy array of appropriate size.",
    )

    def _get_run_count(self, parameters: list = None) -> int:
        """Return the maximum number of runs for the method based on current method arguments."""
        return self.n_particles * self.n_iter

    def _run(self, parameters: Tuple[ParameterType, ...], run_fn: Callable, console) -> Tuple[Any]:
        """Defines the particle search optimization algorithm for the method.

        Uses the ``pyswarms`` package to carry out a particle search optimization.
        More details of the package can be found `here <https://pyswarms.readthedocs.io/en/latest/index.html>`_.
        """
        try:
            from pyswarms.single.global_best import GlobalBestPSO
        except ImportError:
            raise ImportError(
                "Cannot run particle swarm optimization as 'pyswarms' module not found. Please check installation or run 'pip install pyswarms'."
            )

        # Pyswarms doesn't have a seed set outside of numpy std method
        if self.seed is not None:
            np.random.seed(self.seed)

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

        def fitness_function(solution: np.array) -> np.array:
            """Fitness function for PSO. Input format cannot be changed"""
            # Correct solutions that should be ints
            sol_dict_list = self.sol_array_to_dict(solution, param_keys, param_converter)
            for arg_dict in sol_dict_list:
                self._force_int(arg_dict, parameters)

            store_parameters.append(sol_dict_list)

            sol_out, aux_out = self._extract_output(run_fn(sol_dict_list))

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
            init_pos=self.init_pos,
            # TODO: including oh_strategy would be nice but complicated to specify with pydantic oh_strategy={"w": "exp_decay"},
        )

        opt_out = optimizer.optimize(fitness_function, self.n_iter, verbose=True)

        if console is not None:
            console.log(
                f"Best Result: {opt_out[0]}\n"
                f"Best Parameters: {' '.join([f'{param.name}: {value}' for param, value in zip(parameters, opt_out[1])])}\n"
            )

        # Collapse stores into fn_args and results lists
        fn_args = [val for sublist in store_parameters for val in sublist]
        results = [val for sublist in store_fitness for val in sublist]

        return fn_args, results, store_aux, optimizer


class AbstractMethodRandom(MethodSample, ABC):
    """Select parameters with an object with a ``random`` method."""

    num_points: pd.PositiveInt = pd.Field(
        ...,
        title="Number of Sampling Points",
        description="The number of points to be generated for sampling.",
    )

    seed: pd.PositiveInt = pd.Field(
        default=None,
        title="Seed",
        description="Sets the seed used by the optimizers to set constant random number generation.",
    )

    @abstractmethod
    def _get_sampler(self, parameters: Tuple[ParameterType, ...]) -> qmc.QMCEngine:
        """Sampler for this ``Method`` class. If ``None``, sets a default."""

    def _get_run_count(self, parameters: list = None) -> int:
        """Return the maximum number of runs for the method based on current method arguments."""
        return self.num_points

    def sample(self, parameters: Tuple[ParameterType, ...], **kwargs) -> Dict[str, Any]:
        """Defines how the design parameters are sampled on grid."""

        sampler = self._get_sampler(parameters)
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
    """Select sampling points using Monte Carlo sampling.
    The sampling is done with the Latin Hypercube method from scipy.

    Example
    -------
    >>> import tidy3d.plugins.design as tdd
    >>> method = tdd.MethodMonteCarlo(num_points=20)
    """

    def _get_sampler(self, parameters: Tuple[ParameterType, ...]) -> qmc.QMCEngine:
        """Sampler for this ``Method`` class."""

        d = len(parameters)
        return DEFAULT_MONTE_CARLO_SAMPLER_TYPE(d=d, seed=self.seed)


MethodType = Union[
    MethodMonteCarlo,
    MethodGrid,
    MethodBayOpt,
    MethodGenAlg,
    MethodParticleSwarm,
]
