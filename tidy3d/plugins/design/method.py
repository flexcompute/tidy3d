"""Defines the methods used for parameter sweep."""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Literal, Optional, Tuple, Union

import numpy as np
import pydantic.v1 as pd
import scipy.stats.qmc as qmc
from bayes_opt import BayesianOptimization, UtilityFunction
from bayes_opt.event import Events
from bayes_opt.logger import JSONLogger

from ... import web
from ...components.base import Tidy3dBaseModel
from ...components.simulation import Simulation
from ...log import log
from ...web.api.container import BatchData
from .parameter import ParameterType

DEFAULT_MONTE_CARLO_SAMPLER_TYPE = qmc.LatinHypercube


class Method(Tidy3dBaseModel, ABC):
    """Spec for a sweep algorithm, with a method to run it."""

    name: str = pd.Field(None, title="Name", description="Optional name for the sweep method.")

    batch_size: pd.PositiveInt = pd.Field(
        ...,
        title="Size of batch to be run",
        description="TBD",
    )

    num_batches: pd.PositiveInt = pd.Field(
        ...,
        title="Number of batches to be run",
        description="TBD",
    )

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

    def _eval_run(self, fn_args: Dict[str, tuple], pre_fn, run_loc):
        """Decide whether this is a local, single online, or batch online job"""

        if run_loc == "local":
            result = []
            for arg_dict in fn_args:
                fn_output = pre_fn(**arg_dict)
                result.append(fn_output)

            return result
        else:
            print("Tidy3d coming soon")

    def _tidy3d_run(self):
        """Generic logic for running a pre_fn on tidy3d servers"""

        # Maybe it's just a check for the number of simulations given?
        # Simulation building done by individual runs / submethod methods and then given to this

        # If something given/determined shows only one thing to be run
        web.run()

        # Elif if is clearly to be run as batches
        batch = web.Batch()
        batch.run()

    @staticmethod
    def _check_pre_output(self, parameters: Tuple[ParameterType, ...], pre_fn: Callable):
        """See if pre produces an arrangement of td.Simulation or is unrelated"""

        # Run fn_pre with the lower bound of each parameter
        lower_params = {param.name: param.span[0] for param in parameters}
        result = pre_fn(**lower_params)

        # Determine if pre_fn will need to run locally or on tidy3d
        if isinstance(result, Simulation):  # Single Simulation
            run_loc = "tidy3d"

        elif isinstance(result, dict):  # Dict of simulations
            if isinstance(result[next(iter(result))], Simulation):
                run_loc = "tidy3d"

        elif isinstance(result, list):  # List of dicts of simulations
            if isinstance(result[0], dict):
                if isinstance(result[0][next(iter(result[0]))], Simulation):
                    run_loc = "tidy3d"

        else:
            print("Not a recognised way of organising Tidy3D simulation objects")
            run_loc = "local"

        return run_loc


class MethodSample(Method, ABC):
    """A sweep method where all points are independently computed."""

    num_points: pd.PositiveInt = pd.Field(
        ...,
        title="Number of points for sampling",
        description="TBD",
    )

    @abstractmethod
    def sample(self, parameters: Tuple[ParameterType, ...], **kwargs) -> Dict[str, Any]:
        """Defines how the design parameters are sampled."""

    def _assemble_args(
        self,
        parameters: Tuple[ParameterType, ...],
        pre_fn: Callable,
    ) -> Tuple[dict, int]:
        """Sample design parameters, check the args are hashable and compute number of points."""

        fn_args = self.sample(parameters)
        run_loc = self._check_pre_output(self, parameters, pre_fn)
        # self.assert_hashable(fn_args)
        # self.assert_num_points(fn_args)
        return fn_args, run_loc

    def run(
        self, parameters: Tuple[ParameterType, ...], pre_fn: Callable, post_fn: Callable
    ) -> Tuple[Any]:
        """Defines the search algorithm (sequential)."""

        # get all function inputs
        fn_args, run_loc = self._assemble_args(parameters, pre_fn)

        # for each point, construct the function inputs, run it, record output
        results = self._eval_run(fn_args, pre_fn, run_loc)

        # Post process the data
        processed_result = []
        for res in results:
            processed_result.append(post_fn(res))

        return fn_args, processed_result

    def _run_batch(
        self, simulations: Dict[str, Simulation], path_dir: str = None, **kwargs
    ) -> BatchData:
        """Create a batch of simulations and run it. Mainly separated out for ease of testing."""
        batch = web.Batch(simulations=simulations, simulation_type="tidy3d_design", **kwargs)

        if path_dir:
            run_kwargs = dict(path_dir=path_dir)
        else:
            run_kwargs = {}
        return batch.run(**run_kwargs)

    def run_batch(
        self,
        parameters: Tuple[ParameterType, ...],
        fn_pre: Callable,
        fn_post: Callable,
        path_dir: str = None,
        **batch_kwargs,
    ) -> Tuple[Any]:
        """Defines the search algorithm (batched)."""

        # get all function inputs
        fn_args = self._assemble_args(parameters)

        def get_task_name(pt_index: int, sim_index: int, fn_kwargs: dict) -> str:
            """Get task name for 'index'-th set of function kwargs."""
            try:
                kwarg_str = str(fn_kwargs)
                if sim_index is not None:
                    return f"{kwarg_str}_{sim_index}"
                return kwarg_str
            # just to be safe, handle the case if this does not work
            except ValueError:
                return f"{pt_index}_{sim_index}"

        # for each point, construct the simulation inputs into a dict
        simulations = {}
        task_name_mappings = []
        for i in range(fn_args):
            fn_kwargs = {key: vals[i] for key, vals in fn_args.items()}
            sim = fn_pre(**fn_kwargs)
            if isinstance(sim, Simulation):
                task_name = get_task_name(pt_index=i, sim_index=None, fn_kwargs=fn_kwargs)
                simulations[task_name] = sim
                task_name_mappings.append([task_name])
            elif isinstance(sim, dict):
                task_name_mappings.append({})
                for name, _sim in sim.items():
                    task_name = get_task_name(pt_index=1, sim_index=name, fn_kwargs=fn_kwargs)
                    simulations[task_name] = _sim
                    task_name_mappings[i][name] = task_name
            else:
                task_name_mappings.append([])
                for j, _sim in enumerate(sim):
                    task_name = get_task_name(pt_index=i, sim_index=j, fn_kwargs=fn_kwargs)
                    simulations[task_name] = _sim
                    task_name_mappings[i].append(task_name)

        # run in batch
        batch_data = self._run_batch(simulations=simulations, path_dir=path_dir, **batch_kwargs)
        task_id_dict = batch_data.task_ids

        # run post processing on each data
        result = []
        task_ids = []
        for task_names_i in task_name_mappings:
            if isinstance(task_names_i, dict):
                task_ids.append([task_id_dict[task_name] for task_name in task_names_i.values()])
                kwargs_dict = {
                    kwarg_name: batch_data[task_name]
                    for kwarg_name, task_name in task_names_i.items()
                }
                val = fn_post(**kwargs_dict)
            else:
                task_ids.append([task_id_dict[task_name] for task_name in task_names_i])
                args_list = (batch_data[task_name] for task_name in task_names_i)
                val = fn_post(*args_list)

            result.append(val)

        return fn_args, result, task_ids, batch_data


class MethodOptimise(Method, ABC):
    """A method for handling design searches that optimise the design"""

    def create_boundary_dict(
        self,
        parameters: Tuple[ParameterType, ...],
    ):
        """Reshape parameter spans to dict of boundaries"""

        return {design_var.name: design_var.span for design_var in parameters}


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
        return dict(zip(vals_each_dim.keys(), vals_grid))


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
        """Defines the search algorithm (sequential)."""

        boundary_dict = self.create_boundary_dict(parameters)
        run_loc = self._check_pre_output(self, parameters, pre_fn)

        # Fn can be defined here to be a combined func of pre, run_batch, post for BO to use
        utility = UtilityFunction(kind=self.acq_func, kappa=2.5, xi=0.0)
        opt = BayesianOptimization(
            f=pre_fn, pbounds=boundary_dict, random_state=1, allow_duplicate_points=True
        )

        logger = JSONLogger(path="./logs")
        opt.subscribe(Events.OPTIMIZATION_STEP, logger)

        # opt.maximize(
        #     init_points=self.initial_iter,
        #     n_iter=self.n_iter,
        #     acquisition_function=utility,
        # )

        # OR
        # Need method of running initial_iter random points
        # Need to iterate over n_iter points

        if run_loc == "local":
            for _ in range(self.n_iter):
                next_point = opt.suggest(utility)
                next_out = pre_fn(**next_point)
                opt.register(params=next_point, target=next_out)
        elif run_loc == "tidy3d":
            print("Tidy3d coming soon")

        # Output results from the BO.opt object
        result = []
        fn_args = []
        for output in opt.res:
            result.append(output["target"])
            fn_args.append(output["params"])

        return fn_args, result

    def run_batch(
        self,
        parameters: Tuple[ParameterType, ...],
        fn_pre: Callable,
        fn_post: Callable,
        path_dir: str = None,
        **batch_kwargs,
    ) -> Tuple[Any]:
        """Defines the search algorithm (batched)."""

        def get_task_name(pt_index: int, sim_index: int, fn_kwargs: dict) -> str:
            """Get task name for 'index'-th set of function kwargs."""
            try:
                kwarg_str = str(fn_kwargs)
                if sim_index is not None:
                    return f"{kwarg_str}_{sim_index}"
                return kwarg_str
            # just to be safe, handle the case if this does not work
            except ValueError:
                return f"{pt_index}_{sim_index}"


class AbstractMethodRandom(MethodSample, ABC):
    """Select parameters with an object with a ``random`` method."""

    @abstractmethod
    def get_sampler(self, parameters: Tuple[ParameterType, ...]) -> qmc.QMCEngine:
        """Sampler for this ``Method`` class. If ``None``, sets a default."""

    def sample(self, parameters: Tuple[ParameterType, ...], **kwargs) -> Dict[str, Any]:
        """Defines how the design parameters are sampled on grid."""

        sampler = self.get_sampler(parameters)
        pts_01 = sampler.random(self.num_points)

        # Get output list of kwargs for pre_fn
        keys = [param.name for param in parameters]
        result = [{keys[j]: row[j] for j in range(len(keys))} for row in pts_01]

        # for each dimension, sample `num_points` points and combine them all
        # result = {}
        # for i, design_var in enumerate(parameters):
        #     pts_i_01 = pts_01[..., i]
        #     values = design_var.select_from_01(pts_i_01)
        #     result[design_var.name] = values

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


MethodType = Union[MethodMonteCarlo, MethodGrid, MethodRandom, MethodRandomCustom, MethodBayOpt]
