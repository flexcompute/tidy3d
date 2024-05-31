"""Defines the methods used for parameter sweep."""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Tuple, Union

import numpy as np
import pydantic.v1 as pd
import scipy.stats.qmc as qmc

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

    @abstractmethod
    def run(self, parameters: Tuple[ParameterType, ...], fn: Callable) -> Tuple[Any]:
        """Defines the search algorithm (sequential)."""

    @abstractmethod
    def sample(self, parameters: Tuple[ParameterType, ...], **kwargs) -> Dict[str, Any]:
        """Defines how the design parameters are sampled."""

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


class MethodIndependent(Method, ABC):
    """A sweep method where all points are independently computed."""

    @staticmethod
    def get_num_points(fn_args: Dict[str, tuple]) -> int:
        """Compute number of points from the function arguments and do error checking."""
        num_points_each_dim = [len(val) for val in fn_args.values()]
        if len(set(num_points_each_dim)) != 1:
            raise ValueError(
                f"Found different number of points: {num_points_each_dim} along each dimension. "
                "This suggests a bug in the parameter sweep tool. "
                "Please raise an issue on the front end GitHub repository."
            )
        return num_points_each_dim[0]

    def _assemble_args(self, parameters: Tuple[ParameterType, ...]) -> Tuple[dict, int]:
        """Sample design parameters, check the args are hashable and compute number of points."""

        fn_args = self.sample(parameters)
        self.assert_hashable(fn_args)
        num_points = self.get_num_points(fn_args)
        return fn_args, num_points

    def run(self, parameters: Tuple[ParameterType, ...], fn: Callable) -> Tuple[Any]:
        """Defines the search algorithm (sequential)."""

        # get all function inputs
        fn_args, num_points = self._assemble_args(parameters)

        # for each point, construct the function inputs, run it, record output
        result = []
        for i in range(num_points):
            fn_kwargs = {key: vals[i] for key, vals in fn_args.items()}
            fn_output = fn(**fn_kwargs)
            result.append(fn_output)

        return fn_args, result

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
        fn_args, num_points = self._assemble_args(parameters)

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
        for i in range(num_points):
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


class MethodGrid(MethodIndependent):
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


class AbstractMethodRandom(MethodIndependent, ABC):
    """Select parameters with an object with a ``random`` method."""

    num_points: pd.PositiveInt = pd.Field(
        ...,
        title="Number of Points",
        description="Maximum number of sampling points to perform in the sweep.",
    )

    @abstractmethod
    def get_sampler(self, parameters: Tuple[ParameterType, ...]) -> qmc.QMCEngine:
        """Sampler for this ``Method`` class. If ``None``, sets a default."""

    def sample(self, parameters: Tuple[ParameterType, ...], **kwargs) -> Dict[str, Any]:
        """Defines how the design parameters are sampled on grid."""

        sampler = self.get_sampler(parameters)
        pts_01 = sampler.random(self.num_points)

        # for each dimension, sample `num_points` points and combine them all
        result = {}
        for i, design_var in enumerate(parameters):
            pts_i_01 = pts_01[..., i]
            values = design_var.select_from_01(pts_i_01)
            result[design_var.name] = values
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


MethodType = Union[MethodMonteCarlo, MethodGrid, MethodRandom, MethodRandomCustom]
