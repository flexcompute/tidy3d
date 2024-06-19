"""Defines design space specification for tidy3d."""

from __future__ import annotations

import inspect
from typing import Any, Callable, Dict, List, Tuple

import pydantic.v1 as pd

import tidy3d.web as web

from ...components.base import Tidy3dBaseModel, cached_property
from ...components.simulation import Simulation
from ...web.api.container import BatchData
from .method import MethodType
from .parameter import ParameterType
from .result import Result


class DesignSpace(Tidy3dBaseModel):
    """Specification of a design problem / combination of several parameters + algorithm.

    Example
    -------
    >>> import tidy3d.plugins.design as tdd
    >>> param = tdd.ParameterFloat(name="x", span=(0,1))
    >>> method = tdd.MethodMonteCarlo(num_points=10)
    >>> design_space = tdd.DesignSpace(parameters=[param], method=method)
    >>> fn = lambda x: x**2
    >>> result = design_space.run(fn)
    >>> df = result.to_dataframe()
    >>> im = df.plot()
    """

    parameters: Tuple[ParameterType, ...] = pd.Field(
        (),
        title="Parameters",
        description="Set of parameters defining the dimensions and allowed values for the design space.",
    )

    method: MethodType = pd.Field(
        ...,
        title="Search Type",
        description="Specifications for the procedure used to explore the parameter space.",
    )

    name: str = pd.Field(None, title="Name", description="Optional name for the design space.")

    @cached_property
    def dims(self) -> Tuple[str]:
        """dimensions defined by the design parameter names."""
        return tuple(param.name for param in self.parameters)

    @cached_property
    def design_parameter_dict(self) -> Dict[str, ParameterType]:
        """Mapping of design parameter name to design parameter."""
        return dict(zip(self.dims, self.parameters))

    def _package_run_results(
        self,
        fn_args: Dict[str, tuple],
        fn_values: List[Any],
        fn_source: str,
        task_ids: Tuple[str] = None,
        batch_data: BatchData = None,
    ) -> Result:
        """How to package results from ``method.run`` and ``method.run_batch``"""

        fn_args_coords = tuple(
            [[arg_dict[key] for arg_dict in fn_args] for key in fn_args[0].keys()]
        )

        fn_args_coords_T = list(map(list, zip(*fn_args_coords)))

        return Result(
            dims=self.dims,
            values=fn_values,
            coords=fn_args_coords_T,
            fn_source=fn_source,
            task_ids=task_ids,
            batch_data=batch_data,
        )

    @staticmethod
    def get_fn_source(function: Callable) -> str:
        """Get the function source as a string, return ``None`` if not available."""
        try:
            return inspect.getsource(function)
        except (TypeError, OSError):
            return None

    def run(self, fn: Callable, fn_post: Callable = None, **kwargs) -> Result:
        """Run the design problem on a user defined function of the design parameters.

        Parameters
        ----------
        function : Callable
            Function accepting arguments that correspond to the ``.name`` fields
            of the ``DesignSpace.parameters``.

        Returns
        -------
        :class:`.Result`
            Object containing the results of the design space exploration.
            Can be converted to ``pandas.DataFrame`` with ``.to_dataframe()``.
        """

        # Run based on how many functions the user provides
        if fn_post is None:
            fn_args, fn_values = self.run_single(fn)

        else:
            fn_args, fn_values = self.run_pre_post(fn_pre=fn, fn_post=fn_post)

        fn_source = self.get_fn_source(fn)

        # Package the result
        return self._package_run_results(fn_args=fn_args, fn_values=fn_values, fn_source=fn_source)

    def run_single(self, fn: Callable):
        """Run a single function of parameter inputs."""
        evaluate_fn = self._get_evaluate_fn_single(fn=fn)
        return self.method.run(run_fn=evaluate_fn, parameters=self.parameters)

    def run_pre_post(self, fn_pre: Callable, fn_post: Callable):
        """Run a function with tidy3d implicitly called in between."""
        evaluate_fn = self._get_evaluate_fn_pre_post(fn_pre=fn_pre, fn_post=fn_post)
        return self.method.run(run_fn=evaluate_fn, parameters=self.parameters)

    """ Helpers """

    def _get_evaluate_fn_single(self, fn: Callable) -> list[Any]:
        """Get function that sequentially evaluates single `fn` for a list of arguments."""

        def evaluate(args_list: list) -> list[Any]:
            """Evaluate a list of arguments passed to ``fn``."""
            return [fn(**args) for args in args_list]

        return evaluate

    def _get_evaluate_fn_pre_post(self, fn_pre: Callable, fn_post: Callable) -> list[Any]:
        """Get function that tries to use batch processing on a set of arguments."""

        # def fn_pre_batched(args_list: list) -> dict[str, Any]:
        #     """Generate inputs."""
        #     return {fn_pre(**args) for args in args_list}

        # def fn_post_batch(data_dict_list: dict[str, Any]) -> list[Any]:
        #     """Evaluate the outputs."""
        #     return [fn_post(val) for val in data_dict_list.values()]

        def evaluate(args_list: list[tuple[Any, ...]]) -> list[Any]:
            """Put together into one pipeline."""
            combined_fn = self._stitch_pre_post(fn_pre=fn_pre, fn_post=fn_post)
            out = combined_fn(args_list)
            return out

        return evaluate

    def _stitch_pre_post(self, fn_pre: Callable, fn_post: Callable) -> Callable:
        """Combine pre and post into one big function, stitching tidy3d calls between if needed."""

        def fn_combined(args_list):
            sim_dict = {idx: fn_pre(**arg_list) for idx, arg_list in enumerate(args_list)}
            data = self.fn_mid(sim_dict)
            post_out = [fn_post(val) for val in data]
            return post_out

        return fn_combined

    @staticmethod
    def fn_mid(pre_out: Any) -> Any:
        """A function of the output of ``fn_pre`` that gives the input to ``fn_post``."""
        # if isinstance(pre_out, Simulation):
        #     data = web.run(pre_out)
        # elif isinstance(pre_out, list) and all(isinstance(sim, Simulation) for sim in pre_out):
        #     data = web.Batch({idx: sim for idx, sim in enumerate(pre_out)}).run()
        #     data = list(data.values())
        if isinstance(pre_out, dict) and all(
            isinstance(sim, Simulation) for sim in pre_out.values()
        ):
            data = web.Batch(simulations=pre_out).run()
            data = [sim_tuple[1] for sim_tuple in data.items()]
        # TODO: eventually handed nested dict of list/tuple of Sims...
        else:
            # user just wants to split into pre and post, without tidy3d I guess
            data = list(pre_out.values())
            # or we just error
            # raise ValueError(f'Bad outputs from "fn_pre", cant run')

        return data
