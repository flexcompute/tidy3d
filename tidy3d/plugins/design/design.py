"""Defines design space specification for tidy3d."""

from __future__ import annotations

import inspect
from typing import Any, Callable, Dict, List, Tuple, Union

import pydantic.v1 as pd

from ...components.base import Tidy3dBaseModel, cached_property
from ...components.data.sim_data import SimulationData
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

        fn_args_coords = tuple(fn_args.values())

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

    def run(self, function: Callable, **kwargs) -> Result:
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

        # run the function from the method
        fn_args, fn_values = self.method.run(parameters=self.parameters, fn=function)

        fn_source = self.get_fn_source(function)

        # package the result
        return self._package_run_results(fn_args=fn_args, fn_values=fn_values, fn_source=fn_source)

    @staticmethod
    def _make_batch_fn_source(fn_source_pre: str, fn_source_post: str) -> str:
        """How to make the full function source from the pre and post functions."""

        if (fn_source_pre is None) and (fn_source_post is None):
            return None

        return str(fn_source_pre) + "\n\n" + str(fn_source_post)

    def run_batch(
        self,
        fn_pre: Callable[Any, Union[Simulation, List[Simulation], Dict[str, Simulation]]],
        fn_post: Callable[
            Union[SimulationData, List[SimulationData], Dict[str, SimulationData]], Any
        ],
        path_dir: str = None,
        **batch_kwargs,
    ) -> Result:
        """Run a design problem where the function is split into pre and post processing steps.

        Parameters
        ----------
        fn_pre : Callable[Any, Union[Simulation, List[Simulation]]]
            Function accepting arguments that correspond to the ``.name`` fields
            of the ``DesignSpace.parameters``. Returns either a :class:`.Simulation`,
            `list` of :class:`.Simulation`s or a `dict` of :class:`.Simulation`s to be run
            in a batch.
        fn_pre : Callable[Union[SimulationData, List[SimulationData], Dict[str, SimulationData]], Any]
            Function accepting the :class:`.SimulationData` object(s) corresponding to the
            ``fn_pre`` and returning the result of the parameter sweep function.
            If ``fn_pre`` returns a single simulation, it will be passed as a single argument to
            ``fn_post``.
            If ``fn_pre`` returns a list of simulations, their data will be passed as ``*args``.
            If ``fn_pre`` returns a dict of simulations, their data will be passed as ``**kwargs``
            with the keys corresponding to the argument names.
        path_dir : str = None
            Optional directory in which to store the batch results.

        Returns
        -------
        :class:`.Result`
            Object containing the results of the design space exploration.
            Can be converted to ``pandas.DataFrame`` with ``.to_dataframe()``.
        """

        # run the functions using the `method.run_batch`
        fn_args, fn_values, task_ids, batch_data = self.method.run_batch(
            parameters=self.parameters,
            fn_pre=fn_pre,
            fn_post=fn_post,
            path_dir=path_dir,
            **batch_kwargs,
        )

        # store the pre and post functions in a single source code
        fn_source_pre = self.get_fn_source(fn_pre)
        fn_source_post = self.get_fn_source(fn_post)
        fn_source = self._make_batch_fn_source(fn_source_pre, fn_source_post)

        # package the result
        return self._package_run_results(
            fn_args=fn_args,
            fn_values=fn_values,
            fn_source=fn_source,
            task_ids=task_ids,
            batch_data=batch_data,
        )
