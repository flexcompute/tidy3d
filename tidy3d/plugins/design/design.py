"""Defines design space specification for tidy3d."""

from __future__ import annotations

import inspect
from collections import defaultdict
from typing import Any, Callable, Dict, Generator, List, Tuple, Union

import pydantic.v1 as pd

import tidy3d.web as web
from tidy3d.log import log

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

    task_name: str = pd.Field(
        None,
        title="Task Name",
        description="Task name assigned to tasks along with a simulation counter. Only used when pre-post functions are supplied.",
    )

    path_dir: str = pd.Field(
        ".",
        title="Path Directory",
        description="Directory for files to output to. Only used when pre-post functions are supplied.",
    )

    folder_name: str = pd.Field(
        "default",
        title="Folder Name",
        description="Folder path where the simulation will be uploaded in the Tidy3D Workspace. Will use 'default' if no path is set.",
    )

    _dict_name: str = pd.PrivateAttr(default=None)

    _task_names: list = pd.PrivateAttr(default=[])

    _name_generator: Generator = pd.PrivateAttr(default=None)

    @cached_property
    def dims(self) -> Tuple[str]:
        """dimensions defined by the design parameter names."""
        return tuple(param.name for param in self.parameters)

    def _package_run_results(
        self,
        fn_args: list[dict[str, Any]],
        fn_values: List[Any],
        fn_source: str,
        task_ids: Tuple[str] = None,
        aux_values: List[Any] = None,
    ) -> Result:
        """How to package results from ``method.run`` and ``method.run_batch``"""

        fn_args_coords = tuple(
            [[arg_dict[key] for arg_dict in fn_args] for key in fn_args[0].keys()]
        )

        fn_args_coords_T = list(map(list, zip(*fn_args_coords)))

        # Format fn_values as appropriate

        return Result(
            dims=self.dims,
            values=fn_values,
            coords=fn_args_coords_T,
            fn_source=fn_source,
            task_ids=task_ids,
            aux_values=aux_values,
        )

    def _create_name_generator(self):
        """Initialise the name generator used for simulation task names"""
        counter = 0

        while True:
            # Check if user has supplied a dict key
            if self._dict_name is not None:
                suffix = f"{self._dict_name}_{counter}"
            else:
                suffix = f"{counter}"

            if self.task_name is not None:
                task_name = f"{self.task_name}_{suffix}"
            else:
                task_name = f"{suffix}"

            yield (task_name)
            counter += 1

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

        self._name_generator = self._create_name_generator()
        next(self._name_generator)

        # Run based on how many functions the user provides
        if fn_post is None:
            fn_args, fn_values, aux_values = self.run_single(fn)

        else:
            fn_args, fn_values, aux_values = self.run_pre_post(fn_pre=fn, fn_post=fn_post)

        fn_source = self.get_fn_source(fn)

        # Package the result
        return self._package_run_results(
            fn_args=fn_args,
            fn_values=fn_values,
            fn_source=fn_source,
            aux_values=aux_values,
            task_ids=self._task_names,
        )

    def run_single(self, fn: Callable) -> Tuple(list[dict], list, list[Any]):
        """Run a single function of parameter inputs."""
        evaluate_fn = self._get_evaluate_fn_single(fn=fn)
        return self.method.run(run_fn=evaluate_fn, parameters=self.parameters)

    def run_pre_post(self, fn_pre: Callable, fn_post: Callable) -> Tuple(
        list[dict], list[dict], list[Any]
    ):
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
            data = self._fn_mid(sim_dict)
            post_out = [fn_post(val[1]) for val in data.items()]
            return post_out

        return fn_combined

    def _fn_mid(self, pre_out: dict[int, Any]) -> Union[dict[int, Any], BatchData]:
        """A function of the output of ``fn_pre`` that gives the input to ``fn_post``."""

        # New plan - element wise
        # Keep list coversion but no need to check for simulation
        # If no sims or batches found then dump all the work up and just return pre_out - for non-td problems
        # Don't unpack batches but allow for sequential running

        # Convert list of sim inputs to dict of dict before passing through checks
        was_list = False
        if all(isinstance(sim, list) for sim in pre_out.values()) and any(
            isinstance(sim, Simulation) for sim in pre_out[0]
        ):
            pre_out = {
                idx: dict(enumerate(sim_list)) for idx, sim_list in enumerate(pre_out.values())
            }
            was_list = True

        # Handle most common case where fn_combined builds a dict of sims
        if all(isinstance(sim, Simulation) for sim in pre_out.values()):
            named_sims = {next(self._name_generator): sim for sim in pre_out.values()}
            self._task_names.extend(list(named_sims.keys()))
            data = web.Batch(
                simulations=named_sims,
                folder_name=self.folder_name,
                simulation_type="tidy3d_design",
            ).run(path_dir=self.path_dir)

        # Can "index" dict here because fn_combined uses idx as the key
        elif all(isinstance(sim, Dict) for sim in pre_out.values()) and any(
            isinstance(sim, Simulation) for sim in pre_out[0].values()
        ):
            # Flatten dict of dicts whilst storing combined keys for later
            flattened_sims = {}
            original_structure = defaultdict(dict)
            dict_keys = []
            for dict_idx, sub_dict in pre_out.items():
                for sub_dict_idx, sim_like in sub_dict.items():
                    sub_dict_idx = str(sub_dict_idx)  # Needs to be string for restoration later
                    if isinstance(sim_like, Simulation):
                        task_name = f"{dict_idx}_{sub_dict_idx}"
                        flattened_sims[task_name] = sim_like
                        original_structure[dict_idx][sub_dict_idx] = None

                        # Add user specified keys to the naming if a dict was supplied
                        if not was_list:
                            dict_keys.append(sub_dict_idx)
                        else:
                            dict_keys.append(None)
                    else:
                        original_structure[dict_idx][sub_dict_idx] = sim_like

            # Run sims with flattened dict
            named_sims = {}
            for sim, key_name in zip(flattened_sims.values(), dict_keys):
                self._dict_name = key_name
                named_sims[next(self._name_generator)] = sim
            translation_dict = dict(zip(named_sims, flattened_sims))
            self._task_names.extend(list(named_sims.keys()))
            batch_out = web.Batch(
                simulations=named_sims,
                folder_name=self.folder_name,
                simulation_type="tidy3d_design",
            ).run(path_dir=self.path_dir)

            # Unflatten structure whilst running fn_post
            for task_id in batch_out.task_ids:
                search_key = translation_dict[task_id]
                dict_idx, sub_dict_idx = search_key.split("_", 1)
                original_structure[int(dict_idx)][sub_dict_idx] = batch_out[task_id]

            # Restore to lists if user supplied lists
            if was_list:
                data = {
                    dict_idx: list(sub_dict.values())
                    for dict_idx, sub_dict in original_structure.items()
                }
            else:
                data = original_structure

        else:
            # user just wants to split into pre and post, without tidy3d I guess
            data = pre_out

        return data

    def run_batch(
        self,
        fn_pre: Callable[Any, Union[Simulation, List[Simulation], Dict[str, Simulation]]],
        fn_post: Callable[
            Union[SimulationData, List[SimulationData], Dict[str, SimulationData]], Any
        ],
        path_dir: str = None,
        **batch_kwargs,
    ) -> Result:
        """
        This function has been superceded by `run`, please use `run` for batched simulations.
        """
        log.warning(
            "In version 2.8.0, the 'run_batch' method is replaced by 'run'."
            "While the original syntax will still be supported, future functionality will be added to the new method moving forward."
        )

        if len(batch_kwargs) > 0:
            log.warning(
                "Batch_kwargs supplied here will no longer be used for within the simulation."
            )

        new_self = self.updated_copy(path_dir=path_dir)
        result = new_self.run(fn=fn_pre, fn_post=fn_post)

        return result
