"""Defines design space specification for tidy3d."""

from __future__ import annotations

import dataclasses
import gc
import inspect
from typing import TYPE_CHECKING, Any, get_args

from pydantic import Field, model_validator

from tidy3d.components.base import Tidy3dBaseModel, cached_property
from tidy3d.components.types import TYPE_TAG_STR
from tidy3d.components.types.workflow import WorkflowDataType, WorkflowOperationType
from tidy3d.log import get_logging_console, log
from tidy3d.web.api.container import Batch, Job

from .method import (
    MethodBayOpt,
    MethodGenAlg,
    MethodOptimize,
    MethodParticleSwarm,
    MethodSample,
    MethodType,
)
from .parameter import ParameterAny, ParameterInt, ParameterType
from .result import Result

if TYPE_CHECKING:
    from collections.abc import Callable

    from tidy3d.log import Console
    from tidy3d.web.api.container import BatchData

WORKFLOW_TYPES = get_args(WorkflowOperationType)
WORKFLOW_DATA_TYPES = get_args(WorkflowDataType)


@dataclasses.dataclass
class _FnMidResult:
    """Intermediate fn_pre execution data passed into fn_post."""

    data: dict[int, Any]
    task_names: list[Any]
    task_ids: list[Any]
    task_paths: list[Any]
    sim_counter: int
    post_input_loader: Callable[[str, Any], Any] | None = None


class DesignSpace(Tidy3dBaseModel):
    """Manages all exploration of a parameter space within specified parameters using a supplied search method.
    The ``DesignSpace`` forms the basis of the ``Design`` plugin, and receives a ``Method`` and ``Parameter`` list that
    define the scope of the design space and how it should be searched. ``DesignSpace.run()`` can then be called with
    a function(s) to generate different solutions from parameters suggested by the ``Method``. The ``Method`` can either
    sample the design space systematically or randomly, or can optimize for a given problem through an iterative search
    and evaluate approach.


    Notes
    -----

        Schematic outline of how to use the ``Design`` plugin to explore a design space.

        .. image:: /_static/img/design.png
            :width: 80%
            :align: center

        The `Design <https://www.flexcompute.com/tidy3d/examples/notebooks/Design/>`_ notebook contains an overview of the
        ``Design`` plugin and is the best place to learn how to get started.
        Detailed examples using the ``Design`` plugin can be found in the following notebooks:

        * `All-Dielectric Structural Colors <https://www.flexcompute.com/tidy3d/examples/notebooks/AllDielectricStructuralColor/>`_
        * `Bayesian Optimization of Y-Junction <https://www.flexcompute.com/tidy3d/examples/notebooks/BayesianOptimizationYJunction/>`_
        * `Genetic Algorithm Reflector <https://www.flexcompute.com/tidy3d/examples/notebooks/GeneticAlgorithmReflector/>`_
        * `Particle Swarm Optimizer PBS <https://www.flexcompute.com/tidy3d/examples/notebooks/ParticleSwarmOptimizedPBS/>`_
        * `Particle Swarm Optimizer Bullseye Cavity <https://www.flexcompute.com/tidy3d/examples/notebooks/BullseyeCavityPSO/>`_

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

    parameters: tuple[ParameterType, ...] = Field(
        (),
        title="Parameters",
        description="Set of parameters defining the dimensions and allowed values for the design space.",
    )

    method: MethodType = Field(
        title="Search Type",
        description="Specifications for the procedure used to explore the parameter space.",
        discriminator=TYPE_TAG_STR,  # Stops pydantic trying to validate every method whilst checking MethodType
    )

    task_name: str = Field(
        "",
        title="Task Name",
        description="Task name assigned to tasks along with a simulation counter in the form of {task_name}_{sim_index}_{counter} where ``sim_index`` is "
        "the index of the workflow/simulation object from the pre function output. "
        "If the pre function outputs a dictionary the key will be included in the task name as {task_name}_{dict_key}_{counter}. "
        "Only used when pre-post functions are supplied.",
    )

    name: str | None = Field(
        None,
        title="Name",
        description="Optional name for the design space.",
    )

    path_dir: str = Field(
        ".",
        title="Path Directory",
        description="Directory where simulation data files will be locally saved to. Only used when pre and post functions are supplied.",
    )

    folder_name: str = Field(
        "default",
        title="Folder Name",
        description="Folder path where the simulation will be uploaded in the Tidy3D Workspace. Will use 'default' if no path is set.",
    )

    @cached_property
    def dims(self) -> tuple[str]:
        """dimensions defined by the design parameter names."""
        return tuple(param.name for param in self.parameters)

    @model_validator(mode="after")
    def _validate_method_filter_func(self) -> DesignSpace:
        """Validate method filter callback shape when available."""
        if not isinstance(self.method, MethodSample) or self.method.filter_func is None:
            return self

        filter_func = self.method.filter_func
        filter_sig = inspect.signature(filter_func)
        sample_kwargs = {param.name: param.sample_first() for param in self.parameters}

        try:
            filter_sig.bind(**sample_kwargs)
        except TypeError as exc:
            raise ValueError(
                "'filter_func' must accept keyword arguments for all design parameter names "
                f"{tuple(sample_kwargs.keys())}. Signature validation failed: {exc}"
            ) from exc

        return self

    def _package_run_results(
        self,
        fn_args: list[dict[str, Any]],
        fn_values: list[Any],
        fn_source: str,
        task_names: tuple[str] | None = None,
        task_ids: list | None = None,
        task_paths: list | None = None,
        aux_values: list[Any] | None = None,
        opt_output: Any = None,
    ) -> Result:
        """How to package results from ``method.run`` and ``method.run_batch``"""

        fn_args_coords = tuple([[arg_dict[dim] for arg_dict in fn_args] for dim in self.dims])

        fn_args_coords_T = list(map(list, zip(*fn_args_coords)))

        return Result(
            dims=self.dims,
            values=fn_values,
            coords=fn_args_coords_T,
            fn_source=fn_source,
            task_names=task_names,
            task_ids=task_ids,
            task_paths=task_paths,
            aux_values=aux_values,
            optimizer=opt_output,
        )

    @staticmethod
    def get_fn_source(function: Callable) -> str:
        """Get the function source as a string, return ``None`` if not available."""
        try:
            return inspect.getsource(function)
        except (TypeError, OSError):
            return None

    def run(
        self,
        fn: Callable,
        fn_post: Callable | None = None,
        verbose: bool = True,
        priority: int | None = None,
    ) -> Result:
        """Explore a parameter space with a supplied method using the user supplied function.
        Supplied functions are used to evaluate the design space and are called within the method.
        For optimization methods these functions act as the fitness function. A single function can be
        supplied which will contain the preprocessing, computation, and analysis of the desired problem.
        If running a Tidy3D simulation, it is recommended to split this function into a pre function, that creates a Simulation object(s),
        and a post function which analyses the SimulationData produced by the pre function Simulations. This allows the DesignSpace to
        manage the batching of Simulations, which varies between Method used, and saving time writing their own batching code.
        It also efficiently submits simulations to the cloud servers allowing for the fastest exploration of a design space.

        The ``fn`` function must take a dictionary input - this can be stored a dictionary ``def example_fn(**params)``
        or left as keyword arguments ``def example_fn(arg1, arg2)`` where the keywords correspond to the ``name`` of the parameters in the design space.

        If used as a pre function, the output of ``fn`` must be a float, a ``WorkflowOperationType`` (for example ``Simulation``, ``ModeSimulation``, ``EMESimulation``),
        a ``Batch``, list, or dict. Supplied ``Batch`` objects are run without modification and are run in series. A list or dict of workflow objects is flattened
        into a single ``Batch`` to enable parallel computation on the cloud. The original structure is then restored for output; all workflow objects are replaced by their corresponding data objects.
        Example pre return formats and associated post inputs can be seen in the table below.

        .. list-table:: Pre return formats and post input formats
            :widths: 50 50
            :header-rows: 1

            * - fn_pre return
              - fn_post call
            * - 1.0
              - fn_post(1.0)
            * - [1,2,3]
              - fn_post(1,2,3)
            * - {'a': 2, 'b': 'hi'}
              - fn_post(a=2, b='hi')
            * - Simulation
              - fn_post(SimulationData)
            * - Batch
              - fn_post(BatchData)
            * - [Simulation, Simulation]
              - fn_post(SimulationData, SimulationData)
            * - [Simulation, 1.0]
              - fn_post(SimulationData, 1.0)
            * - [Simulation, Batch]
              - fn_post(SimulationData, BatchData)
            * - {'a': Simulation, 'b': Batch, 'c': 2.0}
              - fn_post(a=SimulationData, b=BatchData, c=2.0)

        The output of ``fn_post`` (or ``fn`` if only one function is supplied) must be a float
        or a container where the first element is a ``float`` and second element is a ``list`` / ``dict`` e,g. [float {"aux_1": str}].
        The float is used by the optimizers as the return of the fitness function.
        The second element is for auxiliary data from the analysis that the user may want to keep.
        Sampling methods (``MethodGrid`` or ``MethodMonteCarlo``) can have any return type.

        Parameters
        ----------
        fn : Callable
            Function accepting arguments that correspond to the ``name`` fields
            of the ``DesignSpace.parameters``.
            Must return in the expected format for the ``method`` used in ``DesignSpace``,
            or return an object that fn_post can accept as an input.

        fn_post : Callable = None
            Optional function performing postprocessing on the output of ``fn``.
            It is recommended to supply fn_post when working with Simulation objects.
            Must return in the expected format for the ``method`` used in ``DesignSpace``.

        verbose : bool = True
            Toggle the output of statements stored in the logging console.

        priority : int = None
            Optional vGPU queue priority (1 = lowest, 10 = highest) applied to simulations run via
            ``fn`` / ``fn_post``. Ignored when ``fn_post`` is not provided because no automatic
            batching occurs in that mode.

        Returns
        -------
        :class:`Result`
            Object containing the results of the design space exploration.
            Can be converted to ``pandas.DataFrame`` with ``.to_dataframe()``.
        """

        if priority is not None and (priority < 1 or priority > 10):
            raise ValueError("'priority' must be between 1 and 10 (inclusive).")

        # Get the console
        # Method.run checks for console is None instead of being passed console and verbose
        console = get_logging_console() if verbose else None

        # Run based on how many functions the user provides
        if fn_post is None:
            if priority is not None:
                log.warning(
                    "'priority' has no effect unless both fn and fn_post are provided;"
                    " please split your function to allow automatic batching.",
                    log_once=True,
                )
            fn_args, fn_values, aux_values, opt_output = self.run_single(fn, console)
            sim_names = None
            sim_ids = None
            sim_paths = None

        else:
            (
                fn_args,
                fn_values,
                aux_values,
                opt_output,
                sim_names,
                sim_ids,
                sim_paths,
            ) = self.run_pre_post(
                fn_pre=fn,
                fn_post=fn_post,
                console=console,
                priority=priority,
            )

            if len(sim_names) == 0:
                sim_names = None
                sim_ids = None
                sim_paths = None

        fn_source = self.get_fn_source(fn)

        # Package the result
        return self._package_run_results(
            fn_args=fn_args,
            fn_values=fn_values,
            fn_source=fn_source,
            aux_values=aux_values,
            task_names=sim_names,
            task_ids=sim_ids,
            task_paths=sim_paths,
            opt_output=opt_output,
        )

    def run_single(self, fn: Callable, console: Console) -> tuple[list[dict], list, list[Any]]:
        """Run a single function of parameter inputs."""
        evaluate_fn = self._get_evaluate_fn_single(fn=fn)
        return self.method._run(run_fn=evaluate_fn, parameters=self.parameters, console=console)

    def run_pre_post(
        self,
        fn_pre: Callable,
        fn_post: Callable,
        console: Console,
        priority: int | None = None,
    ) -> tuple[list[dict], list[dict], list[Any]]:
        """Run a function with Tidy3D implicitly called in between."""
        handler = self._get_evaluate_fn_pre_post(
            fn_pre=fn_pre,
            fn_post=fn_post,
            fn_mid=self._fn_mid,
            console=console,
            priority=priority,
        )
        fn_args, fn_values, aux_values, opt_output = self.method._run(
            run_fn=handler.fn_combined, parameters=self.parameters, console=console
        )
        return (
            fn_args,
            fn_values,
            aux_values,
            opt_output,
            handler.sim_names,
            handler.sim_ids,
            handler.sim_paths,
        )

    """ Helpers """

    def _get_evaluate_fn_single(self, fn: Callable) -> list[Any]:
        """Get function that sequentially evaluates single `fn` for a list of arguments."""

        def evaluate(args_list: list) -> list[Any]:
            """Evaluate a list of arguments passed to ``fn``."""
            return [fn(**args) for args in args_list]

        return evaluate

    def _get_evaluate_fn_pre_post(
        self,
        fn_pre: Callable,
        fn_post: Callable,
        fn_mid: Callable,
        console: Console,
        priority: int | None,
    ) -> Any:
        """Get function that tries to use batch processing on a set of arguments."""

        postprocess_fn_mid_result = self._postprocess_fn_mid_result

        class Pre_Post_Handler:
            def __init__(self, console: Console, priority: int | None) -> None:
                self.sim_counter = 0
                self.sim_names = []
                self.sim_ids = []
                self.sim_paths = []
                self.console = console
                self.priority = priority

            def fn_combined(self, args_list: list[dict[str, Any]]) -> list[Any]:
                """Compute fn_pre and fn_post functions and capture other outputs."""
                sim_dict = {str(idx): fn_pre(**arg_list) for idx, arg_list in enumerate(args_list)}

                if not all(
                    isinstance(val, (int, float, *WORKFLOW_TYPES, Batch, list, dict))
                    for val in sim_dict.values()
                ):
                    raise ValueError(
                        "Unrecognized output of fn_pre. Please change the return of fn_pre."
                    )

                fn_mid_result = fn_mid(
                    sim_dict,
                    self.sim_counter,
                    self.console,
                    self.priority,
                )
                self.sim_names.extend(fn_mid_result.task_names)
                self.sim_ids.extend(fn_mid_result.task_ids)
                self.sim_paths.extend(fn_mid_result.task_paths)
                self.sim_counter = fn_mid_result.sim_counter
                return postprocess_fn_mid_result(fn_post, fn_mid_result)

        handler = Pre_Post_Handler(console, priority)

        return handler

    @staticmethod
    def _postprocess_fn_mid_result(
        fn_post: Callable,
        fn_mid_result: _FnMidResult,
    ) -> list[Any]:
        """Run fn_post, releasing workflow data after each top-level item."""

        if not fn_mid_result.task_names:
            return [fn_post(val) for val in fn_mid_result.data.values()]

        post_out = []
        try:
            while fn_mid_result.data:
                key = next(iter(fn_mid_result.data))
                value = fn_mid_result.data.pop(key)
                post_input = None
                try:
                    if fn_mid_result.post_input_loader is None:
                        post_input = value
                    else:
                        post_input = fn_mid_result.post_input_loader(key, value)
                    post_out.append(fn_post(post_input))
                finally:
                    del value
                    del post_input
            return post_out
        finally:
            if fn_mid_result.data:
                fn_mid_result.data.clear()
            gc.collect()

    @staticmethod
    def _run_batch(
        batch: Batch,
        path_dir: str,
        priority: int | None = None,
    ) -> BatchData:
        """Run a batch and return the BatchData."""
        batch_out = batch.run(path_dir=path_dir, priority=priority)
        return batch_out

    def _fn_mid(
        self,
        pre_out: dict[int, Any],
        sim_counter: int,
        console: Console,
        priority: int | None,
    ) -> _FnMidResult:
        """A function of the output of ``fn_pre`` that gives the input to ``fn_post``."""

        # Keep copy of original to use if no tidy3d simulation required
        original_pre_out = pre_out.copy()

        # Convert list input to dict of dict before passing through checks
        was_list = False
        if all(isinstance(sim, list) for sim in pre_out.values()):
            pre_out = {
                str(idx): {str(sub_idx): sub_list for sub_idx, sub_list in enumerate(sim_list)}
                for idx, sim_list in enumerate(pre_out.values())
            }
            was_list = True

        def _find_and_map(
            search_dict: dict,
            search_type: Any,
            output_dict: dict,
            naming_dict: dict,
            previous_key: str = "",
        ) -> None:
            """Recursively search for search_type objects within a dictionary."""
            current_key = previous_key
            for key, value in search_dict.items():
                if not len(previous_key):
                    latest_key = str(key)
                else:
                    latest_key = f"{current_key}_{key}"

                if isinstance(value, dict):
                    _find_and_map(value, search_type, output_dict, naming_dict, latest_key)
                elif isinstance(value, search_type):
                    output_dict[latest_key] = value
                    naming_dict[latest_key] = key

        simulations = {}
        batches = {}
        naming_keys = {}

        _find_and_map(pre_out, WORKFLOW_TYPES, simulations, naming_keys)
        _find_and_map(pre_out, Batch, batches, naming_keys)

        # Exit fn_mid here if no td computation is required
        if not simulations and not batches:
            return _FnMidResult(
                data=original_pre_out,
                task_names=[],
                task_ids=[],
                task_paths=[],
                sim_counter=sim_counter,
            )

        # Create task names for simulations
        named_sims = {}
        translate_sims = {}
        for sim_key, sim in simulations.items():
            # Checks stop standard indexing keys being included in the name
            suffix = f"{naming_keys[sim_key]}_{sim_counter}"

            # Handle if the user does not want a task name
            if len(self.task_name) > 0:
                sim_name = f"{self.task_name}_{suffix}"
            else:
                sim_name = suffix
            named_sims[sim_name] = sim
            translate_sims[sim_name] = sim_key
            sim_counter += 1

        # Log the simulations and batches for the user
        if console is not None:
            # Writen like this to include batches on the same line if present
            run_statement = f"{len(named_sims)} Simulations"
            if len(batches) > 0:
                run_statement = run_statement + f" and {len(batches)} user Batches"

            console.log(f"Running {run_statement}")

        # Running simulations and batches
        batch = Batch(
            simulations=named_sims,
            folder_name=self.folder_name,
            simulation_type="tidy3d_design",
            verbose=False,  # Using a custom output instead of Batch.monitor updates
        )
        sims_out = self._run_batch(batch, path_dir=self.path_dir, priority=priority)

        batch_results = {}
        for batch_key, batch in batches.items():
            batch_out = self._run_batch(batch, path_dir=self.path_dir, priority=priority)
            batch_results[batch_key] = batch_out

        sim_key_to_name = {sim_key: sim_name for sim_name, sim_key in translate_sims.items()}

        def _get_sim_metadata(sim_name: str, attr_name: str) -> str:
            """Get task metadata for an automatically batched simulation."""
            if attr_name == "task_name":
                return sim_name
            if attr_name == "task_id":
                return sims_out.task_ids[sim_name]
            return sims_out.task_paths[sim_name]

        def _get_batch_metadata(batch_data: BatchData, attr_name: str) -> list[str]:
            """Get task metadata for a user-supplied batch."""
            if attr_name == "task_name":
                return list(batch_data.task_ids.keys())
            if attr_name == "task_id":
                return list(batch_data.task_ids.values())
            return list(batch_data.task_paths.values())

        def _remove_or_replace(
            search_dict: dict,
            attr_name: str,
            previous_key: str = "",
        ) -> dict:
            """Recursively build task metadata without loading workflow data."""
            new_dict = {}
            for key, value in search_dict.items():
                if not len(previous_key):
                    latest_key = str(key)
                else:
                    latest_key = f"{previous_key}_{key}"

                if isinstance(value, dict):
                    new_sub_dict = _remove_or_replace(value, attr_name, latest_key)
                    new_dict[key] = new_sub_dict

                else:
                    if latest_key in sim_key_to_name:
                        sim_name = sim_key_to_name[latest_key]
                        new_dict[key] = _get_sim_metadata(sim_name, attr_name)

                    elif latest_key in batch_results:
                        batch_data = batch_results[latest_key]
                        new_dict[key] = _get_batch_metadata(batch_data, attr_name)

            return new_dict

        def _load_value(value: Any, current_key: str) -> Any:
            """Load workflow data for a single top-level postprocessing input."""
            if isinstance(value, dict):
                return {
                    key: _load_value(sub_value, f"{current_key}_{key}")
                    for key, sub_value in value.items()
                }

            if current_key in sim_key_to_name:
                sim_name = sim_key_to_name[current_key]
                sim_data = sims_out[sim_name]
                sim_data.attrs["task_name"] = sim_name
                sim_data.attrs["task_id"] = sims_out.task_ids[sim_name]
                sim_data.attrs["task_path"] = sims_out.task_paths[sim_name]
                return sim_data

            if current_key in batch_results:
                return batch_results.pop(current_key)

            return value

        def _load_post_input(top_key: str, top_value: Any) -> Any:
            """Restore the original top-level fn_pre output shape for fn_post."""
            post_input = _load_value(top_value, top_key)
            if was_list:
                return list(post_input.values())
            return post_input

        # Build out a dict of task_name or task_path in the same shape as the original data
        task_names = _remove_or_replace(pre_out.copy(), "task_name")
        task_ids = _remove_or_replace(pre_out.copy(), "task_id")
        task_paths = _remove_or_replace(pre_out.copy(), "task_path")

        # Reduce down to a list to be extended later
        task_names = list(task_names.values())
        task_ids = list(task_ids.values())
        task_paths = list(task_paths.values())

        # Restore output to a list if a list was supplied
        if was_list:
            task_names = [list(sub_dict.values()) for sub_dict in task_names]
            task_ids = [list(sub_dict.values()) for sub_dict in task_ids]
            task_paths = [list(sub_dict.values()) for sub_dict in task_paths]

        return _FnMidResult(
            data=pre_out,
            task_names=task_names,
            task_ids=task_ids,
            task_paths=task_paths,
            sim_counter=sim_counter,
            post_input_loader=_load_post_input,
        )

    def run_batch(
        self,
        fn_pre: Callable[
            Any,
            WorkflowOperationType | list[WorkflowOperationType] | dict[str, WorkflowOperationType],
        ],
        fn_post: Callable[
            [WorkflowDataType | list[WorkflowDataType] | dict[str, WorkflowDataType]], Any
        ],
        path_dir: str = ".",
        priority: int | None = None,
        **batch_kwargs: Any,
    ) -> Result:
        """
        This function has been superceded by `run`, please use `run` for batched simulations.
        """
        log.warning(
            "In version 2.8.0, the 'run_batch' method is replaced by 'run'."
            "'fn_pre' has become 'fn', whilst 'fn_post' remains the same."
            "While the original syntax will still be supported, future functionality will be added to the new method moving forward."
        )

        if len(batch_kwargs) > 0:
            log.warning(
                "'batch_kwargs' supplied here will no longer be used within the simulation."
            )

        new_self = self.updated_copy(path_dir=path_dir)
        result = new_self.run(fn=fn_pre, fn_post=fn_post, priority=priority)

        return result

    def estimate_cost(self, fn_pre: Callable) -> float:
        """Compute the maximum FlexCredit charge for the ``DesignSpace.run`` computation.

        Require a pre function that should return a ``WorkflowOperationType`` object, a ``Batch`` object, or collection of either.
        The pre function is called to estimate the cost - complicated pre functions may cause long runtimes. The cost per
        iteration is multiplied by the theoretical maximum number of iterations to give the maximum cost.

        Parameters
        ----------
        fn_pre : Callable
            Function accepting arguments that correspond to the ``name`` fields
            of the ``DesignSpace.parameters``. Should return a ``WorkflowOperationType`` or ``Batch`` object, or a
            top-level ``list`` / nested ``dict`` tree of these objects.

        Returns
        -------
        float
            Estimated maximum cost for the ``DesignSpace.run``.

        """
        # Get output fn_pre for paramters at the lowest span / default
        arg_dict = {}
        for param in self.parameters:
            arg_dict[param.name] = param.sample_first()

        # Compute fn_pre
        pre_out = fn_pre(**arg_dict)

        def _estimate_sim_cost(workflow: WorkflowOperationType) -> float:
            job = Job(simulation=workflow, task_name="estimate_cost")

            estimate = job.estimate_cost()
            job.delete()  # Deleted as only a test with initial parameters

            return estimate

        def _estimate_container_cost(value: Any, allow_list: bool) -> float | None:
            """Recursively estimate costs for supported container outputs."""
            if isinstance(value, WORKFLOW_TYPES):
                return _estimate_sim_cost(value)

            if isinstance(value, Batch):
                estimate = value.estimate_cost()
                value.delete()  # Deleted as only a test with initial parameters
                return estimate

            if isinstance(value, dict):
                total_estimate = 0.0
                for sub_value in value.values():
                    sub_estimate = _estimate_container_cost(sub_value, allow_list=False)
                    if sub_estimate is None:
                        return None
                    total_estimate += sub_estimate
                return total_estimate

            if allow_list and isinstance(value, list):
                total_estimate = 0.0
                for sub_value in value:
                    sub_estimate = _estimate_container_cost(sub_value, allow_list=False)
                    if sub_estimate is None:
                        return None
                    total_estimate += sub_estimate
                return total_estimate

            # Non-workflow values inside a supported container are treated as free.
            return 0.0

        if isinstance(pre_out, WORKFLOW_TYPES):
            per_run_estimate = _estimate_sim_cost(pre_out)

        elif isinstance(pre_out, Batch):
            per_run_estimate = pre_out.estimate_cost()
            pre_out.delete()  # Deleted as only a test with initial parameters

        elif isinstance(pre_out, list):
            per_run_estimate = _estimate_container_cost(pre_out, allow_list=True)

        elif isinstance(pre_out, dict):
            per_run_estimate = _estimate_container_cost(pre_out, allow_list=False)

        else:
            raise ValueError("Unrecognized output from pre-function, unable to estimate cost.")

        # Calculate maximum number of runs for different methods
        run_count = self.method._get_run_count(self.parameters)

        # For if tidy3d server cannot determine the estimate
        if per_run_estimate is None:
            return None
        return round(per_run_estimate * run_count, 3)

    def summarize(self, fn_pre: Callable | None = None, verbose: bool = True) -> dict[str, Any]:
        """Summarize the setup of the DesignSpace

        Prints a summary of the DesignSpace including the method and associated args, the parameters,
        and the maximum number of runs expected. If ``fn_pre`` is provided an estimated cost will
        also be included. Additional notes are printed where relevant. All data is returned as a dict.

        Parameters
        ----------
        fn_pre : Callable = None
            Function accepting arguments that correspond to the ``name`` fields
            of the ``DesignSpace.parameters``. Allows for estimated cost to be included
            in the summary.
        verbose: bool = True
            Toggle if the summary should be output to log. If False, the dict is returned silently.

        Returns
        -------
        summary_dict: dict
            Dictionary containing the summary information.
        """

        # Get output console
        console = get_logging_console()

        # Assemble message
        # If check stops it printing standard attributes
        arg_values = [
            f"{field}: {getattr(self.method, field)}\n"
            for field in type(self.method).model_fields
            if field not in MethodOptimize.model_fields
        ]

        param_values = []
        for param in self.parameters:
            if isinstance(param, ParameterAny):
                param_values.append(f"{param.name}: {param.type} {param.allowed_values}\n")
            else:
                param_values.append(f"{param.name}: {param.type} {param.span}\n")

        run_count = self.method._get_run_count(self.parameters)

        # Compile data into a dict for return
        summary_dict = {
            "method": self.method.type,
            "method_args": "".join(arg_values),
            "param_count": len(self.parameters),
            "param_names": ", ".join([param.name for param in self.parameters]),
            "param_vals": "".join(param_values),
            "max_run_count": run_count,
        }

        if verbose:
            console.log(
                "\nSummary of DesignSpace\n\n"
                f"Method: {summary_dict['method']}\n"
                f"Method Args\n{summary_dict['method_args']}\n"
                f"No. of Parameters: {summary_dict['param_count']}\n"
                f"Parameters: {summary_dict['param_names']}\n"
                f"{summary_dict['param_vals']}\n"
                f"Maximum Run Count: {summary_dict['max_run_count']}\n"
            )

            if fn_pre is not None:
                cost_estimate = self.estimate_cost(fn_pre)
                summary_dict["cost_estimate"] = cost_estimate
                console.log(f"Estimated Maximum Cost: {cost_estimate} FlexCredits")

                # NOTE: Could then add more details regarding the output of fn_pre - confirm batching?

        # Include additional notes/warnings
        notes = []

        if isinstance(self.method, MethodGenAlg):
            notes.append(
                "The maximum run count for MethodGenAlg is difficult to predict. "
                "Repeated solutions are not executed, reducing the total number of simulations. "
                "High crossover and mutation probabilities may result in an increased number of simulations, potentially exceeding the predicted maximum run count."
            )

        if isinstance(self.method, (MethodBayOpt, MethodGenAlg, MethodParticleSwarm)):
            if any(isinstance(param, ParameterInt) for param in self.parameters):
                if any(isinstance(param, ParameterAny) for param in self.parameters):
                    notes.append(
                        "Discrete 'ParameterAny' values are automatically converted to 'int' values to be optimized.\n"
                    )

                notes.append(
                    "Discrete 'int' values are automatically rounded if optimizers generate 'float' predictions.\n"
                )

        if len(notes) > 0 and verbose:
            console.log(
                "Notes:",
            )
            for note in notes:
                console.log(note)

        return summary_dict
