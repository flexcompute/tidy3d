"""Defines design space specification for tidy3d."""

from __future__ import annotations

import inspect
from typing import Any, Callable, Dict, List, Tuple, Union

import pydantic.v1 as pd

from ...components.base import TYPE_TAG_STR, Tidy3dBaseModel, cached_property
from ...components.data.sim_data import SimulationData
from ...components.simulation import Simulation
from ...log import get_logging_console, log
from ...web.api.container import Batch, BatchData, Job
from .method import (
    MethodBayOpt,
    MethodGenAlg,
    MethodOptimize,
    MethodParticleSwarm,
    MethodType,
)
from .parameter import ParameterAny, ParameterInt, ParameterType
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
        discriminator=TYPE_TAG_STR,  # Stops pydantic trying to validate every method whilst checking MethodType
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
        description="Directory where simulation data files will be saved to. Only used when pre-post functions are supplied.",
    )

    folder_name: str = pd.Field(
        "default",
        title="Folder Name",
        description="Folder path where the simulation will be uploaded in the Tidy3D Workspace. Will use 'default' if no path is set.",
    )

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
        opt_output: Any = None,
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

    def run(self, fn: Callable, fn_post: Callable = None, verbose: bool = True) -> Result:
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

        # Get the console
        # Method.run checks for console is None instead of being passed console and verbose
        console = get_logging_console() if verbose else None

        # Run based on how many functions the user provides
        if fn_post is None:
            fn_args, fn_values, aux_values, opt_output = self.run_single(fn, console)
            sim_names = None

        else:
            fn_args, fn_values, aux_values, opt_output, sim_names = self.run_pre_post(
                fn_pre=fn, fn_post=fn_post, console=console
            )

            sim_names = tuple(sim_names)

            if len(sim_names) == 0:
                sim_names = None

        fn_source = self.get_fn_source(fn)

        # Package the result
        return self._package_run_results(
            fn_args=fn_args,
            fn_values=fn_values,
            fn_source=fn_source,
            aux_values=aux_values,
            task_ids=sim_names,
            opt_output=opt_output,
        )

    def run_single(self, fn: Callable, console) -> Tuple(list[dict], list, list[Any]):
        """Run a single function of parameter inputs."""
        evaluate_fn = self._get_evaluate_fn_single(fn=fn)
        return self.method.run(run_fn=evaluate_fn, parameters=self.parameters, console=console)

    def run_pre_post(self, fn_pre: Callable, fn_post: Callable, console) -> Tuple(
        list[dict], list[dict], list[Any]
    ):
        """Run a function with tidy3d implicitly called in between."""
        handler = self._get_evaluate_fn_pre_post(
            fn_pre=fn_pre, fn_post=fn_post, fn_mid=self._fn_mid
        )
        fn_args, fn_values, aux_values, opt_output = self.method.run(
            run_fn=handler.fn_combined, parameters=self.parameters, console=console
        )
        return fn_args, fn_values, aux_values, opt_output, handler.sim_names

    """ Helpers """

    def _get_evaluate_fn_single(self, fn: Callable) -> list[Any]:
        """Get function that sequentially evaluates single `fn` for a list of arguments."""

        def evaluate(args_list: list) -> list[Any]:
            """Evaluate a list of arguments passed to ``fn``."""
            return [fn(**args) for args in args_list]

        return evaluate

    def _get_evaluate_fn_pre_post(self, fn_pre: Callable, fn_post: Callable, fn_mid: Callable):
        """Get function that tries to use batch processing on a set of arguments."""

        class Pre_Post_Handler:
            def __init__(self):
                self.sim_counter = 0
                self.sim_names = []

            def fn_combined(self, args_list: list[dict[str, Any]]) -> list[Any]:
                """Compute fn_pre and fn_post functions and capture other outputs."""
                sim_dict = {str(idx): fn_pre(**arg_list) for idx, arg_list in enumerate(args_list)}

                if not all(
                    isinstance(val, (int, float, Simulation, Batch, list, dict))
                    for val in sim_dict.values()
                ):
                    raise ValueError(
                        "Unrecognized output of fn_pre. Please change the return of fn_pre."
                    )

                data, sim_names = fn_mid(sim_dict, self.sim_counter)
                self.sim_names.extend(sim_names)
                self.sim_counter += len(sim_names)
                post_out = [fn_post(val) for val in data.values()]
                return post_out

        handler = Pre_Post_Handler()

        return handler

    def _fn_mid(
        self, pre_out: dict[int, Any], sim_counter: int
    ) -> Union[dict[int, Any], BatchData]:
        """A function of the output of ``fn_pre`` that gives the input to ``fn_post``."""

        # NOTE: Need to check that pre_out has string keys!

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
        ):
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

        _find_and_map(pre_out, Simulation, simulations, naming_keys)
        _find_and_map(pre_out, Batch, batches, naming_keys)

        # Exit fn_mid here if no td computation is required
        if not len(simulations) and not len(batches):
            return original_pre_out, list()

        # Compute sims and batches and return to pre_out
        named_sims = {}
        translate_sims = {}
        for sim_key, sim in simulations.items():
            # Checks stop standard indexing keys being included in the name
            suffix = (
                f"{naming_keys[sim_key]}_{sim_counter}"
                if naming_keys[sim_key] != str(sim_counter) and not was_list
                else sim_counter
            )
            sim_name = f"{self.task_name}_{suffix}"
            named_sims[sim_name] = sim
            translate_sims[sim_name] = sim_key
            sim_counter += 1

        sims_out = Batch(
            simulations=named_sims,
            folder_name=self.folder_name,
            simulation_type="tidy3d_design",
        ).run(path_dir=self.path_dir)

        batch_results = {}
        for batch_key, batch in batches.items():
            batch_out = batch.run(path_dir=self.path_dir)
            batch_results[batch_key] = batch_out

        def _return_to_dict(return_dict, key, return_obj):
            """Recursively insert items into a dict by keys split with underscore. Only works for dict or dict of dict inputs."""
            split_key = key.split("_", 1)
            if len(split_key) > 1:
                _return_to_dict(return_dict[split_key[0]], split_key[1], return_obj)
            else:
                return_dict[split_key[0]] = return_obj

        for sim_name, sim in sims_out.items():
            translated_name = translate_sims[sim_name]
            _return_to_dict(pre_out, translated_name, sim)

        for batch_name, batch in batch_results.items():
            _return_to_dict(pre_out, batch_name, batch)

        # Restore output to a list if a list was supplied
        if was_list:
            return {
                dict_idx: list(sub_dict.values()) for dict_idx, sub_dict in pre_out.items()
            }, list(named_sims.keys())

        return pre_out, list(named_sims.keys())

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
            "'fn_pre' has become 'fn', whilst 'fn_post' remains the same."
            "While the original syntax will still be supported, future functionality will be added to the new method moving forward."
        )

        if len(batch_kwargs) > 0:
            log.warning(
                "'batch_kwargs' supplied here will no longer be used within the simulation."
            )

        new_self = self.updated_copy(path_dir=path_dir)
        result = new_self.run(fn=fn_pre, fn_post=fn_post)

        return result

    def estimate_cost(self, fn_pre: Callable) -> float:
        """
        Compute the maximum FlexCredit charge for the entire design space computation.
        Require a pre function that should return a Simulation object, Batch object, or collection of either.
        The pre function is called to estimate the cost - complicated pre functions may cause long runtimes.
        """
        # Get output fn_pre for paramters at the lowest span / default
        arg_dict = {}
        for param in self.parameters:
            arg_dict[param.name] = param.sample_first()

        # Compute fn_pre
        pre_out = fn_pre(**arg_dict)

        def _estimate_sim_cost(sim):
            job = Job(simulation=sim, task_name="estimate_cost")

            estimate = job.estimate_cost()
            job.delete()  # Deleted as only a test with initial parameters

            return estimate

        if isinstance(pre_out, Simulation):
            per_run_estimate = _estimate_sim_cost(pre_out)

        elif isinstance(pre_out, Batch):
            per_run_estimate = pre_out.estimate_cost()
            pre_out.delete()  # Deleted as only a test with initial parameters

        elif isinstance(pre_out, (list, dict)):
            # Iterate through container to get simulations and batches and sum cost
            # Accept list or dict inputs

            if isinstance(pre_out, dict):
                pre_out = list(pre_out.values())

            sims = []
            batches = []
            for value in pre_out:
                if isinstance(value, Simulation):
                    sims.append(value)
                elif isinstance(value, Batch):
                    batches.append(value)

            calculated_estimates = []
            for sim in sims:
                calculated_estimates.append(_estimate_sim_cost(sim))

            for batch in batches:
                calculated_estimates.append(batch.estimate_cost())
                batch.delete()  # Deleted as only a test with initial parameters

            if None in calculated_estimates:
                per_run_estimate = None
            else:
                per_run_estimate = sum(calculated_estimates)

        else:
            raise ValueError("Unrecognized output from pre-function, unable to estimate cost.")

        # Calculate maximum number of runs for different methods
        run_count = self.method.get_run_count(self.parameters)

        # For if tidy3d server cannot determine the estimate
        if per_run_estimate is None:
            return None
        else:
            return round(per_run_estimate * run_count, 3)

    def summarize(self, fn_pre: Callable = None) -> dict[str, Any]:
        """Summarize the setup of the DesignSpace"""

        # Get output console
        console = get_logging_console()

        # Assemble message
        # If check stops it printing standard attributes
        arg_values = [
            f"{field}: {getattr(self.method, field)}\n"
            for field in self.method.__fields__
            if field not in MethodOptimize.__fields__
        ]

        param_values = []
        for param in self.parameters:
            if isinstance(param, ParameterAny):
                param_values.append(f"{param.name}: {param.type} {param.allowed_values}\n")
            else:
                param_values.append(f"{param.name}: {param.type} {param.span}\n")

        run_count = self.method.get_run_count(self.parameters)

        # Compile data into a dict for return
        summary_dict = {
            "method": self.method.type,
            "method_args": "".join(arg_values),
            "param_count": len(self.parameters),
            "param_names": ", ".join([param.name for param in self.parameters]),
            "param_vals": "".join(param_values),
            "max_run_count": run_count,
        }

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
                "The maximum run count for MethodGenAlg is difficult to predict."
                "Repeated solutions are not executed, reducing the total number of simulations."
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

        if len(notes) > 0:
            console.log(
                "Notes:",
            )
            for note in notes:
                console.log(note)

        return summary_dict
