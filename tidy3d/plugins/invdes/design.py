# container for everything defining the inverse design

from __future__ import annotations

import abc
import typing

import autograd.numpy as anp
import numpy as np
import pydantic.v1 as pd

import tidy3d as td
from tidy3d.components.autograd import get_static
from tidy3d.exceptions import ValidationError
from tidy3d.plugins.expressions.metrics import Metric, generate_validation_data
from tidy3d.plugins.expressions.types import ExpressionType

from .base import InvdesBaseModel
from .region import DesignRegionType
from .validators import check_pixel_size

PostProcessFnType = typing.Callable[[td.SimulationData], float]


class AbstractInverseDesign(InvdesBaseModel, abc.ABC):
    """Container for an inverse design problem."""

    design_region: DesignRegionType = pd.Field(
        ...,
        title="Design Region",
        description="Region within which we will optimize the simulation.",
    )

    task_name: str = pd.Field(
        ...,
        title="Task Name",
        description="Task name to use in the objective function when running the ``JaxSimulation``.",
    )

    verbose: bool = pd.Field(
        False,
        title="Task Verbosity",
        description="If ``True``, will print the regular output from ``web`` functions.",
    )

    metric: typing.Optional[ExpressionType] = pd.Field(
        None,
        title="Objective Metric",
        description="Serializable expression defining the objective function.",
    )

    def make_objective_fn(
        self, post_process_fn: typing.Optional[typing.Callable] = None, maximize: bool = True
    ) -> typing.Callable[[anp.ndarray], tuple[float, dict]]:
        """Construct the objective function for this InverseDesign object."""

        if (post_process_fn is None) and (self.metric is None):
            raise ValueError("Either 'post_process_fn' or 'metric' must be provided.")

        if (post_process_fn is not None) and (self.metric is not None):
            raise ValueError("Provide only one of 'post_process_fn' or 'metric', not both.")

        direction_multiplier = 1 if maximize else -1

        def objective_fn(params: anp.ndarray, aux_data: dict = None) -> float:
            """Full objective function."""
            data = self.to_simulation_data(params=params)

            if self.metric is None:
                post_process_val = post_process_fn(data)
            elif isinstance(data, td.SimulationData):
                post_process_val = self.metric.evaluate(data)
            elif getattr(data, "type", None) == "BatchData":
                raise NotImplementedError("Metrics currently do not support 'BatchData'")
            else:
                raise ValueError(f"Invalid data type: {type(data)}")

            penalty_value = self.design_region.penalty_value(params)
            objective_fn_val = direction_multiplier * post_process_val - penalty_value

            # Store auxiliary data if provided
            if aux_data is not None:
                aux_data["penalty"] = get_static(penalty_value)
                aux_data["post_process_val"] = get_static(post_process_val)
                aux_data["objective_fn_val"] = get_static(objective_fn_val) * direction_multiplier
                if isinstance(data, td.SimulationData):
                    aux_data["sim_data"] = data.to_static()
                else:
                    aux_data["sim_data"] = {k: v.to_static() for k, v in data.items()}
                aux_data["params"] = params

            return objective_fn_val

        return objective_fn

    @property
    def initial_simulation(self) -> td.Simulation:
        """Return a simulation with the initial design region parameters."""
        initial_params = self.design_region.initial_parameters
        return self.to_simulation(initial_params)

    def run(self, simulation, **kwargs) -> td.SimulationData:
        """Run a single tidy3d simulation."""
        from tidy3d.web import run

        kwargs.setdefault("verbose", self.verbose)
        kwargs.setdefault("task_name", self.task_name)
        return run(simulation, **kwargs)

    def run_async(self, simulations, **kwargs) -> web.BatchData:  # noqa: F821
        """Run a batch of tidy3d simulations."""
        from tidy3d.web import run_async

        kwargs.setdefault("verbose", self.verbose)
        return run_async(simulations, **kwargs)


class InverseDesign(AbstractInverseDesign):
    """Container for an inverse design problem."""

    simulation: td.Simulation = pd.Field(
        ...,
        title="Base Simulation",
        description="Simulation without the design regions or monitors used in the objective fn.",
    )

    output_monitor_names: typing.Tuple[str, ...] = pd.Field(
        None,
        title="Output Monitor Names",
        description="Optional names of monitors whose data the differentiable output depends on."
        "If this field is left ``None``, the plugin will try to add all compatible monitors to "
        "``JaxSimulation.output_monitors``. While this will work, there may be warnings if the "
        "monitors are not compatible with the ``adjoint`` plugin, for example if there are "
        "``FieldMonitor`` instances with ``.colocate != False``.",
    )

    _check_sim_pixel_size = check_pixel_size("simulation")

    @pd.root_validator(pre=False)
    def _validate_model(cls, values: dict) -> dict:
        cls._validate_metric(values)
        return values

    @staticmethod
    def _validate_metric(values: dict) -> dict:
        metric_expr = values.get("metric")
        if not metric_expr:
            return values
        simulation = values.get("simulation")
        for metric in metric_expr.filter(Metric):
            InverseDesign._validate_metric_monitor_name(metric, simulation)
            InverseDesign._validate_metric_mode_index(metric, simulation)
            InverseDesign._validate_metric_f(metric, simulation)
        InverseDesign._validate_metric_data(metric_expr, simulation)
        return values

    @staticmethod
    def _validate_metric_monitor_name(metric: Metric, simulation: td.Simulation) -> None:
        """Validate that the monitor name of the metric exists in the simulation."""
        monitor = next((m for m in simulation.monitors if m.name == metric.monitor_name), None)
        if monitor is None:
            raise ValidationError(
                f"Monitor named '{metric.monitor_name}' associated with the metric not found in the simulation monitors."
            )

    @staticmethod
    def _validate_metric_mode_index(metric: Metric, simulation: td.Simulation) -> None:
        """Validate that the mode index of the metric is within the bounds of the monitor's ``ModeSpec.num_modes``."""
        monitor = next((m for m in simulation.monitors if m.name == metric.monitor_name), None)
        if metric.mode_index >= monitor.mode_spec.num_modes:
            raise ValidationError(
                f"Mode index '{metric.mode_index}' for metric associated with monitor "
                f"'{metric.monitor_name}' is out of bounds. "
                f"Maximum allowed mode index is '{monitor.mode_spec.num_modes - 1}'."
            )

    @staticmethod
    def _validate_metric_f(metric: Metric, simulation: td.Simulation) -> None:
        """Validate that the frequencies of the metric are present in the monitor."""
        monitor = next((m for m in simulation.monitors if m.name == metric.monitor_name), None)
        if metric.f is not None:
            metric_f_list = [metric.f] if isinstance(metric.f, float) else metric.f
            if len(metric_f_list) != 1:
                raise ValidationError("Only a single frequency is supported for the metric.")
            for freq in metric_f_list:
                if not any(np.isclose(freq, monitor.freqs, atol=1.0)):
                    raise ValidationError(
                        f"Frequency '{freq}' for metric associated with monitor "
                        f"'{metric.monitor_name}' not found in monitor frequencies."
                    )
        else:
            if len(monitor.freqs) != 1:
                raise ValidationError(
                    f"Monitor '{metric.monitor_name}' must contain only a single frequency when metric.f is None."
                )

    @staticmethod
    def _validate_metric_data(expr: ExpressionType, simulation: td.Simulation) -> None:
        """Validate that expression can be evaluated and returns a real scalar."""
        data = generate_validation_data(expr)
        try:
            result = expr(data)
        except Exception as e:
            raise ValidationError(f"Failed to evaluate the metric expression: {str(e)}") from e
        if len(np.ravel(result)) > 1:
            raise ValidationError(
                f"The expression must return a scalar value or an array of length 1 (got {result})."
            )
        if not np.all(np.isreal(result)):
            raise ValidationError(
                f"The expression must return a real (not complex) value (got {result})."
            )

    def is_output_monitor(self, monitor: td.Monitor) -> bool:
        """Whether a monitor is added to the ``JaxSimulation`` as an ``output_monitor``."""

        output_mnt_types = td.components.simulation.OutputMonitorTypes

        if self.output_monitor_names is None:
            return any(isinstance(monitor, mnt_type) for mnt_type in output_mnt_types)

        return monitor.name in self.output_monitor_names

    def separate_output_monitors(self, monitors: typing.Tuple[td.Monitor]) -> dict:
        """Separate monitors into output_monitors and regular monitors."""

        monitor_fields = dict(monitors=[], output_monitors=[])

        for monitor in monitors:
            key = "output_monitors" if self.is_output_monitor(monitor) else "monitors"
            monitor_fields[key].append(monitor)

        return monitor_fields

    def to_simulation(self, params: anp.ndarray) -> td.Simulation:
        """Convert the ``InverseDesign`` to a corresponding ``td.Simulation`` with traced fields."""

        # construct the design region to a regular structure
        design_region_structure = self.design_region.to_structure(params)

        # construct mesh override structures and a new grid spec, if applicable
        grid_spec = self.simulation.grid_spec
        mesh_override_structure = self.design_region.mesh_override_structure
        if mesh_override_structure:
            override_structures = list(self.simulation.grid_spec.override_structures)
            override_structures += [mesh_override_structure]
            grid_spec = grid_spec.updated_copy(override_structures=override_structures)

        return self.simulation.updated_copy(
            structures=list(self.simulation.structures) + [design_region_structure],
            grid_spec=grid_spec,
        )

    def to_simulation_data(self, params: anp.ndarray, **kwargs) -> td.SimulationData:
        """Convert the ``InverseDesign`` to a ``td.Simulation`` and run it."""
        simulation = self.to_simulation(params=params)
        return self.run(simulation, **kwargs)


class InverseDesignMulti(AbstractInverseDesign):
    """``InverseDesign`` with multiple simulations and corresponding postprocess functions."""

    simulations: typing.Tuple[td.Simulation, ...] = pd.Field(
        ...,
        title="Base Simulations",
        description="Set of simulation without the design regions or monitors used in the objective fn.",
    )

    output_monitor_names: typing.Tuple[typing.Union[typing.Tuple[str, ...], None], ...] = pd.Field(
        None,
        title="Output Monitor Names",
        description="Optional names of monitors whose data the differentiable output depends on."
        "If this field is left ``None``, the plugin will try to add all compatible monitors to "
        "``JaxSimulation.output_monitors``. While this will work, there may be warnings if the "
        "monitors are not compatible with the ``adjoint`` plugin, for example if there are "
        "``FieldMonitor`` instances with ``.colocate != False``.",
    )

    _check_sim_pixel_size = check_pixel_size("simulations")

    @pd.root_validator()
    def _check_lengths(cls, values):
        """Check the lengths of all of the multi fields."""

        keys = ("simulations", "post_process_fns", "output_monitor_names", "override_structure_dl")
        multi_dict = {key: values.get(key) for key in keys}
        sizes = {key: len(val) for key, val in multi_dict.items() if val is not None}

        if len(set(sizes.values())) != 1:
            raise ValueError(
                f"'MultiInverseDesign' requires that the fields {keys} must either "
                "have the same length or be left ``None``, if optional. Given fields with "
                "corresponding sizes of '{sizes}'."
            )

        return values

    @property
    def task_names(self) -> list[str]:
        """Task names associated with each of the simulations."""
        return [f"{self.task_name}_{i}" for i in range(len(self.simulations))]

    @property
    def designs(self) -> typing.List[InverseDesign]:
        """List of individual ``InverseDesign`` objects corresponding to this instance."""

        designs_list = []
        for i, (task_name, sim) in enumerate(zip(self.task_names, self.simulations)):
            des_i = InverseDesign(
                design_region=self.design_region,
                simulation=sim,
                verbose=self.verbose,
                task_name=task_name,
            )
            if self.output_monitor_names is not None:
                des_i = des_i.updated_copy(output_monitor_names=self.output_monitor_names[i])

            designs_list.append(des_i)

        return designs_list

    def to_simulation(self, params: anp.ndarray) -> dict[str, td.Simulation]:
        """Convert the ``InverseDesign`` to a corresponding dict of ``td.Simulation``s."""
        simulation_list = [design.to_simulation(params) for design in self.designs]
        return dict(zip(self.task_names, simulation_list))

    def to_simulation_data(self, params: anp.ndarray, **kwargs) -> web.BatchData:  # noqa: F821
        """Convert the ``InverseDesignMulti`` to a set of ``td.Simulation``s and run async."""
        simulations = self.to_simulation(params)
        return self.run_async(simulations, **kwargs)


InverseDesignType = typing.Union[InverseDesign, InverseDesignMulti]
