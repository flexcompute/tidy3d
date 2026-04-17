# container for everything defining the inverse design

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Any, Callable, Optional, Union

import numpy as np
from pydantic import Field, field_validator, model_validator

import tidy3d as td
from tidy3d.components.autograd import get_static
from tidy3d.exceptions import ValidationError
from tidy3d.plugins.expressions.metrics import Metric, generate_validation_data
from tidy3d.plugins.expressions.types import ExpressionType

from .base import InvdesBaseModel
from .region import DesignRegionType
from .validators import check_pixel_size

if TYPE_CHECKING:
    import autograd.numpy as anp

    from tidy3d.compat import Self
    from tidy3d.plugins.autograd.invdes.symmetries import MirrorSymmetry

PostProcessFnType = Callable[[td.SimulationData], float]


def _resolve_mirror_symmetry(
    *,
    array_shape: tuple[int, ...],
    design_region: DesignRegionType,
    simulation: td.Simulation,
) -> Optional[MirrorSymmetry]:
    """Mirror boundaries relevant to a design-region array inside a simulation."""
    rmin, rmax = design_region.geometry.bounds
    mirror_symmetry = []
    for size_axis, sim_sym, low, high, sim_center in zip(
        array_shape, simulation.symmetry, rmin, rmax, simulation.center
    ):
        if sim_sym == 0 or size_axis <= 1:
            mirror_symmetry.append(None)
            continue

        touches_low = np.isclose(low, sim_center, rtol=0.0, atol=1e-12)
        touches_high = np.isclose(high, sim_center, rtol=0.0, atol=1e-12)

        if touches_low and not touches_high:
            mirror_symmetry.append("low")
        elif touches_high and not touches_low:
            mirror_symmetry.append("high")
        else:
            mirror_symmetry.append(None)

    if any(side is not None for side in mirror_symmetry):
        return tuple(mirror_symmetry)
    return None


class AbstractInverseDesign(InvdesBaseModel, abc.ABC):
    """Container for an inverse design problem."""

    design_region: DesignRegionType = Field(
        title="Design Region",
        description="Region within which we will optimize the simulation.",
    )

    task_name: str = Field(
        title="Task Name",
        description="Task name to use in the objective function when running the ``JaxSimulation``.",
    )

    verbose: bool = Field(
        False,
        title="Task Verbosity",
        description="If ``True``, will print the regular output from ``web`` functions.",
    )

    metric: Optional[ExpressionType] = Field(
        None,
        title="Objective Metric",
        description="Serializable expression defining the objective function.",
    )

    @abc.abstractmethod
    def _resolve_mirror_symmetry(self, array_shape: tuple[int, ...]) -> Optional[MirrorSymmetry]:
        """Mirror boundaries relevant to a design-region array for this inverse design."""

    def make_objective_fn(
        self, post_process_fn: Optional[Callable] = None, maximize: bool = True
    ) -> Callable[[anp.ndarray], tuple[float, dict]]:
        """Construct the objective function for this InverseDesign object."""

        if (post_process_fn is None) and (self.metric is None):
            raise ValueError("Either 'post_process_fn' or 'metric' must be provided.")

        if (post_process_fn is not None) and (self.metric is not None):
            raise ValueError("Provide only one of 'post_process_fn' or 'metric', not both.")

        direction_multiplier = 1 if maximize else -1

        def objective_fn(params: anp.ndarray, aux_data: Optional[dict] = None) -> float:
            """Full objective function."""
            symmetry = self._resolve_mirror_symmetry(params.shape)
            data = self.to_simulation_data(params=params, symmetry=symmetry)

            if self.metric is None:
                post_process_val = post_process_fn(data)
            elif isinstance(data, td.SimulationData):
                post_process_val = self.metric.evaluate(data)
            elif getattr(data, "type", None) == "BatchData":
                raise NotImplementedError("Metrics currently do not support 'BatchData'")
            else:
                raise ValueError(f"Invalid data type: {type(data)}")

            penalty_value = self.design_region.penalty_value(params, symmetry=symmetry)
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

    def run(self, simulation: td.Simulation, **kwargs: Any) -> td.SimulationData:
        """Run a single tidy3d simulation."""
        from tidy3d.web import run

        kwargs.setdefault("verbose", self.verbose)
        kwargs.setdefault("task_name", self.task_name)
        return run(simulation, **kwargs)

    def run_async(self, simulations: dict[str, td.Simulation], **kwargs: Any) -> td.web.BatchData:
        """Run a batch of tidy3d simulations."""
        from tidy3d.web import run_async

        kwargs.setdefault("verbose", self.verbose)
        return run_async(simulations, **kwargs)


class InverseDesign(AbstractInverseDesign):
    """Container for an inverse design problem."""

    simulation: td.Simulation = Field(
        title="Base Simulation",
        description="Simulation without the design regions or monitors used in the objective fn.",
    )

    output_monitor_names: Optional[tuple[str, ...]] = Field(
        None,
        title="Output Monitor Names",
        description="Optional names of monitors whose data the differentiable output depends on."
        "If this field is left ``None``, the plugin will try to add all compatible monitors "
        "automatically. While this will work, there may be warnings if certain monitor types are "
        "not fully supported, for example ``FieldMonitor`` instances with ``.colocate != False``.",
    )

    _check_sim_pixel_size = check_pixel_size("simulation")

    @model_validator(mode="after")
    def _validate_model(self) -> Self:
        if not self.metric:
            return self
        for metric in self.metric.filter(Metric):
            InverseDesign._validate_metric_monitor_name(metric, self.simulation)
            InverseDesign._validate_metric_mode_index(metric, self.simulation)
            InverseDesign._validate_metric_f(metric, self.simulation)
        InverseDesign._validate_metric_data(self.metric, self.simulation)
        return self

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
            raise ValidationError(f"Failed to evaluate the metric expression: {e!s}") from e
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

    def separate_output_monitors(self, monitors: tuple[td.Monitor]) -> dict:
        """Separate monitors into output_monitors and regular monitors."""

        monitor_fields = {"monitors": [], "output_monitors": []}

        for monitor in monitors:
            key = "output_monitors" if self.is_output_monitor(monitor) else "monitors"
            monitor_fields[key].append(monitor)

        return monitor_fields

    def _resolve_mirror_symmetry(self, array_shape: tuple[int, ...]) -> Optional[MirrorSymmetry]:
        """Mirror boundaries relevant to a design-region array in this simulation."""
        return _resolve_mirror_symmetry(
            array_shape=array_shape,
            design_region=self.design_region,
            simulation=self.simulation,
        )

    def to_simulation(
        self, params: anp.ndarray, symmetry: Optional[MirrorSymmetry] = None
    ) -> td.Simulation:
        """Convert the ``InverseDesign`` to a corresponding ``td.Simulation`` with traced fields."""
        if symmetry is None:
            symmetry = self._resolve_mirror_symmetry(params.shape)

        # construct the design region to a regular structure
        design_region_structure = self.design_region.to_structure(params, symmetry=symmetry)

        # construct mesh override structures and a new grid spec, if applicable
        grid_spec = self.simulation.grid_spec
        mesh_override_structure = self.design_region.mesh_override_structure
        if mesh_override_structure:
            override_structures = list(self.simulation.grid_spec.override_structures)
            override_structures += [mesh_override_structure]
            grid_spec = grid_spec.updated_copy(override_structures=override_structures)

        return self.simulation.updated_copy(
            structures=[*list(self.simulation.structures), design_region_structure],
            grid_spec=grid_spec,
        )

    def to_simulation_data(
        self, params: anp.ndarray, symmetry: Optional[MirrorSymmetry] = None, **kwargs: Any
    ) -> td.SimulationData:
        """Convert the ``InverseDesign`` to a ``td.Simulation`` and run it."""
        simulation = self.to_simulation(params=params, symmetry=symmetry)
        return self.run(simulation, **kwargs)


class InverseDesignMulti(AbstractInverseDesign):
    """``InverseDesign`` with multiple simulations and corresponding postprocess functions."""

    simulations: tuple[td.Simulation, ...] = Field(
        title="Base Simulations",
        description="Set of simulation without the design regions or monitors used in the objective fn.",
    )

    output_monitor_names: Optional[tuple[Union[tuple[str, ...], None], ...]] = Field(
        None,
        title="Output Monitor Names",
        description="Optional names of monitors whose data the differentiable output depends on."
        "If this field is left ``None``, the plugin will try to add all compatible monitors "
        "automatically. While this will work, there may be warnings if certain monitor types are "
        "not fully supported, for example ``FieldMonitor`` instances with ``.colocate != False``.",
    )

    @field_validator("output_monitor_names", mode="before")
    @classmethod
    def _convert_list_to_tuple(cls, v: Any) -> Any:
        """Convert lists to tuples for output_monitor_names."""
        if v is None:
            return v
        if isinstance(v, list):
            # Convert outer list to tuple
            result = []
            for item in v:
                if isinstance(item, list):
                    # Convert inner list to tuple
                    result.append(tuple(item))
                else:
                    result.append(item)
            return tuple(result)
        return v

    _check_sim_pixel_size = check_pixel_size("simulations")

    @model_validator(mode="after")
    def _check_lengths(self) -> Self:
        """Check the lengths of all of the multi fields."""

        keys = ("simulations", "output_monitor_names")
        multi_dict = {key: getattr(self, key) for key in keys}
        sizes = {key: len(val) for key, val in multi_dict.items() if val is not None}

        if len(set(sizes.values())) != 1:
            raise ValueError(
                f"'MultiInverseDesign' requires that the fields {keys} must either "
                "have the same length or be left ``None``, if optional. Given fields with "
                "corresponding sizes of '{sizes}'."
            )

        self._resolve_mirror_symmetry(self.design_region.params_shape)
        return self

    @property
    def task_names(self) -> list[str]:
        """Task names associated with each of the simulations."""
        return [f"{self.task_name}_{i}" for i in range(len(self.simulations))]

    @property
    def designs(self) -> list[InverseDesign]:
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

    def _resolve_mirror_symmetry(self, array_shape: tuple[int, ...]) -> Optional[MirrorSymmetry]:
        """Mirror boundaries relevant to a shared design-region array across all simulations."""
        mirror_symmetries = [
            _resolve_mirror_symmetry(
                array_shape=array_shape,
                design_region=self.design_region,
                simulation=simulation,
            )
            for simulation in self.simulations
        ]

        if not mirror_symmetries:
            return None

        mirror_symmetry = mirror_symmetries[0]
        if any(sym != mirror_symmetry for sym in mirror_symmetries[1:]):
            raise ValidationError(
                "All simulations in 'InverseDesignMulti' must resolve to the same mirror "
                "symmetry for the shared design region."
            )
        return mirror_symmetry

    def to_simulation(
        self, params: anp.ndarray, symmetry: Optional[MirrorSymmetry] = None
    ) -> dict[str, td.Simulation]:
        r"""Convert the ``InverseDesign`` to a corresponding dict of ``td.Simulation``\s."""
        if symmetry is None:
            symmetry = self._resolve_mirror_symmetry(params.shape)
        simulation_list = [
            design.to_simulation(params, symmetry=symmetry) for design in self.designs
        ]
        return dict(zip(self.task_names, simulation_list))

    def to_simulation_data(
        self, params: anp.ndarray, symmetry: Optional[MirrorSymmetry] = None, **kwargs: Any
    ) -> td.web.BatchData:
        r"""Convert the ``InverseDesignMulti`` to a set of ``td.Simulation``\s and run async."""
        simulations = self.to_simulation(params, symmetry=symmetry)
        return self.run_async(simulations, **kwargs)


InverseDesignType = Union[InverseDesign, InverseDesignMulti]
