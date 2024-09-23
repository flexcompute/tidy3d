# container for everything defining the inverse design

from __future__ import annotations

import abc
import typing

import autograd.numpy as anp
import pydantic.v1 as pd

import tidy3d as td
import tidy3d.web as web
from tidy3d.components.autograd import get_static
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
            elif isinstance(data, web.BatchData):
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
        kwargs.setdefault("task_name", self.task_name)
        return web.run(simulation, verbose=self.verbose, **kwargs)


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

    def to_simulation_data(self, params: anp.ndarray, **kwargs) -> web.BatchData:
        """Convert the ``InverseDesignMulti`` to a set of ``td.Simulation``s and run async."""
        simulations = self.to_simulation(params)
        kwargs.setdefault("verbose", self.verbose)
        return web.run_async(simulations, **kwargs)


InverseDesignType = typing.Union[InverseDesign, InverseDesignMulti]
