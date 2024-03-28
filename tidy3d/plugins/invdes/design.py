# container for everything defining the inverse design

import pydantic.v1 as pd
import typing
import abc

import jax.numpy as jnp

import tidy3d as td
import tidy3d.plugins.adjoint as tda

from .base import InvdesBaseModel
from .region import DesignRegionType


PostProcessFnType = typing.Callable[[tda.JaxSimulationData], float]


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

    @property
    @abc.abstractmethod
    def objective_fn(self) -> typing.Callable[[jnp.ndarray], float]:
        """construct the objective function for this ``InverseDesign`` object."""

    def _add_verbosity_to_kwargs(self, **kwargs) -> dict:
        """Add ``self.verbose`` to the ``kwargs`` passed to ``web`` functions."""
        if "verbose" not in kwargs:
            kwargs["verbose"] = self.verbose
        return kwargs


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

    post_process_fn: PostProcessFnType = pd.Field(
        ...,
        title="Post-Process Function",
        description="Function of ``JaxSimulationData`` that returns a ``float`` contribution "
        "to the objective function",
    )

    @pd.validator("post_process_fn", always=True)
    def _add_kwargs(cls, val):
        """Make sure the call signature of the post process function accepts kwargs."""

        def post_process_fn_with_kwargs(*args, **kwargs):
            try:
                return val(*args, **kwargs)
            except TypeError:
                return val(*args)

        return post_process_fn_with_kwargs

    def is_output_monitor(self, monitor: td.Monitor) -> bool:
        """Whether a monitor is added to the ``JaxSimulation`` as an ``output_monitor``."""

        output_mnt_types = tda.components.simulation.OutputMonitorTypes

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

    def to_jax_simulation(self, params: jnp.ndarray) -> tda.JaxSimulation:
        """Convert the ``InverseDesign`` to a corresponding ``tda.JaxSimulation`` given make_."""

        # construct the jax structure design region
        design_region_structure = self.design_region.to_jax_structure(params)

        # construct mesh override structures and a new grid spec, if applicable
        grid_spec = self.simulation.grid_spec
        mesh_override_structure = self.design_region.mesh_override_structure
        if mesh_override_structure:
            override_structures = list(self.simulation.grid_spec.override_structures)
            override_structures += [mesh_override_structure]
            grid_spec = grid_spec.updated_copy(override_structures=override_structures)

        # make a base jax simulation corresponding to this InverseDesign
        jax_info = tda.components.simulation.JaxInfo(
            num_input_structures=0,
            num_output_monitors=0,
            num_grad_monitors=0,
            num_grad_eps_monitors=0,
        )
        jax_sim = tda.JaxSimulation.from_simulation(
            self.simulation,
            jax_info=jax_info,
        )

        # split monitors into regular an output
        monitor_fields = self.separate_output_monitors(monitors=jax_sim.monitors)

        return jax_sim.updated_copy(
            input_structures=[design_region_structure],
            grid_spec=grid_spec,
            **monitor_fields,
        )

    def to_simulation(self, params: jnp.ndarray) -> td.Simulation:
        """Convert the ``InverseDesign`` to a corresponding ``td.Simulation`` given params."""
        jax_sim = self.to_jax_simulation(params=params)
        return jax_sim.to_simulation()[0]

    def to_simulation_data(
        self, params: jnp.ndarray, task_name: str, **kwargs
    ) -> td.SimulationData:
        """Convert the ``InverseDesign`` to a ``td.Simulation`` and run it."""
        sim = self.to_simulation(params=params)

        kwargs = self._add_verbosity_to_kwargs(**kwargs)

        sim_data = td.web.run(sim, task_name=task_name, **kwargs)
        return sim_data

    @property
    def objective_fn(self) -> typing.Callable[[jnp.ndarray], float]:
        """construct the objective function for this ``InverseDesign`` object."""

        def objective_fn(params: jnp.ndarray, **kwargs_postprocess) -> float:
            """Full objective function."""

            jax_sim = self.to_jax_simulation(params=params)

            # run the jax simulation
            jax_sim_data = tda.web.run(jax_sim, task_name=self.task_name, verbose=self.verbose)

            # construct objective function values
            post_process_val = self.post_process_fn(jax_sim_data, **kwargs_postprocess)
            penalty_value = self.design_region.penalty_value(params)
            objective_fn_val = post_process_val - penalty_value

            # return objective value and auxiliary data
            aux_data = dict(
                penalty=penalty_value,
                post_process_val=post_process_val,
            )
            return objective_fn_val, aux_data

        return objective_fn


class InverseDesignMulti(AbstractInverseDesign):
    """``InverseDesign`` with multiple simulations and corresponding postprocess functions."""

    simulations: typing.Tuple[td.Simulation, ...] = pd.Field(
        ...,
        title="Base Simulations",
        description="Set of simulation without the design regions or monitors used in the objective fn.",
    )

    post_process_fn: PostProcessFnType = pd.Field(
        ...,
        title="Post-Process Function",
        description="``list`` of ``JaxSimulationData`` instances corresponding to "
        "each ``Simulation`` in ``.simulations``. "
        "Should return a ``float`` contribution to the objective function.",
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
    def designs(self) -> typing.List[InverseDesign]:
        """List of individual ``InverseDesign`` objects corresponding to this instance."""

        designs_list = []
        for i, sim in enumerate(self.simulations):
            des_i = InverseDesign(
                design_region=self.design_region,
                simulation=sim,
                post_process_fn=lambda *args, **kwargs: 0.0,
                verbose=self.verbose,
                task_name=self.task_name + "_{i}",
            )
            if self.output_monitor_names is not None:
                des_i = des_i.updated_copy(output_monitor_names=self.output_monitor_names[i])

            designs_list.append(des_i)

        return designs_list

    @property
    def objective_fn(self) -> typing.Callable[[jnp.ndarray], float]:
        """construct the objective function for this ``InverseDesign`` object."""

        designs = self.designs

        def objective_fn(params: jnp.ndarray, **kwargs_postprocess) -> float:
            """Full objective function."""

            # construct the jax simulations
            jax_sims = [design.to_jax_simulation(params=params) for design in designs]

            # run the jax simulations
            sim_data_list = tda.web.run_async(jax_sims, verbose=self.verbose)

            # compute objective function values and sum them
            post_process_val = self.post_process_fn(sim_data_list, **kwargs_postprocess)

            # construct penalty value
            penalty_value = self.design_region.penalty_value(params)

            # combine objective
            objective_fn_val = post_process_val - penalty_value

            # return objective value and auxiliary data
            aux_data = dict(
                penalty=penalty_value,
                post_process_val=post_process_val,
            )
            return objective_fn_val, aux_data

        return objective_fn

    def to_simulation(self, params: jnp.ndarray) -> typing.List[td.Simulation]:
        """Convert the ``InverseDesign`` to a corresponding list of ``td.Simulation``s given params."""
        return [design.to_simulation(params) for design in self.designs]

    def to_simulation_data(
        self, params: jnp.ndarray, task_name: str, **kwargs
    ) -> typing.List[td.SimulationData]:
        """Convert the ``InverseDesignMulti`` to a set of ``td.Simulation``s and run async."""
        simulations = self.to_simulation(params)

        def get_task_name(i: int) -> str:
            """task name for the i-th task."""
            return f"{task_name}_{str(i)}"

        task_names = [get_task_name(i) for i in range(len(simulations))]
        sim_dict = dict(zip(task_names, simulations))

        kwargs = self._add_verbosity_to_kwargs(**kwargs)

        batch_data = td.web.run_async(sim_dict, **kwargs)
        return [batch_data[tn] for tn in task_names]


InverseDesignType = typing.Union[InverseDesign, InverseDesignMulti]
