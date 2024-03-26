# container for everything defining the inverse design

import pydantic.v1 as pd
import typing

import jax.numpy as jnp

import tidy3d as td
import tidy3d.plugins.adjoint as tda

from .base import InvdesBaseModel
from .region import DesignRegionType


PostProcessFnType = typing.Callable[[tda.JaxSimulationData], float]


class InverseDesign(InvdesBaseModel):
    """Container for an inverse design problem."""

    simulation: td.Simulation = pd.Field(
        ...,
        title="Base Simulation",
        description="Simulation without the design regions or monitors used in the objective fn.",
    )

    design_region: DesignRegionType = pd.Field(
        ...,
        title="Design Region",
        description="Region within which we will optimize the simulation.",
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

    task_name: str = pd.Field(
        ...,
        title="Task Name",
        description="Task name to use in the objective function when running the ``JaxSimulation``.",
    )

    verbose: bool = pd.Field(
        False,
        title="Task Verbosity",
        description="If ``True``, will print the regular output from ``web.run(...)``.",
    )

    # TODO: test all of the options
    override_structure_dl: typing.Union[td.components.types.Size, typing.Literal[False]] = pd.Field(
        None,
        title="Design Region Override Structure Grid Sizes",
        description="Defines grid size when adding an ``override_structure`` to the "
        "``JaxSimulation.grid_spec`` corresponding to this design region. "
        "If left ``None``, ``invdes`` will mesh the simulation with the same resolution as the "
        "pixel resolution defined by the parameter array within the design region. "
        "This is advised if the pixel size is relatively close to the FDTD grid size. "
        "Specifying a ``tuple`` of 3 grid sizes for x, y, z will override this setting "
        "with the supplied values. Supplying ``False`` will completely leave out the "
        " override structure. We recommend setting this to ``False`` or specifying your own values "
        "if the pixel size of your design region is much larger than your simulation grid cell size"
        " as in that case, the mesh may create too low of a resolution in the design region. ",
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

    def separate_monitors(self, monitors: typing.Tuple[td.Monitor]) -> dict:
        """Separate monitors into output_monitors and regular monitors."""

        monitor_dict = dict(
            monitors=[],
            output_monitors=[],
        )

        for monitor in monitors:
            key = "output_monitors" if self.is_output_monitor(monitor) else "monitors"
            monitor_dict[key].append(monitor)

        return monitor_dict

    def to_jax_simulation(self, params: jnp.ndarray) -> tda.JaxSimulation:
        """Convert the ``InverseDesign`` to a corresponding ``tda.JaxSimulation`` given make_."""

        # construct the jax simulation from the simulation + design region
        design_region_structure = self.design_region.to_jax_structure(params)
        grid_spec = self.simulation.grid_spec

        # add override structure to grid spec, if specified
        if self.override_structure_dl is not False:
            if self.override_structure_dl is None:
                dl_mesh_override = self.design_region.step_sizes
            else:
                dl_mesh_override = self.override_structure_dl

            mesh_override_structure = td.MeshOverrideStructure(
                geometry=design_region_structure.geometry,
                dl=dl_mesh_override,
                enforce=True,
            )
            grid_spec = grid_spec.updated_copy(
                override_structures=list(self.simulation.grid_spec.override_structures)
                + [mesh_override_structure]
            )

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

        # separate regular monitors from output monitors
        monitor_dict = self.separate_monitors(monitors=jax_sim.monitors)
        # monitors = []
        # output_monitors = []
        # for mnt in jax_sim.monitors:
        #     output_mnt_types = tda.components.simulation.OutputMonitorTypes

        #     is_output_monitor_type = any(isinstance(mnt, mnt_type) for mnt_type in output_mnt_types)
        #     is_output_monitor_specified = (
        #         self.output_monitor_names is not None and mnt.name in self.output_monitor_names
        #     )

        #     if is_output_monitor_type and is_output_monitor_specified:
        #         output_monitors.append(mnt)
        #     else:
        #         monitors.append(mnt)

        return jax_sim.updated_copy(
            input_structures=[design_region_structure],
            grid_spec=grid_spec,
            **monitor_dict,
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
        sim_data = td.web.run(sim, task_name=task_name, verbose=self.verbose, **kwargs)
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
