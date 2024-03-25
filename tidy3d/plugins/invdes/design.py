# container for everything defining the inverse design

import pydantic.v1 as pd
import typing

import jax.numpy as jnp

import tidy3d as td
from tidy3d.plugins.adjoint.components.simulation import JaxInfo

import tidy3d.plugins.adjoint as tda

from .region import DesignRegion


class InverseDesign(td.components.base.Tidy3dBaseModel):
    """Container for an inverse design problem."""

    simulation: td.Simulation = pd.Field(
        ...,
        title="Base Simulation",
        description="Simulation without the design regions or monitors used in the objective fn.",
    )

    design_region: DesignRegion = pd.Field(
        ...,
        title="Design Region",
        description="Region within which we will optimize the simulation.",
    )

    output_monitor_names: typing.Tuple[str, ...] = pd.Field(
        (),
        title="Output Monitor Names",
        description="Name of monitors whose data the differentiable output depends on.",
    )

    post_process_fn: typing.Callable[[tda.JaxSimulationData], float] = pd.Field(...)

    task_name: str = pd.Field(...)

    def to_jax_simulation(self, params: jnp.ndarray) -> tda.JaxSimulation:
        """Convert the ``InverseDesign`` to a corresponding ``tda.JaxSimulation`` given make_."""

        # construct the jax simulation from the simulation + design region
        design_region_structure = self.design_region.to_jax_structure(params)

        # TODO: do we want to add mesh override structure if the pixels are large / low res?
        mesh_override_structure = td.MeshOverrideStructure(
            geometry=design_region_structure.geometry,
            dl=self.design_region.step_sizes,
            enforce=True,
        )
        grid_spec = self.simulation.grid_spec.updated_copy(
            override_structures=list(self.simulation.grid_spec.override_structures)
            + [mesh_override_structure]
        )
        jax_info = JaxInfo(
            num_input_structures=0,
            num_output_monitors=0,
            num_grad_monitors=0,
            num_grad_eps_monitors=0,
        )

        jax_sim = tda.JaxSimulation.from_simulation(
            self.simulation,
            jax_info=jax_info,
        )

        monitors = []
        output_monitors = []
        for mnt in jax_sim.monitors:
            if mnt.name in self.output_monitor_names:
                output_monitors.append(mnt)
            else:
                monitors.append(mnt)

        return jax_sim.updated_copy(
            input_structures=[design_region_structure],
            output_monitors=output_monitors,
            monitors=monitors,
            grid_spec=grid_spec,
        )

    @property
    def objective_fn(self) -> typing.Callable[[jnp.ndarray], float]:
        """construct the objective function for this ``InverseDesign`` object."""

        def objective_fn(params: jnp.ndarray) -> float:
            """Full objective function."""

            # TODO: I dont think post_proc_kwargs is ever exposed to the user

            jax_sim = self.to_jax_simulation(params=params)

            # run the jax simulation
            jax_sim_data = tda.web.run(jax_sim, task_name=self.task_name, verbose=False)

            # construct objective function values
            post_process_val = self.post_process_fn(jax_sim_data)
            penalty_value = self.design_region.penalty_value(params)
            objective_fn_val = post_process_val - penalty_value

            # return objective value and auxiliary data
            aux_data = dict(
                penalty=penalty_value,
                post_process_val=post_process_val,
                simulation=jax_sim.to_simulation()[0],
            )
            return objective_fn_val, aux_data

        return objective_fn
