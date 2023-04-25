"""Defines a jax-compatible SimulationData."""
from __future__ import annotations

from typing import Tuple, Dict, Union, List

import pydantic as pd

from jax.tree_util import register_pytree_node_class

from .....components.data.monitor_data import MonitorDataType, FieldData, PermittivityData
from .....components.data.sim_data import SimulationData

from ..base import JaxObject
from ..simulation import JaxSimulation, JaxInfo
from .monitor_data import JaxMonitorDataType, JAX_MONITOR_DATA_MAP


@register_pytree_node_class
class JaxSimulationData(SimulationData, JaxObject):
    """A :class:`.SimulationData` registered with jax."""

    output_data: Tuple[JaxMonitorDataType, ...] = pd.Field(
        (),
        title="Jax Data",
        description="Tuple of Jax-compatible data associated with output monitors.",
        jax_field=True,
    )

    grad_data: Tuple[FieldData, ...] = pd.Field(
        (),
        title="Gradient Field Data",
        description="Tuple of monitor data storing fields associated with the input structures.",
    )

    grad_eps_data: Tuple[PermittivityData, ...] = pd.Field(
        (),
        title="Gradient Permittivity Data",
        description="Tuple of monitor data storing epsilon associated with the input structures.",
    )

    simulation: JaxSimulation = pd.Field(
        ...,
        title="Simulation",
        description="The jax-compatible simulation corresponding to the data.",
    )

    @property
    def output_monitor_data(self) -> Dict[str, JaxMonitorDataType]:
        """Dictionary of ``.output_data`` monitor ``.name`` to the corresponding data."""
        return {monitor_data.monitor.name: monitor_data for monitor_data in self.output_data}

    @property
    def monitor_data(self) -> Dict[str, Union[JaxMonitorDataType, MonitorDataType]]:
        """Dictionary of ``.output_data`` monitor ``.name`` to the corresponding data."""
        reg_mnt_data = {monitor_data.monitor.name: monitor_data for monitor_data in self.data}
        reg_mnt_data.update(self.output_monitor_data)
        return reg_mnt_data

    @staticmethod
    def split_data(
        mnt_data: List[MonitorDataType], jax_info: JaxInfo
    ) -> Dict[str, List[MonitorDataType]]:
        """Split list of monitor data into data, output_data, grad_data, and grad_eps_data."""
        # Get information needed to split the full data list
        len_output_data = jax_info.num_output_monitors
        len_grad_data = jax_info.num_grad_monitors
        len_grad_eps_data = jax_info.num_grad_eps_monitors
        len_data = len(mnt_data) - len_output_data - len_grad_data - len_grad_eps_data

        # split the data list into regular data, output_data, and grad_data
        all_data = list(mnt_data)
        data = all_data[:len_data]
        output_data = all_data[len_data : len_data + len_output_data]
        grad_data = all_data[
            len_data + len_output_data : len_data + len_output_data + len_grad_data
        ]
        grad_eps_data = all_data[len_data + len_output_data + len_grad_data :]

        return dict(
            data=data, output_data=output_data, grad_data=grad_data, grad_eps_data=grad_eps_data
        )

    @classmethod
    def from_sim_data(cls, sim_data: SimulationData, jax_info: JaxInfo) -> JaxSimulationData:
        """Construct a :class:`.JaxSimulationData` instance from a :class:`.SimulationData`."""

        self_dict = sim_data.dict(exclude={"type", "simulation", "data"})

        # convert the simulation to JaxSimulation
        jax_sim = JaxSimulation.from_simulation(simulation=sim_data.simulation, jax_info=jax_info)

        # construct JaxSimulationData with no data (yet)
        self_dict["simulation"] = jax_sim
        self_dict["data"] = ()

        data_dict = cls.split_data(mnt_data=sim_data.data, jax_info=jax_info)

        # convert the output data to the proper jax type
        data_dict["output_data"] = [
            JAX_MONITOR_DATA_MAP[type(x)].from_monitor_data(x) for x in data_dict["output_data"]
        ]

        self_dict.update(data_dict)

        return cls.parse_obj(self_dict)

    # def to_sim_data(self) -> Tuple[SimulationData, JaxInfo]:
    #     """Convert regular :class:`.Simulation` to a `JaxSimulationData`."""

    #     sim, jax_info = self.simulation.to_simulation()

    #     data_combined = self.data + self.output_data + self.grad_data + self.grad_eps_data

    #     sim_data = SimulationData(simulation=sim, data=data_combined)

    #     return sim_data, jax_info

    @classmethod
    def split_fwd_sim_data(
        cls, sim_data: SimulationData, jax_info: JaxInfo
    ) -> Tuple[SimulationData, SimulationData]:
        """Split a :class:`.SimulationData` into two parts, containing user and gradient data."""

        data_dict = cls.split_data(mnt_data=sim_data.data, jax_info=jax_info)

        user_data = data_dict["data"] + data_dict["output_data"]
        adjoint_data = data_dict["grad_data"] + data_dict["grad_eps_data"]

        user_sim_data = sim_data.updated_copy(data=user_data)
        adjoint_sim_data = sim_data.updated_copy(data=adjoint_data)

        return user_sim_data, adjoint_sim_data

    def make_adjoint_simulation(self, fwidth: float) -> JaxSimulation:
        """Make an adjoint simulation out of the data provided (generally, the vjp sim data)."""

        # grab boundary conditions with flipped bloch vectors (for adjoint)
        bc_adj = self.simulation.boundary_spec.flipped_bloch_vecs

        # add all adjoint sources and boundary conditions (at same time for BC validators to work)
        adj_srcs = []
        for mnt_data_vjp in self.output_data:
            for adj_source in mnt_data_vjp.to_adjoint_sources(fwidth=fwidth):
                adj_srcs.append(adj_source)

        update_dict = dict(boundary_spec=bc_adj, sources=adj_srcs, monitors=(), output_monitors=())
        update_dict.update(self.simulation.get_grad_monitors())
        return self.simulation.updated_copy(**update_dict)
