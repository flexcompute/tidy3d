"""Defines a jax-compatible SimulationData."""

from __future__ import annotations

from typing import Dict, List, Tuple, Union

import numpy as np
import pydantic.v1 as pd
import xarray as xr
from jax.tree_util import register_pytree_node_class

from .....components.data.monitor_data import FieldData, MonitorDataType, PermittivityData
from .....components.data.sim_data import SimulationData
from .....components.source.current import PointDipole
from .....components.source.time import GaussianPulse
from .....log import log
from ..base import JaxObject
from ..simulation import JaxInfo, JaxSimulation
from .monitor_data import JAX_MONITOR_DATA_MAP, JaxMonitorDataType


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

    task_id: str = pd.Field(
        None,
        title="Task ID",
        description="Optional field storing the task_id for the original JaxSimulation.",
    )

    def get_poynting_vector(self, field_monitor_name: str) -> xr.Dataset:
        """return ``xarray.Dataset`` of the Poynting vector at Yee cell centers.

        Calculated values represent the instantaneous Poynting vector for time-domain fields and the
        complex vector for frequency-domain: ``S = 1/2 E × conj(H)``.

        Only the available components are returned, e.g., if the indicated monitor doesn't include
        field component `"Ex"`, then `"Sy"` and `"Sz"` will not be calculated.

        Parameters
        ----------
        field_monitor_name : str
            Name of field monitor used in the original :class:`Simulation`.

        Returns
        -------
        xarray.DataArray
            DataArray containing the Poynting vector calculated based on the field components
            colocated at the center locations of the Yee grid.
        """

        if field_monitor_name in self.output_monitor_data:
            raise NotImplementedError(
                "Adjoint support for differentiation with respect to Poynting vector not available."
            )

        return super().get_poynting_vector(field_monitor_name)

    @property
    def grad_data_symmetry(self) -> Tuple[FieldData, ...]:
        """``self.grad_data`` but with ``symmetry_expanded_copy`` applied."""
        return tuple(data.symmetry_expanded_copy for data in self.grad_data)

    @property
    def grad_eps_data_symmetry(self) -> Tuple[FieldData, ...]:
        """``self.grad_eps_data`` but with ``symmetry_expanded_copy`` applied."""
        return tuple(data.symmetry_expanded_copy for data in self.grad_eps_data)

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
    def from_sim_data(
        cls, sim_data: SimulationData, jax_info: JaxInfo, task_id: str = None
    ) -> JaxSimulationData:
        """Construct a :class:`.JaxSimulationData` instance from a :class:`.SimulationData`."""

        self_dict = sim_data.dict(exclude={"type", "simulation", "data"})

        # convert the simulation to JaxSimulation
        jax_sim = JaxSimulation.from_simulation(simulation=sim_data.simulation, jax_info=jax_info)

        # construct JaxSimulationData with no data (yet)
        self_dict["simulation"] = jax_sim
        self_dict["data"] = ()

        data_dict = cls.split_data(mnt_data=sim_data.data, jax_info=jax_info)

        # convert the output data to the proper jax type
        output_data_list = []
        for mnt_data in data_dict["output_data"]:
            mnt_data_type_str = type(mnt_data)
            if mnt_data_type_str not in JAX_MONITOR_DATA_MAP:
                raise KeyError(
                    f"MonitorData type '{mnt_data_type_str}' "
                    "not currently supported by adjoint plugin."
                )
            mnt_data_type = JAX_MONITOR_DATA_MAP[mnt_data_type_str]
            jax_mnt_data = mnt_data_type.from_monitor_data(mnt_data)
            output_data_list.append(jax_mnt_data)
        data_dict["output_data"] = output_data_list
        self_dict.update(data_dict)
        self_dict.update(dict(task_id=task_id))

        return cls.parse_obj(self_dict)

    @classmethod
    def split_fwd_sim_data(
        cls, sim_data: SimulationData, jax_info: JaxInfo
    ) -> Tuple[SimulationData, SimulationData]:
        """Split a :class:`.SimulationData` into two parts, containing user and gradient data."""

        sim = sim_data.simulation

        data_dict = cls.split_data(mnt_data=sim_data.data, jax_info=jax_info)
        user_data = data_dict["data"] + data_dict["output_data"]
        adjoint_data = data_dict["grad_data"] + data_dict["grad_eps_data"]

        mnt_dict = JaxSimulation.split_monitors(
            monitors=sim_data.simulation.monitors, jax_info=jax_info
        )
        user_mnts = mnt_dict["monitors"] + mnt_dict["output_monitors"]
        adjoint_mnts = mnt_dict["grad_monitors"] + mnt_dict["grad_eps_monitors"]

        user_sim = sim.updated_copy(monitors=user_mnts)
        adjoint_sim = sim.updated_copy(monitors=adjoint_mnts)

        user_sim_data = sim_data.updated_copy(data=user_data, simulation=user_sim)
        adjoint_sim_data = sim_data.updated_copy(data=adjoint_data, simulation=adjoint_sim)

        return user_sim_data, adjoint_sim_data

    def make_adjoint_simulation(self, fwidth: float, run_time: float) -> JaxSimulation:
        """Make an adjoint simulation out of the data provided (generally, the vjp sim data)."""

        sim_fwd = self.simulation

        # grab boundary conditions with flipped bloch vectors (for adjoint)
        bc_adj = sim_fwd.boundary_spec.flipped_bloch_vecs

        # add all adjoint sources and boundary conditions (at same time for BC validators to work)
        adj_srcs = []
        for mnt_data_vjp in self.output_data:
            for adj_source in mnt_data_vjp.to_adjoint_sources(fwidth=fwidth):
                adj_srcs.append(adj_source)

        # in this case (no adjoint sources) give it an "empty" source
        if not adj_srcs:
            log.warning(
                "No adjoint sources, making a mock source with amplitude = 0. "
                "All gradients will be zero for anything depending on this simulation's data. "
                "This comes up when a simulation's data contributes to the value of an objective "
                "function but the contribution from each member of the data is 0. "
                "If this is intended (eg. if using 'jnp.max()' of several simulation results), "
                "please ignore. Otherwise, this can suggest a mistake in your objective function."
            )

            # set a zero-amplitude source
            adj_srcs.append(
                PointDipole(
                    center=sim_fwd.center,
                    polarization="Ez",
                    source_time=GaussianPulse(
                        freq0=sim_fwd.freqs_adjoint[0],
                        fwidth=sim_fwd._fwidth_adjoint,
                        amplitude=0.0,
                    ),
                )
            )

            # set a very short run time relative to the fwidth
            run_time = 2 / fwidth

        update_dict = dict(
            boundary_spec=bc_adj,
            sources=adj_srcs,
            monitors=(),
            output_monitors=(),
            run_time=run_time,
            normalize_index=None,  # normalize later, frequency-by-frequency
        )

        update_dict.update(
            sim_fwd.get_grad_monitors(
                input_structures=sim_fwd.input_structures,
                freqs_adjoint=sim_fwd.freqs_adjoint,
                include_eps_mnts=False,
            )
        )

        # set the ADJ grid spec wavelength to the FWD wavelength (for same meshing)
        grid_spec_fwd = sim_fwd.grid_spec
        if len(sim_fwd.sources) and grid_spec_fwd.wavelength is None:
            wavelength_fwd = grid_spec_fwd.wavelength_from_sources(sim_fwd.sources)
            grid_spec_adj = grid_spec_fwd.updated_copy(wavelength=wavelength_fwd)
            update_dict.update(dict(grid_spec=grid_spec_adj))

        return sim_fwd.updated_copy(**update_dict)

    def normalize_adjoint_fields(self) -> JaxSimulationData:
        """Make copy of jax_sim_data with grad_data (fields) normalized by adjoint sources."""

        grad_data_norm = []
        for field_data in self.grad_data:
            field_components_norm = {}
            for field_name, field_component in field_data.field_components.items():
                freqs = field_component.coords["f"]
                norm_factor_f = np.zeros(len(freqs), dtype=complex)
                for i, freq in enumerate(freqs):
                    freq = float(freq)
                    for source_index, source in enumerate(self.simulation.sources):
                        if source.source_time.freq0 == freq and source.source_time.amplitude > 0:
                            spectrum_fn = self.source_spectrum(source_index)
                            norm_factor_f[i] = complex(spectrum_fn([freq])[0])

                norm_factor_f_darr = xr.DataArray(norm_factor_f, coords=dict(f=freqs))
                field_component_norm = field_component / norm_factor_f_darr
                field_components_norm[field_name] = field_component_norm

            field_data_norm = field_data.updated_copy(**field_components_norm)
            grad_data_norm.append(field_data_norm)

        return self.updated_copy(grad_data=grad_data_norm)
