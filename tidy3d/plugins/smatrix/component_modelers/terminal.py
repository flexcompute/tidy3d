"""Tool for generating an S matrix automatically from a Tidy3d simulation and lumped port definitions."""

from __future__ import annotations

from typing import Dict, Tuple, Union

import numpy as np
import pydantic.v1 as pd

from ....components.base import cached_property
from ....components.data.sim_data import SimulationData
from ....components.geometry.utils_2d import snap_coordinate_to_grid
from ....components.simulation import Simulation
from ....components.source import GaussianPulse
from ....components.types import Ax
from ....components.viz import add_ax_if_none, equal_aspect
from ....constants import C_0, fp_eps
from ....exceptions import ValidationError
from ....web.api.container import BatchData
from ..ports.base_lumped import LumpedPortDataArray
from ..ports.coaxial_lumped import CoaxialLumpedPort
from ..ports.rectangular_lumped import LumpedPort
from .base import FWIDTH_FRAC, AbstractComponentModeler


class TerminalComponentModeler(AbstractComponentModeler):
    """Tool for modeling two-terminal multiport devices and computing port parameters
    with lumped ports."""

    ports: Tuple[Union[LumpedPort, CoaxialLumpedPort], ...] = pd.Field(
        (),
        title="Lumped ports",
        description="Collection of lumped ports associated with the network. "
        "For each port, one simulation will be run with a lumped port source.",
    )

    @equal_aspect
    @add_ax_if_none
    def plot_sim(
        self, x: float = None, y: float = None, z: float = None, ax: Ax = None, **kwargs
    ) -> Ax:
        """Plot a :class:`.Simulation` with all sources added for each port, for troubleshooting."""

        plot_sources = []
        for port_source in self.ports:
            source_0 = port_source.to_source(
                self._source_time, snap_center=None, grid=self.simulation.grid
            )
            plot_sources.append(source_0)
        sim_plot = self.simulation.copy(update=dict(sources=plot_sources))
        return sim_plot.plot(x=x, y=y, z=z, ax=ax, **kwargs)

    @equal_aspect
    @add_ax_if_none
    def plot_sim_eps(
        self, x: float = None, y: float = None, z: float = None, ax: Ax = None, **kwargs
    ) -> Ax:
        """Plot permittivity of the :class:`.Simulation` with all sources added for each port."""

        plot_sources = []
        for port_source in self.ports:
            source_0 = port_source.to_source(
                self._source_time, snap_center=None, grid=self.simulation.grid
            )
            plot_sources.append(source_0)
        sim_plot = self.simulation.copy(update=dict(sources=plot_sources))
        return sim_plot.plot_eps(x=x, y=y, z=z, ax=ax, **kwargs)

    @cached_property
    def sim_dict(self) -> Dict[str, Simulation]:
        """Generate all the :class:`.Simulation` objects for the port parameter calculation."""

        sim_dict = {}

        port_voltage_monitors = [
            port.to_voltage_monitor(self.freqs, snap_center=None) for port in self.ports
        ]
        port_current_monitors = [
            port.to_current_monitor(self.freqs, snap_center=None) for port in self.ports
        ]
        lumped_resistors = [port.to_load(snap_center=None) for port in self.ports]

        # Create a mesh override for each port in case refinement is needed.
        # The port is a flat surface, but when computing the port current,
        # we'll eventually integrate the magnetic field just above and below
        # this surface, so the mesh override needs to ensure that the mesh
        # is fine enough not only in plane, but also in the normal direction.
        # So in the normal direction, we'll make sure there are at least
        # 2 cell layers above and below whose size is the same as the in-plane
        # cell size in the override region. Also, to ensure that the port itself
        # is aligned with a grid boundary in the normal direction, two separate
        # override regions are defined, one above and one below the analytical
        # port region.
        mesh_overrides = []
        for port, lumped_resistor in zip(self.ports, lumped_resistors):
            if port.num_grid_cells:
                mesh_overrides.extend(lumped_resistor.to_mesh_overrides())

        new_mnts = list(self.simulation.monitors) + port_voltage_monitors + port_current_monitors

        # also, use the highest frequency in the simulation to define the grid, rather than the
        # source's central frequency, to ensure an accurate solution over the entire range
        grid_spec = self.simulation.grid_spec.copy(
            update={
                "wavelength": C_0 / np.max(self.freqs),
                "override_structures": list(self.simulation.grid_spec.override_structures)
                + mesh_overrides,
            }
        )

        # Checking if snapping is required needs the simulation to be created, because the
        # elements may impact the final grid discretization
        snap_and_recreate = False
        snap_centers = []
        for port in self.ports:
            update_dict = dict(
                monitors=new_mnts,
                lumped_elements=lumped_resistors,
                grid_spec=grid_spec,
            )

            sim_copy = self.simulation.copy(update=update_dict)
            port_source = port.to_source(self._source_time, snap_center=None, grid=sim_copy.grid)
            sim_copy = sim_copy.updated_copy(sources=[port_source])
            task_name = self._task_name(port=port)
            sim_dict[task_name] = sim_copy

            # Check if snapping to grid is needed
            port_center_on_axis = port.center[port.injection_axis]
            new_port_center = snap_coordinate_to_grid(
                sim_copy.grid, port_center_on_axis, port.injection_axis
            )
            snap_centers.append(new_port_center)
            if not np.isclose(port_center_on_axis, new_port_center, fp_eps, fp_eps):
                snap_and_recreate = True

        # Check if snapping was needed and if it was recreate the simulations
        if snap_and_recreate:
            sim_dict.clear()
            port_voltage_monitors = [
                port.to_voltage_monitor(self.freqs, snap_center=center)
                for port, center in zip(self.ports, snap_centers)
            ]
            port_current_monitors = [
                port.to_current_monitor(self.freqs, snap_center=center)
                for port, center in zip(self.ports, snap_centers)
            ]
            lumped_resistors = [
                port.to_load(snap_center=center) for port, center in zip(self.ports, snap_centers)
            ]
            new_mnts = (
                list(self.simulation.monitors) + port_voltage_monitors + port_current_monitors
            )
            for port, center in zip(self.ports, snap_centers):
                update_dict = dict(
                    monitors=new_mnts,
                    lumped_elements=lumped_resistors,
                    grid_spec=grid_spec,
                )

                sim_copy = self.simulation.copy(update=update_dict)
                port_source = port.to_source(
                    self._source_time, snap_center=center, grid=sim_copy.grid
                )
                sim_copy = sim_copy.updated_copy(sources=[port_source])
                task_name = self._task_name(port=port)
                sim_dict[task_name] = sim_copy

        # Check final simulations for grid size at ports
        for _, sim in sim_dict.items():
            TerminalComponentModeler._check_grid_size_at_ports(sim, self.ports)
        return sim_dict

    @cached_property
    def _source_time(self):
        """Helper to create a time domain pulse for the frequeny range of interest."""
        freq0 = np.mean(self.freqs)
        fdiff = max(self.freqs) - min(self.freqs)
        fwidth = max(fdiff, freq0 * FWIDTH_FRAC)
        return GaussianPulse(
            freq0=freq0, fwidth=fwidth, remove_dc_component=self.remove_dc_component
        )

    def _construct_smatrix(self, batch_data: BatchData) -> LumpedPortDataArray:
        """Post process ``BatchData`` to generate scattering matrix."""

        port_names = [port.name for port in self.ports]

        values = np.zeros(
            (len(port_names), len(port_names), len(self.freqs)),
            dtype=complex,
        )
        coords = dict(
            port_out=port_names,
            port_in=port_names,
            f=np.array(self.freqs),
        )
        a_matrix = LumpedPortDataArray(values, coords=coords)
        b_matrix = a_matrix.copy(deep=True)

        def port_ab(port: Union[LumpedPort, CoaxialLumpedPort], sim_data: SimulationData):
            """Helper to compute the port incident and reflected power waves."""
            voltage = port.compute_voltage(sim_data)
            current = port.compute_current(sim_data)

            power_a = (voltage + port.impedance * current) / 2 / np.sqrt(np.real(port.impedance))
            power_b = (voltage - port.impedance * current) / 2 / np.sqrt(np.real(port.impedance))
            return power_a, power_b

        # loop through source ports
        for port_in in self.ports:
            sim_data = batch_data[self._task_name(port=port_in)]

            for port_out in self.ports:
                a_out, b_out = port_ab(port_out, sim_data)
                a_matrix.loc[
                    dict(
                        port_in=port_in.name,
                        port_out=port_out.name,
                    )
                ] = a_out

                b_matrix.loc[
                    dict(
                        port_in=port_in.name,
                        port_out=port_out.name,
                    )
                ] = b_out

        s_matrix = self.ab_to_s(a_matrix, b_matrix)

        return s_matrix

    @pd.validator("simulation")
    def _validate_3d_simulation(cls, val):
        """Error if :class:`.Simulation` is not a 3D simulation"""

        if val.size.count(0.0) > 0:
            raise ValidationError(
                f"'{cls.__name__}' must be setup with a 3D simulation with all sizes greater than 0."
            )
        return val

    @staticmethod
    def _check_grid_size_at_ports(
        simulation: Simulation, ports: list[Union[LumpedPort, CoaxialLumpedPort]]
    ):
        """Raises :class:`.SetupError` if the grid is too coarse at port locations"""
        yee_grid = simulation.grid.yee
        for port in ports:
            port._check_grid_size(yee_grid)
