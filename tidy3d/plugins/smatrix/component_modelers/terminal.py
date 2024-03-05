"""Tool for generating an S matrix automatically from a Tidy3d simulation and lumped port definitions."""

from __future__ import annotations

from typing import Tuple, Dict

import pydantic.v1 as pd
import numpy as np
import xarray as xr

from ....constants import C_0, fp_eps
from ....components.simulation import Simulation
from ....components.geometry.utils_2d import snap_coordinate_to_grid
from ....components.data.sim_data import SimulationData
from ....components.source import GaussianPulse
from ....components.types import Ax
from ....components.viz import add_ax_if_none, equal_aspect
from ....components.base import cached_property
from ....exceptions import ValidationError
from ....web.api.container import BatchData

from .base import AbstractComponentModeler, FWIDTH_FRAC
from ..ports.lumped import LumpedPortDataArray, LumpedPort


class TerminalComponentModeler(AbstractComponentModeler):
    """Tool for modeling two-terminal multiport devices and computing port parameters
    with lumped ports."""

    ports: Tuple[LumpedPort, ...] = pd.Field(
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
        """Plot a :class:`Simulation` with all sources added for each port, for troubleshooting."""

        plot_sources = []
        for port_source in self.ports:
            source_0 = port_source.to_source(self._source_time, snap_center=None)
            plot_sources.append(source_0)
        sim_plot = self.simulation.copy(update=dict(sources=plot_sources))
        return sim_plot.plot(x=x, y=y, z=z, ax=ax, **kwargs)

    @equal_aspect
    @add_ax_if_none
    def plot_sim_eps(
        self, x: float = None, y: float = None, z: float = None, ax: Ax = None, **kwargs
    ) -> Ax:
        """Plot permittivity of the :class:`Simulation` with all sources added for each port."""

        plot_sources = []
        for port_source in self.ports:
            source_0 = port_source.to_source(self._source_time, snap_center=None)
            plot_sources.append(source_0)
        sim_plot = self.simulation.copy(update=dict(sources=plot_sources))
        return sim_plot.plot_eps(x=x, y=y, z=z, ax=ax, **kwargs)

    @cached_property
    def sim_dict(self) -> Dict[str, Simulation]:
        """Generate all the :class:`Simulation` objects for the port parameter calculation."""

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
            port_source = port.to_source(self._source_time, snap_center=None)
            update_dict = dict(
                sources=[port_source],
                monitors=new_mnts,
                lumped_elements=lumped_resistors,
                grid_spec=grid_spec,
            )

            sim_copy = self.simulation.copy(update=update_dict)
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
                port_source = self.to_source(self._source_time, port=port, snap_center=center)
                update_dict = dict(
                    sources=[port_source],
                    monitors=new_mnts,
                    lumped_elements=lumped_resistors,
                    grid_spec=grid_spec,
                )

                sim_copy = self.simulation.copy(update=update_dict)
                task_name = self._task_name(port=port)
                sim_dict[task_name] = sim_copy

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
        """Post process `BatchData` to generate scattering matrix."""

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

        def port_voltage(port: LumpedPort, sim_data: SimulationData) -> xr.DataArray:
            """Helper to compute voltage across the port."""
            e_component = "xyz"[port.voltage_axis]
            field_data = sim_data[f"{port.name}_E{e_component}"]
            e_field = field_data.field_components[f"E{e_component}"]

            # Remove E field outside the port region
            e_field = e_field.sel(
                {
                    e_component: slice(
                        port.bounds[0][port.voltage_axis], port.bounds[1][port.voltage_axis]
                    )
                }
            )
            e_coords = [e_field.x, e_field.y, e_field.z]
            # Integration is along the original coordinates plus two additional endpoints corresponding to the precise bounds of the port
            e_coords_interp = np.array([port.bounds[0][port.voltage_axis]])
            e_coords_interp = np.concatenate((e_coords_interp, e_coords[port.voltage_axis].values))
            e_coords_interp = np.concatenate((e_coords_interp, [port.bounds[1][port.voltage_axis]]))
            e_coords_interp = {e_component: e_coords_interp}
            # Use extrapolation for the 2 additional endpoints
            e_field = e_field.interp(
                **e_coords_interp, method="linear", kwargs={"fill_value": "extrapolate"}
            )
            voltage = -e_field.integrate(coord=e_component).squeeze(drop=True)
            # Return data array of voltage with coordinates of frequency
            return voltage

        def port_current(port: LumpedPort, sim_data: SimulationData) -> xr.DataArray:
            """Helper to compute current flowing through the port."""

            # Diagram of contour integral, dashed line indicates location of sheet resistance
            # and electric field used for voltage computation. Voltage axis is out-of-page.
            #
            #                                    current_axis = ->
            #                                    injection_axis = ^
            #
            #                  |                   h2_field ->             |
            #    h_cap_minus ^  -------------------------------------------  h_cap_plus ^
            #                  |                   h1_field ->             |

            # Get h field tangent to resistive sheet
            h_component = "xyz"[port.current_axis]
            orth_component = "xyz"[port.injection_axis]
            field_data = sim_data[f"{port.name}_H{h_component}"]
            h_field = field_data.field_components[f"H{h_component}"]
            h_coords = [h_field.x, h_field.y, h_field.z]
            # h_cap represents the very short section (single cell) of the H contour that
            # is in the injection_axis direction. It is needed to fully enclose the sheet.
            h_cap_field = field_data.field_components[f"H{orth_component}"]
            h_cap_coords = [h_cap_field.x, h_cap_field.y, h_cap_field.z]

            # Use the coordinates of h_cap since it lies on the same grid that the
            # lumped resistor is snapped to
            orth_index = np.argmin(
                np.abs(h_cap_coords[port.injection_axis].values - port.center[port.injection_axis])
            )
            # Some sanity checks, tangent H field coordinates should be directly above
            # and below the coordinates of the resistive sheet
            assert orth_index > 0
            assert (
                h_cap_coords[port.injection_axis].values[orth_index]
                < h_coords[port.injection_axis].values[orth_index]
            )
            assert (
                h_coords[port.injection_axis].values[orth_index - 1]
                < h_cap_coords[port.injection_axis].values[orth_index]
            )

            # Extract field just below and just above sheet
            h1_field = h_field.isel({orth_component: orth_index - 1})
            h2_field = h_field.isel({orth_component: orth_index})
            h_field = h1_field - h2_field
            # Extract cap field which is coincident with sheet
            h_cap = h_cap_field.isel({orth_component: orth_index})

            # Need to make sure to use the nearest coordinate that is
            # at least greater than the port bounds
            hcap_minus = h_cap.sel({h_component: slice(-np.inf, port.bounds[0][port.current_axis])})
            hcap_plus = h_cap.sel({h_component: slice(port.bounds[1][port.current_axis], np.inf)})
            hcap_minus = hcap_minus.isel({h_component: -1})
            hcap_plus = hcap_plus.isel({h_component: 0})
            # Length of integration along the h_cap contour is a single cell width
            dcap = (
                h_coords[port.injection_axis].values[orth_index]
                - h_coords[port.injection_axis].values[orth_index - 1]
            )

            h_min_bound = hcap_minus.coords[h_component].values
            h_max_bound = hcap_plus.coords[h_component].values
            h_coords_interp = {
                h_component: np.linspace(
                    h_min_bound,
                    h_max_bound,
                    len(h_coords[port.current_axis] + 2),
                )
            }
            # Integration that corresponds to the tangent H field
            h_field = h_field.interp(**h_coords_interp)
            current = h_field.integrate(coord=h_component).squeeze(drop=True)

            # Integration that corresponds with the contribution to current from cap contours
            hcap_current = (
                ((hcap_plus - hcap_minus) * dcap).squeeze(drop=True).reset_coords(drop=True)
            )
            # Add the contribution from the hcap integral
            current = current + hcap_current
            # Make sure we compute current flowing from plus to minus voltage
            if port.current_axis != (port.voltage_axis + 1) % 3:
                current *= -1
            # Return data array of current with coordinates of frequency
            return current

        def port_ab(port: LumpedPort, sim_data: SimulationData):
            """Helper to compute the port incident and reflected power waves."""
            voltage = port_voltage(port, sim_data)
            current = port_current(port, sim_data)
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
        """Error if Simulation is not a 3D simulation"""

        if val.size.count(0.0) > 0:
            raise ValidationError(
                f"'{cls.__name__}' must be setup with a 3D simulation with all sizes greater than 0."
            )
        return val
