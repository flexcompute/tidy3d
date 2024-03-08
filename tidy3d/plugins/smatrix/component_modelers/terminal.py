"""Tool for generating an S matrix automatically from a Tidy3d simulation and lumped port definitions."""

from __future__ import annotations

from typing import Tuple, Dict

import pydantic.v1 as pd
import numpy as np
import xarray as xr

from ....constants import C_0, fp_eps
from ....exceptions import SetupError
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

from ...microwave import VoltageIntegralAxisAligned, CurrentIntegralAxisAligned


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
                port_source = port.to_source(self._source_time, snap_center=center)
                update_dict = dict(
                    sources=[port_source],
                    monitors=new_mnts,
                    lumped_elements=lumped_resistors,
                    grid_spec=grid_spec,
                )

                sim_copy = self.simulation.copy(update=update_dict)
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

        def select_within_bounds(coords: np.array, min, max):
            """Helper to return indices of coordinates within min and max bounds,"""
            """including a tolerance. xarray does not have this functionality yet. """
            min_idx = np.searchsorted(coords, min, "left")
            # If a coordinate is close enough, it is considered included
            if min_idx > 0 and np.isclose(coords[min_idx - 1], min, rtol=fp_eps, atol=fp_eps):
                min_idx -= 1
            max_idx = np.searchsorted(coords, max, "left")
            if max_idx < len(coords) and np.isclose(coords[max_idx], max, rtol=fp_eps, atol=fp_eps):
                max_idx += 1

            return (min_idx, max_idx - 1)

        def port_voltage(port: LumpedPort, sim_data: SimulationData) -> xr.DataArray:
            """Helper to compute voltage across the port."""
            e_component = "xyz"[port.voltage_axis]
            field_data = sim_data[f"{port.name}_E{e_component}"]

            size = list(port.size)
            size[port.current_axis] = 0
            voltage_integral = VoltageIntegralAxisAligned(
                center=port.center,
                size=size,
                extrapolate_to_endpoints=True,
                snap_path_to_grid=True,
                sign="+",
            )
            voltage = voltage_integral.compute_voltage(field_data)
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
            inject_component = "xyz"[port.injection_axis]
            monitor_data = sim_data[f"{port.name}_H{h_component}"]
            field_data = monitor_data.field_components
            h_field = field_data[f"H{h_component}"]
            # Coordinates as numpy array for h_field along curren and injection axis
            h_coords_along_current = h_field.coords[h_component].values
            h_coords_along_injection = h_field.coords[inject_component].values
            # h_cap represents the very short section (single cell) of the H contour that
            # is in the injection_axis direction. It is needed to fully enclose the sheet.
            h_cap_field = field_data[f"H{inject_component}"]
            # Coordinates of h_cap field as numpy arrays
            h_cap_coords_along_current = h_cap_field.coords[h_component].values
            h_cap_coords_along_injection = h_cap_field.coords[inject_component].values

            # Use the coordinates of h_cap since it lies on the same grid that the
            # lumped resistor is snapped to
            orth_index = np.argmin(
                np.abs(h_cap_coords_along_injection - port.center[port.injection_axis])
            )
            inject_center = h_cap_coords_along_injection[orth_index]
            # Some sanity checks, tangent H field coordinates should be directly above
            # and below the coordinates of the resistive sheet
            assert orth_index > 0
            assert inject_center < h_coords_along_injection[orth_index]
            assert h_coords_along_injection[orth_index - 1] < inject_center
            # Distance between the h1_field and h2_field, a single cell size
            dcap = h_coords_along_injection[orth_index] - h_coords_along_injection[orth_index - 1]

            # Next find the size in the current_axis direction
            # Find exact bounds of port taking into consideration the Yee grid
            # Select bounds carefully and allow for h_cap very close to the port bounds
            port_min = port.bounds[0][port.current_axis]
            port_max = port.bounds[1][port.current_axis]

            (idx_min, idx_max) = select_within_bounds(h_coords_along_current, port_min, port_max)
            # Use these indices to select the exact positions of the h_cap field
            h_min_bound = h_cap_coords_along_current[idx_min - 1]
            h_max_bound = h_cap_coords_along_current[idx_max]

            # Setup axis aligned contour integral, which is defined by a plane
            # The path integral is snapped to the grid, so center and size will
            # be slightly modified when compared to the original port.
            center = list(port.center)
            center[port.injection_axis] = inject_center
            center[port.current_axis] = (h_max_bound + h_min_bound) / 2
            size = [0, 0, 0]
            size[port.current_axis] = h_max_bound - h_min_bound
            size[port.injection_axis] = dcap

            # H field is continuous at integral bounds, so extrapolation is turned off
            I_integral = CurrentIntegralAxisAligned(
                center=center,
                size=size,
                sign="+",
                extrapolate_to_endpoints=False,
                snap_contour_to_grid=True,
            )
            return I_integral.compute_current(monitor_data)

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

    @staticmethod
    def _check_grid_size_at_ports(simulation: Simulation, ports: list[LumpedPort]):
        """Raises :class:`SetupError` if the grid is too coarse at port locations"""
        yee_grid = simulation.grid.yee
        for port in ports:
            e_component = "xyz"[port.voltage_axis]
            e_yee_grid = yee_grid.grid_dict[f"E{e_component}"]
            coords = e_yee_grid.to_dict[e_component]
            min_bound = port.bounds[0][port.voltage_axis]
            max_bound = port.bounds[1][port.voltage_axis]
            coords_within_port = np.any(np.logical_and(coords > min_bound, coords < max_bound))
            if not coords_within_port:
                raise SetupError(
                    f"Grid is too coarse along '{e_component}' direction for the lumped port "
                    f"at location {port.center}. Either set the port's 'num_grid_cells' to "
                    f"a nonzero integer or modify the 'GridSpec'. "
                )
