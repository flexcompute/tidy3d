"""RF TerminalComponentModeler: copied from smatrix and namespaced under rf."""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pydantic.v1 as pd

from tidy3d.components.base import cached_property
from tidy3d.components.geometry.utils_2d import snap_coordinate_to_grid
from tidy3d.components.monitor import DirectivityMonitor
from tidy3d.components.simulation import Simulation
from tidy3d.components.source.time import GaussianPulse
from tidy3d.components.types import Ax
from tidy3d.components.viz import add_ax_if_none, equal_aspect
from tidy3d.constants import C_0, OHM
from tidy3d.exceptions import SetupError, Tidy3dKeyError, ValidationError
from tidy3d.log import log
from tidy3d.plugins.rf.data.data_array import PortDataArray
from tidy3d.plugins.rf.ports.base_lumped import AbstractLumpedPort
from tidy3d.plugins.rf.ports.coaxial_lumped import CoaxialLumpedPort
from tidy3d.plugins.rf.ports.rectangular_lumped import LumpedPort
from tidy3d.plugins.rf.ports.wave import WavePort
from tidy3d.plugins.smatrix.component_modelers.base import AbstractComponentModeler


class TerminalComponentModeler(AbstractComponentModeler):
    ports: tuple[Union[LumpedPort, CoaxialLumpedPort, WavePort], ...] = pd.Field(
        (),
        title="Terminal Ports",
        description="Collection of lumped and wave ports associated with the network.",
    )

    radiation_monitors: tuple[DirectivityMonitor, ...] = pd.Field(
        (),
        title="Radiation Monitors",
        description="Included in every simulation to record radiated fields.",
    )

    @pd.root_validator(pre=False)
    def _warn_rf_license(cls, values):
        log.warning(
            "ℹ️ ⚠️ RF simulations are subject to new license requirements in the future. You have instantiated at least one RF-specific component.",
            log_once=True,
        )
        return values

    @equal_aspect
    @add_ax_if_none
    def plot_sim(
        self,
        x: Optional[float] = None,
        y: Optional[float] = None,
        z: Optional[float] = None,
        ax: Ax = None,
        **kwargs,
    ) -> Ax:
        plot_sources = []
        for port_source in self.ports:
            source_0 = port_source.to_source(self._source_time)
            plot_sources.append(source_0)
        sim_plot = self.simulation.copy(update={"sources": plot_sources})
        return sim_plot.plot(x=x, y=y, z=z, ax=ax, **kwargs)

    @equal_aspect
    @add_ax_if_none
    def plot_sim_eps(
        self,
        x: Optional[float] = None,
        y: Optional[float] = None,
        z: Optional[float] = None,
        ax: Ax = None,
        **kwargs,
    ) -> Ax:
        plot_sources = []
        for port_source in self.ports:
            source_0 = port_source.to_source(self._source_time)
            plot_sources.append(source_0)
        sim_plot = self.simulation.copy(update={"sources": plot_sources})
        return sim_plot.plot_eps(x=x, y=y, z=z, ax=ax, **kwargs)

    @cached_property
    def sim_dict(self) -> dict[str, Simulation]:
        sim_dict = {}

        lumped_resistors = [port.to_load() for port in self._lumped_ports]

        grid_spec = self.simulation.grid_spec.copy(
            update={
                "wavelength": C_0 / np.max(self.freqs),
            }
        )

        sim_wo_source = self.simulation.updated_copy(
            grid_spec=grid_spec, lumped_elements=lumped_resistors
        )
        snap_centers = {}
        for port in self._lumped_ports:
            port_center_on_axis = port.center[port.injection_axis]
            new_port_center = snap_coordinate_to_grid(
                sim_wo_source.grid, port_center_on_axis, port.injection_axis
            )
            snap_centers[port.name] = new_port_center

        field_monitors = [
            mon
            for port in self.ports
            for mon in port.to_monitors(
                self.freqs, snap_center=snap_centers.get(port.name), grid=sim_wo_source.grid
            )
        ]

        new_mnts = list(self.simulation.monitors) + field_monitors

        if self.radiation_monitors is not None:
            new_mnts = new_mnts + list(self.radiation_monitors)

        new_lumped_elements = list(self.simulation.lumped_elements) + [
            port.to_load(snap_center=snap_centers[port.name]) for port in self._lumped_ports
        ]

        mesh_overrides = list(sim_wo_source.grid_spec.override_structures)
        for wave_port in self._wave_ports:
            if wave_port.num_grid_cells is not None:
                mesh_overrides.extend(wave_port.to_mesh_overrides())
        new_grid_spec = sim_wo_source.grid_spec.updated_copy(override_structures=mesh_overrides)

        update_dict = {
            "monitors": new_mnts,
            "lumped_elements": new_lumped_elements,
            "grid_spec": new_grid_spec,
        }

        sim_wo_source = sim_wo_source.copy(update=update_dict)

        for port in self._lumped_ports:
            port_source = port.to_source(
                self._source_time, snap_center=snap_centers[port.name], grid=sim_wo_source.grid
            )
            task_name = self.get_task_name(port=port)
            sim_dict[task_name] = sim_wo_source.updated_copy(sources=[port_source])

        for wave_port in self._wave_ports:
            mode_src_pos = wave_port.center[wave_port.injection_axis] + self._shift_value_signed(
                wave_port
            )
            port_source = wave_port.to_source(self._source_time, snap_center=mode_src_pos)
            update_dict = {"sources": [port_source]}
            task_name = self.get_task_name(port=wave_port)
            sim_dict[task_name] = sim_wo_source.copy(update=update_dict)

        for _, sim in sim_dict.items():
            TerminalComponentModeler._check_grid_size_at_ports(sim, self._lumped_ports)
            TerminalComponentModeler._check_grid_size_at_wave_ports(sim, self._wave_ports)

        return sim_dict

    @cached_property
    def _source_time(self):
        return GaussianPulse.from_frequency_range(
            fmin=min(self.freqs), fmax=max(self.freqs), remove_dc_component=self.remove_dc_component
        )

    @pd.validator("simulation")
    def _validate_3d_simulation(cls, val):
        if val.size.count(0.0) > 0:
            raise ValidationError(
                f"'{cls.__name__}' must be setup with a 3D simulation with all sizes greater than 0."
            )
        return val

    @pd.validator("radiation_monitors")
    def _validate_radiation_monitors(cls, val, values):
        freqs = set(values.get("freqs"))
        for rad_mon in val:
            mon_freqs = rad_mon.freqs
            is_subset = freqs.issuperset(mon_freqs)
            if not is_subset:
                raise ValidationError(
                    f"The frequencies in the radiation monitor '{rad_mon.name}' must be equal to or a subset of the frequencies in the '{cls.__name__}'."
                )
        return val

    @staticmethod
    def _check_grid_size_at_ports(
        simulation: Simulation, ports: list[Union[LumpedPort, CoaxialLumpedPort]]
    ):
        yee_grid = simulation.grid.yee
        for port in ports:
            port._check_grid_size(yee_grid)

    @staticmethod
    def _check_grid_size_at_wave_ports(simulation: Simulation, ports: list[WavePort]):
        for port in ports:
            disc_grid = simulation.discretize(port)
            check_axes = port.transverse_axes
            msg_header = f"'WavePort' '{port.name}' "
            for axis in check_axes:
                sim_size = simulation.size[axis]
                dim_cells = disc_grid.num_cells[axis]
                if sim_size > 0 and dim_cells <= 2:
                    small_dim = "xyz"[axis]
                    raise SetupError(
                        msg_header + f"is too small along the "
                        f"'{small_dim}' axis. Less than '3' grid cells were detected. "
                        "Please ensure that the port's 'num_grid_cells' is not 'None'. "
                        "You also may need to use an 'AutoGrid' or `QuasiUniformGrid` "
                        "for the simulation passed to the 'TerminalComponentModeler'."
                    )

    @cached_property
    def _lumped_ports(self) -> list[AbstractLumpedPort]:
        return [port for port in self.ports if isinstance(port, AbstractLumpedPort)]

    @cached_property
    def _wave_ports(self) -> list[WavePort]:
        return [port for port in self.ports if isinstance(port, WavePort)]

    @staticmethod
    def _set_port_data_array_attributes(data_array: PortDataArray) -> PortDataArray:
        data_array.name = "Z0"
        return data_array.assign_attrs(units=OHM, long_name="characteristic impedance")

    def get_radiation_monitor_by_name(self, monitor_name: str) -> DirectivityMonitor:
        for monitor in self.radiation_monitors:
            if monitor.name == monitor_name:
                return monitor
        raise Tidy3dKeyError(f"No radiation monitor named '{monitor_name}'.")


TerminalComponentModeler.update_forward_refs()
