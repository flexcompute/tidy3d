"""Tool for generating an S matrix automatically from a Tidy3d simulation and modal port definitions."""

from __future__ import annotations

from typing import Optional

import autograd.numpy as np
import pydantic.v1 as pd

from tidy3d.components.base import cached_property
from tidy3d.components.data.sim_data import SimulationData
from tidy3d.components.monitor import ModeMonitor
from tidy3d.components.simulation import Simulation
from tidy3d.components.source.field import ModeSource
from tidy3d.components.source.time import GaussianPulse
from tidy3d.components.types import Ax
from tidy3d.components.viz import add_ax_if_none, equal_aspect
from tidy3d.plugins.smatrix.ports.modal import Port

from .base import FWIDTH_FRAC, AbstractComponentModeler

MatrixIndex = tuple[str, pd.NonNegativeInt]  # the 'i' in S_ij
Element = tuple[MatrixIndex, MatrixIndex]  # the 'ij' in S_ij


class ModalComponentModeler(AbstractComponentModeler[MatrixIndex, Element]):
    """
    Tool for modeling devices and computing scattering matrix elements.

    .. TODO missing basic example

    See Also
    --------

    **Notebooks**
        * `Computing the scattering matrix of a device <../../notebooks/SMatrix.html>`_
    """

    ports: tuple[Port, ...] = pd.Field(
        (),
        title="Ports",
        description="Collection of ports describing the scattering matrix elements. "
        "For each input mode, one simulation will be run with a modal source.",
    )

    @cached_property
    def sim_dict(self) -> dict[str, Simulation]:
        """Generate all the :class:`.Simulation` objects for the S matrix calculation."""

        sim_dict = {}
        mode_monitors = [self.to_monitor(port=port) for port in self.ports]

        for port_name, mode_index in self.matrix_indices_run_sim:
            port = self.get_port_by_name(port_name=port_name)

            port_source = self.shift_port(port=port)
            mode_source = self.to_source(port=port_source, mode_index=mode_index)

            new_mnts = list(self.simulation.monitors) + mode_monitors
            sim_copy = self.simulation.copy(update={"sources": [mode_source], "monitors": new_mnts})
            task_name = self.get_task_name(port=port, mode_index=mode_index)
            sim_dict[task_name] = sim_copy
        return sim_dict

    @cached_property
    def matrix_indices_monitor(self) -> tuple[MatrixIndex, ...]:
        """Tuple of all the possible matrix indices (port, mode_index) in the Component Modeler."""
        matrix_indices = []
        for port in self.ports:
            for mode_index in range(port.mode_spec.num_modes):
                matrix_indices.append((port.name, mode_index))
        return tuple(matrix_indices)

    @cached_property
    def port_names(self) -> tuple[list[str], list[str]]:
        """List of port names for inputs and outputs, respectively."""

        def get_port_names(matrix_elements: tuple[str, int]) -> list[str]:
            """Get the port names from a list of (port name, mode index)."""
            port_names = []
            for port_name, _ in matrix_elements:
                if port_name not in port_names:
                    port_names.append(port_name)
            return port_names

        port_names_in = get_port_names(self.matrix_indices_source)
        port_names_out = get_port_names(self.matrix_indices_monitor)

        return port_names_out, port_names_in

    def to_monitor(self, port: Port) -> ModeMonitor:
        """Creates a mode monitor from a given port."""
        return ModeMonitor(
            center=port.center,
            size=port.size,
            freqs=self.freqs,
            mode_spec=port.mode_spec,
            name=port.name,
        )

    def to_source(
        self, port: Port, mode_index: int, num_freqs: int = 1, **kwargs
    ) -> list[ModeSource]:
        """Creates a list of mode sources from a given port."""
        freq0 = np.mean(self.freqs)
        fdiff = max(self.freqs) - min(self.freqs)
        fwidth = max(fdiff, freq0 * FWIDTH_FRAC)
        return ModeSource(
            center=port.center,
            size=port.size,
            source_time=GaussianPulse(freq0=freq0, fwidth=fwidth),
            mode_spec=port.mode_spec,
            mode_index=mode_index,
            direction=port.direction,
            name=port.name,
            num_freqs=num_freqs,
            **kwargs,
        )

    def shift_port(self, port: Port) -> Port:
        """Generate a new port shifted by the shift amount in normal direction."""

        shift_value = self._shift_value_signed(port=port)
        center_shifted = list(port.center)
        center_shifted[port.size.index(0.0)] += shift_value
        port_shifted = port.copy(update={"center": center_shifted})
        return port_shifted

    @equal_aspect
    @add_ax_if_none
    def plot_sim(
        self,
        x: Optional[float] = None,
        y: Optional[float] = None,
        z: Optional[float] = None,
        ax: Ax = None,
    ) -> Ax:
        """Plot a :class:`.Simulation` with all sources added for each port, for troubleshooting."""

        plot_sources = []
        for port_source in self.ports:
            mode_source_0 = self.to_source(port=port_source, mode_index=0)
            plot_sources.append(mode_source_0)
        sim_plot = self.simulation.copy(update={"sources": plot_sources})
        return sim_plot.plot(x=x, y=y, z=z, ax=ax)

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
        """Plot permittivity of the :class:`.Simulation` with all sources added for each port."""

        plot_sources = []
        for port_source in self.ports:
            mode_source_0 = self.to_source(port=port_source, mode_index=0)
            plot_sources.append(mode_source_0)
        sim_plot = self.simulation.copy(update={"sources": plot_sources})
        return sim_plot.plot_eps(x=x, y=y, z=z, ax=ax, **kwargs)

    def _normalization_factor(self, port_source: Port, sim_data: SimulationData) -> complex:
        """Compute the normalization amplitude based on the measured input mode amplitude."""

        port_monitor_data = sim_data[port_source.name]
        mode_index = sim_data.simulation.sources[0].mode_index

        normalize_amps = port_monitor_data.amps.sel(
            f=np.array(self.freqs),
            direction=port_source.direction,
            mode_index=mode_index,
        )

        return normalize_amps.values

    @cached_property
    def max_mode_index(self) -> tuple[int, int]:
        """maximum mode indices for the smatrix dataset for the in and out ports, respectively."""

        def get_max_mode_indices(matrix_elements: tuple[str, int]) -> int:
            """Get the maximum mode index for a list of (port name, mode index)."""
            return max(mode_index for _, mode_index in matrix_elements)

        max_mode_index_out = get_max_mode_indices(self.matrix_indices_monitor)
        max_mode_index_in = get_max_mode_indices(self.matrix_indices_source)

        return max_mode_index_out, max_mode_index_in
