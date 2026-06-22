"""Tool for generating an S matrix automatically from a Tidy3d simulation and modal port definitions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import autograd.numpy as np
from pydantic import Field

from tidy3d.components.base import cached_property
from tidy3d.components.index import SimulationMap
from tidy3d.components.run_time_spec import RunTimeSpec
from tidy3d.components.source.time import GaussianPulse
from tidy3d.components.types import Complex
from tidy3d.components.types.base import discriminated_union
from tidy3d.components.viz import add_ax_if_none, equal_aspect
from tidy3d.constants import GLANCING_CUTOFF
from tidy3d.exceptions import SetupError
from tidy3d.plugins.smatrix.ports.modal import (
    AbstractGaussianPort,
    GaussianPort,
    ModalPortType,
)
from tidy3d.plugins.smatrix.types import Element, MatrixIndex

from .base import FWIDTH_FRAC, AbstractComponentModeler

if TYPE_CHECKING:
    from tidy3d.components.data.sim_data import SimulationData
    from tidy3d.components.monitor import AbstractOverlapMonitor
    from tidy3d.components.simulation import Simulation
    from tidy3d.components.source.field import DirectionalSource
    from tidy3d.components.types import Ax
    from tidy3d.plugins.smatrix.ports.modal import Port


class ModalComponentModeler(AbstractComponentModeler):
    """A tool for modeling devices and computing scattering matrix elements.

    Notes
    -----
        This class orchestrates the process of running multiple simulations to
        derive the scattering matrix (S-matrix) of a component. It uses modal
        or Gaussian sources and monitors defined by a set of ports.

    See Also
    --------
    **Notebooks**
        * `Computing the scattering matrix of a device <../../notebooks/SMatrix.html>`_
    """

    ports: tuple[discriminated_union(ModalPortType), ...] = Field(
        (),
        title="Ports",
        description="Collection of ports describing the scattering matrix elements. "
        "For each input mode, one simulation will be run with a modal source.",
    )

    run_only: tuple[MatrixIndex, ...] | None = Field(
        None,
        title="Run Only",
        description="Set of matrix indices that define the simulations to run. "
        "If ``None``, simulations will be run for all indices in the scattering matrix. "
        "If a tuple is given, simulations will be run only for the given matrix indices.",
    )

    element_mappings: tuple[tuple[Element, Element, Complex], ...] = Field(
        (),
        title="Element Mappings",
        description="Tuple of S matrix element mappings, each described by a tuple of "
        "(input_element, output_element, coefficient), where the coefficient is the "
        "element_mapping coefficient describing the relationship between the input and output "
        "matrix element. If all elements of a given column of the scattering matrix are defined "
        "by ``element_mappings``, the simulation corresponding to this column is skipped automatically.",
    )

    @property
    def base_sim(self) -> Simulation:
        """The base simulation."""
        base = self.simulation
        # Resolve a 'RunTimeSpec' run_time using the modeler's known excitation pulse, so the
        # base simulation carries a concrete, source-independent run_time. base_sim has no port
        # sources, so a 'RunTimeSpec' would otherwise resolve far too short.
        if isinstance(base.run_time, RunTimeSpec):
            base = base.updated_copy(run_time=base._resolve_run_time([self._source_time]))
        return base

    @cached_property
    def _source_time(self) -> GaussianPulse:
        """Time-domain pulse used for the port excitations (also used to resolve run_time)."""
        if self.custom_source_time is not None:
            return self.custom_source_time
        freq0 = 0.5 * (max(self.freqs) + min(self.freqs))
        fwidth = max(max(self.freqs) - min(self.freqs), freq0 * FWIDTH_FRAC)
        return GaussianPulse(
            freq0=freq0, fwidth=fwidth, remove_dc_component=self.remove_dc_component
        )

    @cached_property
    def sim_dict(self) -> SimulationMap:
        """Generates all :class:`.Simulation` objects for the S-matrix calculation.

        Returns
        -------
        Dict[str, Simulation]
            A dictionary where keys are task names and values are the
            corresponding :class:`.Simulation` objects. Each simulation is
            configured to excite a specific mode at a specific port and
            includes all necessary monitors.
        """

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
        return SimulationMap(keys=tuple(sim_dict.keys()), values=tuple(sim_dict.values()))

    def _construct_matrix_indices_monitor(
        self, ports: tuple[ModalPortType, ...]
    ) -> tuple[MatrixIndex, ...]:
        """Construct matrix indices for monitoring from modal ports.

        Parameters
        ----------
        ports : tuple[Port, ...]
            Tuple of Port objects.

        Returns
        -------
        tuple[MatrixIndex, ...]
            Tuple of (port_name, mode_index) pairs.
        """
        matrix_indices = []
        for port in ports:
            for mode_index in range(port.num_modes):
                matrix_indices.append((port.name, mode_index))
        return tuple(matrix_indices)

    @cached_property
    def matrix_indices_monitor(self) -> tuple[MatrixIndex, ...]:
        """Returns a tuple of all possible matrix indices for monitoring.

        Each matrix index is a tuple of (port_name, mode_index).

        Returns
        -------
        Tuple[MatrixIndex, ...]
            A tuple of all possible matrix indices for the monitoring ports.
        """
        return self._construct_matrix_indices_monitor(self.ports)

    @cached_property
    def matrix_indices_source(self) -> tuple[MatrixIndex, ...]:
        """Tuple of all the source matrix indices, which may be less than the total number of
        ports."""
        return super().matrix_indices_source

    @cached_property
    def matrix_indices_run_sim(self) -> tuple[MatrixIndex, ...]:
        """Tuple of all the matrix indices that will be used to run simulations."""
        return super().matrix_indices_run_sim

    @cached_property
    def port_names(self) -> tuple[list[str], list[str]]:
        """Returns lists of port names for inputs and outputs.

        Returns
        -------
        Tuple[List[str], List[str]]
            A tuple containing two lists: the first with the names of the
            output ports, and the second with the names of the input ports.
        """

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

    def to_monitor(self, port: ModalPortType) -> AbstractOverlapMonitor:
        """Creates an overlap monitor from a given port (modal or gaussian)."""
        return port.to_monitor(freqs=self.freqs)

    def to_source(
        self, port: ModalPortType, mode_index: int, num_freqs: int = 1, **kwargs: Any
    ) -> DirectionalSource:
        """Creates a source from a given port (modal or gaussian)."""
        freqs = self.freqs
        freq0 = 0.5 * (max(freqs) + min(freqs))
        fdiff = max(freqs) - min(freqs)
        fwidth = max(fdiff, freq0 * FWIDTH_FRAC)
        source_time = self._source_time
        return port.to_source(
            freq0=freq0,
            fwidth=fwidth,
            mode_index=mode_index,
            num_freqs=num_freqs,
            source_time=source_time,
            **kwargs,
        )

    def shift_port(self, port: ModalPortType) -> ModalPortType:
        """Generates a new port shifted slightly in the normal direction.

        This is to ensure that the source is placed just inside the
        simulation domain, away from the PML.

        Parameters
        ----------
        port : Port
            The port to shift.

        Returns
        -------
        Port
            A new :class:`.Port` object with its center shifted.
        """

        shift_value = self._shift_value_signed(port=port, simulation=self.simulation)
        center_shifted = list(port.center)
        normal_dim, plane_dims = port.pop_axis([0, 1, 2], port.size.index(0.0))
        center_shifted[normal_dim] += shift_value
        update = {}
        if isinstance(port, AbstractGaussianPort):
            theta = port.angle_theta
            phi = port.angle_phi
            cos_theta = np.cos(theta)
            if np.abs(cos_theta) < GLANCING_CUTOFF:
                raise SetupError(
                    "Cannot shift Gaussian port at glancing incidence. "
                    "Adjust angle_theta or use a different injection axis."
                )
            tan_theta = np.sin(theta) / cos_theta
            center_shifted[plane_dims[0]] += shift_value * tan_theta * np.cos(phi)
            center_shifted[plane_dims[1]] += shift_value * tan_theta * np.sin(phi)
            if isinstance(port, GaussianPort):
                waist_distance = port.waist_distance + shift_value / cos_theta
                update["waist_distance"] = waist_distance
            else:
                waist_distances = list(port.waist_distances)
                waist_distances[0] = waist_distances[0] + shift_value / cos_theta
                waist_distances[1] = waist_distances[1] + shift_value / cos_theta
                update["waist_distances"] = waist_distances
        port_shifted = port.updated_copy(center=center_shifted, **update)
        return port_shifted

    @equal_aspect
    @add_ax_if_none
    def plot_sim(
        self,
        x: float | None = None,
        y: float | None = None,
        z: float | None = None,
        ax: Ax = None,
    ) -> Ax:
        """Plots the simulation with all sources added for troubleshooting.

        This method creates a temporary simulation with all mode sources
        activated to help visualize the setup.

        Parameters
        ----------
        x : float, optional
            The x-coordinate of the cross-section, by default None.
        y : float, optional
            The y-coordinate of the cross-section, by default None.
        z : float, optional
            The z-coordinate of the cross-section, by default None.
        ax : Ax, optional
            The matplotlib axes to plot on, by default None.

        Returns
        -------
        Ax
            The matplotlib axes with the plot.
        """

        plot_sources = []
        for port_source in self.ports:
            # for plotting, use mode_index=0 (gaussian ignores it)
            src0 = self.to_source(port=port_source, mode_index=0)
            plot_sources.append(src0)
        sim_plot = self.simulation.copy(update={"sources": tuple(plot_sources)})
        return sim_plot.plot(x=x, y=y, z=z, ax=ax)

    @equal_aspect
    @add_ax_if_none
    def plot_sim_eps(
        self,
        x: float | None = None,
        y: float | None = None,
        z: float | None = None,
        ax: Ax = None,
        **kwargs: Any,
    ) -> Ax:
        """Plots the permittivity of the simulation with all sources.

        This method is useful for visualizing the device geometry along
        with the placement of the sources.

        Parameters
        ----------
        x : float, optional
            The x-coordinate of the cross-section, by default None.
        y : float, optional
            The y-coordinate of the cross-section, by default None.
        z : float, optional
            The z-coordinate of the cross-section, by default None.
        ax : Ax, optional
            The matplotlib axes to plot on, by default None.
        **kwargs
            Additional keyword arguments passed to the plotter.

        Returns
        -------
        Ax
            The matplotlib axes with the plot.
        """

        plot_sources = []
        for port_source in self.ports:
            src0 = self.to_source(port=port_source, mode_index=0)
            plot_sources.append(src0)
        sim_plot = self.simulation.copy(update={"sources": tuple(plot_sources)})
        return sim_plot.plot_eps(x=x, y=y, z=z, ax=ax, **kwargs)

    def _normalization_factor(self, port_source: Port, sim_data: SimulationData) -> complex:
        """Computes the normalization amplitude for the input mode.

        This is used to normalize the S-matrix elements.

        Parameters
        ----------
        port_source : Port
            The port that was excited.
        sim_data : SimulationData
            The data from the simulation run.

        Returns
        -------
        complex
            The complex amplitude of the input mode.
        """

        port_monitor_data = sim_data[port_source.name]
        # some sources (GaussianBeam) don't have 'mode_index'; default to 0
        src = sim_data.simulation.sources[0]
        mode_index = getattr(src, "mode_index", 0)
        normalize_amps = port_monitor_data.amps.sel(
            f=np.array(self.freqs),
            direction=port_source.direction,
            mode_index=mode_index,
        )

        return normalize_amps.values

    @cached_property
    def max_mode_index(self) -> tuple[int, int]:
        """Returns the maximum mode indices for the in and out ports.

        Returns
        -------
        Tuple[int, int]
            A tuple containing the maximum mode index for the output ports
            and the maximum mode index for the input ports.
        """

        def get_max_mode_indices(matrix_elements: tuple[str, int]) -> int:
            """Get the maximum mode index for a list of (port name, mode index)."""
            return max(mode_index for _, mode_index in matrix_elements)

        max_mode_index_out = get_max_mode_indices(self.matrix_indices_monitor)
        max_mode_index_in = get_max_mode_indices(self.matrix_indices_source)

        return max_mode_index_out, max_mode_index_in

    def task_name_from_index(self, matrix_index: MatrixIndex) -> str:
        """Compute task name for a given (port_name, mode_index) without constructing simulations."""
        port_name, mode_index = matrix_index
        port = self.get_port_by_name(port_name=port_name)
        return self.get_task_name(port=port, mode_index=mode_index)
