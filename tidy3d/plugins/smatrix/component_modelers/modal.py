"""Tool for generating an S matrix automatically from a Tidy3d simulation and modal port definitions."""

# TODO: The names "ComponentModeler" and "Port" should be changed to "ModalComponentModeler" and
# "ModalPort" to explicitly differentiate these from "TerminalComponentModeler" and "LumpedPort".
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pydantic.v1 as pd

from ....components.base import cached_property
from ....components.data.sim_data import SimulationData
from ....components.monitor import ModeMonitor
from ....components.simulation import Simulation
from ....components.source.field import ModeSource
from ....components.source.time import GaussianPulse
from ....components.types import Ax, Complex
from ....components.viz import add_ax_if_none, equal_aspect
from ....exceptions import SetupError
from ....web.api.container import BatchData
from ..ports.modal import ModalPortDataArray, Port
from .base import FWIDTH_FRAC, AbstractComponentModeler

MatrixIndex = Tuple[str, pd.NonNegativeInt]  # the 'i' in S_ij
Element = Tuple[MatrixIndex, MatrixIndex]  # the 'ij' in S_ij


class ComponentModeler(AbstractComponentModeler):
    """
    Tool for modeling devices and computing scattering matrix elements.

    .. TODO missing basic example

    See Also
    --------

    **Notebooks**
        * `Computing the scattering matrix of a device <../../notebooks/SMatrix.html>`_
    """

    ports: Tuple[Port, ...] = pd.Field(
        (),
        title="Ports",
        description="Collection of ports describing the scattering matrix elements. "
        "For each input mode, one simulation will be run with a modal source.",
    )

    element_mappings: Tuple[Tuple[Element, Element, Complex], ...] = pd.Field(
        (),
        title="Element Mappings",
        description="Mapping between elements of the scattering matrix, "
        "as specified by pairs of ``(port name, mode index)`` matrix indices, where the "
        "first element of the pair is the output and the second element of the pair is the input."
        "Each item of ``element_mappings`` is a tuple of ``(element1, element2, c)``, where "
        "the scattering matrix ``Smatrix[element2]`` is set equal to ``c * Smatrix[element1]``."
        "If all elements of a given column of the scattering matrix are defined by "
        " ``element_mappings``, the simulation corresponding to this column "
        "is skipped automatically.",
    )

    run_only: Optional[Tuple[MatrixIndex, ...]] = pd.Field(
        None,
        title="Run Only",
        description="If given, a tuple of matrix indices, specified by (:class:`.Port`, ``int``),"
        " to run only, excluding the other rows from the scattering matrix. "
        "If this option is used, "
        "the data corresponding to other inputs will be missing in the resulting matrix.",
    )
    """Finally, to exclude some rows of the scattering matrix, one can supply a ``run_only`` parameter to the
    :class:`ComponentModeler`. ``run_only`` contains the scattering matrix indices that the user wants to run as a
    source. If any indices are excluded, they will not be run."""

    verbose: bool = pd.Field(
        False,
        title="Verbosity",
        description="Whether the :class:`.ComponentModeler` should print status and progressbars.",
    )

    callback_url: str = pd.Field(
        None,
        title="Callback URL",
        description="Http PUT url to receive simulation finish event. "
        "The body content is a json file with fields "
        "``{'id', 'status', 'name', 'workUnit', 'solverVersion'}``.",
    )

    @pd.validator("simulation", always=True)
    def _sim_has_no_sources(cls, val):
        """Make sure simulation has no sources as they interfere with tool."""
        if len(val.sources) > 0:
            raise SetupError("'ComponentModeler.simulation' must not have any sources.")
        return val

    @cached_property
    def sim_dict(self) -> Dict[str, Simulation]:
        """Generate all the :class:`.Simulation` objects for the S matrix calculation."""

        sim_dict = {}
        mode_monitors = [self.to_monitor(port=port) for port in self.ports]

        for port_name, mode_index in self.matrix_indices_run_sim:
            port = self.get_port_by_name(port_name=port_name)

            port_source = self.shift_port(port=port)
            mode_source = self.to_source(port=port_source, mode_index=mode_index)

            new_mnts = list(self.simulation.monitors) + mode_monitors
            sim_copy = self.simulation.copy(update=dict(sources=[mode_source], monitors=new_mnts))
            task_name = self._task_name(port=port, mode_index=mode_index)
            sim_dict[task_name] = sim_copy
        return sim_dict

    @cached_property
    def matrix_indices_monitor(self) -> Tuple[MatrixIndex, ...]:
        """Tuple of all the possible matrix indices (port, mode_index) in the Component Modeler."""
        matrix_indices = []
        for port in self.ports:
            for mode_index in range(port.mode_spec.num_modes):
                matrix_indices.append((port.name, mode_index))
        return tuple(matrix_indices)

    @cached_property
    def matrix_indices_source(self) -> Tuple[MatrixIndex, ...]:
        """Tuple of all the source matrix indices (port, mode_index) in the Component Modeler."""
        if self.run_only is not None:
            return self.run_only
        return self.matrix_indices_monitor

    @cached_property
    def matrix_indices_run_sim(self) -> Tuple[MatrixIndex, ...]:
        """Tuple of all the source matrix indices (port, mode_index) in the Component Modeler."""

        if self.element_mappings is None or self.element_mappings == {}:
            return self.matrix_indices_source

        # all the (i, j) pairs in `S_ij` that are tagged as covered by `element_mappings`
        elements_determined_by_map = [element_out for (_, element_out, _) in self.element_mappings]

        # loop through rows of the full s matrix and record rows that still need running.
        source_indices_needed = []
        for col_index in self.matrix_indices_source:
            # loop through columns and keep track of whether each element is covered by mapping.
            matrix_elements_covered = []
            for row_index in self.matrix_indices_monitor:
                element = (row_index, col_index)
                element_covered_by_map = element in elements_determined_by_map
                matrix_elements_covered.append(element_covered_by_map)

            # if any matrix elements in row still not covered by map, a source is needed for row.
            if not all(matrix_elements_covered):
                source_indices_needed.append(col_index)

        return source_indices_needed

    @cached_property
    def port_names(self) -> Tuple[List[str], List[str]]:
        """List of port names for inputs and outputs, respectively."""

        def get_port_names(matrix_elements: Tuple[str, int]) -> List[str]:
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
    ) -> List[ModeSource]:
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
        port_shifted = port.copy(update=dict(center=center_shifted))
        return port_shifted

    @equal_aspect
    @add_ax_if_none
    def plot_sim(self, x: float = None, y: float = None, z: float = None, ax: Ax = None) -> Ax:
        """Plot a :class:`.Simulation` with all sources added for each port, for troubleshooting."""

        plot_sources = []
        for port_source in self.ports:
            mode_source_0 = self.to_source(port=port_source, mode_index=0)
            plot_sources.append(mode_source_0)
        sim_plot = self.simulation.copy(update=dict(sources=plot_sources))
        return sim_plot.plot(x=x, y=y, z=z, ax=ax)

    @equal_aspect
    @add_ax_if_none
    def plot_sim_eps(
        self, x: float = None, y: float = None, z: float = None, ax: Ax = None, **kwargs
    ) -> Ax:
        """Plot permittivity of the :class:`.Simulation` with all sources added for each port."""

        plot_sources = []
        for port_source in self.ports:
            mode_source_0 = self.to_source(port=port_source, mode_index=0)
            plot_sources.append(mode_source_0)
        sim_plot = self.simulation.copy(update=dict(sources=plot_sources))
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
    def max_mode_index(self) -> Tuple[int, int]:
        """maximum mode indices for the smatrix dataset for the in and out ports, respectively."""

        def get_max_mode_indices(matrix_elements: Tuple[str, int]) -> int:
            """Get the maximum mode index for a list of (port name, mode index)."""
            return max(mode_index for _, mode_index in matrix_elements)

        max_mode_index_out = get_max_mode_indices(self.matrix_indices_monitor)
        max_mode_index_in = get_max_mode_indices(self.matrix_indices_source)

        return max_mode_index_out, max_mode_index_in

    def _construct_smatrix(self) -> ModalPortDataArray:
        """Post process :class:`.BatchData` to generate scattering matrix."""
        return self._internal_construct_smatrix(batch_data=self.batch_data)

    def _internal_construct_smatrix(self, batch_data: BatchData) -> ModalPortDataArray:
        """Post process :class:`.BatchData` to generate scattering matrix, for internal use only."""

        max_mode_index_out, max_mode_index_in = self.max_mode_index
        num_modes_out = max_mode_index_out + 1
        num_modes_in = max_mode_index_in + 1
        port_names_out, port_names_in = self.port_names

        values = np.zeros(
            (len(port_names_out), len(port_names_in), num_modes_out, num_modes_in, len(self.freqs)),
            dtype=complex,
        )
        coords = dict(
            port_out=port_names_out,
            port_in=port_names_in,
            mode_index_out=range(num_modes_out),
            mode_index_in=range(num_modes_in),
            f=np.array(self.freqs),
        )
        s_matrix = ModalPortDataArray(values, coords=coords)

        # loop through source ports
        for col_index in self.matrix_indices_run_sim:
            port_name_in, mode_index_in = col_index
            port_in = self.get_port_by_name(port_name=port_name_in)

            sim_data = batch_data[self._task_name(port=port_in, mode_index=mode_index_in)]

            for row_index in self.matrix_indices_monitor:
                port_name_out, mode_index_out = row_index
                port_out = self.get_port_by_name(port_name=port_name_out)

                # directly compute the element
                mode_amps_data = sim_data[port_out.name].copy().amps
                dir_out = "-" if port_out.direction == "+" else "+"
                amp = mode_amps_data.sel(
                    f=coords["f"], direction=dir_out, mode_index=mode_index_out
                )
                source_norm = self._normalization_factor(port_in, sim_data)
                s_matrix_elements = np.array(amp.data) / np.array(source_norm)
                s_matrix.loc[
                    dict(
                        port_in=port_name_in,
                        mode_index_in=mode_index_in,
                        port_out=port_name_out,
                        mode_index_out=mode_index_out,
                    )
                ] = s_matrix_elements

        # element can be determined by user-defined mapping
        for (row_in, col_in), (row_out, col_out), mult_by in self.element_mappings:
            port_out_from, mode_index_out_from = row_in
            port_in_from, mode_index_in_from = col_in
            coords_from = dict(
                port_in=port_in_from,
                mode_index_in=mode_index_in_from,
                port_out=port_out_from,
                mode_index_out=mode_index_out_from,
            )

            port_out_to, mode_index_out_to = row_out
            port_in_to, mode_index_in_to = col_out
            coords_to = dict(
                port_in=port_in_to,
                mode_index_in=mode_index_in_to,
                port_out=port_out_to,
                mode_index_out=mode_index_out_to,
            )
            s_matrix.loc[coords_to] = mult_by * s_matrix.loc[coords_from].values

        return s_matrix
