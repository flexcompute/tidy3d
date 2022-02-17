"""Tools for generating an S matrix automatically from tidy3d simulation and port definitions."""

from typing import List, Tuple, Dict

import pydantic as pd
import numpy as np

from ... import Simulation, Box, ModeSpec, ModeMonitor, ModeSource, GaussianPulse
from ...components.types import Direction
from ...constants import MICROMETER
from ...web import Batch
from ...web.container import DEFAULT_DATA_DIR

# fwidth of gaussian pulse in units of freq0
FWIDTH_FRAC = 1.0/10

class Port(Box):
    """Specifies a port in the scattering matrix."""

    direction: Direction = pd.Field(
        ...,
        title="Direction",
        description="Direction ('+' or '-') defining the port."
    )
    mode_spec: ModeSpec = pd.Field(
        ModeSpec(),
        title="Mode Specification",
        description="Specifies how the mode solver will solve for the modes of the port."
    )
    mode_indices: Tuple[pd.NonNegativeInt] = pd.Field(
        None,
        title="Mode Indices.",
        description="Indices into modes returned to mode solver to use in the port. "
        "If ``None``, all modes returned by the mode solver are used in the scattering matrix. "
        "Otherwise, the scattering matrix will include elements for each supplied mode index."
    )
    name: str = pd.Field(
        ...,
        title="Name",
        description="Unique name for the port.",
        min_length=1,
    )

    @pd.validator('mode_indices', always=True)
    def evaluate_mode_indices(cls, val, values):
        """Evaluates mode indices based on number of modes in mode spec."""
        if val is None:
            num_modes = values.get("mode_indices").num_modes
            val = tuple(range(num_modes))
        return val

# S[port1][port2] gives the scattering matrix between all mode indices between ports1 and ports2
SMatrixType = Dict[Port, Dict[Port, Dict[Direction, List[List[complex]]]]]

class ComponentModeler(pd.BaseModel):
    """Tool for modeling devices and computing scattering matrix elements."""

    simulation: Simulation = pd.Field(
        ...,
        title="Simulation",
        description="Simulation describing the device without any sources or monitors present.",
    )
    ports: List[Port] = pd.Field(
        [],
        title="Ports",
        description="Collection of ports describing the scattering matrix elements. "
        "For each port, one siulation will be run with a modal source.",
    )
    freq: float = pd.Field(
        ...,
        title="Frequency",
        description="Frequency at which to evaluate the scattering matrix.",
        units=MICROMETER
    )

    @pd.validator("simulation", always=True)
    def _empty_sim(cls, val):
        """Make sure simulation is empty."""
        if len(val.sources) > 0:
            raise SetupError(f"Simulation must not have sources.")
        if len(val.monitors) > 0:
            raise SetupError(f"Simulation must not have monitors.")
        return val

    # these will ghave to go into Component Modeller too

    def _to_monitor(self, port : Port) -> ModeMonitor:
        """Creates a mode monitor from a given port."""
        return ModeMonitor(
            center=port.center,
            size=port.size,
            freqs=[self.freq0],
            mode_spec=port.mode_spec,
            name=port.name,
        )

    def _to_sources(self, port : Port) -> List[ModeSource]:
        """Creates a list of mode sources from a given port."""
        return [ModeSource(
            center=port.center,
            size=port.size,
            source_time=GaussianPulse(freq0=self.freq0, fwidth=self.freq0*FWIDTH_FRAC),
            mode_spec=port.mode_spec,
            mode_index=mode_index,
            direction=port.direction,
            name=port.name,
        ) for mode_index in port.mode_indices]

    def _shift_port(self, port : Port) -> Port:
        """Generate a new port shifted by one grid cell in normal direction."""

        normal_index = port.size.index(0.0)
        dl = self.simulation.grid_size[normal_index]
        if not isinstance(dl, float):
            raise NotImplementedError("doesn't support nonuniform. How many grid cells to shift?")
        shift_value = dl if port.direction == '+' else -1 * dl

        center_shifted = list(port.center)
        center_shifted[normal_index] += shift
        port_shifted = port.copy(deep=True)
        port_shifted.center = center_shifted
        return port_shifted

    def solve(self, folder_name: str = "default", path_dir: str = DEFAULT_DATA_DIR) -> SMatrixType:
        """Solves for the scattering matrix of the system."""
        sims = {}
        for port_source in ports:
            for mode_source in self._to_sources(port_source):
                sim_copy = self.simulation.copy(deep=True)
                sim_copy.sources = [mode_source]
                sim_copy.monitors = []

                for port_monitor in ports:
                    if port_source == port_monitor:
                        port_monitor = self._shift_port(port_source)
                    mode_monitor = self._to_monitor(port_monitor)
                    sim_copy.monitors += mode_monitor

                task_name = f'smatrix_port{port_source.name}_mode{mode_source.mode_index}'
        batch = Batch(simulations=sims, folder_name=folder_name)
        batch.upload()
        batch.start()
        batch.monitor()

        s_matrix = []
        for task_name, sim_data in batch.items(path_dir=path_dir):
            for monitor in sim_data.simulation.monitors:
                mode_data = sim_data[monitor.name].amps.values
                s_matrix.append(mode_data)
        return np.array(s_matrx)












