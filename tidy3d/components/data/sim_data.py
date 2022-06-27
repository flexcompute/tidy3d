""" Simulation Level Data """
from typing import Dict, Optional

import pydantic as pd

from .base import Tidy3dData
from .monitor_data import MonitorDataType, AbstractFieldData
from ..base import cached_property
from ..simulation import Simulation
from ...log import log

# TODO: normalization
# TODO: final decay value
# TODO: centering of field data
# TODO: plotting (put some stuff in viz?)
# TODO: saving and loading from hdf5 group or json file
# TODO: docstring examples?
# TODO: at centers
# TODO: ModeSolverData?


class SimulationData(Tidy3dData):
    """Stores data from a collection of :class:`.Monitor` objects in a :class:`.Simulation`."""

    simulation: Simulation = pd.Field(
        ...,
        title="Simulation",
        description="Original :class:`.Simulation` associated with the data.",
    )

    monitor_data: Dict[str, MonitorDataType] = pd.Field(
        ...,
        title="Monitor Data",
        description="Mapping of monitor name to :class:`.MonitorData` instance.",
    )

    log: str = pd.Field(
        None,
        title="Solver Log",
        description="A string containing the log information from the simulation run.",
    )

    diverged: bool = pd.Field(
        False,
        title="Diverged",
        description="A boolean flag denoting whether the simulation run diverged.",
    )

    normalize_index: Optional[pd.NonNegativeInt] = pd.Field(
        0,
        title="Normalization index",
        description="Index of the source in the simulation.sources to use to normalize the data.",
    )

    @pd.validator("normalize_index", always=True)
    def _check_normalize_index(cls, val, values):
        """Check validity of normalize index in context of simulation.sources."""

        # not normalizing
        if val is None:
            return val

        assert val >= 0, "normalize_index can't be negative."

        num_sources = len(values.get("simulation").sources)

        # no sources, just skip normalization
        if num_sources == 0:
            log.warning(f"normalize_index={val} supplied but no sources found, not normalizing.")
            return None  # TODO: do we want this behavior though?

        assert val < num_sources, f"{num_sources} sources greater than normalize_index of {val}"

        return val

    def __getitem__(self, monitor_name: str) -> MonitorDataType:
        """Get a :class:`.MonitorData` by name. Apply symmetry and normalize if applicable."""

        monitor_data = self.monitor_data[monitor_name]
        monitor_data = self.apply_symmetry(monitor_data)
        monitor_data = self.normalize_monitor_data(monitor_data)

    def apply_symmetry(self, monitor_data: MonitorDataType) -> MonitorDataType:
        """Return copy of :class:`.MonitorData` object with symmetry values applied."""
        grid_expanded = self.simulation.discretize(monitor_data.monitor, extend=True)
        return monitor_data.apply_symmetry(
            symmetry=self.simulation.symmetry,
            symmetry_center=self.simulation.center,
            grid_expanded=grid_expanded,
        )

    def normalize_monitor_data(self, monitor_data: MonitorDataType) -> MonitorDataType:
        """Return copy of :class:`.MonitorData` object with data normalized to source."""

        # if no normalize index, just return the new copy right away.
        if self.normalize_index is None:
            return monitor_data.copy()

        # get source time information
        source = self.simulation.sources[self.normalize_index]
        source_time = source.source_time
        times = self.simulation.tmesh
        dt = self.simulation.dt
        user_defined_phase = np.exp(1j * source_time.phase)

        # get boundary information to determine whether to use complex fields
        boundaries = self.simulation.boundary_spec.to_list
        boundaries_1d = [boundary_1d for dim_boundary in boundaries for boundary_1d in dim_boundary]
        complex_fields = any(isinstance(boundary, BlochBoundary) for boundary in boundaries_1d)

        # plug in mornitor_data frequency domain information
        def source_spectrum_fn(freqs):
            """Source amplitude as function of frequency."""
            spectrum = source_time.spectrum(times, freqs, dt, complex_fields)

            # remove user defined phase from normalization so its effect is present in the result
            return spectrum * np.conj(user_defined_phase)

        return monitor_data.normalize(source_spectrum_fn)
