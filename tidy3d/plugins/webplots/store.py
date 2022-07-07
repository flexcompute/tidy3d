""" Data store to hold hdf5 data. """
from abc import ABC, abstractmethod

from ...components.data import SimulationData
from .simulation import SimulationPlotly
from .data import DataPlotly

FrontEndStore = {}


class Store(ABC):
    """Abstract class for data store."""

    @abstractmethod
    def get_simulation_data(self, front_end_store: FrontEndStore) -> SimulationData:
        """Get simulation data from store."""

    def get_simulation_plotly(self, store: []) -> SimulationPlotly:
        """Get simulation plotly from store."""
        data = self.get_simulation_data(store)
        return SimulationPlotly(simulation=data.simulation)

    def get_data_plotly_by_name(self, store: FrontEndStore, name: str) -> DataPlotly:
        """Get data plotly from store by name."""
        sim = self.get_simulation_data(store)
        return next(
            (
                DataPlotly.from_monitor_data(monitor_data=monitor_data, monitor_name=monitor_name)
                for monitor_name, monitor_data in sim.monitor_data.items()
                if monitor_name == name
            ),
            None,
        )


class LocalStore(Store):
    """Local data store."""

    def __init__(self, fname: str):
        """Initialize local store."""
        self.sim_data = SimulationData.from_file(fname)

    def get_simulation_data(self, front_end_store: FrontEndStore):
        """Get simulation data from store."""
        return self.sim_data


class EmptyStore(Store):
    """Empty store."""

    def get_simulation_data(self, front_end_store: FrontEndStore):
        pass


_DEFAULT_STORE = EmptyStore()


def set_store(store: Store):
    """Set default store."""
    global _DEFAULT_STORE  # pylint: disable=global-statement
    _DEFAULT_STORE = store


def get_store() -> Store:
    """Get default store."""
    return _DEFAULT_STORE
