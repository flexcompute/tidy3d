from abc import ABC, abstractmethod

from ...components import SimulationData
from .simulation import SimulationPlotly
from .data import DataPlotly

FrontEndStore = {}

class Store(ABC):
    @abstractmethod
    def get_simulation_data(self, front_end_store: FrontEndStore) -> SimulationData:
        pass

    def get_simulation_plotly(self, store: []) -> SimulationPlotly:
        data = self.get_simulation_data(store)
        return SimulationPlotly(simulation=data.simulation)

    def get_data_plotly_by_name(self, store: FrontEndStore, name: str) -> DataPlotly:
        sim = self.get_simulation_data(store)
        for monitor_name, monitor_data in sim.monitor_data.items():
            if monitor_name == name:
                return DataPlotly.from_monitor_data(
                    monitor_data=monitor_data, monitor_name=monitor_name
                )


class LocalStore(Store):
    def __init__(self, fname: str):
        self.sim_data = SimulationData.from_file(fname)

    def get_simulation_data(self, front_end_store: FrontEndStore):
        return self.sim_data


class EmptyStore(Store):
    def get_simulation_data(self, front_end_store: FrontEndStore):
        pass

_DEFAULT_STORE = EmptyStore()


def set_store(store: Store):
    global _DEFAULT_STORE
    _DEFAULT_STORE = store


def get_store() -> Store:
    return _DEFAULT_STORE
