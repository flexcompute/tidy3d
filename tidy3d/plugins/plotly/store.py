from abc import ABC, abstractmethod

from tidy3d import SimulationData
from tidy3d.plugins.plotly import SimulationPlotly, SimulationDataApp
from tidy3d.plugins.plotly.data import DataPlotly

FrontEndStore = {}


class Store(ABC):
    @abstractmethod
    def get_simulation_data(self, front_end_store: FrontEndStore):
        pass

    def get_simulation_data(self, store: FrontEndStore) -> SimulationData:
        data = self.get_simulation_data(store)
        return SimulationPlotly(simulation=data.simulation)

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
        self.sim_data = SimulationDataApp.from_file(fname)

    def get_simulation_data(self, front_end_store: FrontEndStore):
        return self.sim_data


class S3Store(Store):

    def get_simulation_data(self, front_end_store: FrontEndStore):
        if front_end_store and front_end_store.get("task_id") == "1":
            sim_data = SimulationDataApp.from_file("../../../data/monitor_data.hdf5")
        else:
            sim_data = SimulationDataApp.from_file("../../../data/has_source.hdf5")
        return sim_data.sim_data


_DEFAULT_STORE = S3Store()


def set_store(store: Store):
    _DEFAULT_STORE = store


def get_store() -> Store:
    return _DEFAULT_STORE
