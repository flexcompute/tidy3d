from tidy3d import SimulationData
from tidy3d.plugins.plotly import SimulationPlotly, SimulationDataApp
from tidy3d.plugins.plotly.data import DataPlotly


def get_simulation_data(store: {}) -> SimulationData:
    if store and store.get("task_id") == "1":
        sim_data = SimulationDataApp.from_file("../../../data/monitor_data.hdf5")
    else:
        sim_data = SimulationDataApp.from_file("../../../data/no_source.hdf5")
    return sim_data.sim_data


def get_simulation_plotly(store: []) -> SimulationPlotly:
    data = get_simulation_data(store)
    return SimulationPlotly(simulation=data.simulation)


def get_data_plotly_by_name(store: [], name: str) -> DataPlotly:
    sim = get_simulation_data(store)
    for monitor_name, monitor_data in sim.monitor_data.items():
        if monitor_name == name:
            return DataPlotly.from_monitor_data(
                monitor_data=monitor_data, monitor_name=monitor_name
            )
