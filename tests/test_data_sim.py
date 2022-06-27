from tidy3d.components.simulation import Simulation
from tidy3d.components.grid.grid_spec import GridSpec
from tidy3d.components.data.sim_data import SimulationData

from .test_data_monitor import make_field_data, make_field_time_data, make_permittivity_data
from .test_data_monitor import make_mode_data, make_mode_field_data
from .test_data_monitor import make_flux_data, make_flux_time_data

# monitor data instances

FIELD = make_field_data()
FIELD_TIME = make_field_time_data()
PERMITTIVITY = make_permittivity_data()
MODE = make_mode_data()
MODE_FIELD = make_mode_field_data()
FLUX = make_flux_data()
FLUX_TIME = make_flux_time_data()

# for constructing SimulationData

MONITOR_DATA = (FIELD, FIELD_TIME, PERMITTIVITY, MODE, MODE_FIELD, FLUX, FLUX_TIME)
MONITOR_DATA_DICT = {data.monitor.name: data for data in MONITOR_DATA}
MONITORS = [data.monitor for data in MONITOR_DATA]
SIZE = (2, 2, 2)
GRID_SPEC = GridSpec(wavelength=1.0)
RUN_TIME = 1e-12


def make_sim_data():
    simulation = Simulation(monitors=MONITORS, size=SIZE, run_time=RUN_TIME, grid_spec=GRID_SPEC)
    return SimulationData(simulation=simulation, monitor_data=MONITOR_DATA_DICT)


def test_sim_data():
    sim_data = make_sim_data()


def test_getitem():
    sim_data = make_sim_data()
    for mon in sim_data.simulation.monitors:
        data = sim_data[mon.name]
