""" Loads solver results into SimulationData"""
from tidy3d import Simulation, SimulationData, FieldData, data_type_map

from .solver import SolverDataDict


def load_solver_results(
    simulation: Simulation,
    solver_data_dict: SolverDataDict,
    log_string: str = None,
) -> SimulationData:
    """load the solver_data_dict and simulation into SimulationData"""

    # constuct monitor_data dictionary
    monitor_data = {}
    for name, monitor in simulation.monitors.items():
        monitor_data_dict = solver_data_dict[name]
        monitor_data_type = data_type_map[monitor.data_type]
        if monitor.type in ("FieldMonitor", "FieldTimeMonitor"):
            field_data = {}
            for field_name, data_dict in monitor_data_dict.items():
                field_data[field_name] = monitor_data_type(**data_dict)
            monitor_data[name] = FieldData(data_dict=field_data)
        else:
            monitor_data[name] = monitor_data_type(**monitor_data_dict)
    return SimulationData(simulation=simulation, monitor_data=monitor_data, log_string=log_string)


def save_solver_results(path: str, sim: Simulation, solver_dict: SolverDataDict) -> None:
    """save the solver_data_dict and simulation json to file"""

    # create SimulationData object
    sim_data = load_solver_results(sim, solver_dict)

    # export the solver data to path
    sim_data.export(path)
    # potentially: export HTML for viz or other files here?
