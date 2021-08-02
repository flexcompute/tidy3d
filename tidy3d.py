from dataclasses import dataclass, asdict
from collections import defaultdict
import numpy as np
import shutil

""" ==== structure ==== """

@dataclass
class Geometry:
    def inside(x,y,z):
        pass

@dataclass
class Medium:
    permittivity: float = 1.0
    conductivity: float = 0.0

@dataclass
class Structure:
    geometry: Geometry
    medium: Medium

""" ==== source ==== """

@dataclass
class Source:
    geometry: Geometry
    source_time: list[float]

""" ==== monitor ==== """

@dataclass
class Monitor:
    geometry: Geometry
    monitor_time: list[float]

@dataclass
class Data:
    monitor: Monitor

""" ==== simulation ==== """

@dataclass
class Mesh:
    size: tuple[float]
    dl: tuple[list[float]]

@dataclass
class Simulation:
    mesh: Mesh
    structures: dict[Structure]
    sources: dict[Source]
    monitors: dict[Monitor]
    data: dict[Data]

sim = Simulation(
    mesh = Mesh(
        size=(1., 2., 1.),
        dl=(0.01, 0.01, 0.01),

    ),
    structures = {
        'square': Structure(
            geometry=Geometry(),
            medium=Medium(permittivity=2.0)
        ),
        'box': Structure(
            geometry=Geometry(),
            medium=Medium(conductivity=3.0)
        )        
    },
    sources = {
        'planewave': Source(
            geometry=Geometry(),
            source_time=[0.0, 0.01, 0.1, 0.2, 0.1, 0.01, 0.0]
        )
    },
    monitors = {
        'point': Monitor(
            geometry=Geometry(),
            monitor_time=[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        ),
        'plane': Monitor(
            geometry=Geometry(),
            monitor_time=[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        )
    },
    data = {}
)


def export(sim : Simulation) -> str:
    return asdict(sim)

def new_project(json : dict) -> int:
    task_id = 101
    monitors = json['monitors']
    for name, mon in monitors.items():
        data = np.random.random((4, 4))
        np.save(f'data_server/task_{task_id}_monitor_{name}.npy', data)
    return task_id

def load(sim : Simulation, task_id : int) -> None:
    monitors = sim.monitors
    for name, mon in monitors.items():
        fname = f'task_{task_id}_monitor_{name}.npy'
        shutil.copyfile(f'data_server/{fname}', f'data_user/{fname}')
        data = np.load(f'data_user/{fname}')
        sim.data[name] = data

def viz_data(sim : Simulation, monitor_name : str) -> None:
    data = sim.data[monitor_name]
    import matplotlib.pylab as plt
    plt.imshow(data)
    plt.show()

# example usage
if __name__ == '__main__':
    json = export(sim)                      # validation, schema matching
    task_id = new_project(json)             # export to server
    load(sim, task_id)                      # load data into sim.data containers
    viz_data(sim, 'plane')                   # vizualize
