from tidy3d import Simulation
import matplotlib.pylab as plt

def viz_data(sim: Simulation, monitor_name: str) -> None:
    data = sim.data[monitor_name]

    plt.imshow(data)
    plt.show()
