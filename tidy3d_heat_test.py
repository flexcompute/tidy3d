from tidy3d.web.core.environment import Env
from tidy3d import web, HeatSimulation

Env.dev.active()

sim = HeatSimulation.from_file("C:/Users/ywujo/OneDrive/Desktop/tidy3d/heat_sim.json")
sim_data = web.run(sim, "heat-test")
