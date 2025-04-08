# standard python imports

# tidy3d imports
from tidy3d import Simulation, web
from tidy3d.web.api.container import Batch
from tidy3d.web.environment import Env

# from tidy3d.web import simulation_task

Env.dev.active()
web.configure("AITmcZOOrPfQcQJqcJwueSm3s21QsNdTRRlUj4VA7q5l71SC")
# web.configure("ZAadiucEubb9e3lFztGDTt9Wda4cuQ5x3RYRDDtxUgJ0ZETK") // momchil

# web.account()
# Env.uat.active()
# web.configure("42Xu66tfjGVqPhAr6ou6gDPDZukuiMgtYjkwnT0muy7cylf8")

# Env.prod.active()
# web.configure("VyusDxqFEsQquauM1EQy0WpEblcswkJoI9G3C6oysBcziyuJ")

Env.enable_caching(False)
sim = Simulation.from_file("FDTD_2.7.7.hdf5")


sims = {}

for i in range(1):
    key = f"fdtd_{i}"
    value = sim
    sims[key] = value

custom_batch = Batch(simulations=sims, folder_name="pythonclient-batch")
batch_data = custom_batch.run()  # doctest: +SKIP


# task_id = web.run(sim, folder_name="PythonClientTasks", task_name="fdtd-pre2.6")
# print("Max flex unit cost: ", web.estimate_cost(task_id))

# web.estimate_cost("fdve-67ee097c-d332-4d62-9f95-346fb5ce2bedv1")
# task_id = web.load_simulation("fdve-f0e1295b-4ff9-4e16-866d-573e9688d234v2", "/tmp/simulation.json")
