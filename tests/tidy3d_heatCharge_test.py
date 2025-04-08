# standard python imports


# tidy3d imports
import tidy3d as td
from tidy3d import web
from tidy3d.web.environment import Env

# from tidy3d.web import simulation_task

Env.dev.active()
# web.configure("AITmcZOOrPfQcQJqcJwueSm3s21QsNdTRRlUj4VA7q5l71SC")
web.configure("Nc9IKGtKueHqcU3674m3oTW1ClP3nbOHXCasx3EPxI1jLs4Y")  # shijie
# web.configure("ZAadiucEubb9e3lFztGDTt9Wda4cuQ5x3RYRDDtxUgJ0ZETK") // momchil

# web.account()
# Env.uat2.active()
# web.configure("42Xu66tfjGVqPhAr6ou6gDPDZukuiMgtYjkwnT0muy7cylf8")

# Env.prod.active()
# web.configure("VyusDxqFEsQquauM1EQy0WpEblcswkJoI9G3C6oysBcziyuJ")

Env.prod.active()
# web.configure("AITmcZOOrPfQcQJqcJwueSm3s21QsNdTRRlUj4VA7q5l71SC")
# web.configure("42Xu66tfjGVqPhAr6ou6gDPDZukuiMgtYjkwnT0muy7cylf8") # uat
web.configure("VyusDxqFEsQquauM1EQy0WpEblcswkJoI9G3C6oysBcziyuJ")

sim = td.HeatChargeSimulation.from_file("heat-charge-dev.hdf5")

result = web.run(sim, folder_name="Mode-HeatCharge", task_name="heatCharge")
