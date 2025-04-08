# standard python imports

import numpy as np

# tidy3d imports
import tidy3d as td
from tidy3d import web
from tidy3d.plugins.mode import ModeSolver
from tidy3d.plugins.mode import web as modeweb
from tidy3d.web.environment import Env

# from tidy3d.web import simulation_task


def create_task_sim():
    global freq0, fwidth, sim
    # Simulation domain size (in micron)
    sim_size = [4, 4, 4]
    # Central frequency and bandwidth of pulsed excitation, in Hz
    freq0 = 2e14
    fwidth = 1e13
    # apply a PML in all directions
    boundary_spec = td.BoundarySpec.all_sides(boundary=td.PML())
    # Total time to run in seconds
    run_time = 2 / fwidth
    # Lossless dielectric specified directly using relative permittivity
    material1 = td.Medium(permittivity=6.0)
    # Lossy dielectric defined from the real and imaginary part of the refractive index
    material2 = td.Medium.from_nk(n=1.5, k=0.0, freq=freq0)
    # material2 = td.Medium(permittivity=2.)
    # Rectangular slab, extending infinitely in x and y with medium `material1`
    box = td.Structure(
        geometry=td.Box(center=[0, 0, 0], size=[td.inf, td.inf, 1]), medium=material1
    )
    # Triangle in the xy-plane with a finite extent in z
    equi_tri_verts = [[-1 / 2, -1 / 4], [1 / 2, -1 / 4], [0, np.sqrt(3) / 2 - 1 / 4]]
    poly = td.Structure(
        geometry=td.PolySlab(
            vertices=(2 * np.array(equi_tri_verts)).tolist(),
            # vertices=equi_tri_verts,
            slab_bounds=(0.5, 1.0),
            axis=2,
        ),
        medium=material2,
    )
    psource = td.PlaneWave(
        center=(0, 0, 1.5),
        direction="-",
        size=(td.inf, td.inf, 0),
        source_time=td.GaussianPulse(freq0=freq0, fwidth=fwidth),
        pol_angle=np.pi / 2,
    )
    # measure time domain fields at center location, measure every 5 time steps
    time_mnt = td.FieldTimeMonitor(center=[0, 0, 0], size=[0, 0, 0], interval=8, name="field_time")
    # measure the steady state fields at central frequency in the xy plane and the xz plane.
    freq_mnt1 = td.FieldMonitor(center=[0, 0, -1], size=[20, 20, 0], freqs=[freq0], name="field1")
    # freq_mnt2 = td.FieldMonitor(center=[0, 0, 0], size=[20, 0, 20], freqs=[freq0], name="field2")
    # Initialize simulation
    sim = td.Simulation(
        size=sim_size,
        grid_spec=td.GridSpec.auto(min_steps_per_wvl=20),
        structures=[box, poly],
        sources=[psource],
        monitors=[time_mnt, freq_mnt1],
        run_time=run_time,
        boundary_spec=boundary_spec,
    )
    return sim


# task = simulation_task.SimulationTask.create(None, "test task", "default")
# task.submit()
# Env.dev.active()
# web.configure("vRmoURI8xHjYHsEgdmeerP7sUOstxGhh7pp3xJc7x3w1RJSX")


# Env.dev.active()
# web.configure("AITmcZOOrPfQcQJqcJwueSm3s21QsNdTRRlUj4VA7q5l71SC")
# web.configure("ZAadiucEubb9e3lFztGDTt9Wda4cuQ5x3RYRDDtxUgJ0ZETK") // momchil

# web.account()
# Env.uat2.active()
# web.configure("42Xu66tfjGVqPhAr6ou6gDPDZukuiMgtYjkwnT0muy7cylf8")

Env.prod.active()
web.configure("VyusDxqFEsQquauM1EQy0WpEblcswkJoI9G3C6oysBcziyuJ")

Env.enable_caching(False)
# sim = create_task_sim()
# sim=Simulation.from_file("simulation.json")
# result = subprocess.run(['md5sum', "111.hdf5"], capture_output=True, text=True)
# print(result)
# Env.enable_caching(False)

sim = ModeSolver.from_file("ModeSolver.hdf5")

result = modeweb.run(sim, folder_name="modeSimulation", task_name="modeSimulation")
