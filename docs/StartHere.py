#!/usr/bin/env python
# coding: utf-8

import sys

sys.path.append("..")

import tidy3d as td

# set up parameters of simulation
dl = 0.1
pml = td.PMLLayer(profile="standard", num_layers=10)
sim_size = [4, 4, 4]
freq0 = 3e14
fwidth = 1e13
run_time = 2 / fwidth

# create structure
dielectric = td.nk_to_medium(n=2, k=0, freq=freq0)
square = td.Structure(geometry=td.Box(center=[0, 0, 0], size=[1.5, 1.5, 1.5]), medium=dielectric)

# create source
source = td.PlaneWave(
    center=(0, -1.5, 0),
    size=(td.inf, 0, td.inf),
    source_time=td.GaussianPulse(freq0=freq0, fwidth=fwidth),
    polarization="Jx",
    direction="+",
)

# create monitor
monitor = td.FieldMonitor(
    fields=["Ex", "Hy"], center=(0, 0, 0), size=(td.inf, 0, td.inf), freqs=[freq0]
)

# Initialize simulation
sim = td.Simulation(
    size=sim_size,
    grid_size=(dl, dl, dl),
    structures=[square],
    sources={"plane_wave": source},
    monitors={"field_monitor": monitor},
    run_time=run_time,
    pml_layers=(pml, pml, pml),
)


# # run simulation
# project = web.new_project(sim.export(), task_name='quickstart')
# web.monitor_project(project['taskId'])

# # download results
# web.download_results(project['taskId'], target_folder='out/')
# sim.load_results('out/monitor_data.hdf5')

# # get field data as a numpy array
# E = sim.data(freq_mnt)["E"]

# # plot results
# clim = (1, 3)
# fig, (ax_top, ax_bot) = plt.subplots(2, 2, figsize=(9, 7), tight_layout=True)

# sim.viz_eps_2D(normal='y', ax=ax_top[0], cbar=True, clim=clim, monitor_alpha=0)
# im_re = sim.viz_field_2D(freq_mnt, ax=ax_top[1], cbar=True, comp='y', val='re')

# im_ab = sim.viz_field_2D(freq_mnt, ax=ax_bot[0], cbar=True, comp='y', val='abs')
# im_int = sim.viz_field_2D(freq_mnt, ax=ax_bot[1], cbar=True, comp='y', val='int')

# plt.show()
