#!/usr/bin/env python
# coding: utf-8

# # Start Here

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import matplotlib.pylab as plt

import sys
sys.path.append('..')

import tidy3d as td
import tidy3d.web as web


# set up parameters of simulation
dl=0.01
pml = td.PML(num_layers=10)
sim_size = [4, 4, 4]
freq0 = 3e14
fwidth = 1e13
run_time = 10/fwidth

# create structure
dielectric = td.nk_to_medium(n=2, k=0, freq=freq0)
square = td.Structure(
    geometry=td.Box(center=[0, 0, 0], size=[1.5, 1.5, 1.5]),
    medium=dielectric)

# create source
source = td.VolumeSource(
    center=(-1.5, 0, 0),
    size=(0, 0.4, 0.4),
    source_time = td.GaussianPulse(
        freq0=freq0,
        fwidth=fwidth),
    polarization='Ex')

# create monitor
monitor = td.FieldMonitor(
    fields=['Ex', 'Hy'],
    center=(0, 0, 0),
    size=(4, 4, 0),
    freqs=[freq0])

# Initialize simulation
sim = td.Simulation(size=sim_size,
                    grid_size=(dl, dl, dl),
                    structures=[square],
                    sources={'source': source},
                    monitors={'monitor': monitor},
                    run_time=run_time,
                    pml_layers=(pml, pml, pml))


task_id = web.upload(sim, task_name='quickstart')
web.start(task_id)
web.monitor(task_id)


web.download(task_id, simulation=sim, path='data/sim_data.hdf5')


sim_data = web.load_data(task_id, simulation=sim, path='data/sim_data.hdf5')
ax = sim_data['monitor'].Ex.isel(f=0, z=0).imag.plot.pcolormesh(x='x', y='y', vmin=-5e-13, vmax=5e-13, cmap='RdBu')

plt.show()


sim_data.log




