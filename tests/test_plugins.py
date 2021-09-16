import pytest
import numpy as np
import pydantic

import sys
sys.path.append('./')

from tidy3d.plugins.data_analyzer import *
from tidy3d import *

SIM = Simulation(
    size=(2.0, 2.0, 2.0),
    grid_size=(0.01, 0.01, 0.01),
    monitors={
        "field_freq": FieldMonitor(size=(0,0,0), center=(0,0,0), sampler=FreqSampler(freqs=[1,2, 5, 3, 3])),
        "field_time": FieldMonitor(size=(1,1,0), center=(0,0,0), sampler=TimeSampler(times=[1])),
        "flux_freq": FluxMonitor(size=(1,1,0), center=(0,0,0), sampler=FreqSampler(freqs=[1,2, 1, 5])),
        "flux_time": FluxMonitor(size=(1,1,0), center=(0,0,0), sampler=TimeSampler(times=[1,2, 3])),
        "mode": ModeMonitor(size=(1,1,0), center=(0,0,0), sampler=FreqSampler(freqs=[1.90,2.01, 2.0]), modes=[Mode(mode_index=1)])
    },
)

TASK_ID = 1001

def test_data():
	pass

def test_data_field():

	data_size = Nx, Ny, Nz = (5, 4, 3)
	xs, ys, zs = (np.random.random(s) for s in data_size)

	freq_mon = SIM.monitors['field_freq']
	Nf = len(freq_mon.sampler)
	E = (1 + 1j) * np.random.random((Nx, Ny, Nz, Nf))
	H = (1 + 1j) * np.random.random((Nx, Ny, Nz, Nf))

	f = FieldData(
		monitor=freq_mon,
		xs = xs,
		ys = ys,
		zs = zs,
		E = E,
		H = H)

	time_mon = SIM.monitors['field_time']
	Nt = len(time_mon.sampler)
	E = (1 + 1j) * np.random.random((Nx, Ny, Nz, Nt))
	H = (1 + 1j) * np.random.random((Nx, Ny, Nz, Nt))

	t = FieldData(
		monitor=time_mon,
		xs = xs,
		ys = ys,
		zs = zs,
		E = E,
		H = H)

def test_data_flux():

	data_size = (5, 4, 3)

	freq_mon = SIM.monitors['flux_freq']
	time_mon = SIM.monitors['flux_time']

	flux_freq = np.random.random(len(freq_mon.sampler.freqs))
	flux_time = np.random.random(len(time_mon.sampler.times))

	f = FluxData(
		monitor=freq_mon,
		flux=flux_freq)

	t = FluxData(
		monitor=time_mon,
		flux=flux_time)

def test_data_mode():

	data_size = (5, 4, 3)

	mode_mon = SIM.monitors['mode']
	Nf = len(mode_mon.sampler.freqs)
	Nm = len(mode_mon.modes)

	mode_amps = (1+1j)*np.random.random((Nm, Nf))

	f = ModeData(
		monitor=mode_mon,
		mode_amps=mode_amps)
