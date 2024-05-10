# test autograd integration into tidy3d

import pytest

import autograd as ag
import autograd.numpy as npa
import tidy3d as td
from tidy3d.components.autograd import primitive, defvjp
import typing

from ..utils import run_emulated

_run_was_emulated = [False]

@pytest.fixture
def use_emulated_run(monkeypatch):
    """If this fixture is used, the `tests.utils.run_emulated` function is used for simulation."""
    import tidy3d.web.api.webapi as webapi
    monkeypatch.setattr(webapi, "run", run_emulated)
    run_was_emulated[0] = True

WVL = 1.0
FREQ0 = td.C_0 / WVL

# sim and structure sizes in x
BX = 2 * WVL
NUM_STCRS = 5
LX = 2 * NUM_STCRS * WVL

NUM_MNTS = 3
MNT_NAME = 'mnt'

def mnt_name_i(i: int):
	return f"{MNT_NAME}_{i}"

# for logging output
td.config.logging_level = "INFO"

def test_autograd_objective(use_emulated_run):

	# import here so it uses emulated run
	from tidy3d.web.api.autograd import run as run_ag
	run_was_emulated = _run_was_emulated[0]

	def objective(params):

		permittivities = params

		# medium = td.Medium(permittivity=permittivity)
		# box = td.Box(size=(BX,1,1), center=(0,0,0))

		structure_centers = npa.linspace(-LX/2 + BX, LX/2 - BX, NUM_STCRS)

		structures = [
			td.Structure(
				geometry=td.Box(size=(BX,1,1), center=(x0,0,0)),
				medium=td.Medium(permittivity=eps)
			)
		for eps, x0 in zip(params, structure_centers)]

		mnts = []
		for i in range(NUM_MNTS):
			mnt_i = td.ModeMonitor(
				size=(2,2,0),
				center=(0,0,1),
				mode_spec=td.ModeSpec(),
				freqs=[FREQ0],
				name=mnt_name_i(i),
			)
			mnts.append(mnt_i)

		src = td.PointDipole(
			center=(0,0,-1),
			polarization="Ey",
			source_time=td.GaussianPulse(
				freq0=FREQ0,
				fwidth=FREQ0/10.0,
				amplitude=1.0,
			)
		)
		sim = td.Simulation(
			size=(LX,3,3),
			run_time=1e-12,
			grid_spec=td.GridSpec.auto(min_steps_per_wvl=25),
			structures=structures,
			sources=[src],
			monitors=mnts,
		)

		data = run_ag(sim)

		value = 0.0
		for i in range(NUM_MNTS):
			name = mnt_name_i(i)
			amps_i = data[name].amps
			value_i = npa.sum(abs(amps_i.values))**2
			value += value_i

		return value

	params0 = NUM_STCRS * [2.0]

	if True or run_was_emulated:
		val, grad = ag.value_and_grad(objective)(params0)
	else:
		val = 6.431317914419032e-09
		grad = npa.array([6.6703622e-05, 3.5446883e-05, 0.00852042, 7.34919959e-05, 7.27415099e-05])

	print(val, grad)

	assert not npa.allclose(grad, 0.0)

	# numerical gradient (if not emulating run)

	if not run_was_emulated:

		# grad_numerical = npa.array([-2.72940140e-06, -7.69690465e-06,  7.14835332e-06, -9.46320901e-07, -2.73337469e-06])

		delta = 4e-2
		grad_numerical = npa.zeros_like(grad)
		for i in range(NUM_STCRS):
			for sign in (-1, 1):
				_params_i = npa.array(params0).copy()
				_params_i[i] += sign * delta
				print(f' params {i}: {_params_i}')
				_val_i = objective(_params_i)
				print(f' val {i}: {_val_i}')
				grad_numerical[i] += sign * _val_i / 2 / delta

		print(grad_numerical)
		import pdb; pdb.set_trace()










