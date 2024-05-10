# test autograd integration into tidy3d

import pytest

import autograd as ag
import autograd.numpy as npa
import tidy3d as td
from tidy3d.components.autograd import primitive, defvjp
import typing

from ..utils import run_emulated

@pytest.fixture
def use_emulated_run(monkeypatch):
    """If this fixture is used, the `tests.utils.run_emulated` function is used for simulation."""
    import tidy3d.web.api.webapi as webapi
    monkeypatch.setattr(webapi, "run", run_emulated)

WVL = 1.0
FREQ0 = td.C_0 / WVL
MNT_NAME = 'mnt'

# for logging output
td.config.logging_level = "INFO"

def test_autograd_objective(use_emulated_run):

	# import here so it uses emulated run
	from tidy3d.web.api.autograd import run as run_ag

	def objective(params):
		permittivity = params
		medium = td.Medium(permittivity=permittivity)
		box = td.Box(size=(1,1,1), center=(0,0,0))
		structure = td.Structure(geometry=box, medium=medium)
		mnt = td.ModeMonitor(
			size=(1,1,0),
			center=(1,0,0),
			mode_spec=td.ModeSpec(),
			freqs=[FREQ0],
			name=MNT_NAME,
		)
		sim = td.Simulation(
			size=(5,5,5),
			run_time=1e-12,
			sources=[],
			grid_spec=td.GridSpec.auto(wavelength=WVL),
			structures=[structure],
			monitors=[mnt],
		)

		data = run_ag(sim)
		return npa.sum(abs(data[MNT_NAME].amps.values))**2

	grad = ag.grad(objective)(1.0)

	print(grad)

	assert grad != 0.0

