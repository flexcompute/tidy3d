# Test the inverse design plugin
import pytest

import numpy as np
import jax.numpy as jnp

import tidy3d as td
import tidy3d.plugins.adjoint as tda
import tidy3d.plugins.invdes as tdi

from .test_adjoint import use_emulated_run

FREQ0 = 1e14
L_SIM = 2.0
MNT_NAME = "mnt_name"
PARAMS_SHAPE = (18,19,20)
PARAMS_0 = np.random.random(PARAMS_SHAPE)

def test_pipeline(use_emulated_run):
	"""Test a full run with all of the possible objects."""

	simulation = td.Simulation(
		size=(L_SIM,L_SIM,L_SIM),
		grid_spec=td.GridSpec.auto(wavelength=td.C_0/FREQ0),
		sources=[td.PointDipole(
			center=(0,0,0),
			source_time=td.GaussianPulse(freq0=FREQ0, fwidth=FREQ0/10),
			polarization="Ez",
		)],
		run_time=1e-12,
	)

	class custom_transformation(tdi.Transformation):
		def evaluate(self, data: jnp.ndarray) -> jnp.ndarray:
			return data / jnp.mean(data)

	class custom_penalty(tdi.Penalty):
		def evaluate(self, data: jnp.ndarray) -> float:
			return jnp.mean(data)

	design_region = tdi.TopologyDesignRegion(
		size=(0.4 * L_SIM, 0.4 * L_SIM, 0.4 * L_SIM),
		center=(0, 0, 0),
		eps_bounds=(1.0, 4.0),
		symmetry=(0, 1, -1),
		params_shape=PARAMS_SHAPE,
		transformations=[
			tdi.CircularFilter(radius=0.2, design_region_dl=0.1),
			tdi.BinaryProjector(beta=2.0, vmin=0.0, vmax=1.0),
			# custom_transformation, # TODO: fix these
			tdi.ConicFilter(radius=0.2, design_region_dl=0.1),
		],
		penalties=[
			tdi.ErosionDilationPenalty(length_scale=0.2, pixel_size=0.1),
			# custom_penalty, # TODO: fix these
		],
		penalty_weights=[0.2],
	)

	# test some design region functions
	design_region.material_density(PARAMS_0)
	design_region.penalty_value(PARAMS_0)
	design_region.updated_copy(penalty_weights=None).penalty_value(PARAMS_0)
	design_region_no_penalties = design_region.updated_copy(penalties=[], penalty_weights=None)
	with pytest.raises(ValueError):
		design_region_no_penalties._penalty_weights
	assert design_region_no_penalties.penalty_value(PARAMS_0) == 0.0


	mnt = td.FieldMonitor(
		center=(L_SIM/3.0, 0, 0),
		size=(0, td.inf, td.inf),
		freqs=[FREQ0],
		name=MNT_NAME
	)

	optimizer = tdi.Optimizer(
		learning_rate=0.2,
		num_steps=10,
	)

	invdes = tdi.InverseDesign(
		simulation=simulation,
		design_region=design_region,
		output_monitors=[mnt],
		optimizer=optimizer,
		params0=np.random.random(PARAMS_SHAPE).tolist(),
	)

	def post_process_fn(sim_data: tda.JaxSimulationData, scale: float = 2.0) -> float:
		intensity = sim_data.get_intensity(MNT_NAME)
		return scale * jnp.sum(intensity.values)

	result = invdes.run(post_process_fn, task_name='blah')

