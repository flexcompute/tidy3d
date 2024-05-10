# test autograd integration into tidy3d

import autograd as ag
import autograd.numpy as npa
import tidy3d as td
from tidy3d.components.autograd import primitive, defvjp
import typing

from ..utils import run_emulated

WVL = 1.0
FREQ0 = td.C_0 / WVL
MNT_NAME = 'mnt'

"""This is code that will need to go into the regular components eventually."""

AUX_KEY_SIM_DATA = 'sim_data'

def _run_static(sim: td.Simulation) -> td.SimulationData:
    """Run a simulation without any tracers (would call web API)."""
    td.log.info('running static simulation _run_static()')
    return run_emulated(sim, task_name='autograd my life')
    
@primitive
def _run(sim_fields: npa.ndarray, sim: td.Simulation, aux_data: dict) -> tuple:
    """Autograd-traced running function, runs a simulation under the hood and stitches in data."""
    td.log.info('running primitive _run()')

    # TODO: implement
    sim_data = _run_static(sim.to_static())

    # TODO: refactor this to do it more systematically (see TODO in vjp too).
    datas = [npa.array(d.amps.values) for d in sim_data.data]

    aux_data[AUX_KEY_SIM_DATA] = sim_data
    return npa.array(datas)

def run(sim: td.Simulation) -> td.SimulationData:
    """User-facing run function, passes autograd primitive _run() the array of traced fields."""
    td.log.info('running user-facing run()')

    # TODO: implement
    sim_fields = sim.traced_fields()
    aux_data = {}
    data_fields = _run(sim_fields, sim=sim, aux_data=aux_data)
    if AUX_KEY_SIM_DATA not in aux_data:
        raise KeyError(f"Could not grab 'td.SimulationData' from 'aux_data[{AUX_KEY_SIM_DATA}]'. It might not have been properly stored.")
    sim_data = aux_data[AUX_KEY_SIM_DATA]

    # TODO: fix
    data = []
    for d, values in zip(sim_data.data, data_fields):
    	amps = d.amps.copy()
    	amps.values = values
    	data.append(d.copy(update=dict(amps=amps)))

    return sim_data.copy(update=dict(data=data))

def _run_bwd(data_fields_fwd: npa.ndarray, sim_fields_fwd : npa.ndarray, sim: td.Simulation, aux_data: dict) -> typing.Callable[[npa.ndarray], npa.ndarray]:
    """VJP-maker for _run(). Constructs and runs adjoint simulation, does postprocessing."""

    sim_data_fwd = aux_data[AUX_KEY_SIM_DATA]
    mnts = sim.monitors

    td.log.info('constructing custom vjp')

    def vjp(data_fields_vjp: npa.ndarray) -> npa.ndarray:
        """dJ/d{sim.traced_fields()} as a function of Function of dJ/d{data.traced_fields()}"""
        td.log.info('running custom vjp')
        amps_adj = [-1 * data_vjp for data_vjp in data_fields_vjp]
        sources_adj = []
        for mnt, amps in zip(mnts, amps_adj):

        	for direction, amp in zip('+-', amps):

	        	amplitude = float(npa.abs(amp))
	        	phase = float(npa.angle(amp))

		        src_adj = td.ModeSource(
		        	source_time=td.GaussianPulse(
		        		amplitude=amplitude,
		        		phase=phase,
		        		freq0=FREQ0,
		        		fwidth=FREQ0/10
		        	),
		        	size=mnt.size,
		        	center=mnt.center,
		        	direction=direction,
			    )
		        sources_adj.append(src_adj)

        sim_adj = sim.copy(update=dict(sources=sources_adj))
        sim_data_adj = run(sim_adj)

        # TODO: refactor this to do it more systematically
        data_fields_adj = [npa.array(d.amps.values) for d in sim_data_adj.data]

        # TODO: write the actual VJP rules
        vjp_values = []
        for fwd, adj in zip(data_fields_fwd, data_fields_adj):
        	vjp_value = npa.array(data_fields_fwd * data_fields_adj)
        	vjp_value = npa.sum(npa.abs(vjp_value))
        	vjp_values.append(vjp_value)
        return vjp_values

    return vjp

defvjp(_run, _run_bwd, argnums=[0])

"""END This is code that will need to go into the regular components eventually."""


def test_autograd_objective():


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

		data = run(sim)
		return npa.sum(abs(data[MNT_NAME].amps.values))**2

	grad = ag.grad(objective)(1.0)

	print(grad._value)

	assert grad != 0.0

