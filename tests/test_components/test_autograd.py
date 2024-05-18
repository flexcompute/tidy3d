# test autograd integration into tidy3d

import pytest
import matplotlib.pylab as plt
import numpy as np

import autograd as ag
import autograd.numpy as npa

import tidy3d as td

from ..utils import run_emulated


WVL = 1.0
FREQ0 = td.C_0 / WVL

# sim and structure sizes in x
BX = 2 * WVL
NUM_STCRS = 5
LX = 3 * NUM_STCRS * WVL
LZ = 7 * WVL

NUM_MNTS = 3
MNT_NAME = "mnt"

PLOT_SIM = False

DA_SHAPE = (3, 4, 1)


SIM_BASE = td.Simulation(
    size=(LX, 3, LZ),
    run_time=1e-12,
    sources=[
        td.PointDipole(
            center=(0, 0, -LZ / 2 + WVL),
            polarization="Ey",
            source_time=td.GaussianPulse(
                freq0=FREQ0,
                fwidth=FREQ0 / 10.0,
                amplitude=1.0,
            ),
        )
    ],
    structures=[
        td.Structure(
            geometry=td.Box(
                size=(0.5, 0.5, LZ / 2),
                center=(0, 0, LZ / 2),
            ),
            medium=td.Medium(permittivity=2.0),
        )
    ],
    monitors=[
        td.FieldMonitor(
            center=(0,0,0),
            size=(0, 0, 0),
            freqs=[FREQ0],
            name='extraneous',
        )
    ],
    boundary_spec=td.BoundarySpec.pml(x=False, y=False, z=True),
)

# variable to store whether the emulated run as used
_run_was_emulated = [False]


@pytest.fixture
def use_emulated_run(monkeypatch):
    """If this fixture is used, the `tests.utils.run_emulated` function is used for simulation."""
    import tidy3d.web.api.webapi as webapi
    monkeypatch.setattr(webapi, "run", run_emulated)
    _run_was_emulated[0] = True


# default starting args
eps0 = 2.0
center0 = (0.0, 0.0, 0.0)
size_y0 = 1.0
eps_arr0 = np.random.random(DA_SHAPE) + 1.0
args0 = (eps0, center0, size_y0, eps_arr0)
argnum = tuple(range(len(args0)))

def make_structures(eps, center, size_y, eps_arr) -> dict[str, td.Structure]:
    """Make a dictionary of the structures given the parameters."""

    box = td.Box(center=(0,0,0), size=(1,1,1))
    med = td.Medium(permittivity=2.0)

    medium = td.Structure(
        geometry=box,
        medium=td.Medium(permittivity=eps, conductivity=eps/10.0),
    )

    center_list = td.Structure(
        geometry=td.Box(center=center, size=(1,1,1)),
        medium=med,
    )

    size_element = td.Structure(
        geometry=td.Box(center=(0,0,0), size=(1,size_y,1)),
        medium=med,
    )


    nx, ny, nz = eps_arr.shape
    custom_med = td.Structure(
        geometry=box,
        medium=td.CustomMedium(
            permittivity=td.SpatialDataArray(
                eps_arr,
                coords=dict(
                    x=np.linspace(-0.5, 0.5, nx),
                    y=np.linspace(-0.5, 0.5, ny),
                    z=np.linspace(-0.5, 0.5, nz),
                ),
            ),
        ),
    )

    return dict(
        medium=medium,
        center_list=center_list,
        size_element=size_element,
        custom_med=custom_med,
    )

def make_monitors() -> dict[str, td.Monitor]:
    """Make a dictionary of all the possible monitors in the simulation."""

    mode = td.ModeMonitor(
        size=(2, 2, 0),
        center=(0, 0, LZ / 2 - WVL),
        mode_spec=td.ModeSpec(),
        freqs=[FREQ0],
        name="mode",
    )

    mode_pp = lambda mnt_data: npa.sum(abs(mnt_data.amps.values)**2)

    diff = td.DiffractionMonitor(
        size=(td.inf, td.inf, 0),
        center=(0, 0, -LZ / 2 + WVL),
        freqs=[FREQ0],
        normal_dir="+",
        name="diff",
    )

    diff_pp = lambda mnt_data: npa.sum(abs(mnt_data.amps.values)**2)

    return dict(
        mode=(mode, mode_pp),
        diff=(diff, diff_pp),
    )

def plot_sim(sim: td.Simulation) -> None:
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, tight_layout=True)
    sim.plot(x=0, ax=ax1)
    sim.plot(y=0, ax=ax2)
    sim.plot(z=0, ax=ax3)
    plt.show()

# TODO: grab these automatically
structure_keys_ = ('medium', 'center_list', 'size_element', 'custom_med')
monitor_keys_ = ('mode', 'diff')

ALL_KEY = "<ALL>"

args = []
for s in structure_keys_:
    args.append((s, ALL_KEY))

for m in monitor_keys_:
    args.append((ALL_KEY, m))

@pytest.mark.parametrize('structure_key, monitor_key', args)
def test_autograd_objective(use_emulated_run, structure_key, monitor_key):
    """Test an objective function through tidy3d autograd."""

    if structure_key == ALL_KEY:
        structure_keys = structure_keys_
    else:
        structure_keys = [structure_key]

    if monitor_key == ALL_KEY:
        monitor_keys = monitor_keys_
    else:
        monitor_keys = [monitor_key]

    # import here so it uses emulated run
    from tidy3d.web.api.autograd import run as run_ag

    # for logging output
    td.config.logging_level = "INFO"

    monitor_dict = make_monitors()

    monitors = list(SIM_BASE.monitors)
    monitor_pp_fns = {}
    for monitor_key in monitor_keys:
        monitor_traced, monitor_pp_fn = monitor_dict[monitor_key]
        monitors.append(monitor_traced)
        monitor_pp_fns[monitor_key] = monitor_pp_fn

    def make_sim(eps, center, size_y, eps_arr) -> td.Simulation:
        """Make the simulation with all of the fields."""

        structures_traced_dict = make_structures(eps, center, size_y, eps_arr)

        structures = list(SIM_BASE.structures)
        for structure_key in structure_keys:
            structures.append(structures_traced_dict[structure_key])

        return SIM_BASE.updated_copy(
            structures=structures,
            monitors=monitors
        )

    def postprocess(data: td.SimulationData) -> float:
        """Postprocess the dataset."""
        mnt_data = data[monitor_key]
        return monitor_pp_fn(mnt_data)


    def objective(*args):
        """Objective function."""
        sim = make_sim(*args)
        if PLOT_SIM:
            plot_sim(sim)
        data = run_ag(sim)
        value = postprocess(data)
        return value

    val, grad = ag.value_and_grad(objective, argnum=argnum)(*args0)

    grad_are_0 = [np.all(np.array(grad_i) == 0) for grad_i in grad]

    assert not npa.all(grad_are_0), "All gradient elements were 0, this combo was un-traced."

    # for logging output
    td.config.logging_level = "WARNING"
