# test autograd integration into tidy3d

import pytest
import matplotlib.pylab as plt
import numpy as np
import cProfile
import typing

import autograd as ag
import autograd.numpy as npa

import tidy3d as td
import tidy3d.web as web

from ..utils import run_emulated

""" Test configuration """

"""Test modes
    pipeline: just run with emulated data, make sure gradient is not 0.0
    adjoint: run pipeline with real data through web API
    numerical: adjoint with an extra numerical derivative test after
    speed: pipeline with cProfile to analyze performance
"""

# make it faster to toggle this
TEST_CUSTOM_MEDIUM_SPEED = True
TEST_POLYSLAB_SPEED = False


TEST_MODES = ("pipeline", "adjoint", "numerical", "speed")
TEST_MODE = "speed" if TEST_POLYSLAB_SPEED else "pipeline"

# number of elements in the parameters / input to the objective function
N_PARAMS = 100

# default starting args
np.random.seed(1)
params0 = np.random.random(N_PARAMS)

# whether to plot the simulation within the objective function
PLOT_SIM = False

# whether to include a call to `objective(params)` in addition to gradient
CALL_OBJECTIVE = False

""" simulation configuration """

WVL = 1.0
FREQ0 = td.C_0 / WVL

# sim sizes
LZ = 7 * WVL

# NOTE: regular stuff is broken in 2D need to change volume and face integration to handle this
IS_3D = True

# TODO: test 2D and 3D parameterized

LX = 4 * WVL if IS_3D else 0.0
PML_X = True if IS_3D else False


# shape of the custom medium
DA_SHAPE_X = 1 if IS_3D else 1
DA_SHAPE = (DA_SHAPE_X, 1_000, 1_000) if TEST_CUSTOM_MEDIUM_SPEED else (DA_SHAPE_X, 50, 50)

# number of vertices in the polyslab
NUM_VERTICES = 100_000 if TEST_POLYSLAB_SPEED else 12

# sim that we add traced structures and monitors to
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
            center=(0, 0, 0),
            size=(0, 0, 0),
            freqs=[FREQ0],
            name="extraneous",
        )
    ],
    boundary_spec=td.BoundarySpec.pml(x=False, y=False, z=True),
)

# variable to store whether the emulated run as used
_run_was_emulated = [False]


@pytest.fixture
def use_emulated_run(monkeypatch):
    """If this fixture is used, the `tests.utils.run_emulated` function is used for simulation."""

    if TEST_MODE in ("pipeline", "speed"):
        import tidy3d.web.api.webapi as webapi

        monkeypatch.setattr(webapi, "run", run_emulated)
        _run_was_emulated[0] = True


def make_structures(params: npa.ndarray) -> dict[str, td.Structure]:
    """Make a dictionary of the structures given the parameters."""

    params_average = npa.mean(params)

    # static components
    box = td.Box(center=(0, 0, 0), size=(1, 1, 1))
    med = td.Medium(permittivity=2.0)

    # Structure with variable .medium
    eps = 1 + np.random.random(N_PARAMS) @ params
    conductivity = eps / 10.0
    medium = td.Structure(
        geometry=box,
        medium=td.Medium(permittivity=eps, conductivity=conductivity),
    )

    # Structure with variable Box.center
    center = (np.random.random((3, N_PARAMS)) - 0.5) @ params
    x0, y0, z0 = center
    center_list = td.Structure(
        geometry=td.Box(center=(x0, y0, z0), size=(1, 1, 1)),
        medium=med,
    )

    # Structure with variable Box.center
    size_y = 1 + np.random.random(N_PARAMS) @ params
    size_element = td.Structure(
        geometry=td.Box(center=(0, 0, 0), size=(1, size_y, 1)),
        medium=med,
    )

    # custom medium with variable permittivity data
    len_arr = np.prod(DA_SHAPE)
    num_params = len_arr // N_PARAMS

    # eps_arr = 1.0 + npa.stack(num_params * [params]).flatten()
    # eps_arr = eps_arr.reshape(DA_SHAPE)
    eps_arr = 1.0 + npa.dot(np.random.random((len_arr, N_PARAMS)), params).reshape(DA_SHAPE)

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

    # Polyslab with variable radius about origin
    radii = 1 + 0.1 * npa.dot(np.random.random((NUM_VERTICES, N_PARAMS)), params)
    phis = 2 * npa.pi * npa.linspace(0, 1, NUM_VERTICES + 1)[:NUM_VERTICES]
    xs = radii * npa.cos(phis)
    ys = radii * npa.sin(phis)
    vertices = npa.stack((xs, ys), axis=-1)
    polyslab = td.Structure(
        geometry=td.PolySlab(
            vertices=vertices,
            slab_bounds=(-0.5, 0.5),
            axis=1,
        ),
        medium=med,
    )

    return dict(
        medium=medium,
        center_list=center_list,
        size_element=size_element,
        custom_med=custom_med,
        polyslab=polyslab,
    )


def make_monitors() -> dict[str, tuple[td.Monitor, typing.Callable[[td.SimulationData], float]]]:
    """Make a dictionary of all the possible monitors in the simulation."""

    mode_mnt = td.ModeMonitor(
        size=(2, 2, 0),
        center=(0, 0, LZ / 2 - WVL),
        mode_spec=td.ModeSpec(),
        freqs=[FREQ0],
        name="mode",
    )

    mode_postprocess_fn = lambda mnt_data: npa.sum(abs(mnt_data.amps.values) ** 2)

    diff_mnt = td.DiffractionMonitor(
        size=(td.inf, td.inf, 0),
        center=(0, 0, -LZ / 2 + WVL),
        freqs=[FREQ0],
        normal_dir="+",
        name="diff",
    )

    diff_postprocess_fn = lambda mnt_data: npa.sum(abs(mnt_data.amps.values) ** 2)

    return dict(
        mode=(mode_mnt, mode_postprocess_fn),
        diff=(diff_mnt, diff_postprocess_fn),
    )


def plot_sim(sim: td.Simulation, plot_eps: bool = False) -> None:
    """Plot the simulation."""

    plot_fn = sim.plot_eps if plot_eps else sim.plot

    f, (ax1, ax2, ax3) = plt.subplots(1, 3, tight_layout=True)
    plot_fn(x=0, ax=ax1)
    plot_fn(y=0, ax=ax2)
    plot_fn(z=0, ax=ax3)
    plt.show()


# TODO: grab these automatically
structure_keys_ = ("medium", "center_list", "size_element", "custom_med", "polyslab")
monitor_keys_ = ("mode", "diff")

# generate combos of all structures with each monitor and all monitors with each structure
ALL_KEY = "<ALL>"
args = []
for s in structure_keys_:
    args.append((s, ALL_KEY))

for m in monitor_keys_:
    args.append((ALL_KEY, m))

# or just set args manually to test certain things
if TEST_CUSTOM_MEDIUM_SPEED:
    args = [("custom_med", "mode")]

if TEST_POLYSLAB_SPEED:
    args = [("polyslab", "mode")]


@pytest.mark.parametrize("structure_key, monitor_key", args)
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
    td.config.logging_level = "ERROR"

    monitor_dict = make_monitors()

    monitors = list(SIM_BASE.monitors)
    monitor_pp_fns = {}
    for monitor_key in monitor_keys:
        monitor_traced, monitor_pp_fn = monitor_dict[monitor_key]
        monitors.append(monitor_traced)
        monitor_pp_fns[monitor_key] = monitor_pp_fn

    def make_sim(*args) -> td.Simulation:
        """Make the simulation with all of the fields."""

        structures_traced_dict = make_structures(*args)

        structures = list(SIM_BASE.structures)
        for structure_key in structure_keys:
            structures.append(structures_traced_dict[structure_key])

        return SIM_BASE.updated_copy(structures=structures, monitors=monitors)

    def postprocess(data: td.SimulationData) -> float:
        """Postprocess the dataset."""
        mnt_data = data[monitor_key]
        return monitor_pp_fn(mnt_data)

    def objective(*args):
        """Objective function."""
        sim = make_sim(*args)
        if PLOT_SIM:
            plot_sim(sim, plot_eps=True)
        data = run_ag(sim)
        value = postprocess(data)
        return value

    # if speed test, get the profile
    if TEST_MODE == "speed":
        with cProfile.Profile() as pr:
            val, grad = ag.value_and_grad(objective)(params0)
            pr.print_stats(sort="cumtime")
            pr.dump_stats("results.prof")

    # otherwise, just test that it ran and the gradients are all non-zero
    else:
        if CALL_OBJECTIVE:
            val = objective(params0)
        val, grad = ag.value_and_grad(objective)(params0)
        print(val, grad)
        assert npa.all(grad != 0.0), "some gradients are 0"

    # if 'numerical', we do a numerical gradient check
    if TEST_MODE == "numerical":
        delta = 1e-8
        sims_numerical = {}

        params_num = np.zeros((N_PARAMS, N_PARAMS))

        for i in range(N_PARAMS):
            for sign, pm_string in zip((-1, 1), "-+"):
                task_name = f"{i}_{pm_string}"
                params_i = np.copy(params0)
                params_i[i] += sign * delta
                params_num[i] = params_i.copy()
                sim_i = make_sim(params_i)
                sims_numerical[task_name] = sim_i

        datas = web.run_async(sims_numerical, path_dir="data")

        grad_num = np.zeros_like(grad)
        objectives_num = np.zeros((len(params0), 2))
        for i in range(N_PARAMS):
            for sign, pm_string in zip((-1, 1), "-+"):
                task_name = f"{i}_{pm_string}"
                sim_data_i = datas[task_name]
                obj_i = postprocess(sim_data_i)
                objectives_num[i, (sign + 1) // 2] = obj_i
                grad_num[i] += sign * obj_i / 2 / delta

        print("adjoint: ", grad)
        print("numerical: ", grad_num)

        assert np.allclose(grad, grad_num), "gradients dont match"

    # for logging output
    td.config.logging_level = "WARNING"
