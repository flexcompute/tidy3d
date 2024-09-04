# test autograd integration into tidy3d

import copy
import cProfile
import typing
import warnings
from importlib import reload
from os.path import join

import autograd as ag
import autograd.numpy as anp
import matplotlib.pylab as plt
import numpy as np
import numpy.testing as npt
import pytest
import tidy3d as td
import tidy3d.web as web
import xarray as xr
from autograd.test_util import check_grads
from tidy3d.components.autograd.derivative_utils import DerivativeInfo
from tidy3d.components.autograd.utils import is_tidy_box
from tidy3d.components.data.data_array import DataArray
from tidy3d.web import run, run_async
from tidy3d.web.api.autograd.utils import FieldMap

from ..utils import SIM_FULL, AssertLogLevel, run_emulated, tracer_arr

""" Test configuration """

"""Test modes
    pipeline: just run with emulated data, make sure gradient is not 0.0
    adjoint: run pipeline with real data through web API
    numerical: adjoint with an extra numerical derivative test after
    speed: pipeline with cProfile to analyze performance
"""

# make it faster to toggle this
TEST_CUSTOM_MEDIUM_SPEED = False
TEST_POLYSLAB_SPEED = False

# whether to run numerical gradient tests, off by default because it runs real simulations
RUN_NUMERICAL = False
_NUMERICAL_COMBINATION = ("size_element", "mode")

TEST_MODES = ("pipeline", "adjoint", "speed")
TEST_MODE = "speed" if TEST_POLYSLAB_SPEED else "pipeline"

# number of elements in the parameters / input to the objective function
N_PARAMS = 10

# default starting args
np.random.seed(1)
params0 = np.random.random(N_PARAMS) - 0.5
params0 /= np.linalg.norm(params0)

# whether to plot the simulation within the objective function
PLOT_SIM = False

# whether to include a call to `objective(params)` in addition to gradient
CALL_OBJECTIVE = False

""" simulation configuration """

WVL = 1.0
FREQ0 = td.C_0 / WVL
FREQS = [FREQ0]
FWIDTH = FREQ0 / 10

# sim sizes
LZ = 7.0 * WVL

IS_3D = False

# TODO: test 2D and 3D parameterized

LX = 0.5 * WVL if IS_3D else 0.0
PML_X = True if IS_3D else False


# shape of the custom medium
DA_SHAPE_X = 1 if IS_3D else 1
DA_SHAPE = (DA_SHAPE_X, 1_000, 1_000) if TEST_CUSTOM_MEDIUM_SPEED else (DA_SHAPE_X, 12, 12)

# number of vertices in the polyslab
NUM_VERTICES = 100_000 if TEST_POLYSLAB_SPEED else 110

PNT_DIPOLE = td.PointDipole(
    center=(0, 0, -LZ / 2 + WVL),
    polarization="Ey",
    source_time=td.GaussianPulse(
        freq0=FREQ0,
        fwidth=FWIDTH,
        amplitude=1.0,
    ),
)

PLANE_WAVE = td.PlaneWave(
    center=(0, 0, -LZ / 2 + WVL),
    size=(td.inf, td.inf, 0),
    direction="+",
    source_time=td.GaussianPulse(
        freq0=FREQ0,
        fwidth=FWIDTH,
        amplitude=1.0,
    ),
)

# sim that we add traced structures and monitors to
SIM_BASE = td.Simulation(
    size=(LX, 3.15, LZ),
    run_time=200 / FWIDTH,
    sources=[PLANE_WAVE],
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
    boundary_spec=td.BoundarySpec.pml(x=False, y=True, z=True),
    grid_spec=td.GridSpec.uniform(dl=0.01 * td.C_0 / FREQ0),
)

# variable to store whether the emulated run as used
_run_was_emulated = [False]


@pytest.fixture
def use_emulated_run(monkeypatch):
    """If this fixture is used, the `tests.utils.run_emulated` function is used for simulation."""

    import tidy3d

    if TEST_MODE in ("pipeline", "speed"):
        task_id_fwd = "task_fwd"
        AUX_KEY_SIM_FIELDS_KEYS = "sim_fields_keys"

        cache = {}

        import tidy3d.web.api.webapi as webapi

        # reload(tidy3d.web.api.autograd.autograd)
        from tidy3d.web.api.autograd.autograd import (
            AUX_KEY_SIM_DATA_FWD,
            AUX_KEY_SIM_DATA_ORIGINAL,
            postprocess_adj,
            postprocess_fwd,
        )

        def emulated_run_fwd(simulation, task_name, **run_kwargs) -> td.SimulationData:
            """What gets called instead of ``web/api/autograd/autograd.py::_run_tidy3d``."""
            task_id_fwd = task_name
            if run_kwargs.get("simulation_type") == "autograd_fwd":
                sim_original = simulation
                sim_fields_keys = run_kwargs["sim_fields_keys"]
                # add gradient monitors and make combined simulation
                sim_combined = sim_original.with_adjoint_monitors(sim_fields_keys)
                sim_data_combined = run_emulated(sim_combined, task_name=task_name)

                # store both original and fwd data aux_data
                aux_data = {}

                _ = postprocess_fwd(
                    sim_data_combined=sim_data_combined,
                    sim_original=sim_original,
                    aux_data=aux_data,
                )

                # cache original and fwd data locally for test
                cache[task_id_fwd] = copy.copy(aux_data)
                cache[task_id_fwd][AUX_KEY_SIM_FIELDS_KEYS] = sim_fields_keys
                # return original data only
                return aux_data[AUX_KEY_SIM_DATA_ORIGINAL], task_id_fwd
            else:
                return run_emulated(simulation, task_name=task_name), task_id_fwd

        def emulated_run_bwd(simulation, task_name, **run_kwargs) -> td.SimulationData:
            """What gets called instead of ``web/api/autograd/autograd.py::_run_tidy3d_bwd``."""

            task_id_fwd = task_name[:-8]

            # run the adjoint sim
            sim_data_adj = run_emulated(simulation, task_name="task_name")

            # grab the fwd and original data from the cache
            aux_data_fwd = cache[task_id_fwd]
            sim_data_orig = aux_data_fwd[AUX_KEY_SIM_DATA_ORIGINAL]
            sim_data_fwd = aux_data_fwd[AUX_KEY_SIM_DATA_FWD]

            # get the original traced fields
            sim_fields_keys = cache[task_id_fwd][AUX_KEY_SIM_FIELDS_KEYS]

            # postprocess (compute adjoint gradients)
            traced_fields_vjp = postprocess_adj(
                sim_data_adj=sim_data_adj,
                sim_data_orig=sim_data_orig,
                sim_data_fwd=sim_data_fwd,
                sim_fields_keys=sim_fields_keys,
            )

            return traced_fields_vjp

        def emulated_run_async_fwd(simulations, **run_kwargs) -> td.SimulationData:
            batch_data_orig, task_ids_fwd = {}, {}
            sim_fields_keys_dict = run_kwargs.pop("sim_fields_keys_dict", None)
            for task_name, simulation in simulations.items():
                if sim_fields_keys_dict is not None:
                    run_kwargs["sim_fields_keys"] = sim_fields_keys_dict[task_name]
                sim_data_orig, task_id_fwd = emulated_run_fwd(simulation, task_name, **run_kwargs)
                batch_data_orig[task_name] = sim_data_orig
                task_ids_fwd[task_name] = task_id_fwd

            return batch_data_orig, task_ids_fwd

        def emulated_run_async_bwd(simulations, **run_kwargs) -> td.SimulationData:
            vjp_dict = {}
            for task_name, simulation in simulations.items():
                task_id_fwd = task_name[:-8]
                vjp_dict[task_name] = emulated_run_bwd(simulation, task_name, **run_kwargs)
            return vjp_dict

        monkeypatch.setattr(webapi, "run", run_emulated)
        monkeypatch.setattr(tidy3d.web.api.autograd.autograd, "_run_tidy3d", emulated_run_fwd)
        monkeypatch.setattr(tidy3d.web.api.autograd.autograd, "_run_tidy3d_bwd", emulated_run_bwd)
        monkeypatch.setattr(
            tidy3d.web.api.autograd.autograd, "_run_async_tidy3d", emulated_run_async_fwd
        )
        monkeypatch.setattr(
            tidy3d.web.api.autograd.autograd, "_run_async_tidy3d_bwd", emulated_run_async_bwd
        )

        _run_was_emulated[0] = True
        return emulated_run_fwd, emulated_run_bwd


def make_structures(params: anp.ndarray) -> dict[str, td.Structure]:
    """Make a dictionary of the structures given the parameters."""

    np.random.seed(0)

    vector = np.random.random(N_PARAMS) - 0.5
    vector = vector / np.linalg.norm(vector)

    # static components
    box = td.Box(center=(0, 0, 0), size=(1, 1, 1))
    med = td.Medium(permittivity=3.0)

    # Structure with variable .medium
    eps = 1 + anp.abs(vector @ params)
    sigma = 0.1 * (anp.tanh(vector @ params) + 1)

    permittivity, conductivity = eps, sigma

    medium = td.Structure(
        geometry=box,
        medium=td.Medium(permittivity=permittivity, conductivity=conductivity),
    )

    # Structure with variable Box.center
    matrix = np.random.random((3, N_PARAMS)) - 0.5
    matrix /= np.linalg.norm(matrix)
    center = anp.tanh(matrix @ params)
    x0, y0, z0 = center
    center_list = td.Structure(
        geometry=td.Box(center=(x0, y0, z0), size=(1, 1, 1)),
        medium=med,
    )

    # Structure with variable Box.center
    size_y = anp.abs(vector @ params)
    size_element = td.Structure(
        geometry=td.Box(center=(0, 0, 0), size=(1, size_y, 1)),
        medium=med,
        background_permittivity=5.0,
    )

    # custom medium with variable permittivity data
    len_arr = np.prod(DA_SHAPE)
    matrix = np.random.random((len_arr, N_PARAMS))
    # matrix /= np.linalg.norm(matrix)

    eps_arr = 1.01 + 0.5 * (anp.tanh(matrix @ params).reshape(DA_SHAPE) + 1)

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

    # custom medium with vector valued permittivity data
    eps_ii = td.ScalarFieldDataArray(
        eps_arr.reshape(nx, ny, nz, 1),
        coords=dict(
            x=np.linspace(-0.5, 0.5, nx),
            y=np.linspace(-0.5, 0.5, ny),
            z=np.linspace(-0.5, 0.5, nz),
            f=[td.C_0],
        ),
    )

    custom_med_vec = td.Structure(
        geometry=box,
        medium=td.CustomMedium(
            eps_dataset=td.PermittivityDataset(eps_xx=eps_ii, eps_yy=eps_ii, eps_zz=eps_ii)
        ),
    )

    # Polyslab with variable radius about origin
    # matrix = np.random.random((NUM_VERTICES, N_PARAMS)) - 0.5
    # params_01 = 0.5 * (anp.tanh(matrix @ params / 3) + 1)
    matrix = np.random.random((N_PARAMS,)) - 0.5
    params_01 = 0.5 * (anp.tanh(matrix @ params / 3) + 1)

    radii = 1.0 + 0.5 * params_01

    phis = 2 * anp.pi * anp.linspace(0, 1, NUM_VERTICES + 1)[:NUM_VERTICES]
    xs = radii * anp.cos(phis)
    ys = radii * anp.sin(phis)
    vertices = anp.stack((xs, ys), axis=-1)
    polyslab = td.Structure(
        geometry=td.PolySlab(
            vertices=vertices,
            slab_bounds=(-0.5, 0.5),
            axis=0,
            sidewall_angle=0.01,
            dilation=0.01,
        ),
        medium=med,
    )

    # geometry group
    geo_group = td.Structure(
        geometry=td.GeometryGroup(
            geometries=[
                medium.geometry,
                center_list.geometry,
                size_element.geometry,
            ],
        ),
        medium=td.Medium(permittivity=eps, conductivity=conductivity),
    )

    # dispersive medium
    eps_inf = 1 + anp.abs(vector @ params)
    box = td.Box(center=(0, 0, 0), size=(1, 1, 1))

    a0 = -FREQ0 * eps_inf + 1j * FREQ0 * eps_inf
    c0 = FREQ0 * eps_inf + 1j * FREQ0 * eps_inf
    a1 = -2 * FREQ0 * eps_inf + 1j * FREQ0 * eps_inf
    c1 = 2 * FREQ0 * eps_inf + 1j * FREQ0 * eps_inf

    med = td.PoleResidue(eps_inf=eps_inf, poles=[(a0, c0), (a1, c1)])
    pole_res = td.Structure(geometry=box, medium=med)

    # custom dispersive medium
    len_arr = np.prod(DA_SHAPE)
    matrix = np.random.random((len_arr, N_PARAMS))
    matrix /= np.linalg.norm(matrix)

    eps_arr = 1.01 + 0.5 * (anp.tanh(matrix @ params).reshape(DA_SHAPE) + 1)
    custom_disp_values = 1.01 + (0.5 + 0.5j) * (anp.tanh(matrix @ params).reshape(DA_SHAPE) + 1)

    nx, ny, nz = custom_disp_values.shape
    x = np.linspace(-0.5, 0.5, nx)
    y = np.linspace(-0.5, 0.5, ny)
    z = np.linspace(-0.5, 0.5, nz)
    coords = dict(x=x, y=y, z=z)

    eps_inf = td.SpatialDataArray(anp.real(custom_disp_values), coords=coords)
    a1 = td.SpatialDataArray(-custom_disp_values, coords=coords)
    c1 = td.SpatialDataArray(custom_disp_values, coords=coords)
    a2 = td.SpatialDataArray(-custom_disp_values, coords=coords)
    c2 = td.SpatialDataArray(custom_disp_values, coords=coords)
    custom_med_pole_res = td.CustomPoleResidue(eps_inf=eps_inf, poles=[(a1, c1), (a2, c2)])
    custom_pole_res = td.Structure(geometry=box, medium=custom_med_pole_res)

    radius = 0.4 * (1 + anp.abs(vector @ params))
    cyl_center_y = 0  # vector @ params
    cyl_center_z = 0  # -vector @ params
    cylinder_geo = td.Cylinder(
        radius=radii,
        center=(0, cyl_center_y, cyl_center_z),
        axis=0,
        length=LX / 2 if IS_3D else td.inf,
    )
    cylinder = td.Structure(geometry=cylinder_geo, medium=polyslab.medium)

    return dict(
        medium=medium,
        center_list=center_list,
        size_element=size_element,
        custom_med=custom_med,
        custom_med_vec=custom_med_vec,
        polyslab=polyslab,
        geo_group=geo_group,
        pole_res=pole_res,
        custom_pole_res=custom_pole_res,
        cylinder=cylinder,
    )


def make_monitors() -> dict[str, tuple[td.Monitor, typing.Callable[[td.SimulationData], float]]]:
    """Make a dictionary of all the possible monitors in the simulation."""

    X = 0.75

    mode_mnt = td.ModeMonitor(
        size=(2, 2, 0),
        center=(0, 0, +LZ / 2 - X * WVL),
        mode_spec=td.ModeSpec(),
        freqs=[FREQ0],
        name="mode",
    )

    def mode_postprocess_fn(sim_data, mnt_data):
        return anp.sum(abs(mnt_data.amps.values) ** 2)

    diff_mnt = td.DiffractionMonitor(
        size=(td.inf, td.inf, 0),
        center=(0, 0, +LZ / 2 - 2 * WVL),
        freqs=[FREQ0],
        normal_dir="+",
        name="diff",
    )

    def diff_postprocess_fn(sim_data, mnt_data):
        return anp.sum(abs(mnt_data.amps.sel(polarization=["s", "p"]).values) ** 2)

    field_vol = td.FieldMonitor(
        size=(1, 1, 0),
        center=(0, 0, +LZ / 2 - X * WVL),
        freqs=[FREQ0],
        name="field_vol",
    )

    def field_vol_postprocess_fn(sim_data, mnt_data):
        value = 0.0
        for _, val in mnt_data.field_components.items():
            value = value + abs(anp.sum(val.values))
        intensity = anp.nan_to_num(anp.sum(sim_data.get_intensity(mnt_data.monitor.name).values))
        value += intensity
        value += anp.sum(mnt_data.flux.values)
        return value

    field_point = td.FieldMonitor(
        size=(0, 0, 0),
        center=(0, 0, LZ / 2 - WVL),
        freqs=[FREQ0],
        name="field_point",
    )

    def field_point_postprocess_fn(sim_data, mnt_data):
        value = 0.0
        for _, val in mnt_data.field_components.items():
            value += abs(anp.sum(abs(val.values)))
        value += anp.sum(sim_data.get_intensity(mnt_data.monitor.name).values)
        return value

    return dict(
        mode=(mode_mnt, mode_postprocess_fn),
        diff=(diff_mnt, diff_postprocess_fn),
        field_vol=(field_vol, field_vol_postprocess_fn),
        field_point=(field_point, field_point_postprocess_fn),
    )


def plot_sim(sim: td.Simulation, plot_eps: bool = True) -> None:
    """Plot the simulation."""

    sim = sim.to_static()

    plot_fn = sim.plot_eps if plot_eps else sim.plot

    f, (ax1, ax2, ax3) = plt.subplots(1, 3, tight_layout=True)
    plot_fn(x=0, ax=ax1)
    plot_fn(y=0, ax=ax2)
    plot_fn(z=0, ax=ax3)
    plt.show()


# TODO: grab these automatically
structure_keys_ = (
    "medium",
    "center_list",
    "size_element",
    "custom_med",
    "custom_med_vec",
    "polyslab",
    "geo_group",
    "pole_res",
    "custom_pole_res",
    "cylinder",
)
monitor_keys_ = ("mode", "diff", "field_vol", "field_point")

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


# args = [("size_element", "mode")]


def get_functions(structure_key: str, monitor_key: str) -> typing.Callable:
    if structure_key == ALL_KEY:
        structure_keys = structure_keys_
    else:
        structure_keys = [structure_key]

    if monitor_key == ALL_KEY:
        monitor_keys = monitor_keys_
    else:
        monitor_keys = [monitor_key]

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

        sim = SIM_BASE
        if "diff" in monitor_dict:
            sim = sim.updated_copy(boundary_spec=td.BoundarySpec.pml(x=False, y=False, z=True))
        sim = sim.updated_copy(structures=structures, monitors=monitors)

        return sim

    def postprocess(data: td.SimulationData) -> float:
        """Postprocess the dataset."""
        mnt_data = data[monitor_key]
        return monitor_pp_fn(data, mnt_data)

    return dict(sim=make_sim, postprocess=postprocess)


@pytest.mark.parametrize("axis", (0, 1, 2))
def test_polyslab_axis_ops(axis):
    vertices = ((0, 0), (0, 1), (1, 1), (1, 0))
    p = td.PolySlab(vertices=vertices, axis=axis, slab_bounds=(0, 1))

    ax_coords = np.array([0, 1, 2, 3])
    plane_coords = np.array([[4, 5], [6, 7], [8, 9], [10, 11]])
    coord = p.unpop_axis_vect(ax_coords=ax_coords, plane_coords=plane_coords)

    assert np.all(coord[:, axis] == ax_coords)

    _ax_coords, _plane_coords = p.pop_axis_vect(coord=coord)

    assert np.all(_ax_coords == ax_coords)
    assert np.all(_plane_coords == plane_coords)

    vertices_next = np.roll(vertices, axis=0, shift=-1)
    edges = vertices_next - vertices

    basis_vecs = p.edge_basis_vectors(edges=edges)


@pytest.mark.skipif(not RUN_NUMERICAL, reason="Numerical gradient tests runs through web API.")
@pytest.mark.parametrize("structure_key, monitor_key", (_NUMERICAL_COMBINATION,))
def test_autograd_numerical(structure_key, monitor_key):
    """Test an objective function through tidy3d autograd."""

    import tidy3d.web as web

    fn_dict = get_functions(structure_key, monitor_key)
    make_sim = fn_dict["sim"]
    postprocess = fn_dict["postprocess"]

    def objective(*args):
        """Objective function."""
        sim = make_sim(*args)
        if PLOT_SIM:
            plot_sim(sim, plot_eps=True)
        data = web.run(sim, task_name="autograd_test_numerical", verbose=False)
        value = postprocess(data)
        return value

    val, grad = ag.value_and_grad(objective)(params0)
    print(val, grad)
    assert anp.all(grad != 0.0), "some gradients are 0"

    # numerical gradients
    delta = 1e-3
    sims_numerical = {}

    params_num = np.zeros((N_PARAMS, N_PARAMS))

    def task_name_fn(i: int, sign: int) -> str:
        """Task name for a given index into grad num and sign."""
        pm_string = "+" if sign > 0 else "-"
        return f"{i}_{pm_string}"

    for i in range(N_PARAMS):
        for j, sign in enumerate((-1, 1)):
            task_name = task_name_fn(i, sign)
            params_i = np.copy(params0)
            params_i[i] += sign * delta
            params_num[:, j] = params_i.copy()
            sim_i = make_sim(params_i)
            sims_numerical[task_name] = sim_i

    datas = web.Batch(simulations=sims_numerical).run(path_dir="data")

    grad_num = np.zeros_like(grad)
    objectives_num = np.zeros((len(params0), 2))
    for i in range(N_PARAMS):
        for j, sign in enumerate((-1, 1)):
            task_name = task_name_fn(i, sign)
            sim_data_i = datas[task_name]
            obj_i = postprocess(sim_data_i)
            objectives_num[i, j] = obj_i
            grad_num[i] += sign * obj_i / 2 / delta

    print("adjoint: ", grad)
    print("numerical: ", grad_num)

    print(objectives_num)

    grad_normalized = grad / np.linalg.norm(grad)
    grad_num_normalized = grad_num / np.linalg.norm(grad_num)

    rms_error = np.linalg.norm(grad_normalized - grad_num_normalized)
    norm_factor = np.linalg.norm(grad) / np.linalg.norm(grad_num)

    diff_objectives_num = np.mean(abs(np.diff(objectives_num, axis=-1)))

    print(f"rms_error = {rms_error:.4f}")
    print(f"|grad| / |grad_num| = {norm_factor:.4f}")
    print(f"avg(diff(objectives)) = {diff_objectives_num:.4f}")


@pytest.mark.parametrize("structure_key, monitor_key", args)
def test_autograd_objective(use_emulated_run, structure_key, monitor_key):
    """Test an objective function through tidy3d autograd."""

    fn_dict = get_functions(structure_key, monitor_key)
    make_sim = fn_dict["sim"]
    postprocess = fn_dict["postprocess"]

    def objective(*args):
        """Objective function."""
        sim = make_sim(*args)
        if PLOT_SIM:
            plot_sim(sim, plot_eps=True)
        data = run(sim, task_name="autograd_test", verbose=False)
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
        assert anp.all(grad != 0.0), "some gradients are 0"


@pytest.mark.parametrize("structure_key, monitor_key", args)
def test_autograd_async(use_emulated_run, structure_key, monitor_key):
    """Test an objective function through tidy3d autograd."""

    fn_dict = get_functions(structure_key, monitor_key)
    make_sim = fn_dict["sim"]
    postprocess = fn_dict["postprocess"]

    task_names = {"1", "2", "3", "4"}

    def objective(*args):
        """Objective function."""

        sims = {task_name: make_sim(*args) for task_name in task_names}
        batch_data = run_async(sims, verbose=False)
        value = 0.0
        for _, sim_data in batch_data.items():
            value += postprocess(sim_data)
        return value

    val, grad = ag.value_and_grad(objective)(params0)
    print(val, grad)
    assert anp.all(grad != 0.0), "some gradients are 0"


def test_autograd_speed_num_structures(use_emulated_run):
    """Test an objective function through tidy3d autograd."""

    num_structures_test = 10

    import time

    fn_dict = get_functions(ALL_KEY, ALL_KEY)

    monitor_key = "mode"
    structure_key = "size_element"
    monitor, postprocess = make_monitors()[monitor_key]

    def make_sim(*args):
        structure = make_structures(*args)[structure_key]
        structures = num_structures_test * [structure]
        return SIM_BASE.updated_copy(structures=structures, monitors=[monitor])

    def objective(*args):
        """Objective function."""
        sim = make_sim(*args)
        data = run(sim, task_name="autograd_test", verbose=False)
        value = postprocess(data, data[monitor_key])
        return value

    # if speed test, get the profile
    with cProfile.Profile() as pr:
        t = time.time()
        val, grad = ag.value_and_grad(objective)(params0)
        t2 = time.time() - t
        pr.print_stats(sort="cumtime")
        pr.dump_stats("results.prof")
        print(f"{num_structures_test} structures took {t2:.2e} seconds")


@pytest.mark.parametrize("monitor_key", ("mode",))
def test_autograd_polyslab_cylinder(use_emulated_run, monitor_key):
    """Test an objective function through tidy3d autograd."""

    fn_dict_polyslab = get_functions("polyslab", monitor_key)
    make_sim_polyslab = fn_dict_polyslab["sim"]

    fn_dict_cylinder = get_functions("cylinder", monitor_key)
    make_sim_cylinder = fn_dict_cylinder["sim"]

    postprocess = fn_dict_cylinder["postprocess"]

    def objective_polyslab(*args):
        """Objective function."""
        sim = make_sim_polyslab(*args)
        if PLOT_SIM:
            plot_sim(sim, plot_eps=True)
        data = run(sim, task_name="autograd_test", verbose=False)
        value = postprocess(data)
        return value

    val_polyslab, grad_polyslab = ag.value_and_grad(objective_polyslab)(params0)
    print(val_polyslab, grad_polyslab)
    assert anp.all(grad_polyslab != 0.0), "some gradients are 0"

    def objective_cylinder(*args):
        """Objective function."""
        sim = make_sim_cylinder(*args)
        if PLOT_SIM:
            plot_sim(sim, plot_eps=True)
        data = run(sim, task_name="autograd_test", verbose=False)
        value = postprocess(data)
        return value

    val_cylinder, grad_cylinder = ag.value_and_grad(objective_cylinder)(params0)
    print(val_cylinder, grad_cylinder)
    assert anp.all(grad_cylinder != 0.0), "some gradients are 0"

    # just make sure they're somewhat close (use different discretizations)
    assert np.allclose(grad_cylinder, grad_polyslab, rtol=0.3)


@pytest.mark.parametrize("structure_key, monitor_key", args)
def test_autograd_server(use_emulated_run, structure_key, monitor_key):
    """Test an objective function through tidy3d autograd."""

    fn_dict = get_functions(structure_key, monitor_key)
    make_sim = fn_dict["sim"]
    postprocess = fn_dict["postprocess"]

    def objective(*args):
        """Objective function."""
        sim = make_sim(*args)
        data = run(sim, task_name="autograd_test", verbose=False, local_gradient=False)
        value = postprocess(data)
        return value

        val, grad = ag.value_and_grad(objective)(params0)
        print(val, grad)
        assert anp.all(grad != 0.0), "some gradients are 0"

    val, grad = ag.value_and_grad(objective)(params0)


@pytest.mark.parametrize("structure_key, monitor_key", args)
def test_autograd_async_server(use_emulated_run, structure_key, monitor_key):
    """Test an async objective function through tidy3d autograd."""

    fn_dict = get_functions(structure_key, monitor_key)
    make_sim = fn_dict["sim"]
    postprocess = fn_dict["postprocess"]

    def objective(*args):
        """Objective function."""
        sim = make_sim(*args)
        sims = {"autograd_test1": sim, "autograd_test2": sim}
        batch_data = run_async(sims, verbose=False, local_gradient=False)
        value = 0.0
        for _, sim_data in batch_data.items():
            value = value + postprocess(sim_data)
        return value

        val, grad = ag.value_and_grad(objective)(params0)
        print(val, grad)
        assert anp.all(grad != 0.0), "some gradients are 0"

    val, grad = ag.value_and_grad(objective)(params0)


@pytest.mark.parametrize("structure_key", ("custom_med",))
def test_sim_full_ops(structure_key):
    """make sure the autograd operations don't error on a simulation containing everything."""

    def objective(*params):
        s = make_structures(*params)[structure_key]
        s = s.updated_copy(geometry=s.geometry.updated_copy(center=(2, 2, 2), size=(0, 0, 0)))
        sim_full_traced = SIM_FULL.updated_copy(structures=list(SIM_FULL.structures) + [s])

        sim_full_static = sim_full_traced.to_static()

        sim_fields = sim_full_traced.strip_traced_fields()

        # note: there is one traced structure in SIM_FULL already with 6 fields + 1 = 7
        assert len(sim_fields) == 10

        sim_traced = sim_full_static.insert_traced_fields(sim_fields)

        assert sim_traced == sim_full_traced

        return anp.sum(sim_full_traced.structures[-1].medium.permittivity.values)

    ag.grad(objective)(params0)


def test_sim_traced_override_structures(log_capture):
    """Make sure that sims with traced override structures are handled properly."""

    def f(x):
        override_structure = td.MeshOverrideStructure(
            geometry=td.Box(center=(0, 0, 0), size=(1, 1, x)),
            dl=[1, 1, 1],
        )
        sim = SIM_FULL.updated_copy(override_structures=[override_structure], path="grid_spec")
        return sim.grid_spec.override_structures[0].geometry.size[2]

    with AssertLogLevel(log_capture, "WARNING", contains_str="override structures"):
        ag.grad(f)(1.0)


@pytest.mark.parametrize("structure_key", ("custom_med",))
def test_sim_fields_io(structure_key, tmp_path):
    """Test that converging and AutogradFieldMap dictionary to a FieldMap object, saving and loading
    from file, and then converting back, returns the same object."""
    s = make_structures(params0)[structure_key]
    s = s.updated_copy(geometry=s.geometry.updated_copy(center=(2, 2, 2), size=(0, 0, 0)))
    sim_full_traced = SIM_FULL.updated_copy(structures=list(SIM_FULL.structures) + [s])
    sim_fields = sim_full_traced.strip_traced_fields()

    field_map = FieldMap.from_autograd_field_map(sim_fields)
    field_map_file = join(tmp_path, "test_sim_fields.hdf5.gz")
    field_map.to_file(field_map_file)
    autograd_field_map = FieldMap.from_file(field_map_file).to_autograd_field_map
    for path, data in sim_fields.items():
        assert np.all(data == autograd_field_map[path])


def test_web_incompatible_inputs(log_capture, monkeypatch):
    """Test what happens when bad inputs passed to web.run()."""

    def catch(*args, **kwargs):
        """Just raise an exception."""
        raise AssertionError()

    monkeypatch.setattr(td.web.api.webapi, "run", catch)
    monkeypatch.setattr(td.web.api.container.Job, "run", catch)
    monkeypatch.setattr(td.web.api.asynchronous, "run_async", catch)

    from tidy3d.web.api.autograd import autograd

    reload(autograd)

    # no tracers

    with pytest.raises(AssertionError):
        td.web.run(SIM_BASE, task_name="task_name")

    with pytest.raises(AssertionError):
        td.web.run_async({"task_name": SIM_BASE})

    with pytest.raises(AssertionError):
        autograd._run(SIM_BASE, task_name="task_name")

    # wrong input types

    with pytest.raises(AssertionError):
        td.web.run([SIM_BASE], task_name="test")

    with pytest.raises(AssertionError):
        td.web.run_async([SIM_BASE])


def test_too_many_traced_structures(monkeypatch, log_capture, use_emulated_run):
    """More traced structures than allowed."""

    from tidy3d.web.api.autograd.autograd import MAX_NUM_TRACED_STRUCTURES

    monitor_key = "mode"
    structure_key = "size_element"
    monitor, postprocess = make_monitors()[monitor_key]

    def make_sim(*args):
        structure = make_structures(*args)[structure_key]
        return SIM_BASE.updated_copy(
            structures=(MAX_NUM_TRACED_STRUCTURES + 1) * [structure], monitors=[monitor]
        )

    def objective(*args):
        """Objective function."""
        sim = make_sim(*args)
        data = run(sim, task_name="autograd_test", verbose=False)
        value = postprocess(data, data[monitor_key])
        return value

    with pytest.raises(ValueError):
        ag.grad(objective)(params0)


@pytest.mark.parametrize("colocate", [True, False])
@pytest.mark.parametrize("objtype", ["flux", "intensity"])
def test_interp_objectives(use_emulated_run, colocate, objtype):
    monitor = td.FieldMonitor(
        center=(0, 0, 0),
        size=(td.inf, td.inf, 0),
        freqs=[FREQ0],
        name="monitor",
        colocate=colocate,
    )

    def objective(args):
        structures_traced_dict = make_structures(args)
        structures = list(SIM_BASE.structures)
        for structure_key in structure_keys_:
            structures.append(structures_traced_dict[structure_key])

        sim = SIM_BASE.updated_copy(monitors=[monitor], structures=structures)
        data = run(sim, task_name="autograd_test", verbose=False)

        if objtype == "flux":
            return data[monitor.name].flux.item()
        elif objtype == "intensity":
            return anp.sum(data.get_intensity(monitor.name).values)

    grads = ag.grad(objective)(params0)
    assert np.any(grads > 0)


@pytest.mark.parametrize("far_field_approx", [True, False])
@pytest.mark.parametrize("projection_type", ["angular", "cartesian"])
@pytest.mark.parametrize("sim_2d", [True, False])
class TestFieldProjection:
    @staticmethod
    def setup(far_field_approx, projection_type, sim_2d):
        if sim_2d and not far_field_approx:
            pytest.skip("Exact field projection not implemented for 2d simulations")

        r_proj = 50 * WVL
        monitor = td.FieldMonitor(
            center=(0, SIM_BASE.size[1] / 2 - 0.1, 0),
            size=(td.inf, 0, td.inf),
            freqs=[FREQ0],
            name="near_field",
            colocate=False,
        )

        if projection_type == "angular":
            theta_proj = np.linspace(np.pi / 10, np.pi - np.pi / 10, 2)
            phi_proj = np.linspace(np.pi / 10, np.pi - np.pi / 10, 3)
            monitor_far = td.FieldProjectionAngleMonitor(
                center=monitor.center,
                size=monitor.size,
                freqs=monitor.freqs,
                phi=tuple(phi_proj),
                theta=tuple(theta_proj),
                proj_distance=r_proj,
                far_field_approx=far_field_approx,
                name="far_field",
            )
        elif projection_type == "cartesian":
            x_proj = np.linspace(-10, 10, 2)
            y_proj = np.linspace(-10, 10, 3)
            monitor_far = td.FieldProjectionCartesianMonitor(
                center=monitor.center,
                size=monitor.size,
                freqs=monitor.freqs,
                x=x_proj,
                y=y_proj,
                proj_axis=1,
                proj_distance=r_proj,
                far_field_approx=far_field_approx,
                name="far_field",
            )

        sim = SIM_BASE.updated_copy(monitors=[monitor])

        if sim_2d and IS_3D:
            sim = sim.updated_copy(size=(0, *sim.size[1:]))

        return sim, monitor_far

    @staticmethod
    def objective(sim_data, monitor_far):
        projector = td.FieldProjector.from_near_field_monitors(
            sim_data=sim_data,
            near_monitors=[sim_data.simulation.monitors[0]],
            normal_dirs=["+"],
        )

        projected_fields = projector.project_fields(monitor_far)

        return projected_fields.power.sum().item()

    def test_field_projection_grad_prop(
        self, use_emulated_run, far_field_approx, projection_type, sim_2d
    ):
        """Tests whether field projection gradients are propagated through simulation.
        x0 <-> structures <-> sim <-> run <-> fields <-> projection <-> objval
        Does _not_ test gradient accuracy!
        """
        sim_base, monitor_far = self.setup(far_field_approx, projection_type, sim_2d)

        def objective(args):
            structures_traced_dict = make_structures(args)
            structures = list(SIM_BASE.structures)
            for structure_key in structure_keys_:
                structures.append(structures_traced_dict[structure_key])

            sim = sim_base.updated_copy(structures=structures)
            sim_data = run(sim, task_name="field_projection_test")

            return self.objective(sim_data, monitor_far)

        grads = ag.grad(objective)(params0)
        assert np.linalg.norm(grads) > 0

    def test_field_projection_grads(
        self, use_emulated_run, far_field_approx, projection_type, sim_2d
    ):
        """Tests projection gradient accuracy w.r.t. fields.
        fields <-> projection <-> objval
        """
        sim_base, monitor_far = self.setup(far_field_approx, projection_type, sim_2d)

        def objective(x0):
            sim_data = run_emulated(sim_base, task_name="field_projection_test", x0=x0)
            return self.objective(sim_data, monitor_far)

        check_grads(objective, modes=["rev"], order=1)(1.0)


def test_autograd_deepcopy():
    """make sure deepcopy works as expected in autograd."""

    def post(x, y):
        return 3 * x + y

    def f1(x):
        y = copy.deepcopy(x)
        return post(x, y)

    def f2(x):
        y = copy.copy(x)
        return post(x, y)

    def f3(x):
        y = x
        return post(x, y)

    x0 = 12.0

    val1, grad1 = ag.value_and_grad(f1)(x0)
    val2, grad2 = ag.value_and_grad(f2)(x0)
    val3, grad3 = ag.value_and_grad(f3)(x0)

    assert val1 == val2 == val3
    assert grad1 == grad2 == grad3


def test_pole_residue(monkeypatch):
    """Test that computed pole residue derivatives match."""

    def J(eps):
        return abs(eps)

    freq = 3e8

    eps_inf = 2.0
    p = td.C_0 * (1 + 1j)
    poles = [(-p, p), (-2 * p, 2 * p)]
    pr = td.PoleResidue(eps_inf=2.0, poles=poles)
    eps0 = pr.eps_model(freq)

    dJ_deps = ag.holomorphic_grad(J)(eps0)

    monkeypatch.setattr(
        td.PoleResidue, "derivative_eps_complex_volume", lambda self, E_der_map, bounds: dJ_deps
    )

    import importlib

    importlib.reload(td)

    poles = [(-p, p), (-2 * p, 2 * p)]
    pr = td.PoleResidue(eps_inf=2.0, poles=poles)
    field_paths = [("eps_inf",)]
    for i in range(len(poles)):
        for j in range(2):
            field_paths.append(("poles", i, j))

    info = DerivativeInfo(
        paths=field_paths,
        E_der_map={},
        D_der_map={},
        E_fwd={},
        D_fwd={},
        E_adj={},
        D_adj={},
        eps_data={},
        eps_in=2.0,
        eps_out=1.0,
        frequency=freq,
        bounds=((-1, -1, -1), (1, 1, 1)),
    )

    grads_computed = pr.compute_derivatives(derivative_info=info)

    def f(eps_inf, poles):
        eps = td.PoleResidue._eps_model(eps_inf, poles, freq)
        return J(eps)

    gfn = ag.holomorphic_grad(f, argnum=(0, 1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        grad_eps_inf, grad_poles = gfn(eps_inf, poles)

    assert np.isclose(grads_computed[("eps_inf",)], grad_eps_inf)

    for i in range(len(poles)):
        for j in range(2):
            field_path = ("poles", i, j)
            assert np.isclose(grads_computed[field_path], grad_poles[i][j])


def test_custom_pole_residue(monkeypatch):
    """Test that computed pole residue derivatives match."""

    nx, ny, nz = shape = (4, 5, 6)
    values = np.random.random((nx, ny, nz)) * (2 + 2j) * td.C_0

    nx, ny, nz = values.shape
    x = np.linspace(-0.5, 0.5, nx)
    y = np.linspace(-0.5, 0.5, ny)
    z = np.linspace(-0.5, 0.5, nz)
    coords = dict(x=x, y=y, z=z)

    eps_inf = td.SpatialDataArray(anp.real(values), coords=coords)
    a1 = td.SpatialDataArray(-values, coords=coords)
    c1 = td.SpatialDataArray(values, coords=coords)
    a2 = td.SpatialDataArray(-values, coords=coords)
    c2 = td.SpatialDataArray(values, coords=coords)
    poles = [(a1, c1), (a2, c2)]
    custom_med_pole_res = td.CustomPoleResidue(eps_inf=eps_inf, poles=poles)

    def J(eps):
        return anp.sum(abs(eps))

    freq = 3e8
    pr = td.CustomPoleResidue(eps_inf=eps_inf, poles=poles)
    eps0 = pr.eps_model(freq)

    dJ_deps = ag.holomorphic_grad(J)(eps0)

    monkeypatch.setattr(
        td.CustomPoleResidue,
        "_derivative_field_cmp",
        lambda self, E_der_map, eps_data, dim: dJ_deps,
    )

    import importlib

    importlib.reload(td)

    pr = td.CustomPoleResidue(eps_inf=eps_inf, poles=poles)
    field_paths = [("eps_inf",)]
    for i in range(len(poles)):
        for j in range(2):
            field_paths.append(("poles", i, j))

    info = DerivativeInfo(
        paths=field_paths,
        E_der_map={},
        D_der_map={},
        E_fwd={},
        D_fwd={},
        E_adj={},
        D_adj={},
        eps_data={},
        eps_in=2.0,
        eps_out=1.0,
        frequency=freq,
        bounds=((-1, -1, -1), (1, 1, 1)),
    )

    grads_computed = pr.compute_derivatives(derivative_info=info)

    poles_complex = [
        (np.array(a.values, dtype=complex), np.array(c.values, dtype=complex)) for a, c in poles
    ]
    poles_complex = np.stack(poles_complex, axis=0)

    def f(eps_inf, poles):
        eps = td.CustomPoleResidue._eps_model(eps_inf, poles, freq)
        return J(eps)

    gfn = ag.holomorphic_grad(f, argnum=(0, 1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        grad_eps_inf, grad_poles = gfn(eps_inf.values, poles_complex)

    assert np.allclose(grads_computed[("eps_inf",)], grad_eps_inf)

    for i in range(len(poles)):
        for j in range(2):
            field_path = ("poles", i, j)
            assert np.allclose(grads_computed[field_path], grad_poles[i][j])


# @pytest.mark.timeout(18.0)
def _test_many_structures():
    """Test that a metalens-like simulation with many structures can be initialized fast enough."""

    with cProfile.Profile() as pr:
        import time

        t = time.time()

        N_length = 200
        Nx, Ny = N_length, N_length
        sim_size = [Nx, Ny, 5]

        def f(x):
            monitor, postprocess = make_monitors()["field_point"]
            monitor = monitor.updated_copy(center=(0, 0, 0))

            geoms = []
            for ix in range(Nx):
                for iy in range(Ny):
                    ix = ix + x
                    iy = iy + x
                    verts = ((ix, iy), (ix + 0.5, iy), (ix + 0.5, iy + 0.5), (ix, iy + 0.5))
                    geom = td.PolySlab(slab_bounds=(0, 1), vertices=verts)
                    geoms.append(geom)

            metalens = td.Structure(
                geometry=td.GeometryGroup(geometries=geoms),
                medium=td.material_library["Si3N4"]["Horiba"],
            )

            src = td.PlaneWave(
                source_time=td.GaussianPulse(freq0=2.5e14, fwidth=1e13),
                center=(0, 0, -1),
                size=(td.inf, td.inf, 0),
                direction="+",
            )

            sim = td.Simulation(
                size=sim_size,
                structures=[metalens],
                sources=[src],
                monitors=[monitor],
                run_time=1e-12,
            )

            data = run_emulated(sim, task_name="test")
            return postprocess(data, data[monitor.name])

        x0 = 0.0
        ag.grad(f)(x0)

        t2 = time.time() - t
        pr.print_stats(sort="cumtime")
        pr.dump_stats("sim_test.prof")
        print(f"structures took {t2} seconds")


""" times (tyler's system)
* original : 35 sec
* no copy : 16 sec
* no to_static(): 13 sec
"""

FREQ1 = FREQ0 * 1.6

mnt_single = td.ModeMonitor(
    size=(2, 2, 0),
    center=(0, 0, LZ / 2 - WVL),
    mode_spec=td.ModeSpec(num_modes=2),
    freqs=[FREQ0],
    name="single",
)

mnt_multi = td.ModeMonitor(
    size=(2, 2, 0),
    center=(0, 0, LZ / 2 - WVL),
    mode_spec=td.ModeSpec(num_modes=2),
    freqs=[FREQ0, FREQ1],
    name="multi",
)


def make_objective(postprocess_fn: typing.Callable, structure_key: str) -> typing.Callable:
    def objective(params):
        structure_traced = make_structures(params)[structure_key]
        sim = SIM_BASE.updated_copy(
            structures=[structure_traced],
            monitors=list(SIM_BASE.monitors) + [mnt_single, mnt_multi],
        )
        data = run(sim, task_name="multifreq_test")
        return postprocess_fn(data)

    return objective


def get_amps(sim_data: td.SimulationData, mnt_name: str) -> xr.DataArray:
    return sim_data[mnt_name].amps


def power(amps: xr.DataArray) -> float:
    """Reduce a selected DataArray into just a float for objective function."""
    return anp.sum(anp.abs(amps.values) ** 2)


def postprocess_0_src(sim_data: td.SimulationData) -> float:
    """Postprocess function that should return 0 adjoint sources."""
    return 0.0


def compute_grad(postprocess_fn: typing.Callable, structure_key: str) -> typing.Callable:
    objective = make_objective(postprocess_fn, structure_key=structure_key)
    params = params0 + 1.0  # +1 is to avoid a warning in size_element with value 0
    return ag.grad(objective)(params)


def check_1_src_single(log_capture, structure_key):
    def postprocess(sim_data: td.SimulationData) -> float:
        """Postprocess function that should return 1 adjoint sources."""
        amps = get_amps(sim_data, "single").sel(mode_index=0, direction="+")
        return power(amps)

    return postprocess


def check_2_src_single(log_capture, structure_key):
    def postprocess(sim_data: td.SimulationData) -> float:
        """Postprocess function that should return 2 different adjoint sources."""
        amps = get_amps(sim_data, "single").sel(mode_index=0)
        return power(amps)

    return postprocess


def check_1_src_multi(log_capture, structure_key):
    def postprocess(sim_data: td.SimulationData) -> float:
        """Postprocess function that should return 1 adjoint sources."""
        amps = get_amps(sim_data, "multi").sel(mode_index=0, direction="+", f=FREQ0)
        return power(amps)

    return postprocess


def check_2_src_multi(log_capture, structure_key):
    def postprocess(sim_data: td.SimulationData) -> float:
        """Postprocess function that should return 2 different adjoint sources."""
        amps = get_amps(sim_data, "multi").sel(mode_index=0, f=FREQ1)
        return power(amps)

    return postprocess


def check_2_src_both(log_capture, structure_key):
    def postprocess(sim_data: td.SimulationData) -> float:
        """Postprocess function that should return 2 different adjoint sources."""
        amps_single = get_amps(sim_data, "single").sel(mode_index=0, direction="+")
        amps_multi = get_amps(sim_data, "multi").sel(mode_index=0, direction="+", f=FREQ0)
        return power(amps_single) + power(amps_multi)

    return postprocess


def check_1_multisrc(log_capture, structure_key):
    def postprocess(sim_data: td.SimulationData) -> float:
        """Postprocess function that should raise ValueError because diff sources, diff freqs."""
        amps_single = get_amps(sim_data, "single").sel(mode_index=0, direction="+")
        amps_multi = get_amps(sim_data, "multi").sel(mode_index=0, direction="+", f=FREQ1)
        return power(amps_single) + power(amps_multi)

    return postprocess


def check_2_multisrc(log_capture, structure_key):
    def postprocess(sim_data: td.SimulationData) -> float:
        """Postprocess function that should raise ValueError because diff sources, diff freqs."""
        amps_single = get_amps(sim_data, "single").sel(mode_index=0, direction="+")
        amps_multi = get_amps(sim_data, "multi").sel(mode_index=0, direction="+")
        return power(amps_single) + power(amps_multi)

    return postprocess


def check_1_src_broadband(log_capture, structure_key):
    def postprocess(sim_data: td.SimulationData) -> float:
        """Postprocess function that should return 1 broadband adjoint sources with many freqs."""
        amps = get_amps(sim_data, "multi").sel(mode_index=0, direction="+")
        return power(amps)

    return postprocess


MULT_FREQ_TEST_CASES = dict(
    src_1_freq_1=check_1_src_single,
    src_2_freq_1=check_2_src_single,
    src_1_freq_2=check_1_src_multi,
    src_2_freq_1_mon_1=check_1_src_multi,
    src_2_freq_1_mon_2=check_2_src_both,
    src_2_freq_2_mon_1=check_1_multisrc,
    src_2_freq_2_mon_2=check_2_multisrc,
    src_1_freq_2_broadband=check_1_src_broadband,
)

checks = list(MULT_FREQ_TEST_CASES.items())


@pytest.mark.parametrize("label, check_fn", checks)
@pytest.mark.parametrize("structure_key", ("custom_med",))
def test_multi_freq_edge_cases(
    log_capture, use_emulated_run, structure_key, label, check_fn, monkeypatch
):
    # test multi-frequency adjoint handling

    import tidy3d.components.data.sim_data as sd

    monkeypatch.setattr(sd, "RESIDUAL_CUTOFF_ADJOINT", 1)
    reload(td)

    postprocess_fn = check_fn(structure_key=structure_key, log_capture=log_capture)

    def objective(params):
        structure_traced = make_structures(params)[structure_key]
        sim = SIM_BASE.updated_copy(
            structures=[structure_traced],
            monitors=list(SIM_BASE.monitors) + [mnt_single, mnt_multi],
        )
        data = run(sim, task_name="multifreq_test")
        return postprocess_fn(data)

    if label == "src_2_freq_2_mon_2":
        with pytest.raises(NotImplementedError):
            g = ag.grad(objective)(params0)
    else:
        g = ag.grad(objective)(params0)
        print(g)


@pytest.mark.parametrize("structure_key", structure_keys_)
def test_multi_frequency_equivalence(use_emulated_run, structure_key):
    """Test an objective function through tidy3d autograd."""

    def objective_indi(params, structure_key) -> float:
        power_sum = 0.0

        for f in mnt_multi.freqs:
            structure_traced = make_structures(params)[structure_key]
            sim = SIM_BASE.updated_copy(
                structures=[structure_traced],
                monitors=list(SIM_BASE.monitors) + [mnt_multi],
            )

            sim_data = web.run(sim, task_name="multifreq_test")
            amps_i = get_amps(sim_data, "multi").sel(mode_index=0, direction="+", f=f)
            power_i = power(amps_i)
            power_sum = power_sum + power_i

        return power_sum

    def objective_multi(params, structure_key) -> float:
        structure_traced = make_structures(params)[structure_key]
        sim = SIM_BASE.updated_copy(
            structures=[structure_traced],
            monitors=list(SIM_BASE.monitors) + [mnt_multi],
        )
        sim_data = web.run(sim, task_name="multifreq_test")
        amps = get_amps(sim_data, "multi").sel(mode_index=0, direction="+")
        return power(amps)

    params0_ = params0 + 1.0

    # J_indi = objective_indi(params0_, structure_key)
    # J_multi = objective_multi(params0_, structure_key)

    # np.testing.assert_allclose(J_indi, J_multi)

    grad_indi = ag.grad(objective_indi)(params0_, structure_key=structure_key)
    grad_multi = ag.grad(objective_multi)(params0_, structure_key=structure_key)

    assert not np.any(np.isclose(grad_indi, 0))
    assert not np.any(np.isclose(grad_multi, 0))


def test_error_flux(use_emulated_run, log_capture):
    """Make sure proper error raised if differentiating w.r.t. FluxData."""

    def objective(params):
        structure_traced = make_structures(params)["medium"]
        sim = SIM_BASE.updated_copy(
            structures=[structure_traced],
            monitors=[td.FluxMonitor(size=(1, 1, 0), center=(0, 0, 0), freqs=[FREQ0], name="flux")],
        )
        data = run(sim, task_name="flux_error")
        return anp.sum(data["flux"].flux.values)

    with pytest.raises(
        NotImplementedError, match="Could not formulate adjoint source for 'FluxMonitor' output"
    ):
        g = ag.grad(objective)(params0)


def test_extraneous_field(use_emulated_run, log_capture):
    """Make sure this doesnt fail."""

    def objective(params):
        structure_traced = make_structures(params)["medium"]
        sim = SIM_BASE.updated_copy(
            structures=[structure_traced],
            monitors=[
                SIM_BASE.monitors[0],
                td.ModeMonitor(
                    size=(1, 1, 0),
                    center=(0, 0, 0),
                    mode_spec=td.ModeSpec(),
                    freqs=[FREQ0 * 0.9, FREQ0 * 1.1],
                    name="mode",
                ),
            ],
        )
        data = run(sim, task_name="extra_field")
        amp = data["mode"].amps.sel(direction="+", f=FREQ0 * 0.9, mode_index=0).values
        return abs(amp.item()) ** 2

    g = ag.grad(objective)(params0)


class TestTidyArrayBox:
    def test_is_tidy_box(self):
        da = DataArray(tracer_arr, dims=map(str, range(tracer_arr.ndim)))
        assert is_tidy_box(da.data)

    def test_real(self):
        npt.assert_allclose(tracer_arr.real._value, tracer_arr._value.real)

    def test_imag(self):
        npt.assert_allclose(tracer_arr.imag._value, tracer_arr._value.imag)

    def test_conj(self):
        npt.assert_allclose(tracer_arr.conj()._value, tracer_arr._value.conj())

    def test_item(self):
        assert tracer_arr.item() == tracer_arr._value.item()


class TestDataArrayGrads:
    @pytest.mark.parametrize("attr", ["real", "imag", "conj"])
    def test_custom_methods_grads(self, attr):
        """Test grads of TidyArrayBox methods implemented in autograd/boxes.py"""

        def objective(x, attr):
            da = DataArray(x, dims=map(str, range(x.ndim)))
            attr_value = getattr(da, attr)
            val = attr_value() if callable(attr_value) else attr_value
            return val.item()

        x = np.array([1.0])
        check_grads(objective, modes=["fwd", "rev"], order=2)(x, attr)

    def test_multiply_at_grads(self, rng):
        """Test grads of DataArray.multiply_at method"""

        def objective(a, b):
            coords = {str(i): np.arange(a.shape[i]) for i in range(a.ndim)}
            da = DataArray(a, coords=coords, dims=map(str, range(a.ndim)))
            da_mult = da.multiply_at(b, "0", [0, 1]) ** 2
            return np.sum(da_mult).item()

        a = rng.uniform(-1, 1, (3, 3))
        b = 1.0
        check_grads(lambda x: objective(x, b), modes=["fwd", "rev"], order=1)(a)
        check_grads(lambda x: objective(a, x), modes=["fwd", "rev"], order=1)(b)
