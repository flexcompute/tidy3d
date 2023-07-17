"""Tests adjoint plugin."""

from typing import Callable, Tuple, Dict

import pytest
import pydantic
import jax.numpy as jnp
import numpy as np
from jax import grad, custom_vjp
import jax
from numpy.random import random
import time
import matplotlib.pylab as plt
import h5py

import tidy3d as td
from typing import Tuple, Any, List

from tidy3d.exceptions import DataError, Tidy3dKeyError, AdjointError
from tidy3d.plugins.adjoint.components.base import JaxObject
from tidy3d.plugins.adjoint.components.geometry import JaxBox, JaxPolySlab, MAX_NUM_VERTICES
from tidy3d.plugins.adjoint.components.geometry import JaxGeometryGroup
from tidy3d.plugins.adjoint.components.medium import JaxMedium, JaxAnisotropicMedium
from tidy3d.plugins.adjoint.components.medium import JaxCustomMedium, MAX_NUM_CELLS_CUSTOM_MEDIUM
from tidy3d.plugins.adjoint.components.structure import JaxStructure
from tidy3d.plugins.adjoint.components.simulation import JaxSimulation, JaxInfo
from tidy3d.plugins.adjoint.components.simulation import MAX_NUM_INPUT_STRUCTURES
from tidy3d.plugins.adjoint.components.data.sim_data import JaxSimulationData
from tidy3d.plugins.adjoint.components.data.monitor_data import JaxModeData, JaxDiffractionData
from tidy3d.plugins.adjoint.components.data.data_array import JaxDataArray, JAX_DATA_ARRAY_TAG
from tidy3d.plugins.adjoint.components.data.dataset import JaxPermittivityDataset
from tidy3d.plugins.adjoint.web import run, run_async
from tidy3d.plugins.adjoint.web import run_local, run_async_local
from tidy3d.plugins.adjoint.components.data.data_array import VALUE_FILTER_THRESHOLD
from tidy3d.plugins.adjoint.utils.penalty import RadiusPenalty
from tidy3d.plugins.adjoint.utils.filter import ConicFilter, BinaryProjector
from tidy3d.web.container import BatchData
from tidy3d.web import run as run_regular

from ..utils import run_emulated, assert_log_level, log_capture, run_async_emulated
from ..utils import SIM_DATA_PATH, SIM_FULL, TMP_DIR

FWD_SIM_DATA_FILE = TMP_DIR + "adjoint_grad_data_fwd.hdf5"
SIM_VJP_FILE = TMP_DIR + "adjoint_sim_vjp_file.hdf5"
RUN_PATH = TMP_DIR + "simulation.hdf5"
NUM_PROC_PARALLEL = 2

EPS = 2.0
SIZE = (1.0, 2.0, 3.0)
CENTER = (2.0, -1.0, 1.0)
VERTICES = ((-1.0, -1.0), (0.0, 0.0), (-1.0, 0.0))
POLYSLAB_AXIS = 2
FREQ0 = 2e14
BASE_EPS_VAL = 2.0

# name of the output monitor used in tests
MNT_NAME = "mode"

# Emulated forward and backward run functions
def run_emulated_fwd(
    simulation: td.Simulation,
    jax_info: JaxInfo,
    task_name: str,
    folder_name: str,
    path: str,
    callback_url: str,
    verbose: bool,
):
    """Runs the forward simulation on our servers, stores the gradient data for later."""

    # Simulation data with both original and gradient data
    sim_data = run_emulated(
        simulation=simulation,
        task_name=str(task_name),
        path=path,
    )

    # simulation data (without gradient data), written to the path file
    sim_data_orig, sim_data_store = JaxSimulationData.split_fwd_sim_data(
        sim_data=sim_data, jax_info=jax_info
    )

    # Test file IO
    sim_data_orig.to_file(path)
    sim_data_orig = td.SimulationData.from_file(path)

    # gradient data stored for later use
    jax_sim_data_store = JaxSimulationData.from_sim_data(sim_data_store, jax_info)
    jax_sim_data_store.to_file(FWD_SIM_DATA_FILE)

    task_id = "test"
    return sim_data_orig, task_id


def run_emulated_bwd(
    sim_adj: td.Simulation,
    jax_info_adj: JaxInfo,
    fwd_task_id: str,
    task_name: str,
    folder_name: str,
    callback_url: str,
    verbose: bool,
    num_proc: int = NUM_PROC_PARALLEL,
) -> JaxSimulation:
    """Runs adjoint simulation on our servers, grabs the gradient data from fwd for processing."""

    # Forward data
    sim_data_fwd = JaxSimulationData.from_file(FWD_SIM_DATA_FILE)
    grad_data_fwd = sim_data_fwd.grad_data_symmetry
    grad_eps_data_fwd = sim_data_fwd.grad_eps_data_symmetry

    # Adjoint data
    sim_data_adj = run_emulated(
        simulation=sim_adj,
        task_name=str(task_name),
        path=RUN_PATH,
    )

    jax_sim_data_adj = JaxSimulationData.from_sim_data(sim_data_adj, jax_info_adj)
    grad_data_adj = jax_sim_data_adj.grad_data_symmetry

    # get gradient and insert into the resulting simulation structure medium
    sim_vjp = jax_sim_data_adj.simulation.store_vjp(
        grad_data_fwd, grad_data_adj, grad_eps_data_fwd, num_proc=num_proc
    )

    # write VJP sim to and from file to emulate webapi download and loading
    sim_vjp.to_file(SIM_VJP_FILE)
    sim_vjp = JaxSimulation.from_file(SIM_VJP_FILE)

    return sim_vjp


# Emulated forward and backward run functions
def run_async_emulated_fwd(
    simulations: Tuple[td.Simulation, ...],
    jax_infos: Tuple[JaxInfo, ...],
    folder_name: str,
    path_dir: str,
    callback_url: str,
    verbose: bool,
) -> Tuple[BatchData, Dict[str, str]]:
    """Runs the forward simulation on our servers, stores the gradient data for later."""

    sim_datas_orig = {}
    task_ids = []

    for i, (sim, jax_info) in enumerate(zip(simulations, jax_infos)):
        sim_data_orig, task_id = run_emulated_fwd(
            simulation=sim,
            jax_info=jax_info,
            task_name=str(i),
            folder_name=folder_name,
            path=path_dir + str(i) + ".hdf5",
            callback_url=callback_url,
            verbose=verbose,
        )
        task_ids.append(task_id)
        sim_datas_orig[str(i)] = sim_data_orig

    return sim_datas_orig, task_ids


def run_async_emulated_bwd(
    simulations: Tuple[td.Simulation, ...],
    jax_infos: Tuple[JaxInfo, ...],
    folder_name: str,
    path_dir: str,
    callback_url: str,
    verbose: bool,
    parent_tasks: List[List[str]],
) -> List[JaxSimulation]:
    """Runs adjoint simulation on our servers, grabs the gradient data from fwd for processing."""

    sim_vjps_orig = []

    for i, (sim, jax_info, parent_tasks_i) in enumerate(zip(simulations, jax_infos, parent_tasks)):
        sim_vjp = run_emulated_bwd(
            sim_adj=sim,
            jax_info_adj=jax_info,
            fwd_task_id="test",
            task_name=str(i),
            folder_name=folder_name,
            callback_url=callback_url,
            verbose=verbose,
            num_proc=NUM_PROC_PARALLEL,
        )
        sim_vjps_orig.append(sim_vjp)

    return sim_vjps_orig


def make_sim(
    permittivity: float, size: Tuple[float, float, float], vertices: tuple, base_eps_val: float
) -> JaxSimulation:
    """Construt a simulation out of some input parameters."""

    box = td.Box(size=(0.2, 0.2, 0.2), center=(5, 0, 2))
    med = td.Medium(permittivity=2.0)
    extraneous_structure = td.Structure(geometry=box, medium=med)

    # NOTE: Any new input structures should be added below as they are made

    # JaxBox
    jax_box1 = JaxBox(size=size, center=(1, 0, 2))
    jax_med1 = JaxMedium(permittivity=permittivity, conductivity=permittivity * 0.1)
    jax_struct1 = JaxStructure(geometry=jax_box1, medium=jax_med1)

    jax_box2 = JaxBox(size=size, center=(-1, 0, -3))
    jax_med2 = JaxAnisotropicMedium(
        xx=JaxMedium(permittivity=permittivity),
        yy=JaxMedium(permittivity=permittivity + 2),
        zz=JaxMedium(permittivity=permittivity * 2),
    )
    jax_struct2 = JaxStructure(geometry=jax_box2, medium=jax_med2)

    jax_polyslab1 = JaxPolySlab(axis=POLYSLAB_AXIS, vertices=vertices, slab_bounds=(-1, 1))
    jax_struct3 = JaxStructure(geometry=jax_polyslab1, medium=jax_med1)

    # custom medium
    Nx, Ny, Nz = 10, 1, 10
    (xmin, ymin, zmin), (xmax, ymax, zmax) = jax_box1.bounds
    coords = dict(
        x=np.linspace(xmin, xmax, Nx).tolist(),
        y=np.linspace(ymin, ymax, Ny).tolist(),
        z=np.linspace(zmin, zmax, Nz).tolist(),
        f=[FREQ0],
    )

    jax_box_custom = JaxBox(size=size, center=(1, 0, 2))
    values = base_eps_val + np.random.random((Nx, Ny, Nz, 1))

    # adding this line breaks things without enforcing that the vjp for custom medium is complex
    values = (1 + 1j) * values
    values = values + (1 + 1j) * values / 0.5

    eps_ii = JaxDataArray(values=values, coords=coords)
    field_components = {f"eps_{dim}{dim}": eps_ii for dim in "xyz"}
    jax_eps_dataset = JaxPermittivityDataset(**field_components)
    jax_med_custom = JaxCustomMedium(eps_dataset=jax_eps_dataset)
    jax_struct_custom = JaxStructure(geometry=jax_box_custom, medium=jax_med_custom)

    jax_geo_group = JaxGeometryGroup(geometries=[jax_polyslab1, jax_polyslab1])
    jax_struct_group = JaxStructure(geometry=jax_geo_group, medium=jax_med1)
    # TODO: Add new geometries as they are created.

    # NOTE: Any new output monitors should be added below as they are made

    # ModeMonitors
    output_mnt1 = td.ModeMonitor(
        size=(10, 10, 0),
        mode_spec=td.ModeSpec(num_modes=3),
        freqs=[FREQ0],
        name=MNT_NAME + "1",
    )

    # DiffractionMonitor
    output_mnt2 = td.DiffractionMonitor(
        center=(0, 0, 4),
        size=(td.inf, td.inf, 0),
        normal_dir="+",
        freqs=[FREQ0],
        name=MNT_NAME + "2",
    )

    output_mnt3 = td.FieldMonitor(
        size=(2, 0, 2),
        freqs=[FREQ0],
        name=MNT_NAME + "3",
    )

    output_mnt4 = td.FieldMonitor(
        size=(0, 0, 0),
        freqs=[FREQ0],
        name=MNT_NAME + "4",
    )

    extraneous_field_monitor = td.FieldMonitor(
        size=(10, 10, 0),
        freqs=[1e14, 2e14],
        name="field",
    )

    sim = JaxSimulation(
        size=(10, 10, 10),
        run_time=1e-12,
        grid_spec=td.GridSpec(wavelength=4.0),
        monitors=(extraneous_field_monitor,),
        structures=(extraneous_structure,),
        input_structures=(
            jax_struct1,
            jax_struct2,
            jax_struct_custom,
            jax_struct3,
            jax_struct_group,
        ),
        output_monitors=(output_mnt1, output_mnt2, output_mnt3, output_mnt4),
        boundary_spec=td.BoundarySpec.pml(x=False, y=False, z=False),
        symmetry=(0, 1, -1),
    )

    return sim


def objective(amp: complex) -> float:
    """Objective function as a function of the complex amplitude."""
    return abs(amp) ** 2


def test_run_flux(use_emulated_run):
    td.config.logging_level = "ERROR"

    def make_components(eps, size, vertices, base_eps_val):
        sim = make_sim(permittivity=eps, size=size, vertices=vertices, base_eps_val=base_eps_val)
        # sim = sim.to_simulation()[0]
        td.config.logging_level = "WARNING"
        sim = sim.updated_copy(
            sources=[
                td.PointDipole(
                    center=(0, 0, 0),
                    polarization="Ex",
                    source_time=td.GaussianPulse(freq0=2e14, fwidth=1e15),
                )
            ]
        )
        sim_data = run_local(sim, task_name="test", path=RUN_PATH)
        mnt_data = sim_data[MNT_NAME + "3"]
        flat_components = {}
        for key, fld in mnt_data.field_components.items():
            values = jnp.array(jax.lax.stop_gradient(fld.values))[:, 1, ...]
            values = values[:, None, ...]
            coords = dict(fld.coords).copy()
            coords["y"] = [0.0]
            if isinstance(fld, td.ScalarFieldDataArray):
                flat_components[key] = td.ScalarFieldDataArray(values, coords=coords)
            else:
                flat_components[key] = fld.updated_copy(values=values, coords=coords)
        return mnt_data.updated_copy(**flat_components)

    mnt_data = make_components(EPS, SIZE, VERTICES, BASE_EPS_VAL)

    # whether to run the flux pipeline through jax (True) or regular tidy3d (False)
    use_jax = True
    if not use_jax:

        td_field_components = {}
        for fld, jax_data_array in mnt_data.field_components.items():
            data_array = td.ScalarFieldDataArray(
                np.array(jax_data_array.values), coords=jax_data_array.coords
            )
            td_field_components[fld] = data_array

        mnt_data = td.FieldData(monitor=mnt_data.monitor, **td_field_components)

    def get_flux(x):

        fld_components = {}
        for fld, fld_component in mnt_data.field_components.items():
            new_values = x * fld_component.values
            if isinstance(fld_component, td.ScalarFieldDataArray):
                fld_data = td.ScalarFieldDataArray(new_values, coords=fld_component.coords)
            else:
                fld_data = fld_component.updated_copy(values=new_values)
            fld_components[fld] = fld_data

        mnt_data2 = mnt_data.updated_copy(**fld_components)

        return jnp.sum(mnt_data2.flux)

    f = get_flux(1.0)

    if use_jax:
        get_flux_grad = jax.grad(get_flux)
        g = get_flux_grad(1.0)


def extract_amp(sim_data: td.SimulationData) -> complex:
    """get the amplitude from a simulation data object."""

    ret_value = 0.0

    # ModeData
    mnt_name = MNT_NAME + "1"
    mnt_data = sim_data[mnt_name]
    amps = mnt_data.amps
    ret_value += amps.sel(direction="+", f=2e14, mode_index=0)
    ret_value += amps.isel(direction=0, f=0, mode_index=0)
    ret_value += amps.sel(direction="-", f=2e14, mode_index=1)
    ret_value += amps.sel(mode_index=1, f=2e14, direction="-")
    ret_value += amps.sel(direction="-", f=2e14).isel(mode_index=1)

    # DiffractionData
    mnt_name = MNT_NAME + "2"
    mnt_data = sim_data[mnt_name]
    ret_value += mnt_data.amps.sel(orders_x=0, orders_y=0, f=2e14, polarization="p")
    ret_value += mnt_data.amps.sel(orders_x=-1, orders_y=1, f=2e14, polarization="p")
    ret_value += mnt_data.amps.isel(orders_x=0, orders_y=1, f=0, polarization=0)
    ret_value += mnt_data.Er.isel(orders_x=0, orders_y=1, f=0)
    ret_value += mnt_data.power.sel(orders_x=-1, orders_y=1, f=2e14)

    # FieldData
    mnt_name = MNT_NAME + "3"
    mnt_data = sim_data[mnt_name]
    ret_value += jnp.sum(jnp.array(mnt_data.Ex.values))
    ret_value += jnp.sum(jnp.array(mnt_data.Ex.interp(z=0).values))

    # this should work when we figure out a jax version of xr.DataArray
    sim_data.get_intensity(mnt_name)
    # ret_value += jnp.sum(jnp.array(mnt_data.flux().values))

    # FieldData (dipole)
    mnt_name = MNT_NAME + "4"
    mnt_data = sim_data[mnt_name]
    ret_value += jnp.sum(jnp.array(mnt_data.Ex.values))

    return ret_value


@pytest.fixture
def use_emulated_run(monkeypatch):
    """If this fixture is used, the `tests.utils.run_emulated` function is used for simulation."""
    import tidy3d.plugins.adjoint.web as adjoint_web

    monkeypatch.setattr(adjoint_web, "tidy3d_run_fn", run_emulated)
    monkeypatch.setattr(adjoint_web, "webapi_run_adjoint_fwd", run_emulated_fwd)
    monkeypatch.setattr(adjoint_web, "webapi_run_adjoint_bwd", run_emulated_bwd)


@pytest.fixture
def use_emulated_run_async(monkeypatch):
    """If this fixture is used, the `tests.utils.run_emulated` function is used for simulation."""
    import tidy3d.plugins.adjoint.web as adjoint_web

    monkeypatch.setattr(adjoint_web, "tidy3d_run_async_fn", run_async_emulated)
    monkeypatch.setattr(adjoint_web, "webapi_run_async_adjoint_fwd", run_async_emulated_fwd)
    monkeypatch.setattr(adjoint_web, "webapi_run_async_adjoint_bwd", run_async_emulated_bwd)


@pytest.mark.parametrize("local", (True, False))
def test_adjoint_pipeline(local, use_emulated_run):
    """Test computing gradient using jax."""

    run_fn = run_local if local else run

    sim = make_sim(permittivity=EPS, size=SIZE, vertices=VERTICES, base_eps_val=BASE_EPS_VAL)
    sim_data = run_fn(sim, task_name="test", path=RUN_PATH)

    def f(permittivity, size, vertices, base_eps_val):
        sim = make_sim(
            permittivity=permittivity, size=size, vertices=vertices, base_eps_val=base_eps_val
        )
        sim_data = run_fn(sim, task_name="test", path=RUN_PATH)
        amp = extract_amp(sim_data)
        return objective(amp)

    grad_f = grad(f, argnums=(0, 1, 2, 3))
    df_deps, df_dsize, df_dvertices, d_eps_base = grad_f(EPS, SIZE, VERTICES, BASE_EPS_VAL)

    print("gradient: ", df_deps, df_dsize, df_dvertices, d_eps_base)


@pytest.mark.parametrize("local", (True, False))
def test_adjoint_pipeline_2d(local, use_emulated_run):

    run_fn = run_local if local else run

    sim = make_sim(permittivity=EPS, size=SIZE, vertices=VERTICES, base_eps_val=BASE_EPS_VAL)

    sim_size_2d = list(sim.size)
    sim_size_2d[1] = 0
    sim = sim.updated_copy(size=sim_size_2d)

    sim_data = run_fn(sim, task_name="test", path=RUN_PATH)

    def f(permittivity, size, vertices, base_eps_val):
        sim = make_sim(
            permittivity=permittivity, size=size, vertices=vertices, base_eps_val=base_eps_val
        )
        sim_size_2d = list(sim.size)
        sim_size_2d[1] = 0

        sim = sim.updated_copy(size=sim_size_2d)

        sim_data = run_fn(sim, task_name="test", path=RUN_PATH)
        amp = extract_amp(sim_data)
        return objective(amp)

    grad_f = grad(f, argnums=(0, 1, 2, 3))
    df_deps, df_dsize, df_dvertices, d_eps_base = grad_f(EPS, SIZE, VERTICES, BASE_EPS_VAL)


def test_adjoint_setup_fwd(use_emulated_run):
    """Test that the forward pass works as expected."""
    sim = make_sim(permittivity=EPS, size=SIZE, vertices=VERTICES, base_eps_val=BASE_EPS_VAL)
    sim_data_orig, (task_id_fwd) = run.fwd(
        simulation=sim,
        task_name="test",
        folder_name="default",
        path=RUN_PATH,
        callback_url=None,
        verbose=False,
    )


def _test_adjoint_setup_adj(use_emulated_run):
    """Test that the adjoint pass works as expected."""
    sim_orig = make_sim(permittivity=EPS, size=SIZE, vertices=VERTICES, base_eps_val=BASE_EPS_VAL)

    # call forward pass
    sim_data_fwd, (sim_data_fwd,) = run.fwd(
        simulation=sim_orig,
        task_name="test",
        folder_name="default",
        path=RUN_PATH,
        callback_url=None,
    )

    # create some contrived vjp sim_data to be able to call backward pass
    sim_data_vjp = sim_data_fwd.copy()
    output_data_vjp = []
    for mode_data in sim_data_vjp.output_data:
        new_values = 0 * np.array(mode_data.amps.values)
        new_values[0, 0, 0] = 1 + 1j
        amps_vjp = mode_data.amps.copy(update=dict(values=new_values.tolist()))
        mode_data_vjp = mode_data.copy(update=dict(amps=amps_vjp))
        output_data_vjp.append(mode_data_vjp)
    sim_data_vjp = sim_data_vjp.copy(update=dict(output_data=output_data_vjp))
    (sim_vjp,) = run.bwd(
        task_name="test",
        folder_name="default",
        path=RUN_PATH,
        callback_url=None,
        res=(sim_data_fwd,),
        sim_data_vjp=sim_data_vjp,
    )

    # check the lengths of various tuples are correct
    assert len(sim_vjp.monitors) == len(sim_orig.monitors)
    assert len(sim_vjp.structures) == len(sim_orig.structures)
    assert len(sim_vjp.input_structures) == len(sim_orig.input_structures)


# @pytest.mark.parametrize("add_grad_monitors", (True, False))
# def test_convert_tidy3d_to_jax(add_grad_monitors):
#     """test conversion of JaxSimulation to Simulation and SimulationData to JaxSimulationData."""
#     jax_sim = make_sim(permittivity=EPS, size=SIZE, vertices=VERTICES, base_eps_val=BASE_EPS_VAL)
#     if add_grad_monitors:
#         jax_sim = jax_sim.add_grad_monitors()
#     sim, jax_info = jax_sim.to_simulation()
#     assert type(sim) == td.Simulation
#     assert sim.type == "Simulation"
#     sim_data = run_emulated(sim)
#     jax_sim_data = JaxSimulationData.from_sim_data(sim_data, jax_info)
#     jax_sim2 = jax_sim_data.simulation
#     assert jax_sim_data.simulation == jax_sim


def test_multiple_freqs():
    """Test that sim validation fails when output monitors have multiple frequencies."""

    output_mnt = td.ModeMonitor(
        size=(10, 10, 0),
        mode_spec=td.ModeSpec(num_modes=3),
        freqs=[1e14, 2e14],
        name=MNT_NAME,
    )

    with pytest.raises(pydantic.ValidationError):
        sim = JaxSimulation(
            size=(10, 10, 10),
            run_time=1e-12,
            grid_spec=td.GridSpec(wavelength=1.0),
            monitors=(),
            structures=(),
            output_monitors=(output_mnt,),
            input_structures=(),
        )


def test_different_freqs():
    """Test that sim validation fails when output monitors have different frequencies."""

    output_mnt1 = td.ModeMonitor(
        size=(10, 10, 0),
        mode_spec=td.ModeSpec(num_modes=3),
        freqs=[1e14],
        name=MNT_NAME + "1",
    )
    output_mnt2 = td.ModeMonitor(
        size=(10, 10, 0),
        mode_spec=td.ModeSpec(num_modes=3),
        freqs=[2e14],
        name=MNT_NAME + "2",
    )
    with pytest.raises(pydantic.ValidationError):
        sim = JaxSimulation(
            size=(10, 10, 10),
            run_time=1e-12,
            grid_spec=td.GridSpec(wavelength=1.0),
            monitors=(),
            structures=(),
            output_monitors=(output_mnt1, output_mnt2),
            input_structures=(),
        )


def test_get_freq_adjoint():
    """Test that the adjoint frequency property works as expected."""

    sim = JaxSimulation(
        size=(10, 10, 10),
        run_time=1e-12,
        grid_spec=td.GridSpec(wavelength=1.0),
        monitors=(),
        structures=(),
        output_monitors=(),
        input_structures=(),
    )

    with pytest.raises(AdjointError):
        f = sim.freq_adjoint

    freq0 = 2e14
    output_mnt1 = td.ModeMonitor(
        size=(10, 10, 0),
        mode_spec=td.ModeSpec(num_modes=3),
        freqs=[freq0],
        name=MNT_NAME + "1",
    )
    output_mnt2 = td.ModeMonitor(
        size=(10, 10, 0),
        mode_spec=td.ModeSpec(num_modes=3),
        freqs=[freq0],
        name=MNT_NAME + "2",
    )
    sim = JaxSimulation(
        size=(10, 10, 10),
        run_time=1e-12,
        grid_spec=td.GridSpec(wavelength=1.0),
        monitors=(),
        structures=(),
        output_monitors=(output_mnt1, output_mnt2),
        input_structures=(),
    )
    assert sim.freq_adjoint == freq0


def test_get_fwidth_adjoint():
    """Test that the adjoint fwidth property works as expected."""

    from tidy3d.plugins.adjoint.components.simulation import FWIDTH_FACTOR

    freq0 = 2e14
    mnt = td.ModeMonitor(
        size=(10, 10, 0),
        mode_spec=td.ModeSpec(num_modes=3),
        freqs=[freq0],
        name=MNT_NAME + "1",
    )

    def make_sim(sources=(), fwidth_adjoint=None):
        """Make a sim with given sources and fwidth_adjoint specified."""
        return JaxSimulation(
            size=(10, 10, 10),
            run_time=1e-12,
            grid_spec=td.GridSpec(wavelength=1.0),
            monitors=(),
            structures=(),
            output_monitors=(mnt,),
            input_structures=(),
            sources=sources,
            fwidth_adjoint=fwidth_adjoint,
        )

    # no sources, use FWIDTH * freq0
    sim = make_sim(sources=(), fwidth_adjoint=None)
    assert np.isclose(sim._fwidth_adjoint, FWIDTH_FACTOR * freq0)

    # a few sources, use average of fwidths
    fwidths = [1e14, 2e14, 3e14, 4e14]
    src_times = [td.GaussianPulse(freq0=freq0, fwidth=fwidth) for fwidth in fwidths]
    srcs = [td.PointDipole(source_time=src_time, polarization="Ex") for src_time in src_times]
    sim = make_sim(sources=srcs, fwidth_adjoint=None)
    assert np.isclose(sim._fwidth_adjoint, np.mean(fwidths))

    # a few sources, with custom fwidth specified
    fwidth_custom = 3e13
    sim = make_sim(sources=srcs, fwidth_adjoint=fwidth_custom)
    assert np.isclose(sim._fwidth_adjoint, fwidth_custom)

    # no sources, custom fwidth specified
    sim = make_sim(sources=(), fwidth_adjoint=fwidth_custom)
    assert np.isclose(sim._fwidth_adjoint, fwidth_custom)


def test_jax_data_array():
    """Test mechanics of the JaxDataArray."""

    a = [1, 2, 3]
    b = [2, 3]
    c = [4]
    values = np.random.random((len(a), len(b), len(c)))
    coords = dict(a=a, b=b, c=c)

    # validate missing coord
    # with pytest.raises(AdjointError):
    # da = JaxDataArray(values=values, coords=dict(a=a, b=b))

    # validate coords in wrong order
    # with pytest.raises(AdjointError):
    # da = JaxDataArray(values=values, coords=dict(c=c, b=b, a=a))

    # creation
    da = JaxDataArray(values=values, coords=coords)
    _ = da.real
    _ = da.imag
    _ = da.as_list

    # isel multi
    z = da.isel(a=1, b=[0, 1], c=0)
    assert z.shape == (2,)

    # isel
    z = da.isel(a=1, b=1, c=0)
    z = da.isel(c=0, b=1, a=1)

    # sel
    z = da.sel(a=1, b=2, c=4)
    z = da.sel(c=4, b=2, a=1)

    # isel and sel
    z = da.sel(c=4, b=2).isel(a=0)
    z = da.isel(c=0, b=1).sel(a=1)

    # errors if coordinate not in data
    with pytest.raises(Tidy3dKeyError):
        da.sel(d=1)

    # errors if index out of range
    with pytest.raises(DataError):
        da.isel(c=1)
    with pytest.raises(DataError):
        da.isel(c=-1)
    with pytest.raises(DataError):
        da.sel(c=5)

    # not implemented
    # with pytest.raises(NotImplementedError):
    da.interp(b=2.5)

    assert np.isclose(da.interp(a=2, b=3, c=4), values[1, 1, 0])
    assert np.isclose(da.interp(a=1, b=2, c=4), values[0, 0, 0])

    with pytest.raises(Tidy3dKeyError):
        da.interp(d=3)

    da1d = JaxDataArray(values=[0.0, 1.0, 2.0, 3.0], coords=dict(x=[0, 1, 2, 3]))
    assert np.isclose(da1d.interp(x=0.5), 0.5)


def test_jax_sim_data(use_emulated_run):
    """Test mechanics of the JaxSimulationData."""

    sim = make_sim(permittivity=EPS, size=SIZE, vertices=VERTICES, base_eps_val=BASE_EPS_VAL)
    sim_data = run(sim, task_name="test", path=RUN_PATH)

    for i in range(len(sim.output_monitors)):
        mnt_name = MNT_NAME + str(i + 1)
        mnt_data_a = sim_data.output_data[i]
        mnt_data_b = sim_data.output_monitor_data[mnt_name]
        mnt_data_c = sim_data[mnt_name]


def test_intersect_structures(log_capture):
    """Test validators for structures touching and intersecting."""

    SIZE_X = 1.0
    OVERLAP = 1e-4

    def make_sim_intersect(spacing: float, is_vjp: bool = False) -> JaxSimulation:
        """Make a sim with two boxes spaced by some variable amount."""
        box1 = JaxBox(center=(-SIZE_X / 2 - spacing / 2, 0, 0), size=(SIZE_X, 1, 1))
        box2 = JaxBox(center=(+SIZE_X / 2 + spacing / 2, 0, 0), size=(SIZE_X, 1, 1))
        medium = JaxMedium(permittivity=2.0)
        struct1 = JaxStructure(geometry=box1, medium=medium)
        struct2 = JaxStructure(geometry=box2, medium=medium)
        src = td.PointDipole(
            source_time=td.GaussianPulse(freq0=2e14, fwidth=1e13),
            polarization="Ex",
        )
        return JaxSimulation(
            size=(2, 2, 2),
            input_structures=(struct1, struct2),
            grid_spec=td.GridSpec(wavelength=1.0),
            run_time=1e-12,
            sources=(src,),
            boundary_spec=td.BoundarySpec.pml(x=True, y=True, z=True),
        )

    # shouldnt error, boxes spaced enough
    sim = make_sim_intersect(spacing=+OVERLAP)

    # shouldnt error, just warn because of touching but not intersecting
    sim = make_sim_intersect(spacing=0.0)
    assert_log_level(log_capture, "WARNING")


def test_structure_overlaps():
    """Test weird overlap edge cases, eg with box out of bounds and 2D sim."""

    box = JaxBox(center=(0, 0, 0), size=(td.inf, 2, 1))
    medium = JaxMedium(permittivity=2.0)
    struct = JaxStructure(geometry=box, medium=medium)
    src = td.PointDipole(
        source_time=td.GaussianPulse(freq0=2e14, fwidth=1e13),
        polarization="Ex",
    )

    sim = JaxSimulation(
        size=(2, 0, 2),
        input_structures=(struct,),
        grid_spec=td.GridSpec(wavelength=1.0),
        run_time=1e-12,
        sources=(src,),
    )


def test_validate_subpixel():
    """Make sure errors if subpixel is off."""
    with pytest.raises(pydantic.ValidationError):
        sim = JaxSimulation(
            size=(10, 10, 10),
            run_time=1e-12,
            grid_spec=td.GridSpec(wavelength=1.0),
            subpixel=False,
        )


# def test_validate_3D_geometry():
#     """Make sure it errors if the size of a JaxBox is 1d or 2d."""

#     b = JaxBox(center=(0,0,0), size=(1,1,1))

#     with pytest.raises(AdjointError):
#         b = JaxBox(center=(0,0,0), size=(0,1,1))

#     with pytest.raises(AdjointError):
#         b = JaxBox(center=(0,0,0), size=(0,1,0))

#     p = JaxPolySlab(vertices=VERTICES, axis=2, slab_bounds=(0,1))

#     with pytest.raises(AdjointError):
#         p = JaxPolySlab(vertices=VERTICES, axis=2, slab_bounds=(0,0))


def test_plot_sims():
    """Make sure plotting functions without erroring."""

    sim = JaxSimulation(
        size=(10, 10, 10),
        run_time=1e-12,
        grid_spec=td.GridSpec(wavelength=1.0),
    )
    sim.plot(x=0)
    sim.plot_eps(x=0)


def test_flip_direction():
    """Make sure flip direction fails when direction happens to neither be '+' nor '-'"""
    with pytest.raises(AdjointError):
        JaxModeData.flip_direction("NOT+OR-")


def test_strict_types():
    """Test that things fail if you try to use just any object in a Jax component."""
    with pytest.raises(pydantic.ValidationError):
        b = JaxBox(size=(1, 1, [1, 2]), center=(0, 0, 0))


def _test_polyslab_box(use_emulated_run):
    """Make sure box made with polyslab gives equivalent gradients.
    Note: doesn't pass now since JaxBox samples the permittivity inside and outside the box,
    and a random permittivity data is created by the emulated run function. JaxPolySlab just
    uses the slab permittivity and the background simulation permittivity."""

    np.random.seed(0)

    def f(size, center, is_box=True):

        jax_med = JaxMedium(permittivity=2.0)
        POLYSLAB_AXIS = 2

        if is_box:

            size = list(size)
            size[POLYSLAB_AXIS] = jax.lax.stop_gradient(size[POLYSLAB_AXIS])
            center = list(center)
            center[POLYSLAB_AXIS] = jax.lax.stop_gradient(center[POLYSLAB_AXIS])

            # JaxBox
            jax_box = JaxBox(size=size, center=center)
            jax_struct = JaxStructure(geometry=jax_box, medium=jax_med)

        else:

            size_axis, (size_1, size_2) = JaxPolySlab.pop_axis(size, axis=POLYSLAB_AXIS)
            cent_axis, (cent_1, cent_2) = JaxPolySlab.pop_axis(center, axis=POLYSLAB_AXIS)

            pos_x2 = cent_1 + size_1 / 2.0
            pos_x1 = cent_1 - size_1 / 2.0
            pos_y1 = cent_2 - size_2 / 2.0
            pos_y2 = cent_2 + size_2 / 2.0

            vertices = ((pos_x1, pos_y1), (pos_x2, pos_y1), (pos_x2, pos_y2), (pos_x1, pos_y2))
            slab_bounds = (cent_axis - size_axis / 2, cent_axis + size_axis / 2)
            slab_bounds = tuple(jax.lax.stop_gradient(x) for x in slab_bounds)
            jax_polyslab = JaxPolySlab(
                vertices=vertices, axis=POLYSLAB_AXIS, slab_bounds=slab_bounds
            )
            jax_struct = JaxStructure(geometry=jax_polyslab, medium=jax_med)

        # ModeMonitors
        output_mnt1 = td.ModeMonitor(
            size=(td.inf, td.inf, 0),
            mode_spec=td.ModeSpec(num_modes=3),
            freqs=[2e14],
            name=MNT_NAME + "1",
        )

        # DiffractionMonitor
        output_mnt2 = td.DiffractionMonitor(
            center=(0, 0, 4),
            size=(td.inf, td.inf, 0),
            normal_dir="+",
            freqs=[2e14],
            name=MNT_NAME + "2",
        )

        sim = JaxSimulation(
            size=(10, 10, 10),
            run_time=1e-12,
            grid_spec=td.GridSpec(wavelength=1.0),
            boundary_spec=td.BoundarySpec.all_sides(boundary=td.Periodic()),
            output_monitors=(output_mnt1, output_mnt2),
            input_structures=(jax_struct,),
            sources=[
                td.PointDipole(
                    source_time=td.GaussianPulse(freq0=1e14, fwidth=1e14),
                    center=(0, 0, 0),
                    polarization="Ex",
                )
            ],
        )

        sim_data = run(sim, task_name="test")
        amp = extract_amp(sim_data)
        return objective(amp)

    f_b = lambda size, center: f(size, center, is_box=True)
    f_p = lambda size, center: f(size, center, is_box=False)

    g_b = grad(f_b, argnums=(0, 1))
    g_p = grad(f_p, argnums=(0, 1))

    gs_b, gc_b = g_b(SIZE, CENTER)
    gs_p, gc_p = g_p(SIZE, CENTER)

    gs_b, gc_b, gs_p, gc_p = map(np.array, (gs_b, gc_b, gs_p, gc_p))

    print("grad_size_box  = ", gs_b)
    print("grad_size_poly = ", gs_p)
    print("grad_cent_box  = ", gc_b)
    print("grad_cent_poly = ", gc_p)

    print(gs_b / (gs_p + 1e-12))
    print(gc_b / (gc_p + 1e-12))

    assert np.allclose(gs_b, gs_p), f"size gradients dont match, got {gs_b} and {gs_p}"
    assert np.allclose(gc_b, gc_p), f"center gradients dont match, got {gc_b} and {gc_p}"


@pytest.mark.parametrize("sim_size_axis", [0, 10])
def test_polyslab_2d(sim_size_axis, use_emulated_run):
    """Make sure box made with polyslab gives equivalent gradients (note, doesn't pass now)."""

    np.random.seed(0)

    def f(size, center):

        jax_med = JaxMedium(permittivity=2.0)
        POLYSLAB_AXIS = 2

        size_axis, (size_1, size_2) = JaxPolySlab.pop_axis(size, axis=POLYSLAB_AXIS)
        cent_axis, (cent_1, cent_2) = JaxPolySlab.pop_axis(center, axis=POLYSLAB_AXIS)

        pos_x2 = cent_1 + size_1 / 2.0
        pos_x1 = cent_1 - size_1 / 2.0
        pos_y1 = cent_2 - size_2 / 2.0
        pos_y2 = cent_2 + size_2 / 2.0

        vertices = ((pos_x1, pos_y1), (pos_x2, pos_y1), (pos_x2, pos_y2), (pos_x1, pos_y2))
        slab_bounds = (cent_axis - size_axis / 2, cent_axis + size_axis / 2)
        slab_bounds = tuple(jax.lax.stop_gradient(x) for x in slab_bounds)
        jax_polyslab = JaxPolySlab(vertices=vertices, axis=POLYSLAB_AXIS, slab_bounds=slab_bounds)
        jax_struct = JaxStructure(geometry=jax_polyslab, medium=jax_med)

        # ModeMonitors
        output_mnt1 = td.ModeMonitor(
            size=(td.inf, td.inf, 0),
            mode_spec=td.ModeSpec(num_modes=3),
            freqs=[2e14],
            name=MNT_NAME + "1",
        )

        # DiffractionMonitor
        output_mnt2 = td.DiffractionMonitor(
            center=(0, 4, 0),
            size=(td.inf, 0, td.inf),
            normal_dir="+",
            freqs=[2e14],
            name=MNT_NAME + "2",
        )

        output_mnt3 = td.FieldMonitor(
            size=(2, 0, 2),
            freqs=[FREQ0],
            name=MNT_NAME + "3",
        )

        output_mnt4 = td.FieldMonitor(
            size=(0, 0, 0),
            freqs=[FREQ0],
            name=MNT_NAME + "4",
        )

        sim = JaxSimulation(
            size=(10, 10, sim_size_axis),
            run_time=1e-12,
            grid_spec=td.GridSpec(wavelength=1.0),
            boundary_spec=td.BoundarySpec.all_sides(boundary=td.Periodic()),
            output_monitors=(output_mnt1, output_mnt2, output_mnt3, output_mnt4),
            input_structures=(jax_struct,),
            sources=[
                td.PointDipole(
                    source_time=td.GaussianPulse(freq0=1e14, fwidth=1e14),
                    center=(0, 0, 0),
                    polarization="Ex",
                )
            ],
        )

        sim_data = run(sim, task_name="test")
        amp = extract_amp(sim_data)
        return objective(amp)

    f_b = lambda size, center: f(size, center)

    g_b = grad(f_b, argnums=(0, 1))

    gs_b, gc_b = g_b((1.0, 2.0, 100.0), CENTER)


@pytest.mark.parametrize("local", (True, False))
def test_adjoint_run_async(local, use_emulated_run_async):
    """Test differnetiating thorugh async adjoint runs"""

    run_fn = run_async_local if local else run_async

    def make_sim_simple(permittivity: float) -> JaxSimulation:
        """Make a sim as a function of a single parameter."""
        return make_sim(
            permittivity=permittivity, size=SIZE, vertices=VERTICES, base_eps_val=BASE_EPS_VAL
        )

    def f(x):
        """Objective function to differentiate."""

        sims = []
        for i in range(1):
            permittivity = x + 1.0
            sims.append(make_sim_simple(permittivity=permittivity))

        sim_data_list = run_fn(sims, path_dir=TMP_DIR)

        result = 0.0
        for sim_data in sim_data_list:
            amp = extract_amp(sim_data)
            result += objective(amp)

        return result

    # test evaluating the function
    x0 = 1.0
    # f0 = await f(x0)

    # and its derivatve
    f0 = f(x0)
    g = jax.grad(f)
    g0 = g(x0)


@pytest.mark.parametrize("axis", (0, 1, 2))
def test_diff_data_angles(axis):

    center = td.DiffractionMonitor.unpop_axis(2, (0, 0), axis)
    size = td.DiffractionMonitor.unpop_axis(0, (td.inf, td.inf), axis)

    SIZE_2D = 1.0
    ORDERS_X = [-1, 0, 1]
    ORDERS_Y = [-1, 0, 1]
    FS = [2e14]

    DIFFRACTION_MONITOR = td.DiffractionMonitor(
        center=center,
        size=size,
        freqs=FS,
        name="diffraction",
    )

    values = (1 + 1j) * np.random.random((len(ORDERS_X), len(ORDERS_Y), len(FS)))
    sim_size = [SIZE_2D, SIZE_2D]
    bloch_vecs = [0, 0]
    data = JaxDataArray(values=values, coords=dict(orders_x=ORDERS_X, orders_y=ORDERS_Y, f=FS))

    diff_data = JaxDiffractionData(
        monitor=DIFFRACTION_MONITOR,
        Etheta=data,
        Ephi=data,
        Er=data,
        Htheta=data,
        Hphi=data,
        Hr=data,
        sim_size=sim_size,
        bloch_vecs=bloch_vecs,
    )

    thetas, phis = diff_data.angles
    zeroth_order_theta = thetas.sel(orders_x=0, orders_y=0).isel(f=0)

    assert np.isclose(zeroth_order_theta, 0.0)


def _test_error_regular_web():
    """Test that a custom error is raised if running tidy3d through web.run()"""

    sim = make_sim(permittivity=EPS, size=SIZE, vertices=VERTICES, base_eps_val=BASE_EPS_VAL)
    import tidy3d.web as web

    with pytest.raises(ValueError):
        web.run(sim, task_name="test")


def test_value_filter():
    """Ensure value filter works as expected."""

    values = np.array([1, 0.5 * VALUE_FILTER_THRESHOLD, 2 * VALUE_FILTER_THRESHOLD, 0])
    coords = dict(x=list(range(4)))
    data = JaxDataArray(values=values, coords=coords)

    values_after, _ = data.nonzero_val_coords

    # assert that the terms <= VALUE_FILTER_THRESHOLD should be removed
    values_expected = np.array([1, 2 * VALUE_FILTER_THRESHOLD])
    assert np.allclose(np.array(values_after), values_expected)


def test_jax_info_to_file():
    """Test writing jax info to file."""

    sim = make_sim(permittivity=EPS, size=SIZE, vertices=VERTICES, base_eps_val=BASE_EPS_VAL)
    _, jax_info = sim.to_simulation()
    jax_info.to_file("tests/tmp/jax_info.json")


def test_split_fwd_sim_data():
    """Test splitting of regular simulation data into user and server data."""

    jax_sim = make_sim(permittivity=EPS, size=SIZE, vertices=VERTICES, base_eps_val=BASE_EPS_VAL)
    sim, jax_info = jax_sim.to_simulation()
    sim_data = run_emulated(sim, task_name="test", path=SIM_DATA_PATH)
    data_user, data_adj = JaxSimulationData.split_fwd_sim_data(sim_data=sim_data, jax_info=jax_info)


def test_save_load_simdata(use_emulated_run):
    """Make sure a simulation data can be saved and loaded from file and retain info."""

    sim = make_sim(permittivity=EPS, size=SIZE, vertices=VERTICES, base_eps_val=BASE_EPS_VAL)
    sim_data = run(sim, task_name="test", path=RUN_PATH)
    sim_data.to_file("tests/tmp/adjoint_simdata.hdf5")
    sim_data2 = JaxSimulationData.from_file("tests/tmp/adjoint_simdata.hdf5")
    assert sim_data == sim_data2


def _test_polyslab_scale(use_emulated_run):
    """Make sure box made with polyslab gives equivalent gradients (note, doesn't pass now)."""

    nums = np.logspace(np.log10(3), 3, 13)
    times = []
    for num_vertices in nums:
        num_vertices = int(num_vertices)

        angles = 2 * np.pi * np.arange(num_vertices) / num_vertices
        xs = np.cos(angles)
        ys = np.sin(angles)
        vertices = np.stack((xs, ys), axis=1).tolist()
        np.random.seed(0)
        start_time = time.time()

        def f(scale=1.0):

            jax_med = JaxMedium(permittivity=2.0)
            POLYSLAB_AXIS = 2

            size_axis, (size_1, size_2) = JaxPolySlab.pop_axis(SIZE, axis=POLYSLAB_AXIS)
            cent_axis, (cent_1, cent_2) = JaxPolySlab.pop_axis(CENTER, axis=POLYSLAB_AXIS)

            vertices_jax = [(scale * x, scale * y) for x, y in vertices]
            # vertices_jax = [(x, y) for x, y in vertices]

            slab_bounds = (cent_axis - size_axis / 2, cent_axis + size_axis / 2)
            slab_bounds = tuple(jax.lax.stop_gradient(x) for x in slab_bounds)
            jax_polyslab = JaxPolySlab(
                vertices=vertices_jax, axis=POLYSLAB_AXIS, slab_bounds=slab_bounds
            )
            jax_struct = JaxStructure(geometry=jax_polyslab, medium=jax_med)

            # ModeMonitors
            output_mnt1 = td.ModeMonitor(
                size=(td.inf, td.inf, 0),
                mode_spec=td.ModeSpec(num_modes=3),
                freqs=[2e14],
                name=MNT_NAME + "1",
            )

            # DiffractionMonitor
            output_mnt2 = td.DiffractionMonitor(
                center=(0, 4, 0),
                size=(td.inf, 0, td.inf),
                normal_dir="+",
                freqs=[2e14],
                name=MNT_NAME + "2",
            )

            sim = JaxSimulation(
                size=(10, 10, 10),
                run_time=1e-12,
                grid_spec=td.GridSpec(wavelength=1.0),
                boundary_spec=td.BoundarySpec.all_sides(boundary=td.Periodic()),
                output_monitors=(output_mnt1, output_mnt2),
                input_structures=(jax_struct,),
                sources=[
                    td.PointDipole(
                        source_time=td.GaussianPulse(freq0=1e14, fwidth=1e14),
                        center=(0, 0, 0),
                        polarization="Ex",
                    )
                ],
            )

            sim_data = run(sim, task_name="test")
            amp = extract_amp(sim_data)
            return objective(amp)

        g = grad(f)

        g_eval = g(1.0)

        total_time = time.time() - start_time
        print(f"{num_vertices} vertices took {total_time:.2e} seconds")
        times.append(total_time)

    plt.plot(nums, times)
    plt.xlabel("number of vertices")
    plt.ylabel("time to compute gradient")
    plt.xscale("log")
    plt.yscale("log")
    plt.show()


def test_validate_vertices():
    """Test the maximum number of vertices."""
    vertices = np.random.rand(MAX_NUM_VERTICES, 2)
    poly = JaxPolySlab(vertices=vertices, slab_bounds=(-1, 1))
    vertices = np.random.rand(MAX_NUM_VERTICES + 1, 2)
    with pytest.raises(pydantic.ValidationError):
        poly = JaxPolySlab(vertices=vertices, slab_bounds=(-1, 1))


def _test_custom_medium_3D(use_emulated_run):
    """Ensure custom medium fails if 3D pixelated grid."""
    # NOTE: turned off since we relaxed this restriction

    jax_box = JaxBox(size=(1, 1, 1), center=(0, 0, 0))

    def make_custom_medium(Nx: int, Ny: int, Nz: int) -> JaxCustomMedium:

        # custom medium
        (xmin, ymin, zmin), (xmax, ymax, zmax) = jax_box.bounds
        coords = dict(
            x=np.linspace(xmin, xmax, Nx).tolist(),
            y=np.linspace(ymin, ymax, Ny).tolist(),
            z=np.linspace(zmin, zmax, Nz).tolist(),
            f=[FREQ0],
        )

        values = np.random.random((Nx, Ny, Nz, 1))
        eps_ii = JaxDataArray(values=values, coords=coords)
        field_components = {f"eps_{dim}{dim}": eps_ii for dim in "xyz"}
        jax_eps_dataset = JaxPermittivityDataset(**field_components)
        return JaxCustomMedium(eps_dataset=jax_eps_dataset)

    make_custom_medium(1, 1, 1)
    make_custom_medium(10, 1, 1)
    make_custom_medium(1, 10, 1)
    make_custom_medium(1, 1, 10)
    make_custom_medium(1, 10, 10)
    make_custom_medium(10, 1, 10)
    make_custom_medium(10, 10, 1)
    with pytest.raises(pydantic.ValidationError):
        make_custom_medium(10, 10, 10)


def test_custom_medium_size(use_emulated_run):
    """Ensure custom medium fails if too many cells provided."""

    jax_box = JaxBox(size=(1, 1, 1), center=(0, 0, 0))

    def make_custom_medium(num_cells: int) -> JaxCustomMedium:

        Nx = num_cells
        Ny = Nz = 1

        # custom medium
        (xmin, ymin, zmin), (xmax, ymax, zmax) = jax_box.bounds
        coords = dict(
            x=np.linspace(xmin, xmax, Nx).tolist(),
            y=np.linspace(ymin, ymax, Ny).tolist(),
            z=np.linspace(zmin, zmax, Nz).tolist(),
            f=[FREQ0],
        )

        values = np.random.random((Nx, Ny, Nz, 1))
        eps_ii = JaxDataArray(values=values, coords=coords)
        field_components = {f"eps_{dim}{dim}": eps_ii for dim in "xyz"}
        jax_eps_dataset = JaxPermittivityDataset(**field_components)
        return JaxCustomMedium(eps_dataset=jax_eps_dataset)

    make_custom_medium(num_cells=1)
    make_custom_medium(num_cells=MAX_NUM_CELLS_CUSTOM_MEDIUM)
    with pytest.raises(pydantic.ValidationError):
        make_custom_medium(num_cells=MAX_NUM_CELLS_CUSTOM_MEDIUM + 1)


def test_jax_sim_io():

    jax_box = JaxBox(size=(1, 1, 1), center=(0, 0, 0))

    def make_custom_medium(num_cells: int) -> JaxCustomMedium:

        n = int(np.sqrt(num_cells))
        Nx = n
        Ny = n
        Nz = 1

        # custom medium
        (xmin, ymin, zmin), (xmax, ymax, zmax) = jax_box.bounds
        coords = dict(
            x=np.linspace(xmin, xmax, Nx).tolist(),
            y=np.linspace(ymin, ymax, Ny).tolist(),
            z=np.linspace(zmin, zmax, Nz).tolist(),
            f=[FREQ0],
        )

        values = np.random.random((Nx, Ny, Nz, 1)) + 1.0
        eps_ii = JaxDataArray(values=values, coords=coords)
        field_components = {f"eps_{dim}{dim}": eps_ii for dim in "xyz"}
        jax_eps_dataset = JaxPermittivityDataset(**field_components)
        return JaxCustomMedium(eps_dataset=jax_eps_dataset)

    num_cells = 200 * 200
    struct = JaxStructure(geometry=jax_box, medium=make_custom_medium(num_cells=num_cells))
    sim = JaxSimulation(
        size=(2, 2, 2),
        input_structures=[struct],
        run_time=1e-12,
        grid_spec=td.GridSpec.auto(wavelength=1.0),
    )

    fname = "tests/tmp/jax_sim_io_tmp.hdf5"
    sim.to_file(fname)

    with h5py.File(fname, "r") as f:
        assert "input_structures" in f.keys()
        json_string = str(f["JSON_STRING"][()])
        assert JAX_DATA_ARRAY_TAG in json_string

    sim2 = JaxSimulation.from_file(fname)

    assert sim == sim2


def test_num_input_structures():
    """Assert proper error is raised if number of input structures is too large."""

    def make_sim_(num_input_structures: int) -> JaxSimulation:

        sim = make_sim(permittivity=EPS, size=SIZE, vertices=VERTICES, base_eps_val=BASE_EPS_VAL)
        struct = sim.input_structures[0]
        return sim.updated_copy(input_structures=num_input_structures * [struct])

    sim = make_sim_(num_input_structures=MAX_NUM_INPUT_STRUCTURES)

    with pytest.raises(pydantic.ValidationError):
        sim = make_sim_(num_input_structures=MAX_NUM_INPUT_STRUCTURES + 1)


@pytest.mark.parametrize("strict_binarize", (True, False))
def test_adjoint_utils(strict_binarize):
    """Test filtering, projection, and optimization routines."""

    sim = make_sim(permittivity=EPS, size=SIZE, vertices=VERTICES, base_eps_val=BASE_EPS_VAL)

    # projection / filtering
    image = sim.input_structures[2].medium.eps_dataset.eps_xx.values

    filter = ConicFilter(feature_size=1.5, design_region_dl=0.1)
    filter.evaluate(image)
    projector = BinaryProjector(vmin=1.0, vmax=2.0, beta=1.5, strict_binarize=strict_binarize)
    projector.evaluate(image)

    # radius of curvature

    polyslab = sim.input_structures[3].geometry

    radius_penalty = RadiusPenalty(min_radius=0.2, wrap=True)
    penalty = radius_penalty.evaluate(polyslab.vertices)
