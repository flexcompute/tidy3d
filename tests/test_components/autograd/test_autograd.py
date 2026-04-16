# test autograd integration into tidy3d
from __future__ import annotations

import copy
import cProfile
import re
import typing
import warnings
from contextlib import nullcontext
from dataclasses import dataclass
from importlib import reload
from os.path import join
from types import MethodType

import autograd as ag
import autograd.numpy as anp
import h5py
import matplotlib.pylab as plt
import numpy as np
import numpy.testing as npt
import pytest
import xarray as xr
from autograd.test_util import check_grads

import tidy3d as td
import tidy3d.web as web
from tidy3d import Box, Geometry, GeometryGroup
from tidy3d.components.autograd.derivative_utils import DerivativeInfo
from tidy3d.components.autograd.field_map import FieldMap
from tidy3d.components.autograd.utils import get_static, hasbox, is_tidy_box
from tidy3d.components.base import TRACED_FIELD_KEYS_ATTR
from tidy3d.components.data.data_array import DataArray
from tidy3d.components.geometry.primitives import discretization_wavelength
from tidy3d.config import config
from tidy3d.exceptions import AdjointError
from tidy3d.plugins.polyslab import ComplexPolySlab
from tidy3d.plugins.smatrix import ComponentModeler, Port
from tidy3d.plugins.smatrix.run import _run_local
from tidy3d.web import run, run_async
from tidy3d.web.api.autograd import autograd as autograd_module
from tidy3d.web.api.autograd.autograd import run_async_custom, run_custom, verify_custom_vjp
from tidy3d.web.api.autograd.context import AutogradContext
from tidy3d.web.api.autograd.types import CustomVJPConfig, NumericalStructureConfig

from ...utils import (
    SIM_FULL,
    AssertLogLevel,
    AssertLogStr,
    custom_poleresidue_u,
    run_emulated,
    tracer_arr,
)

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
_NUMERICAL_COMBINATION = ("polyslab", "mode")

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


# --- helpers for custom dispersive tests ---
def _patch_cmp_custom_to_const(monkeypatch, cls, dJ_const):
    """Monkeypatch `_derivative_field_cmp_custom` to return a constant split across xyz."""

    def _fake(self, E_der_map, spatial_data, dim, sum_over_freqs=True, **kwargs):
        if sum_over_freqs:
            if dJ_const.ndim > 3:
                return dJ_const.sum(axis=-1) / 3.0
            return dJ_const / 3.0
        dJ_with_freq = dJ_const[..., None]
        return dJ_with_freq / 3.0

    monkeypatch.setattr(cls, "_derivative_field_cmp_custom", _fake)


def _make_di(paths, freq):
    """Construct a minimal DerivativeInfo shared by custom dispersive tests."""

    eps_keys = ["eps_xx", "eps_yy", "eps_zz"]

    return DerivativeInfo(
        paths=paths,
        E_der_map={},
        D_der_map={},
        E_fwd={},
        D_fwd={},
        E_adj={},
        D_adj={},
        eps_data={
            key: td.ScalarFieldDataArray(
                [[[[2.0]]]], coords={"x": [0], "y": [0], "z": [0], "f": [200e12]}
            )
            for key in eps_keys
        },
        frequencies=[freq],
        bounds=((-1, -1, -1), (1, 1, 1)),
        bounds_intersect=((-1, -1, -1), (1, 1, 1)),
        simulation_bounds=((-2, -2, -2), (2, 2, 2)),
        updated_epsilon=lambda geom: td.ScalarFieldDataArray(
            [[[[1.0]]]], coords={"x": [0], "y": [0], "z": [0], "f": [freq]}
        ),
    )


""" simulation configuration """

WVL = 1.0
FREQ0 = td.C_0 / WVL
FREQS = [FREQ0]
FWIDTH = FREQ0 / 10

# sim sizes
LZ = 7.0 * WVL

IS_3D = False
POLYSLAB_AXIS = 2
POLYSLAB_SELECT_VERTICES = 0

# angle of the measurement waveguide
ROT_ANGLE_WG = 0 * np.pi / 4

# position of output mode monitor
MODE_FIELD_SPC = 0.75
MODE_FLD_MNT_SPC = MODE_FIELD_SPC * WVL

LX = 3.5 * WVL if IS_3D else 0.0
PML_X = True if IS_3D else False

# shape of the custom medium
DA_SHAPE_X = 1
DA_SHAPE = (DA_SHAPE_X, 1_000, 1_000) if TEST_CUSTOM_MEDIUM_SPEED else (DA_SHAPE_X, 12, 12)

# number of vertices in the polyslab
NUM_VERTICES = 100_000 if TEST_POLYSLAB_SPEED else 25

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
    pol_angle=0,
)

# sim that we add traced structures and monitors to
SIM_BASE = td.Simulation(
    size=(LX, 3.15, LZ),
    run_time=200 / FWIDTH,
    sources=(PLANE_WAVE,),
    structures=(
        td.Structure(
            geometry=td.Box(
                size=(0.5, 0.5, LZ / 2),
                center=(0, 0, 0),
            )
            .rotated(ROT_ANGLE_WG, axis=0)
            .translated(x=0, y=-np.tan(ROT_ANGLE_WG) * MODE_FIELD_SPC, z=LZ / 2),
            medium=td.Medium(permittivity=2.0),
        ),
    ),
    monitors=(
        td.FieldMonitor(
            center=(0, 0, 0),
            size=(0, 0, 0),
            freqs=[FREQ0],
            name="extraneous",
        ),
    ),
    boundary_spec=td.BoundarySpec.pml(x=PML_X, y=True, z=True),
    grid_spec=td.GridSpec.uniform(dl=0.01 * td.C_0 / FREQ0),
)

# variable to store whether the emulated run as used
_run_was_emulated = [False]


@pytest.fixture
def use_emulated_run(monkeypatch):
    """If this fixture is used, the `tests.utils.run_emulated` function is used for simulation."""

    import tidy3d

    if TEST_MODE in ("pipeline", "speed"):
        task_name_fwd = "task_fwd"
        AUX_KEY_SIM_FIELDS_KEYS = "sim_fields_keys"

        cache = {}

        import tidy3d.web.api.webapi as webapi

        # reload(tidy3d.web.api.autograd.autograd)
        from tidy3d.web.api.autograd.autograd import (
            postprocess_adj,
            postprocess_fwd,
        )

        def emulated_run_fwd(simulation, task_name, **run_kwargs) -> td.SimulationData:
            """What gets called instead of ``web/api/autograd/autograd.py::_run_tidy3d``."""
            task_name_fwd = task_name
            if run_kwargs.get("simulation_type") == "autograd_fwd":
                sim_original = simulation
                sim_fields_keys = run_kwargs["sim_fields_keys"]
                # add gradient monitors and make combined simulation
                sim_combined = sim_original._with_adjoint_monitors(sim_fields_keys)
                sim_data_combined = run_emulated(sim_combined, task_name=task_name)

                # store both original and forward data in the typed context
                context = AutogradContext()

                _ = postprocess_fwd(
                    sim_data_combined=sim_data_combined,
                    sim_original=sim_original,
                    context=context,
                )

                # cache original and forward data locally for test
                cache[task_name_fwd] = {
                    "context": copy.copy(context),
                    AUX_KEY_SIM_FIELDS_KEYS: sim_fields_keys,
                }
                # return original data only
                return context.simulation_data_original, task_name_fwd
            else:
                return run_emulated(simulation, task_name=task_name), task_name_fwd

        def emulated_run_bwd(simulation, task_name, **run_kwargs) -> td.SimulationData:
            """What gets called instead of ``web/api/autograd/autograd.py::_run_tidy3d_bwd``."""
            task_name_fwd = "".join(task_name.partition("_adjoint")[:-2])

            # run the adjoint sim
            sim_data_adj = run_emulated(simulation, task_name="task_name")

            # grab the fwd and original data from the cache
            cached_context = cache[task_name_fwd]["context"]
            sim_data_orig = cached_context.simulation_data_original
            sim_data_fwd = cached_context.simulation_data_forward

            # Reuse forward permittivity monitor payloads in the emulated adjoint data so
            # geometry-gradient spacing logic sees consistent permittivity on both sides.
            data_adj = []
            for mnt_data in sim_data_adj.data:
                if (
                    mnt_data.monitor.name.startswith("adjoint_eps_")
                    and mnt_data.monitor.name in sim_data_fwd.monitor_data
                ):
                    fwd_eps_data = sim_data_fwd.monitor_data[mnt_data.monitor.name]
                    monitor_freqs = np.array(mnt_data.monitor.freqs)
                    eps_components = {
                        key: value.sel(f=monitor_freqs)
                        for key, value in fwd_eps_data.field_components.items()
                    }
                    mnt_data = fwd_eps_data.updated_copy(monitor=mnt_data.monitor, **eps_components)
                data_adj.append(mnt_data)
            data_adj = tuple(data_adj)
            sim_data_adj = sim_data_adj.updated_copy(data=data_adj)

            # get the original traced fields
            sim_fields_keys = cache[task_name_fwd][AUX_KEY_SIM_FIELDS_KEYS]

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
                run_kwargs_task = dict(run_kwargs)
                if sim_fields_keys_dict is not None:
                    sim_fields_keys = sim_fields_keys_dict.get(task_name)
                    if sim_fields_keys is not None and "_parallel_adj_" not in task_name:
                        run_kwargs_task["sim_fields_keys"] = sim_fields_keys
                    else:
                        run_kwargs_task.pop("sim_fields_keys", None)
                sim_data_orig, task_name_fwd = emulated_run_fwd(
                    simulation, task_name, **run_kwargs_task
                )
                batch_data_orig[task_name] = sim_data_orig
                task_ids_fwd[task_name] = task_name_fwd

            class EmulatedBatchData(web.BatchData):
                def load_sim_data(self, task_name):
                    return batch_data_orig[task_name]

            task_paths = dict.fromkeys(simulations.keys(), "")

            batch_data = EmulatedBatchData(
                task_paths=task_paths,
                task_ids=task_ids_fwd,
                verbose=False,
            )

            return batch_data, task_ids_fwd

        def emulated_run_async_bwd(simulations, **run_kwargs) -> td.SimulationData:
            vjp_dict = {}
            for task_name, simulation in simulations.items():
                vjp_dict[task_name] = emulated_run_bwd(simulation, task_name, **run_kwargs)
            return vjp_dict

        monkeypatch.setattr(webapi, "run", run_emulated)
        monkeypatch.setattr(tidy3d.web.api.autograd.autograd, "_run_tidy3d", emulated_run_fwd)
        monkeypatch.setattr(
            tidy3d.web.api.autograd.autograd, "_run_async_tidy3d", emulated_run_async_fwd
        )
        monkeypatch.setattr(
            tidy3d.web.api.autograd.autograd, "_run_async_tidy3d_bwd", emulated_run_async_bwd
        )

        _run_was_emulated[0] = True
        return emulated_run_fwd, emulated_run_bwd


def make_structures(
    params: anp.ndarray, polyslab_axis: int = POLYSLAB_AXIS
) -> dict[str, td.Structure]:
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
        background_medium=td.Medium(permittivity=5.0),
    )

    # custom medium with variable permittivity data
    len_arr = np.prod(DA_SHAPE)
    matrix = np.random.random((len_arr, N_PARAMS))
    # matrix /= np.linalg.norm(matrix)

    eps_arr = 1.01 + 0.5 * (anp.tanh(matrix @ params).reshape(DA_SHAPE) + 1)

    nx, ny, nz = eps_arr.shape
    da_coords = {
        "x": np.linspace(-0.5, 0.5, nx),
        "y": np.linspace(-0.5, 0.5, ny),
        "z": np.linspace(-0.5, 0.5, nz),
    }

    custom_med = td.Structure(
        geometry=box,
        medium=td.CustomMedium(
            permittivity=td.SpatialDataArray(
                eps_arr,
                coords=da_coords,
            ),
        ),
    )

    # custom medium with variable permittivity and conductivity data
    conductivity_arr = 0.01 * (anp.tanh(matrix @ params).reshape(DA_SHAPE) + 1)
    custom_med_with_conductivity = td.Structure(
        geometry=box,
        medium=td.CustomMedium(
            permittivity=td.SpatialDataArray(
                eps_arr,
                coords=da_coords,
            ),
            conductivity=td.SpatialDataArray(
                conductivity_arr,
                coords=da_coords,
            ),
        ),
    )

    # custom medium with vector valued permittivity data
    eps_ii = td.ScalarFieldDataArray(
        eps_arr.reshape(nx, ny, nz, 1),
        coords=da_coords | {"f": [td.C_0]},
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

    free_param = "vertices" if polyslab_axis == POLYSLAB_SELECT_VERTICES else "slab_bounds"
    if free_param == "vertices":
        radii = 0.5 + 0.5 * params_01
        slab_bounds = (-0.5, 0.5)
    elif free_param == "slab_bounds":
        radii = 1.0
        shift = 0.1 * params_01
        slab_bounds = (-0.5 + shift, 0.5 + shift)

    phis = 2 * anp.pi * anp.linspace(0, 1, NUM_VERTICES + 1)[:NUM_VERTICES]
    xs = radii * anp.cos(phis)
    ys = radii * anp.sin(phis)
    vertices = anp.stack((xs, ys), axis=-1)

    polyslab = td.Structure(
        geometry=td.PolySlab(
            vertices=vertices,
            slab_bounds=slab_bounds,
            axis=polyslab_axis,
            sidewall_angle=0.00,
            dilation=0.00,
        ),
        medium=med,
    )

    polyslab_dispersive = td.Structure(
        geometry=td.PolySlab(
            vertices=vertices,
            slab_bounds=slab_bounds,
            axis=polyslab_axis,
            sidewall_angle=0.00,
            dilation=0.00,
        ),
        medium=td.material_library["Si3N4"]["Philipp1973Sellmeier"],
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

    # complex polyslab
    polyslab_combined = ComplexPolySlab(
        vertices=(
            (-eps, 0),
            (-eps, eps),
            (0, eps / 10),
            (eps, eps),
            (eps, 0),
        ),
        slab_bounds=(-0.5, 0.5),
        axis=1,
        sidewall_angle=np.pi / 100,
    )

    polyslab_geometries = []
    for sub_polyslab in polyslab_combined.sub_polyslabs:
        polyslab_geometries.append(sub_polyslab)

    assert len(polyslab_geometries) >= 2, "need more polyslabs for a proper test of ComplexPolySlab"

    complex_polyslab_geo_group = td.Structure(
        geometry=td.GeometryGroup(geometries=polyslab_geometries),
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
    coords = {"x": x, "y": y, "z": z}

    eps_inf = td.SpatialDataArray(anp.real(custom_disp_values), coords=coords)
    a1 = td.SpatialDataArray(-custom_disp_values, coords=coords)
    c1 = td.SpatialDataArray(custom_disp_values, coords=coords)
    a2 = td.SpatialDataArray(-custom_disp_values, coords=coords)
    c2 = td.SpatialDataArray(custom_disp_values, coords=coords)
    custom_med_pole_res = td.CustomPoleResidue(eps_inf=eps_inf, poles=[(a1, c1), (a2, c2)])
    custom_pole_res = td.Structure(geometry=box, medium=custom_med_pole_res)

    radius = 0.4 * (1 + anp.abs(vector @ params))
    cyl_center_y = vector @ params
    cyl_center_z = -vector @ params
    cylinder_geo = td.Cylinder(
        radius=anp.mean(radii) * 0.5,
        center=(0, cyl_center_y, cyl_center_z),
        axis=0,
        length=LX / 2 if IS_3D else td.inf,
    )
    cylinder = td.Structure(geometry=cylinder_geo, medium=polyslab.medium)

    # triangle mesh geometry with param-dependent medium and vertex response
    # use first parameter for eps, the rest for vertices
    base_vertices = anp.array(
        [
            (0.0, 0.0, 0.0),
            (0.6, 0.0, 0.0),
            (0.0, 0.6, 0.0),
            (0.0, 0.0, 0.6),
        ],
    )
    base_vertices = base_vertices + anp.mean(params[1:]) - get_static(anp.mean(params[1:]))
    faces = anp.array(
        [
            (0, 2, 1),
            (0, 1, 3),
            (0, 3, 2),
            (1, 2, 3),
        ],
        dtype=int,
    )
    triangles = base_vertices[faces]
    triangle_mesh_geo = td.TriangleMesh.from_triangles(triangles)
    mesh_eps = 1.8 + params[0]
    triangle_mesh = td.Structure(
        geometry=triangle_mesh_geo,
        medium=td.Medium(permittivity=mesh_eps),
    )

    # use first 4 params for radius/center, rest for eps
    cx, cy, cz = params[1:4]
    sphere_geom = td.Sphere(radius=params[0] + 1, center=(cx, cy, cz))
    sphere = td.Structure(
        geometry=sphere_geom,
        medium=td.Medium(permittivity=anp.mean(params[4:]) + 2),
    )

    return {
        "medium": medium,
        "center_list": center_list,
        "size_element": size_element,
        "custom_med": custom_med,
        "custom_med_with_conductivity": custom_med_with_conductivity,
        "custom_med_vec": custom_med_vec,
        "polyslab": polyslab,
        "polyslab_dispersive": polyslab_dispersive,
        "geo_group": geo_group,
        "complex_polyslab": complex_polyslab_geo_group,
        "pole_res": pole_res,
        "custom_pole_res": custom_pole_res,
        "cylinder": cylinder,
        "triangle_mesh": triangle_mesh,
        "sphere": sphere,
    }


def make_monitors() -> dict[str, tuple[td.Monitor, typing.Callable[[td.SimulationData], float]]]:
    """Make a dictionary of all the possible monitors in the simulation."""

    mode_mnt = td.ModeMonitor(
        size=(2, 2, 0),
        center=(0, 0, +LZ / 2 - MODE_FIELD_SPC),
        mode_spec=td.ModeSpec(
            angle_theta=ROT_ANGLE_WG,
            angle_phi=3 * np.pi / 2,
        ),
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
        center=(0, 0, +LZ / 2 - MODE_FIELD_SPC),
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

    return {
        "mode": (mode_mnt, mode_postprocess_fn),
        "diff": (diff_mnt, diff_postprocess_fn),
        "field_vol": (field_vol, field_vol_postprocess_fn),
        "field_point": (field_point, field_point_postprocess_fn),
    }


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
    "custom_med_with_conductivity",
    "custom_med_vec",
    "polyslab",
    "complex_polyslab",
    "geo_group",
    "pole_res",
    "custom_pole_res",
    "cylinder",
    "triangle_mesh",
    "sphere",
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


def get_functions(structure_key: str, monitor_key: str) -> dict[str, typing.Callable]:
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

    def make_sim(*args, polyslab_axis=POLYSLAB_AXIS) -> td.Simulation:
        """Make the simulation with all of the fields."""

        structures_traced_dict = make_structures(*args, polyslab_axis=polyslab_axis)

        structures = list(SIM_BASE.structures)
        for structure_key in structure_keys:
            structures.append(structures_traced_dict[structure_key])

        sim = SIM_BASE
        if "diff" in monitor_keys:
            sim = sim.updated_copy(boundary_spec=td.BoundarySpec.pml(x=False, y=False, z=True))
        sim = sim.updated_copy(structures=structures, monitors=monitors)

        return sim

    def postprocess(data: td.SimulationData) -> float:
        """Postprocess the dataset."""
        mnt_data = data[monitor_key]
        return monitor_pp_fn(data, mnt_data)

    return {"sim": make_sim, "postprocess": postprocess}


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


def make_polyslab_custom_vjp(custom_vjp_val):
    def polyslab_custom_vjp(polyslab, derivative_info):
        vjps = {}

        for path in derivative_info.paths:
            if path[0:2] == ("geometry", "vertices"):
                vjps[path] = custom_vjp_val * np.ones(polyslab.vertices.shape)
            elif path[0:2] == ("geometry", "slab_bounds"):
                if len(path) == 3:
                    vjps[path] = (custom_vjp_val, custom_vjp_val)[path[2]]
                else:
                    vjps[path] = (custom_vjp_val, custom_vjp_val)

        return vjps

    return polyslab_custom_vjp


def make_polyslab_numerical_vjp(numerical_val):
    def polyslab_numerical_vjp(parameters, derivative_info):
        vjps = {}
        for path in derivative_info.paths:
            vjps[path] = numerical_val
        return vjps

    return polyslab_numerical_vjp


def make_polyslab_from_params(parameters, polyslab_axis):
    params_arr = np.array(parameters)
    return make_structures(params_arr, polyslab_axis=polyslab_axis)["polyslab"]


def test_run_custom_rejects_numerical_structures_for_unsupported_workflow_type():
    numerical_structure = NumericalStructureConfig(
        create=lambda params: td.Structure(
            geometry=td.Box(size=(1, 1, 1)),
            medium=td.Medium(permittivity=1.0),
        ),
        compute_derivatives=lambda parameters, derivative_info: {},
        parameters=np.array([0.1]),
    )

    with pytest.raises(
        td.exceptions.AdjointError,
        match="numerical_structures is only supported for 'Simulation' and ComponentModeler workflows",
    ):
        run_custom(
            simulation=object(),
            task_name="unsupported_numerical_structures",
            numerical_structures=numerical_structure,
            local_gradient=True,
        )


@pytest.mark.parametrize("structure_key, monitor_key", [("polyslab", "mode")])
@pytest.mark.parametrize("polyslab_axis", [0, 1, 2])
@pytest.mark.parametrize("use_run_async", [True, False])
@pytest.mark.parametrize("use_task_names", [True, False])
@pytest.mark.parametrize("use_single_custom_vjp", [True, False])
@pytest.mark.parametrize("specify_custom_vjp_by_type", [True, False])
@pytest.mark.parametrize("local_gradient", [True, False])
def test_autograd_custom_vjp(
    use_emulated_run,
    structure_key,
    monitor_key,
    polyslab_axis,
    use_run_async,
    use_task_names,
    use_single_custom_vjp,
    specify_custom_vjp_by_type,
    local_gradient,
):
    """Test that we can override a vjp with a user defined function."""

    fn_dict = get_functions(structure_key, monitor_key)
    make_sim = fn_dict["sim"]
    postprocess = fn_dict["postprocess"]

    task_names = ["test_a", "adjoint", "_test"]

    def make_objective(custom_vjp_val):
        polyslab_custom_vjp = make_polyslab_custom_vjp(custom_vjp_val)

        structure_index = td.PolySlab if specify_custom_vjp_by_type else 1

        path_key = (
            "geometry",
            "vertices" if polyslab_axis == POLYSLAB_SELECT_VERTICES else "slab_bounds",
        )

        custom_vjp_tuple = (
            CustomVJPConfig(
                structure=structure_index,
                compute_derivatives=polyslab_custom_vjp,
                path_key=path_key,
            ),
        )

        custom_vjp_single = CustomVJPConfig(
            structure=structure_index,
            compute_derivatives=polyslab_custom_vjp,
        )

        custom_vjp_element = custom_vjp_single if use_single_custom_vjp else custom_vjp_tuple

        def objective(*args):
            if use_task_names:
                sims = {
                    task_name: make_sim(*args, polyslab_axis=polyslab_axis)
                    for task_name in task_names
                }
                custom_vjp = dict.fromkeys(sims.keys(), custom_vjp_element)
            else:
                sims = [make_sim(*args, polyslab_axis=polyslab_axis)] * len(task_names)
                custom_vjp = [custom_vjp_element] * len(task_names)
            batch_data = {}
            if use_run_async:
                batch_data = run_async_custom(
                    sims, custom_vjp=custom_vjp, local_gradient=local_gradient
                )
            else:
                if use_task_names:
                    for task_name, sim in sims.items():
                        batch_data[task_name] = run_custom(
                            sim,
                            task_name,
                            custom_vjp=custom_vjp[task_name],
                            local_gradient=local_gradient,
                        )
                else:
                    for idx, sim in enumerate(sims):
                        batch_data[idx] = run_custom(
                            sim, custom_vjp=custom_vjp[idx], local_gradient=local_gradient
                        )

            value = 0.0

            for _, sim_data in batch_data.items():
                value += postprocess(sim_data)
            return value

        return objective

    custom_vjp_val = 1.0
    custom_vjp_val_scale = 10.0 * custom_vjp_val

    if not local_gradient:
        with pytest.raises(
            td.exceptions.AdjointError,
            match="custom_vjp specified for a remote gradient not supported.",
        ):
            val, grad = ag.value_and_grad(make_objective(custom_vjp_val))(params0)
    else:
        val, grad = ag.value_and_grad(make_objective(custom_vjp_val))(params0)
        val_scale, grad_scale = ag.value_and_grad(make_objective(custom_vjp_val_scale))(params0)

        assert np.isclose(
            np.sum(np.abs(grad * (custom_vjp_val_scale / custom_vjp_val) - grad_scale)), 0.0
        ), "Gradients were not set by the user vjp"


@pytest.mark.parametrize("error_type", ["create_num_args", "vjp_num_args", "vjp_arg_name"])
def test_numerical_structure_signature_validation(error_type):
    def create_ok(parameters):
        return td.Structure(geometry=td.Box(size=(1, 1, 1)), medium=td.Medium(permittivity=1.0))

    def vjp_ok(parameters, derivative_info):
        return {}

    def create_bad_num_args(parameters, extra_arg):
        return create_ok(parameters)

    def vjp_bad_num_args(parameters, derivative_info, extra_arg):
        return {}

    def vjp_bad_arg_name(parameters, d_info):
        return {}

    create_fn = create_ok
    vjp_fn = vjp_ok
    expected_error = ""

    if error_type == "create_num_args":
        create_fn = create_bad_num_args
        expected_error = (
            "NumericalStructureConfig.create should accept one argument "
            r"\(the parameters vector used for structure creation\), and it currently "
            "accepts 2 arguments."
        )
    elif error_type == "vjp_num_args":
        vjp_fn = vjp_bad_num_args
        expected_error = (
            "NumericalStructureConfig.compute_derivatives should accept two arguments "
            r"\(parameters, derivative_info\), and it currently accepts 3 arguments."
        )
    else:
        vjp_fn = vjp_bad_arg_name
        expected_error = (
            "NumericalStructureConfig.compute_derivatives second argument name is d_info "
            "but it should be derivative_info."
        )

    with pytest.raises(td.exceptions.AdjointError, match=expected_error):
        NumericalStructureConfig(
            create=create_fn,
            compute_derivatives=vjp_fn,
            parameters=np.array([0.1]),
        )


@pytest.mark.parametrize("polyslab_axis", [0, 1, 2])
@pytest.mark.parametrize("use_run_async", [True, False])
def test_autograd_numerical_structures(
    use_emulated_run,
    polyslab_axis,
    use_run_async,
    redirect_stdout_to_stderr,
):
    """Test that numerical_structures can override gradients for traced parameters."""

    structure_key = "polyslab"
    monitor_key = "mode"
    fn_dict = get_functions(structure_key, monitor_key)
    make_sim = fn_dict["sim"]
    postprocess = fn_dict["postprocess"]

    def make_objective(numerical_val):
        polyslab_numerical_vjp = make_polyslab_numerical_vjp(numerical_val)

        def objective(*args):
            static_args = [get_static(arg) for arg in args]
            base_sim = make_sim(*static_args, polyslab_axis=polyslab_axis)
            structures = [
                structure
                for structure in base_sim.structures
                if not isinstance(structure.geometry, td.PolySlab)
            ]
            sim_strip_structure = base_sim.updated_copy(structures=structures)

            numerical_structure = NumericalStructureConfig(
                create=lambda params: make_polyslab_from_params(params, polyslab_axis),
                compute_derivatives=polyslab_numerical_vjp,
                parameters=np.array(args).flatten(),
            )

            batch_data = {}
            if use_run_async:
                sims = {"test_a": sim_strip_structure}
                numerical_structures = {"test_a": numerical_structure}
                batch_data = run_async_custom(
                    sims,
                    numerical_structures=numerical_structures,
                    local_gradient=True,
                )
            else:
                batch_data["test_a"] = run_custom(
                    sim_strip_structure,
                    numerical_structures=numerical_structure,
                    local_gradient=True,
                )

            value = 0.0
            for _, sim_data in batch_data.items():
                value += postprocess(sim_data)
            return value

        return objective

    numerical_val = 1.0
    numerical_val_scale = 10.0 * numerical_val

    val, grad = ag.value_and_grad(make_objective(numerical_val))(params0)
    val_scale, grad_scale = ag.value_and_grad(make_objective(numerical_val_scale))(params0)

    assert np.isclose(
        np.sum(np.abs(grad * (numerical_val_scale / numerical_val) - grad_scale)), 0.0
    ), "Gradients were not set by numerical_structures"


@pytest.mark.parametrize("use_run_async", [True, False])
def test_autograd_numerical_structures_remote_gradient_unsupported(
    use_emulated_run,
    use_run_async,
):
    """numerical_structures should raise when local_gradient=False."""

    fn_dict = get_functions("polyslab", "mode")
    make_sim = fn_dict["sim"]

    polyslab_axis = 0
    polyslab_numerical_vjp = make_polyslab_numerical_vjp(1.0)

    def objective(*args):
        static_args = [get_static(arg) for arg in args]
        base_sim = make_sim(*static_args, polyslab_axis=polyslab_axis)
        structures = [
            structure
            for structure in base_sim.structures
            if not isinstance(structure.geometry, td.PolySlab)
        ]
        sim_strip_structure = base_sim.updated_copy(structures=structures)

        numerical_structure = NumericalStructureConfig(
            create=lambda params: make_polyslab_from_params(params, polyslab_axis),
            compute_derivatives=polyslab_numerical_vjp,
            parameters=np.array(args).flatten(),
        )

        if use_run_async:
            run_async_custom(
                {"test_a": sim_strip_structure},
                numerical_structures={"test_a": numerical_structure},
                local_gradient=False,
            )
        else:
            run_custom(
                sim_strip_structure,
                numerical_structures=numerical_structure,
                local_gradient=False,
            )

        return 0.0

    with pytest.raises(
        td.exceptions.AdjointError,
        match="numerical_structures specified for a remote gradient not supported.",
    ):
        ag.value_and_grad(objective)(params0)


@pytest.mark.parametrize("structure_key, monitor_key", [("polyslab", "mode")])
@pytest.mark.parametrize("polyslab_axis", [0, 1, 2])
@pytest.mark.parametrize("use_run_async", [True, False])
@pytest.mark.parametrize("use_task_names", [True, False])
@pytest.mark.parametrize("use_single_custom_vjp", [True, False])
def test_autograd_custom_vjp_selective(
    use_emulated_run,
    structure_key,
    monitor_key,
    polyslab_axis,
    use_run_async,
    use_task_names,
    use_single_custom_vjp,
):
    """Test that we can selectively override a vjp with a user defined function that covers some of, but not all, gradient keys."""

    fn_dict = get_functions(structure_key, monitor_key)
    make_sim = fn_dict["sim"]
    postprocess = fn_dict["postprocess"]

    task_names = ["test_a", "adjoint", "_test"]

    def make_objective(custom_vjp_val):
        polyslab_custom_vjp = make_polyslab_custom_vjp(custom_vjp_val)

        custom_vjp_tuple = (
            CustomVJPConfig(
                structure=1,
                compute_derivatives=polyslab_custom_vjp,
                path_key=(
                    (
                        "geometry",
                        "vertices",
                    )
                ),
            ),
        )

        custom_vjp_single = CustomVJPConfig(
            structure=1,
            compute_derivatives=polyslab_custom_vjp,
            path_key=(
                (
                    "geometry",
                    "vertices",
                )
            ),
        )

        custom_vjp_element = custom_vjp_single if use_single_custom_vjp else custom_vjp_tuple
        if not (polyslab_axis == POLYSLAB_SELECT_VERTICES):
            custom_vjp_element = None

        def objective(*args):
            if custom_vjp_element:
                if use_task_names:
                    custom_vjp = dict.fromkeys(task_names, custom_vjp_element)
                else:
                    sims = [make_sim(*args, polyslab_axis=polyslab_axis)] * len(task_names)
                    custom_vjp = [custom_vjp_element] * len(task_names)
            else:
                custom_vjp = None

            if use_task_names:
                sims = {
                    task_name: make_sim(*args, polyslab_axis=polyslab_axis)
                    for task_name in task_names
                }
            else:
                sims = [make_sim(*args, polyslab_axis=polyslab_axis)] * len(task_names)

            batch_data = {}
            if use_run_async:
                batch_data = run_async_custom(sims, custom_vjp=custom_vjp, local_gradient=True)
            else:
                if use_task_names:
                    for task_name, sim in sims.items():
                        batch_data[task_name] = run_custom(
                            sim,
                            task_name,
                            custom_vjp=custom_vjp and custom_vjp[task_name],
                            local_gradient=True,
                        )
                else:
                    for idx, sim in enumerate(sims):
                        batch_data[idx] = run_custom(
                            sim, custom_vjp=custom_vjp and custom_vjp[idx], local_gradient=True
                        )

            value = 0.0

            for _, sim_data in batch_data.items():
                value += postprocess(sim_data)
            return value

        return objective

    custom_vjp_val = 1.0
    custom_vjp_val_scale = 10.0 * custom_vjp_val

    val, grad = ag.value_and_grad(make_objective(custom_vjp_val))(params0)
    val_scale, grad_scale = ag.value_and_grad(make_objective(custom_vjp_val_scale))(params0)

    if polyslab_axis == POLYSLAB_SELECT_VERTICES:
        assert np.isclose(
            np.sum(np.abs(grad * (custom_vjp_val_scale / custom_vjp_val) - grad_scale)), 0.0
        ), "Gradients were not set by the user vjp when they should have been"
    else:
        assert not np.isclose(
            np.sum(np.abs(grad * (custom_vjp_val_scale / custom_vjp_val) - grad_scale)), 0.0
        ), "Gradients were set by the user vjp when they should not have been"


@pytest.mark.parametrize("structure_key, monitor_key", [("polyslab", "mode")])
@pytest.mark.parametrize("use_run_async", [True, False])
def test_autograd_error_custom_vjp_indices_and_paths(
    use_emulated_run,
    structure_key,
    monitor_key,
    use_run_async,
):
    """Test error checking for custom_vjp when structure index or path does not
    exist in traced vjp structure paths.
    """

    fn_dict = get_functions(structure_key, monitor_key)
    make_sim = fn_dict["sim"]
    postprocess = fn_dict["postprocess"]

    task_names = ["test_a", "adjoint", "_test"]
    custom_vjp_paths = {
        "test_a": ("geometry", "vertices"),
        "adjoint": ("geometry", "vertices"),
        "_test": ("geometry", "verslab"),
    }
    custom_vjp_structure_indices = {"test_a": 1, "adjoint": 5, "_test": 1}

    def make_objective(custom_vjp_val):
        polyslab_custom_vjp = make_polyslab_custom_vjp(custom_vjp_val)

        custom_vjp_dict = {}
        custom_vjp_dict_bad_structure_index = {}
        custom_vjp_dict_bad_structure_path = {}
        for task_name in task_names:
            custom_vjp_dict[task_name] = CustomVJPConfig(
                structure=custom_vjp_structure_indices[task_name],
                compute_derivatives=polyslab_custom_vjp,
                path_key=custom_vjp_paths[task_name],
            )

            custom_vjp_dict_bad_structure_index[task_name] = CustomVJPConfig(
                structure=custom_vjp_structure_indices[task_name],
                compute_derivatives=polyslab_custom_vjp,
                path_key=("geometry", "vertices"),
            )

            custom_vjp_dict_bad_structure_path[task_name] = CustomVJPConfig(
                structure=1,
                compute_derivatives=polyslab_custom_vjp,
                path_key=custom_vjp_paths[task_name],
            )

        def objective(*args):
            sims = {task_name: make_sim(*args, polyslab_axis=0) for task_name in task_names}

            batch_data = {}
            if use_run_async:
                with pytest.raises(
                    td.exceptions.AdjointError,
                    match=f"CustomVJPConfig structure index {custom_vjp_structure_indices['adjoint']} not in traced structure indices.",
                ):
                    batch_data = run_async_custom(
                        sims, custom_vjp=custom_vjp_dict_bad_structure_index, local_gradient=True
                    )
                with pytest.raises(
                    td.exceptions.AdjointError,
                    match=re.escape(
                        f"CustomVJPConfig path {custom_vjp_paths['_test']} not in traced structure paths"
                    )
                    + r".*",
                ):
                    batch_data = run_async_custom(
                        sims, custom_vjp=custom_vjp_dict_bad_structure_path, local_gradient=True
                    )
            else:
                for task_name, sim in sims.items():
                    if task_name == "adjoint":
                        with pytest.raises(
                            td.exceptions.AdjointError,
                            match=f"CustomVJPConfig structure index {custom_vjp_structure_indices[task_name]} not in traced structure indices.",
                        ):
                            batch_data[task_name] = run_custom(
                                sim,
                                task_name,
                                custom_vjp=custom_vjp_dict[task_name],
                                local_gradient=True,
                            )

                    elif task_name == "_test":
                        with pytest.raises(
                            td.exceptions.AdjointError,
                            match=re.escape(
                                f"CustomVJPConfig path {custom_vjp_paths[task_name]} not in traced structure paths"
                            )
                            + r".*",
                        ):
                            batch_data[task_name] = run_custom(
                                sim,
                                task_name,
                                custom_vjp=custom_vjp_dict[task_name],
                                local_gradient=True,
                            )
                    else:
                        batch_data[task_name] = run_custom(
                            sim,
                            task_name,
                            custom_vjp=custom_vjp_dict[task_name],
                            local_gradient=True,
                        )

            value = 0.0

            for _, sim_data in batch_data.items():
                value += postprocess(sim_data)
            return value

        return objective

    custom_vjp_val = 1.0
    # The async branch intentionally exercises only failing custom_vjp configs, so the
    # objective returns a constant 0.0 and autograd is expected to warn here.
    warn_context = (
        pytest.warns(UserWarning, match="Output seems independent of input.")
        if use_run_async
        else nullcontext()
    )
    with warn_context:
        val, grad = ag.value_and_grad(make_objective(custom_vjp_val))(params0)


def test_custom_vjp_rejects_numerical_namespace_structure_indices():
    """Custom VJP indices must come from original structures namespace only."""

    def custom_vjp_fn(target, derivative_info):
        return {}

    traced_fields = {
        ("structures", 0, "geometry", "vertices"): 1.0,
        ("numerical", 1, 0): 2.0,
    }
    custom_vjp = (
        CustomVJPConfig(
            structure=1,  # present only in numerical namespace
            compute_derivatives=custom_vjp_fn,
            path_key=None,
        ),
    )

    with pytest.raises(
        td.exceptions.AdjointError,
        match="CustomVJPConfig structure index 1 not in traced structure indices.",
    ):
        autograd_module.verify_custom_vjp(custom_vjp, traced_fields)


def test_autograd_error_custom_vjp_function():
    """Test construction-time error checking for custom_vjp signature."""

    def polyslab_custom_vjp_bad_num_args(polyslab, derivative_info, extra_arg):
        return {}

    def polyslab_custom_vjp_bad_arg_name(polyslab, d_info):
        return {}

    with pytest.raises(
        td.exceptions.AdjointError,
        match=(
            "CustomVJPConfig compute_derivatives function should accept two arguments "
            r"\(target, derivative_info\), and it currently accepts 3 arguments\."
        ),
    ):
        CustomVJPConfig(
            structure=td.PolySlab,
            compute_derivatives=polyslab_custom_vjp_bad_num_args,
        )

    with pytest.raises(
        td.exceptions.AdjointError,
        match=(
            "CustomVJPConfig compute_derivatives function second argument name is d_info "
            "but it should be derivative_info."
        ),
    ):
        CustomVJPConfig(
            structure=td.PolySlab,
            compute_derivatives=polyslab_custom_vjp_bad_arg_name,
        )


@pytest.mark.parametrize("structure_key, monitor_key", [("polyslab", "mode")])
@pytest.mark.parametrize("polyslab_axis", [0, 1, 2])
@pytest.mark.parametrize("run_function", [_run_local, run_custom])
@pytest.mark.parametrize("use_single_custom_vjp", [True, False])
@pytest.mark.parametrize("specify_custom_vjp_by_type", [True, False])
@pytest.mark.parametrize("local_gradient", [True, False])
def test_autograd_cm_custom_vjp(
    use_emulated_run,
    structure_key,
    monitor_key,
    polyslab_axis,
    run_function,
    use_single_custom_vjp,
    specify_custom_vjp_by_type,
    local_gradient,
):
    """Test that we can override a vjp with a user defined function in component modeler simulations."""

    fn_dict = get_functions(structure_key, monitor_key)
    make_sim = fn_dict["sim"]
    postprocess = fn_dict["postprocess"]

    def make_objective(custom_vjp_val):
        polyslab_custom_vjp = make_polyslab_custom_vjp(custom_vjp_val)

        structure_index = td.PolySlab if specify_custom_vjp_by_type else 1

        path_key = (
            "geometry",
            "vertices" if polyslab_axis == POLYSLAB_SELECT_VERTICES else "slab_bounds",
        )

        custom_vjp_tuple = (
            CustomVJPConfig(
                structure=structure_index,
                compute_derivatives=polyslab_custom_vjp,
                path_key=path_key,
            ),
        )

        custom_vjp_single = CustomVJPConfig(
            structure=structure_index,
            compute_derivatives=polyslab_custom_vjp,
        )

        custom_vjp_element = custom_vjp_single if use_single_custom_vjp else custom_vjp_tuple

        def objective(*args):
            base_sim = make_sim(*args, polyslab_axis=polyslab_axis)
            find_mode_monitors = [
                monitor for monitor in base_sim.monitors if isinstance(monitor, td.ModeMonitor)
            ]

            select_mode_monitor = find_mode_monitors[0]

            stripped_sim = base_sim.updated_copy(sources=[], monitors=[])

            input_port = Port(
                center=select_mode_monitor.center,
                size=select_mode_monitor.size,
                mode_spec=select_mode_monitor.mode_spec,
                direction="-",
                name="input_port",
            )

            modeler = ComponentModeler(
                simulation=stripped_sim,
                ports=[input_port],
                freqs=select_mode_monitor.freqs,
            )

            smatrix = run_function(
                modeler,
                custom_vjp=custom_vjp_element,
                local_gradient=local_gradient,
            )
            return np.sum(np.abs(smatrix.smatrix().values) ** 2)

        return objective

    custom_vjp_val = 1.0
    custom_vjp_val_scale = 10.0 * custom_vjp_val

    if not local_gradient:
        with pytest.raises(
            td.exceptions.AdjointError,
            match="custom_vjp specified for a remote gradient not supported.",
        ):
            val, grad = ag.value_and_grad(make_objective(custom_vjp_val))(params0)
    else:
        val, grad = ag.value_and_grad(make_objective(custom_vjp_val))(params0)
        val_scale, grad_scale = ag.value_and_grad(make_objective(custom_vjp_val_scale))(params0)

        assert np.isclose(
            np.sum(np.abs(grad * (custom_vjp_val_scale / custom_vjp_val) - grad_scale)), 0.0
        ), "Gradients were not set by the user vjp"


def test_verify_custom_vjp_ignores_source_traces():
    """Custom VJP validation must only consider traced structures, not traced sources."""

    def _dummy_vjp(_field, derivative_info):
        return dict.fromkeys(derivative_info.paths, 0.0)

    traced_fields = {("sources", 0, "current_dataset", "Ex"): np.array([1.0])}
    custom_vjp = (
        CustomVJPConfig(
            structure=0,
            compute_derivatives=_dummy_vjp,
            path_key=("geometry", "vertices"),
        ),
    )

    with pytest.raises(
        AdjointError,
        match="CustomVJPConfig structure index 0 not in traced structure indices.",
    ):
        verify_custom_vjp(custom_vjp, traced_fields)


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

        data = web.run(sim, task_name="autograd_test_numerical", verbose=False, local_gradient=True)
        value = postprocess(data)
        return value

    val, grad = ag.value_and_grad(objective)(params0)
    print(val, grad)
    assert anp.all(grad != 0.0), "some gradients are 0"

    # numerical gradients
    delta = 1e-1
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


def test_run_zero_grad(use_emulated_run):
    """Test warning if no adjoint sim is run (no adjoint sources).

    This checks the case where a simulation is still part of the computational
    graph (i.e. the output technically depends on the simulation),
    but no adjoint sources are placed because their amplitudes are zero and thus
    no adjoint simulation is run.
    """

    # only needs to be checked for one monitor
    fn_dict = get_functions(args[0][0], args[0][1])
    make_sim = fn_dict["sim"]
    postprocess = fn_dict["postprocess"]

    def objective(*args):
        sim = make_sim(*args)
        sim_data = run(sim, task_name="adjoint_test", verbose=False)
        return 0 * postprocess(sim_data)

    with AssertLogLevel("WARNING", contains_str="no sources"):
        grad = ag.grad(objective)(params0)


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
@pytest.mark.parametrize("use_task_names", [True, False])
def test_autograd_async(use_emulated_run, structure_key, monitor_key, use_task_names):
    """Test an objective function through tidy3d autograd."""

    fn_dict = get_functions(structure_key, monitor_key)
    make_sim = fn_dict["sim"]
    postprocess = fn_dict["postprocess"]

    task_names = {"test_a", "adjoint", "_test"}

    def objective(*args):
        if use_task_names:
            sims = {task_name: make_sim(*args) for task_name in task_names}
        else:
            sims = [make_sim(*args)] * len(task_names)
        batch_data = run_async(sims, verbose=False)
        value = 0.0
        for _, sim_data in batch_data.items():
            value += postprocess(sim_data)
        return value

    val, grad = ag.value_and_grad(objective)(params0)
    print(val, grad)
    assert anp.all(grad != 0.0), "some gradients are 0"


class TestTupleGrads:
    center0 = (0.0, 0.0, 0.0)
    size0 = (0.5, 1.0, 1.5)

    @staticmethod
    def make_simulation(center: tuple, size: tuple) -> td.Simulation:
        wavelength = 1.0
        freq0 = td.C_0 / wavelength

        src = td.PointDipole(
            center=(-1.4, 0, 0),
            source_time=td.GaussianPulse(freq0=freq0, fwidth=freq0 / 10),
            polarization="Ex",
        )

        mnt = td.FieldMonitor(
            size=(0, 0, 1),
            center=(1.4, 0, 0),
            freqs=[freq0, freq0 + freq0 / 50],
            name="fields",
        )

        scatterer = td.Structure(
            geometry=td.Box(center=center, size=size),
            medium=td.Medium(permittivity=3.0),
        )

        return td.Simulation(
            size=(3, 3, 3),
            run_time=2e-13,
            structures=[scatterer],
            sources=[src],
            monitors=[mnt],
            boundary_spec=td.BoundarySpec.all_sides(td.PML()),
            grid_spec=td.GridSpec.auto(min_steps_per_wvl=30),
        )

    @pytest.mark.parametrize("run_async", [False, True])
    @pytest.mark.parametrize("zero", [False, True])
    @pytest.mark.parametrize("local_gradient", [False, True])
    def test_zero_grad_tuple(self, use_emulated_run, run_async, zero, local_gradient, tmp_path):
        """Checks that tuple gradients don't return empty tuples"""

        def obj(center: tuple, size: tuple) -> float:
            sim = self.make_simulation(center=center, size=size)
            if run_async:
                batch_data = web.run_async(
                    {"lossy_test_async": sim},
                    path_dir=tmp_path,
                    local_gradient=local_gradient,
                )
                sim_data = list(batch_data.values())[0]
            else:
                sim_data = web.run(
                    sim,
                    task_name="lossy_test",
                    local_gradient=local_gradient,
                )
            objval = anp.mean(sim_data["fields"].intensity.data).item()
            if zero:
                objval *= 0
            return objval

        d_power = ag.value_and_grad(obj, argnum=(0, 1))
        val, (dp_dcenter, dp_dsize) = d_power(self.center0, self.size0)

        assert len(dp_dcenter) == 3
        assert len(dp_dsize) == 3

        if zero:
            assert np.allclose(dp_dcenter, 0)
            assert np.allclose(dp_dsize, 0)
        else:
            assert not np.allclose(dp_dcenter, 0)
            assert not np.allclose(dp_dsize, 0)


@pytest.mark.parametrize("structure_key, monitor_key", args)
def test_autograd_async_some_zero_grad(use_emulated_run, structure_key, monitor_key):
    """Test objective where only some simulations in batch have adjoint sources."""

    fn_dict = get_functions(structure_key, monitor_key)
    make_sim = fn_dict["sim"]
    postprocess = fn_dict["postprocess"]

    task_names = {"1", "2"}

    def objective(*args):
        sims = {task_name: make_sim(*args) for task_name in task_names}
        batch_data = run_async(sims, verbose=False)
        values = []
        for _, sim_data in batch_data.items():
            values.append(postprocess(sim_data))
        return min(values)

    val, grad = ag.value_and_grad(objective)(params0)

    assert anp.all(grad != 0.0), "some gradients are 0"


def test_autograd_async_all_zero_grad(use_emulated_run):
    """Test objective where no simulation in batch has adjoint sources."""

    fn_dict = get_functions(args[0][0], args[0][1])
    make_sim = fn_dict["sim"]
    postprocess = fn_dict["postprocess"]

    task_names = {"1", "2"}

    def objective(*args):
        sims = {task_name: make_sim(*args) for task_name in task_names}
        batch_data = run_async(sims, verbose=False)
        values = []
        for _, sim_data in batch_data.items():
            values.append(postprocess(sim_data))
        return 0 * sum(values)

    with AssertLogLevel("WARNING", contains_str="contains adjoint sources"):
        grad = ag.grad(objective)(params0)


def test_async_vjp_parallel_only_keeps_full_sim_field_keys(monkeypatch):
    """Ensure async VJP returns full per-task key coverage when only parallel contribution exists."""

    task_name = "task_parallel_only"
    sim_fields_original = {
        ("structures", 0, "geometry", "size"): np.array([1.0, 2.0]),
        ("structures", 0, "geometry", "center"): np.array([3.0, 4.0]),
    }

    def _fake_prepare_adjoints_from_vjp(**_kwargs):
        return (
            {("structures", 0, "geometry", "size"): np.array([5.0, 6.0])},
            [],
            True,
        )

    monkeypatch.setattr(
        autograd_module, "_prepare_adjoints_from_vjp", _fake_prepare_adjoints_from_vjp
    )

    vjp_fn = autograd_module._run_async_bwd(
        data_fields_original_dict={task_name: {("data", 0, "amps"): np.array([1.0])}},
        sim_fields_original_dict={task_name: sim_fields_original},
        sims_original={task_name: SIM_BASE},
        contexts={
            task_name: AutogradContext(
                simulation_data_original=object(),
            )
        },
        local_gradient=False,
        max_num_adjoint_per_fwd=1,
        numerical_structures={},
        custom_vjp=None,
    )

    vjp_result = vjp_fn({task_name: {("data", 0, "amps"): np.array([1.0])}})

    assert task_name in vjp_result
    assert set(vjp_result[task_name]) == set(sim_fields_original)
    assert np.allclose(vjp_result[task_name][("structures", 0, "geometry", "size")], [5.0, 6.0])
    assert np.allclose(vjp_result[task_name][("structures", 0, "geometry", "center")], [0.0, 0.0])


def test_async_vjp_sequential_path_keeps_full_sim_field_keys(monkeypatch):
    """Ensure async VJP keeps full key coverage when sequential adjoints return sparse maps."""

    task_name = "task_sequential_sparse"
    sim_fields_original = {
        ("structures", 0, "geometry", "size"): np.array([1.0, 2.0]),
        ("structures", 0, "geometry", "center"): np.array([3.0, 4.0]),
    }

    def _fake_prepare_adjoints_from_vjp(**_kwargs):
        return ({}, [SIM_BASE], True)

    def _fake_run_async_tidy3d_bwd(simulations, **_kwargs):
        assert f"{task_name}_adjoint_0" in simulations
        return {
            f"{task_name}_adjoint_0": {("structures", 0, "geometry", "size"): np.array([9.0, 10.0])}
        }

    monkeypatch.setattr(
        autograd_module, "_prepare_adjoints_from_vjp", _fake_prepare_adjoints_from_vjp
    )
    monkeypatch.setattr(autograd_module, "_run_async_tidy3d_bwd", _fake_run_async_tidy3d_bwd)

    vjp_fn = autograd_module._run_async_bwd(
        data_fields_original_dict={task_name: {("data", 0, "amps"): np.array([1.0])}},
        sim_fields_original_dict={task_name: sim_fields_original},
        sims_original={task_name: SIM_BASE},
        contexts={
            task_name: AutogradContext(
                simulation_data_original=object(),
                forward_task_id="fwd_task_id",
            )
        },
        local_gradient=False,
        max_num_adjoint_per_fwd=1,
        numerical_structures={},
        custom_vjp=None,
    )

    vjp_result = vjp_fn({task_name: {("data", 0, "amps"): np.array([1.0])}})

    assert task_name in vjp_result
    assert set(vjp_result[task_name]) == set(sim_fields_original)
    assert np.allclose(vjp_result[task_name][("structures", 0, "geometry", "size")], [9.0, 10.0])
    assert np.allclose(vjp_result[task_name][("structures", 0, "geometry", "center")], [0.0, 0.0])


def test_vjp_sequential_path_keeps_full_sim_field_keys(monkeypatch):
    """Ensure single-run VJP keeps full key coverage when sequential adjoints return sparse maps."""

    task_name = "task_single_sequential_sparse"
    sim_fields_original = {
        ("structures", 0, "geometry", "size"): np.array([1.0, 2.0]),
        ("structures", 0, "geometry", "center"): np.array([3.0, 4.0]),
    }

    def _fake_prepare_adjoints_from_vjp(**_kwargs):
        return ({}, [SIM_BASE], True)

    def _fake_run_async_tidy3d_bwd(simulations, **_kwargs):
        assert f"{task_name}_adjoint_0" in simulations
        return {
            f"{task_name}_adjoint_0": {("structures", 0, "geometry", "size"): np.array([9.0, 10.0])}
        }

    monkeypatch.setattr(
        autograd_module, "_prepare_adjoints_from_vjp", _fake_prepare_adjoints_from_vjp
    )
    monkeypatch.setattr(autograd_module, "_run_async_tidy3d_bwd", _fake_run_async_tidy3d_bwd)

    vjp_fn = autograd_module._run_bwd(
        data_fields_original={("data", 0, "amps"): np.array([1.0])},
        sim_fields_original=sim_fields_original,
        sim_original=SIM_BASE,
        task_name=task_name,
        context=AutogradContext(
            simulation_data_original=object(),
            forward_task_id="fwd_task_id",
        ),
        local_gradient=False,
        max_num_adjoint_per_fwd=1,
        numerical_structures={},
        custom_vjp=None,
    )

    vjp_result = vjp_fn({("data", 0, "amps"): np.array([1.0])})

    assert set(vjp_result) == set(sim_fields_original)
    assert np.allclose(vjp_result[("structures", 0, "geometry", "size")], [9.0, 10.0])
    assert np.allclose(vjp_result[("structures", 0, "geometry", "center")], [0.0, 0.0])


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
        structures = tuple(num_structures_test * [structure])
        return SIM_BASE.updated_copy(structures=structures, monitors=(monitor,))

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

    t0 = 1.0
    axis = 0

    num_pts = 819

    monitor, postprocess = make_monitors()[monitor_key]

    def make_cylinder(radius, x0, y0, t):
        return td.Cylinder(
            center=td.Cylinder.unpop_axis(0.0, (x0, y0), axis=axis),
            radius=radius,
            length=t,
            axis=axis,
        ).to_polyslab(num_pts)

    def make_polyslab(radius, x0, y0, t):
        phis = anp.linspace(0, 2 * np.pi, num_pts + 1)[:-1]

        xs = radius * anp.cos(phis) + x0
        ys = radius * anp.sin(phis) + y0

        vertices = anp.stack((xs, ys), axis=-1)

        return td.PolySlab(
            vertices=vertices,
            axis=axis,
            slab_bounds=(-t / 2, t / 2),
        )

    def make_sim(params, geo_maker):
        geo = geo_maker(*params)
        structure = td.Structure(geometry=geo, medium=td.Medium(permittivity=2))

        return SIM_BASE.updated_copy(structures=(structure,), monitors=(monitor,))

    p0 = [1.0, 0.0, 0.0, t0]

    def objective_polyslab(params):
        """Objective function."""
        sim = make_sim(params, geo_maker=make_polyslab)
        if PLOT_SIM:
            plot_sim(sim, plot_eps=True)
        data = run(sim, task_name="autograd_test", verbose=False)
        return anp.sum(anp.abs(data[monitor.name].amps)).item()

    val_polyslab, grad_polyslab = ag.value_and_grad(objective_polyslab)(p0)
    print(val_polyslab, grad_polyslab)
    assert anp.all(grad_polyslab != 0.0), "some gradients are 0"

    def objective_cylinder(params):
        """Objective function."""
        sim = make_sim(params, geo_maker=make_cylinder)
        if PLOT_SIM:
            plot_sim(sim, plot_eps=True)
        data = run(sim, task_name="autograd_test", verbose=False)
        return anp.sum(anp.abs(data[monitor.name].amps)).item()

    val_cylinder, grad_cylinder = ag.value_and_grad(objective_cylinder)(p0)
    print(val_cylinder, grad_cylinder)
    assert anp.all(grad_cylinder != 0.0), "some gradients are 0"


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
    assert np.all(np.abs(grad) > 0), "some gradients are 0"


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
    assert np.all(np.abs(grad) > 0), "some gradients are 0"


@pytest.mark.parametrize("structure_key", ("custom_med",))
def test_sim_full_ops(structure_key):
    """make sure the autograd operations don't error on a simulation containing everything."""

    def objective(*params):
        s = make_structures(*params)[structure_key]
        s = s.updated_copy(geometry=s.geometry.updated_copy(center=(2, 2, 2), size=(0, 0, 0)))
        sim_full_traced = SIM_FULL.updated_copy(structures=(*SIM_FULL.structures, s))

        sim_full_static = sim_full_traced.to_static()

        sim_fields = sim_full_traced._strip_traced_fields()

        # note: there is one traced structure in SIM_FULL already with 6 fields + 1 = 7
        assert len(sim_fields) == 10

        sim_traced = sim_full_static._insert_traced_fields(sim_fields)

        assert sim_traced == sim_full_traced

        return anp.sum(sim_full_traced.structures[-1].medium.permittivity.values)

    ag.grad(objective)(params0)


def test_sim_hash_changes_with_traced_keys():
    """Ensure the model hash accounts for autograd traced paths."""

    sim_traced = SIM_FULL.copy()
    original_field_map = sim_traced._strip_traced_fields()

    structures = list(sim_traced.structures)
    structures[0] = structures[0].to_static()
    sim_modified = sim_traced.updated_copy(structures=tuple(structures))

    modified_field_map = sim_modified._strip_traced_fields()
    assert original_field_map != modified_field_map
    assert sim_traced._hash_self() != sim_modified._hash_self()


def test_sim_hdf5_records_traced_keys(tmp_path):
    """HDF5 exports should include traced-key metadata for caching."""

    sim_traced = SIM_FULL.copy()
    expected_payload = sim_traced._serialized_traced_field_keys()
    assert expected_payload, "simulation fixture must yield traced keys"

    sim_traced.attrs.pop(TRACED_FIELD_KEYS_ATTR, None)

    export_path = tmp_path / "sim_traced.hdf5"
    sim_traced.to_hdf5(str(export_path))

    with h5py.File(export_path, "r") as handle:
        assert TRACED_FIELD_KEYS_ATTR in handle.attrs
        assert handle.attrs[TRACED_FIELD_KEYS_ATTR] == expected_payload

    static_export = tmp_path / "sim_traced_static.hdf5"
    sim_traced.attrs[TRACED_FIELD_KEYS_ATTR] = expected_payload
    sim_static = sim_traced.to_static()
    sim_static.to_hdf5(str(static_export))

    with h5py.File(static_export, "r") as handle:
        assert TRACED_FIELD_KEYS_ATTR in handle.attrs
        assert handle.attrs[TRACED_FIELD_KEYS_ATTR] == expected_payload


def test_web_run_duplicate_simulations(monkeypatch):
    """Repeated simulation objects should reuse cached data without hash mismatches."""

    sim = SIM_FULL.copy()
    sim.attrs.pop(TRACED_FIELD_KEYS_ATTR, None)

    copy_calls = {"count": 0}

    class DummyData:
        def __init__(self, label: str):
            self.label = label

        def copy(self):
            copy_calls["count"] += 1
            return DummyData(f"{self.label}_copy{copy_calls['count']}")

    dummy = DummyData("root")

    def fake_run_autograd(*args, **kwargs):
        return dummy

    monkeypatch.setattr("tidy3d.web.api.run.run_autograd", fake_run_autograd)

    results = web.run([sim, sim])

    assert isinstance(results, list)
    assert len(results) == 2
    assert results[0] is dummy
    assert results[1] is not dummy
    assert copy_calls["count"] == 1


def test_autograd_run_does_not_mutate_input_attrs(monkeypatch):
    """Autograd run should attach traced metadata only to the exported static copy."""

    sim = SIM_FULL.copy()
    sim.attrs.pop(TRACED_FIELD_KEYS_ATTR, None)
    payload = sim._serialized_traced_field_keys()
    assert payload

    captured: dict[str, typing.Any] = {}

    def fake_run_primitive(
        sim_fields,
        sim_original,
        task_name,
        context,
        local_gradient,
        max_num_adjoint_per_fwd,
        **run_kwargs,
    ):
        captured["sim_original"] = sim_original
        captured["payload"] = sim_original.attrs.get(TRACED_FIELD_KEYS_ATTR)
        captured["sim_fields"] = sim_fields
        captured["context"] = context
        return sim_fields

    def fake_postprocess_run(traced_fields_data, context):
        captured["postprocess_data"] = traced_fields_data
        captured["postprocess_context"] = context
        return "sentinel"

    monkeypatch.setattr(autograd_module, "_run_primitive", fake_run_primitive)
    monkeypatch.setattr(autograd_module, "postprocess_run", fake_postprocess_run)

    result = autograd_module._run(simulation=sim, task_name="dummy")

    assert result == "sentinel"
    assert sim.attrs.get(TRACED_FIELD_KEYS_ATTR) is None
    assert captured["payload"] == payload
    assert captured["sim_original"] is not sim
    assert captured["sim_original"].attrs.get(TRACED_FIELD_KEYS_ATTR) == payload
    assert captured["postprocess_data"] == captured["sim_fields"]
    assert captured["postprocess_context"] is captured["context"]
    assert isinstance(captured["postprocess_context"], AutogradContext)
    assert captured["postprocess_context"].simulation_data_original is None
    assert captured["postprocess_context"].simulation_data_forward is None
    assert captured["postprocess_context"].forward_task_id is None
    assert captured["postprocess_context"].parallel_adjoint_state is None


def test_sim_traced_override_structures():
    """Make sure that sims with traced override structures are handled properly."""

    def f(x):
        override_structure = td.MeshOverrideStructure(
            geometry=td.Box(center=(0, 0, 0), size=(1, 1, x)),
            dl=[1, 1, 1],
        )
        sim = SIM_FULL.updated_copy(override_structures=(override_structure,), path="grid_spec")
        return sim.grid_spec.override_structures[0].geometry.size[2]

    with AssertLogLevel("WARNING", contains_str="override structures"):
        ag.grad(f)(1.0)


@pytest.mark.parametrize("structure_key", ("custom_med",))
def test_sim_fields_io(structure_key, tmp_path):
    """Test that converging and AutogradFieldMap dictionary to a FieldMap object, saving and loading
    from file, and then converting back, returns the same object."""
    s = make_structures(params0)[structure_key]
    s = s.updated_copy(geometry=s.geometry.updated_copy(center=(2, 2, 2), size=(0, 0, 0)))
    sim_full_traced = SIM_FULL.updated_copy(structures=(*SIM_FULL.structures, s))
    sim_fields = sim_full_traced._strip_traced_fields()

    field_map = FieldMap.from_autograd_field_map(sim_fields)
    field_map_file = join(tmp_path, "test_sim_fields.hdf5.gz")
    field_map.to_file(field_map_file)
    autograd_field_map = FieldMap.from_file(field_map_file).to_autograd_field_map
    for path, data in sim_fields.items():
        assert np.all(data == autograd_field_map[path])


def test_field_map_complex_array_dict_io(tmp_path):
    """Complex array tracer payloads should roundtrip through FieldMap IO."""
    payload = {
        "tracers": (
            {
                "path": ("sources", 0, "field_dataset", "Ex", "values"),
                "data": {"real": [[1.0, 2.0]], "imag": [[3.0, 4.0]]},
            },
        )
    }
    field_map = FieldMap.model_validate(payload)
    field_map_file = join(tmp_path, "test_complex_array_field_map.hdf5.gz")
    field_map.to_file(field_map_file)

    reloaded = FieldMap.from_file(field_map_file)
    autograd_field_map = reloaded.to_autograd_field_map

    expected = np.array([[1.0 + 3.0j, 2.0 + 4.0j]])
    np.testing.assert_allclose(reloaded.tracers[0].data, expected)
    np.testing.assert_allclose(
        autograd_field_map[("sources", 0, "field_dataset", "Ex", "values")], expected
    )


def test_web_incompatible_inputs(monkeypatch):
    """Test what happens when bad inputs passed to web.run()."""

    def catch(*args, **kwargs):
        """Just raise an exception."""
        raise AssertionError

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


def test_too_many_traced_structures(monkeypatch, use_emulated_run):
    """More traced structures than allowed."""

    monitor_key = "mode"
    structure_key = "size_element"
    monitor, postprocess = make_monitors()[monitor_key]

    def make_sim(*args):
        structure = make_structures(*args)[structure_key]
        return SIM_BASE.updated_copy(
            structures=(config.adjoint.max_traced_structures + 1) * (structure,),
            monitors=(monitor,),
        )

    def objective(*args):
        """Objective function."""
        sim = make_sim(*args)
        data = run(sim, task_name="autograd_test", verbose=False)
        value = postprocess(data, data[monitor_key])
        return value

    with pytest.raises(ValueError):
        ag.grad(objective)(params0)


def test_no_freq_adjoint(monkeypatch, use_emulated_run):
    """No frequency adjoint."""

    def objective(args):
        structures_traced_dict = make_structures(args)
        structures = list(SIM_BASE.structures)

        for structure_key in structure_keys_:
            structures.append(structures_traced_dict[structure_key])

        sim = SIM_BASE.updated_copy(
            structures=structures,
            monitors=(td.FieldTimeMonitor(size=(0, 0, 0), name="time_monitor_only"),),
        )
        # doesn't need to be a valid objective since this should error when calling web.run
        return web.run(sim, task_name="autograd_test", verbose=False)

    with pytest.raises(AdjointError, match="No frequency-domain data"):
        ag.grad(objective)(params0)


def _make_mock_combined_sim_data(
    sim_original: td.Simulation,
    task_name: str,
    include_eps_monitor: bool = True,
) -> td.SimulationData:
    """Build deterministic combined SimulationData with named adjoint monitors."""
    sim_data_original = run_emulated(sim_original, task_name=task_name)
    template_monitor = sim_original.monitors[0]
    template_data = sim_data_original.data[0]

    adjoint_monitors = [template_monitor.updated_copy(name="adjoint_fld_0")]
    adjoint_data = [template_data.updated_copy(monitor=adjoint_monitors[0])]

    if include_eps_monitor:
        adjoint_monitors.append(template_monitor.updated_copy(name="adjoint_eps_0"))
        adjoint_data.append(template_data.updated_copy(monitor=adjoint_monitors[-1]))

    sim_combined = sim_original.updated_copy(monitors=(*sim_original.monitors, *adjoint_monitors))
    sim_data_combined = sim_data_original.updated_copy(
        simulation=sim_combined,
        data=(*sim_data_original.data, *adjoint_data),
    )
    return sim_data_combined


def test_split_adjoint_data_warns_on_missing_original_monitor_data():
    """Warn if combined SimulationData does not contain all expected original monitor data."""
    monitor, _ = make_monitors()["field_point"]
    structure = make_structures(params0)["custom_med"]
    sim_original = SIM_BASE.updated_copy(
        structures=(*SIM_BASE.structures, structure),
        monitors=(*SIM_BASE.monitors, monitor),
    )
    sim_data_combined = _make_mock_combined_sim_data(
        sim_original, task_name="missing_original_monitor_data"
    )

    removed_name = sim_original.monitors[0].name
    sim_data_missing = sim_data_combined.updated_copy(
        data=tuple(data for data in sim_data_combined.data if data.monitor.name != removed_name)
    )

    with AssertLogLevel("WARNING", contains_str="missing expected original monitor data"):
        sim_data_missing._split_adjoint_data(num_mnts_original=len(sim_original.monitors))


def test_split_adjoint_data_warns_on_missing_adjoint_monitor_data():
    """Warn if combined SimulationData does not contain all expected adjoint monitor data."""
    monitor, _ = make_monitors()["field_point"]
    structure = make_structures(params0)["custom_med"]
    sim_original = SIM_BASE.updated_copy(
        structures=(*SIM_BASE.structures, structure),
        monitors=(*SIM_BASE.monitors, monitor),
    )
    sim_data_combined = _make_mock_combined_sim_data(
        sim_original, task_name="missing_adjoint_monitor_data"
    )

    removed_name = "adjoint_fld_0"
    sim_data_missing = sim_data_combined.updated_copy(
        data=tuple(data for data in sim_data_combined.data if data.monitor.name != removed_name)
    )

    with AssertLogLevel("WARNING", contains_str="missing expected adjoint monitor data"):
        sim_data_missing._split_adjoint_data(num_mnts_original=len(sim_original.monitors))


def test_split_adjoint_data_warns_on_unordered_monitor_data():
    """Warn if monitor data order differs from combined simulation monitor order."""
    monitor, _ = make_monitors()["field_point"]
    structure = make_structures(params0)["custom_med"]
    sim_original = SIM_BASE.updated_copy(
        structures=(*SIM_BASE.structures, structure),
        monitors=(*SIM_BASE.monitors, monitor),
    )
    sim_data_combined = _make_mock_combined_sim_data(
        sim_original, task_name="unordered_monitor_data"
    )

    assert len(sim_data_combined.data) >= 2, (
        "Expected at least two monitor datasets to test ordering."
    )
    data_unordered = tuple(reversed(sim_data_combined.data))
    sim_data_unordered = sim_data_combined.updated_copy(data=data_unordered)

    with AssertLogLevel(
        "WARNING",
        contains_str="monitor data order does not match combined simulation monitor order",
    ):
        sim_data_unordered._split_adjoint_data(num_mnts_original=len(sim_original.monitors))


def test_split_adjoint_data_preserves_raw_monitor_data_under_symmetry():
    """Regression test: _split_adjoint_data should not apply symmetry_expanded_copy via __getitem__."""
    sim_symmetry = SIM_BASE.updated_copy(symmetry=(0, 1, 0))
    sim_data = run_emulated(sim_symmetry, task_name="split_adjoint_data_raw_symmetry")

    data_original, _ = sim_data._split_adjoint_data(num_mnts_original=len(sim_symmetry.monitors))

    monitor_name = sim_symmetry.monitors[0].name
    raw_monitor_data = sim_data.monitor_data[monitor_name]
    split_monitor_data = next(data for data in data_original if data.monitor.name == monitor_name)

    assert split_monitor_data is raw_monitor_data
    assert sim_data[monitor_name] is not raw_monitor_data


def test_adjoint_src_width():
    """Test the adjoint source width for single sources decays by f=0."""

    f0 = td.C_0 / 1.55
    fwidth = f0

    fwidths = f0 * np.linspace(0.1, 1.0, 5)

    adj_srcs = [
        td.PointDipole(
            center=(0, 0, 0),
            source_time=td.GaussianPulse(freq0=f0, fwidth=fwidth),
            polarization="Ex",
        )
        for fwidth in fwidths
    ]

    adj_srcs_fwidth = td.SimulationData._adjoint_src_width_single(adj_srcs)

    for src in adj_srcs_fwidth:
        assert np.isclose((src.source_time.freq0 - f0) / f0, 0.0), (
            "f0 of adjoint source should be centered on original f0"
        )

        check_fwidth = (
            src.source_time.freq0
            - td.components.data.sim_data.NUM_ADJOINT_FWIDTH_TO_ZERO * src.source_time.fwidth
        ) / src.source_time.freq0

        assert np.isclose(check_fwidth, 0.0) or (check_fwidth > 0.0), (
            "fwidth of adjoint source should decay sufficiently before f=0"
        )


def test_broadband_adjoint_src_width():
    """Test the broadband adjoint source handling for choosing fwidth."""

    # Test the case where we have a custom current source and a wide adjoint source width that overlaps with zero.
    # In this case, we want to issue a warning to the user about the adjoint accuracy of this setup.
    f0_high = td.C_0 / 1.55
    f0_low = 0.1 * f0_high

    f0_adj_all = [f0_low, f0_high]

    fwidth = 0.1 * f0_high

    adj_srcs = []
    x = np.array([0.0])
    y = np.array([0.0])
    z = np.array([0.0])
    for f0 in f0_adj_all:
        f = np.array([f0])

        coords = {"x": x, "y": y, "z": z, "f": f}

        dataset = td.FieldDataset(Ex=td.ScalarFieldDataArray(np.ones((1, 1, 1, 1)), coords=coords))

        adj_srcs.append(
            td.CustomCurrentSource(
                center=(0, 0, 0),
                size=(0, 0, 0),
                source_time=td.GaussianPulse(freq0=f0, fwidth=fwidth),
                current_dataset=dataset,
            )
        )

    EXPECTED_WARNING_MSG_PIECE = (
        "Adjoint source generated with a frequency spectrum that extends to or overlaps with 0 Hz"
    )
    with AssertLogLevel("WARNING", contains_str=EXPECTED_WARNING_MSG_PIECE):
        broadband_f0, broadband_fwidth = td.SimulationData._adjoint_src_width_broadband(adj_srcs)

        f0_expected = 0.5 * (np.max(f0_adj_all) + np.min(f0_adj_all))

        fwidth_expected = (
            f0_expected - np.min(f0_adj_all)
        ) / td.components.data.sim_data.NUM_ADJOINT_FWIDTH_TO_FMIN

        assert np.isclose((f0_expected - broadband_f0) / f0_expected, 0.0), (
            "Expected freq0 not matching for broadband source"
        )
        assert np.isclose((fwidth_expected - broadband_fwidth) / fwidth_expected, 0.0), (
            "Expected fwidth not matching for broadband source"
        )

    # Test the case where we need a wider pulse to cover all the adjoint frequencies than we would otherwise choose for
    # each individual adjoint source
    f0_broadband = np.linspace(f0_low, f0_high, 10)
    fwidth_broadband = 0.1 * np.mean(f0_broadband)

    adj_srcs = [
        td.PointDipole(
            center=(0, 0, 0),
            source_time=td.GaussianPulse(freq0=f0, fwidth=fwidth_broadband),
            polarization="Ex",
        )
        for f0 in f0_broadband
    ]

    broadband_f0, broadband_fwidth = td.SimulationData._adjoint_src_width_broadband(adj_srcs)

    f0_expected = 0.5 * (np.max(f0_broadband) + np.min(f0_broadband))
    fwidth_expected = (
        f0_expected - np.min(f0_broadband)
    ) / td.components.data.sim_data.NUM_ADJOINT_FWIDTH_TO_FMIN

    assert np.isclose((f0_expected - broadband_f0) / f0_expected, 0.0), (
        "Expected freq0 not matching for broadband source"
    )
    assert np.isclose((fwidth_expected - broadband_fwidth) / fwidth_expected, 0.0), (
        "Expected fwidth not matching for broadband source"
    )

    # Test the case where we have a narrow set of frequencies for the adjoint sources and so we can
    # choose a wider overall source than is needed for covering those frequencies. This larger pulse width
    # in frequency will shorten the time pulse.
    f0_broadband = np.linspace(0.95 * f0_high, 1.05 * f0_high, 10)
    fwidth_broadband = 0.1 * np.mean(f0_broadband)

    adj_srcs = [
        td.PointDipole(
            center=(0, 0, 0),
            source_time=td.GaussianPulse(freq0=f0, fwidth=fwidth_broadband),
            polarization="Ex",
        )
        for f0 in f0_broadband
    ]

    broadband_f0, broadband_fwidth = td.SimulationData._adjoint_src_width_broadband(adj_srcs)

    f0_expected = 0.5 * (np.max(f0_broadband) + np.min(f0_broadband))
    fwidth_expected = f0_expected / td.components.data.sim_data.NUM_ADJOINT_FWIDTH_TO_ZERO

    assert np.isclose((f0_expected - broadband_f0) / f0_expected, 0.0), (
        "Expected freq0 not matching for broadband source"
    )
    assert np.isclose((fwidth_expected - broadband_fwidth) / fwidth_expected, 0.0), (
        "Expected fwidth not matching for broadband source"
    )


def test_autograd_multi_source_normalize_index(use_emulated_run):
    """Gradient run with multi-source normalization does not raise and returns finite grads."""

    freq0 = 2e14
    fwidth = 5e13

    source_time = td.GaussianPulse(freq0=freq0, fwidth=fwidth)
    source0 = td.UniformCurrentSource(
        size=(0, 0, 0),
        center=(0.0, -0.2, 0.0),
        polarization="Hx",
        source_time=source_time,
        current_amplitude_definition="total",
    )
    source1 = source0.updated_copy(center=(0.0, 0.2, 0.0))

    monitor = td.FieldMonitor(
        size=(0, 0, 0),
        center=(0.0, 0.0, 0.0),
        freqs=[freq0],
        fields=["Ex"],
        name="field",
    )

    base_sim = td.Simulation(
        size=(1.0, 1.0, 1.0),
        run_time=8 / fwidth,
        grid_spec=td.GridSpec.uniform(dl=0.1),
        sources=[source0, source1],
        monitors=[monitor],
        normalize_index=1,
    )

    def objective(params):
        eps = 2.0 + params[0]
        structure = td.Structure(
            geometry=td.Box(size=(0.5, 0.5, 0.5), center=(0.0, 0.0, 0.0)),
            medium=td.Medium(permittivity=eps),
        )

        sim = base_sim.updated_copy(structures=[structure])
        data = run(sim, task_name="normalize_index_clamp", verbose=False)
        field = data["field"].Ex.sel(f=freq0).values
        return anp.real(field).sum()

    params = anp.array([0.1])
    val, grad = ag.value_and_grad(objective)(params)

    assert anp.isfinite(val)
    assert grad.shape == params.shape
    assert anp.all(anp.isfinite(grad))


def test_gaussian_broadband_source_num_freqs_selection():
    """Gaussian broadband grouping uses threshold constant for num_freqs selection."""

    f0 = FREQ0
    base_sim = td.Simulation(
        size=(1.0, 1.0, 1.0),
        run_time=1e-12,
        grid_spec=td.GridSpec.uniform(dl=0.1),
        sources=[
            td.PointDipole(
                center=(0.0, 0.0, 0.0),
                polarization="Ex",
                source_time=td.GaussianPulse(freq0=f0, fwidth=0.1 * f0),
            )
        ],
        monitors=[],
    )
    sim_data = td.SimulationData(simulation=base_sim, data=())
    bandwidth_threshold = td.components.data.sim_data.GAUSSIAN_WIDE_BANDWIDTH_THRESHOLD
    narrow_delta = 0.45 * bandwidth_threshold
    wide_delta = 0.55 * bandwidth_threshold

    def make_gaussian_source(freq: float) -> td.GaussianBeam:
        return td.GaussianBeam(
            center=(0.0, 0.0, 0.0),
            size=(1.0, 1.0, 0.0),
            source_time=td.GaussianPulse(freq0=freq, fwidth=0.05 * f0),
            direction="+",
            waist_radius=0.5,
            pol_angle=np.pi / 2,
        )

    def make_astig_source(freq: float) -> td.AstigmaticGaussianBeam:
        return td.AstigmaticGaussianBeam(
            center=(0.0, 0.0, 0.0),
            size=(1.0, 1.0, 0.0),
            source_time=td.GaussianPulse(freq0=freq, fwidth=0.05 * f0),
            direction="+",
            waist_sizes=(0.5, 0.4),
            waist_distances=(0.0, 0.1),
            pol_angle=np.pi / 2,
        )

    # Single-frequency case -> keep num_freqs=1.
    src_single = sim_data._make_broadband_source([make_gaussian_source(f0)])
    assert isinstance(src_single, td.GaussianBeam)
    assert src_single.num_freqs == 1

    # Grouped span <= threshold of center frequency -> keep num_freqs=1.
    src_narrow = sim_data._make_broadband_source(
        [
            make_gaussian_source((1 - narrow_delta) * f0),
            make_gaussian_source((1 + narrow_delta) * f0),
        ]
    )
    assert isinstance(src_narrow, td.GaussianBeam)
    assert src_narrow.num_freqs == 1

    # Grouped span > threshold of center frequency -> set num_freqs=3.
    src_wide = sim_data._make_broadband_source(
        [make_gaussian_source((1 - wide_delta) * f0), make_gaussian_source((1 + wide_delta) * f0)]
    )
    assert isinstance(src_wide, td.GaussianBeam)
    assert src_wide.num_freqs == 3

    # Astigmatic GaussianBeam follows the same rule.
    src_astig_single = sim_data._make_broadband_source([make_astig_source(f0)])
    assert isinstance(src_astig_single, td.AstigmaticGaussianBeam)
    assert src_astig_single.num_freqs == 1

    src_astig_wide = sim_data._make_broadband_source(
        [make_astig_source((1 - wide_delta) * f0), make_astig_source((1 + wide_delta) * f0)]
    )
    assert isinstance(src_astig_wide, td.AstigmaticGaussianBeam)
    assert src_astig_wide.num_freqs == 3


def test_make_post_norm_amps_deduplicates_equal_frequencies():
    """Repeated same-frequency broadband-source normalizations collapse to one entry."""

    freq0 = 2e14
    source0 = td.PointDipole(
        center=(0.0, 0.0, 0.0),
        polarization="Ex",
        source_time=td.GaussianPulse(freq0=freq0, fwidth=0.1 * freq0, amplitude=2.0, phase=0.3),
    )
    source1 = source0.updated_copy(center=(0.1, 0.0, 0.0))

    post_norm = td.SimulationData._make_post_norm_amps([source0, source1])

    assert np.allclose(post_norm.coords["f"].values, [freq0])
    assert np.allclose(post_norm.values, [source0.source_time.amplitude * np.exp(1j * 0.3)])


def test_make_post_norm_amps_rejects_conflicting_same_frequency_values():
    """Repeated same-frequency broadband-source normalizations must agree."""

    freq0 = 2e14
    source0 = td.PointDipole(
        center=(0.0, 0.0, 0.0),
        polarization="Ex",
        source_time=td.GaussianPulse(freq0=freq0, fwidth=0.1 * freq0, amplitude=1.0, phase=0.0),
    )
    source1 = source0.updated_copy(
        center=(0.1, 0.0, 0.0),
        source_time=td.GaussianPulse(freq0=freq0, fwidth=0.1 * freq0, amplitude=2.0, phase=0.0),
    )

    with pytest.raises(AdjointError, match="conflicting post-normalization values"):
        td.SimulationData._make_post_norm_amps([source0, source1])


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

        sim = SIM_BASE.updated_copy(monitors=(monitor,), structures=tuple(structures))
        data = run(sim, task_name="autograd_test", verbose=False)

        if objtype == "flux":
            return data[monitor.name].flux.item()
        elif objtype == "intensity":
            return anp.sum(data.get_intensity(monitor.name).values)

    grads = ag.grad(objective)(params0)
    assert np.any(grads > 0)


@pytest.mark.parametrize("far_field_approx", [True, False])
@pytest.mark.parametrize("projection_type", ["angular", "cartesian", "kspace"])
@pytest.mark.parametrize("sim_2d", [True, False])
class TestFieldProjection:
    @staticmethod
    def setup(far_field_approx, projection_type, sim_2d):
        if (sim_2d or not IS_3D) and not far_field_approx:
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
            monitor_far = td.FieldProjectionAngleMonitor(
                center=monitor.center,
                size=monitor.size,
                freqs=monitor.freqs,
                phi=(np.pi / 2, 3 * np.pi / 2),
                theta=tuple(theta_proj),
                proj_distance=r_proj,
                far_field_approx=far_field_approx,
                name="far_field",
            )
        elif projection_type == "cartesian":
            y_proj = np.linspace(-10, 10, 3)
            monitor_far = td.FieldProjectionCartesianMonitor(
                center=monitor.center,
                size=monitor.size,
                freqs=monitor.freqs,
                x=[0],
                y=y_proj,
                proj_axis=1,
                proj_distance=r_proj,
                far_field_approx=far_field_approx,
                name="far_field",
            )
        elif projection_type == "kspace":
            uy = np.linspace(-0.7, 0.7, 3)
            monitor_far = td.FieldProjectionKSpaceMonitor(
                center=monitor.center,
                size=monitor.size,
                freqs=monitor.freqs,
                ux=[0],
                uy=uy,
                proj_axis=1,
                proj_distance=r_proj,
                far_field_approx=far_field_approx,
                name="far_field",
            )

        sim = SIM_BASE.updated_copy(monitors=(monitor,))

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

            sim = sim_base.updated_copy(structures=tuple(structures))
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

    def test_error_if_server_side_projection(
        self, use_emulated_run, far_field_approx, projection_type, sim_2d
    ):
        """Using a far field monitor directly should error"""
        # build a projection‐only monitor sim
        sim_base, monitor_far = self.setup(far_field_approx, projection_type, sim_2d)
        sim_base = sim_base.updated_copy(monitors=(monitor_far,))

        def objective(args):
            structures_traced_dict = make_structures(args)
            structures = list(SIM_BASE.structures)
            for structure_key in structure_keys_:
                structures.append(structures_traced_dict[structure_key])
            sim = sim_base.updated_copy(structures=tuple(structures))
            sim_data = run(sim, task_name="field_projection_test")
            return sim_data["far_field"].power.sum().item()

        with pytest.raises(NotImplementedError):
            ag.grad(objective)(params0)


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

    # Wrap the scalar as a DataArray to match expected return type
    import xarray as xr

    dJ_deps_array = xr.DataArray([dJ_deps], dims=["f"], coords={"f": [freq]})

    monkeypatch.setattr(
        td.PoleResidue,
        "_derivative_eps_complex_volume",
        lambda self, E_der_map, bounds: dJ_deps_array,
    )

    import importlib

    importlib.reload(td)

    poles = [(-p, p), (-2 * p, 2 * p)]
    pr = td.PoleResidue(eps_inf=2.0, poles=poles)
    field_paths = [("eps_inf",)]
    for i in range(len(poles)):
        for j in range(2):
            field_paths.append(("poles", i, j))

    eps_keys = ["eps_xx", "eps_yy", "eps_zz"]

    info = DerivativeInfo(
        paths=field_paths,
        E_der_map={},
        D_der_map={},
        E_fwd={},
        D_fwd={},
        E_adj={},
        D_adj={},
        eps_data={
            key: td.ScalarFieldDataArray(
                [[[[2.0]]]], coords={"x": [0], "y": [0], "z": [0], "f": [200e12]}
            )
            for key in eps_keys
        },
        frequencies=[freq],
        bounds=((-1, -1, -1), (1, 1, 1)),
        bounds_intersect=((-1, -1, -1), (1, 1, 1)),
        simulation_bounds=((-2, -2, -2), (2, 2, 2)),
        updated_epsilon=lambda geom: td.ScalarFieldDataArray(
            [[[[1.0]]]], coords={"x": [0], "y": [0], "z": [0], "f": [1.94e14]}
        ),
    )

    grads_computed = pr._compute_derivatives(derivative_info=info)

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


@pytest.mark.parametrize("eps_real", [1e6, -1e8])
def test_adaptive_spacing(eps_real):
    freq = 5e9

    eps_keys = ["eps_xx", "eps_yy", "eps_zz"]

    info = DerivativeInfo(
        paths={},
        E_der_map={},
        D_der_map={},
        E_fwd={},
        D_fwd={},
        E_adj={},
        D_adj={},
        eps_data={
            key: td.ScalarFieldDataArray(
                [[[[eps_real]]]], coords={"x": [0], "y": [0], "z": [0], "f": [200e12]}
            )
            for key in eps_keys
        },
        frequencies=[freq],
        bounds=((-1, -1, -1), (1, 1, 1)),
        bounds_intersect=((-1, -1, -1), (1, 1, 1)),
        simulation_bounds=((-2, -2, -2), (2, 2, 2)),
        updated_epsilon=lambda geom: td.ScalarFieldDataArray(
            [[[[1.0]]]], coords={"x": [0], "y": [0], "z": [0], "f": [freq]}
        ),
    )

    with AssertLogLevel("WARNING", contains_str="Based on the material, the adaptive spacing"):
        expected_vjp_spacing = info.wavelength_min * config.adjoint.minimum_spacing_fraction
        vjp_spacing = info.adaptive_vjp_spacing()

        assert np.isclose(expected_vjp_spacing, vjp_spacing), "Unexpected adaptive vjp spacing!"


def test_adaptive_spacing_cache(rng, redirect_stdout_to_stderr, monkeypatch):
    """Ensure the cache is not affecting `GeometryGroup` results or persisting after derivative computation."""
    x_geom = [-0.5, 0.5]
    y_geom = [-0.5, 0.5]

    radius = 0.3

    geometries = []
    for x_center in x_geom:
        for y_center in y_geom:
            geometries.append(
                td.Cylinder(center=(x_center, y_center, 0.0), length=0.5, radius=radius)
            )

    field_paths = []
    for idx in range(len(geometries)):
        field_paths.append(("geometries", idx, "radius"))

    geometry = td.GeometryGroup(geometries=geometries)

    eps_keys = ["eps_xx", "eps_yy", "eps_zz"]

    N = 20
    xcoord = np.linspace(-1, 1, N)
    ycoord = np.linspace(-1, 1, N)
    zcoord = np.linspace(-1, 1, N)

    eps_out_data = rng.uniform(1, 2, (N, N, N, 1))
    eps_in_data = rng.uniform(1, 2, (N, N, N, 1))

    freq = 1.94e14

    def random_scalar_data_array():
        return td.ScalarFieldDataArray(
            rng.uniform(1, 2, (N, N, N, 1)),
            coords={"x": xcoord, "y": ycoord, "z": zcoord, "f": [freq]},
        )

    E_fwd = {key: random_scalar_data_array() for key in ["Ex", "Ey", "Ez"]}
    E_adj = {key: random_scalar_data_array() for key in ["Ex", "Ey", "Ez"]}
    D_fwd = {key: random_scalar_data_array() for key in ["Ex", "Ey", "Ez"]}
    D_adj = {key: random_scalar_data_array() for key in ["Ex", "Ey", "Ez"]}
    E_der_map = {key: random_scalar_data_array() for key in ["Ex", "Ey", "Ez"]}
    D_der_map = {key: random_scalar_data_array() for key in ["Ex", "Ey", "Ez"]}

    derivative_info = DerivativeInfo(
        paths=tuple(field_paths),
        E_der_map=E_der_map,
        D_der_map=D_der_map,
        E_fwd=E_fwd,
        D_fwd=D_fwd,
        E_adj=E_adj,
        D_adj=D_adj,
        eps_data={
            key: td.ScalarFieldDataArray(
                rng.uniform(1, 2, (N, N, N, 1)),
                coords={"x": xcoord, "y": ycoord, "z": zcoord, "f": [freq]},
            )
            for key in eps_keys
        },
        frequencies=[freq],
        bounds=((-1, -1, -1), (1, 1, 1)),
        bounds_intersect=((-1, -1, -1), (1, 1, 1)),
        simulation_bounds=((-2, -2, -2), (2, 2, 2)),
        updated_epsilon=lambda geom: td.ScalarFieldDataArray(
            eps_out_data, coords={"x": xcoord, "y": ycoord, "z": zcoord, "f": [freq]}
        ),
    )

    vjp_with_cache = geometry._compute_derivatives(derivative_info)

    assert derivative_info.cached_min_spacing_from_permittivity is None, (
        "Unexpected cached variable persistence."
    )

    monkeypatch.setattr(
        derivative_info, "cache_min_spacing_from_permittivity", lambda: nullcontext()
    )

    vjp_without_cache = geometry._compute_derivatives(derivative_info)

    for k, v in vjp_with_cache.items():
        assert v == vjp_without_cache[k], (
            "Geometry group computation changed when running with cache."
        )


@pytest.mark.parametrize("eps_real", [1e6, -1e8])
def test_cylinder_discretization(eps_real):
    freq = 5e9

    eps_keys = ["eps_xx", "eps_yy", "eps_zz"]

    info = DerivativeInfo(
        paths={},
        E_der_map={},
        D_der_map={},
        E_fwd={},
        D_fwd={},
        E_adj={},
        D_adj={},
        eps_data={
            key: td.ScalarFieldDataArray(
                [[[[eps_real]]]], coords={"x": [0], "y": [0], "z": [0], "f": [200e12]}
            )
            for key in eps_keys
        },
        frequencies=[freq],
        bounds=((-1, -1, -1), (1, 1, 1)),
        bounds_intersect=((-1, -1, -1), (1, 1, 1)),
        simulation_bounds=((-2, -2, -2), (2, 2, 2)),
        updated_epsilon=lambda geom: td.ScalarFieldDataArray(
            [[[[1.0]]]], coords={"x": [0], "y": [0], "z": [0], "f": [freq]}
        ),
    )

    with AssertLogLevel(
        "WARNING", contains_str="The minimum wavelength inside the cylinder material"
    ):
        expected_wvl_mat = info.wavelength_min * config.adjoint.min_wvl_fraction
        wvl_mat = discretization_wavelength(info, "cylinder")

        assert np.isclose(expected_wvl_mat, wvl_mat), (
            "Unexpected wavelength for discretizing cylinder!"
        )


def test_outside_snapped_points_reach_simulation_boundary_true():
    freq = 2.0e14
    coords = {"x": [0.0, 0.2], "y": [0.0], "z": [0.0], "f": [freq]}

    def scalar_field(value):
        return td.ScalarFieldDataArray(
            np.full((2, 1, 1, 1), value, dtype=float),
            coords=coords,
            dims=("x", "y", "z", "f"),
        )

    info = DerivativeInfo(
        paths={},
        E_der_map={},
        D_der_map={},
        E_fwd={},
        D_fwd={},
        E_adj={},
        D_adj={},
        eps_data={key: scalar_field(2.0) for key in ("eps_xx", "eps_yy", "eps_zz")},
        frequencies=[freq],
        bounds=((0.0, -1.0, -1.0), (1.0, 1.0, 1.0)),
        bounds_intersect=((0.0, -1.0, -1.0), (1.0, 1.0, 1.0)),
        simulation_bounds=((0.0, -1.0, -1.0), (1.0, 1.0, 1.0)),
        updated_epsilon=lambda geom: td.ScalarFieldDataArray(
            [[[[1.0]]]], coords={"x": [0], "y": [0], "z": [0], "f": [freq]}
        ),
    )

    spatial_coords = np.array([[1.0, 0.0, 0.0]])
    normals = np.array([[1.0, 0.0, 0.0]])
    assert info._outside_snapped_points_reach_simulation_boundary(
        spatial_coords=spatial_coords, normals=normals
    )


def test_outside_snapped_points_reach_simulation_boundary_false():
    freq = 2.0e14
    coords = {"x": [0.0, 0.2], "y": [0.0], "z": [0.0], "f": [freq]}

    def scalar_field(value):
        return td.ScalarFieldDataArray(
            np.full((2, 1, 1, 1), value, dtype=float),
            coords=coords,
            dims=("x", "y", "z", "f"),
        )

    info = DerivativeInfo(
        paths={},
        E_der_map={},
        D_der_map={},
        E_fwd={},
        D_fwd={},
        E_adj={},
        D_adj={},
        eps_data={key: scalar_field(2.0) for key in ("eps_xx", "eps_yy", "eps_zz")},
        frequencies=[freq],
        bounds=((0.0, -1.0, -1.0), (1.0, 1.0, 1.0)),
        bounds_intersect=((0.0, -1.0, -1.0), (1.0, 1.0, 1.0)),
        simulation_bounds=((0.0, -1.0, -1.0), (1.0, 1.0, 1.0)),
        updated_epsilon=lambda geom: td.ScalarFieldDataArray(
            [[[[1.0]]]], coords={"x": [0], "y": [0], "z": [0], "f": [freq]}
        ),
    )

    spatial_coords = np.array([[0.5, 0.0, 0.0]])
    normals = np.array([[1.0, 0.0, 0.0]])
    assert not info._outside_snapped_points_reach_simulation_boundary(
        spatial_coords=spatial_coords, normals=normals
    )


def test_outside_snapped_points_reach_simulation_boundary_false_for_inward_normal():
    freq = 2.0e14
    coords = {"x": [0.0, 0.2], "y": [0.0], "z": [0.0], "f": [freq]}

    def scalar_field(value):
        return td.ScalarFieldDataArray(
            np.full((2, 1, 1, 1), value, dtype=float),
            coords=coords,
            dims=("x", "y", "z", "f"),
        )

    info = DerivativeInfo(
        paths={},
        E_der_map={},
        D_der_map={},
        E_fwd={},
        D_fwd={},
        E_adj={},
        D_adj={},
        eps_data={key: scalar_field(2.0) for key in ("eps_xx", "eps_yy", "eps_zz")},
        frequencies=[freq],
        bounds=((0.0, -1.0, -1.0), (1.0, 1.0, 1.0)),
        bounds_intersect=((0.0, -1.0, -1.0), (1.0, 1.0, 1.0)),
        simulation_bounds=((0.0, -1.0, -1.0), (1.0, 1.0, 1.0)),
        updated_epsilon=lambda geom: td.ScalarFieldDataArray(
            [[[[1.0]]]], coords={"x": [0], "y": [0], "z": [0], "f": [freq]}
        ),
    )

    spatial_coords = np.array([[1.0, 0.0, 0.0]])
    normals = np.array([[-1.0, 0.0, 0.0]])
    assert not info._outside_snapped_points_reach_simulation_boundary(
        spatial_coords=spatial_coords, normals=normals
    )


def test_custom_pole_residue(monkeypatch):
    """Test that computed pole residue derivatives match."""

    nx, ny, nz = shape = (4, 5, 6)
    values = np.random.random((nx, ny, nz)) * (2 + 2j) * td.C_0

    nx, ny, nz = values.shape
    x = np.linspace(-0.5, 0.5, nx)
    y = np.linspace(-0.5, 0.5, ny)
    z = np.linspace(-0.5, 0.5, nz)
    coords = {"x": x, "y": y, "z": z}

    eps_inf = td.SpatialDataArray(anp.real(values), coords=coords)
    a1 = td.SpatialDataArray(-values, coords=coords)
    c1 = td.SpatialDataArray(values, coords=coords)
    a2 = td.SpatialDataArray(-values, coords=coords)
    c2 = td.SpatialDataArray(values, coords=coords)
    poles = [(a1, c1), (a2, c2)]
    custom_med_pole_res = td.CustomPoleResidue(eps_inf=eps_inf, poles=poles)

    def J(eps):
        return anp.sum(anp.abs(eps))

    freq = 3e8
    pr = td.CustomPoleResidue(eps_inf=eps_inf, poles=poles)
    eps0 = pr.eps_model(freq)

    dJ_deps = np.conj(ag.holomorphic_grad(J)(eps0))

    import importlib

    importlib.reload(td)

    _patch_cmp_custom_to_const(monkeypatch, td.CustomPoleResidue, dJ_deps)

    pr = td.CustomPoleResidue(eps_inf=eps_inf, poles=poles)
    field_paths = [("eps_inf",)]
    for i in range(len(poles)):
        for j in range(2):
            field_paths.append(("poles", i, j))

    eps_keys = ["eps_xx", "eps_yy", "eps_zz"]

    info = DerivativeInfo(
        paths=field_paths,
        E_der_map={},
        D_der_map={},
        E_fwd={},
        D_fwd={},
        E_adj={},
        D_adj={},
        eps_data={
            key: td.ScalarFieldDataArray(
                [[[[2.0]]]], coords={"x": [0], "y": [0], "z": [0], "f": [200e12]}
            )
            for key in eps_keys
        },
        frequencies=[freq],
        bounds=((-1, -1, -1), (1, 1, 1)),
        bounds_intersect=((-1, -1, -1), (1, 1, 1)),
        simulation_bounds=((-2, -2, -2), (2, 2, 2)),
        updated_epsilon=lambda geom: td.ScalarFieldDataArray(
            [[[[1.0]]]], coords={"x": [0], "y": [0], "z": [0], "f": [freq]}
        ),
    )

    grads_computed = pr._compute_derivatives(derivative_info=info)

    poles_complex = [
        (np.array(a.values, dtype=complex), np.array(c.values, dtype=complex)) for a, c in poles
    ]
    poles_complex = np.stack(poles_complex, axis=0)

    def f(eps_inf, poles):
        eps = td.CustomPoleResidue._eps_model(eps_inf, poles, freq)
        return J(eps)

    gfn = ag.grad(lambda x: f(x, poles_complex))
    grad_eps_inf = gfn(eps_inf.values)

    assert np.allclose(grads_computed[("eps_inf",)], grad_eps_inf)

    gfn = ag.holomorphic_grad(lambda x: f(eps_inf.values, x))
    grad_poles = gfn(poles_complex)

    for i in range(len(poles)):
        for j in range(2):
            field_path = ("poles", i, j)
            assert np.allclose(grads_computed[field_path], np.conj(grad_poles[i][j]))


def test_custom_pole_residue_unstructured_derivatives():
    """Ensure unstructured pole residue adjoints are explicitly unsupported."""
    pr = custom_poleresidue_u
    field_paths = [("eps_inf",), ("poles", 0, 0), ("poles", 0, 1)]

    eps_keys = ["eps_xx", "eps_yy", "eps_zz"]

    info = DerivativeInfo(
        paths=field_paths,
        E_der_map={},
        D_der_map={},
        E_fwd={},
        D_fwd={},
        E_adj={},
        D_adj={},
        eps_data={
            key: td.ScalarFieldDataArray(
                [[[[2.0]]]], coords={"x": [0], "y": [0], "z": [0], "f": [200e12]}
            )
            for key in eps_keys
        },
        frequencies=[3e8],
        bounds=((-1, -1, -1), (1, 1, 1)),
        bounds_intersect=((-1, -1, -1), (1, 1, 1)),
        simulation_bounds=((-2, -2, -2), (2, 2, 2)),
        updated_epsilon=lambda geom: td.ScalarFieldDataArray(
            [[[[1.0]]]], coords={"x": [0], "y": [0], "z": [0], "f": [3e8]}
        ),
    )

    with pytest.raises(NotImplementedError, match="unstructured"):
        pr._compute_derivatives(derivative_info=info)


def test_custom_sellmeier(monkeypatch):
    """Test that computed CustomSellmeier derivatives match analytic mapping."""

    rng = np.random.RandomState(0)
    shape = (2, 2, 2)
    coords = {
        "x": np.linspace(-0.5, 0.5, shape[0]),
        "y": np.linspace(-0.5, 0.5, shape[1]),
        "z": np.linspace(-0.5, 0.5, shape[2]),
    }

    # positive B, C
    B1 = td.SpatialDataArray(0.2 + 0.5 * rng.rand(*shape), coords=coords)
    C1 = td.SpatialDataArray(0.3 + 0.5 * rng.rand(*shape), coords=coords)
    B2 = td.SpatialDataArray(0.2 + 0.5 * rng.rand(*shape), coords=coords)
    C2 = td.SpatialDataArray(0.3 + 0.5 * rng.rand(*shape), coords=coords)
    med = td.CustomSellmeier(coeffs=[(B1, C1), (B2, C2)])

    freq = 2.5e14
    lam2 = td.C_0 / freq
    lam2 = lam2 * lam2

    def eps_from(B, C):
        return 1.0 + B * lam2 / (lam2 - C)

    eps_arr = eps_from(B1.values, C1.values) + eps_from(B2.values, C2.values) + 0j
    dJ = np.conj(ag.holomorphic_grad(lambda e: anp.sum(anp.abs(e)))(eps_arr))

    _patch_cmp_custom_to_const(monkeypatch, td.CustomSellmeier, dJ)

    di = _make_di(
        paths=[("coeffs", 0, 0), ("coeffs", 0, 1), ("coeffs", 1, 0), ("coeffs", 1, 1)],
        freq=freq,
    )
    grads = med._compute_derivatives(di)

    def obj(B1_, C1_, B2_, C2_):
        eps = eps_from(B1_, C1_) + eps_from(B2_, C2_)
        return anp.sum(anp.abs(eps))

    gB1 = ag.grad(lambda x: obj(x, C1.values, B2.values, C2.values))(B1.values)
    gC1 = ag.grad(lambda x: obj(B1.values, x, B2.values, C2.values))(C1.values)
    gB2 = ag.grad(lambda x: obj(B1.values, C1.values, x, C2.values))(B2.values)
    gC2 = ag.grad(lambda x: obj(B1.values, C1.values, B2.values, x))(C2.values)

    np.testing.assert_allclose(grads[("coeffs", 0, 0)], gB1, rtol=5e-6, atol=5e-7)
    np.testing.assert_allclose(grads[("coeffs", 0, 1)], gC1, rtol=5e-6, atol=5e-7)
    np.testing.assert_allclose(grads[("coeffs", 1, 0)], gB2, rtol=5e-6, atol=5e-7)
    np.testing.assert_allclose(grads[("coeffs", 1, 1)], gC2, rtol=5e-6, atol=5e-7)


def test_custom_lorentz(monkeypatch):
    rng = np.random.RandomState(1)
    shape = (2, 2, 2)
    coords = {
        "x": np.linspace(-0.5, 0.5, shape[0]),
        "y": np.linspace(-0.5, 0.5, shape[1]),
        "z": np.linspace(-0.5, 0.5, shape[2]),
    }

    eps_inf = td.SpatialDataArray(1.2 + 0.3 * rng.rand(*shape), coords=coords)
    de1 = td.SpatialDataArray(0.1 + 0.4 * rng.rand(*shape), coords=coords)
    f01 = td.SpatialDataArray(3.0e14 + 0.1e14 * rng.rand(*shape), coords=coords)
    dl1 = td.SpatialDataArray(0.1e14 + 0.1e14 * rng.rand(*shape), coords=coords)
    de2 = td.SpatialDataArray(0.1 + 0.4 * rng.rand(*shape), coords=coords)
    f02 = td.SpatialDataArray(3.5e14 + 0.1e14 * rng.rand(*shape), coords=coords)
    dl2 = td.SpatialDataArray(0.1e14 + 0.1e14 * rng.rand(*shape), coords=coords)
    med = td.CustomLorentz(eps_inf=eps_inf, coeffs=[(de1, f01, dl1), (de2, f02, dl2)])

    freq = 2.0e14

    def term(de, f0, dl):
        den = (f0**2) - 2j * (freq * dl) - (freq**2)
        return (de * (f0**2)) / den

    eps_arr = (
        eps_inf.values
        + term(de1.values, f01.values, dl1.values)
        + term(de2.values, f02.values, dl2.values)
    )
    dJ = np.conj(ag.holomorphic_grad(lambda e: anp.sum(anp.abs(e)))(eps_arr))

    _patch_cmp_custom_to_const(monkeypatch, td.CustomLorentz, dJ)

    di = _make_di(
        paths=[
            ("eps_inf",),
            ("coeffs", 0, 0),
            ("coeffs", 0, 1),
            ("coeffs", 0, 2),
            ("coeffs", 1, 0),
            ("coeffs", 1, 1),
            ("coeffs", 1, 2),
        ],
        freq=freq,
    )
    grads = med._compute_derivatives(di)

    def obj(ei, de1_, f01_, dl1_, de2_, f02_, dl2_):
        def t(de, f0, dl):
            den = (f0**2) - 2j * (freq * dl) - (freq**2)
            return (de * (f0**2)) / den

        eps = ei + t(de1_, f01_, dl1_) + t(de2_, f02_, dl2_)
        return anp.sum(anp.abs(eps))

    g_ei = ag.grad(
        lambda x: obj(x, de1.values, f01.values, dl1.values, de2.values, f02.values, dl2.values)
    )(eps_inf.values)
    g_de1 = ag.grad(
        lambda x: obj(eps_inf.values, x, f01.values, dl1.values, de2.values, f02.values, dl2.values)
    )(de1.values)
    g_f01 = ag.grad(
        lambda x: obj(eps_inf.values, de1.values, x, dl1.values, de2.values, f02.values, dl2.values)
    )(f01.values)
    g_dl1 = ag.grad(
        lambda x: obj(eps_inf.values, de1.values, f01.values, x, de2.values, f02.values, dl2.values)
    )(dl1.values)
    g_de2 = ag.grad(
        lambda x: obj(eps_inf.values, de1.values, f01.values, dl1.values, x, f02.values, dl2.values)
    )(de2.values)
    g_f02 = ag.grad(
        lambda x: obj(eps_inf.values, de1.values, f01.values, dl1.values, de2.values, x, dl2.values)
    )(f02.values)
    g_dl2 = ag.grad(
        lambda x: obj(eps_inf.values, de1.values, f01.values, dl1.values, de2.values, f02.values, x)
    )(dl2.values)

    np.testing.assert_allclose(grads[("eps_inf",)], g_ei, rtol=5e-6, atol=5e-7)
    np.testing.assert_allclose(grads[("coeffs", 0, 0)], g_de1, rtol=5e-6, atol=5e-7)
    np.testing.assert_allclose(grads[("coeffs", 0, 1)], g_f01, rtol=5e-6, atol=5e-7)
    np.testing.assert_allclose(grads[("coeffs", 0, 2)], g_dl1, rtol=5e-6, atol=5e-7)
    np.testing.assert_allclose(grads[("coeffs", 1, 0)], g_de2, rtol=5e-6, atol=5e-7)
    np.testing.assert_allclose(grads[("coeffs", 1, 1)], g_f02, rtol=5e-6, atol=5e-7)
    np.testing.assert_allclose(grads[("coeffs", 1, 2)], g_dl2, rtol=5e-6, atol=5e-7)


def test_custom_drude(monkeypatch):
    rng = np.random.RandomState(2)
    shape = (2, 2, 2)
    coords = {
        "x": np.linspace(-0.5, 0.5, shape[0]),
        "y": np.linspace(-0.5, 0.5, shape[1]),
        "z": np.linspace(-0.5, 0.5, shape[2]),
    }

    eps_inf = td.SpatialDataArray(1.1 + 0.2 * rng.rand(*shape), coords=coords)
    fp1 = td.SpatialDataArray(2.0e14 + 0.2e14 * rng.rand(*shape), coords=coords)
    dl1 = td.SpatialDataArray(0.05e14 + 0.05e14 * rng.rand(*shape), coords=coords)
    fp2 = td.SpatialDataArray(2.5e14 + 0.2e14 * rng.rand(*shape), coords=coords)
    dl2 = td.SpatialDataArray(0.05e14 + 0.05e14 * rng.rand(*shape), coords=coords)
    med = td.CustomDrude(eps_inf=eps_inf, coeffs=[(fp1, dl1), (fp2, dl2)])

    freq = 2.2e14

    def term(fp, dl):
        den = (freq**2) + 1j * (freq * dl)
        return -(fp**2) / den

    eps_arr = eps_inf.values + term(fp1.values, dl1.values) + term(fp2.values, dl2.values)
    dJ = np.conj(ag.holomorphic_grad(lambda e: anp.sum(anp.abs(e)))(eps_arr))

    _patch_cmp_custom_to_const(monkeypatch, td.CustomDrude, dJ)

    di = _make_di(
        paths=[
            ("eps_inf",),
            ("coeffs", 0, 0),
            ("coeffs", 0, 1),
            ("coeffs", 1, 0),
            ("coeffs", 1, 1),
        ],
        freq=freq,
    )
    grads = med._compute_derivatives(di)

    def obj(ei, fp1_, dl1_, fp2_, dl2_):
        def t(fp, dl):
            den = (freq**2) + 1j * (freq * dl)
            return -(fp**2) / den

        eps = ei + t(fp1_, dl1_) + t(fp2_, dl2_)
        return anp.sum(anp.abs(eps))

    g_ei = ag.grad(lambda x: obj(x, fp1.values, dl1.values, fp2.values, dl2.values))(eps_inf.values)
    g_fp1 = ag.grad(lambda x: obj(eps_inf.values, x, dl1.values, fp2.values, dl2.values))(
        fp1.values
    )
    g_dl1 = ag.grad(lambda x: obj(eps_inf.values, fp1.values, x, fp2.values, dl2.values))(
        dl1.values
    )
    g_fp2 = ag.grad(lambda x: obj(eps_inf.values, fp1.values, dl1.values, x, dl2.values))(
        fp2.values
    )
    g_dl2 = ag.grad(lambda x: obj(eps_inf.values, fp1.values, dl1.values, fp2.values, x))(
        dl2.values
    )

    np.testing.assert_allclose(grads[("eps_inf",)], g_ei, rtol=5e-6, atol=5e-7)
    np.testing.assert_allclose(grads[("coeffs", 0, 0)], g_fp1, rtol=5e-6, atol=5e-7)
    np.testing.assert_allclose(grads[("coeffs", 0, 1)], g_dl1, rtol=5e-6, atol=5e-7)
    np.testing.assert_allclose(grads[("coeffs", 1, 0)], g_fp2, rtol=5e-6, atol=5e-7)
    np.testing.assert_allclose(grads[("coeffs", 1, 1)], g_dl2, rtol=5e-6, atol=5e-7)


def test_custom_debye(monkeypatch):
    rng = np.random.RandomState(3)
    shape = (2, 2, 2)
    coords = {
        "x": np.linspace(-0.5, 0.5, shape[0]),
        "y": np.linspace(-0.5, 0.5, shape[1]),
        "z": np.linspace(-0.5, 0.5, shape[2]),
    }

    eps_inf = td.SpatialDataArray(1.05 + 0.2 * rng.rand(*shape), coords=coords)
    de1 = td.SpatialDataArray(0.1 + 0.4 * rng.rand(*shape), coords=coords)
    tau1 = td.SpatialDataArray(0.5e-14 + 0.5e-14 * rng.rand(*shape), coords=coords)
    de2 = td.SpatialDataArray(0.1 + 0.4 * rng.rand(*shape), coords=coords)
    tau2 = td.SpatialDataArray(0.5e-14 + 0.5e-14 * rng.rand(*shape), coords=coords)
    med = td.CustomDebye(eps_inf=eps_inf, coeffs=[(de1, tau1), (de2, tau2)])

    freq = 1.8e14

    def term(de, tau):
        den = 1.0 - 1j * (freq * tau)
        return de / den

    eps_arr = eps_inf.values + term(de1.values, tau1.values) + term(de2.values, tau2.values)
    dJ = np.conj(ag.holomorphic_grad(lambda e: anp.sum(anp.abs(e)))(eps_arr))

    _patch_cmp_custom_to_const(monkeypatch, td.CustomDebye, dJ)

    di = _make_di(
        paths=[
            ("eps_inf",),
            ("coeffs", 0, 0),
            ("coeffs", 0, 1),
            ("coeffs", 1, 0),
            ("coeffs", 1, 1),
        ],
        freq=freq,
    )
    grads = med._compute_derivatives(di)

    def obj(ei, de1_, tau1_, de2_, tau2_):
        def t(de, tau):
            den = 1.0 - 1j * (freq * tau)
            return de / den

        eps = ei + t(de1_, tau1_) + t(de2_, tau2_)
        return anp.sum(anp.abs(eps))

    g_ei = ag.grad(lambda x: obj(x, de1.values, tau1.values, de2.values, tau2.values))(
        eps_inf.values
    )
    g_de1 = ag.grad(lambda x: obj(eps_inf.values, x, tau1.values, de2.values, tau2.values))(
        de1.values
    )
    g_tau1 = ag.grad(lambda x: obj(eps_inf.values, de1.values, x, de2.values, tau2.values))(
        tau1.values
    )
    g_de2 = ag.grad(lambda x: obj(eps_inf.values, de1.values, tau1.values, x, tau2.values))(
        de2.values
    )
    g_tau2 = ag.grad(lambda x: obj(eps_inf.values, de1.values, tau1.values, de2.values, x))(
        tau2.values
    )

    np.testing.assert_allclose(grads[("eps_inf",)], g_ei, rtol=5e-6, atol=5e-7)
    np.testing.assert_allclose(grads[("coeffs", 0, 0)], g_de1, rtol=5e-6, atol=5e-7)
    np.testing.assert_allclose(grads[("coeffs", 0, 1)], g_tau1, rtol=5e-6, atol=5e-7)
    np.testing.assert_allclose(grads[("coeffs", 1, 0)], g_de2, rtol=5e-6, atol=5e-7)
    np.testing.assert_allclose(grads[("coeffs", 1, 1)], g_tau2, rtol=5e-6, atol=5e-7)


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
            structures=(structure_traced,),
            monitors=(*SIM_BASE.monitors, mnt_single, mnt_multi),
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


def check_1_src_single(structure_key):
    def postprocess(sim_data: td.SimulationData) -> float:
        """Postprocess function that should return 1 adjoint sources."""
        amps = get_amps(sim_data, "single").sel(mode_index=0, direction="+")
        return power(amps)

    return postprocess


def check_2_src_single(structure_key):
    def postprocess(sim_data: td.SimulationData) -> float:
        """Postprocess function that should return 2 different adjoint sources."""
        amps = get_amps(sim_data, "single").sel(mode_index=0)
        return power(amps)

    return postprocess


def check_1_src_multi(structure_key):
    def postprocess(sim_data: td.SimulationData) -> float:
        """Postprocess function that should return 1 adjoint sources."""
        amps = get_amps(sim_data, "multi").sel(mode_index=0, direction="+", f=FREQ0)
        return power(amps)

    return postprocess


def check_2_src_multi(structure_key):
    def postprocess(sim_data: td.SimulationData) -> float:
        """Postprocess function that should return 2 different adjoint sources."""
        amps = get_amps(sim_data, "multi").sel(mode_index=0, f=FREQ1)
        return power(amps)

    return postprocess


def check_2_src_both(structure_key):
    def postprocess(sim_data: td.SimulationData) -> float:
        """Postprocess function that should return 2 different adjoint sources."""
        amps_single = get_amps(sim_data, "single").sel(mode_index=0, direction="+")
        amps_multi = get_amps(sim_data, "multi").sel(mode_index=0, direction="+", f=FREQ0)
        return power(amps_single) + power(amps_multi)

    return postprocess


def check_1_multisrc(structure_key):
    def postprocess(sim_data: td.SimulationData) -> float:
        """Postprocess function that should raise ValueError because diff sources, diff freqs."""
        amps_single = get_amps(sim_data, "single").sel(mode_index=0, direction="+")
        amps_multi = get_amps(sim_data, "multi").sel(mode_index=0, direction="+", f=FREQ1)
        return power(amps_single) + power(amps_multi)

    return postprocess


def check_2_multisrc(structure_key):
    def postprocess(sim_data: td.SimulationData) -> float:
        """Postprocess function that should raise ValueError because diff sources, diff freqs."""
        amps_single = get_amps(sim_data, "single").sel(mode_index=0, direction="+")
        amps_multi = get_amps(sim_data, "multi").sel(mode_index=0, direction="+")
        return power(amps_single) + power(amps_multi)

    return postprocess


def check_1_src_broadband(structure_key):
    def postprocess(sim_data: td.SimulationData) -> float:
        """Postprocess function that should return 1 broadband adjoint sources with many freqs."""
        amps = get_amps(sim_data, "multi").sel(mode_index=0, direction="+")
        return power(amps)

    return postprocess


MULT_FREQ_TEST_CASES = {
    "src_1_freq_1": check_1_src_single,
    "src_2_freq_1": check_2_src_single,
    "src_1_freq_2": check_1_src_multi,
    "src_2_freq_1_mon_1": check_1_src_multi,
    "src_2_freq_1_mon_2": check_2_src_both,
    "src_2_freq_2_mon_1": check_1_multisrc,
    "src_2_freq_2_mon_2": check_2_multisrc,
    "src_1_freq_2_broadband": check_1_src_broadband,
}

checks = list(MULT_FREQ_TEST_CASES.items())


@pytest.mark.parametrize("label, check_fn", checks)
@pytest.mark.parametrize("structure_key", ("custom_med",))
def test_multi_freq_edge_cases(use_emulated_run, structure_key, label, check_fn, monkeypatch):
    # test multi-frequency adjoint handling

    postprocess_fn = check_fn(structure_key=structure_key)

    def objective(params):
        structure_traced = make_structures(params)[structure_key]
        sim = SIM_BASE.updated_copy(
            structures=(structure_traced,),
            monitors=(*SIM_BASE.monitors, mnt_single, mnt_multi),
        )
        data = run(sim, task_name="multifreq_test")
        return postprocess_fn(data)

    if label == "src_2_freq_2_mon_2":
        with pytest.raises(AdjointError, match="conflicting post-normalization values"):
            ag.grad(objective)(params0)
    else:
        g = ag.grad(objective)(params0)
        assert np.all(np.isfinite(g)), f"non-finite gradient for {label}"


@pytest.mark.parametrize("structure_key", structure_keys_)
def test_multi_frequency_equivalence(use_emulated_run, structure_key):
    """Test an objective function through tidy3d autograd."""

    def objective_indi(params, structure_key) -> float:
        power_sum = 0.0

        for f in mnt_multi.freqs:
            structure_traced = make_structures(params)[structure_key]
            sim = SIM_BASE.updated_copy(
                structures=(structure_traced,),
                monitors=(*SIM_BASE.monitors, mnt_multi),
            )

            sim_data = web.run(sim, task_name="multifreq_test")
            amps_i = get_amps(sim_data, "multi").sel(mode_index=0, direction="+", f=f)
            power_i = power(amps_i)
            power_sum = power_sum + power_i

        return power_sum

    def objective_multi(params, structure_key) -> float:
        structure_traced = make_structures(params)[structure_key]
        sim = SIM_BASE.updated_copy(
            structures=(structure_traced,),
            monitors=(*SIM_BASE.monitors, mnt_multi),
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


@pytest.mark.parametrize(
    "monitor_cls", (td.GaussianOverlapMonitor, td.AstigmaticGaussianOverlapMonitor)
)
def test_gaussian_overlap_gradient_pipeline(use_emulated_run, monitor_cls):
    """Test autograd pipeline with Gaussian overlap amplitudes in objective."""

    fwd_params = params0 + 1.0
    freq1 = FREQ0 * 1.6

    def make_monitor():
        kwargs = {
            "size": (2, 2, 0),
            "center": (0, 0, LZ / 2 - WVL),
            "freqs": [FREQ0, freq1],
            "name": "gaussian_overlap",
            "pol_angle": np.pi / 2,
        }
        if monitor_cls is td.GaussianOverlapMonitor:
            return monitor_cls(waist_radius=0.8, waist_distance=0.2, **kwargs)
        return monitor_cls(waist_sizes=(0.8, 0.6), waist_distances=(0.2, -0.1), **kwargs)

    def objective(params):
        structure_traced = make_structures(params)["custom_med"]
        sim = SIM_BASE.updated_copy(
            structures=(structure_traced,),
            monitors=(*SIM_BASE.monitors, make_monitor()),
        )
        sim_data = run(sim, task_name="gaussian_overlap_grad")
        amps = sim_data["gaussian_overlap"].amps.sel(direction="+")
        return anp.sum(anp.abs(amps.values) ** 2)

    grad = ag.grad(objective)(fwd_params)

    assert np.all(np.isfinite(grad))
    assert not np.all(np.isclose(grad, 0.0))


def test_gaussian_overlap_frequency_selection_pipeline(use_emulated_run):
    """Test selecting one frequency and one direction from Gaussian overlap amplitudes."""

    fwd_params = params0 + 1.0
    freq1 = FREQ0 * 1.6

    monitor = td.GaussianOverlapMonitor(
        size=(2, 2, 0),
        center=(0, 0, LZ / 2 - WVL),
        freqs=[FREQ0, freq1],
        name="gaussian_overlap",
        pol_angle=np.pi / 2,
        waist_radius=0.8,
        waist_distance=0.2,
    )

    def objective(params):
        structure_traced = make_structures(params)["custom_med"]
        sim = SIM_BASE.updated_copy(
            structures=(structure_traced,),
            monitors=(*SIM_BASE.monitors, monitor),
        )
        sim_data = run(sim, task_name="gaussian_overlap_freq")
        amps = sim_data["gaussian_overlap"].amps.sel(direction="+", f=FREQ0, mode_index=0)
        return anp.sum(anp.abs(amps.values) ** 2)

    grad = ag.grad(objective)(fwd_params)

    assert np.all(np.isfinite(grad))
    assert not np.all(np.isclose(grad, 0.0))


def test_gaussian_overlap_multifreq_grouped_to_one_adjoint_source(use_emulated_run, monkeypatch):
    """Single Gaussian overlap monitor with multi-f objective should produce one grouped adjoint source."""

    fwd_params = params0 + 1.0
    freqs = [FREQ0 * 0.9, FREQ0, FREQ0 * 1.1, FREQ0 * 1.2]

    monitor = td.GaussianOverlapMonitor(
        size=(2, 2, 0),
        center=(0, 0, LZ / 2 - WVL),
        freqs=freqs,
        name="gaussian_overlap",
        pol_angle=np.pi / 2,
        waist_radius=0.8,
        waist_distance=0.2,
    )

    setup_adj_orig = autograd_module.setup_adj
    process_adj_orig = td.SimulationData._process_adjoint_sources
    captured = {"num_sims_adj": None, "num_sources_first_sim": None}
    captured_infos = []

    def process_adj_capture(self, adj_srcs):
        infos = process_adj_orig(self, adj_srcs)
        captured_infos.extend(infos)
        return infos

    def setup_adj_capture(*args, **kwargs):
        sims_adj = setup_adj_orig(*args, **kwargs)
        captured["num_sims_adj"] = len(sims_adj)
        captured["num_sources_first_sim"] = len(sims_adj[0].sources) if sims_adj else 0
        return sims_adj

    monkeypatch.setattr(autograd_module, "setup_adj", setup_adj_capture)
    monkeypatch.setattr(td.SimulationData, "_process_adjoint_sources", process_adj_capture)

    def objective(params):
        structure_traced = make_structures(params)["custom_med"]
        sim = SIM_BASE.updated_copy(
            structures=(structure_traced,),
            monitors=(*SIM_BASE.monitors, monitor),
        )
        sim_data = run(sim, task_name="gaussian_overlap_grouping")
        amps = sim_data["gaussian_overlap"].amps.sel(direction="+")
        return anp.sum(anp.abs(amps.values) ** 2)

    grad = ag.grad(objective)(fwd_params)

    assert np.all(np.isfinite(grad))
    assert captured["num_sims_adj"] == 1
    assert captured["num_sources_first_sim"] == 1
    assert captured_infos
    grouped_info = captured_infos[0]
    grouped_freqs = np.array(grouped_info.post_norm.coords["f"].values, dtype=float)
    assert np.allclose(np.sort(grouped_freqs), np.sort(freqs))
    assert grouped_info.sources[0].num_freqs == 3


def test_error_flux(use_emulated_run):
    """Make sure proper error raised if differentiating w.r.t. FluxData."""

    def objective(params):
        structure_traced = make_structures(params)["medium"]
        sim = SIM_BASE.updated_copy(
            structures=(structure_traced,),
            monitors=(
                td.FluxMonitor(size=(1, 1, 0), center=(0, 0, 0), freqs=[FREQ0], name="flux"),
                td.FieldMonitor(size=(1, 1, 0), center=(0, 0, 0), freqs=[FREQ0], name="field"),
            ),
        )
        data = run(sim, task_name="flux_error")
        return anp.sum(data["flux"].flux.values)

    with pytest.raises(
        NotImplementedError, match="Could not formulate adjoint source for 'FluxMonitor' output"
    ):
        g = ag.grad(objective)(params0)


def test_error_gaussian_overlap_unsupported_dataset_name_raises_not_implemented():
    """Ensure unsupported FieldOverlapData dataset names raise NotImplementedError."""

    monitor = td.GaussianOverlapMonitor(
        center=(0, 0, 0),
        size=(1, 1, 0),
        freqs=[FREQ0],
        store_fields_direction="+",
        name="gauss_overlap",
    )
    amps = td.ModeAmpsDataArray(
        np.zeros((2, 1, 1), dtype=complex),
        coords={"direction": ["+", "-"], "f": [FREQ0], "mode_index": [0]},
    )
    overlap_data = td.FieldOverlapData(monitor=monitor, amps=amps)

    with pytest.raises(NotImplementedError, match="Unsupported adjoint field 'Ex'"):
        overlap_data._make_adjoint_sources(dataset_names=["Ex"], fwidth=FWIDTH)


def test_extraneous_field(use_emulated_run):
    """Make sure this doesnt fail."""

    def objective(params):
        structure_traced = make_structures(params)["medium"]
        sim = SIM_BASE.updated_copy(
            structures=(structure_traced,),
            monitors=(
                SIM_BASE.monitors[0],
                td.ModeMonitor(
                    size=(1, 1, 0),
                    center=(0, 0, 0),
                    mode_spec=td.ModeSpec(),
                    freqs=[FREQ0 * 0.9, FREQ0 * 1.1],
                    name="mode",
                ),
            ),
        )
        data = run(sim, task_name="extra_field")
        amp = data["mode"].amps.sel(direction="+", f=FREQ0 * 0.9, mode_index=0).values
        return abs(amp.item()) ** 2

    g = ag.grad(objective)(params0)


def test_background_medium():
    geo = td.Box(size=(1, 1, 1), center=(0, 0, 0))
    med = td.Medium(permittivity=2.0)

    background_permittivity = 5.0
    background_medium = td.Medium(permittivity=background_permittivity)

    # nothing
    s = td.Structure(
        geometry=geo,
        medium=med,
    )

    # both supplied, consistent
    td.Structure(
        geometry=geo,
        medium=med,
        background_permittivity=background_permittivity,
        background_medium=background_medium,
    )

    # both supplied, inconsistent
    with pytest.raises(ValueError):
        td.Structure(
            geometry=geo,
            medium=med,
            background_permittivity=background_permittivity + 1,
            background_medium=background_medium,
        )

    # background medium (preferred)
    s = td.Structure(
        geometry=geo,
        medium=med,
        background_medium=background_medium,
    )

    # background permittivity (deprecated)
    with AssertLogLevel("WARNING", contains_str="deprecated"):
        s_warn = td.Structure(
            geometry=geo,
            medium=med,
            background_permittivity=background_permittivity,
        )

        assert s_warn.background_medium is not None
        assert s_warn.background_medium.permittivity == background_permittivity


class TestTidyArrayBox:
    def test_is_tidy_box(self):
        da = DataArray(tracer_arr, dims=tuple(map(str, range(tracer_arr.ndim))))
        assert is_tidy_box(da.data)

    def test_hasbox_dataclass(self):
        @dataclass(frozen=True)
        class _Leaf:
            value: object

        @dataclass(frozen=True)
        class _Node:
            leaf: _Leaf

        assert not hasbox(_Node(leaf=_Leaf(value=1.0)))
        assert hasbox(_Node(leaf=_Leaf(value=tracer_arr)))

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
            da = DataArray(x)
            attr_value = getattr(da, attr)
            val = attr_value() if callable(attr_value) else attr_value
            return val.item()

        x = np.array([1.0])
        check_grads(objective, modes=["fwd", "rev"], order=2)(x, attr)

    def test_multiply_at_grads(self, rng):
        """Test grads of DataArray.multiply_at method"""

        def objective(a, b):
            coords = {str(i): np.arange(a.shape[i]) for i in range(a.ndim)}
            da = DataArray(a, coords=coords)
            da_mult = da.multiply_at(b, "0", [0, 1]) ** 2
            return np.sum(da_mult).item()

        a = rng.uniform(-1, 1, (3, 3))
        b = 1.0
        check_grads(lambda x: objective(x, b), modes=["fwd", "rev"], order=2)(a)
        check_grads(lambda x: objective(a, x), modes=["fwd", "rev"], order=2)(b)


@pytest.fixture
def polyslab() -> td.PolySlab:
    """Creates a PolySlab instance for testing affine transformations."""
    vertices = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    axis = 1  # 0 for x, 1 for y, 2 for z
    slab_bounds = (-1.0, 1.0)

    return td.PolySlab(vertices=vertices, axis=axis, slab_bounds=slab_bounds)


@pytest.mark.parametrize(
    "x, y, z",
    [
        (0.0, 0.0, 0.0),  # No translation (edge case)
        (0.1, 0.2, 0.3),  # Small positive values
        (-0.1, -0.2, -0.3),  # Small negative values
        (1e-5, 1e-5, 1e-5),  # Near-zero translation
        (10.0, 10.0, 10.0),  # Large values
    ],
)
def test_polyslab_translated_grad(polyslab: td.PolySlab, x: float, y: float, z: float) -> None:
    """Checks the differentiability of the translation operation of PolySlab."""
    poly = polyslab

    def translated_grad_with_vertices(x: float, y: float, z: float) -> anp.ndarray:
        """Computes the translated vertices of a PolySlab object."""
        new_poly = poly.translated(x, y, z)
        return new_poly.vertices

    def translated_grad_with_slab_bounds(x: float, y: float, z: float) -> anp.ndarray:
        """Computes the translated slab bounds of a PolySlab object."""
        new_poly = poly.translated(x, y, z)
        return anp.array([new_poly.slab_bounds[0], new_poly.slab_bounds[1]])

    if poly.axis == 0:
        check_grads(lambda x: translated_grad_with_slab_bounds(x, y, z), modes=["rev"])(x)
    else:
        check_grads(lambda x: translated_grad_with_vertices(x, y, z), modes=["rev"])(x)

    if poly.axis == 1:
        check_grads(lambda y: translated_grad_with_slab_bounds(x, y, z), modes=["rev"])(y)
    else:
        check_grads(lambda y: translated_grad_with_vertices(x, y, z), modes=["rev"])(y)

    if poly.axis == 2:
        check_grads(lambda z: translated_grad_with_slab_bounds(x, y, z), modes=["rev"])(z)
    else:
        check_grads(lambda z: translated_grad_with_vertices(x, y, z), modes=["rev"])(z)


@pytest.mark.parametrize(
    "x, y, z, expect_exception",
    [
        (0.0, 0.0, 0.0, True),  # No scaling
        (0.1, 0.2, 0.3, False),  # Small positive values
        (-0.1, 0.2, -0.3, False),  # Reflect along x and z axes
        (0.1, -0.2, 0.3, True),  # Flips slab bounds (pydantic validation to fail)
        (1e-5, 1e-5, 1e-5, True),  # Near-zero scaling (polygon almost collapses to a 1D curve)
        (10.0, 10.0, 10.0, False),  # Large values
    ],
)
def test_polyslab_scaled_grad(
    polyslab: td.PolySlab, x: float, y: float, z: float, expect_exception: bool
) -> None:
    """Checks the differentiability of the scaling operation of PolySlab."""
    poly = polyslab

    def scaled_grad_with_vertices(x: float, y: float, z: float) -> anp.ndarray:
        """Computes the scaled vertices of a PolySlab object."""
        new_poly = poly.scaled(x, y, z)
        return new_poly.vertices

    def scaled_grad_with_slab_bounds(x: float, y: float, z: float) -> anp.ndarray:
        """Computes the scaled slab bounds of a PolySlab object."""
        new_poly = poly.scaled(x, y, z)
        return anp.array([new_poly.slab_bounds[0], new_poly.slab_bounds[1]])

    if expect_exception:
        with pytest.raises(ValueError):
            check_grads(lambda x: scaled_grad_with_vertices(x, y, z), modes=["rev"])(x)
            check_grads(lambda y: scaled_grad_with_vertices(x, y, z), modes=["rev"])(y)
            check_grads(lambda z: scaled_grad_with_vertices(x, y, z), modes=["rev"])(z)
    else:
        if poly.axis == 0:
            check_grads(lambda x: scaled_grad_with_slab_bounds(x, y, z), modes=["rev"])(x)
        else:
            check_grads(lambda x: scaled_grad_with_vertices(x, y, z), modes=["rev"])(x)

        if poly.axis == 1:
            check_grads(lambda y: scaled_grad_with_slab_bounds(x, y, z), modes=["rev"])(y)
        else:
            check_grads(lambda y: scaled_grad_with_vertices(x, y, z), modes=["rev"])(y)

        if poly.axis == 2:
            check_grads(lambda z: scaled_grad_with_slab_bounds(x, y, z), modes=["rev"])(z)
        else:
            check_grads(lambda z: scaled_grad_with_vertices(x, y, z), modes=["rev"])(z)


@pytest.mark.parametrize(
    "theta, axis",
    [
        (0.0, 0),  # No rotation around x-axis
        (np.pi / 6, 1),  # Small rotation around y-axis
        (-np.pi / 4, 2),  # Rotation around z-axis
        (np.pi / 4, 1),  # 90-degree rotation around y-axis
        (np.pi, 1),  # 180-degree rotation around y-axis
    ],
)
def test_polyslab_rotated_grad(polyslab: td.PolySlab, theta: float, axis: int) -> None:
    """Checks the differentiability of the rotation operation of PolySlab."""
    poly = polyslab
    expect_exception = axis != poly.axis  # Rotation about different axis will fail

    def rotated_grad(angle: float, axis: int) -> np.ndarray:
        """Computes the rotated vertices of a PolySlab object."""
        return poly.rotated(angle, axis).vertices

    if expect_exception:
        with pytest.raises(
            AttributeError, match=".*'Transformed' object has no attribute 'vertices'.*"
        ):
            rotated_grad(theta, axis)
    else:
        check_grads(lambda theta: rotated_grad(theta, axis), modes=["rev"])(theta)


def test_flux_monitor_freq_exclusion(use_emulated_run):
    """Checks if we are excluding flux monitor frequencies from the adjoint frequencies since
    we cannot differentiate through flux data."""

    monitors_just_field = (
        td.FieldMonitor(
            size=(1, 1, 0),
            center=(0, 0, 0),
            freqs=[FREQ0],
            name="field",
        ),
    )

    monitors_with_flux = (
        td.FieldMonitor(size=(1, 1, 0), center=(0, 0, 0), freqs=[FREQ0], name="field"),
        td.FluxMonitor(
            size=(1, 1, 0), center=(0, 0, 0), freqs=[FREQ0 - FWIDTH, FREQ0 + FWIDTH], name="flux"
        ),
    )

    def objective_with_monitors(monitors):
        def objective(params):
            structure_traced = make_structures(params)["medium"]
            sim = SIM_BASE.updated_copy(structures=(structure_traced,), monitors=monitors)
            data = run(sim, task_name="adjoint_freq_test")
            assert data.simulation._freqs_adjoint == [FREQ0]
            return anp.sum(data["field"].flux.values)

        return objective

    grad_no_flux_monitors = ag.grad(objective_with_monitors(monitors_just_field))(params0)
    grad_with_flux_monitors = ag.grad(objective_with_monitors(monitors_with_flux))(params0)


def test_dispersive_no_inf(use_emulated_run):
    """Test that automatic permittivity grabbing uses the correct freq_adj to
    retrieve permittivity in dispersive material models.
    """

    fn_dict = get_functions(args[0][0], args[0][1])
    make_sim = fn_dict["sim"]
    postprocess = fn_dict["postprocess"]

    def objective(args):
        structure_traced = make_structures(args)["polyslab_dispersive"]
        sim = make_sim(args).updated_copy(structures=(structure_traced,))
        sim_data = run(sim, task_name="adjoint_test", verbose=False)
        return postprocess(sim_data)

    # Fail if dispersive eps_model() is evaluated without a proper finite frequency.
    with AssertLogStr("WARNING", excludes_str="frequency passed to 'Medium.eps_model()'"):
        grad = ag.grad(objective)(params0)
    assert np.all(np.isfinite(grad))


def test_sim_traced_center_size(use_emulated_run):
    fn_dict = get_functions(args[0][0], args[0][1])
    make_sim = fn_dict["sim"]
    postprocess = fn_dict["postprocess"]
    base_sim = make_sim(params0)

    def objective(center, size):
        sim = base_sim.updated_copy(center=center, size=size)
        sim_data = run_emulated(sim, task_name="adjoint_test")
        return postprocess(sim_data)

    with (
        AssertLogLevel("WARNING", contains_str="autograd tracer"),
        pytest.warns(UserWarning, match="Output seems independent of input."),
    ):
        grad = ag.grad(objective, argnum=0)(base_sim.center, base_sim.size)

    with (
        AssertLogLevel("WARNING", contains_str="autograd tracer"),
        pytest.warns(UserWarning, match="Output seems independent of input."),
    ):
        grad = ag.grad(objective, argnum=1)(base_sim.center, base_sim.size)


def test_error_clip(use_emulated_run):
    """Make sure proper error raised if differentiating a ``ClipOperation``."""

    def objective(x):
        box1 = td.Box(center=(0, 0, 0), size=(x, x, x))
        box2 = td.Box(center=(1, 1, 1), size=(x, x, x))
        union = td.ClipOperation(operation="union", geometry_a=box1, geometry_b=box2)
        structure = td.Structure(geometry=union, medium=td.Medium(permittivity=2))
        sim = SIM_BASE.updated_copy(
            structures=(structure,),
            monitors=(
                td.FieldMonitor(size=(0, 0, 0), center=(0, 0, 0), freqs=[FREQ0], name="field"),
            ),
        )
        data = run(sim, task_name="clip_error")
        return anp.sum(data["field"].intensity.item())

    with pytest.raises(ValueError):
        g = ag.grad(objective)(1.0)


def test_custom_medium_conductivity_only_gradient(rng, use_emulated_run, tmp_path):
    """Test conductivity gradients for CustomMedium with constant permittivity."""

    monitor, postprocess = make_monitors()["field_point"]

    def objective(params):
        """Objective function testing only conductivity gradient (constant permittivity)."""
        len_arr = np.prod(DA_SHAPE)
        matrix = rng.random((len_arr, N_PARAMS))

        # constant permittivity
        eps_arr = np.ones(DA_SHAPE) * 2.0

        # variable conductivity
        conductivity_arr = 0.05 * (anp.tanh(3 * matrix @ params).reshape(DA_SHAPE) + 1)

        nx, ny, nz = DA_SHAPE
        coords = {
            "x": np.linspace(-0.5, 0.5, nx),
            "y": np.linspace(-0.5, 0.5, ny),
            "z": np.linspace(-0.5, 0.5, nz),
        }

        custom_med_struct = td.Structure(
            geometry=td.Box(center=(0, 0, 0), size=(1, 1, 1)),
            medium=td.CustomMedium(
                permittivity=td.SpatialDataArray(eps_arr, coords=coords),
                conductivity=td.SpatialDataArray(conductivity_arr, coords=coords),
            ),
        )

        sim = SIM_BASE.updated_copy(
            structures=[custom_med_struct],
            monitors=[monitor],
        )

        data = run(
            sim,
            path=str(tmp_path / "sim_test.hdf5"),
            task_name="conductivity_only_grad_test",
            verbose=False,
        )
        return postprocess(data, data[monitor.name])

    val, grad = ag.value_and_grad(objective)(params0)

    assert anp.all(grad != 0.0), "some gradients are 0 for conductivity-only test"


@pytest.mark.parametrize("use_run_async", (False, True))
def test_error_custom_medium_and_geometry_traced(rng, use_run_async, use_emulated_run, tmp_path):
    """Test that we properly error when there is a combination of custom medium and
    geometry gradients."""
    monitor, postprocess = make_monitors()["field_point"]

    def objective(all_params):
        """Objective function testing only conductivity gradient (constant permittivity)."""
        params = all_params[0:-3]
        size_params = all_params[-3:]

        len_arr = np.prod(DA_SHAPE)
        matrix = rng.random((len_arr, N_PARAMS))

        # variable permittivity
        eps_arr = 1.5 + 1.5 * (anp.tanh(3 * matrix @ params).reshape(DA_SHAPE) + 1)

        nx, ny, nz = DA_SHAPE
        coords = {
            "x": np.linspace(-0.5, 0.5, nx),
            "y": np.linspace(-0.5, 0.5, ny),
            "z": np.linspace(-0.5, 0.5, nz),
        }

        custom_med_struct = td.Structure(
            geometry=td.Box(center=(0, 0, 0), size=tuple(size_params)),
            medium=td.CustomMedium(
                permittivity=td.SpatialDataArray(eps_arr, coords=coords),
            ),
        )

        sim = SIM_BASE.updated_copy(
            structures=[custom_med_struct],
            monitors=[monitor],
        )

        if use_run_async:
            data = run_async(
                [sim],
                path_dir=str(tmp_path),
                verbose=False,
            )[0]
        else:
            data = run(
                sim,
                path=str(tmp_path / "sim_test.hdf5"),
                task_name="error_custom_medium_and_geometry_traced_test",
                verbose=False,
            )
        return postprocess(data, data[monitor.name])

    box_sizes = [1.0, 1.0, 1.0]
    all_params = np.array(list(params0) + box_sizes)

    with pytest.raises(
        AdjointError,
        match="Detected structure at index 0 containing a CustomMedium "
        "type and traced geometry attributes.",
    ):
        val, grad = ag.value_and_grad(objective)(all_params)


@pytest.mark.parametrize("structure_key, monitor_key", args)
def test_vjp_nan(use_emulated_run, structure_key, monitor_key):
    """Test vjp data that has nan in it is flagged as an error."""

    fn_dict = get_functions(structure_key, monitor_key)
    make_sim = fn_dict["sim"]
    postprocess = fn_dict["postprocess"]

    def objective(*args):
        """Objective function."""
        sim = make_sim(*args)
        if PLOT_SIM:
            plot_sim(sim, plot_eps=True)
        data = run(sim, task_name="autograd_test", verbose=False)
        value = (postprocess(data) + float("nan")) ** 2
        return value

    with pytest.raises(AdjointError, match="aN values detected for data field"):
        grad = ag.grad(objective)(params0)


@pytest.mark.parametrize("monitor_key", ("mode",))
def test_autograd_polyslab_sidewall(use_emulated_run, monitor_key):
    """Sidewall-angle gradient propagates via autograd."""
    monitor, _ = make_monitors()[monitor_key]

    def make_polyslab(theta):
        verts = anp.array([[-0.4, -0.3], [0.4, -0.3], [0.4, 0.3], [-0.4, 0.3]])
        return td.PolySlab(
            vertices=verts,
            slab_bounds=(-0.5, 0.5),
            axis=POLYSLAB_AXIS,
            sidewall_angle=theta,
            dilation=0.0,
        )

    def make_sim(theta):
        geom = make_polyslab(theta)
        struct = td.Structure(geometry=geom, medium=td.Medium(permittivity=2.5))
        return SIM_BASE.updated_copy(structures=[struct], monitors=[monitor])

    def objective(theta_raw):
        theta = 0.30 * anp.tanh(theta_raw)
        sim = make_sim(theta)
        data = run(sim, task_name="autograd_sidewall_e2e", verbose=False)
        return anp.sum(anp.abs(data[monitor.name].amps)).item()

    val, grad = ag.value_and_grad(objective)(0.2)

    assert np.isfinite(val)
    assert grad != 0.0


def test_frequency_coordinate_alignment():
    """Test that frequency coordinate handling is robust to floating-point drift.

    Regression test for FXC-4349: KeyError in adjoint postprocessing due to
    frequency coordinate mismatch between forward and adjoint data.
    """
    from tidy3d.web.api.autograd.backward import _slice_field_data

    # Typical optical frequency
    freq = 2e14

    # Create field data with exact frequency (single frequency, single value)
    data = xr.DataArray(
        np.array([1.0]),
        coords={"f": [freq]},
        dims=["f"],
    )
    field_data = {"Ex": data, "Ey": data, "Ez": data}

    # Test 1: Exact match should work
    result = _slice_field_data(field_data, slice(0, 1))
    assert len(result) == 3
    assert all(k in result for k in ["Ex", "Ey", "Ez"])

    # Test 2: Component indicator filtering works
    result_e_only = _slice_field_data(field_data, slice(0, 1), component_indicator="E")
    assert len(result_e_only) == 3

    # Test 3: Multiple frequencies
    freqs_multi = [1e14, 2e14, 3e14]
    data_multi = xr.DataArray(
        np.array([1.0, 2.0, 3.0]),
        coords={"f": freqs_multi},
        dims=["f"],
    )
    field_data_multi = {"Ex": data_multi}

    # Selecting subset should work
    result_subset = _slice_field_data(
        field_data_multi, slice(freqs_multi.index(2e14), 1 + freqs_multi.index(2e14))
    )
    assert result_subset["Ex"].sizes["f"] == 1

    # Selecting non-existent frequency should fail
    with pytest.raises(IndexError):
        _slice_field_data(field_data_multi, slice(len(freqs_multi), len(freqs_multi) + 1))

    with pytest.raises(IndexError):
        _slice_field_data(field_data_multi, slice(-1, len(freqs_multi)))


def test_require_freq_ascending_rejects_descending_coordinates():
    """Adjoint postprocessing should reject descending frequency coordinates."""

    from tidy3d.web.api.autograd.backward import _require_freq_ascending

    data_desc = xr.DataArray(
        np.array([2.0, 1.0]),
        coords={"f": [2e14, 1e14]},
        dims=["f"],
    )
    field_data = {"Ex": data_desc}

    class DummyDataset:
        def __init__(self, field_components):
            self.field_components = field_components

    with pytest.raises(ValueError, match="expects ascending frequency coordinates"):
        _require_freq_ascending(
            DummyDataset(field_data),
            component_type="structure",
            component_index=0,
            dataset_name="forward field data",
        )


def _make_field_data_for_frequency_invariant_test(
    data_freqs: list[float], monitor_freqs: list[float] | None = None
) -> td.FieldData:
    """Create minimal field data with configurable data and monitor frequency order."""
    monitor_freqs = data_freqs if monitor_freqs is None else monitor_freqs
    coords = {"x": [0.0], "y": [0.0], "z": [0.0], "f": data_freqs}
    ex = td.ScalarFieldDataArray(np.ones((1, 1, 1, len(data_freqs))), coords=coords)
    monitor = td.FieldMonitor(
        center=(0, 0, 0),
        size=(0, 0, 0),
        freqs=monitor_freqs,
        fields=["Ex"],
        name="field",
    )
    return td.FieldData(monitor=monitor, Ex=ex)


def test_validate_adjoint_frequencies_accepts_exact_ascending_match():
    """Adjoint postprocessing should accept identical ascending frequency order."""

    from tidy3d.web.api.autograd.backward import _validate_adjoint_frequencies

    freqs = np.array([1e14, 2e14, 3e14])
    _validate_adjoint_frequencies(
        adjoint_frequencies=freqs,
        monitor_freqs=freqs,
        component_type="structure",
        component_index=0,
    )


def test_validate_adjoint_frequencies_rejects_monitor_order_mismatch():
    """Adjoint postprocessing should reject monitor frequencies in a different order."""

    from tidy3d.web.api.autograd.backward import _validate_adjoint_frequencies

    with pytest.raises(ValueError, match="Frequency mismatch in adjoint postprocessing"):
        _validate_adjoint_frequencies(
            adjoint_frequencies=np.array([1e14, 2e14, 3e14]),
            monitor_freqs=np.array([3e14, 2e14, 1e14]),
            component_type="structure",
            component_index=0,
        )


def test_filter_frequency_data_updates_monitor_and_preserves_ascending_order():
    """Filtered forward field data should preserve ascending monitor and coordinate order."""

    from tidy3d.web.api.autograd.backward import _filter_frequency_data

    fld_fwd = _make_field_data_for_frequency_invariant_test([1e14, 2e14, 3e14])
    filtered = _filter_frequency_data(
        fld_fwd,
        np.array([1e14, 3e14]),
        component_type="structure",
        component_index=0,
        dataset_name="filtered forward field data",
    )

    np.testing.assert_allclose(np.asarray(filtered.monitor.freqs, dtype=float), [1e14, 3e14])
    np.testing.assert_allclose(
        np.asarray(filtered.Ex.coords["f"].values, dtype=float), [1e14, 3e14]
    )


def test_filtered_forward_field_data_matches_adjoint_monitor_frequencies():
    """Filtered forward field data should match the adjoint monitor frequencies exactly."""

    from tidy3d.web.api.autograd.backward import (
        _filter_frequency_data,
        _get_freq_coords,
        _validate_adjoint_frequencies,
    )

    adjoint_monitor_freqs = np.array([1e14, 3e14])
    fld_fwd = _make_field_data_for_frequency_invariant_test([1e14, 2e14, 3e14])
    filtered = _filter_frequency_data(
        fld_fwd,
        adjoint_monitor_freqs,
        component_type="structure",
        component_index=0,
        dataset_name="filtered forward field data",
    )

    _validate_adjoint_frequencies(
        adjoint_frequencies=_get_freq_coords(filtered),
        monitor_freqs=adjoint_monitor_freqs,
        component_type="structure",
        component_index=0,
    )


def test_filter_frequency_data_rejects_descending_filtered_order():
    """Filtered forward field data should reject descending frequency order."""

    from tidy3d.web.api.autograd.backward import _filter_frequency_data

    fld_fwd = _make_field_data_for_frequency_invariant_test([1e14, 2e14])

    with pytest.raises(ValueError, match="expects ascending frequency coordinates"):
        _filter_frequency_data(
            fld_fwd,
            np.array([2e14, 1e14]),
            component_type="structure",
            component_index=0,
            dataset_name="filtered forward field data",
        )


def test_validate_adjoint_frequencies_rejects_filtered_forward_data_mismatch():
    """Filtered forward field data should fail if it does not match adjoint monitor frequencies."""

    from tidy3d.web.api.autograd.backward import (
        _filter_frequency_data,
        _get_freq_coords,
        _validate_adjoint_frequencies,
    )

    fld_fwd = _make_field_data_for_frequency_invariant_test([1e14, 2e14, 3e14])
    filtered = _filter_frequency_data(
        fld_fwd,
        np.array([1e14, 3e14]),
        component_type="structure",
        component_index=0,
        dataset_name="filtered forward field data",
    )

    with pytest.raises(ValueError, match="Frequency mismatch in adjoint postprocessing"):
        _validate_adjoint_frequencies(
            adjoint_frequencies=_get_freq_coords(filtered),
            monitor_freqs=np.array([1e14, 2e14]),
            component_type="structure",
            component_index=0,
        )


def test_resolve_freq_chunk_size_falls_back_without_memory(monkeypatch):
    """Fallback chunk size is used when available memory cannot be determined."""

    from tidy3d.web.api.autograd import backward as backward_module

    n_freqs = 10
    fallback_num_freqs = 4
    monkeypatch.setattr(backward_module.system_utils, "get_available_memory_bytes", lambda: -1)
    chunk_size = backward_module._resolve_freq_chunk_size(
        n_freqs=n_freqs,
        max_freqs_from_budget=lambda _available_bytes: n_freqs,
        fallback_num_freqs=fallback_num_freqs,
    )
    assert chunk_size == min(n_freqs, fallback_num_freqs)


def test_resolve_freq_chunk_size_auto_scales_to_budget(monkeypatch):
    """Auto chunk size uses the current memory heuristic."""

    from tidy3d.web.api.autograd import backward as backward_module

    n_freqs = 10
    combined_data_size = 100
    available_bytes = 1000
    monkeypatch.setattr(
        backward_module.system_utils, "get_available_memory_bytes", lambda: available_bytes
    )
    chunk_size = backward_module._resolve_freq_chunk_size(
        n_freqs=n_freqs,
        max_freqs_from_budget=lambda available_bytes: int(
            n_freqs
            * (
                int(available_bytes * config.adjoint.memory_allotment_fraction)
                - backward_module.ADJOINT_MEMORY_BASELINE_MULTIPLIER * combined_data_size
            )
            / (combined_data_size * backward_module.ADJOINT_MEMORY_MULTIPLIER)
        )
        if combined_data_size > 0
        else n_freqs,
    )
    budget_bytes = int(available_bytes * config.adjoint.memory_allotment_fraction)
    numerator = (
        budget_bytes - backward_module.ADJOINT_MEMORY_BASELINE_MULTIPLIER * combined_data_size
    )
    denominator = combined_data_size * backward_module.ADJOINT_MEMORY_MULTIPLIER
    expected = max(1, min(n_freqs, int(n_freqs * numerator / denominator)))
    assert chunk_size == expected


def test_resolve_freq_chunk_size_warns_when_auto_forces_chunk_one(monkeypatch):
    """Auto chunk sizing warns when even one frequency appears over budget."""

    from tidy3d.web.api.autograd import backward as backward_module

    n_freqs = 10
    combined_data_size = 100
    available_bytes = 300
    monkeypatch.setattr(
        backward_module.system_utils, "get_available_memory_bytes", lambda: available_bytes
    )
    with AssertLogLevel("WARNING", contains_str="one frequency per chunk"):
        chunk_size = backward_module._resolve_freq_chunk_size(
            n_freqs=n_freqs,
            max_freqs_from_budget=lambda available_bytes: int(
                n_freqs
                * (
                    int(available_bytes * config.adjoint.memory_allotment_fraction)
                    - backward_module.ADJOINT_MEMORY_BASELINE_MULTIPLIER * combined_data_size
                )
                / (combined_data_size * backward_module.ADJOINT_MEMORY_MULTIPLIER)
            )
            if combined_data_size > 0
            else n_freqs,
        )
    budget_bytes = int(available_bytes * config.adjoint.memory_allotment_fraction)
    numerator = (
        budget_bytes - backward_module.ADJOINT_MEMORY_BASELINE_MULTIPLIER * combined_data_size
    )
    denominator = combined_data_size * backward_module.ADJOINT_MEMORY_MULTIPLIER
    expected = max(1, min(n_freqs, int(n_freqs * numerator / denominator)))
    assert chunk_size == expected


def test_resolve_freq_chunk_size_falls_back_without_available_memory(monkeypatch):
    """Unavailable memory information falls back to the configured/default behavior."""

    from tidy3d.web.api.autograd import backward as backward_module

    monkeypatch.setattr(backward_module.system_utils, "get_available_memory_bytes", lambda: -1)

    assert (
        backward_module._resolve_freq_chunk_size(
            n_freqs=10, max_freqs_from_budget=lambda _available_bytes: 3, fallback_num_freqs=10
        )
        == 10
    )
    assert (
        backward_module._resolve_freq_chunk_size(
            n_freqs=10, max_freqs_from_budget=lambda _available_bytes: 3, fallback_num_freqs=4
        )
        == 4
    )


def test_geometry_group_passes_intersected_bounds_to_children():
    """GeometryGroup should clip bounds_intersect for each child geometry."""

    @dataclass
    class SimpleDerivativeInfo:
        paths: list[tuple]
        bounds: tuple
        bounds_intersect: tuple
        simulation_bounds: tuple
        interpolators: dict | None = None
        cached_min_spacing_from_permittivity: float | None = None
        eps_data = {
            key: td.ScalarFieldDataArray(
                [[[[2.0]]]], coords={"x": [0], "y": [0], "z": [0], "f": [200e12]}
            )
            for key in ["eps_xx", "eps_yy", "eps_zz"]
        }
        frequencies = [200e12]

        cache_min_spacing_from_permittivity = DerivativeInfo.cache_min_spacing_from_permittivity
        min_spacing_from_permittivity = DerivativeInfo.min_spacing_from_permittivity
        wavelength_min = property(lambda self: DerivativeInfo.wavelength_min.fget(self))

        def create_interpolators(self, dtype: float = float):
            return self.interpolators or {}

        def updated_copy(self, **kwargs):
            data = {
                "paths": self.paths,
                "bounds": self.bounds,
                "bounds_intersect": self.bounds_intersect,
                "simulation_bounds": self.simulation_bounds,
                "interpolators": self.interpolators,
                "cached_min_spacing_from_permittivity": self.cached_min_spacing_from_permittivity,
            }
            data.update({k: v for k, v in kwargs.items() if k in data})
            return SimpleDerivativeInfo(**data)

    fully_inside_box = Box(center=(-1.0, 0.0, 0.0), size=(1.0, 1.0, 1.0))
    big_box = Box(center=(0.0, 0.0, 0.0), size=(10.0, 10.0, 10.0))

    def record_method(self, derivative_info):
        object.__setattr__(self, "recorded_bounds_intersect", derivative_info.bounds_intersect)
        return {derivative_info.paths[0]: 0.0}

    boxes = (fully_inside_box, big_box)

    for box in boxes:
        object.__setattr__(box, "recorded_bounds_intersect", None)
        object.__setattr__(box, "_compute_derivatives", MethodType(record_method, box))
    group = GeometryGroup(geometries=boxes)

    # case where group bounds bigger than sim bounds
    sim_bounds = ((-5.0, -5.0, -5.0), (5.0, 5.0, 5.0))

    deriv_info = SimpleDerivativeInfo(
        paths=[("geom", idx, "dummy") for idx, _ in enumerate(boxes)],
        bounds=group.bounds,
        bounds_intersect=Geometry.bounds_intersection(group.bounds, sim_bounds),
        simulation_bounds=sim_bounds,
        interpolators={},
    )

    group._compute_derivatives(deriv_info)

    assert (
        object.__getattribute__(fully_inside_box, "recorded_bounds_intersect")
        == fully_inside_box.bounds
    )
    assert object.__getattribute__(big_box, "recorded_bounds_intersect") == sim_bounds

    # case where sim bounds bigger than group bounds
    sim_bounds = ((-20.0, -20.0, -20.0), (20.0, 20.0, 20.0))

    deriv_info = SimpleDerivativeInfo(
        paths=[("geom", idx, "dummy") for idx, _ in enumerate(boxes)],
        bounds=group.bounds,
        bounds_intersect=Geometry.bounds_intersection(group.bounds, sim_bounds),
        simulation_bounds=sim_bounds,
        interpolators={},
    )

    group._compute_derivatives(deriv_info)

    assert object.__getattribute__(big_box, "recorded_bounds_intersect") == group.bounds, (
        f"got {object.__getattribute__(big_box, 'recorded_bounds_intersect')} and {group.bounds}"
    )


@pytest.mark.parametrize("monitor_key", ("mode",))
def test_autograd_sphere_0_radius(use_emulated_run, monitor_key):
    """Integration test that Sphere gradients are non-zero (mirrors cylinder check)."""

    monitor, postprocess = make_monitors()[monitor_key]

    def make_sphere(radius, x0, y0, z0):
        return td.Sphere(center=(x0, y0, z0), radius=radius)

    def make_sim(params):
        geometry = make_sphere(*params)
        structure = td.Structure(geometry=geometry, medium=td.Medium(permittivity=2))
        return SIM_BASE.updated_copy(structures=[structure], monitors=[monitor])

    p0 = [0.0, 0.0, 0.0, 0.0]

    def objective(params):
        sim = make_sim(params)
        if PLOT_SIM:
            plot_sim(sim, plot_eps=True)
        data = run(sim, task_name="autograd_test", verbose=False)
        return anp.sum(anp.abs(data[monitor.name].amps)).item()

    with AssertLogLevel("WARNING", contains_str="cannot be computed"):
        val_sphere, grad_sphere = ag.value_and_grad(objective)(p0)
    # first 4 parameters are related to the geometry
    geom_grad = np.asarray(get_static(grad_sphere[:4]), dtype=float)
    assert np.allclose(geom_grad, 0.0)
