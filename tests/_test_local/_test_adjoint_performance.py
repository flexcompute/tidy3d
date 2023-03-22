import pytest
import numpy as np
import os
import sys
from memory_profiler import profile
import matplotlib.pylab as plt
import time

from tidy3d.components.data.sim_data import SimulationData
from tidy3d.components.data.monitor_data import FieldData
from tidy3d.components.data.data_array import ScalarFieldDataArray
from tidy3d.components.monitor import FieldMonitor
from tidy3d.components.simulation import Simulation
from tidy3d.components.source import PointDipole, GaussianPulse
from tidy3d.components.grid.grid_spec import GridSpec


from typing import Callable, Tuple

import pytest
import pydantic
import jax.numpy as jnp
import numpy as np
from jax import grad, custom_vjp
import jax
from numpy.random import random

import tidy3d as td
from typing import Tuple, Any

from tidy3d.exceptions import DataError, Tidy3dKeyError, AdjointError
from tidy3d.plugins.adjoint.components.base import JaxObject
from tidy3d.plugins.adjoint.components.geometry import JaxBox, JaxPolySlab
from tidy3d.plugins.adjoint.components.medium import (
    JaxMedium,
    JaxAnisotropicMedium,
    JaxCustomMedium,
)
from tidy3d.plugins.adjoint.components.structure import JaxStructure
from tidy3d.plugins.adjoint.components.simulation import JaxSimulation
from tidy3d.plugins.adjoint.components.data.sim_data import JaxSimulationData
from tidy3d.plugins.adjoint.components.data.monitor_data import JaxModeData
from tidy3d.plugins.adjoint.components.data.data_array import JaxDataArray
from tidy3d.plugins.adjoint.components.data.dataset import JaxPermittivityDataset
from tidy3d.plugins.adjoint.web import run

from ..utils import run_emulated

import tidy3d as td

sys.path.append("/users/twhughes/Documents/Flexcompute/tidy3d-core")
from tidy3d_backend.utils import Profile


EPS = 2.0
SIZE = (1.0, 2.0, 3.0)
CENTER = (2.0, -1.0, 1.0)
VERTICES = ((-1.0, -1.0), (0.0, 0.0), (-1.0, 0.0))
POLYSLAB_AXIS = 2
FREQ0 = 2e14
BASE_EPS_VAL = 2.0

# name of the output monitor used in tests
MNT_NAME = "mode"


# 300 : 7448/7379
# 300 : 7448/7379


def make_sim(eps_values: np.ndarray) -> JaxSimulation:
    """Construt a simulation out of some input parameters."""

    Nx, Ny, Nz, Nf = eps_values.shape

    box = td.Box(size=(1, 1, 1), center=(1, 2, 2))
    med = td.Medium(permittivity=2.0)
    extraneous_structure = td.Structure(geometry=box, medium=med)

    # NOTE: Any new input structures should be added below as they are made

    # JaxBox
    jax_box = JaxBox(size=SIZE, center=(1, 0, 2))

    # custom medium
    (xmin, ymin, zmin), (xmax, ymax, zmax) = jax_box.bounds
    coords = dict(
        x=np.linspace(xmin, xmax, Nx).tolist(),
        y=np.linspace(ymin, ymax, Ny).tolist(),
        z=np.linspace(zmin, zmax, Nz).tolist(),
        f=[FREQ0],
    )

    eps_ii = JaxDataArray(values=eps_values, coords=coords)
    field_components = {f"eps_{dim}{dim}": eps_ii for dim in "xyz"}
    jax_eps_dataset = JaxPermittivityDataset(**field_components)
    jax_med_custom = JaxCustomMedium(eps_dataset=jax_eps_dataset)
    jax_struct_custom = JaxStructure(geometry=jax_box, medium=jax_med_custom)

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

    extraneous_field_monitor = td.FieldMonitor(
        size=(10, 10, 0),
        freqs=[1e14, 2e14],
        name="field",
    )

    sim = JaxSimulation(
        size=(10, 10, 10),
        run_time=1e-12,
        grid_spec=td.GridSpec(wavelength=1.0),
        monitors=(extraneous_field_monitor,),
        structures=(extraneous_structure,),
        output_monitors=(output_mnt1, output_mnt2),
        input_structures=(jax_struct_custom,),
    )

    return sim


def objective(amp: complex) -> float:
    """Objective function as a function of the complex amplitude."""
    return abs(amp) ** 2


def extract_amp(sim_data: td.SimulationData) -> complex:
    """get the amplitude from a simulation data object."""

    ret_value = 0.0

    # ModeData
    mnt_name = MNT_NAME + "1"
    mnt_data = sim_data.output_monitor_data[mnt_name]
    amps = mnt_data.amps
    ret_value += amps.sel(direction="+", f=2e14, mode_index=0)
    ret_value += amps.isel(direction=0, f=0, mode_index=0)
    ret_value += amps.sel(direction="-", f=2e14, mode_index=1)
    ret_value += amps.sel(mode_index=1, f=2e14, direction="-")
    ret_value += amps.sel(direction="-", f=2e14).isel(mode_index=1)

    # DiffractionData
    mnt_name = MNT_NAME + "2"
    mnt_data = sim_data.output_monitor_data[mnt_name]
    ret_value += mnt_data.amps.sel(orders_x=0, orders_y=0, f=2e14, polarization="p")
    ret_value += mnt_data.amps.sel(orders_x=-1, orders_y=1, f=2e14, polarization="p")
    ret_value += mnt_data.amps.isel(orders_x=0, orders_y=1, f=0, polarization=0)
    ret_value += mnt_data.Er.isel(orders_x=0, orders_y=1, f=0)
    ret_value += mnt_data.power.sel(orders_x=-1, orders_y=1, f=2e14)

    return ret_value


@pytest.fixture
def use_emulated_run(monkeypatch):
    """If this fixture is used, the `tests.utils.run_emulated` function is used for simulation."""
    import tidy3d.plugins.adjoint.web as adjoint_web

    monkeypatch.setattr(adjoint_web, "tidy3d_run_fn", run_emulated)


@profile
def test_large_custom_medium(use_emulated_run):
    def f(eps_values):
        sim = make_sim(eps_values=eps_values)
        sim_data = run(sim, task_name="test")
        amp = extract_amp(sim_data)
        return objective(amp)

    Nx, Ny, Nz = 100, 100, 1
    EPS_VALUES = 1 + np.random.random((Nx, Ny, Nz, 1))

    with Profile():
        grad_f = grad(f)
        df_eps_values = grad_f(EPS_VALUES)


def test_time_custom_medium(use_emulated_run):

    num_tests = 50
    nxs = np.logspace(0, 2.0, num_tests)
    times_sec = np.zeros(num_tests)
    num_pixels = np.zeros(num_tests)

    def f(eps_values):
        sim = make_sim(eps_values=eps_values)
        sim_data = run(sim, task_name="test")
        amp = extract_amp(sim_data)
        return objective(amp)

    g = grad(f)

    # run it once to compile for accurate comparison
    g(np.ones((1, 1, 1, 1)))

    for i, nx in enumerate(nxs):
        nx = int(nx)
        ny = nx
        eps_array = 1 + np.random.random((nx, ny, 1, 1))

        tstart = time.time()
        deps = g(eps_array)
        tend = time.time()
        times_sec[i] = tend - tstart
        num_pixels[i] = eps_array.size

    plt.scatter(num_pixels, times_sec)
    plt.plot(num_pixels, times_sec)
    plt.xlabel("number of pixels")
    plt.ylabel("grad(eps) time (sec)")
    plt.yscale("log")
    plt.xscale("log")
    plt.show()


def test_simple_jax_data_array(use_emulated_run):
    def make_data_array(values):
        coords = {}
        for dim, n in zip("xyzf", values.shape):
            coords[dim] = np.linspace(0, 1, n).tolist()
        return JaxDataArray(values=values, coords=coords)

    def make_custom_medium(values):
        components = {f"eps_{dim}{dim}": make_data_array(values) for dim in "xyz"}
        eps_dataset = JaxPermittivityDataset(**components)
        return JaxCustomMedium(eps_dataset=eps_dataset)

    def make_custom_structure(values):
        medium = make_custom_medium(values)
        geometry = JaxBox(size=(1, 1, 1), center=(0.5, 0.5, 0.5))
        return JaxStructure(geometry=geometry, medium=medium)

    def make_sim(values):
        structure = make_custom_structure(values)
        output_mnt = td.ModeMonitor(
            size=(10, 10, 0),
            mode_spec=td.ModeSpec(num_modes=3),
            freqs=[FREQ0],
            name="mode",
        )
        return JaxSimulation(
            size=(1, 1, 1),
            center=(0.5, 0.5, 0.5),
            grid_spec=td.GridSpec.auto(wavelength=0.1),
            input_structures=[structure],
            output_monitors=[output_mnt],
            run_time=1e-12,
        )

    Nx, Ny, Nz, Nf = 300, 300, 1, 1
    VALUES = 2 + np.random.random((Nx, Ny, Nz, Nf))
    data_array = make_data_array(VALUES)

    # the first 3 are fast, last one is slow

    def f(values):
        data_array = make_data_array(values)
        return jnp.sum(data_array.values)

    def f(values):
        custom_medium = make_custom_medium(values)
        return jnp.sum(custom_medium.eps_dataset.eps_xx.values)

    def f(values):
        custom_structure = make_custom_structure(values)
        return jnp.sum(custom_structure.medium.eps_dataset.eps_xx.values)

    def f(values):
        sim = make_sim(values)
        sim_data = run(sim, task_name="test")
        return abs(sim_data["mode"].amps.sel(f=FREQ0, mode_index=0, direction="+"))

    with Profile():
        g = grad(f)
        g(VALUES)
