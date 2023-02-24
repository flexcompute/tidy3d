"""Tests adjoint plugin."""

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

from tidy3d.log import DataError, Tidy3dKeyError
from tidy3d.plugins.adjoint.components.base import JaxObject
from tidy3d.plugins.adjoint.components.geometry import JaxBox, JaxPolySlab
from tidy3d.plugins.adjoint.components.medium import JaxMedium, JaxAnisotropicMedium
from tidy3d.plugins.adjoint.components.medium import JaxCustomMedium
from tidy3d.plugins.adjoint.components.structure import JaxStructure
from tidy3d.plugins.adjoint.components.simulation import JaxSimulation
from tidy3d.plugins.adjoint.components.data.sim_data import JaxSimulationData
from tidy3d.plugins.adjoint.components.data.monitor_data import JaxModeData, JaxDiffractionData
from tidy3d.plugins.adjoint.components.data.data_array import JaxDataArray
from tidy3d.plugins.adjoint.components.data.dataset import JaxPermittivityDataset
from tidy3d.plugins.adjoint.web import run, run_async
from tidy3d.plugins.adjoint.log import AdjointError

from ..utils import run_emulated, assert_log_level, run_async_emulated


EPS = 2.0
SIZE = (1.0, 2.0, 3.0)
CENTER = (2.0, -1.0, 1.0)
VERTICES = ((-1.0, -1.0), (0.0, 0.0), (-1.0, 0.0))
POLYSLAB_AXIS = 2
FREQ0 = 2e14
BASE_EPS_VAL = 2.0

# name of the output monitor used in tests
MNT_NAME = "mode"


def make_sim(
    permittivity: float, size: Tuple[float, float, float], vertices: tuple, base_eps_val: float
) -> JaxSimulation:
    """Construt a simulation out of some input parameters."""

    box = td.Box(size=(1, 1, 1), center=(1, 2, 2))
    med = td.Medium(permittivity=2.0)
    extraneous_structure = td.Structure(geometry=box, medium=med)

    # NOTE: Any new input structures should be added below as they are made

    # JaxBox
    jax_box1 = JaxBox(size=size, center=(1, 0, 2))
    jax_med1 = JaxMedium(permittivity=permittivity)
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
    Nx, Ny, Nz = 10, 10, 1
    (xmin, ymin, zmin), (xmax, ymax, zmax) = jax_box1.bounds
    coords = dict(
        x=np.linspace(xmin, xmax, Nx).tolist(),
        y=np.linspace(ymin, ymax, Ny).tolist(),
        z=np.linspace(zmin, zmax, Nz).tolist(),
        f=[FREQ0],
    )

    values = base_eps_val + np.random.random((Nx, Ny, Nz, 1))
    eps_ii = JaxDataArray(values=values, coords=coords)
    field_components = {f"eps_{dim}{dim}": eps_ii for dim in "xyz"}
    jax_eps_dataset = JaxPermittivityDataset(**field_components)
    jax_med_custom = JaxCustomMedium(eps_dataset=jax_eps_dataset)
    jax_struct_custom = JaxStructure(geometry=jax_box1, medium=jax_med_custom)

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
        size=(10, 2, 0),
        freqs=[FREQ0],
        name=MNT_NAME + "3",
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
        output_monitors=(output_mnt1, output_mnt2, output_mnt3),
        input_structures=(jax_struct1, jax_struct2, jax_struct3, jax_struct_custom),
        # input_structures=(jax_struct_custom,),
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


@pytest.fixture
def use_emulated_run_async(monkeypatch):
    """If this fixture is used, the `tests.utils.run_emulated` function is used for simulation."""
    import tidy3d.plugins.adjoint.web as adjoint_web

    monkeypatch.setattr(adjoint_web, "tidy3d_run_async_fn", run_async_emulated)


def test_adjoint_pipeline(use_emulated_run):
    """Test computing gradient using jax."""

    sim = make_sim(permittivity=EPS, size=SIZE, vertices=VERTICES, base_eps_val=BASE_EPS_VAL)
    sim_data = run(sim, task_name="test")

    def f(permittivity, size, vertices, base_eps_val):
        sim = make_sim(
            permittivity=permittivity, size=size, vertices=vertices, base_eps_val=base_eps_val
        )
        sim_data = run(sim, task_name="test")
        amp = extract_amp(sim_data)
        return objective(amp)

    grad_f = grad(f, argnums=(0, 1, 2, 3))
    df_deps, df_dsize, df_dvertices, d_eps_base = grad_f(EPS, SIZE, VERTICES, BASE_EPS_VAL)

    print("gradient: ", df_deps, df_dsize, df_dvertices, d_eps_base)


def test_adjoint_setup_fwd(use_emulated_run):
    """Test that the forward pass works as expected."""
    sim = make_sim(permittivity=EPS, size=SIZE, vertices=VERTICES, base_eps_val=BASE_EPS_VAL)
    sim_data_orig, (sim_data_fwd,) = run.fwd(
        simulation=sim,
        task_name="test",
        folder_name="default",
        path="simulation_data.hdf5",
        callback_url=None,
        verbose=False,
    )
    sim_orig = sim_data_orig.simulation
    sim_fwd = sim_data_fwd.simulation

    # check the cached objects are as expected
    assert sim_orig == sim, "original simulation wasnt cached properly"
    assert len(sim_orig.monitors) == len(sim_data_fwd.data) == len(sim_data_orig.data)
    assert len(sim_orig.output_monitors) == len(sim_data_fwd.output_data)
    assert len(sim_orig.input_structures) == len(sim_data_fwd.grad_data)
    assert len(sim_data_fwd.grad_data) == len(sim_fwd.grad_monitors)


def _test_adjoint_setup_adj(use_emulated_run):
    """Test that the adjoint pass works as expected."""
    sim_orig = make_sim(permittivity=EPS, size=SIZE, vertices=VERTICES, base_eps_val=BASE_EPS_VAL)

    # call forward pass
    sim_data_fwd, (sim_data_fwd,) = run.fwd(
        simulation=sim_orig,
        task_name="test",
        folder_name="default",
        path="simulation_data.hdf5",
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
        path="simulation_data.hdf5",
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

    with pytest.raises(AdjointError):
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
    with pytest.raises(AdjointError):
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
    with pytest.raises(NotImplementedError):
        da.interp(b=2.5)


def test_jax_sim_data(use_emulated_run):
    """Test mechanics of the JaxSimulationData."""

    sim = make_sim(permittivity=EPS, size=SIZE, vertices=VERTICES, base_eps_val=BASE_EPS_VAL)
    sim_data = run(sim, task_name="test")

    for i in range(len(sim.output_monitors)):
        mnt_name = MNT_NAME + str(i + 1)
        mnt_data_a = sim_data.output_data[i]
        mnt_data_b = sim_data.output_monitor_data[mnt_name]
        mnt_data_c = sim_data[mnt_name]


def test_intersect_structures(caplog):
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
    assert_log_level(caplog, "warning")


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
    with pytest.raises(AdjointError):
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
    """Make sure box made with polyslab gives equivalent gradients (note, doesn't pass now)."""

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

            pos_x1 = cent_1 - size_1 / 2.0
            pos_x2 = cent_1 + size_1 / 2.0
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
            size=(10, 10, 0),
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

        extraneous_field_monitor = td.FieldMonitor(
            size=(10, 10, 0),
            freqs=[1e14, 2e14],
            name="field",
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

    assert np.allclose(gs_b, gs_p), f"size gradients dont match, got {gs_b} and {gs_p}"
    assert np.allclose(gc_b, gc_p), f"center gradients dont match, got {gc_b} and {gc_p}"


# @pytest.mark.asyncio
def test_adjoint_run_async(use_emulated_run_async):
    """Test differnetiating thorugh async adjoint runs"""

    def make_sim_simple(permittivity: float) -> JaxSimulation:
        """Make a sim as a function of a single parameter."""
        return make_sim(
            permittivity=permittivity, size=SIZE, vertices=VERTICES, base_eps_val=BASE_EPS_VAL
        )

    def f(x):
        """Objective function to differentiate."""

        sims = []
        for i in range(2):
            permittivity = x + float(1.0 + i)
            sims.append(make_sim_simple(permittivity=permittivity))

        sim_data_list = run_async(sims)

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
