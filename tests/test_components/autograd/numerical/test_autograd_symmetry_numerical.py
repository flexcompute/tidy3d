# test autograd for field sources when symmetry is used in the simulation
from __future__ import annotations

import operator
from pathlib import Path

import autograd as ag
import matplotlib.pylab as plt
import numpy as np
import pytest

import tidy3d as td
import tidy3d.web as web

PLOT_SYMMETRY_COMPARISON = False
NUM_FINITE_DIFFERENCE = 10
SAVE_FD_ADJ_DATA = False
SAVE_FD_LOC = 0
SAVE_ADJ_LOC = 1
LOCAL_GRADIENT = False
VERBOSE = False

RMS_THRESHOLD = 0.25

if PLOT_SYMMETRY_COMPARISON:
    pytestmark = pytest.mark.usefixtures("mpl_config_interactive")
else:
    pytestmark = pytest.mark.usefixtures("mpl_config_noninteractive")


FINITE_DIFF_PERM_SEED = 1.5**2
MESH_FACTOR_DESIGN = 30.0


def get_sim_geometry(mesh_wvl_um):
    return td.Box(size=(5 * mesh_wvl_um, 5 * mesh_wvl_um, 7 * mesh_wvl_um), center=(0, 0, 0))


def make_base_sim(
    mesh_wvl_um,
    adj_wvl_um,
    monitor_size_wvl,
    box_for_override,
    symmetry,
    monitor_bg_index=1.0,
    run_time=1e-11,
):
    sim_geometry = get_sim_geometry(mesh_wvl_um)
    sim_size_um = sim_geometry.size
    sim_center_um = sim_geometry.center

    boundary_spec = td.BoundarySpec(
        x=td.Boundary.pml(),
        y=td.Boundary.pml(),
        z=td.Boundary.pml(),
    )

    dl_design = mesh_wvl_um / MESH_FACTOR_DESIGN

    mesh_overrides = []
    mesh_overrides.extend(
        [
            td.MeshOverrideStructure(
                geometry=box_for_override,
                dl=[dl_design, dl_design, dl_design],
            ),
        ]
    )

    src_size = (*sim_size_um[0:2], 0)

    wl_min_src_um = 0.9 * adj_wvl_um
    wl_max_src_um = 1.1 * adj_wvl_um

    fwidth_src = td.C_0 * ((1.0 / wl_min_src_um) - (1.0 / wl_max_src_um))
    freq0 = td.C_0 / adj_wvl_um

    pulse = td.GaussianPulse(freq0=freq0, fwidth=fwidth_src)

    src = td.PlaneWave(
        center=(0, 0, -2 * mesh_wvl_um),
        size=(td.inf, td.inf, 0),
        direction="+",
        pol_angle=0,
        angle_theta=0,
        source_time=pulse,
    )

    field_monitor = td.FieldMonitor(
        center=(0, 0, 0.25 * sim_size_um[2]),
        size=tuple(dim * mesh_wvl_um for dim in monitor_size_wvl),
        name="monitor_fields",
        freqs=[freq0],
    )

    monitor_index_block = td.Box(
        center=(0, 0, 0.25 * sim_size_um[2] + mesh_wvl_um),
        size=(*tuple(2 * size for size in sim_size_um[0:2]), mesh_wvl_um + 0.5 * sim_size_um[2]),
    )
    monitor_index_block_structure = td.Structure(
        geometry=monitor_index_block, medium=td.Medium(permittivity=monitor_bg_index**2)
    )

    sim_base = td.Simulation(
        center=sim_center_um,
        size=sim_size_um,
        grid_spec=td.GridSpec.auto(
            min_steps_per_wvl=30,
            wavelength=mesh_wvl_um,
            override_structures=mesh_overrides,
        ),
        structures=[monitor_index_block_structure],
        sources=[src],
        monitors=[field_monitor],
        run_time=run_time,
        boundary_spec=boundary_spec,
        subpixel=True,
        symmetry=symmetry,
    )

    return sim_base


def create_objective_functions(geometry, create_sim_base, eval_fn, sim_path_dir):
    sim_path_dir = Path(sim_path_dir)
    sim_path_dir.mkdir(parents=True, exist_ok=True)

    def objective_(perm_array, symmetry):
        sim_base = create_sim_base(symmetry)

        block_structure = td.Structure.from_permittivity_array(
            eps_data=perm_array,
            geometry=geometry,
        )

        sim_with_block = sim_base.updated_copy(structures=(*sim_base.structures, block_structure))

        symmetry_tag = "_".join(str(val) for val in symmetry)
        result_path = sim_path_dir / f"symmetry_{symmetry_tag}.hdf5"

        sim_data = web.run(
            sim_with_block,
            task_name="symmetry_field_testing",
            path=str(result_path),
            local_gradient=LOCAL_GRADIENT,
            verbose=VERBOSE,
        )

        objective_val = eval_fn(sim_data)

        return objective_val

    def objective_no_symmetry(perm_array):
        return objective_(perm_array=perm_array, symmetry=(0, 0, 0))

    def objective_x_symmetry(perm_array):
        return objective_(perm_array=perm_array, symmetry=(-1, 0, 0))

    def objective_y_symmetry(perm_array):
        return objective_(perm_array=perm_array, symmetry=(0, 1, 0))

    def objective_xy_symmetry(perm_array):
        return objective_(perm_array=perm_array, symmetry=(-1, 1, 0))

    return objective_no_symmetry, objective_x_symmetry, objective_y_symmetry, objective_xy_symmetry


def make_eval_fns(monitor_size_wvl):
    num_nonzero_spatial_dims = 3 - np.sum(np.isclose(monitor_size_wvl, 0))

    def intensity(sim_data):
        field_data = sim_data["monitor_fields"]
        _shape_x, _shape_y, _shape_z, *_ = field_data.Ex.values.shape

        return np.sum(np.abs(field_data.Ex.values) ** 2 + np.abs(field_data.Ey.values) ** 2)

    eval_fns = [intensity]
    eval_fn_names = ["intensity"]

    if num_nonzero_spatial_dims == 2:

        def flux(sim_data):
            field_data = sim_data["monitor_fields"]

            return np.sum(field_data.flux.values)

        eval_fns.append(flux)
        eval_fn_names.append("flux")

    return eval_fns, eval_fn_names


background_indices = [1.0]
mesh_wvls_um = [1.55]
adj_wvls_um = [1.55]
monitor_sizes_3d_wvl = [(0.5, 0.5, 0)]

field_symmetry_test_parameters = []

test_number = 0
for idx in range(len(mesh_wvls_um)):
    mesh_wvl_um = mesh_wvls_um[idx]
    adj_wvl_um = adj_wvls_um[idx]

    for monitor_size_wvl in monitor_sizes_3d_wvl:
        eval_fns, eval_fn_names = make_eval_fns(monitor_size_wvl)

        for monitor_bg_index in background_indices:
            for eval_fn_idx, eval_fn in enumerate(eval_fns):
                field_symmetry_test_parameters.append(
                    {
                        "mesh_wvl_um": mesh_wvl_um,
                        "adj_wvl_um": adj_wvl_um,
                        "monitor_size_wvl": monitor_size_wvl,
                        "monitor_bg_index": monitor_bg_index,
                        "eval_fn": eval_fn,
                        "eval_fn_name": eval_fn_names[eval_fn_idx],
                        "test_number": test_number,
                    }
                )

                test_number += 1


@pytest.mark.numerical
@pytest.mark.parametrize("field_symmetry_test_parameters", field_symmetry_test_parameters)
def test_adjoint_difference_symmetry(
    field_symmetry_test_parameters, rng, numerical_case_dir, redirect_stdout_to_stderr
):
    """Test the gradient is not affected by symmetry when using field sources."""

    num_tests = 0
    for monitor_size_wvl in monitor_sizes_3d_wvl:
        eval_fns, _ = make_eval_fns(monitor_size_wvl)
        num_tests += len(eval_fns) * len(background_indices) * len(mesh_wvls_um)

    test_results = np.zeros((2, NUM_FINITE_DIFFERENCE))

    test_number = field_symmetry_test_parameters["test_number"]

    (
        mesh_wvl_um,
        adj_wvl_um,
        monitor_size_wvl,
        monitor_bg_index,
        eval_fn,
        eval_fn_name,
        test_number,
    ) = operator.itemgetter(
        "mesh_wvl_um",
        "adj_wvl_um",
        "monitor_size_wvl",
        "monitor_bg_index",
        "eval_fn",
        "eval_fn_name",
        "test_number",
    )(field_symmetry_test_parameters)

    dim_um = mesh_wvl_um
    dim_um = mesh_wvl_um
    thickness_um = 0.5 * mesh_wvl_um
    block = td.Box(center=(0, 0, 0), size=(dim_um, dim_um, thickness_um))

    dim = 1 + int(dim_um / (mesh_wvl_um / MESH_FACTOR_DESIGN))
    Nz = 1 + int(thickness_um / (mesh_wvl_um / MESH_FACTOR_DESIGN))

    sim_geometry = get_sim_geometry(mesh_wvl_um)

    box_for_override = td.Box(
        center=(0, 0, 0), size=(*sim_geometry.size[0:2], thickness_um + mesh_wvl_um)
    )

    eval_fns, _eval_fn_names = make_eval_fns(monitor_size_wvl)

    sim_path_dir = numerical_case_dir / "simulations" / f"test{test_number}"
    sim_path_dir.mkdir(parents=True, exist_ok=True)

    objective_no_symmetry, objective_x_symmetry, objective_y_symmetry, objective_xy_symmetry = (
        create_objective_functions(
            block,
            lambda symmetry,
            mesh_wvl_um=mesh_wvl_um,
            adj_wvl_um=adj_wvl_um,
            monitor_size_wvl=monitor_size_wvl,
            box_for_override=box_for_override,
            monitor_bg_index=monitor_bg_index: make_base_sim(
                mesh_wvl_um=mesh_wvl_um,
                adj_wvl_um=adj_wvl_um,
                monitor_size_wvl=monitor_size_wvl,
                box_for_override=box_for_override,
                monitor_bg_index=monitor_bg_index,
                symmetry=symmetry,
            ),
            eval_fn,
            sim_path_dir=str(sim_path_dir),
        )
    )

    obj_val_and_grad_no_symmetry = ag.value_and_grad(objective_no_symmetry)
    obj_val_and_grad_x_symmetry = ag.value_and_grad(objective_x_symmetry)
    obj_val_and_grad_y_symmetry = ag.value_and_grad(objective_y_symmetry)
    obj_val_and_grad_xy_symmetry = ag.value_and_grad(objective_xy_symmetry)

    objs_val_and_grad = [
        obj_val_and_grad_no_symmetry,
        obj_val_and_grad_x_symmetry,
        obj_val_and_grad_y_symmetry,
        obj_val_and_grad_xy_symmetry,
    ]

    symmetries = ["none", "x", "y", "xy"]

    objs = []
    adj_grads = []

    perm_init = FINITE_DIFF_PERM_SEED * np.ones((dim, dim, Nz))

    for obj_val_and_grad in objs_val_and_grad:
        obj, adj_grad = obj_val_and_grad(perm_init)

        objs.append(obj)
        adj_grads.append(np.array(adj_grad))

    grad_data_base = adj_grads[0] / objs[0]
    for idx in range(1, len(adj_grads)):
        # field magnitudes can be different for different symmetries so we expect the gradients
        # to scale with the objecive values
        grad_data = adj_grads[idx] / objs[idx]

        mag_base = np.sqrt(np.mean(grad_data_base**2))
        mag_compare = np.sqrt(np.mean(grad_data**2))
        rms_error = np.sqrt(np.mean((grad_data_base - grad_data) ** 2))

        print(f"Testing {eval_fn_name} objective")
        print(f"Symmetry comparison: {symmetries[0]}, {symmetries[idx]}")
        print(f"RMS error (normalized): {rms_error / np.sqrt(mag_base * mag_compare)}")

        assert np.isclose(rms_error / np.sqrt(mag_base * mag_compare), 0.0, atol=0.075), (
            "Expected adjoint gradients to be the same with and without symmetry"
        )

        if PLOT_SYMMETRY_COMPARISON:
            plot_grad_data_base = np.squeeze(grad_data_base)
            plot_grad_data = np.squeeze(grad_data)
            plot_diff = plot_grad_data - plot_grad_data_base

            plt.subplot(1, 3, 1)
            plt.imshow(plot_grad_data_base[:, :, plot_grad_data_base.shape[2] // 2])
            plt.title(f"Symmetry: {symmetries[0]}")
            plt.colorbar()
            plt.subplot(1, 3, 2)
            plt.imshow(plot_grad_data[:, :, plot_grad_data.shape[2] // 2])
            plt.title(f"Symmetry: {symmetries[idx]}")
            plt.colorbar()
            plt.subplot(1, 3, 3)
            plt.imshow(plot_diff[:, :, plot_diff.shape[2] // 2])
            plt.title("Difference")
            plt.colorbar()
            plt.show()
