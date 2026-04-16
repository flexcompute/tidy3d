"""Numerical diffraction-based shape-gradient checks for anisotropic PolySlab cases."""

from __future__ import annotations

import operator
import sys

import autograd as ag
import autograd.numpy as anp
import matplotlib.pylab as plt
import numpy as np
import pytest

import tidy3d as td
import tidy3d.web as web

PLOT_FD_ADJ_COMPARISON = False
NUM_FINITE_DIFFERENCE = 10
SAVE_FD_ADJ_DATA = False
SAVE_FD_LOC = 0
SAVE_ADJ_LOC = 1
LOCAL_GRADIENT = True
VERBOSE = False
NUMERICAL_RESULTS_SUBDIR = "numerical_anisotropic_shape_test"

RMS_THRESHOLD = 0.25

if PLOT_FD_ADJ_COMPARISON:
    pytestmark = pytest.mark.usefixtures("mpl_config_interactive")
else:
    pytestmark = pytest.mark.usefixtures("mpl_config_noninteractive")

MESH_FACTOR_DESIGN = 30.0
ROTATION_ANGLE_DEG = 25.0
ROTATION_ANGLE_RAD = np.deg2rad(ROTATION_ANGLE_DEG)


def get_sim_geometry(mesh_wvl_um):
    return td.Box(
        size=(0.75 * mesh_wvl_um, 0.75 * mesh_wvl_um, 7 * mesh_wvl_um),
        center=(0, 0, 0),
    )


def _make_medium(kind: str, eps_vals: tuple[float, float, float]):
    if kind == "isotropic":
        return td.Medium(permittivity=eps_vals[0])
    if kind == "anisotropic":
        return td.AnisotropicMedium(
            xx=td.Medium(permittivity=eps_vals[0]),
            yy=td.Medium(permittivity=eps_vals[1]),
            zz=td.Medium(permittivity=eps_vals[2]),
        )
    raise ValueError(f"Unsupported medium kind: {kind!r}")


def _encasing_box_geometry(mesh_wvl_um, adj_wvl_um):
    x_half = 1.1 * mesh_wvl_um + 0.2 * adj_wvl_um
    y_half = 0.8 * mesh_wvl_um + 0.2 * adj_wvl_um
    z_half = 0.5 * (POLYSLAB_HEIGHT_WVL * adj_wvl_um) + 0.2 * adj_wvl_um
    return td.Box(
        center=(0.0, 0.0, 0.0),
        size=(2 * x_half, 2 * y_half, 2 * z_half),
    )


def _rotated_rectangle_vertices(
    width: float, height: float, angle_rad: float
) -> list[tuple[float, float]]:
    half_w = 0.5 * width
    half_h = 0.5 * height
    base = anp.array(
        [
            [-half_w, -half_h],
            [half_w, -half_h],
            [half_w, half_h],
            [-half_w, half_h],
        ]
    )
    rot = anp.array(
        [
            [anp.cos(angle_rad), -anp.sin(angle_rad)],
            [anp.sin(angle_rad), anp.cos(angle_rad)],
        ]
    )
    verts = base @ rot.T
    return [tuple(v) for v in verts]


def make_base_sim(
    mesh_wvl_um,
    adj_wvl_um,
    box_for_override,
    monitor_bg_index=1.0,
    run_time=1e-10,
):
    sim_geometry = get_sim_geometry(mesh_wvl_um)
    sim_size_um = sim_geometry.size
    sim_center_um = sim_geometry.center

    dl_design = mesh_wvl_um / MESH_FACTOR_DESIGN

    mesh_overrides = [
        td.MeshOverrideStructure(
            geometry=box_for_override,
            dl=[dl_design, dl_design, dl_design],
        ),
    ]

    wl_min_src_um = 0.9 * adj_wvl_um
    wl_max_src_um = 1.1 * adj_wvl_um
    fwidth_src = td.C_0 * ((1.0 / wl_min_src_um) - (1.0 / wl_max_src_um))
    freq0 = td.C_0 / adj_wvl_um

    src = td.PlaneWave(
        center=(0.0, 0, -0.25 * sim_size_um[2]),
        size=[td.inf, td.inf, 0],
        source_time=td.GaussianPulse(freq0=freq0, fwidth=fwidth_src),
        direction="+",
        angle_theta=0.0,
        # pol_angle=np.pi / 2,
        # pol_angle=np.pi / 4,
        pol_angle=ROTATION_ANGLE_RAD,
    )

    bloch_x = td.Boundary.bloch_from_source(source=src, domain_size=sim_size_um[0], axis=0)
    bloch_y = td.Boundary.bloch_from_source(source=src, domain_size=sim_size_um[1], axis=1)

    boundary_spec = td.BoundarySpec(
        x=bloch_x,
        y=bloch_y,
        z=td.Boundary.pml(num_layers=48),
    )

    diffraction_monitor = td.DiffractionMonitor(
        center=(0, sim_center_um[1], 0.25 * sim_size_um[2]),
        size=(np.inf, np.inf, 0),
        name="monitor_diffraction",
        freqs=[freq0],
        normal_dir="+",
    )

    monitor_index_block = td.Box(
        center=(sim_center_um[0], sim_center_um[1], 0.25 * sim_size_um[2] + mesh_wvl_um),
        size=(*tuple(2 * size for size in sim_size_um[0:2]), mesh_wvl_um + 0.5 * sim_size_um[2]),
    )
    monitor_index_block_structure = td.Structure(
        geometry=monitor_index_block,
        medium=td.Medium(permittivity=monitor_bg_index**2),
    )

    sim_base = td.Simulation(
        center=sim_center_um,
        size=sim_size_um,
        grid_spec=td.GridSpec.auto(
            min_steps_per_wvl=30,
            wavelength=mesh_wvl_um,
            override_structures=mesh_overrides,
        ),
        # structures=[monitor_index_block_structure],
        structures=[],
        sources=[src],
        monitors=[diffraction_monitor],
        run_time=run_time,
        boundary_spec=boundary_spec,
        subpixel=True,
    )

    return sim_base


def create_objective_function(
    create_sim_base,
    eval_fn,
    sim_path_dir,
    polyslab_height_um,
    polyslab_medium_kind,
    polyslab_eps_vals,
    encasing_medium_kind,
    encasing_eps_vals,
    mesh_wvl_um,
    adj_wvl_um,
):
    encasing_box = _encasing_box_geometry(mesh_wvl_um, adj_wvl_um)
    encasing_medium = _make_medium(encasing_medium_kind, encasing_eps_vals)
    encasing_structure = td.Structure(geometry=encasing_box, medium=encasing_medium)
    polyslab_medium = _make_medium(polyslab_medium_kind, polyslab_eps_vals)

    def objective(params):
        sim_base = create_sim_base()

        simulation_dict = {}
        for idx in range(len(params)):
            width = params[idx][0]
            height = params[idx][1]
            vertices = _rotated_rectangle_vertices(width, height, ROTATION_ANGLE_RAD)

            polyslab = td.PolySlab(
                slab_bounds=(-0.5 * polyslab_height_um, 0.5 * polyslab_height_um),
                axis=2,
                vertices=vertices,
            )

            polyslab_structure = td.Structure(geometry=polyslab, medium=polyslab_medium)

            sim_with_polyslab = sim_base.updated_copy(
                structures=(*sim_base.structures, encasing_structure, polyslab_structure)
            )

            simulation_dict[f"numerical_aniso_shape_testing_{idx}"] = sim_with_polyslab.copy()

        sim_data = web.run_async(
            simulation_dict, path_dir=sim_path_dir, local_gradient=LOCAL_GRADIENT, verbose=VERBOSE
        )

        objective_vals = []
        for idx in range(len(params)):
            objective_vals.append(eval_fn(sim_data[f"numerical_aniso_shape_testing_{idx}"]))

        if len(params) == 1:
            return objective_vals[0]
        return objective_vals

    return objective


MESH_ADJ_WVL = 1.5
POLYSLAB_HEIGHT_WVL = MESH_ADJ_WVL / 2.0
SUBSTRATE_INDEX = 1.0
WG_INDEX = 3.5

NUM_VERTICES = 4

background_indices = [1.0]
mesh_wvls_um = [MESH_ADJ_WVL]
adj_wvls_um = [MESH_ADJ_WVL]
orders_x = (0,)
orders_y = (0,)
pw_angles_deg = [0.0]
grating_modes = ["transmission"]

periodic_test_parameters = [
    {
        "case_name": "polyslab_aniso_encasing_iso",
        "mesh_wvl_um": MESH_ADJ_WVL,
        "adj_wvl_um": MESH_ADJ_WVL,
        "monitor_bg_index": 1.0,
        "pw_angle_deg": 0.0,
        "order_x": (0,),
        "order_y": (0,),
        "grating_mode": "transmission",
        "polyslab_medium_kind": "anisotropic",
        "polyslab_eps_vals": (WG_INDEX**2, WG_INDEX**2 - 4.0, WG_INDEX**2 - 2.0),
        "encasing_medium_kind": "isotropic",
        "encasing_eps_vals": (1.8, 1.8, 1.8),
        "test_number": 0,
    },
    {
        "case_name": "polyslab_iso_encasing_aniso",
        "mesh_wvl_um": MESH_ADJ_WVL,
        "adj_wvl_um": MESH_ADJ_WVL,
        "monitor_bg_index": 1.0,
        "pw_angle_deg": 0.0,
        "order_x": (0,),
        "order_y": (0,),
        "grating_mode": "transmission",
        "polyslab_medium_kind": "isotropic",
        "polyslab_eps_vals": (WG_INDEX**2, WG_INDEX**2, WG_INDEX**2),
        "encasing_medium_kind": "anisotropic",
        "encasing_eps_vals": (1.6, 2.3, 1.65),
        "test_number": 1,
    },
    {
        "case_name": "polyslab_aniso_encasing_aniso",
        "mesh_wvl_um": MESH_ADJ_WVL,
        "adj_wvl_um": MESH_ADJ_WVL,
        "monitor_bg_index": 1.0,
        "pw_angle_deg": 0.0,
        "order_x": (0,),
        "order_y": (0,),
        "grating_mode": "transmission",
        "polyslab_medium_kind": "anisotropic",
        "polyslab_eps_vals": (WG_INDEX**2 - 4.0, WG_INDEX**2, WG_INDEX**2 - 0.2),
        "encasing_medium_kind": "anisotropic",
        "encasing_eps_vals": (1.6, 2.3, 1.65),
        "test_number": 2,
    },
]


@pytest.mark.numerical
@pytest.mark.parametrize(
    "periodic_test_parameters", periodic_test_parameters, ids=lambda p: p["case_name"]
)
def test_finite_difference_anisotropic_shape(
    periodic_test_parameters, rng, numerical_case_dir, redirect_stdout_to_stderr
):
    """Compare FD vs adjoint diffraction-based shape gradients for anisotropic PolySlab cases."""

    test_results = np.zeros((2, 2))

    (
        case_name,
        mesh_wvl_um,
        adj_wvl_um,
        monitor_bg_index,
        pw_angle_deg,
        order_x,
        order_y,
        grating_mode,
        polyslab_medium_kind,
        polyslab_eps_vals,
        encasing_medium_kind,
        encasing_eps_vals,
        test_number,
    ) = operator.itemgetter(
        "case_name",
        "mesh_wvl_um",
        "adj_wvl_um",
        "monitor_bg_index",
        "pw_angle_deg",
        "order_x",
        "order_y",
        "grating_mode",
        "polyslab_medium_kind",
        "polyslab_eps_vals",
        "encasing_medium_kind",
        "encasing_eps_vals",
        "test_number",
    )(periodic_test_parameters)

    box_for_override = td.Box(
        center=(0, 0, 0), size=(np.inf, np.inf, POLYSLAB_HEIGHT_WVL * adj_wvl_um + mesh_wvl_um)
    )

    sim_path_dir = numerical_case_dir / "simulations" / case_name
    sim_path_dir.mkdir(parents=True, exist_ok=True)

    def eval_fn(sim_data):
        amps = sim_data["monitor_diffraction"].amps.sel(orders_x=0, orders_y=0)
        amp_s = amps.sel(polarization="s").data
        amp_p = amps.sel(polarization="p").data
        total = 0.0
        for order_x_val in orders_x:
            for order_y_val in orders_y:
                amp_p = (
                    sim_data["monitor_diffraction"]
                    .amps.sel(polarization="p", orders_x=order_x_val, orders_y=order_y_val)
                    .data
                )
                amp_s = (
                    sim_data["monitor_diffraction"]
                    .amps.sel(polarization="s", orders_x=order_x_val, orders_y=order_y_val)
                    .data
                )

                total += (
                    np.sum(
                        np.abs(
                            amp_p * np.cos(ROTATION_ANGLE_RAD + 0.5 * np.pi)
                            + amp_s * np.sin(ROTATION_ANGLE_RAD + 0.5 * np.pi)
                        )
                    )
                    ** 2
                )

        return total

    polyslab_height_um = POLYSLAB_HEIGHT_WVL * adj_wvl_um

    objective = create_objective_function(
        lambda mesh_wvl_um=mesh_wvl_um,
        adj_wvl_um=adj_wvl_um,
        box_for_override=box_for_override,
        monitor_bg_index=monitor_bg_index: make_base_sim(
            mesh_wvl_um=mesh_wvl_um,
            adj_wvl_um=adj_wvl_um,
            box_for_override=box_for_override,
            monitor_bg_index=monitor_bg_index,
        ),
        eval_fn,
        sim_path_dir=str(sim_path_dir),
        polyslab_height_um=polyslab_height_um,
        polyslab_medium_kind=polyslab_medium_kind,
        polyslab_eps_vals=polyslab_eps_vals,
        encasing_medium_kind=encasing_medium_kind,
        encasing_eps_vals=encasing_eps_vals,
        mesh_wvl_um=mesh_wvl_um,
        adj_wvl_um=adj_wvl_um,
    )

    obj_val_and_grad = ag.value_and_grad(objective)
    fd_step = 0.25 * 0.025 * adj_wvl_um

    width0 = 0.4 * 0.75 * mesh_wvl_um
    height0 = 0.35 * 0.75 * mesh_wvl_um
    params0 = [width0, height0]

    obj, adj_grad = obj_val_and_grad([params0])

    all_params = [
        [width0 + fd_step, height0],
        [width0 - fd_step, height0],
        [width0, height0 + fd_step],
        [width0, height0 - fd_step],
    ]
    all_obj = objective(all_params)

    fd_grad = np.zeros(2)
    fd_grad[0] = (all_obj[0] - all_obj[1]) / (2 * fd_step)
    fd_grad[1] = (all_obj[2] - all_obj[3]) / (2 * fd_step)
    rms_error = np.linalg.norm(fd_grad - adj_grad)
    fd_mag = np.linalg.norm(fd_grad)
    adj_mag = np.linalg.norm(adj_grad)
    percentage_error = 100.0 * np.mean(
        np.abs(fd_grad - adj_grad) / (np.abs(fd_grad) + np.finfo(np.float64).eps)
    )

    print("\n" * 3)
    print("-" * 20)
    print(f"Numerical test #{test_number}: {case_name}")
    print(f"Mesh and adjoint wavelengths: {mesh_wvl_um}, {adj_wvl_um}")
    print(f"Orders: x={order_x}, y={order_y}, output polarization='p'")
    print(f"PolySlab medium: {polyslab_medium_kind} {polyslab_eps_vals}")
    print(f"Encasing medium: {encasing_medium_kind} {encasing_eps_vals}")
    print(f"RMS Error: {rms_error}")
    print(f"FD, Adj magnitudes: {fd_mag}, {adj_mag}")
    print(f"Percentage Error: {percentage_error}")
    print(
        f"[{case_name}] width_fd={fd_grad[0]:.6e} height_fd={fd_grad[1]:.6e}",
        file=sys.stderr,
    )
    print(
        f"[{case_name}] width_adj={adj_grad[0][0]:.6e} height_adj={adj_grad[0][1]:.6e}",
        file=sys.stderr,
    )
    print("-" * 20)
    print("\n" * 3)

    test_results[SAVE_FD_LOC, :] = fd_grad
    test_results[SAVE_ADJ_LOC, :] = adj_grad[0]

    save_idx = test_number + 1
    save_path = None
    if SAVE_FD_ADJ_DATA:
        results_dir = numerical_case_dir / NUMERICAL_RESULTS_SUBDIR
        results_dir.mkdir(parents=True, exist_ok=True)
        save_path = results_dir / f"results_{save_idx}.npy"

    try:
        assert rms_error < RMS_THRESHOLD * fd_mag, "RMS error magnitude too large"
    finally:
        if save_path is not None:
            np.save(save_path, test_results)

    if PLOT_FD_ADJ_COMPARISON:
        plt.plot(adj_grad, color="g", linewidth=2.0)
        plt.plot(fd_grad, color="b", linewidth=1.5, linestyle="--")
        plt.title(f"Width/Height Gradient: {case_name}")
        plt.legend(["Adjoint", "Finite difference"])
        plt.xlabel("Parameter index")
        plt.ylabel("Gradient value")
        plt.legend()
        plt.show()
