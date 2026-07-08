"""Numerical diffraction-based shape-gradient checks for anisotropic PolySlab cases."""

from __future__ import annotations

import sys
from pathlib import Path

import autograd as ag
import autograd.numpy as anp
import matplotlib.pylab as plt
import numpy as np
import pytest
from pydantic import BaseModel

import tidy3d as td
import tidy3d.web as web

from .numerical_test_helpers import (
    EvaluationData,
    GradientComparisonDiagnostics,
    MetricGroups,
    case_identity_from_parameters,
    case_identity_id,
    evaluate_fd_adjoint_gradient_agreement,
    finalize_result,
    load_or_collect_evaluation_data,
)

PLOT_FD_ADJ_COMPARISON = False
LOCAL_GRADIENT = True
VERBOSE = False

RMS_THRESHOLD = 0.25


class AnisotropicShapeCaseIdentity(BaseModel):
    """Semantic identity for one anisotropic PolySlab diffraction-gradient case."""

    case_name: str
    mesh_wvl_um: float
    adj_wvl_um: float
    monitor_bg_index: float
    pw_angle_deg: float
    order_x: tuple[int, ...]
    order_y: tuple[int, ...]
    grating_mode: str
    polyslab_medium_kind: str
    polyslab_eps_vals: tuple[float, float, float]
    encasing_medium_kind: str
    encasing_eps_vals: tuple[float, float, float]
    rms_threshold: float


class AnisotropicShapeTestParameters(AnisotropicShapeCaseIdentity):
    """Full parameter bundle for one anisotropic shape test invocation."""

    test_number: int


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
            simulation_dict,
            path_dir=sim_path_dir,
            local_gradient=LOCAL_GRADIENT,
            verbose=VERBOSE,
            lazy=False,
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

ANISOTROPIC_SHAPE_PARAMETERS = [
    AnisotropicShapeTestParameters(
        case_name="polyslab_aniso_encasing_iso",
        mesh_wvl_um=MESH_ADJ_WVL,
        adj_wvl_um=MESH_ADJ_WVL,
        monitor_bg_index=1.0,
        pw_angle_deg=0.0,
        order_x=(0,),
        order_y=(0,),
        grating_mode="transmission",
        polyslab_medium_kind="anisotropic",
        polyslab_eps_vals=(WG_INDEX**2, WG_INDEX**2 - 4.0, WG_INDEX**2 - 2.0),
        encasing_medium_kind="isotropic",
        encasing_eps_vals=(1.8, 1.8, 1.8),
        rms_threshold=RMS_THRESHOLD,
        test_number=0,
    ),
    AnisotropicShapeTestParameters(
        case_name="polyslab_iso_encasing_aniso",
        mesh_wvl_um=MESH_ADJ_WVL,
        adj_wvl_um=MESH_ADJ_WVL,
        monitor_bg_index=1.0,
        pw_angle_deg=0.0,
        order_x=(0,),
        order_y=(0,),
        grating_mode="transmission",
        polyslab_medium_kind="isotropic",
        polyslab_eps_vals=(WG_INDEX**2, WG_INDEX**2, WG_INDEX**2),
        encasing_medium_kind="anisotropic",
        encasing_eps_vals=(1.6, 2.3, 1.65),
        rms_threshold=RMS_THRESHOLD,
        test_number=1,
    ),
    AnisotropicShapeTestParameters(
        case_name="polyslab_aniso_encasing_aniso",
        mesh_wvl_um=MESH_ADJ_WVL,
        adj_wvl_um=MESH_ADJ_WVL,
        monitor_bg_index=1.0,
        pw_angle_deg=0.0,
        order_x=(0,),
        order_y=(0,),
        grating_mode="transmission",
        polyslab_medium_kind="anisotropic",
        polyslab_eps_vals=(WG_INDEX**2 - 4.0, WG_INDEX**2, WG_INDEX**2 - 0.2),
        encasing_medium_kind="anisotropic",
        encasing_eps_vals=(1.6, 2.3, 1.65),
        rms_threshold=RMS_THRESHOLD,
        test_number=2,
    ),
]


def _case_identity(
    anisotropic_shape_parameters: AnisotropicShapeTestParameters,
) -> AnisotropicShapeCaseIdentity:
    return case_identity_from_parameters(AnisotropicShapeCaseIdentity, anisotropic_shape_parameters)


def _collect_anisotropic_shape_evaluation_data(
    anisotropic_shape_parameters: AnisotropicShapeTestParameters,
    numerical_case_dir: Path,
) -> EvaluationData:
    case_name = anisotropic_shape_parameters.case_name
    mesh_wvl_um = anisotropic_shape_parameters.mesh_wvl_um
    adj_wvl_um = anisotropic_shape_parameters.adj_wvl_um
    monitor_bg_index = anisotropic_shape_parameters.monitor_bg_index
    order_x = anisotropic_shape_parameters.order_x
    order_y = anisotropic_shape_parameters.order_y
    polyslab_medium_kind = anisotropic_shape_parameters.polyslab_medium_kind
    polyslab_eps_vals = anisotropic_shape_parameters.polyslab_eps_vals
    encasing_medium_kind = anisotropic_shape_parameters.encasing_medium_kind
    encasing_eps_vals = anisotropic_shape_parameters.encasing_eps_vals

    box_for_override = td.Box(
        center=(0, 0, 0), size=(np.inf, np.inf, POLYSLAB_HEIGHT_WVL * adj_wvl_um + mesh_wvl_um)
    )

    sim_path_dir = numerical_case_dir / "simulations" / case_name
    sim_path_dir.mkdir(parents=True, exist_ok=True)

    def eval_fn(sim_data):
        total = 0.0
        for order_x_val in order_x:
            for order_y_val in order_y:
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

    _obj, adj_grad = obj_val_and_grad([params0])

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

    return {
        "fd_grad": np.asarray(fd_grad, dtype=float),
        "adj_grad_projected": np.asarray(adj_grad[0], dtype=float),
        "fd_step": np.asarray(fd_step, dtype=float),
        "params0": np.asarray(params0, dtype=float),
    }


def _evaluate_anisotropic_shape_evaluation_data(evaluation_data: EvaluationData) -> MetricGroups:
    return evaluate_fd_adjoint_gradient_agreement(
        fd_grad=np.asarray(evaluation_data["fd_grad"], dtype=float),
        adj_grad_projected=np.asarray(evaluation_data["adj_grad_projected"], dtype=float),
        relative_rms_threshold=RMS_THRESHOLD,
    )


def _print_anisotropic_shape_summary(
    anisotropic_shape_parameters: AnisotropicShapeTestParameters,
    evaluation_data: EvaluationData,
    diagnostics: GradientComparisonDiagnostics,
    *,
    eval_only: bool,
) -> None:
    mode_label = "saved-artifact re-evaluation" if eval_only else "fresh data collection"
    fd_grad = np.asarray(evaluation_data["fd_grad"], dtype=float)
    adj_grad = np.asarray(evaluation_data["adj_grad_projected"], dtype=float)

    print("\n" * 3)
    print("-" * 20)
    print(
        f"Numerical test #{anisotropic_shape_parameters.test_number}: {anisotropic_shape_parameters.case_name}"
    )
    print(f"Evaluation mode: {mode_label}")
    print(
        "Mesh and adjoint wavelengths: "
        f"{anisotropic_shape_parameters.mesh_wvl_um}, {anisotropic_shape_parameters.adj_wvl_um}"
    )
    print(
        f"Orders: x={anisotropic_shape_parameters.order_x}, "
        f"y={anisotropic_shape_parameters.order_y}, output polarization='p'"
    )
    print(
        "PolySlab medium: "
        f"{anisotropic_shape_parameters.polyslab_medium_kind} "
        f"{anisotropic_shape_parameters.polyslab_eps_vals}"
    )
    print(
        "Encasing medium: "
        f"{anisotropic_shape_parameters.encasing_medium_kind} "
        f"{anisotropic_shape_parameters.encasing_eps_vals}"
    )
    print(f"RMS Error: {diagnostics['rms_error']}")
    print(f"FD, Adj magnitudes: {diagnostics['fd_mag']}, {diagnostics['adj_mag']}")
    print(f"Percentage Error: {diagnostics['percentage_error']}")
    print(
        f"[{anisotropic_shape_parameters.case_name}] width_fd={fd_grad[0]:.6e} "
        f"height_fd={fd_grad[1]:.6e}",
        file=sys.stderr,
    )
    print(
        f"[{anisotropic_shape_parameters.case_name}] width_adj={adj_grad[0]:.6e} "
        f"height_adj={adj_grad[1]:.6e}",
        file=sys.stderr,
    )
    print("-" * 20)
    print("\n" * 3)


def _plot_anisotropic_shape_comparison(
    anisotropic_shape_parameters: AnisotropicShapeTestParameters, evaluation_data: EvaluationData
) -> None:
    plt.plot(evaluation_data["adj_grad_projected"], color="g", linewidth=2.0)
    plt.plot(evaluation_data["fd_grad"], color="b", linewidth=1.5, linestyle="--")
    plt.title(f"Width/Height Gradient: {anisotropic_shape_parameters.case_name}")
    plt.legend(["Adjoint", "Finite difference"])
    plt.xlabel("Parameter index")
    plt.ylabel("Gradient value")
    plt.legend()
    plt.show()


@pytest.mark.numerical
@pytest.mark.parametrize(
    "anisotropic_shape_parameters",
    ANISOTROPIC_SHAPE_PARAMETERS,
    ids=lambda params: case_identity_id(_case_identity(params), prefix="aniso-shape"),
)
def test_finite_difference_anisotropic_shape(
    request: pytest.FixtureRequest,
    anisotropic_shape_parameters: AnisotropicShapeTestParameters,
    numerical_case_dir: Path,
    numerical_eval_only: bool,
    redirect_stdout_to_stderr: None,
) -> None:
    """Compare FD vs adjoint diffraction-based shape gradients for anisotropic PolySlab cases."""
    case_identity = _case_identity(anisotropic_shape_parameters)
    evaluation_data = load_or_collect_evaluation_data(
        numerical_case_dir=numerical_case_dir,
        numerical_eval_only=numerical_eval_only,
        case_identity=case_identity,
        collect_evaluation_data=lambda: _collect_anisotropic_shape_evaluation_data(
            anisotropic_shape_parameters, numerical_case_dir
        ),
    )
    regression_metrics, observation_metrics, diagnostics = (
        _evaluate_anisotropic_shape_evaluation_data(evaluation_data)
    )
    _print_anisotropic_shape_summary(
        anisotropic_shape_parameters,
        evaluation_data,
        diagnostics,
        eval_only=numerical_eval_only,
    )

    if PLOT_FD_ADJ_COMPARISON:
        _plot_anisotropic_shape_comparison(anisotropic_shape_parameters, evaluation_data)

    finalize_result(
        pytest_nodeid=request.node.nodeid,
        numerical_case_dir=numerical_case_dir,
        regression_metrics=regression_metrics,
        observation_metrics=observation_metrics,
        failure_message=(
            "Anisotropic shape RMS error magnitude too large; inspect "
            f"{numerical_case_dir / 'evaluation_data.npz'} and {numerical_case_dir / 'result.json'}"
        ),
    )
