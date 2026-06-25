"""Numerical gradient check against TMM for a multilayer slab."""

from __future__ import annotations

import autograd as ag
import autograd.numpy as anp
import matplotlib.pylab as plt
import numpy as np
import pytest
from autograd.tracer import getval
from pydantic import BaseModel

import tidy3d as td
import tidy3d.web as web

from .numerical_test_helpers import (
    EvaluationData,
    case_identity_id,
    finalize_result,
    load_or_collect_evaluation_data,
)
from .result_models import Metric

tmm = pytest.importorskip("tmm")

PLOT_TMM_ADJ_COMPARISON = False
LOCAL_GRADIENT = True

RMS_RELATIVE_THRESHOLD_EPS = 0.02
RMS_RELATIVE_THRESHOLD_DS = 0.002

# multilayer slab parameters
ADJ_FREQ0 = 2.0e14
WAVELENGTH_UM = td.C_0 / ADJ_FREQ0
BCK_EPS = 1.0**2
SPC = 0.0
SLAB_EPS0 = np.array([2.0**2, 1.8**2, 1.5**2, 1.9**2])
SLAB_DS0 = np.array([0.5, 0.25, 0.5, 0.5])
THETA = 0.0

DL_UM = 0.01
DOMAIN_XY_UM = WAVELENGTH_UM
RUN_TIME = 1e-11

BASE_TOTAL_THICKNESS = np.sum(SLAB_DS0) + (SLAB_DS0.size - 1) * SPC

SIM_Z_SIZE = BASE_TOTAL_THICKNESS + 4 * WAVELENGTH_UM
SIM_Z_HALF = 0.5 * SIM_Z_SIZE
SRC_Z = -SIM_Z_HALF + WAVELENGTH_UM
MONITOR_Z = SIM_Z_HALF - WAVELENGTH_UM

if PLOT_TMM_ADJ_COMPARISON:
    pytestmark = pytest.mark.usefixtures("mpl_config_interactive")
else:
    pytestmark = pytest.mark.usefixtures("mpl_config_noninteractive")


def compute_T_tmm(slab_eps: np.ndarray, slab_ds: np.ndarray) -> float:
    """Get transmission as a function of slab permittivities and thicknesses."""
    eps_list = [BCK_EPS]
    d_list = [np.inf]

    for eps, d in zip(slab_eps, slab_ds):
        eps_list.append(eps)
        d_list.append(d)
        if SPC > 0:
            eps_list.append(BCK_EPS)
            d_list.append(SPC)

    eps_list.append(BCK_EPS)
    d_list.append(np.inf)

    n_list = np.sqrt(eps_list)
    return tmm.coh_tmm("p", n_list, d_list, THETA, WAVELENGTH_UM)["T"]


def compute_T_tmm_grad(slab_eps: np.ndarray, slab_ds: np.ndarray) -> np.ndarray:
    """Compute numerical gradient of transmission w.r.t. each of the slab permittivities and thicknesses using TMM."""

    delta = 1e-4

    # set up containers to store gradient and perturbed arguments
    num_slabs = len(slab_eps)
    grad_tmm = np.zeros((2, num_slabs), dtype=float)
    args = np.stack((slab_eps, slab_ds), axis=0)

    # loop through slab index and argument index (eps, d)
    for arg_index in range(2):
        for slab_index in range(num_slabs):
            grad = 0.0

            # perturb the argument by delta in each + and - direction
            for pm in (-1, +1):
                args_num = args.copy()
                args_num[arg_index][slab_index] += delta * pm

                # NEW: for slab thickness gradient, need to modify neighboring slabs too
                if arg_index == 1 and SPC == 0:
                    if slab_index > 0:
                        args_num[arg_index][slab_index - 1] -= delta * pm / 2
                    if slab_index < num_slabs - 1:
                        args_num[arg_index][slab_index + 1] -= delta * pm / 2

                # compute argument perturbed T and add to finite difference gradient contribution
                T_tmm = compute_T_tmm(slab_eps=args_num[0], slab_ds=args_num[1])
                grad += pm * T_tmm / 2 / delta

            grad_tmm[arg_index][slab_index] = grad
    grad_eps, grad_ds = grad_tmm

    return np.concatenate([grad_eps, grad_ds])


def make_simulation(slab_eps: np.ndarray, slab_ds: np.ndarray, geom_type: str) -> td.Simulation:
    """Create a Simulation given the slab permittivities and thicknesses."""

    # frequency setup
    fwidth = ADJ_FREQ0 / 10.0
    freqs = [ADJ_FREQ0]

    # geometry setup
    bck_medium = td.Medium(permittivity=BCK_EPS)

    space_above = 2
    space_below = 2

    length_x = 1.0
    length_y = 1.0
    length_z = space_below + sum(SLAB_DS0) + space_above + (len(SLAB_DS0) - 1) * SPC
    sim_size = (length_x, length_y, length_z)

    # make structures
    slabs = []
    z_start = -length_z / 2 + space_below
    for d, eps in zip(slab_ds, slab_eps):
        # don't track the gradient through the center of each slab
        # as tidy3d doesn't have enough information to properly process the interface between touching Box objects
        z_center = z_start + d / 2
        z_center = getval(z_center)

        vertex_x = (-2 * length_x, -2 * length_x, 2 * length_x, 2 * length_x)
        vertex_y = (-2 * length_y, 2 * length_y, 2 * length_y, -2 * length_y)

        if geom_type == "box":
            geometry = td.Box(center=[0, 0, z_center], size=[td.inf, td.inf, d])
        elif geom_type == "polyslab":
            geometry = td.PolySlab(
                vertices=list(zip(vertex_x, vertex_y)),
                slab_bounds=(z_center - 0.5 * d, z_center + 0.5 * d),
            )
        else:
            raise ValueError(f"Unknown geometry type {geom_type}. Expected 'box' or 'polyslab'.")

        slab = td.Structure(geometry=geometry, medium=td.Medium(permittivity=eps))

        slabs.append(slab)
        z_start = z_start + d + SPC

    # source setup
    gaussian = td.GaussianPulse(freq0=ADJ_FREQ0, fwidth=fwidth)
    src_z = -length_z / 2 + space_below / 2.0

    source = td.PlaneWave(
        center=(0, 0, src_z),
        size=(td.inf, td.inf, 0),
        source_time=gaussian,
        direction="+",
        angle_theta=THETA,
        angle_phi=0,
        pol_angle=0,
    )

    # boundaries
    boundary_x = td.Boundary.bloch_from_source(
        source=source, domain_size=sim_size[0], axis=0, medium=bck_medium
    )
    boundary_y = td.Boundary.bloch_from_source(
        source=source, domain_size=sim_size[1], axis=1, medium=bck_medium
    )
    boundary_spec = td.BoundarySpec(x=boundary_x, y=boundary_y, z=td.Boundary.pml(num_layers=40))

    # monitors
    mnt_z = length_z / 2 - space_above / 2.0
    monitor_1 = td.DiffractionMonitor(
        center=[0.0, 0.0, mnt_z],
        size=[td.inf, td.inf, 0],
        freqs=freqs,
        name="diffraction",
        normal_dir="+",
    )

    # make simulation
    return td.Simulation(
        size=sim_size,
        grid_spec=td.GridSpec.uniform(dl=0.01),
        structures=slabs,
        sources=[source],
        monitors=[monitor_1],
        run_time=10 / fwidth,
        boundary_spec=boundary_spec,
        medium=bck_medium,
        subpixel=True,
        shutoff=1e-8,
    )


def transmission_from_sim(sim_data: td.SimulationData) -> anp.ndarray:
    amps = sim_data["diffraction"].amps.sel(orders_x=0, orders_y=0, polarization="p").data
    return anp.sum(anp.abs(amps) ** 2)


class TMMGradientCaseIdentity(BaseModel):
    """Semantic identity for one TMM gradient comparison case."""

    geom_type: str
    adj_freq0: float
    slab_eps: tuple[float, ...]
    slab_ds: tuple[float, ...]
    spc: float
    theta: float


test_parameters = [
    TMMGradientCaseIdentity(
        geom_type=geom_type,
        adj_freq0=ADJ_FREQ0,
        slab_eps=tuple(float(eps) for eps in SLAB_EPS0),
        slab_ds=tuple(float(ds) for ds in SLAB_DS0),
        spc=SPC,
        theta=THETA,
    )
    for geom_type in ("box", "polyslab")
]


def _collect_tmm_gradient_evaluation_data(
    test_parameters: TMMGradientCaseIdentity,
    numerical_case_dir,
) -> EvaluationData:
    geom_type = test_parameters.geom_type
    params0 = np.concatenate([SLAB_EPS0, SLAB_DS0])
    tmm_grad = compute_T_tmm_grad(SLAB_EPS0, SLAB_DS0)
    sim_path_dir = numerical_case_dir / "simulations"
    sim_path_dir.mkdir(parents=True, exist_ok=True)

    def objective(params):
        slab_eps = params[: SLAB_EPS0.size]
        slab_ds = params[SLAB_EPS0.size :]
        sim = make_simulation(slab_eps, slab_ds, geom_type)
        sim_path = sim_path_dir / "tmm_gradients.hdf5"
        sim_data = web.run(sim, path=str(sim_path), local_gradient=LOCAL_GRADIENT)
        return transmission_from_sim(sim_data)

    obj_val, adj_grad = ag.value_and_grad(objective)(params0)
    adj_grad = np.asarray(adj_grad, dtype=float)

    return {
        "tmm_grad": np.asarray(tmm_grad, dtype=float),
        "adj_grad": np.asarray(adj_grad, dtype=float),
        "obj_val": np.asarray(obj_val, dtype=float),
    }


def _evaluate_tmm_gradient_evaluation_data(evaluation_data: EvaluationData):
    tmm_grad = np.asarray(evaluation_data["tmm_grad"], dtype=float)
    adj_grad = np.asarray(evaluation_data["adj_grad"], dtype=float)

    tmm_grad_eps = tmm_grad[0 : len(SLAB_EPS0)]
    tmm_grad_ds = tmm_grad[len(SLAB_EPS0) :]

    adj_grad_eps = adj_grad[0 : len(SLAB_EPS0)]
    adj_grad_ds = adj_grad[len(SLAB_EPS0) :]

    rms_relative_eps = float(
        np.linalg.norm(adj_grad_eps - tmm_grad_eps) / np.linalg.norm(tmm_grad_eps)
    )
    rms_relative_ds = float(np.linalg.norm(adj_grad_ds - tmm_grad_ds) / np.linalg.norm(tmm_grad_ds))

    regression_metrics = [
        Metric(
            name="rms_relative_eps",
            observed=rms_relative_eps,
            expected=RMS_RELATIVE_THRESHOLD_EPS,
            comparator="lt",
        ),
        Metric(
            name="rms_relative_ds",
            observed=rms_relative_ds,
            expected=RMS_RELATIVE_THRESHOLD_DS,
            comparator="lt",
        ),
    ]
    observation_metrics = [
        Metric(
            name="tmm_grad_norm",
            observed=float(np.linalg.norm(tmm_grad)),
            expected=0.0,
            comparator="gte",
        ),
        Metric(
            name="adj_grad_norm",
            observed=float(np.linalg.norm(adj_grad)),
            expected=0.0,
            comparator="gte",
        ),
    ]
    diagnostics = {
        "obj_val": float(np.asarray(evaluation_data["obj_val"])),
        "tmm_grad_eps": tmm_grad_eps,
        "adj_grad_eps": adj_grad_eps,
        "tmm_grad_ds": tmm_grad_ds,
        "adj_grad_ds": adj_grad_ds,
        "rms_relative_eps": rms_relative_eps,
        "rms_relative_ds": rms_relative_ds,
    }
    return regression_metrics, observation_metrics, diagnostics


def _print_tmm_gradient_summary(
    test_parameters: TMMGradientCaseIdentity,
    diagnostics,
    *,
    eval_only: bool,
) -> None:
    mode_label = "saved-artifact re-evaluation" if eval_only else "fresh data collection"
    print("\n" * 2)
    print("-" * 20)
    print("TMM vs FDTD gradients")
    print(f"Evaluation mode: {mode_label}")
    print(f"Geometry type: {test_parameters.geom_type}")
    print(f"Objective value: {diagnostics['obj_val']}")
    print(f"TMM grad (eps): {diagnostics['tmm_grad_eps']}")
    print(f"Adj grad (eps): {diagnostics['adj_grad_eps']}")
    print(f"TMM grad (ds): {diagnostics['tmm_grad_ds']}")
    print(f"Adj grad (ds): {diagnostics['adj_grad_ds']}")
    print(f"RMS relative error (eps): {diagnostics['rms_relative_eps']}")
    print(f"RMS relative error (ds): {diagnostics['rms_relative_ds']}")
    print("-" * 20)
    print("\n" * 2)


def _plot_tmm_gradient_comparison(
    test_parameters: TMMGradientCaseIdentity,
    evaluation_data: EvaluationData,
) -> None:
    if PLOT_TMM_ADJ_COMPARISON:
        tmm_grad = np.asarray(evaluation_data["tmm_grad"], dtype=float)
        adj_grad = np.asarray(evaluation_data["adj_grad"], dtype=float)
        plt.plot(tmm_grad, color="b", linewidth=1.5)
        plt.plot(adj_grad, color="g", linewidth=1.5, linestyle="--")
        plt.title(f"TMM vs FDTD gradients (geom type: {test_parameters.geom_type})")
        plt.legend(["TMM (finite diff)", "FDTD (adjoint)"])
        plt.xlabel("Parameter index (eps first, then thickness)")
        plt.ylabel("Gradient value")
        plt.show()


@pytest.mark.numerical
@pytest.mark.parametrize(
    "test_parameters",
    test_parameters,
    ids=lambda params: case_identity_id(params, prefix="tmm"),
)
def test_tmm_gradient_match(
    request: pytest.FixtureRequest,
    test_parameters: TMMGradientCaseIdentity,
    numerical_case_dir,
    numerical_eval_only: bool,
    redirect_stdout_to_stderr,
):
    case_identity = test_parameters
    evaluation_data = load_or_collect_evaluation_data(
        numerical_case_dir=numerical_case_dir,
        numerical_eval_only=numerical_eval_only,
        case_identity=case_identity,
        collect_evaluation_data=lambda: _collect_tmm_gradient_evaluation_data(
            test_parameters, numerical_case_dir
        ),
    )
    regression_metrics, observation_metrics, diagnostics = _evaluate_tmm_gradient_evaluation_data(
        evaluation_data
    )
    _print_tmm_gradient_summary(test_parameters, diagnostics, eval_only=numerical_eval_only)
    _plot_tmm_gradient_comparison(test_parameters, evaluation_data)

    finalize_result(
        pytest_nodeid=request.node.nodeid,
        numerical_case_dir=numerical_case_dir,
        regression_metrics=regression_metrics,
        observation_metrics=observation_metrics,
        failure_message=(
            "TMM and adjoint gradients diverged; inspect "
            f"{numerical_case_dir / 'evaluation_data.npz'} and {numerical_case_dir / 'result.json'}"
        ),
    )
