from __future__ import annotations

import autograd.numpy as anp
import numpy as np
import pytest
from autograd import value_and_grad
from autograd.tracer import getval

import tidy3d as td
import tidy3d.web as web

THETAS_DEG = [-10, -5, 0, 5, 10]
AXES = [0, 1, 2]
REF_PLANES = ["bottom", "middle", "top"]
H = 2e-2  # finite-difference step in degrees
RTOL = 5e-2
ATOL = 1e-3
MSPW = 20  # min steps per wavelength for auto grid
UNIFORM_DX = None  # set to float to force uniform grid
DOMAIN = (4.0, 4.0, 4.0)
LOCAL_GRADIENT = True


def _build_sim(theta_deg: float, axis: int, reference_plane: str) -> td.Simulation:
    """Construct simulation matching the runner's setup (Gaussian beam + field monitor)."""
    wvl = 1.0
    freq0 = td.C_0 / wvl

    half_x = DOMAIN[0] / 2.0
    src_x = -(half_x - 0.1)
    mnt_x = half_x - 0.1

    src = td.GaussianBeam(
        center=(src_x, 0, 0),
        size=(0, td.inf, td.inf),
        direction="+",
        waist_radius=0.5,
        source_time=td.GaussianPulse(freq0=freq0, fwidth=freq0 / 10),
    )
    mnt = td.FieldMonitor(
        size=(0, 3, 3),
        center=(mnt_x, 0, 0),
        freqs=[freq0],
        name="field",
    )

    theta = anp.deg2rad(theta_deg)
    verts = anp.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])
    geom = td.PolySlab(
        vertices=verts,
        slab_bounds=(-1, 1),
        axis=axis,
        sidewall_angle=theta,
        dilation=0.0,
        reference_plane=reference_plane,
    )
    struct = td.Structure(geometry=geom, medium=td.Medium(permittivity=2))

    grid_spec = (
        td.GridSpec.uniform(dl=UNIFORM_DX)
        if UNIFORM_DX is not None
        else td.GridSpec.auto(min_steps_per_wvl=MSPW)
    )

    return td.Simulation(
        size=DOMAIN,
        grid_spec=grid_spec,
        boundary_spec=td.BoundarySpec.pml(x=True, y=True, z=True),
        sources=[src],
        structures=[struct],
        monitors=[mnt],
        run_time=1e-12,
    )


def _objective(theta_deg: float, axis: int, reference_plane: str, case_dir, verbose: bool) -> float:
    sim = _build_sim(theta_deg, axis=axis, reference_plane=reference_plane)
    task_name = f"obj_axis{axis}_ref{reference_plane}_t{float(getval(theta_deg)):+0.3f}"
    out_path = case_dir / f"{task_name}.hdf5"
    data = web.run(
        sim,
        task_name=task_name,
        local_gradient=LOCAL_GRADIENT,
        verbose=verbose,
        path=str(out_path),
    )
    return data["field"].flux.item()


@pytest.mark.numerical
@pytest.mark.parametrize("axis", AXES)
@pytest.mark.parametrize("reference_plane", REF_PLANES)
@pytest.mark.parametrize("theta0_deg", THETAS_DEG)
def test_autograd_polyslab_sidewall_vs_fd(
    axis, reference_plane, theta0_deg, numerical_case_dir, redirect_stdout_to_stderr
):
    """Adjoint dJ/dtheta matches centered FD for PolySlab.sidewall_angle across axes/ref planes."""

    verbose = False

    # objective and adjoint grad
    objective_dir = numerical_case_dir / "objective"
    objective_dir.mkdir(parents=True, exist_ok=True)

    obj_fun = lambda tdeg: _objective(tdeg, axis, reference_plane, objective_dir, verbose)
    obj, grad_adj = value_and_grad(obj_fun)(anp.array(theta0_deg))

    # centered finite difference
    uid = f"axis{axis}_ref{reference_plane}_t{float(theta0_deg):+0.3f}"
    sims = {
        f"plus_{uid}": _build_sim(theta0_deg + H, axis=axis, reference_plane=reference_plane),
        f"minus_{uid}": _build_sim(theta0_deg - H, axis=axis, reference_plane=reference_plane),
    }
    fd_dir = numerical_case_dir / "finite_difference"
    fd_dir.mkdir(parents=True, exist_ok=True)

    datas = web.run_async(
        sims,
        path_dir=str(fd_dir),
        local_gradient=LOCAL_GRADIENT,
        verbose=verbose,
    )

    obj_plus = float(datas[f"plus_{uid}"]["field"].flux.item())
    obj_minus = float(datas[f"minus_{uid}"]["field"].flux.item())
    grad_fd = (obj_plus - obj_minus) / (2 * H)

    print("\n" * 3)
    print("-" * 20)
    print(f"axis: {axis}")
    print(f"reference_plane: {reference_plane}")
    print(f"theta0_deg: {theta0_deg}")
    print(f"Grad (adjoint): {grad_adj}")
    print(f"Grad (finite difference): {grad_fd}")
    print("-" * 20)
    print("\n" * 3)

    assert np.isfinite(grad_adj)
    assert np.isclose(grad_adj, grad_fd, rtol=RTOL, atol=ATOL)
