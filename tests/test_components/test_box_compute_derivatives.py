import autograd
import autograd.numpy as anp
import numpy as np
import pytest
import tidy3d as td
import tidy3d.web as web

wavelength = 1.5
freq0 = td.C_0 / wavelength
L = 10 * wavelength
buffer = 1.0 * wavelength
run_time = 120 / freq0


SCENARIOS = [
    {
        "name": "(1) normal",
        "has_background": False,
        "background_eps": 3.0,
        "box_eps": 2.0,
        "rotation_deg": None,
        "rotation_axis": None,
    },
    {
        "name": "(2) perm=1.5",
        "has_background": True,
        "background_eps": 1.5,
        "box_eps": 2.0,
        "rotation_deg": None,
        "rotation_axis": None,
    },
    {
        "name": "(3) rotation=0 deg about z",
        "has_background": False,
        "background_eps": 1.5,
        "box_eps": 2.0,
        "rotation_deg": 0.0,
        "rotation_axis": 2,
    },
    {
        "name": "(4) rotation=90 deg about z",
        "has_background": False,
        "background_eps": 1.5,
        "box_eps": 2.0,
        "rotation_deg": 90.0,
        "rotation_axis": 2,
    },
    {
        "name": "(5) rotation=45 deg about y",
        "has_background": False,
        "background_eps": 1.5,
        "box_eps": 2.0,
        "rotation_deg": 45.0,
        "rotation_axis": 1,
    },
    {
        "name": "(6) rotation=45 deg about x",
        "has_background": False,
        "background_eps": 1.5,
        "box_eps": 2.0,
        "rotation_deg": 45.0,
        "rotation_axis": 0,
    },
    {
        "name": "(7) rotation=45 deg about z",
        "has_background": False,
        "background_eps": 1.5,
        "box_eps": 2.0,
        "rotation_deg": 45.0,
        "rotation_axis": 2,
    },
]

PARAM_LABELS = ["center_x", "center_y", "center_z", "size_x", "size_y", "size_z"]


def make_sim(center: tuple, size: tuple, scenario: dict):
    source = td.PointDipole(
        center=(-L / 2 + buffer, 0.0, 0.0),
        source_time=td.GaussianPulse(freq0=freq0, fwidth=freq0 / 10.0),
        polarization="Ez",
    )

    monitor = td.FieldMonitor(
        center=(+L / 2 - buffer, 0.5 * buffer, 0.5 * buffer),
        size=(0, 0, 0),
        freqs=[freq0],
        name="point_out",
    )

    structures = []
    if scenario["has_background"]:
        back_box = td.Box(center=(0.0, 0.0, 0.0), size=(4.0, 1.6, 1.6))
        background_box = td.Structure(
            geometry=back_box,
            medium=td.Medium(permittivity=scenario["background_eps"]),
        )
        structures.append(background_box)

    scatter_box = td.Box(center=center, size=size)

    if scenario["rotation_deg"] is not None:
        angle_rad = np.deg2rad(scenario["rotation_deg"])
        rotated_geom = scatter_box.rotated(angle_rad, scenario["rotation_axis"])
    else:
        rotated_geom = scatter_box

    scatter_struct = td.Structure(
        geometry=rotated_geom,
        medium=td.Medium(permittivity=scenario["box_eps"]),
    )
    structures.append(scatter_struct)

    sim = td.Simulation(
        size=(L, L, L),
        run_time=run_time,
        grid_spec=td.GridSpec.auto(min_steps_per_wvl=50),
        sources=[source],
        monitors=[monitor],
        structures=structures,
    )
    return sim


def objective_fn(center, size, scenario):
    sim = make_sim(center, size, scenario)
    sim_data = web.run(sim, task_name="autograd_vs_fd_scenario", local_gradient=True, verbose=False)
    return anp.sum(sim_data.get_intensity("point_out").values)


def fd_vs_ad_param(center, size, scenario, param_label, delta=1e-3):
    val_and_grad_fn = autograd.value_and_grad(
        lambda c, s: objective_fn(c, s, scenario), argnum=(0, 1)
    )
    _, (grad_center, grad_size) = val_and_grad_fn(center, size)

    param_map = {
        "center_x": (0, "center"),
        "center_y": (1, "center"),
        "center_z": (2, "center"),
        "size_x": (0, "size"),
        "size_y": (1, "size"),
        "size_z": (2, "size"),
    }
    idx, which = param_map[param_label]
    if which == "center":
        ad_val = grad_center[idx]
    else:
        ad_val = grad_size[idx]

    center_arr = np.array(center, dtype=float)
    size_arr = np.array(size, dtype=float)

    if which == "center":
        cplus = center_arr.copy()
        cminus = center_arr.copy()
        cplus[idx] += delta
        cminus[idx] -= delta
        p_plus = objective_fn(tuple(cplus), tuple(size_arr), scenario)
        p_minus = objective_fn(tuple(cminus), tuple(size_arr), scenario)
    else:
        splus = size_arr.copy()
        sminus = size_arr.copy()
        splus[idx] += delta
        sminus[idx] -= delta
        p_plus = objective_fn(tuple(center_arr), tuple(splus), scenario)
        p_minus = objective_fn(tuple(center_arr), tuple(sminus), scenario)

    fd_val = (p_plus - p_minus) / (2.0 * delta)
    return fd_val, ad_val, p_plus, p_minus


@pytest.mark.numerical
@pytest.mark.parametrize("scenario", SCENARIOS, ids=[s["name"] for s in SCENARIOS])
@pytest.mark.parametrize(
    "param_label", ["center_x", "center_y", "center_z", "size_x", "size_y", "size_z"]
)
def test_autograd_vs_fd_scenarios(scenario, param_label):
    center0 = (0.0, 0.0, 0.0)
    size0 = (2.0, 2.0, 2.0)
    delta = 0.03

    fd_val, ad_val, p_plus, p_minus = fd_vs_ad_param(center0, size0, scenario, param_label, delta)

    assert np.isfinite(fd_val), f"FD derivative is not finite for param={param_label}"
    assert np.isfinite(ad_val), f"AD derivative is not finite for param={param_label}"

    denom = max(abs(fd_val), 1e-12)
    rel_diff = abs(fd_val - ad_val) / denom
    assert rel_diff < 0.3, f"Autograd vs FD mismatch: param={param_label}, diff={rel_diff:.1%}"
