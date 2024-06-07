#!/usr/bin/env python3

import autograd.numpy as anp
import matplotlib.pyplot as plt
import nlopt
import numpy as np
from autograd import value_and_grad

import tidy3d as td
from tidy3d.plugins.autograd.functions import rescale
from tidy3d.plugins.autograd.invdes import (
    get_kernel_size_px,
    make_filter,
    tanh_projection,
)
from tidy3d.web import run


def make_simulation(
    design: np.ndarray,
    *,
    wavelength: float = 1.0,
    min_steps_per_wvl: int = 20,
    design_dl: float = 0.01,
    wg_width: float = 0.7,
    eps_wg: float = 2.75,
    buffer: float = 1.0,
    monitor_offset: float = 0.1,
    run_time_mult: int = 10,
    with_field_monitor: bool = False,
) -> td.Simulation:
    nx, ny = design.shape
    lx = int(nx * design_dl)
    ly = int(ny * design_dl)
    sx, sy, sz = (lx + 2 * buffer, ly + 2 * buffer, 0)

    freq0 = td.C_0 / wavelength
    freqw = freq0 / 10
    run_time = run_time_mult / freqw

    eps_r = rescale(design, 1.0, eps_wg)
    permittivity = td.SpatialDataArray(
        anp.reshape(eps_r, (*eps_r.shape, 1)),
        coords={
            "x": np.linspace((-lx + design_dl) / 2, (lx - design_dl) / 2, nx),
            "y": np.linspace((-ly + design_dl) / 2, (ly - design_dl) / 2, ny),
            "z": [0],
        },
    )
    design_region = td.Structure(
        geometry=td.Box(center=(0, 0, 0), size=(lx, ly, td.inf)),
        medium=td.CustomMedium(permittivity=permittivity),
    )
    waveguide = td.Structure(
        geometry=td.Box(size=(td.inf, wg_width, td.inf)),
        medium=td.Medium(permittivity=eps_wg),
    )

    mode_source = td.ModeSource(
        source_time=td.GaussianPulse(freq0=freq0, fwidth=freqw),
        center=(-sx / 2 + monitor_offset, 0, 0),
        size=(0, 3 * wg_width, td.inf),
        mode_index=0,
        mode_spec=td.ModeSpec(num_modes=1, target_neff=np.sqrt(eps_wg)),
        direction="+",
    )

    mode_monitor = td.ModeMonitor(
        center=(sx / 2 - monitor_offset, *mode_source.center[1:]),
        size=mode_source.size,
        freqs=[freq0],
        mode_spec=td.ModeSpec(num_modes=3, target_neff=np.sqrt(eps_wg)),
        name="mode_monitor",
    )
    monitors = [mode_monitor]

    if with_field_monitor:
        monitors.append(
            td.FieldMonitor(
                center=(0, 0, 0),
                size=(td.inf, td.inf, 0),
                freqs=[freq0],
                fields=("Ex", "Ey", "Ez"),
                name="field_monitor",
            )
        )

    grid_spec = td.GridSpec.auto(
        wavelength=wavelength,
        min_steps_per_wvl=min_steps_per_wvl,
        override_structures=[
            td.MeshOverrideStructure(
                geometry=design_region.geometry, dl=[design_dl] * 3, enforce=True
            )
        ],
    )
    boundary_spec = td.BoundarySpec(
        x=td.Boundary.pml(), y=td.Boundary.pml(), z=td.Boundary.periodic()
    )

    return td.Simulation(
        size=(sx, sy, sz),
        grid_spec=grid_spec,
        structures=(waveguide, design_region),
        sources=(mode_source,),
        monitors=monitors,
        run_time=run_time,
        boundary_spec=boundary_spec,
    )


def get_mode_power(sim_data: td.SimulationData, *, mode_index: int = 2) -> float:
    output_amps = sim_data["mode_monitor"].amps
    amp = output_amps.sel(direction="+", mode_index=mode_index).isel(f=0).values
    power = anp.sum(anp.abs(amp) ** 2)
    return power


def main():
    rng = np.random.default_rng(294587)

    sx = 3
    sy = 2
    dl = 0.01
    filter_radius = 0.1
    beta = 20

    x0 = rng.uniform(0.45, 0.55, (int(sx / dl), int(sy / dl)))
    lb = np.zeros_like(x0)
    ub = np.ones_like(x0)

    kernel_size = get_kernel_size_px(filter_radius, dl)
    filter_fn = make_filter("conic", kernel_size)

    def parametrization(x, beta=beta, eta=0.5):
        x = filter_fn(x)
        x = tanh_projection(x, beta, eta)
        return x

    def objective(x):
        x = parametrization(x)
        sim = make_simulation(x, design_dl=dl)
        sim_data = run(sim, task_name="invdes", verbose=False)
        mode_power = get_mode_power(sim_data)
        return mode_power

    vg_fun = value_and_grad(objective)

    cnt = iter(range(100))

    def nlopt_obj(x, gd):
        x = np.reshape(x, x0.shape)

        step = next(cnt)
        fig, ax = plt.subplots(tight_layout=True)
        ax.imshow(parametrization(x), cmap="gray_r", vmin=0, vmax=1)
        fig.savefig(f"step_{step:02d}.png", bbox_inches="tight", pad_inches=0.1, dpi=300)
        plt.close(fig)

        v, g = vg_fun(x)
        print(f"{step}: {v}")
        if gd.size > 0:
            gd[:] = g.ravel()
        return v.item()

    opt = nlopt.opt(nlopt.LD_MMA, x0.size)
    opt.set_lower_bounds(lb.ravel())
    opt.set_upper_bounds(ub.ravel())
    opt.set_max_objective(nlopt_obj)
    opt.set_maxeval(10)

    xopt = opt.optimize(x0.ravel())

    sim = make_simulation(np.reshape(xopt, x0.shape), design_dl=dl)  # noqa: F841


if __name__ == "__main__":
    main()
