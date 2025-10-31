from __future__ import annotations

import autograd.numpy as np
import xarray as xr
from autograd import value_and_grad

import tidy3d as td
from tidy3d import web


def make_sim():
    source = td.PlaneWave(
        center=(0.0, 0.0, -2.0),
        size=(td.inf, td.inf, 0.0),
        source_time=td.GaussianPulse(freq0=td.C_0 / 1.5, fwidth=0.2 * td.C_0 / 1.5),
        direction="+",
    )

    monitor = td.FieldMonitor(
        center=(0.0, 0.0, 1.0), size=(1.5, 1.5, 0.0), freqs=[td.C_0 / 1.5], name="field"
    )

    boundary_spec = td.BoundarySpec(
        x=td.Boundary.pml(),
        y=td.Boundary.pml(),
        z=td.Boundary.pml(),
    )

    return td.Simulation(
        center=(0.0, 0.0, 0.0),
        size=(5.0, 5.0, 5.0),
        grid_spec=td.GridSpec.auto(
            min_steps_per_wvl=30,
            wavelength=1.5,
            # override_structures=mesh_overrides,
        ),
        boundary_spec=boundary_spec,
        sources=[source],
        monitors=[monitor],
        structures=[],
        run_time=1e-11,
    )


def vjp_sphere(sphere, derivative_info):
    max_frequency = np.max(derivative_info.frequencies)
    min_wvl = td.C_0 / max_frequency

    step_size = min_wvl / 20.0

    ps_paths = set()
    # for path in derivative_info.paths:
    ps_paths.update({("permittivity",)})

    # pass interpolators to PolySlab if available to avoid redundant conversions
    update_kwargs = {
        "paths": list(ps_paths),
        "deep": False,
    }
    derivative_info_custom_medium = derivative_info.updated_copy(**update_kwargs)

    vjps = {}
    for path in derivative_info.paths:
        if path == ("radius",):
            sphere_up = sphere.updated_copy(radius=sphere.radius + step_size)
            sphere_down = sphere.updated_copy(radius=sphere.radius - step_size)

            eps_up = derivative_info.updated_epsilon(sphere_up)
            eps_down = derivative_info.updated_epsilon(sphere_down)

            eps_grad = (eps_up - eps_down) / (2 * step_size)

            custom_medium = td.CustomMedium(
                permittivity=xr.ones_like(eps_grad.isel(f=0, drop=True))
            )
            vjps_custom_medium = custom_medium._compute_derivatives(derivative_info_custom_medium)

            total_grad = np.real(
                np.sum(eps_grad.sum("f").data * vjps_custom_medium[("permittivity",)])
            )

            vjps[path] = total_grad
        elif "center" in path:
            if len(path) == 1:
                center_indices = (0, 1, 2)
            else:
                _, center_index = path
                center_indices = [center_index]

            vjp_result = []
            for center_index in center_indices:
                center_up = list(sphere.center)
                center_down = list(sphere.center)

                center_up[center_index] += step_size
                center_down[center_index] -= step_size

                sphere_up = sphere.updated_copy(center=center_up)
                sphere_down = sphere.updated_copy(center=center_down)

                eps_up = derivative_info.updated_epsilon(sphere_up)
                eps_down = derivative_info.updated_epsilon(sphere_down)

                eps_grad = (eps_up - eps_down) / (2 * step_size)

                custom_medium = td.CustomMedium(
                    permittivity=xr.ones_like(eps_grad.isel(f=0, drop=True))
                )
                vjps_custom_medium = custom_medium._compute_derivatives(
                    derivative_info_custom_medium
                )

                total_grad = np.real(
                    np.sum(eps_grad.sum("f").data * vjps_custom_medium[("permittivity",)])
                )

                vjp_result.append(total_grad)

            vjps[path] = vjp_result if len(path) == 1 else vjp_result[0]

    return vjps


def obj_fn(params):
    sphere = td.Structure(
        geometry=td.Sphere(center=(params[0], params[1], params[2]), radius=params[3]),
        medium=td.Medium(permittivity=1.5**2),
    )

    sim_base = make_sim()

    updated_sim = sim_base.updated_copy(structures=[sphere])

    sim_data = web.run(
        updated_sim,
        "sphere_grad",
        local_gradient=True,
        verbose=True,
        user_vjp=((0, "radius", vjp_sphere), (0, "center", vjp_sphere)),
    )

    return np.sum(np.abs(sim_data["field"].flux.data) ** 2)


v_and_g = value_and_grad(obj_fn)

init_r = 0.75
init_params = np.array([0.5, 0.25, 0.0, init_r])
f, g = v_and_g(init_params)

print(f"f = {f}")
print(f"g = {g}")

# fd_grad = []
# for param_idx in range(0, len(init_params)):
#     print(f'Working on finite difference idx {param_idx}')
#     params_up = init_params.copy()
#     params_down = init_params.copy()

#     params_up[param_idx] += 0.02
#     params_down[param_idx] -= 0.02

#     check_g_up = obj_fn(params_up)
#     check_g_down = obj_fn(params_down)

#     check_g = (check_g_up - check_g_down) / (2 * 0.02)

#     fd_grad.append(check_g)

# print(f'fd grad = {fd_grad}')

# print(obj_fn(init_params + 10. * g))
