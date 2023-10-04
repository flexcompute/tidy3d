"""Tests that multi-frequency adjoint gradient matches one based on run_async."""

# grads must match to this tolerance (error is fwidth and diff(freqs) dependent).
RELATIVE_TOLERANCE = 1e-3


import numpy as np
import jax.numpy as jnp
import jax

import tidy3d as td
import tidy3d.plugins.adjoint as tda
from tidy3d.plugins.adjoint.web import run, run_local, run_async


freq_min = 1.8e14
freq_max = 2.2e14
num_freqs = 2
freqs = np.linspace(freq_min, freq_max, num_freqs)
freq0 = np.mean(freqs)

wavelength_max = td.C_0 / freq_min
freqs = freqs.tolist()

permittivity_val = 2.0

lx = wavelength_max
ly = wavelength_max
lz = wavelength_max

buffer = 2 * wavelength_max

Lx = lx + 2 * buffer
Ly = ly + 2 * buffer
Lz = lz + 2 * buffer

src_pos_x = -Lx / 2 + buffer / 2
mnt_pos_x = +Lx / 2 - buffer / 2


def make_sim(permittivity: float, freqs: list) -> tda.JaxSimulation:
    """Make a simulation as a function of the box permittivity and the frequency."""

    box = tda.JaxStructure(
        geometry=tda.JaxBox(center=(0.0, 0.0, 0.0), size=(lx, ly, lz)),
        medium=tda.JaxMedium(permittivity=permittivity),
    )

    src = td.PointDipole(
        center=(src_pos_x, 0, 0),
        polarization="Ey",
        source_time=td.GaussianPulse(
            freq0=freq0,
            fwidth=freq0 / 6,
        ),
    )

    mnt = td.DiffractionMonitor(
        center=(mnt_pos_x, 0, 0),
        size=(0, td.inf, td.inf),
        freqs=freqs,
        name="diffraction",
    )

    return tda.JaxSimulation(
        size=(Lx, Ly, Lz),
        input_structures=[box],
        output_monitors=[mnt],
        sources=[src],
        grid_spec=td.GridSpec.auto(wavelength=td.C_0 / freq0),
        boundary_spec=td.BoundarySpec(
            x=td.Boundary.pml(), y=td.Boundary.periodic(), z=td.Boundary.periodic()
        ),
        run_time=200 / src.source_time.fwidth,
    )


jax_sim = make_sim(permittivity=permittivity_val, freqs=freqs)
ax = jax_sim.plot(z=0)


def post_process(sim_data: tda.JaxSimulationData) -> float:
    """O-th order diffracted power."""
    amp = sim_data["diffraction"].amps.sel(orders_x=0, orders_y=0)
    return jnp.sum(jnp.abs(amp.values) ** 2)


def objective(permittivity: float, freqs: list) -> float:
    """Average of O-th order diffracted power over all frequencies."""
    sim = make_sim(permittivity, freqs)
    sim_data = run_local(sim, task_name="MULTIFREQ", verbose=False)
    power = post_process(sim_data)
    return jnp.sum(jnp.array(power)) / len(freqs)


power_average = objective(permittivity=permittivity_val, freqs=freqs)
print(f"average power (freq) = {power_average:.2e}")

grad_objective = jax.grad(objective)

grad_power_average = grad_objective(permittivity_val, freqs=freqs)
print(f"derivative of average power wrt permittivity = {grad_power_average:.2e}")


def grad_manual(permittivity: float) -> float:
    """Average of O-th order diffracted power over all frequencies."""

    total_grad = 0.0
    for freq in freqs:
        print(f"working on freq = {freq:.2e} (Hz)")
        obj_fn = lambda x: objective(x, freqs=[freq])
        grad_fn = jax.grad(obj_fn)
        gradient = grad_fn(permittivity)
        total_grad += gradient

    return total_grad / len(freqs)


grad_man = grad_manual(permittivity_val)


print(f"gradient (batched) = {grad_power_average:.4e}")
print(f"gradient (looped) = {grad_man:.4e}")

assert np.isclose(
    grad_man, grad_power_average, rtol=RELATIVE_TOLERANCE
), "Async Multifrequency grads dont match."
