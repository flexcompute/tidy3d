#!/usr/bin/env python

"""
Notes:

## single frequency
* RMS error normalized (avg) = 0.3308823080089637 %

## multi-freq (n=3, df = 0.1) [all fields at freq0]
* RMS error normalized (avg) = 143.55391909049393 %
NOTE: when the df gets very small (freq0/200_000), error goes to 0%.

## multi-freq


"""


# NOTE: move to backend test later, uses webapi

import numpy as np
import jax.numpy as jnp
import jax
import tmm
import matplotlib.pyplot as plt
from typing import Tuple, List

import tidy3d as td
from tidy3d.web import run as run_sim
import tidy3d.plugins.adjoint as tda
from tidy3d.plugins.adjoint.web import run_local as run_adjoint

np.random.seed(0)

""" Generic Settings """

# background permittivity
bck_eps = 1.3**2

# space between each slab
spc = 0.0

# slab permittivities and thicknesses
num_slabs = 12
slab_d = 0.5

eps_max = 2.0

slab_eps0 = (1 + (eps_max - 1) * np.random.random(num_slabs)).tolist()
slab_ds0 = (slab_d * np.ones(num_slabs)).tolist()

# incidence angle
theta = 0 * np.pi / 8

# resolution
dl = 0.01


def compute_T_tmm(wavelength, slab_eps=slab_eps0, slab_ds=slab_ds0) -> float:
    """Get transmission as a function of slab permittivities and thicknesses."""

    # construct lists of permittivities and thicknesses including spaces between
    new_slab_eps = []
    new_slab_ds = []
    for eps, d in zip(slab_eps, slab_ds):
        new_slab_eps.append(eps)
        new_slab_eps.append(bck_eps)
        new_slab_ds.append(d)
        new_slab_ds.append(spc)
    slab_eps = new_slab_eps[:-1]
    slab_ds = new_slab_ds[:-1]

    # add the input and output spaces to the lists
    eps_list = [bck_eps] + slab_eps + [bck_eps]
    n_list = np.sqrt(eps_list)
    d_list = [np.inf] + slab_ds + [np.inf]

    # compute transmission with TMM
    return tmm.coh_tmm("p", n_list, d_list, theta, wavelength)["T"]


def compute_grad_tmm(
    wavelength, slab_eps=slab_eps0, slab_ds=slab_ds0
) -> np.ndarray[2, "num_slabs"]:
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
                if arg_index == 1 and spc == 0:
                    if slab_index > 0:
                        args_num[arg_index][slab_index - 1] -= delta * pm / 2
                    if slab_index < num_slabs - 1:
                        args_num[arg_index][slab_index + 1] -= delta * pm / 2

                # compute argument perturbed T and add to finite difference gradient contribution
                T_tmm = compute_T_tmm(wavelength, slab_eps=args_num[0], slab_ds=args_num[1])
                grad += pm * T_tmm / 2 / delta

            grad_tmm[arg_index][slab_index] = grad
    return grad_tmm


def compute_average_T_tmm(wavelengths, slab_eps=slab_eps0, slab_ds=slab_ds0) -> float:
    """Get average transmission as a function of slab permittivities and thicknesses."""

    Ts_tmm = [compute_T_tmm(wvl, slab_eps=slab_eps, slab_ds=slab_ds) for wvl in wavelengths]
    return np.mean(Ts_tmm)


def compute_average_grad_tmm(
    wavelengths, slab_eps=slab_eps0, slab_ds=slab_ds0
) -> np.ndarray[2, "num_slabs"]:
    """Get average gradient as a function of slab permittivities and thicknesses."""

    grads_tmm = [compute_grad_tmm(wvl, slab_eps=slab_eps, slab_ds=slab_ds) for wvl in wavelengths]
    grads_tmm = np.stack(grads_tmm, axis=-1)
    return np.mean(grads_tmm, axis=-1)


""" FDTD parts """


def make_sim(freq0, fwidth, slab_eps=slab_eps0, slab_ds=slab_ds0) -> tda.JaxSimulation:
    """Create a tda.JaxSimulation given the slab permittivities and thicknesses."""

    # geometry setup
    bck_medium = td.Medium(permittivity=bck_eps)

    space_above = 2
    space_below = 2

    length_x = 0.1
    length_y = 0.1
    length_z = space_below + sum(slab_ds0) + space_above + (len(slab_ds0) - 1) * spc
    sim_size = (length_x, length_y, length_z)

    # make structures
    slabs = []
    z_start = -length_z / 2 + space_below
    for (d, eps) in zip(slab_ds, slab_eps):

        # dont track the gradient through the center of each slab
        # as tidy3d doesn't have enough information to properly process the interface between touching tda.JaxBox objects
        z_center = jax.lax.stop_gradient(z_start + d / 2)
        slab = tda.JaxStructure(
            geometry=tda.JaxBox(center=[0, 0, z_center], size=[td.inf, td.inf, d]),
            medium=tda.JaxMedium(permittivity=eps),
        )
        slabs.append(slab)
        z_start += d + spc

    # forward source setup
    gaussian = td.GaussianPulse(freq0=freq0, fwidth=fwidth)
    src_z = -length_z / 2 + space_below / 2.0
    source = td.PlaneWave(
        center=(0, 0, src_z),
        size=(td.inf, td.inf, 0),
        source_time=gaussian,
        direction="+",
        angle_theta=theta,
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
        freqs=freqs.tolist(),
        name="diffraction",
        normal_dir="+",
    )

    # NOTE: IMPORTANT
    run_time = 10 / fwidth

    # make simulation
    return tda.JaxSimulation(
        size=sim_size,
        grid_spec=td.GridSpec.auto(min_steps_per_wvl=100),
        input_structures=slabs,
        sources=[source],
        output_monitors=[monitor_1],
        run_time=run_time,
        boundary_spec=boundary_spec,
        medium=bck_medium,
        subpixel=True,
        shutoff=1e-8,
    )


def post_process_T(sim_data: tda.JaxSimulationData) -> float:
    """Given some tda.JaxSimulationData from the run, return the transmission of "p" polarized light."""
    amps = sim_data.output_monitor_data["diffraction"].amps.sel(polarization="p")
    return jnp.sum(abs(amps.values) ** 2)


def compute_T_fdtd(slab_eps=slab_eps0, slab_ds=slab_ds0) -> float:
    """Given the slab permittivities and thicknesses, compute T, making sure to use `tidy3d.plugins.adjoint.web.run_adjoint`."""
    sim = make_sim(slab_eps=slab_eps, slab_ds=slab_ds, freq0=freq0, fwidth=fwidth)
    print("running FDTD")
    sim_data = run_adjoint(sim, task_name="slab", verbose=False)
    return post_process_T(sim_data)


compute_T_and_grad_fdtd = jax.value_and_grad(compute_T_fdtd, argnums=(0, 1))


def grad_error(freqs, freq0, fwidth, verbose=False):

    # do some variable switcheroo

    # frequencies and wavelengths we want to simulate at
    wavelengths = td.C_0 / freqs

    T_tmm = compute_average_T_tmm(wavelengths, slab_eps=slab_eps0, slab_ds=slab_ds0)

    grad_eps_tmm, grad_ds_tmm = compute_average_grad_tmm(wavelengths)

    # set logging level to ERROR to avoid redundant warnings from adjoint run
    td.config.logging_level = "ERROR"
    T_fdtd, (grad_eps_fdtd, grad_ds_fdtd) = compute_T_and_grad_fdtd(slab_eps0, slab_ds0)

    grad_eps_fdtd = np.array(grad_eps_fdtd)
    grad_ds_fdtd = np.array(grad_ds_fdtd)

    rms_eps = np.linalg.norm(grad_eps_tmm - grad_eps_fdtd) / np.linalg.norm(grad_eps_tmm)
    rms_ds = np.linalg.norm(grad_ds_tmm - grad_ds_fdtd) / np.linalg.norm(grad_ds_tmm)

    def normalize(arr):
        return arr / np.linalg.norm(arr)

    grad_eps_tmm_norm = normalize(grad_eps_tmm)
    grad_ds_tmm_norm = normalize(grad_ds_tmm)
    grad_eps_fdtd_norm = normalize(grad_eps_fdtd)
    grad_ds_fdtd_norm = normalize(grad_ds_fdtd)

    rms_eps_norm = np.linalg.norm(grad_eps_tmm_norm - grad_eps_fdtd_norm) / np.linalg.norm(
        grad_eps_tmm_norm
    )
    rms_ds_norm = np.linalg.norm(grad_ds_tmm_norm - grad_ds_fdtd_norm) / np.linalg.norm(
        grad_ds_tmm_norm
    )

    grad_tmm = np.mean([grad_eps_tmm, grad_ds_tmm])
    grad_fdtd = np.mean([grad_eps_fdtd, grad_ds_fdtd])
    rms = np.mean([rms_eps, rms_ds])
    rms_norm = np.mean([rms_eps_norm, rms_ds_norm])

    if verbose:
        print("RESULTS:\n")
        print(80 * "-", "\n")
        print(f"T (tmm)  = {T_tmm}")
        print(f"T (FDTD) = {T_fdtd}")
        print(80 * "-", "\n")

        print("un-normalized gradients:")
        print("")
        print(f"\tgrad_eps (tmm)  = {grad_eps_tmm}")
        print(f"\tgrad_eps (FDTD)  = {grad_eps_fdtd}")
        print("")
        print(f"\tgrad_ds  (tmm)  = {grad_ds_tmm}")
        print(f"\tgrad_ds  (FDTD)  = {grad_ds_fdtd}")
        print("")
        print(f"RMS error (eps) = {rms_eps * 100} %")
        print(f"RMS error (ds)  = {rms_ds * 100} %")
        print(f"RMS error (avg) = {rms * 100} %")
        print("")
        print(80 * "-", "\n")

        print("normalized gradients:")
        print("")
        print(f"\tgrad_eps normalized (tmm)  = {grad_eps_tmm_norm}")
        print(f"\tgrad_eps normalized (FDTD)  = {grad_eps_fdtd_norm}")
        print("")
        print(f"\tgrad_ds normalized (tmm)  = {grad_ds_tmm_norm}")
        print(f"\tgrad_ds normalized (FDTD)  = {grad_ds_fdtd_norm}")
        print("")
        print(f"RMS error normalized (eps) = {rms_eps_norm * 100} %")
        print(f"RMS error normalized (ds)  = {rms_ds_norm * 100} %")
        print(f"RMS error normalized (avg) = {rms_norm * 100} %")
        print(80 * "-", "\n")

    return rms, rms_norm, grad_tmm, grad_fdtd


""" Main script """

freq0 = 2e14
df = 0.00001e14
num_freqs = 3
fwidth = freq0 / 10

if num_freqs == 1:
    freqs = np.array([freq0])
else:
    freqs = np.linspace(freq0 - df, freq0 + df, num_freqs)
    fwidth = max(fwidth, np.max(freqs) - np.min(freqs))

verbose = True
grad_results = grad_error(freqs, freq0=freq0, fwidth=fwidth, verbose=verbose)
rms_raw, rms_norm, grad_tmm, grad_fdtd = grad_results
