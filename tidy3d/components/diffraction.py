"""Shared helpers for diffraction monitor order accounting."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, cast, get_args

import numpy as np

from tidy3d.components.types.base import DiffractionPolarization
from tidy3d.constants import C_0, EPSILON_0, MU_0, fp_eps

from .boundary import BlochBoundary
from .geometry.base import Geometry

if TYPE_CHECKING:
    from .medium import MediumType
    from .monitor import DiffractionMonitor
    from .simulation import Simulation


AXIS_NAMES = ("x", "y", "z")
BYTES_REAL = 4
BYTES_COMPLEX = 8
COS_THETA_THRESH = 1e-5
DIFFRACTION_POLARIZATIONS = cast(
    tuple[DiffractionPolarization, ...], get_args(DiffractionPolarization)
)


def shifted_orders(orders: tuple[int, ...], bloch_vec: float | np.ndarray) -> np.ndarray:
    """Diffraction orders shifted by the Bloch vector."""

    return bloch_vec + np.atleast_2d(orders).T


def reciprocal_coords(
    orders: np.ndarray,
    size: float,
    bloch_vec: float | np.ndarray,
    f: float,
    medium: MediumType,
) -> np.ndarray:
    """Get the normalized "u" reciprocal coords for a vector of orders, size, and bloch vec."""

    if size == 0:
        return np.atleast_2d(0)
    epsilon = medium.eps_model(f)
    bloch_array = shifted_orders(tuple(int(order) for order in np.atleast_1d(orders)), bloch_vec)
    return bloch_array / size * C_0 / f / np.real(np.sqrt(epsilon))


def compute_angles(
    reciprocal_vectors: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the polar and azimuth angles associated with the given reciprocal vectors."""

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="invalid value encountered in arcsin", category=RuntimeWarning
        )
        ux, uy = reciprocal_vectors
        thetas, phis = Geometry.kspace_2_sph(ux[:, None, :], uy[None, :, :], axis=2)
    return (thetas, phis)


def diffraction_amplitude_norm(theta_data: np.ndarray, eta: np.ndarray | complex) -> np.ndarray:
    """Amplitude normalization used by diffraction monitor power amplitudes."""

    cos_theta = np.cos(np.nan_to_num(theta_data))
    cos_theta[cos_theta <= COS_THETA_THRESH] = np.inf
    return 1.0 / np.sqrt(2.0 * np.asarray(eta)) / np.sqrt(cos_theta)


def diffraction_angle_is_propagating(angle_theta: float) -> bool:
    """Return whether a diffraction angle is valid for a propagating order."""

    return bool(np.isfinite(angle_theta) and np.cos(angle_theta) > COS_THETA_THRESH)


def bloch_vec_at_freq(bloch_vec: float | np.ndarray, freq_index: int) -> float:
    """Return scalar Bloch vector for a specific monitor frequency index."""

    if np.ndim(bloch_vec) == 0:
        return float(bloch_vec)
    return float(np.asarray(bloch_vec)[freq_index])


def diffraction_monitor_medium(sim: Simulation, monitor: DiffractionMonitor) -> MediumType:
    """Return medium on diffraction monitor plane using simulation-level monitor semantics."""

    return sim.monitor_medium(monitor)


def sim_bloch_vecs(
    sim: Simulation, monitor: DiffractionMonitor
) -> tuple[float | np.ndarray, float | np.ndarray]:
    """Get Bloch vectors for the diffraction monitor transverse axes."""

    fixed_angle_sources = sim._fixed_angle_sources
    if len(fixed_angle_sources) > 0:
        src = fixed_angle_sources[0]
        total_structures = (sim.scene.background_structure, *(sim.structures or ()))
        medium_set = sim.scene.intersecting_media(src, total_structures)
        medium = medium_set.pop() if medium_set else sim.medium
        background_n, _ = cast(Any, medium).nk_model(frequency=src.source_time._freq0)

        vec = np.array(src._dir_vector, dtype=float)
        vec[np.abs(vec) < fp_eps] = 0.0
        vec[src._injection_axis] = 0.0
        vec *= np.real(background_n)
        vec = vec * np.sqrt(EPSILON_0 * MU_0) * np.array(sim.size) * np.atleast_2d(monitor.freqs).T
        vec = np.delete(vec, src._injection_axis, axis=1)
        return vec[:, 0], vec[:, 1]

    bloch_x_0 = 0.0
    bloch_y_0 = 0.0
    _, (n_x, n_y) = monitor.pop_axis(AXIS_NAMES, axis=monitor.normal_axis)
    plus_x = sim.boundary_spec[n_x].plus
    plus_y = sim.boundary_spec[n_y].plus
    if isinstance(plus_x, BlochBoundary):
        bloch_x_0 = float(np.real(plus_x.bloch_vec))
    if isinstance(plus_y, BlochBoundary):
        bloch_y_0 = float(np.real(plus_y.bloch_vec))
    return bloch_x_0, bloch_y_0


def domain_size_2d(sim: Simulation, monitor: DiffractionMonitor) -> tuple[float, float]:
    """Get domain size in the diffraction monitor transverse directions."""

    _, (n_x, n_y) = monitor.pop_axis(AXIS_NAMES, axis=monitor.normal_axis)
    domain_x = sim.size[AXIS_NAMES.index(n_x)]
    domain_y = sim.size[AXIS_NAMES.index(n_y)]
    return domain_x, domain_y


def domain_center_2d(sim: Simulation, monitor: DiffractionMonitor) -> tuple[float, float]:
    """Get domain center in the diffraction monitor transverse directions."""

    center = sim.center
    assert center is not None
    _, (n_x, n_y) = monitor.pop_axis(AXIS_NAMES, axis=monitor.normal_axis)
    center_x = center[AXIS_NAMES.index(n_x)]
    center_y = center[AXIS_NAMES.index(n_y)]
    return center_x, center_y


def handle_zero_division(numerator: Any, denominator: float) -> Any:
    """Handle divide-by-zero for zero-size domains with zero Bloch vector."""

    return numerator / denominator if denominator else np.zeros_like(numerator)


def _generate_orders(
    wave_numbers: np.ndarray, bloch: float | np.ndarray, domain_size: float
) -> np.ndarray:
    """Generate the 1D diffraction orders allowed along one transverse axis."""

    allowed_orders = [0]
    if domain_size == 0:
        return np.array(allowed_orders, dtype=int)

    order = 0
    next_order = True
    while next_order:
        order += 1

        order_pos = shifted_orders((order,), bloch)
        pos_vec = handle_zero_division(order_pos * 2.0 * np.pi / wave_numbers, domain_size)

        order_neg = shifted_orders((-order,), bloch)
        neg_vec = handle_zero_division(order_neg * 2.0 * np.pi / wave_numbers, domain_size)

        next_order = False
        if (np.abs(pos_vec) <= 1.0).any():
            allowed_orders.append(order)
            next_order = True
        if (np.abs(neg_vec) <= 1.0).any():
            allowed_orders.append(-order)
            next_order = True

    return np.array(sorted(allowed_orders), dtype=int)


def diffraction_order_grid_axes(
    sim: Simulation, monitor: DiffractionMonitor, medium: MediumType
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute the raw rectangular diffraction order grid before 2D light-cone pruning."""

    bloch_x_0, bloch_y_0 = sim_bloch_vecs(sim, monitor)
    domain_x, domain_y = domain_size_2d(sim, monitor)
    epsilon = np.array([medium.eps_model(float(freq)) for freq in monitor.freqs])
    wave_numbers = np.real(2.0 * np.pi * np.array(monitor.freqs) / C_0 * np.sqrt(epsilon))

    orders_x = _generate_orders(wave_numbers=wave_numbers, bloch=bloch_x_0, domain_size=domain_x)
    orders_y = _generate_orders(wave_numbers=wave_numbers, bloch=bloch_y_0, domain_size=domain_y)

    bloch_x = shifted_orders(tuple(int(order) for order in orders_x), bloch_x_0)
    bloch_y = shifted_orders(tuple(int(order) for order in orders_y), bloch_y_0)

    ux = np.atleast_2d(
        handle_zero_division(bloch_x * 2.0 * np.pi / wave_numbers[None, :], domain_x)
    )
    uy = np.atleast_2d(
        handle_zero_division(bloch_y * 2.0 * np.pi / wave_numbers[None, :], domain_y)
    )
    return orders_x, orders_y, ux, uy


def diffraction_order_grid_size(
    sim: Simulation, monitor: DiffractionMonitor, medium: MediumType
) -> int:
    """Number of diffraction order combinations in the raw rectangular grid."""

    orders_x, orders_y, _, _ = diffraction_order_grid_axes(sim, monitor, medium)
    return int(orders_x.size * orders_y.size)


def diffraction_monitor_storage_size(
    sim: Simulation, monitor: DiffractionMonitor, medium: MediumType
) -> int:
    """Estimated final storage size in bytes for diffraction monitor data."""

    order_grid_size = diffraction_order_grid_size(sim, monitor, medium)
    return BYTES_COMPLEX * order_grid_size * len(monitor.freqs) * 6


def diffraction_orders(
    sim: Simulation, monitor: DiffractionMonitor, medium: MediumType
) -> tuple[np.ndarray, np.ndarray]:
    """Get the allowed diffraction orders based on the simulation setup."""

    orders_x, orders_y, ux, uy = diffraction_order_grid_axes(sim, monitor, medium)
    domain_x, domain_y = domain_size_2d(sim, monitor)

    ind_min = [len(orders_x), len(orders_y)]
    ind_max = [0, 0]
    count = 0

    for f, _ in enumerate(monitor.freqs):
        k_grids = np.meshgrid(ux[:, f], uy[:, f], indexing="ij")
        k_mags = np.abs(np.sqrt(k_grids[0] ** 2 + k_grids[1] ** 2))
        temp_inds = np.nonzero(k_mags <= 1.0)
        count += len(temp_inds[0])

        for idx in range(2):
            ind_arr = temp_inds[idx]
            if ind_arr.size > 0:
                ind_min[idx] = min(ind_min[idx], int(np.min(ind_arr)))
                ind_max[idx] = max(ind_max[idx], int(np.max(ind_arr)))

    if count == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    inds = [range(i_min, i_max + 1) for i_min, i_max in zip(ind_min, ind_max)]
    orders_x = np.array(orders_x)[inds[0]] if domain_x > 0 else np.atleast_1d(0)
    orders_y = np.array(orders_y)[inds[1]] if domain_y > 0 else np.atleast_1d(0)

    return orders_x, orders_y
