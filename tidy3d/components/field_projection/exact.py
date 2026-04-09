"""Exact field projection paths and kernels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import autograd.numpy as anp
import numpy as np
from autograd.extend import defvjp, primitive

from tidy3d.components.autograd import hasbox
from tidy3d.components.data.monitor_data import AbstractFieldProjectionData
from tidy3d.constants import EPSILON_0, MU_0

from .common import (
    EXACT_PROJECTION_BATCH_SIZE,
    _frequency_chunk_slices,
    _track_if_verbose,
    _trapz_weights_1d,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    import xarray as xr
    from numpy.typing import NDArray

    from tidy3d.components.medium import MediumType
    from tidy3d.components.monitor import AbstractFieldProjectionMonitor, FieldProjectionSurface
    from tidy3d.components.types import Axis

    from .common import ArrayLikeN2F


@dataclass(frozen=True)
class _ExactProjectionStatic:
    """Point-independent exact-projection data for one source surface."""

    surface_monitor: AbstractFieldProjectionMonitor
    pts: tuple[np.ndarray, np.ndarray, np.ndarray]
    idx_u: Axis
    idx_v: Axis
    idx_w: Axis
    normal_axis_size: int
    i_omega: ArrayLikeN2F
    wavenumber: ArrayLikeN2F
    epsilon: ArrayLikeN2F
    integration_weights: ArrayLikeN2F

    @property
    def num_frequencies(self) -> int:
        """Return the number of frequencies represented by this prepared chunk."""

        return self.i_omega.shape[-1]


@dataclass(frozen=True)
class _ExactProjectionPrepared:
    """Exact-projection data with point-dependent geometry terms attached."""

    static: _ExactProjectionStatic
    i_omega: ArrayLikeN2F
    wavenumber: ArrayLikeN2F
    epsilon: ArrayLikeN2F
    integration_weights: ArrayLikeN2F
    green: ArrayLikeN2F
    d_green: ArrayLikeN2F
    d_perp: ArrayLikeN2F
    d_parallel: ArrayLikeN2F
    r_hat_x: ArrayLikeN2F
    r_hat_y: ArrayLikeN2F
    r_hat_z: ArrayLikeN2F
    theta_out: ArrayLikeN2F
    phi_out: ArrayLikeN2F


@dataclass(frozen=True)
class _PreparedExactSurfaceProjection:
    """Prepared exact projection data reused across observation points."""

    currents_tangential: np.ndarray
    projection: _ExactProjectionStatic


@dataclass(frozen=True)
class _ExactSurfaceCurrentData:
    """Exact surface current data reused across frequency chunks."""

    surface: FieldProjectionSurface
    currents: xr.Dataset
    currents_tangential: np.ndarray


@dataclass(frozen=True)
class _ExactProjectionGeometry:
    """Point-dependent exact-projection Green-function geometry terms."""

    green: ArrayLikeN2F
    d_green: ArrayLikeN2F
    d_perp: ArrayLikeN2F
    d_parallel: ArrayLikeN2F
    r_hat_x: ArrayLikeN2F
    r_hat_y: ArrayLikeN2F
    r_hat_z: ArrayLikeN2F


def _sph_2_car_field_components(
    f_r: ArrayLikeN2F,
    f_theta: ArrayLikeN2F,
    f_phi: ArrayLikeN2F,
    theta: ArrayLikeN2F,
    phi: ArrayLikeN2F,
) -> tuple[ArrayLikeN2F, ArrayLikeN2F, ArrayLikeN2F]:
    """Convert spherical vector-field components to Cartesian components."""

    sin_theta = anp.sin(theta)
    cos_theta = anp.cos(theta)
    sin_phi = anp.sin(phi)
    cos_phi = anp.cos(phi)
    f_x = f_r * sin_theta * cos_phi + f_theta * cos_theta * cos_phi - f_phi * sin_phi
    f_y = f_r * sin_theta * sin_phi + f_theta * cos_theta * sin_phi + f_phi * cos_phi
    f_z = f_r * cos_theta - f_theta * sin_theta
    return f_x, f_y, f_z


def _car_2_sph_field_components(
    f_x: ArrayLikeN2F,
    f_y: ArrayLikeN2F,
    f_z: ArrayLikeN2F,
    theta: ArrayLikeN2F,
    phi: ArrayLikeN2F,
) -> tuple[ArrayLikeN2F, ArrayLikeN2F, ArrayLikeN2F]:
    """Convert Cartesian vector-field components to spherical components."""

    sin_theta = anp.sin(theta)
    cos_theta = anp.cos(theta)
    sin_phi = anp.sin(phi)
    cos_phi = anp.cos(phi)
    f_r = f_x * sin_theta * cos_phi + f_y * sin_theta * sin_phi + f_z * cos_theta
    f_theta = f_x * cos_theta * cos_phi + f_y * cos_theta * sin_phi - f_z * sin_theta
    f_phi = -f_x * sin_phi + f_y * cos_phi
    return f_r, f_theta, f_phi


def _prepare_exact_projection_geometry(
    x_new: ArrayLikeN2F,
    y_new: ArrayLikeN2F,
    z_new: ArrayLikeN2F,
    surface_monitor: AbstractFieldProjectionMonitor,
    wavenumber: ArrayLikeN2F,
) -> _ExactProjectionGeometry:
    """Build exact Green-function terms and local spherical basis data."""

    r, theta_local, phi_local = surface_monitor.car_2_sph(x_new, y_new, z_new)
    sin_theta = anp.sin(theta_local)
    cos_theta = anp.cos(theta_local)
    sin_phi = anp.sin(phi_local)
    cos_phi = anp.cos(phi_local)

    ikr = 1j * wavenumber * r
    green = anp.exp(ikr) / (4.0 * np.pi * r)
    d_green = green * (ikr - 1.0) / r
    d2_green = d_green * (ikr - 1.0) / r + green / (r**2)

    return _ExactProjectionGeometry(
        green,
        d_green,
        d_green / r,
        d2_green - d_green / r,
        sin_theta * cos_phi,
        sin_theta * sin_phi,
        cos_theta,
    )


def _prepare_exact_surface_projection_static(
    surface: FieldProjectionSurface,
    currents: xr.Dataset,
    medium: MediumType,
    frequencies: ArrayLikeN2F,
) -> _ExactProjectionStatic:
    """Build point-independent exact-projection data for one source surface."""

    freqs = anp.asarray(frequencies)
    pts = tuple(currents[name].values for name in ("x", "y", "z"))

    idx_w, idx_uv = surface.monitor.pop_axis((0, 1, 2), axis=surface.axis)
    idx_u, idx_v = idx_uv

    integration_weights = anp.ones((*tuple(len(pt) for pt in pts), 1), dtype=float)
    for axis in (idx_u, idx_v):
        axis_weights = _trapz_weights_1d(pts[axis])
        reshape = [1, 1, 1, 1]
        reshape[axis] = axis_weights.size
        integration_weights = integration_weights * axis_weights.reshape(reshape)

    wavenumber = AbstractFieldProjectionData.wavenumber(frequency=freqs, medium=medium)
    wavenumber = wavenumber[None, None, None, :]

    return _ExactProjectionStatic(
        surface_monitor=surface.monitor,
        pts=pts,
        idx_u=idx_u,
        idx_v=idx_v,
        idx_w=idx_w,
        normal_axis_size=pts[idx_w].size,
        i_omega=1j * 2.0 * np.pi * freqs[None, None, None, :],
        wavenumber=wavenumber,
        epsilon=EPSILON_0 * medium.eps_model(frequency=freqs)[None, None, None, :],
        integration_weights=integration_weights,
    )


def _prepare_exact_surface_projection_point(
    x: float,
    y: float,
    z: float,
    prepared: _ExactProjectionStatic,
) -> _ExactProjectionPrepared:
    """Add point-dependent exact-projection geometry for one observation point."""

    pts = prepared.pts
    surface_monitor = prepared.surface_monitor
    x_new, y_new, z_new = (pt_obs - pt_src for pt_src, pt_obs in zip(pts, (x, y, z)))

    _, theta_out, phi_out = surface_monitor.car_2_sph(x, y, z)

    geometry = _prepare_exact_projection_geometry(
        x_new[:, None, None, None],
        y_new[None, :, None, None],
        z_new[None, None, :, None],
        surface_monitor,
        prepared.wavenumber,
    )
    return _ExactProjectionPrepared(
        static=prepared,
        i_omega=prepared.i_omega,
        wavenumber=prepared.wavenumber,
        epsilon=prepared.epsilon,
        integration_weights=prepared.integration_weights,
        green=geometry.green,
        d_green=geometry.d_green,
        d_perp=geometry.d_perp,
        d_parallel=geometry.d_parallel,
        r_hat_x=geometry.r_hat_x,
        r_hat_y=geometry.r_hat_y,
        r_hat_z=geometry.r_hat_z,
        theta_out=theta_out,
        phi_out=phi_out,
    )


def _prepare_exact_surface_projection_batch(
    x: ArrayLikeN2F,
    y: ArrayLikeN2F,
    z: ArrayLikeN2F,
    prepared: _ExactProjectionStatic,
) -> _ExactProjectionPrepared:
    """Add point-dependent exact-projection geometry for a batch of observation points."""

    x = anp.asarray(x)
    y = anp.asarray(y)
    z = anp.asarray(z)
    pts = prepared.pts
    surface_monitor = prepared.surface_monitor
    x_new = x[None, None, None, None, :] - pts[0][:, None, None, None, None]
    y_new = y[None, None, None, None, :] - pts[1][None, :, None, None, None]
    z_new = z[None, None, None, None, :] - pts[2][None, None, :, None, None]

    wavenumber = prepared.wavenumber[..., None]
    _, theta_out, phi_out = surface_monitor.car_2_sph(x, y, z)

    geometry = _prepare_exact_projection_geometry(x_new, y_new, z_new, surface_monitor, wavenumber)
    return _ExactProjectionPrepared(
        static=prepared,
        i_omega=prepared.i_omega[..., None],
        wavenumber=wavenumber,
        epsilon=prepared.epsilon[..., None],
        integration_weights=prepared.integration_weights[..., None],
        green=geometry.green,
        d_green=geometry.d_green,
        d_perp=geometry.d_perp,
        d_parallel=geometry.d_parallel,
        r_hat_x=geometry.r_hat_x,
        r_hat_y=geometry.r_hat_y,
        r_hat_z=geometry.r_hat_z,
        theta_out=theta_out,
        phi_out=phi_out,
    )


def _prepare_exact_surface_currents(
    surface: FieldProjectionSurface, currents: xr.Dataset
) -> np.ndarray:
    """Extract and stack tangential currents for one source surface."""

    _, source_names = surface.monitor.pop_axis(("x", "y", "z"), axis=surface.axis)
    cmp_1, cmp_2 = source_names
    return anp.stack(
        [
            currents[f"E{cmp_1}"].data,
            currents[f"E{cmp_2}"].data,
            currents[f"H{cmp_1}"].data,
            currents[f"H{cmp_2}"].data,
        ]
    )


def _prepare_exact_surface_projection_chunk(
    exact_surface_currents: list[_ExactSurfaceCurrentData],
    medium: MediumType,
    frequencies: np.ndarray,
    freq_slice: slice,
) -> list[_PreparedExactSurfaceProjection]:
    """Prepare exact projection data for one frequency chunk across all surfaces."""

    frequencies_chunk = frequencies[freq_slice]
    return [
        _PreparedExactSurfaceProjection(
            currents_tangential=surface_data.currents_tangential[..., freq_slice],
            projection=_prepare_exact_surface_projection_static(
                surface=surface_data.surface,
                currents=surface_data.currents,
                medium=medium,
                frequencies=frequencies_chunk,
            ),
        )
        for surface_data in exact_surface_currents
    ]


def _expand_tangential_surface_currents(
    currents_tangential: np.ndarray,
    prepared: _ExactProjectionPrepared,
) -> tuple[list[ArrayLikeN2F], list[ArrayLikeN2F]]:
    """Insert tangential surface currents into Cartesian 3-vectors."""

    j_u, j_v, m_u, m_v = currents_tangential
    idx_u = prepared.static.idx_u
    idx_v = prepared.static.idx_v
    idx_w = prepared.static.idx_w

    electric = [None] * 3
    magnetic = [None] * 3
    electric[idx_u] = j_u
    electric[idx_v] = j_v
    electric[idx_w] = anp.zeros_like(j_u)
    magnetic[idx_u] = m_u
    magnetic[idx_v] = m_v
    magnetic[idx_w] = anp.zeros_like(m_u)
    return electric, magnetic


def _cross_with_rhat(
    current: list[ArrayLikeN2F], prepared: _ExactProjectionPrepared
) -> list[ArrayLikeN2F]:
    """Compute ``r_hat x current`` pointwise on the source surface."""

    r_hat_x = prepared.r_hat_x
    r_hat_y = prepared.r_hat_y
    r_hat_z = prepared.r_hat_z
    return [
        r_hat_y * current[2] - r_hat_z * current[1],
        r_hat_z * current[0] - r_hat_x * current[2],
        r_hat_x * current[1] - r_hat_y * current[0],
    ]


def _apply_exact_grad_div_operator(
    current: list[ArrayLikeN2F], prepared: _ExactProjectionPrepared
) -> list[ArrayLikeN2F]:
    """Apply the exact Green-function gradient-divergence operator."""

    r_hat_x = prepared.r_hat_x
    r_hat_y = prepared.r_hat_y
    r_hat_z = prepared.r_hat_z
    d_perp = prepared.d_perp
    d_parallel = prepared.d_parallel
    r_dot_current = r_hat_x * current[0] + r_hat_y * current[1] + r_hat_z * current[2]
    return [
        d_perp * current[0] + d_parallel * r_hat_x * r_dot_current,
        d_perp * current[1] + d_parallel * r_hat_y * r_dot_current,
        d_perp * current[2] + d_parallel * r_hat_z * r_dot_current,
    ]


def _integrate_exact_surface_quantity(
    quantity: ArrayLikeN2F, prepared: _ExactProjectionPrepared
) -> ArrayLikeN2F:
    """Integrate a surface quantity over the two tangential monitor axes."""

    integrated = anp.sum(
        quantity * prepared.integration_weights,
        axis=(prepared.static.idx_u, prepared.static.idx_v),
    )
    if prepared.static.normal_axis_size == 1:
        integrated = integrated[0]
    return integrated


def _exact_surface_integrands(
    electric_current: list[ArrayLikeN2F],
    magnetic_current: list[ArrayLikeN2F],
    prepared: _ExactProjectionPrepared,
) -> tuple[list[ArrayLikeN2F], list[ArrayLikeN2F]]:
    """Build exact electric and magnetic field integrands."""

    grad_div_electric = _apply_exact_grad_div_operator(electric_current, prepared)
    grad_div_magnetic = _apply_exact_grad_div_operator(magnetic_current, prepared)
    cross_electric = _cross_with_rhat(electric_current, prepared)
    cross_magnetic = _cross_with_rhat(magnetic_current, prepared)

    i_omega = prepared.i_omega
    inv_wavenumber_sq = 1.0 / (prepared.wavenumber**2)
    epsilon = prepared.epsilon
    green = prepared.green
    d_green = prepared.d_green

    e_integrand = [
        i_omega
        * MU_0
        * (green * electric_current[idx] + grad_div_electric[idx] * inv_wavenumber_sq)
        - d_green * cross_magnetic[idx]
        for idx in range(3)
    ]
    h_integrand = [
        d_green * cross_electric[idx]
        + i_omega
        * epsilon
        * (green * magnetic_current[idx] + grad_div_magnetic[idx] * inv_wavenumber_sq)
        for idx in range(3)
    ]
    return e_integrand, h_integrand


def _integrate_exact_surface_fields(
    e_integrand: list[ArrayLikeN2F],
    h_integrand: list[ArrayLikeN2F],
    prepared: _ExactProjectionPrepared,
) -> tuple[ArrayLikeN2F, ArrayLikeN2F, ArrayLikeN2F, ArrayLikeN2F, ArrayLikeN2F, ArrayLikeN2F]:
    """Integrate exact electric and magnetic field integrands over the source surface."""

    e_x, e_y, e_z = [_integrate_exact_surface_quantity(item, prepared) for item in e_integrand]
    h_x, h_y, h_z = [_integrate_exact_surface_quantity(item, prepared) for item in h_integrand]
    return e_x, e_y, e_z, h_x, h_y, h_z


def _stack_exact_surface_fields(
    e_x: ArrayLikeN2F,
    e_y: ArrayLikeN2F,
    e_z: ArrayLikeN2F,
    h_x: ArrayLikeN2F,
    h_y: ArrayLikeN2F,
    h_z: ArrayLikeN2F,
    prepared: _ExactProjectionPrepared,
    *,
    batch_axis_last: bool = False,
) -> np.ndarray:
    """Convert integrated Cartesian fields to stacked spherical components."""

    e_r, e_theta, e_phi = _car_2_sph_field_components(
        e_x, e_y, e_z, prepared.theta_out, prepared.phi_out
    )
    h_r, h_theta, h_phi = _car_2_sph_field_components(
        h_x, h_y, h_z, prepared.theta_out, prepared.phi_out
    )
    fields = anp.array([e_r, e_theta, e_phi, h_r, h_theta, h_phi])
    if batch_axis_last:
        return anp.moveaxis(fields, -1, 0)
    return fields


def _exact_surface_adjoint_terms(
    g_electric_integrand: list[ArrayLikeN2F],
    g_magnetic_integrand: list[ArrayLikeN2F],
    prepared: _ExactProjectionPrepared,
) -> tuple[list[ArrayLikeN2F], list[ArrayLikeN2F]]:
    """Apply the exact adjoint operator to integrated Cartesian field cotangents."""

    grad_div_electric = _apply_exact_grad_div_operator(g_electric_integrand, prepared)
    grad_div_magnetic = _apply_exact_grad_div_operator(g_magnetic_integrand, prepared)
    cross_electric = _cross_with_rhat(g_electric_integrand, prepared)
    cross_magnetic = _cross_with_rhat(g_magnetic_integrand, prepared)

    i_omega = prepared.i_omega
    inv_wavenumber_sq = 1.0 / (prepared.wavenumber**2)
    epsilon = prepared.epsilon
    green = prepared.green
    d_green = prepared.d_green

    grad_electric = [
        i_omega
        * MU_0
        * (green * g_electric_integrand[idx] + grad_div_electric[idx] * inv_wavenumber_sq)
        - d_green * cross_magnetic[idx]
        for idx in range(3)
    ]
    grad_magnetic = [
        d_green * cross_electric[idx]
        + i_omega
        * epsilon
        * (green * g_magnetic_integrand[idx] + grad_div_magnetic[idx] * inv_wavenumber_sq)
        for idx in range(3)
    ]
    return grad_electric, grad_magnetic


def _stack_exact_surface_current_gradients(
    grad_electric: list[ArrayLikeN2F],
    grad_magnetic: list[ArrayLikeN2F],
    prepared: _ExactProjectionPrepared,
    *,
    sum_batch_axis: bool = False,
) -> np.ndarray:
    """Collect tangential exact-kernel gradients back into stacked current components."""

    idx_u = prepared.static.idx_u
    idx_v = prepared.static.idx_v
    gradients = [
        grad_electric[idx_u],
        grad_electric[idx_v],
        grad_magnetic[idx_u],
        grad_magnetic[idx_v],
    ]
    if sum_batch_axis:
        gradients = [anp.sum(component, axis=-1) for component in gradients]
    return anp.stack(gradients)


def _fields_for_surface_exact_impl(
    currents_tangential: np.ndarray,
    prepared: _ExactProjectionPrepared,
) -> np.ndarray:
    """Exact near-field projection kernel evaluated on tangential surface currents."""

    electric_current, magnetic_current = _expand_tangential_surface_currents(
        currents_tangential, prepared
    )
    e_integrand, h_integrand = _exact_surface_integrands(
        electric_current, magnetic_current, prepared
    )
    return _stack_exact_surface_fields(
        *_integrate_exact_surface_fields(e_integrand, h_integrand, prepared),
        prepared,
    )


def _fields_for_surface_exact_vjp_impl(
    g: np.ndarray, prepared: _ExactProjectionPrepared
) -> np.ndarray:
    """Exact near-field projection VJP with respect to tangential surface currents."""

    g_electric = _sph_2_car_field_components(g[0], g[1], g[2], prepared.theta_out, prepared.phi_out)
    g_magnetic = _sph_2_car_field_components(g[3], g[4], g[5], prepared.theta_out, prepared.phi_out)

    g_electric_integrand = [prepared.integration_weights * item for item in g_electric]
    g_magnetic_integrand = [prepared.integration_weights * item for item in g_magnetic]
    return _stack_exact_surface_current_gradients(
        *_exact_surface_adjoint_terms(g_electric_integrand, g_magnetic_integrand, prepared),
        prepared,
    )


@primitive
def _fields_for_surface_exact_primitive(
    currents_tangential: np.ndarray,
    prepared: _ExactProjectionPrepared,
) -> np.ndarray:
    """Exact near-field projection primitive with custom VJP on tangential currents."""

    return _fields_for_surface_exact_impl(currents_tangential, prepared)


def _fields_for_surface_exact_vjp(
    ans: np.ndarray,
    currents_tangential: np.ndarray,
    prepared: _ExactProjectionPrepared,
) -> Callable[[np.ndarray], np.ndarray]:
    """Return the custom VJP for the stacked exact-kernel current input."""

    return lambda g: _fields_for_surface_exact_vjp_impl(g, prepared)


defvjp(
    _fields_for_surface_exact_primitive,
    _fields_for_surface_exact_vjp,
    argnums=[0],
)


def _fields_for_surface_exact_batch_impl(
    currents_tangential: np.ndarray,
    prepared: _ExactProjectionPrepared,
) -> np.ndarray:
    """Exact near-field projection kernel for a batch of observation points."""

    electric_current, magnetic_current = _expand_tangential_surface_currents(
        currents_tangential, prepared
    )
    electric_current = [component[..., None] for component in electric_current]
    magnetic_current = [component[..., None] for component in magnetic_current]
    e_integrand, h_integrand = _exact_surface_integrands(
        electric_current, magnetic_current, prepared
    )
    return _stack_exact_surface_fields(
        *_integrate_exact_surface_fields(e_integrand, h_integrand, prepared),
        prepared,
        batch_axis_last=True,
    )


def _fields_for_surface_exact_batch_vjp_impl(
    g: np.ndarray, prepared: _ExactProjectionPrepared
) -> np.ndarray:
    """Exact near-field projection VJP for a batch of observation points."""

    g = anp.moveaxis(g, 0, -1)
    g_electric = _sph_2_car_field_components(g[0], g[1], g[2], prepared.theta_out, prepared.phi_out)
    g_magnetic = _sph_2_car_field_components(g[3], g[4], g[5], prepared.theta_out, prepared.phi_out)

    g_electric_integrand = [prepared.integration_weights * item for item in g_electric]
    g_magnetic_integrand = [prepared.integration_weights * item for item in g_magnetic]
    return _stack_exact_surface_current_gradients(
        *_exact_surface_adjoint_terms(g_electric_integrand, g_magnetic_integrand, prepared),
        prepared,
        sum_batch_axis=True,
    )


@primitive
def _fields_for_surface_exact_batch_primitive(
    currents_tangential: np.ndarray,
    prepared: _ExactProjectionPrepared,
) -> np.ndarray:
    """Exact near-field projection primitive for a batch of observation points."""

    return _fields_for_surface_exact_batch_impl(currents_tangential, prepared)


def _fields_for_surface_exact_batch_vjp(
    ans: np.ndarray,
    currents_tangential: np.ndarray,
    prepared: _ExactProjectionPrepared,
) -> Callable[[np.ndarray], np.ndarray]:
    """Return the custom VJP for the batched exact-kernel current input."""

    return lambda g: _fields_for_surface_exact_batch_vjp_impl(g, prepared)


defvjp(
    _fields_for_surface_exact_batch_primitive,
    _fields_for_surface_exact_batch_vjp,
    argnums=[0],
)


class _ExactFieldProjectionMixin:
    def _project_exact_fields_points(
        self,
        x: ArrayLikeN2F,
        y: ArrayLikeN2F,
        z: ArrayLikeN2F,
        surface_currents: list[tuple[FieldProjectionSurface, xr.Dataset]],
        medium: MediumType,
        *,
        verbose: bool,
        freq_chunk_size: int | None,
    ) -> NDArray:
        """Project exact fields for a flat list of observation points."""

        x = np.reshape(x, (-1,))
        y = np.reshape(y, (-1,))
        z = np.reshape(z, (-1,))
        frequencies = np.atleast_1d(self.frequencies)
        exact_surface_currents = [
            _ExactSurfaceCurrentData(
                surface=surface,
                currents=currents,
                currents_tangential=_prepare_exact_surface_currents(surface, currents),
            )
            for surface, currents in surface_currents
        ]
        freq_slices = _frequency_chunk_slices(frequencies, freq_chunk_size)

        num_batches = (x.size + EXACT_PROJECTION_BATCH_SIZE - 1) // EXACT_PROJECTION_BATCH_SIZE
        fields_by_freq_chunk = []
        for idx_chunk, freq_slice in enumerate(freq_slices, start=1):
            exact_surface_data = _prepare_exact_surface_projection_chunk(
                exact_surface_currents=exact_surface_currents,
                medium=medium,
                frequencies=frequencies,
                freq_slice=freq_slice,
            )
            description = "Computing projected fields"
            if len(freq_slices) > 1:
                description = (
                    f"Computing projected fields (freq chunk {idx_chunk}/{len(freq_slices)})"
                )
            chunk_fields = []
            for start in _track_if_verbose(
                range(0, x.size, EXACT_PROJECTION_BATCH_SIZE),
                verbose=verbose,
                description=description,
                total=num_batches,
            ):
                stop = min(start + EXACT_PROJECTION_BATCH_SIZE, x.size)
                chunk_fields.append(
                    self._project_exact_fields_batch(
                        x=x[start:stop],
                        y=y[start:stop],
                        z=z[start:stop],
                        exact_surface_data=exact_surface_data,
                    )
                )
            # Avoid concatenating a single batch: autograd would still allocate
            # and copy the full result even though there is nothing to join.
            fields_by_freq_chunk.append(
                chunk_fields[0] if len(chunk_fields) == 1 else anp.concatenate(chunk_fields, axis=0)
            )

        return (
            # Same idea for the outer frequency loop. The common pytest exact
            # benchmark has one frequency chunk, so this avoids one redundant
            # copy of the full projected field tensor.
            fields_by_freq_chunk[0]
            if len(fields_by_freq_chunk) == 1
            else anp.concatenate(fields_by_freq_chunk, axis=-1)
        )

    def _fields_for_surface_exact(
        self,
        x: float,
        y: float,
        z: float,
        surface: FieldProjectionSurface,
        currents: xr.Dataset,
        medium: MediumType,
    ) -> NDArray:
        """Compute exact projected fields from one source surface."""

        currents_tangential = _prepare_exact_surface_currents(surface, currents)
        prepared_static = _prepare_exact_surface_projection_static(
            surface=surface,
            currents=currents,
            medium=medium,
            frequencies=self.frequencies,
        )
        return self._fields_for_surface_exact_prepared(
            x=x,
            y=y,
            z=z,
            currents_tangential=currents_tangential,
            prepared_static=prepared_static,
        )

    def _fields_for_surface_exact_prepared(
        self,
        x: float,
        y: float,
        z: float,
        currents_tangential: np.ndarray,
        prepared_static: _ExactProjectionStatic,
    ) -> NDArray:
        """Evaluate exact projection from precomputed surface currents and static geometry."""

        prepared = _prepare_exact_surface_projection_point(x=x, y=y, z=z, prepared=prepared_static)

        if hasbox(prepared):
            return _fields_for_surface_exact_impl(currents_tangential, prepared)
        return _fields_for_surface_exact_primitive(currents_tangential, prepared)

    def _fields_for_surface_exact_batch_prepared(
        self,
        x: ArrayLikeN2F,
        y: ArrayLikeN2F,
        z: ArrayLikeN2F,
        currents_tangential: np.ndarray,
        prepared_static: _ExactProjectionStatic,
    ) -> NDArray:
        """Evaluate exact projection for a batch of observation points."""

        prepared = _prepare_exact_surface_projection_batch(x=x, y=y, z=z, prepared=prepared_static)

        if hasbox(prepared):
            return _fields_for_surface_exact_batch_impl(currents_tangential, prepared)
        return _fields_for_surface_exact_batch_primitive(currents_tangential, prepared)

    def _project_exact_fields_batch(
        self,
        x: ArrayLikeN2F,
        y: ArrayLikeN2F,
        z: ArrayLikeN2F,
        exact_surface_data: list[_PreparedExactSurfaceProjection],
    ) -> NDArray:
        """Project a batch of exact observation points for all surfaces."""

        x = np.atleast_1d(x)
        y = np.atleast_1d(y)
        z = np.atleast_1d(z)
        num_frequencies = exact_surface_data[0].projection.num_frequencies
        fields = anp.zeros((x.size, 6, num_frequencies), dtype=complex)
        for prepared_surface in exact_surface_data:
            fields = fields + self._fields_for_surface_exact_batch_prepared(
                x=x,
                y=y,
                z=z,
                currents_tangential=prepared_surface.currents_tangential,
                prepared_static=prepared_surface.projection,
            )
        return fields
