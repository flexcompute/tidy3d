"""Exact field projection paths and kernels."""

from __future__ import annotations

from typing import TYPE_CHECKING

import autograd.numpy as anp
import numpy as np

from tidy3d.components.data.monitor_data import AbstractFieldProjectionData
from tidy3d.constants import EPSILON_0, MU_0

from .common import FIELD_COMPONENT_NAMES, _track_if_verbose

if TYPE_CHECKING:
    import xarray as xr
    from numpy.typing import NDArray

    from tidy3d.components.medium import MediumType
    from tidy3d.components.monitor import FieldProjectionSurface

    from .common import ArrayLikeN2F


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
    ) -> NDArray:
        """Project exact fields for a flat list of observation points."""

        x = np.reshape(x, (-1,))
        y = np.reshape(y, (-1,))
        z = np.reshape(z, (-1,))

        field_shape = (len(FIELD_COMPONENT_NAMES), len(self.frequencies))
        point_fields = []
        for x_obs, y_obs, z_obs in _track_if_verbose(
            zip(x, y, z),
            verbose=verbose,
            description="Computing projected fields",
            total=x.size,
        ):
            fields_sum = anp.zeros(field_shape, dtype=complex)
            for surface, currents in surface_currents:
                fields_surface = self._fields_for_surface_exact(
                    x=x_obs,
                    y=y_obs,
                    z=z_obs,
                    surface=surface,
                    currents=currents,
                    medium=medium,
                )
                fields_surface = anp.reshape(fields_surface, field_shape)
                fields_sum = fields_sum + fields_surface
            point_fields.append(fields_sum)

        return anp.stack(point_fields, axis=0)

    def _fields_for_surface_exact(
        self,
        x: float,
        y: float,
        z: float,
        surface: FieldProjectionSurface,
        currents: xr.Dataset,
        medium: MediumType,
    ) -> NDArray:
        """Compute projected fields in spherical coordinates at a given projection point on a
        Cartesian grid for a given set of surface currents using the exact homogeneous medium
        Green's function without geometric approximations.

        Parameters
        ----------
        x : float
            Observation point x-coordinate (microns) relative to the local origin.
        y : float
            Observation point y-coordinate (microns) relative to the local origin.
        z : float
            Observation point z-coordinate (microns) relative to the local origin.
        surface: :class:`FieldProjectionSurface`
            :class:`FieldProjectionSurface` object to use as source of near field.
        currents : xarray.Dataset
            xarray Dataset containing surface currents associated with the surface monitor.
        medium : :class:`.MediumType`
            Background medium through which to project fields.

        Returns
        -------
        np.ndarray
            With leading dimension containing ``Er``, ``Etheta``, ``Ephi``, ``Hr``, ``Htheta``, ``Hphi``
            projected fields for each frequency.
        """
        freqs = anp.array(self.frequencies)
        i_omega = 1j * 2.0 * np.pi * freqs[None, None, None, :]
        wavenumber = AbstractFieldProjectionData.wavenumber(frequency=freqs, medium=medium)
        wavenumber = wavenumber[None, None, None, :]  # add space dimensions

        eps_complex = medium.eps_model(frequency=freqs)
        epsilon = EPSILON_0 * eps_complex[None, None, None, :]

        # source points
        pts = [currents[name].values for name in ["x", "y", "z"]]

        # transform the coordinate system so that the origin is at the source point
        # then the observation points in the new system are:
        x_new, y_new, z_new = (pt_obs - pt_src for pt_src, pt_obs in zip(pts, [x, y, z]))

        # tangential source components to use
        idx_w, idx_uv = surface.monitor.pop_axis((0, 1, 2), axis=surface.axis)
        _, source_names = surface.monitor.pop_axis(("x", "y", "z"), axis=surface.axis)

        idx_u, idx_v = idx_uv
        cmp_1, cmp_2 = source_names

        # set the surface current density Cartesian components
        J = [None] * 3
        M = [None] * 3
        J[idx_u] = currents[f"E{cmp_1}"].data
        J[idx_v] = currents[f"E{cmp_2}"].data
        J[idx_w] = anp.zeros(J[idx_u].shape)
        M[idx_u] = currents[f"H{cmp_1}"].data
        M[idx_v] = currents[f"H{cmp_2}"].data
        M[idx_w] = anp.zeros(M[idx_u].shape)

        # observation point in the new spherical system
        r, theta_obs, phi_obs = surface.monitor.car_2_sph(
            x_new[:, None, None, None], y_new[None, :, None, None], z_new[None, None, :, None]
        )

        # angle terms
        sin_theta = anp.sin(theta_obs)
        cos_theta = anp.cos(theta_obs)
        sin_phi = anp.sin(phi_obs)
        cos_phi = anp.cos(phi_obs)

        # Green's function and terms related to its derivatives
        ikr = 1j * wavenumber * r
        G = anp.exp(ikr) / (4.0 * np.pi * r)
        dG_dr = G * (ikr - 1.0) / r
        d2G_dr2 = dG_dr * (ikr - 1.0) / r + G / (r**2)

        # operations between unit vectors and currents
        def r_x_current(current: tuple[NDArray, ...]) -> tuple[NDArray, ...]:
            """Cross product between the r unit vector and the current."""
            return [
                sin_theta * sin_phi * current[2] - cos_theta * current[1],
                cos_theta * current[0] - sin_theta * cos_phi * current[2],
                sin_theta * cos_phi * current[1] - sin_theta * sin_phi * current[0],
            ]

        def r_dot_current(current: tuple[NDArray, ...]) -> NDArray:
            """Dot product between the r unit vector and the current."""
            return (
                sin_theta * cos_phi * current[0]
                + sin_theta * sin_phi * current[1]
                + cos_theta * current[2]
            )

        def r_dot_current_dtheta(current: tuple[NDArray, ...]) -> NDArray:
            """Theta derivative of the dot product between the r unit vector and the current."""
            return (
                cos_theta * cos_phi * current[0]
                + cos_theta * sin_phi * current[1]
                - sin_theta * current[2]
            )

        def r_dot_current_dphi_div_sin_theta(current: tuple[NDArray, ...]) -> NDArray:
            """Phi derivative of the dot product between the r unit vector and the current,
            analytically divided by sin theta."""
            return -sin_phi * current[0] + cos_phi * current[1]

        def grad_Gr_r_dot_current(current: tuple[NDArray, ...]) -> tuple[NDArray, ...]:
            """Gradient of the product of the gradient of the Green's function and the dot product
            between the r unit vector and the current."""
            temp = [
                d2G_dr2 * r_dot_current(current),
                dG_dr * r_dot_current_dtheta(current) / r,
                dG_dr * r_dot_current_dphi_div_sin_theta(current) / r,
            ]
            # convert to Cartesian coordinates
            return surface.monitor.sph_2_car_field(temp[0], temp[1], temp[2], theta_obs, phi_obs)

        def potential_terms(
            current: tuple[NDArray, ...], const: complex
        ) -> tuple[list[complex], list[complex], list[complex]]:
            """Assemble vector potential and its derivatives."""
            r_x_c = r_x_current(current)
            pot = [const * item * G for item in current]
            curl_pot = [const * item * dG_dr for item in r_x_c]
            grad_div_pot = grad_Gr_r_dot_current(current)
            grad_div_pot = [const * item for item in grad_div_pot]
            return pot, curl_pot, grad_div_pot

        # magnetic vector potential terms
        A, curl_A, grad_div_A = potential_terms(J, MU_0)

        # electric vector potential terms
        F, curl_F, grad_div_F = potential_terms(M, epsilon)

        # assemble the electric field components (Taflove 8.24, 8.27)
        e_x_integrand, e_y_integrand, e_z_integrand = (
            i_omega * (a + grad_div_a / (wavenumber**2)) - curl_f / epsilon
            for a, grad_div_a, curl_f in zip(A, grad_div_A, curl_F)
        )

        # assemble the magnetic field components (Taflove 8.25, 8.28)
        h_x_integrand, h_y_integrand, h_z_integrand = (
            i_omega * (f + grad_div_f / (wavenumber**2)) + curl_a / MU_0
            for f, grad_div_f, curl_a in zip(F, grad_div_F, curl_A)
        )

        # integrate over the surface
        e_x = self.trapezoid(e_x_integrand, (pts[idx_u], pts[idx_v]), (idx_u, idx_v))
        e_y = self.trapezoid(e_y_integrand, (pts[idx_u], pts[idx_v]), (idx_u, idx_v))
        e_z = self.trapezoid(e_z_integrand, (pts[idx_u], pts[idx_v]), (idx_u, idx_v))
        h_x = self.trapezoid(h_x_integrand, (pts[idx_u], pts[idx_v]), (idx_u, idx_v))
        h_y = self.trapezoid(h_y_integrand, (pts[idx_u], pts[idx_v]), (idx_u, idx_v))
        h_z = self.trapezoid(h_z_integrand, (pts[idx_u], pts[idx_v]), (idx_u, idx_v))

        # observation point in the original spherical system
        _, theta_obs, phi_obs = surface.monitor.car_2_sph(x, y, z)

        # convert fields to the original spherical system
        e_r, e_theta, e_phi = surface.monitor.car_2_sph_field(e_x, e_y, e_z, theta_obs, phi_obs)
        h_r, h_theta, h_phi = surface.monitor.car_2_sph_field(h_x, h_y, h_z, theta_obs, phi_obs)

        return anp.array([e_r, e_theta, e_phi, h_r, h_theta, h_phi])
