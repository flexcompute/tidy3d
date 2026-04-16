"""Angular approximate field projection path."""

from __future__ import annotations

from typing import TYPE_CHECKING

import autograd.numpy as anp
import numpy as np

from tidy3d.components.autograd.functions import add_at
from tidy3d.components.data.data_array import FieldProjectionAngleDataArray
from tidy3d.components.data.monitor_data import (
    AbstractFieldProjectionData,
    FieldProjectionAngleData,
)

from .common import (
    FIELD_COMPONENT_NAMES,
    PROJECTION_FREQ_CHUNK_SIZE,
    _far_field_integral,
    _prepare_far_field_projection,
    _projection_data_from_fields,
    _spherical_far_fields_from_projected_components,
)

if TYPE_CHECKING:
    import xarray as xr
    from numpy.typing import NDArray

    from tidy3d.components.medium import MediumType
    from tidy3d.components.monitor import FieldProjectionAngleMonitor, FieldProjectionSurface
    from tidy3d.components.types import Axis

    from .common import ArrayLikeN2F, _PreparedFarFieldProjection


class _ApproximateAngleProjectionMixin:
    def _far_fields_for_surface(
        self,
        frequency: float,
        theta: ArrayLikeN2F,
        phi: ArrayLikeN2F,
        surface: FieldProjectionSurface,
        currents: xr.Dataset,
        medium: MediumType,
    ) -> NDArray:
        """Compute far fields at angular observation points for one source surface."""

        prepared = _prepare_far_field_projection(
            frequency=frequency,
            surface=surface,
            currents=currents,
            medium=medium,
            is_2d_simulation=self.is_2d_simulation,
            simulation_size=self.effective_simulation_size,
        )
        return self._far_fields_from_prepared(
            theta=theta, phi=phi, prepared=prepared, surface_axis=surface.axis
        )

    def _far_fields_from_prepared(
        self,
        theta: ArrayLikeN2F,
        phi: ArrayLikeN2F,
        prepared: _PreparedFarFieldProjection,
        surface_axis: Axis,
    ) -> NDArray:
        """Evaluate far fields from pre-sliced current data."""

        theta = np.atleast_1d(theta)
        phi = np.atleast_1d(phi)

        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)

        pts = prepared.pts
        propagation_factor = prepared.propagation_factor
        phase_0 = np.exp(
            (propagation_factor * pts[0])[:, None, None]
            * sin_theta[None, :, None]
            * cos_phi[None, None, :]
        )
        phase_1 = np.exp(
            (propagation_factor * pts[1])[:, None, None]
            * sin_theta[None, :, None]
            * sin_phi[None, None, :]
        )
        phase_2 = np.exp((propagation_factor * pts[2])[:, None] * cos_theta[None, :])

        projected_components = []
        phases = (phase_0, phase_1, phase_2)
        for field_component in prepared.field_components:
            projected = _far_field_integral(field_component, phases, prepared.integral)
            projected_components.append(anp.reshape(projected, (len(theta), len(phi))))

        return _spherical_far_fields_from_projected_components(
            projected_components,
            idx_u=prepared.integral.idx_u,
            idx_v=prepared.integral.idx_v,
            surface_axis=surface_axis,
            sin_theta=sin_theta,
            cos_theta=cos_theta,
            sin_phi=sin_phi,
            cos_phi=cos_phi,
            eta=prepared.eta,
            paired_observations=False,
        )

    def _project_fields_angular(
        self,
        monitor: FieldProjectionAngleMonitor,
        verbose: bool = True,
        freq_chunk_size: int | None = PROJECTION_FREQ_CHUNK_SIZE,
    ) -> FieldProjectionAngleData:
        """Compute projected fields on an angle-based grid in spherical coordinates."""

        freqs = np.atleast_1d(self.frequencies)
        theta = np.atleast_1d(monitor.theta)
        phi = np.atleast_1d(monitor.phi)

        fields = np.zeros(
            (len(FIELD_COMPONENT_NAMES), 1, len(theta), len(phi), len(freqs)), dtype=complex
        )

        medium = monitor.medium if monitor.medium else self.medium
        wavenumber = AbstractFieldProjectionData.wavenumber(medium=medium, frequency=freqs)
        phase = np.atleast_1d(
            AbstractFieldProjectionData.propagation_factor(
                dist=monitor.proj_distance,
                k=wavenumber,
                is_2d_simulation=self.is_2d_simulation,
            )
        )

        surface_currents = self._windowed_surface_currents(monitor)

        if monitor.far_field_approx:
            for surface, currents in surface_currents:
                for idx_f, frequency in enumerate(freqs):
                    fields_surface = self._far_fields_for_surface(
                        frequency=frequency,
                        theta=theta,
                        phi=phi,
                        surface=surface,
                        currents=currents,
                        medium=medium,
                    )
                    fields = add_at(fields, [..., idx_f], fields_surface[:, None] * phase[idx_f])
        else:
            theta_grid, phi_grid = np.meshgrid(theta, phi, indexing="ij")
            flat_theta = np.reshape(theta_grid, (-1,))
            flat_phi = np.reshape(phi_grid, (-1,))
            flat_x, flat_y, flat_z = monitor.sph_2_car(monitor.proj_distance, flat_theta, flat_phi)
            stacked_fields = self._project_exact_fields_points(
                x=flat_x,
                y=flat_y,
                z=flat_z,
                surface_currents=surface_currents,
                medium=medium,
                verbose=verbose,
                freq_chunk_size=freq_chunk_size,
            )
            stacked_fields = anp.reshape(
                stacked_fields,
                (len(theta), len(phi), len(FIELD_COMPONENT_NAMES), len(freqs)),
            )
            fields = anp.moveaxis(stacked_fields, 2, 0)[:, None, :, :, :]

        coords = {"r": np.atleast_1d(monitor.proj_distance), "theta": theta, "phi": phi, "f": freqs}
        return _projection_data_from_fields(
            FieldProjectionAngleData,
            FieldProjectionAngleDataArray,
            monitor=monitor,
            projection_surfaces=self.surfaces,
            medium=medium,
            coords=coords,
            fields=fields,
            is_2d_simulation=self.is_2d_simulation,
        )
