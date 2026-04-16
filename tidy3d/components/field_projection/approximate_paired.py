"""Paired-point approximate field projection paths for Cartesian and k-space outputs."""

from __future__ import annotations

from typing import TYPE_CHECKING

import autograd.numpy as anp
import numpy as np

from tidy3d.components.data.data_array import (
    FieldProjectionCartesianDataArray,
    FieldProjectionKSpaceDataArray,
)
from tidy3d.components.data.monitor_data import (
    AbstractFieldProjectionData,
    FieldProjectionCartesianData,
    FieldProjectionKSpaceData,
)

from .common import (
    APPROX_PROJECTION_BATCH_SIZE,
    FIELD_COMPONENT_NAMES,
    PROJECTION_FREQ_CHUNK_SIZE,
    _far_field_integral_pairs,
    _frequency_chunk_slices,
    _prepare_far_field_projection,
    _projection_data_from_fields,
    _spherical_far_fields_from_projected_components,
    _track_if_verbose,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    import xarray as xr
    from numpy.typing import NDArray

    from tidy3d.components.medium import MediumType
    from tidy3d.components.monitor import (
        FieldProjectionCartesianMonitor,
        FieldProjectionKSpaceMonitor,
        FieldProjectionSurface,
    )
    from tidy3d.components.types import Axis

    from .common import ArrayLikeN2F, _PreparedFarFieldProjection


class _ApproximatePairedProjectionMixin:
    def _far_fields_from_prepared_pairs(
        self,
        theta: ArrayLikeN2F,
        phi: ArrayLikeN2F,
        prepared: _PreparedFarFieldProjection,
        surface_axis: Axis,
    ) -> NDArray:
        """Evaluate far fields for paired observation angles."""

        theta = np.reshape(theta, (-1,))
        phi = np.reshape(phi, (-1,))

        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)

        pts = prepared.pts
        propagation_factor = prepared.propagation_factor
        phase_0 = np.exp(
            (propagation_factor * pts[0])[:, None] * sin_theta[None, :] * cos_phi[None, :]
        )
        phase_1 = np.exp(
            (propagation_factor * pts[1])[:, None] * sin_theta[None, :] * sin_phi[None, :]
        )
        phase_2 = np.exp((propagation_factor * pts[2])[:, None] * cos_theta[None, :])

        projected_components = []
        phases = (phase_0, phase_1, phase_2)
        for field_component in prepared.field_components:
            projected = _far_field_integral_pairs(field_component, phases, prepared.integral)
            projected_components.append(anp.reshape(projected, theta.shape))

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
            paired_observations=True,
        )

    def _project_prepared_fields_pairs(
        self,
        theta: np.ndarray,
        phi: np.ndarray,
        prepared_surface_currents: list[
            tuple[FieldProjectionSurface, list[_PreparedFarFieldProjection]]
        ],
        phase_by_freq: np.ndarray,
    ) -> NDArray:
        """Project approximate fields for paired observation points."""

        num_points = theta.size
        phase_by_freq = anp.asarray(phase_by_freq)

        fields_by_freq = []
        for idx_f in range(phase_by_freq.shape[0]):
            fields_sum = anp.zeros((6, num_points), dtype=complex)
            for surface, prepared_by_freq in prepared_surface_currents:
                fields_surface = self._far_fields_from_prepared_pairs(
                    theta=theta,
                    phi=phi,
                    prepared=prepared_by_freq[idx_f],
                    surface_axis=surface.axis,
                )
                fields_sum = fields_sum + fields_surface * phase_by_freq[idx_f][None, :]
            fields_by_freq.append(fields_sum)

        return anp.moveaxis(anp.stack(fields_by_freq, axis=-1), 1, 0)

    def _prepare_far_field_surface_currents(
        self,
        surface_currents: list[tuple[FieldProjectionSurface, xr.Dataset]],
        freqs: np.ndarray,
        medium: MediumType,
    ) -> list[tuple[FieldProjectionSurface, list[_PreparedFarFieldProjection]]]:
        """Prepare reusable per-frequency far-field data for each surface."""

        return [
            (
                surface,
                [
                    _prepare_far_field_projection(
                        frequency=frequency,
                        surface=surface,
                        currents=currents,
                        medium=medium,
                        is_2d_simulation=self.is_2d_simulation,
                        simulation_size=self.effective_simulation_size,
                    )
                    for frequency in freqs
                ],
            )
            for surface, currents in surface_currents
        ]

    def _project_approximate_paired_fields(
        self,
        *,
        theta_obs: np.ndarray,
        phi_obs: np.ndarray,
        freqs: np.ndarray,
        surface_currents: list[tuple[FieldProjectionSurface, xr.Dataset]],
        medium: MediumType,
        phase_by_freq_chunk: Callable[[slice, int, int], np.ndarray],
        verbose: bool,
        freq_chunk_size: int | None,
    ) -> NDArray:
        """Project approximate fields for paired-output monitors."""

        total_points = theta_obs.size
        num_batches = (
            total_points + APPROX_PROJECTION_BATCH_SIZE - 1
        ) // APPROX_PROJECTION_BATCH_SIZE
        freq_slices = _frequency_chunk_slices(freqs, freq_chunk_size)
        fields_by_freq_chunk = []

        for idx_chunk, freq_slice in enumerate(freq_slices, start=1):
            prepared_surface_currents = self._prepare_far_field_surface_currents(
                surface_currents=surface_currents,
                freqs=freqs[freq_slice],
                medium=medium,
            )
            description = "Computing projected fields"
            if len(freq_slices) > 1:
                description = (
                    f"Computing projected fields (freq chunk {idx_chunk}/{len(freq_slices)})"
                )

            chunk_fields = []
            for start in _track_if_verbose(
                range(0, total_points, APPROX_PROJECTION_BATCH_SIZE),
                verbose=verbose,
                description=description,
                total=num_batches,
            ):
                stop = min(start + APPROX_PROJECTION_BATCH_SIZE, total_points)
                chunk_fields.append(
                    self._project_prepared_fields_pairs(
                        theta=theta_obs[start:stop],
                        phi=phi_obs[start:stop],
                        prepared_surface_currents=prepared_surface_currents,
                        phase_by_freq=phase_by_freq_chunk(freq_slice, start, stop),
                    )
                )

            fields_by_freq_chunk.append(anp.concatenate(chunk_fields, axis=0))

        return anp.concatenate(fields_by_freq_chunk, axis=-1)

    def _project_fields_cartesian(
        self,
        monitor: FieldProjectionCartesianMonitor,
        verbose: bool = True,
        freq_chunk_size: int | None = PROJECTION_FREQ_CHUNK_SIZE,
    ) -> FieldProjectionCartesianData:
        """Compute projected fields on a Cartesian grid in spherical coordinates."""

        freqs = np.atleast_1d(self.frequencies)
        x, y, z = monitor.unpop_axis(
            monitor.proj_distance, (monitor.x, monitor.y), axis=monitor.proj_axis
        )
        x, y, z = list(map(np.atleast_1d, [x, y, z]))

        medium = monitor.medium if monitor.medium else self.medium
        surface_currents = self._windowed_surface_currents(monitor)

        if monitor.far_field_approx:
            x_grid, y_grid, z_grid = np.meshgrid(x, y, z, indexing="ij")
            flat_x = np.reshape(x_grid, (-1,))
            flat_y = np.reshape(y_grid, (-1,))
            flat_z = np.reshape(z_grid, (-1,))
            r_obs, theta_obs, phi_obs = monitor.car_2_sph(flat_x, flat_y, flat_z)
            wavenumber = AbstractFieldProjectionData.wavenumber(medium=medium, frequency=freqs)

            stacked_fields = self._project_approximate_paired_fields(
                theta_obs=theta_obs,
                phi_obs=phi_obs,
                freqs=freqs,
                surface_currents=surface_currents,
                medium=medium,
                phase_by_freq_chunk=lambda freq_slice,
                start,
                stop: AbstractFieldProjectionData.propagation_factor(
                    dist=r_obs[None, start:stop],
                    k=wavenumber[freq_slice, None],
                    is_2d_simulation=self.is_2d_simulation,
                ),
                verbose=verbose,
                freq_chunk_size=freq_chunk_size,
            )
            stacked_fields = anp.reshape(
                stacked_fields, (len(x), len(y), len(z), len(FIELD_COMPONENT_NAMES), len(freqs))
            )
            fields = anp.moveaxis(stacked_fields, 3, 0)

            return _projection_data_from_fields(
                FieldProjectionCartesianData,
                FieldProjectionCartesianDataArray,
                monitor=monitor,
                projection_surfaces=self.surfaces,
                medium=medium,
                coords={"x": x, "y": y, "z": z, "f": freqs},
                fields=fields,
            )

        x_grid, y_grid, z_grid = np.meshgrid(x, y, z, indexing="ij")
        stacked_fields = self._project_exact_fields_points(
            x=np.reshape(x_grid, (-1,)),
            y=np.reshape(y_grid, (-1,)),
            z=np.reshape(z_grid, (-1,)),
            surface_currents=surface_currents,
            medium=medium,
            verbose=verbose,
            freq_chunk_size=freq_chunk_size,
        )
        stacked_fields = anp.reshape(
            stacked_fields, (len(x), len(y), len(z), len(FIELD_COMPONENT_NAMES), len(freqs))
        )
        fields = anp.moveaxis(stacked_fields, 3, 0)

        return _projection_data_from_fields(
            FieldProjectionCartesianData,
            FieldProjectionCartesianDataArray,
            monitor=monitor,
            projection_surfaces=self.surfaces,
            medium=medium,
            coords={"x": x, "y": y, "z": z, "f": freqs},
            fields=fields,
        )

    def _project_fields_kspace(
        self,
        monitor: FieldProjectionKSpaceMonitor,
        verbose: bool = True,
        freq_chunk_size: int | None = PROJECTION_FREQ_CHUNK_SIZE,
    ) -> FieldProjectionKSpaceData:
        """Compute projected fields on a k-space grid in spherical coordinates."""

        freqs = np.atleast_1d(self.frequencies)
        ux = np.atleast_1d(monitor.ux)
        uy = np.atleast_1d(monitor.uy)

        medium = monitor.medium if monitor.medium else self.medium
        surface_currents = self._windowed_surface_currents(monitor)

        if monitor.far_field_approx:
            ux_grid, uy_grid = np.meshgrid(ux, uy, indexing="ij")
            theta_obs, phi_obs = monitor.kspace_2_sph(
                np.reshape(ux_grid, (-1,)),
                np.reshape(uy_grid, (-1,)),
                monitor.proj_axis,
            )
            phase = np.atleast_1d(
                AbstractFieldProjectionData.propagation_factor(
                    dist=monitor.proj_distance,
                    k=AbstractFieldProjectionData.wavenumber(medium=medium, frequency=freqs),
                    is_2d_simulation=self.is_2d_simulation,
                )
            )[:, None]

            stacked_fields = self._project_approximate_paired_fields(
                theta_obs=theta_obs,
                phi_obs=phi_obs,
                freqs=freqs,
                surface_currents=surface_currents,
                medium=medium,
                phase_by_freq_chunk=lambda freq_slice, start, stop: phase[freq_slice],
                verbose=verbose,
                freq_chunk_size=freq_chunk_size,
            )
            stacked_fields = anp.reshape(
                stacked_fields, (len(ux), len(uy), len(FIELD_COMPONENT_NAMES), len(freqs))
            )
            fields = anp.moveaxis(stacked_fields, 2, 0)[:, :, :, None, :]

            return _projection_data_from_fields(
                FieldProjectionKSpaceData,
                FieldProjectionKSpaceDataArray,
                monitor=monitor,
                projection_surfaces=self.surfaces,
                medium=medium,
                coords={
                    "ux": np.array(monitor.ux),
                    "uy": np.array(monitor.uy),
                    "r": np.atleast_1d(monitor.proj_distance),
                    "f": freqs,
                },
                fields=fields,
            )

        ux_grid, uy_grid = np.meshgrid(ux, uy, indexing="ij")
        theta_obs, phi_obs = monitor.kspace_2_sph(
            np.reshape(ux_grid, (-1,)),
            np.reshape(uy_grid, (-1,)),
            monitor.proj_axis,
        )
        x_obs, y_obs, z_obs = monitor.sph_2_car(monitor.proj_distance, theta_obs, phi_obs)
        stacked_fields = self._project_exact_fields_points(
            x=x_obs,
            y=y_obs,
            z=z_obs,
            surface_currents=surface_currents,
            medium=medium,
            verbose=verbose,
            freq_chunk_size=freq_chunk_size,
        )
        stacked_fields = anp.reshape(
            stacked_fields, (len(ux), len(uy), len(FIELD_COMPONENT_NAMES), len(freqs))
        )
        fields = anp.moveaxis(stacked_fields, 2, 0)[:, :, :, None, :]

        return _projection_data_from_fields(
            FieldProjectionKSpaceData,
            FieldProjectionKSpaceDataArray,
            monitor=monitor,
            projection_surfaces=self.surfaces,
            medium=medium,
            coords={
                "ux": np.array(monitor.ux),
                "uy": np.array(monitor.uy),
                "r": np.atleast_1d(monitor.proj_distance),
                "f": freqs,
            },
            fields=fields,
        )
