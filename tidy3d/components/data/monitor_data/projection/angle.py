from __future__ import annotations

from math import isclose

import autograd.numpy as np
import xarray as xr
from pydantic import Field

from tidy3d.components.data.data_array import (
    FieldProjectionAngleDataArray,
    FluxDataArray,
)
from tidy3d.components.data.monitor_data._constants import MIN_ANGULAR_SAMPLES_SPHERE
from tidy3d.components.monitor import (
    FieldProjectionAngleMonitor,
    FieldProjectionSurface,
)
from tidy3d.exceptions import DataError

from .base import AbstractFieldProjectionData


class FieldProjectionAngleData(AbstractFieldProjectionData):
    """Data associated with a :class:`.FieldProjectionAngleMonitor`: components of projected fields.

    Example
    -------
    >>> from tidy3d import FieldProjectionAngleDataArray
    >>> f = np.linspace(1e14, 2e14, 10)
    >>> r = np.atleast_1d(5)
    >>> theta = np.linspace(0, np.pi, 10)
    >>> phi = np.linspace(0, 2*np.pi, 20)
    >>> coords = dict(r=r, theta=theta, phi=phi, f=f)
    >>> values = (1+1j) * np.random.random((len(r), len(theta), len(phi), len(f)))
    >>> scalar_field = FieldProjectionAngleDataArray(values, coords=coords)
    >>> monitor = FieldProjectionAngleMonitor(
    ...     center=(1,2,3), size=(2,2,2), freqs=f, name='n2f_monitor', phi=phi, theta=theta
    ...     )
    >>> data = FieldProjectionAngleData(
    ...     monitor=monitor, Er=scalar_field, Etheta=scalar_field, Ephi=scalar_field,
    ...     Hr=scalar_field, Htheta=scalar_field, Hphi=scalar_field,
    ...     projection_surfaces=monitor.projection_surfaces,
    ...     )
    """

    monitor: FieldProjectionAngleMonitor = Field(
        title="Projection monitor",
        description="Field projection monitor with an angle-based projection grid.",
    )

    projection_surfaces: tuple[FieldProjectionSurface, ...] = Field(
        title="Projection surfaces",
        description="Surfaces of the monitor where near fields were recorded for projection",
    )

    Er: FieldProjectionAngleDataArray = Field(
        title="Er",
        description="Spatial distribution of r-component of the electric field.",
    )
    Etheta: FieldProjectionAngleDataArray = Field(
        title="Etheta",
        description="Spatial distribution of the theta-component of the electric field.",
    )
    Ephi: FieldProjectionAngleDataArray = Field(
        title="Ephi",
        description="Spatial distribution of phi-component of the electric field.",
    )
    Hr: FieldProjectionAngleDataArray = Field(
        title="Hr",
        description="Spatial distribution of r-component of the magnetic field.",
    )
    Htheta: FieldProjectionAngleDataArray = Field(
        title="Htheta",
        description="Spatial distribution of theta-component of the magnetic field.",
    )
    Hphi: FieldProjectionAngleDataArray = Field(
        title="Hphi",
        description="Spatial distribution of phi-component of the magnetic field.",
    )

    @property
    def r(self) -> np.ndarray:
        """Radial distance."""
        return self.Etheta.r.values

    @property
    def theta(self) -> np.ndarray:
        """Polar angles."""
        return self.Etheta.theta.values

    @property
    def phi(self) -> np.ndarray:
        """Azimuthal angles."""
        return self.Etheta.phi.values

    def renormalize_fields(self, proj_distance: float) -> FieldProjectionAngleData:
        """Return a :class:`.FieldProjectionAngleData` with fields re-normalized to a new
        projection distance, by applying a phase factor based on ``proj_distance``.

        Parameters
        ----------
        proj_distance : float = None
            (micron) new radial distance relative to the monitor's local origin.

        Returns
        -------
        :class:`.FieldProjectionAngleData`
            Copy of this :class:`.FieldProjectionAngleData` with fields re-projected
            to ``proj_distance``.
        """
        if self.monitor and not self.monitor.far_field_approx:
            raise DataError(
                "Fields projected without invoking the far field approximation "
                "cannot be re-projected to a new distance."
            )

        # the phase factor associated with the old distance must be removed
        r = self.coords_spherical["r"][..., None]
        old_phase = self.propagation_factor(
            dist=r, k=self.k[None, None, None, :], is_2d_simulation=self.is_2d_simulation
        )

        # the phase factor associated with the new distance must be applied
        new_phase = self.propagation_factor(
            dist=proj_distance, k=self.k, is_2d_simulation=self.is_2d_simulation
        )

        # net phase
        phase = new_phase[None, None, None, :] / old_phase

        # compute updated fields and their coordinates
        return self.make_renormalized_data(phase, proj_distance)

    @property
    def tangential_dims(self) -> list[str]:
        """Tangential dimensions to a spherical surface in the spherical coordinate system."""
        tangential_dims = ["theta", "phi"]
        return tangential_dims

    @staticmethod
    def _check_coords_sorted(coord: np.ndarray, name: str) -> None:
        """Helper for checking whether an array is sorted and raises an exception if it is not."""
        is_sorted = np.all(np.diff(coord) >= 0)
        if not is_sorted:
            raise ValueError(f"{name} was not provided as a sorted array.")

    def _check_integration_suitability(self) -> None:
        """Checks whether the sampling of ``theta`` and ``phi`` is suitable for
        integrating over a spherical surface."""
        if (
            len(self.theta) < MIN_ANGULAR_SAMPLES_SPHERE
            or len(self.phi) < 2 * MIN_ANGULAR_SAMPLES_SPHERE
        ):
            raise ValueError(
                "There are not enough sampling points along `theta` or `phi` for accurate integration. "
                f"Currently, {len(self.theta)} samples for `theta` and {len(self.phi)} samples for `phi`. "
                f"Consider using, at the very least, {MIN_ANGULAR_SAMPLES_SPHERE} samples for `theta` and "
                f"{2 * MIN_ANGULAR_SAMPLES_SPHERE} samples for `phi`."
            )
        self._check_coords_sorted(self.theta, "theta")
        self._check_coords_sorted(self.phi, "phi")
        if not isclose(self.theta[0], 0) or not isclose(self.theta[-1], np.pi):
            raise ValueError(
                "Chosen limits for `theta` are not appropriate for integration. "
                "`theta` must range from 0 to π."
            )
        if not isclose(self.phi[0], 0) or not isclose(self.phi[-1], 2 * np.pi):
            raise ValueError(
                "Chosen limits for `phi` are not appropriate for integration. "
                "`phi` must range from 0 to 2π."
            )

    def flux_from_projected_fields(self) -> FluxDataArray:
        """Flux calculated by integrating the projected fields on a spherical surface.

        Returns
        -------
        :class:`.FluxDataArray`
            Flux in the frequency domain.
        """
        self._check_integration_suitability()
        d_solid_angle = np.sin(self.Etheta.theta)
        integrand = (self.power * d_solid_angle).sel(r=self.monitor.proj_distance)
        flux = self.monitor.proj_distance**2 * integrand.integrate(self.tangential_dims)
        return FluxDataArray(flux)

    @staticmethod
    def get_phi_slice(
        field_array: FieldProjectionAngleDataArray, phi: float, symmetric: bool = False
    ) -> FieldProjectionAngleDataArray:
        """Get a planar slice of the :class:`.FieldProjectionAngleDataArray` along a given phi angle.
        Extends theta range from [0, π] to [0, 2π] to create a full slice.

        Parameters
        ----------
        field_array : :class:`.FieldProjectionAngleDataArray`
            Field array to slice.
        phi : float
            Angle phi in radians to slice at.
        symmetric : bool = False
            If True, uses same data for both halves. If False, takes opposite phi angle
            for back half.

        Returns
        -------
        :class:`.FieldProjectionAngleDataArray`
            2D slice with theta going from 0 to 2π.
        """
        slice_phi = field_array.sel(phi=phi, method="nearest")
        slice_phi = slice_phi.where(slice_phi.theta < np.pi)
        if symmetric:
            slice_opposite_phi = field_array.sel(phi=phi, method="nearest")
        else:
            slice_opposite_phi = field_array.sel(phi=phi + np.pi, method="nearest")
        slice_opposite_phi = slice_opposite_phi.where(slice_opposite_phi.theta > 0)
        slice_opposite_phi = slice_opposite_phi.assign_coords(
            theta=(2 * np.pi - slice_opposite_phi.theta)
        )
        data_array = xr.concat(
            (slice_phi, slice_opposite_phi), dim="theta", coords="minimal", compat="override"
        ).sortby("theta")
        return FieldProjectionAngleDataArray(data_array)
