from __future__ import annotations

from typing import TYPE_CHECKING

import autograd.numpy as np
from flexcompute.core._migration.em.numerical.raw import field_data as field_data_numerics
from pydantic import Field

from tidy3d.components.base import cached_property
from tidy3d.components.data.data_array import FieldProjectionCartesianDataArray, FluxDataArray
from tidy3d.components.monitor import (
    FieldProjectionCartesianMonitor,
    FieldProjectionSurface,
)
from tidy3d.exceptions import DataError

from .base import AbstractFieldProjectionData

if TYPE_CHECKING:
    from tidy3d.components.data.data_array import ScalarFieldDataArray


class FieldProjectionCartesianData(AbstractFieldProjectionData):
    """Data associated with a :class:`.FieldProjectionCartesianMonitor`: components of
    projected fields.

    Example
    -------
    >>> from tidy3d import FieldProjectionCartesianDataArray
    >>> f = np.linspace(1e14, 2e14, 10)
    >>> x = np.linspace(0, 5, 10)
    >>> y = np.linspace(0, 10, 20)
    >>> z = np.atleast_1d(5)
    >>> coords = dict(x=x, y=y, z=z, f=f)
    >>> values = (1+1j) * np.random.random((len(x), len(y), len(z), len(f)))
    >>> scalar_field = FieldProjectionCartesianDataArray(values, coords=coords)
    >>> monitor = FieldProjectionCartesianMonitor(
    ...     center=(1,2,3), size=(2,2,2), freqs=f, name='n2f_monitor', x=x, y=y,
    ...     proj_axis=2, proj_distance=50
    ...     )
    >>> data = FieldProjectionCartesianData(
    ...     monitor=monitor, Er=scalar_field, Etheta=scalar_field, Ephi=scalar_field,
    ...     Hr=scalar_field, Htheta=scalar_field, Hphi=scalar_field,
    ...     projection_surfaces=monitor.projection_surfaces,
    ...     )
    """

    monitor: FieldProjectionCartesianMonitor = Field(
        title="Projection monitor",
        description="Field projection monitor with a Cartesian projection grid.",
    )

    projection_surfaces: tuple[FieldProjectionSurface, ...] = Field(
        title="Projection surfaces",
        description="Surfaces of the monitor where near fields were recorded for projection",
    )

    Er: FieldProjectionCartesianDataArray = Field(
        title="Er",
        description="Spatial distribution of r-component of the electric field.",
    )
    Etheta: FieldProjectionCartesianDataArray = Field(
        title="Etheta",
        description="Spatial distribution of the theta-component of the electric field.",
    )
    Ephi: FieldProjectionCartesianDataArray = Field(
        title="Ephi",
        description="Spatial distribution of phi-component of the electric field.",
    )
    Hr: FieldProjectionCartesianDataArray = Field(
        title="Hr",
        description="Spatial distribution of r-component of the magnetic field.",
    )
    Htheta: FieldProjectionCartesianDataArray = Field(
        title="Htheta",
        description="Spatial distribution of theta-component of the magnetic field.",
    )
    Hphi: FieldProjectionCartesianDataArray = Field(
        title="Hphi",
        description="Spatial distribution of phi-component of the magnetic field.",
    )

    @property
    def x(self) -> np.ndarray:
        """X positions."""
        return self.Etheta.x.values

    @property
    def y(self) -> np.ndarray:
        """Y positions."""
        return self.Etheta.y.values

    @property
    def z(self) -> np.ndarray:
        """Z positions."""
        return self.Etheta.z.values

    @property
    def tangential_dims(self) -> list[str]:
        tangential_dims = ["x", "y", "z"]
        tangential_dims.pop(self.monitor.proj_axis)
        return tangential_dims

    @property
    def poynting(self) -> ScalarFieldDataArray:
        """Time-averaged Poynting vector for field data associated to a Cartesian field projection monitor."""
        fc = self.fields_cartesian
        dim1, dim2 = self.tangential_dims

        e1 = fc["E" + dim1]
        e2 = fc["E" + dim2]
        h1 = fc["H" + dim1]
        h2 = fc["H" + dim2]

        return np.real(field_data_numerics.complex_poynting(e1, e2, h1, h2))

    @cached_property
    def flux(self) -> FluxDataArray:
        """Flux for projected field data corresponding to a Cartesian field projection monitor."""
        flux = self.poynting.integrate(self.tangential_dims)
        return FluxDataArray(flux)

    def renormalize_fields(self, proj_distance: float) -> FieldProjectionCartesianData:
        """Return a :class:`.FieldProjectionCartesianData` with fields re-normalized to a new
        projection distance, by applying a phase factor based on ``proj_distance``.

        Parameters
        ----------
        proj_distance : float = None
            (micron) new plane distance relative to the monitor's local origin.

        Returns
        -------
        :class:`.FieldProjectionCartesianData`
            Copy of this :class:`.FieldProjectionCartesianData` with fields re-projected
            to ``proj_distance``.
        """
        if not self.monitor.far_field_approx:
            raise DataError(
                "Fields projected without invoking the far field approximation "
                "cannot be re-projected to a new distance."
            )

        # the phase factor associated with the old distance must be removed
        k = self.k[None, None, None, :]
        r = self.coords_spherical["r"][..., None]
        old_phase = self.propagation_factor(dist=r, k=k, is_2d_simulation=self.is_2d_simulation)

        # update the field components' projection distance
        norm_dir, _ = self.monitor.pop_axis(["x", "y", "z"], axis=self.monitor.proj_axis)
        for field in self.field_components.values():
            field[norm_dir] = np.atleast_1d(proj_distance)

        # the phase factor associated with the new distance must be applied
        r = self.coords_spherical["r"][..., None]
        new_phase = self.propagation_factor(dist=r, k=k, is_2d_simulation=self.is_2d_simulation)

        # net phase
        phase = new_phase / old_phase

        # compute updated fields and their coordinates
        return self.make_renormalized_data(phase, proj_distance)
