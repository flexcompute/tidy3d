from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import Field

from tidy3d.components.data.data_array import FieldProjectionKSpaceDataArray
from tidy3d.components.monitor import (
    FieldProjectionKSpaceMonitor,
    FieldProjectionSurface,
)
from tidy3d.exceptions import DataError

from .base import AbstractFieldProjectionData

if TYPE_CHECKING:
    import autograd.numpy as np


class FieldProjectionKSpaceData(AbstractFieldProjectionData):
    """Data associated with a :class:`.FieldProjectionKSpaceMonitor`: components of
    projected fields.

    Example
    -------
    >>> from tidy3d import FieldProjectionKSpaceDataArray
    >>> import numpy as np
    >>> f = np.linspace(1e14, 2e14, 10)
    >>> ux = np.linspace(0, 0.4, 10)
    >>> uy = np.linspace(0, 0.6, 20)
    >>> r = np.atleast_1d(5)
    >>> coords = dict(ux=ux, uy=uy, r=r, f=f)
    >>> values = (1+1j) * np.random.random((len(ux), len(uy), len(r), len(f)))
    >>> scalar_field = FieldProjectionKSpaceDataArray(values, coords=coords)
    >>> monitor = FieldProjectionKSpaceMonitor(
    ...     center=(1,2,3), size=(2,2,2), freqs=f, name='n2f_monitor', ux=ux, uy=uy, proj_axis=2
    ...     )
    >>> data = FieldProjectionKSpaceData(
    ...     monitor=monitor, Er=scalar_field, Etheta=scalar_field, Ephi=scalar_field,
    ...     Hr=scalar_field, Htheta=scalar_field, Hphi=scalar_field,
    ...     projection_surfaces=monitor.projection_surfaces,
    ...     )
    """

    monitor: FieldProjectionKSpaceMonitor = Field(
        title="Projection monitor",
        description="Field projection monitor with a projection grid defined in k-space.",
    )

    projection_surfaces: tuple[FieldProjectionSurface, ...] = Field(
        title="Projection surfaces",
        description="Surfaces of the monitor where near fields were recorded for projection",
    )

    Er: FieldProjectionKSpaceDataArray = Field(
        title="Er",
        description="Spatial distribution of r-component of the electric field.",
    )
    Etheta: FieldProjectionKSpaceDataArray = Field(
        title="Etheta",
        description="Spatial distribution of the theta-component of the electric field.",
    )
    Ephi: FieldProjectionKSpaceDataArray = Field(
        title="Ephi",
        description="Spatial distribution of phi-component of the electric field.",
    )
    Hr: FieldProjectionKSpaceDataArray = Field(
        title="Hr",
        description="Spatial distribution of r-component of the magnetic field.",
    )
    Htheta: FieldProjectionKSpaceDataArray = Field(
        title="Htheta",
        description="Spatial distribution of theta-component of the magnetic field.",
    )
    Hphi: FieldProjectionKSpaceDataArray = Field(
        title="Hphi",
        description="Spatial distribution of phi-component of the magnetic field.",
    )

    @property
    def ux(self) -> np.ndarray:
        """Reciprocal X positions."""
        return self.Etheta.ux.values

    @property
    def uy(self) -> np.ndarray:
        """Reciprocal Y positions."""
        return self.Etheta.uy.values

    @property
    def r(self) -> np.ndarray:
        """Radial distance."""
        return self.Etheta.r.values

    def renormalize_fields(self, proj_distance: float) -> FieldProjectionKSpaceData:
        """Return a :class:`.FieldProjectionKSpaceData` with fields re-normalized to a new
        projection distance, by applying a phase factor based on ``proj_distance``.

        Parameters
        ----------
        proj_distance : float = None
            (micron) new radial distance relative to the monitor's local origin.

        Returns
        -------
        :class:`.FieldProjectionKSpaceData`
            Copy of this :class:`.FieldProjectionKSpaceData` with fields re-projected
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
