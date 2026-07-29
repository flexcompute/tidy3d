"""Spatially varying anisotropic medium models."""

from __future__ import annotations

from typing import TYPE_CHECKING

import autograd.numpy as np
from pydantic import Field, field_validator

from tidy3d.components.base import cached_property
from tidy3d.components.types import TYPE_TAG_STR, InterpMethod
from tidy3d.exceptions import SetupError
from tidy3d.log import log

if TYPE_CHECKING:
    from pydantic import FieldValidationInfo

    from tidy3d.compat import Self
    from tidy3d.components.data.utils import CustomSpatialDataType
    from tidy3d.components.medium.medium_types import (
        IsotropicCustomMediumInternalType,
        IsotropicCustomMediumType,
    )
    from tidy3d.components.types import (
        Axis,
        Bound,
        PermittivityComponent,
    )

from tidy3d.components.medium.abstract_custom import AbstractCustomMedium
from tidy3d.components.medium.anisotropic import AnisotropicMedium

from .isotropic import CustomMedium


class CustomAnisotropicMedium(AbstractCustomMedium, AnisotropicMedium):
    """Diagonally anisotropic medium with spatially varying permittivity in each component.

    Note
    ----
        Only diagonal anisotropy is currently supported.

    Example
    -------
    >>> import numpy as np
    >>> import tidy3d as td
    >>> Nx, Ny, Nz = 10, 9, 8
    >>> x = np.linspace(-1, 1, Nx)
    >>> y = np.linspace(-1, 1, Ny)
    >>> z = np.linspace(-1, 1, Nz)
    >>> coords = dict(x=x, y=y, z=z)
    >>> permittivity = td.SpatialDataArray(np.ones((Nx, Ny, Nz)), coords=coords)
    >>> conductivity = td.SpatialDataArray(np.ones((Nx, Ny, Nz)), coords=coords)
    >>> medium_xx = td.CustomMedium(permittivity=permittivity, conductivity=conductivity)
    >>> medium_yy = td.CustomMedium(permittivity=permittivity, conductivity=conductivity)
    >>> d_epsilon = td.SpatialDataArray(np.random.random((Nx, Ny, Nz)), coords=coords)
    >>> f = td.SpatialDataArray(1 + np.random.random((Nx, Ny, Nz)), coords=coords)
    >>> delta = td.SpatialDataArray(np.random.random((Nx, Ny, Nz)), coords=coords)
    >>> medium_zz = td.CustomLorentz(eps_inf=permittivity, coeffs=[(d_epsilon, f, delta)])
    >>> anisotropic_dielectric = td.CustomAnisotropicMedium(
    ...     xx=medium_xx, yy=medium_yy, zz=medium_zz
    ... )

    See Also
    --------

    :class:`AnisotropicMedium`
        Diagonally anisotropic medium.

    **Notebooks**
        * `Broadband polarizer assisted by anisotropic metamaterial <../../notebooks/SWGBroadbandPolarizer.html>`_
        * `Thin film lithium niobate adiabatic waveguide coupler <../../notebooks/AdiabaticCouplerLN.html>`_
        * `Defining fully anisotropic materials <../../notebooks/FullyAnisotropic.html>`_
    """

    xx: IsotropicCustomMediumType | CustomMedium = Field(
        title="XX Component",
        description="Medium describing the xx-component of the diagonal permittivity tensor.",
        discriminator=TYPE_TAG_STR,
    )

    yy: IsotropicCustomMediumType | CustomMedium = Field(
        title="YY Component",
        description="Medium describing the yy-component of the diagonal permittivity tensor.",
        discriminator=TYPE_TAG_STR,
    )

    zz: IsotropicCustomMediumType | CustomMedium = Field(
        title="ZZ Component",
        description="Medium describing the zz-component of the diagonal permittivity tensor.",
        discriminator=TYPE_TAG_STR,
    )

    interp_method: InterpMethod | None = Field(
        None,
        title="Interpolation method",
        description="When the value is ``None`` each component will follow its own "
        "interpolation method. When the value is other than ``None`` the interpolation "
        "method specified by this field will override the one in each component.",
    )

    allow_gain: bool | None = Field(
        None,
        title="Allow gain medium",
        description="This field is ignored. Please set ``allow_gain`` in each component",
    )

    subpixel: bool | None = Field(
        None,
        title="Subpixel averaging",
        description="This field is ignored. Please set ``subpixel`` in each component",
    )

    @field_validator("xx", "yy", "zz")
    @classmethod
    def _isotropic_xx(
        cls, val: IsotropicCustomMediumType | CustomMedium, info: FieldValidationInfo
    ) -> IsotropicCustomMediumType | CustomMedium:
        """If it's `CustomMedium`, make sure it's isotropic."""
        if isinstance(val, CustomMedium) and not val.is_isotropic:
            raise SetupError(f"The {info.field_name}-component medium type is not isotropic.")
        return val

    def _ignored_fields(self) -> Self:
        """The field is ignored."""
        if self.xx is not None:
            if self.allow_gain is not None:
                log.warning(
                    "The field 'allow_gain' is ignored. Please set 'allow_gain' in each component."
                )
            if self.subpixel is not None:
                log.warning(
                    "The field 'subpixel' is ignored. Please set 'subpixel' in each component."
                )
        return self

    @cached_property
    def is_spatially_uniform(self) -> bool:
        """Whether the medium is spatially uniform."""
        return any(comp.is_spatially_uniform for comp in self.components.values())

    @cached_property
    def n_cfl(self) -> float:
        """This property computes the index of refraction related to CFL condition, so that
        the FDTD with this medium is stable when the time step size that doesn't take
        material factor into account is multiplied by ``n_cfl``.

        For this medium, it takes the minimal of ``n_clf`` in all components.
        """
        return min(mat_component.n_cfl for mat_component in self.components.values())

    @cached_property
    def is_isotropic(self) -> bool:
        """Whether the medium is isotropic."""
        return False

    def _interp_method(self, comp: Axis) -> InterpMethod:
        """Interpolation method applied to comp."""
        # override `interp_method` in components if self.interp_method is not None
        if self.interp_method is not None:
            return self.interp_method
        # use component's interp_method
        comp_map = ["xx", "yy", "zz"]
        return self.components[comp_map[comp]].interp_method

    def eps_dataarray_freq(
        self, frequency: float
    ) -> tuple[CustomSpatialDataType, CustomSpatialDataType, CustomSpatialDataType]:
        """Permittivity array at ``frequency``.

        Parameters
        ----------
        frequency : float
            Frequency to evaluate permittivity at (Hz).

        Returns
        -------
        tuple[Union[:class:`.SpatialDataArray`, :class:`.TriangularGridDataset`, :class:`.TetrahedralGridDataset`], Union[:class:`.SpatialDataArray`, :class:`.TriangularGridDataset`, :class:`.TetrahedralGridDataset`], Union[:class:`.SpatialDataArray`, :class:`.TriangularGridDataset`, :class:`.TetrahedralGridDataset`]]
            The permittivity evaluated at ``frequency``.
        """
        return tuple(
            mat_component.eps_dataarray_freq(frequency)[ind]
            for ind, mat_component in enumerate(self.components.values())
        )

    def _eps_bounds(
        self,
        frequency: float | None = None,
        eps_component: PermittivityComponent | None = None,
    ) -> tuple[float, float]:
        """Returns permittivity bounds for setting the color bounds when plotting.

        Parameters
        ----------
        frequency : float = None
            Frequency to evaluate the relative permittivity of all mediums.
            If not specified, evaluates at infinite frequency.
        eps_component : Optional[PermittivityComponent] = None
            Component of the permittivity tensor to plot for anisotropic materials,
            e.g. ``"xx"``, ``"yy"``, ``"zz"``, ``"xy"``, ``"yz"``, ...
            Defaults to ``None``, which returns the average of the diagonal values.

        Returns
        -------
        tuple[float, float]
            The min and max values of the permittivity for the selected component and evaluated at ``frequency``.
        """
        comps = ["xx", "yy", "zz"]
        if eps_component in comps:
            # Return the bounds of a specific component
            eps_dataarray = self.eps_dataarray_freq(frequency)
            eps = self._get_real_vals(eps_dataarray[comps.index(eps_component)])
            return (np.min(eps), np.max(eps))
        if eps_component is None:
            # Returns the bounds across all components
            return super()._eps_bounds(frequency=frequency)
        raise ValueError(
            f"Plotting component '{eps_component}' of a diagonally-anisotropic permittivity tensor is not supported."
        )

    def _sel_custom_data_inside(self, bounds: Bound) -> Self:
        return self


class CustomAnisotropicMediumInternal(CustomAnisotropicMedium):
    """Diagonally anisotropic medium with spatially varying permittivity in each component.

    Notes
    -----

        Only diagonal anisotropy is currently supported.

    Example
    -------
    >>> import numpy as np
    >>> import tidy3d as td
    >>> Nx, Ny, Nz = 10, 9, 8
    >>> X = np.linspace(-1, 1, Nx)
    >>> Y = np.linspace(-1, 1, Ny)
    >>> Z = np.linspace(-1, 1, Nz)
    >>> coords = dict(x=X, y=Y, z=Z)
    >>> permittivity = td.SpatialDataArray(np.ones((Nx, Ny, Nz)), coords=coords)
    >>> conductivity = td.SpatialDataArray(np.ones((Nx, Ny, Nz)), coords=coords)
    >>> medium_xx = td.CustomMedium(permittivity=permittivity, conductivity=conductivity)
    >>> medium_yy = td.CustomMedium(permittivity=permittivity, conductivity=conductivity)
    >>> d_epsilon = td.SpatialDataArray(np.random.random((Nx, Ny, Nz)), coords=coords)
    >>> f = td.SpatialDataArray(1 + np.random.random((Nx, Ny, Nz)), coords=coords)
    >>> delta = td.SpatialDataArray(np.random.random((Nx, Ny, Nz)), coords=coords)
    >>> medium_zz = td.CustomLorentz(eps_inf=permittivity, coeffs=[(d_epsilon, f, delta)])
    >>> anisotropic_dielectric = td.CustomAnisotropicMedium(
    ...     xx=medium_xx, yy=medium_yy, zz=medium_zz
    ... )
    """

    xx: IsotropicCustomMediumInternalType | CustomMedium = Field(
        title="XX Component",
        description="Medium describing the xx-component of the diagonal permittivity tensor.",
        discriminator=TYPE_TAG_STR,
    )

    yy: IsotropicCustomMediumInternalType | CustomMedium = Field(
        title="YY Component",
        description="Medium describing the yy-component of the diagonal permittivity tensor.",
        discriminator=TYPE_TAG_STR,
    )

    zz: IsotropicCustomMediumInternalType | CustomMedium = Field(
        title="ZZ Component",
        description="Medium describing the zz-component of the diagonal permittivity tensor.",
        discriminator=TYPE_TAG_STR,
    )
