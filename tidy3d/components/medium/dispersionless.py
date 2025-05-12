from __future__ import annotations

import functools
from math import isclose
from typing import Dict, List, Optional, Tuple, Union

import autograd.numpy as np
import pydantic.v1 as pd
import xarray as xr
from numpy.typing import NDArray

from tidy3d.components.autograd.derivative_utils import DerivativeInfo, integrate_within_bounds
from tidy3d.components.autograd.types import AutogradFieldMap, TracedFloat
from tidy3d.components.base import cached_property, skip_if_fields_missing
from tidy3d.components.data.data_array import DATA_ARRAY_MAP, ScalarFieldDataArray, SpatialDataArray
from tidy3d.components.data.dataset import ElectromagneticFieldDataset, PermittivityDataset
from tidy3d.components.data.utils import (
    CustomSpatialDataType,
    CustomSpatialDataTypeAnnotated,
    _check_same_coordinates,
    _get_numpy_array,
    _zeros_like,
)
from tidy3d.components.data.validators import validate_no_nans
from tidy3d.components.grid.grid import Coords, Grid
from tidy3d.components.types import ArrayComplex3D, Axis, Bound, InterpMethod
from tidy3d.constants import CONDUCTIVITY, EPSILON_0, PERMITTIVITY, pec_val
from tidy3d.exceptions import SetupError, ValidationError
from tidy3d.log import log

from .base import AbstractCustomMedium, AbstractMedium
from .utils import ensure_freq_in_range


class PECMedium(AbstractMedium):
    """Perfect electrical conductor class.

    Note
    ----

        To avoid confusion from duplicate PECs, must import ``tidy3d.PEC`` instance directly.



    """

    @pd.validator("modulation_spec", always=True)
    def _validate_modulation_spec(cls, val):
        """Check compatibility with modulation_spec."""
        if val is not None:
            raise ValidationError(
                f"A 'modulation_spec' of class {type(val)} is not "
                f"currently supported for medium class {cls}."
            )
        return val

    @ensure_freq_in_range
    def eps_model(self, frequency: float) -> complex:
        # return something like frequency with value of pec_val + 0j
        return 0j * frequency + pec_val

    @cached_property
    def n_cfl(self):
        """This property computes the index of refraction related to CFL condition, so that
        the FDTD with this medium is stable when the time step size that doesn't take
        material factor into account is multiplied by ``n_cfl``.
        """
        return 1.0

    @cached_property
    def is_pec(self):
        """Whether the medium is a PEC."""
        return True


class Medium(AbstractMedium):
    """Dispersionless medium. Mediums define the optical properties of the materials within the simulation.

    Notes
    -----

        In a dispersion-less medium, the displacement field :math:`D(t)` reacts instantaneously to the applied
        electric field :math:`E(t)`.

        .. math::

            D(t) = \\epsilon E(t)

    Example
    -------
    >>> dielectric = Medium(permittivity=4.0, name='my_medium')
    >>> eps = dielectric.eps_model(200e12)

    See Also
    --------

    **Notebooks**
        * `Introduction on Tidy3D working principles <../../notebooks/Primer.html#Mediums>`_
        * `Index <../../notebooks/docs/features/medium.html>`_

    **Lectures**
        * `Modeling dispersive material in FDTD <https://www.flexcompute.com/fdtd101/Lecture-5-Modeling-dispersive-material-in-FDTD/>`_

    **GUI**
        * `Mediums <https://www.flexcompute.com/tidy3d/learning-center/tidy3d-gui/Lecture-2-Mediums/>`_

    """

    permittivity: TracedFloat = pd.Field(
        1.0, ge=1.0, title="Permittivity", description="Relative permittivity.", units=PERMITTIVITY
    )

    conductivity: TracedFloat = pd.Field(
        0.0,
        title="Conductivity",
        description="Electric conductivity. Defined such that the imaginary part of the complex "
        "permittivity at angular frequency omega is given by conductivity/omega.",
        units=CONDUCTIVITY,
    )

    @pd.validator("conductivity", always=True)
    @skip_if_fields_missing(["allow_gain"])
    def _passivity_validation(cls, val, values):
        """Assert passive medium if ``allow_gain`` is False."""
        if not values.get("allow_gain") and val < 0:
            raise ValidationError(
                "For passive medium, 'conductivity' must be non-negative. "
                "To simulate a gain medium, please set 'allow_gain=True'. "
                "Caution: simulations with a gain medium are unstable, and are likely to diverge."
            )
        return val

    @pd.validator("permittivity", always=True)
    @skip_if_fields_missing(["modulation_spec"])
    def _permittivity_modulation_validation(cls, val, values):
        """Assert modulated permittivity cannot be <= 0."""
        modulation = values.get("modulation_spec")
        if modulation is None or modulation.permittivity is None:
            return val

        min_eps_inf = np.min(_get_numpy_array(val))
        if min_eps_inf - modulation.permittivity.max_modulation <= 0:
            raise ValidationError(
                "The minimum permittivity value with modulation applied was found to be negative."
            )
        return val

    @pd.validator("conductivity", always=True)
    @skip_if_fields_missing(["modulation_spec", "allow_gain"])
    def _passivity_modulation_validation(cls, val, values):
        """Assert passive medium if ``allow_gain`` is False."""
        modulation = values.get("modulation_spec")
        if modulation is None or modulation.conductivity is None:
            return val

        min_sigma = np.min(_get_numpy_array(val))
        if not values.get("allow_gain") and min_sigma - modulation.conductivity.max_modulation < 0:
            raise ValidationError(
                "For passive medium, 'conductivity' must be non-negative at any time."
                "With conductivity modulation, this medium can sometimes be active. "
                "Please set 'allow_gain=True'. "
                "Caution: simulations with a gain medium are unstable, "
                "and are likely to diverge."
            )
        return val

    @cached_property
    def n_cfl(self):
        """This property computes the index of refraction related to CFL condition, so that
        the FDTD with this medium is stable when the time step size that doesn't take
        material factor into account is multiplied by ``n_cfl``.

        For dispersiveless medium, it equals ``sqrt(permittivity)``.
        """
        permittivity = self.permittivity
        if self.modulation_spec is not None and self.modulation_spec.permittivity is not None:
            permittivity -= self.modulation_spec.permittivity.max_modulation
        n, _ = self.eps_complex_to_nk(permittivity)
        return n

    @staticmethod
    def _eps_model(permittivity: float, conductivity: float, frequency: float) -> complex:
        """Complex-valued permittivity as a function of frequency."""

        return AbstractMedium.eps_sigma_to_eps_complex(permittivity, conductivity, frequency)

    @ensure_freq_in_range
    def eps_model(self, frequency: float) -> complex:
        """Complex-valued permittivity as a function of frequency."""

        return self._eps_model(self.permittivity, self.conductivity, frequency)

    @classmethod
    def from_nk(cls, n: float, k: float, freq: float, **kwargs):
        """Convert ``n`` and ``k`` values at frequency ``freq`` to :class:`Medium`.

        Parameters
        ----------
        n : float
            Real part of refractive index.
        k : float = 0
            Imaginary part of refrative index.
        freq : float
            Frequency to evaluate permittivity at (Hz).
        kwargs: dict
            Keyword arguments passed to the medium construction.

        Returns
        -------
        :class:`Medium`
            medium containing the corresponding ``permittivity`` and ``conductivity``.
        """
        eps, sigma = AbstractMedium.nk_to_eps_sigma(n, k, freq)
        if eps < 1:
            raise ValidationError(
                "Dispersiveless medium must have 'permittivity>=1`. "
                "Please use 'Lorentz.from_nk()' to covert to a Lorentz medium, or the utility "
                "function 'td.medium_from_nk()' to automatically return the proper medium type."
            )
        return cls(permittivity=eps, conductivity=sigma, **kwargs)

    def compute_derivatives(self, derivative_info: DerivativeInfo) -> AutogradFieldMap:
        """Compute the adjoint derivatives for this object."""

        # get vjps w.r.t. permittivity and conductivity of the bulk
        vjps_volume = self.derivative_eps_sigma_volume(
            E_der_map=derivative_info.E_der_map,
            bounds=derivative_info.bounds,
            freqs=np.atleast_1d(derivative_info.frequency),
        )

        # store the fields asked for by ``field_paths``
        derivative_map = {}
        for field_path in derivative_info.paths:
            field_name, *_ = field_path
            if field_name in vjps_volume:
                derivative_map[field_path] = vjps_volume[field_name]

        return derivative_map

    def derivative_eps_sigma_volume(
        self, E_der_map: ElectromagneticFieldDataset, bounds: Bound, freqs: NDArray
    ) -> dict[str, xr.DataArray]:
        """Get the derivative w.r.t permittivity and conductivity in the volume."""

        vjp_eps_complex = self.derivative_eps_complex_volume(
            E_der_map=E_der_map, bounds=bounds, freqs=freqs
        )

        values = vjp_eps_complex.values

        # vjp of eps_complex_to_eps_sigma
        omegas = 2 * np.pi * freqs
        eps_vjp = np.real(values)
        sigma_vjp = -np.imag(values) / omegas / EPSILON_0

        eps_vjp = np.sum(eps_vjp)
        sigma_vjp = np.sum(sigma_vjp)

        return dict(permittivity=eps_vjp, conductivity=sigma_vjp)

    def derivative_eps_complex_volume(
        self, E_der_map: ElectromagneticFieldDataset, bounds: Bound, freqs: NDArray
    ) -> xr.DataArray:
        """Get the derivative w.r.t complex-valued permittivity in the volume."""

        vjp_value = 0.0
        for field_name in ("Ex", "Ey", "Ez"):
            fld = E_der_map[field_name].sel(f=freqs)
            vjp_value_fld = integrate_within_bounds(
                arr=fld,
                dims=("x", "y", "z"),
                bounds=bounds,
            )
            vjp_value += vjp_value_fld

        return vjp_value.sum("f")


class CustomIsotropicMedium(AbstractCustomMedium, Medium):
    """:class:`.Medium` with user-supplied permittivity distribution.
    (This class is for internal use in v2.0; it will be renamed as `CustomMedium` in v3.0.)

    Example
    -------
    >>> Nx, Ny, Nz = 10, 9, 8
    >>> X = np.linspace(-1, 1, Nx)
    >>> Y = np.linspace(-1, 1, Ny)
    >>> Z = np.linspace(-1, 1, Nz)
    >>> coords = dict(x=X, y=Y, z=Z)
    >>> permittivity= SpatialDataArray(np.ones((Nx, Ny, Nz)), coords=coords)
    >>> conductivity= SpatialDataArray(np.ones((Nx, Ny, Nz)), coords=coords)
    >>> dielectric = CustomIsotropicMedium(permittivity=permittivity, conductivity=conductivity)
    >>> eps = dielectric.eps_model(200e12)
    """

    permittivity: CustomSpatialDataTypeAnnotated = pd.Field(
        ...,
        title="Permittivity",
        description="Relative permittivity.",
        units=PERMITTIVITY,
    )

    conductivity: Optional[CustomSpatialDataTypeAnnotated] = pd.Field(
        None,
        title="Conductivity",
        description="Electric conductivity. Defined such that the imaginary part of the complex "
        "permittivity at angular frequency omega is given by conductivity/omega.",
        units=CONDUCTIVITY,
    )

    _no_nans_eps = validate_no_nans("permittivity")
    _no_nans_sigma = validate_no_nans("conductivity")

    @pd.validator("permittivity", always=True)
    def _eps_inf_greater_no_less_than_one(cls, val):
        """Assert any eps_inf must be >=1"""

        if not CustomIsotropicMedium._validate_isreal_dataarray(val):
            raise SetupError("'permittivity' must be real.")

        if np.any(_get_numpy_array(val) < 1):
            raise SetupError("'permittivity' must be no less than one.")

        return val

    @pd.validator("conductivity", always=True)
    @skip_if_fields_missing(["permittivity"])
    def _conductivity_real_and_correct_shape(cls, val, values):
        """Assert conductivity is real and of right shape."""

        if val is None:
            return val

        if not CustomIsotropicMedium._validate_isreal_dataarray(val):
            raise SetupError("'conductivity' must be real.")

        if not _check_same_coordinates(values["permittivity"], val):
            raise SetupError("'permittivity' and 'conductivity' must have the same coordinates.")
        return val

    @pd.validator("conductivity", always=True)
    @skip_if_fields_missing(["allow_gain"])
    def _passivity_validation(cls, val, values):
        """Assert passive medium if ``allow_gain`` is False."""
        if val is None:
            return val
        if not values.get("allow_gain") and np.any(_get_numpy_array(val) < 0):
            raise ValidationError(
                "For passive medium, 'conductivity' must be non-negative. "
                "To simulate a gain medium, please set 'allow_gain=True'. "
                "Caution: simulations with a gain medium are unstable, and are likely to diverge."
            )
        return val

    @cached_property
    def is_spatially_uniform(self) -> bool:
        """Whether the medium is spatially uniform."""
        if self.conductivity is None:
            return self.permittivity.is_uniform
        return self.permittivity.is_uniform and self.conductivity.is_uniform

    @cached_property
    def n_cfl(self):
        """This property computes the index of refraction related to CFL condition, so that
        the FDTD with this medium is stable when the time step size that doesn't take
        material factor into account is multiplied by ``n_cfl``.

        For dispersiveless medium, it equals ``sqrt(permittivity)``.
        """
        permittivity = np.min(_get_numpy_array(self.permittivity))
        if self.modulation_spec is not None and self.modulation_spec.permittivity is not None:
            permittivity -= self.modulation_spec.permittivity.max_modulation
        n, _ = self.eps_complex_to_nk(permittivity)
        return n

    @cached_property
    def is_isotropic(self):
        """Whether the medium is isotropic."""
        return True

    def eps_dataarray_freq(
        self, frequency: float
    ) -> Tuple[CustomSpatialDataType, CustomSpatialDataType, CustomSpatialDataType]:
        """Permittivity array at ``frequency``.

        Parameters
        ----------
        frequency : float
            Frequency to evaluate permittivity at (Hz).

        Returns
        -------
        Tuple[
            Union[
                :class:`.SpatialDataArray`,
                :class:`.TriangularGridDataset`,
                :class:`.TetrahedralGridDataset`
            ],
            Union[
                :class:`.SpatialDataArray`,
                :class:`.TriangularGridDataset`,
                :class:`.TetrahedralGridDataset`
            ],
            Union[
                :class:`.SpatialDataArray`,
                :class:`.TriangularGridDataset`,
                :class:`.TetrahedralGridDataset`
            ],
        ]
            The permittivity evaluated at ``frequency``.
        """
        conductivity = self.conductivity
        if conductivity is None:
            conductivity = _zeros_like(self.permittivity)
        eps = self.eps_sigma_to_eps_complex(self.permittivity, conductivity, frequency)
        return (eps, eps, eps)

    def _sel_custom_data_inside(self, bounds: Bound):
        """Return a new custom medium that contains the minimal amount data necessary to cover
        a spatial region defined by ``bounds``.


        Parameters
        ----------
        bounds : Tuple[float, float, float], Tuple[float, float float]
            Min and max bounds packaged as ``(minx, miny, minz), (maxx, maxy, maxz)``.

        Returns
        -------
        CustomMedium
            CustomMedium with reduced data.
        """
        if not self.permittivity.does_cover(bounds=bounds):
            log.warning(
                "Permittivity spatial data array does not fully cover the requested region."
            )
        perm_reduced = self.permittivity.sel_inside(bounds=bounds)
        cond_reduced = None
        if self.conductivity is not None:
            if not self.conductivity.does_cover(bounds=bounds):
                log.warning(
                    "Conductivity spatial data array does not fully cover the requested region."
                )
            cond_reduced = self.conductivity.sel_inside(bounds=bounds)

        return self.updated_copy(
            permittivity=perm_reduced,
            conductivity=cond_reduced,
        )


class CustomMedium(AbstractCustomMedium):
    """:class:`.Medium` with user-supplied permittivity distribution.

    Example
    -------
    >>> Nx, Ny, Nz = 10, 9, 8
    >>> X = np.linspace(-1, 1, Nx)
    >>> Y = np.linspace(-1, 1, Ny)
    >>> Z = np.linspace(-1, 1, Nz)
    >>> coords = dict(x=X, y=Y, z=Z)
    >>> permittivity= SpatialDataArray(np.ones((Nx, Ny, Nz)), coords=coords)
    >>> conductivity= SpatialDataArray(np.ones((Nx, Ny, Nz)), coords=coords)
    >>> dielectric = CustomMedium(permittivity=permittivity, conductivity=conductivity)
    >>> eps = dielectric.eps_model(200e12)
    """

    eps_dataset: Optional[PermittivityDataset] = pd.Field(
        None,
        title="Permittivity Dataset",
        description="[To be deprecated] User-supplied dataset containing complex-valued "
        "permittivity as a function of space. Permittivity distribution over the Yee-grid "
        "will be interpolated based on ``interp_method``.",
    )

    permittivity: Optional[CustomSpatialDataTypeAnnotated] = pd.Field(
        None,
        title="Permittivity",
        description="Spatial profile of relative permittivity.",
        units=PERMITTIVITY,
    )

    conductivity: Optional[CustomSpatialDataTypeAnnotated] = pd.Field(
        None,
        title="Conductivity",
        description="Spatial profile Electric conductivity. Defined such "
        "that the imaginary part of the complex permittivity at angular "
        "frequency omega is given by conductivity/omega.",
        units=CONDUCTIVITY,
    )

    _no_nans_eps_dataset = validate_no_nans("eps_dataset")
    _no_nans_permittivity = validate_no_nans("permittivity")
    _no_nans_sigma = validate_no_nans("conductivity")

    @pd.root_validator(pre=True)
    def _warn_if_none(cls, values):
        """Warn if the data array fails to load, and return a vacuum medium."""
        eps_dataset = values.get("eps_dataset")
        permittivity = values.get("permittivity")
        conductivity = values.get("conductivity")
        fail_load = False
        if cls._not_loaded(permittivity):
            log.warning(
                "Loading 'permittivity' without data; constructing a vacuum medium instead."
            )
            fail_load = True
        if cls._not_loaded(conductivity):
            log.warning(
                "Loading 'conductivity' without data; constructing a vacuum medium instead."
            )
            fail_load = True
        if isinstance(eps_dataset, dict):
            if any((v in DATA_ARRAY_MAP for _, v in eps_dataset.items() if isinstance(v, str))):
                log.warning(
                    "Loading 'eps_dataset' without data; constructing a vacuum medium instead."
                )
                fail_load = True
        if fail_load:
            eps_real = SpatialDataArray(np.ones((1, 1, 1)), coords=dict(x=[0], y=[0], z=[0]))
            return dict(permittivity=eps_real)
        return values

    @pd.root_validator(pre=True)
    def _deprecation_dataset(cls, values):
        """Raise deprecation warning if dataset supplied and convert to dataset."""

        eps_dataset = values.get("eps_dataset")
        permittivity = values.get("permittivity")
        conductivity = values.get("conductivity")

        # Incomplete custom medium definition.
        if eps_dataset is None and permittivity is None and conductivity is None:
            raise SetupError("Missing spatial profiles of 'permittivity' or 'eps_dataset'.")
        if eps_dataset is None and permittivity is None:
            raise SetupError("Missing spatial profiles of 'permittivity'.")

        # Definition racing
        if eps_dataset is not None and (permittivity is not None or conductivity is not None):
            raise SetupError(
                "Please either define 'permittivity' and 'conductivity', or 'eps_dataset', "
                "but not both simultaneously."
            )

        if eps_dataset is None:
            return values

        # TODO: sometime before 3.0, uncomment these lines to warn users to start using new API
        # if isinstance(eps_dataset, dict):
        #     eps_components = [eps_dataset[f"eps_{dim}{dim}"] for dim in "xyz"]
        # else:
        #     eps_components = [eps_dataset.eps_xx, eps_dataset.eps_yy, eps_dataset.eps_zz]

        # is_isotropic = eps_components[0] == eps_components[1] == eps_components[2]

        # if is_isotropic:
        #     # deprecation warning for isotropic custom medium
        #     log.warning(
        #         "For spatially varying isotropic medium, the 'eps_dataset' field "
        #         "is being replaced by 'permittivity' and 'conductivity' in v3.0. "
        #         "We recommend you change your scripts to be compatible with the new API."
        #     )
        # else:
        #     # deprecation warning for anisotropic custom medium
        #     log.warning(
        #         "For spatially varying anisotropic medium, this class is being replaced "
        #         "by 'CustomAnisotropicMedium' in v3.0. "
        #         "We recommend you change your scripts to be compatible with the new API."
        #     )

        return values

    @pd.validator("eps_dataset", always=True)
    def _eps_dataset_single_frequency(cls, val):
        """Assert only one frequency supplied."""
        if val is None:
            return val

        for name, eps_dataset_component in val.field_components.items():
            freqs = eps_dataset_component.f
            if len(freqs) != 1:
                raise SetupError(
                    f"'eps_dataset.{name}' must have a single frequency, "
                    f"but it contains {len(freqs)} frequencies."
                )
        return val

    @pd.validator("eps_dataset", always=True)
    @skip_if_fields_missing(["modulation_spec", "allow_gain"])
    def _eps_dataset_eps_inf_greater_no_less_than_one_sigma_positive(cls, val, values):
        """Assert any eps_inf must be >=1"""
        if val is None:
            return val
        modulation = values.get("modulation_spec")

        for comp in ["eps_xx", "eps_yy", "eps_zz"]:
            eps_real, sigma = CustomMedium.eps_complex_to_eps_sigma(
                val.field_components[comp], val.field_components[comp].f
            )
            if np.any(_get_numpy_array(eps_real) < 1):
                raise SetupError(
                    "Permittivity at infinite frequency at any spatial point "
                    "must be no less than one."
                )

            if modulation is not None and modulation.permittivity is not None:
                if np.any(_get_numpy_array(eps_real) - modulation.permittivity.max_modulation <= 0):
                    raise ValidationError(
                        "The minimum permittivity value with modulation applied "
                        "was found to be negative."
                    )

            if not values.get("allow_gain") and np.any(_get_numpy_array(sigma) < 0):
                raise ValidationError(
                    "For passive medium, imaginary part of permittivity must be non-negative. "
                    "To simulate a gain medium, please set 'allow_gain=True'. "
                    "Caution: simulations with a gain medium are unstable, "
                    "and are likely to diverge."
                )

            if (
                not values.get("allow_gain")
                and modulation is not None
                and modulation.conductivity is not None
                and np.any(_get_numpy_array(sigma) - modulation.conductivity.max_modulation <= 0)
            ):
                raise ValidationError(
                    "For passive medium, imaginary part of permittivity must be non-negative "
                    "at any time. "
                    "With conductivity modulation, this medium can sometimes be active. "
                    "Please set 'allow_gain=True'. "
                    "Caution: simulations with a gain medium are unstable, "
                    "and are likely to diverge."
                )
        return val

    @pd.validator("permittivity", always=True)
    @skip_if_fields_missing(["modulation_spec"])
    def _eps_inf_greater_no_less_than_one(cls, val, values):
        """Assert any eps_inf must be >=1"""
        if val is None:
            return val

        if not CustomMedium._validate_isreal_dataarray(val):
            raise SetupError("'permittivity' must be real.")

        if np.any(_get_numpy_array(val) < 1):
            raise SetupError("'permittivity' must be no less than one.")

        modulation = values.get("modulation_spec")
        if modulation is None or modulation.permittivity is None:
            return val

        if np.any(_get_numpy_array(val) - modulation.permittivity.max_modulation <= 0):
            raise ValidationError(
                "The minimum permittivity value with modulation applied was found to be negative."
            )

        return val

    @pd.validator("conductivity", always=True)
    @skip_if_fields_missing(["permittivity", "allow_gain"])
    def _conductivity_non_negative_correct_shape(cls, val, values):
        """Assert conductivity>=0"""

        if val is None:
            return val

        if not CustomMedium._validate_isreal_dataarray(val):
            raise SetupError("'conductivity' must be real.")

        if not values.get("allow_gain") and np.any(_get_numpy_array(val) < 0):
            raise ValidationError(
                "For passive medium, 'conductivity' must be non-negative. "
                "To simulate a gain medium, please set 'allow_gain=True'. "
                "Caution: simulations with a gain medium are unstable, "
                "and are likely to diverge."
            )

        if not _check_same_coordinates(values["permittivity"], val):
            raise SetupError("'permittivity' and 'conductivity' must have the same coordinates.")

        return val

    @pd.validator("conductivity", always=True)
    @skip_if_fields_missing(["eps_dataset", "modulation_spec", "allow_gain"])
    def _passivity_modulation_validation(cls, val, values):
        """Assert passive medium at any time during modulation if ``allow_gain`` is False."""

        # validated already when the data is supplied through `eps_dataset`
        if values.get("eps_dataset"):
            return val

        # permittivity defined with ``permittivity`` and ``conductivity``
        modulation = values.get("modulation_spec")
        if values.get("allow_gain") or modulation is None or modulation.conductivity is None:
            return val
        if val is None or np.any(
            _get_numpy_array(val) - modulation.conductivity.max_modulation < 0
        ):
            raise ValidationError(
                "For passive medium, 'conductivity' must be non-negative at any time. "
                "With conductivity modulation, this medium can sometimes be active. "
                "Please set 'allow_gain=True'. "
                "Caution: simulations with a gain medium are unstable, "
                "and are likely to diverge."
            )
        return val

    @pd.validator("permittivity", "conductivity", always=True)
    def _check_permittivity_conductivity_interpolate(cls, val, values, field):
        """Check that the custom medium 'SpatialDataArrays' can be interpolated."""

        if isinstance(val, SpatialDataArray):
            val._interp_validator(field.name)

        return val

    @cached_property
    def is_isotropic(self) -> bool:
        """Check if the medium is isotropic or anisotropic."""
        if self.eps_dataset is None:
            return True
        if self.eps_dataset.eps_xx == self.eps_dataset.eps_yy == self.eps_dataset.eps_zz:
            return True
        return False

    @cached_property
    def is_spatially_uniform(self) -> bool:
        """Whether the medium is spatially uniform."""
        return self._medium.is_spatially_uniform

    @cached_property
    def freqs(self) -> np.ndarray:
        """float array of frequencies.
        This field is to be deprecated in v3.0.
        """
        # return dummy values in this case
        if self.eps_dataset is None:
            return np.array([0, 0, 0])
        return np.array(
            [
                self.eps_dataset.eps_xx.coords["f"],
                self.eps_dataset.eps_yy.coords["f"],
                self.eps_dataset.eps_zz.coords["f"],
            ]
        )

    @cached_property
    def _medium(self):
        """Internal representation in the form of
        either `CustomIsotropicMedium` or `CustomAnisotropicMedium`.
        """
        from .anisotropic import CustomAnisotropicMediumInternal

        self_dict = self.dict(exclude={"type", "eps_dataset"})
        # isotropic
        if self.eps_dataset is None:
            self_dict.update({"permittivity": self.permittivity, "conductivity": self.conductivity})
            return CustomIsotropicMedium.parse_obj(self_dict)

        def get_eps_sigma(eps_complex: SpatialDataArray, freq: float) -> tuple:
            """Convert a complex permittivity to real permittivity and conductivity."""
            eps_values = np.array(eps_complex.values)

            eps_real, sigma = CustomMedium.eps_complex_to_eps_sigma(eps_values, freq)
            coords = eps_complex.coords

            eps_real = ScalarFieldDataArray(eps_real, coords=coords)
            sigma = ScalarFieldDataArray(sigma, coords=coords)

            eps_real = SpatialDataArray(eps_real.squeeze(dim="f", drop=True))
            sigma = SpatialDataArray(sigma.squeeze(dim="f", drop=True))

            return eps_real, sigma

        # isotropic, but with `eps_dataset`
        if self.is_isotropic:
            eps_complex = self.eps_dataset.eps_xx
            eps_real, sigma = get_eps_sigma(eps_complex, freq=self.freqs[0])

            self_dict.update({"permittivity": eps_real, "conductivity": sigma})
            return CustomIsotropicMedium.parse_obj(self_dict)

        # anisotropic
        mat_comp = {"interp_method": self.interp_method}
        for freq, comp in zip(self.freqs, ["xx", "yy", "zz"]):
            eps_complex = self.eps_dataset.field_components["eps_" + comp]
            eps_real, sigma = get_eps_sigma(eps_complex, freq=freq)

            comp_dict = self_dict.copy()
            comp_dict.update({"permittivity": eps_real, "conductivity": sigma})
            mat_comp.update({comp: CustomIsotropicMedium.parse_obj(comp_dict)})
        return CustomAnisotropicMediumInternal(**mat_comp)

    def _interp_method(self, comp: Axis) -> InterpMethod:
        """Interpolation method applied to comp."""
        return self._medium._interp_method(comp)

    @cached_property
    def n_cfl(self):
        """This property computes the index of refraction related to CFL condition, so that
        the FDTD with this medium is stable when the time step size that doesn't take
        material factor into account is multiplied by ``n_cfl```.

        For dispersiveless custom medium, it equals ``min[sqrt(eps_inf)]``, where ``min``
        is performed over all components and spatial points.
        """
        return self._medium.n_cfl

    def eps_dataarray_freq(
        self, frequency: float
    ) -> Tuple[CustomSpatialDataType, CustomSpatialDataType, CustomSpatialDataType]:
        """Permittivity array at ``frequency``. ()

        Parameters
        ----------
        frequency : float
            Frequency to evaluate permittivity at (Hz).

        Returns
        -------
        Tuple[
            Union[
                :class:`.SpatialDataArray`,
                :class:`.TriangularGridDataset`,
                :class:`.TetrahedralGridDataset`,
            ],
            Union[
                :class:`.SpatialDataArray`,
                :class:`.TriangularGridDataset`,
                :class:`.TetrahedralGridDataset`,
            ],
            Union[
                :class:`.SpatialDataArray`,
                :class:`.TriangularGridDataset`,
                :class:`.TetrahedralGridDataset`,
            ],
        ]
            The permittivity evaluated at ``frequency``.
        """
        return self._medium.eps_dataarray_freq(frequency)

    def eps_diagonal_on_grid(
        self,
        frequency: float,
        coords: Coords,
    ) -> Tuple[ArrayComplex3D, ArrayComplex3D, ArrayComplex3D]:
        """Spatial profile of main diagonal of the complex-valued permittivity
        at ``frequency`` interpolated at the supplied coordinates.

        Parameters
        ----------
        frequency : float
            Frequency to evaluate permittivity at (Hz).
        coords : :class:`.Coords`
            The grid point coordinates over which interpolation is performed.

        Returns
        -------
        Tuple[ArrayComplex3D, ArrayComplex3D, ArrayComplex3D]
            The complex-valued permittivity tensor at ``frequency`` interpolated
            at the supplied coordinate.
        """
        return self._medium.eps_diagonal_on_grid(frequency, coords)

    @ensure_freq_in_range
    def eps_diagonal(self, frequency: float) -> Tuple[complex, complex, complex]:
        """Main diagonal of the complex-valued permittivity tensor
        at ``frequency``. Spatially, we take max{|eps|}, so that autoMesh generation
        works appropriately.
        """
        return self._medium.eps_diagonal(frequency)

    @ensure_freq_in_range
    def eps_model(self, frequency: float) -> complex:
        """Spatial and polarizaiton average of complex-valued permittivity
        as a function of frequency.
        """
        return self._medium.eps_model(frequency)

    @classmethod
    def from_eps_raw(
        cls,
        eps: Union[ScalarFieldDataArray, CustomSpatialDataType],
        freq: float = None,
        interp_method: InterpMethod = "nearest",
        **kwargs,
    ) -> CustomMedium:
        """Construct a :class:`.CustomMedium` from datasets containing raw permittivity values.

        Parameters
        ----------
        eps : Union[
                :class:`.SpatialDataArray`,
                :class:`.ScalarFieldDataArray`,
                :class:`.TriangularGridDataset`,
                :class:`.TetrahedralGridDataset`,
            ]
            Dataset containing complex-valued permittivity as a function of space.
        freq : float, optional
            Frequency at which ``eps`` are defined.
        interp_method : :class:`.InterpMethod`, optional
            Interpolation method to obtain permittivity values that are not supplied
            at the Yee grids.

        Notes
        -----

            For lossy medium that has a complex-valued ``eps``, if ``eps`` is supplied through
            :class:`.SpatialDataArray`, which doesn't contain frequency information,
            the ``freq`` kwarg will be used to evaluate the permittivity and conductivity.
            Alternatively, ``eps`` can be supplied through :class:`.ScalarFieldDataArray`,
            which contains a frequency coordinate.
            In this case, leave ``freq`` kwarg as the default of ``None``.

        Returns
        -------
        :class:`.CustomMedium`
            Medium containing the spatially varying permittivity data.
        """
        if isinstance(eps, CustomSpatialDataType.__args__):
            # purely real, not need to know `freq`
            if CustomMedium._validate_isreal_dataarray(eps):
                return cls(permittivity=eps, interp_method=interp_method, **kwargs)
            # complex permittivity, needs to know `freq`
            if freq is None:
                raise SetupError(
                    "For a complex 'eps', 'freq' at which 'eps' is defined must be supplied",
                )
            eps_real, sigma = CustomMedium.eps_complex_to_eps_sigma(eps, freq)
            return cls(
                permittivity=eps_real, conductivity=sigma, interp_method=interp_method, **kwargs
            )

        # eps is ScalarFieldDataArray
        # contradictory definition of frequency
        freq_data = eps.coords["f"].data[0]
        if freq is not None and not isclose(freq, freq_data):
            raise SetupError(
                "'freq' value is inconsistent with the coordinate 'f'"
                "in 'eps' DataArray. It's unclear at which frequency 'eps' "
                "is defined. Please leave 'freq=None' to use the frequency "
                "value in the DataArray."
            )
        eps_real, sigma = CustomMedium.eps_complex_to_eps_sigma(eps, freq_data)
        eps_real = SpatialDataArray(eps_real.squeeze(dim="f", drop=True))
        sigma = SpatialDataArray(sigma.squeeze(dim="f", drop=True))
        return cls(permittivity=eps_real, conductivity=sigma, interp_method=interp_method, **kwargs)

    @classmethod
    def from_nk(
        cls,
        n: Union[ScalarFieldDataArray, CustomSpatialDataType],
        k: Optional[Union[ScalarFieldDataArray, CustomSpatialDataType]] = None,
        freq: float = None,
        interp_method: InterpMethod = "nearest",
        **kwargs,
    ) -> CustomMedium:
        """Construct a :class:`.CustomMedium` from datasets containing n and k values.

        Parameters
        ----------
        n : Union[
                :class:`.SpatialDataArray`,
                :class:`.ScalarFieldDataArray`,
                :class:`.TriangularGridDataset`,
                :class:`.TetrahedralGridDataset`,
            ]
            Real part of refractive index.
        k : Union[
                :class:`.SpatialDataArray`,
                :class:`.ScalarFieldDataArray`,
                :class:`.TriangularGridDataset`,
                :class:`.TetrahedralGridDataset`,
            ], optional
            Imaginary part of refrative index for lossy medium.
        freq : float, optional
            Frequency at which ``n`` and ``k`` are defined.
        interp_method : :class:`.InterpMethod`, optional
            Interpolation method to obtain permittivity values that are not supplied
            at the Yee grids.
        kwargs: dict
            Keyword arguments passed to the medium construction.

        Note
        ----
        For lossy medium, if both ``n`` and ``k`` are supplied through
        :class:`.SpatialDataArray`, which doesn't contain frequency information,
        the ``freq`` kwarg will be used to evaluate the permittivity and conductivity.
        Alternatively, ``n`` and ``k`` can be supplied through :class:`.ScalarFieldDataArray`,
        which contains a frequency coordinate.
        In this case, leave ``freq`` kwarg as the default of ``None``.

        Returns
        -------
        :class:`.CustomMedium`
            Medium containing the spatially varying permittivity data.
        """
        # lossless
        if k is None:
            if isinstance(n, ScalarFieldDataArray):
                n = SpatialDataArray(n.squeeze(dim="f", drop=True))
            freq = 0  # dummy value
            eps_real, _ = CustomMedium.nk_to_eps_sigma(n, 0 * n, freq)
            return cls(permittivity=eps_real, interp_method=interp_method, **kwargs)

        # lossy case
        if not _check_same_coordinates(n, k):
            raise SetupError("'n' and 'k' must be of the same type and must have same coordinates.")

        # k is a SpatialDataArray
        if isinstance(k, CustomSpatialDataType.__args__):
            if freq is None:
                raise SetupError(
                    "For a lossy medium, must supply 'freq' at which to convert 'n' "
                    "and 'k' to a complex valued permittivity."
                )
            eps_real, sigma = CustomMedium.nk_to_eps_sigma(n, k, freq)
            return cls(
                permittivity=eps_real, conductivity=sigma, interp_method=interp_method, **kwargs
            )

        # k is a ScalarFieldDataArray
        freq_data = k.coords["f"].data[0]
        if freq is not None and not isclose(freq, freq_data):
            raise SetupError(
                "'freq' value is inconsistent with the coordinate 'f'"
                "in 'k' DataArray. It's unclear at which frequency 'k' "
                "is defined. Please leave 'freq=None' to use the frequency "
                "value in the DataArray."
            )

        eps_real, sigma = CustomMedium.nk_to_eps_sigma(n, k, freq_data)
        eps_real = SpatialDataArray(eps_real.squeeze(dim="f", drop=True))
        sigma = SpatialDataArray(sigma.squeeze(dim="f", drop=True))
        return cls(permittivity=eps_real, conductivity=sigma, interp_method=interp_method, **kwargs)

    def grids(self, bounds: Bound) -> Dict[str, Grid]:
        """Make a :class:`.Grid` corresponding to the data in each ``eps_ii`` component.
        The min and max coordinates along each dimension are bounded by ``bounds``."""

        rmin, rmax = bounds
        pt_mins = dict(zip("xyz", rmin))
        pt_maxs = dict(zip("xyz", rmax))

        def make_grid(scalar_field: Union[ScalarFieldDataArray, SpatialDataArray]) -> Grid:
            """Make a grid for a single dataset."""

            def make_bound_coords(coords: np.ndarray, pt_min: float, pt_max: float) -> List[float]:
                """Convert user supplied coords into boundary coords to use in :class:`.Grid`."""

                # get coordinates of the bondaries halfway between user-supplied data
                coord_bounds = (coords[1:] + coords[:-1]) / 2.0

                # res-set coord boundaries that lie outside geometry bounds to the boundary (0 vol.)
                coord_bounds[coord_bounds <= pt_min] = pt_min
                coord_bounds[coord_bounds >= pt_max] = pt_max

                # add the geometry bounds in explicitly
                return [pt_min] + coord_bounds.tolist() + [pt_max]

            # grab user supplied data long this dimension
            coords = {key: np.array(val) for key, val in scalar_field.coords.items()}
            spatial_coords = {key: coords[key] for key in "xyz"}

            # convert each spatial coord to boundary coords
            bound_coords = {}
            for key, coords in spatial_coords.items():
                pt_min = pt_mins[key]
                pt_max = pt_maxs[key]
                bound_coords[key] = make_bound_coords(coords=coords, pt_min=pt_min, pt_max=pt_max)

            # construct grid
            boundaries = Coords(**bound_coords)
            return Grid(boundaries=boundaries)

        grids = {}
        for field_name in ("eps_xx", "eps_yy", "eps_zz"):
            # grab user supplied data long this dimension
            scalar_field = self.eps_dataset.field_components[field_name]

            # feed it to make_grid
            grids[field_name] = make_grid(scalar_field)

        return grids

    def _sel_custom_data_inside(self, bounds: Bound):
        """Return a new custom medium that contains the minimal amount data necessary to cover
        a spatial region defined by ``bounds``.


        Parameters
        ----------
        bounds : Tuple[float, float, float], Tuple[float, float float]
            Min and max bounds packaged as ``(minx, miny, minz), (maxx, maxy, maxz)``.

        Returns
        -------
        CustomMedium
            CustomMedium with reduced data.
        """

        perm_reduced = None
        if self.permittivity is not None:
            if not self.permittivity.does_cover(bounds=bounds):
                log.warning(
                    "Permittivity spatial data array does not fully cover the requested region."
                )
            perm_reduced = self.permittivity.sel_inside(bounds=bounds)

        cond_reduced = None
        if self.conductivity is not None:
            if not self.conductivity.does_cover(bounds=bounds):
                log.warning(
                    "Conductivity spatial data array does not fully cover the requested region."
                )
            cond_reduced = self.conductivity.sel_inside(bounds=bounds)

        eps_reduced = None
        if self.eps_dataset is not None:
            eps_reduced_dict = {}
            for key, comp in self.eps_dataset.field_components.items():
                if not comp.does_cover(bounds=bounds):
                    log.warning(
                        f"{key} spatial data array does not fully cover the requested region."
                    )
                eps_reduced_dict[key] = comp.sel_inside(bounds=bounds)
            eps_reduced = PermittivityDataset(**eps_reduced_dict)

        return self.updated_copy(
            permittivity=perm_reduced,
            conductivity=cond_reduced,
            eps_dataset=eps_reduced,
        )

    def compute_derivatives(self, derivative_info: DerivativeInfo) -> AutogradFieldMap:
        """Compute the adjoint derivatives for this object."""

        vjps = {}

        for field_path in derivative_info.paths:
            if field_path == ("permittivity",):
                vjp_array = 0.0
                for dim in "xyz":
                    vjp_array += self._derivative_field_cmp(
                        E_der_map=derivative_info.E_der_map,
                        eps_data=self.permittivity,
                        dim=dim,
                        freqs=np.atleast_1d(derivative_info.frequency),
                    )
                vjps[field_path] = vjp_array

            elif field_path[0] == "eps_dataset":
                key = field_path[1]
                dim = key[-1]
                vjps[field_path] = self._derivative_field_cmp(
                    E_der_map=derivative_info.E_der_map,
                    eps_data=self.eps_dataset.field_components[key],
                    dim=dim,
                    freqs=np.atleast_1d(derivative_info.frequency),
                )

            else:
                raise NotImplementedError(
                    f"No derivative defined for 'CustomMedium' field: {field_path}."
                )

        return vjps

    def _derivative_field_cmp(
        self,
        E_der_map: ElectromagneticFieldDataset,
        eps_data: PermittivityDataset,
        dim: str,
        freqs: NDArray,
    ) -> np.ndarray:
        """Compute derivative with respect to the ``dim`` components within the custom medium."""

        coords_interp = {key: eps_data.coords[key] for key in "xyz"}
        coords_interp = {key: val for key, val in coords_interp.items() if len(val) > 1}

        E_der_dim_interp = E_der_map[f"E{dim}"].sel(f=freqs)

        for dim_ in "xyz":
            if dim_ not in coords_interp:
                bound_max = np.max(E_der_dim_interp.coords[dim_])
                bound_min = np.min(E_der_dim_interp.coords[dim_])
                dimension_size = bound_max - bound_min

                if dimension_size > 0.0:
                    E_der_dim_interp = E_der_dim_interp.integrate(dim_)

        # compute sizes along each of the interpolation dimensions
        sizes_list = []
        for _, coords in coords_interp.items():
            num_coords = len(coords)
            coords = np.array(coords)

            # compute distances between midpoints for all internal coords
            mid_points = (coords[1:] + coords[:-1]) / 2.0
            dists = np.diff(mid_points)
            sizes = np.zeros(num_coords)
            sizes[1:-1] = dists

            # estimate the sizes on the edges using 2 x the midpoint distance
            sizes[0] = 2 * abs(mid_points[0] - coords[0])
            sizes[-1] = 2 * abs(coords[-1] - mid_points[-1])

            sizes_list.append(sizes)

        # turn this into a volume element, should be re-sizeable to the gradient shape
        if sizes_list:
            d_vol = functools.reduce(np.outer, sizes_list)
        else:
            # if sizes_list is empty, then reduce() fails
            d_vol = np.array(1.0)

        # TODO: probably this could be more robust. eg if the DataArray has weird edge cases
        E_der_dim_interp = (
            E_der_dim_interp.interp(**coords_interp, assume_sorted=True).fillna(0.0).real.sum("f")
        )

        try:
            E_der_dim_interp = E_der_dim_interp * d_vol.reshape(E_der_dim_interp.shape)
        except ValueError:
            log.warning(
                "Skipping volume element normalization of 'CustomMedium' gradients. "
                f"Could not reshape the volume elements of shape {d_vol.shape} "
                f"to the shape of the fields {E_der_dim_interp.shape}. "
                "If you encounter this warning, gradient direction will be accurate but the norm "
                "will be inaccurate. Please raise an issue on the tidy3d front end with this "
                "message and some information about your simulation setup and we will investigate. "
            )
        vjp_array = E_der_dim_interp.values
        vjp_array = vjp_array.reshape(eps_data.shape)

        return vjp_array


PECMedium.update_forward_refs()
PEC = PECMedium(name="PEC")
