"""Spatially varying isotropic medium models."""

from __future__ import annotations

from math import isclose
from typing import TYPE_CHECKING, Any, ClassVar

import autograd.numpy as np
import xarray as xr
from pydantic import Field, field_validator, model_validator

from tidy3d.components.autograd.path_utils import (
    AutogradRoute,
)
from tidy3d.components.autograd.types import PathType
from tidy3d.components.base import cached_property
from tidy3d.components.data.data_array import DATA_ARRAY_MAP, ScalarFieldDataArray, SpatialDataArray
from tidy3d.components.data.dataset import PermittivityDataset
from tidy3d.components.data.utils import (
    CustomSpatialDataType,
    CustomSpatialDataTypeAnnotated,
    _check_same_coordinates,
    _get_numpy_array,
    _zeros_like,
)
from tidy3d.components.data.validators import validate_no_nans
from tidy3d.components.grid.grid import Coords, Grid
from tidy3d.components.types import TYPE_TAG_STR
from tidy3d.constants import (
    CONDUCTIVITY,
    PERMITTIVITY,
)
from tidy3d.exceptions import SetupError, ValidationError
from tidy3d.log import log

if TYPE_CHECKING:
    from pydantic import FieldValidationInfo

    from tidy3d.compat import Self
    from tidy3d.components.autograd.derivative_utils import DerivativeInfo
    from tidy3d.components.autograd.types import AutogradFieldMap
    from tidy3d.components.medium.base import ArrayFloat
    from tidy3d.components.types import (
        ArrayComplex3D,
        Axis,
        Bound,
        InterpMethod,
    )

    from .anisotropic import CustomAnisotropicMedium

from tidy3d.components.medium.abstract_custom import AbstractCustomMedium
from tidy3d.components.medium.base import (
    _EPS_SIGMA_TRACED_PATHS,
    AbstractMedium,
    _normalize_frequency_input,
    _validate_traced_custom_data_path,
    ensure_freq_in_range,
)
from tidy3d.components.medium.isotropic import Medium


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

    permittivity: CustomSpatialDataTypeAnnotated = Field(
        title="Permittivity",
        description="Relative permittivity.",
        json_schema_extra={"units": PERMITTIVITY},
    )

    conductivity: CustomSpatialDataTypeAnnotated | None = Field(
        default=None,
        title="Conductivity",
        description="Electric conductivity. Defined such that the imaginary part of the complex "
        "permittivity at angular frequency omega is given by conductivity/omega.",
        json_schema_extra={"units": CONDUCTIVITY},
    )

    _no_nans = validate_no_nans("permittivity", "conductivity")

    @field_validator("permittivity")
    @classmethod
    def _eps_inf_greater_no_less_than_one(
        cls, val: CustomSpatialDataTypeAnnotated | None
    ) -> CustomSpatialDataTypeAnnotated | None:
        """Assert any eps_inf must be >=1"""

        if not CustomIsotropicMedium._validate_isreal_dataarray(val):
            raise SetupError("'permittivity' must be real.")

        if np.any(_get_numpy_array(val) < 1):
            raise SetupError("'permittivity' must be no less than one.")

        return val

    @model_validator(mode="after")
    def _run_after_validators(self) -> Self:
        """Run post-init validations in an explicit, dependency-aware order."""
        # Keep ordering explicit; avoid super() to prevent passivity from running before
        # conductivity shape checks (and to avoid duplicate passivity calls).
        AbstractMedium._run_after_validators(self)
        self._permittivity_modulation_validation()
        self._passivity_modulation_validation()
        self._conductivity_real_and_correct_shape()
        self._passivity_validation()
        return self

    def _conductivity_real_and_correct_shape(self) -> Self:
        """Assert conductivity is real and of right shape."""
        val = self.conductivity

        if val is None:
            return self

        if not CustomIsotropicMedium._validate_isreal_dataarray(val):
            raise SetupError("'conductivity' must be real.")

        if not _check_same_coordinates(self.permittivity, val):
            raise SetupError("'permittivity' and 'conductivity' must have the same coordinates.")
        return self

    def _passivity_validation(self) -> Self:
        """Assert passive medium if ``allow_gain`` is False."""
        val = self.conductivity
        if val is None:
            return self
        if not self.allow_gain and np.any(_get_numpy_array(val) < 0):
            raise ValidationError(
                "For passive medium, 'conductivity' must be non-negative. "
                "To simulate a gain medium, please set 'allow_gain=True'. "
                "Caution: simulations with a gain medium are unstable, and are likely to diverge."
            )
        return self

    @cached_property
    def is_spatially_uniform(self) -> bool:
        """Whether the medium is spatially uniform."""
        if self.conductivity is None:
            return self.permittivity.is_uniform
        return self.permittivity.is_uniform and self.conductivity.is_uniform

    @cached_property
    def n_cfl(self) -> float:
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
    def is_isotropic(self) -> bool:
        """Whether the medium is isotropic."""
        return True

    @cached_property
    def _permittivity_mean(self) -> complex:
        """Spatial mean of the real permittivity profile."""
        return np.mean(_get_numpy_array(self.permittivity))

    @cached_property
    def _conductivity_mean(self) -> complex:
        """Spatial mean of the conductivity profile."""
        if self.conductivity is None:
            return 0.0
        return np.mean(_get_numpy_array(self.conductivity))

    @ensure_freq_in_range
    def eps_model(self, frequency: float) -> complex:
        """Complex-valued spatially averaged permittivity as a function of frequency."""
        return self.eps_sigma_to_eps_complex(
            self._permittivity_mean, self._conductivity_mean, frequency
        )

    @ensure_freq_in_range
    def eps_diagonal(self, frequency: float) -> tuple[complex, complex, complex]:
        """Main diagonal of the complex-valued permittivity tensor at ``frequency``."""
        if self.conductivity is None:
            eps = np.max(_get_numpy_array(self.permittivity))
            return (eps, eps, eps)
        return super().eps_diagonal(frequency)

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
        frequency = _normalize_frequency_input(frequency)
        conductivity = self.conductivity
        if conductivity is None:
            conductivity = _zeros_like(self.permittivity)

        if not np.isscalar(frequency) and isinstance(self.permittivity, SpatialDataArray):
            omega = 2 * np.pi * xr.DataArray(frequency, coords={"f": frequency}, dims=("f",))
            eps = self._eps_sigma_to_eps_complex_from_omega(self.permittivity, conductivity, omega)
            return (eps, eps, eps)

        eps = self.eps_sigma_to_eps_complex(self.permittivity, conductivity, frequency)
        return (eps, eps, eps)

    def _sel_custom_data_inside(self, bounds: Bound) -> Self:
        """Return a new custom medium that contains the minimal amount data necessary to cover
        a spatial region defined by ``bounds``.


        Parameters
        ----------
        bounds : tuple[float, float, float], tuple[float, float float]
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

    Notes
    -----

        **Practical Advice**

        Use ``CustomMedium`` when permittivity varies spatially — for example, graded-index
        (GRIN) lenses or topology-optimized design regions. Define the permittivity on a
        rectangular grid using ``SpatialDataArray``::

            from tidy3d import SpatialDataArray
            import numpy as np

            x = np.linspace(-5, 5, 100)
            y = np.linspace(-5, 5, 100)
            z = [0]  # 2D variation
            X, Y = np.meshgrid(x, y, indexing="ij")
            eps_data = 1 + 3 * np.exp(-(X**2 + Y**2) / 4)
            eps_data = eps_data[:, :, np.newaxis]

            permittivity = SpatialDataArray(eps_data, coords=dict(x=x, y=y, z=z))
            custom_medium = CustomMedium(permittivity=permittivity)

        For uniform pixelated grids (e.g. topology optimization), consider the convenience method
        :meth:`Structure.from_permittivity_array`, which creates a ``Structure`` with a ``CustomMedium``
        directly from a 3D numpy array and a geometry.

        For wavelength-independent homogeneous materials, use :class:`Medium` instead.
        For dispersive materials, use :class:`FastDispersionFitter` or an analytical model.

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

    _traced_supported_paths: ClassVar[tuple[PathType, ...]] = _EPS_SIGMA_TRACED_PATHS

    eps_dataset: PermittivityDataset | None = Field(
        default=None,
        title="Permittivity Dataset",
        description="[To be deprecated] User-supplied dataset containing complex-valued "
        "permittivity as a function of space. Permittivity distribution over the Yee-grid "
        "will be interpolated based on ``interp_method``.",
    )

    permittivity: CustomSpatialDataTypeAnnotated | None = Field(
        default=None,
        title="Permittivity",
        description="Spatial profile of relative permittivity.",
        json_schema_extra={"units": PERMITTIVITY},
    )

    conductivity: CustomSpatialDataTypeAnnotated | None = Field(
        default=None,
        title="Conductivity",
        description="Spatial profile Electric conductivity. Defined such "
        "that the imaginary part of the complex permittivity at angular "
        "frequency omega is given by conductivity/omega.",
        json_schema_extra={"units": CONDUCTIVITY},
    )

    _no_nans = validate_no_nans("eps_dataset", "permittivity", "conductivity")

    @model_validator(mode="before")
    @classmethod
    def _warn_if_none(cls, data: dict) -> dict:
        """Warn if the data array fails to load, and return a vacuum medium."""
        fail_load = False
        if cls._not_loaded(data.get("permittivity")):
            log.warning(
                "Loading 'permittivity' without data; constructing a vacuum medium instead."
            )
            fail_load = True
        if cls._not_loaded(data.get("conductivity")):
            log.warning(
                "Loading 'conductivity' without data; constructing a vacuum medium instead."
            )
            fail_load = True
        eps_ds = data.get("eps_dataset")
        if isinstance(eps_ds, dict):
            if any(isinstance(v, str) and v in DATA_ARRAY_MAP for v in eps_ds.values()):
                log.warning(
                    "Loading 'eps_dataset' without data; constructing a vacuum medium instead."
                )
                fail_load = True
        if fail_load:
            data["eps_dataset"] = None
            data["conductivity"] = None
            data["modulation_spec"] = None
            data["permittivity"] = SpatialDataArray(
                np.ones((1, 1, 1)), coords={"x": [0], "y": [0], "z": [0]}
            )
        return data

    @model_validator(mode="after")
    def _run_after_validators(self) -> Self:
        """Run post-init validations in an explicit, dependency-aware order."""
        super()._run_after_validators()
        self._deprecation_dataset()
        self._eps_dataset_eps_inf_greater_no_less_than_one_sigma_positive()
        self._eps_inf_greater_no_less_than_one()
        self._conductivity_non_negative_correct_shape()
        self._passivity_modulation_validation()
        return self

    def _deprecation_dataset(self) -> Self:
        """Raise deprecation warning if dataset supplied and convert to dataset."""

        eps_dataset = self.eps_dataset
        permittivity = self.permittivity
        conductivity = self.conductivity

        # Incomplete custom medium definition.
        if eps_dataset is None and permittivity is None and conductivity is None:
            self._raise_validation_error_at_loc(
                SetupError("Missing spatial profiles of 'permittivity' or 'eps_dataset'."),
                "permittivity",
            )
        if eps_dataset is None and permittivity is None:
            self._raise_validation_error_at_loc(
                SetupError("Missing spatial profiles of 'permittivity'."), "permittivity"
            )

        # Definition racing
        if eps_dataset is not None and (permittivity is not None or conductivity is not None):
            self._raise_validation_error_at_loc(
                SetupError(
                    "Please either define 'permittivity' and 'conductivity', or 'eps_dataset', "
                    "but not both simultaneously."
                ),
                "eps_dataset",
            )

        if eps_dataset is None:
            return self

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

        return self

    @field_validator("eps_dataset")
    @classmethod
    def _eps_dataset_single_frequency(
        cls, val: PermittivityDataset | None
    ) -> PermittivityDataset | None:
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

    def _eps_dataset_eps_inf_greater_no_less_than_one_sigma_positive(self) -> Self:
        """Assert any eps_inf must be >=1"""
        val = self.eps_dataset
        if val is None:
            return self
        modulation = self.modulation_spec

        for comp in ["eps_xx", "eps_yy", "eps_zz"]:
            eps_real, sigma = CustomMedium.eps_complex_to_eps_sigma(
                val.field_components[comp], val.field_components[comp].f
            )
            if np.any(_get_numpy_array(eps_real) < 1):
                self._raise_validation_error_at_loc(
                    SetupError(
                        "Permittivity at infinite frequency at any spatial point "
                        "must be no less than one."
                    ),
                    "eps_dataset",
                    comp,
                )

            if modulation is not None and modulation.permittivity is not None:
                if np.any(_get_numpy_array(eps_real) - modulation.permittivity.max_modulation <= 0):
                    self._raise_validation_error_at_loc(
                        ValidationError(
                            "The minimum permittivity value with modulation applied "
                            "was found to be negative."
                        ),
                        "eps_dataset",
                        comp,
                    )

            if not self.allow_gain and np.any(_get_numpy_array(sigma) < 0):
                self._raise_validation_error_at_loc(
                    ValidationError(
                        "For passive medium, imaginary part of permittivity must be non-negative. "
                        "To simulate a gain medium, please set 'allow_gain=True'. "
                        "Caution: simulations with a gain medium are unstable, "
                        "and are likely to diverge."
                    ),
                    "eps_dataset",
                    comp,
                )

            if (
                not self.allow_gain
                and modulation is not None
                and modulation.conductivity is not None
                and np.any(_get_numpy_array(sigma) - modulation.conductivity.max_modulation <= 0)
            ):
                self._raise_validation_error_at_loc(
                    ValidationError(
                        "For passive medium, imaginary part of permittivity must be non-negative "
                        "at any time. "
                        "With conductivity modulation, this medium can sometimes be active. "
                        "Please set 'allow_gain=True'. "
                        "Caution: simulations with a gain medium are unstable, "
                        "and are likely to diverge."
                    ),
                    "eps_dataset",
                    comp,
                )
        return self

    def _eps_inf_greater_no_less_than_one(self) -> Self:
        """Assert any eps_inf must be >=1"""
        val = self.permittivity
        if val is None:
            return self

        if not CustomMedium._validate_isreal_dataarray(val):
            self._raise_validation_error_at_loc(
                SetupError("'permittivity' must be real."), "permittivity"
            )

        if np.any(_get_numpy_array(val) < 1):
            self._raise_validation_error_at_loc(
                SetupError("'permittivity' must be no less than one."), "permittivity"
            )

        modulation = self.modulation_spec
        if modulation is None or modulation.permittivity is None:
            return self

        if np.any(_get_numpy_array(val) - modulation.permittivity.max_modulation <= 0):
            self._raise_validation_error_at_loc(
                ValidationError(
                    "The minimum permittivity value with modulation applied was found to be negative."
                ),
                "permittivity",
            )

        return self

    def _conductivity_non_negative_correct_shape(self) -> Self:
        """Assert conductivity>=0"""
        val = self.conductivity

        if val is None:
            return self

        if not CustomMedium._validate_isreal_dataarray(val):
            self._raise_validation_error_at_loc(
                SetupError("'conductivity' must be real."), "conductivity"
            )

        if not self.allow_gain and np.any(_get_numpy_array(val) < 0):
            self._raise_validation_error_at_loc(
                ValidationError(
                    "For passive medium, 'conductivity' must be non-negative. "
                    "To simulate a gain medium, please set 'allow_gain=True'. "
                    "Caution: simulations with a gain medium are unstable, "
                    "and are likely to diverge."
                ),
                "conductivity",
            )

        if not _check_same_coordinates(self.permittivity, val):
            self._raise_validation_error_at_loc(
                SetupError("'permittivity' and 'conductivity' must have the same coordinates."),
                "permittivity",
            )

        return self

    def _passivity_modulation_validation(self) -> Self:
        """Assert passive medium at any time during modulation if ``allow_gain`` is False."""
        val = self.conductivity

        # validated already when the data is supplied through `eps_dataset`
        if self.eps_dataset:
            return self

        # permittivity defined with ``permittivity`` and ``conductivity``
        modulation = self.modulation_spec
        if self.allow_gain or modulation is None or modulation.conductivity is None:
            return self
        if val is None or np.any(
            _get_numpy_array(val) - modulation.conductivity.max_modulation < 0
        ):
            self._raise_validation_error_at_loc(
                ValidationError(
                    "For passive medium, 'conductivity' must be non-negative at any time. "
                    "With conductivity modulation, this medium can sometimes be active. "
                    "Please set 'allow_gain=True'. "
                    "Caution: simulations with a gain medium are unstable, "
                    "and are likely to diverge."
                ),
                "conductivity",
            )
        return self

    @field_validator("permittivity", "conductivity")
    @classmethod
    def _check_permittivity_conductivity_interpolate(
        cls, val: CustomSpatialDataType | None, info: FieldValidationInfo
    ) -> CustomSpatialDataType | None:
        """Check that the custom medium 'SpatialDataArrays' can be interpolated."""

        if isinstance(val, SpatialDataArray):
            val._interp_validator(info.field_name)

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
    def _permittivity_sorted(self) -> SpatialDataArray | None:
        """Cached copy of permittivity sorted along spatial axes."""
        if self.permittivity is None:
            return None
        return self.permittivity._spatially_sorted

    @cached_property
    def _conductivity_sorted(self) -> SpatialDataArray | None:
        """Cached copy of conductivity sorted along spatial axes."""
        if self.conductivity is None:
            return None
        return self.conductivity._spatially_sorted

    @cached_property
    def _eps_components_sorted(self) -> dict[str, ScalarFieldDataArray]:
        """Cached copies of dataset components sorted along spatial axes."""
        if self.eps_dataset is None:
            return {}
        return {
            key: comp._spatially_sorted for key, comp in self.eps_dataset.field_components.items()
        }

    @cached_property
    def _permittivity_mean(self) -> complex | None:
        """Spatial mean of the real permittivity profile."""
        if self.permittivity is None:
            return None
        return np.mean(_get_numpy_array(self.permittivity))

    @cached_property
    def _conductivity_mean(self) -> complex:
        """Spatial mean of the conductivity profile."""
        if self.conductivity is None:
            return 0.0
        return np.mean(_get_numpy_array(self.conductivity))

    @cached_property
    def freqs(self) -> ArrayFloat:
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
    def _medium(self) -> CustomAnisotropicMedium:
        """Internal representation in the form of
        either `CustomIsotropicMedium` or `CustomAnisotropicMedium`.
        """
        # Imported here to keep the custom-family module graph acyclic.
        from .anisotropic import CustomAnisotropicMediumInternal

        self_dict = self.model_dump(exclude={TYPE_TAG_STR, "eps_dataset"})
        # isotropic
        if self.eps_dataset is None:
            self_dict.update({"permittivity": self.permittivity, "conductivity": self.conductivity})
            return CustomIsotropicMedium.model_validate(self_dict)

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
            return CustomIsotropicMedium.model_validate(self_dict)

        # anisotropic
        mat_comp = {"interp_method": self.interp_method}
        for freq, comp in zip(self.freqs, ["xx", "yy", "zz"]):
            eps_complex = self.eps_dataset.field_components["eps_" + comp]
            eps_real, sigma = get_eps_sigma(eps_complex, freq=freq)

            comp_dict = self_dict.copy()
            comp_dict.update({"permittivity": eps_real, "conductivity": sigma})
            mat_comp.update({comp: CustomIsotropicMedium.model_validate(comp_dict)})
        return CustomAnisotropicMediumInternal(**mat_comp)

    def _interp_method(self, comp: Axis) -> InterpMethod:
        """Interpolation method applied to comp."""
        return self._medium._interp_method(comp)

    @cached_property
    def n_cfl(self) -> float:
        """This property computes the index of refraction related to CFL condition, so that
        the FDTD with this medium is stable when the time step size that doesn't take
        material factor into account is multiplied by ``n_cfl```.

        For dispersiveless custom medium, it equals ``min[sqrt(eps_inf)]``, where ``min``
        is performed over all components and spatial points.
        """
        return self._medium.n_cfl

    def eps_dataarray_freq(
        self, frequency: float
    ) -> tuple[CustomSpatialDataType, CustomSpatialDataType, CustomSpatialDataType]:
        """Permittivity array at ``frequency``. ()

        Parameters
        ----------
        frequency : float
            Frequency to evaluate permittivity at (Hz).

        Returns
        -------
        tuple[Union[:class:`.SpatialDataArray`, :class:`.TriangularGridDataset`, :class:`.TetrahedralGridDataset`], Union[:class:`.SpatialDataArray`, :class:`.TriangularGridDataset`, :class:`.TetrahedralGridDataset`], Union[:class:`.SpatialDataArray`, :class:`.TriangularGridDataset`, :class:`.TetrahedralGridDataset`]]
            The permittivity evaluated at ``frequency``.
        """
        return self._medium.eps_dataarray_freq(frequency)

    def eps_diagonal_on_grid(
        self,
        frequency: float,
        coords: Coords,
    ) -> tuple[ArrayComplex3D, ArrayComplex3D, ArrayComplex3D]:
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
        tuple[ArrayComplex3D, ArrayComplex3D, ArrayComplex3D]
            The complex-valued permittivity tensor at ``frequency`` interpolated
            at the supplied coordinate.
        """
        return self._medium.eps_diagonal_on_grid(frequency, coords)

    @ensure_freq_in_range
    def eps_diagonal(self, frequency: float) -> tuple[complex, complex, complex]:
        """Main diagonal of the complex-valued permittivity tensor
        at ``frequency``. Spatially, we take :math:`\\max\\{|\\varepsilon|\\}`, so that autoMesh generation
        works appropriately.
        """
        if self.eps_dataset is None and self.permittivity is not None and self.conductivity is None:
            eps = np.max(_get_numpy_array(self.permittivity))
            return (eps, eps, eps)
        return self._medium.eps_diagonal(frequency)

    @ensure_freq_in_range
    def eps_model(self, frequency: float) -> complex:
        """Spatial and polarizaiton average of complex-valued permittivity
        as a function of frequency.
        """
        if self.eps_dataset is None and self.permittivity is not None:
            return self.eps_sigma_to_eps_complex(
                self._permittivity_mean, self._conductivity_mean, frequency
            )
        return self._medium.eps_model(frequency)

    @classmethod
    def from_eps_raw(
        cls,
        eps: ScalarFieldDataArray | CustomSpatialDataType,
        freq: float | None = None,
        interp_method: InterpMethod = "nearest",
        **kwargs: Any,
    ) -> Self:
        """Construct a :class:`.CustomMedium` from datasets containing raw permittivity values.

        Parameters
        ----------
        eps : Union[:class:`.SpatialDataArray`, :class:`.ScalarFieldDataArray`, :class:`.TriangularGridDataset`, :class:`.TetrahedralGridDataset`]
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
        n: ScalarFieldDataArray | CustomSpatialDataType,
        k: ScalarFieldDataArray | CustomSpatialDataType | None = None,
        freq: float | None = None,
        interp_method: InterpMethod = "nearest",
        **kwargs: Any,
    ) -> Self:
        """Construct a :class:`.CustomMedium` from datasets containing n and k values.

        Parameters
        ----------
        n : Union[:class:`.SpatialDataArray`, :class:`.ScalarFieldDataArray`, :class:`.TriangularGridDataset`, :class:`.TetrahedralGridDataset`]
            Real part of refractive index.
        k : Union[:class:`.SpatialDataArray`, :class:`.ScalarFieldDataArray`, :class:`.TriangularGridDataset`, :class:`.TetrahedralGridDataset`], optional
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

    def grids(self, bounds: Bound) -> dict[str, Grid]:
        """Make a :class:`.Grid` corresponding to the data in each ``eps_ii`` component.
        The min and max coordinates along each dimension are bounded by ``bounds``."""

        rmin, rmax = bounds
        pt_mins = dict(zip("xyz", rmin))
        pt_maxs = dict(zip("xyz", rmax))

        def make_grid(scalar_field: ScalarFieldDataArray | SpatialDataArray) -> Grid:
            """Make a grid for a single dataset."""

            def make_bound_coords(coords: ArrayFloat, pt_min: float, pt_max: float) -> list[float]:
                """Convert user supplied coords into boundary coords to use in :class:`.Grid`."""

                # get coordinates of the bondaries halfway between user-supplied data
                coord_bounds = (coords[1:] + coords[:-1]) / 2.0

                # res-set coord boundaries that lie outside geometry bounds to the boundary (0 vol.)
                coord_bounds[coord_bounds <= pt_min] = pt_min
                coord_bounds[coord_bounds >= pt_max] = pt_max

                # add the geometry bounds in explicitly
                return [pt_min, *coord_bounds.tolist(), pt_max]

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

    def _sel_custom_data_inside(self, bounds: Bound) -> Self:
        """Return a new custom medium that contains the minimal amount data necessary to cover
        a spatial region defined by ``bounds``.


        Parameters
        ----------
        bounds : tuple[float, float, float], tuple[float, float float]
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

    def _resolve_autograd_route(self, field_path: tuple[Any, ...]) -> AutogradRoute:
        """Resolve and validate one traced CustomMedium path for adjoint routing."""
        scalar_data = {
            "permittivity": self.permittivity,
            "conductivity": self.conductivity,
        }
        active_scalar_data = {name: data for name, data in scalar_data.items() if data is not None}

        if field_path and field_path[0] in active_scalar_data:
            _validate_traced_custom_data_path(
                type(self).__name__,
                field_path,
                scalar_data=active_scalar_data,
            )
        if field_path in self._traced_supported_paths and field_path[0] in active_scalar_data:
            return AutogradRoute(local_path=field_path)

        eps_components = (
            tuple(self.eps_dataset.field_components.keys()) if self.eps_dataset is not None else ()
        )
        if (
            len(field_path) == 2
            and field_path[0] == "eps_dataset"
            and field_path[1] in eps_components
        ):
            return AutogradRoute(local_path=field_path)

        supported_scalar = tuple(active_scalar_data)
        supported_eps = tuple(f"eps_dataset.{component}" for component in eps_components)
        self._raise_unsupported_traced_path(
            field_path,
            supported_parameters=(*supported_scalar, *supported_eps),
        )

    def _compute_derivatives(self, derivative_info: DerivativeInfo) -> AutogradFieldMap:
        """Compute the adjoint derivatives for this object."""

        vjps = {}

        for field_path in derivative_info.paths:
            if field_path[0] == "permittivity":
                spatial_data = self._permittivity_sorted
                if spatial_data is None:
                    continue
                vjp_array = 0.0
                for dim in "xyz":
                    vjp_array += self._derivative_field_cmp_custom(
                        E_der_map=derivative_info.E_der_map,
                        spatial_data=spatial_data,
                        dim=dim,
                        bounds=derivative_info.bounds_intersect,
                        component="real",
                    )
                vjps[field_path] = vjp_array

            elif field_path[0] == "conductivity":
                spatial_data = self._conductivity_sorted
                if spatial_data is None:
                    continue
                vjp_array = 0.0
                for dim in "xyz":
                    vjp_array += self._derivative_field_cmp_custom(
                        E_der_map=derivative_info.E_der_map,
                        spatial_data=spatial_data,
                        dim=dim,
                        bounds=derivative_info.bounds_intersect,
                        component="sigma",
                    )
                vjps[field_path] = vjp_array

            elif field_path[0] == "eps_dataset":
                key = field_path[1]
                spatial_data = self._eps_components_sorted.get(key)
                if spatial_data is None:
                    continue
                dim = key[-1]
                component = (
                    "complex"
                    if np.issubdtype(np.asarray(spatial_data.values).dtype, np.complexfloating)
                    else "real"
                )
                vjps[field_path] = self._derivative_field_cmp_custom(
                    E_der_map=derivative_info.E_der_map,
                    spatial_data=spatial_data,
                    dim=dim,
                    bounds=derivative_info.bounds_intersect,
                    component=component,
                )

        return vjps
