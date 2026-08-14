"""Spatially varying dispersive medium models."""

from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING, Any

import autograd.numpy as np
from pydantic import Field, field_validator, model_validator

from tidy3d.components.base import cached_property
from tidy3d.components.data.data_array import SpatialDataArray
from tidy3d.components.data.utils import (
    CustomSpatialDataTypeAnnotated,
    UnstructuredGridDatasetType,
    _check_same_coordinates,
    _get_numpy_array,
    _ones_like,
    _zeros_like,
)
from tidy3d.components.data.validators import validate_no_nans
from tidy3d.components.types import TYPE_TAG_STR
from tidy3d.constants import (
    EPSILON_0,
    HERTZ,
    LARGEST_FP_NUMBER,
    MICROMETER,
    PERMITTIVITY,
    RADPERSEC,
    SECOND,
    fp_eps,
)
from tidy3d.exceptions import SetupError, ValidationError
from tidy3d.log import log

if TYPE_CHECKING:
    from collections.abc import Callable

    from pydantic import PositiveFloat

    from tidy3d.compat import Self
    from tidy3d.components.autograd.derivative_utils import DerivativeInfo
    from tidy3d.components.autograd.path_utils import AutogradRoute
    from tidy3d.components.autograd.types import AutogradFieldMap, TracedPositiveFloat
    from tidy3d.components.data.dataset import PermittivityDataset
    from tidy3d.components.data.utils import CustomSpatialDataType
    from tidy3d.components.grid.grid import Coords
    from tidy3d.components.medium.base import (
        AbstractMedium,
        ArrayComplex,
        ArrayFloat,
        FrequencyArray,
        WeightFunction,
    )
    from tidy3d.components.types import (
        ArrayComplex3D,
        Bound,
        InterpMethod,
    )

from tidy3d.components.medium.abstract_custom import AbstractCustomMedium
from tidy3d.components.medium.base import _validate_traced_custom_data_path
from tidy3d.components.medium.debye import Debye
from tidy3d.components.medium.drude import Drude
from tidy3d.components.medium.lorentz import Lorentz
from tidy3d.components.medium.pole_residue import DispersiveMedium, PoleResidue
from tidy3d.components.medium.sellmeier import Sellmeier

from .isotropic import CustomMedium


class CustomDispersiveMedium(AbstractCustomMedium, DispersiveMedium, ABC):
    """A spatially varying dispersive medium."""

    def _resolve_autograd_route(self, field_path: tuple[Any, ...]) -> AutogradRoute:
        """Resolve and validate one traced custom dispersive medium path."""
        root = self._traced_indexed_root
        scalar_data = {path[0]: getattr(self, path[0]) for path in self._traced_supported_paths}
        indexed_data = {root: getattr(self, root)} if root is not None else {}
        _validate_traced_custom_data_path(
            type(self).__name__,
            field_path,
            scalar_data=scalar_data,
            indexed_data=indexed_data,
        )
        return super()._resolve_autograd_route(field_path)

    @cached_property
    def n_cfl(self) -> float:
        """This property computes the index of refraction related to CFL condition, so that
        the FDTD with this medium is stable when the time step size that doesn't take
        material factor into account is multiplied by ``n_cfl``.

        For PoleResidue model, it equals ``sqrt(eps_inf)``
        [https://ieeexplore.ieee.org/document/9082879].
        """
        permittivity = np.min(_get_numpy_array(self.pole_residue.eps_inf))
        if self.modulation_spec is not None and self.modulation_spec.permittivity is not None:
            permittivity -= self.modulation_spec.permittivity.max_modulation
        n, _ = self.eps_complex_to_nk(permittivity)
        return n

    @cached_property
    def is_isotropic(self) -> bool:
        """Whether the medium is isotropic."""
        return True

    @cached_property
    def pole_residue(self) -> CustomPoleResidue:
        """Representation of Medium as a pole-residue model."""
        return CustomPoleResidue(
            **self._pole_residue_dict(),
            interp_method=self.interp_method,
            allow_gain=self.allow_gain,
            subpixel=self.subpixel,
        )

    @staticmethod
    def _warn_if_data_none(
        nested_tuple_field: str,
    ) -> Callable[[type[AbstractMedium], dict[str, Any]], dict[str, Any]]:
        """Warn if any of `eps_inf` and nested_tuple_field are not loaded,
        and return a vacuum with eps_inf = 1.
        """

        @model_validator(mode="before")
        @classmethod
        def _warn_if_none(cls: type[AbstractMedium], data: dict[str, Any]) -> dict[str, Any]:
            is_not_loaded = AbstractCustomMedium._not_loaded

            eps_inf = data.get("eps_inf")
            coeffs = data.get(nested_tuple_field, ())

            eps_bad = is_not_loaded(eps_inf)
            coeff_bad = any(is_not_loaded(c) for coeff in coeffs for c in coeff)

            if not (eps_bad or coeff_bad):
                return data

            if eps_bad:
                log.warning("Loading 'eps_inf' without data; constructing a vacuum medium instead.")
            if coeff_bad:
                log.warning(
                    f"Loading '{nested_tuple_field}' without data; constructing a vacuum medium instead."
                )

            data[nested_tuple_field] = ()
            if eps_inf is not None:
                data["eps_inf"] = SpatialDataArray(
                    np.ones((1, 1, 1)), coords={"x": [0], "y": [0], "z": [0]}
                )

            return data

        return _warn_if_none

    # --- helpers for custom dispersive adjoints ---
    def _sum_complex_eps_sensitivity(
        self,
        derivative_info: DerivativeInfo,
        spatial_ref: PermittivityDataset,
    ) -> ArrayComplex:
        """Sum complex permittivity sensitivities over xyz on the given spatial grid.

        Parameters
        ----------
        derivative_info : DerivativeInfo
            Info bundle carrying field maps and frequencies.
        spatial_ref : PermittivityDataset
            Spatial dataset to define the grid/coords for interpolation and summation.

        Returns
        -------
        np.ndarray
            Complex-valued aggregated dJ array with frequency as the last axis.
        """
        dJ = 0.0 + 0.0j
        for dim in "xyz":
            dJ += self._derivative_field_cmp_custom(
                E_der_map=derivative_info.E_der_map,
                spatial_data=spatial_ref,
                dim=dim,
                bounds=derivative_info.bounds_intersect,
                component="complex",
                sum_over_freqs=False,
            )
        return dJ

    @staticmethod
    def _accum_real_inner(dJ: ArrayComplex, weight: ArrayComplex) -> ArrayFloat:
        """Compute Re(dJ * weight) with proper broadcasting."""
        return np.real(dJ * weight)

    def _sum_real_over_freqs(self, dJ: np.ndarray, freqs: list[float] | np.ndarray) -> np.ndarray:
        """Sum real parts over the frequency axis using the shared accumulator."""
        freqs = np.asarray(freqs, float)
        if freqs.size == 0:
            raise ValueError("freqs must not be empty")
        if np.ndim(dJ) == 0 or np.shape(dJ)[-1] != freqs.size:
            raise ValueError(
                f"Expected frequency axis on dJ with size {freqs.size}, got shape {np.shape(dJ)}."
            )

        ones = np.ones_like(dJ[..., 0], dtype=float)
        return self._sum_over_freqs(freqs=freqs, dJ=dJ, weight_fn=lambda _f, ones=ones: ones)

    def _sum_over_freqs(
        self, freqs: FrequencyArray, dJ: ArrayComplex, weight_fn: WeightFunction
    ) -> ArrayFloat:
        """Accumulate gradient contributions over frequencies using provided weight function.

        Parameters
        ----------
        freqs : array-like
            Frequencies to accumulate over.
        dJ : np.ndarray
            Complex dataset sensitivity with spatial shape.
        weight_fn : Callable[[float], np.ndarray]
            Function mapping frequency to weight array broadcastable to dJ.

        Returns
        -------
        np.ndarray
            Real-valued gradient array matching dJ's broadcasted shape.
        """
        freqs = np.asarray(freqs, float)
        if freqs.size == 0:
            raise ValueError("freqs must not be empty")

        weight0 = weight_fn(freqs[0])
        if dJ.ndim == np.ndim(weight0) + 1 and dJ.shape[-1] == freqs.size:
            g = self._accum_real_inner(dJ[..., 0], weight0)
            for idx, f in enumerate(freqs[1:], start=1):
                g = g + self._accum_real_inner(dJ[..., idx], weight_fn(f))
            return g

        g = self._accum_real_inner(dJ, weight0)
        for f in freqs[1:]:
            g = g + self._accum_real_inner(dJ, weight_fn(f))
        return g


class CustomPoleResidue(CustomDispersiveMedium, PoleResidue):
    """A spatially varying dispersive medium described by the pole-residue pair model.

    Notes
    -----

        In this method, the frequency-dependent permittivity :math:`\\epsilon(\\omega)` is expressed as a sum of
        resonant material poles [1]_.

        .. math::

            \\epsilon(\\omega) = \\epsilon_\\infty - \\sum_i
            \\left[\\frac{c_i}{j \\omega + a_i} +
            \\frac{c_i^*}{j \\omega + a_i^*}\\right]

        For each of these resonant poles identified by the index :math:`i`, an auxiliary differential equation is
        used to relate the auxiliary current :math:`J_i(t)` to the applied electric field :math:`E(t)`.
        The sum of all these auxiliary current contributions describes the total dielectric response of the material.

        .. math::

            \\frac{d}{dt} J_i (t) - a_i J_i (t) = \\epsilon_0 c_i \\frac{d}{dt} E (t)

        Hence, the computational cost increases with the number of poles.

        **References**

        .. [1]   M. Han, R.W. Dutton and S. Fan, IEEE Microwave and Wireless Component Letters, 16, 119 (2006).

        .. TODO add links to notebooks using this.

    Example
    -------
    >>> x = np.linspace(-1, 1, 5)
    >>> y = np.linspace(-1, 1, 6)
    >>> z = np.linspace(-1, 1, 7)
    >>> coords = dict(x=x, y=y, z=z)
    >>> eps_inf = SpatialDataArray(np.ones((5, 6, 7)), coords=coords)
    >>> a1 = SpatialDataArray(-np.random.random((5, 6, 7)), coords=coords)
    >>> c1 = SpatialDataArray(np.random.random((5, 6, 7)), coords=coords)
    >>> a2 = SpatialDataArray(-np.random.random((5, 6, 7)), coords=coords)
    >>> c2 = SpatialDataArray(np.random.random((5, 6, 7)), coords=coords)
    >>> pole_res = CustomPoleResidue(eps_inf=eps_inf, poles=[(a1, c1), (a2, c2)])
    >>> eps = pole_res.eps_model(200e12)

    See Also
    --------

    **Notebooks**

    * `Fitting dispersive material models <../../notebooks/Fitting.html>`_

    **Lectures**

    * `Modeling dispersive material in FDTD <https://www.flexcompute.com/fdtd101/Lecture-5-Modeling-dispersive-material-in-FDTD/>`_
    """

    eps_inf: CustomSpatialDataTypeAnnotated = Field(
        title="Epsilon at Infinity",
        description="Relative permittivity at infinite frequency (:math:`\\epsilon_\\infty`).",
        json_schema_extra={"units": PERMITTIVITY},
    )

    poles: tuple[tuple[CustomSpatialDataTypeAnnotated, CustomSpatialDataTypeAnnotated], ...] = (
        Field(
            (),
            title="Poles",
            description="Tuple of complex-valued (:math:`a_i, c_i`) poles for the model.",
            json_schema_extra={"units": (RADPERSEC, RADPERSEC)},
        )
    )

    _no_nans = validate_no_nans("eps_inf", "poles")
    _warn_if_none = CustomDispersiveMedium._warn_if_data_none("poles")

    @field_validator("eps_inf")
    @classmethod
    def _eps_inf_positive(cls, val: CustomSpatialDataType) -> CustomSpatialDataType:
        """eps_inf must be positive"""
        if not CustomDispersiveMedium._validate_isreal_dataarray(val):
            raise SetupError("'eps_inf' must be real.")
        if np.any(_get_numpy_array(val) < 0):
            raise SetupError("'eps_inf' must be positive.")
        return val

    @model_validator(mode="after")
    def _run_after_validators(self) -> Self:
        """Run post-init validations in an explicit, dependency-aware order."""
        super()._run_after_validators()
        self._poles_correct_shape()
        return self

    def _poles_correct_shape(self) -> Self:
        """poles must have the same shape."""
        val = self.poles

        for coeffs in val:
            for coeff in coeffs:
                if not _check_same_coordinates(coeff, self.eps_inf):
                    self._raise_validation_error_at_loc(
                        SetupError(
                            "All pole coefficients 'a' and 'c' must have the same coordinates; "
                            "The coordinates must also be consistent with 'eps_inf'."
                        ),
                        "poles",
                    )
        return self

    @cached_property
    def is_spatially_uniform(self) -> bool:
        """Whether the medium is spatially uniform."""
        if not self.eps_inf.is_uniform:
            return False

        for coeffs in self.poles:
            for coeff in coeffs:
                if not coeff.is_uniform:
                    return False
        return True

    @staticmethod
    def _sorted_spatial_data(
        data: CustomSpatialDataTypeAnnotated,
    ) -> CustomSpatialDataTypeAnnotated:
        """Return spatial data sorted along its coordinates if applicable."""
        if isinstance(data, SpatialDataArray):
            return data._spatially_sorted
        return data

    @cached_property
    def _eps_inf_sorted(self) -> CustomSpatialDataTypeAnnotated:
        """Cached sorted copy of eps_inf when structured data is provided."""
        return self._sorted_spatial_data(self.eps_inf)

    @cached_property
    def _poles_sorted(
        self,
    ) -> tuple[tuple[CustomSpatialDataTypeAnnotated, CustomSpatialDataTypeAnnotated], ...]:
        """Cached sorted copies of pole coefficients when structured data is provided."""
        return tuple(
            (self._sorted_spatial_data(a), self._sorted_spatial_data(c)) for a, c in self.poles
        )

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
        eps = PoleResidue.eps_model(self, frequency)
        return (eps, eps, eps)

    def poles_on_grid(self, coords: Coords) -> tuple[tuple[ArrayComplex3D, ArrayComplex3D], ...]:
        """Spatial profile of poles interpolated at the supplied coordinates.

        Parameters
        ----------
        coords : :class:`.Coords`
            The grid point coordinates over which interpolation is performed.

        Returns
        -------
        tuple[tuple[ArrayComplex3D, ArrayComplex3D], ...]
            The poles interpolated at the supplied coordinate.
        """

        def fun_interp(input_data: SpatialDataArray) -> ArrayComplex3D:
            return _get_numpy_array(coords.spatial_interp(input_data, self.interp_method))

        return tuple((fun_interp(a), fun_interp(c)) for (a, c) in self.poles)

    @classmethod
    def from_medium(cls, medium: CustomMedium) -> Self:
        """Convert a :class:`.CustomMedium` to a pole residue model.

        Parameters
        ----------
        medium: :class:`.CustomMedium`
            The medium with permittivity and conductivity to convert.

        Returns
        -------
        :class:`.CustomPoleResidue`
            The pole residue equivalent.
        """
        poles = [(_zeros_like(medium.conductivity), medium.conductivity / (2 * EPSILON_0))]
        medium_dict = medium.model_dump(
            exclude={TYPE_TAG_STR, "eps_dataset", "permittivity", "conductivity"}
        )
        medium_dict.update({"eps_inf": medium.permittivity, "poles": poles})
        return CustomPoleResidue.model_validate(medium_dict)

    def to_medium(self) -> CustomMedium:
        """Convert to a :class:`.CustomMedium`.
        Requires the pole residue model to only have a pole at 0 frequency,
        corresponding to a constant conductivity term.

        Returns
        -------
        :class:`.CustomMedium`
            The non-dispersive equivalent with constant permittivity and conductivity.
        """
        res = 0
        for a, c in self.poles:
            if np.any(abs(_get_numpy_array(a)) > fp_eps):
                raise ValidationError(
                    "Cannot convert dispersive 'CustomPoleResidue' to 'CustomMedium'."
                )
            res = res + (c + np.conj(c)) / 2
        sigma = res * 2 * EPSILON_0

        self_dict = self.model_dump(exclude={TYPE_TAG_STR, "eps_inf", "poles"})
        self_dict.update({"permittivity": self.eps_inf, "conductivity": np.real(sigma)})
        return CustomMedium.model_validate(self_dict)

    @cached_property
    def loss_upper_bound(self) -> float:
        """Not implemented yet."""
        raise SetupError("To be implemented.")

    def _sel_custom_data_inside(self, bounds: Bound) -> Self:
        """Return a new custom medium that contains the minimal amount data necessary to cover
        a spatial region defined by ``bounds``.


        Parameters
        ----------
        bounds : tuple[float, float, float], tuple[float, float float]
            Min and max bounds packaged as ``(minx, miny, minz), (maxx, maxy, maxz)``.

        Returns
        -------
        CustomPoleResidue
            CustomPoleResidue with reduced data.
        """
        if not self.eps_inf.does_cover(bounds=bounds):
            log.warning("eps_inf spatial data array does not fully cover the requested region.")
        eps_inf_reduced = self.eps_inf.sel_inside(bounds=bounds)
        poles_reduced = []
        for pole, residue in self.poles:
            if not pole.does_cover(bounds=bounds):
                log.warning("Pole spatial data array does not fully cover the requested region.")

            if not residue.does_cover(bounds=bounds):
                log.warning("Residue spatial data array does not fully cover the requested region.")

            poles_reduced.append((pole.sel_inside(bounds), residue.sel_inside(bounds)))

        return self.updated_copy(eps_inf=eps_inf_reduced, poles=tuple(poles_reduced))

    def _compute_derivatives(self, derivative_info: DerivativeInfo) -> AutogradFieldMap:
        """Compute adjoint derivatives by preparing array data and calling the static helper."""

        eps_inf_sorted = self._eps_inf_sorted
        is_unstructured = isinstance(eps_inf_sorted, UnstructuredGridDatasetType)
        if is_unstructured:
            raise NotImplementedError(
                "Adjoint derivatives for unstructured custom media are not supported."
            )

        dJ_deps_complex = 0.0 + 0.0j
        for dim in "xyz":
            dJ_deps_complex += self._derivative_field_cmp_custom(
                E_der_map=derivative_info.E_der_map,
                spatial_data=eps_inf_sorted,
                dim=dim,
                bounds=derivative_info.bounds_intersect,
                component="complex",
                sum_over_freqs=False,
            )

        poles_vals = [
            (np.array(a_sorted.values, dtype=complex), np.array(c_sorted.values, dtype=complex))
            for a_sorted, c_sorted in self._poles_sorted
        ]

        freqs = np.asarray(derivative_info.frequencies, float)
        vjps_total = {}
        for idx, freq in enumerate(freqs):
            dJ_deps_complex_f = dJ_deps_complex[..., idx]
            vjps_f = PoleResidue._get_vjps_from_params(
                dJ_deps_complex=dJ_deps_complex_f,
                poles_vals=poles_vals,
                omega=2 * np.pi * freq,
                requested_paths=derivative_info.paths,
                project_real=False,
            )
            for path, vjp in vjps_f.items():
                if path not in vjps_total:
                    vjps_total[path] = vjp
                else:
                    vjps_total[path] += vjp
        return vjps_total


class CustomSellmeier(CustomDispersiveMedium, Sellmeier):
    """A spatially varying dispersive medium described by the Sellmeier model.

    Notes
    -----

        The frequency-dependence of the refractive index is described by:

        .. math::

            n(\\lambda)^2 = 1 + \\sum_i \\frac{B_i \\lambda^2}{\\lambda^2 - C_i}

    Example
    -------
    >>> x = np.linspace(-1, 1, 5)
    >>> y = np.linspace(-1, 1, 6)
    >>> z = np.linspace(-1, 1, 7)
    >>> coords = dict(x=x, y=y, z=z)
    >>> b1 = SpatialDataArray(np.random.random((5, 6, 7)), coords=coords)
    >>> c1 = SpatialDataArray(1 + np.random.random((5, 6, 7)), coords=coords)
    >>> sellmeier_medium = CustomSellmeier(coeffs=[(b1,c1),])
    >>> eps = sellmeier_medium.eps_model(200e12)

    See Also
    --------

    :class:`Sellmeier`
        A dispersive medium described by the Sellmeier model.

    **Notebooks**
        * `Fitting dispersive material models <../../notebooks/Fitting.html>`_

    **Lectures**
        * `Modeling dispersive material in FDTD <https://www.flexcompute.com/fdtd101/Lecture-5-Modeling-dispersive-material-in-FDTD/>`_
    """

    coeffs: tuple[tuple[CustomSpatialDataTypeAnnotated, CustomSpatialDataTypeAnnotated], ...] = (
        Field(
            title="Coefficients",
            description="List of Sellmeier (:math:`B_i, C_i`) coefficients.",
            json_schema_extra={"units": (None, MICROMETER + "^2")},
        )
    )

    _no_nans = validate_no_nans("coeffs")
    _warn_if_none = CustomDispersiveMedium._warn_if_data_none("coeffs")

    @field_validator("coeffs")
    @classmethod
    def _correct_shape_and_sign(
        cls,
        val: tuple[tuple[CustomSpatialDataType, CustomSpatialDataType], ...],
    ) -> tuple[tuple[CustomSpatialDataType, CustomSpatialDataType], ...]:
        """every term in coeffs must have the same shape, and B>=0 and C>0."""
        if len(val) == 0:
            return val
        for B, C in val:
            if not _check_same_coordinates(B, val[0][0]) or not _check_same_coordinates(
                C, val[0][0]
            ):
                raise SetupError("Every term in 'coeffs' must have the same coordinates.")
            if not CustomDispersiveMedium._validate_isreal_dataarray_tuple((B, C)):
                raise SetupError("'B' and 'C' must be real.")
            if np.any(_get_numpy_array(C) <= 0):
                raise SetupError("'C' must be positive.")
        return val

    def _passivity_validation(self) -> Self:
        """Assert passive medium if `allow_gain` is False."""
        val = self.coeffs
        if self.allow_gain:
            return self
        for B, _ in val:
            if np.any(_get_numpy_array(B) < 0):
                raise ValidationError(
                    "For passive medium, 'B_i' must be non-negative. "
                    "To simulate a gain medium, please set 'allow_gain=True'. "
                    "Caution: simulations with a gain medium are unstable, "
                    "and are likely to diverge."
                )
        return self

    @field_validator("coeffs")
    @classmethod
    def _coeffs_C_all_near_zero_or_much_greater(
        cls, val: tuple[tuple[float, PositiveFloat], ...]
    ) -> tuple[tuple[float, PositiveFloat], ...]:
        """We restrict either all C~=0, or very different from 0."""
        for _, C in val:
            c_array_near_zero = np.isclose(_get_numpy_array(C), 0)
            if np.any(c_array_near_zero) and not np.all(c_array_near_zero):
                raise SetupError(
                    "Coefficients 'C_i' are restricted to be "
                    "either all near zero or much greater than 0."
                )
        return val

    @cached_property
    def is_spatially_uniform(self) -> bool:
        """Whether the medium is spatially uniform."""
        for coeffs in self.coeffs:
            for coeff in coeffs:
                if not coeff.is_uniform:
                    return False
        return True

    def _pole_residue_dict(self) -> dict:
        """Dict representation of Medium as a pole-residue model."""
        poles_dict = Sellmeier._pole_residue_dict(self)
        if len(self.coeffs) > 0:
            poles_dict.update({"eps_inf": _ones_like(self.coeffs[0][0])})
        return poles_dict

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
        eps = Sellmeier.eps_model(self, frequency)
        # if `eps` is simply a float, convert it to a SpatialDataArray ; this is possible when
        # `coeffs` is empty.
        if isinstance(eps, (int, float, complex)):
            eps = SpatialDataArray(eps * np.ones((1, 1, 1)), coords={"x": [0], "y": [0], "z": [0]})
        return (eps, eps, eps)

    @classmethod
    def from_dispersion(
        cls,
        n: CustomSpatialDataType,
        freq: float,
        dn_dwvl: CustomSpatialDataType,
        interp_method: InterpMethod = "nearest",
        **kwargs: Any,
    ) -> Self:
        """Convert ``n`` and wavelength dispersion ``dn_dwvl`` values at frequency ``freq`` to
        a single-pole :class:`CustomSellmeier` medium.

        Parameters
        ----------
        n : Union[:class:`.SpatialDataArray`, :class:`.TriangularGridDataset`, :class:`.TetrahedralGridDataset`]
            Real part of refractive index. Must be larger than or equal to one.
        dn_dwvl : Union[:class:`.SpatialDataArray`, :class:`.TriangularGridDataset`, :class:`.TetrahedralGridDataset`]
            Derivative of the refractive index with wavelength (1/um). Must be negative.
        freq : float
            Frequency at which ``n`` and ``dn_dwvl`` are sampled.
        interp_method : :class:`.InterpMethod`, optional
            Interpolation method to obtain permittivity values that are not supplied
            at the Yee grids.

        Returns
        -------
        :class:`.CustomSellmeier`
            Single-pole Sellmeier medium with the prvoided refractive index and index dispersion
            valuesat at the prvoided frequency.
        """

        if not _check_same_coordinates(n, dn_dwvl):
            raise ValidationError("'n' and'dn_dwvl' must have the same dimension.")
        if np.any(_get_numpy_array(dn_dwvl) >= 0):
            raise ValidationError("Dispersion ``dn_dwvl`` must be smaller than zero.")
        if np.any(_get_numpy_array(n) < 1):
            raise ValidationError("Refractive index ``n`` cannot be smaller than one.")
        return cls(
            coeffs=cls._from_dispersion_to_coeffs(n, freq, dn_dwvl),
            interp_method=interp_method,
            **kwargs,
        )

    def _sel_custom_data_inside(self, bounds: Bound) -> Self:
        """Return a new custom medium that contains the minimal amount data necessary to cover
        a spatial region defined by ``bounds``.


        Parameters
        ----------
        bounds : tuple[float, float, float], tuple[float, float float]
            Min and max bounds packaged as ``(minx, miny, minz), (maxx, maxy, maxz)``.

        Returns
        -------
        CustomSellmeier
            CustomSellmeier with reduced data.
        """
        coeffs_reduced = []
        for b_coeff, c_coeff in self.coeffs:
            if not b_coeff.does_cover(bounds=bounds):
                log.warning(
                    "Sellmeier B coeff spatial data array does not fully cover the requested region."
                )

            if not c_coeff.does_cover(bounds=bounds):
                log.warning(
                    "Sellmeier C coeff spatial data array does not fully cover the requested region."
                )

            coeffs_reduced.append((b_coeff.sel_inside(bounds), c_coeff.sel_inside(bounds)))

        return self.updated_copy(coeffs=tuple(coeffs_reduced))

    def _compute_derivatives(self, derivative_info: DerivativeInfo) -> AutogradFieldMap:
        """Adjoint derivatives for CustomSellmeier via analytic chain rule.

        Uses the complex permittivity derivative aggregated over spatial dims and
        applies frequency-dependent weights per Sellmeier term.
        """

        if len(self.coeffs) == 0:
            return {}

        # accumulate complex-valued sensitivity across xyz using B's grid as reference
        ref = self.coeffs[0][0]
        dJ = self._sum_complex_eps_sensitivity(derivative_info, spatial_ref=ref)

        # prepare gradients map
        grads: AutogradFieldMap = {}

        # iterate coefficients and requested paths
        for i, (B_da, C_da) in enumerate(self.coeffs):
            need_B = ("coeffs", i, 0) in derivative_info.paths
            need_C = ("coeffs", i, 1) in derivative_info.paths
            if not (need_B or need_C):
                continue

            Bv = np.array(B_da.values, dtype=float)
            Cv = np.array(C_da.values, dtype=float)

            gB = 0.0 if not need_B else np.zeros_like(Bv, dtype=float)
            gC = 0.0 if not need_C else np.zeros_like(Cv, dtype=float)

            if need_B:
                gB = gB + self._sum_over_freqs(
                    derivative_info.frequencies,
                    dJ,
                    weight_fn=lambda f, Cv=Cv: Sellmeier._w_B(f, Cv),
                )
            if need_C:
                gC = gC + self._sum_over_freqs(
                    derivative_info.frequencies,
                    dJ,
                    weight_fn=lambda f, Bv=Bv, Cv=Cv: Sellmeier._w_C(f, Bv, Cv),
                )

            if need_B:
                grads[("coeffs", i, 0)] = gB
            if need_C:
                grads[("coeffs", i, 1)] = gC

        return grads


class CustomLorentz(CustomDispersiveMedium, Lorentz):
    """A spatially varying dispersive medium described by the Lorentz model.

    Notes
    -----

        The frequency-dependence of the complex-valued permittivity is described by:

        .. math::

            \\epsilon(f) = \\epsilon_\\infty + \\sum_i
            \\frac{\\Delta\\epsilon_i f_i^2}{f_i^2 - 2jf\\delta_i - f^2}

    Example
    -------
    >>> x = np.linspace(-1, 1, 5)
    >>> y = np.linspace(-1, 1, 6)
    >>> z = np.linspace(-1, 1, 7)
    >>> coords = dict(x=x, y=y, z=z)
    >>> eps_inf = SpatialDataArray(np.ones((5, 6, 7)), coords=coords)
    >>> d_epsilon = SpatialDataArray(np.random.random((5, 6, 7)), coords=coords)
    >>> f = SpatialDataArray(1+np.random.random((5, 6, 7)), coords=coords)
    >>> delta = SpatialDataArray(np.random.random((5, 6, 7)), coords=coords)
    >>> lorentz_medium = CustomLorentz(eps_inf=eps_inf, coeffs=[(d_epsilon,f,delta),])
    >>> eps = lorentz_medium.eps_model(200e12)

    See Also
    --------

    :class:`CustomPoleResidue`:
        A spatially varying dispersive medium described by the pole-residue pair model.

    **Notebooks**
        * `Fitting dispersive material models <../../notebooks/Fitting.html>`_

    **Lectures**
        * `Modeling dispersive material in FDTD <https://www.flexcompute.com/fdtd101/Lecture-5-Modeling-dispersive-material-in-FDTD/>`_
    """

    eps_inf: CustomSpatialDataTypeAnnotated = Field(
        title="Epsilon at Infinity",
        description="Relative permittivity at infinite frequency (:math:`\\epsilon_\\infty`).",
        json_schema_extra={"units": PERMITTIVITY},
    )

    coeffs: tuple[
        tuple[
            CustomSpatialDataTypeAnnotated,
            CustomSpatialDataTypeAnnotated,
            CustomSpatialDataTypeAnnotated,
        ],
        ...,
    ] = Field(
        title="Coefficients",
        description="List of (:math:`\\Delta\\epsilon_i, f_i, \\delta_i`) values for model.",
        json_schema_extra={"units": (PERMITTIVITY, HERTZ, HERTZ)},
    )

    _no_nans = validate_no_nans("eps_inf", "coeffs")
    _warn_if_none = CustomDispersiveMedium._warn_if_data_none("coeffs")

    @field_validator("eps_inf")
    @classmethod
    def _eps_inf_positive(cls, val: CustomSpatialDataType) -> CustomSpatialDataType:
        """eps_inf must be positive"""
        if not CustomDispersiveMedium._validate_isreal_dataarray(val):
            raise SetupError("'eps_inf' must be real.")
        if np.any(_get_numpy_array(val) < 0):
            raise SetupError("'eps_inf' must be positive.")
        return val

    @field_validator("coeffs")
    @classmethod
    def _coeffs_unequal_f_delta(
        cls, val: tuple[tuple[CustomSpatialDataType, CustomSpatialDataType], ...]
    ) -> tuple[tuple[CustomSpatialDataType, CustomSpatialDataType], ...]:
        """f and delta cannot be exactly the same.
        Not needed for now because we have a more strict
        validator `_coeffs_delta_all_smaller_or_larger_than_fi`.
        """
        return val

    def _coeffs_correct_shape(self) -> Self:
        """coeffs must have consistent shape."""
        val = self.coeffs
        for de, f, delta in val:
            if (
                not _check_same_coordinates(de, self.eps_inf)
                or not _check_same_coordinates(f, self.eps_inf)
                or not _check_same_coordinates(delta, self.eps_inf)
            ):
                raise SetupError(
                    "All terms in 'coeffs' must have the same coordinates; "
                    "The coordinates must also be consistent with 'eps_inf'."
                )
            if not CustomDispersiveMedium._validate_isreal_dataarray_tuple((de, f, delta)):
                raise SetupError("All terms in 'coeffs' must be real.")
        return self

    def _validate_coeffs_shape(self) -> Self:
        return self._coeffs_correct_shape()

    @field_validator("coeffs")
    @classmethod
    def _coeffs_delta_all_smaller_or_larger_than_fi(
        cls,
        val: tuple[tuple[CustomSpatialDataType, CustomSpatialDataType, CustomSpatialDataType], ...],
    ) -> tuple[tuple[CustomSpatialDataType, CustomSpatialDataType, CustomSpatialDataType], ...]:
        """We restrict either all f**2>delta**2 or all f**2<delta**2 for now."""
        for _, f, delta in val:
            f2 = f**2
            delta2 = delta**2
            if not (Lorentz._all_larger(f2, delta2) or Lorentz._all_larger(delta2, f2)):
                raise SetupError(
                    "Coefficients in 'coeffs' are restricted to have "
                    "either all 'delta**2'<'f**2' or all 'delta**2'>'f**2'."
                )
        return val

    def _passivity_validation(self) -> Self:
        """Assert passive medium if ``allow_gain`` is False."""
        val = self.coeffs
        allow_gain = self.allow_gain
        for del_ep, _, delta in val:
            if np.any(_get_numpy_array(delta) < 0):
                raise ValidationError("For stable medium, 'delta_i' must be non-negative.")
            if not allow_gain and np.any(_get_numpy_array(del_ep) < 0):
                raise ValidationError(
                    "For passive medium, 'Delta epsilon_i' must be non-negative. "
                    "To simulate a gain medium, please set 'allow_gain=True'. "
                    "Caution: simulations with a gain medium are unstable, "
                    "and are likely to diverge."
                )
        return self

    @cached_property
    def is_spatially_uniform(self) -> bool:
        """Whether the medium is spatially uniform."""
        if not self.eps_inf.is_uniform:
            return False
        for coeffs in self.coeffs:
            for coeff in coeffs:
                if not coeff.is_uniform:
                    return False
        return True

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
        eps = Lorentz.eps_model(self, frequency)
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
        CustomLorentz
            CustomLorentz with reduced data.
        """
        if not self.eps_inf.does_cover(bounds=bounds):
            log.warning("Eps inf spatial data array does not fully cover the requested region.")
        eps_inf_reduced = self.eps_inf.sel_inside(bounds=bounds)
        coeffs_reduced = []
        for de, f, delta in self.coeffs:
            if not de.does_cover(bounds=bounds):
                log.warning(
                    "Lorentz 'de' spatial data array does not fully cover the requested region."
                )

            if not f.does_cover(bounds=bounds):
                log.warning(
                    "Lorentz 'f' spatial data array does not fully cover the requested region."
                )

            if not delta.does_cover(bounds=bounds):
                log.warning(
                    "Lorentz 'delta' spatial data array does not fully cover the requested region."
                )

            coeffs_reduced.append(
                (de.sel_inside(bounds), f.sel_inside(bounds), delta.sel_inside(bounds))
            )

        return self.updated_copy(eps_inf=eps_inf_reduced, coeffs=tuple(coeffs_reduced))

    def _compute_derivatives(self, derivative_info: DerivativeInfo) -> AutogradFieldMap:
        """Adjoint derivatives for CustomLorentz via analytic chain rule."""

        # complex epsilon sensitivity over xyz aligned to eps_inf grid
        dJ = self._sum_complex_eps_sensitivity(derivative_info, spatial_ref=self.eps_inf)

        grads: AutogradFieldMap = {}

        # eps_inf path
        if ("eps_inf",) in derivative_info.paths:
            grads[("eps_inf",)] = self._sum_real_over_freqs(dJ, derivative_info.frequencies)

        # per-coefficient contributions
        for i, (de_da, f0_da, dl_da) in enumerate(self.coeffs):
            need_de = ("coeffs", i, 0) in derivative_info.paths
            need_f0 = ("coeffs", i, 1) in derivative_info.paths
            need_dl = ("coeffs", i, 2) in derivative_info.paths
            if not (need_de or need_f0 or need_dl):
                continue

            de = np.array(de_da.values, dtype=float)
            f0 = np.array(f0_da.values, dtype=float)
            dl = np.array(dl_da.values, dtype=float)

            g_de = 0.0 if not need_de else np.zeros_like(de, dtype=float)
            g_f0 = 0.0 if not need_f0 else np.zeros_like(f0, dtype=float)
            g_dl = 0.0 if not need_dl else np.zeros_like(dl, dtype=float)

            if need_de:
                g_de = g_de + self._sum_over_freqs(
                    derivative_info.frequencies,
                    dJ,
                    weight_fn=lambda f, f0=f0, dl=dl: Lorentz._w_de(f, f0, dl),
                )
            if need_f0:
                # d/d f0 of (de f0^2 / den) = (2 de f0 (den - f0^2)) / den^2
                g_f0 = g_f0 + self._sum_over_freqs(
                    derivative_info.frequencies,
                    dJ,
                    weight_fn=lambda f, de=de, f0=f0, dl=dl: Lorentz._w_f0(f, de, f0, dl),
                )
            if need_dl:
                # d/d delta of (de f0^2 / den) = (2 j f de f0^2) / den^2
                g_dl = g_dl + self._sum_over_freqs(
                    derivative_info.frequencies,
                    dJ,
                    weight_fn=lambda f, de=de, f0=f0, dl=dl: Lorentz._w_delta(f, de, f0, dl),
                )

            if need_de:
                grads[("coeffs", i, 0)] = g_de
            if need_f0:
                grads[("coeffs", i, 1)] = g_f0
            if need_dl:
                grads[("coeffs", i, 2)] = g_dl

        return grads


class CustomDrude(CustomDispersiveMedium, Drude):
    """A spatially varying dispersive medium described by the Drude model.


    Notes
    -----

        The frequency-dependence of the complex-valued permittivity is described by:

        .. math::

            \\epsilon(f) = \\epsilon_\\infty - \\sum_i
            \\frac{ f_i^2}{f^2 + jf\\delta_i}

    Example
    -------
    >>> x = np.linspace(-1, 1, 5)
    >>> y = np.linspace(-1, 1, 6)
    >>> z = np.linspace(-1, 1, 7)
    >>> coords = dict(x=x, y=y, z=z)
    >>> eps_inf = SpatialDataArray(np.ones((5, 6, 7)), coords=coords)
    >>> f1 = SpatialDataArray(np.random.random((5, 6, 7)), coords=coords)
    >>> delta1 = SpatialDataArray(np.random.random((5, 6, 7)), coords=coords)
    >>> drude_medium = CustomDrude(eps_inf=eps_inf, coeffs=[(f1,delta1),])
    >>> eps = drude_medium.eps_model(200e12)

    See Also
    --------

    :class:`Drude`:
        A dispersive medium described by the Drude model.

    **Notebooks**
        * `Fitting dispersive material models <../../notebooks/Fitting.html>`_

    **Lectures**
        * `Modeling dispersive material in FDTD <https://www.flexcompute.com/fdtd101/Lecture-5-Modeling-dispersive-material-in-FDTD/>`_
    """

    eps_inf: CustomSpatialDataTypeAnnotated = Field(
        title="Epsilon at Infinity",
        description="Relative permittivity at infinite frequency (:math:`\\epsilon_\\infty`).",
        json_schema_extra={"units": PERMITTIVITY},
    )

    coeffs: tuple[tuple[CustomSpatialDataTypeAnnotated, CustomSpatialDataTypeAnnotated], ...] = (
        Field(
            title="Coefficients",
            description="List of (:math:`f_i, \\delta_i`) values for model.",
            json_schema_extra={"units": (HERTZ, HERTZ)},
        )
    )

    _no_nans = validate_no_nans("eps_inf", "coeffs")
    _warn_if_none = CustomDispersiveMedium._warn_if_data_none("coeffs")

    @field_validator("eps_inf")
    @classmethod
    def _eps_inf_positive(cls, val: TracedPositiveFloat) -> TracedPositiveFloat:
        """eps_inf must be positive"""
        if not CustomDispersiveMedium._validate_isreal_dataarray(val):
            raise SetupError("'eps_inf' must be real.")
        if np.any(_get_numpy_array(val) < 0):
            raise SetupError("'eps_inf' must be positive.")
        return val

    @model_validator(mode="after")
    def _run_after_validators(self) -> Self:
        """Run post-init validations in an explicit, dependency-aware order."""
        super()._run_after_validators()
        self._coeffs_correct_shape_and_sign()
        return self

    def _coeffs_correct_shape_and_sign(self) -> Self:
        """coeffs must have consistent shape and sign."""
        val = self.coeffs
        for f, delta in val:
            if not _check_same_coordinates(f, self.eps_inf) or not _check_same_coordinates(
                delta, self.eps_inf
            ):
                self._raise_validation_error_at_loc(
                    SetupError(
                        "All terms in 'coeffs' must have the same coordinates; "
                        "The coordinates must also be consistent with 'eps_inf'."
                    ),
                    "coeffs",
                )
            if not CustomDispersiveMedium._validate_isreal_dataarray_tuple((f, delta)):
                self._raise_validation_error_at_loc(
                    SetupError("All terms in 'coeffs' must be real."), "coeffs"
                )
            if np.any(_get_numpy_array(delta) <= 0):
                self._raise_validation_error_at_loc(
                    SetupError("For stable medium, 'delta' must be positive."), "coeffs"
                )
        return self

    @cached_property
    def is_spatially_uniform(self) -> bool:
        """Whether the medium is spatially uniform."""
        if not self.eps_inf.is_uniform:
            return False
        for coeffs in self.coeffs:
            for coeff in coeffs:
                if not coeff.is_uniform:
                    return False
        return True

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
        eps = Drude.eps_model(self, frequency)
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
        CustomDrude
            CustomDrude with reduced data.
        """
        if not self.eps_inf.does_cover(bounds=bounds):
            log.warning("Eps inf spatial data array does not fully cover the requested region.")
        eps_inf_reduced = self.eps_inf.sel_inside(bounds=bounds)
        coeffs_reduced = []
        for f, delta in self.coeffs:
            if not f.does_cover(bounds=bounds):
                log.warning(
                    "Drude 'f' spatial data array does not fully cover the requested region."
                )

            if not delta.does_cover(bounds=bounds):
                log.warning(
                    "Drude 'delta' spatial data array does not fully cover the requested region."
                )

            coeffs_reduced.append((f.sel_inside(bounds), delta.sel_inside(bounds)))

        return self.updated_copy(eps_inf=eps_inf_reduced, coeffs=tuple(coeffs_reduced))

    def _compute_derivatives(self, derivative_info: DerivativeInfo) -> AutogradFieldMap:
        """Adjoint derivatives for CustomDrude via analytic chain rule."""

        dJ = self._sum_complex_eps_sensitivity(derivative_info, spatial_ref=self.eps_inf)

        grads: AutogradFieldMap = {}
        if ("eps_inf",) in derivative_info.paths:
            grads[("eps_inf",)] = self._sum_real_over_freqs(dJ, derivative_info.frequencies)

        for i, (fp_da, dl_da) in enumerate(self.coeffs):
            need_fp = ("coeffs", i, 0) in derivative_info.paths
            need_dl = ("coeffs", i, 1) in derivative_info.paths
            if not (need_fp or need_dl):
                continue

            fp = np.array(fp_da.values, dtype=float)
            dl = np.array(dl_da.values, dtype=float)

            g_fp = 0.0 if not need_fp else np.zeros_like(fp, dtype=float)
            g_dl = 0.0 if not need_dl else np.zeros_like(dl, dtype=float)

            if need_fp:
                g_fp = g_fp + self._sum_over_freqs(
                    derivative_info.frequencies,
                    dJ,
                    weight_fn=lambda f, fp=fp, dl=dl: Drude._w_fp(f, fp, dl),
                )
            if need_dl:
                g_dl = g_dl + self._sum_over_freqs(
                    derivative_info.frequencies,
                    dJ,
                    weight_fn=lambda f, fp=fp, dl=dl: Drude._w_delta(f, fp, dl),
                )

            if need_fp:
                grads[("coeffs", i, 0)] = g_fp
            if need_dl:
                grads[("coeffs", i, 1)] = g_dl

        return grads


class CustomDebye(CustomDispersiveMedium, Debye):
    """A spatially varying dispersive medium described by the Debye model.

    Notes
    -----

        The frequency-dependence of the complex-valued permittivity is described by:

        .. math::

            \\epsilon(f) = \\epsilon_\\infty + \\sum_i
            \\frac{\\Delta\\epsilon_i}{1 - jf\\tau_i}

    Example
    -------
    >>> x = np.linspace(-1, 1, 5)
    >>> y = np.linspace(-1, 1, 6)
    >>> z = np.linspace(-1, 1, 7)
    >>> coords = dict(x=x, y=y, z=z)
    >>> eps_inf = SpatialDataArray(1+np.random.random((5, 6, 7)), coords=coords)
    >>> eps1 = SpatialDataArray(np.random.random((5, 6, 7)), coords=coords)
    >>> tau1 = SpatialDataArray(np.random.random((5, 6, 7)), coords=coords)
    >>> debye_medium = CustomDebye(eps_inf=eps_inf, coeffs=[(eps1,tau1),])
    >>> eps = debye_medium.eps_model(200e12)

    See Also
    --------

    :class:`Debye`
        A dispersive medium described by the Debye model.

    **Notebooks**
        * `Fitting dispersive material models <../../notebooks/Fitting.html>`_

    **Lectures**
        * `Modeling dispersive material in FDTD <https://www.flexcompute.com/fdtd101/Lecture-5-Modeling-dispersive-material-in-FDTD/>`_
    """

    eps_inf: CustomSpatialDataTypeAnnotated = Field(
        title="Epsilon at Infinity",
        description="Relative permittivity at infinite frequency (:math:`\\epsilon_\\infty`).",
        json_schema_extra={"units": PERMITTIVITY},
    )

    coeffs: tuple[tuple[CustomSpatialDataTypeAnnotated, CustomSpatialDataTypeAnnotated], ...] = (
        Field(
            title="Coefficients",
            description="List of (:math:`\\Delta\\epsilon_i, \\tau_i`) values for model.",
            json_schema_extra={"units": (PERMITTIVITY, SECOND)},
        )
    )

    _no_nans = validate_no_nans("eps_inf", "coeffs")
    _warn_if_none = CustomDispersiveMedium._warn_if_data_none("coeffs")

    @field_validator("eps_inf")
    @classmethod
    def _eps_inf_positive(cls, val: TracedPositiveFloat) -> TracedPositiveFloat:
        """eps_inf must be positive"""
        if not CustomDispersiveMedium._validate_isreal_dataarray(val):
            raise SetupError("'eps_inf' must be real.")
        if np.any(_get_numpy_array(val) < 0):
            raise SetupError("'eps_inf' must be positive.")
        return val

    def _coeffs_correct_shape(self) -> Self:
        """coeffs must have consistent shape."""
        val = self.coeffs
        for de, tau in val:
            if not _check_same_coordinates(de, self.eps_inf) or not _check_same_coordinates(
                tau, self.eps_inf
            ):
                raise SetupError(
                    "All terms in 'coeffs' must have the same coordinates; "
                    "The coordinates must also be consistent with 'eps_inf'."
                )
            if not CustomDispersiveMedium._validate_isreal_dataarray_tuple((de, tau)):
                raise SetupError("All terms in 'coeffs' must be real.")
        return self

    def _validate_coeffs_shape(self) -> Self:
        return self._coeffs_correct_shape()

    @field_validator("coeffs")
    @classmethod
    def _coeffs_tau_all_sufficient_positive(
        cls, val: tuple[tuple[CustomSpatialDataType, CustomSpatialDataType], ...]
    ) -> tuple[tuple[CustomSpatialDataType, CustomSpatialDataType], ...]:
        """We restrict either all tau is sufficently greater than 0."""
        for _, tau in val:
            if np.any(_get_numpy_array(tau) < 1 / 2 / np.pi / LARGEST_FP_NUMBER):
                raise SetupError(
                    "Coefficients 'tau_i' are restricted to be sufficiently greater than 0."
                )
        return val

    def _compute_derivatives(self, derivative_info: DerivativeInfo) -> AutogradFieldMap:
        """Adjoint derivatives for CustomDebye via analytic chain rule."""

        dJ = self._sum_complex_eps_sensitivity(derivative_info, spatial_ref=self.eps_inf)

        grads: AutogradFieldMap = {}
        if ("eps_inf",) in derivative_info.paths:
            grads[("eps_inf",)] = self._sum_real_over_freqs(dJ, derivative_info.frequencies)

        for i, (de_da, tau_da) in enumerate(self.coeffs):
            need_de = ("coeffs", i, 0) in derivative_info.paths
            need_tau = ("coeffs", i, 1) in derivative_info.paths
            if not (need_de or need_tau):
                continue

            de = np.array(de_da.values, dtype=float)
            tau = np.array(tau_da.values, dtype=float)

            g_de = 0.0 if not need_de else np.zeros_like(de, dtype=float)
            g_tau = 0.0 if not need_tau else np.zeros_like(tau, dtype=float)

            if need_de:
                g_de = g_de + self._sum_over_freqs(
                    derivative_info.frequencies,
                    dJ,
                    weight_fn=lambda f, tau=tau: Debye._w_de(f, tau),
                )
            if need_tau:
                g_tau = g_tau + self._sum_over_freqs(
                    derivative_info.frequencies,
                    dJ,
                    weight_fn=lambda f, de=de, tau=tau: Debye._w_tau(f, de, tau),
                )

            if need_de:
                grads[("coeffs", i, 0)] = g_de
            if need_tau:
                grads[("coeffs", i, 1)] = g_tau

        return grads

    def _passivity_validation(self) -> Self:
        """Assert passive medium if ``allow_gain`` is False."""
        val = self.coeffs
        allow_gain = self.allow_gain
        for del_ep, tau in val:
            if np.any(_get_numpy_array(tau) <= 0):
                raise SetupError("For stable medium, 'tau_i' must be positive.")
            if not allow_gain and np.any(_get_numpy_array(del_ep) < 0):
                raise ValidationError(
                    "For passive medium, 'Delta epsilon_i' must be non-negative. "
                    "To simulate a gain medium, please set 'allow_gain=True'. "
                    "Caution: simulations with a gain medium are unstable, "
                    "and are likely to diverge."
                )
        return self

    @cached_property
    def is_spatially_uniform(self) -> bool:
        """Whether the medium is spatially uniform."""
        if not self.eps_inf.is_uniform:
            return False
        for coeffs in self.coeffs:
            for coeff in coeffs:
                if not coeff.is_uniform:
                    return False
        return True

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
        eps = Debye.eps_model(self, frequency)
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
        CustomDebye
            CustomDebye with reduced data.
        """
        if not self.eps_inf.does_cover(bounds=bounds):
            log.warning("Eps inf spatial data array does not fully cover the requested region.")
        eps_inf_reduced = self.eps_inf.sel_inside(bounds=bounds)
        coeffs_reduced = []
        for de, tau in self.coeffs:
            if not de.does_cover(bounds=bounds):
                log.warning(
                    "Debye 'f' spatial data array does not fully cover the requested region."
                )

            if not tau.does_cover(bounds=bounds):
                log.warning(
                    "Debye 'tau' spatial data array does not fully cover the requested region."
                )

            coeffs_reduced.append((de.sel_inside(bounds), tau.sel_inside(bounds)))

        return self.updated_copy(eps_inf=eps_inf_reduced, coeffs=tuple(coeffs_reduced))
