"""Spatially varying dispersive medium models."""

from __future__ import annotations

from abc import ABC
from functools import partial
from typing import TYPE_CHECKING, Any

import autograd.numpy as np
from pydantic import Field, model_validator

from tidy3d.components.base import cached_property
from tidy3d.components.data.data_array import SpatialDataArray
from tidy3d.components.data.utils import (
    CustomSpatialDataTypeAnnotated,
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

    from tidy3d.compat import Self
    from tidy3d.components.autograd.derivative_terms import Compiled
    from tidy3d.components.autograd.types import PathType
    from tidy3d.components.data.utils import CustomSpatialDataType
    from tidy3d.components.grid.grid import Coords
    from tidy3d.components.medium.base import (
        AbstractMedium,
    )
    from tidy3d.components.medium.derivative_terms import MediumTarget
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

    def _derivative_terms(self, paths: list[PathType], target: MediumTarget) -> Compiled:
        """One term: project onto the parameter grid, sum components, contract with d(eps)/dp."""

        from flexcompute.core._migration.em.derivatives import (
            ContractMaterialDerivative,
            DispersiveMaterial,
            FieldProductKernel,
            SumComponents,
        )
        from flexcompute.core._migration.em.numerical.raw.medium import (
            DISPERSIVE_TERM_PARAMETERS,
        )

        from tidy3d.components.autograd.derivative_terms import Binding, Compiled
        from tidy3d.components.autograd.utils import get_static
        from tidy3d.components.medium.derivative_terms import projection, restore_output

        names = DISPERSIVE_TERM_PARAMETERS[self._derivative_model]
        eps_inf = getattr(self, "eps_inf", None)
        coefficients = getattr(self, self._traced_indexed_root)

        def on_grid(data: Any) -> Any:
            # a copy: a traced value can be the caller's own parameter array, which the
            # program's array type would otherwise mark read-only
            return np.array(get_static(data._spatially_sorted.values))

        material = DispersiveMaterial(
            model=self._derivative_model,
            epsilon_infinity=None if eps_inf is None else on_grid(eps_inf),
            terms=tuple(tuple(on_grid(value) for value in term) for term in coefficients),
        )
        selected = [
            (("epsilon_infinity",), eps_inf)
            if path == ("eps_inf",)
            else (("term", path[1], names[path[2]]), coefficients[path[1]][path[2]])
            for path in paths
        ]
        basis = (coefficients[0][0] if eps_inf is None else eps_inf)._spatially_sorted
        kernel = FieldProductKernel(
            monitor=target.monitor,
            components=target.components,
            reductions=(
                projection(target, basis, target.interpolation or self.interp_method),
                SumComponents(),
                ContractMaterialDerivative(
                    material=material, parameters=tuple(dict.fromkeys(key for key, _ in selected))
                ),
            ),
        )
        bindings = tuple(
            Binding.one(
                (*target.key, *path), target.key, partial(restore_output, key=key, data=data)
            )
            for path, (key, data) in zip(paths, selected, strict=True)
        )
        return Compiled(terms=((target.key, kernel),), bindings=bindings)

    def _resolve_autograd_route(self, field_path: tuple[Any, ...]) -> None:
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
            default=(),
            title="Poles",
            description="Tuple of complex-valued (:math:`a_i, c_i`) poles for the model.",
            json_schema_extra={"units": (RADPERSEC, RADPERSEC)},
        )
    )

    _no_nans = validate_no_nans("eps_inf", "poles")
    _warn_if_none = CustomDispersiveMedium._warn_if_data_none("poles")

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

    def _passivity_validation(self) -> Self:
        """Assert passive medium if `allow_gain` is False."""
        # Thin check that calls the core rule with the code of the custom media, which has no
        # location (``Sellmeier`` reports at ``coeffs``). It runs here, and not only on the
        # construction path, so that it comes before the modulation check, as before.
        from flexcompute.core._migration.em.material.medium.validation import (
            CUSTOM_PASSIVE_SELLMEIER_B,
            validate_sellmeier_passivity,
        )

        from tidy3d.em.translate.components import _present

        _present(
            self,
            validate_sellmeier_passivity(
                [_get_numpy_array(b) for b, _ in self.coeffs],
                self.allow_gain,
                (),
                CUSTOM_PASSIVE_SELLMEIER_B,
            ),
        )
        return self

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
