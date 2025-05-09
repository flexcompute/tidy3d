from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union

import pydantic.v1 as pd

from tidy3d.components.base import Tidy3dBaseModel
from tidy3d.components.data.utils import (
    CustomSpatialDataType,
)
from tidy3d.components.parameter_perturbation import (
    IndexPerturbation,
    ParameterPerturbation,
    PermittivityPerturbation,
)
from tidy3d.components.types import TYPE_TAG_STR, InterpMethod
from tidy3d.components.validators import _warn_potential_error, validate_parameter_perturbation
from tidy3d.constants import CONDUCTIVITY, EPSILON_0, PERMITTIVITY, RADPERSEC
from tidy3d.exceptions import SetupError

from .base import AbstractCustomMedium, AbstractMedium
from .dispersionless import Medium
from .dispersive import PoleResidue


class AbstractPerturbationMedium(ABC, Tidy3dBaseModel):
    """Abstract class for medium perturbation."""

    subpixel: bool = pd.Field(
        True,
        title="Subpixel averaging",
        description="This value will be transferred to the resulting custom medium. That is, "
        "if ``True``, the subpixel averaging will be applied to the custom medium. The type "
        "of subpixel averaging method applied is specified in ``Simulation``'s field ``subpixel``. "
        "If the resulting medium is not a custom medium (no perturbations), this field does not "
        "have an effect.",
    )

    perturbation_spec: Optional[Union[PermittivityPerturbation, IndexPerturbation]] = pd.Field(
        None,
        title="Perturbation Spec",
        description="Specification of medium perturbation as one of predefined types.",
        discriminator=TYPE_TAG_STR,
    )

    @abstractmethod
    def perturbed_copy(
        self,
        temperature: CustomSpatialDataType = None,
        electron_density: CustomSpatialDataType = None,
        hole_density: CustomSpatialDataType = None,
        interp_method: InterpMethod = "linear",
    ) -> Union[AbstractMedium, AbstractCustomMedium]:
        """Sample perturbations on provided heat and/or charge data and create a custom medium.
        Any of ``temperature``, ``electron_density``, and ``hole_density`` can be ``None``.
        If all passed arguments are ``None`` then a non-custom medium is returned.
        All provided fields must have identical coords.

        Parameters
        ----------
        temperature : Union[
                :class:`.SpatialDataArray`,
                :class:`.TriangularGridDataset`,
                :class:`.TetrahedralGridDataset`,
            ] = None
            Temperature field data.
        electron_density : Union[
                :class:`.SpatialDataArray`,
                :class:`.TriangularGridDataset`,
                :class:`.TetrahedralGridDataset`,
            ] = None
            Electron density field data.
        hole_density : Union[
                :class:`.SpatialDataArray`,
                :class:`.TriangularGridDataset`,
                :class:`.TetrahedralGridDataset`,
            ] = None
            Hole density field data.
        interp_method : :class:`.InterpMethod`, optional
            Interpolation method to obtain heat and/or charge values that are not supplied
            at the Yee grids.

        Returns
        -------
        Union[AbstractMedium, AbstractCustomMedium]
            Medium specification after application of heat and/or charge data.
        """

    @classmethod
    def from_unperturbed(
        cls,
        medium: Union[Medium, DispersiveMedium],
        subpixel: bool = True,
        perturbation_spec: Union[PermittivityPerturbation, IndexPerturbation] = None,
        **kwargs,
    ) -> AbstractPerturbationMedium:
        """Construct a medium with pertubation models from an unpertubed one.

        Parameters
        ----------
        medium : Union[
                :class:`.Medium`,
                :class:`.DispersiveMedium`,
            ]
            A medium with no perturbation models.
        subpixel : bool = True
            Subpixel averaging of derivative custom medium.
        perturbation_spec : Union[
                :class:`.PermittivityPerturbation`,
                :class:`.IndexPerturbation`,
            ] = None
            Perturbation model specification.

        Returns
        -------
        :class:`.AbstractPerturbationMedium`
            Resulting medium with perturbation model.
        """

        new_dict = medium.dict(
            exclude={
                "type",
            }
        )

        new_dict["perturbation_spec"] = perturbation_spec
        new_dict["subpixel"] = subpixel

        new_dict.update(kwargs)

        return cls.parse_obj(new_dict)


class PerturbationMedium(Medium, AbstractPerturbationMedium):
    """Dispersionless medium with perturbations. Perturbation model can be defined either directly
    through providing ``permittivity_perturbation`` and ``conductivity_perturbation`` or via
    providing a specific perturbation model (:class:`PermittivityPerturbation`,
    :class:`IndexPerturbation`) as ``perturbaiton_spec``.

    Example
    -------
    >>> from tidy3d import ParameterPerturbation, LinearHeatPerturbation
    >>> dielectric = PerturbationMedium(
    ...     permittivity=4.0,
    ...     permittivity_perturbation=ParameterPerturbation(
    ...         heat=LinearHeatPerturbation(temperature_ref=300, coeff=0.0001),
    ...     ),
    ...     name='my_medium',
    ... )
    """

    permittivity_perturbation: Optional[ParameterPerturbation] = pd.Field(
        None,
        title="Permittivity Perturbation",
        description="List of heat and/or charge perturbations to permittivity.",
        units=PERMITTIVITY,
    )

    conductivity_perturbation: Optional[ParameterPerturbation] = pd.Field(
        None,
        title="Permittivity Perturbation",
        description="List of heat and/or charge perturbations to permittivity.",
        units=CONDUCTIVITY,
    )

    _permittivity_perturbation_validator = validate_parameter_perturbation(
        "permittivity_perturbation",
        "permittivity",
        allowed_real_range=[(1.0, None)],
        allowed_imag_range=[None],
        allowed_complex=False,
    )

    _conductivity_perturbation_validator = validate_parameter_perturbation(
        "conductivity_perturbation",
        "conductivity",
        allowed_real_range=[(0.0, None)],
        allowed_imag_range=[None],
        allowed_complex=False,
    )

    @pd.root_validator(pre=True)
    def _check_overdefining(cls, values):
        """Check that perturbation model is provided either directly or through
        ``perturbation_spec``, but not both.
        """

        perm_p = values.get("permittivity_perturbation") is not None
        cond_p = values.get("conductivity_perturbation") is not None
        p_spec = values.get("perturbation_spec") is not None

        if p_spec and (perm_p or cond_p):
            raise SetupError(
                "Must provide perturbation model either as 'perturbation_spec' or as "
                "'permittivity_perturbation' and 'conductivity_perturbation', "
                "but not in both ways simultaneously."
            )

        return values

    @pd.root_validator(skip_on_failure=True)
    def _check_perturbation_spec_ranges(cls, values):
        """Check perturbation ranges if defined as ``perturbation_spec``."""
        p_spec = values["perturbation_spec"]
        if p_spec is None:
            return values

        perm = values["permittivity"]
        cond = values["conductivity"]

        if isinstance(p_spec, IndexPerturbation):
            eps_complex = Medium._eps_model(
                permittivity=perm, conductivity=cond, frequency=p_spec.freq
            )
            n, k = Medium.eps_complex_to_nk(eps_c=eps_complex)
            delta_eps_range, delta_sigma_range = p_spec._delta_eps_delta_sigma_ranges(n, k)
        elif isinstance(p_spec, PermittivityPerturbation):
            delta_eps_range, delta_sigma_range = p_spec._delta_eps_delta_sigma_ranges()
        else:
            raise SetupError("Unknown type of 'perturbation_spec'.")

        _warn_potential_error(
            field_name="permittivity",
            base_value=perm,
            val_change_range=delta_eps_range,
            allowed_real_range=(1.0, None),
            allowed_imag_range=None,
        )

        _warn_potential_error(
            field_name="conductivity",
            base_value=cond,
            val_change_range=delta_sigma_range,
            allowed_real_range=(0.0, None),
            allowed_imag_range=None,
        )
        return values

    def perturbed_copy(
        self,
        temperature: CustomSpatialDataType = None,
        electron_density: CustomSpatialDataType = None,
        hole_density: CustomSpatialDataType = None,
        interp_method: InterpMethod = "linear",
    ) -> Union[Medium, CustomMedium]:
        """Sample perturbations on provided heat and/or charge data and return 'CustomMedium'.
        Any of temperature, electron_density, and hole_density can be 'None'. If all passed
        arguments are 'None' then a 'Medium' object is returned. All provided fields must have
        identical coords.

        Parameters
        ----------
        temperature : Union[
                :class:`.SpatialDataArray`,
                :class:`.TriangularGridDataset`,
                :class:`.TetrahedralGridDataset`,
            ] = None
            Temperature field data.
        electron_density : Union[
                :class:`.SpatialDataArray`,
                :class:`.TriangularGridDataset`,
                :class:`.TetrahedralGridDataset`,
            ] = None
            Electron density field data.
        hole_density : Union[
                :class:`.SpatialDataArray`,
                :class:`.TriangularGridDataset`,
                :class:`.TetrahedralGridDataset`,
            ] = None
            Hole density field data.
        interp_method : :class:`.InterpMethod`, optional
            Interpolation method to obtain heat and/or charge values that are not supplied
            at the Yee grids.

        Returns
        -------
        Union[Medium, CustomMedium]
            Medium specification after application of heat and/or charge data.
        """

        new_dict = self.dict(
            exclude={
                "permittivity_perturbation",
                "conductivity_perturbation",
                "perturbation_spec",
                "type",
            }
        )

        if all(x is None for x in [temperature, electron_density, hole_density]):
            new_dict.pop("subpixel")
            return Medium.parse_obj(new_dict)

        permittivity_field = self.permittivity + ParameterPerturbation._zeros_like(
            temperature, electron_density, hole_density
        )

        delta_eps = None
        delta_sigma = None

        if self.perturbation_spec is not None:
            pspec = self.perturbation_spec
            if isinstance(pspec, PermittivityPerturbation):
                delta_eps, delta_sigma = pspec._sample_delta_eps_delta_sigma(
                    temperature, electron_density, hole_density
                )
            elif isinstance(pspec, IndexPerturbation):
                n, k = self.nk_model(frequency=pspec.freq)
                delta_eps, delta_sigma = pspec._sample_delta_eps_delta_sigma(
                    n, k, temperature, electron_density, hole_density
                )
        else:
            if self.permittivity_perturbation is not None:
                delta_eps = self.permittivity_perturbation.apply_data(
                    temperature, electron_density, hole_density
                )

            if self.conductivity_perturbation is not None:
                delta_sigma = self.conductivity_perturbation.apply_data(
                    temperature, electron_density, hole_density
                )

        if delta_eps is not None:
            permittivity_field = permittivity_field + delta_eps

        conductivity_field = None
        if delta_sigma is not None:
            conductivity_field = self.conductivity + delta_sigma

        new_dict["permittivity"] = permittivity_field
        new_dict["conductivity"] = conductivity_field
        new_dict["interp_method"] = interp_method

        return CustomMedium.parse_obj(new_dict)


class PerturbationPoleResidue(PoleResidue, AbstractPerturbationMedium):
    """A dispersive medium described by the pole-residue pair model with perturbations.
    Perturbation model can be defined either directly
    through providing ``eps_inf_perturbation`` and ``poles_perturbation`` or via
    providing a specific perturbation model (:class:`PermittivityPerturbation`,
    :class:`IndexPerturbation`) as ``perturbaiton_spec``.

    Notes
    -----

        The frequency-dependence of the complex-valued permittivity is described by:

        .. math::

            \\epsilon(\\omega) = \\epsilon_\\infty - \\sum_i
            \\left[\\frac{c_i}{j \\omega + a_i} +
            \\frac{c_i^*}{j \\omega + a_i^*}\\right]

    Example
    -------
    >>> from tidy3d import ParameterPerturbation, LinearHeatPerturbation
    >>> c0_perturbation = ParameterPerturbation(
    ...     heat=LinearHeatPerturbation(temperature_ref=300, coeff=0.0001),
    ... )
    >>> pole_res = PerturbationPoleResidue(
    ...     eps_inf=2.0,
    ...     poles=[((-1+2j), (3+4j)), ((-5+6j), (7+8j))],
    ...     poles_perturbation=[(None, c0_perturbation), (None, None)],
    ... )
    """

    eps_inf_perturbation: Optional[ParameterPerturbation] = pd.Field(
        None,
        title="Perturbation of Epsilon at Infinity",
        description="Perturbations to relative permittivity at infinite frequency "
        "(:math:`\\epsilon_\\infty`).",
        units=PERMITTIVITY,
    )

    poles_perturbation: Optional[
        Tuple[Tuple[Optional[ParameterPerturbation], Optional[ParameterPerturbation]], ...]
    ] = pd.Field(
        None,
        title="Perturbations of Poles",
        description="Perturbations to poles of the model.",
        units=(RADPERSEC, RADPERSEC),
    )

    _eps_inf_perturbation_validator = validate_parameter_perturbation(
        "eps_inf_perturbation",
        "eps_inf",
        allowed_real_range=[(0.0, None)],
        allowed_imag_range=[None],
        allowed_complex=False,
    )

    _poles_perturbation_validator = validate_parameter_perturbation(
        "poles_perturbation",
        "poles",
        allowed_real_range=[(None, 0.0), (None, None)],
        allowed_imag_range=[None, None],
    )

    @pd.root_validator(pre=True)
    def _check_overdefining(cls, values):
        """Check that perturbation model is provided either directly or through
        ``perturbation_spec``, but not both.
        """

        eps_i_p = values.get("eps_inf_perturbation") is not None
        poles_p = values.get("poles_perturbation") is not None
        p_spec = values.get("perturbation_spec") is not None

        if p_spec and (eps_i_p or poles_p):
            raise SetupError(
                "Must provide perturbation model either as 'perturbation_spec' or as "
                "'eps_inf_perturbation' and 'poles_perturbation', "
                "but not in both ways simultaneously."
            )

        return values

    @pd.root_validator(skip_on_failure=True)
    def _check_perturbation_spec_ranges(cls, values):
        """Check perturbation ranges if defined as ``perturbation_spec``."""
        p_spec = values["perturbation_spec"]
        if p_spec is None:
            return values

        eps_inf = values["eps_inf"]
        poles = values["poles"]

        if isinstance(p_spec, IndexPerturbation):
            eps_complex = PoleResidue._eps_model(
                eps_inf=eps_inf, poles=poles, frequency=p_spec.freq
            )
            n, k = Medium.eps_complex_to_nk(eps_c=eps_complex)
            delta_eps_range, _ = p_spec._delta_eps_delta_sigma_ranges(n, k)
        elif isinstance(p_spec, PermittivityPerturbation):
            delta_eps_range, _ = p_spec._delta_eps_delta_sigma_ranges()
        else:
            raise SetupError("Unknown type of 'perturbation_spec'.")

        _warn_potential_error(
            field_name="eps_inf",
            base_value=eps_inf,
            val_change_range=delta_eps_range,
            allowed_real_range=(0.0, None),
            allowed_imag_range=None,
        )

        return values

    def perturbed_copy(
        self,
        temperature: CustomSpatialDataType = None,
        electron_density: CustomSpatialDataType = None,
        hole_density: CustomSpatialDataType = None,
        interp_method: InterpMethod = "linear",
    ) -> Union[PoleResidue, CustomPoleResidue]:
        """Sample perturbations on provided heat and/or charge data and return 'CustomPoleResidue'.
        Any of temperature, electron_density, and hole_density can be 'None'. If all passed
        arguments are 'None' then a 'PoleResidue' object is returned. All provided fields must have
        identical coords.

        Parameters
        ----------
        temperature : Union[
                :class:`.SpatialDataArray`,
                :class:`.TriangularGridDataset`,
                :class:`.TetrahedralGridDataset`,
            ] = None
            Temperature field data.
        electron_density : Union[
                :class:`.SpatialDataArray`,
                :class:`.TriangularGridDataset`,
                :class:`.TetrahedralGridDataset`,
            ] = None
            Electron density field data.
        hole_density : Union[
                :class:`.SpatialDataArray`,
                :class:`.TriangularGridDataset`,
                :class:`.TetrahedralGridDataset`,
            ] = None
            Hole density field data.
        interp_method : :class:`.InterpMethod`, optional
            Interpolation method to obtain heat and/or charge values that are not supplied
            at the Yee grids.

        Returns
        -------
        Union[PoleResidue, CustomPoleResidue]
            Medium specification after application of heat and/or charge data.
        """

        new_dict = self.dict(
            exclude={"eps_inf_perturbation", "poles_perturbation", "perturbation_spec", "type"}
        )

        if all(x is None for x in [temperature, electron_density, hole_density]):
            new_dict.pop("subpixel")
            return PoleResidue.parse_obj(new_dict)

        zeros = ParameterPerturbation._zeros_like(temperature, electron_density, hole_density)

        eps_inf_field = self.eps_inf + zeros
        poles_field = [[a + zeros, c + zeros] for a, c in self.poles]

        if self.perturbation_spec is not None:
            pspec = self.perturbation_spec
            if isinstance(pspec, PermittivityPerturbation):
                delta_eps, delta_sigma = pspec._sample_delta_eps_delta_sigma(
                    temperature, electron_density, hole_density
                )
            elif isinstance(pspec, IndexPerturbation):
                n, k = self.nk_model(frequency=pspec.freq)
                delta_eps, delta_sigma = pspec._sample_delta_eps_delta_sigma(
                    n, k, temperature, electron_density, hole_density
                )

            if delta_eps is not None:
                eps_inf_field = eps_inf_field + delta_eps

            if delta_sigma is not None:
                poles_field = poles_field + [[zeros, 0.5 * delta_sigma / EPSILON_0]]
        else:
            # sample eps_inf
            if self.eps_inf_perturbation is not None:
                eps_inf_field = eps_inf_field + self.eps_inf_perturbation.apply_data(
                    temperature, electron_density, hole_density
                )

            # sample poles
            if self.poles_perturbation is not None:
                for ind, ((a_perturb, c_perturb), (a_field, c_field)) in enumerate(
                    zip(self.poles_perturbation, poles_field)
                ):
                    if a_perturb is not None:
                        a_field = a_field + a_perturb.apply_data(
                            temperature, electron_density, hole_density
                        )
                    if c_perturb is not None:
                        c_field = c_field + c_perturb.apply_data(
                            temperature, electron_density, hole_density
                        )
                    poles_field[ind] = [a_field, c_field]

        new_dict["eps_inf"] = eps_inf_field
        new_dict["poles"] = poles_field
        new_dict["interp_method"] = interp_method

        return CustomPoleResidue.parse_obj(new_dict)
