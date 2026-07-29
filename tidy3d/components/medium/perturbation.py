"""Temperature and carrier-density perturbation medium models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from pydantic import (
    Field,
    model_validator,
)

from tidy3d.components.base import Tidy3dBaseModel
from tidy3d.components.parameter_perturbation import (
    IndexPerturbation,
    ParameterPerturbation,
    PermittivityPerturbation,
)
from tidy3d.components.types import TYPE_TAG_STR
from tidy3d.components.validators import (
    validate_parameter_perturbation,
)
from tidy3d.constants import (
    CONDUCTIVITY,
    EPSILON_0,
    PERMITTIVITY,
    RADPERSEC,
)
from tidy3d.exceptions import SetupError

if TYPE_CHECKING:
    from tidy3d.compat import Self
    from tidy3d.components.data.utils import CustomSpatialDataType
    from tidy3d.components.types import InterpMethod

    from .abstract_custom import AbstractCustomMedium
    from .base import AbstractMedium
    from .pole_residue import DispersiveMedium

from .custom import CustomMedium, CustomPoleResidue
from .isotropic import Medium
from .pole_residue import PoleResidue


class AbstractPerturbationMedium(ABC, Tidy3dBaseModel):
    """Abstract class for medium perturbation."""

    subpixel: bool = Field(
        True,
        title="Subpixel averaging",
        description="This value will be transferred to the resulting custom medium. That is, "
        "if ``True``, the subpixel averaging will be applied to the custom medium. The type "
        "of subpixel averaging method applied is specified in ``Simulation``'s field ``subpixel``. "
        "If the resulting medium is not a custom medium (no perturbations), this field does not "
        "have an effect.",
    )

    perturbation_spec: PermittivityPerturbation | IndexPerturbation | None = Field(
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
    ) -> AbstractMedium | AbstractCustomMedium:
        """Sample perturbations on provided heat and/or charge data and create a custom medium.
        Any of ``temperature``, ``electron_density``, and ``hole_density`` can be ``None``.
        If all passed arguments are ``None`` then a non-custom medium is returned.
        All provided fields must have identical coords.

        Parameters
        ----------
        temperature : Union[:class:`.SpatialDataArray`, :class:`.TriangularGridDataset`, :class:`.TetrahedralGridDataset`] = None
            Temperature field data.
        electron_density : Union[:class:`.SpatialDataArray`, :class:`.TriangularGridDataset`, :class:`.TetrahedralGridDataset`] = None
            Electron density field data.
        hole_density : Union[:class:`.SpatialDataArray`, :class:`.TriangularGridDataset`, :class:`.TetrahedralGridDataset`] = None
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
        medium: Medium | DispersiveMedium,
        subpixel: bool = True,
        perturbation_spec: PermittivityPerturbation | IndexPerturbation = None,
        **kwargs: Any,
    ) -> Self:
        """Construct a medium with pertubation models from an unpertubed one.

        Parameters
        ----------
        medium : Union[:class:`.Medium`, :class:`.DispersiveMedium`]
            A medium with no perturbation models.
        subpixel : bool = True
            Subpixel averaging of derivative custom medium.
        perturbation_spec : Union[:class:`.PermittivityPerturbation`, :class:`.IndexPerturbation`] = None
            Perturbation model specification.

        Returns
        -------
        :class:`.AbstractPerturbationMedium`
            Resulting medium with perturbation model.
        """

        new_dict = medium.model_dump(exclude={TYPE_TAG_STR})

        new_dict["perturbation_spec"] = perturbation_spec
        new_dict["subpixel"] = subpixel

        new_dict.update(kwargs)

        return cls.model_validate(new_dict)


class PerturbationMedium(Medium, AbstractPerturbationMedium):
    """Dispersionless medium with perturbations. Perturbation model can be defined either directly
    through providing ``permittivity_perturbation`` and ``conductivity_perturbation`` or via
    providing a specific perturbation model (:class:`~tidy3d.PermittivityPerturbation`,
    :class:`~tidy3d.IndexPerturbation`) as ``perturbaiton_spec``.

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

    permittivity_perturbation: ParameterPerturbation | None = Field(
        None,
        title="Permittivity Perturbation",
        description="List of heat and/or charge perturbations to permittivity.",
        json_schema_extra={"units": PERMITTIVITY},
    )

    conductivity_perturbation: ParameterPerturbation | None = Field(
        None,
        title="Permittivity Perturbation",
        description="List of heat and/or charge perturbations to permittivity.",
        json_schema_extra={"units": CONDUCTIVITY},
    )

    _permittivity_perturbation_validator = validate_parameter_perturbation(
        "permittivity_perturbation",
        "permittivity",
        allowed_complex=False,
    )

    _conductivity_perturbation_validator = validate_parameter_perturbation(
        "conductivity_perturbation",
        "conductivity",
        allowed_complex=False,
    )

    @model_validator(mode="after")
    def _run_after_validators(self) -> Self:
        """Run post-init validations in an explicit, dependency-aware order."""
        super()._run_after_validators()
        self._check_overdefining()
        return self

    def _check_overdefining(self) -> Self:
        """Check that perturbation model is provided either directly or through
        ``perturbation_spec``, but not both.
        """

        perm_p = self.permittivity_perturbation is not None
        cond_p = self.conductivity_perturbation is not None
        p_spec = self.perturbation_spec is not None

        if p_spec and (perm_p or cond_p):
            self._raise_validation_error_at_loc(
                SetupError(
                    "Must provide perturbation model either as 'perturbation_spec' or as "
                    "'permittivity_perturbation' and 'conductivity_perturbation', "
                    "but not in both ways simultaneously."
                ),
                "perturbation_spec",
            )

        return self

    def perturbed_copy(
        self,
        temperature: CustomSpatialDataType = None,
        electron_density: CustomSpatialDataType = None,
        hole_density: CustomSpatialDataType = None,
        interp_method: InterpMethod = "linear",
    ) -> PerturbationMedium | CustomMedium:
        """Sample perturbations on provided heat and/or charge data and return 'CustomMedium'.
        Any of temperature, electron_density, and hole_density can be 'None'. If all passed
        arguments are 'None' then a 'Medium' object is returned. All provided fields must have
        identical coords.

        Parameters
        ----------
        temperature : Union[:class:`.SpatialDataArray`, :class:`.TriangularGridDataset`, :class:`.TetrahedralGridDataset`] = None
            Temperature field data.
        electron_density : Union[:class:`.SpatialDataArray`, :class:`.TriangularGridDataset`, :class:`.TetrahedralGridDataset`] = None
            Electron density field data.
        hole_density : Union[:class:`.SpatialDataArray`, :class:`.TriangularGridDataset`, :class:`.TetrahedralGridDataset`] = None
            Hole density field data.
        interp_method : :class:`.InterpMethod`, optional
            Interpolation method to obtain heat and/or charge values that are not supplied
            at the Yee grids.

        Returns
        -------
        Union[PerturbationMedium, CustomMedium]
            Medium specification after application of heat and/or charge data.
        """

        # in the absence of perturbation
        if all(x is None for x in [temperature, electron_density, hole_density]):
            return self

        new_dict = self.model_dump(
            exclude={
                "permittivity_perturbation",
                "conductivity_perturbation",
                "perturbation_spec",
                "type",
            }
        )

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
        new_dict["derived_from"] = self

        return CustomMedium.model_validate(new_dict)


class PerturbationPoleResidue(PoleResidue, AbstractPerturbationMedium):
    """A dispersive medium described by the pole-residue pair model with perturbations.
    Perturbation model can be defined either directly
    through providing ``eps_inf_perturbation`` and ``poles_perturbation`` or via
    providing a specific perturbation model (:class:`~tidy3d.PermittivityPerturbation`,
    :class:`~tidy3d.IndexPerturbation`) as ``perturbaiton_spec``.

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

    eps_inf_perturbation: ParameterPerturbation | None = Field(
        None,
        title="Perturbation of Epsilon at Infinity",
        description="Perturbations to relative permittivity at infinite frequency "
        "(:math:`\\epsilon_\\infty`).",
        json_schema_extra={"units": PERMITTIVITY},
    )

    poles_perturbation: (
        tuple[tuple[ParameterPerturbation | None, ParameterPerturbation | None], ...] | None
    ) = Field(
        None,
        title="Perturbations of Poles",
        description="Perturbations to poles of the model.",
        json_schema_extra={"units": (RADPERSEC, RADPERSEC)},
    )

    _eps_inf_perturbation_validator = validate_parameter_perturbation(
        "eps_inf_perturbation",
        "eps_inf",
        allowed_complex=False,
    )

    _poles_perturbation_validator = validate_parameter_perturbation(
        "poles_perturbation",
        "poles",
    )

    @model_validator(mode="after")
    def _run_after_validators(self) -> Self:
        """Run post-init validations in an explicit, dependency-aware order."""
        super()._run_after_validators()
        self._check_overdefining()
        return self

    def _check_overdefining(self) -> Self:
        """Check that perturbation model is provided either directly or through
        ``perturbation_spec``, but not both.
        """

        eps_i_p = self.eps_inf_perturbation is not None
        poles_p = self.poles_perturbation is not None
        p_spec = self.perturbation_spec is not None

        if p_spec and (eps_i_p or poles_p):
            self._raise_validation_error_at_loc(
                SetupError(
                    "Must provide perturbation model either as 'perturbation_spec' or as "
                    "'eps_inf_perturbation' and 'poles_perturbation', "
                    "but not in both ways simultaneously."
                ),
                "perturbation_spec",
            )

        return self

    def perturbed_copy(
        self,
        temperature: CustomSpatialDataType = None,
        electron_density: CustomSpatialDataType = None,
        hole_density: CustomSpatialDataType = None,
        interp_method: InterpMethod = "linear",
    ) -> PerturbationPoleResidue | CustomPoleResidue:
        """Sample perturbations on provided heat and/or charge data and return 'CustomPoleResidue'.
        Any of temperature, electron_density, and hole_density can be 'None'. If all passed
        arguments are 'None' then a 'PoleResidue' object is returned. All provided fields must have
        identical coords.

        Parameters
        ----------
        temperature : Union[:class:`.SpatialDataArray`, :class:`.TriangularGridDataset`, :class:`.TetrahedralGridDataset`] = None
            Temperature field data.
        electron_density : Union[:class:`.SpatialDataArray`, :class:`.TriangularGridDataset`, :class:`.TetrahedralGridDataset`] = None
            Electron density field data.
        hole_density : Union[:class:`.SpatialDataArray`, :class:`.TriangularGridDataset`, :class:`.TetrahedralGridDataset`] = None
            Hole density field data.
        interp_method : :class:`.InterpMethod`, optional
            Interpolation method to obtain heat and/or charge values that are not supplied
            at the Yee grids.

        Returns
        -------
        Union[PerturbationPoleResidue, CustomPoleResidue]
            Medium specification after application of heat and/or charge data.
        """

        # in the absence of perturbation
        if all(x is None for x in [temperature, electron_density, hole_density]):
            return self

        new_dict = self.model_dump(
            exclude={
                "eps_inf_perturbation",
                "poles_perturbation",
                "perturbation_spec",
                TYPE_TAG_STR,
            }
        )

        if all(x is None for x in [temperature, electron_density, hole_density]):
            new_dict.pop("subpixel")
            return PoleResidue.model_validate(new_dict)

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
                poles_field = [*poles_field, [zeros, 0.5 * delta_sigma / EPSILON_0]]
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
        new_dict["derived_from"] = self

        return CustomPoleResidue.model_validate(new_dict)


# types of mediums that can be used in Simulation and Structures

PerturbationMediumType = PerturbationMedium | PerturbationPoleResidue
