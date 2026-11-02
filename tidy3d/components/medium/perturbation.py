"""Temperature, carrier-density, and electric-field perturbation medium models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from pydantic import (
    Field,
)

from tidy3d.components.base import Tidy3dBaseModel
from tidy3d.components.data.utils import _as_electric_field_components
from tidy3d.components.parameter_perturbation import (
    ElectroOpticPerturbation,
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
    from tidy3d.components.data.utils import (
        CustomSpatialDataType,
        ElectricFieldComponentsType,
        ElectricFieldType,
    )
    from tidy3d.components.types import InterpMethod

    from .abstract_custom import AbstractCustomMedium
    from .base import AbstractMedium
    from .pole_residue import DispersiveMedium

from .anisotropic import AnisotropicMedium
from .custom import CustomAnisotropicMedium, CustomMedium, CustomPoleResidue
from .debye import Debye
from .drude import Drude
from .isotropic import Medium
from .lorentz import Lorentz
from .pole_residue import PoleResidue
from .sellmeier import Sellmeier


class AbstractPerturbationMedium(ABC, Tidy3dBaseModel):
    """Abstract class for medium perturbation."""

    subpixel: bool = Field(
        default=True,
        title="Subpixel averaging",
        description="This value will be transferred to the resulting custom medium. That is, "
        "if ``True``, the subpixel averaging will be applied to the custom medium. The type "
        "of subpixel averaging method applied is specified in ``Simulation``'s field ``subpixel``. "
        "If the resulting medium is not a custom medium (no perturbations), this field does not "
        "have an effect.",
    )

    electro_optic_spec: ElectroOpticPerturbation | None = Field(
        default=None,
        title="Electro-Optic Perturbation",
        description="Change of the refractive index by a static electric field, applied when "
        "an ``electric_field`` is given to ``perturbed_copy``. The medium then becomes a "
        ":class:`.CustomAnisotropicMedium`, to which the heat and charge perturbations add. A "
        "medium with ``electro_optic_spec`` cannot have a ``nonlinear_spec``.",
    )

    @abstractmethod
    def perturbed_copy(
        self,
        temperature: CustomSpatialDataType = None,
        electron_density: CustomSpatialDataType = None,
        hole_density: CustomSpatialDataType = None,
        interp_method: InterpMethod = "linear",
        electric_field: ElectricFieldType | None = None,
    ) -> AbstractMedium | AbstractCustomMedium:
        """Sample perturbations on provided heat, charge, and electric field data and create a
        custom medium. Any of ``temperature``, ``electron_density``, ``hole_density``, and
        ``electric_field`` can be ``None``. If all passed arguments are ``None``, or only
        ``electric_field`` is given to a medium without ``electro_optic_spec``, the medium is
        returned unchanged. All provided fields must have identical coords.

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
        electric_field : Union[tuple, :class:`.TriangularGridDataset`, :class:`.TetrahedralGridDataset`] = None
            Static electric field (V/um), applied through ``electro_optic_spec``. Either a tuple
            ``(Ex, Ey, Ez)`` of spatial fields, ``None`` for a zero component, or an unstructured
            dataset with an ``axis`` dimension, such as ``SteadyElectricFieldData.E`` at one
            voltage.

        Returns
        -------
        Union[AbstractMedium, AbstractCustomMedium]
            Medium specification after application of heat, charge, and electric field data.
        """

    @classmethod
    def from_unperturbed(
        cls,
        medium: Medium | DispersiveMedium | AnisotropicMedium,
        subpixel: bool = True,
        **kwargs: Any,
    ) -> Self:
        """Construct a medium with perturbation models from an unperturbed one.

        Parameters
        ----------
        medium : Union[:class:`.Medium`, :class:`.DispersiveMedium`, :class:`.AnisotropicMedium`]
            A medium with no perturbation models.
        subpixel : bool = True
            Subpixel averaging of derivative custom medium.
        **kwargs
            Perturbation models of the new medium, such as ``electro_optic_spec``.

        Returns
        -------
        :class:`.AbstractPerturbationMedium`
            Resulting medium with perturbation model.
        """

        new_dict = medium.model_dump(exclude={TYPE_TAG_STR})
        new_dict["subpixel"] = subpixel
        new_dict.update(kwargs)

        return cls.model_validate(new_dict)

    def _electric_field_components(
        self, electric_field: ElectricFieldType | None
    ) -> ElectricFieldComponentsType | None:
        """Components of ``electric_field`` along x, y, z, or ``None`` when it has no effect."""
        if electric_field is None or self.electro_optic_spec is None:
            return None
        return _as_electric_field_components(electric_field)


def _custom_anisotropic_copy(
    medium: AbstractPerturbationMedium, components: list[AbstractCustomMedium]
) -> CustomAnisotropicMedium:
    """Diagonal custom medium of the three perturbed components of ``medium``."""
    xx, yy, zz = components
    return CustomAnisotropicMedium(
        xx=xx,
        yy=yy,
        zz=zz,
        name=medium.name,
        frequency_range=medium.frequency_range,
        viz_spec=medium.viz_spec,
        heat_spec=medium.heat_spec,
        derived_from=medium,
    )


class AbstractIsotropicPerturbationMedium(AbstractPerturbationMedium):
    """Abstract class for isotropic medium perturbation."""

    perturbation_spec: PermittivityPerturbation | IndexPerturbation | None = Field(
        default=None,
        title="Perturbation Spec",
        description="Specification of medium perturbation as one of predefined types.",
        discriminator=TYPE_TAG_STR,
    )

    @classmethod
    def from_unperturbed(
        cls,
        medium: Medium | DispersiveMedium,
        subpixel: bool = True,
        perturbation_spec: PermittivityPerturbation | IndexPerturbation | None = None,
        **kwargs: Any,
    ) -> Self:
        """Construct a medium with perturbation models from an unperturbed one.

        Parameters
        ----------
        medium : Union[:class:`.Medium`, :class:`.DispersiveMedium`]
            A medium with no perturbation models.
        subpixel : bool = True
            Subpixel averaging of derivative custom medium.
        perturbation_spec : Union[:class:`.PermittivityPerturbation`, :class:`.IndexPerturbation`] = None
            Perturbation model specification.
        **kwargs
            Further perturbation models of the new medium, such as ``electro_optic_spec``.

        Returns
        -------
        :class:`.AbstractIsotropicPerturbationMedium`
            Resulting medium with perturbation model.
        """
        return super().from_unperturbed(
            medium, subpixel=subpixel, perturbation_spec=perturbation_spec, **kwargs
        )

    @abstractmethod
    def _perturbed_fields(
        self,
        zeros: CustomSpatialDataType,
        temperature: CustomSpatialDataType | None,
        electron_density: CustomSpatialDataType | None,
        hole_density: CustomSpatialDataType | None,
    ) -> dict[str, Any]:
        """Fields of the custom medium after the heat and charge perturbations."""

    @abstractmethod
    def _add_delta_eps_sigma(
        self,
        fields: dict[str, Any],
        delta_eps: CustomSpatialDataType,
        delta_sigma: CustomSpatialDataType | None,
        zeros: CustomSpatialDataType,
    ) -> dict[str, Any]:
        """Fields of the custom medium with a further permittivity and conductivity change."""

    @abstractmethod
    def _custom_copy(
        self,
        fields: dict[str, Any],
        interp_method: InterpMethod,
        derived_from: AbstractPerturbationMedium | None,
    ) -> AbstractCustomMedium:
        """Custom medium with the given fields."""

    def perturbed_copy(
        self,
        temperature: CustomSpatialDataType = None,
        electron_density: CustomSpatialDataType = None,
        hole_density: CustomSpatialDataType = None,
        interp_method: InterpMethod = "linear",
        electric_field: ElectricFieldType | None = None,
    ) -> AbstractIsotropicPerturbationMedium | AbstractCustomMedium:
        electric_field = self._electric_field_components(electric_field)

        # in the absence of perturbation
        if all(x is None for x in [temperature, electron_density, hole_density, electric_field]):
            return self

        zeros = ParameterPerturbation._zeros_like(
            temperature, electron_density, hole_density, electric_field
        )
        fields = self._perturbed_fields(zeros, temperature, electron_density, hole_density)

        if electric_field is None:
            return self._custom_copy(fields, interp_method, derived_from=self)

        # the electro-optic change differs per axis, so each diagonal component gets its own copy
        nk = self.nk_model(frequency=self.electro_optic_spec.freq)
        delta_eps, delta_sigma = self.electro_optic_spec._sample_delta_eps_delta_sigma(
            (nk, nk, nk), electric_field
        )
        # only the anisotropic wrapper records the source medium
        components = [
            self._custom_copy(
                {**self._add_delta_eps_sigma(fields, d_eps, d_sigma, zeros), "name": None},
                interp_method,
                derived_from=None,
            )
            for d_eps, d_sigma in zip(delta_eps, delta_sigma)
        ]
        return _custom_anisotropic_copy(self, components)

    perturbed_copy.__doc__ = AbstractPerturbationMedium.perturbed_copy.__doc__


class PerturbationMedium(Medium, AbstractIsotropicPerturbationMedium):
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
        default=None,
        title="Permittivity Perturbation",
        description="List of heat and/or charge perturbations to permittivity.",
        json_schema_extra={"units": PERMITTIVITY},
    )

    conductivity_perturbation: ParameterPerturbation | None = Field(
        default=None,
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

    def _perturbed_fields(
        self,
        zeros: CustomSpatialDataType,
        temperature: CustomSpatialDataType | None,
        electron_density: CustomSpatialDataType | None,
        hole_density: CustomSpatialDataType | None,
    ) -> dict[str, Any]:
        new_dict = self.model_dump(
            exclude={
                "permittivity_perturbation",
                "conductivity_perturbation",
                "perturbation_spec",
                "electro_optic_spec",
                TYPE_TAG_STR,
            }
        )

        permittivity_field = self.permittivity + zeros
        conductivity_field = None if self.conductivity == 0 else self.conductivity + zeros

        # with only an electric field, the heat and charge models have no data to sample
        if all(x is None for x in (temperature, electron_density, hole_density)):
            return {
                **new_dict,
                "permittivity": permittivity_field,
                "conductivity": conductivity_field,
            }

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

        if delta_sigma is not None:
            conductivity_field = self.conductivity + delta_sigma

        new_dict["permittivity"] = permittivity_field
        new_dict["conductivity"] = conductivity_field
        return new_dict

    def _add_delta_eps_sigma(
        self,
        fields: dict[str, Any],
        delta_eps: CustomSpatialDataType,
        delta_sigma: CustomSpatialDataType | None,
        zeros: CustomSpatialDataType,
    ) -> dict[str, Any]:
        conductivity_field = fields["conductivity"]
        if delta_sigma is not None:
            # a conductivity change needs a lossy medium, whose fields hold its conductivity
            conductivity_field = conductivity_field + delta_sigma
        return {
            **fields,
            "permittivity": fields["permittivity"] + delta_eps,
            "conductivity": conductivity_field,
        }

    def _custom_copy(
        self,
        fields: dict[str, Any],
        interp_method: InterpMethod,
        derived_from: AbstractPerturbationMedium | None,
    ) -> CustomMedium:
        return CustomMedium.model_validate(
            {**fields, "interp_method": interp_method, "derived_from": derived_from}
        )


class PerturbationPoleResidue(PoleResidue, AbstractIsotropicPerturbationMedium):
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
        default=None,
        title="Perturbation of Epsilon at Infinity",
        description="Perturbations to relative permittivity at infinite frequency "
        "(:math:`\\epsilon_\\infty`).",
        json_schema_extra={"units": PERMITTIVITY},
    )

    poles_perturbation: (
        tuple[tuple[ParameterPerturbation | None, ParameterPerturbation | None], ...] | None
    ) = Field(
        default=None,
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

    def _perturbed_fields(
        self,
        zeros: CustomSpatialDataType,
        temperature: CustomSpatialDataType | None,
        electron_density: CustomSpatialDataType | None,
        hole_density: CustomSpatialDataType | None,
    ) -> dict[str, Any]:
        new_dict = self.model_dump(
            exclude={
                "eps_inf_perturbation",
                "poles_perturbation",
                "perturbation_spec",
                "electro_optic_spec",
                TYPE_TAG_STR,
            }
        )

        eps_inf_field = self.eps_inf + zeros
        poles_field = [[a + zeros, c + zeros] for a, c in self.poles]

        # with only an electric field, the heat and charge models have no data to sample
        if all(x is None for x in (temperature, electron_density, hole_density)):
            return {**new_dict, "eps_inf": eps_inf_field, "poles": poles_field}

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
        return new_dict

    def _add_delta_eps_sigma(
        self,
        fields: dict[str, Any],
        delta_eps: CustomSpatialDataType,
        delta_sigma: CustomSpatialDataType | None,
        zeros: CustomSpatialDataType,
    ) -> dict[str, Any]:
        poles_field = fields["poles"]
        if delta_sigma is not None:
            poles_field = [*poles_field, [zeros, 0.5 * delta_sigma / EPSILON_0]]
        return {**fields, "eps_inf": fields["eps_inf"] + delta_eps, "poles": poles_field}

    def _custom_copy(
        self,
        fields: dict[str, Any],
        interp_method: InterpMethod,
        derived_from: AbstractPerturbationMedium | None,
    ) -> CustomPoleResidue:
        return CustomPoleResidue.model_validate(
            {**fields, "interp_method": interp_method, "derived_from": derived_from}
        )


PerturbationAnisotropicComponentType = (
    Medium
    | PoleResidue
    | Sellmeier
    | Lorentz
    | Debye
    | Drude
    | PerturbationMedium
    | PerturbationPoleResidue
)


class PerturbationAnisotropicMedium(AnisotropicMedium, AbstractPerturbationMedium):
    """Diagonally anisotropic medium with perturbations. Each diagonal component is a uniform
    isotropic medium, and may carry its own heat and charge perturbations as a
    :class:`.PerturbationMedium` or :class:`.PerturbationPoleResidue`. The ``electro_optic_spec``
    model of the medium changes the three components differently.

    Note
    ----
    ``subpixel`` applies to every component of the resulting custom medium, and the components
    cannot have their own ``electro_optic_spec`` model.

    Example
    -------
    >>> import numpy as np
    >>> from tidy3d import C_0, ElectroOpticPerturbation, Medium, RotationAroundAxis
    >>> eo = ElectroOpticPerturbation.from_trigonal_3m(
    ...     r13=10e-6,
    ...     r22=5e-6,
    ...     r33=30e-6,
    ...     r51=30e-6,
    ...     freq=C_0 / 1.55,
    ...     rotation=RotationAroundAxis(axis=0, angle=-np.pi / 2),
    ... )
    >>> crystal = PerturbationAnisotropicMedium(
    ...     xx=Medium(permittivity=4.9),
    ...     yy=Medium(permittivity=4.6),
    ...     zz=Medium(permittivity=4.9),
    ...     electro_optic_spec=eo,
    ... )
    """

    xx: PerturbationAnisotropicComponentType = Field(
        title="XX Component",
        description="Medium describing the xx-component of the diagonal permittivity tensor.",
        discriminator=TYPE_TAG_STR,
    )

    yy: PerturbationAnisotropicComponentType = Field(
        title="YY Component",
        description="Medium describing the yy-component of the diagonal permittivity tensor.",
        discriminator=TYPE_TAG_STR,
    )

    zz: PerturbationAnisotropicComponentType = Field(
        title="ZZ Component",
        description="Medium describing the zz-component of the diagonal permittivity tensor.",
        discriminator=TYPE_TAG_STR,
    )

    @classmethod
    def from_unperturbed(
        cls, medium: AnisotropicMedium, subpixel: bool = True, **kwargs: Any
    ) -> Self:
        """Construct a medium with perturbation models from an unperturbed anisotropic one.

        Parameters
        ----------
        medium : :class:`.AnisotropicMedium`
            A medium with no perturbation models.
        subpixel : bool = True
            Subpixel averaging of derivative custom medium.
        **kwargs
            Perturbation models of the new medium, such as ``electro_optic_spec``.

        Returns
        -------
        :class:`.PerturbationAnisotropicMedium`
            Resulting medium with perturbation model.
        """
        if "perturbation_spec" in kwargs:
            raise SetupError(
                "'PerturbationAnisotropicMedium' has no 'perturbation_spec'. Give the components "
                "their own heat and charge models as 'PerturbationMedium' or "
                "'PerturbationPoleResidue'."
            )
        return super().from_unperturbed(medium, subpixel=subpixel, **kwargs)

    def _perturbation_component(
        self, medium: Medium | DispersiveMedium
    ) -> AbstractIsotropicPerturbationMedium:
        """Component as an isotropic perturbation medium, with ``subpixel`` of this medium."""
        if isinstance(medium, AbstractIsotropicPerturbationMedium):
            return medium.updated_copy(subpixel=self.subpixel)
        if isinstance(medium, Medium):
            return PerturbationMedium.from_unperturbed(medium, subpixel=self.subpixel)
        # the pole-residue form leaves the modulation out
        return PerturbationPoleResidue.from_unperturbed(
            medium.pole_residue, subpixel=self.subpixel, modulation_spec=medium.modulation_spec
        )

    def _component_support_facts(self) -> dict[str, tuple[bool, ...]]:
        """Which components are nonlinear, rejected as in :class:`.AnisotropicMedium`."""
        return {
            "anisotropic_components_nonlinear": tuple(
                c.nonlinear_spec is not None for c in (self.xx, self.yy, self.zz)
            )
        }

    def perturbed_copy(
        self,
        temperature: CustomSpatialDataType = None,
        electron_density: CustomSpatialDataType = None,
        hole_density: CustomSpatialDataType = None,
        interp_method: InterpMethod = "linear",
        electric_field: ElectricFieldType | None = None,
    ) -> PerturbationAnisotropicMedium | CustomAnisotropicMedium:
        electric_field = self._electric_field_components(electric_field)

        # in the absence of perturbation
        if all(x is None for x in [temperature, electron_density, hole_density, electric_field]):
            return self

        zeros = ParameterPerturbation._zeros_like(
            temperature, electron_density, hole_density, electric_field
        )
        media = [self._perturbation_component(m) for m in (self.xx, self.yy, self.zz)]

        deltas = [None] * 3
        if electric_field is not None:
            nk = tuple(m.nk_model(frequency=self.electro_optic_spec.freq) for m in media)
            deltas = list(
                zip(*self.electro_optic_spec._sample_delta_eps_delta_sigma(nk, electric_field))
            )

        components = []
        for medium, delta in zip(media, deltas):
            fields = medium._perturbed_fields(zeros, temperature, electron_density, hole_density)
            if delta is not None:
                fields = medium._add_delta_eps_sigma(fields, *delta, zeros)
            components.append(
                medium._custom_copy({**fields, "name": None}, interp_method, derived_from=None)
            )
        return _custom_anisotropic_copy(self, components)

    perturbed_copy.__doc__ = AbstractPerturbationMedium.perturbed_copy.__doc__


# types of mediums that can be used in Simulation and Structures

PerturbationMediumType = (
    PerturbationMedium | PerturbationPoleResidue | PerturbationAnisotropicMedium
)
