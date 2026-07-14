"""Holds dispersive models for several commonly used RF materials."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
from pydantic import Field, PositiveFloat, model_validator

from tidy3d.components.base import cached_property
from tidy3d.components.medium import (
    LossyMetalMedium,
    PoleResidue,
    SurfaceImpedanceFitterParam,
    SurfaceRoughnessType,
)
from tidy3d.components.types import TYPE_TAG_STR
from tidy3d.constants import (
    CONDUCTIVITY,
    HERTZ,
    MICROMETER,
    MICROWAVE_FREQUENCY_RANGE,
)
from tidy3d.exceptions import ValidationError
from tidy3d.log import log
from tidy3d.material_library.material_library import (
    AbstractVariantItem,
    MaterialItem,
    VariantItem,
)
from tidy3d.plugins.dispersion.fit_fast import FastDispersionFitter

if TYPE_CHECKING:
    from tidy3d.components.types import FreqBound

from .rf_material_reference import rf_material_refs


class AbstractVariantItemFreqRange(AbstractVariantItem, ABC):
    """:class:`.AbstractVariantItemFreqRange` is an abstract base class for frequency-range parametrized variant items.

    This class defines the interface for variant items that allow for an optional ``frequency_range``
    parameter to generate their medium model. Concrete implementations handle either
    lossy dielectrics or lossy metals.
    """

    @property
    @abstractmethod
    def medium(self) -> PoleResidue | LossyMetalMedium:
        """The default medium for this variant."""

    @abstractmethod
    def medium_in_range(
        self, frequency_range: FreqBound | None = None
    ) -> PoleResidue | LossyMetalMedium:
        """
        Generate medium with specified ``frequency_range``.

        Parameters
        ----------
        frequency_range : Optional[FreqBound]
            Frequency range of validity for the medium, specified as (f_min, f_max) in Hz.
            Optional for dielectrics (uses original range if not provided).
            Optional for metals (uses default microwave frequency range if not provided).

        Returns
        -------
        Union[PoleResidue, LossyMetalMedium]
            The medium model with the specified frequency range.
        """

    @property
    @abstractmethod
    def summarize_mediums(self) -> dict[str, PoleResidue | LossyMetalMedium]:
        """Summarize the mediums in this variant."""


class VariantItemFreqRangeDielectric(AbstractVariantItemFreqRange):
    """:class:`.VariantItemFreqRangeDielectric` is a frequency-range parametrized variant item
    for lossy dielectric materials.

    This class stores a ``PoleResidue`` model and provides a ``medium`` property that
    returns the prefitted medium, and a ``medium_in_range()`` method that returns
    a medium for a specific frequency range.
    """

    loss_tangent: float | list[float] = Field(
        ...,
        title="Loss Tangent",
        description="Loss tangent for lossy dielectric materials. "
        "If the desired frequency range is outside of the prefitted material's frequency range, "
        "averaged loss tangent data is used to fit a new pole residue medium model.",
    )

    eps_real: float | list[float] = Field(
        ...,
        title="Real Permittivity",
        description="Real permittivity for lossy dielectric materials. "
        "If the desired frequency range is outside of the prefitted material's frequency range, "
        "averaged real permittivity data is used to fit a new pole residue medium model.",
    )

    measurement_frequencies: float | list[float] = Field(
        ...,
        title="Measurement Frequencies",
        description="Frequencies at which the material properties were measured.",
        json_schema_extra={"units": HERTZ},
    )

    prefitted_medium: PoleResidue = Field(
        ...,
        title="Pole-Residue Model",
        description="Pole-residue model for lossy dielectric materials.",
    )

    @model_validator(mode="after")
    def _validate_paired_field_lengths(self) -> VariantItemFreqRangeDielectric:
        """Validate that paired fields (``loss_tangent``, ``eps_real``, ``measurement_frequencies``) all have the same length, are not empty, and contain no NaN values."""
        # Get length of each field (single values have length 1, lists/tuples/arrays have their actual length)
        field_lengths = {}
        for name in ["loss_tangent", "eps_real", "measurement_frequencies"]:
            value = getattr(self, name)
            if isinstance(value, (list, tuple, np.ndarray)):
                length = len(value)
                if length == 0:
                    raise ValidationError(
                        f"Field '{name}' cannot be empty. "
                        f"Paired fields (loss_tangent, eps_real, measurement_frequencies) must contain at least one measurement."
                    )
                field_lengths[name] = length
            else:
                field_lengths[name] = 1  # Single value has implicit length 1

            # Check for NaN values (works for both scalars and arrays)
            if np.any(np.isnan(value)):
                raise ValidationError(
                    f"Field '{name}' contains NaN values. "
                    f"Paired fields (loss_tangent, eps_real, measurement_frequencies) must contain valid numeric measurements."
                )

        # All fields must have the same length
        unique_lengths = set(field_lengths.values())
        if len(unique_lengths) > 1:
            raise ValidationError(
                f"Mismatched lengths for paired fields: "
                f"{', '.join(f'{name}={length}' for name, length in field_lengths.items())}. "
                f"All fields must have the same length (single values count as length 1)."
            )

        return self

    @property
    def medium(self) -> PoleResidue:
        """The default medium for this variant (returns the prefitted medium)."""
        return self.prefitted_medium

    def medium_in_range(self, frequency_range: FreqBound | None = None) -> PoleResidue:
        """
        Generate ``PoleResidue`` medium with specified ``frequency_range``.
        If ``frequency_range`` is not provided, returns the original prefitted material model.

        Parameters
        ----------
        frequency_range : Optional[FreqBound],
            Frequency range of validity for the medium, specified as (f_min, f_max) in Hz.
            If not provided, uses the original frequency_range from the stored prefitted model.

        Returns
        -------
        PoleResidue
            The ``PoleResidue`` model with updated ``frequency_range`` attribute (if provided),
            or the original ``prefitted_medium`` model (if ``frequency_range`` is None).

        Notes
        -----
        If ``frequency_range`` is provided and is within the stored ``prefitted_medium.frequency_range``,
        returns the prefitted model.

        If ``frequency_range`` is provided but is outside the stored ``prefitted_medium.frequency_range``,
        a warning is issued and a new ``PoleResidue`` model is created using the constant loss tangent
        fitter with averaged ``eps_real`` and ``loss_tangent`` values.

        If ``frequency_range`` is not provided, returns the stored ``prefitted_medium`` model as-is.
        """
        if frequency_range is None:
            return self.prefitted_medium

        # Check if requested frequency_range is within the stored prefitted_medium frequency_range
        stored_freq_range = self.prefitted_medium.frequency_range
        if stored_freq_range is not None:
            f_min_stored, f_max_stored = stored_freq_range
            f_min_requested, f_max_requested = frequency_range

            # Check if requested range is within stored range
            if f_min_requested >= f_min_stored and f_max_requested <= f_max_stored:
                # Within range: return stored model
                return self.prefitted_medium

        # Outside range (or stored_freq_range is None): create new model using constant loss tangent fitter
        # Calculate average values
        # Handle both single values and lists/arrays
        if isinstance(self.eps_real, (list, tuple, np.ndarray)):
            eps_real_avg = float(np.mean(self.eps_real))
        else:
            eps_real_avg = self.eps_real

        if isinstance(self.loss_tangent, (list, tuple, np.ndarray)):
            loss_tangent_avg = float(np.mean(self.loss_tangent))
        else:
            loss_tangent_avg = self.loss_tangent

        log.warning(
            f"Requested frequency_range {frequency_range} is outside the stored "
            f"prefitted_medium frequency_range {stored_freq_range}. "
            f"The RF material library does not have measurements for loss_tangent and eps_real "
            f"in the desired frequency range. A new PoleResidue model will be created using "
            f"the constant loss tangent fitter with averaged material properties "
            f"(eps_real={eps_real_avg:.6f}, loss_tangent={loss_tangent_avg:.6f}). "
            f"The resulting fit may be less accurate. For more accurate results, "
            f"consider calling the fast fitter directly with frequency-dependent data."
        )

        # Create new model using constant loss tangent fitter
        new_pole_residue = FastDispersionFitter.constant_loss_tangent_model(
            eps_real=eps_real_avg,
            loss_tangent=loss_tangent_avg,
            frequency_range=frequency_range,
            max_num_poles=5,
            tolerance_rms=1e-3,
            show_progress=False,
        )

        new_pole_residue = new_pole_residue.updated_copy(
            frequency_range=frequency_range
        )  # Set frequency range metadata

        return new_pole_residue

    @property
    def summarize_mediums(self) -> dict[str, PoleResidue]:
        """Summarize the mediums in this variant."""
        return {"medium": self.prefitted_medium}


class VariantItemFreqRangeMetal(AbstractVariantItemFreqRange):
    """Frequency-range parametrized variant item for lossy metal materials.

    This class stores conductivity and optional parameters for creating a ``LossyMetalMedium``.

    The optional ``roughness`` parameter applies a frequency-dependent scaling factor to the
    surface impedance, accounting for surface roughness effects that increase losses at higher
    frequencies. The roughness correction is computed based on the skin depth at each frequency.

    The optional ``thickness`` parameter is used when the conductor thickness is not much greater
    than the skin depth. In this case, a 1D transmission line model is applied to compute the
    surface impedance of the thin conductor, accounting for the finite thickness effects.

    The optional ``fit_param`` parameter controls the pole-residue fitting process used to model
    the scaled surface impedance (surface impedance divided by -1j * omega) over the frequency range.
    It specifies the maximum number of poles, fitting tolerance, number of sampling frequencies,
    and whether to use logarithmic or linear frequency sampling.
    """

    conductivity: PositiveFloat = Field(
        ...,
        title="Conductivity",
        description="Electric conductivity for lossy metal materials.",
        json_schema_extra={"units": CONDUCTIVITY},
    )

    roughness: SurfaceRoughnessType | None = Field(
        None,
        discriminator=TYPE_TAG_STR,
        title="Surface Roughness Model",
        description="Surface roughness model that applies a frequency-dependent scaling "
        "factor to surface impedance, accounting for increased losses at higher frequencies "
        "due to surface roughness effects.",
    )

    thickness: PositiveFloat | None = Field(
        None,
        title="Conductor Thickness",
        description="When the thickness is not much greater than the skin depth, "
        "a 1D transmission line model is applied to compute the surface impedance of the thin conductor, "
        "accounting for finite thickness effects.",
        json_schema_extra={"units": MICROMETER},
    )

    fit_param: SurfaceImpedanceFitterParam | None = Field(
        None,
        title="Fitting Parameters For Surface Impedance",
        description="Parameters controlling the pole-residue fitting process for the scaled "
        "surface impedance (surface impedance divided by -1j * omega) over the frequency range. "
        "Includes maximum number of poles, RMS tolerance, number of sampling frequencies, "
        "and logarithmic vs linear frequency sampling.",
    )

    @cached_property
    def medium(self) -> LossyMetalMedium:
        """The default medium for this variant (uses default microwave frequency range)."""
        # Use enhanced fit parameters for the wide default frequency range (0.3-300 GHz)
        # to ensure good fitting quality. For custom ranges, defaults are sufficient.
        if self.fit_param is None:
            fit_param = SurfaceImpedanceFitterParam(
                max_num_poles=12,  # Increased from default 5 for wide frequency range
                tolerance_rms=1e-3,  # Default value
                frequency_sampling_points=50,  # Increased from default 20 for wide frequency range
            )
        else:
            fit_param = self.fit_param

        kwargs = {
            "conductivity": self.conductivity,
            "frequency_range": MICROWAVE_FREQUENCY_RANGE,
            "fit_param": fit_param,  # fit_param is always set (either from self.fit_param or created above)
        }
        if self.roughness is not None:
            kwargs["roughness"] = self.roughness
        if self.thickness is not None:
            kwargs["thickness"] = self.thickness

        return LossyMetalMedium(**kwargs)

    def medium_in_range(self, frequency_range: FreqBound | None = None) -> LossyMetalMedium:
        """
        Generate ``LossyMetalMedium`` with specified ``frequency_range``.

        Parameters
        ----------
        frequency_range : Optional[FreqBound]
            Frequency range of validity for the medium, specified as (f_min, f_max) in Hz.
            If None, returns the default medium (same as ``medium`` property).

        Returns
        -------
        LossyMetalMedium
            A LossyMetalMedium fitted for the specified frequency range.

        Notes
        -----
        The ``LossyMetalMedium`` internally fits a pole-residue model for the scaled surface
        impedance over the specified frequency range. This fitting happens lazily when the
        ``scaled_surface_impedance_model`` property is first accessed, ensuring accuracy for
        the requested frequency range.

        If ``frequency_range`` is not provided (None) or matches the default microwave frequency
        range, returns the ``medium`` property directly, which uses enhanced fit parameters for
        better accuracy over the wide frequency range.
        """
        # If no frequency range specified or matches the default microwave frequency range,
        # return the default medium property (which uses enhanced fit parameters)
        if frequency_range is None or frequency_range == MICROWAVE_FREQUENCY_RANGE:
            return self.medium

        # Use provided fit_param or None for default
        fit_param = self.fit_param

        kwargs = {
            "conductivity": self.conductivity,
            "frequency_range": frequency_range,
        }
        if self.roughness is not None:
            kwargs["roughness"] = self.roughness
        if self.thickness is not None:
            kwargs["thickness"] = self.thickness
        if fit_param is not None:
            kwargs["fit_param"] = fit_param

        return LossyMetalMedium(**kwargs)

    @property
    def summarize_mediums(self) -> dict[str, LossyMetalMedium]:
        """Summarize the mediums in this variant."""
        return {"medium": self.medium}


class MaterialItemFreqRange(MaterialItem):
    """A material that includes several frequency-range parametrized variants."""

    variants: dict[str, AbstractVariantItemFreqRange] = Field(
        ...,
        title="Dictionary of available variants for this material",
        description="A dictionary of available variants for this material "
        "that maps from a key to the variant model.",
    )

    def __getitem__(self, variant_name: str) -> PoleResidue | LossyMetalMedium:
        """Helper function to easily access the medium of a variant."""
        return self.variants[variant_name].medium

    @property
    def medium(self) -> PoleResidue | LossyMetalMedium:
        """The default medium for the default variant.

        Returns the default medium for the default variant. This is a property to maintain
        consistency with the base ``MaterialItem`` class where ``medium`` is also a property.

        Returns
        -------
        Union[PoleResidue, LossyMetalMedium]
            The default medium model for the default variant.

        Examples
        --------
        >>> # Get default medium for default variant (dielectric)
        >>> default_medium = rf_material_library["RT_duroid5880"].medium
        >>>
        >>> # Get default medium for default variant (metal)
        >>> metal_medium = rf_material_library["Aluminum"].medium
        """
        variant = self.variants[self.default]
        if isinstance(variant, (VariantItemFreqRangeDielectric, VariantItemFreqRangeMetal)):
            return variant.medium
        raise ValueError(
            f"The variant type '{variant.__class__.__name__}' for material '{self.name}' "
            f"is currently not supported by MaterialItemFreqRange.medium. "
            f"Supported types are: VariantItemFreqRangeDielectric and VariantItemFreqRangeMetal."
        )

    def medium_in_range(
        self, frequency_range: FreqBound | None = None
    ) -> PoleResidue | LossyMetalMedium:
        """Get medium for the default variant at a specific frequency range.

        Returns the medium for the default variant at the specified frequency range.
        For dielectrics, if ``frequency_range`` is not provided, returns the stored
        ``PoleResidue`` model (same as ``medium`` property).
        If ``frequency_range`` is provided, returns ``variant.medium_in_range(frequency_range)``.
        For metals, if ``frequency_range`` is not provided, uses the default microwave frequency range
        (300 MHz to 300 GHz). If provided, returns ``variant.medium_in_range(frequency_range)``.

        Parameters
        ----------
        frequency_range : Optional[FreqBound]
            Frequency range of validity for the medium, specified as (f_min, f_max) in Hz.
            Optional for both dielectrics and metals. For dielectrics, uses original range if not provided.
            For metals, uses the default microwave frequency range (300 MHz to 300 GHz) if not provided.

        Returns
        -------
        Union[PoleResidue, LossyMetalMedium]
            The medium model for the default variant with the specified frequency range.

        Examples
        --------
        >>> # Get medium for a specific frequency range (dielectric)
        >>> custom_medium = rf_material_library["RT_duroid5880"].medium_in_range(frequency_range=(5e9, 10e9))
        >>>
        >>> # Get medium for a specific frequency range (metal)
        >>> metal_medium = rf_material_library["Aluminum"].medium_in_range(frequency_range=(1e9, 10e9))
        """
        variant = self.variants[self.default]
        if isinstance(variant, (VariantItemFreqRangeDielectric, VariantItemFreqRangeMetal)):
            return variant.medium_in_range(frequency_range)
        raise ValueError(
            f"The variant type '{variant.__class__.__name__}' for material '{self.name}' "
            f"is currently not supported by MaterialItemFreqRange.medium_in_range(). "
            f"Supported types are: VariantItemFreqRangeDielectric and VariantItemFreqRangeMetal."
        )


Rogers3003_design = VariantItem(
    medium=PoleResidue(
        eps_inf=2.899334368423831,
        poles=[
            ((-13726909999112.38 + 0j), (675466950945.4238 - 0j)),
            ((-127757727974.42976 + 0j), (61040421.35354894 - 0j)),
            ((-374813426.0766755 + 0j), (6559263.919691786 - 0j)),
            ((-60931330853.99707 + 0j), (393463576.50244325 - 0j)),
            (
                (-42782469337.27963 - 3516011892.8127995j),
                (-270153900.61712974 + 1210573246.0512795j),
            ),
        ],
        frequency_range=(1e9, 30e9),
    ),
    reference=[rf_material_refs["Rogers3003"]],
)

Rogers3003_process = VariantItem(
    medium=PoleResidue(
        eps_inf=2.899334368423831,
        poles=[
            ((-13726909999112.38 + 0j), (675466950945.4238 - 0j)),
            ((-127757727974.42976 + 0j), (61040421.35354894 - 0j)),
            ((-374813426.0766755 + 0j), (6559263.919691786 - 0j)),
            ((-60931330853.99707 + 0j), (393463576.50244325 - 0j)),
            (
                (-42782469337.27963 - 3516011892.8127995j),
                (-270153900.61712974 + 1210573246.0512795j),
            ),
        ],
        frequency_range=(1e9, 30e9),
    ),
    reference=[rf_material_refs["Rogers3003"]],
)

Rogers3010_design = VariantItem(
    medium=PoleResidue(
        eps_inf=1.0,
        poles=[
            ((-190311688148667.66 + 0j), (970458264222342.4 - 0j)),
            ((-8041664316784.256 + 0j), (-78427613185.98936 + 0j)),
            ((-115629204802.1858 + 0j), (1439677179.1184998 - 0j)),
            ((-1566991917.1737952 + 0j), (56220784.42607903 - 0j)),
            (
                (-32230539979.65299 - 1886547193.4560573j),
                (316057638.0439734 + 1893874944.2030544j),
            ),
        ],
        frequency_range=(1e9, 30e9),
    ),
    reference=[rf_material_refs["Rogers3010"]],
)

Rogers3010_process = VariantItem(
    medium=PoleResidue(
        eps_inf=2.080628548409516,
        poles=[
            ((-190912690404321.97 + 0j), (876459219078710.1 - 0j)),
            ((-980591939345.2362 + 0j), (-324110911.62494695 + 0j)),
            ((-115069654378.4512 + 0j), (1264596020.4434915 - 0j)),
            ((-1899216108.6070127 + 0j), (52116342.18344934 - 0j)),
            (
                (-27368376580.409447 - 3449802441.5683656j),
                (306447750.09117764 + 286444051.423245j),
            ),
        ],
        frequency_range=(1e9, 30e9),
    ),
    reference=[rf_material_refs["Rogers3010"]],
)

Rogers4003C_design = VariantItem(
    medium=PoleResidue(
        eps_inf=2.3991336253434206,
        poles=[
            ((-3536586338220.136 + 0j), (3456126808589.21 - 0j)),
            ((-1279509068106.1462 + 0j), (-591157900891.4213 + 0j)),
            ((-572117773989.671 + 0j), (32733533477.326588 - 0j)),
            ((-115797982419.77081 + 0j), (403571606.7634415 - 0j)),
            ((-25277453186.29566 + 0j), (174655291.79823563 - 0j)),
        ],
        frequency_range=(8e9, 40e9),
    ),
    reference=[rf_material_refs["Rogers4003C"]],
)

Rogers4003C_process = VariantItem(
    medium=PoleResidue(
        eps_inf=2.225560631279651,
        poles=[
            ((-3878105633791.6025 + 0j), (3541339411029.7544 - 0j)),
            ((-1256834223235.502 + 0j), (-486498887909.03485 + 0j)),
            ((-555141016026.9468 + 0j), (26809945913.510426 - 0j)),
            ((-115094093542.04837 + 0j), (389854375.48413974 - 0j)),
            ((-24404614314.56533 + 0j), (162291307.76685056 - 0j)),
        ],
        frequency_range=(8e9, 40e9),
    ),
    reference=[rf_material_refs["Rogers4003C"]],
)

Rogers4350B_design = VariantItem(
    medium=PoleResidue(
        eps_inf=2.093469160990834,
        poles=[
            ((-3333804051304.176 + 0j), (4662760599740.088 - 0j)),
            ((-1291611694925.002 + 0j), (-911430485105.6237 + 0j)),
            ((-580062825673.5575 + 0j), (49925441975.34481 - 0j)),
            ((-116361944426.69427 + 0j), (565566269.0312785 - 0j)),
            ((-25777935631.449436 + 0j), (250419623.14678243 - 0j)),
        ],
        frequency_range=(8e9, 40e9),
    ),
    reference=[rf_material_refs["Rogers4350B"]],
)

Rogers4350B_process = VariantItem(
    medium=PoleResidue(
        eps_inf=1.898535127745988,
        poles=[
            ((-3674202075105.828 + 0j), (4786609173656.517 - 0j)),
            ((-1270793687238.1047 + 0j), (-749073341863.9965 + 0j)),
            ((-565869017147.8951 + 0j), (41539554888.18475 - 0j)),
            ((-115480614694.7206 + 0j), (545223977.4071108 - 0j)),
            ((-24933632234.96286 + 0j), (232336330.1631193 - 0j)),
        ],
        frequency_range=(8e9, 40e9),
    ),
    reference=[rf_material_refs["Rogers4350B"]],
)

ArlonAD255C_design = VariantItem(
    medium=PoleResidue(
        eps_inf=2.593483364821817,
        poles=[
            ((-670722589451.1771 + 0j), (1220957997.8480208 - 0j)),
            ((-135363399940.36879 + 0j), (353381092.0919367 - 0j)),
            ((-82610486125.0233 + 0j), (-109137171.5087939 + 0j)),
            ((-30177216432.73082 + 0j), (87480864.83649774 - 0j)),
            (
                (-637141607.3397331 - 3465289859.422562j),
                (1410591.6500579868 + 21941106.461227544j),
            ),
        ],
        frequency_range=(1e9, 30e9),
    ),
    reference=[rf_material_refs["ArlonAD255C"]],
)

ArlonAD255C_process = VariantItem(
    medium=PoleResidue(
        eps_inf=2.382226773011058,
        poles=[
            ((-1910716790949.3625 + 0j), (250493153062.4902 - 0j)),
            ((-653869720809.9143 + 0j), (-34684509695.923775 + 0j)),
            ((-208669432914.22174 + 0j), (1077587902.2470737 - 0j)),
            ((-40816290353.693214 + 0j), (68052204.45587738 - 0j)),
            ((-5528506878.719737 + 0j), (14902109.737969821 - 0j)),
        ],
        frequency_range=(1e9, 30e9),
    ),
    reference=[rf_material_refs["ArlonAD255C"]],
)

FR4_standard = VariantItem(
    medium=PoleResidue(
        eps_inf=1.0,
        poles=[
            ((-25028761752571.43 + 0j), (36125660716454.16 - 0j)),
            ((-166080567707.1139 + 0j), (6590627601.155302 - 0j)),
            ((-35972125698.10394 + 0j), (1298122687.3466518 - 0j)),
            ((-17425761930.325096 + 0j), (321442180.24505204 - 0j)),
            ((-88586616.12822348 + 0j), (162501389.22666085 - 0j)),
        ],
        frequency_range=(1e9, 3e9),
    ),
    reference=[rf_material_refs["FR4_standard"]],
)

FR4_lowloss = VariantItem(
    medium=PoleResidue(
        eps_inf=1.4048166324577303,
        poles=[
            ((-1111922427678.9827 + 0j), (6389261773005.734 - 0j)),
            ((-821151129265.1252 + 0j), (-4137380909722.168 + 0j)),
            ((-391087754569.5279 + 0j), (163834407633.01984 - 0j)),
            ((-48665388093.04858 + 0j), (788964219.243519 - 0j)),
            ((-7100136485.071744 + 0j), (204322710.98135194 - 0j)),
        ],
        frequency_range=(1e9, 3e9),
    ),
    reference=[rf_material_refs["FR4_lowloss"]],
)

RT_duroid5880 = VariantItemFreqRangeDielectric(
    loss_tangent=[0.0004, 0.0009],
    eps_real=[2.2, 2.2],
    measurement_frequencies=[1e6, 1e10],
    prefitted_medium=PoleResidue(
        eps_inf=2.194082487374544,
        poles=[
            ((-184481509383.02472 + 0j), (521794132.04005456 - 0j)),
            ((-11600151057.285006 + 0j), (8565651.766613413 - 0j)),
            ((-2321769.4837817834 + 0j), (3134.368885081261 - 0j)),
        ],
        frequency_range=(1e9, 1.1e11),
    ),
    reference=[rf_material_refs["RogersRT_duroid_5880"]],
)
RT_duroid_6035HTC_process = VariantItemFreqRangeDielectric(
    loss_tangent=[0.0013],
    eps_real=[3.5],
    measurement_frequencies=[1e10],
    prefitted_medium=PoleResidue(
        eps_inf=1.6050854545822404,
        poles=[((-62801402532552.52 + 0j), (59501988627702.53 - 0j))],
        frequency_range=(8e9, 4e10),
    ),
    reference=[rf_material_refs["RogersRT_duroid_6035HTC"]],
)
RT_duroid_6035HTC_design = VariantItemFreqRangeDielectric(
    loss_tangent=[0.0013],
    eps_real=[3.6],
    measurement_frequencies=[1e10],
    prefitted_medium=PoleResidue(
        eps_inf=1.0,
        poles=[
            ((-7.797810011903168e16 + 0j), (617754938994122.2 - 0j)),
            ((-83265822107991.55 + 0j), (107586276486304.94 - 0j)),
        ],
        frequency_range=(8e9, 4e10),
    ),
    reference=[rf_material_refs["RogersRT_duroid_6035HTC"]],
)
PTFE_solid = VariantItemFreqRangeDielectric(
    loss_tangent=[0.00022],
    eps_real=[2.02],
    measurement_frequencies=[1e10],
    prefitted_medium=PoleResidue(
        eps_inf=2.019212989211342,
        poles=[
            ((-881114136668.1382 + 0j), (306501347.27997166 - 0j)),
            ((-55285601084.21439 + 0j), (9937348.543665012 - 0j)),
            ((-164820074.0670714 + 0j), (1195084.7305780055 - 0j)),
            ((-133999177246.56708 + 0j), (27607743.236883767 - 0j)),
        ],
        frequency_range=(1e9, 1e11),
    ),
    reference=[rf_material_refs["PTFE"]],
)

PTFE_lowloss_low_density = VariantItemFreqRangeDielectric(
    loss_tangent=[0.00005],
    eps_real=[1.7],
    measurement_frequencies=[1e10],
    prefitted_medium=PoleResidue(
        eps_inf=1.699848219980485,
        poles=[
            ((-896218342448.8458 + 0j), (59882766.40053672 - 0j)),
            ((-56239216907.10459 + 0j), (1920302.8987695205 - 0j)),
            ((-228136100.7830333 + 0j), (229444.6174920688 - 0j)),
            ((-134957679158.1368 + 0j), (5327222.369718695 - 0j)),
        ],
        frequency_range=(1e9, 1e11),
    ),
    reference=[rf_material_refs["PTFE"]],
)

PTFE_microporous_expanded = VariantItemFreqRangeDielectric(
    loss_tangent=[0.00005],
    eps_real=[1.4],
    measurement_frequencies=[1e10],
    prefitted_medium=PoleResidue(
        eps_inf=1.3789744018143326,
        poles=[
            ((-256475000600153.2 + 0j), (2692008343604.1704 - 0j)),
            ((-167090647925.4567 + 0j), (-571258184.9031196 + 0j)),
            ((-166247334068.52448 + 0j), (578153720.1028228 - 0j)),
            ((-5777451.652730278 + 0j), (201150.21508024278 - 0j)),
        ],
        frequency_range=(1e9, 1e11),
    ),
    reference=[rf_material_refs["PTFE"]],
)

Alumina_AO700 = VariantItemFreqRangeDielectric(
    loss_tangent=[0.0006, 0.0006],
    eps_real=[9.4, 9.2],
    measurement_frequencies=[1e6, 2e9],
    prefitted_medium=PoleResidue(
        eps_inf=9.28118974994868,
        poles=[
            ((-87564768.67254353 + 0j), (3467649.9365641726 - 0j)),
            ((-4645671113.0985565 + 0j), (7396176.560323613 - 0j)),
            ((-13021106235.910488 + 0j), (49865360.65045654 - 0j)),
        ],
        frequency_range=(1e6, 2e9),
    ),
    reference=[rf_material_refs["Alumina_kyocera"]],
)

Alumina_AO800 = VariantItemFreqRangeDielectric(
    loss_tangent=[0.0004, 0.0013],
    eps_real=[9.4, 9.4],
    measurement_frequencies=[1e6, 2e9],
    prefitted_medium=PoleResidue(
        eps_inf=9.364674283611748,
        poles=[
            ((-34555426073.01337 + 0j), (603806525.3554897 - 0j)),
            ((-3074674.1972897267 + 0j), (14541.974349977063 - 0j)),
            ((-2049180096.9694386 + 0j), (6236869.656041709 - 0j)),
        ],
        frequency_range=(1e6, 2e9),
    ),
    reference=[rf_material_refs["Alumina_kyocera"]],
)

Alumina_AO479U = VariantItemFreqRangeDielectric(
    loss_tangent=[0.0002],
    eps_real=[9.9],
    measurement_frequencies=[2e9],
    prefitted_medium=PoleResidue(
        eps_inf=1.0,
        poles=[((-600143194825769.9 + 0j), (2670637215123859 - 0j))],
        frequency_range=(1e6, 8.5e9),
    ),
    reference=[rf_material_refs["Alumina_AO479U"]],
)

Getek = VariantItemFreqRangeDielectric(
    loss_tangent=[0.011, 0.01, 0.009, 0.009, 0.01],
    eps_real=[3.81, 3.78, 3.6, 3.5, 3.5],
    measurement_frequencies=[1e8, 1e9, 2e9, 5e9, 1e10],
    prefitted_medium=PoleResidue(
        eps_inf=3.5704492926597595,
        poles=[
            ((-773949794.5259495 + 0j), (28994776.99291547 - 0j)),
            ((-6040023513.518989 + 0j), (138741914.11865166 - 0j)),
            ((-60574983010.82395 + 0j), (1658486170.8391733 - 0j)),
        ],
        frequency_range=(1e8, 1e10),
    ),
    reference=[rf_material_refs["Getek"]],
)

Isola_370HR = VariantItemFreqRangeDielectric(
    loss_tangent=[0.015, 0.0161, 0.021, 0.025, 0.025],
    eps_real=[4.24, 4.17, 4.04, 3.92, 3.92],
    measurement_frequencies=[1e8, 1e9, 2e9, 5e9, 1e10],
    prefitted_medium=PoleResidue(
        eps_inf=3.864530563354492,
        poles=[
            ((-947400350.5182724 + 0j), (57081197.755207635 - 0j)),
            ((-66888762103.143105 + 0j), (5286504592.022915 - 0j)),
            ((-12396196320.907772 + 0j), (565084624.681324 - 0j)),
        ],
        frequency_range=(1e8, 1e10),
    ),
    reference=[rf_material_refs["Isola_370HR"]],
)

Isola_FR406 = VariantItemFreqRangeDielectric(
    loss_tangent=[0.013, 0.0161, 0.0167, 0.0172, 0.0172],
    eps_real=[4.0, 3.95, 3.93, 3.92, 3.92],
    measurement_frequencies=[1e8, 1e9, 2e9, 5e9, 1e10],
    prefitted_medium=PoleResidue(
        eps_inf=3.8135552406311035,
        poles=[
            ((-783656308.9300861 + 0j), (35283399.27609267 - 0j)),
            ((-7707369516.325749 + 0j), (319159892.83936816 - 0j)),
            ((-57856620236.39564 + 0j), (3179395807.666597 - 0j)),
        ],
        frequency_range=(1e8, 1e10),
    ),
    reference=[rf_material_refs["Isola_FR406"]],
)
Isola_FR408 = VariantItemFreqRangeDielectric(
    loss_tangent=[0.0094, 0.0117, 0.012, 0.0127, 0.0125],
    eps_real=[3.69, 3.66, 3.67, 3.66, 3.65],
    measurement_frequencies=[1e8, 1e9, 2e9, 5e9, 1e10],
    prefitted_medium=PoleResidue(
        eps_inf=3.5799712240695953,
        poles=[
            ((-716927602.6966791 + 0j), (20903514.826920874 - 0j)),
            ((-6923571248.390689 + 0j), (191205925.70093066 - 0j)),
            ((-54096530754.46649 + 0j), (2080459547.5164423 - 0j)),
        ],
        frequency_range=(1e8, 1e10),
    ),
    reference=[rf_material_refs["Isola_FR408"]],
)
Megtron6_R5775_KG = VariantItemFreqRangeDielectric(
    loss_tangent=[0.002, 0.002, 0.003, 0.003, 0.004, 0.004],
    eps_real=[3.65, 3.58, 3.57, 3.56, 3.56, 3.55],
    measurement_frequencies=[1e9, 2e9, 4e9, 6e9, 8e9, 1e10],
    prefitted_medium=PoleResidue(
        eps_inf=3.535348117351532,
        poles=[
            ((-214434721057.43317 + 0j), (3446127217.699701 - 0j)),
            ((-49465046163.5074 + 0j), (264817592.7617919 - 0j)),
            ((-6530725048.927035 - 1270400350.1790192j), (22525486.607439514 + 42183787.49899139j)),
        ],
        frequency_range=(1e9, 1e10),
    ),
    reference=[rf_material_refs["Megtron6_R5775_R5670(KG)"]],
)
Megtron6_R5670_KG = VariantItemFreqRangeDielectric(
    loss_tangent=[0.002, 0.002, 0.003, 0.003, 0.004, 0.004],
    eps_real=[3.47, 3.40, 3.39, 3.39, 3.39, 3.38],
    measurement_frequencies=[1e9, 2e9, 4e9, 6e9, 8e9, 1e10],
    prefitted_medium=PoleResidue(
        eps_inf=3.3726682662963867,
        poles=[
            ((-6250234872.067123 + 0j), (26175668.546415854 - 0j)),
            ((-104949883789.25421 + 0j), (1486989807.389693 - 0j)),
            ((-23076386432.036922 + 0j), (21245804.24735235 - 0j)),
        ],
        frequency_range=(1e9, 1e10),
    ),
    reference=[rf_material_refs["Megtron6_R5775_R5670(KG)"]],
)
Megtron6_R5775_N = VariantItemFreqRangeDielectric(
    loss_tangent=[0.002, 0.003, 0.004, 0.004, 0.004, 0.004, 0.004, 0.005, 0.005, 0.005],
    eps_real=[3.40, 3.37, 3.35, 3.35, 3.34, 3.34, 3.34, 3.34, 3.34, 3.34],
    measurement_frequencies=[1e9, 6e9, 12e9, 18e9, 23e9, 29e9, 34e9, 40e9, 45e9, 50e9],
    prefitted_medium=PoleResidue(
        eps_inf=3.307225823402405,
        poles=[
            ((-773353257706.6769 + 0j), (15432797221.92099 - 0j)),
            ((-51917915632.41483 + 0j), (540455950.6398327 - 0j)),
            ((-11738278467.08503 - 4376627321.665409j), (-36521596.20551511 + 168696656.94222957j)),
        ],
        frequency_range=(1e9, 5e10),
    ),
    reference=[rf_material_refs["Megtron6_R5775_R5670(N)"]],
)
Megtron6_R5670_N = VariantItemFreqRangeDielectric(
    loss_tangent=[0.002, 0.003, 0.004, 0.004, 0.004, 0.005, 0.005, 0.005, 0.005, 0.005],
    eps_real=[3.22, 3.20, 3.19, 3.18, 3.18, 3.18, 3.18, 3.18, 3.18, 3.18],
    measurement_frequencies=[1e9, 6e9, 12e9, 18e9, 23e9, 29e9, 34e9, 40e9, 45e9, 50e9],
    prefitted_medium=PoleResidue(
        eps_inf=3.1631745100021362,
        poles=[
            ((-273723888684.53394 + 0j), (3897758154.7457604 - 0j)),
            ((-37414521622.92833 + 0j), (204797365.53446326 - 0j)),
            (
                (-10861609286.292263 - 4534672035.728212j),
                (-440659.91379824624 + 84936797.08916609j),
            ),
        ],
        frequency_range=(1e9, 5e10),
    ),
    reference=[rf_material_refs["Megtron6_R5775_R5670(N)"]],
)
Nelco_N4000_6 = VariantItemFreqRangeDielectric(
    loss_tangent=[0.023, 0.022],
    eps_real=[4.3, 4.0],
    measurement_frequencies=[1e6, 2.5e9],
    prefitted_medium=PoleResidue(
        eps_inf=4.005315810441971,
        poles=[
            ((-682861.6156659126 + 0j), (312924.199044801 - 0j)),
            ((-2385002765.5945625 + 0j), (184285707.85959545 - 0j)),
            ((-16700873357.193726 + 0j), (1081861238.3687525 - 0j)),
        ],
        frequency_range=(1e6, 2.5e9),
    ),
    reference=[rf_material_refs["Nelco_N4000-6"]],
)

Nelco_N4000_13EP = VariantItemFreqRangeDielectric(
    loss_tangent=[0.009, 0.008],
    eps_real=[3.7, 3.7],
    measurement_frequencies=[2.5e9, 1e10],
    prefitted_medium=PoleResidue(
        eps_inf=3.6493919491767883,
        poles=[
            ((-153615421425.48874 + 0j), (3075430659.8732424 - 0j)),
            ((-9510138434.945053 + 0j), (395228289.53439903 - 0j)),
            ((-17154794924.367336 - 11317528227.67127j), (145199787.87337863 - 353126133.8076158j)),
        ],
        frequency_range=(1e9, 1e10),
    ),
    reference=[rf_material_refs["Nelco_N4000-13"]],
)

Nelco_N4000_13EP_SI = VariantItemFreqRangeDielectric(
    loss_tangent=[0.008, 0.008],
    eps_real=[3.2, 3.3],
    measurement_frequencies=[2.5e9, 1e10],
    prefitted_medium=PoleResidue(
        eps_inf=2.372894287109375,
        poles=[
            ((-1024113482003.5793 + 0j), (643330783807.0553 - 0j)),
            ((-398377082666.3882 + 0j), (-77760975093.56621 + 0j)),
            ((-16132164203.086113 + 0j), (352184033.05679125 - 0j)),
        ],
        frequency_range=(1e9, 1e10),
    ),
    reference=[rf_material_refs["Nelco_N4000-13"]],
)

# Metals
Copper_Matula = VariantItemFreqRangeMetal(
    conductivity=59.59,
    reference=[rf_material_refs["Matula"]],
)

Annealed_Copper = VariantItemFreqRangeMetal(
    conductivity=58,
    reference=[rf_material_refs["Annealed_Copper"]],
)

Silver_Matula = VariantItemFreqRangeMetal(
    conductivity=63.01,
    reference=[rf_material_refs["Matula"]],
)

Gold_Matula = VariantItemFreqRangeMetal(
    conductivity=45.1671,
    reference=[rf_material_refs["Matula"]],
)

Aluminum = VariantItemFreqRangeMetal(
    conductivity=37.67,
    reference=[rf_material_refs["Alum_293K"]],
)

Brass_C21000 = VariantItemFreqRangeMetal(
    conductivity=32.48,
    reference=[rf_material_refs["Brass"]],
)

Brass_C26000 = VariantItemFreqRangeMetal(
    conductivity=16.24,
    reference=[rf_material_refs["Brass"]],
)

Cobalt = VariantItemFreqRangeMetal(
    conductivity=17,
    reference=[rf_material_refs["CRC_Handbook"]],
)

Tin = VariantItemFreqRangeMetal(
    conductivity=9.17,
    reference=[rf_material_refs["CRC_Handbook"]],
)

Lead = VariantItemFreqRangeMetal(
    conductivity=4.55,
    reference=[rf_material_refs["Raymond_Serway"]],
)

Platinum = VariantItemFreqRangeMetal(
    conductivity=9.43,
    reference=[rf_material_refs["Raymond_Serway"]],
)

Nickel = VariantItemFreqRangeMetal(
    conductivity=14.3,
    reference=[rf_material_refs["CRC_Handbook"]],
)
Tungsten = VariantItemFreqRangeMetal(
    conductivity=17.9,
    reference=[rf_material_refs["Raymond_Serway"]],
)

Zinc = VariantItemFreqRangeMetal(
    conductivity=16.77,
    reference=[rf_material_refs["Zinc"]],
)

Lithium = VariantItemFreqRangeMetal(
    conductivity=10.8,
    reference=[rf_material_refs["CRC_Handbook"]],
)

AISI_1008 = VariantItemFreqRangeMetal(
    conductivity=6.13,
    reference=[rf_material_refs["AISI_1008"]],
)

Titanium = VariantItemFreqRangeMetal(
    conductivity=2.38,
    reference=[rf_material_refs["CRC_Handbook"]],
)

Iridium = VariantItemFreqRangeMetal(
    conductivity=21.3,
    reference=[rf_material_refs["CRC_Handbook"]],
)

Magnesium = VariantItemFreqRangeMetal(
    conductivity=22.6,
    reference=[rf_material_refs["CRC_Handbook"]],
)

rf_material_library = {
    "RO3003": MaterialItem(
        name="Rogers3003",
        variants={
            "design": Rogers3003_design,
            "process": Rogers3003_process,
        },
        default="design",
    ),
    "RO3010": MaterialItem(
        name="Rogers3010",
        variants={
            "design": Rogers3010_design,
            "process": Rogers3010_process,
        },
        default="design",
    ),
    "RO4003C": MaterialItem(
        name="Rogers4003C",
        variants={
            "design": Rogers4003C_design,
            "process": Rogers4003C_process,
        },
        default="design",
    ),
    "RO4350B": MaterialItem(
        name="Rogers4350B",
        variants={
            "design": Rogers4350B_design,
            "process": Rogers4350B_process,
        },
        default="design",
    ),
    "AD255C": MaterialItem(
        name="ArlonAD255C",
        variants={
            "design": ArlonAD255C_design,
            "process": ArlonAD255C_process,
        },
        default="design",
    ),
    "FR4": MaterialItem(
        name="FR4",
        variants={
            "standard": FR4_standard,
            "lowloss": FR4_lowloss,
        },
        default="standard",
    ),
    "RT_duroid5880": MaterialItemFreqRange(
        name="Rogers RT_duroid 5880",
        variants={
            "standard": RT_duroid5880,
        },
        default="standard",
    ),
    "RT_duroid_6035HTC": MaterialItemFreqRange(
        name="Rogers RT_duroid 6035HTC",
        variants={
            "design": RT_duroid_6035HTC_design,
            "process": RT_duroid_6035HTC_process,
        },
        default="design",
    ),
    "PTFE_solid": MaterialItemFreqRange(
        name="PTFE_solid",
        variants={
            "standard": PTFE_solid,
        },
        default="standard",
    ),
    "PTFE_lowloss_low_density": MaterialItemFreqRange(
        name="PTFE_lowloss_low_density",
        variants={
            "standard": PTFE_lowloss_low_density,
        },
        default="standard",
    ),
    "PTFE_microporous_expanded": MaterialItemFreqRange(
        name="PTFE_microporous_expanded",
        variants={
            "standard": PTFE_microporous_expanded,
        },
        default="standard",
    ),
    "Alumina_AO700": MaterialItemFreqRange(
        name="Alumina AO700",
        variants={
            "standard": Alumina_AO700,
        },
        default="standard",
    ),
    "Alumina_AO800": MaterialItemFreqRange(
        name="Alumina AO800",
        variants={
            "standard": Alumina_AO800,
        },
        default="standard",
    ),
    "Alumina_AO479U": MaterialItemFreqRange(
        name="Alumina AO479U",
        variants={
            "standard": Alumina_AO479U,
        },
        default="standard",
    ),
    "Getek": MaterialItemFreqRange(
        name="Getek",
        variants={
            "standard": Getek,
        },
        default="standard",
    ),
    "Isola_370HR": MaterialItemFreqRange(
        name="Isola 370HR",
        variants={
            "standard": Isola_370HR,
        },
        default="standard",
    ),
    "Isola_FR406": MaterialItemFreqRange(
        name="Isola FR406",
        variants={
            "standard": Isola_FR406,
        },
        default="standard",
    ),
    "Isola_FR408": MaterialItemFreqRange(
        name="Isola FR408",
        variants={
            "standard": Isola_FR408,
        },
        default="standard",
    ),
    "Megtron6_R5775_KG": MaterialItemFreqRange(
        name="Megtron6 R5775_KG",
        variants={
            "standard": Megtron6_R5775_KG,
        },
        default="standard",
    ),
    "Megtron6_R5670_KG": MaterialItemFreqRange(
        name="Megtron6 R5670_KG",
        variants={
            "standard": Megtron6_R5670_KG,
        },
        default="standard",
    ),
    "Megtron6_R5775_N": MaterialItemFreqRange(
        name="Megtron6 R5775_N",
        variants={
            "standard": Megtron6_R5775_N,
        },
        default="standard",
    ),
    "Megtron6_R5670_N": MaterialItemFreqRange(
        name="Megtron6 R5670_N",
        variants={
            "standard": Megtron6_R5670_N,
        },
        default="standard",
    ),
    "Nelco_N4000_6": MaterialItemFreqRange(
        name="Nelco N4000-6",
        variants={
            "standard": Nelco_N4000_6,
        },
        default="standard",
    ),
    "Nelco_N4000_13EP": MaterialItemFreqRange(
        name="Nelco N4000-13 EP",
        variants={
            "standard": Nelco_N4000_13EP,
        },
        default="standard",
    ),
    "Nelco_N4000_13EP_SI": MaterialItemFreqRange(
        name="Nelco N4000-13 EP SI",
        variants={
            "SI": Nelco_N4000_13EP_SI,
        },
        default="SI",
    ),
    # Metals
    "Copper_Matula": MaterialItemFreqRange(
        name="Copper (Matula)",
        variants={
            "standard": Copper_Matula,
        },
        default="standard",
    ),
    "Annealed_Copper": MaterialItemFreqRange(
        name="Annealed Copper",
        variants={
            "standard": Annealed_Copper,
        },
        default="standard",
    ),
    "Silver_Matula": MaterialItemFreqRange(
        name="Silver (Matula)",
        variants={
            "standard": Silver_Matula,
        },
        default="standard",
    ),
    "Gold_Matula": MaterialItemFreqRange(
        name="Gold (Matula)",
        variants={
            "standard": Gold_Matula,
        },
        default="standard",
    ),
    "Aluminum": MaterialItemFreqRange(
        name="Aluminum",
        variants={
            "standard": Aluminum,
        },
        default="standard",
    ),
    "Brass_C21000": MaterialItemFreqRange(
        name="Brass C21000",
        variants={
            "standard": Brass_C21000,
        },
        default="standard",
    ),
    "Brass_C26000": MaterialItemFreqRange(
        name="Brass C26000",
        variants={
            "standard": Brass_C26000,
        },
        default="standard",
    ),
    "Cobalt": MaterialItemFreqRange(
        name="Cobalt",
        variants={
            "standard": Cobalt,
        },
        default="standard",
    ),
    "Tin": MaterialItemFreqRange(
        name="Tin",
        variants={
            "standard": Tin,
        },
        default="standard",
    ),
    "Lead": MaterialItemFreqRange(
        name="Lead",
        variants={
            "standard": Lead,
        },
        default="standard",
    ),
    "Platinum": MaterialItemFreqRange(
        name="Platinum",
        variants={
            "standard": Platinum,
        },
        default="standard",
    ),
    "Nickel": MaterialItemFreqRange(
        name="Nickel",
        variants={
            "standard": Nickel,
        },
        default="standard",
    ),
    "Tungsten": MaterialItemFreqRange(
        name="Tungsten",
        variants={
            "standard": Tungsten,
        },
        default="standard",
    ),
    "Zinc": MaterialItemFreqRange(
        name="Zinc",
        variants={
            "standard": Zinc,
        },
        default="standard",
    ),
    "Lithium": MaterialItemFreqRange(
        name="Lithium",
        variants={
            "standard": Lithium,
        },
        default="standard",
    ),
    "AISI_1008": MaterialItemFreqRange(
        name="AISI 1008 Steel",
        variants={
            "standard": AISI_1008,
        },
        default="standard",
    ),
    "Titanium": MaterialItemFreqRange(
        name="Titanium",
        variants={
            "standard": Titanium,
        },
        default="standard",
    ),
    "Iridium": MaterialItemFreqRange(
        name="Iridium",
        variants={
            "standard": Iridium,
        },
        default="standard",
    ),
    "Magnesium": MaterialItemFreqRange(
        name="Magnesium",
        variants={
            "standard": Magnesium,
        },
        default="standard",
    ),
}
