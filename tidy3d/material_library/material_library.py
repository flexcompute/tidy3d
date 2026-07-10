"""Holds dispersive models for several commonly used optical materials."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from pydantic import Field, model_validator

from tidy3d.components.base import Tidy3dBaseModel
from tidy3d.components.material.multi_physics import MultiPhysicsMedium
from tidy3d.components.material.tcad.charge import SemiconductorMedium
from tidy3d.components.medium import AnisotropicMedium, Medium2D, PoleResidue, Sellmeier
from tidy3d.components.tcad.bandgap import ConstantEnergyBandGap
from tidy3d.components.tcad.types import (
    AugerRecombination,
    CaugheyThomasMobility,
    ConstantEffectiveDOS,
    RadiativeRecombination,
    ShockleyReedHallRecombination,
    SlotboomBandGapNarrowing,
)
from tidy3d.components.types import TYPE_TAG_STR
from tidy3d.exceptions import SetupError
from tidy3d.log import log

from .material_reference import ReferenceData, material_refs
from .parametric_materials import Graphene
from .util import (
    repr_pretty_with_rich,
    summarize_material_item,
    summarize_material_item_rich,
    summarize_material_library,
    summarize_material_library_rich,
    summarize_variant_item,
    summarize_variant_item_rich,
)

if TYPE_CHECKING:
    from os import PathLike

    from IPython.lib.pretty import RepresentationPrinter
    from rich.panel import Panel
    from rich.table import Table

    from tidy3d.compat import Self
    from tidy3d.components.types import Axis


CSI_GREEN2008_VARIANT = "Green2008"
CSI_PALIK_LOWLOSS_VARIANT = "Palik_LowLoss"
CSI_PALIK_LOSSY_VARIANT = "Palik_Lossy"


def _variant_name(variant: str, rst: bool = False) -> str:
    """Format a material variant name for warning text or generated docs."""
    if rst:
        return f"``'{variant}'``"
    return f"'{variant}'"


def _material_variant_access(material: str, variant: str, rst: bool = False) -> str:
    """Format a material-library variant lookup for warning text or generated docs."""
    access = f"material_library['{material}']['{variant}']"
    if rst:
        return f"``{access}``"
    return access


def csi_default_migration_message(rst: bool = False, include_suppression: bool = False) -> str:
    """Migration guidance for the crystalline silicon default-variant change."""
    unit = r":math:`{\mu}m`" if rst else "um"
    green2008 = _variant_name(CSI_GREEN2008_VARIANT, rst)
    palik_lowloss = _variant_name(CSI_PALIK_LOWLOSS_VARIANT, rst)
    palik_lossy = _variant_name(CSI_PALIK_LOSSY_VARIANT, rst)
    explicit_lowloss = _material_variant_access("cSi", CSI_PALIK_LOWLOSS_VARIANT, rst)
    explicit_green2008 = _material_variant_access("cSi", CSI_GREEN2008_VARIANT, rst)

    message = (
        f"The default variant for crystalline silicon changed from {green2008} "
        f"to {palik_lowloss}, which uses the fitted Palik model with small loss. "
        f"Use {green2008} explicitly to preserve pre-change results, including "
        f"the 1.2 to 1.45 {unit} overlap where both variants are valid. "
        f"{palik_lowloss} is valid from 1.2 to 250 {unit}. "
        f"Use {green2008} for pre-change results from 0.25 to 1.2 {unit}. "
        f"{palik_lossy} is also available from 0.1 to 1.4 {unit} when the lossy "
        f"Palik model is desired."
    )
    if include_suppression:
        message += (
            " To suppress this warning, request a variant explicitly, "
            f"for example {explicit_lowloss} or {explicit_green2008}."
        )
    return message


def export_matlib_to_file(fname: PathLike = "matlib.json") -> None:
    """Write the material library to a .json file."""
    mat_lib_dict = {
        f'{mat.name} ("{mat_name}")': {
            var_name: json.loads(var.medium._json_string) for var_name, var in mat.variants.items()
        }
        for mat_name, mat in material_library.items()
        if not isinstance(mat, type | MaterialItemUniaxial)
    }

    # Uniaxial medium treated differently
    mat_lib_dict.update(
        {
            f'{mat.name} ("{mat_name}")': {
                var_name: {
                    "ordinary": json.loads(var.ordinary._json_string),
                    "extraordinary": json.loads(var.extraordinary._json_string),
                }
                for var_name, var in mat.variants.items()
            }
            for mat_name, mat in material_library.items()
            if isinstance(mat, MaterialItemUniaxial)
        }
    )

    with open(fname, "w") as f:
        json.dump(mat_lib_dict, f)


class AbstractVariantItem(Tidy3dBaseModel):
    """Reference, and data_source for a variant of a material."""

    reference: list[ReferenceData] | None = Field(
        None,
        title="Reference information",
        description="A list of references related to this variant model.",
    )

    data_url: str | None = Field(
        None,
        title="Dispersion data URL",
        description="The URL to access the dispersion data upon which the material "
        "model is fitted.",
    )

    @property
    def summarize_mediums(self) -> dict[str, PoleResidue | Medium2D | MultiPhysicsMedium]:
        return {}

    def __str__(self) -> str:
        return summarize_variant_item(self)

    def __rich__(self) -> Panel:
        return summarize_variant_item_rich(self)

    def _repr_pretty_(self, p: RepresentationPrinter, cycle: bool) -> None:
        return repr_pretty_with_rich(self, p, cycle)


class VariantItem(AbstractVariantItem):
    """Reference, data_source, and material model for a variant of a material."""

    medium: PoleResidue | MultiPhysicsMedium = Field(
        discriminator=TYPE_TAG_STR,
        title="Material dispersion model",
        description="A dispersive medium described by the pole-residue pair model.",
    )

    @property
    def summarize_mediums(self) -> dict[str, PoleResidue | Medium2D | MultiPhysicsMedium]:
        return {"medium": self.medium}


class MaterialItem(Tidy3dBaseModel):
    """A material that includes several variants."""

    name: str = Field(title="Name", description="Unique name for the medium.")
    variants: dict[str, VariantItem] = Field(
        title="Dictionary of available variants for this material",
        description="A dictionary of available variants for this material "
        "that maps from a key to the variant model.",
    )
    default: str = Field(title="default variant", description="The default type of variant.")

    @model_validator(mode="after")
    def _default_in_variants(self: Self) -> Self:
        """Make sure the default variant is already included in the ``variants``."""
        if self.default not in self.variants:
            raise SetupError(
                f"The data of the default variant '{self.default}' is not supplied; "
                "please include it in the 'variants'."
            )
        _with_palik_lossless_alias(self)
        return self

    def __getitem__(self, variant_name: str) -> PoleResidue | MultiPhysicsMedium:
        """Helper function to easily access the medium of a variant"""
        return self.variants[variant_name].medium

    @property
    def medium(self) -> PoleResidue | MultiPhysicsMedium:
        """The default medium."""
        if self.name == "Silicon Dioxide":
            log.warning(
                "The default variant for silicon dioxide is 'Palik_LowLoss', which was "
                "previously named 'Palik_Lossless'. Use 'Palik_NoLoss' for a zero-loss "
                "Palik model."
            )
        if self._uses_migrated_csi_default:
            log.warning(csi_default_migration_message(include_suppression=True))
        return self.variants[self.default].medium

    @property
    def _uses_migrated_csi_default(self) -> bool:
        """Whether this item matches the migrated built-in cSi default."""
        library = globals().get("material_library")
        return (
            library is not None and self is library.get("cSi") and self.default == "Palik_LowLoss"
        )

    def __str__(self) -> str:
        return summarize_material_item(self)

    def __rich__(self) -> Panel:
        return summarize_material_item_rich(self)

    def _repr_pretty_(self, p: RepresentationPrinter, cycle: bool) -> None:
        return repr_pretty_with_rich(self, p, cycle)


class VariantItem2D(AbstractVariantItem):
    """Reference, data_source, and material model for a variant of a 2D material."""

    medium: Medium2D = Field(
        title="Material dispersion model",
        description="A dispersive 2D medium described by a surface conductivity model, "
        "which is handled as an anisotropic medium with pole-residue pair models "
        "defined for the in-plane directions of the 2D geometry.",
    )

    @property
    def summarize_mediums(self) -> dict[str, PoleResidue | Medium2D | MultiPhysicsMedium]:
        return {"medium": self.medium}


class MaterialItem2D(MaterialItem):
    """A 2D material that includes several variants."""

    variants: dict[str, VariantItem2D] = Field(
        title="Dictionary of available variants for this material",
        description="A dictionary of available variants for this material "
        "that maps from a key to the variant model.",
    )


class VariantItemUniaxial(AbstractVariantItem):
    """Reference, data_source, and material model for a variant of an uniaxial material."""

    ordinary: PoleResidue = Field(
        title="Ordinary Component",
        description="Medium describing the ordinary component.",
    )

    extraordinary: PoleResidue = Field(
        title="Extraordinary Component",
        description="Medium describing the extraordinary component.",
    )

    def medium(self, optical_axis: Axis) -> AnisotropicMedium:
        """
        Generate anisotropic medium.

        Parameters
        ----------
        optical_axis : Axis
            Optical axis of the uniaxial medium.

        Returns
        -------
        :class:`.AnisotropicMedium`
            The anisotropic medium representing the uniaxial medium.
        """

        components = ["xx", "yy", "zz"]
        mat_dict = dict.fromkeys(components, self.ordinary)
        mat_dict.update({components[optical_axis]: self.extraordinary})
        return AnisotropicMedium.model_validate(mat_dict)

    @property
    def summarize_mediums(self) -> dict[str, PoleResidue | Medium2D | MultiPhysicsMedium]:
        return {"ordinary": self.ordinary, "extraordinary": self.extraordinary}


class MaterialItemUniaxial(MaterialItem):
    """A material that includes several variants."""

    variants: dict[str, VariantItemUniaxial] = Field(
        title="Dictionary of available variants for this material",
        description="A dictionary of available variants for this material "
        "that maps from a key to the variant model.",
    )

    def medium(self, optical_axis: Axis) -> AnisotropicMedium:
        """The default medium."""
        return self.variants[self.default].medium(optical_axis)


PALIK_LOSSLESS_VARIANT = "Palik_Lossless"
PALIK_NOLOSS_VARIANT = "Palik_NoLoss"
PALIK_LOWLOSS_VARIANT = "Palik_LowLoss"

PALIK_LOSSLESS_ALIAS_WARNING = (
    f"The material-library variant '{PALIK_LOSSLESS_VARIANT}' is deprecated and maps to "
    f"'{PALIK_LOWLOSS_VARIANT}' because it contains a tiny fitted loss despite its name. "
    "Use 'Palik_NoLoss' where available for a zero-loss Palik model."
)

PALIK_LOSSLESS_ALIAS_MEDIUM_NAMES = {
    "Gallium Arsenide": "GaAs_Palik_LowLoss",
    "Germanium": "Ge_Palik_LowLoss",
    "Indium Phosphide": "InP_Palik_LowLoss",
    "Silicon (Crystalline)": "cSi_PalikLowLoss",
    "Silicon Dioxide": "SiO2_Palik_LowLoss",
}


class _DeprecatedVariantAliasDict(dict[str, VariantItem]):
    """Dictionary with deprecated lookup aliases that are hidden from iteration."""

    def __init__(
        self, variants: dict[str, VariantItem], aliases: dict[str, str] | None = None
    ) -> None:
        super().__init__(variants)
        self._aliases = aliases or {}

    def _resolve_alias(self, variant_name: str) -> str:
        """Return the canonical variant for a deprecated alias and warn once per lookup."""
        alias = self._aliases.get(variant_name)
        if alias is None:
            return variant_name
        log.warning(PALIK_LOSSLESS_ALIAS_WARNING)
        return alias

    def __contains__(self, variant_name: object) -> bool:
        """Treat deprecated aliases as present when their canonical target exists."""
        if isinstance(variant_name, str):
            alias = self._aliases.get(variant_name)
            if alias is not None:
                return super().__contains__(alias)
        return super().__contains__(variant_name)

    def __getitem__(self, variant_name: str) -> VariantItem:
        """Return deprecated aliases while preserving normal dict iteration."""
        return super().__getitem__(self._resolve_alias(variant_name))

    def get(self, variant_name: str, default: VariantItem | None = None) -> VariantItem | None:
        """Return deprecated aliases through the usual mapping helper."""
        alias = self._aliases.get(variant_name)
        if alias is not None:
            log.warning(PALIK_LOSSLESS_ALIAS_WARNING)
            return super().get(alias, default)
        return super().get(variant_name, default)

    def copy(self) -> _DeprecatedVariantAliasDict:
        """Preserve alias lookups when users explicitly copy the variants mapping."""
        return _DeprecatedVariantAliasDict(dict(self), aliases=dict(self._aliases))


def _with_palik_lossless_alias(material_item: MaterialItem) -> MaterialItem:
    """Keep old Palik_Lossless lookups available without listing them as variants."""
    if isinstance(material_item.variants, _DeprecatedVariantAliasDict):
        return material_item
    if PALIK_LOSSLESS_VARIANT in material_item.variants:
        return material_item

    alias_medium_names = PALIK_LOSSLESS_ALIAS_MEDIUM_NAMES.get(material_item.name)
    if alias_medium_names is None:
        return material_item

    lowloss_medium_name = alias_medium_names
    lowloss_variant = material_item.variants.get(PALIK_LOWLOSS_VARIANT)
    if lowloss_variant is None:
        return material_item

    if lowloss_variant.medium.name != lowloss_medium_name:
        return material_item

    object.__setattr__(
        material_item,
        "variants",
        _DeprecatedVariantAliasDict(
            material_item.variants,
            aliases={PALIK_LOSSLESS_VARIANT: PALIK_LOWLOSS_VARIANT},
        ),
    )
    return material_item


LiNbO3_Zelmon1997 = VariantItemUniaxial(
    ordinary=Sellmeier(
        coeffs=((2.6734, 0.01764), (1.2290, 0.05914), (12.614, 474.60)),
        frequency_range=(59958491600000.0, 749481145000000.0),
        name="LiNbO3_Zelmon1997",
    ).pole_residue,
    extraordinary=Sellmeier(
        coeffs=((2.9804, 0.02047), (0.5981, 0.0666), (8.9543, 416.08)),
        frequency_range=(59958491600000.0, 749481145000000.0),
        name="LiNbO3_Zelmon1997",
    ).pole_residue,
    reference=[material_refs["Zelmon1997"]],
    data_url="https://refractiveindex.info/data_csv.php?datafile=database/data-nk/main/LiNbO3/Zelmon-e.yml",
)

Ag_Rakic1998BB = VariantItem(
    medium=PoleResidue(
        name="Ag_Rakic1998BB",
        eps_inf=1.0,
        poles=[
            (
                (-2.483370555410649e16 + 1j * 0.0),
                (1.5744113130000786e16 + 1j * 0.0),
            ),
            (
                (-427051442026657.8 - 1j * 7267130897123369.0),
                (285966966530009.5 - 1j * 259085203357223.06),
            ),
            (
                (-458495605638445.94 - 1j * 6501478008755004.0),
                (950826754614684.4 + 1j * 1709069477589139.5),
            ),
            (
                (-200096391057984.5 + 1j * 0.0),
                (868238077367508.6 + 1j * 0.0),
            ),
            (
                (-18638609734.40867 + 1j * 0.0),
                (1.033846354957434e18 + 1j * 0.0),
            ),
            (
                (-74449645789281.8 + 1j * 0.0),
                (-1.0335652795499411e18 + 1j * 0.0),
            ),
        ],
        frequency_range=(24179892422719.273, 1208994621135963.5),
    ),
    reference=[material_refs["Rakic1998"]],
    data_url="https://refractiveindex.info/data_csv.php?datafile=database/data-nk/"
    "main/Ag/Rakic-BB.yml",
)

Ag_Rakic1998BB_IR = VariantItem(
    medium=PoleResidue(
        name="Ag_Rakic1998BB_IR",
        eps_inf=2.080628548409516,
        poles=[
            (
                (-74116405167315.4 - 1j * 0.0),
                (-1.0385354711010449e18 - 1j * 0.0),
            ),
            (
                (-199290207342.26654 - 1j * 0.0),
                (1.0396417727844411e18 - 1j * 0.0),
            ),
            (
                (-622425347820110.2 - 1j * 6539570627133650.0),
                (936046890626063.0 + 1j * 1966533189396127.8),
            ),
        ],
        frequency_range=(24179892422719.273, 214137470000000.0),
    ),
    reference=[material_refs["Rakic1998"]],
    data_url="https://refractiveindex.info/data_csv.php?datafile=database/data-nk/"
    "main/Ag/Rakic-BB.yml",
)

Ag_RakicLorentzDrude1998 = VariantItem(
    medium=PoleResidue(
        name="Ag_RakicLorentzDrude1998",
        eps_inf=1.0,
        poles=(
            (0j, (1.085598639948276e18 + 0j)),
            ((-72924837503814.11 + 0j), (-1.085598639948276e18 - 0j)),
            ((-272940800787927.5 + 0j), (1136578330456760.5 + 0j)),
            ((-5630932502125024 + 0j), (-1136578330456760.5 - 0j)),
            ((-343354443247124.75 - 6799173351259867j), 1708652013864486.5j),
            ((-49376192059874.13 - 1.2435106032980426e16j), 82876469878486.64j),
            ((-695824491182226.4 - 1.3781951983423364e16j), 5710269496109004j),
            ((-1837553978351315.8 - 3.0771118889340676e16j), 1.7190386342847058e16j),
        ),
        frequency_range=(24179892422719.273, 1208994621135963.5),
    ),
    reference=[material_refs["Rakic1998"]],
    data_url="https://refractiveindex.info/data_csv.php?datafile=database/data-nk/"
    "main/Ag/Rakic-LD.yml",
)

Ag_JohnsonChristy1972 = VariantItem(
    medium=PoleResidue(
        name="Ag_JohnsonChristy1972",
        eps_inf=1.0,
        poles=[
            (
                (-2208321413508536.0 - 1j * 5948722238054062.0),
                (6900545964172845.0 + 1j * 2859999298140436.0),
            ),
            (
                (-454071262600809.06 - 1j * 6045413448570748.0),
                (900170460600995.0 - 1j * 571876746089699.8),
            ),
            (
                (-440069917626400.94 - 1j * 1501005120704881.8),
                (-624427112339892.1 - 1j * 709709047591569.2),
            ),
            (
                (-90645222183131.28 - 1j * 1207503089909680.2),
                (36466030087984.664 - 1j * 105435949158129.28),
            ),
            (
                (-6882166770889.508 - 1j * 101422373698470.4),
                (2204704963300273.8 + 1j * 9.259919916690406e17),
            ),
        ],
        frequency_range=(154771532566312.25, 1595489401708072.2),
    ),
    reference=[material_refs["JohnsonChristy1972"]],
    data_url="https://refractiveindex.info/data_csv.php?datafile=database/data-nk/"
    "main/Ag/Johnson.yml",
)

Ag_Yang2015Drude = VariantItem(
    medium=PoleResidue(
        name="Ag_Yang2015Drude",
        eps_inf=1.0,
        poles=[
            (
                (-36518002732.60446 - 1j * 0.0),
                (1.6187698336621732e18 - 1j * 0.0),
            ),
            (
                (-55686381370850.8 - 1j * 0.0),
                (-1.6187268610953428e18 - 1j * 0.0),
            ),
            (
                (-7162984458004041.0 - 1j * 2943343329124741.0),
                (1.5545804484194954e16 - 1j * 1.936229457319725e16),
            ),
        ],
        frequency_range=(154771532566312.25, 1595489401708072.2),
    ),
    reference=[material_refs["Yang2015"]],
    data_url="https://refractiveindex.info/data_csv.php?datafile=database/data-nk/main/Ag/Yang.yml",
)

Al_Rakic1995 = VariantItem(
    medium=PoleResidue(
        name="Al_Rakic1995",
        eps_inf=1.0,
        poles=[
            (
                (-3478273083283395.0 - 1j * 2663678407664229.5),
                (-1.7387584269061222e16 - 1j * 4351056211589479.5),
            ),
            (
                (-357768743237230.5 - 1j * 2218498642414248.8),
                (9464952935418408.0 + 1j * 1.0850783579943464e16),
            ),
            (
                (-242766411045799.88 - 1j * 1820240789275247.2),
                (-1362328161533225.2 + 1j * 1583869480765488.2),
            ),
            (
                (-8704513168412.099 - 1j * 0.0),
                (1.2660614263809472e18 - 1j * 0.0),
            ),
            (
                (-160944103454154.38 - 1j * 0.0),
                (-1.2565462108051812e18 - 1j * 0.0),
            ),
        ],
        frequency_range=(151926744799612.75, 1.5192674479961274e16),
    ),
    reference=[material_refs["Rakic1995"]],
    data_url="https://refractiveindex.info/data_csv.php?datafile=database/data-nk/"
    "main/Al/Rakic.yml",
)

Al_RakicLorentzDrude1998 = VariantItem(
    medium=PoleResidue(
        name="Al_RakicLorentzDrude1998",
        eps_inf=1.0,
        poles=(
            (0j, (1.896844347324609e18 + 0j)),
            ((-71405570055817.98 + 0j), (-1.896844347324609e18 - 0j)),
            ((-194545209645174.6 + 0j), (5.0321060071503546e17 + 0j)),
            ((-311370850537535.75 + 0j), (-5.0321060071503546e17 - 0j)),
            ((-237005721887395.88 - 2333745139453868j), 5548539400655871j),
            ((-1026265161121383.9 - 2547917843202808.5j), 1.6872706975652862e16j),
            ((-2569081254561451.5 - 4608729293067524j), 1685784870483934.2j),
        ),
        frequency_range=(1208986804855.426, 4835986224028907.0),
    ),
    reference=[material_refs["Rakic1998"]],
    data_url="https://refractiveindex.info/data_csv.php?datafile=database/data-nk/"
    "main/Al/Rakic-LD.yml",
)

Al2O3_Horiba = VariantItem(
    medium=PoleResidue(
        name="Al2O3_Horiba",
        eps_inf=1.0,
        poles=[((-0.0 - 1j * 1.856240967961668e16), (0.0 + 1j * 1.4107431356508676e16))],
        frequency_range=(145079354536315.6, 1450793545363156.0),
    ),
    reference=[material_refs["Horiba"]],
)

AlAs_Horiba = VariantItem(
    medium=PoleResidue(
        name="AlAs_Horiba",
        eps_inf=1.0,
        poles=[
            (
                (-287141547671268.06 - 1j * 6859562349716031.0),
                (0.0 + 1j * 2.4978200955702556e16),
            )
        ],
        frequency_range=(0.0, 725396772681578.0),
    ),
    reference=[material_refs["Horiba"]],
)

AlAs_FernOnton1971 = VariantItem(
    medium=PoleResidue(
        name="AlAs_FernOnton1971",
        eps_inf=2.0792,
        poles=[
            ((0.0 + 1j * 6674881541314847.0), (-0.0 - 1j * 2.0304989648679764e16)),
            ((0.0 + 1j * 68198825885555.74), (-0.0 - 1j * 64788884591277.95)),
        ],
        frequency_range=(136269299354975.81, 535343676037405.0),
    ),
    reference=[material_refs["FernOnton1971"]],
    data_url="https://refractiveindex.info/data_csv.php?datafile=database/data-nk/"
    "main/AlAs/Fern.yml",
)

AlGaN_Horiba = VariantItem(
    medium=PoleResidue(
        name="AlGaN_Horiba",
        eps_inf=1.0,
        poles=[
            (
                (-96473482947754.08 - 1j * 1.0968686723518324e16),
                (0.0 + 1j * 1.974516343551917e16),
            )
        ],
        frequency_range=(145079354536315.6, 967195696908770.8),
    ),
    reference=[material_refs["Horiba"]],
)

AlN_Horiba = VariantItem(
    medium=PoleResidue(
        name="AlN_Horiba",
        eps_inf=1.0,
        poles=[((-0.0 - 1j * 1.354578856633347e16), (0.0 + 1j * 2.2391188500149228e16))],
        frequency_range=(181349193170394.5, 1148544890079165.2),
    ),
    reference=[material_refs["Horiba"]],
)

AlxOy_Horiba = VariantItem(
    medium=PoleResidue(
        name="AlxOy_Horiba",
        eps_inf=1.0,
        poles=[
            (
                (-654044636362332.8 - 1j * 1.9535949662203744e16),
                (0.0 + 1j * 2.123004231270711e16),
            )
        ],
        frequency_range=(145079354536315.6, 1450793545363156.0),
    ),
    reference=[material_refs["Horiba"]],
)

Aminoacid_Horiba = VariantItem(
    medium=PoleResidue(
        name="Aminoacid_Horiba",
        eps_inf=1.0,
        poles=[((-0.0 - 1j * 2.2518582114198596e16), (0.0 + 1j * 5472015453750259.0))],
        frequency_range=(362698386340789.0, 1208994621135963.5),
    ),
    reference=[material_refs["Horiba"]],
)

Au_Olmon2012evaporated = VariantItem(
    medium=PoleResidue(
        name="Au_Olmon2012evaporated",
        eps_inf=1.0,
        poles=[
            (
                (-1520562960691848.0 - 1j * 3928154716170055.0),
                (1.1174011615554154e16 + 1j * 1251700157597656.8),
            ),
            (
                (-1507358279340936.5 + 1j * 0.0),
                (3408362175303679.0 + 1j * 0.0),
            ),
            (
                (-44944854555.44156 - 1j * 104969866980268.64),
                (-1875596504315.7332 - 1j * 2796754031259.1025),
            ),
            (
                (-85925833547817.62 + 1j * 0.0),
                (-7.855548809693609e17 + 1j * 0.0),
            ),
            (
                (-190719132146.27554 - 1j * 2277191501416.505),
                (7.852261343792833e17 + 1j * 8.554273616444449e18),
            ),
        ],
        frequency_range=(12025369359446.29, 999308193769986.8),
    ),
    reference=[material_refs["Olmon2012"]],
    data_url="https://refractiveindex.info/data_csv.php?datafile=database/data-nk/"
    "main/Au/Olmon-ev.yml",
)

Au_Olmon2012evaporated_IR = VariantItem(
    medium=PoleResidue(
        name="Au_Olmon2012evaporated_IR",
        eps_inf=5.632132676065586,
        poles=(
            (
                (-208702733035001.06 - 205285605362650.1j),
                (-5278287093117479 + 1877992342820785.5j),
            ),
            (
                (-5802337384288.284 - 6750566414892.662j),
                (4391102400709820 + 6.164348337888482e18j),
            ),
            (
                (-56597670698540.76 - 8080114483410.944j),
                (895004078070708.5 + 5.346045584373232e18j),
            ),
        ),
        frequency_range=(12025369359446.29, 85654988000000.0),
    ),
    reference=[material_refs["Olmon2012"]],
    data_url="https://refractiveindex.info/data_csv.php?datafile=database/data-nk/"
    "main/Au/Olmon-ev.yml",
)

Au_Olmon2012stripped = VariantItem(
    medium=PoleResidue(
        name="Au_Olmon2012stripped",
        eps_inf=1.8661249761826162,
        poles=(
            (
                (-909376873.6996255 - 4596858854036.634j),
                (6.746525460331022e16 + 5.926266046979877e18j),
            ),
            (
                (-2211438487782.0527 + 0j),
                (5.660718217037341e17 + 6.245539733887402e18j),
            ),
            (
                (-102715947550852.86 - 10649989484.773024j),
                (-6.333331223161453e17 + 5.199295820846523e18j),
            ),
        ),
        frequency_range=(12025369359446.29, 999308193769986.8),
    ),
    reference=[material_refs["Olmon2012"]],
    data_url="https://refractiveindex.info/data_csv.php?datafile=database/data-nk/"
    "main/Au/Olmon-ts.yml",
)

Au_Olmon2012crystal = VariantItem(
    medium=PoleResidue(
        name="Au_Olmon2012crystal",
        eps_inf=2.6361315520011614,
        poles=[
            (
                (-193665366999934.53 - 1j * 0.0),
                (3926767015155341.0 - 1j * 0.0),
            ),
            (
                (-128971779806825.61 - 1j * 0.0),
                (-3.5927350834150784e17 - 1j * 0.0),
            ),
            (
                (-41481046459.60286 - 1j * 2244366051795.782),
                (3.554312841202226e17 + 1j * 1.3907487154141815e19),
            ),
        ],
        frequency_range=(12025369359446.29, 999308193769986.8),
    ),
    reference=[material_refs["Olmon2012"]],
    data_url="https://refractiveindex.info/data_csv.php?datafile=database/data-nk/"
    "main/Au/Olmon-sc.yml",
)

Au_Olmon2012Drude = VariantItem(
    medium=PoleResidue(
        name="Au_Olmon2012Drude",
        eps_inf=2.6361315520011614,
        poles=[
            (
                (-193665366999934.53 - 1j * 0.0),
                (3926767015155341.0 - 1j * 0.0),
            ),
            (
                (-128971779806825.61 - 1j * 0.0),
                (-3.5927350834150784e17 - 1j * 0.0),
            ),
            (
                (-41481046459.60286 - 1j * 2244366051795.782),
                (3.554312841202226e17 + 1j * 1.3907487154141815e19),
            ),
        ],
        frequency_range=(12025369359446.29, 241798930000000),
    ),
    reference=[material_refs["Olmon2012"]],
    data_url="https://refractiveindex.info/data_csv.php?datafile=database/data-nk/"
    "main/Au/Olmon-sc.yml",
)

Au_JohnsonChristy1972 = VariantItem(
    medium=PoleResidue(
        name="Au_JohnsonChristy1972",
        eps_inf=1.0,
        poles=[
            (
                (-569801961707506.2 - 1j * 5919036552773929.0),
                (547547687971021.3 + 1j * 294290120688174.94),
            ),
            (
                (-1589807268517576.2 - 1j * 3491672819475904.0),
                (1.0929163739969444e16 + 1j * 422154272517619.8),
            ),
            (
                (-291288687923238.2 - 1j * 3648189488671429.0),
                (252866071416221.06 - 1j * 269873948615172.56),
            ),
            (
                (-249515901504479.2 - 1j * 1081669083787891.2),
                (-343676981321731.5 - 1j * 707306264998813.6),
            ),
            (
                (-53249179312995.555 - 1j * 283474011170933.5),
                (2393359178586368.5 + 1j * 2.9476885206131206e17),
            ),
        ],
        frequency_range=(154751311505403.34, 1595872899899471.8),
    ),
    reference=[material_refs["JohnsonChristy1972"]],
    data_url="https://refractiveindex.info/data_csv.php?datafile=database/data-nk/"
    "main/Au/Johnson.yml",
)

Au_RakicLorentzDrude1998 = VariantItem(
    medium=PoleResidue(
        name="Au_RakicLorentzDrude1998",
        eps_inf=1.0,
        poles=(
            (0j, (8.882136852663547e17 + 0j)),
            ((-80521174743794.75 + 0j), (-8.882136852663547e17 - 0j)),
            ((-183071727483533.3 - 603332360445186.9j), 3743420309393973.5j),
            ((-262073634779331.9 - 1233457817766871.8j), 762938741152796.4j),
            ((-660881339878315.4 - 4462028230599516j), 1497407504712811j),
            ((-1894526507651170.8 - 6258461223088549j), 9036929133946472j),
            ((-1681829064931712.8 - 2.0166634496554556e16j), 2.0457430700884664e16j),
        ),
        frequency_range=(48359862240289.07, 1208986804855426.0),
    ),
    reference=[material_refs["Rakic1998"]],
    data_url="https://refractiveindex.info/data_csv.php?datafile=database/data-nk/"
    "main/Au/Rakic-LD.yml",
)

BK7_Zemax = VariantItem(
    medium=PoleResidue(
        name="BK7_Zemax",
        eps_inf=1,
        poles=[
            ((0.0 + 1j * 2.431642149296798e16), (-0.0 - 1j * 1.2639823249559002e16)),
            ((0.0 + 1j * 1.3313466757556814e16), (-0.0 - 1j * 1542979833250087.0)),
            ((0.0 + 1j * 185098620483566.44), (-0.0 - 1j * 93518250617894.06)),
        ],
        frequency_range=(119916983432378.72, 999308195269822.8),
    ),
    reference=[material_refs["Zemax"]],
    data_url="https://refractiveindex.info/data_csv.php?datafile=data/glass/schott/N-BK7.yml",
)

Be_Rakic1998BB = VariantItem(
    medium=PoleResidue(
        name="Be_Rakic1998BB",
        eps_inf=1.0,
        poles=[
            (
                (-4793730492700408.0 - 1j * 5850528104977031.0),
                (-4.2135306149588344e16 + 1j * 3.4517986755194536e16),
            ),
            (
                (-3524988896441455.5 - 1j * 0.0),
                (2.6127316531938252e16 - 1j * 0.0),
            ),
            (
                (-982200146551476.4 - 1j * 0.0),
                (5522570243163596.0 - 1j * 0.0),
            ),
            (
                (-10935721489917.447 - 1j * 58713070368029.39),
                (16347856591442.65 + 1j * 6140980155310.667),
            ),
            (
                (-15335088367.172012 - 1j * 0.0),
                (6.053398252993956e17 - 1j * 0.0),
            ),
            (
                (-54087838046595.13 - 1j * 0.0),
                (-6.790592340263299e17 - 1j * 0.0),
            ),
            (
                (-61355594698420.5 - 1j * 0.0),
                (8.1642489720691e16 - 1j * 0.0),
            ),
            (
                (-306968735452312.44 - 1j * 0.0),
                (2633724857568732.0 - 1j * 0.0),
            ),
        ],
        frequency_range=(4835978484543.8545, 1208994621135963.5),
    ),
    reference=[material_refs["Rakic1998"]],
    data_url="https://refractiveindex.info/data_csv.php?datafile=database/data-nk/"
    "main/Be/Rakic-BB.yml",
)

Be_Rakic1998BB_IR = VariantItem(
    medium=PoleResidue(
        name="Be_Rakic1998BB_IR",
        eps_inf=1.0,
        poles=[
            (
                (-1737739552967275.2 - 1j * 0.0),
                (2.3924381023090224e16 - 1j * 0.0),
            ),
            (
                (-151352273074186.28 - 1j * 0.0),
                (4367049766016236.5 - 1j * 0.0),
            ),
            (
                (-53296876831178.09 - 1j * 0.0),
                (-6.001139611206947e17 - 1j * 0.0),
            ),
            (
                (-20238020062.550835 - 1j * 0.0),
                (6.055916356024831e17 - 1j * 0.0),
            ),
        ],
        frequency_range=(4835978484543.8545, 299792458000000.0),
    ),
    reference=[material_refs["Rakic1998"]],
    data_url="https://refractiveindex.info/data_csv.php?datafile=database/data-nk/"
    "main/Be/Rakic-BB.yml",
)

Be_RakicLorentzDrude1998 = VariantItem(
    medium=PoleResidue(
        name="Be_RakicLorentzDrude1998",
        eps_inf=1.0,
        poles=(
            (0j, (6.246378779510136e17 + 0j)),
            ((-53174360679864.46 + 0j), (-6.246378779510136e17 - 0j)),
            ((-9163427576987.25 + 0j), (4884108194218923 + 0j)),
            ((-2518897605888568 + 0j), (-4884108194218923 - 0j)),
            ((-531334375653411.75 + 0j), (1.351759826496727e16 + 0j)),
            ((-4626578610293440 + 0j), (-1.351759826496727e16 - 0j)),
            ((-3383408606687375.5 - 3455109465888044.5j), 6.065482659167511e16j),
            ((-1368859970644510.8 - 6859457195810405j), 7493848504616172j),
        ),
        frequency_range=(4835978484543.8545, 1208994621135963.5),
    ),
    reference=[material_refs["Rakic1998"]],
    data_url="https://refractiveindex.info/data_csv.php?datafile=database/data-nk/"
    "main/Be/Rakic-LD.yml",
)

CaF2_Horiba = VariantItem(
    medium=PoleResidue(
        name="CaF2_Horiba",
        eps_inf=1.0,
        poles=[((-0.0 - 1j * 2.376134288665943e16), (0.0 + 1j * 1.2308375615289586e16))],
        frequency_range=(181349193170394.5, 1148544890079165.2),
    ),
    reference=[material_refs["Horiba"]],
)

Cellulose_Sultanova2009 = VariantItem(
    medium=PoleResidue(
        name="Cellulose_Sultanova2009",
        eps_inf=1,
        poles=[((0.0 + 1j * 1.7889308287957964e16), (-0.0 - 1j * 1.0053791257832376e16))],
        frequency_range=(284973819943865.75, 686338046201801.2),
    ),
    reference=[material_refs["Sultanova2009"]],
    data_url="https://refractiveindex.info/data_csv.php?"
    "datafile=data/organic/(C6H10O5)n%20-%20cellulose/Sultanova.yml",
)

Cr_Rakic1998BB = VariantItem(
    medium=PoleResidue(
        name="Cr_Rakic1998BB",
        eps_inf=1.0,
        poles=[
            (
                (-1.7008332048761648e16 - 1j * 0.0),
                (-2933671096810391.0 - 1j * 0.0),
            ),
            (
                (-2005256053898001.5 - 1j * 2381844289356685.0),
                (-9650588319181022.0 + 1j * 5.668939669664152e16),
            ),
            (
                (-220104169827627.3 - 1j * 0.0),
                (3249240187754787.0 - 1j * 0.0),
            ),
            (
                (-5697781842.205581 - 1j * 0.0),
                (2.8278212440731635e17 - 1j * 0.0),
            ),
            (
                (-21648484807783.637 - 1j * 0.0),
                (4446929285969332.0 - 1j * 0.0),
            ),
            (
                (-73023335173331.34 - 1j * 0.0),
                (-2.7744487842401034e17 - 1j * 0.0),
            ),
        ],
        frequency_range=(4835362227919.29, 1208840556979822.5),
    ),
    reference=[material_refs["Rakic1998"]],
    data_url="https://refractiveindex.info/data_csv.php?datafile=database/data-nk/"
    "main/Cr/Rakic-BB.yml",
)

Cr_Rakic1998BB_IR = VariantItem(
    medium=PoleResidue(
        name="Cr_Rakic1998BB_IR",
        eps_inf=1.0,
        poles=[
            (
                (-73056488139432.73 - 1j * 0.0),
                (-2.7457982793225763e17 - 1j * 0.0),
            ),
            (
                (-145384800564.84518 - 1j * 0.0),
                (2.8558672134946093e17 - 1j * 0.0),
            ),
            (
                (-2137728163059224.0 - 1j * 740097502616341.5),
                (5846984237158586.0 + 1j * 9.545555973191486e16),
            ),
        ],
        frequency_range=(4835362227919.29, 93685143125000.0),
    ),
    reference=[material_refs["Rakic1998"]],
    data_url="https://refractiveindex.info/data_csv.php?datafile=database/data-nk/"
    "main/Cr/Rakic-BB.yml",
)

Cr_RakicLorentzDrude1998 = VariantItem(
    medium=PoleResidue(
        name="Cr_RakicLorentzDrude1998",
        eps_inf=1.0,
        poles=(
            (0j, (3.137852964800087e17 + 0j)),
            ((-71405570055817.98 + 0j), (-3.137852964800087e17 - 0j)),
            ((-7016061501736.5 + 0j), (4187160341714059 + 0j)),
            ((-4816658085885968 + 0j), (-4187160341714059 - 0j)),
            ((-441634229628193.8 + 0j), (1.8197032850966132e16 + 0j)),
            ((-1541009790006752 + 0j), (-1.8197032850966132e16 - 0j)),
            ((-2032779845418818.5 - 2196724138579423.8j), 6.975894511603245e16j),
            ((-1014111021537414.9 - 1.3292945008240806e16j), 8277289379024516j),
        ),
        frequency_range=(4835978484543.8545, 1208994621135963.5),
    ),
    reference=[material_refs["Rakic1998"]],
    data_url="https://refractiveindex.info/data_csv.php?datafile=database/data-nk/"
    "main/Cr/Rakic-LD.yml",
)

Cu_JohnsonChristy1972 = VariantItem(
    medium=PoleResidue(
        name="Cu_JohnsonChristy1972",
        eps_inf=1.0,
        poles=[
            (
                (-1252374269166904.5 - 1j * 7829718683182146.0),
                (-660427953437394.4 + 1j * 2056312746029814.8),
            ),
            (
                (-500398492478025.6 - 1j * 3123892988543211.0),
                (2348376270614990.0 - 1j * 1390125983450377.5),
            ),
            (
                (-775228900492209.9 - 1j * 1254493598977193.5),
                (-7078896427414573.0 - 1j * 1.007782055107454e16),
            ),
            (
                (-92770480154285.34 - 1j * 1365410212347161.2),
                (323897486922091.44 + 1j * 93507890692118.31),
            ),
            (
                (-8965554692589.553 - 1j * 256329468465111.16),
                (1.6798480681493582e16 + 1j * 2.8078798578850288e17),
            ),
        ],
        frequency_range=(154771532266391.3, 1595489398616285.2),
    ),
    reference=[material_refs["JohnsonChristy1972"]],
    data_url="https://refractiveindex.info/data_csv.php?datafile=database/data-nk/"
    "main/Cu/Johnson.yml",
)

Cu_RakicLorentzDrude1998 = VariantItem(
    medium=PoleResidue(
        name="Cu_RakicLorentzDrude1998",
        eps_inf=1.0,
        poles=(
            (0j, (1.7076849079038659e18 + 0j)),
            ((-45578023439883.81 + 0j), (-1.7076849079038659e18 - 0j)),
            ((-287141547671268.06 - 336166890703636.9j), 2.4562370654105788e16j),
            ((-802173212541955.2 - 4420275938629774.5j), 3184779293720060.5j),
            ((-2440703155205778.5 - 7673302022556902j), 1.2754146107549982e16j),
            ((-3270223181811663.5 - 1.6667627171842064e16j), 5181342297925362j),
        ),
        frequency_range=(24176811129032.258, 1450795867208672.2),
    ),
    reference=[material_refs["Rakic1998"]],
    data_url="https://refractiveindex.info/data_csv.php?datafile=database/data-nk/"
    "main/Cu/Rakic-LD.yml",
)

FusedSilica_Zemax = VariantItem(
    medium=PoleResidue(
        name="FusedSilica_Zemax",
        eps_inf=1,
        poles=[
            ((0.0 + 1j * 2.7537034527932452e16), (-0.0 - 1j * 9585177720141492.0)),
            ((0.0 + 1j * 1.620465316968868e16), (-0.0 - 1j * 3305284173070520.5)),
            ((0.0 + 1j * 190341645710801.38), (-0.0 - 1j * 85413852993771.3)),
        ],
        frequency_range=(44745143071783.1, 1427583136099746.8),
    ),
    reference=[material_refs["Malitson1965"], material_refs["Tan1998"]],
    data_url="https://refractiveindex.info/data_csv.php?datafile=database/data-nk/"
    "main/SiO2/Malitson.yml",
)

FusedSilica_Zemax_Visible_PMLStable = VariantItem(
    medium=PoleResidue(
        name="FusedSilica_Zemax_Visible_PMLStable",
        eps_inf=1,
        poles=((-2.0054061849947e16j, 1.1008717135056432e16j),),
        frequency_range=(382925607524582.94, 739315556426623.9),
    ),
    reference=[material_refs["Malitson1965"], material_refs["Tan1998"]],
    data_url="https://refractiveindex.info/data_csv.php?datafile=database/data-nk/"
    "main/SiO2/Malitson.yml",
)

FusedSilica_Zemax_PMLStable = VariantItem(
    medium=PoleResidue(
        name="FusedSilica_Zemax_PMLStable",
        eps_inf=1,
        poles=((-1.7312422399228024e16j, 9389865424501702j),),
        frequency_range=(150347270878132.4, 739315556426623.9),
    ),
    reference=[material_refs["Malitson1965"], material_refs["Tan1998"]],
    data_url="https://refractiveindex.info/data_csv.php?datafile=database/data-nk/"
    "main/SiO2/Malitson.yml",
)

GaAs_Skauli2003 = VariantItem(
    medium=PoleResidue(
        name="GaAs_Skauli2003",
        eps_inf=5.372514,
        poles=[
            ((0.0 + 1j * 4250781024557878.5), (-0.0 - 1j * 1.1618961579876792e16)),
            ((0.0 + 1j * 2153617667595138.0), (-0.0 - 1j * 26166023937747.41)),
            ((0.0 + 1j * 51024513930292.87), (-0.0 - 1j * 49940804278927.375)),
        ],
        frequency_range=(17634850504761.58, 309064390289635.9),
    ),
    reference=[material_refs["Skauli2003"]],
    data_url="https://refractiveindex.info/data_csv.php?datafile=database/data-nk/"
    "main/GaAs/Skauli.yml",
)

GaAs_Palik_Lossy = VariantItem(
    medium=PoleResidue(
        name="GaAs_Palik_Lossy",
        eps_inf=1.0,
        poles=[
            (
                (-279009922360229.66 - 1j * 7540534540879450.0),
                (-2098570260345821.5 + 1j * 1068731155756624.8),
            ),
            (
                (-848502325580467.2 - 1j * 6776421302325818.0),
                (1.1355772698538406e16 + 1j * 1.151506674670278e16),
            ),
            (
                (-333841468018367.75 - 1j * 2161251923134449.0),
                (113942922925879.11 - 1j * 190870726210118.72),
            ),
            (
                (-1137007722945610.5 - 1j * 5590695255927143.0),
                (-3141402186504010.5 + 1j * 1.3358549633112706e16),
            ),
            (
                (-303689161580505.9 - 1j * 4698565400865453.0),
                (1067141610043567.8 + 1j * 3313576218621913.5),
            ),
            (
                (-122020624036776.08 - 1j * 4449479376721530.5),
                (-264108481879804.78 + 1j * 1003958179608862.0),
            ),
        ],
        frequency_range=(230609583076923.06, 1362692990909091.0),
    ),
    reference=[material_refs["Palik_Lossy"]],
)

GaAs_Palik_NoLoss = VariantItem(
    medium=PoleResidue(
        name="GaAs_Palik_NoLoss",
        eps_inf=1.0,
        poles=[
            (
                (0.0 + 1j * 333524452168.78815),
                (0.0 - 1j * 6392996319943263.0),
            ),
            (
                (0.0 + 1j * 5350121052897750.0),
                (0.0 - 1j * 2.6358366870929732e16),
            ),
        ],
        frequency_range=(44087126176470.59, 272538598181818.16),
    ),
    reference=[material_refs["Palik_NoLoss"]],
)

GaAs_Palik_LowLoss = VariantItem(
    medium=PoleResidue(
        name="GaAs_Palik_LowLoss",
        eps_inf=1.2402134414081076,
        poles=[
            (
                (-11863066.22341989 - 1j * 52037903228727.2),
                (6445302.8779493505 + 1j * 42726737123058.85),
            ),
            (
                (-54720906910.3452 - 1j * 5268083432765163.0),
                (529749352233.0459 + 1j * 2.5313510635938664e16),
            ),
        ],
        frequency_range=(9993081933333.334, 272538598181818.16),
    ),
    reference=[material_refs["Palik_LowLoss"]],
)

Ge_Icenogle1976 = VariantItem(
    medium=PoleResidue(
        name="Ge_Icenogle1976",
        eps_inf=9.28156000004953,
        poles=[
            ((0.0 + 1j * 2836329349380603.5), (-0.0 - 1j * 9542546463056102.0)),
            ((0.0 + 1j * 30278857121656.766), (-0.0 - 1j * 3225758043455.7036)),
        ],
        frequency_range=(24982704881745.566, 119916983432378.72),
    ),
    reference=[material_refs["Icenogle1976"], material_refs["Barnes1979"]],
    data_url="https://refractiveindex.info/data_csv.php?datafile=database/data-nk/"
    "main/Ge/Icenogle.yml",
)

Ge_Palik_NoLoss = VariantItem(
    medium=PoleResidue(
        name="Ge_Palik_NoLoss",
        eps_inf=1.0,
        poles=[
            (
                (0.0 + 1j * 4000960662304286.5),
                (0.0 - 1j * 3.0004708993106404e16),
            ),
        ],
        frequency_range=(20675341931034.484, 149896229000000.0),
    ),
    reference=[material_refs["Palik_NoLoss"]],
)

Ge_Palik_LowLoss = VariantItem(
    medium=PoleResidue(
        name="Ge_Palik_LowLoss",
        eps_inf=1.0,
        poles=[
            (
                (-22827287748.34302 - 1j * 3788297395976083.5),
                (341986806721.7298 + 1j * 2.8377214482187104e16),
            ),
        ],
        frequency_range=(14989622900000.0, 249827048333333.34),
    ),
    reference=[material_refs["Palik_LowLoss"]],
)

Ge_Palik_Lossy = VariantItem(
    medium=PoleResidue(
        name="Ge_Palik_Lossy",
        eps_inf=1.0,
        poles=[
            (
                (-294614885251039.44 - 1j * 6573759918096796.0),
                (-2746914920270423.0 + 1j * 3753037064385678.5),
            ),
            (
                (-1883765676655915.5 - 1j * 5607839821270834.0),
                (4723235811351269.0 + 1j * 3.2155902184095756e16),
            ),
            (
                (-409425255369698.5 - 1j * 3436854587631159.5),
                (3222567364435628.0 + 1j * 5525170432168548.0),
            ),
            (
                (-4565802472669.914 - 1j * 3111619249174371.0),
                (2903778433269.1646 - 1j * 11126863944525.096),
            ),
            (
                (-131917314333404.67 - 1j * 3202793315510643.5),
                (214800976241258.6 + 1j * 937971280362587.1),
            ),
        ],
        frequency_range=(214137470000000.0, 1199169832000000.0),
    ),
    reference=[material_refs["Palik_Lossy"]],
)

Ge_Nunley = VariantItem(
    medium=PoleResidue(
        name="Ge_Nunley",
        eps_inf=1.0,
        poles=[
            (
                (-1055692444928990.6 - 1.2069979211686232e16j),
                (1107423663161164.6 - 3093285987380348.5j),
            ),
            ((-3216853350289244 - 8291440266759688j), (5700802820282152 + 2.5732993737266788e16j)),
            ((-541881030908034.7 - 8791646324216410j), (-937105384931749 + 541020934229084.2j)),
            (
                (-976547995477318.4 - 6543303598934721j),
                (-1.0737211590127532e16 + 1.3992511131747758e16j),
            ),
            ((-284492778464386.2 - 6696785988114651j), (-2168480500787241.8 - 696493506373301.2j)),
            ((-900354155072.7748 - 6663372533420019j), (183103725786.0085 - 141583942490.17123j)),
            ((-7073535582440.927 - 6580195498223051j), (262790152.22284198 + 378843848585.63043j)),
            ((-270911770733075.62 - 6447924271229311j), (1207997886295188 + 1471602437337227.5j)),
            ((-255993466255361.62 - 5008805217676040j), (445613879952076.8 - 223801118798973.3j)),
            ((-344512832764646.75 - 4858405095425429j), (563341672141455.9 + 1286782730502855.5j)),
            ((-695769101918184.8 - 3620945512916543j), (5328924572609390 + 5529617018233712j)),
            ((-252629750588162 - 3513125220372237.5j), (109779730682375.69 + 2230058891609707j)),
            (
                (-141325408476762.66 - 3230461058493017.5j),
                (126429312165475.8 + 1257642646101307.8j),
            ),
            ((-498950664609906.4 - 1179396998903782.5j), (180816169912939.7 - 297981189005482.4j)),
            ((-37573278339073.01 - 1213096851555221.8j), (13359584691343.223 - 3071910249946.702j)),
            ((-99651215401004 - 1635758773069449.5j), (5479484262079.737 - 6773804202146.6045j)),
        ],
        frequency_range=(1.209e14, 1.595e15),
    ),
    reference=[material_refs["Nunley2016"]],
    data_url="https://refractiveindex.info/data_csv.php?datafile=database/data/main/Ge/nk/Nunley.yml",
)

GeOx_Horiba = VariantItem(
    medium=PoleResidue(
        name="GeOx_Horiba",
        eps_inf=1.0,
        poles=[
            (
                (-351710414211103.44 - 1j * 2.4646085673376252e16),
                (0.0 + 1j * 2.02755336442934e16),
            )
        ],
        frequency_range=(145079354536315.6, 967195696908770.8),
    ),
    reference=[material_refs["Horiba"]],
)

H2O_Horiba = VariantItem(
    medium=PoleResidue(
        name="H2O_Horiba",
        eps_inf=1.0,
        poles=[((-0.0 - 1j * 1.7289263558195928e16), (0.0 + 1j * 5938862032240302.0))],
        frequency_range=(362698386340789.0, 1450793545363156.0),
    ),
    reference=[material_refs["Horiba"]],
)

HMDS_Horiba = VariantItem(
    medium=PoleResidue(
        name="HMDS_Horiba",
        eps_inf=1.0,
        poles=[
            (
                (-379816861999031.8 - 1j * 1.8227252520914852e16),
                (0.0 + 1j * 1.0029341899480378e16),
            )
        ],
        frequency_range=(362698386340789.0, 1571693007476752.5),
    ),
    reference=[material_refs["Horiba"]],
)

HfO2_Horiba = VariantItem(
    medium=PoleResidue(
        name="HfO2_Horiba",
        eps_inf=1.0,
        poles=[
            (
                (-2278901171994190.5 - 1j * 1.4098114301144558e16),
                (0.0 + 1j * 1.3743164680834702e16),
            )
        ],
        frequency_range=(362698386340789.0, 1450793545363156.0),
    ),
    reference=[material_refs["Horiba"]],
)

ITO_Horiba = VariantItem(
    medium=PoleResidue(
        name="ITO_Horiba",
        eps_inf=1.0,
        poles=[
            (
                (-483886682186766.56 - 1j * 1.031968022520672e16),
                (0.0 + 1j * 1.292796190658882e16),
            )
        ],
        frequency_range=(362698386340789.0, 1450793545363156.0),
    ),
    reference=[material_refs["Horiba"]],
)

InAs_Palik = VariantItem(
    medium=PoleResidue(
        name="InAs_Palik",
        eps_inf=6.169295480278222,
        poles=(
            (
                (-110738420632975.47 - 4797247857720928j),
                (61433546381780.16 + 1.3356669256010974e16j),
            ),
            (
                (-89906741691385.8 - 2141190071662963j),
                (25362746938200.98 - 13367622759633.719j),
            ),
            (
                (-716541564870285.5 - 2211195587846909.2j),
                (164186583366674.1 + 57657881084640.46j),
            ),
        ),
        frequency_range=(214137470000000.0, 545077196363636.3),
    ),
    reference=[
        material_refs["Palik"],
    ],
)

InP_Pettit1965 = VariantItem(
    medium=PoleResidue(
        name="InP_Pettit1965",
        eps_inf=7.255000000015208,
        poles=[
            ((0.0 + 1j * 3007586733129570.0), (-0.0 - 1j * 3482785436964042.0)),
            ((0.0 + 1j * 57193003520845.59), (-0.0 - 1j * 79069327367569.03)),
        ],
        frequency_range=(29979245858094.68, 315571009032575.6),
    ),
    reference=[
        material_refs["Pettit1965"],
        material_refs["Pikhtin1978"],
        material_refs["HandbookOptics"],
    ],
    data_url="https://refractiveindex.info/data_csv.php?datafile=database/data-nk/"
    "main/InP/Pettit.yml",
)

InP_Palik_Lossy = VariantItem(
    medium=PoleResidue(
        name="InP_Palik_Lossy",
        eps_inf=1.0,
        poles=[
            (
                (-2436835594727707.5 - 1j * 6012940840882128.0),
                (5209289049886790.0 + 1j * 2.0722903121497028e16),
            ),
            (
                (-181822046571594.84 - 1j * 7643953136098615.0),
                (-397737124208487.1 + 1j * 232408558197787.72),
            ),
            (
                (-332301323575900.5 - 1j * 7123495517140508.0),
                (829584621565063.6 + 1j * 4388096421439121.5),
            ),
            (
                (-99068755067236.3 - 1j * 4782684534836000.0),
                (-38276916712131.36 + 1j * 171113238190472.66),
            ),
            (
                (-369016862332144.3 - 1j * 4823797003938133.0),
                (984268377580870.6 + 1j * 3584419241684899.0),
            ),
        ],
        frequency_range=(365600558536585.4, 1362692990909091.0),
    ),
    reference=[
        material_refs["Palik_Lossy"],
    ],
)

InP_Palik_LowLoss = VariantItem(
    medium=PoleResidue(
        name="InP_Palik_LowLoss",
        eps_inf=1.0,
        poles=[
            (
                (-637021331.3063034 - 1j * 4632017612362640.0),
                (5353676504.962148 + 1j * 1.9464314977679076e16),
            ),
        ],
        frequency_range=(29979245800000.0, 322357481720430.06),
    ),
    reference=[
        material_refs["Palik_LowLoss"],
    ],
)

MgF2_Horiba = VariantItem(
    medium=PoleResidue(
        name="MgF2_Horiba",
        eps_inf=1.0,
        poles=[((-0.0 - 1j * 2.5358092974503356e16), (0.0 + 1j * 1.1398462792039258e16))],
        frequency_range=(193439139381754.16, 918835912063332.1),
    ),
    reference=[material_refs["Horiba"]],
)

MgO_StephensMalitson1952 = VariantItem(
    medium=PoleResidue(
        name="MgO_StephensMalitson1952",
        eps_inf=1.4351800718235839,
        poles=[
            (
                (-0.48094220428010026 - 1j * 79394307783843.27),
                (-0.03402577740242653 + 1j * 237833097466154.6),
            ),
            (
                (-26.994597423819396 - 1j * 1.5689163692407274e16),
                (37.17971519984528 + 1j * 1.1933512715464772e16),
            ),
        ],
        frequency_range=(55517121959434.59, 832756829391519.0),
    ),
    reference=[material_refs["StephensMalitson1952"]],
    data_url="https://refractiveindex.info/data_csv.php?datafile=database/data-nk/"
    "main/MgO/Stephens.yml",
)

MoS2_Li2014 = VariantItem2D(
    medium=Medium2D.from_dispersive_medium(
        PoleResidue(
            name="MoS2_Li2014",
            eps_inf=7,
            poles=(
                ((-359315575683882.94 - 4351037853607888j), 1.3176033127808174e16j),
                ((-47550212723398.46 - 2830800196611070.5j), 423989981007996.2j),
                ((-115583767884472.11 - 3044501424941655.5j), 1228598693488551.8j),
                ((-71809429716556.45 - 4843776341355436j), 3676495982332201.5j),
                ((-357036299948221.06 - 3522742014142554j), 1439441065103469.5j),
            ),
            frequency_range=(359760000000000, 719520000000000),
        ),
        thickness=6.15e-4,
    ),
    reference=[material_refs["Li2014"]],
)

MoSe2_Li2014 = VariantItem2D(
    medium=Medium2D.from_dispersive_medium(
        PoleResidue(
            name="MoSe2_Li2014",
            eps_inf=2.98,
            poles=(
                ((-36761326958106.516 - 2346800992876732.5j), 338220688925072j),
                ((-529696146171994.1 - 3250011358803138j), 2592639640470081.5j),
                ((-83845324190119.6 - 2655257170055444.5j), 600182265785651.4j),
                ((-460941134311120.06 - 3946269084308785.5j), 1.1521315248761458e16j),
                ((-365616548688667.1 - 5272054887123941j), 1.176321407277452e16j),
            ),
            frequency_range=(359760000000000, 719520000000000),
        ),
        thickness=6.46e-4,
    ),
    reference=[material_refs["Li2014"]],
)

Ni_JohnsonChristy1972 = VariantItem(
    medium=PoleResidue(
        name="Ni_JohnsonChristy1972",
        eps_inf=1.0,
        poles=(
            (
                (-727865855817055.1 - 781480451098244.8j),
                (-1555004460279525.5 + 3.602847327083701e16j),
            ),
            (
                (-1298720752173337.8 - 2121335877180779j),
                (1083033754695040 + 1.0925578521887692e16j),
            ),
            (
                (-1819477367096665 - 586975596758.178j),
                (3506055897617337 + 1.561199088683625e18j),
            ),
            (
                (-2001079540362000.8 - 6914798333407941j),
                (999447311644327.9 + 8623994636438280j),
            ),
            (
                (-3956384974540.076 - 12646403210723.701j),
                (8260543758347535 + 3.3147262955373885e18j),
            ),
        ),
        frequency_range=(154771532266391.3, 1594640734042553.2),
    ),
    reference=[material_refs["JohnsonChristy1972"]],
    data_url="https://refractiveindex.info/data_csv.php?datafile=database/data-nk/"
    "main/Ni/Johnson.yml",
)

Ni_RakicLorentzDrude1998 = VariantItem(
    medium=PoleResidue(
        name="Ni_RakicLorentzDrude1998",
        eps_inf=1.0,
        poles=(
            (0j, (3.850528653318057e17 + 0j)),
            ((-72924837503814.11 + 0j), (-3.850528653318057e17 - 0j)),
            ((-10211922369538.5 + 0j), (4280689317169589.5 + 0j)),
            ((-6843203535540992 + 0j), (-4280689317169589.5 - 0j)),
            ((-518328915630820.1 + 0j), (3.988443595266849e16 + 0j)),
            ((-1508373859996013.5 + 0j), (-3.988443595266849e16 - 0j)),
            ((-1654482250867782.5 - 1774676068987181.8j), 1.7470742743872058e16j),
            ((-4779615391395816 - 7920412739409055j), 2.692181349054443e16j),
        ),
        frequency_range=(48359784845438.54, 1208994621135963.5),
    ),
    reference=[material_refs["Rakic1998"]],
    data_url="https://refractiveindex.info/data_csv.php?datafile=database/data-nk/"
    "main/Ni/Rakic-LD.yml",
)

PEI_Horiba = VariantItem(
    medium=PoleResidue(
        name="PEI_Horiba",
        eps_inf=1.0,
        poles=[((-0.0 - 1j * 1.8231209375953524e16), (0.0 + 1j * 9936009109894670.0))],
        frequency_range=(181349193170394.5, 1148544890079165.2),
    ),
    reference=[material_refs["Horiba"]],
)

PEN_Horiba = VariantItem(
    medium=PoleResidue(
        name="PEN_Horiba",
        eps_inf=1.0,
        poles=[((-0.0 - 1j * 6981033923542204.0), (0.0 + 1j * 5117097865956436.0))],
        frequency_range=(362698386340789.0, 773756557527016.6),
    ),
    reference=[material_refs["Horiba"]],
)

PET_Horiba = VariantItem(
    medium=PoleResidue(
        name="PET_Horiba",
        eps_inf=1.0,
        poles=[((-0.0 - 1j * 1.063487213597289e16), (0.0 + 1j * 1.169835934957018e16))],
        frequency_range=(362698386340789.0, 773756557527016.6),
    ),
    reference=[material_refs["Horiba"]],
)

PMMA_Horiba = VariantItem(
    medium=PoleResidue(
        name="PMMA_Horiba",
        eps_inf=1.0,
        poles=[((-0.0 - 1j * 1.7360669128251744e16), (0.0 + 1j * 1.015599144002727e16))],
        frequency_range=(181349193170394.5, 1100185105233726.6),
    ),
    reference=[material_refs["Horiba"]],
)

PMMA_Sultanova2009 = VariantItem(
    medium=PoleResidue(
        name="PMMA_Sultanova2009",
        eps_inf=1,
        poles=[((0.0 + 1j * 1.7709719337156064e16), (-0.0 - 1j * 1.0465558642292376e16))],
        frequency_range=(284973819943865.75, 686338046201801.2),
    ),
    reference=[material_refs["Sultanova2009"]],
    data_url="https://refractiveindex.info/data_csv.php?datafile"
    "=data/organic/(C5H8O2)n%20-%20poly(methyl%20methacrylate)/Sultanova.yml",
)

PTFE_Horiba = VariantItem(
    medium=PoleResidue(
        name="PTFE_Horiba",
        eps_inf=1.0,
        poles=[((-0.0 - 1j * 2.5039046810424176e16), (0.0 + 1j * 8763666383648461.0))],
        frequency_range=(362698386340789.0, 1571693007476752.5),
    ),
    reference=[material_refs["Horiba"]],
)

PVC_Horiba = VariantItem(
    medium=PoleResidue(
        name="PVC_Horiba",
        eps_inf=1.0,
        poles=[((-0.0 - 1j * 1.8551774807480708e16), (0.0 + 1j * 1.209575717447742e16))],
        frequency_range=(362698386340789.0, 1148544890079165.2),
    ),
    reference=[material_refs["Horiba"]],
)

Pd_JohnsonChristy1972 = VariantItem(
    medium=PoleResidue(
        name="Pd_JohnsonChristy1972",
        eps_inf=1.0,
        poles=[
            (
                (-27947601188212.62 - 1j * 88012749128378.45),
                (-116820857784644.19 + 1j * 4.431305747926611e17),
            ),
            (
                (-42421241831450.59 + 1j * 0.0),
                (2.0926917440899536e16 - 1j * 2.322604734166214e17),
            ),
            (
                (-1156114791888924.0 - 1j * 459830394883492.75),
                (-2205692318269041.5 + 1j * 5.882192811019071e16),
            ),
            (
                (-16850504828430.291 - 1j * 19945795950186.92),
                (-2244562993366961.8 + 1j * 2.2399893428156035e17),
            ),
            (
                (-1.0165311890218712e16 - 1j * 6195195244753680.0),
                (-8682197716799510.0 - 1j * 2496615613677907.5),
            ),
        ],
        frequency_range=(154751311505403.34, 1595872899899471.8),
    ),
    reference=[material_refs["JohnsonChristy1972"]],
    data_url="https://refractiveindex.info/data_csv.php?datafile=database/data-nk/"
    "main/Pd/Johnson.yml",
)

Pd_RakicLorentzDrude1998 = VariantItem(
    medium=PoleResidue(
        name="Pd_RakicLorentzDrude1998",
        eps_inf=1.0,
        poles=(
            (0j, (2.96047037671187e18 + 0j)),
            ((-12154139583969.018 + 0j), (-2.96047037671187e18 - 0j)),
            ((-58916603694997.75 + 0j), (1.6215501354199708e16 + 0j)),
            ((-4422922367893578 + 0j), (-1.6215501354199708e16 - 0j)),
            ((-421596716818925.3 - 633727137461217.1j), 2.0818721955845844e16j),
            ((-1067065603800966.5 + 0j), (1.423647063905693e16 + 0j)),
            ((-5953469273389138 + 0j), (-1.423647063905693e16 - 0j)),
            ((-2458174730857734 - 8327373750489667j), 5931453695969745j),
        ),
        frequency_range=(24179892422719.27, 1208994621135963.5),
    ),
    reference=[material_refs["Rakic1998"]],
    data_url="https://refractiveindex.info/data_csv.php?datafile=database/data-nk/"
    "main/Pd/Rakic-LD.yml",
)

Polycarbonate_Horiba = VariantItem(
    medium=PoleResidue(
        name="Polycarbonate_Horiba",
        eps_inf=1.0,
        poles=[((-0.0 - 1j * 1.8240324980641504e16), (0.0 + 1j * 1.3716724385442412e16))],
        frequency_range=(362698386340789.0, 967195696908770.8),
    ),
    reference=[material_refs["Horiba"]],
)

Polycarbonate_Sultanova2009 = VariantItem(
    medium=PoleResidue(
        name="Polycarbonate_Sultanova2009",
        eps_inf=1,
        poles=[((0.0 + 1j * 1.290535618305202e16), (-0.0 - 1j * 9151188069402186.0))],
        frequency_range=(284973819943865.75, 686338046201801.2),
    ),
    reference=[material_refs["Sultanova2009"]],
    data_url="https://refractiveindex.info/data_csv.php?datafile="
    "data/organic/(C16H14O3)n%20-%20polycarbonate/Sultanova.yml",
)

Polystyrene_Sultanova2009 = VariantItem(
    medium=PoleResidue(
        name="Polystyrene_Sultanova2009",
        eps_inf=1,
        poles=[((0.0 + 1j * 1.3248080478547494e16), (-0.0 - 1j * 9561802085391654.0))],
        frequency_range=(284973819943865.75, 686338046201801.2),
    ),
    reference=[material_refs["Sultanova2009"]],
    data_url="https://refractiveindex.info/data_csv.php?datafile="
    "data/organic/(C8H8)n%20-%20polystyren/Sultanova.yml",
)

Pt_Werner2009 = VariantItem(
    medium=PoleResidue(
        name="Pt_Werner2009",
        eps_inf=1.0,
        poles=[
            (
                (-7275744144059393.0 - 1j * 9955657528822276.0),
                (1.4662671453199282e16 + 1j * 2.984423192519277e16),
            ),
            (
                (-970134238553685.2 - 1j * 6795364286984015.0),
                (-217022068099918.25 + 1j * 2348231691221299.0),
            ),
            (
                (-588183430601066.2 - 1j * 5374967881680110.0),
                (-486571495902717.4 + 1j * 3447358361444697.0),
            ),
            (
                (-2600194242881.716 + 1j * 0.0),
                (8.307089446211048e17 + 1j * 0.0),
            ),
            (
                (-250977399463139.06 + 1j * 0.0),
                (-8.305157912555209e17 + 1j * 0.0),
            ),
        ],
        frequency_range=(120884055879414.03, 2997924585809468.0),
    ),
    reference=[material_refs["Werner2009"]],
    data_url="https://refractiveindex.info/data_csv.php?datafile=database/data-nk/"
    "main/Pt/Werner.yml",
)

Pt_RakicLorentzDrude1998 = VariantItem(
    medium=PoleResidue(
        name="Pt_RakicLorentzDrude1998",
        eps_inf=1.0,
        poles=(
            (0j, (2.9080086759055955e17 + 0j)),
            ((-121541395839690.19 + 0j), (-2.9080086759055955e17 - 0j)),
            ((-392730635306998.9 - 1118058787933578.1j), 1.813194578357386e16j),
            ((-1396206784708441 - 1426846131279793.8j), 4.902120207541369e16j),
            ((-2786336499624897.5 - 3874079860313212j), 1.4986300662355044e16j),
            ((-6469800427291507 - 1.2473655652689588e16j), 3.042842289267071e16j),
        ),
        frequency_range=(24179892422719.273, 1208994621135963.5),
    ),
    reference=[material_refs["Rakic1998"]],
    data_url="https://refractiveindex.info/data_csv.php?datafile=database/data-nk/"
    "main/Pt/Rakic-LD.yml",
)

Sapphire_Horiba = VariantItem(
    medium=PoleResidue(
        name="Sapphire_Horiba",
        eps_inf=1.0,
        poles=[((-0.0 - 1j * 2.0143967092980652e16), (0.0 + 1j * 2.105044561216478e16))],
        frequency_range=(362698386340789.0, 1329894083249559.8),
    ),
    reference=[material_refs["Horiba"]],
)

Si3N4_Horiba = VariantItem(
    medium=PoleResidue(
        name="Si3N4_Horiba",
        eps_inf=1.0,
        poles=[
            (
                (-1357465464784539.5 - 1j * 4646140872332419.0),
                (0.0 + 1j * 1.103606337254506e16),
            )
        ],
        frequency_range=(362698386340789.0, 1329894083249559.8),
    ),
    reference=[material_refs["Horiba"]],
)

Si3N4_Luke2015 = VariantItem(
    medium=PoleResidue(
        name="Si3N4_Luke2015",
        eps_inf=1,
        poles=[
            ((0.0 + 1j * 1.391786035350109e16), (-0.0 - 1j * 2.1050067891652724e16)),
            ((0.0 + 1j * 1519267431623.5857), (-0.0 - 1j * 3.0623873619236616e16)),
        ],
        frequency_range=(54468106573573.19, 967072447035312.2),
    ),
    reference=[material_refs["Luke2015"]],
    data_url="https://refractiveindex.info/data_csv.php?datafile=database/data-nk/"
    "main/Si3N4/Luke.yml",
)

Si3N4_Luke2015_PMLStable = VariantItem(
    medium=PoleResidue(
        name="Si3N4_Luke2015_PMLStable",
        eps_inf=3.031225983820944,
        poles=(
            (-7534484687295489j, 3530332266482328j),
            (-4550924050946271j, 7233481618.869821j),
        ),
        frequency_range=(152024573088740.38, 724311326723836.8),
    ),
    reference=[material_refs["Luke2015"]],
    data_url="https://refractiveindex.info/data_csv.php?datafile=database/data-nk/"
    "main/Si3N4/Luke.yml",
)

Si3N4_Philipp1973 = VariantItem(
    medium=PoleResidue(
        name="Si3N4_Philipp1973",
        eps_inf=1,
        poles=[((0.0 + 1j * 1.348644355236665e16), (-0.0 - 1j * 1.9514209498096924e16))],
        frequency_range=(241768111758828.06, 1448272746767859.0),
    ),
    reference=[material_refs["Philipp1973"], material_refs["Baak1982"]],
    data_url="https://refractiveindex.info/data_csv.php?datafile=database/data-nk/"
    "main/Si3N4/Philipp.yml",
)

SiC_Horiba = VariantItem(
    medium=PoleResidue(
        name="SiC_Horiba",
        eps_inf=3.0,
        poles=[((-0.0 - 1j * 1.2154139583969018e16), (0.0 + 1j * 2.3092865209541132e16))],
        frequency_range=(145079354536315.6, 967195696908770.8),
    ),
    reference=[material_refs["Horiba"]],
)

SiN_Horiba = VariantItem(
    medium=PoleResidue(
        name="SiN_Horiba",
        eps_inf=2.32,
        poles=[
            (
                (-302334222151229.3 - 1j * 9863009385232968.0),
                (0.0 + 1j * 6244215164693547.0),
            )
        ],
        frequency_range=(145079354536315.6, 1450793545363156.0),
    ),
    reference=[material_refs["Horiba"]],
)

SiO2_Horiba = VariantItem(
    medium=PoleResidue(
        name="SiO2_Horiba",
        eps_inf=1.0,
        poles=[
            (
                (-75963372399806.36 - 1j * 1.823105111824081e16),
                (0.0 + 1j * 1.0209565875622414e16),
            )
        ],
        frequency_range=(169259246959034.88, 1208994621135963.5),
    ),
    reference=[material_refs["Horiba"]],
)

SiO2_Palik_NoLoss = VariantItem(
    medium=PoleResidue(
        name="SiO2_Palik_NoLoss",
        eps_inf=1.0,
        poles=[
            (
                (0.0 + 1j * 184980080369957.72),
                (0.0 - 1j * 88758428582369.77),
            ),
            (
                (0.0 + 1j * 2.1152615798545252e16),
                (0.0 - 1j * 1.167984925219996e16),
            ),
        ],
        frequency_range=(59958491600000.0, 545077196363636.4),
    ),
    reference=[material_refs["Palik_NoLoss"]],
)

SiO2_Palik_LowLoss = VariantItem(
    medium=PoleResidue(
        name="SiO2_Palik_LowLoss",
        eps_inf=1.5385442336875639,
        poles=[
            (
                (-11504139.374277674 - 1j * 1.595196740783775e16),
                (7507685.43042605 + 1j * 4535416182817100.0),
            ),
            (
                (-249390.3565044153 - 1j * 172280738540723.53),
                (46272.506981344035 + 1j * 99704543223121.88),
            ),
        ],
        frequency_range=(59958491600000.0, 1998616386666666.8),
    ),
    reference=[material_refs["Palik_LowLoss"]],
)

SiO2_Palik_Lossy = VariantItem(
    medium=PoleResidue(
        name="SiO2_Palik_Lossy",
        eps_inf=2.1560362571240765,
        poles=[
            (
                (-3781744691507.2856 - 1j * 207719670863343.84),
                (-18676276825273.156 - 1j * 6355596169134.299),
            ),
            (
                (-9306968330309.3 - 1j * 199739685682949.9),
                (26685644798963.88 + 1j * 81265966041216.78),
            ),
            (
                (-11649519584911.078 - 1j * 161489841654821.16),
                (-13040029201085.318 + 1j * 2679209910871.1226),
            ),
            (
                (-3052239610863.719 - 1j * 88355407251640.77),
                (-24299959225698.41 + 1j * 3850586684365.262),
            ),
            (
                (-7182184304431.551 - 1j * 84819227587180.16),
                (29330620453153.605 + 1j * 39789511603200.61),
            ),
        ],
        frequency_range=(1199169832000.0, 74948114500000.0),
    ),
    reference=[material_refs["Palik_Lossy"]],
)

SiON_Horiba = VariantItem(
    medium=PoleResidue(
        name="SiON_Horiba",
        eps_inf=1.0,
        poles=[((-0.0 - 1j * 1.651139862482191e16), (0.0 + 1j * 1.1079148477255502e16))],
        frequency_range=(181349193170394.5, 725396772681578.0),
    ),
    reference=[material_refs["Horiba"]],
)

Ta2O5_Horiba = VariantItem(
    medium=PoleResidue(
        name="Ta2O5_Horiba",
        eps_inf=1.0,
        poles=[
            (
                (-618341851334423.8 - 1j * 1.205777404193952e16),
                (0.0 + 1j * 1.8938176054079756e16),
            )
        ],
        frequency_range=(181349193170394.5, 967195696908770.8),
    ),
    reference=[material_refs["Horiba"]],
)

Ti_Werner2009 = VariantItem(
    medium=PoleResidue(
        name="Ti_Werner2009",
        eps_inf=1.1169666751532588,
        poles=[
            (
                (-7499149193091210.0 - 1j * 1.2408737021591132e16),
                (2094198081127931.8 + 1j * 1.623942919672785e16),
            ),
            (
                (-2354831105417196.0 - 1j * 8085421034651400.0),
                (-140823625142081.7 + 1j * 5567280255011079.0),
            ),
            (
                (-920843403539743.6 - 1j * 5513453042989389.0),
                (-28870755188206.27 + 1j * 4663773279612380.0),
            ),
            (
                (-2211596411771.0137 + 1j * 0.0),
                (3.6436335858752307e18 + 1j * 0.0),
            ),
            (
                (-38985398241158.02 + 1j * 0.0),
                (-3.643594667795836e18 + 1j * 0.0),
            ),
            (
                (-379374981045792.8 - 1j * 1471699416157019.0),
                (-9274224477696.99 + 1j * 6138254498624140.0),
            ),
        ],
        frequency_range=(120884055879414.03, 2997924585809468.0),
    ),
    reference=[material_refs["Werner2009"]],
    data_url="https://refractiveindex.info/data_csv.php?datafile=database/data-nk/"
    "main/Ti/Werner.yml",
)

Ti_RakicLorentzDrude1998 = VariantItem(
    medium=PoleResidue(
        name="Ti_RakicLorentzDrude1998",
        eps_inf=1.0,
        poles=(
            (0j, (7.286301814080211e16 + 0j)),
            ((-124579930735682.44 + 0j), (-7.286301814080211e16 - 0j)),
            ((-465726048089722.25 + 0j), (2.1824836537305828e16 + 0j)),
            ((-2992126663549463 + 0j), (-2.1824836537305828e16 - 0j)),
            ((-1912757717027124 - 1360524146154421j), 1.7716577274303776e16j),
            ((-1263270883008779.8 - 3596426881658456.5j), 3189068866500566j),
            ((-1338474621684588.2 - 2.9489006173628724e16j), 2079856587113.8086j),
        ),
        frequency_range=(9670724451612.902, 1208986804855426.0),
    ),
    reference=[material_refs["Rakic1998"]],
    data_url="https://refractiveindex.info/data_csv.php?datafile=database/data-nk/"
    "main/Ti/Rakic-LD.yml",
)

TiOx_Horiba = VariantItem(
    medium=PoleResidue(
        name="TiOx_Horiba",
        eps_inf=0.29,
        poles=[((-0.0 - 1j * 9875238411974826.0), (0.0 + 1j * 1.7429795797135566e16))],
        frequency_range=(145079354536315.6, 725396772681578.0),
    ),
    reference=[material_refs["Horiba"]],
)

TiOx_HoribaStable = VariantItem(
    medium=PoleResidue(
        name="TiOx_HoribaStable",
        eps_inf=1.0,
        poles=(
            (-9092895987017908j, 1.2878308348235048e16j),
            (-2393718508037.5645j, 2861174795691055.5j),
        ),
        frequency_range=(145079354536315.6, 725396772681578.0),
    ),
    reference=[material_refs["Horiba"]],
)

W_Werner2009 = VariantItem(
    medium=PoleResidue(
        name="W_Werner2009",
        eps_inf=1.0,
        poles=[
            (
                (-252828226350812.1 - 1j * 5784339105708298.0),
                (1948434702479989.5 + 1j * 2132849526706848.8),
            ),
            (
                (-609339941453361.0 - 1j * 3915807795417961.0),
                (5854104539645628.0 + 1j * 9628118738288506.0),
            ),
            (
                (-351192772413652.7 - 1j * 1494382744693159.8),
                (-1716717002731465.8 + 1j * 3.5658946710009096e16),
            ),
            (
                (-5781257140720.449 - 1j * 134119053600439.05),
                (1.1436600467759824e16 + 1j * 1.3607215135185057e18),
            ),
        ],
        frequency_range=(120884055879414.03, 2997924585809468.0),
    ),
    reference=[material_refs["Werner2009"]],
    data_url="https://refractiveindex.info/data_csv.php?datafile=database/data-nk/"
    "main/W/Werner.yml",
)

W_RakicLorentzDrude1998 = VariantItem(
    medium=PoleResidue(
        name="W_RakicLorentzDrude1998",
        eps_inf=1.0,
        poles=(
            (0j, (4.2732115514080845e17 + 0j)),
            ((-97233116671752.14 + 0j), (-4.2732115514080845e17 - 0j)),
            ((-402605873718973.75 - 1471252666401400j), 7403002173803196j),
            ((-973090800441519.4 - 2745063931489722.5j), 1.2197111799530032e16j),
            ((-2531099568361548 - 4814146946972908j), 2.9579221430831016e16j),
            ((-4433222413252700 - 1.0493429699239636e16j), 4.978330061510858e16j),
        ),
        frequency_range=(24176811129032.258, 1208986804855426.0),
    ),
    reference=[material_refs["Rakic1998"]],
    data_url="https://refractiveindex.info/data_csv.php?datafile=database/data-nk/"
    "main/W/Rakic-LD.yml",
)

WS2_Li2014 = VariantItem2D(
    medium=Medium2D.from_dispersive_medium(
        PoleResidue(
            name="WS2_Li2014",
            eps_inf=6.18,
            poles=(
                ((-24007743182653.15 - 3052251370817458j), 716432281919880.8j),
                ((-137029680488199.55 - 3645019841622440.5j), 1010556928646303.5j),
                ((-209636856712237.66 - 4307630639419263j), 3158371314580892.5j),
                ((-466855949030110.3 - 4891967555229964j), 1.1703841436358186e16j),
            ),
            frequency_range=(359760000000000, 719520000000000),
        ),
        thickness=6.18e-4,
    ),
    reference=[material_refs["Li2014"]],
)

WSe2_Li2014 = VariantItem2D(
    medium=Medium2D.from_dispersive_medium(
        PoleResidue(
            name="WSe2_Li2014",
            eps_inf=6.29,
            poles=(
                ((-32911988143375.11 - 2509059529797599.5j), 280960681034011.66j),
                ((-138781520516435.52 - 3149502378897181.5j), 690812354204714j),
                ((-28255484326386.598 - 5055392293836629j), 4415551968968067j),
                ((-258471752123009.2 - 3675437972450793.5j), 2703088180177408.5j),
                ((-354954812602843.94 - 4404690392224204.5j), 7012794593077168j),
            ),
            frequency_range=(359760000000000, 719520000000000),
        ),
        thickness=6.49e-4,
    ),
    reference=[material_refs["Li2014"]],
)

Y2O3_Horiba = VariantItem(
    medium=PoleResidue(
        name="Y2O3_Horiba",
        eps_inf=1.0,
        poles=[((-0.0 - 1j * 1.3814698904628784e16), (0.0 + 1j * 1.1846104310719182e16))],
        frequency_range=(374788332552148.7, 967195696908770.8),
    ),
    reference=[material_refs["Horiba"]],
)

Y2O3_Nigara1968 = VariantItem(
    medium=PoleResidue(
        name="Y2O3_Nigara1968",
        eps_inf=1,
        poles=[
            ((0.0 + 1j * 1.3580761146063806e16), (-0.0 - 1j * 1.7505601117276244e16)),
            ((0.0 + 1j * 82126420080181.8), (-0.0 - 1j * 161583731507757.7)),
        ],
        frequency_range=(31228381102181.96, 1199169834323787.2),
    ),
    reference=[material_refs["Nigara1968"]],
    data_url="https://refractiveindex.info/data_csv.php?datafile=database/data-nk/"
    "main/Y2O3/Nigara.yml",
)

YAG_Zelmon1998 = VariantItem(
    medium=PoleResidue(
        name="YAG_Zelmon1998",
        eps_inf=1,
        poles=[
            ((0.0 + 1j * 1.7303796419562446e16), (-0.0 - 1j * 1.974363171472075e16)),
            ((0.0 + 1j * 112024123195387.16), (-0.0 - 1j * 183520159101147.16)),
        ],
        frequency_range=(59958491716189.36, 749481146452367.0),
    ),
    reference=[material_refs["Zelmon1998"]],
    data_url="https://refractiveindex.info/data_csv.php?datafile=database/data-nk/"
    "main/Y3Al5O12/Zelmon.yml",
)

ZrO2_Horiba = VariantItem(
    medium=PoleResidue(
        name="ZrO2_Horiba",
        eps_inf=1.0,
        poles=[
            (
                (-97233116671752.14 - 1j * 1.446765717253359e16),
                (0.0 + 1j * 2.0465425413547396e16),
            )
        ],
        frequency_range=(362698386340789.0, 725396772681578.0),
    ),
    reference=[material_refs["Horiba"]],
)

aSi_Horiba = VariantItem(
    medium=PoleResidue(
        name="aSi_Horiba",
        eps_inf=3.109,
        poles=[
            (
                (-1458496750076282.0 - 1j * 5789844327200831.0),
                (0.0 + 1j * 4.485863370051096e16),
            )
        ],
        frequency_range=(362698386340789.0, 1450793545363156.0),
    ),
    reference=[material_refs["Horiba"]],
)

cSi_SalzbergVilla1957 = VariantItem(
    medium=PoleResidue(
        name="cSi_SalzbergVilla1957",
        eps_inf=1.0,
        poles=[((0.0 + 1j * 6206417594288582.0), (-0.0 - 1j * 3.311074436985222e16))],
        frequency_range=(27253859870995.164, 220435631309519.7),
    ),
    reference=[material_refs["SalzbergVilla1957"], material_refs["Tatian1984"]],
    data_url="https://refractiveindex.info/data_csv.php?datafile=database/data-nk/"
    "main/Si/Salzberg.yml",
)

cSi_Li1993_293K = VariantItem(
    medium=PoleResidue(
        name="cSi_Li1993_293K",
        eps_inf=1.0,
        poles=[((0.0 + 1j * 6241549589084091.0), (0.0 - 1j * 3.3254308736142404e16))],
        frequency_range=(21413747041496.2, 249827048817455.7),
    ),
    reference=[material_refs["Li1993_293K"]],
    data_url="https://refractiveindex.info/data_csv.php?datafile=database/data-nk/"
    "main/Si/Li-293K.yml",
)

cSi_Green2008 = VariantItem(
    medium=PoleResidue(
        name="cSi_Green2008",
        eps_inf=1.0,
        poles=[
            (
                (-1222609703462548.8 - 1j * 8050379549196754.0),
                (-459622937683711.4 + 1j * 9267605970169190.0),
            ),
            (
                (-175523251487055.25 - 1j * 5102103225457989.0),
                (2999238051888977.5 + 1j * 3308731934026053.5),
            ),
            (
                (-639256909304283.8 - 1j * 5540083093730861.0),
                (3951666456017081.5 + 1j * 8328629776149453.0),
            ),
            (
                (-41762311617006.414 - 1j * 6390358961251164.0),
                (235737176572234.88 + 1j * 7336777258751.023),
            ),
            (
                (-408427946633920.56 - 1j * 6473818314696734.0),
                (-3701782820069415.5 + 1j * 1.3120427121052996e16),
            ),
        ],
        frequency_range=(206753419710997.8, 1199169834323787.2),
    ),
    reference=[material_refs["Green2008"]],
    data_url="https://refractiveindex.info/data_csv.php?datafile=database/data-nk/"
    "main/Si/Green-2008.yml",
)

cSi_Green2008Lossless = VariantItem(
    medium=PoleResidue(
        name="cSi_Green2008Lossless",
        eps_inf=8.735527704181576,
        poles=[
            (
                (-3618638294867195j),
                (5372233772327493j),
            ),
        ],
        frequency_range=(206753419710997.8, 249827048333333.34),
    ),
    reference=[material_refs["Green2008"]],
    data_url="https://refractiveindex.info/data_csv.php?datafile=database/data-nk/"
    "main/Si/Green-2008.yml",
)

cSi_PalikLossy = VariantItem(
    medium=PoleResidue(
        name="cSi_PalikLossy",
        eps_inf=1.0,
        poles=[
            (
                (-1412334036440776.5 - 1j * 7736965309820769.0),
                (2074279678884574.5 + 1j * 1.24660337157771e16),
            ),
            (
                (-450397701359903.6 - 1j * 6618528100117753.0),
                (-6140228015093651.0 + 1j * 1.0715818241196356e16),
            ),
            (
                (-151644372105463.22 - 1j * 6524259387210869.0),
                (-1172059456363202.2 + 1j * 843491118862394.5),
            ),
            (
                (-175346131688212.56 - 1j * 5130142673758083.0),
                (2416851869777591.0 + 1j * 3365218119235760.5),
            ),
            (
                (-664592759241954.5 - 1j * 5457951786866534.0),
                (6067672014919645.0 + 1j * 7145799033442451.0),
            ),
        ],
        frequency_range=(214137470000000.0, 2997924580000000.0),
    ),
    reference=[material_refs["Palik_Lossy"]],
)

cSi_PalikNoLoss = VariantItem(
    medium=PoleResidue(
        name="cSi_PalikNoLoss",
        eps_inf=1.0,
        poles=[
            (
                (0.0 - 1j * 6445155000140348.0),
                (0.0 + 1j * 3.4470730496439236e16),
            ),
        ],
        frequency_range=(49965409666666.664, 249827048333333.34),
    ),
    reference=[material_refs["Palik_NoLoss"]],
)

cSi_PalikLowLoss = VariantItem(
    medium=PoleResidue(
        name="cSi_PalikLowLoss",
        eps_inf=1.0,
        poles=[
            (
                (-1.7473849958109988 - 1j * 6409829457220535.0),
                (0.06947645444424029 + 1j * 3.4268436708700284e16),
            ),
        ],
        frequency_range=(1199169832000.0, 249827048333333.34),
    ),
    reference=[material_refs["Palik_LowLoss"]],
)

cSi_MultiPhysics = VariantItem(
    medium=MultiPhysicsMedium(
        optical=cSi_PalikLowLoss.medium,
        charge=SemiconductorMedium(
            permittivity=11.7,
            N_c=ConstantEffectiveDOS(N=2.86e19),
            N_v=ConstantEffectiveDOS(N=3.1e19),
            E_g=ConstantEnergyBandGap(eg=1.11),
            mobility_n=CaugheyThomasMobility(
                mu_min=52.2,
                mu=1471.0,
                ref_N=9.68e16,
                exp_N=0.68,
                exp_1=-0.57,
                exp_2=-2.33,
                exp_3=2.4,
                exp_4=-0.146,
            ),
            mobility_p=CaugheyThomasMobility(
                mu_min=44.9,
                mu=470.5,
                ref_N=2.23e17,
                exp_N=0.719,
                exp_1=-0.57,
                exp_2=-2.33,
                exp_3=2.4,
                exp_4=-0.146,
            ),
            R=[
                ShockleyReedHallRecombination(tau_n=3.3e-6, tau_p=4e-6),
                RadiativeRecombination(r_const=1.6e-14),
                AugerRecombination(c_n=2.8e-31, c_p=9.9e-32),
            ],
            delta_E_g=SlotboomBandGapNarrowing(
                v1=6.92 * 1e-3,
                n2=1.3e17,
                c2=0.5,
                min_N=1e15,
            ),
        ),
    ),
    reference=[material_refs["Palik_LowLoss"]],
)


class MaterialLibrary(dict):
    def __str__(self) -> str:
        return summarize_material_library(self)

    def __rich__(self) -> Table:
        return summarize_material_library_rich(self)

    def _repr_pretty_(self, p: RepresentationPrinter, cycle: bool) -> None:
        return repr_pretty_with_rich(self, p, cycle)


material_library = MaterialLibrary(
    Ag=MaterialItem(
        name="Silver",
        variants={
            "Rakic1998BB": Ag_Rakic1998BB,
            "Rakic1998BB_IR": Ag_Rakic1998BB_IR,
            "JohnsonChristy1972": Ag_JohnsonChristy1972,
            "RakicLorentzDrude1998": Ag_RakicLorentzDrude1998,
            "Yang2015Drude": Ag_Yang2015Drude,
        },
        default="Rakic1998BB",
    ),
    Al=MaterialItem(
        name="Aluminum",
        variants={
            "Rakic1995": Al_Rakic1995,
            "RakicLorentzDrude1998": Al_RakicLorentzDrude1998,
        },
        default="Rakic1995",
    ),
    Al2O3=MaterialItem(
        name="Alumina",
        variants={
            "Horiba": Al2O3_Horiba,
        },
        default="Horiba",
    ),
    AlAs=MaterialItem(
        name="Aluminum Arsenide",
        variants={
            "Horiba": AlAs_Horiba,
            "FernOnton1971": AlAs_FernOnton1971,
        },
        default="Horiba",
    ),
    AlGaN=MaterialItem(
        name="Aluminum Gallium Nitride",
        variants={
            "Horiba": AlGaN_Horiba,
        },
        default="Horiba",
    ),
    AlN=MaterialItem(
        name="Aluminum Nitride",
        variants={
            "Horiba": AlN_Horiba,
        },
        default="Horiba",
    ),
    AlxOy=MaterialItem(
        name="Aluminum Oxide",
        variants={
            "Horiba": AlxOy_Horiba,
        },
        default="Horiba",
    ),
    Aminoacid=MaterialItem(
        name="Amino Acid",
        variants={
            "Horiba": Aminoacid_Horiba,
        },
        default="Horiba",
    ),
    Au=MaterialItem(
        name="Gold",
        variants={
            "Olmon2012crystal": Au_Olmon2012crystal,
            "Olmon2012stripped": Au_Olmon2012stripped,
            "Olmon2012evaporated": Au_Olmon2012evaporated,
            "Olmon2012evaporated_IR": Au_Olmon2012evaporated_IR,
            "Olmon2012Drude": Au_Olmon2012Drude,
            "JohnsonChristy1972": Au_JohnsonChristy1972,
            "RakicLorentzDrude1998": Au_RakicLorentzDrude1998,
        },
        default="Olmon2012evaporated",
    ),
    BK7=MaterialItem(
        name="N-BK7 Borosilicate Glass",
        variants={
            "Zemax": BK7_Zemax,
        },
        default="Zemax",
    ),
    Be=MaterialItem(
        name="Beryllium",
        variants={
            "Rakic1998BB": Be_Rakic1998BB,
            "Rakic1998BB_IR": Be_Rakic1998BB_IR,
            "RakicLorentzDrude1998": Be_RakicLorentzDrude1998,
        },
        default="Rakic1998BB",
    ),
    CaF2=MaterialItem(
        name="Calcium Fluoride",
        variants={
            "Horiba": CaF2_Horiba,
        },
        default="Horiba",
    ),
    Cellulose=MaterialItem(
        name="Cellulose",
        variants={
            "Sultanova2009": Cellulose_Sultanova2009,
        },
        default="Sultanova2009",
    ),
    Cr=MaterialItem(
        name="Chromium",
        variants={
            "Rakic1998BB": Cr_Rakic1998BB,
            "Rakic1998BB_IR": Cr_Rakic1998BB_IR,
            "RakicLorentzDrude1998": Cr_RakicLorentzDrude1998,
        },
        default="Rakic1998BB",
    ),
    Cu=MaterialItem(
        name="Copper",
        variants={
            "JohnsonChristy1972": Cu_JohnsonChristy1972,
            "RakicLorentzDrude1998": Cu_RakicLorentzDrude1998,
        },
        default="JohnsonChristy1972",
    ),
    FusedSilica=MaterialItem(
        name="Fused Silica",
        variants={
            "ZemaxSellmeier": FusedSilica_Zemax,
            "ZemaxVisiblePMLStable": FusedSilica_Zemax_Visible_PMLStable,
            "ZemaxPMLStable": FusedSilica_Zemax_PMLStable,
        },
        default="ZemaxPMLStable",
    ),
    GaAs=_with_palik_lossless_alias(
        MaterialItem(
            name="Gallium Arsenide",
            variants={
                "Palik_NoLoss": GaAs_Palik_NoLoss,
                "Palik_LowLoss": GaAs_Palik_LowLoss,
                "Palik_Lossy": GaAs_Palik_Lossy,
                "Skauli2003": GaAs_Skauli2003,
            },
            default="Skauli2003",
        )
    ),
    Ge=_with_palik_lossless_alias(
        MaterialItem(
            name="Germanium",
            variants={
                "Palik_NoLoss": Ge_Palik_NoLoss,
                "Palik_LowLoss": Ge_Palik_LowLoss,
                "Palik_Lossy": Ge_Palik_Lossy,
                "Icenogle1976": Ge_Icenogle1976,
                "Nunley": Ge_Nunley,
            },
            default="Icenogle1976",
        )
    ),
    GeOx=MaterialItem(
        name="Germanium Oxide",
        variants={
            "Horiba": GeOx_Horiba,
        },
        default="Horiba",
    ),
    H2O=MaterialItem(
        name="Water",
        variants={
            "Horiba": H2O_Horiba,
        },
        default="Horiba",
    ),
    HMDS=MaterialItem(
        name="Hexamethyldisilazane, or Bis(trimethylsilyl)amine",
        variants={
            "Horiba": HMDS_Horiba,
        },
        default="Horiba",
    ),
    HfO2=MaterialItem(
        name="Hafnium Oxide",
        variants={
            "Horiba": HfO2_Horiba,
        },
        default="Horiba",
    ),
    ITO=MaterialItem(
        name="Indium Tin Oxide",
        variants={
            "Horiba": ITO_Horiba,
        },
        default="Horiba",
    ),
    InAs=MaterialItem(
        name="Indium Arsenide",
        variants={
            "Palik": InAs_Palik,
        },
        default="Palik",
    ),
    InP=_with_palik_lossless_alias(
        MaterialItem(
            name="Indium Phosphide",
            variants={
                "Palik_LowLoss": InP_Palik_LowLoss,
                "Palik_Lossy": InP_Palik_Lossy,
                "Pettit1965": InP_Pettit1965,
            },
            default="Pettit1965",
        )
    ),
    MgF2=MaterialItem(
        name="Magnesium Fluoride",
        variants={
            "Horiba": MgF2_Horiba,
        },
        default="Horiba",
    ),
    MgO=MaterialItem(
        name="Magnesium Oxide",
        variants={
            "StephensMalitson1952": MgO_StephensMalitson1952,
        },
        default="StephensMalitson1952",
    ),
    MoS2=MaterialItem2D(
        name="Molybdenum Disulfide",
        variants={
            "Li2014": MoS2_Li2014,
        },
        default="Li2014",
    ),
    MoSe2=MaterialItem2D(
        name="Molybdenum Diselenide",
        variants={
            "Li2014": MoSe2_Li2014,
        },
        default="Li2014",
    ),
    Ni=MaterialItem(
        name="Nickel",
        variants={
            "JohnsonChristy1972": Ni_JohnsonChristy1972,
            "RakicLorentzDrude1998": Ni_RakicLorentzDrude1998,
        },
        default="JohnsonChristy1972",
    ),
    PEI=MaterialItem(
        name="Polyetherimide",
        variants={
            "Horiba": PEI_Horiba,
        },
        default="Horiba",
    ),
    PEN=MaterialItem(
        name="Polyethylene Naphthalate",
        variants={
            "Horiba": PEN_Horiba,
        },
        default="Horiba",
    ),
    PET=MaterialItem(
        name="Polyethylene Terephthalate",
        variants={
            "Horiba": PET_Horiba,
        },
        default="Horiba",
    ),
    PMMA=MaterialItem(
        name="Poly(methyl Methacrylate)",
        variants={
            "Horiba": PMMA_Horiba,
            "Sultanova2009": PMMA_Sultanova2009,
        },
        default="Sultanova2009",
    ),
    PTFE=MaterialItem(
        name="Polytetrafluoroethylene, or Teflon",
        variants={
            "Horiba": PTFE_Horiba,
        },
        default="Horiba",
    ),
    PVC=MaterialItem(
        name="Polyvinyl Chloride",
        variants={
            "Horiba": PVC_Horiba,
        },
        default="Horiba",
    ),
    Pd=MaterialItem(
        name="Palladium",
        variants={
            "JohnsonChristy1972": Pd_JohnsonChristy1972,
            "RakicLorentzDrude1998": Pd_RakicLorentzDrude1998,
        },
        default="JohnsonChristy1972",
    ),
    Polycarbonate=MaterialItem(
        name="Polycarbonate",
        variants={
            "Horiba": Polycarbonate_Horiba,
            "Sultanova2009": Polycarbonate_Sultanova2009,
        },
        default="Sultanova2009",
    ),
    Polystyrene=MaterialItem(
        name="Polystyrene",
        variants={
            "Sultanova2009": Polystyrene_Sultanova2009,
        },
        default="Sultanova2009",
    ),
    Pt=MaterialItem(
        name="Platinum",
        variants={
            "Werner2009": Pt_Werner2009,
            "RakicLorentzDrude1998": Pt_RakicLorentzDrude1998,
        },
        default="Werner2009",
    ),
    Sapphire=MaterialItem(
        name="Sapphire",
        variants={
            "Horiba": Sapphire_Horiba,
        },
        default="Horiba",
    ),
    Si3N4=MaterialItem(
        name="Silicon Nitride",
        variants={
            "Horiba": Si3N4_Horiba,
            "Luke2015Sellmeier": Si3N4_Luke2015,
            "Luke2015PMLStable": Si3N4_Luke2015_PMLStable,
            "Philipp1973Sellmeier": Si3N4_Philipp1973,
        },
        default="Horiba",
    ),
    SiC=MaterialItem(
        name="Silicon Carbide",
        variants={
            "Horiba": SiC_Horiba,
        },
        default="Horiba",
    ),
    SiN=MaterialItem(
        name="Silicon Mononitride",
        variants={
            "Horiba": SiN_Horiba,
        },
        default="Horiba",
    ),
    SiO2=_with_palik_lossless_alias(
        MaterialItem(
            name="Silicon Dioxide",
            variants={
                "Palik_NoLoss": SiO2_Palik_NoLoss,
                "Palik_LowLoss": SiO2_Palik_LowLoss,
                "Palik_Lossy": SiO2_Palik_Lossy,
                "Horiba": SiO2_Horiba,
            },
            default="Palik_LowLoss",
        )
    ),
    SiON=MaterialItem(
        name="Silicon Oxynitride",
        variants={
            "Horiba": SiON_Horiba,
        },
        default="Horiba",
    ),
    Ta2O5=MaterialItem(
        name="Tantalum Pentoxide",
        variants={
            "Horiba": Ta2O5_Horiba,
        },
        default="Horiba",
    ),
    Ti=MaterialItem(
        name="Titanium",
        variants={
            "Werner2009": Ti_Werner2009,
            "RakicLorentzDrude1998": Ti_RakicLorentzDrude1998,
        },
        default="Werner2009",
    ),
    TiOx=MaterialItem(
        name="Titanium Oxide",
        variants={
            "Horiba": TiOx_Horiba,
            "HorbiaStable": TiOx_HoribaStable,
        },
        default="Horiba",
    ),
    W=MaterialItem(
        name="Tungsten",
        variants={
            "Werner2009": W_Werner2009,
            "RakicLorentzDrude1998": W_RakicLorentzDrude1998,
        },
        default="Werner2009",
    ),
    WS2=MaterialItem2D(
        name="Tungsten Disulfide",
        variants={
            "Li2014": WS2_Li2014,
        },
        default="Li2014",
    ),
    WSe2=MaterialItem2D(
        name="Tungsten Diselenide",
        variants={
            "Li2014": WSe2_Li2014,
        },
        default="Li2014",
    ),
    Y2O3=MaterialItem(
        name="Yttrium Oxide",
        variants={
            "Horiba": Y2O3_Horiba,
            "Nigara1968": Y2O3_Nigara1968,
        },
        default="Horiba",
    ),
    YAG=MaterialItem(
        name="Yttrium Aluminium Garnet",
        variants={
            "Zelmon1998": YAG_Zelmon1998,
        },
        default="Zelmon1998",
    ),
    ZrO2=MaterialItem(
        name="Zirconium Oxide",
        variants={
            "Horiba": ZrO2_Horiba,
        },
        default="Horiba",
    ),
    aSi=MaterialItem(
        name="Silicon (Amorphous)",
        variants={
            "Horiba": aSi_Horiba,
        },
        default="Horiba",
    ),
    cSi=_with_palik_lossless_alias(
        MaterialItem(
            name="Silicon (Crystalline)",
            variants={
                "Palik_NoLoss": cSi_PalikNoLoss,
                "Palik_LowLoss": cSi_PalikLowLoss,
                "Palik_Lossy": cSi_PalikLossy,
                "SalzbergVilla1957": cSi_SalzbergVilla1957,
                "Li1993_293K": cSi_Li1993_293K,
                "Green2008": cSi_Green2008,
                "Green2008_Lossless": cSi_Green2008Lossless,
                "Si_MultiPhysics": cSi_MultiPhysics,
            },
            default="Palik_LowLoss",
        )
    ),
    LiNbO3=MaterialItemUniaxial(
        name="Lithium niobate",
        variants={"Zelmon1997": LiNbO3_Zelmon1997},
        default="Zelmon1997",
    ),
    graphene=Graphene,
)
