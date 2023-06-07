"""Holds dispersive models for several commonly used optical materials."""  # pylint: disable=too-many-lines
import json
from typing import Dict, List
import pydantic as pd

from ..components.medium import PoleResidue, Medium2D
from ..components.base import Tidy3dBaseModel
from ..exceptions import SetupError
from .material_reference import material_refs, ReferenceData
from .parametric_materials import Graphene


def export_matlib_to_file(fname: str = "matlib.json") -> None:
    """Write the material library to a .json file."""

    mat_lib_dict = {
        f'{mat.name} ("{mat_name}")': {
            var_name: json.loads(var.medium._json_string)  # pylint: disable=protected-access
            for var_name, var in mat.variants.items()
        }
        for mat_name, mat in material_library.items()
        if not isinstance(mat, type)
    }

    with open(fname, "w") as f:
        json.dump(mat_lib_dict, f)


class VariantItem(Tidy3dBaseModel):
    """Reference, data_source, and material model for a variant of a material."""

    medium: PoleResidue = pd.Field(
        ...,
        title="Material dispersion model",
        description="A dispersive medium described by the pole-residue pair model.",
    )

    reference: List[ReferenceData] = pd.Field(
        None,
        title="Reference information",
        description="A list of references related to this variant model.",
    )

    data_url: str = pd.Field(
        None,
        title="Dispersion data URL",
        description="The URL to access the dispersion data upon which the material "
        "model is fitted.",
    )


class MaterialItem(Tidy3dBaseModel):
    """A material that includes several variants."""

    name: str = pd.Field(..., title="Name", description="Unique name for the medium.")
    variants: Dict[str, VariantItem] = pd.Field(
        ...,
        title="Dictionary of available variants for this material",
        description="A dictionary of available variants for this material "
        "that maps from a key to the variant model.",
    )
    default: str = pd.Field(
        ..., title="default variant", description="The default type of variant."
    )

    @pd.validator("default", always=True)
    def _default_in_variants(cls, val, values):
        """Make sure the default variant is already included in the ``variants``."""
        if not val in values["variants"]:
            raise SetupError(
                f"The data of the default variant '{val}' is not supplied; "
                "please include it in the 'variants'."
            )
        return val

    def __getitem__(self, variant_name):
        """Helper function to easily access the medium of a variant"""
        return self.variants[variant_name].medium

    @property
    def medium(self):
        """The default medium."""
        return self.variants[self.default].medium


class VariantItem2D(VariantItem):
    """Reference, data_source, and material model for a variant of a 2D material."""

    medium: Medium2D = pd.Field(
        ...,
        title="Material dispersion model",
        description="A dispersive 2D medium described by a surface conductivity model, "
        "which is handled as an anisotropic medium with pole-residue pair models "
        "defined for the in-plane directions of the 2D geometry.",
    )


class MaterialItem2D(MaterialItem):
    """A 2D material that includes several variants."""

    variants: Dict[str, VariantItem2D] = pd.Field(
        ...,
        title="Dictionary of available variants for this material",
        description="A dictionary of available variants for this material "
        "that maps from a key to the variant model.",
    )


Ag_Rakic1998BB = VariantItem(
    medium=PoleResidue(
        eps_inf=1.0,
        poles=[
            (
                (-275580863813647.1 + 1j * 312504541922578.7),
                (410592688830514.8 - 1j * 1.3173437570517746e16),
            ),
            (
                (-1148310840598705.2 + 1j * 8055992835194972.0),
                (227736607453638.5 - 1j * 1042414461766764.9),
            ),
            (
                (-381116695232772.56 + 1j * 6594145937912653.0),
                (161555291564323.06 - 1j * 1397161265004318.2),
            ),
            (
                (-1.2755935758322332e16 + 1j * 4213421975115564.5),
                (1.718968422861484e16 + 1j * 2.293341935281984e16),
            ),
            (
                (-1037538194.0633082 - 1j * 71105682833114.89),
                (117311511.37080565 + 1j * 6.61015554492372e17),
            ),
            (
                (-76642436669493.88 + 1j * 123745349008080.44),
                (129838572187083.62 - 1j * 2.1821880909947117e17),
            ),
        ],
        frequency_range=(24179892422719.273, 1208994621135963.5),
    ),
    reference=[material_refs["Rakic1998"]],
    data_url="https://refractiveindex.info/data_csv.php?datafile=database/data-nk/"
    "main/Ag/Rakic-BB.yml",
)

Ag_RakicLorentzDrude1998 = VariantItem(
    medium=PoleResidue(
        eps_inf=1.0,
        poles=(
            ((-800062663506.125 + 0j), (1.1099533667426209e18 + 0j)),
            ((-72124774840307.98 + 0j), (-1.1099533667426209e18 - 0j)),
            ((-272940800787927.5 + 0j), (1136578330456760.5 + 0j)),
            ((-5630932502125024 + 0j), (-1136578330456760.5 - 0j)),
            ((-343354443247124.8 - 6799173351259867j), 1708652013864486.5j),
            ((-49376192059874.14 - 1.2435106032980426e16j), 82876469878486.64j),
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
        eps_inf=1.0000740219977509,
        poles=(
            ((-1306213662214179.8 - 6200791340906446j), (5316579866298263 + 770314552771784.5j)),
            (
                (-825550023625.1349 - 43574416058741.68j),
                (7414298483024743 + 2.1001409321547041e18j),
            ),
            (
                (-1414117800340546 - 841892406600516.1j),
                (-3015811271404633.5 - 1.627264404485923e16j),
            ),
        ),
        frequency_range=(154771532566312.25, 1595489401708072.2),
    ),
    reference=[material_refs["JohnsonChristy1972"]],
    data_url="https://refractiveindex.info/data_csv.php?datafile=database/data-nk/"
    "main/Ag/Johnson.yml",
)

Ag_Yang2015Drude = VariantItem(
    medium=PoleResidue(
        eps_inf=5.0,
        poles=(
            (
                (0.0j),
                (1.5540587685959158e18 + 0j),
            ),
            (
                (-58823530000000 + 0j),
                (-1.5540587685959158e18 - 0j),
            ),
        ),
        frequency_range=(154771532566312.25, 1595489401708072.2),
    ),
    reference=[material_refs["Yang2015"]],
    data_url="https://refractiveindex.info/data_csv.php?datafile=database/data-nk/"
    "main/Ag/Yang.yml",
)

Al_Rakic1995 = VariantItem(
    medium=PoleResidue(
        eps_inf=1.0,
        poles=[
            (
                (-38634980988505.31 - 1j * 48273958812026.45),
                (4035140886647080.0 + 1j * 2.835977690098632e18),
            ),
            ((-1373449221156.457 + 1j * 0.0), (7.630343339215653e16 + 1j * 2.252091523762478e17)),
            (
                (-1.0762187388103686e16 - 1j * 799978314126058.1),
                (-1.5289438747838848e16 + 1j * 4.746731963865045e16),
            ),
            (
                (-179338332256147.1 - 1j * 243607346238054.5),
                (-4.625363670034073e16 + 1j * 7.703073947098675e16),
            ),
            (
                (-1.0180997365823526e16 - 1j * 5542555481403632.0),
                (-1.6978040336362288e16 - 1j * 1.4140848316870884e16),
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
        eps_inf=1.0,
        poles=(
            ((-16956600687280.727 + 0j), (3.6126085572070707e18 + 0j)),
            ((-54448969368537.26 + 0j), (-3.6126085572070707e18 - 0j)),
            ((-194545209645174.6 + 0j), (5.0321060071503546e17 + 0j)),
            ((-311370850537535.75 + 0j), (-5.0321060071503546e17 - 0j)),
            ((-237005721887395.88 - 2333745139453868j), 5548539400655874j),
            ((-1026265161121384.1 - 2547917843202809j), 1.6872706975652858e16j),
            ((-2569081254561451.5 - 4608729293067524j), 1685784870483934.5j),
        ),
        frequency_range=(1208994621135.9636, 4835978484543854.0),
    ),
    reference=[material_refs["Rakic1998"]],
    data_url="https://refractiveindex.info/data_csv.php?datafile=database/data-nk/"
    "main/Al/Rakic-LD.yml",
)

Al2O3_Horiba = VariantItem(
    medium=PoleResidue(
        eps_inf=1.0,
        poles=[((-0.0 - 1j * 1.856240967961668e16), (0.0 + 1j * 1.4107431356508676e16))],
        frequency_range=(145079354536315.6, 1450793545363156.0),
    ),
    reference=[material_refs["Horiba"]],
)

AlAs_Horiba = VariantItem(
    medium=PoleResidue(
        eps_inf=1.0,
        poles=[
            ((-287141547671268.06 - 1j * 6859562349716031.0), (0.0 + 1j * 2.4978200955702556e16))
        ],
        frequency_range=(0.0, 725396772681578.0),
    ),
    reference=[material_refs["Horiba"]],
)

AlAs_FernOnton1971 = VariantItem(
    medium=PoleResidue(
        eps_inf=1,
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
        eps_inf=1.0,
        poles=[
            ((-96473482947754.08 - 1j * 1.0968686723518324e16), (0.0 + 1j * 1.974516343551917e16))
        ],
        frequency_range=(145079354536315.6, 967195696908770.8),
    ),
    reference=[material_refs["Horiba"]],
)

AlN_Horiba = VariantItem(
    medium=PoleResidue(
        eps_inf=1.0,
        poles=[((-0.0 - 1j * 1.354578856633347e16), (0.0 + 1j * 2.2391188500149228e16))],
        frequency_range=(181349193170394.5, 1148544890079165.2),
    ),
    reference=[material_refs["Horiba"]],
)

AlxOy_Horiba = VariantItem(
    medium=PoleResidue(
        eps_inf=1.0,
        poles=[
            ((-654044636362332.8 - 1j * 1.9535949662203744e16), (0.0 + 1j * 2.123004231270711e16))
        ],
        frequency_range=(145079354536315.6, 1450793545363156.0),
    ),
    reference=[material_refs["Horiba"]],
)

Aminoacid_Horiba = VariantItem(
    medium=PoleResidue(
        eps_inf=1.0,
        poles=[((-0.0 - 1j * 2.2518582114198596e16), (0.0 + 1j * 5472015453750259.0))],
        frequency_range=(362698386340789.0, 1208994621135963.5),
    ),
    reference=[material_refs["Horiba"]],
)

Au_Olmon2012evaporated = VariantItem(
    medium=PoleResidue(
        eps_inf=5.632132676065586,
        poles=(
            ((-208702733035001.06 - 205285605362650.1j), (-5278287093117479 + 1877992342820785.5j)),
            ((-5802337384288.284 - 6750566414892.662j), (4391102400709820 + 6.164348337888482e18j)),
            (
                (-56597670698540.76 - 8080114483410.944j),
                (895004078070708.5 + 5.346045584373232e18j),
            ),
        ),
        frequency_range=(12025369359446.29, 999308193769986.8),
    ),
    reference=[material_refs["Olmon2012"]],
    data_url="https://refractiveindex.info/data_csv.php?datafile=database/data-nk/"
    "main/Au/Olmon-ev.yml",
)

Au_Olmon2012stripped = VariantItem(
    medium=PoleResidue(
        eps_inf=1.8661249761826162,
        poles=(
            (
                (-909376873.6996255 - 4596858854036.634j),
                (6.746525460331022e16 + 5.926266046979877e18j),
            ),
            ((-2211438487782.0527 + 0j), (5.660718217037341e17 + 6.245539733887402e18j)),
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
        eps_inf=2.093707117588658,
        poles=(
            (
                (-90837872195258.47 - 285647368378.67224j),
                (3.62598600536222e16 - 4.1780217126662364e18j),
            ),
            (
                (-143173969385.42313 - 5077357179706.552j),
                (3.491965542317068e17 + 6.19626961036357e18j),
            ),
            (
                (-112863245755655.73 - 1485493875832.8145j),
                (-3.854564084910335e17 + 4.175729670090279e18j),
            ),
        ),
        frequency_range=(12025369359446.29, 999308193769986.8),
    ),
    reference=[material_refs["Olmon2012"]],
    data_url="https://refractiveindex.info/data_csv.php?datafile=database/data-nk/"
    "main/Au/Olmon-sc.yml",
)

Au_Olmon2012Drude = VariantItem(
    medium=PoleResidue(
        eps_inf=1.0,
        poles=(
            (
                (0.0j),
                ((1.153665672616558e18 + 0j)),
            ),
            (
                (-71428570000000 + 0j),
                (-1.153665672616558e18 - 0j),
            ),
        ),
        frequency_range=(12025369359446.29, 241798930000000),
    ),
    reference=[material_refs["Olmon2012"]],
    data_url="https://refractiveindex.info/data_csv.php?datafile=database/data-nk/"
    "main/Au/Olmon-sc.yml",
)

Au_JohnsonChristy1972 = VariantItem(
    medium=PoleResidue(
        eps_inf=1.0,
        poles=[
            (
                (-2734662976094585.0 - 1j * 5109708411015428.0),
                (6336826024756207.0 + 1j * 4435873101906770.0),
            ),
            (
                (-1350147983711818.5 - 1j * 5489311548525578.0),
                (1313699470597296.0 + 1j * 2519572763961442.0),
            ),
            (
                (-617052918383578.8 - 1j * 4245316498596240.5),
                (577794256452581.6 + 1j * 1959978954055246.2),
            ),
            (
                (-49323313828269.45 + 1j * 357801380626459.0),
                (107506676273403.77 - 1j * 1.4556042795341494e17),
            ),
            (
                (-1443242886602454.5 + 1j * 1.2515133019565118e16),
                (230166586216985.78 - 1j * 3809468920144284.5),
            ),
            (
                (-258129278193.38495 + 1j * 126209156799910.83),
                (972898514880373.2 - 1j * 2.6164309961808477e17),
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
        eps_inf=1.0,
        poles=[
            ((-1146636944.7421875 + 0j), (8.88238982652701e17 + 0j)),
            ((-80520028106850 + 0j), (-8.88238982652701e17 - 0j)),
            ((-183071727483533.34 - 603332360445186.9j), 3743420309393974.5j),
            ((-262073634779331.97 - 1233457817766871.8j), 762938741152796.6j),
            ((-660881339878315.4 - 4462028230599516j), 1497407504712811j),
            ((-1894526507651171.2 - 6258461223088549j), 9036929133946472j),
            ((-1681829064931713 - 2.0166634496554556e16j), 2.0457430700884664e16j),
        ],
        frequency_range=(4.83598623e13, 1.20898681e15),
    ),
    reference=[material_refs["Rakic1998"]],
    data_url="https://refractiveindex.info/data_csv.php?datafile=database/data-nk/"
    "main/Au/Rakic-LD.yml",
)

BK7_Zemax = VariantItem(
    medium=PoleResidue(
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
        eps_inf=1.0,
        poles=[
            (
                (-1895389650993988.8 + 1j * 97908760254751.03),
                (40119229416830.445 - 1j * 6.072472443146835e17),
            ),
            (
                (-173563254483411.3 - 1j * 39098441331858.36),
                (17327582796970.727 + 1j * 2.1782706819526035e17),
            ),
            (
                (-3894265931723855.5 + 1j * 4182034916796805.5),
                (12304771601918.207 - 1j * 7.207815056419813e16),
            ),
            (
                (-21593264136101.0 + 1j * 15791763527.314959),
                (10898385976899.773 - 1j * 1.844312751315413e21),
            ),
        ],
        frequency_range=(4835978484543.8545, 1208994621135963.5),
    ),
    reference=[material_refs["Rakic1998"]],
    data_url="https://refractiveindex.info/data_csv.php?datafile=database/data-nk/"
    "main/Be/Rakic-BB.yml",
)

Be_RakicLorentzDrude1998 = VariantItem(
    medium=PoleResidue(
        eps_inf=1.0,
        poles=(
            ((-1108290667017.4727 + 0j), (6.51808619662519e17 + 0j)),
            ((-52066070012846.984 + 0j), (-6.51808619662519e17 - 0j)),
            ((-9163427576987.25 + 0j), (4884108194218923 + 0j)),
            ((-2518897605888569 + 0j), (-4884108194218923 - 0j)),
            ((-531334375653411.75 + 0j), (1.351759826496727e16 + 0j)),
            ((-4626578610293440 + 0j), (-1.351759826496727e16 - 0j)),
            ((-3383408606687375.5 - 3455109465888045.5j), 6.06548265916751e16j),
            ((-1368859970644510.8 - 6859457195810405j), 7493848504616175j),
        ),
        frequency_range=(4835978484543.8545, 1208994621135963.5),
    ),
    reference=[material_refs["Rakic1998"]],
    data_url="https://refractiveindex.info/data_csv.php?datafile=database/data-nk/"
    "main/Be/Rakic-LD.yml",
)

CaF2_Horiba = VariantItem(
    medium=PoleResidue(
        eps_inf=1.0,
        poles=[((-0.0 - 1j * 2.376134288665943e16), (0.0 + 1j * 1.2308375615289586e16))],
        frequency_range=(181349193170394.5, 1148544890079165.2),
    ),
    reference=[material_refs["Horiba"]],
)

Cellulose_Sultanova2009 = VariantItem(
    medium=PoleResidue(
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
        eps_inf=1.0,
        poles=[
            (
                (-1986166383636938.8 - 1j * 2164878977347264.2),
                (7556808013710.747 + 1j * 7.049099034302554e16),
            ),
            (
                (-721541271079502.1 - 1j * 373401161923.8366),
                (310196803320813.3 + 1j * 3.9059060187608424e19),
            ),
            (
                (-63813936856379.42 - 1j * 74339943925.90295),
                (9692153948376.459 + 1j * 1.677574997330204e20),
            ),
            (
                (-14969882528204.193 + 1j * 2792246309026.462),
                (1365296575589394.2 - 1j * 3.587733271017399e18),
            ),
        ],
        frequency_range=(4835362227919.29, 1208840556979822.5),
    ),
    reference=[material_refs["Rakic1998"]],
    data_url="https://refractiveindex.info/data_csv.php?datafile=database/data-nk/"
    "main/Cr/Rakic-BB.yml",
)

Cr_RakicLorentzDrude1998 = VariantItem(
    medium=PoleResidue(
        eps_inf=1.0,
        poles=(
            ((-817479822341.9102 + 0j), (3.211383394563691e17 + 0j)),
            ((-70588090233476.08 + 0j), (-3.211383394563691e17 - 0j)),
            ((-7016061501736.5 + 0j), (4187160341714059 + 0j)),
            ((-4816658085885968 + 0j), (-4187160341714059 - 0j)),
            ((-441634229628194.1 + 0j), (1.8197032850966144e16 + 0j)),
            ((-1541009790006751.5 + 0j), (-1.8197032850966144e16 - 0j)),
            ((-2032779845418818.5 - 2196724138579424.2j), 6.975894511603244e16j),
            ((-1014111021537414.9 - 1.3292945008240806e16j), 8277289379024513j),
        ),
        frequency_range=(4835978484543.8545, 1208994621135963.5),
    ),
    reference=[material_refs["Rakic1998"]],
    data_url="https://refractiveindex.info/data_csv.php?datafile=database/data-nk/"
    "main/Cr/Rakic-LD.yml",
)

Cu_JohnsonChristy1972 = VariantItem(
    medium=PoleResidue(
        eps_inf=1.0717963962915042,
        poles=(
            (
                (-1875546376431306 - 3382015888128589.5j),
                (4501596678511654.5 - 4.40812176484843e16j),
            ),
            (
                (-4395398245090650 - 1571964241780689.5j),
                (3281817129816946.5 + 9.445791153795858e16j),
            ),
            (
                (-57069946429575.93 - 17754659293216.31j),
                (1226529686083922.8 + 4.5093110134769004e18j),
            ),
            (
                (-208846699968783.94 - 2170839355554.1362j),
                (-1518719353508992.2 + 4.4544685629246894e18j),
            ),
            ((-1292418447472384 - 3584018186373431j), (2786681303341121.5 + 2.134483285446143e16j)),
        ),
        frequency_range=(154771532266391.3, 1595489398616285.2),
    ),
    reference=[material_refs["JohnsonChristy1972"]],
    data_url="https://refractiveindex.info/data_csv.php?datafile=database/data-nk/"
    "main/Cu/Johnson.yml",
)

Cu_RakicLorentzDrude1998 = VariantItem(
    medium=PoleResidue(
        eps_inf=1.0,
        poles=(
            ((-506299046.578125 + 0j), (1.7077228480506522e18 + 0j)),
            ((-45577517140837.24 + 0j), (-1.7077228480506522e18 - 0j)),
            ((-287141547671268.06 - 336166890703636.9j), 2.4562370654105788e16j),
            ((-802173212541955.2 - 4420275938629775j), 3184779293720060j),
            ((-2440703155205778.5 - 7673302022556904j), 1.275414610754998e16j),
            ((-3270223181811664 - 1.6667627171842064e16j), 5181342297925362j),
        ),
        frequency_range=(2.41768111e13, 1.44827274e15),
    ),
    reference=[material_refs["Rakic1998"]],
    data_url="https://refractiveindex.info/data_csv.php?datafile=database/data-nk/"
    "main/Cu/Rakic-LD.yml",
)

FusedSilica_Zemax = VariantItem(
    medium=PoleResidue(
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

Ge_Icenogle1976 = VariantItem(
    medium=PoleResidue(
        eps_inf=1,
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

GeOx_Horiba = VariantItem(
    medium=PoleResidue(
        eps_inf=1.0,
        poles=[
            ((-351710414211103.44 - 1j * 2.4646085673376252e16), (0.0 + 1j * 2.02755336442934e16))
        ],
        frequency_range=(145079354536315.6, 967195696908770.8),
    ),
    reference=[material_refs["Horiba"]],
)

H2O_Horiba = VariantItem(
    medium=PoleResidue(
        eps_inf=1.0,
        poles=[((-0.0 - 1j * 1.7289263558195928e16), (0.0 + 1j * 5938862032240302.0))],
        frequency_range=(362698386340789.0, 1450793545363156.0),
    ),
    reference=[material_refs["Horiba"]],
)

HMDS_Horiba = VariantItem(
    medium=PoleResidue(
        eps_inf=1.0,
        poles=[
            ((-379816861999031.8 - 1j * 1.8227252520914852e16), (0.0 + 1j * 1.0029341899480378e16))
        ],
        frequency_range=(362698386340789.0, 1571693007476752.5),
    ),
    reference=[material_refs["Horiba"]],
)

HfO2_Horiba = VariantItem(
    medium=PoleResidue(
        eps_inf=1.0,
        poles=[
            ((-2278901171994190.5 - 1j * 1.4098114301144558e16), (0.0 + 1j * 1.3743164680834702e16))
        ],
        frequency_range=(362698386340789.0, 1450793545363156.0),
    ),
    reference=[material_refs["Horiba"]],
)

ITO_Horiba = VariantItem(
    medium=PoleResidue(
        eps_inf=1.0,
        poles=[
            ((-483886682186766.56 - 1j * 1.031968022520672e16), (0.0 + 1j * 1.292796190658882e16))
        ],
        frequency_range=(362698386340789.0, 1450793545363156.0),
    ),
    reference=[material_refs["Horiba"]],
)

InP_Pettit1965 = VariantItem(
    medium=PoleResidue(
        eps_inf=1,
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

MgF2_Horiba = VariantItem(
    medium=PoleResidue(
        eps_inf=1.0,
        poles=[((-0.0 - 1j * 2.5358092974503356e16), (0.0 + 1j * 1.1398462792039258e16))],
        frequency_range=(193439139381754.16, 918835912063332.1),
    ),
    reference=[material_refs["Horiba"]],
)

MgO_StephensMalitson1952 = VariantItem(
    medium=PoleResidue(
        eps_inf=1.0,
        poles=[
            (
                (-56577071909034.84 + 1j * 1.709097252165159e16),
                (104656337098134.19 - 1j * 1.5807476741024398e16),
            ),
            (
                (-1.4437966258192067e17 - 1j * 2258757151354688.5),
                (1.5132011505098516e16 - 1j * 4.810654072512032e17),
            ),
            (
                (-982824644.4296285 - 1j * 4252237346494.8228),
                (338287950556.00256 + 1j * 4386571425642974.0),
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
            ((-1819477367096665 - 586975596758.178j), (3506055897617337 + 1.561199088683625e18j)),
            ((-2001079540362000.8 - 6914798333407941j), (999447311644327.9 + 8623994636438280j)),
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
        eps_inf=1.0,
        poles=(
            ((-800062663506.125 + 0j), (3.936912856387643e17 + 0j)),
            ((-72124774840307.98 + 0j), (-3.936912856387643e17 - 0j)),
            ((-10211922369538.5 + 0j), (4280689317169589.5 + 0j)),
            ((-6843203535540992 + 0j), (-4280689317169589.5 - 0j)),
            ((-518328915630819.8 + 0j), (3.988443595266845e16 + 0j)),
            ((-1508373859996014.5 + 0j), (-3.988443595266845e16 - 0j)),
            ((-1654482250867782.5 - 1774676068987181.8j), 1.7470742743872056e16j),
            ((-4779615391395816 - 7920412739409055j), 2.6921813490544444e16j),
        ),
        frequency_range=(48359784845438.55, 1208994621135963.5),
    ),
    reference=[material_refs["Rakic1998"]],
    data_url="https://refractiveindex.info/data_csv.php?datafile=database/data-nk/"
    "main/Ni/Rakic-LD.yml",
)

PEI_Horiba = VariantItem(
    medium=PoleResidue(
        eps_inf=1.0,
        poles=[((-0.0 - 1j * 1.8231209375953524e16), (0.0 + 1j * 9936009109894670.0))],
        frequency_range=(181349193170394.5, 1148544890079165.2),
    ),
    reference=[material_refs["Horiba"]],
)

PEN_Horiba = VariantItem(
    medium=PoleResidue(
        eps_inf=1.0,
        poles=[((-0.0 - 1j * 6981033923542204.0), (0.0 + 1j * 5117097865956436.0))],
        frequency_range=(362698386340789.0, 773756557527016.6),
    ),
    reference=[material_refs["Horiba"]],
)

PET_Horiba = VariantItem(
    medium=PoleResidue(
        eps_inf=1.0,
        poles=[((-0.0 - 1j * 1.063487213597289e16), (0.0 + 1j * 1.169835934957018e16))],
    ),
    reference=[material_refs["Horiba"]],
)

PMMA_Horiba = VariantItem(
    medium=PoleResidue(
        eps_inf=1.0,
        poles=[((-0.0 - 1j * 1.7360669128251744e16), (0.0 + 1j * 1.015599144002727e16))],
        frequency_range=(181349193170394.5, 1100185105233726.6),
    ),
    reference=[material_refs["Horiba"]],
)

PMMA_Sultanova2009 = VariantItem(
    medium=PoleResidue(
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
        eps_inf=1.0,
        poles=[((-0.0 - 1j * 2.5039046810424176e16), (0.0 + 1j * 8763666383648461.0))],
        frequency_range=(362698386340789.0, 1571693007476752.5),
    ),
    reference=[material_refs["Horiba"]],
)

PVC_Horiba = VariantItem(
    medium=PoleResidue(
        eps_inf=1.0,
        poles=[((-0.0 - 1j * 1.8551774807480708e16), (0.0 + 1j * 1.209575717447742e16))],
        frequency_range=(362698386340789.0, 1148544890079165.2),
    ),
    reference=[material_refs["Horiba"]],
)

Pd_JohnsonChristy1972 = VariantItem(
    medium=PoleResidue(
        eps_inf=1.0,
        poles=[
            (
                (-27947601188212.62 - 1j * 88012749128378.45),
                (-116820857784644.19 + 1j * 4.431305747926611e17),
            ),
            ((-42421241831450.59 + 1j * 0.0), (2.0926917440899536e16 - 1j * 2.322604734166214e17)),
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
        eps_inf=1.0,
        poles=(
            ((-6077069791984.51 - 4557802343988.38j), 7.894587671231656e18j),
            ((-58916603694997.75 + 0j), (1.6215501354199708e16 + 0j)),
            ((-4422922367893578 + 0j), (-1.6215501354199708e16 - 0j)),
            ((-421596716818925.4 - 633727137461217.1j), 2.0818721955845844e16j),
            ((-1067065603800966.5 + 0j), (1.4236470639056928e16 + 0j)),
            ((-5953469273389138 + 0j), (-1.4236470639056928e16 - 0j)),
            ((-2458174730857734 - 8327373750489667j), 5931453695969745j),
        ),
        frequency_range=(24179892422719.273, 1208994621135963.5),
    ),
    reference=[material_refs["Rakic1998"]],
    data_url="https://refractiveindex.info/data_csv.php?datafile=database/data-nk/"
    "main/Pd/Rakic-LD.yml",
)

Polycarbonate_Horiba = VariantItem(
    medium=PoleResidue(
        eps_inf=1.0,
        poles=[((-0.0 - 1j * 1.8240324980641504e16), (0.0 + 1j * 1.3716724385442412e16))],
        frequency_range=(362698386340789.0, 967195696908770.8),
    ),
    reference=[material_refs["Horiba"]],
)

Polycarbonate_Sultanova2009 = VariantItem(
    medium=PoleResidue(
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
        eps_inf=1.0,
        poles=[
            (
                (-101718046412896.23 - 1j * 222407105780688.0),
                (4736075731111783.0 + 1j * 7.146182537352074e17),
            ),
            (
                (-78076341531946.67 - 1j * 60477052937666.555),
                (5454987478240738.0 + 1j * 4.413657205572709e17),
            ),
            (
                (-6487635330201033.0 - 1j * 155489439108998.5),
                (5343260155670645.0 + 1j * 2.067963085430939e17),
            ),
            (
                (-2281398148570798.5 - 1j * 64631536899092.15),
                (-1930595420879896.2 - 1j * 4.8251418308161344e17),
            ),
            (
                (-9967323231923196.0 - 1j * 4041974141709040.5),
                (-501748269346742.7 + 1j * 6.883385112306915e16),
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
        eps_inf=1.0,
        poles=(
            ((-476640283942.46875 + 0j), (2.9309972445197843e17 + 0j)),
            ((-121064755555747.72 + 0j), (-2.9309972445197843e17 - 0j)),
            ((-392730635306998.94 - 1118058787933578.1j), 1.813194578357386e16j),
            ((-1396206784708441 - 1426846131279794.5j), 4.9021202075413656e16j),
            ((-2786336499624897.5 - 3874079860313212j), 1.498630066235504e16j),
            ((-6469800427291508 - 1.2473655652689588e16j), 3.042842289267071e16j),
        ),
        frequency_range=(24179892422719.273, 1208994621135963.5),
    ),
    reference=[material_refs["Rakic1998"]],
    data_url="https://refractiveindex.info/data_csv.php?datafile=database/data-nk/"
    "main/Pt/Rakic-LD.yml",
)

Sapphire_Horiba = VariantItem(
    medium=PoleResidue(
        eps_inf=1.0,
        poles=[((-0.0 - 1j * 2.0143967092980652e16), (0.0 + 1j * 2.105044561216478e16))],
        frequency_range=(362698386340789.0, 1329894083249559.8),
    ),
    reference=[material_refs["Horiba"]],
)

Si3N4_Horiba = VariantItem(
    medium=PoleResidue(
        eps_inf=1.0,
        poles=[
            ((-1357465464784539.5 - 1j * 4646140872332419.0), (0.0 + 1j * 1.103606337254506e16))
        ],
        frequency_range=(362698386340789.0, 1329894083249559.8),
    ),
    reference=[material_refs["Horiba"]],
)

Si3N4_Luke2015 = VariantItem(
    medium=PoleResidue(
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
        eps_inf=3.031225983820944,
        poles=((-7534484687295489j, 3530332266482328j), (-4550924050946271j, 7233481618.869821j)),
        frequency_range=(152024573088740.38, 724311326723836.8),
    ),
    reference=[material_refs["Luke2015"]],
    data_url="https://refractiveindex.info/data_csv.php?datafile=database/data-nk/"
    "main/Si3N4/Luke.yml",
)

Si3N4_Philipp1973 = VariantItem(
    medium=PoleResidue(
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
        eps_inf=3.0,
        poles=[((-0.0 - 1j * 1.2154139583969018e16), (0.0 + 1j * 2.3092865209541132e16))],
        frequency_range=(145079354536315.6, 967195696908770.8),
    ),
    reference=[material_refs["Horiba"]],
)

SiN_Horiba = VariantItem(
    medium=PoleResidue(
        eps_inf=2.32,
        poles=[((-302334222151229.3 - 1j * 9863009385232968.0), (0.0 + 1j * 6244215164693547.0))],
        frequency_range=(145079354536315.6, 1450793545363156.0),
    ),
    reference=[material_refs["Horiba"]],
)

SiO2_Horiba = VariantItem(
    medium=PoleResidue(
        eps_inf=1.0,
        poles=[
            ((-75963372399806.36 - 1j * 1.823105111824081e16), (0.0 + 1j * 1.0209565875622414e16))
        ],
        frequency_range=(169259246959034.88, 1208994621135963.5),
    ),
    reference=[material_refs["Horiba"]],
)

SiON_Horiba = VariantItem(
    medium=PoleResidue(
        eps_inf=1.0,
        poles=[((-0.0 - 1j * 1.651139862482191e16), (0.0 + 1j * 1.1079148477255502e16))],
        frequency_range=(181349193170394.5, 725396772681578.0),
    ),
    reference=[material_refs["Horiba"]],
)

Ta2O5_Horiba = VariantItem(
    medium=PoleResidue(
        eps_inf=1.0,
        poles=[
            ((-618341851334423.8 - 1j * 1.205777404193952e16), (0.0 + 1j * 1.8938176054079756e16))
        ],
        frequency_range=(181349193170394.5, 967195696908770.8),
    ),
    reference=[material_refs["Horiba"]],
)

Ti_Werner2009 = VariantItem(
    medium=PoleResidue(
        eps_inf=1.0,
        poles=[
            ((-55002727357489.695 - 1j * 103457301057900.64), (0.0 + 1j * 1.4157836508658926e18)),
            ((-3889516074161299.0 - 1j * 6.314261108475189e16), (0.0 + 1j * 2192302508847248.2)),
            ((-2919746613155850.5 - 1j * 7.211858151732786e16), (0.0 + 1j * 744301222539582.0)),
            ((-4635394958195360.0 - 1j * 5.622429893839941e16), (0.0 + 1j * 2101343798471838.0)),
            ((-9774364062177540.0 - 1j * 4844300045008988.0), (0.0 + 1j * 7.377824793744533e16)),
        ],
        frequency_range=(120884055879414.03, 2997924585809468.0),
    ),
    reference=[material_refs["Werner2009"]],
    data_url="https://refractiveindex.info/data_csv.php?datafile=database/data-nk/"
    "main/Ti/Werner.yml",
)

Ti_RakicLorentzDrude1998 = VariantItem(
    medium=PoleResidue(
        eps_inf=1.0,
        poles=(
            ((-464807447764.2969 + 0j), (7.341080927693171e16 + 0j)),
            ((-124115123287918.16 + 0j), (-7.341080927693171e16 - 0j)),
            ((-465726048089722.5 + 0j), (2.1824836537305828e16 + 0j)),
            ((-2992126663549463 + 0j), (-2.1824836537305828e16 - 0j)),
            ((-1912757717027124.2 - 1360524146154420.5j), 1.7716577274303782e16j),
            ((-1263270883008780 - 3596426881658456.5j), 3189068866500566j),
            ((-1338474621684588.2 - 2.9489006173628724e16j), 2079856587113.8086j),
        ),
        frequency_range=(9.67072446e12, 1.20884056e15),
    ),
    reference=[material_refs["Rakic1998"]],
    data_url="https://refractiveindex.info/data_csv.php?datafile=database/data-nk/"
    "main/Ti/Rakic-LD.yml",
)

TiOx_Horiba = VariantItem(
    medium=PoleResidue(
        eps_inf=0.29,
        poles=[((-0.0 - 1j * 9875238411974826.0), (0.0 + 1j * 1.7429795797135566e16))],
        frequency_range=(145079354536315.6, 725396772681578.0),
    ),
    reference=[material_refs["Horiba"]],
)

TiOx_HoribaStable = VariantItem(
    medium=PoleResidue(
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
        eps_inf=1.0,
        poles=[
            (
                (-6008545281436.0 - 1j * 273822982315836.25),
                (2874701466157776.0 + 1j * 6.354855141434104e17),
            ),
            (
                (-18716635733325.97 - 1j * 7984905262277.852),
                (2669048417776342.0 + 1j * 1.4111869583971584e17),
            ),
            (
                (-7709052771634303.0 - 1j * 64340875428723.28),
                (501889387931716.2 + 1j * 5.510078120444142e16),
            ),
            (
                (-330546522884264.1 - 1j * 1422878310689065.0),
                (584859595267922.1 + 1j * 3.664402566039364e16),
            ),
            (
                (-3989296857299139.0 - 1j * 3986090497375137.0),
                (-352374832782093.06 + 1j * 6.323677441887342e16),
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
        eps_inf=1.0,
        poles=(
            ((-596977842693.5781 + 0j), (4.3263359766348934e17 + 0j)),
            ((-96636138829058.58 + 0j), (-4.3263359766348934e17 - 0j)),
            ((-402605873718973.75 - 1471252666401400j), 7403002173803200j),
            ((-973090800441519.6 - 2745063931489722j), 1.219711179953004e16j),
            ((-2531099568361548 - 4814146946972908j), 2.9579221430831028e16j),
            ((-4433222413252700 - 1.0493429699239636e16j), 4.978330061510859e16j),
        ),
        frequency_range=(2.41768111e13, 1.20884056e15),
    ),
    reference=[material_refs["Rakic1998"]],
    data_url="https://refractiveindex.info/data_csv.php?datafile=database/data-nk/"
    "main/W/Rakic-LD.yml",
)

WS2_Li2014 = VariantItem2D(
    medium=Medium2D.from_dispersive_medium(
        PoleResidue(
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
        eps_inf=1.0,
        poles=[((-0.0 - 1j * 1.3814698904628784e16), (0.0 + 1j * 1.1846104310719182e16))],
        frequency_range=(374788332552148.7, 967195696908770.8),
    ),
    reference=[material_refs["Horiba"]],
)

Y2O3_Nigara1968 = VariantItem(
    medium=PoleResidue(
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
        eps_inf=1.0,
        poles=[
            ((-97233116671752.14 - 1j * 1.446765717253359e16), (0.0 + 1j * 2.0465425413547396e16))
        ],
        frequency_range=(362698386340789.0, 725396772681578.0),
    ),
    reference=[material_refs["Horiba"]],
)

aSi_Horiba = VariantItem(
    medium=PoleResidue(
        eps_inf=3.109,
        poles=[
            ((-1458496750076282.0 - 1j * 5789844327200831.0), (0.0 + 1j * 4.485863370051096e16))
        ],
        frequency_range=(362698386340789.0, 1450793545363156.0),
    ),
    reference=[material_refs["Horiba"]],
)

cSi_SalzbergVilla1957 = VariantItem(
    medium=PoleResidue(
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
        eps_inf=1.0,
        poles=[
            (
                (-516580533476358.94 - 1j * 7988869406082532.0),
                (531784950915900.1 + 1j * 4114144409090735.5),
            ),
            (
                (-422564506478804.25 - 1j * 6388843514992565.0),
                (2212987364690094.5 + 1j * 1.665883190033301e16),
            ),
            (
                (-169315596364414.94 + 1j * 5194420450502291.0),
                (301374428182025.6 - 1j * 4618167601749804.0),
            ),
            (
                (-379444981070553.4 + 1j * 5656363945615038.0),
                (1105733518717537.1 - 1j * 8204725853411607.0),
            ),
        ],
        frequency_range=(206753419710997.8, 1199169834323787.2),
    ),
    reference=[material_refs["Green2008"]],
    data_url="https://refractiveindex.info/data_csv.php?datafile=database/data-nk/"
    "main/Si/Green-2008.yml",
)


material_library = dict(
    Ag=MaterialItem(
        name="Silver",
        variants=dict(
            Rakic1998BB=Ag_Rakic1998BB,
            JohnsonChristy1972=Ag_JohnsonChristy1972,
            RakicLorentzDrude1998=Ag_RakicLorentzDrude1998,
            Yang2015Drude=Ag_Yang2015Drude,
        ),
        default="Rakic1998BB",
    ),
    Al=MaterialItem(
        name="Aluminum",
        variants=dict(
            Rakic1995=Al_Rakic1995,
            RakicLorentzDrude1998=Al_RakicLorentzDrude1998,
        ),
        default="Rakic1995",
    ),
    Al2O3=MaterialItem(
        name="Alumina",
        variants=dict(
            Horiba=Al2O3_Horiba,
        ),
        default="Horiba",
    ),
    AlAs=MaterialItem(
        name="Aluminum Arsenide",
        variants=dict(
            Horiba=AlAs_Horiba,
            FernOnton1971=AlAs_FernOnton1971,
        ),
        default="Horiba",
    ),
    AlGaN=MaterialItem(
        name="Aluminum Gallium Nitride",
        variants=dict(
            Horiba=AlGaN_Horiba,
        ),
        default="Horiba",
    ),
    AlN=MaterialItem(
        name="Aluminum Nitride",
        variants=dict(
            Horiba=AlN_Horiba,
        ),
        default="Horiba",
    ),
    AlxOy=MaterialItem(
        name="Aluminum Oxide",
        variants=dict(
            Horiba=AlxOy_Horiba,
        ),
        default="Horiba",
    ),
    Aminoacid=MaterialItem(
        name="Amino Acid",
        variants=dict(
            Horiba=Aminoacid_Horiba,
        ),
        default="Horiba",
    ),
    Au=MaterialItem(
        name="Gold",
        variants=dict(
            Olmon2012crystal=Au_Olmon2012crystal,
            Olmon2012stripped=Au_Olmon2012stripped,
            Olmon2012evaporated=Au_Olmon2012evaporated,
            Olmon2012Drude=Au_Olmon2012Drude,
            JohnsonChristy1972=Au_JohnsonChristy1972,
            RakicLorentzDrude1998=Au_RakicLorentzDrude1998,
        ),
        default="Olmon2012evaporated",
    ),
    BK7=MaterialItem(
        name="N-BK7 Borosilicate Glass",
        variants=dict(
            Zemax=BK7_Zemax,
        ),
        default="Zemax",
    ),
    Be=MaterialItem(
        name="Beryllium",
        variants=dict(
            Rakic1998BB=Be_Rakic1998BB,
            RakicLorentzDrude1998=Be_RakicLorentzDrude1998,
        ),
        default="Rakic1998BB",
    ),
    CaF2=MaterialItem(
        name="Calcium Fluoride",
        variants=dict(
            Horiba=CaF2_Horiba,
        ),
        default="Horiba",
    ),
    Cellulose=MaterialItem(
        name="Cellulose",
        variants=dict(
            Sultanova2009=Cellulose_Sultanova2009,
        ),
        default="Sultanova2009",
    ),
    Cr=MaterialItem(
        name="Chromium",
        variants=dict(
            Rakic1998BB=Cr_Rakic1998BB,
            RakicLorentzDrude1998=Cr_RakicLorentzDrude1998,
        ),
        default="Rakic1998BB",
    ),
    Cu=MaterialItem(
        name="Copper",
        variants=dict(
            JohnsonChristy1972=Cu_JohnsonChristy1972,
            RakicLorentzDrude1998=Cu_RakicLorentzDrude1998,
        ),
        default="JohnsonChristy1972",
    ),
    FusedSilica=MaterialItem(
        name="Fused Silica",
        variants=dict(
            ZemaxSellmeier=FusedSilica_Zemax,
            ZemaxVisiblePMLStable=FusedSilica_Zemax_Visible_PMLStable,
            ZemaxPMLStable=FusedSilica_Zemax_PMLStable,
        ),
        default="ZemaxPMLStable",
    ),
    GaAs=MaterialItem(
        name="Gallium Arsenide",
        variants=dict(
            Skauli2003=GaAs_Skauli2003,
        ),
        default="Skauli2003",
    ),
    Ge=MaterialItem(
        name="Germanium",
        variants=dict(
            Icenogle1976=Ge_Icenogle1976,
        ),
        default="Icenogle1976",
    ),
    GeOx=MaterialItem(
        name="Germanium Oxide",
        variants=dict(
            Horiba=GeOx_Horiba,
        ),
        default="Horiba",
    ),
    H2O=MaterialItem(
        name="Water",
        variants=dict(
            Horiba=H2O_Horiba,
        ),
        default="Horiba",
    ),
    HMDS=MaterialItem(
        name="Hexamethyldisilazane, or Bis(trimethylsilyl)amine",
        variants=dict(
            Horiba=HMDS_Horiba,
        ),
        default="Horiba",
    ),
    HfO2=MaterialItem(
        name="Hafnium Oxide",
        variants=dict(
            Horiba=HfO2_Horiba,
        ),
        default="Horiba",
    ),
    ITO=MaterialItem(
        name="Indium Tin Oxide",
        variants=dict(
            Horiba=ITO_Horiba,
        ),
        default="Horiba",
    ),
    InP=MaterialItem(
        name="Indium Phosphide",
        variants=dict(
            Pettit1965=InP_Pettit1965,
        ),
        default="Pettit1965",
    ),
    MgF2=MaterialItem(
        name="Magnesium Fluoride",
        variants=dict(
            Horiba=MgF2_Horiba,
        ),
        default="Horiba",
    ),
    MgO=MaterialItem(
        name="Magnesium Oxide",
        variants=dict(
            StephensMalitson1952=MgO_StephensMalitson1952,
        ),
        default="StephensMalitson1952",
    ),
    MoS2=MaterialItem2D(
        name="Molybdenum Disulfide",
        variants=dict(
            Li2014=MoS2_Li2014,
        ),
        default="Li2014",
    ),
    MoSe2=MaterialItem2D(
        name="Molybdenum Diselenide",
        variants=dict(
            Li2014=MoSe2_Li2014,
        ),
        default="Li2014",
    ),
    Ni=MaterialItem(
        name="Nickel",
        variants=dict(
            JohnsonChristy1972=Ni_JohnsonChristy1972,
            RakicLorentzDrude1998=Ni_RakicLorentzDrude1998,
        ),
        default="JohnsonChristy1972",
    ),
    PEI=MaterialItem(
        name="Polyetherimide",
        variants=dict(
            Horiba=PEI_Horiba,
        ),
        default="Horiba",
    ),
    PEN=MaterialItem(
        name="Polyethylene Naphthalate",
        variants=dict(
            Horiba=PEN_Horiba,
        ),
        default="Horiba",
    ),
    PET=MaterialItem(
        name="Polyethylene Terephthalate",
        variants=dict(
            Horiba=PET_Horiba,
        ),
        default="Horiba",
    ),
    PMMA=MaterialItem(
        name="Poly(methyl Methacrylate)",
        variants=dict(
            Horiba=PMMA_Horiba,
            Sultanova2009=PMMA_Sultanova2009,
        ),
        default="Sultanova2009",
    ),
    PTFE=MaterialItem(
        name="Polytetrafluoroethylene, or Teflon",
        variants=dict(
            Horiba=PTFE_Horiba,
        ),
        default="Horiba",
    ),
    PVC=MaterialItem(
        name="Polyvinyl Chloride",
        variants=dict(
            Horiba=PVC_Horiba,
        ),
        default="Horiba",
    ),
    Pd=MaterialItem(
        name="Palladium",
        variants=dict(
            JohnsonChristy1972=Pd_JohnsonChristy1972,
            RakicLorentzDrude1998=Pd_RakicLorentzDrude1998,
        ),
        default="JohnsonChristy1972",
    ),
    Polycarbonate=MaterialItem(
        name="Polycarbonate",
        variants=dict(
            Horiba=Polycarbonate_Horiba,
            Sultanova2009=Polycarbonate_Sultanova2009,
        ),
        default="Sultanova2009",
    ),
    Polystyrene=MaterialItem(
        name="Polystyrene",
        variants=dict(
            Sultanova2009=Polystyrene_Sultanova2009,
        ),
        default="Sultanova2009",
    ),
    Pt=MaterialItem(
        name="Platinum",
        variants=dict(
            Werner2009=Pt_Werner2009,
            RakicLorentzDrude1998=Pt_RakicLorentzDrude1998,
        ),
        default="Werner2009",
    ),
    Sapphire=MaterialItem(
        name="Sapphire",
        variants=dict(
            Horiba=Sapphire_Horiba,
        ),
        default="Horiba",
    ),
    Si3N4=MaterialItem(
        name="Silicon Nitride",
        variants=dict(
            Horiba=Si3N4_Horiba,
            Luke2015Sellmeier=Si3N4_Luke2015,
            Luke2015PMLStable=Si3N4_Luke2015_PMLStable,
            Philipp1973Sellmeier=Si3N4_Philipp1973,
        ),
        default="Horiba",
    ),
    SiC=MaterialItem(
        name="Silicon Carbide",
        variants=dict(
            Horiba=SiC_Horiba,
        ),
        default="Horiba",
    ),
    SiN=MaterialItem(
        name="Silicon Mononitride",
        variants=dict(
            Horiba=SiN_Horiba,
        ),
        default="Horiba",
    ),
    SiO2=MaterialItem(
        name="Silicon Dioxide",
        variants=dict(
            Horiba=SiO2_Horiba,
        ),
        default="Horiba",
    ),
    SiON=MaterialItem(
        name="Silicon Oxynitride",
        variants=dict(
            Horiba=SiON_Horiba,
        ),
        default="Horiba",
    ),
    Ta2O5=MaterialItem(
        name="Tantalum Pentoxide",
        variants=dict(
            Horiba=Ta2O5_Horiba,
        ),
        default="Horiba",
    ),
    Ti=MaterialItem(
        name="Titanium",
        variants=dict(
            Werner2009=Ti_Werner2009,
            RakicLorentzDrude1998=Ti_RakicLorentzDrude1998,
        ),
        default="Werner2009",
    ),
    TiOx=MaterialItem(
        name="Titanium Oxide",
        variants=dict(
            Horiba=TiOx_Horiba,
            HorbiaStable=TiOx_HoribaStable,
        ),
        default="Horiba",
    ),
    W=MaterialItem(
        name="Tungsten",
        variants=dict(
            Werner2009=W_Werner2009,
            RakicLorentzDrude1998=W_RakicLorentzDrude1998,
        ),
        default="Werner2009",
    ),
    WS2=MaterialItem2D(
        name="Tungsten Disulfide",
        variants=dict(
            Li2014=WS2_Li2014,
        ),
        default="Li2014",
    ),
    WSe2=MaterialItem2D(
        name="Tungsten Diselenide",
        variants=dict(
            Li2014=WSe2_Li2014,
        ),
        default="Li2014",
    ),
    Y2O3=MaterialItem(
        name="Yttrium Oxide",
        variants=dict(
            Horiba=Y2O3_Horiba,
            Nigara1968=Y2O3_Nigara1968,
        ),
        default="Horiba",
    ),
    YAG=MaterialItem(
        name="Yttrium Aluminium Garnet",
        variants=dict(
            Zelmon1998=YAG_Zelmon1998,
        ),
        default="Zelmon1998",
    ),
    ZrO2=MaterialItem(
        name="Zirconium Oxide",
        variants=dict(
            Horiba=ZrO2_Horiba,
        ),
        default="Horiba",
    ),
    aSi=MaterialItem(
        name="Amorphous Silicon",
        variants=dict(
            Horiba=aSi_Horiba,
        ),
        default="Horiba",
    ),
    cSi=MaterialItem(
        name="Crystalline Silicon",
        variants=dict(
            SalzbergVilla1957=cSi_SalzbergVilla1957,
            Li1993_293K=cSi_Li1993_293K,
            Green2008=cSi_Green2008,
        ),
        default="SalzbergVilla1957",
    ),
    graphene=Graphene,
)
