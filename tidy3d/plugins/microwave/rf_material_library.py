"""Holds dispersive models for several commonly used RF materials."""

# from ...components.base import Tidy3dBaseModel
from ...components.medium import PoleResidue
from ...material_library.material_library import MaterialItem, VariantItem
from .rf_material_reference import rf_material_refs

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

rf_material_library = dict(
    RO3003=MaterialItem(
        name="Rogers3003",
        variants=dict(
            design=Rogers3003_design,
            process=Rogers3003_process,
        ),
        default="design",
    ),
    RO3010=MaterialItem(
        name="Rogers3010",
        variants=dict(
            design=Rogers3010_design,
            process=Rogers3010_process,
        ),
        default="design",
    ),
    RO4003C=MaterialItem(
        name="Rogers4003C",
        variants=dict(
            design=Rogers4003C_design,
            process=Rogers4003C_process,
        ),
        default="design",
    ),
    RO4350B=MaterialItem(
        name="Rogers4350B",
        variants=dict(
            design=Rogers4350B_design,
            process=Rogers4350B_process,
        ),
        default="design",
    ),
    AD255C=MaterialItem(
        name="ArlonAD255C",
        variants=dict(
            design=ArlonAD255C_design,
            process=ArlonAD255C_process,
        ),
        default="design",
    ),
    FR4=MaterialItem(
        name="FR4",
        variants=dict(
            standard=FR4_standard,
            lowloss=FR4_lowloss,
        ),
        default="standard",
    ),
)
