"""Tests space time modulation."""

from math import isclose

import numpy as np
import pydantic.v1 as pydantic
import pytest
import tidy3d as td

from ..utils import cartesian_to_unstructured

np.random.seed(4)

# space
NX, NY, NZ = 10, 9, 8
X = np.linspace(-1, 1, NX)
Y = np.linspace(-1, 1, NY)
Z = np.linspace(-1, 1, NZ)
COORDS = dict(x=X, y=Y, z=Z)
ARRAY_CMP = td.SpatialDataArray(np.random.random((NX, NY, NZ)) + 0.1j, coords=COORDS)
ARRAY = td.SpatialDataArray(np.random.random((NX, NY, NZ)), coords=COORDS)

SP_UNIFORM = td.SpaceModulation()

# time
FREQ_MODULATE = 1e12
AMP_TIME = 1.1
PHASE_TIME = 0
CW = td.ContinuousWaveTimeModulation(freq0=FREQ_MODULATE, amplitude=AMP_TIME, phase=PHASE_TIME)

# combined
ST = td.SpaceTimeModulation(
    time_modulation=CW,
)

# medium modulation spec
MODULATION_SPEC = td.ModulationSpec()

SUBSECTION = td.Box(size=(0.3, 0.4, 0.35), center=(0.4, 0.4, 0.4))


def reduce(obj):
    return obj.sel_inside(SUBSECTION.bounds)


def check_reduction(obj, obj_reduced):
    for field in ["amplitude", "phase"]:
        original = getattr(obj, field)
        reduced = getattr(obj_reduced, field)

        if isinstance(original, float):
            assert reduced == original
            continue

        diff = (original - reduced).abs
        assert diff.does_cover(SUBSECTION.bounds)
        assert np.allclose(diff, 0)


def check_sp_reduction(sp):
    check_reduction(sp, reduce(sp))


def check_st_reduction(st):
    check_reduction(st.space_modulation, reduce(st).space_modulation)


def check_med_reduction(med):
    med_red = reduce(med)

    for field in ["permittivity", "conductivity"]:
        field_mod = getattr(med.modulation_spec, field)
        field_mod_red = getattr(med_red.modulation_spec, field)
        if field_mod is None:
            assert field_mod_red is None
        else:
            check_reduction(field_mod.space_modulation, field_mod_red.space_modulation)


def check_ani_med_reduction(med):
    reduced_med = reduce(med)

    for comp, comp_red in zip(
        [med.xx, med.yy, med.zz], [reduced_med.xx, reduced_med.yy, reduced_med.zz]
    ):
        if comp.modulation_spec is None:
            assert comp_red.modulation_spec is None
        else:
            for field in ["permittivity", "conductivity"]:
                field_mod = getattr(comp.modulation_spec, field)
                field_mod_red = getattr(comp_red.modulation_spec, field)
                if field_mod is None:
                    assert field_mod_red is None
                else:
                    check_reduction(field_mod.space_modulation, field_mod_red.space_modulation)


def test_time_modulation():
    """time modulation: only supporting CW for now."""
    assert isclose(np.real(CW.amp_time(1 / FREQ_MODULATE)), AMP_TIME)
    assert isclose(CW.max_modulation, AMP_TIME)

    cw = CW.updated_copy(phase=np.pi / 4, amplitude=10)
    assert isclose(np.real(cw.amp_time(1 / FREQ_MODULATE)), np.sqrt(2) / 2 * 10)
    assert isclose(cw.max_modulation, 10)


def test_space_modulation():
    """uniform or custom space modulation"""
    # uniform in both amplitude and phase
    assert isclose(SP_UNIFORM.max_modulation, 1)
    check_sp_reduction(SP_UNIFORM)

    # uniform in phase, but custom in amplitude
    with pytest.raises(pydantic.ValidationError):
        sp = SP_UNIFORM.updated_copy(amplitude=ARRAY_CMP)

    sp = SP_UNIFORM.updated_copy(amplitude=ARRAY)
    assert isclose(sp.max_modulation, np.max(ARRAY))
    check_sp_reduction(sp)

    # uniform in amplitude, but custom in phase
    with pytest.raises(pydantic.ValidationError):
        sp = SP_UNIFORM.updated_copy(phase=ARRAY_CMP)
    sp = SP_UNIFORM.updated_copy(phase=ARRAY)
    assert isclose(sp.max_modulation, 1)
    check_sp_reduction(sp)

    # custom in both
    with pytest.raises(pydantic.ValidationError):
        sp = SP_UNIFORM.updated_copy(phase=ARRAY_CMP, amplitude=ARRAY_CMP)
    sp = SP_UNIFORM.updated_copy(phase=ARRAY, amplitude=ARRAY)
    check_sp_reduction(sp)


def test_space_time_modulation():
    # cw modulation, uniform in space
    assert isclose(ST.max_modulation, AMP_TIME)
    assert not ST.negligible_modulation
    check_st_reduction(ST)

    # cw modulation, but 0 amplitude
    st = ST.updated_copy(time_modulation=CW.updated_copy(amplitude=0))
    assert st.negligible_modulation
    check_st_reduction(st)

    st = ST.updated_copy(space_modulation=td.SpaceModulation(amplitude=0))
    assert st.negligible_modulation
    check_st_reduction(st)

    # cw modulation, nonuniform in space
    st = ST.updated_copy(space_modulation=td.SpaceModulation(amplitude=ARRAY, phase=ARRAY))
    assert not st.negligible_modulation
    assert isclose(st.max_modulation, AMP_TIME * np.max(ARRAY))
    check_st_reduction(st)


def test_modulated_medium():
    """time modulated medium"""
    # unmodulated
    medium = td.Medium()
    assert medium.modulation_spec is None
    assert not medium.is_time_modulated
    reduce(medium)

    assert not MODULATION_SPEC.applied_modulation
    medium = medium.updated_copy(modulation_spec=MODULATION_SPEC)
    assert not medium.is_time_modulated
    reduce(medium)

    # permittivity modulated
    modulation_spec = MODULATION_SPEC.updated_copy(permittivity=ST)
    # modulated permitivity <= 0
    with pytest.raises(pydantic.ValidationError):
        medium = td.Medium(modulation_spec=modulation_spec)
    medium = td.Medium(permittivity=2, modulation_spec=modulation_spec)
    assert isclose(medium.n_cfl, np.sqrt(2 - AMP_TIME))
    check_med_reduction(medium)

    # conductivity modulated
    modulation_spec = MODULATION_SPEC.updated_copy(conductivity=ST)
    # modulated conductivity <= 0
    with pytest.raises(pydantic.ValidationError):
        medium = td.Medium(modulation_spec=modulation_spec)
    medium_sometimes_active = td.Medium(modulation_spec=modulation_spec, allow_gain=True)
    medium = td.Medium(conductivity=2, modulation_spec=modulation_spec)
    check_med_reduction(medium)
    check_med_reduction(medium_sometimes_active)

    # both modulated, but different time modulation: error
    st_freq2 = ST.updated_copy(
        time_modulation=td.ContinuousWaveTimeModulation(freq0=2e12, amplitude=2)
    )
    with pytest.raises(pydantic.ValidationError):
        modulation_spec = MODULATION_SPEC.updated_copy(permittivity=ST, conductivity=st_freq2)
    # both modulated, but different space modulation: fine
    st_space2 = ST.updated_copy(space_modulation=td.SpaceModulation(amplitude=0.1))
    modulation_spec = MODULATION_SPEC.updated_copy(permittivity=ST, conductivity=st_space2)
    medium = td.Medium(
        permittivity=3,
        conductivity=1,
        modulation_spec=modulation_spec,
    )
    check_med_reduction(medium)


def test_unsupported_modulated_medium_types():
    """Unsupported types of time modulated medium"""
    modulation_spec = MODULATION_SPEC.updated_copy(permittivity=ST)

    # PEC cannot be modulated
    with pytest.raises(pydantic.ValidationError):
        td.PECMedium(modulation_spec=modulation_spec)

    # For Anisotropic medium, one should modulate the components, not the whole medium
    with pytest.raises(pydantic.ValidationError):
        td.AnisotropicMedium(
            xx=td.Medium(), yy=td.Medium(), zz=td.Medium(), modulation_spec=modulation_spec
        )

    # Modulation to fully Anisotropic medium unsupported
    with pytest.raises(pydantic.ValidationError):
        td.FullyAnisotropicMedium(modulation_spec=modulation_spec)

    # 2D material
    with pytest.raises(pydantic.ValidationError):
        drude_medium = td.Drude(eps_inf=2.0, coeffs=[(1, 2), (3, 4)])
        td.Medium2D(ss=drude_medium, tt=drude_medium, modulation_spec=modulation_spec)

    # together with nonlinear_spec
    with pytest.raises(pydantic.ValidationError):
        td.Medium(
            permittivity=2,
            nonlinear_spec=td.NonlinearSusceptibility(chi3=1),
            modulation_spec=modulation_spec,
        )


@pytest.mark.parametrize("unstructured", [True, False])
@pytest.mark.parametrize("z", [[0], [0, 1]])
def test_supported_modulated_medium_types(unstructured, z):
    """Supported types of time modulated medium"""
    modulation_spec = MODULATION_SPEC.updated_copy(permittivity=ST)
    modulation_both_spec = modulation_spec.updated_copy(conductivity=ST)

    # Dispersive
    mat_p = td.PoleResidue(
        eps_inf=2.0, poles=[((-1 + 2j), (3 + 4j))], modulation_spec=modulation_spec
    )
    assert mat_p.is_time_modulated
    assert isclose(mat_p.n_cfl, np.sqrt(2 - AMP_TIME))
    # too much modulation resulting in eps_inf < 0
    with pytest.raises(pydantic.ValidationError):
        mat = mat_p.updated_copy(eps_inf=1.0)
    # conductivity modulation
    with pytest.raises(pydantic.ValidationError):
        mat = mat_p.updated_copy(modulation_spec=modulation_both_spec)
    mat = mat_p.updated_copy(modulation_spec=modulation_both_spec, allow_gain=True)
    check_med_reduction(mat)
    check_med_reduction(mat_p)

    # custom
    permittivity = td.SpatialDataArray(
        np.ones((2, 2, len(z))) * 2, coords=dict(x=[1, 2], y=[1, 3], z=z)
    )
    if unstructured:
        permittivity = cartesian_to_unstructured(permittivity, seed=345)
    mat_c = td.CustomMedium(permittivity=permittivity, modulation_spec=modulation_spec)
    assert mat_c.is_time_modulated
    assert isclose(mat_c.n_cfl, np.sqrt(2 - AMP_TIME))
    # too much modulation resulting in eps_inf < 0
    with pytest.raises(pydantic.ValidationError):
        mat = mat_c.updated_copy(permittivity=permittivity * 0.5)
    # conductivity modulation
    with pytest.raises(pydantic.ValidationError):
        mat = mat_c.updated_copy(modulation_spec=modulation_both_spec)
    mat = mat_c.updated_copy(modulation_spec=modulation_both_spec, allow_gain=True)
    check_med_reduction(mat_c)
    check_med_reduction(mat)

    # anisotropic medium component
    mat = td.AnisotropicMedium(xx=td.Medium(), yy=mat_p, zz=td.Medium())
    assert mat.is_time_modulated
    assert isclose(mat.n_cfl, np.sqrt(2 - AMP_TIME))
    check_ani_med_reduction(mat)

    # custom anistropic medium component
    mat_uc = td.CustomMedium(permittivity=permittivity)
    mat = td.CustomAnisotropicMedium(xx=mat_uc, yy=mat_c, zz=mat_uc)
    assert mat.is_time_modulated
    assert isclose(mat.n_cfl, np.sqrt(2 - AMP_TIME))
    check_ani_med_reduction(mat)
