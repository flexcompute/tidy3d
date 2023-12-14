"""Tests space time modulation."""
import numpy as np
import pytest
from math import isclose
import pydantic.v1 as pydantic
import tidy3d as td
from tidy3d.exceptions import ValidationError

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

    # uniform in phase, but custom in amplitude
    with pytest.raises(pydantic.ValidationError):
        sp = SP_UNIFORM.updated_copy(amplitude=ARRAY_CMP)

    sp = SP_UNIFORM.updated_copy(amplitude=ARRAY)
    assert isclose(sp.max_modulation, np.max(ARRAY))

    # uniform in amplitude, but custom in phase
    with pytest.raises(pydantic.ValidationError):
        sp = SP_UNIFORM.updated_copy(phase=ARRAY_CMP)
    sp = SP_UNIFORM.updated_copy(phase=ARRAY)
    assert isclose(sp.max_modulation, 1)

    # custom in both
    with pytest.raises(pydantic.ValidationError):
        sp = SP_UNIFORM.updated_copy(phase=ARRAY_CMP, amplitude=ARRAY_CMP)
    sp = SP_UNIFORM.updated_copy(phase=ARRAY, amplitude=ARRAY)


def test_space_time_modulation():
    # cw modulation, uniform in space
    assert isclose(ST.max_modulation, AMP_TIME)
    assert not ST.negligible_modulation

    # cw modulation, but 0 amplitude
    st = ST.updated_copy(time_modulation=CW.updated_copy(amplitude=0))
    assert st.negligible_modulation

    st = ST.updated_copy(space_modulation=td.SpaceModulation(amplitude=0))
    assert st.negligible_modulation

    # cw modulation, nonuniform in space
    st = ST.updated_copy(space_modulation=td.SpaceModulation(amplitude=ARRAY, phase=ARRAY))
    assert not st.negligible_modulation
    assert isclose(st.max_modulation, AMP_TIME * np.max(ARRAY))


def test_modulated_medium():
    """time modulated medium"""
    # unmodulated
    medium = td.Medium()
    assert medium.modulation_spec is None
    assert medium.time_modulated == False

    assert MODULATION_SPEC.applied_modulation == False
    medium = medium.updated_copy(modulation_spec=MODULATION_SPEC)
    assert medium.time_modulated == False

    # permittivity modulated
    modulation_spec = MODULATION_SPEC.updated_copy(permittivity=ST)
    # modulated permitivity <= 0
    with pytest.raises(pydantic.ValidationError):
        medium = td.Medium(modulation_spec=modulation_spec)
    medium = td.Medium(permittivity=2, modulation_spec=modulation_spec)
    assert isclose(medium.n_cfl, np.sqrt(2 - AMP_TIME))

    # conductivity modulated
    modulation_spec = MODULATION_SPEC.updated_copy(conductivity=ST)
    # modulated conductivity <= 0
    with pytest.raises(pydantic.ValidationError):
        medium = td.Medium(modulation_spec=modulation_spec)
    medium_sometimes_active = td.Medium(modulation_spec=modulation_spec, allow_gain=True)
    medium = td.Medium(conductivity=2, modulation_spec=modulation_spec)

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


def test_unsupported_modulated_medium_types():
    """Unsupported types of time modulated medium"""
    modulation_spec = MODULATION_SPEC.updated_copy(permittivity=ST)

    # PEC cannot be modulated
    with pytest.raises(pydantic.ValidationError):
        mat = td.PECMedium(modulation_spec=modulation_spec)

    # For Anisotropic medium, one should modulate the components, not the whole medium
    with pytest.raises(pydantic.ValidationError):
        mat = td.AnisotropicMedium(
            xx=td.Medium(), yy=td.Medium(), zz=td.Medium(), modulation_spec=modulation_spec
        )

    # Modulation to fully Anisotropic medium unsupported
    with pytest.raises(pydantic.ValidationError):
        mat = td.FullyAnisotropicMedium(modulation_spec=modulation_spec)

    # 2D material
    with pytest.raises(pydantic.ValidationError):
        drude_medium = td.Drude(eps_inf=2.0, coeffs=[(1, 2), (3, 4)])
        medium2d = td.Medium2D(ss=drude_medium, tt=drude_medium, modulation_spec=modulation_spec)

    # together with nonlinear_spec
    with pytest.raises(pydantic.ValidationError):
        mat = td.Medium(
            permittivity=2,
            nonlinear_spec=td.NonlinearSusceptibility(chi3=1),
            modulation_spec=modulation_spec,
        )


def test_supported_modulated_medium_types():
    """Supported types of time modulated medium"""
    modulation_spec = MODULATION_SPEC.updated_copy(permittivity=ST)
    modulation_both_spec = modulation_spec.updated_copy(conductivity=ST)

    # Dispersive
    mat_p = td.PoleResidue(
        eps_inf=2.0, poles=[((-1 + 2j), (3 + 4j))], modulation_spec=modulation_spec
    )
    assert mat_p.time_modulated
    assert isclose(mat_p.n_cfl, np.sqrt(2 - AMP_TIME))
    # too much modulation resulting in eps_inf < 0
    with pytest.raises(pydantic.ValidationError):
        mat = mat_p.updated_copy(eps_inf=1.0)
    # conductivity modulation
    with pytest.raises(pydantic.ValidationError):
        mat = mat_p.updated_copy(modulation_spec=modulation_both_spec)
    mat = mat_p.updated_copy(modulation_spec=modulation_both_spec, allow_gain=True)

    # custom
    permittivity = td.SpatialDataArray(np.ones((1, 1, 1)) * 2, coords=dict(x=[1], y=[1], z=[1]))
    mat_c = td.CustomMedium(permittivity=permittivity, modulation_spec=modulation_spec)
    assert mat_c.time_modulated
    assert isclose(mat_c.n_cfl, np.sqrt(2 - AMP_TIME))
    # too much modulation resulting in eps_inf < 0
    with pytest.raises(pydantic.ValidationError):
        mat = mat_c.updated_copy(permittivity=permittivity * 0.5)
    # conductivity modulation
    with pytest.raises(pydantic.ValidationError):
        mat = mat_c.updated_copy(modulation_spec=modulation_both_spec)
    mat = mat_c.updated_copy(modulation_spec=modulation_both_spec, allow_gain=True)

    # anisotropic medium component
    mat = td.AnisotropicMedium(xx=td.Medium(), yy=mat_p, zz=td.Medium())
    assert mat.time_modulated
    assert isclose(mat.n_cfl, np.sqrt(2 - AMP_TIME))

    # custom anistropic medium component
    mat_uc = td.CustomMedium(permittivity=permittivity)
    mat = td.CustomAnisotropicMedium(xx=mat_uc, yy=mat_c, zz=mat_uc)
    assert mat.time_modulated
    assert isclose(mat.n_cfl, np.sqrt(2 - AMP_TIME))
