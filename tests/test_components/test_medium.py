"""Tests mediums."""

from __future__ import annotations

import json
import logging

import matplotlib.pyplot as plt
import numpy as np
import pydantic as pd
import pytest

import tidy3d as td
from tidy3d.exceptions import ValidationError
from tidy3d.material_library.material_library import MaterialItem, VariantItem

from ..utils import SIM_FULL, AssertLogLevel

MEDIUM = td.Medium()
ANIS_MEDIUM = td.AnisotropicMedium(xx=MEDIUM, yy=MEDIUM, zz=MEDIUM)
PEC = td.PECMedium()
PMC = td.PMCMedium()
PR = td.PoleResidue(poles=[(-1 + 1j, 2 + 2j)])
SM = td.Sellmeier(coeffs=[(1, 2)])
LZ = td.Lorentz(coeffs=[(1, 2, 3)])
DR = td.Drude(coeffs=[(1, 2)])
DB = td.Debye(coeffs=[(1, 2)])
MEDIUMS = [MEDIUM, ANIS_MEDIUM, PEC, PR, SM, LZ, DR, DB, PMC]

f, AX = plt.subplots()

RTOL = 0.001
PALIK_LOWLOSS_MATERIALS = ("GaAs", "Ge", "InP", "SiO2", "cSi")
PALIK_NOLOSS_MATERIALS = ("GaAs", "Ge", "SiO2", "cSi")
PALIK_NOLOSS_VARIANT = "Palik_NoLoss"
PALIK_LOWLOSS_VARIANT = "Palik_LowLoss"


@pytest.mark.parametrize("component", MEDIUMS)
def test_plot(component):
    _ = component.plot(freqs=[2e14, 3e14], ax=AX)
    plt.close()


def test_eps_model_accepts_scalar_list_and_array_across_sim_full_media():
    freqs_scalar = 2e14
    freqs_list = [2e14, 3e14]
    freqs_array = np.array(freqs_list)

    media = [SIM_FULL.medium, *(structure.medium for structure in SIM_FULL.structures)]

    for medium in media:
        scalar_result = medium.eps_model(freqs_scalar)
        assert np.asarray(scalar_result).shape == ()

        expected = np.array([medium.eps_model(freq) for freq in freqs_array])
        for freqs in (freqs_list, freqs_array):
            result = np.asarray(medium.eps_model(freqs))
            assert result.shape == expected.shape
            if result.dtype != object and expected.dtype != object:
                np.testing.assert_allclose(result, expected)


def test_eps_sigma_freq_none():
    EPS_REAL = 2.0
    eps = MEDIUM.eps_sigma_to_eps_complex(eps_real=EPS_REAL, sigma=1.0, freq=None)
    assert eps == EPS_REAL


def test_tuple_complex_convert():
    assert LZ.tuple_to_complex((1, 2)) == 1 + 2j
    assert LZ.complex_to_tuple(1 + 2j) == (1, 2)


def test_str():
    _ = str(PR)


def test_from_n_less_than_1():
    with pytest.raises(ValidationError):
        td.Sellmeier.from_dispersion(n=0.5, freq=1.0, dn_dwvl=-1)


def test_medium():
    # mediums error with unacceptable values
    with pytest.raises(pd.ValidationError):
        _ = td.Medium(permittivity=0.0)
    with pytest.raises(pd.ValidationError):
        _ = td.Medium(conductivity=-1.0)


def test_empty_heat_spec_dict_stays_invalid_for_direct_validation():
    raw_medium = td.Medium(permittivity=2.0, heat_spec=td.SolidSpec(conductivity=1.0)).model_dump()
    raw_medium["heat_spec"] = {}

    with pytest.raises(
        pd.ValidationError, match="Unable to extract tag using discriminator 'type'"
    ):
        td.Medium.model_validate(raw_medium)


def test_empty_heat_spec_dict_loads_as_none_from_file_without_errors(caplog, tmp_path):
    raw_medium = td.Medium(permittivity=2.0, heat_spec=td.SolidSpec(conductivity=1.0)).model_dump()
    raw_medium["heat_spec"] = {}
    medium_path = tmp_path / "legacy_medium.json"
    medium_path.write_text(json.dumps(raw_medium))

    with caplog.at_level(logging.ERROR):
        medium = td.Medium.from_file(medium_path)

    assert medium.heat_spec is None
    assert not caplog.records


def test_validate_largest_pole_parameters():
    # error for large pole parameters
    with pytest.raises(pd.ValidationError):
        _ = td.PoleResidue(poles=[((-1e50 + 2j), (1 + 3j))])

    with pytest.raises(pd.ValidationError):
        _ = td.PoleResidue(poles=[((-1 + 2j), (1e50 + 3j))])


def test_pole_residue_json_roundtrip():
    pole_residue = td.PoleResidue(
        eps_inf=2.0,
        poles=((-1 + 2j, 3 + 4j), (-2 + 5j, 6 + 7j)),
    )

    payload = json.loads(pole_residue.model_dump_json())
    assert payload["poles"][0][0] == {"real": -1.0, "imag": 2.0}
    assert payload["poles"][1][1] == {"real": 6.0, "imag": 7.0}

    roundtrip = td.PoleResidue.model_validate_json(json.dumps(payload))
    assert roundtrip == pole_residue
    assert isinstance(roundtrip.poles, tuple)
    assert all(isinstance(pole, tuple) for pole in roundtrip.poles)
    assert all(isinstance(value, complex) for pole in roundtrip.poles for value in pole)


def test_medium_conversions():
    n = 4.0
    k = 1.0
    freq = 3.0

    # test medium creation
    medium = td.Medium.from_nk(n, k, freq)

    # test consistency
    eps_z = medium.nk_to_eps_complex(n, k)
    eps, sig = medium.nk_to_eps_sigma(n, k, freq)
    eps_z_ = medium.eps_sigma_to_eps_complex(eps, sig, freq)
    assert np.isclose(eps_z, eps_z_)

    eps_, sig_ = medium.eps_complex_to_eps_sigma(eps_z, freq)
    assert np.isclose(eps_, eps)
    assert np.isclose(sig_, sig)

    n_, k_ = medium.eps_complex_to_nk(eps_z)
    assert np.isclose(n, n_)
    assert np.isclose(k, k_)


def test_lorentz_medium_conversions():
    freq = 3.0

    # lossless, eps_r > 1
    eps_complex = 2 + 0j
    n, k = td.Lorentz.eps_complex_to_nk(eps_complex)
    with AssertLogLevel("WARNING"):
        medium = td.Lorentz.from_nk(n, k, freq)
    eps_model = medium.eps_model(freq)
    assert np.isclose(eps_complex, eps_model)

    # lossless, eps_r < 1
    eps_complex = 0.5 + 0j
    n, k = td.Lorentz.eps_complex_to_nk(eps_complex)
    medium = td.Lorentz.from_nk(n, k, freq)
    eps_model = medium.eps_model(freq)
    assert np.isclose(eps_complex, eps_model)

    # lossy, eps_r < 1
    eps_complex = 0.5 + 0.1j
    n, k = td.Lorentz.eps_complex_to_nk(eps_complex)
    medium = td.Lorentz.from_nk(n, k, freq)
    eps_model = medium.eps_model(freq)
    assert np.isclose(eps_complex, eps_model)

    # lossy, eps_r > 1
    eps_complex = 1.5 + 2j
    n, k = td.Lorentz.eps_complex_to_nk(eps_complex)
    with AssertLogLevel("WARNING"):
        medium = td.Lorentz.from_nk(n, k, freq)
    eps_model = medium.eps_model(freq)
    assert np.isclose(eps_complex, eps_model)


def test_medium_from_nk():
    freq = 3.0

    # lossy, eps_r < 1
    eps_complex = 0.5 + 0.1j
    n, k = td.AbstractMedium.eps_complex_to_nk(eps_complex)
    medium = td.medium_from_nk(n, k, freq)
    eps_model = medium.eps_model(freq)
    assert np.isclose(eps_complex, eps_model)
    assert medium.type == "Lorentz"

    # lossy, eps_r > 1
    eps_complex = 1.5 + 2j
    n, k = td.AbstractMedium.eps_complex_to_nk(eps_complex)
    medium = td.medium_from_nk(n, k, freq)
    eps_model = medium.eps_model(freq)
    assert np.isclose(eps_complex, eps_model)
    assert medium.type == "Medium"


def test_PEC():
    _ = td.Structure(geometry=td.Box(size=(1, 1, 1)), medium=td.PEC)


def test_PMC():
    _ = td.Structure(geometry=td.Box(size=(1, 1, 1)), medium=td.PMC)


def test_lossy_metal():
    # frequency_range shouldn't be None
    with pytest.raises(pd.ValidationError):
        _ = td.LossyMetalMedium(conductivity=1)
    # frequency_range shouldn't contain non-positive values
    with pytest.raises(pd.ValidationError):
        _ = td.LossyMetalMedium(conductivity=1, frequency_range=(0, 10))
    with pytest.raises(pd.ValidationError):
        _ = td.LossyMetalMedium(conductivity=1, frequency_range=(-10, 10))

    # frequency_range should be finite
    with pytest.raises(pd.ValidationError):
        _ = td.LossyMetalMedium(conductivity=1, frequency_range=(10, np.inf))
    with pytest.raises(pd.ValidationError):
        _ = td.LossyMetalMedium(conductivity=1, frequency_range=(-np.inf, 10))

    # allow_gain cannot be true
    with pytest.raises(pd.ValidationError):
        _ = td.LossyMetalMedium(allow_gain=True, conductivity=1, frequency_range=(10, 20))

    # conductivity cannot be negative
    with pytest.raises(pd.ValidationError):
        _ = td.LossyMetalMedium(conductivity=-1, frequency_range=(10, 20))

    # conductivity cannot be 0
    with pytest.raises(pd.ValidationError):
        _ = td.LossyMetalMedium(conductivity=0, frequency_range=(10, 20))

    # default fitting
    mat = td.LossyMetalMedium(conductivity=1.0, frequency_range=(1e14, 4e14))
    model = mat.scaled_surface_impedance_model
    num_poles = mat.num_poles

    # thickness
    mat = td.LossyMetalMedium(conductivity=1.0, frequency_range=(1e14, 4e14), thickness=0.1)
    model = mat.scaled_surface_impedance_model
    num_poles = mat.num_poles

    # a penetrable lossy metal is a regular conductive medium, so it is a valid anisotropic
    # component whose conductivity enters the diagonal permittivity tensor
    lossy_inside = td.LossyMetalMedium(
        conductivity=1.0, frequency_range=(1e14, 4e14), penetrable=True
    )
    aniso = td.AnisotropicMedium(xx=lossy_inside, yy=td.Medium(), zz=td.Medium())
    assert aniso.eps_diagonal(2e14)[0].imag != 0

    # a non-penetrable lossy metal has no anisotropic-component formulation, so it is rejected
    surface_metal = td.LossyMetalMedium(conductivity=1.0, frequency_range=(1e14, 4e14))
    with pytest.raises(pd.ValidationError):
        _ = td.AnisotropicMedium(xx=surface_metal, yy=td.Medium(), zz=td.Medium())


def test_medium2d_volumetric_equivalent_lossy_metal_normal():
    """A surface-impedance lossy metal neighboring a ``Medium2D`` (e.g. a 2D lumped element
    sitting flush on a conductor face) has no volumetric tensor formulation, so it enters the
    volumetric equivalent in its penetrable form (a regular conductive medium) instead of
    tripping the ``AnisotropicMedium`` component validator."""
    sheet = td.Medium2D(ss=td.Medium(conductivity=1.0), tt=td.Medium(conductivity=1.0))
    surface_metal = td.LossyMetalMedium(conductivity=35.0, frequency_range=(10e9, 40e9))
    penetrable_metal = surface_metal.updated_copy(penetrable=True)
    dielectric = td.Medium(permittivity=3.9)

    def assert_no_surface_lossy_metal(aniso):
        for comp in (aniso.xx, aniso.yy, aniso.zz):
            if isinstance(comp, td.LossyMetalMedium):
                assert comp.penetrable

    # a lossy metal on the + side backs the normal (zz) component: its penetrable form,
    # regardless of the `penetrable` setting of the neighbor itself
    for metal in (surface_metal, penetrable_metal):
        aniso = sheet.volumetric_equivalent(
            axis=2, adjacent_media=(dielectric, metal), adjacent_dls=(0.1, 0.1)
        )
        assert type(aniso.zz) is td.LossyMetalMedium
        assert aniso.zz == metal.updated_copy(penetrable=True)
        assert_no_surface_lossy_metal(aniso)

    # a lossy metal encountered on either or both sides never yields a non-penetrable component
    for media in [(surface_metal, dielectric), (surface_metal, surface_metal)]:
        assert_no_surface_lossy_metal(
            sheet.volumetric_equivalent(axis=2, adjacent_media=media, adjacent_dls=(0.1, 0.1))
        )


def test_lossy_metal_surface_roughness():
    mat_orig = td.LossyMetalMedium(
        conductivity=41.0,
        frequency_range=(1e9, 10e9),
    )
    skin_depth = 1.1
    frequency = 1e12

    # Hammerstad
    rq = 0.5
    mat = mat_orig.updated_copy(roughness=td.HammerstadSurfaceRoughness(rq=rq))
    # verify power loss correction factor compared to analytical formula
    complex_factor = mat.roughness.roughness_correction_factor(frequency, skin_depth)
    power_factor = 1 + 2 / np.pi * np.arctan(1.4 * (rq / skin_depth) ** 2)
    assert np.isclose(np.real(complex_factor) + np.imag(complex_factor), power_factor)
    _, residue = mat._fitting_result
    assert residue < 1e-2  # small enough residue indicating causality

    # Huray
    mat = mat_orig.updated_copy(roughness=td.HuraySurfaceRoughness.from_cannonball_huray(rq))
    # verify power loss correction factor compared to analytical formula
    complex_factor = mat.roughness.roughness_correction_factor(frequency, skin_depth)
    power_factor = 1 + 7 / 3 * np.pi / (1 + skin_depth / rq + (skin_depth / rq) ** 2 / 2)
    assert np.isclose(np.real(complex_factor) + np.imag(complex_factor), power_factor)
    _, residue = mat._fitting_result
    assert residue < 1e-2  # small enough residue indicates causality


def test_medium_dispersion():
    # construct media
    m_PR = td.PoleResidue(eps_inf=1.0, poles=[((-1 + 2j), (1 + 3j)), ((-2 + 4j), (1 + 5j))])
    m_SM = td.Sellmeier(coeffs=[(2, 3), (2, 4)])
    m_LZ = td.Lorentz(eps_inf=1.0, coeffs=[(1, 3, 2), (2, 4, 1)])
    m_LZ2 = td.Lorentz(eps_inf=1.0, coeffs=[(1, 2, 3), (2, 1, 4)])
    m_DR = td.Drude(eps_inf=1.0, coeffs=[(1, 3), (2, 4)])
    m_DB = td.Debye(eps_inf=1.0, coeffs=[(1, 3), (2, 4)])

    with pytest.raises(pd.ValidationError):
        _ = td.Sellmeier(coeffs=[(2, 0), (2, 4)])

    with pytest.raises(pd.ValidationError):
        _ = td.Drude(eps_inf=1.0, coeffs=[(1, 0), (2, 4)])

    with pytest.raises(pd.ValidationError):
        _ = td.Debye(eps_inf=1.0, coeffs=[(1, 0), (2, 4)])

    freqs = np.linspace(0.01, 1, 1001)
    for medium in [m_PR, m_SM, m_LZ, m_LZ2, m_DR, m_DB]:
        eps_c = medium.eps_model(freqs)

    for medium in [m_SM, m_LZ, m_LZ2, m_DR, m_DB]:
        eps_c = medium.eps_model(freqs)
        assert np.all(eps_c.imag >= 0)

    # test eps_model for int arguments
    m_SM.eps_model(np.array([1, 2]))

    # test LO-TO form
    poles = [(1, 0.1, 2, 5), (3, 0.4, 1, 0.4)]
    m_LO_TO = td.PoleResidue.from_lo_to(poles=poles, eps_inf=2)
    assert np.allclose(
        m_LO_TO.eps_model(freqs),
        td.PoleResidue.lo_to_eps_model(poles=poles, eps_inf=2, frequency=freqs),
    )


def test_medium_dispersion_conversion():
    m_PR = td.PoleResidue(eps_inf=1.0, poles=[((-1 + 2j), (1 + 3j)), ((-2 + 4j), (1 + 5j))])
    m_SM = td.Sellmeier(coeffs=[(2, 3), (2, 4)])
    m_SM_small_C = td.Sellmeier(coeffs=[(2, 3), (2, 1e-20)])
    m_LZ = td.Lorentz(eps_inf=1.0, coeffs=[(1, 3, 2), (2, 4, 1)])
    m_LZ2 = td.Lorentz(eps_inf=1.0, coeffs=[(1, 2, 3), (2, 1, 4)])
    m_DR = td.Drude(eps_inf=1.0, coeffs=[(1, 3), (2, 4)])
    m_DB = td.Debye(eps_inf=1.0, coeffs=[(1, 3), (2, 4)])
    m_DB_small_tau = td.Debye(eps_inf=1.0, coeffs=[(1, 3), (2, 1e-50)])

    freqs = np.linspace(0.01, 1, 1001)
    for medium in [m_PR, m_SM, m_SM_small_C, m_DB, m_DB_small_tau, m_LZ, m_DR, m_LZ2]:  # , m_DB]:
        eps_model = medium.eps_model(freqs)
        eps_pr = medium.pole_residue.eps_model(freqs)
        np.testing.assert_allclose(eps_model, eps_pr)


def test_medium_dispersion_create():
    m_PR = td.PoleResidue(eps_inf=1.0, poles=[((-1 + 2j), (1 + 3j)), ((-2 + 4j), (1 + 5j))])
    m_SM = td.Sellmeier(coeffs=[(2, 3), (2, 4)])
    m_LZ = td.Lorentz(eps_inf=1.0, coeffs=[(1, 3, 2), (2, 4, 1)])
    m_LZ2 = td.Lorentz(eps_inf=1.0, coeffs=[(1, 2, 3), (2, 1, 4)])
    m_DR = td.Drude(eps_inf=1.0, coeffs=[(1, 3), (2, 4)])
    m_DB = td.Debye(eps_inf=1.0, coeffs=[(1, 3), (2, 4)])

    for medium in [m_PR, m_SM, m_DB, m_LZ, m_DR, m_LZ2]:
        _ = td.Structure(geometry=td.Box(size=(1, 1, 1)), medium=medium)


def test_sellmeier_from_dispersion():
    n = 3.5
    wvl = 0.5
    freq = td.C_0 / wvl
    dn_dwvl = -0.1
    with pytest.raises(ValidationError):
        # Check that postivie dispersion raises an error
        medium = td.Sellmeier.from_dispersion(n=n, freq=freq, dn_dwvl=-dn_dwvl)

    # Check that medium properties are as epected
    medium = td.Sellmeier.from_dispersion(n=n, freq=freq, dn_dwvl=dn_dwvl)
    epses = [medium.eps_model(f) for f in [0.99 * freq, freq, 1.01 * freq]]
    ns = np.sqrt(epses)
    dn_df = (ns[2] - ns[0]) / 0.02 / freq

    assert np.allclose(ns[1], n)
    assert np.allclose(-dn_df * td.C_0 / wvl**2, dn_dwvl)


def eps_compare(medium: td.Medium, expected: dict, tol: float = 1e-5):
    for freq, val in expected.items():
        assert np.abs(medium.eps_model(freq) - val) < tol


def test_pole_residue_loss_upper_bound():
    """Test if `loss_upper_bound` in PoleResidue behaves correctly."""
    mat_lorentz = td.Lorentz(coeffs=((15, 1e14, 0.3e14), (10, 1.5e14, 0.2e14)))
    mat_sellmeier = td.Sellmeier(coeffs=((2, 4),))
    mat_combined = td.PoleResidue(
        poles=(mat_lorentz.pole_residue.poles + mat_sellmeier.pole_residue.poles)
    )
    # compute overall Im[eps] upper bound when `frequency_range = None`
    assert mat_combined.loss_upper_bound > 40
    # Im[eps] upper bound within the frequency range
    mat_new = mat_combined.copy(update={"frequency_range": (6e13, 1.2e14)})
    assert mat_new.loss_upper_bound > 30 and mat_new.loss_upper_bound < 35

    for material_name in PALIK_NOLOSS_MATERIALS:
        assert td.material_library[material_name][PALIK_NOLOSS_VARIANT].loss_upper_bound == 0


def test_palik_noloss_materials_have_zero_loss():
    """Make sure Palik_NoLoss variants are truly lossless across valid ranges."""
    for material_name in PALIK_NOLOSS_MATERIALS:
        medium = td.material_library[material_name][PALIK_NOLOSS_VARIANT]
        freqs = np.geomspace(*medium.frequency_range, num=11)
        _, index_k = medium.nk_model(freqs)

        assert np.all(medium.eps_model(freqs).imag == 0)
        assert np.all(index_k == 0)


def test_palik_noloss_stays_close_to_lowloss_real_response():
    """Make sure the raw-data no-loss refit remains close to the previous fit."""
    for material_name in PALIK_NOLOSS_MATERIALS:
        noloss_medium = td.material_library[material_name][PALIK_NOLOSS_VARIANT]
        lowloss_medium = td.material_library[material_name][PALIK_LOWLOSS_VARIANT]
        freqs = np.geomspace(*noloss_medium.frequency_range, num=101)
        rel_error = np.abs(
            (noloss_medium.eps_model(freqs).real - lowloss_medium.eps_model(freqs).real)
            / lowloss_medium.eps_model(freqs).real
        )

        assert np.max(rel_error) < 0.07


def test_palik_lowloss_materials_preserve_fitted_loss():
    """Make sure the old fitted Palik variants remain available as Palik_LowLoss."""
    for material_name in PALIK_LOWLOSS_MATERIALS:
        medium = td.material_library[material_name][PALIK_LOWLOSS_VARIANT]
        freqs = np.geomspace(*medium.frequency_range, num=11)
        eps_imag = medium.eps_model(freqs).imag
        _, index_k = medium.nk_model(freqs)

        assert 0 < medium.loss_upper_bound < 1e-4
        assert np.any(np.abs(eps_imag) > 0)
        assert np.any(np.abs(index_k) > 0)


def test_palik_lossless_alias_points_to_lowloss():
    """Make sure the ambiguous old Palik_Lossless name remains a low-loss alias."""
    for material_name in PALIK_LOWLOSS_MATERIALS:
        material = td.material_library[material_name]
        lowloss_medium = material[PALIK_LOWLOSS_VARIANT]

        with AssertLogLevel("WARNING", contains_str="Palik_Lossless"):
            legacy_medium = material["Palik_Lossless"]
        with AssertLogLevel("WARNING", contains_str="Palik_Lossless"):
            legacy_variant = material.variants["Palik_Lossless"]
        with AssertLogLevel("WARNING", contains_str="Palik_Lossless"):
            legacy_variant_from_get = material.variants.get("Palik_Lossless")

        assert legacy_medium == lowloss_medium
        assert legacy_variant.medium == lowloss_medium
        assert legacy_variant_from_get == material.variants[PALIK_LOWLOSS_VARIANT]
        assert "Palik_Lossless" in material.variants
        assert "Palik_Lossless" not in material.variants.keys()


def test_palik_lossless_alias_get_returns_default_when_target_missing():
    """Make sure the deprecated alias mapping preserves the dict.get contract."""
    variants = td.material_library["SiO2"].variants.copy()
    default_variant = VariantItem(medium=td.PoleResidue(name="fallback"))
    variants.pop(PALIK_LOWLOSS_VARIANT)

    with AssertLogLevel("WARNING", contains_str="Palik_Lossless"):
        legacy_variant = variants.get("Palik_Lossless", default_variant)

    assert legacy_variant == default_variant


def test_palik_lossless_alias_survives_round_trip():
    """Make sure Palik_Lossless remains available after normal model round-trips."""
    for material_name in PALIK_LOWLOSS_MATERIALS:
        material = td.material_library[material_name]

        round_tripped_materials = (
            material.copy(),
            material.copy(validate=False),
            material.updated_copy(),
            material.updated_copy(validate=False),
            type(material).model_validate(material.model_dump()),
        )

        for round_tripped_material in round_tripped_materials:
            with AssertLogLevel("WARNING", contains_str="Palik_Lossless"):
                legacy_medium = round_tripped_material["Palik_Lossless"]
            with AssertLogLevel("WARNING", contains_str="Palik_Lossless"):
                legacy_variant = round_tripped_material.variants["Palik_Lossless"]

            assert legacy_medium == round_tripped_material[PALIK_LOWLOSS_VARIANT]
            assert legacy_variant == round_tripped_material.variants[PALIK_LOWLOSS_VARIANT]


def test_palik_lossless_user_variant_is_not_rewritten():
    """Make sure the built-in alias does not rewrite user-defined material variants."""
    lossless_variant = VariantItem(medium=td.PoleResidue(name="custom_lossless"))
    lowloss_variant = VariantItem(medium=td.PoleResidue(name="custom_lowloss"))
    material = MaterialItem(
        name="Custom Material",
        variants={
            "Palik_Lossless": lossless_variant,
            "Palik_LowLoss": lowloss_variant,
        },
        default="Palik_Lossless",
    )

    with AssertLogLevel(None):
        legacy_medium = material["Palik_Lossless"]

    assert legacy_medium == lossless_variant.medium


def test_palik_lossless_alias_does_not_apply_to_similarly_named_user_material():
    """Make sure built-in Palik aliases are not inferred from material names alone."""
    material = MaterialItem(
        name="Silicon Dioxide",
        variants={
            PALIK_NOLOSS_VARIANT: VariantItem(medium=td.PoleResidue(name="custom_noloss")),
            PALIK_LOWLOSS_VARIANT: VariantItem(medium=td.PoleResidue(name="custom_lowloss")),
        },
        default=PALIK_LOWLOSS_VARIANT,
    )

    assert "Palik_Lossless" not in material.variants
    with pytest.raises(KeyError):
        _ = material["Palik_Lossless"]


def test_epsilon_eval():
    """Compare epsilon evaluated from a dispersive various models to expected."""

    # Dispersive silver model
    poles_silver = [
        (a / td.HBAR, c / td.HBAR)
        for (a, c) in [
            ((-2.502e-2 - 8.626e-3j), (5.987e-1 + 4.195e3j)),
            ((-2.021e-1 - 9.407e-1j), (-2.211e-1 + 2.680e-1j)),
            ((-1.467e1 - 1.338e0j), (-4.240e0 + 7.324e2j)),
            ((-2.997e-1 - 4.034e0j), (6.391e-1 - 7.186e-2j)),
            ((-1.896e0 - 4.808e0j), (1.806e0 + 4.563e0j)),
            ((-9.396e0 - 6.477e0j), (1.443e0 - 8.219e1j)),
        ]
    ]

    material = td.PoleResidue(poles=poles_silver)
    expected = {
        2e14: (-102.18389652032306 + 9.22771912188222j),
        5e14: (-13.517709933590542 + 0.9384819052893092j),
    }
    eps_compare(material, expected)

    # Constant and eps, zero sigma
    material = td.Medium(permittivity=1.5**2)
    expected = {
        2e14: 2.25,
        5e14: 2.25,
    }
    eps_compare(material, expected)

    # Constant eps and sigma
    material = td.Medium(permittivity=1.5**2, conductivity=0.1)
    expected = {
        2e14: 2.25 + 8.987552009401353j,
        5e14: 2.25 + 3.5950208037605416j,
    }
    eps_compare(material, expected)

    # Constant n and k at a given frequency
    material = td.Medium.from_nk(n=1.5, k=0.1, freq=td.C_0 / 0.8)
    expected = {
        2e14: 2.24 + 0.5621108598392753j,
        5e14: 2.24 + 0.22484434393571015j,
    }
    eps_compare(material, expected)

    # Anisotropic material
    eps = (1.5, 2.0, 2.3)
    sig = (0.01, 0.03, 0.015)
    mediums = [td.Medium(permittivity=eps[i], conductivity=sig[i]) for i in range(3)]
    material = td.AnisotropicMedium(xx=mediums[0], yy=mediums[1], zz=mediums[2])

    eps_diag_2 = material.eps_diagonal(2e14)
    eps_diag_5 = material.eps_diagonal(5e14)
    assert np.all(np.array(eps_diag_2) == np.array([medium.eps_model(2e14) for medium in mediums]))

    expected = {2e14: np.mean(eps_diag_2), 5e14: np.mean(eps_diag_5)}
    eps_compare(material, expected)

    # Anisotropic material with dispersion
    eps = 1.5
    sig = 0.01
    mediums = [
        td.Medium(permittivity=eps, conductivity=sig),
        td.PoleResidue(poles=poles_silver),
        td.PoleResidue(poles=poles_silver),
    ]
    material = td.AnisotropicMedium(xx=mediums[0], yy=mediums[1], zz=mediums[2])

    eps_diag_2 = material.eps_diagonal(2e14)
    eps_diag_5 = material.eps_diagonal(5e14)
    assert np.all(np.array(eps_diag_2) == np.array([medium.eps_model(2e14) for medium in mediums]))

    expected = {2e14: np.mean(eps_diag_2), 5e14: np.mean(eps_diag_5)}
    eps_compare(material, expected)


def test_n_cfl():
    """Test ``n_cfl`` is computed correctly."""
    # dispersiveless medium
    assert MEDIUM.n_cfl == 1
    material = td.Medium(permittivity=4, conductivity=2)
    assert material.n_cfl == 2
    # PEC
    assert PEC.n_cfl == 1
    # PMC
    assert PMC.n_cfl == 1
    # anisotropic
    material = td.AnisotropicMedium(xx=MEDIUM, yy=td.Medium(permittivity=4), zz=MEDIUM)
    assert material.n_cfl == 1
    # dispersive
    material = td.PoleResidue(eps_inf=0.16, poles=[(-1 + 1j, 2 + 2j)])
    assert material.n_cfl == 0.4
    assert SM.n_cfl == 1
    material = td.Lorentz(eps_inf=0.04, coeffs=[(1, 2, 3)])
    assert material.n_cfl == 0.2
    material = td.Drude(eps_inf=4, coeffs=[(1, 2)])
    assert material.n_cfl == 2
    material = td.Debye(eps_inf=4, coeffs=[(1, 2)])
    assert material.n_cfl == 2


def test_gain_medium():
    """Test passive and gain medium validations."""
    # non-dispersive
    with pytest.raises(pd.ValidationError):
        _ = td.Medium(conductivity=-0.1)
    with pytest.raises(pd.ValidationError):
        _ = td.Medium(conductivity=-1.0, allow_gain=False)
    _ = td.Medium(conductivity=-1.0, allow_gain=True)

    # pole residue, causality
    with pytest.raises(pd.ValidationError):
        _ = td.PoleResidue(eps_inf=0.16, poles=[(1 + 1j, 2 + 2j)])

    # Sellmeier
    with pytest.raises(pd.ValidationError):
        _ = td.Sellmeier(coeffs=((-1, 1),))
    mS = td.Sellmeier(coeffs=((-1, 1),), allow_gain=True)

    # Lorentz
    # causality, negative gamma
    with pytest.raises(pd.ValidationError):
        _ = td.Lorentz(eps_inf=0.04, coeffs=[(1, 2, -3)])
    # gain, negative Delta epsilon
    with pytest.raises(pd.ValidationError):
        _ = td.Lorentz(eps_inf=0.04, coeffs=[(-1, 2, 3)])
    mL = td.Lorentz(eps_inf=0.04, coeffs=[(-1, 2, 3)], allow_gain=True)
    assert mL.pole_residue.allow_gain

    # f_i can take whatever sign
    _ = td.Lorentz(eps_inf=0.04, coeffs=[(1, -2, 3)])

    # Drude, only causality constraint
    with pytest.raises(pd.ValidationError):
        _ = td.Drude(eps_inf=0.04, coeffs=[(1, -2)])

    # anisotropic medium, warn allow_gain is ignored

    with AssertLogLevel("WARNING"):
        _ = td.AnisotropicMedium(xx=td.Medium(), yy=mL, zz=mS, allow_gain=True)

    with AssertLogLevel("WARNING"):
        _ = td.AnisotropicMedium(xx=td.Medium(), yy=mL, zz=mS, allow_gain=False)


def test_medium2d():
    sigma = 0.45
    thickness = 0.01
    cond_med = td.Medium(conductivity=sigma)
    medium = td.Medium2D.from_medium(cond_med, thickness=thickness)

    _ = medium.plot_sigma(freqs=[2e14, 3e14], ax=AX)
    plt.close()
    assert np.isclose(medium.ss.to_medium().conductivity, sigma * thickness, rtol=RTOL)
    aniso_medium = td.AnisotropicMedium(xx=td.Medium(permittivity=2), yy=cond_med, zz=td.Medium())
    medium = td.Medium2D.from_anisotropic_medium(aniso_medium, axis=2, thickness=thickness)
    medium3d = medium.to_anisotropic_medium(axis=2, thickness=1.5 * thickness)
    assert np.isclose(medium3d.xx.to_medium().permittivity, 1 + (2 - 1) / 1.5, rtol=RTOL)
    assert np.isclose(medium3d.yy.to_medium().conductivity, sigma / 1.5, rtol=RTOL)
    assert np.isclose(medium3d.zz.permittivity, 1, rtol=RTOL)
    assert np.isclose(
        medium.to_medium(thickness=1.5 * thickness).conductivity, sigma / 3, rtol=RTOL
    )
    assert np.isclose(
        medium.to_pole_residue(thickness=1.5 * thickness).to_medium().conductivity,
        sigma / 3,
        rtol=RTOL,
    )

    td.Structure(medium=medium3d, geometry=td.Box(size=(1, 1, 1)))

    # this should also not warn, since it could be used for override structure
    td.Structure(medium=medium3d, geometry=td.Box(size=(1, 0, 1)))

    with AssertLogLevel("WARNING"):
        _ = medium.plot(freqs=[2e14, 3e14], ax=AX)
    plt.close()

    with pytest.raises(pd.ValidationError):
        _ = td.Medium2D(ss=td.PECMedium(), tt=td.Medium())


def test_rotation():
    # check that transpose is inverse
    axis = np.random.random(3)
    rot = td.RotationAroundAxis(axis=tuple(axis), angle=1.23)

    R = rot.matrix

    assert np.all(np.abs(np.matmul(np.transpose(R), R) - np.eye(3)) < 1.0e-15)
    assert np.all(np.abs(np.matmul(R, np.transpose(R)) - np.eye(3)) < 1.0e-15)

    # check that rotation around x, y, z by 90 degrees works as expected
    tan_dims = [[1, 2], [2, 0], [0, 1]]

    for dim in range(3):
        axis = [0, 0, 0]
        axis[dim] = 1
        rot = td.RotationAroundAxis(axis=axis, angle=np.pi / 2)

        v0 = np.random.random(3)
        vr = rot.rotate_vector(v0)
        assert np.abs(v0[dim] - vr[dim]) < 1.0e-15
        assert np.abs(v0[tan_dims[dim][0]] - vr[tan_dims[dim][1]]) < 1.0e-15
        assert np.abs(v0[tan_dims[dim][1]] + vr[tan_dims[dim][0]]) < 1.0e-15


def test_fully_anisotropic_media():
    perm_diag = [[1, 0, 0], [0, 2, 0], [0, 0, 3]]
    cond_diag = [[4, 0, 0], [0, 5, 0], [0, 0, 6]]

    rot = td.RotationAroundAxis(axis=(1, 2, 3), angle=1.23)
    rot2 = td.RotationAroundAxis(axis=(3, 2, 1), angle=1.23)

    perm = rot.rotate_tensor(perm_diag)
    cond = rot.rotate_tensor(cond_diag)
    cond2 = rot2.rotate_tensor(cond_diag)

    _ = td.FullyAnisotropicMedium(permittivity=perm, conductivity=cond)

    # check that tensors are provided
    with pytest.raises(pd.ValidationError):
        td.FullyAnisotropicMedium(permittivity=2)
    with pytest.raises(pd.ValidationError):
        td.FullyAnisotropicMedium(permittivity=[3, 4, 2])

    # check that permittivity >= 1 and conductivity >= 0
    with pytest.raises(pd.ValidationError):
        td.FullyAnisotropicMedium(permittivity=[[3, 0, 0], [0, 0.5, 0], [0, 0, 1]])
    with pytest.raises(pd.ValidationError):
        td.FullyAnisotropicMedium(conductivity=[[-3, 0, 0], [0, 0.5, 0], [0, 0, 1]])
    td.FullyAnisotropicMedium(conductivity=[[-3, 0, 0], [0, 0.5, 0], [0, 0, 1]], allow_gain=True)

    # check that permittivity needs to be symmetric
    with pytest.raises(pd.ValidationError):
        td.FullyAnisotropicMedium(permittivity=[[3, 0.1, 0], [0.2, 2, 0], [0, 0, 1]])

    # check that differently oriented permittivity and conductivity are not accepted
    with pytest.raises(pd.ValidationError):
        td.FullyAnisotropicMedium(permittivity=perm, conductivity=cond2)

    # check creation from diagonal medium
    m = td.FullyAnisotropicMedium.from_diagonal(
        xx=td.Medium(permittivity=perm_diag[0][0], conductivity=cond_diag[0][0]),
        yy=td.Medium(permittivity=perm_diag[1][1], conductivity=cond_diag[1][1]),
        zz=td.Medium(permittivity=perm_diag[2][2], conductivity=cond_diag[2][2]),
        rotation=rot,
    )

    # check eps_model can be called with an array of frequencies
    m.eps_model(np.linspace(1e12, 2e12, 10))

    assert np.allclose(m.permittivity, perm)
    assert np.allclose(m.conductivity, cond)

    perm_d, cond_d, _ = m.eps_sigma_diag

    assert all(np.isin(np.round(perm_d), np.round(np.diag(perm_diag))))
    assert all(np.isin(np.round(cond_d), np.round(np.diag(cond_diag))))

    with pytest.raises(ValidationError):
        _ = td.FullyAnisotropicMedium.from_diagonal(
            xx=td.Medium(
                permittivity=2,
                nonlinear_spec=td.NonlinearSpec(
                    models=[
                        td.NonlinearSusceptibility(chi3=2),
                        td.TwoPhotonAbsorption(beta=1.3),
                        td.KerrNonlinearity(n2=1.3),
                    ]
                ),
            ),
            yy=td.Medium(permittivity=4),
            zz=td.Medium(permittivity=1),
            rotation=td.RotationAroundAxis(axis=2, angle=np.pi / 4),
        )

    with pytest.raises(ValidationError):
        _ = td.FullyAnisotropicMedium.from_diagonal(
            xx=td.Medium(permittivity=2),
            yy=td.Medium(
                permittivity=4,
                modulation_spec=td.ModulationSpec(
                    permittivity=td.SpaceTimeModulation(
                        time_modulation=td.ContinuousWaveTimeModulation(freq0=1e12, amplitude=0.02)
                    )
                ),
            ),
            zz=td.Medium(permittivity=1),
            rotation=td.RotationAroundAxis(axis=2, angle=np.pi / 4),
        )


def test_nonlinear_medium():
    med = td.Medium(
        nonlinear_spec=td.NonlinearSpec(
            models=[
                td.NonlinearSusceptibility(chi3=1.5),
                td.TwoPhotonAbsorption(beta=1, sigma=1, tau=1, e_e=1, e_h=0.8, c_e=1, c_h=1),
                td.KerrNonlinearity(n2=1),
            ],
            num_iters=20,
        )
    )

    assert med._nonlinear_num_iters == 20
    assert td.Medium()._nonlinear_num_iters == 0
    assert (
        td.Medium(
            nonlinear_spec=td.NonlinearSpec(
                models=[td.NonlinearSusceptibility(chi3=1.5)], num_iters=1
            )
        )._nonlinear_num_iters
        == 1
    )
    assert (
        td.Medium(
            nonlinear_spec=td.NonlinearSpec(
                models=[td.NonlinearSusceptibility(chi3=1.5)], num_iters=2
            )
        )._nonlinear_num_iters
        == 2
    )
    assert td.Medium()._nonlinear_models == []
    assert td.Medium(nonlinear_spec=td.NonlinearSpec())._nonlinear_models == []

    # warn about deprecated api
    with AssertLogLevel("WARNING"):
        med = td.Medium(nonlinear_spec=td.NonlinearSusceptibility(chi3=1.5))

    # don't use deprecated numiters
    with pytest.raises(pd.ValidationError):
        med = td.Medium(
            nonlinear_spec=td.NonlinearSpec(models=[td.NonlinearSusceptibility(chi3=1, numiters=2)])
        )

    # dispersive support
    med = td.PoleResidue(
        poles=[(-1, 1)],
        nonlinear_spec=td.NonlinearSpec(models=[td.NonlinearSusceptibility(chi3=1.5)]),
    )

    # unsupported material types
    with pytest.raises(pd.ValidationError):
        med = td.AnisotropicMedium(
            xx=med,
            yy=med,
            zz=med,
            nonlinear_spec=td.NonlinearSpec(models=[td.NonlinearSusceptibility(chi3=1.5)]),
        )

    # numiters too large
    with AssertLogLevel("ERROR", contains_str="numiters"):
        with pytest.raises(pd.ValidationError):
            med = td.Medium(nonlinear_spec=td.NonlinearSusceptibility(chi3=1.5, numiters=200))
    with pytest.raises(pd.ValidationError):
        med = td.Medium(
            nonlinear_spec=td.NonlinearSpec(
                num_iters=200, models=[td.NonlinearSusceptibility(chi3=1.5)]
            )
        )

    # duplicate models
    with pytest.raises(pd.ValidationError):
        med = td.Medium(
            nonlinear_spec=td.NonlinearSpec(
                models=[
                    td.NonlinearSusceptibility(chi3=1.5),
                    td.NonlinearSusceptibility(chi3=1),
                ]
            )
        )

    # active materials
    with pytest.raises(pd.ValidationError):
        med = td.Medium(
            nonlinear_spec=td.NonlinearSpec(models=[td.TwoPhotonAbsorption(beta=-1, n0=1, freq0=1)])
        )

    # automatic detection of n0 and freq0
    n0 = 2
    freq0 = td.C_0 / 1
    nonlinear_spec = td.NonlinearSpec(models=[td.KerrNonlinearity(n2=1)])
    medium = td.Sellmeier.from_dispersion(n=n0, freq=freq0, dn_dwvl=-0.2).updated_copy(
        nonlinear_spec=nonlinear_spec
    )
    source_time = td.GaussianPulse(freq0=freq0, fwidth=freq0 / 10)
    source = td.PointDipole(center=(0, 0, 0), source_time=source_time, polarization="Ex")
    monitor = td.FieldMonitor(size=(td.inf, td.inf, 0), freqs=[freq0], name="field")
    structure = td.Structure(geometry=td.Box(size=(5, 5, 5)), medium=medium)
    sim = td.Simulation(
        size=(10, 10, 10),
        run_time=1e-12,
        grid_spec=td.GridSpec.uniform(dl=0.1),
        sources=[source],
        monitors=[monitor],
        structures=[structure],
    )

    # subsection with nonlinear materials preserves sources
    sim2 = sim.updated_copy(center=(-4, -4, -4), path="sources/0")
    sim2 = sim2.updated_copy(
        models=(td.TwoPhotonAbsorption(beta=1),), path="structures/0/medium/nonlinear_spec"
    )
    sim2 = sim2.subsection(region=td.Box(center=(0, 0, 0), size=(1, 1, 0)))

    nonlinear_spec = td.NonlinearSpec(models=[td.KerrNonlinearity(n2=1, n0=1)])
    structure = structure.updated_copy(medium=medium.updated_copy(nonlinear_spec=nonlinear_spec))
    sim = sim.updated_copy(structures=(structure,))

    nonlinear_spec = td.NonlinearSpec(models=[td.TwoPhotonAbsorption(beta=1, n0=1)])
    structure = structure.updated_copy(medium=medium.updated_copy(nonlinear_spec=nonlinear_spec))
    sim = sim.updated_copy(structures=[structure])
    nonlinear_spec = td.NonlinearSpec(models=[td.TwoPhotonAbsorption(beta=1, n0=1, freq0=1)])
    structure = structure.updated_copy(medium=medium.updated_copy(nonlinear_spec=nonlinear_spec))
    sim = sim.updated_copy(structures=(structure,))

    # active materials with automatic detection of n0
    nonlinear_spec_active = td.NonlinearSpec(models=[td.TwoPhotonAbsorption(beta=-1)])
    with pytest.raises(pd.ValidationError):
        medium_active = medium.updated_copy(nonlinear_spec=nonlinear_spec_active)

    # inconsistent n0
    with pytest.raises(pd.ValidationError):
        _ = td.NonlinearSpec(
            models=[td.KerrNonlinearity(n0=1, n2=1), td.TwoPhotonAbsorption(beta=1, n0=2)]
        )

    # nonlinear or time-modulation on medium2d
    # time-modulated
    FREQ_MODULATE = 1e12
    AMP_TIME = 1.1
    PHASE_TIME = 0
    CW = td.ContinuousWaveTimeModulation(freq0=FREQ_MODULATE, amplitude=AMP_TIME, phase=PHASE_TIME)
    ST = td.SpaceTimeModulation(
        time_modulation=CW,
    )
    MODULATION_SPEC = td.ModulationSpec()
    modulation_spec = MODULATION_SPEC.updated_copy(permittivity=ST)
    modulated = td.Medium(permittivity=2, modulation_spec=modulation_spec)
    with pytest.raises(pd.ValidationError):
        td.Medium2D(ss=medium, tt=medium)
    with pytest.raises(pd.ValidationError):
        td.Medium2D(ss=modulated, tt=modulated)

    grid_spec = td.GridSpec.auto(min_steps_per_wvl=10, wavelength=1)
    sim = sim.updated_copy(grid_spec=grid_spec)
    aux_fields = ("Nfz",)
    with AssertLogLevel(None):
        med = td.Medium(
            nonlinear_spec=td.NonlinearSpec(models=[td.TwoPhotonAbsorption(beta=1, tau=1)])
        )
        monitor = td.AuxFieldTimeMonitor(
            interval=1, size=(0, 0, 0), name="aux_field_time", fields=aux_fields
        )
        sim = sim.updated_copy(medium=med, path="structures/0")
        sim = sim.updated_copy(monitors=(monitor,))

    with AssertLogLevel("WARNING", contains_str="stores field"):
        med = td.Medium(
            nonlinear_spec=td.NonlinearSpec(models=[td.TwoPhotonAbsorption(beta=1, tau=0)])
        )
        _ = sim.updated_copy(medium=med, path="structures/0")

    with AssertLogLevel("WARNING", contains_str="stores field"):
        med = td.Medium(nonlinear_spec=td.NonlinearSpec(models=[td.KerrNonlinearity(n2=1)]))
        _ = sim.updated_copy(medium=med, path="structures/0")


def test_empty_nonlinear_spec_dict_stays_invalid_for_direct_validation():
    assert td.Medium(nonlinear_spec=td.NonlinearSpec()).nonlinear_spec is not None

    medium_dict = td.Medium().model_dump()
    medium_dict["nonlinear_spec"] = {}

    with pytest.raises(
        pd.ValidationError, match="Unable to extract tag using discriminator 'type'"
    ):
        td.Medium.model_validate(medium_dict)


def test_empty_nonlinear_spec_dict_loads_as_none_from_file_without_errors(caplog, tmp_path):
    medium_dict = td.Medium().model_dump()
    medium_dict["nonlinear_spec"] = {}
    medium_path = tmp_path / "legacy_medium_nonlinear.json"
    medium_path.write_text(json.dumps(medium_dict))

    with caplog.at_level(logging.ERROR):
        medium = td.Medium.from_file(medium_path)

    assert medium.nonlinear_spec is None
    assert not caplog.records


def test_legacy_nonlinear_spec_dicts_validate_without_explicit_type():
    with AssertLogLevel("WARNING", contains_str="nonlinear_spec=model"):
        medium = td.Medium(nonlinear_spec={"chi3": 1.0})
    assert isinstance(medium.nonlinear_spec, td.NonlinearSusceptibility)
    assert medium.nonlinear_spec.chi3 == 1.0

    medium = td.Medium.model_validate(
        {
            "permittivity": 2.0,
            "nonlinear_spec": {
                "models": [{"type": "KerrNonlinearity", "n2": 1.0}],
                "num_iters": 3,
            },
        }
    )
    assert isinstance(medium.nonlinear_spec, td.NonlinearSpec)
    assert isinstance(medium.nonlinear_spec.models[0], td.KerrNonlinearity)
    assert medium.nonlinear_spec.num_iters == 3


def test_custom_medium():
    Nx, Ny, Nz, Nf = 4, 3, 1, 1
    X = np.linspace(-1, 1, Nx)
    Y = np.linspace(-1, 1, Ny)
    Z = [0]
    freqs = [2e14]
    n_data = np.ones((Nx, Ny, Nz, Nf))
    n_dataset = td.ScalarFieldDataArray(n_data, coords={"x": X, "y": Y, "z": Z, "f": freqs})

    def create_mediums(n_dataset):
        ## Three equivalent ways of defining custom medium for the lens

        # define custom medium with n/k data
        _ = td.CustomMedium.from_nk(n_dataset, interp_method="nearest")

        # define custom medium with permittivity data
        eps_dataset = td.ScalarFieldDataArray(
            n_dataset**2, coords={"x": X, "y": Y, "z": Z, "f": freqs}
        )
        _ = td.CustomMedium.from_eps_raw(eps_dataset, interp_method="nearest")

        # define each component of permittivity via "PermittivityDataset"
        eps_xyz_dataset = td.PermittivityDataset(
            eps_xx=eps_dataset, eps_yy=eps_dataset, eps_zz=eps_dataset
        )
        _ = td.CustomMedium(eps_dataset=eps_xyz_dataset, interp_method="nearest")

    with AssertLogLevel(None):
        create_mediums(n_dataset=n_dataset)

    with pytest.raises(pd.ValidationError):
        # repeat some entries so data cannot be interpolated
        X2 = [X[0], *list(X)]
        n_data2 = np.vstack((n_data[0, :, :, :].reshape(1, Ny, Nz, Nf), n_data))
        n_dataset2 = td.ScalarFieldDataArray(n_data2, coords={"x": X2, "y": Y, "z": Z, "f": freqs})
        create_mediums(n_dataset=n_dataset2)


def test_medium_from_admittance_coeffs():
    """Test that ``from_admittance_coeffs`` produces same PoleResidue model as
    conversions from Drude and Lorentz models. Also test some special cases."""
    freqs = np.linspace(0.01, 1, 1001)
    twopi = 2 * np.pi
    m_DR = td.Drude(eps_inf=1.0, coeffs=[(1.5, 3)])
    # test from transfer function using Drude model
    f1 = m_DR.coeffs[0][0]
    d1 = m_DR.coeffs[0][1]
    # admittance function in Laplace domain (a/b) modeling Drude differential equation is:
    a = np.array([0, td.EPSILON_0 * (twopi * f1) ** 2, 0])
    b = np.array([0, twopi * d1, 1])

    m_transfer = td.PoleResidue.from_admittance_coeffs(a, b)
    assert np.allclose(
        m_transfer.eps_model(freqs),
        m_DR.eps_model(freqs),
    )

    # test from transfer function using Lorentz model
    m_L = td.Lorentz(eps_inf=1.0, coeffs=[(1.5, 3, 5)])
    deps = m_L.coeffs[0][0]
    f1 = m_L.coeffs[0][1]
    d1 = m_L.coeffs[0][2]
    # admittance function in Laplace domain (a/b) modeling Lorentz differential equation is:
    a = np.array([0, td.EPSILON_0 * (twopi * f1) ** 2 * deps, 0])
    b = np.array([(twopi * f1) ** 2, 2 * twopi * d1, 1])

    m_transfer = td.PoleResidue.from_admittance_coeffs(a, b)
    assert np.allclose(
        m_transfer.eps_model(freqs),
        m_L.eps_model(freqs),
    )

    # Example network Taflove Sec 15.9.6
    L1 = 1e-9
    L2 = 1.5e-9
    C1 = 0.2e-12
    C2 = 0.2e-12
    R1 = 10
    R2 = 250
    R3 = 50
    # Admittance transfer function for circuit
    a = [1, C1 * R1 + C1 * R2 + C2 * R2, C1 * C2 * R1 * R2 + C1 * L2, C1 * C2 * L2 * R2, 0]
    b = [
        R1 + R2 + R3,
        C1 * R1 * R3 + C1 * R2 * R3 + C2 * R1 * R2 + C2 * R2 * R3 + L1 + L2,
        C1 * C2 * R1 * R2 * R3
        + C1 * L1 * R1
        + C1 * L1 * R2
        + C1 * L2 * R3
        + C2 * L1 * R2
        + C2 * L2 * R2,
        C1 * C2 * L1 * R1 * R2 + C1 * C2 * L2 * R2 * R3 + C1 * L1 * L2,
        C1 * C2 * L1 * L2 * R2,
    ]

    # Should be no warnings due to passivity
    # (although numerically there is a small negative part at high frequencies)
    with AssertLogLevel(None):
        m_transfer = td.PoleResidue.from_admittance_coeffs(np.array(a), np.array(b))

    # test corner case of an admittance function representing a pure capacitance
    C = 1e-12  # 1 pF capacitor
    a = np.array([0, C])
    b = np.array([1, 0])
    m_transfer = td.PoleResidue.from_admittance_coeffs(a, b)

    assert len(m_transfer.poles) == 0
    assert np.isclose(1 + C / td.EPSILON_0, m_transfer.eps_inf)

    #### Test validation of inputs
    # Test improper admittance function resulting in a negative direct polynomial part
    a = np.array([0, -C])
    b = np.array([1, 0])
    with pytest.raises(ValidationError):
        _ = td.PoleResidue.from_admittance_coeffs(a, b)

    # Test improper admittance function with numerator order too large
    a = np.array([0, 1, 2])
    b = np.array([1, 0])
    with pytest.raises(ValidationError):
        _ = td.PoleResidue.from_admittance_coeffs(a, b)

    # Test transfer function that will result in a higher order pole
    a = np.array([1])
    b = np.array([0, 2])
    with pytest.raises(ValidationError):
        _ = td.PoleResidue.from_admittance_coeffs(a, b)

    # Test transfer function that will result in a higher order pole, but is not in simplest form
    a = np.array([0, 1])
    b = np.array([0, 2])
    _ = td.PoleResidue.from_admittance_coeffs(a, b)
