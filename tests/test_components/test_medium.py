"""Tests mediums."""
import numpy as np
import pytest
import pydantic
import matplotlib.pylab as plt
import tidy3d as td
from tidy3d.exceptions import ValidationError
from ..utils import assert_log_level, log_capture
from typing import Dict

MEDIUM = td.Medium()
ANIS_MEDIUM = td.AnisotropicMedium(xx=MEDIUM, yy=MEDIUM, zz=MEDIUM)
PEC = td.PECMedium()
PR = td.PoleResidue(poles=[(-1 + 1j, 2 + 2j)])
SM = td.Sellmeier(coeffs=[(1, 2)])
LZ = td.Lorentz(coeffs=[(1, 2, 3)])
DR = td.Drude(coeffs=[(1, 2)])
DB = td.Debye(coeffs=[(1, 2)])
MEDIUMS = [MEDIUM, ANIS_MEDIUM, PEC, PR, SM, LZ, DR, DB]

f, AX = plt.subplots()

RTOL = 0.001


@pytest.mark.parametrize("component", MEDIUMS)
def test_plot(component):
    _ = component.plot(freqs=[2e14, 3e14], ax=AX)


def test_eps_sigma_freq_none():
    EPS_REAL = 2.0
    eps = MEDIUM.eps_sigma_to_eps_complex(eps_real=EPS_REAL, sigma=1.0, freq=None)
    assert eps == EPS_REAL


def test_tuple_complex_convert():
    assert LZ.tuple_to_complex((1, 2)) == 1 + 2j
    assert LZ.complex_to_tuple(1 + 2j) == (1, 2)


def test_str():
    s = str(PR)


def test_from_n_less_than_1():
    with pytest.raises(ValidationError):
        td.Sellmeier.from_dispersion(n=0.5, freq=1.0, dn_dwvl=-1)


def test_medium():

    # mediums error with unacceptable values
    with pytest.raises(pydantic.ValidationError) as e_info:
        m = td.Medium(permittivity=0.0)
    with pytest.raises(pydantic.ValidationError) as e_info:
        m = td.Medium(conductivity=-1.0)


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


def test_PEC():

    struct = td.Structure(geometry=td.Box(size=(1, 1, 1)), medium=td.PEC)


def test_medium_dispersion():

    # construct media
    m_PR = td.PoleResidue(eps_inf=1.0, poles=[((-1 + 2j), (1 + 3j)), ((-2 + 4j), (1 + 5j))])
    m_SM = td.Sellmeier(coeffs=[(2, 3), (2, 4)])
    m_LZ = td.Lorentz(eps_inf=1.0, coeffs=[(1, 3, 2), (2, 4, 1)])
    m_LZ2 = td.Lorentz(eps_inf=1.0, coeffs=[(1, 2, 3), (2, 1, 4)])
    m_DR = td.Drude(eps_inf=1.0, coeffs=[(1, 3), (2, 4)])
    m_DB = td.Debye(eps_inf=1.0, coeffs=[(1, 3), (2, 4)])

    with pytest.raises(pydantic.ValidationError) as e_info:
        mf_SM = td.Sellmeier(coeffs=[(2, 0), (2, 4)])

    with pytest.raises(pydantic.ValidationError) as e_info:
        mf_DR = td.Drude(eps_inf=1.0, coeffs=[(1, 0), (2, 4)])

    with pytest.raises(pydantic.ValidationError) as e_info:
        mf_DB = td.Debye(eps_inf=1.0, coeffs=[(1, 0), (2, 4)])

    freqs = np.linspace(0.01, 1, 1001)
    for medium in [m_PR, m_SM, m_LZ, m_LZ2, m_DR, m_DB]:
        eps_c = medium.eps_model(freqs)

    for medium in [m_SM, m_LZ, m_LZ2, m_DR, m_DB]:
        eps_c = medium.eps_model(freqs)
        assert np.all(eps_c.imag >= 0)


def test_medium_dispersion_conversion():

    m_PR = td.PoleResidue(eps_inf=1.0, poles=[((-1 + 2j), (1 + 3j)), ((-2 + 4j), (1 + 5j))])
    m_SM = td.Sellmeier(coeffs=[(2, 3), (2, 4)])
    m_LZ = td.Lorentz(eps_inf=1.0, coeffs=[(1, 3, 2), (2, 4, 1)])
    m_LZ2 = td.Lorentz(eps_inf=1.0, coeffs=[(1, 2, 3), (2, 1, 4)])
    m_DR = td.Drude(eps_inf=1.0, coeffs=[(1, 3), (2, 4)])
    m_DB = td.Debye(eps_inf=1.0, coeffs=[(1, 3), (2, 4)])

    freqs = np.linspace(0.01, 1, 1001)
    for medium in [m_PR, m_SM, m_DB, m_LZ, m_DR, m_LZ2]:  # , m_DB]:
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
        struct = td.Structure(geometry=td.Box(size=(1, 1, 1)), medium=medium)


def test_sellmeier_from_dispersion():
    n = 3.5
    wvl = 0.5
    freq = td.C_0 / wvl
    dn_dwvl = -0.1
    with pytest.raises(ValidationError) as e:
        # Check that postivie dispersion raises an error
        medium = td.Sellmeier.from_dispersion(n=n, freq=freq, dn_dwvl=-dn_dwvl)

    # Check that medium properties are as epected
    medium = td.Sellmeier.from_dispersion(n=n, freq=freq, dn_dwvl=dn_dwvl)
    epses = [medium.eps_model(f) for f in [0.99 * freq, freq, 1.01 * freq]]
    ns = np.sqrt(epses)
    dn_df = (ns[2] - ns[0]) / 0.02 / freq

    assert np.allclose(ns[1], n)
    assert np.allclose(-dn_df * td.C_0 / wvl**2, dn_dwvl)


def eps_compare(medium: td.Medium, expected: Dict, tol: float = 1e-5):

    for freq, val in expected.items():
        assert np.abs(medium.eps_model(freq) - val) < tol


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


def test_gain_medium(log_capture):
    """Test passive and gain medium validations."""
    # non-dispersive
    with pytest.raises(pydantic.ValidationError) as e_info:
        m = td.Medium(conductivity=-0.1)
    with pytest.raises(pydantic.ValidationError) as e_info:
        m = td.Medium(conductivity=-1.0, allow_gain=False)
    mM = td.Medium(conductivity=-1.0, allow_gain=True)

    # pole residue, causality
    with pytest.raises(pydantic.ValidationError) as e_info:
        m = td.PoleResidue(eps_inf=0.16, poles=[(1 + 1j, 2 + 2j)])

    # Sellmeier
    with pytest.raises(pydantic.ValidationError) as e_info:
        m = td.Sellmeier(coeffs=((-1, 1),))
    mS = td.Sellmeier(coeffs=((-1, 1),), allow_gain=True)

    # Lorentz
    # causality, negative gamma
    with pytest.raises(pydantic.ValidationError) as e_info:
        m = td.Lorentz(eps_inf=0.04, coeffs=[(1, 2, -3)])
    # gain, negative Delta epsilon
    with pytest.raises(pydantic.ValidationError) as e_info:
        m = td.Lorentz(eps_inf=0.04, coeffs=[(-1, 2, 3)])
    mL = td.Lorentz(eps_inf=0.04, coeffs=[(-1, 2, 3)], allow_gain=True)
    assert mL.pole_residue.allow_gain

    # f_i can take whatever sign
    m = td.Lorentz(eps_inf=0.04, coeffs=[(1, -2, 3)])

    # Drude, only causality constraint
    with pytest.raises(pydantic.ValidationError) as e_info:
        m = td.Drude(eps_inf=0.04, coeffs=[(1, -2)])

    # anisotropic medium, warn allow_gain is ignored
    m = td.AnisotropicMedium(xx=td.Medium(), yy=mL, zz=mS, allow_gain=True)
    assert_log_level(log_capture, "WARNING")

    m = td.AnisotropicMedium(xx=td.Medium(), yy=mL, zz=mS, allow_gain=False)
    assert_log_level(log_capture, "WARNING")


def test_medium2d():
    sigma = 0.45
    thickness = 0.01
    cond_med = td.Medium(conductivity=sigma)
    medium = td.Medium2D.from_medium(cond_med, thickness=thickness)

    _ = medium.plot(freqs=[2e14, 3e14], ax=AX)
    _ = medium.plot_sigma(freqs=[2e14, 3e14], ax=AX)
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
    with pytest.raises(pydantic.ValidationError):
        td.FullyAnisotropicMedium(permittivity=2)
    with pytest.raises(pydantic.ValidationError):
        td.FullyAnisotropicMedium(permittivity=[3, 4, 2])

    # check that permittivity >= 1 and conductivity >= 0
    with pytest.raises(pydantic.ValidationError):
        td.FullyAnisotropicMedium(permittivity=[[3, 0, 0], [0, 0.5, 0], [0, 0, 1]])
    with pytest.raises(pydantic.ValidationError):
        td.FullyAnisotropicMedium(conductivity=[[-3, 0, 0], [0, 0.5, 0], [0, 0, 1]])
    td.FullyAnisotropicMedium(conductivity=[[-3, 0, 0], [0, 0.5, 0], [0, 0, 1]], allow_gain=True)

    # check that permittivity needs to be symmetric
    with pytest.raises(pydantic.ValidationError):
        td.FullyAnisotropicMedium(permittivity=[[3, 0.1, 0], [0.2, 2, 0], [0, 0, 1]])

    # check that differently oriented permittivity and conductivity are not accepted
    with pytest.raises(pydantic.ValidationError):
        td.FullyAnisotropicMedium(permittivity=perm, conductivity=cond2)

    # check creation from diagonal medium
    m = td.FullyAnisotropicMedium.from_diagonal(
        xx=td.Medium(permittivity=perm_diag[0][0], conductivity=cond_diag[0][0]),
        yy=td.Medium(permittivity=perm_diag[1][1], conductivity=cond_diag[1][1]),
        zz=td.Medium(permittivity=perm_diag[2][2], conductivity=cond_diag[2][2]),
        rotation=rot,
    )

    assert np.allclose(m.permittivity, perm)
    assert np.allclose(m.conductivity, cond)

    perm_d, cond_d, _ = m.eps_sigma_diag

    assert all(np.isin(np.round(perm_d), np.round(np.diag(perm_diag))))
    assert all(np.isin(np.round(cond_d), np.round(np.diag(cond_diag))))
