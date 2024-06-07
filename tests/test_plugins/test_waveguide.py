import numpy as np
import pytest
import tidy3d as td
from pydantic.v1 import ValidationError
from tidy3d.plugins import waveguide


def test_array_validators():
    with pytest.raises(ValidationError, match="negative"):
        waveguide.RectangularDielectric(
            wavelength=1.55,
            core_width=[0.1, -0.2],
            core_thickness=0.22,
            core_medium=td.Medium(permittivity=3.48**2),
            clad_medium=td.Medium(permittivity=1.45**2),
        )

    with pytest.raises(ValidationError, match="negative"):
        waveguide.RectangularDielectric(
            wavelength=1.55,
            core_width=-0.3,
            core_thickness=0.22,
            core_medium=td.Medium(permittivity=3.48**2),
            clad_medium=td.Medium(permittivity=1.45**2),
        )

    with pytest.raises(ValidationError, match="negative"):
        waveguide.RectangularDielectric(
            wavelength=1.55,
            core_width=0.3,
            clad_thickness=[-0.3],
            core_thickness=0.22,
            core_medium=td.Medium(permittivity=3.48**2),
            clad_medium=td.Medium(permittivity=1.45**2),
        )

    with pytest.raises(ValidationError, match="negative"):
        waveguide.RectangularDielectric(
            wavelength=1.55,
            core_width=0.3,
            box_thickness=[0.1, -0.3],
            core_thickness=0.22,
            core_medium=td.Medium(permittivity=3.48**2),
            clad_medium=td.Medium(permittivity=1.45**2),
            box_medium=[td.Medium()] * 2,
        )


def test_layer_validators():
    with pytest.raises(ValidationError, match="Number"):
        waveguide.RectangularDielectric(
            wavelength=1.55,
            core_width=0.3,
            core_thickness=0.22,
            core_medium=td.Medium(permittivity=3.48**2),
            clad_medium=[td.Medium(permittivity=1.45**2)] * 2,
        )

    with pytest.raises(ValidationError, match="Number"):
        waveguide.RectangularDielectric(
            wavelength=1.55,
            core_width=0.3,
            core_thickness=0.22,
            core_medium=td.Medium(permittivity=3.48**2),
            clad_thickness=[1, 1],
            clad_medium=[td.Medium(permittivity=1.45**2)] * 2,
            box_thickness=[1, 1],
        )

    waveguide.RectangularDielectric(
        wavelength=1.55,
        core_width=0.3,
        core_thickness=0.22,
        core_medium=td.Medium(permittivity=3.48**2),
        clad_thickness=[1, 1],
        clad_medium=[td.Medium(permittivity=1.45**2)] * 2,
    )


def test_rectangular_dielectric_consistency_validations():
    """Rectangular dielectric waveguide validations"""
    with pytest.raises(ValidationError, match=r"gaps"):
        waveguide.RectangularDielectric(
            wavelength=1.55,
            core_width=(0.5, 0.5),
            core_thickness=0.22,
            core_medium=td.Medium(permittivity=3.48**2),
            clad_medium=td.Medium(permittivity=1.45**2),
            gap=(0.1, 0.1),
        )

    with pytest.raises(ValidationError, match=r"sidewall thickness"):
        waveguide.RectangularDielectric(
            wavelength=1.55,
            core_width=0.5,
            core_thickness=0.22,
            core_medium=td.Medium(permittivity=3.48**2),
            clad_medium=td.Medium(permittivity=1.45**2),
            sidewall_thickness=0.01,
        )

    with pytest.raises(ValidationError, match=r"surface thickness"):
        waveguide.RectangularDielectric(
            wavelength=1.55,
            core_width=0.5,
            core_thickness=0.22,
            core_medium=td.Medium(permittivity=3.48**2),
            clad_medium=td.Medium(permittivity=1.45**2),
            surface_thickness=0.01,
        )


def test_rectangular_dielectric_strip():
    """Rectangular dielectric strip waveguide"""
    wg = waveguide.RectangularDielectric(
        wavelength=1.55,
        core_width=0.5,
        core_thickness=0.22,
        core_medium=td.Medium(permittivity=3.48**2),
        clad_medium=td.Medium(permittivity=1.45**2),
        mode_spec=td.ModeSpec(num_modes=2, precision="double"),
    )
    assert np.allclose(wg.n_eff.values, [2.4510009616579067, 1.7792425611078422], atol=1e-2)


def test_rectangular_dielectric_rib():
    """Rectangular dielectric rib waveguide"""
    wg = waveguide.RectangularDielectric(
        wavelength=1.55,
        core_width=0.45,
        core_thickness=0.22,
        slab_thickness=0.07,
        core_medium=td.Medium(permittivity=3.48**2),
        clad_medium=td.Medium(permittivity=1.0),
        box_medium=td.Medium(permittivity=1.45**2),
        sidewall_angle=0.5,
        mode_spec=td.ModeSpec(num_modes=2, precision="double"),
    )
    assert np.allclose(wg.n_eff.values, [2.543478205771245, 1.9583342349805217], atol=5e-2)


def test_rectangular_dielectric_coupled():
    """Rectangular dielectric coupled waveguides"""
    wg = waveguide.RectangularDielectric(
        wavelength=1.55,
        core_width=[0.38, 0.38],
        core_thickness=0.22,
        core_medium=td.Medium(permittivity=3.48**2),
        clad_medium=td.Medium(permittivity=1.45**2),
        sidewall_angle=0.2,
        gap=0.1,
        mode_spec=td.ModeSpec(num_modes=4, precision="double"),
    )
    assert np.allclose(wg.n_eff.values, [2.4453077, 2.2707212, 1.8694501, 1.5907708], atol=1e-1)


def test_layered_clad():
    wg = waveguide.RectangularDielectric(
        wavelength=1.55,
        core_width=0.5,
        core_thickness=0.22,
        clad_thickness=[0.1, 2],
        core_medium=td.Medium(permittivity=3.48**2),
        clad_medium=[td.Medium(permittivity=1.45**2), td.Medium(permittivity=1)],
        mode_spec=td.ModeSpec(num_modes=2, precision="double"),
    )
    assert np.allclose(wg.n_eff.values, [2.42377731, 1.69501951], atol=1e-2)


def test_layered_box():
    wg = waveguide.RectangularDielectric(
        wavelength=1.55,
        core_width=0.4,
        core_thickness=0.22,
        box_thickness=[0.1, 2],
        core_medium=td.Medium(permittivity=3.48**2),
        clad_medium=td.Medium(permittivity=1),
        box_medium=[td.Medium(permittivity=1.45**2), td.Medium(permittivity=1)],
        mode_spec=td.ModeSpec(num_modes=2, precision="double"),
    )
    assert np.allclose(wg.n_eff.values, [2.11032315, 1.29379651], atol=1e-2)
