import numpy as np
import pydantic.v1 as pydantic
import pytest
import tidy3d as td
from tidy3d.material_library.material_library import (
    MaterialItem,
    MaterialItemUniaxial,
    ReferenceData,
    VariantItem,
    VariantItemUniaxial,
    export_matlib_to_file,
    material_library,
)

from ..utils import AssertLogLevel


def test_warning_default_variant_switching():
    """Issue warning for switching default medium variant."""

    # no warning for most materials with no default change
    with AssertLogLevel(None):
        _ = td.material_library["cSi"].medium

    # issue warning for SiO2
    with AssertLogLevel("WARNING"):
        _ = td.material_library["SiO2"].medium


def test_VariantItem():
    """Test if the variant class is working as expected."""
    _ = VariantItem(
        medium=td.PoleResidue(),
        reference=[ReferenceData(doi="etc.com", journal="paper", url="www")],
    )


def test_MaterialItem():
    """Test if the material class is working as expected."""
    variant1 = VariantItem(
        medium=td.PoleResidue(),
        reference=[ReferenceData(doi="etc.com", journal="paper", url="www")],
    )

    variant2 = VariantItem(
        medium=td.PoleResidue(),
        reference=[ReferenceData(doi="etc2.com", journal="paper2", url="www2")],
    )
    material = MaterialItem(name="material", variants=dict(v1=variant1, v2=variant2), default="v1")
    assert material["v1"] == material.medium

    with pytest.raises(pydantic.ValidationError):
        material = MaterialItem(
            name="material", variants=dict(v1=variant1, v2=variant2), default="v3"
        )


def test_library():
    """for each member of material library, ensure that it evaluates eps_model correctly"""
    for material_name, material in material_library.items():
        if isinstance(material, type):
            continue
        for variant_name, variant in material.variants.items():
            if not isinstance(variant, VariantItemUniaxial):
                if variant.medium.frequency_range:
                    fmin, fmax = variant.medium.frequency_range
                else:
                    fmin, fmax = 100e12, 300e12
                freqs = np.linspace(fmin, fmax, 10011)
                # two ways of access
                eps_complex1 = variant.medium.eps_model(freqs)
                eps_complex2 = material_library[material_name][variant_name].eps_model(freqs)
                assert np.allclose(eps_complex1, eps_complex2)
            else:
                for optical_axis in range(3):
                    if variant.medium(optical_axis).frequency_range:
                        fmin, fmax = variant.medium(optical_axis).frequency_range
                    else:
                        fmin, fmax = 100e12, 300e12
                    freqs = np.linspace(fmin, fmax, 10011)
                    # two ways of access
                    eps_complex1 = variant.medium(optical_axis).eps_model(freqs)
                    eps_complex2 = material_library[material_name][variant_name](
                        optical_axis
                    ).eps_model(freqs)
                    assert np.allclose(eps_complex1, eps_complex2)


def test_test_export(tmp_path):
    export_matlib_to_file(str(tmp_path / "matlib.json"))


def test_uniaxial_variant():
    """Test if the variant class is working as expected."""
    eps_ordinary = 2
    eps_extraordinary = 4
    mat = VariantItemUniaxial(
        ordinary=td.PoleResidue(eps_inf=eps_ordinary),
        extraordinary=td.PoleResidue(eps_inf=eps_extraordinary),
    )

    # optical axis along x-axis
    medium = mat.medium(0)
    assert medium.xx.eps_inf == eps_extraordinary
    assert medium.yy.eps_inf == eps_ordinary
    assert medium.zz.eps_inf == eps_ordinary

    # optical axis along y-axis
    medium = mat.medium(1)
    assert medium.xx.eps_inf == eps_ordinary
    assert medium.yy.eps_inf == eps_extraordinary
    assert medium.zz.eps_inf == eps_ordinary

    # optical axis along z-axis
    medium = mat.medium(2)
    assert medium.xx.eps_inf == eps_ordinary
    assert medium.yy.eps_inf == eps_ordinary
    assert medium.zz.eps_inf == eps_extraordinary


def test_uniaxial_material():
    """Test if the uniaxial material class is working as expected."""
    variant1 = VariantItemUniaxial(
        ordinary=td.PoleResidue(eps_inf=2),
        extraordinary=td.PoleResidue(eps_inf=4),
    )

    variant2 = VariantItemUniaxial(
        ordinary=td.PoleResidue(eps_inf=3),
        extraordinary=td.PoleResidue(eps_inf=6),
    )
    material = MaterialItemUniaxial(
        name="material", variants=dict(v1=variant1, v2=variant2), default="v1"
    )

    for optical_axis in range(3):
        assert material["v1"](optical_axis) == material.medium(optical_axis)
