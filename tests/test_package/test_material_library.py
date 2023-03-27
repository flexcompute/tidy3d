import pytest
import numpy as np
import pydantic

from tidy3d.material_library.material_library import (
    VariantItem,
    MaterialItem,
    ReferenceData,
    material_library,
    export_matlib_to_file,
)
import tidy3d as td
from tidy3d.exceptions import SetupError
from ..utils import clear_tmp


def test_VariantItem():
    """Test if the variant class is working as expected."""
    variant = VariantItem(
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
            if variant.medium.frequency_range:
                fmin, fmax = variant.medium.frequency_range
            else:
                fmin, fmax = 100e12, 300e12
            freqs = np.linspace(fmin, fmax, 10011)
            # two ways of access
            eps_complex1 = variant.medium.eps_model(freqs)
            eps_complex2 = material_library[material_name][variant_name].eps_model(freqs)
            assert np.allclose(eps_complex1, eps_complex2)


@clear_tmp
def test_test_export():
    export_matlib_to_file("tests/tmp/matlib.json")
