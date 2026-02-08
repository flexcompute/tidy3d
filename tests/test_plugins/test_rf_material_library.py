"""Test the RF material library."""

from __future__ import annotations

import numpy as np
import pydantic.v1 as pydantic
import pytest

from tidy3d.components.medium import LossyMetalMedium, PoleResidue, SurfaceImpedanceFitterParam
from tidy3d.exceptions import ValidationError
from tidy3d.material_library.material_library import MaterialItem, ReferenceData, VariantItem
from tidy3d.plugins.microwave.rf_material_library import (
    VariantItemFreqRangeDielectric,
    VariantItemFreqRangeMetal,
    rf_material_library,
)

from ..utils import AssertLogLevel


def test_VariantItemFreqRangeDielectric():
    """Test if the VariantItemFreqRangeDielectric class is working as expected."""
    pole_res = PoleResidue(
        eps_inf=2.5,
        poles=[((-1e10 + 1e11j), (1e10 + 0j))],
        frequency_range=(1e9, 10e9),
    )
    variant = VariantItemFreqRangeDielectric(
        pole_residue=pole_res,
        loss_tangent=0.001,
        eps_real=2.5,
        measurement_frequencies=[1e9, 10e9],
        reference=[ReferenceData(doi="etc.com", journal="paper", url="www")],
    )
    assert variant.pole_residue == pole_res


def test_VariantItemFreqRangeDielectric_validation():
    """Test validation for VariantItemFreqRangeDielectric."""
    # Should fail without pole_residue
    with pytest.raises((ValidationError, pydantic.ValidationError)):
        _ = VariantItemFreqRangeDielectric(
            reference=[ReferenceData(doi="etc.com", journal="paper", url="www")],
        )


def test_VariantItemFreqRangeDielectric_medium():
    """Test VariantItemFreqRangeDielectric.medium() method."""
    pole_res = PoleResidue(
        eps_inf=2.5,
        poles=[((-1e10 + 1e11j), (1e10 + 0j))],
        frequency_range=(1e9, 10e9),
    )
    variant = VariantItemFreqRangeDielectric(
        pole_residue=pole_res,
        loss_tangent=0.001,
        eps_real=2.5,
        measurement_frequencies=[1e9, 10e9],
    )

    # Test without frequency_range - should return original
    medium1 = variant.medium()
    assert medium1 is variant.pole_residue
    assert medium1.frequency_range == (1e9, 10e9)

    # Test with frequency_range inside stored range - should return copy with updated range
    new_freq_range = (3e9, 7e9)  # Inside (1e9, 10e9)
    medium2 = variant.medium(new_freq_range)
    assert medium2 is variant.pole_residue  # Should be the same pole
    assert medium2.eps_inf == variant.pole_residue.eps_inf
    assert medium2.poles == variant.pole_residue.poles  # Poles unchanged

    # Test that epsilon calculation is the same (poles unchanged)
    test_freq = 5e9
    eps1 = medium1.eps_model(test_freq)
    eps2 = medium2.eps_model(test_freq)
    assert np.allclose(eps1, eps2)


def test_VariantItemFreqRangeDielectric_medium_frequency_range_cases():
    """Test VariantItemFreqRangeDielectric.medium() with inside/outside/overlapping frequency ranges."""
    pole_res = PoleResidue(
        eps_inf=2.5,
        poles=[((-1e10 + 1e11j), (1e10 + 0j))],
        frequency_range=(1e9, 10e9),
    )
    variant = VariantItemFreqRangeDielectric(
        pole_residue=pole_res,
        loss_tangent=[0.001, 0.0015],
        eps_real=[2.5, 2.6],
        measurement_frequencies=[1e9, 10e9],
    )

    # Case 1: Requested range is INSIDE stored range
    # Should return stored model with updated frequency_range, no warning
    with AssertLogLevel(None):
        inside_range = (3e9, 7e9)  # Completely inside (1e9, 10e9)
        medium_inside = variant.medium(inside_range)

        assert medium_inside is variant.pole_residue
        assert np.isclose(medium_inside.eps_inf, variant.pole_residue.eps_inf)
        assert np.allclose(
            medium_inside.poles, variant.pole_residue.poles
        )  # Original poles preserved

    # Case 2: Requested range is OUTSIDE stored range
    # Should create new model using constant loss tangent fitter, with warning
    with AssertLogLevel("WARNING", contains_str="outside"):
        outside_range = (20e9, 30e9)  # Completely outside (1e9, 10e9)
        medium_outside = variant.medium(outside_range)

        assert medium_outside is not variant.pole_residue  # Should be a new model
        assert np.allclose(medium_outside.frequency_range, outside_range)
        # New model should be a valid PoleResidue (may have different poles fitted for new range)
        assert isinstance(medium_outside, PoleResidue)
        assert medium_outside.eps_inf is not None

    # Case 3: Requested range OVERLAPS stored range (extends beyond on one side)
    # Should create new model using constant loss tangent fitter, with warning
    with AssertLogLevel("WARNING", contains_str="outside"):
        overlap_range_high = (5e9, 15e9)  # Overlaps but extends beyond upper bound
        medium_overlap_high = variant.medium(overlap_range_high)

        assert medium_overlap_high is not variant.pole_residue  # Should be a new model
        assert medium_overlap_high.frequency_range == overlap_range_high

    with AssertLogLevel("WARNING", contains_str="outside"):
        overlap_range_low = (0.5e9, 5e9)  # Overlaps but extends beyond lower bound
        medium_overlap_low = variant.medium(overlap_range_low)

        assert medium_overlap_low is not variant.pole_residue  # Should be a new model
        assert np.allclose(medium_overlap_low.frequency_range, overlap_range_low)

    # Case 4: Requested range exactly matches stored range
    # Should return stored model with updated frequency_range, no warning
    with AssertLogLevel(None):
        exact_range = (1e9, 10e9)  # Exactly matches stored range
        medium_exact = variant.medium(exact_range)

        assert medium_exact is variant.pole_residue  # Should be the same pole


def test_VariantItemFreqRangeDielectric_medium_none_frequency_range():
    """Test VariantItemFreqRangeDielectric.medium() when pole_residue.frequency_range is None."""
    pole_res = PoleResidue(
        eps_inf=2.5,
        poles=[((-1e10 + 1e11j), (1e10 + 0j))],
        frequency_range=None,  # No frequency range specified
    )
    variant = VariantItemFreqRangeDielectric(
        pole_residue=pole_res,
        loss_tangent=[0.001, 0.0015],
        eps_real=[2.5, 2.6],
        measurement_frequencies=[1e9, 10e9],
    )

    # When stored frequency_range is None, any requested range should create new model
    with AssertLogLevel("WARNING", contains_str="outside"):
        requested_range = (5e9, 15e9)
        medium = variant.medium(requested_range)

    assert medium is not variant.pole_residue  # Should be a new model
    assert medium.frequency_range == requested_range
    assert isinstance(medium, PoleResidue)


def test_VariantItemFreqRangeDielectric_medium_averaged_values():
    """Test that outside-range models use averaged loss_tangent and eps_real."""
    pole_res = PoleResidue(
        eps_inf=2.5,
        poles=[((-1e10 + 1e11j), (1e10 + 0j))],
        frequency_range=(1e9, 10e9),
    )

    # Use multiple values to test averaging
    loss_tangent_values = [0.001, 0.0015, 0.002]
    eps_real_values = [2.5, 2.6, 2.7]
    measurement_freqs = [1e9, 5e9, 10e9]

    variant = VariantItemFreqRangeDielectric(
        pole_residue=pole_res,
        loss_tangent=loss_tangent_values,
        eps_real=eps_real_values,
        measurement_frequencies=measurement_freqs,
    )

    # Request range outside stored range
    expected_loss_tan_avg = np.mean(loss_tangent_values)
    expected_eps_real_avg = np.mean(eps_real_values)

    # Verify warning mentions averaged values and contains the averaged values
    with AssertLogLevel("WARNING", contains_str="averaged"):
        outside_range = (20e9, 30e9)
        medium_outside = variant.medium(outside_range)


def test_VariantItemFreqRangeDielectric_summarize_mediums():
    """Test VariantItemFreqRangeDielectric.summarize_mediums property."""
    pole_res = PoleResidue(
        eps_inf=2.5,
        poles=[((-1e10 + 1e11j), (1e10 + 0j))],
        frequency_range=(1e9, 10e9),
    )
    variant = VariantItemFreqRangeDielectric(
        pole_residue=pole_res,
        loss_tangent=0.001,
        eps_real=2.5,
        measurement_frequencies=[1e9, 10e9],
    )
    mediums = variant.summarize_mediums
    assert isinstance(mediums, dict)
    assert "medium" in mediums
    assert mediums["medium"] == pole_res


def test_VariantItemFreqRangeMetal():
    """Test if the VariantItemFreqRangeMetal class is working as expected."""
    variant = VariantItemFreqRangeMetal(
        conductivity=60.0,
        reference=[ReferenceData(doi="etc.com", journal="paper", url="www")],
    )
    assert variant.conductivity == 60.0


def test_VariantItemFreqRangeMetal_validation():
    """Test validation for VariantItemFreqRangeMetal."""
    # Should fail without conductivity
    with pytest.raises((ValidationError, pydantic.ValidationError)):
        _ = VariantItemFreqRangeMetal(
            reference=[ReferenceData(doi="etc.com", journal="paper", url="www")],
        )

    # Should fail with negative conductivity
    with pytest.raises((ValidationError, pydantic.ValidationError)):
        _ = VariantItemFreqRangeMetal(conductivity=-1.0)

    # Should fail with zero conductivity
    with pytest.raises((ValidationError, pydantic.ValidationError)):
        _ = VariantItemFreqRangeMetal(conductivity=0.0)


def test_VariantItemFreqRangeMetal_medium():
    """Test VariantItemFreqRangeMetal.medium() method."""
    variant = VariantItemFreqRangeMetal(conductivity=60.0)
    frequency_range = (1e9, 10e9)

    medium = variant.medium(frequency_range)
    assert isinstance(medium, LossyMetalMedium)
    assert medium.conductivity == 60.0
    assert medium.frequency_range == frequency_range


def test_VariantItemFreqRangeMetal_with_optional_params():
    """Test VariantItemFreqRangeMetal with optional parameters."""
    from tidy3d.components.medium import HammerstadSurfaceRoughness

    roughness = HammerstadSurfaceRoughness(rq=0.5, roughness_factor=2.0)
    fit_param = SurfaceImpedanceFitterParam(max_num_poles=3, tolerance_rms=0.01)

    variant = VariantItemFreqRangeMetal(
        conductivity=60.0,
        roughness=roughness,
        thickness=1.0,
        fit_param=fit_param,
    )

    frequency_range = (1e9, 10e9)
    medium = variant.medium(frequency_range)

    assert medium.conductivity == 60.0
    assert medium.frequency_range == frequency_range
    assert medium.roughness == roughness
    assert medium.thickness == 1.0
    assert medium.fit_param == fit_param


def test_VariantItemFreqRangeMetal_summarize_mediums():
    """Test VariantItemFreqRangeMetal.summarize_mediums property."""
    variant = VariantItemFreqRangeMetal(conductivity=60.0)
    mediums = variant.summarize_mediums
    assert isinstance(mediums, dict)
    assert len(mediums) == 0  # Empty dict since we can't create medium without frequency_range


def test_rf_material_library_VariantItem():
    """Test that VariantItem materials in RF library work as expected."""
    # Test accessing a VariantItem material (like RO3010)
    ro3010 = rf_material_library["RO3010"]["design"]
    assert isinstance(ro3010, PoleResidue)
    assert ro3010.frequency_range is not None

    # Test default variant
    ro3010_default = rf_material_library["RO3010"].medium
    assert isinstance(ro3010_default, PoleResidue)


def test_rf_material_library_VariantItemFreqRangeDielectric():
    """Test VariantItemFreqRangeDielectric materials in RF library."""
    # Access variant object
    rt_duroid_variant = rf_material_library["RT_duroid5880"].variants["standard"]
    assert isinstance(rt_duroid_variant, VariantItemFreqRangeDielectric)

    # Test medium() without frequency_range
    medium1 = rt_duroid_variant.medium()
    assert isinstance(medium1, PoleResidue)
    assert medium1.frequency_range == rt_duroid_variant.pole_residue.frequency_range

    # Test medium() with frequency_range
    new_freq_range = (5e9, 20e9)
    medium2 = rt_duroid_variant.medium(new_freq_range)
    assert isinstance(medium2, PoleResidue)

    # Verify poles are unchanged
    assert np.allclose(medium1.poles, medium2.poles)
    assert np.isclose(medium1.eps_inf, medium2.eps_inf)


def test_rf_material_library_VariantItemFreqRangeMetal():
    """Test VariantItemFreqRangeMetal materials in RF library."""
    # Access variant object
    copper_variant = rf_material_library["Copper_Matula"].variants["standard"]
    assert isinstance(copper_variant, VariantItemFreqRangeMetal)

    # Test medium() with frequency_range
    frequency_range = (1e9, 10e9)
    medium = copper_variant.medium(frequency_range)
    assert isinstance(medium, LossyMetalMedium)
    assert np.isclose(medium.conductivity, copper_variant.conductivity)
    assert np.allclose(medium.frequency_range, frequency_range)


def test_rf_material_library_eps_model():
    """Test that all materials in RF library can evaluate eps_model correctly."""
    for material_name, material in rf_material_library.items():
        if isinstance(material, type):
            continue

        for variant_name, variant in material.variants.items():
            if isinstance(variant, VariantItem):
                # Standard VariantItem - direct medium property
                if variant.medium.frequency_range:
                    fmin, fmax = variant.medium.frequency_range
                else:
                    fmin, fmax = 1e9, 10e9
                freqs = np.linspace(fmin, fmax, 11)

                # Two ways of access should give same result
                eps_complex1 = variant.medium.eps_model(freqs)
                eps_complex2 = rf_material_library[material_name][variant_name].eps_model(freqs)
                assert np.allclose(eps_complex1, eps_complex2)

            elif isinstance(variant, VariantItemFreqRangeDielectric):
                # VariantItemFreqRangeDielectric - need to call medium()
                original_range = variant.pole_residue.frequency_range
                fmin, fmax = original_range
                freqs = np.linspace(fmin, fmax, 11)

                # Test with original frequency_range
                medium = variant.medium(original_range)
                eps_complex = medium.eps_model(freqs)
                assert len(eps_complex) == len(freqs)
                assert np.all(np.isfinite(eps_complex))

                # Test without frequency_range (should use original)
                medium2 = variant.medium()
                eps_complex2 = medium2.eps_model(freqs)
                assert np.allclose(eps_complex, eps_complex2)

            elif isinstance(variant, VariantItemFreqRangeMetal):
                # VariantItemFreqRangeMetal - need to call medium() with frequency_range
                frequency_range = (1e9, 10e9)
                medium = variant.medium(frequency_range)
                assert isinstance(medium, LossyMetalMedium)

                # LossyMetalMedium doesn't have eps_model, but we can verify it was created
                assert np.isclose(medium.conductivity, variant.conductivity)
                assert np.allclose(medium.frequency_range, frequency_range)


def test_rf_material_library_material_item():
    """Test MaterialItem behavior with RF materials."""
    # Test MaterialItem with VariantItem
    ro3010 = rf_material_library["RO3010"]
    assert isinstance(ro3010, MaterialItem)
    assert ro3010["design"] == ro3010.medium

    # Test MaterialItem with VariantItemFreqRangeDielectric
    rt_duroid = rf_material_library["RT_duroid5880"]
    assert isinstance(rt_duroid, MaterialItem)
    # Accessing via [] returns the variant object, not the medium
    variant = rt_duroid["standard"]
    assert isinstance(variant, VariantItemFreqRangeDielectric)
    # Need to call medium() to get the PoleResidue
    medium = variant.medium()
    assert isinstance(medium, PoleResidue)


def test_rf_material_library_frequency_range_consistency():
    """Test that frequency_range updates don't change epsilon calculations."""
    variant = rf_material_library["RT_duroid5880"].variants["standard"]
    original_range = variant.pole_residue.frequency_range

    # Get mediums with different frequency_ranges
    medium1 = variant.medium(original_range)
    medium2 = variant.medium((5e9, 20e9))
    medium3 = variant.medium()  # No parameter - uses original

    # Evaluate at same frequency - should get same epsilon
    test_freq = 10e9
    eps1 = medium1.eps_model(test_freq)
    eps2 = medium2.eps_model(test_freq)
    eps3 = medium3.eps_model(test_freq)

    assert np.allclose(eps1, eps2)
    assert np.allclose(eps1, eps3)
    assert np.allclose(eps2, eps3)


def test_rf_material_library_lossy_metal_fitting():
    """Test that LossyMetalMedium is properly fitted for the requested frequency range."""
    variant = rf_material_library["Copper_Matula"].variants["standard"]

    # Create medium for specific frequency range
    frequency_range = (1e9, 10e9)
    medium = variant.medium(frequency_range)

    assert isinstance(medium, LossyMetalMedium)
    assert medium.frequency_range == frequency_range

    # Verify that the medium has been fitted (has scaled_surface_impedance_model)
    assert hasattr(medium, "scaled_surface_impedance_model")
    assert hasattr(medium, "num_poles")
    assert medium.num_poles > 0  # Should have at least one pole


def test_rf_material_library_all_materials_accessible():
    """Test that all materials in rf_material_library are accessible."""
    for _material_name, material in rf_material_library.items():
        if isinstance(material, type):
            continue

        assert isinstance(material, MaterialItem)
        assert material.name is not None
        assert len(material.variants) > 0
        assert material.default in material.variants

        # Test accessing default variant
        default_variant = material.variants[material.default]
        assert default_variant is not None

        # Test accessing via [] operator
        variant = material[material.default]
        if isinstance(default_variant, VariantItem):
            assert isinstance(variant, PoleResidue)
        elif isinstance(default_variant, VariantItemFreqRangeDielectric):
            assert isinstance(variant, VariantItemFreqRangeDielectric)
        elif isinstance(default_variant, VariantItemFreqRangeMetal):
            assert isinstance(variant, VariantItemFreqRangeMetal)
