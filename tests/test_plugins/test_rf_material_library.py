"""Test the RF material library."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from tidy3d.components.medium import LossyMetalMedium, PoleResidue, SurfaceImpedanceFitterParam
from tidy3d.constants import MICROWAVE_FREQUENCY_RANGE
from tidy3d.material_library.material_library import MaterialItem, ReferenceData, VariantItem
from tidy3d.plugins.microwave.rf_material_library import (
    AbstractVariantItemFreqRange,
    MaterialItemFreqRange,
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
        prefitted_medium=pole_res,
        loss_tangent=[0.001, 0.001],
        eps_real=[2.5, 2.5],
        measurement_frequencies=[1e9, 10e9],
        reference=[ReferenceData(doi="etc.com", journal="paper", url="www")],
    )
    assert variant.prefitted_medium == pole_res


def test_VariantItemFreqRangeDielectric_validation():
    """Test validation for VariantItemFreqRangeDielectric."""
    # Should fail without prefitted_medium
    with pytest.raises(ValidationError):
        _ = VariantItemFreqRangeDielectric(
            reference=[ReferenceData(doi="etc.com", journal="paper", url="www")],
        )


def test_VariantItemFreqRangeDielectric_medium():
    """Test VariantItemFreqRangeDielectric.medium property."""
    pole_res = PoleResidue(
        eps_inf=2.5,
        poles=[((-1e10 + 1e11j), (1e10 + 0j))],
        frequency_range=(1e9, 10e9),
    )
    variant = VariantItemFreqRangeDielectric(
        prefitted_medium=pole_res,
        loss_tangent=[0.001, 0.001],
        eps_real=[2.5, 2.5],
        measurement_frequencies=[1e9, 10e9],
    )

    # Test medium property - should return original prefitted medium
    medium1 = variant.medium
    assert medium1 is variant.prefitted_medium
    assert medium1.frequency_range == (1e9, 10e9)

    # Test with frequency_range inside stored range - should return copy with updated range
    new_freq_range = (3e9, 7e9)  # Inside (1e9, 10e9)
    medium2 = variant.medium_in_range(new_freq_range)
    assert medium2 is variant.prefitted_medium  # Should be the same pole
    assert medium2.eps_inf == variant.prefitted_medium.eps_inf
    assert medium2.poles == variant.prefitted_medium.poles  # Poles unchanged

    # Test that epsilon calculation is the same (poles unchanged)
    test_freq = 5e9
    eps1 = medium1.eps_model(test_freq)
    eps2 = medium2.eps_model(test_freq)
    assert np.allclose(eps1, eps2)


def test_VariantItemFreqRangeDielectric_medium_frequency_range_cases():
    """Test VariantItemFreqRangeDielectric.medium_in_range() with inside/outside/overlapping frequency ranges."""
    pole_res = PoleResidue(
        eps_inf=2.5,
        poles=[((-1e10 + 1e11j), (1e10 + 0j))],
        frequency_range=(1e9, 10e9),
    )
    variant = VariantItemFreqRangeDielectric(
        prefitted_medium=pole_res,
        loss_tangent=[0.001, 0.0015],
        eps_real=[2.5, 2.6],
        measurement_frequencies=[1e9, 10e9],
    )

    # Case 1: Requested range is INSIDE stored range
    # Should return stored model with updated frequency_range, no warning
    with AssertLogLevel(None):
        inside_range = (3e9, 7e9)  # Completely inside (1e9, 10e9)
        medium_inside = variant.medium_in_range(inside_range)

        assert medium_inside is variant.prefitted_medium
        assert np.isclose(medium_inside.eps_inf, variant.prefitted_medium.eps_inf)
        assert np.allclose(
            medium_inside.poles, variant.prefitted_medium.poles
        )  # Original poles preserved

    # Case 2: Requested range is OUTSIDE stored range
    # Should create new model using constant loss tangent fitter, with warning
    with AssertLogLevel("WARNING", contains_str="outside"):
        outside_range = (20e9, 30e9)  # Completely outside (1e9, 10e9)
        medium_outside = variant.medium_in_range(outside_range)

        assert medium_outside is not variant.prefitted_medium  # Should be a new model
        assert np.allclose(medium_outside.frequency_range, outside_range)
        # New model should be a valid PoleResidue (may have different poles fitted for new range)
        assert isinstance(medium_outside, PoleResidue)
        assert medium_outside.eps_inf is not None

    # Case 3: Requested range OVERLAPS stored range (extends beyond on one side)
    # Should create new model using constant loss tangent fitter, with warning
    with AssertLogLevel("WARNING", contains_str="outside"):
        overlap_range_high = (5e9, 15e9)  # Overlaps but extends beyond upper bound
        medium_overlap_high = variant.medium_in_range(overlap_range_high)

        assert medium_overlap_high is not variant.prefitted_medium  # Should be a new model
        assert medium_overlap_high.frequency_range == overlap_range_high

    with AssertLogLevel("WARNING", contains_str="outside"):
        overlap_range_low = (0.5e9, 5e9)  # Overlaps but extends beyond lower bound
        medium_overlap_low = variant.medium_in_range(overlap_range_low)

        assert medium_overlap_low is not variant.prefitted_medium  # Should be a new model
        assert np.allclose(medium_overlap_low.frequency_range, overlap_range_low)

    # Case 4: Requested range exactly matches stored range
    # Should return stored model with updated frequency_range, no warning
    with AssertLogLevel(None):
        exact_range = (1e9, 10e9)  # Exactly matches stored range
        medium_exact = variant.medium_in_range(exact_range)

        assert medium_exact is variant.prefitted_medium  # Should be the same pole


def test_VariantItemFreqRangeDielectric_medium_none_frequency_range():
    """Test VariantItemFreqRangeDielectric.medium_in_range() when prefitted_medium.frequency_range is None."""
    pole_res = PoleResidue(
        eps_inf=2.5,
        poles=[((-1e10 + 1e11j), (1e10 + 0j))],
        frequency_range=None,  # No frequency range specified
    )
    variant = VariantItemFreqRangeDielectric(
        prefitted_medium=pole_res,
        loss_tangent=[0.001, 0.0015],
        eps_real=[2.5, 2.6],
        measurement_frequencies=[1e9, 10e9],
    )

    # When stored frequency_range is None, any requested range should create new model
    with AssertLogLevel("WARNING", contains_str="outside"):
        requested_range = (5e9, 15e9)
        medium = variant.medium_in_range(requested_range)

    assert medium is not variant.prefitted_medium  # Should be a new model
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
        prefitted_medium=pole_res,
        loss_tangent=loss_tangent_values,
        eps_real=eps_real_values,
        measurement_frequencies=measurement_freqs,
    )

    # Request range outside stored range
    expected_loss_tan_avg = np.mean(loss_tangent_values)
    expected_eps_real_avg = np.mean(eps_real_values)

    # Verify warning mentions averaged values and contains the averaged values
    with AssertLogLevel("WARNING", contains_str="averaged") as ctx:
        outside_range = (20e9, 30e9)
        medium_outside = variant.medium_in_range(outside_range)

    # Check that warning contains the averaged values (with some tolerance for formatting)
    # Records are tuples of (level, message)
    warning_msg = " ".join([record[1] for record in ctx.records])
    assert (
        f"{expected_loss_tan_avg:.6f}" in warning_msg
        or f"{expected_loss_tan_avg:.5f}" in warning_msg
    )
    assert (
        f"{expected_eps_real_avg:.6f}" in warning_msg
        or f"{expected_eps_real_avg:.5f}" in warning_msg
    )


def test_VariantItemFreqRangeDielectric_medium_single_float_values():
    """Test that single float values (not lists) work correctly for averaging."""
    pole_res = PoleResidue(
        eps_inf=2.5,
        poles=[((-1e10 + 1e11j), (1e10 + 0j))],
        frequency_range=(1e9, 10e9),
    )

    # Use single float values (not lists) - all must be same length
    variant = VariantItemFreqRangeDielectric(
        prefitted_medium=pole_res,
        loss_tangent=0.0015,  # Single float, length 1
        eps_real=2.6,  # Single float, length 1
        measurement_frequencies=1e9,  # Single float, length 1
    )

    # Request range outside stored range - should use single float values directly
    with AssertLogLevel("WARNING", contains_str="outside"):
        outside_range = (20e9, 30e9)
        medium_outside = variant.medium_in_range(outside_range)

    assert isinstance(medium_outside, PoleResidue)
    assert medium_outside.frequency_range == outside_range


def test_VariantItemFreqRangeDielectric_paired_field_lengths():
    """Test validation for matching lengths of paired fields (all must have same length)."""
    pole_res = PoleResidue(
        eps_inf=2.5,
        poles=[((-1e10 + 1e11j), (1e10 + 0j))],
        frequency_range=(1e9, 10e9),
    )
    reference = [ReferenceData(doi="test.com", journal="test", url="test")]

    # Test case 1: All single values (all length 1) - should pass
    variant1 = VariantItemFreqRangeDielectric(
        prefitted_medium=pole_res,
        loss_tangent=0.001,
        eps_real=2.5,
        measurement_frequencies=1e9,
        reference=reference,
    )
    assert variant1.loss_tangent == 0.001
    assert variant1.eps_real == 2.5

    # Test case 2: All lists with matching lengths - should pass
    variant2 = VariantItemFreqRangeDielectric(
        prefitted_medium=pole_res,
        loss_tangent=[0.001, 0.002],
        eps_real=[2.5, 2.6],
        measurement_frequencies=[1e9, 2e9],
        reference=reference,
    )
    assert len(variant2.loss_tangent) == 2
    assert len(variant2.eps_real) == 2

    # Test case 3: All lists of length 1 (equivalent to single values) - should pass
    variant3 = VariantItemFreqRangeDielectric(
        prefitted_medium=pole_res,
        loss_tangent=[0.001],
        eps_real=[2.5],
        measurement_frequencies=[1e9],
        reference=reference,
    )
    assert isinstance(variant3.loss_tangent, list)
    assert len(variant3.loss_tangent) == 1

    # Test case 4: Mix of single values and length-1 lists - should pass (all length 1)
    variant4 = VariantItemFreqRangeDielectric(
        prefitted_medium=pole_res,
        loss_tangent=[0.001],  # length 1
        eps_real=2.5,  # length 1 (single value)
        measurement_frequencies=1e9,  # length 1 (single value)
        reference=reference,
    )
    assert len(variant4.loss_tangent) == 1

    # Test case 5: Mix of single value and list of length > 1 - should raise ValidationError
    with pytest.raises(ValidationError, match="Mismatched lengths"):
        _ = VariantItemFreqRangeDielectric(
            prefitted_medium=pole_res,
            loss_tangent=[0.001, 0.002],  # length 2
            eps_real=2.5,  # length 1 (single value)
            measurement_frequencies=1e9,  # length 1 (single value)
            reference=reference,
        )

    # Test case 6: All lists with mismatched lengths - should raise ValidationError
    with pytest.raises(ValidationError, match="Mismatched lengths"):
        _ = VariantItemFreqRangeDielectric(
            prefitted_medium=pole_res,
            loss_tangent=[0.001, 0.002],  # length 2
            eps_real=[2.5],  # length 1
            measurement_frequencies=[1e9, 2e9],  # length 2
            reference=reference,
        )

    # Test case 7: Two lists with mismatched lengths - should raise ValidationError
    with pytest.raises(ValidationError, match="Mismatched lengths"):
        _ = VariantItemFreqRangeDielectric(
            prefitted_medium=pole_res,
            loss_tangent=[0.001, 0.002, 0.003],  # length 3
            eps_real=[2.5, 2.6],  # length 2
            measurement_frequencies=1e9,  # length 1 (single value)
            reference=reference,
        )

    # Test case 8: Tuple and numpy array with matching lengths - should pass
    variant8 = VariantItemFreqRangeDielectric(
        prefitted_medium=pole_res,
        loss_tangent=(0.001, 0.002),  # tuple, length 2
        eps_real=np.array([2.5, 2.6]),  # numpy array, length 2
        measurement_frequencies=[1e9, 2e9],  # list, length 2
        reference=reference,
    )
    assert len(variant8.loss_tangent) == 2
    assert len(variant8.eps_real) == 2

    # Test case 9: Tuple and list with mismatched lengths - should raise ValidationError
    with pytest.raises(ValidationError, match="Mismatched lengths"):
        _ = VariantItemFreqRangeDielectric(
            prefitted_medium=pole_res,
            loss_tangent=(0.001, 0.002),  # length 2
            eps_real=np.array([2.5]),  # length 1
            measurement_frequencies=[1e9, 2e9],  # length 2
            reference=reference,
        )

    # Test case 10: Empty lists - should raise ValidationError (fail-fast)
    with pytest.raises(ValidationError, match="cannot be empty"):
        _ = VariantItemFreqRangeDielectric(
            prefitted_medium=pole_res,
            loss_tangent=[],  # empty list
            eps_real=[],  # empty list
            measurement_frequencies=[],  # empty list
            reference=reference,
        )

    # Test case 11: Empty tuple - should raise ValidationError
    with pytest.raises(ValidationError, match="cannot be empty"):
        _ = VariantItemFreqRangeDielectric(
            prefitted_medium=pole_res,
            loss_tangent=(),  # empty tuple
            eps_real=(),  # empty tuple
            measurement_frequencies=(),  # empty tuple
            reference=reference,
        )

    # Test case 12: Empty numpy array - should raise ValidationError
    with pytest.raises(ValidationError, match="cannot be empty"):
        _ = VariantItemFreqRangeDielectric(
            prefitted_medium=pole_res,
            loss_tangent=np.array([]),  # empty array
            eps_real=np.array([]),  # empty array
            measurement_frequencies=np.array([]),  # empty array
            reference=reference,
        )

    # Test case 13: Single NaN value - should raise ValidationError
    with pytest.raises(ValidationError, match="contains NaN values"):
        _ = VariantItemFreqRangeDielectric(
            prefitted_medium=pole_res,
            loss_tangent=float("nan"),
            eps_real=2.5,
            measurement_frequencies=1e9,
            reference=reference,
        )

    # Test case 14: NaN in list - should raise ValidationError
    with pytest.raises(ValidationError, match="contains NaN values"):
        _ = VariantItemFreqRangeDielectric(
            prefitted_medium=pole_res,
            loss_tangent=[0.001, float("nan")],
            eps_real=[2.5, 2.6],
            measurement_frequencies=[1e9, 2e9],
            reference=reference,
        )

    # Test case 15: NaN in numpy array - should raise ValidationError
    with pytest.raises(ValidationError, match="contains NaN values"):
        _ = VariantItemFreqRangeDielectric(
            prefitted_medium=pole_res,
            loss_tangent=np.array([0.001, float("nan")]),
            eps_real=np.array([2.5, 2.6]),
            measurement_frequencies=np.array([1e9, 2e9]),
            reference=reference,
        )


def test_VariantItemFreqRangeDielectric_summarize_mediums():
    """Test VariantItemFreqRangeDielectric.summarize_mediums property."""
    pole_res = PoleResidue(
        eps_inf=2.5,
        poles=[((-1e10 + 1e11j), (1e10 + 0j))],
        frequency_range=(1e9, 10e9),
    )
    variant = VariantItemFreqRangeDielectric(
        prefitted_medium=pole_res,
        loss_tangent=[0.001, 0.001],
        eps_real=[2.5, 2.5],
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
    with pytest.raises(ValidationError):
        _ = VariantItemFreqRangeMetal(
            reference=[ReferenceData(doi="etc.com", journal="paper", url="www")],
        )

    # Should fail with negative conductivity
    with pytest.raises(ValidationError):
        _ = VariantItemFreqRangeMetal(conductivity=-1.0)

    # Should fail with zero conductivity
    with pytest.raises(ValidationError):
        _ = VariantItemFreqRangeMetal(conductivity=0.0)


def test_VariantItemFreqRangeMetal_medium():
    """Test VariantItemFreqRangeMetal.medium property and medium_in_range() method."""
    variant = VariantItemFreqRangeMetal(conductivity=60.0)
    frequency_range = (1e9, 10e9)

    # Test with explicit frequency_range
    medium = variant.medium_in_range(frequency_range)
    assert isinstance(medium, LossyMetalMedium)
    assert medium.conductivity == 60.0
    assert medium.frequency_range == frequency_range

    # Test medium property - should use default RF frequency range
    default_medium = variant.medium
    assert isinstance(default_medium, LossyMetalMedium)
    assert default_medium.conductivity == 60.0
    assert default_medium.frequency_range == MICROWAVE_FREQUENCY_RANGE


def test_VariantItemFreqRangeMetal_medium_none_frequency_range():
    """Test that VariantItemFreqRangeMetal.medium_in_range(None) and medium_in_range(MICROWAVE_FREQUENCY_RANGE) return equivalent medium to the medium property."""
    variant = VariantItemFreqRangeMetal(conductivity=60.0)

    # medium_in_range(None) should return equivalent medium to medium property
    medium1 = variant.medium_in_range(None)
    medium2 = variant.medium

    # Should have equivalent properties (not necessarily same object since medium is a property)
    assert isinstance(medium1, LossyMetalMedium)
    assert isinstance(medium2, LossyMetalMedium)
    assert medium1.frequency_range == medium2.frequency_range == MICROWAVE_FREQUENCY_RANGE
    assert medium1.conductivity == medium2.conductivity == 60.0
    assert medium1.roughness == medium2.roughness
    assert medium1.thickness == medium2.thickness

    # Verify fit parameters use enhanced values for default frequency range when fit_param is None
    assert medium1.fit_param is not None
    assert medium2.fit_param is not None
    assert (
        medium1.fit_param.max_num_poles == medium2.fit_param.max_num_poles == 12
    )  # Enhanced for wide frequency range
    assert (
        medium1.fit_param.frequency_sampling_points
        == medium2.fit_param.frequency_sampling_points
        == 50
    )  # Enhanced for wide frequency range
    assert (
        medium1.fit_param.tolerance_rms == medium2.fit_param.tolerance_rms == 1e-3
    )  # Default value

    # medium_in_range(MICROWAVE_FREQUENCY_RANGE) should also return equivalent medium to medium property
    medium3 = variant.medium_in_range(MICROWAVE_FREQUENCY_RANGE)
    assert isinstance(medium3, LossyMetalMedium)
    assert medium3.frequency_range == medium2.frequency_range == MICROWAVE_FREQUENCY_RANGE
    assert medium3.conductivity == medium2.conductivity == 60.0
    assert medium3.roughness == medium2.roughness
    assert medium3.thickness == medium2.thickness
    # Should use enhanced fit parameters (same as medium property)
    assert medium3.fit_param.max_num_poles == medium2.fit_param.max_num_poles == 12
    assert (
        medium3.fit_param.frequency_sampling_points
        == medium2.fit_param.frequency_sampling_points
        == 50
    )
    assert medium3.fit_param.tolerance_rms == medium2.fit_param.tolerance_rms == 1e-3


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
    medium = variant.medium_in_range(frequency_range)

    assert medium.conductivity == 60.0
    assert medium.frequency_range == frequency_range
    assert medium.roughness == roughness
    assert medium.thickness == 1.0
    assert medium.fit_param == fit_param

    # When custom fit_param is provided, medium_in_range(None) should return equivalent to medium property
    medium_default = variant.medium_in_range(None)
    medium_prop = variant.medium
    # Should have equivalent properties (not necessarily same object since medium is a property)
    assert (
        medium_default.frequency_range == medium_prop.frequency_range == MICROWAVE_FREQUENCY_RANGE
    )
    assert medium_default.conductivity == medium_prop.conductivity == 60.0
    assert medium_default.fit_param == medium_prop.fit_param == fit_param
    assert medium_default.fit_param.max_num_poles == 3  # Custom value


def test_VariantItemFreqRangeMetal_fit_param_consistency():
    """Test that medium and medium_in_range(None/MICROWAVE_FREQUENCY_RANGE) return equivalent mediums with consistent fit parameters."""
    variant = VariantItemFreqRangeMetal(conductivity=60.0)

    # medium_in_range(None) should return equivalent medium to medium property
    medium_prop = variant.medium
    medium_method = variant.medium_in_range(None)

    # Should have equivalent properties (not necessarily same object since medium is a property)
    assert medium_prop.frequency_range == medium_method.frequency_range == MICROWAVE_FREQUENCY_RANGE
    assert medium_prop.conductivity == medium_method.conductivity == 60.0

    # Verify fit parameters use enhanced values for default frequency range when fit_param is None
    assert medium_prop.fit_param is not None
    assert medium_method.fit_param is not None
    assert (
        medium_prop.fit_param.max_num_poles == medium_method.fit_param.max_num_poles == 12
    )  # Enhanced for wide frequency range
    assert (
        medium_prop.fit_param.frequency_sampling_points
        == medium_method.fit_param.frequency_sampling_points
        == 50
    )  # Enhanced for wide frequency range
    assert (
        medium_prop.fit_param.tolerance_rms == medium_method.fit_param.tolerance_rms == 1e-3
    )  # Default value

    # When using the default microwave frequency range explicitly, should return same as medium property
    medium_explicit_default = variant.medium_in_range(MICROWAVE_FREQUENCY_RANGE)
    assert (
        medium_explicit_default.frequency_range
        == medium_prop.frequency_range
        == MICROWAVE_FREQUENCY_RANGE
    )
    assert medium_explicit_default.conductivity == medium_prop.conductivity == 60.0
    # Should use enhanced fit parameters (same as medium property)
    assert (
        medium_explicit_default.fit_param.max_num_poles == medium_prop.fit_param.max_num_poles == 12
    )
    assert (
        medium_explicit_default.fit_param.frequency_sampling_points
        == medium_prop.fit_param.frequency_sampling_points
        == 50
    )

    # When using a different frequency range, should also use default fit_param
    medium_custom_range = variant.medium_in_range((1e9, 10e9))
    assert (
        medium_custom_range.fit_param is not None
    )  # LossyMetalMedium creates default via default_factory
    assert medium_custom_range.fit_param.max_num_poles == 5  # Default value
    assert medium_custom_range.fit_param.frequency_sampling_points == 20  # Default value


def test_VariantItemFreqRangeMetal_summarize_mediums():
    """Test VariantItemFreqRangeMetal.summarize_mediums property."""
    variant = VariantItemFreqRangeMetal(conductivity=60.0)
    mediums = variant.summarize_mediums
    assert isinstance(mediums, dict)
    assert len(mediums) == 1
    assert "medium" in mediums
    assert isinstance(mediums["medium"], LossyMetalMedium)
    # Should use default microwave frequency range (300 MHz to 300 GHz)
    assert mediums["medium"].frequency_range == (0.3e9, 300e9)


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

    # Test medium property
    medium1 = rt_duroid_variant.medium
    assert isinstance(medium1, PoleResidue)
    assert medium1.frequency_range == rt_duroid_variant.prefitted_medium.frequency_range

    # Test medium_in_range() with frequency_range
    new_freq_range = (5e9, 20e9)
    medium2 = rt_duroid_variant.medium_in_range(new_freq_range)
    assert isinstance(medium2, PoleResidue)

    # Verify poles are unchanged
    assert np.allclose(medium1.poles, medium2.poles)
    assert np.isclose(medium1.eps_inf, medium2.eps_inf)


def test_rf_material_library_VariantItemFreqRangeMetal():
    """Test VariantItemFreqRangeMetal materials in RF library."""
    # Access variant object
    copper_variant = rf_material_library["Copper_Matula"].variants["standard"]
    assert isinstance(copper_variant, VariantItemFreqRangeMetal)

    # Test medium_in_range() with frequency_range
    frequency_range = (1e9, 10e9)
    medium = copper_variant.medium_in_range(frequency_range)
    assert isinstance(medium, LossyMetalMedium)
    assert np.isclose(medium.conductivity, copper_variant.conductivity)
    assert np.allclose(medium.frequency_range, frequency_range)

    # Test medium property - should use default RF frequency range
    default_medium = copper_variant.medium
    assert isinstance(default_medium, LossyMetalMedium)
    assert np.isclose(default_medium.conductivity, copper_variant.conductivity)
    assert default_medium.frequency_range == MICROWAVE_FREQUENCY_RANGE


def test_MaterialItemFreqRange_medium_property():
    """Test MaterialItemFreqRange.medium property and medium_in_range() method."""
    # Test with dielectric material
    rt_duroid = rf_material_library["RT_duroid5880"]
    assert isinstance(rt_duroid, MaterialItemFreqRange)

    # Should return PoleResidue for dielectric (medium property)
    default_medium = rt_duroid.medium
    assert isinstance(default_medium, PoleResidue)
    assert default_medium == rt_duroid.variants[rt_duroid.default].medium

    # Test with metal material - medium property uses default RF range
    copper_material = rf_material_library["Copper_Matula"]
    assert isinstance(copper_material, MaterialItemFreqRange)

    # Should return LossyMetalMedium (medium property uses default RF range)
    default_metal_medium = copper_material.medium
    assert isinstance(default_metal_medium, LossyMetalMedium)
    # Check that it uses the default microwave frequency range (300 MHz to 300 GHz)
    assert default_metal_medium.frequency_range == (0.3e9, 300e9)

    # Test with custom frequency range
    custom_freq_range = (1e9, 10e9)
    custom_metal_medium = copper_material.medium_in_range(frequency_range=custom_freq_range)
    assert isinstance(custom_metal_medium, LossyMetalMedium)
    assert custom_metal_medium.frequency_range == custom_freq_range


def test_MaterialItemFreqRange_medium_unsupported_variant():
    """Test MaterialItemFreqRange.medium property raises error for unsupported variant types."""

    # Create a custom variant class that inherits from AbstractVariantItemFreqRange
    # but isn't one of the two supported types (VariantItemFreqRangeDielectric or VariantItemFreqRangeMetal)
    class UnsupportedVariant(AbstractVariantItemFreqRange):
        """An unsupported variant type for testing."""

        @property
        def medium(self):
            """Dummy implementation."""
            return PoleResidue(
                eps_inf=1.0,
                poles=[],
                frequency_range=(1e9, 10e9),
            )

        def medium_in_range(self, frequency_range=None):
            """Dummy implementation."""
            return PoleResidue(
                eps_inf=1.0,
                poles=[],
                frequency_range=frequency_range or (1e9, 10e9),
            )

        @property
        def summarize_mediums(self):
            """Dummy implementation."""
            return {}

    # Create a MaterialItemFreqRange with the unsupported variant
    unsupported_variant = UnsupportedVariant(
        reference=[ReferenceData(doi="test.com", journal="test", url="test")]
    )
    material = MaterialItemFreqRange(
        name="TestMaterial",
        variants={"standard": unsupported_variant},
        default="standard",
    )

    # Accessing .medium should raise ValueError with appropriate message
    with pytest.raises(ValueError) as exc_info:
        _ = material.medium

    # Verify the error message contains expected information
    error_message = str(exc_info.value)
    assert "UnsupportedVariant" in error_message
    assert "TestMaterial" in error_message
    assert "MaterialItemFreqRange.medium" in error_message
    assert "VariantItemFreqRangeDielectric" in error_message
    assert "VariantItemFreqRangeMetal" in error_message


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
                # VariantItemFreqRangeDielectric - use medium property or medium_in_range()
                original_range = variant.prefitted_medium.frequency_range
                fmin, fmax = original_range
                freqs = np.linspace(fmin, fmax, 11)

                # Test with original frequency_range
                medium = variant.medium_in_range(original_range)
                eps_complex = medium.eps_model(freqs)
                assert len(eps_complex) == len(freqs)
                assert np.all(np.isfinite(eps_complex))

                # Test medium property (should use original)
                medium2 = variant.medium
                eps_complex2 = medium2.eps_model(freqs)
                assert np.allclose(eps_complex, eps_complex2)

            elif isinstance(variant, VariantItemFreqRangeMetal):
                # VariantItemFreqRangeMetal - use medium property or medium_in_range()
                frequency_range = (1e9, 10e9)
                medium = variant.medium_in_range(frequency_range)
                assert isinstance(medium, LossyMetalMedium)

                # LossyMetalMedium doesn't have eps_model, but we can verify it was created
                assert np.isclose(medium.conductivity, variant.conductivity)
                assert np.allclose(medium.frequency_range, frequency_range)

                # Test medium property - should use default RF frequency range
                default_medium = variant.medium
                assert isinstance(default_medium, LossyMetalMedium)
                assert default_medium.frequency_range == MICROWAVE_FREQUENCY_RANGE


def test_rf_material_library_material_item():
    """Test MaterialItem behavior with RF materials."""
    # Test MaterialItem with VariantItem
    ro3010 = rf_material_library["RO3010"]
    assert isinstance(ro3010, MaterialItem)
    assert ro3010["design"] == ro3010.medium

    # Test MaterialItem with VariantItemFreqRangeDielectric (same LSP as VariantItem)
    rt_duroid = rf_material_library["RT_duroid5880"]
    assert isinstance(rt_duroid, MaterialItem)
    # Accessing via [] returns the medium, consistent with MaterialItem
    medium = rt_duroid["standard"]
    assert isinstance(medium, PoleResidue)


def test_rf_material_library_frequency_range_consistency():
    """Test that frequency_range updates don't change epsilon calculations."""
    variant = rf_material_library["RT_duroid5880"].variants["standard"]
    original_range = variant.prefitted_medium.frequency_range

    # Get mediums with different frequency_ranges
    medium1 = variant.medium_in_range(original_range)
    medium2 = variant.medium_in_range((5e9, 20e9))
    medium3 = variant.medium  # Property - uses original prefitted medium

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
    medium = variant.medium_in_range(frequency_range)

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

        # Test accessing via [] operator (returns medium for all variant types)
        variant_medium = material[material.default]
        if isinstance(default_variant, VariantItem):
            assert isinstance(variant_medium, PoleResidue)
        elif isinstance(default_variant, VariantItemFreqRangeDielectric):
            assert isinstance(variant_medium, PoleResidue)
        elif isinstance(default_variant, VariantItemFreqRangeMetal):
            assert isinstance(variant_medium, LossyMetalMedium)
