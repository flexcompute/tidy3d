"""Test the microwave plugin."""

from __future__ import annotations

from math import isclose

import matplotlib.pyplot as plt
import numpy as np
import pytest
from pydantic import ValidationError
from skrf import Frequency
from skrf.media import MLine

import tidy3d.plugins.microwave as mw

MAKE_PLOTS = False
if MAKE_PLOTS:
    # Interative plotting for debugging
    from matplotlib import use

    use("TkAgg")


def test_microstrip_models():
    """Test that the microstrip model computes transmission line parameters accurately."""
    width = 3.0
    height = 1.0
    thickness = 0.0
    eps_r = 4.4

    # Check zero thickness parameters
    Z0, eps_eff = mw.models.microstrip.compute_line_params(eps_r, width, height, thickness)
    freqs = Frequency(start=1, stop=1, npoints=1, unit="ghz")
    mline = MLine(frequency=freqs, w=width, h=height, t=thickness, ep_r=eps_r, disp="none")

    assert np.isclose(Z0, mline.z0[0])
    assert np.isclose(eps_eff, mline.ep_reff[0])

    # Check end effect length computation
    dL = mw.models.microstrip.compute_end_effect_length(eps_r, eps_eff, width, height)
    assert np.isclose(dL, 0.54, rtol=0.01)

    # Check finite thickness parameters
    thickness = 0.1
    Z0, eps_eff = mw.models.microstrip.compute_line_params(eps_r, width, height, thickness)
    mline = MLine(frequency=freqs, w=width, h=height, t=thickness, ep_r=eps_r, disp="none")

    assert np.isclose(Z0, mline.z0[0])
    assert np.isclose(eps_eff, mline.ep_reff[0])


def test_coupled_microstrip_model():
    """Test that the coupled microstrip model computes transmission line parameters accurately."""
    w1 = 1.416
    w2 = 2.396
    height = 1.56
    g1 = 0.134
    g2 = 0.386
    eps_r = 4.3
    # Compare to:   Taoufik, Ragani, N. Amar Touhami, and M. Agoutane. "Designing a Microstrip coupled line bandpass filter."
    #               International Journal of Engineering & Technology 2, no. 4 (2013): 266.
    # and notebook "CoupledLineBandpassFilter"

    (Z_even, Z_odd, eps_even, eps_odd) = mw.models.coupled_microstrip.compute_line_params(
        eps_r, w1, height, g1
    )
    assert np.isclose(Z_even, 101.5, rtol=0.01)
    assert np.isclose(Z_odd, 38.5, rtol=0.01)
    assert np.isclose(eps_even, 3.26, rtol=0.01)
    assert np.isclose(eps_odd, 2.71, rtol=0.01)

    (Z_even, Z_odd, eps_even, eps_odd) = mw.models.coupled_microstrip.compute_line_params(
        eps_r, w2, height, g2
    )
    assert np.isclose(Z_even, 71, rtol=0.01)
    assert np.isclose(Z_odd, 39, rtol=0.01)
    assert np.isclose(eps_even, 3.42, rtol=0.01)
    assert np.isclose(eps_odd, 2.80, rtol=0.01)


def test_lobe_measurer_validation():
    """ "Make sure the lobe measurer validates inputs correctly."""
    theta = np.linspace(0, 2 * np.pi, 101, endpoint=False)
    Urad = np.cos(theta)

    # Raise error when radiation pattern is negative
    with pytest.raises(ValidationError):
        mw.LobeMeasurer(
            angle=theta,
            radiation_pattern=Urad,
        )

    Urad = np.cos(theta) + 1j * np.sin(theta)
    # Raise error when radiation pattern is complex
    with pytest.raises(ValidationError), pytest.warns(np.exceptions.ComplexWarning):
        mw.LobeMeasurer(
            angle=theta,
            radiation_pattern=Urad,
        )

    theta = np.linspace(-np.pi, np.pi, 101, endpoint=False)
    Urad = np.cos(theta) ** 2
    # No error when cyclic extension is disabled
    mw.LobeMeasurer(angle=theta, radiation_pattern=Urad, apply_cyclic_extension=False)

    # Raise error when cyclic extension is enabled and angle array is not in [0, 2π)
    with pytest.raises(ValidationError):
        mw.LobeMeasurer(
            angle=theta,
            radiation_pattern=Urad,
        )

    theta = np.linspace(-np.pi, np.pi, 101, endpoint=False)
    theta[10] = theta[75]
    Urad = np.cos(theta) ** 2
    # Make sure array is sorted
    with pytest.raises(ValidationError):
        mw.LobeMeasurer(angle=theta, radiation_pattern=Urad, apply_cyclic_extension=False)


@pytest.mark.parametrize("apply_cyclic_extension", [True, False])
@pytest.mark.parametrize("include_endpoint", [True, False])
def test_lobe_measurements(apply_cyclic_extension, include_endpoint):
    """Run the lobe measurer on some test data and check the results."""

    # Example 2.4 from Antenna Theory by Balanis
    theta = np.linspace(0, 2 * np.pi, 301, endpoint=include_endpoint)
    Urad_raw = np.cos(theta) ** 2 * np.cos(3 * theta) ** 2

    angular_tol = 2 * np.pi / len(theta)

    # Smooth out small variations using vectorized operations
    # This will help check if the lobe measurer correctly places peaks
    # at the center of any plateau regions
    tolerance = 1e-7
    diffs = np.abs(np.diff(Urad_raw))
    mask = np.append(False, diffs < tolerance)  # False for first element to maintain array size
    Urad = np.where(mask, np.roll(Urad_raw, 1), Urad_raw)  # Replace small variations

    # Check against easy numpy peak finding
    if apply_cyclic_extension:
        max_lobe_loc = np.argmax(Urad)
        max_lobe_level = Urad[max_lobe_loc]
    else:
        # When the signal is not cyclically extended, the peak finding
        # will not find the first peak near the boundary
        max_lobe_loc = np.argmax(Urad[1:]) + 1
        max_lobe_level = Urad[max_lobe_loc]

    lobe_measurer = mw.LobeMeasurer(
        angle=theta,
        radiation_pattern=Urad,
        width_measure=0.5,
        apply_cyclic_extension=apply_cyclic_extension,
    )
    lobe_measures = lobe_measurer.lobe_measures
    num_rows, num_cols = lobe_measures.shape
    assert num_cols == 7
    # Check the number of lobes found
    if apply_cyclic_extension:
        assert num_rows == 6
    else:
        assert num_rows == 5

    assert isclose(lobe_measurer.main_lobe["magnitude"], max_lobe_level)
    assert isclose(lobe_measurer.main_lobe["direction"], theta[max_lobe_loc], abs_tol=angular_tol)
    assert isclose(lobe_measurer.main_lobe["beamwidth"], 0.5, abs_tol=angular_tol)
    assert isclose(lobe_measurer.main_lobe["FNBW"], np.pi / 3, abs_tol=angular_tol)

    sidelobe_level = lobe_measurer.sidelobe_level

    # The side lobe level will not be accurate for this pattern
    # when using cyclic extension, since there are two main lobes with similar magnitude
    if not apply_cyclic_extension:
        assert isclose(sidelobe_level, 0.3167, rel_tol=1e-2)

    # Plot the radiation pattern and the lobe measures
    # for debugging purposes
    # These additional lobe measures are useful for plotting
    # the calculated beamwidths.
    # x_start = lobe_measures["beamwidth bounds"][0]
    # x_end = lobe_measures["beamwidth bounds"]
    # width_heights = lobe_measures["beamwidth magnitude"]
    # import matplotlib.pyplot as plt
    # from matplotlib import use
    # use("TkAgg")
    # _, ax = plt.subplots(1, 1, tight_layout=True)
    # ax.plot(theta, Urad)
    # ax.plot(lobe_measures["direction"], lobe_measures["magnitude"], "x")
    # for index, _ in lobe_measures.iterrows():
    #     x = [x_start[index], x_end[index]]
    #     y = [width_heights[index], width_heights[index]]
    #     ax.plot(x, y, "-r")
    # plt.show()

    # Now test when no lobes are found
    lobe_measurer = mw.LobeMeasurer(
        angle=theta,
        radiation_pattern=np.zeros_like(Urad),
        width_measure=0.5,
        apply_cyclic_extension=apply_cyclic_extension,
    )
    assert lobe_measurer.main_lobe.empty
    lobe_measures = lobe_measurer.lobe_measures
    assert lobe_measures.empty
    sidelobe_level = lobe_measurer.sidelobe_level
    assert sidelobe_level is None


@pytest.mark.parametrize("min_value", [0.0, 1.0])
def test_lobe_plots(min_value):
    """Run the lobe measurer on some test data and plot the results."""
    theta = np.linspace(0, 2 * np.pi, 301)
    Urad = np.cos(theta) ** 2 * np.cos(3 * theta) ** 2 + min_value
    lobe_measurer = mw.LobeMeasurer(angle=theta, radiation_pattern=Urad)
    _, ax = plt.subplots(1, 1, subplot_kw={"projection": "polar"})
    ax.plot(theta, Urad, "k")
    lobe_measurer.plot(0, ax)
    if MAKE_PLOTS:
        plt.show()
