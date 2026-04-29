"""Tests mode objects."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pytest
from pydantic import ValidationError

import tidy3d as td

from ..utils import assert_single_value_error_loc


def test_apodization():
    _ = td.ApodizationSpec(width=0.2)
    _ = td.ApodizationSpec(start=1, width=0.2)
    _ = td.ApodizationSpec(end=2, width=0.2)
    _ = td.ApodizationSpec(start=1, end=2, width=0.2)


def test_end_lt_start():
    with pytest.raises(ValidationError) as excinfo:
        _ = td.ApodizationSpec(start=2, end=1, width=0.2)
    assert_single_value_error_loc(excinfo, ("end",), "End apodization begins before start")


def test_no_width():
    with pytest.raises(ValidationError) as excinfo:
        _ = td.ApodizationSpec(start=1, end=2)
    assert_single_value_error_loc(excinfo, ("width",), "Apodization width must be set")
    with pytest.raises(ValidationError) as excinfo:
        _ = td.ApodizationSpec(start=1)
    assert_single_value_error_loc(excinfo, ("width",), "Apodization width must be set")
    with pytest.raises(ValidationError) as excinfo:
        _ = td.ApodizationSpec(end=2)
    assert_single_value_error_loc(excinfo, ("width",), "Apodization width must be set")


def test_negative_times():
    with pytest.raises(ValidationError):
        _ = td.ApodizationSpec(start=-2, end=-1, width=0.2)

    with pytest.raises(ValidationError):
        _ = td.ApodizationSpec(start=1, end=2, width=-0.2)

    with pytest.raises(ValidationError):
        _ = td.ApodizationSpec(start=1, end=2, width=0)


def test_plot():
    run_time = 1.0e-13
    times = [0, 2.0e-14, 4.0e-14, 6.0e-14, 8.0e-14, 1.0e-13]

    a = td.ApodizationSpec(start=0.2 * run_time, end=0.8 * run_time, width=0.02 * run_time)
    a.plot(times)
    plt.close()

    _fig, ax = plt.subplots(1, 1)
    a.plot(times, ax=ax)
    plt.close()
