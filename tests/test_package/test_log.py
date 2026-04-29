"""Test the logging."""

from __future__ import annotations

import json

import numpy as np
import pytest
from pydantic import ValidationError, model_validator

import tidy3d as td
from tidy3d.exceptions import Tidy3dError
from tidy3d.log import DEFAULT_LEVEL, _get_level_int, set_logging_level

from ..utils import AssertLogLevel


def test_log():
    td.log.debug("debug test")
    td.log.info("info test")
    td.log.warning("warning test")
    td.log.error("error test")
    td.log.critical("critical test")
    td.log.log(0, "zero test")


def test_log_config(tmp_path):
    td.config.logging.level = "DEBUG"
    td.set_logging_file(str(tmp_path / "test.log"))
    assert len(td.log.handlers) == 2
    assert td.log.handlers["console"].level == _get_level_int("DEBUG")
    assert td.log.handlers["file"].level == _get_level_int(DEFAULT_LEVEL)
    del td.log.handlers["file"]


def test_log_level_not_found():
    with pytest.raises(ValueError):
        set_logging_level("NOT_A_LEVEL")


def test_set_logging_level_deprecated():
    with pytest.raises(DeprecationWarning):
        td.set_logging_level("WARNING")


def test_exception_message():
    MESSAGE = "message"
    e = Tidy3dError(MESSAGE)
    assert str(e) == MESSAGE


def test_logging_upper():
    """Make sure we get an error if lowercase."""
    td.config.logging.level = "WARNING"
    with pytest.raises(ValidationError):
        td.config.logging.level = "warning"


def test_logging_unrecognized():
    """If unrecognized option, raise validation error."""
    with pytest.raises(ValidationError):
        td.config.logging.level = "blah"


def test_logging_warning_capture():
    # create sim with warnings
    domain_size = 12

    td.log.set_capture(True)
    wavelength = 1
    f0 = td.C_0 / wavelength
    fwidth = f0 / 10.0
    source_time = td.GaussianPulse(freq0=f0, fwidth=fwidth)
    freqs = np.linspace(f0 - fwidth, f0 + fwidth, 11)

    # 1 warning: too long run_time
    run_time = 10000 / fwidth

    # 2 warnings: frequency outside of source frequency range; too many points
    mode_mnt = td.ModeMonitor(
        center=(0, 0, 0),
        size=(domain_size, 0, domain_size),
        # additional frequency is outside the source range, but is inside the allowed validator range
        freqs=[*list(freqs), 0.1 * f0],
        mode_spec=td.ModeSpec(num_modes=3),
        name="mode",
    )

    # 1 warning: too many points
    mode_source = td.ModeSource(
        size=(domain_size, 0, domain_size),
        source_time=source_time,
        mode_spec=td.ModeSpec(num_modes=2, precision="single"),
        mode_index=1,
        num_freqs=10,
        direction="-",
    )

    # 1 warning: ignoring "normal_dir"
    monitor_flux = td.FluxMonitor(
        center=(0, 0, 0),
        size=(8, 8, 8),
        freqs=list(freqs),
        name="flux",
        normal_dir="+",
    )

    # 1 warning: large monitor size
    monitor_time = td.FieldTimeMonitor(
        center=(0, 0, 0),
        size=(2, 2, 2),
        stop=1 / fwidth,
        name="time",
    )

    # 2 warnings * 4 sources = 8 total: too close to each PML
    # 1 warning * 2 DFT monitors = 2 total: medium frequency range does not cover monitors freqs
    box = td.Structure(
        geometry=td.Box(center=(0, 0, 0), size=(11.5, 11.5, 11.5)),
        medium=td.Medium(permittivity=2, frequency_range=[0.5, 1]),
    )

    # 1 warning: inside pml
    box_in_pml = td.Structure(
        geometry=td.Box(center=(0, 0, 0), size=(domain_size * 1.001, 5, 5)),
        medium=td.Medium(permittivity=10),
    )

    # 2 warnings: exactly on sim edge
    box_on_boundary = td.Structure(
        geometry=td.Box(center=(0, 0, 0), size=(domain_size, 5, 5)),
        medium=td.Medium(permittivity=20),
    )

    # 1 warning: outside of domain
    box_outside = td.Structure(
        geometry=td.Box(center=(50, 0, 0), size=(domain_size, 5, 5)),
        medium=td.Medium(permittivity=6),
    )

    # 1 warning: glancing angle
    gaussian_beam = td.GaussianBeam(
        center=(4, 0, 0),
        size=(0, 2, 1),
        waist_radius=2.0,
        waist_distance=1,
        source_time=source_time,
        direction="+",
        angle_theta=np.pi / 2.1,
    )

    plane_wave = td.PlaneWave(
        center=(4, 0, 0),
        size=(0, 1, 2),
        source_time=source_time,
        direction="+",
    )

    # 2 warnings: non-uniform grid along y and z
    tfsf = td.TFSF(
        size=(10, 15, 15),
        source_time=source_time,
        direction="-",
        injection_axis=0,
    )

    # 1 warning: bloch boundary is inconsistent with plane_wave
    bspec = td.BoundarySpec(
        x=td.Boundary.pml(),
        y=td.Boundary.periodic(),
        z=td.Boundary.bloch(bloch_vec=0.2),
    )

    # 1 warning * 1 structures (perm=20) * 4 sources = 20 total: large grid step along x
    gspec = td.GridSpec(
        grid_x=td.UniformGrid(dl=0.05),
        grid_y=td.AutoGrid(min_steps_per_wvl=15),
        grid_z=td.AutoGrid(min_steps_per_wvl=15),
        override_structures=[
            td.Structure(geometry=td.Box(size=(3, 2, 1)), medium=td.Medium(permittivity=4))
        ],
    )

    sim = td.Simulation(
        size=[domain_size, 20, 20],
        sources=[gaussian_beam, mode_source, plane_wave, tfsf],
        structures=[box, box_in_pml, box_on_boundary, box_outside],
        monitors=[monitor_flux, mode_mnt],
        run_time=run_time,
        boundary_spec=bspec,
        grid_spec=gspec,
    )

    # parse the entire simulation at once to capture warnings hierarchically
    sim_dict = sim.model_dump()

    sim = td.Simulation.model_validate(sim_dict)
    print(sim.monitors_data_size)
    sim.validate_pre_upload()
    warning_list = td.log.captured_warnings()
    print(json.dumps(warning_list, indent=4))
    # assert len(warning_list) >= 29
    # TODO FIXME
    td.log.set_capture(False)

    # check that capture doesn't change validation errors

    # validation error during model_validate()
    sim_dict_no_source = sim.model_dump()
    sim_dict_no_source.update({"sources": []})

    # validation error during validate_pre_upload()
    sim_dict_large_mnt = sim.model_dump()
    sim_dict_large_mnt.update({"monitors": [monitor_time.updated_copy(size=(10, 10, 10))]})

    # for sim_dict in [sim_dict_no_source, sim_dict_large_mnt]:
    for sim_dict in [sim_dict_no_source]:
        try:
            sim = td.Simulation.model_validate(sim_dict)
            sim.validate_pre_upload()
        except ValidationError as e:
            error_without = e.errors()
        except Exception as e:
            error_without = str(e)

        td.log.set_capture(True)
        try:
            sim = td.Simulation.model_validate(sim_dict)
            sim.validate_pre_upload()
        except ValidationError as e:
            error_with = e.errors()
        except Exception as e:
            error_with = str(e)
        td.log.set_capture(False)

        assert str(error_without) == str(error_with)


def test_warning_capture_during_model_validation():
    from tidy3d.components.base import Tidy3dBaseModel
    from tidy3d.log import log

    class _CaptureChild(Tidy3dBaseModel):
        x: int

        @model_validator(mode="after")
        def _warn_child(self):
            log.warning("child warning")
            return self

    class _CaptureParent(Tidy3dBaseModel):
        child: _CaptureChild

        @model_validator(mode="after")
        def _warn_parent(self):
            log.warning("parent warning")
            return self

    td.log.set_capture(True)
    _CaptureParent(child={"x": 1})
    warning_list = td.log.captured_warnings()
    td.log.set_capture(False)

    assert {"loc": [], "msg": "parent warning"} in warning_list
    assert {"loc": ["child"], "msg": "child warning"} in warning_list


def test_log_suppression():
    with td.log as suppressed_log:
        assert td.log._counts is not None
        for _ in range(4):
            suppressed_log.warning("Warning message")
        assert td.log._counts[30] == 3

    td.config.logging.suppression = False
    with td.log as suppressed_log:
        assert td.log._counts is None
        for _ in range(4):
            suppressed_log.warning("Warning message")
        assert td.log._counts is None

    td.config.logging.suppression = True


def test_warn_once():
    """Test that warn_once setting causes each unique warning to only be shown once."""

    # Clear the static cache to ensure clean test state
    td.log._static_cache.clear()

    # By default, warn_once should be False
    assert td.log.warn_once is False
    assert td.config.logging.warn_once is False

    # Enable warn_once via config
    td.config.logging.warn_once = True
    assert td.log.warn_once is True

    # First warning should go through
    initial_cache_size = len(td.log._static_cache)
    td.log.warning("unique_test_warning_message_1234")
    assert len(td.log._static_cache) == initial_cache_size + 1
    assert "unique_test_warning_message_1234" in td.log._static_cache

    # Same warning should be skipped (cache doesn't grow)
    td.log.warning("unique_test_warning_message_1234")
    assert len(td.log._static_cache) == initial_cache_size + 1

    # Different warning should go through
    td.log.warning("different_warning_message_5678")
    assert len(td.log._static_cache) == initial_cache_size + 2
    assert "different_warning_message_5678" in td.log._static_cache

    # Info messages should NOT be affected by warn_once
    td.log.info("info_message_should_not_cache")
    td.log.info("info_message_should_not_cache")
    # Info messages don't use the cache when warn_once is enabled (only warnings do)
    assert "info_message_should_not_cache" not in td.log._static_cache

    # Error messages should NOT be affected by warn_once
    td.log.error("error_message_should_not_cache")
    td.log.error("error_message_should_not_cache")
    assert "error_message_should_not_cache" not in td.log._static_cache

    # Critical messages should NOT be affected by warn_once
    td.log.critical("critical_message_should_not_cache")
    td.log.critical("critical_message_should_not_cache")
    assert "critical_message_should_not_cache" not in td.log._static_cache

    # Disable warn_once
    td.config.logging.warn_once = False
    assert td.log.warn_once is False

    # Clear cache for cleanup
    td.log._static_cache.clear()


def test_assert_log_level():
    """Test features of the assert_log_level"""

    # log was captured
    with AssertLogLevel("WARNING", contains_str="ABC"):
        td.log.warning("ABC")

    # Test when log message doesn't contain expected string
    with pytest.raises(AssertionError), AssertLogLevel("WARNING", contains_str="DEF"):
        td.log.warning("ABC")

    # Test when log message is at incorrect level
    with pytest.raises(AssertionError), AssertLogLevel("WARNING", contains_str="ABC"):
        td.log.info("ABC")  # Should fail since INFO < WARNING

    # Test when log level is higher than expected
    with pytest.raises(AssertionError), AssertLogLevel("INFO"):
        td.log.warning("ABC")  # Should fail since WARNING > INFO


def test_suppress_output():
    """Test that suppress_output() context manager prevents log messages from being emitted."""
    from ..utils import AssertLogLevelHandler

    # Register a handler to capture log records
    handler = AssertLogLevelHandler()
    td.log.handlers["test_suppress"] = handler

    try:
        # Without suppression, messages should be captured
        td.log.warning("visible warning")
        td.log.error("visible error")
        assert len(handler.records) == 2

        # With suppression, messages should not be captured
        with td.log.suppress_output():
            td.log.warning("suppressed warning")
            td.log.error("suppressed error")
            td.log.info("suppressed info")

        # Still only 2 records from before
        assert len(handler.records) == 2

        # After exiting context, messages should be captured again
        td.log.warning("visible again")
        assert len(handler.records) == 3

        # Nested suppression should work correctly
        with td.log.suppress_output():
            td.log.warning("outer suppressed")
            with td.log.suppress_output():
                td.log.warning("inner suppressed")
            td.log.warning("still outer suppressed")

        # Still only 3 records
        assert len(handler.records) == 3

    finally:
        del td.log.handlers["test_suppress"]


def test_suppress_output_during_repr():
    """Test that suppress_output prevents spurious errors during repr of models that can't be default-instantiated.

    This tests the scenario where:
    1. A model has optional fields with defaults (so Pydantic can attempt to instantiate it)
    2. A validator raises SetupError when instantiated with defaults (which logs an error)
    3. The repr optimization tries to create a default instance to compare against
    4. Without suppress_output, the error would be logged even though it's caught

    This mirrors what happens with CustomMedium (permittivity=None by default, but validator
    requires either permittivity or eps_dataset to be provided).
    """

    from pydantic import model_validator

    from tidy3d.components.base import Tidy3dBaseModel
    from tidy3d.exceptions import SetupError

    from ..utils import AssertLogLevelHandler

    # Create a model that:
    # - Has optional field with None default (so Pydantic can instantiate it)
    # - Raises SetupError in validator when value is None (which logs an error before raising)
    class _ModelRequiringValue(Tidy3dBaseModel):
        value: float | None = None

        @model_validator(mode="after")
        def _check_value(self):
            if self.value is None:
                raise SetupError("test error: value cannot be None")
            return self

    # Verify the model raises when instantiated with no args
    # (SetupError is wrapped in Pydantic's ValidationError)
    with pytest.raises(ValidationError, match="value cannot be None"):
        _ModelRequiringValue()

    # Register a handler to capture log records
    handler = AssertLogLevelHandler()
    td.log.handlers["test_repr"] = handler

    try:
        # Create a valid instance (with value provided)
        model = _ModelRequiringValue(value=1.0)

        # repr() internally tries to create _ModelRequiringValue() to compare defaults.
        # This fails with SetupError, which logs an error before raising.
        # The suppress_output context manager should prevent this error from appearing.
        _ = repr(model)

        # Check that no ERROR level messages were logged
        error_records = [r for r in handler.records if r[0] >= _get_level_int("ERROR")]
        assert len(error_records) == 0, f"Unexpected error logs during repr: {error_records}"

    finally:
        del td.log.handlers["test_repr"]
