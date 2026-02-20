from __future__ import annotations

import builtins

import pytest

from tidy3d.config import config, reload_config
from tidy3d.packaging import (
    Tidy3dImportError,
    check_import,
    check_tidy3d_extras_licensed_feature,
    supports_local_subpixel,
    tidy3d_extras,
    verify_packages_import,
)

assert check_import("tidy3d") is True


@pytest.fixture(autouse=True)
def _reset_tidy3d_extras():
    """Reinitialize tidy3d_extras config before and after every test."""
    tidy3d_extras["mod"] = None
    tidy3d_extras["use_local_subpixel"] = None
    reload_config(profile="default")
    yield
    tidy3d_extras["mod"] = None
    tidy3d_extras["use_local_subpixel"] = None
    reload_config(profile="default")


# Mock module import function to simulate availability
def mock_check_import(module_name):
    """
    Mock the check_import function to simulate availability of tidy3d module.
    """
    if module_name == "tidy3d":
        return True
    return False


def test_verify_packages_import_all_required():
    """
    Test the verify_packages_import function with all required. Verifies that the decorator works to trigger a
    Tidy3dImportError when module2 is unavailable.
    """

    @verify_packages_import(["tidy3d", "module2"], required="all")
    def my_function():
        pass

    with pytest.raises(Tidy3dImportError):
        my_function()


def test_verify_packages_import_either_required():
    """
    Test the verify_packages_import function with either required. Verifies the decorator works by not triggering an
    error when the module2 is not found. However it should throw an error when either module2 or module3 are found.
    """

    @verify_packages_import(["tidy3d", "module2"], required="any")
    def my_function():
        pass

    # When at least one module is imported, it should not raise an error
    my_function()

    @verify_packages_import(["module2", "module3"], required="any")
    def my_function2():
        pass

    with pytest.raises(Tidy3dImportError):
        my_function2()


def test_check_import():
    """
    Test the check_import function with mock_check_import. Just standard test to verify the mock function works
    compared to check_import.
    """
    import sys

    sys.modules["tidy3d"].check_import = mock_check_import

    assert mock_check_import("tidy3d") is True
    assert mock_check_import("module2") is False


def test_tidy3d_extras():
    import importlib

    has_tidy3d_extras = importlib.util.find_spec("tidy3d_extras") is not None
    print(f"has_tidy3d_extras = {has_tidy3d_extras}")

    @supports_local_subpixel
    def get_eps():
        if has_tidy3d_extras:
            assert tidy3d_extras["mod"] is not None
            features = tidy3d_extras["mod"].extension._features()
            assert tidy3d_extras["use_local_subpixel"] == ("local_subpixel" in features)
            if tidy3d_extras["use_local_subpixel"]:
                check_tidy3d_extras_licensed_feature("local_subpixel")
        else:
            print(f"has_tidy3d_extras = {has_tidy3d_extras}")
            assert tidy3d_extras["use_local_subpixel"] is False
            assert tidy3d_extras["mod"] is None

    get_eps()


def test_broadband_feature_raises_when_extras_missing(monkeypatch):
    """check_tidy3d_extras_licensed_feature must raise when tidy3d-extras is absent."""
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "tidy3d_extras":
            raise ImportError("forced failure")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(Tidy3dImportError, match="tidy3d-extras"):
        check_tidy3d_extras_licensed_feature("BroadbandPulse")


def test_broadband_feature_raises_when_not_licensed():
    """check_tidy3d_extras_licensed_feature must raise when feature is not in the license."""
    from types import SimpleNamespace

    mock_extension = SimpleNamespace(_features=lambda: {"other_feature"})
    tidy3d_extras["mod"] = SimpleNamespace(extension=mock_extension)

    with pytest.raises(Tidy3dImportError, match="not available with your license"):
        check_tidy3d_extras_licensed_feature("BroadbandPulse")


def test_broadband_feature_succeeds_when_licensed():
    """check_tidy3d_extras_licensed_feature must not raise when feature is licensed."""
    from types import SimpleNamespace

    mock_extension = SimpleNamespace(_features=lambda: {"BroadbandPulse"})
    tidy3d_extras["mod"] = SimpleNamespace(extension=mock_extension)

    check_tidy3d_extras_licensed_feature("BroadbandPulse")


def test_supports_local_subpixel_respects_config_false():
    tidy3d_extras["mod"] = object()
    tidy3d_extras["use_local_subpixel"] = True

    config.update_section("simulation", use_local_subpixel=False)

    @supports_local_subpixel
    def get_flag():
        return tidy3d_extras["use_local_subpixel"]

    assert get_flag() is False


def test_supports_local_subpixel_requires_extras_when_forced(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "tidy3d_extras":
            raise ImportError("forced failure")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    config.update_section("simulation", use_local_subpixel=True)

    @supports_local_subpixel
    def get_flag():
        return tidy3d_extras["use_local_subpixel"]

    with pytest.raises(Tidy3dImportError):
        get_flag()


def test_supports_local_subpixel_no_error_logged_when_optional(monkeypatch, caplog):
    """Regression test for FXC-4374: no ERROR should be logged when preference=None
    and tidy3d-extras is unavailable, since the decorator handles it gracefully."""
    import logging

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "tidy3d_extras":
            raise ImportError("forced failure")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    # preference=None is the default - feature is optional
    config.update_section("simulation", use_local_subpixel=None)

    @supports_local_subpixel
    def get_flag():
        return tidy3d_extras["use_local_subpixel"]

    with caplog.at_level(logging.ERROR):
        result = get_flag()

    # Should fall back gracefully without logging errors
    assert result is False
    assert not any("tidy3d-extras" in record.message for record in caplog.records), (
        "ERROR was logged but should have been suppressed for optional feature check"
    )


def test_solve_warning_suppressed_when_subpixel_enabled():
    """The accuracy warning in ModeSolver.solve() should not fire when local subpixel is active."""
    from types import SimpleNamespace

    from tidy3d.log import log

    # Build a mock module that satisfies _check_tidy3d_extras_available (mod is not
    # None ⇒ early return) and check_tidy3d_extras_licensed_feature (needs
    # mod.extension._features() containing "local_subpixel").
    mock_extension = SimpleNamespace(_features=lambda: {"local_subpixel"})
    mock_mod = SimpleNamespace(extension=mock_extension)

    tidy3d_extras["mod"] = mock_mod

    _LogCapture = type(
        "_LogCapture", (), {"records": [], "handle": lambda s, *a: s.records.append(a)}
    )
    handler = _LogCapture()
    log.handlers["_test_capture"] = handler

    try:
        config.update_section("simulation", use_local_subpixel=True)

        @supports_local_subpixel
        def _guarded():
            from tidy3d.packaging import tidy3d_extras as _te

            # Mimics the warning guard in ModeSolver.solve()
            if not _te["use_local_subpixel"]:
                log.warning("remote mode solver", log_once=True)

        _guarded()
        assert not any("remote mode solver" in str(r) for r in handler.records), (
            "Accuracy warning should be suppressed when local subpixel is enabled"
        )
    finally:
        del log.handlers["_test_capture"]


def test_solve_warning_emitted_when_subpixel_disabled(monkeypatch):
    """The accuracy warning in ModeSolver.solve() should fire when local subpixel is off."""
    from tidy3d.log import log

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "tidy3d_extras":
            raise ImportError("forced failure")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    _LogCapture = type(
        "_LogCapture", (), {"records": [], "handle": lambda s, *a: s.records.append(a)}
    )
    handler = _LogCapture()
    log.handlers["_test_capture"] = handler

    try:
        config.update_section("simulation", use_local_subpixel=None)

        @supports_local_subpixel
        def _guarded():
            from tidy3d.packaging import tidy3d_extras as _te

            # Mimics the warning guard in ModeSolver.solve()
            if not _te["use_local_subpixel"]:
                log.warning("remote mode solver", log_once=True)

        _guarded()
        assert any("remote mode solver" in str(r) for r in handler.records), (
            "Accuracy warning should be emitted when local subpixel is unavailable"
        )
    finally:
        del log.handlers["_test_capture"]


if __name__ == "__main__":
    pytest.main()
