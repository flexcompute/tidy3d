from __future__ import annotations

import numpy as np
import pytest

from tidy3d.config import Env, get_manager, reload_config


def test_default_web_settings(config_manager):
    web = config_manager.get_section("web")
    assert str(web.api_endpoint) == "https://tidy3d-api.simulation.cloud"
    assert str(web.website_endpoint) == "https://tidy3d.simulation.cloud"
    assert web.ssl_verify is True


def test_update_section_runtime_overlay(config_manager):
    config_manager.update_section("logging", level="DEBUG", suppression=False)
    logging_section = config_manager.get_section("logging")
    assert logging_section.level == "DEBUG"
    assert logging_section.suppression is False


def test_runtime_isolated_per_profile(config_manager):
    config_manager.update_section("web", timeout=45)
    config_manager.switch_profile("customer")
    assert config_manager.get_section("web").timeout == 120
    config_manager.switch_profile("default")
    assert config_manager.get_section("web").timeout == 45


def test_runtime_overrides_env(monkeypatch, config_manager):
    monkeypatch.setenv("TIDY3D_LOGGING__LEVEL", "WARNING")
    config_manager.switch_profile(config_manager.profile)
    config_manager.update_section("logging", level="DEBUG")
    logging_section = config_manager.get_section("logging")
    # runtime change should override the environment variable
    assert logging_section.level == "DEBUG"


def test_env_applies_without_runtime_override(monkeypatch, config_manager):
    monkeypatch.setenv("TIDY3D_LOGGING__LEVEL", "WARNING")
    config_manager.switch_profile(config_manager.profile)
    logging_section = config_manager.get_section("logging")
    assert logging_section.level == "WARNING"


@pytest.mark.parametrize("profile", ["dev", "uat"])
def test_builtin_profiles(profile, config_manager):
    config_manager.switch_profile(profile)
    web = config_manager.get_section("web")
    assert web.s3_region is not None


def test_uppercase_profile_normalization(monkeypatch):
    monkeypatch.setenv("TIDY3D_ENV", "DEV")
    try:
        reload_config()
        manager = get_manager()
        assert manager.profile == "dev"
        web = manager.get_section("web")
        assert str(web.api_endpoint) == "https://tidy3d-api.dev-simulation.cloud"
        assert Env.current.name == "dev"
    finally:
        reload_config(profile="default")


def test_adjoint_defaults(config_manager):
    adjoint = config_manager.get_section("adjoint")
    assert adjoint.min_wvl_fraction == pytest.approx(5e-2)
    assert adjoint.points_per_wavelength == 10
    assert adjoint.monitor_interval_poly == (1, 1, 1)
    assert adjoint.quadrature_sample_fraction == pytest.approx(0.4)
    assert adjoint.gauss_quadrature_order == 7
    assert adjoint.edge_clip_tolerance == pytest.approx(1e-9)
    assert adjoint.minimum_spacing_fraction == pytest.approx(1e-2)
    assert adjoint.gradient_precision == "single"
    assert adjoint.max_traced_structures == 500
    assert adjoint.max_adjoint_per_fwd == 10


def test_adjoint_update_section(config_manager):
    config_manager.update_section(
        "adjoint",
        min_wvl_fraction=0.08,
        points_per_wavelength=12,
        solver_freq_chunk_size=3,
        gradient_precision="double",
        minimum_spacing_fraction=0.02,
        gauss_quadrature_order=5,
        edge_clip_tolerance=2e-9,
        max_traced_structures=600,
        max_adjoint_per_fwd=7,
    )
    adjoint = config_manager.get_section("adjoint")
    assert adjoint.min_wvl_fraction == pytest.approx(0.08)
    assert adjoint.points_per_wavelength == 12
    assert adjoint.solver_freq_chunk_size == 3
    assert adjoint.gauss_quadrature_order == 5
    assert adjoint.edge_clip_tolerance == pytest.approx(2e-9)
    assert adjoint.minimum_spacing_fraction == pytest.approx(0.02)
    assert adjoint.gradient_precision == "double"
    assert adjoint.max_traced_structures == 600
    assert adjoint.max_adjoint_per_fwd == 7

    assert adjoint.gradient_dtype_float is np.float64
    assert adjoint.gradient_dtype_complex is np.complex128


def test_config_str_formatting(config_manager):
    text = str(config_manager)
    assert "Config (profile='default')" in text
    assert "├── adjoint" in text
    assert "├── logging" in text
    assert "└── web" in text
    assert "'api_endpoint': 'https://tidy3d-api.simulation.cloud'" in text
    assert "'s3_region': 'us-gov-west-1'" in text


def test_section_accessor_str_formatting(config_manager):
    text = str(config_manager.adjoint)
    assert "adjoint" in text
    assert "'gradient_precision': 'single'" in text
    assert "'monitor_interval_poly': [" in text


def test_as_dict_includes_defaults(config_manager):
    data = config_manager.as_dict()
    assert "logging" in data
    assert data["logging"]["level"] == "WARNING"
    assert "adjoint" in data
    assert data["adjoint"]["local_adjoint_dir"] == "adjoint_data"
    assert "simulation" in data
