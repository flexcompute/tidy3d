from __future__ import annotations

import ssl

import pytest

from tidy3d.config.sections import WebConfig


def test_build_api_url_joins_paths():
    web = WebConfig(api_endpoint="https://example.com/api")
    assert web.build_api_url("v1/tasks") == "https://example.com/api/v1/tasks"


def test_build_api_url_strips_leading_slashes():
    web = WebConfig(api_endpoint="https://example.com/api/")
    assert web.build_api_url("/v1/tasks") == "https://example.com/api/v1/tasks"


def test_build_api_url_returns_base_for_empty_path():
    web = WebConfig(api_endpoint="https://example.com/api")
    assert web.build_api_url("") == "https://example.com/api"


def test_build_api_url_without_base_returns_path():
    web = WebConfig.model_construct(api_endpoint="")
    assert web.build_api_url("/v1/tasks") == "v1/tasks"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (ssl.TLSVersion.TLSv1, "TLSv1"),
        (ssl.TLSVersion.TLSv1_1, "TLSv1_1"),
    ],
)
def test_web_config_normalizes_ssl_version_aliases(value, expected):
    web = WebConfig(ssl_version=value)
    assert web.ssl_version == expected


@pytest.mark.parametrize("value", ["", "TLSv2", "SSLv3", "udp1.0"])
def test_web_config_rejects_invalid_ssl_version(value):
    with pytest.raises(ValueError):
        WebConfig(ssl_version=value)
