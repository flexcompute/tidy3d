"""Built-in configuration section schemas and handlers."""

from __future__ import annotations

import ssl
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PositiveInt,
    SecretStr,
    field_serializer,
    field_validator,
)

from tidy3d.log import DEFAULT_LEVEL, LogLevel, log, set_log_suppression, set_logging_level

from .registry import get_manager as _get_attached_manager
from .registry import register_handler, register_section


class ConfigSection(BaseModel):
    """Base class for configuration sections."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    def to_dict(self, *, mask_secrets: bool = True) -> dict[str, Any]:
        """Convert section to a serializable dictionary."""

        data = self.model_dump(exclude_unset=True)
        if mask_secrets:
            return data

        unmasked: dict[str, Any] = {}
        for key, value in data.items():
            if isinstance(value, SecretStr):
                unmasked[key] = value.get_secret_value()
            else:
                unmasked[key] = value
        return unmasked


@register_section("logging")
class LoggingConfig(ConfigSection):
    """Logging configuration."""

    level: LogLevel = Field(
        DEFAULT_LEVEL,
        title="Logging level",
        description="Lowest logging level that will be emitted.",
        json_schema_extra={"persist": True},
    )

    suppression: bool = Field(
        True,
        title="Log suppression",
        description="Suppress repeated log messages when True.",
    )


@register_handler("logging")
def apply_logging(config: LoggingConfig) -> None:
    """Apply logging configuration globally."""

    set_logging_level(config.level)
    set_log_suppression(config.suppression)


@register_section("simulation")
class SimulationConfig(ConfigSection):
    """Simulation-related configuration."""

    use_local_subpixel: bool | None = Field(
        None,
        title="Use local subpixel",
        description=(
            "If True, force local subpixel averaging; False disables it; None keeps default behavior."
        ),
    )


@register_section("microwave")
class MicrowaveConfig(ConfigSection):
    """Microwave solver configuration."""

    suppress_rf_license_warning: bool = Field(
        False,
        title="Suppress RF license warning",
        description="If true, do not emit microwave license availability warnings.",
    )


@register_section("adjoint")
class AdjointConfig(ConfigSection):
    """Adjoint (autograd) configuration section."""

    min_wvl_fraction: float = Field(
        5e-2,
        title="Minimum wavelength fraction",
        description=(
            "Minimum fraction of the smallest free-space wavelength used when discretizing "
            "cylindrical structures for autograd derivatives."
        ),
        ge=0.0,
    )

    points_per_wavelength: PositiveInt = Field(
        10,
        title="Points per wavelength",
        description=(
            "Default number of material sample points per wavelength when discretizing "
            "cylinders for autograd derivatives."
        ),
    )

    default_wavelength_fraction: float = Field(
        0.1,
        title="Default wavelength fraction",
        description=(
            "Fallback fraction of the minimum wavelength used when autograd needs to "
            "estimate adaptive spacing."
        ),
        ge=0.0,
    )

    minimum_spacing_fraction: float = Field(
        1e-2,
        title="Minimum spacing fraction",
        description=(
            "Minimum normalized spacing allowed when constructing adaptive finite-difference "
            "stencils for autograd evaluations."
        ),
        ge=0.0,
    )

    local_gradient: bool = Field(
        False,
        title="Enable local gradients",
        description=(
            "When True, autograd runs download intermediate data and compute gradients locally. "
            "Remote (default) gradients always use server-side limits regardless of other settings."
        ),
        json_schema_extra={"persist": True},
    )

    local_adjoint_dir: Path = Field(
        Path("adjoint_data"),
        title="Local gradient directory",
        description=(
            "Relative directory name used to store intermediate results when local gradients are enabled."
        ),
        json_schema_extra={"persist": True},
    )

    gradient_precision: Literal["single", "double"] = Field(
        "single",
        title="Gradient precision",
        description="Floating-point precision used for autograd gradient calculations.",
    )

    monitor_interval_poly: tuple[int, int, int] = Field(
        (1, 1, 1),
        title="Polynomial monitor spacing",
        description=(
            "Default spatial interval (in cells) between samples for polynomial autograd monitors."
        ),
    )

    monitor_interval_custom: tuple[int, int, int] = Field(
        (1, 1, 1),
        title="Custom monitor spacing",
        description=(
            "Default spatial interval (in cells) between samples for custom autograd monitors."
        ),
    )

    quadrature_sample_fraction: float = Field(
        0.4,
        title="Quadrature sample fraction",
        description=(
            "Fraction of uniform samples reused when building Gauss quadrature nodes for "
            "autograd surface integrations."
        ),
        ge=0.0,
        le=1.0,
    )

    gauss_quadrature_order: PositiveInt = Field(
        7,
        title="Gauss quadrature order",
        description=(
            "Maximum Gauss-Legendre order used when constructing composite quadrature rules "
            "for autograd surface integrations."
        ),
    )

    edge_clip_tolerance: float = Field(
        1e-9,
        title="Edge clipping tolerance",
        description=(
            "Padding tolerance applied when clipping polygon edges against simulation bounds "
            "in autograd surface integrations."
        ),
        ge=0.0,
    )

    solver_freq_chunk_size: PositiveInt | None = Field(
        None,
        title="Adjoint frequency chunk size",
        description=(
            "Maximum number of frequencies to process per chunk during adjoint gradient "
            "evaluation. Use `None` to disable chunking."
        ),
    )

    max_traced_structures: PositiveInt = Field(
        500,
        title="Max traced structures",
        description="Maximum number of structures that can have traced fields in an adjoint run.",
    )

    max_adjoint_per_fwd: PositiveInt = Field(
        10,
        title="Max adjoint solves per forward",
        description="Maximum number of adjoint simulations dispatched per forward solve.",
    )

    @property
    def gradient_dtype_float(self) -> np.dtype:
        """Floating-point dtype implied by ``gradient_precision``."""

        return np.float64 if self.gradient_precision == "double" else np.float32

    @property
    def gradient_dtype_complex(self) -> np.dtype:
        """Complex dtype implied by ``gradient_precision``."""

        return np.complex128 if self.gradient_precision == "double" else np.complex64

    @field_serializer("local_adjoint_dir")
    def _serialize_local_adjoint_dir(self, value: Path) -> str:
        """Persist local gradient directories as strings."""

        return str(value)


@register_handler("adjoint")
def apply_adjoint(config: AdjointConfig) -> None:
    """Warn when remote gradients will ignore autograd overrides."""

    if config.local_gradient:
        return

    defaults = AdjointConfig()
    overridden = [
        name
        for name in config.model_fields
        if name != "local_gradient" and getattr(config, name) != getattr(defaults, name)
    ]
    if not overridden:
        return

    overrides = ", ".join(sorted(overridden))
    log.warning(
        f"Autograd configuration overrides ({overrides}) are active while "
        "'autograd.local_gradient' is False. Remote gradients ignore these "
        "values. Enable local gradients to apply them locally."
    )


@register_section("web")
class WebConfig(ConfigSection):
    """Web/HTTP configuration."""

    apikey: SecretStr | None = Field(
        None,
        title="API key",
        description="Tidy3D API key.",
        json_schema_extra={"persist": True},
    )

    ssl_verify: bool = Field(
        True,
        title="SSL verification",
        description="Verify SSL certificates for API requests.",
    )

    enable_caching: bool = Field(
        True,
        title="Enable server-side caching",
        description="Allow the web service to return cached simulation results.",
        json_schema_extra={"persist": True},
    )

    api_endpoint: str = Field(
        "https://tidy3d-api.simulation.cloud",
        title="API endpoint",
        description="Tidy3D API base URL.",
    )

    website_endpoint: str = Field(
        "https://tidy3d.simulation.cloud",
        title="Website endpoint",
        description="Tidy3D website URL.",
    )

    s3_region: str = Field(
        "us-gov-west-1",
        title="S3 region",
        description="AWS S3 region used by the platform.",
    )

    timeout: int = Field(
        120,
        title="HTTP timeout",
        description="HTTP request timeout in seconds.",
        ge=0,
        le=300,
    )

    ssl_version: ssl.TLSVersion | None = Field(
        None,
        title="SSL/TLS version",
        description="Optional SSL/TLS version to enforce for requests.",
    )

    env_vars: dict[str, str] = Field(
        default_factory=dict,
        title="Environment variable overrides",
        description="Environment variables to export when this config is applied.",
    )

    def to_dict(self, *, mask_secrets: bool = True) -> dict[str, Any]:
        data = super().to_dict(mask_secrets=mask_secrets)
        if mask_secrets:
            if isinstance(data.get("apikey"), SecretStr):
                data["apikey"] = None
        else:
            secret = data.get("apikey")
            if isinstance(secret, SecretStr):
                data["apikey"] = secret.get_secret_value()
        ssl_version = data.get("ssl_version")
        if isinstance(ssl_version, ssl.TLSVersion):
            data["ssl_version"] = ssl_version.value
        for field in ("api_endpoint", "website_endpoint"):
            if field in data and data[field] is not None:
                data[field] = str(data[field])
        return data

    @field_validator("api_endpoint", "website_endpoint", mode="before")
    @classmethod
    def _validate_http_url(cls, value: Any) -> str:
        if value is None:
            return value
        parsed = urlparse(str(value))
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("Value must be an HTTP or HTTPS URL")
        normalized = parsed.geturl()
        if (
            parsed.path in {"", "/"}
            and not parsed.params
            and not parsed.query
            and not parsed.fragment
        ):
            normalized = normalized.rstrip("/")
        return normalized


@register_handler("web")
def apply_web(config: WebConfig) -> None:
    """Apply web-related environment variable overrides."""

    manager = _get_attached_manager()
    if manager is None:
        raise RuntimeError("Configuration manager not attached; cannot apply web env overrides.")
    manager.apply_web_env(dict(config.env_vars))


@register_section("plugins")
class PluginsContainer(ConfigSection):
    """Container that holds plugin-specific configuration sections."""

    model_config = ConfigDict(extra="allow")


__all__ = [
    "AdjointConfig",
    "LoggingConfig",
    "MicrowaveConfig",
    "PluginsContainer",
    "SimulationConfig",
    "WebConfig",
]
