"""Built-in configuration section schemas and handlers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import urlparse

import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    DirectoryPath,
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    NonPositiveFloat,
    PositiveInt,
    SecretStr,
    field_serializer,
    field_validator,
)

from tidy3d._runtime import WASM_BUILD
from tidy3d.log import (
    DEFAULT_LEVEL,
    LogLevel,
    log,
    set_log_suppression,
    set_logging_level,
    set_warn_once,
)

from .registry import get_manager as _get_attached_manager
from .registry import register_handler, register_section

if TYPE_CHECKING:
    from os import PathLike

VALID_VGPU_ALLOCATIONS = (1, 2, 4, 8)

TLS_VERSION_CHOICES = {"TLSv1", "TLSv1_1", "TLSv1_2", "TLSv1_3"}
ParallelAdjointModeDirectionPolicy = Literal[
    "assume_outgoing",
    "run_both_directions",
]


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
        default=DEFAULT_LEVEL,
        title="Logging level",
        description="Lowest logging level that will be emitted.",
        json_schema_extra={"persist": True},
    )

    suppression: bool = Field(
        default=True,
        title="Log suppression",
        description="Suppress repeated log messages when True.",
    )

    warn_once: bool = Field(
        default=False,
        title="Warn once",
        description="When True, each unique warning message is only shown once per process.",
    )


@register_handler("logging")
def apply_logging(config: LoggingConfig) -> None:
    """Apply logging configuration globally."""

    set_logging_level(config.level)
    set_log_suppression(config.suppression)
    set_warn_once(config.warn_once)


@register_section("simulation")
class SimulationConfig(ConfigSection):
    """Simulation-related configuration."""

    use_local_subpixel: bool | None = Field(
        default=None,
        title="Use local subpixel",
        description=(
            "Controls whether local subpixel averaging is used in "
            "'Simulation.epsilon', 'ModeSimulation.run_local', and 'ModeSolver.solve'. "
            "Subpixel averaging improves the accuracy of these computations. "
            "This feature requires the 'tidy3d-extras' package, which "
            r"can be installed using 'pip install tidy3d\[extras]'. "
            "If True, local subpixel averaging is enabled, and these functions will "
            "raise an error if 'tidy3d-extras' is not installed. "
            "If False, local subpixel averaging is disabled and these functions "
            "will use permittivity staircasing instead. "
            "If None (the default), local subpixel averaging will be used when "
            "'tidy3d-extras' is installed and will silently fall back "
            "to permittivity staircasing otherwise."
        ),
    )

    skip_size_checks: bool = Field(
        default=False,
        title="Skip size checks",
        description=(
            "Skip client-side simulation and monitor size checks before upload. "
            "This is not intended to be changed by cloud users, and cloud servers still reject "
            "simulations that violate the default cloud limits."
        ),
        json_schema_extra={"persist": True},
    )


@register_section("microwave")
class MicrowaveConfig(ConfigSection):
    """Microwave solver configuration."""

    suppress_rf_license_warning: bool = Field(
        default=False,
        title="Suppress RF license warning",
        description="If true, do not emit microwave license availability warnings.",
    )


@register_section("adjoint")
class AdjointConfig(ConfigSection):
    """Adjoint (autograd) configuration section."""

    min_wvl_fraction: float = Field(
        default=5e-2,
        title="Minimum wavelength fraction",
        description=(
            "Minimum fraction of the smallest free-space wavelength used when discretizing "
            "cylindrical structures for autograd derivatives."
        ),
        ge=0.0,
    )

    points_per_wavelength: PositiveInt = Field(
        default=10,
        title="Points per wavelength",
        description=(
            "Default number of material sample points per wavelength when discretizing "
            "cylinders for autograd derivatives."
        ),
    )

    default_wavelength_fraction: float = Field(
        default=0.1,
        title="Default wavelength fraction",
        description=(
            "Fallback fraction of the minimum wavelength used when autograd needs to "
            "estimate adaptive spacing."
        ),
        ge=0.0,
    )

    minimum_spacing_fraction: float = Field(
        default=1e-3,
        title="Minimum spacing fraction",
        description=(
            "Minimum fraction of the shortest free-space adjoint wavelength used as the lower "
            "bound for adaptive shape-gradient surface-sampling spacing."
        ),
        ge=0.0,
    )

    boundary_snapping_fraction: float = Field(
        default=1.0,
        title="Boundary snapping fraction",
        description=(
            "Fraction of minimum local grid size to use for snapping coordinates outside of "
            "a boundary when computing shape gradients. Should be at least 0.5."
        ),
        ge=0.5,
    )

    pec_detection_threshold: NonPositiveFloat = Field(
        default=-100.0,
        title="PEC detection threshold",
        description=(
            "Value the real permittivity should be below to consider it a PEC material in "
            "the shape gradient boundary integration."
        ),
    )

    local_gradient: bool = Field(
        default=False,
        title="Enable local gradients",
        description=(
            "When True, autograd runs download intermediate data and compute gradients locally. "
            "Remote (default) gradients ignore local-only execution and numerical overrides, "
            "while client-side source-planning settings such as field-source PCA still apply."
        ),
        json_schema_extra={"persist": True},
    )

    local_adjoint_dir: Path = Field(
        default=Path("adjoint_data"),
        title="Local gradient directory",
        description=(
            "Relative directory name used to store intermediate results when local gradients are enabled."
        ),
        json_schema_extra={"persist": True},
    )

    parallel_run: bool = Field(
        default=False,
        title="Enable parallel adjoint sources",
        description=(
            "When True, run canonical adjoint simulations in parallel with the forward solve for "
            "supported monitor types when local gradients are enabled."
        ),
        json_schema_extra={"persist": True},
    )

    parallel_adjoint_mode_direction_policy: ParallelAdjointModeDirectionPolicy = Field(
        default="assume_outgoing",
        title="Parallel adjoint mode direction policy",
        description=(
            "Policy for selecting propagation directions when launching parallel adjoint mode "
            "simulations. 'assume_outgoing' uses the monitor position relative to the simulation "
            "center to choose a single outgoing direction and flips it for the adjoint source. "
            "'run_both_directions' launches adjoint sources for both '+' and '-' mode directions."
        ),
        json_schema_extra={"persist": True},
    )

    field_source_reduction_mode: Literal["pca"] | None = Field(
        default=None,
        title="Field-source adjoint reduction mode",
        description=(
            "Strategy for compressing compatible FieldData-derived adjoint current sources. "
            "None applies no reduction and uses standard adjoint source grouping. 'pca' "
            "compresses sources into principal components, one adjoint simulation per "
            "component."
        ),
    )

    field_source_pca_min_energy_coverage: float = Field(
        default=0.999,
        title="Field-source PCA minimum energy coverage",
        description=(
            "Minimum fraction of weighted current-source profile energy retained per "
            "current type (electric and magnetic blocks are truncated independently, each "
            "in its own units) by the field-source PCA spatial decomposition. The fraction "
            "is measured over all sources decomposed together, so a monitor holding a small "
            "share of that energy can be represented coarsely, or dropped, while the "
            "requested coverage is still reported as met. The retained fraction describes "
            "the injected source profiles, not the gradient: the discarded component still "
            "contributes through the simulated field response, which can amplify it, so "
            "this setting constrains gradient error indirectly rather than bounding it, and "
            "the gradient of such a weak monitor on its own can be inaccurate. Coverage is "
            "measured in energy, so a target of 0.999 can discard up to roughly 3% of the "
            "source amplitude. A value of 1.0 keeps the full numerical rank, representing "
            "every source and frequency exactly."
        ),
        ge=0.0,
        le=1.0,
    )

    field_source_pca_max_matrix_entries: PositiveInt = Field(
        default=20_000_000,
        title="Field-source PCA max matrix entries",
        description=(
            "Maximum number of dense complex entries allowed in an individual field-source "
            "PCA profile matrix. Larger PCA blocks are skipped and handled by standard "
            "port-versus-frequency grouping. This is a storage guard, not a runtime one: "
            "peak memory during the decomposition runs several times the entry count "
            "because of the singular-value solver's own factors and workspace, and "
            "decomposition time additionally grows with the number of frequencies, so "
            "blocks with equal entry counts can take very different times."
        ),
        json_schema_extra={"persist": True},
    )

    gradient_precision: Literal["single", "double"] = Field(
        default="single",
        title="Gradient precision",
        description="Floating-point precision used for autograd gradient calculations.",
    )

    monitor_interval_poly: tuple[int, int, int] = Field(
        default=(1, 1, 1),
        title="Polynomial monitor spacing",
        description=(
            "Spatial interval (in cells) between samples for the volumetric adjoint monitors "
            "of geometry paths handled volumetrically (numerical structures and "
            "custom-vjp-owned paths). Standard shape gradients record surface point clouds "
            "instead; their sampling density is set by 'default_wavelength_fraction', not by "
            "this interval."
        ),
    )

    monitor_interval_custom: tuple[int, int, int] = Field(
        default=(1, 1, 1),
        title="Custom monitor spacing",
        description=(
            "Spatial interval (in cells) between samples for the volumetric adjoint monitors "
            "of structures with traced medium paths. Shape-gradient surface point clouds are "
            "unaffected."
        ),
    )

    quadrature_sample_fraction: float = Field(
        default=0.4,
        title="Quadrature sample fraction",
        description=(
            "Fraction of uniform samples reused when building Gauss quadrature nodes for "
            "autograd surface integrations."
        ),
        ge=0.0,
        le=1.0,
    )

    gauss_quadrature_order: PositiveInt = Field(
        default=7,
        title="Gauss quadrature order",
        description=(
            "Maximum Gauss-Legendre order used when constructing composite quadrature rules "
            "for autograd surface integrations."
        ),
    )

    edge_clip_tolerance: float = Field(
        default=1e-9,
        title="Edge clipping tolerance",
        description=(
            "Padding tolerance applied when clipping polygon edges against simulation bounds "
            "in autograd surface integrations."
        ),
        ge=0.0,
    )

    solver_freq_chunk_size: PositiveInt | None = Field(
        default=None,
        title="Adjoint frequency chunk size",
        description=(
            "Maximum number of frequencies to process per chunk during adjoint gradient "
            "evaluation. Use `None` to disable chunking."
        ),
    )

    memory_allotment_fraction: float = Field(
        default=0.75,
        title="Adjoint memory allotment fraction",
        description=(
            "Fraction of reported available RAM reserved for local adjoint postprocessing "
            "when auto-selecting frequency chunk sizes."
        ),
        ge=0.0,
        le=1.0,
        json_schema_extra={"persist": True},
    )

    max_traced_structures: PositiveInt = Field(
        default=500,
        title="Max traced structures",
        description="Maximum number of structures that can have traced fields in an adjoint run.",
    )

    max_adjoint_per_fwd: PositiveInt = Field(
        default=10,
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
    """Warn when remote gradients will ignore local-only autograd overrides."""

    if config.local_gradient:
        return

    client_side_fields = {
        "field_source_reduction_mode",
        "field_source_pca_min_energy_coverage",
        "field_source_pca_max_matrix_entries",
    }
    defaults = AdjointConfig()
    overridden = [
        name
        for name in type(config).model_fields
        if name != "local_gradient"
        and name not in client_side_fields
        and getattr(config, name) != getattr(defaults, name)
    ]
    if not overridden:
        return

    overrides = ", ".join(sorted(overridden))
    log.warning(
        f"Autograd configuration overrides ({overrides}) are active while "
        "'autograd.local_gradient' is False. Remote gradients ignore these "
        "values. Enable local gradients to apply them locally."
    )


@register_section("run")
class RunConfig(ConfigSection):
    """Default run configuration for web submissions."""

    solver_version: str | None = Field(
        default=None,
        title="Solver version",
        description="Internal usage only. Default solver version to use for web runs.",
    )

    worker_group: str | None = Field(
        default=None,
        title="Worker group",
        description="Internal usage only. Default worker group to use for web runs.",
    )

    simulation_type: str = Field(
        default="tidy3d",
        title="Simulation type",
        description="Internal usage only. Default simulation type label for uploaded tasks.",
    )

    additional_payload: dict[str, Any] | None = Field(
        default=None,
        title="Additional payload",
        description="Internal usage only. Additional submit payload serialized to JSON and sent "
        "under 'additionalPayload'.",
    )

    pay_type: str = Field(
        default="AUTO",
        title="Payment type",
        description="Internal usage only. Default payment type for web runs.",
    )

    @field_validator("pay_type", mode="before")
    @classmethod
    def _validate_pay_type(cls, value: Any) -> str:
        from tidy3d.web.core.types import PayType

        candidate = getattr(value, "value", value)
        return PayType(candidate).value


@register_section("vgpu")
class VgpuConfig(ConfigSection):
    """Default vGPU configuration for web runs."""

    priority: int | None = Field(
        default=None,
        title="Priority",
        description="Default queue priority for vGPU runs (1 = lowest, 10 = highest).",
    )

    vgpu_allocation: int | None = Field(
        default=None,
        title="vGPU allocation",
        description="Default virtual GPU allocation for vGPU runs.",
    )

    ignore_memory_limit: bool | None = Field(
        default=None,
        title="Ignore memory limit",
        description="Default flag to allow vGPU runs above the estimated memory limit.",
    )

    @field_validator("priority")
    @classmethod
    def _validate_priority(cls, value: int | None) -> int | None:
        if value is None:
            return value
        if value < 1 or value > 10:
            raise ValueError("Priority must be between '1' and '10' if specified.")
        return value

    @field_validator("vgpu_allocation")
    @classmethod
    def _validate_vgpu_allocation(cls, value: int | None) -> int | None:
        if value is None:
            return value
        if value not in VALID_VGPU_ALLOCATIONS:
            raise ValueError(
                f"vgpu_allocation must be one of {list(VALID_VGPU_ALLOCATIONS)} if specified."
            )
        return value


class WebConfig(ConfigSection):
    """Web/HTTP configuration."""

    apikey: SecretStr | None = Field(
        default=None,
        title="API key",
        description="Tidy3D API key.",
        json_schema_extra={"persist": True},
    )

    ssl_verify: bool = Field(
        default=True,
        title="SSL verification",
        description="Verify SSL certificates for API requests.",
    )

    enable_caching: bool = Field(
        default=True,
        title="Enable server-side caching",
        description="Allow the web service to return cached simulation results.",
        json_schema_extra={"persist": True},
    )

    api_endpoint: str = Field(
        default="https://tidy3d-api.simulation.cloud",
        title="API endpoint",
        description="Tidy3D API base URL.",
    )

    website_endpoint: str = Field(
        default="https://tidy3d.simulation.cloud",
        title="Website endpoint",
        description="Tidy3D website URL.",
    )

    s3_region: str = Field(
        default="us-gov-west-1",
        title="S3 region",
        description="AWS S3 region used by the platform.",
    )

    timeout: int = Field(
        default=120,
        title="HTTP timeout",
        description="HTTP request timeout in seconds.",
        ge=0,
        le=300,
    )

    default_num_workers: PositiveInt = Field(
        default=10,
        title="Default batch workers",
        description=(
            "Default worker count for configurable ``Batch`` thread pools when ``num_workers`` "
            "is not provided. Upload/start uses a fixed concurrency of 64 workers."
        ),
    )

    ssl_version: str | None = Field(
        default=None,
        title="SSL/TLS version",
        description=(
            "Optional TLS version override to enforce for requests. Accepts values such as "
            "'TLSv1_2'."
        ),
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
        for field in ("api_endpoint", "website_endpoint"):
            if field in data and data[field] is not None:
                data[field] = str(data[field])
        return data

    @field_validator("ssl_version", mode="before")
    @classmethod
    def _convert_and_check_ssl_version_name(cls, value: Any) -> str | None:
        """Convert SSL enum to string and check if valid.

        Accepted examples:
            "TLSv1"
            "TLSv1_2"
            ssl.TLSVersion.TLSv1_2.name  -> "TLSv1_2"
        """
        if value is None:
            return None

        # Prefer enum.name if present, otherwise raw string
        candidate = getattr(value, "name", value)
        candidate = str(candidate).strip()

        if candidate not in TLS_VERSION_CHOICES:
            allowed = ", ".join(sorted(TLS_VERSION_CHOICES))
            raise ValueError(f"Invalid TLS version {candidate!r}. Must be one of: {allowed}")

        return candidate

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

    def build_api_url(self, path: str) -> str:
        """Join the configured API endpoint with a request path."""

        base = self.api_endpoint or ""
        path_str = path or ""
        if not base:
            return path_str.lstrip("/")
        if not path_str:
            return base
        return "/".join([base.rstrip("/"), path_str.lstrip("/")])


def apply_web(config: WebConfig) -> None:
    """Apply web-related environment variable overrides."""

    manager = _get_attached_manager()
    if manager is None:
        raise RuntimeError("Configuration manager not attached; cannot apply web env overrides.")
    manager.apply_web_env(dict(config.env_vars))


def _default_cache_directory() -> Path:
    """Determine the default on-disk cache directory respecting platform conventions."""

    base_override = os.getenv("TIDY3D_BASE_DIR")
    if base_override:
        base = Path(base_override).expanduser().resolve()
        return (base / "cache" / "simulations").resolve()
    else:
        xdg_cache = os.getenv("XDG_CACHE_HOME")
        if xdg_cache:
            base = Path(xdg_cache).expanduser().resolve()
        else:
            base = Path.home() / ".cache"
    return (base / "tidy3d" / "simulations").resolve()


class LocalCacheConfig(ConfigSection):
    """Settings controlling the optional local simulation cache."""

    enabled: bool = Field(
        default=True,
        title="Enable cache",
        description="Enable or disable the local simulation cache.",
        json_schema_extra={"persist": True},
    )

    directory: DirectoryPath = Field(
        default_factory=_default_cache_directory,
        title="Cache directory",
        description="Directory where cached artifacts are stored.",
        json_schema_extra={"persist": True},
    )

    max_size_gb: NonNegativeFloat = Field(
        default=10.0,
        title="Maximum cache size (GB)",
        description="Maximum cache size in gigabytes. Set to 0 for no size limit.",
        json_schema_extra={"persist": True},
    )

    max_entries: NonNegativeInt = Field(
        default=0,
        title="Maximum cache entries",
        description="Maximum number of cache entries. Set to 0 for no limit.",
        json_schema_extra={"persist": True},
    )

    @field_validator("directory", mode="before")
    @classmethod
    def _ensure_directory_exists(cls, v: PathLike) -> Path:
        """Expand ~, resolve path, and create directory if missing before DirectoryPath validation."""
        p = Path(v).expanduser().resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p

    @field_serializer("directory")
    def _serialize_directory(self, value: Path) -> str:
        """Persist directory as strings."""
        return str(value)


class BatchDataCacheConfig(ConfigSection):
    """Settings controlling in-memory caching for batch data."""

    enabled: bool = Field(
        default=True,
        title="Enable batch data cache",
        description="Cache batch results in memory when files are below the size threshold.",
    )

    max_total_size_gb: NonNegativeFloat = Field(
        default=1.0,
        title="Maximum total batch data size (GB)",
        description=(
            "Cache batch task data only when the combined size of all task data files is at or "
            "below this threshold. Set to 0 to disable."
        ),
    )


@register_section("plugins")
class PluginsContainer(ConfigSection):
    """Container that holds plugin-specific configuration sections."""

    model_config = ConfigDict(extra="allow")


# Register web and local_cache sections only in non-WASM environments
# where filesystem and network features are available
if not WASM_BUILD:
    register_section("web")(WebConfig)
    register_handler("web")(apply_web)
    register_section("local_cache")(LocalCacheConfig)
    register_section("batch_data_cache")(BatchDataCacheConfig)


__all__ = [
    "AdjointConfig",
    "BatchDataCacheConfig",
    "LocalCacheConfig",
    "LoggingConfig",
    "MicrowaveConfig",
    "PluginsContainer",
    "RunConfig",
    "SimulationConfig",
    "WebConfig",
]
