"""Protected optical medium references and the policy shared with protected doping."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, ClassVar, Literal, get_args

from pydantic import Field, field_validator, model_validator

from tidy3d.components.base import Tidy3dBaseModel, cached_property
from tidy3d.components.bound_ops import bounds_contains
from tidy3d.components.data.binary import OPAQUE_BINARY_DATA_TYPE, OpaqueBinaryData
from tidy3d.components.types import TYPE_TAG_STR
from tidy3d.exceptions import DataError, SetupError, ValidationError
from tidy3d.log import log

from .debye import Debye
from .drude import Drude
from .isotropic import Medium
from .lorentz import Lorentz
from .pole_residue import PoleResidue
from .sellmeier import Sellmeier

if TYPE_CHECKING:
    from pydantic import ValidationInfo

    from tidy3d.components.types import Ax, Axis, FreqBound, PermittivityComponent
    from tidy3d.components.viz import VisualizationSpec

    from .base import AbstractMedium


PublicOpticalMediumType = Medium | PoleResidue | Sellmeier | Lorentz | Debye | Drude

PROTECTED_SURROGATE_DISCLOSURE = (
    "Protected assets are present. Local validation, grid generation, "
    "resource estimates, and diagnostics use foundry-approved public surrogate "
    "values. Cloud validation using the protected assets is authoritative."
)

PROTECTED_FDTD_ALLOWED_MONITOR_TYPES: frozenset[str] = frozenset(
    {
        "FieldMonitor",
        "FieldTimeMonitor",
        "PointCloudFieldMonitor",
        "SurfaceFieldMonitor",
        "SurfaceFieldTimeMonitor",
        "FluxMonitor",
        "FluxTimeMonitor",
        "ModeMonitor",
        "ModeSolverMonitor",
        "ModeTimeMonitor",
        "GaussianOverlapMonitor",
        "AstigmaticGaussianOverlapMonitor",
        "ThinLensOverlapMonitor",
        "MicrowaveModeMonitor",
        "MicrowaveModeSolverMonitor",
        "DirectivityMonitor",
    }
)

PROTECTED_FDTD_MATERIAL_MONITOR_TYPES: frozenset[str] = frozenset(
    {
        "PermittivityMonitor",
        "MediumMonitor",
        "PointCloudPermittivityMonitor",
        "FieldStructureMonitor",
        "FieldStructureTimeMonitor",
        "SurfaceFieldStructureMonitor",
    }
)

# Hosted charge simulations expose no spatial output for protected doping: carrier, potential,
# band, field and current-density maps stay denied. The terminal device characteristics are not
# monitors and are always returned, and the only monitor allowed is the capacitance over the whole
# simulation, which is a single curve over the bias.
PROTECTED_CHARGE_ALLOWED_MONITOR_TYPES: frozenset[str] = frozenset({"SteadyCapacitanceMonitor"})

# The solvers of these simulations do not read doping, so protected doping inside one of them is
# inert: it is never decrypted, and it does not make the simulation, or a modeler or batch that
# wraps it, protected. Doping elsewhere, for example in an EME simulation, still counts.
PROTECTED_DOPING_INERT_SIMULATION_TYPES: frozenset[str] = frozenset(
    {"Simulation", "ModeSimulation", "ModeSolver"}
)

ProtectedMediumSurfacePolicy = Literal[
    "owned-public-ref",
    "serialized-public-surrogate",
    "delegate-surrogate",
    "fixed-safe-value",
    "unsupported",
    "not-applicable",
]

PROTECTED_MEDIUM_SURFACE_POLICY_CATEGORIES: frozenset[str] = frozenset(
    get_args(ProtectedMediumSurfacePolicy)
)

ProtectedAssetKind = Literal["optical_medium", "doping"]

# One solver-compatibility label per asset kind: the label is bound into the authenticated
# envelope, so a ciphertext published for one kind cannot be reinterpreted as another.
PROTECTED_SOLVER_COMPATIBILITY_BY_KIND: dict[ProtectedAssetKind, str] = {
    "optical_medium": "tidy3d-public-medium-3d-v1",
    "doping": "tidy3d-doping-box-v1",
}

PROTECTED_MEDIUM_PUBLIC_ID_MAX_LENGTH = 128
PROTECTED_MEDIUM_ENCRYPTED_DATA_KEY_MAX_BYTES = 8192
PROTECTED_MEDIUM_NONCE_BYTES = 12
PROTECTED_MEDIUM_PUBLIC_ID_CHARS = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._:-"
)

PROTECTED_MEDIUM_SURFACE_POLICY: dict[str, ProtectedMediumSurfacePolicy] = {
    "asset_id": "owned-public-ref",
    "asset_kind": "owned-public-ref",
    "attrs": "owned-public-ref",
    "encrypted_payload": "owned-public-ref",
    "is_protected": "owned-public-ref",
    "name": "owned-public-ref",
    "optical": "owned-public-ref",
    "public_name": "owned-public-ref",
    "public_surrogate": "owned-public-ref",
    "revision": "owned-public-ref",
    "schema_version": "owned-public-ref",
    "type": "owned-public-ref",
    "surrogate": "serialized-public-surrogate",
    "_compatible_with": "delegate-surrogate",
    "_eps_bounds": "delegate-surrogate",
    "_eps_plot": "delegate-surrogate",
    "_has_incompatibilities": "delegate-surrogate",
    "_incompatible_material_types": "delegate-surrogate",
    "eps_comp": "delegate-surrogate",
    "eps_complex_to_nk": "delegate-surrogate",
    "eps_diagonal": "delegate-surrogate",
    "eps_diagonal_numerical": "delegate-surrogate",
    "eps_model": "delegate-surrogate",
    "frequency_range": "delegate-surrogate",
    "is_pec": "delegate-surrogate",
    "is_pec_like": "delegate-surrogate",
    "is_pmc": "delegate-surrogate",
    "is_spatially_uniform": "delegate-surrogate",
    "loss_tangent_model": "delegate-surrogate",
    "n_cfl": "delegate-surrogate",
    "nk_model": "delegate-surrogate",
    "plot": "delegate-surrogate",
    "pole_residue": "delegate-surrogate",
    "sigma_model": "delegate-surrogate",
    "viz_spec": "delegate-surrogate",
    "allow_gain": "fixed-safe-value",
    "charge": "fixed-safe-value",
    "electrical": "fixed-safe-value",
    "heat": "fixed-safe-value",
    "heat_spec": "fixed-safe-value",
    "is_custom": "fixed-safe-value",
    "is_fully_anisotropic": "fixed-safe-value",
    "is_isotropic": "delegate-surrogate",
    "is_nonlinear": "fixed-safe-value",
    "is_time_modulated": "fixed-safe-value",
    "modulation_spec": "fixed-safe-value",
    "nonlinear_spec": "fixed-safe-value",
    "background_index_from_freqs": "unsupported",
    "sel_inside": "unsupported",
    "Hz_to_angular_freq": "not-applicable",
    "angular_freq_to_Hz": "not-applicable",
    "angular_freq_to_eV": "not-applicable",
    "eps_complex_to_eps_loss_tangent": "not-applicable",
    "eps_complex_to_eps_sigma": "not-applicable",
    "eps_loss_tangent_to_eps_complex": "not-applicable",
    "eps_sigma_to_eps_complex": "not-applicable",
    "eV_to_angular_freq": "not-applicable",
    "nk_to_eps_complex": "not-applicable",
    "nk_to_eps_sigma": "not-applicable",
}


def protected_surrogate_digest(surrogate: Any) -> str:
    """Return the canonical digest binding an encrypted payload to its public surrogate."""
    value = surrogate.model_dump(mode="json", exclude_unset=False)
    serialized = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def protected_fdtd_monitor_policy_error(monitor: Any, monitor_index: int) -> str | None:
    """Return a protected-output policy error for an FDTD monitor, if any."""
    monitor_type = getattr(monitor, "type", type(monitor).__name__)
    if monitor_type in PROTECTED_FDTD_MATERIAL_MONITOR_TYPES:
        return (
            "Protected media do not support material-property monitors "
            f"for hosted execution. Monitor 'monitors[{monitor_index}]' has type "
            f"'{monitor_type}', which could reveal protected constitutive values."
        )
    if monitor_type not in PROTECTED_FDTD_ALLOWED_MONITOR_TYPES:
        return (
            "Protected media use a fail-closed monitor policy. "
            f"Monitor 'monitors[{monitor_index}]' has type '{monitor_type}', which is "
            "not in the protected-output allowlist. Allowed FDTD outputs are E/H field "
            "data, flux data, and overlap or mode amplitudes/profiles."
        )
    if monitor_type == "FluxMonitor" and getattr(monitor, "enable_adjoint", False):
        return (
            "Protected media do not support FluxMonitor(enable_adjoint=True) "
            "because adjoint workflows are unsupported."
        )
    if monitor_type == "PointCloudFieldMonitor" and any(
        field.startswith("D") for field in getattr(monitor, "fields", ())
    ):
        return (
            "Protected media do not support point-cloud 'D' field outputs "
            f"for hosted execution. Monitor 'monitors[{monitor_index}]' reconstructs "
            "'D' from electric fields and local permittivity."
        )
    return None


def _covers_simulation(monitor_bounds: Any, simulation_bounds: Any) -> bool:
    """Whether a monitor covers the whole simulation.

    The tolerance follows the size of the domain and not its coordinates: a relative tolerance on
    coordinates far from the origin would let a small monitor pass for the whole domain. An
    unbounded axis has no size to scale by, so only the finite sizes count: the monitor has to be
    unbounded along that axis as well.
    """
    from math import isfinite

    spans = (float(upper) - float(lower) for lower, upper in zip(*simulation_bounds))
    extent = max((span for span in spans if isfinite(span)), default=0.0)
    return bounds_contains(monitor_bounds, simulation_bounds, rtol=0.0, atol=1e-9 * extent)


def protected_heat_charge_monitor_policy_error(
    monitor: Any, monitor_index: int, *, simulation_bounds: Any
) -> str | None:
    """Return a protected-output policy error for a heat or charge monitor, if any."""
    monitor_type = getattr(monitor, "type", type(monitor).__name__)
    if monitor_type not in PROTECTED_CHARGE_ALLOWED_MONITOR_TYPES:
        return (
            "Protected assets use a fail-closed monitor policy for heat and charge simulations. "
            f"Monitor 'monitors[{monitor_index}]' has type '{monitor_type}', which is not in the "
            "protected-output allowlist. Protected doping allows only a "
            "'SteadyCapacitanceMonitor' that covers the whole simulation "
            "(size=(td.inf, td.inf, td.inf)). The terminal device characteristics are returned "
            "without monitors."
        )
    if not _covers_simulation(monitor.bounds, simulation_bounds):
        return (
            "Protected doping only supports a 'SteadyCapacitanceMonitor' that covers the "
            f"whole simulation. Monitor 'monitors[{monitor_index}]' covers a smaller region, "
            "where its capacitance could map the protected carrier distribution. "
            "Use size=(td.inf, td.inf, td.inf)."
        )
    return None


class ProtectedMediumEncryptedPayload(Tidy3dBaseModel):
    """Authenticated encrypted solver-ready asset (optical medium or doping box) carried by a protected reference."""

    envelope_version: Literal["1"] = Field(
        default="1",
        title="Protected Medium Envelope Version",
        description="Version of the encrypted protected-medium payload envelope.",
    )

    algorithm: Literal["RSA-OAEP-SHA256+A256GCM"] = Field(
        default="RSA-OAEP-SHA256+A256GCM",
        title="Protected Medium Encryption Algorithm",
        description="Hybrid public-key envelope used for the protected payload.",
    )

    foundry_id: str = Field(
        title="Foundry ID",
        description="Public foundry key-domain identifier for selecting the runtime key.",
        min_length=1,
        max_length=PROTECTED_MEDIUM_PUBLIC_ID_MAX_LENGTH,
    )

    key_id: str = Field(
        title="Foundry Key ID",
        description="Public key-generation identifier for selecting the runtime key.",
        min_length=1,
        max_length=PROTECTED_MEDIUM_PUBLIC_ID_MAX_LENGTH,
    )

    surrogate_digest: str = Field(
        title="Surrogate Digest",
        description="SHA-256 digest binding the public surrogate to the encrypted payload.",
        min_length=64,
        max_length=64,
    )

    payload_format: Literal["tidy3d-hdf5"] = Field(
        title="Protected Payload Format",
        description="Serialization format of the decrypted solver-ready asset.",
    )

    payload_format_version: Literal["1"] = Field(
        title="Protected Payload Format Version",
        description="Version of the decrypted solver-ready asset serialization contract.",
    )

    solver_compatibility: Literal[
        "tidy3d-public-medium-3d-v1",
        "tidy3d-doping-box-v1",
    ] = Field(
        title="Protected Payload Solver Compatibility",
        description="Solver compatibility contract expected for the decrypted payload.",
    )

    ciphertext_sha256: str = Field(
        title="Ciphertext Digest",
        description="SHA-256 digest of the encrypted solver-ready asset payload bytes.",
        min_length=64,
        max_length=64,
    )

    encrypted_data_key: OpaqueBinaryData = Field(
        title="Encrypted Data Key",
        description=(
            "Binary HDF5 payload data containing the content key encrypted for the "
            "foundry runtime key. JSON and YAML serialization store only a placeholder."
        ),
    )

    nonce: OpaqueBinaryData = Field(
        title="Payload Nonce",
        description=(
            "Binary HDF5 payload data containing the AES-GCM nonce for the encrypted "
            "payload. JSON and YAML serialization store only a placeholder."
        ),
    )

    ciphertext: OpaqueBinaryData = Field(
        title="Payload Ciphertext",
        description=(
            "Binary HDF5 payload data containing the authenticated ciphertext for the "
            "solver-ready asset. JSON and YAML serialization store only a placeholder."
        ),
    )

    @field_validator("foundry_id", "key_id")
    @classmethod
    def _validate_non_empty_public_id(cls, value: str) -> str:
        """Validate public key-domain identifiers."""
        if not value or value.strip() != value:
            raise ValidationError("Protected medium encrypted payload identifiers must be set.")
        if any(char not in PROTECTED_MEDIUM_PUBLIC_ID_CHARS for char in value):
            raise ValidationError(
                "Protected medium encrypted payload identifiers contain unsupported characters."
            )
        return value

    @field_validator("surrogate_digest", "ciphertext_sha256")
    @classmethod
    def _validate_digest(cls, value: str) -> str:
        """Validate public digest syntax without revealing payload details."""
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise ValidationError("Protected medium encrypted payload digest must be SHA-256 hex.")
        return value

    @staticmethod
    def _to_opaque_binary_data(value: Any, *, field_name: str) -> OpaqueBinaryData:
        """Normalize binary payload input to opaque HDF5-backed bytes."""
        if isinstance(value, str):
            raise ValidationError(
                f"Protected medium encrypted payload {field_name} must be binary HDF5 data."
            )
        if isinstance(value, dict) and value.get("type") == OPAQUE_BINARY_DATA_TYPE:
            raise ValidationError(
                f"Protected medium encrypted payload {field_name} must be binary HDF5 data."
            )
        try:
            return OpaqueBinaryData(value)
        except (DataError, ValueError):
            raise ValidationError(
                f"Protected medium encrypted payload {field_name} must be one-dimensional bytes."
            ) from None

    @staticmethod
    def _validate_binary_size(
        value: OpaqueBinaryData,
        *,
        field_name: str,
        max_bytes: int | None = None,
        exact_bytes: int | None = None,
    ) -> None:
        """Validate binary payload length without exposing payload bytes."""
        size = len(value)
        if size == 0:
            raise ValidationError(
                f"Protected medium encrypted payload {field_name} must be non-empty."
            )
        if exact_bytes is not None and size != exact_bytes:
            raise ValidationError(
                f"Protected medium encrypted payload {field_name} has invalid length."
            )
        if max_bytes is not None and size > max_bytes:
            raise ValidationError(
                f"Protected medium encrypted payload {field_name} has invalid length."
            )

    @field_validator("encrypted_data_key", "nonce", "ciphertext", mode="before")
    @classmethod
    def _normalize_binary_payload_field(cls, value: Any, info: ValidationInfo) -> OpaqueBinaryData:
        """Normalize encrypted payload fields to binary HDF5 data arrays."""
        return cls._to_opaque_binary_data(value, field_name=info.field_name)

    @field_validator("encrypted_data_key")
    @classmethod
    def _validate_encrypted_data_key(cls, value: OpaqueBinaryData) -> OpaqueBinaryData:
        """Validate encrypted content-key shape without exposing it."""
        cls._validate_binary_size(
            value,
            field_name="data key",
            max_bytes=PROTECTED_MEDIUM_ENCRYPTED_DATA_KEY_MAX_BYTES,
        )
        return value

    @field_validator("nonce")
    @classmethod
    def _validate_nonce(cls, value: OpaqueBinaryData) -> OpaqueBinaryData:
        """Validate AES-GCM nonce shape."""
        cls._validate_binary_size(
            value,
            field_name="nonce",
            exact_bytes=PROTECTED_MEDIUM_NONCE_BYTES,
        )
        return value

    @field_validator("ciphertext")
    @classmethod
    def _validate_ciphertext(cls, value: OpaqueBinaryData) -> OpaqueBinaryData:
        """Validate encrypted payload shape without exposing it."""
        cls._validate_binary_size(
            value,
            field_name="ciphertext",
        )
        return value


def validate_protected_payload_binding(reference: Any) -> None:
    """Ensure an encrypted payload matches the asset kind and public surrogate of its reference."""
    payload = reference.encrypted_payload
    if payload is None:
        return
    if payload.solver_compatibility != PROTECTED_SOLVER_COMPATIBILITY_BY_KIND[reference.asset_kind]:
        raise ValidationError(  # post-init-tidy3d-error: ignore
            "Protected asset encrypted payload does not match its asset kind."
        )
    if payload.surrogate_digest != protected_surrogate_digest(reference.public_surrogate):
        raise ValidationError(  # post-init-tidy3d-error: ignore
            "Protected asset encrypted payload does not match its surrogate. The public fields "
            "of a protected reference must not be changed."
        )


class ProtectedOpticalMediumRef(Tidy3dBaseModel):
    """Reference to a hosted protected optical medium with a public local model."""

    PROTECTED_REFERENCE: ClassVar[bool] = True

    SURROGATE_DELEGATED_ATTRIBUTES: ClassVar[frozenset[str]] = frozenset(
        name
        for name, category in PROTECTED_MEDIUM_SURFACE_POLICY.items()
        if category == "delegate-surrogate"
    )

    asset_id: str = Field(
        title="Protected Asset ID",
        description="Stable public identifier bound into the protected medium payload.",
    )

    revision: str = Field(
        title="Protected Asset Revision",
        description=("Immutable public revision token bound into the protected medium payload."),
    )

    asset_kind: Literal["optical_medium"] = Field(
        default="optical_medium",
        title="Protected Asset Kind",
        description="Public schema discriminator for the protected asset class.",
    )

    schema_version: Literal["1.0"] = Field(
        default="1.0",
        title="Protected Reference Schema Version",
        description="Public schema version of the protected optical reference envelope.",
    )

    public_name: str | None = Field(
        default=None,
        title="Public Name",
        description="Optional public display name for the protected optical medium.",
    )

    surrogate: PublicOpticalMediumType = Field(
        title="Surrogate",
        description=(
            "Foundry-approved public optical surrogate used for local validation and preview."
        ),
        discriminator=TYPE_TAG_STR,
    )

    encrypted_payload: ProtectedMediumEncryptedPayload | None = Field(
        default=None,
        title="Encrypted Payload",
        description=(
            "Optional package-carried encrypted solver-ready medium. Hosted execution may "
            "decrypt this payload with a trusted foundry runtime key."
        ),
    )

    @field_validator("asset_id", "revision", "schema_version")
    @classmethod
    def _validate_non_empty_identifier(cls, value: str) -> str:
        """Validate public identifier strings without contacting external services."""
        if not value or value.strip() != value:
            raise ValidationError("Protected optical medium identifiers must be non-empty strings.")
        return value

    @field_validator("surrogate")
    @classmethod
    def _validate_surrogate(cls, value: AbstractMedium) -> AbstractMedium:
        """Restrict public surrogate models to simple passive optical media."""
        if value.is_custom:
            raise ValidationError("Protected optical public surrogates must be non-custom.")
        if value.is_time_modulated or value.is_nonlinear:
            raise ValidationError(
                "Protected optical public surrogates must not be time-modulated or nonlinear."
            )
        if value.allow_gain:
            raise ValidationError("Protected optical public surrogates must not allow gain.")
        if value.heat_spec is not None:
            raise ValidationError(
                "Protected optical public surrogates must not include heat specs."
            )
        if value.is_pec or value.is_pmc:
            raise ValidationError(
                "PEC/PMC are not useful public surrogates for protected optical assets."
            )
        return value

    @model_validator(mode="after")
    def _validate_encrypted_payload_surrogate_digest(self) -> ProtectedOpticalMediumRef:
        """Ensure local surrogate behavior matches the encrypted payload binding."""
        validate_protected_payload_binding(self)
        return self

    @property
    def public_surrogate(self) -> AbstractMedium:
        """Public object that the encrypted payload is bound to."""
        return self.surrogate

    @property
    def is_protected(self) -> bool:
        """Whether this medium-like object refers to a protected hosted asset."""
        return True

    @property
    def name(self) -> str | None:
        """Public display name used by existing medium-name consumers."""
        return self.public_name or self.surrogate.name

    def __getattr__(self, name: str) -> Any:
        """Delegate reviewed medium-like helpers to the public surrogate."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            pass

        if name not in self.SURROGATE_DELEGATED_ATTRIBUTES:
            raise AttributeError(
                f"ProtectedOpticalMediumRef has no public delegated attribute {name!r}."
            )

        try:
            surrogate = object.__getattribute__(self, "surrogate")
        except AttributeError as original_error:
            raise AttributeError(
                "ProtectedOpticalMediumRef cannot delegate "
                f"{name!r} because the surrogate is unavailable: {original_error}"
            ) from original_error
        return getattr(surrogate, name)

    @property
    def frequency_range(self) -> FreqBound | None:
        """Frequency range of the public local model."""
        return self.surrogate.frequency_range

    @property
    def viz_spec(self) -> VisualizationSpec | None:
        """Visualization specification of the public local model."""
        return self.surrogate.viz_spec

    @property
    def allow_gain(self) -> Literal[False]:
        """Protected optical media do not allow client-side gain."""
        return False

    @property
    def nonlinear_spec(self) -> None:
        """Protected optical media do not allow client-side nonlinear specs."""
        return None

    @property
    def modulation_spec(self) -> None:
        """Protected optical media do not allow client-side modulation specs."""
        return None

    @property
    def heat_spec(self) -> None:
        """Protected optical media do not expose heat specs."""
        return None

    @property
    def charge(self) -> None:
        return None

    @property
    def electrical(self) -> None:
        return None

    @property
    def heat(self) -> None:
        return None

    @property
    def optical(self) -> ProtectedOpticalMediumRef:
        return self

    @property
    def _optical_medium_key(self) -> ProtectedOpticalMediumRef:
        """Keep protected asset identity when comparing unresolved optical references."""
        return self

    @cached_property
    def is_spatially_uniform(self) -> bool:
        return self.surrogate.is_spatially_uniform

    @cached_property
    def is_time_modulated(self) -> bool:
        return False

    @cached_property
    def is_nonlinear(self) -> bool:
        return False

    @cached_property
    def is_custom(self) -> bool:
        return False

    @cached_property
    def is_fully_anisotropic(self) -> bool:
        return False

    @cached_property
    def is_isotropic(self) -> bool:
        return getattr(self.surrogate, "is_isotropic", True)

    @cached_property
    def _incompatible_material_types(self) -> list[str]:
        return self.surrogate._incompatible_material_types

    @cached_property
    def _has_incompatibilities(self) -> bool:
        return self.surrogate._has_incompatibilities

    @cached_property
    def is_pec(self) -> bool:
        return self.surrogate.is_pec

    @cached_property
    def is_pmc(self) -> bool:
        return self.surrogate.is_pmc

    @cached_property
    def is_pec_like(self) -> bool:
        return self.surrogate.is_pec_like

    @cached_property
    def n_cfl(self) -> float:
        return self.surrogate.n_cfl

    @cached_property
    def pole_residue(self) -> PoleResidue:
        return self.surrogate.pole_residue

    def _compatible_with(self, other: AbstractMedium | ProtectedOpticalMediumRef) -> bool:
        """Whether the public local model is compatible with another medium-like object."""
        return self.surrogate._compatible_with(public_medium(other))

    def _warn_material_query_uses_surrogate(self) -> None:
        """Warn that an explicit material query is using the public surrogate."""
        warn_protected_surrogate(self, "Protected medium material query")

    def eps_model(self, frequency: float) -> complex:
        """Complex permittivity of the public local model."""
        self._warn_material_query_uses_surrogate()
        return self.surrogate.eps_model(frequency)

    def nk_model(self, frequency: float) -> tuple[float, float]:
        """Refractive index of the public local model."""
        self._warn_material_query_uses_surrogate()
        return self.surrogate.nk_model(frequency)

    def loss_tangent_model(self, frequency: float) -> tuple[float, float]:
        """Loss tangent of the public local model."""
        self._warn_material_query_uses_surrogate()
        return self.surrogate.loss_tangent_model(frequency)

    def eps_diagonal(self, frequency: float) -> tuple[complex, complex, complex]:
        """Diagonal permittivity tensor of the public local model."""
        self._warn_material_query_uses_surrogate()
        return self.surrogate.eps_diagonal(frequency)

    def eps_diagonal_numerical(self, frequency: float) -> tuple[complex, complex, complex]:
        """Numerical diagonal permittivity tensor of the public local model."""
        self._warn_material_query_uses_surrogate()
        return self.surrogate.eps_diagonal_numerical(frequency)

    def eps_comp(self, row: Axis, col: Axis, frequency: float) -> complex:
        """Permittivity component of the public local model."""
        self._warn_material_query_uses_surrogate()
        return self.surrogate.eps_comp(row=row, col=col, frequency=frequency)

    def sigma_model(self, freq: float) -> complex:
        """Conductivity of the public local model."""
        self._warn_material_query_uses_surrogate()
        return self.surrogate.sigma_model(freq)

    def _eps_plot(
        self, frequency: float, eps_component: PermittivityComponent | None = None
    ) -> float:
        """Plotting permittivity of the public local model."""
        return self.surrogate._eps_plot(frequency=frequency, eps_component=eps_component)

    def _eps_bounds(
        self, frequency: float, eps_component: PermittivityComponent | None = None
    ) -> tuple[float, float]:
        """Permittivity plot bounds of the public local model."""
        return self.surrogate._eps_bounds(frequency=frequency, eps_component=eps_component)

    def plot(self, freqs: float, ax: Ax = None) -> Ax:
        """Plot the public local model and mark it as a protected-medium preview."""
        ax = self.surrogate.plot(freqs=freqs, ax=ax)
        public_name = f": {self.public_name}" if self.public_name else ""
        ax.set_title(f"public surrogate for protected optical medium{public_name}")
        return ax


def is_protected_medium(obj: object) -> bool:
    """Whether ``obj`` is a protected optical medium reference.

    Protected doping is not an optical medium, use ``is_protected_reference`` to test for any kind.
    """
    return isinstance(obj, ProtectedOpticalMediumRef)


def is_protected_reference(obj: object) -> bool:
    """Whether ``obj`` is a protected reference of any asset kind."""
    return getattr(type(obj), "PROTECTED_REFERENCE", False) is True


def public_medium(medium: Any) -> AbstractMedium | None:
    """Return the public optical medium used for local client behavior."""
    # Avoid importing MultiPhysicsMedium here; material solver types import this package.
    if type(medium).__name__ == "MultiPhysicsMedium":
        medium = medium.optical
    if isinstance(medium, ProtectedOpticalMediumRef):
        return medium.surrogate
    return medium


def protected_surrogate_disclosure(surface: str | None = None) -> str:
    """Return user-facing disclosure text for surrogate-based protected previews."""
    if surface is None:
        return PROTECTED_SURROGATE_DISCLOSURE
    return f"{surface}: {PROTECTED_SURROGATE_DISCLOSURE}"


def _iter_model_children(obj: Tidy3dBaseModel) -> Any:
    """Yield pydantic field values relevant for execution-policy traversal."""
    for field_name in type(obj).model_fields:
        if field_name == "attrs":
            continue
        try:
            yield getattr(obj, field_name)
        except AttributeError:
            continue


def _iter_children(obj: object) -> Any:
    """Yield child objects for protected-reference discovery."""
    if isinstance(obj, Tidy3dBaseModel):
        yield from _iter_model_children(obj)
    elif isinstance(obj, Mapping):
        yield from obj.values()
    elif isinstance(obj, (tuple, list, set, frozenset)):
        yield from obj


def _ignores_protected_doping(obj: object) -> bool:
    """Whether the class of ``obj``, or a base class, is named in the inert-doping types."""
    return any(cls.__name__ in PROTECTED_DOPING_INERT_SIMULATION_TYPES for cls in type(obj).__mro__)


def protected_asset_refs(obj: object) -> tuple[Any, ...]:
    """Collect the protected references reachable from ``obj``.

    Protected doping inside a simulation of ``PROTECTED_DOPING_INERT_SIMULATION_TYPES`` is left
    out, because that solver never reads it. A structure or medium passed on its own yields every
    reference.
    """
    refs: dict[int, Any] = {}
    seen: set[tuple[int, bool]] = set()
    stack = [(obj, False)]

    while stack:
        current, inert = stack.pop()
        # The same object can sit inside an optical simulation and elsewhere, e.g. in a batch.
        if (id(current), inert) in seen:
            continue
        seen.add((id(current), inert))

        if is_protected_reference(current):
            if not (inert and current.asset_kind == "doping"):
                refs[id(current)] = current
            continue

        inert = inert or _ignores_protected_doping(current)
        stack.extend((child, inert) for child in _iter_children(current))

    return tuple(refs.values())


def has_protected_media(obj: object) -> bool:
    """Whether ``protected_asset_refs(obj)`` is non-empty."""
    return bool(protected_asset_refs(obj))


def warn_protected_surrogate(
    obj: object, surface: str | None = None, asset_kind: ProtectedAssetKind | None = None
) -> None:
    """Warn once when a user-facing surface depends on protected surrogates.

    ``asset_kind`` limits the check to the kind that the surface shows.
    """
    if any(asset_kind in (None, ref.asset_kind) for ref in protected_asset_refs(obj)):
        log.warning(protected_surrogate_disclosure(surface), log_once=True)


def assert_no_protected_media(obj: object, feature: str) -> None:
    """Reject use of protected media in an unsupported client feature."""
    if not has_protected_media(obj):
        return
    raise SetupError(
        f"Protected media are not supported for {feature}. Protected optical media are only "
        "supported in hosted FDTD and mode simulations, and protected doping only in hosted heat "
        "and charge simulations, run through tidy3d.web."
    )


PROTECTED_DOPING_SOLVER_ERROR = (
    "Protected doping requires the accelerated charge solver. Remove "
    "'use_accelerated_solver=False' to use it."
)


def protected_asset_loc(simulation: Any, asset_kind: str) -> tuple[Any, ...]:
    """Validation location of the first ``asset_kind`` reference in a simulation.

    Points at the background medium or the structure that holds the reference, so a client can
    highlight the offending object. Falls back to the structure list.
    """
    if any(
        reference.asset_kind == asset_kind
        for reference in protected_asset_refs(getattr(simulation, "medium", None))
    ):
        return ("medium",)
    for index, structure in enumerate(getattr(simulation, "structures", ())):
        if any(reference.asset_kind == asset_kind for reference in protected_asset_refs(structure)):
            return ("structures", index)
    return ("structures",)


def validate_protected_heat_charge_assets(simulation: Any) -> None:
    """Enforce the protected-asset capability and output policy of heat and charge simulations."""
    references = protected_asset_refs(simulation)
    if not references:
        return
    if any(reference.asset_kind == "optical_medium" for reference in references):
        simulation._raise_validation_error_at_loc(
            "Protected optical media are not supported for charge, heat, or heat-charge "
            "simulations. Protected doping is supported.",
            *protected_asset_loc(simulation, "optical_medium"),
        )
    if getattr(simulation, "use_accelerated_solver", None) is False:
        simulation._raise_validation_error_at_loc(
            PROTECTED_DOPING_SOLVER_ERROR, "use_accelerated_solver"
        )
    for monitor_index, monitor in enumerate(simulation.monitors):
        error_message = protected_heat_charge_monitor_policy_error(
            monitor, monitor_index, simulation_bounds=simulation.bounds
        )
        if error_message:
            simulation._raise_validation_error_at_loc(error_message, "monitors", monitor_index)


__all__ = [
    "PROTECTED_CHARGE_ALLOWED_MONITOR_TYPES",
    "PROTECTED_DOPING_INERT_SIMULATION_TYPES",
    "PROTECTED_DOPING_SOLVER_ERROR",
    "PROTECTED_MEDIUM_SURFACE_POLICY",
    "PROTECTED_MEDIUM_SURFACE_POLICY_CATEGORIES",
    "PROTECTED_SOLVER_COMPATIBILITY_BY_KIND",
    "PROTECTED_SURROGATE_DISCLOSURE",
    "ProtectedAssetKind",
    "ProtectedMediumEncryptedPayload",
    "ProtectedMediumSurfacePolicy",
    "ProtectedOpticalMediumRef",
    "PublicOpticalMediumType",
    "assert_no_protected_media",
    "has_protected_media",
    "is_protected_medium",
    "is_protected_reference",
    "protected_asset_loc",
    "protected_asset_refs",
    "protected_heat_charge_monitor_policy_error",
    "protected_surrogate_digest",
    "protected_surrogate_disclosure",
    "public_medium",
    "validate_protected_heat_charge_assets",
    "validate_protected_payload_binding",
    "warn_protected_surrogate",
]
