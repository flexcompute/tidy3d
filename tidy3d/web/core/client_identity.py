"""Request-local client identity used by internal web request builders."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Generator


@dataclass(frozen=True)
class ClientIdentity:
    """Immutable protocol identity captured by a product web client."""

    protocol_version: str
    solver_version: str | None = None


_CURRENT_CLIENT_IDENTITY: ContextVar[ClientIdentity | None] = ContextVar(
    "tidy3d_web_client_identity",
    default=None,
)


def get_client_identity() -> ClientIdentity | None:
    """Return the identity active for the current request context."""

    return _CURRENT_CLIENT_IDENTITY.get()


@contextmanager
def use_client_identity(identity: ClientIdentity) -> Generator[None, None, None]:
    """Scope one client identity to the current thread or async context."""

    token = _CURRENT_CLIENT_IDENTITY.set(identity)
    try:
        yield
    finally:
        _CURRENT_CLIENT_IDENTITY.reset(token)


def resolve_protocol_version(*, persisted: str | None = None) -> str | None:
    """Resolve persisted task identity before request-local client identity."""

    if persisted is not None:
        return persisted
    identity = get_client_identity()
    return identity.protocol_version if identity is not None else None
