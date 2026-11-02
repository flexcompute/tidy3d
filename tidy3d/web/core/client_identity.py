"""Request-local client identity used by internal web request builders."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Mapping
    from contextlib import AbstractContextManager
    from typing import Any


@dataclass(frozen=True)
class ClientIdentity:
    """Immutable protocol identity captured by a product web client."""

    protocol_version: str
    solver_version: str | None = None
    model_classes: Mapping[str, type] = field(default_factory=dict)
    """Classes the product reads for the type tags it shares with Tidy3D, by tag."""
    task_types: Mapping[type, str] = field(default_factory=dict)
    """Server task type of each product model class, e.g. ``"FDTD"`` or ``"TERMINAL_CM"``.

    A class listed here, or a subclass of one, is a product model the client submits.
    Classes not listed resolve through Tidy3D's own table.
    """
    client_context: Callable[[ClientIdentity], AbstractContextManager[object]] | None = None
    """The product's client context, entered as ``client_context(self)``.

    Tidy3D re-enters it for work it does later on the client's behalf, such as the adjoint
    runs of an autograd backward pass.
    """
    modeler_runners: Mapping[type, Callable[..., Any]] = field(default_factory=dict)
    """Runner of each product component modeler, by class.

    Autograd runs a traced modeler as its port simulations through its runner, called with the
    arguments of Tidy3D's own ``ModalComponentModeler`` runner, instead of as one modeler task.
    """


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
