"""Send the messages the core logger emits while Tidy3D validates a schema to the Tidy3D log."""

from __future__ import annotations

import threading
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

from flexcompute.core.log import log as core_log

from tidy3d.log import log

if TYPE_CHECKING:
    from collections.abc import Generator

_RELAYING: ContextVar[bool] = ContextVar("relaying_core_log", default=False)
_INSTALL_LOCK = threading.Lock()


class _MutedWhileRelaying:
    """A core log handler that stays quiet while the calling context relays."""

    def __init__(self, handler: Any) -> None:
        object.__setattr__(self, "handler", handler)

    def handle(self, level: int, level_name: str, message: str) -> None:
        if not _RELAYING.get():
            self.handler.handle(level, level_name, message)

    def __getattr__(self, name: str) -> Any:
        # Settings such as ``level`` stay on the wrapped handler. ``handler`` itself is missing
        # only on an instance that is not initialized, where forwarding it would recurse.
        if name == "handler":
            raise AttributeError(name)
        return getattr(self.handler, name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self.handler, name, value)


class _Tidy3dLogHandler:
    """The core log handler that hands a relayed message to the Tidy3D log."""

    def handle(self, level: int, level_name: str, message: str) -> None:
        if _RELAYING.get():
            log.log(level, message)


@contextmanager
def relay_core_log() -> Generator[None, None, None]:
    """Send the core log's messages in the calling context to the Tidy3D log instead.

    The schema logs through the core logger while its rules run, for example the warnings of
    its grid build. Inside this block they reach the Tidy3D handlers and capture, and the core
    handlers stay quiet. The core logger applies its suppression, ``log_once`` and consolidation
    first, so a consolidated block relays one message and its summary. Capture records a relayed
    message at no location. Other threads and contexts keep the core handlers. The wrappers stay on
    the core handlers once installed and are quiet only while a block relays.
    """
    with _INSTALL_LOCK:
        for key, handler in tuple(core_log.handlers.items()):
            if not isinstance(handler, (_MutedWhileRelaying, _Tidy3dLogHandler)):
                core_log.handlers[key] = _MutedWhileRelaying(handler)
        core_log.handlers.setdefault("tidy3d_log", _Tidy3dLogHandler())
    token = _RELAYING.set(True)
    try:
        yield
    finally:
        _RELAYING.reset(token)
