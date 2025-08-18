from __future__ import annotations

from typing import Union

from .modal import ComponentModeler
from .terminal import TerminalComponentModeler

ComponentModelerType = Union[ComponentModeler, TerminalComponentModeler]
