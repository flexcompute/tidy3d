# base class for all of the invdes fields
from __future__ import annotations

from abc import ABC

from tidy3d.components.base import Tidy3dBaseModel


class InvdesBaseModel(Tidy3dBaseModel, ABC):
    """Base class for ``invdes`` components, in case we need it."""
