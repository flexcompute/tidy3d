# base class for all of the invdes fields
from __future__ import annotations

import abc

import tidy3d as td


class InvdesBaseModel(td.components.base.Tidy3dBaseModel, abc.ABC):
    """Base class for ``invdes`` components, in case we need it."""
