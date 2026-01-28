"""LayeredGeometry: 2D geometry with layer and net assignment."""

from __future__ import annotations

from typing import Optional

import pydantic.v1 as pydantic

from tidy3d.components.base import Tidy3dBaseModel
from tidy3d.components.geometry.geometry2d import Geometry2DType
from tidy3d.components.types import TYPE_TAG_STR


class LayeredGeometry(Tidy3dBaseModel):
    """A 2D geometry assigned to a layer with optional net.

    This is the atomic storage unit in LayeredStructure. It associates
    a pure geometric primitive with layer and net metadata.

    Users typically don't create these directly - use LayeredStructure.add()
    which creates LayeredGeometry wrappers internally.

    Example
    -------
    >>> from tidy3d.components.geometry.geometry2d import Circle2D
    >>> # Direct creation (rarely needed)
    >>> lg = LayeredGeometry(
    ...     geometry=Circle2D(center=(0, 0), radius=0.5),
    ...     layer="top_copper",
    ...     net="GND",
    ... )
    """

    geometry: Geometry2DType = pydantic.Field(
        ...,
        title="Geometry",
        description="The 2D geometry shape.",
        discriminator=TYPE_TAG_STR,
    )

    layer: str = pydantic.Field(
        ...,
        title="Layer",
        description="Name of the layer this geometry belongs to. "
        "Must match a layer name in the stackup.",
    )

    net: Optional[str] = pydantic.Field(
        None,
        title="Net",
        description="Optional net name for grouping/querying. "
        "None indicates the geometry is not assigned to any net.",
    )

