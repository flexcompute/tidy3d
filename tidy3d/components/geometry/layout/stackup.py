"""Stackup: collection of layer specifications."""

from __future__ import annotations

from typing import Optional

import pydantic.v1 as pydantic

from tidy3d.components.base import Tidy3dBaseModel
from tidy3d.components.geometry.layout.layer_spec import LayerSpec
from tidy3d.exceptions import Tidy3dKeyError


class Stackup(Tidy3dBaseModel):
    """Collection of layer specifications defining a complete stackup.

    A stackup defines all the physical layers in a design, including
    their z-positions and materials. It is used to convert 2D geometry
    layers into 3D structures.

    Example
    -------
    >>> from tidy3d import Medium, PEC
    >>> copper = PEC
    >>> fr4 = Medium(permittivity=4.2)
    >>>
    >>> stackup = Stackup(layers=(
    ...     LayerSpec(name="substrate", z_min=-1.6, z_max=0.0, medium=fr4),
    ...     LayerSpec(name="top_copper", z_min=0.0, z_max=0.035, medium=copper),
    ...     LayerSpec(name="gnd_plane", z_min=-0.235, z_max=-0.2, medium=copper),
    ... ))
    >>>
    >>> # Access by name
    >>> top = stackup["top_copper"]
    >>> print(top.thickness)
    0.035
    >>>
    >>> # Check if layer exists
    >>> "top_copper" in stackup
    True
    """

    layers: tuple[LayerSpec, ...] = pydantic.Field(
        (),
        title="Layers",
        description="Tuple of LayerSpec objects defining the stackup.",
    )

    def __getitem__(self, name: str) -> LayerSpec:
        """Get a layer specification by name.

        Parameters
        ----------
        name : str
            Name of the layer to retrieve.

        Returns
        -------
        LayerSpec
            The layer specification with the given name.

        Raises
        ------
        Tidy3dKeyError
            If no layer with the given name exists.
        """
        for layer in self.layers:
            if layer.name == name:
                return layer
        raise Tidy3dKeyError(f"Layer '{name}' not found in stackup. "
                            f"Available layers: {self.layer_names}")

    def __contains__(self, name: str) -> bool:
        """Check if a layer with the given name exists.

        Parameters
        ----------
        name : str
            Name of the layer to check.

        Returns
        -------
        bool
            True if a layer with this name exists.
        """
        return any(layer.name == name for layer in self.layers)

    def __len__(self) -> int:
        """Number of layers in the stackup."""
        return len(self.layers)

    @property
    def layer_names(self) -> list[str]:
        """List of all layer names in the stackup."""
        return [layer.name for layer in self.layers]

    @property
    def z_bounds(self) -> tuple[float, float]:
        """Overall z-bounds of the stackup (min, max)."""
        if not self.layers:
            return (0.0, 0.0)
        z_mins = [layer.z_min for layer in self.layers]
        z_maxs = [layer.z_max for layer in self.layers]
        return (min(z_mins), max(z_maxs))

    def get(self, name: str, default: Optional[LayerSpec] = None) -> Optional[LayerSpec]:
        """Get a layer by name, returning default if not found.

        Parameters
        ----------
        name : str
            Name of the layer to retrieve.
        default : LayerSpec, optional
            Value to return if layer not found.

        Returns
        -------
        LayerSpec or None
            The layer specification, or default if not found.
        """
        for layer in self.layers:
            if layer.name == name:
                return layer
        return default

    @pydantic.validator("layers")
    def _unique_layer_names(cls, val):
        """Ensure all layer names are unique."""
        names = [layer.name for layer in val]
        duplicates = [name for name in names if names.count(name) > 1]
        if duplicates:
            raise ValueError(f"Duplicate layer names: {set(duplicates)}")
        return val

