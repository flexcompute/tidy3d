"""LayeredStructure: collection of 2D geometries organized by layer."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Optional

import pydantic.v1 as pydantic
from typing_extensions import Self

from tidy3d.components.base import Tidy3dBaseModel, cached_property
from tidy3d.components.geometry.geometry2d import Geometry2D, Geometry2DType
from tidy3d.components.geometry.layout.layered_geometry import LayeredGeometry
from tidy3d.components.geometry.layout.stackup import Stackup

if TYPE_CHECKING:
    from tidy3d.components.structure import Structure


class LayeredStructure(Tidy3dBaseModel):
    """Collection of 2D geometries organized by layer.

    The top-level container for PCB/photonic layouts. Geometries are stored
    in a flat tuple of LayeredGeometry for efficient serialization (each
    geometry stored exactly once).

    Query by layer or net using the computed `layers` and `nets` properties.

    Example
    -------
    >>> from tidy3d import PEC, Medium
    >>> from tidy3d.components.geometry.geometry2d import Circle2D, Rectangle2D
    >>> from tidy3d.components.geometry.layout import LayerSpec, Stackup, LayeredStructure
    >>>
    >>> # Define stackup
    >>> stackup = Stackup(layers=(
    ...     LayerSpec(name="substrate", z_bounds=(-1.6, 0.0), medium=Medium(permittivity=4.2)),
    ...     LayerSpec(name="top_copper", z_bounds=(0.0, 0.035), medium=PEC),
    ... ))
    >>>
    >>> # Create structure and add geometries
    >>> board = LayeredStructure(stackup=stackup)
    >>> board = board.add(
    ...     layer="top_copper",
    ...     geometries=[
    ...         Rectangle2D(center=(0, 0), size=(10, 0.5)),
    ...         Circle2D(center=(5, 0), radius=0.3),
    ...     ],
    ...     net="CLK",
    ... )
    >>>
    >>> # Query by layer
    >>> board.layers["top_copper"]  # Returns tuple of geometries
    >>>
    >>> # Query by net
    >>> board.nets["CLK"]  # Returns tuple of geometries
    """

    stackup: Stackup = pydantic.Field(
        ...,
        title="Stackup",
        description="Layer stackup defining physical layer properties.",
    )

    geometries: tuple[LayeredGeometry, ...] = pydantic.Field(
        (),
        title="Geometries",
        description="Flat tuple of LayeredGeometry objects. "
        "Use add() method to append geometries.",
    )

    def add(
        self,
        layer: str,
        geometries: Sequence[Geometry2D],
        net: Optional[str] = None,
    ) -> Self:
        """Add geometries to a layer with optional net assignment.

        Creates LayeredGeometry wrappers internally. Returns a new
        LayeredStructure with the geometries added (immutable pattern).

        Parameters
        ----------
        layer : str
            Name of the layer to add geometries to.
            Must exist in the stackup.
        geometries : Sequence[Geometry2D]
            List or tuple of 2D geometries to add.
        net : str, optional
            Net name to assign to all added geometries.

        Returns
        -------
        LayeredStructure
            New LayeredStructure with geometries added.

        Example
        -------
        >>> board = board.add("top_copper", [trace1, trace2, pad1], net="CLK")
        >>> board = board.add("gnd_plane", [gnd_fill], net="GND")
        """
        # Validate layer exists in stackup
        if layer not in self.stackup:
            raise ValueError(
                f"Layer '{layer}' not found in stackup. "
                f"Available layers: {self.stackup.layer_names}"
            )

        new_entries = tuple(
            LayeredGeometry(geometry=g, layer=layer, net=net) for g in geometries
        )
        return self.copy(update={"geometries": self.geometries + new_entries})

    @cached_property
    def layers(self) -> dict[str, tuple[Geometry2D, ...]]:
        """Group geometries by layer name.

        Returns
        -------
        dict[str, tuple[Geometry2D, ...]]
            Mapping from layer name to tuple of geometries on that layer.
        """
        result: dict[str, list[Geometry2D]] = {}
        for lg in self.geometries:
            result.setdefault(lg.layer, []).append(lg.geometry)
        return {k: tuple(v) for k, v in result.items()}

    @cached_property
    def nets(self) -> dict[str, tuple[Geometry2D, ...]]:
        """Group geometries by net name.

        Only includes geometries that have a net assigned.
        Use unassigned_geometries() for geometries without net.

        Returns
        -------
        dict[str, tuple[Geometry2D, ...]]
            Mapping from net name to tuple of geometries in that net.
        """
        result: dict[str, list[Geometry2D]] = {}
        for lg in self.geometries:
            if lg.net is not None:
                result.setdefault(lg.net, []).append(lg.geometry)
        return {k: tuple(v) for k, v in result.items()}

    @property
    def layer_names(self) -> list[str]:
        """Layer names that have geometries assigned.

        Note: This returns layers with geometries, not all stackup layers.
        """
        # Preserve order of first appearance
        seen = set()
        result = []
        for lg in self.geometries:
            if lg.layer not in seen:
                seen.add(lg.layer)
                result.append(lg.layer)
        return result

    @property
    def net_names(self) -> list[str]:
        """Net names that have geometries assigned."""
        seen = set()
        result = []
        for lg in self.geometries:
            if lg.net is not None and lg.net not in seen:
                seen.add(lg.net)
                result.append(lg.net)
        return result

    def geometries_on_layer(self, layer: str) -> tuple[Geometry2D, ...]:
        """Get all geometries on a specific layer.

        Parameters
        ----------
        layer : str
            Layer name.

        Returns
        -------
        tuple[Geometry2D, ...]
            Geometries on that layer, or empty tuple if none.
        """
        return self.layers.get(layer, ())

    def geometries_in_net(self, net: str) -> tuple[Geometry2D, ...]:
        """Get all geometries assigned to a specific net.

        Parameters
        ----------
        net : str
            Net name.

        Returns
        -------
        tuple[Geometry2D, ...]
            Geometries in that net, or empty tuple if none.
        """
        return self.nets.get(net, ())

    def unassigned_geometries(self) -> tuple[Geometry2D, ...]:
        """Get geometries not assigned to any net.

        Returns
        -------
        tuple[Geometry2D, ...]
            Geometries where net is None.
        """
        return tuple(lg.geometry for lg in self.geometries if lg.net is None)

    def __len__(self) -> int:
        """Number of geometry entries."""
        return len(self.geometries)

    @pydantic.validator("geometries", each_item=True)
    def _layer_exists_in_stackup(cls, lg: LayeredGeometry, values) -> LayeredGeometry:
        """Validate that layer names exist in stackup."""
        stackup = values.get("stackup")
        if stackup is not None and lg.layer not in stackup:
            raise ValueError(
                f"Layer '{lg.layer}' not found in stackup. "
                f"Available layers: {stackup.layer_names}"
            )
        return lg

    def to_structures(self) -> list["Structure"]:
        """Convert all geometries to 3D tidy3d Structures.

        Each LayeredGeometry is converted to a Structure using:
        - z_bounds, sidewall_angle, axis from the corresponding LayerSpec
        - medium from the LayerSpec (required)

        Returns
        -------
        list[Structure]
            List of 3D Structure objects ready for simulation.

        Raises
        ------
        ValueError
            If any LayerSpec has medium=None.

        Example
        -------
        >>> structures = board.to_structures()
        >>> sim = Simulation(..., structures=structures)
        """
        # Import here to avoid circular imports
        from tidy3d.components.structure import Structure

        structures = []
        for lg in self.geometries:
            layer_spec = self.stackup[lg.layer]

            if layer_spec.medium is None:
                raise ValueError(
                    f"LayerSpec '{layer_spec.name}' has medium=None. "
                    "A medium is required to create a Structure."
                )

            geometry_3d = lg.geometry.to_3d_geometry(
                slab_bounds=layer_spec.slab_bounds,
                axis=layer_spec.axis,
                sidewall_angle=layer_spec.sidewall_angle,
            )

            structures.append(
                Structure(geometry=geometry_3d, medium=layer_spec.medium)
            )

        return structures

