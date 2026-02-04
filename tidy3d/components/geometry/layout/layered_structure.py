"""LayeredStructure: collection of 2D geometries organized by layer."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Optional, Union

import pydantic.v1 as pydantic
from typing_extensions import Self

from tidy3d.components.base import Tidy3dBaseModel, cached_property
from tidy3d.components.geometry.geometry2d import Geometry2D, Geometry2DType
from tidy3d.components.geometry.geometry2d.base import Bound2D
from tidy3d.components.geometry.layout.layered_geometry import LayeredGeometry
from tidy3d.components.geometry.layout.stackup import Stackup
from tidy3d.components.types import Ax, Bound, Coordinate, Coordinate2D, Shapely
from tidy3d.components.viz import add_ax_if_none, equal_aspect, polygon_patch

if TYPE_CHECKING:
    from tidy3d.components.structure import Structure

# Default layer color palette for plotting
LAYER_CMAP = [
    "#689DBC",  # blue
    "#D0698E",  # pink
    "#5E6EAD",  # purple
    "#C6224E",  # red
    "#BDB3E2",  # lavender
    "#9EC3E0",  # light blue
    "#77B88D",  # green
    "#877EBC",  # violet
    "#E8A838",  # orange
    "#5DADE2",  # sky blue
]


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
        layer: Union[str, Sequence[str]],
        geometries: Union[Geometry2D, Sequence[Geometry2D]],
        net: Union[str, Sequence[str], None] = None,
    ) -> Self:
        """Add geometries to layer(s) with optional net assignment(s).

        Creates LayeredGeometry wrappers internally. Returns a new
        LayeredStructure with the geometries added (immutable pattern).

        Supports broadcasting: a single layer or net value is applied to all
        geometries. When a sequence of layers or nets is provided, it must
        match the number of geometries.

        Parameters
        ----------
        layer : str or Sequence[str]
            Layer name(s). If a single string, all geometries are added to
            that layer. If a sequence, must match length of geometries.
            All layers must exist in the stackup.
        geometries : Geometry2D or Sequence[Geometry2D]
            Single geometry or sequence of 2D geometries to add.
        net : str, Sequence[str], or None, optional
            Net name(s). If a single string or None, applied to all geometries.
            If a sequence, must match length of geometries.

        Returns
        -------
        LayeredStructure
            New LayeredStructure with geometries added.

        Example
        -------
        >>> # Single geometry
        >>> board = board.add("top_copper", trace, net="CLK")
        >>>
        >>> # Multiple geometries, single layer/net (broadcast)
        >>> board = board.add("top_copper", [trace1, trace2, pad1], net="CLK")
        >>>
        >>> # Multiple geometries, multiple layers, single net
        >>> board = board.add(["top_copper", "bottom_copper"], [trace1, trace2], net="GND")
        >>>
        >>> # Multiple geometries, single layer, multiple nets
        >>> board = board.add("top_copper", [trace1, trace2], net=["CLK", "DATA"])
        >>>
        >>> # Multiple geometries, multiple layers, multiple nets
        >>> board = board.add(["top", "bottom"], [g1, g2], net=["NET1", "NET2"])
        """
        # Normalize geometries to a list
        if isinstance(geometries, Geometry2D):
            geom_list = [geometries]
        else:
            geom_list = list(geometries)

        n = len(geom_list)

        # Normalize layer to a list
        if isinstance(layer, str):
            layer_list = [layer] * n
        else:
            layer_list = list(layer)
            if len(layer_list) != n:
                raise ValueError(
                    f"Length of layer sequence ({len(layer_list)}) must match "
                    f"number of geometries ({n})."
                )

        # Normalize net to a list
        if net is None or isinstance(net, str):
            net_list = [net] * n
        else:
            net_list = list(net)
            if len(net_list) != n:
                raise ValueError(
                    f"Length of net sequence ({len(net_list)}) must match "
                    f"number of geometries ({n})."
                )

        # Validate all layers exist in stackup
        for lyr in layer_list:
            if lyr not in self.stackup:
                raise ValueError(
                    f"Layer '{lyr}' not found in stackup. "
                    f"Available layers: {self.stackup.layer_names}"
                )

        new_entries = tuple(
            LayeredGeometry(geometry=g, layer=lyr, net=nt)
            for g, lyr, nt in zip(geom_list, layer_list, net_list)
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

    @property
    def bounds_2d(self) -> Optional[Bound2D]:
        """Get the overall 2D bounding box across all geometries.

        Returns
        -------
        Optional[Bound2D]
            Tuple of ((min_x, min_y), (max_x, max_y)), or None if no geometries.
        """
        if not self.geometries:
            return None

        min_x = min_y = float("inf")
        max_x = max_y = float("-inf")

        for lg in self.geometries:
            (x0, y0), (x1, y1) = lg.geometry.bounds_2d
            min_x = min(min_x, x0)
            min_y = min(min_y, y0)
            max_x = max(max_x, x1)
            max_y = max(max_y, y1)

        return ((min_x, min_y), (max_x, max_y))

    @property
    def bounds_3d(self) -> Optional[Bound]:
        """Get the full 3D bounding box including layer heights from stackup.

        Only considers z-bounds of layers that have geometries assigned.

        Returns
        -------
        Optional[Bound]
            Tuple of ((min_x, min_y, min_z), (max_x, max_y, max_z)),
            or None if no geometries.
        """
        bounds_2d = self.bounds_2d
        if bounds_2d is None:
            return None

        (min_x, min_y), (max_x, max_y) = bounds_2d

        # Get z-bounds only from layers that have geometries
        min_z = float("inf")
        max_z = float("-inf")

        for layer_name in self.layer_names:
            layer_spec = self.stackup[layer_name]
            z0, z1 = layer_spec.z_bounds
            min_z = min(min_z, z0)
            max_z = max(max_z, z1)

        return ((min_x, min_y, min_z), (max_x, max_y, max_z))

    @property
    def center_2d(self) -> Optional[Coordinate2D]:
        """Get the center point of the 2D bounding box.

        Returns
        -------
        Optional[Coordinate2D]
            Tuple of (center_x, center_y), or None if no geometries.
        """
        bounds = self.bounds_2d
        if bounds is None:
            return None

        (min_x, min_y), (max_x, max_y) = bounds
        return ((min_x + max_x) / 2, (min_y + max_y) / 2)

    @property
    def size_2d(self) -> Optional[Coordinate2D]:
        """Get the size of the 2D bounding box.

        Returns
        -------
        Optional[Coordinate2D]
            Tuple of (size_x, size_y), or None if no geometries.
        """
        bounds = self.bounds_2d
        if bounds is None:
            return None

        (min_x, min_y), (max_x, max_y) = bounds
        return (max_x - min_x, max_y - min_y)

    @property
    def center_3d(self) -> Optional[Coordinate]:
        """Get the center point of the 3D bounding box.

        Returns
        -------
        Optional[Coordinate]
            Tuple of (center_x, center_y, center_z), or None if no geometries.
        """
        bounds = self.bounds_3d
        if bounds is None:
            return None

        (min_x, min_y, min_z), (max_x, max_y, max_z) = bounds
        return ((min_x + max_x) / 2, (min_y + max_y) / 2, (min_z + max_z) / 2)

    @property
    def size_3d(self) -> Optional[Coordinate]:
        """Get the size of the 3D bounding box.

        Returns
        -------
        Optional[Coordinate]
            Tuple of (size_x, size_y, size_z), or None if no geometries.
        """
        bounds = self.bounds_3d
        if bounds is None:
            return None

        (min_x, min_y, min_z), (max_x, max_y, max_z) = bounds
        return (max_x - min_x, max_y - min_y, max_z - min_z)

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

    def to_structures(self, split_by_net: bool = False) -> list["Structure"]:
        """Convert all geometries to 3D tidy3d Structures.

        Geometries are grouped and combined into a GeometryGroup per group.
        This is more efficient than creating one Structure per geometry.

        Parameters
        ----------
        split_by_net : bool, optional
            If False (default), geometries are grouped by layer only, producing
            one Structure per layer named "<layer>".
            If True, geometries are grouped by (layer, net), producing one
            Structure per unique combination named "<layer>_<net>" (or "<layer>"
            if net is None).

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
        >>> structures = board.to_structures()  # One per layer
        >>> structures = board.to_structures(split_by_net=True)  # One per (layer, net)
        >>> sim = Simulation(..., structures=structures)
        """
        from collections import defaultdict

        from tidy3d.components.geometry.base import GeometryGroup
        from tidy3d.components.structure import Structure

        # Group geometries by layer only, or by (layer, net)
        if split_by_net:
            groups: dict[tuple[str, Optional[str]], list[Geometry2D]] = defaultdict(list)
            for lg in self.geometries:
                groups[(lg.layer, lg.net)].append(lg.geometry)
        else:
            groups: dict[tuple[str, Optional[str]], list[Geometry2D]] = defaultdict(list)
            for lg in self.geometries:
                groups[(lg.layer, None)].append(lg.geometry)

        structures = []
        for (layer_name, net_name), geom_2d_list in groups.items():
            layer_spec = self.stackup[layer_name]

            if layer_spec.medium is None:
                raise ValueError(
                    f"LayerSpec '{layer_spec.name}' has medium=None. "
                    "A medium is required to create a Structure."
                )

            # Convert all 2D geometries to 3D, flattening any nested GeometryGroups
            geometries_3d = []
            for g in geom_2d_list:
                geom_3d = g.to_3d_geometry(
                    slab_bounds=layer_spec.slab_bounds,
                    axis=layer_spec.axis,
                    sidewall_angle=layer_spec.sidewall_angle,
                )
                if isinstance(geom_3d, GeometryGroup):
                    geometries_3d.extend(geom_3d.geometries)
                else:
                    geometries_3d.append(geom_3d)

            # Build structure name: <layer>_<net> or just <layer> if no net
            if split_by_net and net_name is not None:
                struct_name = f"{layer_name}_{net_name}"
            else:
                struct_name = layer_name

            # Use geometry directly if only one, otherwise wrap in GeometryGroup
            if len(geometries_3d) == 1:
                geometry = geometries_3d[0]
            else:
                geometry = GeometryGroup(geometries=tuple(geometries_3d))

            structures.append(
                Structure(geometry=geometry, medium=layer_spec.medium, name=struct_name)
            )

        return structures

    @equal_aspect
    @add_ax_if_none
    def plot(
        self,
        layers: Optional[Union[str, Sequence[str]]] = None,
        nets: Optional[Union[str, Sequence[str]]] = None,
        ax: Ax = None,
        xlim: Optional[tuple[float, float]] = None,
        ylim: Optional[tuple[float, float]] = None,
        alpha: float = 0.8,
        edgecolor: str = "black",
        linewidth: float = 0.5,
        layer_colors: Optional[dict[str, str]] = None,
        show_legend: bool = True,
        **patch_kwargs: Any,
    ) -> Ax:
        """Plot 2D geometries on a matplotlib axes.

        Renders the 2D geometries from selected layers, with optional net
        filtering. Each layer is assigned a distinct color from a default
        palette or custom color mapping.

        Parameters
        ----------
        layers : str, Sequence[str], or None
            Layer name(s) to plot. If None, plots all layers with geometries.
        nets : str, Sequence[str], or None
            Net name(s) to filter. If None, no net filtering is applied
            (all geometries on selected layers are plotted).
        ax : matplotlib.axes._subplots.Axes
            Matplotlib axes to plot on. If None, one is created.
        xlim : tuple[float, float], optional
            X-axis limits (min_x, max_x). Auto-calculated from bounds if None.
        ylim : tuple[float, float], optional
            Y-axis limits (min_y, max_y). Auto-calculated from bounds if None.
        alpha : float
            Opacity for fill color (0-1). Default 0.8.
        edgecolor : str
            Edge color for all shapes. Default "black".
        linewidth : float
            Line width for edges. Default 0.5.
        layer_colors : dict[str, str], optional
            Custom color mapping {layer_name: color}. Uses default colormap
            for layers not specified.
        show_legend : bool
            Whether to show a legend for layers. Default True.
        **patch_kwargs
            Additional kwargs passed to matplotlib patches.

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The matplotlib axes with plotted geometries.

        Example
        -------
        >>> # Plot all layers
        >>> board.plot()
        >>>
        >>> # Plot specific layers
        >>> board.plot(layers=["top_copper", "bottom_copper"])
        >>>
        >>> # Plot single layer filtered by net
        >>> board.plot(layers="top_copper", nets="CLK")
        >>>
        >>> # Custom colors and styling
        >>> board.plot(
        ...     layer_colors={"top_copper": "gold", "substrate": "#8B4513"},
        ...     alpha=0.6,
        ...     edgecolor="navy",
        ... )
        """
        from matplotlib.collections import PatchCollection

        # Determine which layers to plot
        if layers is None:
            layers_to_plot = self.layer_names
        elif isinstance(layers, str):
            layers_to_plot = [layers]
        else:
            layers_to_plot = list(layers)

        # Validate layers exist
        for lyr in layers_to_plot:
            if lyr not in self.stackup:
                raise ValueError(
                    f"Layer '{lyr}' not found in stackup. "
                    f"Available layers: {self.stackup.layer_names}"
                )

        # Build color map for layers
        if layer_colors is None:
            layer_colors = {}
        color_map = {}
        for i, layer in enumerate(layers_to_plot):
            color_map[layer] = layer_colors.get(layer, LAYER_CMAP[i % len(LAYER_CMAP)])

        # Build net filter set
        net_set: Optional[set[str]] = None
        if nets is not None:
            net_set = {nets} if isinstance(nets, str) else set(nets)

        # Determine view bounds for pre-filtering (if xlim/ylim specified)
        # Use slightly expanded bounds to avoid clipping edge geometries
        view_x_min = xlim[0] if xlim else float("-inf")
        view_x_max = xlim[1] if xlim else float("inf")
        view_y_min = ylim[0] if ylim else float("-inf")
        view_y_max = ylim[1] if ylim else float("inf")
        has_view_filter = xlim is not None or ylim is not None

        # Collect patches per layer for batch rendering
        layer_patches: dict[str, list] = {lyr: [] for lyr in layers_to_plot}
        # Separate storage for lines and points (can't go in PatchCollection)
        layer_lines: dict[str, list[tuple]] = {lyr: [] for lyr in layers_to_plot}
        layer_points: dict[str, list[tuple]] = {lyr: [] for lyr in layers_to_plot}

        # Collect geometries
        for lg in self.geometries:
            if lg.layer not in layers_to_plot:
                continue
            if net_set is not None and lg.net not in net_set:
                continue

            # Bounding box pre-filtering: skip geometries outside view bounds
            if has_view_filter:
                (gx0, gy0), (gx1, gy1) = lg.geometry.bounds_2d
                if gx1 < view_x_min or gx0 > view_x_max:
                    continue  # Entirely outside x range
                if gy1 < view_y_min or gy0 > view_y_max:
                    continue  # Entirely outside y range

            # Convert to shapely and collect patches
            shape = lg.geometry.to_shapely()
            self._collect_patches(
                shape=shape,
                patches_list=layer_patches[lg.layer],
                lines_list=layer_lines[lg.layer],
                points_list=layer_points[lg.layer],
            )

        # Render patches using PatchCollection (batch rendering for performance)
        plotted_layers: set[str] = set()
        for layer in layers_to_plot:
            patches = layer_patches[layer]
            lines = layer_lines[layer]
            points = layer_points[layer]

            if patches or lines or points:
                plotted_layers.add(layer)
                facecolor = color_map[layer]

                # Add polygon patches as a collection
                if patches:
                    collection = PatchCollection(
                        patches,
                        facecolors=facecolor,
                        edgecolors=edgecolor,
                        alpha=alpha,
                        linewidths=linewidth,
                        **patch_kwargs,
                    )
                    ax.add_collection(collection)

                # Add lines (can't be batched in PatchCollection)
                for xs, ys in lines:
                    ax.plot(xs, ys, color=facecolor, linewidth=linewidth)

                # Add points
                if points:
                    px = [p[0] for p in points]
                    py = [p[1] for p in points]
                    ax.scatter(px, py, color=facecolor, s=20)

        # Set axis limits
        bounds = self.bounds_2d
        if xlim is not None:
            ax.set_xlim(xlim)
        elif bounds is not None:
            (x0, _), (x1, _) = bounds
            margin = (x1 - x0) * 0.05 if x1 > x0 else 0.1
            ax.set_xlim(x0 - margin, x1 + margin)

        if ylim is not None:
            ax.set_ylim(ylim)
        elif bounds is not None:
            (_, y0), (_, y1) = bounds
            margin = (y1 - y0) * 0.05 if y1 > y0 else 0.1
            ax.set_ylim(y0 - margin, y1 + margin)

        # Add legend
        if show_legend and plotted_layers:
            from matplotlib.patches import Patch

            # Preserve layer order from layers_to_plot
            handles = [
                Patch(facecolor=color_map[lyr], edgecolor=edgecolor, label=lyr, alpha=alpha)
                for lyr in layers_to_plot
                if lyr in plotted_layers
            ]
            if handles:
                ax.legend(handles=handles, loc="best")

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("LayeredStructure")

        return ax

    def _collect_patches(
        self,
        shape: Shapely,
        patches_list: list,
        lines_list: list[tuple],
        points_list: list[tuple],
    ) -> None:
        """Collect matplotlib patches from a shapely geometry.

        Flattens multi-geometries and separates polygons, lines, and points
        for efficient batch rendering.
        """
        # Handle multi-geometries by flattening
        if shape.geom_type in ("MultiPolygon", "GeometryCollection", "MultiLineString"):
            for sub_shape in shape.geoms:
                self._collect_patches(sub_shape, patches_list, lines_list, points_list)
            return

        if shape.geom_type == "Polygon" and not shape.is_empty:
            # Create patch but don't set colors - PatchCollection will handle that
            patch = polygon_patch(shape)
            patches_list.append(patch)
        elif shape.geom_type == "LineString" and not shape.is_empty:
            xs, ys = zip(*shape.coords)
            lines_list.append((xs, ys))
        elif shape.geom_type == "Point" and not shape.is_empty:
            points_list.append((shape.x, shape.y))
        # Silently skip unsupported geometry types

