"""ODB++ to LayeredStructure loader.

High-level API for loading ODB++ PCB design files into tidy3d's
2D geometry layer system.
"""

from __future__ import annotations

import math
import warnings
import tidy3d as td
from pathlib import Path
from typing import Generator, List, Optional, Union

from tidy3d.components.geometry.geometry2d import (
    Array2D,
    Circle2D,
    Geometry2D,
    Path2D,
    Polygon2D,
    Rectangle2D,
    Transformed2D,
)
from tidy3d.components.geometry.layout import (
    LayeredGeometry,
    LayeredStructure,
    LayerSpec,
    Stackup,
)
from tidy3d.plugins.odb.parser import (
    ArcRecord,
    EDAData,
    FeaturesData,
    LineRecord,
    MatrixData,
    PadRecord,
    PolygonContour,
    SurfaceRecord,
    read_eda_data,
    read_features,
    read_matrix,
    read_profile_data,
)
from tidy3d.plugins.odb.symbols import SymbolInfo, parse_symbol
from tidy3d.plugins.odb.stackup_builder import StackupBuilder
from tidy3d.plugins.odb.utils import (
    arc_to_bulge,
    convert_to_microns,
    is_full_circle,
    split_full_circle,
)

# Tolerance for endpoint matching (in microns) - matches RF GUI approach
MERGE_TOLERANCE = 1e-6


def _endpoints_match(x1: float, y1: float, x2: float, y2: float) -> bool:
    """Check if two points are within tolerance."""
    return abs(x1 - x2) <= MERGE_TOLERANCE and abs(y1 - y2) <= MERGE_TOLERANCE


def _widths_match(w1: float, w2: float) -> bool:
    """Check if two widths are within tolerance."""
    return abs(w1 - w2) <= MERGE_TOLERANCE


class ODBLoader:
    """Load ODB++ files into LayeredStructure.

    Parameters
    ----------
    path : str or Path
        Path to ODB++ directory (unzipped).

    Example
    -------
    >>> from tidy3d.plugins.odb import ODBLoader
    >>>
    >>> loader = ODBLoader("./my_design.odb")
    >>> structure = loader.load()
    >>>
    >>> # Check what was loaded
    >>> print(structure.layer_names)
    >>> print(len(structure.geometries))
    >>>
    >>> # Filter to specific layers
    >>> structure = loader.load(layers=["TRACE", "GND"])

    Notes
    -----
    This is the MVP implementation with the following limitations:

    - Trace grouping not implemented (each L/A record is a separate Path2D)
    - Unknown symbols are skipped with warning
    - Oval symbols are approximated as rectangles
    - Rotation for Rectangle2D not fully supported
    - Stackup z_bounds are placeholders (user must configure)
    """

    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"ODB++ path not found: {self.path}")

        self._matrix: Optional[MatrixData] = None
        self._warnings: list[str] = []

    @property
    def matrix(self) -> MatrixData:
        """Parsed matrix data (lazy loaded).

        Returns
        -------
        MatrixData
            Parsed layer and step definitions.
        """
        if self._matrix is None:
            self._matrix = read_matrix(self.path)
        return self._matrix

    @property
    def layer_names(self) -> list[str]:
        """List of layer names from the matrix file.

        Returns
        -------
        list[str]
            Layer names in stackup order.
        """
        return [layer.name for layer in self.matrix.layers]

    @property
    def step_names(self) -> list[str]:
        """List of step names from the matrix file.

        Returns
        -------
        list[str]
            Step names.
        """
        return [step.name for step in self.matrix.steps]

    def load(
        self,
        step: Optional[str] = None,
        layers: Optional[list[str]] = None,
        stackup: Optional[Stackup] = None,
        layer_thicknesses: Optional[dict[str, float]] = None,
        default_conductor_thickness: float = 35.0,
        default_dielectric_thickness: float = 200.0,
        default_permittivity: float = 4.2,
        include_dielectrics: bool = True,
        frequency_range: tuple[float, float] = (0.1e9, 10e9),
        use_lossy_metal: bool = True,
        use_lossy_dielectric: bool = True,
        group_traces: bool = True,
    ) -> LayeredStructure:
        """Load ODB++ into LayeredStructure.

        Parameters
        ----------
        step : str, optional
            Step name to load. Defaults to first step.
        layers : list[str], optional
            Layer names to load. Defaults to all signal/power layers.
        stackup : Stackup, optional
            User-provided stackup. If provided, skips automatic stackup construction.
        layer_thicknesses : dict[str, float], optional
            Per-layer thickness overrides in microns.
        default_conductor_thickness : float
            Default conductor thickness in microns (default: 35 µm = 1 oz copper).
        default_dielectric_thickness : float
            Default dielectric thickness in microns (default: 200 µm).
        default_permittivity : float
            Default dielectric constant for FR4 (default: 4.2).
        include_dielectrics : bool
            If True, include dielectric layers in stackup (default: True).
        frequency_range : tuple[float, float]
            Frequency range (f_min, f_max) in Hz for lossy material models.
            Default is (0.1e9, 10e9) Hz.
        use_lossy_metal : bool
            If True, use LossyMetalMedium for conductor layers (with conductivity
            from ODB++ attributes or default copper value of 58 S/µm).
            If False, use PECMedium. Default is True.
        use_lossy_dielectric : bool
            If True, use FastDispersionFitter.constant_loss_tangent_model() for
            dielectric layers with loss tangent from ODB++ attributes.
            If False, use lossless Medium. Default is True.
        group_traces : bool
            If True (default), merge consecutive connected Line/Arc records
            into multi-segment Path2D objects. This reduces geometry count
            and produces proper traces with end caps only at trace endpoints.

        Returns
        -------
        LayeredStructure
            Loaded geometry organized by layers with proper stackup.

        Notes
        -----
        - Stackup z_bounds are in microns (tidy3d default units)
        - Conductor layers use LossyMetalMedium by default (conductivity from
          bulk_resistivity attribute, or 58 S/µm for copper)
        - Dielectric layers use constant_loss_tangent_model if loss tangent is
          available in ODB++ attributes
        - Unknown symbols are skipped with warning
        - Trace grouping merges consecutive L/A records with matching width,
          net, and connected endpoints into single Path2D objects

        Example
        -------
        >>> loader = ODBLoader("./design.odb")
        >>> # Load with automatic stackup and lossy materials
        >>> structure = loader.load()
        >>>
        >>> # Load with PEC conductors (faster, simpler)
        >>> structure = loader.load(use_lossy_metal=False)
        >>>
        >>> # Load with custom frequency range
        >>> structure = loader.load(frequency_range=(1e9, 20e9))
        >>>
        >>> # Load with user-provided stackup
        >>> from tidy3d.components.geometry.layout import Stackup, LayerSpec
        >>> my_stackup = Stackup(layers=(...))
        >>> structure = loader.load(stackup=my_stackup)
        """
        self._warnings.clear()

        # Determine step
        if step is None:
            if not self.matrix.steps:
                raise ValueError("No steps found in ODB++ matrix")
            step = self.matrix.steps[0].name

        # Build layer filter
        layer_filter = set(layers) if layers else None

        # Build stackup
        if stackup is None:
            # Use StackupBuilder to construct from ODB++ data
            builder = StackupBuilder(self.path, self.matrix, step)

            # Apply user-provided thickness overrides
            if layer_thicknesses:
                for layer_info in builder.stackup_data.layers:
                    if layer_info.name in layer_thicknesses:
                        layer_info.thickness = layer_thicknesses[layer_info.name]

            stackup = builder.build(
                layer_filter=layer_filter,
                default_conductor_thickness=default_conductor_thickness,
                default_dielectric_thickness=default_dielectric_thickness,
                default_permittivity=default_permittivity,
                include_dielectrics=include_dielectrics,
                frequency_range=frequency_range,
                use_lossy_metal=use_lossy_metal,
                use_lossy_dielectric=use_lossy_dielectric,
            )

        # Determine which layers have geometry to load
        # (only load layers that are in the stackup)
        stackup_layer_names = {spec.name for spec in stackup.layers}

        # Read step profile (board outline) for auto-filling empty dielectric layers
        profile_data = read_profile_data(self.path, step)
        step_profile = profile_data.surface
        profile_units = profile_data.units

        # Read EDA data for net assignments
        eda_data = read_eda_data(self.path, step)

        # Build case-insensitive layer name mapping for EDA data
        # EDA layer names may be lowercase while matrix names are uppercase
        eda_layer_map: dict[str, str] = {}
        for eda_layer in eda_data.layer_names:
            eda_layer_map[eda_layer.upper()] = eda_layer

        # Collect LayeredGeometry objects directly using construct() to skip
        # per-object validation. This is safe because:
        # 1. Geometry2D objects are already validated when created
        # 2. Layer names are checked against stackup below
        # 3. Net names are just strings from parsed EDA data
        all_layered_geoms: list[LayeredGeometry] = []

        # Load each layer's features
        for layer_def in self.matrix.layers:
            # Skip if not in filter
            if layer_filter and layer_def.name not in layer_filter:
                continue

            # Skip if not in stackup (e.g., document layers)
            if layer_def.name not in stackup_layer_names:
                continue

            features_data = read_features(self.path, step, layer_def.name)

            # Convert features with net assignments
            # Returns list of (geometry, net_name) tuples
            # Use EDA layer name (may be different case than matrix layer name)
            eda_layer_name = eda_layer_map.get(layer_def.name.upper(), layer_def.name)
            
            # Use optimized DRILL layer conversion for via/drill layers
            if layer_def.type == "DRILL":
                geom_net_pairs = list(
                    self._convert_drill_features_with_nets(
                        features_data, eda_layer_name, eda_data
                    )
                )
            else:
                geom_net_pairs = list(
                    self._convert_features_with_nets(
                        features_data, eda_layer_name, eda_data, group_traces
                    )
                )

            # Auto-fill empty dielectric layers with board profile
            if not geom_net_pairs and layer_def.type in ("DIELECTRIC", "SOLDER_MASK"):
                # Try layer-specific profile first, then step profile
                layer_profile_data = read_profile_data(self.path, step, layer_def.name)
                profile = layer_profile_data.surface or step_profile
                units = layer_profile_data.units if layer_profile_data.surface else profile_units

                if profile:
                    profile_geom = self._convert_surface(profile, units)
                    if profile_geom:
                        geom_net_pairs = [(profile_geom, None)]
                        self._warnings.append(
                            f"Layer '{layer_def.name}' has no features, using board profile"
                        )
                else:
                    self._warnings.append(
                        f"Dielectric layer '{layer_def.name}' is empty and no profile found"
                    )

            # Build LayeredGeometry objects directly, skipping per-object validation
            for geom, net in geom_net_pairs:
                all_layered_geoms.append(
                    LayeredGeometry.construct(
                        geometry=geom,
                        layer=layer_def.name,
                        net=net,
                    )
                )

        # Create structure with pre-built geometries (final validation happens here)
        structure = LayeredStructure(
            stackup=stackup,
            geometries=tuple(all_layered_geoms),
        )

        # Report warnings
        for warning in self._warnings:
            warnings.warn(warning, stacklevel=2)

        return structure

    def _convert_features(
        self, data: FeaturesData, layer_name: str
    ) -> Generator[Geometry2D, None, None]:
        """Convert parsed features to Geometry2D objects.

        Parameters
        ----------
        data : FeaturesData
            Parsed features data.
        layer_name : str
            Layer name (for error messages).

        Yields
        ------
        Geometry2D
            Converted geometry objects.
        """
        for geom, _ in self._convert_features_with_nets(
            data, layer_name, EDAData(), group_traces=True
        ):
            yield geom

    def _convert_features_with_nets(
        self,
        data: FeaturesData,
        layer_name: str,
        eda_data: EDAData,
        group_traces: bool = True,
    ) -> Generator[tuple[Geometry2D, Optional[str]], None, None]:
        """Convert parsed features to Geometry2D objects with net assignments.

        Parameters
        ----------
        data : FeaturesData
            Parsed features data.
        layer_name : str
            Layer name (for net lookup and error messages).
        eda_data : EDAData
            Parsed EDA data for net assignments.
        group_traces : bool
            If True, merge consecutive connected Line/Arc records into
            multi-segment Path2D objects.

        Yields
        ------
        tuple[Geometry2D, Optional[str]]
            (geometry, net_name) tuples. net_name is None if not assigned.

        Notes
        -----
        Net assignment is looked up from two sources:
        1. EDA data file (eda/data) via layer_name and feature index
        2. Feature-level attributes (.net_name attribute in features file)
        Source 1 takes precedence if both are present.
        """
        units = data.units
        features = data.features
        symbols = data.symbols

        # Find .net_name attribute index in features file
        net_name_attr_idx: Optional[int] = None
        for attr_idx, attr_name in data.attr_names.items():
            if attr_name == ".net_name":
                net_name_attr_idx = attr_idx
                break

        def get_net_from_feature(feature) -> Optional[str]:
            """Get net name from feature's own attributes."""
            if net_name_attr_idx is None:
                return None
            if not hasattr(feature, "attributes") or not feature.attributes:
                return None
            if net_name_attr_idx not in feature.attributes:
                return None
            try:
                text_idx = int(feature.attributes[net_name_attr_idx])
                net = data.attr_texts.get(text_idx)
                return None if net == "$NONE$" else net
            except (ValueError, TypeError):
                return None

        def get_net(idx: int, feature=None) -> Optional[str]:
            """Look up net for feature index, checking EDA data then feature attributes."""
            # First try EDA data
            net = eda_data.net_assignments.get((layer_name, idx))
            if net is not None:
                return None if net == "$NONE$" else net
            # Fall back to feature attributes
            if feature is not None:
                return get_net_from_feature(feature)
            return None

        i = 0
        while i < len(features):
            feature = features[i]
            net_name = get_net(i, feature)

            # Try trace grouping for Line/Arc records
            if group_traces and isinstance(feature, (LineRecord, ArcRecord)):
                # Find extent of mergeable consecutive records
                merge_end = self._find_merge_extent(
                    features, i, symbols, units, layer_name, eda_data,
                    data.attr_names, data.attr_texts
                )

                if merge_end > i:
                    # Create merged multi-segment Path2D
                    geom = self._create_merged_trace(
                        features[i : merge_end + 1], symbols, units
                    )
                    if geom:
                        yield geom, net_name
                    i = merge_end + 1
                    continue

            # Single feature conversion
            if isinstance(feature, LineRecord):
                geom = self._convert_line(feature, symbols, units)
                if geom:
                    yield geom, net_name

            elif isinstance(feature, ArcRecord):
                geom = self._convert_arc(feature, symbols, units)
                if geom:
                    yield geom, net_name

            elif isinstance(feature, PadRecord):
                geom = self._convert_pad(feature, symbols, units)
                if geom:
                    yield geom, net_name

            elif isinstance(feature, SurfaceRecord):
                geom = self._convert_surface(feature, units)
                if geom:
                    yield geom, net_name

            i += 1

    def _convert_drill_features_with_nets(
        self,
        data: FeaturesData,
        layer_name: str,
        eda_data: EDAData,
    ) -> Generator[tuple[Geometry2D, Optional[str]], None, None]:
        """Convert DRILL layer features to Array2D objects grouped by (symbol, net).

        For DRILL layers (vias), this method groups vias by their symbol (size/shape)
        and net assignment, creating compact Array2D objects instead of individual
        Circle2D objects.

        Parameters
        ----------
        data : FeaturesData
            Parsed features data.
        layer_name : str
            Layer name (for net lookup and error messages).
        eda_data : EDAData
            Parsed EDA data for net assignments.

        Yields
        ------
        tuple[Geometry2D, Optional[str]]
            (geometry, net_name) tuples. geometry is Array2D for grouped vias.
        """
        units = data.units
        features = data.features
        symbols = data.symbols

        # Find .net_name attribute index in features file
        net_name_attr_idx: Optional[int] = None
        for attr_idx, attr_name in data.attr_names.items():
            if attr_name == ".net_name":
                net_name_attr_idx = attr_idx
                break

        def get_net_from_feature(feature) -> Optional[str]:
            """Get net name from feature's own attributes."""
            if net_name_attr_idx is None:
                return None
            if not hasattr(feature, "attributes") or not feature.attributes:
                return None
            if net_name_attr_idx not in feature.attributes:
                return None
            try:
                text_idx = int(feature.attributes[net_name_attr_idx])
                net = data.attr_texts.get(text_idx)
                return None if net == "$NONE$" else net
            except (ValueError, TypeError):
                return None

        def get_net(idx: int, feature=None) -> Optional[str]:
            """Look up net for feature index."""
            # First try EDA data
            net = eda_data.net_assignments.get((layer_name, idx))
            if net is not None:
                return None if net == "$NONE$" else net
            # Fall back to feature attributes
            if feature is not None:
                return get_net_from_feature(feature)
            return None

        # Group pads by (symbol_name, net_name)
        # Key: (symbol_name, net_name)
        # Value: list of (x, y) positions
        groups: dict[tuple[str, Optional[str]], list[tuple[float, float]]] = {}
        
        # Track non-pad features to yield individually
        non_pad_features: list[tuple[int, object]] = []

        for i, feature in enumerate(features):
            if isinstance(feature, PadRecord):
                sym_name = symbols.get(feature.symbol_num, "")
                if not sym_name:
                    self._warnings.append(
                        f"Pad record references missing symbol {feature.symbol_num}"
                    )
                    continue

                net_name = get_net(i, feature)
                x = convert_to_microns(feature.x, units)
                y = convert_to_microns(feature.y, units)

                key = (sym_name, net_name)
                if key not in groups:
                    groups[key] = []
                groups[key].append((x, y))
            else:
                # Non-pad features (lines, arcs, surfaces) - track for individual yield
                non_pad_features.append((i, feature))

        # Convert each group to Array2D
        for (sym_name, net_name), positions in groups.items():
            sym_info = parse_symbol(sym_name)
            
            if sym_info.type == "unknown":
                self._warnings.append(f"Unknown symbol '{sym_name}' in DRILL layer, skipping")
                continue

            # Create base shape at origin
            base_shape: Optional[Geometry2D] = None

            if sym_info.type == "round":
                diameter = convert_to_microns(
                    sym_info.params["diameter"], units, is_symbol_dim=True
                )
                base_shape = Circle2D(center=(0, 0), radius=diameter / 2)

            elif sym_info.type == "square":
                side = convert_to_microns(sym_info.params["side"], units, is_symbol_dim=True)
                base_shape = Rectangle2D(center=(0, 0), size=(side, side))

            elif sym_info.type == "rect":
                width = convert_to_microns(sym_info.params["width"], units, is_symbol_dim=True)
                height = convert_to_microns(sym_info.params["height"], units, is_symbol_dim=True)
                base_shape = Rectangle2D(center=(0, 0), size=(width, height))

            elif sym_info.type == "oval":
                # Oval (stadium shape) - use accurate representation
                width = convert_to_microns(sym_info.params["width"], units, is_symbol_dim=True)
                height = convert_to_microns(sym_info.params["height"], units, is_symbol_dim=True)
                base_shape = self._create_oval_polygon(width, height)

            elif sym_info.type == "diamond":
                # Diamond (rhombus)
                width = convert_to_microns(sym_info.params["width"], units, is_symbol_dim=True)
                height = convert_to_microns(sym_info.params["height"], units, is_symbol_dim=True)
                hw, hh = width / 2, height / 2
                vertices = [(hw, 0), (0, hh), (-hw, 0), (0, -hh)]
                base_shape = Polygon2D(vertices=vertices)

            elif sym_info.type == "octagon":
                # Octagon (chamfered rectangle)
                width = convert_to_microns(sym_info.params["width"], units, is_symbol_dim=True)
                height = convert_to_microns(sym_info.params["height"], units, is_symbol_dim=True)
                corner = convert_to_microns(sym_info.params["corner"], units, is_symbol_dim=True)
                vertices = self._create_octagon_vertices(width, height, corner)
                base_shape = Polygon2D(vertices=vertices)

            else:
                # For other symbol types (donuts, rounded rect, etc.), fall back to individual conversion
                # These shapes often have holes which can't be efficiently grouped in Array2D
                for px, py in positions:
                    geom = self._convert_symbol_to_geometry(sym_name, px, py, units)
                    if geom:
                        yield geom, net_name
                continue

            if base_shape is None:
                continue

            # Create Array2D with all positions
            array = Array2D(
                base_shape=base_shape,
                positions=tuple(positions),
            )
            yield array, net_name

        # Yield non-pad features individually (lines, arcs, surfaces in drill layers)
        for i, feature in non_pad_features:
            net_name = get_net(i, feature)

            if isinstance(feature, LineRecord):
                geom = self._convert_line(feature, symbols, units)
                if geom:
                    yield geom, net_name

            elif isinstance(feature, ArcRecord):
                geom = self._convert_arc(feature, symbols, units)
                if geom:
                    yield geom, net_name

            elif isinstance(feature, SurfaceRecord):
                geom = self._convert_surface(feature, units)
                if geom:
                    yield geom, net_name

    def _convert_symbol_to_geometry(
        self, sym_name: str, x: float, y: float, units: str
    ) -> Optional[Geometry2D]:
        """Convert a symbol to geometry at a specific position.

        Helper for DRILL layer fallback when symbol type doesn't support Array2D grouping.
        Uses accurate geometry representations for all symbol types.

        Parameters
        ----------
        sym_name : str
            Symbol name string.
        x, y : float
            Position in microns.
        units : str
            Units string for symbol dimensions.

        Returns
        -------
        Optional[Geometry2D]
            Geometry at position, or None if conversion failed.
        """
        sym_info = parse_symbol(sym_name)

        if sym_info.type == "unknown":
            return None

        if sym_info.type == "round":
            diameter = convert_to_microns(sym_info.params["diameter"], units, is_symbol_dim=True)
            return Circle2D(center=(x, y), radius=diameter / 2)

        elif sym_info.type == "square":
            side = convert_to_microns(sym_info.params["side"], units, is_symbol_dim=True)
            return Rectangle2D(center=(x, y), size=(side, side))

        elif sym_info.type == "rect":
            width = convert_to_microns(sym_info.params["width"], units, is_symbol_dim=True)
            height = convert_to_microns(sym_info.params["height"], units, is_symbol_dim=True)
            return Rectangle2D(center=(x, y), size=(width, height))

        elif sym_info.type == "oval":
            # Oval (stadium shape)
            width = convert_to_microns(sym_info.params["width"], units, is_symbol_dim=True)
            height = convert_to_microns(sym_info.params["height"], units, is_symbol_dim=True)
            base = self._create_oval_polygon(width, height)
            return self._translate_polygon(base, x, y)

        elif sym_info.type == "diamond":
            # Diamond (rhombus)
            width = convert_to_microns(sym_info.params["width"], units, is_symbol_dim=True)
            height = convert_to_microns(sym_info.params["height"], units, is_symbol_dim=True)
            hw, hh = width / 2, height / 2
            vertices = [(x + hw, y), (x, y + hh), (x - hw, y), (x, y - hh)]
            return Polygon2D(vertices=vertices)

        elif sym_info.type == "octagon":
            # Octagon (chamfered rectangle)
            width = convert_to_microns(sym_info.params["width"], units, is_symbol_dim=True)
            height = convert_to_microns(sym_info.params["height"], units, is_symbol_dim=True)
            corner = convert_to_microns(sym_info.params["corner"], units, is_symbol_dim=True)
            vertices = self._create_octagon_vertices(width, height, corner)
            return Polygon2D(vertices=[(vx + x, vy + y) for vx, vy in vertices])

        elif sym_info.type == "donut_r":
            # Round donut (annular ring)
            outer_d = convert_to_microns(
                sym_info.params["outer_diameter"], units, is_symbol_dim=True
            )
            inner_d = convert_to_microns(
                sym_info.params["inner_diameter"], units, is_symbol_dim=True
            )
            outer_r = outer_d / 2
            inner_r = inner_d / 2
            # Use quarter-circle bulges for accurate circle
            bulge_quarter = math.tan(math.pi / 8)
            vertices = [
                (x + outer_r, y),
                (x, y + outer_r),
                (x - outer_r, y),
                (x, y - outer_r),
            ]
            bulges = (bulge_quarter, bulge_quarter, bulge_quarter, bulge_quarter)
            hole = Circle2D(center=(x, y), radius=inner_r)
            return Polygon2D(vertices=vertices, bulges=bulges, holes=(hole,))

        elif sym_info.type == "donut_s":
            # Square donut
            outer_s = convert_to_microns(
                sym_info.params["outer_side"], units, is_symbol_dim=True
            )
            inner_s = convert_to_microns(
                sym_info.params["inner_side"], units, is_symbol_dim=True
            )
            half_outer = outer_s / 2
            vertices = [
                (x - half_outer, y - half_outer),
                (x + half_outer, y - half_outer),
                (x + half_outer, y + half_outer),
                (x - half_outer, y + half_outer),
            ]
            hole = Rectangle2D(center=(x, y), size=(inner_s, inner_s))
            return Polygon2D(vertices=vertices, holes=(hole,))

        elif sym_info.type == "donut_s_rounded":
            # Rounded square donut - both outer and inner squares have rounded corners
            outer_s = convert_to_microns(
                sym_info.params["outer_side"], units, is_symbol_dim=True
            )
            inner_s = convert_to_microns(
                sym_info.params["inner_side"], units, is_symbol_dim=True
            )
            corner_radius = convert_to_microns(
                sym_info.params["corner_radius"], units, is_symbol_dim=True
            )
            corners = sym_info.params.get("corners", "1234")
            # Outer rounded rectangle
            outer_verts, outer_bulges = self._create_rounded_rect_vertices(
                outer_s, outer_s, corner_radius, corners
            )
            outer_verts = [(vx + x, vy + y) for vx, vy in outer_verts]
            # Inner rounded rectangle (hole) - scale corner radius proportionally
            inner_corner_radius = min(corner_radius, inner_s / 2)
            inner_verts, inner_bulges = self._create_rounded_rect_vertices(
                inner_s, inner_s, inner_corner_radius, corners
            )
            inner_hole = Polygon2D(
                vertices=[(vx + x, vy + y) for vx, vy in inner_verts],
                bulges=inner_bulges,
            )
            return Polygon2D(vertices=outer_verts, bulges=outer_bulges, holes=(inner_hole,))

        elif sym_info.type == "donut_sr":
            # Square with round hole
            outer_s = convert_to_microns(
                sym_info.params["outer_side"], units, is_symbol_dim=True
            )
            inner_d = convert_to_microns(
                sym_info.params["inner_diameter"], units, is_symbol_dim=True
            )
            half_outer = outer_s / 2
            inner_r = inner_d / 2
            vertices = [
                (x - half_outer, y - half_outer),
                (x + half_outer, y - half_outer),
                (x + half_outer, y + half_outer),
                (x - half_outer, y + half_outer),
            ]
            hole = Circle2D(center=(x, y), radius=inner_r)
            return Polygon2D(vertices=vertices, holes=(hole,))

        elif sym_info.type == "rounded_rect":
            # Rounded rectangle
            width = convert_to_microns(sym_info.params["width"], units, is_symbol_dim=True)
            height = convert_to_microns(sym_info.params["height"], units, is_symbol_dim=True)
            corner_radius = convert_to_microns(
                sym_info.params["corner_radius"], units, is_symbol_dim=True
            )
            corners = sym_info.params.get("corners", "1234")
            vertices, bulges = self._create_rounded_rect_vertices(
                width, height, corner_radius, corners
            )
            return Polygon2D(
                vertices=[(vx + x, vy + y) for vx, vy in vertices],
                bulges=bulges,
            )

        elif sym_info.type == "chamfered_rect":
            # Chamfered rectangle
            width = convert_to_microns(sym_info.params["width"], units, is_symbol_dim=True)
            height = convert_to_microns(sym_info.params["height"], units, is_symbol_dim=True)
            corner_radius = convert_to_microns(
                sym_info.params["corner_radius"], units, is_symbol_dim=True
            )
            corners = sym_info.params.get("corners", "1234")
            vertices = self._create_chamfered_rect_vertices(
                width, height, corner_radius, corners
            )
            return Polygon2D(vertices=[(vx + x, vy + y) for vx, vy in vertices])

        return None

    def _find_merge_extent(
        self,
        features: list,
        start_idx: int,
        symbols: dict,
        units: str,
        layer_name: str,
        eda_data: EDAData,
        attr_names: dict,
        attr_texts: dict,
    ) -> int:
        """Find the last index of consecutive mergeable Line/Arc records.

        Parameters
        ----------
        features : list
            List of feature records.
        start_idx : int
            Starting index.
        symbols : dict
            Symbol index to name mapping.
        units : str
            Units string.
        layer_name : str
            Layer name for net lookup.
        eda_data : EDAData
            EDA data for net lookup.
        attr_names : dict
            Attribute index to name mapping from features file.
        attr_texts : dict
            Attribute text index to text mapping from features file.

        Returns
        -------
        int
            Last index of mergeable records (>= start_idx).
            Returns start_idx if no merging possible.
        """
        # Find .net_name attribute index
        net_name_attr_idx: Optional[int] = None
        for attr_idx, attr_name in attr_names.items():
            if attr_name == ".net_name":
                net_name_attr_idx = attr_idx
                break

        def get_net_from_feature(feature) -> Optional[str]:
            """Get net name from feature's own attributes."""
            if net_name_attr_idx is None:
                return None
            if not hasattr(feature, "attributes") or not feature.attributes:
                return None
            if net_name_attr_idx not in feature.attributes:
                return None
            try:
                text_idx = int(feature.attributes[net_name_attr_idx])
                net = attr_texts.get(text_idx)
                return None if net == "$NONE$" else net
            except (ValueError, TypeError):
                return None

        def get_net(idx: int, feature=None) -> Optional[str]:
            # First try EDA data
            net = eda_data.net_assignments.get((layer_name, idx))
            if net is not None:
                return None if net == "$NONE$" else net
            # Fall back to feature attributes
            if feature is not None:
                return get_net_from_feature(feature)
            return None

        def get_width_and_cap(
            record: Union[LineRecord, ArcRecord]
        ) -> tuple[Optional[float], str]:
            """Get width (in microns) and cap style for a record."""
            sym_name = symbols.get(record.symbol_num, "")
            if not sym_name:
                return None, ""
            sym_info = parse_symbol(sym_name)
            if sym_info.type == "unknown" or sym_info.width is None:
                return None, ""
            width_um = convert_to_microns(sym_info.width, units, is_symbol_dim=True)
            cap = "round" if sym_info.type == "round" else "square"
            return width_um, cap

        def get_end_point(record: Union[LineRecord, ArcRecord]) -> tuple[float, float]:
            """Get end point in microns."""
            return (
                convert_to_microns(record.xe, units),
                convert_to_microns(record.ye, units),
            )

        def get_start_point(record: Union[LineRecord, ArcRecord]) -> tuple[float, float]:
            """Get start point in microns."""
            return (
                convert_to_microns(record.xs, units),
                convert_to_microns(record.ys, units),
            )

        merge_end = start_idx
        prev = features[start_idx]
        prev_width, prev_cap = get_width_and_cap(prev)
        prev_net = get_net(start_idx, prev)

        if prev_width is None:
            return start_idx

        for j in range(start_idx + 1, len(features)):
            curr = features[j]

            # Must be Line or Arc
            if not isinstance(curr, (LineRecord, ArcRecord)):
                break

            # Check width and cap
            curr_width, curr_cap = get_width_and_cap(curr)
            if curr_width is None:
                break
            if not _widths_match(prev_width, curr_width):
                break
            if curr_cap != prev_cap:
                break

            # Check net
            curr_net = get_net(j, curr)
            if curr_net != prev_net:
                break

            # Check endpoint connectivity
            prev_end = get_end_point(prev)
            curr_start = get_start_point(curr)
            if not _endpoints_match(prev_end[0], prev_end[1], curr_start[0], curr_start[1]):
                break

            # Can merge
            merge_end = j
            prev = curr

        return merge_end

    def _create_merged_trace(
        self,
        records: list[Union[LineRecord, ArcRecord]],
        symbols: dict,
        units: str,
    ) -> Optional[Path2D]:
        """Create a multi-segment Path2D from merged Line/Arc records.

        Parameters
        ----------
        records : list
            List of LineRecord and/or ArcRecord to merge.
        symbols : dict
            Symbol index to name mapping.
        units : str
            Units string.

        Returns
        -------
        Optional[Path2D]
            Merged path, or None if conversion failed.
        """
        if not records:
            return None

        first = records[0]
        sym_name = symbols.get(first.symbol_num, "")
        sym_info = parse_symbol(sym_name)

        if sym_info.type == "unknown" or sym_info.width is None:
            return None

        width_um = convert_to_microns(sym_info.width, units, is_symbol_dim=True)
        if width_um <= 0:
            return None

        end_cap = "round" if sym_info.type == "round" else "square"

        # Build vertices list: start with first record's start point
        vertices: list[tuple[float, float]] = [
            (
                convert_to_microns(first.xs, units),
                convert_to_microns(first.ys, units),
            )
        ]

        # Add each record's end point (and arc info if applicable)
        segment_types: list[str] = []  # "line" or "arc"
        arc_centers: list[Optional[tuple[float, float]]] = []
        arc_clockwise: list[bool] = []

        for record in records:
            xe = convert_to_microns(record.xe, units)
            ye = convert_to_microns(record.ye, units)
            vertices.append((xe, ye))

            if isinstance(record, ArcRecord):
                segment_types.append("arc")
                xc = convert_to_microns(record.xc, units)
                yc = convert_to_microns(record.yc, units)
                arc_centers.append((xc, yc))
                arc_clockwise.append(record.clockwise)
            else:
                segment_types.append("line")
                arc_centers.append(None)
                arc_clockwise.append(False)

        # If all segments are lines, use simple Path2D
        if all(st == "line" for st in segment_types):
            return Path2D(
                vertices=tuple(vertices),
                width=width_um,
                end_cap=end_cap,
            )

        # For mixed line/arc, we need to use Path2D with proper arc handling
        # Build the path segment by segment
        # Note: Path2D currently supports from_line and from_arc for single segments
        # For multi-segment with arcs, we need to build manually

        # For now, create a polyline approximation for arcs
        # TODO: Enhance Path2D to support mixed line/arc segments natively
        final_vertices: list[tuple[float, float]] = [vertices[0]]

        for idx, seg_type in enumerate(segment_types):
            if seg_type == "line":
                final_vertices.append(vertices[idx + 1])
            else:
                # Arc segment - approximate with line for now
                # A proper implementation would store arc info in Path2D
                final_vertices.append(vertices[idx + 1])
                self._warnings.append(
                    f"Arc segment in merged trace approximated as line"
                )

        return Path2D(
            vertices=tuple(final_vertices),
            width=width_um,
            end_cap=end_cap,
        )

    def _convert_line(
        self, record: LineRecord, symbols: dict, units: str
    ) -> Optional[Path2D]:
        """Convert L record to Path2D.

        Parameters
        ----------
        record : LineRecord
            Parsed line record.
        symbols : dict
            Symbol index to name mapping.
        units : str
            Units string ("MM" or "INCH").

        Returns
        -------
        Optional[Path2D]
            Converted path, or None if conversion failed.
        """
        sym_name = symbols.get(record.symbol_num, "")
        if not sym_name:
            self._warnings.append(f"Line record references missing symbol {record.symbol_num}")
            return None

        sym_info = parse_symbol(sym_name)

        if sym_info.type == "unknown":
            # Warning already issued by parse_symbol
            return None

        width = sym_info.width
        if width is None:
            self._warnings.append(f"Symbol '{sym_name}' not suitable for line stroke")
            return None

        # Convert units to microns
        width_um = convert_to_microns(width, units, is_symbol_dim=True)
        
        # Validate width is positive
        if width_um <= 0:
            self._warnings.append(f"Line with zero/negative width ({width_um}), skipping")
            return None
        
        xs = convert_to_microns(record.xs, units)
        ys = convert_to_microns(record.ys, units)
        xe = convert_to_microns(record.xe, units)
        ye = convert_to_microns(record.ye, units)

        # Skip zero-length lines
        length_sq = (xe - xs) ** 2 + (ye - ys) ** 2
        if length_sq < 1e-12:
            return None

        # Determine end cap style
        end_cap = "round" if sym_info.type == "round" else "square"

        return Path2D.from_line(
            start=(xs, ys),
            end=(xe, ye),
            width=width_um,
            end_cap=end_cap,
        )

    def _convert_arc(
        self, record: ArcRecord, symbols: dict, units: str
    ) -> Optional[Path2D]:
        """Convert A record to Path2D with arc segment.

        Parameters
        ----------
        record : ArcRecord
            Parsed arc record.
        symbols : dict
            Symbol index to name mapping.
        units : str
            Units string ("MM" or "INCH").

        Returns
        -------
        Optional[Path2D]
            Converted path, or None if conversion failed.
        """
        sym_name = symbols.get(record.symbol_num, "")
        if not sym_name:
            self._warnings.append(f"Arc record references missing symbol {record.symbol_num}")
            return None

        sym_info = parse_symbol(sym_name)

        if sym_info.type == "unknown":
            return None

        if sym_info.type != "round":
            self._warnings.append(f"Arc requires round symbol, got {sym_info.type}")
            return None

        width = sym_info.width
        if width is None:
            return None

        width_um = convert_to_microns(width, units, is_symbol_dim=True)
        
        # Validate width is positive
        if width_um <= 0:
            self._warnings.append(f"Arc with zero/negative width ({width_um}), skipping")
            return None
        
        xs = convert_to_microns(record.xs, units)
        ys = convert_to_microns(record.ys, units)
        xe = convert_to_microns(record.xe, units)
        ye = convert_to_microns(record.ye, units)
        xc = convert_to_microns(record.xc, units)
        yc = convert_to_microns(record.yc, units)

        return Path2D.from_arc(
            start=(xs, ys),
            end=(xe, ye),
            center=(xc, yc),
            width=width_um,
            clockwise=record.clockwise,
            end_cap="round",
        )

    def _convert_pad(
        self, record: PadRecord, symbols: dict, units: str
    ) -> Optional[Geometry2D]:
        """Convert P record to Geometry2D, applying rotation/mirror via Transformed2D.

        Parameters
        ----------
        record : PadRecord
            Parsed pad record.
        symbols : dict
            Symbol index to name mapping.
        units : str
            Units string ("MM" or "INCH").

        Returns
        -------
        Optional[Geometry2D]
            Converted geometry, or None if conversion failed.
        """
        sym_name = symbols.get(record.symbol_num, "")
        if not sym_name:
            self._warnings.append(f"Pad record references missing symbol {record.symbol_num}")
            return None

        sym_info = parse_symbol(sym_name)

        if sym_info.type == "unknown":
            return None

        x = convert_to_microns(record.x, units)
        y = convert_to_microns(record.y, units)

        # Parse orientation: rotation (degrees) and mirror flag
        rotation_deg, mirror_x = self._parse_orient_def(record.orient_def)
        rotation_deg += sym_info.rotation
        rotation_deg = rotation_deg % 360  # Normalize to [0, 360)

        # Build base shape at origin first, then apply transform
        base_shape: Optional[Geometry2D] = None

        if sym_info.type == "round":
            diameter = convert_to_microns(sym_info.params["diameter"], units, is_symbol_dim=True)
            # Circles are rotationally symmetric, no transform needed
            return Circle2D(center=(x, y), radius=diameter / 2)

        elif sym_info.type == "square":
            side = convert_to_microns(sym_info.params["side"], units, is_symbol_dim=True)
            # Squares are symmetric under 90° rotation, only mirror matters
            if mirror_x and rotation_deg % 90 != 0:
                # Mirror + non-90° rotation needs transform
                base_shape = Rectangle2D(center=(0, 0), size=(side, side))
            elif rotation_deg != 0 and rotation_deg % 90 != 0:
                # Non-90° rotation needs transform
                base_shape = Rectangle2D(center=(0, 0), size=(side, side))
            else:
                return Rectangle2D(center=(x, y), size=(side, side))

        elif sym_info.type == "rect":
            width = convert_to_microns(sym_info.params["width"], units, is_symbol_dim=True)
            height = convert_to_microns(sym_info.params["height"], units, is_symbol_dim=True)
            
            # For axis-aligned rectangles, we can optimize 90° rotations
            if not mirror_x and rotation_deg in (0, 180):
                return Rectangle2D(center=(x, y), size=(width, height))
            elif not mirror_x and rotation_deg in (90, 270):
                return Rectangle2D(center=(x, y), size=(height, width))
            else:
                # Non-90° rotation or mirror: use Transformed2D
                base_shape = Rectangle2D(center=(0, 0), size=(width, height))

        elif sym_info.type == "oval":
            # Oval → Stadium shape (rectangle with semicircle endcaps)
            width = convert_to_microns(sym_info.params["width"], units, is_symbol_dim=True)
            height = convert_to_microns(sym_info.params["height"], units, is_symbol_dim=True)
            base_shape = self._create_oval_polygon(width, height)
            # For axis-aligned cases without mirror, just translate
            if not mirror_x and rotation_deg in (0, 180):
                return self._translate_polygon(base_shape, x, y)
            elif not mirror_x and rotation_deg in (90, 270):
                # Swap dimensions for 90/270 rotation
                base_shape = self._create_oval_polygon(height, width)
                return self._translate_polygon(base_shape, x, y)

        elif sym_info.type == "diamond":
            # Diamond → 4-vertex rhombus polygon
            width = convert_to_microns(sym_info.params["width"], units, is_symbol_dim=True)
            height = convert_to_microns(sym_info.params["height"], units, is_symbol_dim=True)
            hw, hh = width / 2, height / 2
            # Vertices at cardinal points (right, top, left, bottom) - CCW order
            vertices = [(hw, 0), (0, hh), (-hw, 0), (0, -hh)]
            base_shape = Polygon2D(vertices=vertices)
            # For axis-aligned cases without mirror, just translate
            if not mirror_x and rotation_deg in (0, 180):
                return Polygon2D(vertices=[(vx + x, vy + y) for vx, vy in vertices])
            elif not mirror_x and rotation_deg in (90, 270):
                # Swap dimensions for 90/270 rotation
                vertices_rotated = [(hh, 0), (0, hw), (-hh, 0), (0, -hw)]
                return Polygon2D(vertices=[(vx + x, vy + y) for vx, vy in vertices_rotated])

        elif sym_info.type == "octagon":
            # Octagon → 8-vertex polygon (chamfered rectangle)
            width = convert_to_microns(sym_info.params["width"], units, is_symbol_dim=True)
            height = convert_to_microns(sym_info.params["height"], units, is_symbol_dim=True)
            corner = convert_to_microns(sym_info.params["corner"], units, is_symbol_dim=True)
            vertices = self._create_octagon_vertices(width, height, corner)
            base_shape = Polygon2D(vertices=vertices)
            # For axis-aligned cases without mirror, just translate
            if not mirror_x and rotation_deg in (0, 180):
                return Polygon2D(vertices=[(vx + x, vy + y) for vx, vy in vertices])
            elif not mirror_x and rotation_deg in (90, 270):
                # Swap dimensions for 90/270 rotation
                vertices_rotated = self._create_octagon_vertices(height, width, corner)
                return Polygon2D(vertices=[(vx + x, vy + y) for vx, vy in vertices_rotated])

        elif sym_info.type == "donut_r":
            # Round donut (annular ring) → Polygon2D with circular exterior and hole
            outer_d = convert_to_microns(
                sym_info.params["outer_diameter"], units, is_symbol_dim=True
            )
            inner_d = convert_to_microns(
                sym_info.params["inner_diameter"], units, is_symbol_dim=True
            )
            outer_r = outer_d / 2
            inner_r = inner_d / 2
            # Approximate outer circle with 32 vertices (centered at origin)
            n = 32
            vertices = [
                (outer_r * math.cos(2 * math.pi * i / n),
                 outer_r * math.sin(2 * math.pi * i / n))
                for i in range(n)
            ]
            hole = Circle2D(center=(0, 0), radius=inner_r)
            base_shape = Polygon2D(vertices=vertices, holes=(hole,))
            # Round donuts are rotationally symmetric, only need translation (and mirror)
            if not mirror_x:
                # Just translate
                return Polygon2D(
                    vertices=[(vx + x, vy + y) for vx, vy in vertices],
                    holes=(Circle2D(center=(x, y), radius=inner_r),)
                )

        elif sym_info.type == "donut_s":
            # Square donut → Rectangle exterior with square hole
            outer_s = convert_to_microns(
                sym_info.params["outer_side"], units, is_symbol_dim=True
            )
            inner_s = convert_to_microns(
                sym_info.params["inner_side"], units, is_symbol_dim=True
            )
            half_outer = outer_s / 2
            # Centered at origin
            vertices = [
                (-half_outer, -half_outer),
                (half_outer, -half_outer),
                (half_outer, half_outer),
                (-half_outer, half_outer),
            ]
            hole = Rectangle2D(center=(0, 0), size=(inner_s, inner_s))
            base_shape = Polygon2D(vertices=vertices, holes=(hole,))
            # For axis-aligned cases, just translate
            if not mirror_x and rotation_deg in (0, 90, 180, 270):
                return Polygon2D(
                    vertices=[(vx + x, vy + y) for vx, vy in vertices],
                    holes=(Rectangle2D(center=(x, y), size=(inner_s, inner_s)),)
                )

        elif sym_info.type == "donut_s_rounded":
            # Rounded square donut - both outer and inner squares have rounded corners
            outer_s = convert_to_microns(
                sym_info.params["outer_side"], units, is_symbol_dim=True
            )
            inner_s = convert_to_microns(
                sym_info.params["inner_side"], units, is_symbol_dim=True
            )
            corner_radius = convert_to_microns(
                sym_info.params["corner_radius"], units, is_symbol_dim=True
            )
            corners = sym_info.params.get("corners", "1234")
            # Outer rounded rectangle (centered at origin)
            outer_verts, outer_bulges = self._create_rounded_rect_vertices(
                outer_s, outer_s, corner_radius, corners
            )
            # Inner rounded rectangle (hole) - scale corner radius proportionally
            inner_corner_radius = min(corner_radius, inner_s / 2)
            inner_verts, inner_bulges = self._create_rounded_rect_vertices(
                inner_s, inner_s, inner_corner_radius, corners
            )
            inner_hole = Polygon2D(vertices=inner_verts, bulges=inner_bulges)
            base_shape = Polygon2D(
                vertices=outer_verts, bulges=outer_bulges, holes=(inner_hole,)
            )
            # For axis-aligned cases without mirror, just translate
            if not mirror_x and rotation_deg in (0, 90, 180, 270):
                return Polygon2D(
                    vertices=[(vx + x, vy + y) for vx, vy in outer_verts],
                    bulges=outer_bulges,
                    holes=(
                        Polygon2D(
                            vertices=[(vx + x, vy + y) for vx, vy in inner_verts],
                            bulges=inner_bulges,
                        ),
                    ),
                )

        elif sym_info.type == "donut_sr":
            # Square with round hole
            outer_s = convert_to_microns(
                sym_info.params["outer_side"], units, is_symbol_dim=True
            )
            inner_d = convert_to_microns(
                sym_info.params["inner_diameter"], units, is_symbol_dim=True
            )
            half_outer = outer_s / 2
            inner_r = inner_d / 2
            # Centered at origin
            vertices = [
                (-half_outer, -half_outer),
                (half_outer, -half_outer),
                (half_outer, half_outer),
                (-half_outer, half_outer),
            ]
            hole = Circle2D(center=(0, 0), radius=inner_r)
            base_shape = Polygon2D(vertices=vertices, holes=(hole,))
            # For axis-aligned cases, just translate
            if not mirror_x and rotation_deg in (0, 90, 180, 270):
                return Polygon2D(
                    vertices=[(vx + x, vy + y) for vx, vy in vertices],
                    holes=(Circle2D(center=(x, y), radius=inner_r),)
                )

        elif sym_info.type == "rounded_rect":
            # Rounded rectangle → Polygon2D with corner bulges
            width = convert_to_microns(sym_info.params["width"], units, is_symbol_dim=True)
            height = convert_to_microns(sym_info.params["height"], units, is_symbol_dim=True)
            corner_radius = convert_to_microns(
                sym_info.params["corner_radius"], units, is_symbol_dim=True
            )
            corners = sym_info.params.get("corners", "1234")
            vertices, bulges = self._create_rounded_rect_vertices(
                width, height, corner_radius, corners
            )
            base_shape = Polygon2D(vertices=vertices, bulges=bulges)
            # For axis-aligned cases without mirror, just translate
            if not mirror_x and rotation_deg in (0, 180):
                return Polygon2D(
                    vertices=[(vx + x, vy + y) for vx, vy in vertices],
                    bulges=bulges,
                )
            elif not mirror_x and rotation_deg in (90, 270):
                # Swap dimensions for 90/270 rotation
                vertices_r, bulges_r = self._create_rounded_rect_vertices(
                    height, width, corner_radius, corners
                )
                return Polygon2D(
                    vertices=[(vx + x, vy + y) for vx, vy in vertices_r],
                    bulges=bulges_r,
                )

        elif sym_info.type == "chamfered_rect":
            # Chamfered rectangle → 8-vertex polygon
            width = convert_to_microns(sym_info.params["width"], units, is_symbol_dim=True)
            height = convert_to_microns(sym_info.params["height"], units, is_symbol_dim=True)
            corner_radius = convert_to_microns(
                sym_info.params["corner_radius"], units, is_symbol_dim=True
            )
            corners = sym_info.params.get("corners", "1234")
            vertices = self._create_chamfered_rect_vertices(
                width, height, corner_radius, corners
            )
            base_shape = Polygon2D(vertices=vertices)
            # For axis-aligned cases without mirror, just translate
            if not mirror_x and rotation_deg in (0, 180):
                return Polygon2D(vertices=[(vx + x, vy + y) for vx, vy in vertices])
            elif not mirror_x and rotation_deg in (90, 270):
                # Swap dimensions for 90/270 rotation
                vertices_r = self._create_chamfered_rect_vertices(
                    height, width, corner_radius, corners
                )
                return Polygon2D(vertices=[(vx + x, vy + y) for vx, vy in vertices_r])

        # If we have a base shape that needs transformation
        if base_shape is not None:
            return self._apply_pad_transform(base_shape, x, y, rotation_deg, mirror_x)

        return None

    def _apply_pad_transform(
        self,
        base_shape: Geometry2D,
        x: float,
        y: float,
        rotation_deg: float,
        mirror_x: bool,
    ) -> Geometry2D:
        """Apply rotation, mirror, and translation to a pad shape.

        Parameters
        ----------
        base_shape : Geometry2D
            Shape centered at origin.
        x, y : float
            Target position (microns).
        rotation_deg : float
            Rotation angle in degrees (counter-clockwise).
        mirror_x : bool
            Whether to mirror across X-axis (flip Y).

        Returns
        -------
        Geometry2D
            Transformed shape at target position.
        """
        import numpy as np

        # Build composite transform: mirror → rotate → translate
        # Transform applied right-to-left, so we build: T @ R @ M
        transform = np.eye(3)

        # 1. Mirror (if needed) - reflect across X-axis (negate Y)
        if mirror_x:
            transform = transform @ np.array(Transformed2D.reflection("x"))

        # 2. Rotate (if needed)
        if rotation_deg != 0:
            angle_rad = math.radians(rotation_deg)
            transform = transform @ np.array(Transformed2D.rotation(angle_rad))

        # 3. Translate to final position
        transform = np.array(Transformed2D.translation(x, y)) @ transform

        # Convert to list for pydantic validation
        return Transformed2D(geometry=base_shape, transform=transform.tolist())

    def _create_oval_polygon(
        self, width: float, height: float
    ) -> Polygon2D:
        """Create an oval (stadium) shape as Polygon2D with bulges.

        The oval is a rectangle with semicircular endcaps. Uses bulge=1.0
        for semicircles (180° arcs).

        Parameters
        ----------
        width : float
            Total width of the oval.
        height : float
            Total height of the oval.

        Returns
        -------
        Polygon2D
            Stadium shape centered at origin.
        """
        hw, hh = width / 2, height / 2

        if abs(width - height) < 1e-9:
            # Circle case: use 4 quarter-circle arcs
            # Vertices at cardinal points, each edge is a quarter circle (bulge ≈ 0.414)
            r = hw
            # tan(90°/4) = tan(22.5°) ≈ 0.4142
            bulge_quarter = math.tan(math.pi / 8)
            vertices = [(r, 0), (0, r), (-r, 0), (0, -r)]
            bulges = (bulge_quarter, bulge_quarter, bulge_quarter, bulge_quarter)
            return Polygon2D(vertices=vertices, bulges=bulges)

        if width > height:
            # Horizontal stadium: semicircles on left and right
            # The semicircle radius is hh (half height)
            hw2 = hw - hh  # Half-width of the straight section
            # Vertices: bottom-left, bottom-right, top-right, top-left (CCW)
            # Edges: bottom (straight), right semicircle, top (straight), left semicircle
            vertices = [(-hw2, -hh), (hw2, -hh), (hw2, hh), (-hw2, hh)]
            # bulge = 1.0 for semicircle (180° arc), 0 for straight
            bulges = (0.0, 1.0, 0.0, 1.0)
        else:
            # Vertical stadium: semicircles on top and bottom
            hh2 = hh - hw  # Half-height of the straight section
            # Vertices: right-bottom, right-top, left-top, left-bottom (CCW)
            vertices = [(hw, -hh2), (hw, hh2), (-hw, hh2), (-hw, -hh2)]
            # bulge = 1.0 for semicircle
            bulges = (1.0, 0.0, 1.0, 0.0)

        return Polygon2D(vertices=vertices, bulges=bulges)

    def _translate_polygon(
        self, polygon: Polygon2D, dx: float, dy: float
    ) -> Polygon2D:
        """Translate a Polygon2D by (dx, dy).

        Parameters
        ----------
        polygon : Polygon2D
            Polygon to translate.
        dx, dy : float
            Translation offsets.

        Returns
        -------
        Polygon2D
            Translated polygon.
        """
        new_vertices = [
            (float(v[0]) + dx, float(v[1]) + dy) for v in polygon.vertices
        ]
        # Translate holes if present
        new_holes = []
        for hole in polygon.holes:
            if isinstance(hole, Circle2D):
                new_holes.append(Circle2D(
                    center=(hole.center[0] + dx, hole.center[1] + dy),
                    radius=hole.radius,
                ))
            elif isinstance(hole, Rectangle2D):
                new_holes.append(Rectangle2D(
                    center=(hole.center[0] + dx, hole.center[1] + dy),
                    size=hole.size,
                ))
            elif isinstance(hole, Polygon2D):
                new_holes.append(self._translate_polygon(hole, dx, dy))
            else:
                # For other types, use Transformed2D
                new_holes.append(Transformed2D(
                    geometry=hole,
                    transform=Transformed2D.translation(dx, dy),
                ))

        return Polygon2D(
            vertices=new_vertices,
            bulges=polygon.bulges,
            holes=tuple(new_holes) if new_holes else (),
        )

    def _create_octagon_vertices(
        self, width: float, height: float, corner: float
    ) -> list[tuple[float, float]]:
        """Create octagon vertices (chamfered rectangle).

        Parameters
        ----------
        width : float
            Total width.
        height : float
            Total height.
        corner : float
            Corner chamfer size.

        Returns
        -------
        list[tuple[float, float]]
            8 vertices in CCW order, centered at origin.
        """
        hw, hh = width / 2, height / 2
        c = min(corner, hw, hh)  # Clamp corner to valid range

        # 8 vertices starting from bottom-right, going CCW
        return [
            (hw, -hh + c),       # bottom-right, above corner
            (hw, hh - c),        # top-right, below corner
            (hw - c, hh),        # top-right, left of corner
            (-hw + c, hh),       # top-left, right of corner
            (-hw, hh - c),       # top-left, below corner
            (-hw, -hh + c),      # bottom-left, above corner
            (-hw + c, -hh),      # bottom-left, right of corner
            (hw - c, -hh),       # bottom-right, left of corner
        ]

    def _create_rounded_rect_vertices(
        self,
        width: float,
        height: float,
        corner_radius: float,
        corners: str = "1234",
    ) -> tuple[list[tuple[float, float]], tuple[float, ...]]:
        """Create rounded rectangle vertices with bulges for corner arcs.

        Parameters
        ----------
        width : float
            Total width.
        height : float
            Total height.
        corner_radius : float
            Corner radius.
        corners : str
            Which corners to round: "1"=top-right, "2"=top-left,
            "3"=bottom-left, "4"=bottom-right. Default "1234" = all.

        Returns
        -------
        tuple[list, tuple]
            (vertices, bulges) for Polygon2D.
        """
        hw, hh = width / 2, height / 2
        r = min(corner_radius, hw, hh)  # Clamp radius

        # Quarter-circle bulge: tan(90°/4) = tan(22.5°)
        bulge_quarter = math.tan(math.pi / 8)

        # Build vertices and bulges going CCW from bottom-right
        vertices = []
        bulges = []

        # Corner 4: bottom-right
        if "4" in corners and r > 0:
            vertices.extend([(hw, -hh + r), (hw - r, -hh)])
            bulges.extend([bulge_quarter, 0.0])
        else:
            vertices.append((hw, -hh))
            bulges.append(0.0)

        # Corner 3: bottom-left
        if "3" in corners and r > 0:
            vertices.extend([(-hw + r, -hh), (-hw, -hh + r)])
            bulges.extend([bulge_quarter, 0.0])
        else:
            vertices.append((-hw, -hh))
            bulges.append(0.0)

        # Corner 2: top-left
        if "2" in corners and r > 0:
            vertices.extend([(-hw, hh - r), (-hw + r, hh)])
            bulges.extend([bulge_quarter, 0.0])
        else:
            vertices.append((-hw, hh))
            bulges.append(0.0)

        # Corner 1: top-right
        if "1" in corners and r > 0:
            vertices.extend([(hw - r, hh), (hw, hh - r)])
            bulges.extend([bulge_quarter, 0.0])
        else:
            vertices.append((hw, hh))
            bulges.append(0.0)

        # Fix the last bulge to close the polygon properly
        # The last edge connects back to the first vertex
        if len(bulges) > 0:
            bulges[-1] = 0.0  # Last edge to first vertex is straight

        return vertices, tuple(bulges)

    def _create_chamfered_rect_vertices(
        self,
        width: float,
        height: float,
        corner_radius: float,
        corners: str = "1234",
    ) -> list[tuple[float, float]]:
        """Create chamfered rectangle vertices.

        Parameters
        ----------
        width : float
            Total width.
        height : float
            Total height.
        corner_radius : float
            Corner chamfer size.
        corners : str
            Which corners to chamfer: "1"=top-right, "2"=top-left,
            "3"=bottom-left, "4"=bottom-right. Default "1234" = all.

        Returns
        -------
        list[tuple[float, float]]
            Vertices in CCW order, centered at origin.
        """
        hw, hh = width / 2, height / 2
        c = min(corner_radius, hw, hh)  # Clamp chamfer

        vertices = []

        # Corner 4: bottom-right
        if "4" in corners and c > 0:
            vertices.extend([(hw, -hh + c), (hw - c, -hh)])
        else:
            vertices.append((hw, -hh))

        # Corner 3: bottom-left
        if "3" in corners and c > 0:
            vertices.extend([(-hw + c, -hh), (-hw, -hh + c)])
        else:
            vertices.append((-hw, -hh))

        # Corner 2: top-left
        if "2" in corners and c > 0:
            vertices.extend([(-hw, hh - c), (-hw + c, hh)])
        else:
            vertices.append((-hw, hh))

        # Corner 1: top-right
        if "1" in corners and c > 0:
            vertices.extend([(hw - c, hh), (hw, hh - c)])
        else:
            vertices.append((hw, hh))

        return vertices

    def _convert_surface(self, record: SurfaceRecord, units: str) -> Optional[Polygon2D]:
        """Convert S record to Polygon2D.

        Parameters
        ----------
        record : SurfaceRecord
            Parsed surface record.
        units : str
            Units string ("MM" or "INCH").

        Returns
        -------
        Optional[Polygon2D]
            Converted polygon, or None if conversion failed.
        """
        if not record.contours:
            return None

        # Find island (exterior) contour
        islands = [c for c in record.contours if not c.is_hole]
        holes = [c for c in record.contours if c.is_hole]

        if not islands:
            self._warnings.append("Surface has no island contour, skipping")
            return None

        if len(islands) > 1:
            self._warnings.append("Surface has multiple islands, using first only")

        # Convert island to vertices and bulges
        vertices, bulges = self._contour_to_vertices(islands[0], units)

        if not vertices or len(vertices) < 3:
            self._warnings.append("Surface has insufficient vertices, skipping")
            return None

        # Convert holes to Geometry2D objects
        hole_geoms: list[Geometry2D] = []
        for hole_contour in holes:
            hole_geom = self._convert_hole(hole_contour, units)
            if hole_geom:
                hole_geoms.append(hole_geom)

        # Determine if we need bulges (only if any are non-zero)
        has_bulges = any(abs(b) > 1e-12 for b in bulges)

        return Polygon2D(
            vertices=vertices,
            bulges=tuple(bulges) if has_bulges else None,
            holes=tuple(hole_geoms) if hole_geoms else (),
        )

    def _convert_hole(
        self, contour: PolygonContour, units: str
    ) -> Optional[Geometry2D]:
        """Convert a hole contour to geometry.

        Parameters
        ----------
        contour : PolygonContour
            Hole contour.
        units : str
            Units string.

        Returns
        -------
        Optional[Geometry2D]
            Circle2D for circular holes, Polygon2D otherwise.
        """
        # Check if it's a circular hole (single full-circle arc)
        if self._is_circular_hole(contour):
            center, radius = self._extract_circle_from_contour(contour, units)
            if radius > 0:
                return Circle2D(center=center, radius=radius)

        # Otherwise, convert to polygon
        vertices, bulges = self._contour_to_vertices(contour, units)
        if not vertices or len(vertices) < 3:
            return None

        has_bulges = any(abs(b) > 1e-12 for b in bulges)

        return Polygon2D(
            vertices=vertices,
            bulges=tuple(bulges) if has_bulges else None,
        )

    def _contour_to_vertices(
        self, contour: PolygonContour, units: str
    ) -> tuple[list[tuple[float, float]], list[float]]:
        """Convert contour segments to vertices and bulges.

        Parameters
        ----------
        contour : PolygonContour
            Contour to convert.
        units : str
            Units string.

        Returns
        -------
        tuple[list, list]
            (vertices, bulges) where vertices is list of (x, y) tuples
            and bulges is list of bulge values for each edge.
        """
        vertices: list[tuple[float, float]] = []
        bulges: list[float] = []

        prev_point: Optional[tuple[float, float]] = None

        for seg in contour.segments:
            if seg["type"] == "start":
                x = convert_to_microns(seg["x"], units)
                y = convert_to_microns(seg["y"], units)
                vertices.append((x, y))
                prev_point = (x, y)

            elif seg["type"] == "line":
                x = convert_to_microns(seg["x"], units)
                y = convert_to_microns(seg["y"], units)
                # Bulge for previous edge (straight line = 0)
                if vertices:
                    bulges.append(0.0)
                vertices.append((x, y))
                prev_point = (x, y)

            elif seg["type"] == "arc":
                xe = convert_to_microns(seg["xe"], units)
                ye = convert_to_microns(seg["ye"], units)
                xc = convert_to_microns(seg["xc"], units)
                yc = convert_to_microns(seg["yc"], units)
                cw = seg["cw"]

                if prev_point is None:
                    self._warnings.append("Arc segment without previous point")
                    continue

                # Check for full circle
                if is_full_circle(prev_point, (xe, ye), (xc, yc)):
                    # Split into two semicircles
                    midpoint, bulge1, bulge2 = split_full_circle(prev_point, (xc, yc), cw)
                    bulges.append(bulge1)
                    vertices.append(midpoint)
                    bulges.append(bulge2)
                    vertices.append((xe, ye))
                else:
                    # Calculate bulge
                    bulge = arc_to_bulge(prev_point, (xe, ye), (xc, yc), cw)
                    bulges.append(bulge)
                    vertices.append((xe, ye))

                prev_point = (xe, ye)

        # Final bulge for closing edge (last vertex to first vertex)
        if len(bulges) < len(vertices):
            bulges.append(0.0)  # Assume straight closing edge

        # Remove duplicate closing vertex if present
        if len(vertices) > 1:
            dist = (
                (vertices[0][0] - vertices[-1][0]) ** 2
                + (vertices[0][1] - vertices[-1][1]) ** 2
            ) ** 0.5
            if dist < 1e-9:
                vertices = vertices[:-1]
                bulges = bulges[:-1]

        return vertices, bulges

    def _is_circular_hole(self, contour: PolygonContour) -> bool:
        """Check if contour represents a circular hole (single full-circle arc).

        Parameters
        ----------
        contour : PolygonContour
            Contour to check.

        Returns
        -------
        bool
            True if contour is a single circular arc representing a full circle.
        """
        arc_count = sum(1 for s in contour.segments if s["type"] == "arc")
        line_count = sum(1 for s in contour.segments if s["type"] == "line")
        return arc_count == 1 and line_count == 0

    def _extract_circle_from_contour(
        self, contour: PolygonContour, units: str
    ) -> tuple[tuple[float, float], float]:
        """Extract center and radius from circular hole contour.

        Parameters
        ----------
        contour : PolygonContour
            Circular contour.
        units : str
            Units string.

        Returns
        -------
        tuple[tuple[float, float], float]
            ((center_x, center_y), radius)
        """
        for seg in contour.segments:
            if seg["type"] == "arc":
                xc = convert_to_microns(seg["xc"], units)
                yc = convert_to_microns(seg["yc"], units)
                xe = convert_to_microns(seg["xe"], units)
                ye = convert_to_microns(seg["ye"], units)
                radius = ((xe - xc) ** 2 + (ye - yc) ** 2) ** 0.5
                return (xc, yc), radius

        # Fallback - shouldn't happen if _is_circular_hole returned True
        return (0.0, 0.0), 0.0

    def _parse_orient_def(self, orient_def: str) -> tuple[float, bool]:
        """Parse orientation definition to rotation degrees and mirror flag.

        Parameters
        ----------
        orient_def : str
            Orientation definition string.

        Returns
        -------
        tuple[float, bool]
            (rotation_degrees, mirror_x) where rotation is 0-360 and mirror_x
            indicates X-axis reflection.

        Notes
        -----
        ODB++ orientation encoding:
        - 0-3: No mirror, rotation = value * 90°
        - 4-7: Mirror X, rotation = (value - 4) * 90°
        - "8 <angle>": Arbitrary rotation, no mirror
        - "9 <angle>": Arbitrary rotation, with mirror X
        """
        try:
            val = int(orient_def)
            if val < 4:
                return (float(val * 90 % 360), False)
            elif val < 8:
                return (float((val - 4) * 90 % 360), True)
        except ValueError:
            pass

        # Check for "8 <angle>" or "9 <angle>" format
        parts = orient_def.split()
        if len(parts) == 2 and parts[0] in ("8", "9"):
            try:
                angle = float(parts[1])
                mirror = parts[0] == "9"
                return (angle, mirror)
            except ValueError:
                pass

        return (0.0, False)

