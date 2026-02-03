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
    Circle2D,
    Geometry2D,
    Path2D,
    Polygon2D,
    Rectangle2D,
)
from tidy3d.components.geometry.layout import (
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

        structure = LayeredStructure(stackup=stackup)

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

            # Group geometries by net for efficient adding
            if geom_net_pairs:
                net_groups: dict[Optional[str], list[Geometry2D]] = {}
                for geom, net in geom_net_pairs:
                    net_groups.setdefault(net, []).append(geom)

                # Add each net group to structure
                for net, geoms in net_groups.items():
                    structure = structure.add(layer_def.name, geoms, net=net)

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
        """
        units = data.units
        features = data.features
        symbols = data.symbols

        def get_net(idx: int) -> Optional[str]:
            """Look up net for feature index."""
            net = eda_data.net_assignments.get((layer_name, idx))
            return None if net == "$NONE$" else net

        i = 0
        while i < len(features):
            feature = features[i]
            net_name = get_net(i)

            # Try trace grouping for Line/Arc records
            if group_traces and isinstance(feature, (LineRecord, ArcRecord)):
                # Find extent of mergeable consecutive records
                merge_end = self._find_merge_extent(
                    features, i, symbols, units, layer_name, eda_data
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

    def _find_merge_extent(
        self,
        features: list,
        start_idx: int,
        symbols: dict,
        units: str,
        layer_name: str,
        eda_data: EDAData,
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

        Returns
        -------
        int
            Last index of mergeable records (>= start_idx).
            Returns start_idx if no merging possible.
        """

        def get_net(idx: int) -> Optional[str]:
            net = eda_data.net_assignments.get((layer_name, idx))
            return None if net == "$NONE$" else net

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
        prev_net = get_net(start_idx)

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
            curr_net = get_net(j)
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
        """Convert P record to Circle2D or Rectangle2D.

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

        # Parse orientation (simplified - full implementation would handle mirror)
        rotation = self._parse_orient_def(record.orient_def)
        rotation += sym_info.rotation

        if sym_info.type == "round":
            diameter = convert_to_microns(sym_info.params["diameter"], units, is_symbol_dim=True)
            return Circle2D(center=(x, y), radius=diameter / 2)

        elif sym_info.type == "square":
            side = convert_to_microns(sym_info.params["side"], units, is_symbol_dim=True)
            if rotation != 0 and rotation % 90 != 0:
                self._warnings.append(f"Pad rotation ({rotation}°) not fully supported for square")
            return Rectangle2D(center=(x, y), size=(side, side))

        elif sym_info.type == "rect":
            width = convert_to_microns(sym_info.params["width"], units, is_symbol_dim=True)
            height = convert_to_microns(sym_info.params["height"], units, is_symbol_dim=True)
            if rotation != 0 and rotation % 90 != 0:
                self._warnings.append(f"Pad rotation ({rotation}°) not fully supported for rect")
            # Handle 90° rotations by swapping dimensions
            if rotation == 90 or rotation == 270:
                width, height = height, width
            return Rectangle2D(center=(x, y), size=(width, height))

        elif sym_info.type == "oval":
            # Oval → approximate as rectangle for now
            width = convert_to_microns(sym_info.params["width"], units, is_symbol_dim=True)
            height = convert_to_microns(sym_info.params["height"], units, is_symbol_dim=True)
            self._warnings.append(f"Oval symbol '{sym_name}' approximated as rectangle")
            return Rectangle2D(center=(x, y), size=(width, height))

        elif sym_info.type == "diamond":
            # Diamond → approximate as polygon would be better, use rectangle for now
            width = convert_to_microns(sym_info.params["width"], units, is_symbol_dim=True)
            height = convert_to_microns(sym_info.params["height"], units, is_symbol_dim=True)
            self._warnings.append(f"Diamond symbol '{sym_name}' approximated as rectangle")
            return Rectangle2D(center=(x, y), size=(width, height))

        elif sym_info.type == "octagon":
            # Octagon → approximate as rectangle
            width = convert_to_microns(sym_info.params["width"], units, is_symbol_dim=True)
            height = convert_to_microns(sym_info.params["height"], units, is_symbol_dim=True)
            self._warnings.append(f"Octagon symbol '{sym_name}' approximated as rectangle")
            return Rectangle2D(center=(x, y), size=(width, height))

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
            # Create circular polygon exterior (using bulges for full circle approximation)
            # For now, approximate as a polygon without inner hole
            # TODO: Use Polygon2D with Circle2D hole when properly supported
            # Approximate outer circle with 32 vertices
            n = 32
            vertices = [
                (x + outer_r * math.cos(2 * math.pi * i / n),
                 y + outer_r * math.sin(2 * math.pi * i / n))
                for i in range(n)
            ]
            hole = Circle2D(center=(x, y), radius=inner_r)
            return Polygon2D(vertices=vertices, holes=(hole,))

        elif sym_info.type == "donut_s":
            # Square donut → Rectangle exterior with square hole
            outer_s = convert_to_microns(
                sym_info.params["outer_side"], units, is_symbol_dim=True
            )
            inner_s = convert_to_microns(
                sym_info.params["inner_side"], units, is_symbol_dim=True
            )
            # Outer rectangle vertices (CCW)
            half_outer = outer_s / 2
            vertices = [
                (x - half_outer, y - half_outer),
                (x + half_outer, y - half_outer),
                (x + half_outer, y + half_outer),
                (x - half_outer, y + half_outer),
            ]
            # Inner rectangle as hole (using Rectangle2D)
            hole = Rectangle2D(center=(x, y), size=(inner_s, inner_s))
            return Polygon2D(vertices=vertices, holes=(hole,))

        return None

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

    def _parse_orient_def(self, orient_def: str) -> float:
        """Parse orientation definition to rotation degrees.

        Parameters
        ----------
        orient_def : str
            Orientation definition string.

        Returns
        -------
        float
            Rotation in degrees (0, 90, 180, 270, or arbitrary angle).

        Notes
        -----
        ODB++ orientation encoding:
        - 0-7: Legacy values (0=0°, 1=90°, 2=180°, 3=270°, 4-7=mirrored)
        - "8 <angle>": Arbitrary rotation, no mirror
        - "9 <angle>": Arbitrary rotation, with mirror
        """
        # Simplified - full implementation handles 0-7, 8+angle, 9+angle (mirror)
        try:
            val = int(orient_def)
            if val < 4:
                return float(val * 90 % 360)
            elif val < 8:
                # 4-7 are mirrored versions
                self._warnings.append("Mirrored orientation not fully supported")
                return float((val - 4) * 90 % 360)
        except ValueError:
            pass

        # Check for "8 <angle>" or "9 <angle>" format
        parts = orient_def.split()
        if len(parts) == 2 and parts[0] in ("8", "9"):
            try:
                angle = float(parts[1])
                if parts[0] == "9":
                    self._warnings.append("Mirrored orientation not fully supported")
                return angle
            except ValueError:
                pass

        return 0.0

