"""ODB++ stackup parsing and construction.

Build tidy3d Stackup from ODB++ layer information including:
- Layer ordering from matrix file
- Physical properties from layer attrlist files
- Board-level info from misc/attrlist

Units
-----
All dimensions are in microns (tidy3d default units).
Frequencies are in Hz.
Conductivity is in S/µm.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

from tidy3d.components.geometry.layout import LayerSpec, Stackup
from tidy3d.components.medium import Medium, PECMedium
from tidy3d.plugins.odb.parser import LayerDef, MatrixData

# Default frequency range for lossy materials (Hz)
DEFAULT_FREQ_RANGE: Tuple[float, float] = (0.1e9, 10e9)

# Default copper conductivity in S/µm (58e6 S/m = 58 S/µm)
DEFAULT_COPPER_CONDUCTIVITY: float = 58.0


# --- Data Classes ---


@dataclass
class LayerStackupInfo:
    """Physical stackup properties for a single layer.

    Attributes
    ----------
    name : str
        Layer name.
    layer_type : str
        Layer type from matrix (SIGNAL, DIELECTRIC, DRILL, etc.).
    thickness : Optional[float]
        Layer thickness in microns, if known.
    dielectric_constant : Optional[float]
        Relative permittivity (εr).
    loss_tangent : Optional[float]
        Loss tangent tan(δ).
    conductivity : Optional[float]
        Electrical conductivity in S/µm (for conductor layers).
        Derived from bulk_resistivity if available.
    copper_weight : Optional[float]
        Copper weight in oz/ft² (for signal layers).
    start_layer : str
        For DRILL layers, the starting layer name.
    end_layer : str
        For DRILL layers, the ending layer name.
    polarity : str
        POSITIVE or NEGATIVE.
    row : int
        Row number in stackup (1-based).
    """

    name: str
    layer_type: str
    thickness: Optional[float] = None
    dielectric_constant: Optional[float] = None
    loss_tangent: Optional[float] = None
    conductivity: Optional[float] = None
    copper_weight: Optional[float] = None
    start_layer: str = ""
    end_layer: str = ""
    polarity: str = "POSITIVE"
    row: int = 0


@dataclass
class BoardInfo:
    """Board-level information from misc/attrlist.

    Attributes
    ----------
    thickness : Optional[float]
        Total board thickness in mm.
    units : str
        Units (MM or INCH).
    primary_side : str
        Primary side (top or bottom).
    """

    thickness: Optional[float] = None
    units: str = "MM"
    primary_side: str = "top"


@dataclass
class StackupData:
    """Complete stackup data parsed from ODB++.

    Attributes
    ----------
    layers : list[LayerStackupInfo]
        Layer stackup info sorted by row.
    board_info : BoardInfo
        Board-level information.
    """

    layers: list[LayerStackupInfo] = field(default_factory=list)
    board_info: BoardInfo = field(default_factory=BoardInfo)


# --- Parsing Functions ---


def parse_attrlist(content: str) -> dict[str, str]:
    """Parse attrlist file content into key-value dict.

    Parameters
    ----------
    content : str
        Raw attrlist file content.

    Returns
    -------
    dict[str, str]
        Attribute name to value mapping.
    """
    result: dict[str, str] = {}
    for line in content.strip().split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            # Handle both "key=value" and ".key = value" formats
            key, _, value = line.partition("=")
            key = key.strip().lstrip(".")
            value = value.strip().strip('"')
            result[key] = value
    return result


def parse_board_attrlist(odb_path: Path) -> BoardInfo:
    """Parse board-level attributes from misc/attrlist.

    Parameters
    ----------
    odb_path : Path
        ODB++ root directory.

    Returns
    -------
    BoardInfo
        Parsed board information.
    """
    attrlist_path = odb_path / "misc" / "attrlist"
    info = BoardInfo()

    if not attrlist_path.exists():
        return info

    try:
        content = attrlist_path.read_text(encoding="utf-8", errors="ignore")
        attrs = parse_attrlist(content)

        # Board thickness (in mm typically)
        if "board_thickness" in attrs:
            try:
                info.thickness = float(attrs["board_thickness"])
            except ValueError:
                pass

        # Units
        if "UNITS" in attrs:
            info.units = attrs["UNITS"]

        # Primary side
        if "primary_side" in attrs:
            info.primary_side = attrs["primary_side"].lower()

    except Exception as e:
        warnings.warn(f"Failed to parse board attrlist: {e}", stacklevel=2)

    return info


def parse_layer_attrlist(odb_path: Path, step_name: str, layer_name: str) -> dict[str, str]:
    """Parse layer-specific attributes from layer attrlist.

    Parameters
    ----------
    odb_path : Path
        ODB++ root directory.
    step_name : str
        Step name.
    layer_name : str
        Layer name.

    Returns
    -------
    dict[str, str]
        Layer attributes.
    """
    attrlist_path = odb_path / "steps" / step_name / "layers" / layer_name / "attrlist"

    if not attrlist_path.exists():
        return {}

    try:
        content = attrlist_path.read_text(encoding="utf-8", errors="ignore")
        return parse_attrlist(content)
    except Exception:
        return {}


def parse_stackup_data(
    odb_path: Path,
    matrix: MatrixData,
    step_name: str,
) -> StackupData:
    """Parse complete stackup data from ODB++.

    Parameters
    ----------
    odb_path : Path
        ODB++ root directory.
    matrix : MatrixData
        Parsed matrix data.
    step_name : str
        Step name.

    Returns
    -------
    StackupData
        Complete stackup information.
    """
    data = StackupData()
    data.board_info = parse_board_attrlist(odb_path)

    for layer_def in matrix.layers:
        layer_attrs = parse_layer_attrlist(odb_path, step_name, layer_def.name)

        info = LayerStackupInfo(
            name=layer_def.name,
            layer_type=layer_def.type,
            polarity=layer_def.polarity,
            row=layer_def.row,
            start_layer=layer_def.start_name,
            end_layer=layer_def.end_name,
        )

        # Parse thickness
        # For dielectrics: .layer_dielectric is thickness in mm
        if "layer_dielectric" in layer_attrs:
            try:
                thickness_mm = float(layer_attrs["layer_dielectric"])
                if thickness_mm > 0:
                    info.thickness = thickness_mm * 1000  # Convert to µm
            except ValueError:
                pass

        # Copper weight (oz/ft²) - derive thickness: 1 oz ≈ 35 µm
        # This is the most reliable way to get conductor thickness
        if "copper_weight" in layer_attrs:
            try:
                # copper_weight can be in various formats:
                # - Direct oz value (0.5, 1, 2)
                # - Some tools use 16ths of oz (8=0.5oz, 16=1oz, 32=2oz)
                cw = float(layer_attrs["copper_weight"])
                if cw > 0:
                    info.copper_weight = cw
                    # Determine if it's direct oz or 16ths
                    # If value is small (< 10), assume direct oz
                    # If value is larger, assume 16ths of oz
                    if cw < 10:
                        # Direct oz: 0.5oz=17.5µm, 1oz=35µm, 2oz=70µm
                        info.thickness = cw * 35.0
                    else:
                        # 16ths of oz: 8=0.5oz, 16=1oz, 32=2oz
                        info.thickness = (cw / 16.0) * 35.0
            except ValueError:
                pass

        # Dielectric constant
        if "dielectric_constant" in layer_attrs:
            try:
                er = float(layer_attrs["dielectric_constant"])
                if er > 1:  # Ignore placeholder values like 1.0 for metals
                    info.dielectric_constant = er
            except ValueError:
                pass

        # Loss tangent
        if "loss_tangent" in layer_attrs:
            try:
                lt = float(layer_attrs["loss_tangent"])
                if lt > 0:
                    info.loss_tangent = lt
            except ValueError:
                pass

        # Bulk resistivity → conductivity
        # bulk_resistivity is typically in nΩ·cm (nano-ohm-cm)
        # For copper: ~17.2 nΩ·cm at 20°C
        # Convert to conductivity in S/µm:
        #
        # ρ in nΩ·cm = ρ × 1e-9 Ω·cm = ρ × 1e-9 × 1e-2 Ω·m = ρ × 1e-11 Ω·m
        # σ in S/m = 1 / (ρ × 1e-11) = 1e11 / ρ
        # σ in S/µm = σ_S_m × 1e-6 = (1e11 / ρ) × 1e-6 = 1e5 / ρ
        #
        # Wait, that gives 5800 S/µm for copper, but we expect 58 S/µm.
        # Let's recalculate:
        #   ρ = 17.2 nΩ·cm = 1.72e-8 Ω·m
        #   σ = 1/ρ = 5.81e7 S/m = 58.1 S/µm
        #
        # So the correct formula is: σ (S/µm) = 1e3 / ρ (nΩ·cm)
        if "bulk_resistivity" in layer_attrs:
            try:
                rho = float(layer_attrs["bulk_resistivity"])
                if rho > 0:
                    # σ (S/µm) = 1e3 / ρ (nΩ·cm)
                    # For copper ρ=17.2 → σ = 1000/17.2 = 58.1 S/µm ✓
                    info.conductivity = 1e3 / rho
            except ValueError:
                pass

        data.layers.append(info)

    # Sort by row
    data.layers.sort(key=lambda x: x.row)

    return data


# --- Stackup Builder ---


# Default thicknesses when not specified (in microns)
DEFAULT_THICKNESSES = {
    "SIGNAL": 35.0,  # 1 oz copper
    "POWER_GROUND": 35.0,
    "MIXED": 35.0,
    "DIELECTRIC": 200.0,  # Typical prepreg/core
    "SOLDER_MASK": 25.0,
    "SOLDER_PASTE": 0.0,  # Not physical
    "SILK_SCREEN": 0.0,  # Not physical
    "COMPONENT": 0.0,  # Not physical
    "DOCUMENT": 0.0,
    "DRILL": 0.0,  # Handled separately
    "ROUT": 0.0,
}

# Layer types that represent physical conductive layers
CONDUCTIVE_TYPES = {"SIGNAL", "POWER_GROUND", "MIXED"}

# Layer types that represent physical dielectric layers
DIELECTRIC_TYPES = {"DIELECTRIC", "SOLDER_MASK"}

# Layer types to skip (non-physical or handled separately)
SKIP_TYPES = {"COMPONENT", "DOCUMENT", "SILK_SCREEN", "SOLDER_PASTE", "ROUT"}


class StackupBuilder:
    """Build tidy3d Stackup from ODB++ data.

    Example
    -------
    >>> from tidy3d.plugins.odb.stackup_builder import StackupBuilder
    >>>
    >>> builder = StackupBuilder(odb_path, matrix, step_name)
    >>> stackup = builder.build()
    """

    def __init__(
        self,
        odb_path: Path,
        matrix: MatrixData,
        step_name: str,
    ):
        """Initialize stackup builder.

        Parameters
        ----------
        odb_path : Path
            ODB++ root directory.
        matrix : MatrixData
            Parsed matrix data.
        step_name : str
            Step name.
        """
        self.odb_path = odb_path
        self.matrix = matrix
        self.step_name = step_name
        self._stackup_data: Optional[StackupData] = None
        self._warnings: list[str] = []

    @property
    def stackup_data(self) -> StackupData:
        """Parsed stackup data (lazy loaded)."""
        if self._stackup_data is None:
            self._stackup_data = parse_stackup_data(
                self.odb_path, self.matrix, self.step_name
            )
        return self._stackup_data

    def build(
        self,
        layer_filter: Optional[set[str]] = None,
        default_conductor_thickness: float = 35.0,
        default_dielectric_thickness: float = 200.0,
        default_permittivity: float = 4.2,
        include_dielectrics: bool = False,
        frequency_range: Tuple[float, float] = DEFAULT_FREQ_RANGE,
        use_lossy_metal: bool = True,
        use_lossy_dielectric: bool = True,
    ) -> Stackup:
        """Build Stackup from ODB++ data.

        Parameters
        ----------
        layer_filter : set[str], optional
            Layer names to include. If None, includes all signal/power layers.
        default_conductor_thickness : float
            Default conductor thickness in µm when not specified.
        default_dielectric_thickness : float
            Default dielectric thickness in µm when not specified.
        default_permittivity : float
            Default dielectric constant for FR4.
        include_dielectrics : bool
            If True, include dielectric layers in stackup.
        frequency_range : tuple[float, float]
            Frequency range (f_min, f_max) in Hz for lossy material models.
            Default is (0.1e9, 10e9) Hz.
        use_lossy_metal : bool
            If True, use LossyMetalMedium for conductor layers.
            If False, use PECMedium. Default is True.
        use_lossy_dielectric : bool
            If True, use FastDispersionFitter.constant_loss_tangent_model()
            for dielectric layers with loss tangent. Default is True.

        Returns
        -------
        Stackup
            Constructed stackup with z_bounds and mediums.
        """
        self._warnings.clear()
        layer_specs: list[LayerSpec] = []

        # Filter to physical layers
        physical_layers = self._get_physical_layers(include_dielectrics)

        # Calculate z_bounds from bottom up
        z_bounds_map = self._calculate_z_bounds(
            physical_layers,
            default_conductor_thickness,
            default_dielectric_thickness,
        )

        for layer_info in physical_layers:
            if layer_filter and layer_info.name not in layer_filter:
                continue

            if layer_info.name not in z_bounds_map:
                continue

            z_bounds = z_bounds_map[layer_info.name]
            medium = self._get_medium(
                layer_info,
                default_permittivity,
                frequency_range,
                use_lossy_metal,
                use_lossy_dielectric,
            )

            layer_specs.append(
                LayerSpec(
                    name=layer_info.name,
                    z_bounds=z_bounds,
                    medium=medium,
                )
            )

        # Report warnings
        for warning in self._warnings:
            warnings.warn(warning, stacklevel=2)

        return Stackup(layers=tuple(layer_specs))

    def _get_physical_layers(
        self, include_dielectrics: bool
    ) -> list[LayerStackupInfo]:
        """Get physical layers in stackup order.

        Parameters
        ----------
        include_dielectrics : bool
            If True, include dielectric layers.

        Returns
        -------
        list[LayerStackupInfo]
            Physical layers sorted by row.
        """
        result = []
        for layer in self.stackup_data.layers:
            if layer.layer_type in SKIP_TYPES:
                continue
            if layer.layer_type == "DRILL":
                continue  # Handle separately
            if not include_dielectrics and layer.layer_type in DIELECTRIC_TYPES:
                continue
            result.append(layer)
        return result

    def _calculate_z_bounds(
        self,
        layers: list[LayerStackupInfo],
        default_conductor_thickness: float,
        default_dielectric_thickness: float,
    ) -> dict[str, tuple[float, float]]:
        """Calculate z_bounds for each layer.

        Parameters
        ----------
        layers : list[LayerStackupInfo]
            Physical layers in order.
        default_conductor_thickness : float
            Default conductor thickness in µm.
        default_dielectric_thickness : float
            Default dielectric thickness in µm.

        Returns
        -------
        dict[str, tuple[float, float]]
            Layer name to (z_min, z_max) mapping.
        """
        z_bounds: dict[str, tuple[float, float]] = {}

        # Start from z=0 at the bottom
        z_current = 0.0

        # Process layers from bottom to top (reversed row order)
        for layer in reversed(layers):
            # Determine thickness
            if layer.thickness is not None and layer.thickness > 0:
                thickness = layer.thickness
            elif layer.layer_type in CONDUCTIVE_TYPES:
                thickness = default_conductor_thickness
                self._warnings.append(
                    f"Layer '{layer.name}' has no thickness, using default {thickness} µm"
                )
            elif layer.layer_type in DIELECTRIC_TYPES:
                thickness = default_dielectric_thickness
                self._warnings.append(
                    f"Layer '{layer.name}' has no thickness, using default {thickness} µm"
                )
            else:
                thickness = DEFAULT_THICKNESSES.get(layer.layer_type, 0.0)

            if thickness > 0:
                z_min = z_current
                z_max = z_current + thickness
                z_bounds[layer.name] = (z_min, z_max)
                z_current = z_max

        return z_bounds

    def _get_medium(
        self,
        layer: LayerStackupInfo,
        default_permittivity: float,
        frequency_range: Tuple[float, float],
        use_lossy_metal: bool,
        use_lossy_dielectric: bool,
    ):
        """Get medium for layer.

        Parameters
        ----------
        layer : LayerStackupInfo
            Layer info.
        default_permittivity : float
            Default permittivity for dielectrics.
        frequency_range : tuple[float, float]
            Frequency range (f_min, f_max) in Hz for lossy materials.
        use_lossy_metal : bool
            If True, use LossyMetalMedium for conductors.
        use_lossy_dielectric : bool
            If True, use constant_loss_tangent_model for dielectrics.

        Returns
        -------
        Medium or PECMedium or LossyMetalMedium or None
            Appropriate medium for the layer.
        """
        if layer.layer_type in CONDUCTIVE_TYPES:
            if use_lossy_metal:
                # Use LossyMetalMedium for realistic conductor modeling
                try:
                    import tidy3d.rf as rf

                    # Use parsed conductivity or default copper value
                    conductivity = layer.conductivity or DEFAULT_COPPER_CONDUCTIVITY
                    return rf.LossyMetalMedium(
                        conductivity=conductivity,
                        frequency_range=frequency_range,
                    )
                except ImportError:
                    self._warnings.append(
                        f"tidy3d.rf not available, using PECMedium for '{layer.name}'"
                    )
                    return PECMedium()
            else:
                return PECMedium()

        if layer.layer_type in DIELECTRIC_TYPES:
            permittivity = layer.dielectric_constant or default_permittivity

            # Use lossy dielectric model if loss tangent is available
            if use_lossy_dielectric and layer.loss_tangent and layer.loss_tangent > 0:
                try:
                    from tidy3d.plugins.dispersion import FastDispersionFitter

                    return FastDispersionFitter.constant_loss_tangent_model(
                        eps_real=permittivity,
                        loss_tangent=layer.loss_tangent,
                        frequency_range=frequency_range,
                    )
                except ImportError:
                    self._warnings.append(
                        f"FastDispersionFitter not available, using lossless Medium for '{layer.name}'"
                    )
                    return Medium(permittivity=permittivity)
                except Exception as e:
                    self._warnings.append(
                        f"Failed to create lossy dielectric for '{layer.name}': {e}"
                    )
                    return Medium(permittivity=permittivity)

            return Medium(permittivity=permittivity)

        return None

    def get_drill_layers(self) -> list[LayerStackupInfo]:
        """Get drill layer definitions.

        Returns
        -------
        list[LayerStackupInfo]
            Drill layers with start/end layer references.
        """
        return [
            layer for layer in self.stackup_data.layers
            if layer.layer_type == "DRILL"
        ]

    def get_drill_z_span(
        self, drill_layer: LayerStackupInfo, z_bounds_map: dict[str, tuple[float, float]]
    ) -> Optional[tuple[float, float]]:
        """Get z-span for a drill layer.

        Parameters
        ----------
        drill_layer : LayerStackupInfo
            Drill layer info.
        z_bounds_map : dict[str, tuple[float, float]]
            Layer name to z_bounds mapping.

        Returns
        -------
        Optional[tuple[float, float]]
            (z_min, z_max) spanning from start to end layer.
        """
        start = drill_layer.start_layer
        end = drill_layer.end_layer

        if not start or not end:
            return None

        start_bounds = z_bounds_map.get(start)
        end_bounds = z_bounds_map.get(end)

        if not start_bounds or not end_bounds:
            return None

        # Drill spans from top of start layer to bottom of end layer
        # (or vice versa depending on direction)
        z_min = min(start_bounds[0], end_bounds[0])
        z_max = max(start_bounds[1], end_bounds[1])

        return (z_min, z_max)

