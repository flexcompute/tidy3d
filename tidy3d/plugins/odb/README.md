# ODB++ Import Plugin

Load ODB++ PCB design files into tidy3d's 2D geometry layer system.

## Quick Start

```python
from tidy3d.plugins.odb import ODBLoader

# Load ODB++ design
loader = ODBLoader("./my_design.odb")
structure = loader.load()

# Check what was loaded
print(f"Layers: {structure.layer_names}")
print(f"Total geometries: {len(structure.geometries)}")

# Access geometries by layer
for layer_name in structure.layer_names:
    geoms = structure.geometries_on_layer(layer_name)
    print(f"  {layer_name}: {len(geoms)} geometries")
```

## Features

### Supported ODB++ Features

| ODB++ Record | Maps To | Notes |
|--------------|---------|-------|
| L (Line) | Path2D | Uses symbol width for stroke |
| A (Arc) | Path2D | Uses symbol width, round symbols only |
| P (Pad) | Circle2D, Rectangle2D, Polygon2D | Depends on symbol type |
| S (Surface) | Polygon2D | With holes and arc edges |

### Supported Symbols

| Symbol | Example | Notes |
|--------|---------|-------|
| Round | `r200` | Diameter in microns (MM mode) |
| Square | `s150` | Side length |
| Rectangle | `rect250x150` | Width × height |
| Oval | `oval200x100` | Approximated as rectangle |
| Donut Round | `donut_r300x150` | Outer × inner diameter |
| Donut Square | `donut_s400x200` | Outer × inner side |

### Filter by Layer

```python
# Load only specific layers
structure = loader.load(layers=["TRACE", "GND", "VIAS"])
```

### Multiple Steps

```python
# Check available steps
print(loader.step_names)

# Load specific step
structure = loader.load(step="pcb")
```

## Automatic Stackup Construction

The loader automatically constructs a physical stackup from ODB++ data:

```python
# Load with automatic stackup
structure = loader.load()

# Inspect stackup
for layer_spec in structure.stackup.layers:
    print(f"{layer_spec.name}: z={layer_spec.z_bounds}, medium={layer_spec.medium}")
```

### Stackup Data Sources

The loader parses stackup information from:

| Source | Data |
|--------|------|
| `matrix/matrix` | Layer names, types, order |
| `misc/attrlist` | Board thickness |
| `layers/<name>/attrlist` | Thickness, permittivity, loss tangent |

### Custom Thickness Overrides

```python
# Override specific layer thicknesses (in microns)
structure = loader.load(
    layer_thicknesses={
        "TOP": 70.0,     # 2oz copper
        "BOTTOM": 35.0,  # 1oz copper
    }
)
```

### Default Values

| Parameter | Default | Description |
|-----------|---------|-------------|
| `default_conductor_thickness` | 35.0 µm | 1 oz copper |
| `default_dielectric_thickness` | 200.0 µm | Typical prepreg |
| `default_permittivity` | 4.2 | FR4 dielectric constant |
| `frequency_range` | (0.1e9, 10e9) Hz | For lossy material models |
| `DEFAULT_COPPER_CONDUCTIVITY` | 58.0 S/µm | Used when bulk_resistivity not available |

### Lossy Material Models

The loader automatically creates appropriate lossy material models based on
ODB++ attributes:

**Conductors (LossyMetalMedium):**
```python
import tidy3d.rf as rf

# Conductivity from bulk_resistivity attribute (nΩ·cm → S/µm)
# Or default copper: 58 S/µm
copper = rf.LossyMetalMedium(conductivity=58, frequency_range=(0.1e9, 10e9))
```

**Lossy Dielectrics (PoleResidue from FastDispersionFitter):**
```python
from tidy3d.plugins.dispersion import FastDispersionFitter

# From dielectric_constant and loss_tangent attributes
lossy_fr4 = FastDispersionFitter.constant_loss_tangent_model(
    eps_real=4.2,
    loss_tangent=0.02,
    frequency_range=(0.1e9, 10e9),
)
```

### Control Lossy Material Behavior

```python
# Use lossy metals and dielectrics (default)
structure = loader.load()

# Use PEC for conductors (faster simulation)
structure = loader.load(use_lossy_metal=False)

# Use lossless dielectrics
structure = loader.load(use_lossy_dielectric=False)

# Custom frequency range
structure = loader.load(frequency_range=(1e9, 20e9))
```

### Include Dielectric Layers

```python
# Include dielectric layers in stackup
structure = loader.load(include_dielectrics=True)
```

### Drill Layer Span

Drill layers specify `START_NAME` and `END_NAME` for via spans:

```python
from tidy3d.plugins.odb import StackupBuilder

builder = StackupBuilder(odb_path, loader.matrix, step_name)
for drill in builder.get_drill_layers():
    print(f"{drill.name}: {drill.start_layer} -> {drill.end_layer}")
```

## Limitations

This is an MVP implementation with the following limitations:

1. **Net assignment**: Not implemented yet. All geometries have `net=None`.

2. **Trace grouping**: Each L/A record creates a separate Path2D. Connected
   traces are not merged.

3. **Rotation support**: Limited for non-90° rotations on rectangular pads.

4. **Complex symbols**: Some exotic symbols (donut_rc, thermal, etc.) are
   not yet supported.

## Extending Symbol Support

Register custom symbol parsers for unsupported symbols:

```python
from tidy3d.plugins.odb import register_symbol_parser, SymbolInfo
import re

def parse_custom_symbol(name: str):
    match = re.match(r"^custom_(\d+)x(\d+)$", name)
    if match:
        return SymbolInfo(
            type="unknown",  # Or add new type
            params={
                "width": float(match.group(1)),
                "height": float(match.group(2)),
            }
        )
    return None

register_symbol_parser("custom_", parse_custom_symbol)
```

## API Reference

### ODBLoader

```python
class ODBLoader:
    def __init__(self, path: str | Path):
        """Create loader for ODB++ directory."""
    
    @property
    def layer_names(self) -> list[str]:
        """Layer names from matrix file."""
    
    @property
    def step_names(self) -> list[str]:
        """Step names from matrix file."""
    
    def load(
        self,
        step: str = None,  # Default: first step
        layers: list[str] = None,  # Default: all layers
    ) -> LayeredStructure:
        """Load ODB++ into LayeredStructure."""
```

### StackupBuilder

```python
from tidy3d.plugins.odb import StackupBuilder

class StackupBuilder:
    def __init__(self, odb_path: Path, matrix: MatrixData, step_name: str):
        """Create builder for stackup construction."""
    
    @property
    def stackup_data(self) -> StackupData:
        """Parsed stackup data including layer properties."""
    
    def build(
        self,
        layer_filter: set[str] = None,
        default_conductor_thickness: float = 35.0,
        default_dielectric_thickness: float = 200.0,
        default_permittivity: float = 4.2,
        include_dielectrics: bool = False,
    ) -> Stackup:
        """Build Stackup with z_bounds and mediums."""
    
    def get_drill_layers(self) -> list[LayerStackupInfo]:
        """Get drill layer definitions with start/end spans."""
```

### Low-level Parsing

```python
from tidy3d.plugins.odb import parse_matrix, parse_features

# Parse matrix file directly
with open("design.odb/matrix/matrix") as f:
    matrix_data = parse_matrix(f.read())

# Parse features file directly
with open("design.odb/steps/pcb/layers/TRACE/features") as f:
    features_data = parse_features(f.read())
```

