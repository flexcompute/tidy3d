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

## Limitations

This is an MVP implementation with the following limitations:

1. **Net assignment**: Not implemented yet. All geometries have `net=None`.

2. **Trace grouping**: Each L/A record creates a separate Path2D. Connected
   traces are not merged.

3. **Stackup z_bounds**: Set to placeholder values `(0.0, 1.0)`. User must
   configure actual values for 3D conversion.

4. **Rotation support**: Limited for non-90° rotations on rectangular pads.

5. **Complex symbols**: Some exotic symbols (donut_rc, thermal, etc.) are
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

