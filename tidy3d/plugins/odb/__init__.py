"""ODB++ import plugin for tidy3d.

Load ODB++ PCB design files into the 2D geometry layer system.

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
>>> # Access stackup info
>>> for layer_spec in structure.stackup.layers:
>>>     print(f"{layer_spec.name}: z={layer_spec.z_bounds}")
"""

from tidy3d.plugins.odb.loader import ODBLoader
from tidy3d.plugins.odb.parser import (
    FeaturesData,
    LayerDef,
    MatrixData,
    parse_features,
    parse_matrix,
)
from tidy3d.plugins.odb.stackup_builder import (
    DEFAULT_COPPER_CONDUCTIVITY,
    DEFAULT_FREQ_RANGE,
    BoardInfo,
    LayerStackupInfo,
    StackupBuilder,
    StackupData,
    parse_board_attrlist,
    parse_layer_attrlist,
    parse_stackup_data,
)
from tidy3d.plugins.odb.symbols import SymbolInfo, parse_symbol, register_symbol_parser

__all__ = [
    # Core loader
    "ODBLoader",
    # Parser types
    "FeaturesData",
    "LayerDef",
    "MatrixData",
    "parse_features",
    "parse_matrix",
    # Stackup
    "BoardInfo",
    "DEFAULT_COPPER_CONDUCTIVITY",
    "DEFAULT_FREQ_RANGE",
    "LayerStackupInfo",
    "StackupBuilder",
    "StackupData",
    "parse_board_attrlist",
    "parse_layer_attrlist",
    "parse_stackup_data",
    # Symbols
    "SymbolInfo",
    "parse_symbol",
    "register_symbol_parser",
]

