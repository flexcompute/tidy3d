"""GDS/OASIS import plugin for tidy3d.

Load GDSII and OASIS layout files into the 2D geometry layer system.

Example
-------
>>> from tidy3d.plugins.gds import GDSLoader
>>>
>>> loader = GDSLoader("design.gds")
>>> structure = loader.load(gds_scale=1e-3)  # GDS in nm → µm
>>>
>>> # View auto-generated layer names
>>> print(structure.stackup.layer_names)
>>> # ['(1, 0)', '(2, 0)', '(10, 5)']
>>>
>>> # Convert to 3D structures for simulation
>>> structures_3d = structure.to_structures()
"""

from tidy3d.plugins.gds.loader import GDSLoader

__all__ = [
    "GDSLoader",
]

