"""GDS and gdstk export helpers for concrete FDTD simulations."""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING, Any

import autograd.numpy as np

from tidy3d.components.geometry.base import Geometry
from tidy3d.components.structure import Structure
from tidy3d.exceptions import (
    Tidy3dError,
    Tidy3dImportError,
)

if TYPE_CHECKING:
    from os import PathLike

    from pydantic import NonNegativeFloat, NonNegativeInt, PositiveFloat

    from tidy3d.components.material.types import StructureMediumType
    from tidy3d.components.medium import AbstractMedium


OpticalMediumExportKey = dict[str, Any] | None

try:
    gdstk_available = True
    import gdstk
except ImportError:
    gdstk_available = False


def to_gdstk(
    self: Any,
    x: float | None = None,
    y: float | None = None,
    z: float | None = None,
    permittivity_threshold: NonNegativeFloat = 1,
    frequency: PositiveFloat = 0,
    gds_layer_dtype_map: dict[AbstractMedium, tuple[NonNegativeInt, NonNegativeInt]] | None = None,
    pixel_exact: bool = False,
) -> list:
    """Convert a simulation's planar slice to a .gds type polygon list.

    Parameters
    ----------
    x : float = None
        Position of plane in x direction, only one of x,y,z can be specified to define plane.
    y : float = None
        Position of plane in y direction, only one of x,y,z can be specified to define plane.
    z : float = None
        Position of plane in z direction, only one of x,y,z can be specified to define plane.
    permittivity_threshold : float = 1
        Permittivity value used to define the shape boundaries for structures with custom
        medim
    frequency : float = 0
        Frequency for permittivity evaluation in case of custom medium (Hz).
    gds_layer_dtype_map : Dict
        Dictionary mapping mediums to GDSII layer and data type tuples.
    pixel_exact : bool = False
        If true export gds as pixel exact rectangles instead of gdstk contour if a custom medium is provided.

    Return
    ------
    List
        List of `gdstk.Polygon`.
    """
    if gds_layer_dtype_map is None:
        gds_layer_dtype_map = {}

    axis, _ = self.geometry.parse_xyz_kwargs(x=x, y=y, z=z)
    _, bmin = self.pop_axis(self.bounds[0], axis)
    _, bmax = self.pop_axis(self.bounds[1], axis)

    _, symmetry = self.pop_axis(self.symmetry, axis)
    if symmetry[0] != 0:
        bmin = (0, bmin[1])
    if symmetry[1] != 0:
        bmin = (bmin[0], 0)
    clip = gdstk.rectangle(bmin, bmax)

    optical_medium_export_key_cache: dict[StructureMediumType | None, OpticalMediumExportKey] = {}
    background_medium_key = self._optical_medium_export_key(
        Structure._get_optical_medium(self.medium), optical_medium_export_key_cache
    )

    polygons_by_layer: dict[tuple[int, int], list] = {}
    deferred_background_polygons_by_layer: dict[tuple[int, int], list] = {}
    layer_has_filled_region: dict[tuple[int, int], bool] = {}
    for structure in self.scene.sorted_structures:
        gds_layer, gds_dtype = gds_layer_dtype_map.get(structure.medium, (0, 0))
        structure_polygons = []
        for polygon in structure.to_gdstk(
            x=x,
            y=y,
            z=z,
            permittivity_threshold=permittivity_threshold,
            frequency=frequency,
            gds_layer=gds_layer,
            gds_dtype=gds_dtype,
            pixel_exact=pixel_exact,
        ):
            pmin, pmax = polygon.bounding_box()
            if pmin[0] < bmin[0] or pmin[1] < bmin[1] or pmax[0] > bmax[0] or pmax[1] > bmax[1]:
                structure_polygons.extend(
                    gdstk.boolean(clip, polygon, "and", layer=gds_layer, datatype=gds_dtype)
                )
            else:
                structure_polygons.append(polygon)

        if not structure_polygons:
            continue

        layer_key = (gds_layer, gds_dtype)
        layer_polygons = polygons_by_layer.get(layer_key, [])
        if self._structure_exports_as_filled_region(
            structure,
            background_medium_key=background_medium_key,
            optical_medium_export_key_cache=optical_medium_export_key_cache,
        ):
            if layer_polygons:
                polygons_by_layer[layer_key] = gdstk.boolean(
                    layer_polygons,
                    structure_polygons,
                    "or",
                    layer=gds_layer,
                    datatype=gds_dtype,
                )
            else:
                polygons_by_layer[layer_key] = structure_polygons
            deferred_background_polygons_by_layer.pop(layer_key, None)
            layer_has_filled_region[layer_key] = True
        elif layer_has_filled_region.get(layer_key, False):
            polygons_by_layer[layer_key] = gdstk.boolean(
                layer_polygons,
                structure_polygons,
                "not",
                layer=gds_layer,
                datatype=gds_dtype,
            )
        else:
            deferred_polygons = deferred_background_polygons_by_layer.get(layer_key, [])
            if deferred_polygons:
                deferred_background_polygons_by_layer[layer_key] = gdstk.boolean(
                    deferred_polygons,
                    structure_polygons,
                    "or",
                    layer=gds_layer,
                    datatype=gds_dtype,
                )
            else:
                deferred_background_polygons_by_layer[layer_key] = structure_polygons

    for layer_key, deferred_polygons in deferred_background_polygons_by_layer.items():
        if layer_has_filled_region.get(layer_key, False):
            continue

        gds_layer, gds_dtype = layer_key
        layer_polygons = polygons_by_layer.get(layer_key, [])
        if layer_polygons:
            polygons_by_layer[layer_key] = gdstk.boolean(
                layer_polygons,
                deferred_polygons,
                "or",
                layer=gds_layer,
                datatype=gds_dtype,
            )
        else:
            # Preserve legacy default-layer output for unmapped background structures when no
            # filled region was accumulated on that layer.
            polygons_by_layer[layer_key] = deferred_polygons

    polygons = []
    for layer_polygons in polygons_by_layer.values():
        polygons.extend(layer_polygons)
    return polygons


@staticmethod
def _structure_exports_as_filled_region(
    structure: Structure,
    *,
    background_medium_key: OpticalMediumExportKey,
    optical_medium_export_key_cache: dict[StructureMediumType | None, OpticalMediumExportKey],
) -> bool:
    """Whether a structure should add or clear area on its export layer."""
    return (
        _optical_medium_export_key(
            Structure._get_optical_medium(structure.medium), optical_medium_export_key_cache
        )
        != background_medium_key
    )


@staticmethod
def _optical_medium_export_key(
    medium: StructureMediumType | None,
    cache: dict[StructureMediumType | None, OpticalMediumExportKey],
) -> OpticalMediumExportKey:
    """Normalized optical-medium key used for GDS export semantics."""
    if medium in cache:
        return cache[medium]
    if medium is None:
        cache[medium] = None
        return None
    exclude_fields = {"name", "attrs"}
    cache[medium] = medium.model_dump(exclude=exclude_fields, round_trip=True)
    return cache[medium]


def to_gds(
    self: Any,
    cell: gdstk.Cell,
    x: float | None = None,
    y: float | None = None,
    z: float | None = None,
    permittivity_threshold: NonNegativeFloat = 1,
    frequency: PositiveFloat = 0,
    gds_layer_dtype_map: dict[AbstractMedium, tuple[NonNegativeInt, NonNegativeInt]] | None = None,
    pixel_exact: bool = False,
) -> None:
    """Append the simulation structures to a .gds cell.

    Parameters
    ----------
    cell : ``gdstk.Cell``
        Cell object to which the generated polygons are added.
    x : float = None
        Position of plane in x direction, only one of x,y,z can be specified to define plane.
    y : float = None
        Position of plane in y direction, only one of x,y,z can be specified to define plane.
    z : float = None
        Position of plane in z direction, only one of x,y,z can be specified to define plane.
    permittivity_threshold : float = 1
        Permittivity value used to define the shape boundaries for structures with custom
        medim
    frequency : float = 0
        Frequency for permittivity evaluation in case of custom medium (Hz).
    gds_layer_dtype_map : Dict
        Dictionary mapping mediums to GDSII layer and data type tuples.
    pixel_exact : bool = False
        If true export gds as pixel exact rectangles instead of gdstk contour if a custom medium is provided.
    """
    if gds_layer_dtype_map is None:
        gds_layer_dtype_map = {}

    if gdstk_available and isinstance(cell, gdstk.Cell):
        polygons = self.to_gdstk(
            x=x,
            y=y,
            z=z,
            permittivity_threshold=permittivity_threshold,
            frequency=frequency,
            gds_layer_dtype_map=gds_layer_dtype_map,
            pixel_exact=pixel_exact,
        )
        if len(polygons) > 0:
            cell.add(*polygons)

    elif not gdstk_available:
        raise Tidy3dImportError(
            "Module 'gdstk' not found. It is required to export shapes to gdstk cells."
        )
    else:
        raise Tidy3dError("Argument 'cell' must be an instance of 'gdstk.Cell'.")


def to_gds_file(
    self: Any,
    fname: PathLike,
    x: float | None = None,
    y: float | None = None,
    z: float | None = None,
    permittivity_threshold: NonNegativeFloat = 1,
    frequency: PositiveFloat = 0,
    gds_layer_dtype_map: dict[AbstractMedium, tuple[NonNegativeInt, NonNegativeInt]] | None = None,
    gds_cell_name: str = "MAIN",
    pixel_exact: bool = False,
    gds_precision: PositiveFloat = 1e-3,
) -> None:
    """Append the simulation structures to a .gds cell.

    Parameters
    ----------
    fname : PathLike
        Full path to the .gds file to save the :class:`.Simulation` slice to.
    x : float = None
        Position of plane in x direction, only one of x,y,z can be specified to define plane.
    y : float = None
        Position of plane in y direction, only one of x,y,z can be specified to define plane.
    z : float = None
        Position of plane in z direction, only one of x,y,z can be specified to define plane.
    permittivity_threshold : float = 1
        Permittivity value used to define the shape boundaries for structures with custom
        medim
    frequency : float = 0
        Frequency for permittivity evaluation in case of custom medium (Hz).
    gds_layer_dtype_map : Dict
        Dictionary mapping mediums to GDSII layer and data type tuples.
    gds_cell_name : str = 'MAIN'
        Name of the cell created in the .gds file to store the geometry.
    pixel_exact : bool = False
        If true export gds as pixel exact rectangles instead of gdstk contour if a custom medium is provided.
    gds_precision : float = 1e-3
        Coordinate precision for the written GDS file in micrometers. The default matches
        the gdstk default of ``1e-9`` meters. If the requested precision is too fine for the
        written slice coordinates, export raises :class:`.SetupError`. The minimum safe value
        scales with the maximum absolute written planar coordinate as
        ``max_abs_coord / (2**31 - 1)``.
    """
    if gdstk_available:
        polygons = self.to_gdstk(
            x=x,
            y=y,
            z=z,
            permittivity_threshold=permittivity_threshold,
            frequency=frequency,
            gds_layer_dtype_map=gds_layer_dtype_map,
            pixel_exact=pixel_exact,
        )
        gds_precision = Geometry._validate_gds_precision(
            polygons=polygons,
            gds_precision=gds_precision,
            context="Simulation.to_gds_file()",
        )
        library = gdstk.Library(unit=1e-6, precision=gds_precision * 1e-6)
        reference = gdstk.Reference
        rotation = np.pi
    else:
        raise Tidy3dImportError(
            "Python module 'gdstk' not found. To export geometries to .gds "
            "files, please install 'gdstk'."
        )
    cell = library.new_cell(gds_cell_name)

    axis, _ = self.geometry.parse_xyz_kwargs(x=x, y=y, z=z)
    _, symmetry = self.pop_axis(self.symmetry, axis)
    if symmetry[0] != 0:
        outer_cell = cell
        cell = library.new_cell(gds_cell_name + "_X")
        outer_cell.add(reference(cell))
        outer_cell.add(reference(cell, rotation=rotation, x_reflection=True))
    if symmetry[1] != 0:
        outer_cell = cell
        cell = library.new_cell(gds_cell_name + "_Y")
        outer_cell.add(reference(cell))
        outer_cell.add(reference(cell, x_reflection=True))

    if polygons:
        cell.add(*polygons)
    fname = pathlib.Path(fname)
    fname.parent.mkdir(parents=True, exist_ok=True)
    library.write_gds(fname)
