"""GDS import and export helpers for geometry."""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING, Any

import shapely

from tidy3d.exceptions import (
    SetupError,
    Tidy3dError,
    Tidy3dImportError,
    Tidy3dKeyError,
    ValidationError,
    format_chained_exception_message,
)
from tidy3d.log import log
from tidy3d.packaging import verify_packages_import

from .constants import POLY_GRID_SIZE

if TYPE_CHECKING:
    from collections.abc import Iterable
    from os import PathLike

    from gdstk import Cell
    from numpy.typing import ArrayFloat2D, NDArray
    from pydantic import NonNegativeInt, PositiveFloat

    from tidy3d.components.types import Axis, PlanePosition, Shapely

    from .core import Geometry


@verify_packages_import(["gdstk"])
def load_gds_vertices_gdstk(
    gds_cell: Cell,
    gds_layer: int,
    gds_dtype: int | None = None,
    gds_scale: PositiveFloat = 1.0,
) -> list[ArrayFloat2D]:
    """Load polygon vertices from a ``gdstk.Cell``.

    Parameters
    ----------
    gds_cell : gdstk.Cell
        ``gdstk.Cell`` containing 2D geometric data.
    gds_layer : int
        Layer index in the ``gds_cell``.
    gds_dtype : int = None
        Data-type index in the ``gds_cell``. If ``None``, imports all data for this layer into
        the returned list.
    gds_scale : float = 1.0
        Length scale used in GDS file in units of micrometer. For example, if gds file uses
        nanometers, set ``gds_scale=1e-3``. Must be positive.

    Returns
    -------
    list[ArrayFloat2D]
        List of polygon vertices
    """

    # apply desired scaling and load the polygon vertices
    if gds_dtype is not None:
        # if both layer and datatype are specified, let gdstk do the filtering for better
        # performance on large layouts
        all_vertices = [
            polygon.scale(gds_scale).points
            for polygon in gds_cell.get_polygons(layer=gds_layer, datatype=gds_dtype)
        ]
    else:
        all_vertices = [
            polygon.scale(gds_scale).points
            for polygon in gds_cell.get_polygons()
            if polygon.layer == gds_layer
        ]
    # make sure something got loaded, otherwise error
    if not all_vertices:
        raise Tidy3dKeyError(
            f"Couldn't load gds_cell, no vertices found at gds_layer={gds_layer} "
            f"with specified gds_dtype={gds_dtype}."
        )

    return all_vertices


@verify_packages_import(["gdstk"])
def from_gds(
    gds_cell: Cell,
    axis: Axis,
    slab_bounds: tuple[float, float],
    gds_layer: int,
    gds_dtype: int | None = None,
    gds_scale: PositiveFloat = 1.0,
    dilation: float = 0.0,
    sidewall_angle: float = 0,
    reference_plane: PlanePosition = "middle",
    merge_adjacent: bool = False,
) -> Geometry:
    """Import a ``gdstk.Cell`` and extrude it into a GeometryGroup.

    Parameters
    ----------
    gds_cell : gdstk.Cell
        ``gdstk.Cell`` containing 2D geometric data.
    axis : int
        Integer index defining the extrusion axis: 0 (x), 1 (y), or 2 (z).
    slab_bounds: tuple[float, float]
        Minimal and maximal positions of the extruded slab along ``axis``.
    gds_layer : int
        Layer index in the ``gds_cell``.
    gds_dtype : int = None
        Data-type index in the ``gds_cell``. If ``None``, imports all data for this layer into
        the returned list.
    gds_scale : float = 1.0
        Length scale used in GDS file in units of micrometer. For example, if gds file uses
        nanometers, set ``gds_scale=1e-3``. Must be positive.
    dilation : float = 0.0
        Dilation (positive) or erosion (negative) amount to be applied to the original polygons.
    sidewall_angle : float = 0
        Angle of the extrusion sidewalls, away from the vertical direction, in radians. Positive
        (negative) values result in slabs larger (smaller) at the base than at the top.
    reference_plane : PlanePosition = "middle"
        Reference position of the (dilated/eroded) polygons along the slab axis. One of
        ``"middle"`` (polygons correspond to the center of the slab bounds), ``"bottom"``
        (minimal slab bound position), or ``"top"`` (maximal slab bound position). This value
        has no effect if ``sidewall_angle == 0``.
    merge_adjacent : bool = False
        Merge polygons that become fractured into multiple adjacent GDS polygons, for example
        due to the GDS vertex limit. Enable to import those fragments as a single merged shape.

    Returns
    -------
    :class:`~tidy3d.Geometry`
        Geometries created from the 2D data.
    """
    import gdstk

    from .core import Geometry

    if not isinstance(gds_cell, gdstk.Cell):
        # Check if it might be a gdstk cell but gdstk is not found (should be caught by decorator)
        # or if it's an entirely different type.
        if "gdstk" in gds_cell.__class__.__name__.lower():
            raise Tidy3dImportError(
                "Module 'gdstk' not found. It is required to import gdstk cells."
            )
        raise Tidy3dImportError("Argument 'gds_cell' must be an instance of 'gdstk.Cell'.")

    def iter_import_shapes(shape: Shapely) -> Iterable[Shapely]:
        if shape.is_empty:
            return
        if shape.geom_type in {"MultiPolygon", "GeometryCollection"}:
            for subshape in shape.geoms:
                yield from iter_import_shapes(subshape)
        else:
            yield shape

    def cleaned_shape(vertices: NDArray, consolidated_logger: Any) -> Shapely | None:
        shape = shapely.set_precision(shapely.Polygon(vertices).buffer(0), POLY_GRID_SIZE)
        if shape.is_empty:
            consolidated_logger.warning(
                "A GDS polygon collapsed during topology cleanup in "
                "'Geometry.from_gds()' and will be skipped."
            )
            return None
        return shape

    geometries = []
    with log as consolidated_logger:
        gds_loader_fn = Geometry.load_gds_vertices_gdstk
        all_vertices = gds_loader_fn(gds_cell, gds_layer, gds_dtype, gds_scale)

        if merge_adjacent:
            shapes = []
            for vertices in all_vertices:
                shape = cleaned_shape(vertices, consolidated_logger)
                if shape is not None:
                    shapes.append(shape)

            if len(shapes) > 1:
                shapes = [shapely.set_precision(shapely.union_all(shapes), POLY_GRID_SIZE)]

            import_shapes = (
                import_shape for shape in shapes for import_shape in iter_import_shapes(shape)
            )
        else:
            import_shapes = (
                import_shape
                for vertices in all_vertices
                for shape in [cleaned_shape(vertices, consolidated_logger)]
                if shape is not None
                for import_shape in iter_import_shapes(shape)
            )

        from tidy3d.components.geometry import base as geometry_base

        for import_shape in import_shapes:
            try:
                geometries.append(
                    geometry_base.from_shapely(
                        import_shape,
                        axis,
                        slab_bounds,
                        dilation,
                        sidewall_angle,
                        reference_plane,
                    )
                )
            except ValidationError as error:
                consolidated_logger.warning(str(error))
            except Tidy3dError as error:
                consolidated_logger.warning(str(error))
    if not geometries:
        raise SetupError(
            "Couldn't import any valid geometries from 'gds_cell' at "
            f"gds_layer={gds_layer} with specified gds_dtype={gds_dtype}. "
            "All polygons were skipped during cleanup or failed conversion."
        )
    if len(geometries) == 1:
        return geometries[0]
    from .geometry_group import GeometryGroup

    return GeometryGroup(geometries=geometries)


def from_shapely(
    shape: Shapely,
    axis: Axis,
    slab_bounds: tuple[float, float],
    dilation: float = 0.0,
    sidewall_angle: float = 0,
    reference_plane: PlanePosition = "middle",
) -> Geometry:
    """Convert a shapely primitive into a geometry instance by extrusion.

    Parameters
    ----------
    shape : shapely.geometry.base.BaseGeometry
        Shapely primitive to be converted. It must be a linear ring, a polygon or a collection
        of any of those.
    axis : int
        Integer index defining the extrusion axis: 0 (x), 1 (y), or 2 (z).
    slab_bounds: tuple[float, float]
        Minimal and maximal positions of the extruded slab along ``axis``.
    dilation : float
        Dilation of the polygon in the base by shifting each edge along its normal outwards
        direction by a distance; a negative value corresponds to erosion.
    sidewall_angle : float = 0
        Angle of the extrusion sidewalls, away from the vertical direction, in radians. Positive
        (negative) values result in slabs larger (smaller) at the base than at the top.
    reference_plane : PlanePosition = "middle"
        Reference position of the (dilated/eroded) polygons along the slab axis. One of
        ``"middle"`` (polygons correspond to the center of the slab bounds), ``"bottom"``
        (minimal slab bound position), or ``"top"`` (maximal slab bound position). This value
        has no effect if ``sidewall_angle == 0``.

    Returns
    -------
    :class:`~tidy3d.Geometry`
        Geometry extruded from the 2D data.
    """
    from tidy3d.components.geometry import base as geometry_base

    return geometry_base.from_shapely(
        shape, axis, slab_bounds, dilation, sidewall_angle, reference_plane
    )


@verify_packages_import(["gdstk"])
def to_gdstk(
    self,  # pyrefly: ignore[implicit-any-parameter]
    x: float | None = None,
    y: float | None = None,
    z: float | None = None,
    gds_layer: NonNegativeInt = 0,
    gds_dtype: NonNegativeInt = 0,
) -> list:
    """Convert a Geometry object's planar slice to a .gds type polygon.

    Parameters
    ----------
    x : float = None
        Position of plane in x direction, only one of x,y,z can be specified to define plane.
    y : float = None
        Position of plane in y direction, only one of x,y,z can be specified to define plane.
    z : float = None
        Position of plane in z direction, only one of x,y,z can be specified to define plane.
    gds_layer : int = 0
        Layer index to use for the shapes stored in the .gds file.
    gds_dtype : int = 0
        Data-type index to use for the shapes stored in the .gds file.

    Return
    ------
    List
        List of `gdstk.Polygon`.
    """
    import gdstk

    shapes = self.intersections_plane(x=x, y=y, z=z)
    polygons = []
    for shape in shapes:
        from tidy3d.components.geometry import base as geometry_base

        for vertices in geometry_base.vertices_from_shapely(shape):
            if len(vertices) == 1:
                polygons.append(gdstk.Polygon(vertices[0], gds_layer, gds_dtype))
            else:
                polygons.extend(
                    gdstk.boolean(
                        vertices[:1],
                        vertices[1:],
                        "not",
                        layer=gds_layer,
                        datatype=gds_dtype,
                    )
                )
    return polygons


@verify_packages_import(["gdstk"])
def to_gds(
    self,  # pyrefly: ignore[implicit-any-parameter]
    cell: Cell,
    x: float | None = None,
    y: float | None = None,
    z: float | None = None,
    gds_layer: NonNegativeInt = 0,
    gds_dtype: NonNegativeInt = 0,
) -> None:
    """Append a Geometry object's planar slice to a .gds cell.

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
    gds_layer : int = 0
        Layer index to use for the shapes stored in the .gds file.
    gds_dtype : int = 0
        Data-type index to use for the shapes stored in the .gds file.
    """
    import gdstk

    if not isinstance(cell, gdstk.Cell):
        if "gdstk" in cell.__class__.__name__.lower():
            raise Tidy3dImportError(
                "Module 'gdstk' not found. It is required to export shapes to gdstk cells."
            )
        raise Tidy3dImportError("Argument 'cell' must be an instance of 'gdstk.Cell'.")

    polygons = self.to_gdstk(x=x, y=y, z=z, gds_layer=gds_layer, gds_dtype=gds_dtype)
    if polygons:
        cell.add(*polygons)


@verify_packages_import(["gdstk"])
def to_gds_file(
    self,  # pyrefly: ignore[implicit-any-parameter]
    fname: PathLike,
    x: float | None = None,
    y: float | None = None,
    z: float | None = None,
    gds_layer: NonNegativeInt = 0,
    gds_dtype: NonNegativeInt = 0,
    gds_cell_name: str = "MAIN",
    gds_precision: PositiveFloat = 1e-3,
) -> None:
    """Export a Geometry object's planar slice to a .gds file.

    Parameters
    ----------
    fname : PathLike
        Full path to the .gds file to save the :class:`~tidy3d.Geometry` slice to.
    x : float = None
        Position of plane in x direction, only one of x,y,z can be specified to define plane.
    y : float = None
        Position of plane in y direction, only one of x,y,z can be specified to define plane.
    z : float = None
        Position of plane in z direction, only one of x,y,z can be specified to define plane.
    gds_layer : int = 0
        Layer index to use for the shapes stored in the .gds file.
    gds_dtype : int = 0
        Data-type index to use for the shapes stored in the .gds file.
    gds_cell_name : str = 'MAIN'
        Name of the cell created in the .gds file to store the geometry.
    gds_precision : float = 1e-3
        Coordinate precision for the written GDS file in micrometers. The default matches
        the gdstk default of ``1e-9`` meters. If the requested precision is too fine for the
        written slice coordinates, export raises :class:`.SetupError`. The minimum safe value
        scales with the maximum absolute written planar coordinate as
        ``max_abs_coord / (2**31 - 1)``.
    """
    try:
        import gdstk
    except ImportError as e:
        raise Tidy3dImportError(
            format_chained_exception_message(
                "Python module 'gdstk' not found. To export geometries to .gds files, "
                "please install it",
                e,
            )
        ) from e

    polygons = self.to_gdstk(
        x=x,
        y=y,
        z=z,
        gds_layer=gds_layer,
        gds_dtype=gds_dtype,
    )
    gds_precision = self._validate_gds_precision(
        polygons=polygons,
        gds_precision=gds_precision,
        context="Geometry.to_gds_file()",
    )
    library = gdstk.Library(unit=1e-6, precision=gds_precision * 1e-6)
    cell = library.new_cell(gds_cell_name)
    if polygons:
        cell.add(*polygons)
    fname = pathlib.Path(fname)
    fname.parent.mkdir(parents=True, exist_ok=True)
    library.write_gds(fname)
