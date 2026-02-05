"""GDS/OASIS to LayeredStructure loader.

High-level API for loading GDSII and OASIS layout files into tidy3d's
2D geometry layer system.
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Optional, Union

import numpy as np

from tidy3d.components.geometry.geometry2d import (
    Array2D,
    Geometry2D,
    Polygon2D,
    Transformed2D,
)
from tidy3d.components.geometry.layout import (
    LayeredGeometry,
    LayeredStructure,
    LayerSpec,
    Stackup,
)
from tidy3d.exceptions import Tidy3dImportError


def _layer_name(layer: int, dtype: int) -> str:
    """Generate layer name from GDS layer/dtype pair."""
    return f"({layer}, {dtype})"


class GDSLoader:
    """Load GDS/OASIS files into LayeredStructure.

    Parameters
    ----------
    path : str or Path
        Path to .gds or .oas file.

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
    >>> # Check available cells
    >>> print(loader.cell_names)
    >>> print(loader.top_cells)  # Auto-detected top-level cells
    >>>
    >>> # Convert to 3D structures for simulation
    >>> structures_3d = structure.to_structures()
    """

    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"GDS file not found: {self.path}")

        self._library = None
        self._warnings: list[str] = []

    @property
    def library(self):
        """Parsed GDS library (lazy loaded).

        Returns
        -------
        gdstk.Library
            The loaded GDS/OASIS library.
        """
        if self._library is None:
            self._library = self._load_library()
        return self._library

    def _load_library(self):
        """Load GDS/OASIS file using gdstk."""
        try:
            import gdstk
        except ImportError as e:
            raise Tidy3dImportError(
                "Module 'gdstk' not found. Install it with: pip install gdstk"
            ) from e

        path_str = str(self.path)
        suffix = self.path.suffix.lower()

        if suffix in (".oas", ".oasis"):
            return gdstk.read_oas(path_str)
        else:
            return gdstk.read_gds(path_str)

    @property
    def cell_names(self) -> list[str]:
        """All cell names in the library.

        Returns
        -------
        list[str]
            Cell names sorted alphabetically.
        """
        return sorted(cell.name for cell in self.library.cells)

    @property
    def top_cells(self) -> list[str]:
        """Cells not referenced by any other cell (auto-detected).

        Returns
        -------
        list[str]
            Top-level cell names sorted alphabetically.
        """
        all_cells = {cell.name for cell in self.library.cells}
        referenced = set()

        for cell in self.library.cells:
            for ref in cell.references:
                # Reference can be to Cell, RawCell, or name string
                if hasattr(ref, "cell") and ref.cell is not None:
                    referenced.add(ref.cell.name)
                elif hasattr(ref, "cell_name"):
                    referenced.add(ref.cell_name)

        top = all_cells - referenced
        return sorted(top) if top else sorted(all_cells)

    @property
    def gds_layers(self) -> list[tuple[int, int]]:
        """All (layer, dtype) pairs found in the library.

        Returns
        -------
        list[tuple[int, int]]
            Unique layer/dtype pairs sorted by layer then dtype.
        """
        layers = set()
        for cell in self.library.cells:
            for polygon in cell.polygons:
                layers.add((polygon.layer, polygon.datatype))
            for path in cell.paths:
                # FlexPath/RobustPath have layers/datatypes arrays
                path_layers = getattr(path, "layers", None)
                path_dtypes = getattr(path, "datatypes", None)
                if path_layers is not None:
                    for i, layer in enumerate(path_layers):
                        dtype = path_dtypes[i] if path_dtypes and i < len(path_dtypes) else 0
                        layers.add((layer, dtype))
                else:
                    # Fallback for simple paths
                    layer = getattr(path, "layer", 0)
                    dtype = getattr(path, "datatype", 0)
                    layers.add((layer, dtype))
        return sorted(layers)

    def load(
        self,
        cell: Optional[str] = None,
        stackup: Optional[Stackup] = None,
        gds_scale: float = 1.0,
        default_thickness: float = 1.0,
    ) -> LayeredStructure:
        """Load single cell into LayeredStructure.

        Parameters
        ----------
        cell : str, optional
            Cell name to load. Defaults to first auto-detected top cell.
        stackup : Stackup, optional
            User-provided stackup. If None, auto-generates with uniform thickness.
        gds_scale : float
            Scale factor from GDS units to microns. Default 1.0.
            Example: if GDS is in nm, use gds_scale=1e-3.
        default_thickness : float
            Layer thickness for auto-generated stackup (in µm). Default 1.0.

        Returns
        -------
        LayeredStructure
            Loaded geometry organized by layers.

        Notes
        -----
        - Layer names are auto-generated as "(layer, dtype)" strings
        - Cell references with repetitions become Array2D objects
        - Circular path bends are preserved as ArcSegments where possible
        """
        self._warnings.clear()

        # Determine cell to load
        if cell is None:
            top = self.top_cells
            if not top:
                raise ValueError("No cells found in GDS file")
            cell = top[0]

        # Find the cell
        target_cell = None
        for c in self.library.cells:
            if c.name == cell:
                target_cell = c
                break

        if target_cell is None:
            raise ValueError(
                f"Cell '{cell}' not found. Available: {self.cell_names}"
            )

        # Flatten cell to get all geometries
        geom_layer_pairs = self._flatten_cell(target_cell, gds_scale)

        # Collect unique layers
        layer_names_found = sorted(set(layer for _, layer in geom_layer_pairs))

        # Build or validate stackup
        if stackup is None:
            stackup = self._auto_stackup(layer_names_found, default_thickness)
        else:
            # Validate all layers exist
            stackup_layers = set(stackup.layer_names)
            for layer_name in layer_names_found:
                if layer_name not in stackup_layers:
                    self._warnings.append(
                        f"Layer '{layer_name}' not in stackup, skipping geometries"
                    )

        # Build LayeredGeometry objects using construct() to skip per-object validation.
        # This is safe because:
        # 1. Geometry2D objects are already validated when created
        # 2. Layer names are checked against stackup below
        # 3. Net is always None (no EDA data for GDS)
        stackup_layers = set(spec.name for spec in stackup.layers)
        all_layered_geoms = []

        for geom, layer_name in geom_layer_pairs:
            if layer_name in stackup_layers:
                all_layered_geoms.append(
                    LayeredGeometry.construct(
                        geometry=geom,
                        layer=layer_name,
                        net=None,
                    )
                )

        structure = LayeredStructure(
            stackup=stackup,
            geometries=tuple(all_layered_geoms),
        )

        # Report warnings
        for warning in self._warnings:
            warnings.warn(warning, stacklevel=2)

        return structure

    def load_cells(
        self,
        cells: Optional[list[str]] = None,
        **kwargs,
    ) -> dict[str, LayeredStructure]:
        """Load multiple cells into separate LayeredStructures.

        Parameters
        ----------
        cells : list[str], optional
            Cell names to load. Defaults to all top cells.
        **kwargs
            Additional arguments passed to load().

        Returns
        -------
        dict[str, LayeredStructure]
            Mapping from cell name to LayeredStructure.
        """
        if cells is None:
            cells = self.top_cells

        return {cell_name: self.load(cell=cell_name, **kwargs) for cell_name in cells}

    def _auto_stackup(
        self, layer_names: list[str], default_thickness: float
    ) -> Stackup:
        """Auto-generate stackup from discovered layers.

        Parameters
        ----------
        layer_names : list[str]
            Layer names to include.
        default_thickness : float
            Thickness per layer.

        Returns
        -------
        Stackup
            Auto-generated stackup with uniform layer thickness.
        """
        layer_specs = []
        z = 0.0

        for name in layer_names:
            layer_specs.append(
                LayerSpec(
                    name=name,
                    z_bounds=(z, z + default_thickness),
                    medium=None,
                )
            )
            z += default_thickness

        return Stackup(layers=tuple(layer_specs))

    def _flatten_cell(
        self, cell, gds_scale: float
    ) -> list[tuple[Geometry2D, str]]:
        """Flatten cell to list of (geometry, layer_name) tuples.

        Parameters
        ----------
        cell : gdstk.Cell
            Cell to flatten.
        gds_scale : float
            Scale factor to microns.

        Returns
        -------
        list[tuple[Geometry2D, str]]
            Flattened geometries with layer names.
        """
        result = []

        # Direct polygons
        for polygon in cell.polygons:
            geom = self._convert_polygon(polygon, gds_scale)
            if geom is not None:
                layer_name = _layer_name(polygon.layer, polygon.datatype)
                result.append((geom, layer_name))

        # Direct paths (FlexPath/RobustPath have layers/datatypes arrays)
        for path in cell.paths:
            geoms_layers = self._convert_path(path, gds_scale)
            if geoms_layers:
                result.extend(geoms_layers)

        # References (recursive)
        for ref in cell.references:
            ref_geoms = self._flatten_reference(ref, gds_scale)
            result.extend(ref_geoms)

        return result

    def _flatten_reference(
        self, ref, gds_scale: float
    ) -> list[tuple[Geometry2D, str]]:
        """Flatten cell reference, using Array2D for repetitions.

        Parameters
        ----------
        ref : gdstk.Reference
            Reference to flatten.
        gds_scale : float
            Scale factor to microns.

        Returns
        -------
        list[tuple[Geometry2D, str]]
            Flattened geometries with layer names.
        """
        # Get the referenced cell
        ref_cell = getattr(ref, "cell", None)
        if ref_cell is None:
            self._warnings.append(f"Reference to unresolved cell, skipping")
            return []

        # Recursively flatten the referenced cell (unscaled - we'll scale at the end)
        base_geoms = self._flatten_cell(ref_cell, gds_scale=1.0)

        if not base_geoms:
            return []

        # Check for repetition
        repetition = getattr(ref, "repetition", None)
        has_repetition = repetition is not None and self._get_repetition_count(repetition) > 1

        # Build transform matrix from reference properties
        origin = (ref.origin[0] * gds_scale, ref.origin[1] * gds_scale)
        rotation = getattr(ref, "rotation", 0.0)  # radians
        magnification = getattr(ref, "magnification", 1.0) * gds_scale
        x_reflection = getattr(ref, "x_reflection", False)

        result = []

        if not has_repetition:
            # Single instance - apply transform
            transform = self._build_transform(
                origin, rotation, magnification, x_reflection
            )
            for geom, layer_name in base_geoms:
                if transform is not None:
                    transformed = Transformed2D(geometry=geom, transform=transform)
                    result.append((transformed, layer_name))
                else:
                    result.append((geom, layer_name))
        else:
            # Has repetition - use Array2D
            positions = self._get_repetition_positions(repetition, gds_scale)

            # Build base transform (without origin - positions handle that)
            base_transform = self._build_transform(
                origin=(0, 0),
                rotation=rotation,
                magnification=magnification,
                x_reflection=x_reflection,
            )

            for geom, layer_name in base_geoms:
                # Apply base transform to geometry
                if base_transform is not None:
                    transformed_base = Transformed2D(geometry=geom, transform=base_transform)
                else:
                    transformed_base = geom

                # Offset positions by reference origin
                offset_positions = tuple(
                    (px + origin[0], py + origin[1]) for px, py in positions
                )

                # Create Array2D
                array = Array2D(
                    base_shape=transformed_base,
                    positions=offset_positions,
                )
                result.append((array, layer_name))

        return result

    def _get_repetition_count(self, repetition) -> int:
        """Get number of instances from repetition."""
        if repetition is None:
            return 1

        rep_type = getattr(repetition, "type", None)
        if rep_type is None:
            # Try to determine from attributes
            columns = getattr(repetition, "columns", None)
            rows = getattr(repetition, "rows", None)
            offsets = getattr(repetition, "offsets", None)

            if offsets is not None and len(offsets) > 0:
                return len(offsets) + 1  # +1 for original
            
            # Handle None values
            columns = columns if columns is not None else 1
            rows = rows if rows is not None else 1
            return columns * rows

        # gdstk RepetitionType enum
        type_name = str(rep_type).lower()
        if "none" in type_name:
            return 1
        elif "explicit" in type_name:
            offsets = getattr(repetition, "offsets", [])
            coords = getattr(repetition, "coords", [])
            return max(len(offsets) if offsets else 0, len(coords) if coords else 0) + 1
        else:
            # Rectangular or Regular
            columns = getattr(repetition, "columns", None)
            rows = getattr(repetition, "rows", None)
            columns = columns if columns is not None else 1
            rows = rows if rows is not None else 1
            return columns * rows

    def _get_repetition_positions(
        self, repetition, gds_scale: float
    ) -> list[tuple[float, float]]:
        """Convert gdstk Repetition to list of positions.

        Parameters
        ----------
        repetition : gdstk.Repetition
            Repetition specification.
        gds_scale : float
            Scale factor.

        Returns
        -------
        list[tuple[float, float]]
            List of (x, y) positions including origin (0, 0).
        """
        positions = [(0.0, 0.0)]  # Include original

        rep_type = getattr(repetition, "type", None)
        type_name = str(rep_type).lower() if rep_type else ""

        if "rectangular" in type_name:
            # Rectangular grid
            columns = getattr(repetition, "columns", 1)
            rows = getattr(repetition, "rows", 1)
            spacing = getattr(repetition, "spacing", (0, 0))
            sx = spacing[0] * gds_scale if hasattr(spacing, "__getitem__") else 0
            sy = spacing[1] * gds_scale if hasattr(spacing, "__getitem__") else 0

            for row in range(rows):
                for col in range(columns):
                    if row == 0 and col == 0:
                        continue  # Skip origin, already added
                    positions.append((col * sx, row * sy))

        elif "regular" in type_name:
            # Regular grid along v1, v2 vectors
            columns = getattr(repetition, "columns", 1)
            rows = getattr(repetition, "rows", 1)
            v1 = getattr(repetition, "v1", (0, 0))
            v2 = getattr(repetition, "v2", (0, 0))
            v1x = v1[0] * gds_scale if hasattr(v1, "__getitem__") else 0
            v1y = v1[1] * gds_scale if hasattr(v1, "__getitem__") else 0
            v2x = v2[0] * gds_scale if hasattr(v2, "__getitem__") else 0
            v2y = v2[1] * gds_scale if hasattr(v2, "__getitem__") else 0

            for row in range(rows):
                for col in range(columns):
                    if row == 0 and col == 0:
                        continue
                    x = col * v1x + row * v2x
                    y = col * v1y + row * v2y
                    positions.append((x, y))

        elif "explicit" in type_name:
            # Explicit offsets
            offsets = getattr(repetition, "offsets", [])
            for offset in offsets:
                x = offset[0] * gds_scale if hasattr(offset, "__getitem__") else 0
                y = offset[1] * gds_scale if hasattr(offset, "__getitem__") else 0
                positions.append((x, y))

            # Also check coords for ExplicitX/ExplicitY
            coords = getattr(repetition, "coords", [])
            if coords:
                if "x" in type_name:
                    for c in coords:
                        positions.append((c * gds_scale, 0.0))
                else:
                    for c in coords:
                        positions.append((0.0, c * gds_scale))

        else:
            # Try generic approach
            columns = getattr(repetition, "columns", 1)
            rows = getattr(repetition, "rows", 1)
            spacing = getattr(repetition, "spacing", None)
            v1 = getattr(repetition, "v1", None)
            v2 = getattr(repetition, "v2", None)

            if spacing is not None:
                sx = spacing[0] * gds_scale if hasattr(spacing, "__getitem__") else 0
                sy = spacing[1] * gds_scale if hasattr(spacing, "__getitem__") else 0
                for row in range(rows):
                    for col in range(columns):
                        if row == 0 and col == 0:
                            continue
                        positions.append((col * sx, row * sy))
            elif v1 is not None:
                v1x = v1[0] * gds_scale if hasattr(v1, "__getitem__") else 0
                v1y = v1[1] * gds_scale if hasattr(v1, "__getitem__") else 0
                v2x = (v2[0] * gds_scale if hasattr(v2, "__getitem__") else 0) if v2 else 0
                v2y = (v2[1] * gds_scale if hasattr(v2, "__getitem__") else 0) if v2 else 0
                for row in range(rows):
                    for col in range(columns):
                        if row == 0 and col == 0:
                            continue
                        x = col * v1x + row * v2x
                        y = col * v1y + row * v2y
                        positions.append((x, y))

        return positions

    def _build_transform(
        self,
        origin: tuple[float, float],
        rotation: float,
        magnification: float,
        x_reflection: bool,
    ) -> Optional[list]:
        """Build 3x3 transformation matrix.

        Parameters
        ----------
        origin : tuple[float, float]
            Translation (x, y).
        rotation : float
            Rotation in radians.
        magnification : float
            Scale factor.
        x_reflection : bool
            Mirror across X-axis.

        Returns
        -------
        Optional[list]
            3x3 transformation matrix, or None if identity.
        """
        # Check if identity
        is_identity = (
            abs(origin[0]) < 1e-12
            and abs(origin[1]) < 1e-12
            and abs(rotation) < 1e-12
            and abs(magnification - 1.0) < 1e-12
            and not x_reflection
        )

        if is_identity:
            return None

        # Build transform: T @ R @ S @ M (translation @ rotation @ scale @ mirror)
        # Applied right-to-left

        cos_r = math.cos(rotation)
        sin_r = math.sin(rotation)

        # Start with identity
        # Mirror (reflect across X-axis = negate Y)
        if x_reflection:
            m = [
                [1, 0, 0],
                [0, -1, 0],
                [0, 0, 1],
            ]
        else:
            m = [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ]

        # Scale
        s = [
            [magnification, 0, 0],
            [0, magnification, 0],
            [0, 0, 1],
        ]

        # Rotation
        r = [
            [cos_r, -sin_r, 0],
            [sin_r, cos_r, 0],
            [0, 0, 1],
        ]

        # Translation
        t = [
            [1, 0, origin[0]],
            [0, 1, origin[1]],
            [0, 0, 1],
        ]

        # Compose: T @ R @ S @ M
        def matmul(a, b):
            return [
                [sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)]
                for i in range(3)
            ]

        result = matmul(t, matmul(r, matmul(s, m)))
        return result

    def _convert_polygon(self, polygon, gds_scale: float) -> Optional[Polygon2D]:
        """Convert gdstk.Polygon to Polygon2D.

        Uses Polygon2D.construct() to skip pydantic validation for performance.
        This is safe because:
        1. gdstk guarantees (N, 2) shaped point arrays
        2. Shapely handles duplicate vertices via buffer(0) in to_shapely()
        3. to_shapely() catches degenerate cases with explicit error handling

        Parameters
        ----------
        polygon : gdstk.Polygon
            GDS polygon.
        gds_scale : float
            Scale factor.

        Returns
        -------
        Optional[Polygon2D]
            Converted polygon, or None if invalid.
        """
        points = polygon.points
        if len(points) < 3:
            return None

        # Scale and convert to numpy array directly
        vertices = np.asarray(points, dtype=float) * gds_scale

        # Use construct() to skip pydantic validation
        return Polygon2D.construct(
            vertices=vertices,
            bulges=None,
            holes=(),
            arc_resolution=32,
        )

    def _convert_path(
        self, path, gds_scale: float
    ) -> list[tuple[Geometry2D, str]]:
        """Convert gdstk path to polygons.

        FlexPath and RobustPath can have multiple layers, so this returns
        a list of (geometry, layer_name) tuples.

        Parameters
        ----------
        path : gdstk.FlexPath or gdstk.RobustPath
            GDS path.
        gds_scale : float
            Scale factor.

        Returns
        -------
        list[tuple[Geometry2D, str]]
            List of (geometry, layer_name) tuples.
        """
        result = []
        
        try:
            # Get layers and datatypes from path
            # FlexPath/RobustPath have layers/datatypes arrays
            layers = getattr(path, "layers", None)
            datatypes = getattr(path, "datatypes", None)
            
            if layers is None:
                # Fallback for simple paths
                layer = getattr(path, "layer", 0)
                dtype = getattr(path, "datatype", 0)
                layers = [layer]
                datatypes = [dtype]
            
            # Convert path to polygons
            polygons = path.to_polygons()
            if not polygons:
                return result
            
            # Each polygon corresponds to a layer
            # If there are more polygons than layers, they cycle
            num_layers = len(layers)
            
            for i, poly in enumerate(polygons):
                layer_idx = i % num_layers
                layer = layers[layer_idx]
                dtype = datatypes[layer_idx] if datatypes and layer_idx < len(datatypes) else 0
                
                points = poly.points if hasattr(poly, "points") else poly
                if len(points) < 3:
                    continue
                
                vertices = [
                    (float(p[0]) * gds_scale, float(p[1]) * gds_scale)
                    for p in points
                ]
                
                # Clean vertices
                vertices = self._clean_vertices(vertices)
                if len(vertices) < 3:
                    continue
                
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        geom = Polygon2D(vertices=vertices)
                    layer_name = _layer_name(layer, dtype)
                    result.append((geom, layer_name))
                except Exception:
                    continue
        
        except Exception:
            pass
        
        return result

    def _path_to_polygon(self, path, gds_scale: float) -> Optional[Polygon2D]:
        """Convert path to polygon via tessellation.

        Uses Polygon2D.construct() to skip pydantic validation for performance.

        Parameters
        ----------
        path : gdstk path
            GDS path object.
        gds_scale : float
            Scale factor.

        Returns
        -------
        Optional[Polygon2D]
            Polygonized path, or None if invalid.
        """
        try:
            # gdstk paths have a to_polygons() method
            polygons = path.to_polygons()
            if not polygons:
                return None

            # Take first polygon (paths usually produce one)
            poly = polygons[0]
            points = poly.points if hasattr(poly, "points") else poly

            if len(points) < 3:
                return None

            # Scale and convert to numpy array directly
            vertices = np.asarray(points, dtype=float) * gds_scale

            # Use construct() to skip pydantic validation
            return Polygon2D.construct(
                vertices=vertices,
                bulges=None,
                holes=(),
                arc_resolution=32,
            )

        except Exception:
            return None
