"""Compatibility facade for historical grid_spec imports."""

from __future__ import annotations

from typing import Literal as _Literal

from pydantic import NonNegativeFloat as _NonNegativeFloat
from pydantic import NonNegativeInt as _NonNegativeInt
from pydantic import PositiveFloat as _PositiveFloat

from tidy3d.compat import Self as _Self
from tidy3d.components.geometry.base import Box as _Box
from tidy3d.components.grid.corner_finder import CornerFinderSpec
from tidy3d.components.grid.grid import Coords1D as _Coords1D
from tidy3d.components.grid.grid import Grid as _Grid
from tidy3d.components.grid.mesher import MesherType as _MesherType
from tidy3d.components.lumped_element import LumpedElementType as _LumpedElementType
from tidy3d.components.source.utils import SourceType as _SourceType
from tidy3d.components.structure import MeshOverrideStructure as _MeshOverrideStructure
from tidy3d.components.structure import Structure as _Structure
from tidy3d.components.structure import StructureType as _StructureType
from tidy3d.components.types import ArrayFloat1D as _ArrayFloat1D
from tidy3d.components.types import ArrayFloat2D as _ArrayFloat2D
from tidy3d.components.types import Axis as _Axis
from tidy3d.components.types import Coordinate as _Coordinate
from tidy3d.components.types import CoordinateOptional as _CoordinateOptional
from tidy3d.components.types import PriorityMode as _PriorityMode
from tidy3d.components.types import Shapely as _Shapely
from tidy3d.components.types import Symmetry as _Symmetry

from .constants import (
    DEFAULT_REFINEMENT_FACTOR,
    DL_MIN_FROM_GAPS_FRACTION,
    GAP_MESHING_TOL,
    GAP_REFINEMENT_WARNING_THRESH,
    INPLANE_OVERRIDE_UNION_MAX_ITERS,
    MIN_GRID_SPACING,
    MIN_STEP_BOUND_SCALE,
    UNITS_HELP_URL,
    CornersAndConvexity,
    _GeneratedGridSizeError,
)
from .grid_1d import (
    AbstractAutoGrid,
    AutoGrid,
    CustomGrid,
    CustomGridBoundaries,
    GridSpec1d,
    GridType,
    QuasiUniformGrid,
    UniformGrid,
)
from .grid_1d import auto as _auto
from .grid_1d import base as _base
from .grid_1d import manual as _manual
from .refinement import GridRefinement, LayerRefinementSpec
from .refinement import construction as _construction
from .refinement import edges as _edges
from .refinement import gaps as _gaps
from .refinement import inplane as _inplane
from .refinement import model as _refinement_model
from .refinement import properties as _properties
from .spec import GridSpec
from .spec import entities as _entities
from .spec import factories as _factories
from .spec import generation as _generation
from .spec import localization as _localization
from .spec import localization_helpers as _localization_helpers
from .spec import model as _model
from .spec import sizing as _sizing

# The provider functions are bound to the canonical models after import. Register their annotation
# dependencies here, after both models exist, so ``typing.get_type_hints`` remains usable without
# introducing circular implementation imports.
_base.Axis = _Axis
_base.Box = _Box
_base.CoordinateOptional = _CoordinateOptional
_base.Coords1D = _Coords1D
_base.NonNegativeInt = _NonNegativeInt
_base.PositiveFloat = _PositiveFloat
_base.Structure = _Structure
_base.StructureType = _StructureType
_base.Symmetry = _Symmetry
_auto.Axis = _Axis
_auto.CoordinateOptional = _CoordinateOptional
_auto.Coords1D = _Coords1D
_auto.StructureType = _StructureType
_auto.Symmetry = _Symmetry
_manual.Axis = _Axis
_manual.Structure = _Structure
_manual.StructureType = _StructureType
_refinement_model.Coordinate = _Coordinate
_refinement_model.CoordinateOptional = _CoordinateOptional
_refinement_model.Self = _Self
_localization_helpers.CoordinateOptional = _CoordinateOptional
_localization_helpers.LayerRefinementSpec = LayerRefinementSpec
_model.SourceType = _SourceType
_entities.CornersAndConvexity = CornersAndConvexity
_entities.CoordinateOptional = _CoordinateOptional
_entities.GridSpec = GridSpec
_entities.LumpedElementType = _LumpedElementType
_entities.MeshOverrideStructure = _MeshOverrideStructure
_entities.PositiveFloat = _PositiveFloat
_entities.PriorityMode = _PriorityMode
_entities.Shapely = _Shapely
_entities.StructureType = _StructureType
_sizing.GridSpec = GridSpec
_sizing.LumpedElementType = _LumpedElementType
_sizing.Shapely = _Shapely
_sizing.StructureType = _StructureType
_generation.GridSpec = GridSpec
_generation.CoordinateOptional = _CoordinateOptional
_generation.LumpedElementType = _LumpedElementType
_generation.MeshOverrideStructure = _MeshOverrideStructure
_generation.NonNegativeInt = _NonNegativeInt
_generation.PositiveFloat = _PositiveFloat
_generation.PriorityMode = _PriorityMode
_generation.Shapely = _Shapely
_generation.SourceType = _SourceType
_generation.StructureType = _StructureType
_generation.Symmetry = _Symmetry
_localization.Box = _Box
_localization.GridSpec = GridSpec
_localization.Self = _Self
_factories.Grid = _Grid
_factories.GridSpec = GridSpec
_factories.LayerRefinementSpec = LayerRefinementSpec
_factories.MesherType = _MesherType
_factories.NonNegativeFloat = _NonNegativeFloat
_factories.PositiveFloat = _PositiveFloat
_factories.StructureType = _StructureType
_factories.CoordinateOptional = _CoordinateOptional
_construction.Axis = _Axis
_construction.Coordinate = _Coordinate
_construction.LayerRefinementSpec = LayerRefinementSpec
_construction.Literal = _Literal
_construction.NonNegativeInt = _NonNegativeInt
_construction.PositiveFloat = _PositiveFloat
_construction.Self = _Self
_construction.Structure = _Structure
_edges.ArrayFloat2D = _ArrayFloat2D
_edges.CornersAndConvexity = CornersAndConvexity
_edges.CoordinateOptional = _CoordinateOptional
_edges.Grid = _Grid
_edges.LayerRefinementSpec = LayerRefinementSpec
_edges.Shapely = _Shapely
_edges.Structure = _Structure
_gaps.ArrayFloat1D = _ArrayFloat1D
_gaps.ArrayFloat2D = _ArrayFloat2D
_gaps.CoordinateOptional = _CoordinateOptional
_gaps.Grid = _Grid
_gaps.LayerRefinementSpec = LayerRefinementSpec
_gaps.Shapely = _Shapely
_inplane.ArrayFloat1D = _ArrayFloat1D
_inplane.ArrayFloat2D = _ArrayFloat2D
_inplane.CoordinateOptional = _CoordinateOptional
_inplane.CornersAndConvexity = CornersAndConvexity
_inplane.LayerRefinementSpec = LayerRefinementSpec
_inplane.Shapely = _Shapely
_inplane.Structure = _Structure
_properties.LayerRefinementSpec = LayerRefinementSpec

del (
    _ArrayFloat1D,
    _ArrayFloat2D,
    _auto,
    _Axis,
    _base,
    _Box,
    _construction,
    _Coordinate,
    _CoordinateOptional,
    _Coords1D,
    _edges,
    _entities,
    _factories,
    _gaps,
    _generation,
    _Grid,
    _inplane,
    _localization,
    _localization_helpers,
    _Literal,
    _LumpedElementType,
    _MeshOverrideStructure,
    _MesherType,
    _model,
    _NonNegativeFloat,
    _NonNegativeInt,
    _PositiveFloat,
    _PriorityMode,
    _properties,
    _refinement_model,
    _Self,
    _Shapely,
    _sizing,
    _SourceType,
    _Structure,
    _StructureType,
    _Symmetry,
    _manual,
)

__all__ = [
    "DEFAULT_REFINEMENT_FACTOR",
    "DL_MIN_FROM_GAPS_FRACTION",
    "GAP_MESHING_TOL",
    "GAP_REFINEMENT_WARNING_THRESH",
    "INPLANE_OVERRIDE_UNION_MAX_ITERS",
    "MIN_GRID_SPACING",
    "MIN_STEP_BOUND_SCALE",
    "UNITS_HELP_URL",
    "AbstractAutoGrid",
    "AutoGrid",
    "CornerFinderSpec",
    "CornersAndConvexity",
    "CustomGrid",
    "CustomGridBoundaries",
    "GridRefinement",
    "GridSpec",
    "GridSpec1d",
    "GridType",
    "LayerRefinementSpec",
    "QuasiUniformGrid",
    "UniformGrid",
    "_GeneratedGridSizeError",
]
