from __future__ import annotations

import tidy3d.plugins.expressions
from tidy3d.components.geometry.contour_conversion import smooth_polygon_vertices

from . import utils
from .base import InvdesBaseModel
from .contours import curvature_penalty
from .design import AbstractInverseDesign, InverseDesign, InverseDesignMulti, InverseDesignType
from .initialization import (
    AbstractInitializationSpec,
    CustomInitializationSpec,
    InitializationSpecType,
    RandomInitializationSpec,
    UniformInitializationSpec,
)
from .optimizer import AbstractOptimizer, AdamOptimizer
from .penalty import AbstractPenalty, ErosionDilationPenalty, PenaltyType
from .polyslab_set import PolySlabSet
from .region import DesignRegion, DesignRegionType, TopologyDesignRegion
from .result import InverseDesignResult
from .self_intersection import SelfIntersectionStatus
from .transformation import AbstractTransformation, FilterProject, TransformationType

rebuild_context_namespace = tidy3d.plugins.expressions._local_vars.copy()
AbstractInverseDesign.model_rebuild(_types_namespace=rebuild_context_namespace)
InverseDesign.model_rebuild(_types_namespace=rebuild_context_namespace)
InverseDesignMulti.model_rebuild(_types_namespace=rebuild_context_namespace)

__all__ = (
    "AbstractInitializationSpec",
    "AbstractInverseDesign",
    "AbstractOptimizer",
    "AbstractPenalty",
    "AbstractTransformation",
    "AdamOptimizer",
    "CustomInitializationSpec",
    "DesignRegion",
    "DesignRegionType",
    "ErosionDilationPenalty",
    "FilterProject",
    "InitializationSpecType",
    "InvdesBaseModel",
    "InverseDesign",
    "InverseDesignMulti",
    "InverseDesignResult",
    "InverseDesignType",
    "PenaltyType",
    "PolySlabSet",
    "RandomInitializationSpec",
    "SelfIntersectionStatus",
    "TopologyDesignRegion",
    "TransformationType",
    "UniformInitializationSpec",
    "curvature_penalty",
    "smooth_polygon_vertices",
    "utils",
)
