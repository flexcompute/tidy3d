# imports from tidy3d.plugins.invdes as tdi

from . import utils
from .design import (
    FixedIterationTerminationSpec,
    InverseDesign,  # , InverseDesignMulti
)
from .dummy_simulation import DummySimulation
from .initialization import (
    CustomInitializationSpec,
    RandomInitializationSpec,
    UniformInitializationSpec,
)
from .objective import EMObjective, MaxSpec, MultiObjective, PenaltyObjective, SumSpec
from .optimizer import GradientAscentOptimizer, GradientAscentOptimizerState

# from .optimizer import AdamOptimizer
from .penalty import ErosionDilationPenalty
from .region import DesignRegion, DesignRegion2

# from .region import TopologyDesignRegion
from .result import Result
from .transformation import FilterProject

__all__ = (
    "InverseDesign",
    # "InverseDesignMulti",
    "FilterProject",
    "ErosionDilationPenalty",
    "DummySimulation",
    "EMObjective",
    "PenaltyObjective",
    "MultiObjective",
    "DesignRegion",
    "DesignRegion2",
    "GradientAscentOptimizer",
    "GradientAscentOptimizerState",
    "MaxSpec",
    "SumSpec",
    # "AdamOptimizer",
    "InverseDesign",
    "FixedIterationTerminationSpec",
    "Result",
    "RandomInitializationSpec",
    "UniformInitializationSpec",
    "CustomInitializationSpec",
    "utils",
)
