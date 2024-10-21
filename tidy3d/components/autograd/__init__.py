import autograd.numpy as np
from autograd.extend import VJPNode
from autograd.numpy.numpy_vjps import register_notrace

from .boxes import TidyArrayBox
from .functions import interpn
from .types import (
    AutogradFieldMap,
    AutogradTraced,
    TracedCoordinate,
    TracedFloat,
    TracedSize,
    TracedSize1D,
    TracedVertices,
)
from .utils import get_static, is_tidy_box, split_list

register_notrace(VJPNode, np.full_like)
__all__ = [
    "TidyArrayBox",
    "TracedFloat",
    "TracedSize1D",
    "TracedSize",
    "TracedCoordinate",
    "TracedVertices",
    "AutogradTraced",
    "AutogradFieldMap",
    "get_static",
    "interpn",
    "split_list",
    "is_tidy_box",
]
