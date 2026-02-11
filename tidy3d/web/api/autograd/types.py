from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, NamedTuple, Optional, Union

if TYPE_CHECKING:
    from tidy3d.components.autograd import AutogradFieldMap
    from tidy3d.components.geometry.utils import GeometryType
    from tidy3d.components.medium import MediumType
    from tidy3d.components.simulation import Simulation


@dataclass
class CustomVJPConfig:
    structure: Union[int, type[GeometryType], type[MediumType]]
    """Index for structure to replace vjp or specification of geometry or medium type. If a type is provided,
    the custom vjp will be applied to all structures in the simulation with the geometry or medium type.
    """

    compute_derivatives: Callable
    """Function for computing the targeted vjp value. The function should accept the geometry or medium in the
    structure depending on if this is a geometry or medium path (see path_key) as the first argument. The second
    argument should be named derivative_info and accept a DerivativeInfo object that contains important for computing
    the gradient. The function should return a dict object that maps the path to the computed gradient value.
    """

    path_key: Optional[tuple[str, ...]] = None
    """Path key corresponding to the vjp. For example, this could be ('geometry', 'radius') if you are targeting
    the radius parameter in the given structure geometry. It can also target the medium by specifying medium first
    (i.e. - ('medium', 'permittivity') will target the permittivity variable in the structure's medium). If not
    specified or set to None, the supplied function applies for all possible vjp paths.
    """


CustomVJPSpec = Union[
    CustomVJPConfig,
    dict[str, CustomVJPConfig],
    Sequence[CustomVJPConfig],
    dict[str, Sequence[CustomVJPConfig]],
    Sequence[Sequence[CustomVJPConfig]],
]


class SetupRunResult(NamedTuple):
    sim_fields: AutogradFieldMap
    simulation: Simulation
