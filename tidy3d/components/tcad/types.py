"""File containing classes required for the setup of a DEVSIM case."""

from tidy3d.components.tcad.bandgap import SlotboomNarrowingBandGap
from tidy3d.components.tcad.generation_recombination import (
    AugerRecombination,
    RadiativeRecombination,
    ShockleyReedHallRecombination,
)
from tidy3d.components.tcad.mobility import CaugheyThomasMobility
from tidy3d.components.types import Union

MobilityModelTypes = Union[CaugheyThomasMobility]
RecombinationModelTypes = Union[
    AugerRecombination, RadiativeRecombination, ShockleyReedHallRecombination
]
BandGapModelTypes = Union[SlotboomNarrowingBandGap]
