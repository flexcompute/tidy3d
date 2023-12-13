import pytest
import pydantic.v1 as pd
import numpy as np

import tidy3d as td
from tidy3d.components.bc_placement import (
    StructureBoundary,
    StructureStructureInterface,
    SimulationBoundary,
    StructureSimulationBoundary,
    MediumMediumInterface,
)


def test_bc_placement():
    _ = StructureBoundary(structure="box")
    _ = SimulationBoundary()
    _ = StructureSimulationBoundary(structure="box")
    _ = StructureStructureInterface(structures=["box", "sphere"])
    _ = MediumMediumInterface(mediums=["dieletric", "metal"])
