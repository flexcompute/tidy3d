from tidy3d.components.bc_placement import (
    MediumMediumInterface,
    SimulationBoundary,
    StructureBoundary,
    StructureSimulationBoundary,
    StructureStructureInterface,
)


def test_bc_placement():
    _ = StructureBoundary(structure="box")
    _ = SimulationBoundary()
    _ = StructureSimulationBoundary(structure="box")
    _ = StructureStructureInterface(structures=["box", "sphere"])
    _ = MediumMediumInterface(mediums=["dieletric", "metal"])
