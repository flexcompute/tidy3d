import pytest
import pydantic as pd
import numpy as np

import tidy3d as td

from tidy3d.components.heat_spec import FluidSpec, SolidSpec
from tidy3d.components.heat.source import UniformHeatSource
from tidy3d.components.heat.boundary import TemperatureBC, HeatFluxBC, ConvectionBC, HeatBoundarySpec
from tidy3d.components.bc_placement import (
    StructureBoundary,
    StructureStructureInterface,
    SimulationBoundary,
    StructureSimulationBoundary,
    MediumMediumInterface,
)
from tidy3d.components.heat.grid import UniformHeatGrid
from tidy3d.components.heat.simulation import HeatSimulation
from tidy3d.components.heat.sim_data import HeatSimulationData


def make_heat_mediums():
    fluid_medium = td.Medium(
        permittivity=3,
        heat_spec=FluidSpec(),
        name="fluid_medium",
    )
    solid_medium = td.Medium(
        permittivity=5,
        conductivity=0.01,
        heat_spec=SolidSpec(
            capacity=2,
            conductivity=3,
        ),
        name="solid_medium",
    )

    return fluid_medium, solid_medium


def test_heat_medium():
    _, solid_medium = make_heat_mediums()

    with pytest.raises(pd.ValidationError):
        _ = solid_medium.heat_spec.updated_copy(capacity=-1)

    with pytest.raises(pd.ValidationError):
        _ = solid_medium.heat_spec.updated_copy(conductivity=-1)


def make_heat_structures():
    fluid_medium, solid_medium = make_heat_mediums()

    box = td.Box(center=(0, 0, 0), size=(1, 1, 1))

    fluid_structure = td.Structure(
        geometry=box,
        medium=fluid_medium,
        name="fluid_structure",
    )

    solid_structure = td.Structure(
        geometry=box.updated_copy(center=(1, 1, 1)),
        medium=solid_medium,
        name="solid_structure",
    )

    return fluid_structure, solid_structure


def test_heat_structures():
    _, _ = make_heat_structures()


def make_heat_bcs():
    bc_temp = TemperatureBC(temperature=300)
    bc_flux = HeatFluxBC(flux=20)
    bc_conv = ConvectionBC(ambient_temperature=400, transfer_coeff=0.2)

    return bc_temp, bc_flux, bc_conv


def test_heat_bcs():
    bc_temp, bc_flux, bc_conv = make_heat_bcs()

    with pytest.raises(pd.ValidationError):
        _ = TemperatureBC(temperature=-10)

    with pytest.raises(pd.ValidationError):
        _ = ConvectionBC(ambient_temperature=-400, transfer_coeff=0.2)

    with pytest.raises(pd.ValidationError):
        _ = ConvectionBC(ambient_temperature=400, transfer_coeff=-0.2)


def make_grid_spec():
    return UniformHeatGrid(dl=0.1, min_edges_per_circumference=5, min_edges_per_side=3)


def test_grid_spec():
    grid_spec = make_grid_spec()
    with pytest.raises(pd.ValidationError):
        _ = grid_spec.updated_copy(dl=0)
        _ = grid_spec.updated_copy(min_edges_per_circumference=-1)
        _ = grid_spec.updated_copy(min_edges_per_side=-1)


def make_heat_source():
    return UniformHeatSource(structures=["solid_structure"], rate=100)


def test_heat_source():
    _ = make_heat_source()


def make_heat_sim():
    fluid_medium, solid_medium = make_heat_mediums()
    fluid_structure, solid_structure = make_heat_structures()
    bc_temp, bc_flux, bc_conv = make_heat_bcs()
    heat_source = make_heat_source()

    pl1 = HeatBoundarySpec(condition=bc_conv, placement=MediumMediumInterface(mediums=["fluid_medium", "solid_medium"]))
    pl2 = HeatBoundarySpec(condition=bc_flux, placement=StructureBoundary(structure="solid_structure"))
    pl3 = HeatBoundarySpec(condition=bc_flux, placement=StructureStructureInterface(structures=["fluid_structure", "solid_structure"]))
    pl4 = HeatBoundarySpec(condition=bc_temp, placement=SimulationBoundary())
    pl5 = HeatBoundarySpec(condition=bc_temp, placement=StructureSimulationBoundary(structure="fluid_structure"))

    grid_spec = make_grid_spec()

    heat_sim = HeatSimulation(
        scene=td.Scene(
            center=(0, 0, 0),
            size=(2, 3, 3),
            medium=fluid_medium,
            structures=[fluid_structure, solid_structure],
        ),
        boundary_specs=[pl1, pl2, pl3, pl4, pl5],
        grid_spec=grid_spec,
        heat_domain=td.Box(center=(0, 0, 0), size=(2, 2, 2)),
        heat_sources=[heat_source],
    )

    return heat_sim


def test_heat_sim():
    bc_temp, bc_flux, bc_conv = make_heat_bcs()
    heat_sim = make_heat_sim()

    _ = heat_sim.plot(x=0)

    # wrong names given
    for pl in [
        HeatBoundarySpec(condition=bc_temp, placement=MediumMediumInterface(mediums=["badname", "fluid_medium"])),
        HeatBoundarySpec(condition=bc_flux, placement=StructureBoundary(structure="no_box")),
        HeatBoundarySpec(condition=bc_conv, placement=StructureStructureInterface(structures=["no_box", "solid_structure"])),
        HeatBoundarySpec(condition=bc_temp, placement=StructureSimulationBoundary(structure="no_mesh")),
    ]:
        with pytest.raises(pd.ValidationError):
            _ = heat_sim.updated_copy(boundary_specs=[pl])

    with pytest.raises(pd.ValidationError):
        _ = heat_sim.updated_copy(heat_sources=[UniformHeatSource(structures=["noname"])], rate=-10)

    # test unsupported yet geometries
    vertices = np.array([(0, 0), (1, 0), (1, 1)])
    p = td.PolySlab(vertices=vertices, axis=2, slab_bounds=(-1, 1))
    _, structure = make_heat_structures()
    structure = structure.updated_copy(geometry=p)

    with pytest.raises(pd.ValidationError):
        _ = heat_sim.updated_copy(structures=[structure])

    # test unsupported yet sezi dimension domains
    with pytest.raises(pd.ValidationError):
        _ = heat_sim.updated_copy(heat_domain=td.Box(center=(0, 0, 0), size=(0, 2, 2)))

    with pytest.raises(pd.ValidationError):
        _ = heat_sim.updated_copy(heat_domain=td.Box(center=(0, 0, 0), size=(1, 0, 0)))

    with pytest.raises(pd.ValidationError):
        _ = heat_sim.updated_copy(
            heat_domain=None, scene=heat_sim.scene.updated_copy(size=(1, 0, 1))
        )


def make_heat_sim_data():
    heat_sim = make_heat_sim()

    nx, ny, nz = 9, 6, 5
    x = np.linspace(-1, 1, nx)
    y = np.linspace(-2, 2, ny)
    z = np.linspace(-3, 3, nz)
    T = np.random.default_rng().uniform(300, 350, (nx, ny, nz))
    coords = dict(x=x, y=y, z=z)
    temperature_field = td.SpatialDataArray(T, coords=coords)

    heat_sim_data = HeatSimulationData(
        heat_simulation=heat_sim,
        temperature_data=temperature_field,
    )

    return heat_sim_data


def test_sim_data():
    heat_sim_data = make_heat_sim_data()

    _ = heat_sim_data.perturbed_mediums_scene()
