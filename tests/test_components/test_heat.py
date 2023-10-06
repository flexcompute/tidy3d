import pytest
import pydantic.v1 as pd
import numpy as np
from matplotlib import pyplot as plt

import tidy3d as td

from tidy3d import FluidSpec, SolidSpec
from tidy3d import UniformHeatSource
from tidy3d import (
    TemperatureBC,
    HeatFluxBC,
    ConvectionBC,
    HeatBoundarySpec,
)
from tidy3d import (
    StructureBoundary,
    StructureStructureInterface,
    SimulationBoundary,
    StructureSimulationBoundary,
    MediumMediumInterface,
)
from tidy3d import UniformUnstructuredGrid, DistanceUnstructuredGrid
from tidy3d import HeatSimulation
from tidy3d import HeatSimulationData
from tidy3d import TemperatureMonitor
from tidy3d import TemperatureData

from tidy3d.exceptions import DataError
from ..utils import STL_GEO, assert_log_level, log_capture


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


def make_heat_mnt():
    return TemperatureMonitor(size=(1, 2, 3), name="test")


def test_heat_mnt():
    temp_mnt = make_heat_mnt()

    with pytest.raises(pd.ValidationError):
        _ = temp_mnt.updated_copy(name=None)

    with pytest.raises(pd.ValidationError):
        _ = temp_mnt.updated_copy(size=(-1, 2, 3))


def make_heat_mnt_data():
    temp_mnt = make_heat_mnt()

    nx, ny, nz = 9, 6, 5
    x = np.linspace(-1, 1, nx)
    y = np.linspace(-2, 2, ny)
    z = np.linspace(-3, 3, nz)
    T = np.random.default_rng().uniform(300, 350, (nx, ny, nz))
    coords = dict(x=x, y=y, z=z)
    temperature_field = td.SpatialDataArray(T, coords=coords)

    return TemperatureData(monitor=temp_mnt, temperature=temperature_field)


def test_heat_mnt_data():
    _ = make_heat_mnt_data


def make_uniform_grid_spec():
    return UniformUnstructuredGrid(dl=0.1, min_edges_per_circumference=5, min_edges_per_side=3)


def make_distance_grid_spec():
    return DistanceUnstructuredGrid(
        dl_interface=0.1, dl_bulk=1, distance_interface=1, distance_bulk=2
    )


def test_grid_spec():
    grid_spec = make_uniform_grid_spec()
    with pytest.raises(pd.ValidationError):
        _ = grid_spec.updated_copy(dl=0)
    with pytest.raises(pd.ValidationError):
        _ = grid_spec.updated_copy(min_edges_per_circumference=-1)
    with pytest.raises(pd.ValidationError):
        _ = grid_spec.updated_copy(min_edges_per_side=-1)

    grid_spec = make_distance_grid_spec()
    with pytest.raises(pd.ValidationError):
        grid_spec.updated_copy(dl_interface=-1)
    with pytest.raises(pd.ValidationError):
        grid_spec.updated_copy(distance_interface=2, distance_bulk=1)


def make_heat_source():
    return UniformHeatSource(structures=["solid_structure"], rate=100)


def test_heat_source():
    _ = make_heat_source()


def make_heat_sim():
    fluid_medium, solid_medium = make_heat_mediums()
    fluid_structure, solid_structure = make_heat_structures()
    bc_temp, bc_flux, bc_conv = make_heat_bcs()
    heat_source = make_heat_source()

    pl1 = HeatBoundarySpec(
        condition=bc_conv, placement=MediumMediumInterface(mediums=["fluid_medium", "solid_medium"])
    )
    pl2 = HeatBoundarySpec(
        condition=bc_flux, placement=StructureBoundary(structure="solid_structure")
    )
    pl3 = HeatBoundarySpec(
        condition=bc_flux,
        placement=StructureStructureInterface(structures=["fluid_structure", "solid_structure"]),
    )
    pl4 = HeatBoundarySpec(condition=bc_temp, placement=SimulationBoundary())
    pl5 = HeatBoundarySpec(
        condition=bc_temp, placement=StructureSimulationBoundary(structure="fluid_structure")
    )

    grid_spec = make_uniform_grid_spec()

    temp_mnt = make_heat_mnt()

    heat_sim = HeatSimulation(
        medium=fluid_medium,
        structures=[fluid_structure, solid_structure],
        center=(0, 0, 0),
        size=(2, 2, 2),
        boundary_spec=[pl1, pl2, pl3, pl4, pl5],
        grid_spec=grid_spec,
        sources=[heat_source],
        monitors=[temp_mnt],
    )

    return heat_sim


def test_heat_sim():
    bc_temp, bc_flux, bc_conv = make_heat_bcs()
    heat_sim = make_heat_sim()

    _ = heat_sim.plot(x=0)

    # wrong names given
    for pl in [
        HeatBoundarySpec(
            condition=bc_temp, placement=MediumMediumInterface(mediums=["badname", "fluid_medium"])
        ),
        HeatBoundarySpec(condition=bc_flux, placement=StructureBoundary(structure="no_box")),
        HeatBoundarySpec(
            condition=bc_conv,
            placement=StructureStructureInterface(structures=["no_box", "solid_structure"]),
        ),
        HeatBoundarySpec(
            condition=bc_temp, placement=StructureSimulationBoundary(structure="no_mesh")
        ),
    ]:
        with pytest.raises(pd.ValidationError):
            _ = heat_sim.updated_copy(boundary_spec=[pl])

    with pytest.raises(pd.ValidationError):
        _ = heat_sim.updated_copy(sources=[UniformHeatSource(structures=["noname"])], rate=-10)

    # polyslab support
    vertices = np.array([(0, 0), (1, 0), (1, 1)])
    p = td.PolySlab(vertices=vertices, axis=2, slab_bounds=(-1, 1))
    _, structure = make_heat_structures()
    structure = structure.updated_copy(geometry=p, name="polyslab")
    _ = heat_sim.updated_copy(structures=list(heat_sim.structures) + [structure])

    # test unsupported yet geometries
    structure = structure.updated_copy(geometry=STL_GEO, name="stl")
    with pytest.raises(pd.ValidationError):
        _ = heat_sim.updated_copy(structures=list(heat_sim.structures) + [structure])

    # test unsupported yet zero dimension domains
    with pytest.raises(pd.ValidationError):
        _ = heat_sim.updated_copy(center=(0, 0, 0), size=(0, 2, 2))

    with pytest.raises(pd.ValidationError):
        _ = heat_sim.updated_copy(center=(0, 0, 0), size=(1, 0, 0))

    temp_mnt = make_heat_mnt()

    with pytest.raises(pd.ValidationError):
        heat_sim.updated_copy(monitors=[temp_mnt, temp_mnt])

    _ = heat_sim.plot(x=0)
    plt.close()

    _ = heat_sim.plot_heat_conductivity(y=0)
    plt.close()

    heat_sim = heat_sim.updated_copy(symmetry=(0, 1, 1))
    _ = heat_sim.plot_heat_conductivity(z=0, colorbar="source")
    plt.close()

    with pytest.raises(pd.ValidationError):
        _ = heat_sim.updated_copy(symmetry=(-1, 0, 1))


@pytest.mark.parametrize("shift_amount, log_level", ((1, None), (2, "WARNING")))
def test_heat_sim_bounds(shift_amount, log_level, log_capture):
    """make sure bounds are working correctly"""

    # make sure all things are shifted to this central location
    CENTER_SHIFT = (-1.0, 1.0, 100.0)

    def place_box(center_offset):

        shifted_center = tuple(c + s for (c, s) in zip(center_offset, CENTER_SHIFT))

        _ = td.HeatSimulation(
            size=(1.5, 1.5, 1.5),
            center=CENTER_SHIFT,
            structures=[
                td.Structure(
                    geometry=td.Box(size=(1, 1, 1), center=shifted_center), medium=td.Medium()
                )
            ],
            grid_spec=td.UniformUnstructuredGrid(dl=0.1),
        )

    # create all permutations of squares being shifted 1, -1, or zero in all three directions
    bin_strings = [list(format(i, "03b")) for i in range(8)]
    bin_ints = [[int(b) for b in bin_string] for bin_string in bin_strings]
    bin_ints = np.array(bin_ints)
    bin_signs = 2 * (bin_ints - 0.5)

    # test all cases where box is shifted +/- 1 in x,y,z and still intersects
    for amp in bin_ints:
        for sign in bin_signs:
            center = shift_amount * amp * sign
            if np.sum(center) < 1e-12:
                continue
            place_box(tuple(center))
    assert_log_level(log_capture, log_level)


@pytest.mark.parametrize(
    "box_size,log_level",
    [
        ((1, 0.1, 0.1), "WARNING"),
        ((0.1, 1, 0.1), "WARNING"),
        ((0.1, 0.1, 1), "WARNING"),
    ],
)
def test_sim_structure_extent(log_capture, box_size, log_level):
    """Make sure we warn if structure extends exactly to simulation edges."""

    box = td.Structure(geometry=td.Box(size=box_size), medium=td.Medium(permittivity=2))
    _ = td.HeatSimulation(
        size=(1, 1, 1),
        structures=[box],
        grid_spec=td.UniformUnstructuredGrid(dl=0.1),
    )

    assert_log_level(log_capture, log_level)


def make_heat_sim_data():
    heat_sim = make_heat_sim()
    temp_data = make_heat_mnt_data()

    heat_sim_data = HeatSimulationData(
        simulation=heat_sim,
        data=[temp_data],
    )

    return heat_sim_data


def test_sim_data():
    heat_sim_data = make_heat_sim_data()
    _ = heat_sim_data.plot_field("test", z=0)
    plt.close()

    with pytest.raises(DataError):
        _ = heat_sim_data.plot_field("test3", x=0)

    with pytest.raises(pd.ValidationError):
        _ = heat_sim_data.updated_copy(data=[heat_sim_data.data[0]] * 2)

    temp_mnt = make_heat_mnt()
    temp_mnt = temp_mnt.updated_copy(name="test2")

    sim = heat_sim_data.simulation.updated_copy(monitors=[temp_mnt])

    with pytest.raises(pd.ValidationError):
        _ = heat_sim_data.updated_copy(simulation=sim)
