import pytest
import pydantic.v1 as pd
import numpy as np
from matplotlib import pyplot as plt

import tidy3d as td

from tidy3d import FluidSpec, SolidSpec, ConductorSpec, InsulatorSpec
from tidy3d import UniformHeatSource, HeatSource
from tidy3d import (
    TemperatureBC,
    HeatFluxBC,
    ConvectionBC,
    VoltageBC,
    CurrentBC,
    DeviceBoundarySpec,
)
from tidy3d import (
    StructureBoundary,
    StructureStructureInterface,
    SimulationBoundary,
    StructureSimulationBoundary,
    MediumMediumInterface,
)
from tidy3d import UniformUnstructuredGrid, DistanceUnstructuredGrid
from tidy3d import DeviceSimulation
from tidy3d import DeviceSimulationData
from tidy3d import VoltageMonitor, TemperatureMonitor
from tidy3d import VoltageData, TemperatureData
from tidy3d.exceptions import DataError

from ..utils import STL_GEO, assert_log_level, log_capture


def make_device_mediums():
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
        electric_spec=ConductorSpec(
            conductivity=1,
        ),
        name="solid_medium",
    )

    solid_noHeat = td.Medium(
        permittivity=5,
        conductivity=0.01,
        electric_spec=ConductorSpec(
            conductivity=1,
        ),
        name="solid_medium",
    )

    solid_noElect = td.Medium(
        permittivity=5,
        conductivity=0.01,
        heat_spec=SolidSpec(
            capacity=2,
            conductivity=3,
        ),
        name="solid_medium",
    )

    insulator_medium = td.Medium(
        permittivity=3,
        electric_spec=InsulatorSpec(),
        name="insulator_medium",
    )

    return fluid_medium, solid_medium, solid_noHeat, solid_noElect, insulator_medium


def test_device_medium():
    _, solid_medium, _, _, _ = make_device_mediums()

    with pytest.raises(pd.ValidationError):
        _ = solid_medium.heat_spec.updated_copy(capacity=-1)

    with pytest.raises(pd.ValidationError):
        _ = solid_medium.heat_spec.updated_copy(conductivity=-1)

    with pytest.raises(pd.ValidationError):
        _ = solid_medium.electric_spec.updated_copy(conductivity=-1)


def make_device_structures():
    fluid_medium, solid_medium, solid_noHeat, solid_noElect, insulator_med = make_device_mediums()

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

    solid_struct_noHeat = td.Structure(
        geometry=box.updated_copy(center=(1, 1, 1)),
        medium=solid_noHeat,
        name="solid_struct_noHeat",
    )

    solid_struct_noElect = td.Structure(
        geometry=box.updated_copy(center=(1, 1, 1)),
        medium=solid_noElect,
        name="solid_struct_noElect",
    )

    insulator_structure = td.Structure(
        geometry=box,
        medium=insulator_med,
        name="insulator_structure",
    )

    return (
        fluid_structure,
        solid_structure,
        solid_struct_noHeat,
        solid_struct_noElect,
        insulator_structure,
    )


def test_device_structures():
    _, _, _, _, _ = make_device_structures()


def make_device_bcs():
    bc_temp = TemperatureBC(temperature=300)
    bc_flux = HeatFluxBC(flux=20)
    bc_conv = ConvectionBC(ambient_temperature=400, transfer_coeff=0.2)
    bc_volt = VoltageBC(voltage=1)
    bc_current = CurrentBC(current_density=3e-1)

    return [bc_temp, bc_flux, bc_conv, bc_volt, bc_current]


def test_device_bcs():
    with pytest.raises(pd.ValidationError):
        _ = TemperatureBC(temperature=-10)

    with pytest.raises(pd.ValidationError):
        _ = ConvectionBC(ambient_temperature=-400, transfer_coeff=0.2)

    with pytest.raises(pd.ValidationError):
        _ = ConvectionBC(ambient_temperature=400, transfer_coeff=-0.2)

    with pytest.raises(pd.ValidationError):
        _ = VoltageBC(voltage=td.inf)

    with pytest.raises(pd.ValidationError):
        _ = CurrentBC(current_density=td.inf)


def make_device_mnts():
    temp_mnt1 = TemperatureMonitor(size=(1, 2, 3), name="test")
    temp_mnt2 = TemperatureMonitor(size=(1, 2, 3), name="tet", unstructured=True)
    temp_mnt3 = TemperatureMonitor(size=(1, 0, 3), name="tri", unstructured=True, conformal=True)
    temp_mnt4 = TemperatureMonitor(size=(1, 0, 3), name="empty", unstructured=True, conformal=False)

    volt_mnt1 = VoltageMonitor(size=(1, 2, 3), name="v_test")
    volt_mnt2 = VoltageMonitor(size=(1, 2, 3), name="v_tet", unstructured=True)
    volt_mnt3 = VoltageMonitor(size=(1, 0, 3), name="v_tri", unstructured=True, conformal=True)
    volt_mnt4 = VoltageMonitor(size=(1, 0, 3), name="v_empty", unstructured=True, conformal=False)

    return [temp_mnt1, temp_mnt2, temp_mnt3, temp_mnt4, volt_mnt1, volt_mnt2, volt_mnt3, volt_mnt4]


def test_device_mnt():
    """Checking for no name and negative size in monitors"""
    # NOTE: both Temperature and Voltage monitors derive from the same class. Since
    # both of these classes are empty we're actually checking the base class.

    mnts = make_device_mnts()
    temp_mnt = mnts[0]

    with pytest.raises(pd.ValidationError):
        _ = temp_mnt.updated_copy(name=None)

    with pytest.raises(pd.ValidationError):
        _ = temp_mnt.updated_copy(size=(-1, 2, 3))


def make_temperature_mnt_data():
    temp_mnt1, temp_mnt2, temp_mnt3, temp_mnt4, _, _, _, _ = make_device_mnts()

    nx, ny, nz = 9, 6, 5
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 2, ny)
    z = np.linspace(0, 3, nz)
    T = np.random.default_rng().uniform(300, 350, (nx, ny, nz))
    coords = dict(x=x, y=y, z=z)
    temperature_field = td.SpatialDataArray(T, coords=coords)

    mnt_data1 = TemperatureData(monitor=temp_mnt1, temperature=temperature_field)

    tet_grid_points = td.PointDataArray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dims=("index", "axis"),
    )

    tet_grid_cells = td.CellDataArray(
        [[0, 1, 2, 4], [1, 2, 3, 4]],
        dims=("cell_index", "vertex_index"),
    )

    tet_grid_values = td.IndexedDataArray(
        [1.0, 2.0, 3.0, 4.0, 5.0],
        dims=("index"),
        name="T",
    )

    tet_grid = td.TetrahedralGridDataset(
        points=tet_grid_points,
        cells=tet_grid_cells,
        values=tet_grid_values,
    )

    mnt_data2 = TemperatureData(monitor=temp_mnt2, temperature=tet_grid)

    tri_grid_points = td.PointDataArray(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        dims=("index", "axis"),
    )

    tri_grid_cells = td.CellDataArray(
        [[0, 1, 2], [1, 2, 3]],
        dims=("cell_index", "vertex_index"),
    )

    tri_grid_values = td.IndexedDataArray(
        [1.0, 2.0, 3.0, 4.0],
        dims=("index"),
        name="T",
    )

    tri_grid = td.TriangularGridDataset(
        normal_axis=1,
        normal_pos=0,
        points=tri_grid_points,
        cells=tri_grid_cells,
        values=tri_grid_values,
    )

    mnt_data3 = TemperatureData(monitor=temp_mnt3, temperature=tri_grid)

    mnt_data4 = TemperatureData(monitor=temp_mnt4, temperature=None)

    return (mnt_data1, mnt_data2, mnt_data3, mnt_data4)


def make_voltage_mnt_data():
    _, _, _, _, volt_mnt1, volt_mnt2, volt_mnt3, volt_mnt4 = make_device_mnts()

    nx, ny, nz = 9, 6, 5
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 2, ny)
    z = np.linspace(0, 3, nz)
    T = np.random.default_rng().uniform(-5, 5, (nx, ny, nz))
    coords = dict(x=x, y=y, z=z)
    voltage_field = td.SpatialDataArray(T, coords=coords)

    mnt_data1 = VoltageData(monitor=volt_mnt1, voltage=voltage_field)

    tet_grid_points = td.PointDataArray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dims=("index", "axis"),
    )

    tet_grid_cells = td.CellDataArray(
        [[0, 1, 2, 4], [1, 2, 3, 4]],
        dims=("cell_index", "vertex_index"),
    )

    tet_grid_values = td.IndexedDataArray(
        [1.0, 2.0, 3.0, 4.0, 5.0],
        dims=("index"),
        name="T",
    )

    tet_grid = td.TetrahedralGridDataset(
        points=tet_grid_points,
        cells=tet_grid_cells,
        values=tet_grid_values,
    )

    mnt_data2 = VoltageData(monitor=volt_mnt2, voltage=tet_grid)

    tri_grid_points = td.PointDataArray(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        dims=("index", "axis"),
    )

    tri_grid_cells = td.CellDataArray(
        [[0, 1, 2], [1, 2, 3]],
        dims=("cell_index", "vertex_index"),
    )

    tri_grid_values = td.IndexedDataArray(
        [1.0, 2.0, 3.0, 4.0],
        dims=("index"),
        name="T",
    )

    tri_grid = td.TriangularGridDataset(
        normal_axis=1,
        normal_pos=0,
        points=tri_grid_points,
        cells=tri_grid_cells,
        values=tri_grid_values,
    )

    mnt_data3 = VoltageData(monitor=volt_mnt3, voltage=tri_grid)

    mnt_data4 = VoltageData(monitor=volt_mnt4, voltage=None)

    return (mnt_data1, mnt_data2, mnt_data3, mnt_data4)


def test_device_mnt_data():
    _ = make_temperature_mnt_data()
    _ = make_voltage_mnt_data()


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


def test_device_sources(log_capture):
    # this should issue warning
    _ = HeatSource(structures=["solid_structure"], rate=100)
    assert len(log_capture) == 0

    # this should issue warning
    _ = UniformHeatSource(structures=["solid_structure"], rate=100)
    assert_log_level(log_capture, "WARNING")

    # this shouldn't issue warning
    _ = HeatSource(structures=["solid_structure"], rate="100")
    assert len(log_capture) == 1


def make_device_heat_sim():
    fluid_medium, solid_medium, solid_noHeat, _, _ = make_device_mediums()
    fluid_structure, solid_structure, _, _, _ = make_device_structures()
    bc_temp, bc_flux, bc_conv, bc_volt, bc_current = make_device_bcs()
    heat_source = HeatSource(structures=["solid_structure"], rate=100)

    pl1 = DeviceBoundarySpec(
        condition=bc_conv, placement=MediumMediumInterface(mediums=["fluid_medium", "solid_medium"])
    )
    pl2 = DeviceBoundarySpec(
        condition=bc_flux, placement=StructureBoundary(structure="solid_structure")
    )
    pl3 = DeviceBoundarySpec(
        condition=bc_temp,
        placement=StructureStructureInterface(structures=["fluid_structure", "solid_structure"]),
    )

    grid_spec = make_uniform_grid_spec()

    temp_mnts = make_device_mnts()

    heat_sim = DeviceSimulation(
        medium=fluid_medium,
        structures=[fluid_structure, solid_structure],
        center=(0, 0, 0),
        size=(2, 2, 2),
        boundary_spec=[pl1, pl2, pl3],
        grid_spec=grid_spec,
        sources=[heat_source],
        monitors=temp_mnts[0:4],
    )

    return heat_sim


def make_device_cond_sim():
    _, solid_medium, _, solid_noElect, insulator_medium = make_device_mediums()
    _, solid_structure, _, _, insulator = make_device_structures()
    bc_temp, bc_flux, bc_conv, bc_volt, bc_current = make_device_bcs()

    pl4 = DeviceBoundarySpec(condition=bc_volt, placement=SimulationBoundary())
    pl5 = DeviceBoundarySpec(
        condition=bc_current, placement=StructureSimulationBoundary(structure="insulator_structure")
    )

    grid_spec = make_uniform_grid_spec()

    temp_mnts = make_device_mnts()

    cond_sim = DeviceSimulation(
        medium=insulator_medium,
        structures=[insulator, solid_structure],
        center=(0, 0, 0),
        size=(2, 2, 2),
        boundary_spec=[pl4, pl5],
        grid_spec=grid_spec,
        sources=[],
        monitors=temp_mnts[4:8],
    )

    return cond_sim


def test_device_sim(log_capture):
    bc_temp, bc_flux, bc_conv, bc_volt, bc_current = make_device_bcs()
    (
        fluid_structure,
        solid_structure,
        solid_struct_noHeat,
        solid_struct_noElect,
        insulator,
    ) = make_device_structures()
    heat_sim = make_device_heat_sim()
    cond_sim = make_device_cond_sim()
    sim_types = [heat_sim, cond_sim]

    # COMMON TESTS FOR ALL SIMULATION TYPES
    for sim in sim_types:
        _ = sim.plot(x=0)

        # polyslab support
        vertices = np.array([(0, 0), (1, 0), (1, 1)])
        p = td.PolySlab(vertices=vertices, axis=2, slab_bounds=(-1, 1))
        structure = solid_structure.updated_copy(geometry=p, name="polyslab")
        _ = sim.updated_copy(structures=list(sim.structures) + [structure])

        # stl support
        structure = solid_structure.updated_copy(geometry=STL_GEO, name="stl")
        _ = sim.updated_copy(structures=list(sim.structures) + [structure])

        # run 2D case
        _ = sim.updated_copy(center=(0, 0, 0), size=(0, 2, 2))

        # test unsupported 1D heat domains
        with pytest.raises(pd.ValidationError):
            _ = sim.updated_copy(center=(0, 0, 0), size=(1, 0, 0))

        _ = sim.plot(x=0)
        plt.close()

        with pytest.raises(pd.ValidationError):
            _ = sim.updated_copy(symmetry=(-1, 0, 1))

    # SPECIFIC TESTS FOR EACH SIMUALTION TYPE
    # wrong names given
    for pl in [
        DeviceBoundarySpec(
            condition=bc_temp, placement=MediumMediumInterface(mediums=["badname", "fluid_medium"])
        ),
        DeviceBoundarySpec(condition=bc_flux, placement=StructureBoundary(structure="no_box")),
        DeviceBoundarySpec(
            condition=bc_conv,
            placement=StructureStructureInterface(structures=["no_box", "solid_structure"]),
        ),
        DeviceBoundarySpec(
            condition=bc_temp, placement=StructureSimulationBoundary(structure="no_mesh")
        ),
    ]:
        with pytest.raises(pd.ValidationError):
            _ = heat_sim.updated_copy(boundary_spec=[pl])

    with pytest.raises(pd.ValidationError):
        _ = heat_sim.updated_copy(sources=[HeatSource(structures=["noname"])], rate=-10)

    temp_mnt = heat_sim.monitors[0]
    with pytest.raises(pd.ValidationError):
        heat_sim.updated_copy(monitors=[temp_mnt, temp_mnt])

    _ = heat_sim.plot_scene_specs(plot_type="heat_conductivity", y=0)
    plt.close()

    heat_sim = heat_sim.updated_copy(symmetry=(0, 1, 1))
    _ = heat_sim.plot_scene_specs(plot_type="heat_conductivity", z=0)
    plt.close()

    # fail if assigning structs without heat_spec
    with pytest.raises(pd.ValidationError):
        _ = heat_sim.updated_copy(structures=[fluid_structure, solid_struct_noHeat])

    # fail if assigning structs without electric_spec
    with pytest.raises(pd.ValidationError):
        _ = cond_sim.updated_copy(structures=[insulator, solid_struct_noElect])


@pytest.mark.parametrize("shift_amount, log_level", ((1, None), (2, "WARNING")))
def test_device_sim_bounds(shift_amount, log_level, log_capture):
    """make sure bounds are working correctly"""

    # make sure all things are shifted to this central location
    CENTER_SHIFT = (-1.0, 1.0, 100.0)

    def place_box(center_offset):
        shifted_center = tuple(c + s for (c, s) in zip(center_offset, CENTER_SHIFT))

        _ = td.DeviceSimulation(
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
    _ = td.DeviceSimulation(
        size=(1, 1, 1),
        structures=[box],
        grid_spec=td.UniformUnstructuredGrid(dl=0.1),
    )

    assert_log_level(log_capture, log_level)


def make_device_sim_data():
    heat_sim = make_device_heat_sim()
    temp_data = make_temperature_mnt_data()

    heat_sim_data = DeviceSimulationData(
        simulation=heat_sim,
        data=temp_data,
    )

    cond_sim = make_device_cond_sim()
    volt_data = make_voltage_mnt_data()

    cond_sim_data = DeviceSimulationData(
        simulation=cond_sim,
        data=volt_data,
    )

    return [heat_sim_data, cond_sim_data]


def test_sim_data():
    heat_sim_data, cond_sim_data = make_device_sim_data()
    _ = heat_sim_data.plot_field("test", z=0)
    _ = heat_sim_data.plot_field("tri")
    _ = heat_sim_data.plot_field("tet", y=0.5)

    _ = cond_sim_data.plot_field("v_test", z=0)
    _ = cond_sim_data.plot_field("v_tri")
    _ = cond_sim_data.plot_field("v_tet", y=0.5)
    plt.close()

    with pytest.raises(DataError):
        _ = heat_sim_data.plot_field("empty")

    with pytest.raises(DataError):
        _ = heat_sim_data.plot_field("test")

    with pytest.raises(KeyError):
        _ = heat_sim_data.plot_field("test3", x=0)

    with pytest.raises(pd.ValidationError):
        _ = heat_sim_data.updated_copy(data=[heat_sim_data.data[0]] * 2)

    temp_mnt = TemperatureMonitor(size=(1, 2, 3), name="test")
    temp_mnt = temp_mnt.updated_copy(name="test2")

    sim = heat_sim_data.simulation.updated_copy(monitors=[temp_mnt])

    with pytest.raises(pd.ValidationError):
        _ = heat_sim_data.updated_copy(simulation=sim)
