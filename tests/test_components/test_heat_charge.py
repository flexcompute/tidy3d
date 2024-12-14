"""Test for heat-charge simulation objects and data"""

import numpy as np
import pydantic.v1 as pd
import pytest
import tidy3d as td
from matplotlib import pyplot as plt
from tidy3d.exceptions import DataError

from ..utils import assert_log_level


class CHARGE_SIMULATION:
    """This class contains all elements to be tested"""

    # dimensions semiconductors
    width = 0.2  # um
    hight = 0.2  # um
    z_dim = width / 2

    # simulation size
    sim_size = (3 * width, 2 * hight, z_dim)

    # doping
    acceptors = 1e17
    donors = 5e17


def make_heat_charge_mediums():
    """Creates mediums with different specs"""
    fluid_medium = td.Medium(
        permittivity=3,
        heat_spec=td.FluidSpec(),
        name="fluid_medium",
    )
    solid_medium = td.Medium(
        permittivity=5,
        conductivity=0.01,
        heat_spec=td.SolidSpec(
            capacity=2,
            conductivity=3,
        ),
        electric_spec=td.ConductorSpec(
            conductivity=1,
        ),
        name="solid_medium",
    )

    solid_noHeat = td.Medium(
        permittivity=5,
        conductivity=0.01,
        electric_spec=td.ConductorSpec(
            conductivity=1,
        ),
        name="solid_medium",
    )

    solid_noElect = td.Medium(
        permittivity=5,
        conductivity=0.01,
        heat_spec=td.SolidSpec(
            capacity=2,
            conductivity=3,
        ),
        name="solid_medium",
    )

    insulator_medium = td.Medium(
        permittivity=3,
        electric_spec=td.InsulatorSpec(),
        name="insulator_medium",
    )

    return fluid_medium, solid_medium, solid_noHeat, solid_noElect, insulator_medium


def test_heat_charge_medium():
    """Tests validation errors for mediums"""
    _, solid_medium, _, _, _ = make_heat_charge_mediums()

    with pytest.raises(pd.ValidationError):
        _ = solid_medium.heat_spec.updated_copy(capacity=-1)

    with pytest.raises(pd.ValidationError):
        _ = solid_medium.heat_spec.updated_copy(conductivity=-1)

    with pytest.raises(pd.ValidationError):
        _ = solid_medium.electric_spec.updated_copy(conductivity=-1)


def make_heat_charge_structures():
    """Makes structures with different mediums and sizes"""
    (
        fluid_medium,
        solid_medium,
        solid_noHeat,
        solid_noElect,
        insulator_med,
    ) = make_heat_charge_mediums()

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


def test_heat_charge_structures():
    """Tests that different structures with different mediums can be created"""
    _, _, _, _, _ = make_heat_charge_structures()


def make_heat_charge_bcs():
    bc_temp = td.TemperatureBC(temperature=300)
    bc_flux = td.HeatFluxBC(flux=20)
    bc_conv = td.ConvectionBC(ambient_temperature=400, transfer_coeff=0.2)
    bc_volt = td.VoltageBC(voltage=1)
    bc_current = td.CurrentBC(current_density=3e-1)

    return [bc_temp, bc_flux, bc_conv, bc_volt, bc_current]


def test_heat_charge_bcs():
    """Tests the validators for boundary conditions"""
    with pytest.raises(pd.ValidationError):
        _ = td.TemperatureBC(temperature=-10)

    with pytest.raises(pd.ValidationError):
        _ = td.ConvectionBC(ambient_temperature=-400, transfer_coeff=0.2)

    with pytest.raises(pd.ValidationError):
        _ = td.ConvectionBC(ambient_temperature=400, transfer_coeff=-0.2)

    with pytest.raises(pd.ValidationError):
        _ = td.VoltageBC(voltage=td.inf)

    with pytest.raises(pd.ValidationError):
        _ = td.CurrentBC(current_density=td.inf)


def make_heat_charge_mnts():
    """Creates monitors of different types and sized"""
    temp_mnt1 = td.TemperatureMonitor(size=(1.6, 2, 3), name="test")
    temp_mnt2 = td.TemperatureMonitor(size=(1.6, 2, 3), name="tet", unstructured=True)
    temp_mnt3 = td.TemperatureMonitor(
        center=(0, 0.9, 0), size=(1.6, 0, 3), name="tri", unstructured=True, conformal=True
    )
    temp_mnt4 = td.TemperatureMonitor(
        center=(0, 0.9, 0), size=(1.6, 0, 3), name="empty", unstructured=True, conformal=False
    )

    volt_mnt1 = td.StaticVoltageMonitor(size=(1.6, 2, 3), name="v_test")
    volt_mnt2 = td.StaticVoltageMonitor(size=(1.6, 2, 3), name="v_tet", unstructured=True)
    volt_mnt3 = td.StaticVoltageMonitor(
        center=(0, 0.9, 0), size=(1.6, 0, 3), name="v_tri", unstructured=True, conformal=True
    )
    volt_mnt4 = td.StaticVoltageMonitor(
        center=(0, 0.9, 0), size=(1.6, 0, 3), name="v_empty", unstructured=True, conformal=False
    )

    return [temp_mnt1, temp_mnt2, temp_mnt3, temp_mnt4, volt_mnt1, volt_mnt2, volt_mnt3, volt_mnt4]


def test_heat_charge_mnt():
    """Checking for no name and negative size in monitors"""
    # NOTE: both Temperature and Voltage monitors derive from the same class. Since
    # both of these classes are empty we're actually checking the base class.

    mnts = make_heat_charge_mnts()
    temp_mnt = mnts[0]

    with pytest.raises(pd.ValidationError):
        _ = temp_mnt.updated_copy(name=None)

    with pytest.raises(pd.ValidationError):
        _ = temp_mnt.updated_copy(size=(-1, 2, 3))


def test_monitor_crosses_medium():
    """Test whether monitor crosses structures with relevant material specification"""
    _, _, solid_noHeat, solid_noElect, _ = make_heat_charge_mediums()
    _, _, solid_struct_noHeat, solid_struct_noElect, _ = make_heat_charge_structures()
    heat_sim = make_heat_charge_heat_sim()
    cond_sim = make_heat_charge_cond_sim()

    volt_monitor = td.StaticVoltageMonitor(
        center=(0, 0, 0), size=(td.inf, td.inf, td.inf), name="voltage"
    )
    # a volt monitor in a heat sim should throw error
    # if no materials with ConductorSpec present
    with pytest.raises(pd.ValidationError):
        _ = heat_sim.updated_copy(
            medium=solid_noElect, structures=[solid_struct_noElect], monitors=[volt_monitor]
        )

    temp_monitor = td.TemperatureMonitor(
        center=(0, 0, 0), size=(td.inf, td.inf, td.inf), name="temperature"
    )
    # a temperature monitor should throw error in a conduction simulation
    # if no material with SolidSpec present
    with pytest.raises(pd.ValidationError):
        _ = cond_sim.updated_copy(
            medium=solid_noHeat, structures=[solid_struct_noHeat], monitors=[temp_monitor]
        )


def make_temperature_mnt_data():
    """
    Creates different temperature data. The first one uses
    a 'SpatialDataArray', the second a 'TetrahedralGridDataset', the third one
    a 'TriangularGridDataset' and the last one uses 'None' as data."""
    temp_mnt1, temp_mnt2, temp_mnt3, temp_mnt4, _, _, _, _ = make_heat_charge_mnts()

    nx, ny, nz = 9, 6, 5
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 2, ny)
    z = np.linspace(0, 3, nz)
    T = np.random.default_rng().uniform(300, 350, (nx, ny, nz))
    coords = dict(x=x, y=y, z=z)
    temperature_field = td.SpatialDataArray(T, coords=coords)

    mnt_data1 = td.TemperatureData(monitor=temp_mnt1, temperature=temperature_field)

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

    mnt_data2 = td.TemperatureData(monitor=temp_mnt2, temperature=tet_grid)

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

    mnt_data3 = td.TemperatureData(monitor=temp_mnt3, temperature=tri_grid)

    mnt_data4 = td.TemperatureData(monitor=temp_mnt4, temperature=None)

    return (mnt_data1, mnt_data2, mnt_data3, mnt_data4)


def make_voltage_mnt_data():
    """
    Creates different voltage data. The first one uses
    a 'SpatialDataArray', the second a 'TetrahedralGridDataset', the third one
    a 'TriangularGridDataset' and the last one uses 'None' as data."""
    _, _, _, _, volt_mnt1, volt_mnt2, volt_mnt3, volt_mnt4 = make_heat_charge_mnts()

    nx, ny, nz = 9, 6, 5
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 2, ny)
    z = np.linspace(0, 3, nz)
    T = np.random.default_rng().uniform(-5, 5, (nx, ny, nz))
    coords = dict(x=x, y=y, z=z)
    voltage_field = td.SpatialDataArray(T, coords=coords)

    mnt_data1 = td.StaticVoltageData(monitor=volt_mnt1, voltage=voltage_field)

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

    mnt_data2 = td.StaticVoltageData(monitor=volt_mnt2, voltage=tet_grid)

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

    mnt_data3 = td.StaticVoltageData(monitor=volt_mnt3, voltage=tri_grid)

    mnt_data4 = td.StaticVoltageData(monitor=volt_mnt4, voltage=None)

    return (mnt_data1, mnt_data2, mnt_data3, mnt_data4)


def test_heat_charge_mnt_data():
    """Tests whether different heat-charge monitor data can be created"""
    _ = make_temperature_mnt_data()
    _ = make_voltage_mnt_data()


def make_uniform_grid_spec():
    """Generates a uniform unstructured grid"""
    return td.UniformUnstructuredGrid(dl=0.1, min_edges_per_circumference=5, min_edges_per_side=3)


def make_distance_grid_spec():
    """Generates a distance-based unstructured grid"""
    return td.DistanceUnstructuredGrid(
        dl_interface=0.1, dl_bulk=1, distance_interface=1, distance_bulk=2
    )


def test_grid_spec():
    """Tests whether unstructured grids can be created and different validators for them."""
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


def test_heat_charge_sources(log_capture):  # noqa: F811
    """ "Tests whether heat-charge sources can be created and associated warnings"""
    # this shouldn't issue warning
    _ = td.HeatSource(structures=["solid_structure"], rate=100)
    assert len(log_capture) == 0

    # this should issue warning
    _ = td.UniformHeatSource(structures=["solid_structure"], rate=100)
    assert_log_level(log_capture, "WARNING")

    # this shouldn't issue warning
    _ = td.HeatSource(structures=["solid_structure"], rate="100")
    assert len(log_capture) == 1


def make_heat_charge_heat_sim():
    """Generates heat-charge heat simulation"""
    fluid_medium, solid_medium, solid_noHeat, _, _ = make_heat_charge_mediums()
    fluid_structure, solid_structure, _, _, _ = make_heat_charge_structures()
    bc_temp, bc_flux, bc_conv, bc_volt, bc_current = make_heat_charge_bcs()
    heat_source = td.HeatSource(structures=["solid_structure"], rate=100)

    pl1 = td.HeatChargeBoundarySpec(
        condition=bc_conv,
        placement=td.MediumMediumInterface(mediums=["fluid_medium", "solid_medium"]),
    )
    pl2 = td.HeatChargeBoundarySpec(
        condition=bc_flux, placement=td.StructureBoundary(structure="solid_structure")
    )
    pl3 = td.HeatChargeBoundarySpec(
        condition=bc_temp,
        placement=td.StructureStructureInterface(structures=["fluid_structure", "solid_structure"]),
    )

    grid_spec = make_uniform_grid_spec()

    temp_mnts = make_heat_charge_mnts()

    heat_sim = td.HeatChargeSimulation(
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


def make_heat_charge_cond_sim():
    """Creates a heat-charge conduction simulation"""
    _, solid_medium, _, solid_noElect, insulator_medium = make_heat_charge_mediums()
    _, solid_structure, _, _, insulator = make_heat_charge_structures()
    bc_temp, bc_flux, bc_conv, bc_volt, bc_current = make_heat_charge_bcs()

    pl4 = td.HeatChargeBoundarySpec(condition=bc_volt, placement=td.SimulationBoundary())
    pl5 = td.HeatChargeBoundarySpec(
        condition=bc_current,
        placement=td.StructureSimulationBoundary(structure="insulator_structure"),
    )

    grid_spec = make_uniform_grid_spec()

    temp_mnts = make_heat_charge_mnts()

    cond_sim = td.HeatChargeSimulation(
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


def test_heat_charge_sim(log_capture):  # noqa: F811
    """Creates heat-charge simulations and performs a bunch of
    checks for different conditions"""
    bc_temp, bc_flux, bc_conv, bc_volt, bc_current = make_heat_charge_bcs()
    (
        fluid_structure,
        solid_structure,
        solid_struct_noHeat,
        solid_struct_noElect,
        insulator,
    ) = make_heat_charge_structures()
    heat_sim = make_heat_charge_heat_sim()
    cond_sim = make_heat_charge_cond_sim()
    sim_types = [heat_sim, cond_sim]

    # COMMON TESTS FOR ALL SIMULATION TYPES
    for sim in sim_types:
        _ = sim.plot(x=0)

        # run 2D case
        _ = sim.updated_copy(center=(0.7, 0, 0), size=(0, 2, 2))

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
        td.HeatChargeBoundarySpec(
            condition=bc_temp,
            placement=td.MediumMediumInterface(mediums=["badname", "fluid_medium"]),
        ),
        td.HeatChargeBoundarySpec(
            condition=bc_flux, placement=td.StructureBoundary(structure="no_box")
        ),
        td.HeatChargeBoundarySpec(
            condition=bc_conv,
            placement=td.StructureStructureInterface(structures=["no_box", "solid_structure"]),
        ),
        td.HeatChargeBoundarySpec(
            condition=bc_temp, placement=td.StructureSimulationBoundary(structure="no_mesh")
        ),
    ]:
        with pytest.raises(pd.ValidationError):
            _ = heat_sim.updated_copy(boundary_spec=[pl])

    with pytest.raises(pd.ValidationError):
        _ = heat_sim.updated_copy(sources=[td.HeatSource(structures=["noname"])], rate=-10)

    temp_mnt = heat_sim.monitors[0]
    with pytest.raises(pd.ValidationError):
        heat_sim.updated_copy(monitors=[temp_mnt, temp_mnt])

    _ = heat_sim.plot_property(property="heat_conductivity", y=0)
    plt.close()

    heat_sim = heat_sim.updated_copy(symmetry=(0, 1, 1))
    _ = heat_sim.plot_property(property="heat_conductivity", z=0)
    plt.close()

    # fail if assigning structs without heat_spec
    with pytest.raises(pd.ValidationError):
        _ = heat_sim.updated_copy(structures=[fluid_structure, solid_struct_noHeat])

    # fail if assigning structs without electric_spec
    with pytest.raises(pd.ValidationError):
        _ = cond_sim.updated_copy(structures=[insulator, solid_struct_noElect])

    # no data expected inside a monitor
    for mnt_size in [(0.2, 0.2, 0.2), (0, 1, 1), (0, 2, 0), (0, 0, 0)]:
        temp_mnt = td.TemperatureMonitor(center=(0, 0, 0), size=mnt_size, name="test")
        with pytest.raises(pd.ValidationError):
            _ = heat_sim.updated_copy(monitors=[temp_mnt])

    # fail if 'HeatFromElectricSource' is provided in simulations where only BCs/sources
    # are provided that are either HEAT or CONDUCTION
    bc_spec_HEAT = td.HeatChargeBoundarySpec(
        condition=bc_temp, placement=td.StructureBoundary(structure=solid_structure.name)
    )
    sim = td.HeatChargeSimulation(
        medium=solid_structure.medium,
        center=(0, 0, 0),
        size=(3, 3, 3),
        grid_spec=td.UniformUnstructuredGrid(dl=0.2),
        structures=[solid_structure],
        boundary_spec=[bc_spec_HEAT],
    )
    with pytest.raises(pd.ValidationError):
        _ = sim.updated_copy(sources=[td.HeatFromElectricSource()])

    # now lets make sim have conduction BC
    bc_spec_COND = bc_spec_HEAT.updated_copy(condition=bc_volt)
    sim = sim.updated_copy(boundary_spec=[bc_spec_COND])
    with pytest.raises(pd.ValidationError):
        _ = sim.updated_copy(sources=[td.HeatFromElectricSource()])

    # now let's make a coupled simulation
    sim = sim.updated_copy(boundary_spec=[bc_spec_COND, bc_spec_HEAT])
    _ = sim.updated_copy(sources=[td.HeatFromElectricSource()])


@pytest.mark.parametrize("shift_amount, log_level", ((1, None), (2, "WARNING")))
def test_heat_charge_sim_bounds(shift_amount, log_level, log_capture):  # noqa: F811
    """make sure bounds are working correctly"""

    # make sure all things are shifted to this central location
    CENTER_SHIFT = (-1.0, 1.0, 100.0)

    def place_box(center_offset):
        shifted_center = tuple(c + s for (c, s) in zip(center_offset, CENTER_SHIFT))

        _ = td.HeatChargeSimulation(
            size=(1.5, 1.5, 1.5),
            center=CENTER_SHIFT,
            medium=td.Medium(electric_spec=td.ConductorSpec(conductivity=1)),
            structures=[
                td.Structure(
                    geometry=td.Box(size=(1, 1, 1), center=shifted_center), medium=td.Medium()
                )
            ],
            boundary_spec=[
                td.HeatChargeBoundarySpec(
                    condition=td.VoltageBC(voltage=1), placement=td.SimulationBoundary()
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
def test_sim_structure_extent(log_capture, box_size, log_level):  # noqa: F811
    """Make sure we warn if structure extends exactly to simulation edges."""

    box = td.Structure(geometry=td.Box(size=box_size), medium=td.Medium(permittivity=2))
    _ = td.HeatChargeSimulation(
        size=(1, 1, 1),
        structures=[box],
        medium=td.Medium(electric_spec=td.ConductorSpec(conductivity=1)),
        boundary_spec=[
            td.HeatChargeBoundarySpec(
                placement=td.SimulationBoundary(), condition=td.VoltageBC(voltage=1)
            )
        ],
        grid_spec=td.UniformUnstructuredGrid(dl=0.1),
    )

    assert_log_level(log_capture, log_level)


def make_heat_charge_sim_data():
    """Creates 'HeatChargeSimulationData' for both HEAT and CONDUCTION simulations"""
    heat_sim = make_heat_charge_heat_sim()
    temp_data = make_temperature_mnt_data()

    heat_sim_data = td.HeatChargeSimulationData(
        simulation=heat_sim,
        data=temp_data,
    )

    cond_sim = make_heat_charge_cond_sim()
    volt_data = make_voltage_mnt_data()

    cond_sim_data = td.HeatChargeSimulationData(
        simulation=cond_sim,
        data=volt_data,
    )

    return [heat_sim_data, cond_sim_data]


def test_sim_data():
    """Tests 'HeatChargeSimulationData'. In particular, checks whether data can plotted
    and appropriate errors are issued (no data, data is not 2D, etc.)"""
    heat_sim_data, cond_sim_data = make_heat_charge_sim_data()
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

    temp_mnt = td.TemperatureMonitor(size=(1, 2, 3), name="test")
    temp_mnt = temp_mnt.updated_copy(name="test2")

    sim = heat_sim_data.simulation.updated_copy(monitors=[temp_mnt])

    with pytest.raises(pd.ValidationError):
        _ = heat_sim_data.updated_copy(simulation=sim)


class TestCharge:
    # define semiconductor materials
    @pytest.fixture(scope="class")
    def Si_p(self):
        return td.Medium(
            electric_spec=td.SemiConductorSpec(
                conductivity=1,
                permittivity=11.7,
                donors=0,
                acceptors=CHARGE_SIMULATION.acceptors,
            ),
            name="Si_p",
        )

    @pytest.fixture(scope="class")
    def Si_n(self):
        return td.Medium(
            electric_spec=td.SemiConductorSpec(
                conductivity=1,
                permittivity=11.7,
                donors=CHARGE_SIMULATION.donors,
                acceptors=0,
            ),
            name="Si_n",
        )

    @pytest.fixture(scope="class")
    def SiO2(self):
        return td.Medium(
            electric_spec=td.InsulatorSpec(permittivity=3.9),
            name="SiO2",
        )

    # define structures
    @pytest.fixture(scope="class")
    def oxide(self, SiO2):
        return td.Structure(
            geometry=td.Box(center=(0, 0, 0), size=CHARGE_SIMULATION.sim_size),
            medium=SiO2,
            name="oxide",
        )

    @pytest.fixture(scope="class")
    def p_side(self, Si_p):
        return td.Structure(
            geometry=td.Box(
                center=(-CHARGE_SIMULATION.width / 2, 0, 0),
                size=(CHARGE_SIMULATION.width, CHARGE_SIMULATION.hight, CHARGE_SIMULATION.z_dim),
            ),
            medium=Si_p,
            name="p_side",
        )

    @pytest.fixture(scope="class")
    def n_side(self, Si_n):
        return td.Structure(
            geometry=td.Box(
                center=(CHARGE_SIMULATION.width / 2, 0, 0),
                size=(CHARGE_SIMULATION.width, CHARGE_SIMULATION.hight, CHARGE_SIMULATION.z_dim),
            ),
            medium=Si_n,
            name="n_side",
        )

    # define BCs
    @pytest.fixture(scope="class")
    def bc_p(self, SiO2, Si_p):
        return td.HeatChargeBoundarySpec(
            condition=td.VoltageBC(voltage=0),
            placement=td.MediumMediumInterface(mediums=[SiO2.name, Si_p.name]),
        )

    @pytest.fixture(scope="class")
    def bc_n(self, SiO2, Si_n):
        return td.HeatChargeBoundarySpec(
            condition=td.VoltageBC(voltage=0),
            placement=td.MediumMediumInterface(mediums=[SiO2.name, Si_n.name]),
        )

    # monitors
    @pytest.fixture(scope="class")
    def charge_global_mnt(self):
        return td.StaticChargeCarrierMonitor(
            center=(0, 0, 0),
            size=(td.inf, td.inf, td.inf),
            name="charge_global_mnt",
            unstructured=True,
        )

    @pytest.fixture(scope="class")
    def potential_global_mnt(self):
        return td.StaticVoltageMonitor(
            center=(0, 0, 0),
            size=(td.inf, td.inf, td.inf),
            name="potential_global_mnt",
            unstructured=True,
        )

    @pytest.fixture(scope="class")
    def capacitance_global_mnt(self):
        return td.StaticCapacitanceMonitor(
            center=(0, 0, 0),
            size=(td.inf, td.inf, td.inf),
            name="capacitance_global_mnt",
            unstructured=True,
        )

    # Charge settings
    @pytest.fixture(scope="class")
    def charge_tolerance(self):
        return td.ChargeToleranceSpec(rel_tol=1e5, abs_tol=1e3, max_iters=400)

    @pytest.fixture(scope="class")
    def charge_dc_regime(self):
        return td.DCSpec(dv=1)

    def test_charge_simulation(
        self,
        oxide,
        p_side,
        n_side,
        charge_global_mnt,
        potential_global_mnt,
        capacitance_global_mnt,
        bc_n,
        bc_p,
        charge_tolerance,
        charge_dc_regime,
    ):
        """Make sure charge simulation produces the correct errors when needed."""

        sim = td.HeatChargeSimulation(
            structures=[oxide, p_side, n_side],
            medium=td.Medium(heat_spec=td.FluidSpec(), name="air"),
            monitors=[charge_global_mnt, potential_global_mnt, capacitance_global_mnt],
            center=(0, 0, 0),
            size=CHARGE_SIMULATION.sim_size,
            grid_spec=td.UniformUnstructuredGrid(dl=0.05),
            boundary_spec=[bc_n, bc_p],
            charge_tolerance=charge_tolerance,
            charge_regime=charge_dc_regime,
        )

        # at least one ChargeSimulationMonitor should be added
        with pytest.raises(pd.ValidationError):
            _ = sim.updated_copy(monitors=[])

        # at least 2 VoltageBCs should be defined
        with pytest.raises(pd.ValidationError):
            _ = sim.updated_copy(boundary_spec=[bc_n])

        # define ChargeSimulation with no Semiconductor materials
        medium = td.Medium(
            electric_spec=td.ConductorSpec(permittivity=1, conductivity=1),
            name="medium",
        )
        new_structures = [struct.updated_copy(medium=medium) for struct in sim.structures]

        with pytest.raises(pd.ValidationError):
            _ = sim.updated_copy(structures=new_structures)
