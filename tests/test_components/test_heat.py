import numpy as np
import pydantic.v1 as pd
import pytest
import tidy3d as td
from matplotlib import pyplot as plt
from tidy3d import (
    ConvectionBC,
    DistanceUnstructuredGrid,
    FluidSpec,
    HeatBoundarySpec,
    HeatFluxBC,
    HeatSimulation,
    HeatSimulationData,
    MediumMediumInterface,
    SimulationBoundary,
    SolidSpec,
    StructureBoundary,
    StructureSimulationBoundary,
    StructureStructureInterface,
    TemperatureBC,
    TemperatureData,
    TemperatureMonitor,
    UniformHeatSource,
    UniformUnstructuredGrid,
)
from tidy3d.exceptions import DataError

from ..utils import AssertLogLevel, cartesian_to_unstructured


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


def make_heat_mnts():
    temp_mnt1 = TemperatureMonitor(size=(1.6, 2, 3), name="test")
    temp_mnt2 = TemperatureMonitor(size=(1.6, 2, 3), name="tet", unstructured=True)
    temp_mnt3 = TemperatureMonitor(
        center=(0, 0.9, 0), size=(1.6, 0, 3), name="tri", unstructured=True, conformal=True
    )
    temp_mnt4 = TemperatureMonitor(
        center=(0, 0.9, 0), size=(1.6, 0, 3), name="empty", unstructured=True, conformal=False
    )
    temp_mnt5 = TemperatureMonitor(center=(0, 0.7, 0.8), size=(3, 0, 0), name="line")
    temp_mnt6 = TemperatureMonitor(center=(0.7, 0.6, 0.8), size=(0, 0, 0), name="point")

    return (temp_mnt1, temp_mnt2, temp_mnt3, temp_mnt4, temp_mnt5, temp_mnt6)


def test_heat_mnt():
    temp_mnt, _, _, _, _, _ = make_heat_mnts()

    with pytest.raises(pd.ValidationError):
        _ = temp_mnt.updated_copy(name=None)

    with pytest.raises(pd.ValidationError):
        _ = temp_mnt.updated_copy(size=(-1, 2, 3))


def make_heat_mnt_data():
    temp_mnt1, temp_mnt2, temp_mnt3, temp_mnt4, temp_mnt5, temp_mnt6 = make_heat_mnts()

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

    nx, ny, nz = 9, 1, 1
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 2, ny)
    z = np.linspace(0, 3, nz)
    T = np.random.default_rng().uniform(300, 350, (nx, ny, nz))
    coords = dict(x=x, y=y, z=z)
    temperature_field = td.SpatialDataArray(T, coords=coords)

    mnt_data5 = TemperatureData(monitor=temp_mnt5, temperature=temperature_field)

    nx, ny, nz = 1, 1, 1
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 2, ny)
    z = np.linspace(0, 3, nz)
    T = np.random.default_rng().uniform(300, 350, (nx, ny, nz))
    coords = dict(x=x, y=y, z=z)
    temperature_field = td.SpatialDataArray(T, coords=coords)

    mnt_data6 = TemperatureData(monitor=temp_mnt6, temperature=temperature_field)

    return (mnt_data1, mnt_data2, mnt_data3, mnt_data4, mnt_data5, mnt_data6)


def test_heat_mnt_data():
    _ = make_heat_mnt_data()


def make_uniform_grid_spec():
    return UniformUnstructuredGrid(
        dl=0.1, min_edges_per_circumference=5, min_edges_per_side=3, relative_min_dl=1e-3
    )


def make_distance_grid_spec():
    return DistanceUnstructuredGrid(
        dl_interface=0.1, dl_bulk=1, distance_interface=1, distance_bulk=2, relative_min_dl=1e-5
    )


def test_grid_spec():
    grid_spec = make_uniform_grid_spec()
    with pytest.raises(pd.ValidationError):
        _ = grid_spec.updated_copy(dl=0)
    with pytest.raises(pd.ValidationError):
        _ = grid_spec.updated_copy(min_edges_per_circumference=-1)
    with pytest.raises(pd.ValidationError):
        _ = grid_spec.updated_copy(min_edges_per_side=-1)
    with pytest.raises(pd.ValidationError):
        _ = grid_spec.updated_copy(relative_min_dl=-1e-4)

    grid_spec = make_distance_grid_spec()
    _ = grid_spec.updated_copy(relative_min_dl=0)
    with pytest.raises(pd.ValidationError):
        _ = grid_spec.updated_copy(dl_interface=-1)
    with pytest.raises(pd.ValidationError):
        _ = grid_spec.updated_copy(distance_interface=2, distance_bulk=1)


def make_heat_source():
    return UniformHeatSource(structures=["solid_structure"], rate=100)


def test_heat_source():
    source = make_heat_source()
    with pytest.raises(pd.ValidationError):
        _ = source.updated_copy(structures=[])


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

    temp_mnts = make_heat_mnts()

    heat_sim = HeatSimulation(
        medium=fluid_medium,
        structures=[fluid_structure, solid_structure],
        center=(0, 0, 0),
        size=(2, 2, 2),
        boundary_spec=[pl1, pl2, pl3, pl4, pl5],
        grid_spec=grid_spec,
        sources=[heat_source],
        monitors=temp_mnts,
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

    # run 2D case
    _ = heat_sim.updated_copy(center=(0.7, 0, 0), size=(0, 2, 2), monitors=heat_sim.monitors[:5])

    # test unsupported 1D heat domains
    with pytest.raises(pd.ValidationError):
        _ = heat_sim.updated_copy(center=(1, 1, 1), size=(1, 0, 0))

    temp_mnt = heat_sim.monitors[0]

    with pytest.raises(pd.ValidationError):
        heat_sim.updated_copy(monitors=[temp_mnt, temp_mnt])

    _ = heat_sim.plot(x=0)
    plt.close()

    _ = heat_sim.plot_heat_conductivity(y=0)
    plt.close()

    heat_sim_sym = heat_sim.updated_copy(symmetry=(0, 1, 1))
    _ = heat_sim_sym.plot_heat_conductivity(z=0, colorbar="source")
    plt.close()

    # no negative symmetry
    with pytest.raises(pd.ValidationError):
        _ = heat_sim.updated_copy(symmetry=(-1, 0, 1))

    # no SolidSpec in the entire simulation
    bc_spec = td.HeatBoundarySpec(
        placement=td.SimulationBoundary(), condition=td.TemperatureBC(temperature=300)
    )
    solid_med = heat_sim.structures[1].medium

    _ = heat_sim.updated_copy(structures=[], medium=solid_med, sources=[], boundary_spec=[bc_spec])
    with pytest.raises(pd.ValidationError):
        _ = heat_sim.updated_copy(structures=[], sources=[], boundary_spec=[bc_spec], monitors=[])

    _ = heat_sim.updated_copy(
        structures=[heat_sim.structures[0]], medium=solid_med, boundary_spec=[bc_spec], sources=[]
    )
    with pytest.raises(pd.ValidationError):
        _ = heat_sim.updated_copy(
            structures=[heat_sim.structures[0]], boundary_spec=[bc_spec], sources=[], monitors=[]
        )

    # 1D and 2D structures
    struct_1d = td.Structure(
        geometry=td.Box(size=(1, 0, 0)),
        medium=solid_med,
    )
    struct_2d = td.Structure(
        geometry=td.Box(size=(1, 0, 1)),
        medium=heat_sim.medium,
    )
    with pytest.raises(pd.ValidationError):
        _ = heat_sim.updated_copy(structures=list(heat_sim.structures) + [struct_1d])

    with pytest.raises(pd.ValidationError):
        _ = heat_sim.updated_copy(structures=list(heat_sim.structures) + [struct_2d])

    # no data expected inside a monitor
    for mnt_size in [(0.2, 0.2, 0.2), (0, 1, 1), (0, 2, 0), (0, 0, 0)]:
        temp_mnt = td.TemperatureMonitor(center=(0, 0, 0), size=mnt_size, name="test")

        with pytest.raises(pd.ValidationError):
            _ = heat_sim.updated_copy(monitors=[temp_mnt])


@pytest.mark.parametrize("shift_amount, log_level", ((1, None), (2, "WARNING")))
def test_heat_sim_bounds(shift_amount, log_level):
    """make sure bounds are working correctly"""

    # make sure all things are shifted to this central location
    CENTER_SHIFT = (-1.0, 1.0, 100.0)

    def place_box(center_offset):
        shifted_center = tuple(c + s for (c, s) in zip(center_offset, CENTER_SHIFT))

        _ = td.HeatChargeSimulation(
            size=(1.5, 1.5, 1.5),
            center=CENTER_SHIFT,
            medium=td.Medium(heat_spec=td.SolidSpec(conductivity=1, capacity=1)),
            structures=[
                td.Structure(
                    geometry=td.Box(size=(1, 1, 1), center=shifted_center), medium=td.Medium()
                )
            ],
            boundary_spec=[
                td.HeatBoundarySpec(
                    placement=td.SimulationBoundary(), condition=td.TemperatureBC(temperature=300)
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
            with AssertLogLevel(log_level):
                place_box(tuple(center))


@pytest.mark.parametrize(
    "box_size,log_level",
    [
        ((1, 0.1, 0.1), "WARNING"),
        ((0.1, 1, 0.1), "WARNING"),
        ((0.1, 0.1, 1), "WARNING"),
    ],
)
def test_sim_structure_extent(box_size, log_level):
    """Make sure we warn if structure extends exactly to simulation edges."""

    box = td.Structure(geometry=td.Box(size=box_size), medium=td.Medium(permittivity=2))

    with AssertLogLevel(log_level):
        _ = td.HeatSimulation(
            size=(1, 1, 1),
            medium=td.Medium(heat_spec=td.SolidSpec(conductivity=1, capacity=1)),
            structures=[box],
            boundary_spec=[
                td.HeatBoundarySpec(
                    placement=td.SimulationBoundary(), condition=td.TemperatureBC(temperature=300)
                )
            ],
            grid_spec=td.UniformUnstructuredGrid(dl=0.1),
        )


def make_heat_sim_data():
    heat_sim = make_heat_sim()
    temp_data = make_heat_mnt_data()

    heat_sim_data = HeatSimulationData(
        simulation=heat_sim,
        data=temp_data,
    )

    return heat_sim_data


def test_sim_data():
    heat_sim_data = make_heat_sim_data()
    _ = heat_sim_data.plot_field("test", z=0)
    _ = heat_sim_data.plot_field("tri")
    _ = heat_sim_data.plot_field("tet", y=0.5)
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


def test_relative_min_dl_warning():
    with AssertLogLevel("WARNING"):
        _ = td.HeatSimulation(
            size=(1, 1, 1),
            medium=td.Medium(heat_spec=td.SolidSpec(conductivity=1, capacity=2)),
            grid_spec=td.UniformUnstructuredGrid(dl=0.0001, relative_min_dl=1e-2),
            boundary_spec=[
                td.HeatBoundarySpec(
                    placement=td.SimulationBoundary(), condition=td.TemperatureBC(temperature=300)
                )
            ],
        )

    with AssertLogLevel("WARNING"):
        _ = td.HeatSimulation(
            size=(1, 1, 1),
            medium=td.Medium(heat_spec=td.SolidSpec(conductivity=1, capacity=2)),
            grid_spec=td.DistanceUnstructuredGrid(
                dl_interface=0.0001,
                dl_bulk=0.1,
                distance_interface=0.01,
                distance_bulk=0.5,
                relative_min_dl=1e-2,
            ),
            boundary_spec=[
                td.HeatBoundarySpec(
                    placement=td.SimulationBoundary(), condition=td.TemperatureBC(temperature=300)
                )
            ],
        )

    with AssertLogLevel("WARNING"):
        _ = td.HeatSimulation(
            size=(1, 1, 1),
            medium=td.Medium(heat_spec=td.SolidSpec(conductivity=1, capacity=2)),
            grid_spec=td.DistanceUnstructuredGrid(
                dl_interface=0.1,
                dl_bulk=0.0001,
                distance_interface=0.01,
                distance_bulk=0.5,
                relative_min_dl=1e-2,
            ),
            boundary_spec=[
                td.HeatBoundarySpec(
                    placement=td.SimulationBoundary(), condition=td.TemperatureBC(temperature=300)
                )
            ],
        )


def test_sim_version_update():
    heat_sim = make_heat_sim()
    heat_sim_dict = heat_sim.dict()
    heat_sim_dict["version"] = "ancient_version"

    with AssertLogLevel("WARNING"):
        heat_sim_new = td.HeatSimulation.parse_obj(heat_sim_dict)

    assert heat_sim_new.version == td.__version__


@pytest.mark.parametrize("zero_dim_axis", [None, 0, 2])
def test_symmetry_expanded(zero_dim_axis):
    symmetry_center = [2, 0.5, 0]
    symmetry = [1, 1, 1]

    lens = [1, 2, 2]
    num_points = [7, 4, 11]

    if zero_dim_axis is not None:
        lens[zero_dim_axis] = 0
        num_points[zero_dim_axis] = 1

    mnt_span_x = [1 - lens[0], 1]
    mnt_span_y = [-lens[1] / 2, lens[1] / 2]
    mnt_span_z = [1, 1 + lens[2]]

    # symmetric around symmetry_center
    data_span_x = [3, 3 + lens[0]]
    data_span_y = [0.5, 0.5 + lens[1]]
    data_span_z = [1, 1 + lens[2]]

    mnt_bounds = np.array(list(zip(mnt_span_x, mnt_span_y, mnt_span_z)))
    mnt_size = tuple(mnt_bounds[1] - mnt_bounds[0])
    mnt_center = tuple((mnt_bounds[1] + mnt_bounds[0]) / 2)

    x = np.linspace(*data_span_x, num_points[0])
    y = np.linspace(*data_span_y, num_points[1])
    z = np.linspace(*data_span_z, num_points[2])
    v = np.sin(x[:, None, None]) * np.cos(y[None, :, None]) * np.exp(z[None, None, :])

    data_cart = td.SpatialDataArray(v, coords=dict(x=x, y=y, z=z))
    data_ugrid = cartesian_to_unstructured(data_cart, seed=33342)

    mnt_cart = td.TemperatureMonitor(
        center=mnt_center, size=mnt_size, name="test", unstructured=False
    )
    mnt_ugrid = td.TemperatureMonitor(
        center=mnt_center, size=mnt_size, name="test", unstructured=True
    )

    mnt_data_cart = td.TemperatureData(
        temperature=data_cart, monitor=mnt_cart, symmetry=symmetry, symmetry_center=symmetry_center
    )
    mnt_data_ugrid = td.TemperatureData(
        temperature=data_ugrid,
        monitor=mnt_ugrid,
        symmetry=symmetry,
        symmetry_center=symmetry_center,
    )

    mnt_data_cart_expanded = mnt_data_cart.symmetry_expanded_copy
    mnt_data_ugrid_expanded = mnt_data_ugrid.symmetry_expanded_copy

    assert mnt_data_cart_expanded.symmetry == (0, 0, 0)
    assert mnt_data_ugrid_expanded.symmetry == (0, 0, 0)

    data_expanded_cart = mnt_data_cart_expanded.temperature
    data_expanded_ugrid = mnt_data_ugrid_expanded.temperature

    print(data_expanded_ugrid.bounds)
    print(mnt_bounds)

    assert np.all(data_expanded_ugrid.bounds == mnt_bounds)
    assert data_expanded_cart.does_cover(mnt_bounds)
