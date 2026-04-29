"""Test suite for heat-charge simulation objects and data using pytest fixtures."""

from __future__ import annotations

import numpy as np
import pytest
from matplotlib import pyplot as plt
from pydantic import ValidationError

import tidy3d as td
from tidy3d.components.tcad.simulation.heat_charge import TCADAnalysisTypes
from tidy3d.components.tcad.types import (
    AugerRecombination,
    CaugheyThomasMobility,
    ConstantEffectiveDOS,
    ConstantEnergyBandGap,
    SlotboomBandGapNarrowing,
)
from tidy3d.exceptions import DataError

from ..utils import AssertLogLevel, assert_single_value_error_loc


class CHARGE_SIMULATION:
    """This class contains all elements to be tested."""

    # Dimensions of semiconductors
    width = 0.2  # um
    height = 0.2  # um
    z_dim = width / 2

    # Simulation size
    sim_size = (3 * width, 2 * height, z_dim)

    # Doping concentrations
    acceptors = 1e17
    donors = 5e17

    # intrinsic semiconductor
    intrinsic_Si = td.MultiPhysicsMedium(
        charge=td.SemiconductorMedium(
            permittivity=11.7,
            N_d=0,
            N_a=0,
            N_c=ConstantEffectiveDOS(N=2.86e19),
            N_v=ConstantEffectiveDOS(N=3.1e19),
            E_g=ConstantEnergyBandGap(eg=1.11),
            mobility_n=CaugheyThomasMobility(
                mu_min=52.2,
                mu=1471.0,
                ref_N=9.68e16,
                exp_N=0.68,
                exp_1=-0.57,
                exp_2=-2.33,
                exp_3=2.4,
                exp_4=-0.146,
            ),
            mobility_p=CaugheyThomasMobility(
                mu_min=44.9,
                mu=470.5,
                ref_N=2.23e17,
                exp_N=0.719,
                exp_1=-0.57,
                exp_2=-2.33,
                exp_3=2.4,
                exp_4=-0.146,
            ),
            R=[
                AugerRecombination(c_n=2.8e-31, c_p=9.9e-32),
            ],
            delta_E_g=SlotboomBandGapNarrowing(
                v1=6.92 * 1e-3,
                n2=1.3e17,
                c2=0.5,
                min_N=1e15,
            ),
        ),
        name="Si_intrinsic",
    )


# --------------------------
# Pytest Fixtures
# --------------------------


@pytest.fixture(scope="module")
def mediums():
    """Creates mediums with different specifications."""
    fluid_medium = td.Medium(
        permittivity=3,
        heat_spec=td.FluidSpec(),
        name="fluid_medium",
    )
    solid_medium = td.MultiPhysicsMedium(
        optical=td.Medium(
            permittivity=5,
            conductivity=0.01,
            heat_spec=td.SolidSpec(
                capacity=2,
                conductivity=3,
            ),
        ),
        charge=td.ChargeConductorMedium(
            conductivity=1,
        ),
        heat=td.SolidMedium(conductivity=1.1, capacity=1.2, density=2.3),
        name="solid_medium",
    )

    solid_no_heat = td.MultiPhysicsMedium(
        optical=td.Medium(
            permittivity=5,
            conductivity=0.01,
        ),
        charge=td.ChargeConductorMedium(
            conductivity=1,
        ),
        name="solid_no_heat",
    )

    solid_no_elect = td.Medium(
        permittivity=5,
        conductivity=0.01,
        heat_spec=td.SolidSpec(
            capacity=2,
            conductivity=3,
        ),
        name="solid_no_elect",
    )

    insulator_medium = td.MultiPhysicsMedium(
        optical=td.Medium(
            permittivity=3,
        ),
        charge=td.ChargeInsulatorMedium(),
        name="insulator_medium",
    )

    semiconductor_medium = td.MultiPhysicsMedium(
        optical=td.Medium(
            permittivity=5,
            conductivity=0.01,
            heat_spec=td.SolidSpec(
                capacity=2,
                conductivity=3,
            ),
        ),
        charge=td.SemiconductorMedium(
            N_c=td.ConstantEffectiveDOS(N=1e10),
            N_v=td.ConstantEffectiveDOS(N=1e10),
            E_g=td.ConstantEnergyBandGap(eg=1),
            mobility_n=td.ConstantMobilityModel(mu=1500),
            mobility_p=td.ConstantMobilityModel(mu=1500),
        ),
        name="solid_medium",
    )

    return {
        "fluid_medium": fluid_medium,
        "solid_medium": solid_medium,
        "solid_no_heat": solid_no_heat,
        "solid_no_elect": solid_no_elect,
        "insulator_medium": insulator_medium,
        "semiconductor_medium": semiconductor_medium,
    }


@pytest.fixture(scope="module")
def structures(mediums):
    """Creates structures with different mediums and sizes."""
    box = td.Box(center=(0, 0, 0), size=(1, 1, 1))  # Adjusted size for consistency

    fluid_structure = td.Structure(
        geometry=box,
        medium=mediums["fluid_medium"],
        name="fluid_structure",
    )

    solid_structure = td.Structure(
        geometry=box.updated_copy(center=(1, 1, 1)),
        medium=mediums["solid_medium"],
        name="solid_structure",
    )

    solid_struct_no_heat = td.Structure(
        geometry=box.updated_copy(center=(1, 1, 1)),
        medium=mediums["solid_no_heat"],
        name="solid_struct_no_heat",
    )

    solid_struct_no_elect = td.Structure(
        geometry=box.updated_copy(center=(1, 1, 1)),
        medium=mediums["solid_no_elect"],
        name="solid_struct_no_elect",
    )

    insulator_structure = td.Structure(
        geometry=box,
        medium=mediums["insulator_medium"],
        name="insulator_structure",
    )

    semiconductor_structure = td.Structure(
        geometry=box,
        medium=mediums["semiconductor_medium"],
        name="semiconductor_structure",
    )

    return {
        "fluid_structure": fluid_structure,
        "solid_structure": solid_structure,
        "solid_struct_no_heat": solid_struct_no_heat,
        "solid_struct_no_elect": solid_struct_no_elect,
        "insulator_structure": insulator_structure,
        "semiconductor_structure": semiconductor_structure,
    }


@pytest.fixture(scope="module")
def boundary_conditions():
    """Creates a list of boundary conditions."""
    bc_temp = td.TemperatureBC(temperature=300)
    bc_flux = td.HeatFluxBC(flux=20)
    bc_conv = td.ConvectionBC(ambient_temperature=400, transfer_coeff=0.2)
    bc_volt = td.VoltageBC(source=td.DCVoltageSource(voltage=[1]))
    bc_current = td.CurrentBC(source=td.DCCurrentSource(current=3e-1))

    return [bc_temp, bc_flux, bc_conv, bc_volt, bc_current]


@pytest.fixture(scope="module")
def monitors():
    """Creates monitors of different types and sizes."""
    temp_mnt1 = td.TemperatureMonitor(size=(1.6, 2, 3), name="test", unstructured=False)
    temp_mnt2 = td.TemperatureMonitor(size=(1.6, 2, 3), name="tet", unstructured=True)
    temp_mnt3 = td.TemperatureMonitor(
        center=(0, 0.9, 0), size=(1.6, 0, 3), name="tri", unstructured=True, conformal=True
    )
    temp_mnt4 = td.TemperatureMonitor(
        center=(0, 0.9, 0), size=(1.6, 0, 3), name="empty", unstructured=True, conformal=False
    )

    volt_mnt1 = td.SteadyPotentialMonitor(size=(1.6, 2, 3), name="v_test", unstructured=False)
    volt_mnt2 = td.SteadyPotentialMonitor(size=(1.6, 2, 3), name="v_tet", unstructured=True)
    volt_mnt3 = td.SteadyPotentialMonitor(
        center=(0, 0.9, 0), size=(1.6, 0, 3), name="v_tri", unstructured=True, conformal=True
    )
    volt_mnt4 = td.SteadyPotentialMonitor(
        center=(0, 0.9, 0), size=(1.6, 0, 3), name="v_empty", unstructured=True, conformal=False
    )

    capacitance_mnt1 = td.SteadyCapacitanceMonitor(size=(1.6, 2, 3), name="cmnt_test")

    free_carrier_mnt1 = td.SteadyFreeCarrierMonitor(size=(1.6, 2, 3), name="carrier_test")

    energy_band_mnt1 = td.SteadyEnergyBandMonitor(size=(1.6, 2, 3), name="bandgap_test")

    mesh_mnt = td.VolumeMeshMonitor(size=(1.6, 2, 3), name="mesh_test")

    electric_field_mnt = td.SteadyElectricFieldMonitor(size=(1.6, 2, 3), name="electric_field_test")

    current_density_mnt = td.SteadyCurrentDensityMonitor(
        size=(1.6, 2, 3), name="current_density_mnt"
    )

    return [
        temp_mnt1,  # 0
        temp_mnt2,  # 1
        temp_mnt3,  # 2
        temp_mnt4,  # 3
        volt_mnt1,  # 4
        volt_mnt2,  # 5
        volt_mnt3,  # 6
        volt_mnt4,  # 7
        capacitance_mnt1,  # 8
        free_carrier_mnt1,  # 9
        energy_band_mnt1,  # 10
        mesh_mnt,  # 11
        electric_field_mnt,  # 12
        current_density_mnt,  # 13
    ]


@pytest.fixture(scope="module")
def grid_specs():
    """Creates grid specifications."""
    uniform_grid = td.UniformUnstructuredGrid(
        dl=0.1, min_edges_per_circumference=5, min_edges_per_side=3
    )
    distance_grid = td.DistanceUnstructuredGrid(
        dl_interface=0.1, dl_bulk=1, distance_interface=1, distance_bulk=2
    )
    return {
        "uniform": uniform_grid,
        "distance": distance_grid,
    }


@pytest.fixture(scope="module")
def heat_simulation(mediums, structures, boundary_conditions, monitors, grid_specs):
    """Generates a heat-charge heat simulation."""
    heat_source = td.HeatSource(structures=["solid_structure"], rate=100)

    pl1 = td.HeatChargeBoundarySpec(
        condition=boundary_conditions[2],  # bc_conv
        placement=td.MediumMediumInterface(mediums=["fluid_medium", "solid_medium"]),
    )
    pl2 = td.HeatChargeBoundarySpec(
        condition=boundary_conditions[1],  # bc_flux
        placement=td.StructureBoundary(structure="solid_structure"),
    )
    pl3 = td.HeatChargeBoundarySpec(
        condition=boundary_conditions[0],  # bc_temp
        placement=td.StructureStructureInterface(structures=["fluid_structure", "solid_structure"]),
    )

    heat_sim = td.HeatChargeSimulation(
        medium=mediums["fluid_medium"],
        structures=[structures["fluid_structure"], structures["solid_structure"]],
        center=(0, 0, 0),
        size=(2, 2, 2),
        boundary_spec=[pl1, pl2, pl3],
        grid_spec=grid_specs["uniform"],
        sources=[heat_source],
        monitors=monitors[0:4],
    )

    return heat_sim


@pytest.fixture(scope="module")
def conduction_simulation(mediums, structures, boundary_conditions, monitors, grid_specs):
    """Creates a heat-charge conduction simulation."""
    pl4 = td.HeatChargeBoundarySpec(
        condition=boundary_conditions[3],  # bc_volt
        placement=td.SimulationBoundary(),
    )
    pl5 = td.HeatChargeBoundarySpec(
        condition=boundary_conditions[4],  # bc_current
        placement=td.StructureSimulationBoundary(structure="insulator_structure"),
    )

    cond_sim = td.HeatChargeSimulation(
        medium=mediums["insulator_medium"],
        structures=[structures["insulator_structure"], structures["solid_structure"]],
        center=(0, 0, 0),
        size=(2, 2, 2),
        boundary_spec=[pl4, pl5],
        grid_spec=grid_specs["uniform"],
        sources=[],
        monitors=monitors[4:8],
    )

    return cond_sim


@pytest.fixture(scope="module")
def voltage_capacitance_simulation(mediums, structures, boundary_conditions, monitors, grid_specs):
    """
    Creates a HeatChargeSimulation that focuses on voltage sweeping (for capacitance).
    Specifically, we define a voltage BC with multiple voltage values (an array)
    so that 'SteadyCapacitanceMonitor' can compute capacitance over this array.
    """
    # We will define our own VoltageBC with an array of voltages for the sweep
    voltage_bc_array = td.VoltageBC(
        source=td.DCVoltageSource(voltage=[0.0, 1.0, 2.0]),
    )

    # For illustration, we can reuse the insulator structure as background (like conduction).
    # Suppose we want to set the voltage array at the simulation boundary
    pl6 = td.HeatChargeBoundarySpec(
        condition=voltage_bc_array,
        placement=td.SimulationBoundary(),
    )

    # We can optionally define a second boundary condition if desired, e.g. an insulating BC:
    bc_insulating = td.InsulatingBC()
    pl7 = td.HeatChargeBoundarySpec(
        condition=bc_insulating,
        placement=td.StructureBoundary(structure="semiconductor_structure"),
    )

    # we need two voltage BCs for Charge simulations
    pl8 = pl7.updated_copy(
        condition=td.VoltageBC(source=td.DCVoltageSource(voltage=0)),
    )

    # Let’s pick a couple of monitors. We'll definitely include the CapacitanceMonitor
    # (monitors[8] -> 'cap_mt1') so that we can measure capacitance. We can also include
    # a potential monitor to see the fields, e.g. monitors[4] -> volt_mnt1 for demonstration.
    cap_monitor = monitors[8]  # 'capacitance_mnt1'
    volt_monitor = monitors[4]  # 'volt_mnt1'
    chosen_monitors = [cap_monitor, volt_monitor]

    # Build a new HeatChargeSimulation
    voltage_cap_sim = td.HeatChargeSimulation(
        medium=mediums["insulator_medium"],
        structures=[structures["insulator_structure"], structures["semiconductor_structure"]],
        center=(0, 0, 0),
        size=(2, 2, 2),
        boundary_spec=[pl6, pl7, pl8],
        grid_spec=grid_specs["uniform"],
        sources=[],
        monitors=chosen_monitors,
    )

    return voltage_cap_sim


@pytest.fixture(scope="module")
def current_voltage_simulation(mediums, structures, boundary_conditions, monitors, grid_specs):
    """
    Creates a HeatChargeSimulation for a scenario combining a current BC and a voltage BC.
    This can be used to measure conduction properties and free carriers with different
    monitors, e.g. potential monitors and free carrier monitors.
    """
    # We'll reuse bc_volt=boundary_conditions[3] and bc_current=boundary_conditions[4]
    bc_volt = boundary_conditions[3]  # VoltageBC(source=td.DCVoltageSource(voltage=[1]))
    bc_current = boundary_conditions[4]  # CurrentBC(source=td.DCCurrentSource(current=3e-1))

    # Place the voltage BC at the simulation boundary
    pl6 = td.HeatChargeBoundarySpec(
        condition=bc_volt,
        placement=td.SimulationBoundary(),
    )
    # Place the current BC at the boundary of the "insulator_structure" (arbitrary choice here)
    pl7 = td.HeatChargeBoundarySpec(
        condition=bc_current,
        placement=td.StructureBoundary(structure="insulator_structure"),
    )

    # Pick a voltage monitor and a free carrier monitor
    # e.g., monitors[5] -> 'volt_mnt2', monitors[9] -> 'free_carrier_mnt1'
    volt_monitor = monitors[4]
    free_carrier_monitor = monitors[9]
    chosen_monitors = [volt_monitor, free_carrier_monitor]

    current_volt_sim = td.HeatChargeSimulation(
        medium=mediums["insulator_medium"],
        structures=[structures["insulator_structure"], structures["solid_structure"]],
        center=(0, 0, 0),
        size=(2, 2, 2),
        boundary_spec=[pl6, pl7],
        grid_spec=grid_specs["uniform"],
        sources=[],
        monitors=chosen_monitors,
    )

    return current_volt_sim


@pytest.fixture(scope="module")
def temperature_monitor_data(monitors):
    """Creates different temperature monitor data."""
    temp_mnt1, temp_mnt2, temp_mnt3, temp_mnt4, *_ = monitors

    # SpatialDataArray
    nx, ny, nz = 9, 6, 5
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 2, ny)
    z = np.linspace(0, 3, nz)
    T = np.random.default_rng().uniform(300, 350, (nx, ny, nz))
    coords = {"x": x, "y": y, "z": z}
    temperature_field = td.SpatialDataArray(T, coords=coords)

    mnt_data1 = td.TemperatureData(monitor=temp_mnt1, temperature=temperature_field)

    # TetrahedralGridDataset
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
        dims=("index",),
        name="T",
    )

    tet_grid = td.TetrahedralGridDataset(
        points=tet_grid_points,
        cells=tet_grid_cells,
        values=tet_grid_values,
    )

    mnt_data2 = td.TemperatureData(monitor=temp_mnt2, temperature=tet_grid)

    # TriangularGridDataset
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
        dims=("index",),
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

    default_field_name = mnt_data3.field_name()
    target_field_name = mnt_data3.field_name("abs^2")
    assert default_field_name is not None
    assert target_field_name is not None

    return (mnt_data1, mnt_data2, mnt_data3, mnt_data4)


@pytest.fixture(scope="module")
def voltage_monitor_data(monitors):
    """Creates different voltage monitor data."""
    volt_mnt1 = monitors[4]
    volt_mnt2 = monitors[5]
    volt_mnt3 = monitors[6]
    volt_mnt4 = monitors[7]

    # SpatialDataArray
    nx, ny, nz = 9, 6, 5
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 2, ny)
    z = np.linspace(0, 3, nz)
    T = np.random.default_rng().uniform(-5, 5, (nx, ny, nz))
    coords = {"x": x, "y": y, "z": z}
    voltage_field = td.SpatialDataArray(T, coords=coords)

    mnt_data1 = td.SteadyPotentialData(monitor=volt_mnt1, potential=voltage_field)

    # TetrahedralGridDataset
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
        dims=("index",),
        name="T",
    )

    tet_grid = td.TetrahedralGridDataset(
        points=tet_grid_points,
        cells=tet_grid_cells,
        values=tet_grid_values,
    )

    mnt_data2 = td.SteadyPotentialData(monitor=volt_mnt2, potential=tet_grid)

    # TriangularGridDataset
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
        dims=("index",),
        name="T",
    )

    tri_grid = td.TriangularGridDataset(
        normal_axis=1,
        normal_pos=0,
        points=tri_grid_points,
        cells=tri_grid_cells,
        values=tri_grid_values,
    )

    mnt_data3 = td.SteadyPotentialData(monitor=volt_mnt3, potential=tri_grid)

    mnt_data4 = td.SteadyPotentialData(monitor=volt_mnt4, potential=None)

    return (mnt_data1, mnt_data2, mnt_data3, mnt_data4)


@pytest.fixture(scope="module")
def capacitance_monitor_data(monitors):
    """Creates different voltage monitor data."""
    cap_mt1 = monitors[8]

    # SpatialDataArray
    cap_data1 = td.SteadyCapacitanceData(monitor=cap_mt1)
    cap_data2 = cap_data1.symmetry_expanded_copy

    return (cap_data1,)


@pytest.fixture(scope="module")
def mesh_monitor_data(monitors):
    """Creates different voltage monitor data."""
    mesh_mnt = monitors[11]

    # TetrahedralGridDataset
    tet_grid_points = td.PointDataArray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dims=("index", "axis"),
    )

    tet_grid_cells = td.CellDataArray(
        [[0, 1, 2, 4], [1, 2, 3, 4]],
        dims=("cell_index", "vertex_index"),
    )

    tet_grid_values = td.IndexedDataArray(
        np.zeros((tet_grid_points.shape[0],)),
        dims=("index",),
        name="Mesh",
    )

    tet_grid = td.TetrahedralGridDataset(
        points=tet_grid_points,
        cells=tet_grid_cells,
        values=tet_grid_values,
    )

    # SpatialDataArray
    mesh_data = td.VolumeMeshData(monitor=mesh_mnt, mesh=tet_grid)

    return (mesh_data,)


@pytest.fixture(scope="module")
def free_carrier_monitor_data(monitors):
    """Creates different voltage monitor data."""
    fc_mnt = monitors[9]

    # SpatialDataArray
    fc_data1 = td.SteadyFreeCarrierData(monitor=fc_mnt)
    fc_data2 = fc_data1.symmetry_expanded_copy
    assert fc_data2 is not None

    field_components = fc_data1.field_components

    fc_fields = fc_data1.field_name("abs^2")
    assert fc_fields is not None
    fc_fields_default = fc_data1.field_name()
    assert fc_fields_default is not None

    assert field_components is not None

    return (fc_data1,)


@pytest.fixture(scope="module")
def energy_band_monitor_data(monitors):
    """Creates different voltage monitor data."""
    eb_mnt = monitors[10]

    # SpatialDataArray
    eb_data1 = td.SteadyEnergyBandData(monitor=eb_mnt)
    eb_data2 = eb_data1.symmetry_expanded_copy
    assert eb_data2 is not None

    field_components = eb_data1.field_components

    eb_fields = eb_data1.field_name("abs^2")
    assert eb_fields is not None
    eb_fields_default = eb_data1.field_name()
    assert eb_fields_default is not None

    assert field_components is not None

    return (eb_data1,)


@pytest.fixture(scope="module")
def electric_field_monitor_data(monitors):
    """Creates different electric field monitor data."""
    monitor = monitors[12]

    # TetrahedralGridDataset
    tet_grid_points = td.PointDataArray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dims=("index", "axis"),
    )

    tet_grid_cells = td.CellDataArray(
        [[0, 1, 2, 4], [1, 2, 3, 4]],
        dims=("cell_index", "vertex_index"),
    )

    tet_grid_values = td.PointDataArray(
        [[0.0, 1.0, 0.0], [1.0, 1.0, 1.0], [3.0, 5.0, 1.0], [4.0, 5.0, 3.0], [5.0, 2.0, 1.0]],
        dims=(
            "index",
            "axis",
        ),
        name="T",
    )

    tet_grid = td.TetrahedralGridDataset(
        points=tet_grid_points,
        cells=tet_grid_cells,
        values=tet_grid_values,
    )

    mnt_data1 = td.SteadyElectricFieldData(monitor=monitor, E=tet_grid)

    # TriangularGridDataset
    tri_grid_points = td.PointDataArray(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        dims=("index", "axis"),
    )

    tri_grid_cells = td.CellDataArray(
        [[0, 1, 2], [1, 2, 3]],
        dims=("cell_index", "vertex_index"),
    )

    tri_grid_values = td.IndexedFieldVoltageDataArray(
        [
            [[1.0, 1.5], [-1.0, 1.1], [5.1, 0.0]],
            [[1.0, 1.5], [-1.0, 1.1], [5.1, 0.0]],
            [[1.0, 1.5], [-1.0, 1.1], [5.1, 0.0]],
            [[1.0, 1.5], [-1.0, 1.1], [5.1, 0.0]],
        ],
        coords={"index": np.arange(4), "axis": np.arange(3), "voltage": [-1, 1]},
        name="T",
    )

    tri_grid = td.TriangularGridDataset(
        normal_axis=1,
        normal_pos=0,
        points=tri_grid_points,
        cells=tri_grid_cells,
        values=tri_grid_values,
    )

    mnt_data2 = td.SteadyElectricFieldData(monitor=monitor, E=tri_grid)

    mnt_data3 = td.SteadyElectricFieldData(monitor=monitor, E=None)

    return (mnt_data1, mnt_data2, mnt_data3)


@pytest.fixture(scope="module")
def current_density_monitor_data(monitors, electric_field_monitor_data):
    """Creates different current density monitor data."""
    monitor = monitors[13]
    e_data1, e_data2, e_data3 = electric_field_monitor_data

    mnt_data1 = td.SteadyCurrentDensityData(monitor=monitor, J=e_data1.E)
    mnt_data2 = td.SteadyCurrentDensityData(monitor=monitor, J=e_data2.E)
    mnt_data3 = td.SteadyCurrentDensityData(monitor=monitor, J=e_data3.E)

    return (mnt_data1, mnt_data2, mnt_data3)


@pytest.fixture(scope="module")
def simulation_data(
    heat_simulation,
    conduction_simulation,
    voltage_capacitance_simulation,
    current_voltage_simulation,
    temperature_monitor_data,
    voltage_monitor_data,
    capacitance_monitor_data,
    free_carrier_monitor_data,
    energy_band_monitor_data,
    mesh_monitor_data,
):
    """Creates 'HeatChargeSimulationData' for both Heat and Conduction simulations."""
    heat_sim_data = td.HeatChargeSimulationData(
        simulation=heat_simulation,
        data=temperature_monitor_data,
    )

    cond_sim_data = td.HeatChargeSimulationData(
        simulation=conduction_simulation,
        data=voltage_monitor_data,
    )

    voltage_capacitance_sim_data = td.HeatChargeSimulationData(
        simulation=voltage_capacitance_simulation,
        data=(capacitance_monitor_data[0], voltage_monitor_data[0]),
    )

    current_voltage_sim_data = td.HeatChargeSimulationData(
        simulation=current_voltage_simulation,
        data=(voltage_monitor_data[0], free_carrier_monitor_data[0]),
    )

    mesh_monitor = mesh_monitor_data[0].monitor
    mesh_data = td.VolumeMesherData(
        simulation=conduction_simulation,
        data=mesh_monitor_data,
        monitors=[mesh_monitor],
    )

    return [
        heat_sim_data,
        cond_sim_data,
        voltage_capacitance_sim_data,
        current_voltage_sim_data,
        mesh_data,
    ]


# --------------------------
# Test Functions
# --------------------------


def test_heat_charge_medium_validation(mediums):
    """Tests validation errors for mediums."""
    solid_medium = mediums["solid_medium"]

    # Test invalid capacity
    with pytest.raises(ValidationError):
        solid_medium.heat_spec.updated_copy(capacity=-1)

    # Test invalid conductivity
    with pytest.raises(ValidationError):
        solid_medium.heat_spec.updated_copy(conductivity=-1)

    # Test invalid charge conductivity
    with pytest.raises(ValidationError):
        solid_medium.charge.updated_copy(conductivity=-1)


def test_constant_mobility():
    constant_mobility = td.ConstantMobilityModel(mu=1500)

    with pytest.raises(ValidationError):
        _ = constant_mobility.updated_copy(mu=-1)


def test_heat_charge_structures_creation(structures):
    """Tests that different structures with different mediums can be created."""
    fluid_structure = structures["fluid_structure"]
    solid_structure = structures["solid_structure"]
    solid_struct_no_heat = structures["solid_struct_no_heat"]
    solid_struct_no_elect = structures["solid_struct_no_elect"]
    insulator_structure = structures["insulator_structure"]

    assert fluid_structure.medium.name == "fluid_medium"
    assert solid_structure.medium.name == "solid_medium"
    assert solid_struct_no_heat.medium.name == "solid_no_heat"
    assert solid_struct_no_elect.medium.name == "solid_no_elect"
    assert insulator_structure.medium.name == "insulator_medium"


def test_heat_charge_bcs_validation(boundary_conditions):
    """Tests the validators for boundary conditions."""
    _bc_temp, _bc_flux, _bc_conv, _bc_volt, _bc_current = boundary_conditions

    # Invalid TemperatureBC
    with pytest.raises(ValidationError):
        td.TemperatureBC(temperature=-10)

    # Invalid ConvectionBC: negative ambient temperature
    with pytest.raises(ValidationError):
        td.ConvectionBC(ambient_temperature=-400, transfer_coeff=0.2)

    # Invalid ConvectionBC: negative transfer coefficient
    with pytest.raises(ValidationError):
        td.ConvectionBC(ambient_temperature=400, transfer_coeff=-0.2)

    # Invalid VoltageBC: infinite voltage
    with pytest.raises(ValidationError):
        td.VoltageBC(source=td.DCVoltageSource(voltage=[td.inf]))

    # Invalid CurrentBC: infinite current density
    with pytest.raises(ValidationError):
        td.CurrentBC(source=td.DCCurrentSource(current=td.inf))

    with pytest.raises(ValidationError):
        td.VoltageBC(source=td.DCVoltageSource(voltage=np.array([td.inf, 0, 1])))

    # Invalid SSACVoltageSource: infinite voltage
    with pytest.raises(ValidationError):
        td.VoltageBC(source=td.SSACVoltageSource(voltage=np.array([td.inf, 0, 1]), amplitude=1e-2))


def test_repeated_voltage_warning():
    """Test that a warning is raised when repeated voltage values are present."""
    # No warning for unique values
    with AssertLogLevel(None):
        td.DCVoltageSource(voltage=[0, 1, 2, 3])

    # Warning for repeated values
    with AssertLogLevel("WARNING"):
        td.DCVoltageSource(voltage=[1, 2, 2, 3])

    # Warning for 0 and -0 (treated as duplicates)
    with AssertLogLevel("WARNING"):
        td.DCVoltageSource(voltage=[0.0, -0.0, 1, 2])

    # Warning for multiple repeated values
    with AssertLogLevel("WARNING"):
        td.DCVoltageSource(voltage=[1, 1, 2, 2, 3])


def test_freqs_validation():
    """Test validation that freqs requires SSACVoltageSource."""
    solid_box_1 = td.Box(center=(0, 0, 0), size=(2, 2, 2))
    solid_box_2 = td.Box(center=(1, 1, 1), size=(2, 2, 2))
    solid_box_3 = td.Box(center=(2, 2, 2), size=(2, 2, 2))

    metal_medium = td.MultiPhysicsMedium(
        heat=td.SolidMedium(conductivity=1, capacity=1),
        charge=td.ChargeConductorMedium(conductivity=1),
        name="metal",
    )

    cathode = td.Structure(
        geometry=solid_box_1,
        medium=metal_medium,
        name="cathode",
    )
    silicon = td.Structure(
        geometry=solid_box_2,
        medium=CHARGE_SIMULATION.intrinsic_Si,
        name="silicon",
    )
    anode = td.Structure(
        geometry=solid_box_3,
        medium=metal_medium,
        name="anode",
    )
    structures = [cathode, silicon, anode]

    volt_monitor = td.SteadyPotentialMonitor(
        center=(0, 0, 0), size=(td.inf, td.inf, td.inf), name="voltage", unstructured=False
    )

    charge_tolerance = td.ChargeToleranceSpec(rel_tol=1e5, abs_tol=1e3, max_iters=400)
    freqs_input = [1e3, 1e4, 1e5]
    isothermal_spec = td.IsothermalSSACAnalysis(
        temperature=300,
        tolerance_settings=charge_tolerance,
        fermi_dirac=True,
        freqs=freqs_input,
    )

    # Test that freqs with SSACVoltageSource works
    ssac_source = td.SSACVoltageSource(voltage=[0, 1, 2], amplitude=1e-3)
    sim = td.HeatChargeSimulation(
        size=(8, 8, 8),
        center=(0, 0, 0),
        structures=structures,
        boundary_spec=[
            td.HeatChargeBoundarySpec(
                placement=td.StructureStructureInterface(structures=["anode", "silicon"]),
                condition=td.VoltageBC(source=ssac_source),
            ),
            td.HeatChargeBoundarySpec(
                placement=td.StructureStructureInterface(structures=["cathode", "silicon"]),
                condition=td.VoltageBC(source=td.GroundVoltage()),
            ),
        ],
        grid_spec=td.UniformUnstructuredGrid(dl=0.1),
        monitors=[volt_monitor],
        analysis_spec=isothermal_spec,
    )

    # Test that freqs without SSACVoltageSource raises error
    with pytest.raises(
        ValidationError,
        match=r"If 'freqs' is provided and not empty, at least one 'SSACVoltageSource' must be present in the boundary conditions.",
    ):
        sim.updated_copy(
            boundary_spec=[
                td.HeatChargeBoundarySpec(
                    placement=td.StructureStructureInterface(structures=["cathode", "silicon"]),
                    condition=td.VoltageBC(source=td.DCVoltageSource(voltage=0)),
                ),
            ],
        )

    # test the getter function
    freqs, amplitude = sim._get_ssac_frequency_and_amplitude()
    assert np.isclose(freqs, freqs_input).all()
    assert np.isclose(1e-3, amplitude)

    with pytest.raises(ValidationError, match=r"'freqs' cannot contain infinite frequencies."):
        sim.updated_copy(analysis_spec=sim.analysis_spec.updated_copy(freqs=[1e2, np.inf]))

    with pytest.raises(ValidationError, match=r"'freqs' cannot contain negative frequencies."):
        sim.updated_copy(analysis_spec=sim.analysis_spec.updated_copy(freqs=[1e2, -1e2]))


def test_vertical_natural_convection():
    solid_box_l = td.Box(center=(0, 0, 0), size=(2, 2, 2))
    solid_box_r = td.Box(center=(1, 1, 1), size=(2, 2, 2))
    fluid_box_r = td.Box(center=(1, 1, 1), size=(2, 2, 2))

    solid_medium = td.MultiPhysicsMedium(
        heat=td.SolidMedium(conductivity=1, capacity=1), name="solid"
    )
    air = td.MultiPhysicsMedium(
        heat=td.FluidMedium.from_si_units(
            thermal_conductivity=0.026,
            viscosity=1.8e-5,
            specific_heat=1005,
            density=1.2,
            expansivity=1 / 300.0,
        ),
        name="air",
    )
    solid_structure_l = td.Structure(
        geometry=solid_box_l,
        medium=solid_medium,
        name="solid_l",
    )
    solid_structure_r = td.Structure(
        geometry=solid_box_r,
        medium=solid_medium,
        name="solid_r",
    )
    fluid_structure_r = td.Structure(
        geometry=fluid_box_r,
        medium=air,
        name="fluid_r",
    )

    coeff_model = td.VerticalNaturalConvectionCoeffModel(plate_length=1)
    sim = td.HeatChargeSimulation(
        size=(2, 2, 2),
        center=(0, 0, 0),
        medium=td.MultiPhysicsMedium(heat=td.FluidMedium()),
        structures=[solid_structure_l, fluid_structure_r],
        boundary_spec=[
            td.HeatBoundarySpec(
                placement=td.MediumMediumInterface(mediums=["air", "solid"]),
                condition=td.ConvectionBC(ambient_temperature=300, transfer_coeff=coeff_model),
            )
        ],
        grid_spec=td.UniformUnstructuredGrid(dl=0.1),
        monitors=[
            td.TemperatureMonitor(
                center=(0, 0, 0),
                size=(td.inf, td.inf, td.inf),
                name="test_monitor",
                unstructured=True,
            )
        ],
    )

    # Test that the model can be placed on an interface defined by structures
    sim.updated_copy(
        boundary_spec=[
            td.HeatBoundarySpec(
                placement=td.StructureStructureInterface(structures=["solid_l", "fluid_r"]),
                condition=td.ConvectionBC(ambient_temperature=300, transfer_coeff=coeff_model),
            )
        ],
    )

    # Verify that placing the model on an interface between two solid media
    # correctly raises a validation error.
    with pytest.raises(ValidationError):
        sim.updated_copy(
            structures=[solid_structure_l, solid_structure_r],
            boundary_spec=[
                td.HeatBoundarySpec(
                    placement=td.StructureStructureInterface(structures=["solid_l", "solid_r"]),
                    condition=td.ConvectionBC(ambient_temperature=300, transfer_coeff=coeff_model),
                )
            ],
        )

    # Verify that using a fluid medium with incomplete physical properties
    # for the natural convection calculation raises a validation error.
    incomplete_air = td.MultiPhysicsMedium(
        heat=td.FluidMedium(expansivity=1 / 300.0), name="incomplete_air"
    )
    with pytest.raises(ValidationError):
        new_fluid_structure_r = fluid_structure_r.updated_copy(medium=incomplete_air)
        sim.updated_copy(
            structures=[solid_structure_l, new_fluid_structure_r],
            boundary_spec=[
                td.HeatBoundarySpec(
                    placement=td.MediumMediumInterface(mediums=["incomplete_air", "solid"]),
                    condition=td.ConvectionBC(ambient_temperature=300, transfer_coeff=coeff_model),
                )
            ],
        )

    # Test the case where the convection model has its own fluid medium explicitly defined.
    # The simulation should use the properties from the model's medium and ignore the
    # fluid present at the interface.
    full_coeff_model = td.VerticalNaturalConvectionCoeffModel(medium=air.heat, plate_length=1)
    sim.updated_copy(
        boundary_spec=[
            td.HeatBoundarySpec(
                placement=td.StructureStructureInterface(structures=["solid_l", "fluid_r"]),
                condition=td.ConvectionBC(ambient_temperature=300, transfer_coeff=full_coeff_model),
            )
        ],
    )

    # Verify that a validation error is raised if the medium supplied directly to the
    # coefficient model has incomplete properties for the natural convection calculation.
    incomplete_coeff_model = coeff_model.updated_copy(medium=incomplete_air.heat)
    with pytest.raises(ValidationError):
        sim.updated_copy(
            boundary_spec=[
                td.HeatBoundarySpec(
                    placement=td.MediumMediumInterface(mediums=["air", "solid"]),
                    condition=td.ConvectionBC(
                        ambient_temperature=300, transfer_coeff=incomplete_coeff_model
                    ),
                ),
            ]
        )


def test_heat_charge_monitors_validation(monitors):
    """Checks for no name and negative size in monitors."""
    temp_mnt = monitors[0]
    mesh_mnt = monitors[11]

    # Invalid monitor name
    with pytest.raises(ValidationError):
        temp_mnt.updated_copy(name=None)

    # Invalid monitor size (negative dimension)
    with pytest.raises(ValidationError):
        temp_mnt.updated_copy(size=(-1, 2, 3))

    # Mesh monitor 1D
    with pytest.raises(ValidationError):
        mesh_mnt.updated_copy(size=(0, 1, 0))


def test_monitor_crosses_medium(mediums, structures, heat_simulation, conduction_simulation):
    """Tests whether monitor crosses structures with relevant material specifications."""
    solid_no_heat = mediums["solid_no_heat"]
    solid_no_elect = mediums["solid_no_elect"]
    solid_struct_no_heat = structures["solid_struct_no_heat"]
    solid_struct_no_elect = structures["solid_struct_no_elect"]

    # Voltage monitor
    volt_monitor = td.SteadyPotentialMonitor(
        center=(0, 0, 0), size=(td.inf, td.inf, td.inf), name="voltage", unstructured=False
    )
    # A voltage monitor in a heat simulation should throw error if no ChargeConductorMedium is present
    with pytest.raises(ValidationError):
        heat_simulation.updated_copy(
            medium=solid_no_elect, structures=(solid_struct_no_elect,), monitors=(volt_monitor,)
        )

    # Temperature monitor
    temp_monitor = td.TemperatureMonitor(
        center=(0, 0, 0), size=(td.inf, td.inf, td.inf), name="temperature", unstructured=False
    )
    # A temperature monitor should throw error in a conduction simulation if no SolidSpec is present
    with pytest.raises(ValidationError):
        conduction_simulation.updated_copy(
            medium=solid_no_heat, structures=(solid_struct_no_heat,), monitors=(temp_monitor,)
        )

    # check error is raised in voltage monitor doesn't cross a conducting medium
    with pytest.raises(ValidationError):
        volt_mnt = td.SteadyPotentialMonitor(
            center=(0, 0, 0), size=(0, td.inf, td.inf), unstructured=False
        )
        _ = conduction_simulation.updated_copy(monitors=(volt_mnt,))


def test_heat_charge_mnt_data(
    temperature_monitor_data,
    voltage_monitor_data,
    electric_field_monitor_data,
    current_density_monitor_data,
):
    """Tests whether different heat-charge monitor data can be created."""
    assert len(temperature_monitor_data) == 4, "Expected 4 temperature monitor data entries."
    assert len(voltage_monitor_data) == 4, "Expected 4 voltage monitor data entries."
    assert len(electric_field_monitor_data) == 3, "Expected 3 electric field monitor data entries."
    assert len(current_density_monitor_data) == 3, (
        "Expected 3 current density monitor data entries."
    )

    for var, mnt_data_lists in [
        ("E", electric_field_monitor_data),
        ("J", current_density_monitor_data),
    ]:
        for mnt_data in mnt_data_lists:
            assert var in mnt_data.field_components.keys()

            symm_data = mnt_data.symmetry_expanded_copy
            if var == "E":
                assert symm_data.E == mnt_data.E
            elif var == "J":
                assert symm_data.J == mnt_data.J

            names = mnt_data.field_name("abs^2")
            assert names == var + "²"
            names = mnt_data.field_name()
            assert names == var

            # make sure an error is raised if we don't use a field data array
            # TriangularGridDataset
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
                dims=("index",),
                name="T",
            )

            tri_grid = td.TriangularGridDataset(
                normal_axis=1,
                normal_pos=0,
                points=tri_grid_points,
                cells=tri_grid_cells,
                values=tri_grid_values,
            )

            with pytest.raises(ValidationError):
                if var == "E":
                    _ = mnt_data.updated_copy(E=tri_grid)
                elif var == "J":
                    _ = mnt_data.updated_copy(J=tri_grid)


def test_grid_spec_validation(grid_specs):
    """Tests whether unstructured grids can be created and different validators for them."""
    # Test UniformUnstructuredGrid
    uniform_grid = grid_specs["uniform"]
    with pytest.raises(ValidationError):
        uniform_grid.updated_copy(dl=0)
    with pytest.raises(ValidationError):
        uniform_grid.updated_copy(min_edges_per_circumference=-1)
    with pytest.raises(ValidationError):
        uniform_grid.updated_copy(min_edges_per_side=-1)

    # Test DistanceUnstructuredGrid
    distance_grid = grid_specs["distance"]
    with pytest.raises(ValidationError):
        distance_grid.updated_copy(dl_interface=-1)
    with pytest.raises(ValidationError):
        distance_grid.updated_copy(distance_interface=2, distance_bulk=1)
    with pytest.raises(ValidationError) as excinfo:
        _ = td.GridRefinementRegion(
            center=(0, 0, 0),
            size=(1, 0, 0),
            dl_internal=0.1,
            transition_thickness=0.2,
        )
    assert_single_value_error_loc(excinfo, ("size",), "volumetric or planar")
    with pytest.raises(ValidationError) as excinfo:
        _ = td.GridRefinementLine(
            r1=(0, 0, 0),
            r2=(1e-7, 0, 0),
            dl_near=0.1,
            distance_near=0.2,
            distance_bulk=0.4,
        )
    assert_single_value_error_loc(excinfo, ("r2",), "line length must be greater than")


def test_min_mesh_size(grid_specs):
    """Tests the min_mesh_size property for unstructured grids."""
    # UniformUnstructuredGrid: min_mesh_size is simply dl
    uniform_grid = grid_specs["uniform"]
    assert uniform_grid.min_mesh_size == uniform_grid.dl

    # DistanceUnstructuredGrid without refinements: min_mesh_size is dl_interface
    distance_grid = grid_specs["distance"]
    assert distance_grid.min_mesh_size == distance_grid.dl_interface

    # DistanceUnstructuredGrid with refinement region smaller than dl_interface
    region = td.GridRefinementRegion(
        center=(0, 0, 0),
        size=(1, 1, 1),
        dl_internal=0.01,
        transition_thickness=0.5,
    )
    grid_with_region = distance_grid.updated_copy(mesh_refinements=[region])
    assert grid_with_region.min_mesh_size == region.dl_internal

    # DistanceUnstructuredGrid with refinement region larger than dl_interface
    large_region = td.GridRefinementRegion(
        center=(0, 0, 0),
        size=(1, 1, 1),
        dl_internal=0.5,
        transition_thickness=0.5,
    )
    grid_with_large_region = distance_grid.updated_copy(mesh_refinements=[large_region])
    assert grid_with_large_region.min_mesh_size == distance_grid.dl_interface


def test_device_characteristics():
    C = [0, 1, 4]
    V = [-1, -0.5, 0]
    intensities = [0.1, 1.5, 3.6]
    capacitance = td.SteadyVoltageDataArray(data=C, coords={"v": V})
    current_voltage = td.SteadyVoltageDataArray(data=intensities, coords={"v": V})
    _ = td.DeviceCharacteristics(
        steady_dc_hole_capacitance=capacitance,
        steady_dc_electron_capacitance=capacitance,
        steady_dc_current_voltage=current_voltage,
        steady_dc_resistance_voltage=current_voltage,
    )


def test_heat_charge_sources(structures):
    """Tests whether heat-charge sources can be created and associated warnings."""
    # this shouldn't issue warning
    with AssertLogLevel(None):
        _ = td.HeatSource(structures=["solid_structure"], rate=100)

    # this should issue warning
    with AssertLogLevel("WARNING"):
        _ = td.UniformHeatSource(structures=["solid_structure"], rate=100)

    # this shouldn't issue warning but rate is a string, assuming it's allowed
    with AssertLogLevel(None):
        _ = td.HeatSource(structures=["solid_structure"], rate="100")


def test_heat_charge_simulation(simulation_data):
    """Tests 'HeatChargeSimulation' and 'ConductionSimulation' objects."""
    (
        heat_sim_data,
        cond_sim_data,
        voltage_capacitance_sim_data,
        current_voltage_simulation_data,
        mesh_data,
    ) = simulation_data

    # Test Heat Simulation
    heat_sim = heat_sim_data.simulation
    assert heat_sim is not None, "Heat simulation should be created successfully."

    # Test Conduction Simulation
    cond_sim = cond_sim_data.simulation
    assert cond_sim is not None, "Conduction simulation should be created successfully."

    voltage_capacitance_sim = voltage_capacitance_sim_data.simulation
    assert voltage_capacitance_sim is not None, (
        "Voltage-Capacitance simulation should be created successfully."
    )

    current_voltage_sim = current_voltage_simulation_data.simulation
    assert current_voltage_sim is not None, (
        "Current-Voltage simulation should be created successfully."
    )

    mesher = mesh_data.mesher
    assert mesher is not None, "VolumeMesher should be created successfully."


def test_sim_data_plotting(simulation_data):
    """Tests whether simulation data can be plotted and appropriate errors are raised."""
    heat_sim_data, cond_sim_data, _cap_sim_data, _fc_sim_data, _mesh_data = simulation_data

    # Plotting temperature data
    heat_sim_data.plot_field("test", z=0)
    heat_sim_data.plot_field("tri")
    heat_sim_data.plot_field("tet", y=0.5)

    # grid=True on structured data should raise a helpful error
    with pytest.raises(DataError, match=r"only supported for unstructured"):
        heat_sim_data["test"].temperature.sel(z=0).plot(grid=True)

    # Plotting voltage data
    cond_sim_data.plot_field("v_test", z=0)
    cond_sim_data.plot_field("v_tri")
    cond_sim_data.plot_field("v_tet", y=0.5)
    plt.close()

    # Test plotting with no data
    with pytest.raises(DataError):
        heat_sim_data.plot_field("empty")

    # Test plotting with 3D data
    with pytest.raises(DataError):
        heat_sim_data.plot_field("test")

    # Test plotting with invalid key
    with pytest.raises(KeyError):
        heat_sim_data.plot_field("test3", x=0)

    # Test updating simulation data with duplicate data
    with pytest.raises(ValidationError):
        heat_sim_data.updated_copy(data=(heat_sim_data.data[0],) * 2)

    # Test updating simulation data with invalid simulation
    temp_mnt = td.TemperatureMonitor(size=(1, 2, 3), name="test", unstructured=False)
    temp_mnt = temp_mnt.updated_copy(name="test2")

    sim = heat_sim_data.simulation.updated_copy(monitors=(temp_mnt,))

    with pytest.raises(ValidationError):
        heat_sim_data.updated_copy(simulation=sim)


def test_mesh_plotting(simulation_data):
    """Tests whether mesh can be plotted and appropriate errors are raised."""
    heat_sim_data, cond_sim_data, _cap_sim_data, _fc_sim_data, mesh_data = simulation_data

    # Plotting mesh from unstructured temperature data
    heat_sim_data.plot_mesh("tri")
    heat_sim_data.plot_mesh("tri", y=0)  # redundant normal-axis sel should be a no-op
    heat_sim_data.plot_mesh("tet", y=0.5)

    # Plotting mesh from unstructured voltage data
    cond_sim_data.plot_mesh("v_tri", structures_fill=False)
    cond_sim_data.plot_mesh("v_tet", y=0.5)

    # Plotting mesh from mesh data
    mesh_data.plot_mesh("mesh_test", z=0)

    plt.close()

    # Test plotting from structured data
    with pytest.raises(DataError):
        heat_sim_data.plot_mesh("test")

    # Test plotting with no data
    with pytest.raises(DataError):
        heat_sim_data.plot_mesh("empty")

    # Test plotting with 3D data
    with pytest.raises(DataError):
        heat_sim_data.plot_mesh("tet")

    # Test plotting with invalid key
    with pytest.raises(KeyError):
        heat_sim_data.plot_mesh("test3", x=0)

    # Test plotting with invalid field_name
    with pytest.raises(DataError):
        mesh_data.plot_mesh("mesh_test", z=0, field_name="wrong")


def test_plot_mesh_2d_auto_sel():
    """Test that plot_mesh auto-selects the collapsed dimension for 2D simulations."""

    solid_medium = td.Medium(
        permittivity=2.0,
        heat_spec=td.SolidSpec(conductivity=1, capacity=1),
        name="solid",
    )
    fluid_medium = td.Medium(permittivity=3.0, heat_spec=td.FluidSpec(), name="fluid")
    structure = td.Structure(
        geometry=td.Box(size=(0.5, 0.5, 0.5), center=(0, 0, 0)),
        medium=solid_medium,
        name="box",
    )

    temp_mnt_2d = td.TemperatureMonitor(
        center=(0, 0, 0), size=(0.8, 0, 0.8), name="temp_2d", unstructured=True
    )
    mesh_mnt_2d = td.VolumeMeshMonitor(center=(0, 0, 0), size=(0.8, 0, 0.8), name="mesh_2d")

    heat_sim_2d = td.HeatChargeSimulation(
        medium=fluid_medium,
        structures=[structure],
        center=(0, 0, 0),
        size=(1, 0, 1),
        grid_spec=td.UniformUnstructuredGrid(dl=0.1),
        sources=[td.HeatSource(rate=1, structures=["box"])],
        boundary_spec=[
            td.HeatChargeBoundarySpec(
                placement=td.StructureBoundary(structure="box"),
                condition=td.TemperatureBC(temperature=500),
            )
        ],
        monitors=[temp_mnt_2d],
    )

    tet_grid_points = td.PointDataArray(
        [
            [0.0, -0.01, 0.0],
            [0.4, -0.01, 0.0],
            [0.0, -0.01, 0.4],
            [0.4, -0.01, 0.4],
            [0.2, 0.01, 0.2],
        ],
        dims=("index", "axis"),
    )
    tet_grid_cells = td.CellDataArray(
        [[0, 1, 2, 4], [1, 2, 3, 4]],
        dims=("cell_index", "vertex_index"),
    )
    tet_grid_values = td.IndexedDataArray(
        [300.0, 310.0, 320.0, 330.0, 340.0],
        dims=("index",),
        name="T",
    )
    tet_grid = td.TetrahedralGridDataset(
        points=tet_grid_points, cells=tet_grid_cells, values=tet_grid_values
    )

    sim_data_2d = td.HeatChargeSimulationData(
        simulation=heat_sim_2d,
        data=[td.TemperatureData(monitor=temp_mnt_2d, temperature=tet_grid)],
    )

    # Should auto-detect the zero-size y dimension and slice without needing y=0
    sim_data_2d.plot_mesh("temp_2d")
    plt.close()

    # Also test plot_field auto-selection
    sim_data_2d.plot_field("temp_2d")
    plt.close()

    # VolumeMesherData case
    mesh_mnt_data = td.VolumeMeshData(monitor=mesh_mnt_2d, mesh=tet_grid)
    mesher_data_2d = td.VolumeMesherData(
        simulation=heat_sim_2d, data=[mesh_mnt_data], monitors=[mesh_mnt_2d]
    )
    mesher_data_2d.plot_mesh("mesh_2d")
    plt.close()


def test_conduction_monitors_cross_conductors(conduction_simulation, structures):
    """Conduction voltage monitors must intersect conducting materials."""

    with pytest.raises(ValidationError) as excinfo:
        _ = conduction_simulation.updated_copy(
            structures=(structures["insulator_structure"],),
        )
    assert_single_value_error_loc(
        excinfo, ("monitors", 0), "does not cross any conducting materials"
    )


def test_coupling_source(conduction_simulation, heat_simulation):
    """Test whether the coupling source can be applied."""

    with pytest.raises(ValidationError):
        _ = conduction_simulation.updated_copy(sources=(td.HeatFromElectricSource(),))

    with pytest.raises(ValidationError):
        _ = heat_simulation.updated_copy(sources=(td.HeatFromElectricSource(),))


def test_heat_charge_analysis_spec_error_loc(heat_simulation):
    with pytest.raises(ValidationError) as excinfo:
        _ = heat_simulation.updated_copy(analysis_spec=td.SSACAnalysis(freqs=[1e12]))
    assert_single_value_error_loc(excinfo, ("boundary_spec",), "SSACVoltageSource")


def test_charge_simulation_voltage_bc_error_loc(heat_simulation):
    with pytest.raises(ValidationError) as excinfo:
        _ = heat_simulation.updated_copy(
            analysis_spec=td.IsothermalSteadyChargeDCAnalysis(temperature=300)
        )
    assert_single_value_error_loc(excinfo, ("boundary_spec",), "VoltageBC")


def test_heat_charge_monitor_error_loc(heat_simulation):
    monitors = tuple(
        monitor
        for monitor in heat_simulation.monitors
        if not isinstance(monitor, td.TemperatureMonitor)
    )
    with pytest.raises(ValidationError) as excinfo:
        _ = heat_simulation.updated_copy(monitors=monitors)
    assert_single_value_error_loc(excinfo, ("monitors",), "TemperatureMonitor")


def test_conduction_sim_monitor_error_loc(conduction_simulation):
    with pytest.raises(ValidationError) as excinfo:
        _ = conduction_simulation.updated_copy(monitors=())
    assert_single_value_error_loc(excinfo, ("monitors",), "SteadyPotentialMonitor")


def test_conduction_sim_voltage_array_error_loc(conduction_simulation):
    boundary_spec = list(conduction_simulation.boundary_spec)
    for ind, bc in enumerate(boundary_spec):
        if isinstance(bc.condition, td.VoltageBC):
            boundary_spec[ind] = bc.updated_copy(
                condition=td.VoltageBC(source=td.DCVoltageSource(voltage=[0.0, 1.0]))
            )
            break
    with pytest.raises(ValidationError) as excinfo:
        _ = conduction_simulation.updated_copy(boundary_spec=tuple(boundary_spec))
    assert_single_value_error_loc(excinfo, ("boundary_spec",), "array of voltages")


# --------------------------
# Test Classes with Fixtures
# --------------------------


class TestCharge:
    """Group of tests related to charge simulations."""

    # Define semiconductor materials as fixtures within the class
    @pytest.fixture(scope="class")
    def Si_p(self):
        semiconductor = CHARGE_SIMULATION.intrinsic_Si.charge
        semiconductor = semiconductor.updated_copy(
            N_a=CHARGE_SIMULATION.acceptors,
        )
        return CHARGE_SIMULATION.intrinsic_Si.updated_copy(
            charge=semiconductor,
            heat=td.SolidMedium(conductivity=1),
            name="Si_p",
        )

    @pytest.fixture(scope="class")
    def Si_n(self):
        semiconductor = CHARGE_SIMULATION.intrinsic_Si.charge
        semiconductor = semiconductor.updated_copy(
            N_d=CHARGE_SIMULATION.donors,
        )
        return CHARGE_SIMULATION.intrinsic_Si.updated_copy(
            charge=semiconductor,
            heat=td.SolidMedium(conductivity=1),
            name="Si_n",
        )

    @pytest.fixture(scope="class")
    def SiO2(self):
        return td.MultiPhysicsMedium(
            charge=td.ChargeInsulatorMedium(permittivity=3.9),
            heat=td.SolidMedium(conductivity=2),
            name="SiO2",
        )

    # Define structures as fixtures within the class
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
                size=(CHARGE_SIMULATION.width, CHARGE_SIMULATION.height, CHARGE_SIMULATION.z_dim),
            ),
            medium=Si_p,
            name="p_side",
        )

    @pytest.fixture(scope="class")
    def n_side(self, Si_n):
        return td.Structure(
            geometry=td.Box(
                center=(CHARGE_SIMULATION.width / 2, 0, 0),
                size=(CHARGE_SIMULATION.width, CHARGE_SIMULATION.height, CHARGE_SIMULATION.z_dim),
            ),
            medium=Si_n,
            name="n_side",
        )

    # Define boundary conditions as fixtures within the class
    @pytest.fixture(scope="class")
    def bc_p(self, SiO2, Si_p):
        return td.HeatChargeBoundarySpec(
            condition=td.VoltageBC(source=td.DCVoltageSource(voltage=[0])),
            placement=td.MediumMediumInterface(mediums=[SiO2.name, Si_p.name]),
        )

    @pytest.fixture(scope="class")
    def bc_n(self, SiO2, Si_n):
        return td.HeatChargeBoundarySpec(
            condition=td.VoltageBC(source=td.DCVoltageSource(voltage=[0, 1])),
            placement=td.MediumMediumInterface(mediums=[SiO2.name, Si_n.name]),
        )

    # Define monitors as fixtures within the class
    @pytest.fixture(scope="class")
    def charge_global_mnt(self):
        return td.SteadyFreeCarrierMonitor(
            center=(0, 0, 0),
            size=(td.inf, td.inf, td.inf),
            name="charge_global_mnt",
            unstructured=True,
        )

    @pytest.fixture(scope="class")
    def potential_global_mnt(self):
        return td.SteadyPotentialMonitor(
            center=(0, 0, 0),
            size=(td.inf, td.inf, td.inf),
            name="potential_global_mnt",
            unstructured=True,
        )

    @pytest.fixture(scope="class")
    def capacitance_global_mnt(self):
        return td.SteadyCapacitanceMonitor(
            center=(0, 0, 0),
            size=(td.inf, td.inf, td.inf),
            name="capacitance_global_mnt",
            unstructured=True,
        )

    # Define charge settings as fixtures within the class
    @pytest.fixture(scope="class")
    def charge_tolerance(self):
        return td.ChargeToleranceSpec(rel_tol=1e5, abs_tol=1e3, max_iters=400)

    def test_charge_simulation(
        self,
        Si_n,
        Si_p,
        SiO2,
        oxide,
        p_side,
        n_side,
        charge_global_mnt,
        potential_global_mnt,
        capacitance_global_mnt,
        bc_n,
        bc_p,
        charge_tolerance,
    ):
        """Ensure charge simulation produces the correct errors when needed."""
        # NOTE: start tests with isothermal spec
        isothermal_spec = td.IsothermalSteadyChargeDCAnalysis(
            temperature=300,
            tolerance_settings=charge_tolerance,
            fermi_dirac=True,
        )
        sim = td.HeatChargeSimulation(
            structures=[oxide, p_side, n_side],
            medium=td.MultiPhysicsMedium(
                heat=td.FluidSpec(), charge=td.ChargeConductorMedium(conductivity=1), name="air"
            ),
            monitors=[charge_global_mnt, potential_global_mnt, capacitance_global_mnt],
            center=(0, 0, 0),
            size=CHARGE_SIMULATION.sim_size,
            grid_spec=td.UniformUnstructuredGrid(dl=0.05),
            boundary_spec=[bc_n, bc_p],
            analysis_spec=isothermal_spec,
        )

        # At least one ChargeSimulationMonitor should be added
        with pytest.raises(ValidationError):
            sim.updated_copy(monitors=())

        # At least 2 VoltageBCs should be defined
        with pytest.raises(ValidationError):
            sim.updated_copy(boundary_spec=(bc_n,))

        condition_ssac_n = td.VoltageBC(source=td.SSACVoltageSource(voltage=[0, 1], amplitude=1e-3))
        condition_ssac_p = td.VoltageBC(source=td.SSACVoltageSource(voltage=[0, 1], amplitude=1e-3))
        # Two AC sources cannot be defined
        with pytest.raises(
            ValidationError, match=r"Only a single 'SSACVoltageSource' source can be supplied."
        ):
            analysis = td.IsothermalSSACAnalysis(freqs=[1e2, 1e3], temperature=300)
            sim.updated_copy(
                boundary_spec=[
                    bc_n.updated_copy(condition=condition_ssac_n),
                    bc_p.updated_copy(condition=condition_ssac_p),
                ],
                analysis_spec=analysis,
            )

        # Test SSACAnalysis as well
        with pytest.raises(
            ValidationError, match=r"Only a single 'SSACVoltageSource' source can be supplied."
        ):
            analysis_ssac = td.SSACAnalysis(freqs=[1e2, 1e3], tolerance_settings=charge_tolerance)
            sim.updated_copy(
                boundary_spec=[
                    bc_n.updated_copy(condition=condition_ssac_n),
                    bc_p.updated_copy(condition=condition_ssac_p),
                ],
                analysis_spec=analysis_ssac,
            )

        # Define ChargeSimulation with no Semiconductor materials
        medium = td.MultiPhysicsMedium(
            charge=td.ChargeConductorMedium(permittivity=1, conductivity=1),
            name="medium",
        )
        new_structures = [struct.updated_copy(medium=medium) for struct in sim.structures]

        with pytest.raises(ValidationError):
            sim.updated_copy(structures=tuple(new_structures))

        # test a voltage array is provided when a capacitance monitor is present
        with pytest.raises(ValidationError):
            new_bc_n = bc_n.updated_copy(
                condition=td.VoltageBC(source=td.DCVoltageSource(voltage=1))
            )
            _ = sim.updated_copy(boundary_spec=(bc_p, new_bc_n))

        # test error is raised when more than one voltage array is provided
        with pytest.raises(ValidationError):
            new_bc_p = bc_p.updated_copy(
                condition=td.VoltageBC(source=td.DCVoltageSource(voltage=[1, 2]))
            )
            _ = sim.updated_copy(boundary_spec=(new_bc_p, bc_n))

        # test non isothermal spec
        non_isothermal_spec = td.SteadyChargeDCAnalysis(tolerance_settings=charge_tolerance)

        sim = sim.updated_copy(analysis_spec=non_isothermal_spec)
        with pytest.raises(ValidationError):
            # remove heat from mediums
            new_structs = []
            for struct in sim.structures:
                new_structs.append(
                    struct.updated_copy(medium=struct.medium.updated_copy(heat=None))
                )
            _ = sim.updated_copy(structures=new_structs)

        with pytest.raises(ValidationError):
            # remove charge from mediums
            new_structs = []
            for struct in sim.structures:
                new_structs.append(
                    struct.updated_copy(medium=struct.medium.updated_copy(charge=None))
                )
            _ = sim.updated_copy(structures=new_structs)

        with pytest.raises(ValidationError):
            # make sure there is at least one semiconductor
            new_structs = []
            for struct in sim.structures:
                if isinstance(struct.medium.charge, td.SemiconductorMedium):
                    new_structs.append(
                        struct.updated_copy(
                            medium=struct.medium.updated_copy(
                                charge=td.ChargeInsulatorMedium(permittivity=1),
                                heat=None,
                            )
                        )
                    )
                else:
                    new_structs.append(struct)
            _ = sim.updated_copy(structures=new_structs)

    def test_doping_distributions(self):
        """Test doping distributions."""
        # Implementation needed
        # This test was empty in the original code.


# --------------------------
# Additional Tests
# --------------------------
def test_semiconductor_medium():
    """Make sure we can create a semiconductor with different models."""
    # Create a semiconductor medium with different mobility models
    intrinsic_Si = td.SemiconductorMedium(
        permittivity=11.7,
        N_d=0,
        N_a=0,
        N_c=ConstantEffectiveDOS(N=2.86e19),
        N_v=ConstantEffectiveDOS(N=3.1e19),
        E_g=ConstantEnergyBandGap(eg=1.11),
        mobility_n=td.ConstantMobilityModel(mu=1350),
        mobility_p=td.ConstantMobilityModel(mu=480),
        R=[],
        delta_E_g=None,
    )

    ct_mobility = CaugheyThomasMobility(
        mu_min=52.2,
        mu=1471.0,
        ref_N=9.68e16,
        exp_N=0.68,
        exp_1=-0.57,
        exp_2=-2.33,
        exp_3=2.4,
        exp_4=-0.146,
    )
    # Try different mobility models
    _ = intrinsic_Si.updated_copy(
        mobility_n=ct_mobility,
        mobility_p=ct_mobility,
    )

    fossum = td.FossumCarrierLifetime(
        tau_300=3.3e-6, alpha_T=-0.5, N0=7.1e15, A=1, B=0, C=1, alpha=1
    )
    R = [
        AugerRecombination(c_n=2.8e-31, c_p=9.9e-32),
        td.RadiativeRecombination(r_const=1.6e-14),
        td.ShockleyReedHallRecombination(
            tau_n=3.3e-6,
            tau_p=4e-6,
        ),
        td.ShockleyReedHallRecombination(
            tau_n=fossum,
            tau_p=fossum,
        ),
    ]
    # try the different recombination models
    _ = intrinsic_Si.updated_copy(R=R)

    # Try band gap narrowing model
    _ = intrinsic_Si.updated_copy(
        delta_E_g=SlotboomBandGapNarrowing(
            v1=6.92 * 1e-3,
            n2=1.3e17,
            c2=0.5,
            min_N=1e15,
        ),
    )

    # Try the different effective DOS models
    N_models = [
        ConstantEffectiveDOS(N=2.86e19),
        td.IsotropicEffectiveDOS(m_eff=1.08),
        td.MultiValleyEffectiveDOS(m_eff_long=1.08, m_eff_trans=0.19, N_valley=3),
        td.DualValleyEffectiveDOS(m_eff_hh=0.49, m_eff_lh=0.16),
    ]
    for m in N_models:
        _ = intrinsic_Si.updated_copy(N_c=m, N_v=m)


@pytest.mark.parametrize("shift_amount, log_level", [(1, None), (2, "WARNING")])
def test_heat_charge_sim_bounds(shift_amount, log_level):
    """Ensure bounds are working correctly."""
    # Make sure all things are shifted to this central location
    CENTER_SHIFT = (-1.0, 1.0, 100.0)

    def place_box(center_offset):
        shifted_center = tuple(c + s for (c, s) in zip(center_offset, CENTER_SHIFT))

        _ = td.HeatChargeSimulation(
            size=(1.5, 1.5, 1.5),
            center=CENTER_SHIFT,
            medium=td.MultiPhysicsMedium(charge=td.ChargeConductorMedium(conductivity=1)),
            structures=[
                td.Structure(
                    geometry=td.Box(size=(1, 1, 1), center=shifted_center),
                    medium=td.MultiPhysicsMedium(charge=td.ChargeConductorMedium(conductivity=1)),
                )
            ],
            boundary_spec=[
                td.HeatChargeBoundarySpec(
                    condition=td.VoltageBC(source=td.DCVoltageSource(voltage=[1])),
                    placement=td.SimulationBoundary(),
                )
            ],
            grid_spec=td.UniformUnstructuredGrid(dl=0.1),
            monitors=[
                td.SteadyPotentialMonitor(
                    center=[0, 0, 0],
                    size=(td.inf, td.inf, td.inf),
                    name="test_monitor",
                    unstructured=False,
                )
            ],
        )

    # Create all permutations of squares being shifted 1, -1, or zero in all three directions
    bin_strings = [format(i, "03b") for i in range(8)]
    bin_ints = [[int(b) for b in bin_string] for bin_string in bin_strings]
    bin_ints = np.array(bin_ints)
    bin_signs = 2 * (bin_ints - 0.5)

    # Test all cases where box is shifted +/- 1 in x,y,z and still intersects
    for amp in bin_ints:
        for sign in bin_signs:
            center = tuple(shift_amount * a * s for a, s in zip(amp, sign))
            if np.sum(np.abs(center)) < 1e-12:
                continue
            with AssertLogLevel(log_level):
                place_box(center)


@pytest.mark.parametrize(
    "box_size, log_level",
    [
        ((1, 0.1, 0.1), "WARNING"),
        ((0.1, 1, 0.1), "WARNING"),
        ((0.1, 0.1, 1), "WARNING"),
    ],
)
def test_sim_structure_extent(box_size, log_level):
    """Ensure we warn if structure extends exactly to simulation edges."""
    box = td.Structure(
        geometry=td.Box(size=box_size),
        medium=td.MultiPhysicsMedium(charge=td.ChargeConductorMedium(conductivity=1)),
    )

    with AssertLogLevel(log_level):
        _ = td.HeatChargeSimulation(
            size=(1, 1, 1),
            structures=(box,),
            medium=td.MultiPhysicsMedium(charge=td.ChargeConductorMedium(conductivity=1)),
            boundary_spec=[
                td.HeatChargeBoundarySpec(
                    placement=td.SimulationBoundary(),
                    condition=td.VoltageBC(source=td.DCVoltageSource(voltage=[1])),
                )
            ],
            grid_spec=td.UniformUnstructuredGrid(dl=0.1),
            monitors=[
                td.SteadyPotentialMonitor(
                    center=(0, 0, 0),
                    size=(td.inf, td.inf, td.inf),
                    name="test_monitor",
                    unstructured=False,
                )
            ],
        )


def test_abstract_doping_box_with_box_coords():
    """Test AbstractDopingBox with provided box_coords."""
    box_coords = ((-1, -1, -1), (1, 1, 1))
    box = td.ConstantDoping.from_bounds(rmin=box_coords[0], rmax=box_coords[1])
    assert box.size == (2, 2, 2), "Size should be calculated based on box_coords."
    assert box.center == (0, 0, 0), "Center should be calculated based on box_coords."


def test_constant_doping_initialization():
    """Test initialization of ConstantDoping."""
    box = td.ConstantDoping(center=(0, 0, 0), size=(1, 1, 1), concentration=1e18)
    assert box.concentration == 1e18, "Concentration should be set to 1e18."


def test_gaussian_doping_initialization():
    """Test initialization of GaussianDoping."""
    box = td.GaussianDoping(
        size=(1, 1, 1), ref_con=1e15, concentration=1e18, width=0.1, source="xmin"
    )
    assert box.ref_con == 1e15, "Reference concentration should be 1e15."
    assert box.concentration == 1e18, "Concentration should be 1e18."
    assert box.width == 0.1, "Width should be 0.1."
    assert box.source == "xmin", "Source should be 'xmin'."


def test_gaussian_doping_sigma_calculation():
    """Test sigma calculation in GaussianDoping and ref_con validator."""
    box = td.GaussianDoping(
        size=(1, 1, 1), ref_con=1e15, concentration=1e18, width=0.1, source="xmin"
    )
    expected_sigma = np.sqrt(-(0.1**2) / (2 * np.log(1e15 / 1e18)))
    assert np.isclose(box.sigma, expected_sigma), "Sigma calculation is incorrect."

    with pytest.raises(ValidationError, match=r"must be less than.*concentration"):
        _ = td.GaussianDoping(
            size=(1, 1, 1), ref_con=1e19, concentration=1e18, width=0.1, source="xmin"
        )


def test_gaussian_doping_get_contrib():
    """Test _get_contrib method in GaussianDoping."""
    max_N = 1e18
    min_N = 1e15
    width = 0.1

    box = td.GaussianDoping(
        size=(1, 1, 1), ref_con=min_N, concentration=max_N, width=width, source="xmin"
    )

    coords = {"x": [0], "y": [0], "z": [0]}
    contrib = box._get_contrib(coords)
    assert np.isclose(contrib.item(), max_N, rtol=1e-6)

    coords = {"x": [0.5], "y": [0], "z": [0]}
    contrib = box._get_contrib(coords)
    assert np.isclose(contrib.item(), min_N, rtol=1e-6)

    coords = {"x": [0.5 - width / 2], "y": [0], "z": [0]}
    contrib = box._get_contrib(coords)
    expected_value = max_N * np.exp(-width * width / 4 / box.sigma / box.sigma / 2)
    assert np.isclose(contrib.item(), expected_value, rtol=1e-6)


def test_gaussian_doping_get_contrib_2d_coords():
    """Test _get_contrib method in GaussianDoping with 2D coordinates."""
    box = td.GaussianDoping(
        size=(1, 1, 1), ref_con=1e15, concentration=1e18, width=0.1, source="xmin"
    )
    coords = {"x": [0], "y": [0], "z": [-0.5, 0, 0.5]}
    _ = box._get_contrib(coords)


def test_gaussian_doping_bounds_behavior():
    """Test GaussianDoping bounds behavior."""
    box_coords = ((-1, -1, -1), (1, 1, 1))
    box = td.GaussianDoping.from_bounds(
        rmin=box_coords[0],
        rmax=box_coords[1],
        ref_con=1e15,
        concentration=1e18,
        width=0.1,
        source="xmin",
    )
    assert box.bounds == box_coords, "Bounds should match provided box_coords."


def test_gaussian_doping_validator_source():
    """Test validator for source face."""
    valid_sources = ["xmin", "xmax", "ymin", "ymax", "zmin", "zmax"]
    for source in valid_sources:
        _ = td.GaussianDoping(
            size=(1, 1, 1), ref_con=1e15, concentration=1e18, width=0.1, source=source
        )

    with pytest.raises(ValidationError):
        _ = td.GaussianDoping(
            size=(1, 1, 1), ref_con=1e15, concentration=1e18, width=0.1, source="invalid"
        )


def test_gaussian_doping_validator_width():
    """Test validator for width vs size."""
    width = 0.1
    _ = td.GaussianDoping(
        size=(0.2, 0.2, 0.2), ref_con=1e15, concentration=1e18, width=width, source="xmin"
    )
    _ = td.GaussianDoping(
        size=(np.inf, 1, 1), ref_con=1e15, concentration=1e18, width=width, source="xmin"
    )

    with AssertLogLevel("WARNING", contains_str="'x' direction"):
        _ = td.GaussianDoping(
            size=(0.15, 1, 1), ref_con=1e15, concentration=1e18, width=width, source="xmin"
        )
    with AssertLogLevel("WARNING", contains_str="'y' direction"):
        _ = td.GaussianDoping(
            size=(1, 0.15, 1), ref_con=1e15, concentration=1e18, width=width, source="xmin"
        )
    with AssertLogLevel("WARNING", contains_str="'z' direction"):
        _ = td.GaussianDoping(
            size=(1, 1, 0.15), ref_con=1e15, concentration=1e18, width=width, source="xmin"
        )


def test_2D_doping_box():
    """Check that the doping boxes can handle 2D cases correctly."""

    _ = td.ConstantDoping(size=(1, 1, np.inf), concentration=1)

    _ = td.ConstantDoping.from_bounds(rmin=(-td.inf, -1, -1), rmax=(td.inf, 1, 1), concentration=1)


def test_edge_case_boundary_conditions():
    """Test boundary conditions with extreme values."""
    # Zero heat flux
    bc_zero_flux = td.HeatFluxBC(flux=0)
    assert bc_zero_flux.flux == 0

    # Negative heat flux
    bc_neg_flux = td.HeatFluxBC(flux=-10)
    assert bc_neg_flux.flux == -10


def test_simulation_initialization_invalid_parameters(
    mediums, structures, boundary_conditions, monitors, grid_specs
):
    """Test simulation initialization with invalid parameters."""
    # Invalid simulation size
    with pytest.raises(ValidationError):
        td.HeatChargeSimulation(
            medium=mediums["fluid_medium"],
            structures=[structures["fluid_structure"]],
            center=(0, 0, 0),
            size=(-1, 2, 2),  # Negative size
            boundary_spec=[],
            grid_spec=grid_specs["uniform"],
            sources=[],
            monitors=[],
        )

    # Invalid monitor type
    with pytest.raises(ValidationError):
        td.HeatChargeSimulation(
            medium=mediums["fluid_medium"],
            structures=[structures["fluid_structure"]],
            center=(0, 0, 0),
            size=(2, 2, 2),
            boundary_spec=[],
            grid_spec=grid_specs["uniform"],
            sources=[],
            monitors=["invalid_monitor"],  # Should be monitor objects
        )


def test_simulation_with_multiple_sources_and_monitors(
    mediums, structures, boundary_conditions, grid_specs
):
    """Test simulation with multiple heat sources and monitors."""
    sources = [
        td.HeatSource(structures=["solid_structure"], rate=100),
        td.HeatSource(structures=["fluid_structure"], rate=200),
    ]

    monitors = [
        td.TemperatureMonitor(size=(1.6, 2, 3), name="temp_mnt1", unstructured=False),
        td.SteadyPotentialMonitor(size=(1.6, 2, 3), name="volt_mnt1", unstructured=False),
    ]

    boundary_spec = [
        td.HeatChargeBoundarySpec(
            condition=boundary_conditions[0],  # TemperatureBC
            placement=td.SimulationBoundary(),
        ),
        td.HeatChargeBoundarySpec(
            condition=boundary_conditions[1],  # HeatFluxBC
            placement=td.StructureBoundary(structure="solid_structure"),
        ),
    ]

    sim = td.HeatChargeSimulation(
        medium=mediums["solid_medium"],
        structures=[structures["solid_structure"], structures["fluid_structure"]],
        center=(0, 0, 0),
        size=(2, 2, 2),
        boundary_spec=boundary_spec,
        grid_spec=grid_specs["uniform"],
        sources=sources,
        monitors=monitors,
    )

    assert len(sim.sources) == 2
    assert len(sim.monitors) == 2


def test_dynamic_simulation_updates(heat_simulation):
    """Test updating simulation parameters after initialization."""
    # Update simulation size
    new_size = (3, 3, 3)
    updated_sim = heat_simulation.updated_copy(size=new_size)
    assert updated_sim.size == new_size

    # Update center
    new_center = (1, 1, 1)
    updated_sim = heat_simulation.updated_copy(center=new_center)
    assert updated_sim.center == new_center

    # Add a new monitor
    new_monitor = td.TemperatureMonitor(size=(1, 1, 1), name="new_temp_mnt", unstructured=False)
    updated_sim = heat_simulation.updated_copy(monitors=(*heat_simulation.monitors, new_monitor))
    assert len(updated_sim.monitors) == len(heat_simulation.monitors) + 1
    assert updated_sim.monitors[-1].name == "new_temp_mnt"


def test_plotting_functions(simulation_data):
    """Test plotting functions with various data."""
    heat_sim_data, cond_sim_data, _cap_sim_data, _fc_sim_data, _mesh_data = simulation_data

    # Valid plotting
    try:
        heat_sim_data.plot_field("test", z=0)
        cond_sim_data.plot_field("v_test", y=1)
    except Exception as e:
        pytest.fail(f"Plotting raised an exception unexpectedly: {e}")

    # Invalid field name
    with pytest.raises(KeyError):
        heat_sim_data.plot_field("non_existent_field")

    # Invalid plotting parameters
    with pytest.raises(KeyError):
        heat_sim_data.plot_field("test", invalid_param=0)


def test_bandgap_monitor():
    """Test energy bandgap monitor ploting function."""
    # create a triangle grid
    tri_grid_points = td.PointDataArray(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        dims=("index", "axis"),
    )

    tri_grid_cells = td.CellDataArray(
        [[0, 1, 2], [1, 2, 3]],
        dims=("cell_index", "vertex_index"),
    )

    tri_grid_values_single_voltage = td.IndexedVoltageDataArray(
        [[0.0], [0], [3], [3]],
        coords={"index": np.arange(4), "voltage": [1]},
        name="test",
    )

    tri_grid_values_multi_voltage = td.IndexedVoltageDataArray(
        [[0.0, 0.0], [0, 0], [3, -3], [3, -3]],
        coords={"index": np.arange(4), "voltage": [-1, 1]},
        name="test",
    )

    tri_grid_single_voltage = td.TriangularGridDataset(
        normal_axis=1,
        normal_pos=0,
        points=tri_grid_points,
        cells=tri_grid_cells,
        values=tri_grid_values_single_voltage,
    )

    tri_grid_multi_voltage = td.TriangularGridDataset(
        normal_axis=1,
        normal_pos=0,
        points=tri_grid_points,
        cells=tri_grid_cells,
        values=tri_grid_values_multi_voltage,
    )

    # create a tet mesh
    tet_grid_points = td.PointDataArray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dims=("index", "axis"),
    )

    tet_grid_cells = td.CellDataArray(
        [[0, 1, 3, 7], [0, 2, 7, 3], [0, 2, 6, 7], [0, 4, 7, 6], [0, 4, 5, 7], [0, 1, 7, 5]],
        dims=("cell_index", "vertex_index"),
    )

    tet_grid_values_single_voltage = td.IndexedVoltageDataArray(
        [[0.0], [0.0], [0.0], [0.0], [3.0], [3.0], [3.0], [3.0]],
        coords={"index": np.arange(8), "voltage": [1]},
        name="test_tet",
    )

    tet_grid_values_multi_voltage = td.IndexedVoltageDataArray(
        [
            [0.0, 0.5],
            [0.0, 0.5],
            [0.0, 0.5],
            [0.0, 0.5],
            [3.0, 3.5],
            [3.0, 3.5],
            [3.0, 3.5],
            [3.0, 3.5],
        ],
        coords={"index": np.arange(8), "voltage": [-1, 1]},
        name="test_tet",
    )

    tet_grid_single_voltage = td.TetrahedralGridDataset(
        points=tet_grid_points,
        cells=tet_grid_cells,
        values=tet_grid_values_single_voltage,
    )

    tet_grid_multi_voltage = td.TetrahedralGridDataset(
        points=tet_grid_points,
        cells=tet_grid_cells,
        values=tet_grid_values_multi_voltage,
    )

    aux_monitor_2D = td.SteadyEnergyBandMonitor(
        center=(0, 0.14, 0), size=(0.6, 0.3, 0), name="bands_2D", unstructured=True
    )

    aux_monitor_3D = td.SteadyEnergyBandMonitor(
        center=(0, 0.14, 0.0), size=(0.6, 0.3, 0.5), name="bands_3D", unstructured=True
    )

    tri_single_voltage_data = td.SteadyEnergyBandData(
        monitor=aux_monitor_2D,
        Ec=tri_grid_single_voltage,
        Ev=tri_grid_single_voltage,
        Ei=tri_grid_single_voltage,
        Efn=tri_grid_single_voltage,
        Efp=tri_grid_single_voltage,
    )

    tri_multi_voltage_data = td.SteadyEnergyBandData(
        monitor=aux_monitor_2D,
        Ec=tri_grid_multi_voltage,
        Ev=tri_grid_multi_voltage,
        Ei=tri_grid_multi_voltage,
        Efn=tri_grid_multi_voltage,
        Efp=tri_grid_multi_voltage,
    )

    tet_single_voltage_data = td.SteadyEnergyBandData(
        monitor=aux_monitor_3D,
        Ec=tet_grid_single_voltage,
        Ev=tet_grid_single_voltage,
        Ei=tet_grid_single_voltage,
        Efn=tet_grid_single_voltage,
        Efp=tet_grid_single_voltage,
    )

    tet_multi_voltage_data = td.SteadyEnergyBandData(
        monitor=aux_monitor_3D,
        Ec=tet_grid_multi_voltage,
        Ev=tet_grid_multi_voltage,
        Ei=tet_grid_multi_voltage,
        Efn=tet_grid_multi_voltage,
        Efp=tet_grid_multi_voltage,
    )

    # test check for the voltage value in the list of arguments

    tri_single_voltage_data.plot(x=0.0)
    tri_multi_voltage_data.plot(x=0.0, voltage=1.0)

    with pytest.raises(DataError):
        tri_multi_voltage_data.plot(x=0.0)

    tet_single_voltage_data.plot(x=0.0, y=0.0)
    tet_multi_voltage_data.plot(x=0.0, y=0.0, voltage=1.0)

    with pytest.raises(DataError):
        tri_multi_voltage_data.plot(x=0.0, y=0.0)

    # test check for the number of coordinates in the list of arguments

    with pytest.raises(DataError):
        tri_single_voltage_data.plot()

    with pytest.raises(DataError):
        tri_single_voltage_data.plot(x=0.0, y=0.0)

    with pytest.raises(DataError):
        tet_single_voltage_data.plot()

    with pytest.raises(DataError):
        tet_single_voltage_data.plot(x=0.0)

    with pytest.raises(DataError):
        tet_single_voltage_data.plot(x=0.0, y=0.0, z=0.0)

    # test check for the incorrect cross-section plane

    with pytest.raises(DataError):
        tri_single_voltage_data.plot(y=0.0)


def test_additional_edge_cases():
    """Test additional edge cases and error handling."""
    # Attempt to create a monitor with zero size
    td.TemperatureMonitor(size=(0, 0, 0), name="zero_size_mnt", unstructured=False)

    # Create a simulation with overlapping structures
    td.HeatChargeSimulation(
        medium=td.MultiPhysicsMedium(
            optical=td.Medium(permittivity=5),
            charge=td.ChargeConductorMedium(conductivity=1),
            name="overlap_medium",
        ),
        structures=[
            td.Structure(
                geometry=td.Box(center=(0, 0, 0), size=(1, 1, 1)), medium=td.Medium(name="medium1")
            ),
            td.Structure(
                geometry=td.Box(center=(0, 0, 0), size=(1, 1, 1)), medium=td.Medium(name="medium2")
            ),
        ],
        center=(0, 0, 0),
        size=(2, 2, 2),
        boundary_spec=[],
        grid_spec=td.UniformUnstructuredGrid(dl=0.1),
        sources=[],
        monitors=[],
    )


def test_fossum():
    """Check that fossum model can be defined."""

    _ = td.FossumCarrierLifetime(tau_300=3.3e-6, alpha_T=-0.5, N0=7.1e15, A=1, B=0, C=1, alpha=1)


@pytest.mark.parametrize("symmetry", [(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 0)])
def test_symmetry_capacitance(symmetry):
    """Check that symmetry_expanded_copy works as expected"""

    data = [1, 2, 3]
    voltages = [0, 1, 2]

    hole_capacitance = td.SteadyVoltageDataArray(
        data=data,
        coords={"v": voltages},
    )

    electron_capacitance = td.SteadyVoltageDataArray(
        data=data,
        coords={"v": voltages},
    )

    monitor = td.SteadyCapacitanceMonitor(
        center=(0, 0, 0),
        size=(1, 1, 1),
        name="test_monitor",
    )

    mnt_data = td.SteadyCapacitanceData(
        monitor=monitor,
        hole_capacitance=hole_capacitance,
        electron_capacitance=electron_capacitance,
        symmetry=symmetry,
    )

    num_symmetries = np.sum(np.array([1 if d > 0 else 0 for d in symmetry]))
    scaling_factor = np.power(2, num_symmetries)

    for n in range(len(data)):
        assert mnt_data.symmetry_expanded_copy.hole_capacitance.data[n] == data[n] * scaling_factor
        assert (
            mnt_data.symmetry_expanded_copy.electron_capacitance.data[n] == data[n] * scaling_factor
        )


def test_unsteady_parameters():
    """Test that unsteady parameters are set correctly."""

    _ = td.UnsteadyHeatAnalysis(
        initial_temperature=300,
        unsteady_spec=td.UnsteadySpec(time_step=0.1, total_time_steps=1),
    )

    # test non-positive initial temperature raises error
    with pytest.raises(ValidationError):
        _ = td.UnsteadyHeatAnalysis(
            initial_temperature=0,
            unsteady_spec=td.UnsteadySpec(time_step=0.1, total_time_steps=1),
        )

    # test negative time step raises error
    with pytest.raises(ValidationError):
        _ = td.UnsteadyHeatAnalysis(
            initial_temperature=10,
            unsteady_spec=td.UnsteadySpec(time_step=-0.1, total_time_steps=1),
        )

    # test negative total time steps raises error
    with pytest.raises(ValidationError):
        _ = td.UnsteadyHeatAnalysis(
            initial_temperature=10,
            unsteady_spec=td.UnsteadySpec(time_step=0.1, total_time_steps=-1),
        )


def test_unsteady_heat_analysis(heat_simulation):
    """Test that the validators for unsteady heat analysis are working."""

    unsteady_analysis_spec = td.UnsteadyHeatAnalysis(
        initial_temperature=300,
        unsteady_spec=td.UnsteadySpec(time_step=0.1, total_time_steps=1),
    )

    temp_mnt = td.TemperatureMonitor(
        center=(0, 0, 0),
        size=(td.inf, td.inf, td.inf),
        name="temperature",
        unstructured=True,
        interval=2,
    )

    # this should work since the monitor is unstructured
    unsteady_sim = heat_simulation.updated_copy(
        analysis_spec=unsteady_analysis_spec, monitors=(temp_mnt,)
    )

    with pytest.raises(ValidationError) as excinfo:
        temp_mnt = temp_mnt.updated_copy(unstructured=False)
        _ = unsteady_sim.updated_copy(monitors=(temp_mnt,))
    assert_single_value_error_loc(excinfo, ("monitors",), "to be unstructured")

    with pytest.raises(ValidationError):
        temp_mnt = temp_mnt.updated_copy(unstructured=True, interval=0)
        _ = unsteady_sim.updated_copy(monitors=(temp_mnt,))

    # try simulation with excessive time steps
    with pytest.raises(ValidationError) as excinfo:
        mew_spex = td.UnsteadyHeatAnalysis(
            initial_temperature=300,
            unsteady_spec=td.UnsteadySpec(time_step=0.1, total_time_steps=100000),
        )
        _ = unsteady_sim.updated_copy(analysis_spec=mew_spex)
    assert_single_value_error_loc(excinfo, ("analysis_spec",), "number of time-steps")


def test_heat_conduction_simulations():
    """Test that heat-conduction simulations have necessary components."""

    # let's create some mediums
    solid_medium = td.MultiPhysicsMedium(
        heat=td.SolidSpec(conductivity=1),
        charge=td.ChargeConductorMedium(conductivity=1),
        name="solid_medium",
    )
    air = td.MultiPhysicsMedium(heat=td.FluidSpec(), charge=td.ChargeInsulatorMedium(), name="air")

    struct1 = td.Structure(
        geometry=td.Box(center=(0, 0, 0), size=(1, 1, 1)),
        medium=solid_medium,
        name="struct1",
    )

    # thermal BC
    thermal_bc = td.HeatChargeBoundarySpec(
        condition=td.TemperatureBC(temperature=300),
        placement=td.StructureBoundary(structure="struct1"),
    )

    # electric BCs
    electric_bc = td.HeatChargeBoundarySpec(
        condition=td.VoltageBC(source=td.DCVoltageSource(voltage=[1])),
        placement=td.StructureBoundary(structure="struct1"),
    )

    # thermal monitors
    temp_monitor = td.TemperatureMonitor(
        center=(0, 0, 0), size=(1, 1, 1), name="temp_monitor", unstructured=True
    )
    # electric monitors
    voltage_monitor = td.SteadyPotentialMonitor(
        center=(0, 0, 0), size=(1, 1, 1), name="voltage_monitor", unstructured=True
    )

    sim = td.HeatChargeSimulation(
        medium=air,
        structures=[struct1],
        center=(0, 0, 0),
        size=(3, 3, 3),
        boundary_spec=[thermal_bc, electric_bc],
        grid_spec=td.UniformUnstructuredGrid(dl=0.1),
        sources=[],
        monitors=[temp_monitor, voltage_monitor],
    )

    with pytest.raises(ValidationError):
        # no thermal monitors
        _ = sim.updated_copy(monitors=[voltage_monitor])

    with pytest.raises(ValidationError):
        # voltage array in electric BC
        _ = sim.updated_copy(
            boundary_spec=[
                thermal_bc,
                electric_bc.updated_copy(
                    condition=td.VoltageBC(source=td.DCVoltageSource(voltage=[1, 2]))
                ),
            ]
        )

    # this doesn't raise error
    _ = sim.updated_copy(sources=[td.HeatFromElectricSource()])

    with pytest.raises(ValidationError):
        # This should error since the conduction simulation doesn't have a monitor
        _ = sim.updated_copy(monitors=[temp_monitor])

    # test error if structures defined with Medium instead of MultiPhysicsMedium
    with pytest.raises(ValidationError):
        struct_error = struct1.updated_copy(medium=td.Medium(conductivity=1))
        _ = sim.updated_copy(structures=[struct_error])

    # test error if structures aren't conducting
    with pytest.raises(ValidationError):
        struct_error = struct1.updated_copy(
            medium=struct1.medium.updated_copy(charge=td.ChargeInsulatorMedium())
        )
        _ = sim.updated_copy(structures=[struct_error])


def test_generation_recombination():
    """Test that generation and recombination models are properly defined."""

    # Create a spatial data array for generation rate
    x = [1, 2]
    y = [2, 3, 4]
    z = [3, 4, 5, 6]
    coords = {"x": x, "y": y, "z": z}
    fd = td.SpatialDataArray(np.random.random((2, 3, 4)), coords=coords)

    # make sure we can create a DistributedGeneration
    _ = td.DistributedGeneration(rate=fd)

    # check that unit conversion works
    g_um3 = fd.sel(x=1, y=2, z=3).item()
    new_g = td.DistributedGeneration.from_rate_um3(fd)
    assert new_g.rate.sel(x=1, y=2, z=3).item() == g_um3 * 1e12

    # make sure an error is raised if input array is 1D
    with pytest.raises(ValueError):
        rate1D = td.SpatialDataArray(
            np.random.random((5, 1, 1)), coords={"x": [1, 2, 3, 4, 5], "y": [1], "z": [1]}
        )
        _ = td.DistributedGeneration(rate=rate1D)

    # make sure we can build Fossum
    tau_fossum = td.FossumCarrierLifetime(
        tau_300=3.3e-6, alpha_T=-0.5, N0=7.1e15, A=1, B=0, C=1, alpha=1
    )

    # make sure we can build AugerRecombination
    _ = td.AugerRecombination(
        c_n=2.8e-31,
        c_p=9.9e-32,
    )

    # make sure we can build RadiativeRecombination
    _ = td.RadiativeRecombination(r_const=1.6e-14)

    # make sure we can build ShockleyReedHallRecombination
    _ = td.ShockleyReedHallRecombination(
        tau_n=3.3e-6,
        tau_p=4e-6,
    )

    _ = td.ShockleyReedHallRecombination(
        tau_n=tau_fossum,
        tau_p=tau_fossum,
    )

    # make sure we can build a HurkxDirectBandToBandTunneling
    _ = td.HurkxDirectBandToBandTunneling(
        A=1e19,
        B=1.9e6,
        E_0=1,
        sigma=2,
    )

    # make sure we can build a SelberherrImpactIonization
    _ = td.SelberherrImpactIonization(
        alpha_n_inf=7.03e5,
        alpha_p_inf=1.582e6,
        E_n_crit=1.23e6,
        E_p_crit=2.03e6,
        beta_n=1,
        beta_p=1,
    )


def test_heat_only_simulation_with_semiconductor():
    """Test that a heat-only simulation with semiconductors does not trigger charge simulation.
    Charge simulations are only triggered when `analysis_spec` is provided, not just when
    semiconductors are present in the simulation.
    """

    # Create a semiconductor medium
    semiconductor_medium = td.MultiPhysicsMedium(
        optical=td.Medium(permittivity=5, conductivity=0.01),
        heat=td.SolidMedium(conductivity=3, capacity=2),
        charge=td.SemiconductorMedium(
            N_c=td.ConstantEffectiveDOS(N=1e10),
            N_v=td.ConstantEffectiveDOS(N=1e10),
            E_g=td.ConstantEnergyBandGap(eg=1),
            mobility_n=td.ConstantMobilityModel(mu=1500),
            mobility_p=td.ConstantMobilityModel(mu=1500),
        ),
        name="semiconductor",
    )

    # Create a non-semiconductor solid medium
    solid_medium = td.MultiPhysicsMedium(
        optical=td.Medium(permittivity=5, conductivity=0.01),
        heat=td.SolidMedium(conductivity=1, capacity=1),
        charge=td.ChargeConductorMedium(conductivity=1),
        name="solid",
    )

    # Create structures with both semiconductor and other materials
    semiconductor_structure = td.Structure(
        geometry=td.Box(center=(-0.5, 0, 0), size=(1, 1, 1)),
        medium=semiconductor_medium,
        name="semiconductor_structure",
    )

    solid_structure = td.Structure(
        geometry=td.Box(center=(0.5, 0, 0), size=(1, 1, 1)),
        medium=solid_medium,
        name="solid_structure",
    )

    # Create heat-only boundary conditions (no electric BCs)
    thermal_bc = td.HeatChargeBoundarySpec(
        condition=td.TemperatureBC(temperature=300),
        placement=td.StructureBoundary(structure="solid_structure"),
    )

    # Create heat source
    heat_source = td.HeatSource(structures=["solid_structure"], rate=100)

    # Create heat monitor (no charge monitors)
    temp_monitor = td.TemperatureMonitor(
        center=(0, 0, 0), size=(2, 1, 1), name="temp_monitor", unstructured=True
    )

    # Create heat-only simulation (no analysis_spec, no electric BCs)
    heat_sim = td.HeatChargeSimulation(
        medium=td.MultiPhysicsMedium(
            heat=td.FluidMedium(), charge=td.ChargeInsulatorMedium(), name="air"
        ),
        structures=[semiconductor_structure, solid_structure],
        center=(0, 0, 0),
        size=(3, 3, 3),
        boundary_spec=[thermal_bc],
        grid_spec=td.UniformUnstructuredGrid(dl=0.1),
        sources=[heat_source],
        monitors=[temp_monitor],
    )

    # Verify that only HEAT simulation type is returned, not CHARGE
    simulation_types = heat_sim._get_simulation_types()
    assert TCADAnalysisTypes.HEAT in simulation_types, (
        "Heat simulation should be triggered when heat sources/BCs are present."
    )
    assert TCADAnalysisTypes.CHARGE not in simulation_types, (
        "Charge simulation should NOT be triggered when ChargeTypes analysis_spec is not provided, "
        "even if semiconductors are present in the simulation."
    )
    assert TCADAnalysisTypes.CONDUCTION not in simulation_types, (
        "Conduction simulation should NOT be triggered when no electric BCs are present."
    )


def test_heat_charge_simulation_plot():
    """Test the HeatChargeSimulation.plot() method adds BCs based on simulation type."""

    # Create mediums
    solid_medium = td.MultiPhysicsMedium(
        heat=td.SolidMedium(conductivity=1, capacity=1),
        name="solid",
    )
    fluid_medium = td.MultiPhysicsMedium(
        heat=td.FluidMedium(),
        name="fluid",
    )

    # Create structures
    solid_structure = td.Structure(
        geometry=td.Box(size=(1, 1, 1), center=(0, 0, 0)),
        medium=solid_medium,
        name="solid_structure",
    )

    # Create boundary conditions for heat simulation
    bc_temp = td.HeatChargeBoundarySpec(
        condition=td.TemperatureBC(temperature=300),
        placement=td.StructureBoundary(structure="solid_structure"),
    )

    # Create heat source
    heat_source = td.UniformHeatSource(rate=1e3, structures=["solid_structure"])

    # Create monitor
    temp_monitor = td.TemperatureMonitor(
        center=(0, 0, 0),
        size=(1, 1, 0),
        name="temp_mnt",
        unstructured=False,
    )

    # Create a HEAT simulation
    heat_sim = td.HeatChargeSimulation(
        medium=fluid_medium,
        structures=[solid_structure],
        center=(0, 0, 0),
        size=(2, 2, 2),
        boundary_spec=[bc_temp],
        grid_spec=td.UniformUnstructuredGrid(dl=0.1),
        sources=[heat_source],
        monitors=[temp_monitor],
    )

    # Test plot for HEAT simulation - should add heat BCs
    _, ax_scene_only = plt.subplots()
    heat_sim.scene.plot(z=0, ax=ax_scene_only)
    num_children_scene_only = len(ax_scene_only.get_children())
    plt.close()

    _, ax_with_bc = plt.subplots()
    heat_sim.plot(z=0, ax=ax_with_bc)
    num_children_with_bc = len(ax_with_bc.get_children())
    plt.close()

    # heat_sim.plot() should have more visual elements than scene.plot()
    # because it adds monitors and heat boundaries for HEAT simulations
    assert num_children_with_bc - num_children_scene_only >= 2, (
        "heat_sim.plot() should add at least monitors and heat boundaries "
        "for HEAT simulations, resulting in at least 2 more visual elements "
        "than heat_sim.scene.plot()"
    )

    # Now test with a CHARGE simulation
    semicon = td.material_library["cSi"].variants["Si_MultiPhysics"].medium.charge
    Si_n = semicon.updated_copy(N_d=[td.ConstantDoping(concentration=1e16)], name="Si_n")
    Si_p = semicon.updated_copy(N_a=[td.ConstantDoping(concentration=1e16)], name="Si_p")

    n_side = td.Structure(
        geometry=td.Box(center=(-0.25, 0, 0), size=(0.5, 1, 1)),
        medium=Si_n,
        name="n_side",
    )
    p_side = td.Structure(
        geometry=td.Box(center=(0.25, 0, 0), size=(0.5, 1, 1)),
        medium=Si_p,
        name="p_side",
    )

    bc_v1 = td.HeatChargeBoundarySpec(
        condition=td.VoltageBC(source=td.DCVoltageSource(voltage=0)),
        placement=td.MediumMediumInterface(mediums=[fluid_medium.name, Si_n.name]),
    )
    bc_v2 = td.HeatChargeBoundarySpec(
        condition=td.VoltageBC(source=td.DCVoltageSource(voltage=0.5)),
        placement=td.MediumMediumInterface(mediums=[fluid_medium.name, Si_p.name]),
    )

    volt_monitor = td.SteadyPotentialMonitor(
        center=(0, 0, 0),
        size=(1, 1, 0),
        name="volt_mnt",
        unstructured=True,
    )

    charge_sim = td.HeatChargeSimulation(
        structures=[n_side, p_side],
        medium=fluid_medium,
        monitors=[volt_monitor],
        center=(0, 0, 0),
        size=(2, 2, 2),
        grid_spec=td.UniformUnstructuredGrid(dl=0.05),
        boundary_spec=[bc_v1, bc_v2],
        analysis_spec=td.IsothermalSteadyChargeDCAnalysis(temperature=300),
    )

    # Test plot for CHARGE simulation - should add electric BCs
    _, ax_scene_only = plt.subplots()
    charge_sim.scene.plot(z=0, ax=ax_scene_only)
    num_children_scene_only = len(ax_scene_only.get_children())
    plt.close()

    _, ax_with_bc = plt.subplots()
    charge_sim.plot(z=0, ax=ax_with_bc)
    num_children_with_bc = len(ax_with_bc.get_children())
    plt.close()

    # charge_sim.plot() should have more visual elements than scene.plot()
    # because it adds monitors and electric boundaries for CHARGE simulations
    assert num_children_with_bc - num_children_scene_only >= 2, (
        "charge_sim.plot() should add at least monitors and electric boundaries "
        "for CHARGE simulations, resulting in at least 2 more visual elements "
        "than charge_sim.scene.plot()"
    )


def test_cylinder_small_radius_warning():
    """Test that warning is issued for very small cylinder radii in HeatChargeSimulation."""
    solid = td.MultiPhysicsMedium(
        heat=td.SolidSpec(conductivity=1, capacity=1),
        name="solid",
    )
    background = td.MultiPhysicsMedium(
        heat=td.FluidSpec(),
        name="background",
    )

    # Test non-tapered cylinder with tiny radius
    tiny_cylinder = td.Structure(
        geometry=td.Cylinder(center=(0, 0, 0), radius=1e-8, length=1, axis=2),
        medium=solid,
        name="tiny",
    )
    with AssertLogLevel("WARNING", contains_str="radius"):
        _ = td.HeatChargeSimulation(
            center=(0, 0, 0),
            size=(2, 2, 2),
            medium=background,
            structures=[tiny_cylinder],
            grid_spec=td.UniformUnstructuredGrid(dl=0.1),
            monitors=[td.TemperatureMonitor(size=(1, 1, 1), name="tmp")],
        )

    # Test transformed (translated) cylinder with tiny radius
    tiny_cylinder_transformed = td.Structure(
        geometry=td.Cylinder(center=(0, 0, 0), radius=1e-8, length=1, axis=2).translated(
            x=0.1, y=0.0, z=0.0
        ),
        medium=solid,
        name="tiny_transformed",
    )
    with AssertLogLevel("WARNING", contains_str="radius"):
        _ = td.HeatChargeSimulation(
            center=(0, 0, 0),
            size=(2, 2, 2),
            medium=background,
            structures=[tiny_cylinder_transformed],
            grid_spec=td.UniformUnstructuredGrid(dl=0.1),
            monitors=[td.TemperatureMonitor(size=(1, 1, 1), name="tmp")],
        )

    # Test tapered cylinder with steep sidewall causing negative radius_top
    tapered_cylinder = td.Structure(
        geometry=td.Cylinder(
            center=(0, 0, 0),
            radius=0.1,
            length=1,
            axis=2,
            sidewall_angle=np.pi / 3,  # 60 degrees - causes negative radius_top
            reference_plane="bottom",
        ),
        medium=solid,
        name="tapered",
    )
    with AssertLogLevel("WARNING", contains_str="radius_top"):
        _ = td.HeatChargeSimulation(
            center=(0, 0, 0),
            size=(2, 2, 2),
            medium=background,
            structures=[tapered_cylinder],
            grid_spec=td.UniformUnstructuredGrid(dl=0.1),
            monitors=[td.TemperatureMonitor(size=(1, 1, 1), name="tmp")],
        )


@pytest.mark.parametrize(
    "geometry,expect_error",
    [
        (
            td.PolySlab(
                vertices=((0, 0), (1, 0), (1, 1), (0, 1)),
                bulges=[0.3, 0, 0, 0],
                slab_bounds=(-0.5, 0.5),
                axis=2,
            ),
            True,
        ),
        (
            td.PolySlab(
                vertices=((0, 0), (1, 0), (1, 1), (0, 1)),
                bulges=[0.3, 0, 0, 0],
                slab_bounds=(-0.5, 0.5),
                axis=2,
            ).translated(x=0.1, y=0.0, z=0.0),
            True,
        ),
        (
            td.PolySlab(
                vertices=((0, 0), (1, 0), (1, 1), (0, 1)),
                slab_bounds=(-0.5, 0.5),
                axis=2,
            ),
            False,
        ),
        (
            td.PolySlab(
                vertices=((0, 0), (1, 0), (1, 1), (0, 1)),
                bulges=[0, 0, 0, 0],
                slab_bounds=(-0.5, 0.5),
                axis=2,
            ),
            False,
        ),
    ],
)
def test_polyslab_arc_unsupported(geometry, expect_error):
    """Curved PolySlabs should be rejected by TCAD simulation validation."""
    solid = td.MultiPhysicsMedium(
        heat=td.SolidSpec(conductivity=1, capacity=1),
        name="solid",
    )
    background = td.MultiPhysicsMedium(
        heat=td.FluidSpec(),
        name="background",
    )
    structure = td.Structure(geometry=geometry, medium=solid, name="poly")
    kwargs = {
        "center": (0, 0, 0),
        "size": (2, 2, 2),
        "medium": background,
        "structures": [structure],
        "grid_spec": td.UniformUnstructuredGrid(dl=0.1),
        "monitors": [td.TemperatureMonitor(size=(1, 1, 1), name="tmp")],
    }

    if expect_error:
        with pytest.raises(ValidationError, match=r"arc segments in 'PolySlab'"):
            td.HeatChargeSimulation(**kwargs)
    else:
        td.HeatChargeSimulation(**kwargs)
