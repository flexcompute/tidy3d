from __future__ import annotations

import warnings

import numpy as np
import pytest
from matplotlib import pyplot as plt
from pydantic import ValidationError

import tidy3d as td
from tidy3d import (
    ConvectionBC,
    DistanceUnstructuredGrid,
    FluidSpec,
    HeatBoundarySpec,
    HeatFluxBC,
    HeatSimulation,
    HeatSimulationData,
    HeatSource,
    MediumMediumInterface,
    SimulationBoundary,
    SolidSpec,
    StructureBoundary,
    StructureSimulationBoundary,
    StructureStructureInterface,
    TemperatureBC,
    TemperatureData,
    TemperatureMonitor,
    UniformUnstructuredGrid,
)
from tidy3d.exceptions import DataError

from ..utils import AssertLogLevel, assert_single_value_error_loc, cartesian_to_unstructured


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
            density=1,
        ),
        name="solid_medium",
    )

    return fluid_medium, solid_medium


def test_heat_medium():
    _, solid_medium = make_heat_mediums()

    with pytest.raises(ValidationError):
        _ = solid_medium.heat_spec.updated_copy(capacity=-1)

    with pytest.raises(ValidationError):
        _ = solid_medium.heat_spec.updated_copy(conductivity=-1)

    # advection velocity defaults to None (pure conduction)
    assert solid_medium.heat_spec.velocity is None

    # List and ndarray inputs are intentional coverage. `updated_copy()` should normalize
    # accepted sequence inputs to tuples before Pydantic serializes the copied model.
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        # velocity can be set as a 3-vector (tidy3d units, um/s); tuple, list, and
        # np.ndarray inputs all normalize to the same tuple
        for vel in [(1.0, 0.0, -2.0), [1.0, 0.0, -2.0], np.array([1.0, 0.0, -2.0])]:
            moving_solid = solid_medium.heat_spec.updated_copy(velocity=vel)
            assert moving_solid.velocity == (1.0, 0.0, -2.0)

    # check we can create solid medium from  SI units; list/array inputs normalize too
    for vel in [(1.0, 0.0, -2.0), [1.0, 0.0, -2.0], np.array([1.0, 0.0, -2.0])]:
        solid_from_si = td.SolidMedium.from_si_units(
            conductivity=1,
            capacity=1,
            density=1,
            velocity=vel,
        )
        assert solid_from_si.conductivity == 1e-6
        assert solid_from_si.density == 1e-18
        # m/s -> um/s
        assert solid_from_si.velocity == (1e6, 0.0, -2e6)

    # velocity is optional in from_si_units (defaults to None)
    assert td.SolidMedium.from_si_units(conductivity=1).velocity is None

    assert solid_from_si == solid_from_si.heat

    with pytest.raises(ValueError):
        _ = solid_from_si.charge

    with pytest.raises(ValueError):
        _ = solid_from_si.electrical

    with pytest.raises(ValueError):
        _ = solid_from_si.optical


def _tensor_matrix(components):
    """Rebuild the symmetric 3x3 from the packed [kxx,kyy,kzz,kxy,kxz,kyz]."""
    kxx, kyy, kzz, kxy, kxz, kyz = components
    return np.array([[kxx, kxy, kxz], [kxy, kyy, kyz], [kxz, kyz, kzz]])


def test_anisotropic_conductivity_to_tensor():
    """AnisotropicConductivity resolves principal values + optional rotation to the six
    packed symmetric tensor components [kxx,kyy,kzz,kxy,kxz,kyz]."""
    # diagonal (no rotation): principals on the diagonal, zero off-diagonals
    aniso = td.AnisotropicConductivity(xx=1.0, yy=2.0, zz=3.0)
    assert aniso.to_tensor() == (1.0, 2.0, 3.0, 0.0, 0.0, 0.0)

    # a 90-degree rotation about z swaps the xx and yy principals
    rot_z90 = td.AnisotropicConductivity(
        xx=1.0, yy=2.0, zz=3.0, rotation=td.RotationAroundAxis(axis=2, angle=np.pi / 2)
    )
    assert np.allclose(rot_z90.to_tensor(), (2.0, 1.0, 3.0, 0.0, 0.0, 0.0))

    # general rotation: the reconstructed tensor equals R @ diag @ R.T, and is
    # symmetric positive-definite (SPD by construction from positive principals)
    rotation = td.RotationAroundAxis(axis=(1, 2, 3), angle=0.9)
    aniso_rot = td.AnisotropicConductivity(xx=1.0, yy=2.0, zz=3.0, rotation=rotation)
    matrix = _tensor_matrix(aniso_rot.to_tensor())
    expected = rotation.matrix @ np.diag([1.0, 2.0, 3.0]) @ rotation.matrix.T
    assert np.allclose(matrix, expected)
    assert np.allclose(matrix, matrix.T)  # symmetric
    assert np.all(np.linalg.eigvalsh(matrix) > 0)  # positive-definite
    # eigenvalues are invariant under rotation -> the principals are preserved
    assert np.allclose(sorted(np.linalg.eigvalsh(matrix)), [1.0, 2.0, 3.0])


def test_anisotropic_conductivity_rotation_invariance():
    """Rotating an isotropic tensor (equal principals) leaves it isotropic: no
    off-diagonals appear and the diagonal is unchanged for any rotation."""
    for axis, angle in [(2, np.pi / 2), ((1, 1, 0), 0.7), ((1, 2, 3), 1.23)]:
        iso = td.AnisotropicConductivity(
            xx=5.0, yy=5.0, zz=5.0, rotation=td.RotationAroundAxis(axis=axis, angle=angle)
        )
        assert np.allclose(iso.to_tensor(), (5.0, 5.0, 5.0, 0.0, 0.0, 0.0), atol=1e-12)


def test_anisotropic_conductivity_validation():
    """Principal conductivities must be positive."""
    for bad in ((-1.0, 1.0, 1.0), (1.0, 0.0, 1.0)):
        with pytest.raises(ValidationError):
            td.AnisotropicConductivity(xx=bad[0], yy=bad[1], zz=bad[2])


def test_anisotropic_conductivity_from_components():
    """from_components diagonalizes the six symmetric components into the equivalent
    principal-values-plus-rotation form, round-tripping through to_tensor exactly."""
    # general SPD tensor with all three off-diagonals nonzero
    comps = (2.0, 3.0, 5.0, 0.7, -0.4, 0.9)  # [kxx,kyy,kzz,kxy,kxz,kyz]
    aniso = td.AnisotropicConductivity.from_components(*comps)
    assert np.allclose(aniso.to_tensor(), comps)
    # principals are the eigenvalues of the input matrix
    assert np.allclose(
        sorted([aniso.xx, aniso.yy, aniso.zz]),
        sorted(np.linalg.eigvalsh(_tensor_matrix(comps))),
    )
    # a purely diagonal input needs no rotation
    diag = td.AnisotropicConductivity.from_components(kxx=1.0, kyy=2.0, kzz=3.0)
    assert diag.rotation is None
    assert diag.to_tensor() == (1.0, 2.0, 3.0, 0.0, 0.0, 0.0)


def test_anisotropic_conductivity_from_components_rejects_non_spd():
    """from_components only accepts symmetric positive-definite tensors."""
    from tidy3d.exceptions import ValidationError as Tidy3dValidationError

    # indefinite: large off-diagonal drives a negative eigenvalue
    with pytest.raises(Tidy3dValidationError):
        td.AnisotropicConductivity.from_components(kxx=1.0, kyy=1.0, kzz=1.0, kxy=5.0)
    # negative diagonal
    with pytest.raises(Tidy3dValidationError):
        td.AnisotropicConductivity.from_components(kxx=-1.0, kyy=2.0, kzz=3.0)


def test_solid_medium_accepts_anisotropic_conductivity():
    """SolidMedium.conductivity accepts both an isotropic scalar and an
    AnisotropicConductivity tensor."""
    scalar = td.SolidMedium(capacity=2, conductivity=3)
    assert scalar.conductivity == 3

    aniso = td.AnisotropicConductivity(xx=1.0, yy=2.0, zz=3.0)
    tensor = td.SolidMedium(capacity=2, conductivity=aniso)
    assert isinstance(tensor.conductivity, td.AnisotropicConductivity)
    assert tensor.conductivity.to_tensor() == (1.0, 2.0, 3.0, 0.0, 0.0, 0.0)


def test_solid_medium_from_si_units_anisotropic():
    """from_si_units accepts an AnisotropicConductivity given in SI units and converts its
    principals to tidy3d units (W/m/K -> W/um/K); the rotation is unitless and preserved."""
    rot = td.RotationAroundAxis(axis=2, angle=0.5)
    aniso_si = td.AnisotropicConductivity(xx=7.0, yy=1.5, zz=3.0, rotation=rot)
    solid = td.SolidMedium.from_si_units(conductivity=aniso_si, capacity=1)

    assert isinstance(solid.conductivity, td.AnisotropicConductivity)
    assert solid.conductivity.rotation == rot
    # principals scaled by 1e-6 -> eigenvalues of the resolved tensor scale likewise
    matrix = _tensor_matrix(solid.conductivity.to_tensor())
    assert np.allclose(sorted(np.linalg.eigvalsh(matrix)), [1.5e-6, 3.0e-6, 7.0e-6])

    # scalar path is unchanged
    assert td.SolidMedium.from_si_units(conductivity=1.0).conductivity == 1e-6


def test_solid_medium_velocity_requires_capacity_and_density():
    """A nonzero advection velocity requires both capacity and density (the
    convection coefficient rho*cp = capacity*density must be well defined)."""
    # nonzero velocity without density -> error anchored at 'velocity'
    with pytest.raises(ValidationError) as excinfo:
        td.SolidMedium(conductivity=3, capacity=1, velocity=(1.0, 0.0, 0.0))
    assert_single_value_error_loc(excinfo, ("velocity",))

    # nonzero velocity without capacity -> error
    with pytest.raises(ValidationError) as excinfo:
        td.SolidMedium(conductivity=3, density=1, velocity=(1.0, 0.0, 0.0))
    assert_single_value_error_loc(excinfo, ("velocity",))

    # with both set, a nonzero velocity is accepted
    medium = td.SolidMedium(conductivity=3, capacity=1, density=1, velocity=(1.0, 0.0, 0.0))
    assert medium.velocity == (1.0, 0.0, 0.0)

    # a zero velocity does not trigger the requirement (pure conduction)
    zero_vel = td.SolidMedium(conductivity=3, velocity=(0.0, 0.0, 0.0))
    assert zero_vel.velocity == (0.0, 0.0, 0.0)

    # the error loc is anchored at 'velocity' regardless of list/array input form
    for vel in [[1.0, 0.0, 0.0], np.array([1.0, 0.0, 0.0])]:
        with pytest.raises(ValidationError) as excinfo:
            td.SolidMedium(conductivity=3, capacity=1, velocity=vel)
        assert_single_value_error_loc(excinfo, ("velocity",))


def test_solid_medium_velocity_must_be_finite():
    """Non-finite velocity components are rejected (the solver consumes velocity as a
    real advection speed)."""
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValidationError) as excinfo:
            td.SolidMedium(conductivity=3, capacity=1, density=1, velocity=(bad, 0.0, 0.0))
        assert_single_value_error_loc(excinfo, ("velocity",))


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
    _bc_temp, _bc_flux, _bc_conv = make_heat_bcs()

    with pytest.raises(ValidationError):
        _ = TemperatureBC(temperature=-10)

    with pytest.raises(ValidationError):
        _ = ConvectionBC(ambient_temperature=-400, transfer_coeff=0.2)

    with pytest.raises(ValidationError):
        _ = ConvectionBC(ambient_temperature=400, transfer_coeff=-0.2)

    # Test vertical natural convection model in ConvectionBC
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

    with pytest.raises(ValidationError):
        td.VerticalNaturalConvectionCoeffModel(medium=air.heat, plate_length=-10)

    _, solid_medium = make_heat_mediums()
    with pytest.raises(ValidationError):
        td.VerticalNaturalConvectionCoeffModel(medium=solid_medium.heat_spec, plate_length=1e5)


def test_thermal_contact_resistance():
    """Interfacial thermal resistance BC: units, SI conversion, value and placement checks."""
    bc = td.ThermalContactResistance(resistance=3e3)
    assert bc.resistance == 3e3

    # SI input is in m^2*K/W; tidy3d-native units are K*um^2/W
    bc_si = td.ThermalContactResistance.from_si_units(resistance=3e-9)
    assert np.isclose(bc_si.resistance, 3e3)

    for bad_resistance in (0, -1, np.inf, np.nan):
        with pytest.raises(ValidationError):
            _ = td.ThermalContactResistance(resistance=bad_resistance)

    # placement validation: only material interfaces are allowed
    solid = td.Medium(
        heat_spec=td.SolidSpec(conductivity=1, capacity=1, density=1),
        name="solid",
    )
    slab1 = td.Structure(
        geometry=td.Box(center=(-0.5, 0, 0), size=(1, 1, 1)), medium=solid, name="slab1"
    )
    slab2 = td.Structure(
        geometry=td.Box(center=(0.5, 0, 0), size=(1, 1, 1)), medium=solid, name="slab2"
    )

    bc_anchor = td.HeatChargeBoundarySpec(
        placement=StructureBoundary(structure="slab1"),
        condition=TemperatureBC(temperature=300),
    )
    bc_interface = td.HeatChargeBoundarySpec(
        placement=StructureStructureInterface(structures=["slab1", "slab2"]),
        condition=bc,
    )

    sim = td.HeatChargeSimulation(
        size=(3, 2, 2),
        medium=td.Medium(heat_spec=td.FluidSpec(), name="fluid"),
        structures=[slab1, slab2],
        grid_spec=UniformUnstructuredGrid(
            dl=0.2, min_edges_per_circumference=15, min_edges_per_side=2
        ),
        boundary_spec=[bc_anchor, bc_interface],
        monitors=[TemperatureMonitor(size=(3, 2, 2), name="temperature")],
    )
    assert isinstance(sim.boundary_spec[1].condition, td.ThermalContactResistance)

    with pytest.raises(ValidationError) as excinfo:
        _ = sim.updated_copy(
            boundary_spec=[
                bc_anchor,
                td.HeatChargeBoundarySpec(placement=SimulationBoundary(), condition=bc),
            ]
        )
    assert_single_value_error_loc(excinfo, ("boundary_spec", 1, "placement"))

    # both sides of the interface must be solid heat regions: an interface that references
    # a non-solid (fluid) side is rejected at construction, anchored to the placement loc
    fluid_slab = td.Structure(
        geometry=td.Box(center=(1.5, 0, 0), size=(1, 1, 1)),
        medium=td.Medium(heat_spec=td.FluidSpec(), name="fluid_slab"),
        name="fluid_slab",
    )
    with pytest.raises(ValidationError) as excinfo:
        _ = sim.updated_copy(
            structures=[slab1, slab2, fluid_slab],
            boundary_spec=[
                bc_anchor,
                td.HeatChargeBoundarySpec(
                    placement=StructureStructureInterface(structures=["slab2", "fluid_slab"]),
                    condition=bc,
                ),
            ],
        )
    assert_single_value_error_loc(excinfo, ("boundary_spec", 1, "placement"))

    # a simulation whose only heat BCs are resistances has no temperature anchor
    with pytest.raises(ValidationError):
        _ = sim.updated_copy(boundary_spec=[bc_interface])


def make_heat_mnts():
    temp_mnt1 = TemperatureMonitor(size=(1.6, 2, 3), name="test")
    temp_mnt2 = TemperatureMonitor(size=(1.6, 2, 3), name="tet", unstructured=True)
    temp_mnt3 = TemperatureMonitor(
        center=(0, 0.9, 0), size=(1.6, 0, 3), name="tri", unstructured=True
    )
    temp_mnt4 = TemperatureMonitor(
        center=(0, 0.9, 0), size=(1.6, 0, 3), name="empty", unstructured=True
    )
    temp_mnt5 = TemperatureMonitor(center=(0, 0.7, 0.8), size=(3, 0, 0), name="line")
    temp_mnt6 = TemperatureMonitor(center=(0.7, 0.6, 0.8), size=(0, 0, 0), name="point")

    return (temp_mnt1, temp_mnt2, temp_mnt3, temp_mnt4, temp_mnt5, temp_mnt6)


def test_heat_mnt():
    temp_mnt, _, _, _, _, _ = make_heat_mnts()

    with pytest.raises(ValidationError):
        _ = temp_mnt.updated_copy(name=None)

    with pytest.raises(ValidationError):
        _ = temp_mnt.updated_copy(size=(-1, 2, 3))


def make_heat_mnt_data():
    temp_mnt1, temp_mnt2, temp_mnt3, temp_mnt4, temp_mnt5, temp_mnt6 = make_heat_mnts()

    nx, ny, nz = 9, 6, 5
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 2, ny)
    z = np.linspace(0, 3, nz)
    T = np.random.default_rng().uniform(300, 350, (nx, ny, nz))
    coords = {"x": x, "y": y, "z": z}
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
    coords = {"x": x, "y": y, "z": z}
    temperature_field = td.SpatialDataArray(T, coords=coords)

    mnt_data5 = TemperatureData(monitor=temp_mnt5, temperature=temperature_field)

    nx, ny, nz = 1, 1, 1
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 2, ny)
    z = np.linspace(0, 3, nz)
    T = np.random.default_rng().uniform(300, 350, (nx, ny, nz))
    coords = {"x": x, "y": y, "z": z}
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
    with pytest.raises(ValidationError):
        _ = grid_spec.updated_copy(dl=0)
    with pytest.raises(ValidationError):
        _ = grid_spec.updated_copy(min_edges_per_circumference=-1)
    with pytest.raises(ValidationError):
        _ = grid_spec.updated_copy(min_edges_per_side=-1)
    with pytest.raises(ValidationError):
        _ = grid_spec.updated_copy(relative_min_dl=-1e-4)

    # Zero is accepted on both fields and skips the corresponding sizing contribution.
    _ = grid_spec.updated_copy(min_edges_per_circumference=0, min_edges_per_side=0)

    # Direct construction without the two fields warns: defaults will change to 0.
    with AssertLogLevel("WARNING"):
        _ = UniformUnstructuredGrid(dl=0.1)
    # Setting only one of the two still warns about the unset field.
    with AssertLogLevel("WARNING"):
        _ = UniformUnstructuredGrid(dl=0.1, min_edges_per_circumference=15)
    # Setting both fields explicitly silences the warning, including 0.
    with AssertLogLevel(None):
        _ = UniformUnstructuredGrid(dl=0.1, min_edges_per_circumference=0, min_edges_per_side=0)
    with AssertLogLevel(None):
        _ = UniformUnstructuredGrid(dl=0.1, min_edges_per_circumference=15, min_edges_per_side=2)

    grid_spec = make_distance_grid_spec()
    _ = grid_spec.updated_copy(relative_min_dl=0)
    with pytest.raises(ValidationError):
        _ = grid_spec.updated_copy(dl_interface=-1)
    with pytest.raises(ValidationError):
        _ = grid_spec.updated_copy(distance_interface=2, distance_bulk=1)


def make_heat_source():
    return HeatSource(structures=["solid_structure"], rate=100)


def make_custom_heat_source():
    return HeatSource(
        structures=["solid_structure"],
        rate=td.SpatialDataArray(
            np.ones((1, 2, 3)), coords={"x": [0], "y": [1, 2], "z": [3, 4, 5]}
        ),
    )


def test_heat_source():
    source = make_heat_source()
    source = make_custom_heat_source()
    with pytest.raises(ValidationError):
        _ = source.updated_copy(structures=())


def make_heat_sim(include_custom_source: bool = True):
    fluid_medium, _solid_medium = make_heat_mediums()
    fluid_structure, solid_structure = make_heat_structures()
    bc_temp, bc_flux, bc_conv = make_heat_bcs()
    sources = [make_heat_source()]
    if include_custom_source:
        sources += [make_custom_heat_source()]

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
        sources=sources,
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
        with pytest.raises(ValidationError):
            _ = heat_sim.updated_copy(boundary_spec=(pl,))

    with pytest.raises(ValidationError):
        _ = heat_sim.updated_copy(sources=(HeatSource(structures=["noname"]),), rate=-10)

    # run 2D case
    _ = heat_sim.updated_copy(
        center=(0.7, 0, 0), size=(0, 2, 2), monitors=tuple(heat_sim.monitors[:5])
    )

    # test unsupported 1D heat domains
    with pytest.raises(ValidationError):
        _ = heat_sim.updated_copy(center=(1, 1, 1), size=(1, 0, 0))

    temp_mnt = heat_sim.monitors[0]

    with pytest.raises(ValidationError):
        heat_sim.updated_copy(monitors=(temp_mnt, temp_mnt))

    _ = heat_sim.plot(x=0)
    plt.close()

    _ = heat_sim.plot_property(y=0, property="heat_conductivity")
    plt.close()

    heat_sim_sym = heat_sim.updated_copy(symmetry=(0, 1, 1))
    _ = heat_sim_sym.plot_property(z=0, property="source")
    plt.close()

    # no negative symmetry
    with pytest.raises(ValidationError):
        _ = heat_sim.updated_copy(symmetry=(-1, 0, 1))

    # no SolidSpec in the entire simulation
    bc_spec = td.HeatBoundarySpec(
        placement=td.SimulationBoundary(), condition=td.TemperatureBC(temperature=300)
    )
    solid_med = heat_sim.structures[1].medium

    _ = heat_sim.updated_copy(structures=(), medium=solid_med, sources=(), boundary_spec=(bc_spec,))
    with pytest.raises(ValidationError):
        _ = heat_sim.updated_copy(structures=(), sources=(), boundary_spec=(bc_spec,), monitors=())

    _ = heat_sim.updated_copy(
        structures=(heat_sim.structures[0],), medium=solid_med, boundary_spec=(bc_spec,), sources=()
    )
    with pytest.raises(ValidationError):
        _ = heat_sim.updated_copy(
            structures=(heat_sim.structures[0],), boundary_spec=(bc_spec,), sources=(), monitors=()
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
    with pytest.raises(ValidationError):
        _ = heat_sim.updated_copy(structures=(*heat_sim.structures, struct_1d))

    with pytest.raises(ValidationError):
        _ = heat_sim.updated_copy(structures=(*heat_sim.structures, struct_2d))

    # no data expected inside a monitor
    for mnt_size in [(0.2, 0.2, 0.2), (0, 1, 1), (0, 2, 0), (0, 0, 0)]:
        temp_mnt = td.TemperatureMonitor(center=(0, 0, 0), size=mnt_size, name="test")

        with pytest.raises(ValidationError):
            _ = heat_sim.updated_copy(monitors=(temp_mnt,))


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
            grid_spec=td.UniformUnstructuredGrid(
                dl=0.1, min_edges_per_circumference=15, min_edges_per_side=2
            ),
            monitors=[
                td.TemperatureMonitor(
                    center=(0, 0, 0),
                    size=(td.inf, td.inf, td.inf),
                    name="test_monitor",
                    unstructured=True,
                )
            ],
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
            monitors=[
                td.TemperatureMonitor(
                    center=(0, 0, 0),
                    size=(td.inf, td.inf, td.inf),
                    name="test_monitor",
                    unstructured=True,
                )
            ],
            grid_spec=td.UniformUnstructuredGrid(
                dl=0.1, min_edges_per_circumference=15, min_edges_per_side=2
            ),
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

    with pytest.raises(ValidationError):
        _ = heat_sim_data.updated_copy(data=(heat_sim_data.data[0],) * 2)

    temp_mnt = TemperatureMonitor(size=(1, 2, 3), name="test")
    temp_mnt = temp_mnt.updated_copy(name="test2")

    sim = heat_sim_data.simulation.updated_copy(monitors=(temp_mnt,))

    with pytest.raises(ValidationError):
        _ = heat_sim_data.updated_copy(simulation=sim)


def test_relative_min_dl_warning():
    with AssertLogLevel("WARNING"):
        _ = td.HeatSimulation(
            size=(1, 1, 1),
            medium=td.Medium(heat_spec=td.SolidSpec(conductivity=1, capacity=2)),
            grid_spec=td.UniformUnstructuredGrid(
                dl=0.0001,
                min_edges_per_circumference=15,
                min_edges_per_side=2,
                relative_min_dl=1e-2,
            ),
            boundary_spec=[
                td.HeatBoundarySpec(
                    placement=td.SimulationBoundary(), condition=td.TemperatureBC(temperature=300)
                )
            ],
            monitors=[
                td.TemperatureMonitor(
                    center=(0, 0, 0),
                    size=(td.inf, td.inf, td.inf),
                    name="test_monitor",
                    unstructured=True,
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
            monitors=[
                td.TemperatureMonitor(
                    center=(0, 0, 0),
                    size=(td.inf, td.inf, td.inf),
                    name="test_monitor",
                    unstructured=True,
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
            monitors=[
                td.TemperatureMonitor(
                    center=(0, 0, 0),
                    size=(td.inf, td.inf, td.inf),
                    name="test_monitor",
                    unstructured=True,
                )
            ],
        )


def test_sim_version_update():
    heat_sim = make_heat_sim()
    heat_sim_dict = heat_sim.model_dump()
    heat_sim_dict["version"] = "ancient_version"

    with AssertLogLevel("WARNING"):
        heat_sim_new = td.HeatSimulation.model_validate(heat_sim_dict)

    assert heat_sim_new.version == td.__version__


@pytest.mark.parametrize("zero_dim_axis", [None, 0, 2])
def test_symmetry_expanded(zero_dim_axis):
    # Test symmetry expansion with mesh that conforms to symmetry axis but not monitor bounds.
    # The mesh boundary is at the symmetry center, so after expansion there are no duplicate
    # cells at the join. But the expanded data may extend beyond the monitor, requiring clipping.
    # sel_inside keeps boundary cells, so final bounds may slightly exceed monitor bounds.
    symmetry_center = [2, 0.5, 0]
    symmetry = [1, 1, 1]

    lens = [1, 2, 2]
    num_points = [7, 5, 11]

    if zero_dim_axis is not None:
        lens[zero_dim_axis] = 0
        num_points[zero_dim_axis] = 1

    # Monitor is smaller than the expanded data, requiring clipping
    # For x: data [3, 4] mirrors around x=2 to [0, 1], monitor = [0, 1] (exact match)
    # For y: data [0.5, 2.5] expands around y=0.5 to [-1.5, 2.5], monitor = [-1, 1] (needs clipping)
    # For z: no expansion needed (monitor doesn't extend into mirror region)
    mnt_span_x = [1 - lens[0], 1]  # [0, 1] for lens[0]=1
    mnt_span_y = [-lens[1] / 2, lens[1] / 2]  # [-1, 1] for lens[1]=2
    mnt_span_z = [1, 1 + lens[2]]  # [1, 3] for lens[2]=2

    # Data with mesh boundary at symmetry center (conforms to symmetry axis)
    # - x: reflection_only mirrors [3, 4] to [0, 1]
    # - y: data starts at symmetry center, expands to cover both sides
    # - z: no expansion needed
    data_span_x = [3, 3 + lens[0]]  # [3, 4] → [0, 1] via reflection_only
    data_span_y = [symmetry_center[1], symmetry_center[1] + lens[1]]  # [0.5, 2.5] → [-1.5, 2.5]
    data_span_z = [1, 1 + lens[2]]  # [1, 3] stays as is

    mnt_bounds = np.array(list(zip(mnt_span_x, mnt_span_y, mnt_span_z)))
    mnt_size = tuple(mnt_bounds[1] - mnt_bounds[0])
    mnt_center = tuple((mnt_bounds[1] + mnt_bounds[0]) / 2)

    x = np.linspace(*data_span_x, num_points[0])
    y = np.linspace(*data_span_y, num_points[1])
    z = np.linspace(*data_span_z, num_points[2])
    v = np.sin(x[:, None, None]) * np.cos(y[None, :, None]) * np.exp(z[None, None, :])

    data_cart = td.SpatialDataArray(v, coords={"x": x, "y": y, "z": z})
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

    # Unstructured data uses sel_inside for clipping, which keeps boundary cells.
    # So the final bounds may extend slightly beyond monitor bounds, but must cover them.
    assert data_expanded_ugrid.does_cover(mnt_bounds)
    assert data_expanded_cart.does_cover(mnt_bounds)


def test_unsteady_setup():
    """Test that unsteady setup works correctly."""

    _, solid_medium = make_heat_mediums()
    solid_structure = td.Structure(
        geometry=td.Box(center=(0, 0, 0), size=(1, 1, 1)),
        medium=solid_medium,
        name="solid_structure",
    )

    heat_sim = make_heat_sim(include_custom_source=False)
    unsteady_spec = td.UnsteadyHeatAnalysis(
        initial_temperature=300, unsteady_spec=td.UnsteadySpec(time_step=0.1, total_time_steps=100)
    )

    temp_mnt = TemperatureMonitor(size=(1, 1, 1), name="mnt", unstructured=True)
    bc = HeatBoundarySpec(
        condition=td.TemperatureBC(temperature=300),
        placement=StructureBoundary(structure="solid_structure"),
    )

    heat_sim = heat_sim.updated_copy(
        structures=(solid_structure,),
        analysis_spec=unsteady_spec,
        monitors=(temp_mnt,),
        boundary_spec=(bc,),
    )

    with pytest.raises(ValidationError):
        solid_medium = td.MultiPhysicsMedium(
            heat=td.SolidMedium(
                conductivity=3,
            ),
            name="solid_medium",
        )
        new_struct = solid_structure.updated_copy(medium=solid_medium)
        _ = heat_sim.updated_copy(structures=(new_struct,))

    with pytest.raises(ValidationError):
        solid_medium = td.MultiPhysicsMedium(
            heat=td.SolidMedium(
                conductivity=3,
                capacity=2,
            ),
            name="solid_medium",
        )
        new_struct = solid_structure.updated_copy(medium=solid_medium)
        _ = heat_sim.updated_copy(structures=(new_struct,))

    with pytest.raises(ValidationError):
        solid_medium = td.MultiPhysicsMedium(
            heat=td.SolidMedium(
                conductivity=3,
                density=2,
            ),
            name="solid_medium",
        )
        new_struct = solid_structure.updated_copy(medium=solid_medium)
        _ = heat_sim.updated_copy(structures=(new_struct,))
