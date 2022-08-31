from typing import Dict
import pytest
import numpy as np
import pydantic

from tidy3d import *
from tidy3d.log import ValidationError, SetupError
from tidy3d.components.simulation import MAX_NUM_MEDIUMS
from .utils import assert_log_level


def test_sim():
    """make sure a simulation can be initialized"""

    sim = Simulation(
        size=(2.0, 2.0, 2.0),
        run_time=1e-12,
        structures=[
            Structure(
                geometry=Box(size=(1, 1, 1), center=(-1, 0, 0)),
                medium=Medium(permittivity=2.0),
            ),
            Structure(
                geometry=Box(size=(1, 1, 1), center=(0, 0, 0)),
                medium=Medium(permittivity=1.0, conductivity=3.0),
            ),
            Structure(geometry=Sphere(radius=1.4, center=(1.0, 0.0, 1.0)), medium=Medium()),
            Structure(
                geometry=Cylinder(radius=1.4, length=2.0, center=(1.0, 0.0, -1.0), axis=1),
                medium=Medium(),
            ),
        ],
        sources=[
            UniformCurrentSource(
                size=(0, 0, 0),
                center=(0, -0.5, 0),
                polarization="Hx",
                source_time=GaussianPulse(
                    freq0=1e14,
                    fwidth=1e12,
                ),
                name="my_dipole",
            ),
            PointDipole(
                center=(0, 0, 0),
                polarization="Ex",
                source_time=GaussianPulse(
                    freq0=1e14,
                    fwidth=1e12,
                ),
            ),
        ],
        monitors=[
            FieldMonitor(size=(0, 0, 0), center=(0, 0, 0), freqs=[1, 2], name="point"),
            FluxTimeMonitor(size=(1, 1, 0), center=(0, 0, 0), interval=10, name="plane"),
        ],
        symmetry=(0, 1, -1),
        boundary_spec=BoundarySpec(
            x=Boundary.pml(num_layers=20),
            y=Boundary.stable_pml(num_layers=30),
            z=Boundary.absorber(num_layers=100),
        ),
        shutoff=1e-6,
        courant=0.8,
        subpixel=False,
    )

    dt = sim.dt
    tm = sim.tmesh
    sim.validate_pre_upload()
    ms = sim.mediums
    mm = sim.medium_map
    m = sim.get_monitor_by_name("point")
    s = sim.background_structure
    # sim.plot(x=0)
    # sim.plot_eps(x=0)
    sim.num_pml_layers
    # sim.plot_grid(x=0)
    sim.frequency_range
    sim.grid
    sim.num_cells
    sim.discretize(m)
    sim.epsilon(m)


def _test_version():
    """ensure there's a version in simulation"""

    sim = Simulation(
        size=(1, 1, 1),
        run_time=1e-12,
    )
    path = "tests/tmp/simulation.json"
    sim.to_file("tests/tmp/simulation.json")
    with open(path, "r") as f:
        s = f.read()
        assert '"version": ' in s


def test_sim_bounds():
    """make sure bounds are working correctly"""

    # make sure all things are shifted to this central location
    CENTER_SHIFT = (-1.0, 1.0, 100.0)

    def place_box(center_offset):

        shifted_center = tuple(c + s for (c, s) in zip(center_offset, CENTER_SHIFT))

        sim = Simulation(
            size=(1, 1, 1),
            center=CENTER_SHIFT,
            grid_spec=GridSpec(wavelength=1.0),
            run_time=1e-12,
            structures=[
                Structure(geometry=Box(size=(1, 1, 1), center=shifted_center), medium=Medium())
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
            center = amp * sign
            place_box(tuple(center))

    # test all cases where box is shifted +/- 2 in x,y,z and no longer intersects
    for amp in bin_ints:
        for sign in bin_signs:
            center = 2 * amp * sign
            if np.sum(center) < 1e-12:
                continue
            with pytest.raises(SetupError) as e_info:
                place_box(tuple(center))


def test_sim_size():

    mesh1d = UniformGrid(dl=1e-6)
    grid_spec = GridSpec(grid_x=mesh1d, grid_y=mesh1d, grid_z=mesh1d)

    with pytest.raises(SetupError):
        s = Simulation(size=(1, 1, 1), grid_spec=grid_spec, run_time=1e-12)
        s._validate_size()

    with pytest.raises(SetupError):
        s = Simulation(size=(1, 1, 1), run_time=1e-7)
        s._validate_size()


def _test_monitor_size():

    with pytest.raises(SetupError):
        s = Simulation(
            size=(1, 1, 1),
            grid_spec=GridSpec.uniform(1e-3),
            monitors=[
                FieldMonitor(size=(inf, inf, inf), freqs=np.linspace(0, 200e12, 10001), name="test")
            ],
            run_time=1e-12,
        )
        s.validate_pre_upload()


@pytest.mark.parametrize("freq, log_level", [(1.5, 30), (2.5, None), (3.5, 30)])
def test_monitor_medium_frequency_range(caplog, freq, log_level):
    # monitor frequency above or below a given medium's range should throw a warning

    size = (1, 1, 1)
    medium = Medium(frequency_range=(2, 3))
    box = Structure(geometry=Box(size=(0.1, 0.1, 0.1)), medium=medium)
    mnt = FieldMonitor(size=(0, 0, 0), name="freq", freqs=[freq])
    src = UniformCurrentSource(
        source_time=GaussianPulse(freq0=2.5, fwidth=0.5),
        size=(0, 0, 0),
        polarization="Ex",
    )
    sim = Simulation(
        size=(1, 1, 1),
        structures=[box],
        monitors=[mnt],
        sources=[src],
        run_time=1e-12,
    )
    assert_log_level(caplog, log_level)


@pytest.mark.parametrize("fwidth, log_level", [(0.1, 30), (2, None)])
def test_monitor_simulation_frequency_range(caplog, fwidth, log_level):
    # monitor frequency outside of the simulation's frequency range should throw a warning

    size = (1, 1, 1)
    src = UniformCurrentSource(
        source_time=GaussianPulse(freq0=2.0, fwidth=fwidth),
        size=(0, 0, 0),
        polarization="Ex",
    )
    mnt = FieldMonitor(size=(0, 0, 0), name="freq", freqs=[1.5])
    sim = Simulation(size=(1, 1, 1), monitors=[mnt], sources=[src], run_time=1e-12)
    assert_log_level(caplog, log_level)


def test_monitor_freqs_empty():
    # errors when no frequencies supplied

    with pytest.raises(ValidationError):
        monitor = FieldMonitor(
            size=(inf, inf, inf),
            freqs=[],
            name="test",
            interval_space=(1, 1, 1),
        )


def test_monitor_downsampling():
    # test that the downsampling parameters can be set and the colocation validator works

    monitor = FieldMonitor(
        size=(inf, inf, inf),
        freqs=np.linspace(0, 200e12, 10001),
        name="test",
        interval_space=(1, 1, 1),
    )
    assert monitor.colocate is False

    monitor = FieldMonitor(
        size=(inf, inf, inf),
        freqs=np.linspace(0, 200e12, 10001),
        name="test",
        interval_space=(1, 2, 3),
    )
    assert monitor.colocate is True

    monitor = FieldMonitor(
        size=(inf, inf, inf),
        freqs=np.linspace(0, 200e12, 10001),
        name="test",
        interval_space=(1, 2, 3),
        colocate=False,
    )
    assert monitor.colocate is False


def test_diffraction_validators():

    # ensure error if boundaries are not periodic
    boundary_spec = BoundarySpec(
        x=Boundary.pml(),
        y=Boundary.periodic(),
        z=Boundary.pml(),
    )
    with pytest.raises(SetupError) as e_info:
        sim = Simulation(
            size=(2, 2, 2),
            run_time=1e-12,
            structures=[Structure(geometry=Box(size=(1, 1, 1)), medium=Medium())],
            boundary_spec=boundary_spec,
            monitors=[DiffractionMonitor(size=[inf, inf, 0], freqs=[1e12], name="de")],
        )

    # ensure error if monitor isn't infinite in two directions
    with pytest.raises(SetupError) as e_info:
        monitor = DiffractionMonitor(size=[inf, 4, 0], freqs=[1e12], name="de")

    # ensure error if orders are non-unique
    with pytest.raises(SetupError) as e_info:
        monitor = DiffractionMonitor(
            size=[inf, inf, 0], freqs=[1e12], name="de", orders_x=[0, 1, 1]
        )
    with pytest.raises(SetupError) as e_info:
        monitor = DiffractionMonitor(
            size=[inf, inf, 0], freqs=[1e12], name="de", orders_y=[0, 1, 1]
        )


@pytest.mark.parametrize("grid_size,log_level", [(0.001, None), (3, 30)])
def test_large_grid_size(caplog, grid_size, log_level):
    # small fwidth should be inside range, large one should throw warning

    medium = Medium(permittivity=2, frequency_range=(2e14, 3e14))
    box = Structure(geometry=Box(size=(0.1, 0.1, 0.1)), medium=medium)
    src = PointDipole(
        source_time=GaussianPulse(freq0=2.5e14, fwidth=1e12),
        polarization="Ex",
    )
    _ = Simulation(
        size=(1, 1, 1),
        grid_spec=GridSpec.uniform(dl=grid_size),
        structures=[box],
        sources=[src],
        run_time=1e-12,
    )

    assert_log_level(caplog, log_level)


@pytest.mark.parametrize("box_size,log_level", [(0.001, None), (9.9, 30), (20, None)])
def test_sim_structure_gap(caplog, box_size, log_level):
    """Make sure the gap between a structure and PML is not too small compared to lambda0."""
    medium = Medium(permittivity=2)
    box = Structure(geometry=Box(size=(box_size, box_size, box_size)), medium=medium)
    src = UniformCurrentSource(
        source_time=GaussianPulse(freq0=3e14, fwidth=1e13),
        size=(0, 0, 0),
        polarization="Ex",
    )
    sim = Simulation(
        size=(10, 10, 10),
        structures=[box],
        sources=[src],
        boundary_spec=BoundarySpec(
            x=Boundary.pml(num_layers=5), y=Boundary.pml(num_layers=5), z=Boundary.pml(num_layers=5)
        ),
        run_time=1e-12,
    )
    assert_log_level(caplog, log_level)


def test_sim_plane_wave_error():
    """ "Make sure we error if plane wave is not intersecting homogeneous region of simulation."""

    medium_bg = Medium(permittivity=2)
    medium_air = Medium(permittivity=1)

    box = Structure(geometry=Box(size=(0.1, 0.1, 0.1)), medium=medium_air)

    box_transparent = Structure(geometry=Box(size=(0.1, 0.1, 0.1)), medium=medium_bg)

    src = PlaneWave(
        source_time=GaussianPulse(freq0=2.5e14, fwidth=1e13),
        center=(0, 0, 0),
        size=(inf, inf, 0),
        direction="+",
        pol_angle=-1.0,
    )

    # with transparent box continue
    _ = Simulation(
        size=(1, 1, 1),
        medium=medium_bg,
        structures=[box_transparent],
        sources=[src],
        run_time=1e-12,
    )

    # with non-transparent box, raise
    with pytest.raises(SetupError):
        _ = Simulation(
            size=(1, 1, 1),
            medium=medium_bg,
            structures=[box_transparent, box],
            sources=[src],
        )


@pytest.mark.parametrize(
    "box_size,log_level",
    [((0.1, 0.1, 0.1), None), ((1, 0.1, 0.1), 30), ((0.1, 1, 0.1), 30), ((0.1, 0.1, 1), 30)],
)
def test_sim_structure_extent(caplog, box_size, log_level):
    """Make sure we warn if structure extends exactly to simulation edges."""

    src = UniformCurrentSource(
        source_time=GaussianPulse(freq0=3e14, fwidth=1e13),
        size=(0, 0, 0),
        polarization="Ex",
    )
    box = Structure(geometry=Box(size=box_size), medium=Medium(permittivity=2))
    sim = Simulation(size=(1, 1, 1), structures=[box], sources=[src], run_time=1e-12)

    assert_log_level(caplog, log_level)


def test_num_mediums():
    """Make sure we error if too many mediums supplied."""

    structures = []
    grid_spec = GridSpec.auto(wavelength=1.0)
    for i in range(MAX_NUM_MEDIUMS):
        structures.append(
            Structure(geometry=Box(size=(1, 1, 1)), medium=Medium(permittivity=i + 1))
        )
    sim = Simulation(size=(5, 5, 5), grid_spec=grid_spec, structures=structures, run_time=1e-12)

    with pytest.raises(SetupError):
        structures.append(
            Structure(geometry=Box(size=(1, 1, 1)), medium=Medium(permittivity=i + 2))
        )
        sim = Simulation(size=(5, 5, 5), grid_spec=grid_spec, structures=structures, run_time=1e-12)


""" geometry """


def test_geometry():

    b = Box(size=(1, 1, 1), center=(0, 0, 0))
    s = Sphere(radius=1, center=(0, 0, 0))
    s = Cylinder(radius=1, center=(0, 0, 0), axis=1, length=1)
    s = PolySlab(vertices=((1, 2), (3, 4), (5, 4)), slab_bounds=(-1, 1), axis=2)
    # vertices_np = np.array(s.vertices)
    # s_np = PolySlab(vertices=vertices_np, slab_bounds=(-1, 1), axis=1)

    # make sure wrong axis arguments error
    with pytest.raises(pydantic.ValidationError) as e_info:
        s = Cylinder(radius=1, center=(0, 0, 0), axis=-1, length=1)
    with pytest.raises(pydantic.ValidationError) as e_info:
        s = PolySlab(radius=1, center=(0, 0, 0), axis=-1, slab_bounds=(-0.5, 0.5))
    with pytest.raises(pydantic.ValidationError) as e_info:
        s = Cylinder(radius=1, center=(0, 0, 0), axis=3, length=1)
    with pytest.raises(pydantic.ValidationError) as e_info:
        s = PolySlab(radius=1, center=(0, 0, 0), axis=3, slab_bounds=(-0.5, 0.5))

    # make sure negative values error
    with pytest.raises(pydantic.ValidationError) as e_info:
        s = Sphere(radius=-1, center=(0, 0, 0))
    with pytest.raises(pydantic.ValidationError) as e_info:
        s = Cylinder(radius=-1, center=(0, 0, 0), axis=3, length=1)
    with pytest.raises(pydantic.ValidationError) as e_info:
        s = Cylinder(radius=1, center=(0, 0, 0), axis=3, length=-1)


def test_geometry_sizes():

    # negative in size kwargs errors
    for size in (-1, 1, 1), (1, -1, 1), (1, 1, -1):
        with pytest.raises(pydantic.ValidationError) as e_info:
            a = Box(size=size, center=(0, 0, 0))
        with pytest.raises(pydantic.ValidationError) as e_info:
            s = Simulation(size=size, run_time=1e-12, grid_spec=GridSpec(wavelength=1.0))

    # negative grid sizes error?
    with pytest.raises(pydantic.ValidationError) as e_info:
        s = Simulation(size=(1, 1, 1), grid_spec=GridSpec.uniform(dl=-1.0), run_time=1e-12)


@pytest.mark.parametrize("x0", [5])
def test_geometry_touching_intersections(x0):
    """Two touching boxes should show at least one intersection at plane where they touch."""

    # size of each box
    # L = 1 # works
    # L = 0.1 # works
    # L = 0.12 # assertion errors
    L = 0.24  # assertion errors
    # L = 0.25 # works

    # one box to the left of x0 and one box to the right of x0, touching at x0
    b1 = Box(center=(x0 - L / 2, 0, 0), size=(L, L, L))
    b2 = Box(center=(x0 + L / 2, 0, 0), size=(L, L, L))

    ints1 = b1.intersections(x=x0)
    ints2 = b2.intersections(x=x0)

    ints_total = ints1 + ints2

    assert len(ints_total) > 0, "no intersections found at plane where two boxes touch"


def test_pop_axis():
    b = Box(size=(1, 1, 1))
    for axis in range(3):
        coords = (1, 2, 3)
        Lz, (Lx, Ly) = b.pop_axis(coords, axis=axis)
        _coords = b.unpop_axis(Lz, (Lx, Ly), axis=axis)
        assert all(c == _c for (c, _c) in zip(coords, _coords))
        _Lz, (_Lx, _Ly) = b.pop_axis(_coords, axis=axis)
        assert Lz == _Lz
        assert Lx == _Lx
        assert Ly == _Ly


def test_polyslab_merge():
    """make sure polyslabs from gds get merged when they should."""

    import gdspy

    def make_polyslabs(gap_size):
        """Construct two rectangular polyslabs separated by a gap."""
        lib = gdspy.GdsLibrary()
        cell = lib.new_cell(f"polygons_{gap_size:.2f}")
        rect1 = gdspy.Rectangle((gap_size / 2, 0), (1, 1))
        rect2 = gdspy.Rectangle((-1, 0), (-gap_size / 2, 1))
        cell.add(rect1)
        cell.add(rect2)
        return PolySlab.from_gds(gds_cell=cell, gds_layer=0, axis=2, slab_bounds=(-1, 1))

    polyslabs_gap = make_polyslabs(gap_size=0.3)
    assert len(polyslabs_gap) == 2, "untouching polylsabs were merged incorrectly."

    polyslabs_touching = make_polyslabs(gap_size=0)
    assert len(polyslabs_touching) == 1, "polyslabs didnt merge correctly."


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_polyslab_axis(axis):
    ps = PolySlab(slab_bounds=(-1, 1), vertices=((-5, -5), (-5, 5), (5, 5), (5, -5)), axis=axis)

    # bound test
    bounds_ideal = [-5, -5]
    bounds_ideal.insert(axis, -1)
    bounds_ideal = np.array(bounds_ideal)
    np.allclose(ps.bounds[0], bounds_ideal)
    np.allclose(ps.bounds[1], -bounds_ideal)

    # inside
    point = [0, 0]
    point.insert(axis, 3)
    assert ps.inside(point[0], point[1], point[2]) == False

    # intersections
    plane_coord = [None] * 3
    plane_coord[axis] = 3
    assert ps.intersects_plane(x=plane_coord[0], y=plane_coord[1], z=plane_coord[2]) == False
    plane_coord[axis] = -3
    assert ps.intersects_plane(x=plane_coord[0], y=plane_coord[1], z=plane_coord[2]) == False


""" medium """


def test_medium():

    # mediums error with unacceptable values
    with pytest.raises(pydantic.ValidationError) as e_info:
        m = Medium(permittivity=0.0)
    with pytest.raises(pydantic.ValidationError) as e_info:
        m = Medium(conductivity=-1.0)


def test_medium_conversions():
    n = 4.0
    k = 1.0
    freq = 3.0

    # test medium creation
    medium = Medium.from_nk(n, k, freq)

    # test consistency
    eps_z = AbstractMedium.nk_to_eps_complex(n, k)
    eps, sig = AbstractMedium.nk_to_eps_sigma(n, k, freq)
    eps_z_ = AbstractMedium.eps_sigma_to_eps_complex(eps, sig, freq)
    assert np.isclose(eps_z, eps_z_)

    n_, k_ = AbstractMedium.eps_complex_to_nk(eps_z)
    assert np.isclose(n, n_)
    assert np.isclose(k, k_)


def test_PEC():

    struct = Structure(geometry=Box(size=(1, 1, 1)), medium=PEC)


def test_medium_dispersion():

    # construct media
    m_PR = PoleResidue(eps_inf=1.0, poles=[((1 + 2j), (1 + 3j)), ((2 + 4j), (1 + 5j))])
    m_SM = Sellmeier(coeffs=[(2, 3), (2, 4)])
    m_LZ = Lorentz(eps_inf=1.0, coeffs=[(1, 3, 2), (2, 4, 1)])
    m_LZ2 = Lorentz(eps_inf=1.0, coeffs=[(1, 2, 3), (2, 1, 4)])
    m_DR = Drude(eps_inf=1.0, coeffs=[(1, 3), (2, 4)])
    m_DB = Debye(eps_inf=1.0, coeffs=[(1, 3), (2, 4)])

    with pytest.raises(pydantic.ValidationError) as e_info:
        mf_SM = Sellmeier(coeffs=[(2, 0), (2, 4)])

    with pytest.raises(pydantic.ValidationError) as e_info:
        mf_DR = Drude(eps_inf=1.0, coeffs=[(1, 0), (2, 4)])

    with pytest.raises(pydantic.ValidationError) as e_info:
        mf_DB = Debye(eps_inf=1.0, coeffs=[(1, 0), (2, 4)])

    freqs = np.linspace(0.01, 1, 1001)
    for medium in [m_PR, m_SM, m_LZ, m_LZ2, m_DR, m_DB]:
        eps_c = medium.eps_model(freqs)

    for medium in [m_SM, m_LZ, m_LZ2, m_DR, m_DB]:
        eps_c = medium.eps_model(freqs)
        assert np.all(eps_c.imag >= 0)


def test_medium_dispersion_conversion():

    m_PR = PoleResidue(eps_inf=1.0, poles=[((1 + 2j), (1 + 3j)), ((2 + 4j), (1 + 5j))])
    m_SM = Sellmeier(coeffs=[(2, 3), (2, 4)])
    m_LZ = Lorentz(eps_inf=1.0, coeffs=[(1, 3, 2), (2, 4, 1)])
    m_LZ2 = Lorentz(eps_inf=1.0, coeffs=[(1, 2, 3), (2, 1, 4)])
    m_DR = Drude(eps_inf=1.0, coeffs=[(1, 3), (2, 4)])
    m_DB = Debye(eps_inf=1.0, coeffs=[(1, 3), (2, 4)])

    freqs = np.linspace(0.01, 1, 1001)
    for medium in [m_PR, m_SM, m_DB, m_LZ, m_DR, m_LZ2]:  # , m_DB]:
        eps_model = medium.eps_model(freqs)
        eps_pr = medium.pole_residue.eps_model(freqs)
        np.testing.assert_allclose(eps_model, eps_pr)


def test_medium_dispersion_create():

    m_PR = PoleResidue(eps_inf=1.0, poles=[((1 + 2j), (1 + 3j)), ((2 + 4j), (1 + 5j))])
    m_SM = Sellmeier(coeffs=[(2, 3), (2, 4)])
    m_LZ = Lorentz(eps_inf=1.0, coeffs=[(1, 3, 2), (2, 4, 1)])
    m_LZ2 = Lorentz(eps_inf=1.0, coeffs=[(1, 2, 3), (2, 1, 4)])
    m_DR = Drude(eps_inf=1.0, coeffs=[(1, 3), (2, 4)])
    m_DB = Debye(eps_inf=1.0, coeffs=[(1, 3), (2, 4)])

    for medium in [m_PR, m_SM, m_DB, m_LZ, m_DR, m_LZ2]:
        struct = Structure(geometry=Box(size=(1, 1, 1)), medium=medium)


def test_sellmeier_from_dispersion():
    n = 3.5
    wvl = 0.5
    freq = C_0 / wvl
    dn_dwvl = -0.1
    with pytest.raises(ValidationError) as e:
        # Check that postivie dispersion raises an error
        medium = Sellmeier.from_dispersion(n=n, freq=freq, dn_dwvl=-dn_dwvl)

    # Check that medium properties are as epected
    medium = Sellmeier.from_dispersion(n=n, freq=freq, dn_dwvl=dn_dwvl)
    epses = [medium.eps_model(f) for f in [0.99 * freq, freq, 1.01 * freq]]
    ns = np.sqrt(epses)
    dn_df = (ns[2] - ns[0]) / 0.02 / freq

    assert np.allclose(ns[1], n)
    assert np.allclose(-dn_df * C_0 / wvl**2, dn_dwvl)


def eps_compare(medium: Medium, expected: Dict, tol: float = 1e-5):

    for freq, val in expected.items():
        assert np.abs(medium.eps_model(freq) - val) < tol


def test_epsilon_eval():
    """Compare epsilon evaluated from a dispersive various models to expected."""

    # Dispersive silver model
    poles_silver = [
        (a / HBAR, c / HBAR)
        for (a, c) in [
            ((-2.502e-2 - 8.626e-3j), (5.987e-1 + 4.195e3j)),
            ((-2.021e-1 - 9.407e-1j), (-2.211e-1 + 2.680e-1j)),
            ((-1.467e1 - 1.338e0j), (-4.240e0 + 7.324e2j)),
            ((-2.997e-1 - 4.034e0j), (6.391e-1 - 7.186e-2j)),
            ((-1.896e0 - 4.808e0j), (1.806e0 + 4.563e0j)),
            ((-9.396e0 - 6.477e0j), (1.443e0 - 8.219e1j)),
        ]
    ]

    material = PoleResidue(poles=poles_silver)
    expected = {
        2e14: (-102.18389652032306 + 9.22771912188222j),
        5e14: (-13.517709933590542 + 0.9384819052893092j),
    }
    eps_compare(material, expected)

    # Constant and eps, zero sigma
    material = Medium(permittivity=1.5**2)
    expected = {
        2e14: 2.25,
        5e14: 2.25,
    }
    eps_compare(material, expected)

    # Constant eps and sigma
    material = Medium(permittivity=1.5**2, conductivity=0.1)
    expected = {
        2e14: 2.25 + 8.987552009401353j,
        5e14: 2.25 + 3.5950208037605416j,
    }
    eps_compare(material, expected)

    # Constant n and k at a given frequency
    material = Medium.from_nk(n=1.5, k=0.1, freq=C_0 / 0.8)
    expected = {
        2e14: 2.24 + 0.5621108598392753j,
        5e14: 2.24 + 0.22484434393571015j,
    }
    eps_compare(material, expected)

    # Anisotropic material
    eps = (1.5, 2.0, 2.3)
    sig = (0.01, 0.03, 0.015)
    mediums = [Medium(permittivity=eps[i], conductivity=sig[i]) for i in range(3)]
    material = AnisotropicMedium(xx=mediums[0], yy=mediums[1], zz=mediums[2])

    eps_diag_2 = material.eps_diagonal(2e14)
    eps_diag_5 = material.eps_diagonal(5e14)
    assert np.all(np.array(eps_diag_2) == np.array([medium.eps_model(2e14) for medium in mediums]))

    expected = {2e14: np.mean(eps_diag_2), 5e14: np.mean(eps_diag_5)}
    eps_compare(material, expected)


""" modes """


def test_modes():

    m = ModeSpec(num_modes=2)
    m = ModeSpec(num_modes=1, target_neff=1.0)


""" names """


def _test_names_default():
    """makes sure default names are set"""

    sim = Simulation(
        size=(2.0, 2.0, 2.0),
        run_time=1e-12,
        structures=[
            Structure(
                geometry=Box(size=(1, 1, 1), center=(-1, 0, 0)),
                medium=Medium(permittivity=2.0),
            ),
            Structure(
                geometry=Box(size=(1, 1, 1), center=(0, 0, 0)),
                medium=Medium(permittivity=2.0),
            ),
            Structure(geometry=Sphere(radius=1.4, center=(1.0, 0.0, 1.0)), medium=Medium()),
            Structure(
                geometry=Cylinder(radius=1.4, length=2.0, center=(1.0, 0.0, -1.0), axis=1),
                medium=Medium(),
            ),
        ],
        sources=[
            UniformCurrentSource(
                size=(0, 0, 0),
                center=(0, -0.5, 0),
                polarization="Hx",
                source_time=GaussianPulse(freq0=1e14, fwidth=1e12),
            ),
            UniformCurrentSource(
                size=(0, 0, 0),
                center=(0, -0.5, 0),
                polarization="Ex",
                source_time=GaussianPulse(freq0=1e14, fwidth=1e12),
            ),
            UniformCurrentSource(
                size=(0, 0, 0),
                center=(0, -0.5, 0),
                polarization="Ey",
                source_time=GaussianPulse(freq0=1e14, fwidth=1e12),
            ),
        ],
        monitors=[
            FluxMonitor(size=(1, 1, 0), center=(0, -0.5, 0), freqs=[1], name="mon1"),
            FluxMonitor(size=(0, 1, 1), center=(0, -0.5, 0), freqs=[1], name="mon2"),
            FluxMonitor(size=(1, 0, 1), center=(0, -0.5, 0), freqs=[1], name="mon3"),
        ],
    )

    for i, structure in enumerate(sim.structures):
        assert structure.name == f"structures[{i}]"

    for i, source in enumerate(sim.sources):
        assert source.name == f"sources[{i}]"


def test_names_unique():

    with pytest.raises(SetupError) as e:
        sim = Simulation(
            size=(2.0, 2.0, 2.0),
            run_time=1e-12,
            structures=[
                Structure(
                    geometry=Box(size=(1, 1, 1), center=(-1, 0, 0)),
                    medium=Medium(permittivity=2.0),
                    name="struct1",
                ),
                Structure(
                    geometry=Box(size=(1, 1, 1), center=(0, 0, 0)),
                    medium=Medium(permittivity=2.0),
                    name="struct1",
                ),
            ],
        )

    with pytest.raises(SetupError) as e:
        sim = Simulation(
            size=(2.0, 2.0, 2.0),
            run_time=1e-12,
            sources=[
                UniformCurrentSource(
                    size=(0, 0, 0),
                    center=(0, -0.5, 0),
                    polarization="Hx",
                    source_time=GaussianPulse(freq0=1e14, fwidth=1e12),
                    name="source1",
                ),
                UniformCurrentSource(
                    size=(0, 0, 0),
                    center=(0, -0.5, 0),
                    polarization="Ex",
                    source_time=GaussianPulse(freq0=1e14, fwidth=1e12),
                    name="source1",
                ),
            ],
        )

    with pytest.raises(SetupError) as e:
        sim = Simulation(
            size=(2.0, 2.0, 2.0),
            run_time=1e-12,
            monitors=[
                FluxMonitor(size=(1, 1, 0), center=(0, -0.5, 0), freqs=[1], name="mon1"),
                FluxMonitor(size=(0, 1, 1), center=(0, -0.5, 0), freqs=[1], name="mon1"),
            ],
        )


""" UniformCurrentSources """


def test_UniformCurrentSource():

    g = GaussianPulse(freq0=1, fwidth=0.1)

    # test we can make generic UniformCurrentSource
    s = UniformCurrentSource(size=(1, 1, 1), source_time=g, polarization="Ez")


def test_source_times():

    # test we can make gaussian pulse
    g = GaussianPulse(freq0=1, fwidth=0.1)
    ts = np.linspace(0, 30, 1001)
    g.amp_time(ts)
    # g.plot(ts)

    # test we can make cw pulse
    from tidy3d.components.source import ContinuousWave

    c = ContinuousWave(freq0=1, fwidth=0.1)
    ts = np.linspace(0, 30, 1001)
    c.amp_time(ts)


def test_dipole():

    g = GaussianPulse(freq0=1, fwidth=0.1)
    p = PointDipole(center=(1, 2, 3), source_time=g, polarization="Ex")
    # p.plot(y=2)

    with pytest.raises(pydantic.ValidationError) as e_info:
        p = PointDipole(size=(1, 1, 1), source_time=g, center=(1, 2, 3), polarization="Ex")


def test_FieldSource():
    g = GaussianPulse(freq0=1, fwidth=0.1)
    mode_spec = ModeSpec(num_modes=2)

    # test we can make planewave
    s = PlaneWave(size=(0, inf, inf), source_time=g, pol_angle=np.pi / 2, direction="+")
    # s.plot(y=0)

    # test we can make gaussian beam
    s = GaussianBeam(size=(0, 1, 1), source_time=g, pol_angle=np.pi / 2, direction="+")
    # s.plot(y=0)

    # test we can make an astigmatic gaussian beam
    s = AstigmaticGaussianBeam(
        size=(0, 1, 1),
        source_time=g,
        pol_angle=np.pi / 2,
        direction="+",
        waist_sizes=(0.2, 0.4),
        waist_distances=(0.1, 0.3),
    )

    # test we can make mode source
    s = ModeSource(size=(0, 1, 1), direction="+", source_time=g, mode_spec=mode_spec, mode_index=0)
    # s.plot(y=0)

    # test that non-planar geometry crashes plane wave and gaussian beams
    with pytest.raises(ValidationError) as e_info:
        s = PlaneWave(size=(1, 1, 1), source_time=g, pol_angle=np.pi / 2, direction="+")
    with pytest.raises(ValidationError) as e_info:
        s = GaussianBeam(size=(1, 1, 1), source_time=g, pol_angle=np.pi / 2, direction="+")
    with pytest.raises(ValidationError) as e_info:
        s = AstigmaticGaussianBeam(
            size=(1, 1, 1),
            source_time=g,
            pol_angle=np.pi / 2,
            direction="+",
            waist_sizes=(0.2, 0.4),
            waist_distances=(0.1, 0.3),
        )
    with pytest.raises(ValidationError) as e_info:
        s = ModeSource(size=(1, 1, 1), source_time=g, mode_spec=mode_spec)

    from tidy3d.components.source import TFSF

    s = TFSF(size=(1, 1, 1), direction="+", source_time=g, injection_axis=2)
    # s.plot(z=0)


def test_pol_arrow():

    g = GaussianPulse(freq0=1, fwidth=0.1)

    def get_pol_dir(axis, pol_angle=0, angle_theta=0, angle_phi=0):

        size = [inf, inf, inf]
        size[axis] = 0

        pw = PlaneWave(
            size=size,
            source_time=g,
            pol_angle=pol_angle,
            angle_theta=angle_theta,
            angle_phi=angle_phi,
            direction="+",
        )

        return pw._pol_vector

    assert np.allclose(get_pol_dir(axis=0), (0, 1, 0))
    assert np.allclose(get_pol_dir(axis=1), (1, 0, 0))
    assert np.allclose(get_pol_dir(axis=2), (1, 0, 0))
    assert np.allclose(get_pol_dir(axis=0, angle_phi=np.pi / 2), (0, 0, +1))
    assert np.allclose(get_pol_dir(axis=1, angle_phi=np.pi / 2), (0, 0, -1))
    assert np.allclose(get_pol_dir(axis=2, angle_phi=np.pi / 2), (0, +1, 0))
    assert np.allclose(get_pol_dir(axis=0, pol_angle=np.pi / 2), (0, 0, +1))
    assert np.allclose(get_pol_dir(axis=1, pol_angle=np.pi / 2), (0, 0, -1))
    assert np.allclose(get_pol_dir(axis=2, pol_angle=np.pi / 2), (0, +1, 0))
    assert np.allclose(
        get_pol_dir(axis=0, angle_theta=np.pi / 4), (+1 / np.sqrt(2), -1 / np.sqrt(2), 0)
    )
    assert np.allclose(
        get_pol_dir(axis=1, angle_theta=np.pi / 4), (-1 / np.sqrt(2), +1 / np.sqrt(2), 0)
    )
    assert np.allclose(
        get_pol_dir(axis=2, angle_theta=np.pi / 4), (-1 / np.sqrt(2), 0, +1 / np.sqrt(2))
    )


""" monitors """


def test_monitor():

    size = (1, 2, 3)
    center = (1, 2, 3)

    m1 = FieldMonitor(size=size, center=center, freqs=[1, 2, 3], name="test_monitor")
    m1s = FieldMonitor.surfaces(size=size, center=center, freqs=[1, 2, 3], name="test_monitor")
    m2 = FieldTimeMonitor(size=size, center=center, name="test_mon")
    m3 = FluxMonitor(size=(1, 1, 0), center=center, freqs=[1, 2, 3], name="test_mon")
    m4 = FluxTimeMonitor(size=(1, 1, 0), center=center, name="test_mon")
    m5 = ModeMonitor(
        size=(1, 1, 0), center=center, mode_spec=ModeSpec(), freqs=[1, 2, 3], name="test_mon"
    )
    m6 = ModeSolverMonitor(
        size=(1, 1, 0), center=center, mode_spec=ModeSpec(), freqs=[1, 2, 3], name="test_mon"
    )
    m7 = PermittivityMonitor(size=size, center=center, freqs=[1, 2, 3], name="perm")

    tmesh = np.linspace(0, 1, 10)

    for m in [m1, m2, m3, m4, m5, m6, m7]:
        # m.plot(y=2)
        m.storage_size(num_cells=100, tmesh=tmesh)

    for m in [m2, m4]:
        m.time_inds(tmesh=tmesh)
        m.num_steps(tmesh=tmesh)


def test_monitor_plane():

    freqs = [1, 2, 3]

    # make sure flux, mode and diffraction monitors fail with non planar geometries
    for size in ((0, 0, 0), (1, 0, 0), (1, 1, 1)):
        with pytest.raises(ValidationError) as e_info:
            ModeMonitor(size=size, freqs=freqs, modes=[])
        with pytest.raises(ValidationError) as e_info:
            ModeSolverMonitor(size=size, freqs=freqs, modes=[])
        with pytest.raises(ValidationError) as e_info:
            DiffractionMonitor(size=size, freqs=freqs, name="de")


def _test_freqs_nonempty():
    with pytest.raises(ValidationError) as e_info:
        FieldMonitor(size=(1, 1, 1), freqs=[])


def test_monitor_surfaces_from_volume():

    center = (1, 2, 3)

    # make sure that monitors with zero volume raise an error (adapted from test_monitor_plane())
    for size in ((0, 0, 0), (1, 0, 0), (1, 1, 0)):
        with pytest.raises(SetupError) as e_info:
            mon_surfaces = FieldMonitor.surfaces(
                size=size, center=center, freqs=[1, 2, 3], name="test_monitor"
            )

    # test that the surface monitors can be extracted from a volume monitor
    size = (1, 2, 3)
    monitor_surfaces = FieldMonitor.surfaces(
        size=size, center=center, freqs=[1, 2, 3], name="test_monitor"
    )

    # x- surface
    assert monitor_surfaces[0].center == (center[0] - size[0] / 2.0, center[1], center[2])
    assert monitor_surfaces[0].size == (0.0, size[1], size[2])

    # x+ surface
    assert monitor_surfaces[1].center == (center[0] + size[0] / 2.0, center[1], center[2])
    assert monitor_surfaces[1].size == (0.0, size[1], size[2])

    # y- surface
    assert monitor_surfaces[2].center == (center[0], center[1] - size[1] / 2.0, center[2])
    assert monitor_surfaces[2].size == (size[0], 0.0, size[2])

    # y+ surface
    assert monitor_surfaces[3].center == (center[0], center[1] + size[1] / 2.0, center[2])
    assert monitor_surfaces[3].size == (size[0], 0.0, size[2])

    # z- surface
    assert monitor_surfaces[4].center == (center[0], center[1], center[2] - size[2] / 2.0)
    assert monitor_surfaces[4].size == (size[0], size[1], 0.0)

    # z+ surface
    assert monitor_surfaces[5].center == (center[0], center[1], center[2] + size[2] / 2.0)
    assert monitor_surfaces[5].size == (size[0], size[1], 0.0)


""" modes """


def test_mode_object_syms():
    """Test that errors are raised if a mode object is not placed right in the presence of syms."""
    g = GaussianPulse(freq0=1, fwidth=0.1)

    # wrong mode source
    with pytest.raises(SetupError) as e_info:
        sim = Simulation(
            center=(1.0, -1.0, 0.5),
            size=(2.0, 2.0, 2.0),
            grid_spec=GridSpec.auto(wavelength=C_0 / 1.0),
            run_time=1e-12,
            symmetry=(1, -1, 0),
            sources=[ModeSource(size=(2, 2, 0), direction="+", source_time=g)],
        )

    # wrong mode monitor
    with pytest.raises(SetupError) as e_info:
        sim = Simulation(
            center=(1.0, -1.0, 0.5),
            size=(2.0, 2.0, 2.0),
            grid_spec=GridSpec.auto(wavelength=C_0 / 1.0),
            run_time=1e-12,
            symmetry=(1, -1, 0),
            monitors=[ModeMonitor(size=(2, 2, 0), name="mnt", freqs=[2], mode_spec=ModeSpec())],
        )

    # right mode source (centered on the symmetry)
    sim = Simulation(
        center=(1.0, -1.0, 0.5),
        size=(2.0, 2.0, 2.0),
        grid_spec=GridSpec.auto(wavelength=C_0 / 1.0),
        run_time=1e-12,
        symmetry=(1, -1, 0),
        sources=[ModeSource(center=(1, -1, 1), size=(2, 2, 0), direction="+", source_time=g)],
    )

    # right mode monitor (entirely in the main quadrant)
    sim = Simulation(
        center=(1.0, -1.0, 0.5),
        size=(2.0, 2.0, 2.0),
        grid_spec=GridSpec.auto(wavelength=C_0 / 1.0),
        run_time=1e-12,
        symmetry=(1, -1, 0),
        monitors=[
            ModeMonitor(
                center=(2, 0, 1), size=(2, 2, 0), name="mnt", freqs=[2], mode_spec=ModeSpec()
            )
        ],
    )


def test_deep_copy():
    """Make sure deep copying works as expceted with defaults."""
    b = Box(size=(1, 1, 1))
    m = Medium(permittivity=1)

    s = Structure(
        geometry=b,
        medium=m,
    )

    # s_shallow = s.copy(deep=False)
    # with shallow copy, these should be the same objects
    # assert id(s.geometry) == id(s_shallow.geometry)
    # assert id(s.medium) == id(s_shallow.medium)

    s_deep = s.copy(deep=True)

    # with deep copy, these should be different objects
    assert id(s.geometry) != id(s_deep.geometry)
    assert id(s.medium) != id(s_deep.medium)

    # default should be deep
    s_default = s.copy()
    assert id(s.geometry) != id(s_default.geometry)
    assert id(s.medium) != id(s_default.medium)

    # make sure other kwargs work, here we update the geometry to a sphere and shallow copy medium
    # s_kwargs = s.copy(deep=False, update=dict(geometry=Sphere(radius=1.0)))
    # assert id(s.medium) == id(s_kwargs.medium)
    # assert id(s.geometry) != id(s_kwargs.geometry)

    # behavior of modifying attributes
    s_default = s.copy(update=dict(geometry=Sphere(radius=1.0)))
    assert id(s.geometry) != id(s_default.geometry)

    # s_shallow = s.copy(deep=False, update=dict(geometry=Sphere(radius=1.0)))
    # assert id(s.geometry) != id(s_shallow.geometry)

    # behavior of modifying attributes of attributes
    new_geometry = s.geometry.copy(update=dict(size=(2, 2, 2)))
    s_default = s.copy(update=dict(geometry=new_geometry))
    assert id(s.geometry) != id(s_default.geometry)

    # s_shallow = s.copy(deep=False)
    # new_geometry = s.geometry.copy(update=dict(size=(2,2,2)))
    # s_shallow = s_shallow.copy(update=dict(geometry=new_geometry))
    # assert id(s.geometry) == id(s_shallow.geometry)
