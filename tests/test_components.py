import pytest
import numpy as np
import pydantic

from tidy3d import *
from tidy3d.log import ValidationError, SetupError


def test_sim():
    """make sure a simulation can be initialized"""

    sim = Simulation(
        size=(2.0, 2.0, 2.0),
        grid_size=(0.01, 0.01, 0.01),
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
            VolumeSource(
                size=(0, 0, 0),
                center=(0, -0.5, 0),
                polarization="Hx",
                source_time=GaussianPulse(
                    freq0=1e14,
                    fwidth=1e12,
                ),
                name="my_dipole",
            )
        ],
        monitors=[
            FieldMonitor(size=(0, 0, 0), center=(0, 0, 0), freqs=[1, 2], name="point"),
            FluxTimeMonitor(size=(1, 1, 0), center=(0, 0, 0), interval=10, name="plane"),
        ],
        symmetry=(0, -1, 1),
        pml_layers=(
            PML(num_layers=20),
            StablePML(num_layers=30),
            Absorber(num_layers=100),
        ),
        shutoff=1e-6,
        courant=0.8,
        subpixel=False,
    )


def _test_version():
    """ensure there's a version in simulation"""

    sim = Simulation(
        size=(1, 1, 1),
        grid_size=(0.1, 0.1, 0.1),
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
            grid_size=(0.1, 0.1, 0.1),
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


def test_sim_grid_size():

    size = (1, 1, 1)
    s = Simulation(size=size, grid_size=(1.0, 1.0, 1.0))


""" geometry """


def test_geometry():

    b = Box(size=(1, 1, 1), center=(0, 0, 0))
    s = Sphere(radius=1, center=(0, 0, 0))
    s = Cylinder(radius=1, center=(0, 0, 0), axis=1, length=1)
    s = PolySlab(vertices=((1, 2), (3, 4), (5, 4)), slab_bounds=(-1, 1), axis=1)
    vertices_np = np.array(s.vertices)
    s_np = PolySlab(vertices=vertices_np, slab_bounds=(-1, 1), axis=1)

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
            s = Simulation(size=size, grid_size=(1.0, 1.0, 1.0))
        with pytest.raises(pydantic.ValidationError) as e_info:
            s = Simulation(size=(1, 1, 1), grid_size=size)

    # negative grid sizes error?
    with pytest.raises(pydantic.ValidationError) as e_info:
        s = Simulation(size=(1, 1, 1), grid_size=-1.0)


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
    m_PR = PoleResidue(eps_inf=1.0, poles=[(1 + 2j, 1 + 3j), (2 + 4j, 1 + 5j)])
    m_SM = Sellmeier(coeffs=[(2, 3), (2, 4)])
    m_LZ = Lorentz(eps_inf=1.0, coeffs=[(1, 3, 2), (2, 4, 1)])
    m_DR = Drude(eps_inf=1.0, coeffs=[(1, 3), (2, 4)])
    m_DB = Debye(eps_inf=1.0, coeffs=[(1, 3), (2, 4)])

    freqs = np.linspace(0.01, 1, 1001)
    for medium in [m_PR, m_SM, m_LZ, m_DR, m_DB]:
        eps_c = medium.eps_model(freqs)


def test_medium_dispersion_conversion():

    m_PR = PoleResidue(eps_inf=1.0, poles=[((1 + 2j), (1 + 3j)), ((2 + 4j), (1 + 5j))])
    m_SM = Sellmeier(coeffs=[(2, 3), (2, 4)])
    m_LZ = Lorentz(eps_inf=1.0, coeffs=[(1, 3, 2), (2, 4, 1)])
    m_DR = Drude(eps_inf=1.0, coeffs=[(1, 3), (2, 4)])
    m_DB = Debye(eps_inf=1.0, coeffs=[(1, 3), (2, 4)])

    freqs = np.linspace(0.01, 1, 1001)
    for medium in [m_PR, m_SM, m_DB, m_LZ, m_DR]:  # , m_DB]:
        eps_model = medium.eps_model(freqs)
        eps_pr = medium.pole_residue.eps_model(freqs)
        np.testing.assert_allclose(eps_model, eps_pr)


def test_medium_dispersion_create():

    m_PR = PoleResidue(eps_inf=1.0, poles=[((1 + 2j), (1 + 3j)), ((2 + 4j), (1 + 5j))])
    m_SM = Sellmeier(coeffs=[(2, 3), (2, 4)])
    m_LZ = Lorentz(eps_inf=1.0, coeffs=[(1, 3, 2), (2, 4, 1)])
    m_DR = Drude(eps_inf=1.0, coeffs=[(1, 3), (2, 4)])
    m_DB = Debye(eps_inf=1.0, coeffs=[(1, 3), (2, 4)])

    for medium in [m_PR, m_SM, m_DB, m_LZ, m_DR]:
        struct = Structure(geometry=Box(size=(1, 1, 1)), medium=medium)


""" modes """


def test_modes():

    m = Mode(mode_index=0)
    m = Mode(mode_index=0, num_modes=1)

    # not enough modes
    with pytest.raises(SetupError) as e:
        m = Mode(mode_index=1, num_modes=1)


""" names """


def test_names_default():
    """makes sure default names are set"""

    sim = Simulation(
        size=(2.0, 2.0, 2.0),
        grid_size=(0.01, 0.01, 0.01),
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
            VolumeSource(
                size=(0, 0, 0),
                center=(0, -0.5, 0),
                polarization="Hx",
                source_time=GaussianPulse(freq0=1e14, fwidth=1e12),
            ),
            VolumeSource(
                size=(0, 0, 0),
                center=(0, -0.5, 0),
                polarization="Ex",
                source_time=GaussianPulse(freq0=1e14, fwidth=1e12),
            ),
            VolumeSource(
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

    # distinct_mediums = [f"mediums[{i}]" for i in range(len(sim.mediums))]
    # for i, medium in enumerate(sim.mediums):
    #     assert medium.name in distinct_mediums
    #     distinct_mediums.pop(distinct_mediums.index(medium.name))


def test_names_unique():

    with pytest.raises(SetupError) as e:
        sim = Simulation(
            size=(2.0, 2.0, 2.0),
            grid_size=(0.01, 0.01, 0.01),
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
            grid_size=(0.01, 0.01, 0.01),
            run_time=1e-12,
            sources=[
                VolumeSource(
                    size=(0, 0, 0),
                    center=(0, -0.5, 0),
                    polarization="Hx",
                    source_time=GaussianPulse(freq0=1e14, fwidth=1e12),
                    name="source1",
                ),
                VolumeSource(
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
            grid_size=(0.01, 0.01, 0.01),
            run_time=1e-12,
            monitors=[
                FluxMonitor(size=(1, 1, 0), center=(0, -0.5, 0), freqs=[1], name="mon1"),
                FluxMonitor(size=(0, 1, 1), center=(0, -0.5, 0), freqs=[1], name="mon1"),
            ],
        )


""" VolumeSources """


def test_VolumeSource():

    g = GaussianPulse(freq0=1, fwidth=0.1)

    # test we can make generic VolumeSource
    s = VolumeSource(size=(1, 1, 1), source_time=g, polarization="Ez")


def test_source_times():

    # test we can make gaussian pulse
    g = GaussianPulse(freq0=1, fwidth=0.1)
    ts = np.linspace(0, 30, 1001)
    g.amp_time(ts)

    # test we can make cq pulse
    # c = CW(freq0=1, fwidth=0.1)
    # ts = np.linspace(0, 30, 1001)
    # c.amp_time(ts)


def test_VolumeSource_directional():
    g = GaussianPulse(freq0=1, fwidth=0.1)

    # test we can make planewave
    s = PlaneWave(size=(0, 1, 1), source_time=g, polarization="Ez", direction="+")

    # test we can make planewave
    s = GaussianBeam(size=(0,1,1), source_time=g, polarization='Ez', direction='+')

    # test that non-planar geometry crashes plane wave
    with pytest.raises(ValidationError) as e_info:
        s = PlaneWave(size=(1, 1, 1), source_time=g, polarization="Ez", direction="+")

    # test that non-planar geometry crashes plane wave and gaussian beam
    with pytest.raises(ValidationError) as e_info:
        s = PlaneWave(size=(1, 1, 0), source_time=g, polarization="Ez", direction="+")
    with pytest.raises(ValidationError) as e_info:
        s = GaussianBeam(size=(1,1,1), source_time=g, polarization='Ez', direction='+')


def test_VolumeSource_modal():
    g = GaussianPulse(freq0=1, fwidth=0.1)
    mode = Mode(mode_index=0)
    m = ModeSource(size=(0, 1, 1), direction="+", source_time=g, mode=mode)


""" monitors """


def test_monitor():

    size = (1, 2, 3)
    center = (1, 2, 3)

    m = FieldMonitor(size=size, center=center, freqs=[1, 2, 3], name="test_monitor")


def test_monitor_plane():

    freqs = [1, 2, 3]

    # make sure flux and mode monitors fail with non planar geometries
    for size in ((0, 0, 0), (1, 0, 0), (1, 1, 1)):
        with pytest.raises(ValidationError) as e_info:
            ModeMonitor(size=size, freqs=freqs, modes=[])
        with pytest.raises(ValidationError) as e_info:
            FluxMonitor(size=size, freqs=freqs, modes=[])


def test_freqs_nonempty():

    with pytest.raises(ValidationError) as e_info:
        FieldMonitor(size=(1, 1, 1), freqs=[])
