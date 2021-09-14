import pytest
import numpy as np

import sys
sys.path.append('./')

from tidy3d_client import *

""" ==== Example simulation instance ==== """

def test_sim_full():
    sim = Simulation(
        size=(2.0, 2.0, 2.0),
        grid_size=(0.01, 0.01, 0.01),
        run_time=1e-12,
        structures={
            "square": Structure(
                geometry=Box(size=(1, 1, 1), center=(-1, 0, 0)),
                medium=Medium(permittivity=2.0),
            ),
            "box": Structure(
                geometry=Box(size=(1, 1, 1), center=(0, 0, 0)),
                medium=Medium(permittivity=1.0, conductivity=3.0),
            ),
            "sphere": Structure(
                geometry=Sphere(
                    radius=1.4,
                    center=(1.0, 0.0, 1.0)
                ),
                medium=Medium()
            ),
            "cylinder": Structure(
                geometry=Cylinder(
                    radius=1.4,
                    length=2.0,
                    center=(1.0, 0.0, -1.0),
                    axis=1
                ),
                medium=Medium()
            )        
        },
        sources={
            "dipole": Source(
                geometry=Box(size=(0, 0, 0), center=(0, -0.5, 0)),
                polarization=(1, 0, 1),
                source_time=GaussianPulse(
                    freq0=1e14,
                    fwidth=1e12,
                ),
            )
        },
        monitors={
            "point": Monitor(
                geometry=Box(size=(0, 0, 0), center=(0, 1, 0)),
            ),
            "plane": Monitor(
                geometry=Box(size=(1, 1, 0), center=(0, 0, 0)),
            ),
        },
        symmetry=(0, -1, 1),
        pml_layers=(
            PMLLayer(profile="absorber", num_layers=20),
            PMLLayer(profile="stable", num_layers=30),
            PMLLayer(profile="standard"),
        ),
        shutoff=1e-6,
        courant=0.8,
        subpixel=False,
    )

def test_negative_sizes():

    # negative in size kwargs errors
    for size in (-1, 1, 1), (1, -1, 1), (1, 1, -1):
        with pytest.raises(pydantic.ValidationError) as e_info:
            a = Box(size=size, center=(0, 0, 0))
        with pytest.raises(pydantic.ValidationError) as e_info:        
            s = Simulation(size=size, grid_size=1.0)
        with pytest.raises(pydantic.ValidationError) as e_info:
            s = Simulation(size=(1,1,1), grid_size=size)

    # negative grid sizes error?
    with pytest.raises(pydantic.ValidationError) as e_info:
        s = Simulation(size=(1,1,1), grid_size=-1.0)

def test_grid_size_def():

    size = (1,1,1)
    s = Simulation(size=size, grid_size=1.0)
    s = Simulation(size=size, grid_size=(1.0, 1.0, 1.0))
    s = Simulation(size=size, grid_size=((1.0, 2.0), 1.0, 1.0))
    s = Simulation(size=size, grid_size=(1.0, (1.0, 2.0), 1.0))
    s = Simulation(size=size, grid_size=(1.0, 1.0, (1.0, 2.0)))
    s = Simulation(size=size, grid_size=(1.0, (1.0, 2.0), (1.0, 2.0)))
    s = Simulation(size=size, grid_size=((1.0, 2.0), 1.0, (1.0, 2.0)))
    s = Simulation(size=size, grid_size=((1.0, 2.0), (1.0, 2.0), 1.0))
    s = Simulation(size=size, grid_size=((1.0, 2.0), (1.0, 2.0), (1.0, 2.0)))

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
    medium = nk_to_medium(n, k, freq)

    # test consistency
    eps_z = nk_to_eps_complex(n, k)
    eps, sig = nk_to_eps_sigma(n, k, freq)
    _eps_z = eps_sigma_to_eps_complex(eps, sig, freq)
    assert np.isclose(eps_z, _eps_z)

def test_dispersion_models():

    # construct media
    m_PR = PoleResidue(eps_inf=1.0, poles=[((1,2),(1,3)), ((2,4),(1,5))])
    m_SM = Sellmeier(coeffs=[(2,3), (2,4)])
    m_LZ = Lorentz(eps_inf=1.0, coeffs=[(1,3,2), (2,4,1)])
    m_DB = Debye(eps_inf=1.0, coeffs=[(1,3), (2,4)])

    freqs = np.linspace(0.01, 1, 1001)
    for medium in [m_PR, m_SM, m_LZ, m_DB]:
        eps_c = medium.eps_model(freqs)

def test_geometry():
    b = Box(size=(1,1,1), center=(0,0,0))
    s = Sphere(radius=1, center=(0,0,0))
    s = Cylinder(radius=1, center=(0,0,0), axis=1, length=1)
    s = PolySlab(vertices=((1, 2), (3, 4), (5, 4)), slab_bounds=(-1, 1), axis=1)

    # make sure wrong axis arguments error
    with pytest.raises(pydantic.ValidationError) as e_info:
        s = Cylinder(radius=1, center=(0,0,0), axis=-1, length=1)
    with pytest.raises(pydantic.ValidationError) as e_info:
        s = PolySlab(radius=1, center=(0,0,0), axis=-1, slab_bounds=1)
    with pytest.raises(pydantic.ValidationError) as e_info:
        s = Cylinder(radius=1, center=(0,0,0), axis=3, length=1)
    with pytest.raises(pydantic.ValidationError) as e_info:
        s = PolySlab(radius=1, center=(0,0,0), axis=3, slab_bounds=1)

    # make sure negative values error
    with pytest.raises(pydantic.ValidationError) as e_info:
        s = Sphere(radius=-1, center=(0,0,0))        
    with pytest.raises(pydantic.ValidationError) as e_info:
        s = Cylinder(radius=-1, center=(0,0,0), axis=3, length=1)
    with pytest.raises(pydantic.ValidationError) as e_info:
        s = Cylinder(radius=1, center=(0,0,0), axis=3, length=-1)

def test_bounds():

    # make sure all things are shifted to this central location
    CENTER_SHIFT = (-1., 1., 100.)

    def place_box(center_offset):

        shifted_center = tuple(c + s for (c, s) in zip(center_offset, CENTER_SHIFT))

        sim = Simulation(
            size=(1,1,1),
            center=CENTER_SHIFT,
            grid_size=(0.1, 0.1, 0.1),
            run_time=1e-12,
            structures={
                'box': Structure(
                    geometry=Box(
                        size=(1, 1, 1),
                        center=shifted_center
                    ),
                    medium=Medium()
                )
            }
        )

    # create all permutations of squares being shifted 1, -1, or zero in all three directions
    bin_strings = [list(format(i, '03b')) for i in range(8)]
    bin_ints = [[int(b) for b in bin_string] for bin_string in bin_strings]
    bin_ints = np.array(bin_ints)
    bin_signs = 2*(bin_ints - 0.5)

    # test all cases where box is shifted +/- 1 in x,y,z and still intersects
    for amp in bin_ints:
        for sign in bin_signs:
            center = amp*sign
            place_box(tuple(center))

    # test all cases where box is shifted +/- 2 in x,y,z and no longer intersects
    for amp in bin_ints:
        for sign in bin_signs:
            center = 2 * amp * sign
            if np.sum(center) < 1e-12:
                continue
            with pytest.raises(AssertionError) as e_info:
                place_box(tuple(center))

def test_source():

    # test we can make gaussian pulse
    g = GaussianPulse(freq0=1, fwidth=0.1)
    ts = np.linspace(0, 30, 1001)
    g.amp_time(ts)

    # test we can make generic source
    s = Source(geometry=Box(size=(1,1,1)), source_time=g, polarization=(1,1,1))

    # test we can make planewave
    s = PlaneWave(geometry=Box(size=(0,1,1)), source_time=g, polarization=(1,1,1), direction='+')

    # test that non-planar geometry crashes plane wave
    with pytest.raises(pydantic.ValidationError) as e_info:
        s = PlaneWave(geometry=Box(size=(1,1,1)), source_time=g, polarization=(1,1,1), direction='+')


if __name__ == '__main__':
    test_run()