import math
import pytest
import numpy as np

from tidy3d import *
from tidy3d.log import SetupError
from .utils import assert_log_level

CENTER = (0, 0, 0)
SIZE = (2, 2, 2)
F0 = 1
FREQS = [F0, 1.1 * F0]
THETAS = [0, np.pi / 2]
PHIS = [np.pi / 6, np.pi / 4, np.pi / 3]

def test_near2far_monitor():
    """Make sure a simulation can be initialized with a near2far monitor."""

    vol_monitor = Near2FarMonitor(
        size=SIZE,
        center=CENTER,
        freqs=FREQS,
        name="near2far_vol",
        angles_theta=THETAS,
        angles_phi=PHIS,
        custom_origin=[0, 0, 1],
        exclude_surfaces=['z-'],
        medium=Medium(permittivity=2)
        )

    surf_monitor = Near2FarMonitor(
        size=(2, 2, 0),
        center=CENTER,
        freqs=FREQS,
        name="near2far_surf",
        angles_theta=THETAS,
        angles_phi=PHIS,
        medium=Medium(permittivity=1),
        normal_dir='+'
        )

    sim = Simulation(
        size=(5.0, 5.0, 5.0),
        run_time=1e-12,
        sources=[
            PointDipole(
                center=(0, 0, 0),
                polarization="Ex",
                source_time=GaussianPulse(
                    freq0=F0,
                    fwidth=F0 / 10,
                ),
            ),
        ],
        monitors=[vol_monitor, surf_monitor],
    )

    assert sim.monitors[0].local_origin == (0, 0, 1)
    assert sim.monitors[1].axis == 2


@pytest.mark.parametrize("normal_dir,log_level", [(None, None), ('+', 30)])
def test_box_normal_dir(caplog, normal_dir, log_level):
    """Make sure a warning is issued when ``normal_dir`` is specified for a volume monitor."""
    vol_monitor = Near2FarMonitor(
        size=SIZE,
        center=CENTER,
        freqs=FREQS,
        name="near2far_vol",
        angles_theta=THETAS,
        angles_phi=PHIS,
        normal_dir=normal_dir
        )
    assert_log_level(caplog, log_level)

def test_surf_normal_dir():
    """Make sure an error is issued when ``normal_dir`` is not given for a surface monitor."""
    with pytest.raises(SetupError) as e_info:
        surf_monitor = Near2FarMonitor(
            size=(2, 2, 0),
            center=CENTER,
            freqs=FREQS,
            name="near2far_surf",
            angles_theta=THETAS,
            angles_phi=PHIS,
            )

def test_surf_excluded_surfaces():
    """Make sure an error is issued when ``excluded_surfaces`` is given for a surface monitor."""
    with pytest.raises(SetupError) as e_info:
        surf_monitor = Near2FarMonitor(
            size=(2, 2, 0),
            center=CENTER,
            freqs=FREQS,
            name="near2far_surf",
            angles_theta=THETAS,
            angles_phi=PHIS,
            exclude_surfaces=['x+']
            )

def test_coord_conversions():
    """Test the conversion between spherical and cartesian coordinates."""
    x = np.linspace(0, 10, 11)
    y = np.linspace(0, 6, 7)
    z = np.linspace(1, 3, 4)

    for _x in x:
        for _y in y:
            for _z in z:
                r, theta, phi = Near2FarMonitor.car_2_sph(_x, _y, _z)
                _x2, _y2, _z2 = Near2FarMonitor.sph_2_car(r, theta, phi)
                print(_x, _x2, _y, _y2, _z, _z2)
                assert math.isclose(_x, _x2, rel_tol=1e-9, abs_tol=1e-15) and \
                       math.isclose(_y, _y2, rel_tol=1e-9, abs_tol=1e-15) and \
                       math.isclose(_z, _z2, rel_tol=1e-9, abs_tol=1e-15)

def test_near2far_data():
    """Test the radiation vector data structure."""
    values = (1+1j) * np.random.random((len(THETAS), len(PHIS), len(FREQS)))
    field = RadiationVector(values=values, theta=THETAS, phi=PHIS, f=FREQS)
    rad_vecs = Near2FarData(
        data_dict={'Ntheta': field, 'Nphi': field, 'Ltheta': field, 'Lphi': field})

    rad_vecs.fields_spherical()
    rad_vecs.fields_spherical(r=2)
    rad_vecs.radar_cross_section()
    rad_vecs.power_spherical(r=5)

    # single inputs
    # rad_vecs.power_cartesian(x=1, y=1, z=1)
    rad_vecs.fields_cartesian(x=1, y=1, z=1)

    # vectorized inputs
    pts1 = [0, 1]
    pts2 = [0, 1, 2]
    pts3 = [3, 4, 5]
    # rad_vecs.power_cartesian(pts1, pts2, pts3)
    rad_vecs.fields_cartesian(pts1, pts2, pts3)


def test_near2far_clientside():
    """Test the client-side radiation vector computations."""

    monitors = FieldMonitor.surfaces(size=SIZE, center=CENTER, freqs=FREQS, name="near_field")

    sim_size = (5, 5, 5)
    sim = Simulation(
        size=sim_size,
        grid_spec=GridSpec.auto(wavelength=C_0 / F0),
        monitors=monitors,
        run_time=1e-12,
    )

    def rand_data():
        N = 10
        return ScalarFieldData(
            x=np.linspace(-1, 1, N),
            y=np.linspace(-1, 1, N),
            z=np.linspace(-1, 1, N),
            f=FREQS,
            values=np.random.random((N, N, N, len(FREQS))),
        )

    fields = ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]
    data_dict = {field: rand_data() for field in fields}
    field_data = FieldData(data_dict=data_dict)

    data_dict_mon = {mon.name: field_data for mon in monitors}
    sim_data = SimulationData(simulation=sim, monitor_data=data_dict_mon)

    n2f = Near2Far.from_near_field_monitors(
        sim_data=sim_data,
        monitors=monitors,
        normal_dirs=["-", "+", "-", "+", "-", "+"],
    )

    rad_vecs = n2f.radiation_vectors(theta=THETAS, phi=PHIS)

    rad_vecs.fields_spherical()
    rad_vecs.fields_spherical(r=2)
    rad_vecs.radar_cross_section()
    rad_vecs.power_spherical(r=5)

    # single inputs
    # rad_vecs.power_cartesian(x=1, y=1, z=1)
    rad_vecs.fields_cartesian(x=1, y=1, z=1)

    # vectorized inputs
    pts1 = [0, 1]
    pts2 = [0, 1, 2]
    pts3 = [3, 4, 5]
    # rad_vecs.power_cartesian(pts1, pts2, pts3)
    rad_vecs.fields_cartesian(pts1, pts2, pts3)

