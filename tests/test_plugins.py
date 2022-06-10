import pytest
import numpy as np
import pydantic

import gdspy
import tidy3d as td

from tidy3d.plugins import DispersionFitter
from tidy3d.plugins.webplots import SimulationPlotly, SimulationDataApp
from tidy3d.plugins import ModeSolver
from tidy3d.plugins import Near2Far
from tidy3d import FieldData, ScalarFieldData, FieldMonitor
from tidy3d.plugins.smatrix.smatrix import Port
from tidy3d.plugins.smatrix.smatrix import ComponentModeler
from .utils import clear_tmp


def test_near2far():
    """make sure Near2Far runs"""

    center = (0, 0, 0)
    size = (2, 2, 2)
    f0 = 1
    monitors = FieldMonitor(size=size, center=center, freqs=[f0], name="near_field").surfaces()

    sim_size = (5, 5, 5)
    sim = td.Simulation(
        size=sim_size,
        grid_spec=td.GridSpec.auto(wavelength=td.C_0 / f0),
        monitors=monitors,
        run_time=1e-12,
    )

    def rand_data():
        return ScalarFieldData(
            x=np.linspace(-1, 1, 10),
            y=np.linspace(-1, 1, 10),
            z=np.linspace(-1, 1, 10),
            f=[f0],
            values=np.random.random((10, 10, 10, 1)),
        )

    fields = ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]
    data_dict = {field: rand_data() for field in fields}
    field_data = FieldData(data_dict=data_dict)

    data_dict_mon = {mon.name: field_data for mon in monitors}
    sim_data = td.SimulationData(simulation=sim, monitor_data=data_dict_mon)

    n2f = Near2Far.from_surface_monitors(
        sim_data=sim_data,
        monitors=monitors,
        normal_dirs=["-", "+", "-", "+", "-", "+"],
        frequency=f0,
    )

    # single inputs
    n2f.radar_cross_section(1, 1)
    n2f.power_spherical(1, 1, 1)
    n2f.power_cartesian(1, 1, 1)
    n2f.fields_spherical(1, 1, 1)
    n2f.fields_cartesian(1, 1, 1)

    # vectorized inputs
    pts1 = [0, 1]
    pts2 = [0, 1, 2]
    pts3 = [3, 4, 5]
    n2f.radar_cross_section(pts1, pts2)
    n2f.power_spherical(1, pts2, pts3)
    n2f.power_cartesian(pts1, pts2, pts3)
    n2f.fields_spherical(1, pts2, pts3)
    n2f.fields_cartesian(pts1, pts2, pts3)


def test_mode_solver():
    """make sure mode solver runs"""
    waveguide = td.Structure(
        geometry=td.Box(size=(100, 0.5, 0.5)), medium=td.Medium(permittivity=4.0)
    )
    simulation = td.Simulation(
        size=(2, 2, 2),
        grid_spec=td.GridSpec(wavelength=1.0),
        structures=[waveguide],
        run_time=1e-12,
    )
    plane = td.Box(center=(0, 0, 0), size=(0, 1, 1))
    mode_spec = td.ModeSpec(
        num_modes=3,
        target_neff=2.0,
        bend_radius=3.0,
        bend_axis=0,
        num_pml=(10, 10),
    )
    ms = ModeSolver(
        simulation=simulation, plane=plane, mode_spec=mode_spec, freqs=[td.constants.C_0 / 1.0]
    )
    modes = ms.solve()


def _test_coeffs():
    """make sure pack_coeffs and unpack_coeffs are reciprocal"""
    num_poles = 10
    coeffs = np.random.random(4 * num_poles)
    a, c = _unpack_coeffs(coeffs)
    coeffs_ = _pack_coeffs(a, c)
    a_, c_ = _unpack_coeffs(coeffs_)
    assert np.allclose(coeffs, coeffs_)
    assert np.allclose(a, a_)
    assert np.allclose(c, c_)


def _test_pole_coeffs():
    """make sure coeffs_to_poles and poles_to_coeffs are reciprocal"""
    num_poles = 10
    coeffs = np.random.random(4 * num_poles)
    poles = _coeffs_to_poles(coeffs)
    coeffs_ = _poles_to_coeffs(poles)
    poles_ = _coeffs_to_poles(coeffs_)
    assert np.allclose(coeffs, coeffs_)


@clear_tmp
def test_dispersion():
    """performs a fit on some random data"""
    num_data = 10
    n_data = np.random.random(num_data)
    wvls = np.linspace(1, 2, num_data)
    fitter = DispersionFitter(wvl_um=wvls, n_data=n_data)
    medium, rms = fitter._fit_single()
    medium, rms = fitter.fit(num_tries=2)
    medium.to_file("tests/tmp/medium_fit.json")

    k_data = np.random.random(num_data)
    fitter = DispersionFitter(wvl_um=wvls, n_data=n_data, k_data=k_data)


def test_dispersion_load():
    """loads dispersion model from nk data file"""
    fitter = DispersionFitter.from_file("tests/data/nk_data.csv", skiprows=1, delimiter=",")
    medium, rms = fitter.fit(num_tries=20)


def test_dispersion_plot():
    """plots a medium fit from file"""
    fitter = DispersionFitter.from_file("tests/data/nk_data.csv", skiprows=1, delimiter=",")
    medium, rms = fitter.fit(num_tries=20)
    fitter.plot(medium)


def test_dispersion_set_wvg_range():
    """set wavelength range function"""
    num_data = 50
    n_data = np.random.random(num_data)
    wvls = np.linspace(1, 2, num_data)
    fitter = DispersionFitter(wvl_um=wvls, n_data=n_data)

    wvl_min = np.random.random(1)[0] * 0.5 + 1
    wvl_max = wvl_min + 0.5
    fitter = fitter.copy(update=dict(wvl_range=[wvl_min, wvl_max]))
    assert len(fitter.freqs) < num_data
    medium, rms = fitter.fit(num_tries=2)


def test_plotly():
    """Tests plotly plotting."""
    s = td.Simulation(size=(1, 1, 1), grid_spec=td.GridSpec.auto(wavelength=1), run_time=1e-12)
    sp = SimulationPlotly(simulation=s)
    fig = sp.plotly(x=0)


def test_app():
    """Tests plotly app."""
    center = (0, 0, 0)
    size = (2, 2, 2)
    f0 = 1
    monitors = FieldMonitor(size=size, center=center, freqs=[f0], name="near_field").surfaces()

    sim_size = (5, 5, 5)
    sim = td.Simulation(
        size=sim_size,
        grid_spec=td.GridSpec.auto(wavelength=td.C_0 / f0),
        monitors=monitors,
        run_time=1e-12,
    )

    def rand_data():
        return ScalarFieldData(
            x=np.linspace(-1, 1, 10),
            y=np.linspace(-1, 1, 10),
            z=np.linspace(-1, 1, 10),
            f=[f0],
            values=np.random.random((10, 10, 10, 1)),
        )

    fields = ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]
    data_dict = {field: rand_data() for field in fields}
    field_data = FieldData(data_dict=data_dict)

    data_dict_mon = {mon.name: field_data for mon in monitors}
    sim_data = td.SimulationData(simulation=sim, monitor_data=data_dict_mon)

    app = SimulationDataApp(sim_data=sim_data)
    _app = app.app


def test_component_modeler():
    """Tests S matrix loading."""
    # wavelength / frequency
    lambda0 = 1.550  # all length scales in microns
    freq0 = td.constants.C_0 / lambda0
    fwidth = freq0 / 10

    # Spatial grid specification
    grid_spec = td.GridSpec.auto(min_steps_per_wvl=14, wavelength=lambda0)

    # Permittivity of waveguide and substrate
    wg_n = 3.48
    sub_n = 1.45
    mat_wg = td.Medium(permittivity=wg_n**2)
    mat_sub = td.Medium(permittivity=sub_n**2)

    # Waveguide dimensions

    # Waveguide height
    wg_height = 0.22
    # Waveguide width
    wg_width = 1.0
    # Waveguide separation in the beginning/end
    wg_spacing_in = 8
    # length of coupling region (um)
    coup_length = 6.0
    # spacing between waveguides in coupling region (um)
    wg_spacing_coup = 0.05
    # Total device length along propagation direction
    device_length = 100
    # Length of the bend region
    bend_length = 16
    # Straight waveguide sections on each side
    straight_wg_length = 4
    # space between waveguide and PML
    pml_spacing = 2

    def bend_pts(bend_length, width, npts=10):
        """Set of points describing a tanh bend from (0, 0) to (length, width)"""
        x = np.linspace(0, bend_length, npts)
        y = width * (1 + np.tanh(6 * (x / bend_length - 0.5))) / 2
        return np.stack((x, y), axis=1)

    def arm_pts(length, width, coup_length, bend_length, npts_bend=30):
        """Set of points defining one arm of an integrated coupler"""
        ### Make the right half of the coupler arm first
        # Make bend and offset by coup_length/2
        bend = bend_pts(bend_length, width, npts_bend)
        bend[:, 0] += coup_length / 2
        # Add starting point as (0, 0)
        right_half = np.concatenate(([[0, 0]], bend))
        # Add an extra point to make sure waveguide is straight past the bend
        right_half = np.concatenate((right_half, [[right_half[-1, 0] + 0.1, width]]))
        # Add end point as (length/2, width)
        right_half = np.concatenate((right_half, [[length / 2, width]]))

        # Make the left half by reflecting and omitting the (0, 0) point
        left_half = np.copy(right_half)[1:, :]
        left_half[:, 0] = -left_half[::-1, 0]
        left_half[:, 1] = left_half[::-1, 1]

        return np.concatenate((left_half, right_half), axis=0)

    def make_coupler(
        length, wg_spacing_in, wg_width, wg_spacing_coup, coup_length, bend_length, npts_bend=30
    ):
        """Make an integrated coupler using the gdspy FlexPath object."""

        # Compute one arm of the coupler
        arm_width = (wg_spacing_in - wg_width - wg_spacing_coup) / 2
        arm = arm_pts(length, arm_width, coup_length, bend_length, npts_bend)
        # Reflect and offset bottom arm
        coup_bot = np.copy(arm)
        coup_bot[:, 1] = -coup_bot[::-1, 1] - wg_width / 2 - wg_spacing_coup / 2
        # Offset top arm
        coup_top = np.copy(arm)
        coup_top[:, 1] += wg_width / 2 + wg_spacing_coup / 2

        # Create waveguides as GDS paths
        path_bot = gdspy.FlexPath(coup_bot, wg_width, layer=1, datatype=0)
        path_top = gdspy.FlexPath(coup_top, wg_width, layer=1, datatype=1)

        return [path_bot, path_top]

    gdspy.current_library = gdspy.GdsLibrary()
    lib = gdspy.GdsLibrary()

    # Geometry must be placed in GDS cells to import into Tidy3D
    coup_cell = lib.new_cell("Coupler")

    substrate = gdspy.Rectangle(
        (-device_length / 2, -wg_spacing_in / 2 - 10),
        (device_length / 2, wg_spacing_in / 2 + 10),
        layer=0,
    )
    coup_cell.add(substrate)

    # Add the coupler to a gdspy cell
    gds_coup = make_coupler(
        device_length, wg_spacing_in, wg_width, wg_spacing_coup, coup_length, bend_length
    )
    coup_cell.add(gds_coup)

    # Substrate
    [oxide_geo] = td.PolySlab.from_gds(
        gds_cell=coup_cell, gds_layer=0, gds_dtype=0, slab_bounds=(-10, 0), axis=2
    )

    oxide = td.Structure(geometry=oxide_geo, medium=mat_sub)

    # Waveguides (import all datatypes if gds_dtype not specified)
    coupler1_geo, coupler2_geo = td.PolySlab.from_gds(
        gds_cell=coup_cell, gds_layer=1, slab_bounds=(0, wg_height), axis=2
    )

    coupler1 = td.Structure(geometry=coupler1_geo, medium=mat_wg)

    coupler2 = td.Structure(geometry=coupler2_geo, medium=mat_wg)

    # Simulation size along propagation direction
    sim_length = 2 * straight_wg_length + 2 * bend_length + coup_length

    # Spacing between waveguides and PML
    sim_size = [sim_length, wg_spacing_in + wg_width + 2 * pml_spacing, wg_height + 2 * pml_spacing]

    # source
    src_pos = sim_length / 2 - straight_wg_length / 2

    # in-plane field monitor (optional, increases required data storage)
    domain_monitor = td.FieldMonitor(
        center=[0, 0, wg_height / 2], size=[td.inf, td.inf, 0], freqs=[freq0], name="field"
    )

    # initialize the simulation
    sim = td.Simulation(
        size=sim_size,
        grid_spec=grid_spec,
        structures=[oxide, coupler1, coupler2],
        sources=[],
        monitors=[domain_monitor],
        run_time=20 / fwidth,
        boundary_spec=td.BoundarySpec.all_sides(boundary=td.PML()),
    )

    port_right_top = Port(
        center=[src_pos, wg_spacing_in / 2, wg_height / 2],
        size=[0, 4, 2],
        mode_spec=td.ModeSpec(num_modes=2),
        direction="-",
        name="right_top",
    )

    port_right_bot = Port(
        center=[src_pos, -wg_spacing_in / 2, wg_height / 2],
        size=[0, 4, 2],
        mode_spec=td.ModeSpec(num_modes=2),
        direction="-",
        name="right_bot",
    )

    port_left_top = Port(
        center=[-src_pos, wg_spacing_in / 2, wg_height / 2],
        size=[0, 4, 2],
        mode_spec=td.ModeSpec(num_modes=2),
        direction="+",
        name="left_top",
    )

    port_left_bot = Port(
        center=[-src_pos, -wg_spacing_in / 2, wg_height / 2],
        size=[0, 4, 2],
        mode_spec=td.ModeSpec(num_modes=2),
        direction="+",
        name="left_bot",
    )

    ports = [port_right_top, port_right_bot, port_left_top, port_left_bot]
    modeler = ComponentModeler(simulation=sim, ports=ports, freq=freq0)
