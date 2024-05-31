"""Test the microwave plugin."""

import numpy as np
import pydantic.v1 as pydantic
import pytest
import tidy3d as td
from skrf import Frequency
from skrf.media import MLine
from tidy3d import FieldData
from tidy3d.constants import ETA_0
from tidy3d.exceptions import DataError
from tidy3d.plugins.microwave import (
    CurrentIntegralAxisAligned,
    CustomCurrentIntegral2D,
    CustomVoltageIntegral2D,
    ImpedanceCalculator,
    VoltageIntegralAxisAligned,
)
from tidy3d.plugins.microwave.models import coupled_microstrip, microstrip

from ..utils import run_emulated

# Using similar code as "test_data/test_data_arrays.py"
MON_SIZE = (2, 1, 0)
FIELDS = ("Ex", "Ey", "Hx", "Hy")
FSTART = 0.5e9
FSTOP = 1.5e9
F0 = (FSTART + FSTOP) / 2
FWIDTH = FSTOP - FSTART
FS = np.linspace(FSTART, FSTOP, 3)
FIELD_MONITOR = td.FieldMonitor(
    size=MON_SIZE, fields=FIELDS, name="strip_field", freqs=FS, colocate=False
)
STRIP_WIDTH = 1.5
STRIP_HEIGHT = 0.5

SIM_Z = td.Simulation(
    size=(2, 1, 1),
    grid_spec=td.GridSpec.uniform(dl=0.04),
    monitors=[
        FIELD_MONITOR,
        td.FieldMonitor(center=(0, 0, 0), size=(1, 1, 1), freqs=FS, name="field"),
        td.FieldMonitor(
            center=(0, 0, 0), size=(1, 1, 1), freqs=FS, fields=["Ex", "Hx"], name="ExHx"
        ),
        td.FieldTimeMonitor(center=(0, 0, 0), size=(1, 1, 0), colocate=False, name="field_time"),
        td.ModeSolverMonitor(
            center=(0, 0, 0),
            size=(1, 1, 0),
            freqs=FS,
            mode_spec=td.ModeSpec(num_modes=2),
            name="mode",
        ),
    ],
    sources=[
        td.PointDipole(
            center=(0, 0, 0),
            polarization="Ex",
            source_time=td.GaussianPulse(freq0=F0, fwidth=FWIDTH),
        )
    ],
    run_time=5e-16,
)

SIM_Z_DATA = run_emulated(SIM_Z)

""" Generate the data arrays for testing path integral computations """


def get_xyz(
    monitor: td.components.monitor.MonitorType, grid_key: str
) -> tuple[list[float], list[float], list[float]]:
    grid = SIM_Z.discretize_monitor(monitor)
    x, y, z = grid[grid_key].to_list
    return x, y, z


def make_stripline_scalar_field_data_array(grid_key: str):
    """Populate FIELD_MONITOR with a idealized stripline mode, where fringing fields are assumed 0."""
    XS, YS, ZS = get_xyz(FIELD_MONITOR, grid_key)
    XGRID, YGRID = np.meshgrid(XS, YS, indexing="ij")
    XGRID = XGRID.reshape((len(XS), len(YS), 1, 1))
    YGRID = YGRID.reshape((len(XS), len(YS), 1, 1))
    values = np.zeros((len(XS), len(YS), len(ZS), len(FS)))
    ones = np.ones((len(XS), len(YS), len(ZS), len(FS)))
    XGRID = np.broadcast_to(XGRID, values.shape)
    YGRID = np.broadcast_to(YGRID, values.shape)

    # Numpy masks for quickly determining location
    above_in_strip = np.logical_and(YGRID >= 0, YGRID <= STRIP_HEIGHT / 2)
    below_in_strip = np.logical_and(YGRID < 0, YGRID >= -STRIP_HEIGHT / 2)
    within_strip_width = np.logical_and(XGRID >= -STRIP_WIDTH / 2, XGRID < STRIP_WIDTH / 2)
    above_and_within = np.logical_and(above_in_strip, within_strip_width)
    below_and_within = np.logical_and(below_in_strip, within_strip_width)
    # E field is perpendicular to strip surface and magnetic field is parallel
    if grid_key == "Ey":
        values = np.where(above_and_within, ones, values)
        values = np.where(below_and_within, -ones, values)
    elif grid_key == "Hx":
        values = np.where(above_and_within, -ones / ETA_0, values)
        values = np.where(below_and_within, ones / ETA_0, values)

    return td.ScalarFieldDataArray(values, coords=dict(x=XS, y=YS, z=ZS, f=FS))


def make_field_data():
    return FieldData(
        monitor=FIELD_MONITOR,
        Ex=make_stripline_scalar_field_data_array("Ex"),
        Ey=make_stripline_scalar_field_data_array("Ey"),
        Hx=make_stripline_scalar_field_data_array("Hx"),
        Hy=make_stripline_scalar_field_data_array("Hy"),
        symmetry=SIM_Z.symmetry,
        symmetry_center=SIM_Z.center,
        grid_expanded=SIM_Z.discretize_monitor(FIELD_MONITOR),
    )


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_voltage_integral_axes(axis):
    """Check VoltageIntegralAxisAligned runs."""
    length = 0.5
    size = [0, 0, 0]
    size[axis] = length
    center = [0, 0, 0]
    voltage_integral = VoltageIntegralAxisAligned(
        center=center,
        size=size,
        sign="+",
    )

    _ = voltage_integral.compute_voltage(SIM_Z_DATA["field"])


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_current_integral_axes(axis):
    """Check CurrentIntegralAxisAligned runs."""
    length = 0.5
    size = [length, length, length]
    size[axis] = 0.0
    center = [0, 0, 0]
    current_integral = CurrentIntegralAxisAligned(
        center=center,
        size=size,
        sign="+",
    )
    _ = current_integral.compute_current(SIM_Z_DATA["field"])


def test_voltage_integral_toggles():
    """Check VoltageIntegralAxisAligned runs with toggles."""
    length = 0.5
    size = [0, 0, 0]
    size[0] = length
    center = [0, 0, 0]
    voltage_integral = VoltageIntegralAxisAligned(
        center=center,
        size=size,
        extrapolate_to_endpoints=True,
        snap_path_to_grid=True,
        sign="-",
    )
    _ = voltage_integral.compute_voltage(SIM_Z_DATA["field"])


def test_current_integral_toggles():
    """Check CurrentIntegralAxisAligned runs with toggles."""
    length = 0.5
    size = [length, length, length]
    size[0] = 0.0
    center = [0, 0, 0]
    current_integral = CurrentIntegralAxisAligned(
        center=center,
        size=size,
        extrapolate_to_endpoints=True,
        snap_contour_to_grid=True,
        sign="-",
    )
    _ = current_integral.compute_current(SIM_Z_DATA["field"])


def test_voltage_missing_fields():
    """Check validation of VoltageIntegralAxisAligned with missing fields."""
    length = 0.5
    size = [0, 0, 0]
    size[1] = length
    center = [0, 0, 0]
    voltage_integral = VoltageIntegralAxisAligned(
        center=center,
        size=size,
        sign="+",
    )

    with pytest.raises(DataError):
        _ = voltage_integral.compute_voltage(SIM_Z_DATA["ExHx"])


def test_current_missing_fields():
    """Check validation of CurrentIntegralAxisAligned with missing fields."""
    length = 0.5
    size = [length, length, length]
    size[0] = 0.0
    center = [0, 0, 0]
    current_integral = CurrentIntegralAxisAligned(
        center=center,
        size=size,
        sign="+",
    )

    with pytest.raises(DataError):
        _ = current_integral.compute_current(SIM_Z_DATA["ExHx"])


def test_time_monitor_voltage_integral():
    """Check VoltageIntegralAxisAligned runs on time domain data."""
    length = 0.5
    size = [0, 0, 0]
    size[1] = length
    center = [0, 0, 0]
    voltage_integral = VoltageIntegralAxisAligned(
        center=center,
        size=size,
        sign="+",
    )

    voltage_integral.compute_voltage(SIM_Z_DATA["field_time"])


def test_mode_solver_monitor_voltage_integral():
    """Check VoltageIntegralAxisAligned runs on mode solver data."""
    length = 0.5
    size = [0, 0, 0]
    size[1] = length
    center = [0, 0, 0]
    voltage_integral = VoltageIntegralAxisAligned(
        center=center,
        size=size,
        sign="+",
    )

    voltage_integral.compute_voltage(SIM_Z_DATA["mode"])


def test_tiny_voltage_path():
    """Check VoltageIntegralAxisAligned runs when given a very short path."""
    length = 0.02
    size = [0, 0, 0]
    size[1] = length
    center = [0, 0, 0]
    voltage_integral = VoltageIntegralAxisAligned(
        center=center, size=size, sign="+", extrapolate_to_endpoints=True
    )

    _ = voltage_integral.compute_voltage(SIM_Z_DATA["field"])


def test_impedance_calculator():
    """Check validation of ImpedanceCalculator when integrals are missing."""
    with pytest.raises(pydantic.ValidationError):
        _ = ImpedanceCalculator(voltage_integral=None, current_integral=None)


def test_impedance_calculator_on_time_data():
    """Check ImpedanceCalculator runs on time domain data."""
    # Setup path integrals
    length = 0.5
    size = [0, length, 0]
    size[1] = length
    center = [0, 0, 0]
    voltage_integral = VoltageIntegralAxisAligned(
        center=center, size=size, sign="+", extrapolate_to_endpoints=True
    )

    size = [length, length, 0]
    current_integral = CurrentIntegralAxisAligned(center=center, size=size, sign="+")

    # Compute impedance using the tool
    Z_calc = ImpedanceCalculator(
        voltage_integral=voltage_integral, current_integral=current_integral
    )
    _ = Z_calc.compute_impedance(SIM_Z_DATA["field_time"])
    Z_calc = ImpedanceCalculator(voltage_integral=voltage_integral, current_integral=None)
    _ = Z_calc.compute_impedance(SIM_Z_DATA["field_time"])
    Z_calc = ImpedanceCalculator(voltage_integral=None, current_integral=current_integral)
    _ = Z_calc.compute_impedance(SIM_Z_DATA["field_time"])


def test_impedance_accuracy():
    """Test the accuracy of the ImpedanceCalculator."""
    field_data = make_field_data()
    # Setup path integrals
    size = [0, STRIP_HEIGHT / 2, 0]
    center = [0, -STRIP_HEIGHT / 4, 0]
    voltage_integral = VoltageIntegralAxisAligned(
        center=center, size=size, sign="+", extrapolate_to_endpoints=True
    )

    size = [STRIP_WIDTH * 1.25, STRIP_HEIGHT / 2, 0]
    center = [0, 0, 0]
    current_integral = CurrentIntegralAxisAligned(center=center, size=size, sign="+")

    def impedance_of_stripline(width, height):
        # Assuming no fringing fields, is the same as a parallel plate
        # with half the height and carrying twice the current
        Z0_parallel_plate = 0.5 * height / width * td.ETA_0
        return Z0_parallel_plate / 2

    analytic_impedance = impedance_of_stripline(STRIP_WIDTH, STRIP_HEIGHT)

    # Compute impedance using the tool
    Z_calc = ImpedanceCalculator(
        voltage_integral=voltage_integral, current_integral=current_integral
    )
    Z1 = Z_calc.compute_impedance(field_data)
    Z_calc = ImpedanceCalculator(voltage_integral=voltage_integral, current_integral=None)
    Z2 = Z_calc.compute_impedance(field_data)
    Z_calc = ImpedanceCalculator(voltage_integral=None, current_integral=current_integral)
    Z3 = Z_calc.compute_impedance(field_data)

    # Computation that uses the flux is less accurate, due to staircasing the field
    assert np.all(np.isclose(Z1, analytic_impedance, rtol=0.02))
    assert np.all(np.isclose(Z2, analytic_impedance, atol=3.5))
    assert np.all(np.isclose(Z3, analytic_impedance, atol=3.5))


def test_microstrip_models():
    """Test that the microstrip model computes transmission line parameters accurately."""
    width = 3.0
    height = 1.0
    thickness = 0.0
    eps_r = 4.4

    # Check zero thickness parameters
    Z0, eps_eff = microstrip.compute_line_params(eps_r, width, height, thickness)
    freqs = Frequency(start=1, stop=1, npoints=1, unit="ghz")
    mline = MLine(frequency=freqs, w=width, h=height, t=thickness, ep_r=eps_r, disp="none")

    assert np.isclose(Z0, mline.Z0[0])
    assert np.isclose(eps_eff, mline.ep_reff[0])

    # Check end effect length computation
    dL = microstrip.compute_end_effect_length(eps_r, eps_eff, width, height)
    assert np.isclose(dL, 0.54, rtol=0.01)

    # Check finite thickness parameters
    thickness = 0.1
    Z0, eps_eff = microstrip.compute_line_params(eps_r, width, height, thickness)
    mline = MLine(frequency=freqs, w=width, h=height, t=thickness, ep_r=eps_r, disp="none")

    assert np.isclose(Z0, mline.Z0[0])
    assert np.isclose(eps_eff, mline.ep_reff[0])


def test_coupled_microstrip_model():
    """Test that the coupled microstrip model computes transmission line parameters accurately."""
    w1 = 1.416
    w2 = 2.396
    height = 1.56
    g1 = 0.134
    g2 = 0.386
    eps_r = 4.3
    # Compare to:   Taoufik, Ragani, N. Amar Touhami, and M. Agoutane. "Designing a Microstrip coupled line bandpass filter."
    #               International Journal of Engineering & Technology 2, no. 4 (2013): 266.
    # and notebook "CoupledLineBandpassFilter"

    (Z_even, Z_odd, eps_even, eps_odd) = coupled_microstrip.compute_line_params(
        eps_r, w1, height, g1
    )
    assert np.isclose(Z_even, 101.5, rtol=0.01)
    assert np.isclose(Z_odd, 38.5, rtol=0.01)
    assert np.isclose(eps_even, 3.26, rtol=0.01)
    assert np.isclose(eps_odd, 2.71, rtol=0.01)

    (Z_even, Z_odd, eps_even, eps_odd) = coupled_microstrip.compute_line_params(
        eps_r, w2, height, g2
    )
    assert np.isclose(Z_even, 71, rtol=0.01)
    assert np.isclose(Z_odd, 39, rtol=0.01)
    assert np.isclose(eps_even, 3.42, rtol=0.01)
    assert np.isclose(eps_odd, 2.80, rtol=0.01)


def test_frequency_monitor_custom_voltage_integral():
    length = 0.5
    size = [0, 0, 0]
    size[1] = length
    # Make line
    vertices = [(0, 0), (0, 0.2), (0, 0.4)]
    voltage_integral = CustomVoltageIntegral2D(axis=2, position=0, vertices=vertices)
    voltage_integral.compute_voltage(SIM_Z_DATA["field"])


def test_vertices_validator_custom_current_integral():
    length = 0.5
    size = [0, 0, 0]
    size[1] = length
    # Make wrong box
    vertices = [(0.2, -0.2, 0.5), (0.2, 0.2), (-0.2, 0.2), (-0.2, -0.2), (0.2, -0.2)]

    with pytest.raises(pydantic.ValidationError):
        _ = CustomCurrentIntegral2D(axis=2, position=0, vertices=vertices)

    # Make wrong box shape
    vertices = [(0.2, 0.2, -0.2, -0.2, 0.2), (-0.2, 0.2, 0.2, -0.2, 0.2)]
    with pytest.raises(pydantic.ValidationError):
        _ = CustomCurrentIntegral2D(axis=2, position=0, vertices=vertices)


def test_fields_missing_custom_current_integral():
    length = 0.5
    size = [0, 0, 0]
    size[1] = length
    # Make box
    vertices = [(0.2, -0.2), (0.2, 0.2), (-0.2, 0.2), (-0.2, -0.2), (0.2, -0.2)]
    current_integral = CustomCurrentIntegral2D(axis=2, position=0, vertices=vertices)
    with pytest.raises(DataError):
        current_integral.compute_current(SIM_Z_DATA["ExHx"])


def test_time_monitor_custom_current_integral():
    length = 0.5
    size = [0, 0, 0]
    size[1] = length
    # Make box
    vertices = [(0.2, -0.2), (0.2, 0.2), (-0.2, 0.2), (-0.2, -0.2), (0.2, -0.2)]
    current_integral = CustomCurrentIntegral2D(axis=2, position=0, vertices=vertices)
    current_integral.compute_current(SIM_Z_DATA["field_time"])


def test_mode_solver_custom_current_integral():
    length = 0.5
    size = [0, 0, 0]
    size[1] = length
    # Make box
    vertices = [(0.2, -0.2), (0.2, 0.2), (-0.2, 0.2), (-0.2, -0.2), (0.2, -0.2)]
    current_integral = CustomCurrentIntegral2D(axis=2, position=0, vertices=vertices)
    current_integral.compute_current(SIM_Z_DATA["mode"])
