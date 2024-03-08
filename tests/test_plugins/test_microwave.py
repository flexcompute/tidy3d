import pytest
import numpy as np

import tidy3d as td
from tidy3d import FieldData
from tidy3d.constants import ETA_0
from tidy3d.plugins.microwave import (
    VoltageIntegralAxisAligned,
    CurrentIntegralAxisAligned,
    ImpedanceCalculator,
)
import pydantic.v1 as pydantic
from tidy3d.exceptions import DataError
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
    length = 0.02
    size = [0, 0, 0]
    size[1] = length
    center = [0, 0, 0]
    voltage_integral = VoltageIntegralAxisAligned(
        center=center, size=size, sign="+", extrapolate_to_endpoints=True
    )

    _ = voltage_integral.compute_voltage(SIM_Z_DATA["field"])


def test_impedance_calculator():
    with pytest.raises(pydantic.ValidationError):
        _ = ImpedanceCalculator(voltage_integral=None, current_integral=None)


def test_impedance_calculator_on_time_data():
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
