"""Tests microwave tools."""

from __future__ import annotations

from math import isclose
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pydantic as pd
import pytest
import xarray as xr
from shapely import LineString

import tidy3d as td
from tidy3d.components.data.data_array import FreqModeDataArray
from tidy3d.components.data.monitor_data import FreqDataArray
from tidy3d.components.microwave.formulas.circuit_parameters import (
    capacitance_colinear_cylindrical_wire_segments,
    capacitance_rectangular_sheets,
    inductance_straight_rectangular_wire,
    mutual_inductance_colinear_wire_segments,
    total_inductance_colinear_rectangular_wire_segments,
)
from tidy3d.components.microwave.path_integrals.factory import (
    make_current_integral,
    make_path_integrals,
    make_voltage_integral,
)
from tidy3d.components.microwave.path_integrals.mode_plane_analyzer import (
    ModePlaneAnalyzer,
)
from tidy3d.components.mode.mode_solver import ModeSolver
from tidy3d.constants import EPSILON_0
from tidy3d.exceptions import DataError, SetupError, ValidationError

from ..test_data.test_monitor_data import make_directivity_data
from ..utils import AssertLogLevel, get_spatial_coords_dict, run_emulated

MAKE_PLOTS = False
if MAKE_PLOTS:
    # Interactive plotting for debugging
    from matplotlib import use

    use("TkAgg")

# Constant parameters and simulation for path integral tests
mm = 1e3
# Using similar code as "test_data/test_data_arrays.py"
MON_SIZE = (2, 1, 0)
FIELDS = ("Ex", "Ey", "Hx", "Hy")
FSTART = 0.5e9
FSTOP = 1.5e9
F0 = (FSTART + FSTOP) / 2
FWIDTH = FSTOP - FSTART
FS = np.linspace(FSTART, FSTOP, 3)
FIELD_MONITOR = td.FieldMonitor(size=MON_SIZE, fields=FIELDS, name="strip_field", freqs=FS)
STRIP_WIDTH = 1.5
STRIP_HEIGHT = 0.5
COAX_R1 = 0.04
COAX_R2 = 0.5

SIM_Z = td.Simulation(
    size=(2, 1, 1),
    grid_spec=td.GridSpec.uniform(dl=0.04),
    monitors=[
        FIELD_MONITOR,
        td.FieldMonitor(center=(0, 0, 0), size=(1, 1, 1), freqs=FS, name="field", colocate=False),
        td.FieldMonitor(
            center=(0, 0, 0), size=(1, 1, 1), freqs=FS, fields=["Ex", "Hx"], name="ExHx"
        ),
        td.FieldTimeMonitor(center=(0, 0, 0), size=(1, 1, 0), colocate=False, name="field_time"),
        td.ModeSolverMonitor(
            center=(0, 0, 0),
            size=(1, 1, 0),
            freqs=FS,
            mode_spec=td.ModeSpec(num_modes=2),
            name="mode_solver",
        ),
        td.ModeMonitor(
            center=(0, 0, 0),
            size=(1, 1, 0),
            freqs=FS,
            mode_spec=td.ModeSpec(num_modes=3),
            store_fields_direction="+",
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


def make_stripline_scalar_field_data_array(grid_key: str):
    """Populate FIELD_MONITOR with a idealized stripline mode, where fringing fields are assumed 0."""
    XS, YS, ZS = get_spatial_coords_dict(SIM_Z, FIELD_MONITOR, grid_key).values()
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
        values = np.where(above_and_within, -ones / td.ETA_0, values)
        values = np.where(below_and_within, ones / td.ETA_0, values)

    return td.ScalarFieldDataArray(values, coords={"x": XS, "y": YS, "z": ZS, "f": FS})


def make_coaxial_field_data_array(grid_key: str):
    """Populate FIELD_MONITOR with a coaxial transmission line mode."""

    # Get a normalized electric field that represents the electric field within a coaxial cable transmission line.
    def compute_coax_radial_electric(rin, rout, x, y, is_x):
        # Radial distance
        r = np.sqrt((x) ** 2 + (y) ** 2)
        # Remove division by 0
        r_valid = np.where(r == 0.0, 1, r)
        # Compute current density so that the total current
        # is 1 flowing through surfaces of constant r
        # Extra r is for changing to Cartesian coordinates
        denominator = 2 * np.pi * r_valid**2
        if is_x:
            Exy = np.where(r <= rin, 0, (x / denominator))
            Exy = np.where(r >= rout, 0, Exy)
        else:
            Exy = np.where(r <= rin, 0, (y / denominator))
            Exy = np.where(r >= rout, 0, Exy)
        return Exy

    XS, YS, ZS = get_spatial_coords_dict(SIM_Z, FIELD_MONITOR, grid_key).values()
    XGRID, YGRID = np.meshgrid(XS, YS, indexing="ij")
    XGRID = XGRID.reshape((len(XS), len(YS), 1, 1))
    YGRID = YGRID.reshape((len(XS), len(YS), 1, 1))
    values = np.zeros((len(XS), len(YS), len(ZS), len(FS)))

    XGRID = np.broadcast_to(XGRID, values.shape)
    YGRID = np.broadcast_to(YGRID, values.shape)

    is_x = grid_key[1] == "x"
    if grid_key[0] == "E":
        field = compute_coax_radial_electric(COAX_R1, COAX_R2, XGRID, YGRID, is_x)
    else:
        field = compute_coax_radial_electric(COAX_R1, COAX_R2, XGRID, YGRID, not is_x)
        # H field is perpendicular and oriented for positive z propagation
        # We want to compute Hx which is -Ey/eta0, Hy which is Ex/eta0
        if is_x:
            field /= -td.ETA_0
        else:
            field /= td.ETA_0

    return td.ScalarFieldDataArray(field, coords={"x": XS, "y": YS, "z": ZS, "f": FS})


def make_field_data():
    return td.FieldData(
        monitor=FIELD_MONITOR,
        Ex=make_stripline_scalar_field_data_array("Ex"),
        Ey=make_stripline_scalar_field_data_array("Ey"),
        Hx=make_stripline_scalar_field_data_array("Hx"),
        Hy=make_stripline_scalar_field_data_array("Hy"),
        symmetry=SIM_Z.symmetry,
        symmetry_center=SIM_Z.center,
        grid_expanded=SIM_Z.discretize_monitor(FIELD_MONITOR),
    )


def make_coax_field_data():
    return td.FieldData(
        monitor=FIELD_MONITOR,
        Ex=make_coaxial_field_data_array("Ex"),
        Ey=make_coaxial_field_data_array("Ey"),
        Hx=make_coaxial_field_data_array("Hx"),
        Hy=make_coaxial_field_data_array("Hy"),
        symmetry=SIM_Z.symmetry,
        symmetry_center=SIM_Z.center,
        grid_expanded=SIM_Z.discretize_monitor(FIELD_MONITOR),
    )


def make_mw_sim(
    use_2D: bool = False,
    colocate: bool = False,
    transmission_line_type: Literal["microstrip", "cpw", "coax", "stripline"] = "microstrip",
    width=3 * mm,
    height=1 * mm,
    metal_thickness=0.2 * mm,
) -> td.Simulation:
    """Helper to create a microwave simulation with a single type of transmission line present."""

    freq_start = 1e9
    freq_stop = 10e9

    freq0 = (freq_start + freq_stop) / 2
    fwidth = freq_stop - freq_start
    freqs = np.arange(freq_start, freq_stop, 1e9)

    run_time = 60 / fwidth

    length = 40 * mm
    sim_width = length

    pec = td.PEC
    if use_2D:
        metal_thickness = 0.0
        pec = td.PEC2D

    epsr = 4.4
    diel = td.Medium(permittivity=epsr)

    metal_geos = []

    if transmission_line_type == "microstrip":
        substrate = td.Structure(
            geometry=td.Box(
                center=[0, 0, 0],
                size=[td.inf, td.inf, 2 * height],
            ),
            medium=diel,
        )
        metal_geos.append(
            td.Box(
                center=[0, 0, height + metal_thickness / 2],
                size=[td.inf, width, metal_thickness],
            )
        )
    elif transmission_line_type == "cpw":
        substrate = td.Structure(
            geometry=td.Box(
                center=[0, 0, 0],
                size=[td.inf, td.inf, 2 * height],
            ),
            medium=diel,
        )
        metal_geos.append(
            td.Box(
                center=[0, 0, height + metal_thickness / 2],
                size=[td.inf, width, metal_thickness],
            )
        )
        gnd_width = 10 * width
        gap = width / 5
        gnd_shift = gnd_width / 2 + gap + width / 2
        metal_geos.append(
            td.Box(
                center=[0, -gnd_shift, height + metal_thickness / 2],
                size=[td.inf, gnd_width, metal_thickness],
            )
        )
        metal_geos.append(
            td.Box(
                center=[0, gnd_shift, height + metal_thickness / 2],
                size=[td.inf, gnd_width, metal_thickness],
            )
        )
    elif transmission_line_type == "coax":
        substrate = td.Structure(
            geometry=td.Box(
                center=[0, 0, 0],
                size=[td.inf, td.inf, 2 * height],
            ),
            medium=diel,
        )
        metal_geos.append(
            td.GeometryGroup(
                geometries=(
                    td.ClipOperation(
                        operation="difference",
                        geometry_a=td.Cylinder(
                            axis=0, radius=2 * mm, center=(0, 0, 5 * mm), length=td.inf
                        ),
                        geometry_b=td.Cylinder(
                            axis=0, radius=1.8 * mm, center=(0, 0, 5 * mm), length=td.inf
                        ),
                    ),
                    td.Cylinder(axis=0, radius=0.6 * mm, center=(0, 0, 5 * mm), length=td.inf),
                )
            )
        )
    elif transmission_line_type == "stripline":
        substrate = td.Structure(
            geometry=td.Box(
                center=[0, 0, 0],
                size=[td.inf, td.inf, 2 * height + metal_thickness],
            ),
            medium=diel,
        )
        metal_geos.append(
            td.Box(
                center=[0, 0, 0],
                size=[td.inf, width, metal_thickness],
            )
        )
        gnd_width = 10 * width
        metal_geos.append(
            td.Box(
                center=[0, 0, height + metal_thickness],
                size=[td.inf, gnd_width, metal_thickness],
            )
        )
        metal_geos.append(
            td.Box(
                center=[0, 0, -height - metal_thickness],
                size=[td.inf, gnd_width, metal_thickness],
            )
        )
    else:
        raise AssertionError("Incorrect argument")

    metal_structures = [td.Structure(geometry=geo, medium=pec) for geo in metal_geos]
    structures = [substrate, *metal_structures]
    boundary_spec = td.BoundarySpec(
        x=td.Boundary(plus=td.PML(), minus=td.PML()),
        y=td.Boundary(plus=td.PML(), minus=td.PML()),
        z=td.Boundary(plus=td.PML(), minus=td.PECBoundary()),
    )

    size_sim = [
        length + 2 * width,
        sim_width,
        20 * mm + height + metal_thickness,
    ]
    center_sim = [0, 0, size_sim[2] / 2]
    # Slightly different setup for stripline substrate sandwiched between ground planes
    if transmission_line_type == "stripline":
        center_sim[2] = 0
        boundary_spec = td.BoundarySpec(
            x=td.Boundary(plus=td.PML(), minus=td.PML()),
            y=td.Boundary(plus=td.PML(), minus=td.PML()),
            z=td.Boundary(plus=td.PML(), minus=td.PML()),
        )
    size_port = [0, sim_width, size_sim[2]]
    center_port = [0, 0, center_sim[2]]
    impedance_specs = (td.AutoImpedanceSpec(),) * 4
    mode_spec = td.MicrowaveModeSpec(
        num_modes=4,
        target_neff=1.8,
        impedance_specs=impedance_specs,
    )

    mode_monitor = td.MicrowaveModeMonitor(
        center=center_port, size=size_port, freqs=freqs, name="mode_1", colocate=colocate
    )

    gaussian = td.GaussianPulse(freq0=freq0, fwidth=fwidth)
    mode_src = td.ModeSource(
        center=(-length / 2, 0, center_sim[2]),
        size=size_port,
        direction="+",
        mode_spec=mode_spec,
        mode_index=0,
        source_time=gaussian,
    )
    sim = td.Simulation(
        center=center_sim,
        size=size_sim,
        grid_spec=td.GridSpec.uniform(dl=0.1 * mm),
        structures=structures,
        sources=[mode_src],
        monitors=[mode_monitor],
        run_time=run_time,
        boundary_spec=boundary_spec,
        plot_length_units="mm",
        symmetry=(0, 0, 0),
    )
    return sim


def test_inductance_formulas():
    """Run the formulas for inductance and compare to precomputed results."""
    bar_size = (1000e4, 1e4, 1e4)  # case from reference
    L1 = inductance_straight_rectangular_wire(bar_size, 0)
    assert isclose(L1, 14.816e-6, rel_tol=1e-4)
    length = 1e3
    L2 = mutual_inductance_colinear_wire_segments(length, length, length / 10)
    assert isclose(L2, 0.11181e-9, rel_tol=1e-4)
    side = length / 10
    L3 = total_inductance_colinear_rectangular_wire_segments(
        (side, length, side), (side, length, side), length / 10, 1
    )
    assert isclose(L3, 1.3625e-9, rel_tol=1e-4)


def test_capacitance_formulas():
    """Run the formulas for capacitance and compare to precomputed results."""
    width = 3e3
    length = 1e3
    d = length / 4.5  # case from reference
    C1 = capacitance_rectangular_sheets(width, length, d)
    result = 2.347 * EPSILON_0 * width  # from reference
    assert isclose(C1, result, rel_tol=1e-3)

    # case from reference
    radius = 0.1e-3
    C2 = capacitance_colinear_cylindrical_wire_segments(radius, length, length / 5)
    D2 = 0.345
    C_ref = np.pi * EPSILON_0 * length / (np.log(length / radius) - 2.303 * D2)
    assert isclose(C2, C_ref, rel_tol=1e-3)

    # case from reference
    C3 = capacitance_colinear_cylindrical_wire_segments(radius, length, length * 5)
    D2 = 0.144
    C_ref = np.pi * EPSILON_0 * length / (np.log(length / radius) - 2.303 * D2)
    assert isclose(C3, C_ref, rel_tol=1e-2)


def test_antenna_parameters():
    """Test basic antenna parameters computation and validation."""

    # Create from random directivity data
    directivity_data = make_directivity_data()
    f = directivity_data.coords["f"]
    power_inc = FreqDataArray(0.8 * np.ones(len(f)), coords={"f": f})
    power_refl = 0.25 * power_inc
    antenna_params = td.AntennaMetricsData.from_directivity_data(
        directivity_data, power_inc, power_refl
    )

    # Test that all essential parameters exist and are correct type
    assert isinstance(antenna_params.radiation_efficiency, FreqDataArray)
    assert isinstance(antenna_params.reflection_efficiency, FreqDataArray)
    assert np.allclose(antenna_params.reflection_efficiency, 0.75)
    assert isinstance(antenna_params.gain, xr.DataArray)
    assert isinstance(antenna_params.realized_gain, xr.DataArray)

    # Test partial gain computations in linear basis
    partial_gain_linear = antenna_params.partial_gain(pol_basis="linear")
    assert isinstance(partial_gain_linear, xr.Dataset)
    assert "Gtheta" in partial_gain_linear
    assert "Gphi" in partial_gain_linear

    # Test partial gain computations in linear basis with tilt angle = 0 matches partial gain in the original basis
    partial_gain_linear_tilted = antenna_params.partial_gain(pol_basis="linear", tilt_angle=0)
    assert isinstance(partial_gain_linear_tilted, xr.Dataset)
    assert "Gco" in partial_gain_linear_tilted
    assert "Gcross" in partial_gain_linear_tilted
    assert np.allclose(partial_gain_linear_tilted.Gco, partial_gain_linear.Gtheta)
    assert np.allclose(partial_gain_linear_tilted.Gcross, partial_gain_linear.Gphi)

    # Test validation of tilt angle that only works with linear basis
    with pytest.raises(ValueError):
        antenna_params.partial_gain(pol_basis="circular", tilt_angle=1)

    # Test partial gain computations in circular basis
    partial_gain_circular = antenna_params.partial_gain(pol_basis="circular")
    assert isinstance(partial_gain_circular, xr.Dataset)
    assert "Gright" in partial_gain_circular
    assert "Gleft" in partial_gain_circular

    # Test partial realized gain computations in both bases
    assert isinstance(antenna_params.partial_realized_gain("linear"), xr.Dataset)
    assert isinstance(antenna_params.partial_realized_gain("circular"), xr.Dataset)

    # Test validation of pol_basis parameter
    with pytest.raises(ValueError):
        antenna_params.partial_gain("invalid")
    with pytest.raises(ValueError):
        antenna_params.partial_realized_gain("invalid")


def test_path_spec_plotting():
    """Test that all types of path specification correctly plot themselves."""

    mean_radius = (COAX_R2 + COAX_R1) * 0.5
    size = [COAX_R2 - COAX_R1, 0, 0]
    center = [mean_radius, 0, 0]

    voltage_integral = td.AxisAlignedVoltageIntegralSpec(
        center=center, size=size, sign="-", extrapolate_to_endpoints=True, snap_path_to_grid=True
    )

    current_integral = td.Custom2DCurrentIntegralSpec.from_circular_path(
        center=(0, 0, 0), radius=0.4, num_points=31, normal_axis=2, clockwise=False
    )

    ax = voltage_integral.plot(z=0)
    current_integral.plot(z=0, ax=ax)
    plt.close()

    # Test off center plotting
    ax = voltage_integral.plot(z=2)
    current_integral.plot(z=2, ax=ax)
    plt.close()

    # Plot
    voltage_integral = td.Custom2DVoltageIntegralSpec(
        axis=1, position=0, vertices=[(-1, -1), (0, 0), (1, 1)]
    )

    current_integral = td.AxisAlignedCurrentIntegralSpec(
        center=(0, 0, 0),
        size=(2, 0, 1),
        sign="-",
        extrapolate_to_endpoints=False,
        snap_contour_to_grid=False,
    )

    ax = voltage_integral.plot(y=0)
    current_integral.plot(y=0, ax=ax)
    plt.close()

    # Test off center plotting
    ax = voltage_integral.plot(y=2)
    current_integral.plot(y=2, ax=ax)
    plt.close()

    current_integral = td.CompositeCurrentIntegralSpec(
        path_specs=(
            td.AxisAlignedCurrentIntegralSpec(center=(-1, -1, 0), size=(1, 1, 0), sign="-"),
            td.AxisAlignedCurrentIntegralSpec(center=(1, 1, 0), size=(1, 1, 0), sign="-"),
        ),
        sum_spec="sum",
    )
    ax = current_integral.plot(z=0)
    plt.close()


@pytest.mark.parametrize("clockwise", [False, True])
def test_custom_current_specification_sign(clockwise):
    """Make sure the sign is correctly calculated for custom current specs."""
    current_integral = td.Custom2DCurrentIntegralSpec.from_circular_path(
        center=(0, 0, 0), radius=0.4, num_points=31, normal_axis=2, clockwise=clockwise
    )
    if clockwise:
        assert current_integral.sign == "-"
    else:
        assert current_integral.sign == "+"

    # When aligned with y, the sign has to be handled carefully
    current_integral = td.Custom2DCurrentIntegralSpec.from_circular_path(
        center=(0, 0, 0), radius=0.4, num_points=31, normal_axis=1, clockwise=clockwise
    )
    if clockwise:
        assert current_integral.sign == "-"
    else:
        assert current_integral.sign == "+"


def test_composite_current_integral_validation():
    """Ensures that the CompositeCurrentIntegralSpec is validated correctly."""

    current_spec = td.AxisAlignedCurrentIntegralSpec(center=(1, 2, 3), size=(0, 1, 1), sign="-")
    voltage_spec = td.AxisAlignedVoltageIntegralSpec(center=(1, 2, 3), size=(0, 0, 1), sign="-")
    path_spec = td.CompositeCurrentIntegralSpec(path_specs=[current_spec], sum_spec="sum")

    with pytest.raises(pd.ValidationError):
        path_spec.updated_copy(path_specs=[])

    with pytest.raises(pd.ValidationError):
        path_spec.updated_copy(path_specs=[voltage_spec])


def test_composite_current_integral_bounds():
    """Test that CompositeCurrentIntegralSpec correctly computes overall bounding box."""

    # Single axis-aligned spec
    spec1 = td.AxisAlignedCurrentIntegralSpec(center=(0, 0, 0), size=(2, 2, 0), sign="+")
    composite1 = td.CompositeCurrentIntegralSpec(path_specs=(spec1,), sum_spec="sum")
    expected_bounds1 = ((-1.0, -1.0, 0.0), (1.0, 1.0, 0.0))
    assert composite1.bounds == expected_bounds1

    # Two disjoint axis-aligned specs
    spec2 = td.AxisAlignedCurrentIntegralSpec(center=(5, 5, 0), size=(2, 2, 0), sign="+")
    composite2 = td.CompositeCurrentIntegralSpec(path_specs=(spec1, spec2), sum_spec="sum")
    # Should span from (-1, -1, 0) to (6, 6, 0)
    expected_bounds2 = ((-1.0, -1.0, 0.0), (6.0, 6.0, 0.0))
    assert composite2.bounds == expected_bounds2

    # Custom 2D spec
    vertices = np.array([[0, 0], [2, 0], [2, 1], [0, 1], [0, 0]])
    spec3 = td.Custom2DCurrentIntegralSpec(axis=2, position=0, vertices=vertices)
    composite3 = td.CompositeCurrentIntegralSpec(path_specs=(spec3,), sum_spec="sum")
    expected_bounds3 = ((0.0, 0.0, 0.0), (2.0, 1.0, 0.0))
    assert composite3.bounds == expected_bounds3

    # Mixed spec types (axis-aligned + custom 2D)
    spec4 = td.AxisAlignedCurrentIntegralSpec(center=(-3, -3, 0), size=(1, 1, 0), sign="-")
    composite4 = td.CompositeCurrentIntegralSpec(path_specs=(spec3, spec4), sum_spec="split")
    # Custom spec: (0, 0, 0) to (2, 1, 0)
    # Axis-aligned spec: (-3.5, -3.5, 0) to (-2.5, -2.5, 0)
    # Overall: (-3.5, -3.5, 0) to (2, 1, 0)
    expected_bounds4 = ((-3.5, -3.5, 0.0), (2.0, 1.0, 0.0))
    assert composite4.bounds == expected_bounds4

    # Overlapping specs
    spec5 = td.AxisAlignedCurrentIntegralSpec(center=(1, 1, 0), size=(4, 4, 0), sign="+")
    spec6 = td.AxisAlignedCurrentIntegralSpec(center=(2, 2, 0), size=(2, 2, 0), sign="+")
    composite5 = td.CompositeCurrentIntegralSpec(path_specs=(spec5, spec6), sum_spec="sum")
    # spec5: (-1, -1, 0) to (3, 3, 0)
    # spec6: (1, 1, 0) to (3, 3, 0)
    # Overall: (-1, -1, 0) to (3, 3, 0)
    expected_bounds5 = ((-1.0, -1.0, 0.0), (3.0, 3.0, 0.0))
    assert composite5.bounds == expected_bounds5


def test_path_integral_creation():
    """Check that path integrals are correctly constructed from path specifications."""

    path_spec = td.AxisAlignedVoltageIntegralSpec(center=(1, 2, 3), size=(0, 0, 1), sign="-")
    voltage_integral = make_voltage_integral(path_spec)

    path_spec = td.AxisAlignedCurrentIntegralSpec(center=(1, 2, 3), size=(0, 1, 1), sign="-")
    current_integral = make_current_integral(path_spec)

    path_spec = td.Custom2DVoltageIntegralSpec(vertices=[(0, 1), (0, 4)], axis=1, position=2)
    voltage_integral = make_voltage_integral(path_spec)

    path_spec = td.Custom2DCurrentIntegralSpec(
        vertices=[
            (0, 1),
            (0, 4),
            (3, 4),
            (3, 1),
        ],
        axis=1,
        position=2,
    )
    _ = make_current_integral(path_spec)

    with pytest.raises(pd.ValidationError):
        path_spec = td.Custom2DCurrentIntegralSpec(
            vertices=[
                (0, 1, 3),
                (0, 4, 5),
                (3, 4, 5),
                (3, 1, 5),
            ],
            axis=1,
            position=2,
        )


def test_mode_plane_analyzer_errors():
    """Check that the ModePlaneAnalyzer reports errors properly."""

    path_spec_gen = ModePlaneAnalyzer(size=(0, 2, 2), field_data_colocated=False)

    # First some quick sanity checks with the helper
    test_path = td.Box(center=(0, 0, 0), size=(0, 0.9, 0.1))
    test_shapely = [LineString([(-1, 0), (1, 0)])]
    assert path_spec_gen._check_box_intersects_with_conductors(test_shapely, test_path)

    test_path = td.Box(center=(0, 0, 0), size=(0, 2.1, 0.1))
    test_shapely = [LineString([(-1, 0), (1, 0)])]
    assert not path_spec_gen._check_box_intersects_with_conductors(test_shapely, test_path)

    sim = make_mw_sim(False, False, "microstrip")
    coax = td.GeometryGroup(
        geometries=(
            td.ClipOperation(
                operation="difference",
                geometry_a=td.Cylinder(axis=0, radius=2 * mm, center=(0, 0, 5 * mm), length=td.inf),
                geometry_b=td.Cylinder(
                    axis=0, radius=1.4 * mm, center=(0, 0, 5 * mm), length=td.inf
                ),
            ),
            td.Cylinder(axis=0, radius=1 * mm, center=(0, 0, 5 * mm), length=td.inf),
        )
    )
    coax_struct = td.Structure(geometry=coax, medium=td.PEC)
    sim = sim.updated_copy(structures=[coax_struct])
    mode_monitor = sim.monitors[0]
    modal_plane = td.Box(center=mode_monitor.center, size=mode_monitor.size)
    path_spec_gen = ModePlaneAnalyzer(
        center=modal_plane.center,
        size=modal_plane.size,
        field_data_colocated=mode_monitor.colocate,
    )
    with pytest.raises(SetupError):
        path_spec_gen.get_conductor_bounding_boxes(
            sim.structures,
            sim.grid,
            sim.symmetry,
            sim.bounding_box,
        )

    # Error when no conductors intersecting mode plane
    path_spec_gen = path_spec_gen.updated_copy(size=(0, 0.1, 0.1), center=(0, 0, 1.5))
    with pytest.raises(SetupError):
        path_spec_gen.get_conductor_bounding_boxes(
            sim.structures,
            sim.grid,
            sim.symmetry,
            sim.bounding_box,
        )


@pytest.mark.parametrize("colocate", [False, True])
@pytest.mark.parametrize("tline_type", ["microstrip", "cpw", "coax"])
def test_mode_plane_analyzer_canonical_shapes(colocate, tline_type):
    """Test canonical transmission line types to make sure the correct path integrals are generated."""
    sim = make_mw_sim(False, colocate, tline_type)
    mode_monitor = sim.monitors[0]
    modal_plane = td.Box(center=mode_monitor.center, size=mode_monitor.size)
    mode_plane_analyzer = ModePlaneAnalyzer(
        center=modal_plane.center,
        size=modal_plane.size,
        field_data_colocated=mode_monitor.colocate,
    )
    bounding_boxes, geos = mode_plane_analyzer.get_conductor_bounding_boxes(
        sim.structures,
        sim.grid,
        sim.symmetry,
        sim.bounding_box,
    )

    if tline_type == "coax":
        assert len(bounding_boxes) == 2
        for path_spec in bounding_boxes:
            assert np.all(np.isclose(path_spec.center, (0, 0, 5 * mm)))
    else:
        assert len(bounding_boxes) == 1
        assert np.all(np.isclose(bounding_boxes[0].center, (0, 0, 1.1 * mm)))


@pytest.mark.parametrize("use_2D", [False, True])
@pytest.mark.parametrize("symmetry", [(0, 0, 1), (0, 1, 1), (0, 1, 0)])
def test_mode_plane_analyzer_advanced(use_2D, symmetry):
    """The various symmetry permutations as well as with and without 2D structures."""
    sim = make_mw_sim(use_2D, False, "stripline")

    # Add shapes outside the portion considered for the symmetric simulation
    bottom_left = td.Structure(
        geometry=td.Box(
            center=[0, -5 * mm, -5 * mm],
            size=[td.inf, 1 * mm, 1 * mm],
        ),
        medium=td.PEC,
    )
    # Add shape only in the symmetric portion
    top_right = td.Structure(
        geometry=td.Box(
            center=[0, 5 * mm, 5 * mm],
            size=[td.inf, 1 * mm, 1 * mm],
        ),
        medium=td.PEC,
    )

    structures = [*list(sim.structures), bottom_left, top_right]
    sim = sim.updated_copy(symmetry=symmetry, structures=structures)
    mode_monitor = sim.monitors[0]

    modal_plane = td.Box(center=mode_monitor.center, size=mode_monitor.size)
    mode_plane_analyzer = ModePlaneAnalyzer(
        center=modal_plane.center,
        size=modal_plane.size,
        field_data_colocated=mode_monitor.colocate,
    )
    bounding_boxes, geos = mode_plane_analyzer.get_conductor_bounding_boxes(
        sim.structures,
        sim.grid,
        sim.symmetry,
        sim.bounding_box,
    )

    if symmetry[1] == 1 and symmetry[2] == 1:
        assert len(bounding_boxes) == 7
    else:
        assert len(bounding_boxes) == 5


@pytest.mark.parametrize(
    "mode_size", [(1.4 * mm, 1.0 * mm, 0), (1.4 * mm, 2 * mm, 0), (1.4 * mm - 1, 1.0 * mm + 1, 0)]
)
@pytest.mark.parametrize("symmetry", [(0, 0, 0), (0, 1, 0), (1, 1, 0)])
def test_mode_plane_analyzer_mode_bounds(mode_size, symmetry):
    """Test that the the mode plane bounds matches the mode solver grid bounds exactly."""

    dl = 0.1 * mm

    freq0 = (5e9) / 2
    fwidth = 4e9
    run_time = 60 / fwidth

    boundary_spec = td.BoundarySpec(
        x=td.Boundary(plus=td.PECBoundary(), minus=td.PECBoundary()),
        y=td.Boundary(plus=td.PECBoundary(), minus=td.PECBoundary()),
        z=td.Boundary(plus=td.PECBoundary(), minus=td.PECBoundary()),
    )
    impedance_specs = (td.AutoImpedanceSpec(),) * 4
    mode_spec = td.MicrowaveModeSpec(
        num_modes=4,
        target_neff=1.8,
        impedance_specs=impedance_specs,
    )

    metal_box = td.Structure(
        geometry=td.Box.from_bounds(
            rmin=(0.5 * mm, 0.5 * mm, 0.5 * mm), rmax=(1 * mm - 1 * dl, 0.5 * mm, 0.5 * mm)
        ),
        medium=td.PEC,
    )
    sim = td.Simulation(
        center=(0, 0, 0),
        size=(2 * mm, 2 * mm, 2 * mm),
        grid_spec=td.GridSpec.uniform(dl=dl),
        structures=(metal_box,),
        run_time=run_time,
        boundary_spec=boundary_spec,
        plot_length_units="mm",
        symmetry=symmetry,
    )

    mode_center = [0, 0, 0]
    mode_plane = td.Box(center=mode_center, size=mode_size)

    mms = ModeSolver(
        simulation=sim,
        plane=mode_plane,
        mode_spec=mode_spec,
        colocate=True,
        freqs=[freq0],
    )
    mode_solver_boundaries = mms._solver_grid.boundaries.to_list
    mode_plane_analyzer = ModePlaneAnalyzer(
        center=mode_center,
        size=mode_size,
        field_data_colocated=False,
    )
    mode_plane_limits = mode_plane_analyzer._get_mode_limits(sim.grid, sim.symmetry)

    for dim in (0, 1):
        solver_dim_boundaries = mode_solver_boundaries[dim]
        # TODO: Need the second check because the mode solver erroneously adds
        # an extra grid cell even when touching the simulation boundary
        assert (
            solver_dim_boundaries[0] == mode_plane_limits[0][dim]
            or mode_plane_limits[0][dim] == sim.bounds[0][dim]
        )
        assert (
            solver_dim_boundaries[-1] == mode_plane_limits[1][dim]
            or mode_plane_limits[1][dim] == sim.bounds[1][dim]
        )


def test_impedance_spec_validation():
    """Check that the various allowed methods for supplying path specifications are validated."""

    _ = td.AutoImpedanceSpec()

    v_spec = td.AxisAlignedVoltageIntegralSpec(center=(1, 2, 3), size=(0, 0, 1), sign="-")
    i_spec = td.AxisAlignedCurrentIntegralSpec(center=(1, 2, 3), size=(0, 1, 1), sign="-")

    # All valid methods
    both = td.CustomImpedanceSpec(voltage_spec=v_spec, current_spec=i_spec)
    voltage_only = td.CustomImpedanceSpec(
        voltage_spec=v_spec,
    )
    current_only = td.CustomImpedanceSpec(
        current_spec=i_spec,
    )

    # Invalid
    with pytest.raises(pd.ValidationError):
        _ = td.CustomImpedanceSpec(voltage_spec=None, current_spec=None)

    _ = td.MicrowaveModeSpec(num_modes=4, impedance_specs=(both, voltage_only, current_only, None))


def test_path_integral_factory_voltage_validation():
    """Test make_voltage_integral validation and error handling."""

    # Valid voltage specs
    axis_aligned_spec = td.AxisAlignedVoltageIntegralSpec(
        center=(1, 2, 3), size=(0, 0, 1), sign="-"
    )
    custom_2d_spec = td.Custom2DVoltageIntegralSpec(vertices=[(0, 1), (0, 4)], axis=1, position=2)

    # Test successful creation with axis-aligned spec
    voltage_integral = make_voltage_integral(axis_aligned_spec)
    assert voltage_integral is not None
    assert voltage_integral.center == (1, 2, 3)
    assert voltage_integral.size == (0, 0, 1)

    # Test successful creation with custom 2D spec
    voltage_integral = make_voltage_integral(custom_2d_spec)
    assert voltage_integral is not None
    assert voltage_integral.axis == 1
    assert voltage_integral.position == 2

    # Test ValidationError with unsupported type
    class UnsupportedVoltageSpec:
        def dict(self, exclude=None):
            return {}

    with pytest.raises(ValidationError, match="Unsupported voltage path specification type"):
        make_voltage_integral(UnsupportedVoltageSpec())


def test_path_integral_factory_current_validation():
    """Test make_current_integral validation and error handling."""

    # Valid current specs
    axis_aligned_spec = td.AxisAlignedCurrentIntegralSpec(
        center=(1, 2, 3), size=(0, 1, 1), sign="-"
    )
    custom_2d_spec = td.Custom2DCurrentIntegralSpec(
        vertices=[(0, 1), (0, 4), (3, 4), (3, 1)], axis=1, position=2
    )
    composite_spec = td.CompositeCurrentIntegralSpec(path_specs=[axis_aligned_spec], sum_spec="sum")

    # Test successful creation with axis-aligned spec
    current_integral = make_current_integral(axis_aligned_spec)
    assert current_integral is not None
    assert current_integral.center == (1, 2, 3)
    assert current_integral.size == (0, 1, 1)

    # Test successful creation with custom 2D spec
    current_integral = make_current_integral(custom_2d_spec)
    assert current_integral is not None
    assert current_integral.axis == 1
    assert current_integral.position == 2

    # Test successful creation with composite spec
    current_integral = make_current_integral(composite_spec)
    assert current_integral is not None
    assert len(current_integral.path_specs) == 1

    # Test ValidationError with unsupported type
    class UnsupportedCurrentSpec:
        def dict(self, exclude=None):
            return {}

    with pytest.raises(ValidationError, match="Unsupported current path specification type"):
        make_current_integral(UnsupportedCurrentSpec())


def test_make_path_integrals_validation():
    """Test make_path_integrals validation and error handling."""

    # Create a basic simulation setup
    sim = make_mw_sim(False, False, "microstrip")
    mode_monitor = sim.monitors[0]

    # Valid microwave mode spec with explicit specs
    v_spec = td.AxisAlignedVoltageIntegralSpec(center=(1, 2, 3), size=(0, 0, 1), sign="-")
    i_spec = td.AxisAlignedCurrentIntegralSpec(center=(1, 2, 3), size=(0, 1, 1), sign="-")

    impedance_spec = td.CustomImpedanceSpec(
        voltage_spec=v_spec,
        current_spec=i_spec,
    )
    microwave_spec = td.MicrowaveModeSpec(impedance_specs=(impedance_spec,))

    # Test successful creation
    voltage_integrals, current_integrals = make_path_integrals(microwave_spec)
    assert len(voltage_integrals) == 1
    assert len(current_integrals) == 1
    assert voltage_integrals[0] is not None
    assert current_integrals[0] is not None

    # Test with None specs - when both are None, use_automatic_setup is True
    # This means current integrals will be auto-generated, not None
    microwave_spec_none = td.MicrowaveModeSpec(impedance_specs=(None,))
    voltage_integrals, current_integrals = make_path_integrals(microwave_spec_none)
    assert len(voltage_integrals) == mode_monitor.mode_spec.num_modes
    assert len(current_integrals) == mode_monitor.mode_spec.num_modes
    assert all(vi is None for vi in voltage_integrals)
    assert all(ci is None for ci in current_integrals)


def test_make_path_integrals_construction_errors(monkeypatch):
    """Test that make_path_integrals handles construction errors properly."""

    # Create a valid spec
    v_spec = td.AxisAlignedVoltageIntegralSpec(center=(1, 2, 3), size=(0, 0, 1), sign="-")

    impedance_spec = td.CustomImpedanceSpec(voltage_spec=v_spec, current_spec=None)
    microwave_spec = td.MicrowaveModeSpec(impedance_specs=(impedance_spec,))

    # Mock make_voltage_integral to raise an exception
    def mock_make_voltage_integral(path_spec):
        raise RuntimeError("Intentional construction failure")

    monkeypatch.setattr(
        "tidy3d.components.microwave.path_integrals.factory.make_voltage_integral",
        mock_make_voltage_integral,
    )

    # This should raise a SetupError due to construction failure
    with pytest.raises(SetupError, match="Failed to construct path integrals"):
        make_path_integrals(microwave_spec)


def test_path_integral_factory_composite_current():
    """Test make_current_integral with CompositeCurrentIntegralSpec."""

    # Create base specs for the composite
    axis_aligned_spec1 = td.AxisAlignedCurrentIntegralSpec(
        center=(1, 2, 3), size=(0, 1, 1), sign="-"
    )
    axis_aligned_spec2 = td.AxisAlignedCurrentIntegralSpec(
        center=(2, 2, 3), size=(0, 1, 1), sign="+"
    )

    # Test creation of CompositeCurrentIntegralSpec
    composite_spec = td.CompositeCurrentIntegralSpec(
        path_specs=[axis_aligned_spec1, axis_aligned_spec2],
        sum_spec="sum",
    )

    # Test successful creation with composite spec
    current_integral = make_current_integral(composite_spec)
    assert current_integral is not None
    assert len(current_integral.path_specs) == 2

    # Test with different sum_spec options
    composite_spec_split = td.CompositeCurrentIntegralSpec(
        path_specs=[axis_aligned_spec1, axis_aligned_spec2],
        sum_spec="split",
    )
    current_integral_split = make_current_integral(composite_spec_split)
    assert current_integral_split is not None
    assert current_integral_split.sum_spec == "split"


def test_path_integral_factory_mixed_specs():
    """Test make_path_integrals with mixed voltage and current specs (some None)."""

    # Create specs where some are None
    v_spec = td.AxisAlignedVoltageIntegralSpec(center=(1, 2, 3), size=(0, 0, 1), sign="-")
    i_spec = td.AxisAlignedCurrentIntegralSpec(center=(1, 2, 3), size=(0, 1, 1), sign="-")

    # Test with mixed specs - some None, some specified
    impedance_spec1 = td.CustomImpedanceSpec(voltage_spec=v_spec, current_spec=None)
    impedance_spec2 = td.CustomImpedanceSpec(voltage_spec=None, current_spec=i_spec)
    microwave_spec = td.MicrowaveModeSpec(
        num_modes=2,
        impedance_specs=(
            impedance_spec1,
            impedance_spec2,
        ),
    )

    voltage_integrals, current_integrals = make_path_integrals(microwave_spec)

    assert len(voltage_integrals) == 2
    assert len(current_integrals) == 2
    assert voltage_integrals[0] is not None  # First mode has voltage spec
    assert voltage_integrals[1] is None  # Second mode has no voltage spec
    assert current_integrals[0] is None  # First mode has no current spec
    assert current_integrals[1] is not None  # Second mode has current spec


def test_mode_spec_with_microwave_mode_spec():
    """Test that the number of impedance specs is validated against the number of modes in MicrowaveModeSpec."""

    impedance_specs = (td.AutoImpedanceSpec(),)

    # Should work when impedance_specs matches num_modes
    mw_spec = td.MicrowaveModeSpec(num_modes=1, impedance_specs=impedance_specs)
    assert mw_spec.num_modes == 1
    assert len(mw_spec.impedance_specs) == 1

    # Should fail when impedance_specs doesn't match num_modes
    with pytest.raises(pd.ValidationError):
        td.MicrowaveModeSpec(num_modes=2, impedance_specs=impedance_specs)


def test_mode_solver_with_microwave_mode_spec():
    """Test running the mode locally and see if impedance is close to correct."""

    width = 1.0 * mm
    height = 0.5 * mm
    metal_thickness = 0.1 * mm

    stripline_sim = make_mw_sim(
        transmission_line_type="stripline",
        width=width,
        height=height,
        metal_thickness=metal_thickness,
    )
    dl = 0.05 * mm
    stripline_sim = stripline_sim.updated_copy(grid_spec=td.GridSpec.uniform(dl=dl))

    plane = td.Box(center=(0, 0, 0), size=(0, 10 * width, 2 * height + metal_thickness))
    num_modes = 3
    impedance_specs = td.AutoImpedanceSpec()
    mode_spec = td.MicrowaveModeSpec(
        num_modes=num_modes,
        target_neff=2.2,
        impedance_specs=impedance_specs,
    )
    mms = ModeSolver(
        simulation=stripline_sim,
        plane=plane,
        mode_spec=mode_spec,
        colocate=False,
        freqs=[1e9, 5e9, 10e9],
    )

    # _, ax = plt.subplots(1, 1, tight_layout=True, figsize=(15, 15))
    # mms.plot(ax=ax)
    # mms.plot_grid(ax=ax)
    # ax.set_aspect("equal")
    # plt.show()

    # This should raise a SetupError because the auto impedance spec cannot be used with the local mode solver
    with pytest.raises(SetupError, match="Auto path specification is not available"):
        mms_data: td.MicrowaveModeSolverData = mms.data

    # Manually defined impedance spec will work
    custom_spec = td.CustomImpedanceSpec(
        voltage_spec=None,
        current_spec=td.AxisAlignedCurrentIntegralSpec(
            size=(0, width + dl, metal_thickness + dl), sign="+"
        ),
    )
    impedance_specs = (custom_spec, None, None)
    microwave_spec_custom = td.MicrowaveModeSpec(
        num_modes=num_modes, target_neff=2.2, impedance_specs=impedance_specs
    )
    mms = mms.updated_copy(mode_spec=microwave_spec_custom)
    mms_data: td.MicrowaveModeSolverData = mms.data

    # _, ax = plt.subplots(1, 1, tight_layout=True, figsize=(15, 15))
    # mms_data.field_components["Ez"].isel(mode_index=0, f=0).real.plot(ax=ax)
    # ax.set_aspect("equal")
    # plt.show()
    mms_data.to_dataframe()

    assert np.all(
        np.isclose(mms_data.transmission_line_data.Z0.real.sel(mode_index=0), 28.6, rtol=0.2)
    )

    # Make sure a single spec can be used
    microwave_spec_custom = td.MicrowaveModeSpec(
        num_modes=num_modes, target_neff=2.2, impedance_specs=custom_spec
    )
    mms = mms.updated_copy(mode_spec=microwave_spec_custom)
    mms_data: td.MicrowaveModeSolverData = mms.data
    assert np.all(
        np.isclose(mms_data.transmission_line_data.Z0.real.sel(mode_index=0), 28.6, rtol=0.2)
    )


def test_mode_solver_with_microwave_group_index():
    """Test that group_index calculation with MicrowaveModeSpec correctly filters frequencies."""

    width = 1.0 * mm
    height = 0.5 * mm
    metal_thickness = 0.1 * mm

    stripline_sim = make_mw_sim(
        transmission_line_type="stripline",
        width=width,
        height=height,
        metal_thickness=metal_thickness,
    )
    dl = 0.05 * mm
    stripline_sim = stripline_sim.updated_copy(grid_spec=td.GridSpec.uniform(dl=dl))

    plane = td.Box(center=(0, 0, 0), size=(0, 10 * width, 2 * height + metal_thickness))
    num_modes = 1

    # Define original frequencies that we want in the final result
    original_freqs = [1e9, 5e9, 10e9]

    # Create custom impedance spec (AutoImpedanceSpec won't work with local mode solver)
    custom_spec = td.CustomImpedanceSpec(
        voltage_spec=None,
        current_spec=td.AxisAlignedCurrentIntegralSpec(
            size=(0, width + dl, metal_thickness + dl), sign="+"
        ),
    )

    # Enable group_index calculation
    mode_spec = td.MicrowaveModeSpec(
        num_modes=num_modes,
        target_neff=2.2,
        impedance_specs=custom_spec,
        group_index_step=True,  # This will expand frequencies to triplets
    )

    mms = ModeSolver(
        simulation=stripline_sim,
        plane=plane,
        mode_spec=mode_spec,
        colocate=False,
        freqs=original_freqs,
    )

    # Get the mode solver data
    mms_data: td.MicrowaveModeSolverData = mms.data

    # Verify that group index was calculated
    assert mms_data.n_group is not None, "Group index should be calculated"

    # Verify that transmission line data exists
    assert mms_data.transmission_line_data is not None, "Transmission line data should exist"

    # Verify that the frequencies in transmission_line_data match the original frequencies
    # (not the expanded triplet used internally for group index calculation)
    tl_freqs_Z0 = mms_data.transmission_line_data.Z0.coords["f"].values
    tl_freqs_voltage = mms_data.transmission_line_data.voltage_coeffs.coords["f"].values
    tl_freqs_current = mms_data.transmission_line_data.current_coeffs.coords["f"].values

    assert len(tl_freqs_Z0) == len(original_freqs), (
        f"Z0 should have {len(original_freqs)} frequencies, got {len(tl_freqs_Z0)}"
    )
    assert len(tl_freqs_voltage) == len(original_freqs), (
        f"voltage_coeffs should have {len(original_freqs)} frequencies, got {len(tl_freqs_voltage)}"
    )
    assert len(tl_freqs_current) == len(original_freqs), (
        f"current_coeffs should have {len(original_freqs)} frequencies, got {len(tl_freqs_current)}"
    )

    assert np.allclose(tl_freqs_Z0, original_freqs), (
        f"Z0 frequencies {tl_freqs_Z0} should match original {original_freqs}"
    )
    assert np.allclose(tl_freqs_voltage, original_freqs), (
        f"voltage_coeffs frequencies {tl_freqs_voltage} should match original {original_freqs}"
    )
    assert np.allclose(tl_freqs_current, original_freqs), (
        f"current_coeffs frequencies {tl_freqs_current} should match original {original_freqs}"
    )


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_voltage_integral_axes(axis):
    """Check AxisAlignedVoltageIntegral runs."""
    length = 0.5
    size = [0, 0, 0]
    size[axis] = length
    center = [0, 0, 0]
    voltage_integral = td.AxisAlignedVoltageIntegral(
        center=center,
        size=size,
        sign="+",
    )

    _ = voltage_integral.compute_voltage(SIM_Z_DATA["field"])


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_current_integral_axes(axis):
    """Check AxisAlignedCurrentIntegral runs."""
    length = 0.5
    size = [length, length, length]
    size[axis] = 0.0
    center = [0, 0, 0]
    current_integral = td.AxisAlignedCurrentIntegral(
        center=center,
        size=size,
        sign="+",
    )
    _ = current_integral.compute_current(SIM_Z_DATA["field"])


def test_voltage_integral_toggles():
    """Check AxisAlignedVoltageIntegral runs with toggles."""
    length = 0.5
    size = [0, 0, 0]
    size[0] = length
    center = [0, 0, 0]
    voltage_integral = td.AxisAlignedVoltageIntegral(
        center=center,
        size=size,
        extrapolate_to_endpoints=True,
        snap_path_to_grid=True,
        sign="-",
    )
    _ = voltage_integral.compute_voltage(SIM_Z_DATA["field"])


def test_current_integral_toggles():
    """Check AxisAlignedCurrentIntegral runs with toggles."""
    length = 0.5
    size = [length, length, length]
    size[0] = 0.0
    center = [0, 0, 0]
    current_integral = td.AxisAlignedCurrentIntegral(
        center=center,
        size=size,
        extrapolate_to_endpoints=True,
        snap_contour_to_grid=True,
        sign="-",
    )
    _ = current_integral.compute_current(SIM_Z_DATA["field"])


def test_voltage_missing_fields():
    """Check validation of AxisAlignedVoltageIntegral with missing fields."""
    length = 0.5
    size = [0, 0, 0]
    size[1] = length
    center = [0, 0, 0]
    voltage_integral = td.AxisAlignedVoltageIntegral(
        center=center,
        size=size,
        sign="+",
    )

    with pytest.raises(DataError):
        _ = voltage_integral.compute_voltage(SIM_Z_DATA["ExHx"])


def test_current_missing_fields():
    """Check validation of AxisAlignedCurrentIntegral with missing fields."""
    length = 0.5
    size = [length, length, length]
    size[0] = 0.0
    center = [0, 0, 0]
    current_integral = td.AxisAlignedCurrentIntegral(
        center=center,
        size=size,
        sign="+",
    )

    with pytest.raises(DataError):
        _ = current_integral.compute_current(SIM_Z_DATA["ExHx"])

    current_integral = td.AxisAlignedCurrentIntegral(
        center=center,
        size=[length, length, 0],
        sign="+",
    )

    with pytest.raises(DataError):
        _ = current_integral.compute_current(SIM_Z_DATA["ExHx"])


def test_time_monitor_voltage_integral():
    """Check AxisAlignedVoltageIntegral runs on time domain data."""
    length = 0.5
    size = [0, 0, 0]
    size[1] = length
    center = [0, 0, 0]
    voltage_integral = td.AxisAlignedVoltageIntegral(
        center=center,
        size=size,
        sign="+",
    )

    voltage_integral.compute_voltage(SIM_Z_DATA["field_time"])


def test_mode_solver_monitor_voltage_integral():
    """Check AxisAlignedVoltageIntegral runs on ModeData and ModeSolverData."""
    length = 0.5
    size = [0, 0, 0]
    size[1] = length
    center = [0, 0, 0]
    voltage_integral = td.AxisAlignedVoltageIntegral(
        center=center,
        size=size,
        sign="+",
    )

    voltage_integral.compute_voltage(SIM_Z_DATA["mode_solver"])
    voltage_integral.compute_voltage(SIM_Z_DATA["mode"])


def test_tiny_voltage_path():
    """Check AxisAlignedVoltageIntegral runs when given a very short path."""
    length = 0.02
    size = [0, 0, 0]
    size[1] = length
    center = [0, 0, 0]
    voltage_integral = td.AxisAlignedVoltageIntegral(
        center=center, size=size, sign="+", extrapolate_to_endpoints=True
    )

    _ = voltage_integral.compute_voltage(SIM_Z_DATA["field"])


def test_impedance_calculator():
    """Check validation of ImpedanceCalculator when integrals are missing."""
    with pytest.raises(pd.ValidationError):
        _ = td.ImpedanceCalculator(voltage_integral=None, current_integral=None)


def test_impedance_calculator_on_time_data():
    """Check ImpedanceCalculator runs on time domain data."""
    # Setup path integrals
    length = 0.5
    size = [0, length, 0]
    size[1] = length
    center = [0, 0, 0]
    voltage_integral = td.AxisAlignedVoltageIntegral(
        center=center, size=size, sign="+", extrapolate_to_endpoints=True
    )

    size = [length, length, 0]
    current_integral = td.AxisAlignedCurrentIntegral(center=center, size=size, sign="+")

    # Compute impedance using the tool
    Z_calc = td.ImpedanceCalculator(
        voltage_integral=voltage_integral, current_integral=current_integral
    )
    _ = Z_calc.compute_impedance(SIM_Z_DATA["field_time"])
    Z_calc = td.ImpedanceCalculator(voltage_integral=voltage_integral, current_integral=None)
    _ = Z_calc.compute_impedance(SIM_Z_DATA["field_time"])
    Z_calc = td.ImpedanceCalculator(voltage_integral=None, current_integral=current_integral)
    _ = Z_calc.compute_impedance(SIM_Z_DATA["field_time"])


def test_impedance_accuracy():
    """Test the accuracy of the ImpedanceCalculator."""
    field_data = make_field_data()
    # Setup path integrals
    size = [0, STRIP_HEIGHT / 2, 0]
    center = [0, -STRIP_HEIGHT / 4, 0]
    voltage_integral = td.AxisAlignedVoltageIntegral(
        center=center, size=size, sign="+", extrapolate_to_endpoints=True
    )

    size = [STRIP_WIDTH * 1.25, STRIP_HEIGHT / 2, 0]
    center = [0, 0, 0]
    current_integral = td.AxisAlignedCurrentIntegral(center=center, size=size, sign="+")

    def impedance_of_stripline(width, height):
        # Assuming no fringing fields, is the same as a parallel plate
        # with half the height and carrying twice the current
        Z0_parallel_plate = 0.5 * height / width * td.ETA_0
        return Z0_parallel_plate / 2

    analytic_impedance = impedance_of_stripline(STRIP_WIDTH, STRIP_HEIGHT)

    # Compute impedance using the tool
    Z_calc = td.ImpedanceCalculator(
        voltage_integral=voltage_integral, current_integral=current_integral
    )
    Z1 = Z_calc.compute_impedance(field_data)
    Z_calc = td.ImpedanceCalculator(voltage_integral=voltage_integral, current_integral=None)
    Z2 = Z_calc.compute_impedance(field_data)
    Z_calc = td.ImpedanceCalculator(voltage_integral=None, current_integral=current_integral)
    Z3 = Z_calc.compute_impedance(field_data)

    # Computation that uses the flux is less accurate, due to staircasing the field
    assert np.all(np.isclose(Z1, analytic_impedance, rtol=0.02))
    assert np.all(np.isclose(Z2, analytic_impedance, atol=3.5))
    assert np.all(np.isclose(Z3, analytic_impedance, atol=3.5))


def test_frequency_monitor_custom_voltage_integral():
    length = 0.5
    size = [0, 0, 0]
    size[1] = length
    # Make line
    vertices = [(0, 0), (0, 0.2), (0, 0.4)]
    voltage_integral = td.Custom2DVoltageIntegral(axis=2, position=0, vertices=vertices)
    voltage_integral.compute_voltage(SIM_Z_DATA["field"])


def test_vertices_validator_custom_current_integral():
    length = 0.5
    size = [0, 0, 0]
    size[1] = length
    # Make wrong box
    vertices = [(0.2, -0.2, 0.5), (0.2, 0.2), (-0.2, 0.2), (-0.2, -0.2), (0.2, -0.2)]

    with pytest.raises(pd.ValidationError):
        _ = td.Custom2DCurrentIntegral(axis=2, position=0, vertices=vertices)

    # Make wrong box shape
    vertices = [(0.2, 0.2, -0.2, -0.2, 0.2), (-0.2, 0.2, 0.2, -0.2, 0.2)]
    with pytest.raises(pd.ValidationError):
        _ = td.Custom2DCurrentIntegral(axis=2, position=0, vertices=vertices)


def test_fields_missing_custom_current_integral():
    length = 0.5
    size = [0, 0, 0]
    size[1] = length
    # Make box
    vertices = [(0.2, -0.2), (0.2, 0.2), (-0.2, 0.2), (-0.2, -0.2), (0.2, -0.2)]
    current_integral = td.Custom2DCurrentIntegral(axis=2, position=0, vertices=vertices)
    with pytest.raises(DataError):
        current_integral.compute_current(SIM_Z_DATA["ExHx"])


def test_wrong_monitor_data():
    """Check that the function arguments to integrals are correctly validated."""
    length = 0.5
    size = [0, 0, 0]
    size[1] = length
    # Make box
    vertices = [(0.2, -0.2), (0.2, 0.2), (-0.2, 0.2), (-0.2, -0.2), (0.2, -0.2)]
    current_integral = td.Custom2DCurrentIntegral(axis=2, position=0, vertices=vertices)
    with pytest.raises(DataError):
        current_integral.compute_current(SIM_Z.sources[0].source_time)
    with pytest.raises(DataError):
        current_integral.compute_current(em_field=SIM_Z.sources[0].source_time)


def test_time_monitor_custom_current_integral():
    length = 0.5
    size = [0, 0, 0]
    size[1] = length
    # Make box
    vertices = [(0.2, -0.2), (0.2, 0.2), (-0.2, 0.2), (-0.2, -0.2), (0.2, -0.2)]
    current_integral = td.Custom2DCurrentIntegral(axis=2, position=0, vertices=vertices)
    current_integral.compute_current(SIM_Z_DATA["field_time"])


def test_mode_solver_custom_current_integral():
    """Test that both ModeData and ModeSolverData are allowed types."""
    length = 0.5
    size = [0, 0, 0]
    size[1] = length
    # Make box
    vertices = [(0.2, -0.2), (0.2, 0.2), (-0.2, 0.2), (-0.2, -0.2), (0.2, -0.2)]
    current_integral = td.Custom2DCurrentIntegral(axis=2, position=0, vertices=vertices)
    current_integral.compute_current(SIM_Z_DATA["mode_solver"])
    current_integral.compute_current(SIM_Z_DATA["mode"])


def test_custom_current_integral_normal_y():
    current_integral = td.Custom2DCurrentIntegral.from_circular_path(
        center=(0, 0, 0), radius=0.4, num_points=31, normal_axis=1, clockwise=False
    )
    current_integral.compute_current(SIM_Z_DATA["field"])


def test_composite_current_integral_warnings():
    """Ensures that the checks function correctly on some test data."""
    f = [2e9, 3e9, 4e9]
    mode_index = list(np.arange(5))
    coords = {"f": f, "mode_index": mode_index}
    values = np.ones((3, 5))

    path_spec = td.AxisAlignedCurrentIntegralSpec(center=(0, 0, 0), size=(2, 2, 0), sign="+")
    composite_integral = td.CompositeCurrentIntegral(path_specs=[path_spec], sum_spec="split")

    phase_diff = FreqModeDataArray(np.angle(values), coords=coords)
    with AssertLogLevel(None):
        assert composite_integral._check_phase_sign_consistency(phase_diff)

    values[1, 2:] = -1
    phase_diff = FreqModeDataArray(np.angle(values), coords=coords)
    with AssertLogLevel("WARNING"):
        assert not composite_integral._check_phase_sign_consistency(phase_diff)

    values = np.ones((3, 5))
    in_phase = FreqModeDataArray(values, coords=coords)
    values = 0.5 * np.ones((3, 5))
    out_phase = FreqModeDataArray(values, coords=coords)
    with AssertLogLevel(None):
        assert composite_integral._check_phase_amplitude_consistency(in_phase, out_phase)

    values = 0.5 * np.ones((3, 5))
    values[2, 4:] = 1.5
    out_phase = FreqModeDataArray(values, coords=coords)
    with AssertLogLevel("WARNING"):
        assert not composite_integral._check_phase_amplitude_consistency(in_phase, out_phase)


def test_custom_path_integral_accuracy():
    """Test the accuracy of the custom path integral."""
    field_data = make_coax_field_data()

    current_integral = td.Custom2DCurrentIntegral.from_circular_path(
        center=(0, 0, 0), radius=0.4, num_points=31, normal_axis=2, clockwise=False
    )
    current = current_integral.compute_current(field_data)
    assert np.allclose(current.values, 1.0 / td.ETA_0, rtol=0.01)

    current_integral = td.Custom2DCurrentIntegral.from_circular_path(
        center=(0, 0, 0), radius=0.4, num_points=31, normal_axis=2, clockwise=True
    )
    current = current_integral.compute_current(field_data)
    assert np.allclose(current.values, -1.0 / td.ETA_0, rtol=0.01)


def test_impedance_accuracy_on_coaxial():
    """Test the accuracy of the ImpedanceCalculator."""
    field_data = make_coax_field_data()
    # Setup path integrals
    mean_radius = (COAX_R2 + COAX_R1) * 0.5
    size = [COAX_R2 - COAX_R1, 0, 0]
    center = [mean_radius, 0, 0]

    voltage_integral = td.AxisAlignedVoltageIntegral(
        center=center, size=size, sign="-", extrapolate_to_endpoints=True, snap_path_to_grid=True
    )
    current_integral = td.Custom2DCurrentIntegral.from_circular_path(
        center=(0, 0, 0), radius=mean_radius, num_points=31, normal_axis=2, clockwise=False
    )

    def impedance_of_coaxial_cable(r1, r2, wave_impedance=td.ETA_0):
        return np.log(r2 / r1) * wave_impedance / 2 / np.pi

    Z_analytic = impedance_of_coaxial_cable(COAX_R1, COAX_R2)

    # Compute impedance using the tool
    Z_calculator = td.ImpedanceCalculator(
        voltage_integral=voltage_integral, current_integral=current_integral
    )
    Z_calc = Z_calculator.compute_impedance(field_data)
    assert np.allclose(Z_calc, Z_analytic, rtol=0.04)


def test_creation_from_terminal_positions():
    """Test creating an AxisAlignedVoltageIntegral using terminal positions."""
    _ = td.AxisAlignedVoltageIntegral.from_terminal_positions(
        plus_terminal=2, minus_terminal=1, y=2.2, z=1
    )
    _ = td.AxisAlignedVoltageIntegral.from_terminal_positions(
        plus_terminal=1, minus_terminal=2, y=2.2, z=1
    )


def test_auto_path_integrals_for_lumped_element():
    """Test the auto creation of path integrals around lumped elements."""
    Rval = 75
    Cval = 0.2 * 1e-12
    fstart = 1e9
    fstop = 30e9
    freqs = np.linspace(fstart, fstop, 100)

    RLC = td.RLCNetwork(
        resistance=Rval,
        capacitance=Cval,
        network_topology="series",
    )

    linear_element = td.LinearLumpedElement(
        center=[0, 0, 0],
        size=[0.2, 0, 0.5],
        voltage_axis=2,
        network=RLC,
        name="RLC",
    )

    SIM_Z_with_element = SIM_Z.updated_copy(lumped_elements=[linear_element])

    _, _ = td.path_integrals_from_lumped_element(linear_element, SIM_Z_with_element.grid, "+")


def test_composite_current_integral_compute_current():
    """Test CompositeCurrentIntegral.compute_current method with different sum_spec behaviors."""

    # Create individual path specs for the composite
    path_spec1 = td.AxisAlignedCurrentIntegral(center=(0, 0, 0), size=(0.5, 0.5, 0), sign="+")
    path_spec2 = td.AxisAlignedCurrentIntegral(center=(0.25, 0, 0), size=(0.5, 0.5, 0), sign="-")

    # Test with sum_spec="sum"
    composite_integral_sum = td.CompositeCurrentIntegral(
        path_specs=[path_spec1, path_spec2], sum_spec="sum"
    )

    current_sum = composite_integral_sum.compute_current(SIM_Z_DATA["field"])
    assert current_sum is not None
    assert hasattr(current_sum, "values")

    # Test with sum_spec="split"
    composite_integral_split = td.CompositeCurrentIntegral(
        path_specs=[path_spec1, path_spec2], sum_spec="split"
    )

    current_split = composite_integral_split.compute_current(SIM_Z_DATA["field"])
    assert current_split is not None
    assert hasattr(current_split, "values")

    # Test that both methods return results with the same dimensions
    assert current_sum.dims == current_split.dims


def test_composite_current_integral_time_domain_error():
    """Test that CompositeCurrentIntegral raises error for time domain data with split sum_spec."""

    path_spec = td.AxisAlignedCurrentIntegral(center=(0, 0, 0), size=(0.5, 0.5, 0), sign="+")

    composite_integral = td.CompositeCurrentIntegral(path_specs=[path_spec], sum_spec="split")

    # Should raise DataError for time domain data with split sum_spec
    with pytest.raises(
        td.exceptions.DataError, match="Only frequency domain field data is supported"
    ):
        composite_integral.compute_current(SIM_Z_DATA["field_time"])


def test_composite_current_integral_phase_consistency_warnings():
    """Test CompositeCurrentIntegral phase consistency warning methods."""
    from tidy3d.components.data.data_array import FreqModeDataArray

    # Create a composite integral for testing
    path_spec = td.AxisAlignedCurrentIntegral(center=(0, 0, 0), size=(0.5, 0.5, 0), sign="+")

    composite_integral = td.CompositeCurrentIntegral(path_specs=[path_spec], sum_spec="split")

    # Test _check_phase_sign_consistency with consistent data
    f = [2e9, 3e9, 4e9]
    mode_index = list(np.arange(3))
    coords = {"f": f, "mode_index": mode_index}

    # Phase difference data that is consistent (all in phase)
    consistent_phase_values = np.zeros((3, 3))  # All zeros = in phase
    consistent_phase_diff = FreqModeDataArray(consistent_phase_values, coords=coords)

    # This should return True (no warning)
    result = composite_integral._check_phase_sign_consistency(consistent_phase_diff)
    assert result is True

    # Phase difference data that is inconsistent
    inconsistent_phase_values = np.array([[0, 0, 0], [0, np.pi, 0], [0, 0, np.pi]])  # Mixed phases
    inconsistent_phase_diff = FreqModeDataArray(inconsistent_phase_values, coords=coords)

    # This should return False and emit a warning
    # Note: The warning is logged, but we'll just test the return value here
    result = composite_integral._check_phase_sign_consistency(inconsistent_phase_diff)
    assert result is False

    # Test _check_phase_amplitude_consistency
    current_values = np.ones((3, 3))
    current_in_phase = FreqModeDataArray(current_values, coords=coords)
    current_out_phase = FreqModeDataArray(0.5 * current_values, coords=coords)

    # Consistent amplitudes (in_phase always larger)
    result = composite_integral._check_phase_amplitude_consistency(
        current_in_phase, current_out_phase
    )
    assert result is True

    # Inconsistent amplitudes (mix of which is larger)
    inconsistent_out_phase = FreqModeDataArray(
        np.array([[0.5, 0.5, 0.5], [1.5, 0.5, 0.5], [0.5, 1.5, 0.5]]), coords=coords
    )

    # This should return False and emit a warning
    # Note: The warning is logged, but we'll just test the return value here
    result = composite_integral._check_phase_amplitude_consistency(
        current_in_phase, inconsistent_out_phase
    )
    assert result is False


def test_impedance_calculator_compute_impedance_with_return_extras():
    """Test ImpedanceCalculator.compute_impedance with return_voltage_and_current=True."""

    # Setup path integrals
    voltage_integral = td.AxisAlignedVoltageIntegral(
        center=(0, 0, 0), size=(0, 0.5, 0), sign="+", extrapolate_to_endpoints=True
    )
    current_integral = td.AxisAlignedCurrentIntegral(center=(0, 0, 0), size=(0.5, 0.5, 0), sign="+")

    # Test with both voltage and current integrals
    Z_calc = td.ImpedanceCalculator(
        voltage_integral=voltage_integral, current_integral=current_integral
    )

    # Test with mode data that supports flux calculations
    result = Z_calc.compute_impedance(SIM_Z_DATA["mode"], return_voltage_and_current=True)

    # Should return a tuple of (impedance, voltage, current)
    assert isinstance(result, tuple)
    assert len(result) == 3
    impedance, voltage, current = result

    assert impedance is not None
    assert voltage is not None
    assert current is not None
    assert hasattr(impedance, "values")
    assert hasattr(voltage, "values")
    assert hasattr(current, "values")

    # Test with only voltage integral (current computed from flux)
    Z_calc_voltage_only = td.ImpedanceCalculator(voltage_integral=voltage_integral)

    result_voltage_only = Z_calc_voltage_only.compute_impedance(
        SIM_Z_DATA["mode"], return_voltage_and_current=True
    )

    assert isinstance(result_voltage_only, tuple)
    assert len(result_voltage_only) == 3
    impedance_v, voltage_v, current_v = result_voltage_only

    assert impedance_v is not None
    assert voltage_v is not None
    assert current_v is not None  # Should be computed from flux

    # Test with only current integral (voltage computed from flux)
    Z_calc_current_only = td.ImpedanceCalculator(current_integral=current_integral)

    result_current_only = Z_calc_current_only.compute_impedance(
        SIM_Z_DATA["mode"], return_voltage_and_current=True
    )

    assert isinstance(result_current_only, tuple)
    assert len(result_current_only) == 3
    impedance_c, voltage_c, current_c = result_current_only

    assert impedance_c is not None
    assert voltage_c is not None  # Should be computed from flux
    assert current_c is not None


def test_composite_current_integral_freq_mode_data():
    """Test CompositeCurrentIntegral works correctly with FreqModeDataArray."""

    # Create individual path specs for the composite
    path_spec1 = td.AxisAlignedCurrentIntegral(center=(0, 0, 0), size=(0.5, 0.5, 0), sign="+")
    path_spec2 = td.AxisAlignedCurrentIntegral(center=(0.25, 0, 0), size=(0.5, 0.5, 0), sign="-")

    # Test with sum_spec="sum" - should work with FreqModeDataArray
    composite_integral_sum = td.CompositeCurrentIntegral(
        path_specs=[path_spec1, path_spec2], sum_spec="sum"
    )

    # Use mode data which provides FreqModeDataArray
    current_sum = composite_integral_sum.compute_current(SIM_Z_DATA["mode"])
    assert current_sum is not None
    assert hasattr(current_sum, "values")

    # Verify it's a FreqModeDataArray by checking dimensions
    assert "f" in current_sum.dims
    assert "mode_index" in current_sum.dims

    # Test with sum_spec="split" - should also work with FreqModeDataArray
    composite_integral_split = td.CompositeCurrentIntegral(
        path_specs=[path_spec1, path_spec2], sum_spec="split"
    )

    current_split = composite_integral_split.compute_current(SIM_Z_DATA["mode"])
    assert current_split is not None
    assert hasattr(current_split, "values")

    # Verify it's a FreqModeDataArray by checking dimensions
    assert "f" in current_split.dims
    assert "mode_index" in current_split.dims

    # Test that both methods return compatible results
    assert current_sum.dims == current_split.dims
    assert current_sum.shape == current_split.shape


def test_impedance_calculator_mode_direction_handling():
    """Test that ImpedanceCalculator properly handles mode direction for flux calculation."""

    current_integral = td.AxisAlignedCurrentIntegral(center=(0, 0, 0), size=(0.5, 0.5, 0), sign="+")

    # Test with ModeSolverMonitor data
    Z_calc = td.ImpedanceCalculator(current_integral=current_integral)

    impedance_mode_solver = Z_calc.compute_impedance(SIM_Z_DATA["mode_solver"])
    assert impedance_mode_solver is not None

    # Test with ModeMonitor data
    impedance_mode = Z_calc.compute_impedance(SIM_Z_DATA["mode"])
    assert impedance_mode is not None

    # Both should produce valid impedance values
    assert hasattr(impedance_mode_solver, "values")
    assert hasattr(impedance_mode, "values")


def test_RF_license_suppression():
    """Ensure license warnings are being emitted properly and default instantiations avoid the warning."""
    original_setting = td.config.microwave.suppress_rf_license_warning
    td.config.microwave.suppress_rf_license_warning = False
    with AssertLogLevel("WARNING", contains_str="new license requirements"):
        mode_spec = td.MicrowaveModeSpec()
    with AssertLogLevel(None):
        mode_spec = td.MicrowaveModeSpec._default_without_license_warning()
    td.config.microwave.suppress_rf_license_warning = original_setting


def test_microwave_mode_data_reordering_with_transmission_line_data():
    """Test that transmission_line_data is correctly reordered when modes are reordered."""
    from tidy3d.components.data.data_array import (
        CurrentFreqModeDataArray,
        ImpedanceFreqModeDataArray,
        ModeIndexDataArray,
        ScalarModeFieldDataArray,
        VoltageFreqModeDataArray,
    )
    from tidy3d.components.microwave.data.dataset import TransmissionLineDataset

    # Setup coordinates
    x = [-1, 1, 3]
    y = [-2, 0]
    z = [-3, -1, 1, 3, 5]
    f = [2e14, 3e14]
    mode_index = np.arange(3)

    grid = td.Grid(boundaries=td.Coords(x=x, y=y, z=z))
    field_coords = {"x": x[:-1], "y": y[:-1], "z": z[:-1], "f": f, "mode_index": mode_index}
    index_coords = {"f": f, "mode_index": mode_index}

    # Create field data with distinct values for each mode
    field_values = np.zeros((2, 1, 4, 2, 3), dtype=complex)
    for mode_idx in range(3):
        # Each mode gets a unique value to track reordering
        field_values[:, :, :, :, mode_idx] = (mode_idx + 1) * (1 + 1j)

    field = ScalarModeFieldDataArray(field_values, coords=field_coords)

    # Create mode index data with distinct values for each mode
    index_values = np.zeros((2, 3), dtype=complex)
    for mode_idx in range(3):
        index_values[:, mode_idx] = (mode_idx + 1) * 1.5 + 0.1j
    index_data = ModeIndexDataArray(index_values, coords=index_coords)

    # Create transmission line data with distinct values for each mode
    impedance_values = np.zeros((2, 3))
    voltage_values = np.zeros((2, 3), dtype=complex)
    current_values = np.zeros((2, 3), dtype=complex)

    for mode_idx in range(3):
        # Each mode gets unique impedance, voltage, and current values
        impedance_values[:, mode_idx] = 50 * (mode_idx + 1)
        voltage_values[:, mode_idx] = (mode_idx + 1) * (10 + 5j)
        current_values[:, mode_idx] = (mode_idx + 1) * (0.2 + 0.1j)

    impedance_data = ImpedanceFreqModeDataArray(impedance_values, coords=index_coords)
    voltage_data = VoltageFreqModeDataArray(voltage_values, coords=index_coords)
    current_data = CurrentFreqModeDataArray(current_values, coords=index_coords)

    tl_data = TransmissionLineDataset(
        Z0=impedance_data, voltage_coeffs=voltage_data, current_coeffs=current_data
    )

    # Create monitor
    monitor = td.MicrowaveModeSolverMonitor(
        center=(0, 0, 0),
        size=(2, 0, 6),
        freqs=[2e14, 3e14],
        mode_spec=td.MicrowaveModeSpec(num_modes=3, impedance_specs=td.AutoImpedanceSpec()),
        name="microwave_mode_solver",
    )

    # Create MicrowaveModeSolverData
    data = td.MicrowaveModeSolverData(
        monitor=monitor,
        Ex=field,
        Ey=field,
        Ez=field,
        Hx=field,
        Hy=field,
        Hz=field,
        n_complex=index_data,
        grid_expanded=grid,
        transmission_line_data=tl_data,
    )

    # Define a reordering: reverse the mode order for each frequency
    # Shape: (num_freqs, num_modes) = (2, 3)
    # Original order: [0, 1, 2] -> New order: [2, 1, 0]
    sort_inds_2d = np.array([[2, 1, 0], [2, 1, 0]])

    # Apply mode reordering
    reordered_data = data._apply_mode_reorder(sort_inds_2d)

    # Verify that the main mode data is reordered correctly
    # Original mode 2 should now be at index 0
    original_mode_2_value = (2 + 1) * (1 + 1j)  # Mode 2 had value 3*(1+1j)
    assert np.allclose(
        reordered_data.Ex.isel(mode_index=0, x=0, y=0, z=0).values, original_mode_2_value
    ), "Main field data not reordered correctly"

    # Original mode 0 should now be at index 2
    original_mode_0_value = (0 + 1) * (1 + 1j)  # Mode 0 had value 1*(1+1j)
    assert np.allclose(
        reordered_data.Ex.isel(mode_index=2, x=0, y=0, z=0).values, original_mode_0_value
    ), "Main field data not reordered correctly"

    # Verify that transmission_line_data is also reordered correctly
    assert reordered_data.transmission_line_data is not None, (
        "transmission_line_data should not be None"
    )

    # Check Z0 reordering
    # Original mode 2 had Z0 = 50 * 3 = 150
    assert np.allclose(reordered_data.transmission_line_data.Z0.isel(mode_index=0).values, 150.0), (
        "transmission_line_data.Z0 not reordered correctly"
    )

    # Original mode 0 had Z0 = 50 * 1 = 50
    assert np.allclose(reordered_data.transmission_line_data.Z0.isel(mode_index=2).values, 50.0), (
        "transmission_line_data.Z0 not reordered correctly"
    )

    # Check voltage_coeffs reordering
    # Original mode 2 had voltage = 3 * (10 + 5j)
    assert np.allclose(
        reordered_data.transmission_line_data.voltage_coeffs.isel(mode_index=0).values,
        3 * (10 + 5j),
    ), "transmission_line_data.voltage_coeffs not reordered correctly"

    # Original mode 0 had voltage = 1 * (10 + 5j)
    assert np.allclose(
        reordered_data.transmission_line_data.voltage_coeffs.isel(mode_index=2).values,
        1 * (10 + 5j),
    ), "transmission_line_data.voltage_coeffs not reordered correctly"

    # Check current_coeffs reordering
    # Original mode 2 had current = 3 * (0.2 + 0.1j)
    assert np.allclose(
        reordered_data.transmission_line_data.current_coeffs.isel(mode_index=0).values,
        3 * (0.2 + 0.1j),
    ), "transmission_line_data.current_coeffs not reordered correctly"

    # Original mode 0 had current = 1 * (0.2 + 0.1j)
    assert np.allclose(
        reordered_data.transmission_line_data.current_coeffs.isel(mode_index=2).values,
        1 * (0.2 + 0.1j),
    ), "transmission_line_data.current_coeffs not reordered correctly"

    # Verify mode index data is also reordered
    # Original mode 2 had n_complex = 3 * 1.5 + 0.1j
    assert np.allclose(reordered_data.n_complex.isel(mode_index=0).values, 3 * 1.5 + 0.1j), (
        "n_complex not reordered correctly"
    )


def test_microwave_mode_data_interpolation():
    """Test that MicrowaveModeSolverData interpolation correctly handles transmission_line_data."""
    from tidy3d.components.data.data_array import (
        CurrentFreqModeDataArray,
        ImpedanceFreqModeDataArray,
        ModeIndexDataArray,
        ScalarModeFieldDataArray,
        VoltageFreqModeDataArray,
    )
    from tidy3d.components.microwave.data.dataset import TransmissionLineDataset

    # Setup coordinates with sparse frequencies
    x = [-1, 1, 3]
    y = [-2, 0]
    z = [-3, -1, 1, 3, 5]
    f_sparse = np.array([1e14, 1.5e14, 2e14])  # 3 source frequencies
    mode_index = np.arange(2)

    grid = td.Grid(boundaries=td.Coords(x=x, y=y, z=z))
    field_coords = {"x": x[:-1], "y": y[:-1], "z": z[:-1], "f": f_sparse, "mode_index": mode_index}
    index_coords = {"f": f_sparse, "mode_index": mode_index}

    # Create field data with frequency-dependent values
    field_values = np.zeros(
        (len(x) - 1, len(y) - 1, len(z) - 1, len(f_sparse), len(mode_index)), dtype=complex
    )
    for f_idx, freq in enumerate(f_sparse):
        for mode_idx in range(len(mode_index)):
            # Value depends on both frequency and mode: (freq/1e14) * (mode+1) * (1+1j)
            field_values[:, :, :, f_idx, mode_idx] = (freq / 1e14) * (mode_idx + 1) * (1 + 1j)

    field = ScalarModeFieldDataArray(field_values, coords=field_coords)

    # Create mode index data with frequency dependence
    index_values = np.zeros((len(f_sparse), len(mode_index)), dtype=complex)
    for f_idx, freq in enumerate(f_sparse):
        for mode_idx in range(len(mode_index)):
            # n_eff increases with frequency: 1.5 + (freq/1e14)*0.1 + mode_idx*0.2
            index_values[f_idx, mode_idx] = 1.5 + (freq / 1e14) * 0.1 + mode_idx * 0.2 + 0.01j
    index_data = ModeIndexDataArray(index_values, coords=index_coords)

    # Create transmission line data with frequency dependence
    impedance_values = np.zeros((len(f_sparse), len(mode_index)))
    voltage_values = np.zeros((len(f_sparse), len(mode_index)), dtype=complex)
    current_values = np.zeros((len(f_sparse), len(mode_index)), dtype=complex)

    for f_idx, freq in enumerate(f_sparse):
        for mode_idx in range(len(mode_index)):
            # Impedance varies with frequency: 50 + (freq/1e14)*10 + mode_idx*20
            impedance_values[f_idx, mode_idx] = 50 + (freq / 1e14) * 10 + mode_idx * 20
            # Voltage varies with frequency
            voltage_values[f_idx, mode_idx] = ((freq / 1e14) + mode_idx) * (10 + 5j)
            # Current varies with frequency
            current_values[f_idx, mode_idx] = ((freq / 1e14) + mode_idx) * (0.2 + 0.1j)

    impedance_data = ImpedanceFreqModeDataArray(impedance_values, coords=index_coords)
    voltage_data = VoltageFreqModeDataArray(voltage_values, coords=index_coords)
    current_data = CurrentFreqModeDataArray(current_values, coords=index_coords)

    tl_data = TransmissionLineDataset(
        Z0=impedance_data, voltage_coeffs=voltage_data, current_coeffs=current_data
    )

    # Create monitor
    monitor = td.MicrowaveModeSolverMonitor(
        center=(0, 0, 0),
        size=(2, 0, 6),
        freqs=f_sparse,
        mode_spec=td.MicrowaveModeSpec(num_modes=2, impedance_specs=td.AutoImpedanceSpec()),
        name="microwave_mode_solver",
    )

    # Create MicrowaveModeSolverData
    data = td.MicrowaveModeSolverData(
        monitor=monitor,
        Ex=field,
        Ey=field,
        Ez=field,
        Hx=field,
        Hy=field,
        Hz=field,
        n_complex=index_data,
        grid_expanded=grid,
        transmission_line_data=tl_data,
    )

    # Interpolate to denser frequency grid
    f_dense = np.linspace(1e14, 2e14, 11)

    # Test linear interpolation
    data_interp_linear = data.interp_in_freq(freqs=f_dense, method="linear", renormalize=False)

    # Verify that interpolated data has correct shape
    assert len(data_interp_linear.monitor.freqs) == len(f_dense), (
        "Interpolated data should have new frequency count"
    )
    assert data_interp_linear.Ex.shape[-1] == len(mode_index), "Mode count should be preserved"
    assert data_interp_linear.Ex.shape[-2] == len(f_dense), (
        "Frequency dimension should match target"
    )

    # Verify that transmission_line_data is also interpolated
    assert data_interp_linear.transmission_line_data is not None, (
        "transmission_line_data should be interpolated"
    )
    assert len(data_interp_linear.transmission_line_data.Z0.coords["f"]) == len(f_dense), (
        "transmission_line_data.Z0 should be interpolated to new frequencies"
    )
    assert len(data_interp_linear.transmission_line_data.voltage_coeffs.coords["f"]) == len(
        f_dense
    ), "transmission_line_data.voltage_coeffs should be interpolated to new frequencies"
    assert len(data_interp_linear.transmission_line_data.current_coeffs.coords["f"]) == len(
        f_dense
    ), "transmission_line_data.current_coeffs should be interpolated to new frequencies"

    # Test interpolation accuracy at midpoint
    f_mid = 1.5e14

    # Check that field interpolation is reasonable
    # At f=1.5e14, mode 0 should have value approximately (1.5) * 1 * (1+1j) = 1.5*(1+1j)
    field_at_mid_mode0 = data_interp_linear.Ex.sel(
        f=f_mid, mode_index=0, x=0, y=0, z=0, method="nearest"
    ).values
    expected_field_mode0 = 1.5 * 1 * (1 + 1j)
    assert np.allclose(field_at_mid_mode0, expected_field_mode0, rtol=0.01), (
        f"Field interpolation for mode 0: expected {expected_field_mode0}, got {field_at_mid_mode0}"
    )

    # Check that n_complex interpolation is reasonable
    # At f=1.5e14, mode 0 should have n_eff approximately 1.5 + 1.5*0.1 + 0*0.2 = 1.65
    n_complex_at_mid_mode0 = data_interp_linear.n_complex.sel(
        f=f_mid, mode_index=0, method="nearest"
    ).values
    expected_n_complex_mode0 = 1.5 + 1.5 * 0.1 + 0 * 0.2 + 0.01j
    assert np.allclose(n_complex_at_mid_mode0, expected_n_complex_mode0, rtol=0.01), (
        f"n_complex interpolation for mode 0: expected {expected_n_complex_mode0}, got {n_complex_at_mid_mode0}"
    )

    # Check that transmission line data interpolation is reasonable
    # At f=1.5e14, mode 0 should have Z0 approximately 50 + 1.5*10 + 0*20 = 65
    Z0_at_mid_mode0 = data_interp_linear.transmission_line_data.Z0.sel(
        f=f_mid, mode_index=0, method="nearest"
    ).values
    expected_Z0_mode0 = 50 + 1.5 * 10 + 0 * 20
    assert np.allclose(Z0_at_mid_mode0, expected_Z0_mode0, rtol=0.01), (
        f"Z0 interpolation for mode 0: expected {expected_Z0_mode0}, got {Z0_at_mid_mode0}"
    )

    # At f=1.5e14, mode 0 should have voltage approximately (1.5 + 0) * (10 + 5j) = 1.5*(10+5j)
    voltage_at_mid_mode0 = data_interp_linear.transmission_line_data.voltage_coeffs.sel(
        f=f_mid, mode_index=0, method="nearest"
    ).values
    expected_voltage_mode0 = 1.5 * (10 + 5j)
    assert np.allclose(voltage_at_mid_mode0, expected_voltage_mode0, rtol=0.01), (
        f"voltage_coeffs interpolation for mode 0: expected {expected_voltage_mode0}, got {voltage_at_mid_mode0}"
    )

    # At f=1.5e14, mode 0 should have current approximately (1.5 + 0) * (0.2 + 0.1j) = 1.5*(0.2+0.1j)
    current_at_mid_mode0 = data_interp_linear.transmission_line_data.current_coeffs.sel(
        f=f_mid, mode_index=0, method="nearest"
    ).values
    expected_current_mode0 = 1.5 * (0.2 + 0.1j)
    assert np.allclose(current_at_mid_mode0, expected_current_mode0, rtol=0.01), (
        f"current_coeffs interpolation for mode 0: expected {expected_current_mode0}, got {current_at_mid_mode0}"
    )

    # Test at endpoints to ensure they match original values
    # At f=1e14, mode 1 should have Z0 = 50 + 1*10 + 1*20 = 80
    Z0_at_start_mode1 = data_interp_linear.transmission_line_data.Z0.sel(
        f=1e14, mode_index=1, method="nearest"
    ).values
    expected_Z0_start_mode1 = 50 + 1 * 10 + 1 * 20
    assert np.allclose(Z0_at_start_mode1, expected_Z0_start_mode1, rtol=1e-6), (
        f"Z0 at endpoint should match original: expected {expected_Z0_start_mode1}, got {Z0_at_start_mode1}"
    )

    # At f=2e14, mode 1 should have Z0 = 50 + 2*10 + 1*20 = 90
    Z0_at_end_mode1 = data_interp_linear.transmission_line_data.Z0.sel(
        f=2e14, mode_index=1, method="nearest"
    ).values
    expected_Z0_end_mode1 = 50 + 2 * 10 + 1 * 20
    assert np.allclose(Z0_at_end_mode1, expected_Z0_end_mode1, rtol=1e-6), (
        f"Z0 at endpoint should match original: expected {expected_Z0_end_mode1}, got {Z0_at_end_mode1}"
    )
