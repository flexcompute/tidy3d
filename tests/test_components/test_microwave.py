"""Tests microwave tools."""

from __future__ import annotations

from math import isclose
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pydantic.v1 as pd
import pytest
import xarray as xr
from shapely.geometry import LineString, Polygon
from shapely.plotting import plot_line, plot_polygon

import tidy3d as td
import tidy3d.components.microwave.path_integrals.current_spec
import tidy3d.components.microwave.path_integrals.voltage_spec
from tidy3d.components.data.monitor_data import FreqDataArray, ModeSolverData
from tidy3d.components.microwave.formulas.circuit_parameters import (
    capacitance_colinear_cylindrical_wire_segments,
    capacitance_rectangular_sheets,
    inductance_straight_rectangular_wire,
    mutual_inductance_colinear_wire_segments,
    total_inductance_colinear_rectangular_wire_segments,
)
from tidy3d.components.microwave.path_integrals.impedance_spec import PathSpecGenerator
from tidy3d.components.microwave.path_integrals.path_integral_factory import (
    make_current_integral,
    make_path_integrals,
    make_voltage_integral,
)
from tidy3d.components.mode.mode_solver import ModeSolver
from tidy3d.components.types import Ax, Shapely
from tidy3d.constants import EPSILON_0
from tidy3d.exceptions import SetupError, ValidationError

from ..test_data.test_monitor_data import make_directivity_data

mm = 1e3

MAKE_PLOTS = False
if MAKE_PLOTS:
    # Interative plotting for debugging
    from matplotlib import use

    use("TkAgg")

COAX_R1 = 0.04
COAX_R2 = 0.5


def make_mw_sim(
    use_2D: bool = False,
    colocate: bool = False,
    transmission_line_type: Literal["microstrip", "cpw", "coax", "stripline"] = "microstrip",
    width=3 * mm,
    height=1 * mm,
) -> td.Simulation:
    """Helper to create a microwave simulation with a single type of transmission line present."""

    freq_start = 1e9
    freq_stop = 10e9

    freq0 = (freq_start + freq_stop) / 2
    fwidth = freq_stop - freq_start
    freqs = np.arange(freq_start, freq_stop, 1e9)

    run_time = 60 / fwidth

    length = 40 * mm
    thickness = 0.2 * mm
    sim_width = length

    pec = td.PEC
    if use_2D:
        thickness = 0.0
        pec = td.PEC2D

    epsr = 4.4
    diel = td.Medium(permittivity=epsr)

    metal_geos = []

    if transmission_line_type == "microstrip":
        metal_geos.append(
            td.Box(
                center=[0, 0, height + thickness / 2],
                size=[td.inf, width, thickness],
            )
        )
    elif transmission_line_type == "cpw":
        metal_geos.append(
            td.Box(
                center=[0, 0, height + thickness / 2],
                size=[td.inf, width, thickness],
            )
        )
        gnd_width = 10 * width
        gap = width / 5
        gnd_shift = gnd_width / 2 + gap + width / 2
        metal_geos.append(
            td.Box(
                center=[0, -gnd_shift, height + thickness / 2],
                size=[td.inf, gnd_width, thickness],
            )
        )
        metal_geos.append(
            td.Box(
                center=[0, gnd_shift, height + thickness / 2],
                size=[td.inf, gnd_width, thickness],
            )
        )
    elif transmission_line_type == "coax":
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
        metal_geos.append(
            td.Box(
                center=[0, 0, 0],
                size=[td.inf, width, thickness],
            )
        )
        gnd_width = 10 * width
        metal_geos.append(
            td.Box(
                center=[0, 0, height + thickness / 2],
                size=[td.inf, gnd_width, thickness],
            )
        )
        metal_geos.append(
            td.Box(
                center=[0, 0, -height - thickness / 2],
                size=[td.inf, gnd_width, thickness],
            )
        )
    else:
        raise AssertionError("Incorrect argument")

    metal_structures = [td.Structure(geometry=geo, medium=pec) for geo in metal_geos]
    substrate = td.Structure(
        geometry=td.Box(
            center=[0, 0, 0],
            size=[td.inf, td.inf, 2 * height],
        ),
        medium=diel,
    )

    structures = [substrate, *metal_structures]
    boundary_spec = td.BoundarySpec(
        x=td.Boundary(plus=td.PML(), minus=td.PML()),
        y=td.Boundary(plus=td.PML(), minus=td.PML()),
        z=td.Boundary(plus=td.PML(), minus=td.PECBoundary()),
    )

    size_sim = [
        length + 2 * width,
        sim_width,
        20 * mm + height + thickness,
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
    impedance_spec = (td.AutoImpedanceSpec(),) * 4
    mode_spec = td.ModeSpec(
        num_modes=4,
        target_neff=1.8,
        microwave_mode_spec=td.MicrowaveModeSpec(impedance_spec=impedance_spec),
    )

    mode_monitor = td.ModeMonitor(
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


def plot_auto_path_spec(
    path_spec: td.CompositeCurrentIntegralSpec,
    geoms: list[Shapely],
    ax: Ax = None,
) -> Ax:
    """Helper to plot composite path specifications along with the Shapely geometries used to generate them."""
    if ax is None:
        _, ax = plt.subplots(1, 1, tight_layout=True, figsize=(15, 15))

    for geom in geoms:
        if isinstance(geom, Polygon):
            plot_polygon(geom, ax=ax)
        elif isinstance(geom, LineString):
            plot_line(geom, ax=ax)
    i_integral = make_current_integral(path_spec)
    i_integral.plot(x=i_integral.center[0], ax=ax)


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


def test_path_integral_plotting():
    """Test that all types of path integrals correctly plot themselves."""

    mean_radius = (COAX_R2 + COAX_R1) * 0.5
    size = [COAX_R2 - COAX_R1, 0, 0]
    center = [mean_radius, 0, 0]

    voltage_integral = td.VoltageIntegralAxisAlignedSpec(
        center=center, size=size, sign="-", extrapolate_to_endpoints=True, snap_path_to_grid=True
    )

    current_integral = td.CustomCurrentIntegral2DSpec.from_circular_path(
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
    voltage_integral = td.CustomVoltageIntegral2DSpec(
        axis=1, position=0, vertices=[(-1, -1), (0, 0), (1, 1)]
    )

    current_integral = td.CurrentIntegralAxisAlignedSpec(
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


@pytest.mark.parametrize("colocate", [False, True])
@pytest.mark.parametrize("tline_type", ["microstrip", "cpw", "coax"])
def test_auto_path_spec_canonical_shapes(colocate, tline_type):
    """Test canonical transmission line types to make sure the correct path integrals are generated."""
    sim = make_mw_sim(False, colocate, tline_type)
    mode_monitor = sim.monitors[0]
    modal_plane = td.Box(center=mode_monitor.center, size=mode_monitor.size)
    path_spec_gen = PathSpecGenerator(
        center=modal_plane.center,
        size=modal_plane.size,
        field_data_colocated=mode_monitor.colocate,
    )
    comp_path_spec, geos = path_spec_gen.create_current_path_specs(
        sim.structures,
        sim.grid,
        sim.symmetry,
        sim.bounding_box,
    )

    if tline_type == "coax":
        assert len(comp_path_spec.path_specs) == 2
        for path_spec in comp_path_spec.path_specs:
            assert np.all(np.isclose(path_spec.center, (0, 0, 5 * mm)))
    else:
        assert len(comp_path_spec.path_specs) == 1
        assert np.all(np.isclose(comp_path_spec.path_specs[0].center, (0, 0, 1.1 * mm)))

    _, ax = plt.subplots(1, 1, tight_layout=True, figsize=(15, 15))
    sim.plot(x=modal_plane.center[0], ax=ax, monitor_alpha=0)
    sim.plot_grid(x=modal_plane.center[0], ax=ax, monitor_alpha=0)
    plot_auto_path_spec(comp_path_spec, geos, ax)
    ax.set_aspect("equal")
    ax.set_xlim(modal_plane.bounds[0][1], modal_plane.bounds[1][1])
    ax.set_ylim(modal_plane.bounds[0][2], modal_plane.bounds[1][2])
    if MAKE_PLOTS:
        plt.show()


@pytest.mark.parametrize("use_2D", [False, True])
@pytest.mark.parametrize("symmetry", [(0, 0, 1), (0, 1, 1), (0, 1, 0)])
def test_auto_path_spec_advanced(use_2D, symmetry):
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
    path_spec_gen = PathSpecGenerator(
        center=modal_plane.center,
        size=modal_plane.size,
        field_data_colocated=mode_monitor.colocate,
    )
    comp_path_spec, geos = path_spec_gen.create_current_path_specs(
        sim.structures,
        sim.grid,
        sim.symmetry,
        sim.bounding_box,
    )

    if symmetry[1] == 1 and symmetry[2] == 1:
        assert len(comp_path_spec.path_specs) == 7
    else:
        assert len(comp_path_spec.path_specs) == 5

    _, ax = plt.subplots(1, 1, tight_layout=True, figsize=(15, 15))
    sim.plot(x=modal_plane.center[0], ax=ax, monitor_alpha=0)
    sim.plot_grid(x=modal_plane.center[0], ax=ax, monitor_alpha=0)
    plot_auto_path_spec(comp_path_spec, geos, ax)
    ax.set_aspect("equal")
    ax.set_xlim(modal_plane.bounds[0][1], modal_plane.bounds[1][1])
    ax.set_ylim(modal_plane.bounds[0][2], modal_plane.bounds[1][2])
    if MAKE_PLOTS:
        plt.show()


def test_auto_path_spec_validation():
    """Check that the auto path specs are validated properly."""

    # First some quick sanity checks with the helper
    test_path = td.Box(center=(0, 0, 0), size=(0, 0.9, 0.1))
    test_shapely = [LineString([(-1, 0), (1, 0)])]
    assert PathSpecGenerator._check_path_intersects_with_conductors(test_shapely, test_path)

    test_path = td.Box(center=(0, 0, 0), size=(0, 2.1, 0.1))
    test_shapely = [LineString([(-1, 0), (1, 0)])]
    assert not PathSpecGenerator._check_path_intersects_with_conductors(test_shapely, test_path)

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
    path_spec_gen = PathSpecGenerator(
        center=modal_plane.center,
        size=modal_plane.size,
        field_data_colocated=mode_monitor.colocate,
    )
    with pytest.raises(ValidationError):
        path_spec_gen.create_current_path_specs(
            sim.structures,
            sim.grid,
            sim.symmetry,
            sim.bounding_box,
        )


def test_composite_current_integral_validation():
    """Ensures that the CompositeCurrentIntegralSpec is validated correctly."""

    current_spec = td.CurrentIntegralAxisAlignedSpec(center=(1, 2, 3), size=(0, 1, 1), sign="-")
    voltage_spec = td.VoltageIntegralAxisAlignedSpec(center=(1, 2, 3), size=(0, 0, 1), sign="-")
    path_spec = td.CompositeCurrentIntegralSpec(
        center=(1, 2, 3), size=(0, 1, 1), path_specs=[current_spec], sum_spec="sum"
    )

    with pytest.raises(pd.ValidationError):
        path_spec.updated_copy(path_specs=[])

    with pytest.raises(pd.ValidationError):
        path_spec.updated_copy(path_specs=[voltage_spec])


def test_path_integral_creation():
    """Check that path integrals are correctly constructed from path specifications."""

    path_spec = (
        tidy3d.components.microwave.path_integrals.voltage_spec.VoltageIntegralAxisAlignedSpec(
            center=(1, 2, 3), size=(0, 0, 1), sign="-"
        )
    )
    voltage_integral = make_voltage_integral(path_spec)

    path_spec = (
        tidy3d.components.microwave.path_integrals.current_spec.CurrentIntegralAxisAlignedSpec(
            center=(1, 2, 3), size=(0, 1, 1), sign="-"
        )
    )
    current_integral = make_current_integral(path_spec)

    path_spec = td.CustomVoltageIntegral2DSpec(vertices=[(0, 1), (0, 4)], axis=1, position=2)
    voltage_integral = make_voltage_integral(path_spec)

    path_spec = td.CustomCurrentIntegral2DSpec(
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
        path_spec = td.CustomCurrentIntegral2DSpec(
            vertices=[
                (0, 1, 3),
                (0, 4, 5),
                (3, 4, 5),
                (3, 1, 5),
            ],
            axis=1,
            position=2,
        )


def test_microwave_mode_spec_validation():
    """Check that the various allowed methods for supplying path specifications are validated."""

    _ = td.AutoImpedanceSpec()

    v_spec = td.VoltageIntegralAxisAlignedSpec(center=(1, 2, 3), size=(0, 0, 1), sign="-")
    i_spec = td.CurrentIntegralAxisAlignedSpec(center=(1, 2, 3), size=(0, 1, 1), sign="-")

    # All valid methods
    both = td.CustomImpedanceSpec(voltage_spec=v_spec, current_spec=i_spec)
    voltage_only = td.CustomImpedanceSpec(
        voltage_spec=v_spec,
    )
    current_only = td.CustomImpedanceSpec(
        current_spec=i_spec,
    )

    _ = td.MicrowaveModeSpec(impedance_spec=(both, voltage_only, current_only, None))


def test_path_integral_factory_voltage_validation():
    """Test make_voltage_integral validation and error handling."""

    # Valid voltage specs
    axis_aligned_spec = td.VoltageIntegralAxisAlignedSpec(
        center=(1, 2, 3), size=(0, 0, 1), sign="-"
    )
    custom_2d_spec = td.CustomVoltageIntegral2DSpec(vertices=[(0, 1), (0, 4)], axis=1, position=2)

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
    axis_aligned_spec = td.CurrentIntegralAxisAlignedSpec(
        center=(1, 2, 3), size=(0, 1, 1), sign="-"
    )
    custom_2d_spec = td.CustomCurrentIntegral2DSpec(
        vertices=[(0, 1), (0, 4), (3, 4), (3, 1)], axis=1, position=2
    )
    composite_spec = td.CompositeCurrentIntegralSpec(
        center=(1, 2, 3), size=(0, 1, 1), path_specs=[axis_aligned_spec], sum_spec="sum"
    )

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
    assert current_integral.center == (1, 2, 3)
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
    v_spec = td.VoltageIntegralAxisAlignedSpec(center=(1, 2, 3), size=(0, 0, 1), sign="-")
    i_spec = td.CurrentIntegralAxisAlignedSpec(center=(1, 2, 3), size=(0, 1, 1), sign="-")

    impedance_spec = td.CustomImpedanceSpec(
        voltage_spec=v_spec,
        current_spec=i_spec,
    )
    microwave_mode_spec = td.MicrowaveModeSpec(impedance_spec=(impedance_spec,))

    # Test successful creation
    voltage_integrals, current_integrals = make_path_integrals(
        microwave_mode_spec, mode_monitor, sim
    )
    assert len(voltage_integrals) == 1
    assert len(current_integrals) == 1
    assert voltage_integrals[0] is not None
    assert current_integrals[0] is not None

    # Test with None specs - when both are None, use_automatic_setup is True
    # This means current integrals will be auto-generated, not None
    microwave_mode_spec_none = td.MicrowaveModeSpec(impedance_spec=(None,))
    voltage_integrals, current_integrals = make_path_integrals(
        microwave_mode_spec_none, mode_monitor, sim
    )
    assert len(voltage_integrals) == mode_monitor.mode_spec.num_modes
    assert len(current_integrals) == mode_monitor.mode_spec.num_modes
    assert all(vi is None for vi in voltage_integrals)
    assert all(ci is None for ci in current_integrals)

    # Test with automatic setup enabled (same as above, but explicit)
    microwave_mode_spec_auto = td.MicrowaveModeSpec(impedance_spec=(td.AutoImpedanceSpec(),))

    voltage_integrals_auto, current_integrals_auto = make_path_integrals(
        microwave_mode_spec_auto, mode_monitor, sim
    )
    assert len(voltage_integrals_auto) == mode_monitor.mode_spec.num_modes
    assert len(current_integrals_auto) == mode_monitor.mode_spec.num_modes
    assert all(vi is None for vi in voltage_integrals_auto)
    assert all(ci is not None for ci in current_integrals_auto)


def test_make_path_integrals_setup_errors():
    """Test that make_path_integrals raises appropriate SetupErrors."""

    # Create a simulation that will cause path spec generation to fail
    sim = make_mw_sim(False, False, "microstrip")
    mode_monitor = sim.monitors[0]

    # Create a microwave mode spec that will trigger auto-generation but fail
    # We'll modify the simulation to have structures that will cause validation errors
    bad_coax = td.GeometryGroup(
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
    bad_struct = td.Structure(geometry=bad_coax, medium=td.PEC)
    bad_sim = sim.updated_copy(structures=[bad_struct])

    microwave_mode_spec_auto = td.MicrowaveModeSpec(impedance_spec=(td.AutoImpedanceSpec(),))
    # This should raise a SetupError due to auto-generation failure
    with pytest.raises(SetupError, match="Failed to auto-generate path specification"):
        make_path_integrals(microwave_mode_spec_auto, mode_monitor, bad_sim)


def test_make_path_integrals_construction_errors(monkeypatch):
    """Test that make_path_integrals handles construction errors properly."""

    sim = make_mw_sim(False, False, "microstrip")
    mode_monitor = sim.monitors[0]

    # Create a valid spec
    v_spec = td.VoltageIntegralAxisAlignedSpec(center=(1, 2, 3), size=(0, 0, 1), sign="-")

    impedance_spec = td.CustomImpedanceSpec(voltage_spec=v_spec, current_spec=None)
    microwave_mode_spec = td.MicrowaveModeSpec(impedance_spec=(impedance_spec,))

    # Mock make_voltage_integral to raise an exception
    def mock_make_voltage_integral(path_spec):
        raise RuntimeError("Intentional construction failure")

    monkeypatch.setattr(
        "tidy3d.components.microwave.path_integrals.path_integral_factory.make_voltage_integral",
        mock_make_voltage_integral,
    )

    # This should raise a SetupError due to construction failure
    with pytest.raises(SetupError, match="Failed to construct path integrals"):
        make_path_integrals(microwave_mode_spec, mode_monitor, sim)


def test_path_integral_factory_composite_current():
    """Test make_current_integral with CompositeCurrentIntegralSpec and integration."""

    # Create base specs for the composite
    axis_aligned_spec1 = td.CurrentIntegralAxisAlignedSpec(
        center=(1, 2, 3), size=(0, 1, 1), sign="-"
    )
    axis_aligned_spec2 = td.CurrentIntegralAxisAlignedSpec(
        center=(2, 2, 3), size=(0, 1, 1), sign="+"
    )

    # Test creation of CompositeCurrentIntegralSpec
    composite_spec = td.CompositeCurrentIntegralSpec(
        center=(1.5, 2, 3),
        size=(2, 1, 1),
        path_specs=[axis_aligned_spec1, axis_aligned_spec2],
        sum_spec="sum",
    )

    # Test successful creation with composite spec
    current_integral = make_current_integral(composite_spec)
    assert current_integral is not None
    assert current_integral.center == (1.5, 2, 3)
    assert len(current_integral.path_specs) == 2

    # Test with different sum_spec options
    composite_spec_split = td.CompositeCurrentIntegralSpec(
        center=(1.5, 2, 3),
        size=(2, 1, 1),
        path_specs=[axis_aligned_spec1, axis_aligned_spec2],
        sum_spec="split",
    )
    current_integral_split = make_current_integral(composite_spec_split)
    assert current_integral_split is not None
    assert current_integral_split.sum_spec == "split"


def test_path_integral_factory_mixed_specs():
    """Test make_path_integrals with mixed voltage and current specs (some None)."""

    sim = make_mw_sim(False, False, "microstrip")
    mode_monitor = sim.monitors[0]

    # Create specs where some are None
    v_spec = td.VoltageIntegralAxisAlignedSpec(center=(1, 2, 3), size=(0, 0, 1), sign="-")
    i_spec = td.CurrentIntegralAxisAlignedSpec(center=(1, 2, 3), size=(0, 1, 1), sign="-")

    # Test with mixed specs - some None, some specified
    impedance_spec1 = td.CustomImpedanceSpec(voltage_spec=v_spec, current_spec=None)
    impedance_spec2 = td.CustomImpedanceSpec(voltage_spec=None, current_spec=i_spec)
    microwave_mode_spec = td.MicrowaveModeSpec(
        impedance_spec=(
            impedance_spec1,
            impedance_spec2,
        )
    )
    voltage_integrals, current_integrals = make_path_integrals(
        microwave_mode_spec, mode_monitor, sim
    )

    assert len(voltage_integrals) == 2
    assert len(current_integrals) == 2
    assert voltage_integrals[0] is not None  # First mode has voltage spec
    assert voltage_integrals[1] is None  # Second mode has no voltage spec
    assert current_integrals[0] is None  # First mode has no current spec
    assert current_integrals[1] is not None  # Second mode has current spec


def test_mode_solver_with_microwave_mode_spec():
    """Test make_path_integrals with mixed voltage and current specs (some None)."""

    def analytical_stripline_impedance(er, width, height, thickness):
        assert width > 0.35 * height
        term1 = 1 - thickness / height
        Cf = 2 / np.pi * np.log(1 / term1 + 1)
        if thickness > 0:
            Cf -= thickness / np.pi / height * np.log(1 / term1**2 - 1)
        return 30 * np.pi / np.sqrt(er) * (term1) / (width / height + Cf)

    stripline_sim = make_mw_sim(transmission_line_type="stripline", width=2 * mm, height=1 * mm)

    stripline_sim = stripline_sim.updated_copy(grid_spec=td.GridSpec.uniform(dl=0.05 * mm))
    analytical_stripline_impedance(5.6, 0.5, 1, 0)
    height = 1 * mm
    width = 10 * mm
    thickness = 0.1 * mm
    plane = td.Box(center=(0, 0, 0), size=(0, width, 2 * height))
    num_modes = 3
    impedance_spec = (td.AutoImpedanceSpec(), None, None)
    mode_spec = td.ModeSpec(
        num_modes=num_modes,
        target_neff=2.2,
        microwave_mode_spec=td.MicrowaveModeSpec(impedance_spec=impedance_spec),
    )

    mms = ModeSolver(
        simulation=stripline_sim,
        plane=plane,
        mode_spec=mode_spec,
        freqs=[1e9, 5e9, 10e9],
    )
    snapped_grid: td.Grid = mms.grid_snapped

    print(snapped_grid.num_cells)

    mms_data: ModeSolverData = mms.data
    print(mms_data.to_dataframe())
    print(mms_data.microwave_data.Z0)
    print(mms_data.microwave_data.voltage_coeffs)
    print(mms_data.microwave_data.current_coeffs)
    print(analytical_stripline_impedance(4.4, 2 * mm, 2 * height, thickness))

    _, ax = plt.subplots(1, 1, tight_layout=True, figsize=(15, 15))
    mms.plot(ax=ax)
    mms.plot_grid(ax=ax)
    ax.set_aspect("equal")
    if MAKE_PLOTS:
        plt.show()
