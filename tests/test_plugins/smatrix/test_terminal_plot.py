"""Tests for terminal port plotting (plot_port) across all port types."""

from __future__ import annotations

import itertools

import matplotlib.pyplot as plt
import numpy as np
import pytest

import tidy3d as td
from tidy3d.components.geometry.float_utils import increment_float
from tidy3d.plugins.smatrix import (
    CoaxialLumpedPort,
    LumpedPort,
    TerminalComponentModeler,
    TerminalWavePort,
    WavePort,
)
from tidy3d.plugins.smatrix.component_modelers.terminal import _pack_label_centers_1d

# ---------------------------------------------------------------------------
# Geometry constants
# ---------------------------------------------------------------------------

# Microstrip
_MS_STRIP_LENGTH = 75e3
_MS_STRIP_WIDTH = 3e3
_MS_GAP = 1e3
_MS_GND_WIDTH = _MS_STRIP_WIDTH * 8
_MS_METAL_THICKNESS = 0.2e3

# Coaxial
_COAX_RINNER = 0.2768e3
_COAX_ROUTER = 1.0e3
_COAX_LENGTH = 75e3

# Stripline
_MIL = 25.4
_SL_W = 3.2 * _MIL
_SL_T = 0.7 * _MIL
_SL_H = 10.7 * _MIL
_SL_SE = 7 * _MIL
_SL_L = 4000 * _MIL

# Shared
_PEC = td.PECMedium()
_DIEL = td.Medium(permittivity=4.4)
_FREQ_STOP = 10e9
_WAVELENGTH0 = td.C_0 / 5e9
_FREQS = np.array([1e9, 10e9])

# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------


def _make_microstrip_sim() -> td.Simulation:
    """Microstrip line: ground plane + substrate + signal strip."""
    ground = td.Structure(
        geometry=td.Box(
            center=[0, 0, _MS_METAL_THICKNESS / 2],
            size=[_MS_STRIP_LENGTH, _MS_GND_WIDTH, _MS_METAL_THICKNESS],
        ),
        medium=_PEC,
    )
    substrate = td.Structure(
        geometry=td.Box(
            center=[0, 0, _MS_METAL_THICKNESS + _MS_GAP / 2],
            size=[_MS_STRIP_LENGTH, _MS_GND_WIDTH, _MS_GAP],
        ),
        medium=_DIEL,
    )
    strip = td.Structure(
        geometry=td.Box(
            center=[0, 0, _MS_METAL_THICKNESS + _MS_GAP + _MS_METAL_THICKNESS / 2],
            size=[_MS_STRIP_LENGTH, _MS_STRIP_WIDTH, _MS_METAL_THICKNESS],
        ),
        medium=_PEC,
    )
    center_z = _MS_METAL_THICKNESS + _MS_GAP / 2 + _MS_GAP * 2
    return td.Simulation(
        center=[0, 0, center_z],
        size=[
            _MS_STRIP_LENGTH + 0.5 * _WAVELENGTH0,
            _MS_GND_WIDTH + 0.5 * _WAVELENGTH0,
            2 * _MS_METAL_THICKNESS + _MS_GAP + 0.5 * _WAVELENGTH0,
        ],
        grid_spec=td.GridSpec.auto(min_steps_per_wvl=10, wavelength=td.C_0 / _FREQ_STOP),
        structures=[substrate, strip, ground],
        sources=[],
        monitors=[],
        run_time=60 / (9e9),
        boundary_spec=td.BoundarySpec.all_sides(boundary=td.PML()),
        shutoff=1e-4,
    )


def _make_coaxial_sim() -> td.Simulation:
    """Coaxial cable: inner conductor + outer shell."""
    inner_conductor = td.Structure(
        geometry=td.Cylinder(center=(0, 0, 0), radius=_COAX_RINNER, length=_COAX_LENGTH, axis=2),
        medium=_PEC,
    )
    outer_shell = td.Structure(
        geometry=td.ClipOperation(
            operation="difference",
            geometry_a=td.Cylinder(
                center=(0, 0, 0), radius=_COAX_ROUTER * 1.1, length=_COAX_LENGTH, axis=2
            ),
            geometry_b=td.Cylinder(
                center=(0, 0, 0), radius=_COAX_ROUTER, length=_COAX_LENGTH, axis=2
            ),
        ),
        medium=_PEC,
    )
    return td.Simulation(
        center=[0, 0, 0],
        size=[4 * _COAX_ROUTER, 4 * _COAX_ROUTER, _COAX_LENGTH + 0.5 * _WAVELENGTH0],
        grid_spec=td.GridSpec.auto(min_steps_per_wvl=10, wavelength=td.C_0 / _FREQ_STOP),
        structures=[inner_conductor, outer_shell],
        sources=[],
        monitors=[],
        run_time=60 / (9e9),
        boundary_spec=td.BoundarySpec.all_sides(boundary=td.PML()),
        shutoff=1e-4,
    )


def _make_stripline_sim() -> td.Simulation:
    """Differential stripline: 2 signal traces + 2 ground planes + substrate."""
    len_inf = 1e6
    f_max = 70e9

    left_strip = td.Box(center=(-(_SL_SE + _SL_W) / 2, 0, 0), size=(_SL_W, _SL_T, _SL_L))
    right_strip = td.Box(center=((_SL_SE + _SL_W) / 2, 0, 0), size=(_SL_W, _SL_T, _SL_L))

    str_sub = td.Structure(
        geometry=td.Box(center=(0, 0, 0), size=(len_inf, _SL_H, _SL_L)),
        medium=_DIEL,
    )
    str_signal = td.Structure(
        geometry=td.GeometryGroup(geometries=[left_strip, right_strip]),
        medium=td.PEC,
    )
    str_gnd_top = td.Structure(
        geometry=td.Box(center=(0, _SL_H / 2 + _SL_T / 2, 0), size=(len_inf, _SL_T, _SL_L)),
        medium=td.PEC,
    )
    str_gnd_bot = td.Structure(
        geometry=td.Box(center=(0, -_SL_H / 2 - _SL_T / 2, 0), size=(len_inf, _SL_T, _SL_L)),
        medium=td.PEC,
    )

    lr_spec = td.LayerRefinementSpec.from_structures(
        structures=[str_signal],
        axis=1,
        min_steps_along_axis=10,
        refinement_inside_sim_only=False,
        bounds_snapping="bounds",
        corner_refinement=td.GridRefinement(dl=_SL_T / 10, num_cells=2),
    )
    lr_spec_top = lr_spec.updated_copy(
        center=(0, _SL_H / 2 + _SL_T / 2, 0), size=(len_inf, _SL_T, _SL_L)
    )
    lr_spec_bot = lr_spec.updated_copy(
        center=(0, -_SL_H / 2 - _SL_T / 2, 0), size=(len_inf, _SL_T, _SL_L)
    )

    grid_spec = td.GridSpec.auto(
        wavelength=td.C_0 / f_max,
        min_steps_per_wvl=30,
        layer_refinement_specs=[lr_spec, lr_spec_top, lr_spec_bot],
    )

    return td.Simulation(
        size=(50 * _MIL, _SL_H + 2 * _SL_T, 1.05 * _SL_L),
        center=(0, 0, 0),
        grid_spec=grid_spec,
        boundary_spec=td.BoundarySpec(
            x=td.Boundary.pml(), y=td.Boundary.pec(), z=td.Boundary.pml()
        ),
        structures=[str_sub, str_signal, str_gnd_top, str_gnd_bot],
        monitors=[],
        run_time=2e-9,
        shutoff=1e-7,
    )


# ---------------------------------------------------------------------------
# Label packing tests
# ---------------------------------------------------------------------------


def _assert_non_overlapping_centers(
    centers_px: np.ndarray, widths_px: np.ndarray, pad_px: float
) -> None:
    order = np.argsort(centers_px)
    for left_idx, right_idx in itertools.pairwise(order):
        left_right_edge = centers_px[left_idx] + widths_px[left_idx] / 2
        right_left_edge = centers_px[right_idx] - widths_px[right_idx] / 2
        assert right_left_edge >= left_right_edge + pad_px - 1e-9


def test_pack_label_centers_1d_non_overlapping_and_in_bounds():
    """Packed labels stay non-overlapping and within the given bounds."""
    anchors = np.array([10.0, 11.0, 12.0])
    widths = np.array([10.0, 10.0, 10.0])
    x_min, x_max = 0.0, 40.0
    pad = 2.0

    centers = _pack_label_centers_1d(
        anchor_centers_px=anchors, widths_px=widths, x_min_px=x_min, x_max_px=x_max, pad_px=pad
    )

    _assert_non_overlapping_centers(centers_px=centers, widths_px=widths, pad_px=pad)
    assert np.all(centers - widths / 2 >= x_min - 1e-9)
    assert np.all(centers + widths / 2 <= x_max + 1e-9)


def test_pack_label_centers_1d_handles_right_overflow():
    """Labels anchored near the right edge are shifted left to stay in bounds."""
    anchors = np.array([30.0, 31.0, 32.0])
    widths = np.array([12.0, 12.0, 12.0])
    x_min, x_max = 0.0, 40.0
    pad = 2.0

    centers = _pack_label_centers_1d(
        anchor_centers_px=anchors, widths_px=widths, x_min_px=x_min, x_max_px=x_max, pad_px=pad
    )

    _assert_non_overlapping_centers(centers_px=centers, widths_px=widths, pad_px=pad)
    assert np.all(centers + widths / 2 <= x_max + 1e-9)


# ---------------------------------------------------------------------------
# TerminalWavePort plot tests
# ---------------------------------------------------------------------------


def test_plot_port_terminal_wave_port():
    """Test plot_port for TerminalWavePort with differential pairs on stripline."""
    sim = _make_stripline_sim()
    port = TerminalWavePort(
        center=(0, 0, -_SL_L / 2),
        size=(1e6, 1e6, 0),
        direction="+",
        name="terminal",
        differential_pairs=(("T0", "T1"),),
    )
    modeler = TerminalComponentModeler(simulation=sim, ports=[port], freqs=_FREQS)

    ax = modeler.plot_port("terminal")
    assert ax is not None
    plt.close()

    ax = modeler.plot_port(port)
    assert ax is not None
    plt.close()

    modeler.plot_sim_grid(z=port.center[2])
    plt.close()


def test_plot_port_terminal_wave_port_custom_specs():
    """Test plot_port with explicit voltage specs and invalid port type error."""
    spec0 = td.CustomImpedanceSpec(
        voltage_spec=td.AxisAlignedVoltageIntegralSpec(
            center=(0, 0, 0),
            size=(1, 0, 0),
            extrapolate_to_endpoints=True,
            snap_path_to_grid=True,
            sign="+",
        )
    )
    spec1 = td.CustomImpedanceSpec(
        voltage_spec=td.AxisAlignedVoltageIntegralSpec(
            center=(0.5, 0, 0),
            size=(0.5, 0, 0),
            extrapolate_to_endpoints=True,
            snap_path_to_grid=True,
            sign="+",
        )
    )
    port = TerminalWavePort(
        center=(0, 0, 0),
        size=(2, 2, 0),
        direction="+",
        name="twp_volt",
        terminal_specs=(spec0, spec1),
    )
    sim = td.Simulation(
        size=(4, 4, 4),
        center=(0, 0, 0),
        grid_spec=td.GridSpec.auto(wavelength=td.C_0 / 10e9, min_steps_per_wvl=10),
        boundary_spec=td.BoundarySpec.all_sides(boundary=td.PML()),
        structures=[],
        monitors=[],
        run_time=1e-9,
    )
    modeler = TerminalComponentModeler(simulation=sim, ports=[port], freqs=_FREQS)

    ax = modeler.plot_port("twp_volt")
    assert ax is not None
    plt.close()

    with pytest.raises(ValueError, match="Invalid port type"):
        modeler.plot_port(42)


# ---------------------------------------------------------------------------
# LumpedPort plot test
# ---------------------------------------------------------------------------


def test_plot_port_lumped_port():
    """Test plot_port for LumpedPort on microstrip."""
    sim = _make_microstrip_sim()
    port_center_z = _MS_METAL_THICKNESS + _MS_GAP / 2
    port = LumpedPort(
        center=[_MS_STRIP_LENGTH / 2, 0, port_center_z],
        size=[0, _MS_STRIP_WIDTH, _MS_GAP],
        voltage_axis=2,
        name="lp",
        impedance=50,
    )
    modeler = TerminalComponentModeler(simulation=sim, ports=[port], freqs=_FREQS)

    ax = modeler.plot_port("lp")
    assert ax is not None
    plt.close()


def test_plot_lumped_load_relaxed_for_small_plane_offset():
    """A snapped lumped load should remain plottable at the original plane within float precision."""
    z_expected = 0.3
    z_offset = float(increment_float(z_expected, 1.0))
    port = LumpedPort(
        center=(0, 0, z_expected),
        size=(2, 1, 0),
        voltage_axis=0,
        name="lp_offset",
        impedance=50,
    )
    load_geom = port.to_load(snap_center=z_offset).to_geometry()

    ax = load_geom.plot(z=z_expected)

    assert len(load_geom.intersections_plane(z=z_expected)) == 0
    assert len(ax.patches) > 0
    plt.close()


# ---------------------------------------------------------------------------
# CoaxialLumpedPort plot test
# ---------------------------------------------------------------------------


def test_plot_port_coaxial_lumped_port():
    """Test plot_port for CoaxialLumpedPort on coaxial cable."""
    sim = _make_coaxial_sim()
    port = CoaxialLumpedPort(
        center=[0, 0, -_COAX_LENGTH / 2],
        outer_diameter=2 * _COAX_ROUTER,
        inner_diameter=2 * _COAX_RINNER,
        normal_axis=2,
        direction="+",
        name="coax",
        impedance=50,
    )
    modeler = TerminalComponentModeler(simulation=sim, ports=[port], freqs=_FREQS)

    ax = modeler.plot_port("coax")
    assert ax is not None
    plt.close()


# ---------------------------------------------------------------------------
# WavePort plot tests
# ---------------------------------------------------------------------------


def test_plot_port_wave_port_auto_impedance():
    """Test plot_port for WavePort with AutoImpedanceSpec on stripline."""
    sim = _make_stripline_sim()
    port = WavePort(
        center=(0, 0, -_SL_L / 2),
        size=(50 * _MIL, _SL_H + 2 * _SL_T, 0),
        direction="+",
        name="wp_auto",
        mode_spec=td.MicrowaveModeSpec(num_modes=1),
        num_grid_cells=5,
    )
    modeler = TerminalComponentModeler(simulation=sim, ports=[port], freqs=_FREQS)

    ax = modeler.plot_port("wp_auto")
    assert ax is not None
    plt.close()


def test_plot_port_wave_port_custom_impedance():
    """Test plot_port for WavePort with CustomImpedanceSpec on microstrip."""
    sim = _make_microstrip_sim()
    port_center_z = _MS_METAL_THICKNESS + _MS_GAP / 2
    v_spec = td.AxisAlignedVoltageIntegralSpec(
        center=[-_MS_STRIP_LENGTH / 2, 0, port_center_z],
        size=[0, 0, _MS_GAP],
        sign="+",
    )
    i_spec = td.AxisAlignedCurrentIntegralSpec(
        center=[-_MS_STRIP_LENGTH / 2, 0, port_center_z],
        size=[0, 2 * _MS_STRIP_WIDTH, _MS_GAP + 2 * _MS_METAL_THICKNESS],
        sign="+",
    )
    mode_spec = td.MicrowaveModeSpec(
        num_modes=1,
        impedance_specs=td.CustomImpedanceSpec(voltage_spec=v_spec, current_spec=i_spec),
    )
    port = WavePort(
        center=[-_MS_STRIP_LENGTH / 2, 0, port_center_z],
        size=[0, 6 * _MS_STRIP_WIDTH, 2 * _MS_METAL_THICKNESS + _MS_GAP + 2e3],
        direction="+",
        name="wp_custom",
        mode_spec=mode_spec,
        num_grid_cells=5,
    )
    modeler = TerminalComponentModeler(simulation=sim, ports=[port], freqs=_FREQS)

    ax = modeler.plot_port("wp_custom")
    assert ax is not None
    plt.close()


def test_plot_port_wave_port_mixed_impedance():
    """Test plot_port for WavePort with mixed Custom + Auto impedance specs."""
    sim = _make_stripline_sim()
    left_cx = -(_SL_SE + _SL_W) / 2
    v_spec = td.AxisAlignedVoltageIntegralSpec(
        center=(left_cx, (-_SL_H / 2 + (-_SL_T / 2)) / 2, -_SL_L / 2),
        size=(0, (-_SL_T / 2) - (-_SL_H / 2), 0),
        sign="+",
    )
    i_spec = td.AxisAlignedCurrentIntegralSpec(
        center=(left_cx, 0, -_SL_L / 2),
        size=(2 * _SL_W, 0.5 * _SL_H, 0),
        sign="+",
    )
    mode_spec = td.MicrowaveModeSpec(
        num_modes=2,
        impedance_specs=(
            td.CustomImpedanceSpec(voltage_spec=v_spec, current_spec=i_spec),
            td.AutoImpedanceSpec(),
        ),
    )
    port = WavePort(
        center=(0, 0, -_SL_L / 2),
        size=(50 * _MIL, _SL_H + 2 * _SL_T, 0),
        direction="+",
        name="wp_mixed",
        mode_spec=mode_spec,
        num_grid_cells=5,
    )
    modeler = TerminalComponentModeler(simulation=sim, ports=[port], freqs=_FREQS)

    ax = modeler.plot_port("wp_mixed")
    assert ax is not None
    plt.close()
