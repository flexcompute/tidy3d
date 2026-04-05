"""Tests the simulation and its validators."""

from __future__ import annotations

import uuid

import gdstk
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.testing.compare import compare_images
from pydantic import ValidationError

import tidy3d as td
from tidy3d.components import scene, simulation
from tidy3d.components.medium import AnisotropicMediumFromMedium2D
from tidy3d.components.simulation import MAX_NUM_SOURCES
from tidy3d.exceptions import SetupError, Tidy3dError, Tidy3dKeyError
from tidy3d.plugins.mode import ModeSolver

from ..utils import (
    SIM_FULL,
    SIM_FULL_FIELD_PROJECTION,
    AssertLogLevel,
    AssertLogLevelHandler,
    AssertLogStr,
    cartesian_to_unstructured,
    run_emulated,
)

SIM = td.Simulation(size=(1, 1, 1), run_time=1e-12, grid_spec=td.GridSpec(wavelength=1.0))

RTOL = 0.01
TEST_MAX_NUM_MEDIUMS = 3


def test_sim_init():
    """make sure a simulation can be initialized"""

    sim = td.Simulation(
        size=(2.0, 2.0, 2.0),
        run_time=1e-12,
        structures=(
            td.Structure(
                geometry=td.Box(size=(1, 1, 1), center=(-1, 0, 0)),
                medium=td.Medium(permittivity=2.0),
            ),
            td.Structure(
                geometry=td.Box(size=(1, 1, 1), center=(0, 0, 0)),
                medium=td.Medium(permittivity=1.0, conductivity=3.0),
            ),
            td.Structure(
                geometry=td.Sphere(radius=1.4, center=(1.0, 0.0, 1.0)), medium=td.Medium()
            ),
            td.Structure(
                geometry=td.Cylinder(radius=1.4, length=2.0, center=(1.0, 0.0, -1.0), axis=1),
                medium=td.Medium(),
            ),
        ),
        sources=(
            td.UniformCurrentSource(
                size=(0, 0, 0),
                center=(0, -0.5, 0),
                polarization="Hx",
                source_time=td.GaussianPulse(
                    freq0=1e14,
                    fwidth=1e12,
                ),
                name="my_dipole",
                current_amplitude_definition="total",
            ),
            td.PointDipole(
                center=(0, 0, 0),
                polarization="Ex",
                source_time=td.GaussianPulse(
                    freq0=1e14,
                    fwidth=1e12,
                ),
            ),
        ),
        monitors=(
            td.FieldMonitor(size=(0, 0, 0), center=(0, 0, 0), freqs=[1e14, 2e14], name="point"),
            td.FluxTimeMonitor(size=(1, 1, 0), center=(0, 0, 0), interval=10, name="plane"),
        ),
        symmetry=(0, 1, -1),
        boundary_spec=td.BoundarySpec(
            x=td.Boundary.pml(num_layers=20),
            y=td.Boundary.stable_pml(num_layers=30),
            z=td.Boundary.absorber(num_layers=100),
        ),
        shutoff=1e-6,
        courant=0.8,
        subpixel=False,
    )

    _ = sim.dt
    _ = sim.tmesh
    sim.validate_pre_upload()
    m = sim.get_monitor_by_name("point")
    # will not work in 3.0
    _ = sim.mediums
    _ = sim.medium_map
    _ = sim.background_structure
    # will continue working in 3.0
    _ = sim.scene.mediums
    _ = sim.scene.medium_map
    _ = sim.scene.background_structure
    # sim.plot(x=0)
    # plt.close()
    # sim.plot_eps(x=0)
    # plt.close()
    _ = sim.num_pml_layers
    # sim.plot_grid(x=0)
    # plt.close()
    _ = sim.frequency_range
    _ = sim.grid
    _ = sim.num_cells
    sim.discretize(m)
    sim.epsilon(m)
    _ = sim.grid_info


def test_num_cells():
    """Test num_cells and num_computational_grid_points."""

    sim = td.Simulation(
        size=(1, 1, 1),
        run_time=1e-12,
        grid_spec=td.GridSpec.uniform(dl=0.1),
        sources=(
            td.PointDipole(
                center=(0, 0, 0),
                polarization="Ex",
                source_time=td.GaussianPulse(freq0=2e14, fwidth=1e14),
            ),
        ),
    )
    assert sim.num_computational_grid_points > sim.num_cells  # due to extra pixels at boundaries

    sim = sim.updated_copy(symmetry=(1, 0, 0))
    assert sim.num_computational_grid_points < sim.num_cells  # due to symmetry


def test_monitors_data_size():
    """make sure a simulation can be initialized"""

    sim = td.Simulation(
        size=(2.0, 2.0, 2.0),
        run_time=1e-12,
        structures=(
            td.Structure(
                geometry=td.Box(size=(1, 1, 1), center=(-1, 0, 0)),
                medium=td.Medium(permittivity=2.0),
            ),
            td.Structure(
                geometry=td.Box(size=(1, 1, 1), center=(0, 0, 0)),
                medium=td.Medium(permittivity=1.0, conductivity=3.0),
            ),
            td.Structure(
                geometry=td.Sphere(radius=1.4, center=(1.0, 0.0, 1.0)), medium=td.Medium()
            ),
            td.Structure(
                geometry=td.Cylinder(radius=1.4, length=2.0, center=(1.0, 0.0, -1.0), axis=1),
                medium=td.Medium(),
            ),
        ),
        sources=(
            td.UniformCurrentSource(
                size=(0, 0, 0),
                center=(0, -0.5, 0),
                polarization="Hx",
                source_time=td.GaussianPulse(
                    freq0=1e14,
                    fwidth=1e12,
                ),
                name="my_dipole",
                current_amplitude_definition="total",
            ),
            td.PointDipole(
                center=(0, 0, 0),
                polarization="Ex",
                source_time=td.GaussianPulse(
                    freq0=1e14,
                    fwidth=1e12,
                ),
            ),
        ),
        monitors=(
            td.FieldMonitor(size=(0, 0, 0), center=(0, 0, 0), freqs=[1e12, 2e12], name="point"),
            td.FluxTimeMonitor(size=(1, 1, 0), center=(0, 0, 0), interval=10, name="plane"),
        ),
        symmetry=(0, 1, -1),
        boundary_spec=td.BoundarySpec(
            x=td.Boundary.pml(num_layers=20),
            y=td.Boundary.stable_pml(num_layers=30),
            z=td.Boundary.absorber(num_layers=100),
        ),
        shutoff=1e-6,
        courant=0.8,
        subpixel=False,
    )

    datas = sim.monitors_data_size
    assert len(datas) == 2


def test_deprecation_defaults():
    """Make sure deprecation warnings NOT thrown if defaults used."""
    with AssertLogLevel(None):
        _ = td.Simulation(
            size=(1, 1, 1),
            run_time=1e-12,
            grid_spec=td.GridSpec.uniform(dl=0.1),
            sources=(
                td.PointDipole(
                    center=(0, 0, 0),
                    polarization="Ex",
                    source_time=td.GaussianPulse(freq0=2e14, fwidth=1e14),
                ),
            ),
        )


@pytest.mark.parametrize("shift_amount, log_level", ((1, None), (2, "WARNING")))
def test_sim_bounds(shift_amount, log_level):
    """make sure bounds are working correctly"""

    # make sure all things are shifted to this central location
    CENTER_SHIFT = (-1.0, 1.0, 100.0)

    def place_box(center_offset):
        shifted_center = tuple(c + s for (c, s) in zip(center_offset, CENTER_SHIFT))

        _ = td.Simulation(
            size=(1.5, 1.5, 1.5),
            center=CENTER_SHIFT,
            grid_spec=td.GridSpec(wavelength=1.0),
            run_time=1e-12,
            structures=(
                td.Structure(
                    geometry=td.Box(size=(1, 1, 1), center=shifted_center), medium=td.Medium()
                ),
            ),
            boundary_spec=td.BoundarySpec.all_sides(boundary=td.Periodic()),
            sources=(
                td.PointDipole(
                    center=CENTER_SHIFT,
                    polarization="Ex",
                    source_time=td.GaussianPulse(freq0=td.C_0, fwidth=td.C_0),
                ),
            ),
        )

    # create all permutations of squares being shifted 1, -1, or zero in all three directions
    bin_strings = [list(format(i, "03b")) for i in range(8)]
    bin_ints = [[int(b) for b in bin_string] for bin_string in bin_strings]
    bin_ints = np.array(bin_ints)
    bin_signs = 2 * (bin_ints - 0.5)

    # test all cases where box is shifted +/- 1 in x,y,z and still intersects
    for amp in bin_ints:
        for sign in bin_signs:
            center = shift_amount * amp * sign
            if np.sum(center) < 1e-12:
                continue
            if log_level is None:
                with AssertLogStr("WARNING", excludes_str="outside of the simulation domain"):
                    place_box(tuple(center))
            else:
                with AssertLogStr("WARNING", contains_str="outside of the simulation domain"):
                    place_box(tuple(center))


def test_sim_size():
    # note dl may need to change if we change the maximum allowed number of cells
    mesh1d = td.UniformGrid(dl=2e-4)
    grid_spec = td.GridSpec(grid_x=mesh1d, grid_y=mesh1d, grid_z=mesh1d)

    # check too many cells
    with pytest.raises(SetupError):
        s = td.Simulation(
            size=(1, 1, 1),
            grid_spec=grid_spec,
            run_time=1e-13,
        )
        s._validate_size()

    # should pass if symmetries applied
    s = td.Simulation(
        size=(1, 1, 1),
        grid_spec=grid_spec,
        run_time=1e-13,
        symmetry=(1, -1, 1),
    )
    s._validate_size()

    # check too many time steps
    with pytest.raises(ValidationError):
        s = td.Simulation(
            size=(1, 1, 1),
            run_time=1e-7,
        )
        s._validate_size()


def _test_monitor_size():
    with pytest.raises(SetupError):
        s = td.Simulation(
            size=(1, 1, 1),
            grid_spec=td.GridSpec.uniform(1e-3),
            monitors=(
                td.FieldMonitor(
                    size=(td.inf, td.inf, td.inf), freqs=np.linspace(0, 200e12, 10001), name="test"
                ),
            ),
            run_time=1e-12,
            boundary_spec=td.BoundarySpec.all_sides(boundary=td.Periodic()),
        )
        s.validate_pre_upload()


@pytest.mark.parametrize("freq, log_level", [(1.5, "WARNING"), (2.5, "INFO"), (3.5, "WARNING")])
def test_monitor_medium_frequency_range(freq, log_level):
    # monitor frequency above or below a given medium's range should throw a warning

    medium = td.Medium(frequency_range=(2e12, 3e12))
    box = td.Structure(geometry=td.Box(size=(0.1, 0.1, 0.1)), medium=medium)
    mnt = td.FieldMonitor(size=(0, 0, 0), name="freq", freqs=[freq * 1e12])
    src = td.UniformCurrentSource(
        source_time=td.GaussianPulse(freq0=2.5e12, fwidth=0.5e12),
        size=(0, 0, 0),
        polarization="Ex",
        current_amplitude_definition="total",
    )
    with AssertLogLevel(log_level):
        _ = td.Simulation(
            size=(1, 1, 1),
            structures=(box,),
            monitors=(mnt,),
            sources=(src,),
            run_time=1e-12,
            boundary_spec=td.BoundarySpec.all_sides(boundary=td.Periodic()),
        )


@pytest.mark.parametrize(
    "monitor_freq, log_level", [(5e10, "WARNING"), (2e12, "INFO"), (5e13, "WARNING")]
)
def test_monitor_simulation_frequency_range(monitor_freq, log_level):
    # monitor frequency outside of the simulation's frequency range should throw a warning

    src = td.UniformCurrentSource(
        source_time=td.GaussianPulse(freq0=2.0e12, fwidth=0.1e12),
        size=(0, 0, 0),
        polarization="Ex",
        current_amplitude_definition="total",
    )
    mnt = td.FieldMonitor(size=(0, 0, 0), name="freq", freqs=[monitor_freq])

    with AssertLogLevel(log_level):
        _ = td.Simulation(
            size=(1, 1, 1),
            monitors=(mnt,),
            sources=(src,),
            run_time=1e-12,
            boundary_spec=td.BoundarySpec.all_sides(boundary=td.Periodic()),
        )


def test_validate_monitor_simulation_frequency_range():
    # monitor frequency outside of the simulation's frequency range should throw an error

    src = td.UniformCurrentSource(
        source_time=td.GaussianPulse(freq0=2.0e12, fwidth=0.1e12),
        size=(0, 0, 0),
        polarization="Ex",
        current_amplitude_definition="total",
    )

    mnt = td.FieldMonitor(size=(0, 0, 0), name="freq", freqs=[2e12])
    s = td.Simulation(
        size=(1, 1, 1),
        monitors=(mnt,),
        sources=(src,),
        run_time=1e-12,
        boundary_spec=td.BoundarySpec.all_sides(boundary=td.Periodic()),
    )
    s._validate_freq_monitors_freq_range()

    with pytest.raises(SetupError):
        mnt = td.FieldMonitor(size=(0, 0, 0), name="freq", freqs=[5e10])
        s = td.Simulation(
            size=(1, 1, 1),
            monitors=(mnt,),
            sources=(src,),
            run_time=1e-12,
            boundary_spec=td.BoundarySpec.all_sides(boundary=td.Periodic()),
        )
        s._validate_freq_monitors_freq_range()

    with pytest.raises(SetupError):
        mnt = td.FieldMonitor(size=(0, 0, 0), name="freq", freqs=[5e13])
        s = td.Simulation(
            size=(1, 1, 1),
            monitors=(mnt,),
            sources=(src,),
            run_time=1e-12,
            boundary_spec=td.BoundarySpec.all_sides(boundary=td.Periodic()),
        )
        s._validate_freq_monitors_freq_range()


def test_validate_bloch_with_symmetry():
    with pytest.raises(ValidationError):
        td.Simulation(
            size=(1, 1, 1),
            run_time=1e-12,
            boundary_spec=td.BoundarySpec(
                x=td.Boundary.bloch(bloch_vec=1.0),
                y=td.Boundary.bloch(bloch_vec=1.0),
                z=td.Boundary.bloch(bloch_vec=1.0),
            ),
            symmetry=(1, 1, 1),
            grid_spec=td.GridSpec(wavelength=1.0),
        )


def test_validate_normalize_index():
    src = td.UniformCurrentSource(
        source_time=td.GaussianPulse(freq0=2.0e12, fwidth=1.0e12),
        size=(0, 0, 0),
        polarization="Ex",
        current_amplitude_definition="total",
    )

    # negative normalize index
    with pytest.raises(ValidationError):
        td.Simulation(
            size=(1, 1, 1),
            run_time=1e-12,
            grid_spec=td.GridSpec.uniform(dl=0.1),
            normalize_index=-1,
        )

    # normalize index out of bounds
    with pytest.raises(ValidationError):
        td.Simulation(
            size=(1, 1, 1),
            run_time=1e-12,
            grid_spec=td.GridSpec.uniform(dl=0.1),
            sources=(src,),
            normalize_index=1,
        )
    # skipped if no sources
    td.Simulation(
        size=(1, 1, 1), run_time=1e-12, grid_spec=td.GridSpec.uniform(dl=0.1), normalize_index=1
    )

    # normalize by zero-amplitude source
    with pytest.warns(
        RuntimeWarning,
        match=r"invalid value encountered in scalar divide",
    ):
        src0 = td.UniformCurrentSource(
            source_time=td.GaussianPulse(freq0=2.0e12, fwidth=1.0e12, amplitude=0),
            size=(0, 0, 0),
            polarization="Ex",
        )
        with pytest.raises(ValidationError):
            td.Simulation(
                size=(1, 1, 1),
                run_time=1e-12,
                grid_spec=td.GridSpec.uniform(dl=0.1),
                sources=(src0,),
            )


def test_simulation_validator_warning_order():
    src_cw = td.UniformCurrentSource(
        source_time=td.ContinuousWave(freq0=2.0e12, fwidth=0.5e12),
        size=(0, 0, 0),
        polarization="Ex",
    )
    src_pulse = td.UniformCurrentSource(
        source_time=td.GaussianPulse(freq0=3.0e12, fwidth=0.5e12),
        size=(0, 0, 0),
        polarization="Ex",
    )

    handler = AssertLogLevelHandler()
    td.log.handlers["validator_order"] = handler
    try:
        _ = td.Simulation(
            size=(1, 1, 1),
            run_time=1e-12,
            grid_spec=td.GridSpec.uniform(dl=0.01),
            sources=(src_cw, src_pulse),
            normalize_index=0,
            boundary_spec=td.BoundarySpec.all_sides(boundary=td.ABCBoundary()),
        )
    finally:
        del td.log.handlers["validator_order"]

    messages = [message for _level, message in handler.records]
    mode_abc_msg = (
        "At least one 'ModeABCBoundary' does not specify frequency at which the absorbed mode "
        "must be evaluated. The central frequency of the first source will be used."
    )
    normalize_msg = (
        "'normalize_index' 0 is a source with 'ContinuousWave' time dependence. Normalizing "
        "frequency-domain monitors by this source is not meaningful because field decay does "
        "not occur. Consider setting 'normalize_index' to 'None' instead."
    )

    assert mode_abc_msg in messages
    assert normalize_msg in messages
    assert messages.index(mode_abc_msg) < messages.index(normalize_msg)


def test_validate_plane_wave_boundaries():
    src1 = td.PlaneWave(
        source_time=td.GaussianPulse(freq0=2.5e14, fwidth=1e13),
        center=(0, 0, 0),
        size=(td.inf, td.inf, 0),
        direction="+",
        pol_angle=-1.0,
    )

    src2 = td.PlaneWave(
        source_time=td.GaussianPulse(freq0=2.5e14, fwidth=1e13),
        center=(0, 0, 0),
        size=(td.inf, td.inf, 0),
        direction="+",
        pol_angle=-1.0,
        angle_theta=np.pi / 4,
    )

    mnt = td.DiffractionMonitor(
        center=(0, 0, 0),
        size=(td.inf, td.inf, 0),
        freqs=[250e12, 300e12],
        name="monitor_diffraction",
        normal_dir="+",
    )

    bspec1 = td.BoundarySpec(
        x=td.Boundary.pml(),
        y=td.Boundary.absorber(),
        z=td.Boundary.stable_pml(),
    )

    bspec2 = td.BoundarySpec(
        x=td.Boundary.bloch_from_source(source=src2, domain_size=1, axis=0),
        y=td.Boundary.bloch_from_source(source=src2, domain_size=1, axis=1),
        z=td.Boundary.stable_pml(),
    )

    bspec3 = td.BoundarySpec(
        x=td.Boundary.bloch(bloch_vec=-3 + bspec2.x.plus.bloch_vec),
        y=td.Boundary.bloch(bloch_vec=2 + bspec2.y.plus.bloch_vec),
        z=td.Boundary.stable_pml(),
    )

    bspec4 = td.BoundarySpec(
        x=td.Boundary.bloch(bloch_vec=-3.1 + bspec2.x.plus.bloch_vec),
        y=td.Boundary.bloch(bloch_vec=1.8 + bspec2.y.plus.bloch_vec),
        z=td.Boundary.stable_pml(),
    )

    # normally incident plane wave with PMLs / absorbers is fine
    td.Simulation(
        size=(1, 1, 1),
        run_time=1e-12,
        sources=(src1,),
        boundary_spec=bspec1,
    )

    # angled incidence plane wave with PMLs / absorbers should error
    with pytest.raises(ValidationError):
        td.Simulation(
            size=(1, 1, 1),
            run_time=1e-12,
            sources=(src2,),
            boundary_spec=bspec1,
        )

    # angled incidence plane wave with periodic boundaries should warn
    with AssertLogLevel("WARNING", contains_str="incorrectly set"):
        td.Simulation(
            size=(1, 1, 1),
            run_time=1e-12,
            sources=(src2,),
            boundary_spec=td.BoundarySpec.all_sides(td.Periodic()),
        )

    # angled incidence plane wave with an integer-offset Bloch vector should warn
    with AssertLogLevel("WARNING", contains_str="integer reciprocal"):
        td.Simulation(
            size=(1, 1, 1),
            run_time=1e-12,
            sources=(src2,),
            boundary_spec=bspec3,
            monitors=(mnt,),
        )

    # angled incidence plane wave with wrong Bloch vector should warn
    with AssertLogLevel("WARNING", contains_str="incorrectly set"):
        td.Simulation(
            size=(1, 1, 1),
            run_time=1e-12,
            sources=(src2,),
            boundary_spec=bspec4,
        )


def test_validate_zero_dim_boundaries():
    # zero-dim simulation with an absorbing boundary in that direction should error
    src = td.PlaneWave(
        source_time=td.GaussianPulse(freq0=2.5e14, fwidth=1e13),
        center=(0, 0, 0),
        size=(td.inf, 0, td.inf),
        direction="+",
        pol_angle=0.0,
    )

    with AssertLogLevel("WARNING", contains_str="Periodic"):
        assert (
            td.Simulation(
                size=(1, 1, 0),
                run_time=1e-12,
                sources=[src],
                boundary_spec=td.BoundarySpec(
                    x=td.Boundary.periodic(),
                    y=td.Boundary.periodic(),
                    z=td.Boundary.pml(),
                ),
            ).boundary_spec.z
            == td.Boundary.periodic()
        )

    # zero-dim simulation with an absorbing boundary any other direction should not error
    td.Simulation(
        size=(1, 1, 0),
        run_time=1e-12,
        sources=(src,),
        boundary_spec=td.BoundarySpec(
            x=td.Boundary.pml(),
            y=td.Boundary.stable_pml(),
            z=td.Boundary.periodic(),
        ),
    )


def test_validate_symmetry_boundaries():
    # simulation with symmetry along an axis should have the same boundaries defined on both sides
    td.Simulation(
        size=(1, 1, 1),
        symmetry=(1, 1, 1),
        grid_spec=td.GridSpec.uniform(dl=0.1),
        run_time=1e-12,
        boundary_spec=td.BoundarySpec(
            x=td.Boundary.periodic(),
            y=td.Boundary(
                # Now give the plus and minus boundaries different names to confirm it does not matter.
                plus=td.PML(name="b1"),
                minus=td.PML(name="b2"),
            ),
            z=td.Boundary.pml(),
        ),
    )
    with pytest.raises(ValidationError, match="Symmetry"):
        td.Simulation(
            size=(1, 1, 1),
            symmetry=(1, 1, 1),
            grid_spec=td.GridSpec.uniform(dl=0.1),
            run_time=1e-12,
            boundary_spec=td.BoundarySpec(
                x=td.Boundary.periodic(),
                y=td.Boundary(plus=td.PML(num_layers=10), minus=td.PML(num_layers=20)),
                z=td.Boundary.pml(),
            ),
        )


def test_validate_components_none():
    assert type(SIM)._validate_num_sources(val=None) is None
    assert SIM._structures_not_at_edges() is SIM
    assert SIM._warn_monitor_mediums_frequency_range() is SIM
    assert SIM._warn_monitor_simulation_frequency_range() is SIM
    assert SIM._warn_grid_size_too_small() is SIM
    assert SIM._source_homogeneous_isotropic() is SIM


def test_sources_edge_case_validation():
    values = SIM.model_dump()
    values.pop("sources")
    td.Simulation.model_validate(values)


def test_validate_size_run_time(monkeypatch):
    monkeypatch.setattr(simulation, "MAX_TIME_STEPS", 1)
    with pytest.raises(SetupError):
        s = SIM.copy(update={"run_time": 1e-12})
        s._validate_size()


def test_validate_size_spatial_and_time(monkeypatch):
    monkeypatch.setattr(simulation, "MAX_CELLS_TIMES_STEPS", 1)
    with pytest.raises(SetupError):
        s = SIM.copy(update={"run_time": 1e-12})
        s._validate_size()


def test_validate_mnt_size(monkeypatch):
    # warning for monitor size
    monkeypatch.setattr(simulation, "WARN_MONITOR_DATA_SIZE_GB", 1 / 2**30)
    s = SIM.copy(update={"monitors": (td.FieldMonitor(name="f", freqs=[1e12], size=(1, 1, 1)),)})
    with AssertLogLevel("WARNING"):
        s._validate_monitor_size()

    # error for simulation size
    monkeypatch.setattr(simulation, "MAX_SIMULATION_DATA_SIZE_GB", 1 / 2**30)
    with pytest.raises(SetupError):
        s = SIM.copy(
            update={"monitors": (td.FieldMonitor(name="f", freqs=[1e12], size=(1, 1, 1)),)}
        )
        s._validate_monitor_size()


# def test_max_geometry_validation():
#     gs = td.GridSpec(wavelength=1.0)
#     too_many = [td.Box(size=(1, 1, 1)) for _ in range(MAX_GEOMETRY_COUNT + 1)]

#     fine = [
#         td.Structure(
#             geometry=td.ClipOperation(
#                 operation="union",
#                 geometry_a=td.Box(size=(1, 1, 1)),
#                 geometry_b=td.GeometryGroup(geometries=too_many),
#             ),
#             medium=td.Medium(permittivity=2.0),
#         ),
#         td.Structure(
#             geometry=td.GeometryGroup(geometries=too_many),
#             medium=td.Medium(permittivity=2.0),
#         ),
#     ]
#     _ = td.Simulation(size=(1, 1, 1), run_time=1, grid_spec=gs, structures=fine)

#     not_fine = [
#         td.Structure(
#             geometry=td.ClipOperation(
#                 operation="difference",
#                 geometry_a=td.Box(size=(1, 1, 1)),
#                 geometry_b=td.GeometryGroup(geometries=too_many),
#             ),
#             medium=td.Medium(permittivity=2.0),
#         ),
#     ]
#     with pytest.raises(ValidationError, match=f" {MAX_GEOMETRY_COUNT + 2} "):
#         _ = td.Simulation(size=(1, 1, 1), run_time=1, grid_spec=gs, structures=not_fine)


def test_no_monitor():
    with pytest.raises(Tidy3dKeyError):
        SIM.get_monitor_by_name("NOPE")


def test_wvl_mat_min_error():
    """Make sure we get an error when there are no sources in the simulation but
    we ask for the minimum wavelength in material."""

    with pytest.raises(Tidy3dError):
        SIM.wvl_mat_min()


def test_plot_structure():
    _ = SIM_FULL.structures[0].plot(x=0)
    plt.close()


def test_plot_eps():
    _ = SIM_FULL.plot_eps(x=0)
    plt.close()


def test_plot_eps_bounds():
    _ = SIM_FULL.plot_eps(x=0, hlim=[-0.45, 0.45])
    plt.close()
    _ = SIM_FULL.plot_eps(x=0, vlim=[-0.45, 0.45])
    plt.close()
    _ = SIM_FULL.plot_eps(x=0, hlim=[-0.45, 0.45], vlim=[-0.45, 0.45])
    plt.close()


class TestAnisotropicPlotting:
    """Tests for plotting anisotropic media"""

    diag_comps = ["xx", "yy", "zz"]
    offdiag_comps = ["xy", "yx", "xz", "zx", "yz", "zy"]
    allcomps = diag_comps + offdiag_comps

    medium_diag = td.AnisotropicMedium(
        xx=td.Medium(permittivity=5), yy=td.Medium(permittivity=10), zz=td.Medium(permittivity=15)
    )

    medium_fullyani = td.FullyAnisotropicMedium(permittivity=[[6, 2, 3], [2, 7, 4], [3, 4, 8]])

    @pytest.fixture(scope="class")
    def medium_customani(self):
        """based this custom medium on
        https://docs.flexcompute.com/projects/tidy3d/en/latest/api/_autosummary/tidy3d.CustomAnisotropicMedium.html
        """
        Nx, Ny, Nz = 100, 100, 100
        x = np.linspace(-1, 1, Nx)
        y = np.linspace(-1, 1, Ny)
        z = np.linspace(-1, 1, Nz)
        coords = {"x": x, "y": y, "z": z}
        permittivity = td.SpatialDataArray(2 * np.ones((Nx, Ny, Nz)), coords=coords)
        conductivity = td.SpatialDataArray(np.ones((Nx, Ny, Nz)), coords=coords)
        medium_xx = td.CustomMedium(permittivity=permittivity, conductivity=conductivity)
        medium_yy = td.CustomMedium(permittivity=2 * permittivity, conductivity=conductivity)

        # make the zz component a spatially varying medium
        # define coordinate array
        x_mesh, y_mesh, _ = np.meshgrid(x, y, z, indexing="ij")
        r_mesh = np.sqrt(x_mesh**2 + y_mesh**2)  # radial distance

        # index of refraction array
        # assign the refractive index value to the array according to the desired profile
        n_data = np.ones((Nx, Ny, Nz))
        n0 = 2
        A = 0.5
        r = 1
        n_data[r_mesh <= r] = n0 * (1 - A * r_mesh[r_mesh <= r] ** 2)
        # convert to dataset array
        n_dataset = td.SpatialDataArray(n_data, coords={"x": x, "y": y, "z": z})
        medium_zz = td.CustomMedium.from_nk(n_dataset, interp_method="nearest")

        return td.CustomAnisotropicMedium(xx=medium_xx, yy=medium_yy, zz=medium_zz)

    def make_sim(self, medium):
        L = 5

        source = td.UniformCurrentSource(
            center=(0, 0, -L / 3),
            size=(L, L / 2, 0),
            polarization="Ex",
            source_time=td.GaussianPulse(
                freq0=td.C_0,
                fwidth=10e14,
            ),
            current_amplitude_definition="total",
        )
        structures = (td.Structure(geometry=td.Sphere(center=(0, 0, 0), radius=1), medium=medium),)

        return td.Simulation(
            size=(L, L, L),
            grid_spec=td.GridSpec.uniform(dl=0.01),
            structures=structures,
            sources=(source,),
            run_time=1e-12,
        )

    def compare_eps_images(self, tmp_path, eps_comp1, eps_comp2, expected, medium):
        """Asserts that two epsilon component plots are different"""
        sim = self.make_sim(medium)

        # plot and save epsilon component 1
        fname1 = tmp_path / (str(uuid.uuid4()) + ".png")
        f1, ax1 = plt.subplots()
        sim.plot_eps(x=0, eps_component=eps_comp1, ax=ax1)
        f1.savefig(fname1)

        # plot and save epsilon component 2
        fname2 = tmp_path / (str(uuid.uuid4()) + ".png")
        f2, ax2 = plt.subplots()
        sim.plot_eps(x=0, eps_component=eps_comp2, ax=ax2)
        f2.savefig(fname2)

        # compare_images only returns None if the two images are the same
        assert (compare_images(fname1, fname2, tol=0.001) is None) == expected

    @pytest.mark.parametrize("eps_comp", ("xyz", "123", "", 5))
    def test_bad_eps_arg(self, eps_comp):
        """Tests that an incorrect component raises the proper exception."""
        with pytest.raises(ValueError, match=f"eps_component '{eps_comp}' is not supported. "):
            self.make_sim(self.medium_diag).plot_eps(x=0, eps_component=eps_comp)

    @pytest.mark.parametrize(
        "eps_comp",
        [None, *diag_comps],
    )
    def test_plot_anisotropic_medium(self, eps_comp):
        """Test plotting diagonal components of a diagonally anisotropic medium succeeds or not.
        diagonal components and ``None`` should succeed.
        """
        self.make_sim(self.medium_diag).plot_eps(x=0, eps_component=eps_comp)

    @pytest.mark.parametrize("eps_comp", offdiag_comps)
    def test_plot_anisotropic_medium_offdiagfail(self, eps_comp):
        """Tests that plotting off-diagonal components of a diagonally anisotropic medium raises an exception."""
        with pytest.raises(
            ValueError,
            match=f"Plotting component '{eps_comp}' of a diagonally-anisotropic permittivity tensor is not supported",
        ):
            self.make_sim(self.medium_diag).plot_eps(x=0, eps_component=eps_comp)

    @pytest.mark.parametrize(
        "eps_comp1,eps_comp2,expected",
        (
            pytest.param("xx", "yy", False),
            pytest.param("xx", "zz", False),
            pytest.param("yy", "zz", False),
        ),
    )
    def test_plot_anisotropic_medium_diff(self, tmp_path, eps_comp1, eps_comp2, expected):
        """Tests that the plots of different components of an AnisotropicMedium are actually different."""
        self.compare_eps_images(tmp_path, eps_comp1, eps_comp2, expected, self.medium_diag)

    @pytest.mark.parametrize(
        "eps_comp",
        [None, *diag_comps, *offdiag_comps],
    )
    def test_plot_fully_anisotropic_medium(self, eps_comp):
        """Test plotting all components of a fully anisotropic medium.
        All plots should succeed.
        """
        sim = self.make_sim(self.medium_fullyani)
        sim.plot_eps(x=0, eps_component=eps_comp)

    # Test parameters for comparing plots of a FullyAnisotropicMedium
    fullyani_testplot_diff_params = []
    for eps_comp1 in allcomps:
        for eps_comp2 in allcomps:
            if eps_comp1 == eps_comp2 or eps_comp1[::-1] == eps_comp2:
                # Same components, or transposed components (eg. xy and yx) should plot the same
                fullyani_testplot_diff_params.append((eps_comp1, eps_comp2, True))
            else:
                # All other component pairs should plot differently
                fullyani_testplot_diff_params.append(pytest.param(eps_comp1, eps_comp2, False))

    @pytest.mark.parametrize("eps_comp1,eps_comp2,expected", fullyani_testplot_diff_params)
    def test_plot_fully_anisotropic_medium_diff(self, tmp_path, eps_comp1, eps_comp2, expected):
        """Tests that the plots of different components of a FullyAnisotropicMedium are actually different."""
        self.compare_eps_images(tmp_path, eps_comp1, eps_comp2, expected, self.medium_fullyani)

    @pytest.mark.parametrize(
        "eps_comp",
        [None, *diag_comps],
    )
    def test_plot_customanisotropic_medium(self, eps_comp, medium_customani):
        """Test plotting diagonal components of a diagonally anisotropic custom medium.
        diagonal components and ``None`` should succeed.
        """
        self.make_sim(medium_customani).plot_eps(x=0, eps_component=eps_comp)

    @pytest.mark.parametrize("eps_comp", offdiag_comps)
    def test_plot_customanisotropic_medium_offdiagfail(self, eps_comp, medium_customani):
        """Tests that plotting off-diagonal components of a diagonally anisotropic custom medium raises an exception."""
        with pytest.raises(
            ValueError,
            match=f"Plotting component '{eps_comp}' of a diagonally-anisotropic permittivity tensor is not supported.",
        ):
            self.make_sim(medium_customani).plot_eps(x=0, eps_component=eps_comp)

    @pytest.mark.parametrize(
        "eps_comp1,eps_comp2,expected",
        (
            pytest.param("xx", "yy", False),
            pytest.param("xx", "zz", False),
            pytest.param("yy", "zz", False),
        ),
    )
    def test_plot_customanisotropic_medium_diff(
        self, tmp_path, eps_comp1, eps_comp2, expected, medium_customani
    ):
        """Tests that the plots of different components of a CustomAnisotropicMedium are actually different."""
        self.compare_eps_images(tmp_path, eps_comp1, eps_comp2, expected, medium_customani)


def test_plot():
    SIM_FULL.plot(x=0)
    plt.close()


def test_plot_with_units():
    sim_with_units = SIM_FULL.updated_copy(plot_length_units="nm")
    sim_with_units.plot(x=-0.5)


def test_plot_1d_sim():
    mesh1d = td.UniformGrid(dl=2e-4)
    grid_spec = td.GridSpec(grid_x=mesh1d, grid_y=mesh1d, grid_z=mesh1d)
    s = td.Simulation(
        size=(0, 0, 1),
        grid_spec=grid_spec,
        run_time=1e-13,
        boundary_spec=td.BoundarySpec.all_sides(boundary=td.Periodic()),
    )
    _ = s.plot(y=0)
    plt.close()


def test_plot_bounds():
    _ = SIM_FULL.plot(x=0, hlim=[-0.45, 0.45])
    plt.close()
    _ = SIM_FULL.plot(x=0, vlim=[-0.45, 0.45])
    plt.close()
    _ = SIM_FULL.plot(x=0, hlim=[-0.45, 0.45], vlim=[-0.45, 0.45])
    plt.close()


def test_plot_3d():
    SIM_FULL.plot_3d()
    plt.close()


def test_structure_alpha():
    _ = SIM_FULL.plot_structures_eps(x=0, alpha=None)
    plt.close()
    _ = SIM_FULL.plot_structures_eps(x=0, alpha=-1)
    plt.close()
    _ = SIM_FULL.plot_structures_eps(x=0, alpha=1)
    plt.close()
    _ = SIM_FULL.plot_structures_eps(x=0, alpha=0.5)
    plt.close()
    _ = SIM_FULL.plot_structures_eps(x=0, alpha=0.5, cbar=True)
    plt.close()
    new_structs = [
        td.Structure(geometry=s.geometry, medium=SIM_FULL.medium) for s in SIM_FULL.structures
    ]
    S2 = SIM_FULL.copy(update={"structures": tuple(new_structs)})
    _ = S2.plot_structures_eps(x=0, alpha=0.5)
    plt.close()


def test_plot_eps_with_default_frequency():
    """Make sure that when possible the permittivity is plotted using
    central frequency of the first source added to the simulation.
    """
    freq0 = 2e14
    src = td.PointDipole(polarization="Ex", source_time=td.GaussianPulse(freq0=freq0, fwidth=1e11))
    chromium = td.material_library["Cr"]["RakicLorentzDrude1998"]
    box = td.Structure(medium=chromium, geometry=td.Box(size=(0.2, 0.2, 0.2), center=(0, 0, 0)))
    sim = td.Simulation(
        size=(1, 1, 1),
        structures=(box,),
        sources=(src,),
        run_time=1e-12,
        boundary_spec=td.BoundarySpec.all_sides(boundary=td.PECBoundary()),
        grid_spec=td.GridSpec.uniform(dl=0.01),
    )
    # Source frequency is in range, so no warning
    with AssertLogLevel(None):
        _ = sim.plot_structures_eps(x=0)
    plt.close()

    freq0 = 20e14
    sim = sim.updated_copy(path="sources/0/source_time", freq0=freq0)
    # Source frequency is out of range, so give warning
    with AssertLogLevel("WARNING"):
        _ = sim.plot_structures_eps(x=0)
    plt.close()

    src2 = td.PointDipole(polarization="Ex", source_time=td.GaussianPulse(freq0=3e14, fwidth=1e11))
    sim = sim.updated_copy(sources=(src, src2))
    # Source frequencies do not agree, so give warning about evaluating at infinite frequency is out of range.
    with AssertLogLevel("WARNING"):
        _ = sim.plot_structures_eps(x=0)
    plt.close()


def test_plot_symmetries():
    S2 = SIM.copy(update={"symmetry": (1, 0, -1)})
    S2.plot_symmetries(x=0)
    plt.close()


def test_plot_grid():
    override = td.Structure(geometry=td.Box(size=(1, 1, 1)), medium=td.Medium())
    S2 = SIM_FULL.copy(
        update={"grid_spec": td.GridSpec(wavelength=1.0, override_structures=(override,))}
    )
    S2.plot_grid(x=0)
    plt.close()


def test_plot_boundaries():
    bound_spec = td.BoundarySpec(
        x=td.Boundary(plus=td.PECBoundary(), minus=td.PMCBoundary()),
        y=td.Boundary(
            plus=td.BlochBoundary(bloch_vec=1.0),
            minus=td.BlochBoundary(bloch_vec=1.0),
        ),
        z=td.Boundary(plus=td.Periodic(), minus=td.Periodic()),
    )
    S2 = SIM_FULL.copy(update={"boundary_spec": bound_spec})
    S2.plot_boundaries(z=0)
    plt.close()


def test_plot_with_lumped_elements():
    load = td.LumpedResistor(
        center=(0, 0, 0), size=(1, 2, 0), name="resistor", voltage_axis=0, resistance=50
    )
    sim_test = SIM_FULL.updated_copy(lumped_elements=(load,))
    sim_test.plot(z=0)
    plt.close()


def test_wvl_mat_grid():
    td.Simulation.wvl_mat_min.fget(SIM_FULL)


def test_complex_fields():
    assert not SIM.complex_fields
    bound_spec = td.BoundarySpec(
        x=td.Boundary(plus=td.PECBoundary(), minus=td.PMCBoundary()),
        y=td.Boundary(
            plus=td.BlochBoundary(bloch_vec=1.0),
            minus=td.BlochBoundary(bloch_vec=1.0),
        ),
        z=td.Boundary(plus=td.Periodic(), minus=td.Periodic()),
    )
    S2 = SIM_FULL.copy(update={"boundary_spec": bound_spec})
    assert S2.complex_fields


def test_nyquist():
    S = SIM.copy(
        update={
            "sources": (
                td.PointDipole(
                    polarization="Ex", source_time=td.GaussianPulse(freq0=2e14, fwidth=1e11)
                ),
            ),
        }
    )
    assert S.nyquist_step > 1

    # nyquist step decreses to 1 when the frequency-domain monitor is at high frequency
    S_MONITOR = S.copy(
        update={"monitors": (td.FluxMonitor(size=(1, 1, 0), freqs=[1e14, 1e20], name="flux"),)}
    )
    assert S_MONITOR.nyquist_step == 1

    # fake a scenario where the fmax of the simulation is negative?
    class MockSim:
        frequency_range = (-2, -1)
        monitors = ()
        _cached_properties = {}
        _fixed_angle_sources = ()

    m = MockSim()
    assert td.Simulation.nyquist_step.fget(m) == 1


def test_discretize_non_intersect():
    with AssertLogLevel("ERROR"):
        SIM.discretize(box=td.Box(center=(-20, -20, -20), size=(1, 1, 1)))


def test_warn_sim_background_medium_freq_range():
    with AssertLogLevel("WARNING"):
        _ = SIM.copy(
            update={
                "sources": (
                    td.PointDipole(
                        polarization="Ex", source_time=td.GaussianPulse(freq0=2e14, fwidth=1e11)
                    ),
                ),
                "monitors": (td.FluxMonitor(name="test", freqs=[2e12], size=(1, 1, 0)),),
                "medium": td.Medium(frequency_range=(0, 1e12)),
            }
        )


@pytest.mark.parametrize("grid_size,log_level", [(0.001, None), (3, "WARNING")])
def test_large_grid_size(grid_size, log_level):
    # small fwidth should be inside range, large one should throw warning

    medium = td.Medium(permittivity=2, frequency_range=(2e14, 3e14))
    box = td.Structure(geometry=td.Box(size=(0.1, 0.1, 0.1)), medium=medium)
    src = td.PointDipole(
        source_time=td.GaussianPulse(freq0=2.5e14, fwidth=1e12),
        polarization="Ex",
    )

    with AssertLogLevel(log_level):
        _ = td.Simulation(
            size=(1, 1, 1),
            grid_spec=td.GridSpec.uniform(dl=grid_size),
            structures=(box,),
            sources=(src,),
            run_time=1e-12,
            boundary_spec=td.BoundarySpec.all_sides(boundary=td.Periodic()),
        )


@pytest.mark.parametrize("box_size,log_level", [(0.1, "INFO"), (9.9, "WARNING"), (20, "INFO")])
def test_sim_structure_gap(box_size, log_level):
    """Make sure the gap between a structure and PML is not too small compared to lambda0."""
    medium = td.Medium(permittivity=2)
    box = td.Structure(geometry=td.Box(size=(box_size, box_size, box_size)), medium=medium)
    src = td.UniformCurrentSource(
        source_time=td.GaussianPulse(freq0=3e14, fwidth=1e13),
        size=(0, 0, 0),
        polarization="Ex",
        current_amplitude_definition="total",
    )

    with AssertLogLevel(log_level):
        _ = td.Simulation(
            size=(10, 10, 10),
            structures=(box,),
            sources=(src,),
            boundary_spec=td.BoundarySpec(
                x=td.Boundary.pml(num_layers=6),
                y=td.Boundary.pml(num_layers=6),
                z=td.Boundary.pml(num_layers=6),
            ),
            run_time=1e-12,
        )


def test_sim_plane_wave_error():
    """ "Make sure we error if plane wave is not intersecting homogeneous region of simulation."""

    medium_bg = td.Medium(permittivity=2)
    medium_air = td.Medium(permittivity=1)
    medium_bg_diag = td.AnisotropicMedium(
        xx=td.Medium(permittivity=1),
        yy=td.Medium(permittivity=2),
        zz=td.Medium(permittivity=3),
    )
    medium_bg_full = td.FullyAnisotropicMedium(permittivity=[[4, 0.1, 0], [0.1, 2, 0], [0, 0, 3]])

    box = td.Structure(geometry=td.Box(size=(0.1, 0.1, 0.1)), medium=medium_air)

    box_transparent = td.Structure(geometry=td.Box(size=(0.1, 0.1, 0.1)), medium=medium_bg)

    src = td.PlaneWave(
        source_time=td.GaussianPulse(freq0=2.5e14, fwidth=1e13),
        center=(0, 0, 0),
        size=(td.inf, td.inf, 0),
        direction="+",
        pol_angle=-1.0,
    )

    # with transparent box continue
    _ = td.Simulation(
        size=(1, 1, 1),
        medium=medium_bg,
        structures=(box_transparent,),
        sources=(src,),
        run_time=1e-12,
        boundary_spec=td.BoundarySpec.all_sides(boundary=td.Periodic()),
    )

    # with non-transparent box, raise
    with pytest.raises(ValidationError):
        _ = td.Simulation(
            size=(1, 1, 1),
            medium=medium_bg,
            structures=(box_transparent, box),
            sources=(src),
            boundary_spec=td.BoundarySpec.all_sides(boundary=td.Periodic()),
        )

    # raise with anisotropic medium
    with pytest.raises(ValidationError):
        _ = td.Simulation(
            size=(1, 1, 1),
            medium=medium_bg_diag,
            sources=(src,),
            boundary_spec=td.BoundarySpec.all_sides(boundary=td.Periodic()),
        )

    with pytest.raises(ValidationError):
        _ = td.Simulation(
            size=(1, 1, 1),
            medium=medium_bg_full,
            sources=(src,),
            boundary_spec=td.BoundarySpec.all_sides(boundary=td.Periodic()),
        )


def test_sim_monitor_homogeneous():
    """Make sure we error if a field projection monitor is not intersecting a
    homogeneous region of the simulation.
    """

    medium_bg = td.Medium(permittivity=2)
    medium_air = td.Medium(permittivity=1)

    box = td.Structure(geometry=td.Box(size=(0.2, 0.1, 0.1)), medium=medium_air)

    box_transparent = td.Structure(geometry=td.Box(size=(0.2, 0.1, 0.1)), medium=medium_bg)

    _ = td.FieldProjectionAngleMonitor(
        center=(0, 0, 0),
        size=(td.inf, td.inf, 0),
        freqs=[250e12, 300e12],
        name="monitor_n2f",
        theta=[0],
        phi=[0],
    )

    monitor_n2f_vol = td.FieldProjectionAngleMonitor(
        center=(0.1, 0, 0),
        size=(0.04, 0.04, 0.04),
        freqs=[250e12, 300e12],
        name="monitor_n2f_vol",
        theta=[0],
        phi=[0],
    )

    _ = td.DiffractionMonitor(
        center=(0, 0, 0),
        size=(td.inf, td.inf, 0),
        freqs=[250e12, 300e12],
        name="monitor_diffraction",
        normal_dir="+",
    )

    src = td.PointDipole(
        source_time=td.GaussianPulse(freq0=2.5e14, fwidth=1e13),
        center=(0, 0, 0),
        polarization="Ex",
    )

    for monitor in [monitor_n2f_vol]:
        # with transparent box continue
        _ = td.Simulation(
            size=(1, 1, 1),
            medium=medium_bg,
            structures=(box_transparent,),
            sources=(src,),
            run_time=1e-12,
            monitors=(monitor,),
            boundary_spec=td.BoundarySpec.all_sides(boundary=td.PML()),
        )

        # with non-transparent box, raise
        with pytest.raises(ValidationError):
            _ = td.Simulation(
                size=(1, 1, 1),
                medium=medium_bg,
                structures=(box,),
                sources=(src,),
                monitors=(monitor,),
                run_time=1e-12,
                boundary_spec=td.BoundarySpec.all_sides(boundary=td.PML()),
            )

    # will be removed in 3.0
    mediums = td.Simulation.intersecting_media(monitor_n2f_vol, [box])
    assert len(mediums) == 1
    mediums = td.Simulation.intersecting_media(monitor_n2f_vol, [box_transparent])
    assert len(mediums) == 1

    # continue in 3.0
    mediums = td.Scene.intersecting_media(monitor_n2f_vol, [box])
    assert len(mediums) == 1
    mediums = td.Scene.intersecting_media(monitor_n2f_vol, [box_transparent])
    assert len(mediums) == 1

    # when another medium intersects an excluded surface, no errors should be raised
    monitor_n2f_vol_exclude = td.FieldProjectionAngleMonitor(
        center=(0.2, 0, 0.2),
        size=(0.4, 0.4, 0.4),
        freqs=[250e12, 300e12],
        name="monitor_n2f_vol",
        theta=[0],
        phi=[0],
        exclude_surfaces=["x-", "z-"],
    )

    _ = td.Simulation(
        size=(1, 1, 1),
        medium=medium_bg,
        structures=(box_transparent, box),
        sources=(src,),
        monitors=(monitor_n2f_vol_exclude,),
        run_time=1e-12,
        boundary_spec=td.BoundarySpec.all_sides(boundary=td.PML()),
    )

    # structures outside the simulation domain should not affect the homogeneity check
    box_outside_sim = td.Structure(
        geometry=td.Box(center=(0, 0, 1.5), size=(0.2, 0.2, 2.0)),
        medium=medium_air,
    )

    monitor_n2f_outside_sim = td.FieldProjectionAngleMonitor(
        center=(0, 0, 0.5),
        size=(0.1, 0.1, 1.0),
        freqs=[250e12, 300e12],
        name="monitor_n2f_outside_sim",
        theta=[np.pi / 2],
        phi=[0],
    )

    with AssertLogStr("WARNING", contains_str="outside of the simulation domain"):
        _ = td.Simulation(
            size=(1, 1, 0),
            medium=medium_bg,
            structures=(box_outside_sim,),
            sources=(src,),
            monitors=(monitor_n2f_outside_sim,),
            run_time=1e-12,
            boundary_spec=td.BoundarySpec.all_sides(boundary=td.Periodic()),
        )

    # in 2D, the in-domain line traces of a 3D projection monitor should still be checked
    box_in_2d_sim = td.Structure(
        geometry=td.Box(center=(0.075, 0, 0), size=(0.1, 0.2, 2.0)),
        medium=medium_air,
    )

    with pytest.raises(ValidationError):
        _ = td.Simulation(
            size=(1, 1, 0),
            medium=medium_bg,
            structures=(box_in_2d_sim,),
            sources=(src,),
            monitors=(monitor_n2f_outside_sim,),
            run_time=1e-12,
            boundary_spec=td.BoundarySpec.all_sides(boundary=td.Periodic()),
        )

    # mixed planar and zero-measure clipped projection surfaces should still be accepted
    monitor_n2f_edge_touch = td.FieldProjectionAngleMonitor(
        center=(0.75, 1.0, 0.0),
        size=(1, 1, 1),
        freqs=[250e12, 300e12],
        name="monitor_n2f_edge_touch",
        theta=[0],
        phi=[0],
    )

    mediums = td.Simulation._projection_monitor_mediums_in_bounds(
        center=(0, 0, 0),
        size=(1, 1, 1),
        monitor=monitor_n2f_edge_touch,
        structures=[
            td.Structure(
                geometry=td.Box(center=(0, 0, 0), size=(1, 1, 1)),
                medium=medium_bg,
            )
        ],
    )
    assert mediums == {medium_bg}

    # purely zero-measure clipped projection surfaces should error explicitly
    monitor_n2f_corner_touch = td.FieldProjectionAngleMonitor(
        center=(1.0, 1.0, 1.0),
        size=(1, 1, 1),
        freqs=[250e12, 300e12],
        name="monitor_n2f_corner_touch",
        theta=[0],
        phi=[0],
    )

    with pytest.raises(SetupError, match="zero-measure sets"):
        _ = td.Simulation._projection_monitor_mediums_in_bounds(
            center=(0, 0, 0),
            size=(1, 1, 1),
            monitor=monitor_n2f_corner_touch,
            structures=[
                td.Structure(
                    geometry=td.Box(center=(0, 0, 0), size=(1, 1, 1)),
                    medium=medium_bg,
                )
            ],
        )


def test_proj_monitor_periodic_bloch_boundaries_3d():
    """Make sure 3D field projection monitors error with periodic or Bloch boundaries."""

    monitor_n2f = td.FieldProjectionAngleMonitor(
        center=(0, 0, 0),
        size=(2, 2, 0),
        freqs=[2.5e14],
        name="monitor_n2f",
        theta=[0],
        phi=[0],
        proj_distance=1e5,
    )
    src = td.PointDipole(
        center=(0, 0, 0),
        polarization="Ex",
        source_time=td.GaussianPulse(freq0=1e14, fwidth=1e12),
    )

    for boundary_spec in (
        td.BoundarySpec.all_sides(boundary=td.Periodic()),
        td.BoundarySpec(
            x=td.Boundary.bloch(bloch_vec=0.2),
            y=td.Boundary.pml(),
            z=td.Boundary.pml(),
        ),
    ):
        with pytest.raises(ValidationError, match="periodic/Bloch boundaries"):
            _ = td.Simulation(
                size=(2.2, 2.2, 2),
                structures=(),
                sources=(src,),
                run_time=1e-12,
                boundary_spec=boundary_spec,
                monitors=(monitor_n2f,),
            )


def test_proj_monitor_distance():
    """Make sure a warning is issued if the projection distance for exact projections
    is very large compared to the simulation domain size.
    """

    monitor_n2f = td.FieldProjectionAngleMonitor(
        center=(0, 0, 0),
        size=(td.inf, td.inf, 0),
        freqs=[220e12, 280e12],
        name="monitor_n2f",
        theta=[0],
        phi=[0],
        proj_distance=1e3,
        far_field_approx=False,
    )

    monitor_n2f_far = td.FieldProjectionAngleMonitor(
        center=(0, 0, 0),
        size=(td.inf, td.inf, 0),
        freqs=[220e12, 280e12],
        name="monitor_n2f_far",
        theta=[0],
        phi=[0],
        proj_distance=1e5,
        far_field_approx=False,
    )

    monitor_n2f_approx = td.FieldProjectionAngleMonitor(
        center=(0, 0, 0),
        size=(td.inf, td.inf, 0),
        freqs=[220e12, 280e12],
        name="monitor_n2f_approx",
        theta=[0],
        phi=[0],
        proj_distance=1e5,
        far_field_approx=True,
    )

    src = td.PlaneWave(
        source_time=td.GaussianPulse(freq0=2.5e14, fwidth=1e13),
        center=(0, 0, 0),
        size=(td.inf, td.inf, 0),
        direction="+",
        pol_angle=-1.0,
    )

    # proj_distance large - warn
    with AssertLogLevel("WARNING"):
        _ = td.Simulation(
            size=(1, 1, 0.3),
            structures=(),
            sources=(src,),
            run_time=1e-12,
            monitors=(monitor_n2f_far,),
            boundary_spec=td.BoundarySpec.all_sides(boundary=td.PML()),
        )

    # proj_distance not too large - don't warn
    with AssertLogLevel(None):
        _ = td.Simulation(
            size=(1, 1, 0.3),
            structures=(),
            sources=(src,),
            run_time=1e-12,
            monitors=(monitor_n2f,),
            grid_spec=td.GridSpec.auto(wavelength=src.source_time.freq0),
            boundary_spec=td.BoundarySpec.all_sides(boundary=td.PML()),
        )

    # proj_distance large but using approximations - don't warn
    with AssertLogLevel(None):
        _ = td.Simulation(
            size=(1, 1, 0.3),
            structures=(),
            sources=(src,),
            run_time=1e-12,
            monitors=(monitor_n2f_approx,),
            grid_spec=td.GridSpec.auto(wavelength=src.source_time.freq0),
            boundary_spec=td.BoundarySpec.all_sides(boundary=td.PML()),
        )


@pytest.mark.parametrize(
    "monitor_type,monitor_kwargs,custom_origin,normal_dir",
    [
        # Cartesian monitor projecting backwards
        (
            td.FieldProjectionCartesianMonitor,
            {"x": [4], "y": [5], "proj_distance": -1e5, "proj_axis": 2},
            None,
            "+",
        ),
        # Cartesian monitor with custom origin projecting backwards
        (
            td.FieldProjectionCartesianMonitor,
            {"x": [4], "y": [5], "proj_distance": 39, "proj_axis": 2},
            (1, 2, -40),
            "+",
        ),
        # Cartesian monitor with custom origin projecting backwards with normal_dir '-'
        (
            td.FieldProjectionCartesianMonitor,
            {"x": [4], "y": [5], "proj_distance": 41, "proj_axis": 2},
            (1, 2, -40),
            "-",
        ),
        # Angle monitor projecting backwards
        (
            td.FieldProjectionAngleMonitor,
            {"theta": [np.pi / 2 + 1e-2], "phi": [0], "proj_distance": 1e3},
            None,
            "+",
        ),
        # Angle monitor projecting backwards with custom origin
        (
            td.FieldProjectionAngleMonitor,
            {"theta": [np.pi / 2 - 0.02], "phi": [0], "proj_distance": 10},
            (0, 0, -0.5),
            "+",
        ),
        # Angle monitor projecting backwards with custom origin and normal_dir '-'
        (
            td.FieldProjectionAngleMonitor,
            {"theta": [np.pi / 2 + 0.02], "phi": [0], "proj_distance": 10},
            (0, 0, 0.5),
            "-",
        ),
        # Cartesian monitor using approximations but too short proj_distance
        (
            td.FieldProjectionCartesianMonitor,
            {"x": [4], "y": [5], "proj_distance": 9, "proj_axis": 2},
            None,
            "+",
        ),
    ],
)
def test_proj_monitor_warnings(monitor_type, monitor_kwargs, custom_origin, normal_dir):
    """Test the validator that warns if projecting backwards."""

    src = td.PlaneWave(
        source_time=td.GaussianPulse(freq0=2.5e14, fwidth=1e13),
        center=(0, 0, -0.4),
        size=(td.inf, td.inf, 0),
        direction="+",
        pol_angle=-1.0,
    )

    monitor_kwargs.update(
        center=(0, 0, 0),
        size=(td.inf, td.inf, 0),
        freqs=[2.5e14],
        name="monitor_n2f",
    )

    if custom_origin is not None:
        monitor_kwargs["custom_origin"] = custom_origin

    if normal_dir != "+":
        monitor_kwargs["normal_dir"] = normal_dir

    monitor = monitor_type(**monitor_kwargs)

    with AssertLogLevel("WARNING"):
        _ = td.Simulation(
            size=(1, 1, 1),
            structures=(),
            sources=(src,),
            run_time=1e-12,
            monitors=(monitor,),
        )


def test_diffraction_medium():
    """Make sure we error if a diffraction monitor is in a lossy medium."""

    medium_cond = td.Medium(permittivity=2, conductivity=1)
    medium_disp = td.Lorentz(eps_inf=1.0, coeffs=[(1, 3, 2), (2, 4, 1)])

    box_cond = td.Structure(geometry=td.Box(size=(td.inf, td.inf, 1)), medium=medium_cond)
    box_disp = td.Structure(geometry=td.Box(size=(td.inf, td.inf, 1)), medium=medium_disp)

    monitor = td.DiffractionMonitor(
        center=(0, 0, 0),
        size=(td.inf, td.inf, 0),
        freqs=[250e12, 300e12],
        name="monitor_diffraction",
        normal_dir="+",
    )

    src = td.PlaneWave(
        source_time=td.GaussianPulse(freq0=2.5e14, fwidth=1e13),
        center=(0, 0, 0),
        size=(td.inf, td.inf, 0),
        direction="+",
        pol_angle=-1.0,
    )

    with pytest.raises(ValidationError):
        _ = td.Simulation(
            size=(2, 2, 2),
            structures=(box_cond,),
            sources=(src,),
            run_time=1e-12,
            monitors=(monitor,),
            boundary_spec=td.BoundarySpec.all_sides(boundary=td.Periodic()),
        )

    with pytest.raises(ValidationError):
        _ = td.Simulation(
            size=(2, 2, 2),
            structures=(box_disp,),
            sources=(src,),
            monitors=(monitor,),
            run_time=1e-12,
            boundary_spec=td.BoundarySpec.all_sides(boundary=td.Periodic()),
        )


def test_diffraction_monitor_order_grid_size():
    """Make sure overly large diffraction order grids fail during simulation creation."""

    monitor = td.DiffractionMonitor(
        center=(0, 0, 0),
        size=(td.inf, td.inf, 0),
        freqs=[td.C_0 / 1.5],
        name="monitor_diffraction",
        normal_dir="+",
    )

    with pytest.raises(ValidationError, match="100000000"):
        _ = td.Simulation(
            size=(2000, 2000, 1),
            medium=td.Medium(permittivity=16),
            run_time=1e-12,
            grid_spec=td.GridSpec.uniform(dl=500),
            monitors=(monitor,),
            boundary_spec=td.BoundarySpec.all_sides(boundary=td.Periodic()),
        )


def test_diffraction_monitor_storage_size():
    """Make sure diffraction monitors use the standard storage-size validation."""

    monitor = td.DiffractionMonitor(
        center=(0, 0, 0),
        size=(td.inf, td.inf, 0),
        freqs=np.linspace(td.C_0 / 1.55, td.C_0 / 1.45, 11),
        name="monitor_diffraction",
        normal_dir="+",
    )

    sim = td.Simulation(
        size=(1800, 1800, 1),
        medium=td.Medium(permittivity=16),
        run_time=1e-12,
        grid_spec=td.GridSpec.uniform(dl=500),
        monitors=(monitor,),
        boundary_spec=td.BoundarySpec.all_sides(boundary=td.Periodic()),
    )

    with pytest.raises(SetupError, match="maximum of 50.00GB"):
        sim.validate_pre_upload(source_required=False)


def test_diffraction_monitor_fixed_angle_source_setup():
    """Make sure fixed-angle sources don't crash diffraction setup."""

    freq0 = td.C_0
    fwidth = freq0 / 5

    source = td.PlaneWave(
        angle_phi=np.pi / 6,
        angle_theta=np.pi / 5,
        angular_spec=td.FixedAngleSpec(),
        direction="+",
        center=(-0.4, 0, 0),
        size=(0, td.inf, td.inf),
        pol_angle=np.pi / 4,
        source_time=td.GaussianPulse(freq0=freq0, fwidth=fwidth),
    )
    monitor = td.DiffractionMonitor(
        center=(0, 0, 0),
        size=(0, td.inf, td.inf),
        freqs=[freq0],
        name="monitor_diffraction",
        normal_dir="+",
    )

    sim = td.Simulation(
        size=(2.0, 2.0, 2.0),
        sources=(source,),
        monitors=(monitor,),
        run_time=10 / fwidth,
        grid_spec=td.GridSpec.auto(min_steps_per_wvl=10),
        boundary_spec=td.BoundarySpec(
            x=td.Boundary.absorber(),
            y=td.Boundary.periodic(),
            z=td.Boundary.periodic(),
        ),
    )

    assert sim._is_fixed_angle


def test_diffraction_monitor_fixed_angle_no_spurious_warning():
    """FixedAngleSpec + Periodic boundaries + DiffractionMonitor should not warn about Bloch vec."""

    freq0 = td.C_0
    fwidth = freq0 / 5

    source = td.PlaneWave(
        angle_phi=np.pi / 6,
        angle_theta=np.pi / 5,
        angular_spec=td.FixedAngleSpec(),
        direction="+",
        center=(-0.4, 0, 0),
        size=(0, td.inf, td.inf),
        pol_angle=np.pi / 4,
        source_time=td.GaussianPulse(freq0=freq0, fwidth=fwidth),
    )
    monitor = td.DiffractionMonitor(
        center=(0, 0, 0),
        size=(0, td.inf, td.inf),
        freqs=[freq0],
        name="monitor_diffraction",
        normal_dir="+",
    )

    with AssertLogStr("WARNING", excludes_str="incorrectly set"):
        td.Simulation(
            size=(2.0, 2.0, 2.0),
            sources=(source,),
            monitors=(monitor,),
            run_time=10 / fwidth,
            grid_spec=td.GridSpec.auto(min_steps_per_wvl=10),
            boundary_spec=td.BoundarySpec(
                x=td.Boundary.absorber(),
                y=td.Boundary.periodic(),
                z=td.Boundary.periodic(),
            ),
        )


@pytest.mark.parametrize(
    "box_size,log_level",
    [
        ((0.1, 0.1, 0.1), "INFO"),
        ((1, 0.1, 0.1), "WARNING"),
        ((0.1, 1, 0.1), "WARNING"),
        ((0.1, 0.1, 1), "WARNING"),
    ],
)
def test_sim_structure_extent(box_size, log_level):
    """Make sure we warn if structure extends exactly to simulation edges."""

    src = td.UniformCurrentSource(
        source_time=td.GaussianPulse(freq0=3e14, fwidth=1e13),
        size=(0, 0, 0),
        polarization="Ex",
        current_amplitude_definition="total",
    )
    box = td.Structure(geometry=td.Box(size=box_size), medium=td.Medium(permittivity=2))

    with AssertLogLevel(log_level):
        _ = td.Simulation(
            size=(1, 1, 1),
            structures=(box,),
            sources=(src,),
            run_time=1e-12,
            boundary_spec=td.BoundarySpec.all_sides(boundary=td.Periodic()),
        )


def test_warn_lumped_elements_outside_sim_bounds():
    """Test that warning is emitted for lumped elements that are not entirely contained within simulation bounds."""

    sim_center = (0, 0, 0)
    sim_size = (2, 2, 2)
    src = td.UniformCurrentSource(
        source_time=td.GaussianPulse(freq0=10e9, fwidth=8e9),
        size=(0, 0, 0),
        polarization="Ex",
        current_amplitude_definition="total",
    )

    # Lumped element fully contained - should work
    resistor_in = td.LumpedResistor(
        size=(0.5, 1, 0),
        center=(0, 0, 0),
        voltage_axis=1,
        resistance=50,
        name="resistor_inside",
    )
    with AssertLogStr("WARNING", excludes_str="not completely inside"):
        sim_good = td.Simulation(
            size=sim_size,
            center=sim_center,
            sources=(src,),
            run_time=1e-12,
            lumped_elements=(resistor_in,),
            boundary_spec=td.BoundarySpec.all_sides(boundary=td.Periodic()),
        )
    assert len(sim_good.volumetric_structures) == 1

    # Lumped element is touching the boundary along one of its nonzero dims
    resistor_in = td.LumpedResistor(
        size=(0.5, 1, 0),
        center=(0, 0.5, 0),
        voltage_axis=1,
        resistance=50,
        name="resistor_touching",
    )
    with AssertLogStr("WARNING", excludes_str="not completely inside"):
        sim_good = td.Simulation(
            size=sim_size,
            center=sim_center,
            sources=(src,),
            run_time=1e-12,
            lumped_elements=(resistor_in,),
            boundary_spec=td.BoundarySpec.all_sides(boundary=td.Periodic()),
        )
    assert len(sim_good.volumetric_structures) == 1

    # Lumped element outside - should emit warning and not be added
    resistor_out = td.LumpedResistor(
        size=(0.5, 1, 0),
        center=(0, 2, 0),
        voltage_axis=1,
        resistance=50,
        name="resistor_outside",
    )
    with AssertLogStr("WARNING", contains_str="not completely inside"):
        sim_bad = sim_good.updated_copy(lumped_elements=(resistor_out,))
    assert len(sim_bad.volumetric_structures) == 0

    # Lumped element is flush against boundary along its zero size dimension
    resistor_edge = td.LumpedResistor(
        size=(0.5, 1, 0),
        center=(0, 0.5, 1),
        voltage_axis=1,
        resistance=50,
        name="resistor_edge",
    )
    with AssertLogStr("WARNING", contains_str="not completely inside"):
        sim_bad = sim_good.updated_copy(lumped_elements=(resistor_edge,))
    assert len(sim_bad.volumetric_structures) == 0


@pytest.mark.parametrize(
    "box_length,absorb_type,log_level",
    [
        (0.0001, "PML", None),
        (1, "PML", "WARNING"),
        (1.5, "absorber", None),
        (2.0, "PML", None),
    ],
)
def test_sim_validate_structure_bounds_pml(box_length, absorb_type, log_level):
    """Make sure we warn if structure bounds are within the PML exactly to simulation edges."""

    # For PML, set extrude_structures=False to test the warning behavior
    # (with extrude_structures=True, structures are automatically extended so no warning is needed)
    boundary = td.PML(extrude_structures=False) if absorb_type == "PML" else td.Absorber()

    src = td.UniformCurrentSource(
        source_time=td.GaussianPulse(freq0=3e14, fwidth=1e13),
        size=(0, 0, 0),
        polarization="Ex",
        current_amplitude_definition="total",
    )
    box = td.Structure(
        geometry=td.Box(size=(box_length, 0.5, 0.5), center=(0, 0, 0)),
        medium=td.Medium(permittivity=2),
    )

    with AssertLogLevel(log_level):
        _ = td.Simulation(
            size=(1, 1, 1),
            structures=(box,),
            grid_spec=td.GridSpec.auto(wavelength=0.001),
            sources=(src,),
            run_time=1e-12,
            boundary_spec=td.BoundarySpec(
                x=td.Boundary(plus=boundary, minus=boundary),
                y=td.Boundary.pec(),
                z=td.Boundary.pec(),
            ),
        )


def test_num_mediums(monkeypatch):
    """Make sure we error if too many mediums supplied."""
    monkeypatch.setattr(simulation, "MAX_NUM_MEDIUMS", TEST_MAX_NUM_MEDIUMS)
    structures = []
    grid_spec = td.GridSpec.auto(wavelength=1.0)
    for i in range(TEST_MAX_NUM_MEDIUMS):
        structures.append(
            td.Structure(geometry=td.Box(size=(1, 1, 1)), medium=td.Medium(permittivity=i + 1))
        )
    _ = td.Simulation(
        size=(5, 5, 5),
        grid_spec=grid_spec,
        structures=structures,
        run_time=1e-12,
        boundary_spec=td.BoundarySpec.all_sides(boundary=td.Periodic()),
    )

    with pytest.raises(ValidationError):
        structures.append(
            td.Structure(geometry=td.Box(size=(1, 1, 1)), medium=td.Medium(permittivity=i + 2))
        )
        _ = td.Simulation(
            size=(5, 5, 5), grid_spec=grid_spec, structures=structures, run_time=1e-12
        )


def test_unique_medium_names():
    """Warn if non-unique medium names supplied."""

    with AssertLogLevel("WARNING", contains_str="unique names"):
        _ = td.Simulation(
            size=(5, 5, 5),
            structures=[
                td.Structure(
                    geometry=td.Box(size=(1, 1, 1)),
                    medium=td.Medium(permittivity=2, name="medium1"),
                ),
                td.Structure(
                    geometry=td.Box(size=(1, 1, 1), center=(1, 0, 0)),
                    medium=td.Medium(permittivity=3, name="medium1"),
                ),
            ],
            run_time=1e-12,
            grid_spec=td.GridSpec.uniform(dl=0.02),
        )


def test_num_sources():
    """Make sure we error if too many sources supplied."""

    src = td.PlaneWave(
        source_time=td.GaussianPulse(freq0=2.5e14, fwidth=1e13),
        center=(0, 0, 0),
        size=(td.inf, td.inf, 0),
        direction="+",
    )

    _ = td.Simulation(size=(5, 5, 5), run_time=1e-12, sources=(src,) * MAX_NUM_SOURCES)

    with pytest.raises(ValidationError):
        _ = td.Simulation(size=(5, 5, 5), run_time=1e-12, sources=(src,) * (MAX_NUM_SOURCES + 1))


def _test_names_default():
    """makes sure default names are set"""

    sim = td.Simulation(
        size=(2.0, 2.0, 2.0),
        run_time=1e-12,
        structures=(
            td.Structure(
                geometry=td.Box(size=(1, 1, 1), center=(-1, 0, 0)),
                medium=td.Medium(permittivity=2.0),
            ),
            td.Structure(
                geometry=td.Box(size=(1, 1, 1), center=(0, 0, 0)),
                medium=td.Medium(permittivity=2.0),
            ),
            td.Structure(
                geometry=td.Sphere(radius=1.4, center=(1.0, 0.0, 1.0)), medium=td.Medium()
            ),
            td.Structure(
                geometry=td.Cylinder(radius=1.4, length=2.0, center=(1.0, 0.0, -1.0), axis=1),
                medium=td.Medium(),
            ),
        ),
        sources=(
            td.UniformCurrentSource(
                size=(0, 0, 0),
                center=(0, -0.5, 0),
                polarization="Hx",
                source_time=td.GaussianPulse(freq0=1e14, fwidth=1e12),
                current_amplitude_definition="total",
            ),
            td.UniformCurrentSource(
                size=(0, 0, 0),
                center=(0, -0.5, 0),
                polarization="Ex",
                source_time=td.GaussianPulse(freq0=1e14, fwidth=1e12),
                current_amplitude_definition="total",
            ),
            td.UniformCurrentSource(
                size=(0, 0, 0),
                center=(0, -0.5, 0),
                polarization="Ey",
                source_time=td.GaussianPulse(freq0=1e14, fwidth=1e12),
                current_amplitude_definition="total",
            ),
        ),
        monitors=(
            td.FluxMonitor(size=(1, 1, 0), center=(0, -0.5, 0), freqs=[1e12], name="mon1"),
            td.FluxMonitor(size=(0, 1, 1), center=(0, -0.5, 0), freqs=[1e12], name="mon2"),
            td.FluxMonitor(size=(1, 0, 1), center=(0, -0.5, 0), freqs=[1e12], name="mon3"),
        ),
        boundary_spec=td.BoundarySpec.all_sides(boundary=td.Periodic()),
    )

    for i, structure in enumerate(sim.structures):
        assert structure.name == f"structures[{i}]"

    for i, source in enumerate(sim.sources):
        assert source.name == f"sources[{i}]"


def test_names_unique():
    with pytest.raises(ValidationError):
        _ = td.Simulation(
            size=(2.0, 2.0, 2.0),
            run_time=1e-12,
            structures=(
                td.Structure(
                    geometry=td.Box(size=(1, 1, 1), center=(-1, 0, 0)),
                    medium=td.Medium(permittivity=2.0),
                    name="struct1",
                ),
                td.Structure(
                    geometry=td.Box(size=(1, 1, 1), center=(0, 0, 0)),
                    medium=td.Medium(permittivity=2.0),
                    name="struct1",
                ),
            ),
            boundary_spec=td.BoundarySpec.all_sides(boundary=td.Periodic()),
        )

    with pytest.raises(ValidationError):
        _ = td.Simulation(
            size=(2.0, 2.0, 2.0),
            run_time=1e-12,
            sources=(
                td.UniformCurrentSource(
                    size=(0, 0, 0),
                    center=(0, -0.5, 0),
                    polarization="Hx",
                    source_time=td.GaussianPulse(freq0=1e14, fwidth=1e12),
                    name="source1",
                    current_amplitude_definition="total",
                ),
                td.UniformCurrentSource(
                    size=(0, 0, 0),
                    center=(0, -0.5, 0),
                    polarization="Ex",
                    source_time=td.GaussianPulse(freq0=1e14, fwidth=1e12),
                    name="source1",
                    current_amplitude_definition="total",
                ),
            ),
            boundary_spec=td.BoundarySpec.all_sides(boundary=td.Periodic()),
        )

    with pytest.raises(ValidationError):
        _ = td.Simulation(
            size=(2.0, 2.0, 2.0),
            run_time=1e-12,
            monitors=(
                td.FluxMonitor(size=(1, 1, 0), center=(0, -0.5, 0), freqs=[1e12], name="mon1"),
                td.FluxMonitor(size=(0, 1, 1), center=(0, -0.5, 0), freqs=[1e12], name="mon1"),
            ),
            boundary_spec=td.BoundarySpec.all_sides(boundary=td.Periodic()),
        )


def test_mode_object_syms():
    """Test that errors are raised if a mode object is not placed right in the presence of syms."""
    g = td.GaussianPulse(freq0=1e12, fwidth=0.1e12)

    # wrong mode source
    with pytest.raises(ValidationError):
        _ = td.Simulation(
            center=(1.0, -1.0, 0.5),
            size=(2.0, 2.0, 2.0),
            grid_spec=td.GridSpec.auto(wavelength=td.C_0 / 1.0),
            run_time=1e-12,
            symmetry=(1, -1, 0),
            sources=(td.ModeSource(size=(2, 2, 0), direction="+", source_time=g),),
            boundary_spec=td.BoundarySpec.all_sides(boundary=td.Periodic()),
        )

    # wrong mode monitor
    with pytest.raises(ValidationError):
        _ = td.Simulation(
            center=(1.0, -1.0, 0.5),
            size=(2.0, 2.0, 2.0),
            grid_spec=td.GridSpec.auto(wavelength=td.C_0 / 1.0),
            run_time=1e-12,
            symmetry=(1, -1, 0),
            monitors=(
                td.ModeMonitor(size=(2, 2, 0), name="mnt", freqs=[2e12], mode_spec=td.ModeSpec()),
            ),
            boundary_spec=td.BoundarySpec.all_sides(boundary=td.Periodic()),
        )

    # right mode source (centered on the symmetry)
    _ = td.Simulation(
        center=(1.0, -1.0, 0.5),
        size=(2.0, 2.0, 2.0),
        grid_spec=td.GridSpec.auto(wavelength=td.C_0 / 1.0),
        run_time=1e-12,
        symmetry=(1, -1, 0),
        sources=(td.ModeSource(center=(1, -1, 1), size=(2, 2, 0), direction="+", source_time=g),),
        boundary_spec=td.BoundarySpec.all_sides(boundary=td.Periodic()),
    )

    # right mode monitor (entirely in the main quadrant)
    _ = td.Simulation(
        center=(1.0, -1.0, 0.5),
        size=(2.0, 2.0, 2.0),
        grid_spec=td.GridSpec.auto(wavelength=td.C_0 / 1.0),
        run_time=1e-12,
        symmetry=(1, -1, 0),
        monitors=(
            td.ModeMonitor(
                center=(2, 0, 1), size=(2, 2, 0), name="mnt", freqs=[2e12], mode_spec=td.ModeSpec()
            ),
        ),
        boundary_spec=td.BoundarySpec.all_sides(boundary=td.Periodic()),
    )


def test_tfsf_symmetry():
    """Test that a TFSF source cannot be set in the presence of symmetries."""
    src_time = td.GaussianPulse(freq0=1e12, fwidth=0.1e12)

    source = td.TFSF(
        size=[1, 1, 1],
        source_time=src_time,
        pol_angle=0,
        angle_theta=np.pi / 4,
        angle_phi=np.pi / 6,
        direction="+",
        injection_axis=2,
    )

    with pytest.raises(ValidationError):
        _ = td.Simulation(
            size=(2.0, 2.0, 2.0),
            grid_spec=td.GridSpec.auto(wavelength=td.C_0 / 1.0),
            run_time=1e-12,
            symmetry=(0, -1, 0),
            sources=(source,),
        )


def test_tfsf_aux_source_outside_domain():
    """Test that a TFSF source cannot be too close to the simulation domain boundaries
    along the injection direction."""
    src_time = td.GaussianPulse(freq0=1e12, fwidth=0.1e12)

    source = td.TFSF(
        size=[1, 1, 1],
        source_time=src_time,
        pol_angle=0,
        angle_theta=np.pi / 4,
        angle_phi=np.pi / 6,
        direction="+",
        injection_axis=2,
    )

    with pytest.raises(ValidationError):
        _ = td.Simulation(
            size=(2.0, 2.0, 1.01),
            grid_spec=td.GridSpec.auto(wavelength=td.C_0 / 1.0),
            run_time=1e-12,
            sources=(source,),
        )


def test_tfsf_boundaries():
    """Test that a TFSF source is allowed to cross boundaries only in particular cases."""
    src_time = td.GaussianPulse(freq0=td.C_0, fwidth=0.1e12)

    source = td.TFSF(
        size=[1, 1, 1],
        source_time=src_time,
        pol_angle=0,
        angle_theta=np.pi / 4,
        angle_phi=np.pi / 6,
        direction="+",
        injection_axis=2,
    )

    # can cross periodic boundaries in the transverse directions
    _ = td.Simulation(
        size=(2.0, 0.5, 2.0),
        grid_spec=td.GridSpec.auto(wavelength=1.0),
        boundary_spec=td.BoundarySpec(
            x=td.Boundary.periodic(),
            y=td.Boundary.periodic(),
            z=td.Boundary.periodic(),
        ),
        run_time=1e-12,
        sources=(source,),
    )

    # can cross Bloch boundaries in the transverse directions
    _ = td.Simulation(
        size=(0.5, 0.5, 2.0),
        grid_spec=td.GridSpec.auto(wavelength=1.0),
        run_time=1e-12,
        sources=(source,),
        boundary_spec=td.BoundarySpec(
            x=td.Boundary.bloch_from_source(source=source, domain_size=0.5, axis=0, medium=None),
            y=td.Boundary.bloch_from_source(source=source, domain_size=0.5, axis=1, medium=None),
            z=td.Boundary.pml(),
        ),
    )

    # warn if Bloch boundaries are crossed in the transverse directions but
    # the Bloch vector is incorrect
    with AssertLogLevel("WARNING"):
        _ = td.Simulation(
            size=(0.5, 0.5, 2.0),
            grid_spec=td.GridSpec.auto(wavelength=1.0),
            run_time=1e-12,
            sources=(source,),
            boundary_spec=td.BoundarySpec(
                x=td.Boundary.bloch_from_source(
                    source=source,
                    domain_size=0.5 * 1.1,
                    axis=0,
                    medium=None,  # wrong domain size
                ),
                y=td.Boundary.bloch_from_source(
                    source=source,
                    domain_size=0.5 * 1.1,
                    axis=1,
                    medium=None,  # wrong domain size
                ),
                z=td.Boundary.pml(),
            ),
        )

    # cannot cross any boundary in the direction of injection
    with pytest.raises(ValidationError):
        _ = td.Simulation(
            size=(2.0, 2.0, 0.5),
            grid_spec=td.GridSpec.auto(wavelength=1.0),
            run_time=1e-12,
            sources=(source,),
        )

    # cannot cross any non-periodic boundary in the transverse direction
    with pytest.raises(ValidationError):
        _ = td.Simulation(
            center=(0.5, 0, 0),  # also check the case when the boundary is crossed only on one side
            size=(0.5, 0.5, 2.0),
            grid_spec=td.GridSpec.auto(wavelength=1.0),
            run_time=1e-12,
            sources=(source,),
            boundary_spec=td.BoundarySpec(
                x=td.Boundary.pml(),
                y=td.Boundary.absorber(),
            ),
        )


def test_tfsf_structures_grid():
    """Test that a TFSF source is allowed to intersect structures only in particular cases."""
    src_time = td.GaussianPulse(freq0=td.C_0, fwidth=0.1e12)

    source = td.TFSF(
        size=[1, 1, 1],
        source_time=src_time,
        pol_angle=0,
        angle_theta=np.pi / 4,
        angle_phi=np.pi / 6,
        direction="+",
        injection_axis=2,
    )

    # a non-uniform mesh along the transverse directions should issue a warning
    with AssertLogLevel("WARNING"):
        sim = td.Simulation(
            size=(2.0, 2.0, 2.0),
            grid_spec=td.GridSpec.auto(wavelength=1.0),
            run_time=1e-12,
            sources=(source,),
            structures=(
                td.Structure(
                    geometry=td.Box(center=(0, 0, -1), size=(0.5, 0.5, 0.5)),
                    medium=td.Medium(permittivity=2),
                ),
            ),
        )

    sim.validate_pre_upload()

    # must not have different material profiles on different faces along the injection axis
    sim = td.Simulation(
        size=(2.0, 2.0, 2.0),
        grid_spec=td.GridSpec.auto(wavelength=1.0),
        run_time=1e-12,
        sources=(source,),
        structures=(
            td.Structure(
                geometry=td.Box(center=(0.5, 0, 0), size=(0.25, 0.25, 0.25)),
                medium=td.Medium(permittivity=2),
            ),
        ),
    )
    with pytest.raises(SetupError):
        sim.validate_pre_upload()

    # different structures *are* allowed on different faces as long as material properties match
    sim = td.Simulation(
        size=(2.0, 2.0, 2.0),
        grid_spec=td.GridSpec.auto(wavelength=1.0),
        run_time=1e-12,
        sources=(source,),
        structures=(
            td.Structure(
                geometry=td.Box(center=(0.5, 0, 0), size=(0.25, 0.25, 0.25)), medium=td.Medium()
            ),
        ),
    )

    # TFSF box must not intersect a custom medium
    Nx, Ny, Nz = 10, 9, 8
    X = np.linspace(-1, 1, Nx)
    Y = np.linspace(-1, 1, Ny)
    Z = np.linspace(-1, 1, Nz)
    data = np.ones((Nx, Ny, Nz, 1))
    eps_diagonal_data = td.ScalarFieldDataArray(
        data, coords={"x": X, "y": Y, "z": Z, "f": [td.C_0]}
    )
    eps_components = {f"eps_{d}{d}": eps_diagonal_data for d in "xyz"}
    eps_dataset = td.PermittivityDataset(**eps_components)
    custom_medium = td.CustomMedium(eps_dataset=eps_dataset, name="my_medium")
    sim = td.Simulation(
        size=(2.0, 2.0, 2.0),
        grid_spec=td.GridSpec.auto(wavelength=1.0),
        run_time=1e-12,
        sources=(source,),
        structures=(
            td.Structure(
                geometry=td.Box(center=(0.5, 0, 0), size=(td.inf, td.inf, 0.25)),
                medium=custom_medium,
            ),
        ),
    )
    with pytest.raises(SetupError):
        sim.validate_pre_upload()

    # TFSF box must not intersect a fully anisotropic medium
    anisotropic_medium = td.FullyAnisotropicMedium(
        permittivity=np.eye(3).tolist(), conductivity=np.eye(3).tolist()
    )
    sim = td.Simulation(
        size=(2.0, 2.0, 2.0),
        grid_spec=td.GridSpec.auto(wavelength=1.0),
        run_time=1e-12,
        sources=(source,),
        structures=(
            td.Structure(
                geometry=td.Box(center=(0.5, 0, 0), size=(td.inf, td.inf, 0.25)),
                medium=anisotropic_medium,
            ),
        ),
    )
    with pytest.raises(SetupError):
        sim.validate_pre_upload()


@pytest.mark.parametrize(
    "size, num_struct, log_level", [(1, 1, None), (2, 1, "WARNING"), (1, 11, "WARNING")]
)
@td.packaging.disable_local_subpixel
def test_warn_large_epsilon(monkeypatch, size, num_struct, log_level):
    """Make sure we get a warning if the epsilon grid is too large."""

    monkeypatch.setattr(simulation, "NUM_STRUCTURES_WARN_EPSILON", 10)
    monkeypatch.setattr(simulation, "NUM_CELLS_WARN_EPSILON", 2_000)
    structures = [
        td.Structure(
            geometry=td.Box(center=(0, 0, 0), size=(0.1, 0.1, 0.1)),
            medium=td.Medium(permittivity=eps),
        )
        for eps in np.linspace(1, 2, num_struct)
    ]
    sim = td.Simulation(
        size=(size, size, size),
        grid_spec=td.GridSpec.uniform(dl=0.1),
        run_time=1e-12,
        boundary_spec=td.BoundarySpec.all_sides(boundary=td.Periodic()),
        sources=(
            td.ModeSource(
                center=(0, 0, 0),
                size=(td.inf, td.inf, 0),
                direction="+",
                source_time=td.GaussianPulse(freq0=1e12, fwidth=0.1e12),
            ),
        ),
        structures=tuple(structures),
    )

    with AssertLogLevel(log_level):
        sim.epsilon(box=td.Box(size=(size, size, size)))


@pytest.mark.parametrize("dl, log_level", [(0.1, None), (0.005, "WARNING")])
def test_warn_large_mode_monitor(dl, log_level):
    """Make sure we get a warning if the mode monitor grid is too large."""

    sim = td.Simulation(
        size=(2.0, 2.0, 2.0),
        grid_spec=td.GridSpec.uniform(dl=dl),
        run_time=1e-12,
        sources=(
            td.ModeSource(
                size=(0.4, 0.4, 0),
                direction="+",
                source_time=td.GaussianPulse(freq0=1e12, fwidth=0.1e12),
            ),
        ),
        monitors=(
            td.ModeMonitor(
                size=(td.inf, 0, td.inf), freqs=[1e12], name="test", mode_spec=td.ModeSpec()
            ),
        ),
    )

    with AssertLogLevel(log_level):
        sim.validate_pre_upload()


@pytest.mark.parametrize("dl, log_level", [(0.1, None), (0.005, "WARNING")])
def test_warn_large_mode_source(dl, log_level):
    """Make sure we get a warning if the mode source grid is too large."""

    sim = td.Simulation(
        size=(2.0, 2.0, 2.0),
        grid_spec=td.GridSpec.uniform(dl=dl),
        run_time=1e-12,
        sources=(
            td.ModeSource(
                size=(td.inf, td.inf, 0),
                direction="+",
                source_time=td.GaussianPulse(freq0=1e12, fwidth=0.1e12),
            ),
        ),
    )

    with AssertLogLevel(log_level):
        sim.validate_pre_upload()


mnt_size = (td.inf, 0, td.inf)
mnt_test = [
    td.ModeMonitor(size=mnt_size, freqs=[1e12], name="test", mode_spec=td.ModeSpec()),
    td.FluxMonitor(size=mnt_size, freqs=[1e12], name="test"),
    td.FluxTimeMonitor(size=mnt_size, name="test"),
    td.DiffractionMonitor(size=mnt_size, freqs=[1e12], name="test"),
    td.FieldProjectionAngleMonitor(size=mnt_size, freqs=[1e12], name="test", theta=[0], phi=[0]),
    td.FieldMonitor(size=mnt_size, freqs=[1e12], name="test", fields=["Ex", "Hx"]),
    td.FieldTimeMonitor(size=mnt_size, stop=1e-17, name="test", fields=["Ex", "Hx"]),
]


@pytest.mark.parametrize("monitor", mnt_test)
def test_error_large_monitors(monitor):
    """Test if various large monitors cause pre-upload validation to error."""

    sim_large = td.Simulation(
        size=(40.0, 0, 40.0),
        grid_spec=td.GridSpec.uniform(dl=0.001),
        run_time=1e-12,
        boundary_spec=td.BoundarySpec.all_sides(boundary=td.Periodic()),
        sources=(
            td.ModeSource(
                size=(0.1, 0.1, 0),
                direction="+",
                source_time=td.GaussianPulse(freq0=1e12, fwidth=0.1e12),
            ),
        ),
        monitors=(monitor,),
    )

    # small sim should not error
    sim_small = sim_large.updated_copy(size=(4.0, 0, 4.0))
    sim_small.validate_pre_upload()

    # large sim should error
    with pytest.raises(SetupError):
        sim_large.validate_pre_upload()


def test_error_max_time_monitor_steps():
    """Test if a time monitor with too many time steps causes pre upload error."""

    sim = td.Simulation(
        size=(5, 5, 5),
        run_time=1e-12,
        grid_spec=td.GridSpec.uniform(dl=0.01),
        sources=(
            td.ModeSource(
                size=(0.1, 0.1, 0),
                direction="+",
                source_time=td.GaussianPulse(freq0=2e14, fwidth=0.1e14),
            ),
        ),
    )

    # simulation with a 0D time monitor should not error
    monitor = td.FieldTimeMonitor(center=(0, 0, 0), size=(0, 0, 0), name="time")
    sim = sim.updated_copy(monitors=(monitor,))
    sim.validate_pre_upload()

    # 1D monitor should error
    with pytest.raises(SetupError):
        monitor = monitor.updated_copy(size=(1, 0, 0))
        sim = sim.updated_copy(monitors=(monitor,))
        sim.validate_pre_upload()

    # setting a large enough interval should again not error
    monitor = monitor.updated_copy(interval=20)
    sim = sim.updated_copy(monitors=(monitor,))
    sim.validate_pre_upload()


def test_monitor_num_cells():
    """Test the computation of number of cells in monitor."""
    sim = td.Simulation(
        size=(2.0, 2.0, 2.0),
        grid_spec=td.GridSpec.uniform(dl=0.01),
        run_time=1e-12,
    )
    monitor_3d = td.FluxMonitor(size=[1, 1, 1], freqs=[1e12], name="test")
    monitor_2d = td.FluxMonitor(size=[1, 0, 1], freqs=[1e12], name="test")
    downsample = 3
    monitor_downsample = td.FieldMonitor(
        size=[1, 0, 1], freqs=[1e12], name="test", interval_space=[downsample] * 3
    )
    num_cells_3d = sim._monitor_num_cells(monitor_3d)
    num_cells_2d = sim._monitor_num_cells(monitor_2d)
    num_cells_downsample = sim._monitor_num_cells(monitor_downsample)
    assert num_cells_2d * 6 == num_cells_3d
    # downsampling is not exact
    assert np.isclose(num_cells_downsample, num_cells_2d / downsample**2, rtol=0.1)


@pytest.mark.parametrize("start, log_level", [(1e-12, None), (1, "WARNING")])
def test_warn_time_monitor_outside_run_time(start, log_level):
    """Make sure we get a warning if the mode monitor grid is too large."""

    sim = td.Simulation(
        size=(2.0, 2.0, 2.0),
        grid_spec=td.GridSpec.uniform(dl=0.1),
        run_time=1e-12,
        sources=(
            td.ModeSource(
                size=(0.4, 0.4, 0),
                direction="+",
                source_time=td.GaussianPulse(freq0=1e12, fwidth=0.1e12),
            ),
        ),
        monitors=(td.FieldTimeMonitor(size=(td.inf, 0, td.inf), start=start, name="test"),),
    )
    with AssertLogLevel(log_level, contains_str="start time"):
        sim.validate_pre_upload()


def test_dt():
    """make sure dt is reduced when there is a medium with eps_inf < 1."""
    sim = td.Simulation(
        size=(2.0, 2.0, 2.0),
        run_time=1e-12,
        grid_spec=td.GridSpec.uniform(dl=0.1),
    )
    dt = sim.dt

    # simulation with eps_inf < 1
    structure = td.Structure(
        geometry=td.Box(size=(1, 1, 1), center=(-1, 0, 0)),
        medium=td.PoleResidue(eps_inf=0.16, poles=[(-1 + 1j, 2 + 2j)]),
    )
    sim_new = sim.copy(update={"structures": (structure,)})
    assert sim_new.dt == 0.4 * dt


def test_conformal_dt():
    """make sure dt is reduced when PEC structures are present and PECConformal is used."""
    box = td.Structure(
        geometry=td.Box(size=(1, 1, 1)),
        medium=td.PECMedium(),
    )
    sim = td.Simulation(
        size=(2.0, 2.0, 2.0),
        run_time=1e-12,
        structures=(box,),
        grid_spec=td.GridSpec.uniform(dl=0.1),
        subpixel=td.SubpixelSpec(pec=td.Staircasing()),
    )
    dt = sim.dt

    # Conformal
    sim_conformal = sim.updated_copy(subpixel=td.SubpixelSpec(pec=td.PECConformal()))
    assert sim_conformal.dt < dt

    # Conformal: same courant
    sim_conformal2 = sim.updated_copy(
        subpixel=td.SubpixelSpec(pec=td.PECConformal(timestep_reduction=0))
    )
    assert sim_conformal2.dt == dt

    # heuristic
    sim_heuristic = sim.updated_copy(subpixel=td.SubpixelSpec(pec=td.HeuristicPECStaircasing()))
    assert sim_heuristic.dt == dt


def test_edge_correction():
    """make sure edge correction can be enabled for PEC and lossy meal."""
    sim = td.Simulation(
        size=(2.0, 2.0, 2.0),
        run_time=1e-12,
        structures=[],
        grid_spec=td.GridSpec.uniform(dl=0.1),
        subpixel=td.SubpixelSpec(
            pec=td.PECConformal(edge_singularity_correction=False),
            lossy_metal=td.SurfaceImpedance(edge_singularity_correction=False),
        ),
    )

    sim = sim.updated_copy(
        subpixel=td.SubpixelSpec(
            pec=td.PECConformal(edge_singularity_correction=True),
            lossy_metal=td.SurfaceImpedance(edge_singularity_correction=True),
        )
    )


def test_sim_volumetric_structures(tmp_path):
    """Test volumetric equivalent of 2D materials."""
    sigma = 0.45
    thickness = 0.01
    medium = td.Medium2D.from_medium(td.Medium(conductivity=sigma), thickness=thickness)
    grid_dl = 0.03
    box = td.Structure(geometry=td.Box(size=(td.inf, td.inf, 0)), medium=medium)
    cyl = td.Structure(geometry=td.Cylinder(radius=1, length=0), medium=medium)
    pslab = td.Structure(
        geometry=td.PolySlab(vertices=[(-1, -1), (-1, 1), (1, 1), (1, -1)], slab_bounds=(0, 0)),
        medium=medium,
    )
    src = td.UniformCurrentSource(
        source_time=td.GaussianPulse(freq0=1.5e14, fwidth=0.5e14),
        size=(0, 0, 0),
        polarization="Ex",
        current_amplitude_definition="total",
    )
    for struct in [box, cyl, pslab]:
        sim = td.Simulation(
            size=(10, 10, 10),
            structures=(struct,),
            sources=(src,),
            boundary_spec=td.BoundarySpec(
                x=td.Boundary.pml(num_layers=6),
                y=td.Boundary.pml(num_layers=6),
                z=td.Boundary.pml(num_layers=6),
            ),
            grid_spec=td.GridSpec.uniform(dl=grid_dl),
            run_time=1e-12,
        )
        if isinstance(struct.geometry, td.Box):
            assert np.isclose(
                sim.volumetric_structures[0].geometry.bounding_box.size[2], 0, rtol=RTOL
            )
        else:
            assert np.isclose(sim.volumetric_structures[0].geometry.length_axis, 0, rtol=RTOL)
        assert np.isclose(
            sim.volumetric_structures[0].medium.xx.to_medium().conductivity,
            sigma * thickness / grid_dl,
            rtol=RTOL,
        )
    # now with a substrate and anisotropy
    aniso_medium = td.AnisotropicMedium(
        xx=td.Medium(permittivity=2), yy=td.Medium(), zz=td.Medium()
    )
    box = td.Structure(
        geometry=td.Box(size=(td.inf, td.inf, 0)),
        medium=td.Medium2D.from_medium(td.Medium(permittivity=1), thickness=thickness),
    )
    below = td.Structure(
        geometry=td.Box.from_bounds([-td.inf, -td.inf, -1000], [td.inf, td.inf, 0]),
        medium=aniso_medium,
    )
    monitor = td.FieldMonitor(
        center=(0, 0, 0),
        size=(td.inf, 0, td.inf),
        freqs=(1.5e14),
        name="field_xz",
    )
    sim = td.Simulation(
        size=(10, 10, 10),
        structures=(below, box),
        sources=(src,),
        monitors=(monitor,),
        boundary_spec=td.BoundarySpec(
            x=td.Boundary.pml(num_layers=6),
            y=td.Boundary.pml(num_layers=6),
            z=td.Boundary.pml(num_layers=6),
        ),
        grid_spec=td.GridSpec.uniform(dl=grid_dl),
        run_time=1e-12,
    )
    assert np.isclose(
        sim.volumetric_structures[1].medium.xx.to_medium().permittivity,
        1.5,
        rtol=RTOL,
    )
    assert np.isclose(sim.volumetric_structures[1].medium.yy.to_medium().permittivity, 1, rtol=RTOL)

    # PEC
    box = td.Structure(
        geometry=td.Box(size=(td.inf, td.inf, 0)),
        medium=td.PEC2D,
    )
    sim = td.Simulation(
        size=(10, 10, 10),
        structures=(below, box),
        sources=(src,),
        monitors=(monitor,),
        boundary_spec=td.BoundarySpec(
            x=td.Boundary.pml(num_layers=6),
            y=td.Boundary.pml(num_layers=6),
            z=td.Boundary.pml(num_layers=6),
        ),
        grid_spec=td.GridSpec.uniform(dl=grid_dl),
        run_time=1e-12,
    )
    assert isinstance(sim.volumetric_structures[1].medium.xx, td.PECMedium)

    # plotting should not raise warning
    with AssertLogLevel(None):
        # check that plotting 2d material doesn't raise an error
        sim_data = run_emulated(sim)
        sim_data.plot_field(field_monitor_name="field_xz", field_name="Ex", val="real")
        plt.close()
        _ = sim.plot_eps(x=0, alpha=0.2)
        plt.close()
        _ = sim.plot(x=0)
        plt.close()

    # nonuniform sub/super-strate should not error
    below_half = td.Structure(
        geometry=td.Box.from_bounds([-100, -td.inf, -1000], [0, td.inf, 0]),
        medium=aniso_medium,
    )

    sim = td.Simulation(
        size=(10, 10, 10),
        structures=(below_half, box),
        sources=(src,),
        boundary_spec=td.BoundarySpec(
            x=td.Boundary.pml(num_layers=6),
            y=td.Boundary.pml(num_layers=6),
            z=td.Boundary.pml(num_layers=6),
        ),
        grid_spec=td.GridSpec.uniform(dl=grid_dl),
        run_time=1e-12,
    )

    _ = sim.volumetric_structures

    # structure overlaying the 2D material should overwrite it like normal
    sim = td.Simulation(
        size=(10, 10, 10),
        structures=(box, below),
        sources=(src,),
        boundary_spec=td.BoundarySpec(
            x=td.Boundary.pml(num_layers=6),
            y=td.Boundary.pml(num_layers=6),
            z=td.Boundary.pml(num_layers=6),
        ),
        grid_spec=td.GridSpec.uniform(dl=grid_dl),
        run_time=1e-12,
    )

    assert np.isclose(sim.volumetric_structures[1].medium.xx.permittivity, 2, rtol=RTOL)

    # test simulation.medium can't be Medium2D
    with pytest.raises(ValidationError):
        sim = td.Simulation(
            size=(10, 10, 10),
            structures=(),
            sources=(src,),
            medium=box.medium,
            boundary_spec=td.BoundarySpec(
                x=td.Boundary.pml(num_layers=6),
                y=td.Boundary.pml(num_layers=6),
                z=td.Boundary.pml(num_layers=6),
            ),
            grid_spec=td.GridSpec.uniform(dl=grid_dl),
            run_time=1e-12,
        )

    # test 2d medium is added to 2d geometry
    with pytest.raises(ValidationError):
        _ = td.Structure(geometry=td.Box(center=(0, 0, 0), size=(1, 1, 1)), medium=box.medium)
    with pytest.raises(ValidationError):
        _ = td.Structure(geometry=td.Cylinder(radius=1, length=1), medium=box.medium)
    with pytest.raises(ValidationError):
        _ = td.Structure(
            geometry=td.PolySlab(vertices=[(0, 0), (1, 0), (1, 1)], slab_bounds=(-1, 1)),
            medium=box.medium,
        )
    with pytest.raises(ValidationError):
        _ = td.Structure(geometry=td.Sphere(radius=1), medium=box.medium)

    # test warning for 2d geometry in simulation without Medium2D
    with AssertLogLevel("WARNING"):
        struct = td.Structure(medium=td.Medium(), geometry=td.Box(size=(1, 0, 1)))
        sim = td.Simulation(
            size=(10, 10, 10),
            structures=(struct,),
            sources=(src,),
            boundary_spec=td.BoundarySpec(
                x=td.Boundary.pml(num_layers=6),
                y=td.Boundary.pml(num_layers=6),
                z=td.Boundary.pml(num_layers=6),
            ),
            grid_spec=td.GridSpec.uniform(dl=grid_dl),
            run_time=1e-12,
        )


@pytest.mark.parametrize("normal_axis", (0, 1, 2))
def test_pml_boxes_2D(normal_axis):
    """Ensure pml boxes have non-zero dimension for 2D sim."""

    sim_size = [1, 1, 1]
    sim_size[normal_axis] = 0
    pml_on_kwargs = {dim: axis != normal_axis for axis, dim in enumerate("xyz")}

    sim2d = td.Simulation(
        size=sim_size,
        run_time=1e-12,
        grid_spec=td.GridSpec(wavelength=1.0),
        sources=(
            td.PointDipole(
                center=(0, 0, 0),
                polarization="Ex",
                source_time=td.GaussianPulse(
                    freq0=1e14,
                    fwidth=1e12,
                ),
            ),
        ),
        boundary_spec=td.BoundarySpec.pml(**pml_on_kwargs),
    )

    pml_boxes = sim2d._make_pml_boxes(normal_axis=normal_axis)

    for pml_box in pml_boxes:
        assert pml_box.size[normal_axis] > 0, "PML box has size of 0 in normal direction of 2D sim."


def test_allow_gain():
    """Test if simulation allows gain."""

    medium = td.Medium(permittivity=2.0)
    medium_gain = td.Medium(permittivity=2.0, allow_gain=True)
    medium_ani = td.AnisotropicMedium(xx=medium, yy=medium, zz=medium)
    medium_gain_ani = td.AnisotropicMedium(xx=medium, yy=medium_gain, zz=medium)

    # Test simulation medium
    sim = td.Simulation(
        size=(10, 10, 10), run_time=1e-12, medium=medium, grid_spec=td.GridSpec.uniform(dl=0.1)
    )
    assert not sim.allow_gain
    sim = sim.updated_copy(medium=medium_gain)
    assert sim.allow_gain

    # Test structure with anisotropic gain medium
    struct = td.Structure(geometry=td.Box(center=(0, 0, 0), size=(1, 1, 1)), medium=medium_ani)
    struct_gain = struct.updated_copy(medium=medium_gain_ani)
    sim = td.Simulation(
        size=(1, 1, 1),
        run_time=1e-12,
        medium=medium,
        grid_spec=td.GridSpec.uniform(dl=0.1),
        structures=(struct,),
    )
    assert not sim.allow_gain
    sim = sim.updated_copy(structures=(struct_gain,))
    assert sim.allow_gain


@pytest.mark.parametrize("z", [[5, 6], [5.5]])
@pytest.mark.parametrize("unstructured", [True, False])
def test_perturbed_mediums_copy(unstructured, z):
    # Non-dispersive
    pp_real = td.ParameterPerturbation(
        heat=td.LinearHeatPerturbation(
            coeff=-0.01,
            temperature_ref=300,
            temperature_range=(200, 500),
        ),
    )

    pp_complex = td.ParameterPerturbation(
        heat=td.LinearHeatPerturbation(
            coeff=0.01j,
            temperature_ref=300,
            temperature_range=(200, 500),
        ),
        charge=td.LinearChargePerturbation(
            electron_coeff=-1e-21,
            electron_ref=0,
            electron_range=(0, 1e20),
            hole_coeff=-2e-21,
            hole_ref=0,
            hole_range=(0, 0.5e20),
        ),
    )

    coords = {"x": [1, 2], "y": [3, 4], "z": z}
    temperature = td.SpatialDataArray(300 * np.ones((2, 2, len(z))), coords=coords)
    electron_density = td.SpatialDataArray(1e18 * np.ones((2, 2, len(z))), coords=coords)
    hole_density = td.SpatialDataArray(2e18 * np.ones((2, 2, len(z))), coords=coords)

    if unstructured:
        seed = 654
        temperature = cartesian_to_unstructured(temperature, seed=seed)
        electron_density = cartesian_to_unstructured(electron_density, seed=seed)
        hole_density = cartesian_to_unstructured(hole_density, seed=seed)

    pmed1 = td.PerturbationMedium(permittivity=3, permittivity_perturbation=pp_real)

    pmed2 = td.PerturbationPoleResidue(
        poles=[(1j, 3), (2j, 4)],
        poles_perturbation=[(None, pp_real), (pp_complex, None)],
    )

    struct = td.Structure(geometry=td.Box(center=(0, 0, 0), size=(1, 1, 1)), medium=pmed2)

    sim = td.Simulation(
        size=(1, 1, 1),
        run_time=1e-12,
        medium=pmed1,
        grid_spec=td.GridSpec.uniform(dl=0.1),
        structures=(struct,),
    )

    # no perturbations provided -> regular mediums
    new_sim = sim.perturbed_mediums_copy()

    assert isinstance(new_sim.medium, td.Medium)
    assert isinstance(new_sim.structures[0].medium, td.PoleResidue)

    # perturbations provided -> custom mediums
    new_sim = sim.perturbed_mediums_copy(temperature)
    new_sim = sim.perturbed_mediums_copy(temperature, None, hole_density)
    new_sim = sim.perturbed_mediums_copy(temperature, electron_density, hole_density)

    assert isinstance(new_sim.medium, td.CustomMedium)
    assert isinstance(new_sim.structures[0].medium, td.CustomPoleResidue)


def test_scene_from_scene():
    """Test .scene and .from_scene functionality."""

    scene = SIM_FULL.scene

    sim = td.Simulation.from_scene(
        scene=scene,
        **SIM_FULL.model_dump(exclude={"structures", "medium"}),
    )

    assert sim == SIM_FULL


def test_to_gds(tmp_path):
    sim = td.Simulation(
        size=(2.0, 2.0, 2.0),
        run_time=1e-12,
        structures=(
            td.Structure(
                geometry=td.Box(size=(1, 1, 1), center=(-1, 0, 0)),
                medium=td.Medium(permittivity=2.0),
            ),
            td.Structure(
                geometry=td.Sphere(radius=1.4, center=(1.0, 0.0, 1.0)),
                medium=td.Medium(permittivity=1.5),
            ),
            td.Structure(
                geometry=td.Cylinder(radius=1.4, length=2.0, center=(1.0, 0.0, -1.0), axis=1),
                medium=td.Medium(),
            ),
        ),
        sources=(
            td.PointDipole(
                center=(0, 0, 0),
                polarization="Ex",
                source_time=td.GaussianPulse(freq0=1e14, fwidth=1e12),
            ),
        ),
        monitors=(
            td.FieldMonitor(size=(0, 0, 0), center=(0, 0, 0), freqs=[1e12, 2e12], name="point"),
        ),
        boundary_spec=td.BoundarySpec(
            x=td.Boundary.pml(num_layers=20),
            y=td.Boundary.stable_pml(num_layers=30),
            z=td.Boundary.absorber(num_layers=100),
        ),
        shutoff=1e-6,
    )

    fname = str(tmp_path / "simulation_z.gds")
    sim.to_gds_file(
        fname, z=0, gds_layer_dtype_map={td.Medium(permittivity=2.0): (2, 1), td.Medium(): (1, 0)}
    )
    cell = gdstk.read_gds(fname).cells[0]
    assert cell.name == "MAIN"
    assert len(cell.polygons) >= 3
    areas = cell.area(True)
    assert (2, 1) in areas
    assert (1, 0) in areas
    assert (0, 0) in areas
    assert np.allclose(areas[(2, 1)], 0.5)
    assert np.allclose(areas[(1, 0)], 2.0 * (1.4**2 - 1) ** 0.5, atol=1e-2)
    assert np.allclose(areas[(0, 0)], 0.5 * np.pi * (1.4**2 - 1), atol=1e-2)

    fname = str(tmp_path / "simulation_y.gds")
    sim.to_gds_file(
        fname, y=0, gds_layer_dtype_map={td.Medium(permittivity=2.0): (2, 1), td.Medium(): (1, 0)}
    )
    cell = gdstk.read_gds(fname).cells[0]
    assert cell.name == "MAIN"
    assert len(cell.polygons) >= 3
    areas = cell.area(True)
    assert (2, 1) in areas
    assert (1, 0) in areas
    assert (0, 0) in areas
    assert np.allclose(areas[(2, 1)], 0.5)
    assert np.allclose(areas[(1, 0)], 0.25 * np.pi * 1.4**2, atol=1e-2)
    assert np.allclose(areas[(0, 0)], 0.25 * np.pi * 1.4**2, atol=1e-2)


@pytest.mark.parametrize("nz", [13, 1])
@pytest.mark.parametrize("unstructured", [True, False])
def test_sim_subsection(unstructured, nz):
    region = td.Box(size=(0.3, 0.5, 0.7), center=(0.1, 0.05, 0.02))
    region_xy = td.Box(size=(0.3, 0.5, 0), center=(0.1, 0.05, 0.02))

    sim_red = SIM_FULL.subsection(region=region)
    # Ensure that in this first test case the lumped element is safely excluded
    assert len(sim_red.lumped_elements) == 0
    assert sim_red.structures != SIM_FULL.structures

    sim_full_sym = SIM_FULL.updated_copy(
        boundary_spec=td.BoundarySpec(
            x=td.Boundary.pml(),
            y=td.Boundary.periodic(),
            z=td.Boundary.periodic(),
        ),
    )
    # Need to update BCs to be symmetrice when we include symmetries
    sim_red = sim_full_sym.subsection(
        region=region,
        symmetry=(1, 0, -1),
        monitors=tuple(mnt for mnt in SIM_FULL.monitors if not isinstance(mnt, td.ModeMonitor)),
    )
    assert sim_red.symmetry == (1, 0, -1)
    sim_red = SIM_FULL.subsection(
        region=region, boundary_spec=td.BoundarySpec.all_sides(td.Periodic())
    )
    sim_red = SIM_FULL.subsection(
        region=region,
        sources=(),
        grid_spec=td.GridSpec.uniform(dl=20),
    )
    assert len(sim_red.sources) == 0
    sim_red = SIM_FULL.subsection(region=region, monitors=())
    assert len(sim_red.monitors) == 0
    sim_red = SIM_FULL.subsection(region=region, remove_outside_structures=False)
    assert len(sim_red.structures) == len(SIM_FULL.structures)
    for strc_red, strc in zip(sim_red.structures, SIM_FULL.structures):
        if strc.medium.nonlinear_spec is None:
            assert strc == strc_red
    sim_red = SIM_FULL.subsection(region=region, remove_outside_custom_mediums=True)

    perm = td.SpatialDataArray(
        1 + np.random.random((11, 12, nz)),
        coords={
            "x": np.linspace(-0.51, 0.52, 11),
            "y": np.linspace(-1.02, 1.04, 12),
            "z": np.linspace(-1.51, 1.51, nz),
        },
    )

    if unstructured:
        perm = cartesian_to_unstructured(perm, seed=523)

    fine_custom_medium = td.CustomMedium(permittivity=perm)

    sim = SIM_FULL.updated_copy(
        structures=(
            td.Structure(
                geometry=td.Box(size=(1, 2, 3)),
                medium=fine_custom_medium,
            ),
        ),
        medium=fine_custom_medium,
    )
    sim_red = sim.subsection(region=region, remove_outside_custom_mediums=True)

    # check automatic symmetry expansion
    sim_sym = sim_full_sym.updated_copy(
        symmetry=(-1, 0, 1),
        sources=tuple(src for src in SIM_FULL.sources if not isinstance(src, td.TFSF)),
    )
    sim_red = sim_sym.subsection(region=region)
    assert np.allclose(sim_red.center, (0, 0.05, 0.0))

    # check grid is preserved when requested
    sim_red = SIM_FULL.subsection(
        region=region,
        grid_spec="identical",
        boundary_spec=td.BoundarySpec.all_sides(td.Periodic()),
    )
    grids_1d = SIM_FULL.grid.boundaries
    grids_1d_red = sim_red.grid.boundaries
    tol = 1e-8
    for full_grid, red_grid in zip(
        [grids_1d.x, grids_1d.y, grids_1d.z], [grids_1d_red.x, grids_1d_red.y, grids_1d_red.z]
    ):
        # find index into full grid at which reduced grid is starting
        start = red_grid[0]
        ind = np.argmax(np.logical_and(full_grid >= start - tol, full_grid <= start + tol))
        # compare
        assert np.allclose(red_grid, full_grid[ind : ind + len(red_grid)])

    subsection_monitors = (
        mnt
        for mnt in SIM_FULL_FIELD_PROJECTION.monitors
        if region_xy.intersects(mnt)
        and getattr(mnt, "far_field_approx", True)  # unsupported in 2d
        and not isinstance(
            mnt, (td.FieldProjectionCartesianMonitor, td.FieldProjectionKSpaceMonitor)
        )
    )
    sim_red = SIM_FULL_FIELD_PROJECTION.subsection(
        region=region_xy,
        grid_spec="identical",
        boundary_spec=td.BoundarySpec.all_sides(td.Periodic()),
        # Set theta to 'pi/2' for 2D simulation in the x-y plane
        monitors=tuple(
            mnt.updated_copy(theta=np.pi / 2)
            if isinstance(mnt, td.FieldProjectionAngleMonitor)
            else mnt
            for mnt in subsection_monitors
        ),
    )
    assert sim_red.size[2] == 0
    assert isinstance(sim_red.boundary_spec.z.minus, td.Periodic)
    assert isinstance(sim_red.boundary_spec.z.plus, td.Periodic)

    # check behavior for zero-size dimensions
    sim_2d = SIM.updated_copy(
        size=(SIM.size[0], 0, SIM.size[2]),
        boundary_spec=td.BoundarySpec.pml(x=True, z=True),
    )
    sim_2d_red = sim_2d.subsection(
        region=region, remove_outside_structures=True, remove_outside_custom_mediums=True
    )
    assert sim_2d_red.size[1] == 0

    sim_red = sim_2d.subsection(
        region=region_xy,
        grid_spec="identical",
        boundary_spec=td.BoundarySpec.all_sides(td.Periodic()),
    )
    assert sim_red.size[1] == 0
    assert sim_red.size[2] == 0
    assert isinstance(sim_red.boundary_spec.y.minus, td.Periodic)
    assert isinstance(sim_red.boundary_spec.y.plus, td.Periodic)
    assert isinstance(sim_red.boundary_spec.z.minus, td.Periodic)
    assert isinstance(sim_red.boundary_spec.z.plus, td.Periodic)

    sim_1d = SIM.updated_copy(
        size=(0, SIM.size[1], 0),
        boundary_spec=td.BoundarySpec.pml(y=True),
    )
    sim_1d_red = sim_1d.subsection(
        region=region, remove_outside_structures=True, remove_outside_custom_mediums=True
    )
    assert sim_1d_red.size[0] == 0
    assert sim_1d_red.size[2] == 0


def _make_auto_grid_subsection_sim():
    keep_override = td.MeshOverrideStructure(
        geometry=td.Box(center=(10, 0, 0), size=(1, 1, 1)),
        dl=(None, 0.1, 0.1),
    )
    drop_override = td.MeshOverrideStructure(
        geometry=td.Box(center=(10, 10, 0), size=(1, 1, 1)),
        dl=(0.1, 0.1, 0.1),
    )
    layer_spec = td.LayerRefinementSpec.from_bounds(
        axis=2,
        rmin=(1.5, -1.0, -4.0),
        rmax=(3.5, 1.0, 4.0),
    )
    simulation = td.Simulation(
        center=(0, 0, 0),
        size=(12, 12, 12),
        run_time=1e-12,
        structures=(
            td.Structure(
                geometry=td.Box(center=(0, 0, 0), size=(2, 2, 2)),
                medium=td.Medium(permittivity=2.0),
            ),
        ),
        grid_spec=td.GridSpec.auto(
            wavelength=1.0,
            override_structures=(keep_override, drop_override),
            layer_refinement_specs=(layer_spec,),
            snapping_points=((0.0, 0.0, 5.0), (10.0, 0.0, 5.0), (10.0, 10.0, 5.0)),
        ),
    )
    region = td.Box(center=(0, 0, 0), size=(4.0, 6.0, td.inf))
    return simulation, region, layer_spec


def test_sim_subsection_filters_auto_grid_entities():
    simulation, region, layer_spec = _make_auto_grid_subsection_sim()

    sim_full = simulation.subsection(region=region, remove_outside_grid_spec=False)
    assert len(sim_full.grid_spec.override_structures) == 2
    assert len(sim_full.grid_spec.layer_refinement_specs) == 1
    assert len(sim_full.grid_spec.snapping_points) == 3

    sim_red = simulation.subsection(region=region, remove_outside_grid_spec=True)
    assert len(sim_red.grid_spec.override_structures) == 2
    assert sim_red.grid_spec.override_structures[0].dl == (None, 0.1, 0.1)
    assert sim_red.grid_spec.override_structures[1].dl == (None, None, 0.1)

    clipped_spec = sim_red.grid_spec.layer_refinement_specs[0]
    assert clipped_spec.bounds[0][0] == pytest.approx(layer_spec.bounds[0][0])
    assert clipped_spec.bounds[1][0] == pytest.approx(region.bounds[1][0])
    assert clipped_spec.bounds[0][2] == pytest.approx(layer_spec.bounds[0][2])
    assert clipped_spec.bounds[1][2] == pytest.approx(layer_spec.bounds[1][2])

    assert sim_red.grid_spec.snapping_points == (
        (0.0, 0.0, 5.0),
        (None, 0.0, 5.0),
        (None, None, 5.0),
    )


def test_sim_subsection_prunes_grouped_geometries():
    region = td.Box(center=(0, 0, 0), size=(2, 2, 2))
    geometry_in = td.Box(center=(0, 0, 0), size=(1, 1, 1))
    geometry_out = td.Box(center=(4, 0, 0), size=(1, 1, 1))
    grouped_geometry = td.GeometryGroup(geometries=(geometry_in, geometry_out))
    structure = td.Structure(geometry=grouped_geometry, medium=td.Medium(permittivity=2.0))
    sim = td.Simulation(
        center=(0, 0, 0),
        size=(10, 10, 10),
        run_time=1e-12,
        grid_spec=td.GridSpec.uniform(dl=0.5),
        structures=(structure,),
    )

    sim_red = sim.subsection(region=region)
    assert len(sim_red.structures) == 1
    assert sim_red.structures[0].geometry == geometry_in

    sim_full = sim.subsection(region=region, remove_outside_structures=False)
    assert sim_full.structures[0].geometry == grouped_geometry


def test_sim_subsection_preserves_geometry_array_structures():
    region = td.Box(center=(1, 0, 0), size=(4, 2, 2))
    geometry_array = td.GeometryArray(
        geometry=td.Box(size=(1, 1, 1)),
        offsets=[[0, 0, 0], [2, 0, 0], [6, 0, 0]],
    )
    structure = td.Structure(geometry=geometry_array, medium=td.Medium(permittivity=2.0))
    sim = td.Simulation(
        center=(0, 0, 0),
        size=(12, 12, 12),
        run_time=1e-12,
        grid_spec=td.GridSpec.uniform(dl=0.5),
        structures=(structure,),
    )

    sim_red = sim.subsection(region=region)

    assert len(sim_red.structures) == 1
    assert isinstance(sim_red.structures[0].geometry, td.GeometryArray)
    assert sim_red.structures[0].geometry.offsets == (
        (0.0, 0.0, 0.0),
        (2.0, 0.0, 0.0),
    )


def test_2d_material_subdivision():
    units = 1e3
    plane_pos = 1.0 * units
    plane_width = 1.0 * units
    plane_height = 1.0 * units

    two = td.Medium(permittivity=2.0)
    three = td.Medium(permittivity=3.0)
    four = td.Medium(permittivity=4.0)
    five = td.Medium(permittivity=5.0)

    # ~Copper
    conductor = td.Medium(conductivity=5.8e7)

    freq_start = 1e1
    freq_stop = 10e9
    freq0 = (freq_start + freq_stop) / 2
    wavelength0 = td.C_0 / freq0

    # Setup simulation size
    size_sim = [
        4 * abs(plane_pos),
        4 * abs(plane_width),
        4 * abs(plane_height),
    ]
    center_sim = [plane_pos, 0, 0]

    face = td.Structure(
        geometry=td.Box(
            center=[plane_pos / 2, 0, 0],
            size=[plane_pos, 0.9 * plane_width, 0.9 * plane_height],
        ),
        medium=two,
    )

    left_center = [plane_pos / 2, -0.25 * plane_width, 0.25 * plane_height]
    left_top = td.Structure(
        geometry=td.Box(
            center=left_center,
            size=[plane_pos, 0.2 * plane_width, 0.2 * plane_height],
        ),
        medium=three,
    )
    right_center = [plane_pos / 2, 0.25 * plane_width, 0.25 * plane_height]
    right_top = td.Structure(
        geometry=td.Box(
            center=right_center,
            size=[plane_pos, 0.2 * plane_width, 0.2 * plane_height],
        ),
        medium=four,
    )
    # This object fully extrudes through the 2d material
    bottom_center = [plane_pos, 0, -0.25 * plane_height]
    bottom = td.Structure(
        geometry=td.Box(
            center=bottom_center,
            size=[1.8 * plane_pos, 0.5 * plane_width, 0.3 * plane_height],
        ),
        medium=five,
    )

    med_2d = td.Medium2D(ss=conductor, tt=conductor)
    plane_size = [0, 1.5 * plane_width, 1.5 * plane_height]
    plane_material = td.Structure(
        geometry=td.Box(size=plane_size, center=[plane_pos, 0, 0]), medium=med_2d, name="plane"
    )

    structures = [face, left_top, right_top, bottom, plane_material]

    uni_grid = td.UniformGrid(dl=wavelength0 / 1000)

    sim_td = td.Simulation(
        center=center_sim,
        size=size_sim,
        grid_spec=td.GridSpec(grid_x=uni_grid, grid_y=uni_grid, grid_z=uni_grid),
        structures=structures,
        sources=(),
        monitors=(),
        run_time=1e-12,
    )

    _ = sim_td._finalized

    volume = td.Box(center=(plane_pos, 0, 0), size=(0, 2 * plane_width, 2 * plane_height))
    eps_centers = sim_td.epsilon(box=volume, freq=freq0, coord_key="Ey")
    # Plot should give a smiley face
    # f, (ax1, ax2) = plt.subplots(1, 2, tight_layout=True, figsize=(10, 4))
    # eps_centers.real.plot(x="y", y="z", cmap="Greys", ax=ax1)
    # eps_centers.imag.plot(x="y", y="z", cmap="Greys", ax=ax2)

    # Test some positions to make sure the correct volumetric permittivity was computed. All positions should take on the same volumetric version of the conductivity
    assert np.isclose(
        np.real(eps_centers.sel(x=plane_pos, y=0, z=-0.4 * plane_size[2], method="nearest").values),
        1,
    )
    assert np.isclose(
        np.imag(eps_centers.sel(x=plane_pos, y=0, z=-0.4 * plane_size[2], method="nearest").values),
        3492562622979.975,
    )

    assert np.isclose(np.real(eps_centers.sel(x=plane_pos, y=0, z=0, method="nearest").values), 1.5)
    assert np.isclose(
        np.imag(eps_centers.sel(x=plane_pos, y=0, z=0, method="nearest").values), 3492562622979.975
    )

    assert np.isclose(
        np.real(
            eps_centers.sel(
                x=plane_pos, y=left_center[1], z=left_center[2], method="nearest"
            ).values
        ),
        2,
    )
    assert np.isclose(
        np.imag(
            eps_centers.sel(
                x=plane_pos, y=left_center[1], z=left_center[2], method="nearest"
            ).values
        ),
        3492562622979.975,
    )

    assert np.isclose(
        np.real(
            eps_centers.sel(
                x=plane_pos, y=right_center[1], z=right_center[2], method="nearest"
            ).values
        ),
        2.5,
    )
    assert np.isclose(
        np.imag(
            eps_centers.sel(
                x=plane_pos, y=right_center[1], z=right_center[2], method="nearest"
            ).values
        ),
        3492562622979.975,
    )
    # In this position the substrate and superstrate are the same so the average value should be the original
    assert np.isclose(
        np.real(eps_centers.sel(x=plane_pos, y=0, z=bottom_center[2], method="nearest").values), 5.0
    )
    assert np.isclose(
        np.imag(eps_centers.sel(x=plane_pos, y=0, z=bottom_center[2], method="nearest").values),
        3492562622979.975,
    )


def test_advanced_material_intersection():
    src_time = td.GaussianPulse(freq0=td.C_0, fwidth=0.1e12)
    source = td.PlaneWave(center=(0, 0, -1.9), size=[1, 1, 0], source_time=src_time, direction="+")

    # custom
    Nx, Ny, Nz = 10, 9, 8
    X = np.linspace(-1, 1, Nx)
    Y = np.linspace(-1, 1, Ny)
    Z = np.linspace(-1, 1, Nz)
    data = np.ones((Nx, Ny, Nz, 1))
    eps_diagonal_data = td.ScalarFieldDataArray(
        data, coords={"x": X, "y": Y, "z": Z, "f": [td.C_0]}
    )
    eps_components = {f"eps_{d}{d}": eps_diagonal_data for d in "xyz"}
    eps_dataset = td.PermittivityDataset(**eps_components)
    custom_medium = td.CustomMedium(eps_dataset=eps_dataset, name="my_medium")

    # nonlinear
    nonlinear_medium = td.Medium(
        nonlinear_spec=td.NonlinearSpec(models=[td.KerrNonlinearity(n2=1)])
    )

    # time-modulated
    FREQ_MODULATE = 1e12
    AMP_TIME = 1.1
    PHASE_TIME = 0
    CW = td.ContinuousWaveTimeModulation(freq0=FREQ_MODULATE, amplitude=AMP_TIME, phase=PHASE_TIME)
    ST = td.SpaceTimeModulation(
        time_modulation=CW,
    )
    MODULATION_SPEC = td.ModulationSpec()
    modulation_spec = MODULATION_SPEC.updated_copy(permittivity=ST)
    time_modulated_medium = td.Medium(permittivity=2, modulation_spec=modulation_spec)

    # fully anisotropic
    perm_diag = [[1, 0, 0], [0, 2, 0], [0, 0, 3]]
    cond_diag = [[4, 0, 0], [0, 5, 0], [0, 0, 6]]

    rot = td.RotationAroundAxis(axis=(1, 2, 3), angle=1.23)
    rot2 = td.RotationAroundAxis(axis=(3, 2, 1), angle=1.23)

    perm = rot.rotate_tensor(perm_diag)
    cond = rot.rotate_tensor(cond_diag)
    _ = rot2.rotate_tensor(cond_diag)

    fully_anisotropic_medium = td.FullyAnisotropicMedium(permittivity=perm, conductivity=cond)

    # compatible and incompatible media
    media = [custom_medium, nonlinear_medium, time_modulated_medium, fully_anisotropic_medium]
    compatible_pairs = [(custom_medium, fully_anisotropic_medium)]
    for medium in media:
        compatible_pairs.append((medium, medium))
    incompatible_pairs = [(custom_medium, med) for med in media[1:3]]
    incompatible_pairs += [(nonlinear_medium, med) for med in media[2:]]
    incompatible_pairs += [(time_modulated_medium, fully_anisotropic_medium)]
    # check in other order
    compatible_pairs += [(pair[1], pair[0]) for pair in compatible_pairs if pair[0] != pair[1]]
    incompatible_pairs += [(pair[1], pair[0]) for pair in incompatible_pairs if pair[0] != pair[1]]

    # base sim
    sim = td.Simulation(
        size=(4.0, 4.0, 4.0),
        grid_spec=td.GridSpec.auto(wavelength=1.0),
        run_time=1e-12,
        sources=(source,),
        structures=(),
    )

    for pair in compatible_pairs:
        struct1 = td.Structure(geometry=td.Box(size=(1, 1, 1), center=(0, 0, 0.5)), medium=pair[0])
        struct2 = td.Structure(geometry=td.Box(size=(1, 1, 1), center=(0, 0, -0.5)), medium=pair[1])
        # this pair can intersect
        sim = sim.updated_copy(structures=(struct1, struct2))

    for pair in incompatible_pairs:
        struct1 = td.Structure(geometry=td.Box(size=(1, 1, 1), center=(0, 0, 0.5)), medium=pair[0])
        struct2 = td.Structure(geometry=td.Box(size=(1, 1, 1), center=(0, 0, -0.5)), medium=pair[1])
        # this pair cannot intersect
        with pytest.raises(ValidationError):
            sim = sim.updated_copy(structures=(struct1, struct2))

    for pair in incompatible_pairs:
        struct1 = td.Structure(geometry=td.Box(size=(1, 1, 1), center=(0, 0, 0.75)), medium=pair[0])
        struct2 = td.Structure(
            geometry=td.Box(size=(1, 1, 1), center=(0, 0, -0.75)), medium=pair[1]
        )
        # it's ok if these are both present as long as they don't intersect
        sim = sim.updated_copy(structures=(struct1, struct2))


def test_num_lumped_elements(monkeypatch):
    """Make sure we error if too many lumped elements supplied."""
    monkeypatch.setattr(simulation, "MAX_NUM_MEDIUMS", TEST_MAX_NUM_MEDIUMS)
    resistor = td.LumpedResistor(
        size=(0, 1, 2), center=(0, 0, 0), name="R1", voltage_axis=2, resistance=75
    )
    grid_spec = td.GridSpec.auto(wavelength=1.0)

    _ = td.Simulation(
        size=(5, 5, 5),
        grid_spec=grid_spec,
        structures=(),
        lumped_elements=(resistor,) * TEST_MAX_NUM_MEDIUMS,
        run_time=1e-12,
    )
    with pytest.raises(ValidationError):
        _ = td.Simulation(
            size=(5, 5, 5),
            grid_spec=grid_spec,
            structures=(),
            lumped_elements=(resistor,) * (TEST_MAX_NUM_MEDIUMS + 1),
            run_time=1e-12,
        )


def test_validate_lumped_elements():
    resistor = td.LumpedResistor(
        size=(0, 1, 2), center=(0, 0, 0), name="R1", voltage_axis=2, resistance=75
    )

    _ = td.Simulation(
        size=(1, 2, 3),
        run_time=1e-12,
        grid_spec=td.GridSpec.uniform(dl=0.1),
        lumped_elements=(resistor,),
    )
    # error for 1D/2D simulation with lumped elements
    with pytest.raises(ValidationError):
        td.Simulation(
            size=(1, 0, 3),
            run_time=1e-12,
            grid_spec=td.GridSpec.uniform(dl=0.1),
            lumped_elements=(resistor,),
        )

    with pytest.raises(ValidationError):
        td.Simulation(
            size=(1, 0, 0),
            run_time=1e-12,
            grid_spec=td.GridSpec.uniform(dl=0.1),
            lumped_elements=(resistor,),
        )


def test_suggested_mesh_overrides():
    resistor = td.LumpedResistor(
        size=(0, 1, 2), center=(0, 0, 0), name="R1", voltage_axis=2, resistance=75
    )
    sim = td.Simulation(
        size=(1, 2, 3),
        run_time=1e-12,
        grid_spec=td.GridSpec.auto(wavelength=1),
        lumped_elements=(resistor,),
    )
    assert len(sim.internal_override_structures) == 1
    assert len(sim.internal_snapping_points) == 3

    coax_resistor = td.CoaxialLumpedResistor(
        resistance=50.0,
        center=[0, 0, 0],
        outer_diameter=2,
        inner_diameter=0.5,
        normal_axis=0,
        name="R",
    )

    sim = sim.updated_copy(
        lumped_elements=(coax_resistor,),
    )
    assert len(sim.internal_override_structures) == 1
    assert len(sim.internal_snapping_points) == 1


def test_run_time_spec():
    run_time_spec = td.RunTimeSpec(quality_factor=3.0)

    sim = SIM_FULL.updated_copy(run_time=run_time_spec)

    assert sim._run_time > 0


def test_run_time_spec_lossy_metal():
    freq0 = 1e9
    run_time_spec = td.RunTimeSpec(quality_factor=3.0)
    src_time = td.GaussianPulse(freq0=freq0, fwidth=freq0 * 0.5)
    source = td.PlaneWave(
        center=(0, 0, -0.5e3), size=[td.inf, td.inf, 0], source_time=src_time, direction="+"
    )
    box = td.Structure(
        geometry=td.Box(size=(0.1e3, 0.1e3, 0.1e3)),
        medium=td.LossyMetalMedium(conductivity=50, frequency_range=(freq0 * 0.5, freq0 * 1.5)),
    )
    sim = td.Simulation(
        run_time=run_time_spec,
        size=(1e4, 1e4, 2e3),
        sources=(source,),
        structures=(box,),
    )
    assert max(sim.get_refractive_indices(freq0)) < 2
    # if lossymetal is not handled properly, _run_time can approach 1e-6
    assert sim._run_time < 5e-8


def test_validate_low_num_cells_in_mode_objects():
    pulse = td.GaussianPulse(freq0=200e12, fwidth=20e12)
    mode_spec = td.ModeSpec(target_neff=2.0)
    mode_source = td.ModeSource(
        center=(0, 0, 0),
        size=(1, 0.02, 0.0),
        source_time=pulse,
        name="Small Source",
        mode_spec=mode_spec,
        mode_index=1,
        direction="+",
    )

    sim = SIM.updated_copy(sources=(mode_source,))

    # check with mode source that is too small
    with pytest.raises(SetupError):
        sim._validate_num_cells_in_mode_objects()

    sim_2d_size = list(sim.size)
    sim_2d_size[1] = 0
    # Should be fine if the simulation is 2D
    sim2d = td.Simulation(
        size=sim_2d_size,
        run_time=1e-12,
        grid_spec=td.GridSpec(wavelength=1.0),
        sources=(mode_source,),
        boundary_spec=td.BoundarySpec(
            x=td.Boundary.pml(num_layers=6),
            y=td.Boundary.pec(),
            z=td.Boundary.pml(num_layers=6),
        ),
    )
    sim2d._validate_num_cells_in_mode_objects()

    # Now try with a mode monitor
    mode_monitor = td.ModeMonitor(
        center=(0, 0, 0),
        size=(1, 0.02, 0.0),
        name="Small Monitor",
        mode_spec=mode_spec,
        freqs=[1e12],
    )
    sim = SIM.updated_copy(monitors=(mode_monitor,))
    with pytest.raises(SetupError):
        sim._validate_num_cells_in_mode_objects()


def test_validate_sources_monitors_in_bounds():
    pulse = td.GaussianPulse(freq0=200e12, fwidth=20e12)
    mode_source = td.ModeSource(
        center=(0, -1, 0),
        size=(1, 0, 1),
        source_time=pulse,
        direction="+",
    )
    mode_monitor = td.ModeMonitor(
        center=(0, 1, 0),
        size=(1, 0, 1),
        freqs=[1e12],
        name="test_in_bounds",
        mode_spec=td.ModeSpec(),
    )

    # check that a source at y- simulation domain edge errors
    with pytest.raises(ValidationError):
        sim = td.Simulation(
            size=(2, 2, 2),
            run_time=1e-12,
            grid_spec=td.GridSpec(wavelength=1.0),
            sources=(mode_source,),
        )
    # check that a monitor at y+ simulation domain edge errors
    with pytest.raises(ValidationError):
        sim = td.Simulation(
            size=(2, 2, 2),
            run_time=1e-12,
            grid_spec=td.GridSpec(wavelength=1.0),
            monitors=(mode_monitor,),
        )


def test_mode_pml_warning():
    sim_size = (3, 3, 3)
    lambda0 = 1.55
    freq0 = td.C_0 / lambda0
    si = td.material_library["cSi"]["Li1993_293K"]
    sio2 = td.material_library["SiO2"]["Horiba"]
    wg = td.Structure(geometry=td.Box(size=(0.22, 0.5, td.inf)), medium=si)
    mode_plane = td.Box(size=(2, 2, 0))
    mode_spec = td.ModeSpec(num_pml=(22, 22))
    grid_spec = td.GridSpec.auto(wavelength=lambda0, min_steps_per_wvl=30)
    symmetry = (0, 0, 0)
    with AssertLogLevel(None):
        sim = td.Simulation(
            size=sim_size,
            medium=sio2,
            structures=(wg,),
            grid_spec=grid_spec,
            run_time=1e-30,
            monitors=(
                td.ModeSolverMonitor(
                    size=(2, 2, 0),
                    name="mode",
                    freqs=[freq0],
                    mode_spec=mode_spec.updated_copy(num_pml=(10, 10)),
                ),
            ),
            symmetry=symmetry,
        )
    with AssertLogLevel("WARNING", contains_str="covers more than"):
        sim = td.Simulation(
            size=sim_size,
            medium=sio2,
            structures=(wg,),
            grid_spec=grid_spec,
            run_time=1e-30,
            monitors=(
                td.ModeSolverMonitor(
                    size=(2, 2, 0), name="mode", freqs=[freq0], mode_spec=mode_spec
                ),
            ),
            symmetry=symmetry,
        )
    with AssertLogLevel("WARNING", contains_str="covers more than"):
        sim = td.Simulation(
            size=sim_size,
            medium=sio2,
            structures=(wg,),
            grid_spec=grid_spec,
            run_time=1e-30,
            sources=(
                td.ModeSource(
                    size=(2, 2, 0),
                    direction="+",
                    source_time=td.GaussianPulse(freq0=freq0, fwidth=0.1 * freq0),
                    mode_spec=mode_spec,
                ),
            ),
            symmetry=symmetry,
        )
    with AssertLogLevel("WARNING", contains_str="covers more than"):
        mode_solver = ModeSolver(
            simulation=sim, plane=mode_plane, mode_spec=mode_spec, freqs=[freq0]
        )
        size = mode_solver._mode_plane_size(simulation=sim, plane=mode_plane)
        size_no_pml = mode_solver._mode_plane_size_no_pml(
            simulation=sim, plane=mode_plane, mode_spec=mode_spec
        )
        for i in [0, 1]:
            assert size_no_pml[i] / size[i] < 0.5
    with AssertLogLevel("WARNING", contains_str="covers more than"):
        mode_sim = td.ModeSimulation(
            size=sim_size,
            medium=sio2,
            structures=(wg,),
            grid_spec=grid_spec,
            plane=mode_plane,
            mode_spec=mode_spec,
            freqs=[freq0],
        )


def test_fixed_angle_sim():
    wvl_um = 1.0
    freq0 = td.C_0 / wvl_um
    fwidth = freq0 / 5

    freqs = freq0 + 0.5 * fwidth * np.linspace(-1, 1, 11)
    med = td.Medium(permittivity=5)
    sphere = td.Structure(
        geometry=td.Sphere(radius=0.5),
        medium=med,
    )
    flux_r_mnt = td.FluxMonitor(
        center=(-1, 0, 0), size=(0, td.inf, td.inf), freqs=freqs, name="flux_r"
    )
    source = td.PlaneWave(
        angle_phi=np.pi / 6,
        angle_theta=np.pi / 5,
        angular_spec=td.FixedAngleSpec(),
        direction="+",
        center=(-0.9, 0, 0),
        size=(0, td.inf, td.inf),
        pol_angle=np.pi / 4,
        source_time=td.GaussianPulse(freq0=freq0, fwidth=fwidth),
    )
    sim_size = (2.2, 2.2, 2.2)
    sim = td.Simulation(
        structures=(sphere,),
        sources=(source,),
        monitors=(flux_r_mnt,),
        size=sim_size,
        grid_spec=td.GridSpec.auto(min_steps_per_wvl=15),
        boundary_spec=td.BoundarySpec(
            x=td.Boundary.absorber(), y=td.Boundary.periodic(), z=td.Boundary.periodic()
        ),
        run_time=10 / fwidth,
    )

    assert sim._is_fixed_angle

    with pytest.raises(ValidationError):
        _ = sim.updated_copy(
            boundary_spec=td.BoundarySpec(
                x=td.Boundary.pml(),
                y=td.Boundary.bloch_from_source(source=source, axis=1, domain_size=2.2),
                z=td.Boundary.bloch_from_source(source=source, axis=2, domain_size=2.2),
            )
        )

    with pytest.raises(KeyError):
        _ = sim.updated_copy(med=td.Medium(conductivity=0.001))

    anisotropic_med = td.FullyAnisotropicMedium(permittivity=[[2, 0, 0], [0, 1, 0], [0, 0, 3]])
    with pytest.raises(ValidationError):
        _ = sim.updated_copy(structures=(sphere.updated_copy(medium=anisotropic_med),))

    with pytest.raises(ValidationError):
        _ = sim.updated_copy(sources=(source, source))

    with pytest.raises(ValidationError):
        _ = sim.updated_copy(
            structures=(sphere.updated_copy(medium=td.Medium(conductivity=-0.1, allow_gain=True)),)
        )

    with pytest.raises(ValidationError):
        _ = sim.updated_copy(monitors=(td.FieldTimeMonitor(size=[td.inf, td.inf, 0], name="time"),))

    with pytest.raises(ValidationError):
        _ = sim.updated_copy(monitors=(td.FluxTimeMonitor(size=[td.inf, td.inf, 0], name="time"),))

    nonlinear_med = td.Medium(
        permittivity=3,
        nonlinear_spec=td.NonlinearSpec(
            models=[
                td.KerrNonlinearity(n2=1, n0=1),
            ],
            num_iters=20,
        ),
    )
    with pytest.raises(ValidationError):
        _ = sim.updated_copy(structures=(sphere.updated_copy(medium=nonlinear_med),))

    time_modulated_med = td.Medium(
        permittivity=2,
        modulation_spec=td.ModulationSpec(
            permittivity=td.SpaceTimeModulation(
                time_modulation=td.ContinuousWaveTimeModulation(freq0=td.C_0, amplitude=1, phase=0),
            )
        ),
    )
    with pytest.raises(ValidationError):
        _ = sim.updated_copy(structures=(sphere.updated_copy(medium=time_modulated_med),))


def test_sim_volumetric_structures_with_lumped_elements(tmp_path):
    """Test volumetric equivalent of lumped elements."""
    grid_dl = 0.1
    center = (-2, 0, 0)
    network = td.RLCNetwork(resistance=42, capacitance=5e-12, network_topology="parallel")
    resistor = td.LumpedResistor(
        center=center, size=(0, 1, 2), name="resistor", voltage_axis=1, resistance=54
    )
    coax_resistor = td.CoaxialLumpedResistor(
        center=center,
        outer_diameter=3,
        inner_diameter=0.5,
        name="coax_resistor",
        normal_axis=0,
        resistance=54,
    )
    linear_element = td.LinearLumpedElement(
        center=center, size=(0, 1, 2), name="linear_element", voltage_axis=1, network=network
    )
    src = td.UniformCurrentSource(
        source_time=td.GaussianPulse(freq0=1.5e14, fwidth=0.5e14),
        size=(0, 0, 0),
        polarization="Ex",
        current_amplitude_definition="total",
    )
    substrate = td.Structure(
        geometry=td.Box(size=(4, td.inf, td.inf)), medium=td.Medium(permittivity=3.5)
    )
    for element in [resistor, coax_resistor, linear_element]:
        sim = td.Simulation(
            size=(10, 10, 10),
            structures=(substrate,),
            sources=(src,),
            boundary_spec=td.BoundarySpec(
                x=td.Boundary.pml(num_layers=6),
                y=td.Boundary.pml(num_layers=6),
                z=td.Boundary.pml(num_layers=6),
            ),
            lumped_elements=(element,),
            grid_spec=td.GridSpec.uniform(dl=grid_dl),
            run_time=1e-12,
        )
        vol_structures = sim.volumetric_structures
        assert len(vol_structures) == 2
        assert np.isclose(vol_structures[1].geometry.bounding_box.size[0], 0, rtol=RTOL)


def test_finalized_sim_with_lumped_element_avoids_2d_medium_warning():
    """Converted lumped-element structures should not trigger the zero-thickness warning."""
    src = td.UniformCurrentSource(
        source_time=td.GaussianPulse(freq0=1.5e14, fwidth=0.5e14),
        size=(0, 0, 0),
        polarization="Ex",
        current_amplitude_definition="total",
    )
    resistor = td.LumpedResistor(
        center=(-2, 0, 0),
        size=(0, 1, 2),
        name="resistor",
        voltage_axis=1,
        resistance=54,
    )
    sim = td.Simulation(
        size=(10, 10, 10),
        sources=(src,),
        boundary_spec=td.BoundarySpec.all_sides(boundary=td.PML(num_layers=6)),
        lumped_elements=(resistor,),
        grid_spec=td.GridSpec.uniform(dl=0.1),
        run_time=1e-12,
    )

    with AssertLogStr("WARNING", excludes_str="zero size along dimensions"):
        finalized = sim._finalized

    assert isinstance(finalized.structures[0].medium, AnisotropicMediumFromMedium2D)


def test_create_sim_multiphysics():
    s = td.Simulation(
        run_time=1e-12,
        size=(10, 10, 10),
        grid_spec=td.GridSpec(wavelength=1.0),
        medium=td.Medium(permittivity=1.0),
        structures=(
            td.Structure(
                geometry=td.Box(size=(1, 1, 1), center=(-1, 0.5, 0.5)),
                medium=td.MultiPhysicsMedium(
                    optical=td.Medium(permittivity=2.0),
                    charge=td.ChargeInsulatorMedium(permittivity=2),
                    name="SiO2",
                ),
            ),
        ),
    )


def test_create_sim_multiphysics_with_incompatibilities():
    modulated = td.Medium(
        permittivity=2,
        modulation_spec=td.ModulationSpec(
            permittivity=td.SpaceTimeModulation(
                time_modulation=td.ContinuousWaveTimeModulation(freq0=1e12, amplitude=1.1, phase=0),
            )
        ),
    )
    assert modulated._has_incompatibilities

    nonlinear = td.Medium(
        nonlinear_spec=td.NonlinearSpec(
            models=[
                td.NonlinearSusceptibility(chi3=1.5),
                td.TwoPhotonAbsorption(beta=1, sigma=1, tau=1, e_e=1, e_h=0.8, c_e=1, c_h=1),
                td.KerrNonlinearity(n2=1),
            ],
            num_iters=20,
        )
    )
    with pytest.raises(ValidationError):
        s = td.Simulation(
            run_time=1e-12,
            size=(10, 10, 10),
            grid_spec=td.GridSpec(wavelength=1.0),
            medium=td.Medium(permittivity=1.0),
            structures=(
                td.Structure(
                    geometry=td.Box(size=(1, 1, 1), center=(-1, 0.5, 0.5)),
                    medium=nonlinear,
                ),
                td.Structure(
                    geometry=td.Box(size=(1, 1, 1), center=(-1, 0.5, 0.5)),
                    medium=td.MultiPhysicsMedium(
                        optical=modulated,
                        charge=td.ChargeInsulatorMedium(permittivity=2),
                        name="SiO2",
                    ),
                ),
            ),
        )


def test_messages_contain_object_names():
    """Make sure that errors and warnings contain the name of the object."""
    # Note: This function currently tests for out-of-bounds errors and warnings.
    # Create an empty simulation.
    sim = td.Simulation(
        size=(1, 1, 1),
        grid_spec=td.GridSpec.auto(wavelength=4),
        run_time=1e-12,
    )

    # Test 1) Create a structure lying outside the simulation boundary.
    # Check that a warning message is generated containing the structure's `name`.
    name = "structure_123"
    structure = td.Structure(
        name=name,
        geometry=td.Box(center=(1.0, 0.0, 0.0), size=(0.5, 0.5, 0.5)),
        medium=td.Medium(permittivity=2.0),
    )
    with AssertLogLevel("WARNING", contains_str=name):
        _ = sim.updated_copy(structures=[structure])

    # Test 2) Create a source lying outside the simulation boundary.
    # Check that an error message is generated containing the source's `name`.
    name = "source_123"
    source = td.UniformCurrentSource(
        name=name,
        center=(0, -1.0, 0),
        size=(1, 0, 0.5),
        polarization="Ex",
        source_time=td.GaussianPulse(freq0=100e14, fwidth=10e14),
        current_amplitude_definition="total",
    )
    with pytest.raises(ValidationError, match=name) as e:
        _ = sim.updated_copy(sources=[source])

    # Test 3) Create a monitor lying outside the simulation boundary.
    # Check that an error message is generated containing the monitor's `name`.
    name = "monitor_123"
    monitor = td.FieldMonitor(name=name, center=(-1.0, 0, 0), size=(0.5, 0, 1), freqs=[100e14])
    with pytest.raises(ValidationError, match=name) as e:
        _ = sim.updated_copy(monitors=[monitor])


def test_structures_per_medium(monkeypatch):
    """Test if structures that share the same medium warn or error appropriately."""

    # Set low thresholds to keep the test fast; ensure len(structures) > MAX to avoid early return
    monkeypatch.setattr(scene, "WARN_STRUCTURES_PER_MEDIUM", 2)
    monkeypatch.setattr(scene, "MAX_STRUCTURES_PER_MEDIUM", 4)

    shared_med = td.Medium(permittivity=2.0)
    # 3 share the same medium -> triggers warning (> WARN), but <= MAX per-medium
    same_medium_structs = [
        td.Structure(geometry=td.Box(size=(1, 1, 1)), medium=shared_med) for _ in range(3)
    ]
    # Add two with different mediums so total structures > MAX but error should not be triggered
    other_structs = [
        td.Structure(geometry=td.Box(size=(1, 1, 1)), medium=td.Medium(permittivity=3.0)),
        td.Structure(geometry=td.Box(size=(1, 1, 1)), medium=td.Medium(permittivity=4.0)),
    ]
    structs = same_medium_structs + other_structs  # total = 5 > MAX = 4

    with AssertLogLevel("WARNING", contains_str="use the same medium"):
        _ = td.Simulation(
            size=(10, 10, 10),
            run_time=1e-12,
            grid_spec=td.GridSpec.uniform(dl=0.02),
            structures=structs,
        )

    # Now test error
    monkeypatch.setattr(scene, "MAX_STRUCTURES_PER_MEDIUM", 3, raising=False)
    structs = [td.Structure(geometry=td.Box(size=(1, 1, 1)), medium=shared_med) for _ in range(4)]

    with pytest.raises(ValidationError, match="use the same medium"):
        _ = td.Simulation(
            size=(10, 10, 10),
            run_time=1e-12,
            grid_spec=td.GridSpec.uniform(dl=0.02),
            structures=structs,
        )


def test_validate_microwave_mode_spec():
    """Test that auto generation ande user supplied path specs are correctly validated."""
    freq0 = 10e9
    mm = 1e3
    run_time_spec = td.RunTimeSpec(quality_factor=3.0)
    size = (10 * mm, 10 * mm, 10 * mm)
    size_mon = (0, 8 * mm, 8 * mm)

    # Currently limited to generation of axis aligned boxes around conductors,
    # so the path may intersect other nearby conductors, like in this coaxial cable
    coaxial = td.Structure(
        geometry=td.GeometryGroup(
            geometries=(
                td.ClipOperation(
                    operation="difference",
                    geometry_a=td.Cylinder(
                        axis=0, radius=2.5 * mm, center=(0, 0, 0), length=td.inf
                    ),
                    geometry_b=td.Cylinder(
                        axis=0, radius=1.3 * mm, center=(0, 0, 0), length=td.inf
                    ),
                ),
                td.Cylinder(axis=0, radius=1 * mm, center=(0, 0, 0), length=td.inf),
            )
        ),
        medium=td.PEC,
    )
    mode_spec = td.MicrowaveModeSpec(
        num_modes=2,
        target_neff=1.8,
        impedance_specs=(td.AutoImpedanceSpec(), td.AutoImpedanceSpec()),
    )

    mode_mon = td.MicrowaveModeMonitor(
        center=(0, 0, 0),
        size=size_mon,
        freqs=[freq0],
        name="mode_1",
        colocate=False,
        mode_spec=mode_spec,
    )
    sim = td.Simulation(
        run_time=run_time_spec,
        size=size,
        sources=[],
        structures=[coaxial],
        grid_spec=td.GridSpec.uniform(dl=0.1 * mm),
        monitors=[mode_mon],
    )

    # check that validation error is caught
    with pytest.raises(SetupError):
        sim._validate_microwave_mode_specs()

    # Custom current spec is too large for mode plane
    custom_spec = td.CustomImpedanceSpec(
        current_spec=td.Custom2DCurrentIntegralSpec.from_circular_path(
            center=(0, 0, 0), radius=10 * mm, num_points=21, normal_axis=0, clockwise=True
        )
    )
    mode_spec = td.MicrowaveModeSpec(
        num_modes=2,
        target_neff=1.8,
        impedance_specs=(custom_spec, td.AutoImpedanceSpec()),
    )

    mode_mon = mode_mon.updated_copy(
        path="mode_spec/", impedance_specs=(custom_spec, td.AutoImpedanceSpec())
    )
    # check that validation error is in the MicrowaveModeSpec
    with pytest.raises(ValidationError):
        sim = sim.updated_copy(
            monitors=[mode_mon],
        )


def test_padded_copy():
    """Test that padding layers are added along simulation boundaries."""
    grid_spec = td.GridSpec.auto(wavelength=1.0)

    sim = td.Simulation(
        size=(5, 5, 5),
        grid_spec=grid_spec,
        structures=[
            td.Structure(geometry=td.Box(size=(10, 13, 7)), medium=td.Medium(permittivity=2.0))
        ],
        lumped_elements=[],
        run_time=1e-12,
    )

    padded_sim = sim.padded_copy(x=(4, 10), y=(1, 2))
    assert np.allclose(np.array(padded_sim.size), np.array([19, 8, 5]))
    assert np.allclose(np.array(padded_sim.center), np.array([3, 0.5, 0]))

    with pytest.raises(ValueError):
        padded_sim = sim.padded_copy(x=(1, -2), z=(-2, 0))
    with pytest.raises(ValueError):
        padded_sim = sim.padded_copy(x=(1))


def test_uniformly_padded_copy():
    """Test that padding layers are uniformly added along simulation boundaries."""
    grid_spec = td.GridSpec.auto(wavelength=1.0)

    sim = td.Simulation(
        size=(5, 5, 5),
        grid_spec=grid_spec,
        structures=[
            td.Structure(geometry=td.Box(size=(3, 2, 4)), medium=td.Medium(permittivity=2.0))
        ],
        lumped_elements=[],
        run_time=1e-12,
    )

    padded_sim = sim.uniformly_padded_copy(padding=5)
    assert np.allclose(np.array(padded_sim.size), np.array([15, 15, 15]))
    assert np.allclose(np.array(padded_sim.center), np.array([0, 0, 0]))

    with pytest.raises(ValueError):
        padded_sim = sim.uniformly_padded_copy(padding=-1)


@pytest.mark.parametrize("structure_priority_mode", ["equal", "conductor"])
def test_finalized_volumetric_structures_respects_priority_mode(structure_priority_mode):
    """Test that _finalized_volumetric_structures respects structure_priority_mode."""
    # Create structures with different media types
    dielectric = td.Structure(
        geometry=td.Box(size=(1, 1, 1), center=(0, 0, 0)),
        medium=td.Medium(permittivity=2.0),
    )
    pec_struct = td.Structure(
        geometry=td.Box(size=(0.5, 0.5, 0.5), center=(0, 0, 0)),
        medium=td.PEC,
    )
    lossy_metal = td.Structure(
        geometry=td.Box(size=(0.8, 0.8, 0.8), center=(0, 0, 0)),
        medium=td.LossyMetalMedium(conductivity=1e7, frequency_range=(1e14, 2e14)),
    )

    structures = [pec_struct, lossy_metal, dielectric]

    sim = td.Simulation(
        size=(2, 2, 2),
        structures=structures,
        structure_priority_mode=structure_priority_mode,
        run_time=1e-12,
        grid_spec=td.GridSpec.uniform(dl=0.1),
    )

    # No 2D materials or lumped elements, so _finalized_volumetric_structures
    # should use scene.sorted_structures
    finalized_structures = sim._finalized_volumetric_structures
    sorted_structures = sim.scene.sorted_structures

    # Should match sorted_structures (modal_frames are empty in this case)
    assert len(finalized_structures) == len(sorted_structures)
    for fin, sorted_s in zip(finalized_structures, sorted_structures):
        assert fin.medium == sorted_s.medium

    if structure_priority_mode == "conductor":
        # In conductor mode: PEC should be last, lossy metal second to last
        assert finalized_structures[-1].medium == td.PEC
        assert isinstance(finalized_structures[-2].medium, td.LossyMetalMedium)
    else:
        # In equal mode: original order preserved
        assert finalized_structures[-1].medium == dielectric.medium
        assert isinstance(finalized_structures[1].medium, td.LossyMetalMedium)
        assert finalized_structures[0].medium == td.PEC


def test_extrusion_suppresses_structure_at_edges_warning():
    """Test that structure at edges warning is suppressed when extrude_structures=True."""

    # Create a structure that extends exactly to x-min edge only
    # Simulation: size=(2, 1, 1), center=(0, 0, 0) → bounds x:[-1, 1], y:[-0.5, 0.5], z:[-0.5, 0.5]
    # Structure: center=(-0.5, 0, 0), size=(1, 0.4, 0.4) → x-min = -1 (touches x-min edge), y:[-0.2, 0.2], z:[-0.2, 0.2] (inside bounds)
    structure = td.Structure(
        geometry=td.Box(size=(1, 0.4, 0.4), center=(-0.5, 0, 0)),  # Touches x-min edge at -1 only
        medium=td.Medium(permittivity=2.0),
    )

    # Use PointDipole to avoid PlaneWave validation issues with structure intersections
    source = td.PointDipole(
        source_time=td.GaussianPulse(freq0=2e14, fwidth=1e13),
        center=(0, 0, 0),
        polarization="Ex",
    )

    # Test 1: Warning should appear when extrude_structures=False
    boundary_spec_no_extrusion = td.BoundarySpec(
        x=td.Boundary(
            minus=td.PML(extrude_structures=False), plus=td.PML(extrude_structures=False)
        ),
        y=td.Boundary.pml(),
        z=td.Boundary.pml(),
    )

    with AssertLogLevel("WARNING", contains_str="has bounds that extend exactly"):
        sim = td.Simulation(
            size=(2, 1, 1),
            center=(0, 0, 0),
            structures=[structure],
            sources=[source],
            boundary_spec=boundary_spec_no_extrusion,
            run_time=1e-12,
        )

    # Test 2: Warning should be suppressed when extrude_structures=True on x-min
    boundary_spec_with_extrusion = td.BoundarySpec(
        x=td.Boundary(minus=td.PML(extrude_structures=True), plus=td.PML(extrude_structures=True)),
        y=td.Boundary.pml(),
        z=td.Boundary.pml(),
    )

    # Verify the "structure at edges" warning is NOT present (other warnings may still appear)
    with AssertLogStr(log_level_expected="WARNING", excludes_str="has bounds that extend exactly"):
        sim = td.Simulation(
            size=(2, 1, 1),
            center=(0, 0, 0),
            structures=[structure],
            sources=[source],
            boundary_spec=boundary_spec_with_extrusion,
            run_time=1e-12,
        )


def test_extrusion_suppresses_structure_in_pml_warning():
    """Test that structure in PML warning is suppressed when extrude_structures=True."""

    # Create a structure that extends into PML region on x-min side only
    # Simulation: size=(2, 1, 1), center=(0, 0, 0) → bounds x:[-1, 1], y:[-0.5, 0.5], z:[-0.5, 0.5]
    # PML extends beyond x=-1, so structure x-min needs to be < -1 to be in PML
    # Structure must be partially inside simulation domain for extrusion to work (within 2 cells of boundary)
    # Structure: center=(-0.95, 0, 0), size=(0.2, 0.4, 0.4) → x-min = -1.05 (in PML), x-max = -0.85 (in simulation domain), y/z inside bounds
    structure = td.Structure(
        geometry=td.Box(
            size=(1.2, 0.4, 0.4), center=(-0.5, 0, 0)
        ),  # Partially in simulation domain, extends into PML
        medium=td.Medium(permittivity=2.0),
    )

    # Use PointDipole to avoid PlaneWave validation issues
    source = td.PointDipole(
        source_time=td.GaussianPulse(freq0=2e14, fwidth=1e13),
        center=(0.5, 0, 0),  # Place away from structure
        polarization="Ex",
    )

    # Test 1: Warning should appear when extrude_structures=False
    boundary_spec_no_extrusion = td.BoundarySpec(
        x=td.Boundary(
            minus=td.PML(extrude_structures=False, num_layers=12),
            plus=td.PML(extrude_structures=False, num_layers=12),
        ),
        y=td.Boundary.pml(),
        z=td.Boundary.pml(),
    )

    with AssertLogLevel("WARNING", contains_str="within the simulation PML"):
        sim = td.Simulation(
            size=(2, 4, 4),
            center=(0, 0, 0),
            structures=[structure],
            sources=[source],
            boundary_spec=boundary_spec_no_extrusion,
            run_time=1e-12,
        )

    # Test 2: Warning should be suppressed when extrude_structures=True
    boundary_spec_with_extrusion = td.BoundarySpec(
        x=td.Boundary(
            minus=td.PML(extrude_structures=True, num_layers=12),
            plus=td.PML(extrude_structures=True, num_layers=12),
        ),
        y=td.Boundary.pml(),
        z=td.Boundary.pml(),
    )

    # Verify the "structure in PML" warning is NOT present (other warnings may still appear)
    with AssertLogStr(log_level_expected="WARNING", excludes_str="within the simulation PML"):
        sim = td.Simulation(
            size=(2, 4, 4),
            center=(0, 0, 0),
            structures=[structure],
            sources=[source],
            boundary_spec=boundary_spec_with_extrusion,
            run_time=1e-12,
        )


def test_extrusion_warns_automatic_extrusion():
    """Test that automatic extrusion warning appears when structure is close to PML with extrude_structures=True."""

    # Create a structure close to PML boundary (within half wavelength AND within 2 cells)
    freq0 = 2e14  # 200 THz
    wavelength = 3e8 / freq0 * 1e6  # Convert to um: ~1.5 um
    half_wavelength = wavelength / 2

    # Structure positioned just inside simulation boundary, close to PML
    # Use longer simulation domain to ensure better grid resolution and clipping margin coverage
    sim_size = (10, 1, 1)  # In um - longer in x direction
    sim_bound_min = -sim_size[0] / 2

    # Structure extends from -4.95 um to 3 um
    structure_x_min = -4.95  # Very close to boundary (within 2 cells for typical grid)
    structure_x_max = 3.0
    structure_size_x = structure_x_max - structure_x_min  # 7.95 um
    structure_center_x = (structure_x_min + structure_x_max) / 2  # -0.975 um
    structure = td.Structure(
        geometry=td.Box(size=(structure_size_x, 10, 10), center=(structure_center_x, 0, 0)),
        medium=td.Medium(permittivity=2.0),
    )

    # Use PointDipole to avoid PlaneWave validation issues
    source = td.PointDipole(
        source_time=td.GaussianPulse(freq0=freq0, fwidth=freq0 * 0.1),
        center=(0, 0, 0),
        polarization="Ex",
    )

    # Test: Warning should appear when extrude_structures=True and structure is within clipping margin
    boundary_spec_with_extrusion = td.BoundarySpec.all_sides(
        boundary=td.PML(extrude_structures=True, num_layers=12)
    )

    with AssertLogLevel("WARNING", contains_str="will be automatically extruded"):
        sim = td.Simulation(
            size=sim_size,
            center=(0, 0, 0),
            structures=[structure],
            sources=[source],
            boundary_spec=boundary_spec_with_extrusion,
            run_time=1e-12,
        )


def test_extrusion_warning_not_triggered_when_far_from_pml():
    """Test extrusion warning behavior based on structure distance from PML.

    Case 1: Structure > half_wavelength AND > 2 cells away → no warning
    Case 2: Structure < half_wavelength BUT > 2 cells away → warns about distance to PML, but does not trigger automatic extrusion
    """

    freq0 = 2e14  # 200 THz
    wavelength = 3e8 / freq0 * 1e6  # Convert to um: ~1.5 um
    half_wavelength = wavelength / 2

    sim_size = (2, 1, 1)  # In um
    sim_bound_min = -sim_size[0] / 2

    # Use PointDipole to avoid PlaneWave validation issues
    source = td.PointDipole(
        source_time=td.GaussianPulse(freq0=freq0, fwidth=freq0 * 0.1),
        center=(0, 0, 0),
        polarization="Ex",
    )

    boundary_spec_with_extrusion = td.BoundarySpec.all_sides(
        boundary=td.PML(extrude_structures=True, num_layers=12)
    )

    # Case 1: Structure is more than half_wavelength AND more than 2 cells away from boundary
    # No warning expected
    # Place structure near center with small size to ensure it's > half_wavelength from both boundaries
    # Structure bounds: x: [-0.1, 0.1], distance from boundaries: 0.9 um > 0.75 um (half_wavelength) ✓
    structure_far = td.Structure(
        geometry=td.Box(size=(0.2, 10, 10), center=(0.0, 0, 0)),
        medium=td.Medium(permittivity=2.0),
    )

    # Verify no warning when structure is far from PML
    with AssertLogStr(
        log_level_expected="WARNING", excludes_str="less than half of a central wavelength"
    ):
        sim = td.Simulation(
            size=sim_size,
            center=(0, 0, 0),
            structures=[structure_far],
            sources=[source],
            boundary_spec=boundary_spec_with_extrusion,
            run_time=1e-12,
        )

    # Case 2: Structure is closer than half_wavelength BUT more than 2 cells away (outside clipping margin)
    # Should warn about distance to PML (not automatic extrusion, since it's outside clipping margin)
    # Place structure so its x-min bound is inside simulation domain but within half_wavelength of boundary
    structure_size_x = 0.3
    structure_x_min_close = (
        sim_bound_min + half_wavelength * 0.4
    )  # x-min close to boundary (< half_wavelength)
    structure_center_x_close = (
        structure_x_min_close + structure_size_x / 2
    )  # Center so x-min is at desired position
    structure_close = td.Structure(
        geometry=td.Box(size=(structure_size_x, 10, 10), center=(structure_center_x_close, 0, 0)),
        medium=td.Medium(permittivity=2.0),
    )

    # Verify warning about distance to PML (contains distance warning, excludes automatic extrusion warning)
    with AssertLogStr(
        log_level_expected="WARNING",
        contains_str="less than half of a central wavelength",
        excludes_str="will be automatically extruded",
    ):
        sim = td.Simulation(
            size=sim_size,
            center=(0, 0, 0),
            structures=[structure_close],
            sources=[source],
            boundary_spec=boundary_spec_with_extrusion,
            run_time=1e-12,
        )


def test_extrusion_warnings_with_absorber():
    """Test that extrusion warnings work correctly with Absorber boundaries."""

    # Structure that touches x-min edge only
    # Simulation: size=(2, 1, 1), center=(0, 0, 0) → bounds x:[-1, 1], y:[-0.5, 0.5], z:[-0.5, 0.5]
    # Structure: center=(-0.5, 0, 0), size=(1, 0.4, 0.4) → x-min = -1 (touches edge), y/z inside bounds
    structure = td.Structure(
        geometry=td.Box(size=(1, 0.4, 0.4), center=(-0.5, 0, 0)),
        medium=td.Medium(permittivity=2.0),
    )

    # Use PointDipole to avoid PlaneWave validation issues
    source = td.PointDipole(
        source_time=td.GaussianPulse(freq0=2e14, fwidth=1e13),
        center=(0, 0, 0),
        polarization="Ex",
    )

    # Test with Absorber (default extrude_structures=False)
    boundary_spec_absorber_no_extrusion = td.BoundarySpec(
        x=td.Boundary(
            minus=td.Absorber(extrude_structures=False), plus=td.Absorber(extrude_structures=False)
        ),
        y=td.Boundary.pml(),
        z=td.Boundary.pml(),
    )

    with AssertLogLevel("WARNING", contains_str="has bounds that extend exactly"):
        sim = td.Simulation(
            size=(2, 1, 1),
            center=(0, 0, 0),
            structures=[structure],
            sources=[source],
            boundary_spec=boundary_spec_absorber_no_extrusion,
            run_time=1e-12,
        )

    # Test with Absorber (extrude_structures=True)
    boundary_spec_absorber_with_extrusion = td.BoundarySpec(
        x=td.Boundary(
            minus=td.Absorber(extrude_structures=True), plus=td.Absorber(extrude_structures=True)
        ),
        y=td.Boundary.pml(),
        z=td.Boundary.pml(),
    )

    # Verify the "structure at edges" warning is NOT present (other warnings may still appear)
    with AssertLogStr(log_level_expected="WARNING", excludes_str="has bounds that extend exactly"):
        sim = td.Simulation(
            size=(2, 1, 1),
            center=(0, 0, 0),
            structures=[structure],
            sources=[source],
            boundary_spec=boundary_spec_absorber_with_extrusion,
            run_time=1e-12,
        )


def test_extrusion_warnings_with_non_pml_boundaries():
    """Test that extrusion warnings are not triggered for non-PML boundaries."""

    # Structure that touches x-min edge only
    # Simulation: size=(2, 1, 1), center=(0, 0, 0) → bounds x:[-1, 1], y:[-0.5, 0.5], z:[-0.5, 0.5]
    # Structure: center=(-0.5, 0, 0), size=(1, 0.4, 0.4) → x-min = -1 (touches edge), y/z inside bounds
    structure = td.Structure(
        geometry=td.Box(size=(1, 0.4, 0.4), center=(-0.5, 0, 0)),
        medium=td.Medium(permittivity=2.0),
    )

    # Use PointDipole to avoid PlaneWave validation issues
    source = td.PointDipole(
        source_time=td.GaussianPulse(freq0=2e14, fwidth=1e13),
        center=(0, 0, 0),
        polarization="Ex",
    )

    # Test with PEC boundary (no extrude_structures attribute)
    boundary_spec_pec = td.BoundarySpec(
        x=td.Boundary(minus=td.PECBoundary(), plus=td.PECBoundary()),
        y=td.Boundary.pml(),
        z=td.Boundary.pml(),
    )

    # Should still warn about structure at edges (extrusion not applicable)
    with AssertLogLevel("WARNING", contains_str="has bounds that extend exactly"):
        sim = td.Simulation(
            size=(2, 1, 1),
            center=(0, 0, 0),
            structures=[structure],
            sources=[source],
            boundary_spec=boundary_spec_pec,
            run_time=1e-12,
        )

    # Test with Periodic boundary (no extrude_structures attribute)
    boundary_spec_periodic = td.BoundarySpec(
        x=td.Boundary(minus=td.Periodic(), plus=td.Periodic()),
        y=td.Boundary.pml(),
        z=td.Boundary.pml(),
    )

    # Should still warn about structure at edges
    with AssertLogLevel("WARNING", contains_str="has bounds that extend exactly"):
        sim = td.Simulation(
            size=(2, 1, 1),
            center=(0, 0, 0),
            structures=[structure],
            sources=[source],
            boundary_spec=boundary_spec_periodic,
            run_time=1e-12,
        )


def test_extrusion_warnings_mixed_boundaries():
    """Test that warnings work correctly when different boundaries are used on different sides."""

    # Structure that touches x-min edge only
    # Simulation: size=(2, 1, 1), center=(0, 0, 0) → bounds x:[-1, 1], y:[-0.5, 0.5], z:[-0.5, 0.5]
    # Structure: center=(-0.5, 0, 0), size=(1, 0.4, 0.4) → x-min = -1 (touches x-min edge only), y/z inside bounds
    structure = td.Structure(
        geometry=td.Box(size=(1, 0.4, 0.4), center=(-0.5, 0, 0)),
        medium=td.Medium(permittivity=2.0),
    )

    # Use PointDipole to avoid PlaneWave validation issues
    source = td.PointDipole(
        source_time=td.GaussianPulse(freq0=2e14, fwidth=1e13),
        center=(0.5, 0, 0),  # Place away from structure
        polarization="Ex",
    )

    # PML with extrusion on x-min, PML without extrusion on x-max
    boundary_spec_mixed = td.BoundarySpec(
        x=td.Boundary(
            minus=td.PML(extrude_structures=True),
            plus=td.PML(extrude_structures=False),
        ),
        y=td.Boundary.pml(),
        z=td.Boundary.pml(),
    )

    # Warning should be suppressed for x-min (extrusion enabled)
    # Since structure only touches x-min, no warning should appear
    with AssertLogStr(log_level_expected="WARNING", excludes_str="has bounds that extend exactly"):
        sim = td.Simulation(
            size=(2, 1, 1),
            center=(0, 0, 0),
            structures=[structure],
            sources=[source],
            boundary_spec=boundary_spec_mixed,
            run_time=1e-12,
        )
