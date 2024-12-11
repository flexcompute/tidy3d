"""Tests the simulation and its validators."""

import gdstk
import matplotlib.pyplot as plt
import numpy as np
import pydantic.v1 as pydantic
import pytest
import tidy3d as td
from tidy3d.components import simulation
from tidy3d.components.scene import MAX_GEOMETRY_COUNT, MAX_NUM_MEDIUMS
from tidy3d.components.simulation import MAX_NUM_SOURCES
from tidy3d.exceptions import SetupError, Tidy3dKeyError

from ..utils import (
    SIM_FULL,
    AssertLogLevel,
    assert_log_level,
    cartesian_to_unstructured,
    run_emulated,
)

SIM = td.Simulation(size=(1, 1, 1), run_time=1e-12, grid_spec=td.GridSpec(wavelength=1.0))

RTOL = 0.01


def test_sim_init():
    """make sure a simulation can be initialized"""

    sim = td.Simulation(
        size=(2.0, 2.0, 2.0),
        run_time=1e-12,
        structures=[
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
        ],
        sources=[
            td.UniformCurrentSource(
                size=(0, 0, 0),
                center=(0, -0.5, 0),
                polarization="Hx",
                source_time=td.GaussianPulse(
                    freq0=1e14,
                    fwidth=1e12,
                ),
                name="my_dipole",
            ),
            td.PointDipole(
                center=(0, 0, 0),
                polarization="Ex",
                source_time=td.GaussianPulse(
                    freq0=1e14,
                    fwidth=1e12,
                ),
            ),
        ],
        monitors=[
            td.FieldMonitor(size=(0, 0, 0), center=(0, 0, 0), freqs=[1e12, 2e12], name="point"),
            td.FluxTimeMonitor(size=(1, 1, 0), center=(0, 0, 0), interval=10, name="plane"),
        ],
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


def test_num_cells():
    """Test num_cells and num_computational_grid_points."""

    sim = td.Simulation(
        size=(1, 1, 1),
        run_time=1e-12,
        grid_spec=td.GridSpec.uniform(dl=0.1),
        sources=[
            td.PointDipole(
                center=(0, 0, 0),
                polarization="Ex",
                source_time=td.GaussianPulse(freq0=2e14, fwidth=1e14),
            )
        ],
    )
    assert sim.num_computational_grid_points > sim.num_cells  # due to extra pixels at boundaries

    sim = sim.updated_copy(symmetry=(1, 0, 0))
    assert sim.num_computational_grid_points < sim.num_cells  # due to symmetry


def test_monitors_data_size():
    """make sure a simulation can be initialized"""

    sim = td.Simulation(
        size=(2.0, 2.0, 2.0),
        run_time=1e-12,
        structures=[
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
        ],
        sources=[
            td.UniformCurrentSource(
                size=(0, 0, 0),
                center=(0, -0.5, 0),
                polarization="Hx",
                source_time=td.GaussianPulse(
                    freq0=1e14,
                    fwidth=1e12,
                ),
                name="my_dipole",
            ),
            td.PointDipole(
                center=(0, 0, 0),
                polarization="Ex",
                source_time=td.GaussianPulse(
                    freq0=1e14,
                    fwidth=1e12,
                ),
            ),
        ],
        monitors=[
            td.FieldMonitor(size=(0, 0, 0), center=(0, 0, 0), freqs=[1e12, 2e12], name="point"),
            td.FluxTimeMonitor(size=(1, 1, 0), center=(0, 0, 0), interval=10, name="plane"),
        ],
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


def test_deprecation_defaults(log_capture):
    """Make sure deprecation warnings NOT thrown if defaults used."""
    _ = td.Simulation(
        size=(1, 1, 1),
        run_time=1e-12,
        grid_spec=td.GridSpec.uniform(dl=0.1),
        sources=[
            td.PointDipole(
                center=(0, 0, 0),
                polarization="Ex",
                source_time=td.GaussianPulse(freq0=2e14, fwidth=1e14),
            )
        ],
    )
    assert_log_level(log_capture, None)


@pytest.mark.parametrize("shift_amount, log_level", ((1, None), (2, "WARNING")))
def test_sim_bounds(shift_amount, log_level, log_capture):
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
            structures=[
                td.Structure(
                    geometry=td.Box(size=(1, 1, 1), center=shifted_center), medium=td.Medium()
                )
            ],
            boundary_spec=td.BoundarySpec.all_sides(boundary=td.Periodic()),
            sources=[
                td.PointDipole(
                    center=CENTER_SHIFT,
                    polarization="Ex",
                    source_time=td.GaussianPulse(freq0=td.C_0, fwidth=td.C_0),
                )
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
            center = shift_amount * amp * sign
            if np.sum(center) < 1e-12:
                continue
            place_box(tuple(center))
    assert_log_level(log_capture, log_level)


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
    with pytest.raises(pydantic.ValidationError):
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
            monitors=[
                td.FieldMonitor(
                    size=(td.inf, td.inf, td.inf), freqs=np.linspace(0, 200e12, 10001), name="test"
                )
            ],
            run_time=1e-12,
            boundary_spec=td.BoundarySpec.all_sides(boundary=td.Periodic()),
        )
        s.validate_pre_upload()


@pytest.mark.parametrize("freq, log_level", [(1.5, "WARNING"), (2.5, "INFO"), (3.5, "WARNING")])
def test_monitor_medium_frequency_range(log_capture, freq, log_level):
    # monitor frequency above or below a given medium's range should throw a warning

    medium = td.Medium(frequency_range=(2e12, 3e12))
    box = td.Structure(geometry=td.Box(size=(0.1, 0.1, 0.1)), medium=medium)
    mnt = td.FieldMonitor(size=(0, 0, 0), name="freq", freqs=[freq * 1e12])
    src = td.UniformCurrentSource(
        source_time=td.GaussianPulse(freq0=2.5e12, fwidth=0.5e12),
        size=(0, 0, 0),
        polarization="Ex",
    )
    _ = td.Simulation(
        size=(1, 1, 1),
        structures=[box],
        monitors=[mnt],
        sources=[src],
        run_time=1e-12,
        boundary_spec=td.BoundarySpec.all_sides(boundary=td.Periodic()),
    )
    assert_log_level(log_capture, log_level)


@pytest.mark.parametrize("fwidth, log_level", [(0.1e12, "WARNING"), (2e12, "INFO")])
def test_monitor_simulation_frequency_range(log_capture, fwidth, log_level):
    # monitor frequency outside of the simulation's frequency range should throw a warning

    src = td.UniformCurrentSource(
        source_time=td.GaussianPulse(freq0=2.0e12, fwidth=fwidth),
        size=(0, 0, 0),
        polarization="Ex",
    )
    mnt = td.FieldMonitor(size=(0, 0, 0), name="freq", freqs=[1.5e12])
    _ = td.Simulation(
        size=(1, 1, 1),
        monitors=[mnt],
        sources=[src],
        run_time=1e-12,
        boundary_spec=td.BoundarySpec.all_sides(boundary=td.Periodic()),
    )
    assert_log_level(log_capture, log_level)


def test_validate_bloch_with_symmetry():
    with pytest.raises(pydantic.ValidationError):
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
    )
    src0 = td.UniformCurrentSource(
        source_time=td.GaussianPulse(freq0=2.0e12, fwidth=1.0e12, amplitude=0),
        size=(0, 0, 0),
        polarization="Ex",
    )

    # negative normalize index
    with pytest.raises(pydantic.ValidationError):
        td.Simulation(
            size=(1, 1, 1),
            run_time=1e-12,
            grid_spec=td.GridSpec.uniform(dl=0.1),
            normalize_index=-1,
        )

    # normalize index out of bounds
    with pytest.raises(pydantic.ValidationError):
        td.Simulation(
            size=(1, 1, 1),
            run_time=1e-12,
            grid_spec=td.GridSpec.uniform(dl=0.1),
            sources=[src],
            normalize_index=1,
        )
    # skipped if no sources
    td.Simulation(
        size=(1, 1, 1), run_time=1e-12, grid_spec=td.GridSpec.uniform(dl=0.1), normalize_index=1
    )

    # normalize by zero-amplitude source
    with pytest.raises(pydantic.ValidationError):
        td.Simulation(
            size=(1, 1, 1),
            run_time=1e-12,
            grid_spec=td.GridSpec.uniform(dl=0.1),
            sources=[src0],
        )


def test_validate_plane_wave_boundaries(log_capture):
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
        sources=[src1],
        boundary_spec=bspec1,
    )

    # angled incidence plane wave with PMLs / absorbers should error
    with pytest.raises(pydantic.ValidationError):
        td.Simulation(
            size=(1, 1, 1),
            run_time=1e-12,
            sources=[src2],
            boundary_spec=bspec1,
        )

    # angled incidence plane wave with periodic boundaries should warn
    with AssertLogLevel(log_capture, "WARNING", contains_str="incorrectly set"):
        td.Simulation(
            size=(1, 1, 1),
            run_time=1e-12,
            sources=[src2],
            boundary_spec=td.BoundarySpec.all_sides(td.Periodic()),
        )

    # angled incidence plane wave with an integer-offset Bloch vector should warn
    with AssertLogLevel(log_capture, "WARNING", contains_str="integer reciprocal"):
        td.Simulation(
            size=(1, 1, 1),
            run_time=1e-12,
            sources=[src2],
            boundary_spec=bspec3,
            monitors=[mnt],
        )

    # angled incidence plane wave with wrong Bloch vector should warn
    with AssertLogLevel(log_capture, "WARNING", contains_str="incorrectly set"):
        td.Simulation(
            size=(1, 1, 1),
            run_time=1e-12,
            sources=[src2],
            boundary_spec=bspec4,
        )


def test_validate_zero_dim_boundaries(log_capture):
    # zero-dim simulation with an absorbing boundary in that direction should error
    src = td.PlaneWave(
        source_time=td.GaussianPulse(freq0=2.5e14, fwidth=1e13),
        center=(0, 0, 0),
        size=(td.inf, 0, td.inf),
        direction="+",
        pol_angle=0.0,
    )

    with pytest.raises(pydantic.ValidationError):
        td.Simulation(
            size=(1, 1, 0),
            run_time=1e-12,
            sources=[src],
            boundary_spec=td.BoundarySpec(
                x=td.Boundary.periodic(),
                y=td.Boundary.periodic(),
                z=td.Boundary.pml(),
            ),
        )

    # zero-dim simulation with an absorbing boundary any other direction should not error
    td.Simulation(
        size=(1, 1, 0),
        run_time=1e-12,
        sources=[src],
        boundary_spec=td.BoundarySpec(
            x=td.Boundary.pml(),
            y=td.Boundary.stable_pml(),
            z=td.Boundary.periodic(),
        ),
    )


def test_validate_components_none():
    assert SIM._structures_not_at_edges(val=None, values=SIM.dict()) is None
    assert SIM._validate_num_sources(val=None) is None
    assert SIM._warn_monitor_mediums_frequency_range(val=None, values=SIM.dict()) is None
    assert SIM._warn_monitor_simulation_frequency_range(val=None, values=SIM.dict()) is None
    assert SIM._warn_grid_size_too_small(val=None, values=SIM.dict()) is None
    assert SIM._source_homogeneous_isotropic(val=None, values=SIM.dict()) is None


def test_sources_edge_case_validation(log_capture):
    values = SIM.dict()
    values.pop("sources")
    SIM._warn_monitor_simulation_frequency_range(val="test", values=values)
    assert_log_level(log_capture, "WARNING")


def test_validate_size_run_time(monkeypatch):
    monkeypatch.setattr(simulation, "MAX_TIME_STEPS", 1)
    with pytest.raises(SetupError):
        s = SIM.copy(update=dict(run_time=1e-12))
        s._validate_size()


def test_validate_size_spatial_and_time(monkeypatch):
    monkeypatch.setattr(simulation, "MAX_CELLS_TIMES_STEPS", 1)
    with pytest.raises(SetupError):
        s = SIM.copy(update=dict(run_time=1e-12))
        s._validate_size()


def test_validate_mnt_size(monkeypatch, log_capture):
    # warning for monitor size
    monkeypatch.setattr(simulation, "WARN_MONITOR_DATA_SIZE_GB", 1 / 2**30)
    s = SIM.copy(update=dict(monitors=(td.FieldMonitor(name="f", freqs=[1e12], size=(1, 1, 1)),)))
    s._validate_monitor_size()
    assert_log_level(log_capture, "WARNING")

    # error for simulation size
    monkeypatch.setattr(simulation, "MAX_SIMULATION_DATA_SIZE_GB", 1 / 2**30)
    with pytest.raises(SetupError):
        s = SIM.copy(
            update=dict(monitors=(td.FieldMonitor(name="f", freqs=[1e12], size=(1, 1, 1)),))
        )
        s._validate_monitor_size()


def test_max_geometry_validation():
    gs = td.GridSpec(wavelength=1.0)
    too_many = [td.Box(size=(1, 1, 1)) for _ in range(MAX_GEOMETRY_COUNT + 1)]

    fine = [
        td.Structure(
            geometry=td.ClipOperation(
                operation="union",
                geometry_a=td.Box(size=(1, 1, 1)),
                geometry_b=td.GeometryGroup(geometries=too_many),
            ),
            medium=td.Medium(permittivity=2.0),
        ),
        td.Structure(
            geometry=td.GeometryGroup(geometries=too_many),
            medium=td.Medium(permittivity=2.0),
        ),
    ]
    _ = td.Simulation(size=(1, 1, 1), run_time=1, grid_spec=gs, structures=fine)

    not_fine = [
        td.Structure(
            geometry=td.ClipOperation(
                operation="difference",
                geometry_a=td.Box(size=(1, 1, 1)),
                geometry_b=td.GeometryGroup(geometries=too_many),
            ),
            medium=td.Medium(permittivity=2.0),
        ),
    ]
    with pytest.raises(pydantic.ValidationError, match=f" {MAX_GEOMETRY_COUNT + 2} "):
        _ = td.Simulation(size=(1, 1, 1), run_time=1, grid_spec=gs, structures=not_fine)


def test_no_monitor():
    with pytest.raises(Tidy3dKeyError):
        SIM.get_monitor_by_name("NOPE")


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
    S2 = SIM_FULL.copy(update=dict(structures=new_structs))
    _ = S2.plot_structures_eps(x=0, alpha=0.5)
    plt.close()


def test_plot_eps_with_default_frequency(log_capture):
    """Make sure that when possible the permittivity is plotted using
    central frequency of the first source added to the simulation.
    """
    freq0 = 2e14
    src = td.PointDipole(polarization="Ex", source_time=td.GaussianPulse(freq0=freq0, fwidth=1e11))
    chromium = td.material_library["Cr"]["RakicLorentzDrude1998"]
    box = td.Structure(medium=chromium, geometry=td.Box(size=(0.2, 0.2, 0.2), center=(0, 0, 0)))
    sim = td.Simulation(
        size=(1, 1, 1),
        structures=[box],
        sources=[src],
        run_time=1e-12,
        boundary_spec=td.BoundarySpec.all_sides(boundary=td.PECBoundary()),
        grid_spec=td.GridSpec.uniform(dl=0.01),
    )
    # Source frequency is in range, so no warning
    log_capture.clear()
    _ = sim.plot_structures_eps(x=0)
    assert_log_level(log_capture, None)
    plt.close()

    freq0 = 20e14
    sim = sim.updated_copy(path="sources/0/source_time", freq0=freq0)
    # Source frequency is out of range, so give warning
    log_capture.clear()
    _ = sim.plot_structures_eps(x=0)
    assert_log_level(log_capture, "WARNING")
    plt.close()

    src2 = td.PointDipole(polarization="Ex", source_time=td.GaussianPulse(freq0=3e14, fwidth=1e11))
    sim = sim.updated_copy(sources=(src, src2))
    # Source frequencies do not agree, so give warning about evaluating at infinite frequency is out of range.
    log_capture.clear()
    _ = sim.plot_structures_eps(x=0)
    assert_log_level(log_capture, "WARNING")
    plt.close()


def test_plot_symmetries():
    S2 = SIM.copy(update=dict(symmetry=(1, 0, -1)))
    S2.plot_symmetries(x=0)
    plt.close()


def test_plot_grid():
    override = td.Structure(geometry=td.Box(size=(1, 1, 1)), medium=td.Medium())
    S2 = SIM_FULL.copy(
        update=dict(grid_spec=td.GridSpec(wavelength=1.0, override_structures=[override]))
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
    S2 = SIM_FULL.copy(update=dict(boundary_spec=bound_spec))
    S2.plot_boundaries(z=0)
    plt.close()


def test_plot_with_lumped_elements():
    load = td.LumpedResistor(
        center=(0, 0, 0), size=(1, 2, 0), name="resistor", voltage_axis=0, resistance=50
    )
    sim_test = SIM_FULL.updated_copy(lumped_elements=[load])
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
    S2 = SIM_FULL.copy(update=dict(boundary_spec=bound_spec))
    assert S2.complex_fields


def test_nyquist():
    S = SIM.copy(
        update=dict(
            sources=(
                td.PointDipole(
                    polarization="Ex", source_time=td.GaussianPulse(freq0=2e14, fwidth=1e11)
                ),
            ),
        )
    )
    assert S.nyquist_step > 1

    # nyquist step decreses to 1 when the frequency-domain monitor is at high frequency
    S_MONITOR = S.copy(
        update=dict(monitors=[td.FluxMonitor(size=(1, 1, 0), freqs=[1e14, 1e20], name="flux")])
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


def test_discretize_non_intersect(log_capture):
    SIM.discretize(box=td.Box(center=(-20, -20, -20), size=(1, 1, 1)))
    assert_log_level(log_capture, "ERROR")


def test_warn_sim_background_medium_freq_range(log_capture):
    _ = SIM.copy(
        update=dict(
            sources=(
                td.PointDipole(
                    polarization="Ex", source_time=td.GaussianPulse(freq0=2e14, fwidth=1e11)
                ),
            ),
            monitors=(td.FluxMonitor(name="test", freqs=[2e12], size=(1, 1, 0)),),
            medium=td.Medium(frequency_range=(0, 1e12)),
        )
    )
    assert_log_level(log_capture, "WARNING")


@pytest.mark.parametrize("grid_size,log_level", [(0.001, None), (3, "WARNING")])
def test_large_grid_size(log_capture, grid_size, log_level):
    # small fwidth should be inside range, large one should throw warning

    medium = td.Medium(permittivity=2, frequency_range=(2e14, 3e14))
    box = td.Structure(geometry=td.Box(size=(0.1, 0.1, 0.1)), medium=medium)
    src = td.PointDipole(
        source_time=td.GaussianPulse(freq0=2.5e14, fwidth=1e12),
        polarization="Ex",
    )
    _ = td.Simulation(
        size=(1, 1, 1),
        grid_spec=td.GridSpec.uniform(dl=grid_size),
        structures=[box],
        sources=[src],
        run_time=1e-12,
        boundary_spec=td.BoundarySpec.all_sides(boundary=td.Periodic()),
    )

    assert_log_level(log_capture, log_level)


@pytest.mark.parametrize("box_size,log_level", [(0.1, "INFO"), (9.9, "WARNING"), (20, "INFO")])
def test_sim_structure_gap(log_capture, box_size, log_level):
    """Make sure the gap between a structure and PML is not too small compared to lambda0."""
    medium = td.Medium(permittivity=2)
    box = td.Structure(geometry=td.Box(size=(box_size, box_size, box_size)), medium=medium)
    src = td.UniformCurrentSource(
        source_time=td.GaussianPulse(freq0=3e14, fwidth=1e13),
        size=(0, 0, 0),
        polarization="Ex",
    )
    _ = td.Simulation(
        size=(10, 10, 10),
        structures=[box],
        sources=[src],
        boundary_spec=td.BoundarySpec(
            x=td.Boundary.pml(num_layers=6),
            y=td.Boundary.pml(num_layers=6),
            z=td.Boundary.pml(num_layers=6),
        ),
        run_time=1e-12,
    )
    assert_log_level(log_capture, log_level)


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
        structures=[box_transparent],
        sources=[src],
        run_time=1e-12,
        boundary_spec=td.BoundarySpec.all_sides(boundary=td.Periodic()),
    )

    # with non-transparent box, raise
    with pytest.raises(pydantic.ValidationError):
        _ = td.Simulation(
            size=(1, 1, 1),
            medium=medium_bg,
            structures=[box_transparent, box],
            sources=[src],
            boundary_spec=td.BoundarySpec.all_sides(boundary=td.Periodic()),
        )

    # raise with anisotropic medium
    with pytest.raises(pydantic.ValidationError):
        _ = td.Simulation(
            size=(1, 1, 1),
            medium=medium_bg_diag,
            sources=[src],
            boundary_spec=td.BoundarySpec.all_sides(boundary=td.Periodic()),
        )

    with pytest.raises(pydantic.ValidationError):
        _ = td.Simulation(
            size=(1, 1, 1),
            medium=medium_bg_full,
            sources=[src],
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
            structures=[box_transparent],
            sources=[src],
            run_time=1e-12,
            monitors=[monitor],
            boundary_spec=td.BoundarySpec.all_sides(boundary=td.Periodic()),
        )

        # with non-transparent box, raise
        with pytest.raises(pydantic.ValidationError):
            _ = td.Simulation(
                size=(1, 1, 1),
                medium=medium_bg,
                structures=[box],
                sources=[src],
                monitors=[monitor],
                run_time=1e-12,
                boundary_spec=td.BoundarySpec.all_sides(boundary=td.Periodic()),
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
        structures=[box_transparent, box],
        sources=[src],
        monitors=[monitor_n2f_vol_exclude],
        run_time=1e-12,
        boundary_spec=td.BoundarySpec.all_sides(boundary=td.Periodic()),
    )


def test_proj_monitor_distance(log_capture):
    """Make sure a warning is issued if the projection distance for exact projections
    is very large compared to the simulation domain size.
    """

    monitor_n2f = td.FieldProjectionAngleMonitor(
        center=(0, 0, 0),
        size=(td.inf, td.inf, 0),
        freqs=[250e12, 300e12],
        name="monitor_n2f",
        theta=[0],
        phi=[0],
        proj_distance=1e3,
        far_field_approx=False,
    )

    monitor_n2f_far = td.FieldProjectionAngleMonitor(
        center=(0, 0, 0),
        size=(td.inf, td.inf, 0),
        freqs=[250e12, 300e12],
        name="monitor_n2f",
        theta=[0],
        phi=[0],
        proj_distance=1e5,
        far_field_approx=False,
    )

    monitor_n2f_approx = td.FieldProjectionAngleMonitor(
        center=(0, 0, 0),
        size=(td.inf, td.inf, 0),
        freqs=[250e12, 300e12],
        name="monitor_n2f",
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
    _ = td.Simulation(
        size=(1, 1, 0.3),
        structures=[],
        sources=[src],
        run_time=1e-12,
        monitors=[monitor_n2f_far],
        boundary_spec=td.BoundarySpec.all_sides(boundary=td.Periodic()),
    )
    assert_log_level(log_capture, "WARNING")

    # proj_distance not too large - don't warn
    _ = td.Simulation(
        size=(1, 1, 0.3),
        structures=[],
        sources=[src],
        run_time=1e-12,
        monitors=[monitor_n2f],
        boundary_spec=td.BoundarySpec.all_sides(boundary=td.Periodic()),
    )

    # proj_distance large but using approximations - don't warn
    _ = td.Simulation(
        size=(1, 1, 0.3),
        structures=[],
        sources=[src],
        run_time=1e-12,
        monitors=[monitor_n2f_approx],
        boundary_spec=td.BoundarySpec.all_sides(boundary=td.Periodic()),
    )


def test_proj_monitor_warnings(log_capture):
    """Test the validator that warns if projecting backwards."""

    src = td.PlaneWave(
        source_time=td.GaussianPulse(freq0=2.5e14, fwidth=1e13),
        center=(0, 0, -0.4),
        size=(td.inf, td.inf, 0),
        direction="+",
        pol_angle=-1.0,
    )

    # Cartesian monitor projecting backwards
    with AssertLogLevel(log_capture, "WARNING"):
        monitor_n2f = td.FieldProjectionCartesianMonitor(
            center=(0, 0, 0),
            size=(td.inf, td.inf, 0),
            freqs=[2.5e14],
            name="monitor_n2f",
            x=[4],
            y=[5],
            proj_distance=-1e5,
            proj_axis=2,
        )
        _ = td.Simulation(
            size=(1, 1, 1),
            structures=[],
            sources=[src],
            run_time=1e-12,
            monitors=[monitor_n2f],
        )

    # Cartesian monitor with custom origin projecting backwards
    with AssertLogLevel(log_capture, "WARNING"):
        monitor_n2f = td.FieldProjectionCartesianMonitor(
            center=(0, 0, 0),
            size=(td.inf, td.inf, 0),
            freqs=[2.5e14],
            name="monitor_n2f",
            x=[4],
            y=[5],
            proj_distance=39,
            proj_axis=2,
            custom_origin=(1, 2, -40),
        )
        _ = td.Simulation(
            size=(1, 1, 1),
            structures=[],
            sources=[src],
            run_time=1e-12,
            monitors=[monitor_n2f],
        )

    # Cartesian monitor with custom origin projecting backwards with normal_dir '-'
    with AssertLogLevel(log_capture, "WARNING"):
        monitor_n2f = td.FieldProjectionCartesianMonitor(
            center=(0, 0, 0),
            size=(td.inf, td.inf, 0),
            freqs=[2.5e14],
            name="monitor_n2f",
            x=[4],
            y=[5],
            proj_distance=41,
            proj_axis=2,
            custom_origin=(1, 2, -40),
            normal_dir="-",
        )
        _ = td.Simulation(
            size=(1, 1, 1),
            structures=[],
            sources=[src],
            run_time=1e-12,
            monitors=[monitor_n2f],
        )

    # Angle monitor projecting backwards
    with AssertLogLevel(log_capture, "WARNING"):
        monitor_n2f = td.FieldProjectionAngleMonitor(
            center=(0, 0, 0),
            size=(td.inf, td.inf, 0),
            freqs=[2.5e14],
            name="monitor_n2f",
            theta=[np.pi / 2 + 1e-2],
            phi=[0],
            proj_distance=1e3,
        )
        _ = td.Simulation(
            size=(1, 1, 1),
            structures=[],
            sources=[src],
            run_time=1e-12,
            monitors=[monitor_n2f],
        )

    # Angle monitor projecting backwards with custom origin
    with AssertLogLevel(log_capture, "WARNING"):
        monitor_n2f = td.FieldProjectionAngleMonitor(
            center=(0, 0, 0),
            size=(td.inf, td.inf, 0),
            freqs=[2.5e14],
            name="monitor_n2f",
            theta=[np.pi / 2 - 0.02],
            phi=[0],
            proj_distance=10,
            custom_origin=(0, 0, -0.5),
        )
        _ = td.Simulation(
            size=(1, 1, 1),
            structures=[],
            sources=[src],
            run_time=1e-12,
            monitors=[monitor_n2f],
        )

    # Angle monitor projecting backwards with custom origin and normal_dir '-'
    with AssertLogLevel(log_capture, "WARNING"):
        monitor_n2f = td.FieldProjectionAngleMonitor(
            center=(0, 0, 0),
            size=(td.inf, td.inf, 0),
            freqs=[2.5e14],
            name="monitor_n2f",
            theta=[np.pi / 2 + 0.02],
            phi=[0],
            proj_distance=10,
            custom_origin=(0, 0, 0.5),
            normal_dir="-",
        )
        _ = td.Simulation(
            size=(1, 1, 1),
            structures=[],
            sources=[src],
            run_time=1e-12,
            monitors=[monitor_n2f],
        )

    # Cartesian monitor using approximations but too short proj_distance
    with AssertLogLevel(log_capture, "WARNING"):
        monitor_n2f = td.FieldProjectionCartesianMonitor(
            center=(0, 0, 0),
            size=(td.inf, td.inf, 0),
            freqs=[2.5e14],
            name="monitor_n2f",
            x=[4],
            y=[5],
            proj_distance=9,
            proj_axis=2,
        )
        _ = td.Simulation(
            size=(1, 1, 1),
            structures=[],
            sources=[src],
            run_time=1e-12,
            monitors=[monitor_n2f],
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

    with pytest.raises(pydantic.ValidationError):
        _ = td.Simulation(
            size=(2, 2, 2),
            structures=[box_cond],
            sources=[src],
            run_time=1e-12,
            monitors=[monitor],
            boundary_spec=td.BoundarySpec.all_sides(boundary=td.Periodic()),
        )

    with pytest.raises(pydantic.ValidationError):
        _ = td.Simulation(
            size=(2, 2, 2),
            structures=[box_disp],
            sources=[src],
            monitors=[monitor],
            run_time=1e-12,
            boundary_spec=td.BoundarySpec.all_sides(boundary=td.Periodic()),
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
def test_sim_structure_extent(log_capture, box_size, log_level):
    """Make sure we warn if structure extends exactly to simulation edges."""

    src = td.UniformCurrentSource(
        source_time=td.GaussianPulse(freq0=3e14, fwidth=1e13),
        size=(0, 0, 0),
        polarization="Ex",
    )
    box = td.Structure(geometry=td.Box(size=box_size), medium=td.Medium(permittivity=2))
    _ = td.Simulation(
        size=(1, 1, 1),
        structures=[box],
        sources=[src],
        run_time=1e-12,
        boundary_spec=td.BoundarySpec.all_sides(boundary=td.Periodic()),
    )

    assert_log_level(log_capture, log_level)


@pytest.mark.parametrize(
    "box_length,absorb_type,log_level",
    [
        (0.0001, "PML", None),
        (1, "PML", "WARNING"),
        (1.5, "absorber", None),
        (2.0, "PML", None),
    ],
)
def test_sim_validate_structure_bounds_pml(log_capture, box_length, absorb_type, log_level):
    """Make sure we warn if structure bounds are within the PML exactly to simulation edges."""

    boundary = td.PML() if absorb_type == "PML" else td.Absorber()

    src = td.UniformCurrentSource(
        source_time=td.GaussianPulse(freq0=3e14, fwidth=1e13),
        size=(0, 0, 0),
        polarization="Ex",
    )
    box = td.Structure(
        geometry=td.Box(size=(box_length, 0.5, 0.5), center=(0, 0, 0)),
        medium=td.Medium(permittivity=2),
    )
    _ = td.Simulation(
        size=(1, 1, 1),
        structures=[box],
        grid_spec=td.GridSpec.auto(wavelength=0.001),
        sources=[src],
        run_time=1e-12,
        boundary_spec=td.BoundarySpec(
            x=td.Boundary(plus=boundary, minus=boundary),
            y=td.Boundary.pec(),
            z=td.Boundary.pec(),
        ),
    )

    assert_log_level(log_capture, log_level)


def test_num_mediums(monkeypatch):
    """Make sure we error if too many mediums supplied."""

    max_num_mediums = 10
    monkeypatch.setattr(simulation, "MAX_NUM_MEDIUMS", max_num_mediums)
    structures = []
    grid_spec = td.GridSpec.auto(wavelength=1.0)
    for i in range(max_num_mediums):
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

    with pytest.raises(pydantic.ValidationError):
        structures.append(
            td.Structure(geometry=td.Box(size=(1, 1, 1)), medium=td.Medium(permittivity=i + 2))
        )
        _ = td.Simulation(
            size=(5, 5, 5), grid_spec=grid_spec, structures=structures, run_time=1e-12
        )


def test_num_sources():
    """Make sure we error if too many sources supplied."""

    src = td.PlaneWave(
        source_time=td.GaussianPulse(freq0=2.5e14, fwidth=1e13),
        center=(0, 0, 0),
        size=(td.inf, td.inf, 0),
        direction="+",
    )

    _ = td.Simulation(size=(5, 5, 5), run_time=1e-12, sources=[src] * MAX_NUM_SOURCES)

    with pytest.raises(pydantic.ValidationError):
        _ = td.Simulation(size=(5, 5, 5), run_time=1e-12, sources=[src] * (MAX_NUM_SOURCES + 1))


def _test_names_default():
    """makes sure default names are set"""

    sim = td.Simulation(
        size=(2.0, 2.0, 2.0),
        run_time=1e-12,
        structures=[
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
        ],
        sources=[
            td.UniformCurrentSource(
                size=(0, 0, 0),
                center=(0, -0.5, 0),
                polarization="Hx",
                source_time=td.GaussianPulse(freq0=1e14, fwidth=1e12),
            ),
            td.UniformCurrentSource(
                size=(0, 0, 0),
                center=(0, -0.5, 0),
                polarization="Ex",
                source_time=td.GaussianPulse(freq0=1e14, fwidth=1e12),
            ),
            td.UniformCurrentSource(
                size=(0, 0, 0),
                center=(0, -0.5, 0),
                polarization="Ey",
                source_time=td.GaussianPulse(freq0=1e14, fwidth=1e12),
            ),
        ],
        monitors=[
            td.FluxMonitor(size=(1, 1, 0), center=(0, -0.5, 0), freqs=[1e12], name="mon1"),
            td.FluxMonitor(size=(0, 1, 1), center=(0, -0.5, 0), freqs=[1e12], name="mon2"),
            td.FluxMonitor(size=(1, 0, 1), center=(0, -0.5, 0), freqs=[1e12], name="mon3"),
        ],
        boundary_spec=td.BoundarySpec.all_sides(boundary=td.Periodic()),
    )

    for i, structure in enumerate(sim.structures):
        assert structure.name == f"structures[{i}]"

    for i, source in enumerate(sim.sources):
        assert source.name == f"sources[{i}]"


def test_names_unique():
    with pytest.raises(pydantic.ValidationError):
        _ = td.Simulation(
            size=(2.0, 2.0, 2.0),
            run_time=1e-12,
            structures=[
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
            ],
            boundary_spec=td.BoundarySpec.all_sides(boundary=td.Periodic()),
        )

    with pytest.raises(pydantic.ValidationError):
        _ = td.Simulation(
            size=(2.0, 2.0, 2.0),
            run_time=1e-12,
            sources=[
                td.UniformCurrentSource(
                    size=(0, 0, 0),
                    center=(0, -0.5, 0),
                    polarization="Hx",
                    source_time=td.GaussianPulse(freq0=1e14, fwidth=1e12),
                    name="source1",
                ),
                td.UniformCurrentSource(
                    size=(0, 0, 0),
                    center=(0, -0.5, 0),
                    polarization="Ex",
                    source_time=td.GaussianPulse(freq0=1e14, fwidth=1e12),
                    name="source1",
                ),
            ],
            boundary_spec=td.BoundarySpec.all_sides(boundary=td.Periodic()),
        )

    with pytest.raises(pydantic.ValidationError):
        _ = td.Simulation(
            size=(2.0, 2.0, 2.0),
            run_time=1e-12,
            monitors=[
                td.FluxMonitor(size=(1, 1, 0), center=(0, -0.5, 0), freqs=[1e12], name="mon1"),
                td.FluxMonitor(size=(0, 1, 1), center=(0, -0.5, 0), freqs=[1e12], name="mon1"),
            ],
            boundary_spec=td.BoundarySpec.all_sides(boundary=td.Periodic()),
        )


def test_mode_object_syms():
    """Test that errors are raised if a mode object is not placed right in the presence of syms."""
    g = td.GaussianPulse(freq0=1e12, fwidth=0.1e12)

    # wrong mode source
    with pytest.raises(pydantic.ValidationError):
        _ = td.Simulation(
            center=(1.0, -1.0, 0.5),
            size=(2.0, 2.0, 2.0),
            grid_spec=td.GridSpec.auto(wavelength=td.C_0 / 1.0),
            run_time=1e-12,
            symmetry=(1, -1, 0),
            sources=[td.ModeSource(size=(2, 2, 0), direction="+", source_time=g)],
            boundary_spec=td.BoundarySpec.all_sides(boundary=td.Periodic()),
        )

    # wrong mode monitor
    with pytest.raises(pydantic.ValidationError):
        _ = td.Simulation(
            center=(1.0, -1.0, 0.5),
            size=(2.0, 2.0, 2.0),
            grid_spec=td.GridSpec.auto(wavelength=td.C_0 / 1.0),
            run_time=1e-12,
            symmetry=(1, -1, 0),
            monitors=[
                td.ModeMonitor(size=(2, 2, 0), name="mnt", freqs=[2e12], mode_spec=td.ModeSpec())
            ],
            boundary_spec=td.BoundarySpec.all_sides(boundary=td.Periodic()),
        )

    # right mode source (centered on the symmetry)
    _ = td.Simulation(
        center=(1.0, -1.0, 0.5),
        size=(2.0, 2.0, 2.0),
        grid_spec=td.GridSpec.auto(wavelength=td.C_0 / 1.0),
        run_time=1e-12,
        symmetry=(1, -1, 0),
        sources=[td.ModeSource(center=(1, -1, 1), size=(2, 2, 0), direction="+", source_time=g)],
        boundary_spec=td.BoundarySpec.all_sides(boundary=td.Periodic()),
    )

    # right mode monitor (entirely in the main quadrant)
    _ = td.Simulation(
        center=(1.0, -1.0, 0.5),
        size=(2.0, 2.0, 2.0),
        grid_spec=td.GridSpec.auto(wavelength=td.C_0 / 1.0),
        run_time=1e-12,
        symmetry=(1, -1, 0),
        monitors=[
            td.ModeMonitor(
                center=(2, 0, 1), size=(2, 2, 0), name="mnt", freqs=[2e12], mode_spec=td.ModeSpec()
            )
        ],
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

    with pytest.raises(pydantic.ValidationError):
        _ = td.Simulation(
            size=(2.0, 2.0, 2.0),
            grid_spec=td.GridSpec.auto(wavelength=td.C_0 / 1.0),
            run_time=1e-12,
            symmetry=(0, -1, 0),
            sources=[source],
        )


def test_tfsf_boundaries(log_capture):
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
        sources=[source],
    )

    # can cross Bloch boundaries in the transverse directions
    _ = td.Simulation(
        size=(0.5, 0.5, 2.0),
        grid_spec=td.GridSpec.auto(wavelength=1.0),
        run_time=1e-12,
        sources=[source],
        boundary_spec=td.BoundarySpec(
            x=td.Boundary.bloch_from_source(source=source, domain_size=0.5, axis=0, medium=None),
            y=td.Boundary.bloch_from_source(source=source, domain_size=0.5, axis=1, medium=None),
            z=td.Boundary.pml(),
        ),
    )

    # warn if Bloch boundaries are crossed in the transverse directions but
    # the Bloch vector is incorrect
    _ = td.Simulation(
        size=(0.5, 0.5, 2.0),
        grid_spec=td.GridSpec.auto(wavelength=1.0),
        run_time=1e-12,
        sources=[source],
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
    assert_log_level(log_capture, "WARNING")

    # cannot cross any boundary in the direction of injection
    with pytest.raises(pydantic.ValidationError):
        _ = td.Simulation(
            size=(2.0, 2.0, 0.5),
            grid_spec=td.GridSpec.auto(wavelength=1.0),
            run_time=1e-12,
            sources=[source],
        )

    # cannot cross any non-periodic boundary in the transverse direction
    with pytest.raises(pydantic.ValidationError):
        _ = td.Simulation(
            center=(0.5, 0, 0),  # also check the case when the boundary is crossed only on one side
            size=(0.5, 0.5, 2.0),
            grid_spec=td.GridSpec.auto(wavelength=1.0),
            run_time=1e-12,
            sources=[source],
            boundary_spec=td.BoundarySpec(
                x=td.Boundary.pml(),
                y=td.Boundary.absorber(),
            ),
        )


def test_tfsf_structures_grid(log_capture):
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
    sim = td.Simulation(
        size=(2.0, 2.0, 2.0),
        grid_spec=td.GridSpec.auto(wavelength=1.0),
        run_time=1e-12,
        sources=[source],
        structures=[
            td.Structure(
                geometry=td.Box(center=(0, 0, -1), size=(0.5, 0.5, 0.5)),
                medium=td.Medium(permittivity=2),
            )
        ],
    )
    sim.validate_pre_upload()
    assert_log_level(log_capture, "WARNING")

    # must not have different material profiles on different faces along the injection axis
    with pytest.raises(SetupError):
        sim = td.Simulation(
            size=(2.0, 2.0, 2.0),
            grid_spec=td.GridSpec.auto(wavelength=1.0),
            run_time=1e-12,
            sources=[source],
            structures=[
                td.Structure(
                    geometry=td.Box(center=(0.5, 0, 0), size=(0.25, 0.25, 0.25)),
                    medium=td.Medium(permittivity=2),
                )
            ],
        )
        sim.validate_pre_upload()

    # different structures *are* allowed on different faces as long as material properties match
    sim = td.Simulation(
        size=(2.0, 2.0, 2.0),
        grid_spec=td.GridSpec.auto(wavelength=1.0),
        run_time=1e-12,
        sources=[source],
        structures=[
            td.Structure(
                geometry=td.Box(center=(0.5, 0, 0), size=(0.25, 0.25, 0.25)), medium=td.Medium()
            )
        ],
    )

    # TFSF box must not intersect a custom medium
    Nx, Ny, Nz = 10, 9, 8
    X = np.linspace(-1, 1, Nx)
    Y = np.linspace(-1, 1, Ny)
    Z = np.linspace(-1, 1, Nz)
    data = np.ones((Nx, Ny, Nz, 1))
    eps_diagonal_data = td.ScalarFieldDataArray(data, coords=dict(x=X, y=Y, z=Z, f=[td.C_0]))
    eps_components = {f"eps_{d}{d}": eps_diagonal_data for d in "xyz"}
    eps_dataset = td.PermittivityDataset(**eps_components)
    custom_medium = td.CustomMedium(eps_dataset=eps_dataset, name="my_medium")
    sim = td.Simulation(
        size=(2.0, 2.0, 2.0),
        grid_spec=td.GridSpec.auto(wavelength=1.0),
        run_time=1e-12,
        sources=[source],
        structures=[
            td.Structure(
                geometry=td.Box(center=(0.5, 0, 0), size=(td.inf, td.inf, 0.25)),
                medium=custom_medium,
            )
        ],
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
        sources=[source],
        structures=[
            td.Structure(
                geometry=td.Box(center=(0.5, 0, 0), size=(td.inf, td.inf, 0.25)),
                medium=anisotropic_medium,
            )
        ],
    )
    with pytest.raises(SetupError):
        sim.validate_pre_upload()


@pytest.mark.parametrize(
    "size, num_struct, log_level", [(1, 1, None), (50, 1, "WARNING"), (1, 11000, "WARNING")]
)
def test_warn_large_epsilon(log_capture, size, num_struct, log_level):
    """Make sure we get a warning if the epsilon grid is too large."""

    structures = [
        td.Structure(
            geometry=td.Box(center=(0, 0, 0), size=(0.1, 0.1, 0.1)),
            medium=td.Medium(permittivity=1.0),
        )
        for _ in range(num_struct)
    ]

    sim = td.Simulation(
        size=(size, size, size),
        grid_spec=td.GridSpec.uniform(dl=0.1),
        run_time=1e-12,
        boundary_spec=td.BoundarySpec.all_sides(boundary=td.Periodic()),
        sources=[
            td.ModeSource(
                center=(0, 0, 0),
                size=(td.inf, td.inf, 0),
                direction="+",
                source_time=td.GaussianPulse(freq0=1e12, fwidth=0.1e12),
            )
        ],
        structures=structures,
    )
    sim.epsilon(box=td.Box(size=(size, size, size)))
    assert_log_level(log_capture, log_level)


@pytest.mark.parametrize("dl, log_level", [(0.1, None), (0.005, "WARNING")])
def test_warn_large_mode_monitor(log_capture, dl, log_level):
    """Make sure we get a warning if the mode monitor grid is too large."""

    sim = td.Simulation(
        size=(2.0, 2.0, 2.0),
        grid_spec=td.GridSpec.uniform(dl=dl),
        run_time=1e-12,
        sources=[
            td.ModeSource(
                size=(0.4, 0.4, 0),
                direction="+",
                source_time=td.GaussianPulse(freq0=1e12, fwidth=0.1e12),
            )
        ],
        monitors=[
            td.ModeMonitor(
                size=(td.inf, 0, td.inf), freqs=[1e12], name="test", mode_spec=td.ModeSpec()
            )
        ],
    )
    sim.validate_pre_upload()
    assert_log_level(log_capture, log_level)


@pytest.mark.parametrize("dl, log_level", [(0.1, None), (0.005, "WARNING")])
def test_warn_large_mode_source(log_capture, dl, log_level):
    """Make sure we get a warning if the mode source grid is too large."""

    sim = td.Simulation(
        size=(2.0, 2.0, 2.0),
        grid_spec=td.GridSpec.uniform(dl=dl),
        run_time=1e-12,
        sources=[
            td.ModeSource(
                size=(td.inf, td.inf, 0),
                direction="+",
                source_time=td.GaussianPulse(freq0=1e12, fwidth=0.1e12),
            )
        ],
    )
    sim.validate_pre_upload()
    assert_log_level(log_capture, log_level)


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
        sources=[
            td.ModeSource(
                size=(0.1, 0.1, 0),
                direction="+",
                source_time=td.GaussianPulse(freq0=1e12, fwidth=0.1e12),
            )
        ],
        monitors=[monitor],
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
        sources=[
            td.ModeSource(
                size=(0.1, 0.1, 0),
                direction="+",
                source_time=td.GaussianPulse(freq0=2e14, fwidth=0.1e14),
            )
        ],
    )

    # simulation with a 0D time monitor should not error
    monitor = td.FieldTimeMonitor(center=(0, 0, 0), size=(0, 0, 0), name="time")
    sim = sim.updated_copy(monitors=[monitor])
    sim.validate_pre_upload()

    # 1D monitor should error
    with pytest.raises(SetupError):
        monitor = monitor.updated_copy(size=(1, 0, 0))
        sim = sim.updated_copy(monitors=[monitor])
        sim.validate_pre_upload()

    # setting a large enough interval should again not error
    monitor = monitor.updated_copy(interval=20)
    sim = sim.updated_copy(monitors=[monitor])
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
def test_warn_time_monitor_outside_run_time(log_capture, start, log_level):
    """Make sure we get a warning if the mode monitor grid is too large."""

    sim = td.Simulation(
        size=(2.0, 2.0, 2.0),
        grid_spec=td.GridSpec.uniform(dl=0.1),
        run_time=1e-12,
        sources=[
            td.ModeSource(
                size=(0.4, 0.4, 0),
                direction="+",
                source_time=td.GaussianPulse(freq0=1e12, fwidth=0.1e12),
            )
        ],
        monitors=[td.FieldTimeMonitor(size=(td.inf, 0, td.inf), start=start, name="test")],
    )
    with AssertLogLevel(log_capture, log_level_expected=log_level, contains_str="start time"):
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
    sim_new = sim.copy(update=dict(structures=[structure]))
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
        structures=[box],
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


def test_sim_volumetric_structures(log_capture, tmp_path):
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
    )
    for struct in [box, cyl, pslab]:
        sim = td.Simulation(
            size=(10, 10, 10),
            structures=[struct],
            sources=[src],
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
        structures=[below, box],
        sources=[src],
        monitors=[monitor],
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
        structures=[below, box],
        sources=[src],
        monitors=[monitor],
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
    with AssertLogLevel(log_capture, None):
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
        structures=[below_half, box],
        sources=[src],
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
        structures=[box, below],
        sources=[src],
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
    with pytest.raises(pydantic.ValidationError):
        sim = td.Simulation(
            size=(10, 10, 10),
            structures=[],
            sources=[src],
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
    with pytest.raises(pydantic.ValidationError):
        _ = td.Structure(geometry=td.Box(center=(0, 0, 0), size=(1, 1, 1)), medium=box.medium)
    with pytest.raises(pydantic.ValidationError):
        _ = td.Structure(geometry=td.Cylinder(radius=1, length=1), medium=box.medium)
    with pytest.raises(pydantic.ValidationError):
        _ = td.Structure(
            geometry=td.PolySlab(vertices=[(0, 0), (1, 0), (1, 1)], slab_bounds=(-1, 1)),
            medium=box.medium,
        )
    with pytest.raises(pydantic.ValidationError):
        _ = td.Structure(geometry=td.Sphere(radius=1), medium=box.medium)

    # test warning for 2d geometry in simulation without Medium2D
    with AssertLogLevel(log_capture, "WARNING"):
        struct = td.Structure(medium=td.Medium(), geometry=td.Box(size=(1, 0, 1)))
        sim = td.Simulation(
            size=(10, 10, 10),
            structures=[struct],
            sources=[src],
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
        sources=[
            td.PointDipole(
                center=(0, 0, 0),
                polarization="Ex",
                source_time=td.GaussianPulse(
                    freq0=1e14,
                    fwidth=1e12,
                ),
            )
        ],
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
        structures=[struct],
    )
    assert not sim.allow_gain
    sim = sim.updated_copy(structures=[struct_gain])
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

    coords = dict(x=[1, 2], y=[3, 4], z=z)
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
        structures=[struct],
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
        **SIM_FULL.dict(exclude={"structures", "medium"}),
    )

    assert sim == SIM_FULL


def test_to_gds(tmp_path):
    sim = td.Simulation(
        size=(2.0, 2.0, 2.0),
        run_time=1e-12,
        structures=[
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
        ],
        sources=[
            td.PointDipole(
                center=(0, 0, 0),
                polarization="Ex",
                source_time=td.GaussianPulse(freq0=1e14, fwidth=1e12),
            )
        ],
        monitors=[
            td.FieldMonitor(size=(0, 0, 0), center=(0, 0, 0), freqs=[1e12, 2e12], name="point"),
        ],
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
def test_sim_subsection(unstructured, nz, log_capture):
    region = td.Box(size=(0.3, 0.5, 0.7), center=(0.1, 0.05, 0.02))
    region_xy = td.Box(size=(0.3, 0.5, 0), center=(0.1, 0.05, 0.02))

    sim_red = SIM_FULL.subsection(region=region)
    # Ensure that in this first test case the lumped element is safely excluded
    _ = sim_red.volumetric_structures
    assert_log_level(log_capture, "WARNING")
    assert sim_red.structures != SIM_FULL.structures
    sim_red = SIM_FULL.subsection(
        region=region,
        symmetry=(1, 0, -1),
        monitors=[mnt for mnt in SIM_FULL.monitors if not isinstance(mnt, td.ModeMonitor)],
    )
    assert sim_red.symmetry == (1, 0, -1)
    sim_red = SIM_FULL.subsection(
        region=region, boundary_spec=td.BoundarySpec.all_sides(td.Periodic())
    )
    sim_red = SIM_FULL.subsection(region=region, sources=[], grid_spec=td.GridSpec.uniform(dl=20))
    assert len(sim_red.sources) == 0
    sim_red = SIM_FULL.subsection(region=region, monitors=[])
    assert len(sim_red.monitors) == 0
    sim_red = SIM_FULL.subsection(region=region, remove_outside_structures=False)
    assert len(sim_red.structures) == len(SIM_FULL.structures)
    for strc_red, strc in zip(sim_red.structures, SIM_FULL.structures):
        if strc.medium.nonlinear_spec is None:
            assert strc == strc_red
    sim_red = SIM_FULL.subsection(region=region, remove_outside_custom_mediums=True)

    perm = td.SpatialDataArray(
        1 + np.random.random((11, 12, nz)),
        coords=dict(
            x=np.linspace(-0.51, 0.52, 11),
            y=np.linspace(-1.02, 1.04, 12),
            z=np.linspace(-1.51, 1.51, nz),
        ),
    )

    if unstructured:
        perm = cartesian_to_unstructured(perm, seed=523)

    fine_custom_medium = td.CustomMedium(permittivity=perm)

    sim = SIM_FULL.updated_copy(
        structures=[
            td.Structure(
                geometry=td.Box(size=(1, 2, 3)),
                medium=fine_custom_medium,
            )
        ],
        medium=fine_custom_medium,
    )
    sim_red = sim.subsection(region=region, remove_outside_custom_mediums=True)

    # check automatic symmetry expansion
    sim_sym = SIM_FULL.updated_copy(
        symmetry=(-1, 0, 1),
        sources=[src for src in SIM_FULL.sources if not isinstance(src, td.TFSF)],
    )
    sim_red = sim_sym.subsection(region=region)
    assert np.allclose(sim_red.center, (0, 0.05, 0.0))

    # check grid is preserved when requested
    sim_red = SIM_FULL.subsection(
        region=region, grid_spec="identical", boundary_spec=td.BoundarySpec.all_sides(td.Periodic())
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

    subsection_monitors = [mnt for mnt in SIM_FULL.monitors if region_xy.intersects(mnt)]
    sim_red = SIM_FULL.subsection(
        region=region_xy,
        grid_spec="identical",
        boundary_spec=td.BoundarySpec.all_sides(td.Periodic()),
        # Set theta to 'pi/2' for 2D simulation in the x-y plane
        monitors=[
            mnt.updated_copy(theta=np.pi / 2)
            if isinstance(mnt, td.FieldProjectionAngleMonitor)
            else mnt
            for mnt in subsection_monitors
            if not isinstance(
                mnt, (td.FieldProjectionCartesianMonitor, td.FieldProjectionKSpaceMonitor)
            )
        ],
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
        geometry=td.Box(size=plane_size, center=[plane_pos, 0, 0]), medium=med_2d
    )

    structures = [face, left_top, right_top, bottom, plane_material]

    uni_grid = td.UniformGrid(dl=wavelength0 / 1000)

    sim_td = td.Simulation(
        center=center_sim,
        size=size_sim,
        grid_spec=td.GridSpec(grid_x=uni_grid, grid_y=uni_grid, grid_z=uni_grid),
        structures=structures,
        sources=[],
        monitors=[],
        run_time=1e-12,
    )

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
    eps_diagonal_data = td.ScalarFieldDataArray(data, coords=dict(x=X, y=Y, z=Z, f=[td.C_0]))
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
        sources=[source],
        structures=[],
    )

    for pair in compatible_pairs:
        struct1 = td.Structure(geometry=td.Box(size=(1, 1, 1), center=(0, 0, 0.5)), medium=pair[0])
        struct2 = td.Structure(geometry=td.Box(size=(1, 1, 1), center=(0, 0, -0.5)), medium=pair[1])
        # this pair can intersect
        sim = sim.updated_copy(structures=[struct1, struct2])

    for pair in incompatible_pairs:
        struct1 = td.Structure(geometry=td.Box(size=(1, 1, 1), center=(0, 0, 0.5)), medium=pair[0])
        struct2 = td.Structure(geometry=td.Box(size=(1, 1, 1), center=(0, 0, -0.5)), medium=pair[1])
        # this pair cannot intersect
        with pytest.raises(pydantic.ValidationError):
            sim = sim.updated_copy(structures=[struct1, struct2])

    for pair in incompatible_pairs:
        struct1 = td.Structure(geometry=td.Box(size=(1, 1, 1), center=(0, 0, 0.75)), medium=pair[0])
        struct2 = td.Structure(
            geometry=td.Box(size=(1, 1, 1), center=(0, 0, -0.75)), medium=pair[1]
        )
        # it's ok if these are both present as long as they don't intersect
        sim = sim.updated_copy(structures=[struct1, struct2])


def test_num_lumped_elements():
    """Make sure we error if too many lumped elements supplied."""

    resistor = td.LumpedResistor(
        size=(0, 1, 2), center=(0, 0, 0), name="R1", voltage_axis=2, resistance=75
    )
    grid_spec = td.GridSpec.auto(wavelength=1.0)

    _ = td.Simulation(
        size=(5, 5, 5),
        grid_spec=grid_spec,
        structures=[],
        lumped_elements=[resistor] * MAX_NUM_MEDIUMS,
        run_time=1e-12,
    )
    with pytest.raises(pydantic.ValidationError):
        _ = td.Simulation(
            size=(5, 5, 5),
            grid_spec=grid_spec,
            structures=[],
            lumped_elements=[resistor] * (MAX_NUM_MEDIUMS + 1),
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
        lumped_elements=[resistor],
    )
    # error for 1D/2D simulation with lumped elements
    with pytest.raises(pydantic.ValidationError):
        td.Simulation(
            size=(1, 0, 3),
            run_time=1e-12,
            grid_spec=td.GridSpec.uniform(dl=0.1),
            lumped_elements=[resistor],
        )

    with pytest.raises(pydantic.ValidationError):
        td.Simulation(
            size=(1, 0, 0),
            run_time=1e-12,
            grid_spec=td.GridSpec.uniform(dl=0.1),
            lumped_elements=[resistor],
        )


def test_suggested_mesh_overrides():
    resistor = td.LumpedResistor(
        size=(0, 1, 2), center=(0, 0, 0), name="R1", voltage_axis=2, resistance=75
    )
    sim = td.Simulation(
        size=(1, 2, 3),
        run_time=1e-12,
        grid_spec=td.GridSpec.uniform(dl=0.1),
        lumped_elements=[resistor],
    )

    def update_sim_with_suggested_overrides(sim):
        suggested_mesh_overrides = sim.suggest_mesh_overrides()
        assert len(suggested_mesh_overrides) == 2
        grid_spec = sim.grid_spec.copy(
            update={
                "override_structures": list(sim.grid_spec.override_structures)
                + suggested_mesh_overrides,
            }
        )

        return sim.updated_copy(
            grid_spec=grid_spec,
        )

    _ = update_sim_with_suggested_overrides(sim)

    coax_resistor = td.CoaxialLumpedResistor(
        resistance=50.0,
        center=[0, 0, 0],
        outer_diameter=2,
        inner_diameter=0.5,
        normal_axis=0,
        name="R",
    )

    sim = sim.updated_copy(
        lumped_elements=[coax_resistor],
        grid_spec=td.GridSpec.uniform(dl=0.1),
    )

    _ = update_sim_with_suggested_overrides(sim)


def test_run_time_spec():
    run_time_spec = td.RunTimeSpec(quality_factor=3.0)

    sim = SIM_FULL.updated_copy(run_time=run_time_spec)

    assert sim._run_time > 0


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

    sim = SIM.updated_copy(sources=[mode_source])

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
        sources=[mode_source],
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
    sim = SIM.updated_copy(monitors=[mode_monitor])
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
    with pytest.raises(pydantic.ValidationError):
        sim = td.Simulation(
            size=(2, 2, 2),
            run_time=1e-12,
            grid_spec=td.GridSpec(wavelength=1.0),
            sources=[mode_source],
        )
    # check that a monitor at y+ simulation domain edge errors
    with pytest.raises(pydantic.ValidationError):
        sim = td.Simulation(
            size=(2, 2, 2),
            run_time=1e-12,
            grid_spec=td.GridSpec(wavelength=1.0),
            monitors=[mode_monitor],
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
        structures=[sphere],
        sources=[source],
        monitors=[flux_r_mnt],
        size=sim_size,
        grid_spec=td.GridSpec.auto(min_steps_per_wvl=15),
        boundary_spec=td.BoundarySpec(
            x=td.Boundary.absorber(), y=td.Boundary.periodic(), z=td.Boundary.periodic()
        ),
        run_time=10 / fwidth,
    )

    assert sim._is_fixed_angle

    with pytest.raises(pydantic.ValidationError):
        _ = sim.updated_copy(
            boundary_spec=td.BoundarySpec(
                x=td.Boundary.pml(),
                y=td.Boundary.bloch_from_source(source=source, axis=1, domain_size=2.2),
                z=td.Boundary.bloch_from_source(source=source, axis=2, domain_size=2.2),
            )
        )

    with pytest.raises(pydantic.ValidationError):
        _ = sim.updated_copy(med=td.Medium(conductivity=0.001))

    anisotropic_med = td.FullyAnisotropicMedium(permittivity=[[2, 0, 0], [0, 1, 0], [0, 0, 3]])
    with pytest.raises(pydantic.ValidationError):
        _ = sim.updated_copy(structures=[sphere.updated_copy(medium=anisotropic_med)])

    with pytest.raises(pydantic.ValidationError):
        _ = sim.updated_copy(sources=[source, source])

    with pytest.raises(pydantic.ValidationError):
        _ = sim.updated_copy(
            structures=[sphere.updated_copy(medium=td.Medium(conductivity=-0.1, allow_gain=True))]
        )

    with pytest.raises(pydantic.ValidationError):
        _ = sim.updated_copy(monitors=[td.FieldTimeMonitor(size=[td.inf, td.inf, 0], name="time")])

    with pytest.raises(pydantic.ValidationError):
        _ = sim.updated_copy(monitors=[td.FluxTimeMonitor(size=[td.inf, td.inf, 0], name="time")])

    nonlinear_med = td.Medium(
        permittivity=3,
        nonlinear_spec=td.NonlinearSpec(
            models=[
                td.KerrNonlinearity(n2=-1 + 1j, n0=1),
            ],
            num_iters=20,
        ),
    )
    with pytest.raises(pydantic.ValidationError):
        _ = sim.updated_copy(structures=[sphere.updated_copy(medium=nonlinear_med)])

    time_modulated_med = td.Medium(
        permittivity=2,
        modulation_spec=td.ModulationSpec(
            permittivity=td.SpaceTimeModulation(
                time_modulation=td.ContinuousWaveTimeModulation(freq0=td.C_0, amplitude=1, phase=0),
            )
        ),
    )
    with pytest.raises(pydantic.ValidationError):
        _ = sim.updated_copy(structures=[sphere.updated_copy(medium=time_modulated_med)])
