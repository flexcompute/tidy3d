"""Tests the simulation and its validators."""
import pytest
import pydantic
import matplotlib.pylab as plt

import numpy as np
import tidy3d as td
from tidy3d.exceptions import SetupError, ValidationError, Tidy3dKeyError
from tidy3d.components import simulation
from tidy3d.components.simulation import MAX_NUM_MEDIUMS
from ..utils import assert_log_level, SIM_FULL, log_capture, run_emulated, clear_tmp
from tidy3d.constants import LARGE_NUMBER

SIM = td.Simulation(size=(1, 1, 1), run_time=1e-12, grid_spec=td.GridSpec(wavelength=1.0))

_, AX = plt.subplots()

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
            td.FieldMonitor(size=(0, 0, 0), center=(0, 0, 0), freqs=[1, 2], name="point"),
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

    dt = sim.dt
    tm = sim.tmesh
    sim.validate_pre_upload()
    ms = sim.mediums
    mm = sim.medium_map
    m = sim.get_monitor_by_name("point")
    s = sim.background_structure
    # sim.plot(x=0)
    # sim.plot_eps(x=0)
    sim.num_pml_layers
    # sim.plot_grid(x=0)
    sim.frequency_range
    sim.grid
    sim.num_cells
    sim.discretize(m)
    sim.epsilon(m)


def test_deprecation_defaults(log_capture):
    """Make sure deprecation warnings NOT thrown if defaults used."""
    s = td.Simulation(
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

        sim = td.Simulation(
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

    size = (1, 1, 1)
    medium = td.Medium(frequency_range=(2, 3))
    box = td.Structure(geometry=td.Box(size=(0.1, 0.1, 0.1)), medium=medium)
    mnt = td.FieldMonitor(size=(0, 0, 0), name="freq", freqs=[freq])
    src = td.UniformCurrentSource(
        source_time=td.GaussianPulse(freq0=2.5, fwidth=0.5),
        size=(0, 0, 0),
        polarization="Ex",
    )
    sim = td.Simulation(
        size=(1, 1, 1),
        structures=[box],
        monitors=[mnt],
        sources=[src],
        run_time=1e-12,
        boundary_spec=td.BoundarySpec.all_sides(boundary=td.Periodic()),
    )
    assert_log_level(log_capture, log_level)


@pytest.mark.parametrize("fwidth, log_level", [(0.1, "WARNING"), (2, "INFO")])
def test_monitor_simulation_frequency_range(log_capture, fwidth, log_level):
    # monitor frequency outside of the simulation's frequency range should throw a warning

    size = (1, 1, 1)
    src = td.UniformCurrentSource(
        source_time=td.GaussianPulse(freq0=2.0, fwidth=fwidth),
        size=(0, 0, 0),
        polarization="Ex",
    )
    mnt = td.FieldMonitor(size=(0, 0, 0), name="freq", freqs=[1.5])
    sim = td.Simulation(
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
        source_time=td.GaussianPulse(freq0=2.0, fwidth=1.0),
        size=(0, 0, 0),
        polarization="Ex",
    )
    src0 = td.UniformCurrentSource(
        source_time=td.GaussianPulse(freq0=2.0, fwidth=1.0, amplitude=0),
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
        x=td.Boundary.bloch(bloch_vec=-3 + bspec2.x.plus.bloch_vec),
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

    # angled incidence plane wave with an integer-offset Bloch vector should warn
    td.Simulation(
        size=(1, 1, 1),
        run_time=1e-12,
        sources=[src2],
        boundary_spec=bspec3,
    )
    assert_log_level(log_capture, "WARNING")

    # angled incidence plane wave with wrong Bloch vector should warn
    td.Simulation(
        size=(1, 1, 1),
        run_time=1e-12,
        sources=[src2],
        boundary_spec=bspec4,
    )
    assert_log_level(log_capture, "WARNING")


def test_validate_zero_dim_boundaries(log_capture):

    # zero-dim simulation with an absorbing boundary in that direction should warn
    src = td.PlaneWave(
        source_time=td.GaussianPulse(freq0=2.5e14, fwidth=1e13),
        center=(0, 0, 0),
        size=(td.inf, 0, td.inf),
        direction="+",
        pol_angle=0.0,
    )

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
    assert_log_level(log_capture, "WARNING")

    # zero-dim simulation with an absorbing boundary any other direction should not warn
    td.Simulation(
        size=(1, 1, 0),
        run_time=1e-12,
        sources=[src],
        boundary_spec=td.BoundarySpec(
            x=td.Boundary.pml(),
            y=td.Boundary.stable_pml(),
            z=td.Boundary.pec(),
        ),
    )


def test_validate_components_none():

    assert SIM._structures_not_at_edges(val=None, values=SIM.dict()) is None
    assert SIM._validate_num_mediums(val=None) is None
    assert SIM._warn_monitor_mediums_frequency_range(val=None, values=SIM.dict()) is None
    assert SIM._warn_monitor_simulation_frequency_range(val=None, values=SIM.dict()) is None
    assert SIM._warn_grid_size_too_small(val=None, values=SIM.dict()) is None
    assert SIM._source_homogeneous_isotropic(val=None, values=SIM.dict()) is None


def test_sources_edge_case_validation():
    values = SIM.dict()
    values.pop("sources")
    with pytest.raises(ValidationError):
        SIM._warn_monitor_simulation_frequency_range(val="test", values=values)


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
    s = SIM.copy(update=dict(monitors=(td.FieldMonitor(name="f", freqs=[1], size=(1, 1, 1)),)))
    s._validate_monitor_size()
    assert_log_level(log_capture, "WARNING")

    # error for simulation size
    monkeypatch.setattr(simulation, "MAX_SIMULATION_DATA_SIZE_GB", 1 / 2**30)
    with pytest.raises(SetupError):
        s = SIM.copy(update=dict(monitors=(td.FieldMonitor(name="f", freqs=[1], size=(1, 1, 1)),)))
        s._validate_monitor_size()


def test_no_monitor():
    with pytest.raises(Tidy3dKeyError):
        SIM.get_monitor_by_name("NOPE")


def test_plot_eps():
    ax = SIM_FULL.plot_eps(ax=AX, x=0)
    SIM_FULL._add_cbar(eps_min=1, eps_max=2, ax=ax)


def test_plot():
    SIM_FULL.plot(x=0, ax=AX)


def test_plot_3d():
    SIM_FULL.plot_3d()


def test_structure_alpha():
    _ = SIM_FULL.plot_structures_eps(x=0, ax=AX, alpha=None)
    _ = SIM_FULL.plot_structures_eps(x=0, ax=AX, alpha=-1)
    _ = SIM_FULL.plot_structures_eps(x=0, ax=AX, alpha=1)
    _ = SIM_FULL.plot_structures_eps(x=0, ax=AX, alpha=0.5)
    _ = SIM_FULL.plot_structures_eps(x=0, ax=AX, alpha=0.5, cbar=True)
    new_structs = [
        td.Structure(geometry=s.geometry, medium=SIM_FULL.medium) for s in SIM_FULL.structures
    ]
    S2 = SIM_FULL.copy(update=dict(structures=new_structs))
    ax5 = S2.plot_structures_eps(x=0, ax=AX, alpha=0.5)


def test_plot_symmetries():
    S2 = SIM.copy(update=dict(symmetry=(1, 0, -1)))
    S2.plot_symmetries(x=0, ax=AX)


def test_plot_grid():
    override = td.Structure(geometry=td.Box(size=(1, 1, 1)), medium=td.Medium())
    S2 = SIM_FULL.copy(
        update=dict(grid_spec=td.GridSpec(wavelength=1.0, override_structures=[override]))
    )
    S2.plot_grid(x=0)


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

    # fake a scenario where the fmax of the simulation is negative?
    class MockSim:
        frequency_range = (-2, -1)
        _cached_properties = {}

    m = MockSim()
    assert td.Simulation.nyquist_step.fget(m) == 1


def test_min_sym_box():
    S = SIM.copy(update=dict(symmetry=(1, 1, 1)))
    b = td.Box(center=(-2, -2, -2), size=(1, 1, 1))
    S.min_sym_box(b)
    b = td.Box(center=(2, 2, 2), size=(1, 1, 1))
    S.min_sym_box(b)
    b = td.Box(size=(1, 1, 1))
    S.min_sym_box(b)


def test_discretize_non_intersect(log_capture):
    SIM.discretize(box=td.Box(center=(-20, -20, -20), size=(1, 1, 1)))
    assert_log_level(log_capture, "ERROR")


def test_filter_structures():
    s1 = td.Structure(geometry=td.Box(size=(1, 1, 1)), medium=SIM.medium)
    s2 = td.Structure(geometry=td.Box(size=(1, 1, 1), center=(1, 1, 1)), medium=SIM.medium)
    plane = td.Box(center=(0, 0, 1.5), size=(td.inf, td.inf, 0))
    SIM._filter_structures_plane(structures=[s1, s2], plane=plane)


def test_get_structure_plot_params():
    pp = SIM_FULL._get_structure_plot_params(mat_index=0, medium=SIM_FULL.medium)
    assert pp.facecolor == "white"
    pp = SIM_FULL._get_structure_plot_params(mat_index=1, medium=td.PEC)
    assert pp.facecolor == "gold"
    pp = SIM_FULL._get_structure_eps_plot_params(
        medium=SIM_FULL.medium, freq=1, eps_min=1, eps_max=2
    )
    assert float(pp.facecolor) == 1.0
    pp = SIM_FULL._get_structure_eps_plot_params(medium=td.PEC, freq=1, eps_min=1, eps_max=2)
    assert pp.facecolor == "gold"


def test_warn_sim_background_medium_freq_range(log_capture):
    S = SIM.copy(
        update=dict(
            sources=(
                td.PointDipole(
                    polarization="Ex", source_time=td.GaussianPulse(freq0=2e14, fwidth=1e11)
                ),
            ),
            monitors=(td.FluxMonitor(name="test", freqs=[2], size=(1, 1, 0)),),
            medium=td.Medium(frequency_range=(0, 1)),
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


@pytest.mark.parametrize("box_size,log_level", [(0.001, "INFO"), (9.9, "WARNING"), (20, "INFO")])
def test_sim_structure_gap(log_capture, box_size, log_level):
    """Make sure the gap between a structure and PML is not too small compared to lambda0."""
    medium = td.Medium(permittivity=2)
    box = td.Structure(geometry=td.Box(size=(box_size, box_size, box_size)), medium=medium)
    src = td.UniformCurrentSource(
        source_time=td.GaussianPulse(freq0=3e14, fwidth=1e13),
        size=(0, 0, 0),
        polarization="Ex",
    )
    sim = td.Simulation(
        size=(10, 10, 10),
        structures=[box],
        sources=[src],
        boundary_spec=td.BoundarySpec(
            x=td.Boundary.pml(num_layers=5),
            y=td.Boundary.pml(num_layers=5),
            z=td.Boundary.pml(num_layers=5),
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

    monitor_n2f = td.FieldProjectionAngleMonitor(
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

    monitor_diffraction = td.DiffractionMonitor(
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
        sim1 = td.Simulation(
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

    mediums = td.Simulation.intersecting_media(monitor_n2f_vol, [box])
    assert len(mediums) == 1
    mediums = td.Simulation.intersecting_media(monitor_n2f_vol, [box_transparent])
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
    sim = td.Simulation(
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
        (0, "PML", None),
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
    sim = td.Simulation(
        size=(1, 1, 1),
        structures=[box],
        grid_spec=td.GridSpec.auto(wavelength=0.001),
        sources=[src],
        run_time=1e-12,
        boundary_spec=td.BoundarySpec.pml(x=True, y=False, z=False),
    )

    assert_log_level(log_capture, log_level)


def test_num_mediums():
    """Make sure we error if too many mediums supplied."""

    structures = []
    grid_spec = td.GridSpec.auto(wavelength=1.0)
    for i in range(MAX_NUM_MEDIUMS):
        structures.append(
            td.Structure(geometry=td.Box(size=(1, 1, 1)), medium=td.Medium(permittivity=i + 1))
        )
    sim = td.Simulation(
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
        sim = td.Simulation(
            size=(5, 5, 5), grid_spec=grid_spec, structures=structures, run_time=1e-12
        )


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
            td.FluxMonitor(size=(1, 1, 0), center=(0, -0.5, 0), freqs=[1], name="mon1"),
            td.FluxMonitor(size=(0, 1, 1), center=(0, -0.5, 0), freqs=[1], name="mon2"),
            td.FluxMonitor(size=(1, 0, 1), center=(0, -0.5, 0), freqs=[1], name="mon3"),
        ],
        boundary_spec=td.BoundarySpec.all_sides(boundary=td.Periodic()),
    )

    for i, structure in enumerate(sim.structures):
        assert structure.name == f"structures[{i}]"

    for i, source in enumerate(sim.sources):
        assert source.name == f"sources[{i}]"


def test_names_unique():

    with pytest.raises(pydantic.ValidationError) as e:
        sim = td.Simulation(
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

    with pytest.raises(pydantic.ValidationError) as e:
        sim = td.Simulation(
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

    with pytest.raises(pydantic.ValidationError) as e:
        sim = td.Simulation(
            size=(2.0, 2.0, 2.0),
            run_time=1e-12,
            monitors=[
                td.FluxMonitor(size=(1, 1, 0), center=(0, -0.5, 0), freqs=[1], name="mon1"),
                td.FluxMonitor(size=(0, 1, 1), center=(0, -0.5, 0), freqs=[1], name="mon1"),
            ],
            boundary_spec=td.BoundarySpec.all_sides(boundary=td.Periodic()),
        )


def test_mode_object_syms():
    """Test that errors are raised if a mode object is not placed right in the presence of syms."""
    g = td.GaussianPulse(freq0=1, fwidth=0.1)

    # wrong mode source
    with pytest.raises(pydantic.ValidationError) as e_info:
        sim = td.Simulation(
            center=(1.0, -1.0, 0.5),
            size=(2.0, 2.0, 2.0),
            grid_spec=td.GridSpec.auto(wavelength=td.C_0 / 1.0),
            run_time=1e-12,
            symmetry=(1, -1, 0),
            sources=[td.ModeSource(size=(2, 2, 0), direction="+", source_time=g)],
            boundary_spec=td.BoundarySpec.all_sides(boundary=td.Periodic()),
        )

    # wrong mode monitor
    with pytest.raises(pydantic.ValidationError) as e_info:
        sim = td.Simulation(
            center=(1.0, -1.0, 0.5),
            size=(2.0, 2.0, 2.0),
            grid_spec=td.GridSpec.auto(wavelength=td.C_0 / 1.0),
            run_time=1e-12,
            symmetry=(1, -1, 0),
            monitors=[
                td.ModeMonitor(size=(2, 2, 0), name="mnt", freqs=[2], mode_spec=td.ModeSpec())
            ],
            boundary_spec=td.BoundarySpec.all_sides(boundary=td.Periodic()),
        )

    # right mode source (centered on the symmetry)
    sim = td.Simulation(
        center=(1.0, -1.0, 0.5),
        size=(2.0, 2.0, 2.0),
        grid_spec=td.GridSpec.auto(wavelength=td.C_0 / 1.0),
        run_time=1e-12,
        symmetry=(1, -1, 0),
        sources=[td.ModeSource(center=(1, -1, 1), size=(2, 2, 0), direction="+", source_time=g)],
        boundary_spec=td.BoundarySpec.all_sides(boundary=td.Periodic()),
    )

    # right mode monitor (entirely in the main quadrant)
    sim = td.Simulation(
        center=(1.0, -1.0, 0.5),
        size=(2.0, 2.0, 2.0),
        grid_spec=td.GridSpec.auto(wavelength=td.C_0 / 1.0),
        run_time=1e-12,
        symmetry=(1, -1, 0),
        monitors=[
            td.ModeMonitor(
                center=(2, 0, 1), size=(2, 2, 0), name="mnt", freqs=[2], mode_spec=td.ModeSpec()
            )
        ],
        boundary_spec=td.BoundarySpec.all_sides(boundary=td.Periodic()),
    )


def test_tfsf_symmetry():
    """Test that a TFSF source cannot be set in the presence of symmetries."""
    src_time = td.GaussianPulse(freq0=1, fwidth=0.1)

    source = td.TFSF(
        size=[1, 1, 1],
        source_time=src_time,
        pol_angle=0,
        angle_theta=np.pi / 4,
        angle_phi=np.pi / 6,
        direction="+",
        injection_axis=2,
    )

    with pytest.raises(pydantic.ValidationError) as e:
        _ = td.Simulation(
            size=(2.0, 2.0, 2.0),
            grid_spec=td.GridSpec.auto(wavelength=td.C_0 / 1.0),
            run_time=1e-12,
            symmetry=(0, -1, 0),
            sources=[source],
        )


def test_tfsf_boundaries(log_capture):
    """Test that a TFSF source is allowed to cross boundaries only in particular cases."""
    src_time = td.GaussianPulse(freq0=td.C_0, fwidth=0.1)

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
                source=source, domain_size=0.5 * 1.1, axis=0, medium=None  # wrong domain size
            ),
            y=td.Boundary.bloch_from_source(
                source=source, domain_size=0.5 * 1.1, axis=1, medium=None  # wrong domain size
            ),
            z=td.Boundary.pml(),
        ),
    )
    assert_log_level(log_capture, "WARNING")

    # cannot cross any boundary in the direction of injection
    with pytest.raises(pydantic.ValidationError) as e:
        _ = td.Simulation(
            size=(2.0, 2.0, 0.5),
            grid_spec=td.GridSpec.auto(wavelength=1.0),
            run_time=1e-12,
            sources=[source],
        )

    # cannot cross any non-periodic boundary in the transverse direction
    with pytest.raises(pydantic.ValidationError) as e:
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
    src_time = td.GaussianPulse(freq0=td.C_0, fwidth=0.1)

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
    with pytest.raises(SetupError) as e:
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
    with pytest.raises(SetupError) as e:
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
    with pytest.raises(SetupError) as e:
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
                source_time=td.GaussianPulse(freq0=1, fwidth=0.1),
            )
        ],
        structures=structures,
    )
    sim.epsilon(box=td.Box(size=(size, size, size)))
    assert_log_level(log_capture, log_level)


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


@clear_tmp
def test_sim_volumetric_structures():
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
                x=td.Boundary.pml(num_layers=5),
                y=td.Boundary.pml(num_layers=5),
                z=td.Boundary.pml(num_layers=5),
            ),
            grid_spec=td.GridSpec.uniform(dl=grid_dl),
            run_time=1e-12,
        )
        if isinstance(struct.geometry, td.Box):
            assert np.isclose(sim.volumetric_structures[0].geometry.size[2], grid_dl, rtol=RTOL)
        else:
            assert np.isclose(sim.volumetric_structures[0].geometry.length_axis, grid_dl, rtol=RTOL)
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
        medium=td.Medium2D.from_medium(td.PEC, thickness=thickness),
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
            x=td.Boundary.pml(num_layers=5),
            y=td.Boundary.pml(num_layers=5),
            z=td.Boundary.pml(num_layers=5),
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
    assert np.isclose(
        sim.volumetric_structures[1].medium.xx.to_medium().conductivity,
        LARGE_NUMBER * thickness / grid_dl,
        rtol=RTOL,
    )
    # check that plotting 2d material doesn't raise an error
    sim_data = run_emulated(sim)
    sim_data.plot_field(ax=AX, field_monitor_name="field_xz", field_name="Ex", val="real")
    _ = sim.plot_eps(ax=AX, x=0, alpha=0.2)
    _ = sim.plot(ax=AX, x=0)

    # nonuniform sub/super-strate should error
    below_half = td.Structure(
        geometry=td.Box.from_bounds([-100, -td.inf, -1000], [0, td.inf, 0]),
        medium=aniso_medium,
    )

    sim = td.Simulation(
        size=(10, 10, 10),
        structures=[below_half, box],
        sources=[src],
        boundary_spec=td.BoundarySpec(
            x=td.Boundary.pml(num_layers=5),
            y=td.Boundary.pml(num_layers=5),
            z=td.Boundary.pml(num_layers=5),
        ),
        grid_spec=td.GridSpec.uniform(dl=grid_dl),
        run_time=1e-12,
    )

    with pytest.raises(SetupError):
        _ = sim.volumetric_structures

    # structure overlaying the 2D material should overwrite it like normal
    sim = td.Simulation(
        size=(10, 10, 10),
        structures=[box, below],
        sources=[src],
        boundary_spec=td.BoundarySpec(
            x=td.Boundary.pml(num_layers=5),
            y=td.Boundary.pml(num_layers=5),
            z=td.Boundary.pml(num_layers=5),
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
                x=td.Boundary.pml(num_layers=5),
                y=td.Boundary.pml(num_layers=5),
                z=td.Boundary.pml(num_layers=5),
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
