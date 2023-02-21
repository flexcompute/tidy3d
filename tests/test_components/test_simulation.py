"""Tests the simulation and its validators."""
import pytest
import pydantic
import matplotlib.pylab as plt

import numpy as np
import tidy3d as td
from tidy3d.log import SetupError, ValidationError, Tidy3dKeyError
from tidy3d.components import simulation
from tidy3d.components.simulation import MAX_NUM_MEDIUMS
from ..utils import assert_log_level, SIM_FULL

SIM = td.Simulation(size=(1, 1, 1), run_time=1e-12, grid_spec=td.GridSpec(wavelength=1.0))

_, AX = plt.subplots()


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


# TODO: remove for 2.0
def test_deprecation_defaults(caplog):
    """Make sure deprecation warnings thrown if defaults used."""
    s = td.Simulation(
        size=(1, 1, 1), run_time=1e-12, grid_spec=td.GridSpec.uniform(dl=0.1), boundary_spec=None
    )
    assert_log_level(caplog, "warning")


def test_sim_bounds():
    """make sure bounds are working correctly"""

    # make sure all things are shifted to this central location
    CENTER_SHIFT = (-1.0, 1.0, 100.0)

    def place_box(center_offset):

        shifted_center = tuple(c + s for (c, s) in zip(center_offset, CENTER_SHIFT))

        sim = td.Simulation(
            size=(1, 1, 1),
            center=CENTER_SHIFT,
            grid_spec=td.GridSpec(wavelength=1.0),
            run_time=1e-12,
            structures=[
                td.Structure(
                    geometry=td.Box(size=(1, 1, 1), center=shifted_center), medium=td.Medium()
                )
            ],
            boundary_spec=td.BoundarySpec.all_sides(boundary=td.Periodic()),
        )

    # create all permutations of squares being shifted 1, -1, or zero in all three directions
    bin_strings = [list(format(i, "03b")) for i in range(8)]
    bin_ints = [[int(b) for b in bin_string] for bin_string in bin_strings]
    bin_ints = np.array(bin_ints)
    bin_signs = 2 * (bin_ints - 0.5)

    # test all cases where box is shifted +/- 1 in x,y,z and still intersects
    for amp in bin_ints:
        for sign in bin_signs:
            center = amp * sign
            place_box(tuple(center))

    # test all cases where box is shifted +/- 2 in x,y,z and no longer intersects
    for amp in bin_ints:
        for sign in bin_signs:
            center = 2 * amp * sign
            if np.sum(center) < 1e-12:
                continue
            with pytest.raises(SetupError) as e_info:
                place_box(tuple(center))


def test_sim_size():

    mesh1d = td.UniformGrid(dl=1e-6)
    grid_spec = td.GridSpec(grid_x=mesh1d, grid_y=mesh1d, grid_z=mesh1d)

    with pytest.raises(SetupError):
        s = td.Simulation(
            size=(1, 1, 1),
            grid_spec=grid_spec,
            run_time=1e-12,
            boundary_spec=td.BoundarySpec.all_sides(boundary=td.Periodic()),
        )
        s._validate_size()

    with pytest.raises(SetupError):
        s = td.Simulation(
            size=(1, 1, 1),
            run_time=1e-7,
            boundary_spec=td.BoundarySpec.all_sides(boundary=td.Periodic()),
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


@pytest.mark.parametrize("freq, log_level", [(1.5, "warning"), (2.5, None), (3.5, "warning")])
def test_monitor_medium_frequency_range(caplog, freq, log_level):
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
    assert_log_level(caplog, log_level)


@pytest.mark.parametrize("fwidth, log_level", [(0.1, "warning"), (2, None)])
def test_monitor_simulation_frequency_range(caplog, fwidth, log_level):
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
    assert_log_level(caplog, log_level)


def test_validate_bloch_with_symmetry():
    with pytest.raises(SetupError):
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


def test_validate_plane_wave_boundaries(caplog):
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
    with pytest.raises(SetupError):
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
    assert_log_level(caplog, "warning")

    # angled incidence plane wave with wrong Bloch vector should warn
    td.Simulation(
        size=(1, 1, 1),
        run_time=1e-12,
        sources=[src2],
        boundary_spec=bspec4,
    )
    assert_log_level(caplog, "warning")


def test_validate_zero_dim_boundaries(caplog):

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
    assert_log_level(caplog, "warning")

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
    assert SIM._source_homogeneous(val=None, values=SIM.dict()) is None


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


def test_validate_mnt_size(monkeypatch):
    monkeypatch.setattr(simulation, "MAX_MONITOR_DATA_SIZE_BYTES", 1)
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


def test_discretize_non_intersect(caplog):
    SIM.discretize(box=td.Box(center=(-20, -20, -20), size=(1, 1, 1)))
    assert_log_level(caplog, "error")


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


def test_warn_sim_background_medium_freq_range(caplog):
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
    assert_log_level(caplog, "warning")


@pytest.mark.parametrize("grid_size,log_level", [(0.001, None), (3, "warning")])
def test_large_grid_size(caplog, grid_size, log_level):
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

    assert_log_level(caplog, log_level)


@pytest.mark.parametrize("box_size,log_level", [(0.001, None), (9.9, "warning"), (20, None)])
def test_sim_structure_gap(caplog, box_size, log_level):
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
    assert_log_level(caplog, log_level)


def test_sim_plane_wave_error():
    """ "Make sure we error if plane wave is not intersecting homogeneous region of simulation."""

    medium_bg = td.Medium(permittivity=2)
    medium_air = td.Medium(permittivity=1)

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
    with pytest.raises(SetupError):
        _ = td.Simulation(
            size=(1, 1, 1),
            medium=medium_bg,
            structures=[box_transparent, box],
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
        with pytest.raises(SetupError):
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


def test_proj_monitor_distance(caplog):
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
    assert_log_level(caplog, "warning")

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

    with pytest.raises(SetupError):
        _ = td.Simulation(
            size=(2, 2, 2),
            structures=[box_cond],
            sources=[src],
            run_time=1e-12,
            monitors=[monitor],
            boundary_spec=td.BoundarySpec.all_sides(boundary=td.Periodic()),
        )

    with pytest.raises(SetupError):
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
        ((0.1, 0.1, 0.1), None),
        ((1, 0.1, 0.1), "warning"),
        ((0.1, 1, 0.1), "warning"),
        ((0.1, 0.1, 1), "warning"),
    ],
)
def test_sim_structure_extent(caplog, box_size, log_level):
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

    assert_log_level(caplog, log_level)


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

    with pytest.raises(SetupError):
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

    with pytest.raises(SetupError) as e:
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

    with pytest.raises(SetupError) as e:
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

    with pytest.raises(SetupError) as e:
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
    with pytest.raises(SetupError) as e_info:
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
    with pytest.raises(SetupError) as e_info:
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


@pytest.mark.parametrize(
    "size, num_struct, log_level", [(1, 1, None), (50, 1, "warning"), (1, 11000, "warning")]
)
def test_warn_large_epsilon(caplog, size, num_struct, log_level):
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
    assert_log_level(caplog, log_level)
