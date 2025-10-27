"""Tests for ModePlaneAnalyzer and terminal specification features."""

from __future__ import annotations

import numpy as np
import pydantic.v1 as pd
import pytest
from shapely import LineString

import tidy3d as td
from tidy3d.components.microwave.path_integrals.mode_plane_analyzer import ModePlaneAnalyzer
from tidy3d.components.mode.mode_solver import ModeSolver
from tidy3d.exceptions import SetupError

from .utils import make_coupled_microstrip_sim, make_minimal_mode_plane_analyzer, make_mw_sim

# Constants
mm = 1e3


def make_mode_plane_analyzer(sim, monitor):
    """Create ModePlaneAnalyzer from simulation and monitor."""
    return ModePlaneAnalyzer(
        center=monitor.center,
        size=monitor.size,
        field_data_colocated=monitor.colocate,
        structures=sim.structures,
        grid=sim.grid,
        symmetry=sim.symmetry,
        sim_box=sim.bounding_box,
    )


def test_mode_plane_analyzer_errors():
    """Check that the ModePlaneAnalyzer reports errors properly."""

    mode_plane_analyzer = make_minimal_mode_plane_analyzer()

    # First some quick sanity checks with the helper
    test_path = td.Box(center=(0, 0, 0), size=(0, 0.9, 0.1))
    test_shapely = [LineString([(-1, 0), (1, 0)])]
    assert mode_plane_analyzer._check_box_intersects_with_conductors(test_shapely, test_path)

    test_path = td.Box(center=(0, 0, 0), size=(0, 2.1, 0.1))
    test_shapely = [LineString([(-1, 0), (1, 0)])]
    assert not mode_plane_analyzer._check_box_intersects_with_conductors(test_shapely, test_path)

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
    mode_plane_analyzer = make_mode_plane_analyzer(sim, mode_monitor)

    with pytest.raises(SetupError):
        _ = mode_plane_analyzer.conductor_bounding_boxes

    # Error when no conductors intersecting mode plane
    mode_plane_analyzer = mode_plane_analyzer.updated_copy(size=(0, 0.1, 0.1), center=(0, 0, 1.5))
    with pytest.raises(SetupError):
        _ = mode_plane_analyzer.conductor_shapes


@pytest.mark.parametrize("colocate", [False, True])
@pytest.mark.parametrize("tline_type", ["microstrip", "cpw", "coax"])
def test_mode_plane_analyzer_canonical_shapes(colocate, tline_type):
    """Test canonical transmission line types to make sure the correct path integrals are generated."""
    sim = make_mw_sim(False, colocate, tline_type)
    mode_monitor = sim.monitors[0]
    mode_plane_analyzer = make_mode_plane_analyzer(sim, mode_monitor)
    bounding_boxes = mode_plane_analyzer.conductor_bounding_boxes
    geos = mode_plane_analyzer.conductor_shapes

    if tline_type == "coax":
        assert mode_plane_analyzer.num_conductors == 2
        assert len(bounding_boxes) == 2
        for path_spec in bounding_boxes:
            assert np.all(np.isclose(path_spec.center, (0, 0, 5 * mm)))
    else:
        assert mode_plane_analyzer.num_conductors == 1
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
    mode_plane_analyzer = make_mode_plane_analyzer(sim, mode_monitor)
    bounding_boxes = mode_plane_analyzer.conductor_bounding_boxes
    geos = mode_plane_analyzer.conductor_shapes

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
    impedance_specs = (td.AutoImpedanceSpec(),)
    mode_spec = td.MicrowaveModeSpec(
        num_modes=1,
        target_neff=1.8,
        impedance_specs=impedance_specs,
    )

    metal_box = td.Structure(
        geometry=td.Box.from_bounds(
            rmin=(-1 * dl, -1 * dl, -0.5 * mm), rmax=(2 * dl, 2 * dl, 0.5 * mm)
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
        colocate=False,
        freqs=[freq0],
    )
    mode_solver_boundaries = mms._solver_grid.boundaries.to_list
    mode_plane_analyzer = ModePlaneAnalyzer(
        center=mode_center,
        size=mode_size,
        field_data_colocated=False,
        structures=sim.structures,
        grid=sim.grid,
        symmetry=sim.symmetry,
        sim_box=sim.bounding_box,
    )
    mode_plane_limits = mode_plane_analyzer.mode_limits

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


def test_validate_conductor_voltage_configurations():
    """Test validation of conductor voltage configurations in ModePlaneAnalyzer.

    Tests common user errors:
    1. Valid configuration
    2. Conductor assigned to both positive and negative terminals
    3. Polarity-reversed duplicates
    """
    analyzer = make_minimal_mode_plane_analyzer()

    # Valid: multiple terminal configurations with different conductors
    valid_config = [
        ({0}, {1}),  # Terminal 1: conductor 0 (+), conductor 1 (-)
        ({2}, {3}),  # Terminal 2: conductor 2 (+), conductor 3 (-)
    ]
    analyzer._validate_conductor_voltage_configurations(valid_config)  # Should not raise

    # ERROR: Conductor in both positive and negative sets
    error_both_polarities = [
        ({0, 1}, {1, 2})  # Conductor 1 is in BOTH positive and negative!
    ]
    with pytest.raises(SetupError, match="cannot be assigned to both a positive and negative"):
        analyzer._validate_conductor_voltage_configurations(error_both_polarities)

    # ERROR: Polarity-reversed duplicates (most common user error!)
    # ({0}, {1}) and ({1}, {0}) represent the same terminal with reversed polarity
    error_polarity_reversed = [
        ({0}, {1}),  # Terminal 1: conductor 0 (+), conductor 1 (-)
        ({1}, {0}),  # Terminal 2: conductor 1 (+), conductor 0 (-) <- SAME TERMINAL!
    ]
    with pytest.raises(SetupError, match="Duplicate voltage configuration"):
        analyzer._validate_conductor_voltage_configurations(error_polarity_reversed)


def test_validate_empty_terminal_sets():
    """Handle empty plus/minus sets."""
    analyzer = make_minimal_mode_plane_analyzer()

    # Valid: One set can be empty (for common mode or single-ended)
    valid_empty_minus = [({0, 1}, set())]  # Common mode: multiple plus, no minus
    analyzer._validate_conductor_voltage_configurations(valid_empty_minus)  # Should not raise

    # This is also valid: no plus, only minus (though unusual)
    valid_empty_plus = [(set(), {0, 1})]
    analyzer._validate_conductor_voltage_configurations(valid_empty_plus)  # Should not raise


def test_validate_exact_duplicates():
    """Not just polarity-reversed, but exact copies."""
    analyzer = make_minimal_mode_plane_analyzer()

    # ERROR: Exact duplicate configurations
    exact_duplicates = [
        ({0}, {1}),  # First terminal spec
        ({0}, {1}),  # Exact duplicate
    ]
    with pytest.raises(SetupError, match="Duplicate voltage configuration"):
        analyzer._validate_conductor_voltage_configurations(exact_duplicates)


def test_validate_complex_multi_conductor():
    """4+ conductor scenarios."""
    analyzer = make_minimal_mode_plane_analyzer()

    # Valid: Complex multi-conductor setup
    valid_complex = [
        ({0, 1}, {2}),  # Terminal 1: conductors 0,1 (+), conductor 2 (-)
        ({3}, {4, 5}),  # Terminal 2: conductor 3 (+), conductors 4,5 (-)
    ]
    analyzer._validate_conductor_voltage_configurations(valid_complex)  # Should not raise


def test_terminal_spec_valid_inputs():
    """Test TerminalSpec validation with valid inputs for plus_terminals and minus_terminals."""
    # Test Case 1.1: Coordinate2D tuples (single coordinate)
    spec = td.TerminalSpec(plus_terminals=((1.0, 0.5),), minus_terminals=((-1.0, 0.5),))
    assert spec is not None
    assert len(spec.plus_terminals) == 1
    assert spec.plus_terminals[0] == (1.0, 0.5)

    # Test Case 1.2: Multiple Coordinate2D tuples
    spec = td.TerminalSpec(plus_terminals=((1.0, 0.5), (2.0, 0.5)), minus_terminals=())
    assert len(spec.plus_terminals) == 2

    # Test Case 1.3: String identifiers
    spec = td.TerminalSpec(
        plus_terminals=("conductor_1", "conductor_2"), minus_terminals=("ground",)
    )
    assert spec.plus_terminals[0] == "conductor_1"
    assert spec.minus_terminals[0] == "ground"

    # Test Case 1.4: ArrayFloat2D with 2 points (linestring)
    spec = td.TerminalSpec(plus_terminals=(np.array([[0.0, 0.0], [1.0, 1.0]]),), minus_terminals=())
    assert isinstance(spec.plus_terminals[0], np.ndarray)
    assert spec.plus_terminals[0].shape == (2, 2)

    # Test Case 1.5: ArrayFloat2D with 3+ points (polygon)
    spec = td.TerminalSpec(
        plus_terminals=(np.array([[0, 0], [1, 0], [1, 1], [0, 1]]),), minus_terminals=()
    )
    assert isinstance(spec.plus_terminals[0], np.ndarray)
    assert spec.plus_terminals[0].shape == (4, 2)

    # Test Case 1.6: Lists converted to numpy by ArrayFloat2D
    spec = td.TerminalSpec(plus_terminals=([[0.0, 0.0], [1.0, 1.0]],), minus_terminals=())
    # List should be converted to numpy array
    assert isinstance(spec.plus_terminals[0], np.ndarray)
    assert spec.plus_terminals[0].shape == (2, 2)

    # Test Case 1.7: Mixed types in same spec
    spec = td.TerminalSpec(
        plus_terminals=(
            (1.0, 0.5),  # Coordinate2D
            "conductor_2",  # str
            np.array([[0, 0], [1, 1]]),  # ArrayFloat2D
        ),
        minus_terminals=(),
    )
    assert isinstance(spec.plus_terminals[0], tuple)
    assert isinstance(spec.plus_terminals[1], str)
    assert isinstance(spec.plus_terminals[2], np.ndarray)

    # Test Case 1.8: Empty minus_terminals (valid for common mode)
    spec = td.TerminalSpec(plus_terminals=((1.0, 0.5),), minus_terminals=())
    assert len(spec.minus_terminals) == 0

    # Test Case 1.9: Exactly 3 points (minimum for polygon - triangle)
    spec = td.TerminalSpec(
        plus_terminals=(np.array([[0, 0], [1, 0], [0.5, 1]]),), minus_terminals=()
    )
    assert spec.plus_terminals[0].shape == (3, 2)

    # Test Case 1.10: Exactly 2 points
    spec = td.TerminalSpec(plus_terminals=(np.array([[0.0, 0.0], [1.0, 0.0]]),), minus_terminals=())
    assert spec.plus_terminals[0].shape == (2, 2)

    # Test Case 1.11: 1 point but using ArrayLike
    spec = td.TerminalSpec(plus_terminals=(np.array([[1.0, 0.0]]),), minus_terminals=())
    assert spec.plus_terminals[0].shape == (1, 2)


def test_terminal_spec_invalid_inputs():
    """Test TerminalSpec validation rejects invalid inputs."""
    # Test Case 2.1: Invalid array shape (not Nx2)
    with pytest.raises(pd.ValidationError):
        td.TerminalSpec(
            plus_terminals=(np.array([[0, 0, 0], [1, 1, 1]]),),  # 3 columns instead of 2
            minus_terminals=(),
        )

    # Test Case 2.2: Invalid polygon (self-intersecting bowtie)
    with pytest.raises(pd.ValidationError, match="valid polygon"):
        td.TerminalSpec(
            plus_terminals=(
                np.array([[0, 0], [1, 1], [1, 0], [0, 1]]),
            ),  # Figure-8 self-intersection
            minus_terminals=(),
        )

    # Test Case 2.3: 1D array instead of 2D
    with pytest.raises(pd.ValidationError):
        td.TerminalSpec(
            plus_terminals=(np.array([0, 1, 2]),),  # 1D array
            minus_terminals=(),
        )

    # Test Case 2.4: Wrong coordinate tuple length (3 elements instead of 2)
    with pytest.raises(pd.ValidationError):
        td.TerminalSpec(
            plus_terminals=((1.0, 0.5, 2.0),),  # 3 elements
            minus_terminals=(),
        )


def test_terminal_spec_microstrip():
    """TerminalSpec → conversion → finding → validation (microstrip)."""
    sim = make_mw_sim(transmission_line_type="microstrip")
    mode_monitor = sim.monitors[0]
    analyzer = make_mode_plane_analyzer(sim, mode_monitor)

    # Define terminal spec: signal line as positive terminal
    terminal_specs = [td.TerminalSpec(plus_terminals=((0.0, 1.1 * mm),), minus_terminals=())]
    voltage_sets = analyzer._identify_conductor_voltage_sets(tuple(terminal_specs))

    # Validate configuration
    analyzer._validate_conductor_voltage_configurations(voltage_sets)  # Should not raise

    # Verify results
    assert len(voltage_sets) == 1
    assert len(voltage_sets[0][0]) == 1  # One plus conductor (signal)
    assert len(voltage_sets[0][1]) == 0  # No minus conductor (single-ended)


def test_terminal_spec_cpw():
    """Test with CPW (3 conductors, but 2 touching PEC boundary)."""
    sim = make_mw_sim(transmission_line_type="cpw")
    mode_monitor = sim.monitors[0]
    analyzer = make_mode_plane_analyzer(sim, mode_monitor)

    # CPW: signal line positive, ground planes can be reference
    terminal_specs = [td.TerminalSpec(plus_terminals=((0.0, 1.1 * mm),), minus_terminals=())]

    voltage_sets = analyzer._identify_conductor_voltage_sets(tuple(terminal_specs))
    analyzer._validate_conductor_voltage_configurations(voltage_sets)

    assert len(voltage_sets) == 1


@pytest.mark.parametrize("terminal_spec_type", ["Point", "Line", "Polygon"])
def test_terminal_spec_coupled_microstrip(terminal_spec_type):
    """Test with coupled microstrip (2 signal lines)."""
    sim = make_coupled_microstrip_sim()
    mode_monitor = sim.monitors[0]
    analyzer = make_mode_plane_analyzer(sim, mode_monitor)

    strip_left = sim.structures[1]
    strip_right = sim.structures[2]
    # Define terminal specifications
    # Differential mode: strip_left is +, strip_right is -
    # Common mode: both strips are +
    if terminal_spec_type == "Point":
        conductor_1_id = np.asarray(
            [[strip_left.geometry.center[1], strip_left.geometry.center[2]]]
        )
        conductor_2_id = np.asarray(
            [[strip_right.geometry.center[1], strip_right.geometry.center[2]]]
        )
    elif terminal_spec_type == "Line":
        conductor_1_id = [strip_left.geometry.bounds[0][1:3], strip_left.geometry.bounds[1][1:3]]
        conductor_2_id = [strip_right.geometry.bounds[0][1:3], strip_right.geometry.bounds[1][1:3]]
    elif terminal_spec_type == "Polygon":
        conductor_1_id = [
            strip_left.geometry.bounds[0][1:3],
            strip_left.geometry.bounds[1][1:3],
            sim.center[1:3],
        ]
        conductor_2_id = [
            strip_right.geometry.bounds[0][1:3],
            strip_right.geometry.bounds[1][1:3],
            sim.center[1:3],
        ]

    diff_spec = td.TerminalSpec(plus_terminals=(conductor_1_id,), minus_terminals=(conductor_2_id,))
    common_spec = td.TerminalSpec(
        plus_terminals=(conductor_1_id, conductor_2_id), minus_terminals=()
    )
    terminal_specs = [diff_spec, common_spec]

    voltage_sets = analyzer._identify_conductor_voltage_sets(tuple(terminal_specs))
    analyzer._validate_conductor_voltage_configurations(voltage_sets)

    assert len(voltage_sets) == 2
    assert voltage_sets[0][0] == {0}  # One plus conductor
    assert voltage_sets[0][1] == {1}  # One minus conductor
    assert voltage_sets[1][0] == {0, 1}  # Both plus conductors
    assert voltage_sets[1][1] == set()  # No conductors


def test_terminal_spec_with_structure_names():
    """Use structure names instead of coordinates."""
    sim = make_coupled_microstrip_sim()
    mode_monitor = sim.monitors[0]
    analyzer = make_mode_plane_analyzer(sim, mode_monitor)

    # Use structure names
    terminal_specs = [td.TerminalSpec(plus_terminals=("signal_1",), minus_terminals=("signal_2",))]

    voltage_sets = analyzer._identify_conductor_voltage_sets(tuple(terminal_specs))
    analyzer._validate_conductor_voltage_configurations(voltage_sets)

    assert len(voltage_sets) == 1
    assert len(voltage_sets[0][0]) == 1  # signal_1
    assert len(voltage_sets[0][1]) == 1  # signal_2


def test_terminal_spec_error_messages_helpful():
    """Verify helpful error messages for common mistakes."""
    sim = make_coupled_microstrip_sim()
    mode_monitor = sim.monitors[0]
    analyzer = make_mode_plane_analyzer(sim, mode_monitor)

    # Error 1: Structure name not found
    terminal_specs = [td.TerminalSpec(plus_terminals=("nonexistent",), minus_terminals=())]
    with pytest.raises(SetupError, match="No structure found with name 'nonexistent'"):
        analyzer._convert_terminal_specifications_to_candidate_geometry(
            sim.structures, terminal_specs
        )

    # Error 2: Terminal doesn't intersect any conductor
    terminal_specs = [td.TerminalSpec(plus_terminals=((100 * mm, 100 * mm),), minus_terminals=())]
    with pytest.raises(SetupError, match="No conductor found intersecting"):
        analyzer._identify_conductor_voltage_sets(tuple(terminal_specs))

    # Error 3: Terminal spec accidentally touches 2 conductors
    strip_left = sim.structures[1]
    strip_right = sim.structures[2]
    line_touching_both = [strip_left.geometry.bounds[1][1:3], strip_right.geometry.bounds[0][1:3]]
    terminal_specs = [td.TerminalSpec(plus_terminals=(line_touching_both,), minus_terminals=())]
    with pytest.raises(SetupError, match="Multiple conductors"):
        analyzer._identify_conductor_voltage_sets(tuple(terminal_specs))

    # Error 4: Conductor in both polarities
    voltage_sets = [({0}, {0})]  # Same conductor in both plus and minus
    with pytest.raises(SetupError, match="cannot be assigned to both"):
        analyzer._validate_conductor_voltage_configurations(voltage_sets)
