import pytest
import numpy as np
import pydantic

import gdstk

import tidy3d as td
from tidy3d.web.container import Batch
from tidy3d.plugins.smatrix.smatrix import Port, ComponentModeler
from tidy3d.exceptions import SetupError, Tidy3dKeyError
from ..utils import clear_tmp, run_emulated

# Waveguide height
wg_height = 0.22
# Waveguide width
wg_width = 1.0
# Waveguide separation in the beginning/end
wg_spacing_in = 8

# length of coupling region (um)
coup_length = 6.0
# spacing between waveguides in coupling region (um)
wg_spacing_coup = 0.05
# Total device length along propagation direction
device_length = 100
# Length of the bend region
bend_length = 16
# Straight waveguide sections on each side
straight_wg_length = 4
# space between waveguide and PML
pml_spacing = 2

PATH_DIR = "tests/tmp"


def make_coupler():
    # wavelength / frequency
    lambda0 = 1.550  # all length scales in microns
    freq0 = td.constants.C_0 / lambda0
    freqs = [freq0, freq0 * 1.1]
    fwidth = freq0 / 10

    # Spatial grid specification
    grid_spec = td.GridSpec.auto(min_steps_per_wvl=14, wavelength=3 * lambda0)

    # Permittivity of waveguide and substrate
    wg_n = 3.48
    sub_n = 1.45
    mat_wg = td.Medium(permittivity=wg_n**2)
    mat_sub = td.Medium(permittivity=sub_n**2)

    def tanh_interp(max_arg):
        """Interpolator for tanh with adjustable extension"""
        scale = 1 / np.tanh(max_arg)
        return lambda u: 0.5 * (1 + scale * np.tanh(max_arg * (u * 2 - 1)))

    def make_coupler(
        length, wg_spacing_in, wg_width, wg_spacing_coup, coup_length, bend_length, npts_bend=30
    ):
        """Make an integrated coupler using the gdstk RobustPath object."""
        # bend interpolator
        interp = tanh_interp(3)
        delta = wg_width + wg_spacing_coup - wg_spacing_in
        offset = lambda u: wg_spacing_in + interp(u) * delta

        coup = gdstk.RobustPath(
            (-0.5 * length, 0),
            (wg_width, wg_width),
            wg_spacing_in,
            simple_path=True,
            layer=1,
            datatype=[0, 1],
        )
        coup.segment((-0.5 * coup_length - bend_length, 0))
        coup.segment(
            (-0.5 * coup_length, 0), offset=[lambda u: -0.5 * offset(u), lambda u: 0.5 * offset(u)]
        )
        coup.segment((0.5 * coup_length, 0))
        coup.segment(
            (0.5 * coup_length + bend_length, 0),
            offset=[lambda u: -0.5 * offset(1 - u), lambda u: 0.5 * offset(1 - u)],
        )
        coup.segment((0.5 * length, 0))
        return coup

    # Geometry must be placed in GDS cells to import into Tidy3D
    coup_cell = gdstk.Cell("Coupler")

    substrate = gdstk.rectangle(
        (-device_length / 2, -wg_spacing_in / 2 - 10),
        (device_length / 2, wg_spacing_in / 2 + 10),
        layer=0,
    )
    coup_cell.add(substrate)

    # Add the coupler to a gdstk cell
    gds_coup = make_coupler(
        device_length, wg_spacing_in, wg_width, wg_spacing_coup, coup_length, bend_length
    )
    coup_cell.add(gds_coup)

    # Substrate
    [oxide_geo] = td.PolySlab.from_gds(
        gds_cell=coup_cell, gds_layer=0, gds_dtype=0, slab_bounds=(-10, 0), axis=2
    )

    oxide = td.Structure(geometry=oxide_geo, medium=mat_sub)

    # Waveguides (import all datatypes if gds_dtype not specified)
    coupler1_geo, coupler2_geo = td.PolySlab.from_gds(
        gds_cell=coup_cell, gds_layer=1, slab_bounds=(0, wg_height), axis=2
    )

    coupler1 = td.Structure(geometry=coupler1_geo, medium=mat_wg)

    coupler2 = td.Structure(geometry=coupler2_geo, medium=mat_wg)

    # Simulation size along propagation direction
    sim_length = 2 * straight_wg_length + 2 * bend_length + coup_length

    # Spacing between waveguides and PML
    sim_size = [sim_length, wg_spacing_in + wg_width + 2 * pml_spacing, wg_height + 2 * pml_spacing]

    # in-plane field monitor (optional, increases required data storage)
    domain_monitor = td.FieldMonitor(
        center=[0, 0, wg_height / 2], size=[td.inf, td.inf, 0], freqs=freqs, name="field"
    )

    # initialize the simulation
    return td.Simulation(
        size=sim_size,
        grid_spec=grid_spec,
        structures=[oxide, coupler1, coupler2],
        sources=[],
        monitors=[domain_monitor],
        run_time=20 / fwidth,
        boundary_spec=td.BoundarySpec.all_sides(boundary=td.PML()),
    )


def make_ports():

    sim = make_coupler()
    # source
    src_pos = sim.size[0] / 2 - straight_wg_length / 2

    port_right_top = Port(
        center=[src_pos, wg_spacing_in / 2, wg_height / 2],
        size=[0, 4, 2],
        mode_spec=td.ModeSpec(num_modes=2),
        direction="-",
        name="right_top",
    )

    port_right_bot = Port(
        center=[src_pos, -wg_spacing_in / 2, wg_height / 2],
        size=[0, 4, 2],
        mode_spec=td.ModeSpec(num_modes=2),
        direction="-",
        name="right_bot",
    )

    port_left_top = Port(
        center=[-src_pos, wg_spacing_in / 2, wg_height / 2],
        size=[0, 4, 2],
        mode_spec=td.ModeSpec(num_modes=2),
        direction="+",
        name="left_top",
    )

    port_left_bot = Port(
        center=[-src_pos, -wg_spacing_in / 2, wg_height / 2],
        size=[0, 4, 2],
        mode_spec=td.ModeSpec(num_modes=2),
        direction="+",
        name="left_bot",
    )

    return [port_right_top, port_right_bot, port_left_top, port_left_bot]


def make_component_modeler(**kwargs):
    """Tests S matrix loading."""

    sim = make_coupler()
    ports = make_ports()
    batch_empty = Batch(simulations={}, folder_name="None")
    return ComponentModeler(
        simulation=sim, ports=ports, freqs=sim.monitors[0].freqs, path_dir=PATH_DIR, **kwargs
    )


def run_component_modeler(monkeypatch, modeler: ComponentModeler):

    values = dict(
        simulation=modeler.simulation,
        ports=modeler.ports,
        freqs=modeler.freqs,
        run_only=modeler.run_only,
        element_mappings=modeler.element_mappings,
    )
    sim_dict = modeler.sim_dict
    batch_data = {task_name: run_emulated(sim) for task_name, sim in sim_dict.items()}
    monkeypatch.setattr(ComponentModeler, "_run_sims", lambda self, path_dir: batch_data)
    s_matrix = modeler.run(path_dir=PATH_DIR)
    return s_matrix


def test_validate_no_sources():
    modeler = make_component_modeler()
    source = td.PointDipole(
        source_time=td.GaussianPulse(freq0=2e14, fwidth=1e14), polarization="Ex"
    )
    sim_w_source = modeler.simulation.copy(update=dict(sources=(source,)))
    with pytest.raises(pydantic.ValidationError):
        modeler_w_source = modeler.copy(update=dict(simulation=sim_w_source))


def test_element_mappings_none():
    modeler = make_component_modeler()
    modeler = modeler.updated_copy(ports=[], element_mappings=())
    modeler.matrix_indices_run_sim


def test_no_port():
    modeler = make_component_modeler()
    ports = modeler.ports
    with pytest.raises(Tidy3dKeyError):
        modeler.get_port_by_name(port_name="NOT_A_PORT")


def test_ports_too_close_boundary():
    modeler = make_component_modeler()
    grid_boundaries = modeler.simulation.grid.boundaries.to_list[0]
    way_outside = grid_boundaries[0] - 1000
    xmin = grid_boundaries[1]
    xmax = grid_boundaries[-2]
    for edge_val, port_dir in zip((way_outside, xmin, xmax), ("+", "+", "-")):
        port_at_edge = modeler.ports[0].copy()
        port_center_at_edge = list(port_at_edge.center)
        port_center_at_edge[0] = edge_val
        port_at_edge = port_at_edge.copy(
            update=dict(center=port_center_at_edge, direction=port_dir)
        )
        with pytest.raises(SetupError):
            modeler._shift_value_signed(port=port_at_edge)


def test_validate_batch_supplied():

    sim = make_coupler()
    modeler = ComponentModeler(
        simulation=sim, ports=[], freqs=sim.monitors[0].freqs, path_dir=PATH_DIR
    )


def test_plot_sim():
    modeler = make_component_modeler()
    modeler.plot_sim(z=0)


def test_make_component_modeler():
    modeler = make_component_modeler()


def test_run(monkeypatch):
    modeler = make_component_modeler()
    monkeypatch.setattr(ComponentModeler, "run", lambda self, path_dir: None)
    modeler.run(path_dir=PATH_DIR)


def test_run_component_modeler(monkeypatch):
    modeler = make_component_modeler()
    s_matrix = run_component_modeler(monkeypatch, modeler)

    for port_in in modeler.ports:
        for mode_index_in in range(port_in.mode_spec.num_modes):

            for port_out in modeler.ports:
                for mode_index_out in range(port_out.mode_spec.num_modes):

                    coords_in = dict(port_in=port_in.name, mode_index_in=mode_index_in)
                    coords_out = dict(port_out=port_out.name, mode_index_out=mode_index_out)

                    try:
                        assert np.all(
                            s_matrix.loc[coords_in] != 0
                        ), "source index not present in S matrix"
                    except:
                        import pdb

                        pdb.set_trace()
                    assert np.all(
                        s_matrix.loc[coords_in].loc[coords_out] != 0
                    ), "monitor index not present in S matrix"


def test_component_modeler_run_only(monkeypatch):
    sim = make_coupler()
    ports = make_ports()
    ONLY_SOURCE = (port_run_only, mode_index_run_only) = ("right_bot", 0)
    run_only = [ONLY_SOURCE]
    modeler = make_component_modeler(run_only=run_only)
    s_matrix = run_component_modeler(monkeypatch, modeler)

    coords_in_run_only = dict(port_in=port_run_only, mode_index_in=mode_index_run_only)

    # make sure the run only mappings are non-zero
    assert np.all(s_matrix.loc[coords_in_run_only] != 0)

    # make sure if we zero out the run_only mappings, everythging is zero
    s_matrix.loc[coords_in_run_only] = 0
    assert np.all(s_matrix.values == 0.0)


def _test_mappings(element_mappings, s_matrix):
    """Makes sure the mappings are reflected in a given S matrix."""
    for (i, j), (k, l), mult_by in element_mappings:

        (port_out_from, mode_index_out_from) = i
        (port_in_from, mode_index_in_from) = j
        (port_out_to, mode_index_out_to) = k
        (port_in_to, mode_index_in_to) = l

        coords_from = dict(
            port_in=port_in_from,
            port_out=port_out_from,
            mode_index_in=mode_index_in_from,
            mode_index_out=mode_index_out_from,
        )

        coords_to = dict(
            port_in=port_in_to,
            port_out=port_out_to,
            mode_index_in=mode_index_in_to,
            mode_index_out=mode_index_out_to,
        )

        assert np.all(
            s_matrix.loc[coords_to].values == mult_by * s_matrix.loc[coords_from].values
        ), "mapping not applied correctly."


def test_run_component_modeler_mappings(monkeypatch):

    element_mappings = (
        ((("left_bot", 0), ("right_bot", 0)), (("left_top", 0), ("right_top", 0)), -1j),
        ((("left_bot", 0), ("right_top", 0)), (("left_top", 0), ("right_bot", 0)), +1),
    )
    modeler = make_component_modeler(element_mappings=element_mappings)
    s_matrix = run_component_modeler(monkeypatch, modeler)
    _test_mappings(element_mappings, s_matrix)


def test_mapping_exclusion(monkeypatch):
    """Make sure that source indices are skipped if totally covered by element mapping."""

    sim = make_coupler()
    ports = make_ports()

    EXCLUDE_INDEX = ("right_bot", 0)
    element_mappings = []

    # add a mapping to each element in the row of EXCLUDE_INDEX
    for port in ports:
        for mode_index in range(port.mode_spec.num_modes):
            row_index = (port.name, mode_index)
            if row_index != EXCLUDE_INDEX:
                mapping = ((row_index, row_index), (row_index, EXCLUDE_INDEX), +1)
                element_mappings.append(mapping)

    # add the self-self coupling element to complete row
    mapping = ((("right_bot", 1), ("right_bot", 1)), (EXCLUDE_INDEX, EXCLUDE_INDEX), +1)
    element_mappings.append(mapping)

    modeler = make_component_modeler(element_mappings=element_mappings)

    run_sim_indices = modeler.matrix_indices_run_sim
    assert EXCLUDE_INDEX not in run_sim_indices, "mapping didnt exclude row properly"

    s_matrix = run_component_modeler(monkeypatch, modeler)
    _test_mappings(element_mappings, s_matrix)
