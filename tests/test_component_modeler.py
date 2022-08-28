import pytest
import numpy as np
import pydantic

import gdspy

import tidy3d as td
from tidy3d.web.container import Batch
from tidy3d.plugins.smatrix.smatrix import Port, ComponentModeler
from .utils import clear_tmp, run_emulated

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


def make_coupler():
    # wavelength / frequency
    lambda0 = 1.550  # all length scales in microns
    freq0 = td.constants.C_0 / lambda0
    fwidth = freq0 / 10

    # Spatial grid specification
    grid_spec = td.GridSpec.auto(min_steps_per_wvl=14, wavelength=lambda0)

    # Permittivity of waveguide and substrate
    wg_n = 3.48
    sub_n = 1.45
    mat_wg = td.Medium(permittivity=wg_n**2)
    mat_sub = td.Medium(permittivity=sub_n**2)

    def bend_pts(bend_length, width, npts=10):
        """Set of points describing a tanh bend from (0, 0) to (length, width)"""
        x = np.linspace(0, bend_length, npts)
        y = width * (1 + np.tanh(6 * (x / bend_length - 0.5))) / 2
        return np.stack((x, y), axis=1)

    def arm_pts(length, width, coup_length, bend_length, npts_bend=30):
        """Set of points defining one arm of an integrated coupler"""
        ### Make the right half of the coupler arm first
        # Make bend and offset by coup_length/2
        bend = bend_pts(bend_length, width, npts_bend)
        bend[:, 0] += coup_length / 2
        # Add starting point as (0, 0)
        right_half = np.concatenate(([[0, 0]], bend))
        # Add an extra point to make sure waveguide is straight past the bend
        right_half = np.concatenate((right_half, [[right_half[-1, 0] + 0.1, width]]))
        # Add end point as (length/2, width)
        right_half = np.concatenate((right_half, [[length / 2, width]]))

        # Make the left half by reflecting and omitting the (0, 0) point
        left_half = np.copy(right_half)[1:, :]
        left_half[:, 0] = -left_half[::-1, 0]
        left_half[:, 1] = left_half[::-1, 1]

        return np.concatenate((left_half, right_half), axis=0)

    def make_coupler(
        length, wg_spacing_in, wg_width, wg_spacing_coup, coup_length, bend_length, npts_bend=30
    ):
        """Make an integrated coupler using the gdspy FlexPath object."""

        # Compute one arm of the coupler
        arm_width = (wg_spacing_in - wg_width - wg_spacing_coup) / 2
        arm = arm_pts(length, arm_width, coup_length, bend_length, npts_bend)
        # Reflect and offset bottom arm
        coup_bot = np.copy(arm)
        coup_bot[:, 1] = -coup_bot[::-1, 1] - wg_width / 2 - wg_spacing_coup / 2
        # Offset top arm
        coup_top = np.copy(arm)
        coup_top[:, 1] += wg_width / 2 + wg_spacing_coup / 2

        # Create waveguides as GDS paths
        path_bot = gdspy.FlexPath(coup_bot, wg_width, layer=1, datatype=0)
        path_top = gdspy.FlexPath(coup_top, wg_width, layer=1, datatype=1)

        return [path_bot, path_top]

    gdspy.current_library = gdspy.GdsLibrary()
    lib = gdspy.GdsLibrary()

    # Geometry must be placed in GDS cells to import into Tidy3D
    coup_cell = lib.new_cell("Coupler")

    substrate = gdspy.Rectangle(
        (-device_length / 2, -wg_spacing_in / 2 - 10),
        (device_length / 2, wg_spacing_in / 2 + 10),
        layer=0,
    )
    coup_cell.add(substrate)

    # Add the coupler to a gdspy cell
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
        center=[0, 0, wg_height / 2], size=[td.inf, td.inf, 0], freqs=[freq0], name="field"
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
        simulation=sim, ports=ports, freq=sim.monitors[0].freqs[0], batch=batch_empty, **kwargs
    )


def run_component_modeler(monkeypatch, modeler: ComponentModeler):

    values = dict(
        simulation=modeler.simulation,
        ports=modeler.ports,
        freq=modeler.freq,
        run_only=modeler.run_only,
        element_mappings=modeler.element_mappings,
    )
    sim_dict = modeler.make_sim_dict(values)
    batch_data = {task_name: run_emulated(sim) for task_name, sim in sim_dict.items()}
    monkeypatch.setattr(ComponentModeler, "_run_sims", lambda self, path_dir: batch_data)
    s_matrix = modeler.run()
    return s_matrix


def test_make_component_modeler():
    modeler = make_component_modeler()


def test_run_component_modeler(monkeypatch):
    modeler = make_component_modeler()
    s_matrix = run_component_modeler(monkeypatch, modeler)

    for port_in in modeler.ports:
        for mode_index_in in range(port_in.mode_spec.num_modes):
            index_in = (port_in.name, mode_index_in)

            for port_out in modeler.ports:
                for mode_index_out in range(port_out.mode_spec.num_modes):
                    index_out = (port_out.name, mode_index_out)
                    assert index_in in s_matrix, "source index not present in S matrix"
                    assert index_out in s_matrix[index_in], "monitor index not present in S matrix"


def test_component_modeler_run_only(monkeypatch):
    sim = make_coupler()
    ports = make_ports()
    ONLY_SOURCE = ("right_bot", 0)
    run_only = [ONLY_SOURCE]
    modeler = make_component_modeler(run_only=run_only)
    s_matrix = run_component_modeler(monkeypatch, modeler)

    for port_in in ports:
        for mode_index_in in range(port_in.mode_spec.num_modes):
            index_in = (port_in.name, mode_index_in)

            for port_out in ports:
                for mode_index_out in range(port_out.mode_spec.num_modes):
                    index_out = (port_out.name, mode_index_out)

                    # make sure only allowed elements are in S matrix
                    if index_in == ONLY_SOURCE:
                        assert index_in in s_matrix, "run_only source index not present in S matrix"
                        assert (
                            index_out in s_matrix[index_in]
                        ), "run_only out data not present in S matrix"
                    else:
                        assert (
                            index_in not in s_matrix
                        ), "source index excluded from run_only not present in S matrix"


def _test_mappings(element_mappings, s_matrix):
    """Makes sure the mappings are reflected in a given S matrix."""
    for (i, j), (k, l), mult_by in element_mappings:
        assert s_matrix[k][l] == mult_by * s_matrix[i][j], "mapping not applied correctly."


def test_run_component_modeler_mappings(monkeypatch):

    element_mappings = (
        ((("left_top", 0), ("right_top", 0)), (("left_bot", 0), ("right_bot", 0)), -1j),
        ((("left_top", 0), ("right_bot", 0)), (("left_bot", 0), ("right_top", 0)), +1),
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
                mapping = ((row_index, row_index), (EXCLUDE_INDEX, row_index), +1)
                element_mappings.append(mapping)

    # add the self-self coupling element to complete row
    mapping = ((("right_bot", 1), ("right_bot", 1)), (EXCLUDE_INDEX, EXCLUDE_INDEX), +1)
    element_mappings.append(mapping)
    modeler = make_component_modeler(element_mappings=element_mappings)

    run_sim_indices = modeler.matrix_indices_run_sim(
        ports=modeler.ports, run_only=modeler.run_only, element_mappings=modeler.element_mappings
    )
    assert EXCLUDE_INDEX not in run_sim_indices, "mapping didnt exclude row properly"

    s_matrix = run_component_modeler(monkeypatch, modeler)
    _test_mappings(element_mappings, s_matrix)
