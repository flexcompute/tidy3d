from __future__ import annotations

import warnings

import gdstk
import matplotlib.pyplot as plt
import numpy as np
import pytest
from pydantic import ValidationError

import tidy3d as td
from tidy3d import SimulationDataMap
from tidy3d.exceptions import SetupError, Tidy3dKeyError
from tidy3d.plugins.smatrix import (
    AstigmaticGaussianPort,
    GaussianPort,
    ModalComponentModeler,
    ModalComponentModelerData,
    Port,
)
from tidy3d.web.api.container import Batch

from ...utils import AssertLogStr, run_emulated

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

        def offset(u):
            return wg_spacing_in + interp(u) * delta

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
        center=(0, 0, wg_height / 2), size=(td.inf, td.inf, 0), freqs=freqs, name="field"
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

    # Gaussian ports on top and bottom
    port_z_bot = AstigmaticGaussianPort(
        center=[0, 0, wg_height + 0.1],
        size=(10, 10, 0),
        direction="-",
        name="z_top",
        angle_theta=0.0,
        angle_phi=0.0,
        pol_angle=0.0,
    )

    port_z_top = GaussianPort(
        center=[0, 0, -0.1],
        size=(10, 10, 0),
        direction="+",
        name="z_bot",
        angle_theta=0.0,
        angle_phi=0.0,
        pol_angle=0.0,
    )

    return [port_right_top, port_right_bot, port_left_top, port_left_bot, port_z_bot, port_z_top]


def make_component_modeler(**kwargs):
    """Tests S matrix loading."""

    sim = make_coupler()
    ports = make_ports()
    _ = Batch(simulations={}, folder_name="None")
    return ModalComponentModeler(simulation=sim, ports=ports, freqs=sim.monitors[0].freqs, **kwargs)


def run_component_modeler(monkeypatch, modeler: ModalComponentModeler) -> ModalComponentModelerData:
    sim_dict = modeler.sim_dict
    batch_data = {task_name: run_emulated(sim) for task_name, sim in sim_dict.items()}
    port_data = SimulationDataMap(
        keys=tuple(batch_data.keys()),
        values=tuple(batch_data.values()),
    )
    modeler_data = ModalComponentModelerData(modeler=modeler, data=port_data)
    return modeler_data


def get_port_data_array(monkeypatch, modeler: ModalComponentModeler):
    modeler_data = run_component_modeler(monkeypatch=monkeypatch, modeler=modeler)
    return modeler_data.smatrix().data


def test_legacy_port_mode_spec_sort_spec_load():
    """Legacy ``sort_key=None`` in port mode specs should load through model validation."""
    sim = td.Simulation(
        size=(1, 1, 1),
        run_time=1e-12,
        structures=[],
        sources=[],
        monitors=[],
        grid_spec=td.GridSpec(
            grid_x=td.UniformGrid(dl=0.1),
            grid_y=td.UniformGrid(dl=0.1),
            grid_z=td.UniformGrid(dl=0.1),
        ),
        boundary_spec=td.BoundarySpec(
            x=td.Boundary.periodic(),
            y=td.Boundary.periodic(),
            z=td.Boundary.periodic(),
        ),
    )
    legacy_sort_spec = {"sort_key": None, "sort_reference": 1.5, "sort_order": "ascending"}

    port = Port(
        center=(0, 0, 0),
        size=(1, 1, 0),
        mode_spec=td.ModeSpec(),
        direction="+",
        name="P0",
    )
    port_dict = port.model_dump(mode="json")
    port_dict["mode_spec"]["sort_spec"] = legacy_sort_spec
    loaded_port = Port.model_validate(port_dict)
    assert loaded_port.mode_spec.sort_spec.sort_key == "n_eff"
    assert loaded_port.mode_spec.sort_spec.sort_reference is None
    assert loaded_port.mode_spec.sort_spec.sort_order == "descending"

    modeler = ModalComponentModeler(simulation=sim, ports=[port], freqs=[2e14])
    modeler_dict = modeler.model_dump(mode="json")
    modeler_dict["ports"][0]["mode_spec"]["sort_spec"] = legacy_sort_spec
    loaded_modeler = ModalComponentModeler.model_validate(modeler_dict)
    assert loaded_modeler.ports[0].mode_spec.sort_spec.sort_key == "n_eff"
    assert loaded_modeler.ports[0].mode_spec.sort_spec.sort_reference is None
    assert loaded_modeler.ports[0].mode_spec.sort_spec.sort_order == "descending"


def test_validate_no_sources():
    modeler = make_component_modeler()
    source = td.PointDipole(
        source_time=td.GaussianPulse(freq0=2e14, fwidth=1e14), polarization="Ex"
    )
    sim_w_source = modeler.simulation.copy(update={"sources": (source,)})
    with pytest.raises(ValidationError):
        _ = modeler.copy(update={"simulation": sim_w_source})


def test_element_mappings_none():
    modeler = make_component_modeler()
    modeler = modeler.updated_copy(ports=(), element_mappings=())
    _ = modeler.matrix_indices_run_sim


def test_no_port():
    modeler = make_component_modeler()
    _ = modeler.ports
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
            update={"center": port_center_at_edge, "direction": port_dir}
        )
        with pytest.raises(SetupError):
            modeler._shift_value_signed(port=port_at_edge, simulation=modeler.simulation)


def test_validate_batch_supplied(tmp_path):
    sim = make_coupler()
    _ = ModalComponentModeler(
        simulation=sim,
        ports=[],
        freqs=sim.monitors[0].freqs,
    )


def test_plot_sim():
    modeler = make_component_modeler()
    modeler.plot_sim(z=0)
    plt.close()


def test_plot_sim_eps():
    modeler = make_component_modeler()
    modeler.plot_sim_eps(z=0)
    plt.close()


def test_make_component_modeler():
    _ = make_component_modeler()


def test_run(monkeypatch):
    modeler = make_component_modeler()
    _ = run_component_modeler(monkeypatch, modeler=modeler)


def test_run_component_modeler(monkeypatch):
    modeler = make_component_modeler()
    modeler_data = run_component_modeler(monkeypatch, modeler=modeler)
    s_matrix = modeler_data.smatrix()

    for port_in in modeler.ports:
        for mode_index_in in range(port_in.num_modes):
            for port_out in modeler.ports:
                for mode_index_out in range(port_out.num_modes):
                    coords_in = {"port_in": port_in.name, "mode_index_in": mode_index_in}
                    coords_out = {"port_out": port_out.name, "mode_index_out": mode_index_out}
                    assert np.all(s_matrix.sel(**coords_in).sel(mode_index_out=0) != 0), (
                        "source index not present in S matrix"
                    )
                    assert np.all(s_matrix.sel(**coords_in).sel(**coords_out) != 0), (
                        "monitor index not present in S matrix"
                    )


def test_component_modeler_run_only(monkeypatch):
    _ = make_coupler()
    _ = make_ports()
    ONLY_SOURCE = (port_run_only, mode_index_run_only) = ("right_bot", 0)
    run_only = (ONLY_SOURCE,)
    modeler = make_component_modeler(run_only=run_only)
    modeler_data = run_component_modeler(monkeypatch, modeler=modeler)
    s_matrix = modeler_data.smatrix()

    coords_in_run_only = {"port_in": port_run_only, "mode_index_in": mode_index_run_only}

    # make sure the run only mappings are non-zero
    assert np.all(s_matrix.sel(**coords_in_run_only).sel(mode_index_out=0) != 0)

    # make sure if we zero out the run_only mappings, everythging is zero
    s_matrix.loc[coords_in_run_only] = 0
    assert np.all(s_matrix.values == 0.0)

    # make sure lists are correctly converted into tuples
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"(?s)Pydantic serializer warnings:.*field_name='run_only'.*",
            category=UserWarning,
        )
        run_only = [list(ONLY_SOURCE)]
        modeler = modeler.updated_copy(run_only=run_only)
        assert ONLY_SOURCE in modeler.matrix_indices_run_sim


def _test_mappings(element_mappings, s_matrix):
    """Makes sure the mappings are reflected in a given S matrix."""
    for (i, j), (k, L), mult_by in element_mappings:
        (port_out_from, mode_index_out_from) = i
        (port_in_from, mode_index_in_from) = j
        (port_out_to, mode_index_out_to) = k
        (port_in_to, mode_index_in_to) = L

        coords_from = {
            "port_in": port_in_from,
            "port_out": port_out_from,
            "mode_index_in": mode_index_in_from,
            "mode_index_out": mode_index_out_from,
        }

        coords_to = {
            "port_in": port_in_to,
            "port_out": port_out_to,
            "mode_index_in": mode_index_in_to,
            "mode_index_out": mode_index_out_to,
        }

        assert np.all(
            s_matrix.sel(**coords_to).values == mult_by * s_matrix.sel(**coords_from).values
        ), "mapping not applied correctly."


def test_run_component_modeler_mappings(monkeypatch):
    element_mappings = (
        ((("left_bot", 0), ("right_bot", 0)), (("left_top", 0), ("right_top", 0)), -1j),
        ((("left_bot", 0), ("right_top", 0)), (("left_top", 0), ("right_bot", 0)), +1),
    )
    modeler = make_component_modeler(element_mappings=element_mappings)
    modeler_data = run_component_modeler(monkeypatch, modeler=modeler)
    s_matrix = modeler_data.smatrix()
    _test_mappings(element_mappings, s_matrix)


def test_mapping_exclusion(monkeypatch):
    """Make sure that source indices are skipped if totally covered by element mapping."""

    _ = make_coupler()
    ports = make_ports()

    EXCLUDE_INDEX = ("right_bot", 0)
    element_mappings = []

    # add a mapping to each element in the row of EXCLUDE_INDEX
    for port in ports:
        for mode_index in range(port.num_modes):
            row_index = (port.name, mode_index)
            if row_index != EXCLUDE_INDEX:
                mapping = ((row_index, row_index), (row_index, EXCLUDE_INDEX), +1)
                element_mappings.append(mapping)

    # add the self-self coupling element to complete row
    mapping = ((("right_bot", 1), ("right_bot", 1)), (EXCLUDE_INDEX, EXCLUDE_INDEX), +1)
    element_mappings.append(mapping)

    modeler = make_component_modeler(element_mappings=tuple(element_mappings))
    modeler_data = run_component_modeler(monkeypatch, modeler=modeler)
    s_matrix = modeler_data.smatrix()

    run_sim_indices = modeler.matrix_indices_run_sim
    assert EXCLUDE_INDEX not in run_sim_indices, "mapping didnt exclude row properly"

    _test_mappings(tuple(element_mappings), s_matrix)


def test_mapping_with_run_only():
    """Make sure that the Modeler is correctly validated when both run_only and
    element_mappings are provided."""
    ports = make_ports()

    EXCLUDE_INDEX = ["right_bot", 0]
    element_mappings = []
    run_only = []
    # add a mapping to each element in the row of EXCLUDE_INDEX
    for port in ports:
        for mode_index in range(port.num_modes):
            # Test that providing a list is properly handled
            row_index = [port.name, mode_index]
            run_only.append(row_index)
            if row_index != EXCLUDE_INDEX:
                mapping = ((row_index, row_index), (row_index, EXCLUDE_INDEX), +1)
                element_mappings.append(mapping)

    # add the self-self coupling element to complete row
    mapping = ((("right_bot", 1), ("right_bot", 1)), (EXCLUDE_INDEX, EXCLUDE_INDEX), +1)
    element_mappings.append(mapping)

    # Will pass, since run_only covers all source indices in element_mapping
    _ = make_component_modeler(element_mappings=element_mappings, run_only=run_only)

    run_only.remove(EXCLUDE_INDEX)
    with pytest.raises(ValidationError):
        _ = make_component_modeler(element_mappings=element_mappings, run_only=run_only)


def test_get_task_name():
    """Test the 'get_task_name' method."""
    port = Port(
        center=[0, 0, 0],
        size=[0, 4, 2],
        mode_spec=td.ModeSpec(num_modes=2),
        direction="-",
        name="port1",
    )

    # Test with mode_index specified
    task_name = ModalComponentModeler.get_task_name(port=port, mode_index=1)
    assert task_name == "port1@1"

    # Test with mode_index unspecified
    task_name = ModalComponentModeler.get_task_name(port=port)
    assert task_name == "port1@0"


def test_custom_source_time(monkeypatch):
    """Test that custom_source_time is properly used in the component modeler."""
    modeler = make_component_modeler()
    freqs = modeler.freqs
    custom_source = td.GaussianPulse.from_frequency_range(fmin=min(freqs), fmax=max(freqs))

    # Create modeler with custom source time
    with AssertLogStr(
        log_level_expected="WARNING", excludes_str="Custom source time does not cover all"
    ):
        modeler = make_component_modeler(custom_source_time=custom_source)

    # Run the modeler and verify it works with custom source time
    modeler_data = run_component_modeler(monkeypatch, modeler=modeler)

    # Verify that simulations were created and run successfully
    s_matrix = modeler_data.smatrix()
    assert s_matrix is not None

    # Verify that the simulations in sim_dict use the custom source time
    for sim in modeler.sim_dict.values():
        # Each simulation should have sources with the custom source time
        assert len(sim.sources) > 0
        for source in sim.sources:
            assert source.source_time.freq0 == custom_source.freq0
            assert source.source_time.fwidth == custom_source.fwidth

    with AssertLogStr(
        log_level_expected="WARNING", contains_str="Custom source time does not cover all"
    ):
        custom_source = td.GaussianPulse(freq0=td.C_0, fwidth=1e12)
        modeler = make_component_modeler(custom_source_time=custom_source)


def test_validate_run_only_uniqueness_modal():
    """Test that run_only validator rejects duplicate entries for ModalComponentModeler."""
    modeler = make_component_modeler()

    # Get valid matrix indices (port_name, mode_index)
    port0_idx = (modeler.ports[0].name, 0)
    port1_idx = (modeler.ports[1].name, 0)

    # Test with duplicate entries - should raise ValidationError
    with pytest.raises(ValidationError, match="duplicate entries"):
        modeler.updated_copy(run_only=(port0_idx, port0_idx, port1_idx))


def test_validate_run_only_membership_modal():
    """Test that run_only validator rejects invalid indices for ModalComponentModeler."""
    modeler = make_component_modeler()

    # Test with invalid port name
    with pytest.raises(ValidationError, match="not present in"):
        modeler.updated_copy(run_only=(("invalid_port", 0),))

    # Test with invalid mode index
    port0_name = modeler.ports[0].name
    invalid_mode = modeler.ports[0].mode_spec.num_modes + 1
    with pytest.raises(ValidationError, match="not present in"):
        modeler.updated_copy(run_only=((port0_name, invalid_mode),))
