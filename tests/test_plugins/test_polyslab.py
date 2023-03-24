import pytest
import numpy as np
import matplotlib.pyplot as plt
import gdstk

import tidy3d as td

from tidy3d.plugins.polyslab import ComplexPolySlab
from ..utils import clear_tmp, assert_log_level, log_capture


def test_divide_simple_events():
    """PolySlab division for simple neighbor vertex-vertex type edge events."""

    # simple edge events occurs during erosion
    vertices_ero = ((0, 0), (1, 0), (1, 1), (0, 1), (0, 0.9), (0, 0.11))
    # simple edge events occurs during dilation
    vertices_dil = ((0, 0), (3, 0), (3, 1), (0, 1), (0, 0.9), (0.5, 0.55), (0.5, 0.45), (0, 0.1))

    vertices_list = [vertices_ero, vertices_dil]
    for vertices in vertices_list:
        for angle in [0, np.pi / 4, -np.pi / 4]:
            for reference_plane in ["top", "middle", "bottom"]:
                s = ComplexPolySlab(
                    vertices=vertices,
                    slab_bounds=(0, 1),
                    axis=2,
                    sidewall_angle=angle,
                    reference_plane=reference_plane,
                )
                subpolyslabs = s.sub_polyslabs
                geometry_group = s.geometry_group

                # for i, poly in enumerate(subpolyslabs):
                #     print(f"------------{i}-th polyglab----------")
                #     print(f"bounds = ({poly.slab_bounds[0]},  {poly.slab_bounds[1]})")
                #     print(np.array(poly.vertices))


def test_many_sub_polyslabs(log_capture):
    """warn when too many subpolyslabs are generated."""

    # generate vertices that can generate at least this number
    # of sub-polyslabs
    num_subpoly = 200
    dl_list = np.linspace(0, 0.1, num_subpoly)
    vertices = [(sum(dl_list[: i + 1]), 0) for i in range(num_subpoly)]
    vertices = vertices + [(5, 20)]

    s = ComplexPolySlab(
        vertices=vertices,
        slab_bounds=(0, 10),
        axis=2,
        sidewall_angle=np.pi / 4,
        reference_plane="bottom",
    )
    struct = td.Structure(
        geometry=s.geometry_group,
        medium=td.Medium(permittivity=2),
    )
    assert_log_level(log_capture, "WARNING")


def test_divide_simulation():
    """Test adding to a simulation."""

    vertices = ((0, 0), (1, 0), (1, 1), (0, 1), (0, 0.9), (0, 0.11))

    s = ComplexPolySlab(
        vertices=vertices,
        slab_bounds=(0, 1),
        axis=2,
        sidewall_angle=np.pi / 4,
        reference_plane="bottom",
    )
    struct = td.Structure(
        geometry=s.geometry_group,
        medium=td.Medium(permittivity=2),
    )
    sim = td.Simulation(
        run_time=1e-12,
        size=(1, 1, 1),
        grid_spec=td.GridSpec.auto(wavelength=1.0),
        structures=(struct,),
    )
    sim2 = td.Simulation(
        run_time=1e-12,
        size=(1, 1, 1),
        grid_spec=td.GridSpec.auto(wavelength=1.0),
        structures=(s.to_structure(td.Medium(permittivity=2)),),
    )


@clear_tmp
def test_gds_import():
    """construct complex polyslabs from gds (mostly from GDSII notebook)"""

    # Waveguide width
    wg_width = 0.45
    # Waveguide separation in the beginning/end
    wg_spacing_in = 8
    # Length of the coupling region
    coup_length = 10
    # Angle of the sidewall deviating from the vertical ones, positive values for the base larger than the top
    sidewall_angle = np.pi / 4
    # Reference plane where the cross section of the device is defined
    reference_plane = "bottom"
    # Length of the bend region
    bend_length = 16
    # Waveguide separation in the coupling region
    wg_spacing_coup = 0.10
    # Total device length along propagation direction
    device_length = 100

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

    # Create a gds cell to add our structures to
    coup_cell = gdstk.Cell("Coupler")

    # make substrate and add to cell
    substrate = gdstk.rectangle(
        (-device_length / 2, -wg_spacing_in / 2 - 10),
        (device_length / 2, wg_spacing_in / 2 + 10),
        layer=0,
    )

    coup_cell.add(substrate)

    # make coupler and add to the cell
    coup = make_coupler(
        device_length, wg_spacing_in, wg_width, wg_spacing_coup, coup_length, bend_length
    )

    coup_cell.add(coup)

    # Create a library for the cell and save it, just so that we can demosntrate loading
    # geometry from a gds file
    gds_path = "tests/tmp/coupler.gds"

    lib = gdstk.Library()
    lib.add(coup_cell)
    lib.write_gds(gds_path)

    lib_loaded = gdstk.read_gds(gds_path)
    coup_cell_loaded = lib_loaded.top_level()[0]

    # Define waveguide height
    wg_height = 0.3
    dilation = 0.02

    [substrate_geo] = ComplexPolySlab.from_gds(
        coup_cell_loaded,
        gds_layer=0,
        gds_dtype=0,
        axis=2,
        slab_bounds=(-430, 0),
        reference_plane=reference_plane,
    )
    arm_geo = ComplexPolySlab.from_gds(
        coup_cell_loaded,
        gds_layer=1,
        axis=2,
        slab_bounds=(0, wg_height),
        sidewall_angle=sidewall_angle,
        dilation=dilation,
    )
