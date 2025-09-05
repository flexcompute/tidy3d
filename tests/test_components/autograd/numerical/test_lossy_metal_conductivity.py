# test autograd and compares to numerically computed finite difference gradients
from __future__ import annotations

import os
import sys

import autograd as ag
import matplotlib.pylab as plt
import numpy as np
import pytest

import tidy3d as td
import tidy3d.plugins.smatrix as smatrix
import tidy3d.web as web
from tidy3d.components.autograd import get_static

td.config.logging_level = "ERROR"

PLOT_FD_ADJ_COMPARISON = False
SAVE_FD_ADJ_DATA = True
LOCAL_GRADIENT = True
VERBOSE = False
SHOW_PRINT_STATEMENTS = True
NUMERICAL_RESULTS_DATA_DIR = "./lossy_metal_conductivity/"

LOSSY_METAL_CONDUCTIVITY_SHAPE = 50
LOSSY_METAL_CONDUCTIVITY_MEAN = 12

GRADIENT_OVERLAP_THRESHOLD_DEGREES = 20


# -----------------------------
# Shared helpers for both tests. Much of the setup code comes from https://www.flexcompute.com/tidy3d/examples/notebooks/CoupledLineBandpassFilter/
# -----------------------------
def build_common_config():
    # Frequency range of interest is 2-10 GHz
    freq0 = 6e9
    freq_stopband = 3e9
    freqs = np.linspace(2, 10, 11) * 1e9
    lda0 = td.C_0 / freq0

    mm = 1e3
    inf_eff = 1000 * mm

    # Geometry parameters common to all designs
    h_sub = 1.56 * mm
    h_trace = 0.035 * mm

    eps_sub = 4.3
    sub_medium = td.Medium(permittivity=eps_sub)

    return {
        "freq0": freq0,
        "freq_stopband": freq_stopband,
        "freqs": freqs,
        "lda0": lda0,
        "mm": mm,
        "inf_eff": inf_eff,
        "h_sub": h_sub,
        "h_trace": h_trace,
        "eps_sub": eps_sub,
        "sub_medium": sub_medium,
    }


def get_coupled_line_geometry(lengths, widths, gaps):
    # Coupled lines are generated from upper left to lower right centered at the origin
    total_length = np.sum(lengths)
    # Compute the total width
    total_width = 0
    prev_width = 0
    # Coupled line segments are aligned differently depending on their relative widths
    for width, gap in zip(widths, gaps):
        total_width += width + gap
        if width > prev_width:
            total_width += width - prev_width
        prev_width = width

    # Compute the starting positions of each segment the left bound and the top bound
    xstarts = []
    ystarts = []

    prev_xstart = -total_length / 2
    prev_ystart = total_width / 2
    prev_length = 0
    prev_width = 0
    prev_gap = 0
    for length, width, gap in zip(lengths, widths, gaps):
        # Compute x position of the segment (left bound)
        xstarts.append(prev_xstart + prev_length)

        # Compute y position of the segment (top bound)
        ystart = prev_ystart - prev_width - prev_gap
        if width < prev_width:
            ystart += width - prev_width
        ystarts.append(ystart)

        prev_xstart = xstarts[-1]
        prev_ystart = ystarts[-1]
        prev_length = length
        prev_width = width
        prev_gap = gap

    return total_length, total_width, xstarts, ystarts


def generate_coupled_lines(
    lengths, widths, gaps, xstarts, ystarts, medium, h_trace, use_polyslab=False
):
    coupled_lines = []
    # Each segment is composed of two microstrips
    for length, width, gap, xstart, ystart in zip(lengths, widths, gaps, xstarts, ystarts):
        if use_polyslab:
            # Create rectangles as PolySlab with same bounds
            verts_top = [
                (xstart, ystart - width),
                (xstart + length, ystart - width),
                (xstart + length, ystart),
                (xstart, ystart),
            ]
            verts_bot = [
                (xstart, ystart - 2 * width - gap),
                (xstart + length, ystart - 2 * width - gap),
                (xstart + length, ystart - width - gap),
                (xstart, ystart - width - gap),
            ]
            geom_top = td.PolySlab(vertices=verts_top, slab_bounds=(0, h_trace), axis=2)
            geom_bot = td.PolySlab(vertices=verts_bot, slab_bounds=(0, h_trace), axis=2)
        else:
            geom_top = td.Box.from_bounds(
                rmin=(xstart, ystart - width, 0),
                rmax=(xstart + length, ystart, h_trace),
            )
            geom_bot = td.Box.from_bounds(
                rmin=(xstart, ystart - 2 * width - gap, 0),
                rmax=(xstart + length, ystart - width - gap, h_trace),
            )
        coupled_lines.append(td.Structure(geometry=geom_top, medium=medium))
        coupled_lines.append(td.Structure(geometry=geom_bot, medium=medium))
    return coupled_lines


def build_mesh_overrides(total_L, total_W, widths, h_sub, h_trace):
    return [
        # The first mesh override ensures the small height of the strips is accurately modeled.
        td.MeshOverrideStructure(
            geometry=td.Box(
                center=[0, 0, h_trace / 2],
                size=[1.1 * (total_L), 1.1 * (total_W), h_trace],
            ),
            dl=[(total_L) / 200, (np.min(widths)) / 20, h_trace],
        ),
        # The second mesh override refined the grid within the substrate.
        td.MeshOverrideStructure(
            geometry=td.Box(
                center=[0, 0, -h_sub / 2],
                size=[1.1 * (total_L), 1.1 * (total_W), h_sub],
            ),
            dl=[(total_L) / 200, (np.min(widths)) / 20, h_sub / 20],
        ),
    ]


def build_ports(xstarts, ystarts, lengths, widths, gaps, h_sub):
    # Compute port xy locations using provided geometry
    port_left = (get_static(xstarts[0]), ystarts[0] - (widths[0] / 2))
    port_right = (get_static(xstarts[-1]) + lengths[-1], ystarts[-1] - gaps[-1] - 1.5 * widths[-1])
    reference_impedance = 50
    port_1_td = smatrix.LumpedPort(
        center=(port_left[0], port_left[1], -h_sub / 2),
        size=(0, widths[0], h_sub),
        voltage_axis=2,
        name="lumped_port_1",
        impedance=reference_impedance,
    )
    port_2_td = smatrix.LumpedPort(
        center=(port_right[0], port_right[1], -h_sub / 2),
        size=(0, widths[-1], h_sub),
        voltage_axis=2,
        name="lumped_port_2",
        impedance=reference_impedance,
    )
    return port_1_td, port_2_td


def run_modeler_and_objective(sim, ports, freqs, f_sample=4.4e9):
    modeler = smatrix.TerminalComponentModeler(
        simulation=sim,
        ports=ports,
        freqs=freqs,
    )
    s_matrix_data = web.run(
        modeler,
        task_name="lossy_metal_numerical_test",
        local_gradient=LOCAL_GRADIENT,
        verbose=VERBOSE,
    )
    s_matrix = s_matrix_data.smatrix()

    # Measure s21
    get_data = s_matrix.data.sel(port_in="lumped_port_1", port_out="lumped_port_2")
    get_data = np.abs(get_data.sel(f=f_sample, method="nearest").data) ** 2
    return get_data


def finite_difference(obj_fn, x0, h):
    grads = []
    for idx in range(len(x0)):
        x_up = x0.copy()
        x_up[idx] += h
        f_up = obj_fn(x_up)
        x_down = x0.copy()
        x_down[idx] -= h
        f_down = obj_fn(x_down)
        grads.append((f_up - f_down) / (2 * h))
    return np.array(grads)


def cosine_similarity(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    if a.size == 0 or b.size == 0:
        return np.nan
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return np.nan
    return float(np.dot(a, b) / denom)


if PLOT_FD_ADJ_COMPARISON:
    pytestmark = pytest.mark.usefixtures("mpl_config_interactive")
else:
    pytestmark = pytest.mark.usefixtures("mpl_config_noninteractive")

if SHOW_PRINT_STATEMENTS:
    sys.stdout = sys.stderr


@pytest.mark.numerical
@pytest.mark.parametrize("use_lossy_medium", [False, True])
@pytest.mark.parametrize("use_polyslab", [False, True])
@pytest.mark.parametrize(
    "dir_name", [NUMERICAL_RESULTS_DATA_DIR] if SAVE_FD_ADJ_DATA else [None], indirect=["dir_name"]
)
def test_finite_difference_lossy_metal_shape(rng, use_lossy_medium, use_polyslab, create_directory):
    """Test shape gradients for `Box` and `PolySlab` when using `LossyMetalMedium` and `PECMedium`."""

    td.config.logging_level = "ERROR"

    cfg = build_common_config()
    freq0 = cfg["freq0"]
    freq_stopband = cfg["freq_stopband"]
    freqs = cfg["freqs"]
    lda0 = cfg["lda0"]
    mm = cfg["mm"]
    inf_eff = cfg["inf_eff"]
    h_sub = cfg["h_sub"]
    h_trace = cfg["h_trace"]
    sub_medium = cfg["sub_medium"]

    # Select medium based on parameter
    metal_medium = (
        td.LossyMetalMedium(
            conductivity=LOSSY_METAL_CONDUCTIVITY_SHAPE,
            frequency_range=(np.min(freqs), np.max(freqs)),
            thickness=h_trace,
        )
        if use_lossy_medium
        else td.PECMedium()
    )

    # Microstrip parameters taken from [1]
    strips_W = [1.4 * mm, 2.4 * mm, 2.4 * mm, 1.4 * mm]
    strips_L = [6 * mm, 6 * mm, 6 * mm, 6 * mm]
    strips_G = [0.2 * mm, 0.2 * mm, 0.2 * mm, 0.2 * mm]

    # Numpy arrays of the microstrip dimensions
    lengths = np.array(strips_L)
    widths = np.array(strips_W)
    gaps = np.array(strips_G)

    num_boxes = len(strips_W)
    all_parameters = strips_L + strips_W + strips_G

    (total_L_static, total_W_static, xstarts_static, ystarts_static) = get_coupled_line_geometry(
        lengths, widths, gaps
    )
    lengths_static = lengths.copy()
    widths_static = widths.copy()
    gaps_static = gaps.copy()

    def objective_fn(params):
        # Create the structures and geometry of the coupled line bandpass filter
        (total_L, total_W, xstarts, ystarts) = get_coupled_line_geometry(lengths, widths, gaps)

        coupled_lines = generate_coupled_lines(
            lengths,
            widths,
            gaps,
            xstarts,
            ystarts,
            metal_medium,
            h_trace,
            use_polyslab=use_polyslab,
        )
        # Create the substrate block
        substrate = td.Structure(
            geometry=td.Box.from_bounds(
                rmin=(-inf_eff, -inf_eff, -inf_eff), rmax=(inf_eff, inf_eff, 0)
            ),
            medium=sub_medium,
        )

        modified_lines = coupled_lines.copy()
        for idx, perturb_idx in enumerate(length_idx_to_modify):
            geom = coupled_lines[perturb_idx].geometry
            direction = modify_direction[idx]

            if not use_polyslab:
                center = geom.center
                size = geom.size
                dim_to_change = 0
                center_dim = center[dim_to_change]
                size_dim = size[dim_to_change]
                diff = params[idx] - size_dim
                size_dim_new = size_dim + diff
                center_dim_new = center_dim + 0.5 * direction * diff
                new_geom = geom.updated_copy(
                    center=[center_dim_new, center[1], center[2]],
                    size=[size_dim_new, size[1], size[2]],
                )
                modified_lines[perturb_idx] = coupled_lines[perturb_idx].updated_copy(
                    geometry=new_geom
                )
            else:
                # derive rectangle from vertices and adjust along x
                verts = list(geom.vertices)
                xs = [v[0] for v in verts]
                ys = [v[1] for v in verts]
                min_x, max_x = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)
                current_length = max_x - min_x
                diff = params[idx] - current_length
                new_length = current_length + diff
                cx = 0.5 * (min_x + max_x)
                new_cx = cx + 0.5 * direction * diff
                new_min_x = new_cx - new_length / 2.0
                new_max_x = new_cx + new_length / 2.0
                new_verts = [
                    (new_min_x, y_min),
                    (new_max_x, y_min),
                    (new_max_x, y_max),
                    (new_min_x, y_max),
                ]
                new_geom = td.PolySlab(
                    vertices=new_verts, slab_bounds=geom.slab_bounds, axis=geom.axis
                )
                modified_lines[perturb_idx] = coupled_lines[perturb_idx].updated_copy(
                    geometry=new_geom
                )

        # Define the simulation domain size with some extra padding based on central wavelength
        Lx = (total_L_static) + lda0 / 5
        Ly = (total_W_static) + lda0 / 5
        Lz = h_sub + lda0 / 8

        # Definition of mesh overrides using shared helper
        mesh_overrides = build_mesh_overrides(
            total_L_static, total_W_static, widths_static, h_sub, h_trace
        )

        # Field monitor to view the electromagnetic fields along the propagation direction.
        field_monitor = td.FieldMonitor(
            center=(0, 0, -h_sub / 2),
            size=(td.inf, td.inf, 0),
            freqs=[freq_stopband, freq0],
            name="field",
        )
        # Boundary conditions are perfectly matched layers, except for the minus z boundary.
        boundary_spec = td.BoundarySpec(
            x=td.Boundary.pml(),
            y=td.Boundary.pml(),
            z=td.Boundary(minus=td.PECBoundary(), plus=td.PML()),
        )
        # The base tidy3D Simulation ready to be used by the TerminalComponentModeler
        sim = td.Simulation(
            center=(0, 0, Lz / 2 - h_sub),
            size=(Lx, Ly, Lz),
            grid_spec=td.GridSpec.auto(
                min_steps_per_wvl=40.0,
                wavelength=lda0,
                dl_min=h_trace / 5,
                override_structures=mesh_overrides,
            ),
            structures=[substrate, *modified_lines],
            sources=[],
            monitors=[field_monitor],
            run_time=10e-9,
            boundary_spec=boundary_spec,
        )

        # Build ports using shared helper
        port_1_td, port_2_td = build_ports(
            xstarts_static, ystarts_static, lengths_static, widths_static, gaps_static, h_sub
        )

        # Use shared helper to run modeler and get objective
        return run_modeler_and_objective(sim, [port_1_td, port_2_td], freqs, f_sample=4.4e9)

    (total_L, total_W, xstarts, ystarts) = get_coupled_line_geometry(lengths, widths, gaps)
    coupled_lines = generate_coupled_lines(
        lengths, widths, gaps, xstarts, ystarts, metal_medium, h_trace, use_polyslab=use_polyslab
    )

    # we only modify some of the box boundaries that have dielectric on the other side to avoid complication with
    # a boundary that moves above or below another box boundary. `modify_direction` tells which side the boundary
    # is on so that we can modify the center position correctly.
    length_idx_to_modify = [0, 2, 3, 4, 5, 7]
    modify_direction = [1, 1, -1, 1, -1, -1]

    init_params = []
    for idx in length_idx_to_modify:
        geom = coupled_lines[idx].geometry
        if use_polyslab:
            verts = list(geom.vertices)
            xs = [v[0] for v in verts]
            init_params.append(max(xs) - min(xs))
        else:
            init_params.append(geom.size[0])

    static_parameters = all_parameters.copy()

    val_and_grad = ag.value_and_grad(objective_fn)

    f, g = val_and_grad(init_params)

    h = 0.15 * mm
    fd_length = finite_difference(objective_fn, init_params, h)

    if PLOT_FD_ADJ_COMPARISON:
        plt.subplot(1, 2, 1)
        plt.plot(g, color="g", linewidth=2.0)
        plt.plot(fd_length, color="b", linewidth=1.5, linestyle="--")
        plt.legend(["Finite difference", "Adjoint"])
        plt.title("Unnormalized Gradient Comparison (shape)")
        plt.xlabel("Sample number")
        plt.ylabel("Gradient value")
        plt.subplot(1, 2, 2)
        plt.plot(g / np.linalg.norm(g), color="g", linewidth=2.0)
        plt.plot(fd_length / np.linalg.norm(fd_length), color="b", linewidth=1.5, linestyle="--")
        plt.title("Normalized Gradient Comparison (shape)")
        plt.xlabel("Sample number")
        plt.ylabel("Gradient value")
        plt.legend(["Finite difference", "Adjoint"])
        plt.show()

    angle_deg = np.arccos(cosine_similarity(g, fd_length)) * 180.0 / np.pi
    if SAVE_FD_ADJ_DATA and NUMERICAL_RESULTS_DATA_DIR:
        medium_tag = "lossy" if use_lossy_medium else "pec"
        geometry_tag = "polyslab" if use_polyslab else "box"
        np.savez(
            os.path.join(
                NUMERICAL_RESULTS_DATA_DIR, f"shape_{medium_tag}_{geometry_tag}_results.npz"
            ),
            adjoint=np.array(g, dtype=float),
            finite_difference=np.array(fd_length, dtype=float),
            similarity=cosine_similarity(g, fd_length),
            angle=angle_deg,
            init_params=np.array(init_params, dtype=float),
        )

    assert angle_deg < GRADIENT_OVERLAP_THRESHOLD_DEGREES, (
        "Shape adjoint gradient direction not lining up with finite difference"
    )


@pytest.mark.numerical
@pytest.mark.parametrize("cond_std_idx,cond_std", list(enumerate([1.0, 2.0, 4.0])))
@pytest.mark.parametrize(
    "dir_name", [NUMERICAL_RESULTS_DATA_DIR] if SAVE_FD_ADJ_DATA else [None], indirect=["dir_name"]
)
def test_finite_difference_lossy_metal_cond(rng, cond_std_idx, cond_std, create_directory):
    """Test conductivity gradients for `Box` and `PolySlab` when using `LossyMetalMedium` and `PECMedium`."""

    td.config.logging_level = "ERROR"

    # pull shared constants
    cfg = build_common_config()
    freq0 = cfg["freq0"]
    freq_stopband = cfg["freq_stopband"]
    freqs = cfg["freqs"]
    lda0 = cfg["lda0"]
    mm = cfg["mm"]
    inf_eff = cfg["inf_eff"]
    h_sub = cfg["h_sub"]
    h_trace = cfg["h_trace"]
    sub_medium = cfg["sub_medium"]

    lossy_metal = td.LossyMetalMedium(
        conductivity=50, frequency_range=(np.min(freqs), np.max(freqs)), thickness=h_trace
    )

    # Microstrip parameters taken from [1]
    strips_W = [1.4 * mm, 2.4 * mm, 2.4 * mm, 1.4 * mm]
    strips_L = [6 * mm, 6 * mm, 6 * mm, 6 * mm]
    strips_G = [0.2 * mm, 0.2 * mm, 0.2 * mm, 0.2 * mm]

    # Numpy arrays of the microstrip dimensions
    lengths = np.array(strips_L)
    widths = np.array(strips_W)
    gaps = np.array(strips_G)

    num_boxes = len(strips_W)
    all_parameters = strips_L + strips_W + strips_G

    (total_L_static, total_W_static, xstarts_static, ystarts_static) = get_coupled_line_geometry(
        lengths, widths, gaps
    )
    lengths_static = lengths.copy()
    widths_static = widths.copy()
    gaps_static = gaps.copy()

    def objective_fn(params):
        # Create the structures and geometry of the coupled line bandpass filter
        (total_L, total_W, xstarts, ystarts) = get_coupled_line_geometry(lengths, widths, gaps)

        coupled_lines = generate_coupled_lines(
            lengths, widths, gaps, xstarts, ystarts, lossy_metal, h_trace
        )
        # Create the substrate block
        substrate = td.Structure(
            geometry=td.Box.from_bounds(
                rmin=(-inf_eff, -inf_eff, -inf_eff), rmax=(inf_eff, inf_eff, 0)
            ),
            medium=sub_medium,
        )

        modified_lines = coupled_lines.copy()
        modified_lines = [
            s.updated_copy(medium=s.medium.updated_copy(conductivity=params[idx]))
            for idx, s in enumerate(coupled_lines)
        ]

        # Define the simulation domain size with some extra padding based on central wavelength
        Lx = (total_L_static) + lda0 / 5
        Ly = (total_W_static) + lda0 / 5
        Lz = h_sub + lda0 / 8

        # Definition of mesh overrides using shared helper
        mesh_overrides = build_mesh_overrides(
            total_L_static, total_W_static, widths_static, h_sub, h_trace
        )

        # Field monitor to view the electromagnetic fields along the propagation direction.
        field_monitor = td.FieldMonitor(
            center=(0, 0, -h_sub / 2),
            size=(td.inf, td.inf, 0),
            freqs=[freq_stopband, freq0],
            name="field",
        )
        # Boundary conditions are perfectly matched layers, except for the minus z boundary.
        boundary_spec = td.BoundarySpec(
            x=td.Boundary.pml(),
            y=td.Boundary.pml(),
            z=td.Boundary(minus=td.PECBoundary(), plus=td.PML()),
        )
        # The base tidy3D Simulation ready to be used by the TerminalComponentModeler
        sim = td.Simulation(
            center=(0, 0, Lz / 2 - h_sub),
            size=(Lx, Ly, Lz),
            grid_spec=td.GridSpec.auto(
                min_steps_per_wvl=40.0,
                wavelength=lda0,
                dl_min=h_trace / 5,
                override_structures=mesh_overrides,
            ),
            structures=[substrate, *modified_lines],
            sources=[],
            monitors=[field_monitor],
            run_time=10e-9,
            boundary_spec=boundary_spec,
        )

        # Build ports using shared helper
        port_1_td, port_2_td = build_ports(
            xstarts_static, ystarts_static, lengths_static, widths_static, gaps_static, h_sub
        )

        # Use shared helper to run modeler and get objective
        return run_modeler_and_objective(sim, [port_1_td, port_2_td], freqs, f_sample=4.4e9)

    (total_L, total_W, xstarts, ystarts) = get_coupled_line_geometry(lengths, widths, gaps)
    coupled_lines = generate_coupled_lines(
        lengths, widths, gaps, xstarts, ystarts, lossy_metal, h_trace, use_polyslab=False
    )

    init_conductivity = rng.normal(LOSSY_METAL_CONDUCTIVITY_MEAN, cond_std, len(coupled_lines))
    static_parameters = all_parameters.copy()

    val_and_grad = ag.value_and_grad(objective_fn)

    f, g = val_and_grad(init_conductivity)

    h = 0.5
    fd_cond = finite_difference(objective_fn, init_conductivity, h)

    if PLOT_FD_ADJ_COMPARISON:
        plt.subplot(1, 2, 1)
        plt.plot(g, color="g", linewidth=2.0)
        plt.plot(fd_cond, color="b", linewidth=1.5, linestyle="--")
        plt.legend(["Finite difference", "Adjoint"])
        plt.title("Unnormalized Gradient Comparison (conductivity)")
        plt.xlabel("Sample number")
        plt.ylabel("Gradient value")
        plt.subplot(1, 2, 2)
        plt.plot(g / np.linalg.norm(g), color="g", linewidth=2.0)
        plt.plot(fd_cond / np.linalg.norm(fd_cond), color="b", linewidth=1.5, linestyle="--")
        plt.title("Normalized Gradient Comparison (conductivity)")
        plt.xlabel("Sample number")
        plt.ylabel("Gradient value")
        plt.legend(["Finite difference", "Adjoint"])
        plt.show()

    angle_deg = np.arccos(cosine_similarity(g, fd_cond)) * 180.0 / np.pi
    if SAVE_FD_ADJ_DATA and NUMERICAL_RESULTS_DATA_DIR:
        np.savez(
            os.path.join(
                NUMERICAL_RESULTS_DATA_DIR, f"conductivity_results_std_{cond_std_idx}.npz"
            ),
            adjoint=np.array(g, dtype=float),
            finite_difference=np.array(fd_cond, dtype=float),
            similarity=cosine_similarity(g, fd_cond),
            angle=angle_deg,
            init_params=np.array(init_conductivity, dtype=float),
            cond_std=float(cond_std),
        )

    assert angle_deg < GRADIENT_OVERLAP_THRESHOLD_DEGREES, (
        "Conductivity adjoint gradient direction not lining up with finite difference"
    )
