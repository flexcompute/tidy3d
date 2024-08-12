import gdstk
import numpy as np
from scipy.interpolate import make_interp_spline

import tidy3d as td
import tidy3d.plugins.design as tdd

lda0 = 1.55  # central wavelength
freq0 = td.C_0 / lda0  # central frequency
n_wav = 100  # Number of wavelengths to sample in the range
ldas = np.linspace(1.5, 1.6, n_wav)  # wavelength range
freqs = td.C_0 / ldas  # frequency range
fwidth = 0.5 * (np.max(freqs) - np.min(freqs))  # width of the source frequency range

si = td.material_library["cSi"]["Palik_Lossless"]
sio2 = td.material_library["SiO2"]["Palik_Lossless"]

t = 0.22  # thickness of the silicon layer
num_d = 13  # dimensional space of the design region

l_in = 1  # input waveguide length
l_junction = 2  # length of the junction
l_bend = 6  # horizontal length of the waveguide bend
h_bend = 2  # vertical offset of the waveguide bend
l_out = 1  # output waveguide length
branch_width = 0.5  # width of one Y branch
branch_sep = 0.2  # distance between y branches at the junction
inf_eff = 100  # effective infinity


def fn_pre(**params):
    w_start = 0.5
    w_end = branch_width * 2 + branch_sep

    widths = [w_start]  # Ensures input waveguide is included in spline for first point of junction
    widths.extend(list(params.values()))
    widths.append(w_end)  # Ensures final point of junction smoothly converts to the branches

    x_junction = np.linspace(
        l_in, l_in + l_junction, num_d + 2
    )  # x coordinates of the top edge vertices
    y_junction = np.array(widths)  # y coordinates of the top edge vertices

    # pass vertices through spline and increase sampling to smooth the geometry
    new_x_junction = np.linspace(
        l_in, l_in + l_junction, 100
    )  # x coordinates of the top edge vertices
    spline = make_interp_spline(x_junction, y_junction, k=2)
    spline_yjunction = spline(new_x_junction)

    # using concatenate to include bottom edge vertices
    x_junction = np.concatenate((new_x_junction, np.flipud(new_x_junction)))
    y_junction = np.concatenate((spline_yjunction / 2, -np.flipud(spline_yjunction / 2)))

    # stacking x and y coordinates to form vertices pairs
    vertices = np.transpose(np.vstack((x_junction, y_junction)))

    junction = td.Structure(
        geometry=td.PolySlab(vertices=vertices, axis=2, slab_bounds=(0, t)), medium=si
    )

    x_start = l_in + l_junction  # x coordinate of the starting point of the waveguide bends

    x_bend = np.linspace(x_start, x_start + l_bend, 100)  # x coordinates of the top edge vertices

    y_bend = (
        (x_bend - x_start) * h_bend / l_bend
        - h_bend * np.sin(2 * np.pi * (x_bend - x_start) / l_bend) / (np.pi * 2)
        + w_end / 2
        - w_start / 2
    )  # y coordinates of the top edge vertices

    # adding the last point to include the straight waveguide at the output
    x_bend = np.append(x_bend, inf_eff)
    y_bend = np.append(y_bend, y_bend[-1])

    # add path to the cell
    cell = gdstk.Cell("bends")
    cell.add(
        gdstk.FlexPath(x_bend + 1j * y_bend, branch_width, layer=1, datatype=0)
    )  # top waveguide bend
    cell.add(
        gdstk.FlexPath(x_bend - 1j * y_bend, branch_width, layer=1, datatype=0)
    )  # bottom waveguide bend

    # define top waveguide bend structure
    wg_bend_1 = td.Structure(
        geometry=td.PolySlab.from_gds(
            cell,
            gds_layer=1,
            axis=2,
            slab_bounds=(0, t),
        )[0],
        medium=si,
    )

    # define bottom waveguide bend structure
    wg_bend_2 = td.Structure(
        geometry=td.PolySlab.from_gds(
            cell,
            gds_layer=1,
            axis=2,
            slab_bounds=(0, t),
        )[1],
        medium=si,
    )

    # straight input waveguide
    wg_in = td.Structure(
        geometry=td.Box.from_bounds(rmin=(-inf_eff, -w_start / 2, 0), rmax=(l_in, w_start / 2, t)),
        medium=si,
    )

    # the entire model is the collection of all structures defined so far
    model_structure = [wg_in, junction, wg_bend_1, wg_bend_2]

    Lx = l_in + l_junction + l_out + l_bend  # simulation domain size in x direction
    Ly = w_end + 2 * h_bend + 1.5 * lda0  # simulation domain size in y direction
    Lz = 10 * t  # simulation domain size in z direction
    sim_size = (Lx, Ly, Lz)

    # add a mode source as excitation
    mode_spec = td.ModeSpec(num_modes=1, target_neff=3.5)
    mode_source = td.ModeSource(
        center=(l_in / 2, 0, t / 2),
        size=(0, 4 * w_start, 6 * t),
        source_time=td.GaussianPulse(freq0=freq0, fwidth=fwidth),
        direction="+",
        mode_spec=mode_spec,
        mode_index=0,
    )

    # add a mode monitor to measure transmission at the output waveguide
    mode_monitor_11 = td.ModeMonitor(
        center=(l_in / 3, 0, t / 2),
        size=(0, 4 * w_start, 6 * t),
        freqs=freqs,
        mode_spec=mode_spec,
        name="mode_11",
    )

    mode_monitor_12 = td.ModeMonitor(
        center=(l_in + l_junction + l_bend + l_out / 2, w_end / 2 - w_start / 2 + h_bend, t / 2),
        size=(0, 4 * w_start, 6 * t),
        freqs=freqs,
        mode_spec=mode_spec,
        name="mode_12",
    )

    # add a field monitor to visualize field distribution at z=t/2
    field_monitor = td.FieldMonitor(
        center=(0, 0, t / 2), size=(td.inf, td.inf, 0), freqs=[freq0], name="field"
    )

    run_time = 5e-13  # simulation run time

    # construct simulation
    sim = td.Simulation(
        center=(Lx / 2, 0, 0),
        size=sim_size,
        grid_spec=td.GridSpec.auto(min_steps_per_wvl=20, wavelength=lda0),
        structures=model_structure,
        sources=[mode_source],
        monitors=[mode_monitor_11, mode_monitor_12, field_monitor],
        run_time=run_time,
        boundary_spec=td.BoundarySpec.all_sides(boundary=td.PML()),
        medium=sio2,
    )

    return sim


def fn_post(sim_data):
    power_reflected = np.squeeze(
        np.abs(sim_data["mode_11"].amps.sel(direction="-", mode_index=0)) ** 2
    )
    power_transmitted = np.squeeze(
        np.abs(sim_data["mode_12"].amps.sel(direction="+", mode_index=0)) ** 2
    )

    loss_fn = 1 / 3 * n_wav * np.sum(power_reflected**2 + 2 * (power_transmitted - 0.5) ** 2)
    output = -float(loss_fn.values)  # Negative value as this is a minimizing loss function

    return output


method = tdd.MethodMonteCarlo(
    num_points=600,
    rng_seed=5,
)

parameters = [tdd.ParameterFloat(name=f"w_{i}", span=(0.5, 1.6)) for i in range(num_d)]
output_dir = "/home/matt/Documents/Flexcompute/y_split/data"
design_space = tdd.DesignSpace(
    method=method,
    parameters=parameters,
    task_name="y_split_5",
    folder_name="YSplitV1",
    path_dir=output_dir,
)

results = design_space.run(fn_pre, fn_post, verbose=False)
