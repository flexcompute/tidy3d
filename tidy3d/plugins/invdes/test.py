import numpy as np

import tidy3d as td
import tidy3d.plugins.invdes as tdi

from .web import Job

# source info
wavelength = 1.0

# waveguide parameters
num_output_waveguides = 3
ly_wg = 0.5 * wavelength
buffer_wg = 0.9 * wavelength

# buffer between design region, pml, and sources
buffer = 1 * wavelength

# relative permittivity of material
eps_mat = 4.0

# resolution (for both the FDTD simulation and for the design region)
min_steps_per_wvl = 30
pixel_size = wavelength / min_steps_per_wvl / np.sqrt(eps_mat)

# spectral information
freq0 = td.C_0 / wavelength
fwidth = freq0 / 10
run_time = 50 / fwidth

# design region size in y
ly_des = num_output_waveguides * (ly_wg + buffer_wg)
lx_des = 4 * wavelength

# simulation size
Lx = 2 * buffer + lx_des + 2 * buffer
Ly = buffer + ly_des + buffer

# source and monitor locations
x_src = -lx_des / 2 - buffer
x_mnt = -x_src

# material Medium
medium = td.Medium(permittivity=eps_mat)

# grid spec
grid_spec = td.GridSpec.auto(wavelength=wavelength, min_steps_per_wvl=min_steps_per_wvl)


# monitor names
def output_monitor_name(i: int) -> str:
    return f"MNT_{i}"


field_mnt_name = "field"

# mode spec
mode_spec = td.ModeSpec(num_modes=1)


if __name__ == "__main__":
    # radius (um) of the conic filter that is convolved with th parameter array.
    # Larger values tend to help create larger feature sizes in the final device.
    projection_radius = 0.120

    # projection strength, larger values lead to more binarization and tend to push intermediate parameters towards (0,1) density
    beta = 10.0

    # transformations on the parameters that lead to the material density array (0,1)
    filter_project = tdi.FilterProject(radius=projection_radius, beta=beta)

    # length scale (um) of the erosion dilation penalty.
    # features smaller than this scale will be penalized
    length_scale = 0.120

    # penalty weight, the penalty contributes its raw value (max of 1) times this weight to the objective function
    weight = 0.8

    # penalties applied to the state of the material density, after these transformations are applied
    penalty = tdi.ErosionDilationPenalty(weight=weight, length_scale=length_scale)

    design_region = tdi.TopologyDesignRegion(
        size=(lx_des, ly_des, td.inf),
        center=(0, 0, 0),
        eps_bounds=(1.0, eps_mat),  # the minimum and maximum permittivity values in the final grid
        transformations=[filter_project],
        penalties=[penalty],
        pixel_size=pixel_size,
    )

    waveguide_in = td.Structure(
        geometry=td.Box(
            size=(Lx, ly_wg, td.inf),
            center=(-Lx + 2 * buffer, 0, 0),
        ),
        medium=medium,
    )

    y_max_wg_centers = ly_des / 2 - buffer_wg / 2 - ly_wg / 2
    wg_y_centers_out = np.linspace(-y_max_wg_centers, y_max_wg_centers, num_output_waveguides)

    # put a waveguide and mode monitor at each of the outputs
    waveguides_out = []
    monitors_out = []
    for i, wg_y_center in enumerate(wg_y_centers_out):
        wg_out = td.Structure(
            geometry=td.Box(
                size=(Lx, ly_wg, td.inf),
                center=(Lx - 2 * buffer, wg_y_center, 0),
            ),
            medium=medium,
        )

        waveguides_out.append(wg_out)

        mnt_out = td.ModeMonitor(
            size=(0, ly_wg + 1.8 * buffer_wg, td.inf),
            center=(x_mnt, wg_y_center, 0),
            freqs=[freq0],
            name=output_monitor_name(i),
            mode_spec=mode_spec,
        )

        monitors_out.append(mnt_out)

    source = td.ModeSource(
        size=(0, ly_wg + 1.8 * buffer_wg, td.inf),
        center=(x_src, 0, 0),
        source_time=td.GaussianPulse(freq0=freq0, fwidth=fwidth),
        mode_index=0,
        direction="+",
    )

    # used to visualize fields in the plane, not for optimization
    fld_mnt = td.FieldMonitor(
        center=(0, 0, 0),
        size=(td.inf, td.inf, 0),
        freqs=[freq0],
        name=field_mnt_name,
    )

    simulation = td.Simulation(
        size=(Lx, Ly, 0),
        grid_spec=grid_spec,
        boundary_spec=td.BoundarySpec.pml(x=True, y=True, z=False),
        run_time=run_time,
        structures=[waveguide_in] + waveguides_out,
        sources=[source],
        monitors=[fld_mnt] + monitors_out,
    )

    design = tdi.InverseDesign(
        simulation=simulation,
        design_region=design_region,
        task_name="invdes",
        output_monitor_names=[mnt.name for mnt in monitors_out],
    )

    optimizer = tdi.AdamOptimizer(
        design=design,
        num_steps=10,
        learning_rate=0.1,
        results_cache_fname="data/invdes_history.hdf5",
    )

    optimizer.to_hdf5("data/invdes_history.hdf5")
    job = Job(optimizer=optimizer)
