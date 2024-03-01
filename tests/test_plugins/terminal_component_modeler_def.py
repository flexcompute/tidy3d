import numpy as np

import tidy3d as td
from tidy3d.plugins.smatrix import (
    LumpedPort,
    TerminalComponentModeler,
)

# Microstrip dimensions
unit = 1e6
default_strip_length = 75e-3 * unit
strip_width = 3e-3 * unit
gap = 1e-3 * unit
gnd_width = strip_width * 8
metal_thickness = 0.2e-3 * unit

# Microstrip materials
pec = td.PECMedium()
pec_cond = td.Medium(conductivity=1e10)
pec2d = td.Medium2D(ss=pec_cond, tt=pec_cond)
diel = td.Medium(permittivity=4.4)

# Frequency setup
freq_start = 1e8
freq_stop = 10e9


def make_simulation(planar_pec: bool, length: float = None):
    if length:
        strip_length = length
    else:
        strip_length = default_strip_length

    if planar_pec:
        height = 0
        metal = pec2d
    else:
        height = metal_thickness
        metal = pec

    # wavelength / frequency
    freq0 = (freq_start + freq_stop) / 2
    fwidth = freq_stop - freq_start
    wavelength0 = td.C_0 / freq0
    run_time = 60 / fwidth

    # Spatial grid specification
    grid_spec = td.GridSpec.auto(min_steps_per_wvl=10, wavelength=td.C_0 / freq_stop)

    # Make structures
    strip = td.Structure(
        geometry=td.Box(
            center=[0, 0, height + gap + height / 2],
            size=[strip_length, strip_width, height],
        ),
        medium=metal,
    )

    ground = td.Structure(
        geometry=td.Box(
            center=[0, 0, height / 2],
            size=[strip_length, gnd_width, height],
        ),
        medium=metal,
    )

    substrate = td.Structure(
        geometry=td.Box(
            center=[0, 0, height + gap / 2],
            size=[strip_length, gnd_width, gap],
        ),
        medium=diel,
    )

    structures = [substrate, strip, ground]

    # Make simulation
    center_sim = [0, 0, height + gap / 2 + gap * 2]
    size_sim = [
        strip_length + 0.5 * wavelength0,
        gnd_width + 0.5 * wavelength0,
        2 * height + gap + 0.5 * wavelength0,
    ]

    sim = td.Simulation(
        center=center_sim,
        size=size_sim,
        grid_spec=grid_spec,
        structures=structures,
        sources=[],
        monitors=[],
        run_time=run_time,
        boundary_spec=td.BoundarySpec.all_sides(boundary=td.PML()),
        shutoff=1e-4,
    )

    return sim


def make_component_modeler(
    planar_pec: bool, reference_impedance: complex = 50, length: float = None, **kwargs
):
    if length:
        strip_length = length
    else:
        strip_length = default_strip_length

    sim = make_simulation(planar_pec, length=length)

    if planar_pec:
        height = 0
    else:
        height = metal_thickness

    center_src1 = [-strip_length / 2, 0, height + gap / 2]
    size_src1 = [0, strip_width, gap]

    center_src2 = [strip_length / 2, 0, height + gap / 2]
    size_src2 = [0, strip_width, gap]

    port_cells = np.ceil(gap / (metal_thickness / 1))

    port_1 = LumpedPort(
        center=center_src1,
        size=size_src1,
        voltage_axis=2,
        name="lumped_port_1",
        num_grid_cells=port_cells,
        impedance=reference_impedance,
    )

    port_2 = LumpedPort(
        center=center_src2,
        size=size_src2,
        voltage_axis=2,
        name="lumped_port_2",
        num_grid_cells=port_cells,
        impedance=reference_impedance,
    )

    ports = [port_1, port_2]
    freqs = np.linspace(freq_start, freq_stop, 100)

    modeler = TerminalComponentModeler(
        simulation=sim, ports=ports, freqs=freqs, remove_dc_component=False, verbose=True, **kwargs
    )

    return modeler
