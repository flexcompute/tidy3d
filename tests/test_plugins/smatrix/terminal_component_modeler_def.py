from __future__ import annotations

import numpy as np

import tidy3d as td
import tidy3d.plugins.microwave as mw
from tidy3d.plugins.smatrix import (
    CoaxialLumpedPort,
    LumpedPort,
    TerminalComponentModeler,
    WavePort,
)

# Microstrip dimensions
mm = 1e3
default_strip_length = 75 * mm
strip_width = 3 * mm
gap = 1 * mm
gnd_width = strip_width * 8
metal_thickness = 0.2 * mm

# Microstrip materials
pec = td.PECMedium()
pec_cond = td.Medium(conductivity=1e10)
pec2d = td.Medium2D(ss=pec_cond, tt=pec_cond)
diel = td.Medium(permittivity=4.4)

# Frequency setup
freq_start = 1e8
freq_stop = 10e9

# Coaxial dimensions
Rinner = 0.2768 * mm
Router = 1.0 * mm


def make_simulation(planar_pec: bool, length: float | None = None, grid_spec: td.GridSpec = None):
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
    if not grid_spec:
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
    planar_pec: bool,
    reference_impedance: complex = 50,
    length: float | None = None,
    port_refinement: bool = True,
    port_snapping: bool = True,
    grid_spec: td.GridSpec = None,
    **kwargs,
):
    if length:
        strip_length = length
    else:
        strip_length = default_strip_length

    sim = make_simulation(planar_pec, length=length, grid_spec=grid_spec)

    if planar_pec:
        height = 0
    else:
        height = metal_thickness

    center_src1 = [-strip_length / 2, 0, height + gap / 2]
    size_src1 = [0, strip_width, gap]

    center_src2 = [strip_length / 2, 0, height + gap / 2]
    size_src2 = [0, strip_width, gap]

    port_cells = None
    if port_refinement:
        port_cells = np.ceil(gap / (metal_thickness / 1))

    port_1 = LumpedPort(
        center=center_src1,
        size=size_src1,
        voltage_axis=2,
        name="lumped_port_1",
        num_grid_cells=port_cells,
        enable_snapping_points=port_snapping,
        impedance=reference_impedance,
    )

    port_2 = LumpedPort(
        center=center_src2,
        size=size_src2,
        voltage_axis=2,
        name="lumped_port_2",
        num_grid_cells=port_cells,
        enable_snapping_points=port_snapping,
        impedance=reference_impedance,
    )

    ports = [port_1, port_2]
    freqs = np.linspace(freq_start, freq_stop, 100)

    modeler = TerminalComponentModeler(
        simulation=sim, ports=ports, freqs=freqs, remove_dc_component=False, **kwargs
    )

    return modeler


def make_coaxial_simulation(length: float | None = None, grid_spec: td.GridSpec = None):
    if not length:
        length = default_strip_length

    # wavelength / frequency
    freq0 = (freq_start + freq_stop) / 2
    fwidth = freq_stop - freq_start
    wavelength0 = td.C_0 / freq0
    run_time = 60 / fwidth

    # Spatial grid specification
    if not grid_spec:
        grid_spec = td.GridSpec.auto(min_steps_per_wvl=10, wavelength=td.C_0 / freq_stop)

    # Make structures
    inner_conductor = td.Cylinder(
        center=(0, 0, 0),
        radius=Rinner,
        length=length,
        axis=2,
    )

    outer_1 = td.Cylinder(
        center=(0, 0, 0),
        radius=Router,
        length=length,
        axis=2,
    )

    outer_2 = td.Cylinder(
        center=(0, 0, 0),
        radius=Router * 1.1,
        length=length,
        axis=2,
    )

    outer_shell_clip = td.ClipOperation(
        operation="difference", geometry_a=outer_2, geometry_b=outer_1
    )

    inner = td.Structure(
        geometry=inner_conductor,
        medium=pec,
    )

    outer_shell = td.Structure(
        geometry=outer_shell_clip,
        medium=pec,
    )

    structures_list = [inner, outer_shell]

    # Make simulation
    center_sim = [0, 0, 0]
    size_sim = [
        4 * Router,
        4 * Router,
        length + 0.5 * wavelength0,
    ]

    sim = td.Simulation(
        center=center_sim,
        size=size_sim,
        grid_spec=grid_spec,
        structures=structures_list,
        sources=[],
        monitors=[],
        run_time=run_time,
        boundary_spec=td.BoundarySpec.all_sides(boundary=td.PML()),
        shutoff=1e-4,
    )

    return sim


def make_coaxial_component_modeler(
    reference_impedance: complex = 50,
    length: float | None = None,
    port_refinement: bool = True,
    grid_spec: td.GridSpec = None,
    port_types: tuple[CoaxialLumpedPort | WavePort, CoaxialLumpedPort | WavePort] = (
        CoaxialLumpedPort,
        CoaxialLumpedPort,
    ),
    use_current: bool = True,
    use_voltage: bool = True,
    **kwargs,
):
    if not length:
        length = default_strip_length

    sim = make_coaxial_simulation(length=length, grid_spec=grid_spec)

    def make_port(center, direction, type, name) -> CoaxialLumpedPort | WavePort:
        if type is CoaxialLumpedPort:
            port_cells = None
            if port_refinement:
                port_cells = 21
            port = CoaxialLumpedPort(
                center=center,
                outer_diameter=2 * Router,
                inner_diameter=2 * Rinner,
                normal_axis=2,
                direction=direction,
                name="coax" + name,
                num_grid_cells=port_cells,
                impedance=reference_impedance,
            )
        else:
            mean_radius = (Router + Rinner) / 2
            voltage_center = list(center)
            voltage_center[0] += mean_radius
            voltage_size = [Router - Rinner, 0, 0]

            voltage_integral = None
            if use_voltage:
                voltage_integral = td.AxisAlignedVoltageIntegral(
                    center=voltage_center,
                    size=voltage_size,
                    extrapolate_to_endpoints=True,
                    snap_path_to_grid=True,
                    sign="+",
                )
            current_integral = None
            if use_current:
                current_integral = td.Custom2DCurrentIntegral.from_circular_path(
                    center=center,
                    radius=mean_radius,
                    num_points=41,
                    normal_axis=2,
                    clockwise=direction != "+",
                )
            port_cells = None
            if port_refinement:
                port_cells = 5
            port = WavePort(
                center=center,
                size=[2 * Router, 2 * Router, 0],
                direction=direction,
                name="wave" + name,
                mode_spec=td.ModeSpec(num_modes=1),
                mode_index=0,
                voltage_integral=voltage_integral,
                current_integral=current_integral,
                num_grid_cells=port_cells,
            )
        return port

    center_src1 = [0, 0, -length / 2]
    port_1 = make_port(center_src1, direction="+", type=port_types[0], name="_1")
    center_src2 = [0, 0, length / 2]
    port_2 = make_port(center_src2, direction="-", type=port_types[1], name="_2")
    ports = [port_1, port_2]
    freqs = np.linspace(freq_start, freq_stop, 100)

    modeler = TerminalComponentModeler(
        simulation=sim, ports=ports, freqs=freqs, remove_dc_component=False, **kwargs
    )

    return modeler


def make_differential_stripline_modeler():
    # Frequency range (Hz)
    f_min, f_max = (1e9, 70e9)

    # Frequency sample points
    freqs = np.linspace(f_min, f_max, 101)

    # Geometry
    mil = 25.4  # conversion to mils to microns (default unit)
    w = 3.2 * mil  # Signal strip width
    t = 0.7 * mil  # Conductor thickness
    h = 10.7 * mil  # Substrate thickness
    se = 7 * mil  # gap between edge-coupled pair
    L = 4000 * mil  # Line length
    len_inf = 1e6  # Effective infinity

    left_end = -L / 2
    right_end = len_inf

    len_z = right_end - left_end
    cent_z = (left_end + right_end) / 2
    waveport_z = L

    # Material properties
    eps = 4.4  # Relative permittivity, substrate

    # define media
    med_sub = td.Medium(permittivity=eps)
    med_metal = td.PEC

    left_strip_geometry = td.Box(center=(-(se + w) / 2, 0, 0), size=(w, t, L))
    right_strip_geometry = td.Box(center=((se + w) / 2, 0, 0), size=(w, t, L))

    # Substrate
    str_sub = td.Structure(geometry=td.Box(center=(0, 0, 0), size=(len_inf, h, L)), medium=med_sub)

    # disjoint signal strips
    str_signal_strips = td.Structure(
        geometry=td.GeometryGroup(geometries=[left_strip_geometry, right_strip_geometry]),
        medium=med_metal,
    )

    # Top ground plane
    str_gnd_top = td.Structure(
        geometry=td.Box(center=(0, h / 2 + t / 2, 0), size=(len_inf, t, L)), medium=med_metal
    )

    # Bottom ground plane
    str_gnd_bot = td.Structure(
        geometry=td.Box(center=(0, -h / 2 - t / 2, 0), size=(len_inf, t, L)), medium=med_metal
    )

    # Create a LayerRefinementSpec from signal trace structures
    lr_spec = td.LayerRefinementSpec.from_structures(
        structures=[str_signal_strips],
        axis=1,  # Layer normal is in y-direction
        min_steps_along_axis=10,  # Min 10 grid cells along normal direction
        refinement_inside_sim_only=False,  # Metal structures extend outside sim domain. Set 'False' to snap to corners outside sim.
        bounds_snapping="bounds",  # snap grid to metal boundaries
        corner_refinement=td.GridRefinement(
            dl=t / 10, num_cells=2
        ),  # snap to corners and apply added refinement
    )

    # Layer refinement for top and bottom ground planes
    lr_spec2 = lr_spec.updated_copy(center=(0, h / 2 + t / 2, cent_z), size=(len_inf, t, len_z))
    lr_spec3 = lr_spec.updated_copy(center=(0, -h / 2 - t / 2, cent_z), size=(len_inf, t, len_z))

    # Define overall grid specification
    grid_spec = td.GridSpec.auto(
        wavelength=td.C_0 / f_max,
        min_steps_per_wvl=30,
        layer_refinement_specs=[lr_spec, lr_spec2, lr_spec3],
    )

    # boundary specs
    boundary_spec = td.BoundarySpec(
        x=td.Boundary.pml(),
        y=td.Boundary.pec(),
        z=td.Boundary.pml(),
    )

    # Define port specification
    wave_port_mode_spec = td.ModeSpec(num_modes=1, target_neff=np.sqrt(eps))

    # Define current and voltage integrals
    current_integral = mw.AxisAlignedCurrentIntegral(
        center=((se + w) / 2, 0, -waveport_z / 2), size=(2 * w, 3 * t, 0), sign="+"
    )
    voltage_integral = mw.AxisAlignedVoltageIntegral(
        center=(0, 0, -waveport_z / 2),
        size=(se, 0, 0),
        extrapolate_to_endpoints=True,
        snap_path_to_grid=True,
        sign="+",
    )

    # Define wave ports
    WP1 = WavePort(
        center=(0, 0, -waveport_z / 2),
        size=(len_inf, len_inf, 0),
        mode_spec=wave_port_mode_spec,
        direction="+",
        name="WP1",
        mode_index=0,
        current_integral=current_integral,
        voltage_integral=voltage_integral,
    )
    WP2 = WP1.updated_copy(
        name="WP2",
        center=(0, 0, waveport_z / 2),
        direction="-",
        current_integral=current_integral.updated_copy(
            center=((se + w) / 2, 0, waveport_z / 2), sign="-"
        ),
        voltage_integral=voltage_integral.updated_copy(center=(0, 0, waveport_z / 2)),
    )

    # define fimulation
    sim = td.Simulation(
        size=(50 * mil, h + 2 * t, 1.05 * L),
        center=(0, 0, 0),
        grid_spec=grid_spec,
        boundary_spec=boundary_spec,
        structures=[str_sub, str_signal_strips, str_gnd_top, str_gnd_bot],
        monitors=[],
        run_time=2e-9,  # simulation run time in seconds
        shutoff=1e-7,  # lower shutoff threshold for more accurate low frequency
        plot_length_units="mm",
        symmetry=(-1, 0, 0),  # odd symmetry in x-direction
    )

    # set up component modeler
    tcm = TerminalComponentModeler(
        simulation=sim,  # simulation, previously defined
        ports=[WP1, WP2],  # wave ports, previously defined
        freqs=freqs,  # S-parameter frequency points
    )

    return tcm
