from __future__ import annotations

import numpy as np

import tidy3d as td
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
            enable_snapping_points = False
            if port_refinement:
                port_cells = 21
                enable_snapping_points = True
            port = CoaxialLumpedPort(
                center=center,
                outer_diameter=2 * Router,
                inner_diameter=2 * Rinner,
                normal_axis=2,
                direction=direction,
                name="coax" + name,
                num_grid_cells=port_cells,
                impedance=reference_impedance,
                enable_snapping_points=enable_snapping_points,
            )
        else:
            mean_radius = (Router + Rinner) / 2
            voltage_center = list(center)
            voltage_center[0] += mean_radius
            voltage_size = [Router - Rinner, 0, 0]

            voltage_spec = None
            if use_voltage:
                voltage_spec = td.AxisAlignedVoltageIntegralSpec(
                    center=voltage_center,
                    size=voltage_size,
                    extrapolate_to_endpoints=True,
                    snap_path_to_grid=True,
                    sign="+",
                )
            current_spec = None
            if use_current:
                current_spec = td.Custom2DCurrentIntegralSpec.from_circular_path(
                    center=center,
                    radius=mean_radius,
                    num_points=41,
                    normal_axis=2,
                    clockwise=direction != "+",
                )
            mw_mode_spec = td.MicrowaveModeSpec(
                num_modes=1,
                impedance_specs=(
                    td.CustomImpedanceSpec(voltage_spec=voltage_spec, current_spec=current_spec),
                ),
            )
            port_cells = None
            if port_refinement:
                port_cells = 5
            port = WavePort(
                center=center,
                size=[2 * Router, 2 * Router, 0],
                direction=direction,
                name="wave" + name,
                mode_spec=mw_mode_spec,
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

    # Define current and voltage integrals
    current_spec = td.AxisAlignedCurrentIntegralSpec(
        center=((se + w) / 2, 0, -waveport_z / 2), size=(2 * w, 3 * t, 0), sign="+"
    )
    voltage_spec = td.AxisAlignedVoltageIntegralSpec(
        center=(0, 0, -waveport_z / 2),
        size=(se, 0, 0),
        extrapolate_to_endpoints=True,
        snap_path_to_grid=True,
        sign="+",
    )

    # Define port specification
    wave_port_mode_spec = td.MicrowaveModeSpec(
        num_modes=1,
        target_neff=np.sqrt(eps),
        impedance_specs=td.CustomImpedanceSpec(
            voltage_spec=voltage_spec, current_spec=current_spec
        ),
    )

    # Define wave ports
    WP1 = WavePort(
        center=(0, 0, -waveport_z / 2),
        size=(len_inf, len_inf, 0),
        mode_spec=wave_port_mode_spec,
        direction="+",
        name="WP1",
    )
    WP2 = WP1.updated_copy(
        name="WP2",
        center=(0, 0, waveport_z / 2),
        direction="-",
        mode_spec=wave_port_mode_spec.updated_copy(
            path="impedance_specs", current_spec=current_spec.updated_copy(sign="-")
        ),
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


def make_patch_antenna_simulation(padding: tuple[float, float, float] = (0.25, 0.25, 0.25)):
    """Create a rectangular patch antenna simulation on a dielectric substrate.

    Closely follows the patch antenna design from AntennaCharacteristics.ipynb,
    based on: Sheen, D.M., Ali, S.M., Abouzahra, M.D. and Kong, J.A., 1990.
    Application of the three-dimensional finite-difference time-domain method
    to the analysis of planar microstrip circuits.

    Parameters
    ----------
    padding : tuple[float, float, float]
        Padding between substrate and simulation domain boundaries in x, y, z directions.

    Returns
    -------
    td.Simulation
        Patch antenna simulation object.
    """
    # Frequency setup for patch antenna (5-11 GHz range for resonances at 7.5 and 10 GHz)
    freq_start_antenna = 5e9
    freq_stop_antenna = 11e9
    freq0 = (freq_start_antenna + freq_stop_antenna) / 2
    wavelength0 = td.C_0 / freq0

    # Metal thickness
    th = 0.05 * mm

    # Substrate parameters
    sub_x = 23.34 * mm
    sub_y = 40 * mm
    sub_z = 0.794 * mm

    # Patch parameters
    patch_x = 12.45 * mm
    patch_y = 16 * mm

    # Feed line parameters
    feed_x = 2.46 * mm
    feed_y = 20 * mm
    feed_offset = 2.09 * mm

    # Materials
    medium_sub = td.Medium(permittivity=2.2, name="Substrate")
    medium_metal = td.PECMedium()

    # Create structures for grid refinement calculation
    ground_plane_temp = td.Structure(
        geometry=td.Box(center=[0, 0, -(sub_z + th) / 2], size=[sub_x, sub_y, th]),
        medium=medium_metal,
        name="Ground",
    )

    feed_line_temp = td.Structure(
        geometry=td.Box.from_bounds(
            rmin=[-patch_x / 2 + feed_offset, -sub_y / 2, sub_z / 2],
            rmax=[-patch_x / 2 + feed_offset + feed_x, -sub_y / 2 + feed_y, sub_z / 2 + th],
        ),
        medium=medium_metal,
        name="Feed line",
    )

    patch_temp = td.Structure(
        geometry=td.Box.from_bounds(
            rmin=[-patch_x / 2, -sub_y / 2 + feed_y, sub_z / 2],
            rmax=[patch_x / 2, -sub_y / 2 + feed_y + patch_y, sub_z / 2 + th],
        ),
        medium=medium_metal,
        name="Patch",
    )

    # Create LayerRefinementSpec helper
    def create_lr_spec(structures_list):
        """Returns LayerRefinementSpec applied to the bounding box of the input structure list"""
        lr_spec = td.LayerRefinementSpec.from_structures(
            structures=structures_list,
            axis=2,  # Layer normal is oriented along the z-axis
            bounds_snapping="bounds",  # Snap grid lines to layer bounds in normal direction
            bounds_refinement=td.GridRefinement(
                dl=th, num_cells=2
            ),  # cell size and num cells at layer boundaries in normal direction
            corner_refinement=td.GridRefinement(
                dl=0.2 * mm, num_cells=2
            ),  # cell size and num cells around in-plane metal corners
        )
        return lr_spec

    # Layer refinement for the ground layer
    lr_spec_1 = create_lr_spec([ground_plane_temp])

    # Layer refinement for the patch antenna layer
    lr_spec_2 = create_lr_spec([feed_line_temp, patch_temp])

    # Define overall grid specification
    grid_spec = td.GridSpec.auto(
        wavelength=wavelength0, min_steps_per_wvl=25, layer_refinement_specs=[lr_spec_1, lr_spec_2]
    )

    # Create substrate
    substrate = td.Structure(
        geometry=td.Box(center=[0, 0, 0], size=[sub_x, sub_y, sub_z]),
        medium=medium_sub,
        name="Substrate",
    )

    # Create ground plane
    ground_plane = td.Structure(
        geometry=td.Box(center=[0, 0, -(sub_z + th) / 2], size=[sub_x, sub_y, th]),
        medium=medium_metal,
        name="Ground",
    )

    # Create feed line
    feed_line = td.Structure(
        geometry=td.Box.from_bounds(
            rmin=[-patch_x / 2 + feed_offset, -sub_y / 2, sub_z / 2],
            rmax=[-patch_x / 2 + feed_offset + feed_x, -sub_y / 2 + feed_y, sub_z / 2 + th],
        ),
        medium=medium_metal,
        name="Feed line",
    )

    # Create patch antenna
    patch = td.Structure(
        geometry=td.Box.from_bounds(
            rmin=[-patch_x / 2, -sub_y / 2 + feed_y, sub_z / 2],
            rmax=[patch_x / 2, -sub_y / 2 + feed_y + patch_y, sub_z / 2 + th],
        ),
        medium=medium_metal,
        name="Patch",
    )

    # Structures list (dielectric first, then metal/PEC)
    structures_list = [substrate, ground_plane, feed_line, patch]

    # Padding distance
    padding_um = td.C_0 / freq_start_antenna * np.array(padding)

    # Simulation size
    sim_x = sub_x + 2 * padding_um[0]
    sim_y = sub_y + 2 * padding_um[1]
    sim_z = sub_z + 2 * padding_um[2]

    # Define PMLs on all sides
    boundary_spec = td.BoundarySpec.pml(x=True, y=True, z=True)

    # Create simulation object
    sim = td.Simulation(
        center=[0, 0, 0],
        size=[sim_x, sim_y, sim_z],
        structures=structures_list,
        sources=[],
        monitors=[],
        boundary_spec=boundary_spec,
        grid_spec=grid_spec,
        run_time=5e-9,
        shutoff=1e-4,
        plot_length_units="mm",
    )

    return sim


def make_patch_antenna_modeler(padding: tuple[float, float, float] = (0.25, 0.25, 0.25)):
    """Create a TerminalComponentModeler for a rectangular patch antenna.

    Closely follows the antenna setup from AntennaCharacteristics.ipynb.

    Returns
    -------
    TerminalComponentModeler
        Component modeler for the patch antenna.
    """
    # Frequency setup for patch antenna
    freq_start_antenna = 5e9
    freq_stop_antenna = 11e9

    sim = make_patch_antenna_simulation(padding=padding)

    # Patch antenna dimensions
    patch_x = 12.45 * mm
    feed_x = 2.46 * mm
    feed_offset = 2.09 * mm
    sub_z = 0.794 * mm
    sub_y = 40 * mm

    # Create lumped port excitation at feed line
    port = LumpedPort(
        name="lumped_port",
        center=[-patch_x / 2 + feed_offset + feed_x / 2, -sub_y / 2, 0],
        size=[feed_x, 0, sub_z],
        voltage_axis=2,
        impedance=50,
    )

    # Frequency sweep (201 points as in notebook)
    freqs = np.linspace(freq_start_antenna, freq_stop_antenna, 201)

    # Target frequencies for field monitor
    freqs_target = [7.5e9, 10e9]

    # Field monitor
    monitor_field = td.FieldMonitor(
        center=(0, 0, sub_z / 2),
        size=(td.inf, td.inf, 0),
        freqs=freqs_target,
        name="field",
    )

    # Update simulation with field monitor
    sim = sim.updated_copy(monitors=[monitor_field])

    modeler = TerminalComponentModeler(
        simulation=sim,
        ports=[port],
        freqs=freqs,
        remove_dc_component=False,
        radiation_monitors=[],
    )

    return modeler


def make_basic_filter_terminals():
    # Frequency
    (f_min, f_max) = (0.1e9, 8e9)

    # Materials
    med_Cu = td.LossyMetalMedium(conductivity=60, frequency_range=(f_min, f_max))

    # Geometry and Structure
    mm = 1000  # Conversion mm to micron
    H = 0.8 * mm  # Substrate thickness
    T = 0.035 * mm  # Metal thickness
    WL = 0.5 * mm
    WC = 4 * mm
    LC = 5.3 * mm
    LL1 = 5.8 * mm
    LL2 = 1.2 * mm
    LL3 = 11.1 * mm
    Lsub = LL1 + WL + LL3
    Wsub = 2 * (LC + WL + LL2)

    geom_C = td.Box.from_bounds(rmin=(-WC / 2, 0, 0), rmax=(WC / 2, LC, T))
    geom_L2 = td.Box.from_bounds(rmin=(-WL / 2, -LL2 - WL, 0), rmax=(WL / 2, 0, T))
    geom_L1 = td.Box.from_bounds(rmin=(-WL / 2 - LL1, -LL2 - WL, 0), rmax=(-WL / 2, -LL2, T))
    geom_L3 = td.Box.from_bounds(rmin=(WL / 2, -LL2 - WL, 0), rmax=(WL / 2 + LL3, -LL2, T))

    geom_resonator_basic = td.GeometryGroup(geometries=[geom_C, geom_L1, geom_L2, geom_L3])

    x0, y0, _z0 = geom_resonator_basic.bounding_box.center  # center (x,y) with circuit
    geom_gnd = td.Box(center=(x0, y0, -H - T / 2), size=(Lsub, Wsub, T))

    str_gnd = td.Structure(geometry=geom_gnd, medium=med_Cu)
    str_resonator_basic = td.Structure(geometry=geom_resonator_basic, medium=med_Cu)

    # add second signal trace to test lateral_coord
    geom_sign = td.Box.from_bounds(
        rmin=(-WL / 2 - LL1, -2 * LL2 - WL, 0), rmax=(WL / 2 + LL3, -2 * LL2, T)
    )
    geom_resonator_modified = td.GeometryGroup(
        geometries=[geom_C, geom_L1, geom_L2, geom_L3, geom_sign]
    )
    str_resonator_modified = td.Structure(geometry=geom_resonator_modified, medium=med_Cu)

    return (str_gnd, str_resonator_basic, str_resonator_modified)
