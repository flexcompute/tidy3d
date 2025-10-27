"""Utility functions for microwave tests.

This module contains helper functions for creating canonical transmission line simulations
and other test fixtures used across microwave-related tests.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

import tidy3d as td

# Constants
mm = 1e3


def make_mw_sim(
    use_2D: bool = False,
    colocate: bool = False,
    transmission_line_type: Literal["microstrip", "cpw", "coax", "stripline"] = "microstrip",
    width=3 * mm,
    height=1 * mm,
    metal_thickness=0.2 * mm,
) -> td.Simulation:
    """Helper to create a microwave simulation with a single type of transmission line present."""

    freq_start = 1e9
    freq_stop = 10e9

    freq0 = (freq_start + freq_stop) / 2
    fwidth = freq_stop - freq_start
    freqs = np.arange(freq_start, freq_stop, 1e9)

    run_time = 60 / fwidth

    length = 40 * mm
    sim_width = length

    pec = td.PEC
    if use_2D:
        metal_thickness = 0.0
        pec = td.PEC2D

    epsr = 4.4
    diel = td.Medium(permittivity=epsr)

    metal_geos = []

    if transmission_line_type == "microstrip":
        substrate = td.Structure(
            geometry=td.Box(
                center=[0, 0, 0],
                size=[td.inf, td.inf, 2 * height],
            ),
            medium=diel,
        )
        metal_geos.append(
            td.Box(
                center=[0, 0, height + metal_thickness / 2],
                size=[td.inf, width, metal_thickness],
            )
        )
    elif transmission_line_type == "cpw":
        substrate = td.Structure(
            geometry=td.Box(
                center=[0, 0, 0],
                size=[td.inf, td.inf, 2 * height],
            ),
            medium=diel,
        )
        metal_geos.append(
            td.Box(
                center=[0, 0, height + metal_thickness / 2],
                size=[td.inf, width, metal_thickness],
            )
        )
        gnd_width = 10 * width
        gap = width / 5
        gnd_shift = gnd_width / 2 + gap + width / 2
        metal_geos.append(
            td.Box(
                center=[0, -gnd_shift, height + metal_thickness / 2],
                size=[td.inf, gnd_width, metal_thickness],
            )
        )
        metal_geos.append(
            td.Box(
                center=[0, gnd_shift, height + metal_thickness / 2],
                size=[td.inf, gnd_width, metal_thickness],
            )
        )
    elif transmission_line_type == "coax":
        substrate = td.Structure(
            geometry=td.Box(
                center=[0, 0, 0],
                size=[td.inf, td.inf, 2 * height],
            ),
            medium=diel,
        )
        metal_geos.append(
            td.GeometryGroup(
                geometries=(
                    td.ClipOperation(
                        operation="difference",
                        geometry_a=td.Cylinder(
                            axis=0, radius=2 * mm, center=(0, 0, 5 * mm), length=td.inf
                        ),
                        geometry_b=td.Cylinder(
                            axis=0, radius=1.8 * mm, center=(0, 0, 5 * mm), length=td.inf
                        ),
                    ),
                    td.Cylinder(axis=0, radius=0.6 * mm, center=(0, 0, 5 * mm), length=td.inf),
                )
            )
        )
    elif transmission_line_type == "stripline":
        substrate = td.Structure(
            geometry=td.Box(
                center=[0, 0, 0],
                size=[td.inf, td.inf, 2 * height + metal_thickness],
            ),
            medium=diel,
        )
        metal_geos.append(
            td.Box(
                center=[0, 0, 0],
                size=[td.inf, width, metal_thickness],
            )
        )
        gnd_width = 10 * width
        metal_geos.append(
            td.Box(
                center=[0, 0, height + metal_thickness],
                size=[td.inf, gnd_width, metal_thickness],
            )
        )
        metal_geos.append(
            td.Box(
                center=[0, 0, -height - metal_thickness],
                size=[td.inf, gnd_width, metal_thickness],
            )
        )
    else:
        raise AssertionError("Incorrect argument")

    metal_structures = [td.Structure(geometry=geo, medium=pec) for geo in metal_geos]
    structures = [substrate, *metal_structures]
    boundary_spec = td.BoundarySpec(
        x=td.Boundary(plus=td.PML(), minus=td.PML()),
        y=td.Boundary(plus=td.PML(), minus=td.PML()),
        z=td.Boundary(plus=td.PECBoundary(), minus=td.PECBoundary()),
    )

    size_sim = [
        length + 2 * width,
        sim_width,
        20 * mm + height + metal_thickness,
    ]
    center_sim = [0, 0, size_sim[2] / 2]
    # Slightly different setup for stripline substrate sandwiched between ground planes
    if transmission_line_type == "stripline":
        center_sim[2] = 0
        boundary_spec = td.BoundarySpec(
            x=td.Boundary(plus=td.PML(), minus=td.PML()),
            y=td.Boundary(plus=td.PML(), minus=td.PML()),
            z=td.Boundary(plus=td.PML(), minus=td.PML()),
        )
    size_port = [0, sim_width, size_sim[2]]
    center_port = [0, 0, center_sim[2]]
    impedance_specs = (td.AutoImpedanceSpec(),) * 4
    mode_spec = td.MicrowaveModeSpec(
        num_modes=4,
        target_neff=1.8,
        impedance_specs=impedance_specs,
    )

    mode_monitor = td.MicrowaveModeMonitor(
        center=center_port, size=size_port, freqs=freqs, name="mode_1", colocate=colocate
    )

    gaussian = td.GaussianPulse(freq0=freq0, fwidth=fwidth)
    mode_src = td.ModeSource(
        center=(-length / 2, 0, center_sim[2]),
        size=size_port,
        direction="+",
        mode_spec=mode_spec,
        mode_index=0,
        source_time=gaussian,
    )
    sim = td.Simulation(
        center=center_sim,
        size=size_sim,
        grid_spec=td.GridSpec.uniform(dl=0.1 * mm),
        structures=structures,
        sources=[mode_src],
        monitors=[mode_monitor],
        run_time=run_time,
        boundary_spec=boundary_spec,
        plot_length_units="mm",
        symmetry=(0, 0, 0),
    )
    return sim


def make_minimal_mode_plane_analyzer(**kwargs):
    """Helper to create a ModePlaneAnalyzer with minimal simulation data for testing."""
    from tidy3d.components.microwave.path_integrals.mode_plane_analyzer import (
        ModePlaneAnalyzer,
    )

    # Create a minimal simulation for testing
    minimal_sim = td.Simulation(
        size=(10, 10, 10),
        grid_spec=td.GridSpec.uniform(dl=1.0),
        run_time=1e-12,
    )

    # Default kwargs
    defaults = {
        "size": (0, 2, 2),
        "center": (0, 0, 0),
        "field_data_colocated": False,
        "structures": minimal_sim.structures,
        "grid": minimal_sim.grid,
        "symmetry": minimal_sim.symmetry,
        "sim_box": minimal_sim.bounding_box,
    }
    defaults.update(kwargs)

    return ModePlaneAnalyzer(**defaults)


def make_coupled_microstrip_sim(gap=1 * mm, width=3 * mm, height=1 * mm):
    """Create simulation with two coupled microstrip lines."""
    freq_start = 1e9
    freq_stop = 10e9
    freq0 = (freq_start + freq_stop) / 2
    fwidth = freq_stop - freq_start
    freqs = np.arange(freq_start, freq_stop, 1e9)
    run_time = 60 / fwidth
    length = 40 * mm
    sim_width = 50 * mm

    pec = td.PEC
    epsr = 4.4
    diel = td.Medium(permittivity=epsr)
    metal_thickness = 0.2 * mm

    # Substrate
    substrate = td.Structure(
        geometry=td.Box(
            center=[0, 0, 0],
            size=[td.inf, td.inf, 2 * height],
        ),
        medium=diel,
        name="substrate",
    )

    # Two signal lines
    signal_1 = td.Box(
        center=[0, -gap / 2 - width / 2, height + metal_thickness / 2],
        size=[td.inf, width, metal_thickness],
    )
    signal_2 = td.Box(
        center=[0, gap / 2 + width / 2, height + metal_thickness / 2],
        size=[td.inf, width, metal_thickness],
    )

    # Ground plane (not needed for the tests, but keeps simulation consistent)
    ground = td.Box(
        center=[0, 0, -metal_thickness / 2],
        size=[td.inf, td.inf, metal_thickness],
    )

    structures = [
        substrate,
        td.Structure(geometry=signal_1, medium=pec, name="signal_1"),
        td.Structure(geometry=signal_2, medium=pec, name="signal_2"),
        td.Structure(geometry=ground, medium=pec, name="ground"),
    ]

    size_sim = [length + 2 * width, sim_width, 20 * mm + height + metal_thickness]
    center_sim = [0, 0, size_sim[2] / 2]
    size_port = [0, sim_width, size_sim[2]]
    center_port = [0, 0, center_sim[2]]

    impedance_specs = (td.AutoImpedanceSpec(),) * 3
    mode_spec = td.MicrowaveModeSpec(
        num_modes=3,
        target_neff=1.8,
        impedance_specs=impedance_specs,
    )

    mode_monitor = td.MicrowaveModeMonitor(
        center=center_port, size=size_port, freqs=freqs, name="mode", colocate=False
    )

    gaussian = td.GaussianPulse(freq0=freq0, fwidth=fwidth)
    mode_src = td.ModeSource(
        center=(-length / 2, 0, center_sim[2]),
        size=size_port,
        direction="+",
        mode_spec=mode_spec,
        mode_index=0,
        source_time=gaussian,
    )

    boundary_spec = td.BoundarySpec(
        x=td.Boundary(plus=td.PML(), minus=td.PML()),
        y=td.Boundary(plus=td.PML(), minus=td.PML()),
        z=td.Boundary(plus=td.PML(), minus=td.PECBoundary()),
    )

    return td.Simulation(
        center=center_sim,
        size=size_sim,
        grid_spec=td.GridSpec.uniform(dl=0.1 * mm),
        structures=structures,
        sources=[mode_src],
        monitors=[mode_monitor],
        run_time=run_time,
        boundary_spec=boundary_spec,
    )


def make_stripline_mode_solver(
    width=1.0 * mm,
    height=0.5 * mm,
    metal_thickness=0.1 * mm,
    dl=0.05 * mm,
    freqs=None,
):
    """Create configured stripline mode solver for testing.

    Parameters
    ----------
    width : float
        Width of the stripline signal conductor.
    height : float
        Height from signal conductor to ground plane.
    metal_thickness : float
        Thickness of metal conductors.
    dl : float
        Grid resolution.
    freqs : list of float, optional
        Frequency points for mode solver. Defaults to [1e9, 5e9, 10e9].

    Returns
    -------
    tuple[ModeSolver, Simulation]
        Configured mode solver and stripline simulation.
    """
    from tidy3d.components.mode.mode_solver import ModeSolver

    if freqs is None:
        freqs = [1e9, 5e9, 10e9]

    stripline_sim = make_mw_sim(
        transmission_line_type="stripline",
        width=width,
        height=height,
        metal_thickness=metal_thickness,
    )
    stripline_sim = stripline_sim.updated_copy(grid_spec=td.GridSpec.uniform(dl=dl))

    plane = td.Box(center=(0, 0, 0), size=(0, 10 * width, 2 * height + metal_thickness))
    impedance_specs = td.AutoImpedanceSpec()
    mode_spec = td.MicrowaveModeSpec(
        num_modes=3,
        target_neff=2.2,
        impedance_specs=impedance_specs,
    )

    mms = ModeSolver(
        simulation=stripline_sim,
        plane=plane,
        mode_spec=mode_spec,
        colocate=False,
        freqs=freqs,
    )

    return mms, stripline_sim
