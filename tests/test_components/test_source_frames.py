"""Tests frames around sources and absorbers."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

import tidy3d as td
from tidy3d.components.mode.mode_solver import ModeSolver


def test_source_absorber_frames():
    _ = td.PECFrame()
    with pytest.raises(ValidationError):
        _ = td.PECFrame(length=0)

    wvl_um = 1
    freq0 = td.C_0 / wvl_um
    mode_source = td.ModeSource(
        size=(1, 1, 0),
        source_time=td.GaussianPulse(freq0=freq0, fwidth=0.2 * freq0),
        mode_spec=td.ModeSpec(num_modes=2),
        mode_index=1,
        frame=td.PECFrame(length=3),
        direction="+",
    )
    sim = td.Simulation(
        center=[0, 0, 0],
        size=[1, 1, 1],
        grid_spec=td.GridSpec.auto(
            min_steps_per_wvl=10,
            wavelength=wvl_um,
        ),
        sources=[mode_source],
        run_time=1e-20,
        internal_absorbers=[
            td.InternalAbsorber(
                size=(0.4, 0.5, 0), direction="-", boundary_spec=td.ABCBoundary(permittivity=1)
            )
        ],
    )

    _ = sim._finalized

    # added frame will collide with projection monitor which requires uniform medium
    bad_sim = td.Simulation(
        center=[0, 0, 0],
        size=[1, 1, 1],
        grid_spec=td.GridSpec.auto(
            min_steps_per_wvl=10,
            wavelength=wvl_um,
        ),
        monitors=[
            td.FieldProjectionAngleMonitor(
                center=[0, 0.25, 0],
                size=[1, 0, 1],
                freqs=[freq0],
                name="n2f_angle",
                phi=[0],
                theta=[0],
                normal_dir="+",
            )
        ],
        sources=[mode_source],
        run_time=1e-20,
        internal_absorbers=[
            td.InternalAbsorber(
                size=(0.4, 0.5, 0), direction="-", boundary_spec=td.ABCBoundary(permittivity=1)
            )
        ],
    )
    with pytest.raises(td.exceptions.Tidy3dError):
        _ = bad_sim._validate_finalized()


def test_pec_frame_matches_mode_solver_pec_boundaries():
    """PEC frame tangential boundaries must match ModeSolver PEC boundary positions."""

    wvl_um = 1
    freq0 = td.C_0 / wvl_um
    mode_source = td.ModeSource(
        size=(0.6, 0.8, 0),
        source_time=td.GaussianPulse(freq0=freq0, fwidth=0.2 * freq0),
        mode_spec=td.ModeSpec(num_modes=1),
        frame=td.PECFrame(length=2),
        direction="+",
    )
    sim = td.Simulation(
        center=[0, 0, 0],
        size=[2, 2, 2],
        grid_spec=td.GridSpec.auto(min_steps_per_wvl=10, wavelength=wvl_um),
        sources=[mode_source],
        run_time=1e-20,
    )

    ms = ModeSolver(
        simulation=sim,
        plane=td.Box(center=mode_source.center, size=mode_source.size),
        freqs=[freq0],
        mode_spec=mode_source.mode_spec,
    )

    # Get PEC frame box and mode solver snapped mode domain
    frame_box, axis, _ = sim._pec_frame_box(mode_source)
    pec_box = ModeSolver._snapped_mode_domain(sim.grid, ms.plane, ms.normal_axis)

    _, tangential_axes = td.Box.pop_axis([0, 1, 2], axis)
    for ax in tangential_axes:
        assert frame_box.bounds[0][ax] == pec_box.bounds[0][ax], (
            f"PEC frame min on axis {ax} ({frame_box.bounds[0][ax]}) "
            f"does not match mode solver PEC ({pec_box.bounds[0][ax]})"
        )
        assert frame_box.bounds[1][ax] == pec_box.bounds[1][ax], (
            f"PEC frame max on axis {ax} ({frame_box.bounds[1][ax]}) "
            f"does not match mode solver PEC ({pec_box.bounds[1][ax]})"
        )
