"""Tests frames around sources and absorbers."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

import tidy3d as td
from tidy3d.components.mode.mode_solver import ModeSolver

from ..utils import assert_single_value_error_loc


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
        size=[2, 2, 2],
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
        size=[2, 2, 2],
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


def _sim_with_mode_source_pec_frame(
    *,
    boundary: td.AbsorberSpec,
    source_center: tuple[float, float, float] = (0, 0, 0),
    source_size: tuple[float, float, float] = (1, 1, 0),
    frame_length: int = 5,
) -> td.Simulation:
    """Build and validate a simulation with a single mode source carrying a PEC frame.

    The default source has tangential extent inside the inner clipping margin so that
    the only PML side that can collide with the frame is the injection axis.
    """
    freq0 = td.C_0 / 1.0
    return td.Simulation(
        size=(4, 4, 4),
        grid_spec=td.GridSpec.uniform(dl=0.1),
        boundary_spec=td.BoundarySpec.all_sides(boundary=boundary),
        run_time=1e-12,
        sources=[
            td.ModeSource(
                center=source_center,
                size=source_size,
                direction="+",
                source_time=td.GaussianPulse(freq0=freq0, fwidth=0.2 * freq0),
                frame=td.PECFrame(length=frame_length),
            )
        ],
    )


def _sim_with_internal_absorber_frame(
    *,
    boundary: td.AbsorberSpec,
    absorber_center: tuple[float, float, float] = (0, 0, 0),
    absorber_size: tuple[float, float, float] = (0.4, 0.4, 0),
) -> td.Simulation:
    """Build a simulation with a single internal absorber that auto-adds a PEC frame."""
    return td.Simulation(
        size=(4, 4, 4),
        grid_spec=td.GridSpec.uniform(dl=0.1),
        boundary_spec=td.BoundarySpec.all_sides(boundary=boundary),
        run_time=1e-12,
        internal_absorbers=[
            td.InternalAbsorber(
                center=absorber_center,
                size=absorber_size,
                direction="+",
                boundary_spec=td.ABCBoundary(permittivity=1),
            )
        ],
    )


_BOUNDARY_CLASSES = pytest.mark.parametrize(
    "boundary_cls",
    [td.PML, td.StablePML, td.Absorber],
    ids=["pml", "stable_pml", "absorber"],
)


@_BOUNDARY_CLASSES
@pytest.mark.parametrize("side", [-1.0, 1.0], ids=["minus", "plus"])
def test_pec_frame_pml_extrusion_overlap_errors(boundary_cls, side):
    """PEC frame reaching into the PML extrusion clipping region raises at the source loc.

    Parametrized over the offending boundary side so both the ``side == 0`` (``-``) and
    ``side == 1`` (``+``) branches of ``_pml_extrusion_clipping_bound_ind()`` are exercised.
    """
    with pytest.raises(ValidationError) as excinfo:
        _sim_with_mode_source_pec_frame(
            boundary=boundary_cls(num_layers=12, extrude_structures=True),
            source_center=(0, 0, side * 1.9),
        )
    assert_single_value_error_loc(
        excinfo,
        ("sources", 0),
        message_contains="extrusion region",
    )


@_BOUNDARY_CLASSES
def test_pec_frame_pml_extrusion_overlap_on_transverse_axis_errors(boundary_cls):
    """A frame with injection-axis clearance still errors when its tangential bbox
    overlaps a transverse PML's extrusion region (validator must check all axes,
    not just the injection axis)."""
    with pytest.raises(ValidationError) as excinfo:
        _sim_with_mode_source_pec_frame(
            boundary=boundary_cls(num_layers=12, extrude_structures=True),
            source_center=(-1.4, 0, 0),
            frame_length=2,
        )
    assert_single_value_error_loc(
        excinfo,
        ("sources", 0),
        message_contains="extrusion region",
    )


@_BOUNDARY_CLASSES
def test_pec_frame_pml_extrusion_ok_when_extrude_disabled(boundary_cls):
    """Same geometry is accepted when 'extrude_structures' is off on the offending side."""
    _sim_with_mode_source_pec_frame(
        boundary=boundary_cls(num_layers=12, extrude_structures=False),
        source_center=(0, 0, -1.9),
    )


@_BOUNDARY_CLASSES
def test_pec_frame_pml_extrusion_ok_with_clearance(boundary_cls):
    """Frame well clear of the PML + clipping margin on every axis validates cleanly."""
    _sim_with_mode_source_pec_frame(
        boundary=boundary_cls(num_layers=12, extrude_structures=True),
    )


@_BOUNDARY_CLASSES
def test_pec_frame_pml_extrusion_internal_absorber_overlap_errors(boundary_cls):
    """Internal-absorber frame reaching into the PML extrusion region raises at its loc."""
    with pytest.raises(ValidationError) as excinfo:
        _sim_with_internal_absorber_frame(
            boundary=boundary_cls(num_layers=12, extrude_structures=True),
            absorber_center=(0, 0, -1.7),
        )
    assert_single_value_error_loc(
        excinfo,
        ("internal_absorbers", 0),
        message_contains="extrusion region",
    )


@_BOUNDARY_CLASSES
def test_pec_frame_pml_extrusion_internal_absorber_ok_with_clearance(boundary_cls):
    """Internal-absorber frame well clear of the PML + clipping margin validates cleanly."""
    _sim_with_internal_absorber_frame(
        boundary=boundary_cls(num_layers=12, extrude_structures=True),
    )


def test_pec_frame_pml_extrusion_caught_at_pre_upload_for_validate_false_sim():
    """The PEC-frame / PML-extrusion check still fires for sims constructed with
    ``updated_copy(..., validate=False)`` once they reach upload-time validation.

    The smatrix WavePort / TerminalComponentModeler path adds per-port sources to
    ``base_sim`` via ``updated_copy(sources=..., validate=False, deep=False)`` and
    relies on ``Simulation.validate_pre_upload`` to catch geometry errors at upload.
    ``validate_pre_upload`` does not call ``_validate_scene`` directly, but routes
    through ``_validate_finalized`` -> ``_finalized.updated_copy(...)`` (default
    ``validate=True``), which re-runs the post-init validators on the finalized
    sim. This regression guards that transitive coverage so a future change that
    silences the finalized-sim re-validation can't re-open the bypass.
    """
    freq0 = td.C_0 / 1.0
    base_sim = td.Simulation(
        size=(4, 4, 4),
        grid_spec=td.GridSpec.uniform(dl=0.1),
        boundary_spec=td.BoundarySpec.all_sides(
            boundary=td.StablePML(num_layers=12, extrude_structures=True)
        ),
        run_time=1e-12,
    )
    bad_source = td.ModeSource(
        center=(0, 0, -1.9),
        size=(1, 1, 0),
        direction="+",
        source_time=td.GaussianPulse(freq0=freq0, fwidth=0.2 * freq0),
        frame=td.PECFrame(length=5),
    )
    augmented = base_sim.updated_copy(sources=[bad_source], validate=False, deep=False)
    with pytest.raises(td.exceptions.Tidy3dError, match="extrusion region"):
        augmented.validate_pre_upload(source_required=False)
