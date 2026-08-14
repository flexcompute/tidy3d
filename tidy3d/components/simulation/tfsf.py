"""TFSF, fixed-angle, and dipole-emission source validation for ``Simulation``."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import autograd.numpy as np

from tidy3d.components.base import cached_property
from tidy3d.components.boundary import BlochBoundary, Periodic
from tidy3d.components.geometry.base import Box
from tidy3d.components.grid.grid_spec import UniformGrid
from tidy3d.components.medium import (
    AbstractCustomMedium,
    FullyAnisotropicMedium,
    LossyMetalMedium,
    PECMedium,
    PMCMedium,
)
from tidy3d.components.monitor import DipoleEmissionMonitor, TimeMonitor
from tidy3d.components.scene import Scene
from tidy3d.components.source.base import Source
from tidy3d.components.source.field import TFSF, FixedAngleSpec, PlaneWave
from tidy3d.components.source.time import CustomSourceTime, Pulse
from tidy3d.components.structure import Structure
from tidy3d.components.validators import is_close_to_glancing_angle, points_outside_bounds
from tidy3d.constants import C_0, GLANCING_CUTOFF, inf
from tidy3d.exceptions import (
    SetupError,
)
from tidy3d.log import log
from tidy3d.packaging import (
    disable_local_subpixel,
)

if TYPE_CHECKING:
    from typing import NoReturn

    from tidy3d.compat import Self
    from tidy3d.components.medium import AbstractMedium
    from tidy3d.components.source.utils import SourceType
    from tidy3d.components.types import ArrayFloat1D


def _raise_setup_error(message: str) -> NoReturn:
    """Raise a setup error from helper paths used outside post-init validation."""
    raise SetupError(message)


def _tfsf_boundaries(self: Any) -> Self:
    """Error if the boundary conditions are incompatible with TFSF sources, if any."""
    boundaries = self.boundary_spec.to_list
    sources = self.sources
    size = self.size
    center = self.center
    sim_medium = self.medium
    structures = self.structures
    sim_bounds = [
        [c - s / 2.0 for c, s in zip(center, size)],
        [c + s / 2.0 for c, s in zip(center, size)],
    ]
    for src_idx, source in enumerate(sources):
        if not isinstance(source, TFSF):
            continue

        norm_dir, tan_dirs = self.pop_axis([0, 1, 2], axis=source.injection_axis)
        src_bounds = source.bounds
        clipped_bounds = Box.bounds_intersection(src_bounds, sim_bounds)
        clipped_tan_sizes = [
            clipped_bounds[1][tan_dir] - clipped_bounds[0][tan_dir] for tan_dir in tan_dirs
        ]

        if not any(size > 0 for size in clipped_tan_sizes):
            self._raise_validation_error_at_loc(
                f"The TFSF source at index '{src_idx}' must have a nonzero in-domain "
                "tangential extent in at least one direction after intersecting with the "
                "simulation domain.",
                "sources",
                src_idx,
            )

        # make a dummy source that represents the injection surface to get the intersecting
        # medium, which is later used to test the Bloch vector for correctness
        temp_size = list(source.size)
        temp_size[source.injection_axis] = 0
        temp_src = Source(
            center=source.injection_plane_center,
            size=temp_size,
            source_time=source.source_time,
        )
        medium_set = Scene.intersecting_media(temp_src, structures)
        medium = medium_set.pop() if medium_set else sim_medium

        # the source shouldn't touch or cross any boundary in the direction of injection
        if (
            src_bounds[0][norm_dir] <= sim_bounds[0][norm_dir]
            or src_bounds[1][norm_dir] >= sim_bounds[1][norm_dir]
        ):
            self._raise_validation_error_at_loc(
                f"The TFSF source at index '{src_idx}' must not touch or cross the "
                f"simulation boundary along its injection axis, '{['x', 'y', 'z'][norm_dir]}'.",
                "sources",
                src_idx,
            )

        # Periodic / Bloch boundaries along the injection axis are
        # physically inconsistent with TFSF — the wave reaches the
        # boundary and gets re-injected, breaking the assumption
        # that the SF region is a pure scattered field.
        inj_boundary = boundaries[norm_dir]
        bad_inj = [bnd for bnd in inj_boundary if isinstance(bnd, BlochBoundary | Periodic)]
        if bad_inj:
            self._raise_validation_error_at_loc(
                f"The TFSF source at index '{src_idx}' cannot use 'BlochBoundary' or "
                f"'Periodic' on its injection axis '{['x', 'y', 'z'][norm_dir]}'; got "
                f"'{type(bad_inj[0]).__name__}'.",
                "sources",
                src_idx,
            )

        for tan_dir in tan_dirs:
            boundary = boundaries[tan_dir]

            # Fixed-angle TFSF forbids ``BlochBoundary`` and ``Periodic``
            # transverse boundaries (both imply an infinite-extent or
            # periodic structure, which contradicts the isolated-scatterer
            # model this path is designed for). The constant-in-plane-k
            # TFSF (the default ``FixedInPlaneKSpec``) is what to use for
            # periodic structures. 2D simulations: a transverse axis with
            # ``sim.size[axis] == 0`` is the out-of-plane axis,
            # conventionally Periodic and carrying no physical width — it
            # is exempt from this rule.
            if isinstance(source.angular_spec, FixedAngleSpec) and self.size[tan_dir] > 0:
                bad = [bnd for bnd in boundary if isinstance(bnd, BlochBoundary | Periodic)]
                if bad:
                    self._raise_validation_error_at_loc(
                        "Fixed-angle TFSF forbids 'BlochBoundary' and 'Periodic' transverse "
                        f"boundaries; got '{type(bad[0]).__name__}' on dimension "
                        f"'{['x', 'y', 'z'][tan_dir]}'. Fixed-angle TFSF models an isolated "
                        "scatterer — for periodic structures with Bloch boundaries, use "
                        "'FixedInPlaneKSpec' (angle exact only at the central frequency).",
                        "sources",
                        src_idx,
                    )

            # 2D simulations exempt the 0-size transverse axis from
            # the absorbing-BC rule (the conventional out-of-plane
            # ``Periodic`` axis carries no physical width), but the
            # wave must still have no k-component along that axis —
            # otherwise the ``Periodic`` BC + 0-size cell is
            # physically inconsistent. ``BlochBoundary`` on the
            # 0-width axis is already rejected upstream by
            # ``_check_zero_dim_domain`` (Bloch's vector definition
            # is incompatible with zero domain size), so no extra
            # rejection is needed here.
            if isinstance(source.angular_spec, FixedAngleSpec) and self.size[tan_dir] == 0:
                # ``Source._dir_vector`` is the wave's unit propagation
                # vector in lab frame; its ``tan_dir`` component is the
                # k-projection that must be ≈ 0 for the ``Periodic`` /
                # 0-size axis to be physically consistent. The 1e-12
                # tolerance is well below any user-meaningful angle (and
                # well above floating-point noise in (θ, φ)).
                k_proj = abs(float(source._dir_vector[tan_dir]))
                if k_proj > 1e-12:
                    self._raise_validation_error_at_loc(
                        "Fixed-angle TFSF in 2D requires the wave's "
                        f"k-vector to have no component along the 0-size "
                        f"axis '{['x', 'y', 'z'][tan_dir]}', but the "
                        f"current (angle_theta, angle_phi, injection_axis) "
                        f"give a projection of {k_proj:.3e}. Set angle_phi "
                        "so the in-plane component points along the "
                        "physical 2D plane (or use angle_theta=0 for "
                        "normal incidence).",
                        "sources",
                        src_idx,
                    )

            # crossing may be allowed for periodic or Bloch boundaries, but not others
            if (
                src_bounds[0][tan_dir] <= sim_bounds[0][tan_dir]
                or src_bounds[1][tan_dir] >= sim_bounds[1][tan_dir]
            ):
                # if the boundary is Bloch periodic, crossing is allowed, but check that the
                # Bloch vector has been correctly set, similar to the check for plane waves
                num_bloch = sum(isinstance(bnd, Periodic | BlochBoundary) for bnd in boundary)
                if num_bloch == 2:
                    self._check_bloch_vec(
                        source=source,
                        source_ind=src_idx,
                        bloch_vec=boundary[0].bloch_vec,
                        dim=tan_dir,
                        medium=medium,
                        domain_size=size[tan_dir],
                    )
                    continue

                # for any other boundary, the source must not cross the boundary
                self._raise_validation_error_at_loc(
                    f"The TFSF source at index '{src_idx}' must not touch or cross the "
                    f"simulation boundary in the '{['x', 'y', 'z'][tan_dir]}' direction, "
                    "unless that boundary is 'Periodic' or 'BlochBoundary'.",
                    "sources",
                    src_idx,
                )

    return self


def _warn_fixed_angle_tfsf_normal_incidence(self: Any) -> Self:
    """Warn if a fixed-angle TFSF is used at normal incidence (θ=0).
    At θ=0 the fixed-angle TFSF path adds setup and per-step cost
    without any physical benefit — the default ``FixedInPlaneKSpec``
    (Bloch TFSF) is exactly equivalent and faster."""
    for src_idx, source in enumerate(self.sources):
        if (
            isinstance(source, TFSF)
            and isinstance(source.angular_spec, FixedAngleSpec)
            and source.angle_theta == 0.0
        ):
            log.warning(
                f"TFSF source at index '{src_idx}' uses 'FixedAngleSpec' with "
                "angle_theta=0. At normal incidence the default "
                "'FixedInPlaneKSpec' (Bloch TFSF) is physically equivalent and "
                "runs faster. Consider switching unless you specifically need "
                "the fixed-angle path.",
                log_once=True,
            )
    return self


def _validate_fixed_angle_tfsf_angle_theta(self: Any) -> Self:
    """Fixed-angle TFSF source-amplitude normalization includes a
    ``1/sqrt(cos(angle_theta))`` factor that is singular at
    ``angle_theta = ±π/2`` and imaginary beyond, producing
    ``inf``/``NaN`` injections. Reject ``angle_theta`` within
    :data:`tidy3d.constants.GLANCING_CUTOFF` of any odd multiple of
    ``π/2``."""
    for src_idx, source in enumerate(self.sources):
        if not (isinstance(source, TFSF) and isinstance(source.angular_spec, FixedAngleSpec)):
            continue
        if is_close_to_glancing_angle(source.angle_theta, GLANCING_CUTOFF):
            cutoff_deg = float(np.rad2deg(GLANCING_CUTOFF))
            self._raise_validation_error_at_loc(
                "Fixed-angle TFSF requires the source's propagation angle to be more "
                f"than ~{cutoff_deg:.1f}° away from glancing (i.e. |angle_theta| ≤ "
                f"π/2 − {GLANCING_CUTOFF:g} rad); got "
                f"angle_theta = {source.angle_theta:.4f} rad.",
                "sources",
                src_idx,
            )
    return self


def _validate_fixed_angle_tfsf_source_time_type(self: Any) -> Self:
    """Fixed-angle TFSF needs a ``Pulse`` source time with an
    analytic ``amp_freq`` (uses ``fwidth`` and ``offset_time`` for
    the bandwidth and offset, and the analytic frequency spectrum
    for normalization). Reject other ``SourceTime`` subclasses,
    and explicitly reject ``CustomSourceTime`` (a ``Pulse``
    subclass but without an analytic ``amp_freq``).
    """
    for src_idx, source in enumerate(self.sources):
        if not (isinstance(source, TFSF) and isinstance(source.angular_spec, FixedAngleSpec)):
            continue
        if isinstance(source.source_time, CustomSourceTime):
            self._raise_validation_error_at_loc(
                "Fixed-angle TFSF does not support 'CustomSourceTime'; an analytic "
                "frequency-domain envelope is required. Use 'GaussianPulse' (or "
                "another analytic 'Pulse' subclass) instead.",
                "sources",
                src_idx,
            )
        if not isinstance(source.source_time, Pulse):
            self._raise_validation_error_at_loc(
                "Fixed-angle TFSF requires a 'Pulse' source time (e.g. "
                f"'GaussianPulse'); got '{source.source_time.type}'.",
                "sources",
                src_idx,
            )
    return self


def _validate_fixed_angle_tfsf_semi_infinite_injection_axis(self: Any) -> Self:
    """Fixed-angle TFSF assumes its top and bottom (injection-axis)
    faces sit in semi-infinite spaces along the injection axis:
    on each side, the region between the box face and the
    simulation edge must be a single medium. Reject otherwise.
    """
    # Include the simulation background as a virtual structure so
    # ``intersecting_media`` catches vacuum/structure mixtures.
    structure_bg = Structure(
        geometry=Box(size=self.size, center=self.center),
        medium=self.medium,
    )
    total_structures = [structure_bg, *list(self.structures or [])]
    for src_idx, source in enumerate(self.sources):
        if not (isinstance(source, TFSF) and isinstance(source.angular_spec, FixedAngleSpec)):
            continue
        axis = source.injection_axis
        sim_lo, sim_hi = self.bounds[0][axis], self.bounds[1][axis]
        box_lo, box_hi = source.bounds[0][axis], source.bounds[1][axis]
        for side_label, z_far, z_near in (
            ("-", sim_lo, box_lo),
            ("+", box_hi, sim_hi),
        ):
            # Probe a column at the source's transverse extent,
            # spanning from the box face out to the sim edge along
            # the injection axis. Skip if the box face touches the
            # sim edge (already caught by ``_tfsf_boundaries``).
            if z_near - z_far <= 0:
                continue
            probe_center = list(source.center)
            probe_center[axis] = 0.5 * (z_far + z_near)
            probe_size = list(source.size)
            probe_size[axis] = z_near - z_far
            probe = Box(center=tuple(probe_center), size=tuple(probe_size))
            # Best-effort check: ``Scene.intersecting_media`` on a
            # volumetric ``Box`` only recurses on its six surfaces,
            # so a finite inclusion fully enclosed inside the
            # probe (no surface contact) can slip through. A
            # genuinely volume-aware test on a setup with up to
            # ~10⁶ structures (e.g., a metalens) is too costly to
            # run at validation time.
            mediums = Scene.intersecting_media(probe, total_structures)
            if len(mediums) > 1:
                self._raise_validation_error_at_loc(
                    f"Fixed-angle TFSF source at index {src_idx} requires the region "
                    f"between its '{side_label}' injection-axis box face and the "
                    f"simulation edge along '{'xyz'[axis]}' to be a single medium "
                    f"(semi-infinite space); got {len(mediums)} distinct media. Either "
                    "extend the structures so they fill the full simulation extent "
                    "along the injection axis (e.g. ``size=td.inf``), or move them "
                    "fully inside the TFSF box.",
                    "sources",
                    src_idx,
                )
    return self


def _validate_fixed_angle_tfsf_source_time_localization(self: Any) -> Self:
    """Fixed-angle TFSF requires the source pulse to have decayed by
    the end of the simulation. Reject if ``|amp_time(run_time)| >
    1e-4 · peak``, where peak is taken over a dense sample of the run
    window (anchored at the pulse-center ``offset_time`` so a long
    ``run_time`` with a short pulse doesn't skip the pulse peak).
    Catches ``ContinuousWave`` (steady-state at ``run_time``) and
    ``CustomSourceTime`` with non-decaying ends.

    We do *not* check the source value at ``t = 0`` — a
    ``GaussianPulse(offset=N)`` has analytic value ``exp(-N²/2) ·
    peak`` at t=0, which is reproduced faithfully even when small
    but non-zero. Users who want a cleaner ramp-up should increase
    ``offset``.
    """
    EPS_REL = 1e-4
    for src_idx, source in enumerate(self.sources):
        if not (isinstance(source, TFSF) and isinstance(source.angular_spec, FixedAngleSpec)):
            continue
        st = source.source_time
        # A source time with unbounded support (no finite ``end_time``,
        # e.g. ``ContinuousWave``) can never satisfy the decay
        # requirement. Reject it here with the actionable, source-localized
        # error *before* evaluating ``self._run_time`` — for a
        # ``RunTimeSpec`` that evaluation would otherwise raise the generic
        # "could not compute source contributions" error first, making the
        # failure mode depend on how ``run_time`` is represented.
        if st.end_time() is None:
            self._raise_validation_error_at_loc(
                "Fixed-angle TFSF requires 'source_time' to have decayed by "
                f"the end of the simulation, but '{st.type}' has unbounded "
                "time support. Use a localized source (e.g. 'GaussianPulse').",
                "sources",
                src_idx,
            )
        # Use the evaluated run time so a ``RunTimeSpec`` (not a plain
        # float) is handled instead of raising ``TypeError`` here.
        run_time = self._run_time
        # Anchor the dense sample at `offset_time` so long-`run_time`
        # sims with a short pulse (run_time >> twidth) don't skip the
        # pulse peak entirely and report a spurious "peak ≈ 0".
        t_dense = np.unique(
            np.concatenate(
                [
                    np.linspace(0.0, run_time, 256),
                    np.array([float(st.offset_time)]),
                ]
            )
        )
        amps = np.abs(np.asarray(st.amp_time(t_dense)))
        peak = float(amps.max())
        if peak <= 0:
            continue
        a_end = float(np.abs(np.atleast_1d(np.asarray(st.amp_time(run_time)))[0]))
        if a_end / peak > EPS_REL:
            self._raise_validation_error_at_loc(
                "Fixed-angle TFSF requires 'source_time' to have decayed "
                "by the end of the simulation. Got |amp_time(run_time)|/peak = "
                f"{a_end / peak:.2e} > {EPS_REL:.0e}. Use a longer "
                "'run_time' so the pulse tail fits inside, or a more "
                "localized source_time (e.g. 'GaussianPulse' instead "
                "of 'ContinuousWave').",
                "sources",
                src_idx,
            )
    return self


def _warn_fixed_angle_tfsf_long_run_time(self: Any) -> Self:
    """Warn if a fixed-angle TFSF source is used with a long
    ``run_time`` / wide ``fwidth``. The fixed-angle TFSF path has
    cost that scales as **`run_time ** 2`**: at long ``run_time`` the
    source's per-step cost grows linearly with ``run_time``, on
    top of the linear growth in the number of time steps. For
    long sims this can put the simulation in a regime where the
    TFSF source is more expensive than the FDTD time-stepping.
    We warn the user so they can shorten ``run_time`` if their
    field decay allows, or narrow ``source_time.fwidth`` if the
    bandwidth is wider than needed."""
    # Heuristic threshold on the dimensionless product
    # `run_time · fwidth`. Empirically chosen so the warning
    # fires roughly when the fixed-angle TFSF cost becomes
    # comparable to the FDTD update cost.
    RUN_TIME_FWIDTH_WARN_THRESHOLD = 500.0
    for src_idx, source in enumerate(self.sources):
        if not (isinstance(source, TFSF) and isinstance(source.angular_spec, FixedAngleSpec)):
            continue
        # Use the evaluated run time so a ``RunTimeSpec`` (not a plain
        # float) is handled instead of raising ``TypeError`` here.
        run_time = self._run_time
        fwidth = float(source.source_time.fwidth)
        if run_time * fwidth > RUN_TIME_FWIDTH_WARN_THRESHOLD:
            log.warning(
                f"TFSF source at index '{src_idx}' uses 'FixedAngleSpec' with "
                f"'run_time' ({run_time:.2e} s) and 'source_time.fwidth' "
                f"({fwidth:.2e} Hz) in a regime where the fixed-angle TFSF "
                "cost scales as `run_time ** 2` and can become comparable to or "
                "larger than the FDTD time-stepping cost. Consider reducing "
                "'run_time' if the field decay allows, or narrowing "
                "'source_time.fwidth' if the bandwidth is wider than needed.",
                log_once=True,
            )
    return self


def _tfsf_with_symmetry(self: Any) -> Self:
    """Error if a TFSF source is applied with symmetry"""
    for source_ind, source in enumerate(self.sources):
        if isinstance(source, TFSF) and not all(sym == 0 for sym in self.symmetry):
            self._raise_validation_error_at_loc(
                "TFSF sources cannot be used with symmetries.", "sources", source_ind
            )
    return self


@staticmethod
def _get_periodic_fixed_angle_sources(
    sources: tuple[SourceType, ...],
) -> tuple[SourceType, ...]:
    """Periodic fixed-angle :class:`PlaneWave` sources.

    ``TFSF`` sources with ``FixedAngleSpec`` are intentionally
    excluded — only a fixed-angle :class:`PlaneWave` is a periodic
    fixed-angle source, and ``_check_fixed_angle_components``
    prohibits combining the two.
    """

    return [
        source
        for source in sources
        if isinstance(source, PlaneWave) and source._is_periodic_fixed_angle
    ]


def _check_fixed_angle_components(self: Any) -> Self:
    """Error if a fixed-angle plane wave is combined with other sources
    or fully anisotropic mediums or gain mediums."""

    fixed_angle_sources = self._get_periodic_fixed_angle_sources(self.sources)

    if len(fixed_angle_sources) > 0:
        # A fixed-angle PlaneWave must be the only source — no
        # other sources of any type.
        if len(self.sources) > 1:
            self._raise_validation_error_at_loc(
                "A fixed-angle plane wave source cannot be combined with other sources.",
                "sources",
            )

        structures = self.structures
        structures = structures or []
        medium_bg = self.medium
        mediums = [medium_bg] + [structure.medium for structure in structures]

        if any(med.is_fully_anisotropic for med in mediums):
            self._raise_validation_error_at_loc(
                "Fixed-angle plane wave sources cannot be used in the presence of 'FullyAnisotropicMedium'.",
                "sources",
            )

        if any(med.is_nonlinear for med in mediums):
            self._raise_validation_error_at_loc(
                "Fixed-angle plane wave sources cannot be used in the presence of nonlinear materials.",
                "sources",
            )

        if any(med.is_time_modulated for med in mediums):
            self._raise_validation_error_at_loc(
                "Fixed-angle plane wave sources cannot be used in the presence of time-modulated materials.",
                "sources",
            )

        if any(med.allow_gain for med in mediums):
            self._raise_validation_error_at_loc(
                "Fixed-angle plane wave sources cannot be used in the presence of gain materials.",
                "sources",
            )

        if any(isinstance(mnt, TimeMonitor) for mnt in self.monitors):
            self._raise_validation_error_at_loc(
                "Time monitors cannot be used in fixed-angle simulations.",
                "monitors",
            )

        if len(self.internal_absorbers) > 0:
            self._raise_validation_error_at_loc(
                "Fixed-angle plane wave sources cannot be used in the presence of internal absorbers.",
                "internal_absorbers",
            )

    return self


def _validate_dipole_emission_monitor_sources(self: Any) -> Self:
    """Error if dipole-emission monitors are not paired with the single TFSF source."""

    if not self.monitors:
        return self

    dipole_emission_monitors = tuple(
        (monitor_ind, monitor)
        for monitor_ind, monitor in enumerate(self.monitors)
        if isinstance(monitor, DipoleEmissionMonitor)
    )
    if not dipole_emission_monitors:
        return self

    if any(size == 0 for size in self.size):
        self._raise_validation_error_at_loc(
            "A simulation containing a DipoleEmissionMonitor must be three-dimensional. "
            "The radiation intensity is a per-solid-angle quantity, so 2D simulations "
            "(a zero-size dimension) are not supported.",
            "size",
        )

    if len(self.sources) != 1 or not isinstance(self.sources[0], TFSF):
        self._raise_validation_error_at_loc(
            "A simulation containing a DipoleEmissionMonitor must contain exactly one "
            "source, and that source must be a TFSF source.",
            "sources",
        )
    source = self.sources[0]

    if not isinstance(source.angular_spec, FixedAngleSpec):
        self._raise_validation_error_at_loc(
            "DipoleEmissionMonitor requires a TFSF source with FixedAngleSpec.",
            "sources",
            0,
            "angular_spec",
        )

    cos_theta = float(np.cos(source.angle_theta))
    if not np.isfinite(cos_theta) or cos_theta <= 0:
        self._raise_validation_error_at_loc(
            "DipoleEmissionMonitor requires a TFSF source with positive cos(angle_theta).",
            "sources",
            0,
            "angle_theta",
        )

    for monitor_ind, monitor in enumerate(self.monitors):
        if not isinstance(monitor, DipoleEmissionMonitor):
            continue

        self._dipole_emission_background_index(
            source,
            monitor.freqs,
            validation_loc=("monitors", monitor_ind),
        )

        points = np.asarray(monitor.points.values, dtype=float)
        bounds = np.asarray(source.bounds, dtype=float)
        strict_inequality = np.asarray([size != 0 for size in source.size], dtype=bool)
        outside = points_outside_bounds(points, bounds, strict_inequality)
        if np.any(outside):
            first_index = int(np.nonzero(outside)[0][0])
            num_outside = int(np.count_nonzero(outside))
            self._raise_validation_error_at_loc(
                f"Dipole-emission monitor '{monitor.name}' has {num_outside} point(s) "
                f"outside TFSF source '{source.name}'. The first outside point has index "
                f"{first_index} and coordinates {points[first_index].tolist()}.",
                "monitors",
                monitor_ind,
                "points",
            )

    return self


def _dipole_emission_tfsf_source(self: Any) -> TFSF:
    """Return the single TFSF source associated with a dipole-emission monitor."""
    if len(self.sources) != 1 or not isinstance(self.sources[0], TFSF):
        raise SetupError(
            "DipoleEmissionMonitor postprocessing requires a simulation with exactly "
            "one source, and that source must be a TFSF source."
        )
    return self.sources[0]


@staticmethod
def _dipole_emission_tfsf_injection_plane(source: TFSF) -> Box:
    """Planar TFSF injection face used as the dipole-emission collection side."""
    injection_plane_size = list(source.size)
    injection_plane_size[source.injection_axis] = 0.0
    return Box(center=tuple(source.injection_plane_center), size=tuple(injection_plane_size))


def _dipole_emission_tfsf_injection_medium(
    self: Any,
    source: TFSF,
    validation_loc: tuple[Any, ...] | None = None,
) -> AbstractMedium:
    """Return the single medium intersecting the TFSF injection face."""
    injection_plane = self._dipole_emission_tfsf_injection_plane(source)
    simulation_background = Structure(
        geometry=Box(size=self.size, center=self.center),
        medium=self.medium,
    )
    plane_media = Scene.intersecting_media(
        injection_plane, [simulation_background, *list(self.structures or [])]
    )
    if len(plane_media) != 1:
        message = (
            "DipoleEmissionMonitor requires a homogeneous medium on the TFSF injection "
            f"plane; found {len(plane_media)} media."
        )
        if validation_loc is not None:
            self._raise_validation_error_at_loc(message, *validation_loc)
        _raise_setup_error(message)
    return next(iter(plane_media))


def _dipole_emission_background_index(
    self: Any,
    source: TFSF,
    freqs: ArrayFloat1D,
    validation_loc: tuple[Any, ...] | None = None,
) -> float:
    """Validate and return the real nondispersive index on the TFSF injection side."""
    medium = self._dipole_emission_tfsf_injection_medium(source, validation_loc)
    background_n = np.asarray(medium.background_index_from_freqs(freqs), dtype=complex)
    if not np.allclose(background_n.imag, 0.0):
        message = "DipoleEmissionMonitor requires real TFSF injection-side refractive index."
        if validation_loc is not None:
            self._raise_validation_error_at_loc(message, *validation_loc)
        _raise_setup_error(message)
    if not np.allclose(background_n.real, background_n.real[0], rtol=1e-12, atol=0.0):
        message = (
            "DipoleEmissionMonitor requires nondispersive TFSF injection-side refractive index."
        )
        if validation_loc is not None:
            self._raise_validation_error_at_loc(message, *validation_loc)
        _raise_setup_error(message)
    return float(background_n.real[0])


def _validate_tfsf_has_grid_cells(self: Any) -> None:
    """Each TFSF source must contain at least one grid center on every
    axis. Fixed-angle TFSF additionally needs at least two cells along
    the injection axis inside the simulation's physical domain."""
    for source_ind, source in enumerate(self.sources):
        if not isinstance(source, TFSF):
            continue
        centers = self.grid.centers.to_list
        tfsf_bounds = source.bounds
        sim_bounds = self.bounds
        for ind in range(3):
            n_in = sum(
                1 for center in centers[ind] if tfsf_bounds[0][ind] <= center <= tfsf_bounds[1][ind]
            )
            if n_in == 0:
                self._raise_validation_error_at_loc(
                    f"TFSF source at index {source_ind} has no grid cells along the "
                    f"'{'xyz'[ind]}' axis within its box. The source size or center is "
                    f"too small relative to the grid spacing, or the box falls outside "
                    f"the simulation domain.",
                    "sources",
                    source_ind,
                )
        if isinstance(source.angular_spec, FixedAngleSpec):
            inj = source.injection_axis
            n_inj_phys = sum(
                1 for c in centers[inj] if sim_bounds[0][inj] <= c <= sim_bounds[1][inj]
            )
            if n_inj_phys < 2:
                self._raise_validation_error_at_loc(
                    f"Fixed-angle TFSF source at index {source_ind} needs at least 2 "
                    f"grid cells along its injection axis '{'xyz'[inj]}' inside the "
                    f"simulation's physical domain (got {n_inj_phys}). Increase the "
                    f"physical-domain extent along that axis, or refine the grid.",
                    "sources",
                    source_ind,
                )


def _validate_tfsf_nonuniform_grid(self: Any) -> None:
    """Warn (or error) if the grid is nonuniform along the directions tangential to the
    injection plane, inside the TFSF box. A fixed-angle TFSF source requires a uniform
    transverse grid and errors out; other TFSF sources only see degraded incident-field
    cancellation, so we warn.
    """
    if not any(isinstance(source, TFSF) for source in self.sources):
        return

    with log as consolidated_logger:
        for source_ind, source in enumerate(self.sources):
            if not isinstance(source, TFSF):
                continue

            fixed_angle = isinstance(source.angular_spec, FixedAngleSpec)
            centers = self.grid.centers.to_list
            sizes = self.grid.sizes.to_list
            tfsf_bounds = source.bounds
            _, plane_inds = source.pop_axis([0, 1, 2], axis=source.injection_axis)
            grid_list = [self.grid_spec.grid_x, self.grid_spec.grid_y, self.grid_spec.grid_z]
            for ind in plane_inds:
                grid_type = grid_list[ind]
                if isinstance(grid_type, UniformGrid):
                    continue

                sizes_in_tfsf = [
                    size
                    for size, center in zip(sizes[ind], centers[ind])
                    if tfsf_bounds[0][ind] <= center <= tfsf_bounds[1][ind]
                ]

                # check if all the grid sizes are sufficiently unequal
                if not np.all(np.isclose(sizes_in_tfsf, sizes_in_tfsf[0])):
                    if fixed_angle:
                        self._raise_validation_error_at_loc(
                            f"Fixed-angle TFSF requires a uniform transverse grid inside the "
                            f"TFSF box, but the grid is nonuniform along the '{'xyz'[ind]}' "
                            f"axis within the source region. Add a 'MeshOverrideStructure' "
                            f"covering the TFSF box with a uniform 'dl' on the non-injection "
                            f"axes to force uniform spacing, or remove the non-uniformity "
                            f"from the structures intersecting the source.",
                            "sources",
                            source_ind,
                        )
                    else:
                        consolidated_logger.warning(
                            f"The grid is nonuniform along the '{'xyz'[ind]}' axis, which may lead "
                            "to sub-optimal cancellation of the incident field in the "
                            "scattered-field region for the total-field scattered-field (TFSF) "
                            f"source '{source.name}'. For best results, we recommended ensuring a "
                            "uniform grid in both directions tangential to the TFSF injection "
                            f"axis, '{'xyz'[source.injection_axis]}'.",
                            custom_loc=["sources", source_ind],
                        )


def _aux_tfsf_source(self: Any, source: TFSF) -> PlaneWave:
    """Create the auxiliary plane wave source for a give TFSF source."""
    # center and size of the plane wave source
    source_size = [inf] * 3
    source_size[source.injection_axis] = 0
    source_center = list(source.injection_plane_center)

    # since we need to access values of the aux self at dual grid locations below the actual
    # injection plane, we need to place the aux sim's source at least one full cell below the
    # location of the injection plane; for good measure, we'll offset the source by two cells
    src_grid = self.discretize(source, extend=False)
    src_grid_sizes = src_grid.sizes.to_list
    if source.direction == "+":
        offset = -sum(src_grid_sizes[source.injection_axis][0:2])
    else:
        offset = sum(src_grid_sizes[source.injection_axis][-1:-3:-1])
    source_center[source.injection_axis] += offset

    # Make sure that the new source center is within the simulation bounds
    sim_axis_bounds = [self.bounds[i][source.injection_axis] for i in range(2)]
    if (
        source_center[source.injection_axis] < sim_axis_bounds[0]
        or source_center[source.injection_axis] > sim_axis_bounds[1]
    ):
        raise SetupError(
            "The TFSF source is too close to the simulation domain boundary along the "
            "injection axis. Slightly increase the simulation domain size along that "
            "dimension, or decrease the source size."
        )

    # Pre-compensate the source-time so the unit-amplitude
    # reference lands at the injection plane (the box face), not
    # at the aux source plane that sits ``|offset|`` along the
    # propagation direction. For lossless source-side media this
    # is purely a phase shift; for lossy media it also pre-
    # amplifies by ``exp(+Im(kz)·|offset|)`` to undo the decay
    # over ``|offset|``. The medium at the injection plane is
    # queried stacking-aware (a structure overlapping the source
    # plane changes the local ``n``); fall back to ``self.medium``
    # if multiple media are visible there.
    source_time = source.source_time
    injection_plane_size = list(source.size)
    injection_plane_size[source.injection_axis] = 0.0
    injection_plane_probe = Box(
        center=tuple(source.injection_plane_center),
        size=tuple(injection_plane_size),
    )
    injection_bg = Structure(geometry=Box(size=self.size, center=self.center), medium=self.medium)
    plane_mediums = Scene.intersecting_media(
        injection_plane_probe, [injection_bg, *list(self.structures or [])]
    )
    injection_medium = next(iter(plane_mediums)) if len(plane_mediums) == 1 else self.medium
    try:
        f0 = float(source_time._freq0)
        n_complex = complex(injection_medium.background_index_from_freqs([f0])[0])
    except (AttributeError, NotImplementedError):
        n_complex = None
    if n_complex is not None:
        kz_continuum_at_f0 = (
            (2.0 * np.pi * f0 / C_0) * n_complex * float(np.cos(source.angle_theta))
        )
        compensation = complex(np.exp(-1j * kz_continuum_at_f0 * abs(offset)))
        amp_factor = float(np.abs(compensation))
        phase_shift = float(np.angle(compensation))
        if not (amp_factor == 1.0 and phase_shift == 0.0):
            source_time = source_time.updated_copy(
                amplitude=amp_factor * source_time.amplitude,
                phase=source_time.phase + phase_shift,
            )

    # Note: broadband injection for TFSF not currently supported.
    return PlaneWave(
        size=source_size,
        center=source_center,
        source_time=source_time,
        angle_theta=source.angle_theta,
        angle_phi=source.angle_phi,
        pol_angle=source.pol_angle,
        direction=source.direction,
        num_freqs=source.num_freqs,
        use_colocated_integration=source.use_colocated_integration,
    )


def _validate_tfsf_aux_sources(self: Any) -> None:
    """Validate that PlaneWave sources auxiliary to TFSF sources can be successfully created."""
    for source_ind, source in enumerate(self.sources):
        if isinstance(source, TFSF):
            _ = self._call_with_validation_loc(
                ["sources", source_ind], self._aux_tfsf_source, source=source
            )


@cached_property
def aux_fields(self: Any) -> list[str]:
    """All aux fields available in the simulation."""
    fields = []
    for medium in self.scene.mediums:
        if medium.nonlinear_spec is not None:
            fields += medium.nonlinear_spec.aux_fields
    return fields


@disable_local_subpixel
def _validate_tfsf_structure_intersections(self: Any) -> None:
    """Error if the 4 sidewalls of a TFSF box don't all intersect the same structures.
    This validator may need to compute permittivities on the grid, so it is called
    pre-upload rather than at the time of definition. Also errors if any side wall
    intersects with a custom medium or a fully anisotropic media.
    """
    for source_idx, source in enumerate(self.sources):
        if not isinstance(source, TFSF):
            continue
        # get all TFSF surfaces
        tfsf_surfaces = Source.surfaces(
            center=source.center, size=source.size, source_time=source.source_time
        )
        sidewall_surfaces = []
        sidewall_structs = []
        # get the structures that intersect each sidewall
        for surface in tfsf_surfaces:
            # ignore the sidewall surface if it falls outside the simulation domain
            if not self.intersects(surface):
                continue

            if surface.name[-2] != "xyz"[source.injection_axis]:
                sidewall_surfaces.append(surface)
                intersecting_structs = Scene.intersecting_structures(
                    test_object=surface, structures=self.structures
                )

                if any(
                    isinstance(struct.medium, AbstractCustomMedium | FullyAnisotropicMedium)
                    for struct in intersecting_structs
                ):
                    raise SetupError(
                        f"The surfaces of TFSF source '{source.name}' must not intersect any "
                        "structures containing a 'CustomMedium' or a 'FullyAnisotropicMedium'."
                    )

                # Surface-BC media (``LossyMetalMedium``, ``PECMedium``, ``PMCMedium``)
                # are only rejected at sidewalls of a fixed-angle TFSF source; the
                # constant-in-plane-k TFSF supports them. Move the structure fully
                # inside the box, or switch the source to ``FixedInPlaneKSpec``.
                if isinstance(source.angular_spec, FixedAngleSpec) and any(
                    isinstance(struct.medium, PECMedium | PMCMedium)
                    or (
                        isinstance(struct.medium, LossyMetalMedium) and not struct.medium.penetrable
                    )
                    for struct in intersecting_structs
                ):
                    self._raise_validation_error_at_loc(
                        f"Fixed-angle TFSF source '{source.name}' cannot have its sidewalls "
                        "intersect a 'LossyMetalMedium', 'PECMedium', or 'PMCMedium'. Move the "
                        "structure fully inside the TFSF box, or use 'FixedInPlaneKSpec' "
                        "instead.",
                        "sources",
                        source_idx,
                    )

                # if no structures intersect, just add a phantom associated with the simulation
                # background, to prevent false positives below
                if not intersecting_structs:
                    sidewall_structs.append(
                        [
                            Structure(
                                geometry=Box(center=self.center, size=self.size),
                                medium=self.medium,
                            )
                        ]
                    )
                else:
                    sidewall_structs.append(intersecting_structs)

        # let the first wall be a reference, and compare the rest of them to the structures
        # intersected by that reference wall
        if len(sidewall_structs) > 1:
            ref_structs = sidewall_structs[0]
            test_structs = sidewall_structs[1:]
            if all(structs == ref_structs for structs in test_structs):
                continue

            # if the == test doesn't pass, that doesn't mean the materials are necessarily
            # different, because it's possible that the sidewalls encounter different
            # `Structure` objects but with an identical material profile, which is still
            # a valid setup; in this case, compute the epsilon profile on the grid for each
            # side wall - the profiles must be the same along the injection axis, so we take
            # a single "stripe" of epsilon as the reference and subtract it from all other
            # stripes, which should result in zero if all the epsilon profiles are the same
            freq0 = source.source_time._freq0
            _, plane_axs = source.pop_axis("xyz", axis=source.injection_axis)
            ref_eps = self.epsilon(box=sidewall_surfaces[0], coord_key="centers", freq=freq0)
            kwargs = {plane_axs[0]: 0, plane_axs[1]: 0}
            ref_eps = ref_eps.isel(**kwargs)
            for surface in sidewall_surfaces:
                test_eps = self.epsilon(box=surface, coord_key="centers", freq=freq0) - ref_eps
                if not np.allclose(test_eps.to_numpy(), 0):
                    raise SetupError(
                        f"All sidewalls of the TFSF source '{source.name}' must intersect "
                        "the same media along the injection axis "
                        f" '{'xyz'[source.injection_axis]}'."
                    )


@cached_property
def _fixed_angle_sources(self: Any) -> tuple[SourceType, ...]:
    """List of plane wave sources with ``FixedAngleSpec``."""
    return self._get_periodic_fixed_angle_sources(self.sources)


@cached_property
def _is_periodic_fixed_angle(self: Any) -> bool:
    """Whether the simulation contains a periodic fixed-angle source —
    i.e. a fixed-angle :class:`PlaneWave` with non-zero ``angle_theta``."""
    return len(self._fixed_angle_sources) > 0
