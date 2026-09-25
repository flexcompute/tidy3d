"""Time-domain waveform synthesis for FieldData-derived adjoint current sources.

This module compresses the per-frequency ``CustomCurrentSource`` objects generated
from ``FieldData`` VJPs into a *single* adjoint simulation. The spatial reduction is
exactly the field-source PCA decomposition (per-batch spatial modes sharing one set
of per-frequency coefficients); this module only replaces how each retained mode
is realized in time. Instead of one broadband simulation per mode, every mode
receives a synthesized ``CustomSourceTime`` waveform whose discrete-time spectrum
matches its complex coefficient at every adjoint frequency -- and zero at every
other adjoint frequency present in the simulation -- to within
``config.adjoint.field_source_temporal_spectrum_rtol``, measured against the largest
target in that waveform. By linearity, the raw
(unnormalized) adjoint fields of the combined simulation then equal the
per-frequency adjoint fields of the standard frequency-grouped pipeline at every
target frequency simultaneously.

The synthesized waveforms are windowed multi-tone signals: one smooth window shared
by all sources, one tone per adjoint frequency, with complex tone amplitudes solving
a small Gram system built from the *actual* ``CustomSourceTime.spectrum`` values so
that spectral leakage between nearby tones is corrected exactly. This is the
closed-form equivalent of a linearly constrained quadratic program restricted to the
smooth multi-tone subspace.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from tidy3d.components.autograd.field_source_pca import FieldSourcePCAProcessor
from tidy3d.components.data.data_array import TimeDataArray
from tidy3d.components.data.dataset import TimeDataset
from tidy3d.components.source.time import CustomSourceTime
from tidy3d.components.source.utils import SourceType
from tidy3d.log import log

if TYPE_CHECKING:
    from tidy3d.components.autograd.field_source_pca import BatchDecomposition
    from tidy3d.components.simulation import Simulation

# Maximum number of Newton-style refinements of the tone amplitudes against the
# actual ``CustomSourceTime.spectrum`` of the combined waveform.
MAX_SPECTRUM_REFINEMENTS = 8

# Fraction of the pulse duration spent in the smooth turn-on and turn-off ramps of
# the Tukey window. A flatter window narrows the tone lobes, improving nearby-tone
# separability at a fixed pulse duration.
WINDOW_RAMP_FRACTION = 0.25

# Maximum peak time-domain amplitude of a synthesized waveform relative to a single
# tone carrying the largest spectral target. Larger solutions rely on cancellation
# between tones and lose accuracy to solver dynamic range, so the pulse duration is
# escalated instead.
MAX_TIME_AMPLIFICATION = 100.0

# Multiplicative step between candidate pulse durations when the synthesis at the
# current duration fails its accuracy or amplification gates.
DURATION_ESCALATION_FACTOR = 1.6

# A rung within this fraction of the budget counts as being at the budget: escalating to a
# duration a fraction of a percent longer costs another full synthesis for no real gain.
ESCALATION_TOLERANCE = 0.999

# Absolute floor on the relative singular values retained in the truncated
# tone-Gram solves, guarding numerically meaningless directions regardless of the
# requested spectrum tolerance.
GRAM_RCOND = 1e-12

# Weight of the zero-DC constraint relative to the spectral targets. A synthesized
# current with nonzero time integral deposits static charge that FDTD cannot relax,
# so its DC content is driven this factor further below the spectrum tolerance.
DC_CONSTRAINT_WEIGHT = 1e4

# every counted entry is one complex128 value, which is what lets the entry budget be
# reported as the memory it actually stands for
BYTES_PER_ENTRY = 16


@dataclass(frozen=True)
class FieldSourceTemporalInfo:
    """Synthesized time-domain adjoint source data for one combined adjoint simulation.

    ``sources`` holds every synthesized ``CustomCurrentSource`` (all spatial profiles
    of all compatible supports) meant to run together in a single adjoint simulation.
    ``post_norm`` has unit modulus at each covered frequency: the synthesized
    waveforms reproduce the target spectra up to a common time shift used to center
    the pulses inside their causal window, which ``post_norm`` compensates exactly.
    ``run_time`` covers the synthesized pulse duration plus a field-decay margin equal
    to the duration the forward simulation actually ran.
    """

    sources: tuple[SourceType, ...]
    post_norm: xr.DataArray
    run_time: float


@dataclass(frozen=True)
class TemporalPlan:
    """An accepted synthesis plan, handed from the planning stage to the solver.

    A plan exists only once every gate that can reject it has passed, so holding one
    means it is worth solving. It carries what the solver and the assembly step need
    from the decomposition -- the retained modes, their target spectra, the durations
    worth attempting, and the output counts -- so neither has to re-derive them.
    """

    decompositions: tuple[BatchDecomposition, ...]
    freqs_all: np.ndarray
    targets: np.ndarray
    duration_candidates: tuple[float, ...]
    num_waveforms: int
    total_sources: int


@dataclass(frozen=True)
class FieldSourceTemporalProcessor:
    """Plan and synthesize time-domain compressed adjoint current sources.

    The processor consumes ``CustomCurrentSource`` objects generated from ``FieldData``
    VJPs, reuses the field-source PCA partitioning and joint spatial decomposition
    unchanged, and replaces the per-frequency source times with synthesized
    multi-tone ``CustomSourceTime`` waveforms so all retained modes can be injected
    simultaneously in one adjoint simulation.
    """

    simulation: Simulation
    forward_duration: float | None = None

    @staticmethod
    def _fallback_sources(decompositions: list[BatchDecomposition]) -> list[SourceType]:
        """Return the caller's own sources behind a list of decompositions.

        The planning sources of a series merge component-split inputs into one source
        per frequency, so handing those to the standard grouping path would group and
        post-normalize them differently than the sources the caller supplied. Rejected
        decompositions therefore fall back as the original, unmerged sources.
        """

        return [
            source
            for decomposition in decompositions
            for series, _modes in decomposition.series_modes
            for source in series.original_sources
        ]

    @staticmethod
    def _warn_source_budget_exceeded(
        decompositions: list[BatchDecomposition], total_sources: int, max_sources: int
    ) -> None:
        """Report a plan abandoned once its source count passed the configured budget.

        The count is what had accumulated when the budget was passed, not the total the
        untouched batches would have reached, because those are deliberately never
        decomposed once the plan is known to be over.
        """

        num_waveforms = sum(decomposition.coefficients.shape[0] for decomposition in decompositions)
        num_supports = sum(len(decomposition.series_modes) for decomposition in decompositions)
        log.warning(
            "Skipping temporal adjoint source synthesis: the combined simulation would "
            f"inject at least {total_sources} sources ({num_waveforms} synthesized waveforms "
            f"across {num_supports} supports), exceeding "
            f"'config.adjoint.field_source_temporal_max_sources'={max_sources}. These sources "
            "fall back to the other reduction paths, so the objective still differentiates "
            "correctly but uses more adjoint simulations. Raise "
            "'config.adjoint.field_source_temporal_max_sources' to allow this plan.",
            log_once=True,
        )

    def _effective_forward_duration(self) -> float:
        """Return the duration the forward simulation actually ran.

        Uses the measured duration recorded by the forward run's field-decay data
        (shutoff can end a run well before its nominal ``run_time``), clamped to that
        nominal value.

        The nominal-only branch is not reachable through the adjoint pipeline: a run
        without a readable decay record fails verification upstream and the synthesis is
        skipped entirely rather than falling back to the nominal duration. It remains
        only so the processor can be constructed directly, as the unit tests do.
        """

        run_time_nominal = float(self.simulation._run_time)
        if self.forward_duration is None:
            return run_time_nominal
        return min(self.forward_duration, run_time_nominal)

    def _pulse_duration_candidates(
        self,
        *,
        pulse_scale: float,
        max_run_time_ratio: float,
    ) -> list[float]:
        """Return escalating candidate pulse durations for the adjoint synthesis.

        Every duration is measured against the duration the forward simulation actually
        ran. The adjoint decays in the same structure, so it is given that same duration
        as a ring-down margin after its pulse ends, making the total adjoint run time
        ``pulse + measured``. Bounding the total at ``max_run_time_ratio`` times the
        measured duration therefore caps the pulse itself at ``max_run_time_ratio - 1``
        times it.

        The exact time-domain adjoint of a truncated-DFT objective needs no more than the
        forward recording window, so the ladder starts at ``pulse_scale`` times the
        measured duration and escalates by ``DURATION_ESCALATION_FACTOR`` per rung up to
        that cap. An empty list means no pulse fits inside the budget at all, and the
        caller falls back rather than exceeding it.
        """

        measured = self._effective_forward_duration()
        cap = (max_run_time_ratio - 1.0) * measured
        floor = 16.0 * self.simulation.dt
        if cap < floor:
            # the budget leaves no room for a pulse the solver could even represent, which
            # includes a forward run so short that the floor alone would overrun the bound
            return []

        # clamp into [floor, cap]: the floor keeps the ladder advancing (a zero-length
        # start could never grow by multiplication) and the cap keeps the first rung
        # inside the budget even when the requested scale overshoots it
        start = min(max(pulse_scale * measured, floor), cap)
        # a start inside the escalation tolerance of the cap becomes the cap: escalation
        # stops there, so leaving it just short would try one rung and never the budget
        # itself, while adding the cap as its own rung would repeat the same synthesis
        if start >= ESCALATION_TOLERANCE * cap:
            start = cap
        candidates = [start]
        while candidates[-1] < ESCALATION_TOLERANCE * cap:
            candidates.append(min(candidates[-1] * DURATION_ESCALATION_FACTOR, cap))
        return candidates

    def _envelope_sample_count(self, pulse_duration: float) -> int:
        """Return the number of time samples one synthesized envelope will hold."""

        return max(int(np.round(pulse_duration / float(self.simulation.dt))), 8) + 1

    @staticmethod
    def _frequency_entry_count(num_freqs: int) -> int:
        """Estimate the dense entries the synthesis holds that scale with frequency count.

        Two phases allocate quadratically in the frequency count and not at all with the
        pulse duration. The group-delay scan of :meth:`_dominant_group_delay` is sized
        ``(8 (frequencies + 1) + 1) x frequencies`` and is released before the solve.
        The solve then holds the constraint matrix, both singular-vector matrices from
        its decomposition, and the truncated pseudoinverse -- four ``(frequencies + 1)^2``
        blocks -- plus the decomposition's own workspace.

        This is an estimate, not a bound: the workspace a dense complex decomposition
        requests is not knowable from here, so it is allowed for with a term of the same
        order. Measured peak usage tracks it within tens of percent across frequency
        counts from a few hundred to a thousand.
        """

        scan_entries = (8 * (num_freqs + 1) + 1) * num_freqs
        # constraint matrix, both singular-vector matrices, the pseudoinverse, and an
        # allowance of the same order for the decomposition's workspace
        solve_entries = 6 * (num_freqs + 1) ** 2
        return scan_entries + solve_entries

    def _duration_entry_count(self, duration: float, num_freqs: int, num_waveforms: int) -> int:
        """Return the dense entries one candidate duration allocates and retains in total.

        Three contributions: the tone-envelope matrix, the frequency-quadratic work
        arrays, and one retained envelope per synthesized waveform. The last is the
        plan's output rather than scratch, so it outlives the synthesis -- and with few
        frequencies and many retained modes it is the largest of the three.
        """

        num_samples = self._envelope_sample_count(duration)
        tone_entries = num_samples * (num_freqs + 1)
        retained_envelopes = num_waveforms * num_samples
        return tone_entries + self._frequency_entry_count(num_freqs) + retained_envelopes

    def _affordable_durations(
        self, duration_candidates: list[float], num_freqs: int, num_waveforms: int
    ) -> list[float]:
        """Drop candidate durations whose allocations would exceed the entry budget.

        The tone-envelope matrix is sized ``samples x (frequencies + 1)``, and its sample
        count follows the pulse duration, already bounded by
        ``field_source_temporal_max_run_time_ratio``. The quadratic allocations of
        :meth:`_frequency_entry_count` bound the remaining axis: an objective spanning
        very many adjoint frequencies. Those do not shrink with a shorter pulse, so a
        frequency-heavy plan filters out at every duration and falls back, while longer
        durations are filtered before escalation walks into a matrix already rejected.
        """

        from tidy3d.config import config

        max_entries = config.adjoint.field_source_temporal_max_waveform_entries
        return [
            duration
            for duration in duration_candidates
            if self._duration_entry_count(duration, num_freqs, num_waveforms) <= max_entries
        ]

    @staticmethod
    def _dominant_group_delay(targets: np.ndarray, freqs_all: np.ndarray) -> float:
        """Return the energy-dominant group delay of the target spectra.

        The targets inherit spectral phase from the forward fields, which acts as a
        time delay of the synthesized pulses. Centering the pulses inside their
        causal window against this delay (compensated exactly through ``post_norm``)
        minimizes the tone amplitudes needed at short pulse durations.
        """

        freq_span = float(np.max(freqs_all) - np.min(freqs_all))
        if freq_span <= 0.0:
            return 0.0
        delay_max = (freqs_all.size + 1) / freq_span
        num_delays = 8 * (freqs_all.size + 1) + 1
        delay_grid = np.linspace(-delay_max, delay_max, num_delays)
        # a target spectrum ~ e^{+2 pi i f tau} corresponds to a pulse delayed by tau
        phases = np.exp(-2j * np.pi * np.outer(delay_grid, freqs_all))
        energy = np.sum(np.abs(phases @ targets.T) ** 2, axis=1)
        return float(delay_grid[int(np.argmax(energy))])

    @staticmethod
    def _make_custom_source_time(
        envelope: np.ndarray,
        times_env: np.ndarray,
        *,
        freq0: float,
        fwidth: float,
    ) -> CustomSourceTime:
        """Wrap one complex envelope sampled on the simulation time step grid."""

        data_array = TimeDataArray(np.asarray(envelope, dtype=complex), coords={"t": times_env})
        return CustomSourceTime(
            freq0=freq0,
            fwidth=fwidth,
            offset=0.0,
            source_time_dataset=TimeDataset(values=data_array),
        )

    def _synthesize_source_times(
        self,
        targets: np.ndarray,
        freqs_all: np.ndarray,
        *,
        pulse_duration: float,
        spectrum_rtol: float,
    ) -> list[CustomSourceTime] | None:
        """Synthesize one multi-tone ``CustomSourceTime`` per target spectrum row.

        ``targets`` has shape ``(num_sources, num_freqs)`` and stores the complex
        spectrum each synthesized waveform must reach at ``freqs_all`` under the
        ``SourceTime.spectrum`` discrete-time convention. Every waveform additionally
        carries a zero-DC constraint: a current with nonzero time integral deposits
        static charge at the source that FDTD can never relax, blocking shutoff and
        polluting the frequency-domain readout. Rows are solved against a shared tone
        Gram matrix (tones at the target frequencies plus one baseband window term
        providing the DC degree of freedom) and refined against the actual
        combined-waveform spectrum until the residual is negligible. ``None`` means
        this pulse duration cannot reach the targets accurately or without excessive
        time-domain amplitude, and the caller should escalate the duration.
        """

        dt = float(self.simulation.dt)
        num_env = self._envelope_sample_count(pulse_duration)
        times_env = np.arange(num_env) * dt
        num_ramp = max(int(WINDOW_RAMP_FRACTION * num_env / 2.0), 2)
        window = np.ones(num_env)
        ramp = np.sin(0.5 * np.pi * np.arange(num_ramp) / num_ramp) ** 2
        window[:num_ramp] = ramp
        window[-num_ramp:] = ramp[::-1]
        window[0] = 0.0
        window[-1] = 0.0

        freq0 = float(np.mean(freqs_all))
        freq_span = float(np.max(freqs_all) - np.min(freqs_all))
        fwidth = max(freq_span, 1.0 / (2.0 * np.pi * times_env[-1]))

        # tone 0 is a baseband window bump (carrier cancelled); its complex amplitude
        # gives the solve two real degrees of freedom to null the complex DC content
        # built as one (num_env x num_tones) matrix rather than a list of tone arrays:
        # the refinement below needs the matrix, and keeping a separate list alive would
        # double the dominant allocation of the whole synthesis
        tone_envelope_matrix = np.empty((num_env, freqs_all.size + 1), dtype=complex)
        tone_envelope_matrix[:, 0] = window * np.exp(2j * np.pi * freq0 * times_env)
        for tone_index, freq in enumerate(freqs_all, start=1):
            tone_envelope_matrix[:, tone_index] = window * np.exp(
                -2j * np.pi * (float(freq) - freq0) * times_env
            )
        # constraint row 0 zeroes the complex time integral of ``amp_time``: the real
        # part of a nonzero integral deposits static charge through the real part of
        # the current dataset, the imaginary part through the imaginary part. The row
        # is weighted so the common tolerance drives the residual static charge far
        # below the shutoff threshold.
        carrier = np.exp(-2j * np.pi * freq0 * times_env)
        constraint_weights = np.concatenate([[DC_CONSTRAINT_WEIGHT], np.ones(freqs_all.size)])
        gram = np.zeros((freqs_all.size + 1, tone_envelope_matrix.shape[1]), dtype=complex)
        for tone_index in range(tone_envelope_matrix.shape[1]):
            tone_envelope = tone_envelope_matrix[:, tone_index]
            tone_source_time = self._make_custom_source_time(
                tone_envelope, times_env, freq0=freq0, fwidth=fwidth
            )
            gram[0, tone_index] = dt * np.sum(tone_envelope * carrier)
            gram[1:, tone_index] = tone_source_time.spectrum(times_env, freqs_all, dt)
        gram = constraint_weights[:, np.newaxis] * gram

        if float(np.max(np.abs(targets))) == 0.0:
            return None

        # peak time amplitude of a single tone carrying a row's largest spectral target
        window_gain = 0.5 * dt * float(np.sum(window))

        u_mat, singular_values, vh_mat = np.linalg.svd(gram)
        numerical_rank = int(np.sum(singular_values > GRAM_RCOND * singular_values[0]))

        # every row solves against the same gram, and the truncated pseudoinverse is a
        # pure function of (u, singular values, vh, rank), so an unchanged rank reuses it
        cached_rank = None
        gram_pinv = None

        source_times = []
        for target in targets:
            row_scale = float(np.max(np.abs(target)))
            if row_scale == 0.0:
                return None
            tolerance = spectrum_rtol * row_scale
            # weighted constraint vector: zero DC target followed by the spectra
            constraint_target = constraint_weights * np.concatenate([[0.0], target])

            # smallest truncation rank whose discarded content fits inside the
            # tolerance, which is also the smallest-amplitude solution meeting it
            projections = u_mat.conj().T @ constraint_target
            # discarded_norms[rank] = l2 norm of the components a rank-``rank``
            # solve cannot represent, for rank = 0 .. num_constraints
            discarded_norms = np.sqrt(
                np.concatenate([np.cumsum(np.abs(projections[::-1]) ** 2)[::-1], [0.0]])
            )
            achievable_ranks = np.nonzero(discarded_norms[: numerical_rank + 1] <= 0.5 * tolerance)[
                0
            ]
            rank = int(achievable_ranks[0]) if achievable_ranks.size else numerical_rank
            rank = max(rank, 1)
            if rank != cached_rank:
                gram_pinv = (
                    vh_mat[:rank].conj().T
                    @ np.diag(1.0 / singular_values[:rank])
                    @ u_mat[:, :rank].conj().T
                )
                cached_rank = rank

            amplitudes = gram_pinv @ constraint_target
            best_envelope = None
            best_residual_norm = np.inf
            for _refinement in range(MAX_SPECTRUM_REFINEMENTS):
                envelope = tone_envelope_matrix @ amplitudes
                source_time = self._make_custom_source_time(
                    envelope, times_env, freq0=freq0, fwidth=fwidth
                )
                achieved = constraint_weights * np.concatenate(
                    [
                        [dt * np.sum(envelope * carrier)],
                        np.asarray(source_time.spectrum(times_env, freqs_all, dt)),
                    ]
                )
                residual = constraint_target - achieved
                residual_norm = float(np.max(np.abs(residual)))
                if residual_norm >= best_residual_norm:
                    break
                best_envelope = envelope
                best_source_time = source_time
                best_residual_norm = residual_norm
                if residual_norm <= tolerance:
                    break
                amplitudes = amplitudes + gram_pinv @ residual

            if best_envelope is None or best_residual_norm > tolerance:
                log.info(
                    "Temporal adjoint synthesis did not converge to its spectral "
                    f"targets at pulse duration {pulse_duration:.3e}s."
                )
                return None

            amplification = float(np.max(np.abs(best_envelope))) * window_gain / row_scale
            if not np.isfinite(amplification) or amplification > MAX_TIME_AMPLIFICATION:
                log.info(
                    "Temporal adjoint synthesis requires excessive amplitude "
                    f"(amplification {amplification:.3g} > {MAX_TIME_AMPLIFICATION:g}) at "
                    f"pulse duration {pulse_duration:.3e}s."
                )
                return None
            source_times.append(best_source_time)
        return source_times

    def _plan(
        self,
        adj_srcs: list[SourceType],
        *,
        min_coverage: float,
        pulse_scale: float,
        max_run_time_ratio: float,
        max_sources: int,
        shared_decompositions: dict[tuple, BatchDecomposition] | None = None,
    ) -> tuple[TemporalPlan | None, list[SourceType]]:
        """Decompose the eligible sources and apply every gate that can reject a plan.

        ``None`` means no plan survived, and the returned list then carries every source
        the caller supplied, routed back for standard grouping. Nothing dense is
        allocated until the last gate has passed, so a rejected plan costs no memory.
        """

        pca = FieldSourcePCAProcessor(simulation=self.simulation)
        # single-frequency supports still benefit from joining the combined simulation
        support_series, remaining_sources = pca.partition_sources(adj_srcs, min_frequencies=1)

        batches: dict[tuple, list] = defaultdict(list)
        for series in support_series:
            batches[pca.batch_key(series)].append(series)

        decompositions: list[BatchDecomposition] = []
        total_sources = 0
        batch_keys = sorted(batches)
        for index, batch_key in enumerate(batch_keys):
            batch = tuple(sorted(batches[batch_key], key=lambda series: series.support_key))
            # a batch the spatial plan already attempted is reused rather than repeated:
            # the two paths differ only in the mode-count policy, applied after the fact.
            # a recorded ``None`` means that attempt failed, which it would again here
            identity = pca.shared_batch_identity(batch)
            if shared_decompositions is not None and identity in shared_decompositions:
                decomposition = shared_decompositions[identity]
            else:
                decomposition = pca.decompose_batch(batch, min_coverage=min_coverage)
            if decomposition is None:
                for series in batch:
                    remaining_sources.extend(series.original_sources)
                continue
            decompositions.append(decomposition)

            # every later batch only adds sources, so once the budget is passed the plan
            # is already doomed and decomposing the rest would be discarded work
            total_sources += decomposition.coefficients.shape[0] * len(decomposition.series_modes)
            if total_sources > max_sources:
                self._warn_source_budget_exceeded(decompositions, total_sources, max_sources)
                remaining_sources.extend(self._fallback_sources(decompositions))
                for later_key in batch_keys[index + 1 :]:
                    for series in batches[later_key]:
                        remaining_sources.extend(series.original_sources)
                return None, remaining_sources

        if not decompositions:
            return None, remaining_sources

        num_waveforms = sum(decomposition.coefficients.shape[0] for decomposition in decompositions)

        freqs_all = np.unique(
            np.concatenate(
                [np.asarray(decomposition.frequencies) for decomposition in decompositions]
            )
        )
        duration_candidates = self._pulse_duration_candidates(
            pulse_scale=pulse_scale,
            max_run_time_ratio=max_run_time_ratio,
        )
        if not duration_candidates:
            log.warning(
                "Skipping temporal adjoint source synthesis: "
                f"'config.adjoint.field_source_temporal_max_run_time_ratio'={max_run_time_ratio} "
                "leaves no room for a pulse after the field-decay margin. Consider "
                "increasing it.",
                log_once=True,
            )
            remaining_sources.extend(self._fallback_sources(decompositions))
            return None, remaining_sources

        from tidy3d.config import config as _config

        max_entries = _config.adjoint.field_source_temporal_max_waveform_entries
        affordable_durations = self._affordable_durations(
            duration_candidates, freqs_all.size, num_waveforms
        )
        if not affordable_durations:
            smallest_samples = self._envelope_sample_count(duration_candidates[0])
            smallest_entries = self._duration_entry_count(
                duration_candidates[0], freqs_all.size, num_waveforms
            )
            frequency_entries = self._frequency_entry_count(freqs_all.size)
            megabytes = BYTES_PER_ENTRY / 1e6
            log.warning(
                "Skipping temporal adjoint source synthesis: the shortest usable pulse would "
                f"allocate {smallest_entries} dense entries "
                f"(~{smallest_entries * megabytes:.1f} MB): {smallest_samples} time samples x "
                f"{freqs_all.size + 1} tones, plus {frequency_entries} entries set by "
                f"{freqs_all.size} adjoint frequencies alone, plus {num_waveforms} retained "
                f"waveforms of {smallest_samples} samples. This exceeds "
                f"'config.adjoint.field_source_temporal_max_waveform_entries'={max_entries} "
                f"(~{max_entries * megabytes:.0f} MB). These sources fall back to the other "
                "reduction paths. Raise that setting to allow this plan, or reduce the number "
                "of adjoint frequencies in the objective.",
                log_once=True,
            )
            remaining_sources.extend(self._fallback_sources(decompositions))
            return None, remaining_sources
        duration_candidates = affordable_durations

        freq_indices = {float(freq): index for index, freq in enumerate(freqs_all)}

        targets = np.zeros((num_waveforms, freqs_all.size), dtype=complex)
        target_row = 0
        for decomposition in decompositions:
            column_indices = [freq_indices[freq] for freq in decomposition.frequencies]
            for coefficient_row in decomposition.coefficients:
                targets[target_row, column_indices] = coefficient_row
                target_row += 1

        return (
            TemporalPlan(
                decompositions=tuple(decompositions),
                freqs_all=freqs_all,
                targets=targets,
                duration_candidates=tuple(duration_candidates),
                num_waveforms=num_waveforms,
                total_sources=total_sources,
            ),
            remaining_sources,
        )

    def _solve(
        self, plan: TemporalPlan, *, spectrum_rtol: float
    ) -> tuple[list[CustomSourceTime], float, float] | None:
        """Return waveforms for the shortest planned duration that meets the gates.

        ``None`` means no duration in the plan reached its spectral targets, which the
        caller answers by falling back rather than injecting waveforms that miss them.
        """

        freqs_all = plan.freqs_all
        targets = plan.targets
        duration_candidates = plan.duration_candidates

        group_delay = self._dominant_group_delay(targets, freqs_all)
        for candidate_duration in duration_candidates:
            # center the pulses inside the causal window against the dominant
            # target group delay; compensated exactly through ``post_norm`` below
            time_shift = group_delay - 0.5 * candidate_duration
            shifted_targets = targets * np.exp(-2j * np.pi * freqs_all * time_shift)
            source_times = self._synthesize_source_times(
                shifted_targets,
                freqs_all,
                pulse_duration=candidate_duration,
                spectrum_rtol=spectrum_rtol,
            )
            if source_times is not None:
                return source_times, candidate_duration, time_shift
        log.warning(
            "Skipping temporal adjoint source synthesis: no pulse duration up to "
            f"{duration_candidates[-1]:.3e}s reached the spectral targets within the "
            "accuracy and amplitude gates. Consider increasing "
            "'config.adjoint.field_source_temporal_max_run_time_ratio'.",
            log_once=True,
        )
        return None

    def _assemble(
        self,
        plan: TemporalPlan,
        source_times: list[CustomSourceTime],
        pulse_duration: float,
        time_shift: float,
    ) -> FieldSourceTemporalInfo:
        """Attach the solved waveforms to their spatial profiles."""

        decompositions = plan.decompositions
        freqs_all = plan.freqs_all
        num_waveforms = plan.num_waveforms
        total_sources = plan.total_sources

        synthesized_sources = []
        waveform_index = 0
        for decomposition in decompositions:
            for mode_index in range(decomposition.coefficients.shape[0]):
                source_time = source_times[waveform_index]
                for series, modes in decomposition.series_modes:
                    dataset = FieldSourcePCAProcessor.dataset_from_vector(
                        modes[:, mode_index],
                        templates=series.templates,
                        components=series.components,
                    )
                    synthesized_sources.append(
                        series.sources[0]
                        .updated_copy(current_dataset=dataset)
                        .updated_copy(source_time=source_time, validate=False)
                    )
                waveform_index += 1

        # the adjoint decays in the same structure as the forward, so the duration the
        # forward actually ran is the ring-down margin; using the nominal run time here
        # would over-provision by the whole shutoff saving
        run_time_adj = pulse_duration + self._effective_forward_duration()
        post_norm = xr.DataArray(
            np.exp(2j * np.pi * freqs_all * time_shift), coords={"f": freqs_all}
        )
        profile_count = sum(
            len(series.sources)
            for decomposition in decompositions
            for series, _modes in decomposition.series_modes
        )
        num_supports = sum(len(decomposition.series_modes) for decomposition in decompositions)
        log.info(
            "Synthesized temporal adjoint sources: "
            f"{profile_count} source profiles across {num_supports} supports and "
            f"{freqs_all.size} frequencies -> {total_sources} sources sharing "
            f"{num_waveforms} synthesized waveforms in one adjoint simulation "
            f"(pulse duration {pulse_duration:.3e}s)."
        )
        return FieldSourceTemporalInfo(
            sources=tuple(synthesized_sources),
            post_norm=post_norm,
            run_time=run_time_adj,
        )

    def adjoint_info(
        self,
        adj_srcs: list[SourceType],
        *,
        min_coverage: float,
        pulse_scale: float,
        max_run_time_ratio: float,
        max_sources: int,
        spectrum_rtol: float,
        shared_decompositions: dict[tuple, BatchDecomposition] | None = None,
    ) -> tuple[FieldSourceTemporalInfo | None, list[SourceType]]:
        """Build one combined time-domain adjoint source info plus leftover sources.

        Eligible ``CustomCurrentSource`` supports are decomposed by the field-source
        PCA machinery into joint spatial modes with shared per-frequency
        coefficients, then every mode receives a synthesized waveform hitting its
        coefficients at its own frequencies and zero at every other adjoint
        frequency in the combined simulation. Sources that cannot participate
        (incompatible layout, volumetric support, duplicate frequencies, failed
        decomposition or synthesis) are returned for the standard adjoint grouping
        path. ``None`` means nothing was synthesized.
        """

        plan, remaining_sources = self._plan(
            adj_srcs,
            min_coverage=min_coverage,
            pulse_scale=pulse_scale,
            max_run_time_ratio=max_run_time_ratio,
            max_sources=max_sources,
            shared_decompositions=shared_decompositions,
        )
        if plan is None:
            return None, remaining_sources

        solved = self._solve(plan, spectrum_rtol=spectrum_rtol)
        if solved is None:
            remaining_sources.extend(self._fallback_sources(plan.decompositions))
            return None, remaining_sources

        return self._assemble(plan, *solved), remaining_sources
