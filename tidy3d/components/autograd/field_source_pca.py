"""PCA compression utilities for FieldData-derived adjoint current sources."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import xarray as xr

from tidy3d.components.autograd.derivative_utils import compute_spatial_weights
from tidy3d.components.autograd.utils import array_digest
from tidy3d.components.data.dataset import FieldDataset
from tidy3d.components.source.current import CustomCurrentSource
from tidy3d.components.source.utils import SourceType
from tidy3d.config import config
from tidy3d.log import log

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from tidy3d.components.simulation import Simulation


# Relative slack applied to the requested coverage when selecting the shared basis,
# so that a full-coverage request is not rejected by floating-point rounding.
COVERAGE_RELATIVE_SLACK = 1.0 - 1e-12


@dataclass(frozen=True)
class FieldSourcePCAInfo:
    """Processed PCA adjoint source data before ``SimulationData`` wraps it.

    Each instance corresponds to one adjoint simulation produced by PCA compression.
    ``sources`` contains one broadband source per compatible spatial support, while
    ``post_norm`` stores the frequency-dependent coefficient used to reconstruct the
    retained source component after the adjoint simulation is run.
    """

    sources: tuple[SourceType, ...]
    post_norm: xr.DataArray


@dataclass(frozen=True)
class SourceSupportSeries:
    """One spatial support sampled over a complete, exact frequency tuple.

    ``sources`` holds one planning source per frequency, each carrying every current
    component of the support; component-split inputs are merged into these.
    ``original_sources`` holds the untouched inputs behind them, so a support that
    turns out to be non-reducible can be handed to standard grouping exactly as the
    caller supplied it.
    """

    support_key: tuple[Any, ...]
    sources: tuple[CustomCurrentSource, ...]
    frequencies: tuple[float, ...]
    components: tuple[str, ...]
    templates: Mapping[str, Any]
    sqrt_weights: np.ndarray
    original_sources: tuple[SourceType, ...]


@dataclass(frozen=True)
class BatchDecomposition:
    """Joint spatial modes and shared per-frequency coefficients for one exact batch.

    ``series_modes`` pairs every support series in the batch with its block of
    unweighted spatial mode vectors (rows follow the flattened series layout, one
    column per retained mode). All series share ``coefficients``: the physical
    current profile of series ``s`` at frequency index ``j`` is reconstructed as
    ``series_modes[s] @ coefficients[:, j]``.
    """

    frequencies: tuple[float, ...]
    series_modes: tuple[tuple[SourceSupportSeries, np.ndarray], ...]
    coefficients: np.ndarray
    coverage: float


@dataclass(frozen=True)
class FieldSourcePCAProcessor:
    """Build PCA-compressed groups of compatible custom current sources.

    The processor intentionally uses a narrow, deterministic policy. It only considers
    point, line, and planar ``CustomCurrentSource`` objects generated from FieldData,
    and plans in three steps:

    1. sources sharing a support, frequency, and source time are merged into one
       source carrying all of their current components, so the plan does not depend on
       whether components were recorded by one monitor or several;
    2. merged sources are grouped into support series with identical spatial and
       component layouts across frequency;
    3. series are batched only when their support dimension (point, line, or plane),
       complete frequency tuple, and component tuple all match exactly, and each batch
       is decomposed onto a shared frequency-coefficient basis (per-current-type SVDs
       merged by subspace union).

    Unsupported, oversized, or non-reducing batches are handed back to the standard
    adjoint source grouping path as the original, unmerged sources.
    """

    simulation: Simulation
    make_broadband_source: Callable[[list[SourceType]], SourceType] | None = None

    @staticmethod
    def source_frequency(source: SourceType) -> float:
        """Return the center frequency of one adjoint source.

        Adjoint sources are always built with a ``GaussianPulse``, whose ``freq0`` is a
        validated field, so the frequency is always available and finite.
        """

        return float(source.source_time._freq0)

    @staticmethod
    def component_frequency(field_data: Any) -> float | None:
        """Return the singleton frequency on a PCA-compatible component array."""

        if "f" not in field_data.coords:
            return None
        freq_values = np.asarray(field_data.coords["f"].values, dtype=float)
        if freq_values.size != 1 or np.any(~np.isfinite(freq_values)):
            return None
        return float(freq_values[0])

    @staticmethod
    def data_array_layout_key(field_data: Any) -> tuple[Any, ...] | None:
        """Return a compatibility key for one source current component.

        PCA combines current profiles across frequency, so the component must contain
        exactly one frequency sample. The key records dimensions, shape, and all
        non-frequency coordinates. The frequency value itself is deliberately excluded
        because it is tracked separately by ``SourceSupportSeries``.

        Coordinates enter the key through ``array_digest`` for two reasons. They must
        match by exact identity, because profiles are stacked positionally into matrix
        rows and row ``i`` has to mean the same sample point in every column. And the
        digest must be reproducible across processes, because these keys are sorted to
        fix the order of support series, batches, and stacked row blocks; a
        per-process-randomized key would reshuffle adjoint simulation order, task
        naming (and therefore cache reuse), and the retained mode vectors between
        otherwise identical runs.
        """

        if (
            "f" not in field_data.dims
            or FieldSourcePCAProcessor.component_frequency(field_data) is None
        ):
            return None

        layout_key = [tuple(field_data.dims), tuple(field_data.shape)]
        for dim in field_data.dims:
            if dim == "f":
                continue
            if dim not in field_data.coords:
                return None
            coord_values = np.asarray(field_data.coords[dim].values)
            coord_digest = array_digest(coord_values)
            layout_key.append((dim, coord_values.dtype.str, coord_values.shape, coord_digest))
        return tuple(layout_key)

    @staticmethod
    def source_dimension(source: CustomCurrentSource) -> int | None:
        """Return the geometric dimension of a current source support.

        Collapsed axes have zero size. Only dimensions 0, 1, and 2 are currently
        eligible because these correspond to point, line, and planar FieldMonitor
        adjoint sources.
        """

        sizes = np.asarray(source.size, dtype=float)
        if sizes.shape != (3,) or np.any(np.isnan(sizes)):
            return None
        return int(np.count_nonzero(~np.isclose(sizes, 0.0)))

    @classmethod
    def support_key(cls, source: SourceType) -> tuple[Any, ...] | None:
        """Return a frequency-independent support key for PCA, or ``None``.

        The key is strict: source type, center, size, interpolation settings, component
        names, and component layouts must all match. This keeps the planner simple and
        avoids inferring absent components or remapping sampled grids. Sources split
        across components are combined beforehand by ``merge_component_sources``, which
        unions recorded data rather than filling in components that were never
        recorded.

        Note that ``hash(source)`` is not a substitute for this key:
        ``Tidy3dBaseModel`` deliberately hashes data arrays by class name alone, so
        sources sampled on entirely different grids would collide and their unrelated
        profiles would be stacked into the same decomposition.
        """

        if not isinstance(source, CustomCurrentSource) or source.current_dataset is None:
            return None
        if cls.source_dimension(source) not in (0, 1, 2):
            return None

        source_freq = cls.source_frequency(source)
        field_components = source.current_dataset.field_components
        if not field_components:
            return None

        component_layouts = []
        for component in sorted(field_components):
            field_data = field_components[component]
            if cls.component_frequency(field_data) != source_freq:
                return None
            layout_key = cls.data_array_layout_key(field_data)
            if layout_key is None:
                return None
            component_layouts.append((component, layout_key))

        return (
            type(source).__name__,
            tuple(float(value) for value in source.center),
            tuple(float(value) for value in source.size),
            source.interpolate,
            source.confine_to_bounds,
            tuple(component_layouts),
        )

    @staticmethod
    def spatial_weight(source: CustomCurrentSource, field_data: Any) -> np.ndarray | None:
        """Return grid-measure weights matching one current component.

        Collapsed axes contribute no geometric measure: point sources get weight 1,
        line sources get local line length, and planar sources get local area. An axis
        with nonzero extent but only one sampled coordinate cannot supply local cell
        sizes, so its single sample represents the whole extent and contributes that
        extent as a uniform factor; otherwise a coarsely meshed line or plane would be
        measured as if it were a point and its energy would depend on the mesh. This
        matches the cell-centered convention of ``compute_spatial_weights``, where a
        sampled axis distributes its extent over one cell per coordinate. An unbounded
        axis keeps a unit factor: it is the invariant direction of a two-dimensional
        simulation, whose fields and currents are already per unit length along it.
        The returned array is broadcast to the full component shape, including the
        singleton frequency axis.
        """

        spatial_dims = []
        extent_factor = 1.0
        for axis, dim in enumerate("xyz"):
            size = float(source.size[axis])
            if np.isclose(size, 0.0):
                continue
            coord_values = (
                np.asarray(field_data.coords[dim].values, dtype=float)
                if dim in field_data.dims
                else np.zeros(0)
            )
            if coord_values.size > 1:
                if np.any(~np.isfinite(coord_values)):
                    return None
                spatial_dims.append(dim)
                continue
            if np.isfinite(size):
                extent_factor *= size

        if not spatial_dims:
            return np.full(field_data.shape, extent_factor, dtype=float)

        weights = compute_spatial_weights(field_data, dims=tuple(spatial_dims))
        weight_values = extent_factor * np.asarray(weights.values, dtype=float)
        shape = [1] * len(field_data.dims)
        for dim_index, dim in enumerate(field_data.dims):
            if dim in weights.dims:
                shape[dim_index] = len(field_data.coords[dim])
        weight = np.broadcast_to(weight_values.reshape(shape), field_data.shape)
        if np.any(weight <= 0.0) or np.any(~np.isfinite(weight)):
            return None
        return np.asarray(weight, dtype=float)

    @classmethod
    def sqrt_weights(
        cls,
        source: CustomCurrentSource,
        components: tuple[str, ...],
    ) -> np.ndarray | None:
        """Return flattened square-root quadrature weights for one source.

        Electric and magnetic current components share the same grid-measure
        weighting; the two current types are never compared against each other
        because ``decompose_batch`` truncates each in its own units.
        """

        sqrt_weight_blocks = []
        dataset = source.current_dataset
        if dataset is None:
            return None
        for component in components:
            field_data = dataset.field_components.get(component)
            if field_data is None:
                return None
            spatial_weight = cls.spatial_weight(source, field_data)
            if spatial_weight is None:
                return None
            sqrt_weight_blocks.append(np.sqrt(spatial_weight).ravel())

        if not sqrt_weight_blocks:
            return None
        return np.concatenate(sqrt_weight_blocks)

    @staticmethod
    def magnetic_row_mask(series: SourceSupportSeries) -> np.ndarray:
        """Return a boolean mask over the flattened rows of magnetic-current components."""

        return np.concatenate(
            [
                np.full(
                    np.asarray(series.templates[component].values).size,
                    component.startswith("H"),
                )
                for component in series.components
            ]
        )

    @classmethod
    def shared_coefficient_basis(
        cls,
        weighted_matrix: np.ndarray,
        magnetic_rows: np.ndarray,
        *,
        min_coverage: float,
    ) -> tuple[np.ndarray, float] | None:
        """Return orthonormal shared frequency-coefficient rows and their coverage.

        Electric and magnetic currents carry different physical units, so no single
        weighted norm can rank their combined modes meaningfully. Each current type is
        therefore truncated independently in its own units, and the shared basis is
        assembled from the union of the retained per-block directions -- no cross-block
        magnitude comparison is ever made.

        The returned basis is the shortest one for which *every* block retains at
        least ``min_coverage`` of its weighted energy, and the returned coverage is
        that measured worst-case value. Selecting by measurement rather than by a
        tolerance on subspace angles matters because two losses would otherwise
        compete for one budget: a block's own truncation, and the collapse of two
        nearly parallel per-block directions into one shared direction. A block whose
        truncation already spent the budget cannot afford the second loss, so the
        basis is extended until the measurement clears the request. Where the blocks
        share frequency structure exactly -- the common case -- the first direction
        already suffices and nothing is added.

        ``None`` means no block retained any direction (all-zero currents).
        """

        blocks = []
        direction_blocks = []
        for block_rows in (~magnetic_rows, magnetic_rows):
            block = weighted_matrix[block_rows]
            if block.size == 0:
                continue
            _u_mat, singular_values, vh_mat = np.linalg.svd(block, full_matrices=False)
            num_directions, _coverage = cls.component_count(
                singular_values,
                min_coverage=min_coverage,
                matrix_shape=block.shape,
            )
            if num_directions == 0:
                continue
            blocks.append(block)
            direction_blocks.append(vh_mat[:num_directions])

        if not direction_blocks:
            return None

        # the stacked per-block directions form an ordered candidate basis: leading
        # directions are the ones the blocks agree on, trailing ones carry their
        # disagreement. Only numerically meaningless directions are discarded here.
        stacked_directions = np.vstack(direction_blocks)
        _u_mat, singular_values, vh_mat = np.linalg.svd(stacked_directions, full_matrices=False)
        rank_floor = np.finfo(float).eps * max(stacked_directions.shape) * float(singular_values[0])
        rank = max(int(np.sum(singular_values > rank_floor)), 1)
        candidates = vh_mat[:rank]

        # walking that basis, record how much of each block's energy every prefix
        # captures. Adding orthonormal directions can only capture more, so the
        # captured fraction is nondecreasing and the shortest prefix meeting the
        # requested coverage for every block can be read off directly.
        captured_by_prefix = []
        for block in blocks:
            projected = block @ candidates.conj().T
            block_energy = float(np.sum(np.abs(block) ** 2))
            captured_by_prefix.append(
                np.cumsum(np.sum(np.abs(projected) ** 2, axis=0)) / block_energy
            )
        worst_by_prefix = np.min(np.stack(captured_by_prefix), axis=0)

        # the full candidate basis spans every block's retained directions exactly, so
        # it always satisfies the request; the relative slack keeps a full-coverage
        # request off a floating-point knife edge.
        target = min_coverage * COVERAGE_RELATIVE_SLACK
        qualifying = np.nonzero(worst_by_prefix >= target)[0]
        num_modes = int(qualifying[0]) + 1 if qualifying.size else rank
        return candidates[:num_modes], min(worst_by_prefix[num_modes - 1].item(), 1.0)

    @staticmethod
    def flatten_source(source: CustomCurrentSource, components: tuple[str, ...]) -> np.ndarray:
        """Flatten one source current dataset and include source-time phase/amplitude."""

        blocks = []
        dataset = source.current_dataset
        if dataset is None:
            return np.array([], dtype=complex)
        for component in components:
            values = np.asarray(dataset.field_components[component].values)
            blocks.append(values.ravel())

        source_time = source.source_time
        source_scale = source_time.amplitude * np.exp(1j * source_time.phase)
        return np.concatenate(blocks).astype(complex, copy=False) * source_scale

    @classmethod
    def dataset_from_vector(
        cls,
        vector: np.ndarray,
        templates: Mapping[str, Any],
        components: tuple[str, ...],
        *,
        frequency: float | None = None,
    ) -> FieldDataset:
        """Build a ``FieldDataset`` from a flattened PCA vector and templates."""

        fields: dict[str, Any] = {}
        start = 0
        for component in components:
            template = templates[component]
            values_size = np.asarray(template.values).size
            component_values = vector[start : start + values_size].reshape(template.shape)
            coords = {dim: np.asarray(template.coords[dim].values) for dim in template.dims}
            if frequency is not None and "f" in coords:
                coords["f"] = np.asarray([frequency], dtype=float)
            fields[component] = type(template)(component_values, coords=coords)
            start += values_size
        return FieldDataset(**fields)

    @staticmethod
    def component_count(
        singular_values: np.ndarray,
        min_coverage: float,
        matrix_shape: tuple[int, ...],
    ) -> tuple[int, float]:
        """Return number of PCA modes required by the requested energy coverage."""

        singular_values = np.asarray(singular_values, dtype=float)
        if singular_values.size == 0:
            return 0, 0.0

        max_singular_value = float(np.max(singular_values))
        if max_singular_value == 0.0:
            return 0, 0.0

        cutoff = np.finfo(float).eps * max(matrix_shape) * max_singular_value
        retained_values = singular_values[singular_values > cutoff]
        if retained_values.size == 0:
            return 0, 0.0

        # full coverage means the full numerical rank by definition. The cumulative
        # energy fraction below cannot express that for ill-conditioned spectra: a
        # trailing value that is tiny yet above the rank cutoff contributes less than
        # one epsilon of the energy total, so the fraction already rounds to 1.0
        # before it is reached.
        if min_coverage >= 1.0:
            return retained_values.size, 1.0

        energies = retained_values**2
        total_energy = float(np.sum(energies))
        cumulative = np.cumsum(energies) / total_energy
        cumulative[-1] = 1.0
        num_components = int(np.searchsorted(cumulative, min_coverage, side="left") + 1)
        num_components = min(num_components, retained_values.size)
        return num_components, float(cumulative[num_components - 1])

    @staticmethod
    def source_matrix_entry_count(batch: tuple[SourceSupportSeries, ...]) -> int:
        """Return dense source-matrix entries needed for one exact PCA batch."""

        if not batch:
            return 0
        num_frequencies = len(batch[0].frequencies)
        num_rows = sum(series.sqrt_weights.size for series in batch)
        return num_rows * num_frequencies

    @staticmethod
    def matrix_entry_count_is_allowed(num_entries: int) -> bool:
        """Return whether one dense PCA matrix is within the configured cap.

        The cap bounds the entries stored for one block, which is a proxy rather than a
        measurement in two respects. Peak memory during the decomposition is several
        times the block itself, since the singular-value solver allocates its own
        factors and workspace. And time is governed by the frequency count as well as
        the entry count -- the thin decomposition costs about ``entries * min(rows,
        frequencies)`` -- so two blocks at the same cap can differ by more than an
        order of magnitude in wall time. Realistic monitors sit orders of magnitude
        below the cap, so it acts as a backstop against pathological blocks rather than
        as a tuning parameter.
        """

        max_matrix_entries = config.adjoint.field_source_pca_max_matrix_entries
        if num_entries <= max_matrix_entries:
            return True

        log.warning(
            "Field-source PCA skipped a source block because its dense matrix would "
            f"contain {num_entries} entries, exceeding "
            f"config.adjoint.field_source_pca_max_matrix_entries={max_matrix_entries}. "
            "Those sources will use standard adjoint grouping.",
            log_once=True,
        )
        return False

    def support_series_from_sources(
        self,
        support_key: tuple[Any, ...],
        sources: list[CustomCurrentSource],
        *,
        min_frequencies: int = 2,
        provenance: dict[int, tuple[SourceType, ...]],
    ) -> SourceSupportSeries | None:
        """Return a valid support series for one support-key source list.

        ``provenance`` maps each planning source to the untouched sources behind it.
        """

        sorted_frequency_sources = sorted(
            ((self.source_frequency(source), source) for source in sources),
            key=lambda item: item[0],
        )
        frequencies = tuple(freq for freq, _source in sorted_frequency_sources)
        sources_by_frequency = tuple(source for _freq, source in sorted_frequency_sources)
        if len(frequencies) < min_frequencies or len(set(frequencies)) != len(frequencies):
            return None

        first_source = sources_by_frequency[0]
        first_dataset = first_source.current_dataset
        if first_dataset is None:
            return None
        components = tuple(sorted(first_dataset.field_components))
        templates = {
            component: first_dataset.field_components[component] for component in components
        }
        # sources only share a support key when every component's coordinate digest
        # matches, so the quadrature weights of the first source hold at every frequency
        sqrt_weights = self.sqrt_weights(sources_by_frequency[0], components)
        if sqrt_weights is None:
            return None

        return SourceSupportSeries(
            support_key=support_key,
            sources=sources_by_frequency,
            frequencies=frequencies,
            components=components,
            templates=templates,
            sqrt_weights=sqrt_weights,
            original_sources=tuple(
                original
                for source in sources_by_frequency
                for original in provenance.get(id(source), (source,))
            ),
        )

    @classmethod
    def merge_key(cls, source: CustomCurrentSource) -> tuple[Any, ...]:
        """Return a component-independent key for sources that may be merged.

        Sources sharing this key occupy the same support at the same frequency and are
        injected identically, so their current components can be carried by one source.
        The source time is compared with amplitude and phase normalized away because
        those scalars are folded into the merged component values.
        """

        source_time = source.source_time.updated_copy(amplitude=1.0, phase=0.0)
        return (
            type(source).__name__,
            tuple(float(value) for value in source.center),
            tuple(float(value) for value in source.size),
            source.interpolate,
            source.confine_to_bounds,
            source_time,
        )

    @classmethod
    def merge_component_sources(
        cls,
        sources: list[CustomCurrentSource],
    ) -> CustomCurrentSource | None:
        """Combine same-support sources into one carrying every current component.

        Injecting several current sources on one support is equivalent by linearity to
        injecting a single source whose dataset holds all of their components, so a
        monitor set that splits components across sources reduces exactly like one that
        records them together. Each contribution is scaled by its own source-time
        amplitude and phase, and the merged source carries unit amplitude and zero
        phase. Components repeated across sources are summed when their coordinates
        match. ``None`` means the sources cannot be represented by one dataset, and the
        caller should leave them untouched.
        """

        merged_fields: dict[str, Any] = {}
        for source in sources:
            dataset = source.current_dataset
            if dataset is None:
                return None
            source_time = source.source_time
            source_scale = source_time.amplitude * np.exp(1j * source_time.phase)
            for component, field_data in dataset.field_components.items():
                scaled = field_data * source_scale
                existing = merged_fields.get(component)
                if existing is None:
                    merged_fields[component] = scaled
                    continue
                # a repeated component only adds coherently on identical coordinates
                if existing.dims != scaled.dims or existing.shape != scaled.shape:
                    return None
                if any(
                    not np.array_equal(
                        np.asarray(existing.coords[dim].values),
                        np.asarray(scaled.coords[dim].values),
                    )
                    for dim in existing.dims
                ):
                    return None
                merged_fields[component] = existing + scaled

        if not merged_fields:
            return None

        seed = sources[0]
        return seed.updated_copy(
            current_dataset=FieldDataset(**merged_fields),
            source_time=seed.source_time.updated_copy(amplitude=1.0, phase=0.0),
        )

    @classmethod
    def merged_sources_with_provenance(
        cls,
        adj_srcs: list[SourceType],
    ) -> tuple[list[SourceType], dict[int, tuple[SourceType, ...]]]:
        """Merge same-support component-split sources, tracking their originals.

        Returns the planning source list and a map from each planning source to the
        original sources behind it, so fallback paths can hand the untouched originals
        to standard adjoint grouping.
        """

        merge_groups: dict[tuple[Any, ...], list[CustomCurrentSource]] = defaultdict(list)
        planning_sources: list[SourceType] = []
        provenance: dict[int, tuple[SourceType, ...]] = {}

        for source in adj_srcs:
            if not isinstance(source, CustomCurrentSource) or cls.support_key(source) is None:
                planning_sources.append(source)
                provenance[id(source)] = (source,)
                continue
            merge_groups[cls.merge_key(source)].append(source)

        for merge_key in sorted(merge_groups, key=str):
            group = merge_groups[merge_key]
            if len(group) == 1:
                planning_sources.append(group[0])
                provenance[id(group[0])] = (group[0],)
                continue
            merged = cls.merge_component_sources(group)
            if merged is None or cls.support_key(merged) is None:
                planning_sources.extend(group)
                for source in group:
                    provenance[id(source)] = (source,)
                continue
            planning_sources.append(merged)
            provenance[id(merged)] = tuple(group)

        return planning_sources, provenance

    def partition_sources(
        self,
        adj_srcs: list[SourceType],
        *,
        min_frequencies: int = 2,
    ) -> tuple[list[SourceSupportSeries], list[SourceType]]:
        """Partition raw adjoint sources into PCA support series and leftovers.

        Sources that share a support and frequency but carry different current
        components are merged first, so the reduction does not depend on whether an
        objective recorded its field components in one monitor or several. Leftovers
        are always the original sources, leaving the standard grouping path unchanged
        when a support turns out to be non-reducible.
        """

        planning_sources, provenance = self.merged_sources_with_provenance(adj_srcs)

        source_groups: dict[tuple[Any, ...], list[CustomCurrentSource]] = defaultdict(list)
        leftovers: list[SourceType] = []

        def add_leftovers(sources: list[SourceType]) -> None:
            for source in sources:
                leftovers.extend(provenance.get(id(source), (source,)))

        for source in planning_sources:
            support_key = self.support_key(source)
            if support_key is None or not isinstance(source, CustomCurrentSource):
                add_leftovers([source])
                continue
            source_groups[support_key].append(source)

        support_series = []
        for support_key in sorted(source_groups):
            series = self.support_series_from_sources(
                support_key,
                source_groups[support_key],
                min_frequencies=min_frequencies,
                provenance=provenance,
            )
            if series is None:
                add_leftovers(source_groups[support_key])
                continue
            support_series.append(series)

        return support_series, leftovers

    @classmethod
    def batch_key(
        cls,
        series: SourceSupportSeries,
    ) -> tuple[int, tuple[float, ...], tuple[str, ...]]:
        """Return the exact dimension/frequency/component key for one PCA batch.

        Batches only combine supports of the same geometric dimension (point with
        point, line with line, plane with plane), so every row block compared in one
        decomposition carries the same physical measure.
        """

        dimension = cls.source_dimension(series.sources[0])
        return (dimension, series.frequencies, series.components)

    @staticmethod
    def source_matrix(series: SourceSupportSeries) -> np.ndarray:
        """Return the unweighted source-profile matrix for one support series."""

        return np.column_stack(
            [
                FieldSourcePCAProcessor.flatten_source(source, series.components)
                for source in series.sources
            ]
        )

    def decompose_batch(
        self,
        batch: tuple[SourceSupportSeries, ...],
        *,
        min_coverage: float,
        max_modes: int | None = None,
    ) -> BatchDecomposition | None:
        """Decompose one exact batch onto a shared frequency-coefficient basis.

        The electric and magnetic current blocks are truncated by separate SVDs,
        each in its own physical units, and the shared coefficient basis is the
        union of the retained per-block frequency subspaces. Every series then
        projects onto that basis, so all series and both current types share one
        set of per-frequency coefficients. ``None`` means the batch was oversized,
        had no retained numerical rank, or needs more than ``max_modes`` modes to
        reach the requested per-block coverage.

        ``max_modes`` is a policy applied to the finished decomposition rather than an
        input to it, so a decomposition computed without one can be reused by a caller
        that imposes one -- see :meth:`decompose_batch_uncapped`.
        """

        decomposition = self.decompose_batch_uncapped(batch, min_coverage=min_coverage)
        if decomposition is None:
            return None
        if max_modes is not None and decomposition.coefficients.shape[0] > max_modes:
            return None
        return decomposition

    def decompose_batch_uncapped(
        self,
        batch: tuple[SourceSupportSeries, ...],
        *,
        min_coverage: float,
    ) -> BatchDecomposition | None:
        """Decompose one exact batch with no limit on the retained mode count.

        This is the shareable half of :meth:`decompose_batch`: the result depends only
        on the batch and ``min_coverage``, so two callers applying different mode-count
        policies can decompose once between them.
        """

        if not batch:
            return None

        frequencies = batch[0].frequencies
        matrix_entries = self.source_matrix_entry_count(batch)
        if not self.matrix_entry_count_is_allowed(matrix_entries):
            return None

        source_matrices = []
        weighted_blocks = []
        magnetic_blocks = []
        for series in batch:
            source_matrix = self.source_matrix(series)
            source_matrices.append(source_matrix)
            weighted_blocks.append(series.sqrt_weights[:, None] * source_matrix)
            magnetic_blocks.append(self.magnetic_row_mask(series))

        basis_result = self.shared_coefficient_basis(
            np.vstack(weighted_blocks),
            np.concatenate(magnetic_blocks),
            min_coverage=min_coverage,
        )
        if basis_result is None:
            return None
        coefficients, coverage = basis_result

        # least-squares projection per row; the quadrature weights drop out because
        # the coefficient rows are orthonormal and the weighting is row-diagonal
        series_modes = [
            (series, source_matrix @ coefficients.conj().T)
            for series, source_matrix in zip(batch, source_matrices)
        ]

        return BatchDecomposition(
            frequencies=frequencies,
            series_modes=tuple(series_modes),
            coefficients=coefficients,
            coverage=coverage,
        )

    def process_batch(
        self,
        batch: tuple[SourceSupportSeries, ...],
        *,
        min_coverage: float,
        decomposition: BatchDecomposition | None = None,
    ) -> tuple[list[FieldSourcePCAInfo], float] | None:
        """Run one exact PCA batch and build broadband PCA source groups.

        ``decomposition`` supplies an already-computed uncapped result for this batch,
        so a caller that needed it for another purpose does not pay for it twice; the
        mode-count policy is applied here either way.

        ``None`` means the batch did not reduce source count, was oversized, or had
        no retained numerical rank. The caller should then route all batch sources
        through the standard adjoint grouping path.
        """

        if not batch:
            return None

        frequencies = batch[0].frequencies
        if len(frequencies) < 2:
            return None

        max_modes = len(frequencies) - 1
        if decomposition is None:
            decomposition = self.decompose_batch(
                batch, min_coverage=min_coverage, max_modes=max_modes
            )
        elif decomposition.coefficients.shape[0] > max_modes:
            decomposition = None
        if decomposition is None:
            return None

        pca_infos = []
        for mode_index in range(decomposition.coefficients.shape[0]):
            component_sources = []
            for series, modes in decomposition.series_modes:
                mode_vector = modes[:, mode_index]
                source_at_frequencies: list[SourceType] = []
                for source in series.sources:
                    dataset = self.dataset_from_vector(
                        mode_vector,
                        templates=series.templates,
                        components=series.components,
                        frequency=source.source_time._freq0,
                    )
                    source_at_frequencies.append(source.updated_copy(current_dataset=dataset))
                component_sources.append(self.make_broadband_source(source_at_frequencies))

            pca_infos.append(
                FieldSourcePCAInfo(
                    sources=tuple(component_sources),
                    post_norm=xr.DataArray(
                        data=decomposition.coefficients[mode_index],
                        coords={"f": list(frequencies)},
                    ),
                )
            )

        return pca_infos, decomposition.coverage

    def adjoint_infos(
        self,
        adj_srcs: list[SourceType],
        *,
        min_coverage: float,
        share_decompositions: dict[tuple[Any, ...], BatchDecomposition] | None = None,
    ) -> tuple[list[FieldSourcePCAInfo], list[SourceType]]:
        """Return PCA source infos and sources that should use standard grouping.

        Pass ``share_decompositions`` to record the uncapped outcome of every batch,
        keyed by batch identity, for a later caller that applies a different mode-count
        policy. A batch that could not be decomposed is recorded as ``None`` so that
        caller does not retry it. Collecting costs the mode projections for batches this
        path rejects, which it would otherwise skip, so it is opt-in rather than
        automatic.
        """

        support_series, leftovers = self.partition_sources(adj_srcs)
        batches: dict[tuple[Any, ...], list[SourceSupportSeries]] = defaultdict(list)
        for series in support_series:
            batches[self.batch_key(series)].append(series)

        pca_infos = []
        for batch_key in sorted(batches):
            batch = tuple(sorted(batches[batch_key], key=lambda series: series.support_key))
            decomposition = None
            if share_decompositions is not None:
                decomposition = self.decompose_batch_uncapped(batch, min_coverage=min_coverage)
                share_decompositions[self.shared_batch_identity(batch)] = decomposition
                if decomposition is None:
                    # process_batch would repeat the attempt only to fail the same way
                    for series in batch:
                        leftovers.extend(series.original_sources)
                    continue
            result = self.process_batch(
                batch, min_coverage=min_coverage, decomposition=decomposition
            )
            if result is None:
                for series in batch:
                    leftovers.extend(series.original_sources)
                continue
            batch_infos, _coverage = result
            pca_infos.extend(batch_infos)

        return pca_infos, leftovers

    @classmethod
    def shared_batch_identity(cls, batch: tuple[SourceSupportSeries, ...]) -> tuple[Any, ...]:
        """Return a key identifying one batch across two independent partitions.

        The batch key alone is not enough: it fixes the dimension, frequencies and
        components but not which supports carry them, so the member support keys are
        part of the identity.
        """

        return (cls.batch_key(batch[0]), tuple(series.support_key for series in batch))
