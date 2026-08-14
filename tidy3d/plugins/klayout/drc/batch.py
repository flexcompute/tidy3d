"""Batched KLayout DRC checks for constraint-aware optimization."""

from __future__ import annotations

import re
import tempfile
import xml.etree.ElementTree as ET
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from math import ceil, isclose, sqrt
from numbers import Integral
from os import close
from pathlib import Path
from shutil import copy2, copytree
from typing import TYPE_CHECKING, Any

from tidy3d.exceptions import ValidationError, format_chained_exception_message
from tidy3d.log import log

from .drc import (
    SUPPORTED_DRC_SUFFIXES,
    DRCRunner,
    _validate_drc_args,
)
from .results import EdgePairMarker

if TYPE_CHECKING:
    from collections.abc import Generator, Iterable, Mapping, Sequence

    from .results import DRCEdge, DRCResults

ExportFn = Callable[[Any, Path], None]

_DEEP_PATTERN = re.compile(r"(?m)^\s*deep(?:\s*\(\s*\))?\s*(?:#.*)?$")
_CELL_PREFIX = "T3D_DRC_CANDIDATE"
_COORDINATE_TOLERANCE = 1e-9


@dataclass(frozen=True)
class _CandidatePlacement:
    """Location and dimensions of one candidate in the stitched layout."""

    index: int
    origin_x: float
    origin_y: float
    width: float
    height: float


@contextmanager
def _runset_with_deep(runset: Path) -> Generator[Path, None, None]:
    """Yield a runset that enables hierarchical result reporting.

    The original runset is returned when it already contains a standalone
    ``deep`` directive. Otherwise a temporary sibling is used so relative
    runset dependencies resolve from the original directory.
    """
    runset = Path(runset)
    if runset.suffix not in SUPPORTED_DRC_SUFFIXES:
        raise ValidationError(
            f"DRC runset file '{runset}' must end with one of "
            f"{', '.join(sorted(SUPPORTED_DRC_SUFFIXES))}."
        )

    try:
        content = runset.read_text()
    except FileNotFoundError as error:
        raise FileNotFoundError(
            format_chained_exception_message(f"Could not read DRC runset '{runset}'", error)
        ) from error

    if runset.suffix == ".drc":
        if _DEEP_PATTERN.search(content):
            yield runset
            return
        tree = None
    else:
        try:
            tree = ET.ElementTree(ET.fromstring(content))
        except ET.ParseError as error:
            raise ValidationError(
                format_chained_exception_message(
                    f"Could not parse KLayout macro runset '{runset}'", error
                )
            ) from error
        text_element = tree.getroot().find(".//text")
        if text_element is None:
            raise ValidationError(
                f"KLayout macro runset '{runset}' does not contain a <text> node."
            )
        if _DEEP_PATTERN.search(text_element.text or ""):
            yield runset
            return
        text_element.text = f"deep\n{text_element.text or ''}"

    try:
        file_descriptor, deep_name = tempfile.mkstemp(
            prefix=f".{runset.stem}_tidy3d_deep_",
            suffix=runset.suffix,
            dir=runset.parent,
        )
        close(file_descriptor)
    except OSError as error:
        raise OSError(
            format_chained_exception_message(
                f"Could not create a temporary runset beside '{runset}'", error
            )
        ) from error

    deep_runset = Path(deep_name)
    try:
        if tree is None:
            deep_runset.write_text(f"deep\n{content}")
        else:
            tree.write(deep_runset, encoding="utf-8", xml_declaration=True)
        yield deep_runset
    finally:
        deep_runset.unlink(missing_ok=True)


def _stitch_gds_cells(
    gds_paths: Sequence[str | Path],
    output_path: str | Path,
) -> dict[str, _CandidatePlacement]:
    """Place flattened layouts into a separated, near-square cell grid.

    Parameters
    ----------
    gds_paths : Sequence[Union[str, Path]]
        Candidate GDS files, each containing exactly one top-level cell.
    output_path : Union[str, Path]
        Destination for the stitched GDS library.
    Returns
    -------
    dict[str, _CandidatePlacement]
        Mapping from candidate cell name to its placement metadata.
    """
    if len(gds_paths) == 0:
        raise ValueError("'gds_paths' must contain at least one candidate.")

    try:
        import gdstk
    except ImportError as error:
        raise ImportError(
            format_chained_exception_message(
                "Batched KLayout DRC requires 'gdstk'. Install Tidy3D with the 'gdstk' extra",
                error,
            )
        ) from error

    output_path = Path(output_path)
    output_library = None
    output_top = None
    candidates = []

    for index, gds_path_value in enumerate(gds_paths):
        gds_path = Path(gds_path_value)
        source_library = gdstk.read_gds(gds_path)
        source_top_cells = source_library.top_level()
        if len(source_top_cells) != 1:
            raise ValueError(
                f"Candidate GDS '{gds_path}' must contain exactly one top-level cell; "
                f"found {len(source_top_cells)}."
            )

        if output_library is None:
            output_library = gdstk.Library(
                unit=source_library.unit,
                precision=source_library.precision,
            )
            output_top = output_library.new_cell("TOP")
        elif not (
            isclose(source_library.unit, output_library.unit)
            and isclose(source_library.precision, output_library.precision)
        ):
            raise ValueError("All candidate GDS files must use the same unit and precision.")

        source_top = source_top_cells[0]
        bounding_box = source_top.bounding_box()
        if bounding_box is None:
            raise ValueError(f"Candidate GDS '{gds_path}' contains no geometry.")
        (min_x, min_y), (max_x, max_y) = bounding_box
        cell_name = f"{_CELL_PREFIX}_{index:06d}"
        candidate = source_top.copy(
            cell_name,
            translation=(-float(min_x), -float(min_y)),
            deep_copy=True,
        )
        candidate.flatten()
        output_library.add(candidate)
        width = float(max_x - min_x)
        height = float(max_y - min_y)
        candidates.append((cell_name, candidate, width, height))

    gap = max(min(width, height) for _, _, width, height in candidates)
    column_count = ceil(sqrt(len(candidates)))
    row_count = ceil(len(candidates) / column_count)
    column_widths = [0.0] * column_count
    row_heights = [0.0] * row_count
    for index, (_, _, width, height) in enumerate(candidates):
        column = index % column_count
        row = index // column_count
        column_widths[column] = max(column_widths[column], width)
        row_heights[row] = max(row_heights[row], height)

    coordinate_grid = output_library.precision / output_library.unit
    column_origins = [0.0]
    for width in column_widths[:-1]:
        next_x = column_origins[-1] + width + gap
        column_origins.append(ceil(next_x / coordinate_grid) * coordinate_grid)
    row_origins = [0.0]
    for height in row_heights[:-1]:
        next_y = row_origins[-1] + height + gap
        row_origins.append(ceil(next_y / coordinate_grid) * coordinate_grid)

    candidate_cells: dict[str, _CandidatePlacement] = {}
    for index, (cell_name, candidate, width, height) in enumerate(candidates):
        origin_x = column_origins[index % column_count]
        origin_y = row_origins[index // column_count]
        output_top.add(gdstk.Reference(candidate, origin=(origin_x, origin_y)))
        candidate_cells[cell_name] = _CandidatePlacement(
            index=index,
            origin_x=origin_x,
            origin_y=origin_y,
            width=width,
            height=height,
        )

    output_library.write_gds(output_path)
    return candidate_cells


def _edge_is_in_candidate(edge: DRCEdge, candidate: _CandidatePlacement) -> bool:
    """Whether a global-coordinate edge lies within a candidate bounding box."""
    min_x = candidate.origin_x - _COORDINATE_TOLERANCE
    max_x = candidate.origin_x + candidate.width + _COORDINATE_TOLERANCE
    min_y = candidate.origin_y - _COORDINATE_TOLERANCE
    max_y = candidate.origin_y + candidate.height + _COORDINATE_TOLERANCE
    return all(min_x <= x <= max_x and min_y <= y <= max_y for x, y in edge)


def _edge_owners(
    edge: DRCEdge,
    candidate_cells: Mapping[str, _CandidatePlacement],
    source_name: str,
) -> set[str]:
    """Find edge owners, preferring its result marker's source cell."""
    if _edge_is_in_candidate(edge, candidate_cells[source_name]):
        return {source_name}
    return {
        name
        for name, candidate in candidate_cells.items()
        if _edge_is_in_candidate(edge, candidate)
    }


def _cross_candidate_interactions(
    results: DRCResults,
    candidate_cells: Mapping[str, _CandidatePlacement],
) -> set[tuple[int, int]]:
    """Find edge-pair markers whose two edges belong to different candidates."""
    interactions = set()
    for violation in results.violations_by_category.values():
        for marker in violation.markers:
            if not isinstance(marker, EdgePairMarker) or marker.cell not in candidate_cells:
                continue

            source = candidate_cells[marker.cell]
            global_edges = tuple(
                tuple((x + source.origin_x, y + source.origin_y) for x, y in edge)
                for edge in marker.edge_pair
            )
            edge_owners = tuple(
                _edge_owners(edge, candidate_cells, marker.cell) for edge in global_edges
            )
            for first_owner in edge_owners[0]:
                for second_owner in edge_owners[1]:
                    if first_owner == second_owner:
                        continue
                    first_index = candidate_cells[first_owner].index
                    second_index = candidate_cells[second_owner].index
                    pair = (
                        min(first_index, second_index),
                        max(first_index, second_index),
                    )
                    interactions.add(pair)
    return interactions


class BatchedDRCChecker:
    """Check multiple designs with one KLayout invocation.

    Starting a separate KLayout process for every design can dominate the cost
    of repeated DRC checks. This checker exports multiple designs, places each
    flattened layout in its own named subcell, and checks them together in one
    process. Each design may be generated from a different parameterization,
    making this useful for optimizer line searches as well as other workflows
    that need to check several related layouts.

    A temporary runset copy enables KLayout's ``deep`` mode so result markers
    retain the design cell names and can be mapped back to one boolean per
    input design.

    Candidates are arranged in a near-square grid. Their separation in both
    axes is the largest shorter bounding-box dimension across the batch. If
    KLayout still reports a cross-candidate edge-pair or a violation outside a
    candidate cell, the checker raises instead of marking candidates invalid.

    Parameters
    ----------
    export_fn : Callable[[Any, Path], None]
        Function that generates one design from an input value and writes it to
        the supplied GDS path.
    drc_runset : Path
        KLayout ``.drc`` or ``.lydrc`` runset.
    drc_args : Mapping[str, object], optional
        Additional KLayout runtime definitions.
    max_results_per_cell : int = 1
        Number of result markers retained per candidate cell.
    verbose : bool = False
        Whether the KLayout runner logs progress.
    debug_dir : Path, optional
        Root directory for failed-batch artifacts. When a candidate is invalid
        or checking raises, artifacts are copied into a unique child directory.
    """

    def __init__(
        self,
        export_fn: ExportFn,
        drc_runset: str | Path,
        *,
        drc_args: Mapping[str, object] | None = None,
        max_results_per_cell: int = 1,
        verbose: bool = False,
        debug_dir: str | Path | None = None,
    ) -> None:
        """Initialize the checker."""
        if not callable(export_fn):
            raise TypeError("'export_fn' must be callable.")
        if isinstance(max_results_per_cell, bool) or not isinstance(max_results_per_cell, Integral):
            raise TypeError("'max_results_per_cell' must be an integer.")
        if max_results_per_cell <= 0:
            raise ValueError("'max_results_per_cell' must be positive.")

        self._export_fn = export_fn
        self._drc_runset = Path(drc_runset)
        self._drc_args = _validate_drc_args(drc_args)
        self._max_results_per_cell = int(max_results_per_cell)
        self._verbose = verbose
        self._debug_dir = None if debug_dir is None else Path(debug_dir)
        self._last_debug_dir: Path | None = None

    def check_candidates(self, candidates: Iterable[Any]) -> list[bool]:
        """Check an iterable of candidates in one KLayout invocation."""
        return self(tuple(candidates))

    @property
    def last_debug_dir(self) -> Path | None:
        """Artifact directory written by the most recent call, if any."""
        return self._last_debug_dir

    def _save_debug_artifacts(self, directory: Path) -> Path | None:
        """Copy available failed-batch artifacts into a persistent directory."""
        if self._debug_dir is None:
            return None

        artifacts = tuple(directory.iterdir())
        if not artifacts:
            return None

        try:
            self._debug_dir.mkdir(parents=True, exist_ok=True)
            debug_dir = Path(
                tempfile.mkdtemp(
                    prefix="batch_",
                    dir=self._debug_dir,
                )
            )
            for source in artifacts:
                destination = debug_dir / source.name
                if source.is_dir():
                    copytree(source, destination)
                else:
                    copy2(source, destination)
        except OSError as error:
            log.warning(
                format_chained_exception_message(
                    f"Could not save batched DRC debug artifacts to '{self._debug_dir}'",
                    error,
                ),
                log_once=True,
            )
            return None

        log.info(f"Batched DRC debug artifacts written to '{debug_dir}'.")
        return debug_dir

    def _check_batch(self, params_batch: Sequence[Any], directory: Path) -> list[bool]:
        """Run one batched DRC check in the supplied working directory."""
        with _runset_with_deep(self._drc_runset) as deep_runset:
            runner = DRCRunner(drc_runset=deep_runset, verbose=self._verbose)
            if self._debug_dir is not None:
                try:
                    copy2(deep_runset, directory / f"runset{deep_runset.suffix}")
                except OSError as error:
                    log.warning(
                        format_chained_exception_message(
                            "Could not stage the effective DRC runset for debugging",
                            error,
                        ),
                        log_once=True,
                    )

            gds_paths = []
            for index, params in enumerate(params_batch):
                gds_path = directory / f"candidate_{index:06d}.gds"
                self._export_fn(params, gds_path)
                if not gds_path.is_file():
                    raise FileNotFoundError(
                        f"'export_fn' did not create the expected GDS file '{gds_path}'."
                    )
                gds_paths.append(gds_path)

            stitched_gds = directory / "candidates.gds"
            candidate_cells = _stitch_gds_cells(
                gds_paths,
                stitched_gds,
            )
            results = runner.run(
                source=stitched_gds,
                resultsfile=directory / "drc_results.lyrdb",
                drc_args=self._drc_args,
                max_results_per_cell=self._max_results_per_cell,
            )

            unknown_cells = set(results.violated_cells).difference(candidate_cells)
            if unknown_cells:
                names = ", ".join(sorted(unknown_cells))
                raise RuntimeError(
                    "KLayout reported violations outside the candidate cells "
                    f"({names}). The runset may not support spatially batched designs; "
                    "ensure it supports hierarchical checks or use a scalar checker."
                )

            interactions = _cross_candidate_interactions(results, candidate_cells)
            if interactions:
                pairs = ", ".join(f"{first}-{second}" for first, second in sorted(interactions))
                raise RuntimeError(
                    "KLayout reported cross-candidate DRC interactions between candidate "
                    f"indices {pairs}. The automatic separation is insufficient for this "
                    "runset; use a scalar checker for these rules."
                )

            valid = [True] * len(params_batch)
            for cell in results.violated_cells:
                valid[candidate_cells[cell].index] = False
            return valid

    def __call__(self, params_batch: Sequence[Any]) -> list[bool]:
        """Export and check all designs with one KLayout process."""
        self._last_debug_dir = None
        if len(params_batch) == 0:
            return []

        with tempfile.TemporaryDirectory(prefix="tidy3d_batched_drc_") as temp_name:
            directory = Path(temp_name)
            try:
                valid = self._check_batch(params_batch, directory)
            except Exception:
                self._last_debug_dir = self._save_debug_artifacts(directory)
                raise
            if not all(valid):
                self._last_debug_dir = self._save_debug_artifacts(directory)
            return valid
