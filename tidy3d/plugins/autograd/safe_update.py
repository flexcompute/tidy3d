"""Constraint-compatible optimizer updates for Tidy3D autograd."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from itertools import chain
from numbers import Integral
from typing import TYPE_CHECKING, Literal, Protocol, get_args, runtime_checkable

import numpy as np

from tidy3d.log import log

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator
    from typing import TypeAlias

ParamLeaf: TypeAlias = np.ndarray | float
Params = ParamLeaf | dict[str, ParamLeaf]
SafeUpdateStatus: TypeAlias = Literal["full", "partial", "rejected"]
CandidateOrder: TypeAlias = Literal["deterministic", "random"]
CANDIDATE_ORDERS = list(get_args(CandidateOrder))
ConstraintFn: TypeAlias = Callable[[Params], bool]
BatchedConstraintFn: TypeAlias = Callable[[Sequence[Params]], Sequence[bool]]
_LARGE_LINE_SEARCH_PARAMETER_COUNT = 1_000


def _tree_map(fn: Callable[..., ParamLeaf], *trees: Params) -> Params:
    """Apply ``fn`` element-wise across one or more matching pytrees (dicts or arrays)."""
    first = trees[0]
    if isinstance(first, dict):
        return {k: _tree_map(fn, *(t[k] for t in trees)) for k in sorted(first)}
    return fn(*trees)


def _tree_reduce(fn: Callable[[ParamLeaf], float], tree: Params, initializer: float = 0.0) -> float:
    """Reduce all leaves of a pytree to a single scalar."""
    if isinstance(tree, dict):
        return sum((_tree_reduce(fn, v, initializer) for v in tree.values()), initializer)
    return fn(tree)


def _parameter_count(params: Params) -> int:
    """Return the total number of scalar values in a parameterization."""
    return int(_tree_reduce(lambda value: float(np.size(value)), params))


def _parameters_equal(first: Params, second: Params) -> bool:
    """Return whether two parameter trees have exactly equal values."""
    if first is second:
        return True
    if isinstance(first, dict):
        return (
            isinstance(second, dict)
            and first.keys() == second.keys()
            and all(_parameters_equal(first[key], second[key]) for key in first)
        )
    if isinstance(second, dict):
        return False
    return np.array_equal(first, second)


def _parameter_values(params: Params) -> np.ndarray:
    """Flatten a parameter tree into one array of scalar values."""
    if isinstance(params, dict):
        values = [_parameter_values(params[key]) for key in sorted(params)]
        return np.concatenate(values) if values else np.array([])
    return np.asarray(params).reshape(-1)


@dataclass(frozen=True)
class SafeUpdateResult:
    """Result returned by a :class:`SafeUpdate` strategy."""

    params: Params
    status: SafeUpdateStatus
    metrics: dict[str, float] = field(default_factory=dict)


@runtime_checkable
class ConstraintChecker(Protocol):
    """Interface for evaluating one or more candidate parameterizations."""

    def check_candidates(self, candidates: Iterable[Params]) -> Iterable[bool]:
        """Return one validity result per candidate, in input order."""
        ...


@dataclass(frozen=True)
class ScalarConstraintChecker:
    """Adapt a scalar constraint function to the candidate-checker interface."""

    check_fn: ConstraintFn

    def __post_init__(self) -> None:
        """Validate the scalar constraint function."""
        if not callable(self.check_fn):
            raise TypeError("'check_fn' must be callable.")

    def check_candidates(self, candidates: Iterable[Params]) -> Iterable[bool]:
        """Check candidates lazily, one at a time."""
        return (self.check_fn(candidate) for candidate in candidates)


@dataclass(frozen=True)
class BatchedConstraintChecker:
    """Adapt a batched constraint function to the candidate-checker interface."""

    check_fn: BatchedConstraintFn

    def __post_init__(self) -> None:
        """Validate the batched constraint function."""
        if not callable(self.check_fn):
            raise TypeError("'check_fn' must be callable.")

    def check_candidates(self, candidates: Iterable[Params]) -> Iterable[bool]:
        """Materialize and check all candidates in one function call."""
        candidate_batch = tuple(candidates)
        results = list(self.check_fn(candidate_batch))
        if len(results) != len(candidate_batch):
            raise ValueError(
                "A batched 'check_fn' must return one result per candidate: "
                f"expected {len(candidate_batch)}, got {len(results)}."
            )
        return results


class SafeUpdate(ABC):
    """Abstract strategy for replacing a proposed update with a valid one."""

    @abstractmethod
    def find_safe_update(self, current: Params, proposed: Params) -> SafeUpdateResult:
        """Return a valid parameterization between ``current`` and ``proposed``."""

    def __call__(self, current: Params, proposed: Params) -> SafeUpdateResult:
        """Apply the strategy."""
        return self.find_safe_update(current=current, proposed=proposed)


def _as_bool(value: object, *, context: str) -> bool:
    """Convert a strict boolean checker result to ``bool``."""
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{context} must return bool values. Got {type(value).__name__}.")
    return bool(value)


def _coordinate_candidate(accepted: Params, proposed: Params, index: int, scale: float) -> Params:
    """Set one flattened parameter coordinate to a scaled proposed value."""
    remaining = index

    def replace_leaf(accepted_value: ParamLeaf, new: ParamLeaf) -> ParamLeaf:
        nonlocal remaining
        if remaining == -1:
            return accepted_value
        accepted_values = np.asarray(accepted_value)
        size = accepted_values.size
        if remaining >= size:
            remaining -= size
            return accepted_value
        candidate = np.array(
            accepted_value,
            dtype=np.result_type(accepted_value, new, np.float64),
            copy=True,
        )
        candidate_values = candidate.reshape(-1)
        proposed_values = np.asarray(new).reshape(-1)
        if scale == 1:
            candidate_values[remaining] = proposed_values[remaining]
        else:
            candidate_values[remaining] += scale * (
                proposed_values[remaining] - candidate_values[remaining]
            )
        remaining = -1
        return float(candidate) if np.ndim(accepted_value) == 0 else candidate

    candidate = _tree_map(replace_leaf, accepted, proposed)
    if remaining != -1:
        raise IndexError(f"Parameter index {index} is out of range.")
    return candidate


def _prefix_candidate(accepted: Params, proposed: Params, indices: np.ndarray) -> Params:
    """Apply full proposed updates to selected flattened parameter indices."""
    selected_indices = np.sort(np.asarray(indices, dtype=int))
    offset = 0

    def replace_leaf(accepted_value: ParamLeaf, new: ParamLeaf) -> ParamLeaf:
        nonlocal offset
        accepted_values = np.asarray(accepted_value)
        size = accepted_values.size
        leaf_indices = (
            selected_indices[(selected_indices >= offset) & (selected_indices < offset + size)]
            - offset
        )
        offset += size
        if leaf_indices.size == 0:
            return accepted_value
        candidate = np.array(
            accepted_value,
            dtype=np.result_type(accepted_value, new, np.float64),
            copy=True,
        )
        candidate.reshape(-1)[leaf_indices] = np.asarray(new).reshape(-1)[leaf_indices]
        return float(candidate) if np.ndim(accepted_value) == 0 else candidate

    return _tree_map(replace_leaf, accepted, proposed)


@dataclass
class _RecoveryContext:
    """Shared state for one per-parameter recovery call."""

    accepted: Params
    proposed: Params
    ordered_indices: list[int]
    candidate_scales: tuple[float, ...]
    parameter_status: np.ndarray
    checked: Callable[[Iterable[Params]], Iterator[tuple[Params, object]]]


class _RecoveryStrategy(ABC):
    """Internal interface for built-in per-parameter recovery policies."""

    @abstractmethod
    def _recover(self, context: _RecoveryContext) -> None:
        """Apply valid updates to ``context``."""


def _recover_coordinate(context: _RecoveryContext, index: int) -> None:
    """Recover one parameter with full and backtracked candidates."""
    coordinate_candidates = (
        _coordinate_candidate(context.accepted, context.proposed, index, scale)
        for scale in context.candidate_scales
    )
    for scale, (candidate, result) in zip(
        context.candidate_scales, context.checked(coordinate_candidates), strict=True
    ):
        if _as_bool(result, context="'checker'"):
            context.accepted = candidate
            context.parameter_status[index] = 2 if scale == 1.0 else 1
            break


@dataclass(frozen=True)
class CoordinateRecovery(_RecoveryStrategy):
    """Recover remaining parameters one at a time."""

    def _recover(self, context: _RecoveryContext) -> None:
        """Try each coordinate in order."""
        for index in context.ordered_indices:
            _recover_coordinate(context, index)


@dataclass(frozen=True)
class SpeculativePrefixRecovery(_RecoveryStrategy):
    """Batch cumulative full-update prefixes before coordinate recovery."""

    max_prefix_checks: int = 10

    def __post_init__(self) -> None:
        """Validate the speculative batch size."""
        if isinstance(self.max_prefix_checks, bool) or not isinstance(
            self.max_prefix_checks, Integral
        ):
            raise TypeError("'max_prefix_checks' must be an integer.")
        if self.max_prefix_checks <= 0:
            raise ValueError("'max_prefix_checks' must be positive.")

    def _recover(self, context: _RecoveryContext) -> None:
        """Recover full prefixes and then remaining coordinates."""
        position = 0
        while position < len(context.ordered_indices):
            if (
                position == 0
                or context.parameter_status[context.ordered_indices[position - 1]] == 2
            ) and len(context.ordered_indices) - position > 1:
                prefix_lengths = [1]
                while (
                    len(prefix_lengths) < self.max_prefix_checks
                    and prefix_lengths[-1] < len(context.ordered_indices) - position
                ):
                    prefix_lengths.append(
                        min(2 * prefix_lengths[-1], len(context.ordered_indices) - position)
                    )
                prefix_indices = np.asarray(
                    context.ordered_indices[position : position + prefix_lengths[-1]]
                )
                accepted_prefix = context.accepted
                accepted_length = 0
                prefix_candidates = (
                    _prefix_candidate(context.accepted, context.proposed, prefix_indices[:length])
                    for length in prefix_lengths
                )
                for length, (candidate, result) in zip(
                    prefix_lengths, context.checked(prefix_candidates), strict=True
                ):
                    if _as_bool(result, context="'checker'"):
                        accepted_prefix = candidate
                        accepted_length = length

                if accepted_length:
                    context.accepted = accepted_prefix
                    context.parameter_status[prefix_indices[:accepted_length]] = 2
                    position += accepted_length
                    if accepted_length == prefix_indices.size:
                        continue

            _recover_coordinate(context, context.ordered_indices[position])
            position += 1


@dataclass(frozen=True)
class BacktrackingSafeUpdate(SafeUpdate):
    """Find a valid update with global and per-parameter backtracking.

    A bare constraint function checks one parameterization at a time. Pass a
    :class:`ConstraintChecker` implementation to control how candidates are
    evaluated, for example :class:`BatchedConstraintChecker` or
    :class:`tidy3d.plugins.klayout.BatchedDRCChecker`.

    The current parameterization is validated first. The full proposed update
    and successively smaller global updates are then checked from largest to
    smallest scale. Unless the full update passes, the configured recovery
    policy improves the accepted global update. If no global candidate passes,
    recovery starts from the current parameterization instead. Parameter order
    is deterministic or reproducibly randomized by ``candidate_order`` and
    ``random_seed``.

    The returned metrics include the accepted global scale, the norm fraction
    of the proposed update retained, the number of constraint checks, and the
    counts of fully accepted, partially accepted, and unchanged parameters.

    Parameters
    ----------
    checker : ConstraintChecker or Callable
        Candidate checker, or a scalar function that receives one
        parameterization and returns whether it is valid. Wrap a function with
        :class:`BatchedConstraintChecker` when it accepts a complete batch.
    max_backtracks : int = 8
        Maximum number of successively shrunken nonzero steps after the full
        proposed step.
    shrink_factor : float = 0.5
        Factor in ``(0, 1)`` used to shrink each successive step.
    candidate_order : {"deterministic", "random"} = "deterministic"
        Order in which parameters are recovered after the global search. Global
        candidates are always checked first from largest to smallest scale.
    random_seed : int, optional = 0
        Seed used for reproducible randomized parameter ordering. Set to
        ``None`` for nondeterministic ordering.
    recovery : CoordinateRecovery or SpeculativePrefixRecovery or None
        Policy used after a rejected full global update. Defaults to
        :class:`SpeculativePrefixRecovery`. Set to ``None`` to retain only the
        global search.
    """

    checker: ConstraintChecker | ConstraintFn
    max_backtracks: int = 8
    shrink_factor: float = 0.5
    candidate_order: CandidateOrder = "deterministic"
    random_seed: int | None = 0
    recovery: CoordinateRecovery | SpeculativePrefixRecovery | None = field(
        default_factory=SpeculativePrefixRecovery
    )
    _rng: np.random.Generator = field(init=False, repr=False, compare=False)
    _constraint_checker: ConstraintChecker = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Validate configuration and initialize the random number generator."""
        if isinstance(self.checker, ConstraintChecker):
            constraint_checker = self.checker
        elif callable(self.checker):
            constraint_checker = ScalarConstraintChecker(self.checker)
        else:
            raise TypeError("'checker' must be a ConstraintChecker or callable.")
        if isinstance(self.max_backtracks, bool) or not isinstance(self.max_backtracks, Integral):
            raise TypeError("'max_backtracks' must be an integer.")
        if self.max_backtracks < 0:
            raise ValueError("'max_backtracks' must be nonnegative.")
        if not np.isfinite(self.shrink_factor) or not 0 < self.shrink_factor < 1:
            raise ValueError("'shrink_factor' must be finite and strictly between 0 and 1.")
        if self.candidate_order not in CANDIDATE_ORDERS:
            raise ValueError(
                f"'candidate_order' must be one of {CANDIDATE_ORDERS}. "
                f"Got {self.candidate_order!r}."
            )
        if self.random_seed is not None and (
            isinstance(self.random_seed, bool) or not isinstance(self.random_seed, Integral)
        ):
            raise TypeError("'random_seed' must be an integer or None.")
        if self.recovery is not None and not isinstance(self.recovery, _RecoveryStrategy):
            raise TypeError(
                "'recovery' must be CoordinateRecovery, SpeculativePrefixRecovery, or None."
            )
        object.__setattr__(self, "_rng", np.random.default_rng(self.random_seed))
        object.__setattr__(self, "_constraint_checker", constraint_checker)

    @property
    def candidate_scales(self) -> tuple[float, ...]:
        """Nonzero candidate step scales from largest to smallest."""
        return tuple(self.shrink_factor**index for index in range(self.max_backtracks + 1))

    def _ordered_parameter_indices(self, params: Params) -> np.ndarray:
        """Return indices in the configured per-parameter recovery order."""
        indices = np.arange(_parameter_count(params))
        if self.candidate_order == "random":
            return self._rng.permutation(indices)
        return indices

    @staticmethod
    def _candidate(current: Params, proposed: Params, scale: float) -> Params:
        """Interpolate a candidate parameterization."""
        return _tree_map(lambda old, new: old + scale * (new - old), current, proposed)

    @staticmethod
    def _result(
        current: Params,
        proposed: Params,
        candidate: Params,
        global_scale: float,
        num_checks: int,
        parameter_status: np.ndarray,
    ) -> SafeUpdateResult:
        """Build the result and summarize its per-parameter recovery."""
        current_values = _parameter_values(current)
        proposed_values = _parameter_values(proposed)
        candidate_values = _parameter_values(candidate)
        changed = proposed_values != current_values
        full = changed & (parameter_status == 2)
        partial = changed & (parameter_status == 1)
        rejected = changed & ~(full | partial)
        if np.any(full | partial):
            status: SafeUpdateStatus = "full" if np.all(full[changed]) else "partial"
        else:
            status = "rejected"
        proposed_norm = float(np.linalg.norm(proposed_values - current_values))
        retained_fraction = (
            float(np.linalg.norm(candidate_values - current_values)) / proposed_norm
            if proposed_norm
            else 0.0
        )
        return SafeUpdateResult(
            params=candidate,
            status=status,
            metrics={
                "global_scale": global_scale,
                "retained_fraction": retained_fraction,
                "checks": float(num_checks),
                "full_parameter_updates": float(np.count_nonzero(full)),
                "partial_parameter_updates": float(np.count_nonzero(partial)),
                "rejected_parameter_updates": float(np.count_nonzero(rejected)),
            },
        )

    def find_safe_update(self, current: Params, proposed: Params) -> SafeUpdateResult:
        """Return the first valid update, or retain a valid current parameterization.

        Raises
        ------
        RuntimeError
            If the current parameterization does not satisfy the constraint.
        """
        if _parameter_count(current) > _LARGE_LINE_SEARCH_PARAMETER_COUNT:
            log.warning(
                "Backtracking safe updates with more than 1,000 parameters can be slow, "
                "especially when the constraint checker is not batched.",
                log_once=True,
            )

        # An unchanged proposal only needs current validation and represents a
        # zero-scale update, even when clipping produced a new parameter tree.
        scales = () if _parameters_equal(current, proposed) else self.candidate_scales
        current_values = _parameter_values(current)
        proposed_values = _parameter_values(proposed)
        changed = proposed_values != current_values
        parameter_status = np.zeros(current_values.size, dtype=np.int8)
        num_checks = 0

        def checked(candidates: Iterable[Params]) -> Iterator[tuple[Params, object]]:
            nonlocal num_checks
            if isinstance(self._constraint_checker, ScalarConstraintChecker):
                for candidate in candidates:
                    num_checks += 1
                    yield candidate, self._constraint_checker.check_fn(candidate)
                return
            candidate_batch = tuple(candidates)

            def tracked_candidates() -> Iterator[Params]:
                nonlocal num_checks
                for candidate in candidate_batch:
                    num_checks += 1
                    yield candidate

            results = self._constraint_checker.check_candidates(tracked_candidates())
            yield from zip(candidate_batch, results, strict=True)

        initial_results = checked(
            chain((current,), (self._candidate(current, proposed, scale) for scale in scales))
        )
        _, current_result = next(initial_results)
        if not _as_bool(current_result, context="'checker'"):
            raise RuntimeError(
                "The current parameters do not satisfy the constraint; "
                "a safe update requires a valid starting point."
            )
        if not scales:
            return self._result(current, proposed, current, 0.0, num_checks, parameter_status)

        accepted = current
        global_scale = 0.0
        for scale, (candidate, result) in zip(scales, initial_results, strict=True):
            if _as_bool(result, context="'checker'"):
                accepted = candidate
                global_scale = scale
                parameter_status[changed] = 2 if scale == 1.0 else 1
                break

        if global_scale != 1.0 and self.recovery is not None:
            ordered_indices = [
                int(index)
                for index in self._ordered_parameter_indices(proposed)
                if current_values[index] != proposed_values[index]
            ]
            context = _RecoveryContext(
                accepted=accepted,
                proposed=proposed,
                ordered_indices=ordered_indices,
                candidate_scales=scales,
                parameter_status=parameter_status,
                checked=checked,
            )
            self.recovery._recover(context)
            accepted = context.accepted

        if _parameters_equal(accepted, current):
            log.warning(
                "The constraint rejected every nonzero candidate; keeping the current parameters.",
                log_once=True,
            )
        return self._result(current, proposed, accepted, global_scale, num_checks, parameter_status)
