"""Standalone optimizers for use with tidy3d autograd.

Mirrors the `optax <https://optax.readthedocs.io>`_ API:

* ``optimizer.init(params)`` → state
* ``optimizer.update(grads, state, params)`` → (updates, new_state)
* ``apply_updates(params, updates)`` → new_params

Supports parameters as either a single ``np.ndarray`` or a ``dict`` mapping
string keys to ``np.ndarray`` / scalar values (pytree-style).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from itertools import tee
from numbers import Integral
from typing import TYPE_CHECKING, Literal, Protocol, get_args, runtime_checkable

import numpy as np
from pydantic import Field, PositiveFloat

from tidy3d.components.base import Tidy3dBaseModel
from tidy3d.log import log

from .differential_operators import value_and_grad

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator
    from typing import TypeAlias

ParamLeaf: TypeAlias = np.ndarray | float
Params = ParamLeaf | dict[str, ParamLeaf]
Bounds = tuple[float | None, float | None] | dict[str, tuple[float | None, float | None]]
ObjectiveValue: TypeAlias = float | np.ndarray
ObjectiveFn: TypeAlias = Callable[[Params], ObjectiveValue]
OptimizeDirection: TypeAlias = Literal["min", "max"]
OPTIMIZE_DIRECTIONS = list(get_args(OptimizeDirection))
SafeUpdateStatus: TypeAlias = Literal["full", "partial", "rejected"]
CandidateOrder: TypeAlias = Literal["deterministic", "random"]
CANDIDATE_ORDERS = list(get_args(CandidateOrder))
AdamState: TypeAlias = dict[str, Params | int]
OptimizeHistory: TypeAlias = dict[str, list[float]]
OptimizeCallback: TypeAlias = Callable[[Params, Params, AdamState, int, ObjectiveValue], None]
ConstraintFn: TypeAlias = Callable[[Params], bool]
BatchedConstraintFn: TypeAlias = Callable[[Sequence[Params]], Sequence[bool]]
_LARGE_LINE_SEARCH_PARAMETER_COUNT = 1_000


def _tree_map(fn: Callable[..., ParamLeaf], *trees: Params) -> Params:
    """Apply ``fn`` element-wise across one or more matching pytrees (dicts or arrays)."""
    first = trees[0]
    if isinstance(first, dict):
        return {k: _tree_map(fn, *(t[k] for t in trees)) for k in first}
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


class Adam(Tidy3dBaseModel):
    """Adam optimizer (optax-compatible interface).

    Supports parameters as a single ``np.ndarray`` or a ``dict`` mapping
    string keys to arrays/scalars.

    Parameters
    ----------
    learning_rate : float
        Step size for the parameter updates.
    beta1 : float = 0.9
        Exponential decay rate for the first moment estimate.
    beta2 : float = 0.999
        Exponential decay rate for the second moment estimate.
    eps : float = 1e-8
        Small constant for numerical stability.

    Example
    -------
    >>> opt = adam(learning_rate=0.01)  # doctest: +SKIP
    >>> state = opt.init(params)  # doctest: +SKIP
    >>> for step in range(100):  # doctest: +SKIP
    ...     val, grad = value_and_grad(obj_fn)(params)
    ...     updates, state = opt.update(grad, state, params)
    ...     params = apply_updates(params, updates)
    """

    learning_rate: PositiveFloat = Field(
        title="Learning Rate",
        description="Step size for the parameter updates.",
    )

    beta1: float = Field(
        0.9,
        ge=0.0,
        le=1.0,
        title="Beta 1",
        description="Exponential decay rate for the first moment estimate.",
    )

    beta2: float = Field(
        0.999,
        ge=0.0,
        le=1.0,
        title="Beta 2",
        description="Exponential decay rate for the second moment estimate.",
    )

    eps: PositiveFloat = Field(
        1e-8,
        title="Epsilon",
        description="Small constant for numerical stability.",
    )

    def init(self, params: Params) -> AdamState:
        """Create the initial optimizer state.

        Parameters
        ----------
        params : np.ndarray or dict
            Initial parameters (array or dict of arrays), used to determine the
            shape of moment estimates.

        Returns
        -------
        dict
            Initial optimizer state with keys ``"m"``, ``"v"``, and ``"t"``.
        """
        return {
            "m": _tree_map(np.zeros_like, params),
            "v": _tree_map(np.zeros_like, params),
            "t": 0,
        }

    def update(
        self, grads: Params, state: AdamState, params: Params | None = None
    ) -> tuple[Params, AdamState]:
        """Compute parameter updates from gradients (optax-compatible).

        Parameters
        ----------
        grads : np.ndarray or dict
            Gradient of the objective with respect to parameters.
        state : dict
            Current optimizer state.
        params : np.ndarray or dict, optional
            Current parameters (unused by Adam, accepted for API compatibility).

        Returns
        -------
        tuple
            ``(updates, new_state)`` where ``updates`` is the additive delta
            to apply via :func:`apply_updates`.
        """
        m = state["m"]
        v = state["v"]
        t = int(state["t"]) + 1

        b1, b2, lr, eps = self.beta1, self.beta2, self.learning_rate, self.eps
        b1_corr = 1 - b1**t
        b2_corr = 1 - b2**t

        m = _tree_map(lambda m_i, g_i: b1 * np.asarray(m_i) + (1 - b1) * g_i, m, grads)
        v = _tree_map(lambda v_i, g_i: b2 * np.asarray(v_i) + (1 - b2) * g_i**2, v, grads)

        updates = _tree_map(
            lambda m_i, v_i: -lr * (m_i / b1_corr) / (np.sqrt(v_i / b2_corr) + eps),
            m,
            v,
        )

        new_state = {"m": m, "v": v, "t": t}
        return updates, new_state


def adam(learning_rate: float, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8) -> Adam:
    """Create an Adam optimizer (convenience factory, mirrors ``optax.adam``).

    Parameters
    ----------
    learning_rate : float
        Step size for the parameter updates.
    beta1 : float = 0.9
        Exponential decay rate for the first moment estimate.
    beta2 : float = 0.999
        Exponential decay rate for the second moment estimate.
    eps : float = 1e-8
        Small constant for numerical stability.

    Returns
    -------
    Adam
        Configured Adam optimizer instance.
    """
    return Adam(learning_rate=learning_rate, beta1=beta1, beta2=beta2, eps=eps)


def apply_updates(params: Params, updates: Params) -> Params:
    """Apply additive updates to parameters (mirrors ``optax.apply_updates``).

    Parameters
    ----------
    params : np.ndarray or dict
        Current parameters.
    updates : np.ndarray or dict
        Additive updates (as returned by ``optimizer.update()``).

    Returns
    -------
    np.ndarray or dict
        Updated parameters (``params + updates``).
    """
    return _tree_map(lambda p, u: p + u, params, updates)


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


@dataclass(frozen=True)
class BacktrackingLineSearch(SafeUpdate):
    """Find a valid update by checking progressively smaller steps.

    A bare constraint function checks one parameterization at a time. Pass a
    :class:`ConstraintChecker` implementation to control how candidates are
    evaluated, for example :class:`BatchedConstraintChecker` or
    :class:`tidy3d.plugins.klayout.BatchedDRCChecker`.

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
        Check largest-to-smallest steps deterministically or shuffle the
        nonzero candidates. The current parameters are always validated first.
    random_seed : int, optional = 0
        Seed used for reproducible randomized candidate ordering. Set to
        ``None`` for nondeterministic ordering.
    """

    checker: ConstraintChecker | ConstraintFn
    max_backtracks: int = 8
    shrink_factor: float = 0.5
    candidate_order: CandidateOrder = "deterministic"
    random_seed: int | None = 0
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
        object.__setattr__(self, "_rng", np.random.default_rng(self.random_seed))
        object.__setattr__(self, "_constraint_checker", constraint_checker)

    @property
    def candidate_scales(self) -> tuple[float, ...]:
        """Nonzero candidate step scales before ordering."""
        return tuple(self.shrink_factor**index for index in range(self.max_backtracks + 1))

    def _ordered_scales(self) -> tuple[float, ...]:
        """Return scales in their evaluation and selection order."""
        scales = self.candidate_scales
        if self.candidate_order == "random":
            permutation = self._rng.permutation(len(scales))
            scales = tuple(scales[int(index)] for index in permutation)
        return scales

    @staticmethod
    def _candidate(current: Params, proposed: Params, scale: float) -> Params:
        """Interpolate a candidate parameterization."""
        return _tree_map(lambda old, new: old + scale * (new - old), current, proposed)

    def _candidate_stream(
        self, current: Params, proposed: Params, scales: Iterable[float]
    ) -> Iterator[Params]:
        """Yield the current parameters followed by nonzero update candidates."""
        yield current
        for scale in scales:
            yield self._candidate(current, proposed, scale)

    @staticmethod
    def _as_bool(value: object, *, context: str) -> bool:
        """Convert a strict boolean checker result to ``bool``."""
        if not isinstance(value, (bool, np.bool_)):
            raise TypeError(f"{context} must return bool values. Got {type(value).__name__}.")
        return bool(value)

    @staticmethod
    def _result(candidate: Params, scale: float, num_checks: int) -> SafeUpdateResult:
        """Build the result for the selected candidate."""
        if scale == 1:
            status: SafeUpdateStatus = "full"
        elif scale == 0:
            status = "rejected"
        else:
            status = "partial"
        return SafeUpdateResult(
            params=candidate,
            status=status,
            metrics={"scale": scale, "checks": float(num_checks)},
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
                "Backtracking line searches with more than 1,000 parameters can be slow, "
                "especially when the constraint checker is not batched.",
                log_once=True,
            )

        # An unchanged proposal only needs current validation and represents a
        # zero-scale update, even when clipping produced a new parameter tree.
        scales = () if _parameters_equal(current, proposed) else self._ordered_scales()
        num_checks = 0

        def tracked_candidates() -> Iterator[Params]:
            nonlocal num_checks
            for candidate in self._candidate_stream(current, proposed, scales):
                num_checks += 1
                yield candidate

        checker_candidates, result_candidates = tee(tracked_candidates())
        results = self._constraint_checker.check_candidates(checker_candidates)
        candidate_results = zip(result_candidates, results, strict=True)

        _, current_result = next(candidate_results)
        if not self._as_bool(current_result, context="'checker'"):
            raise RuntimeError(
                "The current parameters do not satisfy the constraint; "
                "a safe update requires a valid starting point."
            )
        if not scales:
            return self._result(current, 0.0, num_checks)

        for scale, (candidate, result) in zip(scales, candidate_results, strict=True):
            if self._as_bool(result, context="'checker'"):
                return self._result(candidate, scale, num_checks)

        log.warning(
            "The constraint rejected every nonzero candidate; keeping the current parameters.",
            log_once=True,
        )
        return self._result(current, 0.0, num_checks)


def _grad_norm(grad: Params) -> float:
    """Compute the L2 norm of a gradient (array or dict of arrays)."""
    ss = _tree_reduce(lambda x: float(np.sum(np.asarray(x) ** 2)), grad)
    return float(np.sqrt(ss))


def _clip_params(params: Params, bounds: Bounds) -> Params:
    """Clip parameters according to bounds.

    Parameters
    ----------
    params : np.ndarray or dict
        Current parameters.
    bounds : tuple or dict
        For array params: ``(lo, hi)`` tuple where ``None`` means unbounded.
        For dict params: ``dict`` mapping keys to ``(lo, hi)`` tuples.
        Keys missing from ``bounds`` are left unclipped.

    Returns
    -------
    np.ndarray or dict
        Clipped parameters.
    """
    if isinstance(params, dict):
        if isinstance(bounds, dict):
            return {k: np.clip(v, *bounds[k]) if k in bounds else v for k, v in params.items()}
        lo, hi = bounds
        return {k: np.clip(v, lo, hi) for k, v in params.items()}
    lo, hi = bounds
    return np.clip(params, lo, hi)


def _call_safe_update(
    safe_update: SafeUpdate, current: Params, proposed: Params
) -> SafeUpdateResult:
    """Call a safe-update strategy and validate its result."""
    safe_result = safe_update(current=current, proposed=proposed)
    if not isinstance(safe_result, SafeUpdateResult):
        raise TypeError(
            f"'safe_update' must return a SafeUpdateResult. Got {type(safe_result).__name__}."
        )
    return safe_result


def optimize(
    objective_fn: ObjectiveFn,
    params0: Params,
    optimizer: Adam,
    num_steps: int,
    *,
    bounds: Bounds | None = None,
    callback: OptimizeCallback | None = None,
    direction: OptimizeDirection = "min",
    safe_update: SafeUpdate | None = None,
) -> tuple[Params, AdamState, OptimizeHistory]:
    """Run a full gradient-descent optimization loop (convenience wrapper).

    This is a thin convenience wrapper around the optax-style stepping API provided by this
    module (no optax dependency required). It is not intended to grow into a full optimization
    framework. For advanced use cases (custom stopping criteria, checkpointing, schedulers,
    and similar controls), use the lower-level ``optimizer.init`` / ``optimizer.update`` /
    ``apply_updates`` interface directly.

    Uses ``autograd.value_and_grad`` to compute gradients of ``objective_fn`` at each step,
    then updates parameters using the provided ``optimizer``.

    Parameters
    ----------
    objective_fn : Callable[[Params], Union[float, np.ndarray]]
        Scalar-valued objective function evaluated on the current parameters.
        It must accept the same parameter structure as ``params0`` and return a
        differentiable scalar (Python ``float`` or scalar ``np.ndarray``) that
        ``autograd.value_and_grad`` can handle.
    params0 : np.ndarray, float, or dict
        Initial parameter values as a single array, a scalar, or a dict of
        arrays/scalars.
    optimizer : Adam
        Optimizer instance that provides ``.init()`` and ``.update()`` methods.
    num_steps : int
        Number of optimization steps to run.
    bounds : tuple or dict, optional
        Parameter bounds applied to the initial parameters, each proposed
        optimizer update, and the parameters returned by ``safe_update``.
        For array params: a ``(lo, hi)`` tuple where ``None`` disables a side.
        For dict params: a ``dict`` mapping parameter keys to ``(lo, hi)`` tuples.
        Keys absent from the dict are left unclipped.
    callback : Optional[Callable[[Params, Params, AdamState, int, Union[float, np.ndarray]], None]]
        If provided, called each step **before** the parameter update as
        ``callback(params, grad, state, step_index, objective_val)``.
        All arguments reflect the pre-update state of the current iterate, and
        ``grad`` / ``objective_val`` are the raw outputs of ``objective_fn``
        before any min/max direction handling. The final (post-loop) params are
        available in the return value.
    direction : {"min", "max"} = "min"
        Optimization direction. ``"min"`` performs gradient descent on
        ``objective_fn``. ``"max"`` performs gradient ascent by negating the
        gradient passed to the optimizer while still recording the raw objective
        values in ``history["objective_fn_val"]``.
    safe_update : SafeUpdate, optional
        Strategy used to validate the bounds-clipped initial parameters before
        the first objective evaluation and to replace each bounds-clipped
        proposal with a constraint-compatible update.

    Returns
    -------
    tuple
        ``(params, state, history)`` where ``history`` is a dict with keys
        ``"objective_fn_val"`` and ``"grad_norm"``, each a list of per-step
        values of the raw objective and raw gradient norm, respectively.
        Strategy metrics are recorded with a ``"safe_update_"`` prefix.
    """
    for attr in ("init", "update"):
        if not callable(getattr(optimizer, attr, None)):
            raise TypeError(
                f"'optimizer' must have a callable '.{attr}()' method. "
                f"Got {type(optimizer).__name__}. "
                f"Use 'from tidy3d.plugins.autograd import adam; adam(learning_rate=...)' to create one."
            )

    if direction not in OPTIMIZE_DIRECTIONS:
        raise ValueError(f"'direction' must be one of {OPTIMIZE_DIRECTIONS}. Got {direction!r}.")
    if safe_update is not None and not isinstance(safe_update, SafeUpdate):
        raise TypeError(
            f"'safe_update' must be a SafeUpdate instance or None. "
            f"Got {type(safe_update).__name__}."
        )

    params = _clip_params(params0, bounds) if bounds is not None else params0
    if safe_update is not None:
        # Validate the starting point before initialization or objective evaluation.
        # Invalid starts raise; ``status`` describes step acceptance, not validity.
        _call_safe_update(safe_update, current=params, proposed=params)

    state = optimizer.init(params)
    val_and_grad_fn = value_and_grad(objective_fn)

    history = {"objective_fn_val": [], "grad_norm": []}

    for step_index in range(num_steps):
        val, grad = val_and_grad_fn(params)

        history["objective_fn_val"].append(float(val))
        history["grad_norm"].append(_grad_norm(grad))

        if callback is not None:
            callback(params, grad, state, step_index, val)

        step_grad = grad if direction == "min" else _tree_map(np.negative, grad)
        updates, state = optimizer.update(step_grad, state, params)
        proposed = apply_updates(params, updates)

        if bounds is not None:
            proposed = _clip_params(proposed, bounds)

        if safe_update is None:
            params = proposed
        else:
            safe_result = _call_safe_update(safe_update, current=params, proposed=proposed)
            params = safe_result.params
            if bounds is not None:
                params = _clip_params(params, bounds)
            for metric_name, metric_value in safe_result.metrics.items():
                history.setdefault(f"safe_update_{metric_name}", []).append(metric_value)

    return params, state, history
