"""Standalone optimizers for use with tidy3d autograd.

Mirrors the `optax <https://optax.readthedocs.io>`_ API:

* ``optimizer.init(params)`` → state
* ``optimizer.update(grads, state, params)`` → (updates, new_state)
* ``apply_updates(params, updates)`` → new_params

Supports parameters as either a single ``np.ndarray`` or a ``dict`` mapping
string keys to ``np.ndarray`` / scalar values (pytree-style).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Literal, Optional, Union, get_args

import numpy as np
from pydantic import Field, PositiveFloat

from tidy3d.components.base import Tidy3dBaseModel

from .differential_operators import value_and_grad

if TYPE_CHECKING:
    from typing import TypeAlias

ParamLeaf: TypeAlias = Union[np.ndarray, float]
Params = Union[ParamLeaf, dict[str, ParamLeaf]]
Bounds = Union[
    tuple[Optional[float], Optional[float]], dict[str, tuple[Optional[float], Optional[float]]]
]
ObjectiveValue: TypeAlias = Union[float, np.ndarray]
ObjectiveFn: TypeAlias = Callable[[Params], ObjectiveValue]
OptimizeDirection: TypeAlias = Literal["min", "max"]
OPTIMIZE_DIRECTIONS = list(get_args(OptimizeDirection))
AdamState: TypeAlias = dict[str, Union[Params, int]]
OptimizeHistory: TypeAlias = dict[str, list[float]]
OptimizeCallback: TypeAlias = Callable[[Params, Params, AdamState, int, ObjectiveValue], None]


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
        self, grads: Params, state: AdamState, params: Optional[Params] = None
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


def optimize(
    objective_fn: ObjectiveFn,
    params0: Params,
    optimizer: Adam,
    num_steps: int,
    *,
    bounds: Optional[Bounds] = None,
    callback: Optional[OptimizeCallback] = None,
    direction: OptimizeDirection = "min",
) -> tuple[Params, AdamState, OptimizeHistory]:
    """Run a full gradient-descent optimization loop (convenience wrapper).

    This is a thin convenience wrapper around the optax-style stepping API provided by this
    module (no optax dependency required). It is not intended to grow into a full optimization
    framework — for advanced use cases (custom stopping criteria, checkpointing, schedulers,
    DRC-aware updates, etc.) use the lower-level ``optimizer.init`` / ``optimizer.update`` /
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
        Parameter bounds applied after each update step.
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

    Returns
    -------
    tuple
        ``(params, state, history)`` where ``history`` is a dict with keys
        ``"objective_fn_val"`` and ``"grad_norm"``, each a list of per-step
        values of the raw objective and raw gradient norm, respectively.
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

    val_and_grad_fn = value_and_grad(objective_fn)
    state = optimizer.init(params0)
    params = params0

    history = {"objective_fn_val": [], "grad_norm": []}

    for step_index in range(num_steps):
        val, grad = val_and_grad_fn(params)

        history["objective_fn_val"].append(float(val))
        history["grad_norm"].append(_grad_norm(grad))

        if callback is not None:
            callback(params, grad, state, step_index, val)

        step_grad = grad if direction == "min" else _tree_map(np.negative, grad)
        updates, state = optimizer.update(step_grad, state, params)
        params = apply_updates(params, updates)

        if bounds is not None:
            params = _clip_params(params, bounds)

    return params, state, history
