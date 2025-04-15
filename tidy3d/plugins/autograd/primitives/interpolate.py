from typing import Optional

import numpy as np
from autograd.extend import defvjp, primitive
from numpy.typing import NDArray
from scipy.linalg import solve_banded

from tidy3d.log import log


def _assert_strictly_monotonic(x: NDArray) -> None:
    """Raise if ``x`` is not strictly monotonic (all increasing or all decreasing)."""
    dx = np.diff(x)
    if not (np.all(dx > 0) or np.all(dx < 0)):
        raise ValueError(
            "'x' must be strictly monotonic; mixed or duplicated knots "
            "cause division‑by‑zero or wrong interval selection."
        )


def get_interval_indices(x: NDArray, x_eval: NDArray) -> NDArray:
    """Return the index of the interval in which each ``x_eval`` value lies.

    Parameters
    ----------
    x : np.ndarray
        Array of points defining intervals
    x_eval : np.ndarray
        Points to find intervals for

    Returns
    -------
    np.ndarray
        Indices of intervals
    """
    # use right‑continuity to avoid a left‑side step when x_eval == x[i]
    idx = np.searchsorted(x[:-1], x_eval, side="right") - 1
    return np.clip(idx, 0, len(x) - 2)


def accumulate_polynomial_factors(
    x_points: NDArray, x_eval: NDArray, g: NDArray, order: int
) -> tuple[NDArray, NDArray]:
    """Given an array of evaluation points ``x_eval`` and the corresponding gradient
    contributions ``g``, compute the polynomial factors accumulated per interval.

    Parameters
    ----------
    x_points : np.ndarray
        Points defining the intervals
    x_eval : np.ndarray
        Evaluation points
    g : np.ndarray
        Gradient contributions
    order : int
        Polynomial order

    Returns
    -------
    poly_factors : np.ndarray
        Shape ``(n, order+1)``, where ``n = len(x_points)-1``.
        ``poly_factors[i, k]`` holds the sum of ``g_j * (dx_j**k)`` for all
        ``x_eval_j`` in the ``i``-th interval.
    idx : np.ndarray
        Indices of intervals for each ``x_eval`` point.
    """
    n = len(x_points) - 1
    idx = get_interval_indices(x_points, x_eval)
    dx = x_eval - x_points[idx]

    # (m, order+1)
    powers = dx[:, None] ** np.arange(order + 1)
    poly_factors = np.zeros((n, order + 1))
    np.add.at(poly_factors, idx, g[:, None] * powers)

    return poly_factors, idx


def compute_linear_coefficients(x: NDArray, y: NDArray) -> tuple[NDArray, NDArray]:
    """Compute linear spline coefficients.

    .. math::
        S_i(t) = a[i] + b[i](t - x[i])

    Parameters
    ----------
    x : np.ndarray
        X coordinates
    y : np.ndarray
        Y coordinates

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Coefficients ``(a, b)``
    """
    a = y[:-1].copy()
    b = (y[1:] - y[:-1]) / (x[1:] - x[:-1])
    return a, b


def evaluate_linear_spline(
    x: NDArray,
    coeffs: tuple[NDArray, NDArray],
    x_eval: NDArray,
) -> NDArray:
    """Evaluate a linear spline at the specified points.

    Parameters
    ----------
    x : np.ndarray
        X coordinates defining the spline
    coeffs : tuple[np.ndarray, np.ndarray]
        Spline coefficients ``(a, b)``
    x_eval : np.ndarray
        Points at which to evaluate the spline

    Returns
    -------
    np.ndarray
        Evaluated spline values
    """
    a, b = coeffs
    idx = get_interval_indices(x, x_eval)
    dx = x_eval - x[idx]
    return a[idx] + b[idx] * dx


def get_linear_derivative_wrt_y(
    x: NDArray,
    y: NDArray,
) -> tuple[NDArray, NDArray]:
    """Compute derivative of linear spline coefficients wrt ``y``.

    Parameters
    ----------
    x : np.ndarray
        X coordinates
    y : np.ndarray
        Y coordinates

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(da_dy, db_dy)``
    """
    n = len(x) - 1
    da_dy = np.zeros((n, len(y)))
    db_dy = np.zeros((n, len(y)))

    for i in range(n):
        # a[i] = y[i]
        da_dy[i, i] = 1.0

        # b[i] = (y[i+1] - y[i]) / (x[i+1] - x[i])
        h_i = x[i + 1] - x[i]
        db_dy[i, i] = -1.0 / h_i
        db_dy[i, i + 1] = 1.0 / h_i

    return da_dy, db_dy


def compute_quadratic_coefficients(
    x: NDArray,
    y: NDArray,
    left_deriv: Optional[float] = None,
) -> tuple[NDArray, NDArray, NDArray]:
    """Compute quadratic spline coefficients.

    .. math::
        S_i(t) = a[i] + b[i]*(t - x[i]) + c[i]*(t - x[i])^2

    Parameters
    ----------
    x : np.ndarray
        X coordinates
    y : np.ndarray
        Y coordinates
    left_deriv : float = None
        Left endpoint derivative

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Coefficients ``(a, b, c)``
    """
    n = len(x) - 1
    h = np.diff(x)

    a = y[:-1].copy()
    b = np.zeros(n)
    c = np.zeros(n)

    # If left derivative is provided, use it for the first segment
    if left_deriv is not None:
        b[0] = left_deriv
    else:
        b[0] = (y[1] - y[0]) / h[0]

    c[0] = (y[1] - y[0] - b[0] * h[0]) / (h[0] ** 2)

    # For each internal point, ensure continuity of function + 1st derivative
    for i in range(1, n):
        deriv_end = b[i - 1] + 2 * c[i - 1] * h[i - 1]
        b[i] = deriv_end
        c[i] = (y[i + 1] - y[i] - b[i] * h[i]) / (h[i] ** 2)

    return a, b, c


def evaluate_quadratic_spline(
    x: NDArray,
    coeffs: tuple[NDArray, NDArray, NDArray],
    x_eval: NDArray,
) -> NDArray:
    """Evaluate a quadratic spline at the specified points.

    Parameters
    ----------
    x : np.ndarray
        X coordinates defining the spline
    coeffs : tuple[np.ndarray, np.ndarray, np.ndarray]
        Spline coefficients ``(a, b, c)``
    x_eval : np.ndarray
        Points at which to evaluate the spline

    Returns
    -------
    np.ndarray
        Evaluated spline values
    """
    a, b, c = coeffs
    idx = get_interval_indices(x, x_eval)
    dx = x_eval - x[idx]
    return a[idx] + b[idx] * dx + c[idx] * dx**2


def get_quadratic_derivative_wrt_y(
    x: NDArray,
    y: NDArray,
    left_deriv: Optional[float] = None,
) -> tuple[NDArray, NDArray, NDArray]:
    """Compute derivative of quadratic spline coefficients wrt ``y``.

    Parameters
    ----------
    x : np.ndarray
        X coordinates
    y : np.ndarray
        Y coordinates
    left_deriv : float = None
        Left endpoint derivative

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        ``(da_dy, db_dy, dc_dy)``
    """
    n = len(x) - 1
    h = np.diff(x)

    da_dy = np.zeros((n, len(y)))
    db_dy = np.zeros((n, len(y)))
    dc_dy = np.zeros((n, len(y)))

    for i in range(n):
        da_dy[i, i] = 1.0

    if left_deriv is None:
        db_dy[0, 0] = -1.0 / h[0]
        db_dy[0, 1] = 1.0 / h[0]
        dc_dy[0, :] = 0.0
    else:
        dc_dy[0, 0] = -1.0 / (h[0] ** 2)
        dc_dy[0, 1] = 1.0 / (h[0] ** 2)

    for i in range(1, n):
        # Derivative wrt y_j: db_dy[i, j] = db_dy[i-1, j] + 2*h[i-1]*dc_dy[i-1, j]
        db_dy[i, :] = db_dy[i - 1, :] + 2.0 * h[i - 1] * dc_dy[i - 1, :]

        # First term: -db_dy[i,j] / h[i]
        dc_dy[i, :] = -(db_dy[i, :] * h[i]) / (h[i] ** 2)

        # Second term: Add direct contributions from y[i] and y[i+1]
        dc_dy[i, i] += -1.0 / (h[i] ** 2)
        dc_dy[i, i + 1] += 1.0 / (h[i] ** 2)

    return da_dy, db_dy, dc_dy


def setup_cubic_tridiagonal(
    x: NDArray,
    y: NDArray,
    endpoint_derivs: tuple[Optional[float], Optional[float]] = (None, None),
) -> tuple[NDArray, NDArray, NDArray, NDArray, NDArray]:
    """Return (lower, diag, upper, rhs, h) for the cubic spline system.

    Parameters
    ----------
    x : np.ndarray
        X coordinates
    y : np.ndarray
        Y coordinates
    endpoint_derivs : tuple[float, float] = (None, None)
        Derivatives at endpoints (left, right)

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        ``(lower, diag, upper, rhs, h)`` for the tridiagonal system
    """
    n = len(x) - 1
    h = np.diff(x)

    lower = np.zeros(n + 1)
    diag = np.zeros(n + 1)
    upper = np.zeros(n + 1)
    rhs = np.zeros(n + 1)

    left_deriv, right_deriv = endpoint_derivs

    for i in range(1, n):
        lower[i] = h[i - 1]
        diag[i] = 2.0 * (h[i - 1] + h[i])
        upper[i] = h[i]
        rhs[i] = (3.0 / h[i]) * (y[i + 1] - y[i]) - (3.0 / h[i - 1]) * (y[i] - y[i - 1])

    # Left boundary
    if left_deriv is None:
        diag[0] = 2.0 * h[0]
        upper[0] = h[0]
        rhs[0] = 3.0 * (y[1] - y[0]) / h[0]
    else:
        diag[0] = 2.0
        upper[0] = 1.0
        rhs[0] = (3.0 / h[0]) * ((y[1] - y[0]) / h[0] - left_deriv)

    # Right boundary
    if right_deriv is None:
        lower[n] = h[n - 1]
        diag[n] = 2.0 * h[n - 1]
        rhs[n] = 3.0 * (y[n] - y[n - 1]) / h[n - 1]
    else:
        lower[n] = 1.0
        diag[n] = 2.0
        rhs[n] = (3.0 / h[n - 1]) * (right_deriv - (y[n] - y[n - 1]) / h[n - 1])

    return lower, diag, upper, rhs, h


def _solve_tridiagonal(lower: NDArray, diag: NDArray, upper: NDArray, rhs: NDArray) -> NDArray:
    """Solve tridiagonal system (single RHS) using the Thomas algorithm.

    Parameters
    ----------
    lower : np.ndarray
        Lower diagonal elements
    diag : np.ndarray
        Main diagonal elements
    upper : np.ndarray
        Upper diagonal elements
    rhs : np.ndarray
        Right-hand side vector

    Returns
    -------
    np.ndarray
        Solution vector
    """
    n = diag.size
    ab = np.zeros((3, n))
    ab[0, 1:] = upper[:-1]
    ab[1, :] = diag
    ab[2, :-1] = lower[1:]
    return solve_banded((1, 1), ab, rhs, overwrite_ab=False, overwrite_b=False, check_finite=False)


def _solve_tridiagonal_multi(lower: NDArray, diag: NDArray, upper: NDArray, B: NDArray) -> NDArray:
    """Solve tridiagonal system with multiple RHS (columns of *B*).

    Parameters
    ----------
    lower : np.ndarray
        Lower diagonal elements
    diag : np.ndarray
        Main diagonal elements
    upper : np.ndarray
        Upper diagonal elements
    B : np.ndarray
        Right-hand side matrix

    Returns
    -------
    np.ndarray
        Solution matrix
    """
    n, k = B.shape
    c = upper.copy()
    d = diag.copy()
    a = lower.copy()
    W = B.copy()

    for i in range(1, n):
        m = a[i] / d[i - 1]
        d[i] -= m * c[i - 1]
        W[i, :] -= m * W[i - 1, :]

    X = np.empty_like(W)
    X[-1, :] = W[-1, :] / d[-1]
    for i in range(n - 2, -1, -1):
        X[i, :] = (W[i, :] - c[i] * X[i + 1, :]) / d[i]
    return X


def compute_cubic_coefficients(
    x: NDArray,
    y: NDArray,
    c_full: NDArray,
    h: NDArray,
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Computes cubic spline coefficients from the solved ``c`` array:

    .. math::
        S_i(t) = a[i] + b[i](t - x[i]) + c[i](t - x[i])^2 + d[i](t - x[i])^3.

    Parameters
    ----------
    x : np.ndarray
        X coordinates
    y : np.ndarray
        Y coordinates
    c_full : np.ndarray
        Solved ``c`` values
    h : np.ndarray
        Interval widths

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Coefficients ``(a, b, c, d)``
    """
    n = len(x) - 1
    a = y[:-1].copy()
    b = np.zeros(n)
    d = np.zeros(n)

    for i in range(n):
        b[i] = (y[i + 1] - y[i]) / h[i] - (h[i] / 3.0) * (2 * c_full[i] + c_full[i + 1])
        d[i] = (c_full[i + 1] - c_full[i]) / (3.0 * h[i])

    return a, b, c_full[:-1], d


def evaluate_cubic_spline(
    x: NDArray,
    coeffs: tuple[NDArray, NDArray, NDArray, NDArray],
    x_eval: NDArray,
) -> NDArray:
    """Evaluate a cubic spline at the specified points.

    Parameters
    ----------
    x : np.ndarray
        X coordinates defining the spline
    coeffs : tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Spline coefficients (a, b, c, d)
    x_eval : np.ndarray
        Points at which to evaluate the spline

    Returns
    -------
    np.ndarray
        Evaluated spline values
    """
    a, b, c, d = coeffs
    idx = get_interval_indices(x, x_eval)
    dx = x_eval - x[idx]
    return a[idx] + b[idx] * dx + c[idx] * dx**2 + d[idx] * dx**3


def compute_spline_coefficients(
    x: NDArray,
    y: NDArray,
    endpoint_derivs: tuple[Optional[float], Optional[float]] = (None, None),
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Compute the cubic spline coefficients ``(a, b, c, d)``.

    Parameters
    ----------
    x : np.ndarray
        X coordinates
    y : np.ndarray
        Y coordinates
    endpoint_derivs : tuple[float, float] = (None, None)
        Derivatives at endpoints (left, right)

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Coefficients ``(a, b, c, d)``
    """
    lower, diag, upper, rhs, h = setup_cubic_tridiagonal(x, y, endpoint_derivs)
    c_full = _solve_tridiagonal(lower, diag, upper, rhs)
    return compute_cubic_coefficients(x, y, c_full, h)


def get_cubic_derivative_wrt_y(
    x: NDArray,
    y: NDArray,
    endpoint_derivs: tuple[Optional[float], Optional[float]] = (None, None),
) -> tuple[NDArray, NDArray, NDArray, NDArray, NDArray]:
    """Compute derivatives of cubic spline coefficients ``(a, b, c, d)``
    wrt ``y`` values.

    Parameters
    ----------
    x : np.ndarray
        X coordinates
    y : np.ndarray
        Y coordinates
    endpoint_derivs : tuple[float, float] = (None, None)
        Derivatives at endpoints (left, right)

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        ``(da_dy, db_dy, dc_dy, dd_dy, h)``
    """
    n = len(x) - 1
    lower, diag, upper, _, h = setup_cubic_tridiagonal(x, y, endpoint_derivs)

    db_dy = np.zeros((n + 1, len(y)))
    left_deriv, right_deriv = endpoint_derivs

    for i in range(1, n):
        db_dy[i, i + 1] += 3.0 / h[i]
        db_dy[i, i] += -3.0 / h[i] - 3.0 / h[i - 1]
        db_dy[i, i - 1] += 3.0 / h[i - 1]

    if left_deriv is None:
        db_dy[0, 0] = -3.0 / h[0]
        db_dy[0, 1] = 3.0 / h[0]
    else:
        db_dy[0, 0] = (3.0 / h[0]) * (-1.0 / h[0])
        db_dy[0, 1] = (3.0 / h[0]) * (+1.0 / h[0])

    if right_deriv is None:
        db_dy[n, n] = 3.0 / h[n - 1]
        db_dy[n, n - 1] = -3.0 / h[n - 1]
    else:
        db_dy[n, n] = (3.0 / h[n - 1]) * (-1.0 / h[n - 1])
        db_dy[n, n - 1] = (3.0 / h[n - 1]) * (+1.0 / h[n - 1])

    dc_dy = _solve_tridiagonal_multi(lower, diag, upper, db_dy)  # shape (n+1, len(y))

    da_dy = np.zeros((n, len(y)))
    db_dy_out = np.zeros((n, len(y)))
    dd_dy = np.zeros((n, len(y)))

    for i in range(n):
        da_dy[i, i] = 1.0

        # b[i] depends on y[i], y[i+1], and c[i], c[i+1]
        # b[i] = (y[i+1]-y[i])/h[i] - h[i]/3 * (2*c[i] + c[i+1])
        # => partial wrt y[i+1] => +1/h[i]
        # => partial wrt y[i]   => -1/h[i]
        db_dy_out[i, i] += -1.0 / h[i]
        if i + 1 < len(y):
            db_dy_out[i, i + 1] += 1.0 / h[i]

        # partial wrt c[i], c[i+1]
        # => - h[i]/3 * (2 * dc_dy[i, :] + dc_dy[i+1, :])
        db_dy_out[i, :] += -(h[i] / 3.0) * (2.0 * dc_dy[i, :] + dc_dy[i + 1, :])

        # d[i] = (c[i+1] - c[i]) / (3*h[i])
        dd_dy[i, :] = (dc_dy[i + 1, :] - dc_dy[i, :]) / (3.0 * h[i])

    return da_dy, db_dy_out, dc_dy[:-1], dd_dy, h


def compute_spline_coeffs(
    x_points: NDArray,
    y_points: NDArray,
    endpoint_derivatives: tuple[Optional[float], Optional[float]] = (None, None),
    order: int = 3,
) -> tuple:
    """Compute spline coefficients for the given order.

    Parameters
    ----------
    x_points : np.ndarray
        X coordinates
    y_points : np.ndarray
        Y coordinates
    endpoint_derivatives : tuple[float, float] = (None, None)
        Derivatives at endpoints (left, right)
    order : int = 3
        Spline order

    Returns
    -------
    tuple
        order=1 => ``(a, b)``
        order=2 => ``(a, b, c)``
        order=3 => ``(a, b, c, d)``
    """
    if order == 1:
        return compute_linear_coefficients(x_points, y_points)
    elif order == 2:
        left_deriv = endpoint_derivatives[0]
        return compute_quadratic_coefficients(x_points, y_points, left_deriv)
    elif order == 3:
        return compute_spline_coefficients(x_points, y_points, endpoint_derivatives)
    else:
        raise NotImplementedError(f"Spline order '{order}' not implemented.")


def evaluate_spline(x_points: NDArray, coeffs: tuple, x_eval: NDArray) -> NDArray:
    """Evaluate a spline at the specified points.

    Parameters
    ----------
    x_points : np.ndarray
        X coordinates defining the spline
    coeffs : tuple
        Spline coefficients
    x_eval : np.ndarray
        Points at which to evaluate the spline

    Returns
    -------
    np.ndarray
        Evaluated spline values
    """
    order = len(coeffs) - 1

    if order == 1:
        return evaluate_linear_spline(x_points, coeffs, x_eval)
    elif order == 2:
        return evaluate_quadratic_spline(x_points, coeffs, x_eval)
    elif order == 3:
        return evaluate_cubic_spline(x_points, coeffs, x_eval)
    else:
        raise NotImplementedError(f"Spline order '{order}' not implemented.")


def get_spline_derivatives_wrt_y(
    order: int,
    x_points: NDArray,
    y_points: NDArray,
    endpoint_derivatives: tuple[Optional[float], Optional[float]] = (None, None),
):
    """Returns a tuple of derivative arrays for the given spline order.

    Parameters
    ----------
    order : int
        Spline order
    x_points : np.ndarray
        X coordinates
    y_points : np.ndarray
        Y coordinates
    endpoint_derivatives : tuple[float, float] = (None, None)
        Derivatives at endpoints (left, right)

    Returns
    -------
    tuple
        order=1 => ``(da_dy, db_dy)``
        order=2 => ``(da_dy, db_dy, dc_dy)``
        order=3 => ``(da_dy, db_dy, dc_dy, dd_dy, h)``
    """
    if order == 1:
        return get_linear_derivative_wrt_y(x_points, y_points)
    elif order == 2:
        left_deriv = endpoint_derivatives[0]
        return get_quadratic_derivative_wrt_y(x_points, y_points, left_deriv)
    elif order == 3:
        return get_cubic_derivative_wrt_y(x_points, y_points, endpoint_derivatives)
    else:
        raise NotImplementedError(f"Derivatives for spline order '{order}' not implemented.")


@primitive
def interpolate_spline(
    x_points: NDArray,
    y_points: NDArray,
    num_points: int,
    order: int,
    endpoint_derivatives: tuple[Optional[float], Optional[float]] = (None, None),
) -> tuple[NDArray, NDArray]:
    """Primitive function to perform spline interpolation of a given order
    with optional endpoint derivatives.

    Parameters
    ----------
    x_points : np.ndarray
        X coordinates of the data points (must be strictly monotonic)
    y_points : np.ndarray
        Y coordinates of the data points
    num_points : int
        Number of points in the output interpolation
    order : int
        Order of the spline (1=linear, 2=quadratic, 3=cubic)
    endpoint_derivatives : tuple[float, float] = (None, None)
        Derivatives at the endpoints (left, right)
        Note: For order=1 (linear), all endpoint derivatives are ignored.
              For order=2 (quadratic), only the left endpoint derivative is used.
              For order=3 (cubic), both endpoint derivatives are used if provided.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Tuple of (x_interpolated, y_interpolated) values

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([0, 1, 2])
    >>> y = np.array([0, 1, 0])
    >>> # Linear interpolation
    >>> x_interp, y_interp = interpolate_spline(x, y, num_points=5, order=1)
    >>> print(y_interp)
    [0.   0.5  1.   0.5  0. ]

    >>> # Quadratic interpolation with left endpoint derivative
    >>> x_interp, y_interp = interpolate_spline(x, y, num_points=5, endpoint_derivatives=(0, None), order=2)
    >>> print(np.round(y_interp, 3))
    [0.    0.75  1.    0.5   0.  ]

    >>> # Cubic interpolation with both endpoint derivatives
    >>> x_interp, y_interp = interpolate_spline(x, y, num_points=5, endpoint_derivatives=(0, 0), order=3)
    >>> print(np.round(y_interp, 3))
    [0.    0.75  1.    0.75  0.  ]
    """
    if order not in (1, 2, 3):
        raise NotImplementedError(f"Spline order '{order}' not implemented.")

    if x_points.shape != y_points.shape:
        raise ValueError("'x_points' and 'y_points' must have the same shape")

    if order == 1 and (endpoint_derivatives[0] is not None or endpoint_derivatives[1] is not None):
        log.warning("Endpoint derivatives are ignored for linear splines ('order=1').")
    elif order == 2 and endpoint_derivatives[1] is not None:
        log.warning("Right endpoint derivative is ignored for quadratic splines ('order=2').")

    _assert_strictly_monotonic(x_points)

    # allow decreasing input by reversing internally
    reversed_order = x_points[0] > x_points[-1]
    if reversed_order:
        x_points_proc = x_points[::-1]
        y_points_proc = y_points[::-1]
        endpoint_derivatives_proc = (endpoint_derivatives[1], endpoint_derivatives[0])
    else:
        x_points_proc = x_points
        y_points_proc = y_points
        endpoint_derivatives_proc = endpoint_derivatives

    x_min, x_max = x_points_proc[0], x_points_proc[-1]
    x_eval = np.linspace(x_min, x_max, num_points)

    coeffs = compute_spline_coeffs(x_points_proc, y_points_proc, endpoint_derivatives_proc, order)
    y_eval = evaluate_spline(x_points_proc, coeffs, x_eval)

    if reversed_order:
        return x_eval[::-1], y_eval[::-1]
    return x_eval, y_eval


def interpolate_spline_y_vjp(ans, x_points, y_points, num_points, order, endpoint_derivatives):
    """VJP for interpolate_spline wrt y_points."""

    def vjp(g):
        reversed_order = x_points[0] > x_points[-1]
        if reversed_order:
            x_proc = x_points[::-1]
            y_proc = y_points[::-1]
            endpoint_derivatives_proc = (endpoint_derivatives[1], endpoint_derivatives[0])
        else:
            x_proc = x_points
            y_proc = y_points
            endpoint_derivatives_proc = endpoint_derivatives

        derivative_data = get_spline_derivatives_wrt_y(
            order, x_proc, y_proc, endpoint_derivatives_proc
        )

        derivative_arrays = derivative_data[:-1] if order == 3 else derivative_data

        x_min, x_max = x_proc[0], x_proc[-1]
        x_eval = np.linspace(x_min, x_max, num_points)

        # extract the gradient with respect to y values only
        g_y = g[1] if isinstance(g, tuple) else g
        if reversed_order:
            g_y = g_y[::-1]

        poly_factors, _ = accumulate_polynomial_factors(x_proc, x_eval, g_y, order)

        dv_dy = np.zeros(len(y_proc))
        n = len(x_proc) - 1
        for i in range(n):
            for k, derivative_array in enumerate(derivative_arrays):
                dv_dy += derivative_array[i, :] * poly_factors[i, k]
        if reversed_order:
            dv_dy = dv_dy[::-1]
        return dv_dy

    return vjp


defvjp(interpolate_spline, None, interpolate_spline_y_vjp)
