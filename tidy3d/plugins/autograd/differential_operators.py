from typing import Callable

from autograd.builtins import tuple as atuple
from autograd.core import make_vjp
from autograd.extend import vspace
from autograd.wrap_util import unary_to_nary
from numpy.typing import ArrayLike

from .utilities import scalar_objective

__all__ = [
    "value_and_grad",
    "grad",
]


@unary_to_nary
def grad(fun: Callable, x: ArrayLike, *, has_aux: bool = False) -> Callable:
    """Returns a function that computes the gradient of `fun` with respect to `x`.

    Parameters
    ----------
    fun : Callable
        The function to differentiate. Should return a scalar value, or a tuple of
        (scalar_value, auxiliary_data) if `has_aux` is True.
    x : ArrayLike
        The point at which to evaluate the gradient.
    has_aux : bool = False
        If True, `fun` returns auxiliary data as the second element of a tuple.

    Returns
    -------
    Callable
        A function that takes the same arguments as `fun` and returns its gradient at `x`.
    """
    wrapped_fun = scalar_objective(fun, has_aux=has_aux)
    vjp, result = make_vjp(lambda x: atuple(wrapped_fun(x)) if has_aux else wrapped_fun(x), x)

    if has_aux:
        ans, aux = result
        return vjp((vspace(ans).ones(), None)), aux
    ans = result
    return vjp(vspace(ans).ones())


@unary_to_nary
def value_and_grad(fun: Callable, x: ArrayLike, *, has_aux: bool = False) -> Callable:
    """Returns a function that computes both the value and gradient of `fun` with respect to `x`.

    Parameters
    ----------
    fun : Callable
        The function to differentiate. Should return a scalar value, or a tuple of
        (scalar_value, auxiliary_data) if `has_aux` is True.
    x : ArrayLike
        The point at which to evaluate the function and its gradient.
    has_aux : bool = False
        If True, `fun` returns auxiliary data as the second element of a tuple.

    Returns
    -------
    Callable
        A function that takes the same arguments as `fun` and returns its value and gradient at `x`.
    """
    wrapped_fun = scalar_objective(fun, has_aux=has_aux)
    vjp, result = make_vjp(lambda x: atuple(wrapped_fun(x)) if has_aux else wrapped_fun(x), x)

    if has_aux:
        ans, aux = result
        return (ans, vjp((vspace(ans).ones(), None))), aux
    ans = result
    return ans, vjp(vspace(ans).ones())
