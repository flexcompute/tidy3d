from typing import Any, Callable

from autograd import value_and_grad as value_and_grad_ag
from autograd.builtins import tuple as atuple
from autograd.core import make_vjp
from autograd.extend import vspace
from autograd.wrap_util import unary_to_nary
from numpy.typing import ArrayLike

__all__ = [
    "value_and_grad",
]


@unary_to_nary
def value_and_grad(
    fun: Callable, x: ArrayLike, *, has_aux: bool = False
) -> tuple[tuple[float, ArrayLike], Any]:
    """Returns a function that returns both value and gradient.

    This function wraps and extends autograd's 'value_and_grad' function by adding
    support for auxiliary data.

    Parameters
    ----------
    fun : Callable
        The function to differentiate. Should take a single argument and return
        a scalar value, or a tuple where the first element is a scalar value if has_aux is True.
    x : ArrayLike
        The point at which to evaluate the function and its gradient.
    has_aux : bool = False
        If True, the function returns auxiliary data as the second element of a tuple.

    Returns
    -------
    tuple[tuple[float, ArrayLike], Any]
        A tuple containing:
        - A tuple with the function value (float) and its gradient (ArrayLike)
        - The auxiliary data returned by the function (if has_aux is True)

    Raises
    ------
    TypeError
        If the function does not return a scalar value.

    Notes
    -----
    This function uses autograd for automatic differentiation. If the function
    does not return auxiliary data (has_aux is False), it delegates to autograd's
    value_and_grad function. The main extension is the support for auxiliary data
    when has_aux is True.
    """
    if not has_aux:
        return value_and_grad_ag(fun)(x)

    vjp, (ans, aux) = make_vjp(lambda x: atuple(fun(x)), x)

    if not vspace(ans).size == 1:
        raise TypeError(
            "value_and_grad only applies to real scalar-output "
            "functions. Try jacobian, elementwise_grad or "
            "holomorphic_grad."
        )

    return (ans, vjp((vspace(ans).ones(), None))), aux
