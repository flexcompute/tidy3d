import inspect

import torch
from autograd import make_vjp
from autograd.extend import vspace


def to_torch(fun):
    """
    Converts an autograd function to a PyTorch function.

    Parameters
    ----------
    fun : callable
        The autograd function to be converted.

    Returns
    -------
    callable
        A PyTorch function that can be used with PyTorch tensors and supports
        autograd differentiation.

    Examples
    --------
    >>> import autograd.numpy as anp
    >>> import torch
    >>> from tidy3d.plugins.pytorch.wrapper import to_torch
    >>>
    >>> @to_torch
    ... def f(x):
    ...     return anp.sum(anp.sin(x))
    >>>
    >>> x = torch.tensor([0.0, anp.pi / 2, anp.pi], requires_grad=True)
    >>> val = f(x)
    >>> val.backward()
    >>> torch.allclose(x.grad, torch.cos(x))
    True
    """
    sig = inspect.signature(fun)

    class _Wrapper(torch.autograd.Function):
        """A `torch.autograd.Function` wrapper for the autograd function `fun`.

        See Also
        --------
        `PyTorch Autograd Function Documentation <https://pytorch.org/docs/stable/autograd.html#function>`_
        """

        @staticmethod
        def forward(ctx, *args):
            numpy_args = []
            grad_argnums = []

            # assume that all tensors are on the same device (cpu by default)
            device = torch.device("cpu")

            for idx, arg in enumerate(args):
                if torch.is_tensor(arg):
                    numpy_args.append(arg.detach().cpu().numpy())
                    device = arg.device
                    if arg.requires_grad:
                        grad_argnums.append(idx)
                else:
                    numpy_args.append(arg)

            # note: can't support differentiating w.r.t. keyword-only arguments because
            # autograd's unary_to_nary decorator passes all function arguments as positional
            _vjp = make_vjp(fun, argnum=grad_argnums)
            vjp, ans = _vjp(*numpy_args)

            ctx.vjp = vjp
            ctx.device = device
            ctx.num_args = len(args)
            ctx.grad_argnums = grad_argnums

            return torch.as_tensor(ans, device=device)

        @staticmethod
        def backward(ctx, grad_output):
            _grads = ctx.vjp(vspace(grad_output.detach().cpu().numpy()).ones())
            grads = [None] * ctx.num_args
            for idx, grad in zip(ctx.grad_argnums, _grads):
                grads[idx] = torch.as_tensor(grad, device=ctx.device) * grad_output
            return tuple(grads)

    def apply(*args, **kwargs):
        # we bind the full function signature including defaults so that we can pass
        # all values as positional since torch.autograd.Function.apply only accepts
        # positional arguments
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        return _Wrapper.apply(*bound_args.arguments.values())

    return apply
