from __future__ import annotations

import inspect
from functools import wraps
from typing import TYPE_CHECKING

from tidy3d.components.types import LengthUnit
from tidy3d.constants import UnitScaling
from tidy3d.exceptions import Tidy3dKeyError
from tidy3d.packaging import pyvista

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any, ParamSpec, TypeVar

    import matplotlib.ticker as ticker
    from matplotlib.axes import Axes

    P = ParamSpec("P")
    T = TypeVar("T", bound=Callable[..., Axes])

    from tidy3d.components.types import Ax, Axis


def _create_unit_aware_locator() -> ticker.Locator:
    """Create UnitAwareLocator lazily due to matplotlib import restrictions."""
    import matplotlib.ticker as ticker

    class UnitAwareLocator(ticker.Locator):
        """Custom tick locator that places ticks at nice positions in the target unit."""

        def __init__(self, scale_factor: float) -> None:
            """
            Parameters
            ----------
            scale_factor : float
                Factor to convert from micrometers to the target unit.
            """
            super().__init__()
            self.scale_factor = scale_factor

        def __call__(self) -> list[float]:
            vmin, vmax = self.axis.get_view_interval()
            return self.tick_values(vmin, vmax)

        def view_limits(self, vmin: float, vmax: float) -> tuple[float, float]:
            """Override to prevent matplotlib from adjusting our limits."""
            return vmin, vmax

        def tick_values(self, vmin: float, vmax: float) -> list[float]:
            # convert the view range to the target unit
            vmin_unit = vmin * self.scale_factor
            vmax_unit = vmax * self.scale_factor

            # tolerance for floating point comparisons in target unit
            unit_range = vmax_unit - vmin_unit
            unit_tol = unit_range * 1e-8

            locator = ticker.MaxNLocator(nbins=11, prune=None, min_n_ticks=2)

            ticks_unit = locator.tick_values(vmin_unit, vmax_unit)

            # ensure we have ticks that cover the full range
            if len(ticks_unit) > 0:
                if ticks_unit[0] > vmin_unit + unit_tol or ticks_unit[-1] < vmax_unit - unit_tol:
                    # try with fewer bins to get better coverage
                    for n in [10, 9, 8, 7, 6, 5]:
                        locator = ticker.MaxNLocator(nbins=n, prune=None, min_n_ticks=2)
                        ticks_unit = locator.tick_values(vmin_unit, vmax_unit)
                        if (
                            len(ticks_unit) >= 3
                            and ticks_unit[0] <= vmin_unit + unit_tol
                            and ticks_unit[-1] >= vmax_unit - unit_tol
                        ):
                            break

                # if still no good coverage, manually ensure edge coverage
                if len(ticks_unit) > 0:
                    if (
                        ticks_unit[0] > vmin_unit + unit_tol
                        or ticks_unit[-1] < vmax_unit - unit_tol
                    ):
                        # find a reasonable step size from existing ticks
                        if len(ticks_unit) > 1:
                            step = ticks_unit[1] - ticks_unit[0]
                        else:
                            step = unit_range / 5

                        # extend the range to ensure coverage
                        extended_min = vmin_unit - step
                        extended_max = vmax_unit + step

                        # try one more time with extended range
                        locator = ticker.MaxNLocator(nbins=8, prune=None, min_n_ticks=2)
                        ticks_unit = locator.tick_values(extended_min, extended_max)

                        # filter to reasonable bounds around the original range
                        ticks_unit = [
                            t
                            for t in ticks_unit
                            if t >= vmin_unit - step / 2 and t <= vmax_unit + step / 2
                        ]

            # convert the nice ticks back to the original data unit (micrometers)
            ticks_um = ticks_unit / self.scale_factor

            # filter to ensure ticks are within bounds (with small tolerance)
            eps = (vmax - vmin) * 1e-8
            return [tick for tick in ticks_um if vmin - eps <= tick <= vmax + eps]

    return UnitAwareLocator


def make_ax() -> Ax:
    """makes an empty ``ax``."""
    from tidy3d.components.viz import _ensure_tidy3d_style

    _ensure_tidy3d_style()

    import matplotlib.pyplot as plt

    _, ax = plt.subplots(1, 1, tight_layout=True)
    return ax


def add_ax_if_none(plot: T) -> T:
    """Decorates ``plot(*args, **kwargs, ax=None)`` function.
    if ax=None in the function call, creates an ax and feeds it to rest of function.
    Also ensures tidy3d matplotlib style is applied.
    """

    @wraps(plot)
    def _plot(*args: P.args, **kwargs: P.kwargs) -> Axes:
        """New plot function using a generated ax if None."""
        from tidy3d.components.viz import _ensure_tidy3d_style

        _ensure_tidy3d_style()

        if kwargs.get("ax") is None:
            ax = make_ax()
            kwargs["ax"] = ax
        return plot(*args, **kwargs)

    return _plot


def equal_aspect(plot: T) -> T:
    """Decorates a plotting function returning a matplotlib axes.
    Ensures the aspect ratio of the returned axes is set to equal.
    Useful for 2D plots, like sim.plot() or sim_data.plot_fields()
    """

    @wraps(plot)
    def _plot(*args: P.args, **kwargs: P.kwargs) -> Axes:
        """New plot function with equal aspect ratio axes returned."""
        ax = plot(*args, **kwargs)
        ax.set_aspect("equal")
        return ax

    return _plot


def _is_notebook() -> bool:
    """Detect if running in an interactive notebook environment.

    This function detects various notebook environments including:
    - Jupyter Notebook
    - JupyterLab
    - Google Colab
    - VSCode Interactive Window/Notebook
    - Other kernel-backed IPython environments

    Returns
    -------
    bool
        True if running in a notebook environment, False otherwise.
    """
    try:
        # Check for Google Colab first
        import sys

        if "google.colab" in sys.modules:
            return True

        # Check for IPython
        from IPython import get_ipython

        # Get the IPython instance
        ipython = get_ipython()
        if ipython is None:
            return False

        # Kernel-backed IPython sessions (Jupyter/VSCode notebook) expose IPKernelApp.
        if "IPKernelApp" in ipython.config:
            return True

        # ZMQInteractiveShell is used by notebook frontends; terminal IPython uses
        # TerminalInteractiveShell and should not be treated as notebook mode.
        shell_name = ipython.__class__.__name__
        shell_module = ipython.__class__.__module__
        if shell_name == "ZMQInteractiveShell" or "ipykernel" in shell_module:
            return True

        # Check if we're in VSCode's interactive window
        if "jupyter" in shell_name.lower() or "jupyter" in shell_module.lower():
            return True

        return False
    except (ImportError, AttributeError):
        return False


_trame_server_launched = False


def _ensure_trame_server_running() -> None:
    """Pre-launch PyVista's trame Jupyter server using plain ``nest_asyncio``.

    PyVista 0.47+ launches the server synchronously via
    ``pyvista.trame.jupyter.elegantly_launch``, which hard-imports
    ``nest_asyncio2``. That fork's asyncio monkey-patch is incompatible
    with Python 3.13 and crashes the kernel.

    Bring the server up ourselves with plain ``nest_asyncio`` so the
    server is already running by the time ``plotter.show()`` runs and
    PyVista's broken sync launch path is skipped.
    """
    global _trame_server_launched
    if _trame_server_launched:
        return
    try:
        import asyncio

        import nest_asyncio
        from pyvista.trame.jupyter import launch_server

        nest_asyncio.apply()
        asyncio.get_event_loop().run_until_complete(launch_server().ready)
        _trame_server_launched = True
    except Exception as exc:
        from tidy3d.log import log

        log.warning(
            "Failed to pre-launch PyVista's trame Jupyter server "
            "(%s: %s). Inline plots may fall back to a static image. "
            "Workaround: run `from pyvista.trame.jupyter import "
            "launch_server; await launch_server().ready` once at the "
            "top of your notebook.",
            type(exc).__name__,
            exc,
            log_once=True,
        )


def add_plotter_if_none(plot: Callable) -> Callable:
    """Decorates ``plot(*args, **kwargs, plotter=None)`` function for PyVista.
    If plotter=None in the function call, creates a plotter and feeds it to rest of function.

    The wrapped function should accept 'plotter' as first argument after self.
    The wrapped function should return the plotter object.

    This decorator will:
    - Auto-detect notebook environment (or use windowed parameter)
    - Create plotter if None
    - Call plotter.show() at the end if show=True and plotter was created here
    - Return the plotter object or show() result

    Parameters handled by decorator:
    - plotter: If None, creates new plotter
    - show: If True and plotter was created, calls plotter.show()
    - windowed: If None, auto-detects. If True, forces external window.
    - window_size: Tuple for window dimensions (only when creating plotter)
    """

    @wraps(plot)
    def _plot(*args: Any, **kwargs: Any) -> Any:
        """New plot function using a generated plotter if None."""
        # Pop decorator-specific parameters before any forwarding
        show = kwargs.pop("show", True)
        windowed = kwargs.pop("windowed", None)
        window_size = kwargs.pop("window_size", (800, 600))

        # Determine how plotter was passed and its current value.
        # We must distinguish three cases:
        #   1. plotter passed as keyword arg (possibly None)
        #   2. plotter passed positionally (possibly None)
        #   3. plotter not passed at all
        plotter = None
        plotter_in_kwargs = "plotter" in kwargs
        plotter_positional = False
        plotter_idx = None

        if plotter_in_kwargs:
            plotter = kwargs.pop("plotter")
        else:
            sig = inspect.signature(plot)
            params = list(sig.parameters)
            if "plotter" in params:
                plotter_idx = params.index("plotter")
                if plotter_idx < len(args):
                    plotter = args[plotter_idx]
                    plotter_positional = True

        # Determine display mode
        if windowed is None:
            # Auto-detect: windowed=False (inline) in notebooks, True otherwise
            windowed = not _is_notebook()

        # Pre-launch the trame server in notebook mode regardless of whether
        # the plotter is user-provided or created here. PyVista's broken sync
        # launch path is only skipped if the server is already running.
        if not windowed:
            _ensure_trame_server_running()

        # Track if we created the plotter
        plotter_created = plotter is None

        # Create plotter if not provided
        if plotter is None:
            pv = pyvista["mod"]
            # PyVista uses 'notebook' parameter (True=inline, False=window)
            # Our 'windowed' is opposite: True=window, False=inline
            plotter = pv.Plotter(notebook=not windowed, window_size=window_size)

        if plotter_positional:
            # Replace the positional value with the (possibly new) plotter
            args = list(args)
            args[plotter_idx] = plotter
            args = tuple(args)
        else:
            kwargs["plotter"] = plotter

        # Call the wrapped function
        plotter = plot(*args, **kwargs)

        # Show if we created the plotter and show=True
        if plotter_created and show:
            return plotter.show()

        return plotter

    return _plot


def set_default_labels_and_title(
    axis_labels: tuple[str, str],
    axis: Axis,
    position: float,
    ax: Ax,
    plot_length_units: LengthUnit | None = None,
) -> Ax:
    """Adds axis labels and title to plots involving spatial dimensions.
    When the ``plot_length_units`` are specified, the plot axes are scaled, and
    the title and axis labels include the desired units.
    """

    import matplotlib.ticker as ticker

    xlabel = axis_labels[0]
    ylabel = axis_labels[1]
    if plot_length_units is not None:
        if plot_length_units not in UnitScaling:
            raise Tidy3dKeyError(
                f"Provided units '{plot_length_units}' are not supported. "
                f"Please choose one of '{LengthUnit}'."
            )
        ax.set_xlabel(f"{xlabel} ({plot_length_units})")
        ax.set_ylabel(f"{ylabel} ({plot_length_units})")

        scale_factor = UnitScaling[plot_length_units]

        # for imperial units, use custom tick locator for nice tick positions
        if plot_length_units in ["mil", "in"]:
            UnitAwareLocator = _create_unit_aware_locator()
            x_locator = UnitAwareLocator(scale_factor)
            y_locator = UnitAwareLocator(scale_factor)
            ax.xaxis.set_major_locator(x_locator)
            ax.yaxis.set_major_locator(y_locator)

        formatter = ticker.FuncFormatter(lambda y, _: f"{y * scale_factor:.2f}")

        ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_formatter(formatter)

        position_scaled = position * scale_factor
        ax.set_title(f"cross section at {'xyz'[axis]}={position_scaled:.2f} ({plot_length_units})")
    else:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"cross section at {'xyz'[axis]}={position:.2f}")
    return ax
