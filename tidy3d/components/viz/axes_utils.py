from __future__ import annotations

from functools import wraps
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from tidy3d.components.types import Ax, Axis, LengthUnit
from tidy3d.constants import UnitScaling
from tidy3d.exceptions import Tidy3dKeyError


def make_ax() -> Ax:
    """makes an empty ``ax``."""
    _, ax = plt.subplots(1, 1, tight_layout=True)
    return ax


def add_ax_if_none(plot):
    """Decorates ``plot(*args, **kwargs, ax=None)`` function.
    if ax=None in the function call, creates an ax and feeds it to rest of function.
    """

    @wraps(plot)
    def _plot(*args, **kwargs) -> Ax:
        """New plot function using a generated ax if None."""
        if kwargs.get("ax") is None:
            ax = make_ax()
            kwargs["ax"] = ax
        return plot(*args, **kwargs)

    return _plot


def equal_aspect(plot):
    """Decorates a plotting function returning a matplotlib axes.
    Ensures the aspect ratio of the returned axes is set to equal.
    Useful for 2D plots, like sim.plot() or sim_data.plot_fields()
    """

    @wraps(plot)
    def _plot(*args, **kwargs) -> Ax:
        """New plot function with equal aspect ratio axes returned."""
        ax = plot(*args, **kwargs)
        ax.set_aspect("equal")
        return ax

    return _plot


def set_default_labels_and_title(
    axis_labels: tuple[str, str],
    axis: Axis,
    position: float,
    ax: Ax,
    plot_length_units: Optional[LengthUnit] = None,
) -> Ax:
    """Adds axis labels and title to plots involving spatial dimensions.
    When the ``plot_length_units`` are specified, the plot axes are scaled, and
    the title and axis labels include the desired units.
    """
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
        # Formatter to help plot in arbitrary units
        scale_factor = UnitScaling[plot_length_units]
        formatter = ticker.FuncFormatter(lambda y, _: f"{y * scale_factor:.2f}")
        ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_formatter(formatter)
        ax.set_title(
            f"cross section at {'xyz'[axis]}={position * scale_factor:.2f} ({plot_length_units})"
        )
    else:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"cross section at {'xyz'[axis]}={position:.2f}")
    return ax
