from __future__ import annotations

from functools import wraps
from typing import Any, Optional

from tidy3d.components.types import Ax, Axis, LengthUnit
from tidy3d.constants import UnitScaling
from tidy3d.exceptions import Tidy3dKeyError


def _create_unit_aware_locator():
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

        def __call__(self):
            vmin, vmax = self.axis.get_view_interval()
            return self.tick_values(vmin, vmax)

        def view_limits(self, vmin, vmax):
            """Override to prevent matplotlib from adjusting our limits."""
            return vmin, vmax

        def tick_values(self, vmin, vmax):
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
    import matplotlib.pyplot as plt

    _, ax = plt.subplots(1, 1, tight_layout=True)
    return ax


def add_ax_if_none(plot):
    """Decorates ``plot(*args, **kwargs, ax=None)`` function.
    if ax=None in the function call, creates an ax and feeds it to rest of function.
    """

    @wraps(plot)
    def _plot(*args: Any, **kwargs: Any) -> Ax:
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
    def _plot(*args: Any, **kwargs: Any) -> Ax:
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
