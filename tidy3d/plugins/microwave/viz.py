"""Utilities for plotting microwave components"""

from numpy import inf

from ...components.viz import PathPlotParams

""" Constants """
VOLTAGE_COLOR = "red"
CURRENT_COLOR = "blue"
PATH_LINEWIDTH = 2
ARROW_CURRENT = dict(
    arrowstyle="-|>",
    mutation_scale=32,
    linestyle="",
    lw=PATH_LINEWIDTH,
    color=CURRENT_COLOR,
)

plot_params_voltage_path = PathPlotParams(
    alpha=1.0,
    zorder=inf,
    color=VOLTAGE_COLOR,
    linestyle="--",
    linewidth=PATH_LINEWIDTH,
    marker="o",
    markersize=10,
    markeredgecolor=VOLTAGE_COLOR,
    markerfacecolor="white",
)

plot_params_voltage_plus = PathPlotParams(
    alpha=1.0,
    zorder=inf,
    color=VOLTAGE_COLOR,
    marker="+",
    markersize=6,
)

plot_params_voltage_minus = PathPlotParams(
    alpha=1.0,
    zorder=inf,
    color=VOLTAGE_COLOR,
    marker="_",
    markersize=6,
)

plot_params_current_path = PathPlotParams(
    alpha=1.0,
    zorder=inf,
    color=CURRENT_COLOR,
    linestyle="--",
    linewidth=PATH_LINEWIDTH,
    marker="",
)
