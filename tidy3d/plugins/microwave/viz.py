"""Utilities for plotting microwave components"""

from __future__ import annotations

from numpy import inf

from tidy3d.components.viz import PathPlotParams

""" Constants """
LOBE_PEAK_COLOR = "tab:red"
LOBE_WIDTH_COLOR = "tab:orange"
LOBE_FNBW_COLOR = "tab:blue"

plot_params_lobe_peak = PathPlotParams(
    alpha=1.0,
    zorder=inf,
    color=LOBE_PEAK_COLOR,
    linestyle="-",
    linewidth=1,
    marker="",
)

plot_params_lobe_width = PathPlotParams(
    alpha=1.0,
    zorder=inf,
    color=LOBE_WIDTH_COLOR,
    linestyle="--",
    linewidth=1,
    marker="",
)

plot_params_lobe_FNBW = PathPlotParams(
    alpha=1.0,
    zorder=inf,
    color=LOBE_FNBW_COLOR,
    linestyle=":",
    linewidth=1,
    marker="",
)
