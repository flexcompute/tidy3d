"""Utilities for plotting terminal component modeler elements"""

from __future__ import annotations

""" Constants """
TERMINAL_BOX_COLOR = "tab:orange"
TERMINAL_BOX_LINEWIDTH = 1.5
TERMINAL_BOX_LINESTYLE = "--"
TERMINAL_BOX_ZORDER = 10
TERMINAL_LABEL_FONTSIZE = 12
TERMINAL_LABEL_ZORDER = 11
TERMINAL_ARROW_LINEWIDTH = 0.8
TERMINAL_ARROW_ALPHA = 0.6

DIFF_PAIR_BOX_COLOR = "tab:blue"
DIFF_PAIR_BOX_LINEWIDTH = 1.5
DIFF_PAIR_BOX_LINESTYLE = "--"
DIFF_PAIR_BOX_ZORDER = 10
DIFF_PAIR_LABEL_FONTSIZE = 12
DIFF_PAIR_LABEL_ZORDER = 11
DIFF_PAIR_ARROW_LINEWIDTH = 0.8
DIFF_PAIR_ARROW_ALPHA = 0.6

PADDING_SHADE_COLOR = "dimgray"
PADDING_SHADE_ALPHA = 0.4
PADDING_SHADE_HATCH = "x"
PADDING_SHADE_ZORDER = 1000
PADDING_SHADE_EDGECOLOR = "black"
PADDING_SHADE_LINEWIDTH = 0

# Plotting parameters for terminal bounding boxes
plot_params_terminal_box = {
    "linewidth": TERMINAL_BOX_LINEWIDTH,
    "edgecolor": TERMINAL_BOX_COLOR,
    "facecolor": "none",
    "linestyle": TERMINAL_BOX_LINESTYLE,
    "zorder": TERMINAL_BOX_ZORDER,
}

# Plotting parameters for terminal annotation labels (placed above boxes)
plot_params_terminal_label = {
    "fontsize": TERMINAL_LABEL_FONTSIZE,
    "color": TERMINAL_BOX_COLOR,
    "ha": "center",
    "va": "bottom",
    "zorder": TERMINAL_LABEL_ZORDER,
    # Improve readability when labels overlap simulation geometry.
    "bbox": {
        "boxstyle": "round,pad=0.2",
        "facecolor": "white",
        "edgecolor": "none",
        "alpha": 0.7,
    },
}

# Arrow properties for terminal annotations
plot_params_terminal_arrow = {
    "arrowstyle": "-",
    "color": TERMINAL_BOX_COLOR,
    "lw": TERMINAL_ARROW_LINEWIDTH,
    "alpha": TERMINAL_ARROW_ALPHA,
    # Make leader lines look cleaner / less likely to intersect awkwardly.
    "connectionstyle": "angle3",
    "shrinkA": 2,
    "shrinkB": 2,
}

# Plotting parameters for differential pair bounding boxes
plot_params_diff_pair_box = {
    "linewidth": DIFF_PAIR_BOX_LINEWIDTH,
    "edgecolor": DIFF_PAIR_BOX_COLOR,
    "facecolor": "none",
    "linestyle": DIFF_PAIR_BOX_LINESTYLE,
    "zorder": DIFF_PAIR_BOX_ZORDER,
}

# Plotting parameters for differential pair annotation labels (placed below boxes)
plot_params_diff_pair_label = {
    "fontsize": DIFF_PAIR_LABEL_FONTSIZE,
    "color": DIFF_PAIR_BOX_COLOR,
    "ha": "center",
    "va": "top",
    "zorder": DIFF_PAIR_LABEL_ZORDER,
    "bbox": {
        "boxstyle": "round,pad=0.2",
        "facecolor": "white",
        "edgecolor": "none",
        "alpha": 0.7,
    },
}

# Arrow properties for differential pair annotations
plot_params_diff_pair_arrow = {
    "arrowstyle": "-",
    "color": DIFF_PAIR_BOX_COLOR,
    "lw": DIFF_PAIR_ARROW_LINEWIDTH,
    "alpha": DIFF_PAIR_ARROW_ALPHA,
    "connectionstyle": "angle3",
    "shrinkA": 2,
    "shrinkB": 2,
}

# Plotting parameters for padding region shading (outside port bounds)
plot_params_padding_shade = {
    "facecolor": PADDING_SHADE_COLOR,
    "alpha": PADDING_SHADE_ALPHA,
    "hatch": PADDING_SHADE_HATCH,
    "edgecolor": PADDING_SHADE_EDGECOLOR,
    "linewidth": PADDING_SHADE_LINEWIDTH,
    "zorder": PADDING_SHADE_ZORDER,
}
