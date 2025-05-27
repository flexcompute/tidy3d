from __future__ import annotations

try:
    from matplotlib.patches import ArrowStyle

    arrow_style = ArrowStyle.Simple(head_length=11, head_width=9, tail_width=4)
except ImportError:
    arrow_style = None

FLEXCOMPUTE_COLORS = {
    "brand_green": "#00643C",
    "brand_tan": "#B8A18B",
    "brand_blue": "#6DB5DD",
    "brand_purple": "#8851AD",
    "brand_black": "#000000",
    "brand_orange": "#FC7A4C",
}
ARROW_COLOR_SOURCE = FLEXCOMPUTE_COLORS["brand_green"]
ARROW_COLOR_POLARIZATION = FLEXCOMPUTE_COLORS["brand_tan"]
ARROW_COLOR_MONITOR = FLEXCOMPUTE_COLORS["brand_orange"]
PLOT_BUFFER = 0.3
ARROW_ALPHA = 0.8
ARROW_LENGTH = 0.3

# stores color of simulation.structures for given index in simulation.medium_map
MEDIUM_CMAP = [
    "#689DBC",
    "#D0698E",
    "#5E6EAD",
    "#C6224E",
    "#BDB3E2",
    "#9EC3E0",
    "#616161",
    "#877EBC",
]

# colormap for structure's permittivity in plot_eps
STRUCTURE_EPS_CMAP = "gist_yarg"
STRUCTURE_EPS_CMAP_R = "gist_yarg_r"
STRUCTURE_HEAT_COND_CMAP = "gist_yarg"
