""" utilities for heat solver plotting """
from ..viz import PlotParams

""" Constants """

CHARGE_BC_COLOR_POTENTIAL = "orange"
CHARGE_BC_COLOR_INSULATING = "green"
# HEAT_BC_COLOR_CONVECTION = "brown"
CHARGE_DIST_CMAP = "viridis"

plot_params_charge_bc = PlotParams(lw=3)
plot_params_charge_distribution = PlotParams(edgecolor="red", lw=0, hatch="..", fill=False)
