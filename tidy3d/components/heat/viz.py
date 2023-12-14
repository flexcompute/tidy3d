""" utilities for heat solver plotting """
from ..viz import PlotParams

""" Constants """

HEAT_BC_COLOR_TEMPERATURE = "orange"
HEAT_BC_COLOR_FLUX = "green"
HEAT_BC_COLOR_CONVECTION = "brown"
HEAT_SOURCE_CMAP = "coolwarm"

plot_params_heat_bc = PlotParams(lw=3)
plot_params_heat_source = PlotParams(edgecolor="red", lw=0, hatch="..", fill=False)
