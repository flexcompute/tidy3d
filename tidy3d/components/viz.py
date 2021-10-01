# pylint: disable=invalid-name
""" utilities for plotting """
from typing import Any

import matplotlib.pylab as plt
from matplotlib import cm
from pydantic import BaseModel

from .types import AxesSubplot


def make_ax() -> AxesSubplot:
    """makes an empty `ax`"""
    _, ax = plt.subplots(1, 1, tight_layout=True)
    return ax


def add_ax_if_none(plot):
    """decorates `plot(ax=None)` function,
    if ax=None, creates ax and feeds it to `plot`.
    """

    def _plot(*args, **kwargs) -> AxesSubplot:
        """new `plot()` function with `ax=ax`"""
        if kwargs.get("ax") is None:
            ax = make_ax()
            kwargs["ax"] = ax
        return plot(*args, **kwargs)

    return _plot


class PatchParams(BaseModel):
    """holds parameters for matplotlib.patches
    https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.Patch.html
    """

    alpha: Any = None
    edgecolor: Any = None
    facecolor: Any = None


# """ plot parameters for structure plotting"""
# plot_params_struct = PatchParams(edgecolor="black", facecolor="cornflowerblue").dict()


def plot_params_geo_eps(eps: float, eps_max: float) -> dict:
    """plot parameters for geometry plotting with permittivity"""
    chi = eps - 1.0
    chi_max = eps_max - 1.0
    color = 1 - chi / chi_max
    return PatchParams(facecolor=str(color)).dict()


def plot_params_geo_med(medium, medium_map: dict) -> dict:
    """plot parameters for geometry plotting with medium"""
    mat_index = medium_map[medium]
    mat_cmap = cm.Set2  # pylint: disable=no-name-in-module, no-member
    facecolor = mat_cmap(mat_index % len(mat_cmap.colors))
    return PatchParams(facecolor=facecolor).dict()


def plot_params_sym(sym_val: int) -> dict:
    """plot parameters for symmetry plots of different values"""
    if sym_val == 1:
        return PatchParams(alpha=0.5, facecolor="lightsteelblue", edgecolor="lightsteelblue").dict()
    if sym_val == -1:
        return PatchParams(alpha=0.5, facecolor="lightgreen", edgecolor="lightgreen").dict()
    return {}


""" plot parameters for PML"""
plot_params_pml = PatchParams(
    alpha=0.7,
    facecolor="sandybrown",
    edgecolor="sandybrown",
).dict()

""" plot parameters for sources"""
plot_params_src = PatchParams(alpha=0.7, facecolor="blueviolet", edgecolor="blueviolet").dict()

""" plot parameters for monitors"""
plot_params_mon = PatchParams(alpha=0.7, facecolor="crimson", edgecolor="crimson").dict()
