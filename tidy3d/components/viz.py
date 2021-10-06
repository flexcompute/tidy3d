# pylint: disable=invalid-name
""" utilities for plotting """
from typing import Any
from abc import abstractmethod
from functools import wraps

import matplotlib.pylab as plt
from matplotlib import cm
from pydantic import BaseModel

from .types import Ax


def make_ax() -> Ax:
    """makes an empty `ax`"""
    _, ax = plt.subplots(1, 1, tight_layout=True)
    return ax


def add_ax_if_none(plot):
    """decorates `plot(ax=None)` function,
    if ax=None, creates ax and feeds it to `plot`.
    """

    @wraps(plot)
    def _plot(*args, **kwargs) -> Ax:
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
    fill: bool = True


class PatchParamSwitcher(BaseModel):
    """base class for updating parameters based on default values"""

    def update_params(self, **plot_params):
        """return dictionary of plot params updated with fields user supplied **plot_params dict"""
        default_plot_params = self.get_plot_params()
        default_plot_params_dict = default_plot_params.dict().copy()
        default_plot_params_dict.update(plot_params)

        # get rid of pairs with value of None as they will mess up plots down the line
        return {key: val for key, val in default_plot_params_dict.items() if val is not None}

    @abstractmethod
    def get_plot_params(self) -> PatchParams:
        """returns PatchParams based on attributes of self"""


class GeoParams(PatchParamSwitcher):
    """Patch plotting parameters for td.Geometry"""

    def get_plot_params(self) -> PatchParams:
        """returns PatchParams based on attributes of self"""
        return PatchParams(edgecolor="black", facecolor="cornflowerblue")


class SourceParams(PatchParamSwitcher):
    """Patch plotting parameters for `td.Source`"""

    def get_plot_params(self) -> PatchParams:
        """returns PatchParams based on attributes of self"""
        return PatchParams(alpha=0.7, facecolor="blueviolet", edgecolor="blueviolet")


class MonitorParams(PatchParamSwitcher):
    """Patch plotting parameters for `td.Monitor`"""

    def get_plot_params(self) -> PatchParams:
        """returns PatchParams based on attributes of self"""
        return PatchParams(alpha=0.7, facecolor="crimson", edgecolor="crimson")


class StructMediumParams(PatchParamSwitcher):
    """Patch plotting parameters for `td.Structures in `td.Simulation.plot_structures`"""

    medium: Any
    medium_map: dict

    def get_plot_params(self) -> PatchParams:
        """returns PatchParams based on attributes of self"""
        mat_index = self.medium_map[self.medium]
        mat_cmap = cm.Set2  # pylint: disable=no-name-in-module, no-member
        facecolor = mat_cmap(mat_index % len(mat_cmap.colors))
        return PatchParams(facecolor=facecolor)


class StructEpsParams(PatchParamSwitcher):
    """Patch plotting parameters for `td.Structures in `td.Simulation.plot_structures_eps`"""

    eps: float
    eps_max: float

    def get_plot_params(self) -> PatchParams:
        """returns PatchParams based on attributes of self"""
        chi = self.eps - 1.0
        chi_max = self.eps_max - 1.0
        color = 1 - chi / chi_max
        return PatchParams(facecolor=str(color))


class PMLParams(PatchParamSwitcher):
    """Patch plotting parameters for `td.Simulation.pml_layers`"""

    def get_plot_params(self) -> PatchParams:
        """returns PatchParams based on attributes of self"""
        return PatchParams(alpha=0.7, facecolor="sandybrown", edgecolor="sandybrown")


class SymParams(PatchParamSwitcher):
    """Patch plotting parameters for `td.Simulation.symmetry`"""

    sym_value: int

    def get_plot_params(self) -> PatchParams:
        """returns PatchParams based on attributes of self"""
        if self.sym_value == 1:
            return PatchParams(alpha=0.5, facecolor="lightsteelblue", edgecolor="lightsteelblue")
        if self.sym_value == -1:
            return PatchParams(alpha=0.5, facecolor="lightgreen", edgecolor="lightgreen")
        return PatchParams()


class SimDataGeoParams(PatchParamSwitcher):
    """Patch plotting parameters for `td.Simulation.symmetry`"""

    def get_plot_params(self) -> PatchParams:
        """returns PatchParams based on attributes of self"""
        return PatchParams(alpha=0.4, edgecolor="black")
