# pylint: disable=invalid-name
""" utilities for plotting """
from typing import Any
from abc import abstractmethod
from functools import wraps

import matplotlib.pylab as plt
from matplotlib import cm
from pydantic import BaseModel

from .types import Ax
from ..constants import pec_val


def make_ax() -> Ax:
    """makes an empty `ax`."""
    _, ax = plt.subplots(1, 1, tight_layout=True)
    return ax


def add_ax_if_none(plot):
    """Decorates `plot(*args, **kwargs, ax=None)` function.
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


""" Utilities for default plotting parameters."""


class PatchParams(BaseModel):
    """Datastructure holding default parameters for plotting matplotlib.patches.
    See https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.Patch.html for explanation.
    """

    alpha: Any = None
    edgecolor: Any = None
    facecolor: Any = None
    fill: bool = True
    hatch: str = None


class PatchParamSwitcher(BaseModel):
    """Class used for updating plot kwargs based on :class:`PatchParams` values."""

    def update_params(self, **plot_params):
        """return dictionary of plot params updated with fields user supplied **plot_params dict."""

        # update self's plot params with the the supplied plot parameters and return non none ones.
        default_plot_params = self.get_plot_params()
        default_plot_params_dict = default_plot_params.dict().copy()
        default_plot_params_dict.update(plot_params)

        # get rid of pairs with value of None as they will mess up plots down the line.
        return {key: val for key, val in default_plot_params_dict.items() if val is not None}

    @abstractmethod
    def get_plot_params(self) -> PatchParams:
        """Returns :class:`PatchParams` based on user-supplied args.  Implement in subclasses."""


class GeoParams(PatchParamSwitcher):
    """Patch plotting parameters for :class:`Geometry`."""

    def get_plot_params(self) -> PatchParams:
        """Returns :class:`PatchParams` based on user-supplied args."""
        return PatchParams(edgecolor=None, facecolor="cornflowerblue")


class SourceParams(PatchParamSwitcher):
    """Patch plotting parameters for :class:`Source`."""

    def get_plot_params(self) -> PatchParams:
        """Returns :class:`PatchParams` based on user-supplied args."""
        return PatchParams(alpha=0.4, facecolor="blueviolet", edgecolor="blueviolet")


class MonitorParams(PatchParamSwitcher):
    """Patch plotting parameters for :class:`Monitor`."""

    def get_plot_params(self) -> PatchParams:
        """Returns :class:`PatchParams` based on user-supplied args."""
        return PatchParams(alpha=0.4, facecolor="crimson", edgecolor="crimson")


class StructMediumParams(PatchParamSwitcher):
    """Patch plotting parameters for :class:`Structures` in ``Simulation.plot_structures``."""

    medium: Any
    medium_map: dict

    def get_plot_params(self) -> PatchParams:
        """Returns :class:`PatchParams` based on user-supplied args."""
        mat_index = self.medium_map[self.medium]
        mat_cmap = cm.Set2  # pylint: disable=no-name-in-module, no-member
        facecolor = mat_cmap(mat_index % len(mat_cmap.colors))
        if self.medium.name == "PEC":
            return PatchParams(facecolor="black", edgecolor="black", lw=0)
        return PatchParams(facecolor=facecolor, edgecolor=facecolor, lw=0)


class StructEpsParams(PatchParamSwitcher):
    """Patch plotting parameters for :class:`Structures` in `td.Simulation.plot_structures_eps`."""

    eps: float
    eps_max: float

    def get_plot_params(self) -> PatchParams:
        """Returns :class:`PatchParams` based on user-supplied args."""
        chi = self.eps - 1.0
        chi_max = self.eps_max - 1.0
        color = 1 - chi / chi_max
        if self.eps == pec_val:
            return PatchParams(facecolor="gold", edgecolor="k", lw=1)
        return PatchParams(facecolor=str(color), edgecolor=str(color), lw=0)


class PMLParams(PatchParamSwitcher):
    """Patch plotting parameters for :class:`AbsorberSpec` (PML)."""

    def get_plot_params(self) -> PatchParams:
        """Returns :class:`PatchParams` based on user-supplied args."""
        return PatchParams(alpha=0.7, facecolor="sandybrown", edgecolor="sandybrown")


class SymParams(PatchParamSwitcher):
    """Patch plotting parameters for `td.Simulation.symmetry`."""

    sym_value: int

    def get_plot_params(self) -> PatchParams:
        """Returns :class:`PatchParams` based on user-supplied args."""
        if self.sym_value == 1:
            return PatchParams(alpha=0.3, facecolor="lightsteelblue", edgecolor="lightsteelblue")
        if self.sym_value == -1:
            return PatchParams(alpha=0.3, facecolor="lightgreen", edgecolor="lightgreen")
        return PatchParams()


class SimDataGeoParams(PatchParamSwitcher):
    """Patch plotting parameters for `td.SimulationData`."""

    def get_plot_params(self) -> PatchParams:
        """Returns :class:`PatchParams` based on user-supplied args."""
        return PatchParams(alpha=0.4, edgecolor="black")
