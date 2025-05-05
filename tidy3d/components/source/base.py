"""Defines an abstract base for electromagnetic sources."""

from __future__ import annotations

from abc import ABC
from typing import Tuple

import pydantic.v1 as pydantic

from ..base import cached_property
from ..base_sim.source import AbstractSource
from ..geometry.base import Box
from ..types import TYPE_TAG_STR, Ax
from ..validators import _assert_min_freq, _warn_unsupported_traced_argument
from ..viz import (
    ARROW_ALPHA,
    ARROW_COLOR_POLARIZATION,
    ARROW_COLOR_SOURCE,
    PlotParams,
    plot_params_source,
)
from .time import SourceTimeType


class Source(Box, AbstractSource, ABC):
    """Abstract base class for all sources."""

    source_time: SourceTimeType = pydantic.Field(
        ...,
        title="Source Time",
        description="Specification of the source time-dependence.",
        discriminator=TYPE_TAG_STR,
    )

    @cached_property
    def plot_params(self) -> PlotParams:
        """Default parameters for plotting a Source object."""
        return plot_params_source

    @cached_property
    def geometry(self) -> Box:
        """:class:`Box` representation of source."""

        return Box(center=self.center, size=self.size)

    @cached_property
    def _injection_axis(self):
        """Injection axis of the source."""
        return None

    @cached_property
    def _dir_vector(self) -> Tuple[float, float, float]:
        """Returns a vector indicating the source direction for arrow plotting, if not None."""
        return None

    @cached_property
    def _pol_vector(self) -> Tuple[float, float, float]:
        """Returns a vector indicating the source polarization for arrow plotting, if not None."""
        return None

    _warn_traced_center = _warn_unsupported_traced_argument("center")
    _warn_traced_size = _warn_unsupported_traced_argument("size")

    @pydantic.validator("source_time", always=True)
    def _freqs_lower_bound(cls, val):
        """Raise validation error if central frequency is too low."""
        _assert_min_freq(val.freq0, msg_start="'source_time.freq0'")
        return val

    def plot(
        self,
        x: float = None,
        y: float = None,
        z: float = None,
        ax: Ax = None,
        **patch_kwargs,
    ) -> Ax:
        """Plot this source."""

        kwargs_arrow_base = patch_kwargs.pop("arrow_base", None)

        # call the `Source.plot()` function first.
        ax = Box.plot(self, x=x, y=y, z=z, ax=ax, **patch_kwargs)

        kwargs_alpha = patch_kwargs.get("alpha")
        arrow_alpha = ARROW_ALPHA if kwargs_alpha is None else kwargs_alpha

        # then add the arrow based on the propagation direction
        if self._dir_vector is not None:
            bend_radius = None
            bend_axis = None
            if hasattr(self, "mode_spec") and self.mode_spec.bend_radius is not None:
                bend_radius = self.mode_spec.bend_radius
                bend_axis = self._bend_axis
                sign = 1 if self.direction == "+" else -1
                # Curvature has to be reversed because of ploting coordinates
                if (self.size.index(0), bend_axis) in [(1, 2), (2, 0), (2, 1)]:
                    bend_radius *= -sign
                else:
                    bend_radius *= sign

            ax = self._plot_arrow(
                x=x,
                y=y,
                z=z,
                ax=ax,
                direction=self._dir_vector,
                bend_radius=bend_radius,
                bend_axis=bend_axis,
                color=ARROW_COLOR_SOURCE,
                alpha=arrow_alpha,
                both_dirs=False,
                arrow_base=kwargs_arrow_base,
            )

        if self._pol_vector is not None:
            ax = self._plot_arrow(
                x=x,
                y=y,
                z=z,
                ax=ax,
                direction=self._pol_vector,
                color=ARROW_COLOR_POLARIZATION,
                alpha=arrow_alpha,
                both_dirs=False,
                arrow_base=kwargs_arrow_base,
            )

        return ax
