"""Defines an abstract base for electromagnetic sources."""

from __future__ import annotations

from abc import ABC
from numbers import Integral
from typing import TYPE_CHECKING, Any

from pydantic import Field, field_validator

from tidy3d.components.base import cached_property
from tidy3d.components.base_sim.source import AbstractSource
from tidy3d.components.geometry.base import Box
from tidy3d.components.types import TYPE_TAG_STR
from tidy3d.components.types.time import SourceTimeType
from tidy3d.components.validators import _assert_min_freq
from tidy3d.components.viz import (
    ARROW_ALPHA,
    ARROW_COLOR_POLARIZATION,
    ARROW_COLOR_SOURCE,
    plot_params_source,
)

if TYPE_CHECKING:
    from tidy3d.components.autograd import AutogradFieldMap
    from tidy3d.components.autograd.derivative_utils import DerivativeInfo
    from tidy3d.components.types import Ax
    from tidy3d.components.viz import PlotParams


class Source(Box, AbstractSource, ABC):
    """Abstract base class for all sources.

    Notes
    -----

        **Practical Advice**

        **Choosing a Source Type**

        - ``ModeSource`` — excite a specific waveguide mode. Normalized to inject 1W at the
          center frequency. Place in a waveguide section with uniform cross-section (or
          constant bend radius). Typical: waveguides, PICs, couplers.
        - ``PlaneWave`` — uniform illumination across the full simulation cross-section.
          Requires periodic or Bloch boundaries in the tangential dimensions.
          Typical: metasurfaces, thin films, gratings.
        - ``TFSF`` — localized plane wave that can be placed inside the simulation domain
          with PML on all sides. Separates total-field (inside) from scattered-field (outside).
          Typical: nanoparticle scattering, RCS calculations.
        - ``GaussianBeam`` — focused beam with a finite waist.
          Typical: fiber coupling, free-space optics.
        - ``PointDipole`` — single-point current source for emission or LDOS calculations.
          Typical: Purcell factor, spontaneous emission.
    """

    source_time: SourceTimeType = Field(
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
        """:class:`~tidy3d.Box` representation of source."""

        return Box(center=self.center, size=self.size)

    @cached_property
    def _injection_axis(self) -> None:
        """Injection axis of the source."""
        return

    @cached_property
    def _dir_vector(self) -> None:
        """Returns a vector indicating the source direction for arrow plotting, if not None."""
        return None

    @cached_property
    def _pol_vector(self) -> None:
        """Returns a vector indicating the source polarization for arrow plotting, if not None."""
        return None

    _unsupported_traced_source_fields = ("source_time", "size")

    def _compute_derivatives(self, derivative_info: DerivativeInfo) -> AutogradFieldMap:
        """Compute adjoint derivatives for source parameters."""
        raise NotImplementedError(f"Can't compute derivative for 'Source': '{type(self)}'.")

    def _validate_traced_source_path(
        self,
        field_path: tuple[Any, ...],
        *,
        dataset_key: str,
        supported_roots: tuple[str, ...],
    ) -> None:
        """Validate traced source path against explicitly supported top-level source fields."""
        if not field_path:
            raise ValueError(f"Empty traced source path encountered in '{type(self).__name__}'.")

        field_root = field_path[0]
        supported_roots_str = ", ".join(repr(root) for root in supported_roots)
        if field_root in self._unsupported_traced_source_fields:
            raise ValueError(
                f"Automatic differentiation with respect to source field '{field_root}' is not "
                f"supported for '{type(self).__name__}'. Supported top-level fields are "
                f"{supported_roots_str}."
            )

        if field_root not in supported_roots:
            raise ValueError(
                f"Unsupported traced source path '{field_path}' for '{type(self).__name__}'. "
                f"Supported top-level fields are {supported_roots_str}."
            )

        if field_root == dataset_key:
            if len(field_path) < 2:
                raise ValueError(
                    f"Traced source path '{field_path}' for '{type(self).__name__}' is missing "
                    f"a '{dataset_key}' component key."
                )
            return

        if field_root == "center":
            if len(field_path) > 2:
                raise ValueError(
                    f"Unsupported traced source path '{field_path}' for '{type(self).__name__}'. "
                    "Only full-vector paths or single-axis paths are supported for "
                    f"'{field_root}'."
                )
            if len(field_path) == 2:
                axis = field_path[1]
                if not isinstance(axis, Integral) or int(axis) not in (0, 1, 2):
                    raise ValueError(
                        f"Unsupported axis index '{axis}' in traced source path '{field_path}'. "
                        "Axis must be one of 0, 1, 2."
                    )

    @field_validator("source_time")
    @classmethod
    def _freqs_lower_bound(cls, val: SourceTimeType) -> SourceTimeType:
        """Raise validation error if central frequency is too low."""
        _assert_min_freq(val._freq0_sigma_centroid, msg_start="'source_time' central frequency")
        return val

    def plot(
        self,
        x: float | None = None,
        y: float | None = None,
        z: float | None = None,
        ax: Ax = None,
        **patch_kwargs: Any,
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
