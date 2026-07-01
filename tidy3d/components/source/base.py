"""Defines an abstract base for electromagnetic sources."""

from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import Field, field_validator

from tidy3d.components.autograd.path_utils import AutogradRoute, format_traced_path
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
from tidy3d.exceptions import AdjointError

from .adjoint_helpers import validate_no_zero_dim_center_paths, validate_source_field_component

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

    _supported_traced_source_fields: ClassVar[tuple[str, ...]] = ()
    _traced_source_dataset_key: ClassVar[str | None] = None

    def _traced_source_support_message(self) -> str:
        """Describe which top-level source fields may be traced."""
        supported_roots = self._supported_traced_source_fields
        if supported_roots:
            supported = ", ".join(
                "'center' (non-collapsed components only)" if root == "center" else f"'{root}'"
                for root in supported_roots
            )
            return f"Supported source parameters are: {supported}."
        return "This source type does not support traced source parameters."

    def _resolve_autograd_route(self, field_path: tuple[Any, ...]) -> AutogradRoute:
        """Resolve and validate a traced source path for adjoint routing."""
        self._validate_traced_source_path(field_path)
        return AutogradRoute(local_path=field_path)

    def _compute_derivatives(self, derivative_info: DerivativeInfo) -> AutogradFieldMap:
        """Compute adjoint derivatives for source parameters."""
        raise NotImplementedError(f"Can't compute derivative for 'Source': '{type(self)}'.")

    def _validate_traced_source_path(
        self,
        field_path: tuple[Any, ...],
    ) -> None:
        """Validate traced source path against explicitly supported top-level source fields."""
        if not field_path:
            raise AdjointError(
                f"Empty traced source parameter encountered in '{type(self).__name__}'."
            )

        field_root = field_path[0]
        parameter = format_traced_path(field_path)
        supported_roots = self._supported_traced_source_fields
        if not supported_roots:
            raise AdjointError(
                f"Automatic differentiation with respect to source parameter '{parameter}' is not "
                f"supported for source type '{type(self).__name__}'. "
                f"{self._traced_source_support_message()}"
            )

        if field_root not in supported_roots:
            raise AdjointError(
                f"Unsupported traced source parameter '{parameter}' for '{type(self).__name__}'. "
                f"This parameter is not supported. {self._traced_source_support_message()}"
            )

        dataset_key = self._traced_source_dataset_key
        if field_root == dataset_key:
            if len(field_path) != 2:
                raise AdjointError(
                    f"Traced source parameter '{parameter}' for '{type(self).__name__}' is "
                    f"not supported. Use '{dataset_key}.<field_component>'."
                )
            field_name = field_path[1]
            if not isinstance(field_name, str):
                raise AdjointError(
                    f"Traced source parameter '{parameter}' for '{type(self).__name__}' is "
                    f"not supported. Use a field component such as '{dataset_key}.Ex'."
                )
            validate_source_field_component(field_name, source_name=type(self).__name__)

            dataset = getattr(self, dataset_key)
            if getattr(dataset, field_name, None) is None:
                raise AdjointError(
                    f"Traced source parameter '{parameter}' for '{type(self).__name__}' "
                    f"references a field component that is not present in '{dataset_key}'."
                )
            return

        if field_root == "center":
            validate_no_zero_dim_center_paths(
                (field_path,),
                source_size=tuple(self.size),
                source_name=type(self).__name__,
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
