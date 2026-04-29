"""Class and custom data array for representing a scattering matrix port based on waveguide modes."""

from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING, Any

from pydantic import Field, PositiveFloat

from tidy3d.components.data.data_array import DataArray
from tidy3d.components.geometry.base import Box
from tidy3d.components.mode_spec import ModeSpec
from tidy3d.components.monitor import (
    AstigmaticGaussianOverlapMonitor,
    GaussianOverlapMonitor,
    ModeMonitor,
)
from tidy3d.components.source.field import AstigmaticGaussianBeam, GaussianBeam, ModeSource
from tidy3d.components.source.time import GaussianPulse
from tidy3d.components.types import Direction
from tidy3d.constants import MICROMETER, RADIAN
from tidy3d.plugins.smatrix.ports.base import AbstractBasePort

if TYPE_CHECKING:
    from tidy3d.components.types.time import SourceTimeType


class ModalPortDataArray(DataArray):
    """Port parameter matrix elements for modal ports.

    Example
    -------
    >>> import numpy as np
    >>> ports_in = ['port1', 'port2']
    >>> ports_out = ['port1', 'port2']
    >>> mode_index_in = [0, 1]
    >>> mode_index_out = [0, 1]
    >>> f = [2e14]
    >>> coords = dict(
    ...     port_in=ports_in,
    ...     port_out=ports_out,
    ...     mode_index_in=mode_index_in,
    ...     mode_index_out=mode_index_out,
    ...     f=f
    ... )
    >>> fd = ModalPortDataArray((1 + 1j) * np.random.random((2, 2, 2, 2, 1)), coords=coords)
    """

    __slots__ = ()
    _dims = ("port_out", "mode_index_out", "port_in", "mode_index_in", "f")
    _data_attrs = {"long_name": "modal port matrix element"}


class AbstractPort(AbstractBasePort, Box, ABC):
    """Abstract base class for modal and Gaussian ports used in S-matrix calculations.

    Notes
    -----
        A port defines a location and a set of modes or Gaussian beams for which the S-matrix
        is calculated.
    """

    direction: Direction = Field(
        title="Direction",
        description="'+' or '-', defining which direction is considered 'input'.",
    )

    @property
    def num_modes(self) -> int:
        """Number of modes defined on this port."""
        return 1


class Port(AbstractPort):
    """Specifies a modal port for S-matrix calculation."""

    mode_spec: ModeSpec = Field(
        default_factory=ModeSpec,
        title="Mode Specification",
        description="Specifies how the mode solver will solve for the modes of the port.",
    )

    def to_monitor(self, freqs: tuple[float, ...]) -> ModeMonitor:
        """Create a ModeMonitor matching this modal port."""
        return ModeMonitor(
            center=self.center,
            size=self.size,
            freqs=freqs,
            mode_spec=self.mode_spec,
            name=self.name,
        )

    def to_source(
        self,
        freq0: float,
        fwidth: float,
        mode_index: int,
        num_freqs: int = 1,
        source_time: SourceTimeType | None = None,
        **kwargs: Any,
    ) -> ModeSource:
        """Create a ModeSource matching this modal port."""
        if source_time is None:
            source_time = GaussianPulse(freq0=freq0, fwidth=fwidth)
        return ModeSource(
            center=self.center,
            size=self.size,
            source_time=source_time,
            mode_spec=self.mode_spec,
            mode_index=mode_index,
            direction=self.direction,
            name=self.name,
            num_freqs=num_freqs,
            **kwargs,
        )

    @property
    def num_modes(self) -> int:
        """Number of modes defined on this port."""
        return self.mode_spec.num_modes


class AbstractGaussianPort(AbstractPort, ABC):
    """Abstract base for Gaussian-like ports (Gaussian and AstigmaticGaussian)."""

    angle_theta: float = Field(
        0.0,
        title="Polar Angle",
        description="Polar angle of the propagation axis from the injection axis.",
        json_schema_extra={"units": RADIAN},
    )
    angle_phi: float = Field(
        0.0,
        title="Azimuth Angle",
        description="Azimuth angle of the propagation axis in the plane orthogonal to the injection axis.",
        json_schema_extra={"units": RADIAN},
    )
    pol_angle: float = Field(
        0.0,
        title="Polarization Angle",
        description="Angle between E-field polarization and the plane defined by the injection axis and propagation axis. "
        "0 => P polarization, pi/2 => S polarization.",
        json_schema_extra={"units": RADIAN},
    )


class GaussianPort(AbstractGaussianPort):
    """Specifies a Gaussian port for S-matrix calculation."""

    waist_radius: PositiveFloat = Field(
        1.0,
        title="Waist Radius",
        description="Radius of the beam at the waist.",
        json_schema_extra={"units": MICROMETER},
    )
    waist_distance: float = Field(
        0.0,
        title="Waist Distance",
        description="Distance from the beam waist along the propagation direction. "
        "A positive value places the waist behind the port plane (toward the negative normal axis). "
        "A negative value places the waist in front of the port plane. "
        "This definition is independent of the ``direction`` parameter.",
        json_schema_extra={"units": MICROMETER},
    )

    def to_monitor(self, freqs: tuple[float, ...]) -> GaussianOverlapMonitor:
        """Create a GaussianOverlapMonitor matching this gaussian port."""
        return GaussianOverlapMonitor(
            center=self.center,
            size=self.size,
            freqs=freqs,
            angle_theta=self.angle_theta,
            angle_phi=self.angle_phi,
            pol_angle=self.pol_angle,
            waist_radius=self.waist_radius,
            waist_distance=self.waist_distance,
            name=self.name,
        )

    def to_source(
        self,
        freq0: float,
        fwidth: float,
        mode_index: int = 0,
        num_freqs: int = 1,
        source_time: SourceTimeType | None = None,
        **kwargs: Any,
    ) -> GaussianBeam:
        """Create a GaussianBeam matching this gaussian port. Ignores mode_index."""
        if source_time is None:
            source_time = GaussianPulse(freq0=freq0, fwidth=fwidth)
        return GaussianBeam(
            center=self.center,
            size=self.size,
            source_time=source_time,
            angle_theta=self.angle_theta,
            angle_phi=self.angle_phi,
            pol_angle=self.pol_angle,
            waist_radius=self.waist_radius,
            waist_distance=self.waist_distance,
            direction=self.direction,
            name=self.name,
            num_freqs=num_freqs,
            **kwargs,
        )


class AstigmaticGaussianPort(AbstractGaussianPort):
    """Specifies an astigmatic Gaussian port for S-matrix calculation."""

    waist_sizes: tuple[PositiveFloat, PositiveFloat] = Field(
        (1.0, 1.0),
        title="Waist sizes",
        description="Size of the beam at the waist in the local x and y directions.",
        json_schema_extra={"units": MICROMETER},
    )
    waist_distances: tuple[float, float] = Field(
        (0.0, 0.0),
        title="Waist distances",
        description="Distance to the beam waist along the propagation direction "
        "for the waist sizes in the local x and y directions. "
        "Positive values place the waist behind the port plane (toward the negative normal axis); "
        "negative values place the waist in front of the port plane. "
        "This definition is independent of the ``direction`` parameter.",
        json_schema_extra={"units": MICROMETER},
    )

    def to_monitor(self, freqs: tuple[float, ...]) -> AstigmaticGaussianOverlapMonitor:
        """Create an AstigmaticGaussianOverlapMonitor matching this port."""
        return AstigmaticGaussianOverlapMonitor(
            center=self.center,
            size=self.size,
            freqs=freqs,
            angle_theta=self.angle_theta,
            angle_phi=self.angle_phi,
            pol_angle=self.pol_angle,
            waist_sizes=self.waist_sizes,
            waist_distances=self.waist_distances,
            name=self.name,
        )

    def to_source(
        self,
        freq0: float,
        fwidth: float,
        mode_index: int = 0,
        num_freqs: int = 1,
        source_time: SourceTimeType | None = None,
        **kwargs: Any,
    ) -> AstigmaticGaussianBeam:
        """Create an AstigmaticGaussianBeam matching this port. Ignores mode_index."""
        if source_time is None:
            source_time = GaussianPulse(freq0=freq0, fwidth=fwidth)
        return AstigmaticGaussianBeam(
            center=self.center,
            size=self.size,
            source_time=source_time,
            angle_theta=self.angle_theta,
            angle_phi=self.angle_phi,
            pol_angle=self.pol_angle,
            waist_sizes=self.waist_sizes,
            waist_distances=self.waist_distances,
            direction=self.direction,
            name=self.name,
            num_freqs=num_freqs,
            **kwargs,
        )


ModalPortType = Port | GaussianPort | AstigmaticGaussianPort
