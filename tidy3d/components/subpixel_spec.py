# Defines specifications for subpixel averaging
from __future__ import annotations

from typing import Union

import pydantic.v1 as pd

from .base import Tidy3dBaseModel, cached_property
from .types import TYPE_TAG_STR

# Default Courant number reduction rate in PEC conformal's scheme
DEFAULT_COURANT_REDUCTION_PEC_CONFORMAL = 0.3

# Default Courant number reduction rate in Surface impedance conformal's scheme
DEFAULT_COURANT_REDUCTION_SIBC_CONFORMAL = 0.0


class AbstractSubpixelAveragingMethod(Tidy3dBaseModel):
    """Base class defining how to handle material assignment on structure interfaces."""

    @cached_property
    def courant_ratio(self) -> float:
        """The scaling ratio applied to Courant number so that the courant number
        in the simulation is ``sim.courant * courant_ratio``.
        """
        return 1.0


class Staircasing(AbstractSubpixelAveragingMethod):
    """Apply staircasing scheme to material assignment of Yee grids on structure boundaries.

    Note
    ----
    For PEC interface, the algorithm is based on:

        A. Taflove and S. C. Hagness, "Computational electromagnetics: the
        finite-difference time-domain method", Chapter 10.3 (2005).
    """


class PolarizedAveraging(AbstractSubpixelAveragingMethod):
    """Apply a polarized subpixel averaging method to dielectric boundaries, which
    is a phenomenological approximation of :class:`.ContourPathAveraging`.

    Note
    ----
    The algorithm is based on:

        A. Mohammadi, H. Nadgaran and M. Agio, "Contour-path effective
        permittivities for the two-dimensional finite-difference
        time-domain method", Optics express, 13(25), 10367-10381 (2005).
    """


class ContourPathAveraging(AbstractSubpixelAveragingMethod):
    """Apply a contour-path subpixel averaging method to dielectric boundaries.

    Note
    ----
    The algorithm is based on:

        A. Mohammadi, H. Nadgaran and M. Agio, "Contour-path effective
        permittivities for the two-dimensional finite-difference
        time-domain method", Optics express, 13(25), 10367-10381 (2005).
    """


DielectricSubpixelType = Union[Staircasing, PolarizedAveraging, ContourPathAveraging]


class VolumetricAveraging(AbstractSubpixelAveragingMethod):
    """Apply volumetric averaging scheme to material properties of Yee grids on structure boundaries.
    The material property is averaged in the volume surrounding the Yee grid.
    """

    staircase_normal_component: bool = pd.Field(
        True,
        title="Staircasing For Field Components Substantially Normal To Interface",
        description="Volumetric averaging works accurately if the electric field component "
        "is substantially tangential to the interface. If ``True``, apply volumetric averaging only "
        "if the field component is largely tangential to the interface; if ``False``, apply volumetric "
        "averaging regardless of how field component orients with the interface.",
    )


MetalSubpixelType = Union[Staircasing, VolumetricAveraging]


class HeuristicPECStaircasing(AbstractSubpixelAveragingMethod):
    """Apply a variant of staircasing scheme to PEC boundaries: the electric field grid is set to PEC
    if the field is substantially parallel to the interface.
    """


class PECConformal(AbstractSubpixelAveragingMethod):
    """Apply a subpixel averaging method known as conformal mesh scheme to PEC boundaries.

    Note
    ----
    The algorithm is based on:

        S. Dey and R. Mittra, "A locally conformal finite-difference
        time-domain (FDTD) algorithm for modeling three-dimensional
        perfectly conducting objects",
        IEEE Microwave and Guided Wave Letters, 7(9), 273 (1997).

        S. Benkler, N. Chavannes and N. Kuster, "A new 3-D conformal
        PEC FDTD scheme with user-defined geometric precision and derived
        stability criterion",
        IEEE Transactions on Antennas and Propagation, 54(6), 1843 (2006).
    """

    timestep_reduction: float = pd.Field(
        DEFAULT_COURANT_REDUCTION_PEC_CONFORMAL,
        title="Time Step Size Reduction Rate",
        description="Reduction factor between 0 and 1 such that the simulation's time step size "
        "is ``1 - timestep_reduction`` times its default value. "
        "Accuracy can be improved with a smaller time step size, but the simulation time will be increased.",
        lt=1,
        ge=0,
    )

    @cached_property
    def courant_ratio(self) -> float:
        """The scaling ratio applied to Courant number so that the courant number
        in the simulation is ``sim.courant * courant_ratio``.
        """
        return 1 - self.timestep_reduction


PECSubpixelType = Union[Staircasing, HeuristicPECStaircasing, PECConformal]


class SurfaceImpedance(PECConformal):
    """Apply 1st order (Leontovich) surface impedance boundary condition to
    structure made of :class:`.LossyMetalMedium`.
    """

    timestep_reduction: float = pd.Field(
        DEFAULT_COURANT_REDUCTION_SIBC_CONFORMAL,
        title="Time Step Size Reduction Rate",
        description="Reduction factor between 0 and 1 such that the simulation's time step size "
        "is ``1 - timestep_reduction`` times its default value. "
        "Accuracy can be improved with a smaller time step size, but the simulation time will be increased.",
        lt=1,
        ge=0,
    )


LossyMetalSubpixelType = Union[Staircasing, VolumetricAveraging, SurfaceImpedance]


class SubpixelSpec(Tidy3dBaseModel):
    """Defines specification for subpixel averaging schemes when added to ``Simulation.subpixel``."""

    dielectric: DielectricSubpixelType = pd.Field(
        PolarizedAveraging(),
        title="Subpixel Averaging Method For Dielectric Interfaces",
        description="Subpixel averaging method applied to dielectric material interfaces.",
        discriminator=TYPE_TAG_STR,
    )

    metal: MetalSubpixelType = pd.Field(
        Staircasing(),
        title="Subpixel Averaging Method For Metallic Interfaces",
        description="Subpixel averaging method applied to metallic structure interfaces. "
        "A material is considered as metallic if its real part of relative permittivity "
        "is less than 1 at the central frequency.",
        discriminator=TYPE_TAG_STR,
    )

    pec: PECSubpixelType = pd.Field(
        PECConformal(),
        title="Subpixel Averaging Method For PEC Interfaces",
        description="Subpixel averaging method applied to PEC structure interfaces.",
        discriminator=TYPE_TAG_STR,
    )

    lossy_metal: LossyMetalSubpixelType = pd.Field(
        SurfaceImpedance(),
        title="Subpixel Averaging Method for Lossy Metal Interfaces",
        description="Subpixel averaging method applied to ``td.LossyMetalMedium`` material interfaces.",
        discriminator=TYPE_TAG_STR,
    )

    @classmethod
    def staircasing(cls) -> SubpixelSpec:
        """Apply staircasing on all material boundaries."""
        return cls(
            dielectric=Staircasing(),
            metal=Staircasing(),
            pec=Staircasing(),
            lossy_metal=Staircasing(),
        )

    def courant_ratio(self, contain_pec_structures: bool, contain_sibc_structures: bool) -> float:
        """The scaling ratio applied to Courant number so that the courant number
        in the simulation is ``sim.courant * courant_ratio``. So far only PEC subpixel averaging
        scheme and SIBC require deduction of Courant number.
        """
        if contain_pec_structures and contain_sibc_structures:
            return min(self.pec.courant_ratio, self.lossy_metal.courant_ratio)
        if contain_pec_structures:
            return self.pec.courant_ratio
        if contain_sibc_structures:
            return self.lossy_metal.courant_ratio
        return 1.0
