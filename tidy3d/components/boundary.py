"""Defines electromagnetic boundary conditions"""

from __future__ import annotations

from abc import ABC
from typing import List, Tuple, Union

import numpy as np
import pydantic.v1 as pd

from ..constants import EPSILON_0, MU_0, PML_SIGMA
from ..exceptions import DataError, SetupError
from ..log import log
from .base import Tidy3dBaseModel, cached_property
from .medium import Medium
from .source.field import TFSF, GaussianBeam, ModeSource, PlaneWave
from .types import TYPE_TAG_STR, Axis, Complex

MIN_NUM_PML_LAYERS = 6


class BoundaryEdge(ABC, Tidy3dBaseModel):
    """Electromagnetic boundary condition at a domain edge."""

    name: str = pd.Field(None, title="Name", description="Optional unique name for boundary.")


# PBC keyword
class Periodic(BoundaryEdge):
    """Periodic boundary condition class."""

    @property
    def bloch_vec(self):
        """Periodic boundaries are effectively Bloch boundaries with ``bloch_vec == 0``.
        In practice, periodic boundaries do not force the use of complex fields, while Bloch
        boundaries do, even with ``bloch_vec == 0``. Thus, it is more efficient to use periodic.
        """
        return 0


# PEC keyword
class PECBoundary(BoundaryEdge):
    """Perfect electric conductor boundary condition class."""


# PMC keyword
class PMCBoundary(BoundaryEdge):
    """Perfect magnetic conductor boundary condition class."""


# """ Bloch boundary """

# sources from which Bloch boundary conditions can be defined
BlochSourceType = Union[GaussianBeam, ModeSource, PlaneWave, TFSF]


class BlochBoundary(BoundaryEdge):
    """Specifies a Bloch boundary condition along a single dimension.

    Example
    -------
    >>> bloch = BlochBoundary(bloch_vec=1)

    See Also
    --------

    **Notebooks**:
        * `Defining a total-field scattered-field (TFSF) plane wave source <../../notebooks/TFSF.html>`_
        * `Multilevel blazed diffraction grating <../../notebooks/GratingEfficiency.html>`_
    """

    bloch_vec: float = pd.Field(
        ...,
        title="Normalized Bloch vector component",
        description="Normalized component of the Bloch vector in units of "
        "2 * pi / (size along dimension) in the background medium, "
        "along the dimension in which the boundary is specified.",
    )

    @cached_property
    def bloch_phase(self) -> Complex:
        """Returns the forward phase factor associated with ``bloch_vec``."""
        return np.exp(1j * 2.0 * np.pi * self.bloch_vec)

    @classmethod
    def from_source(
        cls, source: BlochSourceType, domain_size: float, axis: Axis, medium: Medium = None
    ) -> BlochBoundary:
        """Set the Bloch vector component based on a given angled source and its center frequency.
           Note that if a broadband angled source is used, only the frequency components near the
           center frequency will exhibit angled incidence at the expect angle. In this case, a
           narrowband source is recommended.

        Parameters
        ----------
        source : Union[:class:`GaussianBeam`, :class:`ModeSource`, :class:`PlaneWave`]
            Angled source.
        domain_size: float
            Size of the domain (micron) in the direction normal to the Bloch boundary.
        axis: int
            Axis normal to the Bloch boundary.
        medium : :class:`.Medium`
            Background medium associated with the Bloch vector.
            Default: free space.

        Returns
        -------
        :class:`BlochBoundary`
            Bloch boundary condition with wave vector defined based on the source angles
            and center frequency.

        Example
        -------
        >>> from tidy3d import GaussianPulse, PlaneWave, inf
        >>> pulse = GaussianPulse(freq0=200e12, fwidth=20e12)
        >>> pw_source = PlaneWave(
        ...     size=(inf,inf,0), source_time=pulse, direction='+', angle_theta=0.2, angle_phi=0.3)
        >>> bloch = BlochBoundary.from_source(source=pw_source, domain_size=5, axis=0)
        """

        if not isinstance(source, BlochSourceType.__args__):
            raise SetupError(
                "The 'source' parameter must be 'GaussianBeam', 'ModeSource', 'PlaneWave', "
                "or 'TFSF' in order to define a Bloch boundary condition."
            )

        if axis == source.injection_axis:
            raise SetupError(
                "Bloch boundary axis must be orthogonal to the injection axis of 'source'."
            )

        if medium is None:
            medium = Medium(permittivity=1.0, name="free_space")

        freq0 = source.source_time.freq0
        eps_complex = medium.eps_model(freq0)
        kmag = np.real(freq0 * np.sqrt(eps_complex * EPSILON_0 * MU_0))

        angle_theta = source.angle_theta
        angle_phi = source.angle_phi

        # handle the special case to avoid tiny numerical fluctuations
        if angle_theta == 0:
            return cls(bloch_vec=0)

        if source.direction == "-":
            angle_theta += np.pi
            # angle_phi += np.pi
            # kmag *= -1.0

        k_local = [
            kmag * np.sin(angle_theta) * np.cos(angle_phi),
            kmag * np.sin(angle_theta) * np.sin(angle_phi),
            kmag * np.cos(angle_theta),
        ]

        k_global = source.unpop_axis(
            k_local[2], (k_local[0], k_local[1]), axis=source.injection_axis
        )

        bloch_vec = domain_size * k_global[axis]
        return cls(bloch_vec=bloch_vec)


# """ Absorber parameters """


class AbsorberParams(Tidy3dBaseModel):
    """Specifies parameters common to Absorbers and PMLs.

    Example
    -------
    >>> params = AbsorberParams(sigma_order=3, sigma_min=0.0, sigma_max=1.5)
    """

    sigma_order: pd.NonNegativeInt = pd.Field(
        3,
        title="Sigma Order",
        description="Order of the polynomial describing the absorber profile (~dist^sigma_order).",
    )

    sigma_min: pd.NonNegativeFloat = pd.Field(
        0.0,
        title="Sigma Minimum",
        description="Minimum value of the absorber conductivity.",
        units=PML_SIGMA,
    )

    sigma_max: pd.NonNegativeFloat = pd.Field(
        1.5,
        title="Sigma Maximum",
        description="Maximum value of the absorber conductivity.",
        units=PML_SIGMA,
    )


class PMLParams(AbsorberParams):
    """Specifies full set of parameters needed for complex, frequency-shifted PML.

    Example
    -------
    >>> params = PMLParams(sigma_order=3, sigma_min=0.0, sigma_max=1.5, kappa_min=0.0)
    """

    kappa_order: pd.NonNegativeInt = pd.Field(
        3,
        title="Kappa Order",
        description="Order of the polynomial describing the PML kappa profile "
        "(kappa~dist^kappa_order).",
    )

    kappa_min: pd.NonNegativeFloat = pd.Field(0.0, title="Kappa Minimum", description="")

    kappa_max: pd.NonNegativeFloat = pd.Field(1.5, title="Kappa Maximum", description="")

    alpha_order: pd.NonNegativeInt = pd.Field(
        3,
        title="Alpha Order",
        description="Order of the polynomial describing the PML alpha profile "
        "(alpha~dist^alpha_order).",
    )

    alpha_min: pd.NonNegativeFloat = pd.Field(
        0.0, title="Alpha Minimum", description="Minimum value of the PML alpha.", units=PML_SIGMA
    )

    alpha_max: pd.NonNegativeFloat = pd.Field(
        1.5, title="Alpha Maximum", description="Maximum value of the PML alpha.", units=PML_SIGMA
    )


""" Default absorber parameters """

DefaultAbsorberParameters = AbsorberParams(sigma_order=3, sigma_min=0.0, sigma_max=6.4)
DefaultPMLParameters = PMLParams(
    sigma_order=3,
    sigma_min=0.0,
    sigma_max=1.5,
    kappa_order=3,
    kappa_min=1.0,
    kappa_max=3.0,
    alpha_order=1,
    alpha_min=0.0,
    alpha_max=0.0,
)
DefaultStablePMLParameters = PMLParams(
    sigma_order=3,
    sigma_min=0.0,
    sigma_max=1.0,
    kappa_order=3,
    kappa_min=1.0,
    kappa_max=5.0,
    alpha_order=1,
    alpha_min=0.0,
    alpha_max=0.9,
)


""" Absorber specifications """


class AbsorberSpec(BoundaryEdge):
    """Specifies the generic absorber properties along a single dimension."""

    num_layers: int = pd.Field(
        ...,
        title="Number of Layers",
        description="Number of layers of standard PML.",
        ge=MIN_NUM_PML_LAYERS,
    )
    parameters: AbsorberParams = pd.Field(
        ...,
        title="Absorber Parameters",
        description="Parameters to fine tune the absorber profile and properties.",
    )


class PML(AbsorberSpec):
    """Specifies a standard PML along a single dimension.

    Notes
    ------

        **1D Model Illustration**

        Consider a transformed wave equation in the :math:`x` dimension below _`[1]`:

        .. math::

             \\left( \\left( \\frac{1}{s(x)} \\frac{\\delta}{\\delta x} \\right)^2 - \\frac{1}{c^2} \\frac{\\delta^2}{\\delta t^2} \\right) E = 0

        where the wave stretch factor :math:`s(x)` depends on the PML boundary position in the :math:`x` dimension.

        .. TODO what is x at 0?

        .. math::

             s(x) = \\left \\{
                        \\begin{array}{lr}
                            1, & \\text{for } x < 0 \\\\
                            1 - \\frac{\\sigma}{i \\omega \\epsilon_0}, & \\text{for } x > 0
                        \\end{array}
                    \\right \\}

        The wave equation can be solved and plotted accordingly as a function of the :math:`x` dimension.

        .. math::

             E(x) = \\left \\{
                        \\begin{array}{lr}
                            e^{i(kx - \\omega t)}, & \\text{for } x < 0 \\\\
                            e^{i(kx - \\omega t)} \\times e^{-\\frac{\\sigma x}{c \\epsilon_0}} & \\text{for } x > 0
                        \\end{array}
                    \\right \\}

        Hence, we see how this PML stretch factor induces frequency-independent exponential attentation and no
        reflection after the boundary at :math:`x=0`.

        .. image:: ../../_static/img/pml_boundary.png

        .. TODO make this image better

        **Usage Caveats**

        A perfectly matched layer (PML) is the most commonly used boundary condition in FDTD simulations to truncate
        a simulation domain and absorb outgoing radiation. However, many divergence issues are associated with the
        use of PML. One of the most common causes of a diverged simulation is structures inserted into PML at an angle.

        .. TODO links to absorber boundaries

        .. image:: ../../notebooks/img/diverged-fdtd-simulation.png

        Incorporating a dispersive material into the PML can also cause simulation divergence in certain scenarios.
        If your simulation lacks any structures inserted into the PML at an angle, but includes dispersive material
        in PML, it is advisable to substitute a nondispersive material for the dispersive material. Alternatively,
        if dispersion is necessary, switching from the :class:`PML` to :class:`Absorber` can effectively address the
        issue.

        The PML can effectively absorb outgoing radiation with minimum reflection as if the radiation just propagates
        into the free space. However, it’s important to keep in mind that the PML only absorbs propagating fields. For
        evanescent fields, the PML can act as an amplification medium and cause a simulation to diverge. In Tidy3D,
        a warning will appear if the distance between a structure is smaller than half of a wavelength to prevent
        evanescent fields from leaking into PML. In most cases, the evanescent field will naturally die off within
        half a wavelength, but in some instances, a larger distance may be required.

        .. image:: ../../notebooks/img/diverged-fdtd-simulation1.png


        **References**

        .. [1]  W.C. Chew and W.H. Weedon, Microwave and Optical Tech. Lett., 7 (13), 599,1994; S. Johnson, arXiv 2108.05348, 2021
        .. [2]  Antonios Giannopoulos, IEEE Transactions on Antennas and Propagation, 56(9), 2995, 2008

    Note
    ----

        For best results, structures that intersect with the PML or simulation edges should extend extend all the way
        through. In many such cases, an “infinite” size ``td.inf`` can be used to define the size along that dimension.

    Example
    -------
    >>> pml = PML(num_layers=10)

    See Also
    --------

    :class:`StablePML`:
         This PML deals handles possibly divergent simulations better, but at the expense of more layers.

    :class:`Absorber`:
         Specifies an adiabatic absorber along a single dimension.

    **Notebooks:**
        * `How to troubleshoot a diverged FDTD simulation <../../notebooks/DivergedFDTDSimulation.html>`_

    **Lectures:**
        * `Using FDTD to Compute a Transmission Spectrum <https://www.flexcompute.com/fdtd101/Lecture-2-Using-FDTD-to-Compute-a-Transmission-Spectrum/>`__
        * `Introduction to perfectly matched layer (PML) tutorial <https://www.flexcompute.com/fdtd101/Lecture-6-Introduction-to-perfectly-matched-layer/>`__

    """

    num_layers: int = pd.Field(
        12,
        title="Number of Layers",
        description="Number of layers of standard PML.",
        ge=MIN_NUM_PML_LAYERS,
    )

    parameters: PMLParams = pd.Field(
        DefaultPMLParameters,
        title="PML Parameters",
        description="Parameters of the complex frequency-shifted absorption poles.",
    )


class StablePML(AbsorberSpec):
    """Specifies a 'stable' PML along a single dimension.
    This PML deals handles possibly divergent simulations better, but at the expense of more layers.

    Example
    -------
    >>> pml = StablePML(num_layers=40)

    See Also
    --------

    :class:`PML`:
         A standard PML along a single dimension.

    :class:`Absorber`:
         Specifies an adiabatic absorber along a single dimension.

    **Notebooks:**
        * `How to troubleshoot a diverged FDTD simulation <../../notebooks/DivergedFDTDSimulation.html>`_

    **Lectures:**
        * `Introduction to perfectly matched layer (PML) tutorial <https://www.flexcompute.com/fdtd101/Lecture-6-Introduction-to-perfectly-matched-layer/>`__
    """

    num_layers: int = pd.Field(
        40,
        title="Number of Layers",
        description="Number of layers of 'stable' PML.",
        ge=MIN_NUM_PML_LAYERS,
    )

    parameters: PMLParams = pd.Field(
        DefaultStablePMLParameters,
        title="Stable PML Parameters",
        description="'Stable' parameters of the complex frequency-shifted absorption poles.",
    )


class Absorber(AbsorberSpec):
    """Specifies an adiabatic absorber along a single dimension.

    Notes
    -----

        This absorber is well-suited for dispersive materials intersecting with absorbing edges of the simulation at the
        expense of more layers.

        **Usage Caveats**

        Using absorber boundary is often a good remedy to resolve divergence issues related to :class:`PML`. The
        adiabatic absorber is a multilayer system with gradually increasing conductivity. The absorber usually has a
        larger undesired reflection compared to :class:`PML`. In practice, this small difference rarely matters,
        but is important to understand for simulations that require high accuracy.

        There are two possible sources for the reflection from absorbers. The first, and more common one, is that the
        ramping up of the conductivity is not sufficiently slow, which can be remedied by increasing the number of
        absorber layers (40 by default). The second one is that the absorption is not high enough, such that the
        light reaches the :class:`PEC` boundary at the end of the :class:`Absorber`, travels back through it,
        and is still not fully attenuated before re-entering the simulation region. If this is the case, increasing
        the maximum conductivity :class:`AbsorberParams` can help. In both cases, changing the order of the scaling
        of the conductivity (:attr:`tidy3d.AbsorberParams.sigma_order`) can also have an effect, but this is a more
        advanced setting that we typically do not recommend modifying.

    Example
    -------
    >>> pml = Absorber(num_layers=40)

    See Also
    --------

    :class:`PML`:
         A standard PML along a single dimension.

    **Notebooks:**
        * `How to troubleshoot a diverged FDTD simulation <../../notebooks/DivergedFDTDSimulation.html>`_
    """

    num_layers: int = pd.Field(
        40,
        title="Number of Layers",
        description="Number of layers of absorber to add to + and - boundaries.",
        ge=MIN_NUM_PML_LAYERS,
    )

    parameters: AbsorberParams = pd.Field(
        DefaultAbsorberParameters,
        title="Absorber Parameters",
        description="Adiabatic absorber parameters.",
    )


# pml types allowed in simulation init
PMLTypes = Union[PML, StablePML, Absorber, None]


# """ boundary specification classes """

# types of boundaries that can be used in Simulation

BoundaryEdgeType = Union[
    Periodic, PECBoundary, PMCBoundary, PML, StablePML, Absorber, BlochBoundary
]


class Boundary(Tidy3dBaseModel):
    """Boundary conditions at the minus and plus extents along a dimension.

    Notes
    -----

         To specify individual boundary conditions along different dimensions, instead of :class:`BoundarySpec`,
         this class is used, which defines the ``plus`` and ``minus`` boundaries along a single
         dimension.

    Example
    -------
    >>> boundary = Boundary(plus = PML(), minus = PECBoundary())

    See Also
    --------

    :class:`BoundarySpec`
        Specifies boundary conditions on each side of the domain and along each dimension.

    :class:`PML`
        A standard PML along a single dimension.

    **Notebooks**
        * `Setting up boundary conditions <../../notebooks/BoundaryConditions.html>`_
        * `Multilevel blazed diffraction grating <../../notebooks/GratingEfficiency.html>`_
    """

    plus: BoundaryEdgeType = pd.Field(
        PML(),
        title="Plus BC",
        description="Boundary condition on the plus side along a dimension.",
        discriminator=TYPE_TAG_STR,
    )

    minus: BoundaryEdgeType = pd.Field(
        PML(),
        title="Minus BC",
        description="Boundary condition on the minus side along a dimension.",
        discriminator=TYPE_TAG_STR,
    )

    @pd.root_validator(skip_on_failure=True)
    def bloch_on_both_sides(cls, values):
        """Error if a Bloch boundary is applied on only one side."""
        plus = values.get("plus")
        minus = values.get("minus")
        num_bloch = isinstance(plus, BlochBoundary) + isinstance(minus, BlochBoundary)
        if num_bloch == 1:
            raise SetupError(
                "Bloch boundaries must be applied either on both sides or on neither side."
            )
        return values

    @pd.root_validator(skip_on_failure=True)
    def periodic_with_pml(cls, values):
        """Error if PBC is specified with a PML."""
        plus = values.get("plus")
        minus = values.get("minus")
        num_pbc = isinstance(plus, Periodic) + isinstance(minus, Periodic)
        num_pml = isinstance(plus, (PML, StablePML, Absorber)) + isinstance(
            minus, (PML, StablePML, Absorber)
        )
        if num_pbc == 1 and num_pml == 1:
            raise SetupError("Cannot have both PML and PBC along the same dimension.")
        return values

    @pd.root_validator(skip_on_failure=True)
    def periodic_with_pec_pmc(cls, values):
        """If a PBC is specified along with PEC or PMC on the other side, manually set the PBC
        to PEC or PMC so that no special treatment of halos is required."""
        plus = values.get("plus")
        minus = values.get("minus")

        switched = False
        if isinstance(minus, (PECBoundary, PMCBoundary)) and isinstance(plus, Periodic):
            plus = minus
            switched = True
        elif isinstance(plus, (PECBoundary, PMCBoundary)) and isinstance(minus, Periodic):
            minus = plus
            switched = True
        if switched:
            values.update({"plus": plus, "minus": minus})
            log.warning(
                "A periodic boundary condition was specified on the opposite side of a perfect "
                "electric or magnetic conductor boundary. This periodic boundary condition will "
                "be replaced by the perfect electric or magnetic conductor across from it."
            )
        return values

    @classmethod
    def periodic(cls):
        """Periodic boundary specification on both sides along a dimension.

        Example
        -------
        >>> pbc = Boundary.periodic()
        """
        plus = Periodic()
        minus = Periodic()
        return cls(plus=plus, minus=minus)

    @classmethod
    def bloch(cls, bloch_vec: complex):
        """Bloch boundary specification on both sides along a dimension.

        Parameters
        ----------
        bloch_vec : complex
            Normalized component of the Bloch vector in units of 2 * pi / (size along dimension)
            in the background medium, along the dimension in which the boundary is specified.

        Example
        -------
        >>> bloch = Boundary.bloch(bloch_vec=1)
        """
        plus = BlochBoundary(bloch_vec=bloch_vec)
        minus = BlochBoundary(bloch_vec=bloch_vec)
        return cls(plus=plus, minus=minus)

    @classmethod
    def bloch_from_source(
        cls, source: BlochSourceType, domain_size: float, axis: Axis, medium: Medium = None
    ):
        """Bloch boundary specification on both sides along a dimension based on a given source.

        Parameters
        ----------
        source : Union[:class:`GaussianBeam`, :class:`ModeSource`, :class:`PlaneWave`]
            Angled source.
        domain_size: float
            Size of the domain in the direction normal to the Bloch boundary
        axis: int
            Axis normal to the Bloch boundary
        medium : :class:`.Medium`
            Background medium associated with the Bloch vector.
            Default: free space.

        Example
        -------
        >>> from tidy3d import GaussianPulse, PlaneWave, inf
        >>> pulse = GaussianPulse(freq0=200e12, fwidth=20e12)
        >>> pw_source = PlaneWave(
        ...     size=(inf,inf,0), source_time=pulse, direction='+', angle_theta=0.2, angle_phi=0.3)
        >>> bloch = Boundary.bloch_from_source(source=pw_source, domain_size=5, axis=0)
        """
        plus = BlochBoundary.from_source(
            source=source, domain_size=domain_size, axis=axis, medium=medium
        )
        minus = BlochBoundary.from_source(
            source=source, domain_size=domain_size, axis=axis, medium=medium
        )
        return cls(plus=plus, minus=minus)

    @classmethod
    def pec(cls):
        """PEC boundary specification on both sides along a dimension.

        Example
        -------
        >>> pec = Boundary.pec()
        """
        plus = PECBoundary()
        minus = PECBoundary()
        return cls(plus=plus, minus=minus)

    @classmethod
    def pmc(cls):
        """PMC boundary specification on both sides along a dimension.

        Example
        -------
        >>> pmc = Boundary.pmc()
        """
        plus = PMCBoundary()
        minus = PMCBoundary()
        return cls(plus=plus, minus=minus)

    @classmethod
    def pml(cls, num_layers: pd.NonNegativeInt = 12, parameters: PMLParams = DefaultPMLParameters):
        """PML boundary specification on both sides along a dimension.

        Parameters
        ----------
        num_layers : int = 12
            Number of layers of standard PML to add to + and - boundaries.
        parameters : :class:`PMLParams`
            Parameters of the complex frequency-shifted absorption poles.

        Example
        -------
        >>> pml = Boundary.pml(num_layers=20)
        """
        plus = PML(num_layers=num_layers, parameters=parameters)
        minus = PML(num_layers=num_layers, parameters=parameters)
        return cls(plus=plus, minus=minus)

    @classmethod
    def stable_pml(
        cls, num_layers: pd.NonNegativeInt = 40, parameters: PMLParams = DefaultStablePMLParameters
    ):
        """Stable PML boundary specification on both sides along a dimension.

        Parameters
        ----------
        num_layers : int = 40
            Number of layers of 'stable' PML to add to + and - boundaries.
        parameters : :class:`PMLParams`
            'Stable' parameters of the complex frequency-shifted absorption poles.

        Example
        -------
        >>> stable_pml = Boundary.stable_pml(num_layers=40)
        """
        plus = StablePML(num_layers=num_layers, parameters=parameters)
        minus = StablePML(num_layers=num_layers, parameters=parameters)
        return cls(plus=plus, minus=minus)

    @classmethod
    def absorber(
        cls, num_layers: pd.NonNegativeInt = 40, parameters: PMLParams = DefaultAbsorberParameters
    ):
        """Adiabatic absorber boundary specification on both sides along a dimension.

        Parameters
        ----------
        num_layers : int = 40
            Number of layers of absorber to add to + and - boundaries.
        parameters : :class:`PMLParams`
            Adiabatic absorber parameters.

        Example
        -------
        >>> absorber = Boundary.absorber(num_layers=40)
        """
        plus = Absorber(num_layers=num_layers, parameters=parameters)
        minus = Absorber(num_layers=num_layers, parameters=parameters)
        return cls(plus=plus, minus=minus)


class BoundarySpec(Tidy3dBaseModel):
    """
    Specifies boundary conditions on each side of the domain and along each dimension.

    Notes
    -----

        This :class:`BoundarySpec` object defines the boundary conditions applied on each of the 6 domain edges,
        and is provided as an input to the simulation.

        A :class:`BoundarySpec` consists of three :class:`Boundary` objects, each defining the boundaries on the plus
        and minus side of each dimension. In most cases, one just wants to specify whether there are absorbing
        :class:`PML` layers along any of the ``x``, ``y``, ``z`` dimensions. By default, ``tidy3d`` simulations have
        :class:`PML` boundaries on all sides.

        If we want to explicitly set the boundaries, we can use the :attr:`tidy3d.BoundarySpec.all_sides` method.
        This can be used to set any type of boundary condition on all sides of the simulation. We can also set
        :class:`PML` on specified sides only by calling the :attr:`tidy3d.BoundarySpec.pml` method, e.g. ``BoundarySpec.pml(
        x=False, y=False, z=False)``. This will put :class:`PML` along the dimensions defined as ``True``,
        and set periodic boundaries along the other dimensions.


    See Also
    --------

    :class:`Boundary`
        Boundary conditions at the minus and plus extents along a dimension.

    **Notebooks**
        * `How to troubleshoot a diverged FDTD simulation <../../notebooks/DivergedFDTDSimulation.html>`_

    **Lectures**
        * `Using FDTD to Compute a Transmission Spectrum <https://www.flexcompute.com/fdtd101/Lecture-2-Using-FDTD-to-Compute-a-Transmission-Spectrum/>`__
    """

    x: Boundary = pd.Field(
        Boundary(),
        title="Boundary condition along x.",
        description="Boundary condition on the plus and minus sides along the x axis. "
        "If ``None``, periodic boundaries are applied. Default will change to PML in 2.0 "
        "so explicitly setting the boundaries is recommended.",
    )

    y: Boundary = pd.Field(
        Boundary(),
        title="Boundary condition along y.",
        description="Boundary condition on the plus and minus sides along the y axis. "
        "If ``None``, periodic boundaries are applied. Default will change to PML in 2.0 "
        "so explicitly setting the boundaries is recommended.",
    )

    z: Boundary = pd.Field(
        Boundary(),
        title="Boundary condition along z.",
        description="Boundary condition on the plus and minus sides along the z axis. "
        "If ``None``, periodic boundaries are applied. Default will change to PML in 2.0 "
        "so explicitly setting the boundaries is recommended.",
    )

    def __getitem__(self, field_name: str) -> Boundary:
        """Get the :class:`Boundary` field by name (``boundary_spec[field_name]``).

        Parameters
        ----------
        field_name : ``str``
            Name of the axis, eg. "y" along which :class:`Boundary` is requested.

        Returns
        -------
        :class:`Boundary`
            Boundary conditions along the given axis.
        """
        if field_name == "x":
            return self.x
        if field_name == "y":
            return self.y
        if field_name == "z":
            return self.z
        raise DataError(f"field_name '{field_name}' not found")

    @classmethod
    def pml(cls, x: bool = False, y: bool = False, z: bool = False):
        """PML along specified directions

        Parameters
        ----------
        x : bool = False
            Toggle whether to set a default PML on both plus and minus sides along the x axis.
        y : bool = False
            Toggle whether to set a default PML on both plus and minus sides along the y axis.
        z : bool = False
            Toggle whether to set a default PML on both plus and minus sides along the z axis.

        Example
        -------
        >>> boundaries = BoundarySpec.pml(y=True)
        """
        return cls(
            x=Boundary.pml() if x else Boundary.periodic(),
            y=Boundary.pml() if y else Boundary.periodic(),
            z=Boundary.pml() if z else Boundary.periodic(),
        )

    @classmethod
    def pec(cls, x: bool = False, y: bool = False, z: bool = False):
        """PEC along specified directions

        Parameters
        ----------
        x : bool = False
            Toggle whether to set a PEC condition on both plus and minus sides along the x axis.
        y : bool = False
            Toggle whether to set a PEC condition on both plus and minus sides along the y axis.
        z : bool = False
            Toggle whether to set a PEC condition on both plus and minus sides along the z axis.

        Example
        -------
        >>> boundaries = BoundarySpec.pec(x=True, z=True)
        """
        return cls(
            x=Boundary.pec() if x else Boundary(),
            y=Boundary.pec() if y else Boundary(),
            z=Boundary.pec() if z else Boundary(),
        )

    @classmethod
    def pmc(cls, x: bool = False, y: bool = False, z: bool = False):
        """PMC along specified directions

        Parameters
        ----------
        x : bool = False
            Toggle whether to set a PMC condition on both plus and minus sides along the x axis.
        y : bool = False
            Toggle whether to set a PMC condition on both plus and minus sides along the y axis.
        z : bool = False
            Toggle whether to set a PMC condition on both plus and minus sides along the z axis.

        Example
        -------
        >>> boundaries = BoundarySpec.pmc(x=True, z=True)
        """
        return cls(
            x=Boundary.pmc() if x else Boundary(),
            y=Boundary.pmc() if y else Boundary(),
            z=Boundary.pmc() if z else Boundary(),
        )

    @classmethod
    def all_sides(cls, boundary: BoundaryEdge):
        """Set a given boundary condition on all six sides of the domain

        Parameters
        ----------
        boundary : :class:`BoundaryEdge`
            Boundary condition to apply on all six sides of the domain.

        Example
        -------
        >>> boundaries = BoundarySpec.all_sides(boundary=PML())
        """
        return cls(
            x=Boundary(minus=boundary, plus=boundary),
            y=Boundary(minus=boundary, plus=boundary),
            z=Boundary(minus=boundary, plus=boundary),
        )

    @cached_property
    def to_list(self) -> List[Tuple[BoundaryEdgeType, BoundaryEdgeType]]:
        """Returns edge-wise boundary conditions along each dimension for internal use."""
        return [
            (self.x.minus, self.x.plus),
            (self.y.minus, self.y.plus),
            (self.z.minus, self.z.plus),
        ]

    @cached_property
    def flipped_bloch_vecs(self) -> BoundarySpec:
        """Return a copy of the instance where all Bloch vectors are multiplied by -1."""
        bound_dims = dict(x=self.x.copy(), y=self.y.copy(), z=self.z.copy())
        for dim_key, bound_dim in bound_dims.items():
            bound_edges = dict(plus=bound_dim.plus.copy(), minus=bound_dim.minus.copy())
            for edge_key, bound_edge in bound_edges.items():
                if isinstance(bound_edge, BlochBoundary):
                    new_bloch_vec = -1 * bound_edge.bloch_vec
                    bound_edges[edge_key] = bound_edge.copy(update=dict(bloch_vec=new_bloch_vec))
            bound_dims[dim_key] = bound_edges
        return self.copy(update=bound_dims)
