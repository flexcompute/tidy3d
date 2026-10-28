"""Concrete FDTD simulation model."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import autograd.numpy as np
from pydantic import (
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    field_validator,
    model_validator,
)

from tidy3d._runtime import always_validate, fast_validation_active
from tidy3d.components.boundary import PML, Absorber, BoundarySpec, StablePML
from tidy3d.components.frequency_extrapolation import LowFrequencySmoothingSpec

# ``Box`` is used by the ``Simulation`` class doctest.
from tidy3d.components.geometry.base import Box, Geometry
from tidy3d.components.geometry.utils import flatten_groups
from tidy3d.components.grid.grid_spec import GridSpec
from tidy3d.components.lumped_element import LumpedElementType
from tidy3d.components.medium import (
    AnisotropicMediumFromMedium2D,
    Medium,
    Medium2D,
    ProtectedOpticalMediumPlacementType3D,
)
from tidy3d.components.medium.protected import (
    has_protected_media,
    protected_fdtd_monitor_policy_error,
)
from tidy3d.components.monitor import (
    AbstractFieldProjectionMonitor,
    AbstractMediumPropertyMonitor,
    FieldProjectionAngleMonitor,
    FieldProjectionKSpaceMonitor,
    FieldStructureMonitor,
    ModeMonitor,
    PointCloudFieldMonitor,
    SurfaceFieldMonitor,
    SurfaceFieldStructureMonitor,
    SurfaceFieldTimeMonitor,
)
from tidy3d.components.run_time_spec import RunTimeSpec
from tidy3d.components.source.field import ModeSource
from tidy3d.components.source.utils import SourceType
from tidy3d.components.structure import Structure
from tidy3d.components.types import TYPE_TAG_STR, Symmetry
from tidy3d.components.types.base import discriminated_union
from tidy3d.components.types.monitor import MonitorType
from tidy3d.components.validators import (
    call_wrapped_validator,
    named_obj_descr,
)
from tidy3d.config import config
from tidy3d.constants import SECOND
from tidy3d.exceptions import (
    SetupError,
    ValidationError,
)
from tidy3d.log import log
from tidy3d.updater import Updater

if TYPE_CHECKING:
    from tidy3d.compat import Self


from . import adjoint as _adjoint_methods
from . import boundaries as _boundaries_methods
from . import construction as _construction_methods
from . import export as _export_methods
from . import grid as _grid_methods
from . import materials as _materials_methods
from . import mode_integration as _mode_integration_methods
from . import monitors as _monitors_methods
from . import rf_integration as _rf_integration_methods
from . import runtime as _runtime_methods
from . import sources as _sources_methods
from . import tfsf as _tfsf_methods
from . import visualization as _visualization_methods
from .boundaries import validate_boundaries_for_zero_dims
from .yee import AbstractYeeGridSimulation

_ABSORBING_BOUNDARIES = (PML, StablePML, Absorber)


class Simulation(AbstractYeeGridSimulation):
    """
    Custom implementation of Maxwell’s equations which represents the physical model to be solved using the FDTD
    method.

    Notes
    -----

        A ``Simulation`` defines a custom implementation of Maxwell's equations which represents the physical model
        to be solved using `the Finite-Difference Time-Domain (FDTD) method
        <https://www.flexcompute.com/fdtd101/Lecture-1-Introduction-to-FDTD-Simulation/>`_. ``tidy3d`` simulations
        run very quickly in the cloud through GPU parallelization.

        .. image:: ../../_static/img/field_update_fdtd.png
            :width: 50%
            :align: left

        FDTD is a method for simulating the interaction of electromagnetic waves with structures and materials. It is
        the most widely used method in photonics design. The Maxwell's
        equations implemented in the ``Simulation`` are solved per time-step in the order shown in this image.

        The simplified input to FDTD solver consists of the permittivity distribution defined by :attr:`structures`
        which describe the device and :attr:`sources` of electromagnetic excitation. This information is used to
        computate the time dynamics of the electric and magnetic fields in this system. From these time-domain
        results, frequency-domain information of the simulation can also be extracted, and used for device design and
        optimization.

        If you are new to the FDTD method, we recommend you get started with the `FDTD 101 Lecture Series
        <https://www.flexcompute.com/tidy3d/learning-center/fdtd101/>`_

        **Dimensions Selection**

        By default, simulations are defined as 3D. To make the simulation 2D, we can just set the simulation
        :attr:`size` in one of the dimensions to be 0. However, note that we still have to define a grid size (eg.
        ``tidy3d.Simulation(size=[size_x, size_y, 0])``) and specify a periodic boundary condition in that direction.

        .. TODO sort out inheritance problem https://aware-moon.cloudvent.net/tidy3d/examples/notebooks/RingResonator/

        See further parameter explanations below.

        **Practical Advice**

        Use :class:`~tidy3d.RunTimeSpec` instead of a hardcoded ``run_time`` to automatically determine simulation
        duration based on field decay::

            sim = Simulation(..., run_time=td.RunTimeSpec(quality_factor=10))

        For grid resolution, use ``min_steps_per_wvl >= 20`` in :class:`AutoGrid` for standard simulations. The
        default value of 10 is suitable only for quick sanity checks. See :class:`AutoGrid` for detailed guidance.

        All lengths are in micrometers (μm), times in seconds (s), and frequencies in Hz. Convert wavelength to
        frequency with ``freq = td.C_0 / wavelength_um``.

    Example
    -------
    >>> from tidy3d import Sphere, Cylinder, PolySlab
    >>> from tidy3d import UniformCurrentSource, GaussianPulse
    >>> from tidy3d import FieldMonitor, FluxMonitor
    >>> from tidy3d import GridSpec, AutoGrid
    >>> from tidy3d import BoundarySpec, Boundary
    >>> from tidy3d import Medium
    >>> sim = Simulation(
    ...     size=(3.0, 3.0, 3.0),
    ...     grid_spec=GridSpec(
    ...         grid_x = AutoGrid(min_steps_per_wvl = 20),
    ...         grid_y = AutoGrid(min_steps_per_wvl = 20),
    ...         grid_z = AutoGrid(min_steps_per_wvl = 20)
    ...     ),
    ...     run_time=40e-11,
    ...     structures=[
    ...         Structure(
    ...             geometry=Box(size=(1, 1, 1), center=(0, 0, 0)),
    ...             medium=Medium(permittivity=2.0),
    ...         ),
    ...     ],
    ...     sources=[
    ...         UniformCurrentSource(
    ...             size=(0, 0, 0),
    ...             center=(0, 0.5, 0),
    ...             polarization="Hx",
    ...             current_amplitude_definition='total',
    ...             source_time=GaussianPulse(
    ...                 freq0=2e14,
    ...                 fwidth=4e13,
    ...             ),
    ...         )
    ...     ],
    ...     monitors=[
    ...         FluxMonitor(size=(1, 1, 0), center=(0, 0, 0), freqs=[2e14, 2.5e14], name='flux'),
    ...     ],
    ...     symmetry=(0, 0, 0),
    ...     boundary_spec=BoundarySpec(
    ...         x = Boundary.pml(num_layers=20),
    ...         y = Boundary.pml(num_layers=30),
    ...         z = Boundary.periodic(),
    ...     ),
    ...     shutoff=1e-6,
    ...     courant=0.8,
    ...     subpixel=False,
    ... )

    See Also
    --------

    **Notebooks:**
        * `Quickstart <../../notebooks/StartHere.html>`_: Usage in a basic simulation flow.
        * `Using automatic nonuniform meshing <../../notebooks/AutoGrid.html>`_
        * See nearly all notebooks for :class:`.Simulation` applications.

    **Lectures:**
        * `Introduction to FDTD Simulation <https://www.flexcompute.com/fdtd101/Lecture-1-Introduction-to-FDTD-Simulation/#presentation-slides>`_: Usage in a basic simulation flow.
        * `Prelude to Integrated Photonics Simulation: Mode Injection <https://www.flexcompute.com/fdtd101/Lecture-4-Prelude-to-Integrated-Photonics-Simulation-Mode-Injection/>`_

    **GUI:**
        * `FDTD Walkthrough <https://www.flexcompute.com/tidy3d/learning-center/tidy3d-gui/Lecture-1-FDTD-Walkthrough/#presentation-slides>`_
    """

    boundary_spec: BoundarySpec = Field(
        default_factory=BoundarySpec,
        title="Boundaries",
        description="Specification of boundary conditions along each dimension. If ``None``, "
        "PML boundary conditions are applied on all sides.",
    )
    """Specification of boundary conditions along each dimension. If ``None``, :class:`PML` boundary conditions are
    applied on all sides.

    Example
    -------
    Simple application reference:

    .. code-block:: python

         Simulation(
            ...
             boundary_spec=BoundarySpec(
                x = Boundary.pml(num_layers=20),
                y = Boundary.pml(num_layers=30),
                z = Boundary.periodic(),
            ),
            ...
         )

    See Also
    --------

    :class:`PML`:
        A perfectly matched layer model.

    :class:`BoundarySpec`:
        Specifies boundary conditions on each side of the domain and along each dimension.

    `Index <../boundary_conditions.html>`__
        All boundary condition models.

    **Notebooks**
        * `How to troubleshoot a diverged FDTD simulation <../../notebooks/DivergedFDTDSimulation.html>`_

    **Lectures**
        * `Using FDTD to Compute a Transmission Spectrum <https://www.flexcompute.com/fdtd101/Lecture-2-Using-FDTD-to-Compute-a-Transmission-Spectrum/>`__
    """

    courant: float = Field(
        default=0.99,
        title="Normalized Courant Factor",
        description="Normalized Courant stability factor that is no larger than 1 when CFL "
        "stability condition is met. It controls time step to spatial step ratio. "
        "Lower values lead to more stable simulations for dispersive materials, "
        "but result in longer simulation times.",
        gt=0.0,
        le=1.0,
    )

    """The Courant-Friedrichs-Lewy (CFL) stability factor :math:`C`, controls time step to spatial step ratio.  A
    physical wave has to propagate slower than the numerical information propagation in a Yee-cell grid. This is
    because in this spatially-discrete grid, information propagates over 1 spatial step :math:`\\Delta x`
    over a time step :math:`\\Delta t`. This constraint enables the correct physics to be captured by the simulation.

    **1D Illustration**

    In a 1D model:

    .. image:: ../../_static/img/courant_instability.png

    Lower values lead to more stable simulations for dispersive materials, but result in longer simulation times. This
    factor is normalized to no larger than 1 when CFL stability condition is met in 3D.

    .. TODO finish this section for 1D, 2D and 3D references.

    For a 1D grid:

    .. math::

        C_{\\text{1D}} = \\frac{c \\Delta t}{\\Delta x} \\leq 1

    **2D Illustration**

    In a 2D uniform grid, where the :math:`E_z` field is at the red dot center surrounded by four green magnetic edge components
    in a square Yee cell grid:

    .. image:: ../../_static/img/courant_instability_2d.png

    .. math::

        C_{\\text{2D}} = \\frac{c\\Delta t}{\\Delta x} \\leq \\frac{1}{\\sqrt{2}}

    Hence, for the same spatial grid, the time step in 2D grid needs to be smaller than the time step in a 1D grid. Note we use
    a normalized Courant number in our simulation, which in 2D is :math:`\\sqrt{2}C_{\\text{2D}}`. CFL stability condition
    is met when the normalized Courant number is no larger than 1.

    **3D Illustration**

    For an isotropic medium with refractive index :math:`n`, the 3D time step condition can be derived to be:

    .. math::

        \\Delta t \\le \\frac{n}{c \\sqrt{\\frac{1}{\\Delta x^2} + \\frac{1}{\\Delta y^2} + \\frac{1}{\\Delta z^2}}}

    In this case, the number of spatial grid points scale by :math:`\\sim \\frac{1}{\\Delta x^3}` where :math:`\\Delta x`
    is the spatial discretization in the :math:`x` dimension. If the total simulation time is kept the same whilst
    maintaining the CFL condition, then the number of time steps required scale by :math:`\\sim \\frac{1}{\\Delta x}`.
    Hence, the spatial grid discretization influences the total time-steps required. The total simulation scaling per
    spatial grid size in this case is by :math:`\\sim \\frac{1}{\\Delta x^4}.`

    As an example, in this case, refining the mesh by a factor or 2 (reducing the spatial step size by half)
    :math:`\\Delta x \\to \\frac{\\Delta x}{2}` will increase the total simulation computational cost by 16.

    **Divergence Caveats**

    ``tidy3d`` uses a default Courant factor of 0.99. When a dispersive material with ``eps_inf < 1`` is used,
    the Courant factor will be automatically adjusted to be smaller than ``sqrt(eps_inf)`` to ensure stability. If
    your simulation still diverges despite addressing any other issues discussed above, reducing the Courant
    factor may help.

    See Also
    --------

    :attr:`grid_spec`
        Specifications for the simulation grid along each of the three directions.

    **Lectures:**
        *  `Time step size and CFL condition in FDTD <https://www.flexcompute.com/fdtd101/Lecture-7-Time-step-size-and-CFL-condition-in-FDTD/>`_
        *  `Numerical dispersion in FDTD <https://www.flexcompute.com/fdtd101/Lecture-8-Numerical-dispersion-in-FDTD/>`_
    """

    relax_courant: bool = Field(
        default=False,
        title="Relax Courant",
        description="Relax the CFL stability condition if possible.",
    )

    precision: Literal["hybrid", "double"] = Field(
        default="hybrid",
        title="Floating-point Precision",
        description="Floating point precision to use in the computations.",
    )
    """
    By default, Tidy3D uses
    a hybrid approach that offers a good balance of speed and accuracy for almost all
    simulations. However, for large simulations (or simulations with a long run time),
    where very high accuracy is needed, the precision can be set to double everywhere.
    Note that this doubles the FlexCredit cost of the simulation. Note that this argument
    affects not only the fields in the time stepping, but also the the structure
    discretization on the grid. Thus, results stored in a ``PermittivityMonitor`` or a
    ``ModeSolverMonitor`` can be affected. For the latter, note also that the precision set
    here affects the structure discretization, and is independent from the
    ``ModeSpec.precision`` argument, which only affects the eigenvalue solver.
    """

    lumped_elements: tuple[LumpedElementType, ...] = Field(
        default=(),
        title="Lumped Elements",
        description="Tuple of lumped elements in the simulation. ",
    )
    """
    Tuple of lumped elements in the simulation.

    Example
    -------
    Simple application reference:

    .. code-block:: python

         Simulation(
            ...
            lumped_elements=[
                LumpedResistor(
                    size=(0, 3, 1),
                    center=(0, 0, 0),
                    voltage_axis=2,
                    resistance=50,
                    name="resistor_1",
                )
            ],
            ...
         )

    See Also
    --------

    `Microwave & RF <../microwave.html>`_:
        Where the RF and microwave APIs live now.
    """

    grid_spec: GridSpec = Field(
        default_factory=GridSpec,
        title="Grid Specification",
        description="Specifications for the simulation grid along each of the three directions.",
    )
    """
    Specifications for the simulation grid along each of the three directions.

    Example
    -------
    Simple application reference:

    .. code-block:: python

         Simulation(
            ...
             grid_spec=GridSpec(
                grid_x = AutoGrid(min_steps_per_wvl = 20),
                grid_y = AutoGrid(min_steps_per_wvl = 20),
                grid_z = AutoGrid(min_steps_per_wvl = 20)
            ),
            ...
         )

    **Usage Recommendations**

    In the *finite-difference* time domain method, the computational domain is discretized by a little cubes called
    the Yee cell. A discrete lattice formed by this Yee cell is used to describe the fields. In 3D, the electric
    fields are distributed on the edge of the Yee cell and the magnetic fields are distributed on the surface of the
    Yee cell.

    .. image:: ../../_static/img/yee_grid_illustration.png

    Note
    ----

        A typical rule of thumb is to choose the discretization to be about :math:`\\frac{\\lambda_m}{20}` where
        :math:`\\lambda_m` is the field wavelength.

    **Numerical Dispersion - 1D Illustration**

    Numerical dispersion is a form of numerical error dependent on the spatial and temporal discretization of the
    fields. In order to reduce it, it is necessary to improve the discretization of the simulation for particular
    frequencies and spatial features. This is an important aspect of defining the grid.

    Consider a standard 1D wave equation in vacuum:

    .. math::

        \\left( \\frac{\\delta ^2 }{\\delta x^2} - \\frac{1}{c^2} \\frac{\\delta^2}{\\delta t^2} \\right) E = 0

    which is ideally solved into a monochromatic travelling wave:

    .. math::

        E(x) = e^{j (kx - \\omega t)}

    This physical wave is described with a wavevector :math:`k` for the spatial field variations and the angular
    frequency :math:`\\omega` for temporal field variations. The spatial and temporal field variations are related by
    a dispersion relation.

    .. TODO explain the above more

    The ideal dispersion relation is:

    .. math::

        \\left( \\frac{\\omega}{c} \\right)^2 = k^2

    However, in the FDTD simulation, the spatial and temporal fields are discrete.

    .. TODO improve the ways figures are represented.

    .. image:: ../../_static/img/numerical_dispersion_grid_1d.png
        :width: 30%
        :align: right

    The same 1D monochromatic wave can be solved using the FDTD method where :math:`m` is the index in the grid:

    .. math::

        \\frac{\\delta^2}{\\delta x^2} E(x_i) \\approx \\frac{1}{\\Delta x^2} \\left[ E(x_i + \\Delta x) + E(x_i -
        \\Delta x) - 2 E(x_i) \\right]

    .. math::

        \\frac{\\delta^2}{\\delta t^2} E(t_{\\alpha}) \\approx \\frac{1}{\\Delta t^2} \\left[ E(t_{\\alpha} + \\Delta
        t) + E(t_{\\alpha} - \\Delta t) - 2 E(t_{\\alpha}) \\right]

    .. TODO define the alpha

    Hence, these discrete fields have this new dispersion relation:

    .. math::

        \\left( \\frac{1}{c \\Delta t} \\text{sin} \\left( \\frac{\\omega \\Delta t}{2} \\right)^2 \\right) = \\left(
        \\frac{1}{\\Delta x} \\text{sin} \\left( \\frac{k \\Delta x}{2} \\right) \\right)^2

    The ideal wave solution and the discrete solution have a mismatch illustrated below as a result of the numerical
    error introduced by numerical dispersion. This plot illustrates the angular frequency as a function of wavevector
    for both the physical ideal wave and the numerical discrete wave implemented in FDTD.

    .. image:: ../../_static/img/numerical_dispersion_discretization_1d.png

    .. TODO improve these images positions

    At lower frequencies, when the discretization of :math:`\\Delta x` is small compared to the wavelength the error
    between the solutions is very low. When this proportionality increases between the spatial step size and the
    angular wavelength, this introduces numerical dispersion errors.

    .. math::

        k \\Delta x = \\frac{2 \\pi}{\\lambda_k} \\Delta x


    **Usage Recommendations**

    *   It is important to understand the relationship between the time-step :math:`\\Delta t` defined by the
        :attr:`courant` factor, and the spatial grid distribution to guarantee simulation stability.

    *   If your structure has small features, consider using a spatially nonuniform grid. This guarantees finer
        spatial resolution near the features, but away from it you use have a larger (and computationally faster) grid.
        In this case, the time step :math:`\\Delta t` is defined by the smallest spatial grid size.

    See Also
    --------

    :attr:`courant`
        The Courant-Friedrichs-Lewy (CFL) stability factor

    :class:`.GridSpec`
        Collective grid specification for all three dimensions.

    :class:`.UniformGrid`
        Uniform 1D grid.

    :class:`.AutoGrid`
        Specification for non-uniform grid along a given dimension.

    **Notebooks:**
        * `Using automatic nonuniform meshing <../../notebooks/AutoGrid.html>`_

    **Lectures:**
        *  `Time step size and CFL condition in FDTD <https://www.flexcompute.com/fdtd101/Lecture-7-Time-step-size-and-CFL-condition-in-FDTD/>`_
        *  `Numerical dispersion in FDTD <https://www.flexcompute.com/fdtd101/Lecture-8-Numerical-dispersion-in-FDTD/>`_
    """

    medium: ProtectedOpticalMediumPlacementType3D = Field(
        default_factory=Medium,
        title="Background Medium",
        description="Background medium of simulation, defaults to vacuum if not specified.",
        discriminator=TYPE_TAG_STR,
    )
    """
    Background medium of simulation, defaults to vacuum if not specified.

    See Also
    --------

    `Material Library <../material_library.html>`_:
        The material library is a dictionary containing various dispersive models from real world materials.

    `Index <../mediums.html>`__:
        Dispersive and dispersionless Mediums models.

    **Notebooks:**

    * `Fitting dispersive material models <../../notebooks/Fitting.html>`_

    **Lectures:**

    * `Modeling dispersive material in FDTD <https://www.flexcompute.com/fdtd101/Lecture-5-Modeling-dispersive-material-in-FDTD/>`_

    **GUI:**

    * `Mediums <https://www.flexcompute.com/tidy3d/learning-center/tidy3d-gui/Lecture-2-Mediums/>`_

    """

    normalize_index: NonNegativeInt | None = Field(
        default=0,
        title="Normalization index",
        description="Index of the source in the tuple of sources whose spectrum will be used to "
        "normalize the frequency-dependent data. If ``None``, the raw field data is returned "
        "unnormalized.",
    )
    """
    Index of the source in the tuple of sources whose spectrum will be used to normalize the frequency-dependent
    data. If ``None``, the raw field data is returned. If ``None``, the raw field data is returned unnormalized.
    """

    monitors: tuple[discriminated_union(MonitorType), ...] = Field(
        default=(),
        title="Monitors",
        description="Tuple of monitors in the simulation. "
        "Note: monitor names are used to access data after simulation is run.",
    )
    """
    Tuple of monitors in the simulation. Monitor names are used to access data after simulation is run.

    See Also
    --------

    `Index <../monitors.html>`__
        All the monitor implementations.
    """

    sources: tuple[discriminated_union(SourceType), ...] = Field(
        default=(),
        title="Sources",
        description="Tuple of electric current sources injecting fields into the simulation.",
    )
    """
    Tuple of electric current sources injecting fields into the simulation.

    Example
    -------
    Simple application reference:

    .. code-block:: python

         Simulation(
            ...
            sources=[
                UniformCurrentSource(
                    size=(0, 0, 0),
                    center=(0, 0.5, 0),
                    polarization="Hx",
                    source_time=GaussianPulse(
                        freq0=2e14,
                        fwidth=4e13,
                    ),
                )
            ],
            ...
         )

    See Also
    --------

    `Index <../sources.html>`__:
        Frequency and time domain source models.
    """

    shutoff: NonNegativeFloat = Field(
        default=1e-5,
        title="Shutoff Condition",
        description="Ratio of the instantaneous integrated E-field intensity to the maximum value "
        "at which the simulation will automatically terminate time stepping. "
        "Used to prevent extraneous run time of simulations with fully decayed fields. "
        "Set to ``0`` to disable this feature.",
    )
    """
    Ratio of the instantaneous integrated E-field intensity to the maximum value
    at which the simulation will automatically terminate time stepping.
    Used to prevent extraneous run time of simulations with fully decayed fields.
    Set to ``0`` to disable this feature.
    """

    structures: tuple[Structure, ...] = Field(
        default=(),
        title="Structures",
        description="Tuple of structures present in simulation. "
        "Note: Structures defined later in this list override the "
        "simulation material properties in regions of spatial overlap.",
    )
    """
    Tuple of structures present in simulation. Structures defined later in this list override the simulation
    material properties in regions of spatial overlap.

    Example
    -------
    Simple application reference:

    .. code-block:: python

        Simulation(
            ...
            structures=[
                 Structure(
                 geometry=Box(size=(1, 1, 1), center=(0, 0, 0)),
                 medium=Medium(permittivity=2.0),
                 ),
            ],
            ...
        )

    **Usage Caveats**

    It is very important to understand the way the dielectric permittivity of the :class:`.Structure` list is resolved
    by the simulation grid. Without :attr:`subpixel` averaging, the structure geometry in relation to the
    grid points can lead to its features permittivity not being fully resolved by the
    simulation.

    For example, in the image below, two silicon slabs with thicknesses 150nm and 175nm centered in a grid with
    spatial discretization :math:`\\Delta z = 25\\text{nm}` will compute equivalently because that grid does
    not resolve the feature permittivity in between grid points without :attr:`subpixel` averaging.

    .. image:: ../../_static/img/permittivity_on_yee_grid.png

    See Also
    --------

    :class:`.Structure`:
        Defines a physical object that interacts with the electromagnetic fields.

    :attr:`subpixel`
        Subpixel averaging of the permittivity based on structure definition, resulting in much higher
        accuracy for a given grid size.

    **Notebooks:**

    * `Visualizing geometries in Tidy3D <../../notebooks/VizSimulation.html>`_

    **Lectures:**

    * `Using FDTD to Compute a Transmission Spectrum <https://www.flexcompute.com/fdtd101/Lecture-2-Using-FDTD-to-Compute-a-Transmission-Spectrum/>`_
    *  `Dielectric constant assignment on Yee grids <https://www.flexcompute.com/fdtd101/Lecture-9-Dielectric-constant-assignment-on-Yee-grids/>`_

    **GUI:**

    * `Structures <https://www.flexcompute.com/tidy3d/learning-center/tidy3d-gui/Lecture-3-Structures/#presentation-slides>`_
    """

    symmetry: tuple[Symmetry, Symmetry, Symmetry] = Field(
        default=(0, 0, 0),
        title="Symmetries",
        description="Tuple of integers defining reflection symmetry across a plane "
        "bisecting the simulation domain normal to the x-, y-, and z-axis "
        "at the simulation center of each axis, respectively. "
        "Each element can be ``0`` (no symmetry), ``1`` (even, i.e. "
        ":class:`~tidy3d.PMCBoundary` symmetry) or ``-1`` (odd, i.e. "
        ":class:`~tidy3d.PECBoundary` symmetry). "
        "Note that the vectorial nature of the fields must be taken into account to correctly "
        "determine the symmetry value.",
    )
    """
    You should set the ``symmetry`` parameter in your :class:`.Simulation` object using a tuple of integers
    defining reflection symmetry across a plane bisecting the simulation domain normal to the x-, y-, and z-axis.
    Each element can be 0 (no symmetry), 1 (even, i.e. :class:`~tidy3d.PMCBoundary` symmetry) or -1 (odd, i.e. :class:`~tidy3d.PECBoundary`
    symmetry). Note that the vectorial nature of the fields must be considered to determine the symmetry value
    correctly.

    The figure below illustrates how the electric and magnetic field components transform under
    :class:`~tidy3d.PECBoundary`- and :class:`~tidy3d.PMCBoundary`-like symmetry planes. You can refer to this figure
    when considering whether a source field conforms to a :class:`~tidy3d.PECBoundary`- or
    :class:`~tidy3d.PMCBoundary`-like symmetry axis. This would be helpful, especially when dealing with optical
    waveguide modes.

    .. image:: ../../notebooks/Symmetry/assets/pec_pmc.png


    .. TODO maybe resize?
    """

    # TODO: at a later time (once well tested) we could consider making default of RunTimeSpec()
    run_time: PositiveFloat | RunTimeSpec = Field(
        title="Run Time",
        description="Total electromagnetic evolution time in seconds. "
        "Note: If simulation 'shutoff' is specified, "
        "simulation will terminate early when shutoff condition met. "
        "Alternatively, user may supply a :class:`RunTimeSpec` to this field, which will auto-"
        "compute the ``run_time`` based on the contents of the spec. If this option is used, "
        "the evaluated ``run_time`` value is available in the ``Simulation._run_time`` property.",
        json_schema_extra={"units": SECOND},
    )
    """
    Total electromagnetic evolution time in seconds. If simulation 'shutoff' is specified, simulation will
    terminate early when shutoff condition met.

    **How long to run a simulation?**

    The frequency-domain response obtained in the FDTD simulation only accurately represents the continuous-wave
    response of the system if the fields at the beginning and at the end of the time stepping are (very close to)
    zero. So, you should run the simulation for a time enough to allow the electromagnetic fields decay to negligible
    values within the simulation domain.

    When dealing with light propagation in a NON-RESONANT device, like a simple optical waveguide, a good initial
    guess to simulation run_time would be the a few times the largest domain dimension (:math:`L`) multiplied by the
    waveguide mode group index (:math:`n_g`), divided by the speed of light in a vacuum (:math:`c_0`),
    plus the ``source_time``:

    .. math::

        t_{sim} \\approx \\frac{n_g L}{c_0} + t_{source}

    By default, ``tidy3d`` checks periodically the total field intensity left in the simulation, and compares that to
    the maximum total field intensity recorded at previous times. If it is found that the ratio of these two values
    is smaller than the default :attr:`shutoff` value :math:`10^{-5}`, the simulation is terminated as the fields
    remaining in the simulation are deemed negligible. The shutoff value can be controlled using the :attr:`shutoff`
    parameter, or completely turned off by setting it to zero. In most cases, the default behavior ensures that
    results are correct, while avoiding unnecessarily long run times. The Flex Unit cost of the simulation is also
    proportionally scaled down when early termination is encountered.

    **Resonant Caveats**

    Should I make sure that fields have fully decayed by the end of the simulation?

    The main use case in which you may want to ignore the field decay warning is when you have high-Q modes in your
    simulation that would require an extremely long run time to decay. In that case, you can use the the
    :class:`tidy3d.plugins.resonance.ResonanceFinder` plugin to analyze the modes, as well as field monitors with
    vaporization to capture the modal profiles. The only thing to note is that the normalization of these modal
    profiles would be arbitrary, and would depend on the exact run time and apodization definition. An example of
    such a use case is presented in our case study.

    .. TODO add links to resonant plugins.

    See Also
    --------

    **Notebooks**

    *   `High-Q silicon resonator <../../notebooks/HighQSi.html>`_

    """

    low_freq_smoothing: LowFrequencySmoothingSpec | None = Field(
        default=None,
        title="Low Frequency Smoothing",
        description="The low frequency smoothing parameters for the simulation.",
    )

    # Model construction and field validation.

    # Bind focused area implementations directly onto this model. This keeps
    # the runtime MRO and generated documentation free of behavioral mixins.

    # SimulationConstruction
    from_scene = _construction_methods.from_scene
    padded_copy = _construction_methods.padded_copy
    uniformly_padded_copy = _construction_methods.uniformly_padded_copy

    # SimulationExport
    to_gdstk = _export_methods.to_gdstk
    _structure_exports_as_filled_region = _export_methods._structure_exports_as_filled_region
    _optical_medium_export_key = _export_methods._optical_medium_export_key
    to_gds = _export_methods.to_gds
    to_gds_file = _export_methods.to_gds_file

    # SimulationAdjoint
    _flux_adjoint_helper_parent_names = _adjoint_methods._flux_adjoint_helper_parent_names
    _is_flux_adjoint_helper_monitor = _adjoint_methods._is_flux_adjoint_helper_monitor
    _monitor_validation_label = _adjoint_methods._monitor_validation_label
    _monitor_validation_index = _adjoint_methods._monitor_validation_index
    _with_adjoint_monitors = _adjoint_methods._with_adjoint_monitors
    _make_adjoint_monitors = _adjoint_methods._make_adjoint_monitors
    _freqs_adjoint = _adjoint_methods._freqs_adjoint

    # SimulationRFIntegration
    validate_rf_type = _rf_integration_methods.validate_rf_type
    requires_enterprise_license = _rf_integration_methods.requires_enterprise_license
    _warn_rf_license = _rf_integration_methods._warn_rf_license
    _validate_microwave_mode_specs = _rf_integration_methods._validate_microwave_mode_specs

    # SimulationModeIntegration
    _validate_internal_absorber_placement = (
        _mode_integration_methods._validate_internal_absorber_placement
    )
    _validate_modes_size = _mode_integration_methods._validate_modes_size
    _validate_num_cells_in_mode_objects = (
        _mode_integration_methods._validate_num_cells_in_mode_objects
    )
    complex_fields = _mode_integration_methods.complex_fields
    _has_lossy_mode_decomposition_feature = (
        _mode_integration_methods._has_lossy_mode_decomposition_feature
    )

    # SimulationRuntime
    _simple_bc = _runtime_methods._simple_bc
    _validate_relax_courant_compatibility = _runtime_methods._validate_relax_courant_compatibility
    _validate_size = _runtime_methods._validate_size
    _run_time = _runtime_methods._run_time
    _resolve_run_time = _runtime_methods._resolve_run_time
    frequency_range = _runtime_methods.frequency_range
    _dt_fixed_angle_reduction_factor = _runtime_methods._dt_fixed_angle_reduction_factor
    scaled_courant = _runtime_methods.scaled_courant
    dt = _runtime_methods.dt
    tmesh = _runtime_methods.tmesh
    num_time_steps = _runtime_methods.num_time_steps
    wvl_mat_min = _runtime_methods.wvl_mat_min
    nyquist_step = _runtime_methods.nyquist_step

    # SimulationMaterials
    _warn_3d_structures_missing_2d_yee_sampling_plane = (
        _materials_methods._warn_3d_structures_missing_2d_yee_sampling_plane
    )
    _validate_scene = _materials_methods._validate_scene
    _check_custom_medium_geometry_overlap = _materials_methods._check_custom_medium_geometry_overlap
    mediums = _materials_methods.mediums
    medium_map = _materials_methods.medium_map
    background_structure = _materials_methods.background_structure
    intersecting_media = _materials_methods.intersecting_media
    intersecting_structures = _materials_methods.intersecting_structures
    self_structure = _materials_methods.self_structure
    all_structures = _materials_methods.all_structures
    get_refractive_indices = _materials_methods.get_refractive_indices
    n_max = _materials_methods.n_max
    custom_datasets = _materials_methods.custom_datasets
    allow_gain = _materials_methods.allow_gain
    perturbed_mediums_copy = _materials_methods.perturbed_mediums_copy

    # SimulationMonitors
    _projection_monitor_mediums_in_bounds = _monitors_methods._projection_monitor_mediums_in_bounds
    _projection_monitor_media_on_plane = _monitors_methods._projection_monitor_media_on_plane
    _validate_monitor_size = _monitors_methods._validate_monitor_size
    _validate_time_monitors_num_steps = _monitors_methods._validate_time_monitors_num_steps
    _validate_freq_monitors_freq_range = _monitors_methods._validate_freq_monitors_freq_range
    _monitors_data_size = _monitors_methods._monitors_data_size
    monitors_data_size = _monitors_methods.monitors_data_size
    _validate_datasets_not_none = _monitors_methods._validate_datasets_not_none
    _warn_time_monitors_outside_run_time = _monitors_methods._warn_time_monitors_outside_run_time
    monitor_medium = _monitors_methods.monitor_medium

    # SimulationTFSF
    _get_periodic_fixed_angle_sources = _tfsf_methods._get_periodic_fixed_angle_sources
    _dipole_emission_tfsf_source = _tfsf_methods._dipole_emission_tfsf_source
    _dipole_emission_tfsf_injection_plane = _tfsf_methods._dipole_emission_tfsf_injection_plane
    _dipole_emission_tfsf_injection_medium = _tfsf_methods._dipole_emission_tfsf_injection_medium
    _dipole_emission_background_index = _tfsf_methods._dipole_emission_background_index
    _aux_tfsf_source = _tfsf_methods._aux_tfsf_source
    aux_fields = _tfsf_methods.aux_fields
    _validate_tfsf_structure_intersections = _tfsf_methods._validate_tfsf_structure_intersections
    _fixed_angle_sources = _tfsf_methods._fixed_angle_sources
    _is_periodic_fixed_angle = _tfsf_methods._is_periodic_fixed_angle

    # SimulationSources
    _source_medium = _sources_methods._source_medium
    _check_source_freq_available = _sources_methods._check_source_freq_available
    _thin_lens_source_plane_cells = _sources_methods._thin_lens_source_plane_cells
    _thin_lens_setup_work_units = _sources_methods._thin_lens_setup_work_units
    _thin_lens_setup_work_limit = _sources_methods._thin_lens_setup_work_limit
    _thin_lens_monitor_setup_evaluations = _sources_methods._thin_lens_monitor_setup_evaluations
    _thin_lens_min_background_index = _sources_methods._thin_lens_min_background_index
    _validate_gaussian_like_beam_background_medium = (
        _sources_methods._validate_gaussian_like_beam_background_medium
    )
    _validate_gaussian_like_beam_backgrounds = (
        _sources_methods._validate_gaussian_like_beam_backgrounds
    )
    _validate_thin_lens_setup_size = _sources_methods._validate_thin_lens_setup_size

    # SimulationBoundaries
    _validate_frequency_mode_abc = _boundaries_methods._validate_frequency_mode_abc
    _validate_absorber_in_zero_dims = _boundaries_methods._validate_absorber_in_zero_dims
    _validate_pec_frame_not_in_pml_extrusion = (
        _boundaries_methods._validate_pec_frame_not_in_pml_extrusion
    )
    _num_non_pml_cells = _boundaries_methods._num_non_pml_cells

    # SimulationGrid
    _validate_auto_grid_wavelength = _grid_methods._validate_auto_grid_wavelength
    _validate_lumped_element_grid_size = _grid_methods._validate_lumped_element_grid_size
    num_cells = _grid_methods._simulation_num_cells
    _num_computational_grid_points_dim = _grid_methods._num_computational_grid_points_dim
    num_computational_grid_points = _grid_methods.num_computational_grid_points

    # SimulationVisualization
    plot_3d = _visualization_methods.plot_3d

    @model_validator(mode="before")
    @classmethod
    def _update_simulation(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Update the simulation if it is an earlier version."""

        # if no version, assume it's already updated
        if "version" not in data:
            return data

        # otherwise, call the updator to update the values dictionary
        updater = Updater(sim_dict=data)
        return updater.update_to_current()

    @model_validator(mode="after")
    @always_validate
    def _run_after_validators(self) -> Self:
        """Run post-init validations in an explicit, dependency-aware order."""
        if fast_validation_active():
            # fast mode keeps only the step that changes the model; every other call
            # in this chain checks and would be skipped by the class-level guard
            call_wrapped_validator(validate_boundaries_for_zero_dims, self)
            return self
        # Normalize zero-dimensional inputs and run shared Yee-grid checks first.
        call_wrapped_validator(validate_boundaries_for_zero_dims, self)
        self._validate_auto_grid_wavelength()
        super()._run_after_validators()
        self._warn_3d_structures_missing_2d_yee_sampling_plane()

        # Validate boundary-dependent runtime and modal-decomposition features.
        self._validate_frequency_mode_abc()
        self._validate_relax_courant_compatibility()
        self._validate_absorber_in_zero_dims()
        self._validate_internal_absorber_placement()

        # Finish with the aggregate scene validation.
        self._validate_scene()
        return self

    @field_validator("structures")
    @classmethod
    def _validate_2d_geometry_has_2d_medium(
        cls, val: tuple[Structure, ...]
    ) -> tuple[Structure, ...]:
        """Warn if a geometry bounding box has zero size in a certain dimension."""

        if val is None:
            return val

        with log as consolidated_logger:
            for i, structure in enumerate(val):
                if isinstance(structure.medium, Medium2D | AnisotropicMediumFromMedium2D):
                    continue
                for geom in flatten_groups(structure.geometry):
                    zero_dims = geom.zero_dims
                    if len(zero_dims) > 0:
                        obj_descr = named_obj_descr(structure, "structures", i)
                        consolidated_logger.warning(
                            f"Structure: {obj_descr} has geometry with zero size along "
                            f"dimensions {zero_dims}, and with a medium that is not a 'Medium2D'. "
                            "This is probably not correct, since the resulting simulation will "
                            "depend on the details of the numerical grid. Consider either "
                            "giving the geometry a nonzero thickness or using a 'Medium2D'."
                        )

        return val

    @field_validator("structures")
    @classmethod
    def _validate_incompatible_material_intersections(
        cls, val: tuple[Structure, ...]
    ) -> tuple[Structure, ...]:
        """Check for intersections of incompatible materials."""
        structures = val
        incompatible_indices = []
        incompatible_structures = []
        # first just isolate the incompatible structures, to avoid unnecessary double looping
        # keep track of indices to give helpful error message
        for i, structure in enumerate(structures):
            if structure.medium._has_incompatibilities:
                incompatible_indices.append(i)
                incompatible_structures.append(structure)
        for i, (ind1, structure_ind1) in enumerate(
            zip(incompatible_indices, incompatible_structures)
        ):
            for ind2, structure_ind2 in zip(
                incompatible_indices[i + 1 :], incompatible_structures[i + 1 :]
            ):
                if not structure_ind1._compatible_with(structure_ind2):
                    raise ValidationError(
                        f"The structure at 'structures[{ind1}]' and the structure at "
                        f"'structures[{ind2}]' have incompatible medium types "
                        f"{structure_ind1.medium._incompatible_material_types} and "
                        f"{structure_ind2.medium._incompatible_material_types} "
                        "respectively, and so are not allowed to intersect. "
                        "Please ensure that the bounding boxes of the two geometries "
                        "do not intersect."
                    )
        return val

    @field_validator("monitors")
    @classmethod
    def _projection_direction(cls, val: tuple[MonitorType, ...]) -> tuple[MonitorType, ...]:
        """Warn if field projection observation points are behind surface projection monitors."""
        # This validator belongs to the concrete simulation model rather than ``monitor.py``:
        # volume monitors are eventually converted to bounding surface projection monitors, and
        # this check must not run during that conversion.
        if val is None:
            return val

        with log as consolidated_logger:
            for monitor_ind, monitor in enumerate(val):
                if isinstance(monitor, AbstractFieldProjectionMonitor):
                    if monitor.size.count(0.0) != 1:
                        continue

                    normal_dir = monitor.projection_surfaces[0].normal_dir
                    normal_ind = monitor.size.index(0.0)

                    projecting_backwards = False
                    if isinstance(monitor, FieldProjectionAngleMonitor):
                        r, theta, phi = np.meshgrid(
                            monitor.proj_distance,
                            monitor.theta,
                            monitor.phi,
                            indexing="ij",
                        )
                        x, y, z = Geometry.sph_2_car(r=r, theta=theta, phi=phi)
                    elif isinstance(monitor, FieldProjectionKSpaceMonitor):
                        uxs, uys, _ = np.meshgrid(
                            monitor.ux,
                            monitor.uy,
                            monitor.proj_distance,
                            indexing="ij",
                        )
                        theta, phi = monitor.kspace_2_sph(uxs, uys, monitor.proj_axis)
                        x, y, z = Geometry.sph_2_car(r=monitor.proj_distance, theta=theta, phi=phi)
                    else:
                        pts = monitor.unpop_axis(
                            monitor.proj_distance, (monitor.x, monitor.y), axis=normal_ind
                        )
                        x, y, z = pts

                    center = np.array(monitor.center) - np.array(monitor.local_origin)
                    pts = [np.array(i) for i in [x, y, z]]
                    normal_displacement = pts[normal_ind] - center[normal_ind]
                    if (np.any(normal_displacement < 0) and normal_dir == "+") or (
                        np.any(normal_displacement > 0) and normal_dir == "-"
                    ):
                        projecting_backwards = True

                    if projecting_backwards:
                        consolidated_logger.warning(
                            f"Field projection monitor '{monitor.name}' has observation points set "
                            "up such that the monitor is projecting backwards with respect to its "
                            "'normal_dir'. If this was not intentional, please take a look at the "
                            "documentation associated with this type of projection monitor to "
                            "check how the observation point coordinate system is defined.",
                            custom_loc=["monitors", monitor_ind],
                        )

        return val

    def _preprocess_cache_ineligibility(self) -> tuple[str, tuple[str | int, ...]] | None:
        """Return why preprocess caching is unsupported and the location of the cause."""
        if has_protected_media(self):
            return "Preprocess caching is not supported for protected optical media.", ()
        if self.simulation_type != "tidy3d":
            return "Preprocess caching is not supported for autograd simulations.", ()

        if self.medium.is_custom:
            return "Preprocess caching does not support custom media.", ("medium",)
        for index, structure in enumerate(self.structures):
            if structure.medium.is_custom:
                return "Preprocess caching does not support custom media.", (
                    "structures",
                    index,
                    "medium",
                )

        unsupported_monitor_types = (
            AbstractMediumPropertyMonitor,
            FieldStructureMonitor,
            SurfaceFieldMonitor,
            SurfaceFieldStructureMonitor,
            SurfaceFieldTimeMonitor,
        )
        for index, monitor in enumerate(self.monitors):
            if isinstance(monitor, unsupported_monitor_types):
                return (
                    f"Preprocess caching does not support '{type(monitor).__name__}' monitors.",
                    ("monitors", index),
                )
            # E/H point-cloud data needs only replayed coefficients; D also needs material data.
            if isinstance(monitor, PointCloudFieldMonitor) and any(
                field.startswith("D") for field in monitor.fields
            ):
                return (
                    "Preprocess caching does not support displacement fields in "
                    "'PointCloudFieldMonitor' monitors.",
                    ("monitors", index),
                )
        return None

    # Pre-upload validation.

    def check_ports_reach_pml(self) -> list[tuple[ModeSource | ModeMonitor, Structure, str]]:
        """Return mode ports whose overlapping structures do not reach through the absorbing region.

        Each result is a ``(port, structure, face)`` tuple. ``face`` is one of ``x-``, ``x+``,
        ``y-``, ``y+``, ``z-``, or ``z+``. Every absorbing face along a port's normal is checked.
        Pre-upload warnings are opt-in via ``config.validation.check_ports_reach_pml = True``
        and run only in full validation mode, for ports within one absorber thickness of a
        boundary. This method also checks farther ports and works regardless of the setting or mode.
        Connected pieces within a structure are checked together.

        Returns
        -------
        list[tuple[ModeSource | ModeMonitor, Structure, str]]
            Port, structure, and face tuples for structures that cross the port window but do not
            reach the outer edge of an absorbing region and are not covered by automatic structure
            extrusion.

        Examples
        --------
        >>> from tidy3d import Box, GaussianPulse, GridSpec, Medium, ModeSource, Structure
        >>> sim = Simulation(
        ...     size=(10, 6, 6),
        ...     grid_spec=GridSpec.auto(wavelength=1.0),
        ...     run_time=1e-12,
        ...     structures=(Structure(geometry=Box(size=(8.2, 1, 1)), medium=Medium(permittivity=2)),),
        ...     sources=(ModeSource(
        ...         center=(-4, 0, 0),
        ...         size=(0, 3, 3),
        ...         source_time=GaussianPulse(freq0=3e14, fwidth=1e13),
        ...         direction="+",
        ...         name="input",
        ...     ),),
        ... )
        >>> [(port.name, face) for port, _, face in sim.check_ports_reach_pml()]
        [('input', 'x-'), ('input', 'x+')]
        """
        return [
            (port, self.structures[structure_index], face)
            for port, structure_index, face in self._find_ports_missing_pml()
        ]

    def _find_ports_missing_pml(
        self, *, near_absorber: bool = False
    ) -> list[tuple[ModeSource | ModeMonitor, int, str]]:
        """Find mode ports whose connected structure parts do not reach an absorber's outer edge."""
        if not self.structures:
            return []

        # Keep exact section results only for this call; avoid hashing geometry vertex arrays.
        section_cache: dict[tuple[int, tuple[float, ...], tuple[float, ...]], list[Any]] = {}

        def sections_for(plane: Box, geometry: Geometry) -> list[Any]:
            key = (id(geometry), plane.center, plane.size)
            if key not in section_cache:
                section_cache[key] = plane.intersections_with(geometry)
            return section_cache[key]

        boundaries = self.boundary_spec.to_list
        offenders = []
        bounds = None
        pml_thicknesses = None
        for port in (*self.sources, *self.monitors):
            if not isinstance(port, (ModeSource, ModeMonitor)):
                continue
            axis = port.size.index(0.0)
            if self.size[axis] == 0:
                continue

            sides = []
            for side, boundary in enumerate(boundaries[axis]):
                if not isinstance(boundary, _ABSORBING_BOUNDARIES):
                    continue
                if near_absorber:
                    if pml_thicknesses is None:
                        pml_thicknesses = self.pml_thicknesses
                        bounds = self.bounds
                    distance = abs(port.center[axis] - bounds[side][axis])
                    if distance > pml_thicknesses[axis][side]:
                        continue
                sides.append((side, boundary))
            if not sides:
                continue

            # The port-plane intersection is independent of which absorbing side is checked.
            structure_parts = []
            for structure_index, structure in enumerate(self.structures):
                geometry = structure.geometry
                port_parts = [
                    part
                    for section in sections_for(port, geometry)
                    for part in getattr(section, "geoms", (section,))
                    if part.area > 0
                ]
                if port_parts:
                    structure_parts.append((structure_index, geometry, port_parts))

            if not structure_parts:
                continue

            for side, boundary in sides:
                face = self.simulation_bounds[side][axis]
                face_center = list(port.center)
                face_center[axis] = face
                face_plane = Box(center=tuple(face_center), size=port.size)
                clipping_plane = None
                if boundary.extrude_structures:
                    clipping_index = self._pml_extrusion_clipping_bound_ind(axis, side)
                    if clipping_index is not None:
                        clipping_center = list(port.center)
                        clipping_center[axis] = self.grid.boundaries.to_list[axis][clipping_index]
                        clipping_plane = Box(center=tuple(clipping_center), size=port.size)
                face_name = "xyz"[axis] + ("-" if side == 0 else "+")
                for structure_index, geometry, port_parts in structure_parts:
                    geometry_face_bound = geometry.bounds[side][axis]
                    can_reach_absorber_edge = (
                        geometry_face_bound <= face if side == 0 else geometry_face_bound >= face
                    )
                    face_sections = (
                        sections_for(face_plane, geometry) if can_reach_absorber_edge else []
                    )
                    clipping_sections = None
                    for port_part in port_parts:
                        reaches_absorber_edge = any(
                            (face_section.intersection(port_part)).area > 0
                            for face_section in face_sections
                        )
                        if reaches_absorber_edge:
                            continue
                        if clipping_sections is None:
                            clipping_sections = (
                                sections_for(clipping_plane, geometry)
                                if clipping_plane is not None
                                else []
                            )
                        is_extruded = any(
                            (clipping_section.intersection(port_part)).area > 0
                            for clipping_section in clipping_sections
                        )
                        if is_extruded:
                            continue
                        offenders.append((port, structure_index, face_name))
                        break
        return offenders

    def _warn_mode_ports_not_reaching_absorbing_boundaries(self) -> None:
        """Optionally warn when nearby mode-port waveguides stop before an absorber's outer edge."""
        if not config.validation.check_ports_reach_pml or fast_validation_active():
            return
        with log as consolidated_logger:
            for port, structure_index, face in self._find_ports_missing_pml(near_absorber=True):
                structure = self.structures[structure_index]
                port_name = port.name or type(port).__name__
                obj_descr = named_obj_descr(structure, "structures", structure_index)
                consolidated_logger.warning(
                    f"Mode port '{port_name}' on {face} crosses {obj_descr}, which does not span "
                    "the full absorbing region. Extend the structure to `td.inf` or fully "
                    "through the absorber to avoid artificial reflections.",
                    custom_loc=["structures", structure_index],
                )

    def validate_pre_upload(self, source_required: bool = True) -> None:
        """Validate the fully initialized simulation is ok for upload to our servers.

        Parameters
        ----------
        source_required: bool = True
            If ``True``, validation will fail in case no sources are found in the simulation.
        """
        # run before super(): catches a degenerate (single-cell transverse axis) line element with a
        # clear message, ahead of the finalized-simulation build that would otherwise surface it as a
        # cryptic "zero volume" probe error
        self._validate_protected_monitor_observation_policy()
        self._validate_lumped_element_grid_size()
        super().validate_pre_upload()
        log.begin_capture()
        self._validate_size()
        self._validate_monitor_size()
        self._validate_gaussian_like_beam_backgrounds()
        self._validate_thin_lens_setup_size()
        self._validate_modes_size()
        self._validate_num_cells_in_mode_objects()
        self._validate_datasets_not_none()
        self._validate_tfsf_structure_intersections()
        self._warn_time_monitors_outside_run_time()
        self._validate_time_monitors_num_steps()
        self._validate_freq_monitors_freq_range()
        self._validate_microwave_mode_specs()
        self._warn_mode_ports_not_reaching_absorbing_boundaries()
        log.end_capture(self)
        if source_required and len(self.sources) == 0:
            raise SetupError("No sources in simulation.")

    def _validate_protected_monitor_observation_policy(self) -> None:
        """Reject monitor outputs that expose protected material properties."""
        if not has_protected_media(self):
            return

        for monitor_index, monitor in enumerate(self.monitors):
            error_message = protected_fdtd_monitor_policy_error(monitor, monitor_index)
            if error_message:
                self._raise_validation_error_at_loc(error_message, "monitors", monitor_index)
