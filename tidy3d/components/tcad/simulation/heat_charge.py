"""Defines heat simulation class"""

from __future__ import annotations

import math
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from pydantic import Field, field_validator, model_validator

from tidy3d.components.base_sim.simulation import AbstractSimulation
from tidy3d.components.bc_placement import (
    MediumMediumInterface,
    SimulationBoundary,
    StructureBoundary,
    StructureSimulationBoundary,
    StructureStructureInterface,
)
from tidy3d.components.geometry.base import Box, Transformed
from tidy3d.components.geometry.polyslab import PolySlab
from tidy3d.components.geometry.primitives import Cylinder
from tidy3d.components.geometry.utils import flatten_groups
from tidy3d.components.material.tcad.charge import (
    AbstractChargeMedium,
    AnisotropicPermittivity,
    ChargeConductorMedium,
    ChargeInsulatorMedium,
    SemiconductorMedium,
)
from tidy3d.components.material.tcad.heat import (
    AbstractConductivityModel,
    AbstractHeatCapacityModel,
    AnisotropicConductivity,
    FluidMedium,
    SolidMedium,
)
from tidy3d.components.material.types import MultiPhysicsMedium, StructureMediumType
from tidy3d.components.medium import Medium
from tidy3d.components.medium.protected import assert_no_protected_media
from tidy3d.components.scene import Scene
from tidy3d.components.spice.analysis.dc import ITERATIVE_ONLY_TOLERANCE_FIELDS
from tidy3d.components.spice.sources.ac import SSACVoltageSource
from tidy3d.components.spice.sources.dc import DCCurrentSource, DCVoltageSource, GroundVoltage
from tidy3d.components.spice.types import (
    ElectricalAnalysisType,
    IsothermalSSACAnalysis,
    IsothermalSteadyChargeDCAnalysis,
    SSACAnalysis,
    SteadyChargeDCAnalysis,
)
from tidy3d.components.structure import Structure
from tidy3d.components.tcad.analysis.heat_simulation_type import UnsteadyHeatAnalysis
from tidy3d.components.tcad.bandgap import VarshniEnergyBandGap
from tidy3d.components.tcad.boundary.heat import VerticalNaturalConvectionCoeffModel
from tidy3d.components.tcad.boundary.specification import HeatBoundarySpec, HeatChargeBoundarySpec
from tidy3d.components.tcad.effective_DOS import ConstantEffectiveDOS
from tidy3d.components.tcad.generation_recombination import (
    HeterojunctionSurfaceShockleyReedHallRecombination,
    PalankovskiQuayApproxCarrierLifetime,
    SelberherrImpactIonization,
    ShockleyReedHallRecombination,
    SurfaceShockleyReedHallRecombination,
)
from tidy3d.components.tcad.grid import (
    AutoUnstructuredGrid,
    DistanceUnstructuredGrid,
    UniformUnstructuredGrid,
    UnstructuredGridType,
)
from tidy3d.components.tcad.mobility import (
    CANALI_BETA_MAX,
    CANALI_BETA_MIN,
    V_SAT_FLOOR,
    AlloyMobility,
    CanaliFieldDependence,
    CaugheyThomasMobility,
    ConstantMobilityModel,
    MasettiMobility,
)
from tidy3d.components.tcad.monitors.charge import (
    SelfHeatingMonitor,
    SteadyCapacitanceMonitor,
    SteadyChargeResidualMonitor,
    SteadyCurrentDensityMonitor,
    SteadyDopingMonitor,
    SteadyElectricFieldMonitor,
    SteadyFreeCarrierMonitor,
    SteadyGenerationRecombinationMonitor,
    SteadyPotentialMonitor,
)
from tidy3d.components.tcad.monitors.heat import TemperatureMonitor
from tidy3d.components.tcad.source.abstract import GlobalHeatChargeSource
from tidy3d.components.tcad.types import (
    ConvectionBC,
    CurrentBC,
    HeatChargeMonitorType,
    HeatChargeSourceType,
    HeatFluxBC,
    HeatFromElectricSource,
    HeatSource,
    InsulatingBC,
    RadiationBC,
    SurfaceRecombinationBC,
    TemperatureBC,
    ThermalContactResistance,
    UniformHeatSource,
    VoltageBC,
)
from tidy3d.components.tcad.viz import (
    CHARGE_BC_INSULATOR,
    HEAT_BC_COLOR_CONVECTION,
    HEAT_BC_COLOR_FLUX,
    HEAT_BC_COLOR_TEMPERATURE,
    HEAT_SOURCE_CMAP,
    plot_params_heat_bc,
    plot_params_heat_source,
)
from tidy3d.components.types import TYPE_TAG_STR, ScalarSymmetry
from tidy3d.components.types.base import discriminated_union
from tidy3d.components.viz import add_ax_if_none, equal_aspect
from tidy3d.constants import VOLUMETRIC_HEAT_RATE, inf
from tidy3d.exceptions import SetupError
from tidy3d.log import log

if TYPE_CHECKING:
    from collections.abc import Iterator

    from pydantic import FiniteFloat

    from tidy3d.compat import Self
    from tidy3d.components.bc_placement import BCPlacementType
    from tidy3d.components.geometry.base import Geometry
    from tidy3d.components.types import Ax, Bound, Shapely
    from tidy3d.components.types.base import ArrayFloat1D, BoxSurface
    from tidy3d.components.viz import PlotParams

HEAT_CHARGE_BACK_STRUCTURE_STR = "<<<HEAT_CHARGE_BACKGROUND_STRUCTURE>>>"

HeatBCTypes = (TemperatureBC, HeatFluxBC, ConvectionBC, RadiationBC, ThermalContactResistance)
# Neumann in the temperature rows, exactly as the adiabatic wall a dropped condition
# leaves behind is: losing one cannot leave the steady solve unanchored.
NeumannHeatBCTypes = (HeatFluxBC, ThermalContactResistance)
HeatSourceTypes = (UniformHeatSource, HeatSource, HeatFromElectricSource)
ChargeSourceTypes = ()
ElectricBCTypes = (VoltageBC, CurrentBC, InsulatingBC, SurfaceRecombinationBC)
ChargeTypes = (
    SteadyChargeDCAnalysis,
    IsothermalSteadyChargeDCAnalysis,
    SSACAnalysis,
    IsothermalSSACAnalysis,
)
ChargeMonitorTypes = (
    SteadyPotentialMonitor,
    SteadyFreeCarrierMonitor,
    SteadyCapacitanceMonitor,
    SteadyCurrentDensityMonitor,
    SteadyChargeResidualMonitor,
    SteadyGenerationRecombinationMonitor,
    SteadyDopingMonitor,
    SelfHeatingMonitor,
)

AnalysisSpecType = ElectricalAnalysisType | UnsteadyHeatAnalysis


# Minimum tolerance for cylinder radii
CYLINDER_RADIUS_TOL = 1e-6
# Minimum radius as fraction of the larger radius (for tapered cylinders)
MIN_CYLINDER_RADIUS_FRACTION = 0.01

# Absolute tolerance (V) for matching SSAC `at_voltages` against the DC sweep.
# Both lists travel through JSON as IEEE-754 doubles, so a machine-epsilon-scale
# value is intentional — any larger tolerance would silently accept a voltage
# the user did not configure. Mirrored by SSAC_VOLTAGE_MATCH_TOL_V in
# Flow360DriftDiffusionSolver.cpp; keep in sync.
SSAC_VOLTAGE_MATCH_TOL_V = 1e-14

# Band parameters further apart than this, relative to the larger (absolute below 1),
# make two semiconductors a heterojunction.
HETEROJUNCTION_REL_TOL = 1e-12


def _band_signature(medium: SemiconductorMedium) -> tuple[tuple[Any, ...], ...]:
    """Band parameters that set a band offset, one ``(kind, *values)`` per field.

    Doping and bandgap narrowing are excluded. An effective DOS reduces to its
    temperature law and ``N(T=1)``, so forms that give the same ``N(T)`` match.
    """
    signature: list[tuple[Any, ...]] = []
    for dos in (medium.N_c, medium.N_v):
        if isinstance(dos, ConstantEffectiveDOS):
            signature.append(("constant", dos.N))
        else:
            signature.append(("temperature_dependent", float(dos.calc_eff_dos(1.0))))
    signature.append(_band_gap_signature(medium))
    signature.append(("affinity", medium.electron_affinity or 0.0))
    return tuple(signature)


def _band_gap_signature(medium: SemiconductorMedium) -> tuple[Any, ...]:
    """The band-gap model as ``(kind, *values)``."""
    band_gap = medium.E_g
    if isinstance(band_gap, VarshniEnergyBandGap):
        return ("varshni", band_gap.eg_0, band_gap.alpha, band_gap.beta)
    return ("constant", band_gap.eg)


def _band_fields_differ(field_a: tuple[Any, ...], field_b: tuple[Any, ...]) -> bool:
    """Whether two ``(kind, *values)`` band fields differ beyond ``HETEROJUNCTION_REL_TOL``."""
    return field_a[0] != field_b[0] or any(
        abs(a - b) > HETEROJUNCTION_REL_TOL * max(1.0, abs(a), abs(b))
        for a, b in zip(field_a[1:], field_b[1:], strict=True)
    )


def _is_material_heterojunction(
    medium_a: SemiconductorMedium, medium_b: SemiconductorMedium
) -> bool:
    """Whether two semiconductors meet in a band offset."""
    return any(
        _band_fields_differ(field_a, field_b)
        for field_a, field_b in zip(
            _band_signature(medium_a), _band_signature(medium_b), strict=True
        )
    )


def _get_cylinder_radii_with_meshing_tol(
    geometry: Cylinder, min_mesh_size: float = 0
) -> tuple[float, float]:
    """Get cylinder radii clamped to the minimum meshing tolerance.

    This function clamps small or negative values to a small positive value to ensure
    valid geometry that can be meshed. The minimum is set relative to the
    larger radius to ensure meshability while still creating a reasonably sharp
    tip for tapered cylinders. If ``min_mesh_size`` is provided, radii are also
    clamped to that value.

    Parameters
    ----------
    geometry : Cylinder
        The cylinder geometry to get radii from.
    min_mesh_size : float, optional
        Minimum mesh size from the grid specification. When positive, radii are
        additionally clamped to this value.

    Returns
    -------
    tuple[float, float]
        ``(r1, r2)`` -- bottom and top radii, clamped to the minimum allowed radius.
    """
    r_bottom = geometry.radius_bottom
    r_top = geometry.radius_top
    is_tapered = not np.isclose(r_bottom, r_top)

    min_radius = max(
        CYLINDER_RADIUS_TOL,
        MIN_CYLINDER_RADIUS_FRACTION * max(abs(r_bottom), abs(r_top)),
    )
    if min_mesh_size > 0:
        min_radius = max(min_radius, min_mesh_size)

    r1 = max(r_bottom, min_radius)
    r2 = max(r_top, min_radius)

    if is_tapered:
        if r1 > r_bottom:
            log.warning(
                f"Cylinder 'radius_bottom' ({r_bottom:.3e}) is below the minimum "
                f"radius for meshing ({r1:.3e}). The sidewall angle may be "
                f"too steep. Will be clamped to {r1:.3e}."
            )
        if r2 > r_top:
            log.warning(
                f"Cylinder 'radius_top' ({r_top:.3e}) is below the minimum "
                f"radius for meshing ({r2:.3e}). The sidewall angle may be "
                f"too steep. Will be clamped to {r2:.3e}."
            )
    else:
        if r1 > r_bottom:
            log.warning(
                f"Cylinder 'radius' ({r_bottom:.3e}) is below the minimum "
                f"radius for meshing ({r1:.3e}). "
                f"Will be clamped to {r1:.3e}."
            )

    return r1, r2


class TCADAnalysisTypes(str, Enum):
    """Enumeration of the types of simulations currently supported"""

    HEAT = "Heat"
    CONDUCTION = "Conduction"
    CHARGE = "Charge"
    MESH = "Mesh"


class HeatChargeSimulation(AbstractSimulation):
    """
    Defines thermoelectric simulations.

    Notes
    -----
        A ``HeatChargeSimulation`` supports different types of simulations. It solves the
        heat and conduction equations using the Finite-Volume (FV) method. This solver
        determines the required computation physics according to the simulation scene definition.
        This is implemented in this way due to the strong multi-physics coupling.

    The ``HeatChargeSimulation`` can solve multiple physics and the intention is to enable close thermo-electrical coupling.

    Currently, this solver supports steady-state heat conduction where :math:`q` is the heat flux, :math:`k`
    is the thermal conductivity, and :math:`T` is the temperature.

    .. math::

        -\\nabla \\cdot (-k \\nabla T) = q

    It is also possible to run transient heat simulations by specifying ``analysis_spec=UnsteadyHeatAnalysis(...)``. This adds
    the temporal terms to the above equations:

    .. math::

        \\frac{\\partial \\rho c_p T}{\\partial t} -\\nabla \\cdot (k \\nabla(T)) = q

    where :math:`\\rho` is the density and :math:`c_p` is the specific heat capacity of the medium.


    The steady-state electrical ``Conduction`` equation depends on the electric conductivity (:math:`\\sigma`)  of a
    medium, and the electric field (:math:`\\mathbf{E} = -\\nabla(\\psi)`) derived from electrical potential (:math:`\\psi`).
    Currently, in this type of simulation, no current sources or sinks are supported.

    .. math::

        \\text{div}(\\sigma \\cdot \\nabla(\\psi)) = 0


    For further details on what equations are solved in ``Charge`` simulations, refer to the :class:`SemiconductorMedium`.

    Let's understand how the physics solving is determined:

        .. list-table::
           :widths: 25 75
           :header-rows: 1

           * - Simulation Type
             - Example Configuration Settings
           * - ``Heat``
             - The heat equation is solved with specified heat sources,
               boundary conditions, etc. Structures should incorporate materials
               with defined heat properties.
           * - ``Conduction``
             - The electrical conduction equation is solved with
               specified boundary conditions such as :class:`VoltageBC`, :class:`CurrentBC`, ...
           * - ``Charge``
             - Drift-diffusion equations are solved for structures containing
               a defined :class:`SemiconductorMedium`. Insulators with a
               :class:`ChargeInsulatorMedium` can also be included. For these, only the
               electric potential field is calculated.

    Examples
    --------
    To run a thermal (``Heat`` |:fire:|) simulation with a solid conductive structure:

    >>> import tidy3d as td
    >>> heat_sim = td.HeatChargeSimulation(
    ...     size=(3.0, 3.0, 3.0),
    ...     structures=[
    ...         td.Structure(
    ...             geometry=td.Box(size=(1, 1, 1), center=(0, 0, 0)),
    ...             medium=td.Medium(
    ...                 permittivity=2.0,
    ...                 heat_spec=td.SolidSpec(
    ...                     conductivity=1,
    ...                     capacity=1,
    ...                 )
    ...             ),
    ...             name="box",
    ...         ),
    ...     ],
    ...     medium=td.Medium(permittivity=3.0, heat_spec=td.FluidSpec()),
    ...     grid_spec=td.UniformUnstructuredGrid(
    ...         dl=0.1, min_edges_per_circumference=15, min_edges_per_side=2
    ...     ),
    ...     sources=[td.HeatSource(rate=1, structures=["box"])],
    ...     boundary_spec=[
    ...         td.HeatChargeBoundarySpec(
    ...             placement=td.StructureBoundary(structure="box"),
    ...             condition=td.TemperatureBC(temperature=500),
    ...         )
    ...     ],
    ...     monitors=[td.TemperatureMonitor(size=(1, 2, 3), name="sample", unstructured=True)],
    ... )

    To run a drift-diffusion (``Charge`` |:zap:|) system:

    >>> import tidy3d as td
    >>> air = td.FluidMedium(
    ...     name="air"
    ... )
    >>> intrinsic_Si = td.material_library['cSi'].variants['Si_MultiPhysics'].medium.charge
    >>> Si_n = intrinsic_Si.updated_copy(N_d=[td.ConstantDoping(concentration=1e16)], name="Si_n")
    >>> Si_p = intrinsic_Si.updated_copy(N_a=[td.ConstantDoping(concentration=1e16)], name="Si_p")
    >>> n_side = td.Structure(
    ...     geometry=td.Box(center=(-0.5, 0, 0), size=(1, 1, 1)),
    ...     medium=Si_n,
    ...     name="n_side"
    ... )
    >>> p_side = td.Structure(
    ...     geometry=td.Box(center=(0.5, 0, 0), size=(1, 1, 1)),
    ...     medium=Si_p,
    ...     name="p_side"
    ... )
    >>> bc_v1 = td.HeatChargeBoundarySpec(
    ...     condition=td.VoltageBC(source=td.DCVoltageSource(voltage=[-1, 0, 0.5])),
    ...     placement=td.MediumMediumInterface(mediums=[air.name, Si_n.name]),
    ... )
    >>> bc_v2 = td.HeatChargeBoundarySpec(
    ...     condition=td.VoltageBC(source=td.DCVoltageSource(voltage=0)),
    ...     placement=td.MediumMediumInterface(mediums=[air.name, Si_p.name]),
    ... )
    >>> charge_sim = td.HeatChargeSimulation(
    ...     structures=[n_side, p_side],
    ...     medium=td.Medium(heat_spec=td.FluidSpec(), name="air"),
    ...     monitors=[td.SteadyFreeCarrierMonitor(
    ...         center=(0, 0, 0), size=(td.inf, td.inf, 0), name="charge_mnt", unstructured=True
    ...     )],
    ...     center=(0, 0, 0),
    ...     size=(3, 3, 3),
    ...     grid_spec=td.UniformUnstructuredGrid(
    ...         dl=0.05, min_edges_per_circumference=15, min_edges_per_side=2
    ...     ),
    ...     boundary_spec=[bc_v1, bc_v2],
    ...     analysis_spec=td.IsothermalSteadyChargeDCAnalysis(
    ...         tolerance_settings=td.ChargeToleranceSpec(),
    ...         convergence_dv=10),
    ...     )


    Coupling between ``Heat`` and electrical ``Conduction`` simulations is currently limited to 1-way.
    This is specified by defining a heat source of type :class:`HeatFromElectricSource`. With this coupling, joule heating is
    calculated as part  of the solution to a ``Conduction`` simulation and translated into the ``Heat`` simulation.

    Two common scenarios can use this coupling definition:
        1. One in which BCs and sources are specified for both ``Heat`` and ``Conduction`` simulations.
            In this case one mesh will be generated and used for both the ``Conduction`` and ``Heat``
            simulations.
        2. Only heat BCs/sources are provided. In this case, only the ``Heat`` equation will be solved.
            Before the simulation starts, it will try to load the heat source from file so a
            previously run ``Conduction`` simulations must have run previously. Since the Conduction
            and ``Heat`` meshes may differ, an interpolation between them will be performed prior to
            starting the ``Heat`` simulation.

    Additional heat sources can be defined, in which case, they will be added on
    top of the coupling heat source.
    """

    medium: StructureMediumType = Field(
        default_factory=Medium,
        title="Background Medium",
        description="Background medium of simulation, defaults to a standard dispersion-less :class:`.Medium` if not "
        "specified.",
    )
    """
    Background medium of simulation, defaults to a standard dispersion-less :class:`.Medium` if not specified.
    """

    sources: tuple[discriminated_union(HeatChargeSourceType), ...] = Field(
        default=(),
        title="Heat and Charge sources",
        description="List of heat and/or charge sources.",
    )

    monitors: tuple[discriminated_union(HeatChargeMonitorType), ...] = Field(
        default=(),
        title="Monitors",
        description="Monitors in the simulation.",
    )

    boundary_spec: tuple[discriminated_union(HeatChargeBoundarySpec | HeatBoundarySpec), ...] = (
        Field(
            default=(),
            title="Boundary Condition Specifications",
            description="List of boundary condition specifications.",
        )
    )
    # NOTE: creating a union with HeatBoundarySpec for backwards compatibility

    grid_spec: UnstructuredGridType = Field(
        title="Grid Specification",
        description="Grid specification for heat-charge simulation.",
        discriminator=TYPE_TAG_STR,
    )

    symmetry: tuple[ScalarSymmetry, ScalarSymmetry, ScalarSymmetry] = Field(
        default=(0, 0, 0),
        title="Symmetries",
        description="Tuple of integers defining reflection symmetry across a plane "
        "bisecting the simulation domain normal to the x-, y-, and z-axis "
        "at the simulation center of each axis, respectively. "
        "Each element can be ``0`` (symmetry off) or ``1`` (symmetry on). "
        "The accelerated solver reports :class:`.DeviceCharacteristics` for the whole device, "
        "while the CPU solver reports them for the simulated portion.",
    )

    analysis_spec: AnalysisSpecType | None = Field(
        default=None,
        discriminator=TYPE_TAG_STR,
        title="Analysis specification.",
        description="The `analysis_spec` is used to specify the type of simulation. Currently, it is used to "
        "specify Charge simulations or transient Heat simulations.",
    )

    use_accelerated_solver: bool = Field(
        default=True,
        title="Use accelerated solver.",
        description="Controls the solver used for charge simulations. When ``True`` "
        "(default), the GPU accelerated charge solver is used. Set to ``False`` "
        "to use the CPU charge solver instead; this is rejected when the simulation uses "
        "a feature available only on the GPU accelerated solver, such as ``MasettiMobility``. "
        "The flag applies only to charge simulations: heat and conduction simulations "
        "always run on the GPU accelerated solver, so ``False`` is not allowed for them.",
    )

    linear_solver: Literal["iterative", "direct"] = Field(
        default="iterative",
        title="Linear solver",
        description="Selects how the linear system arising at each nonlinear iteration is "
        "solved, in every solver this simulation runs. ``'iterative'`` (the default) uses a "
        "preconditioned Krylov method. ``'direct'`` factorizes the system matrix instead, "
        "which removes linear-solver error as a cause of nonlinear stall and is therefore "
        "more robust on cases where the iterative solver converges slowly or fails; in "
        "exchange it needs substantially more memory, increasingly so as the mesh grows, "
        "and can be slower. One setting covers the whole run: a charge analysis that also "
        "solves for the lattice temperature uses the same back-end for the coupled heat "
        "solve, and the two factorizations are sized together before either is allocated. "
        "On a charge analysis the iterative-solver settings in ``ChargeToleranceSpec`` "
        "(``max_iters``, ``preconditioner_iterations``, ``linear_rel_tol``) are ignored when "
        "``'direct'`` is selected. A charge analysis requires the accelerated solver.",
    )

    @field_validator("structures")
    @classmethod
    def _check_unsupported_geometries(cls, val: tuple[Structure, ...]) -> tuple[Structure, ...]:
        """Error if structures contain unsupported yet geometries."""
        for ind, structure in enumerate(val):
            bbox = structure.geometry.bounding_box
            if any(s == 0 for s in bbox.size):
                raise SetupError(
                    f"'HeatSimulation' does not currently support structures with dimensions of zero size ('structures[{ind}]')."
                )
            for geometry in flatten_groups(
                structure.geometry, flatten_nonunion_type=True, flatten_transformed=True
            ):
                base_geometry = geometry.geometry if isinstance(geometry, Transformed) else geometry
                if isinstance(base_geometry, PolySlab) and base_geometry._has_arc_segments:
                    raise SetupError(
                        f"'HeatChargeSimulation' does not currently support arc segments in "
                        f"'PolySlab' geometries ('structures[{ind}]'). Set all 'bulges' to 0 "
                        "to use a straight-edge polyslab."
                    )
        return val

    @field_validator("structures")
    @classmethod
    def _warn_small_cylinder_radius(cls, val: tuple[Structure, ...]) -> tuple[Structure, ...]:
        """Warn if any Cylinder geometry has radius too small for meshing."""
        for structure in val:
            for geometry in flatten_groups(
                structure.geometry, flatten_nonunion_type=True, flatten_transformed=True
            ):
                # Unwrap Transformed to get the base geometry
                base_geometry = geometry.geometry if isinstance(geometry, Transformed) else geometry
                if isinstance(base_geometry, Cylinder):
                    _get_cylinder_radii_with_meshing_tol(base_geometry)
        return val

    def _check_cross_solids(self, objs: tuple[Box, ...]) -> tuple[int, ...]:
        """Given model dictionary ``values``, check whether objects in list ``objs`` cross
        a ``SolidSpec`` medium.
        """

        # NOTE: when considering Conduction or Charge cases, both conductors and semiconductors
        # will be accepted
        valid_electric_medium = (SemiconductorMedium, ChargeConductorMedium)

        # list of structures including background as a Box()
        structure_bg = Structure(
            geometry=Box(
                size=self.size,
                center=self.center,
            ),
            medium=self.medium,
        )

        total_structures = [structure_bg, *list(self.structures)]

        obj_do_not_cross_solid_idx = []
        obj_do_not_cross_cond_idx = []
        for ind, obj in enumerate(objs):
            if obj.size.count(0.0) == 1:
                # for planar objects we could do a rigorous check
                medium_set = Scene.intersecting_media(obj, total_structures)
                crosses_solid = any(
                    isinstance(medium.heat_spec, SolidMedium) for medium in medium_set
                )
                crosses_elec_spec = any(
                    isinstance(medium.charge, valid_electric_medium) for medium in medium_set
                )
            else:
                # approximate check for volumetric objects based on bounding boxes
                # thus, it could still miss a case when there is no data inside the monitor
                crosses_solid = any(
                    obj.intersects(structure.geometry)
                    for structure in total_structures
                    if isinstance(structure.medium.heat_spec, SolidMedium)
                )
                crosses_elec_spec = any(
                    obj.intersects(structure.geometry)
                    for structure in total_structures
                    if isinstance(structure.medium.charge, valid_electric_medium)
                )

            if not crosses_solid:
                obj_do_not_cross_solid_idx.append(ind)
            if not crosses_elec_spec:
                obj_do_not_cross_cond_idx.append(ind)

        return obj_do_not_cross_solid_idx, obj_do_not_cross_cond_idx

    @model_validator(mode="after")
    def _run_after_validators(self) -> Self:
        """Run post-init validations in an explicit, dependency-aware order."""
        self._call_with_validation_loc(("structures",), super()._run_after_validators)
        self._call_with_validation_loc(("structures",), self._structures_not_at_edges)
        self._call_with_validation_loc(("structures",), self._validate_scene)
        self._call_with_validation_loc(("monitors",), self._monitors_cross_solids)
        self._call_with_validation_loc(("monitors",), self._check_monitor_dimensionality)
        self._call_with_validation_loc(("symmetry",), self._check_symmetry_not_normal_to_zero_dim)
        self._check_single_ssac()
        self._check_only_one_voltage_array_provided()
        self._call_with_validation_loc(("boundary_spec",), self._check_voltage_array_if_capacitance)
        self._call_with_validation_loc(("boundary_spec",), self._names_exist_bcs)
        self._call_with_validation_loc(("boundary_spec",), self._check_natural_convection_bc)
        self._call_with_validation_loc(
            ("boundary_spec",), self._check_thermal_contact_resistance_placement
        )
        self._call_with_validation_loc(
            ("structures",), self._check_heat_feature_placement_in_charge
        )
        self._call_with_validation_loc(("structures",), self._check_solids_have_charge_medium)
        self._call_with_validation_loc(("structures",), self._warn_conductor_heat_medium_unused)
        self._call_with_validation_loc(
            ("analysis_spec",), self._check_semiconductor_electric_bcs_require_charge_analysis
        )
        self._call_with_validation_loc(("boundary_spec",), self._check_freqs_requires_ac_source)
        self._call_with_validation_loc(
            ("analysis_spec", "at_voltages"), self._check_ssac_specific_voltages
        )
        simulation_types = self._check_simulation_types()
        self._call_with_validation_loc(
            ("monitors",), self._validate_accelerated_only_monitor_requirements
        )
        self._call_with_validation_loc(
            ("monitors",), self._validate_self_heating_monitor_requirements
        )
        if TCADAnalysisTypes.CHARGE in simulation_types:
            self._check_symmetry_permittivity()
            self._call_with_validation_loc(
                ("boundary_spec",), self._check_charge_simulation_voltage_bcs
            )
            self._call_with_validation_loc(("monitors",), self._check_charge_simulation_monitors)
            self._call_with_validation_loc(
                ("structures",), self._check_charge_simulation_semiconductors
            )
            self._call_with_validation_loc(("structures",), self._check_mobility_model_families)
            self._call_with_validation_loc(("structures",), self._check_masetti_mobility_models)
            self._call_with_validation_loc(("structures",), self._check_alloy_mobility_models)
            self._call_with_validation_loc(
                ("structures",), self._check_field_dependence_mobility_models
            )
            self._call_with_validation_loc(
                ("structures",), self._check_surface_dependence_mobility_models
            )
            # Schottky contacts and surface recombination apply only to
            # charge simulations, so validate them inside the charge guard;
            # heat-only and conduction-only simulations skip these checks.
            self._call_with_validation_loc(("boundary_spec",), self._check_schottky_supported_modes)
            self._call_with_validation_loc(("boundary_spec",), self._check_one_bias_per_metal)
            self._call_with_validation_loc(
                ("boundary_spec",), self._check_surface_recombination_bcs
            )
            self._call_with_validation_loc(
                ("structures",), self._warn_non_accelerated_ignores_electron_affinity
            )
            self._call_with_validation_loc(
                ("structures",), self._warn_heterojunction_electron_affinity
            )
            self._call_with_validation_loc(
                ("structures",), self._warn_same_bands_different_narrowing
            )
            self._call_with_validation_loc(("boundary_spec",), self._check_current_bcs)
            # After the CurrentBC checks: a current-driven gate is rejected outright,
            # so demanding its work function first would send the user after a value
            # that cannot make the input valid.
            self._call_with_validation_loc(("boundary_spec",), self._check_gate_voltage_bcs)
            self._check_grad_quasi_fermi_impact_ionization()
        # Ahead of '_not_all_neumann', whose generic message would hide the real mistake when
        # a 'HeatFromElectricSource' has no heat BCs.
        self._call_with_validation_loc(("sources",), self._check_charge_heat_coupling_unsupported)
        # Ahead of the heat-BC checks below, whose generic messages would hide that the
        # input is ignored.
        self._call_with_validation_loc(
            ("boundary_spec",), self._warn_heat_inputs_need_thermal_solve
        )
        self._call_with_validation_loc(("sources",), self._check_coupling_source_can_be_applied)
        self._call_with_validation_loc(("sources",), self._warn_coupling_source_is_no_op)
        self._call_with_validation_loc(("boundary_spec",), self._heat_bcs_reach_solids)
        self._call_with_validation_loc(("boundary_spec",), self._check_self_heating_has_heat_sink)
        self._call_with_validation_loc(("boundary_spec",), self._not_all_neumann)
        self._call_with_validation_loc(("grid_spec",), self._names_exist_grid_spec)
        self._call_with_validation_loc(("grid_spec",), self._warn_if_minimal_mesh_size_override)
        self._call_with_validation_loc(("sources",), self._names_exist_sources)
        self._call_with_validation_loc(("structures",), self._check_medium_specs)
        # A charge analysis solves its temperature inside the charge run and reports through
        # its own monitors, so a 'TemperatureMonitor' is optional there.
        if TCADAnalysisTypes.HEAT in simulation_types and not isinstance(
            self.analysis_spec, ChargeTypes
        ):
            self._call_with_validation_loc(("monitors",), self._check_heat_sim)
        if TCADAnalysisTypes.CONDUCTION in simulation_types:
            self._call_with_validation_loc(("monitors",), self._check_conduction_sim_monitors)
            self._call_with_validation_loc(
                ("boundary_spec",), self._check_conduction_sim_voltage_arrays
            )
            self._call_with_validation_loc(("structures",), self._check_conduction_sim_structures)
            self._call_with_validation_loc(
                ("boundary_spec",), self._check_current_bc_not_on_conduction
            )
            self._check_conduction_sim_permittivity()
        self._call_with_validation_loc(("grid_spec",), self._estimate_charge_mesh_size)
        self._call_with_validation_loc(("linear_solver",), self._warn_direct_linear_solver)
        return self

    def _check_symmetry_not_normal_to_zero_dim(self) -> Self:
        """Reject a symmetry plane normal to the zero-size axis of a 2D simulation."""
        for axis in self.zero_dims:
            if self.symmetry[axis] != 0:
                self._raise_validation_error_at_loc(
                    f"'symmetry' is on along {'xyz'[axis]}, the axis along which the 2D "
                    "simulation has zero size. Set 'symmetry' to 0 along that axis.",
                    "symmetry",
                    axis,
                )
        return self

    def _check_monitor_dimensionality(self) -> Self:
        """Reject monitors whose recorded quantity needs more than a line.

        A monitor also collapses along the simulation's own invariant dimensions, so one
        zero-size dimension in a 2D simulation already records a line, not a plane.
        """
        needs_two_dims = {
            SteadyCapacitanceMonitor: "integrates charge over mesh cells",
            SteadyElectricFieldMonitor: "records a vector field",
            SteadyCurrentDensityMonitor: "records a vector field",
        }
        sim_zero_dims = set(self.zero_dims)

        for idx, mnt in enumerate(self.monitors):
            reason = next(
                (why for mnt_type, why in needs_two_dims.items() if isinstance(mnt, mnt_type)),
                None,
            )
            if reason is None:
                continue

            zero_dims = sim_zero_dims | set(mnt.zero_dims)
            if len(zero_dims) < 2:
                continue

            axes = ", ".join("xyz"[dim] for dim in sorted(zero_dims))
            note = "" if set(mnt.zero_dims) >= zero_dims else " (with the simulation's own)"
            self._raise_validation_error_at_loc(
                f"'{type(mnt).__name__}' (monitor '{mnt.name}') {reason}, so it needs two "
                f"non-zero dimensions, but it has zero size along {axes}{note} and records "
                "only a line or a point. Give it a non-zero size in two dimensions the "
                "simulation resolves.",
                "monitors",
                idx,
            )

        return self

    def validate_pre_upload(self) -> None:
        """Validate the fully initialized heat/charge simulation before web upload."""
        assert_no_protected_media(self, "charge, heat, or heat-charge simulations")
        self._validate_charge_cartesian_monitors()

    def _monitors_cross_solids(self) -> Self:
        """Error if monitors does not cross any solid medium."""
        val = self.monitors

        failed_solid_idx, failed_elect_idx = self._check_cross_solids(val)

        temp_monitors = [idx for idx, mnt in enumerate(val) if isinstance(mnt, TemperatureMonitor)]
        volt_monitors = [
            idx for idx, mnt in enumerate(val) if isinstance(mnt, SteadyPotentialMonitor)
        ]

        failed_temp_mnt = [idx for idx in temp_monitors if idx in failed_solid_idx]
        failed_volt_mnt = [idx for idx in volt_monitors if idx in failed_elect_idx]

        if len(failed_temp_mnt) > 0:
            failed_idx = failed_temp_mnt[0]
            self._raise_validation_error_at_loc(
                "Temperature monitor does not cross any solid materials "
                "('heat_spec=SolidSpec(...)'). Temperature distribution is only recorded inside solid "
                "materials. Thus, no information will be recorded in this monitor.",
                "monitors",
                failed_idx,
            )

        if len(failed_volt_mnt) > 0:
            failed_idx = failed_volt_mnt[0]
            self._raise_validation_error_at_loc(
                "Steady potential monitor does not cross any conducting materials "
                "('charge=ChargeConductorMedium(...)'). The voltage is only stored inside conducting "
                "materials. Thus, no information will be recorded in this monitor.",
                "monitors",
                failed_idx,
            )

        return self

    def _check_voltage_array_if_capacitance(self) -> Self:
        """Make sure an array of voltages has been defined if a
        SteadyCapacitanceMonitor' has been defined"""
        boundary_spec = self.boundary_spec
        monitors = self.monitors

        is_capacitance_mnt = any(isinstance(mnt, SteadyCapacitanceMonitor) for mnt in monitors)
        voltage_array_present = False
        if is_capacitance_mnt:
            for bc in boundary_spec:
                if isinstance(bc.condition, VoltageBC):
                    if isinstance(bc.condition.source, DCVoltageSource):
                        if len(bc.condition.source.voltage) > 1:
                            voltage_array_present = True
                    elif isinstance(bc.condition.source, SSACVoltageSource):
                        if len(bc.condition.source.voltage) > 1:
                            voltage_array_present = True
        if is_capacitance_mnt and not voltage_array_present:
            raise SetupError(
                "Monitors of type 'SteadyCapacitanceMonitor' have been defined but no array of voltages "
                "has been supplied as voltage source, which is required for this type of monitor. "
                "Voltage arrays can be included in a source in this manner: "
                "'VoltageBC(source=DCVoltageSource(voltage=yourArray))'"
            )
        return self

    def _check_single_ssac(self) -> Self:
        """Error if more than one terminal carries an ``SSACVoltageSource``."""
        terminals = self._voltage_terminals(self._ssac_bc_indices())
        if len(terminals) > 1:
            self._raise_validation_error_at_loc(
                "Only a single 'SSACVoltageSource' source can be supplied. Entries naming one "
                "metal structure with the same source count as one.",
                "boundary_spec",
                terminals[1],
                "condition",
            )
        return self

    def _check_natural_convection_bc(self) -> Self:
        """Make sure that natural convection BCs are defined correctly."""
        boundary_spec = self.boundary_spec
        if not boundary_spec:
            return self

        structures = self.structures
        boundary_spec = self.boundary_spec
        bg_medium = self.medium

        # Create mappings for easy lookup of media and structures by name.
        media = {s.medium.name: s.medium for s in structures if s.medium.name}
        if bg_medium and bg_medium.name:
            media[bg_medium.name] = bg_medium
        structures_map = {s.name: s for s in structures if s.name}

        def check_fluid_medium_attr(fluid_medium: FluidMedium) -> None:
            if (
                (fluid_medium.thermal_conductivity is None)
                or (fluid_medium.viscosity is None)
                or (fluid_medium.specific_heat is None)
                or (fluid_medium.density is None)
                or (fluid_medium.expansivity is None)
            ):
                raise SetupError(
                    f"Boundary spec at index {i}: The fluid medium at the natural convection interface "
                    f"must have 'thermal_conductivity', 'viscosity', 'specific_heat', 'density' and 'expansivity' defined."
                )

        for i, bc in enumerate(boundary_spec):
            if not (
                isinstance(bc.condition, ConvectionBC)
                and isinstance(bc.condition.transfer_coeff, VerticalNaturalConvectionCoeffModel)
            ):
                continue

            natural_conv_model = bc.condition.transfer_coeff
            placement = bc.placement

            # Case 1: The fluid medium is inferred from the placement interface.
            # We use direct dictionary access, assuming '_names_exist_bcs' validator has already run.
            if natural_conv_model.medium is None:
                if isinstance(placement, MediumMediumInterface):
                    med1 = media[placement.mediums[0]]
                    med2 = media[placement.mediums[1]]
                elif isinstance(placement, StructureStructureInterface):
                    med1 = structures_map[placement.structures[0]].medium
                    med2 = structures_map[placement.structures[1]].medium
                else:
                    raise SetupError(
                        f"Boundary spec at index {i}: 'VerticalNaturalConvectionCoeffModel' with no medium specified requires "
                        f"the 'placement' to be of type 'MediumMediumInterface' or 'StructureStructureInterface', "
                        f"but got '{type(placement).__name__}'."
                    )
                specs = [
                    med1.heat if isinstance(med1, MultiPhysicsMedium) else med1,
                    med2.heat if isinstance(med2, MultiPhysicsMedium) else med2,
                ]

                # Check for a single fluid in the interface.
                is_fluid = [isinstance(s, FluidMedium) for s in specs]
                if is_fluid.count(True) != 1:
                    raise SetupError(
                        f"Boundary spec at index {i}: A natural convection boundary at an interface "
                        f"must be between exactly one solid and one fluid medium. "
                        f"Found types '{type(specs[0]).__name__}' and '{type(specs[1]).__name__}'."
                    )
                fluid_medium = specs[is_fluid.index(True)]
                check_fluid_medium_attr(fluid_medium)

            # Case 2: The fluid medium IS specified directly in the convection model.
            else:
                check_fluid_medium_attr(natural_conv_model.medium)
        return self

    def _check_thermal_contact_resistance_placement(self) -> Self:
        """Make sure 'ThermalContactResistance' conditions are placed on an interface
        between two solid heat regions."""
        # name -> medium / structure lookups (assumes '_names_exist_bcs' has already run)
        media = {s.medium.name: s.medium for s in self.structures if s.medium.name}
        if self.medium and self.medium.name:
            media[self.medium.name] = self.medium
        structures_map = {s.name: s for s in self.structures if s.name}

        for i, bc in enumerate(self.boundary_spec):
            if not isinstance(bc.condition, ThermalContactResistance):
                continue
            placement = bc.placement
            if not isinstance(placement, (MediumMediumInterface, StructureStructureInterface)):
                self._raise_validation_error_at_loc(
                    "'ThermalContactResistance' represents an interfacial thermal resistance "
                    "between two touching solids, so its 'placement' must be a "
                    "'MediumMediumInterface' or a 'StructureStructureInterface', "
                    f"but got '{type(placement).__name__}'.",
                    "boundary_spec",
                    i,
                    "placement",
                )

            # Both sides must take part in the heat solve as solids; otherwise the interface
            # is physically meaningless and would only fail later, at mesh time, with an
            # opaque backend error.
            if isinstance(placement, MediumMediumInterface):
                side_media = [media.get(name) for name in placement.mediums]
            else:
                side_media = [
                    structures_map[name].medium if name in structures_map else None
                    for name in placement.structures
                ]
            for medium in side_media:
                if medium is not None and not isinstance(medium.heat_spec, SolidMedium):
                    self._raise_validation_error_at_loc(
                        "'ThermalContactResistance' can only be placed on an interface "
                        "between two solid materials: each side must define a solid heat "
                        f"specification ('SolidSpec'), but medium '{medium.name}' does not.",
                        "boundary_spec",
                        i,
                        "placement",
                    )
        return self

    @field_validator("size")
    @classmethod
    def _check_zero_dim_domain(cls, val: Any) -> Any:
        """Error if heat domain have zero dimensions."""

        dim_names = ["x", "y", "z"]
        zero_dimensions = [False, False, False]
        zero_dim_str = ""
        for n, v in enumerate(val):
            if v == 0:
                zero_dimensions[n] = True
                zero_dim_str += f"{dim_names[n]}- "

        num_zero_dims = np.sum(zero_dimensions)

        if num_zero_dims > 1:
            mssg = f"The current 'HeatChargeSimulation' has zero size along the {zero_dim_str}dimensions. "
            mssg += "Only 2- and 3-D simulations are currently supported."
            raise SetupError(mssg)

        return val

    def _names_exist_bcs(self) -> Self:
        """Error if boundary conditions point to non-existing structures/media."""
        structures = self.structures
        structures_names = {s.name for s in structures}
        mediums_names = {s.medium.name for s in structures}
        mediums_names.add(self.medium.name)

        for bc_ind, bc_spec in enumerate(self.boundary_spec):
            bc_place = bc_spec.placement
            if isinstance(bc_place, (StructureBoundary, StructureSimulationBoundary)):
                if bc_place.structure not in structures_names:
                    raise SetupError(
                        f"Structure '{bc_place.structure}' provided in "
                        f"'boundary_spec[{bc_ind}].placement' (type '{bc_place.type}') "
                        "is not found among simulation structures."
                    )
            if isinstance(bc_place, (StructureStructureInterface)):
                for struct_name in bc_place.structures:
                    if struct_name and struct_name not in structures_names:
                        raise SetupError(
                            f"Structure '{struct_name}' provided in "
                            f"'boundary_spec[{bc_ind}].placement' (type '{bc_place.type}') "
                            "is not found among simulation structures."
                        )
            if isinstance(bc_place, (MediumMediumInterface)):
                for med_name in bc_place.mediums:
                    if med_name not in mediums_names:
                        raise SetupError(
                            f"Material '{med_name}' provided in "
                            f"'boundary_spec[{bc_ind}].placement' (type '{bc_place.type}') "
                            "is not found among simulation mediums."
                        )
        return self

    def _check_only_one_voltage_array_provided(self) -> Self:
        """Error if more than one terminal is swept."""
        terminals = self._voltage_terminals(self._swept_bc_indices())
        if len(terminals) > 1:
            self._raise_validation_error_at_loc(
                "More than one voltage or current array has been provided. Currently such "
                "arrays are supported only for one terminal; entries naming one metal "
                "structure with the same source count as one.",
                "boundary_spec",
                terminals[1],
                "condition",
            )
        return self

    def _ssac_bc_indices(self) -> list[int]:
        """Indices of the ``VoltageBC`` entries with an ``SSACVoltageSource``."""
        return [
            i
            for i, bc in enumerate(self.boundary_spec)
            if isinstance(bc.condition, VoltageBC)
            and isinstance(bc.condition.source, SSACVoltageSource)
        ]

    def _swept_bc_indices(self) -> list[int]:
        """Indices of the contact entries that sweep more than one voltage or current.

        Both DC and SSAC sources carry a DC sweep array, and a ``CurrentBC`` sweeps
        its current; counting only one type would admit an ambiguous two-sweep setup.
        """
        return [
            i
            for i, bc in enumerate(self.boundary_spec)
            if (
                isinstance(bc.condition, VoltageBC)
                and isinstance(bc.condition.source, (DCVoltageSource, SSACVoltageSource))
                and len(bc.condition.source.voltage) > 1
            )
            or self._is_current_sweep(i)
        ]

    def _is_current_sweep(self, index: int) -> bool:
        """Whether ``boundary_spec[index]`` is a ``CurrentBC`` with more than one current."""
        condition = self.boundary_spec[index].condition
        return isinstance(condition, CurrentBC) and np.size(condition.source.current) > 1

    def _driven_terminal_split_across_entries(self) -> bool:
        """Whether a swept or SSAC terminal is named by more than one ``boundary_spec`` entry.

        The legacy solver drives or measures only one of those entries.
        """
        return any(
            len(self._voltage_terminals(indices)) < len(indices)
            for indices in (self._swept_bc_indices(), self._ssac_bc_indices())
        )

    def _voltage_terminals(self, indices: list[int]) -> list[int]:
        """First ``boundary_spec`` index of each terminal among the given contact entries.

        Entries that name a common ``ChargeConductorMedium`` structure and carry the same
        drive are one terminal, as in :meth:`_check_one_bias_per_metal`; any other entry
        is a terminal of its own.
        """
        terminals: list[tuple[int, set[str], Any]] = []
        for i in indices:
            conductors = self._contact_conductors(i)
            drive = self._drive_of(self.boundary_spec[i].condition.source)
            joined = [t for t in terminals if t[2] == drive and t[1] & conductors]
            for terminal in joined:
                terminals.remove(terminal)
                conductors |= terminal[1]
            first = min([i] + [terminal[0] for terminal in joined])
            terminals.append((first, conductors, drive))
        return sorted(terminal[0] for terminal in terminals)

    def _check_freqs_requires_ac_source(self) -> Self:
        """Ensure that if freqs is provided, at least one ACVoltageSource is present."""
        analysis_spec = self.analysis_spec
        if (
            isinstance(analysis_spec, (SSACAnalysis, IsothermalSSACAnalysis))
            and len(analysis_spec.freqs) > 0
        ):
            bcs = self.boundary_spec
            has_ac_source = False
            for bc in bcs:
                if isinstance(bc.condition, VoltageBC):
                    if isinstance(bc.condition.source, SSACVoltageSource):
                        has_ac_source = True
                        break

            if not has_ac_source:
                raise SetupError(
                    "If 'freqs' is provided and not empty, at least one "
                    "'SSACVoltageSource' must be present in the boundary conditions."
                )

        return self

    def _check_ssac_specific_voltages(self) -> Self:
        """Validate user-selected SSAC bias points against the DC voltage sweep."""
        analysis_spec = self.analysis_spec
        if not isinstance(analysis_spec, (SSACAnalysis, IsothermalSSACAnalysis)):
            return self
        if analysis_spec.at_voltages is None:
            return self

        dc_voltages = np.asarray(self._dc_voltages, dtype=float)
        if dc_voltages.size < 2:
            # No sweep: ``_dc_voltages`` is just the first scalar source, so
            # validate against the SSAC operating point (the AC drive's DC bias).
            ssac_voltages = [
                float(v)
                for bc in self.boundary_spec
                if isinstance(bc.condition, VoltageBC)
                and isinstance(bc.condition.source, SSACVoltageSource)
                for v in bc.condition.source.voltage
            ]
            dc_voltages = np.asarray(ssac_voltages, dtype=float)
        if dc_voltages.size == 0:
            return self

        missing_voltages = [
            voltage
            for voltage in analysis_spec.at_voltages
            if not np.any(np.isclose(dc_voltages, voltage, rtol=0.0, atol=SSAC_VOLTAGE_MATCH_TOL_V))
        ]
        if missing_voltages:
            raise SetupError(
                "Every entry in 'at_voltages' must be present in the DC voltage sweep "
                "(with no multi-voltage sweep, the SSAC source bias). "
                f"Missing voltages: {missing_voltages}."
            )

        return self

    def _check_charge_simulation_voltage_bcs(self) -> Self:
        """Require two charge contacts and at least one prescribed potential."""
        voltage_idx = [
            i for i, bc in enumerate(self.boundary_spec) if isinstance(bc.condition, VoltageBC)
        ]
        current_idx = self._current_bc_indices()
        n_entries = len(voltage_idx) + len(current_idx)
        if n_entries < 2:
            raise SetupError(
                "Defining a Charge simulation requires at least two contact boundaries. "
                "Use 'VoltageBC' (Schottky contacts opt in via model=\"schottky_mott\") "
                "or 'CurrentBC'. So far "
                f"{n_entries} contact boundaries have been set."
            )
        if not voltage_idx:
            raise SetupError(
                "Defining a Charge simulation requires at least one 'VoltageBC' to fix the "
                "potential reference. A 'CurrentBC' prescribes a current, not a potential, "
                "so a device driven only by currents leaves the electric potential "
                "undetermined."
            )
        terminals = self._voltage_terminals(voltage_idx) + self._voltage_terminals(current_idx)
        if len(terminals) >= 2:
            return self
        driven = set(self._swept_bc_indices()) | set(self._ssac_bc_indices())
        if driven & set(voltage_idx):
            self._raise_validation_error_at_loc(
                "Defining a Charge simulation requires at least two contact terminals, but "
                "the 'boundary_spec' entries form a single terminal that is swept or carries "
                "an 'SSACVoltageSource'. Entries naming one metal structure count as one "
                "terminal. Add a second terminal.",
                "boundary_spec",
                voltage_idx[0],
                "condition",
            )
        log.warning(
            f"The {n_entries} contact entries in 'boundary_spec' form a single terminal, "
            "because they name one metal structure with one source. A charge simulation "
            "needs two terminals to carry current. Future versions of Tidy3D may reject this "
            "setup."
        )
        return self

    def _check_charge_simulation_monitors(self) -> Self:
        """Validate Charge simulation has at least one charge monitor."""
        if not any(isinstance(mnt, ChargeMonitorTypes) for mnt in self.monitors):
            raise SetupError(
                "Charge simulations require the definition of, at least, one of these monitors: "
                "'[SteadyPotentialMonitor, SteadyFreeCarrierMonitor, SteadyCapacitanceMonitor, SteadyCurrentDensityMonitor, SteadyChargeResidualMonitor, SteadyGenerationRecombinationMonitor, SteadyDopingMonitor, SelfHeatingMonitor]' "
                "but none have been defined."
            )
        # NOTE: in Charge we're only supporting unstructured monitors.
        # only Temperature and Potential monitors can be structured.
        for mnt in self.monitors:
            if isinstance(mnt, SteadyPotentialMonitor) or isinstance(mnt, TemperatureMonitor):
                if not mnt.unstructured:
                    log.warning(
                        "Currently, Charge simulations support only unstructured monitors. Please set "
                        f"monitor '{mnt.name}' to 'unstructured = True'."
                    )
        return self

    def _validate_charge_cartesian_monitors(self) -> None:
        """Reject a Cartesian monitor that a Charge simulation cannot record.

        Charge simulations record Cartesian data only for a ``SteadyPotentialMonitor`` in a 3D
        simulation at a single bias point.
        """
        if TCADAnalysisTypes.CHARGE not in self._get_simulation_types():
            return

        for idx, mnt in enumerate(self.monitors):
            if not isinstance(mnt, (SteadyPotentialMonitor, TemperatureMonitor)):
                continue
            if mnt.unstructured:
                continue
            if isinstance(mnt, TemperatureMonitor):
                reason = "Charge simulations record temperature only on the unstructured grid"
            elif self.zero_dims:
                reason = "2D Charge simulations record no Cartesian data"
            elif len(self._dc_voltages) > 1 or len(self._dc_currents) > 1:
                reason = "Cartesian data covers one bias point and this simulation sweeps the bias"
            else:
                continue
            self._raise_validation_error_at_loc(
                f"Monitor '{mnt.name}' of type '{type(mnt).__name__}' sets "
                f"'unstructured=False', which is not supported here: {reason}. Set "
                "'unstructured=True' and, where a Cartesian grid is needed, resample the recorded "
                "field with 'to_spatial_data_array()' on the monitor data, passing 'voltage' to "
                "pick a bias point.",
                "monitors",
                idx,
            )

    def _validate_accelerated_only_monitor_requirements(self) -> Self:
        """Validate monitors that exist only on the accelerated charge solver.

        ``SteadyChargeResidualMonitor``, ``SteadyGenerationRecombinationMonitor`` and
        ``SteadyDopingMonitor`` require a configured charge analysis and the accelerated
        solver."""
        accelerated_only = (
            SteadyChargeResidualMonitor,
            SteadyGenerationRecombinationMonitor,
            SteadyDopingMonitor,
        )
        simulation_types = self._check_simulation_types()
        charge_configured = TCADAnalysisTypes.CHARGE in simulation_types
        for idx, mnt in enumerate(self.monitors):
            if not isinstance(mnt, accelerated_only):
                continue
            mnt_type = type(mnt).__name__
            if not charge_configured:
                self._raise_validation_error_at_loc(
                    f"'{mnt_type}' is only available when a charge analysis "
                    "is configured (the simulation must include voltage BCs and a "
                    "'SteadyChargeDCAnalysis' or derivative analysis spec).",
                    "monitors",
                    idx,
                )
            if self.use_accelerated_solver is False:
                self._raise_validation_error_at_loc(
                    f"'{mnt_type}' (monitor '{mnt.name}') is only available "
                    "through the accelerated charge solver, but 'use_accelerated_solver=False' "
                    "was set. Remove the monitor or use the accelerated solver (the default).",
                    "monitors",
                    idx,
                )
        return self

    def _check_charge_simulation_semiconductors(self) -> Self:
        """Validate Charge simulation has at least one semiconductor medium."""
        if not any(self._iter_semiconductor_charge_media()):
            raise SetupError(
                f"{TCADAnalysisTypes.CHARGE} simulations require the definition of at least one semiconductor medium."
            )
        return self

    @staticmethod
    def _bc_is_schottky(bc_spec: Any) -> bool:
        """``True`` when this boundary requests Schottky physics."""
        condition = bc_spec.condition
        return isinstance(condition, (VoltageBC, CurrentBC)) and condition.model == "schottky_mott"

    def _check_schottky_supported_modes(self) -> Self:
        """Reject Schottky configurations outside the validated solver model."""
        has_schottky = any(self._bc_is_schottky(bc) for bc in self.boundary_spec)
        if not has_schottky:
            return self

        if self.use_accelerated_solver is False:
            self._raise_validation_error_at_loc(
                'Schottky contacts (model="schottky_mott") are '
                "implemented only by the accelerated charge solver, but "
                "'use_accelerated_solver=False' selects the legacy solver. "
                'Either set model="ohmic" or set '
                "'use_accelerated_solver=True'.",
                "use_accelerated_solver",
                log_error=False,
            )
        for index, bc_spec in enumerate(self.boundary_spec):
            if not self._bc_is_schottky(bc_spec):
                continue
            if isinstance(bc_spec.placement, (StructureSimulationBoundary, SimulationBoundary)):
                self._raise_validation_error_at_loc(
                    'Schottky contacts (model="schottky_mott") '
                    f"cannot be placed on '{type(bc_spec.placement).__name__}': the "
                    "metal-semiconductor contact cannot be identified there. Place "
                    "the condition on the metal structure's 'StructureBoundary' or "
                    "on a 'StructureStructureInterface' between the metal and "
                    "semiconductor structures.",
                    "boundary_spec",
                    index,
                    log_error=False,
                )
        return self

    def _gate_bc_indices(self) -> list[int]:
        """Indices of the ohmic contacts whose terminal reaches only insulators.

        A gate reaches an insulator and no semiconductor; its metal work
        function sets the potential reference. Conductor/conductor and
        metal/fluid interfaces do not qualify.

        Gate identity follows contacted materials, independent of the drive.

        Judged from :meth:`_placement_contact_media`, so a
        ``StructureSimulationBoundary`` counts while ``StructureBoundary`` and
        ``SimulationBoundary`` stay unresolved until the simulation runs. An
        unresolved contact on a candidate metal defers that metal; one naming no
        metal at all could touch any of them and defers the whole simulation.
        """
        placements = []
        metals_touching_semiconductor = set()
        metals_with_unknown_contacts = set()
        has_unidentified_contact = False
        for i, bc in enumerate(self.boundary_spec):
            condition = bc.condition
            if not isinstance(condition, (VoltageBC, CurrentBC)):
                continue
            sides = self._placement_contact_media(bc.placement)
            conductor_names = self._placement_conductor_structures(bc.placement)
            if sides is None:
                metals_with_unknown_contacts.update(conductor_names)
                if not conductor_names:
                    has_unidentified_contact = True
                continue
            charge_specs = [self._charge_spec(side) for side in sides]
            if any(isinstance(spec, SemiconductorMedium) for spec in charge_specs):
                metals_touching_semiconductor.update(conductor_names)
            placements.append((i, condition, charge_specs, conductor_names))

        if has_unidentified_contact:
            return []

        indices = []
        for i, condition, charge_specs, conductor_names in placements:
            if condition.model != "ohmic":
                continue
            if any(isinstance(spec, SemiconductorMedium) for spec in charge_specs):
                continue
            if not any(isinstance(spec, ChargeInsulatorMedium) for spec in charge_specs):
                continue
            if conductor_names and any(
                name in metals_touching_semiconductor or name in metals_with_unknown_contacts
                for name in conductor_names
            ):
                continue
            indices.append(i)
        return indices

    def _insulator_face_bc_indices(self) -> list[int]:
        """Indices of ``VoltageBC`` entries with a face lying on an insulator.

        Such a face has no semiconductor of its own to take its potential
        reference from; only the accelerated charge solver supports it, so the
        legacy solver is refused rather than run with a shifted flat-band voltage.

        Wider than :meth:`_gate_bc_indices`: an oxide-clad ohmic contact is not a
        gate and still has an insulator face. Judged only where
        :meth:`_placement_contact_media` determines the materials.
        """
        indices = []
        for i, bc in enumerate(self.boundary_spec):
            if not isinstance(bc.condition, VoltageBC):
                continue
            sides = self._placement_contact_media(bc.placement)
            if sides is None:
                continue
            if any(isinstance(self._charge_spec(side), ChargeInsulatorMedium) for side in sides):
                indices.append(i)
        return indices

    def _check_gate_voltage_bcs(self) -> Self:
        """Require the metal work function, and the electron affinity, once a gate is present.

        A gate's potential is referenced to the metal work function, which a
        contact touching the semiconductor never has to state. The electron
        affinity cancels out of every potential difference while all contacts
        touch the semiconductor, but against a gate it sets where the bands sit,
        and leaving it unset shifts the flat-band voltage by whole volts.

        The same references apply to voltage- and current-driven gates.
        """
        gate_indices = self._gate_bc_indices()
        if not gate_indices:
            return self

        for i in gate_indices:
            condition_name = type(self.boundary_spec[i].condition).__name__
            charge_specs = [
                self._charge_spec(side)
                for side in self._placement_contact_media(self.boundary_spec[i].placement)
            ]
            has_metal_w = any(
                isinstance(spec, ChargeConductorMedium) and spec.work_function is not None
                for spec in charge_specs
            )
            if not has_metal_w:
                self._raise_validation_error_at_loc(
                    f"'{condition_name}' is placed on an interface where neither side is a "
                    "'SemiconductorMedium', which makes it a gate on an insulator. A gate's "
                    "potential is referenced to the metal work function, so one side must be "
                    "a 'ChargeConductorMedium' carrying 'work_function'. Without it the gate "
                    "potential is off by the metal work function minus the semiconductor "
                    "electron affinity and the flat-band voltage is wrong.",
                    "boundary_spec",
                    i,
                    "placement",
                    log_error=False,
                )

        gate_conditions = sorted(
            {type(self.boundary_spec[i].condition).__name__ for i in gate_indices}
        )
        gate_names = " or ".join(f"'{name}'" for name in gate_conditions)
        for loc, charge in self._iter_semiconductor_charge_media():
            if charge.electron_affinity is not None:
                continue
            self._raise_validation_error_at_loc(
                f"This simulation places a {gate_names} on an insulator (a gate), whose "
                "potential is referenced to the vacuum level. Every 'SemiconductorMedium' must "
                "then carry 'electron_affinity', or its bands sit at an arbitrary offset from "
                "the gate and the flat-band voltage is wrong by several volts.",
                *loc,
                log_error=False,
            )
        return self

    def _check_one_bias_per_metal(self) -> Self:
        """Reject a ``ChargeConductorMedium`` structure driven by two disagreeing sources.

        A metal is equipotential, so the several ``boundary_spec`` entries that
        may name one metal structure — its semiconductor contact and its oxide
        cladding, say — declare one terminal and must agree on its bias.
        Medium-based placements need geometric adjacency to identify the
        contacted structures, so their consistency is checked when running.
        Sources are compared through :meth:`_drive_of`, so differing labels
        alone are not a disagreement. Terminal identity comes from
        :meth:`_contact_conductors`, the same grouping the ``CurrentBC``
        checks use.
        """
        source_by_structure: dict[str, tuple[int, Any]] = {}
        for i, bc in enumerate(self.boundary_spec):
            if not isinstance(bc.condition, VoltageBC):
                continue
            source = self._drive_of(bc.condition.source)
            for name in self._contact_conductors(i):
                first = source_by_structure.setdefault(name, (i, source))
                if first[1] == source:
                    continue
                self._raise_validation_error_at_loc(
                    f"Structure '{name}' is a 'ChargeConductorMedium' named by two 'VoltageBC' "
                    f"entries with different sources ('boundary_spec[{first[0]}]' and "
                    f"'boundary_spec[{i}]'). A metal structure is one terminal and a metal is "
                    "equipotential, so its contacts must share one source. Give both entries "
                    "the same source; or, if this structure is several electrically separate "
                    "pieces of metal -- disjoint islands of one 'GeometryGroup', say -- give "
                    "each piece its own 'Structure', since one structure cannot carry two "
                    "potentials.",
                    "boundary_spec",
                    i,
                    "condition",
                    log_error=False,
                )
        return self

    def _placement_conductor_structures(self, placement: BCPlacementType) -> list[str]:
        """Names of the ``ChargeConductorMedium`` structures a placement names.

        ``MediumMediumInterface`` names media rather than structures, so every
        structure carrying a named medium counts.
        """
        if isinstance(placement, (StructureBoundary, StructureSimulationBoundary)):
            named = {placement.structure}
        elif isinstance(placement, StructureStructureInterface):
            named = set(placement.structures)
        elif isinstance(placement, MediumMediumInterface):
            named = {
                structure.name
                for structure in self.structures
                if structure.medium.name in placement.mediums
            }
        else:
            return []
        return [
            structure.name
            for structure in self.structures
            if structure.name in named
            and isinstance(self._charge_spec(structure.medium), ChargeConductorMedium)
        ]

    def _check_surface_recombination_bcs(self) -> Self:
        """Run all ``SurfaceRecombinationBC`` setup checks."""
        self._check_surface_recombination_requires_accelerated()
        self._check_surface_recombination_not_stacked_with_insulating_bc()
        self._check_surface_recombination_not_stacked_with_schottky()
        self._check_surface_recombination_qf_on_voltage_overlay()
        self._check_surface_recombination_tat_on_voltage_overlay()
        self._check_surface_recombination_heterojunction_not_on_contact()
        self._check_surface_recombination_no_duplicate_placement()
        self._check_surface_recombination_interface_materials()
        return self

    @staticmethod
    def _charge_spec(medium: Any) -> Any:
        """The charge medium of ``medium``, reading through a ``MultiPhysicsMedium``.

        A single-physics medium is its own charge spec; media with no charge
        behaviour (a fluid, say) pass through unchanged and match none of the
        charge medium classes.
        """
        return medium.charge if isinstance(medium, MultiPhysicsMedium) else medium

    @staticmethod
    def _drive_of(source: Any) -> Any:
        """The drive ``source`` prescribes, stripped of its name and attributes.

        ``GroundVoltage`` is documented as equivalent to
        ``DCVoltageSource(voltage=0)`` and compares as that source.
        """
        if isinstance(source, GroundVoltage):
            source = DCVoltageSource(voltage=0)
        if isinstance(source, DCCurrentSource) and np.size(source.current) == 1:
            # A one-element array drives the same current as its scalar.
            source = source.updated_copy(current=float(np.ravel(source.current)[0]))
        metadata = {"name": None, "attrs": {}}
        updates = {
            key: value for key, value in metadata.items() if key in type(source).model_fields
        }
        return source.updated_copy(**updates) if updates else source

    def _placement_side_media(self, placement: BCPlacementType) -> list | None:
        """The two media a placement names, or ``None`` if it names only one side.

        A ``StructureBoundary`` or simulation-boundary placement names one side
        only; the opposing material follows from the geometry and is resolved
        when the simulation runs. Assumes ``_names_exist_bcs`` has already run.
        """
        media = {s.medium.name: s.medium for s in self.structures if s.medium.name}
        if self.medium and self.medium.name:
            media[self.medium.name] = self.medium
        structures_map = {s.name: s for s in self.structures if s.name}

        if isinstance(placement, MediumMediumInterface):
            sides = [media.get(name) for name in placement.mediums]
        elif isinstance(placement, StructureStructureInterface):
            # An unnamed side is the background medium the structure sits in.
            sides = [
                structures_map[name].medium if name in structures_map else self.medium
                for name in placement.structures
            ]
        else:
            return None
        return None if any(side is None for side in sides) else sides

    def _placement_contact_media(self, placement: BCPlacementType) -> list | None:
        """Every medium a contact on ``placement`` is certainly made against.

        Wider than :meth:`_placement_side_media`, which answers only for
        placements naming *two* sides: a ``StructureSimulationBoundary`` names one
        structure and the simulation box, so its contact materials are fully
        determined by that one medium.

        ``None`` where a side could be any material the geometry puts there
        (``StructureBoundary``, ``SimulationBoundary``); resolving those needs
        structure adjacency, which is computed when the simulation runs.
        """
        if isinstance(placement, StructureSimulationBoundary):
            structures_map = {s.name: s for s in self.structures if s.name}
            named = structures_map.get(placement.structure)
            medium = named.medium if named is not None else self.medium
            return None if medium is None else [medium]
        return self._placement_side_media(placement)

    def _check_surface_recombination_interface_materials(self) -> Self:
        """Reject ``SurfaceRecombinationBC`` on an interface that cannot carry it.

        Only placements naming both sides are checked; for the rest the opposing
        material follows from the geometry and is resolved when the simulation
        runs. A heterojunction model must name both sides, so it is always checked.
        """
        for i, bc in enumerate(self.boundary_spec):
            if not isinstance(bc.condition, SurfaceRecombinationBC):
                continue
            heterojunction_model = isinstance(
                bc.condition.model, HeterojunctionSurfaceShockleyReedHallRecombination
            )
            if heterojunction_model:
                self._check_heterojunction_sr_placement(i, bc.condition.model, bc.placement)
            sides = self._placement_side_media(bc.placement)
            if sides is None:
                continue

            semiconductors = [
                spec
                for spec in (self._charge_spec(side) for side in sides)
                if isinstance(spec, SemiconductorMedium)
            ]

            if not semiconductors:
                self._raise_validation_error_at_loc(
                    "'SurfaceRecombinationBC' is placed on an interface where neither "
                    "side is a 'SemiconductorMedium'. Surface recombination consumes "
                    "electrons and holes, so it needs a semiconductor on at least one "
                    "side of the interface.",
                    "boundary_spec",
                    i,
                    "placement",
                )
            at_heterojunction = len(semiconductors) == 2 and _is_material_heterojunction(
                *semiconductors
            )
            if heterojunction_model and not at_heterojunction:
                self._raise_validation_error_at_loc(
                    "'HeterojunctionSurfaceShockleyReedHallRecombination' is placed on an "
                    "interface that is not a heterojunction: the two 'SemiconductorMedium' "
                    "materials must differ in 'N_c', 'N_v', 'E_g' or 'electron_affinity'. Use "
                    "'SurfaceShockleyReedHallRecombination' there: it covers free surfaces, "
                    "semiconductor/insulator interfaces and junctions between regions of "
                    "the same material.",
                    "boundary_spec",
                    i,
                    "placement",
                )
            if not heterojunction_model and at_heterojunction:
                self._raise_validation_error_at_loc(
                    "'SurfaceShockleyReedHallRecombination' is placed on a heterojunction, "
                    "where the interface trap exchanges carriers with both semiconductors. "
                    "Use 'HeterojunctionSurfaceShockleyReedHallRecombination' on this "
                    "placement.",
                    "boundary_spec",
                    i,
                    "placement",
                )
        return self

    def _check_current_bcs(self) -> Self:
        """Run ``CurrentBC`` setup checks."""
        self._check_current_bc_requires_accelerated()
        self._check_one_current_per_metal()
        self._check_current_bc_not_stacked_with_voltage_bc()
        self._check_current_bc_not_on_gate()
        self._check_current_sweep_not_with_ssac_bias_selection()
        return self

    def _check_current_bc_not_on_gate(self) -> Self:
        """Reject ``CurrentBC`` on a contact that resolves to a gate.

        A gate reaches only an insulator, so its terminal current is zero at
        every contact potential and an imposed current, zero included, leaves
        that potential undetermined. Drive the gate with ``VoltageBC`` instead.
        """
        for i in self._gate_bc_indices():
            if not isinstance(self.boundary_spec[i].condition, CurrentBC):
                continue
            self._raise_validation_error_at_loc(
                "'CurrentBC' is placed on a contact that reaches only an insulator and no "
                "semiconductor, which makes it a gate. An imposed terminal current is not "
                "supported on a gate, including a current of exactly zero. Drive this "
                "contact with 'VoltageBC' instead.",
                "boundary_spec",
                i,
                log_error=False,
            )
        return self

    def _current_bc_indices(self) -> list[int]:
        return [
            idx for idx, bc in enumerate(self.boundary_spec) if isinstance(bc.condition, CurrentBC)
        ]

    def _contact_conductors(self, idx: int) -> set[str]:
        """Return named conductors; empty means geometry must identify the terminal."""
        placement = self.boundary_spec[idx].placement
        if isinstance(placement, MediumMediumInterface):
            return set()
        return set(self._placement_conductor_structures(placement))

    def _identified_current_bc_conductors(self) -> list[tuple[int, set[str]]]:
        """Return ``CurrentBC`` entries with identifiable terminals."""
        return [
            (idx, conductors)
            for idx, conductors in (
                (idx, self._contact_conductors(idx)) for idx in self._current_bc_indices()
            )
            if conductors
        ]

    def _check_one_current_per_metal(self) -> Self:
        """Require repeated ``CurrentBC`` entries naming one metal to agree on the current."""
        entries = self._identified_current_bc_conductors()
        for position, (idx, conductors) in enumerate(entries):
            for other_idx, other_conductors in entries[:position]:
                shared = conductors & other_conductors
                if not shared:
                    continue
                if self._drive_of(self.boundary_spec[idx].condition.source) != self._drive_of(
                    self.boundary_spec[other_idx].condition.source
                ):
                    self._raise_validation_error_at_loc(
                        f"Structure '{sorted(shared)[0]}' is a 'ChargeConductorMedium' named "
                        f"by two 'CurrentBC' entries with different sources "
                        f"('boundary_spec[{other_idx}]' and 'boundary_spec[{idx}]'). A metal "
                        "structure is one terminal, so the current through it is one number; "
                        "give both entries the same source.",
                        "boundary_spec",
                        idx,
                        "condition",
                        log_error=False,
                    )
        return self

    def _check_current_bc_not_stacked_with_voltage_bc(self) -> Self:
        """Reject prescribing current and potential on any face of one terminal."""
        for idx in self._current_bc_indices():
            current_bc = self.boundary_spec[idx]
            conductors = self._contact_conductors(idx)
            for other_idx, other_bc in enumerate(self.boundary_spec):
                if other_idx == idx or not isinstance(other_bc.condition, VoltageBC):
                    continue
                if self._placements_overlap(current_bc.placement, other_bc.placement):
                    self._raise_validation_error_at_loc(
                        "'CurrentBC' cannot share a placement with 'VoltageBC'. A "
                        "current-driven contact has its potential solved for, so it "
                        "cannot also be held at a prescribed one.",
                        "boundary_spec",
                        idx,
                    )
                shared = conductors & self._contact_conductors(other_idx)
                if shared:
                    self._raise_validation_error_at_loc(
                        f"Structure '{sorted(shared)[0]}' is a 'ChargeConductorMedium' named "
                        f"by both a 'CurrentBC' ('boundary_spec[{idx}]') and a 'VoltageBC' "
                        f"('boundary_spec[{other_idx}]'). A metal structure is one terminal, "
                        "so the two entries drive one contact, whose potential cannot be both "
                        "solved for and prescribed. Use one or the other.",
                        "boundary_spec",
                        idx,
                    )
        return self

    def _check_current_bc_requires_accelerated(self) -> Self:
        """Reject ``CurrentBC`` when the accelerated charge solver is off."""
        current_bcs = self._current_bc_indices()
        if not current_bcs:
            return self
        if not self._resolve_use_accelerated_solver:
            self._raise_validation_error_at_loc(
                "'CurrentBC' is supported only by the accelerated charge solver. "
                "Either drive this contact with a 'VoltageBC' or set "
                "'use_accelerated_solver=True'.",
                "boundary_spec",
                current_bcs[0],
            )
        return self

    def _check_current_sweep_not_with_ssac_bias_selection(self) -> Self:
        """Reject ``at_voltages`` with a current sweep, whose bias voltages are solved for."""
        analysis_spec = self.analysis_spec
        if not isinstance(analysis_spec, (SSACAnalysis, IsothermalSSACAnalysis)):
            return self
        if analysis_spec.at_voltages is None:
            return self
        swept = [i for i in self._swept_bc_indices() if self._is_current_sweep(i)]
        if swept:
            self._raise_validation_error_at_loc(
                "'at_voltages' selects small-signal bias points by voltage, but "
                f"'boundary_spec[{swept[0]}]' sweeps a current, so the bias voltages are "
                "solved for rather than known in advance. Remove 'at_voltages'; the "
                "small-signal analysis then runs at the last entry of 'current'.",
                "analysis_spec",
                "at_voltages",
            )
        return self

    def _check_current_bc_not_on_conduction(self) -> Self:
        """Reject ``CurrentBC`` on a conduction simulation.

        ``CurrentBC`` prescribes a terminal current and is supported only for
        charge simulations on the accelerated solver. A conduction simulation
        has no such contact; use ``VoltageBC`` instead.
        """
        current_bcs = self._current_bc_indices()
        if not current_bcs:
            return self
        self._raise_validation_error_at_loc(
            "'CurrentBC' prescribes a terminal current and is supported only for charge "
            "simulations on the accelerated solver; for a conduction simulation use "
            "'VoltageBC' instead.",
            "boundary_spec",
            current_bcs[0],
        )
        return self

    def _check_heterojunction_sr_placement(
        self,
        index: int,
        model: HeterojunctionSurfaceShockleyReedHallRecombination,
        placement: BCPlacementType,
    ) -> None:
        """Require a heterojunction SR model on an interface named after its two sides."""
        if isinstance(placement, StructureStructureInterface):
            names, kind = placement.structures, "structure"
        elif isinstance(placement, MediumMediumInterface):
            names, kind = placement.mediums, "medium"
        else:
            self._raise_validation_error_at_loc(
                "'HeterojunctionSurfaceShockleyReedHallRecombination' is placed with a "
                f"'{placement.type}'. It needs a 'StructureStructureInterface' or a "
                "'MediumMediumInterface' that names the two sides of the heterojunction.",
                "boundary_spec",
                index,
                "placement",
            )
        side_names = {side.name for side in model.sides}
        if set(names) != side_names:
            self._raise_validation_error_at_loc(
                f"The side names {sorted(side_names)} of "
                "'HeterojunctionSurfaceShockleyReedHallRecombination' do not match the "
                f"{kind} names {list(names)} of its '{placement.type}' placement. Name "
                f"each side after the {kind} on that side of the interface.",
                "boundary_spec",
                index,
                "placement",
            )

    def _check_surface_recombination_requires_accelerated(self) -> Self:
        """Reject ``SurfaceRecombinationBC`` when the accelerated charge solver is off."""
        has_sr = any(isinstance(bc.condition, SurfaceRecombinationBC) for bc in self.boundary_spec)
        if not has_sr:
            return self
        # May raise if the user forced an unsupported configuration; let it propagate.
        if not self._resolve_use_accelerated_solver:
            raise SetupError(
                "'SurfaceRecombinationBC' is supported only by the "
                "accelerated charge solver. Either remove the surface "
                "recombination boundary condition or set "
                "'use_accelerated_solver=True'."
            )
        return self

    def _check_surface_recombination_not_stacked_with_schottky(self) -> Self:
        """Reject ``SurfaceRecombinationBC`` sharing a placement with a Schottky
        contact, which does not support a surface recombination overlay."""
        for i, sr_bc in enumerate(self.boundary_spec):
            if not isinstance(sr_bc.condition, SurfaceRecombinationBC):
                continue
            for j, other_bc in enumerate(self.boundary_spec):
                if i == j or not self._bc_is_schottky(other_bc):
                    continue
                if self._placements_overlap(sr_bc.placement, other_bc.placement):
                    raise SetupError(
                        "'SurfaceRecombinationBC' cannot share a placement with a "
                        'Schottky contact (model="schottky_mott"). '
                        "Contact surface recombination overlays are supported only "
                        'for ohmic contacts (model="ohmic").'
                    )
        return self

    def _check_surface_recombination_qf_on_voltage_overlay(self) -> Self:
        """Reject non-zero ``Q_f`` on a placement shared with a contact,
        which screens fixed sheet charge."""
        for i, sr_bc in enumerate(self.boundary_spec):
            if not isinstance(sr_bc.condition, SurfaceRecombinationBC):
                continue
            if not sr_bc.condition.Q_f:
                continue
            for j, other_bc in enumerate(self.boundary_spec):
                if i == j or not isinstance(other_bc.condition, (VoltageBC, CurrentBC)):
                    continue
                if self._placements_overlap(sr_bc.placement, other_bc.placement):
                    raise SetupError(
                        "'SurfaceRecombinationBC' with a non-zero 'Q_f' "
                        f"cannot share a placement with '{type(other_bc.condition).__name__}'. The "
                        "metal contact pins the electrostatic potential and "
                        "screens fixed sheet charge, so the Poisson "
                        "contribution from Q_f would be ignored on the "
                        "overlay nodes."
                    )
        return self

    def _check_surface_recombination_tat_on_voltage_overlay(self) -> Self:
        """Reject a trap-assisted tunneling enhancement on a placement shared with
        a contact, where the surface term is contact exchange and not trap SRH."""
        for i, sr_bc in enumerate(self.boundary_spec):
            if not isinstance(sr_bc.condition, SurfaceRecombinationBC):
                continue
            model = sr_bc.condition.model
            if (
                not isinstance(model, SurfaceShockleyReedHallRecombination)
                or model.field_enhancement is None
            ):
                continue
            for j, other_bc in enumerate(self.boundary_spec):
                if i == j or not isinstance(other_bc.condition, (VoltageBC, CurrentBC)):
                    continue
                if self._placements_overlap(sr_bc.placement, other_bc.placement):
                    raise SetupError(
                        "'SurfaceShockleyReedHallRecombination' with a "
                        "'field_enhancement' cannot share a placement with "
                        f"'{type(other_bc.condition).__name__}'. On a contact overlay the surface term "
                        "relaxes the minority-carrier density towards the ohmic "
                        "equilibrium value, which is carrier exchange with the "
                        "metal rather than recombination through a trap, so a "
                        "trap-assisted tunneling enhancement has no meaning "
                        "there. Drop 'field_enhancement' on this face."
                    )
        return self

    def _check_surface_recombination_heterojunction_not_on_contact(self) -> Self:
        """Reject a heterojunction SR model on a placement shared with a contact,
        where the surface term is contact exchange and not trap SRH."""
        for i, sr_bc in enumerate(self.boundary_spec):
            if not isinstance(sr_bc.condition, SurfaceRecombinationBC) or not isinstance(
                sr_bc.condition.model, HeterojunctionSurfaceShockleyReedHallRecombination
            ):
                continue
            for j, other_bc in enumerate(self.boundary_spec):
                if i == j or not isinstance(other_bc.condition, (VoltageBC, CurrentBC)):
                    continue
                if self._placements_overlap(sr_bc.placement, other_bc.placement):
                    self._raise_validation_error_at_loc(
                        "'HeterojunctionSurfaceShockleyReedHallRecombination' cannot share "
                        f"a placement with '{type(other_bc.condition).__name__}': on a contact the surface term is "
                        "carrier exchange with the metal, not recombination through an "
                        "interface trap.",
                        "boundary_spec",
                        i,
                        "placement",
                    )
        return self

    def _check_surface_recombination_not_stacked_with_insulating_bc(self) -> Self:
        """Reject ``SurfaceRecombinationBC`` sharing a placement with ``InsulatingBC``,
        whose zero-flux condition it already replaces."""
        for i, sr_bc in enumerate(self.boundary_spec):
            if not isinstance(sr_bc.condition, SurfaceRecombinationBC):
                continue
            for j, other_bc in enumerate(self.boundary_spec):
                if i == j or not isinstance(other_bc.condition, InsulatingBC):
                    continue
                if self._placements_overlap(sr_bc.placement, other_bc.placement):
                    raise SetupError(
                        "'SurfaceRecombinationBC' cannot share a placement "
                        "with 'InsulatingBC'. The SR Robin term already "
                        "replaces the natural zero-flux BC; drop the "
                        "'InsulatingBC' spec on this face."
                    )
        return self

    def _check_surface_recombination_no_duplicate_placement(self) -> Self:
        """Reject two ``SurfaceRecombinationBC`` entries on the same placement."""
        for i, bc_i in enumerate(self.boundary_spec):
            if not isinstance(bc_i.condition, SurfaceRecombinationBC):
                continue
            for j in range(i + 1, len(self.boundary_spec)):
                bc_j = self.boundary_spec[j]
                if not isinstance(bc_j.condition, SurfaceRecombinationBC):
                    continue
                if self._placements_overlap(bc_i.placement, bc_j.placement):
                    raise SetupError(
                        "Two 'SurfaceRecombinationBC' entries share the same "
                        "placement. Use distinct placements (e.g. one "
                        "'MediumMediumInterface' per semiconductor face) so "
                        "each interface picks up its own kinetic model."
                    )
        return self

    def _medium_pair_for_structure_pair(self, structures: tuple[str, str]) -> frozenset[str] | None:
        """Return the medium-pair key for a structure pair, if it is unambiguous."""
        structure_to_medium = {
            structure.name: structure.medium.name
            for structure in self.structures
            if structure.name is not None and structure.medium.name is not None
        }
        medium_names = tuple(structure_to_medium.get(name) for name in structures)
        if None in medium_names or medium_names[0] == medium_names[1]:
            return None
        return frozenset(medium_names)

    def _placements_overlap(self, first: Any, second: Any) -> bool:
        """Conservatively detect placement overlaps used by SR validation.

        Interface pairs are treated as unordered, ``MediumMediumInterface``
        is treated as a material-wide superset of matching structure
        interfaces, and simulation-boundary surface lists are considered
        overlapping if they share any surface.
        """
        if first == second:
            return True

        if isinstance(first, StructureStructureInterface) and isinstance(
            second, StructureStructureInterface
        ):
            return frozenset(first.structures) == frozenset(second.structures)

        if isinstance(first, MediumMediumInterface) and isinstance(second, MediumMediumInterface):
            return frozenset(first.mediums) == frozenset(second.mediums)

        if isinstance(first, StructureStructureInterface) and isinstance(
            second, MediumMediumInterface
        ):
            return self._medium_pair_for_structure_pair(first.structures) == frozenset(
                second.mediums
            )

        if isinstance(first, MediumMediumInterface) and isinstance(
            second, StructureStructureInterface
        ):
            return self._placements_overlap(second, first)

        if isinstance(first, SimulationBoundary) and isinstance(second, SimulationBoundary):
            return bool(set(first.surfaces) & set(second.surfaces))

        if isinstance(first, StructureSimulationBoundary) and isinstance(
            second, StructureSimulationBoundary
        ):
            return first.structure == second.structure and bool(
                set(first.surfaces) & set(second.surfaces)
            )

        if isinstance(first, SimulationBoundary) and isinstance(
            second, StructureSimulationBoundary
        ):
            return bool(set(first.surfaces) & set(second.surfaces))

        if isinstance(first, StructureSimulationBoundary) and isinstance(
            second, SimulationBoundary
        ):
            return self._placements_overlap(second, first)

        # A StructureBoundary names a whole structure's boundary, and surface
        # recombination never applies to faces a contact already occupies. Such
        # a placement is therefore not an overlap: it is a valid configuration
        # in which the contacted faces carry no surface recombination.

        return False

    def _warn_non_accelerated_ignores_electron_affinity(self) -> Self:
        """Warn when the non-accelerated charge solver path will ignore electron affinity."""
        try:
            use_accelerated = self._resolve_use_accelerated_solver
        except SetupError:
            # Solver routing itself is invalid; the downstream error will surface
            # on its own and there is nothing to warn about here.
            return self
        if use_accelerated:
            return self

        for _loc, semiconductor in self._iter_semiconductor_charge_media():
            if (
                semiconductor.electron_affinity is not None
                and semiconductor.electron_affinity != 0.0
            ):
                log.warning(
                    "'SemiconductorMedium.electron_affinity' is currently supported only "
                    "when 'use_accelerated_solver=True'. With the non-accelerated charge "
                    "solver this value is ignored and an electron affinity of 0 eV is used. "
                    "Set 'use_accelerated_solver=True' to honour the configured value."
                )
                break
        return self

    def _domain_surface_box(self, surface: BoxSurface) -> Box | None:
        """Zero-thickness :class:`.Box` for one domain face; ``None`` if unbounded there."""
        axis = "xyz".index(surface[0])
        position = self.bounds[0 if surface[1] == "-" else 1][axis]
        if not math.isfinite(position):
            return None
        center = list(self.center)
        size = list(self.size)
        center[axis] = position
        size[axis] = 0.0
        return Box(center=center, size=size)

    def _takes_part_in_heat(self, medium: StructureMediumType) -> bool:
        """Whether ``medium`` carries a temperature the heat solve constrains.

        Mirrors ``Zone.remove_from_heat`` in the mesher: a fluid holds no temperature,
        and a charge simulation hands its conductors to the electric solve instead.
        """
        if not isinstance(medium.heat_spec, SolidMedium):
            return False
        if TCADAnalysisTypes.CHARGE in self._check_simulation_types():
            return not isinstance(medium.charge, ChargeConductorMedium)
        return True

    def _overlaps_face(self, face: Box, geometry: Geometry) -> bool:
        """Whether ``geometry`` covers part of ``face`` with positive area.

        Contact along an edge or a corner carries no boundary patch, so touching bounds
        is not enough. A domain flat in a second dimension leaves no area to measure and
        falls back to plain intersection.
        """
        if face.size.count(0.0) != 1:
            return face.intersects(geometry)
        return any(shape.area > 0 for shape in face.intersections_with(geometry))

    def _reaches_heat_solve(self, face: Box) -> bool:
        """Whether a medium taking part in the heat solve covers part of ``face``."""
        background = Structure(geometry=Box(size=self.size, center=self.center), medium=self.medium)
        structures = [background, *self.structures]
        participating = [s for s in structures if self._takes_part_in_heat(s.medium)]
        if not any(self._overlaps_face(face, s.geometry) for s in participating):
            return False
        if face.size.count(0.0) != 1:
            return True
        # and it has to be the medium left exposed there, not one hidden behind another
        return any(
            self._takes_part_in_heat(medium)
            for medium in Scene.intersecting_media(face, structures)
        )

    def _heat_bc_placement_problem(self, placement: BCPlacementType) -> str | None:
        """Why a heat boundary condition at ``placement`` reaches no heat solve, if so.

        Conservative: a placement this side cannot resolve reports no problem, leaving
        the exact call to the mesher, which has the zones. Each branch knows a different
        reason, so it returns its own rather than one blaming fluid for all of them.
        """
        sides = self._placement_side_media(placement)
        if sides is not None:
            if any(self._takes_part_in_heat(side) for side in sides):
                return None
            return "is placed on an interface where neither side takes part in the heat solve"

        if isinstance(placement, SimulationBoundary | StructureSimulationBoundary):
            faces = tuple(self._domain_surface_box(surface) for surface in placement.surfaces)
            if any(face is None for face in faces):
                return None
            if isinstance(placement, SimulationBoundary):
                if any(self._reaches_heat_solve(face) for face in faces):
                    return None
                return (
                    "is placed on simulation boundary faces that reach nothing taking "
                    "part in the heat solve"
                )
            # Only the named structure's share of those faces carries the condition, so
            # it has to reach one of them. Whether it also owns that share depends on
            # the structures above it, which the mesher resolves.
            structure = next((s for s in self.structures if s.name == placement.structure), None)
            if structure is None:
                return None
            if not any(self._overlaps_face(face, structure.geometry) for face in faces):
                return (
                    f"names structure '{placement.structure}', which does not reach the "
                    "simulation boundary faces it lists"
                )
            # The patch is the named structure's share of the face, so it alone decides;
            # a participating background elsewhere on the same face does not rescue it.
            if self._takes_part_in_heat(structure.medium):
                return None
            return f"names structure '{placement.structure}', which takes no part in the heat solve"

        # A fluid structure's boundary can still coincide with the solid facing it.
        return None

    def _heat_bcs_reach_solids(self) -> Self:
        """Report a heat BC placed where no solid material is present.

        Temperature is solved only inside solid media, so the mesher drops such a
        condition and leaves an adiabatic wall in its place. An error when that leaves
        nothing to anchor the temperature -- the steady solve then diverges while the
        task reports success -- and a warning when another condition still anchors it,
        which only makes the dropped one redundant. An unsteady analysis is pinned by
        its initial condition, so nothing there is left undetermined either.

        An anchor is any condition this side cannot disprove, ``_heat_bc_placement_problem``
        leaving what it cannot settle to the mesher. That errs towards the warning: a
        well-posed setup stays buildable, and a setup only this side thought anchored
        still fails at mesh time, where ``_check_unapplied_heat_bcs`` has the zones and
        is exact.
        """
        # (index, condition, message up to the consequence)
        dropped = []
        anchored = isinstance(self.analysis_spec, UnsteadyHeatAnalysis)
        for index, bc in enumerate(self.boundary_spec):
            if not isinstance(bc.condition, HeatBCTypes):
                continue
            problem = self._heat_bc_placement_problem(bc.placement)
            if problem is None:
                anchored |= not isinstance(bc.condition, NeumannHeatBCTypes)
                continue
            dropped.append(
                (
                    index,
                    bc.condition,
                    f"'{type(bc.condition).__name__}' {problem}. Temperature is solved "
                    "only inside solid media -- and a charge simulation solves its "
                    "conductors electrically instead -- so this condition would be "
                    "dropped when the mesh is built and the surface left adiabatic",
                )
            )

        if not dropped:
            return self

        if anchored:
            for index, _condition, message in dropped:
                log.warning(
                    f"{message}, with no effect on the result.",
                    custom_loc=["boundary_spec", index, "placement"],
                )
            return self

        # Name the condition whose loss unanchors the solve, not a Neumann one that
        # happens to come first.
        index, _condition, message = next(
            (item for item in dropped if not isinstance(item[1], NeumannHeatBCTypes)),
            dropped[0],
        )
        self._raise_validation_error_at_loc(
            f"{message}, leaving nothing to anchor the temperature.",
            "boundary_spec",
            index,
            "placement",
        )
        return self

    def _not_all_neumann(self) -> Self:
        """Make sure not all BCs are of Neumann type"""

        # SurfaceRecombinationBC is Robin in the carrier rows, but it does
        # not anchor the electrostatic potential.
        NeumannBCsCharge = (CurrentBC, InsulatingBC, SurfaceRecombinationBC)

        simulation_types = self._check_simulation_types()

        raise_error = False
        for sim_type in simulation_types:
            if sim_type == TCADAnalysisTypes.HEAT:
                # Transient heat is well-posed with all-Neumann BCs (initial
                # condition pins the solution); only steady state is ambiguous.
                if isinstance(self.analysis_spec, UnsteadyHeatAnalysis):
                    continue
                type_bcs = [
                    bc for bc in self.boundary_spec if isinstance(bc.condition, HeatBCTypes)
                ]
                if len(type_bcs) == 0 or all(
                    isinstance(bc.condition, NeumannHeatBCTypes) for bc in type_bcs
                ):
                    raise_error = True
            elif sim_type == TCADAnalysisTypes.CONDUCTION:
                type_bcs = [
                    bc for bc in self.boundary_spec if isinstance(bc.condition, ElectricBCTypes)
                ]
                if len(type_bcs) == 0 or all(
                    isinstance(bc.condition, NeumannBCsCharge) for bc in type_bcs
                ):
                    raise_error = True

        names_neumann_Bcs = [BC.__name__ for BC in NeumannHeatBCTypes]
        names_neumann_Bcs.extend([BC.__name__ for BC in NeumannBCsCharge])
        if raise_error:
            raise SetupError(
                "Current 'HeatChargeSimulation' contains only Neumann-type boundary conditions. "
                "Steady-state solution is undefined in this case. "
                f"Current Neumann BCs are {names_neumann_Bcs}"
            )

        return self

    def _names_exist_grid_spec(self) -> Self:
        """Check grid spec structure/medium references."""
        structures_names = {s.name for s in self.structures}
        mediums_names = {s.medium.name for s in self.structures}
        mediums_names.add(self.medium.name)

        for structure_name in self.grid_spec.non_refined_structures:
            if structure_name not in structures_names:
                log.warning(
                    f"Structure '{structure_name}' listed as a non-refined structure in "
                    "'HeatChargeSimulation.grid_spec' is not present in 'HeatChargeSimulation.structures'"
                )

        if isinstance(self.grid_spec, DistanceUnstructuredGrid | AutoUnstructuredGrid):
            for med_ind, medium_name in enumerate(self.grid_spec.uniform_grid_mediums):
                if medium_name not in mediums_names:
                    self._raise_validation_error_at_loc(
                        f"Medium '{medium_name}' listed in 'grid_spec.uniform_grid_mediums' "
                        "is not found among simulation mediums.",
                        "grid_spec",
                        "uniform_grid_mediums",
                        med_ind,
                    )

        if isinstance(self.grid_spec, AutoUnstructuredGrid):
            for ref_ind, refinement in enumerate(self.grid_spec.interface_refinements):
                selection = refinement.selection
                if isinstance(selection, StructureBoundary):
                    if selection.structure not in structures_names:
                        self._raise_validation_error_at_loc(
                            f"Structure '{selection.structure}' provided in "
                            f"'grid_spec.interface_refinements[{ref_ind}].selection' "
                            f"(type '{selection.type}') is not found among simulation structures.",
                            "grid_spec",
                            "interface_refinements",
                            ref_ind,
                            "selection",
                        )
                elif isinstance(selection, StructureStructureInterface):
                    for struct_name in selection.structures:
                        if struct_name not in structures_names:
                            self._raise_validation_error_at_loc(
                                f"Structure '{struct_name}' provided in "
                                f"'grid_spec.interface_refinements[{ref_ind}].selection' "
                                f"(type '{selection.type}') is not found among simulation structures.",
                                "grid_spec",
                                "interface_refinements",
                                ref_ind,
                                "selection",
                            )
                elif isinstance(selection, MediumMediumInterface):
                    for med_name in selection.mediums:
                        if med_name not in mediums_names:
                            self._raise_validation_error_at_loc(
                                f"Medium '{med_name}' provided in "
                                f"'grid_spec.interface_refinements[{ref_ind}].selection' "
                                f"(type '{selection.type}') is not found among simulation mediums.",
                                "grid_spec",
                                "interface_refinements",
                                ref_ind,
                                "selection",
                            )

        return self

    def _warn_if_minimal_mesh_size_override(self) -> Self:
        """Warn if minimal mesh size limit overrides desired mesh size."""
        val = self.grid_spec
        max_size = np.max(self.size)
        min_dl = val.relative_min_dl * max_size
        desired_min_dl = val.min_mesh_size
        if isinstance(val, DistanceUnstructuredGrid):
            # Keep the grid property's geometry-processing contract unchanged;
            # this warning must also consider a finer requested bulk size.
            desired_min_dl = min(desired_min_dl, val.dl_bulk)

        if desired_min_dl < min_dl:
            log.warning(
                f"The resulting limit for minimal mesh size from parameter 'relative_min_dl={val.relative_min_dl}' is {min_dl}, while provided mesh size in 'grid_spec' is {desired_min_dl}. "
                "Consider lowering parameter 'relative_min_dl' if a finer grid is required."
            )

        return self

    def _names_exist_sources(self) -> Self:
        """Error if a heat-charge source point to non-existing structures."""
        structures_names = {s.name for s in self.structures}

        sources = [s for s in self.sources if not isinstance(s, HeatFromElectricSource)]

        for source in sources:
            for name in source.structures:
                if name not in structures_names:
                    raise SetupError(
                        f"Structure '{name}' provided in a '{source.type}' "
                        "is not found among simulation structures."
                    )
        return self

    def _check_medium_specs(self) -> Self:
        """Error if no appropriate specs."""

        sim_box = (Box(size=self.size, center=self.center),)

        failed_solid_idx, failed_elect_idx = self._check_cross_solids(sim_box)

        simulation_types = self._check_simulation_types()

        for sim_type in simulation_types:
            if sim_type == TCADAnalysisTypes.HEAT:
                if len(failed_solid_idx) > 0:
                    raise SetupError(
                        "No solid materials ('SolidSpec') are detected in heat simulation. Solution domain is empty."
                    )
            elif sim_type == TCADAnalysisTypes.CONDUCTION:
                if len(failed_elect_idx) > 0:
                    raise SetupError(
                        "No conducting materials ('ChargeConductorMedium') are detected in conduction simulation. Solution domain is empty."
                    )

        return self

    def _check_semiconductor_electric_bcs_require_charge_analysis(self) -> Self:
        """Require an explicit charge analysis for electrical semiconductor simulations.

        The rule is deliberately not placement-aware: '_check_simulation_types' suppresses
        the CONDUCTION type from a global 'semiconductor_present' flag, so any electrical
        boundary anywhere in a model that holds a semiconductor anywhere leaves the
        simulation unclassified at validation time while the runtime classifier still calls
        it conduction. Match that predicate here so no such setup slips through."""
        if isinstance(self.analysis_spec, ChargeTypes):
            return self
        if not any(self._iter_semiconductor_charge_media()):
            return self
        if any(isinstance(bc.condition, ElectricBCTypes) for bc in self.boundary_spec):
            raise SetupError(
                "Simulations that contain a 'SemiconductorMedium' together with electrical "
                "boundary conditions require a charge 'analysis_spec', such as "
                "'IsothermalSteadyChargeDCAnalysis'. A charge analysis additionally requires "
                "at least two 'VoltageBC' contacts, so boundaries of another electrical type "
                "may need to be converted as well."
            )
        return self

    def _check_simulation_types(
        self,
        HeatBCTypes: tuple[type, ...] = HeatBCTypes,
        ElectricBCTypes: tuple[type, ...] = ElectricBCTypes,
        HeatSourceTypes: tuple[type, ...] = HeatSourceTypes,
    ) -> list[TCADAnalysisTypes]:
        """Given model dictionary ``values``, check the type of simulations to be run
        based on BCs and sources.
        """
        simulation_types = []

        analysis_spec = self.analysis_spec

        if isinstance(analysis_spec, ChargeTypes):
            simulation_types.append(TCADAnalysisTypes.CHARGE)

        semiconductor_present = any(self._iter_semiconductor_charge_media())

        for boundary in self.boundary_spec:
            if isinstance(boundary.condition, HeatBCTypes):
                simulation_types.append(TCADAnalysisTypes.HEAT)
            if isinstance(boundary.condition, ElectricBCTypes):
                # Add CONDUCTION type if we have no semiconductors
                if not semiconductor_present:
                    simulation_types.append(TCADAnalysisTypes.CONDUCTION)

        for source in self.sources:
            if not isinstance(source, HeatSourceTypes):
                continue
            # A non-isothermal charge analysis ignores 'HeatFromElectricSource'
            # (_warn_coupling_source_is_no_op), so it must not contribute a HEAT type
            # either: that would subject the user to the heat-only checks -- a required
            # 'TemperatureMonitor', not-all-Neumann -- on account of a source that does
            # nothing, right after being told it is ignored.
            if isinstance(source, HeatFromElectricSource) and self._thermal_solver_active:
                continue
            simulation_types.append(TCADAnalysisTypes.HEAT)

        return set(simulation_types)

    def _check_coupling_source_can_be_applied(self) -> Self:
        """Error if 'HeatFromElectricSource' has no conduction solve to draw heating from."""

        HeatSourceTypes_noCoupling = (UniformHeatSource, HeatSource)

        # Exclude the source itself: it must not establish the HEAT type it is validated
        # against.
        simulation_types = self._check_simulation_types(HeatSourceTypes=HeatSourceTypes_noCoupling)

        # Charge is rejected with a specific message elsewhere; let that one win.
        if TCADAnalysisTypes.CHARGE in simulation_types:
            return self

        if (
            TCADAnalysisTypes.HEAT in simulation_types
            and TCADAnalysisTypes.CONDUCTION in simulation_types
        ):
            return self

        for index, source in enumerate(self.sources):
            if not isinstance(source, HeatFromElectricSource):
                continue
            present = ", ".join(sorted(sim_type.name for sim_type in simulation_types)) or "none"
            self._raise_validation_error_at_loc(
                f"Using 'HeatFromElectricSource' requires both "
                f"{TCADAnalysisTypes.CONDUCTION.name} and {TCADAnalysisTypes.HEAT.name}, "
                "since the conduction solve supplies the volumetric heating the heat solve "
                f"consumes. The current simulation setup defines: {present}.",
                "sources",
                index,
            )

        return self

    def _check_charge_heat_coupling_unsupported(self) -> Self:
        """Reject 'HeatFromElectricSource' in a charge simulation; the bias is ambiguous.

        Non-isothermal specs are excluded: there the source is redundant rather than
        ambiguous, and ``_warn_coupling_source_is_no_op`` warns instead.
        """
        if not isinstance(self.analysis_spec, ChargeTypes) or self._thermal_solver_active:
            return self

        for index, source in enumerate(self.sources):
            if not isinstance(source, HeatFromElectricSource):
                continue
            self._raise_validation_error_at_loc(
                "'HeatFromElectricSource' cannot drive a heat simulation from a charge "
                "simulation. A charge simulation solves at a list of bias points while a "
                "heat simulation solves once, so the bias the temperature would correspond "
                "to is ambiguous. Record the charge simulation's self-heating with a "
                "'SelfHeatingMonitor' instead, then pass the bias point you want to a "
                "separate heat simulation: "
                '\'HeatSource(rate=data["self_heat"].to_spatial_data_array('
                "bounds=..., resolution=..., voltage=...), structures=[...])'.",
                "sources",
                index,
            )

        return self

    def _warn_heat_inputs_need_thermal_solve(self) -> Self:
        """Warn about heat BCs and heat sources in a charge analysis that solves no temperature.

        They are accepted and never applied. 'HeatFromElectricSource' has its own
        message in ``_check_charge_heat_coupling_unsupported``.
        """
        if not isinstance(self.analysis_spec, ChargeTypes) or self._thermal_solver_active:
            return self

        if self._temperature_prescribed:
            temperature = "a prescribed lattice temperature field"
        else:
            temperature = f"a uniform lattice temperature of {self.analysis_spec.temperature} K"
        consequence = (
            f"'{type(self.analysis_spec).__name__}' runs on {temperature} and solves no heat "
            "equation, so it is ignored. Remove it, or use 'SteadyChargeDCAnalysis' or "
            "'SSACAnalysis' without a 'temperature' to solve the temperature with self-heating. "
            "Future versions of Tidy3D may reject this setup."
        )
        for index, bc in enumerate(self.boundary_spec):
            if isinstance(bc.condition, HeatBCTypes):
                log.warning(
                    f"'boundary_spec[{index}]' ('{type(bc.condition).__name__}') has no effect "
                    f"here: {consequence}"
                )
        for index, source in enumerate(self.sources):
            if isinstance(source, (UniformHeatSource, HeatSource)):
                log.warning(
                    f"'sources[{index}]' ('{type(source).__name__}') has no effect here: "
                    f"{consequence}"
                )
        return self

    def _warn_coupling_source_is_no_op(self) -> Self:
        """Warn that 'HeatFromElectricSource' does nothing on a non-isothermal charge sim."""
        if not self._thermal_solver_active:
            return self
        if not any(isinstance(source, HeatFromElectricSource) for source in self.sources):
            return self

        log.warning(
            "'HeatFromElectricSource' is ignored in a non-isothermal charge simulation. "
            f"'{type(self.analysis_spec).__name__}' already couples self-heating to the "
            "temperature field self-consistently, so the source adds nothing. It can be "
            "removed."
        )
        return self

    def _validate_self_heating_monitor_requirements(self) -> Self:
        """'SelfHeatingMonitor' requires an electric (charge or conduction) analysis."""
        simulation_types = self._check_simulation_types()
        charge_configured = TCADAnalysisTypes.CHARGE in simulation_types
        electric_configured = charge_configured or TCADAnalysisTypes.CONDUCTION in simulation_types

        for index, mnt in enumerate(self.monitors):
            if not isinstance(mnt, SelfHeatingMonitor):
                continue
            if not electric_configured:
                self._raise_validation_error_at_loc(
                    f"'SelfHeatingMonitor' (monitor '{mnt.name}') records the volumetric "
                    "heating produced by an electric solve, so it is only available when a "
                    f"{TCADAnalysisTypes.CHARGE.name} or {TCADAnalysisTypes.CONDUCTION.name} "
                    "analysis is configured. A heat-only simulation generates no self-heating.",
                    "monitors",
                    index,
                )
            if charge_configured and self.use_accelerated_solver is False:
                self._raise_validation_error_at_loc(
                    f"'SelfHeatingMonitor' (monitor '{mnt.name}') is only available in a "
                    f"{TCADAnalysisTypes.CHARGE.name} simulation through the accelerated "
                    "charge solver, but 'use_accelerated_solver=False' was set. Remove the "
                    "monitor or use the accelerated solver (the default).",
                    "monitors",
                    index,
                )

        return self

    def _check_heat_sim(self) -> Self:
        """Make sure that heat simulations have at least one monitor defined."""
        if not any(isinstance(mnt, TemperatureMonitor) for mnt in self.monitors):
            raise SetupError(
                "Heat simulations require the definition of, at least, one "
                "'TemperatureMonitor' but none have been defined."
            )
        return self

    def _check_conduction_sim_monitors(self) -> Self:
        """Validate conduction simulations have at least one potential monitor."""
        if not any(isinstance(mnt, SteadyPotentialMonitor) for mnt in self.monitors):
            if any(isinstance(s, HeatFromElectricSource) for s in self.sources):
                log.warning(
                    "A Conduction simulation has been defined but no "
                    "SteadyPotentialMonitor has been defined. "
                )
            else:
                raise SetupError(
                    "Conduction simulations require the definition of, at least, one "
                    "'SteadyPotentialMonitor' but none have been defined."
                )
        return self

    def _check_conduction_sim_voltage_arrays(self) -> Self:
        """Validate conduction simulations don't use voltage arrays."""
        for bc in self.boundary_spec:
            if isinstance(bc.condition, VoltageBC):
                if isinstance(bc.condition.source, DCVoltageSource):
                    if len(bc.condition.source.voltage) > 1:
                        raise SetupError(
                            "A Conduction simulation has been defined but a VoltageBC with an array of voltages "
                            "has been defined. This is not supported in Conduction simulations."
                        )
        return self

    def _check_conduction_sim_structures(self) -> Self:
        """Validate conduction simulations include conductive multiphysics media."""
        if all(isinstance(s.medium, Medium) for s in self.structures):
            raise SetupError(
                "Conduction simulations must be defined using 'MultiPhysicsMedium' but none have been defined."
            )
        if not any(isinstance(s.medium.charge, ChargeConductorMedium) for s in self.structures):
            raise SetupError(
                "Conduction simulations require at least one structure with a 'ChargeConductorMedium' "
                "but none have been defined."
            )

        return self

    def _check_conduction_sim_permittivity(self) -> Self:
        """Reject an anisotropic permittivity in conduction simulations, which take only a
        scalar permittivity."""
        for loc in self._anisotropic_permittivity_locs():
            self._raise_validation_error_at_loc(
                "An anisotropic permittivity ('AnisotropicPermittivity') is supported only in "
                "charge simulations. Provide a scalar 'permittivity' for conduction simulations.",
                *loc,
            )
        return self

    def _check_symmetry_permittivity(self) -> Self:
        """Reject reflection symmetry across a plane that an anisotropic permittivity is not
        mirror-symmetric across."""
        for loc, axis in self._symmetry_breaking_permittivity():
            name = "xyz"[axis]
            self._raise_validation_error_at_loc(
                f"'symmetry' is on along {name}, but this 'AnisotropicPermittivity' couples {name} "
                "to the other axes, so the medium is not mirror-symmetric across that plane. Set "
                f"'symmetry' to 0 along {name}, or align a principal axis with {name}.",
                *loc,
            )
        return self

    def _estimate_charge_mesh_size(self) -> Self:
        """Make an estimate of the mesh size and raise a warning if too big.
        NOTE: this is a very rough estimate. The back-end will actually stop
        execution based on actual node-count."""

        if TCADAnalysisTypes.CHARGE not in self._get_simulation_types():
            return self

        # let's raise a warning if the estimate is larger than 2M nodes
        max_nodes = 2e6
        nodes_estimate = 0

        grid_spec = self.grid_spec

        non_refined_structures = grid_spec.non_refined_structures

        sim_center = self.center
        sim_size = self.size

        if isinstance(grid_spec, UniformUnstructuredGrid):
            dl_min = grid_spec.dl
            dl_max = dl_min
        elif isinstance(grid_spec, DistanceUnstructuredGrid):
            dl_min = grid_spec.dl_interface
            dl_max = grid_spec.dl_bulk
        elif isinstance(grid_spec, AutoUnstructuredGrid):
            dl_min = grid_spec._automatic_min_mesh_size
            dl_max = grid_spec.dl_bulk
        else:
            raise TypeError(
                f"Unsupported grid_spec type '{type(grid_spec).__name__}'. "
                "Expected UniformUnstructuredGrid, DistanceUnstructuredGrid, or "
                "AutoUnstructuredGrid."
            )

        for struct in self.structures:
            name = struct.name
            bounds = np.array(struct.geometry.bounds)
            for dim in range(3):
                bounds[0, dim] = max(bounds[0, dim], sim_center[dim] - sim_size[dim] / 2)
                bounds[1, dim] = min(bounds[1, dim], sim_center[dim] + sim_size[dim] / 2)

            dl = dl_min
            if name in non_refined_structures:
                dl = dl_max
            nodes_structure = 1
            for coord_min, coord_max in zip(bounds[0], bounds[1]):
                if (
                    (coord_min != coord_max)
                    and (np.abs(coord_min) != np.inf)
                    and (np.abs(coord_max) != np.inf)
                ):
                    nodes_structure = nodes_structure * (coord_max - coord_min) / dl

            nodes_estimate = nodes_estimate + nodes_structure

        if nodes_estimate > max_nodes:
            log.warning(
                "It is estimated that the mesh will be larger than the currently "
                "supported mesh size for Charge simulations. The simulation may be "
                "submitted but if the maximum number of nodes is indeed exceeded "
                "the pipeline will be stopped. If this happens the grid specification "
                "may need to be modified."
            )
        return self

    def _check_self_heating_has_heat_sink(self) -> Self:
        """Warn if a self-heating charge analysis has no boundary that lets the heat leave.

        Surfaces without a thermal BC are adiabatic, so the steady heat equation has no
        solution unless a 'TemperatureBC', a 'RadiationBC' or a 'ConvectionBC' that exchanges
        heat (by convection or by its 'emissivity') exists. The suggested sinks leave out
        radiation when 'use_accelerated_solver=False', which refuses it. Raises instead when
        '_not_all_neumann' refuses the setup anyway, since its generic message would otherwise
        win.
        """
        if not self._thermal_solver_active:
            return self
        if any(self._is_heat_sink(bc.condition) for bc in self.boundary_spec):
            return self
        if self.use_accelerated_solver:
            sinks = (
                "a 'TemperatureBC', a 'ConvectionBC' with a positive 'transfer_coeff' or "
                "'emissivity', or a 'RadiationBC'"
            )
        else:
            sinks = "a 'TemperatureBC' or a 'ConvectionBC' with a positive 'transfer_coeff'"
        message = (
            "A non-isothermal charge analysis solves a steady heat equation for the "
            "self-heating, which has no solution unless the heat can leave the device: "
            "surfaces without a thermal boundary condition are adiabatic, and a 'HeatFluxBC', "
            "'ThermalContactResistance' or 'ConvectionBC' with a zero 'transfer_coeff' and no "
            f"'emissivity' does not take heat out. Add {sinks} where the device is heat-sunk."
        )
        heat_bcs = [bc for bc in self.boundary_spec if isinstance(bc.condition, HeatBCTypes)]
        if TCADAnalysisTypes.HEAT in self._check_simulation_types() and all(
            isinstance(bc.condition, NeumannHeatBCTypes) for bc in heat_bcs
        ):
            self._raise_validation_error_at_loc(message, "boundary_spec")
        log.warning(f"{message} Future versions of Tidy3D may reject this setup.")
        return self

    @staticmethod
    def _is_heat_sink(condition: Any) -> bool:
        """Whether a heat BC can carry heat out of the domain."""
        if isinstance(condition, (TemperatureBC, RadiationBC)):
            return True
        if isinstance(condition, ConvectionBC):
            return (
                isinstance(condition.transfer_coeff, VerticalNaturalConvectionCoeffModel)
                or condition.transfer_coeff > 0
                or bool(condition.emissivity)
            )
        return False

    def _check_heat_feature_placement_in_charge(self) -> Self:
        """Reject heat features placed where the electric side of a self-heating run cannot follow.

        The heat solve of a non-isothermal charge simulation applies advection and resistive
        interfaces, but the carriers of a moving semiconductor are not advected, a resistive
        interface between two semiconductors would break the carrier coupling across it, and
        conductors take no part in the heat solve. The solver choice is checked by
        '_accelerated_only_features'. Heat, conduction+heat, and isothermal or
        prescribed-temperature charge analyses are unaffected."""
        if not self._thermal_solver_active:
            return self

        for loc, medium in self._iter_media_with_locs():
            heat_spec = (
                medium if isinstance(medium, SolidMedium) else getattr(medium, "heat_spec", None)
            )
            velocity = getattr(heat_spec, "velocity", None)
            moving = velocity is not None and any(v != 0.0 for v in velocity)
            if moving and isinstance(self._charge_spec(medium), SemiconductorMedium):
                self._raise_validation_error_at_loc(
                    f"{self._medium_label(loc)} is a semiconductor with an advection "
                    "velocity ('SolidMedium.velocity'). A non-isothermal charge simulation "
                    "advects the heat of a moving solid but not its carriers, so the two "
                    "would move apart. Remove 'velocity' from this medium.",
                    *loc,
                )

        for i, bc in enumerate(self.boundary_spec):
            if not isinstance(bc.condition, ThermalContactResistance):
                continue
            sides = self._placement_side_media(bc.placement) or []
            charges = [self._charge_spec(side) for side in sides]
            if any(isinstance(charge, ChargeConductorMedium) for charge in charges):
                self._raise_validation_error_at_loc(
                    "'ThermalContactResistance' cannot be placed on a conductor "
                    "('ChargeConductorMedium') in a charge simulation: conductors take no part "
                    "in its heat solve, so this face bounds it. To model a resistive contact to "
                    "a metal at a known temperature, place a 'ConvectionBC' here with "
                    "'transfer_coeff' equal to 1/'resistance' and that temperature as "
                    "'ambient_temperature'.",
                    "boundary_spec",
                    i,
                    "placement",
                )
            if len(charges) == 2 and all(
                isinstance(charge, SemiconductorMedium) for charge in charges
            ):
                self._raise_validation_error_at_loc(
                    "'ThermalContactResistance' between two semiconductors is not supported "
                    "in non-isothermal charge simulations. Where both have the same band "
                    "parameters, the charge solver ties their carrier densities across the "
                    "interface, which a temperature jump would make inconsistent. A resistive "
                    "heterojunction is not supported either. Place it where at least one side "
                    "is an insulator.",
                    "boundary_spec",
                    i,
                    "placement",
                )
        return self

    def _warn_conductor_heat_medium_unused(self) -> Self:
        """Warn that the solid heat medium of a conductor is not used in a self-heating run.

        Conductors take no part in the heat solve of a charge simulation, so the heat medium the
        user gave them has no effect. One warning names every such conductor."""
        if not self._thermal_solver_active:
            return self
        labels = [
            self._medium_label(loc)
            for loc, medium in self._iter_media_with_locs()
            if isinstance(self._charge_spec(medium), ChargeConductorMedium)
            and isinstance(getattr(medium, "heat_spec", None), SolidMedium)
        ]
        if not labels:
            return self
        subject = (
            f"{labels[0]} is a conductor"
            if len(labels) == 1
            else f"{', '.join(labels)} are conductors"
        )
        log.warning(
            f"{subject} whose heat medium is not used: conductors take no part in the heat "
            "solve of a self-heating charge simulation, so the faces they share with other "
            "solids are adiabatic. To let heat leave through a conductor, place a "
            "'TemperatureBC' or 'ConvectionBC' on those faces."
        )
        return self

    def _check_solids_have_charge_medium(self) -> Self:
        """Require a 'charge' medium on every solid of a non-isothermal charge simulation.

        A solid that conducts heat in the coupled thermal solve also takes part in the
        electrostatics, so its electrical role and permittivity must be declared. Fluids, and
        isothermal or prescribed-temperature charge analyses, are unaffected."""
        if not self._thermal_solver_active:
            return self

        media_sources = [(("medium",), self.medium, "The background 'medium'")]
        media_sources.extend(
            (("structures", i), struct.medium, f"Structure '{struct.name}'")
            if struct.name
            else (("structures", i), struct.medium, f"'structures[{i}]'")
            for i, struct in enumerate(self.structures)
        )
        for loc, medium, label in media_sources:
            heat_spec = (
                medium if isinstance(medium, SolidMedium) else getattr(medium, "heat_spec", None)
            )
            if not isinstance(heat_spec, SolidMedium):
                continue
            charge = None if medium is heat_spec else medium.charge
            if charge is None:
                self._raise_validation_error_at_loc(
                    f"{label} has a solid heat medium but no 'charge' medium. In a "
                    "non-isothermal charge simulation every solid also takes part in the "
                    "electrostatics. For an insulating solid, use a 'MultiPhysicsMedium' with "
                    "'charge=ChargeInsulatorMedium(permittivity=...)'.",
                    *loc,
                )
        return self

    @equal_aspect
    @add_ax_if_none
    def plot(
        self,
        x: float | None = None,
        y: float | None = None,
        z: float | None = None,
        ax: Ax = None,
        source_alpha: float | None = None,
        monitor_alpha: float | None = None,
        hlim: tuple[float, float] | None = None,
        vlim: tuple[float, float] | None = None,
        fill_structures: bool = True,
        **patch_kwargs: Any,
    ) -> Ax:
        """Plot each of simulation's components on a plane defined by one nonzero x,y,z coordinate.

        Parameters
        ----------
        x : float = None
            position of plane in x direction, only one of x, y, z must be specified to define plane.
        y : float = None
            position of plane in y direction, only one of x, y, z must be specified to define plane.
        z : float = None
            position of plane in z direction, only one of x, y, z must be specified to define plane.
        ax : matplotlib.axes._subplots.Axes = None
            Matplotlib axes to plot on, if not specified, one is created.
        source_alpha : float = None
            Opacity of the sources. If ``None``, uses Tidy3d default.
        monitor_alpha : float = None
            Opacity of the monitors. If ``None``, uses Tidy3d default.
        hlim : Tuple[float, float] = None
            The x range if plotting on xy or xz planes, y range if plotting on yz plane.
        vlim : Tuple[float, float] = None
            The z range if plotting on xz or yz planes, y plane if plotting on xy plane.
        fill_structures : bool = True
            Whether to fill structures with color or just draw outlines.

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.
        """

        # Call the parent's plot method
        ax = super().plot(
            x=x,
            y=y,
            z=z,
            ax=ax,
            source_alpha=source_alpha,
            monitor_alpha=monitor_alpha,
            hlim=hlim,
            vlim=vlim,
            fill_structures=fill_structures,
            **patch_kwargs,
        )

        # Add boundaries based on simulation type
        # NOTE: there's no need to add heat boundaries since
        # they are already added in the parent 'plot' method.
        simulation_types = self._get_simulation_types()
        if (
            TCADAnalysisTypes.CHARGE in simulation_types
            or TCADAnalysisTypes.CONDUCTION in simulation_types
        ):
            ax = self.plot_boundaries(ax=ax, x=x, y=y, z=z, property="electric_conductivity")

        return ax

    @equal_aspect
    @add_ax_if_none
    def plot_property(
        self,
        x: float | None = None,
        y: float | None = None,
        z: float | None = None,
        ax: Ax = None,
        alpha: float | None = None,
        source_alpha: float | None = None,
        monitor_alpha: float | None = None,
        property: Literal[
            "heat_conductivity", "electric_conductivity", "source"
        ] = "heat_conductivity",
        hlim: tuple[float, float] | None = None,
        vlim: tuple[float, float] | None = None,
    ) -> Ax:
        """Plot each of simulation's components on a plane defined by one nonzero x,y,z coordinate.

        Parameters
        ----------
        x : float = None
            position of plane in x direction, only one of x, y, z must be specified to define plane.
        y : float = None
            position of plane in y direction, only one of x, y, z must be specified to define plane.
        z : float = None
            position of plane in z direction, only one of x, y, z must be specified to define plane.
        ax : matplotlib.axes._subplots.Axes = None
            Matplotlib axes to plot on, if not specified, one is created.
        alpha : float = None
            Opacity of the structures being plotted.
            Defaults to the structure default alpha.
        source_alpha : float = None
            Opacity of the sources. If ``None``, uses Tidy3d default.
        monitor_alpha : float = None
            Opacity of the monitors. If ``None``, uses Tidy3d default.
        property : str = "heat_conductivity"
            Specified the type of simulation for which the plot will be tailored.
            Options are ["heat_conductivity", "electric_conductivity", "source"]
        hlim : tuple[float, float] = None
            The x range if plotting on xy or xz planes, y range if plotting on yz plane.
        vlim : tuple[float, float] = None
            The z range if plotting on xz or yz planes, y plane if plotting on xy plane.

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.
        """

        hlim, vlim = Scene._get_plot_lims(
            bounds=self.simulation_bounds, x=x, y=y, z=z, hlim=hlim, vlim=vlim
        )

        cbar_cond = True

        simulation_types = self._get_simulation_types()
        if property == "source" and len(simulation_types) > 1:
            raise ValueError(
                "'plot_property' must be called with argument 'property' in "
                "'HeatChargeSimulations' with multiple physics, i.e., a 'HeatChargeSimulation' "
                f"with both {TCADAnalysisTypes.HEAT.name} and "
                f"{TCADAnalysisTypes.CONDUCTION.name} simulation properties."
            )
        if len(simulation_types) == 1:
            if (
                property == "heat_conductivity" and TCADAnalysisTypes.CONDUCTION in simulation_types
            ) or (
                property == "electric_conductivity" and TCADAnalysisTypes.HEAT in simulation_types
            ):
                raise ValueError(
                    f"'property' in 'plot_property()' was defined as {property} but the "
                    f"simulation is of type {simulation_types[0]}."
                )

        if property != "source":
            ax = self.scene.plot_heat_charge_property(
                ax=ax,
                x=x,
                y=y,
                z=z,
                cbar=cbar_cond,
                alpha=alpha,
                hlim=hlim,
                vlim=vlim,
                property=property,
            )
        ax = self.plot_sources(
            ax=ax, x=x, y=y, z=z, property=property, alpha=source_alpha, hlim=hlim, vlim=vlim
        )
        ax = self.plot_monitors(ax=ax, x=x, y=y, z=z, alpha=monitor_alpha, hlim=hlim, vlim=vlim)
        ax = self.plot_boundaries(ax=ax, x=x, y=y, z=z, property=property)
        ax = Scene._set_plot_bounds(
            bounds=self.simulation_bounds, ax=ax, x=x, y=y, z=z, hlim=hlim, vlim=vlim
        )
        ax = self.plot_symmetries(ax=ax, x=x, y=y, z=z, hlim=hlim, vlim=vlim)

        if property == "source":
            self._add_source_cbar(ax=ax, property=property)
        return ax

    @equal_aspect
    @add_ax_if_none
    def plot_boundaries(
        self,
        x: float | None = None,
        y: float | None = None,
        z: float | None = None,
        property: str = "heat_conductivity",
        ax: Ax = None,
    ) -> Ax:
        """Plot each of simulation's boundary conditions on a plane defined by one nonzero x,y,z
        coordinate.

        Parameters
        ----------
        x : float = None
            position of plane in x direction, only one of x, y, z must be specified to define plane.
        y : float = None
            position of plane in y direction, only one of x, y, z must be specified to define plane.
        z : float = None
            position of plane in z direction, only one of x, y, z must be specified to define plane.
        property : str = None
            Specified the type of simulation for which the plot will be tailored.
            Options are ["heat_conductivity", "electric_conductivity"]
        ax : matplotlib.axes._subplots.Axes = None
            Matplotlib axes to plot on, if not specified, one is created.

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.
        """

        # get structure list
        structures = [self.simulation_structure]
        structures += list(self.scene.sorted_structures)

        # construct slicing plane
        axis, position = Box.parse_xyz_kwargs(x=x, y=y, z=z)
        center = Box.unpop_axis(position, (0, 0), axis=axis)
        size = Box.unpop_axis(0, (inf, inf), axis=axis)
        plane = Box(center=center, size=size)

        # get boundary conditions in the plane
        boundaries = self._construct_heat_charge_boundaries(
            structures=structures,
            plane=plane,
            boundary_spec=self.boundary_spec,
        )

        # plot boundary conditions
        if property == "heat_conductivity" or property == "source":
            new_boundaries = [(b, s) for b, s in boundaries if isinstance(b.condition, HeatBCTypes)]
        elif property == "electric_conductivity":
            new_boundaries = [
                (b, s) for b, s in boundaries if isinstance(b.condition, ElectricBCTypes)
            ]

        for bc_spec, shape in new_boundaries:
            ax = self._plot_boundary_condition(shape=shape, boundary_spec=bc_spec, ax=ax)

        # clean up the axis display
        ax = self.add_ax_lims(axis=axis, ax=ax)
        ax = Scene._set_plot_bounds(bounds=self.simulation_bounds, ax=ax, x=x, y=y, z=z)
        # Add the default axis labels, tick labels, and title
        ax = Box.add_ax_labels_and_title(
            ax=ax, x=x, y=y, z=z, plot_length_units=self.plot_length_units
        )

        return ax

    def _get_bc_plot_params(self, boundary_spec: HeatChargeBoundarySpec) -> PlotParams:
        """Constructs the plot parameters for given boundary conditions."""

        plot_params = plot_params_heat_bc
        condition = boundary_spec.condition

        if isinstance(condition, (TemperatureBC, VoltageBC)):
            plot_params = plot_params.updated_copy(facecolor=HEAT_BC_COLOR_TEMPERATURE)
        elif isinstance(condition, (HeatFluxBC, CurrentBC, ThermalContactResistance)):
            plot_params = plot_params.updated_copy(facecolor=HEAT_BC_COLOR_FLUX)
        elif isinstance(condition, (ConvectionBC, RadiationBC)):
            plot_params = plot_params.updated_copy(facecolor=HEAT_BC_COLOR_CONVECTION)
        elif isinstance(condition, InsulatingBC):
            plot_params = plot_params.updated_copy(facecolor=CHARGE_BC_INSULATOR)

        return plot_params

    def _plot_boundary_condition(
        self, shape: Shapely, boundary_spec: HeatChargeBoundarySpec, ax: Ax
    ) -> Ax:
        """Plot a structure's cross section shape for a given boundary condition."""
        plot_params_bc = self._get_bc_plot_params(boundary_spec=boundary_spec)
        ax = self.plot_shape(shape=shape, plot_params=plot_params_bc, ax=ax)
        return ax

    @staticmethod
    def _structure_to_bc_spec_map(
        plane: Box,
        structures: tuple[Structure, ...],
        boundary_spec: tuple[HeatChargeBoundarySpec, ...],
    ) -> dict[str, HeatChargeBoundarySpec]:
        """Construct structure name to bc spec inverse mapping. One structure may correspond to
        multiple boundary conditions."""

        named_structures_present = {structure.name for structure in structures if structure.name}

        struct_to_bc_spec = {}
        for bc_spec in boundary_spec:
            bc_place = bc_spec.placement
            if (
                isinstance(bc_place, (StructureBoundary, StructureSimulationBoundary))
                and bc_place.structure in named_structures_present
            ):
                if bc_place.structure in struct_to_bc_spec:
                    struct_to_bc_spec[bc_place.structure] += [bc_spec]
                else:
                    struct_to_bc_spec[bc_place.structure] = [bc_spec]

            if isinstance(bc_place, StructureStructureInterface):
                for structure in bc_place.structures:
                    if structure in named_structures_present:
                        if structure in struct_to_bc_spec:
                            struct_to_bc_spec[structure] += [bc_spec]
                        else:
                            struct_to_bc_spec[structure] = [bc_spec]

            if isinstance(bc_place, SimulationBoundary):
                struct_to_bc_spec[HEAT_CHARGE_BACK_STRUCTURE_STR] = [bc_spec]

        return struct_to_bc_spec

    @staticmethod
    def _medium_to_bc_spec_map(
        plane: Box,
        structures: tuple[Structure, ...],
        boundary_spec: tuple[HeatChargeBoundarySpec, ...],
    ) -> dict[str, HeatChargeBoundarySpec]:
        """Construct medium name to bc spec inverse mapping. One medium may correspond to
        multiple boundary conditions."""

        named_mediums_present = {
            structure.medium.name for structure in structures if structure.medium.name
        }

        med_to_bc_spec = {}
        for bc_spec in boundary_spec:
            bc_place = bc_spec.placement
            if isinstance(bc_place, MediumMediumInterface):
                for med in bc_place.mediums:
                    if med in named_mediums_present:
                        if med in med_to_bc_spec:
                            med_to_bc_spec[med] += [bc_spec]
                        else:
                            med_to_bc_spec[med] = [bc_spec]

        return med_to_bc_spec

    @staticmethod
    def _construct_forward_boundaries(
        shapes: tuple[tuple[str, str, Shapely, tuple[float, float, float, float]], ...],
        struct_to_bc_spec: dict[str, HeatChargeBoundarySpec],
        med_to_bc_spec: dict[str, HeatChargeBoundarySpec],
        background_structure_shape: Shapely,
    ) -> tuple[tuple[HeatChargeBoundarySpec, Shapely], ...]:
        """Construct Simulation, StructureSimulation, Structure, and MediumMedium boundaries."""

        # forward loop to take care of Simulation, StructureSimulation, Structure,
        # and MediumMediums
        boundaries = []  # bc_spec, structure name, shape, bounds
        background_shapes = []
        for name, medium, shape, bounds in shapes:
            # intersect existing boundaries (both structure based and medium based)
            for index, (_bc_spec, _name, _bdry, _bounds) in enumerate(boundaries):
                # simulation bc is overridden only by StructureSimulationBoundary
                if isinstance(_bc_spec.placement, SimulationBoundary):
                    if name not in struct_to_bc_spec:
                        continue
                    if any(
                        not isinstance(bc_spec.placement, StructureSimulationBoundary)
                        for bc_spec in struct_to_bc_spec[name]
                    ):
                        continue

                if Box._do_not_intersect(bounds, _bounds, shape, _bdry):
                    continue

                diff_shape = _bdry - shape

                boundaries[index] = (_bc_spec, _name, diff_shape, diff_shape.bounds)

            # create new structure based boundary

            if name in struct_to_bc_spec:
                for bc_spec in struct_to_bc_spec[name]:
                    if isinstance(bc_spec.placement, StructureBoundary):
                        bdry = shape.exterior
                        bdry = bdry.intersection(background_structure_shape)
                        boundaries.append((bc_spec, name, bdry, bdry.bounds))

                    if isinstance(bc_spec.placement, SimulationBoundary):
                        boundaries.append((bc_spec, name, shape.exterior, shape.exterior.bounds))

                    if isinstance(bc_spec.placement, StructureSimulationBoundary):
                        bdry = background_structure_shape.exterior
                        bdry = bdry.intersection(shape)
                        boundaries.append((bc_spec, name, bdry, bdry.bounds))

            # create new medium based boundary, and cut or merge relevant background shapes

            # loop through background_shapes (note: all background are non-intersecting or merged)
            # this is similar to _filter_structures_plane but only mediums participating in BCs
            # are tracked
            # new interfaces lie on this structure's own shape; `shape` grows below when it
            # merges with background shapes of the same medium, whose interfaces already exist
            structure_shape = shape
            for index, (_medium, _shape, _bounds) in enumerate(background_shapes):
                if Box._do_not_intersect(bounds, _bounds, shape, _shape):
                    continue

                diff_shape = _shape - shape

                # different medium, remove intersection from background shape
                if medium != _medium and len(diff_shape.bounds) > 0:
                    background_shapes[index] = (_medium, diff_shape, diff_shape.bounds)

                    # in case when there is a bc between two media
                    # create a new boundary segment
                    for bc_spec in med_to_bc_spec[_medium.name]:
                        if medium.name in bc_spec.placement.mediums:
                            bdry = structure_shape.boundary.intersection(_shape)
                            # drop edges where the structure covers _shape's own boundary,
                            # since _shape no longer touches them (e.g. a layer whose bottom
                            # face lies on an earlier substrate that already cut _shape)
                            covered = _shape.boundary.intersection(structure_shape) - diff_shape
                            bdry = bdry - covered
                            bdry = bdry.intersection(background_structure_shape)
                            boundaries.append((bc_spec, name, bdry, bdry.bounds))

                # same medium, add diff shape to this shape and mark background shape for removal
                # note: this only happens if this medium is listed in BCs
                else:
                    shape = shape | diff_shape
                    background_shapes[index] = None

            # after doing this with all background shapes, add this shape to the background
            # but only if this medium is listed in BCs
            if medium.name in med_to_bc_spec:
                background_shapes.append((medium, shape, shape.bounds))

            # remove any existing background shapes that have been marked as 'None'
            background_shapes = [b for b in background_shapes if b is not None]

        # filter out empty geometries
        boundaries = [(bc_spec, bdry) for (bc_spec, name, bdry, _) in boundaries if bdry]

        return boundaries

    @staticmethod
    def _construct_reverse_boundaries(
        shapes: tuple[tuple[str, str, Shapely, Bound], ...],
        struct_to_bc_spec: dict[str, HeatChargeBoundarySpec],
        background_structure_shape: Shapely,
    ) -> tuple[tuple[HeatChargeBoundarySpec, Shapely], ...]:
        """Construct StructureStructure boundaries."""

        # backward loop to take care of StructureStructure
        # we do it in this way because we define the boundary between
        # two overlapping structures A and B, where A comes before B, as
        # boundary(B) intersected by A
        # So, in this loop as we go backwards through the structures we:
        # - (1) when come upon B, create boundary(B)
        # - (2) cut away from it by other structures
        # - (3) when come upon A, intersect it with A and mark it as complete,
        #   that is, no more further modifications
        boundaries_reverse = []

        for name, _, shape, bounds in shapes[:0:-1]:
            _minx, _miny, _maxx, _maxy = bounds

            # intersect existing boundaries
            for index, (_bc_spec, _name, _bdry, _bounds, _completed) in enumerate(
                boundaries_reverse
            ):
                if not _completed:
                    if Box._do_not_intersect(bounds, _bounds, shape, _bdry):
                        continue

                    # event (3) from above
                    if name in _bc_spec.placement.structures:
                        new_bdry = _bdry.intersection(shape)
                        boundaries_reverse[index] = (
                            _bc_spec,
                            _name,
                            new_bdry,
                            new_bdry.bounds,
                            True,
                        )

                    # event (2) from above
                    else:
                        new_bdry = _bdry - shape
                        boundaries_reverse[index] = (
                            _bc_spec,
                            _name,
                            new_bdry,
                            new_bdry.bounds,
                            _completed,
                        )

            # create new boundary (event (1) from above)
            if name in struct_to_bc_spec:
                for bc_spec in struct_to_bc_spec[name]:
                    if isinstance(bc_spec.placement, StructureStructureInterface):
                        bdry = shape.exterior
                        bdry = bdry.intersection(background_structure_shape)
                        boundaries_reverse.append((bc_spec, name, bdry, bdry.bounds, False))

        # filter and append completed boundaries to main list
        filtered_boundaries = []
        for bc_spec, _, bdry, _, is_completed in boundaries_reverse:
            if bdry and is_completed:
                filtered_boundaries.append((bc_spec, bdry))

        return filtered_boundaries

    @staticmethod
    def _construct_heat_charge_boundaries(
        structures: list[Structure],
        plane: Box,
        boundary_spec: list[HeatChargeBoundarySpec],
    ) -> list[tuple[HeatChargeBoundarySpec, Shapely]]:
        """Compute list of boundary lines to plot on plane.

        Parameters
        ----------
        structures : list[:class:`.Structure`]
            list of structures to filter on the plane.
        plane : :class:`.Box`
            target plane.
        boundary_spec : list[HeatBoundarySpec]
            list of boundary conditions associated with structures.

        Returns
        -------
        list[tuple[:class:`.HeatBoundarySpec`, shapely.geometry.base.BaseGeometry]]
            List of boundary lines and boundary conditions on the plane after merging.
        """

        # get structures in the plane and present named structures and media
        shapes = []  # structure name, structure medium, shape, bounds
        for structure in structures:
            # get list of Shapely shapes that intersect at the plane
            shapes_plane = plane.intersections_with(structure.geometry)

            # append each of them and their medium information to the list of shapes
            for shape in shapes_plane:
                shapes.append((structure.name, structure.medium, shape, shape.bounds))

        background_structure_shape = shapes[0][2]

        # construct an inverse mapping structure -> bc for present structures
        struct_to_bc_spec = HeatChargeSimulation._structure_to_bc_spec_map(
            plane=plane, structures=structures, boundary_spec=boundary_spec
        )

        # construct an inverse mapping medium -> bc for present mediums
        med_to_bc_spec = HeatChargeSimulation._medium_to_bc_spec_map(
            plane=plane, structures=structures, boundary_spec=boundary_spec
        )

        # construct boundaries in 2 passes:

        # 1. forward loop to take care of Simulation, StructureSimulation, Structure,
        # and MediumMediums
        boundaries = HeatChargeSimulation._construct_forward_boundaries(
            shapes=shapes,
            struct_to_bc_spec=struct_to_bc_spec,
            med_to_bc_spec=med_to_bc_spec,
            background_structure_shape=background_structure_shape,
        )

        # 2. reverse loop: construct structure-structure boundary
        struct_struct_boundaries = HeatChargeSimulation._construct_reverse_boundaries(
            shapes=shapes,
            struct_to_bc_spec=struct_to_bc_spec,
            background_structure_shape=background_structure_shape,
        )

        return boundaries + struct_struct_boundaries

    @equal_aspect
    @add_ax_if_none
    def plot_sources(
        self,
        x: float | None = None,
        y: float | None = None,
        z: float | None = None,
        property: str = "heat_conductivity",
        hlim: tuple[float, float] | None = None,
        vlim: tuple[float, float] | None = None,
        alpha: float | None = None,
        ax: Ax = None,
    ) -> Ax:
        """Plot each of simulation's sources on a plane defined by one nonzero x,y,z coordinate.

        Parameters
        ----------
        x : float = None
            position of plane in x direction, only one of x, y, z must be specified to define plane.
        y : float = None
            position of plane in y direction, only one of x, y, z must be specified to define plane.
        z : float = None
            position of plane in z direction, only one of x, y, z must be specified to define plane.
        property : str = None
            Specified the type of simulation for which the plot will be tailored.
            Options are ["heat_conductivity", "electric_conductivity"]
        hlim : tuple[float, float] = None
            The x range if plotting on xy or xz planes, y range if plotting on yz plane.
        vlim : tuple[float, float] = None
            The z range if plotting on xz or yz planes, y plane if plotting on xy plane.
        alpha : float = None
            Opacity of the sources, If ``None`` uses Tidy3d default.
        ax : matplotlib.axes._subplots.Axes = None
            Matplotlib axes to plot on, if not specified, one is created.

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.
        """

        # background can't have source, so no need to add background structure
        structures = self.scene.sorted_structures

        # alpha is None just means plot without any transparency
        if alpha is None:
            alpha = 1

        if alpha <= 0:
            return ax

        # get appropriate sources
        if property == "heat_conductivity" or property == "source":
            source_list = [s for s in self.sources if isinstance(s, HeatSourceTypes)]
        elif property == "electric_conductivity":
            source_list = [s for s in self.sources if isinstance(s, ChargeSourceTypes)]

        # distribute source where there are assigned
        structure_source_map = {}
        for source in source_list:
            if not isinstance(source, GlobalHeatChargeSource):
                for name in source.structures:
                    structure_source_map[name] = source

        source_list = [structure_source_map.get(structure.name, None) for structure in structures]

        axis, position = Box.parse_xyz_kwargs(x=x, y=y, z=z)
        center = Box.unpop_axis(position, (0, 0), axis=axis)
        size = Box.unpop_axis(0, (inf, inf), axis=axis)
        plane = Box(center=center, size=size)

        source_shapes = self.scene._filter_structures_plane(
            structures=structures,
            plane=plane,
            property_list=source_list,
            section_tolerance_2d=True,
        )

        source_min, source_max = self.source_bounds(property=property)
        for source, shape in source_shapes:
            if source is not None:
                ax = self._plot_shape_structure_source(
                    alpha=alpha,
                    source=source,
                    source_min=source_min,
                    source_max=source_max,
                    shape=shape,
                    ax=ax,
                )

        # clean up the axis display
        ax = self.add_ax_lims(axis=axis, ax=ax)
        ax = Scene._set_plot_bounds(bounds=self.simulation_bounds, ax=ax, x=x, y=y, z=z)
        # Add the default axis labels, tick labels, and title
        ax = Box.add_ax_labels_and_title(
            ax=ax, x=x, y=y, z=z, plot_length_units=self.plot_length_units
        )
        return ax

    def _add_source_cbar(self, ax: Ax, property: str = "heat_conductivity") -> None:
        """Add colorbar for heat sources."""
        source_min, source_max = self.source_bounds(property=property)
        self.scene._add_cbar(
            vmin=source_min,
            vmax=source_max,
            label=f"Volumetric heat rate ({VOLUMETRIC_HEAT_RATE})",
            cmap=HEAT_SOURCE_CMAP,
            ax=ax,
        )

    def source_bounds(self, property: str = "heat_conductivity") -> tuple[float, float]:
        """Compute range of heat sources present in the simulation."""
        if property == "heat_conductivity" or property == "source":
            rate_list = [
                np.mean(source.rate) for source in self.sources if isinstance(source, HeatSource)
            ]
        elif property == "electric_conductivity":
            rate_list = [
                source.rate for source in self.sources if isinstance(source, ChargeSourceTypes)
            ]  # this is currently an empty list

        rate_list.append(0)
        rate_min = min(rate_list)
        rate_max = max(rate_list)
        return rate_min, rate_max

    def _get_structure_source_plot_params(
        self,
        source: HeatChargeSourceType,
        source_min: float,
        source_max: float,
        alpha: float | None = None,
    ) -> PlotParams:
        """Constructs the plot parameters for a given medium in simulation.plot_eps()."""

        plot_params = plot_params_heat_source
        if alpha is not None:
            plot_params = plot_params.copy(update={"alpha": alpha})

        if isinstance(source, HeatSource):
            rate = np.mean(source.rate)
            if rate is not None:
                from matplotlib import colormaps

                delta_rate = rate - source_min
                delta_rate_max = source_max - source_min + 1e-5
                rate_fraction = delta_rate / delta_rate_max
                cmap = colormaps[HEAT_SOURCE_CMAP]
                rgba = cmap(rate_fraction)
                plot_params = plot_params.copy(update={"edgecolor": rgba})

        return plot_params

    def _plot_shape_structure_source(
        self,
        source: HeatChargeSourceType,
        shape: Shapely,
        source_min: float,
        source_max: float,
        ax: Ax,
        alpha: float | None = None,
    ) -> Ax:
        """Plot a structure's cross section shape for a given medium, grayscale for permittivity."""
        plot_params = self._get_structure_source_plot_params(
            source=source,
            source_min=source_min,
            source_max=source_max,
            alpha=alpha,
        )
        ax = self.plot_shape(shape=shape, plot_params=plot_params, ax=ax)
        return ax

    @classmethod
    def from_scene(cls, scene: Scene, **kwargs: Any) -> HeatChargeSimulation:
        """Create a simulation from a :class:`.Scene` instance. Must provide additional parameters
        to define a valid simulation (for example, ``size``, ``grid_spec``, etc).

        Parameters
        ----------
        scene : :class:`.Scene`
            Scene containing structures information.
        **kwargs
            Other arguments

        Example
        -------
        >>> from tidy3d import Scene, Medium, Box, Structure, UniformUnstructuredGrid, TemperatureMonitor
        >>> box = Structure(
        ...     geometry=Box(center=(0, 0, 0), size=(1, 2, 3)),
        ...     medium=Medium(permittivity=5),
        ...     name="box"
        ... )
        >>> scene = Scene(
        ...     structures=[box],
        ...     medium=Medium(
        ...         permittivity=3,
        ...         heat_spec=SolidMedium(
        ...             conductivity=1, capacity=1,
        ...         ),
        ...     ),
        ... )
        >>> sim = HeatChargeSimulation.from_scene(
        ...     scene=scene,
        ...     center=(0, 0, 0),
        ...     size=(5, 6, 7),
        ...     grid_spec=UniformUnstructuredGrid(
        ...         dl=0.4, min_edges_per_circumference=15, min_edges_per_side=2
        ...     ),
        ...     boundary_spec=[
        ...         HeatChargeBoundarySpec(
        ...             placement=StructureBoundary(structure="box"),
        ...             condition=TemperatureBC(temperature=500),
        ...         )
        ...     ],
        ...     monitors=[TemperatureMonitor(name="temp_monitor", center=(0, 0, 0), size=(1, 1, 1), unstructured=True)],
        ... )
        """

        return cls(
            structures=scene.structures,
            medium=scene.medium,
            **kwargs,
        )

    def _get_simulation_types(self) -> list[TCADAnalysisTypes]:
        """
        Checks through BCs and sources and returns the
        types of simulations.
        """
        simulation_types = []

        if isinstance(self.analysis_spec, ChargeTypes):
            return [TCADAnalysisTypes.CHARGE]

        # check if unsteady heat
        if isinstance(self.analysis_spec, UnsteadyHeatAnalysis):
            return [TCADAnalysisTypes.HEAT]

        heat_source_present = any(isinstance(s, HeatSourceTypes) for s in self.sources)

        heat_BCs_present = any(isinstance(bc.condition, HeatBCTypes) for bc in self.boundary_spec)

        if heat_source_present and not heat_BCs_present:
            raise SetupError("Heat sources defined but no heat BCs present.")
        if heat_BCs_present or heat_source_present:
            simulation_types.append(TCADAnalysisTypes.HEAT)

        # check for conduction simulation
        electric_spec_present = any(
            structure.medium.charge is not None for structure in self.structures
        )

        electric_BCs_present = any(
            isinstance(bc.condition, ElectricBCTypes) for bc in self.boundary_spec
        )

        if electric_BCs_present and not electric_spec_present:
            raise SetupError(
                "Electric BC were specified but no structure in the simulation has "
                "a defined '.medium.charge'. Structures with "
                "'.medium.charge=None' are treated as insulators, thus, "
                "the solution domain is empty."
            )
        if electric_BCs_present and electric_spec_present:
            simulation_types.append(TCADAnalysisTypes.CONDUCTION)

        return simulation_types

    @property
    def _dc_voltages(self) -> list[float]:
        """DC bias voltages the charge solver computes steady-state solutions at.

        The sweep array of a ``VoltageBC`` if present (validation permits at most
        one), else the single requested bias, else empty. ``SSACVoltageSource``
        carries DC operating points and sweeps the same way.
        """
        voltages: list[float] = []
        for bc in self.boundary_spec:
            if isinstance(bc.condition, VoltageBC) and isinstance(
                bc.condition.source, (DCVoltageSource, SSACVoltageSource)
            ):
                if len(bc.condition.source.voltage) > len(voltages):
                    voltages = [float(v) for v in bc.condition.source.voltage]
        return voltages

    @property
    def _dc_currents(self) -> list[float]:
        """Currents of the swept ``CurrentBC`` (validation permits one sweep), else empty."""
        for index in self._swept_bc_indices():
            if self._is_current_sweep(index):
                current = self.boundary_spec[index].condition.source.current
                return [float(c) for c in np.atleast_1d(current)]
        return []

    @property
    def _num_dc_solves(self) -> int:
        """Number of nonlinear solves the charge solver runs to cover the DC sweep.

        Mirrors the solver's sweep construction: a single requested voltage is
        one direct solve; otherwise the sweep starts at 0 V and covers positive
        then negative voltages by increasing magnitude, inserting a warm-start
        solve per ``convergence_dv`` interval within each pass. A current sweep is
        ordered the same way but inserts no warm-start solves. Doping ramp-up
        adds ``tolerance_settings.ramp_up_iters - 1`` solves. With impact
        ionization, or with a quasi-Fermi-gradient driving force in a field
        dependence, every bias point is solved twice: first with the
        ionization source frozen and the quasi-Fermi-driven carriers at their
        low-field mobility, then with both active. Ramp levels are solved once.
        Charge sims only.
        """
        voltages = self._dc_voltages

        # Doping ramp: the initial solve runs once per ramp level.
        ramp_solves = self.analysis_spec.tolerance_settings.ramp_up_iters - 1

        # Staged ionization or a quasi-Fermi driving force re-solves every bias point,
        # but not the ramp levels.
        staged = self._uses_impact_ionization() or self._uses_quasi_fermi_driving_force()
        solves_per_point = 2 if staged else 1

        currents = self._dc_currents
        if currents:
            # 1 for the initial zero-current solve, then one per nonzero current.
            bias_points = 1 + sum(1 for c in currents if c != 0)
            return solves_per_point * bias_points + ramp_solves

        if len(voltages) <= 1:
            # No sweep: one direct solve at the requested bias.
            return solves_per_point + ramp_solves

        convergence_dv = self.analysis_spec.convergence_dv
        positive = sorted(v for v in voltages if v > 1e-7)
        negative = sorted((v for v in voltages if v < -1e-7), key=abs)

        def pass_solves(pass_voltages: list[float]) -> int:
            n = 0
            prev = 0.0
            for v in pass_voltages:
                gap = abs(v - prev)
                n += math.ceil(gap / convergence_dv) if gap > convergence_dv else 1
                prev = v
            return n

        # 1 for the initial 0 V solve; each pass warm-starts from the 0 V solution.
        bias_points = 1 + pass_solves(positive) + pass_solves(negative)
        return solves_per_point * bias_points + ramp_solves

    @property
    def _thermal_solver_active(self) -> bool:
        """Whether a coupled thermal solve runs alongside the charge analysis.

        One test suffices -- a thermal solve runs exactly when the lattice temperature is
        unknown -- because neither way of supplying it leaves the field unset: the
        isothermal spec narrows ``temperature`` to a scalar defaulting to 300 K, and a
        prescribed field is frozen rather than solved for. The AC specs inherit from both,
        so they classify with no extra cases.

        Gates the ``residual_temperature`` report, the heat-solver feature checks, and the
        coupling-source checks.
        """
        return (
            isinstance(self.analysis_spec, SteadyChargeDCAnalysis)
            and self.analysis_spec.temperature is None
        )

    @property
    def _temperature_prescribed(self) -> bool:
        """Whether the charge analysis runs on a fixed, spatially varying lattice temperature.

        Distinguishes a prescribed-T run from a uniform isothermal one: both skip the
        thermal solve, but only this one ships a field to the solver.
        """
        return (
            isinstance(self.analysis_spec, SteadyChargeDCAnalysis)
            and not isinstance(self.analysis_spec, IsothermalSteadyChargeDCAnalysis)
            and self.analysis_spec.temperature is not None
        )

    def _accelerated_only_features(self) -> list[str]:
        """Configured features that only the accelerated charge solver supports.

        These cannot run on the CPU charge solver, so requesting
        ``use_accelerated_solver=False`` while any of them is present is an error.
        Returns the human-readable feature names; an empty list means the
        configuration is fully supported by the CPU charge solver.
        """
        features = []
        if any(
            isinstance(charge.mobility_n, MasettiMobility)
            or isinstance(charge.mobility_p, MasettiMobility)
            for _loc, charge in self._iter_semiconductor_charge_media()
        ):
            features.append("MasettiMobility")
        if any(
            isinstance(charge.mobility_n, AlloyMobility)
            or isinstance(charge.mobility_p, AlloyMobility)
            for _loc, charge in self._iter_semiconductor_charge_media()
        ):
            features.append("AlloyMobility")
        if any(
            charge.donor_ionization is not None or charge.acceptor_ionization is not None
            for _loc, charge in self._iter_semiconductor_charge_media()
        ):
            features.append("Incomplete ionization ('donor_ionization', 'acceptor_ionization')")
        field_dependence_names = {
            type(mobility.field_dependence).__name__
            for _loc, charge in self._iter_semiconductor_charge_media()
            for mobility in (charge.mobility_n, charge.mobility_p)
            if mobility.field_dependence is not None
        }
        features.extend(sorted(field_dependence_names))
        if any(
            self._surface_dependence(charge.mobility_n) is not None
            or self._surface_dependence(charge.mobility_p) is not None
            for _loc, charge in self._iter_semiconductor_charge_media()
        ):
            features.append("LombardiSurfaceDependence")

        if self._insulator_face_bc_indices():
            features.append("'VoltageBC' on an insulator face")

        if self._driven_terminal_split_across_entries():
            features.append("Swept or SSAC terminal named by several 'boundary_spec' entries")

        if self._anisotropic_permittivity_locs():
            features.append("AnisotropicPermittivity")

        if self._uses_gpu_only_lifetime_model():
            features.append("PalankovskiQuayApproxCarrierLifetime")

        if self._uses_grad_quasi_fermi_impact_ionization():
            features.append("GradQuasiFermi impact ionization")

        if self._uses_trap_assisted_tunneling():
            features.append("HurkxTrapAssistedTunneling")

        if self._ssac_uses_bias_point_selection():
            features.append("SSAC 'at_voltages' bias-point selection")

        if self.linear_solver == "direct":
            features.append("linear_solver='direct'")

        if self._thermal_solver_active and self._uses_temperature_dependent_thermal_properties():
            features.append("Temperature-dependent thermal properties in a non-isothermal run")

        if self._thermal_solver_active:
            features.extend(self._accelerated_only_heat_features())

        return features

    def _accelerated_only_heat_features(self) -> list[str]:
        """Heat features that only the accelerated solver carries into the thermal solve of a
        non-isothermal charge simulation."""
        heat_specs = [
            medium if isinstance(medium, SolidMedium) else getattr(medium, "heat_spec", None)
            for _loc, medium in self._iter_media_with_locs()
        ]
        conditions = [bc.condition for bc in self.boundary_spec]
        features = []
        if any(
            getattr(spec, "velocity", None) is not None and any(v != 0.0 for v in spec.velocity)
            for spec in heat_specs
        ):
            features.append(
                "Solid-medium advection ('SolidMedium.velocity') in a non-isothermal run"
            )
        if any(
            isinstance(spec, SolidMedium)
            and isinstance(spec._reference_conductivity, AnisotropicConductivity)
            for spec in heat_specs
        ):
            features.append("'AnisotropicConductivity' in a non-isothermal run")
        if any(isinstance(condition, ThermalContactResistance) for condition in conditions):
            features.append("'ThermalContactResistance' in a non-isothermal run")
        if any(
            isinstance(condition, RadiationBC)
            or (isinstance(condition, ConvectionBC) and condition.emissivity)
            for condition in conditions
        ):
            features.append(
                "Surface radiation ('RadiationBC', or 'ConvectionBC' with a positive "
                "'emissivity') in a non-isothermal run"
            )
        return features

    def _uses_temperature_dependent_thermal_properties(self) -> bool:
        """Whether any solid heat spec has a temperature-dependent conductivity or capacity."""
        media = [self.medium] + [struct.medium for struct in self.structures]
        for medium in media:
            heat_spec = (
                medium if isinstance(medium, SolidMedium) else getattr(medium, "heat_spec", None)
            )
            if isinstance(heat_spec, SolidMedium) and (
                isinstance(heat_spec.conductivity, AbstractConductivityModel)
                or isinstance(heat_spec.capacity, AbstractHeatCapacityModel)
            ):
                return True
        return False

    def _warn_heterojunction_electron_affinity(self) -> Self:
        """Warn if an unset electron affinity would decide a band offset.

        The accelerated solver places the bands of two semiconductors by their electron
        affinities (Anderson's rule) and reads an unset one as 0 eV. With one of the pair
        unset, that 0 eV meets the other's affinity. With both unset, a band-gap difference
        lands entirely in the valence band; equal band gaps lose nothing, whatever their
        densities of states. The non-accelerated solver ignores 'electron_affinity'
        (``_warn_non_accelerated_ignores_electron_affinity``).
        """
        try:
            if not self._resolve_use_accelerated_solver:
                return self
        except SetupError:
            return self

        semiconductors = self._distinct_semiconductor_charge_media()
        for index, (loc_a, medium_a) in enumerate(semiconductors):
            for loc_b, medium_b in semiconductors[index + 1 :]:
                missing = [
                    loc
                    for loc, medium in ((loc_a, medium_a), (loc_b, medium_b))
                    if medium.electron_affinity is None
                ]
                pair = f"{self._medium_label(loc_a)} and {self._medium_label(loc_b)}"
                if len(missing) == 1 and _is_material_heterojunction(medium_a, medium_b):
                    message = (
                        f"The band offset between {pair} follows from their electron "
                        f"affinities, and {self._medium_label(missing[0])} has no "
                        "'electron_affinity', which is read as 0 eV against the other's. "
                        "Set it. Future versions of Tidy3D may reject this setup."
                    )
                elif len(missing) == 2 and _band_fields_differ(
                    _band_gap_signature(medium_a), _band_gap_signature(medium_b)
                ):
                    message = (
                        f"{pair} have different band gaps, so their band offset follows from "
                        "their electron affinities. Neither sets 'electron_affinity': both "
                        "are read as 0 eV, which puts the whole band-gap difference into "
                        "the valence band. Set 'electron_affinity' on both. "
                        "Future versions of Tidy3D may reject this setup."
                    )
                else:
                    continue
                log.warning(message)
        return self

    def _warn_same_bands_different_narrowing(self) -> Self:
        """Warn if two semiconductors without a band offset differ in band-gap narrowing.

        Where two such media meet, the accelerated solver keeps the carrier densities
        continuous, which holds only when both sides narrow the band gap alike. Only media
        that may meet are warned about.
        """
        try:
            if not self._resolve_use_accelerated_solver:
                return self
        except SetupError:
            return self

        semiconductors = self._distinct_semiconductor_charge_media()
        for index, (loc_a, medium_a) in enumerate(semiconductors):
            for loc_b, medium_b in semiconductors[index + 1 :]:
                if (
                    medium_a.delta_E_g == medium_b.delta_E_g
                    or _is_material_heterojunction(medium_a, medium_b)
                    or not self._semiconductor_charge_media_may_meet(medium_a, medium_b)
                ):
                    continue
                log.warning(
                    f"{self._medium_label(loc_a)} and {self._medium_label(loc_b)} have the "
                    "same 'N_c', 'N_v', 'E_g' and 'electron_affinity' but different 'delta_E_g'. "
                    "Where they meet, the accelerated charge solver keeps the carrier densities "
                    "continuous, so the difference in band-gap narrowing acts as a voltage source "
                    "there. Use the same 'delta_E_g' on both. "
                    "Future versions of Tidy3D may reject this setup."
                )
        return self

    def _distinct_semiconductor_charge_media(
        self,
    ) -> list[tuple[tuple[Any, ...], SemiconductorMedium]]:
        """``(loc, charge)`` of each distinct ``SemiconductorMedium``, first occurrence kept."""
        semiconductors: list[tuple[tuple[Any, ...], SemiconductorMedium]] = []
        for loc, medium in self._iter_semiconductor_charge_media():
            if not any(medium is other or medium == other for _, other in semiconductors):
                semiconductors.append((loc, medium))
        return semiconductors

    def _semiconductor_charge_media_may_meet(
        self, medium_a: SemiconductorMedium, medium_b: SemiconductorMedium
    ) -> bool:
        """Whether a structure carrying ``medium_a`` and one carrying ``medium_b`` may touch.

        The background meets every structure. Bounding boxes decide, so non-box shapes may
        over-report, but a real contact is never missed.
        """
        locs_a, locs_b = [], []
        for loc, medium in self._iter_semiconductor_charge_media():
            if medium is medium_a or medium == medium_a:
                locs_a.append(loc)
            if medium is medium_b or medium == medium_b:
                locs_b.append(loc)
        if ("medium",) in locs_a or ("medium",) in locs_b:
            return True
        return any(
            self.structures[i].geometry.intersects(self.structures[j].geometry)
            for _, i in locs_a
            for _, j in locs_b
        )

    def _medium_label(self, loc: tuple[Any, ...]) -> str:
        """How messages name the medium at ``loc``: the background or a structure."""
        if loc == ("medium",):
            return "The background 'medium'"
        return f"Structure '{self.structures[loc[1]].name or loc[1]}'"

    def _iter_media_with_locs(self) -> Iterator[tuple[tuple[Any, ...], Any]]:
        """Yield ``(loc, medium)`` for the background ``medium`` and each structure's medium."""
        yield ("medium",), self.medium
        for i, structure in enumerate(self.structures):
            yield ("structures", i), structure.medium

    def _iter_charge_media(self) -> Iterator[tuple[tuple[Any, ...], AbstractChargeMedium]]:
        """Yield ``(loc, charge)`` for every charge medium in the simulation.

        Walks both the background ``self.medium`` and each ``Structure.medium``,
        which the mesher composes together into ``simulation_structure``. Each
        slot may be a raw charge medium or a :class:`MultiPhysicsMedium` that
        carries one as ``.charge``; slots with neither are skipped. ``loc`` is the
        Pydantic field path of the source — ``("medium",)`` for the background,
        ``("structures", i)`` for a structure — so validators can raise
        loc-aware errors that point at the right field. Centralising both the
        traversal and the loc avoids the bugs where iterating only over
        ``.medium.charge`` (or only over ``structures``) silently misses
        raw-medium or background-medium configurations.
        """
        for loc, medium in self._iter_media_with_locs():
            charge = self._charge_spec(medium)
            if isinstance(charge, AbstractChargeMedium):
                yield loc, charge

    def _iter_semiconductor_charge_media(
        self,
    ) -> Iterator[tuple[tuple[Any, ...], SemiconductorMedium]]:
        """Yield ``(loc, charge)`` for every ``SemiconductorMedium`` in the simulation; see
        :meth:`_iter_charge_media`."""
        for loc, charge in self._iter_charge_media():
            if isinstance(charge, SemiconductorMedium):
                yield loc, charge

    def _anisotropic_permittivity_locs(self) -> list[tuple[Any, ...]]:
        """Field paths of the media whose charge medium has an ``AnisotropicPermittivity``."""
        return [
            loc
            for loc, charge in self._iter_charge_media()
            if isinstance(charge.permittivity, AnisotropicPermittivity)
        ]

    def _symmetry_breaking_permittivity(self) -> list[tuple[tuple[Any, ...], int]]:
        """``(loc, axis)`` for each charge medium whose ``AnisotropicPermittivity`` is not
        mirror-symmetric across the symmetry plane normal to ``axis``.

        Reflecting ``axis`` flips the sign of the off-diagonal components that couple it to the
        other axes, so with those nonzero the symmetry plane acts as an extra insulating wall.
        A component that couples to the zero-size axis of a 2D simulation does not enter the
        solve and is ignored.
        """
        zero_dims = set(self.zero_dims)
        # A symmetry plane normal to the zero-size axis is rejected on its own.
        sym_axes = [
            axis for axis, sym in enumerate(self.symmetry) if sym != 0 and axis not in zero_dims
        ]
        if not sym_axes:
            return []
        found = []
        for loc, charge in self._iter_charge_media():
            if not isinstance(charge.permittivity, AnisotropicPermittivity):
                continue
            matrix = charge.permittivity.matrix
            tol = 1e-9 * np.max(np.diag(matrix))
            for axis in sym_axes:
                coupled = (other for other in range(3) if other != axis and other not in zero_dims)
                if any(abs(matrix[axis, other]) > tol for other in coupled):
                    found.append((loc, axis))
        return found

    def _ssac_uses_bias_point_selection(self) -> bool:
        """Whether SSAC selects specific bias points via ``at_voltages``.

        The CPU charge solver always evaluates the AC response at every swept bias
        point and cannot honor an explicit selection, so any ``at_voltages`` is only
        available on the accelerated solver.
        """
        return (
            isinstance(self.analysis_spec, (SSACAnalysis, IsothermalSSACAnalysis))
            and self.analysis_spec.at_voltages is not None
        )

    def _uses_gpu_only_lifetime_model(self) -> bool:
        """Whether the simulation uses an SRH lifetime model unavailable in the CPU charge solver."""
        for _loc, charge in self._iter_semiconductor_charge_media():
            for model in charge.R:
                if not isinstance(model, ShockleyReedHallRecombination):
                    continue
                for tau in (model.tau_n, model.tau_p):
                    if isinstance(tau, PalankovskiQuayApproxCarrierLifetime):
                        return True
        return False

    def _uses_trap_assisted_tunneling(self) -> bool:
        """Whether any bulk or surface SRH model carries a trap-assisted tunneling enhancement."""
        for _loc, charge in self._iter_semiconductor_charge_media():
            for model in charge.R:
                if (
                    isinstance(model, ShockleyReedHallRecombination)
                    and model.field_enhancement is not None
                ):
                    return True

        for bc_spec in self.boundary_spec:
            condition = bc_spec.condition
            if not isinstance(condition, SurfaceRecombinationBC):
                continue
            model = condition.model
            if (
                isinstance(model, SurfaceShockleyReedHallRecombination)
                and model.field_enhancement is not None
            ):
                return True

        return False

    def _uses_impact_ionization(self) -> bool:
        """Whether any semiconductor medium enables an impact-ionization model."""
        for _loc, charge in self._iter_semiconductor_charge_media():
            for model in charge.R:
                if isinstance(model, SelberherrImpactIonization):
                    return True
        return False

    def _uses_quasi_fermi_driving_force(self) -> bool:
        """Whether any semiconductor drives a high-field mobility on a quasi-Fermi gradient."""
        return any(
            getattr(mobility.field_dependence, "driving_force", None) == "quasi_fermi_gradient"
            for _loc, charge in self._iter_semiconductor_charge_media()
            for mobility in (charge.mobility_n, charge.mobility_p)
        )

    def _uses_grad_quasi_fermi_impact_ionization(self) -> bool:
        """Whether any semiconductor selects the GradQuasiFermi formulation."""
        return any(
            isinstance(model, SelberherrImpactIonization) and model.formulation == "GradQuasiFermi"
            for _loc, charge in self._iter_semiconductor_charge_media()
            for model in charge.R
        )

    def _check_grad_quasi_fermi_impact_ionization(self) -> Self:
        """Restrict GradQuasiFermi impact ionization to accelerated isothermal charge."""
        if not self._uses_grad_quasi_fermi_impact_ionization():
            return self

        if not isinstance(
            self.analysis_spec, (IsothermalSteadyChargeDCAnalysis, IsothermalSSACAnalysis)
        ):
            self._raise_validation_error_at_loc(
                "'GradQuasiFermi' impact ionization is supported only for isothermal "
                "charge analyses.",
                "analysis_spec",
            )

        return self

    def _analysis_temperature_extremes(self) -> tuple[float, float] | None:
        """Coldest and hottest lattice temperature the analysis prescribes, if any.

        ``None`` when the analysis has no prescribed temperature — either an
        isothermal spec that carries none, or a self-heating spec whose
        temperature is solved for. A prescribed field makes ``temperature`` a whole
        array, so both extremes are returned: which one is worst depends on the sign
        of the exponent the caller is checking. ``np.asarray`` puts a scalar, a list
        and a :class:`SpatialDataArray` on the same path.
        """
        temperature = getattr(self.analysis_spec, "temperature", None)
        if temperature is None:
            return None
        values = np.asarray(getattr(temperature, "values", temperature), dtype=float).reshape(-1)
        return float(values.min()), float(values.max())

    def _mobility_temperature_candidates(self) -> list[float]:
        """Lattice temperatures [K] a mobility check samples.

        The prescribed extremes when the analysis has any, else 300 K and every
        heat-boundary temperature, which bound a solved-for temperature.
        """
        extremes = self._analysis_temperature_extremes()
        if extremes is not None:
            return list(extremes)
        return [
            300.0,
            *(
                bc.condition.temperature
                if isinstance(bc.condition, TemperatureBC)
                else bc.condition.ambient_temperature
                for bc in self.boundary_spec
                if isinstance(bc.condition, (TemperatureBC, ConvectionBC, RadiationBC))
            ),
        ]

    def _check_mobility_model_families(self) -> Self:
        """Error if a semiconductor's carriers use different low-field mobility families.

        Either charge solver takes one low-field family per medium for both carriers, so the mix
        would be refused only after upload.
        """
        for loc, charge in self._iter_semiconductor_charge_media():
            families = {type(charge.mobility_n), type(charge.mobility_p)}
            if len(families) == 1:
                continue
            hint = ""
            if families == {ConstantMobilityModel, CaugheyThomasMobility}:
                hint = (
                    " A constant mobility is the 'CaugheyThomasMobility' with 'mu_min' and 'mu' "
                    "both set to it and 'exp_1' and 'exp_2' set to 0."
                )
            self._raise_validation_error_at_loc(
                f"The electron mobility model is '{type(charge.mobility_n).__name__}' and the "
                f"hole mobility model is '{type(charge.mobility_p).__name__}'. The charge solver "
                "evaluates one low-field mobility family for both carriers of a medium, so "
                f"they must be of the same type.{hint}",
                *loc,
            )
        return self

    def _check_masetti_mobility_models(self) -> Self:
        """Error if a Masetti model's T-scaled high-doping asymptote is ≤ 0.

        Raises directly at the offending charge medium's loc (``("medium",)``
        or ``("structures", i)``) so the user is pointed at the exact field to
        fix; the iteration helper yields the loc alongside each charge spec.

        The asymptote is checked at every candidate temperature, since the sign of
        'exp_0' decides which one is worst.
        """
        candidate_temperatures = self._mobility_temperature_candidates()
        for loc, charge in self._iter_semiconductor_charge_media():
            for carrier, mobility in (
                ("electron", charge.mobility_n),
                ("hole", charge.mobility_p),
            ):
                if not isinstance(mobility, MasettiMobility):
                    continue
                worst_temperature, high_doping_limit = min(
                    (
                        (T, mobility.mu_0 * (T / 300.0) ** mobility.exp_0 - mobility.mu_1)
                        for T in candidate_temperatures
                    ),
                    key=lambda candidate: candidate[1],
                )
                if high_doping_limit <= 0.0:
                    self._raise_validation_error_at_loc(
                        f"MasettiMobility high-doping asymptote for {carrier} mobility "
                        f"is non-positive at {worst_temperature} K "
                        "('mu_0 * (T/300)**exp_0 - mu_1' <= 0); the accelerated evaluator would "
                        "silently clamp mobility to zero at high doping. Reduce 'mu_1', "
                        "increase 'mu_0', or lower the lattice temperature.",
                        *loc,
                    )
        return self

    def _check_alloy_mobility_models(self) -> Self:
        """Error if an AlloyMobility model's mobility is not positive.

        Only a negative 'mu_alloy' can make '1/mu' non-positive. It is checked at doping 0
        and 1e10-1e23 cm^-3 over the analysis temperature range; when the temperature is
        solved for, that range spans 300 K and every heat-boundary temperature.
        """
        candidates = self._mobility_temperature_candidates()
        temperatures = np.linspace(min(candidates), max(candidates), 9)
        doping = np.concatenate(([0.0], np.logspace(10, 23, 27)))
        for loc, charge in self._iter_semiconductor_charge_media():
            for carrier, mobility in (
                ("electron", charge.mobility_n),
                ("hole", charge.mobility_p),
            ):
                if not isinstance(mobility, AlloyMobility):
                    continue
                if mobility.mu_alloy is None or mobility.mu_alloy > 0.0:
                    continue
                for temperature in temperatures:
                    inv_mu = mobility._inverse_low_field_mobility(doping, temperature)
                    worst = int(np.argmin(inv_mu))
                    if inv_mu[worst] <= 0.0:
                        medium = f" of medium '{charge.name}'" if charge.name else ""
                        self._raise_validation_error_at_loc(
                            f"AlloyMobility {carrier} mobility{medium} is not positive at "
                            f"doping {doping[worst]:.3g} cm^-3 and {temperature:g} K: the "
                            f"negative 'mu_alloy' ({mobility.mu_alloy:g}) outweighs the "
                            "end-member terms of '1/mu'. Increase the magnitude of 'mu_alloy' "
                            "or check the end-member mobilities.",
                            *loc,
                        )
        return self

    @staticmethod
    def _surface_dependence(mobility: Any) -> Any:
        """The ``surface_dependence`` of a mobility model, ``None`` for models without one."""
        return getattr(mobility, "surface_dependence", None)

    def _check_surface_dependence_mobility_models(self) -> Self:
        """Error if surface degradation is set on one carrier only; warn without an insulator."""
        any_surface_dependence = False
        for loc, charge in self._iter_semiconductor_charge_media():
            surface_n = self._surface_dependence(charge.mobility_n)
            surface_p = self._surface_dependence(charge.mobility_p)
            any_surface_dependence = any_surface_dependence or surface_n is not None
            if (surface_n is None) != (surface_p is None):
                present, missing = (
                    ("electron", "hole") if surface_p is None else ("hole", "electron")
                )
                self._raise_validation_error_at_loc(
                    f"'surface_dependence' is set on the {present} mobility model but not on "
                    f"the {missing} one; surface mobility degradation must be enabled for both "
                    "carriers or neither.",
                    *loc,
                )
        if any_surface_dependence and not any(
            isinstance(self._charge_spec(medium), ChargeInsulatorMedium)
            for medium in [self.medium, *(structure.medium for structure in self.structures)]
        ):
            log.warning(
                "'surface_dependence' is set on a semiconductor mobility model, but the "
                "simulation has no insulator medium. The Lombardi surface terms act only "
                "near semiconductor/insulator interfaces and will have no effect."
            )
        return self

    def _check_field_dependence_mobility_models(self) -> Self:
        """Error if a field dependence is single-carrier, or if v_sat(T)/beta(T) is clamped.

        Raises directly at the offending charge medium's loc (``("medium",)``
        or ``("structures", i)``) so the user is pointed at the exact field to
        fix; the iteration helper yields the loc alongside each charge spec.

        Both temperature-scaled quantities are evaluated at the extremes of the
        prescribed lattice temperature, since the sign of each exponent decides
        whether the coldest or the hottest node is its worst case. When the
        temperature is solved for instead, there is nothing to check here and the
        solver applies the floor and the clamp at runtime.
        """
        temperature_samples = self._analysis_temperature_extremes()
        for loc, charge in self._iter_semiconductor_charge_media():
            field_dependence_n = charge.mobility_n.field_dependence
            field_dependence_p = charge.mobility_p.field_dependence
            if (field_dependence_n is None) != (field_dependence_p is None):
                present, missing = (
                    ("electron", "hole") if field_dependence_p is None else ("hole", "electron")
                )
                self._raise_validation_error_at_loc(
                    f"'field_dependence' is set on the {present} mobility model but not on "
                    f"the {missing} one; a high-field mobility model must be set for both "
                    "carriers or neither.",
                    *loc,
                )
            if temperature_samples is None:
                continue
            for carrier, field_dependence in (
                ("electron", field_dependence_n),
                ("hole", field_dependence_p),
            ):
                if field_dependence is None:
                    continue
                for temperature in temperature_samples:
                    scale = temperature / 300.0
                    v_sat = field_dependence.v_sat * scale**field_dependence.exp_v_sat
                    if v_sat <= V_SAT_FLOOR:
                        self._raise_validation_error_at_loc(
                            f"'{type(field_dependence).__name__}' saturation velocity for the "
                            f"{carrier} mobility is {v_sat:.3g} cm/s at {temperature:g} K "
                            "('v_sat * (T/300)**exp_v_sat'), at or below the "
                            f"{V_SAT_FLOOR:g} cm/s floor the accelerated charge solver "
                            "clamps to. Adjust 'v_sat' or 'exp_v_sat'.",
                            *loc,
                        )
                    if not isinstance(field_dependence, CanaliFieldDependence):
                        continue
                    beta = field_dependence.beta * scale**field_dependence.exp_beta
                    if not CANALI_BETA_MIN <= beta <= CANALI_BETA_MAX:
                        self._raise_validation_error_at_loc(
                            f"'CanaliFieldDependence' exponent for the {carrier} mobility is "
                            f"{beta:.3g} at {temperature:g} K ('beta * (T/300)**exp_beta'), "
                            f"outside the [{CANALI_BETA_MIN}, {CANALI_BETA_MAX}] range the "
                            "accelerated charge solver clamps to. Adjust 'beta' or 'exp_beta'.",
                            *loc,
                        )
        return self

    def _warn_direct_linear_solver(self) -> Self:
        """Warn about the direct solver's cost, and about the settings it ignores.

        Both warn rather than raise: neither is a misconfiguration, and the direct path is
        chosen precisely when robustness is worth paying for. The ignored-settings warning
        exists because a user who raised the iterative limits to chase a stall, then
        switched to ``'direct'`` to fix it, would otherwise never learn that the first half
        of that change stopped doing anything. It is charge-only: the heat and conduction
        solvers take their iterative settings from the backend, not from the simulation.
        """
        if self.linear_solver != "direct":
            return self

        log.warning(
            "linear_solver='direct' factorizes the system matrix instead of iterating "
            "on it. This is more robust where the iterative solver converges slowly or "
            "fails, but it uses considerably more memory, particularly on large meshes, "
            "and can be slower.",
            # 'updated_copy' re-validates, so a bias sweep would repeat this per point
            log_once=True,
        )

        if not isinstance(self.analysis_spec, SteadyChargeDCAnalysis):
            return self

        tolerance_settings = self.analysis_spec.tolerance_settings
        tolerance_fields = type(tolerance_settings).model_fields
        ignored = [
            name
            for name in ITERATIVE_ONLY_TOLERANCE_FIELDS
            if getattr(tolerance_settings, name) != tolerance_fields[name].default
        ]
        if ignored:
            log.warning(
                f"{', '.join(ignored)} in 'ChargeToleranceSpec' "
                f"{'configure' if len(ignored) > 1 else 'configures'} the iterative linear "
                "solver and will be ignored with linear_solver='direct'.",
                log_once=True,
            )
        return self

    @property
    def _resolve_use_accelerated_solver(self) -> bool:
        """Resolved value of :attr:`use_accelerated_solver`.

        The accelerated solver is the default for every simulation, and the flag only
        selects the CPU solver for a charge analysis. Heat and conduction always run on the
        accelerated solver, but its charge mesh and solver path does not apply, so they
        resolve to ``False``. The construction path rejects an invalid flag.
        """
        return (
            isinstance(self.analysis_spec, SteadyChargeDCAnalysis) and self.use_accelerated_solver
        )

    def _useHeatSourceFromConductionSim(self) -> bool:
        """Returns True if 'HeatFromElectricSource' has been defined."""
        return any(isinstance(source, HeatFromElectricSource) for source in self.sources)

    def _get_ssac_frequency_and_amplitude(self) -> tuple[ArrayFloat1D, FiniteFloat]:
        if not isinstance(self.analysis_spec, (SSACAnalysis, IsothermalSSACAnalysis)):
            raise SetupError(
                "Invalid analysis type for Small-Signal AC (SSAC). "
                "SSAC requires a 'SSACAnalysis' or 'IsothermalSSACAnalysis', "
                f"but received '{type(self.analysis_spec).__name__}' instead."
            )

        amplitude = None
        for bc in self.boundary_spec:
            if isinstance(bc.condition, VoltageBC) and isinstance(
                bc.condition.source, SSACVoltageSource
            ):
                amplitude = bc.condition.source.amplitude
        return (self.analysis_spec.freqs, amplitude)
