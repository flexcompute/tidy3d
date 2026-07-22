"""Defines heat simulation class"""

from __future__ import annotations

import math
from enum import Enum
from typing import TYPE_CHECKING, Any

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
    ChargeConductorMedium,
    SemiconductorMedium,
)
from tidy3d.components.material.tcad.heat import (
    AnisotropicConductivity,
    FluidMedium,
    SolidMedium,
)
from tidy3d.components.material.types import MultiPhysicsMedium, StructureMediumType
from tidy3d.components.medium import Medium
from tidy3d.components.scene import Scene
from tidy3d.components.spice.sources.ac import SSACVoltageSource
from tidy3d.components.spice.sources.dc import DCVoltageSource
from tidy3d.components.spice.types import (
    ElectricalAnalysisType,
    IsothermalSSACAnalysis,
    IsothermalSteadyChargeDCAnalysis,
    SSACAnalysis,
    SteadyChargeDCAnalysis,
)
from tidy3d.components.structure import Structure
from tidy3d.components.tcad.analysis.heat_simulation_type import UnsteadyHeatAnalysis
from tidy3d.components.tcad.boundary.heat import VerticalNaturalConvectionCoeffModel
from tidy3d.components.tcad.boundary.specification import HeatBoundarySpec, HeatChargeBoundarySpec
from tidy3d.components.tcad.generation_recombination import (
    PalankovskiQuayApproxCarrierLifetime,
    ShockleyReedHallRecombination,
)
from tidy3d.components.tcad.grid import (
    DistanceUnstructuredGrid,
    UniformUnstructuredGrid,
    UnstructuredGridType,
)
from tidy3d.components.tcad.mobility import MasettiMobility
from tidy3d.components.tcad.monitors.charge import (
    SteadyCapacitanceMonitor,
    SteadyChargeResidualMonitor,
    SteadyCurrentDensityMonitor,
    SteadyFreeCarrierMonitor,
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
    from collections.abc import Iterable, Iterator
    from typing import Literal

    from pydantic import FiniteFloat

    from tidy3d.compat import Self
    from tidy3d.components.data.data_array import SpatialDataArray
    from tidy3d.components.types import Ax, Bound, Shapely
    from tidy3d.components.types.base import ArrayFloat1D
    from tidy3d.components.viz import PlotParams

HEAT_CHARGE_BACK_STRUCTURE_STR = "<<<HEAT_CHARGE_BACKGROUND_STRUCTURE>>>"

HeatBCTypes = (TemperatureBC, HeatFluxBC, ConvectionBC, RadiationBC, ThermalContactResistance)
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
)

AnalysisSpecType = ElectricalAnalysisType | UnsteadyHeatAnalysis

# define some limits for transient heat simulations
TRANSIENT_HEAT_MAX_STEPS = 1000

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
        (),
        title="Heat and Charge sources",
        description="List of heat and/or charge sources.",
    )

    monitors: tuple[discriminated_union(HeatChargeMonitorType), ...] = Field(
        (),
        title="Monitors",
        description="Monitors in the simulation.",
    )

    boundary_spec: tuple[discriminated_union(HeatChargeBoundarySpec | HeatBoundarySpec), ...] = (
        Field(
            (),
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
        (0, 0, 0),
        title="Symmetries",
        description="Tuple of integers defining reflection symmetry across a plane "
        "bisecting the simulation domain normal to the x-, y-, and z-axis "
        "at the simulation center of each axis, respectively. "
        "Each element can be ``0`` (symmetry off) or ``1`` (symmetry on).",
    )

    analysis_spec: AnalysisSpecType | None = Field(
        None,
        discriminator=TYPE_TAG_STR,
        title="Analysis specification.",
        description="The `analysis_spec` is used to specify the type of simulation. Currently, it is used to "
        "specify Charge simulations or transient Heat simulations.",
    )

    use_accelerated_solver: bool = Field(
        True,
        title="Use accelerated solver.",
        description="Controls the solver used for charge simulations. When ``True`` "
        "(default), the GPU accelerated charge solver is used. Set to ``False`` "
        "to use the CPU charge solver instead; this is rejected when the simulation uses "
        "a feature available only on the GPU accelerated solver, such as ``MasettiMobility``. "
        "The flag applies only to charge simulations: heat and conduction simulations "
        "always run on the GPU accelerated solver, so ``False`` is not allowed for them.",
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
        self._call_with_validation_loc(("boundary_spec",), self._check_voltage_array_if_capacitance)
        self._call_with_validation_loc(("boundary_spec",), self._names_exist_bcs)
        self._call_with_validation_loc(("boundary_spec",), self._check_natural_convection_bc)
        self._call_with_validation_loc(
            ("boundary_spec",), self._check_thermal_contact_resistance_placement
        )
        self._call_with_validation_loc(("structures",), self._check_heat_only_features_in_charge)
        self._call_with_validation_loc(("boundary_spec",), self._check_freqs_requires_ac_source)
        self._call_with_validation_loc(
            ("analysis_spec", "at_voltages"), self._check_ssac_specific_voltages
        )
        simulation_types = self._check_simulation_types()
        self._call_with_validation_loc(("monitors",), self._validate_residual_monitor_requirements)
        if TCADAnalysisTypes.CHARGE in simulation_types:
            self._call_with_validation_loc(
                ("boundary_spec",), self._check_charge_simulation_voltage_bcs
            )
            self._call_with_validation_loc(("monitors",), self._check_charge_simulation_monitors)
            self._call_with_validation_loc(
                ("structures",), self._check_charge_simulation_semiconductors
            )
            self._call_with_validation_loc(("structures",), self._check_masetti_mobility_models)
            # Schottky contacts and surface recombination apply only to
            # charge simulations, so validate them inside the charge guard;
            # heat-only and conduction-only simulations skip these checks.
            self._call_with_validation_loc(("boundary_spec",), self._check_schottky_supported_modes)
            self._call_with_validation_loc(
                ("boundary_spec",), self._check_surface_recombination_bcs
            )
            self._call_with_validation_loc(
                ("structures",), self._warn_non_accelerated_ignores_electron_affinity
            )
        self._call_with_validation_loc(("boundary_spec",), self._not_all_neumann)
        self._call_with_validation_loc(("grid_spec",), self._names_exist_grid_spec)
        self._call_with_validation_loc(("grid_spec",), self._warn_if_minimal_mesh_size_override)
        self._call_with_validation_loc(("sources",), self._names_exist_sources)
        self._call_with_validation_loc(("structures",), self._check_medium_specs)
        self._call_with_validation_loc(("sources",), self._check_coupling_source_can_be_applied)
        if TCADAnalysisTypes.HEAT in simulation_types:
            self._call_with_validation_loc(("monitors",), self._check_heat_sim)
        if TCADAnalysisTypes.CONDUCTION in simulation_types:
            self._call_with_validation_loc(("monitors",), self._check_conduction_sim_monitors)
            self._call_with_validation_loc(
                ("boundary_spec",), self._check_conduction_sim_voltage_arrays
            )
            self._call_with_validation_loc(("structures",), self._check_conduction_sim_structures)
        self._call_with_validation_loc(("grid_spec",), self._estimate_charge_mesh_size)
        if isinstance(self.analysis_spec, UnsteadyHeatAnalysis):
            self._call_with_validation_loc(("monitors",), self._check_transient_heat_monitors)
            self._call_with_validation_loc(
                ("structures",), self._check_transient_heat_solid_properties
            )
            self._call_with_validation_loc(
                ("analysis_spec",), self._check_transient_heat_time_steps
            )
            self._check_transient_heat_time_warning()
        self._call_with_validation_loc(("structures",), self._check_non_isothermal_is_possible)
        self._call_with_validation_loc(
            ("use_accelerated_solver",), self._check_use_accelerated_solver
        )
        return self

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

    @field_validator("boundary_spec")
    @classmethod
    def _check_single_ssac(
        cls, boundary_spec: HeatChargeBoundarySpec | HeatBoundarySpec
    ) -> HeatChargeBoundarySpec | HeatBoundarySpec:
        ssac_present = False
        for bc in boundary_spec:
            if isinstance(bc.condition, VoltageBC):
                if isinstance(bc.condition.source, SSACVoltageSource):
                    if ssac_present:
                        raise SetupError(
                            "Only a single 'SSACVoltageSource' source can be supplied."
                        )
                    else:
                        ssac_present = True
        return boundary_spec

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

    @field_validator("boundary_spec")
    @classmethod
    def _check_only_one_voltage_array_provided(cls, val: Any) -> Any:
        """Issue error if more than one voltage array is provided.
        Currently we only allow to sweep over one voltage array.
        """
        array_already_provided = False

        for bc in val:
            if isinstance(bc.condition, VoltageBC):
                voltages = []
                # both DC and SSAC sources carry a DC sweep array; counting only
                # one type would admit an ambiguous two-sweep setup
                if isinstance(bc.condition.source, (DCVoltageSource, SSACVoltageSource)):
                    voltages = bc.condition.source.voltage

                if len(voltages) > 1:
                    if not array_already_provided:
                        array_already_provided = True
                    else:
                        raise SetupError(
                            "More than one voltage array has been provided. "
                            "Currently voltage arrays are supported only for one of the BCs."
                        )
        return val

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
        """Validate Charge simulation has enough voltage BCs."""
        voltage_bcs = 0
        for bc in self.boundary_spec:
            if isinstance(bc.condition, VoltageBC):
                voltage_bcs = voltage_bcs + 1
        if voltage_bcs < 2:
            raise SetupError(
                "Defining a Charge simulation requires at least two voltage contact boundaries. "
                "Use 'VoltageBC' (Schottky contacts opt in via model=\"schottky_mott\"). "
                f"So far {voltage_bcs} voltage contact boundaries have been set."
            )
        return self

    def _check_charge_simulation_monitors(self) -> Self:
        """Validate Charge simulation has at least one charge monitor."""
        if not any(isinstance(mnt, ChargeMonitorTypes) for mnt in self.monitors):
            raise SetupError(
                "Charge simulations require the definition of, at least, one of these monitors: "
                "'[SteadyPotentialMonitor, SteadyFreeCarrierMonitor, SteadyCapacitanceMonitor, SteadyCurrentDensityMonitor, SteadyChargeResidualMonitor]' "
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

    def _validate_residual_monitor_requirements(self) -> Self:
        """SteadyChargeResidualMonitor requires charge analysis and the accelerated solver."""
        simulation_types = self._check_simulation_types()
        charge_configured = TCADAnalysisTypes.CHARGE in simulation_types
        for idx, mnt in enumerate(self.monitors):
            if not isinstance(mnt, SteadyChargeResidualMonitor):
                continue
            if not charge_configured:
                self._raise_validation_error_at_loc(
                    "'SteadyChargeResidualMonitor' is only available when a charge analysis "
                    "is configured (the simulation must include voltage BCs and a "
                    "'SteadyChargeDCAnalysis' or derivative analysis spec).",
                    "monitors",
                    idx,
                )
            if self.use_accelerated_solver is False:
                self._raise_validation_error_at_loc(
                    f"'SteadyChargeResidualMonitor' (monitor '{mnt.name}') is only available "
                    "through the accelerated charge solver, but 'use_accelerated_solver=False' "
                    "was set. Remove the monitor or use the accelerated solver (the default).",
                    "monitors",
                    idx,
                )
        return self

    def _check_charge_simulation_semiconductors(self) -> Self:
        """Validate Charge simulation has at least one semiconductor medium."""
        sc_present = HeatChargeSimulation._check_if_semiconductor_present(
            structures=self.structures
        )
        if not sc_present:
            raise SetupError(
                f"{TCADAnalysisTypes.CHARGE} simulations require the definition of at least one semiconductor medium."
            )
        return self

    @staticmethod
    def _bc_is_schottky(bc_spec: Any) -> bool:
        """``True`` when this boundary requests Schottky physics."""
        condition = bc_spec.condition
        return isinstance(condition, VoltageBC) and condition.model == "schottky_mott"

    def _check_schottky_supported_modes(self) -> Self:
        """Reject Schottky configurations outside the validated solver model."""
        has_schottky = any(self._bc_is_schottky(bc) for bc in self.boundary_spec)
        if not has_schottky:
            return self

        if self.use_accelerated_solver is False:
            self._raise_validation_error_at_loc(
                "Schottky contacts ('VoltageBC' with model=\"schottky_mott\") are "
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
                    "Schottky contacts ('VoltageBC' with model=\"schottky_mott\") "
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

    def _check_surface_recombination_bcs(self) -> Self:
        """Run all ``SurfaceRecombinationBC`` setup checks."""
        self._check_surface_recombination_requires_accelerated()
        self._check_surface_recombination_not_stacked_with_current_bc()
        self._check_surface_recombination_not_stacked_with_insulating_bc()
        self._check_surface_recombination_not_stacked_with_schottky()
        self._check_surface_recombination_qf_on_voltage_overlay()
        self._check_surface_recombination_no_duplicate_placement()
        return self

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

    def _check_surface_recombination_not_stacked_with_current_bc(self) -> Self:
        """Reject ``SurfaceRecombinationBC`` sharing a placement with ``CurrentBC``,
        which would prescribe the carrier flux on the same face twice."""
        for i, sr_bc in enumerate(self.boundary_spec):
            if not isinstance(sr_bc.condition, SurfaceRecombinationBC):
                continue
            for j, other_bc in enumerate(self.boundary_spec):
                if i == j or not isinstance(other_bc.condition, CurrentBC):
                    continue
                if self._placements_overlap(sr_bc.placement, other_bc.placement):
                    raise SetupError(
                        "'SurfaceRecombinationBC' cannot share a placement with "
                        "'CurrentBC'. CurrentBC prescribes the normal carrier "
                        "flux; composing it with a Robin SR contribution is "
                        "deferred."
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
                        "Schottky contact ('VoltageBC' with model=\"schottky_mott\"). "
                        "Contact surface recombination overlays are supported only "
                        'for ohmic contacts (model="ohmic").'
                    )
        return self

    def _check_surface_recombination_qf_on_voltage_overlay(self) -> Self:
        """Reject non-zero ``Q_f`` on a placement shared with ``VoltageBC``,
        where the contact screens fixed sheet charge."""
        for i, sr_bc in enumerate(self.boundary_spec):
            if not isinstance(sr_bc.condition, SurfaceRecombinationBC):
                continue
            if not sr_bc.condition.Q_f:
                continue
            for j, other_bc in enumerate(self.boundary_spec):
                if i == j or not isinstance(other_bc.condition, VoltageBC):
                    continue
                if self._placements_overlap(sr_bc.placement, other_bc.placement):
                    raise SetupError(
                        "'SurfaceRecombinationBC' with a non-zero 'Q_f' "
                        "cannot share a placement with 'VoltageBC'. The "
                        "metal contact pins the electrostatic potential and "
                        "screens fixed sheet charge, so the Poisson "
                        "contribution from Q_f would be ignored on the "
                        "overlay nodes."
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

        for semiconductor in self._semiconductor_charge_media(self.structures):
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

    def _not_all_neumann(self) -> Self:
        """Make sure not all BCs are of Neumann type"""

        NeumannBCsHeat = (HeatFluxBC, ThermalContactResistance)
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
                    isinstance(bc.condition, NeumannBCsHeat) for bc in type_bcs
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

        names_neumann_Bcs = [BC.__name__ for BC in NeumannBCsHeat]
        names_neumann_Bcs.extend([BC.__name__ for BC in NeumannBCsCharge])
        if raise_error:
            raise SetupError(
                "Current 'HeatChargeSimulation' contains only Neumann-type boundary conditions. "
                "Steady-state solution is undefined in this case. "
                f"Current Neumann BCs are {names_neumann_Bcs}"
            )

        return self

    def _names_exist_grid_spec(self) -> Self:
        """Warn if 'UniformUnstructuredGrid' points at a non-existing structure."""
        structures_names = {s.name for s in self.structures}
        for structure_name in self.grid_spec.non_refined_structures:
            if structure_name not in structures_names:
                log.warning(
                    f"Structure '{structure_name}' listed as a non-refined structure in "
                    "'HeatChargeSimulation.grid_spec' is not present in 'HeatChargeSimulation.structures'"
                )
        return self

    def _warn_if_minimal_mesh_size_override(self) -> Self:
        """Warn if minimal mesh size limit overrides desired mesh size."""
        val = self.grid_spec
        max_size = np.max(self.size)
        min_dl = val.relative_min_dl * max_size

        if isinstance(val, UniformUnstructuredGrid):
            desired_min_dl = val.dl
        if isinstance(val, DistanceUnstructuredGrid):
            desired_min_dl = min(val.dl_interface, val.dl_bulk)

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

    @staticmethod
    def _semiconductor_charge_media(
        structures: Iterable[Structure],
    ) -> Iterable[SemiconductorMedium]:
        """Yield semiconductor charge media from bare and multiphysics structure media."""
        for structure in structures:
            if isinstance(structure.medium, SemiconductorMedium):
                yield structure.medium
            elif isinstance(structure.medium, MultiPhysicsMedium):
                charge_medium = structure.medium.charge
                if isinstance(charge_medium, SemiconductorMedium):
                    yield charge_medium

    @staticmethod
    def _check_if_semiconductor_present(structures: Iterable[Structure]) -> bool:
        """Checks whether the simulation object can run a Charge simulation."""

        return any(HeatChargeSimulation._semiconductor_charge_media(structures))

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

        semiconductor_present = HeatChargeSimulation._check_if_semiconductor_present(
            structures=self.structures
        )

        for boundary in self.boundary_spec:
            if isinstance(boundary.condition, HeatBCTypes):
                simulation_types.append(TCADAnalysisTypes.HEAT)
            if isinstance(boundary.condition, ElectricBCTypes):
                # Add CONDUCTION type if we have no semiconductors
                if not semiconductor_present:
                    simulation_types.append(TCADAnalysisTypes.CONDUCTION)

        for source in self.sources:
            if isinstance(source, HeatSourceTypes):
                simulation_types.append(TCADAnalysisTypes.HEAT)

        return set(simulation_types)

    def _check_coupling_source_can_be_applied(self) -> Self:
        """Error if material doesn't have the right specifications"""

        HeatSourceTypes_noCoupling = (UniformHeatSource, HeatSource)

        simulation_types = self._check_simulation_types(HeatSourceTypes=HeatSourceTypes_noCoupling)
        simulation_types = list(simulation_types)

        for source in self.sources:
            if isinstance(source, HeatFromElectricSource) and len(simulation_types) < 2:
                raise SetupError(
                    f"Using 'HeatFromElectricSource' requires the definition of both "
                    f"{TCADAnalysisTypes.CONDUCTION.name} and {TCADAnalysisTypes.HEAT.name}. "
                    f"The current simulation setup contains only conditions of type {simulation_types[0].name}"
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

    def _check_transient_heat_monitors(self) -> Self:
        """Validate monitor settings for transient heat simulations."""
        monitors = self.monitors
        for mnt in monitors:
            if isinstance(mnt, TemperatureMonitor):
                if not mnt.unstructured:
                    raise SetupError(
                        f"Unsteady simulations require the temperature monitor '{mnt.name}' to be unstructured."
                    )
        return self

    def _transient_heat_material_lists(
        self,
    ) -> tuple[list[float], list[float], list[float | SpatialDataArray]]:
        """Collect thermal property values used by transient heat validators."""
        capacities = []
        densities = []
        conductivities = []
        structures = self.structures
        for structure in structures:
            heat_properties = None
            if isinstance(structure.medium, MultiPhysicsMedium):
                heat_properties = structure.medium.heat
            # now check legacy Medium too
            elif isinstance(structure.medium, Medium):
                heat_properties = structure.medium.heat_spec

            if isinstance(heat_properties, SolidMedium):
                if heat_properties.capacity is not None:
                    capacities.append(heat_properties.capacity)
                if heat_properties.density is not None:
                    densities.append(heat_properties.density)
                conductivity = heat_properties.conductivity
                # Scalar diffusion-time estimate: reduce a tensor to its mean principal value.
                if isinstance(conductivity, AnisotropicConductivity):
                    conductivity = (conductivity.xx + conductivity.yy + conductivity.zz) / 3
                conductivities.append(conductivity)
        return capacities, densities, conductivities

    def _check_transient_heat_solid_properties(self) -> Self:
        """Validate thermal material properties for transient heat simulations."""
        capacities, densities, conductivities = self._transient_heat_material_lists()
        if len(capacities) == 0 or len(densities) == 0 or len(conductivities) == 0:
            raise SetupError(
                "Unsteady simulations require the SolidSpec to have 'capacity', 'density', and 'conductivity' "
                "defined. Please check the definition of the SolidSpec in the Medium or MultiPhysicsMedium."
            )
        return self

    def _check_transient_heat_time_steps(self) -> Self:
        """Validate transient heat analysis time-step count."""
        analysis_type = self.analysis_spec
        if analysis_type.unsteady_spec.total_time_steps > TRANSIENT_HEAT_MAX_STEPS:
            raise SetupError(
                "Unsteady simulations require the number of time-steps to be less than "
                f"{TRANSIENT_HEAT_MAX_STEPS} but {analysis_type.unsteady_spec.total_time_steps} were provided."
            )
        return self

    def _check_transient_heat_time_warning(self) -> Self:
        """Warn when transient heat simulation time is likely excessive."""
        analysis_type = self.analysis_spec
        capacities, densities, conductivities = self._transient_heat_material_lists()
        if len(capacities) == 0 or len(densities) == 0 or len(conductivities) == 0:
            return self
        domain_length = np.max([d for d in self.size if d != np.inf])
        characteristic_time = (
            domain_length**2
            * np.mean(capacities)
            * np.mean(densities)
            / np.mean(conductivities)
            * 1e-18
        )
        if (
            analysis_type.unsteady_spec.time_step * analysis_type.unsteady_spec.total_time_steps
            > 100 * characteristic_time
        ):
            log.warning(
                "The simulation time is larger than 100 times the estimated characteristic time of the system. "
                "This may lead to unnecessary long simulation times. "
                "Consider reducing the simulation time or the time step size."
            )
        return self

    def _check_non_isothermal_is_possible(self) -> Self:
        """Make sure that when a non-isothermal case is defined the structures
        have both electrical and thermal properties."""

        analysis_spec = self.analysis_spec
        if isinstance(analysis_spec, SteadyChargeDCAnalysis) and not isinstance(
            analysis_spec, IsothermalSteadyChargeDCAnalysis
        ):
            has_heat = False
            has_elec = False
            structures = self.structures
            for struct in structures:
                if isinstance(struct.medium, MultiPhysicsMedium):
                    if struct.medium.heat is not None:
                        if isinstance(struct.medium.heat, SolidMedium):
                            has_heat = True
                    if struct.medium.charge is not None:
                        if isinstance(struct.medium.charge, SemiconductorMedium):
                            has_elec = True

            if not has_heat and has_elec:
                raise SetupError(
                    "The current simulation is defined as non-isothermal but no solid "
                    "materials with heat properties have been defined. "
                )
            elif not has_elec and has_heat:
                raise SetupError(
                    "The current simulation is defined as non-isothermal but no "
                    "semiconductor materials have been defined. "
                )
            elif not has_heat and not has_elec:
                raise SetupError(
                    "The current simulation is defined as non-isothermal but no "
                    "solid or semiconductor materials have been defined. "
                )
        return self

    def _check_heat_only_features_in_charge(self) -> Self:
        """Reject heat-only-solver features in non-isothermal charge simulations.

        Solid-medium advection ('SolidMedium.velocity'), anisotropic thermal conductivity
        ('AnisotropicConductivity'), resistive interfaces ('ThermalContactResistance'), and
        gray-body surface radiation ('RadiationBC', 'ConvectionBC.emissivity') are honored by
        the heat solver, including when it is coupled with electrical conduction. The coupled
        thermal solve that runs alongside a non-isothermal charge analysis does not support any
        of them, so a setup that requests them would silently produce a result that ignores
        them. Flag them here instead. Heat, conduction+heat, and isothermal charge analyses
        (the latter runs no thermal solve) are unaffected."""
        if not self._thermal_solver_active:
            return self

        # Advection velocity and anisotropic conductivity, on the background medium or any
        # structure's solid heat spec.
        media_sources = [(("medium",), self.medium)]
        media_sources.extend(
            (("structures", i), struct.medium) for i, struct in enumerate(self.structures)
        )
        for loc, medium in media_sources:
            heat_spec = (
                medium if isinstance(medium, SolidMedium) else getattr(medium, "heat_spec", None)
            )
            velocity = getattr(heat_spec, "velocity", None)
            if velocity is not None and any(v != 0.0 for v in velocity):
                self._raise_validation_error_at_loc(
                    "Solid-medium advection ('SolidMedium.velocity') is not supported in "
                    "non-isothermal charge (coupled charge+heat) simulations: the coupled "
                    "thermal solve does not apply the convective transport term "
                    "'rho * cp * V . grad(T)', so this velocity would be silently ignored. "
                    "Remove 'velocity' (or set it to 'None') to run this charge simulation.",
                    *loc,
                )
            if isinstance(getattr(heat_spec, "conductivity", None), AnisotropicConductivity):
                self._raise_validation_error_at_loc(
                    "Anisotropic thermal conductivity ('AnisotropicConductivity') is not "
                    "supported in non-isothermal charge (coupled charge+heat) simulations: the "
                    "coupled thermal solve only handles a scalar (isotropic) conductivity, so a "
                    "tensor conductivity would be silently ignored. Provide a scalar "
                    "'conductivity' to run this charge simulation.",
                    *loc,
                )

        # Resistive interfaces and surface radiation.
        for i, bc in enumerate(self.boundary_spec):
            if isinstance(bc.condition, ThermalContactResistance):
                self._raise_validation_error_at_loc(
                    "Resistive interfaces ('ThermalContactResistance') are not supported in "
                    "non-isothermal charge (coupled charge+heat) simulations: the coupled "
                    "thermal solve does not apply the interfacial thermal resistance, so this "
                    "boundary condition would be silently ignored. Remove it to run this "
                    "charge simulation.",
                    "boundary_spec",
                    i,
                )
            if isinstance(bc.condition, RadiationBC) or (
                isinstance(bc.condition, ConvectionBC) and bc.condition.emissivity
            ):
                self._raise_validation_error_at_loc(
                    "Gray-body surface radiation ('RadiationBC', or 'ConvectionBC' with a "
                    "positive 'emissivity') is not yet supported in non-isothermal charge "
                    "(coupled charge+heat) simulations; it is available in heat and "
                    "conduction+heat simulations. Remove the radiative term to run this "
                    "charge simulation.",
                    "boundary_spec",
                    i,
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
                            bdry = shape.exterior.intersection(_shape)
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
    def _num_dc_solves(self) -> int:
        """Number of nonlinear solves the charge solver runs to cover the DC sweep.

        Mirrors the solver's sweep construction: a single requested voltage is
        one direct solve; otherwise the sweep starts at 0 V and covers positive
        then negative voltages by increasing magnitude, inserting a warm-start
        solve per ``convergence_dv`` interval within each pass. Doping ramp-up
        adds ``tolerance_settings.ramp_up_iters - 1`` solves. Charge sims only.
        """
        voltages = self._dc_voltages

        # Doping ramp: the initial solve runs once per ramp level.
        ramp_solves = self.analysis_spec.tolerance_settings.ramp_up_iters - 1

        if len(voltages) <= 1:
            # No sweep: one direct solve at the requested bias.
            return 1 + ramp_solves

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
        return 1 + pass_solves(positive) + pass_solves(negative) + ramp_solves

    @property
    def _thermal_solver_active(self) -> bool:
        """Whether a coupled thermal solve runs alongside the charge analysis.

        Returns ``True`` for non-isothermal :class:`SteadyChargeDCAnalysis` and
        ``False`` for :class:`IsothermalSteadyChargeDCAnalysis`. Determines
        whether the thermal residual ``residual_temperature`` is reported.
        """
        return isinstance(self.analysis_spec, SteadyChargeDCAnalysis) and not isinstance(
            self.analysis_spec, IsothermalSteadyChargeDCAnalysis
        )

    def _accelerated_only_features(self) -> list[str]:
        """Configured features that only the accelerated charge solver supports.

        These cannot run on the CPU charge solver, so requesting
        ``use_accelerated_solver=False`` while any of them is present is an error.
        Returns the human-readable feature names; an empty list means the
        configuration is fully supported by the CPU charge solver.
        """
        features = []
        for _loc, charge in self._iter_semiconductor_charge_media():
            if isinstance(charge.mobility_n, MasettiMobility) or isinstance(
                charge.mobility_p, MasettiMobility
            ):
                features.append("MasettiMobility")
                break

        if self._uses_gpu_only_lifetime_model():
            features.append("PalankovskiQuayApproxCarrierLifetime")

        if self._ssac_uses_bias_point_selection():
            features.append("SSAC 'at_voltages' bias-point selection")

        return features

    def _iter_semiconductor_charge_media(
        self,
    ) -> Iterator[tuple[tuple[Any, ...], SemiconductorMedium]]:
        """Yield ``(loc, charge)`` for every ``SemiconductorMedium`` in the simulation.

        Walks both the background ``self.medium`` and each ``Structure.medium``,
        which the mesher composes together into ``simulation_structure``. Each
        slot may be a raw :class:`SemiconductorMedium` or a
        :class:`MultiPhysicsMedium` that carries one as ``.charge``; slots with
        neither (insulators, conductors, fluids) are skipped. ``loc`` is the
        Pydantic field path of the source — ``("medium",)`` for the background,
        ``("structures", i)`` for a structure — so validators can raise
        loc-aware errors that point at the right field. Centralising both the
        traversal and the loc avoids the bugs where iterating only over
        ``.medium.charge`` (or only over ``structures``) silently misses
        raw-semiconductor or background-medium configurations, and lets every
        validator that uses this helper get the right loc for free.
        """
        sources: list[tuple[tuple[Any, ...], Any]] = [(("medium",), self.medium)]
        sources.extend(
            (("structures", i), structure.medium) for i, structure in enumerate(self.structures)
        )
        for loc, medium in sources:
            if isinstance(medium, SemiconductorMedium):
                yield loc, medium
            else:
                charge = getattr(medium, "charge", None)
                if isinstance(charge, SemiconductorMedium):
                    yield loc, charge

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

    def _check_masetti_mobility_models(self) -> Self:
        """Error if Masetti is mixed with another family or its T-scaled asymptote ≤ 0.

        Raises directly at the offending charge medium's loc (``("medium",)``
        or ``("structures", i)``) so the user is pointed at the exact field to
        fix; the iteration helper yields the loc alongside each charge spec.
        """
        temperature = getattr(self.analysis_spec, "temperature", None)
        for loc, charge in self._iter_semiconductor_charge_media():
            n_is_masetti = isinstance(charge.mobility_n, MasettiMobility)
            p_is_masetti = isinstance(charge.mobility_p, MasettiMobility)
            if n_is_masetti != p_is_masetti:
                self._raise_validation_error_at_loc(
                    "MasettiMobility must be used for both electron and hole mobility "
                    "models in a semiconductor medium. Mixing MasettiMobility with "
                    "another mobility family is not supported by the accelerated "
                    "charge solver.",
                    *loc,
                )
            if temperature is None:
                continue
            for carrier, mobility in (
                ("electron", charge.mobility_n),
                ("hole", charge.mobility_p),
            ):
                if not isinstance(mobility, MasettiMobility):
                    continue
                high_doping_limit = (
                    mobility.mu_0 * (temperature / 300.0) ** mobility.exp_0 - mobility.mu_1
                )
                if high_doping_limit <= 0.0:
                    self._raise_validation_error_at_loc(
                        f"MasettiMobility high-doping asymptote for {carrier} mobility "
                        f"is non-positive at {temperature} K "
                        "('mu_0 * (T/300)**exp_0 - mu_1' <= 0); the accelerated evaluator would "
                        "silently clamp mobility to zero at high doping. Reduce 'mu_1', "
                        "increase 'mu_0', or use a lower isothermal temperature.",
                        *loc,
                    )
        return self

    def _check_use_accelerated_solver(self) -> Self:
        """Validate ``use_accelerated_solver`` for the simulation type and features.

        Resolving the flag raises a ``SetupError`` for invalid combinations
        (``use_accelerated_solver=False`` on a non-charge simulation, or on a charge
        simulation that uses an accelerated-only feature). Triggering it here surfaces
        the error at construction, anchored to the field by ``_call_with_validation_loc``.
        """
        _ = self._resolve_use_accelerated_solver
        return self

    @property
    def _resolve_use_accelerated_solver(self) -> bool:
        """Resolved value of :attr:`use_accelerated_solver`.

        The accelerated solver is the default for every simulation. The
        ``use_accelerated_solver`` flag only applies to charge simulations:

        * Charge simulations use the accelerated solver unless
          ``use_accelerated_solver=False`` selects the CPU charge solver. That
          request raises when the configuration uses a feature only the
          accelerated solver supports (e.g. ``MasettiMobility``).
        * Heat and conduction simulations always run on the accelerated solver, so
          ``use_accelerated_solver=False`` is rejected for them. The accelerated
          *charge* mesh/solver path does not apply, so this resolves to ``False``
          (no charge prism mesh is generated).
        """
        is_charge = isinstance(self.analysis_spec, SteadyChargeDCAnalysis)

        if not is_charge:
            if not self.use_accelerated_solver:
                raise SetupError(
                    "'use_accelerated_solver=False' is only valid for charge simulations; "
                    "heat and conduction simulations always run on the GPU accelerated solver."
                )
            return False

        if not self.use_accelerated_solver:
            accelerated_only = self._accelerated_only_features()
            if accelerated_only:
                raise SetupError(
                    f"{', '.join(accelerated_only)} is supported only by the GPU accelerated "
                    "charge solver. Use 'use_accelerated_solver=True' (the default)."
                )
            return False

        return True

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
