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
    ChargeConductorMedium,
    SemiconductorMedium,
)
from tidy3d.components.material.tcad.heat import (
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
from tidy3d.components.tcad.boundary.heat import VerticalNaturalConvectionCoeffModel
from tidy3d.components.tcad.boundary.specification import HeatBoundarySpec, HeatChargeBoundarySpec
from tidy3d.components.tcad.generation_recombination import (
    SelberherrImpactIonization,
)
from tidy3d.components.tcad.grid import UnstructuredGridType
from tidy3d.components.tcad.monitors.charge import (
    SteadyCapacitanceMonitor,
    SteadyCurrentDensityMonitor,
    SteadyElectricFieldMonitor,
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

AnalysisSpecType = ElectricalAnalysisType | UnsteadyHeatAnalysis


# Minimum tolerance for cylinder radii
CYLINDER_RADIUS_TOL = 1e-6
# Minimum radius as fraction of the larger radius (for tapered cylinders)
MIN_CYLINDER_RADIUS_FRACTION = 0.01


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

    def _voltage_terminals(self, indices: list[int]) -> list[int]:
        """First ``boundary_spec`` index of each terminal among the given contact entries.

        Entries that name a common ``ChargeConductorMedium`` structure and carry the same
        drive are one terminal, as in the ``validate_one_bias_per_metal`` rule; any other
        entry is a terminal of its own.
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

    def _contact_conductors(self, idx: int) -> set[str]:
        """Return named conductors; empty means geometry must identify the terminal."""
        placement = self.boundary_spec[idx].placement
        if isinstance(placement, MediumMediumInterface):
            return set()
        return set(self._placement_conductor_structures(placement))

    def _check_semiconductor_electric_bcs_require_charge_analysis(self) -> Self:
        """Require an explicit charge analysis for electrical semiconductor simulations.

        The rule is deliberately not placement-aware: the simulation type classifier suppresses
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
        types of simulations. The construction path rejects inconsistent
        inputs, so this only reads them.
        """
        simulation_types = []

        if isinstance(self.analysis_spec, ChargeTypes):
            return [TCADAnalysisTypes.CHARGE]

        # check if unsteady heat
        if isinstance(self.analysis_spec, UnsteadyHeatAnalysis):
            return [TCADAnalysisTypes.HEAT]

        heat_source_present = any(isinstance(s, HeatSourceTypes) for s in self.sources)
        heat_BCs_present = any(isinstance(bc.condition, HeatBCTypes) for bc in self.boundary_spec)
        if heat_BCs_present or heat_source_present:
            simulation_types.append(TCADAnalysisTypes.HEAT)

        electric_spec_present = any(
            structure.medium.charge is not None for structure in self.structures
        )
        electric_BCs_present = any(
            isinstance(bc.condition, ElectricBCTypes) for bc in self.boundary_spec
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
