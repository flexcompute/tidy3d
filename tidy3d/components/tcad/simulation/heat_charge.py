# ruff: noqa: W293, W291
"""Defines heat simulation class"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Tuple, Union

import numpy as np
import pydantic.v1 as pd

try:
    from matplotlib import colormaps
except ImportError:
    pass

from tidy3d.components.base import skip_if_fields_missing
from tidy3d.components.base_sim.simulation import AbstractSimulation
from tidy3d.components.bc_placement import (
    MediumMediumInterface,
    SimulationBoundary,
    StructureBoundary,
    StructureSimulationBoundary,
    StructureStructureInterface,
)
from tidy3d.components.geometry.base import Box
from tidy3d.components.material.tcad.charge import (
    ChargeConductorMedium,
    SemiconductorMedium,
)
from tidy3d.components.material.tcad.heat import (
    SolidMedium,
)
from tidy3d.components.material.types import MultiPhysicsMedium, StructureMediumType
from tidy3d.components.medium import Medium
from tidy3d.components.scene import Scene
from tidy3d.components.spice.sources.dc import DCVoltageSource
from tidy3d.components.spice.types import ElectricalAnalysisType
from tidy3d.components.structure import Structure
from tidy3d.components.tcad.boundary.specification import (
    HeatBoundarySpec,
    HeatChargeBoundarySpec,
)
from tidy3d.components.tcad.grid import (
    DistanceUnstructuredGrid,
    UniformUnstructuredGrid,
    UnstructuredGridType,
)
from tidy3d.components.tcad.monitors.charge import (
    SteadyCapacitanceMonitor,
    SteadyFreeCarrierMonitor,
    SteadyPotentialMonitor,
)
from tidy3d.components.tcad.monitors.heat import (
    TemperatureMonitor,
)
from tidy3d.components.tcad.source.abstract import (
    GlobalHeatChargeSource,
)
from tidy3d.components.tcad.types import (
    ConvectionBC,
    CurrentBC,
    HeatChargeMonitorType,
    HeatChargeSourceType,
    HeatFluxBC,
    HeatFromElectricSource,
    HeatSource,
    InsulatingBC,
    TemperatureBC,
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
from tidy3d.components.types import TYPE_TAG_STR, Ax, Bound, ScalarSymmetry, Shapely, annotate_type
from tidy3d.components.viz import PlotParams, add_ax_if_none, equal_aspect
from tidy3d.constants import VOLUMETRIC_HEAT_RATE, inf
from tidy3d.exceptions import SetupError
from tidy3d.log import log

HEAT_CHARGE_BACK_STRUCTURE_STR = "<<<HEAT_CHARGE_BACKGROUND_STRUCTURE>>>"

HeatBCTypes = (TemperatureBC, HeatFluxBC, ConvectionBC)
HeatSourceTypes = (UniformHeatSource, HeatSource, HeatFromElectricSource)
ChargeSourceTypes = ()
ElectricBCTypes = (VoltageBC, CurrentBC, InsulatingBC)

AnalysisSpecType = ElectricalAnalysisType


class TCADAnalysisTypes(str, Enum):
    """Enumeration of the types of simulations currently supported"""

    HEAT = "Heat"
    CONDUCTION = "Conduction"
    CHARGE = "Charge"


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

            -k \\cdot \\nabla(T) = q


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
               specified boundary conditions such as ``SteadyVoltageBC``, ``SteadyCurrentBC``, ...
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
    ...     grid_spec=td.UniformUnstructuredGrid(dl=0.1),
    ...     sources=[td.HeatSource(rate=1, structures=["box"])],
    ...     boundary_spec=[
    ...         td.HeatChargeBoundarySpec(
    ...             placement=td.StructureBoundary(structure="box"),
    ...             condition=td.TemperatureBC(temperature=500),
    ...         )
    ...     ],
    ...     monitors=[td.TemperatureMonitor(size=(1, 2, 3), name="sample")],
    ... )

    To run a drift-diffusion (``Charge`` |:zap:|) system:

    >>> import tidy3d as td
    >>> air = td.FluidMedium(
    ...     name="air"
    ... )
    >>> intrinsic_Si = td.material_library['cSi'].variants['Si_MultiPhysics'].medium.charge
    >>> Si_n = intrinsic_Si.updated_copy(N_d=1e16, name="Si_n")
    >>> Si_p = intrinsic_Si.updated_copy(N_a=1e16, name="Si_p")
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
    ...     grid_spec=td.UniformUnstructuredGrid(dl=0.05),
    ...     boundary_spec=[bc_v1, bc_v2],
    ...     analysis_spec=td.IsothermalSteadyChargeDCAnalysis(
    ...         tolerance_settings=td.ChargeToleranceSpec(rel_tol=1e5, abs_tol=3e3, max_iters=400),
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

    medium: StructureMediumType = pd.Field(
        Medium(),
        title="Background Medium",
        description="Background medium of simulation, defaults to a standard dispersion-less :class:`Medium` if not "
        "specified.",
        discriminator=TYPE_TAG_STR,
    )
    """
    Background medium of simulation, defaults to a standard dispersion-less :class:`Medium` if not specified.
    """

    sources: Tuple[annotate_type(HeatChargeSourceType), ...] = pd.Field(
        (),
        title="Heat and Charge sources",
        description="List of heat and/or charge sources.",
    )

    monitors: Tuple[annotate_type(HeatChargeMonitorType), ...] = pd.Field(
        (),
        title="Monitors",
        description="Monitors in the simulation.",
    )

    boundary_spec: Tuple[annotate_type(Union[HeatChargeBoundarySpec, HeatBoundarySpec]), ...] = (
        pd.Field(
            (),
            title="Boundary Condition Specifications",
            description="List of boundary condition specifications.",
        )
    )
    # NOTE: creating a union with HeatBoundarySpec for backwards compatibility

    grid_spec: UnstructuredGridType = pd.Field(
        title="Grid Specification",
        description="Grid specification for heat-charge simulation.",
        discriminator=TYPE_TAG_STR,
    )

    symmetry: Tuple[ScalarSymmetry, ScalarSymmetry, ScalarSymmetry] = pd.Field(
        (0, 0, 0),
        title="Symmetries",
        description="Tuple of integers defining reflection symmetry across a plane "
        "bisecting the simulation domain normal to the x-, y-, and z-axis "
        "at the simulation center of each axis, respectively. "
        "Each element can be ``0`` (symmetry off) or ``1`` (symmetry on).",
    )

    analysis_spec: AnalysisSpecType = pd.Field(
        None,
        title="Analysis specification.",
        description="The `analysis_spec` is used to validate that the simulation parameters and tolerance settings "
        "are correctly configured as desired by the user.",
    )

    @pd.validator("structures", always=True)
    def check_unsupported_geometries(cls, val):
        """Error if structures contain unsupported yet geometries."""
        for ind, structure in enumerate(val):
            bbox = structure.geometry.bounding_box
            if any(s == 0 for s in bbox.size):
                raise SetupError(
                    f"'HeatSimulation' does not currently support structures with dimensions of zero size ('structures[{ind}]')."
                )
        return val

    @staticmethod
    def _check_cross_solids(objs: Tuple[Box, ...], values: Dict) -> Tuple[int, ...]:
        """Given model dictionary ``values``, check whether objects in list ``objs`` cross
        a ``SolidSpec`` medium.
        """

        # NOTE: when considering Conduction or Charge cases, both conductors and semiconductors
        # will be accepted
        valid_electric_medium = (SemiconductorMedium, ChargeConductorMedium)

        try:
            size = values["size"]
            center = values["center"]
            medium = values["medium"]
            structures = values["structures"]
        except KeyError:
            raise SetupError(
                "Function '_check_cross_solids' assumes dictionary 'values' contains well-defined "
                "'size', 'center',  'medium', and 'structures'. Thus, it should only be used in "
                "validators with @skip_if_fields_missing(['medium', 'center', 'size', 'structures']) "
                "or root validators with option 'skip_on_failure=True'."
            )

        # list of structures including background as a Box()
        structure_bg = Structure(
            geometry=Box(
                size=size,
                center=center,
            ),
            medium=medium,
        )

        total_structures = [structure_bg] + list(structures)

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
                    any([isinstance(medium.charge, medium_i)] for medium_i in valid_electric_medium)
                    for medium in medium_set
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
                    if any(
                        [isinstance(structure.medium.charge, medium_i)]
                        for medium_i in valid_electric_medium
                    )
                )

            if not crosses_solid:
                obj_do_not_cross_solid_idx.append(ind)
            if not crosses_elec_spec:
                obj_do_not_cross_cond_idx.append(ind)

        return obj_do_not_cross_solid_idx, obj_do_not_cross_cond_idx

    @pd.validator("monitors", always=True)
    @skip_if_fields_missing(["medium", "center", "size", "structures"])
    def _monitors_cross_solids(cls, val, values):
        """Error if monitors does not cross any solid medium."""

        if val is None:
            return val

        failed_solid_idx, failed_elect_idx = cls._check_cross_solids(val, values)

        temp_monitors = [idx for idx, mnt in enumerate(val) if isinstance(mnt, TemperatureMonitor)]
        volt_monitors = [
            idx for idx, mnt in enumerate(val) if isinstance(mnt, SteadyPotentialMonitor)
        ]

        failed_temp_mnt = [idx for idx in temp_monitors if idx in failed_solid_idx]
        failed_volt_mnt = [idx for idx in volt_monitors if idx in failed_elect_idx]

        if len(failed_temp_mnt) > 0:
            monitor_names = [f"'{val[ind].name}'" for ind in failed_temp_mnt]
            raise SetupError(
                f"Monitors {monitor_names} do not cross any solid materials "
                "('heat_spec=SolidSpec(...)'). Temperature distribution is only recorded inside solid "
                "materials. Thus, no information will be recorded in these monitors."
            )

        if len(failed_volt_mnt) > 0:
            monitor_names = [f"'{val[ind].name}'" for ind in failed_volt_mnt]
            raise SetupError(
                f"Monitors {monitor_names} do not cross any conducting materials "
                "('charge=ChargeConductorMedium(...)'). The voltage is only stored inside conducting "
                "materials. Thus, no information will be recorded in these monitors."
            )

        return val

    @pd.root_validator(skip_on_failure=True)
    def check_voltage_array_if_capacitance(cls, values):
        """Make sure an array of voltages has been defined if a
        SteadyCapacitanceMonitor' has been defined"""
        bounday_spec = values["boundary_spec"]
        monitors = values["monitors"]

        is_capacitance_mnt = any(isinstance(mnt, SteadyCapacitanceMonitor) for mnt in monitors)
        voltage_array_present = False
        if is_capacitance_mnt:
            for bc in bounday_spec:
                if isinstance(bc.condition, VoltageBC):
                    if isinstance(bc.condition.source, DCVoltageSource):
                        if isinstance(bc.condition.source.voltage, list) or isinstance(
                            bc.condition.source.voltage, tuple
                        ):
                            if len(bc.condition.source.voltage) > 1:
                                voltage_array_present = True
        if is_capacitance_mnt and not voltage_array_present:
            raise SetupError(
                "Monitors of type 'SteadyCapacitanceMonitor' have been defined but no array of voltages "
                "has been supplied as voltage source, which is required for this type of monitor. "
                "Voltage arrays can be included in a source in this manner: "
                "'VoltageBC(source=DCVoltageSource(voltage=yourArray))'"
            )
        return values

    @pd.validator("size", always=True)
    def check_zero_dim_domain(cls, val, values):
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

    @pd.validator("boundary_spec", always=True)
    @skip_if_fields_missing(["structures", "medium"])
    def names_exist_bcs(cls, val, values):
        """Error if boundary conditions point to non-existing structures/media."""

        structures = values.get("structures")
        structures_names = {s.name for s in structures}
        mediums_names = {s.medium.name for s in structures}
        mediums_names.add(values.get("medium").name)

        for bc_ind, bc_spec in enumerate(val):
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
        return val

    @pd.validator("boundary_spec", always=True)
    def check_only_one_voltage_array_provided(cls, val, values):
        """Issue error if more than one voltage array is provided.
        Currently we only allow to sweep over one voltage array.
        """

        array_already_provided = False

        for bc in val:
            if isinstance(bc.condition, VoltageBC):
                voltages = []
                # currently we're only supporting DC BCs, so let's check these values
                if isinstance(bc.condition.source, DCVoltageSource):
                    voltages = bc.condition.source.voltage

                if isinstance(voltages, tuple):
                    if len(voltages) > 1:
                        if not array_already_provided:
                            array_already_provided = True
                        else:
                            raise SetupError(
                                "More than one voltage array has been provided. "
                                "Currently voltage arrays are supported only for one of the BCs."
                            )
        return val

    @pd.root_validator(skip_on_failure=True)
    def check_charge_simulation(cls, values):
        """Makes sure that Charge simulations are set correctly."""

        ChargeMonitorType = (
            SteadyPotentialMonitor,
            SteadyFreeCarrierMonitor,
            SteadyCapacitanceMonitor,
        )

        simulation_types = cls._check_simulation_types(values=values)

        if TCADAnalysisTypes.CHARGE in simulation_types:
            # check that we have at least 2 'VoltageBC's
            boundary_spec = values["boundary_spec"]
            voltage_bcs = 0
            for bc in boundary_spec:
                if isinstance(bc.condition, VoltageBC):
                    voltage_bcs = voltage_bcs + 1
            if voltage_bcs < 2:
                raise SetupError(
                    "Defining a Charge simulation requires the definition of 'VoltageBC' boundaries. "
                    f"So far {voltage_bcs} 'VoltageBC' have been set."
                )

            # check that we have at least one charge monitor
            monitors = values["monitors"]
            if not any(isinstance(mnt, ChargeMonitorType) for mnt in monitors):
                raise SetupError(
                    "Charge simulations require the definition of, at least, one of these monitors: "
                    "'[SteadyPotentialMonitor, SteadyFreeCarrierMonitor, SteadyCapacitanceMonitor]' "
                    "but none have been defined."
                )

            # NOTE: SteadyPotentialMonitor and SteadyFreeCarrierMonitor are only supported
            # for unstructured = True
            for mnt in monitors:
                if isinstance(mnt, SteadyPotentialMonitor) or isinstance(
                    mnt, SteadyFreeCarrierMonitor
                ):
                    if not mnt.unstructured:
                        log.warning(
                            "Currently, Charge simulations support only unstructured monitors. Please set "
                            f"monitor '{mnt.name}' to 'unstructured = True'."
                        )

        return values

    @pd.root_validator(skip_on_failure=True)
    def not_all_neumann(cls, values):
        """Make sure not all BCs are of Neumann type"""

        NeumannBCsHeat = (HeatFluxBC,)
        NeumannBCsCharge = (CurrentBC, InsulatingBC)

        simulation_types = cls._check_simulation_types(values=values)
        bounday_conditions = values["boundary_spec"]

        raise_error = False
        for sim_type in simulation_types:
            if sim_type == TCADAnalysisTypes.HEAT:
                type_bcs = [
                    bc for bc in bounday_conditions if isinstance(bc.condition, HeatBCTypes)
                ]
                if len(type_bcs) == 0 or all(
                    isinstance(bc.condition, NeumannBCsHeat) for bc in type_bcs
                ):
                    raise_error = True
            elif sim_type == TCADAnalysisTypes.CONDUCTION:
                type_bcs = [
                    bc for bc in bounday_conditions if isinstance(bc.condition, ElectricBCTypes)
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

        return values

    @pd.validator("grid_spec", always=True)
    @skip_if_fields_missing(["structures"])
    def names_exist_grid_spec(cls, val, values):
        """Warn if 'UniformUnstructuredGrid' points at a non-existing structure."""

        structures = values.get("structures")
        structures_names = {s.name for s in structures}

        for structure_name in val.non_refined_structures:
            if structure_name not in structures_names:
                log.warning(
                    f"Structure '{structure_name}' listed as a non-refined structure in "
                    "'HeatChargeSimulation.grid_spec' is not present in 'HeatChargeSimulation.structures'"
                )

        return val

    @pd.validator("grid_spec", always=True)
    def warn_if_minimal_mesh_size_override(cls, val, values):
        """Warn if minimal mesh size limit overrides desired mesh size."""

        max_size = np.max(values.get("size"))
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

        return val

    @pd.validator("sources", always=True)
    @skip_if_fields_missing(["structures"])
    def names_exist_sources(cls, val, values):
        """Error if a heat-charge source point to non-existing structures."""
        structures = values.get("structures")
        structures_names = {s.name for s in structures}

        sources = [s for s in val if not isinstance(s, HeatFromElectricSource)]

        for source in sources:
            for name in source.structures:
                if name not in structures_names:
                    raise SetupError(
                        f"Structure '{name}' provided in a '{source.type}' "
                        "is not found among simulation structures."
                    )
        return val

    @pd.root_validator(skip_on_failure=True)
    def check_medium_specs(cls, values):
        """Error if no appropriate specs."""

        sim_box = (
            Box(
                size=values.get("size"),
                center=values.get("center"),
            ),
        )

        failed_solid_idx, failed_elect_idx = cls._check_cross_solids(sim_box, values)

        simulation_types = cls._check_simulation_types(values=values)

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

        return values

    @staticmethod
    def _check_if_semiconductor_present(structures) -> bool:
        """Checks whether the simulation object can run a Charge simulation."""

        charge_sim = False

        # make sure mediums with doping have been defined
        for structure in structures:
            if isinstance(structure.medium, SemiconductorMedium):
                charge_sim = True
            if isinstance(structure.medium, MultiPhysicsMedium):
                if structure.medium.charge is not None:
                    if isinstance(structure.medium.charge, SemiconductorMedium):
                        charge_sim = True
        return charge_sim

    @staticmethod
    def _check_simulation_types(
        values: Dict,
        HeatBCTypes=HeatBCTypes,
        ElectricBCTypes=ElectricBCTypes,
        HeatSourceTypes=HeatSourceTypes,
    ) -> list[TCADAnalysisTypes]:
        """Given model dictionary ``values``, check the type of simulations to be run
        based on BCs and sources.
        """
        simulation_types = []

        boundaries = list(values["boundary_spec"])
        sources = list(values["sources"])

        structures = list(values["structures"])
        semiconductor_present = HeatChargeSimulation._check_if_semiconductor_present(
            structures=structures
        )
        if semiconductor_present:
            simulation_types.append(TCADAnalysisTypes.CHARGE)

        for boundary in boundaries:
            if isinstance(boundary.condition, HeatBCTypes):
                simulation_types.append(TCADAnalysisTypes.HEAT)
            if isinstance(boundary.condition, ElectricBCTypes):
                # for the time being, assume tha the simulation will be of
                # type CHARGE if we have semiconductors
                if semiconductor_present:
                    simulation_types.append(TCADAnalysisTypes.CHARGE)
                else:
                    simulation_types.append(TCADAnalysisTypes.CONDUCTION)

        for source in sources:
            if isinstance(source, HeatSourceTypes):
                simulation_types.append(TCADAnalysisTypes.HEAT)

        return set(simulation_types)

    @pd.root_validator(skip_on_failure=True)
    def check_coupling_source_can_be_applied(cls, values):
        """Error if material doesn't have the right specifications"""

        HeatSourceTypes_noCoupling = (UniformHeatSource, HeatSource)

        simulation_types = cls._check_simulation_types(
            values, HeatSourceTypes=HeatSourceTypes_noCoupling
        )
        simulation_types = list(simulation_types)

        sources = list(values["sources"])

        for source in sources:
            if isinstance(source, HeatFromElectricSource) and len(simulation_types) < 2:
                raise SetupError(
                    f"Using 'HeatFromElectricSource' requires the definition of both "
                    f"{TCADAnalysisTypes.CONDUCTION.name} and {TCADAnalysisTypes.HEAT.name}. "
                    f"The current simulation setup contains only conditions of type {simulation_types[0].name}"
                )

        return values

    @pd.root_validator(skip_on_failure=True)
    def estimate_charge_mesh_size(cls, values):
        """Make an estimate of the mesh size and raise a warning if too big.
        NOTE: this is a very rough estimate. The back-end will actually stop
        execution based on actual node-count."""

        if TCADAnalysisTypes.CHARGE not in cls._check_simulation_types(values=values):
            return values

        # let's raise a warning if the estimate is larger than 2M nodes
        max_nodes = 2e6
        nodes_estimate = 0

        structures = values["structures"]
        grid_spec = values["grid_spec"]

        non_refined_structures = grid_spec.non_refined_structures

        if isinstance(grid_spec, UniformUnstructuredGrid):
            dl_min = grid_spec.dl
            dl_max = dl_min
        elif isinstance(grid_spec, DistanceUnstructuredGrid):
            dl_min = grid_spec.dl_interface
            dl_max = grid_spec.dl_bulk

        for struct in structures:
            name = struct.name
            bounds = struct.geometry.bounds
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
                "WARNING: It has been estimated the mesh to be bigger than the currently "
                "supported mesh size for Charge simulations. The simulation may be "
                "submitted but if the maximum number of nodes is indeed exceeded "
                "the pipeline will be stopped. If this happens the grid specification "
                "may need to be modified."
            )
        return values

    @equal_aspect
    @add_ax_if_none
    def plot_property(
        self,
        x: float = None,
        y: float = None,
        z: float = None,
        ax: Ax = None,
        alpha: float = None,
        source_alpha: float = None,
        monitor_alpha: float = None,
        property: str = "heat_conductivity",
        hlim: Tuple[float, float] = None,
        vlim: Tuple[float, float] = None,
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
        hlim : Tuple[float, float] = None
            The x range if plotting on xy or xz planes, y range if plotting on yz plane.
        vlim : Tuple[float, float] = None
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
    def plot_heat_conductivity(
        self,
        x: float = None,
        y: float = None,
        z: float = None,
        ax: Ax = None,
        alpha: float = None,
        source_alpha: float = None,
        monitor_alpha: float = None,
        colorbar: str = "conductivity",
        hlim: Tuple[float, float] = None,
        vlim: Tuple[float, float] = None,
        **kwargs,
    ) -> Ax:
        """
        DEPRECATED: Method added for backwards compatibility with :class:`HeatSimulation.plot_heat_conductivity`.
        Plot each of simulation's components on a plane defined by one nonzero x,y,z coordinate.

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
        colorbar: str = "conductivity"
            Display colorbar for thermal conductivity ("conductivity") or heat source rate
            ("source").
        hlim : Tuple[float, float] = None
            The x range if plotting on xy or xz planes, y range if plotting on yz plane.
        vlim : Tuple[float, float] = None
            The z range if plotting on xz or yz planes, y plane if plotting on xy plane.

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.
        """
        log.warning(
            """This function `plot_heat_conductivity` is 
            deprecated and will be discontinued. In its place you can use
            `plot_property(property="heat_conductivity")`"""
        )

        plot_type = "heat_conductivity"
        if colorbar == "conductivity":
            plot_type = "heat_conductivity"
        elif colorbar == "source":
            plot_type = "source"

        return self.plot_property(
            x=x,
            y=y,
            z=z,
            ax=ax,
            alpha=alpha,
            source_alpha=source_alpha,
            monitor_alpha=monitor_alpha,
            property=plot_type,
            hlim=hlim,
            vlim=vlim,
        )

    @equal_aspect
    @add_ax_if_none
    def plot_boundaries(
        self,
        x: float = None,
        y: float = None,
        z: float = None,
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
        structures += list(self.structures)

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

        if isinstance(condition, TemperatureBC):
            plot_params = plot_params.updated_copy(facecolor=HEAT_BC_COLOR_TEMPERATURE)
        elif isinstance(condition, HeatFluxBC):
            plot_params = plot_params.updated_copy(facecolor=HEAT_BC_COLOR_FLUX)
        elif isinstance(condition, ConvectionBC):
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
        structures: Tuple[Structure, ...],
        boundary_spec: Tuple[HeatChargeBoundarySpec, ...],
    ) -> Dict[str, HeatChargeBoundarySpec]:
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
        structures: Tuple[Structure, ...],
        boundary_spec: Tuple[HeatChargeBoundarySpec, ...],
    ) -> Dict[str, HeatChargeBoundarySpec]:
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
        shapes: Tuple[Tuple[str, str, Shapely, Tuple[float, float, float, float]], ...],
        struct_to_bc_spec: Dict[str, HeatChargeBoundarySpec],
        med_to_bc_spec: Dict[str, HeatChargeBoundarySpec],
        background_structure_shape: Shapely,
    ) -> Tuple[Tuple[HeatChargeBoundarySpec, Shapely], ...]:
        """Construct Simulation, StructureSimulation, Structure, and MediumMedium boundaries."""

        # forward foop to take care of Simulation, StructureSimulation, Structure,
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
        shapes: Tuple[Tuple[str, str, Shapely, Bound], ...],
        struct_to_bc_spec: Dict[str, HeatChargeBoundarySpec],
        background_structure_shape: Shapely,
    ) -> Tuple[Tuple[HeatChargeBoundarySpec, Shapely], ...]:
        """Construct StructureStructure boundaries."""

        # backward foop to take care of StructureStructure
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
            minx, miny, maxx, maxy = bounds

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
        structures: List[Structure],
        plane: Box,
        boundary_spec: List[HeatChargeBoundarySpec],
    ) -> List[Tuple[HeatChargeBoundarySpec, Shapely]]:
        """Compute list of boundary lines to plot on plane.

        Parameters
        ----------
        structures : List[:class:`.Structure`]
            list of structures to filter on the plane.
        plane : :class:`.Box`
            target plane.
        boundary_spec : List[HeatBoundarySpec]
            list of boundary conditions associated with structures.

        Returns
        -------
        List[Tuple[:class:`.HeatBoundarySpec`, shapely.geometry.base.BaseGeometry]]
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

        # 1. forward foop to take care of Simulation, StructureSimulation, Structure,
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
        x: float = None,
        y: float = None,
        z: float = None,
        property: str = "heat_conductivity",
        hlim: Tuple[float, float] = None,
        vlim: Tuple[float, float] = None,
        alpha: float = None,
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
        hlim : Tuple[float, float] = None
            The x range if plotting on xy or xz planes, y range if plotting on yz plane.
        vlim : Tuple[float, float] = None
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
        structures = self.structures

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
            structures=structures, plane=plane, property_list=source_list
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

    def _add_source_cbar(self, ax: Ax, property: str = "heat_conductivity"):
        """Add colorbar for heat sources."""
        source_min, source_max = self.source_bounds(property=property)
        self.scene._add_cbar(
            vmin=source_min,
            vmax=source_max,
            label=f"Volumetric heat rate ({VOLUMETRIC_HEAT_RATE})",
            cmap=HEAT_SOURCE_CMAP,
            ax=ax,
        )

    def _safe_float_conversion(self, string) -> float:
        """Function to deal with failed string2float conversion when using
        expressions in 'HeatSource'"""
        try:
            return float(string)
        except ValueError:
            return None

    def source_bounds(self, property: str = "heat_conductivity") -> Tuple[float, float]:
        """Compute range of heat sources present in the simulation."""

        if property == "heat_conductivity" or property == "source":
            rate_list = [
                self._safe_float_conversion(source.rate)
                for source in self.sources
                if isinstance(source, HeatSource)
            ]
        elif property == "electric_conductivity":
            rate_list = [
                self._safe_float_conversion(source.rate)
                for source in self.sources
                if isinstance(source, ChargeSourceTypes)
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
        alpha: float = None,
    ) -> PlotParams:
        """Constructs the plot parameters for a given medium in simulation.plot_eps()."""

        plot_params = plot_params_heat_source
        if alpha is not None:
            plot_params = plot_params.copy(update={"alpha": alpha})

        if isinstance(source, HeatSource):
            rate = self._safe_float_conversion(source.rate)
            if rate is not None:
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
        alpha: float = None,
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
    def from_scene(cls, scene: Scene, **kwargs) -> HeatChargeSimulation:
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
        >>> from tidy3d import Scene, Medium, Box, Structure, UniformUnstructuredGrid
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
        ...     grid_spec=UniformUnstructuredGrid(dl=0.4),
        ...     boundary_spec=[
        ...         HeatChargeBoundarySpec(
        ...             placement=StructureBoundary(structure="box"),
        ...             condition=TemperatureBC(temperature=500),
        ...         )
        ...     ],
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

        # NOTE: for the time being, if a simulation has SemiconductorMedium
        # then we consider it of being a 'TCADAnalysisTypes.CHARGE'
        if self._check_if_semiconductor_present(self.structures):
            return [TCADAnalysisTypes.CHARGE]

        heat_source_present = any(isinstance(s, HeatSourceTypes) for s in self.sources)

        heat_BCs_present = any(isinstance(bc.condition, HeatBCTypes) for bc in self.boundary_spec)

        if heat_source_present and not heat_BCs_present:
            raise SetupError("Heat sources defined but no heat BCs present.")
        elif heat_BCs_present or heat_source_present:
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
        elif electric_BCs_present and electric_spec_present:
            simulation_types.append(TCADAnalysisTypes.CONDUCTION)

        return simulation_types

    def _useHeatSourceFromConductionSim(self):
        """Returns True if 'HeatFromElectricSource' has been defined."""

        return any(isinstance(source, HeatFromElectricSource) for source in self.sources)
