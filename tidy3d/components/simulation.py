""" Container holding all information about simulation and its components"""
from __future__ import annotations

from typing import Dict, Tuple, List, Set, Union

import pydantic.v1 as pydantic
import numpy as np
import xarray as xr
import matplotlib as mpl

from .base import cached_property
from .validators import assert_objects_in_sim_bounds
from .validators import validate_mode_objects_symmetry
from .geometry.base import Geometry, Box
from .geometry.primitives import Cylinder
from .geometry.mesh import TriangleMesh
from .geometry.polyslab import PolySlab
from .geometry.utils import flatten_groups, traverse_geometries
from .types import Ax, FreqBound, Axis, annotate_type, InterpMethod
from .grid.grid import Coords1D, Grid, Coords
from .grid.grid_spec import GridSpec, UniformGrid, AutoGrid
from .medium import Medium, MediumType, AbstractMedium
from .medium import AbstractCustomMedium, Medium2D, MediumType3D
from .medium import AnisotropicMedium, FullyAnisotropicMedium, AbstractPerturbationMedium
from .boundary import BoundarySpec, BlochBoundary, PECBoundary, PMCBoundary, Periodic
from .boundary import PML, StablePML, Absorber, AbsorberSpec
from .structure import Structure
from .source import SourceType, PlaneWave, GaussianBeam, AstigmaticGaussianBeam, CustomFieldSource
from .source import CustomCurrentSource, CustomSourceTime, ContinuousWave
from .source import TFSF, Source, ModeSource
from .monitor import MonitorType, Monitor, FreqMonitor, SurfaceIntegrationMonitor
from .monitor import AbstractModeMonitor, FieldMonitor
from .monitor import PermittivityMonitor, DiffractionMonitor, AbstractFieldProjectionMonitor
from .data.dataset import Dataset
from .data.data_array import SpatialDataArray
from .viz import add_ax_if_none, equal_aspect
from .scene import Scene

from .viz import PlotParams
from .viz import plot_params_pml, plot_params_override_structures
from .viz import plot_params_pec, plot_params_pmc, plot_params_bloch, plot_sim_3d

from ..constants import C_0, SECOND, fp_eps
from ..exceptions import SetupError, ValidationError, Tidy3dError, Tidy3dImportError
from ..log import log
from ..updater import Updater

from .base_sim.simulation import AbstractSimulation

try:
    gdstk_available = True
    import gdstk
except ImportError:
    gdstk_available = False

try:
    gdspy_available = True
    import gdspy
except ImportError:
    gdspy_available = False


# minimum number of grid points allowed per central wavelength in a medium
MIN_GRIDS_PER_WVL = 6.0

# maximum number of sources
MAX_NUM_SOURCES = 1000

# maximum numbers of simulation parameters
MAX_TIME_STEPS = 1e7
WARN_TIME_STEPS = 1e6
MAX_GRID_CELLS = 20e9
MAX_CELLS_TIMES_STEPS = 1e16
WARN_MONITOR_DATA_SIZE_GB = 10
MAX_MONITOR_INTERNAL_DATA_SIZE_GB = 50
MAX_SIMULATION_DATA_SIZE_GB = 50
WARN_MODE_NUM_CELLS = 1e5

# number of grid cells at which we warn about slow Simulation.epsilon()
NUM_CELLS_WARN_EPSILON = 100_000_000
# number of structures at which we warn about slow Simulation.epsilon()
NUM_STRUCTURES_WARN_EPSILON = 10_000
# for 2d materials. to find neighboring media, search a distance on either side
# equal to this times the grid size
DIST_NEIGHBOR_REL_2D_MED = 1e-5

# height of the PML plotting boxes along any dimensions where sim.size[dim] == 0
PML_HEIGHT_FOR_0_DIMS = 0.02


class Simulation(AbstractSimulation):
    """Contains all information about Tidy3d simulation.

    Example
    -------
    >>> from tidy3d import Sphere, Cylinder, PolySlab
    >>> from tidy3d import UniformCurrentSource, GaussianPulse
    >>> from tidy3d import FieldMonitor, FluxMonitor
    >>> from tidy3d import GridSpec, AutoGrid
    >>> from tidy3d import BoundarySpec, Boundary
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
    """

    run_time: pydantic.PositiveFloat = pydantic.Field(
        ...,
        title="Run Time",
        description="Total electromagnetic evolution time in seconds. "
        "Note: If simulation 'shutoff' is specified, "
        "simulation will terminate early when shutoff condition met. ",
        units=SECOND,
    )

    sources: Tuple[annotate_type(SourceType), ...] = pydantic.Field(
        (),
        title="Sources",
        description="Tuple of electric current sources injecting fields into the simulation.",
    )

    boundary_spec: BoundarySpec = pydantic.Field(
        BoundarySpec(),
        title="Boundaries",
        description="Specification of boundary conditions along each dimension. If ``None``, "
        "PML boundary conditions are applied on all sides.",
    )

    monitors: Tuple[annotate_type(MonitorType), ...] = pydantic.Field(
        (),
        title="Monitors",
        description="Tuple of monitors in the simulation. "
        "Note: monitor names are used to access data after simulation is run.",
    )

    grid_spec: GridSpec = pydantic.Field(
        GridSpec(),
        title="Grid Specification",
        description="Specifications for the simulation grid along each of the three directions.",
    )

    shutoff: pydantic.NonNegativeFloat = pydantic.Field(
        1e-5,
        title="Shutoff Condition",
        description="Ratio of the instantaneous integrated E-field intensity to the maximum value "
        "at which the simulation will automatically terminate time stepping. "
        "Used to prevent extraneous run time of simulations with fully decayed fields. "
        "Set to ``0`` to disable this feature.",
    )

    subpixel: bool = pydantic.Field(
        True,
        title="Subpixel Averaging",
        description="If ``True``, uses subpixel averaging of the permittivity "
        "based on structure definition, resulting in much higher accuracy for a given grid size.",
    )

    normalize_index: Union[pydantic.NonNegativeInt, None] = pydantic.Field(
        0,
        title="Normalization index",
        description="Index of the source in the tuple of sources whose spectrum will be used to "
        "normalize the frequency-dependent data. If ``None``, the raw field data is returned "
        "unnormalized.",
    )

    courant: float = pydantic.Field(
        0.99,
        title="Courant Factor",
        description="Courant stability factor, controls time step to spatial step ratio. "
        "Lower values lead to more stable simulations for dispersive materials, "
        "but result in longer simulation times. This factor is normalized to no larger than 1 "
        "when CFL stability condition is met in 3D.",
        gt=0.0,
        le=1.0,
    )

    """ Validating setup """

    @pydantic.root_validator(pre=True)
    def _update_simulation(cls, values):
        """Update the simulation if it is an earlier version."""

        # if no version, assume it's already updated
        if "version" not in values:
            return values

        # otherwise, call the updator to update the values dictionary
        updater = Updater(sim_dict=values)
        return updater.update_to_current()

    @pydantic.validator("grid_spec", always=True)
    def _validate_auto_grid_wavelength(cls, val, values):
        """Check that wavelength can be defined if there is auto grid spec."""
        if val.wavelength is None and val.auto_grid_used:
            _ = val.wavelength_from_sources(sources=values.get("sources"))
        return val

    _sources_in_bounds = assert_objects_in_sim_bounds("sources")
    _mode_sources_symmetries = validate_mode_objects_symmetry("sources")
    _mode_monitors_symmetries = validate_mode_objects_symmetry("monitors")

    # _few_enough_mediums = validate_num_mediums()
    # _structures_not_at_edges = validate_structure_bounds_not_at_edges()
    # _gap_size_ok = validate_pml_gap_size()
    # _medium_freq_range_ok = validate_medium_frequency_range()
    # _resolution_fine_enough = validate_resolution()
    # _plane_waves_in_homo = validate_plane_wave_intersections()

    @pydantic.validator("boundary_spec", always=True)
    def bloch_with_symmetry(cls, val, values):
        """Error if a Bloch boundary is applied with symmetry"""
        boundaries = val.to_list
        symmetry = values.get("symmetry")
        for dim, boundary in enumerate(boundaries):
            num_bloch = sum(isinstance(bnd, BlochBoundary) for bnd in boundary)
            if num_bloch > 0 and symmetry[dim] != 0:
                raise SetupError(
                    f"Bloch boundaries cannot be used with a symmetry along dimension {dim}."
                )
        return val

    @pydantic.validator("boundary_spec", always=True)
    def plane_wave_boundaries(cls, val, values):
        """Error if there are plane wave sources incompatible with boundary conditions."""
        boundaries = val.to_list
        sources = values.get("sources")
        size = values.get("size")
        sim_medium = values.get("medium")
        structures = values.get("structures")
        for source_ind, source in enumerate(sources):
            if not isinstance(source, PlaneWave):
                continue

            _, tan_dirs = cls.pop_axis([0, 1, 2], axis=source.injection_axis)
            medium_set = Scene.intersecting_media(source, structures)
            medium = medium_set.pop() if medium_set else sim_medium

            for tan_dir in tan_dirs:
                boundary = boundaries[tan_dir]

                # check the PML/absorber + angled plane wave case
                num_pml = sum(isinstance(bnd, AbsorberSpec) for bnd in boundary)
                if num_pml > 0 and source.angle_theta != 0:
                    raise SetupError(
                        "Angled plane wave sources are not compatible with the absorbing boundary "
                        f"along dimension {tan_dir}. Either set the source ``angle_theta`` to "
                        "``0``, or use Bloch boundaries that match the source angle."
                    )

                # check the Bloch boundary + angled plane wave case
                num_bloch = sum(isinstance(bnd, BlochBoundary) for bnd in boundary)
                if num_bloch > 0:
                    cls._check_bloch_vec(
                        source=source,
                        source_ind=source_ind,
                        bloch_vec=boundary[0].bloch_vec,
                        dim=tan_dir,
                        medium=medium,
                        domain_size=size[tan_dir],
                    )
        return val

    @pydantic.validator("boundary_spec", always=True)
    def tfsf_boundaries(cls, val, values):
        """Error if the boundary conditions are compatible with TFSF sources, if any."""
        boundaries = val.to_list
        sources = values.get("sources")
        size = values.get("size")
        center = values.get("center")
        sim_medium = values.get("medium")
        structures = values.get("structures")
        sim_bounds = [
            [c - s / 2.0 for c, s in zip(center, size)],
            [c + s / 2.0 for c, s in zip(center, size)],
        ]
        for src_idx, source in enumerate(sources):
            if not isinstance(source, TFSF):
                continue

            norm_dir, tan_dirs = cls.pop_axis([0, 1, 2], axis=source.injection_axis)
            src_bounds = source.bounds

            # make a dummy source that represents the injection surface to get the intersecting
            # medium, which is later used to test the Bloch vector for correctness
            temp_size = list(source.size)
            temp_size[source.injection_axis] = 0
            temp_src = Source(
                center=source.injection_plane_center,
                size=temp_size,
                source_time=source.source_time,
            )
            medium_set = Scene.intersecting_media(temp_src, structures)
            medium = medium_set.pop() if medium_set else sim_medium

            # the source shouldn't touch or cross any boundary in the direction of injection
            if (
                src_bounds[0][norm_dir] <= sim_bounds[0][norm_dir]
                or src_bounds[1][norm_dir] >= sim_bounds[1][norm_dir]
            ):
                raise SetupError(
                    f"The TFSF source at index '{src_idx}' must not touch or cross the "
                    f"simulation boundary along its injection axis, '{['x', 'y', 'z'][norm_dir]}'."
                )

            for tan_dir in tan_dirs:
                boundary = boundaries[tan_dir]

                # if the boundary is periodic, the source is allowed to cross the boundary
                # so nothing needs to be done
                num_pbc = sum(isinstance(bnd, Periodic) for bnd in boundary)
                if num_pbc == 2:
                    continue

                # crossing may be allowed for Bloch boundaries, but not others
                if (
                    src_bounds[0][tan_dir] <= sim_bounds[0][tan_dir]
                    or src_bounds[1][tan_dir] >= sim_bounds[1][tan_dir]
                ):
                    # if the boundary is Bloch periodic, crossing is allowed, but check that the
                    # Bloch vector has been correctly set, similar to the check for plane waves
                    num_bloch = sum(isinstance(bnd, BlochBoundary) for bnd in boundary)
                    if num_bloch == 2:
                        cls._check_bloch_vec(
                            source=source,
                            source_ind=src_idx,
                            bloch_vec=boundary[0].bloch_vec,
                            dim=tan_dir,
                            medium=medium,
                            domain_size=size[tan_dir],
                        )
                        continue

                    # for any other boundary, the source must not cross the boundary
                    raise SetupError(
                        f"The TFSF source at index '{src_idx}' must not touch or cross the "
                        f"simulation boundary in the '{['x', 'y', 'z'][tan_dir]}' direction, "
                        "unless that boundary is 'Periodic' or 'BlochBoundary'."
                    )

        return val

    @pydantic.validator("sources", always=True)
    def tfsf_with_symmetry(cls, val, values):
        """Error if a TFSF source is applied with symmetry"""
        symmetry = values.get("symmetry")
        for source in val:
            if isinstance(source, TFSF) and not all(sym == 0 for sym in symmetry):
                raise SetupError("TFSF sources cannot be used with symmetries.")
        return val

    @pydantic.validator("boundary_spec", always=True)
    def boundaries_for_zero_dims(cls, val, values):
        """Error if an absorbing boundary is used along a zero dimension."""
        boundaries = val.to_list
        size = values.get("size")
        for dim, (boundary, size_dim) in enumerate(zip(boundaries, size)):
            num_absorbing_bdries = sum(isinstance(bnd, AbsorberSpec) for bnd in boundary)
            if num_absorbing_bdries > 0 and size_dim == 0:
                raise SetupError(
                    f"The simulation has zero size along the {'xyz'[dim]} axis, so "
                    "using a PML or absorbing boundary along that axis is incorrect. "
                    f"Use either 'Periodic' or 'BlochBoundary' along {'xyz'[dim]}."
                )
        return val

    @pydantic.validator("sources", always=True)
    def _validate_num_sources(cls, val):
        """Error if too many sources present."""

        if val is None:
            return val

        if len(val) > MAX_NUM_SOURCES:
            raise SetupError(
                f"Number of distinct sources exceeds the maximum allowed {MAX_NUM_SOURCES}. "
                "For a complex source setup, consider using 'CustomFieldSource' or "
                "'CustomCurrentSource' to combine multiple sources into one object."
            )

        return val

    @pydantic.validator("structures", always=True)
    def _validate_2d_geometry_has_2d_medium(cls, val, values):
        """Warn if a geometry bounding box has zero size in a certain dimension."""

        if val is None:
            return val

        for i, structure in enumerate(val):
            if isinstance(structure.medium, Medium2D):
                continue
            for geom in flatten_groups(structure.geometry):
                zero_dims = geom.zero_dims
                if len(zero_dims) > 0:
                    log.warning(
                        f"Structure at 'structures[{i}]' has geometry with zero size along "
                        f"dimensions {zero_dims}, and with a medium that is not a 'Medium2D'. "
                        "This is probably not correct, since the resulting simulation will "
                        "depend on the details of the numerical grid. Consider either "
                        "giving the geometry a nonzero thickness or using a 'Medium2D'."
                    )

        return val

    @pydantic.validator("boundary_spec", always=True)
    def _structures_not_close_pml(cls, val, values):
        """Warn if any structures lie at the simulation boundaries."""

        sim_box = Box(size=values.get("size"), center=values.get("center"))
        sim_bound_min, sim_bound_max = sim_box.bounds

        boundaries = val.to_list
        structures = values.get("structures")
        sources = values.get("sources")

        if (not structures) or (not sources):

            return val

        with log as consolidated_logger:

            def warn(istruct, side):
                """Warning message for a structure too close to PML."""
                consolidated_logger.warning(
                    f"Structure at structures[{istruct}] was detected as being less "
                    f"than half of a central wavelength from a PML on side {side}. "
                    "To avoid inaccurate results or divergence, please increase gap between "
                    "any structures and PML or fully extend structure through the pml.",
                    custom_loc=["structures", istruct],
                )

            for istruct, structure in enumerate(structures):
                struct_bound_min, struct_bound_max = structure.geometry.bounds

                for source in sources:
                    lambda0 = C_0 / source.source_time.freq0

                    zipped = zip(["x", "y", "z"], sim_bound_min, struct_bound_min, boundaries)
                    for axis, sim_val, struct_val, boundary in zipped:
                        # The test is required only for PML and stable PML
                        if not isinstance(boundary[0], (PML, StablePML)):
                            continue
                        if (
                            boundary[0].num_layers > 0
                            and struct_val > sim_val
                            and abs(sim_val - struct_val) < lambda0 / 2
                        ):
                            warn(istruct, axis + "-min")

                    zipped = zip(["x", "y", "z"], sim_bound_max, struct_bound_max, boundaries)
                    for axis, sim_val, struct_val, boundary in zipped:
                        # The test is required only for PML and stable PML
                        if not isinstance(boundary[1], (PML, StablePML)):
                            continue
                        if (
                            boundary[1].num_layers > 0
                            and struct_val < sim_val
                            and abs(sim_val - struct_val) < lambda0 / 2
                        ):
                            warn(istruct, axis + "-max")

        return val

    @pydantic.validator("monitors", always=True)
    def _warn_monitor_mediums_frequency_range(cls, val, values):
        """Warn user if any DFT monitors have frequencies outside of medium frequency range."""

        if val is None:
            return val

        structures = values.get("structures")
        structures = structures or []
        medium_bg = values.get("medium")
        mediums = [medium_bg] + [structure.medium for structure in structures]

        with log as consolidated_logger:
            for monitor_index, monitor in enumerate(val):
                if not isinstance(monitor, FreqMonitor):
                    continue

                freqs = np.array(monitor.freqs)
                fmin_mon = freqs.min()
                fmax_mon = freqs.max()
                for medium_index, medium in enumerate(mediums):

                    # skip mediums that have no freq range (all freqs valid)
                    if medium.frequency_range is None:
                        continue

                    # make sure medium frequency range includes all monitor frequencies
                    fmin_med, fmax_med = medium.frequency_range
                    if fmin_mon < fmin_med or fmax_mon > fmax_med:
                        if medium_index == 0:
                            medium_str = "The simulation background medium"
                            custom_loc = ["medium", "frequency_range"]
                        else:
                            medium_str = f"The medium associated with structures[{medium_index-1}]"
                            custom_loc = [
                                "structures",
                                medium_index - 1,
                                "medium",
                                "frequency_range",
                            ]

                        consolidated_logger.warning(
                            f"{medium_str} has a frequency range: ({fmin_med:2e}, {fmax_med:2e}) "
                            "(Hz) that does not fully cover the frequencies contained in "
                            f"monitors[{monitor_index}]. "
                            "This can cause inaccuracies in the recorded results.",
                            custom_loc=custom_loc,
                        )

        return val

    @pydantic.validator("monitors", always=True)
    def _warn_monitor_simulation_frequency_range(cls, val, values):
        """Warn if any DFT monitors have frequencies outside of the simulation frequency range."""

        if val is None:
            return val

        # Get simulation frequency range
        if "sources" not in values:
            raise ValidationError(
                "could not validate `_warn_monitor_simulation_frequency_range` "
                "as `sources` failed validation"
            )

        source_ranges = [source.source_time.frequency_range() for source in values["sources"]]
        if not source_ranges:
            log.info("No sources in simulation.")
            return val

        freq_min = min((freq_range[0] for freq_range in source_ranges), default=0.0)
        freq_max = max((freq_range[1] for freq_range in source_ranges), default=0.0)

        with log as consolidated_logger:
            for monitor_index, monitor in enumerate(val):
                if not isinstance(monitor, FreqMonitor):
                    continue

                freqs = np.array(monitor.freqs)
                if freqs.min() < freq_min or freqs.max() > freq_max:
                    consolidated_logger.warning(
                        f"monitors[{monitor_index}] contains frequencies "
                        f"outside of the simulation frequency range ({freq_min:2e}, {freq_max:2e})"
                        "(Hz) as defined by the sources.",
                        custom_loc=["monitors", monitor_index, "freqs"],
                    )
        return val

    @pydantic.validator("monitors", always=True)
    def diffraction_monitor_boundaries(cls, val, values):
        """If any :class:`.DiffractionMonitor` exists, ensure boundary conditions in the
        transverse directions are periodic or Bloch."""
        monitors = val
        boundary_spec = values.get("boundary_spec")
        for monitor in monitors:
            if isinstance(monitor, DiffractionMonitor):
                _, (n_x, n_y) = monitor.pop_axis(["x", "y", "z"], axis=monitor.normal_axis)
                boundaries = [
                    boundary_spec[n_x].plus,
                    boundary_spec[n_x].minus,
                    boundary_spec[n_y].plus,
                    boundary_spec[n_y].minus,
                ]
                # make sure the transverse boundaries are either periodic or Bloch
                for boundary in boundaries:
                    if not isinstance(boundary, (Periodic, BlochBoundary)):
                        raise SetupError(
                            f"The 'DiffractionMonitor' {monitor.name} requires periodic "
                            f"or Bloch boundaries along dimensions {n_x} and {n_y}."
                        )
        return val

    @pydantic.validator("monitors", always=True)
    def _projection_monitors_homogeneous(cls, val, values):
        """Error if any field projection monitor is not in a homogeneous region."""

        if val is None:
            return val

        # list of structures including background as a Box()
        structure_bg = Structure(
            geometry=Box(
                size=values.get("size"),
                center=values.get("center"),
            ),
            medium=values.get("medium"),
        )

        structures = values.get("structures") or []
        total_structures = [structure_bg] + list(structures)

        for monitor in val:
            if isinstance(monitor, (AbstractFieldProjectionMonitor, DiffractionMonitor)):
                mediums = Scene.intersecting_media(monitor, total_structures)
                # make sure there is no more than one medium in the returned list
                if len(mediums) > 1:
                    raise SetupError(
                        f"{len(mediums)} different mediums detected on plane "
                        f"intersecting a {monitor.type}. Plane must be homogeneous."
                    )

        return val

    @pydantic.validator("monitors", always=True)
    def _integration_surfaces_in_bounds(cls, val, values):
        """Error if any of the integration surfaces are outside of the simulation domain."""

        if val is None:
            return val

        sim_center = values.get("center")
        sim_size = values.get("size")
        sim_box = Box(size=sim_size, center=sim_center)

        for mnt in (mnt for mnt in val if isinstance(mnt, SurfaceIntegrationMonitor)):
            if not any(sim_box.intersects(surf) for surf in mnt.integration_surfaces):
                raise SetupError(
                    f"All integration surfaces of monitor '{mnt.name}' are outside of the "
                    "simulation bounds."
                )

        return val

    @pydantic.validator("monitors", always=True)
    def _projection_monitors_distance(cls, val, values):
        """Warn if the projection distance is large for exact projections."""

        if val is None:
            return val

        sim_size = values.get("size")

        with log as consolidated_logger:
            for idx, monitor in enumerate(val):
                if isinstance(monitor, AbstractFieldProjectionMonitor):
                    if (
                        np.abs(monitor.proj_distance) > 1.0e4 * np.max(sim_size)
                        and not monitor.far_field_approx
                    ):
                        monitor = monitor.copy(update={"far_field_approx": True})
                        val = list(val)
                        val[idx] = monitor
                        val = tuple(val)
                        consolidated_logger.warning(
                            "A very large projection distance was set for the field projection "
                            f"monitor '{monitor.name}'. Using exact field projections may result "
                            "in precision loss for large distances; automatically enabling "
                            "far-field approximations ('far_field_approx = True') for better "
                            "precision. To insist on exact projections, consider using client-side "
                            "projections via the 'FieldProjector' class, where higher precision is "
                            "available.",
                            custom_loc=["monitors", idx, "proj_distance"],
                        )
        return val

    @pydantic.validator("monitors", always=True)
    def diffraction_monitor_medium(cls, val, values):
        """If any :class:`.DiffractionMonitor` exists, ensure is does not lie in a lossy medium."""
        monitors = val
        structures = values.get("structures")
        medium = values.get("medium")
        for monitor in monitors:
            if isinstance(monitor, DiffractionMonitor):
                medium_set = Scene.intersecting_media(monitor, structures)
                medium = medium_set.pop() if medium_set else medium
                _, index_k = medium.nk_model(frequency=np.array(monitor.freqs))
                if not np.all(index_k == 0):
                    raise SetupError("Diffraction monitors must not lie in a lossy medium.")
        return val

    @pydantic.validator("grid_spec", always=True)
    def _warn_grid_size_too_small(cls, val, values):
        """Warn user if any grid size is too large compared to minimum wavelength in material."""

        if val is None:
            return val

        structures = values.get("structures")
        structures = structures or []
        medium_bg = values.get("medium")
        mediums = [medium_bg] + [structure.medium for structure in structures]

        with log as consolidated_logger:
            for source_index, source in enumerate(values.get("sources")):
                freq0 = source.source_time.freq0

                for medium_index, medium in enumerate(mediums):

                    # min wavelength in PEC is meaningless and we'll get divide by inf errors
                    if medium.is_pec:
                        continue
                    # min wavelength in Medium2D is meaningless
                    if isinstance(medium, Medium2D):
                        continue

                    eps_material = medium.eps_model(freq0)
                    n_material, _ = medium.eps_complex_to_nk(eps_material)

                    for comp, (key, grid_spec) in enumerate(
                        zip("xyz", (val.grid_x, val.grid_y, val.grid_z))
                    ):
                        if medium.is_pec or (
                            isinstance(medium, AnisotropicMedium) and medium.is_comp_pec(comp)
                        ):
                            n_material = 1.0
                        lambda_min = C_0 / freq0 / n_material

                        if (
                            isinstance(grid_spec, UniformGrid)
                            and grid_spec.dl > lambda_min / MIN_GRIDS_PER_WVL
                        ):
                            if medium_index == 0:
                                medium_str = "the simulation background medium"
                            else:
                                medium_str = (
                                    f"the medium associated with structures[{medium_index-1}]"
                                )

                            consolidated_logger.warning(
                                f"The grid step in {key} has a value of {grid_spec.dl:.4f} (um)"
                                ", which was detected as being large when compared to the "
                                f"central wavelength of sources[{source_index}] "
                                f"within {medium_str}, given by "
                                f"{lambda_min:.4f} (um). To avoid inaccuracies, "
                                "it is recommended the grid size is reduced. ",
                                custom_loc=["grid_spec", f"grid_{key}", "dl"],
                            )
                            # TODO: warn about custom grid spec

        return val

    @pydantic.validator("sources", always=True)
    def _source_homogeneous_isotropic(cls, val, values):
        """Error if a plane wave or gaussian beam source is not in a homogeneous and isotropic
        region.
        """

        if val is None:
            return val

        # list of structures including background as a Box()
        structure_bg = Structure(
            geometry=Box(
                size=values.get("size"),
                center=values.get("center"),
            ),
            medium=values.get("medium"),
        )

        structures = values.get("structures") or []
        total_structures = [structure_bg] + list(structures)

        # for each plane wave in the sources list
        for source in val:
            if isinstance(source, (PlaneWave, GaussianBeam, AstigmaticGaussianBeam)):
                mediums = Scene.intersecting_media(source, total_structures)
                # make sure there is no more than one medium in the returned list
                if len(mediums) > 1:
                    raise SetupError(
                        f"{len(mediums)} different mediums detected on plane "
                        f"intersecting a {source.type} source. Plane must be homogeneous."
                    )
                if len(mediums) == 1 and isinstance(
                    list(mediums)[0], (AnisotropicMedium, FullyAnisotropicMedium)
                ):
                    raise SetupError(
                        f"An anisotropic medium is detected on plane intersecting a {source.type} "
                        f"source. Injection of {source.type} into anisotropic media currently is "
                        "not supported."
                    )

        return val

    @pydantic.validator("normalize_index", always=True)
    def _check_normalize_index(cls, val, values):
        """Check validity of normalize index in context of simulation.sources."""

        # not normalizing
        if val is None:
            return val

        sources = values.get("sources")
        num_sources = len(sources)
        if num_sources > 0:
            # No check if no sources, but it should be irrelevant anyway
            if val >= num_sources:
                raise ValidationError(
                    f"'normalize_index' {val} out of bounds for number of sources {num_sources}."
                )

            # Also error if normalizing by a zero-amplitude source
            if sources[val].source_time.amplitude == 0:
                raise ValidationError("Cannot set 'normalize_index' to source with zero amplitude.")

            # Warn if normalizing by a ContinuousWave or CustomSourceTime source, if frequency-domain monitors are present.
            if isinstance(sources[val].source_time, ContinuousWave):
                log.warning(
                    f"'normalize_index' {val} is a source with 'ContinuousWave' "
                    "time dependence. Normalizing frequency-domain monitors by this "
                    "source is not meaningful because field decay does not occur. "
                    "Consider setting 'normalize_index' to 'None' instead."
                )
            if isinstance(sources[val].source_time, CustomSourceTime):
                log.warning(
                    f"'normalize_index' {val} is a source with 'CustomSourceTime' "
                    "time dependence. Normalizing frequency-domain monitors by this "
                    "source is only meaningful if field decay occurs."
                )

        return val

    """ Post-init validators """

    def _post_init_validators(self) -> None:
        """Call validators taking z`self` that get run after init."""
        _ = self.scene
        self._validate_no_structures_pml()
        self._validate_tfsf_nonuniform_grid()
        self._validate_nonlinear_specs()

    def _validate_no_structures_pml(self) -> None:
        """Ensure no structures terminate / have bounds inside of PML."""

        pml_thicks = np.array(self.pml_thicknesses).T
        sim_bounds = self.bounds
        bound_spec = self.boundary_spec.to_list

        with log as consolidated_logger:
            for i, structure in enumerate(self.structures):
                geo_bounds = structure.geometry.bounds
                for sim_bound, geo_bound, pml_thick, bound_dim, pm_val in zip(
                    sim_bounds, geo_bounds, pml_thicks, bound_spec, (-1, 1)
                ):
                    for sim_pos, geo_pos, pml, bound_edge in zip(
                        sim_bound, geo_bound, pml_thick, bound_dim
                    ):
                        sim_pos_pml = sim_pos + pm_val * pml
                        in_pml_plus = (pm_val > 0) and (sim_pos < geo_pos <= sim_pos_pml)
                        in_pml_mnus = (pm_val < 0) and (sim_pos > geo_pos >= sim_pos_pml)
                        if not isinstance(bound_edge, Absorber) and (in_pml_plus or in_pml_mnus):
                            consolidated_logger.warning(
                                f"A bound of Simulation.structures[{i}] was detected as being "
                                "within the simulation PML. We recommend extending structures to "
                                "infinity or completely outside of the simulation PML to avoid "
                                "unexpected effects when the structures are not translationally "
                                "invariant within the PML.",
                                custom_loc=["structures", i],
                            )

    def _validate_tfsf_nonuniform_grid(self) -> None:
        """Warn if the grid is nonuniform along the directions tangential to the injection plane,
        inside the TFSF box.
        """
        # if the grid is uniform in all directions, there's no need to proceed
        if not (self.grid_spec.auto_grid_used or self.grid_spec.custom_grid_used):
            return

        with log as consolidated_logger:
            for source_ind, source in enumerate(self.sources):
                if not isinstance(source, TFSF):
                    continue

                centers = self.grid.centers.to_list
                sizes = self.grid.sizes.to_list
                tfsf_bounds = source.bounds
                _, plane_inds = source.pop_axis([0, 1, 2], axis=source.injection_axis)
                grid_list = [self.grid_spec.grid_x, self.grid_spec.grid_y, self.grid_spec.grid_z]
                for ind in plane_inds:
                    grid_type = grid_list[ind]
                    if isinstance(grid_type, UniformGrid):
                        continue

                    sizes_in_tfsf = [
                        size
                        for size, center in zip(sizes[ind], centers[ind])
                        if tfsf_bounds[0][ind] <= center <= tfsf_bounds[1][ind]
                    ]

                    # check if all the grid sizes are sufficiently unequal
                    if not np.all(np.isclose(sizes_in_tfsf, sizes_in_tfsf[0])):
                        consolidated_logger.warning(
                            f"The grid is nonuniform along the '{'xyz'[ind]}' axis, which may lead "
                            "to sub-optimal cancellation of the incident field in the "
                            "scattered-field region for the total-field scattered-field (TFSF) "
                            f"source '{source.name}'. For best results, we recommended ensuring a "
                            "uniform grid in both directions tangential to the TFSF injection "
                            f"axis, '{'xyz'[source.injection_axis]}'.",
                            custom_loc=["sources", source_ind],
                        )

    def _validate_nonlinear_specs(self) -> None:
        """Run :class:`.NonlinearSpec` validators that depend on knowing the central
        frequencies of the sources."""
        freqs = np.array([source.source_time.freq0 for source in self.sources])
        for medium in self.scene.mediums:
            if medium.nonlinear_spec is not None:
                for model in medium._nonlinear_models:
                    model._validate_medium_freqs(medium, freqs)

    """ Pre submit validation (before web.upload()) """

    def validate_pre_upload(self, source_required: bool = True) -> None:
        """Validate the fully initialized simulation is ok for upload to our servers.

        Parameters
        ----------
        source_required: bool = True
            If ``True``, validation will fail in case no sources are found in the simulation.
        """
        log.begin_capture()
        self._validate_size()
        self._validate_monitor_size()
        self._validate_modes_size()
        self._validate_datasets_not_none()
        self._validate_tfsf_structure_intersections()
        # self._validate_run_time()
        _ = self.volumetric_structures
        log.end_capture(self)
        if source_required and len(self.sources) == 0:
            raise SetupError("No sources in simulation.")

    def _validate_size(self) -> None:
        """Ensures the simulation is within size limits before simulation is uploaded."""

        num_comp_cells = self.num_cells / 2 ** (np.sum(np.abs(self.symmetry)))
        if num_comp_cells > MAX_GRID_CELLS:
            raise SetupError(
                f"Simulation has {num_comp_cells:.2e} computational cells, "
                f"a maximum of {MAX_GRID_CELLS:.2e} are allowed."
            )

        num_time_steps = self.num_time_steps
        if num_time_steps > MAX_TIME_STEPS:
            raise SetupError(
                f"Simulation has {num_time_steps:.2e} time steps, "
                f"a maximum of {MAX_TIME_STEPS:.2e} are allowed."
            )
        if num_time_steps > WARN_TIME_STEPS:
            log.warning(
                f"Simulation has {num_time_steps:.2e} time steps. The 'run_time' may be "
                "unnecessarily large, unless there are very long-lived resonances.",
                custom_loc=["run_time"],
            )

        num_cells_times_steps = num_time_steps * num_comp_cells
        if num_cells_times_steps > MAX_CELLS_TIMES_STEPS:
            raise SetupError(
                f"Simulation has {num_cells_times_steps:.2e} grid cells * time steps, "
                f"a maximum of {MAX_CELLS_TIMES_STEPS:.2e} are allowed."
            )

    def _validate_monitor_size(self) -> None:
        """Ensures the monitors aren't storing too much data before simulation is uploaded."""

        total_size_gb = 0
        with log as consolidated_logger:
            datas = self.monitors_data_size
            for monitor_ind, (monitor_name, monitor_size) in enumerate(datas.items()):
                monitor_size_gb = monitor_size / 1e9
                if monitor_size_gb > WARN_MONITOR_DATA_SIZE_GB:
                    consolidated_logger.warning(
                        f"Monitor '{monitor_name}' estimated storage is {monitor_size_gb:1.2f}GB. "
                        "Consider making it smaller, using fewer frequencies, or spatial or "
                        "temporal downsampling using 'interval_space' and 'interval', respectively.",
                        custom_loc=["monitors", monitor_ind],
                    )

                total_size_gb += monitor_size_gb

        if total_size_gb > MAX_SIMULATION_DATA_SIZE_GB:
            raise SetupError(
                f"Simulation's monitors have {total_size_gb:.2f}GB of estimated storage, "
                f"a maximum of {MAX_SIMULATION_DATA_SIZE_GB:.2f}GB are allowed."
            )

        # Some monitors store much less data than what is needed internally. Make sure that the
        # internal storage also does not exceed the limit.
        for monitor in self.monitors:
            num_cells = self._monitor_num_cells(monitor)
            # intermediate storage needed, in GB
            solver_data = monitor._storage_size_solver(num_cells=num_cells, tmesh=self.tmesh) / 1e9
            if solver_data > MAX_MONITOR_INTERNAL_DATA_SIZE_GB:
                raise SetupError(
                    f"Estimated internal storage of monitor '{monitor.name}' is "
                    f"{solver_data:1.2f}GB, which is larger than the maximum allowed "
                    f"{MAX_MONITOR_INTERNAL_DATA_SIZE_GB:.2f}GB. Consider making it smaller, "
                    "using fewer frequencies, or spatial or temporal downsampling using "
                    "'interval_space' and 'interval', respectively."
                )

    def _validate_modes_size(self) -> None:
        """Warn if mode sources or monitors have a large number of points."""

        def warn_mode_size(monitor: AbstractModeMonitor, msg_header: str, custom_loc: List):
            """Warn if a mode component has a large number of points."""
            num_cells = np.prod(self.discretize_monitor(monitor).num_cells)
            if num_cells > WARN_MODE_NUM_CELLS:
                consolidated_logger.warning(
                    msg_header + f"has a large number ({num_cells:1.2e}) of grid points. "
                    "This can lead to solver slow-down and increased cost. "
                    "Consider making the size of the component smaller, as long as the modes "
                    "of interest decay by the plane boundaries.",
                    custom_loc=custom_loc,
                )

        with log as consolidated_logger:
            for src_ind, source in enumerate(self.sources):
                if isinstance(source, ModeSource):
                    # Make a monitor so we can call ``discretize_monitor``
                    monitor = FieldMonitor(
                        center=source.center,
                        size=source.size,
                        name="tmp",
                        freqs=[source.source_time.freq0],
                        colocate=False,
                    )
                    msg_header = f"Mode source at sources[{src_ind}] "
                    custom_loc = ["sources", src_ind]
                    warn_mode_size(monitor=monitor, msg_header=msg_header, custom_loc=custom_loc)

        with log as consolidated_logger:
            for mnt_ind, monitor in enumerate(self.monitors):
                if isinstance(monitor, AbstractModeMonitor):
                    msg_header = f"Mode monitor '{monitor.name}' "
                    custom_loc = ["monitors", mnt_ind]
                    warn_mode_size(monitor=monitor, msg_header=msg_header, custom_loc=custom_loc)

    @cached_property
    def monitors_data_size(self) -> Dict[str, float]:
        """Dictionary mapping monitor names to their estimated storage size in bytes."""
        data_size = {}
        for monitor in self.monitors:
            num_cells = self._monitor_num_cells(monitor)
            storage_size = float(monitor.storage_size(num_cells=num_cells, tmesh=self.tmesh))
            data_size[monitor.name] = storage_size
        return data_size

    def _monitor_num_cells(self, monitor: Monitor) -> int:
        """Total number of cells included by monitor based on simulation grid."""
        num_cells = self.discretize_monitor(monitor).num_cells
        # take monitor downsampling into account
        num_cells = monitor.downsampled_num_cells(num_cells)
        return np.prod(np.array(num_cells, dtype=np.int64))

    def _validate_datasets_not_none(self) -> None:
        """Ensures that all custom datasets are defined."""
        if any(dataset is None for dataset in self.custom_datasets):
            raise SetupError(
                "Data for a custom data component is missing. This can happen for example if the "
                "Simulation has been loaded from json. To save and load simulations with custom "
                "data, use hdf5 format instead."
            )

    def _validate_tfsf_structure_intersections(self) -> None:
        """Error if the 4 sidewalls of a TFSF box don't all intersect the same structures.
        This validator may need to compute permittivities on the grid, so it is called
        pre-upload rather than at the time of definition. Also errors if any side wall
        intersects with a custom medium or a fully anisotropic media.
        """
        for source in self.sources:
            if not isinstance(source, TFSF):
                continue
            # get all TFSF surfaces
            tfsf_surfaces = Source.surfaces(
                center=source.center, size=source.size, source_time=source.source_time
            )
            sidewall_surfaces = []
            sidewall_structs = []
            # get the structures that intersect each sidewall
            for surface in tfsf_surfaces:
                # ignore the sidewall surface if it falls outside the simulation domain
                if not self.intersects(surface):
                    continue

                if surface.name[-2] != "xyz"[source.injection_axis]:
                    sidewall_surfaces.append(surface)
                    intersecting_structs = Scene.intersecting_structures(
                        test_object=surface, structures=self.structures
                    )

                    if any(
                        isinstance(struct.medium, (AbstractCustomMedium, FullyAnisotropicMedium))
                        for struct in intersecting_structs
                    ):
                        raise SetupError(
                            f"The surfaces of TFSF source '{source.name}' must not intersect any "
                            "structures containing a 'CustomMedium' or a 'FullyAnisotropicMedium'."
                        )

                    # if no structures intersect, just add a phantom associated with the simulation
                    # background, to prevent false positives below
                    if not intersecting_structs:
                        sidewall_structs.append(
                            [
                                Structure(
                                    geometry=Box(center=self.center, size=self.size),
                                    medium=self.medium,
                                )
                            ]
                        )
                    else:
                        sidewall_structs.append(intersecting_structs)

            # let the first wall be a reference, and compare the rest of them to the structures
            # intersected by that reference wall
            if len(sidewall_structs) > 1:
                ref_structs = sidewall_structs[0]
                test_structs = sidewall_structs[1:]
                if all(structs == ref_structs for structs in test_structs):
                    continue

                # if the == test doesn't pass, that doesn't mean the materials are necessarily
                # different, because it's possible that the sidewalls encounter different
                # `Structure` objects but with an identical material profile, which is still
                # a valid setup; in this case, compute the epsilon profile on the grid for each
                # side wall - the profiles must be the same along the injection axis, so we take
                # a single "stripe" of epsilon as the reference and subtract it from all other
                # stripes, which should result in zero if all the epsilon profiles are the same
                freq0 = source.source_time.freq0
                _, plane_axs = source.pop_axis("xyz", axis=source.injection_axis)
                ref_eps = self.epsilon(box=sidewall_surfaces[0], coord_key="centers", freq=freq0)
                kwargs = {plane_axs[0]: 0, plane_axs[1]: 0}
                ref_eps = ref_eps.isel(**kwargs)
                for surface in sidewall_surfaces:
                    test_eps = self.epsilon(box=surface, coord_key="centers", freq=freq0) - ref_eps
                    if not np.allclose(test_eps.to_numpy(), 0):
                        raise SetupError(
                            f"All sidewalls of the TFSF source '{source.name}' must intersect "
                            "the same media along the injection axis "
                            f" '{'xyz'[source.injection_axis]}'."
                        )

    """ Accounting """

    # candidate for removal in 3.0
    @cached_property
    def mediums(self) -> Set[MediumType]:
        """Returns set of distinct :class:`.AbstractMedium` in simulation.

        Returns
        -------
        List[:class:`.AbstractMedium`]
            Set of distinct mediums in the simulation.
        """
        log.warning(
            "'Simulation.mediums' will be removed in Tidy3D 3.0. "
            "Use 'Simulation.scene.mediums' instead."
        )
        return self.scene.mediums

    # candidate for removal in 3.0
    @cached_property
    def medium_map(self) -> Dict[MediumType, pydantic.NonNegativeInt]:
        """Returns dict mapping medium to index in material.
        ``medium_map[medium]`` returns unique global index of :class:`.AbstractMedium`
        in simulation.

        Returns
        -------
        Dict[:class:`.AbstractMedium`, int]
            Mapping between distinct mediums to index in simulation.
        """

        log.warning(
            "'Simulation.medium_map' will be removed in Tidy3D 3.0. "
            "Use 'Simulation.scene.medium_map' instead."
        )
        return self.scene.medium_map

    # candidate for removal in 3.0
    @cached_property
    def background_structure(self) -> Structure:
        """Returns structure representing the background of the :class:`.Simulation`."""

        log.warning(
            "'Simulation.background_structure' will be removed in Tidy3D 3.0. "
            "Use 'Simulation.scene.background_structure' instead."
        )
        return self.scene.background_structure

    # candidate for removal in 3.0
    @staticmethod
    def intersecting_media(
        test_object: Box, structures: Tuple[Structure, ...]
    ) -> Tuple[MediumType, ...]:
        """From a given list of structures, returns a list of :class:`.AbstractMedium` associated
        with those structures that intersect with the ``test_object``, if it is a surface, or its
        surfaces, if it is a volume.

        Parameters
        -------
        test_object : :class:`.Box`
            Object for which intersecting media are to be detected.
        structures : List[:class:`.AbstractMedium`]
            List of structures whose media will be tested.

        Returns
        -------
        List[:class:`.AbstractMedium`]
            Set of distinct mediums that intersect with the given planar object.
        """

        log.warning(
            "'Simulation.intersecting_media()' will be removed in Tidy3D 3.0. "
            "Use 'Scene.intersecting_media()' instead."
        )
        return Scene.intersecting_media(test_object=test_object, structures=structures)

    # candidate for removal in 3.0
    @staticmethod
    def intersecting_structures(
        test_object: Box, structures: Tuple[Structure, ...]
    ) -> Tuple[Structure, ...]:
        """From a given list of structures, returns a list of :class:`.Structure` that intersect
        with the ``test_object``, if it is a surface, or its surfaces, if it is a volume.

        Parameters
        -------
        test_object : :class:`.Box`
            Object for which intersecting media are to be detected.
        structures : List[:class:`.AbstractMedium`]
            List of structures whose media will be tested.

        Returns
        -------
        List[:class:`.Structure`]
            Set of distinct structures that intersect with the given surface, or with the surfaces
            of the given volume.
        """

        log.warning(
            "'Simulation.intersecting_structures()' will be removed in Tidy3D 3.0. "
            "Use 'Scene.intersecting_structures()' instead."
        )
        return Scene.intersecting_structures(test_object=test_object, structures=structures)

    def monitor_medium(self, monitor: MonitorType):
        """Return the medium in which the given monitor resides.

        Parameters
        -------
        monitor : :class:`.Monitor`
            Monitor whose associated medium is to be returned.

        Returns
        -------
        :class:`.AbstractMedium`
            Medium associated with the given :class:`.Monitor`.
        """
        medium_set = Scene.intersecting_media(monitor, self.structures)
        if len(medium_set) > 1:
            raise SetupError(f"Monitor '{monitor.name}' intersects more than one medium.")
        medium = medium_set.pop() if medium_set else self.medium
        return medium

    @staticmethod
    def _check_bloch_vec(
        source: SourceType,
        source_ind: int,
        bloch_vec: float,
        dim: Axis,
        medium: MediumType,
        domain_size: float,
    ):
        """Helper to check if a given Bloch vector is consistent with a given source."""

        # make a dummy Bloch boundary to check for correctness
        dummy_bnd = BlochBoundary.from_source(
            source=source, domain_size=domain_size, axis=dim, medium=medium
        )
        expected_bloch_vec = dummy_bnd.bloch_vec

        if bloch_vec != expected_bloch_vec:
            test_val = np.real(expected_bloch_vec - bloch_vec)

            if np.isclose(test_val % 1, 0) and not np.isclose(test_val, 0):
                # the given Bloch vector is offset by an integer
                log.warning(
                    f"The wave vector of source '{source.name}' along dimension "
                    f"'{dim}' is equal to the Bloch vector of the simulation "
                    "boundaries along that dimension plus an integer reciprocal "
                    "lattice vector. If using a 'DiffractionMonitor', diffraction "
                    "order 0 will not correspond to the angle of propagation "
                    "of the source. Consider using 'BlochBoundary.from_source()'.",
                    custom_loc=["boundary_spec", "xyz"[dim]],
                )
            elif not np.isclose(test_val % 1, 0):
                # the given Bloch vector is neither equal to the expected value, nor
                # off by an integer
                log.warning(
                    f"The Bloch vector along dimension '{dim}' may be incorrectly "
                    f"set with respect to the source '{source.name}'. The absolute "
                    "difference between the expected and provided values in "
                    "bandstructure units, up to an integer offset, is greater than "
                    "1e-6. Consider using ``BlochBoundary.from_source()``, or "
                    "double-check that it was defined correctly.",
                    custom_loc=["boundary_spec", "xyz"[dim]],
                )

    def to_gdstk(
        self,
        x: float = None,
        y: float = None,
        z: float = None,
        permittivity_threshold: pydantic.NonNegativeFloat = 1,
        frequency: pydantic.PositiveFloat = 0,
        gds_layer_dtype_map: Dict[
            AbstractMedium, Tuple[pydantic.NonNegativeInt, pydantic.NonNegativeInt]
        ] = None,
    ) -> List:
        """Convert a simulation's planar slice to a .gds type polygon list.

        Parameters
        ----------
        x : float = None
            Position of plane in x direction, only one of x,y,z can be specified to define plane.
        y : float = None
            Position of plane in y direction, only one of x,y,z can be specified to define plane.
        z : float = None
            Position of plane in z direction, only one of x,y,z can be specified to define plane.
        permittivity_threshold : float = 1.001
            Permitivitty value used to define the shape boundaries for structures with custom
            medim
        frequency : float = 0
            Frequency for permittivity evaluaiton in case of custom medium (Hz).
        gds_layer_dtype_map : Dict
            Dictionary mapping mediums to GDSII layer and data type tuples.

        Return
        ------
        List
            List of `gdstk.Polygon`.
        """
        axis, _ = self.geometry.parse_xyz_kwargs(x=x, y=y, z=z)
        _, bmin = self.pop_axis(self.bounds[0], axis)
        _, bmax = self.pop_axis(self.bounds[1], axis)

        _, symmetry = self.pop_axis(self.symmetry, axis)
        if symmetry[0] != 0:
            bmin = (0, bmin[1])
        if symmetry[1] != 0:
            bmin = (bmin[0], 0)
        clip = gdstk.rectangle(bmin, bmax)

        polygons = []
        for structure in self.structures:
            gds_layer, gds_dtype = gds_layer_dtype_map.get(structure.medium, (0, 0))
            for polygon in structure.to_gdstk(
                x=x,
                y=y,
                z=z,
                permittivity_threshold=permittivity_threshold,
                frequency=frequency,
                gds_layer=gds_layer,
                gds_dtype=gds_dtype,
            ):
                pmin, pmax = polygon.bounding_box()
                if pmin[0] < bmin[0] or pmin[1] < bmin[1] or pmax[0] > bmax[0] or pmax[1] > bmax[1]:
                    polygons.extend(
                        gdstk.boolean(clip, polygon, "and", layer=gds_layer, datatype=gds_dtype)
                    )
                else:
                    polygons.append(polygon)

        return polygons

    def to_gdspy(
        self,
        x: float = None,
        y: float = None,
        z: float = None,
        gds_layer_dtype_map: Dict[
            AbstractMedium, Tuple[pydantic.NonNegativeInt, pydantic.NonNegativeInt]
        ] = None,
    ) -> List:
        """Convert a simulation's planar slice to a .gds type polygon list.

        Parameters
        ----------
        x : float = None
            Position of plane in x direction, only one of x,y,z can be specified to define plane.
        y : float = None
            Position of plane in y direction, only one of x,y,z can be specified to define plane.
        z : float = None
            Position of plane in z direction, only one of x,y,z can be specified to define plane.
        gds_layer_dtype_map : Dict
            Dictionary mapping mediums to GDSII layer and data type tuples.

        Return
        ------
        List
            List of `gdspy.Polygon` and `gdspy.PolygonSet`.
        """
        axis, _ = self.geometry.parse_xyz_kwargs(x=x, y=y, z=z)
        _, bmin = self.pop_axis(self.bounds[0], axis)
        _, bmax = self.pop_axis(self.bounds[1], axis)

        _, symmetry = self.pop_axis(self.symmetry, axis)
        if symmetry[0] != 0:
            bmin = (0, bmin[1])
        if symmetry[1] != 0:
            bmin = (bmin[0], 0)
        clip = gdspy.Rectangle(bmin, bmax)

        polygons = []
        for structure in self.structures:
            gds_layer, gds_dtype = gds_layer_dtype_map.get(structure.medium, (0, 0))
            for polygon in structure.to_gdspy(
                x=x,
                y=y,
                z=z,
                gds_layer=gds_layer,
                gds_dtype=gds_dtype,
            ):
                pmin, pmax = polygon.get_bounding_box()
                if pmin[0] < bmin[0] or pmin[1] < bmin[1] or pmax[0] > bmax[0] or pmax[1] > bmax[1]:
                    polygon = gdspy.boolean(
                        clip, polygon, "and", layer=gds_layer, datatype=gds_dtype
                    )
                polygons.append(polygon)
        return polygons

    def to_gds(
        self,
        cell,
        x: float = None,
        y: float = None,
        z: float = None,
        permittivity_threshold: pydantic.NonNegativeFloat = 1,
        frequency: pydantic.PositiveFloat = 0,
        gds_layer_dtype_map: Dict[
            AbstractMedium, Tuple[pydantic.NonNegativeInt, pydantic.NonNegativeInt]
        ] = None,
    ) -> None:
        """Append the simulation structures to a .gds cell.

        Parameters
        ----------
        cell : ``gdstk.Cell`` or ``gdspy.Cell``
            Cell object to which the generated polygons are added.
        x : float = None
            Position of plane in x direction, only one of x,y,z can be specified to define plane.
        y : float = None
            Position of plane in y direction, only one of x,y,z can be specified to define plane.
        z : float = None
            Position of plane in z direction, only one of x,y,z can be specified to define plane.
        permittivity_threshold : float = 1.001
            Permitivitty value used to define the shape boundaries for structures with custom
            medim
        frequency : float = 0
            Frequency for permittivity evaluaiton in case of custom medium (Hz).
        gds_layer_dtype_map : Dict
            Dictionary mapping mediums to GDSII layer and data type tuples.
        """
        if gds_layer_dtype_map is None:
            gds_layer_dtype_map = {}

        if gdstk_available and isinstance(cell, gdstk.Cell):
            polygons = self.to_gdstk(
                x=x,
                y=y,
                z=z,
                permittivity_threshold=permittivity_threshold,
                frequency=frequency,
                gds_layer_dtype_map=gds_layer_dtype_map,
            )
            if len(polygons) > 0:
                cell.add(*polygons)

        elif gdspy_available and isinstance(cell, gdspy.Cell):
            polygons = self.to_gdspy(x=x, y=y, z=z, gds_layer_dtype_map=gds_layer_dtype_map)
            if len(polygons) > 0:
                cell.add(polygons)

        elif "gdstk" in cell.__class__ and not gdstk_available:
            raise Tidy3dImportError(
                "Module 'gdstk' not found. It is required to export shapes to gdstk cells."
            )
        elif "gdspy" in cell.__class__ and not gdspy_available:
            raise Tidy3dImportError(
                "Module 'gdspy' not found. It is required to export shapes to gdspy cells."
            )
        else:
            raise Tidy3dError(
                "Argument 'cell' must be an instance of 'gdstk.Cell' or 'gdspy.Cell'."
            )

    def to_gds_file(
        self,
        fname: str,
        x: float = None,
        y: float = None,
        z: float = None,
        permittivity_threshold: pydantic.NonNegativeFloat = 1,
        frequency: pydantic.PositiveFloat = 0,
        gds_layer_dtype_map: Dict[
            AbstractMedium, Tuple[pydantic.NonNegativeInt, pydantic.NonNegativeInt]
        ] = None,
        gds_cell_name: str = "MAIN",
    ) -> None:
        """Append the simulation structures to a .gds cell.

        Parameters
        ----------
        fname : str
            Full path to the .gds file to save the :class:`Simulation` slice to.
        x : float = None
            Position of plane in x direction, only one of x,y,z can be specified to define plane.
        y : float = None
            Position of plane in y direction, only one of x,y,z can be specified to define plane.
        z : float = None
            Position of plane in z direction, only one of x,y,z can be specified to define plane.
        permittivity_threshold : float = 1.001
            Permitivitty value used to define the shape boundaries for structures with custom
            medim
        frequency : float = 0
            Frequency for permittivity evaluaiton in case of custom medium (Hz).
        gds_layer_dtype_map : Dict
            Dictionary mapping mediums to GDSII layer and data type tuples.
        gds_cell_name : str = 'MAIN'
            Name of the cell created in the .gds file to store the geometry.
        """
        if gdstk_available:
            library = gdstk.Library()
            reference = gdstk.Reference
            rotation = np.pi
        elif gdspy_available:
            library = gdspy.GdsLibrary()
            reference = gdspy.CellReference
            rotation = 180
        else:
            raise Tidy3dImportError(
                "Python modules 'gdspy' and 'gdstk' not found. To export geometries to .gds "
                "files, please install one of those those modules."
            )
        cell = library.new_cell(gds_cell_name)

        axis, _ = self.geometry.parse_xyz_kwargs(x=x, y=y, z=z)
        _, symmetry = self.pop_axis(self.symmetry, axis)
        if symmetry[0] != 0:
            outer_cell = cell
            cell = library.new_cell(gds_cell_name + "_X")
            outer_cell.add(reference(cell))
            outer_cell.add(reference(cell, rotation=rotation, x_reflection=True))
        if symmetry[1] != 0:
            outer_cell = cell
            cell = library.new_cell(gds_cell_name + "_Y")
            outer_cell.add(reference(cell))
            outer_cell.add(reference(cell, x_reflection=True))

        self.to_gds(
            cell,
            x=x,
            y=y,
            z=z,
            permittivity_threshold=permittivity_threshold,
            frequency=frequency,
            gds_layer_dtype_map=gds_layer_dtype_map,
        )
        library.write_gds(fname)

    """ Plotting """

    @equal_aspect
    @add_ax_if_none
    def plot(
        self,
        x: float = None,
        y: float = None,
        z: float = None,
        ax: Ax = None,
        source_alpha: float = None,
        monitor_alpha: float = None,
        hlim: Tuple[float, float] = None,
        vlim: Tuple[float, float] = None,
        **patch_kwargs,
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
        source_alpha : float = None
            Opacity of the sources. If ``None``, uses Tidy3d default.
        monitor_alpha : float = None
            Opacity of the monitors. If ``None``, uses Tidy3d default.
        ax : matplotlib.axes._subplots.Axes = None
            Matplotlib axes to plot on, if not specified, one is created.
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

        ax = self.plot_structures(ax=ax, x=x, y=y, z=z, hlim=hlim, vlim=vlim)
        ax = self.plot_sources(ax=ax, x=x, y=y, z=z, hlim=hlim, vlim=vlim, alpha=source_alpha)
        ax = self.plot_monitors(ax=ax, x=x, y=y, z=z, hlim=hlim, vlim=vlim, alpha=monitor_alpha)
        ax = self.plot_symmetries(ax=ax, x=x, y=y, z=z, hlim=hlim, vlim=vlim)
        ax = self.plot_pml(ax=ax, x=x, y=y, z=z, hlim=hlim, vlim=vlim)
        ax = Scene._set_plot_bounds(
            bounds=self.simulation_bounds, ax=ax, x=x, y=y, z=z, hlim=hlim, vlim=vlim
        )
        ax = self.plot_boundaries(ax=ax, x=x, y=y, z=z)
        return ax

    @equal_aspect
    @add_ax_if_none
    def plot_eps(
        self,
        x: float = None,
        y: float = None,
        z: float = None,
        freq: float = None,
        alpha: float = None,
        source_alpha: float = None,
        monitor_alpha: float = None,
        hlim: Tuple[float, float] = None,
        vlim: Tuple[float, float] = None,
        ax: Ax = None,
    ) -> Ax:
        """Plot each of simulation's components on a plane defined by one nonzero x,y,z coordinate.
        The permittivity is plotted in grayscale based on its value at the specified frequency.

        Parameters
        ----------
        x : float = None
            position of plane in x direction, only one of x, y, z must be specified to define plane.
        y : float = None
            position of plane in y direction, only one of x, y, z must be specified to define plane.
        z : float = None
            position of plane in z direction, only one of x, y, z must be specified to define plane.
        freq : float = None
            Frequency to evaluate the relative permittivity of all mediums.
            If not specified, evaluates at infinite frequency.
        alpha : float = None
            Opacity of the structures being plotted.
            Defaults to the structure default alpha.
        source_alpha : float = None
            Opacity of the sources. If ``None``, uses Tidy3d default.
        monitor_alpha : float = None
            Opacity of the monitors. If ``None``, uses Tidy3d default.
        ax : matplotlib.axes._subplots.Axes = None
            Matplotlib axes to plot on, if not specified, one is created.
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

        ax = self.plot_structures_eps(
            freq=freq,
            cbar=True,
            alpha=alpha,
            ax=ax,
            x=x,
            y=y,
            z=z,
            hlim=hlim,
            vlim=vlim,
        )
        ax = self.plot_sources(ax=ax, x=x, y=y, z=z, hlim=hlim, vlim=vlim, alpha=source_alpha)
        ax = self.plot_monitors(ax=ax, x=x, y=y, z=z, hlim=hlim, vlim=vlim, alpha=monitor_alpha)
        ax = self.plot_symmetries(ax=ax, x=x, y=y, z=z, hlim=hlim, vlim=vlim)
        ax = self.plot_pml(ax=ax, x=x, y=y, z=z, hlim=hlim, vlim=vlim)
        ax = Scene._set_plot_bounds(
            bounds=self.simulation_bounds, ax=ax, x=x, y=y, z=z, hlim=hlim, vlim=vlim
        )
        ax = self.plot_boundaries(ax=ax, x=x, y=y, z=z)
        return ax

    @equal_aspect
    @add_ax_if_none
    def plot_structures_eps(
        self,
        x: float = None,
        y: float = None,
        z: float = None,
        freq: float = None,
        alpha: float = None,
        cbar: bool = True,
        reverse: bool = False,
        ax: Ax = None,
        hlim: Tuple[float, float] = None,
        vlim: Tuple[float, float] = None,
    ) -> Ax:
        """Plot each of simulation's structures on a plane defined by one nonzero x,y,z coordinate.
        The permittivity is plotted in grayscale based on its value at the specified frequency.

        Parameters
        ----------
        x : float = None
            position of plane in x direction, only one of x, y, z must be specified to define plane.
        y : float = None
            position of plane in y direction, only one of x, y, z must be specified to define plane.
        z : float = None
            position of plane in z direction, only one of x, y, z must be specified to define plane.
        freq : float = None
            Frequency to evaluate the relative permittivity of all mediums.
            If not specified, evaluates at infinite frequency.
        reverse : bool = False
            If ``False``, the highest permittivity is plotted in black.
            If ``True``, it is plotteed in white (suitable for black backgrounds).
        cbar : bool = True
            Whether to plot a colorbar for the relative permittivity.
        alpha : float = None
            Opacity of the structures being plotted.
            Defaults to the structure default alpha.
        ax : matplotlib.axes._subplots.Axes = None
            Matplotlib axes to plot on, if not specified, one is created.
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

        return self.scene.plot_structures_eps(
            freq=freq,
            cbar=cbar,
            alpha=alpha,
            ax=ax,
            x=x,
            y=y,
            z=z,
            hlim=hlim,
            vlim=vlim,
            grid=self.grid,
            reverse=reverse,
        )

    # candidate for removal in 3.0
    def eps_bounds(self, freq: float = None) -> Tuple[float, float]:
        """Compute range of (real) permittivity present in the simulation at frequency "freq"."""

        log.warning(
            "'Simulation.eps_bounds()' will be removed in Tidy3D 3.0. "
            "Use 'Simulation.scene.eps_bounds()' instead."
        )
        return self.scene.eps_bounds(freq=freq)

    @cached_property
    def num_pml_layers(self) -> List[Tuple[float, float]]:
        """Number of absorbing layers in all three axes and directions (-, +).

        Returns
        -------
        List[Tuple[float, float]]
            List containing the number of absorber layers in - and + boundaries.
        """
        num_layers = [[0, 0], [0, 0], [0, 0]]

        for idx_i, boundary1d in enumerate(self.boundary_spec.to_list):
            for idx_j, boundary in enumerate(boundary1d):
                if isinstance(boundary, (PML, StablePML, Absorber)):
                    num_layers[idx_i][idx_j] = boundary.num_layers

        return num_layers

    @cached_property
    def pml_thicknesses(self) -> List[Tuple[float, float]]:
        """Thicknesses (um) of absorbers in all three axes and directions (-, +)

        Returns
        -------
        List[Tuple[float, float]]
            List containing the absorber thickness (micron) in - and + boundaries.
        """
        num_layers = self.num_pml_layers
        pml_thicknesses = []
        for num_layer, boundaries in zip(num_layers, self.grid.boundaries.to_list):
            thick_l = boundaries[num_layer[0]] - boundaries[0]
            thick_r = boundaries[-1] - boundaries[-1 - num_layer[1]]
            pml_thicknesses.append((thick_l, thick_r))
        return pml_thicknesses

    # candidate for removal in 3.0
    @cached_property
    def bounds_pml(self) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """Simulation bounds including the PML regions."""
        log.warning(
            "'Simulation.bounds_pml' will be removed in Tidy3D 3.0. "
            "Use 'Simulation.simulation_bounds' instead."
        )
        return self.simulation_bounds

    @cached_property
    def simulation_bounds(self) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """Simulation bounds including the PML regions."""
        pml_thick = self.pml_thicknesses
        bounds_in = self.bounds
        bounds_min = tuple((bmin - pml[0] for bmin, pml in zip(bounds_in[0], pml_thick)))
        bounds_max = tuple((bmax + pml[1] for bmax, pml in zip(bounds_in[1], pml_thick)))

        return (bounds_min, bounds_max)

    @equal_aspect
    @add_ax_if_none
    def plot_pml(
        self,
        x: float = None,
        y: float = None,
        z: float = None,
        hlim: Tuple[float, float] = None,
        vlim: Tuple[float, float] = None,
        ax: Ax = None,
    ) -> Ax:
        """Plot each of simulation's absorbing boundaries
        on a plane defined by one nonzero x,y,z coordinate.

        Parameters
        ----------
        x : float = None
            position of plane in x direction, only one of x, y, z must be specified to define plane.
        y : float = None
            position of plane in y direction, only one of x, y, z must be specified to define plane.
        z : float = None
            position of plane in z direction, only one of x, y, z must be specified to define plane
        hlim : Tuple[float, float] = None
            The x range if plotting on xy or xz planes, y range if plotting on yz plane.
        vlim : Tuple[float, float] = None
            The z range if plotting on xz or yz planes, y plane if plotting on xy plane.
        ax : matplotlib.axes._subplots.Axes = None
            Matplotlib axes to plot on, if not specified, one is created.

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.
        """
        normal_axis, _ = self.parse_xyz_kwargs(x=x, y=y, z=z)
        pml_boxes = self._make_pml_boxes(normal_axis=normal_axis)
        for pml_box in pml_boxes:
            pml_box.plot(x=x, y=y, z=z, ax=ax, **plot_params_pml.to_kwargs())
        ax = Scene._set_plot_bounds(
            bounds=self.simulation_bounds, ax=ax, x=x, y=y, z=z, hlim=hlim, vlim=vlim
        )
        return ax

    def _make_pml_boxes(self, normal_axis: Axis) -> List[Box]:
        """make a list of Box objects representing the pml to plot on plane."""
        pml_boxes = []
        pml_thicks = self.pml_thicknesses
        for pml_axis, num_layers_dim in enumerate(self.num_pml_layers):
            if pml_axis == normal_axis:
                continue
            for sign, pml_height, num_layers in zip((-1, 1), pml_thicks[pml_axis], num_layers_dim):
                if num_layers == 0:
                    continue
                pml_box = self._make_pml_box(pml_axis=pml_axis, pml_height=pml_height, sign=sign)
                pml_boxes.append(pml_box)
        return pml_boxes

    def _make_pml_box(self, pml_axis: Axis, pml_height: float, sign: int) -> Box:
        """Construct a :class:`.Box` representing an arborbing boundary to be plotted."""
        rmin, rmax = (list(bounds) for bounds in self.simulation_bounds)
        if sign == -1:
            rmax[pml_axis] = rmin[pml_axis] + pml_height
        else:
            rmin[pml_axis] = rmax[pml_axis] - pml_height
        pml_box = Box.from_bounds(rmin=rmin, rmax=rmax)

        # if any dimension of the sim has size 0, set the PML to a very small size along that dim
        new_size = list(pml_box.size)
        for dim_index, sim_size in enumerate(self.size):
            if sim_size == 0.0:
                new_size[dim_index] = PML_HEIGHT_FOR_0_DIMS
        pml_box = pml_box.updated_copy(size=new_size)

        return pml_box

    @add_ax_if_none
    def plot_grid(
        self,
        x: float = None,
        y: float = None,
        z: float = None,
        ax: Ax = None,
        hlim: Tuple[float, float] = None,
        vlim: Tuple[float, float] = None,
        **kwargs,
    ) -> Ax:
        """Plot the cell boundaries as lines on a plane defined by one nonzero x,y,z coordinate.

        Parameters
        ----------
        x : float = None
            position of plane in x direction, only one of x, y, z must be specified to define plane.
        y : float = None
            position of plane in y direction, only one of x, y, z must be specified to define plane.
        z : float = None
            position of plane in z direction, only one of x, y, z must be specified to define plane.
        hlim : Tuple[float, float] = None
            The x range if plotting on xy or xz planes, y range if plotting on yz plane.
        vlim : Tuple[float, float] = None
            The z range if plotting on xz or yz planes, y plane if plotting on xy plane.
        ax : matplotlib.axes._subplots.Axes = None
            Matplotlib axes to plot on, if not specified, one is created.
        **kwargs
            Optional keyword arguments passed to the matplotlib ``LineCollection``.
            For details on accepted values, refer to
            `Matplotlib's documentation <https://tinyurl.com/2p97z4cn>`_.

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.
        """
        kwargs.setdefault("linewidth", 0.2)
        kwargs.setdefault("colors", "black")
        cell_boundaries = self.grid.boundaries
        axis, _ = self.parse_xyz_kwargs(x=x, y=y, z=z)
        _, (axis_x, axis_y) = self.pop_axis([0, 1, 2], axis=axis)
        boundaries_x = cell_boundaries.dict()["xyz"[axis_x]]
        boundaries_y = cell_boundaries.dict()["xyz"[axis_y]]
        _, (xmin, ymin) = self.pop_axis(self.simulation_bounds[0], axis=axis)
        _, (xmax, ymax) = self.pop_axis(self.simulation_bounds[1], axis=axis)
        segs_x = [((bound, ymin), (bound, ymax)) for bound in boundaries_x]
        line_segments_x = mpl.collections.LineCollection(segs_x, **kwargs)
        segs_y = [((xmin, bound), (xmax, bound)) for bound in boundaries_y]
        line_segments_y = mpl.collections.LineCollection(segs_y, **kwargs)

        # Plot grid
        ax.add_collection(line_segments_x)
        ax.add_collection(line_segments_y)

        # Plot bounding boxes of override structures
        plot_params = plot_params_override_structures.include_kwargs(
            linewidth=2 * kwargs["linewidth"], edgecolor=kwargs["colors"]
        )
        for structure in self.grid_spec.override_structures:
            bounds = list(zip(*structure.geometry.bounds))
            _, ((xmin, xmax), (ymin, ymax)) = structure.geometry.pop_axis(bounds, axis=axis)
            xmin, xmax, ymin, ymax = (self._evaluate_inf(v) for v in (xmin, xmax, ymin, ymax))
            rect = mpl.patches.Rectangle(
                xy=(xmin, ymin),
                width=(xmax - xmin),
                height=(ymax - ymin),
                **plot_params.to_kwargs(),
            )
            ax.add_patch(rect)

        ax = Scene._set_plot_bounds(
            bounds=self.simulation_bounds, ax=ax, x=x, y=y, z=z, hlim=hlim, vlim=vlim
        )

        return ax

    @equal_aspect
    @add_ax_if_none
    def plot_boundaries(
        self,
        x: float = None,
        y: float = None,
        z: float = None,
        ax: Ax = None,
        **kwargs,
    ) -> Ax:
        """Plot the simulation boundary conditions as lines on a plane
           defined by one nonzero x,y,z coordinate.

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
        **kwargs
            Optional keyword arguments passed to the matplotlib ``LineCollection``.
            For details on accepted values, refer to
            `Matplotlib's documentation <https://tinyurl.com/2p97z4cn>`_.

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.
        """

        def set_plot_params(boundary_edge, lim, side, thickness):
            """Return the line plot properties such as color and opacity based on the boundary"""
            if isinstance(boundary_edge, PECBoundary):
                plot_params = plot_params_pec.copy(deep=True)
            elif isinstance(boundary_edge, PMCBoundary):
                plot_params = plot_params_pmc.copy(deep=True)
            elif isinstance(boundary_edge, BlochBoundary):
                plot_params = plot_params_bloch.copy(deep=True)
            else:
                plot_params = PlotParams(alpha=0)

            # expand axis limit so that the axis ticks and labels aren't covered
            new_lim = lim
            if plot_params.alpha != 0:
                if side == -1:
                    new_lim = lim - thickness
                elif side == 1:
                    new_lim = lim + thickness

            return plot_params, new_lim

        boundaries = self.boundary_spec.to_list

        normal_axis, _ = self.parse_xyz_kwargs(x=x, y=y, z=z)
        _, (dim_u, dim_v) = self.pop_axis([0, 1, 2], axis=normal_axis)

        umin, umax = ax.get_xlim()
        vmin, vmax = ax.get_ylim()

        size_factor = 1.0 / 35.0
        thickness_u = (umax - umin) * size_factor
        thickness_v = (vmax - vmin) * size_factor

        # boundary along the u axis, minus side
        plot_params, ulim_minus = set_plot_params(boundaries[dim_u][0], umin, -1, thickness_u)
        rect = mpl.patches.Rectangle(
            xy=(umin - thickness_u, vmin),
            width=thickness_u,
            height=(vmax - vmin),
            **plot_params.to_kwargs(),
            **kwargs,
        )
        ax.add_patch(rect)

        # boundary along the u axis, plus side
        plot_params, ulim_plus = set_plot_params(boundaries[dim_u][1], umax, 1, thickness_u)
        rect = mpl.patches.Rectangle(
            xy=(umax, vmin),
            width=thickness_u,
            height=(vmax - vmin),
            **plot_params.to_kwargs(),
            **kwargs,
        )
        ax.add_patch(rect)

        # boundary along the v axis, minus side
        plot_params, vlim_minus = set_plot_params(boundaries[dim_v][0], vmin, -1, thickness_v)
        rect = mpl.patches.Rectangle(
            xy=(umin, vmin - thickness_v),
            width=(umax - umin),
            height=thickness_v,
            **plot_params.to_kwargs(),
            **kwargs,
        )
        ax.add_patch(rect)

        # boundary along the v axis, plus side
        plot_params, vlim_plus = set_plot_params(boundaries[dim_v][1], vmax, 1, thickness_v)
        rect = mpl.patches.Rectangle(
            xy=(umin, vmax),
            width=(umax - umin),
            height=thickness_v,
            **plot_params.to_kwargs(),
            **kwargs,
        )
        ax.add_patch(rect)

        # ax = self._set_plot_bounds(ax=ax, x=x, y=y, z=z)
        ax.set_xlim([ulim_minus, ulim_plus])
        ax.set_ylim([vlim_minus, vlim_plus])

        return ax

    @cached_property
    def frequency_range(self) -> FreqBound:
        """Range of frequencies spanning all sources' frequency dependence.

        Returns
        -------
        Tuple[float, float]
            Minumum and maximum frequencies of the power spectrum of the sources.
        """
        source_ranges = [source.source_time.frequency_range() for source in self.sources]
        freq_min = min((freq_range[0] for freq_range in source_ranges), default=0.0)
        freq_max = max((freq_range[1] for freq_range in source_ranges), default=0.0)

        return (freq_min, freq_max)

    def plot_3d(self, width=800, height=800) -> None:
        """Render 3D plot of ``Simulation`` (in jupyter notebook only).
        Parameters
        ----------
        width : float = 800
            width of the 3d view dom's size
        height : float = 800
            height of the 3d view dom's size

        """
        return plot_sim_3d(self, width=width, height=height)

    """ Discretization """

    @cached_property
    def dt(self) -> float:
        """Simulation time step (distance).

        Returns
        -------
        float
            Time step (seconds).
        """
        dl_mins = [np.min(sizes) for sizes in self.grid.sizes.to_list]
        dl_sum_inv_sq = sum(1 / dl**2 for dl in dl_mins)
        dl_avg = 1 / np.sqrt(dl_sum_inv_sq)
        # material factor
        n_cfl = min(min(mat.n_cfl for mat in self.scene.mediums), 1)
        return n_cfl * self.courant * dl_avg / C_0

    @cached_property
    def tmesh(self) -> Coords1D:
        """FDTD time stepping points.

        Returns
        -------
        np.ndarray
            Times (seconds) that the simulation time steps through.
        """
        dt = self.dt
        return np.arange(0.0, self.run_time + dt, dt)

    @cached_property
    def num_time_steps(self) -> int:
        """Number of time steps in simulation."""

        return len(self.tmesh)

    def _grid_corrections_2dmaterials(self, grid: Grid) -> Grid:
        """Correct the grid if 2d materials are present, using their volumetric equivalents."""
        if not any(isinstance(structure.medium, Medium2D) for structure in self.structures):
            return grid

        # when there are 2D materials, need to make grid again with volumetric_structures
        # generated using the first grid

        volumetric_structures = [Structure(geometry=self.geometry, medium=self.medium)]
        volumetric_structures += self._volumetric_structures_grid(grid)

        volumetric_grid = self.grid_spec.make_grid(
            structures=volumetric_structures,
            symmetry=self.symmetry,
            sources=self.sources,
            num_pml_layers=self.num_pml_layers,
        )

        # Handle 2D materials if ``AutoGrid`` is used for in-plane directions
        # must use original grid for the normal directions of all 2d materials
        grid_axes = [False, False, False]
        # must use volumetric grid for the ``AutoGrid`` in-plane directions of 2d materials
        volumetric_grid_axes = [False, False, False]
        with log as consolidated_logger:
            for structure in self.structures:
                if isinstance(structure.medium, Medium2D):

                    normal = structure.geometry._normal_2dmaterial
                    grid_axes[normal] = True
                    for axis, grid_axis in enumerate(
                        [self.grid_spec.grid_x, self.grid_spec.grid_y, self.grid_spec.grid_z]
                    ):
                        if isinstance(grid_axis, AutoGrid):
                            if axis != normal:
                                volumetric_grid_axes[axis] = True
                            else:
                                consolidated_logger.warning(
                                    "Using 'AutoGrid' for the normal direction of a 2D material "
                                    "may generate a grid that is not sufficiently fine."
                                )
        coords_all = [None, None, None]
        for axis in range(3):
            if grid_axes[axis] and volumetric_grid_axes[axis]:
                raise ValidationError(
                    "Unable to generate grid. Cannot use 'AutoGrid' for "
                    "an axis that is in-plane to one 2D material and normal to another."
                )
            if volumetric_grid_axes[axis]:
                coords_all[axis] = volumetric_grid.boundaries.to_list[axis]
            else:
                coords_all[axis] = grid.boundaries.to_list[axis]

        return Grid(boundaries=Coords(**dict(zip("xyz", coords_all))))

    @cached_property
    def grid(self) -> Grid:
        """FDTD grid spatial locations and information.

        Returns
        -------
        :class:`.Grid`
            :class:`.Grid` storing the spatial locations relevant to the simulation.
        """

        # Add a simulation Box as the first structure
        structures = [Structure(geometry=self.geometry, medium=self.medium)]
        structures += self.structures

        grid = self.grid_spec.make_grid(
            structures=structures,
            symmetry=self.symmetry,
            periodic=self._periodic,
            sources=self.sources,
            num_pml_layers=self.num_pml_layers,
        )

        # This would AutoGrid the in-plane directions of the 2D materials
        # return self._grid_corrections_2dmaterials(grid)
        return grid

    @cached_property
    def num_cells(self) -> int:
        """Number of cells in the simulation.

        Returns
        -------
        int
            Number of yee cells in the simulation.
        """

        return np.prod(self.grid.num_cells, dtype=np.int64)

    @cached_property
    def wvl_mat_min(self) -> float:
        """Minimum wavelength in the material.

        Returns
        -------
        float
            Minimum wavelength in the material (microns).
        """
        freq_max = max(source.source_time.freq0 for source in self.sources)
        wvl_min = C_0 / freq_max
        eps_max = max(abs(structure.medium.eps_model(freq_max)) for structure in self.structures)
        n_max, _ = AbstractMedium.eps_complex_to_nk(eps_max)
        return wvl_min / n_max

    @cached_property
    def complex_fields(self) -> bool:
        """Whether complex fields are used in the simulation. Currently this only happens when there
        are Bloch boundaries.

        Returns
        -------
        bool
            Whether the time-stepping fields are real or complex.
        """
        if any(isinstance(boundary[0], BlochBoundary) for boundary in self.boundary_spec.to_list):
            return True
        return False

    @cached_property
    def nyquist_step(self) -> int:
        """Maximum number of discrete time steps to keep sampling below Nyquist limit.

        Returns
        -------
        int
            The largest ``N`` such that ``N * self.dt`` is below the Nyquist limit.
        """

        # source frequency upper bound
        freq_source_max = self.frequency_range[1]
        # monitor frequency upper bound
        freq_monitor_max = max(
            (
                monitor.frequency_range[1]
                for monitor in self.monitors
                if isinstance(monitor, FreqMonitor) and not isinstance(monitor, PermittivityMonitor)
            ),
            default=0.0,
        )
        # combined frequency upper bound
        freq_max = max(freq_source_max, freq_monitor_max)

        if freq_max > 0:
            nyquist_step = int(1 / (2 * freq_max) / self.dt) - 1
            nyquist_step = max(1, nyquist_step)
        else:
            nyquist_step = 1

        return nyquist_step

    def _subgrid(self, span_inds: np.ndarray, grid: Grid = None):
        """Take a subgrid of the simulation grid with cell span defined by ``span_inds`` along the
        three dimensions. Optionally, a grid different from the simulation grid can be provided.
        The ``span_inds`` can also extend beyond the grid, in which case the grid is padded based
        on the boundary conditions of the simulation along the different dimensions."""

        if not grid:
            grid = self.grid

        boundary_dict = {}
        for idim, (dim, periodic) in enumerate(zip("xyz", self._periodic)):
            ind_beg, ind_end = span_inds[idim]
            # ind_end + 1 because we are selecting cell boundaries not cells
            boundary_dict[dim] = grid.extended_subspace(idim, ind_beg, ind_end + 1, periodic)
        return Grid(boundaries=Coords(**boundary_dict))

    def _snap_zero_dim(self, grid: Grid):
        """Snap a grid to the simulation center along any dimension along which simulation is
        effectively 0D, defined as having a single pixel. This is more general than just checking
        size = 0."""
        size_snapped = [
            size if num_cells > 1 else 0 for num_cells, size in zip(self.grid.num_cells, self.size)
        ]
        return grid.snap_to_box_zero_dim(Box(center=self.center, size=size_snapped))

    @cached_property
    def _periodic(self) -> Tuple[bool, bool, bool]:
        """For each dimension, ``True`` if periodic/Bloch boundaries and ``False`` otherwise.
        We check on both sides but in practice there should be no cases in which a periodic/Bloch
        BC is on one side only. This is explicitly validated for Bloch, and implicitly done for
        periodic, in which case we allow PEC/PMC on the other side, but we replace the periodic
        boundary with another PEC/PMC plane upon initialization."""
        periodic = []
        for bcs_1d in self.boundary_spec.to_list:
            periodic.append(all(isinstance(bcs, (Periodic, BlochBoundary)) for bcs in bcs_1d))
        return periodic

    def _discretize_grid(self, box: Box, grid: Grid, extend: bool = False) -> Grid:
        """Grid containing only cells that intersect with a :class:`Box`.

        As opposed to ``Simulation.discretize``, this function operates on a ``grid``
        which may not be the grid of the simulation.
        """

        if not self.intersects(box):
            log.error(f"Box {box} is outside simulation, cannot discretize.")

        span_inds = grid.discretize_inds(box=box, extend=extend)
        return self._subgrid(span_inds=span_inds, grid=grid)

    def _discretize_inds_monitor(self, monitor: Monitor):
        """Start and stopping indexes for the cells where data needs to be recorded to fully cover
        a ``monitor``. This is used during the solver run. The final grid on which a monitor data
        lives is computed in ``discretize_monitor``, with the difference being that 0-sized
        dimensions of the monitor or the simulation are snapped in post-processing."""

        # Expand monitor size slightly to break numerical precision in favor of always having
        # enough data to span the full monitor.
        expand_size = [size + fp_eps if size > fp_eps else size for size in monitor.size]
        box_expanded = Box(center=monitor.center, size=expand_size)
        # Discretize without extension for now
        span_inds = np.array(self.grid.discretize_inds(box_expanded, extend=False))

        # Now add extensions, which are specific for monitors and are determined such that data
        # colocated to grid boundaries can be interpolated anywhere inside the monitor.
        # We always need to expand on the right.
        span_inds[:, 1] += 1
        # Non-colocating monitors also need to expand on the left.
        if not monitor.colocate:
            span_inds[:, 0] -= 1
        return span_inds

    def discretize_monitor(self, monitor: Monitor) -> Grid:
        """Grid on which monitor data corresponding to a given monitor will be computed."""
        span_inds = self._discretize_inds_monitor(monitor)
        grid_snapped = self._subgrid(span_inds=span_inds).snap_to_box_zero_dim(monitor)
        grid_snapped = self._snap_zero_dim(grid=grid_snapped)
        return grid_snapped

    def discretize(self, box: Box, extend: bool = False) -> Grid:
        """Grid containing only cells that intersect with a :class:`.Box`.

        Parameters
        ----------
        box : :class:`.Box`
            Rectangular geometry within simulation to discretize.
        extend : bool = False
            If ``True``, ensure that the returned indexes extend sufficiently in every direction to
            be able to interpolate any field component at any point within the ``box``, for field
            components sampled on the Yee grid.

        Returns
        -------
        :class:`Grid`
            The FDTD subgrid containing simulation points that intersect with ``box``.
        """
        return self._discretize_grid(box=box, grid=self.grid, extend=extend)

    def epsilon(
        self,
        box: Box,
        coord_key: str = "centers",
        freq: float = None,
    ) -> xr.DataArray:
        """Get array of permittivity at volume specified by box and freq.

        Parameters
        ----------
        box : :class:`.Box`
            Rectangular geometry specifying where to measure the permittivity.
        coord_key : str = 'centers'
            Specifies at what part of the grid to return the permittivity at.
            Accepted values are ``{'centers', 'boundaries', 'Ex', 'Ey', 'Ez', 'Exy', 'Exz', 'Eyx',
            'Eyz', 'Ezx', Ezy'}``. The field values (eg. 'Ex') correspond to the correponding field
            locations on the yee lattice. If field values are selected, the corresponding diagonal
            (eg. `eps_xx` in case of `Ex`) or off-diagonal (eg. `eps_xy` in case of `Exy`) epsilon
            component from the epsilon tensor is returned. Otherwise, the average of the main
            values is returned.
        freq : float = None
            The frequency to evaluate the mediums at.
            If not specified, evaluates at infinite frequency.
        Returns
        -------
        xarray.DataArray
            Datastructure containing the relative permittivity values and location coordinates.
            For details on xarray DataArray objects,
            refer to `xarray's Documentaton <https://tinyurl.com/2zrzsp7b>`_.
        """

        sub_grid = self.discretize(box)
        return self.epsilon_on_grid(grid=sub_grid, coord_key=coord_key, freq=freq)

    def epsilon_on_grid(
        self,
        grid: Grid,
        coord_key: str = "centers",
        freq: float = None,
    ) -> xr.DataArray:
        """Get array of permittivity at a given freq on a given grid.

        Parameters
        ----------
        grid : :class:`.Grid`
            Grid specifying where to measure the permittivity.
        coord_key : str = 'centers'
            Specifies at what part of the grid to return the permittivity at.
            Accepted values are ``{'centers', 'boundaries', 'Ex', 'Ey', 'Ez', 'Exy', 'Exz', 'Eyx',
            'Eyz', 'Ezx', Ezy'}``. The field values (eg. 'Ex') correspond to the correponding field
            locations on the yee lattice. If field values are selected, the corresponding diagonal
            (eg. `eps_xx` in case of `Ex`) or off-diagonal (eg. `eps_xy` in case of `Exy`) epsilon
            component from the epsilon tensor is returned. Otherwise, the average of the main
            values is returned.
        freq : float = None
            The frequency to evaluate the mediums at.
            If not specified, evaluates at infinite frequency.
        Returns
        -------
        xarray.DataArray
            Datastructure containing the relative permittivity values and location coordinates.
            For details on xarray DataArray objects,
            refer to `xarray's Documentaton <https://tinyurl.com/2zrzsp7b>`_.
        """

        grid_cells = np.prod(grid.num_cells)
        num_structures = len(self.structures)
        if grid_cells > NUM_CELLS_WARN_EPSILON:
            log.warning(
                f"Requested grid contains {int(grid_cells):.2e} grid cells. "
                "Epsilon calculation may be slow."
            )
        if num_structures > NUM_STRUCTURES_WARN_EPSILON:
            log.warning(
                f"Simulation contains {num_structures:.2e} structures. "
                "Epsilon calculation may be slow."
            )

        def get_eps(structure: Structure, frequency: float, coords: Coords):
            """Select the correct epsilon component if field locations are requested."""
            if coord_key[0] != "E":
                return np.mean(structure.eps_diagonal(frequency, coords), axis=0)
            row = ["x", "y", "z"].index(coord_key[1])
            if len(coord_key) == 2:  # diagonal component in case of Ex, Ey, and Ez
                col = row
            else:  # off-diagonal component in case of Exy, Exz, Eyx, etc
                col = ["x", "y", "z"].index(coord_key[2])
            return structure.eps_comp(row, col, frequency, coords)

        def make_eps_data(coords: Coords):
            """returns epsilon data on grid of points defined by coords"""
            arrays = (np.array(coords.x), np.array(coords.y), np.array(coords.z))
            eps_background = get_eps(
                structure=self.scene.background_structure, frequency=freq, coords=coords
            )
            shape = tuple(len(array) for array in arrays)
            eps_array = eps_background * np.ones(shape, dtype=complex)
            # replace 2d materials with volumetric equivalents
            with log as consolidated_logger:
                for structure in self.volumetric_structures:
                    # Indexing subset within the bounds of the structure

                    inds = structure.geometry._inds_inside_bounds(*arrays)

                    # Get permittivity on meshgrid over the reduced coordinates
                    coords_reduced = tuple(arr[ind] for arr, ind in zip(arrays, inds))
                    if any(coords.size == 0 for coords in coords_reduced):
                        continue

                    red_coords = Coords(**dict(zip("xyz", coords_reduced)))
                    eps_structure = get_eps(structure=structure, frequency=freq, coords=red_coords)

                    if structure.medium.nonlinear_spec is not None:
                        consolidated_logger.warning(
                            "Evaluating permittivity of a nonlinear "
                            "medium ignores the nonlinearity."
                        )

                    if isinstance(structure.geometry, TriangleMesh):
                        consolidated_logger.warning(
                            "Client-side permittivity of a 'TriangleMesh' may be "
                            "inaccurate if the mesh is not unionized. We recommend unionizing "
                            "all meshes before import. A 'PermittivityMonitor' can be used to "
                            "obtain the true permittivity and check that the surface mesh is "
                            "loaded correctly."
                        )

                    # Update permittivity array at selected indexes within the geometry
                    is_inside = structure.geometry.inside_meshgrid(*coords_reduced)
                    eps_array[inds][is_inside] = (eps_structure * is_inside)[is_inside]

            coords = dict(zip("xyz", arrays))
            return xr.DataArray(eps_array, coords=coords, dims=("x", "y", "z"))

        # combine all data into dictionary
        if coord_key[0] == "E":
            # off-diagonal componets are sampled at respective locations (eg. `eps_xy` at `Ex`)
            coords = grid[coord_key[0:2]]
        else:
            coords = grid[coord_key]
        return make_eps_data(coords)

    @property
    def custom_datasets(self) -> List[Dataset]:
        """List of custom datasets for verification purposes. If the list is not empty, then
        the simulation needs to be exported to hdf5 to store the data.
        """
        datasets_source_time = [
            src.source_time.source_time_dataset
            for src in self.sources
            if isinstance(src.source_time, CustomSourceTime)
        ]
        datasets_field_source = [
            src.field_dataset for src in self.sources if isinstance(src, CustomFieldSource)
        ]
        datasets_current_source = [
            src.current_dataset for src in self.sources if isinstance(src, CustomCurrentSource)
        ]
        datasets_medium = [
            mat
            for mat in self.scene.mediums
            if isinstance(mat, AbstractCustomMedium) or mat.time_modulated
        ]
        datasets_geometry = []

        for struct in self.structures:
            for geometry in traverse_geometries(struct.geometry):
                if isinstance(geometry, TriangleMesh):
                    datasets_geometry += [geometry.mesh_dataset]

        return (
            datasets_source_time
            + datasets_field_source
            + datasets_current_source
            + datasets_medium
            + datasets_geometry
        )

    def _volumetric_structures_grid(self, grid: Grid) -> Tuple[Structure]:
        """Generate a tuple of structures wherein any 2D materials are converted to 3D
        volumetric equivalents, using ``grid`` as the simulation grid."""

        if not any(isinstance(medium, Medium2D) for medium in self.scene.mediums):
            return self.structures

        def get_bounds(geom: Geometry, axis: Axis) -> Tuple[float, float]:
            """Get the bounds of a geometry in the axis direction."""
            return (geom.bounds[0][axis], geom.bounds[1][axis])

        def set_bounds(geom: Geometry, bounds: Tuple[float, float], axis: Axis) -> Geometry:
            """Set the bounds of a geometry in the axis direction."""
            if isinstance(geom, Box):
                new_center = list(geom.center)
                new_center[axis] = (bounds[0] + bounds[1]) / 2
                new_size = list(geom.size)
                new_size[axis] = bounds[1] - bounds[0]
                return geom.updated_copy(center=new_center, size=new_size)
            if isinstance(geom, PolySlab):
                return geom.updated_copy(slab_bounds=bounds)
            if isinstance(geom, Cylinder):
                new_center = list(geom.center)
                new_center[axis] = (bounds[0] + bounds[1]) / 2
                new_length = bounds[1] - bounds[0]
                return geom.updated_copy(center=new_center, length=new_length)
            raise ValidationError(
                "'Medium2D' is only compatible with 'Box', 'PolySlab', or 'Cylinder' geometry."
            )

        def get_dls(geom: Geometry, axis: Axis, num_dls: int) -> float:
            """Get grid size around the 2D material."""
            dls = self._discretize_grid(Box.from_bounds(*geom.bounds), grid=grid).sizes.to_list[
                axis
            ]
            if len(dls) != num_dls:
                raise Tidy3dError(
                    "Failed to detect grid size around the 2D material. "
                    "Can't generate volumetric equivalent for this simulation. "
                    "If you received this error, please create an issue in the Tidy3D "
                    "github repository."
                )
            return dls

        def snap_to_grid(geom: Geometry, axis: Axis) -> Geometry:
            """Snap a 2D material to the Yee grid."""
            new_centers = self._discretize_grid(
                Box.from_bounds(*geom.bounds), grid=grid
            ).boundaries.to_list[axis]
            new_center = new_centers[np.argmin(abs(new_centers - get_bounds(geom, axis)[0]))]
            return set_bounds(geom, (new_center, new_center), axis)

        def get_neighboring_media(
            geom: Geometry, axis: Axis, structures: List[Structure]
        ) -> Tuple[List[MediumType3D], List[float]]:
            """Find the neighboring material properties and grid sizes."""
            dl = get_dls(geom, axis, 1)[0]
            center = get_bounds(geom, axis)[0]
            thickness = dl * DIST_NEIGHBOR_REL_2D_MED
            thickened_geom = set_bounds(
                geom, bounds=(center - thickness / 2, center + thickness / 2), axis=axis
            )
            grid_sizes = get_dls(thickened_geom, axis, 2)
            dls_signed = [-grid_sizes[0], grid_sizes[1]]
            neighbors = []
            for _, dl_signed in enumerate(dls_signed):
                geom_shifted = set_bounds(
                    geom, bounds=(center + dl_signed, center + dl_signed), axis=axis
                )
                media = Scene.intersecting_media(Box.from_bounds(*geom_shifted.bounds), structures)
                if len(media) > 1:
                    raise SetupError(
                        "2D materials do not support multiple neighboring media on a side. "
                        "Please split the 2D material into multiple smaller 2D materials, one "
                        "for each background medium."
                    )
                medium_side = Medium() if len(media) == 0 else list(media)[0]
                neighbors.append(medium_side)
            return (neighbors, grid_sizes)

        simulation_background = Structure(geometry=self.geometry, medium=self.medium)
        background_structures = [simulation_background]
        new_structures = []
        for structure in self.structures:
            if not isinstance(structure.medium, Medium2D):
                # found a 3D material; keep it
                background_structures.append(structure)
                new_structures.append(structure)
                continue
            # otherwise, found a 2D material; replace it with volumetric equivalent
            axis = structure.geometry._normal_2dmaterial

            # snap monolayer to grid
            geometry = snap_to_grid(structure.geometry, axis)
            center = get_bounds(geometry, axis)[0]

            # get neighboring media and grid sizes
            (neighbors, dls) = get_neighboring_media(geometry, axis, background_structures)

            if not structure.medium.is_pec:
                new_bounds = (center - dls[0] / 2, center + dls[1] / 2)
            else:
                new_bounds = (center, center)

            new_geometry = set_bounds(structure.geometry, bounds=new_bounds, axis=axis)

            new_medium = structure.medium.volumetric_equivalent(
                axis=axis, adjacent_media=neighbors, adjacent_dls=dls
            )
            new_structures.append(structure.updated_copy(geometry=new_geometry, medium=new_medium))

        return tuple(new_structures)

    @cached_property
    def volumetric_structures(self) -> Tuple[Structure]:
        """Generate a tuple of structures wherein any 2D materials are converted to 3D
        volumetric equivalents."""
        return self._volumetric_structures_grid(self.grid)

    @cached_property
    def allow_gain(self) -> bool:
        """``True`` if any of the mediums in the simulation allows gain."""

        for medium in self.scene.mediums:
            if isinstance(medium, AnisotropicMedium):
                if np.any([med.allow_gain for med in [medium.xx, medium.yy, medium.zz]]):
                    return True
            elif medium.allow_gain:
                return True
        return False

    def perturbed_mediums_copy(
        self,
        temperature: SpatialDataArray = None,
        electron_density: SpatialDataArray = None,
        hole_density: SpatialDataArray = None,
        interp_method: InterpMethod = "linear",
    ) -> Simulation:
        """Return a copy of the simulation with heat and/or charge data applied to all mediums
        that have perturbation models specified. That is, such mediums will be replaced with
        spatially dependent custom mediums that reflect perturbation effects. Any of temperature,
        electron_density, and hole_density can be ``None``. All provided fields must have identical
        coords.

        Parameters
        ----------
        temperature : SpatialDataArray = None
            Temperature field data.
        electron_density : SpatialDataArray = None
            Electron density field data.
        hole_density : SpatialDataArray = None
            Hole density field data.
        interp_method : :class:`.InterpMethod`, optional
            Interpolation method to obtain heat and/or charge values that are not supplied
            at the Yee grids.

        Returns
        -------
        Simulation
            Simulation after application of heat and/or charge data.
        """

        sim_dict = self.dict()
        structures = self.structures
        sim_bounds = self.simulation_bounds
        array_dict = {
            "temperature": temperature,
            "electron_density": electron_density,
            "hole_density": hole_density,
        }

        # For each structure made of mediums with perturbation models, convert those mediums into
        # spatially dependent mediums by selecting minimal amount of heat and charge data points
        # covering the structure, and create a new structure containing the resulting custom medium
        new_structures = []
        for s_ind, structure in enumerate(structures):
            med = structure.medium
            if isinstance(med, AbstractPerturbationMedium):
                # get structure's bounding box
                s_bounds = structure.geometry.bounds

                bounds = [
                    np.max([sim_bounds[0], s_bounds[0]], axis=0),
                    np.min([sim_bounds[1], s_bounds[1]], axis=0),
                ]

                # for each structure select a minimal subset of data that covers it
                restricted_arrays = {}

                for name, array in array_dict.items():
                    if array is not None:
                        restricted_arrays[name] = array.sel_inside(bounds)

                        # check provided data fully cover structure
                        if not array.does_cover(bounds):
                            log.warning(
                                f"Provided '{name}' does not fully cover structures[{s_ind}]."
                            )

                new_medium = med.perturbed_copy(**restricted_arrays, interp_method=interp_method)
                new_structure = structure.updated_copy(medium=new_medium)
                new_structures.append(new_structure)
            else:
                new_structures.append(structure)

        sim_dict["structures"] = new_structures

        # do the same for background medium if it a medium with perturbation models.
        med = self.medium
        if isinstance(med, AbstractPerturbationMedium):

            # get simulation's bounding box
            bounds = sim_bounds

            # for each structure select a minimal subset of data that covers it
            restricted_arrays = {}

            for name, array in array_dict.items():
                if array is not None:
                    restricted_arrays[name] = array.sel_inside(bounds)

                    # check provided data fully cover simulation
                    if not array.does_cover(bounds):
                        log.warning(f"Provided '{name}' does not fully cover simulation domain.")

            sim_dict["medium"] = med.perturbed_copy(
                **restricted_arrays, interp_method=interp_method
            )

        return Simulation.parse_obj(sim_dict)

    @classmethod
    def from_scene(cls, scene: Scene, **kwargs) -> Simulation:
        """Create a simulation from a :class:.`Scene` instance. Must provide additional parameters
        to define a valid simulation (for example, ``run_time``, ``grid_spec``, etc).

        Parameters
        ----------
        scene : :class:.`Scene`
            Size of object in x, y, and z directions.
        **kwargs
            Other arguments

        Example
        -------
        >>> from tidy3d import Scene, Medium, Box, Structure, GridSpec
        >>> box = Structure(
        ...     geometry=Box(center=(0, 0, 0), size=(1, 2, 3)),
        ...     medium=Medium(permittivity=5),
        ... )
        >>> scene = Scene(
        ...     structures=[box],
        ...     medium=Medium(permittivity=3),
        ... )
        >>> sim = Simulation.from_scene(
        ...     scene=scene,
        ...     center=(0, 0, 0),
        ...     size=(5, 6, 7),
        ...     run_time=1e-12,
        ...     grid_spec=GridSpec.uniform(dl=0.4),
        ... )
        """
        return Simulation(
            structures=scene.structures,
            medium=scene.medium,
            **kwargs,
        )
