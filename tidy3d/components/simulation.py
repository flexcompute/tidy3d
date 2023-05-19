# pylint: disable=too-many-lines, too-many-arguments, too-many-statements
""" Container holding all information about simulation and its components"""
from __future__ import annotations

from typing import Dict, Tuple, List, Set, Union
from math import isclose

import pydantic
import numpy as np
import xarray as xr
import matplotlib.pylab as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .base import cached_property
from .validators import assert_unique_names, assert_objects_in_sim_bounds
from .validators import validate_mode_objects_symmetry
from .geometry import Box, TriangleMesh, Geometry, PolySlab, Cylinder, GeometryGroup
from .types import Ax, Shapely, FreqBound, Axis, annotate_type, Symmetry
from .grid.grid import Coords1D, Grid, Coords
from .grid.grid_spec import GridSpec, UniformGrid, AutoGrid
from .medium import Medium, MediumType, AbstractMedium, PECMedium
from .medium import AbstractCustomMedium, Medium2D, MediumType3D
from .medium import AnisotropicMedium, FullyAnisotropicMedium
from .boundary import BoundarySpec, BlochBoundary, PECBoundary, PMCBoundary, Periodic
from .boundary import PML, StablePML, Absorber, AbsorberSpec
from .structure import Structure
from .source import SourceType, PlaneWave, GaussianBeam, AstigmaticGaussianBeam, CustomFieldSource
from .source import CustomCurrentSource, CustomSourceTime
from .source import TFSF, Source
from .monitor import MonitorType, Monitor, FreqMonitor, SurfaceIntegrationMonitor
from .monitor import AbstractFieldMonitor, DiffractionMonitor, AbstractFieldProjectionMonitor
from .data.dataset import Dataset
from .viz import add_ax_if_none, equal_aspect

from .viz import MEDIUM_CMAP, STRUCTURE_EPS_CMAP, PlotParams, plot_params_symmetry, polygon_path
from .viz import plot_params_structure, plot_params_pml, plot_params_override_structures
from .viz import plot_params_pec, plot_params_pmc, plot_params_bloch, plot_sim_3d

from ..version import __version__
from ..constants import C_0, SECOND, inf
from ..exceptions import Tidy3dKeyError, SetupError, ValidationError, Tidy3dError
from ..log import log
from ..updater import Updater


# minimum number of grid points allowed per central wavelength in a medium
MIN_GRIDS_PER_WVL = 6.0

# maximum number of mediums supported
MAX_NUM_MEDIUMS = 65530

# maximum numbers of simulation parameters
MAX_TIME_STEPS = 1e7
WARN_TIME_STEPS = 1e6
MAX_GRID_CELLS = 20e9
MAX_CELLS_TIMES_STEPS = 1e16
WARN_MONITOR_DATA_SIZE_GB = 10
MAX_SIMULATION_DATA_SIZE_GB = 50

# number of grid cells at which we warn about slow Simulation.epsilon()
NUM_CELLS_WARN_EPSILON = 100_000_000
# number of structures at which we warn about slow Simulation.epsilon()
NUM_STRUCTURES_WARN_EPSILON = 10_000
# for 2d materials. to find neighboring media, search a distance on either side
# equal to this times the grid size
DIST_NEIGHBOR_REL_2D_MED = 1e-5

# height of the PML plotting boxes along any dimensions where sim.size[dim] == 0
PML_HEIGHT_FOR_0_DIMS = 0.02

# allow some numerical flexibility before warning about CustomSourceTime
# in units of dt
CUSTOMSOURCETIME_TOL = 1.1


class Simulation(Box):  # pylint:disable=too-many-public-methods
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
    ...         FieldMonitor(size=(0, 0, 0), center=(0, 0, 0), freqs=[1.5e14, 2e14], name='point'),
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

    medium: MediumType3D = pydantic.Field(
        Medium(),
        title="Background Medium",
        description="Background medium of simulation, defaults to vacuum if not specified.",
    )

    symmetry: Tuple[Symmetry, Symmetry, Symmetry] = pydantic.Field(
        (0, 0, 0),
        title="Symmetries",
        description="Tuple of integers defining reflection symmetry across a plane "
        "bisecting the simulation domain normal to the x-, y-, and z-axis "
        "at the simulation center of each axis, respectively. "
        "Each element can be ``0`` (no symmetry), ``1`` (even, i.e. 'PMC' symmetry) or "
        "``-1`` (odd, i.e. 'PEC' symmetry). "
        "Note that the vectorial nature of the fields must be taken into account to correctly "
        "determine the symmetry value.",
    )

    structures: Tuple[Structure, ...] = pydantic.Field(
        (),
        title="Structures",
        description="Tuple of structures present in simulation. "
        "Note: Structures defined later in this list override the "
        "simulation material properties in regions of spatial overlap.",
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
        "periodic boundary conditions are applied on all sides. Default will change to PML in 2.0 "
        "so explicitly setting the boundaries is recommended.",
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

    version: str = pydantic.Field(
        __version__,
        title="Version",
        description="String specifying the front end version number.",
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

    _structures_in_bounds = assert_objects_in_sim_bounds("structures", error=False)
    _sources_in_bounds = assert_objects_in_sim_bounds("sources")
    _monitors_in_bounds = assert_objects_in_sim_bounds("monitors")
    _mode_sources_symmetries = validate_mode_objects_symmetry("sources")
    _mode_monitors_symmetries = validate_mode_objects_symmetry("monitors")

    # make sure all names are unique
    _unique_structure_names = assert_unique_names("structures")
    _unique_source_names = assert_unique_names("sources")
    _unique_monitor_names = assert_unique_names("monitors")

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

    # pylint: disable=too-many-locals
    @pydantic.validator("boundary_spec", always=True)
    def plane_wave_boundaries(cls, val, values):
        """Error if there are plane wave sources incompatible with boundary conditions."""
        boundaries = val.to_list
        sources = values.get("sources")
        size = values.get("size")
        sim_medium = values.get("medium")
        structures = values.get("structures")
        for source in sources:
            if not isinstance(source, PlaneWave):
                continue

            _, tan_dirs = cls.pop_axis([0, 1, 2], axis=source.injection_axis)
            medium_set = cls.intersecting_media(source, structures)
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
            medium_set = cls.intersecting_media(temp_src, structures)
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
        """Warn if an absorbing boundary is used along a zero dimension."""
        boundaries = val.to_list
        size = values.get("size")
        for dim, (boundary, size_dim) in enumerate(zip(boundaries, size)):
            num_absorbing_bdries = sum(isinstance(bnd, AbsorberSpec) for bnd in boundary)
            if num_absorbing_bdries > 0 and size_dim == 0:
                log.warning(
                    f"If the simulation is intended to be 2D in the plane normal to the "
                    f"{'xyz'[dim]} axis, using a PML or absorbing boundary along that axis "
                    f"is incorrect. Consider using a 'Periodic' boundary along {'xyz'[dim]}."
                )
        return val

    @pydantic.validator("structures", always=True)
    def _validate_num_mediums(cls, val):
        """Error if too many mediums present."""

        if val is None:
            return val

        mediums = {structure.medium for structure in val}
        if len(mediums) > MAX_NUM_MEDIUMS:
            raise SetupError(
                f"Tidy3d only supports {MAX_NUM_MEDIUMS} distinct mediums."
                f"{len(mediums)} were supplied."
            )

        return val

    @pydantic.validator("structures", always=True)
    def _structures_not_at_edges(cls, val, values):
        """Warn if any structures lie at the simulation boundaries."""

        if val is None:
            return val

        sim_box = Box(size=values.get("size"), center=values.get("center"))
        sim_bound_min, sim_bound_max = sim_box.bounds
        sim_bounds = list(sim_bound_min) + list(sim_bound_max)

        with log as consolidated_logger:
            for istruct, structure in enumerate(val):
                struct_bound_min, struct_bound_max = structure.geometry.bounds
                struct_bounds = list(struct_bound_min) + list(struct_bound_max)

                for sim_val, struct_val in zip(sim_bounds, struct_bounds):

                    if isclose(sim_val, struct_val):
                        consolidated_logger.warning(
                            f"Structure at structures[{istruct}] has bounds that extend exactly to "
                            "simulation edges. This can cause unexpected behavior. "
                            "If intending to extend the structure to infinity along one dimension, "
                            "use td.inf as a size variable instead to make this explicit."
                        )

        return val

    @pydantic.validator("boundary_spec", always=True)
    def _structures_not_close_pml(cls, val, values):  # pylint:disable=too-many-locals
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
                    "To avoid inaccurate results, please increase gap between "
                    "any structures and PML or fully extend structure through the pml."
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
                        else:
                            medium_str = f"The medium associated with structures[{medium_index-1}]"

                        consolidated_logger.warning(
                            f"{medium_str} has a frequency range: ({fmin_med:2e}, {fmax_med:2e}) "
                            "(Hz) that does not fully cover the frequencies contained in "
                            f"monitors[{monitor_index}]. "
                            "This can cause inaccuracies in the recorded results."
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
                        "(Hz) as defined by the sources."
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

    # pylint: disable=too-many-locals
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
                mediums = cls.intersecting_media(monitor, total_structures)
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
                            "available."
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
                medium_set = Simulation.intersecting_media(monitor, structures)
                medium = medium_set.pop() if medium_set else medium
                _, index_k = medium.nk_model(frequency=np.array(monitor.freqs))
                if not np.all(index_k == 0):
                    raise SetupError("Diffraction monitors must not lie in a lossy medium.")
        return val

    @pydantic.validator("grid_spec", always=True)
    def _warn_grid_size_too_small(cls, val, values):  # pylint:disable=too-many-locals
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
                    if isinstance(medium, PECMedium):
                        continue
                    # min wavelength in Medium2D is meaningless
                    if isinstance(medium, Medium2D):
                        continue

                    eps_material = medium.eps_model(freq0)
                    n_material, _ = medium.eps_complex_to_nk(eps_material)
                    lambda_min = C_0 / freq0 / n_material

                    for key, grid_spec in zip("xyz", (val.grid_x, val.grid_y, val.grid_z)):
                        if (
                            isinstance(grid_spec, UniformGrid)
                            and grid_spec.dl > lambda_min / MIN_GRIDS_PER_WVL
                        ):
                            consolidated_logger.warning(
                                f"The grid step in {key} has a value of {grid_spec.dl:.4f} (um)"
                                ", which was detected as being large when compared to the "
                                f"central wavelength of sources[{source_index}] "
                                f"within the simulation medium "
                                f"associated with structures[{medium_index + 1}], given by "
                                f"{lambda_min:.4f} (um). To avoid inaccuracies, "
                                "it is recommended the grid size is reduced. "
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
                mediums = cls.intersecting_media(source, total_structures)
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

        return val

    """ Post-init validators """

    def _post_init_validators(self) -> None:
        """Call validators taking z`self` that get run after init."""
        self._validate_no_structures_pml()
        self._validate_tfsf_nonuniform_grid()
        self._validate_customsourcetime()

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
                                "invariant within the PML."
                            )

    def _validate_tfsf_nonuniform_grid(self) -> None:
        """Warn if the grid is nonuniform along the directions tangential to the injection plane,
        inside the TFSF box.
        """
        # if the grid is uniform in all directions, there's no need to proceed
        if not (self.grid_spec.auto_grid_used or self.grid_spec.custom_grid_used):
            return

        with log as consolidated_logger:
            for source in self.sources:
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
                            f"axis, '{'xyz'[source.injection_axis]}'."
                        )

    def _validate_customsourcetime(self) -> None:
        """Make sure custom source time is not undersampled.
        Also, make sure that all simulation.tmesh values are covered."""
        for source in self.sources:
            if isinstance(source.source_time, CustomSourceTime):
                dataset = source.source_time.source_time_dataset
                if dataset is None:
                    continue
                times = dataset.values.coords["t"].values
                if (
                    min(times) > self.tmesh[0]
                    or max(times) < self.tmesh[-1] - CUSTOMSOURCETIME_TOL * self.dt
                ):
                    raise ValidationError(
                        "'CustomSourceTime' found with time coordinates "
                        "'times' that do not cover the entire 'Simulation.tmesh'. Currently, "
                        f"'(min(times), max(times)) = ({min(times)}, {max(times)})', while "
                        f"'(min(tmesh), max(tmesh)) = ({self.tmesh[0]}, {self.tmesh[-1]}).' "
                    )
                max_dt = np.amax(np.diff(times))
                if max_dt > self.dt * CUSTOMSOURCETIME_TOL:
                    log.warning(
                        f"'CustomSourceTime' found with time step 'max(dt) = {max_dt:.3g}', "
                        f"while the simulation time step is 'dt={self.dt}'. "
                        "We recommend that the largest time step of the custom source "
                        f"be smaller than the time step of the simulation."
                    )

    """ Pre submit validation (before web.upload()) """

    def validate_pre_upload(self, source_required: bool = True) -> None:
        """Validate the fully initialized simulation is ok for upload to our servers.

        Parameters
        ----------
        source_required: bool = True
            If ``True``, validation will fail in case no sources are found in the simulation.
        """
        self._validate_size()
        self._validate_monitor_size()
        self._validate_datasets_not_none()
        self._validate_tfsf_structure_intersections()
        # self._validate_run_time()
        _ = self.volumetric_structures
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
                "unnecessarily large, unless there are very long-lived resonances."
            )

        num_cells_times_steps = num_time_steps * num_comp_cells
        if num_cells_times_steps > MAX_CELLS_TIMES_STEPS:
            raise SetupError(
                f"Simulation has {num_cells_times_steps:.2e} grid cells * time steps, "
                f"a maximum of {MAX_CELLS_TIMES_STEPS:.2e} are allowed."
            )

    def _validate_monitor_size(self) -> None:
        """Ensures the monitors arent storing too much data before simulation is uploaded."""

        total_size_gb = 0
        with log as consolidated_logger:
            datas = self.monitors_data_size
            for monitor_name, monitor_size in datas.items():
                monitor_size_gb = monitor_size / 2**30
                if monitor_size_gb > WARN_MONITOR_DATA_SIZE_GB:
                    consolidated_logger.warning(
                        f"Monitor '{monitor_name}' estimated storage is {monitor_size_gb:1.2f}GB. "
                        "Consider making it smaller, using fewer frequencies, or spatial or "
                        "temporal downsampling using 'interval_space' and 'interval', respectively."
                    )

                total_size_gb += monitor_size_gb

        if total_size_gb > MAX_SIMULATION_DATA_SIZE_GB:
            raise SetupError(
                f"Simulation's monitors have {total_size_gb:.2f}GB of estimated storage, "
                f"a maximum of {MAX_SIMULATION_DATA_SIZE_GB:.2f}GB are allowed."
            )

    @cached_property
    def monitors_data_size(self) -> Dict[str, float]:
        """Dictionary mapping monitor names to their estimated storage size in bytes."""
        tmesh = self.tmesh
        grid = self.grid
        data_size = {}
        for monitor in self.monitors:
            name = monitor.name
            monitor_inds = grid.discretize_inds(monitor, extend=True)
            num_cells = [inds[1] - inds[0] for inds in monitor_inds]
            # take monitor downsampling into account
            if isinstance(monitor, AbstractFieldMonitor):
                num_cells = monitor.downsampled_num_cells(num_cells)
            num_cells = np.prod(num_cells)
            monitor_size = monitor.storage_size(num_cells=num_cells, tmesh=tmesh)
            data_size[name] = float(monitor_size)

        return data_size

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
                    intersecting_structs = self.intersecting_structures(
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

    @cached_property
    def mediums(self) -> Set[MediumType]:
        """Returns set of distinct :class:`.AbstractMedium` in simulation.

        Returns
        -------
        List[:class:`.AbstractMedium`]
            Set of distinct mediums in the simulation.
        """
        medium_dict = {self.medium: None}
        medium_dict.update({structure.medium: None for structure in self.structures})
        return list(medium_dict.keys())

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

        return {medium: index for index, medium in enumerate(self.mediums)}

    def get_monitor_by_name(self, name: str) -> Monitor:
        """Return monitor named 'name'."""
        for monitor in self.monitors:
            if monitor.name == name:
                return monitor
        raise Tidy3dKeyError(f"No monitor named '{name}'")

    @cached_property
    def background_structure(self) -> Structure:
        """Returns structure representing the background of the :class:`.Simulation`."""
        geometry = Box(size=(inf, inf, inf))
        return Structure(geometry=geometry, medium=self.medium)

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
        if test_object.size.count(0.0) == 1:
            # get all merged structures on the test_object, which is already planar
            structures_merged = Simulation._filter_structures_plane(structures, test_object)
            mediums = {medium for medium, _ in structures_merged}
            return mediums

        # if the test object is a volume, test each surface recursively
        surfaces = test_object.surfaces_with_exclusion(**test_object.dict())
        mediums = set()
        for surface in surfaces:
            _mediums = Simulation.intersecting_media(surface, structures)
            mediums.update(_mediums)
        return mediums

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
        if test_object.size.count(0.0) == 1:
            # get all merged structures on the test_object, which is already planar
            normal_axis_index = test_object.size.index(0.0)
            dim = "xyz"[normal_axis_index]
            pos = test_object.center[normal_axis_index]
            xyz_kwargs = {dim: pos}

            structures_merged = []
            for structure in structures:
                intersections = structure.geometry.intersections_plane(**xyz_kwargs)
                if len(intersections) > 0:
                    structures_merged.append(structure)
            return structures_merged

        # if the test object is a volume, test each surface recursively
        surfaces = test_object.surfaces_with_exclusion(**test_object.dict())
        structures_merged = []
        for surface in surfaces:
            structures_merged += Simulation.intersecting_structures(surface, structures)
        return structures_merged

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
        medium_set = self.intersecting_media(monitor, self.structures)
        if len(medium_set) > 1:
            raise SetupError(f"Monitor '{monitor.name}' intersects more than one medium.")
        medium = medium_set.pop() if medium_set else self.medium
        return medium

    @staticmethod
    def _check_bloch_vec(
        source: SourceType,
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
                    "of the source. Consider using 'BlochBoundary.from_source()'."
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
                    "double-check that it was defined correctly."
                )

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

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.
        """

        ax = self.plot_structures(ax=ax, x=x, y=y, z=z)
        ax = self.plot_sources(ax=ax, x=x, y=y, z=z, alpha=source_alpha)
        ax = self.plot_monitors(ax=ax, x=x, y=y, z=z, alpha=monitor_alpha)
        ax = self.plot_symmetries(ax=ax, x=x, y=y, z=z)
        ax = self.plot_pml(ax=ax, x=x, y=y, z=z)
        ax = self._set_plot_bounds(ax=ax, x=x, y=y, z=z)
        ax = self.plot_boundaries(ax=ax, x=x, y=y, z=z)
        return ax

    @equal_aspect
    @add_ax_if_none
    def plot_eps(  # pylint:disable=too-many-arguments
        self,
        x: float = None,
        y: float = None,
        z: float = None,
        freq: float = None,
        alpha: float = None,
        source_alpha: float = None,
        monitor_alpha: float = None,
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

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.
        """

        ax = self.plot_structures_eps(freq=freq, cbar=True, alpha=alpha, ax=ax, x=x, y=y, z=z)
        ax = self.plot_sources(ax=ax, x=x, y=y, z=z, alpha=source_alpha)
        ax = self.plot_monitors(ax=ax, x=x, y=y, z=z, alpha=monitor_alpha)
        ax = self.plot_symmetries(ax=ax, x=x, y=y, z=z)
        ax = self.plot_pml(ax=ax, x=x, y=y, z=z)
        ax = self._set_plot_bounds(ax=ax, x=x, y=y, z=z)
        ax = self.plot_boundaries(ax=ax, x=x, y=y, z=z)
        return ax

    @equal_aspect
    @add_ax_if_none
    def plot_structures(
        self, x: float = None, y: float = None, z: float = None, ax: Ax = None
    ) -> Ax:
        """Plot each of simulation's structures on a plane defined by one nonzero x,y,z coordinate.

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

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.
        """

        medium_shapes = self._get_structures_plane(structures=self.structures, x=x, y=y, z=z)
        medium_map = self.medium_map

        for (medium, shape) in medium_shapes:
            mat_index = medium_map[medium]
            ax = self._plot_shape_structure(medium=medium, mat_index=mat_index, shape=shape, ax=ax)

        ax = self._set_plot_bounds(ax=ax, x=x, y=y, z=z)

        # clean up the axis display
        axis, position = self.parse_xyz_kwargs(x=x, y=y, z=z)
        ax = self.add_ax_labels_lims(axis=axis, ax=ax)
        ax.set_title(f"cross section at {'xyz'[axis]}={position:.2f}")

        return ax

    def _plot_shape_structure(self, medium: Medium, mat_index: int, shape: Shapely, ax: Ax) -> Ax:
        """Plot a structure's cross section shape for a given medium."""
        plot_params_struct = self._get_structure_plot_params(medium=medium, mat_index=mat_index)
        ax = self.plot_shape(shape=shape, plot_params=plot_params_struct, ax=ax)
        return ax

    def _get_structure_plot_params(self, mat_index: int, medium: Medium) -> PlotParams:
        """Constructs the plot parameters for a given medium in simulation.plot()."""

        plot_params = plot_params_structure.copy(update={"linewidth": 0})

        if mat_index == 0 or medium == self.medium:
            # background medium
            plot_params = plot_params.copy(update={"facecolor": "white", "edgecolor": "white"})
        elif isinstance(medium, PECMedium):
            # perfect electrical conductor
            plot_params = plot_params.copy(
                update={"facecolor": "gold", "edgecolor": "k", "linewidth": 1}
            )
        elif isinstance(medium, Medium2D):
            # 2d material
            plot_params = plot_params.copy(update={"edgecolor": "k", "linewidth": 1})
        else:
            # regular medium
            facecolor = MEDIUM_CMAP[(mat_index - 1) % len(MEDIUM_CMAP)]
            plot_params = plot_params.copy(update={"facecolor": facecolor})

        return plot_params

    @staticmethod
    def _add_cbar(eps_min: float, eps_max: float, ax: Ax = None) -> None:
        """Add a colorbar to eps plot."""
        norm = mpl.colors.Normalize(vmin=eps_min, vmax=eps_max)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.15)
        mappable = mpl.cm.ScalarMappable(norm=norm, cmap=STRUCTURE_EPS_CMAP)
        plt.colorbar(mappable, cax=cax, label=r"$\epsilon_r$")

    @equal_aspect
    @add_ax_if_none
    def plot_structures_eps(  # pylint: disable=too-many-arguments,too-many-locals
        self,
        x: float = None,
        y: float = None,
        z: float = None,
        freq: float = None,
        alpha: float = None,
        cbar: bool = True,
        reverse: bool = False,
        ax: Ax = None,
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

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.
        """

        structures = self.structures

        # alpha is None just means plot without any transparency
        if alpha is None:
            alpha = 1

        if alpha <= 0:
            return ax

        if alpha < 1 and not isinstance(self.medium, AbstractCustomMedium):
            axis, position = Box.parse_xyz_kwargs(x=x, y=y, z=z)
            center = Box.unpop_axis(position, (0, 0), axis=axis)
            size = Box.unpop_axis(0, (inf, inf), axis=axis)
            plane = Box(center=center, size=size)
            medium_shapes = self._filter_structures_plane(structures=structures, plane=plane)
        else:
            structures = [self.background_structure] + list(structures)
            medium_shapes = self._get_structures_plane(structures=structures, x=x, y=y, z=z)

        eps_min, eps_max = self.eps_bounds(freq=freq)
        for (medium, shape) in medium_shapes:
            # if the background medium is custom medium, it needs to be rendered separately
            if medium == self.medium and alpha < 1 and not isinstance(medium, AbstractCustomMedium):
                continue
            # no need to add patches for custom medium
            if not isinstance(medium, AbstractCustomMedium):
                ax = self._plot_shape_structure_eps(
                    freq=freq,
                    alpha=alpha,
                    medium=medium,
                    eps_min=eps_min,
                    eps_max=eps_max,
                    reverse=reverse,
                    shape=shape,
                    ax=ax,
                )
            else:
                # For custom medium, apply pcolormesh clipped by the shape.
                self._pcolormesh_shape_custom_medium_structure_eps(
                    x, y, z, freq, alpha, medium, eps_min, eps_max, reverse, shape, ax
                )

        if cbar:
            self._add_cbar(eps_min=eps_min, eps_max=eps_max, ax=ax)
        ax = self._set_plot_bounds(ax=ax, x=x, y=y, z=z)

        # clean up the axis display
        axis, position = self.parse_xyz_kwargs(x=x, y=y, z=z)
        ax = self.add_ax_labels_lims(axis=axis, ax=ax)
        ax.set_title(f"cross section at {'xyz'[axis]}={position:.2f}")

        return ax

    def eps_bounds(self, freq: float = None) -> Tuple[float, float]:
        """Compute range of (real) permittivity present in the simulation at frequency "freq"."""

        medium_list = [self.medium] + list(self.mediums)
        medium_list = [medium for medium in medium_list if not isinstance(medium, PECMedium)]
        # regular medium
        eps_list = [
            medium.eps_model(freq).real
            for medium in medium_list
            if not isinstance(medium, AbstractCustomMedium)
        ]
        eps_min = min(1, min(eps_list))
        eps_max = max(1, max(eps_list))
        # custom medium, the min and max in the supplied dataset over all components and
        # spatial locations.
        for mat in [medium for medium in medium_list if isinstance(medium, AbstractCustomMedium)]:
            eps_dataarray = mat.eps_dataarray_freq(freq)
            eps_min = min(
                eps_min,
                min((np.min(eps_comp.real.values.ravel()) for eps_comp in eps_dataarray)),
            )
            eps_max = max(
                eps_max,
                max((np.max(eps_comp.real.values.ravel()) for eps_comp in eps_dataarray)),
            )
        return eps_min, eps_max

    def _pcolormesh_shape_custom_medium_structure_eps(
        self,
        x: float,
        y: float,
        z: float,
        freq: float,
        alpha: float,
        medium: Medium,
        eps_min: float,
        eps_max: float,
        reverse: bool,
        shape: Shapely,
        ax: Ax,
    ):
        """
        Plot shape made of custom medium with ``pcolormesh``.
        """
        coords = "xyz"
        normal_axis_ind, normal_position = self.parse_xyz_kwargs(x=x, y=y, z=z)
        normal_axis, plane_axes = self.pop_axis(coords, normal_axis_ind)

        # First, obtain `span_inds` of grids for interpolating permittivity in the
        # bounding box of the shape
        shape_bounds = shape.bounds
        rmin, rmax = [*shape_bounds[:2]], [*shape_bounds[2:]]
        rmin.insert(normal_axis_ind, normal_position)
        rmax.insert(normal_axis_ind, normal_position)
        span_inds = self.grid.discretize_inds(Box.from_bounds(rmin=rmin, rmax=rmax), extend=True)
        # filter negative or too large inds
        n_grid = [len(grid_comp) for grid_comp in self.grid.boundaries.to_list]
        span_inds = [
            (max(fmin, 0), min(fmax, n_grid[f_ind])) for f_ind, (fmin, fmax) in enumerate(span_inds)
        ]

        # assemble the coordinate in the 2d plane
        plane_coord = []
        for plane_axis in range(2):
            ind_axis = "xyz".index(plane_axes[plane_axis])
            plane_coord.append(self.grid.boundaries.to_list[ind_axis][slice(*span_inds[ind_axis])])

        # prepare `Coords` for interpolation
        coord_dict = {
            plane_axes[0]: plane_coord[0],
            plane_axes[1]: plane_coord[1],
            normal_axis: [normal_position],
        }
        coord_shape = Coords(**coord_dict)
        # interpolate permittivity and take the average over components
        eps_shape = np.mean(medium.eps_diagonal_on_grid(frequency=freq, coords=coord_shape), axis=0)
        # remove the normal_axis and take real part
        eps_shape = eps_shape.real.mean(axis=normal_axis_ind)
        # reverse
        if reverse:
            eps_shape = eps_min + eps_max - eps_shape

        # pcolormesh
        plane_xp, plane_yp = np.meshgrid(plane_coord[0], plane_coord[1], indexing="ij")
        ax.pcolormesh(
            plane_xp,
            plane_yp,
            eps_shape,
            clip_path=(polygon_path(shape), ax.transData),
            cmap=STRUCTURE_EPS_CMAP,
            vmin=eps_min,
            vmax=eps_max,
            alpha=alpha,
            clip_box=ax.bbox,
        )

    def _get_structure_eps_plot_params(
        self,
        medium: Medium,
        freq: float,
        eps_min: float,
        eps_max: float,
        reverse: bool = False,
        alpha: float = None,
    ) -> PlotParams:
        """Constructs the plot parameters for a given medium in simulation.plot_eps()."""

        plot_params = plot_params_structure.copy(update={"linewidth": 0})
        if alpha is not None:
            plot_params = plot_params.copy(update={"alpha": alpha})

        if isinstance(medium, PECMedium):
            # perfect electrical conductor
            plot_params = plot_params.copy(
                update={"facecolor": "gold", "edgecolor": "k", "linewidth": 1}
            )
        elif isinstance(medium, Medium2D):
            # 2d material
            plot_params = plot_params.copy(update={"edgecolor": "k", "linewidth": 1})
        else:
            # regular medium
            eps_medium = medium.eps_model(frequency=freq).real
            delta_eps = eps_medium - eps_min
            delta_eps_max = eps_max - eps_min + 1e-5
            eps_fraction = delta_eps / delta_eps_max
            color = eps_fraction if reverse else 1 - eps_fraction
            plot_params = plot_params.copy(update={"facecolor": str(color)})

        return plot_params

    def _plot_shape_structure_eps(
        self,
        freq: float,
        medium: Medium,
        shape: Shapely,
        eps_min: float,
        eps_max: float,
        ax: Ax,
        reverse: bool = False,
        alpha: float = None,
    ) -> Ax:
        """Plot a structure's cross section shape for a given medium, grayscale for permittivity."""
        plot_params = self._get_structure_eps_plot_params(
            medium=medium, freq=freq, eps_min=eps_min, eps_max=eps_max, alpha=alpha, reverse=reverse
        )
        ax = self.plot_shape(shape=shape, plot_params=plot_params, ax=ax)
        return ax

    @equal_aspect
    @add_ax_if_none
    def plot_sources(
        self, x: float = None, y: float = None, z: float = None, alpha: float = None, ax: Ax = None
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
        alpha : float = None
            Opacity of the sources, If ``None`` uses Tidy3d default.
        ax : matplotlib.axes._subplots.Axes = None
            Matplotlib axes to plot on, if not specified, one is created.

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.
        """
        bounds = self.bounds
        for source in self.sources:
            ax = source.plot(x=x, y=y, z=z, alpha=alpha, ax=ax, sim_bounds=bounds)
        ax = self._set_plot_bounds(ax=ax, x=x, y=y, z=z)
        return ax

    @equal_aspect
    @add_ax_if_none
    def plot_monitors(
        self, x: float = None, y: float = None, z: float = None, alpha: float = None, ax: Ax = None
    ) -> Ax:
        """Plot each of simulation's monitors on a plane defined by one nonzero x,y,z coordinate.

        Parameters
        ----------
        x : float = None
            position of plane in x direction, only one of x, y, z must be specified to define plane.
        y : float = None
            position of plane in y direction, only one of x, y, z must be specified to define plane.
        z : float = None
            position of plane in z direction, only one of x, y, z must be specified to define plane.
        alpha : float = None
            Opacity of the sources, If ``None`` uses Tidy3d default.
        ax : matplotlib.axes._subplots.Axes = None
            Matplotlib axes to plot on, if not specified, one is created.

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.
        """
        bounds = self.bounds
        for monitor in self.monitors:
            ax = monitor.plot(x=x, y=y, z=z, alpha=alpha, ax=ax, sim_bounds=bounds)
        ax = self._set_plot_bounds(ax=ax, x=x, y=y, z=z)
        return ax

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

    @cached_property
    def bounds_pml(self) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """Simulation bounds including the PML regions."""
        pml_thick = self.pml_thicknesses
        bounds_in = self.bounds
        bounds_min = tuple((bmin - pml[0] for bmin, pml in zip(bounds_in[0], pml_thick)))
        bounds_max = tuple((bmax + pml[1] for bmax, pml in zip(bounds_in[1], pml_thick)))

        return (bounds_min, bounds_max)

    @cached_property
    def simulation_geometry(self) -> Box:
        """The entire simulation domain including PML layers. It is identical to
        ``sim.geometry`` in the absence of PML.
        """
        rmin, rmax = self.bounds_pml
        return Box.from_bounds(rmin=rmin, rmax=rmax)

    @equal_aspect
    @add_ax_if_none
    def plot_pml(
        self,
        x: float = None,
        y: float = None,
        z: float = None,
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
            position of plane in z direction, only one of x, y, z must be specified to define plane.
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
        ax = self._set_plot_bounds(ax=ax, x=x, y=y, z=z)
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
        rmin, rmax = [list(bounds) for bounds in self.bounds_pml]
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

    @equal_aspect
    @add_ax_if_none
    def plot_symmetries(
        self, x: float = None, y: float = None, z: float = None, ax: Ax = None
    ) -> Ax:
        """Plot each of simulation's symmetries on a plane defined by one nonzero x,y,z coordinate.

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

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.
        """

        normal_axis, _ = self.parse_xyz_kwargs(x=x, y=y, z=z)

        for sym_axis, sym_value in enumerate(self.symmetry):
            if sym_value == 0 or sym_axis == normal_axis:
                continue
            sym_box = self._make_symmetry_box(sym_axis=sym_axis)
            plot_params = self._make_symmetry_plot_params(sym_value=sym_value)
            ax = sym_box.plot(x=x, y=y, z=z, ax=ax, **plot_params.to_kwargs())
        ax = self._set_plot_bounds(ax=ax, x=x, y=y, z=z)
        return ax

    def _make_symmetry_plot_params(self, sym_value: Symmetry) -> PlotParams:
        """Make PlotParams for symmetry."""

        plot_params = plot_params_symmetry.copy()

        if sym_value == 1:
            plot_params = plot_params.copy(
                update={"facecolor": "lightsteelblue", "edgecolor": "lightsteelblue", "hatch": "++"}
            )
        elif sym_value == -1:
            plot_params = plot_params.copy(
                update={"facecolor": "goldenrod", "edgecolor": "goldenrod", "hatch": "--"}
            )

        return plot_params

    def _make_symmetry_box(self, sym_axis: Axis) -> Box:
        """Construct a :class:`.Box` representing the symmetry to be plotted."""
        sym_box = self.simulation_geometry
        size = list(sym_box.size)
        size[sym_axis] /= 2
        center = list(sym_box.center)
        center[sym_axis] -= size[sym_axis] / 2

        return Box(size=size, center=center)

    @add_ax_if_none
    def plot_grid(  # pylint:disable=too-many-locals
        self,
        x: float = None,
        y: float = None,
        z: float = None,
        ax: Ax = None,
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
        _, (xmin, ymin) = self.pop_axis(self.bounds_pml[0], axis=axis)
        _, (xmax, ymax) = self.pop_axis(self.bounds_pml[1], axis=axis)
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
                zorder=np.inf,
                **plot_params.to_kwargs(),
            )
            ax.add_patch(rect)

        ax = self._set_plot_bounds(ax=ax, x=x, y=y, z=z)

        return ax

    @equal_aspect
    @add_ax_if_none
    def plot_boundaries(  # pylint:disable=too-many-locals
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
            zorder=np.inf,
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
            zorder=np.inf,
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
            zorder=np.inf,
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
            zorder=np.inf,
            **plot_params.to_kwargs(),
            **kwargs,
        )
        ax.add_patch(rect)

        # ax = self._set_plot_bounds(ax=ax, x=x, y=y, z=z)
        ax.set_xlim([ulim_minus, ulim_plus])
        ax.set_ylim([vlim_minus, vlim_plus])

        return ax

    def _set_plot_bounds(self, ax: Ax, x: float = None, y: float = None, z: float = None) -> Ax:
        """Sets the xy limits of the simulation at a plane, useful after plotting.

        Parameters
        ----------
        ax : matplotlib.axes._subplots.Axes
            Matplotlib axes to set bounds on.
        x : float = None
            position of plane in x direction, only one of x, y, z must be specified to define plane.
        y : float = None
            position of plane in y direction, only one of x, y, z must be specified to define plane.
        z : float = None
            position of plane in z direction, only one of x, y, z must be specified to define plane.

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The axes after setting the boundaries.
        """

        axis, _ = self.parse_xyz_kwargs(x=x, y=y, z=z)
        _, (xmin, ymin) = self.pop_axis(self.bounds_pml[0], axis=axis)
        _, (xmax, ymax) = self.pop_axis(self.bounds_pml[1], axis=axis)

        if xmin != xmax:
            ax.set_xlim(xmin, xmax)
        if ymin != ymax:
            ax.set_ylim(ymin, ymax)
        return ax

    @staticmethod
    def _get_structures_plane(
        structures: List[Structure], x: float = None, y: float = None, z: float = None
    ) -> List[Tuple[Medium, Shapely]]:
        """Compute list of shapes to plot on plane specified by {x,y,z}.

        Parameters
        ----------
        structures : List[:class:`.Structure`]
            list of structures to filter on the plane.
        x : float = None
            position of plane in x direction, only one of x, y, z must be specified to define plane.
        y : float = None
            position of plane in y direction, only one of x, y, z must be specified to define plane.
        z : float = None
            position of plane in z direction, only one of x, y, z must be specified to define plane.

        Returns
        -------
        List[Tuple[:class:`.AbstractMedium`, shapely.geometry.base.BaseGeometry]]
            List of shapes and mediums on the plane.
        """
        medium_shapes = []
        for structure in structures:
            intersections = structure.geometry.intersections_plane(x=x, y=y, z=z)
            if len(intersections) > 0:
                for shape in intersections:
                    shape = Box.evaluate_inf_shape(shape)
                    medium_shapes.append((structure.medium, shape))
        return medium_shapes

    @staticmethod
    def _filter_structures_plane(  # pylint:disable=too-many-locals
        structures: List[Structure], plane: Box
    ) -> List[Tuple[Medium, Shapely]]:
        """Compute list of shapes to plot on plane specified by {x,y,z}.
        Overlaps are removed or merged depending on medium.

        Parameters
        ----------
        structures : List[:class:`.Structure`]
            list of structures to filter on the plane.
        x : float = None
            position of plane in x direction, only one of x, y, z must be specified to define plane.
        y : float = None
            position of plane in y direction, only one of x, y, z must be specified to define plane.
        z : float = None
            position of plane in z direction, only one of x, y, z must be specified to define plane.

        Returns
        -------
        List[Tuple[:class:`.AbstractMedium`, shapely.geometry.base.BaseGeometry]]
            List of shapes and mediums on the plane after merging.
        """

        shapes = []
        for structure in structures:

            # get list of Shapely shapes that intersect at the plane
            shapes_plane = structure.geometry.intersections_2dbox(plane)

            # Append each of them and their medium information to the list of shapes
            for shape in shapes_plane:
                shapes.append((structure.medium, shape, shape.bounds))

        background_shapes = []
        for medium, shape, bounds in shapes:

            minx, miny, maxx, maxy = bounds

            # loop through background_shapes (note: all background are non-intersecting or merged)
            for index, (_medium, _shape, _bounds) in enumerate(background_shapes):

                _minx, _miny, _maxx, _maxy = _bounds

                # do a bounding box check to see if any intersection to do anything about
                if minx > _maxx or _minx > maxx or miny > _maxy or _miny > maxy:
                    continue

                # look more closely to see if intersected.
                if _shape.is_empty or not shape.intersects(_shape):
                    continue

                diff_shape = _shape - shape

                # different medium, remove intersection from background shape
                if medium != _medium and len(diff_shape.bounds) > 0:
                    background_shapes[index] = (_medium, diff_shape, diff_shape.bounds)

                # same medium, add diff shape to this shape and mark background shape for removal
                else:
                    shape = shape | diff_shape
                    background_shapes[index] = None

            # after doing this with all background shapes, add this shape to the background
            background_shapes.append((medium, shape, shape.bounds))

            # remove any existing background shapes that have been marked as 'None'
            background_shapes = [b for b in background_shapes if b is not None]

        # filter out any remaining None or empty shapes (shapes with area completely removed)
        return [(medium, shape) for (medium, shape, _) in background_shapes if shape]

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
        dl_sum_inv_sq = sum((1 / dl**2 for dl in dl_mins))
        dl_avg = 1 / np.sqrt(dl_sum_inv_sq)
        # material factor
        n_cfl = min(min((mat.n_cfl for mat in self.mediums)), 1)
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
                    # pylint:disable=protected-access
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
        freq_range = self.frequency_range
        if freq_range[1] > 0:
            nyquist_step = int(1 / (2 * freq_range[1]) / self.dt) - 1
            nyquist_step = max(1, nyquist_step)
        else:
            nyquist_step = 1

        return nyquist_step

    def min_sym_box(self, box: Box) -> Box:  # pylint:disable=too-many-locals
        """Compute the smallest Box restricted to the first quadrant in the presence of symmetries
        that fully covers the original Box when symmetries are applied.

        Parameters
        ----------
        box : :class:`.Box`
            Rectangular geometry.

        Returns
        -------
        new_box : :class:`.Box`
            The smallest Box such that any point in ``box`` is either in ``new_box`` or can be
            mapped from ``new_box`` using the simulation symmetries.
        """

        bounds_min, bounds_max = box.bounds
        sim_bs_min, sim_bs_max = self.bounds_pml
        bmin_new, bmax_new = [], []

        zipped = zip(self.center, self.symmetry, bounds_min, bounds_max, sim_bs_min, sim_bs_max)
        for (center, sym, bmin, bmax, sim_bmin, sim_bmax) in zipped:
            if sym == 0 or center < bmin:
                bmin_tmp, bmax_tmp = bmin, bmax
            elif bmax < center:
                bmin_tmp = 2 * center - bmax
                bmax_tmp = 2 * center - bmin
            else:
                # bmin <= center <= bmax
                bmin_tmp = center
                bmax_tmp = max(bmax, 2 * center - bmin)
            # If inf, extend well past the simulation domain but make a finite number
            if np.isinf(bmin_tmp):
                bmin_tmp = sim_bmin - np.amax(self.size)
            if np.isinf(bmax_tmp):
                bmax_tmp = sim_bmax + np.amax(self.size)
            bmin_new.append(bmin_tmp)
            bmax_new.append(bmax_tmp)

        return Box.from_bounds(bmin_new, bmax_new)

    def _discretize_grid(self, box: Box, grid: Grid, snap_zero_dim: bool = False, **kwargs) -> Grid:
        """Grid containing only cells that intersect with a :class:`Box`.

        As opposed to ``Simulation.discretize``, this function operates on a ``grid``
        which may not be the grid of the simulation.
        """
        if not self.intersects(box):
            log.error(f"Box {box} is outside simulation, cannot discretize")

        span_inds = grid.discretize_inds(box, **kwargs)
        boundary_dict = {}
        for idim, (dim, periodic) in enumerate(zip("xyz", self._periodic)):
            ind_beg, ind_end = span_inds[idim]
            # ind_end + 1 because we are selecting cell boundaries not cells
            boundary_dict[dim] = grid.extended_subspace(idim, ind_beg, ind_end + 1, periodic)

            # Overwrite with zero dimension snapped, if requested
            if snap_zero_dim:
                if self.size[idim] == 0:
                    boundary_dict[dim] = [self.center[idim], self.center[idim]]
                elif box.size[idim] == 0:
                    boundary_dict[dim] = [box.center[idim], box.center[idim]]
        return Grid(boundaries=Coords(**boundary_dict))

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

    def discretize(self, box: Box, snap_zero_dim: bool = False, **kwargs) -> Grid:
        """Grid containing only cells that intersect with a :class:`.Box`.

        Parameters
        ----------
        box : :class:`.Box`
            Rectangular geometry within simulation to discretize.
        snap_zero_dim : bool
            If ``True``, and the ``box`` has size zero along a given direction, the ``grid`` is
            defined to also have a zero-sized cell exactly centered at the ``box`` center. If
            false, the ``simulation`` grid cell containing the ``box`` center is instead used.
        kwargs :
            Extra keyword arguments passed to ``discretize_inds`` method of :class:`.Grid`.

        Returns
        -------
        :class:`Grid`
            The FDTD subgrid containing simulation points that intersect with ``box``.
        """
        return self._discretize_grid(box, self.grid, snap_zero_dim, **kwargs)

    def epsilon(
        self,
        box: Box,
        coord_key: str = "centers",
        freq: float = None,
    ) -> Dict[str, xr.DataArray]:
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
    ) -> Dict[str, xr.DataArray]:
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
                structure=self.background_structure, frequency=freq, coords=coords
            )
            shape = tuple(len(array) for array in arrays)
            eps_array = eps_background * np.ones(shape, dtype=complex)
            # replace 2d materials with volumetric equivalents
            with log as consolidated_logger:
                for structure in self.volumetric_structures:
                    # Indexing subset within the bounds of the structure
                    # pylint:disable=protected-access
                    inds = structure.geometry._inds_inside_bounds(*arrays)

                    # Get permittivity on meshgrid over the reduced coordinates
                    coords_reduced = tuple(arr[ind] for arr, ind in zip(arrays, inds))
                    if any(coords.size == 0 for coords in coords_reduced):
                        continue

                    red_coords = Coords(**dict(zip("xyz", coords_reduced)))
                    eps_structure = get_eps(structure=structure, frequency=freq, coords=red_coords)

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
        datasets_medium = [mat for mat in self.mediums if isinstance(mat, AbstractCustomMedium)]
        datasets_geometry = []

        for struct in self.structures:
            if isinstance(struct.geometry, TriangleMesh):
                datasets_geometry += struct.geometry.mesh_dataset
            elif isinstance(struct.geometry, GeometryGroup):
                for geometry in struct.geometry.geometries:
                    datasets_geometry += geometry.mesh_dataset

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

        if not any(isinstance(medium, Medium2D) for medium in self.mediums):
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
                media = self.intersecting_media(Box.from_bounds(*geom_shifted.bounds), structures)
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
            axis = structure.geometry._normal_2dmaterial  # pylint: disable=protected-access

            # snap monolayer to grid
            geometry = snap_to_grid(structure.geometry, axis)
            center = get_bounds(geometry, axis)[0]

            # get neighboring media and grid sizes
            (neighbors, dls) = get_neighboring_media(geometry, axis, background_structures)

            new_bounds = (center - dls[0] / 2, center + dls[1] / 2)
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

        for medium in self.mediums:
            if isinstance(medium, AnisotropicMedium):
                if np.any([med.allow_gain for med in [medium.xx, medium.yy, medium.zz]]):
                    return True
            elif medium.allow_gain:
                return True
        return False
