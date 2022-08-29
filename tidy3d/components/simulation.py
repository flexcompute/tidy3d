# pylint: disable=too-many-lines, too-many-arguments
""" Container holding all information about simulation and its components"""
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
from .geometry import Box
from .types import Ax, Shapely, FreqBound, GridSize, Axis, annotate_type
from .grid import Coords1D, Grid, Coords, GridSpec, UniformGrid
from .medium import Medium, MediumType, AbstractMedium, PECMedium
from .boundary import BoundarySpec, Symmetry, BlochBoundary, PECBoundary, PMCBoundary
from .boundary import PML, StablePML, Absorber
from .structure import Structure
from .source import SourceType, PlaneWave, GaussianBeam, AstigmaticGaussianBeam
from .monitor import MonitorType, Monitor, FreqMonitor
from .monitor import AbstractFieldMonitor, AbstractNear2FarMonitor
from .viz import add_ax_if_none, equal_aspect

from .viz import MEDIUM_CMAP, PlotParams, plot_params_symmetry
from .viz import plot_params_structure, plot_params_pml, plot_params_override_structures
from .viz import plot_params_pec, plot_params_pmc, plot_params_bloch

from ..version import __version__
from ..constants import C_0, MICROMETER, SECOND, inf
from ..log import log, Tidy3dKeyError, SetupError, ValidationError
from ..updater import Updater

# minimum number of grid points allowed per central wavelength in a medium
MIN_GRIDS_PER_WVL = 6.0

# maximum number of mediums supported
MAX_NUM_MEDIUMS = 65530

# maximum numbers of simulation parameters
MAX_TIME_STEPS = 1e8
MAX_GRID_CELLS = 20e9
MAX_CELLS_TIMES_STEPS = 1e17
MAX_MONITOR_DATA_SIZE_BYTES = 10e9


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
    ...     size=(2.0, 2.0, 2.0),
    ...     grid_spec=GridSpec(
    ...         grid_x = AutoGrid(min_steps_per_wvl = 20),
    ...         grid_y = AutoGrid(min_steps_per_wvl = 20),
    ...         grid_z = AutoGrid(min_steps_per_wvl = 20)
    ...     ),
    ...     run_time=40e-11,
    ...     structures=[
    ...         Structure(
    ...             geometry=Box(size=(1, 1, 1), center=(-1, 0, 0)),
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

    grid_size: Union[GridSpec, Tuple[GridSize, GridSize, GridSize]] = pydantic.Field(
        None,
        title="Grid Size",
        description="NOTE: 'grid_size' has been replaced by 'grid_spec'.",
        units=MICROMETER,
    )

    medium: MediumType = pydantic.Field(
        Medium(),
        title="Background Medium",
        description="Background medium of simulation, defaults to vacuum if not specified.",
    )

    symmetry: Tuple[Symmetry, Symmetry, Symmetry] = pydantic.Field(
        (0, 0, 0),
        title="Symmetries",
        description="Tuple of integers defining reflection symmetry across a plane "
        "bisecting the simulation domain normal to the x-, y-, and z-axis "
        "at the simulation center of each axis, respectvely. "
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
        description="Specification of boundary conditions along each dimension.",
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
        0.9,
        title="Courant Factor",
        description="Courant stability factor, controls time step to spatial step ratio. "
        "Lower values lead to more stable simulations for dispersive materials, "
        "but result in longer simulation times.",
        gt=0.0,
        le=1.0,
    )

    version: str = pydantic.Field(
        __version__,
        title="Version",
        description="String specifying the front end version number.",
    )

    """ Validating setup """

    @pydantic.validator("grid_size", always=True)
    def _error_use_grid_size(cls, val):
        """If ``grid_size`` is provided, raise an error."""

        if val is not None:
            raise ValidationError(
                "'grid_size' has been replaced by 'grid_spec'. See the "
                "'GridSpec' documentation for more information."
            )

        return val

    @pydantic.validator("grid_spec", always=True)
    def _validate_auto_grid_wavelength(cls, val, values):
        """Check that wavelength can be defined if there is auto grid spec."""
        if val.wavelength is None and val.auto_grid_used:
            _ = val.wavelength_from_sources(sources=values.get("sources"))
        return val

    _structures_in_bounds = assert_objects_in_sim_bounds("structures")
    _sources_in_bounds = assert_objects_in_sim_bounds("sources")
    _monitors_in_bounds = assert_objects_in_sim_bounds("monitors")
    _mode_sources_symmetries = validate_mode_objects_symmetry("sources")
    _mode_monitors_symmetries = validate_mode_objects_symmetry("monitors")

    # assign names to unnamed structures, sources, and mediums
    # _structure_names = set_names("structures")
    # _source_names = set_names("sources")

    # make sure all names are unique
    _unique_structure_names = assert_unique_names("structures")
    _unique_source_names = assert_unique_names("sources")
    _unique_monitor_names = assert_unique_names("monitors")

    # _unique_medium_names = assert_unique_names("structures", check_mediums=True)

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

        for istruct, structure in enumerate(val):
            struct_bound_min, struct_bound_max = structure.geometry.bounds
            struct_bounds = list(struct_bound_min) + list(struct_bound_max)

            for sim_val, struct_val in zip(sim_bounds, struct_bounds):

                if isclose(sim_val, struct_val):
                    log.warning(
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

        def warn(istruct, side):
            """Warning message for a structure too close to PML."""
            log.warning(
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

        for monitor_index, monitor in enumerate(val):
            if not isinstance(monitor, FreqMonitor):
                continue

            freqs = np.array(monitor.freqs)
            for medium_index, medium in enumerate(mediums):

                # skip mediums that have no freq range (all freqs valid)
                if medium.frequency_range is None:
                    continue

                # make sure medium frequency range includes all monitor frequencies
                fmin_med, fmax_med = medium.frequency_range
                if np.any(freqs < fmin_med) or np.any(freqs > fmax_med):
                    if medium_index == 0:
                        medium_str = "The simulation background medium"
                    else:
                        medium_str = f"The medium associated with structures[{medium_index-1}]"

                    log.warning(
                        f"{medium_str}has a frequency range: ({fmin_med:2e}, {fmax_med:2e}) "
                        "(Hz)that does not fully cover the frequencies contained in "
                        f"monitors[{monitor_index}]. "
                        "This can cause innacuracies in the recorded results."
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
            log.warning("No sources in simulation.")
            return val

        freq_min = min((freq_range[0] for freq_range in source_ranges), default=0.0)
        freq_max = max((freq_range[1] for freq_range in source_ranges), default=0.0)

        for monitor_index, monitor in enumerate(val):
            if not isinstance(monitor, FreqMonitor):
                continue

            freqs = np.array(monitor.freqs)
            if np.any(freqs < freq_min) or np.any(freqs > freq_max):
                log.warning(
                    f"monitors[{monitor_index}] contains frequencies "
                    f"outside of the simulation frequency range ({freq_min:2e}, {freq_max:2e})"
                    "(Hz) as defined by the sources."
                )
        return val

    @pydantic.validator("monitors", always=True)
    def n2f_monitors_with_symmetry(cls, val, values):
        """Error if a server-side near-to-far monitor is used with symmetry"""
        monitors = val
        symmetry = values.get("symmetry")
        if not any(symmetry):
            return val
        for monitor in monitors:
            if isinstance(monitor, AbstractNear2FarMonitor):
                raise SetupError(
                    "Server-side near-to-far calculation is not available with symmetry. "
                    "Use the local near-to-far available through the 'RadiationVectors' class."
                )
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

        for source_index, source in enumerate(values.get("sources")):
            freq0 = source.source_time.freq0

            for medium_index, medium in enumerate(mediums):

                # min wavelength in PEC is meaningless and we'll get divide by inf errors
                if isinstance(medium, PECMedium):
                    continue

                eps_material = medium.eps_model(freq0)
                n_material, _ = medium.eps_complex_to_nk(eps_material)
                lambda_min = C_0 / freq0 / n_material

                for key, grid_spec in zip("xyz", (val.grid_x, val.grid_y, val.grid_z)):
                    if (
                        isinstance(grid_spec, UniformGrid)
                        and grid_spec.dl > lambda_min / MIN_GRIDS_PER_WVL
                    ):
                        log.warning(
                            f"The grid step in {key} has a value of {grid_spec.dl:.4f} (um)"
                            ", which was detected as being large when compared to the "
                            f"central wavelength of sources[{source_index}] "
                            f"within the simulation medium "
                            f"associated with structures[{medium_index + 1}], given by "
                            f"{lambda_min:.4f} (um). "
                            "To avoid inaccuracies, it is reccomended the grid size is reduced."
                        )
                        # TODO: warn about custom grid spec

        return val

    @pydantic.validator("sources", always=True)
    def _source_homogeneous(cls, val, values):
        """Error if a plane wave or gaussian beam source is not in a homogeneous region."""

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

                # get all merged structures on the plane
                normal_axis_index = source.size.index(0.0)
                dim = "xyz"[normal_axis_index]
                pos = source.center[normal_axis_index]
                xyz_kwargs = {dim: pos}
                structures_merged = cls._filter_structures_plane(total_structures, **xyz_kwargs)

                # make sure there is no more than one medium in the returned list
                mediums = {medium for medium, _ in structures_merged}
                if len(mediums) > 1:

                    raise SetupError(
                        f"{len(mediums)} different mediums detected on plane "
                        f"intersecting a {source.type} source. Plane must be homogeneous."
                    )

        return val

    @pydantic.validator("normalize_index", always=True)
    def _check_normalize_index(cls, val, values):
        """Check validity of normalize index in context of simulation.sources."""

        # not normalizing
        if val is None:
            return val

        assert val >= 0, "normalize_index can't be negative."
        num_sources = len(values.get("sources"))
        if num_sources > 0:
            # No check if no sources, but it should be irrelevant anyway
            assert val < num_sources, f"{num_sources} sources smaller than normalize_index of {val}"

        return val

    """ Pre submit validation (before web.upload()) """

    def validate_pre_upload(self) -> None:
        """Validate the fully initialized simulation is ok for upload to our servers."""
        self._validate_size()
        self._validate_monitor_size()
        # self._validate_run_time()

    def _validate_size(self) -> None:
        """Ensures the simulation is within size limits before simulation is uploaded."""

        num_cells = self.num_cells
        if num_cells > MAX_GRID_CELLS:
            raise SetupError(
                f"Simulation has {num_cells:.2e} computational cells, "
                f"a maximum of {MAX_GRID_CELLS:.2e} are allowed."
            )

        num_time_steps = self.num_time_steps
        if num_time_steps > MAX_TIME_STEPS:
            raise SetupError(
                f"Simulation has {num_time_steps:.2e} time steps, "
                f"a maximum of {MAX_TIME_STEPS:.2e} are allowed."
            )

        num_cells_times_steps = num_time_steps * num_cells
        if num_cells_times_steps > MAX_CELLS_TIMES_STEPS:
            raise SetupError(
                f"Simulation has {num_cells_times_steps:.2e} grid cells * time steps, "
                f"a maximum of {MAX_CELLS_TIMES_STEPS:.2e} are allowed."
            )

    def _validate_monitor_size(self) -> None:
        """Ensures the monitors arent storing too much data before simulation is uploaded."""

        tmesh = self.tmesh
        grid = self.grid

        total_size_bytes = 0
        for monitor in self.monitors:
            monitor_inds = grid.discretize_inds(monitor, extend=True)
            num_cells = [inds[1] - inds[0] for inds in monitor_inds]
            # take monitor downsampling into account
            if isinstance(monitor, AbstractFieldMonitor):
                num_cells = monitor.downsampled_num_cells(num_cells)
            num_cells = np.prod(num_cells)
            monitor_size = monitor.storage_size(num_cells=num_cells, tmesh=tmesh)

            total_size_bytes += monitor_size

        if total_size_bytes > MAX_MONITOR_DATA_SIZE_BYTES:
            raise SetupError(
                f"Simulation's monitors have {total_size_bytes:.2e} bytes of estimated storage, "
                f"a maximum of {MAX_MONITOR_DATA_SIZE_BYTES:.2e} are allowed."
            )

    def _validate_run_time(self) -> None:
        """Ensures that the simulation run time is > 0."""

        if self.run_time <= 0:
            raise SetupError(
                "The `Simulation.run_time` parameter was left at its default value of 0.0. "
                "For running a simulation on our servers it must be set to > 0.0."
            )

    """ Accounting """

    @cached_property
    def mediums(self) -> Set[MediumType]:
        """Returns set of distinct :class:`AbstractMedium` in simulation.

        Returns
        -------
        List[:class:`AbstractMedium`]
            Set of distinct mediums in the simulation.
        """
        medium_dict = {self.medium: None}
        medium_dict.update({structure.medium: None for structure in self.structures})
        return list(medium_dict.keys())

    @cached_property
    def medium_map(self) -> Dict[MediumType, pydantic.NonNegativeInt]:
        """Returns dict mapping medium to index in material.
        ``medium_map[medium]`` returns unique global index of :class:`AbstractMedium` in simulation.

        Returns
        -------
        Dict[:class:`AbstractMedium`, int]
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
        """Returns structure representing the background of the :class:`Simulation`."""
        geometry = Box(size=(inf, inf, inf))
        return Structure(geometry=geometry, medium=self.medium)

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

        # TODO: if we want structure alpha, we will have to filter, otherwise just get overlapped
        # medium_shapes = self._filter_structures_plane(self.structures, x=x, y=y, z=z)
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
        mappable = mpl.cm.ScalarMappable(norm=norm, cmap="gist_yarg")
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

        if alpha < 1:
            medium_shapes = self._filter_structures_plane(structures=structures, x=x, y=y, z=z)
        else:
            structures = [self.background_structure] + list(structures)
            medium_shapes = self._get_structures_plane(structures=structures, x=x, y=y, z=z)

        eps_min, eps_max = self.eps_bounds(freq=freq)
        for (medium, shape) in medium_shapes:
            if medium == self.medium and alpha < 1:
                continue
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
        eps_list = [medium.eps_model(freq).real for medium in medium_list]
        eps_min = min(1, min(eps_list))
        eps_max = max(1, max(eps_list))
        return eps_min, eps_max

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
        """Construct a :class:`Box` representing an arborbing boundary to be plotted."""
        rmin, rmax = [list(bounds) for bounds in self.bounds_pml]
        if sign == -1:
            rmax[pml_axis] = rmin[pml_axis] + pml_height
        else:
            rmin[pml_axis] = rmax[pml_axis] - pml_height
        return Box.from_bounds(rmin=rmin, rmax=rmax)

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
        """Construct a :class:`Box` representing the symmetry to be plotted."""
        rmin, rmax = self.bounds_pml
        sym_box = Box.from_bounds(rmin=rmin, rmax=rmax)
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

        kwargs.setdefault("clip_on", False)

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

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        return ax

    @staticmethod
    def _get_structures_plane(
        structures: List[Structure], x: float = None, y: float = None, z: float = None
    ) -> List[Tuple[Medium, Shapely]]:
        """Compute list of shapes to plot on plane specified by {x,y,z}.

        Parameters
        ----------
        structures : List[:class:`Structure`]
            list of structures to filter on the plane.
        x : float = None
            position of plane in x direction, only one of x, y, z must be specified to define plane.
        y : float = None
            position of plane in y direction, only one of x, y, z must be specified to define plane.
        z : float = None
            position of plane in z direction, only one of x, y, z must be specified to define plane.

        Returns
        -------
        List[Tuple[:class:`AbstractMedium`, shapely.geometry.base.BaseGeometry]]
            List of shapes and mediums on the plane.
        """
        medium_shapes = []
        for structure in structures:
            intersections = structure.geometry.intersections(x=x, y=y, z=z)
            if len(intersections) > 0:
                for shape in intersections:
                    shape = Box.evaluate_inf_shape(shape)
                    medium_shapes.append((structure.medium, shape))
        return medium_shapes

    @staticmethod
    def _filter_structures_plane(  # pylint:disable=too-many-locals
        structures: List[Structure], x: float = None, y: float = None, z: float = None
    ) -> List[Tuple[Medium, Shapely]]:
        """Compute list of shapes to plot on plane specified by {x,y,z}.
        Overlaps are removed or merged depending on medium.

        Parameters
        ----------
        structures : List[:class:`Structure`]
            list of structures to filter on the plane.
        x : float = None
            position of plane in x direction, only one of x, y, z must be specified to define plane.
        y : float = None
            position of plane in y direction, only one of x, y, z must be specified to define plane.
        z : float = None
            position of plane in z direction, only one of x, y, z must be specified to define plane.

        Returns
        -------
        List[Tuple[:class:`AbstractMedium`, shapely.geometry.base.BaseGeometry]]
            List of shapes and mediums on the plane after merging.
        """

        shapes = []
        for structure in structures:

            # dont bother with geometries that dont intersect plane
            if not structure.geometry.intersects_plane(x=x, y=y, z=z):
                continue

            # get list of Shapely shapes that intersect at the plane
            shapes_plane = structure.geometry.intersections(x=x, y=y, z=z)

            # Append each of them and their medium information to the list of shapes
            for shape in shapes_plane:
                shape = Box.evaluate_inf_shape(shape)
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
                if not shape.intersects(_shape):
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
        return self.courant * dl_avg / C_0

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

    @cached_property
    def grid(self) -> Grid:
        """FDTD grid spatial locations and information.

        Returns
        -------
        :class:`Grid`
            :class:`Grid` storing the spatial locations relevant to the simulation.
        """

        # Add a simulation Box as the first structure
        structures = [Structure(geometry=self.geometry, medium=self.medium)]
        structures += self.structures

        return self.grid_spec.make_grid(
            structures=structures,
            symmetry=self.symmetry,
            sources=self.sources,
            num_pml_layers=self.num_pml_layers,
        )

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
        wvl_min = C_0 / min(freq_max)
        eps_max = max(abs(structure.medium.get_eps(freq_max)) for structure in self.structures)
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
        box : :class:`Box`
            Rectangular geometry.

        Returns
        -------
        new_box : :class:`Box`
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
            # Extend well past the simulation domain if needed, but truncate if original box
            # is too large, specifically to avoid issues with inf.
            sim_size = sim_bmax - sim_bmin
            bmin_new.append(max(bmin_tmp, sim_bmin - sim_size))
            bmax_new.append(min(bmax_tmp, sim_bmax + sim_size))

        return Box.from_bounds(bmin_new, bmax_new)

    def discretize(self, box: Box, extend: bool = False, snap_zero_dim: bool = False) -> Grid:
        """Grid containing only cells that intersect with a :class:`Box`.

        Parameters
        ----------
        box : :class:`Box`
            Rectangular geometry within simulation to discretize.
        extend : bool
            If ``True``, extra pixels are added to the discretized grid if needed, such that Yee
            grid fields can be interpolated to any point inside the ``box``.
        snap_zero_dim : bool
            If ``True``, and the ``box`` has size zero along a given direction, the ``grid`` is
            defined to also have a zero-sized cell exactly centered at the ``box`` center. If
            false, the ``simulation`` grid cell containing the ``box`` center is instead used.

        Returns
        -------
        :class:`Grid`
            The FDTD subgrid containing simulation points that intersect with ``box``.
        """

        if not self.intersects(box):
            log.error(f"Box {box} is outside simulation, cannot discretize")

        span_inds = self.grid.discretize_inds(box, extend=extend)
        boundary_dict = {}
        for idim, dim in enumerate("xyz"):
            if snap_zero_dim and box.size[idim] == 0:
                boundary_dict[dim] = [box.center[idim], box.center[idim]]
            else:
                ind_beg, ind_end = span_inds[idim]
                # ind_end + 1 because we are selecting cell boundaries not cells
                boundary_dict[dim] = self.grid.periodic_subspace(idim, ind_beg, ind_end + 1)

        return Grid(boundaries=Coords(**boundary_dict))

    def epsilon(
        self,
        box: Box,
        coord_key: str = "centers",
        freq: float = None,
    ) -> Dict[str, xr.DataArray]:
        """Get array of permittivity at volume specified by box and freq.

        Parameters
        ----------
        box : :class:`Box`
            Rectangular geometry specifying where to measure the permittivity.
        coord_key : str = 'centers'
            Specifies at what part of the grid to return the permittivity at.
            Accepted values are ``{'centers', 'boundaries', 'Ex', 'Ey', 'Ez'}``.
            The field values (eg. 'Ex') correspond to the correponding field locations on the yee
            lattice. If field values are selected, the corresponding epsilon component from the
            main diagonal of the epsilon tensor is returned. Otherwise, the average of the diagonal
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
        grid : :class:`Grid`
            Grid specifying where to measure the permittivity.
        coord_key : str = 'centers'
            Specifies at what part of the grid to return the permittivity at.
            Accepted values are ``{'centers', 'boundaries', 'Ex', 'Ey', 'Ez'}``.
            The field values (eg. 'Ex') correspond to the correponding field locations on the yee
            lattice. If field values are selected, the corresponding epsilon component from the
            main diagonal of the epsilon tensor is returned. Otherwise, the average of the diagonal
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

        def get_eps(medium: Medium, freq: float):
            """Select the correct epsilon component if field locations are requested."""
            if coord_key[0] != "E":
                return medium.eps_model(freq)
            component = ["x", "y", "z"].index(coord_key[1])
            return medium.eps_diagonal(freq)[component]

        eps_background = get_eps(self.medium, freq)

        def make_eps_data(coords: Coords):
            """returns epsilon data on grid of points defined by coords"""
            xs, ys, zs = coords.x, coords.y, coords.z
            rmin = tuple(coord[0] for coord in (xs, ys, zs))
            rmax = tuple(coord[-1] for coord in (xs, ys, zs))
            points_box = Box.from_bounds(rmin=rmin, rmax=rmax)
            x, y, z = np.meshgrid(xs, ys, zs, indexing="ij")
            eps_array = eps_background * np.ones(x.shape, dtype=complex)
            for structure in self.structures:
                if not points_box.intersects(structure.geometry):
                    continue
                eps_structure = get_eps(structure.medium, freq)
                is_inside = structure.geometry.inside(x, y, z)
                eps_array[np.where(is_inside)] = eps_structure
            coords = {"x": np.array(xs), "y": np.array(ys), "z": np.array(zs)}
            return xr.DataArray(eps_array, coords=coords, dims=("x", "y", "z"))

        # combine all data into dictionary
        coords = grid[coord_key]
        return make_eps_data(coords)

    @classmethod
    def from_file(cls, fname: str, **parse_kwargs):
        """Loads a :class:`Tidy3dBaseModel` from .yaml or .json file.

        Parameters
        ----------
        fname : str
            Full path to the .yaml or .json file to load the :class:`Tidy3dBaseModel` from.
        **parse_kwargs
            Keyword arguments passed to either pydantic's ``parse_file`` or ``parse_raw`` methods
            for ``.json`` and ``.yaml`` file formats, respectively.
        Returns
        -------
        :class:`Tidy3dBaseModel`
            An instance of the component class calling `load`.

        Example
        -------
        >>> simulation = Simulation.from_file(fname='folder/sim.json') # doctest: +SKIP
        """

        updater = Updater.from_file(fname)
        sim_dict = updater.update_to_current()
        return Simulation.parse_obj(sim_dict, **parse_kwargs)
