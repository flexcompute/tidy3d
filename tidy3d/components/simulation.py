# pylint: disable=too-many-lines
""" Container holding all information about simulation and its components"""
from typing import Dict, Tuple, List, Set

import pydantic
import numpy as np
import xarray as xr
import matplotlib.pylab as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from descartes import PolygonPatch

from .validators import assert_unique_names, assert_objects_in_sim_bounds
from .validators import validate_mode_objects_symmetry
from .geometry import Box
from .types import Symmetry, Ax, Shapely, FreqBound, GridSize
from .grid import Coords1D, Grid, Coords
from .medium import Medium, MediumType, AbstractMedium, PECMedium
from .structure import Structure
from .source import SourceType, PlaneWave
from .monitor import MonitorType, Monitor, FreqMonitor
from .pml import PMLTypes, PML, Absorber
from .viz import StructMediumParams, StructEpsParams, PMLParams, SymParams
from .viz import add_ax_if_none, equal_aspect
from .viz import plotly_sim

from ..version import __version__
from ..constants import C_0, MICROMETER, SECOND, pec_val, inf
from ..log import log, Tidy3dKeyError, SetupError

# for docstring examples
# from .geometry import Sphere, Cylinder, PolySlab  # pylint:disable=unused-import
# from .source import VolumeSource, GaussianPulse  # pylint:disable=unused-import
# from .monitor import FieldMonitor, FluxMonitor, Monitor, FreqMonitor  # pylint:disable=unused-import


# minimum number of grid points allowed per central wavelength in a medium
MIN_GRIDS_PER_WVL = 6.0

# maximum number of mediums supported
MAX_NUM_MEDIUMS = 200

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
    >>> from tidy3d import VolumeSource, GaussianPulse
    >>> from tidy3d import FieldMonitor, FluxMonitor
    >>> sim = Simulation(
    ...     size=(2.0, 2.0, 2.0),
    ...     grid_size=(0.1, 0.1, 0.1),
    ...     run_time=40e-11,
    ...     structures=[
    ...         Structure(
    ...             geometry=Box(size=(1, 1, 1), center=(-1, 0, 0)),
    ...             medium=Medium(permittivity=2.0),
    ...         ),
    ...     ],
    ...     sources=[
    ...         VolumeSource(
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
    ...     pml_layers=(
    ...         PML(num_layers=20),
    ...         PML(num_layers=30),
    ...         None,
    ...     ),
    ...     shutoff=1e-6,
    ...     courant=0.8,
    ...     subpixel=False,
    ... )
    """

    grid_size: Tuple[GridSize, GridSize, GridSize] = pydantic.Field(
        ...,
        title="Grid Size",
        description="If components are float, uniform grid size along x, y, and z. "
        "If components are array like, defines an array of nonuniform grid sizes centered at "
        "the simulation center ."
        " Note: if supplied sizes do not cover the simulation size, the first and last sizes "
        "are repeated to cover size. ",
        units=MICROMETER,
    )

    run_time: pydantic.PositiveFloat = pydantic.Field(
        ...,
        title="Run Time",
        description="Total electromagnetic evolution time in seconds. "
        "Note: If simulation 'shutoff' is specified, "
        "simulation will terminate early when shutoff condition met. ",
        units=SECOND,
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
        "bisecting the simulation domain normal to the x-, y-, and z-axis, respectvely. "
        "Each element can be ``0`` (no symmetry), ``1`` (even, i.e. 'PMC' symmetry) or "
        "``-1`` (odd, i.e. 'PEC' symmetry). "
        "Note that the vectorial nature of the fields must be taken into account to correctly "
        "determine the symmetry value.",
    )

    structures: List[Structure] = pydantic.Field(
        [],
        title="Structures",
        description="List of structures present in simulation. "
        "Note: Structures defined later in this list override the "
        "simulation material properties in regions of spatial overlap.",
    )

    sources: List[SourceType] = pydantic.Field(
        [],
        title="Sources",
        description="List of electric current sources injecting fields into the simulation.",
    )

    monitors: List[MonitorType] = pydantic.Field(
        [],
        title="Monitors",
        description="List of monitors in the simulation. "
        "Note: monitor names are used to access data after simulation is run.",
    )

    pml_layers: Tuple[PMLTypes, PMLTypes, PMLTypes] = pydantic.Field(
        (None, None, None),
        title="Absorbing Layers",
        description="Specifications for the absorbing layers on x, y, and z edges. "
        "If ``None``, no absorber will be added on that dimension "
        "and periodic boundary conditions will be used.",
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

    @pydantic.validator("pml_layers", always=True, allow_reuse=True)
    def set_none_to_zero_layers(cls, val):
        """if any PML layer is None, set it to an empty :class:`PML`."""
        return tuple(PML(num_layers=0) if pml is None else pml for pml in val)

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

                if np.isclose(sim_val, struct_val):
                    log.warning(
                        f"Structure at structures[{istruct}] has bounds that extend exactly to "
                        "simulation edges. This can cause unexpected behavior. "
                        "If intending to extend the structure to infinity along one dimension, "
                        "use td.inf as a size variable instead to make this explicit."
                    )

        return val

    @pydantic.validator("pml_layers", always=True)
    def _structures_not_close_pml(cls, val, values):  # pylint:disable=too-many-locals
        """Warn if any structures lie at the simulation boundaries."""

        if val is None:
            return val

        sim_box = Box(size=values.get("size"), center=values.get("center"))
        sim_bound_min, sim_bound_max = sim_box.bounds

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
                fmin_src, fmax_src = source.source_time.frequency_range()
                f_average = (fmin_src + fmax_src) / 2.0
                lambda0 = C_0 / f_average

                zipped = zip(["x", "y", "z"], sim_bound_min, struct_bound_min, val)
                for axis, sim_val, struct_val, pml in zipped:
                    if (
                        pml.num_layers > 0
                        and struct_val > sim_val
                        and not isinstance(pml, Absorber)
                    ):
                        if abs(sim_val - struct_val) < lambda0 / 2:
                            warn(istruct, axis + "-min")

                zipped = zip(["x", "y", "z"], sim_bound_max, struct_bound_max, val)
                for axis, sim_val, struct_val, pml in zipped:
                    if (
                        pml.num_layers > 0
                        and struct_val < sim_val
                        and not isinstance(pml, Absorber)
                    ):
                        if abs(sim_val - struct_val) < lambda0 / 2:
                            warn(istruct, axis + "-max")

        return val

    @pydantic.validator("monitors", always=True)
    def _warn_monitor_mediums_frequency_range(cls, val, values):
        """Warn user if any DFT monitors have frequencies outside of medium frequency range."""

        if val is None:
            return val

        structures = values.get("structures")
        structures = [] if not structures else structures
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
                        medium_str + f"has a frequency range: ({fmin_med:2e}, {fmax_med:2e}) (Hz)"
                        "that does not fully cover the frequencies contained in "
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
        source_ranges = [source.source_time.frequency_range() for source in values["sources"]]
        if len(source_ranges) == 0:
            log.warning("No sources in simulation.")
            return val

        freq_min = min([freq_range[0] for freq_range in source_ranges], default=0.0)
        freq_max = max([freq_range[1] for freq_range in source_ranges], default=0.0)

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

    @pydantic.validator("sources", always=True)
    def _warn_grid_size_too_small(cls, val, values):  # pylint:disable=too-many-locals
        """Warn user if any grid size is too large compared to minimum wavelength in material."""

        if val is None:
            return val

        structures = values.get("structures")
        structures = [] if not structures else structures
        medium_bg = values.get("medium")
        grid_size = values.get("grid_size")
        mediums = [medium_bg] + [structure.medium for structure in structures]

        for source_index, source in enumerate(val):
            fmin_src, fmax_src = source.source_time.frequency_range()
            f_average = (fmin_src + fmax_src) / 2.0

            for medium_index, medium in enumerate(mediums):

                # min wavelength in PEC is meaningless and we'll get divide by inf errors
                if isinstance(medium, PECMedium):
                    continue

                eps_material = medium.eps_model(f_average)
                n_material, _ = medium.eps_complex_to_nk(eps_material)
                lambda_min = C_0 / f_average / n_material

                for grid_index, dl in enumerate(grid_size):
                    if isinstance(dl, float):
                        if dl > lambda_min / MIN_GRIDS_PER_WVL:
                            log.warning(
                                f"The grid step in {'xyz'[grid_index]} has a value of {dl:.4f} (um)"
                                ", which was detected as being large when compared to the "
                                f"central wavelength of sources[{source_index}] "
                                f"within the simulation medium "
                                f"associated with structures[{medium_index + 1}], given by "
                                f"{lambda_min:.4f} (um). "
                                "To avoid inaccuracies, it is reccomended the grid size is reduced."
                            )
                    # TODO: warn about nonuniform grid

        return val

    @pydantic.validator("sources", always=True)
    def _plane_wave_homogeneous(cls, val, values):
        """Error if plane wave intersects"""

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

        structures = values.get("structures")
        structures = [] if not structures else structures
        total_structures = [structure_bg] + structures

        # for each plane wave in the sources list
        for source in val:
            if isinstance(source, PlaneWave):

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
                        "intersecting a plane wave source.  Plane must be homogeneous."
                    )

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

        total_size_bytes = 0
        for monitor in self.monitors:
            monitor_grid = self.discretize(monitor)
            num_cells = np.prod(monitor_grid.num_cells)
            monitor_size = monitor.storage_size(num_cells=num_cells, tmesh=tmesh)

            total_size_bytes += monitor_size

        if total_size_bytes > MAX_MONITOR_DATA_SIZE_BYTES:
            raise SetupError(
                f"Simulation's monitors have {total_size_bytes:.2e} bytes of estimated storage, "
                f"a maximum of {MAX_MONITOR_DATA_SIZE_BYTES:.2e} are allowed."
            )

    def _validate_run_time(self) -> None:
        """Ensures that the simulation run time is > 0."""

        if not self.run_time > 0:
            raise SetupError(
                "The `Simulation.run_time` parameter was left at its default value of 0.0. "
                "For running a simulation on our servers it must be set to > 0.0."
            )

    """ Accounting """

    @property
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

    @property
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

    """ Plotting """

    @equal_aspect
    @add_ax_if_none
    def plot(  # pylint:disable=too-many-arguments
        self,
        x: float = None,
        y: float = None,
        z: float = None,
        ax: Ax = None,
        **kwargs,
    ) -> Ax:

        ax = self.plot_structures(ax=ax, x=x, y=y, z=z, **kwargs)
        ax = self.plot_sources(ax=ax, x=x, y=y, z=z, **kwargs)
        ax = self.plot_monitors(ax=ax, x=x, y=y, z=z, **kwargs)
        ax = self.plot_symmetries(ax=ax, x=x, y=y, z=z, **kwargs)
        ax = self.plot_pml(ax=ax, x=x, y=y, z=z, **kwargs)
        ax = self._set_plot_bounds(ax=ax, x=x, y=y, z=z)
        return ax

    @equal_aspect
    @add_ax_if_none
    def plot_eps(  # pylint: disable=too-many-arguments
        self,
        x: float = None,
        y: float = None,
        z: float = None,
        freq: float = None,
        ax: Ax = None,
        **kwargs,
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
        ax : matplotlib.axes._subplots.Axes = None
            Matplotlib axes to plot on, if not specified, one is created.
        **kwargs
            Optional keyword arguments passed to the matplotlib patch plotting of structure.
            For details on accepted values, refer to
            `Matplotlib's documentation <https://tinyurl.com/2nf5c2fk>`_.

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.
        """

        ax = self.plot_structures_eps(freq=freq, cbar=True, ax=ax, x=x, y=y, z=z, **kwargs)
        ax = self.plot_sources(ax=ax, x=x, y=y, z=z, **kwargs)
        ax = self.plot_monitors(ax=ax, x=x, y=y, z=z, **kwargs)
        ax = self.plot_symmetries(ax=ax, x=x, y=y, z=z, **kwargs)
        ax = self.plot_pml(ax=ax, x=x, y=y, z=z, **kwargs)
        ax = self._set_plot_bounds(ax=ax, x=x, y=y, z=z)
        return ax

    @equal_aspect
    @add_ax_if_none
    def plot_structures(
        self, x: float = None, y: float = None, z: float = None, ax: Ax = None, **kwargs
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
        **kwargs
            Optional keyword arguments passed to the matplotlib patch plotting of structure.
            For details on accepted values, refer to
            `Matplotlib's documentation <https://tinyurl.com/2nf5c2fk>`_.

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.
        """

        medium_map = self.medium_map
        medium_shapes = self._filter_structures_plane(self.structures, x=x, y=y, z=z)
        for (medium, shape) in medium_shapes:
            params_updater = StructMediumParams(medium=medium, medium_map=medium_map)
            kwargs_struct = params_updater.update_params(**kwargs)
            if medium == self.medium:
                continue
            patch = PolygonPatch(shape, **kwargs_struct)
            ax.add_artist(patch)
        ax = self._set_plot_bounds(ax=ax, x=x, y=y, z=z)

        # clean up the axis display
        axis, position = self.parse_xyz_kwargs(x=x, y=y, z=z)
        ax = self.add_ax_labels_lims(axis=axis, ax=ax)
        ax.set_title(f"cross section at {'xyz'[axis]}={position:.2f}")

        return ax

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
        cbar: bool = True,
        reverse: bool = True,
        ax: Ax = None,
        **kwargs,
    ) -> Ax:
        # pylint:disable=line-too-long
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
        reverse : bool = True
            If ``True``, the highest permittivity is plotted in black; if ``False``: white.
        ax : matplotlib.axes._subplots.Axes = None
            Matplotlib axes to plot on, if not specified, one is created.
        **kwargs
            Optional keyword arguments passed to the matplotlib patch plotting of structure.
            For details on accepted values, refer to
            `Matplotlib's documentation <https://tinyurl.com/2nf5c2fk>`_.

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.
        """
        # pylint:enable=line-too-long

        if freq is None:
            freq = inf
        eps_list = [s.medium.eps_model(freq).real for s in self.structures]
        eps_list.append(self.medium.eps_model(freq).real)
        eps_list = [eps for eps in eps_list if eps != pec_val]
        eps_max = max(eps_list)
        eps_min = min(eps_list)
        medium_shapes = self._filter_structures_plane(self.structures, x=x, y=y, z=z)
        for (medium, shape) in medium_shapes:
            eps = medium.eps_model(freq).real
            params_updater = StructEpsParams(
                eps=eps, eps_min=eps_min, eps_max=eps_max, reverse=reverse
            )
            kwargs_struct = params_updater.update_params(**kwargs)
            if medium == self.medium:
                continue
            patch = PolygonPatch(shape, **kwargs_struct)
            ax.add_artist(patch)
        if cbar:
            self._add_cbar(eps_min=eps_min, eps_max=eps_max, ax=ax)
        ax = self._set_plot_bounds(ax=ax, x=x, y=y, z=z)

        # clean up the axis display
        axis, position = self.parse_xyz_kwargs(x=x, y=y, z=z)
        ax = self.add_ax_labels_lims(axis=axis, ax=ax)
        ax.set_title(f"cross section at {'xyz'[axis]}={position:.2f}")

        return ax

    @equal_aspect
    @add_ax_if_none
    def plot_sources(
        self, x: float = None, y: float = None, z: float = None, ax: Ax = None, **kwargs
    ) -> Ax:
        # pylint:disable=line-too-long
        """Plot each of simulation's sources on a plane defined by one nonzero x,y,z coordinate.

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
            Optional keyword arguments passed to the matplotlib patch plotting of structure.
            For details on accepted values, refer to
            `Matplotlib's documentation <https://tinyurl.com/2nf5c2fk>`_.

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.
        """
        # pylint:enable=line-too-long
        for source in self.sources:
            if source.intersects_plane(x=x, y=y, z=z):
                ax = source.plot(ax=ax, x=x, y=y, z=z, **kwargs)
        ax = self._set_plot_bounds(ax=ax, x=x, y=y, z=z)
        return ax

    @equal_aspect
    @add_ax_if_none
    def plot_monitors(
        self, x: float = None, y: float = None, z: float = None, ax: Ax = None, **kwargs
    ) -> Ax:
        # pylint:disable=line-too-long
        """Plot each of simulation's monitors on a plane defined by one nonzero x,y,z coordinate.

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
            Optional keyword arguments passed to the matplotlib patch plotting of structure.
            For details on accepted values, refer to
            `Matplotlib's documentation <https://tinyurl.com/2nf5c2fk>`_.

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.
        """
        # pylint:enable=line-too-long
        for monitor in self.monitors:
            if monitor.intersects_plane(x=x, y=y, z=z):
                ax = monitor.plot(ax=ax, x=x, y=y, z=z, **kwargs)
        ax = self._set_plot_bounds(ax=ax, x=x, y=y, z=z)
        return ax

    @equal_aspect
    @add_ax_if_none
    def plot_symmetries(
        self, x: float = None, y: float = None, z: float = None, ax: Ax = None, **kwargs
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
        **kwargs
            Optional keyword arguments passed to the matplotlib patch plotting of structure.
            For details on accepted values, refer to
            `Matplotlib's documentation <https://tinyurl.com/2nf5c2fk>`_.

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.
        """

        normal_axis, _ = self.parse_xyz_kwargs(x=x, y=y, z=z)
        for sym_axis, sym_value in enumerate(self.symmetry):
            if sym_value == 0 or sym_axis == normal_axis:
                continue
            sym_size = [1000 * size_dim for size_dim in self.size]
            sym_size[sym_axis] /= 2
            sym_center = list(self.center)
            sym_center[sym_axis] -= sym_size[sym_axis] / 2
            sym_box = Box(center=sym_center, size=sym_size)
            if sym_box.intersects_plane(x=x, y=y, z=z):
                new_kwargs = SymParams(sym_value=sym_value).update_params(**kwargs)
                ax = sym_box.plot(ax=ax, x=x, y=y, z=z, **new_kwargs)
        ax = self._set_plot_bounds(ax=ax, x=x, y=y, z=z)
        return ax

    @property
    def num_pml_layers(self) -> List[Tuple[float, float]]:
        """Number of absorbing layers in all three axes and directions (-, +).

        Returns
        -------
        List[Tuple[float, float]]
            List containing the number of absorber layers in - and + boundaries.
        """

        return [(pml.num_layers, pml.num_layers) for pml in self.pml_layers]

    @property
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

    @property
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
        self, x: float = None, y: float = None, z: float = None, ax: Ax = None, **kwargs
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
        **kwargs
            Optional keyword arguments passed to the matplotlib patch plotting of structure.
            For details on accepted values, refer to
            `Matplotlib's documentation <https://tinyurl.com/2nf5c2fk>`_.

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.
        """
        kwargs = PMLParams().update_params(**kwargs)
        pml_thicks = self.pml_thicknesses
        for pml_axis, pml_layer in enumerate(self.pml_layers):
            if pml_layer.num_layers == 0:
                continue
            for sign, pml_height in zip((-1, 1), pml_thicks[pml_axis]):
                pml_size = [inf, inf, inf]
                pml_size[pml_axis] = pml_height
                pml_center = list(self.center)
                pml_offset_center = (self.size[pml_axis] + pml_height) / 2.0
                pml_center[pml_axis] += sign * pml_offset_center
                pml_box = Box(center=pml_center, size=pml_size)
                if pml_box.intersects_plane(x=x, y=y, z=z):
                    ax = pml_box.plot(ax=ax, x=x, y=y, z=z, **kwargs)
        ax = self._set_plot_bounds(ax=ax, x=x, y=y, z=z)
        return ax

    @add_ax_if_none
    def plot_grid(self, x: float = None, y: float = None, z: float = None, ax: Ax = None) -> Ax:
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

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.
        """
        cell_boundaries = self.grid.boundaries
        axis, _ = self.parse_xyz_kwargs(x=x, y=y, z=z)
        _, (axis_x, axis_y) = self.pop_axis([0, 1, 2], axis=axis)
        boundaries_x = cell_boundaries.dict()["xyz"[axis_x]]
        boundaries_y = cell_boundaries.dict()["xyz"[axis_y]]
        for x_pos in boundaries_x:
            ax.axvline(x=x_pos, linestyle="-", color="black", linewidth=0.2)
        for y_pos in boundaries_y:
            ax.axhline(y=y_pos, linestyle="-", color="black", linewidth=0.2)
        ax = self._set_plot_bounds(ax=ax, x=x, y=y, z=z)
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
    def _filter_structures_plane(
        structures: List[Structure], x: float = None, y: float = None, z: float = None
    ) -> List[Tuple[Medium, Shapely]]:
        """Compute list of shapes to plot on plane specified by {x,y,z}.
        Overlaps are removed or merged depending on medium.

        Parameters
        ----------
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
                shapes.append((structure.medium, shape))

        background_shapes = []
        for medium, shape in shapes:

            shape = Box.evaluate_inf_shape(shape)

            # loop through background_shapes (note: all background are non-intersecting or merged)
            for index, (_medium, _shape) in enumerate(background_shapes):

                # if not intersection, move onto next background shape
                if not shape & _shape:
                    continue

                # different medium, remove intersection from background shape
                if medium != _medium:
                    background_shapes[index] = (_medium, _shape - shape)

                # same medium, add background to this shape and mark background shape for removal
                else:
                    shape = shape | (_shape - shape)
                    background_shapes[index] = None

            # after doing this with all background shapes, add this shape to the background
            background_shapes.append((medium, shape))

            # remove any existing background shapes that have been marked as 'None'
            background_shapes = [b for b in background_shapes if b is not None]

        # filter out any remaining None or empty shapes (shapes with area completely removed)
        return [(medium, shape) for (medium, shape) in background_shapes if shape]

    def plotly(
        self, x: float = None, y: float = None, z: float = None
    ) -> "plotly.graph_objects.Figure":
        """Plot the geometry cross section at single (x,y,z) coordinate using plotly.

        Parameters
        ----------
        x : float = None
            Position of plane in x direction, only one of x,y,z can be specified to define plane.
        y : float = None
            Position of plane in y direction, only one of x,y,z can be specified to define plane.
        z : float = None
            Position of plane in z direction, only one of x,y,z can be specified to define plane.

        Returns
        -------
        plotly.graph_objects.Figure
            A plotly figure.
        """
        return plotly_sim(self, x=x, y=y, z=z)

    @property
    def frequency_range(self) -> FreqBound:
        """Range of frequencies spanning all sources' frequency dependence.

        Returns
        -------
        Tuple[float, float]
            Minumum and maximum frequencies of the power spectrum of the sources.
        """
        source_ranges = [source.source_time.frequency_range() for source in self.sources]
        freq_min = min([freq_range[0] for freq_range in source_ranges], default=0.0)
        freq_max = max([freq_range[1] for freq_range in source_ranges], default=0.0)
        return (freq_min, freq_max)

    """ Discretization """

    @property
    def dt(self) -> float:
        """Simulation time step (distance).

        Returns
        -------
        float
            Time step (seconds).
        """
        dl_mins = [np.min(sizes) for sizes in self.grid.sizes.to_list]
        dl_sum_inv_sq = sum([1 / dl**2 for dl in dl_mins])
        dl_avg = 1 / np.sqrt(dl_sum_inv_sq)
        return self.courant * dl_avg / C_0

    @property
    def tmesh(self) -> Coords1D:
        """FDTD time stepping points.

        Returns
        -------
        np.ndarray
            Times (seconds) that the simulation time steps through.
        """
        dt = self.dt
        return np.arange(0.0, self.run_time + dt, dt)

    @property
    def num_time_steps(self) -> int:
        """Number of time steps in simulation."""

        return len(self.tmesh)

    @staticmethod
    def _make_bound_coords_uniform(dl, center, size):
        """creates coordinate boundaries with uniform mesh (dl is float)"""

        num_cells = int(np.floor(size / dl))

        # Make sure there's at least one cell
        num_cells = max(num_cells, 1)

        # snap to grid, recenter
        size_snapped = dl * num_cells
        bound_coords = center + np.linspace(-size_snapped / 2, size_snapped / 2, num_cells + 1)

        return bound_coords

    @staticmethod
    def _make_bound_coords_nonuniform(dl, center, size):
        """creates coordinate boundaries with non-uniform mesh (dl is arraylike)"""

        # get bounding coordinates
        dl = np.array(dl)
        bound_coords = np.array([np.sum(dl[:i]) for i in range(len(dl) + 1)])

        # place the middle boundary at the center of the simulation along dimension
        bound_coords += center - bound_coords[bound_coords.size // 2]

        # chop off any coords outside of simulation bounds
        bound_min = center - size / 2
        bound_max = center + size / 2
        bound_coords = bound_coords[bound_coords <= bound_max]
        bound_coords = bound_coords[bound_coords >= bound_min]

        # if not extending to simulation bounds, repeat beginning and end
        dl_min = dl[0]
        dl_max = dl[-1]
        while bound_coords[0] - dl_min >= bound_min:
            bound_coords = np.insert(bound_coords, 0, bound_coords[0] - dl_min)
        while bound_coords[-1] + dl_max <= bound_max:
            bound_coords = np.append(bound_coords, bound_coords[-1] + dl_max)

        return bound_coords

    def _make_bound_coords(self, dim):
        """Creates coordinate boundaries along dimension ``dim`` and handle PML and symmetries"""

        dl = self.grid_size[dim]
        center = self.center[dim]

        # Make uniform or nonuniform boundaries depending on dl input
        if isinstance(dl, float):
            bound_coords = self._make_bound_coords_uniform(dl, center, self.size[dim])
        else:
            bound_coords = self._make_bound_coords_nonuniform(dl, center, self.size[dim])

        # Add PML layers in using dl on edges
        bound_coords = self._add_pml_to_bounds(self.num_pml_layers[dim], bound_coords)

        # Enforce a symmetric grid by placing a boundary at the simulation center and
        # reflecting the boundaries on the other side.
        if self.symmetry[dim] != 0:
            bound_coords += center - bound_coords[bound_coords.size // 2]
            bound_coords = bound_coords[bound_coords >= center]
            bound_coords = np.append(2 * center - bound_coords[:0:-1], bound_coords)

        return bound_coords

    @property
    def grid(self) -> Grid:
        """FDTD grid spatial locations and information.

        Returns
        -------
        :class:`Grid`
            :class:`Grid` storing the spatial locations relevant to the simulation.
        """
        cell_boundary_dict = {}
        for dim, key in enumerate("xyz"):
            cell_boundary_dict[key] = self._make_bound_coords(dim)
        boundaries = Coords(**cell_boundary_dict)
        return Grid(boundaries=boundaries)

    @property
    def num_cells(self) -> int:
        """Number of cells in the simulation.

        Returns
        -------
        int
            Number of yee cells in the simulation.
        """

        return np.prod(self.grid.num_cells, dtype=np.int64)

    @staticmethod
    def _add_pml_to_bounds(num_layers: Tuple[int, int], bounds: Coords1D):
        """Append absorber layers to the beginning and end of the simulation bounds
        along one dimension.

        Parameters
        ----------
        num_layers : Tuple[int, int]
            number of layers in the absorber + and - direction along one dimension.
        bound_coords : np.ndarray
            coordinates specifying boundaries between cells along one dimension.

        Returns
        -------
        np.ndarray
            New bound coordinates along dimension taking abosrber into account.
        """
        if bounds.size < 2:
            return bounds

        first_step = bounds[1] - bounds[0]
        last_step = bounds[-1] - bounds[-2]
        add_left = bounds[0] - first_step * np.arange(num_layers[0], 0, -1)
        add_right = bounds[-1] + last_step * np.arange(1, num_layers[1] + 1)
        new_bounds = np.concatenate((add_left, bounds, add_right))

        return new_bounds

    @property
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
            else:
                if bmax < center:
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

    def discretize(self, box: Box) -> Grid:
        """Grid containing only cells that intersect with a :class:`Box`.

        Parameters
        ----------
        box : :class:`Box`
            Rectangular geometry within simulation to discretize.

        Returns
        -------
        :class:`Grid`
            The FDTD subgrid containing simulation points that intersect with ``box``.
        """

        if not self.intersects(box):
            log.error(f"Box {box} is outside simulation, cannot discretize")

        disc_inds = self.grid.discretize_inds(box)
        sub_cell_boundary_dict = {}
        for axis_label, axis_inds in zip("xyz", disc_inds):
            # copy orginal bound coords into subgrid coords
            bound_coords = self.grid.boundaries.dict()[axis_label]
            # axis_inds[1] + 1 because we are selecting cell boundaries not cells
            sub_cell_boundary_dict[axis_label] = bound_coords[axis_inds[0] : axis_inds[1] + 1]

        # construct sub grid
        sub_boundaries = Coords(**sub_cell_boundary_dict)
        return Grid(boundaries=sub_boundaries)

    def epsilon(
        self, box: Box, coord_key: str = "centers", freq: float = None
    ) -> Dict[str, xr.DataArray]:
        """Get array of permittivity at volume specified by box and freq

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

        def get_eps(medium: Medium, freq: float):
            """Select the correct epsilon component if field locations are requested."""
            if coord_key[0] == "E":
                component = ["x", "y", "z"].index(coord_key[1])
                eps = medium.eps_diagonal(freq)[component]
            else:
                eps = medium.eps_model(freq)
            return eps

        eps_background = get_eps(self.medium, freq)

        def make_eps_data(coords: Coords):
            """returns epsilon data on grid of points defined by coords"""
            xs, ys, zs = coords.x, coords.y, coords.z
            x, y, z = np.meshgrid(xs, ys, zs, indexing="ij")
            eps_array = eps_background * np.ones(x.shape, dtype=complex)
            for structure in self.structures:
                eps_structure = get_eps(structure.medium, freq)
                is_inside = structure.geometry.inside(x, y, z)
                eps_array[np.where(is_inside)] = eps_structure
            return xr.DataArray(eps_array, coords={"x": xs, "y": ys, "z": zs}, dims=("x", "y", "z"))

        # combine all data into dictionary
        coords = sub_grid[coord_key]
        return make_eps_data(coords)
