""" Defines classes specifying meshing in 1D and a collective class for 3D """
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple, List, Union

import numpy as np
import pydantic as pd

from .grid import Coords1D, Coords, Grid
from .mesher import GradedMesher, MesherType
from ..base import Tidy3dBaseModel
from ..types import Axis, Symmetry, annotate_type
from ..source import SourceType
from ..structure import Structure, StructureType
from ..geometry import Box
from ...log import log
from ...exceptions import SetupError
from ...constants import MICROMETER, C_0, fp_eps


class GridSpec1d(Tidy3dBaseModel, ABC):

    """Abstract base class, defines 1D grid generation specifications."""

    def make_coords(  # pylint:disable = too-many-arguments
        self,
        axis: Axis,
        structures: List[StructureType],
        symmetry: Tuple[Symmetry, Symmetry, Symmetry],
        periodic: bool,
        wavelength: pd.PositiveFloat,
        num_pml_layers: Tuple[pd.NonNegativeInt, pd.NonNegativeInt],
    ) -> Coords1D:
        """Generate 1D coords to be used as grid boundaries, based on simulation parameters.
        Symmetry, and PML layers will be treated here.

        Parameters
        ----------
        axis : Axis
            Axis of this direction.
        structures : List[StructureType]
            List of structures present in simulation, the first one being the simulation domain.
        symmetry : Tuple[Symmetry, Symmetry, Symmetry]
            Reflection symmetry across a plane bisecting the simulation domain
            normal to each of the three axes.
        wavelength : float
            Free-space wavelength.
        num_pml_layers : Tuple[int, int]
            number of layers in the absorber + and - direction along one dimension.

        Returns
        -------
        :class:`.Coords1D`:
            1D coords to be used as grid boundaries.
        """

        # Determine if one should apply periodic boundary condition.
        # This should only affect auto nonuniform mesh generation for now.
        is_periodic = periodic and symmetry[axis] == 0

        # generate boundaries
        bound_coords = self._make_coords_initial(
            axis=axis,
            structures=structures,
            wavelength=wavelength,
            symmetry=symmetry,
            is_periodic=is_periodic,
        )

        # incooperate symmetries
        if symmetry[axis] != 0:
            # Offset to center if symmetry present
            center = structures[0].geometry.center[axis]
            center_ind = np.argmin(np.abs(center - bound_coords))
            bound_coords += center - bound_coords[center_ind]
            bound_coords = bound_coords[bound_coords >= center]
            bound_coords = np.append(2 * center - bound_coords[:0:-1], bound_coords)

        # Add PML layers in using dl on edges
        bound_coords = self._add_pml_to_bounds(num_pml_layers, bound_coords)
        return bound_coords

    @abstractmethod
    def _make_coords_initial(
        self,
        axis: Axis,
        structures: List[StructureType],
        **kwargs,
    ) -> Coords1D:
        """Generate 1D coords to be used as grid boundaries, based on simulation parameters.
        Symmetry, PML etc. are not considered in this method.

        For auto nonuniform generation, it will take some more arguments.

        Parameters
        ----------
        structures : List[StructureType]
            List of structures present in simulation, the first one being the simulation domain.
        **kwargs
            Other arguments

        Returns
        -------
        :class:`.Coords1D`:
            1D coords to be used as grid boundaries.
        """

    @staticmethod
    def _add_pml_to_bounds(num_layers: Tuple[int, int], bounds: Coords1D) -> Coords1D:
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
        return np.concatenate((add_left, bounds, add_right))


class UniformGrid(GridSpec1d):

    """Uniform 1D grid.

    Example
    -------
    >>> grid_1d = UniformGrid(dl=0.1)
    """

    dl: pd.PositiveFloat = pd.Field(
        ...,
        title="Grid Size",
        description="Grid size for uniform grid generation.",
        units=MICROMETER,
    )

    def _make_coords_initial(
        self,
        axis: Axis,
        structures: List[StructureType],
        **kwargs,
    ) -> Coords1D:
        """Uniform 1D coords to be used as grid boundaries.

        Parameters
        ----------
        axis : Axis
            Axis of this direction.
        structures : List[StructureType]
            List of structures present in simulation, the first one being the simulation domain.
        **kwargs:
            Other arguments all go here.

        Returns
        -------
        :class:`.Coords1D`:
            1D coords to be used as grid boundaries.
        """

        center, size = structures[0].geometry.center[axis], structures[0].geometry.size[axis]

        # Take a number of steps commensurate with the size; make dl a bit smaller if needed
        num_cells = int(np.ceil(size / self.dl))

        # Make sure there's at least one cell
        num_cells = max(num_cells, 1)

        # Adjust step size to fit simulation size exactly
        dl_snapped = size / num_cells if size > 0 else self.dl

        return center - size / 2 + np.arange(num_cells + 1) * dl_snapped


class CustomGrid(GridSpec1d):

    """Custom 1D grid supplied as a list of grid cell sizes centered on the simulation center.

    Example
    -------
    >>> grid_1d = CustomGrid(dl=[0.2, 0.2, 0.1, 0.1, 0.1, 0.2, 0.2])
    """

    dl: Tuple[pd.PositiveFloat, ...] = pd.Field(
        ...,
        title="Customized grid sizes.",
        description="An array of custom nonuniform grid sizes. The resulting grid is centered on "
        "the simulation center such that it spans the region "
        "``(center - sum(dl)/2, center + sum(dl)/2)``, unless a ``custom_offset`` is given. "
        "Note: if supplied sizes do not cover the simulation size, the first and last sizes "
        "are repeated to cover the simulation domain.",
        units=MICROMETER,
    )

    custom_offset: float = pd.Field(
        None,
        title="Customized grid offset.",
        description="The starting coordinate of the grid which defines the simulation center. "
        "If ``None``, the simulation center is set such that it spans the region "
        "``(center - sum(dl)/2, center + sum(dl)/2)``.",
        units=MICROMETER,
    )

    def _make_coords_initial(
        self,
        axis: Axis,
        structures: List[StructureType],
        **kwargs,
    ) -> Coords1D:
        """Customized 1D coords to be used as grid boundaries.

        Parameters
        ----------
        axis : Axis
            Axis of this direction.
        structures : List[StructureType]
            List of structures present in simulation, the first one being the simulation domain.
        *kwargs
            Other arguments all go here.

        Returns
        -------
        :class:`.Coords1D`:
            1D coords to be used as grid boundaries.
        """

        center, size = structures[0].geometry.center[axis], structures[0].geometry.size[axis]

        # get bounding coordinates
        dl = np.array(self.dl)
        bound_coords = np.append(0.0, np.cumsum(dl))

        # place the middle of the bounds at the center of the simulation along dimension,
        # or use the `custom_offset` if provided
        if self.custom_offset is None:
            bound_coords += center - bound_coords[-1] / 2
        else:
            bound_coords += self.custom_offset

        # chop off any coords outside of simulation bounds, beyond some buffer region
        # to take numerical effects into account
        buffer = fp_eps * size
        bound_min = center - size / 2 - buffer
        bound_max = center + size / 2 + buffer
        bound_coords = bound_coords[bound_coords <= bound_max]
        bound_coords = bound_coords[bound_coords >= bound_min]

        # if not extending to simulation bounds, repeat beginning and end
        dl_min = dl[0]
        dl_max = dl[-1]
        while bound_coords[0] - dl_min >= bound_min:
            bound_coords = np.insert(bound_coords, 0, bound_coords[0] - dl_min)
        while bound_coords[-1] + dl_max <= bound_max:
            bound_coords = np.append(bound_coords, bound_coords[-1] + dl_max)

        # in case a `custom_offset` is provided, it's possible the bounds were numerically within
        # the simulation bounds but were still chopped off, which is fixed here
        if self.custom_offset is not None:
            if np.isclose(bound_coords[0] - dl_min, bound_min):
                bound_coords = np.insert(bound_coords, 0, bound_coords[0] - dl_min)
            if np.isclose(bound_coords[-1] + dl_max, bound_max):
                bound_coords = np.append(bound_coords, bound_coords[-1] + dl_max)

        return bound_coords


class AutoGrid(GridSpec1d):
    """Specification for non-uniform grid along a given dimension.

    Example
    -------
    >>> grid_1d = AutoGrid(min_steps_per_wvl=16, max_scale=1.4)
    """

    min_steps_per_wvl: float = pd.Field(
        10.0,
        title="Minimal number of steps per wavelength",
        description="Minimal number of steps per wavelength in each medium.",
        ge=6.0,
    )

    max_scale: float = pd.Field(
        1.4,
        title="Maximum Grid Size Scaling",
        description="Sets the maximum ratio between any two consecutive grid steps.",
        ge=1.2,
        lt=2.0,
    )

    dl_min: pd.NonNegativeFloat = pd.Field(
        0,
        title="Lower bound of grid size",
        description="Lower bound of the grid size along this dimension regardless of "
        "structures present in the simulation, including override structures "
        "with ``enforced=True``. It is a soft bound, meaning that the actual minimal "
        "grid size might be slightly smaller.",
    )

    mesher: MesherType = pd.Field(
        GradedMesher(),
        title="Grid Construction Tool",
        description="The type of mesher to use to generate the grid automatically.",
    )

    def _make_coords_initial(  # pylint:disable=too-many-arguments,arguments-differ,too-many-locals
        self,
        axis: Axis,
        structures: List[StructureType],
        wavelength: float,
        symmetry: Symmetry,
        is_periodic: bool,
    ) -> Coords1D:
        """Customized 1D coords to be used as grid boundaries.

        Parameters
        ----------
        axis : Axis
            Axis of this direction.
        structures : List[StructureType]
            List of structures present in simulation.
        wavelength : float
            Free-space wavelength.
        symmetry : Tuple[Symmetry, Symmetry, Symmetry]
            Reflection symmetry across a plane bisecting the simulation domain
            normal to each of the three axes.
        is_periodic : bool
            Apply periodic boundary condition or not.

        Returns
        -------
        :class:`.Coords1D`:
            1D coords to be used as grid boundaries.
        """

        sim_cent = list(structures[0].geometry.center)
        sim_size = list(structures[0].geometry.size)
        for dim, sym in enumerate(symmetry):
            if sym != 0:
                sim_cent[dim] += sim_size[dim] / 4
                sim_size[dim] /= 2
        symmetry_domain = Box(center=sim_cent, size=sim_size)

        # New list of structures with symmetry applied
        struct_list = [Structure(geometry=symmetry_domain, medium=structures[0].medium)]
        for structure in structures[1:]:
            if symmetry_domain.intersects(structure.geometry):
                struct_list.append(structure)

        # parse structures
        interval_coords, max_dl_list = self.mesher.parse_structures(
            axis,
            struct_list,
            wavelength,
            self.min_steps_per_wvl,
            self.dl_min,
        )
        # Put just a single pixel if 2D-like simulation
        if interval_coords.size == 1:
            dl = wavelength / self.min_steps_per_wvl
            return np.array([sim_cent[axis] - dl / 2, sim_cent[axis] + dl / 2])

        # generate mesh steps
        interval_coords = np.array(interval_coords).flatten()
        max_dl_list = np.array(max_dl_list).flatten()
        len_interval_list = interval_coords[1:] - interval_coords[:-1]
        dl_list = self.mesher.make_grid_multiple_intervals(
            max_dl_list, len_interval_list, self.max_scale, is_periodic
        )

        # generate boundaries
        bound_coords = np.append(0.0, np.cumsum(np.concatenate(dl_list)))
        bound_coords += interval_coords[0]

        # fix simulation domain boundaries which may be slightly off
        domain_bounds = [bound[axis] for bound in symmetry_domain.bounds]
        if not np.all(np.isclose(bound_coords[[0, -1]], domain_bounds)):
            raise SetupError(
                f"AutoGrid coordinates along axis {axis} do not match the simulation "
                "domain, indicating an unexpected error in the meshing. Please create a github "
                "issue so that the problem can be investigated. In the meantime, switch to "
                f"uniform or custom grid along {axis}."
            )
        bound_coords[[0, -1]] = domain_bounds

        return np.array(bound_coords)


GridType = Union[UniformGrid, CustomGrid, AutoGrid]


class GridSpec(Tidy3dBaseModel):

    """Collective grid specification for all three dimensions.

    Example
    -------
    >>> uniform = UniformGrid(dl=0.1)
    >>> custom = CustomGrid(dl=[0.2, 0.2, 0.1, 0.1, 0.1, 0.2, 0.2])
    >>> auto = AutoGrid(min_steps_per_wvl=12)
    >>> grid_spec = GridSpec(grid_x=uniform, grid_y=custom, grid_z=auto, wavelength=1.5)
    """

    grid_x: GridType = pd.Field(
        AutoGrid(),
        title="Grid specification along x-axis",
        description="Grid specification along x-axis",
    )

    grid_y: GridType = pd.Field(
        AutoGrid(),
        title="Grid specification along y-axis",
        description="Grid specification along y-axis",
    )

    grid_z: GridType = pd.Field(
        AutoGrid(),
        title="Grid specification along z-axis",
        description="Grid specification along z-axis",
    )

    wavelength: float = pd.Field(
        None,
        title="Free-space wavelength",
        description="Free-space wavelength for automatic nonuniform grid. It can be 'None' "
        "if there is at least one source in the simulation, in which case it is defined by "
        "the source central frequency. "
        "Note: it only takes effect when at least one of the three dimensions "
        "uses :class:`.AutoGrid`.",
        units=MICROMETER,
    )

    override_structures: Tuple[annotate_type(StructureType), ...] = pd.Field(
        (),
        title="Grid specification override structures",
        description="A set of structures that is added on top of the simulation structures in "
        "the process of generating the grid. This can be used to refine the grid or make it "
        "coarser depending than the expected need for higher/lower resolution regions. "
        "Note: it only takes effect when at least one of the three dimensions "
        "uses :class:`.AutoGrid`.",
    )

    @property
    def auto_grid_used(self) -> bool:
        """True if any of the three dimensions uses :class:`.AutoGrid`."""
        grid_list = [self.grid_x, self.grid_y, self.grid_z]
        return np.any([isinstance(mesh, AutoGrid) for mesh in grid_list])

    @property
    def custom_grid_used(self) -> bool:
        """True if any of the three dimensions uses :class:`.CustomGrid`."""
        grid_list = [self.grid_x, self.grid_y, self.grid_z]
        return np.any([isinstance(mesh, CustomGrid) for mesh in grid_list])

    @staticmethod
    def wavelength_from_sources(sources: List[SourceType]) -> pd.PositiveFloat:
        """Define a wavelength based on supplied sources. Called if auto mesh is used and
        ``self.wavelength is None``."""

        # no sources
        if len(sources) == 0:
            raise SetupError(
                "Automatic grid generation requires the input of 'wavelength' or sources."
            )

        # Use central frequency of sources, if any.
        freqs = np.array([source.source_time.freq0 for source in sources])

        # multiple sources of different central frequencies
        if not np.all(np.isclose(freqs, freqs[0])):
            raise SetupError(
                "Sources of different central frequencies are supplied. "
                "Please supply a 'wavelength' value for 'grid_spec'."
            )

        return C_0 / freqs[0]

    @property
    def override_structures_used(self) -> List[bool, bool, bool]:
        """Along each axis, ``True`` if any override structure is used. However,
        it is still ``False`` if only :class:`.MeshOverrideStructure` is supplied, and
        their ``dl[axis]`` all take the ``None`` value.
        """

        # empty override_structure list
        if len(self.override_structures) == 0:
            return [False] * 3

        override_used = [False] * 3
        for structure in self.override_structures:
            # override used in all axes if any `Structure` is present
            if isinstance(structure, Structure):
                return [True] * 3

            for dl_axis, dl in enumerate(structure.dl):
                if (not override_used[dl_axis]) and (dl is not None):
                    override_used[dl_axis] = True
        return override_used

    # pylint:disable=too-many-locals,too-many-arguments
    def make_grid(
        self,
        structures: List[Structure],
        symmetry: Tuple[Symmetry, Symmetry, Symmetry],
        periodic: Tuple[bool, bool, bool],
        sources: List[SourceType],
        num_pml_layers: List[Tuple[pd.NonNegativeInt, pd.NonNegativeInt]],
    ) -> Grid:
        """Make the entire simulation grid based on some simulation parameters.

        Parameters
        ----------
        structures : List[Structure]
            List of structures present in the simulation. The first structure must be the
            simulation geometry with the simulation background medium.
        symmetry : Tuple[Symmetry, Symmetry, Symmetry]
            Reflection symmetry across a plane bisecting the simulation domain
            normal to each of the three axes.
        sources : List[SourceType]
            List of sources.
        num_pml_layers : List[Tuple[float, float]]
            List containing the number of absorber layers in - and + boundaries.

        Returns
        -------
        Grid:
            Entire simulation grid.
        """

        # Set up wavelength for automatic mesh generation if needed.
        wavelength = self.wavelength
        if wavelength is None and self.auto_grid_used:
            wavelength = self.wavelength_from_sources(sources)
            log.info(f"Auto meshing using wavelength {wavelength:1.4f} defined from sources.")

        # Warn user if ``GridType`` along some axis is not ``AutoGrid`` and
        # ``override_structures`` is not empty. The override structures
        # are not effective along those axes.
        for axis_ind, override_used_axis, grid_axis in zip(
            ["x", "y", "z"], self.override_structures_used, [self.grid_x, self.grid_y, self.grid_z]
        ):
            if override_used_axis and not isinstance(grid_axis, AutoGrid):
                log.warning(
                    f"Override structures take no effect along {axis_ind}-axis. "
                    "If intending to apply override structures to this axis, "
                    "use 'AutoGrid'."
                )

        grids_1d = [self.grid_x, self.grid_y, self.grid_z]
        coords_dict = {}
        for idim, (dim, grid_1d) in enumerate(zip("xyz", grids_1d)):
            coords_dict[dim] = grid_1d.make_coords(
                axis=idim,
                structures=list(structures) + list(self.override_structures),
                symmetry=symmetry,
                periodic=periodic[idim],
                wavelength=wavelength,
                num_pml_layers=num_pml_layers[idim],
            )

        coords = Coords(**coords_dict)
        return Grid(boundaries=coords)

    @classmethod
    def auto(  # pylint:disable=too-many-arguments
        cls,
        wavelength: pd.PositiveFloat = None,
        min_steps_per_wvl: pd.PositiveFloat = 10.0,
        max_scale: pd.PositiveFloat = 1.4,
        override_structures: List[StructureType] = (),
        dl_min: pd.NonNegativeFloat = 0.0,
        mesher: MesherType = GradedMesher(),
    ) -> GridSpec:
        """Use the same :class:`AutoGrid` along each of the three directions.

        Parameters
        ----------
        wavelength : pd.PositiveFloat, optional
            Free-space wavelength for automatic nonuniform grid. It can be 'None'
            if there is at least one source in the simulation, in which case it is defined by
            the source central frequency.
        min_steps_per_wvl : pd.PositiveFloat, optional
            Minimal number of steps per wavelength in each medium.
        max_scale : pd.PositiveFloat, optional
            Sets the maximum ratio between any two consecutive grid steps.
        override_structures : List[StructureType]
            A list of structures that is added on top of the simulation structures in
            the process of generating the grid. This can be used to refine the grid or make it
            coarser depending than the expected need for higher/lower resolution regions.
        dl_min: pd.NonNegativeFloat
            Lower bound of grid size.
        mesher : MesherType = GradedMesher()
            The type of mesher to use to generate the grid automatically.

        Returns
        -------
        GridSpec
            :class:`GridSpec` with the same automatic nonuniform grid settings in each direction.
        """

        grid_1d = AutoGrid(
            min_steps_per_wvl=min_steps_per_wvl,
            max_scale=max_scale,
            dl_min=dl_min,
            mesher=mesher,
        )
        return cls(
            wavelength=wavelength,
            grid_x=grid_1d,
            grid_y=grid_1d,
            grid_z=grid_1d,
            override_structures=override_structures,
        )

    @classmethod
    def uniform(cls, dl: float) -> GridSpec:
        """Use the same :class:`UniformGrid` along each of the three directions.

        Parameters
        ----------
        dl : float
            Grid size for uniform grid generation.

        Returns
        -------
        GridSpec
            :class:`GridSpec` with the same uniform grid size in each direction.
        """

        grid_1d = UniformGrid(dl=dl)
        return cls(grid_x=grid_1d, grid_y=grid_1d, grid_z=grid_1d)
