""" Defines classes specifying meshing in 1D and a collective class for 3D """

from abc import ABC, abstractmethod
from typing import Tuple, List

import numpy as np
import pydantic as pd

from .grid import Coords1D, Coords, Grid
from .auto_mesh import parse_structures, make_mesh_multiple_intervals
from ..base import Tidy3dBaseModel
from ..types import Axis, Symmetry
from ..source import SourceType
from ..structure import Structure
from ...log import SetupError
from ...constants import C_0, MICROMETER


class MeshSpec1d(Tidy3dBaseModel, ABC):

    """Abstract base class, defines 1D mesh generation specifications."""

    def make_coords(  # pylint:disable = too-many-arguments
        self,
        center: float,
        size: float,
        axis: Axis,
        structures: List[Structure],
        symmetry: Symmetry,
        wavelength: float,
        num_pml_layers: Tuple[int, int],
    ) -> Coords1D:
        """Generate 1D coords to be used as grid boundaries, based on simulation parameters.
        Symmetry, and PML layers will be treated here.

        Parameters
        ----------
        center : float
            Center of simulation domain along a given axis.
        size : float
            Size of simulation domain along a given axis.
        axis : Axis
            Axis of this direction.
        structures : List[Structure]
            List of structures present in simulation.
        symmetry : Symmetry
            Reflection symmetry across a plane bisecting the simulation domain normal
            to a given axis.
        wavelength : float
            Free-space wavelength.
        num_pml_layers : Tuple[int, int]
            number of layers in the absorber + and - direction along one dimension.

        Returns
        -------
        Coords1D:
            1D coords to be used as grid boundaries.
        """

        # Determine if one should apply periodic boundary condition.
        # This should only affect auto nonuniform mesh generation for now.
        is_periodic = False
        if num_pml_layers[0] == 0 and num_pml_layers[1] == 0:
            is_periodic = True

        # generate boundaries
        bound_coords = self._make_coords_initial(
            center, size, axis, structures, wavelength, is_periodic
        )

        # incooperate symmetries
        if symmetry != 0:
            # Offset to center if symmetry present
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
        center: float,
        size: float,
        *args,
    ) -> Coords1D:
        """Generate 1D coords to be used as grid boundaries, based on simulation parameters.
        Symmetry, PML etc. are not considered in this method.

        For auto nonuniform generation, it will take some more arguments.

        Parameters
        ----------
        center : float
            Center of simulation domain along a given axis.
        size : float
            Sie of simulation domain along a given axis.
        *args
            Other arguments

        Returns
        -------
        Coords1D:
            1D coords to be used as grid boundaries.
        """

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


class UniformMesh(MeshSpec1d):

    """Uniform 1D mesh generation"""

    dl: pd.PositiveFloat = pd.Field(
        ...,
        title="Grid Size",
        description="Grid size for uniform grid generation.",
        units=MICROMETER,
    )

    def _make_coords_initial(
        self,
        center: float,
        size: float,
        *args,
    ) -> Coords1D:
        """Uniform 1D coords to be used as grid boundaries.

        Parameters
        ----------
        center : float
            Center of simulation domain along a given axis.
        size : float
            Size of simulation domain along a given axis.
        *args:
            Other arguments all go here.

        Returns
        -------
        Coords1D:
            1D coords to be used as grid boundaries.
        """

        num_cells = round(size / self.dl)

        # Make sure there's at least one cell
        num_cells = max(num_cells, 1)

        # Adjust step size to fit simulation size exactly
        dl_snapped = size / num_cells if size > 0 else self.dl

        # Make bounds
        bound_coords = center - size / 2 + np.arange(num_cells + 1) * dl_snapped

        return bound_coords


class CustomMesh(MeshSpec1d):

    """Customized 1D coords."""

    dl: List[pd.PositiveFloat] = pd.Field(
        ...,
        title="Customized grid sizes.",
        description="An array of customized grid sizes.",
    )

    def _make_coords_initial(
        self,
        center: float,
        size: float,
        *args,
    ) -> Coords1D:
        """Customized 1D coords to be used as grid boundaries.

        Parameters
        ----------
        center : float
            Center of simulation domain along a given axis.
        size : float
            Size of simulation domain along a given axis.
        *args
            Other arguments all go here.

        Returns
        -------
        Coords1D:
            1D coords to be used as grid boundaries.
        """

        # get bounding coordinates
        dl = np.array(self.dl)
        bound_coords = np.append(0.0, np.cumsum(dl))

        # place the middle of the bounds at the center of the simulation along dimension
        bound_coords += center - bound_coords[-1] / 2

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


class AutoMesh(MeshSpec1d):
    """Specification for non-uniform grid along a given dimension.

    Example
    -------
    >>> mesh_1d = AutoMesh(min_steps_per_wvl=16, max_scale=1.4)
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

    def _make_coords_initial(  # pylint:disable = arguments-differ, too-many-arguments
        self,
        center: float,
        size: float,
        axis: Axis,
        structures: List[Structure],
        wavelength: float,
        is_periodic: bool,
    ) -> Coords1D:
        """Customized 1D coords to be used as grid boundaries.

        Parameters
        ----------
        center : float
            Center of simulation domain along a given axis.
        size : float
            Size of simulation domain along a given axis.
        axis : Axis
            Axis of this direction.
        structures : List[Structure]
            List of structures present in simulation.
        wavelength : float
            Free-space wavelength.
        is_periodic : bool
            Apply periodic boundary condition or not.

        Returns
        -------
        Coords1D:
            1D coords to be used as grid boundaries.
        """

        # parse structures
        interval_coords, max_dl_list = parse_structures(
            axis, structures, wavelength, self.min_steps_per_wvl
        )

        # generate mesh steps
        interval_coords = np.array(interval_coords).flatten()
        max_dl_list = np.array(max_dl_list).flatten()
        len_interval_list = interval_coords[1:] - interval_coords[:-1]
        dl_list = make_mesh_multiple_intervals(
            max_dl_list, len_interval_list, self.max_scale, is_periodic
        )

        # generate boundaries
        bound_coords = np.append(0.0, np.cumsum(np.concatenate(dl_list)))
        bound_coords += interval_coords[0]
        return np.array(bound_coords)


class MeshSpec(Tidy3dBaseModel):

    """Mesh specifications"""

    mesh_x: MeshSpec1d = pd.Field(
        AutoMesh(),
        title="Mesh specification along x-axis",
        description="Mesh specification along x-axis",
    )

    mesh_y: MeshSpec1d = pd.Field(
        AutoMesh(),
        title="Mesh specification along y-axis",
        description="Mesh specification along y-axis",
    )

    mesh_z: MeshSpec1d = pd.Field(
        AutoMesh(),
        title="Mesh specification along z-axis",
        description="Mesh specification along z-axis",
    )

    wavelength: float = pd.Field(
        None,
        title="Free-space wavelength",
        description="Free-space wavelength for automatic nonuniform mesh. It can be 'None' "
        "if there is at least one source in the simulation, in which case it is defined by "
        "the source central frequency.",
    )

    @property
    def mesh1d_list(self):
        """A list of the MeshSpec1d-s along each axis."""
        return [self.mesh_x, self.mesh_y, self.mesh_z]

    def make_grid(  # pylint:disable = too-many-arguments
        self,
        structures: List[Structure],
        symmetry: Tuple[Symmetry, Symmetry, Symmetry],
        sources: List[SourceType],
        num_pml_layers: List[Tuple[float, float]],
    ) -> Grid:
        """Make the entire simulation grid based on some simulation parameters.

        Parameters
        ----------
        structures : List[Structure]
            List of structures present in simulation.
        symmetry : Tuple[Symmetry, Symmetry, Symmetry]
            Reflection symmetry across a plane bisecting the simulation domain
            normal to the three axis.
        sources : List[SourceType]
            List of sources.
        num_pml_layers : List[Tuple[float, float]]
            List containing the number of absorber layers in - and + boundaries.

        Returns
        -------
        Grid:
            Entire simulation grid
        """

        center, size = structures[0].geometry.center, structures[0].geometry.size

        # Set up wavelength for automatic mesh generation if needed.

        # No need to do anything if automatic mesh is not used
        auto_mesh_used = np.any([isinstance(mesh, AutoMesh) for mesh in self.mesh1d_list])

        # If auto mesh used and wavelength is None, use central frequency of sources, if any
        wavelength = self.wavelength
        if wavelength is None and auto_mesh_used:
            source_ranges = [source.source_time.frequency_range() for source in sources]
            f_center = np.array([np.sum(s_range) / 2 for s_range in source_ranges])
            # lack of wavelength input
            if len(f_center) == 0:
                raise SetupError(
                    "Automatic mesh generation requires the input of 'wavelength' or sources."
                )

            # multiple sources of different central frequencies
            if len(f_center) > 0 and not np.all(np.isclose(f_center, f_center[0])):
                raise SetupError(
                    "Sources of different central frequencies are supplied. "
                    "Please supply a wavelength value for 'mesh_spec'."
                )
            wavelength = C_0 / f_center[0]

        coords_x = self.mesh_x.make_coords(
            center=center[0],
            size=size[0],
            axis=0,
            structures=structures,
            symmetry=symmetry[0],
            wavelength=wavelength,
            num_pml_layers=num_pml_layers[0],
        )
        coords_y = self.mesh_y.make_coords(
            center=center[1],
            size=size[1],
            axis=1,
            structures=structures,
            symmetry=symmetry[1],
            wavelength=wavelength,
            num_pml_layers=num_pml_layers[1],
        )
        coords_z = self.mesh_z.make_coords(
            center=center[2],
            size=size[2],
            axis=2,
            structures=structures,
            symmetry=symmetry[2],
            wavelength=wavelength,
            num_pml_layers=num_pml_layers[2],
        )

        coords = Coords(x=coords_x, y=coords_y, z=coords_z)
        return Grid(boundaries=coords)
