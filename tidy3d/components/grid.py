"""Defines the FDTD grid."""  # pylint:disable = too-many-lines

from abc import ABC, abstractmethod
from typing import Tuple, List

import numpy as np
from scipy.optimize import root_scalar
import pydantic as pd

from .base import Tidy3dBaseModel, TYPE_TAG_STR
from .types import Array, Axis, Symmetry
from .geometry import Box

# from .medium import DispersiveMedium
from .source import SourceType
from .structure import Structure
from ..log import SetupError, ValidationError
from ..constants import C_0, MICROMETER, fp_eps

# data type of one dimensional coordinate array.
Coords1D = Array[float]


class MeshSpec1D(Tidy3dBaseModel, ABC):

    """Abstract base class, defines 1D mesh generation specifications."""

    def make_coords(  # pylint:disable = too-many-arguments
        self,
        center: float,
        size: float,
        axis: Axis,
        structures: List[Structure],
        symmetry: Symmetry,
        sources: List[SourceType],
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
        sources : List[SourceType]
            List of sources.
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
            center, size, axis, structures, sources, is_periodic
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


class UniformMeshSpec(MeshSpec1D):

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


class CustomMeshSpec(MeshSpec1D):

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


class AutoMeshSpec(MeshSpec1D):
    """Specification for non-uniform grid along a given dimension.

    Example
    -------
    >>> mesh_1d = AutoMeshSpec(min_steps_per_wvl=16, max_scale=1.4, wavelength = 1)
    """

    min_steps_per_wvl: pd.PositiveFloat = pd.Field(
        15,
        title="Minimal number of steps per wavelength",
        description="Minimal number of steps per wavelength in each medium.",
    )

    max_scale: float = pd.Field(
        1.4,
        title="Maximum Grid Size Scaling",
        description="Sets the maximum ratio between any two consecutive grid steps.",
        ge=1.2,
        lt=2.0,
    )

    wavelength: float = pd.Field(
        None,
        title="Wavelength for setting up nonuniform mesh",
        description="Wavelength for setting up nonuniform mesh; It can be `None` "
        "if there is at least one source object in the simulation, and the frequency will "
        "be based on the source central frequency. ",
    )

    def _make_coords_initial(  # pylint:disable = arguments-differ, too-many-arguments
        self,
        center: float,
        size: float,
        axis: Axis,
        structures: List[Structure],
        sources: List[SourceType],
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
        sources : List[SourceType]
            List of sources.
        is_periodic : bool
            Apply periodic boundary condition or not.

        Returns
        -------
        Coords1D:
            1D coords to be used as grid boundaries.
        """

        # First, set up wavelength for mesh
        wavelength = self.wavelength
        # if None, use central frequency of the source
        if wavelength is None:
            source_ranges = [source.source_time.frequency_range() for source in sources]
            f_center = np.array([np.sum(s_range) / 2 for s_range in source_ranges])
            # lack of wavelength input
            if len(f_center) == 0:
                raise SetupError(
                    "Automatic mesh generation requires the input of " "the wavelength, or sources."
                )

            # multiple sources of different central frequencies
            if len(f_center) > 0 and not np.all(np.isclose(f_center, f_center[0])):
                raise SetupError(
                    "Sources of different central frequencies are supplied. "
                    "Please supply the wavelength value for setting up mesh."
                )
            wavelength = C_0 / f_center[0]

        # parse structures
        interval_coords, max_dl_list = self._parse_structures(
            axis, structures, wavelength, self.min_steps_per_wvl
        )

        # generate mesh steps
        interval_coords = np.array(interval_coords).flatten()
        max_dl_list = np.array(max_dl_list).flatten()
        len_interval_list = interval_coords[1:] - interval_coords[:-1]
        dl_list = self._make_mesh_multiple_intervals(
            max_dl_list, len_interval_list, self.max_scale, is_periodic
        )

        # generate boundaries
        bound_coords = np.append(0.0, np.cumsum(np.concatenate(dl_list)))
        bound_coords += interval_coords[0]
        return np.array(bound_coords)

    # pylint:disable=too-many-statements,too-many-locals
    @staticmethod
    def _parse_structures(axis, structures, wvl, min_steps_per_wvl):
        """Calculate the positions of all bounding box interfaces along all three axes.
        In most cases the complexity should be O(len(structures)**2), although the worst-case
        complexity may approach O(len(structures)**3). However this should only happen in some
        very contrived cases.

        Returns
        -------
        interval_coords: array_like
            An array of coordinates, where the first element is the simulation min boundary, the
            last element is the simulation max boundary, and the intermediate coordinates are all
            locations where a structure has a bounding box edge along the specified axis.
        max_steps: array_like
            An array of size ``interval_coords.size`` giving the maximum mesh step required in each
            ``interval_coords[i]:interval_coords[i+1]`` interval, depending on the materials in that
            interval, the supplied wavelength, and the minimum required step per wavelength.
            Periodic boundary conditions are applied such that min_steps[0] = min_steps[-1].
        """

        # Simulation boundaries
        sim_bmin, sim_bmax = structures[0].geometry.bounds
        domain_bounds = np.array([sim_bmin[axis], sim_bmax[axis]])

        # Required minimum steps in every material
        medium_steps = []
        for structure in structures:
            n, k = structure.medium.eps_complex_to_nk(structure.medium.eps_model(C_0 / wvl))
            index = max(abs(n), abs(k))
            medium_steps.append(wvl / index / min_steps_per_wvl)
        medium_steps = np.array(medium_steps)

        # If empty simulation, return
        if len(structures) == 1:
            return (domain_bounds, medium_steps)

        # Coordinates of all bounding boxes
        interval_coords = np.array(domain_bounds)

        # Bounding squares in the plane normal to axis (xmin, ymin, xmax, ymax)
        _, pinds = structures[0].geometry.pop_axis([0, 1, 2], axis=axis)
        sim_plane_bbox = [sim_bmin[pinds[0]], sim_bmin[pinds[1]]]
        sim_plane_bbox += [sim_bmax[pinds[0]], sim_bmax[pinds[1]]]
        plane_bbox = [np.array(sim_plane_bbox)]  # will have len equal to len(structures)

        # list of indexes of structures which are contained in each interval
        interval_structs = [[0]]  # will have len equal to len(interval_coords) - 1

        # list of indexes of structures that are contained in 2D inside another structure
        struct_contains = [[]]  # will have len equal to len(structures)

        for struct_ind, structure in enumerate(structures[1:]):
            # get 3D bounding box and write 2D bouding box
            bmin, bmax = structure.geometry.bounds
            bounds_2d = np.array([bmin[pinds[0]], bmin[pinds[1]], bmax[pinds[0]], bmax[pinds[1]]])
            plane_bbox.append(bounds_2d)

            # indexes of structures that are fully covered in 2D by the current structure
            struct_contain_inds = []
            for ind, plane_bounds in enumerate(plane_bbox[:-1]):
                # faster to do the comparison this way than with e.g. np.all
                if (
                    bounds_2d[0] <= plane_bounds[0]
                    and bounds_2d[1] <= plane_bounds[1]
                    and bounds_2d[2] >= plane_bounds[2]
                    and bounds_2d[3] >= plane_bounds[3]
                ):
                    struct_contain_inds.append(ind)
            struct_contains.append(struct_contain_inds)

            # figure out where to place the bounding box coordinates of current structure
            coord_min, coord_max = bmin[axis], bmax[axis]
            indsmin = np.argwhere(coord_min < interval_coords)
            indsmax = np.argwhere(coord_max > interval_coords)

            # Exit if structure is outside of domain bounds
            if (
                indsmin.size == 0
                or indsmax.size == 0
                or np.any(bounds_2d[:2] >= sim_plane_bbox[2:])
                or np.any(bounds_2d[2:] <= sim_plane_bbox[:2])
            ):
                continue

            # Add current structure bounding box coordinates
            indmin = int(indsmin[0])
            indmax = int(indsmax[-1]) + 2
            interval_coords = np.insert(interval_coords, indmin, coord_min)
            interval_coords = np.insert(interval_coords, indmax, coord_max)
            # Copy the structure containment list to the newly created interval
            structs_list_copy = interval_structs[max(0, indmin - 1)].copy()
            interval_structs.insert(indmin, structs_list_copy)
            structs_list_copy = interval_structs[min(indmax - 1, len(interval_structs) - 1)].copy()
            interval_structs.insert(indmax, structs_list_copy)
            for interval_ind in range(indmin, indmax):
                interval_structs[interval_ind].append(struct_ind + 1)

        # Truncate intervals to domain bounds
        b_array = np.array(interval_coords)
        in_domain = np.argwhere((b_array >= domain_bounds[0]) * (b_array <= domain_bounds[1]))
        interval_coords = interval_coords[in_domain]
        interval_structs = [interval_structs[int(i)] for i in in_domain if i < b_array.size - 1]

        # Remove intervals that are smaller than the absolute smallest min_step
        min_step = np.amin(medium_steps)
        coords_filter = [interval_coords[0]]
        structs_filter = []
        for coord_ind, coord in enumerate(interval_coords[1:]):
            if coord - coords_filter[-1] > min_step:
                coords_filter.append(coord)
                structs_filter.append(interval_structs[coord_ind])
        interval_coords = np.array(coords_filter)
        interval_structs = structs_filter

        # Compute the maximum allowed step size in each interval
        max_steps = []
        for coord_ind, _ in enumerate(interval_coords[:-1]):
            # Structure indexes inside current interval; reverse so first structure on top
            struct_list = interval_structs[coord_ind][::-1]
            struct_list_filter = []
            # Handle containment
            for ind, struct_ind in enumerate(struct_list):
                if ind >= len(struct_list):
                    # This can happen because we modify struct_list in the loop
                    break
                # Add current structure to filtered list
                struct_list_filter.append(struct_ind)
                # Remove all structures that current structure overrides
                contains = struct_contains[struct_ind]
                struct_list = [ind for ind in struct_list if ind not in contains]

            # Define the max step as the minimum over all medium steps of media in this interval
            max_step = np.amin(medium_steps[struct_list_filter])
            max_steps.append(float(max_step))

        return interval_coords, max_steps

    def _make_mesh_multiple_intervals(  # pylint:disable=too-many-locals
        self,
        max_dl_list: np.ndarray,
        len_interval_list: np.ndarray,
        max_scale: float,
        is_periodic: bool,
    ) -> List[np.ndarray]:
        """Create mesh steps in multiple connecting intervals of length specified by
        ``len_interval_list``. The maximal allowed step size in each interval is given by
        ``max_dl_list``. The maximum ratio between neighboring steps is bounded by ``max_scale``.

        Parameters
        ----------
        max_dl_list : np.ndarray
            Maximal allowed step size of each interval.
        len_interval_list : np.ndarray
            A list of interval lengths
        max_scale : float
            Maximal ratio between consecutive steps.
        is_periodic : bool
            Apply periodic boundary condition or not.

        Returns
        -------
        List[np.ndarray]
            A list of of step sizes in each interval.
        """

        num_intervals = len(len_interval_list)
        if len(max_dl_list) != num_intervals:
            raise SetupError(
                "Maximal step size list should have the same length as len_interval_list."
            )

        # initialize step size on the left and right boundary of each interval
        # by assuming possible non-integar step number
        left_dl_list, right_dl_list = self._mesh_multiple_interval_analy_refinement(
            max_dl_list, len_interval_list, max_scale, is_periodic
        )

        # initialize mesh steps
        dl_list = []
        for interval_ind in range(num_intervals):
            dl_list.append(
                self._make_mesh_in_interval(
                    left_dl_list[interval_ind],
                    right_dl_list[interval_ind],
                    max_dl_list[interval_ind],
                    max_scale,
                    len_interval_list[interval_ind],
                )
            )

        # refinement
        refine_edge = 1

        while refine_edge > 0:
            refine_edge = 0
            for interval_ind in range(num_intervals):
                # the step size on the left and right boundary
                left_dl = dl_list[interval_ind][0]
                right_dl = dl_list[interval_ind][-1]
                # the step size to the left and right boundary (neighbor interval)
                left_neighbor_dl = dl_list[interval_ind - 1][-1]
                right_neighbor_dl = dl_list[(interval_ind + 1) % num_intervals][0]

                # for non-periodic case
                if not is_periodic:
                    if interval_ind == 0:
                        left_neighbor_dl = left_dl
                    if interval_ind == num_intervals - 1:
                        right_neighbor_dl = right_dl

                # compare to the neighbor
                refine_local = 0
                if left_dl / left_neighbor_dl > max_scale:
                    left_dl = left_neighbor_dl * (max_scale - fp_eps)
                    refine_edge += 1
                    refine_local += 1

                if right_dl / right_neighbor_dl > max_scale:
                    right_dl = right_neighbor_dl * (max_scale - fp_eps)
                    refine_edge += 1
                    refine_local += 1

                # update mesh steps in this interval if necessary
                if refine_local > 0:
                    dl_list[interval_ind] = self._make_mesh_in_interval(
                        left_dl,
                        right_dl,
                        max_dl_list[interval_ind],
                        max_scale,
                        len_interval_list[interval_ind],
                    )

        return dl_list

    def _mesh_multiple_interval_analy_refinement(
        self,
        max_dl_list: np.ndarray,
        len_interval_list: np.ndarray,
        max_scale: float,
        is_periodic: bool,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Analytical refinement for multiple intervals. "analytical" meaning we allow
        non-integar step sizes, so that we don't consider snapping here.

        Parameters
        ----------
        max_dl_list : np.ndarray
            Maximal allowed step size of each interval.
        len_interval_list : np.ndarray
            A list of interval lengths
        max_scale : float
            Maximal ratio between consecutive steps.
        is_periodic : bool
            Apply periodic boundary condition or not.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            left and right step sizes of each interval.
        """

        if len(max_dl_list) != len(len_interval_list):
            raise SetupError(
                "Maximal step size list should have the same length as len_interval_list."
            )

        # left and right step sizes based on maximal step size list
        right_dl = np.roll(max_dl_list, shift=-1)
        left_dl = np.roll(max_dl_list, shift=1)
        # consideration for the first and last interval
        if not is_periodic:
            right_dl[-1] = max_dl_list[-1]
            left_dl[0] = max_dl_list[0]

        # Right and left step size that will be applied for each interval
        right_dl = np.minimum(max_dl_list, right_dl)
        left_dl = np.minimum(max_dl_list, left_dl)

        # Update left and right neighbor step size considering the impact of neighbor intervals
        refine_analy = 1

        while refine_analy > 0:
            refine_analy = 0
            # from left to right, grow to fill up len_interval, minimal 1 step
            tmp_step = 1 - len_interval_list / left_dl * (1 - max_scale)
            num_step = np.maximum(np.log(tmp_step) / np.log(max_scale), 1)
            left_to_right_dl = left_dl * max_scale ** (num_step - 1)
            update_ind = left_to_right_dl < right_dl
            right_dl[update_ind] = left_to_right_dl[update_ind]

            if not is_periodic:
                update_ind[-1] = False

            if np.any(update_ind):
                refine_analy = 1
                left_dl[np.roll(update_ind, shift=1)] = left_to_right_dl[update_ind]

            # from right to left, grow to fill up len_interval, minimal 1 step
            tmp_step = 1 - len_interval_list / right_dl * (1 - max_scale)
            num_step = np.maximum(np.log(tmp_step) / np.log(max_scale), 1)
            right_to_left_dl = right_dl * max_scale ** (num_step - 1)
            update_ind = right_to_left_dl < left_dl
            left_dl[update_ind] = right_to_left_dl[update_ind]

            if not is_periodic:
                update_ind[0] = False

            if np.any(update_ind):
                refine_analy = 1
                right_dl[np.roll(update_ind, shift=-1)] = right_to_left_dl[update_ind]

        if not is_periodic:
            left_dl[0] = max_dl_list[0]
            right_dl[-1] = max_dl_list[-1]

        return left_dl, right_dl

    # pylint:disable=too-many-locals, too-many-return-statements, too-many-arguments
    def _make_mesh_in_interval(
        self,
        left_neighbor_dl: float,
        right_neighbor_dl: float,
        max_dl: float,
        max_scale: float,
        len_interval: float,
    ) -> np.ndarray:
        """Create a set of mesh steps in an interval of length ``len_interval``,
        with first step no larger than ``max_scale * left_neighbor_dl`` and last step no larger than
        ``max_scale * right_neighbor_dl``, with maximum ratio ``max_scale`` between
        neighboring steps. All steps should be no larger than ``max_dl``.

        Parameters
        ----------
        left_neighbor_dl : float
            Step size to left boundary of the interval.
        right_neighbor_dl : float
            Step size to right boundary of the interval.
        max_dl : float
            Maximal step size within the interval.
        max_scale : float
            Maximal ratio between consecutive steps.
        len_interval: float
            Length of the interval.

        Returns
        -------
        np.ndarray
            A list of step sizes in the interval.
        """

        # some validations
        if left_neighbor_dl <= 0 or right_neighbor_dl <= 0 or max_dl <= 0:
            raise ValidationError("Step size needs to be positive.")
        if len_interval <= 0:
            raise ValidationError("The length of the interval must be larger than 0.")
        if max_scale < 1:
            raise ValidationError("max_scale cannot be smaller than 1.")

        # first and last step size
        left_dl = min(max_dl, left_neighbor_dl)
        right_dl = min(max_dl, right_neighbor_dl)

        # classifications:
        mesh_type = self._mesh_type_in_interval(left_dl, right_dl, max_dl, max_scale, len_interval)

        # single pixel
        if mesh_type == -1:
            return np.array([len_interval])

        # uniform and multiple pixels
        if mesh_type == 0:
            even_dl = min(left_dl, right_dl)
            num_cells = int(np.floor(len_interval / even_dl))

            # Length of the interval assuming this num_cells.
            # if it doesn't cover the interval, increase num_cells,
            # which is equivalent of decreasing mesh step size.
            size_snapped = num_cells * even_dl
            if size_snapped < len_interval:
                num_cells += 1
            return np.array([len_interval / num_cells] * num_cells)

        # mesh_type = 1
        # We first set up mesh steps from small to large, and then flip
        # their order if right_dl < left_dl
        small_dl = min(left_dl, right_dl)
        large_dl = max(left_dl, right_dl)
        if mesh_type == 1:
            # Can small_dl scale to large_dl under max_scale within interval?
            # Compute the number of steps it takes to scale from small_dl to large_dl
            # Check the remaining length in the interval
            num_step = 1 + int(np.floor(np.log(large_dl / small_dl) / np.log(max_scale)))
            len_scale = small_dl * (1 - max_scale**num_step) / (1 - max_scale)
            len_remaining = len_interval - len_scale

            # 1) interval length too small, cannot increase to large_dl, or barely can,
            #    but the remaing part is less than large_dl
            if len_remaining < large_dl:
                dl_list = self._mesh_grow_in_interval(small_dl, max_scale, len_interval)
                return dl_list if left_dl <= right_dl else np.flip(dl_list)

            # 2) interval length sufficient, so it will plateau towards large_dl
            dl_list = self._mesh_grow_plateau_in_interval(
                small_dl, large_dl, max_scale, len_interval
            )
            return dl_list if left_dl <= right_dl else np.flip(dl_list)

        # mesh_type = 2
        if mesh_type == 2:
            # Will it be able to plateau?
            # Compute the number of steps it take for both sides to grow to max_it;
            # then compare the length to len_interval
            num_left_step = 1 + int(np.floor(np.log(max_dl / left_dl) / np.log(max_scale)))
            num_right_step = 1 + int(np.floor(np.log(max_dl / right_dl) / np.log(max_scale)))
            len_left = left_dl * (1 - max_scale**num_left_step) / (1 - max_scale)
            len_right = right_dl * (1 - max_scale**num_right_step) / (1 - max_scale)

            len_remaining = len_interval - len_left - len_right

            # able to plateau
            if len_remaining >= max_dl:
                return self._mesh_grow_plateau_decrease_in_interval(
                    left_dl, right_dl, max_dl, max_scale, len_interval
                )

            # unable to plateau
            return self._mesh_grow_decrease_in_interval(left_dl, right_dl, max_scale, len_interval)

        # unlikely to reach here. For future implementation purpose.
        raise ValidationError("Unimplemented mesh type.")

    def _mesh_grow_plateau_decrease_in_interval(  # pylint:disable=too-many-arguments
        self,
        left_dl: float,
        right_dl: float,
        max_dl: float,
        max_scale: float,
        len_interval: float,
    ) -> np.ndarray:
        """In an interval, mesh grows, plateau, and decrease, resembling Lambda letter but
        with plateau in the connection part..

        Parameters
        ----------
        left_dl : float
            Step size at the left boundary.
        right_dl : float
            Step size at the right boundary.
        max_dl : float
            Maximal step size within the interval.
        max_scale : float
            Maximal ratio between consecutive steps.
        len_interval : float
            Length of the interval.
        """

        # Maximum number of steps for undershooting max_dl
        num_left_step = 1 + int(np.floor(np.log(max_dl / left_dl) / np.log(max_scale)))
        num_right_step = 1 + int(np.floor(np.log(max_dl / right_dl) / np.log(max_scale)))

        # step list, in ascending order
        dl_list_left = np.array([left_dl * max_scale**i for i in range(num_left_step)])
        dl_list_right = np.array([right_dl * max_scale**i for i in range(num_right_step)])

        # length
        len_left = left_dl * (1 - max_scale**num_left_step) / (1 - max_scale)
        len_right = right_dl * (1 - max_scale**num_right_step) / (1 - max_scale)

        # remaining part for constant large_dl
        num_const_step = int(np.floor((len_interval - len_left - len_right) / max_dl))
        dl_list_const = np.array([max_dl] * num_const_step)
        len_const = num_const_step * max_dl

        # mismatch
        len_mismatch = len_interval - len_left - len_right - len_const

        # (1) happens to be the right length
        if np.isclose(len_mismatch, 0):
            return np.concatenate((dl_list_left, dl_list_const, np.flip(dl_list_right)))

        # (2) sufficient remaining part, can be inserted to left or right
        if len_mismatch >= left_dl:
            index_mis = np.searchsorted(dl_list_left, len_mismatch)
            dl_list_left = np.insert(dl_list_left, index_mis, len_mismatch)
            return np.concatenate((dl_list_left, dl_list_const, np.flip(dl_list_right)))

        if len_mismatch >= right_dl:
            index_mis = np.searchsorted(dl_list_right, len_mismatch)
            dl_list_right = np.insert(dl_list_right, index_mis, len_mismatch)
            return np.concatenate((dl_list_left, dl_list_const, np.flip(dl_list_right)))

        # nothing more we can do, let's just add smallest step size,
        # and scale each pixel
        if left_dl <= right_dl:
            dl_list_left = np.append(left_dl, dl_list_left)
        else:
            dl_list_right = np.append(right_dl, dl_list_right)
        dl_list = np.concatenate((dl_list_left, dl_list_const, np.flip(dl_list_right)))
        dl_list *= len_interval / np.sum(dl_list)
        return dl_list

    def _mesh_grow_decrease_in_interval(
        self,
        left_dl: float,
        right_dl: float,
        max_scale: float,
        len_interval: float,
    ) -> np.ndarray:
        """In an interval, mesh grows, and decrease, resembling Lambda letter.

        Parameters
        ----------
        left_dl : float
            Step size at the left boundary.
        right_dl : float
            Step size at the right boundary.
        max_scale : float
            Maximal ratio between consecutive steps.
        len_interval : float
            Length of the interval.
        """

        # interval too small, it shouldn't happen if bounding box filter is properly handled
        # just use uniform meshing with min(left_dl, right_dl)
        if len_interval < left_dl + right_dl:
            even_dl = min(left_dl, right_dl)
            num_cells = int(np.floor(len_interval / even_dl))
            size_snapped = num_cells * even_dl
            if size_snapped < len_interval:
                num_cells += 1
            return np.array([len_interval / num_cells] * num_cells)

        # The maximal number of steps for both sides to undershoot the interval,
        # assuming the last step size from both sides grow to the same size before
        # taking ``floor`` to take integar number of steps.

        # The advantage is that even after taking integar number of steps cutoff,
        # the last step size from the two side will not viloate max_scale.

        tmp_num_l = ((left_dl + right_dl) - len_interval * (1 - max_scale)) / 2 / left_dl
        tmp_num_r = ((left_dl + right_dl) - len_interval * (1 - max_scale)) / 2 / right_dl
        num_left_step = max(int(np.floor(np.log(tmp_num_l) / np.log(max_scale))), 0)
        num_right_step = max(int(np.floor(np.log(tmp_num_r) / np.log(max_scale))), 0)

        # step list, in ascending order
        dl_list_left = np.array([left_dl * max_scale**i for i in range(num_left_step)])
        dl_list_right = np.array([right_dl * max_scale**i for i in range(num_right_step)])

        # length
        len_left = left_dl * (1 - max_scale**num_left_step) / (1 - max_scale)
        len_right = right_dl * (1 - max_scale**num_right_step) / (1 - max_scale)

        # mismatch
        len_mismatch = len_interval - len_left - len_right

        # (1) happens to be the right length
        if np.isclose(len_mismatch, 0):
            return np.append(dl_list_left, np.flip(dl_list_right))

        # if len_mismatch is larger than the last step size, insert the last step
        while len(dl_list_left) > 0 and len_mismatch >= dl_list_left[-1]:
            dl_list_left = np.append(dl_list_left, dl_list_left[-1])
            len_mismatch -= dl_list_left[-1]

        while len(dl_list_right) > 0 and len_mismatch >= dl_list_right[-1]:
            dl_list_right = np.append(dl_list_right, dl_list_right[-1])
            len_mismatch -= dl_list_right[-1]

        # (2) sufficient remaining part, can be inserted to dl_left or right
        if len_mismatch >= left_dl:
            index_mis = np.searchsorted(dl_list_left, len_mismatch)
            dl_list_left = np.insert(dl_list_left, index_mis, len_mismatch)
            return np.append(dl_list_left, np.flip(dl_list_right))

        if len_mismatch >= right_dl:
            index_mis = np.searchsorted(dl_list_right, len_mismatch)
            dl_list_right = np.insert(dl_list_right, index_mis, len_mismatch)
            return np.append(dl_list_left, np.flip(dl_list_right))

        # nothing more we can do, let's just add smallest step size,
        # and scale each pixel
        if left_dl <= right_dl:
            dl_list_left = np.append(left_dl, dl_list_left)
        else:
            dl_list_right = np.append(right_dl, dl_list_right)
        dl_list = np.append(dl_list_left, np.flip(dl_list_right))
        dl_list *= len_interval / np.sum(dl_list)
        return dl_list

    def _mesh_grow_plateau_in_interval(
        self,
        small_dl: float,
        large_dl: float,
        max_scale: float,
        len_interval: float,
    ) -> np.ndarray:
        """In an interval, mesh grows, then plateau.

        Parameters
        ----------
        small_dl : float
            The smaller one of step size at the left and right boundaries.
        large_dl : float
            The larger one of step size at the left and right boundaries.
        max_scale : float
            Maximal ratio between consecutive steps.
        len_interval : float
            Length of the interval.

        Returns
        -------
        np.ndarray
            A list of step sizes in the interval, in ascending order.
        """
        # steps for scaling
        num_scale_step = 1 + int(np.floor(np.log(large_dl / small_dl) / np.log(max_scale)))
        dl_list_scale = np.array([small_dl * max_scale**i for i in range(num_scale_step)])
        len_scale = small_dl * (1 - max_scale**num_scale_step) / (1 - max_scale)

        # remaining part for constant large_dl
        num_const_step = int(np.floor((len_interval - len_scale) / large_dl))
        dl_list_const = np.array([large_dl] * num_const_step)
        len_const = large_dl * num_const_step

        # mismatch
        len_mismatch = len_interval - len_scale - len_const

        # (1) happens to be the right length
        if np.isclose(len_mismatch, 0):
            return np.append(dl_list_scale, dl_list_const)

        # (2) sufficient remaining part, can be inserted to dl_list_scale
        if len_mismatch >= small_dl:
            index_mis = np.searchsorted(dl_list_scale, len_mismatch)
            dl_list_scale = np.insert(dl_list_scale, index_mis, len_mismatch)
            return np.append(dl_list_scale, dl_list_const)

        # nothing more we can do, let's just add smallest step size,
        # and scale each pixel
        dl_list_scale = np.append(small_dl, dl_list_scale)
        dl_list = np.append(dl_list_scale, dl_list_const)
        dl_list *= len_interval / np.sum(dl_list)
        return dl_list

    def _mesh_grow_in_interval(
        self,
        small_dl: float,
        max_scale: float,
        len_interval: float,
    ) -> np.ndarray:
        """Mesh simply grows in an interval.

        Parameters
        ----------
        small_dl : float
            The smaller one of step size at the left and right boundaries.
        max_scale : float
            Maximal ratio between consecutive steps.
        len_interval : float
            Length of the interval.

        Returns
        -------
        np.ndarray
            A list of step sizes in the interval, in ascending order.
        """

        # Maximal number of steps for undershooting the interval.
        tmp_step = 1 - len_interval / small_dl * (1 - max_scale)
        num_step = int(np.floor(np.log(tmp_step) / np.log(max_scale)))

        # assuming num_step grids and scaling = max_scale
        dl_list = np.array([small_dl * max_scale**i for i in range(num_step)])
        size_snapped = small_dl * (1 - max_scale**num_step) / (1 - max_scale)

        # mismatch
        len_mismatch = len_interval - size_snapped

        # (1) happens to be the right length
        if np.isclose(len_mismatch, 0):
            return dl_list

        # (2) sufficient remaining part, can be inserted
        if len_mismatch >= small_dl:
            index_mis = np.searchsorted(dl_list, len_mismatch)
            dl_list = np.insert(dl_list, index_mis, len_mismatch)
            return dl_list

        # (3) remaining part not sufficient to insert, but will not
        # violate max_scale by repearting 1st step, and the last step to include
        # the mismatch part
        if num_step >= 2 and len_mismatch >= small_dl - (1 - 1.0 / max_scale**2) * dl_list[-1]:
            dl_list = np.append(small_dl, dl_list)
            dl_list[-1] += len_mismatch - small_dl
            return dl_list

        # (4) let's see if we can squeeze something out of smaller scaling.
        # For this case, duplicate the 1st step size.
        len_mismatch_even = len_interval - num_step * small_dl
        if np.isclose(len_mismatch_even, small_dl):
            return np.array([small_dl] * (num_step + 1))

        if len_mismatch_even > small_dl:

            def fun_scale(new_scale):
                if np.isclose(new_scale, 1.0):
                    return len_interval - small_dl * (1 + num_step)
                return (
                    len_interval
                    - small_dl * (1 - new_scale**num_step) / (1 - new_scale)
                    - small_dl
                )

            # solve for new scaling factor
            sol_scale = root_scalar(fun_scale, bracket=[1, max_scale])
            # if not converged, let's use the last strategy
            if sol_scale.converged:
                new_scale = sol_scale.root
                dl_list = np.array([small_dl * new_scale**i for i in range(num_step)])
                dl_list = np.append(small_dl, dl_list)
                return dl_list

        # nothing more we can do, let's just add smallest step size,
        # and scale each pixel
        dl_list = np.append(small_dl, dl_list)
        dl_list *= len_interval / np.sum(dl_list)
        return dl_list

    def _mesh_type_in_interval(  # pylint:disable=too-many-arguments
        self,
        left_dl: float,
        right_dl: float,
        max_dl: float,
        max_scale: float,
        len_interval: float,
    ) -> int:
        """Mesh type check (in an interval).

        Parameters
        ----------
        left_dl : float
            Step size at left boundary of the interval.
        right_dl : float
            Step size at right boundary of the interval.
        max_dl : float
            Maximal step size within the interval.
        max_scale : float
            Maximal ratio between consecutive steps.
        len_interval: float
            Length of the interval.

        Returns
        -------
        mesh_type : int
            -1 for single pixel mesh
            0 for uniform mesh
            1 for small to large to optionally plateau mesh
            2 for small to large to optionally plateau to small mesh
        """

        # uniform mesh if interval length is no larger than small_dl
        if len_interval <= min(left_dl, right_dl, max_dl):
            return -1
        # uniform mesh if max_scale is too small
        if np.isclose(max_scale, 1):
            return 0
        # uniform mesh if max_dl is the smallest
        if max_dl <= left_dl and max_dl <= right_dl:
            return 0

        # type 1
        if max_dl <= left_dl or max_dl <= right_dl:
            return 1

        return 2


class Coords(Tidy3dBaseModel):
    """Holds data about a set of x,y,z positions on a grid.

    Example
    -------
    >>> x = np.linspace(-1, 1, 10)
    >>> y = np.linspace(-1, 1, 11)
    >>> z = np.linspace(-1, 1, 12)
    >>> coords = Coords(x=x, y=y, z=z)
    """

    x: Coords1D = pd.Field(
        ..., title="X Coordinates", description="1-dimensional array of x coordinates."
    )

    y: Coords1D = pd.Field(
        ..., title="Y Coordinates", description="1-dimensional array of y coordinates."
    )

    z: Coords1D = pd.Field(
        ..., title="Z Coordinates", description="1-dimensional array of z coordinates."
    )

    @property
    def to_list(self):
        """Return a list of the three Coord1D objects."""
        return list(self.dict(exclude={TYPE_TAG_STR}).values())


class FieldGrid(Tidy3dBaseModel):
    """Holds the grid data for a single field.

    Example
    -------
    >>> x = np.linspace(-1, 1, 10)
    >>> y = np.linspace(-1, 1, 11)
    >>> z = np.linspace(-1, 1, 12)
    >>> coords = Coords(x=x, y=y, z=z)
    >>> field_grid = FieldGrid(x=coords, y=coords, z=coords)
    """

    x: Coords = pd.Field(
        ...,
        title="X Positions",
        description="x,y,z coordinates of the locations of the x-component of a vector field.",
    )

    y: Coords = pd.Field(
        ...,
        title="Y Positions",
        description="x,y,z coordinates of the locations of the y-component of a vector field.",
    )

    z: Coords = pd.Field(
        ...,
        title="Z Positions",
        description="x,y,z coordinates of the locations of the z-component of a vector field.",
    )


class YeeGrid(Tidy3dBaseModel):
    """Holds the yee grid coordinates for each of the E and H positions.

    Example
    -------
    >>> x = np.linspace(-1, 1, 10)
    >>> y = np.linspace(-1, 1, 11)
    >>> z = np.linspace(-1, 1, 12)
    >>> coords = Coords(x=x, y=y, z=z)
    >>> field_grid = FieldGrid(x=coords, y=coords, z=coords)
    >>> yee_grid = YeeGrid(E=field_grid, H=field_grid)
    >>> Ex_coords = yee_grid.E.x
    """

    E: FieldGrid = pd.Field(
        ...,
        title="Electric Field Grid",
        description="Coordinates of the locations of all three components of the electric field.",
    )

    H: FieldGrid = pd.Field(
        ...,
        title="Electric Field Grid",
        description="Coordinates of the locations of all three components of the magnetic field.",
    )

    @property
    def grid_dict(self):
        """The Yee grid coordinates associated to various field components as a dictionary."""
        yee_grid_dict = {
            "Ex": self.E.x,
            "Ey": self.E.y,
            "Ez": self.E.z,
            "Hx": self.H.x,
            "Hy": self.H.y,
            "Hz": self.H.z,
        }
        return yee_grid_dict


class Grid(Tidy3dBaseModel):
    """Contains all information about the spatial positions of the FDTD grid.

    Example
    -------
    >>> x = np.linspace(-1, 1, 10)
    >>> y = np.linspace(-1, 1, 11)
    >>> z = np.linspace(-1, 1, 12)
    >>> coords = Coords(x=x, y=y, z=z)
    >>> grid = Grid(boundaries=coords)
    >>> centers = grid.centers
    >>> sizes = grid.sizes
    >>> yee_grid = grid.yee
    """

    boundaries: Coords = pd.Field(
        ...,
        title="Boundary Coordinates",
        description="x,y,z coordinates of the boundaries between cells, defining the FDTD grid.",
    )

    @staticmethod
    def _avg(coords1d: Coords1D):
        """Return average positions of an array of 1D coordinates."""
        return (coords1d[1:] + coords1d[:-1]) / 2.0

    @staticmethod
    def _min(coords1d: Coords1D):
        """Return minus positions of 1D coordinates."""
        return coords1d[:-1]

    @property
    def centers(self) -> Coords:
        """Return centers of the cells in the :class:`Grid`.

        Returns
        -------
        :class:`Coords`
            centers of the FDTD cells in x,y,z stored as :class:`Coords` object.

        Example
        -------
        >>> x = np.linspace(-1, 1, 10)
        >>> y = np.linspace(-1, 1, 11)
        >>> z = np.linspace(-1, 1, 12)
        >>> coords = Coords(x=x, y=y, z=z)
        >>> grid = Grid(boundaries=coords)
        >>> centers = grid.centers
        """
        return Coords(
            **{
                key: self._avg(val)
                for key, val in self.boundaries.dict(exclude={TYPE_TAG_STR}).items()
            }
        )

    @property
    def sizes(self) -> Coords:
        """Return sizes of the cells in the :class:`Grid`.

        Returns
        -------
        :class:`Coords`
            Sizes of the FDTD cells in x,y,z stored as :class:`Coords` object.

        Example
        -------
        >>> x = np.linspace(-1, 1, 10)
        >>> y = np.linspace(-1, 1, 11)
        >>> z = np.linspace(-1, 1, 12)
        >>> coords = Coords(x=x, y=y, z=z)
        >>> grid = Grid(boundaries=coords)
        >>> sizes = grid.sizes
        """
        return Coords(
            **{
                key: np.diff(val)
                for key, val in self.boundaries.dict(exclude={TYPE_TAG_STR}).items()
            }
        )

    @property
    def num_cells(self) -> Tuple[int, int, int]:
        """Return sizes of the cells in the :class:`Grid`.

        Returns
        -------
        tuple[int, int, int]
            Number of cells in the grid in the x, y, z direction.

        Example
        -------
        >>> x = np.linspace(-1, 1, 10)
        >>> y = np.linspace(-1, 1, 11)
        >>> z = np.linspace(-1, 1, 12)
        >>> coords = Coords(x=x, y=y, z=z)
        >>> grid = Grid(boundaries=coords)
        >>> Nx, Ny, Nz = grid.num_cells
        """
        return [
            coords1d.size - 1 for coords1d in self.boundaries.dict(exclude={TYPE_TAG_STR}).values()
        ]

    @property
    def _primal_steps(self) -> Coords:
        """Return primal steps of the cells in the :class:`Grid`.

        Returns
        -------
        :class:`Coords`
            Distances between each of the cell boundaries along each dimension.
        """
        return self.sizes

    @property
    def _dual_steps(self) -> Coords:
        """Return dual steps of the cells in the :class:`Grid`.

        Returns
        -------
        :class:`Coords`
            Distances between each of the cell centers along each dimension, with periodicity
            applied.
        """

        primal_steps = self._primal_steps.dict(exclude={TYPE_TAG_STR})
        dsteps = {}
        for (key, psteps) in primal_steps.items():
            dsteps[key] = (psteps + np.roll(psteps, 1)) / 2

        return Coords(**dsteps)

    @property
    def yee(self) -> YeeGrid:
        """Return the :class:`YeeGrid` defining the yee cell locations for this :class:`Grid`.


        Returns
        -------
        :class:`YeeGrid`
            Stores coordinates of all of the components on the yee lattice.

        Example
        -------
        >>> x = np.linspace(-1, 1, 10)
        >>> y = np.linspace(-1, 1, 11)
        >>> z = np.linspace(-1, 1, 12)
        >>> coords = Coords(x=x, y=y, z=z)
        >>> grid = Grid(boundaries=coords)
        >>> yee_cells = grid.yee
        >>> Ex_positions = yee_cells.E.x
        """
        yee_e_kwargs = {key: self._yee_e(axis=axis) for axis, key in enumerate("xyz")}
        yee_h_kwargs = {key: self._yee_h(axis=axis) for axis, key in enumerate("xyz")}

        yee_e = FieldGrid(**yee_e_kwargs)
        yee_h = FieldGrid(**yee_h_kwargs)
        return YeeGrid(E=yee_e, H=yee_h)

    def __getitem__(self, coord_key: str) -> Coords:
        """quickly get the grid element by grid[key]."""

        coord_dict = {
            "centers": self.centers,
            "sizes": self.sizes,
            "boundaries": self.boundaries,
            "Ex": self.yee.E.x,
            "Ey": self.yee.E.y,
            "Ez": self.yee.E.z,
            "Hx": self.yee.H.x,
            "Hy": self.yee.H.y,
            "Hz": self.yee.H.z,
        }
        if coord_key not in coord_dict:
            raise SetupError(f"key {coord_key} not found in grid with {list(coord_dict.keys())} ")

        return coord_dict.get(coord_key)

    def _yee_e(self, axis: Axis):
        """E field yee lattice sites for axis."""

        boundary_coords = self.boundaries.dict(exclude={TYPE_TAG_STR})

        # initially set all to the minus bounds
        yee_coords = {key: self._min(val) for key, val in boundary_coords.items()}

        # average the axis index between the cell boundaries
        key = "xyz"[axis]
        yee_coords[key] = self._avg(boundary_coords[key])

        return Coords(**yee_coords)

    def _yee_h(self, axis: Axis):
        """H field yee lattice sites for axis."""

        boundary_coords = self.boundaries.dict(exclude={TYPE_TAG_STR})

        # initially set all to centers
        yee_coords = {key: self._avg(val) for key, val in boundary_coords.items()}

        # set the axis index to the minus bounds
        key = "xyz"[axis]
        yee_coords[key] = self._min(boundary_coords[key])

        return Coords(**yee_coords)

    def discretize_inds(self, box: Box, extend: bool = False) -> List[Tuple[int, int]]:
        """Start and stopping indexes for the cells that intersect with a :class:`Box`.

        Parameters
        ----------
        box : :class:`Box`
            Rectangular geometry within simulation to discretize.
        extend : bool = False
            If ``True``, ensure that the returned indexes extend sufficiently in very direction to
            be able to interpolate any field component at any point within the ``box``.

        Returns
        -------
        List[Tuple[int, int]]
            The (start, stop) indexes of the cells that intersect with ``box`` in each of the three
            dimensions.
        """

        pts_min, pts_max = box.bounds
        boundaries = self.boundaries

        inds_list = []

        # for each dimension
        for axis, (pt_min, pt_max) in enumerate(zip(pts_min, pts_max)):
            bound_coords = boundaries.to_list[axis]
            assert pt_min <= pt_max, "min point was greater than max point"

            # index of smallest coord greater than than pt_max
            inds_gt_pt_max = np.where(bound_coords > pt_max)[0]
            ind_max = len(bound_coords) - 1 if len(inds_gt_pt_max) == 0 else inds_gt_pt_max[0]

            # index of largest coord less than or equal to pt_min
            inds_leq_pt_min = np.where(bound_coords <= pt_min)[0]
            ind_min = 0 if len(inds_leq_pt_min) == 0 else inds_leq_pt_min[-1]

            if extend:
                # If the box bounds on the left side are to the left of the closest grid center,
                # we need an extra pixel to be able to interpolate the center components.
                if box.bounds[0][axis] < self.centers.to_list[axis][ind_min]:
                    ind_min -= 1

                # We always need an extra pixel on the right for the surface components.
                ind_max += 1

            # store indexes
            inds_list.append((ind_min, ind_max))

        return inds_list

    def periodic_subspace(self, axis: Axis, ind_beg: int = 0, ind_end: int = 0) -> Coords1D:
        """Pick a subspace of 1D boundaries within ``range(ind_beg, ind_end)``. If any indexes lie
        outside of the grid boundaries array, periodic padding is used, where the zeroth and last
        element of the boundaries are identified.

        Parameters
        ----------
        axis : Axis
            Axis index along which to pick the subspace.
        ind_beg : int = 0
            Starting index for the subspace.
        ind_end : int = 0
            Ending index for the subspace.

        Returns
        -------
        Coords1D
            The subspace of the grid along ``axis``.
        """

        coords = self.boundaries.to_list[axis]
        padded_coords = coords
        num_coords = coords.size
        num_cells = num_coords - 1
        coords_width = coords[-1] - coords[0]

        # Pad on the left if needed
        if ind_beg < 0:
            num_pad = int(np.ceil(-ind_beg / num_cells))
            coords_pad = coords[:-1, None] + (coords_width * np.arange(-num_pad, 0))[None, :]
            coords_pad = coords_pad.T.ravel()
            padded_coords = np.concatenate([coords_pad, padded_coords])
            ind_beg += num_pad * num_cells
            ind_end += num_pad * num_cells

        # Pad on the right if needed
        if ind_end >= padded_coords.size:
            num_pad = int(np.ceil((ind_end - padded_coords.size) / num_cells))
            coords_pad = coords[1:, None] + (coords_width * np.arange(1, num_pad + 1))[None, :]
            coords_pad = coords_pad.T.ravel()
            padded_coords = np.concatenate([padded_coords, coords_pad])

        return padded_coords[ind_beg:ind_end]


class MeshSpec(Tidy3dBaseModel):

    """Mesh specifications"""

    mesh_x: MeshSpec1D = pd.Field(
        AutoMeshSpec(),
        title="Mesh specification along x-axis",
        description="Mesh specification along x-axis",
    )

    mesh_y: MeshSpec1D = pd.Field(
        AutoMeshSpec(),
        title="Mesh specification along y-axis",
        description="Mesh specification along y-axis",
    )

    mesh_z: MeshSpec1D = pd.Field(
        AutoMeshSpec(),
        title="Mesh specification along z-axis",
        description="Mesh specification along z-axis",
    )

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

        coords_x = self.mesh_x.make_coords(
            center=center[0],
            size=size[0],
            axis=0,
            structures=structures,
            symmetry=symmetry[0],
            sources=sources,
            num_pml_layers=num_pml_layers[0],
        )
        coords_y = self.mesh_y.make_coords(
            center=center[1],
            size=size[1],
            axis=1,
            structures=structures,
            symmetry=symmetry[1],
            sources=sources,
            num_pml_layers=num_pml_layers[1],
        )
        coords_z = self.mesh_z.make_coords(
            center=center[2],
            size=size[2],
            axis=2,
            structures=structures,
            symmetry=symmetry[2],
            sources=sources,
            num_pml_layers=num_pml_layers[2],
        )

        coords = Coords(x=coords_x, y=coords_y, z=coords_z)
        return Grid(boundaries=coords)
