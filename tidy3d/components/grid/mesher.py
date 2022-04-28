""" Collection of functions for automatically generating a nonuniform grid. """

from abc import ABC, abstractmethod
from typing import Tuple, List, Union

import pydantic as pd
import numpy as np
from scipy.optimize import root_scalar

from ..base import Tidy3dBaseModel
from ..types import Axis, Array
from ..structure import Structure
from ...log import SetupError, ValidationError
from ...constants import C_0, fp_eps


class Mesher(Tidy3dBaseModel, ABC):
    """Abstract class for automatic meshing."""

    @abstractmethod
    def parse_structures(
        self,
        axis: Axis,
        structures: List[Structure],
        wavelength: pd.PositiveFloat,
        min_steps_per_wvl: pd.NonNegativeInt,
    ) -> Tuple[Array[float], Array[float]]:
        """Calculate the positions of all bounding box interfaces along a given axis."""

    @abstractmethod
    def make_grid_multiple_intervals(
        self,
        max_dl_list: Array[float],
        len_interval_list: Array[float],
        max_scale: float,
        is_periodic: bool,
    ) -> List[Array[float]]:
        """Create grid steps in multiple connecting intervals."""


class GradedMesher(Mesher):
    """Implements automatic nonuniform meshing with a set minimum steps per wavelength and
    a graded mesh expanding from higher- to lower-resolution regions."""

    # pylint:disable=too-many-statements,too-many-locals
    def parse_structures(
        self,
        axis: Axis,
        structures: List[Structure],
        wavelength: pd.PositiveFloat,
        min_steps_per_wvl: pd.NonNegativeInt,
    ) -> Tuple[Array[float], Array[float]]:
        """Calculate the positions of all bounding box interfaces along a given axis.
        In this implementation, in most cases the complexity should be O(len(structures)**2),
        although the worst-case complexity may approach O(len(structures)**3).
        However this should only happen in some very contrived cases.

        Parameters
        ----------
        axis : Axis
            Axis index along which to operate.
        structures : List[Structure]
            List of structures, with the simulation structure being the first item.
        wavelength : pd.PositiveFloat
            Wavelength to use for the step size and for dispersive media epsilon.
        min_steps_per_wvl : pd.NonNegativeInt
            Minimum requested steps per wavelength.

        Returns
        -------
        interval_coords : Array
            An array of coordinates, where the first element is the simulation min boundary, the
            last element is the simulation max boundary, and the intermediate coordinates are all
            locations where a structure has a bounding box edge along the specified axis.
            The boundaries are filtered such that no interval is smaller than the smallest
            of the ``max_steps``.
        max_steps : array_like
            An array of size ``interval_coords.size - 1`` giving the maximum grid step required in
            each ``interval_coords[i]:interval_coords[i+1]`` interval, depending on the materials
            in that interval, the supplied wavelength, and the minimum required step per wavelength.
        """

        # Simulation boundaries
        sim_bmin, sim_bmax = structures[0].geometry.bounds
        domain_bounds = np.array([sim_bmin[axis], sim_bmax[axis]])

        # Required maximum steps in every structure
        structure_steps = self.structure_steps(structures, wavelength, min_steps_per_wvl)

        # If empty simulation, return
        if len(structures) == 1:
            return (domain_bounds, structure_steps)

        # Bounding boxes with the meshing axis rotated to z
        struct_bbox = self.rotate_structure_bounds(structures, axis)

        # Array of coordinates of all intervals; add the simulation domain bounds already
        interval_coords = np.array(domain_bounds)
        # List of indexes of structures which are present in each interval
        interval_structs = [[0]]  # will have len equal to len(interval_coords) - 1
        # List of indexes of structures that every structre contains in 2D
        struct_contains = []  # will have len equal to len(structures)

        for struct_ind in range(len(structures) - 1, 0, -1):
            bbox = struct_bbox[struct_ind]

            # indexes of structures that the current structure contains in 2D
            struct_contains.append(self.contains_2d(bbox, struct_bbox[:struct_ind]))

            # Figure out where to place the bounding box coordinates of current structure
            indsmin = np.argwhere(bbox[0, 2] <= interval_coords)
            indmin = int(indsmin[0])
            bbox0 = np.copy(bbox)
            bbox0[1, 2] = bbox0[0, 2]
            if not self.is_contained(bbox0, struct_bbox[struct_ind + 1 :]):
                # Add current structure bounding box coordinates
                interval_coords = np.insert(interval_coords, indmin, bbox[0, 2])
                # Copy the structure containment list to the newly created interval
                struct_list = interval_structs[max(0, indmin - 1)]
                interval_structs.insert(indmin, struct_list.copy())

            indsmax = np.argwhere(bbox[1, 2] >= interval_coords)
            indmax = int(indsmax[-1])
            bbox0 = np.copy(bbox)
            bbox0[0, 2] = bbox0[1, 2]
            if not self.is_contained(bbox0, struct_bbox[struct_ind + 1 :]):
                indmax += 1
                # Add current structure bounding box coordinates
                interval_coords = np.insert(interval_coords, indmax, bbox[1, 2])
                # Copy the structure containment list to the newly created interval
                struct_list = interval_structs[min(indmax - 1, len(interval_structs) - 1)]
                interval_structs.insert(indmax, struct_list.copy())

            # Add the structure index to all intervals that it spans
            for interval_ind in range(indmin, indmax):
                interval_structs[interval_ind].append(struct_ind)

        # The simulation domain is the lowest structure and doesn't contain anything
        struct_contains.append([])
        # Reverse to match the order in the structures list
        struct_contains = struct_contains[::-1]

        # Truncate intervals to domain bounds
        b_array = np.array(interval_coords)
        in_domain = np.argwhere((b_array >= domain_bounds[0]) * (b_array <= domain_bounds[1]))
        interval_coords = interval_coords[in_domain]
        interval_structs = [interval_structs[int(i)] for i in in_domain if i < b_array.size - 1]

        # Remove intervals that are smaller than the absolute smallest min_step
        min_step = np.amin(structure_steps)
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
            # Structure indexes inside current intervall in order of latter structures come first
            struct_list = interval_structs[coord_ind]
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
            max_step = np.amin(structure_steps[struct_list_filter])
            max_steps.append(float(max_step))

        return interval_coords, np.array(max_steps)

    @staticmethod
    def structure_steps(
        structures: List[Structure], wavelength: float, min_steps_per_wvl: float
    ) -> Array[float]:
        """Get the minimum mesh required in each structure.

        Parameters
        ----------
        structures : List[Structure]
            List of structures, with the simulation structure being the first item.
        wavelength : float
            Wavelength to use for the step size and for dispersive media epsilon.
        min_steps_per_wvl : float
            Minimum requested steps per wavelength.
        """
        min_steps = []
        for structure in structures:
            n, k = structure.medium.eps_complex_to_nk(structure.medium.eps_model(C_0 / wavelength))
            index = max(abs(n), abs(k))
            min_steps.append(wavelength / index / min_steps_per_wvl)
        return np.array(min_steps)

    @staticmethod
    def rotate_structure_bounds(structures: List[Structure], axis: Axis) -> List[Array[float]]:
        """Get sturcture bounding boxes with a given ``axis`` rotated to z.

        Parameters
        ----------
        structures : List[Structure]
            List of structures, with the simulation structure being the first item.
        axis : Axis
            Axis index to place last.

        Returns
        -------
        List[Array[float]]
            A list of the bounding boxes of shape ``(2, 3)`` for each structure, with the bounds
            along ``axis`` being ``(:, 2)``.
        """
        struct_bbox = []
        for structure in structures:
            # Get 3D bounding box and rotate axes
            bmin, bmax = structure.geometry.bounds
            bmin_ax, bmin_plane = structure.geometry.pop_axis(bmin, axis=axis)
            bmax_ax, bmax_plane = structure.geometry.pop_axis(bmax, axis=axis)
            bounds = np.array([list(bmin_plane) + [bmin_ax], list(bmax_plane) + [bmax_ax]])
            struct_bbox.append(bounds)
        return struct_bbox

    @staticmethod
    def is_contained(bbox0: Array[float], bbox_list: List[Array[float]]) -> bool:
        """Check if a bounding box is contained in any of a list of bounding boxes.

        Parameters
        ----------
        bbox0 : Array[float]
            Bounding box to check.
        bbox_list : List[Array[float]]
            List of bounding boxes to check if they contain ``bbox0``.

        Returns
        -------
        contained : bool
            ``True`` if ``bbox0`` is contained in any of the boxes in the list.
        """
        contained = False
        for bounds in bbox_list:
            # It can be much faster to write out the conditions one by one than e.g. to use np.all
            # on the bottom values and np.all on the top values
            if all(
                [
                    bbox0[0, 0] >= bounds[0, 0],
                    bbox0[1, 0] <= bounds[1, 0],
                    bbox0[0, 1] >= bounds[0, 1],
                    bbox0[1, 1] <= bounds[1, 1],
                    bbox0[0, 2] >= bounds[0, 2],
                    bbox0[1, 2] <= bounds[1, 2],
                ]
            ):
                contained = True
        return contained

    @staticmethod
    def contains_2d(bbox0: Array[float], bbox_list: List[Array[float]]) -> List[int]:
        """Check if a bounding box contains along the first two dimensions any of a list of
        bounding boxes.

        Parameters
        ----------
        bbox0 : Array[float]
            Bounding box to check.
        bbox_list : List[Array[float]]
            List of bounding boxes to check if they are contained in ``bbox0``.

        Returns
        -------
        List[int]
            A list with all the indexes into the ``bbox_list`` that are contained in ``bbox0``
            along the first two dimensions.
        """
        struct_contains_inds = []
        for ind, bounds in enumerate(bbox_list):
            if all(
                [
                    bbox0[0, 0] <= bounds[0, 0],
                    bbox0[1, 0] >= bounds[1, 0],
                    bbox0[0, 1] <= bounds[0, 1],
                    bbox0[1, 1] >= bounds[1, 1],
                ]
            ):
                struct_contains_inds.append(ind)
        return struct_contains_inds

    def make_grid_multiple_intervals(  # pylint:disable=too-many-locals
        self,
        max_dl_list: Array[float],
        len_interval_list: Array[float],
        max_scale: float,
        is_periodic: bool,
    ) -> List[Array[float]]:
        """Create grid steps in multiple connecting intervals of length specified by
        ``len_interval_list``. The maximal allowed step size in each interval is given by
        ``max_dl_list``. The maximum ratio between neighboring steps is bounded by ``max_scale``.

        Parameters
        ----------
        max_dl_list : Array[float]
            Maximal allowed step size of each interval.
        len_interval_list : Array[float]
            A list of interval lengths
        max_scale : float
            Maximal ratio between consecutive steps.
        is_periodic : bool
            Apply periodic boundary condition or not.

        Returns
        -------
        List[Array[float]]
            A list of of step sizes in each interval.
        """

        num_intervals = len(len_interval_list)
        if len(max_dl_list) != num_intervals:
            raise SetupError(
                "Maximal step size list should have the same length as len_interval_list."
            )

        # initialize step size on the left and right boundary of each interval
        # by assuming possible non-integar step number
        left_dl_list, right_dl_list = self.grid_multiple_interval_analy_refinement(
            max_dl_list, len_interval_list, max_scale, is_periodic
        )

        # initialize grid steps
        dl_list = []
        for interval_ind in range(num_intervals):
            dl_list.append(
                self.make_grid_in_interval(
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

                # update grid steps in this interval if necessary
                if refine_local > 0:
                    dl_list[interval_ind] = self.make_grid_in_interval(
                        left_dl,
                        right_dl,
                        max_dl_list[interval_ind],
                        max_scale,
                        len_interval_list[interval_ind],
                    )

        return dl_list

    def grid_multiple_interval_analy_refinement(
        self,
        max_dl_list: Array[float],
        len_interval_list: Array[float],
        max_scale: float,
        is_periodic: bool,
    ) -> Tuple[Array[float], Array[float]]:
        """Analytical refinement for multiple intervals. "analytical" meaning we allow
        non-integar step sizes, so that we don't consider snapping here.

        Parameters
        ----------
        max_dl_list : Array[float]
            Maximal allowed step size of each interval.
        len_interval_list : Array[float]
            A list of interval lengths
        max_scale : float
            Maximal ratio between consecutive steps.
        is_periodic : bool
            Apply periodic boundary condition or not.

        Returns
        -------
        Tuple[Array[float], Array[float]]
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
    def make_grid_in_interval(
        self,
        left_neighbor_dl: float,
        right_neighbor_dl: float,
        max_dl: float,
        max_scale: float,
        len_interval: float,
    ) -> Array[float]:
        """Create a set of grid steps in an interval of length ``len_interval``,
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
        Array[float]
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
        grid_type = self.grid_type_in_interval(left_dl, right_dl, max_dl, max_scale, len_interval)

        # single pixel
        if grid_type == -1:
            return np.array([len_interval])

        # uniform and multiple pixels
        if grid_type == 0:
            even_dl = min(left_dl, right_dl)
            num_cells = int(np.ceil(len_interval / even_dl))
            return np.array([len_interval / num_cells] * num_cells)

        # grid_type = 1
        # We first set up grid steps from small to large, and then flip
        # their order if right_dl < left_dl
        small_dl = min(left_dl, right_dl)
        large_dl = max(left_dl, right_dl)
        if grid_type == 1:
            # Can small_dl scale to large_dl under max_scale within interval?
            # Compute the number of steps it takes to scale from small_dl to large_dl
            # Check the remaining length in the interval
            num_step = 1 + int(np.floor(np.log(large_dl / small_dl) / np.log(max_scale)))
            len_scale = small_dl * (1 - max_scale**num_step) / (1 - max_scale)
            len_remaining = len_interval - len_scale

            # 1) interval length too small, cannot increase to large_dl, or barely can,
            #    but the remaing part is less than large_dl
            if len_remaining < large_dl:
                dl_list = self.grid_grow_in_interval(small_dl, max_scale, len_interval)
                return dl_list if left_dl <= right_dl else np.flip(dl_list)

            # 2) interval length sufficient, so it will plateau towards large_dl
            dl_list = self.grid_grow_plateau_in_interval(
                small_dl, large_dl, max_scale, len_interval
            )
            return dl_list if left_dl <= right_dl else np.flip(dl_list)

        # grid_type = 2
        if grid_type == 2:
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
                return self.grid_grow_plateau_decrease_in_interval(
                    left_dl, right_dl, max_dl, max_scale, len_interval
                )

            # unable to plateau
            return self.grid_grow_decrease_in_interval(left_dl, right_dl, max_scale, len_interval)

        # unlikely to reach here. For future implementation purpose.
        raise ValidationError("Unimplemented grid type.")

    @staticmethod
    def grid_grow_plateau_decrease_in_interval(
        left_dl: float,
        right_dl: float,
        max_dl: float,
        max_scale: float,
        len_interval: float,
    ) -> Array[float]:
        """In an interval, grid grows, plateau, and decrease, resembling Lambda letter but
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

    @staticmethod
    def grid_grow_decrease_in_interval(
        left_dl: float,
        right_dl: float,
        max_scale: float,
        len_interval: float,
    ) -> Array[float]:
        """In an interval, grid grows, and decrease, resembling Lambda letter.

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
        # just use uniform griding with min(left_dl, right_dl)
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

    @staticmethod
    def grid_grow_plateau_in_interval(
        small_dl: float,
        large_dl: float,
        max_scale: float,
        len_interval: float,
    ) -> Array[float]:
        """In an interval, grid grows, then plateau.

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
        Array[float]
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

    @staticmethod
    def grid_grow_in_interval(
        small_dl: float,
        max_scale: float,
        len_interval: float,
    ) -> Array[float]:
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
        Array[float]
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

    @staticmethod
    def grid_type_in_interval(
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
        grid_type : int
            -1 for single pixel grid
            0 for uniform grid
            1 for small to large to optionally plateau grid
            2 for small to large to optionally plateau to small grid
        """

        # uniform grid if interval length is no larger than small_dl
        if len_interval <= min(left_dl, right_dl, max_dl):
            return -1
        # uniform grid if max_scale is too small
        if np.isclose(max_scale, 1):
            return 0
        # uniform grid if max_dl is the smallest
        if max_dl <= left_dl and max_dl <= right_dl:
            return 0

        # type 1
        if max_dl <= left_dl or max_dl <= right_dl:
            return 1

        return 2


MesherType = Union[GradedMesher]
