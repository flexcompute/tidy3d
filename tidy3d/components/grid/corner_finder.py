"""Find corners of structures on a 2D plane."""

from __future__ import annotations

from typing import Any, Literal, Optional

import numpy as np
import pydantic.v1 as pd

from tidy3d.components.base import Tidy3dBaseModel, cached_property
from tidy3d.components.geometry.base import Box, ClipOperation
from tidy3d.components.geometry.utils import merging_geometries_on_plane
from tidy3d.components.medium import PEC, LossyMetalMedium
from tidy3d.components.structure import Structure
from tidy3d.components.types import ArrayFloat1D, ArrayFloat2D, Axis, Shapely
from tidy3d.constants import inf

CORNER_ANGLE_THRESOLD = 0.1 * np.pi


class CornerFinderSpec(Tidy3dBaseModel):
    """Specification for corner detection on a 2D plane."""

    medium: Literal["metal", "dielectric", "all"] = pd.Field(
        "metal",
        title="Material Type For Corner Identification",
        description="Find corners of structures made of ``medium``, "
        "which can take value ``metal`` for PEC and lossy metal, ``dielectric`` "
        "for non-metallic materials, and ``all`` for all materials.",
    )

    angle_threshold: float = pd.Field(
        CORNER_ANGLE_THRESOLD,
        title="Angle Threshold In Corner Identification",
        description="A vertex is qualified as a corner if the angle spanned by its two edges "
        "is larger than the supplementary angle of "
        "this threshold value.",
        ge=0,
        lt=np.pi,
    )

    distance_threshold: Optional[pd.PositiveFloat] = pd.Field(
        None,
        title="Distance Threshold In Corner Identification",
        description="If not ``None`` and the distance of the vertex to its neighboring vertices "
        "is below the threshold value based on Douglas-Peucker algorithm, the vertex is disqualified as a corner.",
    )

    concave_resolution: Optional[pd.PositiveInt] = pd.Field(
        None,
        title="Concave Region Resolution.",
        description="Specifies number of steps to use for determining `dl_min` based on concave featues."
        "If set to ``None``, then the corresponding `dl_min` reduction is not applied.",
    )

    convex_resolution: Optional[pd.PositiveInt] = pd.Field(
        None,
        title="Convex Region Resolution.",
        description="Specifies number of steps to use for determining `dl_min` based on convex featues."
        "If set to ``None``, then the corresponding `dl_min` reduction is not applied.",
    )

    mixed_resolution: Optional[pd.PositiveInt] = pd.Field(
        None,
        title="Mixed Region Resolution.",
        description="Specifies number of steps to use for determining `dl_min` based on mixed featues."
        "If set to ``None``, then the corresponding `dl_min` reduction is not applied.",
    )

    @cached_property
    def _no_min_dl_override(self):
        return all(
            (
                self.concave_resolution is None,
                self.convex_resolution is None,
                self.mixed_resolution is None,
            )
        )

    @classmethod
    def _merged_pec_on_plane(
        cls,
        normal_axis: Axis,
        coord: float,
        structure_list: list[Structure],
        center: tuple[float, float] = [0, 0, 0],
        size: tuple[float, float, float] = [inf, inf, inf],
    ) -> list[tuple[Any, Shapely]]:
        """On a 2D plane specified by axis = `normal_axis` and coordinate `coord`, merge geometries made of PEC.

        Parameters
        ----------
        normal_axis : Axis
            Axis normal to the 2D plane.
        coord : float
            Position of plane along the normal axis.
        structure_list : List[Structure]
            List of structures present in simulation.
        center : Tuple[float, float] = [0, 0, 0]
            Center of the 2D plane (coordinate along ``axis`` is ignored)
        size : Tuple[float, float, float] = [inf, inf, inf]
            Size of the 2D plane (size along ``axis`` is ignored)

        Returns
        -------
        List[Tuple[Any, Shapely]]
            List of shapes and their property value on the plane after merging.
        """

        # Construct plane
        slice_center = list(center)
        slice_size = list(size)
        slice_center[normal_axis] = coord
        slice_size[normal_axis] = 0
        plane = Box(center=slice_center, size=slice_size)

        # prepare geometry and medium list
        geometry_list = [structure.geometry for structure in structure_list]
        # For metal, we don't distinguish between LossyMetal and PEC,
        # so they'll be merged to PEC. Other materials are considered as dielectric.
        medium_list = (structure.medium for structure in structure_list)
        medium_list = [
            PEC if (mat.is_pec or isinstance(mat, LossyMetalMedium)) else mat for mat in medium_list
        ]
        # merge geometries
        merged_geos = merging_geometries_on_plane(geometry_list, plane, medium_list)

        return merged_geos

    def _corners_and_convexity(
        self,
        normal_axis: Axis,
        coord: float,
        structure_list: list[Structure],
        ravel: bool,
    ) -> tuple[ArrayFloat2D, ArrayFloat1D]:
        """On a 2D plane specified by axis = `normal_axis` and coordinate `coord`, find out corners of merged
        geometries made of PEC.


        Parameters
        ----------
        normal_axis : Axis
            Axis normal to the 2D plane.
        coord : float
            Position of plane along the normal axis.
        structure_list : List[Structure]
            List of structures present in simulation.
        ravel : bool
            Whether to put the resulting corners in a single list or per polygon.

        Returns
        -------
        Tuple[ArrayFloat2D, ArrayFloat1D]
            Corner coordinates and their convexity.
        """

        # merge geometries
        merged_geos = self._merged_pec_on_plane(
            normal_axis=normal_axis, coord=coord, structure_list=structure_list
        )

        # corner finder
        corner_list = []
        convexity_list = []
        for mat, shapes in merged_geos:
            if self.medium != "all" and mat.is_pec != (self.medium == "metal"):
                continue
            polygon_list = ClipOperation.to_polygon_list(shapes)
            for poly in polygon_list:
                poly = poly.normalize().buffer(0)
                if self.distance_threshold is not None:
                    poly = poly.simplify(self.distance_threshold, preserve_topology=True)
                corners_xy, corners_convexity = self._filter_collinear_vertices(
                    list(poly.exterior.coords)
                )
                corner_list.append(corners_xy)
                convexity_list.append(corners_convexity)
                # in case the polygon has holes
                for poly_inner in poly.interiors:
                    corners_xy, corners_convexity = self._filter_collinear_vertices(
                        list(poly_inner.coords)
                    )
                    corner_list.append(corners_xy)
                    convexity_list.append(corners_convexity)

        if ravel and len(corner_list) > 0:
            corner_list = np.concatenate(corner_list)
            convexity_list = np.concatenate(convexity_list)

        return corner_list, convexity_list

    def corners(
        self,
        normal_axis: Axis,
        coord: float,
        structure_list: list[Structure],
    ) -> ArrayFloat2D:
        """On a 2D plane specified by axis = `normal_axis` and coordinate `coord`, find out corners of merged
        geometries made of `medium`.


        Parameters
        ----------
        normal_axis : Axis
            Axis normal to the 2D plane.
        coord : float
            Position of plane along the normal axis.
        structure_list : List[Structure]
            List of structures present in simulation.

        Returns
        -------
        ArrayFloat2D
            Corner coordinates.
        """

        corner_list, _ = self._corners_and_convexity(
            normal_axis=normal_axis, coord=coord, structure_list=structure_list, ravel=True
        )
        return corner_list

    def _filter_collinear_vertices(
        self, vertices: ArrayFloat2D
    ) -> tuple[ArrayFloat2D, ArrayFloat1D]:
        """Filter collinear vertices of a polygon, and return corners locations and their convexity.

        Parameters
        ----------
        vertices : ArrayFloat2D
            Polygon vertices from shapely.Polygon. The last vertex is identical to the 1st
            vertex to make a valid polygon.

        Returns
        -------
        ArrayFloat2D
            Corner coordinates.
        ArrayFloat1D
            Convexity of corners: True for outer corners, False for inner corners.
        """

        def normalize(v):
            return v / np.linalg.norm(v, axis=-1)[:, np.newaxis]

        # drop the last vertex, which is identical to the 1st one.
        vs_orig = np.array(vertices[:-1])
        # compute unit vector to next and previous vertex
        vs_next = np.roll(vs_orig, axis=0, shift=-1)
        vs_previous = np.roll(vs_orig, axis=0, shift=+1)
        unit_next = normalize(vs_next - vs_orig)
        unit_previous = normalize(vs_previous - vs_orig)
        # angle
        inner_product = np.sum(unit_next * unit_previous, axis=-1)
        inner_product = np.where(inner_product > 1, 1, inner_product)
        inner_product = np.where(inner_product < -1, -1, inner_product)
        angle = np.arccos(inner_product)
        num_vs = len(vs_orig)
        cross_product = np.cross(
            np.hstack([unit_next, np.zeros((num_vs, 1))]),
            np.hstack([unit_previous, np.zeros((num_vs, 1))]),
            axis=-1,
        )
        convexity = cross_product[:, 2] < 0
        ind_filter = angle <= np.pi - self.angle_threshold
        return vs_orig[ind_filter], convexity[ind_filter]
