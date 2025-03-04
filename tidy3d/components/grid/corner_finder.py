"""Find corners of structures on a 2D plane."""

from typing import List, Literal, Optional

import numpy as np
import pydantic.v1 as pd

from ...constants import inf
from ..base import Tidy3dBaseModel
from ..geometry.base import Box, ClipOperation
from ..geometry.utils import merging_geometries_on_plane
from ..medium import PEC, LossyMetalMedium
from ..structure import Structure
from ..types import ArrayFloat2D, Axis

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

    def corners(
        self,
        normal_axis: Axis,
        coord: float,
        structure_list: List[Structure],
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

        # Construct plane
        center = [0, 0, 0]
        size = [inf, inf, inf]
        center[normal_axis] = coord
        size[normal_axis] = 0
        plane = Box(center=center, size=size)

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

        # corner finder
        corner_list = []
        for mat, shapes in merged_geos:
            if self.medium != "all" and mat.is_pec != (self.medium == "metal"):
                continue
            polygon_list = ClipOperation.to_polygon_list(shapes)
            for poly in polygon_list:
                poly = poly.normalize().buffer(0)
                if self.distance_threshold is not None:
                    poly = poly.simplify(self.distance_threshold, preserve_topology=True)
                corner_list.append(self._filter_collinear_vertices(list(poly.exterior.coords)))
                # in case the polygon has holes
                for poly_inner in poly.interiors:
                    corner_list.append(self._filter_collinear_vertices(list(poly_inner.coords)))

        if len(corner_list) > 0:
            corner_list = np.concatenate(corner_list)
        return corner_list

    def _filter_collinear_vertices(self, vertices: ArrayFloat2D) -> ArrayFloat2D:
        """Filter collinear vertices of a polygon, and return corners.

        Parameters
        ----------
        vertices : ArrayFloat2D
            Polygon vertices from shapely.Polygon. The last vertex is identical to the 1st
            vertex to make a valid polygon.

        Returns
        -------
        ArrayFloat2D
            Corner coordinates.
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
        ind_filter = angle <= np.pi - self.angle_threshold
        return vs_orig[ind_filter]
