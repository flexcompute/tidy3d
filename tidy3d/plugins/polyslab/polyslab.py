""" Divide a complex polyslab where self-intersecting polygon can occur during extrusion."""

from typing import List, Tuple
from math import isclose

import pydantic
from shapely.geometry import Polygon

from ...log import log
from ...components.geometry import PolySlab, GeometryGroup
from ...components.medium import MediumType
from ...components.structure import Structure
from ...components.types import Axis, PlanePosition
from ...constants import fp_eps


# Warn for too many divided polyslabs
_WARN_MAX_NUM_DIVISION = 100


class ComplexPolySlab(PolySlab):
    """Interface for dividing a complex polyslab where self-intersecting polygon can
    occur during extrusion.

    Example
    -------
    >>> vertices = ((0, 0), (1, 0), (1, 1), (0, 1), (0, 0.9), (0, 0.11))
    >>> p = ComplexPolySlab(vertices=vertices, axis=2, slab_bounds=(0, 1), sidewall_angle=0.785)
    >>> # To obtain the divided polyslabs, there are two approaches:
    >>> # 1) a list of divided polyslabs
    >>> geo_list = p.sub_polyslabs
    >>> # 2) geometry group containing the divided polyslabs
    >>> geo_group = p.geometry_group
    >>> # Or directly obtain the structure with a user-specified medium
    >>> mat = td.Medium(permittivity=2)
    >>> structure = p.to_structure(mat)

    Note
    ----
    This version is limited to neighboring vertex-vertex crossing type of
    self-intersecting events. Extension to cover all types of self-intersecting
    events is expected in the future.

    The algorithm is as follows (for the convenience of illustration,
    let's consider the reference plane to lie at the bottom of the polyslab),

    1. Starting from the reference plane, find out the critical
    extrusion distance for the first vertices degeneracy
    event when marching towards the top of the polyslab;

    2. Construct a sub-polyslab whose base is the polygon at
    the reference plane and height to be the critical
    extrusion distance;

    3. At the critical extrusion distance, constructing a new polygon
    that keeps only one of the degenerate vertices;

    4. Set the reference plane to the position of the new polygon,
    and  repeating 1-3 to construct sub-polyslabs until reaching
    the top of the polyslab, or all vertices collapsed into a 1D curve
    or a 0D point.
    """

    @pydantic.validator("vertices", always=True)
    def no_self_intersecting_polygon_during_extrusion(cls, val, values):
        """Turn off the validation for this class."""
        return val

    @classmethod
    def from_gds(  # pylint:disable=too-many-arguments
        cls,
        gds_cell,
        axis: Axis,
        slab_bounds: Tuple[float, float],
        gds_layer: int,
        gds_dtype: int = None,
        gds_scale: pydantic.PositiveFloat = 1.0,
        dilation: float = 0.0,
        sidewall_angle: float = 0,
        reference_plane: PlanePosition = "middle",
    ) -> List[PolySlab]:
        """Import :class:`.PolySlab` from a ``gdstk.Cell``.

        Parameters
        ----------
        gds_cell : gdstk.Cell
            ``gdstk.Cell`` containing 2D geometric data.
        axis : int
            Integer index into the polygon's slab axis. (0,1,2) -> (x,y,z).
        slab_bounds: Tuple[float, float]
            Minimum and maximum positions of the slab along ``axis``.
        gds_layer : int
            Layer index in the ``gds_cell``.
        gds_dtype : int = None
            Data-type index in the ``gds_cell``.
            If ``None``, imports all data for this layer into the returned list.
        gds_scale : float = 1.0
            Length scale used in GDS file in units of MICROMETER.
            For example, if gds file uses nanometers, set ``gds_scale=1e-3``.
            Must be positive.
        dilation : float = 0.0
            Dilation of the polygon in the base by shifting each edge along its
            normal outwards direction by a distance;
            a negative value corresponds to erosion.
        sidewall_angle : float = 0
            Angle of the sidewall.
            ``sidewall_angle=0`` (default) specifies vertical wall,
            while ``0<sidewall_angle<np.pi/2`` for the base to be larger than the top.
        reference_plane : PlanePosition = "middle"
            The position of the GDS layer. It can be at the ``bottom``, ``middle``,
            or ``top`` of the PolySlab. E.g. if ``axis=1``, ``bottom`` refers to the
            negative side of y-axis, and ``top`` refers to the positive side of y-axis.

        Returns
        -------
        List[:class:`.PolySlab`]
            List of :class:`.PolySlab` objects sharing ``axis`` and  slab bound properties.
        """

        # TODO: change for 2.0
        # handle reference plane kwarg
        all_vertices = cls._load_gds_vertices(gds_cell, gds_layer, gds_dtype, gds_scale)
        polyslabs = [
            cls(
                vertices=verts,
                axis=axis,
                slab_bounds=slab_bounds,
                dilation=dilation,
                sidewall_angle=sidewall_angle,
                reference_plane=reference_plane,
            )
            for verts in all_vertices
        ]
        return [sub_poly for sub_polys in polyslabs for sub_poly in sub_polys.sub_polyslabs]

    def to_structure(self, medium: MediumType) -> Structure:
        """Construct a structure containing a user-specified medium
        and a GeometryGroup made of all the divided PolySlabs from this object.

        Parameters
        ----------
        medium : :class:`.MediumType`
            Medium for the complex polyslab.

        Returns
        -------
        :class:`.Structure`
            The structure containing all divided polyslabs made of a user-specified
            medium.
        """
        return Structure(geometry=self.geometry_group, medium=medium)

    @property
    def geometry_group(self) -> GeometryGroup:
        """Divide a complex polyslab into a list of simple polyslabs, which
        are assembled into a :class:`.GeometryGroup`.

        Returns
        -------
        :class:`.GeometryGroup`
            GeometryGroup for a list of simple polyslabs divided from the complex
            polyslab.
        """
        return GeometryGroup(geometries=self.sub_polyslabs)

    @property
    def sub_polyslabs(self) -> List[PolySlab]:
        """Divide a complex polyslab into a list of simple polyslabs.
        Only neighboring vertex-vertex crossing events are treated in this
        version.

        Returns
        -------
        List[PolySlab]
            A list of simple polyslabs.
        """
        sub_polyslab_list = []
        num_division_count = 0
        # initialize sub-polyslab parameters
        sub_polyslab_dict = self.dict(exclude={"type"}).copy()
        if isclose(self.sidewall_angle, 0):
            return [PolySlab.parse_obj(sub_polyslab_dict)]

        sub_polyslab_dict.update({"dilation": 0})  # dilation accounted in setup
        # initalize offset distance
        offset_distance = 0

        for dist_val in self._dilation_length:
            dist_now = 0.0
            vertices_now = self.reference_polygon

            # constructing sub-polyslabs until reaching the base/top
            while not isclose(dist_now, dist_val):
                # bounds for sub-polyslabs assuming no self-intersection
                slab_bounds = [
                    self._dilation_value_at_reference_to_coord(dist_now),
                    self._dilation_value_at_reference_to_coord(dist_val),
                ]
                # 1) find out any vertices touching events between the current
                # position to the base/top
                max_dist = PolySlab._neighbor_vertices_crossing_detection(
                    vertices_now, dist_val - dist_now
                )

                # vertices touching events captured, update bounds for sub-polyslab
                if max_dist is not None:
                    # max_dist doesn't have sign, so construct signed offset distance
                    offset_distance = max_dist * dist_val / abs(dist_val)
                    slab_bounds[1] = self._dilation_value_at_reference_to_coord(
                        dist_now + offset_distance
                    )

                # 2) construct sub-polyslab
                slab_bounds.sort()  # for reference_plane=top/bottom, bounds need to be ordered
                # direction of marching
                reference_plane = "bottom" if dist_val / self._tanq < 0 else "top"
                sub_polyslab_dict.update(
                    dict(
                        slab_bounds=tuple(slab_bounds),
                        vertices=vertices_now,
                        reference_plane=reference_plane,
                    )
                )
                sub_polyslab_list.append(PolySlab.parse_obj(sub_polyslab_dict))

                # Now Step 3
                if max_dist is None:
                    break
                dist_now += offset_distance
                # new polygon vertices where collapsing vertices are removed but keep one
                vertices_now = PolySlab._shift_vertices(vertices_now, offset_distance)[0]
                vertices_now = PolySlab._remove_duplicate_vertices(vertices_now)
                # all vertices collapse
                if len(vertices_now) < 3:
                    break
                # polygon collapse into 1D
                if Polygon(vertices_now).buffer(0).area < fp_eps:
                    break
                vertices_now = PolySlab._orient(vertices_now)
                num_division_count += 1

        if num_division_count > _WARN_MAX_NUM_DIVISION:
            log.warning(
                "Two many self-intersecting events: "
                f"The polyslab has been divided into {num_division_count} polyslabs; "
                f"any more than around {_WARN_MAX_NUM_DIVISION} may slow down the simulation."
            )

        return sub_polyslab_list

    @property
    def _dilation_length(self) -> List[float]:
        """dilation length from reference plane to the top/bottom of the polyslab."""

        # for "bottom", only needs to compute the offset length to the top
        dist = [self._extrusion_length_to_offset_distance(self.length_axis)]
        # reverse the dilation value if the reference plane is on the top
        if self.reference_plane == "top":
            dist = [-dist[0]]
        # for middle, both directions
        elif self.reference_plane == "middle":
            dist = [dist[0] / 2, -dist[0] / 2]
        return dist

    def _dilation_value_at_reference_to_coord(self, dilation: float) -> float:
        """Compute the coordinate based on the dilation value to the reference plane."""

        z_coord = -dilation / self._tanq + self.slab_bounds[0]
        if self.reference_plane == "middle":
            return z_coord + self.length_axis / 2
        if self.reference_plane == "top":
            return z_coord + self.length_axis
        # bottom case
        return z_coord
