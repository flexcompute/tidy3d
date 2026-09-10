"""Abstract and primitive geometry base classes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import autograd.numpy as np
from pydantic import Field

from tidy3d.components.autograd import TracedSize
from tidy3d.components.autograd.path_utils import (
    indexed_traced_paths,
    traced_paths,
)
from tidy3d.components.autograd.types import PathType
from tidy3d.components.base import cached_property
from tidy3d.components.geometry.float_utils import increment_float
from tidy3d.components.types import Axis, Coordinate, MatrixReal4x4  # noqa: TC
from tidy3d.components.viz import (
    ARROW_LENGTH,
)
from tidy3d.constants import MICROMETER, fp_eps, inf
from tidy3d.exceptions import (
    AdjointError,
    SetupError,
    ValidationError,
)
from tidy3d.packaging import verify_packages_import

from .abstract_primitives import Centered, SimplePlaneIntersection
from .core import _BOX_FACES, Geometry

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    import pydantic
    from matplotlib.backend_bases import Event
    from matplotlib.patches import FancyArrowPatch
    from numpy.typing import ArrayLike, NDArray
    from typing_extensions import Self

    from tidy3d.components.autograd import AutogradFieldMap
    from tidy3d.components.autograd.derivative_utils import DerivativeInfo
    from tidy3d.components.types import (
        Ax,
        Bound,
        Shapely,
        Size,
    )
    from tidy3d.em.translate.sample_sets import SamplingContext, SurfaceSampleSet


class Box(SimplePlaneIntersection, Centered):
    """Rectangular prism.
       Also base class for :class:`.Simulation`, :class:`Monitor`, and :class:`Source`.

    Example
    -------
    >>> b = Box(center=(1,2,3), size=(2,2,2))
    """

    _traced_supported_paths: ClassVar[tuple[PathType, ...]] = traced_paths(
        "center",
        "size",
        *indexed_traced_paths("center", 3),
        *indexed_traced_paths("size", 3),
    )

    size: TracedSize = Field(
        title="Size",
        description="Size in x, y, and z directions.",
        json_schema_extra={"units": MICROMETER},
    )

    @classmethod
    def from_bounds(cls, rmin: Coordinate, rmax: Coordinate, **kwargs: Any) -> Self:
        """Constructs a :class:`~tidy3d.Box` from minimum and maximum coordinate bounds

        Parameters
        ----------
        rmin : tuple[float, float, float]
            (x, y, z) coordinate of the minimum values.
        rmax : tuple[float, float, float]
            (x, y, z) coordinate of the maximum values.

        Example
        -------
        >>> b = Box.from_bounds(rmin=(-1, -2, -3), rmax=(3, 2, 1))
        """

        center = tuple(cls._get_center(pt_min, pt_max) for pt_min, pt_max in zip(rmin, rmax))
        size = tuple((pt_max - pt_min) for pt_min, pt_max in zip(rmin, rmax))
        return cls(center=center, size=size, **kwargs)

    @cached_property
    def _normal_axis(self) -> Axis:
        """Axis normal to the Box. Errors if box is not planar."""
        if self.size.count(0.0) != 1:
            raise ValidationError(
                f"Tried to get 'normal_axis' of 'Box' that is not planar. Given 'size={self.size}.'"
            )
        return self.size.index(0.0)

    @staticmethod
    def _surface_keys(size: Size) -> tuple[list[str], set[int]]:
        """Return the canonical surface keys and indices dropped for infinite dimensions."""
        surface_keys = [coord + direction for coord in "xyz" for direction in "-+"]
        del_idx = {
            2 * idx + offset for idx, _size in enumerate(size) if _size == inf for offset in (0, 1)
        }
        surface_keys = [key for idx, key in enumerate(surface_keys) if idx not in del_idx]
        return surface_keys, del_idx

    @classmethod
    def surfaces(cls, size: Size, center: Coordinate, **kwargs: Any) -> list[Self]:
        """Returns a list of 6 :class:`~tidy3d.Box` instances corresponding to each surface of a 3D volume.
        The output surfaces are stored in the order [x-, x+, y-, y+, z-, z+], where x, y, and z
        denote which axis is perpendicular to that surface, while "-" and "+" denote the direction
        of the normal vector of that surface. If a name is provided, each output surface's name
        will be that of the provided name appended with the above symbols. E.g., if the provided
        name is "box", the x+ surfaces's name will be "box_x+".

        Parameters
        ----------
        size : tuple[float, float, float]
            Size of object in x, y, and z directions.
        center : tuple[float, float, float]
            Center of object in x, y, and z.

        Example
        -------
        >>> b = Box.surfaces(size=(1, 2, 3), center=(3, 2, 1))
        """

        if any(s == 0.0 for s in size):
            raise SetupError(
                "Can't generate surfaces for the given object because it has zero volume."
            )

        bounds = Box(center=center, size=size).bounds

        # Set up geometry data and names for each surface:
        centers = [list(center) for _ in range(6)]
        sizes = [list(size) for _ in range(6)]

        surface_index = 0
        for dim_index in range(3):
            for min_max_index in range(2):
                new_center = centers[surface_index]
                new_size = sizes[surface_index]

                new_center[dim_index] = bounds[min_max_index][dim_index]
                new_size[dim_index] = 0.0

                centers[surface_index] = new_center
                sizes[surface_index] = new_size

                surface_index += 1

        surface_keys, del_idx = cls._surface_keys(size)
        name_base = kwargs.pop("name", "")
        kwargs.pop("normal_dir", None)

        def del_items(items: Iterable, indices: set[int]) -> list:
            """Delete list items at indices."""
            return [i for j, i in enumerate(items) if j not in indices]

        centers = del_items(centers, del_idx)
        sizes = del_items(sizes, del_idx)
        names = [name_base + "_" + surface_key for surface_key in surface_keys]
        normal_dirs = [surface_key[-1] for surface_key in surface_keys]

        surfaces = []
        for _cent, _size, _name, _normal_dir in zip(centers, sizes, names, normal_dirs):
            if "normal_dir" in cls.model_fields:
                kwargs["normal_dir"] = _normal_dir

            if "name" in cls.model_fields:
                kwargs["name"] = _name

            surface = cls(center=_cent, size=_size, **kwargs)
            surfaces.append(surface)

        return surfaces

    @classmethod
    def surfaces_with_exclusion(cls, size: Size, center: Coordinate, **kwargs: Any) -> list[Self]:
        """Returns a list of 6 :class:`~tidy3d.Box` instances corresponding to each surface of a 3D volume.
        The output surfaces are stored in the order [x-, x+, y-, y+, z-, z+], where x, y, and z
        denote which axis is perpendicular to that surface, while "-" and "+" denote the direction
        of the normal vector of that surface. If a name is provided, each output surface's name
        will be that of the provided name appended with the above symbols. E.g., if the provided
        name is "box", the x+ surfaces's name will be "box_x+". If ``kwargs`` contains an
        ``exclude_surfaces`` parameter, the returned list of surfaces will not include the excluded
        surfaces. Otherwise, the behavior is identical to that of ``surfaces()``.

        Parameters
        ----------
        size : tuple[float, float, float]
            Size of object in x, y, and z directions.
        center : tuple[float, float, float]
            Center of object in x, y, and z.

        Example
        -------
        >>> b = Box.surfaces_with_exclusion(
        ...     size=(1, 2, 3), center=(3, 2, 1), exclude_surfaces=["x-"]
        ... )
        """
        exclude_surfaces = kwargs.pop("exclude_surfaces", None)
        surfaces = cls.surfaces(size=size, center=center, **kwargs)
        if exclude_surfaces:
            surface_keys, _ = cls._surface_keys(size)
            exclude_surfaces = set(exclude_surfaces)
            surfaces = [
                surf
                for surf, surface_key in zip(surfaces, surface_keys)
                if surface_key not in exclude_surfaces
            ]
        return surfaces

    @verify_packages_import(["trimesh"])
    def _do_intersections_tilted_plane(
        self,
        normal: Coordinate,
        origin: Coordinate,
        to_2D: MatrixReal4x4,
        quad_segs: int | None = None,
    ) -> list[Shapely]:
        """Return a list of shapely geometries at the plane specified by normal and origin.

        Parameters
        ----------
        normal : Coordinate
            Vector defining the normal direction to the plane.
        origin : Coordinate
            Vector defining the plane origin.
        to_2D : MatrixReal4x4
            Transformation matrix to apply to resulting shapes.
        quad_segs : Optional[int] = None
            Number of segments used to discretize circular shapes. Not used for Box geometry.

        Returns
        -------
        list[shapely.geometry.base.BaseGeometry]
            List of 2D shapes that intersect plane.
            For more details refer to
            `Shapely's Documentation <https://shapely.readthedocs.io/en/stable/project.html>`_.
        """
        import trimesh

        (x0, y0, z0), (x1, y1, z1) = self.bounds
        vertices = [
            (x0, y0, z0),  # 0
            (x0, y0, z1),  # 1
            (x0, y1, z0),  # 2
            (x0, y1, z1),  # 3
            (x1, y0, z0),  # 4
            (x1, y0, z1),  # 5
            (x1, y1, z0),  # 6
            (x1, y1, z1),  # 7
        ]
        faces = [
            (0, 1, 3, 2),  # -x
            (4, 6, 7, 5),  # +x
            (0, 4, 5, 1),  # -y
            (2, 3, 7, 6),  # +y
            (0, 2, 6, 4),  # -z
            (1, 5, 7, 3),  # +z
        ]
        mesh = trimesh.Trimesh(vertices, faces)

        section = mesh.section(plane_origin=origin, plane_normal=normal)
        if section is None:
            return []
        path, _ = section.to_2D(to_2D=to_2D)
        return path.polygons_full

    def intersections_plane(
        self,
        x: float | None = None,
        y: float | None = None,
        z: float | None = None,
        cleanup: bool = True,
        quad_segs: int | None = None,
        section_tolerance_2d: bool = False,
    ) -> list[Shapely]:
        """Returns shapely geometry at plane specified by one non None value of x,y,z.

        Parameters
        ----------
        x : float = None
            Position of plane in x direction, only one of x,y,z can be specified to define plane.
        y : float = None
            Position of plane in y direction, only one of x,y,z can be specified to define plane.
        z : float = None
            Position of plane in z direction, only one of x,y,z can be specified to define plane.
        cleanup : bool = True
            If True, removes extremely small features from each polygon's boundary.
        quad_segs : Optional[int] = None
            Number of segments used to discretize circular shapes. Not used for Box geometry.
        section_tolerance_2d : bool = False
            See :meth:`Geometry.intersections_plane`.

        Returns
        -------
        list[shapely.geometry.base.BaseGeometry]
            List of 2D shapes that intersect plane.
            For more details refer to
            `Shapely's Documentation <https://shapely.readthedocs.io/en/stable/project.html>`_.
        """
        axis, position = self.parse_xyz_kwargs(x=x, y=y, z=z)
        use_2d_tolerance = section_tolerance_2d and np.isclose(
            self.size[axis], 0.0, rtol=fp_eps, atol=fp_eps
        )
        if not self.intersects_axis_position(axis, position, section_tolerance_2d=use_2d_tolerance):
            return []
        z0, (x0, y0) = self.pop_axis(self.center, axis=axis)
        Lz, (Lx, Ly) = self.pop_axis(self.size, axis=axis)
        if use_2d_tolerance and np.isclose(position, z0, rtol=fp_eps, atol=fp_eps):
            position = z0
        dz = np.abs(z0 - position)
        if dz > Lz / 2 + fp_eps:
            return []

        minx = x0 - Lx / 2
        miny = y0 - Ly / 2
        maxx = x0 + Lx / 2
        maxy = y0 + Ly / 2

        # handle case where the box vertices are identical
        if np.isclose(minx, maxx) and np.isclose(miny, maxy):
            return [self.make_shapely_point(minx, miny)]

        return [self.make_shapely_box(minx, miny, maxx, maxy)]

    def inside(self, x: NDArray[float], y: NDArray[float], z: NDArray[float]) -> NDArray[bool]:
        """For input arrays ``x``, ``y``, ``z`` of arbitrary but identical shape, return an array
        with the same shape which is ``True`` for every point in zip(x, y, z) that is inside the
        volume of the :class:`~tidy3d.Geometry`, and ``False`` otherwise.

        Parameters
        ----------
        x : np.ndarray[float]
            Array of point positions in x direction.
        y : np.ndarray[float]
            Array of point positions in y direction.
        z : np.ndarray[float]
            Array of point positions in z direction.

        Returns
        -------
        np.ndarray[bool]
            ``True`` for every point that is inside the geometry.
        """
        self._ensure_equal_shape(x, y, z)
        x0, y0, z0 = self.center
        Lx, Ly, Lz = self.size
        dist_x = np.abs(x - x0)
        dist_y = np.abs(y - y0)
        dist_z = np.abs(z - z0)
        return (dist_x <= Lx / 2) * (dist_y <= Ly / 2) * (dist_z <= Lz / 2)

    def intersections_with(
        self,
        other: Geometry,
        cleanup: bool = True,
        quad_segs: int | None = None,
        section_tolerance_2d: bool = False,
    ) -> list[Shapely]:
        """Returns list of shapely geometries representing the intersections of the geometry with
        this 2D box.

        Parameters
        ----------
        other : :class:`~tidy3d.Geometry`
            Geometry to intersect with.
        cleanup : bool = True
            If True, removes extremely small features from each polygon's boundary.
        quad_segs : Optional[int] = None
            Number of segments used to discretize circular shapes. If ``None``, uses
            high-quality visualization settings.
        section_tolerance_2d : bool = False
            See :meth:`Geometry.intersections_plane`.

        Returns
        -------
        list[shapely.geometry.base.BaseGeometry]
            List of 2D shapes that intersect this 2D box.
            For more details refer to
            `Shapely's Documentation <https://shapely.readthedocs.io/en/stable/project.html>`_.
        """

        # Verify 2D
        if self.size.count(0.0) != 1:
            raise ValidationError(
                "Intersections with other geometry are only calculated from a 2D box."
            )

        # Don't bother if the geometry doesn't intersect the self at all.
        # Plotting opts into the section-aware precheck so zero-thickness 2D shapes
        # that are only off by ``fp_eps`` still reach ``intersections_plane()`` below.
        normal_ind = self.size.index(0.0)
        if section_tolerance_2d:
            if not other.intersects_axis_position(
                normal_ind,
                self.center[normal_ind],
                section_tolerance_2d=True,
            ):
                return []
        elif not other.intersects(self):
            return []

        # get list of Shapely shapes that intersect at the self
        dim = "xyz"[normal_ind]
        pos = self.center[normal_ind]
        xyz_kwargs = {dim: pos}
        shapes_plane = other.intersections_plane(
            cleanup=cleanup,
            quad_segs=quad_segs,
            section_tolerance_2d=section_tolerance_2d,
            **xyz_kwargs,
        )

        # intersect all shapes with the input self
        bs_min, bs_max = (self.pop_axis(bounds, axis=normal_ind)[1] for bounds in self.bounds)

        shapely_box = self.make_shapely_box(bs_min[0], bs_min[1], bs_max[0], bs_max[1])
        shapely_box = Geometry.evaluate_inf_shape(shapely_box)
        return [Geometry.evaluate_inf_shape(shape) & shapely_box for shape in shapes_plane]

    def slightly_enlarged_copy(self) -> Box:
        """Box size slightly enlarged around machine precision."""
        size = [increment_float(orig_length, 1) for orig_length in self.size]
        return self.updated_copy(size=size)

    def padded_copy(
        self,
        x: tuple[pydantic.NonNegativeFloat, pydantic.NonNegativeFloat] | None = None,
        y: tuple[pydantic.NonNegativeFloat, pydantic.NonNegativeFloat] | None = None,
        z: tuple[pydantic.NonNegativeFloat, pydantic.NonNegativeFloat] | None = None,
    ) -> Box:
        """Created a padded copy of a :class:`~tidy3d.Box` instance.

        Parameters
        ----------
        x : Optional[tuple[pydantic.NonNegativeFloat, pydantic.NonNegativeFloat]] = None
            Padding sizes at the left and right boundaries of the box along x-axis.
        y : Optional[tuple[pydantic.NonNegativeFloat, pydantic.NonNegativeFloat]] = None
            Padding sizes at the left and right boundaries of the box along y-axis.
        z : Optional[tuple[pydantic.NonNegativeFloat, pydantic.NonNegativeFloat]] = None
            Padding sizes at the left and right boundaries of the box along z-axis.

        Returns
        -------
        Box
            Padded instance of :class:`~tidy3d.Box`.
        """

        # Validate that padding values are non-negative
        for axis_name, axis_padding in zip(("x", "y", "z"), (x, y, z)):
            if axis_padding is not None:
                if not isinstance(axis_padding, (tuple, list)) or len(axis_padding) != 2:
                    raise ValueError(f"Padding for {axis_name}-axis must be a tuple of two values.")
                if any(p < 0 for p in axis_padding):
                    raise ValueError(
                        f"Padding values for {axis_name}-axis must be non-negative. Got {axis_padding}."
                    )

        rmin, rmax = self.bounds

        def bound_array(arrs: ArrayLike, idx: int) -> NDArray:
            return np.array([(a[idx] if a is not None else 0) for a in arrs])

        # parse padding sizes for simulation
        drmin = bound_array((x, y, z), 0)
        drmax = bound_array((x, y, z), 1)

        rmin = np.array(rmin) - drmin
        rmax = np.array(rmax) + drmax

        return Box.from_bounds(rmin=rmin, rmax=rmax)

    @cached_property
    def bounds(self) -> Bound:
        """Returns bounding box min and max coordinates.

        Returns
        -------
        tuple[float, float, float], tuple[float, float float]
            Min and max bounds packaged as ``(minx, miny, minz), (maxx, maxy, maxz)``.
        """
        size = self.size
        center = self.center
        coord_min = tuple(c - s / 2 for (s, c) in zip(size, center))
        coord_max = tuple(c + s / 2 for (s, c) in zip(size, center))
        return (coord_min, coord_max)

    @cached_property
    def geometry(self) -> Box:
        """:class:`~tidy3d.Box` representation of self (used for subclasses of Box).

        Returns
        -------
        :class:`~tidy3d.Box`
            Instance of :class:`~tidy3d.Box` representing self's geometry.
        """
        return Box(center=self.center, size=self.size)

    @cached_property
    def zero_dims(self) -> list[Axis]:
        """A list of axes along which the :class:`~tidy3d.Box` is zero-sized."""
        return [dim for dim, size in enumerate(self.size) if size == 0]

    @cached_property
    def _normal_2dmaterial(self) -> Axis:
        """Get the normal to the given geometry, checking that it is a 2D geometry."""
        if np.count_nonzero(self.size) != 2:
            raise ValidationError(
                "'Medium2D' requires exactly one of the 'Box' dimensions to have size zero."
            )
        return self.size.index(0)

    def _update_from_bounds(self, bounds: tuple[float, float], axis: Axis) -> Box:
        """Returns an updated geometry which has been transformed to fit within ``bounds``
        along the ``axis`` direction."""
        new_center = list(self.center)
        new_center[axis] = (bounds[0] + bounds[1]) / 2
        new_size = list(self.size)
        new_size[axis] = bounds[1] - bounds[0]
        return self.updated_copy(center=tuple(new_center), size=tuple(new_size))

    def _plot_arrow(
        self,
        direction: tuple[float, float, float],
        x: float | None = None,
        y: float | None = None,
        z: float | None = None,
        color: str | None = None,
        alpha: float | None = None,
        bend_radius: float | None = None,
        bend_axis: Axis = None,
        both_dirs: bool = False,
        ax: Ax = None,
        arrow_base: Coordinate = None,
    ) -> Ax:
        """Adds an arrow to the axis if with options if certain conditions met.

        Parameters
        ----------
        direction: tuple[float, float, float]
            Normalized vector describing the arrow direction.
        x : float = None
            Position of plotting plane in x direction.
        y : float = None
            Position of plotting plane in y direction.
        z : float = None
            Position of plotting plane in z direction.
        color : str = None
            Color of the arrow.
        alpha : float = None
            Opacity of the arrow (0, 1)
        bend_radius : float = None
            Radius of curvature for this arrow.
        bend_axis : Axis = None
            Axis of curvature of ``bend_radius``.
        both_dirs : bool = False
            If True, plots an arrow pointing in direction and one in -direction.
        arrow_base : :class:`.Coordinate` = None
            Custom base of the arrow. Uses the geometry's center if not provided.

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The matplotlib axes with the arrow added.
        """
        from matplotlib import patches

        from tidy3d.components.viz.styles import arrow_style

        plot_axis, _ = self.parse_xyz_kwargs(x=x, y=y, z=z)
        _, (dx, dy) = self.pop_axis(direction, axis=plot_axis)

        # conditions to check to determine whether to plot arrow, taking into account the
        # possibility of a custom arrow base
        arrow_intersecting_plane = (
            len(self.intersections_plane(x=x, y=y, z=z, section_tolerance_2d=True)) > 0
        )
        center = self.center
        if arrow_base:
            arrow_intersecting_plane = arrow_intersecting_plane and any(
                a == b for a, b in zip(arrow_base, [x, y, z])
            )
            center = arrow_base

        _, (dx, dy) = self.pop_axis(direction, axis=plot_axis)
        components_in_plane = any(not np.isclose(component, 0) for component in (dx, dy))

        # plot if arrow in plotting plane and some non-zero component can be displayed.
        if arrow_intersecting_plane and components_in_plane:
            _, (x0, y0) = self.pop_axis(center, axis=plot_axis)

            # Reasonable value for temporary arrow size.  The correct size and direction
            # have to be calculated after all transforms have been set.  That is why we
            # use a callback to do these calculations only at the drawing phase.
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
            v_x = (xmax - xmin) / 10
            v_y = (ymax - ymin) / 10

            directions = (1.0, -1.0) if both_dirs else (1.0,)
            for sign in directions:
                arrow = patches.FancyArrowPatch(
                    (x0, y0),
                    (x0 + v_x, y0 + v_y),
                    arrowstyle=arrow_style(),
                    color=color,
                    alpha=alpha,
                    zorder=np.inf,
                )
                # Don't draw this arrow until it's been reshaped
                arrow.set_visible(False)

                callback = self._arrow_shape_cb(
                    arrow, (x0, y0), (dx, dy), sign, bend_radius if bend_axis == plot_axis else None
                )
                callback_id = ax.figure.canvas.mpl_connect("draw_event", callback)

                # Store a reference to the callback because mpl_connect does not.
                arrow.set_shape_cb = (callback_id, callback)

                ax.add_patch(arrow)

        return ax

    @staticmethod
    def _arrow_shape_cb(
        arrow: FancyArrowPatch,
        pos: tuple[float, float],
        direction: ArrayLike,
        sign: float,
        bend_radius: float | None,
    ) -> Callable[[Event], None]:
        from matplotlib import patches

        def _cb(event: Event) -> None:
            # We only want to set the shape once, so we disconnect ourselves
            event.canvas.mpl_disconnect(arrow.set_shape_cb[0])

            transform = arrow.axes.transData.transform
            scale_x = transform((1, 0))[0] - transform((0, 0))[0]
            scale_y = transform((0, 1))[1] - transform((0, 0))[1]
            scale = max(scale_x, scale_y)  # <-- Hack: This is a somewhat arbitrary choice.
            arrow_length = ARROW_LENGTH * event.canvas.figure.get_dpi() / scale

            if bend_radius:
                v_norm = (direction[0] ** 2 + direction[1] ** 2) ** 0.5
                vx_norm = direction[0] / v_norm
                vy_norm = direction[1] / v_norm
                bend_angle = -sign * arrow_length / bend_radius
                t_x = 1 - np.cos(bend_angle)
                t_y = np.sin(bend_angle)
                v_x = -bend_radius * (vx_norm * t_y - vy_norm * t_x)
                v_y = -bend_radius * (vx_norm * t_x + vy_norm * t_y)
                tangent_angle = np.arctan2(direction[1], direction[0])
                arrow.set_connectionstyle(
                    patches.ConnectionStyle.Angle3(
                        angleA=180 / np.pi * tangent_angle,
                        angleB=180 / np.pi * (tangent_angle + bend_angle),
                    )
                )

            else:
                v_x = sign * arrow_length * direction[0]
                v_y = sign * arrow_length * direction[1]

            arrow.set_positions(pos, (pos[0] + v_x, pos[1] + v_y))
            arrow.set_visible(True)
            arrow.draw(event.renderer)

        return _cb

    def _volume(self, bounds: Bound) -> float:
        """Returns object's volume within given bounds."""

        volume = 1

        for axis in range(3):
            min_bound = max(self.bounds[0][axis], bounds[0][axis])
            max_bound = min(self.bounds[1][axis], bounds[1][axis])

            volume *= max_bound - min_bound

        return volume

    def _surface_area(self, bounds: Bound) -> float:
        """Returns object's surface area within given bounds."""

        min_bounds = list(self.bounds[0])
        max_bounds = list(self.bounds[1])

        in_bounds_factor = [2, 2, 2]
        length = [0, 0, 0]

        for axis in (0, 1, 2):
            if min_bounds[axis] < bounds[0][axis]:
                min_bounds[axis] = bounds[0][axis]
                in_bounds_factor[axis] -= 1

            if max_bounds[axis] > bounds[1][axis]:
                max_bounds[axis] = bounds[1][axis]
                in_bounds_factor[axis] -= 1

            length[axis] = max_bounds[axis] - min_bounds[axis]

        return (
            length[0] * length[1] * in_bounds_factor[2]
            + length[1] * length[2] * in_bounds_factor[0]
            + length[2] * length[0] * in_bounds_factor[1]
        )

    """ Autograd code """

    def _compute_derivatives(self, derivative_info: DerivativeInfo) -> AutogradFieldMap:
        """Compute the adjoint derivatives for this object."""

        # get gradients w.r.t. each of the 6 faces (in normal direction)
        vjps_faces = self._derivative_faces(derivative_info=derivative_info)

        # post-process these values to give the gradients w.r.t. center and size
        vjps_center_size = self._derivatives_center_size(vjps_faces=vjps_faces)

        # store only the gradients asked for in 'field_paths'
        derivative_map = {}
        for field_path in derivative_info.paths:
            field_name, *index = field_path

            if field_name in vjps_center_size:
                # if the vjp calls for a specific index into the tuple
                if index and len(index) == 1:
                    index = int(index[0])
                    if field_path not in derivative_map:
                        derivative_map[field_path] = vjps_center_size[field_name][index]

                # otherwise, just grab the whole array
                else:
                    derivative_map[field_path] = vjps_center_size[field_name]

        return derivative_map

    @staticmethod
    def _derivatives_center_size(vjps_faces: Bound) -> dict[str, Coordinate]:
        """Derivatives with respect to the ``center`` and ``size`` fields in the ``Box``."""

        vjps_faces_min, vjps_faces_max = np.array(vjps_faces)

        # post-process min and max face gradients into center and size
        vjp_center = vjps_faces_max - vjps_faces_min
        vjp_size = (vjps_faces_min + vjps_faces_max) / 2.0

        return {
            "center": tuple(vjp_center.tolist()),
            "size": tuple(vjp_size.tolist()),
        }

    def _derivative_faces(self, derivative_info: DerivativeInfo) -> Bound:
        """Derivative with respect to normal position of 6 faces of ``Box``."""

        axes_to_compute = (0, 1, 2)
        if len(derivative_info.paths[0]) > 1:
            axes_to_compute = tuple(info[1] for info in derivative_info.paths)

        # change in permittivity between inside and outside
        vjp_faces = np.zeros((2, 3))

        for min_max_index, _ in enumerate((0, -1)):
            for axis in axes_to_compute:
                vjp_face = self._derivative_face(
                    min_max_index=min_max_index,
                    axis_normal=axis,
                    derivative_info=derivative_info,
                )

                # record vjp for this face
                vjp_faces[min_max_index, axis] = vjp_face

        return vjp_faces

    def _derivative_face(
        self,
        min_max_index: int,
        axis_normal: Axis,
        derivative_info: DerivativeInfo,
    ) -> float:
        """Compute the derivative w.r.t. shifting a face in the normal direction."""

        interpolators = derivative_info.interpolators or derivative_info.create_interpolators()
        _, axis_perp = self.pop_axis((0, 1, 2), axis=axis_normal)

        # First, check if the face is outside the simulation domain in which case set the
        # face gradient to 0.
        bounds_normal, _ = self.pop_axis(np.array(derivative_info.bounds).T, axis=axis_normal)
        coord_normal_face = bounds_normal[min_max_index]

        if min_max_index == 0:
            if coord_normal_face < derivative_info.simulation_bounds[0][axis_normal]:
                return 0.0
        else:
            if coord_normal_face > derivative_info.simulation_bounds[1][axis_normal]:
                return 0.0

        intersect_min, intersect_max = map(np.asarray, derivative_info.bounds_intersect)
        extents = intersect_max - intersect_min
        _, intersect_min_perp = self.pop_axis(np.array(intersect_min), axis=axis_normal)
        _, intersect_max_perp = self.pop_axis(np.array(intersect_max), axis=axis_normal)

        is_2d_map = []
        for axis_idx in range(3):
            if axis_idx == axis_normal:
                continue
            is_2d_map.append(np.isclose(extents[axis_idx], 0.0))

        if np.all(is_2d_map):
            return 0.0

        is_2d = np.any(is_2d_map)

        # Build point grid
        adaptive_spacing = derivative_info.adaptive_vjp_spacing()

        def spacing_to_grid_points(
            spacing: float, min_coord: float, max_coord: float
        ) -> NDArray[float]:
            N = np.maximum(3, 1 + int((max_coord - min_coord) / spacing))

            points = np.linspace(min_coord, max_coord, N)
            centers = 0.5 * (points[0:-1] + points[1:])

            return centers

        def verify_integration_interval(bound: tuple[float, float]) -> bool:
            # assume the bounds should not be equal or else this integration interval
            # would be the flat dimension of a 2D geometry.
            return bound[1] > bound[0]

        def compute_integration_weight(grid_points: NDArray[float]) -> float:
            grid_spacing = grid_points[1] - grid_points[0]
            if grid_spacing == 0.0:
                integration_weight = 1.0 / len(grid_points)
            else:
                integration_weight = grid_points[1] - grid_points[0]

            return integration_weight

        if is_2d:
            # build 1D grid for sampling points along the face, which is an edge in the 2D case
            zero_dim = np.where(is_2d_map)[0][0]
            # zero dim is one of the perpendicular directions, so the other perpendicular direction
            # is the nonzero dimension
            nonzero_dim = 1 - zero_dim

            # clip at simulation bounds for integration dimension
            integration_bounds_perp = (
                intersect_min_perp[nonzero_dim],
                intersect_max_perp[nonzero_dim],
            )

            if not verify_integration_interval(integration_bounds_perp):
                return 0.0

            grid_points_linear = spacing_to_grid_points(
                adaptive_spacing, integration_bounds_perp[0], integration_bounds_perp[1]
            )
            integration_weight = compute_integration_weight(grid_points_linear)

            grid_points = np.repeat(np.expand_dims(grid_points_linear.copy(), 1), 3, axis=1)

            # set up grid points to pass into evaluate_gradient_at_points
            grid_points[:, axis_perp[nonzero_dim]] = grid_points_linear
            grid_points[:, axis_perp[zero_dim]] = intersect_min_perp[zero_dim]
            grid_points[:, axis_normal] = coord_normal_face
        else:
            # build 3D grid for sampling points along the face

            # clip at simulation bounds for each integration dimension
            integration_bounds_perp = (
                (intersect_min_perp[0], intersect_max_perp[0]),
                (intersect_min_perp[1], intersect_max_perp[1]),
            )

            if not np.all([verify_integration_interval(b) for b in integration_bounds_perp]):
                return 0.0

            grid_points_perp_1 = spacing_to_grid_points(
                adaptive_spacing, integration_bounds_perp[0][0], integration_bounds_perp[0][1]
            )
            grid_points_perp_2 = spacing_to_grid_points(
                adaptive_spacing, integration_bounds_perp[1][0], integration_bounds_perp[1][1]
            )
            integration_weight = compute_integration_weight(
                grid_points_perp_1
            ) * compute_integration_weight(grid_points_perp_2)

            mesh_perp1, mesh_perp2 = np.meshgrid(grid_points_perp_1, grid_points_perp_2)

            zip_perp_coords = np.array(list(zip(mesh_perp1.flatten(), mesh_perp2.flatten())))

            grid_points = np.pad(zip_perp_coords.copy(), ((0, 0), (1, 0)), mode="constant")

            # set up grid points to pass into evaluate_gradient_at_points
            grid_points[:, axis_perp[0]] = zip_perp_coords[:, 0]
            grid_points[:, axis_perp[1]] = zip_perp_coords[:, 1]
            grid_points[:, axis_normal] = coord_normal_face

        normals = np.zeros_like(grid_points)
        perps1 = np.zeros_like(grid_points)
        perps2 = np.zeros_like(grid_points)

        normals[:, axis_normal] = -1 if (min_max_index == 0) else 1
        perps1[:, axis_perp[0]] = 1
        perps2[:, axis_perp[1]] = 1

        gradient_at_points = derivative_info.evaluate_gradient_at_points(
            spatial_coords=grid_points,
            normals=normals,
            perps1=perps1,
            perps2=perps2,
            interpolators=interpolators,
        )

        vjp_value = np.sum(integration_weight * np.real(gradient_at_points))
        return vjp_value

    @staticmethod
    def _face_axes_for_paths(paths: list[PathType]) -> tuple[int, ...]:
        """Face-normal axes implied by the requested derivative paths.

        All axes unless every requested path names a specific axis (robust to mixed
        indexed/unindexed path lists from grouped router dispatch). Shared by
        generation and consumption so the implied canonical keys cannot disagree.
        """
        if all(len(path) > 1 for path in paths):
            return tuple(sorted({path[1] for path in paths}))
        return (0, 1, 2)

    def _make_adjoint_sample_sets(
        self, paths: list[PathType], ctx: SamplingContext
    ) -> dict[PathType, SurfaceSampleSet]:
        """Generate face sample sets under canonical ``("faces", min_max_index, axis)`` keys.

        Every implied face key is emitted; a face outside the simulation domain or
        collapsed by clipping carries an explicit empty set (zero gradient). The face
        sets serve both the ``center`` and ``size`` derivative paths.
        """
        from tidy3d.em.translate.sample_sets import SurfaceSampleSet

        sample_sets = {}
        for min_max_index in (0, 1):
            for axis_normal in self._face_axes_for_paths(paths):
                sample_set = self._make_face_sample_set(
                    min_max_index=min_max_index,
                    axis_normal=axis_normal,
                    ctx=ctx,
                )
                if sample_set is None:
                    sample_set = SurfaceSampleSet.empty(serves_paths=(("center",), ("size",)))
                sample_sets[(_BOX_FACES, min_max_index, axis_normal)] = sample_set

        return sample_sets

    def _compute_derivatives_from_sample_sets(
        self,
        sample_sets: dict[PathType, SurfaceSampleSet],
        paths: list[PathType],
        derivative_info: DerivativeInfo,
    ) -> AutogradFieldMap:
        """Compute ``center``/``size`` derivatives from pre-generated face sample sets."""
        interpolators = derivative_info.interpolators or derivative_info.create_interpolators()

        # gradients w.r.t. each of the 6 faces (in normal direction)
        vjps_faces = np.zeros((2, 3))
        for min_max_index in (0, 1):
            for axis_normal in self._face_axes_for_paths(paths):
                key = (_BOX_FACES, min_max_index, axis_normal)
                if key not in sample_sets:
                    raise AdjointError(
                        f"'Box' sample sets are missing the canonical {key} key required "
                        f"by derivative paths {tuple(paths)}; got keys {tuple(sample_sets)}."
                    )
                sample_set = sample_sets[key]
                if sample_set.num_points == 0:
                    continue  # explicit zero contribution
                gradient_at_points = sample_set.evaluate(
                    derivative_info, interpolators=interpolators
                )
                vjps_faces[min_max_index, axis_normal] = np.sum(
                    sample_set.weights.values * np.real(gradient_at_points)
                )

        # post-process these values to give the gradients w.r.t. center and size
        vjps_center_size = self._derivatives_center_size(vjps_faces=vjps_faces)

        # store only the gradients asked for in 'field_paths'
        derivative_map = {}
        for field_path in paths:
            field_name, *index = field_path

            if field_name in vjps_center_size:
                # if the vjp calls for a specific index into the tuple
                if index and len(index) == 1:
                    index = int(index[0])
                    if field_path not in derivative_map:
                        derivative_map[field_path] = vjps_center_size[field_name][index]

                # otherwise, just grab the whole array
                else:
                    derivative_map[field_path] = vjps_center_size[field_name]

        return derivative_map

    @staticmethod
    def _derivatives_center_size(vjps_faces: Bound) -> dict[str, Coordinate]:
        """Derivatives with respect to the ``center`` and ``size`` fields in the ``Box``."""

        vjps_faces_min, vjps_faces_max = np.array(vjps_faces)

        # post-process min and max face gradients into center and size
        vjp_center = vjps_faces_max - vjps_faces_min
        vjp_size = (vjps_faces_min + vjps_faces_max) / 2.0

        return {
            "center": tuple(vjp_center.tolist()),
            "size": tuple(vjp_size.tolist()),
        }

    def _make_face_sample_set(
        self,
        min_max_index: int,
        axis_normal: Axis,
        ctx: SamplingContext,
    ) -> SurfaceSampleSet | None:
        """Generate the surface samples for one face, or ``None`` if it contributes none.

        The face's identity is carried by its canonical key ``("faces", min_max_index,
        axis_normal)``; the set itself holds only samples and quadrature weights.
        """
        from tidy3d.em.translate.sample_sets import SurfaceSampleSet

        _, axis_perp = self.pop_axis((0, 1, 2), axis=axis_normal)

        # First, check if the face is outside the simulation domain in which case the
        # face contributes no samples (zero gradient).
        bounds_normal, _ = self.pop_axis(np.array(ctx.bounds).T, axis=axis_normal)
        coord_normal_face = bounds_normal[min_max_index]

        if min_max_index == 0:
            if coord_normal_face < ctx.simulation_bounds[0][axis_normal]:
                return None
        else:
            if coord_normal_face > ctx.simulation_bounds[1][axis_normal]:
                return None

        intersect_min, intersect_max = map(np.asarray, ctx.bounds_intersect)
        extents = intersect_max - intersect_min
        _, intersect_min_perp = self.pop_axis(np.array(intersect_min), axis=axis_normal)
        _, intersect_max_perp = self.pop_axis(np.array(intersect_max), axis=axis_normal)

        is_2d_map = []
        for axis_idx in range(3):
            if axis_idx == axis_normal:
                continue
            is_2d_map.append(np.isclose(extents[axis_idx], 0.0))

        if np.all(is_2d_map):
            return None

        is_2d = np.any(is_2d_map)

        # Build point grid
        adaptive_spacing = ctx.spacing

        def spacing_to_grid_points(
            spacing: float, min_coord: float, max_coord: float
        ) -> NDArray[float]:
            N = np.maximum(3, 1 + int((max_coord - min_coord) / spacing))

            points = np.linspace(min_coord, max_coord, N)
            centers = 0.5 * (points[0:-1] + points[1:])

            return centers

        def verify_integration_interval(bound: tuple[float, float]) -> bool:
            # assume the bounds should not be equal or else this integration interval
            # would be the flat dimension of a 2D geometry.
            return bound[1] > bound[0]

        def compute_integration_weight(grid_points: NDArray[float]) -> float:
            grid_spacing = grid_points[1] - grid_points[0]
            if grid_spacing == 0.0:
                integration_weight = 1.0 / len(grid_points)
            else:
                integration_weight = grid_points[1] - grid_points[0]

            return integration_weight

        if is_2d:
            # build 1D grid for sampling points along the face, which is an edge in the 2D case
            zero_dim = np.where(is_2d_map)[0][0]
            # zero dim is one of the perpendicular directions, so the other perpendicular direction
            # is the nonzero dimension
            nonzero_dim = 1 - zero_dim

            # clip at simulation bounds for integration dimension
            integration_bounds_perp = (
                intersect_min_perp[nonzero_dim],
                intersect_max_perp[nonzero_dim],
            )

            if not verify_integration_interval(integration_bounds_perp):
                return None

            grid_points_linear = spacing_to_grid_points(
                adaptive_spacing, integration_bounds_perp[0], integration_bounds_perp[1]
            )
            integration_weight = compute_integration_weight(grid_points_linear)

            grid_points = np.repeat(np.expand_dims(grid_points_linear.copy(), 1), 3, axis=1)

            # set up grid points to pass into evaluate_gradient_at_points
            grid_points[:, axis_perp[nonzero_dim]] = grid_points_linear
            grid_points[:, axis_perp[zero_dim]] = intersect_min_perp[zero_dim]
            grid_points[:, axis_normal] = coord_normal_face
        else:
            # build 3D grid for sampling points along the face

            # clip at simulation bounds for each integration dimension
            integration_bounds_perp = (
                (intersect_min_perp[0], intersect_max_perp[0]),
                (intersect_min_perp[1], intersect_max_perp[1]),
            )

            if not np.all([verify_integration_interval(b) for b in integration_bounds_perp]):
                return None

            grid_points_perp_1 = spacing_to_grid_points(
                adaptive_spacing, integration_bounds_perp[0][0], integration_bounds_perp[0][1]
            )
            grid_points_perp_2 = spacing_to_grid_points(
                adaptive_spacing, integration_bounds_perp[1][0], integration_bounds_perp[1][1]
            )
            integration_weight = compute_integration_weight(
                grid_points_perp_1
            ) * compute_integration_weight(grid_points_perp_2)

            mesh_perp1, mesh_perp2 = np.meshgrid(grid_points_perp_1, grid_points_perp_2)

            zip_perp_coords = np.array(list(zip(mesh_perp1.flatten(), mesh_perp2.flatten())))

            grid_points = np.pad(zip_perp_coords.copy(), ((0, 0), (1, 0)), mode="constant")

            # set up grid points to pass into evaluate_gradient_at_points
            grid_points[:, axis_perp[0]] = zip_perp_coords[:, 0]
            grid_points[:, axis_perp[1]] = zip_perp_coords[:, 1]
            grid_points[:, axis_normal] = coord_normal_face

        normals = np.zeros_like(grid_points)
        perps1 = np.zeros_like(grid_points)
        perps2 = np.zeros_like(grid_points)

        normals[:, axis_normal] = -1 if (min_max_index == 0) else 1
        perps1[:, axis_perp[0]] = 1
        perps2[:, axis_perp[1]] = 1

        return SurfaceSampleSet.from_arrays(
            points=grid_points,
            normals=normals,
            perps1=perps1,
            perps2=perps2,
            weights=np.full(len(grid_points), integration_weight),
            serves_paths=(("center",), ("size",)),
        )


"""Compound subclasses"""


"""Compound subclasses"""
