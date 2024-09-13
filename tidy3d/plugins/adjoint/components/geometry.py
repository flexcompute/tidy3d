"""Defines jax-compatible geometries and their conversion to grad monitors."""

from __future__ import annotations

from abc import ABC
from typing import Dict, List, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
import pydantic.v1 as pd
import shapely
import xarray as xr
from jax.tree_util import register_pytree_node_class
from joblib import Parallel, delayed

from ....components.base import cached_property
from ....components.data.data_array import ScalarFieldDataArray
from ....components.data.monitor_data import FieldData, PermittivityData
from ....components.geometry.base import Box, Geometry, GeometryGroup
from ....components.geometry.polyslab import (
    _COMPLEX_POLYSLAB_DIVISIONS_WARN,
    _IS_CLOSE_RTOL,
    PolySlab,
)
from ....components.monitor import FieldMonitor, PermittivityMonitor
from ....components.types import ArrayFloat2D, Bound, Coordinate2D  # , annotate_type
from ....constants import MICROMETER, fp_eps
from ....exceptions import AdjointError
from ....log import log
from ...polyslab import ComplexPolySlab
from .base import WEB_ADJOINT_MESSAGE, JaxObject
from .types import JaxFloat

# number of integration points per unit wavelength in material
PTS_PER_WVL_INTEGRATION = 50

# how much to expand the gradient monitors on each side beyond the self.bounds
GRAD_MONITOR_EXPANSION = fp_eps

# maximum number of vertices allowed in JaxPolySlab
MAX_NUM_VERTICES = 1000


class JaxGeometry(Geometry, ABC):
    """Abstract :class:`.Geometry` with methods useful for all Jax subclasses."""

    @property
    def bound_size(self) -> Tuple[float, float, float]:
        """Size of the bounding box of this geometry."""
        rmin, rmax = self.bounds
        return tuple(abs(pt_max - pt_min) for (pt_min, pt_max) in zip(rmin, rmax))

    @property
    def bound_center(self) -> Tuple[float, float, float]:
        """Size of the bounding box of this geometry."""
        rmin, rmax = self.bounds

        def get_center(pt_min: float, pt_max: float) -> float:
            """Get center of bounds, including infinity, calling Geometry._get_center()."""
            pt_min = jax.lax.stop_gradient(pt_min)
            pt_max = jax.lax.stop_gradient(pt_max)
            return self._get_center(pt_min, pt_max)

        return tuple(get_center(pt_min, pt_max) for (pt_min, pt_max) in zip(rmin, rmax))

    @cached_property
    def bounding_box(self):
        """Returns :class:`JaxBox` representation of the bounding box of a :class:`JaxGeometry`.

        Returns
        -------
        :class:`JaxBox`
            Geometric object representing bounding box.
        """
        return JaxBox.from_bounds(*self.bounds)

    def make_grad_monitors(
        self, freqs: List[float], name: str
    ) -> Tuple[FieldMonitor, PermittivityMonitor]:
        """Return gradient monitor associated with this object."""
        size_enlarged = tuple(s + 2 * GRAD_MONITOR_EXPANSION for s in self.bound_size)
        field_mnt = FieldMonitor(
            size=size_enlarged,
            center=self.bound_center,
            fields=["Ex", "Ey", "Ez"],
            freqs=freqs,
            name=name + "_field",
            colocate=False,
        )

        eps_mnt = PermittivityMonitor(
            size=size_enlarged,
            center=self.bound_center,
            freqs=freqs,
            name=name + "_eps",
        )
        return field_mnt, eps_mnt

    @staticmethod
    def compute_dotted_e_d_fields(
        grad_data_fwd: FieldData, grad_data_adj: FieldData, grad_data_eps: PermittivityData
    ) -> Tuple[Dict[str, ScalarFieldDataArray], Dict[str, ScalarFieldDataArray]]:
        """Get the (x,y,z) components of E_fwd * E_adj and D_fwd * D_adj fields in the domain."""

        e_mult_xyz = {}
        d_mult_xyz = {}

        for dim in "xyz":
            # grab the E field components
            e_fld_key = f"E{dim}"
            e_fwd = grad_data_fwd.field_components[e_fld_key]
            e_adj = grad_data_adj.field_components[e_fld_key]

            # grab the epsilon data
            eps_fld_key = f"eps_{dim}{dim}"
            eps = grad_data_eps.field_components[eps_fld_key]

            # compute d fields
            d_fwd = eps * e_fwd
            d_adj = eps * e_adj

            # multiply the fwd and adj fields
            e_mult_xyz[dim] = e_fwd * e_adj
            d_mult_xyz[dim] = d_fwd * d_adj

        return e_mult_xyz, d_mult_xyz


@register_pytree_node_class
class JaxBox(JaxGeometry, Box, JaxObject):
    """A :class:`.Box` registered with jax."""

    _tidy3d_class = Box

    center_jax: Tuple[JaxFloat, JaxFloat, JaxFloat] = pd.Field(
        (0.0, 0.0, 0.0),
        title="Center (Jax)",
        description="Jax traced value for the center of the box in (x, y, z).",
        units=MICROMETER,
        stores_jax_for="center",
    )

    size_jax: Tuple[JaxFloat, JaxFloat, JaxFloat] = pd.Field(
        ...,
        title="Size (Jax)",
        description="Jax-traced value for the size of the box in (x, y, z).",
        units=MICROMETER,
        stores_jax_for="size",
    )

    def store_vjp(
        self,
        grad_data_fwd: FieldData,
        grad_data_adj: FieldData,
        grad_data_eps: PermittivityData,
        sim_bounds: Bound,
        wvl_mat: float,
        eps_out: complex,
        eps_in: complex,
        num_proc: int = 1,
    ) -> JaxBox:
        """Stores the gradient of the box parameters given forward and adjoint field data."""

        # multiply the E and D field components together for fwd and adj
        e_mult_xyz, d_mult_xyz = self.compute_dotted_e_d_fields(
            grad_data_fwd=grad_data_fwd, grad_data_adj=grad_data_adj, grad_data_eps=grad_data_eps
        )

        rmin, rmax = bounds_intersect = self.bounds_intersection(self.bounds, sim_bounds)

        # stores vjps for the min and max surfaces on all dimensions
        vjp_surfs = {dim: np.array([0.0, 0.0]) for dim in "xyz"}

        # loop through all 6 surfaces (x,y,z) & (-, +)
        for dim_index, dim_normal in enumerate("xyz"):
            for min_max_index, min_max_val in enumerate(bounds_intersect):
                # get the normal coordinate of this surface
                normal_coord = {dim_normal: min_max_val[dim_index]}

                # skip if the geometry edge is out of bounds of the simulation
                sim_min_max_val = sim_bounds[min_max_index][dim_index]
                geo_min_max_val = self.bounds[min_max_index][dim_index]
                if (min_max_index == 0) and (geo_min_max_val <= sim_min_max_val):
                    continue
                if (min_max_index == 1) and (geo_min_max_val >= sim_min_max_val):
                    continue

                # get the dimensions and edge values on the plane of this surface
                _, dims_plane = self.pop_axis("xyz", axis=dim_index)
                _, mins_plane = self.pop_axis(rmin, axis=dim_index)
                _, maxs_plane = self.pop_axis(rmax, axis=dim_index)

                # construct differential area value and coordinates evenly spaced along this surface
                d_area = 1.0
                area_coords = {}
                for dim_plane, min_edge, max_edge in zip(dims_plane, mins_plane, maxs_plane):
                    # if there is no thickness along this dimension, skip it
                    length_edge = max_edge - min_edge
                    if length_edge == 0:
                        continue

                    num_cells_dim = int(length_edge * PTS_PER_WVL_INTEGRATION / wvl_mat) + 1

                    # update the differential area value
                    d_len = length_edge / num_cells_dim
                    d_area *= d_len

                    # construct evenly spaced coordinates along this dimension
                    interp_vals = np.linspace(
                        min_edge + d_len / 2, max_edge - d_len / 2, num_cells_dim
                    )
                    area_coords[dim_plane] = interp_vals

                # for each field component
                for field_cmp_dim in "xyz":
                    # select the permittivity data
                    eps_field_name = f"eps_{field_cmp_dim}{field_cmp_dim}"
                    eps_data = grad_data_eps.field_components[eps_field_name]

                    # get the permittivity values just inside and outside the edge

                    num_cells_normal_dim = len(eps_data.coords[dim_normal])

                    # if the eps_data is <=3 grid cells thick, pick the middle cell
                    if num_cells_normal_dim <= 3:
                        isel_ins = num_cells_normal_dim // 2

                    # otherwise, pick the cell 4 pixels from outside cell
                    else:
                        n_cells_in = 3
                        isel_ins = n_cells_in if min_max_index == 0 else -n_cells_in - 1

                    isel_out = 0 if min_max_index == 0 else -1

                    eps2 = eps_data.isel(**{dim_normal: isel_out})
                    eps1 = eps_data.isel(**{dim_normal: isel_ins})

                    # get gradient contribution for normal component using normal D field
                    if field_cmp_dim == dim_normal:
                        # construct normal D fields, dotted together at surface
                        d_normal = d_mult_xyz[field_cmp_dim]
                        d_normal = d_normal.interp(**normal_coord, assume_sorted=True)

                        # compute adjoint contribution using perturbation theory for shifting bounds
                        delta_eps_inv = 1.0 / eps1 - 1.0 / eps2
                        d_integrand = -(delta_eps_inv * d_normal).real
                        d_integrand = d_integrand.interp(**area_coords, assume_sorted=True)
                        grad_contrib = d_area * np.sum(d_integrand.values)

                    # get gradient contribution for parallel components using parallel E fields
                    else:
                        # measure parallel E fields, dotted together at surface
                        e_parallel = e_mult_xyz[field_cmp_dim]
                        e_parallel = e_parallel.interp(**normal_coord, assume_sorted=True)

                        # compute adjoint contribution using perturbation theory for shifting bounds
                        delta_eps = eps1 - eps2
                        e_integrand = +(delta_eps * e_parallel).real
                        e_integrand = e_integrand.interp(**area_coords, assume_sorted=True)
                        grad_contrib = d_area * np.sum(e_integrand.values)

                    # add this field contribution to the dict storing the surface contributions
                    vjp_surfs[dim_normal][min_max_index] += grad_contrib

        # convert surface vjps to center, size vjps. Note, convert these to jax types w/ np.sum()
        vjp_center = tuple(np.sum(vjp_surfs[dim][1] - vjp_surfs[dim][0]) for dim in "xyz")
        vjp_size = tuple(np.sum(0.5 * (vjp_surfs[dim][1] + vjp_surfs[dim][0])) for dim in "xyz")
        return self.copy(update=dict(center_jax=vjp_center, size_jax=vjp_size))


@register_pytree_node_class
class JaxPolySlab(JaxGeometry, PolySlab, JaxObject):
    """A :class:`.PolySlab` registered with jax."""

    _tidy3d_class = PolySlab

    vertices_jax: Tuple[Tuple[JaxFloat, JaxFloat], ...] = pd.Field(
        ...,
        title="Vertices (Jax)",
        description="Jax-traced list of (d1, d2) defining the 2 dimensional positions of the "
        "polygon face vertices at the ``reference_plane``. "
        "The index of dimension should be in the ascending order: e.g. if "
        "the slab normal axis is ``axis=y``, the coordinate of the vertices will be in (x, z)",
        units=MICROMETER,
        stores_jax_for="vertices",
    )

    slab_bounds_jax: Tuple[JaxFloat, JaxFloat] = pd.Field(
        ...,
        title="Slab bounds (Jax)",
        description="Jax-traced list of (h1, h2) defining the minimum and maximum positions "
        "of the slab along the ``axis`` dimension. ",
        units=MICROMETER,
        stores_jax_for="slab_bounds",
    )

    sidewall_angle_jax: JaxFloat = pd.Field(
        default=0.0,
        title="Sidewall angle (Jax)",
        description="Jax-traced float defining the sidewall angle of the slab "
        "along the ``axis`` dimension. ",
        units=MICROMETER,
        stores_jax_for="sidewall_angle",
    )

    dilation_jax: JaxFloat = pd.Field(
        default=0.0,
        title="Dilation (Jax)",
        description="Jax-traced float defining the dilation.",
        units=MICROMETER,
        stores_jax_for="dilation",
    )

    @pd.validator("sidewall_angle", always=True)
    def no_sidewall(cls, val):
        """Warn if sidewall angle present."""
        if not np.isclose(val, 0.0):
            log.warning(
                "'JaxPolySlab' does not yet perform the full adjoint gradient treatment "
                "for slanted sidewalls. "
                "A straight sidewall angle is assumed when computing the gradient with respect "
                "to shifting boundaries of the geometry. Therefore, as 'sidewall_angle' becomes "
                "further from '0.0', the gradient error can be significant. "
                "If high gradient accuracy is needed, please either reduce your 'sidewall_angle' "
                "or wait until this feature is supported fully in a later version.",
                log_once=True,
            )
        return val

    def _validate_web_adjoint(self) -> None:
        """Run validators for this component, only if using ``tda.web.run()``."""
        self._limit_number_of_vertices()

    def _limit_number_of_vertices(self) -> None:
        """Limit the maximum number of vertices."""
        if len(self.vertices_jax) > MAX_NUM_VERTICES:
            raise AdjointError(
                f"For performance, a maximum of {MAX_NUM_VERTICES} are allowed in 'JaxPolySlab'. "
                + WEB_ADJOINT_MESSAGE
            )

    def _extrusion_length_to_offset_distance(self, extrusion: float) -> float:
        """Convert extrusion length to offset distance."""
        if jnp.isclose(self.sidewall_angle_jax, 0):
            return 0.0
        return -extrusion * self._tanq

    @staticmethod
    def _orient(vertices: jnp.ndarray) -> jnp.ndarray:
        """Return a CCW-oriented polygon.

        Parameters
        ----------
        vertices : np.ndarray
            Shape (N, 2) defining the polygon vertices in the xy-plane.

        Returns
        -------
        np.ndarray
            Vertices of a CCW-oriented polygon.
        """
        return vertices if JaxPolySlab._area(vertices) > 0 else vertices[::-1, :]

    @staticmethod
    def _area(vertices: jnp.ndarray) -> float:
        """Compute the signed polygon area (positive for CCW orientation).

        Parameters
        ----------
        vertices : np.ndarray
            Shape (N, 2) defining the polygon vertices in the xy-plane.

        Returns
        -------
        float
            Signed polygon area (positive for CCW orientation).
        """
        x = vertices[:, 0]
        y = vertices[:, 1]
        return jnp.dot(x, jnp.roll(y, -1)) - jnp.dot(y, jnp.roll(x, -1)) / 2

    @staticmethod
    def _shift_vertices(
        vertices: jnp.ndarray, dist
    ) -> Tuple[jnp.ndarray, jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
        """Shifts the vertices of a polygon outward uniformly by distances
        `dists`.

        Parameters
        ----------
        jnp.ndarray
            Shape (N, 2) defining the polygon vertices in the xy-plane.
        dist : float
            Distance to offset.

        Returns
        -------
        Tuple[jnp.ndarray, jnp.narray, Tuple[jnp.ndarray, jnp.ndarray]]
            New polygon vertices;
            and the shift of vertices in direction parallel to the edges.
            Shift along x and y direction.
        """

        if jnp.isclose(dist, 0):
            return vertices, jnp.zeros(vertices.shape[0], dtype=float), None

        def rot90(v):
            return jnp.array([-v[1], v[0]])

        def normalize(v):
            return v / jnp.linalg.norm(v, axis=0)

        vs = vertices.T
        vs_next = jnp.roll(vs, axis=-1, shift=-1)
        vs_previous = jnp.roll(vs, axis=-1, shift=+1)

        asp = normalize(vs_next - vs)
        asm = normalize(vs - vs_previous)

        # the vertex shift is decomposed into parallel and perpendicular directions
        perpendicular_shift = -dist
        det = jnp.cross(asm, asp, axis=0)

        tan_half_angle = jnp.where(
            jnp.isclose(det, 0, rtol=_IS_CLOSE_RTOL),
            0.0,
            jnp.cross(asm, rot90(asm - asp), axis=0)
            / (det + jnp.isclose(det, 0, rtol=_IS_CLOSE_RTOL)),
        )
        parallel_shift = dist * tan_half_angle

        shift_total = perpendicular_shift * rot90(asm) + parallel_shift * asm

        return jnp.swapaxes(vs + shift_total, -2, -1), parallel_shift, shift_total

    @staticmethod
    def _neighbor_vertices_crossing_detection(
        vertices: jnp.ndarray, dist: float, ignore_at_dist: bool = True
    ) -> float:
        """Detect if neighboring vertices will cross after a dilation distance dist.

        Parameters
        ----------
        vertices : jnp.ndarray
            Shape (N, 2) defining the polygon vertices in the xy-plane.
        dist : float
            Distance to offset.
        ignore_at_dist : bool, optional
            whether to ignore the event right at ``dist`.

        Returns
        -------
        float
            the absolute value of the maximal allowed dilation
            if there are any crossing, otherwise return ``None``.
        """
        # ignore the event that occurs right at the offset distance
        if ignore_at_dist:
            dist -= fp_eps * dist / jnp.abs(dist)

        edge_length, edge_reduction = JaxPolySlab._edge_length_and_reduction_rate(vertices)
        length_remaining = edge_length - edge_reduction * dist

        mask = length_remaining < 0
        if jnp.any(mask):
            return jnp.min(jnp.abs(edge_length[mask] / edge_reduction[mask]))
        return None

    @staticmethod
    def _edge_length_and_reduction_rate(vertices: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Edge length of reduction rate of each edge with unit offset length.

        Parameters
        ----------
        vertices : jnp.ndarray
            Shape (N, 2) defining the polygon vertices in the xy-plane.

        Returns
        -------
        Tuple[jnp.ndarray, jnp.narray]
            edge length, and reduction rate
        """

        # edge length
        vs = vertices.T
        vs_next = jnp.roll(vs, axis=-1, shift=-1)
        edge_length = jnp.linalg.norm(vs_next - vs, axis=0)

        # edge length remaining
        parallel_shift = JaxPolySlab._shift_vertices(vertices, 1.0)[1]
        parallel_shift_p = jnp.roll(parallel_shift, shift=-1)
        edge_reduction = -(parallel_shift + parallel_shift_p)
        return edge_length, edge_reduction

    @staticmethod
    def _remove_duplicate_vertices(vertices: jnp.ndarray) -> jnp.ndarray:
        """Remove redundant/identical nearest neighbour vertices.

        Parameters
        ----------
        vertices : jnp.ndarray
            Shape (N, 2) defining the polygon vertices in the xy-plane.

        Returns
        -------
        np.ndarray
            Vertices of polygon.
        """

        vertices_f = jnp.roll(vertices, shift=-1, axis=0)
        vertices_diff = jnp.linalg.norm(vertices - vertices_f, axis=1)
        return vertices[~jnp.isclose(vertices_diff, 0, rtol=_IS_CLOSE_RTOL)]

    @staticmethod
    def _proper_vertices(vertices: ArrayFloat2D) -> jnp.ndarray:
        """convert vertices to jnp.array format,
        removing duplicate neighbouring vertices,
        and oriented in CCW direction.

        Returns
        -------
        ArrayLike[float, float]
           The vertices of the polygon for internal use.
        """

        vertices_np = JaxPolySlab.vertices_to_array(vertices)
        return JaxPolySlab._orient(JaxPolySlab._remove_duplicate_vertices(vertices_np))

    @staticmethod
    def _heal_polygon(vertices: jnp.ndarray) -> jnp.ndarray:
        """heal a self-intersecting polygon."""
        shapely_poly = PolySlab.make_shapely_polygon(jax.lax.stop_gradient(vertices))
        if shapely_poly.is_valid:
            return vertices

        raise NotImplementedError(
            "The dilation caused damage to the polygon. Automatically healing this is "
            "currently not supported for 'JaxPolySlab' objects. Try increasing the spacing "
            "between vertices or reduce the amount of dilation."
        )

    @staticmethod
    def vertices_to_array(vertices_tuple: ArrayFloat2D) -> jnp.ndarray:
        """Converts a list of tuples (vertices) to a jax array."""
        return jnp.asarray(vertices_tuple)

    @cached_property
    def reference_polygon(self) -> jnp.ndarray:
        """The polygon at the reference plane.

        Returns
        -------
        ArrayLike[float, float]
            The vertices of the polygon at the reference plane.
        """
        vertices = JaxPolySlab._proper_vertices(self.vertices_jax)
        if jnp.isclose(self.dilation, 0):
            return vertices
        offset_vertices = self._shift_vertices(vertices, self.dilation)[0]
        return self._heal_polygon(offset_vertices)

    def edge_contrib(
        self,
        vertex_grad: Coordinate2D,
        vertex_stat: Coordinate2D,
        is_next: bool,
        e_mult_xyz: Tuple[Dict[str, ScalarFieldDataArray]],
        d_mult_xyz: Tuple[Dict[str, ScalarFieldDataArray]],
        sim_bounds: Bound,
        wvl_mat: float,
        eps_out: complex,
        eps_in: complex,
    ) -> Coordinate2D:
        """Gradient w.r.t change in ``vertex_grad`` connected to ``vertex_stat``."""

        # TODO: (later) compute these grabbing from grad_data_eps at some distance away
        delta_eps_12 = eps_in - eps_out
        delta_eps_inv_12 = 1.0 / eps_in - 1.0 / eps_out

        # get edge pointing from moving vertex to static vertex
        vertex_grad = np.array(jax.lax.stop_gradient(vertex_grad))
        vertex_stat = np.array(jax.lax.stop_gradient(vertex_stat))
        edge = vertex_stat - vertex_grad
        length_edge = np.linalg.norm(edge)

        # if the edge length is 0 (overlapping vertices), there is no gradient contrib for this edge
        if np.isclose(length_edge, 0.0):
            return 0.0

        # get normalized vectors tangent to and perpendicular to edge in global caresian basis
        tx, ty = edge / length_edge
        normal_vector = np.array((+ty, -tx))

        # ensure normal vector is pointing "out" assuming clockwise vertices
        if is_next:
            normal_vector *= -1
        nx, ny = normal_vector

        # Check if vertices CCW or CW. Multiply by +1 if CCW to ensure normal out
        if self.is_ccw:
            normal_vector *= -1

        def edge_position(s: np.array) -> np.array:
            """Parameterization of position along edge from s=0 (static) to s=1 (gradient)."""
            return (1 - s) * vertex_stat[:, None] + s * vertex_grad[:, None]

        def edge_basis(
            xyz_components: Tuple[FieldData, FieldData, FieldData],
        ) -> Tuple[FieldData, FieldData, FieldData]:
            """Puts a field component from the (x, y, z) basis to the (t, n, z) basis."""
            cmp_z, (cmp_x_edge, cmp_y_edge) = self.pop_axis(xyz_components, axis=self.axis)

            cmp_t = cmp_x_edge * tx + cmp_y_edge * ty
            cmp_n = cmp_x_edge * nx + cmp_y_edge * ny

            return cmp_t, cmp_n, cmp_z

        def compute_integrand(s: np.array, z: np.array) -> np.array:
            """Get integrand at positions ``(s, z)`` along the edge."""

            # grab the position along edge and make dictionary of coords to interp with (s, z)
            x, y = edge_position(s=s)
            x = xr.DataArray(x, coords={"s": s})
            y = xr.DataArray(y, coords={"s": s})
            coords_interp = dict(x=x, y=y, z=z)

            def evaluate(scalar_field: ScalarFieldDataArray) -> float:
                """Evaluate a scalar field at a coordinate along the edge."""

                # if only 1 z coordinate, just isel the data.
                if len(z) == 1:
                    scalar_field = scalar_field.isel(z=0)
                    coords_xy = {key: coords_interp[key] for key in "xy"}
                    return scalar_field.interp(**coords_xy, assume_sorted=True)

                return scalar_field.interp(**coords_interp, assume_sorted=True)

            e_xyz_eval = [evaluate(e_fld) for e_fld in e_mult_xyz.values()]
            d_xyz_eval = [evaluate(d_fld) for d_fld in d_mult_xyz.values()]

            e_t_edge, _, e_z_edge = edge_basis(xyz_components=e_xyz_eval)
            _, d_n_edge, _ = edge_basis(xyz_components=d_xyz_eval)

            # get the correct sign to apply to the new fields
            sign_t, sign_n, sign_z = edge_basis(xyz_components=(1.0, 1.0, 1.0))
            e_t_edge *= sign_t
            d_n_edge *= sign_n
            e_z_edge *= sign_z

            # multiply by the change in epsilon (in, out) terms and sum contributions
            contrib_e_t = delta_eps_12 * e_t_edge
            contrib_e_z = delta_eps_12 * e_z_edge
            contrib_d_n = -delta_eps_inv_12 * d_n_edge
            contrib_total = contrib_e_t + contrib_d_n + contrib_e_z

            # scale the gradient contribution by the normalized distance from the static edge
            # make broadcasting work with both 2D and 3D simulation domains
            return (s * contrib_total.T).T

        # discretize along the edge
        # TODO: handle edge case where a vertex lies far outside simulation domain
        num_cells_edge = int(length_edge * PTS_PER_WVL_INTEGRATION / wvl_mat) + 1
        ds = 1.0 / float(num_cells_edge)
        s_vals = np.linspace(0 + ds / 2, 1 - ds / 2, num_cells_edge)

        # find the min and max of the slab within the simulation bounds
        slab_min, slab_max = self.slab_bounds
        sim_rmin, sim_rmax = sim_bounds
        sim_min = sim_rmin[self.axis]
        sim_max = sim_rmax[self.axis]
        z_max = min(slab_max, sim_max)
        z_min = max(slab_min, sim_min)

        # discretize along z
        length_axis = abs(z_max - z_min)
        num_cells_axis = int(length_axis * PTS_PER_WVL_INTEGRATION / wvl_mat) + 1
        dz = float(length_axis) / float(num_cells_axis)

        # handle a 2D simulation along axis (unitless)
        z_vals = np.linspace(z_min + dz / 2, z_max - dz / 2, num_cells_axis)

        if dz == 0.0:
            dz = 1.0

        # integrate by summing over axis edge (z) and parameterization point (s)
        integrand = compute_integrand(s=s_vals, z=z_vals).sum(dim="f")
        integral_result = np.sum(integrand.fillna(0).values)

        # project to the normal direction
        integral_result *= normal_vector

        # take the real part (from adjoint math) and multiply by area element
        return length_edge * ds * dz * np.real(integral_result)

    def vertex_vjp(
        self,
        i_vertex,
        e_mult_xyz: Tuple[Dict[str, ScalarFieldDataArray]],
        d_mult_xyz: Tuple[Dict[str, ScalarFieldDataArray]],
        sim_bounds: Bound,
        wvl_mat: float,
        eps_out: complex,
        eps_in: complex,
    ):
        """Compute the vjp for every vertex."""

        # get the location of the "previous" and "next" vertices in the polygon
        vertex = self.vertices[i_vertex]
        vertex_prev = self.vertices[(i_vertex - 1) % len(self.vertices)]
        vertex_next = self.vertices[(i_vertex + 1) % len(self.vertices)]

        # taking the current vertex "static", compute the edge contributions from prev and next
        contrib_next = self.edge_contrib(
            vertex_grad=vertex,
            vertex_stat=vertex_next,
            is_next=True,
            e_mult_xyz=e_mult_xyz,
            d_mult_xyz=d_mult_xyz,
            sim_bounds=sim_bounds,
            wvl_mat=wvl_mat,
            eps_out=eps_out,
            eps_in=eps_in,
        )
        contrib_prev = self.edge_contrib(
            vertex_grad=vertex,
            vertex_stat=vertex_prev,
            is_next=False,
            e_mult_xyz=e_mult_xyz,
            d_mult_xyz=d_mult_xyz,
            sim_bounds=sim_bounds,
            wvl_mat=wvl_mat,
            eps_out=eps_out,
            eps_in=eps_in,
        )

        # add the "forward" contribution from the "previous" contribution to get the vertex VJP
        return contrib_prev + contrib_next

    def store_vjp(
        self,
        grad_data_fwd: FieldData,
        grad_data_adj: FieldData,
        grad_data_eps: PermittivityData,
        sim_bounds: Bound,
        wvl_mat: float,
        eps_out: complex,
        eps_in: complex,
        num_proc: int = 1,
    ) -> JaxPolySlab:
        """Stores the gradient of the vertices given forward and adjoint field data."""

        # multiply the E and D field components together for fwd and adj
        e_mult_xyz, d_mult_xyz = self.compute_dotted_e_d_fields(
            grad_data_fwd=grad_data_fwd, grad_data_adj=grad_data_adj, grad_data_eps=grad_data_eps
        )

        if num_proc is not None and num_proc > 1:
            return self.store_vjp_parallel(
                e_mult_xyz=e_mult_xyz,
                d_mult_xyz=d_mult_xyz,
                sim_bounds=sim_bounds,
                wvl_mat=wvl_mat,
                eps_out=eps_out,
                eps_in=eps_in,
                num_proc=num_proc,
            )

        return self.store_vjp_sequential(
            e_mult_xyz=e_mult_xyz,
            d_mult_xyz=d_mult_xyz,
            sim_bounds=sim_bounds,
            wvl_mat=wvl_mat,
            eps_out=eps_out,
            eps_in=eps_in,
        )

    def _make_vertex_args(
        self,
        e_mult_xyz: FieldData,
        d_mult_xyz: FieldData,
        sim_bounds: Bound,
        wvl_mat: float,
        eps_out: complex,
        eps_in: complex,
    ) -> tuple:
        """Generate arguments for ``vertex_vjp``."""

        num_verts = len(self.vertices)

        arg_list = []

        for i in range(num_verts):
            args_i = [i] + [e_mult_xyz, d_mult_xyz, sim_bounds, wvl_mat, eps_out, eps_in]
            arg_list.append(args_i)

        return arg_list

    def store_vjp_sequential(
        self,
        e_mult_xyz: FieldData,
        d_mult_xyz: FieldData,
        sim_bounds: Bound,
        wvl_mat: float,
        eps_out: complex,
        eps_in: complex,
    ) -> JaxPolySlab:
        """Stores the gradient of the vertices given forward and adjoint field data."""
        # Construct arguments to pass to the parallel vertices_vjp computation

        arg_list = self._make_vertex_args(
            e_mult_xyz, d_mult_xyz, sim_bounds, wvl_mat, eps_out, eps_in
        )
        vertices_vjp = tuple(self.vertex_vjp(*args) for args in arg_list)
        vertices_vjp = tuple(tuple(x) for x in vertices_vjp)

        return self.updated_copy(vertices_jax=vertices_vjp)

    def store_vjp_parallel(
        self,
        e_mult_xyz: FieldData,
        d_mult_xyz: FieldData,
        sim_bounds: Bound,
        wvl_mat: float,
        eps_out: complex,
        eps_in: complex,
        num_proc: int = 1,
    ) -> JaxPolySlab:
        """Stores the gradient of the vertices given forward and adjoint field data."""

        args = self._make_vertex_args(e_mult_xyz, d_mult_xyz, sim_bounds, wvl_mat, eps_out, eps_in)
        vertices_vjp = Parallel(n_jobs=num_proc)(delayed(self.vertex_vjp)(*arg) for arg in args)
        vertices_vjp = tuple(tuple(x) for x in vertices_vjp)
        return self.updated_copy(vertices_jax=vertices_vjp)


@register_pytree_node_class
class JaxComplexPolySlab(JaxPolySlab, ComplexPolySlab):
    """A :class:`.ComplexPolySlab` registered with jax."""

    _tidy3d_class = ComplexPolySlab

    @pd.validator("vertices", always=True)
    def no_self_intersecting_polygon_during_extrusion(cls, val, values):
        """Turn off the validation for this class."""
        return val

    @property
    def geometry_group(self) -> None:
        """Divide a complex jax polyslab into a list of simple polyslabs, which
        are assembled into a :class:`.JaxGeometryGroup`.

        Returns
        -------
        :class:`.JaxGeometryGroup`
            JaxGeometryGroup for a list of simple jax polyslabs divided from the complex
            polyslab.
        """
        return JaxGeometryGroup(geometries=self.sub_polyslabs)

    def _dilation_value_at_reference_to_coord(self, dilation: float) -> float:
        """Compute the coordinate based on the dilation value to the reference plane."""

        z_coord = -dilation / self._tanq + self.slab_bounds_jax[0]
        if self.reference_plane == "middle":
            return z_coord + self.finite_length_axis / 2
        if self.reference_plane == "top":
            return z_coord + self.finite_length_axis
        # bottom case
        return z_coord

    @property
    def sub_polyslabs(self) -> List[JaxPolySlab]:
        """Divide a complex polyslab into a list of simple polyslabs.
        Only neighboring vertex-vertex crossing events are treated in this
        version.

        Returns
        -------
        List[JaxPolySlab]
            A list of simple jax polyslabs.
        """
        sub_polyslab_list = []
        num_division_count = 0

        # initialize sub-polyslab parameters

        sub_polyslab_dict = self.dict(
            exclude={  # all of these NEED to be overwritten, so best to exclude them
                "type",
                "vertices",
                "vertices_jax",
                "slab_bounds",
                "slab_bounds_jax",
                "sidewall_angle",
                "sidewall_angle_jax",
                "dilation",
                "reference_plane",
            }
        )
        if jnp.isclose(self.sidewall_angle_jax, 0):
            return [
                JaxPolySlab(
                    vertices=tuple(map(tuple, self.vertices_jax)),
                    slab_bounds=tuple(self.slab_bounds_jax),
                    sidewall_angle=self.sidewall_angle_jax,
                    dilation=self.dilation,
                    reference_plane=self.reference_plane,
                    **sub_polyslab_dict,
                )
            ]

        # initialize offset distance
        offset_distance = 0.0

        for dist_val in self._dilation_length:
            dist_now = 0.0
            vertices_now = self.reference_polygon

            # constructing sub-polyslabs until reaching the base/top
            while not jnp.isclose(dist_now, dist_val):
                # bounds for sub-polyslabs assuming no self-intersection
                slab_bounds = [
                    self._dilation_value_at_reference_to_coord(dist_now),
                    self._dilation_value_at_reference_to_coord(dist_val),
                ]
                # 1) find out any vertices touching events between the current
                # position to the base/top
                max_dist = JaxPolySlab._neighbor_vertices_crossing_detection(
                    vertices_now, dist_val - dist_now
                )

                # vertices touching events captured, update bounds for sub-polyslab
                if max_dist is not None:
                    # max_dist doesn't have sign, so construct signed offset distance
                    offset_distance = max_dist * dist_val / jnp.abs(dist_val)
                    slab_bounds = [
                        self._dilation_value_at_reference_to_coord(dist_now),
                        self._dilation_value_at_reference_to_coord(dist_now + offset_distance),
                    ]

                # 2) construct sub-polyslab
                slab_bounds = jnp.sort(
                    jnp.asarray(slab_bounds)
                )  # for reference_plane=top/bottom, bounds need to be ordered
                # direction of marching
                reference_plane = "bottom" if dist_val / self._tanq < 0 else "top"

                sub_polyslab_list.append(
                    JaxPolySlab(
                        vertices=tuple(map(tuple, vertices_now)),
                        slab_bounds=tuple(slab_bounds),
                        sidewall_angle=self.sidewall_angle_jax,
                        dilation=0.0,  # dilation accounted for in setup
                        reference_plane=reference_plane,
                        **sub_polyslab_dict,
                    )
                )

                # Now Step 3
                if max_dist is None:
                    break
                dist_now += offset_distance
                # new polygon vertices where collapsing vertices are removed but keep one
                vertices_now = JaxPolySlab._shift_vertices(vertices_now, offset_distance)[0]
                vertices_now = JaxPolySlab._remove_duplicate_vertices(vertices_now)
                # all vertices collapse
                if len(vertices_now) < 3:
                    break
                # polygon collapse into 1D
                if shapely.Polygon(jax.lax.stop_gradient(vertices_now)).buffer(0).area < fp_eps:
                    raise RuntimeError("Unhandled shapely transformation in JaxComplexPolySlab.")
                vertices_now = JaxPolySlab._orient(vertices_now)
                num_division_count += 1

        if num_division_count > _COMPLEX_POLYSLAB_DIVISIONS_WARN:
            log.warning(
                f"Too many self-intersecting events: the polyslab has been divided into "
                f"{num_division_count} polyslabs; more than {_COMPLEX_POLYSLAB_DIVISIONS_WARN} may "
                f"slow down the simulation."
            )

        return sub_polyslab_list


JaxSingleGeometryType = Union[JaxBox, JaxPolySlab]


@register_pytree_node_class
class JaxGeometryGroup(JaxGeometry, GeometryGroup, JaxObject):
    """A collection of Geometry objects that can be called as a single geometry object."""

    _tidy3d_class = GeometryGroup

    geometries: Tuple[JaxPolySlab, ...] = pd.Field(
        ...,
        title="Geometries",
        description="Tuple of jax geometries in a single grouping. "
        "Can provide significant performance enhancement in ``JaxStructure`` when all geometries "
        "are assigned the same ``JaxMedium``. Note: at this moment, only ``JaxPolySlab`` "
        "is supported.",
        jax_field=True,
    )

    @staticmethod
    def _store_vjp_geometry(
        geometry: JaxSingleGeometryType,
        grad_data_fwd: FieldData,
        grad_data_adj: FieldData,
        grad_data_eps: PermittivityData,
        sim_bounds: Bound,
        wvl_mat: float,
        eps_out: complex,
        eps_in: complex,
    ) -> JaxSingleGeometryType:
        """Function to store a single vjp for a single geometry."""
        return geometry.store_vjp(
            grad_data_fwd=grad_data_fwd,
            grad_data_adj=grad_data_adj,
            grad_data_eps=grad_data_eps,
            sim_bounds=sim_bounds,
            wvl_mat=wvl_mat,
            eps_out=eps_out,
            eps_in=eps_in,
        )

    def store_vjp(
        self,
        grad_data_fwd: FieldData,
        grad_data_adj: FieldData,
        grad_data_eps: PermittivityData,
        sim_bounds: Bound,
        wvl_mat: float,
        eps_out: complex,
        eps_in: complex,
        num_proc: int = 1,
    ) -> JaxGeometryGroup:
        """Returns a ``JaxGeometryGroup`` where the ``.geometries`` store the gradient info."""

        args_list = []
        for geo in self.geometries:
            args_i = [
                geo,
                grad_data_fwd,
                grad_data_adj,
                grad_data_eps,
                sim_bounds,
                wvl_mat,
                eps_out,
                eps_in,
            ]
            args_list.append(args_i)

        if num_proc == 1:
            geometries_vjp = tuple(self._store_vjp_geometry(*args) for args in args_list)
        else:
            geometries_vjp = tuple(
                Parallel(n_jobs=num_proc)(
                    delayed(self._store_vjp_geometry)(*args) for args in args_list
                )
            )

        return self.updated_copy(geometries=geometries_vjp)


JaxGeometryType = Union[JaxSingleGeometryType, JaxGeometryGroup]


JAX_GEOMETRY_MAP = {
    Box: JaxBox,
    PolySlab: JaxPolySlab,
    ComplexPolySlab: JaxComplexPolySlab,
    GeometryGroup: JaxGeometryGroup,
}
