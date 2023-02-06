# pylint: disable=invalid-name
"""Defines jax-compatible geometries and their conversion to grad monitors."""
from __future__ import annotations

from abc import ABC
from typing import Tuple, Union

import pydantic as pd
import numpy as np
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class
import jax

from ....components.base import cached_property
from ....components.types import Bound, Axis
from ....components.geometry import Geometry, Box, PolySlab
from ....components.data.monitor_data import FieldData, PermittivityData
from ....components.monitor import FieldMonitor, PermittivityMonitor
from ....constants import fp_eps, MICROMETER

from .base import JaxObject
from .types import JaxFloat, validate_jax_tuple, validate_jax_tuple_tuple
from ..log import AdjointError

# number of integration points per unit wavelength in material
PTS_PER_WVL_INTEGRATION = 50

# how much to expand the gradient monitors on each side beyond the self.bounds
GRAD_MONITOR_EXPANSION = fp_eps


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

    def make_grad_monitors(
        self, freq: float, name: str
    ) -> Tuple[FieldMonitor, PermittivityMonitor]:
        """Return gradient monitor associated with this object."""
        size_enlarged = tuple(s + 2 * GRAD_MONITOR_EXPANSION for s in self.bound_size)
        field_mnt = FieldMonitor(
            size=size_enlarged,
            center=self.bound_center,
            fields=["Ex", "Ey", "Ez"],
            freqs=[freq],
            name=name + "_field",
        )

        eps_mnt = PermittivityMonitor(
            size=size_enlarged,
            center=self.bound_center,
            freqs=[freq],
            name=name + "_eps",
        )
        return field_mnt, eps_mnt

    def to_tidy3d(self) -> Geometry:
        """Convert :class:`.JaxGeometry` instance to :class:`.Geometry`"""
        self_dict = self.dict(exclude={"type"})
        map_reverse = {v: k for k, v in JAX_GEOMETRY_MAP.items()}
        tidy3d_type = map_reverse[type(self)]
        return tidy3d_type.parse_obj(self_dict)


@register_pytree_node_class
class JaxBox(JaxGeometry, Box, JaxObject):
    """A :class:`.Box` registered with jax."""

    size: Tuple[JaxFloat, JaxFloat, JaxFloat] = pd.Field(
        ...,
        title="Size",
        description="Size of the box in (x,y,z). May contain ``jax`` ``DeviceArray`` instances.",
        jax_field=True,
    )

    center: Tuple[JaxFloat, JaxFloat, JaxFloat] = pd.Field(
        ...,
        title="Center",
        description="Center of the box in (x,y,z). May contain ``jax`` ``DeviceArray`` instances.",
        jax_field=True,
    )

    _sanitize_size = validate_jax_tuple("size")
    _sanitize_center = validate_jax_tuple("center")

    @cached_property
    def bounds(self):
        size = jax.lax.stop_gradient(self.size)
        center = jax.lax.stop_gradient(self.center)
        coord_min = tuple(c - s / 2 for (s, c) in zip(size, center))
        coord_max = tuple(c + s / 2 for (s, c) in zip(size, center))
        return (coord_min, coord_max)

    @pd.validator("center", always=True)
    def _center_not_inf(cls, val):
        """Overrides validator enforing that val is not inf."""
        return val

    # pylint: disable=too-many-locals, too-many-arguments, too-many-statements, unused-argument
    def store_vjp(
        self,
        grad_data_fwd: FieldData,
        grad_data_adj: FieldData,
        grad_data_eps: PermittivityData,
        sim_bounds: Bound,
        wvl_mat: float,
    ) -> JaxBox:
        """Stores the gradient of the box parameters given forward and adjoint field data."""

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

                    # select the forward and adjoint E fields along this edge
                    field_name = "E" + field_cmp_dim
                    e_fwd = grad_data_fwd.field_components[field_name].isel(f=0)
                    e_adj = grad_data_adj.field_components[field_name].isel(f=0)

                    # select the permittivity data
                    eps_field_name = f"eps_{field_cmp_dim}{field_cmp_dim}"
                    eps_data = grad_data_eps.field_components[eps_field_name].isel(f=0)

                    # get the permittivity values just inside and outside the edge
                    n_cells_in = 2
                    isel_out = 0 if min_max_index == 0 else -1
                    isel_ins = n_cells_in if min_max_index == 0 else -n_cells_in - 1
                    eps2 = eps_data.isel(**{dim_normal: isel_out})
                    eps1 = eps_data.isel(**{dim_normal: isel_ins})

                    # get gradient contribution for normal component using D field
                    if field_cmp_dim == dim_normal:

                        # construct normal D fields, dot together at surface
                        d_fwd = e_fwd * eps_data
                        d_adj = e_adj * eps_data
                        d_normal = (d_fwd * d_adj).interp(**normal_coord)

                        # compute adjoint contribution using perturbation theory for shifting bounds
                        delta_eps_inv = 1.0 / eps1 - 1.0 / eps2
                        d_integrand = -(delta_eps_inv * d_normal).interp(**area_coords).real
                        grad_contrib = d_area * jnp.sum(d_integrand.values)

                    # get gradient contribution for parallel components using E fields
                    else:

                        # measure parallel E fields, dot together at surface
                        e_parallel = (e_fwd * e_adj).interp(**normal_coord)

                        # compute adjoint contribution using perturbation theory for shifting bounds
                        delta_eps = eps1 - eps2
                        e_integrand = +(delta_eps * e_parallel).interp(**area_coords).real
                        grad_contrib = d_area * jnp.sum(e_integrand.values)

                    # grad_contrib *= 1 / k0**3
                    vjp_surfs[dim_normal][min_max_index] += grad_contrib

        # convert surface vjps to center, size vjps. Note, convert these to jax types w/ jnp.sum()
        vjp_center = tuple(jnp.sum(vjp_surfs[dim][1] - vjp_surfs[dim][0]) for dim in "xyz")
        vjp_size = tuple(jnp.sum(0.5 * (vjp_surfs[dim][1] + vjp_surfs[dim][0])) for dim in "xyz")
        return self.copy(update=dict(center=vjp_center, size=vjp_size))


@register_pytree_node_class
class JaxPolySlab(JaxGeometry, PolySlab, JaxObject):
    """A :class:`.PolySlab` registered with jax."""

    vertices: Tuple[Tuple[JaxFloat, JaxFloat], ...] = pd.Field(
        ...,
        title="Vertices",
        description="List of (d1, d2) defining the 2 dimensional positions of the polygon "
        "face vertices at the ``reference_plane``. "
        "The index of dimension should be in the ascending order: e.g. if "
        "the slab normal axis is ``axis=y``, the coordinate of the vertices will be in (x, z)",
        units=MICROMETER,
        jax_field=True,
    )

    # @pd.validator("slab_bounds", always=True)
    # def _is_3d(cls, val):
    #     """Make sure the box is 3D."""
    #     slab_min, slab_max = val
    #     if slab_max <= slab_min:
    #         raise AdjointError(f"'JaxPolySlab' has (min, max) bounds of '{val}', it must be 3D.")
    #     return val

    _sanitize_vertices = validate_jax_tuple_tuple("vertices")

    @cached_property
    def bounds(self) -> Bound:
        """Returns bounding box min and max coordinates. The dilation and slant angle are not
        taken into account exactly for speed. Instead, the polygon may be slightly smaller than
        the returned bounds, but it should always be fully contained.

        Returns
        -------
        Tuple[float, float, float], Tuple[float, float float]
            Min and max bounds packaged as ``(minx, miny, minz), (maxx, maxy, maxz)``.
        """

        # check for the maximum possible contribution from dilation/slant on each side

        xmin, ymin = np.amin(jax.lax.stop_gradient(self.vertices), axis=0)
        xmax, ymax = np.amax(jax.lax.stop_gradient(self.vertices), axis=0)

        # get bounds in (local) z
        zmin, zmax = self.slab_bounds

        # rearrange axes
        coords_min = self.unpop_axis(zmin, (xmin, ymin), axis=self.axis)
        coords_max = self.unpop_axis(zmax, (xmax, ymax), axis=self.axis)
        return (tuple(coords_min), tuple(coords_max))

    @pd.validator("sidewall_angle", always=True)
    def no_sidewall(cls, val):
        """Overrides validator enforing that val is not inf."""
        if not np.isclose(val, 0.0):
            raise AdjointError("'JaxPolySlab' does not support slanted sidewall.")
        return val

    @pd.validator("dilation", always=True)
    def no_dilation(cls, val):
        """Overrides validator enforing that val is not inf."""
        if not np.isclose(val, 0.0):
            raise AdjointError("'JaxPolySlab' does not support dilation.")
        return val

    @pd.validator("vertices", always=True)
    def correct_shape(cls, val):
        """Overrides validator enforing that val is not inf."""
        return val

    @pd.validator("vertices", always=True)
    def no_self_intersecting_polygon_during_extrusion(cls, val, values):
        """Overrides validator enforing that val is not inf."""
        return val

    @pd.validator("vertices", always=True)
    def no_complex_self_intersecting_polygon_at_reference_plane(cls, val, values):
        """Overrides validator enforing that val is not inf."""
        return val

    # pylint: disable=too-many-locals, too-many-arguments, too-many-statements, unused-argument
    def store_vjp(
        self,
        grad_data_fwd: FieldData,
        grad_data_adj: FieldData,
        grad_data_eps: PermittivityData,
        sim_bounds: Bound,
        wvl_mat: float,
    ) -> JaxPolySlab:
        """Stores the gradient of the vertices given forward and adjoint field data."""

        def edge_contrib(
            v: Tuple[float, float], u: Tuple[float, float], axis: Axis
        ) -> Tuple[float, float]:
            """Gradient w.r.t. moving `vertex_grad` (v) in connection to `vertex_fixed` (u)."""

            # get edge between vertices
            v = np.array(jax.lax.stop_gradient(v))
            u = np.array(jax.lax.stop_gradient(u))
            edge = v - u
            length_edge = np.linalg.norm(edge)

            # get normal vectors tangent and perpendicular to edge
            tx, ty = edge / length_edge
            n = nx, ny = np.array((ty, -tx))

            def r(s: float) -> float:
                """Parameterization of position along edge from s=0 (u) to s=1 (v)."""
                return s * v + (1 - s) * u

            def edge_basis(field_data: FieldData, components_xyz: Tuple[str]):
                """Puts a field component from the (x, y, z) basis to the (t, n, z) basis."""

                # TODO: make this nicer and test
                xyz_coords = {
                    key[-1]: field_data.field_components[key].coords[key[-1]][1:-1]
                    for key in components_xyz
                }
                field_data = field_data.colocate(**xyz_coords)

                components_xyz_basis = [field_data[component] for component in components_xyz]

                Ez, (Ex, Ey) = self.pop_axis(components_xyz_basis, axis=axis)

                Et = Ex * tx + Ey * ty  # TODO: colocate at centers to do this properly
                En = Ex * nx + Ey * ny  # TODO: colocate at centers to do this properly
                return Et, En, Ez

            # get forward and adjoint fields in edge basis
            e_t_fwd, e_n_fwd, e_z_fwd = edge_basis(grad_data_fwd, components_xyz=("Ex", "Ey", "Ez"))
            e_t_adj, e_n_adj, e_z_adj = edge_basis(grad_data_adj, components_xyz=("Ex", "Ey", "Ez"))

            # get displacement fields
            _, eps_n, _ = edge_basis(grad_data_eps, components_xyz=("eps_xx", "eps_yy", "eps_zz"))
            d_n_fwd = eps_n * e_n_fwd
            d_n_adj = eps_n * e_n_adj

            # TODO: actually compute these..
            eps1 = 2.0
            eps2 = 1.0
            delta_eps_12 = eps1 - eps2
            delta_eps_inv_12 = 1.0 / eps1 - 1.0 / eps2

            def scalar_integrand(s: float, z: float) -> float:
                """Get integrand at position '(s, z)` along the edge."""

                x, y = r(s)
                coords_interp = dict(x=x, y=y, z=z)

                def evaluate(scalar_field) -> float:
                    scalar_field = scalar_field.isel(f=0)
                    scalar_field = scalar_field.interp(**coords_interp)
                    return scalar_field

                contrib_e_t = evaluate(delta_eps_12 * e_t_fwd * e_t_adj)
                contrib_e_z = evaluate(delta_eps_12 * e_z_fwd * e_z_adj)
                contrib_d_n = -evaluate(delta_eps_inv_12 * d_n_fwd * d_n_adj)
                return s * (contrib_e_t + contrib_d_n + contrib_e_z)

            slab_min, slab_max = self.slab_bounds
            length_axis = slab_max - slab_min

            num_cells_edge = int(length_edge * PTS_PER_WVL_INTEGRATION / wvl_mat) + 1
            num_cells_axis = int(length_axis * PTS_PER_WVL_INTEGRATION / wvl_mat) + 1

            s_vals = np.linspace(0, 1, num_cells_edge)
            z_vals = np.linspace(slab_min, slab_max, num_cells_axis)
            ds = np.mean(np.diff(s_vals)) * length_edge
            dz = np.mean(np.diff(z_vals))

            integrand = 0.0
            for s in s_vals:
                integrand += np.sum(scalar_integrand(s=s, z=z_vals).fillna(0).values)

            return n * np.real(ds * dz * integrand)

        def mod(num: int) -> int:
            """Get index modulo number of vertices."""
            return num % len(self.vertices)

        edge_contributions = []
        for i_vertex, vertex in enumerate(self.vertices):
            vertex_prev = self.vertices[mod(i_vertex - 1)]
            vertex_next = self.vertices[mod(i_vertex + 1)]
            edge_contribution = edge_contrib(vertex, vertex_next, axis=self.axis)
            edge_contribution += edge_contrib(vertex, vertex_prev, axis=self.axis)
            edge_contributions.append(edge_contribution)

        # new_vertices = []
        # for edge_index, edge_i in enumerate(edge_contributions):
        #     prev_edge_index = mod(edge_index - 1)
        #     edge_i_m1 = edge_contributions[prev_edge_index]
        #     vertex_contrib = tuple(val_i + val_i_m1 for val_i, val_i_m1 in zip(edge_i, edge_i_m1))
        #     new_vertices.append(vertex_contrib)

        return self.copy(update=dict(vertices=edge_contributions))


JaxGeometryType = Union[JaxBox, JaxPolySlab]

# pylint: disable=unhashable-member
JAX_GEOMETRY_MAP = {
    Box: JaxBox,
    PolySlab: JaxPolySlab,
}
