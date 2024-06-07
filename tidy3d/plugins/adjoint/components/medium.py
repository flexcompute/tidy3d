"""Defines jax-compatible mediums."""

from __future__ import annotations

from abc import ABC
from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np
import pydantic.v1 as pd
import xarray as xr
from jax.tree_util import register_pytree_node_class

from ....components.data.monitor_data import FieldData
from ....components.geometry.base import Geometry
from ....components.medium import AnisotropicMedium, CustomMedium, Medium
from ....components.types import Bound, Literal
from ....constants import CONDUCTIVITY
from ....exceptions import SetupError
from .base import WEB_ADJOINT_MESSAGE, JaxObject
from .data.data_array import JaxDataArray
from .data.dataset import JaxPermittivityDataset
from .types import JaxFloat

# number of integration points per unit wavelength in material
PTS_PER_WVL_INTEGRATION = 20

# maximum number of pixels allowed in each component of a JaxCustomMedium
MAX_NUM_CELLS_CUSTOM_MEDIUM = 250_000


class AbstractJaxMedium(ABC, JaxObject):
    """Holds some utility functions for Jax medium types."""

    def _get_volume_disc(
        self, grad_data: FieldData, sim_bounds: Bound, wvl_mat: float
    ) -> Tuple[Dict[str, np.ndarray], float]:
        """Get the coordinates and volume element for the inside of the corresponding structure."""

        # find intersecting volume between structure and simulation
        mnt_bounds = grad_data.monitor.geometry.bounds
        rmin, rmax = Geometry.bounds_intersection(mnt_bounds, sim_bounds)

        # assemble volume coordinates and differential volume element
        d_vol = 1.0
        vol_coords = {}
        for coord_name, min_edge, max_edge in zip("xyz", rmin, rmax):
            size = max_edge - min_edge

            # don't discretize this dimension if there is no thickness along it
            if size == 0:
                vol_coords[coord_name] = [max_edge]
                continue

            # update the volume element value
            num_cells_dim = int(size * PTS_PER_WVL_INTEGRATION / wvl_mat) + 1
            d_len = size / num_cells_dim
            d_vol *= d_len

            # construct the interpolation coordinates along this dimension
            coords_interp = np.linspace(min_edge + d_len / 2, max_edge - d_len / 2, num_cells_dim)
            vol_coords[coord_name] = coords_interp

        return vol_coords, d_vol

    @staticmethod
    def make_inside_mask(vol_coords: Dict[str, np.ndarray], inside_fn: Callable) -> xr.DataArray:
        """Make a 3D mask of where the volume coordinates are inside a supplied function."""

        meshgrid_args = [vol_coords[dim] for dim in "xyz" if dim in vol_coords]
        vol_coords_meshgrid = np.meshgrid(*meshgrid_args, indexing="ij")
        inside_kwargs = dict(zip("xyz", vol_coords_meshgrid))
        values = inside_fn(**inside_kwargs)
        return xr.DataArray(values, coords=vol_coords)

    def e_mult_volume(
        self,
        field: Literal["Ex", "Ey", "Ez"],
        grad_data_fwd: FieldData,
        grad_data_adj: FieldData,
        vol_coords: Dict[str, np.ndarray],
        d_vol: float,
        inside_fn: Callable,
    ) -> xr.DataArray:
        """Get the E_fwd * E_adj * dV field distribution inside of the discretized volume."""

        e_fwd = grad_data_fwd.field_components[field]
        e_adj = grad_data_adj.field_components[field]

        e_dotted = e_fwd * e_adj

        inside_mask = self.make_inside_mask(vol_coords=vol_coords, inside_fn=inside_fn)

        isel_kwargs = {
            key: [0]
            for key, value in vol_coords.items()
            if isinstance(value, float) or len(value) <= 1
        }
        interp_kwargs = {key: value for key, value in vol_coords.items() if key not in isel_kwargs}

        fields_eval = e_dotted.isel(**isel_kwargs).interp(**interp_kwargs, assume_sorted=True)

        inside_mask = inside_mask.isel(**isel_kwargs)

        mask_dV = inside_mask * d_vol
        fields_eval = fields_eval.assign_coords(**mask_dV.coords)

        return mask_dV * fields_eval

    def d_eps_map(
        self,
        grad_data_fwd: FieldData,
        grad_data_adj: FieldData,
        sim_bounds: Bound,
        wvl_mat: float,
        inside_fn: Callable,
    ) -> xr.DataArray:
        """Mapping of gradient w.r.t. permittivity at each point in discretized volume."""

        vol_coords, d_vol = self._get_volume_disc(
            grad_data=grad_data_fwd, sim_bounds=sim_bounds, wvl_mat=wvl_mat
        )

        e_mult_sum = 0.0

        for field in ("Ex", "Ey", "Ez"):
            e_mult_sum += self.e_mult_volume(
                field=field,
                grad_data_fwd=grad_data_fwd,
                grad_data_adj=grad_data_adj,
                vol_coords=vol_coords,
                d_vol=d_vol,
                inside_fn=inside_fn,
            )

        return e_mult_sum


@register_pytree_node_class
class JaxMedium(Medium, AbstractJaxMedium):
    """A :class:`.Medium` registered with jax."""

    _tidy3d_class = Medium

    permittivity_jax: JaxFloat = pd.Field(
        1.0,
        title="Permittivity",
        description="Relative permittivity of the medium. May be a ``jax`` ``Array``.",
        stores_jax_for="permittivity",
    )

    conductivity_jax: JaxFloat = pd.Field(
        0.0,
        title="Conductivity",
        description="Electric conductivity. Defined such that the imaginary part of the complex "
        "permittivity at angular frequency omega is given by conductivity/omega.",
        units=CONDUCTIVITY,
        stores_jax_for="conductivity",
    )

    def store_vjp(
        self,
        grad_data_fwd: FieldData,
        grad_data_adj: FieldData,
        sim_bounds: Bound,
        wvl_mat: float,
        inside_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    ) -> JaxMedium:
        """Returns the gradient of the medium parameters given forward and adjoint field data."""

        # integrate the dot product of each E component over the volume, update vjp for epsilon
        d_eps_map = self.d_eps_map(
            grad_data_fwd=grad_data_fwd,
            grad_data_adj=grad_data_adj,
            sim_bounds=sim_bounds,
            wvl_mat=wvl_mat,
            inside_fn=inside_fn,
        )

        vjp_eps_complex = d_eps_map.sum(dim=("x", "y", "z"))

        vjp_eps = 0.0
        vjp_sigma = 0.0

        for freq in d_eps_map.coords["f"]:
            vjp_eps_complex_f = vjp_eps_complex.sel(f=freq)
            _vjp_eps, _vjp_sigma = self.eps_complex_to_eps_sigma(vjp_eps_complex_f, freq)
            vjp_eps += _vjp_eps
            vjp_sigma += _vjp_sigma

        return self.copy(
            update=dict(
                permittivity_jax=vjp_eps,
                conductivity_jax=vjp_sigma,
            )
        )


@register_pytree_node_class
class JaxAnisotropicMedium(AnisotropicMedium, AbstractJaxMedium):
    """A :class:`.Medium` registered with jax."""

    _tidy3d_class = AnisotropicMedium

    xx: JaxMedium = pd.Field(
        ...,
        title="XX Component",
        description="Medium describing the xx-component of the diagonal permittivity tensor.",
        jax_field=True,
    )

    yy: JaxMedium = pd.Field(
        ...,
        title="YY Component",
        description="Medium describing the yy-component of the diagonal permittivity tensor.",
        jax_field=True,
    )

    zz: JaxMedium = pd.Field(
        ...,
        title="ZZ Component",
        description="Medium describing the zz-component of the diagonal permittivity tensor.",
        jax_field=True,
    )

    def store_vjp(
        self,
        grad_data_fwd: FieldData,
        grad_data_adj: FieldData,
        sim_bounds: Bound,
        wvl_mat: float,
        inside_fn: Callable,
    ) -> JaxMedium:
        """Returns the gradient of the medium parameters given forward and adjoint field data."""

        # integrate the dot product of each E component over the volume, update vjp for epsilon
        vol_coords, d_vol = self._get_volume_disc(
            grad_data=grad_data_fwd, sim_bounds=sim_bounds, wvl_mat=wvl_mat
        )

        vjp_fields = {}
        for component in "xyz":
            field_name = "E" + component
            component_name = component + component
            e_mult_dim = self.e_mult_volume(
                field=field_name,
                grad_data_fwd=grad_data_fwd,
                grad_data_adj=grad_data_adj,
                vol_coords=vol_coords,
                d_vol=d_vol,
                inside_fn=inside_fn,
            )

            vjp_eps_complex_ii = e_mult_dim.sum(dim=("x", "y", "z"))
            freq = e_mult_dim.coords["f"][0]

            vjp_eps_ii = 0.0
            vjp_sigma_ii = 0.0

            for freq in e_mult_dim.coords["f"]:
                vjp_eps_complex_ii_f = vjp_eps_complex_ii.sel(f=freq)
                _vjp_eps_ii, _vjp_sigma_ii = self.eps_complex_to_eps_sigma(
                    vjp_eps_complex_ii_f, freq
                )
                vjp_eps_ii += _vjp_eps_ii
                vjp_sigma_ii += _vjp_sigma_ii

            vjp_medium = self.components[component_name]
            vjp_fields[component_name] = vjp_medium.updated_copy(
                permittivity_jax=vjp_eps_ii,
                conductivity_jax=vjp_sigma_ii,
            )

        return self.copy(update=vjp_fields)


@register_pytree_node_class
class JaxCustomMedium(CustomMedium, AbstractJaxMedium):
    """A :class:`.CustomMedium` registered with ``jax``.
    Note: The gradient calculation assumes uniform field across the pixel.
    Therefore, the accuracy degrades as the pixel size becomes large
    with respect to the field variation.
    """

    _tidy3d_class = CustomMedium

    eps_dataset: Optional[JaxPermittivityDataset] = pd.Field(
        None,
        title="Permittivity Dataset",
        description="User-supplied dataset containing complex-valued permittivity "
        "as a function of space. Permittivity distribution over the Yee-grid will be "
        "interpolated based on the data nearest to the grid location.",
        jax_field=True,
    )

    @pd.root_validator(pre=True)
    def _pre_deprecation_dataset(cls, values):
        """Don't allow permittivity as a field until we support it."""
        if values.get("permittivity") or values.get("conductivity"):
            raise SetupError(
                "'permittivity' and 'conductivity' are not yet supported in adjoint plugin. "
                "Please continue to use the 'eps_dataset' field to define the component "
                "of the permittivity tensor."
            )
        return values

    def _validate_web_adjoint(self) -> None:
        """Run validators for this component, only if using ``tda.web.run()``."""
        self._is_not_too_large()

    def _is_not_too_large(self):
        """Ensure number of pixels does not surpass a set amount."""

        field_components = self.eps_dataset.field_components

        for field_dim in "xyz":
            field_name = f"eps_{field_dim}{field_dim}"
            data_array = field_components[field_name]
            coord_lens = [len(data_array.coords[key]) for key in "xyz"]
            num_cells_dim = np.prod(coord_lens)
            if num_cells_dim > MAX_NUM_CELLS_CUSTOM_MEDIUM:
                raise SetupError(
                    "For the adjoint plugin, each component of the 'JaxCustomMedium.eps_dataset' "
                    f"is restricted to have a maximum of {MAX_NUM_CELLS_CUSTOM_MEDIUM} cells. "
                    f"Detected {num_cells_dim} grid cells in the '{field_name}' component. "
                    + WEB_ADJOINT_MESSAGE
                )

    @pd.validator("eps_dataset", always=True)
    def _eps_dataset_single_frequency(cls, val):
        """Override of inherited validator. (still needed)"""
        return val

    @pd.validator("eps_dataset", always=True)
    def _eps_dataset_eps_inf_greater_no_less_than_one_sigma_positive(cls, val, values):
        """Override of inherited validator."""
        return val

    def store_vjp(
        self,
        grad_data_fwd: FieldData,
        grad_data_adj: FieldData,
        sim_bounds: Bound,
        wvl_mat: float,
        inside_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    ) -> JaxMedium:
        """Returns the gradient of the medium parameters given forward and adjoint field data."""

        # get the boundaries of the intersection of the CustomMedium and the Simulation
        mnt_bounds = grad_data_fwd.monitor.geometry.bounds
        bounds_intersect = Geometry.bounds_intersection(mnt_bounds, sim_bounds)

        # get the grids associated with the user-supplied coordinates within these bounds
        grids = self.grids(bounds=bounds_intersect)

        vjp_field_components = {}
        for dim in "xyz":
            eps_field_name = f"eps_{dim}{dim}"

            # grab the original data and its coordinates
            orig_data_array = self.eps_dataset.field_components[eps_field_name]
            coords = orig_data_array.coords

            grid = grids[eps_field_name]
            d_sizes = grid.sizes
            d_sizes = [d_sizes.x, d_sizes.y, d_sizes.z]

            # construct the coordinates for interpolation and selection within the custom medium
            # TODO: extend this to all points within the volume.
            interp_coords = {}
            sum_axes = []

            for dim_index, dim_pt in enumerate("xyz"):
                coord_dim = coords[dim_pt]

                # if it's uniform / single pixel along this dim
                if len(np.array(coord_dim)) == 1:
                    # discretize along this edge like a regular volume

                    # compute the length of the pixel within the sim bounds
                    r_min_coords, r_max_coords = grid.boundaries.to_list[dim_index]
                    r_min_sim, r_max_sim = np.array(sim_bounds).T[dim_index]
                    r_min = max(r_min_coords, r_min_sim)
                    r_max = min(r_max_coords, r_max_sim)
                    size = abs(r_max - r_min)

                    # compute the length element along the dim, handling case of sim.size=0
                    if size > 0:
                        # discretize according to PTS_PER_WVL
                        num_cells_dim = int(size * PTS_PER_WVL_INTEGRATION / wvl_mat) + 1
                        d_len = size / num_cells_dim
                        coords_interp = np.linspace(
                            r_min + d_len / 2, r_max - d_len / 2, num_cells_dim
                        )

                    else:
                        # just interpolate at the single position, dL=1 to normalize out
                        d_len = 1.0
                        coords_interp = np.array([(r_min + r_max) / 2.0])

                    # construct the interpolation coordinates along this dimension
                    d_sizes[dim_index] = np.array([d_len])
                    interp_coords[dim_pt] = coords_interp

                    # only sum this dimension if there are multiple points
                    sum_axes.append(dim_pt)

                # otherwise
                else:
                    # just evaluate at the original data coords
                    interp_coords[dim_pt] = coord_dim

            # outer product all dimensions to get a volume element mask
            d_vols = np.einsum("i, j, k -> ijk", *d_sizes)

            # grab the corresponding dotted fields at these interp_coords and sum over len-1 pixels
            field_name = "E" + dim
            e_dotted = (
                self.e_mult_volume(
                    field=field_name,
                    grad_data_fwd=grad_data_fwd,
                    grad_data_adj=grad_data_adj,
                    vol_coords=interp_coords,
                    d_vol=d_vols,
                    inside_fn=inside_fn,
                )
                .sum(sum_axes)
                .sum(dim="f")
            )

            # reshape values to the expected vjp shape to be more safe
            vjp_shape = tuple(len(coord) for _, coord in coords.items())

            # make sure this has the same dtype as the original
            dtype_orig = np.array(orig_data_array.values).dtype

            vjp_values = e_dotted.values.reshape(vjp_shape)
            if dtype_orig.kind == "f":
                vjp_values = vjp_values.real
            vjp_values = vjp_values.astype(dtype_orig)

            # construct a DataArray storing the vjp
            vjp_data_array = JaxDataArray(values=vjp_values, coords=coords)
            vjp_field_components[eps_field_name] = vjp_data_array

        # package everything into dataset
        vjp_eps_dataset = JaxPermittivityDataset(**vjp_field_components)
        return self.copy(update=dict(eps_dataset=vjp_eps_dataset))


JaxMediumType = Union[JaxMedium, JaxAnisotropicMedium, JaxCustomMedium]


JAX_MEDIUM_MAP = {
    Medium: JaxMedium,
    AnisotropicMedium: JaxAnisotropicMedium,
    CustomMedium: JaxCustomMedium,
}
