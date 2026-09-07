"""Shared abstraction for spatially varying electromagnetic media."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import autograd.numpy as np
from pydantic import Field

from tidy3d.components.autograd.derivative_utils import (
    bounds_slice,
    compute_spatial_weights,
    transpose_interp_axis,
)
from tidy3d.components.base import cached_property
from tidy3d.components.data.data_array import DATA_ARRAY_MAP
from tidy3d.components.data.unstructured.base import UnstructuredGridDataset
from tidy3d.components.data.utils import _get_numpy_array
from tidy3d.components.geometry.contour_conversion import gdstk_contours_from_custom_medium
from tidy3d.components.types import TYPE_TAG_STR, InterpMethod
from tidy3d.constants import EPSILON_0

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from tidy3d.compat import Self
    from tidy3d.components.data.data_array import ScalarFieldDataArray, SpatialDataArray
    from tidy3d.components.data.utils import CustomSpatialDataType
    from tidy3d.components.grid.grid import Coords
    from tidy3d.components.types import (
        ArrayComplex3D,
        Axis,
        Bound,
        Bound2D,
        PermittivityComponent,
    )

    from .base import ArrayFloat, ArrayGeneric, FrequencyArray
    from .perturbation import PerturbationMediumType

from .base import ALLOWED_INTERP_METHODS, AbstractMedium, ensure_freq_in_range


class AbstractCustomMedium(AbstractMedium, ABC):
    """A spatially varying medium."""

    @cached_property
    def is_custom(self) -> bool:
        """Whether the medium is custom."""
        return True

    interp_method: InterpMethod = Field(
        default="nearest",
        title="Interpolation method",
        description="Interpolation method to obtain permittivity values "
        "that are not supplied at the Yee grids; For grids outside the range "
        "of the supplied data, extrapolation will be applied. When the extrapolated "
        "value is smaller (greater) than the minimal (maximal) of the supplied data, "
        "the extrapolated value will take the minimal (maximal) of the supplied data.",
    )

    subpixel: bool = Field(
        default=False,
        title="Subpixel averaging",
        description="If ``True``, apply the subpixel averaging method specified by "
        "``Simulation``'s field ``subpixel`` for this type of material on the "
        "interface of the structure, including exterior boundary and "
        "intersection interfaces with other structures.",
    )

    derived_from: PerturbationMediumType | None = Field(
        default=None,
        discriminator=TYPE_TAG_STR,
        title="Parent Medium",
        description="If not ``None``, it records the parent medium from which this medium was derived.",
    )

    @cached_property
    @abstractmethod
    def is_isotropic(self) -> bool:
        """The medium is isotropic or anisotropic."""

    def _interp_method(self, comp: Axis) -> InterpMethod:
        """Interpolation method applied to comp."""
        return self.interp_method

    @abstractmethod
    def eps_dataarray_freq(
        self, frequency: float
    ) -> tuple[CustomSpatialDataType, CustomSpatialDataType, CustomSpatialDataType]:
        """Permittivity array at ``frequency``.

        Parameters
        ----------
        frequency : float
            Frequency to evaluate permittivity at (Hz).

        Returns
        -------
        tuple[Union[:class:`.SpatialDataArray`, :class:`.TriangularGridDataset`, :class:`.TetrahedralGridDataset`], Union[:class:`.SpatialDataArray`, :class:`.TriangularGridDataset`, :class:`.TetrahedralGridDataset`], Union[:class:`.SpatialDataArray`, :class:`.TriangularGridDataset`, :class:`.TetrahedralGridDataset`]]
            The permittivity evaluated at ``frequency``.
        """

    def eps_diagonal_on_grid(
        self,
        frequency: float,
        coords: Coords,
    ) -> tuple[ArrayComplex3D, ArrayComplex3D, ArrayComplex3D]:
        """Spatial profile of main diagonal of the complex-valued permittivity
        at ``frequency`` interpolated at the supplied coordinates.

        Parameters
        ----------
        frequency : float
            Frequency to evaluate permittivity at (Hz).
        coords : :class:`.Coords`
            The grid point coordinates over which interpolation is performed.

        Returns
        -------
        tuple[ArrayComplex3D, ArrayComplex3D, ArrayComplex3D]
            The complex-valued permittivity tensor at ``frequency`` interpolated
            at the supplied coordinate.
        """
        eps_spatial = self.eps_dataarray_freq(frequency)
        return self._interp_eps_diagonal_on_grid(eps_spatial=eps_spatial, coords=coords)

    def _interp_eps_diagonal_on_grid(
        self,
        eps_spatial: tuple[CustomSpatialDataType, CustomSpatialDataType, CustomSpatialDataType],
        coords: Coords,
    ) -> tuple[ArrayComplex3D, ArrayComplex3D, ArrayComplex3D]:
        """Interpolate already-evaluated permittivity data onto supplied coordinates."""

        def _interp_and_squeeze(eps_comp: Any, comp: int) -> NDArray[Any]:
            """Interpolate spatially and drop any leftover frequency dimension."""
            result = coords.spatial_interp(eps_comp, self._interp_method(comp))
            if hasattr(result, "dims") and "f" in result.dims:
                result = result.squeeze("f", drop=True)
            return _get_numpy_array(result)

        if self.is_isotropic:
            eps_interp = _interp_and_squeeze(eps_spatial[0], 0)
            return (eps_interp, eps_interp, eps_interp)
        return tuple(
            _interp_and_squeeze(eps_comp, comp) for comp, eps_comp in enumerate(eps_spatial)
        )

    def eps_comp_on_grid(
        self,
        row: Axis,
        col: Axis,
        frequency: float,
        coords: Coords,
    ) -> ArrayComplex3D:
        """Spatial profile of a single component of the complex-valued permittivity tensor at
        ``frequency`` interpolated at the supplied coordinates.

        Parameters
        ----------
        row : int
            Component's row in the permittivity tensor (0, 1, or 2 for x, y, or z respectively).
        col : int
            Component's column in the permittivity tensor (0, 1, or 2 for x, y, or z respectively).
        frequency : float
            Frequency to evaluate permittivity at (Hz).
        coords : :class:`.Coords`
            The grid point coordinates over which interpolation is performed.

        Returns
        -------
        ArrayComplex3D
            Single component of the complex-valued permittivity tensor at ``frequency`` interpolated
            at the supplied coordinates.
        """

        if row == col:
            return self.eps_diagonal_on_grid(frequency, coords)[row]
        return 0j

    @staticmethod
    def _spatial_average(
        eps_comp: CustomSpatialDataType, frequency: float | FrequencyArray
    ) -> complex | ArrayGeneric:
        """Average a custom permittivity field over spatial dimensions only."""
        values = _get_numpy_array(eps_comp)

        if np.isscalar(frequency):
            return np.mean(values)

        if hasattr(eps_comp, "dims") and "f" in eps_comp.dims:
            freq_axis = eps_comp.dims.index("f")
            values = np.moveaxis(values, freq_axis, -1)

        num_freqs = np.asarray(frequency, dtype=float).size
        return np.mean(values.reshape(-1, num_freqs), axis=0)

    @ensure_freq_in_range
    def eps_model(self, frequency: float) -> complex:
        """Complex-valued spatially averaged permittivity as a function of frequency."""
        if not np.isscalar(frequency):
            freqs = np.asarray(frequency, dtype=float)
            eps_values = [self.eps_model(float(freq)) for freq in freqs.reshape(-1)]
            return np.array(eps_values).reshape(freqs.shape)

        if self.is_isotropic:
            return self._spatial_average(self.eps_dataarray_freq(frequency)[0], frequency)
        return np.mean(
            [
                self._spatial_average(eps_comp, frequency)
                for eps_comp in self.eps_dataarray_freq(frequency)
            ],
            axis=0,
        )

    @ensure_freq_in_range
    def eps_diagonal(self, frequency: float) -> tuple[complex, complex, complex]:
        """Main diagonal of the complex-valued permittivity tensor
        at ``frequency``. Spatially, we take max{||eps||}, so that autoMesh generation
        works appropriately.
        """
        eps_spatial = self.eps_dataarray_freq(frequency)
        if self.is_isotropic:
            eps_comp = _get_numpy_array(eps_spatial[0]).ravel()
            eps = eps_comp[np.argmax(np.abs(eps_comp))]
            return (eps, eps, eps)
        eps_spatial_array = (_get_numpy_array(eps_comp).ravel() for eps_comp in eps_spatial)
        return tuple(eps_comp[np.argmax(np.abs(eps_comp))] for eps_comp in eps_spatial_array)

    def _get_real_vals(self, x: ArrayGeneric) -> ArrayFloat:
        """Grab the real part of the values in array.
        Used for _eps_bounds()
        """
        return _get_numpy_array(np.real(x)).ravel()

    def _eps_bounds(
        self,
        frequency: float | None = None,
        eps_component: PermittivityComponent | None = None,
    ) -> tuple[float, float]:
        """Returns permittivity bounds for setting the color bounds when plotting.

        Parameters
        ----------
        frequency : float = None
            Frequency to evaluate the relative permittivity of all mediums.
            If not specified, evaluates at infinite frequency.
        eps_component : Optional[PermittivityComponent] = None
            Component of the permittivity tensor to plot for anisotropic materials,
            e.g. ``"xx"``, ``"yy"``, ``"zz"``, ``"xy"``, ``"yz"``, ...
            Defaults to ``None``, which returns the average of the diagonal values.

        Returns
        -------
        tuple[float, float]
            The min and max values of the permittivity for the selected component and evaluated at ``frequency``.
        """
        eps_dataarray = self.eps_dataarray_freq(frequency)
        all_eps = np.concatenate(self._get_real_vals(eps_comp) for eps_comp in eps_dataarray)
        return (np.min(all_eps), np.max(all_eps))

    @staticmethod
    def _validate_isreal_dataarray(dataarray: CustomSpatialDataType) -> bool:
        """Validate that the dataarray is real"""
        return np.all(np.isreal(_get_numpy_array(dataarray)))

    @staticmethod
    def _validate_isreal_dataarray_tuple(
        dataarray_tuple: tuple[CustomSpatialDataType, ...],
    ) -> bool:
        """Validate that the dataarray is real"""
        return np.all([AbstractCustomMedium._validate_isreal_dataarray(f) for f in dataarray_tuple])

    @abstractmethod
    def _sel_custom_data_inside(self, bounds: Bound) -> Self:
        """Return a new medium that contains the minimal amount custom data necessary to cover
        a spatial region defined by ``bounds``."""

    def sel_inside(self, bounds: Bound) -> AbstractCustomMedium:
        """Return a new medium that contains the minimal amount data necessary to cover
        a spatial region defined by ``bounds``.


        Parameters
        ----------
        bounds : tuple[float, float, float], tuple[float, float float]
            Min and max bounds packaged as ``(minx, miny, minz), (maxx, maxy, maxz)``.

        Returns
        -------
        AbstractMedium
            Medium with reduced data.
        """

        self_mod_data_reduced = super().sel_inside(bounds)

        return self_mod_data_reduced._sel_custom_data_inside(bounds)

    @staticmethod
    def _not_loaded(field: Any) -> bool:
        """Check whether data was not loaded."""
        if isinstance(field, str) and field in DATA_ARRAY_MAP:
            return True
        # attempting to construct an UnstructuredGridDataset from a dict
        if isinstance(field, dict) and field.get("type") in (
            "TriangularGridDataset",
            "TetrahedralGridDataset",
        ):
            return any(
                isinstance(subfield, str) and subfield in DATA_ARRAY_MAP
                for subfield in [field["points"], field["cells"], field["values"]]
            )
        # attempting to pass an UnstructuredGridDataset with zero points
        if isinstance(field, UnstructuredGridDataset):
            return any(len(subfield) == 0 for subfield in [field.points, field.cells, field.values])
        return False

    def _gdstk_contours(
        self,
        *,
        axis: int,
        plane_position: float,
        bounds_xyz: tuple[tuple[float, float, float], tuple[float, float, float]],
        permittivity_threshold: float,
        frequency: float,
        pixel_exact: bool,
        eps_components: tuple[CustomSpatialDataType, CustomSpatialDataType, CustomSpatialDataType]
        | None = None,
    ) -> tuple[list[Any], Bound2D, float]:
        """Create GDS contour polygons from this medium on one planar slice."""
        contours, frame_bounds, in_plane_step, *_ = gdstk_contours_from_custom_medium(
            self,
            axis=axis,
            plane_position=plane_position,
            bounds_xyz=bounds_xyz,
            permittivity_threshold=permittivity_threshold,
            frequency=frequency,
            pixel_exact=pixel_exact,
            eps_components=eps_components,
        )
        return contours, frame_bounds, in_plane_step

    def _derivative_field_cmp_custom(
        self,
        E_der_map: dict[str, ScalarFieldDataArray],
        spatial_data: SpatialDataArray,
        dim: str,
        bounds: Bound | None = None,
        component: str = "real",
        interp_method: InterpMethod | None = None,
        sum_over_freqs: bool = True,
    ) -> NDArray:
        """Compute the derivative with respect to a material property component."""
        param_coords = {axis: np.asarray(spatial_data.coords[axis]) for axis in "xyz"}
        eps_shape = [len(param_coords[axis]) for axis in "xyz"]
        dtype_out = complex if component == "complex" else float

        E_der_dim = E_der_map[f"E{dim}"]
        if np.all(E_der_dim.values == 0):
            zero_shape = eps_shape
            if not sum_over_freqs:
                zero_shape = [*eps_shape, len(np.asarray(E_der_dim.coords["f"], float))]
            return np.zeros(zero_shape, dtype=dtype_out)

        field_values_da = E_der_dim

        if bounds is not None:
            (xmin, ymin, zmin), (xmax, ymax, zmax) = bounds
            warning_context = "CustomMedium parameter gradients (adjoint field grid -> medium grid)"
            sx = bounds_slice(
                np.asarray(field_values_da.coords["x"]),
                xmin,
                xmax,
                name="x",
                warning_context=warning_context,
            )
            sy = bounds_slice(
                np.asarray(field_values_da.coords["y"]),
                ymin,
                ymax,
                name="y",
                warning_context=warning_context,
            )
            sz = bounds_slice(
                np.asarray(field_values_da.coords["z"]),
                zmin,
                zmax,
                name="z",
                warning_context=warning_context,
            )
            field_values_da = field_values_da.isel(x=sx, y=sy, z=sz)

        field_coords = {axis: np.asarray(field_values_da.coords[axis]) for axis in "xyz"}
        weights = compute_spatial_weights(field_values_da, dims=("x", "y", "z"))
        weighted_values_da = field_values_da * weights
        # Copy to avoid modifying underlying data through in-place operations below.
        values = weighted_values_da.values.copy()

        method = interp_method if interp_method is not None else self.interp_method

        if method not in ALLOWED_INTERP_METHODS:
            raise ValueError(
                f"Unsupported interpolation method: {method!r}. "
                f"Choose one of: {', '.join(ALLOWED_INTERP_METHODS)}."
            )

        def _interp_axis(
            arr: NDArray, axis: int, field_axis: NDArray, param_axis: NDArray
        ) -> NDArray:
            """Accumulate values from the field grid onto the parameter grid along one axis.

            Moves ``axis`` to the front, applies ``_transpose_interp_axis`` (adjoint of 1D interpolation)
            to map from ``field_axis`` (n_field) to ``param_axis`` (n_param), then moves the axis back.
            """
            moved = np.moveaxis(arr, axis, 0)
            moved = transpose_interp_axis(
                moved,
                field_axis,
                param_axis,
                method=method,
            )
            return np.moveaxis(moved, 0, axis)

        values = _interp_axis(values, 0, field_coords["x"], param_coords["x"])
        values = _interp_axis(values, 1, field_coords["y"], param_coords["y"])
        values = _interp_axis(values, 2, field_coords["z"], param_coords["z"])

        freqs_da = np.asarray(field_values_da.coords["f"])
        if component == "sigma":
            values = values.imag * (-1.0 / (2.0 * np.pi * freqs_da * EPSILON_0))
        elif component == "imag":
            values = values.imag
        elif component == "real":
            values = values.real

        if sum_over_freqs:
            vjp_array = values.sum(axis=-1).reshape(eps_shape)
        else:
            vjp_array = values.reshape([*eps_shape, values.shape[-1]])

        # match derivative dtype to the underlying dataset for real-valued components
        if component != "complex":
            target_array = getattr(spatial_data, "values", None)
            if target_array is None and hasattr(spatial_data, "data"):
                target_array = spatial_data.data
            if target_array is not None:
                target_dtype = np.asarray(target_array).dtype
                if not np.issubdtype(target_dtype, np.complexfloating):
                    vjp_array = np.real(vjp_array).astype(target_dtype, copy=False)

        return vjp_array
