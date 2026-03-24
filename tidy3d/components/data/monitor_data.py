"""Monitor Level Data, store the DataArrays associated with a single monitor."""

from __future__ import annotations

import struct
import warnings
from abc import ABC
from math import isclose
from typing import TYPE_CHECKING, Any, Callable, Optional, Union, get_args

import autograd.numpy as np
import xarray as xr
from pydantic import Field, model_validator

from tidy3d.components.autograd.source_factory import (
    current_component_data_array,
    diffraction_source_from_data,
    mode_source_from_monitor,
)
from tidy3d.components.autograd.source_factory import (
    flip_direction as _flip_direction,
)
from tidy3d.components.base import TYPE_TAG_STR, cached_property
from tidy3d.components.base_sim.data.monitor_data import (
    AbstractMonitorData,
    AbstractUnstructuredMonitorData,
)
from tidy3d.components.data.utils import (
    _complex_power_flow_numpy,
    _dot_numpy,
    _get_broadcast_selection,
    _get_intersection_selection,
    _instantaneous_power_flow_numpy,
    _outer_dot_numpy,
)
from tidy3d.components.diffraction import diffraction_amplitude_norm
from tidy3d.components.grid.grid import Coords, Grid
from tidy3d.components.medium import Medium, MediumType
from tidy3d.components.monitor import (
    AstigmaticGaussianOverlapMonitor,
    AuxFieldTimeMonitor,
    DiffractionMonitor,
    DirectivityMonitor,
    FieldMonitor,
    FieldProjectionAngleMonitor,
    FieldProjectionCartesianMonitor,
    FieldProjectionKSpaceMonitor,
    FieldProjectionSurface,
    FieldTimeMonitor,
    FluxMonitor,
    FluxTimeMonitor,
    GaussianOverlapMonitor,
    MediumMonitor,
    ModeMonitor,
    ModeSolverMonitor,
    PermittivityMonitor,
    SurfaceFieldMonitor,
    SurfaceFieldTimeMonitor,
)
from tidy3d.components.source.current import CustomCurrentSource
from tidy3d.components.source.field import CustomFieldSource
from tidy3d.components.source.time import GaussianPulse
from tidy3d.components.types import (
    ArrayFloat1D,
    Coordinate,
    EpsSpecType,
    Symmetry,
    UnitsZBF,
)
from tidy3d.components.types.monitor import MonitorType
from tidy3d.components.validators import (
    enforce_monitor_fields_present,
    required_if_symmetry_present,
)
from tidy3d.constants import C_0, ETA_0, MICROMETER, UnitScaling, fp_eps
from tidy3d.exceptions import DataError, SetupError, Tidy3dNotImplementedError, ValidationError
from tidy3d.log import log

from .data_array import (
    DataArray,
    DiffractionDataArray,
    EMEFreqModeDataArray,
    FieldProjectionAngleDataArray,
    FieldProjectionCartesianDataArray,
    FieldProjectionKSpaceDataArray,
    FluxDataArray,
    FluxTimeDataArray,
    FreqDataArray,
    FreqModeDataArray,
    GroupIndexDataArray,
    MixedModeDataArray,
    ModeAmpsDataArray,
    ModeDispersionDataArray,
    TimeDataArray,
    _TracedDataset,
)
from .dataset import (
    AbstractFieldDataset,
    AuxFieldTimeDataset,
    ElectromagneticFieldDataset,
    ElectromagneticSurfaceFieldDataset,
    FieldDataset,
    FieldTimeDataset,
    MediumDataset,
    ModeSolverDataset,
    PermittivityDataset,
)

if TYPE_CHECKING:
    from os import PathLike
    from typing import Literal, SupportsComplex

    from numpy.typing import NDArray
    from pandas import DataFrame

    from tidy3d.compat import Self
    from tidy3d.components.geometry.base import Box
    from tidy3d.components.mode_spec import ModeSortSpec, ModeSpec
    from tidy3d.components.source.base import Source
    from tidy3d.components.source.current import PointDipole
    from tidy3d.components.source.field import ModeSource, PlaneWave
    from tidy3d.components.source.time import SourceTimeType
    from tidy3d.components.types import (
        ArrayFloat2D,
        Direction,
        EMField,
        FreqArray,
        PolarizationBasis,
        Size,
        TrackFreq,
    )

    from .data_array import ModeIndexDataArray, ScalarFieldDataArray, ScalarFieldTimeDataArray
    from .dataset import Dataset
    from .unstructured.surface import TriangularSurfaceDataset

Coords1D = ArrayFloat1D

# how much to shift the adjoint field source for 0-D axes dimensions
SHIFT_VALUE_ADJ_FLD_SRC = 1e-5
AXIAL_RATIO_CAP = 1e5
# At this sampling rate, the computed area of a sphere is within ~1% of the true value.
MIN_ANGULAR_SAMPLES_SPHERE = 10
MODE_INTERP_EXTRAPOLATION_TOLERANCE = 1e-2

GRID_CORRECTION_TYPE = Union[
    float,
    FreqDataArray,
    TimeDataArray,
    FreqModeDataArray,
    EMEFreqModeDataArray,
]


class MonitorData(AbstractMonitorData, ABC):
    """
    Abstract base class of objects that store data pertaining to a single :class:`.monitor`.
    """

    monitor: MonitorType = Field(
        title="Monitor",
        description="Monitor associated with the data.",
        discriminator=TYPE_TAG_STR,
    )

    @property
    def symmetry_expanded(self) -> MonitorData:
        """Return self with symmetry applied."""
        return self

    def normalize(self, source_spectrum_fn: Callable[[float], complex]) -> Dataset:
        """Return copy of self after normalization is applied using source spectrum function."""
        return self.copy()

    def scale_fields_by_freq_array(
        self, freq_array: FreqDataArray, method: Optional[str] = None
    ) -> MonitorData:
        """Scale fields in :class:`.MonitorData` by an array of values stored in a :class:`.FreqDataArray`.

        Parameters
        ----------
        freq_array : FreqDataArray
            Array containing the scaling factors in the frequency domain.
        method : str = None
            Interpolation method to use when selecting frequency values. If None, uses default xarray
            method. Passed to xarray's sel() method.

        Returns
        -------
        :class:`.MonitorData`
            A new instance of :class:`.MonitorData` with scaled field values.
        """

        # Reuse the normalize method, so we need the inverse of the scaling amplitude
        def amplitude_fn(freq: list[float]) -> complex:
            return 1.0 / freq_array.sel(f=freq, method=method).values

        return self.normalize(amplitude_fn)

    def _make_adjoint_sources(self, dataset_names: list[str], fwidth: float) -> list[Source]:
        """Generate adjoint sources for this ``MonitorData`` instance."""

        # TODO: if there's data in the MonitorData, but no adjoint source, then
        # user is trying to differentiate something that is un-supported by us
        # warn?

        return []

    @staticmethod
    def flip_direction(direction: Union[str, DataArray]) -> str:
        """Flip the direction of a string ``('+', '-') -> ('-', '+')``."""
        return _flip_direction(direction)

    @staticmethod
    def get_amplitude(x: Union[DataArray, SupportsComplex]) -> complex:
        """Get the complex amplitude out of some data."""

        if isinstance(x, DataArray):
            x = x.values

        return complex(x)


class AbstractFieldData(MonitorData, AbstractFieldDataset, ABC):
    """Collection of scalar fields with some symmetry properties."""

    monitor: Union[
        FieldMonitor,
        FieldTimeMonitor,
        AuxFieldTimeMonitor,
        PermittivityMonitor,
        ModeMonitor,
        MediumMonitor,
    ]

    symmetry: tuple[Symmetry, Symmetry, Symmetry] = Field(
        (0, 0, 0),
        title="Symmetry",
        description="Symmetry eigenvalues of the original simulation in x, y, and z.",
    )

    symmetry_center: Optional[Coordinate] = Field(
        None,
        title="Symmetry Center",
        description="Center of the symmetry planes of the original simulation in x, y, and z. "
        "Required only if any of the ``symmetry`` field are non-zero.",
    )
    grid_expanded: Optional[Grid] = Field(
        None,
        title="Expanded Grid",
        description=":class:`.Grid` discretization of the associated monitor in the simulation "
        "which created the data. Required if symmetries are present, as "
        "well as in order to use some functionalities like getting Poynting vector and flux.",
    )

    @model_validator(mode="after")
    def warn_missing_grid_expanded(self) -> Self:
        """If ``grid_expanded`` not provided and fields data is present, warn that some methods
        will break."""
        field_comps = ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]
        if self.grid_expanded is None and any(
            getattr(self, comp) is not None for comp in field_comps
        ):
            log.warning(
                "Monitor data requires 'grid_expanded' to be defined to compute values like "
                "flux, Poynting and dot product with other data."
            )
        return self

    _require_sym_center: Callable[[Any], Any] = required_if_symmetry_present("symmetry_center")
    _require_grid_expanded: Callable[[Any], Any] = required_if_symmetry_present("grid_expanded")

    def _expanded_grid_field_coords(self, field_name: str) -> Coords:
        """Coordinates in the expanded grid corresponding to a given field component."""
        return self.grid_expanded[self.grid_locations[field_name]]

    @property
    def symmetry_expanded(self) -> Self:
        """Return the :class:`.AbstractFieldData` with fields expanded based on symmetry. If
        any symmetry is nonzero (i.e. expanded), the interpolation implicitly creates a copy of the
        data array. However, if symmetry is not expanded, the returned array contains a view of
        the data, not a copy.

        Returns
        -------
        :class:`AbstractFieldData`
            A data object with the symmetry expanded fields.
        """

        if all(sym == 0 for sym in self.symmetry):
            return self

        return self.updated_copy(**self._symmetry_update_dict, deep=False, validate=False)

    @property
    def symmetry_expanded_copy(self) -> Self:
        """Create a copy of the :class:`.AbstractFieldData` with fields expanded based on symmetry.

        Returns
        -------
        :class:`AbstractFieldData`
            A data object with the symmetry expanded fields.
        """

        if all(sym == 0 for sym in self.symmetry):
            return self.copy()

        return self.copy(update=self._symmetry_update_dict)

    @property
    def _symmetry_update_dict(self) -> dict:
        """Dictionary of data fields to create data with expanded symmetry."""

        update_dict: dict[str, Optional[tuple[float, float, float], DataArray]] = {}
        warn_interp = False
        for field_name, scalar_data in self.field_components.items():
            eigenval_fn = self.symmetry_eigenvalues[field_name]

            # get grid locations for this field component on the expanded grid
            field_coords = self._expanded_grid_field_coords(field_name)

            for sym_dim, (sym_val, sym_loc) in enumerate(zip(self.symmetry, self.symmetry_center)):
                dim_name = "xyz"[sym_dim]

                # Continue if no symmetry along this dimension
                if sym_val == 0:
                    continue

                # Get coordinates for this field component on the expanded grid
                coords = field_coords.to_list[sym_dim]
                coords = self.monitor.downsample(coords, axis=sym_dim)

                # Get indexes of coords that lie on the left of the symmetry center
                flip_inds = np.where(coords < sym_loc)[0]

                # Get the symmetric coordinates on the right
                coords_interp = np.copy(coords)
                coords_interp[flip_inds] = 2 * sym_loc - coords[flip_inds]

                # Interpolate. There generally shouldn't be values out of bounds except potentially
                # when handling modes, in which case they should be at the boundary and close to 0.

                # using sel vs interp is faster, and should always be fine
                # if the data is set up correctly such that its colocation
                # matches the monitor colocation settings. If these do not match,
                # then we need to interpolate, which is slower.
                use_sel = (
                    len(scalar_data.coords[dim_name]) == 1
                    or coords_interp[-1] in scalar_data.coords[dim_name]
                )
                if use_sel:
                    scalar_data = scalar_data.sel(**{dim_name: coords_interp}, method="nearest")
                    scalar_data = scalar_data.assign_coords({dim_name: coords})
                else:
                    warn_interp = True
                    no_flip_inds = np.where(coords >= sym_loc)[0]
                    scalar_data_arrays = []
                    if len(scalar_data.coords[dim_name]) == 1:
                        scalar_data = scalar_data.sel(**{dim_name: coords_interp}, method="nearest")
                    else:
                        if len(flip_inds) > 0:
                            scalar_data_flip = scalar_data.interp(
                                **{dim_name: coords_interp[flip_inds][::-1]},
                                method="linear",
                                kwargs={"fill_value": "extrapolate"},
                                assume_sorted=True,
                            ).isel({dim_name: slice(None, None, -1)})
                            scalar_data_flip = scalar_data_flip.assign_coords(
                                {dim_name: coords[flip_inds]}
                            )
                            scalar_data_arrays.append(scalar_data_flip)
                        if len(no_flip_inds) > 0:
                            scalar_data_no_flip = scalar_data.interp(
                                **{dim_name: coords_interp[no_flip_inds]},
                                method="linear",
                                kwargs={"fill_value": "extrapolate"},
                                assume_sorted=True,
                            )
                            scalar_data_arrays.append(scalar_data_no_flip)
                        scalar_data = xr.concat(scalar_data_arrays, dim=dim_name)

                # apply the symmetry eigenvalue (if defined) to the flipped values
                if eigenval_fn is not None:
                    sym_eigenvalue = eigenval_fn(sym_dim)
                    scalar_data = scalar_data.multiply_at(
                        value=sym_val * sym_eigenvalue, coord_name=dim_name, indices=flip_inds
                    )

            # assign the final scalar data to the update_dict
            update_dict[field_name] = scalar_data

        update_dict.update({"symmetry": (0, 0, 0), "symmetry_center": None})

        if warn_interp:
            log.warning(
                "Interpolating 'ElectromagneticFieldData'. This may be due to "
                "mismatch between monitor colocation and data colocation, "
                "and can lead to performance issues."
            )

        return update_dict

    def at_coords(self, coords: Coords) -> xr.Dataset:
        """Colocate data to some supplied coordinates. This is a convenience method that wraps
        ``colocate``, and skips dimensions for which the data has a single data point only
        (``colocate`` will error in that case.) If the coords are out of bounds for the data
        otherwise, an error will still be produced.

        Parameters
        ----------
        coords : :class:`Coords`
            Coordinates in x, y and z to colocate to.

        Returns
        -------
        xarray.Dataset
            Dataset containing all of the fields in the data interpolated to boundary locations on
            the Yee grid.
        """

        # pass coords if each of the scalar field data have more than one coordinate along a dim
        xyz_kwargs = {}
        for dim, coords_dim in zip("xyz", (coords.x, coords.y, coords.z)):
            scalar_data = list(self.field_components.values())
            coord_lens = [len(data.coords[dim]) for data in scalar_data]
            if all(ncoords > 1 for ncoords in coord_lens):
                xyz_kwargs[dim] = coords_dim

        return self.colocate(**xyz_kwargs)


class ElectromagneticFieldData(AbstractFieldData, ElectromagneticFieldDataset, ABC):
    """Collection of electromagnetic fields."""

    grid_primal_correction: GRID_CORRECTION_TYPE = Field(
        1.0,
        title="Field correction factor",
        description="Correction factor that needs to be applied for data corresponding to a 2D "
        "monitor to take into account the finite grid in the normal direction in the simulation in "
        "which the data was computed. The factor is applied to fields defined on the primal grid "
        "locations along the normal direction.",
    )
    grid_dual_correction: GRID_CORRECTION_TYPE = Field(
        1.0,
        title="Field correction factor",
        description="Correction factor that needs to be applied for data corresponding to a 2D "
        "monitor to take into account the finite grid in the normal direction in the simulation in "
        "which the data was computed. The factor is applied to fields defined on the dual grid "
        "locations along the normal direction.",
    )

    def _expanded_grid_field_coords(self, field_name: str) -> Coords:
        """Coordinates in the expanded grid corresponding to a given field component."""
        if self.monitor.colocate:
            bounds_dict = self.grid_expanded.boundaries.to_dict
            return Coords(**{key: val[:-1] for key, val in bounds_dict.items()})
        return self.grid_expanded[self.grid_locations[field_name]]

    @property
    def _grid_correction_dict(self) -> dict[str, GRID_CORRECTION_TYPE]:
        """Return the primal and dual finite grid correction factors as a dictionary."""
        return {
            "grid_primal_correction": self.grid_primal_correction,
            "grid_dual_correction": self.grid_dual_correction,
        }

    @property
    def _normal_dim(self) -> str:
        """For a 2D monitor data, return the name of the normal dimension. Raise if cannot
        confirm that the associated monitor is 2D."""
        if len(self.monitor.zero_dims) != 1:
            raise DataError("Data must be 2D to get normal dimension.")
        normal_dim = "xyz"[self.monitor.size.index(0)]
        return normal_dim

    @property
    def _tangential_dims(self) -> list[str]:
        """For a 2D monitor data, return the names of the tangential dimensions. Raise if cannot
        confirm that the associated monitor is 2D."""
        if len(self.monitor.zero_dims) != 1:
            raise DataError("Data must be 2D to get tangential dimensions.")
        tangential_dims = ["x", "y", "z"]
        tangential_dims.pop(self.monitor.zero_dims[0])

        return tangential_dims

    def _find_enclosing_boundary(
        self, field_coord: float, boundaries: np.ndarray, side: str
    ) -> float:
        """Find the grid boundary just outside a field coordinate.

        Parameters
        ----------
        field_coord : float
            The field coordinate to find enclosing boundary for.
        boundaries : np.ndarray
            Array of grid boundaries from grid_expanded.
        side : str
            "lower" to find boundary below/at, "upper" to find boundary above/at.

        Returns
        -------
        float
            The grid boundary just outside the field coordinate.
        """
        if side == "lower":
            idx = np.searchsorted(boundaries, field_coord, side="right") - 1
            return boundaries[max(0, idx)]
        else:  # "upper"
            idx = np.searchsorted(boundaries, field_coord, side="left")
            return boundaries[min(len(boundaries) - 1, idx)]

    def _diff_area_at_yee_positions(
        self, truncate_to_monitor_bounds: bool = False
    ) -> tuple[DataArray, DataArray, DataArray, DataArray]:
        """Return differential area elements at Yee grid stagger positions.

        The four returned DataArrays correspond to differential areas at the four
        staggered Yee grid locations for field components:
        - dS_EuHv: area elements at Eu/Hv positions = cell_dim1 x dual_dim2
        - dS_EvHu: area elements at Ev/Hu positions = dual_dim1 x cell_dim2
        - dS_Ew: area elements at Ew positions = dual_dim1 x dual_dim2
        - dS_Hw: area elements at Hw positions = cell_dim1 x cell_dim2

        Parameters
        ----------
        truncate_to_monitor_bounds : bool = False
            If True, clamp integration region to monitor bounds (for flux calculations).
            If False, use grid_expanded bounds enclosing field data (for dot/outer_dot).
            When monitor bounds are np.inf, always uses grid_expanded fallback.

        Returns
        -------
        tuple[DataArray, DataArray, DataArray, DataArray]
            A tuple of (dS_EuHv, dS_EvHu, dS_Ew, dS_Hw) DataArrays with differential
            area elements for Yee grid integration. Each DataArray has dimensions
            corresponding to the two tangential dimensions of the monitor plane.
        """
        if not self.grid_expanded:
            raise DataError(
                "Monitor data requires 'grid_expanded' to compute Yee grid integration sizes."
            )

        _, plane_inds = self.monitor.pop_axis([0, 1, 2], self.monitor.size.index(0.0))
        dims = ["x", "y", "z"]
        mnt_bounds = np.array(self.monitor.bounds)

        cell_sizes = {}
        dual_sizes = {}
        # Use full grid boundaries (not colocation_boundaries) for finding integration limits.
        # This is needed because colocation_boundaries drops first/last for non-colocated monitors,
        # but the actual field data may extend to those edges.
        full_bounds = self.grid_expanded.boundaries.to_dict

        for axis in plane_inds:
            dim = dims[axis]
            # Use full grid boundaries, dropping last boundary
            grid_boundaries = full_bounds[dim][:-1]

            # Do not apply the spurious dl along a dimension where the simulation is 2D.
            # Instead, we just set the boundaries such that the cell size along the zero dimension is 1,
            # such that quantities like flux will come out in units of W / um.
            if grid_boundaries.size == 1:
                dual_sizes[dim] = np.array([1.0])
                cell_sizes[dim] = np.array([1.0])
                continue

            mnt_min = mnt_bounds[0, axis]
            mnt_max = mnt_bounds[1, axis]

            # Compute field data coordinates from grid_expanded boundaries
            full_boundaries = full_bounds[dim]
            field_data_centers = (full_boundaries[:-1] + full_boundaries[1:]) / 2
            field_data_boundaries = full_boundaries[:-1]

            # Determine integration bounds
            if truncate_to_monitor_bounds:
                # Use monitor bounds, handling inf by finding grid boundary outside field data
                if np.isinf(mnt_min):
                    integration_min = self._find_enclosing_boundary(
                        field_data_centers[0], full_boundaries, "lower"
                    )
                else:
                    integration_min = mnt_min

                if np.isinf(mnt_max):
                    integration_max = self._find_enclosing_boundary(
                        field_data_centers[-1], full_boundaries, "upper"
                    )
                else:
                    integration_max = mnt_max
            else:
                # Use grid_expanded bounds that enclose all field data
                integration_min = self._find_enclosing_boundary(
                    field_data_centers[0], full_boundaries, "lower"
                )
                integration_max = self._find_enclosing_boundary(
                    field_data_centers[-1], full_boundaries, "upper"
                )

            # Build coordinate arrays for size calculation
            centers = np.concatenate([[integration_min], field_data_centers])
            boundaries = np.concatenate([field_data_boundaries, [integration_max]])

            # Dual sizes: distances between cell centers, clamped
            dual_coords = np.clip(centers, integration_min, integration_max)
            dual_sizes[dim] = dual_coords[1:] - dual_coords[:-1]

            # Cell/primal sizes: distances between boundaries, clamped
            cell_coords = np.clip(boundaries, integration_min, integration_max)
            cell_sizes[dim] = cell_coords[1:] - cell_coords[:-1]

        dim1 = self._tangential_dims[0]
        dim2 = self._tangential_dims[1]
        dS_EuHv = np.outer(cell_sizes[dim1], dual_sizes[dim2])
        dS_EvHu = np.outer(dual_sizes[dim1], cell_sizes[dim2])
        dS_Ew = np.outer(dual_sizes[dim1], dual_sizes[dim2])
        dS_Hw = np.outer(cell_sizes[dim1], cell_sizes[dim2])

        return (
            DataArray(dS_EuHv, dims=self._tangential_dims),
            DataArray(dS_EvHu, dims=self._tangential_dims),
            DataArray(dS_Ew, dims=self._tangential_dims),
            DataArray(dS_Hw, dims=self._tangential_dims),
        )

    @property
    def colocation_boundaries(self) -> Coords:
        """Coordinates to be used for colocation of the data to grid boundaries."""

        if not self.grid_expanded:
            raise DataError(
                "Monitor data requires 'grid_expanded' to be defined in order to "
                "compute colocation coordinates."
            )

        # Get boundaries from the expanded grid
        grid_bounds = self.grid_expanded.boundaries.to_dict

        # Non-colocating monitors can only colocate starting from the first boundary
        # (unless there's a single data point, in which case data has already been snapped).
        # Regardless of colocation, we also drop the last boundary.
        colocate_bounds = {}
        for dim, bounds in grid_bounds.items():
            cbs = bounds[:-1]
            if not self.monitor.colocate and cbs.size > 1:
                cbs = cbs[1:]
            colocate_bounds[dim] = cbs

        return Coords(**colocate_bounds)

    @property
    def colocation_centers(self) -> Coords:
        """Coordinates to be used for colocation of the data to grid centers."""
        colocate_centers = {}
        for dim, coords in self.colocation_boundaries.to_dict.items():
            colocate_centers[dim] = (coords[1:] + coords[:-1]) / 2

        return Coords(**colocate_centers)

    @property
    def _plane_grid_boundaries(self) -> tuple[Coords1D, Coords1D]:
        """For a 2D monitor data, return the boundaries of the in-plane grid to be used to compute
        differential area and to colocate fields if needed."""
        if np.any(np.array(self.monitor.interval_space) > 1):
            raise Tidy3dNotImplementedError(
                "Cannot determine grid boundaries corresponding to "
                "down-sampled monitor data ('interval_space' > 1 along a direction)."
            )
        dim1, dim2 = self._tangential_dims
        bounds_dict = self.colocation_boundaries.to_dict
        return (bounds_dict[dim1], bounds_dict[dim2])

    @property
    def _plane_grid_centers(self) -> tuple[Coords1D, Coords1D]:
        """For 2D monitor data, return the centers of the in-plane grid"""
        return [(bs[1:] + bs[:-1]) / 2 for bs in self._plane_grid_boundaries]

    @property
    def _diff_area(self) -> DataArray:
        """For a 2D monitor data, return the area of each cell in the plane, for use in numerical
        integrations. This assumes that data is colocated to grid boundaries, and uses the
        difference in the surrounding grid centers to compute the area.
        """

        # Monitor values are interpolated to bounds
        bounds = self._plane_grid_boundaries
        # Coords to compute cell sizes around the interpolation locations
        coords = [bs.copy() for bs in self._plane_grid_centers]

        # Append the first and last boundary
        _, plane_inds = self.monitor.pop_axis([0, 1, 2], self.monitor.size.index(0.0))
        coords[0] = np.array([bounds[0][0], *coords[0].tolist(), bounds[0][-1]])
        coords[1] = np.array([bounds[1][0], *coords[1].tolist(), bounds[1][-1]])

        """Truncate coords to monitor boundaries. This implicitly makes extra pixels which may be
        present have size 0 and so won't be included in the integration. For pixels intersected
        by the monitor edge, the size is truncated to the part covered by the monitor. When using
        the differential area sizes defined in this way together with integrand values
        defined at cell boundaries, the integration is equivalent to trapezoidal rule with the first
        and last values interpolated to the exact monitor start/end location, if the integrand
        is zero outside of the monitor geometry. This should usually be the case for flux and dot
        computations"""
        mnt_bounds = np.array(self.monitor.bounds)
        mnt_bounds = mnt_bounds[:, plane_inds].T
        coords[0][np.argwhere(coords[0] < mnt_bounds[0, 0])] = mnt_bounds[0, 0]
        coords[0][np.argwhere(coords[0] > mnt_bounds[0, 1])] = mnt_bounds[0, 1]
        coords[1][np.argwhere(coords[1] < mnt_bounds[1, 0])] = mnt_bounds[1, 0]
        coords[1][np.argwhere(coords[1] > mnt_bounds[1, 1])] = mnt_bounds[1, 1]

        # Do not apply the spurious dl along a dimension where the simulation is 2D.
        # Instead, we just set the boundaries such that the cell size along the zero dimension is 1,
        # such that quantities like flux will come out in units of W / um.
        sizes_dim0 = coords[0][1:] - coords[0][:-1] if bounds[0].size > 1 else [1.0]
        sizes_dim1 = coords[1][1:] - coords[1][:-1] if bounds[1].size > 1 else [1.0]

        return DataArray(np.outer(sizes_dim0, sizes_dim1), dims=self._tangential_dims)

    def _tangential_corrected(self, fields: dict[str, DataArray]) -> dict[str, DataArray]:
        """For a 2D monitor data, extract the tangential components from fields and orient them
        such that the third component would be the normal axis. This just means that the H field
        gets an extra minus sign if the normal axis is ``"y"``. Raise if any of the tangential
        field components is missing.

        The finite grid correction is also applied, so the intended use of these fields is in
        poynting, flux, and dot-like methods. The normal coordinate is dropped from the field data.
        """

        if len(self.monitor.zero_dims) != 1:
            raise DataError("Data must be 2D to get tangential fields.")

        # Tangential field components
        tan_dims = self._tangential_dims
        components = [fname + dim for fname in "EH" for dim in tan_dims]

        normal_dim = self._normal_dim

        tan_fields = {}
        for component in components:
            if component not in fields:
                raise DataError(f"Tangential field component '{component}' missing in field data.")

            correction = 1

            # sign correction to H
            if normal_dim == "y" and component[0] == "H":
                correction *= -1

            # finite grid correction to all fields
            eig_val = self.symmetry_eigenvalues[component](normal_dim)
            if eig_val < 0:
                correction *= self.grid_dual_correction
            else:
                correction *= self.grid_primal_correction

            field_squeezed = fields[component].squeeze(dim=normal_dim, drop=True)
            # TODO DataArray broadcasting here is a slow portion of the dot method
            tan_fields[component] = field_squeezed * correction

        return tan_fields

    @property
    def _tangential_fields(self) -> dict[str, DataArray]:
        """For a 2D monitor data, get the tangential E and H fields in the 2D plane grid.  Fields
        are oriented such that the third component would be the normal axis. This just means that
        the H field gets an extra minus sign if the normal axis is ``"y"``.

        Note
        ----
            The finite grid correction factors are applied and symmetry is expanded.
        """
        return self._tangential_corrected(self.symmetry_expanded.field_components)

    @property
    def _colocated_fields(self) -> dict[str, DataArray]:
        """For a 2D monitor data, get all E and H fields colocated to the cell boundaries in the 2D
        plane grid, with symmetries expanded.
        """

        field_components = self.symmetry_expanded.field_components

        if self.monitor.colocate:
            return field_components

        # Interpolate field components to cell boundaries
        interp_dict = {"assume_sorted": True}
        for dim, bounds in zip(self._tangential_dims, self._plane_grid_boundaries):
            if bounds.size > 1:
                interp_dict[dim] = bounds

        colocated_fields = {key: val.interp(**interp_dict) for key, val in field_components.items()}
        return colocated_fields

    @property
    def _colocated_tangential_fields(self) -> dict[str, DataArray]:
        """For a 2D monitor data, get the tangential E and H fields colocated to the cell boundaries
        in the 2D plane grid.  Fields are oriented such that the third component would be the normal
        axis. This just means that the H field gets an extra minus sign if the normal axis is
        ``"y"``. Raise if any of the tangential field components is missing.

        Note
        ----
            The finite grid correction factors are applied and symmetry is expanded.
        """
        return self._tangential_corrected(self._colocated_fields)

    @property
    def grid_corrected_copy(self) -> ElectromagneticFieldData:
        """Return a copy of self with grid correction factors applied (if necessary) and symmetry
        expanded."""
        field_data = self.symmetry_expanded_copy
        if len(self.monitor.zero_dims) != 1:
            return field_data

        normal_dim = self._normal_dim
        update = {"grid_primal_correction": 1.0, "grid_dual_correction": 1.0}
        for field_name, field in field_data.field_components.items():
            eig_val = self.symmetry_eigenvalues[field_name](normal_dim)
            if eig_val < 0:
                update[field_name] = field * self.grid_dual_correction
            else:
                update[field_name] = field * self.grid_primal_correction
        return field_data.copy(deep=False, update=update)

    @property
    def intensity(self) -> ScalarFieldDataArray:
        """Return the sum of the squared absolute electric field components."""
        self._check_fields_stored(["Ex", "Ey", "Ez"])

        drop_dims = ["xyz"[dim] for dim in self.monitor.zero_dims]
        fields = self._colocated_fields
        components = ("Ex", "Ey", "Ez")
        if any(cmp not in fields for cmp in components):
            raise KeyError("Can't compute intensity, all E field components must be present.")
        intensity = sum(fields[cmp].abs ** 2 for cmp in components)
        return intensity.squeeze(dim=drop_dims, drop=True)

    @property
    def complex_poynting(self) -> ScalarFieldDataArray:
        """Time-averaged Poynting vector for frequency-domain data associated to a 2D monitor,
        projected to the direction normal to the monitor plane."""

        # Tangential fields are ordered as E1, E2, H1, H2
        tan_fields = self._colocated_tangential_fields
        dim1, dim2 = self._tangential_dims

        e1 = tan_fields["E" + dim1]
        e2 = tan_fields["E" + dim2]
        h1 = tan_fields["H" + dim1]
        h2 = tan_fields["H" + dim2]

        e1_h2 = e1 * h2.conj()
        e2_h1 = e2 * h1.conj()

        e_x_h_star = e1_h2 - e2_h1
        return 0.5 * e_x_h_star

    @property
    def poynting(self) -> ScalarFieldDataArray:
        """Time-averaged Poynting vector for frequency-domain data associated to a 2D monitor,
        projected to the direction normal to the monitor plane."""
        return self.complex_poynting.real

    def _prepare_fields_for_flux(
        self,
        fields: dict[str, DataArray],
    ) -> tuple[dict, dict[str, np.ndarray]]:
        """Make numpy arrays transposed for use in flux calculations.

        Final arrays have spatial dimensions last: (..., Nu, Nv).
        """
        tangential_dims = self._tangential_dims
        test_field = next(iter(fields.values()))

        # Non-spatial dims first (f, optionally mode_index), then spatial
        non_spatial = [d for d in test_field.dims if d not in tangential_dims]
        dim_order = (*non_spatial, *tangential_dims)

        prepped_fields = {key: field.transpose(*dim_order).values for key, field in fields.items()}

        non_spatial_dims = [d for d in test_field.dims if d not in tangential_dims]
        final_coords = {d: test_field.coords[d].values for d in non_spatial_dims}

        return final_coords, prepped_fields

    def package_flux_results(self, flux_values: DataArray) -> Any:
        """How to package flux based on the coordinates present in the data."""
        # Choose appropriate data array type based on coordinates
        if "mode_index" in flux_values.dims:
            return FreqModeDataArray(flux_values)
        return FluxDataArray(flux_values)

    def _compute_complex_flux(self) -> Union[FluxDataArray, FreqModeDataArray]:
        """Compute complex flux."""

        if self.monitor.use_colocated_integration:
            fields = self._colocated_tangential_fields
            dS = self._diff_area.to_numpy()
            dS_numpy = (dS, dS)
        else:
            fields = self._tangential_fields
            dS_EuHv, dS_EvHu, _, _ = self._diff_area_at_yee_positions(
                truncate_to_monitor_bounds=True
            )
            dS_numpy = (dS_EuHv.to_numpy(), dS_EvHu.to_numpy())

        final_coords, prepped_fields = self._prepare_fields_for_flux(fields)

        u, v = self._tangential_dims
        E = (prepped_fields["E" + u], prepped_fields["E" + v])
        H = (prepped_fields["H" + u], prepped_fields["H" + v])

        flux_result = _complex_power_flow_numpy(E, H, dS_numpy)

        if "mode_index" in final_coords:
            return FreqModeDataArray(flux_result, coords=final_coords)
        return FluxDataArray(flux_result, coords=final_coords)

    @cached_property
    def complex_flux(self) -> Union[FluxDataArray, FreqModeDataArray]:
        """Complex flux for data corresponding to a 2D monitor."""
        return self._compute_complex_flux()

    @cached_property
    def flux(self) -> Union[FluxDataArray, FreqModeDataArray]:
        """Flux for data corresponding to a 2D monitor."""
        return self.complex_flux.real

    @cached_property
    def mode_area(self) -> FreqModeDataArray:
        r"""Effective mode area corresponding to a 2D monitor.

        .. math:

           \frac{\left(\int |E|^2 \, {\rm d}S\right)^2}{\int |E|^4 \, {\rm d}S}
        """
        intensity = self.intensity
        # integrate over the plane
        d_area = self._diff_area
        num = (intensity * d_area).sum(dim=d_area.dims) ** 2
        den = (intensity**2 * d_area).sum(dim=d_area.dims)

        area = num / den
        if hasattr(self.monitor, "mode_spec"):
            area *= np.cos(self.monitor.mode_spec.angle_theta)

        return FreqModeDataArray(area)

    def _bounding_box_mask(self, bounding_box: Box) -> DataArray:
        """Create a mask selecting cells whose centers lie within ``bounding_box``."""

        tan_dims = self._tangential_dims
        intensity = self.intensity
        coords = {dim: intensity.coords[dim].values for dim in tan_dims}

        lower, upper = bounding_box.bounds
        axis_indices = ["xyz".index(dim) for dim in tan_dims]

        masks_1d = []
        for dim, axis_idx in zip(tan_dims, axis_indices):
            coord_vals = coords[dim]
            lower_bound = lower[axis_idx]
            upper_bound = upper[axis_idx]
            masks_1d.append((coord_vals >= lower_bound) & (coord_vals <= upper_bound))

        if len(masks_1d) != 2:
            raise DataError("Bounding box masking currently supports planar monitors only.")

        mask_values = (masks_1d[0][:, None] & masks_1d[1][None, :]).astype(float)
        mask = DataArray(mask_values, coords={dim: coords[dim] for dim in tan_dims}, dims=tan_dims)
        return mask

    def _validate_bounding_box_intersection(self, bounding_box: Box) -> None:
        """Ensure bounding box intersects the monitor plane."""

        zero_dims = self.monitor.zero_dims
        if len(zero_dims) != 1:
            raise DataError("Bounding box fill fraction requires a planar monitor.")

        normal_axis = zero_dims[0]
        plane_coord = self.monitor.center[normal_axis]
        lower, upper = bounding_box.bounds
        lower_bound = lower[normal_axis]
        upper_bound = upper[normal_axis]
        tol = fp_eps
        if plane_coord < lower_bound - tol or plane_coord > upper_bound + tol:
            raise ValidationError(
                "Bounding box must intersect the monitor plane when using 'fill_fraction_box'."
            )

    def fill_fraction(self, bounding_box: Box) -> FreqModeDataArray:
        """Return the field-energy fill fraction within ``bounding_box``.
        The fill fraction is defined as the ratio between the integrated field intensity inside
        the bounding box and the total integrated intensity over the monitor plane.

        Parameters
        ----------
        bounding_box : Box
            The bounding box used to compute the fill fraction.

        Returns
        -------
        FreqModeDataArray
            Fill fraction values for each frequency and mode index.
        """

        self._check_fields_stored(["Ex", "Ey", "Ez"])
        self._validate_bounding_box_intersection(bounding_box)

        intensity = self.intensity
        area = self._diff_area
        mask = self._bounding_box_mask(bounding_box)

        weighted_total = (intensity * area).sum(dim=area.dims)
        weighted_box = (intensity * mask * area).sum(dim=area.dims)

        fill_values = (weighted_box / weighted_total.where(weighted_total != 0)).fillna(0.0)

        return FreqModeDataArray(fill_values)

    @cached_property
    def fill_fraction_box(self) -> FreqModeDataArray:
        """Convenience accessor using the :class:`~tidy3d.Box` defined on ``sort_spec``.

        The component of the box along the propagation axis does not influence the fill fraction,
        but the box must intersect the monitor plane.
        """

        sort_spec = getattr(self.monitor.mode_spec, "sort_spec", None)
        bounding_box = None if sort_spec is None else sort_spec.bounding_box
        if bounding_box is None:
            raise DataError(
                "ModeSortSpec.bounding_box must be set to access 'fill_fraction_box' metric."
            )
        return self.fill_fraction(bounding_box)

    def _prepare_fields_for_dot(
        self,
        fields_self: dict[str, DataArray],
        fields_other: dict[str, DataArray],
    ) -> tuple[dict, dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Align coordinates and convert to numpy for "dot" overlap calculations.

        Both returned field dicts have 4-D arrays: ``(N_f, N_modes, Nu, Nv)``.
        When a dataset lacks ``mode_index``, a size-1 dummy dimension is inserted
        (numpy broadcasting handles the mismatch).

        Coordinate alignment:
        - If other has length 1 along ``f`` or ``mode_index``, broadcast to match
          self's coordinate values.
        - Otherwise, the intersection of coordinate values is used.
        """
        tangential_dims = self._tangential_dims
        test_field = "E" + tangential_dims[0]

        # Get coords from both datasets
        self_coords = fields_self[test_field].coords
        other_coords = fields_other[test_field].coords
        self_has_mode = "mode_index" in self_coords
        other_has_mode = "mode_index" in other_coords

        # Compute selections for frequency (always present in both)
        f_sel_self, f_sel_other, final_freqs = _get_broadcast_selection(
            self_coords["f"].values, other_coords["f"].values
        )

        # Compute selections for mode_index
        if self_has_mode and other_has_mode:
            m_sel_self, m_sel_other, final_modes = _get_broadcast_selection(
                self_coords["mode_index"].values, other_coords["mode_index"].values
            )
        elif self_has_mode:
            m_sel_self, m_sel_other = slice(None), None
            final_modes = self_coords["mode_index"].values
        elif other_has_mode:
            m_sel_self, m_sel_other = None, slice(None)
            final_modes = other_coords["mode_index"].values
        else:
            m_sel_self, m_sel_other, final_modes = None, None, None

        def prepare_field_dict(
            fields: dict[str, DataArray],
            f_sel: int | slice | np.ndarray,
            m_sel: int | slice | np.ndarray | None,
            has_mode: bool,
        ) -> dict[str, np.ndarray]:
            """Select, expand dims, transpose, extract numpy."""
            result = {}
            for key, field in fields.items():
                da = field.isel(f=f_sel)
                if has_mode and m_sel is not None:
                    da = da.isel(mode_index=m_sel)
                if not has_mode:
                    da = da.expand_dims("mode_index")
                result[key] = da.transpose("f", "mode_index", *tangential_dims).values
            return result

        prepped_fields_self = prepare_field_dict(fields_self, f_sel_self, m_sel_self, self_has_mode)
        prepped_fields_other = prepare_field_dict(
            fields_other, f_sel_other, m_sel_other, other_has_mode
        )

        # Build final_coords dict
        final_coords = {"f": final_freqs}
        if final_modes is not None:
            final_coords["mode_index"] = final_modes

        return final_coords, prepped_fields_self, prepped_fields_other

    def dot(
        self,
        field_data: FieldData | ModeData | ModeSolverData,
        conjugate: bool = True,
        bidirectional: bool = True,
    ) -> FreqDataArray | FreqModeDataArray:
        r"""Dot product (modal overlap) with another :class:`.FieldData` object. Both datasets have
        to be frequency-domain data associated with a 2D monitor.

        When either monitor uses ``colocate=True`` (default) or
        ``use_colocated_integration=True``, the tangential fields from ``field_data`` are
        interpolated onto this object's grid, so the two datasets may have different spatial
        discretizations. Otherwise, both datasets must share the same tangential grid.

        Along the normal direction, the monitor position may differ and is ignored.
        Non-spatial coordinates (``f``, ``mode_index``) are aligned by intersection;
        broadcasting is also supported when the other dataset has size 1 along a coordinate
        dimension.

        The dot product is defined as:

        .. math:

           \frac{1}{4} \int \left( E_0^* \times H_1 + H_0^* \times E_1 \right) \, {\rm d}S

        If ``bidirectional=False``, the dot product is instead:

        .. math:

           \frac{1}{2} \int \left( E_0^* \times H_1 \right) \, {\rm d}S

        Parameters
        ----------
        field_data : :class:`.FieldData` | :class:`.ModeData` | :class:`.ModeSolverData`
            A data instance to compute the dot product with.
        conjugate : bool, optional
            If ``True`` (default), the dot product is defined as above. If ``False``, the definition
            is similar, but without the complex conjugation of the fields.
        bidirectional : bool = True
            If ``True`` (default), computes the symmetric bidirectional overlap:
            ``1/4 * integral(E1* x H2 + H1* x E2) dS``.
            If ``False``, computes just: ``1/2 * integral(E1* x H2) dS``.

        Returns
        -------
        :class:`.FreqDataArray` | :class:`.FreqModeDataArray`
            Data array with the complex-valued modal overlaps.

            - If neither dataset has ``mode_index``: returns :class:`.FreqDataArray`.
            - If either dataset has ``mode_index``: returns :class:`.FreqModeDataArray`.

        Note
        ----
            The dot product with and without conjugation is equivalent (up to a phase) for
            modes in lossless waveguides but differs for modes in lossy materials. In that case,
            the conjugated dot product can be interpreted as the fraction of the power carried by
            the second mode, but modes are not orthogonal with respect to that product
            and the sum of carried power fractions may be different from the total flux.
            In the non-conjugated definition, modes are orthogonal, but the interpretation of the
            dot product as power carried by a given mode is no longer valid.
        """
        use_colocated = (
            self.monitor.use_colocated_integration or field_data.monitor.use_colocated_integration
        )
        if not use_colocated:
            fields_self = self._tangential_fields
            fields_other = field_data._tangential_fields
            if not self._fields_share_tangential_coords(fields_self, fields_other):
                log.warning(
                    "Tangential field coordinates do not match in 'dot'; "
                    "switching to colocated-based computation."
                )
                use_colocated = True

        if use_colocated:
            fields_self = self._colocated_tangential_fields
            fields_other = field_data._interpolated_tangential_fields(self._plane_grid_boundaries)
            d_area = self._diff_area.to_numpy()
            dS_numpy = (d_area, d_area)
        else:
            dS_EuHv, dS_EvHu, _, _ = self._diff_area_at_yee_positions(
                truncate_to_monitor_bounds=False
            )
            dS_numpy = (dS_EuHv.to_numpy(), dS_EvHu.to_numpy())

        # Determine broadcast behavior and final dimensions (returns numpy arrays directly)
        final_coords, prepped_fields_self, prepped_fields_other = self._prepare_fields_for_dot(
            fields_self, fields_other
        )

        # Extract tangential components as tuples (already numpy arrays)
        u, v = self._tangential_dims
        E1 = (prepped_fields_self["E" + u], prepped_fields_self["E" + v])
        H1 = (prepped_fields_self["H" + u], prepped_fields_self["H" + v])
        E2 = (prepped_fields_other["E" + u], prepped_fields_other["E" + v])
        H2 = (prepped_fields_other["H" + u], prepped_fields_other["H" + v])

        dot_result = _dot_numpy(
            E1, H1, E2, H2, dS_numpy, conjugate=conjugate, bidirectional=bidirectional
        )

        # Squeeze out mode_index dimension (axis 1) if not in final_coords
        if "mode_index" not in final_coords:
            dot_result = dot_result.squeeze(axis=1)
            return FreqDataArray(dot_result, coords=final_coords)
        else:
            return FreqModeDataArray(dot_result, coords=final_coords)

    def _fields_share_tangential_coords(
        self,
        fields_a: dict[str, DataArray],
        fields_b: dict[str, DataArray],
    ) -> bool:
        """Check whether two tangential field sets share the same coordinates."""
        for key in fields_a:
            if key not in fields_b:
                return False
            for dim in self._tangential_dims:
                coords_a = fields_a[key].coords[dim].values
                coords_b = fields_b[key].coords[dim].values
                if coords_a.size != coords_b.size:
                    return False
                if not np.allclose(coords_a, coords_b):
                    return False
        return True

    def _tangential_fields_match_coords(self, coords: ArrayFloat2D) -> bool:
        """Check if the tangential fields already match given coords in the tangential plane."""
        for field in self._tangential_fields.values():
            for idim, dim in enumerate(self._tangential_dims):
                if field.coords[dim].values.size != coords[idim].size or not np.all(
                    field.coords[dim].values == coords[idim]
                ):
                    return False
        return True

    def _interpolated_tangential_fields(self, coords: ArrayFloat2D) -> dict[str, DataArray]:
        """For 2D monitors, interpolate this fields to given coords in the tangential plane.

        Parameters
        ----------
        coords : ArrayFloat2D
            Interpolation coords in the monitor's tangential plane.

        Return
        ------
            Dictionary with interpolated fields.
        """
        fields = self._tangential_fields

        # If coords already match, just return the tangential fields directly.
        if self._tangential_fields_match_coords(coords):
            return fields

        # Interpolate if data has more than one coordinate along a dimension
        interp_dict = {"assume_sorted": True}
        # If single coordinate, just sel "nearest", i.e. just propagate the same data everywhere
        sel_dict = {"method": "nearest"}
        for dim, cents in zip(self._tangential_dims, coords):
            if cents.size > 0:
                if list(fields.values())[0].coords[dim].size > 1:
                    interp_dict[dim] = cents
                else:
                    sel_dict[dim] = cents

        kwargs = {"bounds_error": False, "fill_value": 0.0}
        for component, field in fields.items():
            fields[component] = field.interp(kwargs=kwargs, **interp_dict).sel(**sel_dict)

        return fields

    def _prepare_fields_for_outer_dot(
        self,
        fields_self: dict[str, DataArray],
        fields_other: dict[str, DataArray],
    ) -> tuple[dict, dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Align coordinates and convert to numpy for "outer dot" overlap calculations.

        Both returned field dicts have 4-D arrays: ``(N_f, N_modes, Nu, Nv)``.
        When a dataset lacks ``mode_index``, a size-1 dummy dimension is inserted.

        Coordinate alignment:
        - Frequency: intersection only (no broadcasting).
        - Mode index: each side keeps its own modes (no alignment); ``final_coords``
          uses ``mode_index_0`` / ``mode_index_1`` to distinguish them.
        """
        tangential_dims = self._tangential_dims
        test_field = "E" + tangential_dims[0]

        # Get coords from both datasets
        self_coords = fields_self[test_field].coords
        other_coords = fields_other[test_field].coords
        self_has_mode = "mode_index" in self_coords
        other_has_mode = "mode_index" in other_coords

        # Frequency: intersection only (no broadcasting)
        f_sel_self, f_sel_other, common_freqs = _get_intersection_selection(
            self_coords["f"].values, other_coords["f"].values
        )

        def prepare_field_dict(
            fields: dict[str, DataArray],
            f_sel: int | slice | np.ndarray,
            has_mode: bool,
        ) -> dict[str, np.ndarray]:
            """Select freq, expand dims, transpose, extract numpy."""
            result = {}
            for key, field in fields.items():
                da = field.isel(f=f_sel)
                if not has_mode:
                    da = da.expand_dims("mode_index")
                result[key] = da.transpose("f", "mode_index", *tangential_dims).values
            return result

        prepped_fields_self = prepare_field_dict(fields_self, f_sel_self, self_has_mode)
        prepped_fields_other = prepare_field_dict(fields_other, f_sel_other, other_has_mode)

        # Prepare final coords
        final_coords = {"f": common_freqs}

        # Determine mode_index coordinate handling:
        # - Neither has mode_index → just f → FreqDataArray
        # - At least one has mode_index → f + mode_index_0 + mode_index_1 → MixedModeDataArray
        if self_has_mode and other_has_mode:
            final_coords["mode_index_0"] = self_coords["mode_index"].values
            final_coords["mode_index_1"] = other_coords["mode_index"].values
        elif self_has_mode:
            final_coords["mode_index_0"] = self_coords["mode_index"].values
        elif other_has_mode:
            final_coords["mode_index_1"] = other_coords["mode_index"].values

        return final_coords, prepped_fields_self, prepped_fields_other

    def outer_dot(
        self,
        field_data: FieldData | ModeData | ModeSolverData,
        conjugate: bool = True,
        bidirectional: bool = True,
        truncate_to_monitor_bounds: bool = False,
    ) -> FreqDataArray | MixedModeDataArray:
        r"""Outer dot product (pairwise modal overlap matrix) with another :class:`.FieldData`
        object.

        When either monitor uses ``colocate=True`` (default) or
        ``use_colocated_integration=True``, the tangential fields from ``field_data`` are
        interpolated onto this object's grid, so the two datasets may have different spatial
        discretizations. Otherwise, both datasets must share the same tangential grid.

        The calculation is performed for all common frequencies between the two datasets.

        The dot product is defined as:

        .. math:

           \frac{1}{4} \int \left( E_0^* \times H_1 + H_0^* \times E_1 \right) \, {\rm d}S

        If ``bidirectional=False``, the dot product is instead:

        .. math:

           \frac{1}{2} \int \left( E_0^* \times H_1 \right) \, {\rm d}S

        Parameters
        ----------
        field_data : :class:`.FieldData` | :class:`.ModeData` | :class:`.ModeSolverData`
            A data instance to compute the dot product with.
        conjugate : bool = True
            If ``True`` (default), the dot product is defined as above. If ``False``, the definition
            is similar, but without the complex conjugation of the fields.
        bidirectional : bool = True
            If ``True`` (default), computes the symmetric bidirectional overlap:
            ``1/4 * integral(E1* x H2 + H1* x E2) dS``.
            If ``False``, computes just: ``1/2 * integral(E1* x H2) dS``.
        truncate_to_monitor_bounds : bool = False
            Only used in the non-colocated integration path (when ``colocate=False``).
            If ``True``, clamp integration area to monitor bounds, consistent with
            ``complex_flux``. If ``False`` (default), use grid-enclosing bounds.

        Returns
        -------
        :class:`.FreqDataArray` | :class:`.MixedModeDataArray`
            Data array with the complex-valued modal overlaps.

            - If neither dataset has ``mode_index``: returns :class:`.FreqDataArray`.
            - If only ``self`` has ``mode_index``: returns :class:`.MixedModeDataArray` with
              ``mode_index_0`` coordinate.
            - If only ``field_data`` has ``mode_index``: returns :class:`.MixedModeDataArray` with
              ``mode_index_1`` coordinate.
            - If both datasets have ``mode_index``: returns :class:`.MixedModeDataArray` with
              ``mode_index_0`` and ``mode_index_1`` coordinates.

        See also
        --------
        :member:`dot`
        """

        tan_dims = self._tangential_dims

        if not all(a == b for a, b in zip(tan_dims, field_data._tangential_dims)):
            raise DataError("Tangential dimensions must match between the two monitors.")

        use_colocated = (
            self.monitor.use_colocated_integration or field_data.monitor.use_colocated_integration
        )
        if not use_colocated:
            fields_self = self._tangential_fields
            fields_other = field_data._tangential_fields
            if not self._fields_share_tangential_coords(fields_self, fields_other):
                log.warning(
                    "Tangential field coordinates do not match in 'outer_dot'; "
                    "switching to colocated-based computation."
                )
                use_colocated = True

        if use_colocated:
            fields_self = self._colocated_tangential_fields
            fields_other = field_data._interpolated_tangential_fields(self._plane_grid_boundaries)
            d_area = self._diff_area.to_numpy()
            dS_numpy = (d_area, d_area)
        else:
            dS_EuHv, dS_EvHu, _, _ = self._diff_area_at_yee_positions(
                truncate_to_monitor_bounds=truncate_to_monitor_bounds
            )
            dS_numpy = (dS_EuHv.to_numpy(), dS_EvHu.to_numpy())

        # Determine broadcast behavior and final dimensions (returns numpy arrays directly)
        final_coords, prepped_fields_self, prepped_fields_other = (
            self._prepare_fields_for_outer_dot(fields_self, fields_other)
        )

        # Extract tangential components as tuples (already numpy arrays)
        u, v = self._tangential_dims
        E1 = (prepped_fields_self["E" + u], prepped_fields_self["E" + v])
        H1 = (prepped_fields_self["H" + u], prepped_fields_self["H" + v])
        E2 = (prepped_fields_other["E" + u], prepped_fields_other["E" + v])
        H2 = (prepped_fields_other["H" + u], prepped_fields_other["H" + v])
        numpy_result = _outer_dot_numpy(
            E1, H1, E2, H2, dS_numpy, conjugate=conjugate, bidirectional=bidirectional
        )

        # Determine return type based on final_coords
        # numpy_result shape is (n_freq, n_modes_0, n_modes_1)
        has_mode_index_0 = "mode_index_0" in final_coords
        has_mode_index_1 = "mode_index_1" in final_coords

        if has_mode_index_0 and has_mode_index_1:
            # Both have mode_index: return MixedModeDataArray (no squeeze)
            return MixedModeDataArray(numpy_result, coords=final_coords)
        elif has_mode_index_0:
            # Only self has mode_index: squeeze axis 2 and return MixedModeDataArray
            squeezed = numpy_result.squeeze(axis=2)
            return MixedModeDataArray(squeezed, coords=final_coords)
        elif has_mode_index_1:
            # Only other has mode_index: squeeze axis 1 and return MixedModeDataArray
            squeezed = numpy_result.squeeze(axis=1)
            return MixedModeDataArray(squeezed, coords=final_coords)
        else:
            # Neither has mode_index: return FreqDataArray
            squeezed = numpy_result.squeeze(axis=(1, 2))
            return FreqDataArray(squeezed, coords=final_coords)

    @property
    def time_reversed_copy(self) -> FieldData:
        """Make a copy of the data with time-reversed fields."""

        # Time reversal for frequency-domain fields; overwritten in :class:`FieldTimeData`,
        # :class:`ModeData`, and :class:`ModeSolverData`.
        new_data = {}
        for comp, field in self.field_components.items():
            if comp[0] == "H":
                new_data[comp] = -np.conj(field)
            else:
                new_data[comp] = np.conj(field)
        return self.copy(deep=False, update=new_data)

    def _check_fields_stored(self, components: list[str]) -> None:
        """Check that all requested field components are stored in the data."""
        missing_comps = [comp for comp in components if comp not in self.field_components.keys()]
        if len(missing_comps) > 0:
            raise DataError(
                f"Field components {missing_comps} not included in this data object. Use "
                "the 'fields' argument of a field monitor to select which components are stored."
            )

    def translated_copy(self, vector: Coordinate) -> ElectromagneticFieldData:
        """Create a copy of the :class:`.ElectromagneticFieldData` with fields translated
        by the provided vector. Can be used together with ``dot`` or ``outer_dot``
        to compute overlaps between field data at different locations.

        Parameters
        ----------
        vector: :class:`.Coordinate`
            Translation vector to apply to the field data.

        Returns
        -------
        :class:`ElectromagneticFieldData`
            A data object with the translated fields.
        """
        field_kwargs = {}
        for key, val in self.field_components.items():
            coords = dict(val.coords)
            coords["x"] = coords["x"] + vector[0]
            coords["y"] = coords["y"] + vector[1]
            coords["z"] = coords["z"] + vector[2]
            field_kwargs[key] = val.assign_coords(coords)

        symmetry_center = self.symmetry_center
        if symmetry_center is not None:
            symmetry_center = tuple([x + y for (x, y) in zip(symmetry_center, vector)])
        grid_expanded = self.grid_expanded._translated_copy(vector=vector)

        monitor_center = tuple([x + y for (x, y) in zip(self.monitor.center, vector)])
        monitor = self.monitor.updated_copy(center=monitor_center)

        return self.updated_copy(
            monitor=monitor,
            symmetry=self.symmetry,
            symmetry_center=symmetry_center,
            grid_expanded=grid_expanded,
            **self._grid_correction_dict,
            **field_kwargs,
            deep=False,
        )

    def to_zbf(
        self,
        fname: PathLike,
        units: UnitsZBF = "mm",
        background_refractive_index: float = 1,
        n_x: Optional[int] = None,
        n_y: Optional[int] = None,
        freq: Optional[float] = None,
        mode_index: Optional[int] = None,
        r_x: float = 0,
        r_y: float = 0,
        z_x: float = 0,
        z_y: float = 0,
        rec_efficiency: float = 0,
        sys_efficiency: float = 0,
    ) -> tuple[ScalarFieldDataArray, ScalarFieldDataArray]:
        """For a 2D monitor, export the fields to a Zemax Beam File (``.zbf``).

        The mode area is used to approximate the beam waist, which is only valid
        if the beam profile approximates a Gaussian beam.

        Parameters
        ----------
        fname : PathLike
            Full path to the ``.zbf`` file to be written.
        units : UnitsZBF = "mm"
            Spatial units used for the ``.zbf`` file. Options are ``"mm"``, ``"cm"``, ``"in"``, or ``"m"``.
            Defaults to ``"mm"``.
        background_refractive_index : float = 1
            Refractive index of the medium surrounding the monitor. Defaults to ``1``.
        n_x : Optional[int] = None
            Number of field samples along x.
            Must be a power of 2, between 2^5 and 2^13 inclusive per Zemax's requirements.
            Defaults to ``None``, in which case a value is chosen for the user depending on the coordinates in the field data.
        n_y : Optional[int] = None
            Number of field samples along y.
            Must be a power of 2, between 2^5 and 2^13 inclusive per Zemax's requirements.
            Defaults to ``None``, in which case a value is chosen for the user depending on the coordinates in the field data.
        freq : Optional[float] = None
            Field frequency selection. If ``None``, the average of the recorded frequencies is used.
        mode_index : Optional[int] = None
            For :class:`.ModeData`, choose which mode to save.
        r_x : float = 0
            Pilot beam Rayleigh distance in x, um. Defaults to ``0``.
        r_y : float = 0
            Pilot beam Rayleigh distance in y, um. Defaults to ``0``.
        z_x : float = 0
            Pilot beam z position with respect to the waist in x, um. Defaults to ``0``.
        z_y : float = 0
            Pilot beam z position with respect to the waist in y, um. Defaults to ``0``.
        rec_efficiency : float = 0
            Receiver efficiency, zero if fiber coupling is not computed. Defaults to ``0``.
        sys_efficiency : float = 0
            System efficiency, zero if fiber coupling is not computed. Defaults to ``0``.

        Returns
        -------
        tuple[:class:`.ScalarFieldDataArray`,:class:`.ScalarFieldDataArray`]
            The two E field components being exported to ``.zbf``.
        """
        log.warning(
            "'FieldData.to_zbf()' is currently an experimental feature."
            " If any issues are encountered, please contact Flexcompute support 'https://www.flexcompute.com/tidy3d/technical-support/'"
        )

        # Check that appropriate units are used
        if units not in get_args(UnitsZBF):
            raise ValueError("'units' must be either 'mm', 'cm', 'in', or 'm'.")

        # Mode area calculation ensures all E components are present
        mode_area = self.mode_area
        dim1, dim2 = self._tangential_dims

        # Using file-local coordinates x, y for the tangential components
        e_x = self._tangential_fields["E" + dim1]
        e_y = self._tangential_fields["E" + dim2]
        x = e_x.coords[dim1].values
        y = e_x.coords[dim2].values

        # Use the mean frequency if freq is not specified
        if freq is None:
            log.warning(
                "'freq' was not specified for 'FieldData.to_zbf()'. Defaulting to the mean frequency of the dataset."
            )
            freq = np.mean(e_x.coords["f"].values)
        else:
            freq = np.array(freq)

        if freq.size > 1:
            raise ValueError("'freq' must be a single value, not an array.")
        else:
            freq = freq.item()

        # If the data has just one frequency, avoid Nans at the interpolation
        if len(e_x.f) > 1:
            mode_area = mode_area.interp(f=freq)
            e_x = e_x.interp(f=freq)
            e_y = e_y.interp(f=freq)
        else:
            e_x = e_x.isel(f=0, drop=True)
            e_y = e_y.isel(f=0, drop=True)

        # If the data is ModeData, choose one of the modes to save
        if "mode_index" in e_x.coords:
            if mode_index is None:
                raise ValueError("'mode_index' is required for 'ModeData.to_zbf()'")
            mode_area = mode_area.isel(mode_index=mode_index, drop=True)
            e_x = e_x.isel(mode_index=mode_index, drop=True)
            e_y = e_y.isel(mode_index=mode_index, drop=True)

        # Header info
        version = 1
        polarized = 1
        unit_mapping = {"mm": 0, "cm": 1, "in": 2, "m": 3}
        unit_key = unit_mapping[units]
        unit_scaling = UnitScaling[units]
        lda = C_0 / freq * unit_scaling

        # Pilot (reference) beam waist: use the mode area to approximate the expected value
        w_x = (mode_area.item() / np.pi) ** 0.5 * unit_scaling
        w_y = w_x

        # Pilot beam Rayleigh distance (ignored on input)
        r_x *= unit_scaling
        r_y *= unit_scaling

        # Pilot beam z position w.r.t. the waist
        z_x *= unit_scaling
        z_y *= unit_scaling

        # defaults for n_x and n_y
        if n_x is None:
            n_x = 2 ** min(13, max(5, int(np.log2(x.size) + 1)))
            log.warning(
                f"'n_x' was not specified for 'FieldData.to_zbf()'. Defaulting to 'n_x' = {n_x}."
            )
        if n_y is None:
            n_y = 2 ** min(13, max(5, int(np.log2(y.size) + 1)))
            log.warning(
                f"'n_y' was not specified for 'FieldData.to_zbf()'. Defaulting to 'n_y' = {n_y}."
            )

        # Check that requirements are met for n_x and n_y
        # n_x and n_y must be powers of 2
        if (n_x & (n_x - 1)) != 0:
            raise ValueError("'n_x' must be a power of 2.")
        if (n_y & (n_y - 1)) != 0:
            raise ValueError("'n_y' must be a power of 2.")
        # 32 <= n_x and n_y <= 2^13
        if n_x < 32 or n_x > 2**13:
            raise ValueError("'n_x' must be between 2^5 and 2^13, inclusive.")
        if n_y < 32 or n_y > 2**13:
            raise ValueError("'n_y' must be between 2^5 and 2^13, inclusive.")

        # Interpolating coordinates
        x = np.linspace(x.min(), x.max(), n_x)
        y = np.linspace(y.min(), y.max(), n_y)

        # Interpolate fields
        coords = {dim1: x, dim2: y}
        e_x = e_x.interp(coords, assume_sorted=True)
        e_y = e_y.interp(coords, assume_sorted=True)

        # Sampling distance
        d_x = np.mean(np.diff(x)) * unit_scaling
        d_y = np.mean(np.diff(y)) * unit_scaling

        with open(fname, "wb") as fout:
            fout.write(struct.pack("<5I", version, n_x, n_y, polarized, unit_key))
            fout.write(struct.pack("<4I", 0, 0, 0, 0))  # unused values
            fout.write(struct.pack("<8d", d_x, d_y, z_x, r_x, w_x, z_y, r_y, w_y))
            fout.write(
                struct.pack("<4d", lda, background_refractive_index, rec_efficiency, sys_efficiency)
            )
            fout.write(struct.pack("<8d", 0, 0, 0, 0, 0, 0, 0, 0))  # unused values
            for e in (e_x, e_y):
                e_flat = e.values.flatten(order="F")
                # Interweave real and imaginary parts
                e_values = np.ravel(np.column_stack((e_flat.real, e_flat.imag)))
                fout.write(struct.pack(f"<{2 * n_x * n_y}d", *e_values))

        return e_x, e_y

    def _interpolated_copies_if_needed(
        self, other: ElectromagneticFieldData
    ) -> tuple[ElectromagneticFieldData, ElectromagneticFieldData]:
        """Return interpolated copies of self, other if needed (different interp_spec)."""
        mode_spec1 = self.monitor.mode_spec if isinstance(self, ModeSolverData) else None
        mode_spec2 = other.monitor.mode_spec if isinstance(other, ModeSolverData) else None
        if (
            mode_spec1 is not None
            and mode_spec2 is not None
            and self.monitor.mode_spec._same_nontrivial_interp_spec(other=other.monitor.mode_spec)
        ):
            return self, other
        self_copy = self.interpolated_copy if isinstance(self, ModeSolverData) else self
        other_copy = other.interpolated_copy if isinstance(other, ModeSolverData) else other
        return self_copy, other_copy


class FieldData(FieldDataset, ElectromagneticFieldData):
    """
    Data associated with a :class:`.FieldMonitor`: scalar components of E and H fields.

    Notes
    -----

        The data is stored as a `DataArray <https://docs.xarray.dev/en/stable/generated/xarray.DataArray.html>`_
        object using the `xarray <https://docs.xarray.dev/en/stable/index.html>`_ package.

        This dataset can contain all electric and magnetic field components: ``Ex``, ``Ey``, ``Ez``, ``Hx``, ``Hy``,
        and ``Hz``.

    Example
    -------
    >>> from tidy3d import ScalarFieldDataArray
    >>> x = [-1,1,3]
    >>> y = [-2,0,2,4]
    >>> z = [-3,-1,1,3,5]
    >>> f = [2e14, 3e14]
    >>> coords = dict(x=x[:-1], y=y[:-1], z=z[:-1], f=f)
    >>> grid = Grid(boundaries=Coords(x=x, y=y, z=z))
    >>> scalar_field = ScalarFieldDataArray((1+1j) * np.random.random((2,3,4,2)), coords=coords)
    >>> monitor = FieldMonitor(
    ...     size=(2,4,6), freqs=[2e14, 3e14], name='field', fields=['Ex', 'Hz'], colocate=True
    ... )
    >>> data = FieldData(monitor=monitor, Ex=scalar_field, Hz=scalar_field, grid_expanded=grid)

    .. TODO sort out standalone data example.

    See Also
    --------

    **Notebooks:**
        * `Quickstart <../../notebooks/StartHere.html>`_: Usage in a basic simulation flow.
        * `Performing visualization of simulation data <../../notebooks/VizData.html>`_
        * `Advanced monitor data manipulation and visualization <../../notebooks/XarrayTutorial.html>`_
    """

    monitor: FieldMonitor = Field(
        title="Monitor",
        description="Frequency-domain field monitor associated with the data.",
    )

    _contains_monitor_fields = enforce_monitor_fields_present()

    def normalize(self, source_spectrum_fn: Callable[[float], complex]) -> FieldDataset:
        """Return copy of self after normalization is applied using source spectrum function."""
        fields_norm = {}
        for field_name, field_data in self.field_components.items():
            src_amps = source_spectrum_fn(field_data.f)
            fields_norm[field_name] = (field_data / src_amps).astype(field_data.dtype)

        return self.copy(deep=False, update=fields_norm)

    def to_source(
        self, source_time: SourceTimeType, center: Coordinate, size: Size = None, **kwargs: Any
    ) -> CustomFieldSource:
        """Create a :class:`.CustomFieldSource` from the fields stored in the :class:`.FieldData`.

        Parameters
        ----------
        source_time: :class:`.SourceTime`
            Specification of the source time-dependence.
        center: tuple[float, float, float]
            Source center in x, y and z.
        size: tuple[float, float, float]
            Source size in x, y, and z. If not provided, the size of the monitor associated to the
            data is used.
        **kwargs
            Extra keyword arguments passed to :class:`.CustomFieldSource`.

        Returns
        -------
        :class:`.CustomFieldSource`
            Source injecting the fields stored in the :class:`.FieldData`, with other settings as
            provided in the input arguments.
        """

        if not size:
            size = self.monitor.size

        fields = {}
        for name, field in self.symmetry_expanded.field_components.items():
            fields[name] = field.copy()
            for dim, dim_name in enumerate("xyz"):
                coords_shift = field.coords[dim_name] - self.monitor.center[dim]
                fields[name].coords[dim_name] = coords_shift

        dataset = FieldDataset(**fields)
        return CustomFieldSource(
            field_dataset=dataset, source_time=source_time, center=center, size=size, **kwargs
        )

    def _make_adjoint_sources(
        self, dataset_names: list[str], fwidth: float
    ) -> list[CustomCurrentSource]:
        """Converts a :class:`.FieldData` to a list of adjoint current or point sources."""

        sources = []
        source_geo = self.monitor.geometry
        freqs = self.monitor.freqs

        for freq0 in freqs:
            src_field_components = {}
            for name, field_component in self.field_components.items():
                # get the VJP values at frequency and apply adjoint phase
                field_component = field_component.sel(f=freq0)

                # accounts for the effective size of the source when injecting into a
                # simulation with symmetry
                symmetry_factor = np.prod(field_component.values.shape) / np.prod(
                    self.symmetry_expanded.field_components[name].sel(f=freq0).values.shape
                )
                source_data = current_component_data_array(
                    component=name,
                    base_values=field_component.values,
                    spatial_coords={key: np.array(field_component.coords[key]) for key in "xyz"},
                    source_center=source_geo.center,
                    freq=float(freq0),
                    symmetry_factor=float(symmetry_factor),
                )
                if source_data is not None:
                    src_field_components[name] = source_data

            # dont include this source if no data
            if all(fld_cmp is None for fld_cmp in src_field_components.values()):
                continue

            # construct custom Current source
            dataset = FieldDataset(**src_field_components)
            custom_source = CustomCurrentSource(
                center=source_geo.center,
                size=source_geo.size,
                source_time=GaussianPulse(
                    freq0=freq0,
                    fwidth=fwidth,
                ),
                current_dataset=dataset,
                interpolate=True,
            )

            sources.append(custom_source)

        return sources


class FieldTimeData(FieldTimeDataset, ElectromagneticFieldData):
    """
    Data associated with a :class:`.FieldTimeMonitor`: scalar components of E and H fields.

    Notes
    -----

        The data is stored as a `DataArray <https://docs.xarray.dev/en/stable/generated/xarray.DataArray.html>`_
        object using the `xarray <https://docs.xarray.dev/en/stable/index.html>`_ package.

    Example
    -------
    >>> from tidy3d import ScalarFieldTimeDataArray
    >>> x = [-1,1,3]
    >>> y = [-2,0,2,4]
    >>> z = [-3,-1,1,3,5]
    >>> t = [0, 1e-12, 2e-12]
    >>> coords = dict(x=x[:-1], y=y[:-1], z=z[:-1], t=t)
    >>> grid = Grid(boundaries=Coords(x=x, y=y, z=z))
    >>> scalar_field = ScalarFieldTimeDataArray(np.random.random((2,3,4,3)), coords=coords)
    >>> monitor = FieldTimeMonitor(
    ...     size=(2,4,6), interval=100, name='field', fields=['Ex', 'Hz'], colocate=True
    ... )
    >>> data = FieldTimeData(monitor=monitor, Ex=scalar_field, Hz=scalar_field, grid_expanded=grid)
    """

    monitor: FieldTimeMonitor = Field(
        title="Monitor",
        description="Time-domain field monitor associated with the data.",
    )

    _contains_monitor_fields = enforce_monitor_fields_present()

    @property
    def poynting(self) -> ScalarFieldTimeDataArray:
        """Instantaneous Poynting vector for time-domain data associated to a 2D monitor, projected
        to the direction normal to the monitor plane."""

        # Tangential fields are ordered as E1, E2, H1, H2
        tan_fields = self._colocated_tangential_fields
        dim1, dim2 = self._tangential_dims
        e_x_h = np.real(tan_fields["E" + dim1]) * np.real(tan_fields["H" + dim2])
        e_x_h -= np.real(tan_fields["E" + dim2]) * np.real(tan_fields["H" + dim1])
        return e_x_h

    def _compute_flux(self) -> FluxTimeDataArray:
        """Compute instantaneous flux."""
        dim1, dim2 = self._tangential_dims
        tangential_dims = self._tangential_dims

        if self.monitor.use_colocated_integration:
            fields = self._colocated_tangential_fields
            dS = self._diff_area.to_numpy()
            dS_numpy = (dS, dS)
        else:
            fields = self._tangential_fields
            dS_EuHv, dS_EvHu, _, _ = self._diff_area_at_yee_positions(
                truncate_to_monitor_bounds=True
            )
            dS_numpy = (dS_EuHv.to_numpy(), dS_EvHu.to_numpy())

        # Put spatial dims last: (..., u, v)
        Eu = np.real(fields["E" + dim1].transpose(..., *tangential_dims).to_numpy())
        Ev = np.real(fields["E" + dim2].transpose(..., *tangential_dims).to_numpy())
        Hu = np.real(fields["H" + dim1].transpose(..., *tangential_dims).to_numpy())
        Hv = np.real(fields["H" + dim2].transpose(..., *tangential_dims).to_numpy())

        flux_result = _instantaneous_power_flow_numpy((Eu, Ev), (Hu, Hv), dS_numpy)

        return FluxTimeDataArray(
            flux_result, coords={"t": fields["E" + dim1].coords["t"].to_numpy()}
        )

    @cached_property
    def flux(self) -> FluxTimeDataArray:
        """Flux for data corresponding to a 2D monitor."""
        return self._compute_flux()

    def _compute_complex_flux(self) -> Union[FluxDataArray, FreqModeDataArray]:
        """Complex flux is not defined for time-domain data."""
        raise DataError("Complex power flow is not defined for time-domain data.")

    @cached_property
    def complex_flux(self) -> DataArray:
        """Complex flux is not defined for time-domain data."""
        raise DataError("Complex power flow is not defined for time-domain data.")

    def dot(
        self,
        field_data: ElectromagneticFieldData,
        conjugate: bool = True,
        bidirectional: bool = True,
    ) -> xr.DataArray:
        """Inner product is not defined for time-domain data."""
        raise DataError("Inner product is not defined for time-domain data.")

    def outer_dot(
        self,
        field_data: ElectromagneticFieldData,
        conjugate: bool = True,
        bidirectional: bool = True,
    ) -> xr.DataArray:
        """Outer dot product is not defined for time-domain data."""
        raise DataError("Outer dot product is not defined for time-domain data.")

    @property
    def time_reversed_copy(self) -> FieldTimeData:
        """Make a copy of the data with time-reversed fields. The sign of the magnetic fields is
        flipped, and the data is reversed along the ``t`` dimension, such that for a given field,
        ``field[t_beg + t] -> field[t_end - t]``, where ``t_beg`` and ``t_end`` are the first and
        last coordinates along the ``t`` dimension.
        """
        new_data = {}
        for comp, field in self.field_components.items():
            if comp[0] == "H":
                new_data[comp] = -field
            else:
                new_data[comp] = field
            # Reverse time coordinates
            new_data[comp] = new_data[comp].assign_coords({"t": field.t[::-1]}).sortby("t")
        return self.copy(deep=False, update=new_data)


class AuxFieldTimeData(AuxFieldTimeDataset, AbstractFieldData):
    """
    Data associated with a :class:`.AuxFieldTimeMonitor`: scalar components of aux fields.

    Notes
    -----

        The data is stored as a `DataArray <https://docs.xarray.dev/en/stable/generated/xarray.DataArray.html>`_
        object using the `xarray <https://docs.xarray.dev/en/stable/index.html>`_ package.

    Example
    -------
    >>> from tidy3d import ScalarFieldTimeDataArray
    >>> x = [-1,1,3]
    >>> y = [-2,0,2,4]
    >>> z = [-3,-1,1,3,5]
    >>> t = [0, 1e-12, 2e-12]
    >>> coords = dict(x=x[:-1], y=y[:-1], z=z[:-1], t=t)
    >>> grid = Grid(boundaries=Coords(x=x, y=y, z=z))
    >>> scalar_field = ScalarFieldTimeDataArray(np.random.random((2,3,4,3)), coords=coords)
    >>> monitor = AuxFieldTimeMonitor(
    ...     size=(2,4,6), interval=100, name='field', fields=['Nfx'], colocate=True
    ... )
    >>> data = AuxFieldTimeData(monitor=monitor, Nfx=scalar_field, grid_expanded=grid)
    """

    monitor: AuxFieldTimeMonitor = Field(
        title="Monitor",
        description="Time-domain auxiliary field monitor associated with the data.",
    )

    _contains_monitor_fields = enforce_monitor_fields_present()


class ElectromagneticSurfaceFieldData(
    MonitorData, AbstractUnstructuredMonitorData, ElectromagneticSurfaceFieldDataset, ABC
):
    """Collection of vector fields on a surface with some symmetry properties."""

    monitor: Union[SurfaceFieldMonitor, SurfaceFieldTimeMonitor]

    _contains_monitor_fields = enforce_monitor_fields_present()

    @property
    def symmetry_expanded(self) -> ElectromagneticSurfaceFieldData:
        """Return the :class:`.ElectromagneticSurfaceFieldData` with fields expanded based on symmetry. If
        any symmetry is nonzero (i.e. expanded), the interpolation implicitly creates a copy of the
        data array. However, if symmetry is not expanded, the returned array contains a view of
        the data, not a copy.

        Returns
        -------
        :class:`ElectromagneticSurfaceFieldData`
            A data object with the symmetry expanded fields.
        """

        if all(sym == 0 for sym in self.symmetry):
            return self

        return self.updated_copy(**self._symmetry_update_dict, deep=False, validate=False)

    @property
    def symmetry_expanded_copy(self) -> AbstractFieldData:
        """Create a copy of the :class:`.ElectromagneticSurfaceFieldData` with fields expanded based on symmetry.

        Returns
        -------
        :class:`ElectromagneticSurfaceFieldData`
            A data object with the symmetry expanded fields.
        """

        if all(sym == 0 for sym in self.symmetry):
            return self.copy()

        return self.copy(update=self._symmetry_update_dict)

    @property
    def _symmetry_update_dict(self) -> dict:
        """Dictionary of data fields to create data with expanded symmetry."""

        eigvals = self.symmetry_eigenvalues

        h_symmetry = [
            xr.DataArray(
                [eigvals["Hx"](dim), eigvals["Hy"](dim), eigvals["Hz"](dim)],
                coords={"axis": [0, 1, 2]},
            )
            * self.symmetry[dim]
            for dim in range(3)
        ]

        e_symmetry = [
            xr.DataArray(
                [eigvals["Ex"](dim), eigvals["Ey"](dim), eigvals["Ez"](dim)],
                coords={"axis": [0, 1, 2]},
            )
            * self.symmetry[dim]
            for dim in range(3)
        ]

        # normal field is always even under symmetry
        n_symmetry = [
            xr.DataArray(
                [eigvals["Ex"](dim), eigvals["Ey"](dim), eigvals["Ez"](dim)],
                coords={"axis": [0, 1, 2]},
            )
            for dim in range(3)
        ]

        updated_dict: dict[str, Any] = {}
        if self.E is not None:
            updated_dict["E"] = self._symmetry_expanded_copy_base(self.E, e_symmetry)
        if self.H is not None:
            updated_dict["H"] = self._symmetry_expanded_copy_base(self.H, h_symmetry)

        updated_dict["normal"] = self._symmetry_expanded_copy_base(self.normal, n_symmetry)
        updated_dict.update({"symmetry": (0, 0, 0), "symmetry_center": (0, 0, 0)})
        return updated_dict

    def _check_fields_stored(self, components: list[str]) -> None:
        """Check that all requested field components are stored in the data."""
        missing_comps = [comp for comp in components if comp not in self.field_components.keys()]
        if len(missing_comps) > 0:
            raise DataError(
                f"Field components {missing_comps} not included in this data object. Use "
                "the 'fields' argument of a field monitor to select which components are stored."
            )


class SurfaceFieldData(ElectromagneticSurfaceFieldData):
    """
    Data associated with a :class:`.SurfaceFieldMonitor`: E and H fields on a surface.

    Example
    -------
    >>> from tidy3d import PointDataArray, IndexedSurfaceFieldDataArray, TriangularSurfaceDataset, CellDataArray
    >>> import tidy3d as td
    >>> old_logging_level = td.config.logging_level
    >>> td.config.logging_level = "ERROR"
    >>> points = PointDataArray([[0, 0, 0], [0, 1, 0], [1, 1, 1]], dims=["index", "axis"])
    >>> cells = CellDataArray([[0, 1, 2]], dims=["cell_index", "vertex_index"])
    >>> values = PointDataArray([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dims=["index", "axis"])
    >>> field_values = IndexedSurfaceFieldDataArray(np.ones((3, 1, 3, 1)) + 0j, coords={"index": [0, 1, 2], "side": ["outside"],"axis": [0, 1, 2], "f": [1e10]})
    >>> field = TriangularSurfaceDataset(points=points, cells=cells, values=field_values)
    >>> normal = TriangularSurfaceDataset(points=points, cells=cells, values=values)
    >>> monitor = SurfaceFieldMonitor(
    ...     size=(2,4,6), freqs=[1e10], name='field', fields=['E', 'H']
    ... )
    >>> data = SurfaceFieldData(monitor=monitor, E=field, H=field, normal=normal)
    >>> td.config.logging_level = old_logging_level
    """

    monitor: SurfaceFieldMonitor = Field(
        ..., title="Monitor", description="Frequency-domain field monitor associated with the data."
    )

    @property
    def poynting(self) -> TriangularSurfaceDataset:
        """Time-averaged Poynting vector for frequency-domain data."""

        self._check_fields_stored(["E", "H"])
        e_field = self.E
        h_field = self.H

        poynting = e_field.updated_copy(
            values=0.5 * np.real(xr.cross(e_field.values, np.conj(h_field.values), dim="axis"))
        )

        return poynting

    def normalize(self, source_spectrum_fn: Callable[[float], complex]) -> SurfaceFieldData:
        """Return copy of self after normalization is applied using source spectrum function."""
        fields_norm = {}
        src_amps = FreqDataArray(
            source_spectrum_fn(self.monitor.freqs), coords={"f": list(self.monitor.freqs)}
        )
        for field_name, field_data in self.field_components.items():
            fields_norm[field_name] = field_data.updated_copy(
                values=(field_data.values / src_amps).astype(field_data.values.dtype)
            )

        return self.copy(update=fields_norm)


class SurfaceFieldTimeData(ElectromagneticSurfaceFieldData):
    """

    Example
    -------
    >>> from tidy3d import PointDataArray, IndexedSurfaceFieldTimeDataArray, TriangularSurfaceDataset, CellDataArray
    >>> import tidy3d as td
    >>> old_logging_level = td.config.logging_level
    >>> td.config.logging_level = "ERROR"
    >>> points = PointDataArray([[0, 0, 0], [0, 1, 0], [1, 1, 1]], dims=["index", "axis"])
    >>> cells = CellDataArray([[0, 1, 2]], dims=["cell_index", "vertex_index"])
    >>> values = PointDataArray([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dims=["index", "axis"])
    >>> field_values = IndexedSurfaceFieldTimeDataArray(np.ones((3, 1, 3, 1)) + 0j, coords={"index": [0, 1, 2], "side": ["outside"],"axis": [0, 1, 2], "t": [1e-9]})
    >>> field = TriangularSurfaceDataset(points=points, cells=cells, values=field_values)
    >>> normal = TriangularSurfaceDataset(points=points, cells=cells, values=values)
    >>> monitor = SurfaceFieldTimeMonitor(
    ...     size=(2,4,6), interval=100, name='field', fields=['E', 'H']
    ... )
    >>> data = SurfaceFieldTimeData(monitor=monitor, E=field, H=field, normal=normal)
    >>> td.config.logging_level = old_logging_level
    """

    monitor: SurfaceFieldTimeMonitor = Field(
        ..., title="Monitor", description="Time-domain field monitor associated with the data."
    )

    @property
    def poynting(self) -> TriangularSurfaceDataset:
        """Poynting vector for time-domain data."""

        self._check_fields_stored(["E", "H"])
        e_field = self.E
        h_field = self.H

        poynting = e_field.updated_copy(
            values=xr.cross(np.real(e_field.values), np.real(h_field.values), dim="axis")
        )

        return poynting


class PermittivityData(PermittivityDataset, AbstractFieldData):
    """Data for a :class:`.PermittivityMonitor`: diagonal components of the permittivity tensor.

    Notes
    -----

        The data is stored as a `DataArray <https://docs.xarray.dev/en/stable/generated/xarray.DataArray.html>`_
        object using the `xarray <https://docs.xarray.dev/en/stable/index.html>`_ package.

    Example
    -------
    >>> from tidy3d import ScalarFieldDataArray
    >>> x = [-1,1,3]
    >>> y = [-2,0,2,4]
    >>> z = [-3,-1,1,3,5]
    >>> f = [2e14, 3e14]
    >>> coords = dict(x=x[:-1], y=y[:-1], z=z[:-1], f=f)
    >>> grid = Grid(boundaries=Coords(x=x, y=y, z=z))
    >>> sclr_fld = ScalarFieldDataArray((1+1j) * np.random.random((2,3,4,2)), coords=coords)
    >>> monitor = PermittivityMonitor(size=(2,4,6), freqs=[2e14, 3e14], name='eps')
    >>> data = PermittivityData(
    ...     monitor=monitor, eps_xx=sclr_fld, eps_yy=sclr_fld, eps_zz=sclr_fld, grid_expanded=grid
    ... )
    """

    monitor: PermittivityMonitor = Field(
        title="Monitor",
        description="Permittivity monitor associated with the data.",
    )


class MediumData(MediumDataset, AbstractFieldData):
    """Data for a :class:`.MediumMonitor`: diagonal components of the permittivity and permeability tensor.

    Notes
    -----

        The data is stored as a `DataArray <https://docs.xarray.dev/en/stable/generated/xarray.DataArray.html>`_
        object using the `xarray <https://docs.xarray.dev/en/stable/index.html>`_ package.

    Example
    -------
    >>> from tidy3d import ScalarFieldDataArray
    >>> x = [-1,1,3]
    >>> y = [-2,0,2,4]
    >>> z = [-3,-1,1,3,5]
    >>> f = [2e14, 3e14]
    >>> coords = dict(x=x[:-1], y=y[:-1], z=z[:-1], f=f)
    >>> grid = Grid(boundaries=Coords(x=x, y=y, z=z))
    >>> sclr_fld = ScalarFieldDataArray((1+1j) * np.random.random((2,3,4,2)), coords=coords)
    >>> monitor = MediumMonitor(size=(2,4,6), freqs=[2e14, 3e14], name='medium')
    >>> data = MediumData(
    ...     monitor=monitor, eps_xx=sclr_fld, eps_yy=sclr_fld, eps_zz=sclr_fld, mu_xx=sclr_fld, mu_yy=sclr_fld, mu_zz=sclr_fld, grid_expanded=grid
    ... )
    """

    monitor: MediumMonitor = Field(
        title="Monitor", description="Medium property monitor associated with the data."
    )


class AbstractOverlapData(ElectromagneticFieldData):
    amps: ModeAmpsDataArray = Field(
        title="Amplitudes",
        description="Complex-valued amplitudes of the overlap decomposition.",
    )

    def normalize(self, source_spectrum_fn: Callable) -> AbstractOverlapData:
        """Return copy of self after normalization is applied using source spectrum function."""
        if self.amps is None:
            return self.copy()
        source_freq_amps = source_spectrum_fn(self.amps.f)[None, :, None]
        new_amps = (self.amps / source_freq_amps).astype(self.amps.dtype)
        return self.copy(deep=False, update={"amps": new_amps})

    @property
    def time_reversed_copy(self) -> AbstractOverlapData:
        """Make a copy of the data with direction-reversed fields. In lossy or gyrotropic systems,
        the time-reversed fields will not be the same as the backward-propagating modes.
        Note: this only reverses the store fields, any other stored quantities are untouched.
        """
        mnt = self.monitor

        if not mnt.store_fields_direction:
            return self.copy()

        # Time reversal
        new_data = {}
        for comp, field in self.field_components.items():
            if comp[0] == "H":
                new_data[comp] = -np.conj(field)
            else:
                new_data[comp] = np.conj(field)

        # switch direction in the monitor
        new_dir = "+" if mnt.store_fields_direction == "-" else "-"
        update_dict = {"store_fields_direction": new_dir}
        if hasattr(mnt, "direction"):
            update_dict["direction"] = new_dir
        new_data["monitor"] = mnt.updated_copy(**update_dict)
        return self.copy(deep=False, update=new_data)


class FieldOverlapData(AbstractOverlapData):
    monitor: Union[GaussianOverlapMonitor, AstigmaticGaussianOverlapMonitor] = Field(
        title="Monitor", description="Monitor associated with the data."
    )

    def _make_adjoint_sources(self, dataset_names: list[str], fwidth: float) -> list[Source]:
        """Get all adjoint sources for ``FieldOverlapData``."""
        adjoint_sources = []

        for name in dataset_names:
            if name == "amps":
                adjoint_sources += self._make_adjoint_sources_amps(fwidth=fwidth)
            else:
                raise NotImplementedError(
                    f"Unsupported adjoint field '{name}' for 'FieldOverlapData' "
                    f"on monitor '{self.monitor.name}'. Only 'amps' is supported."
                )

        return adjoint_sources

    def _make_adjoint_sources_amps(self, fwidth: float) -> list[Source]:
        """Generate adjoint sources for ``FieldOverlapData.amps``."""
        coords = self.amps.coords
        adjoint_sources = []

        for freq in coords["f"]:
            for direction in coords["direction"]:
                for mode_index in coords["mode_index"]:
                    amp_single = self.amps.sel(f=freq, direction=direction, mode_index=mode_index)

                    amp_complex = self.get_amplitude(amp_single)
                    if (abs(amp_complex) == 0.0) or np.isnan(amp_complex):
                        continue

                    adjoint_sources.append(self._adjoint_source_amp(amp=amp_single, fwidth=fwidth))

        return adjoint_sources

    def _adjoint_source_amp(self, amp: DataArray, fwidth: float) -> Source:
        """Generate an adjoint Gaussian-like source for a single overlap amplitude."""
        coords = amp.coords
        freq0 = coords["f"]
        direction = coords["direction"]

        amp_complex = self.get_amplitude(amp)
        from tidy3d.components.autograd.source_factory import gaussian_source_from_monitor

        return gaussian_source_from_monitor(
            monitor=self.monitor,
            freq=float(freq0),
            direction=direction,
            coefficient=amp_complex,
            fwidth=fwidth,
        )


class ModeData(ModeSolverDataset, AbstractOverlapData):
    """
    Data associated with a :class:`.ModeMonitor`: modal amplitudes, propagation indices and mode profiles.

    Notes
    -----

        The data is stored as a `DataArray <https://docs.xarray.dev/en/stable/generated/xarray.DataArray.html>`_
        object using the `xarray <https://docs.xarray.dev/en/stable/index.html>`_ package.

        The mode monitor data contains the complex effective indices and the complex mode amplitudes at the monitor
        position calculated by mode decomposition. The data structure of the complex effective
        indices :attr`n_complex` contains two coordinates: ``f`` and ``mode_index``, both of which are specified when
        defining the :class:``ModeMonitor`` in the simulation.

        Besides the effective index, :class:``ModeMonitor`` is primarily used to calculate the transmission of
        certain modes in certain directions. We can extract the complex amplitude and square it to compute the mode
        transmission power.

    Example
    -------
    >>> from tidy3d import ModeSpec
    >>> from tidy3d import ModeAmpsDataArray, ModeIndexDataArray
    >>> direction = ["+", "-"]
    >>> f = [1e14, 2e14, 3e14]
    >>> mode_index = np.arange(5)
    >>> index_coords = dict(f=f, mode_index=mode_index)
    >>> index_data = ModeIndexDataArray((1+1j) * np.random.random((3, 5)), coords=index_coords)
    >>> amp_coords = dict(direction=direction, f=f, mode_index=mode_index)
    >>> amp_data = ModeAmpsDataArray((1+1j) * np.random.random((2, 3, 5)), coords=amp_coords)
    >>> monitor = ModeMonitor(
    ...    size=(2,0,6),
    ...    freqs=[2e14, 3e14],
    ...    mode_spec=ModeSpec(num_modes=5),
    ...    name='mode',
    ... )
    >>> data = ModeData(monitor=monitor, amps=amp_data, n_complex=index_data)
    """

    monitor: ModeMonitor = Field(title="Monitor", description="Monitor associated with the data.")

    eps_spec: Optional[list[EpsSpecType]] = Field(
        None,
        title="Permittivity Specification",
        description="Characterization of the permittivity profile on the plane where modes are "
        "computed. Possible values are 'diagonal', 'tensorial_real', 'tensorial_complex'.",
    )

    @model_validator(mode="after")
    def eps_spec_match_mode_spec(self) -> Self:
        """Raise validation error if frequencies in eps_spec does not match frequency list"""
        if self.n_complex is None:
            return self
        val = self.eps_spec
        if val:
            mode_data_freqs = self.n_complex.coords["f"].values
            if len(val) != len(mode_data_freqs):
                raise ValidationError(
                    "eps_spec must be provided at the same frequencies as mode solver data."
                )
        return self

    def overlap_sort(
        self,
        track_freq: TrackFreq,
        overlap_thresh: float = 0.9,
    ) -> ModeData:
        """Starting from the base frequency defined by parameter ``track_freq``, sort modes at each
        frequency according to their overlap values with the modes at the previous frequency.
        That is, it attempts to rearrange modes in such a way that a given ``mode_index``
        corresponds to physically the same mode at all frequencies. Modes with overlap values over
        ``overlap_thresh`` are considered matching and not rearranged.

        Note
        ----
            The monitor associated to this data is updated so that the deprecated
            ``monitor.mode_spec.track_freq`` is set to ``None``, while
            ``monitor.mode_spec.sort_spec.track_freq`` is set to the provided ``track_freq``.

        Parameters
        ----------
        track_freq : Literal["central", "lowest", "highest"]
            Parameter that specifies which frequency will serve as a starting point in
            the reordering process.
        overlap_thresh : float = 0.9
            Modal overlap threshold above which two modes are considered to be the same and are not
            rearranged. If after the sorting procedure the overlap value between two corresponding
            modes is less than this threshold, a warning about a possible discontinuity is
            displayed.
        """
        if len(self.field_components) == 0:
            return self.copy()

        num_freqs = len(self.monitor._stored_freqs)
        num_modes = self.monitor.mode_spec.num_modes

        if track_freq == "lowest":
            f0_ind = 0
        elif track_freq == "highest":
            f0_ind = num_freqs - 1
        elif track_freq == "central":
            f0_ind = num_freqs // 2

        # Normalizing the flux to 1, does not guarantee self terms of overlap integrals
        # are also normalized to 1 when the non-conjugated product is used.
        data_expanded = self.symmetry_expanded
        if data_expanded.monitor.conjugated_dot_product:
            self_overlap = np.ones((num_freqs, num_modes))
        else:
            self_overlap = data_expanded.dot(data_expanded, self.monitor.conjugated_dot_product)
            self_overlap = np.abs(self_overlap.values)
            threshold_array = overlap_thresh * self_overlap

        # Compute sorting order and overlaps with neighboring frequencies
        sorting = -np.ones((num_freqs, num_modes), dtype=int)
        overlap = np.zeros((num_freqs, num_modes))
        phase = np.zeros((num_freqs, num_modes))
        sorting[f0_ind, :] = np.arange(num_modes)  # base frequency won't change
        overlap[f0_ind, :] = self_overlap[f0_ind, :]

        # Sort in two directions from the base frequency
        for step, last_ind in zip([-1, 1], [-1, num_freqs]):
            # Start with the base frequency
            data_template = data_expanded._isel(f=[f0_ind])

            # March to lower/higher frequencies
            for freq_id in range(f0_ind + step, last_ind, step):
                # Calculate threshold array for this frequency
                if not data_expanded.monitor.conjugated_dot_product:
                    overlap_thresh = threshold_array[freq_id, :]
                # Get next frequency to sort
                data_to_sort = data_expanded._isel(f=[freq_id])
                # Assign to the base frequency so that outer_dot will compare them
                data_to_sort = data_to_sort._assign_coords(f=[self.monitor._stored_freqs[f0_ind]])

                # Compute "sorting w.r.t. to neighbor" and overlap values
                sorting_one_mode, amps_one_mode = data_template._find_ordering_one_freq(
                    data_to_sort, overlap_thresh
                )

                # Transform "sorting w.r.t. neighbor" to "sorting w.r.t. to f0_ind"
                sorting[freq_id, :] = sorting_one_mode[sorting[freq_id - step, :]]
                overlap[freq_id, :] = np.abs(amps_one_mode[sorting[freq_id - step, :]])
                phase[freq_id, :] = phase[freq_id - step, :] + np.angle(
                    amps_one_mode[sorting[freq_id - step, :]]
                )

                # Check for discontinuities and show warning if any
                for mode_ind in list(np.nonzero(overlap[freq_id, :] < overlap_thresh)[0]):
                    log.warning(
                        f"Mode '{mode_ind}' appears to undergo a discontinuous change "
                        f"between frequencies '{self.monitor._stored_freqs[freq_id]}' "
                        f"and '{self.monitor._stored_freqs[freq_id - step]}' "
                        f"(overlap: '{overlap[freq_id, mode_ind]:.2f}')."
                    )

                # Reassign for the next iteration
                data_template = data_to_sort

        # Rearrange modes using computed sorting values

        # 1) Reorder using the shared implementation (creates a copy)
        data_reordered = self._apply_mode_reorder(sorting)

        # 2) Apply phase shifts to field components in-place (data_reordered is already a copy)
        for field in data_reordered.field_components.values():
            phase_fact = np.exp(-1j * phase[None, None, None, :, :]).astype(field.data.dtype)
            field.values *= phase_fact

        # 3) Update mode_spec: prefer sort_spec.track_freq; clear deprecated track_freq
        mspec = data_reordered.monitor.mode_spec
        sort_spec = mspec.sort_spec.updated_copy(track_freq=track_freq)
        mspec_updated = mspec.updated_copy(sort_spec=sort_spec, track_freq=None, validate=False)
        monitor_updated = data_reordered.monitor.updated_copy(
            mode_spec=mspec_updated, validate=False
        )

        return data_reordered.updated_copy(monitor=monitor_updated, deep=False, validate=False)

    def _isel(self, **isel_kwargs: Any) -> Self:
        """Wraps ``xarray.DataArray.isel`` for all data fields that are defined over frequency and
        mode index. Used in ``overlap_sort`` but not officially supported since for example
        ``self.monitor.mode_spec`` and ``self.monitor.freqs`` will no longer be matching the
        newly created data."""

        update_dict = dict(self._grid_correction_dict, **self.field_components)
        update_dict = {
            key: field.isel(**isel_kwargs)
            for key, field in update_dict.items()
            if isinstance(field, DataArray)
        }
        return self.updated_copy(**update_dict, deep=False, validate=False)

    def _assign_coords(self, **assign_coords_kwargs: Any) -> Self:
        """Wraps ``xarray.DataArray.assign_coords`` for all data fields that are defined over frequency and
        mode index. Used in ``overlap_sort`` but not officially supported since for example
        ``self.monitor.mode_spec`` and ``self.monitor.freqs`` will no longer be matching the
        newly created data."""
        update_dict = dict(self._grid_correction_dict, **self.field_components)
        update_dict = {
            key: field.assign_coords(**assign_coords_kwargs) for key, field in update_dict.items()
        }
        return self.updated_copy(**update_dict, deep=False, validate=False)

    def _find_ordering_one_freq(
        self,
        data_to_sort: ModeData,
        overlap_thresh: Union[float, np.array],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Find new ordering of modes in data_to_sort based on their similarity to own modes."""
        num_modes = self.n_complex.sizes["mode_index"]

        # Current pairs and their overlaps
        pairs = np.arange(num_modes)
        complex_amps = self.dot(data_to_sort, self.monitor.conjugated_dot_product).data.ravel()
        if self.monitor.store_fields_direction == "-":
            complex_amps *= -1

        # Check whether modes already match
        modes_to_sort = np.where(np.abs(complex_amps) < overlap_thresh)[0]
        num_modes_to_sort = len(modes_to_sort)
        if num_modes_to_sort <= 1:
            return pairs, complex_amps

        # Extract all modes of interest from template data
        data_template_reduced = self._isel(mode_index=modes_to_sort)

        amps_reduced = data_template_reduced.outer_dot(
            data_to_sort._isel(mode_index=modes_to_sort), self.monitor.conjugated_dot_product
        ).to_numpy()[0, :, :]

        if self.monitor.store_fields_direction == "-":
            amps_reduced *= -1

        # Find the most similar modes and corresponding overlap values
        pairs_reduced, amps_reduced = self._find_closest_pairs(amps_reduced)

        # Insert new sorting and overlap values into arrays with all data
        complex_amps[modes_to_sort] = amps_reduced
        pairs[modes_to_sort] = modes_to_sort[pairs_reduced]

        return pairs, complex_amps

    @staticmethod
    def _find_closest_pairs(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Given a complex overlap matrix pair row and column entries."""

        n, k = np.shape(arr)
        if n != k:
            raise DataError("Overlap matrix must be square.")

        arr_abs = np.abs(arr)
        pairs = -np.ones(n, dtype=int)
        values = np.zeros(n, dtype=np.complex128)
        for _ in range(n):
            imax, jmax = np.unravel_index(np.argmax(arr_abs, axis=None), (n, k))
            pairs[imax] = jmax
            values[imax] = arr[imax, jmax]
            arr_abs[imax, :] = -1
            arr_abs[:, jmax] = -1

        return pairs, values

    def _group_index_freq_slices(self) -> tuple[slice, slice, slice]:
        """Get frequency slices for group index numerical differentiation.

        Group index calculation uses three-point finite differences, requiring
        backward, center, and forward frequency points organized as triplets.

        Returns
        -------
        tuple[slice, slice, slice]
            Slices for (backward, center, forward) frequencies from the frequency array.
        """
        freqs = self.n_complex.coords["f"].values
        num_freqs = freqs.size
        back = slice(0, num_freqs, 3)
        center = slice(1, num_freqs, 3)
        fwd = slice(2, num_freqs, 3)
        return back, center, fwd

    def _group_index_post_process(self, frequency_step: float) -> ModeData:
        """Calculate group index and remove added frequencies used only for this calculation.

        Parameters
        ----------
        frequency_step: float
            Fractional frequency step used to calculate the group index.

        Returns
        -------
        :class:`.ModeData`
            Filtered data with calculated group index.
        """

        back, center, fwd = self._group_index_freq_slices()
        freqs = self.n_complex.coords["f"].values[center]

        # calculate group index
        n_center = self.n_eff.isel(f=center).values
        n_backward = self.n_eff.isel(f=back).values
        n_forward = self.n_eff.isel(f=fwd).values

        inv_step = 1 / frequency_step
        # n_g = n + f * df/dn
        # dn/df = (n+ - n-) / (2 f df)
        n_group_data = n_center + (n_forward - n_backward) * inv_step * 0.5
        # D = -2 * pi * c / lda^2 * d(v_g^-1)/dw = -(f / c)^2 * (2 * dn/df + f * d2n/df2)
        # d2n/df2 = (n+ - 2n + n-) / (f df)^2
        # The '1e18' factor converts from s/um^2 to ps/(nm km)
        dispersion_data = (
            (n_forward * (inv_step + 1) + n_backward * (inv_step - 1) - n_center * inv_step * 2)
            * freqs.reshape((-1, 1))
            * (-1e18 * inv_step / C_0**2)
        )

        mode_index = list(self.n_complex.coords["mode_index"].values)
        f = list(freqs)
        n_group = GroupIndexDataArray(
            n_group_data,
            coords={"f": f, "mode_index": mode_index},
        )

        dispersion = ModeDispersionDataArray(
            dispersion_data,
            coords={"f": f, "mode_index": mode_index},
        )

        # remove data corresponding to frequencies used only for group index calculation
        update_dict = {
            "n_complex": self.n_complex.isel(f=center),
            "n_group_raw": n_group,
            "dispersion_raw": dispersion,
        }

        for key, field in self.field_components.items():
            update_dict[key] = field.isel(f=center)

        for key, data in self._grid_correction_dict.items():
            update_dict[key] = data.isel(f=center)

        if self.eps_spec:
            update_dict["eps_spec"] = self.eps_spec[center]

        update_dict["monitor"] = self.monitor.updated_copy(freqs=freqs)

        return self.copy(deep=False, update=update_dict)

    def _colocated_propagation_axes_field(self, field_name: Literal["E", "H"]) -> DataArray:
        """Collect a field DataArray containing all 3 field components and rotate from frame
        with normal axis along z to frame with propagation axis along z.
        """
        tan_dims = self._tangential_dims
        normal_dim = self._normal_dim
        fields = self._colocated_fields
        fields = {key: val.squeeze(dim=normal_dim, drop=True) for key, val in fields.items()}
        mode_spec = self.monitor.mode_spec

        # fields as a (3, ...) numpy array ordered as [tangential1, tagential2, normal]
        field = [fields[field_name + dim].values for dim in tan_dims]
        field = np.array([*field, fields[field_name + normal_dim].values])

        # rotate axes
        if mode_spec.angle_phi != 0:
            field = self.monitor.rotate_points(field, [0, 0, 1], -mode_spec.angle_phi)
        if mode_spec.angle_theta != 0:
            field = self.monitor.rotate_points(field, [0, 1, 0], -mode_spec.angle_theta)

        # new coords for the (3, ...) array
        coords = {"component": [0, 1, 2]}
        # fields are colocated, so all components should have the same coords
        for dim in fields["Ex"].dims:
            coords.update({dim: fields["Ex"].coords[dim]})

        return DataArray(data=field, coords=coords)

    @cached_property
    def pol_fraction(self) -> xr.Dataset:
        r"""Compute the TE and TM polarization fraction defined as the field intensity along the
        first or the second of the two tangential axes. More precisely, if $E_1$ and $E_2$ are
        the electric field components along the two tangential axes, the TE fraction is defined as:

        .. math::

           \frac{\int |E_1|^2 \, {\rm d}S}{\int \left(|E_1|^2 + |E_2|^2\right) \, {\rm d}S}

        and the TM fraction is equal to one minus the TE fraction. The tangential axes are defined
        by popping the normal axis from the list of ``x, y, z``, so e.g. ``x`` and ``z`` for
        propagation in the ``y`` direction.
        """
        self._check_fields_stored(["Ex", "Ey", "Ez"])

        tan_dims = self._tangential_dims
        e_field = self._colocated_propagation_axes_field("E")
        diff_area = self._diff_area
        tm_int = (diff_area * np.abs(e_field.sel(component=1, drop=True)) ** 2).sum(dim=tan_dims)
        te_int = (diff_area * np.abs(e_field.sel(component=0, drop=True)) ** 2).sum(dim=tan_dims)
        te_frac = te_int / (te_int + tm_int)

        return _TracedDataset(data_vars={"te": te_frac, "tm": 1 - te_frac})

    @cached_property
    def pol_fraction_waveguide(self) -> xr.Dataset:
        r"""Compute the TE and TM polarization fraction using the waveguide definition. If $n$ is
        the propagation direction, the TE fraction is defined as:

        .. math::

           1 - \frac{\int |E \cdot n|^2 \, {\rm d}S}{\int |E|^2 \, {\rm d}S}

        and the TM fraction is defined as

        .. math::

           1 - \frac{\int |H \cdot n|^2 \, {\rm d}S}{\int |H|^2 \, {\rm d}S}

        Note
        ----
            The waveguide TE and TM fractions do not sum to one. For example, TEM modes that
            are completely transverse (zero electric and magnetic field in the propagation
            direction) have TE fraction and TM fraction both equal to one.
        """
        self._check_fields_stored(["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"])

        tan_dims = self._tangential_dims
        e_field = self._colocated_propagation_axes_field("E")
        h_field = self._colocated_propagation_axes_field("H")
        diff_area = self._diff_area

        # te fraction
        field_int = [np.abs(e_field.sel(component=ind, drop=True)) ** 2 for ind in range(3)]
        norm_int = (diff_area * field_int[2]).sum(dim=tan_dims)
        tot_int = norm_int + (diff_area * (field_int[0] + field_int[1])).sum(dim=tan_dims)
        te_frac = 1 - norm_int / tot_int

        # tm fraction
        field_int = [np.abs(h_field.sel(component=ind, drop=True)) ** 2 for ind in range(3)]
        norm_int = (diff_area * field_int[2]).sum(dim=tan_dims)
        tot_int = norm_int + (diff_area * (field_int[0] + field_int[1])).sum(dim=tan_dims)
        tm_frac = 1 - norm_int / tot_int

        return _TracedDataset(data_vars={"te": te_frac, "tm": tm_frac})

    @property
    def TE_fraction(self) -> xr.DataArray:
        """Alias for ``pol_fraction.te``."""
        return self.pol_fraction["te"]

    @property
    def TM_fraction(self) -> xr.DataArray:
        """Alias for ``pol_fraction.tm``."""
        return self.pol_fraction["tm"]

    @property
    def wg_TE_fraction(self) -> xr.DataArray:
        """Alias for ``pol_fraction_waveguide.te``."""
        return self.pol_fraction_waveguide["te"]

    @property
    def wg_TM_fraction(self) -> xr.DataArray:
        """Alias for ``pol_fraction_waveguide.tm``."""
        return self.pol_fraction_waveguide["tm"]

    @property
    def modes_info(self) -> xr.Dataset:
        """Dataset collecting various properties of the stored modes."""

        lambda_cm = C_0 / self.k_eff.f / 1e4
        loss_db_cm = 20 * 2 * np.pi * np.log10(np.e) * self.k_eff / lambda_cm

        info = {
            "wavelength": C_0 / self.n_eff.f,
            "n eff": self.n_eff,
            "k eff": self.k_eff,
            "loss (dB/cm)": loss_db_cm,
            f"TE (E{self._tangential_dims[0]}) fraction": None,
            "wg TE fraction": None,
            "wg TM fraction": None,
            "mode area": None,
            "group index": self.n_group_raw,  # Use raw field to avoid issuing a warning
            "dispersion (ps/(nm km))": self.dispersion_raw,  # Use raw field to avoid issuing a warning
        }

        if self.n_group_raw is not None:
            info["group index"] = self.n_group_raw

        stored = self.field_components
        e_fields_stored = all(c in stored for c in ("Ex", "Ey", "Ez"))

        if e_fields_stored:
            info["mode area"] = self.mode_area
            info[f"TE (E{self._tangential_dims[0]}) fraction"] = self.TE_fraction

            if all(c in stored for c in ("Hx", "Hy", "Hz")):
                info["wg TE fraction"] = self.wg_TE_fraction
                info["wg TM fraction"] = self.wg_TM_fraction

        return _TracedDataset(data_vars=info)

    def to_dataframe(self) -> DataFrame:
        """xarray-like method to export the ``modes_info`` into a pandas dataframe which is e.g.
        simple to visualize as a table."""

        dataset = self.modes_info
        drop = []

        if not np.any(dataset["group index"].values):
            drop.append("group index")
        if not np.any(dataset["dispersion (ps/(nm km))"].values):
            drop.append("dispersion (ps/(nm km))")
        if np.all(dataset["loss (dB/cm)"] == 0):
            drop.append("loss (dB/cm)")

        return dataset.drop_vars(drop).to_dataframe()

    def _check_fields_stored(self, components: list[EMField]) -> None:
        """Check that all requested field components are stored in the data."""

        # ModeData can either have all field components or none
        if len(self.field_components) == 0:
            raise DataError(
                "Field data not included in this ModeData object. Set "
                "'ModeMonitor.store_fields_direction' to the desired propagation direction to "
                "include the mode field profiles in the corresponding 'ModeData'."
            )

    def _make_adjoint_sources(self, dataset_names: list[str], fwidth: float) -> list[ModeSource]:
        """Get all adjoint sources for the ``ModeMonitorData``."""

        adjoint_sources = []

        for name in dataset_names:
            if name == "amps":
                adjoint_sources += self._make_adjoint_sources_amps(fwidth=fwidth)
            elif not np.all(self.n_complex.values == 0.0):
                log.warning(
                    f"Can't create adjoint source for 'ModeData.{type(self)}.{name}'. "
                    f"for monitor '{self.monitor.name}'. "
                    "It's likely your objective function depends on sim data that is un-traced. "
                    "Double check your post-processing function to confirm. "
                )

        return adjoint_sources

    def _make_adjoint_sources_amps(self, fwidth: float) -> list[ModeSource]:
        """Generate adjoint sources for ``ModeMonitorData.amps``."""

        coords = self.amps.coords

        adjoint_sources = []

        # TODO: speed up with ufunc?
        for freq in coords["f"]:
            for direction in coords["direction"]:
                for mode_index in coords["mode_index"]:
                    amp_single = self.amps.sel(f=freq, direction=direction, mode_index=mode_index)

                    if abs(self.get_amplitude(amp_single)) == 0.0:
                        continue

                    adjoint_source = self._adjoint_source_amp(amp=amp_single, fwidth=fwidth)
                    adjoint_sources.append(adjoint_source)

        return adjoint_sources

    def _adjoint_source_amp(self, amp: DataArray, fwidth: float) -> ModeSource:
        """Generate an adjoint ``ModeSource`` for a single amplitude."""

        monitor = self.monitor

        # grab coordinates
        coords = amp.coords
        freq0 = coords["f"]
        direction = coords["direction"]
        mode_index = coords["mode_index"]

        amp_complex = self.get_amplitude(amp)

        return mode_source_from_monitor(
            monitor=monitor,
            freq=float(freq0),
            direction=direction,
            mode_index=int(mode_index),
            coefficient=amp_complex,
            fwidth=fwidth,
        )

    def _apply_mode_reorder(self, sort_inds_2d: NDArray) -> Self:
        """Apply a mode reordering along mode_index for all frequency indices.

        Parameters
        ----------
        sort_inds_2d : np.ndarray
            Array of shape (num_freqs, num_modes) where each row is the
            permutation to apply to the mode_index for that frequency.
        """
        sort_inds_2d = np.asarray(sort_inds_2d, dtype=int)
        num_freqs, num_modes = sort_inds_2d.shape

        # Fast no-op
        identity = np.arange(num_modes)
        if np.all(sort_inds_2d == identity[None, :]):
            return self

        modify_data = {}
        new_mode_index_coord = identity

        for key, data in self.data_arrs.items():
            if "mode_index" not in data.dims or "f" not in data.dims:
                continue

            dims_orig = tuple(data.dims)
            # Preserve coords (as numpy)
            coords_out = {
                k: (v.values if hasattr(v, "values") else np.asarray(v))
                for k, v in data.coords.items()
            }
            f_axis = data.get_axis_num("f")
            m_axis = data.get_axis_num("mode_index")

            # Move axes directly to (f, ..., mode)
            src_order = (
                [f_axis] + [ax for ax in range(data.ndim) if ax not in (f_axis, m_axis)] + [m_axis]
            )
            arr = np.moveaxis(data.data, src_order, range(data.ndim))
            nf, nm = arr.shape[0], arr.shape[-1]
            if nf != num_freqs or nm != num_modes:
                raise DataError(
                    "sort_inds_2d shape does not match array shape in _apply_mode_reorder."
                )

            # Apply sorting
            arr2 = arr.reshape(nf, -1, nm)  # (nf, Nlead, nm)
            inds = sort_inds_2d[:, None, :]  # (nf, 1, nm)
            arr2_sorted = np.take_along_axis(arr2, inds, axis=2)
            arr_sorted = arr2_sorted.reshape(arr.shape)

            # Move axes back to original order
            arr_sorted = np.moveaxis(arr_sorted, range(data.ndim), src_order)

            # Update coords: keep f, reset mode_index to 0..num_modes-1
            coords_out["mode_index"] = new_mode_index_coord
            coords_out["f"] = data.coords["f"].values

            modify_data[key] = DataArray(arr_sorted, coords=coords_out, dims=dims_orig)

        return self.updated_copy(**modify_data, deep=False)

    def _apply_mode_subset(self, subset_inds_2d: np.ndarray) -> ModeSolverData:
        """Return copy of self containing only the selected modes.

        Parameters
        ----------
        subset_inds_2d : np.ndarray
            Array of shape ``(num_freqs, num_modes_keep)`` containing the indices of the original
            modes to retain at each frequency.

        Returns
        -------
        :class:`.ModeSolverData`
            Copy of self with only the retained modes.
        """

        subset_inds_2d = np.asarray(subset_inds_2d, dtype=int)
        if subset_inds_2d.ndim != 2:
            raise DataError(
                "subset_inds_2d must be a 2D array of shape (num_freqs, num_modes_keep)."
            )

        num_freqs, num_keep = subset_inds_2d.shape
        if num_keep == 0:
            raise DataError("Cannot create a mode subset with zero modes.")

        num_modes_full = self.n_eff["mode_index"].size

        modify_data = {}
        new_mode_index_coord = np.arange(num_keep)

        for key, data in self.data_arrs.items():
            if "mode_index" not in data.dims or "f" not in data.dims:
                continue

            dims_orig = tuple(data.dims)
            coords_out = {
                k: (v.values if hasattr(v, "values") else np.asarray(v))
                for k, v in data.coords.items()
            }

            f_axis = data.get_axis_num("f")
            m_axis = data.get_axis_num("mode_index")
            src_order = (
                [f_axis] + [ax for ax in range(data.ndim) if ax not in (f_axis, m_axis)] + [m_axis]
            )

            arr = np.moveaxis(data.data, src_order, range(data.ndim))
            nf, nm = arr.shape[0], arr.shape[-1]
            if nf != num_freqs or nm != num_modes_full:
                raise DataError(
                    "subset_inds_2d shape does not match array shape in _apply_mode_subset."
                )

            arr2 = arr.reshape(nf, -1, nm)
            inds = subset_inds_2d[:, None, :]
            arr2_subset = np.take_along_axis(arr2, inds, axis=2)
            arr_subset = arr2_subset.reshape(arr.shape[:-1] + (num_keep,))
            arr_subset = np.moveaxis(arr_subset, range(data.ndim), src_order)

            coords_out["mode_index"] = new_mode_index_coord
            coords_out["f"] = data.coords["f"].values

            modify_data[key] = DataArray(arr_subset, coords=coords_out, dims=dims_orig)

        return self.updated_copy(**modify_data, deep=False)

    def sort_modes(
        self, sort_spec: Optional[ModeSortSpec] = None, track_freq: Optional[TrackFreq] = None
    ) -> ModeSolverData:
        """Sort modes per frequency according to ``sort_spec``.

        The modes are first filtered if ``sort_spec.filter_key`` is provided. They are then sorted
        within each filtered group according to ``sort_spec.sort_key``. if provided. Finally,
        if a tracking frequency is also provided either in ``sort_spec`` or as a separate argument,
        the tracking is applied . The tracking could reshuffle the filter/sort criteria at
        frequencies away from the tracking frequency.

        Parameters
        ----------
        sort_spec : Optional[:class:`.ModeSortSpec`]
            Specification of how to sort the modes.
        track_freq : Optional[Literal["central", "lowest", "highest"]]
            Specifies that modes should be tracked across frequencies. Overrides
            ``sort_spec.track_freq``, but the returned data will have
            ``monitor.mode_spec.sort_spec.track_freq`` set to the provided value, while
            ``self.monitor.mode_spec.track_freq`` will be set to ``None``.

        Returns
        -------
        :class:`.ModeSolverData`
            Copy of self with modes sorted according to ``sort_spec``.
        """

        # Return the original data if no new sorting / tracking required
        if track_freq is None and sort_spec is None:
            return self

        # If filter_pol is set, preserve its ordering and only allow tracking
        if self.monitor.mode_spec.filter_pol is not None:
            # Check if sort_spec has non-default values
            if sort_spec is not None and sort_spec.has_custom_sort_or_filter:
                raise DataError(
                    "Cannot apply custom 'sort_spec' when 'filter_pol' is set. "
                    "The deprecated 'filter_pol' field is mutually exclusive with 'sort_spec'. "
                    "Please use 'sort_spec' with appropriate filtering instead of 'filter_pol'."
                )
            track_freq = track_freq or (sort_spec.track_freq if sort_spec is not None else None)
            if track_freq and self.n_eff["f"].size > 1:
                return self.overlap_sort(track_freq)
            return self

        data = self
        if sort_spec is not None and sort_spec != self.monitor.mode_spec.sort_spec:
            # replace the monitor sort_spec with the provided sort_spec
            data = self.updated_copy(
                path="monitor/mode_spec", sort_spec=sort_spec, deep=False, validate=False
            )

        num_freqs = data.n_eff["f"].size
        num_modes = data.n_eff["mode_index"].size

        all_inds = np.arange(num_modes)

        # Helper to compute ordered indices within a subset
        def _order_indices(
            indices: NDArray, vals_all: DataArray, sort_order: Literal["ascending", "descending"]
        ) -> NDArray:
            if indices.size == 0:
                return indices
            vals = vals_all.isel(mode_index=indices)
            order = np.argsort(vals)
            if sort_order == "descending":
                order = order[::-1]
            return indices[order]

        # Precompute metrics
        filter_metric = None
        if sort_spec is not None and sort_spec.filter_key is not None:
            filter_metric = getattr(data, sort_spec.filter_key)
        # sort_key is always set (defaults to "n_eff")
        sort_metric = getattr(data, sort_spec.sort_key) if sort_spec is not None else None
        identity = np.arange(num_modes)
        sort_inds_2d = np.tile(identity, (num_freqs, 1))

        for ifreq in range(num_freqs):
            # Build groups according to filter if requested
            if filter_metric is not None:
                vals_filt = filter_metric.isel(f=ifreq).values
                if sort_spec.filter_order == "over":
                    mask_first = vals_filt >= sort_spec.filter_reference
                else:
                    mask_first = vals_filt <= sort_spec.filter_reference
                group1 = all_inds[mask_first]
                group2 = all_inds[~mask_first]
            else:
                group1 = all_inds
                group2 = np.array([], dtype=int)

            # Sort within each group
            if sort_metric is not None:
                vals_sort = sort_metric.isel(f=ifreq)
                if sort_spec.sort_reference is not None:
                    vals_sort = np.abs(vals_sort - sort_spec.sort_reference)
                g1 = _order_indices(group1, vals_sort, sort_spec.sort_order)
                g2 = _order_indices(group2, vals_sort, sort_spec.sort_order)
                sort_inds = np.concatenate([g1, g2])
            else:
                # only filtering applied, keep original ordering within groups
                sort_inds = np.concatenate([group1, group2])

            sort_inds_2d[ifreq, : len(sort_inds)] = sort_inds

        if np.all(sort_inds_2d == np.tile(identity, (num_freqs, 1))):
            if sort_spec is not None:
                data_sorted = data.updated_copy(
                    path="monitor/mode_spec", sort_spec=sort_spec, deep=False, validate=False
                )
            else:
                data_sorted = data
        else:
            data_sorted = data._apply_mode_reorder(sort_inds_2d)
            data_sorted = data_sorted.updated_copy(
                path="monitor/mode_spec", sort_spec=sort_spec, deep=False, validate=False
            )

        # Sort modes across frequencies if requested.
        # Note: after sorting, ``track_freq`` is set in ``sort_spec`` regardless of how it was
        # provided. The deprecated ``mode_spec.track_freq`` is cleared.
        sort_spec_track_freq = sort_spec.track_freq if sort_spec is not None else None
        track_freq = track_freq or sort_spec_track_freq
        if track_freq and num_freqs > 1:
            data_sorted = data_sorted.overlap_sort(track_freq)

        keep_inds = None
        keep_modes = sort_spec.keep_modes if sort_spec is not None else "all"
        keep_mask = None
        filter_metric_sorted = None
        if keep_modes == "filtered" or isinstance(keep_modes, int):
            # Re-evaluate the filter after sorting/tracking so modes are dropped consistently.
            # filter key can be None if keep_modes is an int
            if sort_spec.filter_key is not None:
                if sort_spec.filter_key == "fill_fraction_box":
                    filter_metric_sorted = data_sorted.fill_fraction_box
                else:
                    filter_metric_sorted = getattr(data_sorted, sort_spec.filter_key)
                masks_after = []
                for ifreq in range(num_freqs):
                    vals = filter_metric_sorted.isel(f=ifreq).values
                    if sort_spec.filter_order == "over":
                        mask = vals >= sort_spec.filter_reference
                    else:
                        mask = vals <= sort_spec.filter_reference
                    masks_after.append(mask)

                keep_mask = np.all(np.stack(masks_after, axis=0), axis=0)
            if keep_modes == "filtered":
                # keep_mask and filter_metric_sorted will not be None here
                # because we validate that filter_key is not None when
                # keep_modes == "filtered"
                if not np.any(keep_mask):
                    raise ValidationError(
                        "Filtering removes all modes; relax the filter threshold or change 'keep_modes'."
                    )
                num_modes_sorted = filter_metric_sorted.sizes["mode_index"]
                if keep_mask.sum() < num_modes_sorted:
                    keep_inds = np.where(keep_mask)[0]
            elif isinstance(keep_modes, int):
                if keep_mask is not None and keep_mask.sum() < keep_modes:
                    log.warning(
                        f"'keep_modes={keep_modes}' requests {keep_modes} modes, but only "
                        f"{keep_mask.sum()} modes pass the filter. All {keep_modes} modes will be kept. "
                        "Consider relaxing the filter threshold or lowering 'keep_modes'."
                    )
                if keep_modes > num_modes:
                    raise ValidationError(
                        f"'keep_modes={keep_modes}' is greater than the total number of modes."
                    )
                keep_inds = np.arange(keep_modes)

        if keep_inds is not None:
            subset_inds_2d = np.tile(keep_inds, (num_freqs, 1))
            data_subset = data_sorted._apply_mode_subset(subset_inds_2d)
            mspec = data_subset.monitor.mode_spec
            mspec_updated = mspec.updated_copy(num_modes=keep_inds.size, validate=False)
            monitor_updated = data_subset.monitor.updated_copy(
                mode_spec=mspec_updated, validate=False
            )
            data_sorted = data_subset.updated_copy(
                monitor=monitor_updated, deep=False, validate=False
            )

        return data_sorted


class ModeSolverData(ModeData):
    """
    Data associated with a :class:`.ModeSolverMonitor`: scalar components of E and H fields.

    Notes
    -----

        The data is stored as a `DataArray <https://docs.xarray.dev/en/stable/generated/xarray.DataArray.html>`_
        object using the `xarray <https://docs.xarray.dev/en/stable/index.html>`_ package.

    Example
    -------
    >>> from tidy3d import ModeSpec
    >>> from tidy3d import ScalarModeFieldDataArray, ModeIndexDataArray
    >>> x = [-1,1,3]
    >>> y = [-2,0]
    >>> z = [-3,-1,1,3,5]
    >>> f = [2e14, 3e14]
    >>> mode_index = np.arange(5)
    >>> grid = Grid(boundaries=Coords(x=x, y=y, z=z))
    >>> field_coords = dict(x=x[:-1], y=y[:-1], z=z[:-1], f=f, mode_index=mode_index)
    >>> field = ScalarModeFieldDataArray((1+1j)*np.random.random((2,1,4,2,5)), coords=field_coords)
    >>> index_coords = dict(f=f, mode_index=mode_index)
    >>> index_data = ModeIndexDataArray((1+1j) * np.random.random((2,5)), coords=index_coords)
    >>> monitor = ModeSolverMonitor(
    ...    size=(2,0,6),
    ...    freqs=[2e14, 3e14],
    ...    mode_spec=ModeSpec(num_modes=5),
    ...    name='mode_solver',
    ... )
    >>> data = ModeSolverData(
    ...     monitor=monitor,
    ...     Ex=field,
    ...     Ey=field,
    ...     Ez=field,
    ...     Hx=field,
    ...     Hy=field,
    ...     Hz=field,
    ...     n_complex=index_data,
    ...     grid_expanded=grid
    ... )
    """

    monitor: ModeSolverMonitor = Field(
        title="Monitor",
        description="Mode solver monitor associated with the data.",
    )

    amps: Optional[ModeAmpsDataArray] = Field(
        None,
        title="Amplitudes",
        description="Unused for ModeSolverData.",
    )

    grid_distances_primal: Union[tuple[float], tuple[float, float]] = Field(
        (0.0,),
        title="Distances to the Primal Grid",
        description="Relative distances to the primal grid locations along the normal direction in "
        "the original simulation grid. Needed to recalculate grid corrections after "
        "interpolating in frequency.",
    )

    grid_distances_dual: Union[tuple[float], tuple[float, float]] = Field(
        (0.0,),
        title="Distances to the Dual Grid",
        description="Relative distances to the dual grid locations along the normal direction in "
        "the original simulation grid. Needed to recalculate grid corrections after "
        "interpolating in frequency.",
    )

    log: Optional[str] = Field(
        None,
        title="Solver Log",
        description="A string containing the log information from the mode solver run.",
    )

    def _normalize_modes(self) -> None:
        """Normalize modes. Note: this modifies ``self`` in-place."""
        self_dot = self.dot(self, conjugate=self.monitor.conjugated_dot_product)
        real_part = np.real(self_dot)
        imag_part = np.imag(self_dot)
        tolerance = fp_eps * np.abs(self_dot)
        has_meaningful_real_part = np.abs(real_part) > tolerance
        sign = np.where(has_meaningful_real_part, np.sign(real_part), np.sign(imag_part))
        sign = np.where(sign == 0, 1.0, sign)
        scaling = np.sqrt(sign * self_dot)
        near_zero = np.abs(scaling) < fp_eps
        if np.any(near_zero):
            affected = near_zero.any(dim="f") if "f" in near_zero.dims else near_zero
            affected_modes = [int(m) for m in affected.mode_index.values[affected.values]]
            log.warning(
                f"Mode indices {affected_modes} have a self-overlap magnitude smaller than "
                f"'fp_eps' and cannot be normalized. Skipping normalization for these modes."
            )
            scaling = scaling.where(~near_zero, other=1.0)
        for field in self.field_components.values():
            field /= scaling

    @staticmethod
    def _grid_correction_factors(
        primal_distances: tuple[float, ...],
        dual_distances: tuple[float, ...],
        mode_spec: ModeSpec,
        n_complex: ModeIndexDataArray,
        direction: Direction,
        normal_dim: str,
    ) -> tuple[FreqModeDataArray, FreqModeDataArray]:
        """Calculate the grid correction factors for the primal and dual grid.

        Parameters
        ----------
        primal_distances : tuple[float, ...]
            Relative distances to the primal grid locations along the normal direction in the original simulation grid.
        dual_distances : tuple[float, ...]
            Relative distances to the dual grid locations along the normal direction in the original simulation grid.
        mode_spec : ModeSpec
            Mode specification.
        n_complex : ModeIndexDataArray
            Effective indices of the modes.
        direction : Direction
            Direction of the propagation.
        normal_dim : str
            Name of the normal dimension.

        Returns
        -------
        tuple[FreqModeDataArray, FreqModeDataArray]
            Grid correction factors for the primal and dual grid.
        """

        distances_primal = xr.DataArray(primal_distances, coords={normal_dim: primal_distances})
        distances_dual = xr.DataArray(dual_distances, coords={normal_dim: dual_distances})

        # Propagation phase at the primal and dual locations. The k-vector is along the propagation
        # direction, so angle_theta has to be taken into account. The distance along the propagation
        # direction is the distance along the normal direction over cosine(theta).
        cos_theta = np.cos(mode_spec.angle_theta)
        k_vec = cos_theta * 2 * np.pi * n_complex * n_complex.f / C_0
        if direction == "-":
            k_vec *= -1
        phase_primal = np.exp(1j * k_vec * distances_primal)
        phase_dual = np.exp(1j * k_vec * distances_dual)

        # Fields are modified by a linear interpolation to the exact monitor position
        if distances_primal.size > 1:
            phase_primal = phase_primal.interp(**{normal_dim: 0}).drop_vars(normal_dim)
        else:
            phase_primal = phase_primal.squeeze(dim=normal_dim)
        if distances_dual.size > 1:
            phase_dual = phase_dual.interp(**{normal_dim: 0}).drop_vars(normal_dim)
        else:
            phase_dual = phase_dual.squeeze(dim=normal_dim)

        return FreqModeDataArray(phase_primal), FreqModeDataArray(phase_dual)

    def interp_in_freq(
        self,
        freqs: FreqArray,
        method: Literal["linear", "cubic", "poly"] = "linear",
        renormalize: bool = True,
        recalculate_grid_correction: bool = True,
        assume_sorted: bool = False,
    ) -> ModeSolverData:
        """Interpolate mode data to new frequency points.

        Interpolates all stored mode data (effective indices, field components, group indices,
        and dispersion) from the current frequency grid to a new set of frequencies. This is
        useful for obtaining mode data at many frequencies from computations at fewer frequencies,
        when modes vary smoothly with frequency.

        Parameters
        ----------
        freqs : FreqArray
            New frequency points to interpolate to. Should generally span a similar range
            as the original frequencies to avoid extrapolation.
        method : Literal["linear", "cubic", "poly"]
            Interpolation method. ``"linear"`` for linear interpolation (requires 2+ source
            frequencies), ``"cubic"`` for cubic spline interpolation (requires 4+ source
            frequencies), ``"poly"`` for polynomial interpolation using barycentric
            formula (requires 3+ source frequencies).
            For complex-valued data, real and imaginary parts are interpolated independently.
        renormalize : bool = True
            Whether to renormalize the mode profiles to unity power after interpolation.
        recalculate_grid_correction : bool = True
            Whether to recalculate the grid correction factors after interpolation or use interpolated
            grid corrections.
        assume_sorted: bool = False,
            Whether to assume the frequency points are sorted.

        Returns
        -------
        ModeSolverData
            New :class:`ModeSolverData` object with data interpolated to the requested frequencies.

        Note
        ----
            Interpolation assumes modes vary smoothly with frequency. Results may be inaccurate
            near mode crossings or regions of rapid mode variation. Use frequency tracking
            (``mode_spec.sort_spec.track_freq``) to help maintain mode ordering consistency.

        Example
        -------
        >>> # Compute modes at 5 frequencies
        >>> import numpy as np
        >>> freqs_sparse = np.linspace(1e14, 2e14, 5)
        >>> # ... create mode_solver and compute modes ...
        >>> # mode_data = mode_solver.solve()
        >>> # Interpolate to 50 frequencies
        >>> freqs_dense = np.linspace(1e14, 2e14, 50)
        >>> # mode_data_interp = mode_data.interp(freqs=freqs_dense, method='linear')
        """
        # Validate input
        freqs = np.array(freqs)

        source_freqs = self.monitor._stored_freqs

        # Validate method-specific requirements
        if method == "cubic" and len(source_freqs) < 4:
            raise DataError(
                f"Cubic interpolation requires at least 4 source frequency points. "
                f"Got {len(source_freqs)}. Use method='linear' instead."
            )

        if method == "poly":
            if len(source_freqs) < 3:
                raise DataError(
                    f"Polynomial interpolation requires at least 3 source frequency points. "
                    f"Got {len(source_freqs)}. Use method='linear' instead."
                )

        if method not in ["linear", "cubic", "poly"]:
            raise DataError(
                f"Invalid interpolation method '{method}'. Use 'linear', 'cubic', or 'poly'."
            )

        # Check if we're extrapolating significantly and warn
        freq_min, freq_max = np.min(source_freqs), np.max(source_freqs)
        new_freq_min, new_freq_max = np.min(freqs), np.max(freqs)

        if new_freq_min < freq_min * (
            1 - MODE_INTERP_EXTRAPOLATION_TOLERANCE
        ) or new_freq_max > freq_max * (1 + MODE_INTERP_EXTRAPOLATION_TOLERANCE):
            log.warning(
                f"Interpolating to frequencies outside original range "
                f"[{freq_min:.3e}, {freq_max:.3e}] Hz. New range: "
                f"[{new_freq_min:.3e}, {new_freq_max:.3e}] Hz. "
                "Results may be inaccurate due to extrapolation."
            )

        # Build update dictionary
        update_dict = self._interp_in_freq_update_dict(freqs, method, assume_sorted)

        # Handle eps_spec if present - use nearest neighbor interpolation
        if self.eps_spec is not None:
            update_dict["eps_spec"] = list(
                self._interp_dataarray_in_freq(
                    FreqDataArray(self.eps_spec, coords={"f": source_freqs}),
                    freqs,
                    "nearest",
                ).data
            )

        # Update monitor with new frequencies, remove interp_spece
        update_dict["monitor"] = self.monitor.updated_copy(
            freqs=list(freqs),
            mode_spec=self.monitor.mode_spec.updated_copy(interp_spec=None),
        )

        if recalculate_grid_correction:
            update_dict["grid_primal_correction"], update_dict["grid_dual_correction"] = (
                self._grid_correction_factors(
                    list(self.grid_distances_primal),
                    list(self.grid_distances_dual),
                    self.monitor.mode_spec,
                    update_dict["n_complex"],
                    self.monitor.direction,
                    self._normal_dim,
                )
            )

        updated_data = self.updated_copy(**update_dict, deep=False)
        if renormalize:
            # Detach field arrays before in-place normalization. `_interp_dataarray_in_freq`
            # returns the original DataArray objects when frequencies already match.
            updated_data = updated_data.updated_copy(
                **{name: field.copy() for name, field in updated_data.field_components.items()},
                deep=False,
                validate=False,
            )
            updated_data._normalize_modes()

        return updated_data

    @property
    def _reduced_data(self) -> bool:
        """Whether data will be stored at fewer frequencies than the original number of frequencies."""
        return (
            self.monitor.mode_spec._is_interp_spec_applied(self.monitor.freqs)
            and self.monitor.mode_spec.interp_spec.reduce_data
        )

    @property
    def interpolated_copy(self) -> ModeSolverData:
        """Return a copy of the data with interpolated fields."""
        if self.monitor.mode_spec.interp_spec is None:
            return self
        if not self._reduced_data:
            return self
        interpolated_data = self.interp_in_freq(
            freqs=self.monitor.freqs,
            method=self.monitor.mode_spec.interp_spec.method,
            renormalize=True,
            recalculate_grid_correction=True,
            assume_sorted=True,
        )
        return interpolated_data

    def _check_fields_stored(self, components: list[str]) -> None:
        """Check that all requested field components are stored in the data."""
        missing_comps = [comp for comp in components if comp not in self.field_components.keys()]
        if len(missing_comps) > 0:
            raise DataError(
                f"Field components {missing_comps} not included in this ModeSolverData object. Use "
                "the 'fields' argument of a `ModeSolver` or a `ModeSolverMonitor` to select which "
                "components are stored."
            )


class FluxData(MonitorData):
    """
    Data associated with a :class:`.FluxMonitor`: flux data in the frequency-domain.

    Notes
    -----

        The data is stored as a `DataArray <https://docs.xarray.dev/en/stable/generated/xarray.DataArray.html>`_
        object using the `xarray <https://docs.xarray.dev/en/stable/index.html>`_ package.

        We can access the data for each monitor by indexing into the :class:`SimulationData` with the monitor
        ``.name``. For the flux monitor data, we can access the raw flux data as a function of frequency with
        ``.flux``. As most data are multidimensional, it’s often very helpful to print out the data and directly
        inspect its structure.

    Example
    -------
    >>> from tidy3d import FluxDataArray
    >>> f = [2e14, 3e14]
    >>> coords = dict(f=f)
    >>> flux_data = FluxDataArray(np.random.random(2), coords=coords)
    >>> monitor = FluxMonitor(size=(2,0,6), freqs=[2e14, 3e14], name='flux')
    >>> data = FluxData(monitor=monitor, flux=flux_data)

    See Also
    --------

    **Notebooks:**
        * `Advanced monitor data manipulation and visualization <../../notebooks/XarrayTutorial.html>`_
    """

    monitor: FluxMonitor = Field(
        title="Monitor",
        description="Frequency-domain flux monitor associated with the data.",
    )

    flux: FluxDataArray = Field(
        title="Flux",
        description="Flux values in the frequency-domain.",
    )

    def _make_adjoint_sources(
        self, dataset_names: list[str], fwidth: float
    ) -> list[Union[CustomCurrentSource, PointDipole]]:
        """Converts a :class:`.FieldData` to a list of adjoint current or point sources."""

        # avoids error in edge case where there are extraneous flux monitors not used in objective
        if np.all(self.flux.values == 0.0):
            return []

        raise NotImplementedError(
            "Could not formulate adjoint source for 'FluxMonitor' output. To compute derivatives "
            "with respect to flux data, please use a 'FieldMonitor' and call '.flux' on the "
            "resulting 'FieldData' object. Using 'FluxMonitor' directly is not supported as "
            "the full field information is required to construct the adjoint source for this "
            "problem. The 'FluxData' does not contain the information necessary for gradient "
            "computation."
        )

    def normalize(self, source_spectrum_fn: Callable[[DataArray], NDArray]) -> FluxData:
        """Return copy of self after normalization is applied using source spectrum function."""
        source_freq_amps = source_spectrum_fn(self.flux.f)
        source_power = abs(source_freq_amps) ** 2
        new_flux = (self.flux / source_power).astype(self.flux.dtype)
        return self.copy(deep=False, update={"flux": new_flux})


class FluxTimeData(MonitorData):
    """
    Data associated with a :class:`.FluxTimeMonitor`: flux data in the time-domain.

    Notes
    -----

        The data is stored as a `DataArray <https://docs.xarray.dev/en/stable/generated/xarray.DataArray.html>`_
        object using the `xarray <https://docs.xarray.dev/en/stable/index.html>`_ package.

    Example
    -------
    >>> from tidy3d import FluxTimeDataArray
    >>> t = [0, 1e-12, 2e-12]
    >>> coords = dict(t=t)
    >>> flux_data = FluxTimeDataArray(np.random.random(3), coords=coords)
    >>> monitor = FluxTimeMonitor(size=(2,0,6), interval=100, name='flux_time')
    >>> data = FluxTimeData(monitor=monitor, flux=flux_data)
    """

    monitor: FluxTimeMonitor = Field(
        title="Monitor",
        description="Time-domain flux monitor associated with the data.",
    )

    flux: FluxTimeDataArray = Field(
        title="Flux",
        description="Flux values in the time-domain.",
    )


ProjFieldType = Union[
    FieldProjectionAngleDataArray,
    FieldProjectionCartesianDataArray,
    FieldProjectionKSpaceDataArray,
    DiffractionDataArray,
]

ProjMonitorType = Union[
    FieldProjectionAngleMonitor,
    FieldProjectionCartesianMonitor,
    FieldProjectionKSpaceMonitor,
    DiffractionMonitor,
    DirectivityMonitor,
]


class AbstractFieldProjectionData(MonitorData):
    """Collection of projected fields in spherical coordinates in the frequency domain."""

    monitor: ProjMonitorType = Field(
        title="Projection monitor",
        description="Field projection monitor.",
        discriminator=TYPE_TAG_STR,
    )

    Er: ProjFieldType = Field(
        title="Er",
        description="Spatial distribution of r-component of the electric field.",
    )
    Etheta: ProjFieldType = Field(
        title="Etheta",
        description="Spatial distribution of the theta-component of the electric field.",
    )
    Ephi: ProjFieldType = Field(
        title="Ephi",
        description="Spatial distribution of phi-component of the electric field.",
    )
    Hr: ProjFieldType = Field(
        title="Hr",
        description="Spatial distribution of r-component of the magnetic field.",
    )
    Htheta: ProjFieldType = Field(
        title="Htheta",
        description="Spatial distribution of theta-component of the magnetic field.",
    )
    Hphi: ProjFieldType = Field(
        title="Hphi",
        description="Spatial distribution of phi-component of the magnetic field.",
    )

    medium: MediumType = Field(
        default_factory=Medium,
        title="Background Medium",
        description="Background medium through which to project fields.",
        discriminator=TYPE_TAG_STR,
    )

    is_2d_simulation: bool = Field(
        False,
        title="2D Simulation",
        description="Indicates whether the monitor data is for a 2D simulation.",
    )

    @property
    def field_components(self) -> dict[str, DataArray]:
        """Maps the field components to their associated data."""
        return {
            "Er": self.Er,
            "Etheta": self.Etheta,
            "Ephi": self.Ephi,
            "Hr": self.Hr,
            "Htheta": self.Htheta,
            "Hphi": self.Hphi,
        }

    @property
    def f(self) -> np.ndarray:
        """Frequencies."""
        return np.array(self.Etheta.coords["f"])

    @property
    def coords(self) -> dict[str, np.ndarray]:
        """Coordinates of the fields contained."""
        return self.Etheta.coords

    @property
    def coords_spherical(self) -> dict[str, np.ndarray]:
        """Coordinates grid for the fields in the spherical system."""
        if "theta" in self.coords.keys():
            r, theta, phi = np.meshgrid(
                self.coords["r"].values,
                self.coords["theta"].values,
                self.coords["phi"].values,
                indexing="ij",
            )
        elif "z" in self.coords.keys():
            xs, ys, zs = np.meshgrid(
                self.coords["x"].values,
                self.coords["y"].values,
                self.coords["z"].values,
                indexing="ij",
            )
            r, theta, phi = self.monitor.car_2_sph(xs, ys, zs)
        else:
            uxs, uys, r = np.meshgrid(
                self.coords["ux"].values,
                self.coords["uy"].values,
                self.coords["r"].values,
                indexing="ij",
            )
            theta, phi = self.monitor.kspace_2_sph(uxs, uys, self.monitor.proj_axis)
        return {"r": r, "theta": theta, "phi": phi}

    @property
    def dims(self) -> tuple[str, ...]:
        """Dimensions of the radiation vectors contained."""
        return self.Etheta.dims

    def make_data_array(self, data: np.ndarray) -> DataArray:
        """Make an DataArray with data and same coords and dims as fields of self."""
        return DataArray(data=data, coords=self.coords, dims=self.dims)

    def make_dataset(self, keys: tuple[str, ...], vals: tuple[np.ndarray, ...]) -> xr.Dataset:
        """Make a dataset with keys and data with same coords and dims as fields.

        Using _TracedDataset preserves tidy3d's custom DataArray subclass when items are
        accessed later, which is required for autograd-safe .values handling.
        """
        data_arrays = tuple(map(self.make_data_array, vals))
        return _TracedDataset(dict(zip(keys, data_arrays)))

    def make_renormalized_data(
        self, phase: np.ndarray, proj_distance: float
    ) -> AbstractFieldProjectionData:
        """Helper to apply the re-projection phase to a copied dataset."""
        new_data = self.copy()
        for field in new_data.field_components.values():
            field.values *= phase
            if "r" in self.coords.keys():
                field["r"] = np.atleast_1d(proj_distance)
        return new_data

    def normalize(
        self, source_spectrum_fn: Callable[[float], complex]
    ) -> AbstractFieldProjectionData:
        """Return copy of self after normalization is applied using source spectrum function."""
        fields_norm = {}
        for field_name, field_data in self.field_components.items():
            src_amps = source_spectrum_fn(field_data.f)
            fields_norm[field_name] = (field_data / src_amps).astype(field_data.dtype)

        return self.copy(deep=False, update=fields_norm)

    @staticmethod
    def wavenumber(medium: MediumType, frequency: float) -> complex:
        """Complex valued wavenumber associated with a frequency."""
        index_n, index_k = medium.nk_model(frequency=frequency)
        return (2 * np.pi * frequency / C_0) * (index_n + 1j * index_k)

    @property
    def nk(self) -> tuple[float, float]:
        """Returns the real and imaginary parts of the background medium's refractive index."""
        return self.medium.nk_model(frequency=self.f)

    @property
    def k(self) -> complex:
        """Returns the complex wave number associated with the background medium."""
        return self.wavenumber(medium=self.medium, frequency=self.f)

    @property
    def eta(self) -> complex:
        """Returns the complex wave impedance associated with the background medium."""
        eps_complex = self.medium.eps_model(frequency=self.f)
        return ETA_0 / np.sqrt(eps_complex)

    @staticmethod
    def propagation_factor(dist: Union[float, None], k: complex, is_2d_simulation: bool) -> complex:
        """A normalization factor that includes both phase and amplitude decay associated with propagation over a distance with a given wavenumber."""
        if dist is None:
            return 1.0

        if is_2d_simulation:
            return np.exp(1j * k * dist) * np.sqrt(-1j * k / (8 * np.pi * dist))

        return -1j * k * np.exp(1j * k * dist) / (4 * np.pi * dist)

    @property
    def fields_spherical(self) -> xr.Dataset:
        """Get all field components in spherical coordinates relative to the monitor's
        local origin for all projection grid points and frequencies specified in the
        :class:`~tidy3d.components.monitor.AbstractFieldProjectionMonitor`.

        Returns
        -------
        xarray.Dataset
            xarray-backed dataset containing
            (``Er``, ``Etheta``, ``Ephi``, ``Hr``, ``Htheta``, ``Hphi``)
            in spherical coordinates. Accessing items returns tidy3d DataArrays.
        """
        return self.make_dataset(
            keys=self.field_components.keys(), vals=self.field_components.values()
        )

    @property
    def fields_cartesian(self) -> xr.Dataset:
        """Get all field components in Cartesian coordinates relative to the monitor's
        local origin for all projection grid points and frequencies specified in the
        :class:`~tidy3d.components.monitor.AbstractFieldProjectionMonitor`.

        Returns
        -------
        xarray.Dataset
            xarray-backed dataset containing (``Ex``, ``Ey``, ``Ez``, ``Hx``, ``Hy``, ``Hz``)
            in Cartesian coordinates. Accessing items returns tidy3d DataArrays.
        """
        # convert the field components to the Cartesian coordinate system
        coords_sph = self.coords_spherical
        e_data = self.monitor.sph_2_car_field(
            self.Er.values,
            self.Etheta.values,
            self.Ephi.values,
            coords_sph["theta"][..., None],
            coords_sph["phi"][..., None],
        )
        h_data = self.monitor.sph_2_car_field(
            self.Hr.values,
            self.Htheta.values,
            self.Hphi.values,
            coords_sph["theta"][..., None],
            coords_sph["phi"][..., None],
        )

        # package into dataset
        keys = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
        # Stack traced tuples before concatenation to avoid creating object arrays.
        e_data = np.stack(e_data, axis=0)
        h_data = np.stack(h_data, axis=0)
        field_components = np.concatenate((e_data, h_data), axis=0)
        return self.make_dataset(keys=keys, vals=field_components)

    @property
    def power(self) -> DataArray:
        """Get power measured on the projection grid relative to the monitor's local origin.

        Returns
        -------
        ``xarray.DataArray``
            Power at points relative to the local origin.
        """
        power_theta = 0.5 * np.real(self.Etheta * self.Hphi.conj())
        power_phi = 0.5 * np.real(-self.Ephi * self.Htheta.conj())
        power = power_theta + power_phi

        return self.make_data_array(data=power)

    @property
    def radar_cross_section(self) -> DataArray:
        """Radar cross section in units of incident power."""

        _, index_k = self.nk
        if not np.all(index_k == 0):
            raise SetupError("Can't compute RCS for a lossy background medium.")

        n_leading = max(0, len(self.dims) - 1)
        expand_idx = (None,) * n_leading + (Ellipsis,)
        k = self.k[expand_idx]
        eta = self.eta[expand_idx]

        if self.is_2d_simulation:
            constant = k**2 / (16 * np.pi * eta)
        else:
            constant = k**2 / (8 * np.pi * eta)

        # normalize fields by the distance-based phase factor
        coords_sph = self.coords_spherical
        if coords_sph["r"] is None:
            phase = 1.0
        else:
            phase = self.propagation_factor(
                dist=coords_sph["r"][..., None], k=k, is_2d_simulation=self.is_2d_simulation
            )
        Etheta = self.Etheta.values / phase
        Ephi = self.Ephi.values / phase
        rcs_data = constant * (np.abs(Etheta) ** 2 + np.abs(Ephi) ** 2)

        return self.make_data_array(data=rcs_data)

    def _make_adjoint_sources(
        self, dataset_names: list[str], fwidth: float
    ) -> list[Union[CustomCurrentSource, PointDipole]]:
        """Error if server-side field projection is used for autograd"""

        raise NotImplementedError(
            "Adjoint is currently not implemented for server-side field projections. "
            "To compute derivatives with respect to field projection data, please use a 'FieldMonitor' "
            "and use a local projection in your objective function via 'FieldProjector.from_near_field_monitors'. "
            "Using field projection monitors directly is not supported as the full field information is required "
            "to construct the adjoint source for this problem. The field projection data does not contain the "
            "information necessary for gradient computation."
        )


class FieldProjectionAngleData(AbstractFieldProjectionData):
    """Data associated with a :class:`.FieldProjectionAngleMonitor`: components of projected fields.

    Example
    -------
    >>> from tidy3d import FieldProjectionAngleDataArray
    >>> f = np.linspace(1e14, 2e14, 10)
    >>> r = np.atleast_1d(5)
    >>> theta = np.linspace(0, np.pi, 10)
    >>> phi = np.linspace(0, 2*np.pi, 20)
    >>> coords = dict(r=r, theta=theta, phi=phi, f=f)
    >>> values = (1+1j) * np.random.random((len(r), len(theta), len(phi), len(f)))
    >>> scalar_field = FieldProjectionAngleDataArray(values, coords=coords)
    >>> monitor = FieldProjectionAngleMonitor(
    ...     center=(1,2,3), size=(2,2,2), freqs=f, name='n2f_monitor', phi=phi, theta=theta
    ...     )
    >>> data = FieldProjectionAngleData(
    ...     monitor=monitor, Er=scalar_field, Etheta=scalar_field, Ephi=scalar_field,
    ...     Hr=scalar_field, Htheta=scalar_field, Hphi=scalar_field,
    ...     projection_surfaces=monitor.projection_surfaces,
    ...     )
    """

    monitor: FieldProjectionAngleMonitor = Field(
        title="Projection monitor",
        description="Field projection monitor with an angle-based projection grid.",
    )

    projection_surfaces: tuple[FieldProjectionSurface, ...] = Field(
        title="Projection surfaces",
        description="Surfaces of the monitor where near fields were recorded for projection",
    )

    Er: FieldProjectionAngleDataArray = Field(
        title="Er",
        description="Spatial distribution of r-component of the electric field.",
    )
    Etheta: FieldProjectionAngleDataArray = Field(
        title="Etheta",
        description="Spatial distribution of the theta-component of the electric field.",
    )
    Ephi: FieldProjectionAngleDataArray = Field(
        title="Ephi",
        description="Spatial distribution of phi-component of the electric field.",
    )
    Hr: FieldProjectionAngleDataArray = Field(
        title="Hr",
        description="Spatial distribution of r-component of the magnetic field.",
    )
    Htheta: FieldProjectionAngleDataArray = Field(
        title="Htheta",
        description="Spatial distribution of theta-component of the magnetic field.",
    )
    Hphi: FieldProjectionAngleDataArray = Field(
        title="Hphi",
        description="Spatial distribution of phi-component of the magnetic field.",
    )

    @property
    def r(self) -> np.ndarray:
        """Radial distance."""
        return self.Etheta.r.values

    @property
    def theta(self) -> np.ndarray:
        """Polar angles."""
        return self.Etheta.theta.values

    @property
    def phi(self) -> np.ndarray:
        """Azimuthal angles."""
        return self.Etheta.phi.values

    def renormalize_fields(self, proj_distance: float) -> FieldProjectionAngleData:
        """Return a :class:`.FieldProjectionAngleData` with fields re-normalized to a new
        projection distance, by applying a phase factor based on ``proj_distance``.

        Parameters
        ----------
        proj_distance : float = None
            (micron) new radial distance relative to the monitor's local origin.

        Returns
        -------
        :class:`.FieldProjectionAngleData`
            Copy of this :class:`.FieldProjectionAngleData` with fields re-projected
            to ``proj_distance``.
        """
        if self.monitor and not self.monitor.far_field_approx:
            raise DataError(
                "Fields projected without invoking the far field approximation "
                "cannot be re-projected to a new distance."
            )

        # the phase factor associated with the old distance must be removed
        r = self.coords_spherical["r"][..., None]
        old_phase = self.propagation_factor(
            dist=r, k=self.k[None, None, None, :], is_2d_simulation=self.is_2d_simulation
        )

        # the phase factor associated with the new distance must be applied
        new_phase = self.propagation_factor(
            dist=proj_distance, k=self.k, is_2d_simulation=self.is_2d_simulation
        )

        # net phase
        phase = new_phase[None, None, None, :] / old_phase

        # compute updated fields and their coordinates
        return self.make_renormalized_data(phase, proj_distance)

    @property
    def tangential_dims(self) -> list[str]:
        """Tangential dimensions to a spherical surface in the spherical coordinate system."""
        tangential_dims = ["theta", "phi"]
        return tangential_dims

    @staticmethod
    def _check_coords_sorted(coord: np.ndarray, name: str) -> None:
        """Helper for checking whether an array is sorted and raises an exception if it is not."""
        is_sorted = np.all(np.diff(coord) >= 0)
        if not is_sorted:
            raise ValueError(f"{name} was not provided as a sorted array.")

    def _check_integration_suitability(self) -> None:
        """Checks whether the sampling of ``theta`` and ``phi`` is suitable for
        integrating over a spherical surface."""
        if (
            len(self.theta) < MIN_ANGULAR_SAMPLES_SPHERE
            or len(self.phi) < 2 * MIN_ANGULAR_SAMPLES_SPHERE
        ):
            raise ValueError(
                "There are not enough sampling points along `theta` or `phi` for accurate integration. "
                f"Currently, {len(self.theta)} samples for `theta` and {len(self.phi)} samples for `phi`. "
                f"Consider using, at the very least, {MIN_ANGULAR_SAMPLES_SPHERE} samples for `theta` and "
                f"{2 * MIN_ANGULAR_SAMPLES_SPHERE} samples for `phi`."
            )
        self._check_coords_sorted(self.theta, "theta")
        self._check_coords_sorted(self.phi, "phi")
        if not isclose(self.theta[0], 0) or not isclose(self.theta[-1], np.pi):
            raise ValueError(
                "Chosen limits for `theta` are not appropriate for integration. "
                "`theta` must range from 0 to π."
            )
        if not isclose(self.phi[0], 0) or not isclose(self.phi[-1], 2 * np.pi):
            raise ValueError(
                "Chosen limits for `phi` are not appropriate for integration. "
                "`phi` must range from 0 to 2π."
            )

    def flux_from_projected_fields(self) -> FluxDataArray:
        """Flux calculated by integrating the projected fields on a spherical surface.

        Returns
        -------
        :class:`.FluxDataArray`
            Flux in the frequency domain.
        """
        self._check_integration_suitability()
        d_solid_angle = np.sin(self.Etheta.theta)
        integrand = (self.power * d_solid_angle).sel(r=self.monitor.proj_distance)
        flux = self.monitor.proj_distance**2 * integrand.integrate(self.tangential_dims)
        return FluxDataArray(flux)

    @staticmethod
    def get_phi_slice(
        field_array: FieldProjectionAngleDataArray, phi: float, symmetric: bool = False
    ) -> FieldProjectionAngleDataArray:
        """Get a planar slice of the :class:`.FieldProjectionAngleDataArray` along a given phi angle.
        Extends theta range from [0, π] to [0, 2π] to create a full slice.

        Parameters
        ----------
        field_array : :class:`.FieldProjectionAngleDataArray`
            Field array to slice.
        phi : float
            Angle phi in radians to slice at.
        symmetric : bool = False
            If True, uses same data for both halves. If False, takes opposite phi angle
            for back half.

        Returns
        -------
        :class:`.FieldProjectionAngleDataArray`
            2D slice with theta going from 0 to 2π.
        """
        slice_phi = field_array.sel(phi=phi, method="nearest")
        slice_phi = slice_phi.where(slice_phi.theta < np.pi)
        if symmetric:
            slice_opposite_phi = field_array.sel(phi=phi, method="nearest")
        else:
            slice_opposite_phi = field_array.sel(phi=phi + np.pi, method="nearest")
        slice_opposite_phi = slice_opposite_phi.where(slice_opposite_phi.theta > 0)
        slice_opposite_phi = slice_opposite_phi.assign_coords(
            theta=(2 * np.pi - slice_opposite_phi.theta)
        )
        data_array = xr.concat(
            (slice_phi, slice_opposite_phi), dim="theta", coords="minimal", compat="override"
        ).sortby("theta")
        return FieldProjectionAngleDataArray(data_array)


class FieldProjectionCartesianData(AbstractFieldProjectionData):
    """Data associated with a :class:`.FieldProjectionCartesianMonitor`: components of
    projected fields.

    Example
    -------
    >>> from tidy3d import FieldProjectionCartesianDataArray
    >>> f = np.linspace(1e14, 2e14, 10)
    >>> x = np.linspace(0, 5, 10)
    >>> y = np.linspace(0, 10, 20)
    >>> z = np.atleast_1d(5)
    >>> coords = dict(x=x, y=y, z=z, f=f)
    >>> values = (1+1j) * np.random.random((len(x), len(y), len(z), len(f)))
    >>> scalar_field = FieldProjectionCartesianDataArray(values, coords=coords)
    >>> monitor = FieldProjectionCartesianMonitor(
    ...     center=(1,2,3), size=(2,2,2), freqs=f, name='n2f_monitor', x=x, y=y,
    ...     proj_axis=2, proj_distance=50
    ...     )
    >>> data = FieldProjectionCartesianData(
    ...     monitor=monitor, Er=scalar_field, Etheta=scalar_field, Ephi=scalar_field,
    ...     Hr=scalar_field, Htheta=scalar_field, Hphi=scalar_field,
    ...     projection_surfaces=monitor.projection_surfaces,
    ...     )
    """

    monitor: FieldProjectionCartesianMonitor = Field(
        title="Projection monitor",
        description="Field projection monitor with a Cartesian projection grid.",
    )

    projection_surfaces: tuple[FieldProjectionSurface, ...] = Field(
        title="Projection surfaces",
        description="Surfaces of the monitor where near fields were recorded for projection",
    )

    Er: FieldProjectionCartesianDataArray = Field(
        title="Er",
        description="Spatial distribution of r-component of the electric field.",
    )
    Etheta: FieldProjectionCartesianDataArray = Field(
        title="Etheta",
        description="Spatial distribution of the theta-component of the electric field.",
    )
    Ephi: FieldProjectionCartesianDataArray = Field(
        title="Ephi",
        description="Spatial distribution of phi-component of the electric field.",
    )
    Hr: FieldProjectionCartesianDataArray = Field(
        title="Hr",
        description="Spatial distribution of r-component of the magnetic field.",
    )
    Htheta: FieldProjectionCartesianDataArray = Field(
        title="Htheta",
        description="Spatial distribution of theta-component of the magnetic field.",
    )
    Hphi: FieldProjectionCartesianDataArray = Field(
        title="Hphi",
        description="Spatial distribution of phi-component of the magnetic field.",
    )

    @property
    def x(self) -> np.ndarray:
        """X positions."""
        return self.Etheta.x.values

    @property
    def y(self) -> np.ndarray:
        """Y positions."""
        return self.Etheta.y.values

    @property
    def z(self) -> np.ndarray:
        """Z positions."""
        return self.Etheta.z.values

    @property
    def tangential_dims(self) -> list[str]:
        tangential_dims = ["x", "y", "z"]
        tangential_dims.pop(self.monitor.proj_axis)
        return tangential_dims

    @property
    def poynting(self) -> ScalarFieldDataArray:
        """Time-averaged Poynting vector for field data associated to a Cartesian field projection monitor."""
        fc = self.fields_cartesian
        dim1, dim2 = self.tangential_dims

        e1 = fc["E" + dim1]
        e2 = fc["E" + dim2]
        h1 = fc["H" + dim1]
        h2 = fc["H" + dim2]

        e1_h2 = e1 * h2.conj()
        e2_h1 = e2 * h1.conj()

        e_x_h_star = e1_h2 - e2_h1
        return 0.5 * np.real(e_x_h_star)

    @cached_property
    def flux(self) -> FluxDataArray:
        """Flux for projected field data corresponding to a Cartesian field projection monitor."""
        flux = self.poynting.integrate(self.tangential_dims)
        return FluxDataArray(flux)

    def renormalize_fields(self, proj_distance: float) -> FieldProjectionCartesianData:
        """Return a :class:`.FieldProjectionCartesianData` with fields re-normalized to a new
        projection distance, by applying a phase factor based on ``proj_distance``.

        Parameters
        ----------
        proj_distance : float = None
            (micron) new plane distance relative to the monitor's local origin.

        Returns
        -------
        :class:`.FieldProjectionCartesianData`
            Copy of this :class:`.FieldProjectionCartesianData` with fields re-projected
            to ``proj_distance``.
        """
        if not self.monitor.far_field_approx:
            raise DataError(
                "Fields projected without invoking the far field approximation "
                "cannot be re-projected to a new distance."
            )

        # the phase factor associated with the old distance must be removed
        k = self.k[None, None, None, :]
        r = self.coords_spherical["r"][..., None]
        old_phase = self.propagation_factor(dist=r, k=k, is_2d_simulation=self.is_2d_simulation)

        # update the field components' projection distance
        norm_dir, _ = self.monitor.pop_axis(["x", "y", "z"], axis=self.monitor.proj_axis)
        for field in self.field_components.values():
            field[norm_dir] = np.atleast_1d(proj_distance)

        # the phase factor associated with the new distance must be applied
        r = self.coords_spherical["r"][..., None]
        new_phase = self.propagation_factor(dist=r, k=k, is_2d_simulation=self.is_2d_simulation)

        # net phase
        phase = new_phase / old_phase

        # compute updated fields and their coordinates
        return self.make_renormalized_data(phase, proj_distance)


class FieldProjectionKSpaceData(AbstractFieldProjectionData):
    """Data associated with a :class:`.FieldProjectionKSpaceMonitor`: components of
    projected fields.

    Example
    -------
    >>> from tidy3d import FieldProjectionKSpaceDataArray
    >>> f = np.linspace(1e14, 2e14, 10)
    >>> ux = np.linspace(0, 0.4, 10)
    >>> uy = np.linspace(0, 0.6, 20)
    >>> r = np.atleast_1d(5)
    >>> coords = dict(ux=ux, uy=uy, r=r, f=f)
    >>> values = (1+1j) * np.random.random((len(ux), len(uy), len(r), len(f)))
    >>> scalar_field = FieldProjectionKSpaceDataArray(values, coords=coords)
    >>> monitor = FieldProjectionKSpaceMonitor(
    ...     center=(1,2,3), size=(2,2,2), freqs=f, name='n2f_monitor', ux=ux, uy=uy, proj_axis=2
    ...     )
    >>> data = FieldProjectionKSpaceData(
    ...     monitor=monitor, Er=scalar_field, Etheta=scalar_field, Ephi=scalar_field,
    ...     Hr=scalar_field, Htheta=scalar_field, Hphi=scalar_field,
    ...     projection_surfaces=monitor.projection_surfaces,
    ...     )
    """

    monitor: FieldProjectionKSpaceMonitor = Field(
        title="Projection monitor",
        description="Field projection monitor with a projection grid defined in k-space.",
    )

    projection_surfaces: tuple[FieldProjectionSurface, ...] = Field(
        title="Projection surfaces",
        description="Surfaces of the monitor where near fields were recorded for projection",
    )

    Er: FieldProjectionKSpaceDataArray = Field(
        title="Er",
        description="Spatial distribution of r-component of the electric field.",
    )
    Etheta: FieldProjectionKSpaceDataArray = Field(
        title="Etheta",
        description="Spatial distribution of the theta-component of the electric field.",
    )
    Ephi: FieldProjectionKSpaceDataArray = Field(
        title="Ephi",
        description="Spatial distribution of phi-component of the electric field.",
    )
    Hr: FieldProjectionKSpaceDataArray = Field(
        title="Hr",
        description="Spatial distribution of r-component of the magnetic field.",
    )
    Htheta: FieldProjectionKSpaceDataArray = Field(
        title="Htheta",
        description="Spatial distribution of theta-component of the magnetic field.",
    )
    Hphi: FieldProjectionKSpaceDataArray = Field(
        title="Hphi",
        description="Spatial distribution of phi-component of the magnetic field.",
    )

    @property
    def ux(self) -> np.ndarray:
        """Reciprocal X positions."""
        return self.Etheta.ux.values

    @property
    def uy(self) -> np.ndarray:
        """Reciprocal Y positions."""
        return self.Etheta.uy.values

    @property
    def r(self) -> np.ndarray:
        """Radial distance."""
        return self.Etheta.r.values

    def renormalize_fields(self, proj_distance: float) -> FieldProjectionKSpaceData:
        """Return a :class:`.FieldProjectionKSpaceData` with fields re-normalized to a new
        projection distance, by applying a phase factor based on ``proj_distance``.

        Parameters
        ----------
        proj_distance : float = None
            (micron) new radial distance relative to the monitor's local origin.

        Returns
        -------
        :class:`.FieldProjectionKSpaceData`
            Copy of this :class:`.FieldProjectionKSpaceData` with fields re-projected
            to ``proj_distance``.
        """
        if self.monitor and not self.monitor.far_field_approx:
            raise DataError(
                "Fields projected without invoking the far field approximation "
                "cannot be re-projected to a new distance."
            )

        # the phase factor associated with the old distance must be removed
        r = self.coords_spherical["r"][..., None]
        old_phase = self.propagation_factor(
            dist=r, k=self.k[None, None, None, :], is_2d_simulation=self.is_2d_simulation
        )

        # the phase factor associated with the new distance must be applied
        new_phase = self.propagation_factor(
            dist=proj_distance, k=self.k, is_2d_simulation=self.is_2d_simulation
        )

        # net phase
        phase = new_phase[None, None, None, :] / old_phase

        # compute updated fields and their coordinates
        return self.make_renormalized_data(phase, proj_distance)


class DiffractionData(AbstractFieldProjectionData):
    """Data for a :class:`.DiffractionMonitor`: complex components of diffracted far fields.

    Note
    ----

        The diffraction data are separated into S and P polarizations. At normal incidence when
        S and P are undefined, P(S) corresponds to ``Ey``(``Ez``) polarization for monitor normal
        to x, P(S) corresponds to ``Ex``(``Ez``) polarization for monitor normal to y, and P(S)
        corresponds to ``Ex``(``Ey``) polarization for monitor normal to z.

    Note
    ----

        The power amplitudes per polarization and diffraction order, and correspondingly the power
        per diffraction order, correspond to the power carried by each diffraction order in the
        monitor normal direction. They are not to be confused with power carried by plane waves
        in the propagation direction of each diffraction order, which can be obtained from the
        spherical-coordinate fields which are also stored. The power definition is such that the
        grating efficiency is the recorded power over the input source power, and the direct sum
        over the power in all orders should equal the total power flowing through the monitor.


    Example
    -------
    >>> from tidy3d import DiffractionDataArray
    >>> f = np.linspace(1e14, 2e14, 10)
    >>> orders_x = list(range(-4, 5))
    >>> orders_y = list(range(-6, 7))
    >>> pol = ["s", "p"]
    >>> coords = dict(orders_x=orders_x, orders_y=orders_y, f=f)
    >>> values = (1+1j) * np.random.random((len(orders_x), len(orders_y), len(f)))
    >>> field = DiffractionDataArray(values, coords=coords)
    >>> monitor = DiffractionMonitor(
    ...     center=(1,2,3), size=(np.inf,np.inf,0), freqs=f, name='diffraction'
    ... )
    >>> data = DiffractionData(
    ...     monitor=monitor, sim_size=[1,1], bloch_vecs=[1,2],
    ...     Etheta=field, Ephi=field, Er=field,
    ...     Htheta=field, Hphi=field, Hr=field,
    ... )
    """

    monitor: DiffractionMonitor = Field(
        title="Monitor",
        description="Diffraction monitor associated with the data.",
    )

    Er: DiffractionDataArray = Field(
        title="Er",
        description="Spatial distribution of r-component of the electric field.",
    )
    Etheta: DiffractionDataArray = Field(
        title="Etheta",
        description="Spatial distribution of the theta-component of the electric field.",
    )
    Ephi: DiffractionDataArray = Field(
        title="Ephi",
        description="Spatial distribution of phi-component of the electric field.",
    )
    Hr: DiffractionDataArray = Field(
        title="Hr",
        description="Spatial distribution of r-component of the magnetic field.",
    )
    Htheta: DiffractionDataArray = Field(
        title="Htheta",
        description="Spatial distribution of theta-component of the magnetic field.",
    )
    Hphi: DiffractionDataArray = Field(
        title="Hphi",
        description="Spatial distribution of phi-component of the magnetic field.",
    )

    sim_size: tuple[float, float] = Field(
        title="Domain size",
        description="Size of the near field in the local x and y directions.",
        json_schema_extra={"units": MICROMETER},
    )

    bloch_vecs: Union[tuple[float, float], tuple[ArrayFloat1D, ArrayFloat1D]] = Field(
        title="Bloch vectors",
        description="Bloch vectors along the local x and y directions in units of "
        "``2 * pi / (simulation size along the respective dimension)``.",
    )

    @staticmethod
    def shifted_orders(orders: tuple[int, ...], bloch_vec: Union[float, np.ndarray]) -> np.ndarray:
        """Diffraction orders shifted by the Bloch vector."""
        return bloch_vec + np.atleast_2d(orders).T

    @staticmethod
    def reciprocal_coords(
        orders: np.ndarray,
        size: float,
        bloch_vec: Union[float, np.ndarray],
        f: float,
        medium: MediumType,
    ) -> np.ndarray:
        """Get the normalized "u" reciprocal coords for a vector of orders, size, and bloch vec."""
        if size == 0:
            return np.atleast_2d(0)
        epsilon = medium.eps_model(f)
        bloch_array = DiffractionData.shifted_orders(orders, bloch_vec)
        return bloch_array / size * C_0 / f / np.real(np.sqrt(epsilon))

    @staticmethod
    def compute_angles(
        reciprocal_vectors: tuple[np.ndarray, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute the polar and azimuth angles associated with the given reciprocal vectors."""
        # some wave number pairs are outside the light cone, leading to warnings from numpy.arcsin
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="invalid value encountered in arcsin", category=RuntimeWarning
            )
            ux, uy = reciprocal_vectors
            thetas, phis = DiffractionMonitor.kspace_2_sph(ux[:, None, :], uy[None, :, :], axis=2)
        return (thetas, phis)

    @property
    def coords_spherical(self) -> dict[str, np.ndarray]:
        """Coordinates grid for the fields in the spherical system."""
        theta, phi = self.angles
        return {"r": None, "theta": theta, "phi": phi}

    @property
    def orders_x(self) -> np.ndarray:
        """Allowed orders along x."""
        return np.atleast_1d(np.array(self.Etheta.coords["orders_x"]))

    @property
    def orders_y(self) -> np.ndarray:
        """Allowed orders along y."""
        return np.atleast_1d(np.array(self.Etheta.coords["orders_y"]))

    @property
    def reciprocal_vectors(self) -> tuple[np.ndarray, np.ndarray]:
        """Get the normalized "ux" and "uy" reciprocal vectors."""
        return (self.ux, self.uy)

    @property
    def ux(self) -> np.ndarray:
        """Normalized wave vector along x relative to ``local_origin`` and oriented
        with respect to ``monitor.normal_dir``, normalized by the wave number in the
        projection medium."""
        return self.reciprocal_coords(
            orders=self.orders_x,
            size=self.sim_size[0],
            bloch_vec=self.bloch_vecs[0],
            f=self.f,
            medium=self.medium,
        )

    @property
    def uy(self) -> np.ndarray:
        """Normalized wave vector along y relative to ``local_origin`` and oriented
        with respect to ``monitor.normal_dir``, normalized by the wave number in the
        projection medium."""
        return self.reciprocal_coords(
            orders=self.orders_y,
            size=self.sim_size[1],
            bloch_vec=self.bloch_vecs[1],
            f=self.f,
            medium=self.medium,
        )

    @property
    def angles(self) -> tuple[DataArray]:
        """The (theta, phi) angles corresponding to each allowed pair of diffraction
        orders storeds as data arrays. Disallowed angles are set to ``np.nan``.
        """
        thetas, phis = self.compute_angles(self.reciprocal_vectors)
        theta_data = DataArray(thetas, coords=self.coords)
        phi_data = DataArray(phis, coords=self.coords)
        return theta_data, phi_data

    @property
    def amps(self) -> DataArray:
        """Complex power amplitude in each order for 's' and 'p' polarizations, normalized so that
        the power carried by the wave of that order and polarization equals ``abs(amps)^2``.
        """
        norm = diffraction_amplitude_norm(self.angles[0].values, self.eta)
        amp_theta = self.Etheta.values * norm
        amp_phi = self.Ephi.values * norm

        # stack the amplitudes in s- and p-components along a new polarization axis
        coords = {}
        coords["orders_x"] = np.atleast_1d(self.orders_x)
        coords["orders_y"] = np.atleast_1d(self.orders_y)
        coords["f"] = np.atleast_1d(self.f)
        coords["polarization"] = ["s", "p"]
        return DataArray(np.stack([amp_phi, amp_theta], axis=3), coords=coords)

    @property
    def power(self) -> DataArray:
        """Total power in each order, summed over both polarizations."""
        return (np.abs(self.amps) ** 2).sum(dim="polarization")

    @property
    def radar_cross_section(self) -> DataArray:
        """Radar cross section in units of incident power."""
        raise ValueError("RCS is not a well-defined quantity for diffraction data.")

    @property
    def fields_spherical(self) -> xr.Dataset:
        """Get all field components in spherical coordinates relative to the monitor's
        local origin for all allowed diffraction orders and frequencies specified in the
        :class:`DiffractionMonitor`.

        Returns
        -------
        xarray.Dataset
            xarray-backed dataset containing
            (``Er``, ``Etheta``, ``Ephi``, ``Hr``, ``Htheta``, ``Hphi``)
            in spherical coordinates. Accessing items returns tidy3d DataArrays.
        """
        fields = [field.values for field in self.field_components.values()]
        keys = ["Er", "Etheta", "Ephi", "Hr", "Htheta", "Hphi"]
        return self._make_dataset(fields, keys)

    @property
    def fields_cartesian(self) -> xr.Dataset:
        """Get all field components in Cartesian coordinates relative to the monitor's
        local origin for all allowed diffraction orders and frequencies specified in the
        :class:`DiffractionMonitor`.

        Returns
        -------
        xarray.Dataset
            xarray-backed dataset containing (``Ex``, ``Ey``, ``Ez``, ``Hx``, ``Hy``, ``Hz``)
            in Cartesian coordinates. Accessing items returns tidy3d DataArrays.
        """
        theta, phi = self.angles
        theta = theta.values
        phi = phi.values

        e_x, e_y, e_z = self.monitor.sph_2_car_field(
            0, self.Etheta.values, self.Ephi.values, theta, phi
        )
        h_x, h_y, h_z = self.monitor.sph_2_car_field(
            0, self.Htheta.values, self.Hphi.values, theta, phi
        )
        e_x, e_y, e_z, h_x, h_y, h_z = (
            np.nan_to_num(fld) for fld in [e_x, e_y, e_z, h_x, h_y, h_z]
        )

        fields = [e_x, e_y, e_z, h_x, h_y, h_z]
        keys = ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]
        return self._make_dataset(fields, keys)

    def _make_dataset(self, fields: tuple[np.ndarray, ...], keys: tuple[str, ...]) -> xr.Dataset:
        """Make a dataset for fields with given field names.

        Using _TracedDataset preserves tidy3d's custom DataArray subclass when items are
        accessed later, which is required for autograd-safe .values handling.
        """
        data_arrays = []
        for field in fields:
            data_arrays.append(DataArray(data=field, coords=self.coords, dims=self.dims))
        return _TracedDataset(dict(zip(keys, data_arrays)))

    """ Autograd code """

    def _make_adjoint_sources(self, dataset_names: list[str], fwidth: float) -> list[PlaneWave]:
        """Get all adjoint sources for the ``DiffractionMonitor.amps``."""

        # NOTE: everything just goes through `.amps`, any post-processing is encoded in E-fields
        return self._make_adjoint_sources_amps(fwidth=fwidth)

    def _make_adjoint_sources_amps(self, fwidth: float) -> list[PlaneWave]:
        """Make adjoint sources for outputs that depend on DiffractionData.`amps`."""

        amps = self.amps
        coords = amps.coords

        adjoint_sources = []

        # TODO: speed up with ufunc?
        # loop over all coordinates in the diffraction amplitudes
        for freq in coords["f"]:
            for pol in coords["polarization"]:
                for order_x in coords["orders_x"]:
                    for order_y in coords["orders_y"]:
                        amp_single = amps.sel(
                            f=freq,
                            polarization=pol,
                            orders_x=order_x,
                            orders_y=order_y,
                        )

                        # ignore any amplitudes of 0.0 or nan
                        amp_complex = self.get_amplitude(amp_single)
                        if (abs(amp_complex) == 0.0) or np.isnan(amp_complex):
                            continue

                        # compute a plane wave for this amplitude (if propagating / not None)
                        adjoint_source = self.adjoint_source_amp(amp=amp_single, fwidth=fwidth)
                        if adjoint_source is not None:
                            adjoint_sources.append(adjoint_source)

        return adjoint_sources

    def adjoint_source_amp(self, amp: DataArray, fwidth: float) -> PlaneWave:
        """Generate an adjoint ``PlaneWave`` for a single amplitude."""

        coords = amp.coords
        freq0 = coords["f"]
        pol = coords["polarization"]
        order_x = coords["orders_x"]
        order_y = coords["orders_y"]

        amp_complex = self.get_amplitude(amp)

        return diffraction_source_from_data(
            diff_data=self,
            freq=float(freq0),
            order_x=int(order_x),
            order_y=int(order_y),
            polarization=str(pol.values),
            coefficient=amp_complex,
            fwidth=fwidth,
        )


class DirectivityData(FieldProjectionAngleData):
    """
    Data associated with a :class:`.DirectivityMonitor`.

    Example
    -------
    >>> from tidy3d import FluxDataArray, FieldProjectionAngleDataArray
    >>> f = np.linspace(1e14, 2e14, 10)
    >>> r = np.atleast_1d(1e6)
    >>> theta = np.linspace(0, np.pi, 10)
    >>> phi = np.linspace(0, 2*np.pi, 20)
    >>> coords = dict(r=r, theta=theta, phi=phi, f=f)
    >>> coords_flux = dict(f=f)
    >>> values = (1+1j) * np.random.random((len(r), len(theta), len(phi), len(f)))
    >>> flux_data = FluxDataArray(np.random.random(len(f)), coords=coords_flux)
    >>> scalar_field = FieldProjectionAngleDataArray(values, coords=coords)
    >>> monitor = DirectivityMonitor(center=(1,2,3), size=(2,2,2), freqs=f, name='n2f_monitor', phi=phi, theta=theta)
    >>> data = DirectivityData(monitor=monitor, flux=flux_data, Er=scalar_field, Etheta=scalar_field, Ephi=scalar_field,
    ...     Hr=scalar_field, Htheta=scalar_field, Hphi=scalar_field, projection_surfaces=monitor.projection_surfaces)
    """

    monitor: DirectivityMonitor = Field(
        title="Monitor",
        description="Monitor describing the angle-based projection grid on which to measure directivity data.",
    )

    flux: FluxDataArray = Field(
        title="Flux",
        description="Flux values that are either computed from fields recorded on the "
        "projection surfaces or by integrating the projected fields over a spherical surface.",
    )

    @staticmethod
    def from_spherical_field_dataset(
        monitor: DirectivityMonitor,
        field_dataset: xr.Dataset,
    ) -> DirectivityData:
        """Creates a :class:`.DirectivityData` instance from a spherical field dataset.

        Parameters
        ----------
        monitor : :class:`.DirectivityMonitor`
            Monitor defining measurement parameters.
        field_dataset : ``xr.Dataset``
            Dataset containing spherical field components (Er, Etheta, etc.).
            Must sample the entire spherical surface to compute flux correctly.

        Returns
        -------
        :class:`.DirectivityData`
            New :class:`.DirectivityData` instance with computed flux from spherical field integration.
        """
        field_dataset = _TracedDataset(field_dataset)
        f = list(monitor.freqs)
        flux = FluxDataArray(np.zeros(len(f)), coords={"f": f})
        dir_data = DirectivityData(
            monitor=monitor,
            flux=flux,
            Er=field_dataset.Er,
            Etheta=field_dataset.Etheta,
            Ephi=field_dataset.Ephi,
            Hr=field_dataset.Hr,
            Htheta=field_dataset.Htheta,
            Hphi=field_dataset.Hphi,
            projection_surfaces=monitor.projection_surfaces,
        )
        flux = dir_data.flux_from_projected_fields()
        return dir_data.updated_copy(flux=flux, deep=False, validate=False)

    def __add__(self, other: DirectivityData) -> DirectivityData:
        """Form the superposition of two :class:`.DirectivityData`. Flux is recomputed by
        integrating the projected fields over a sphere.

        Note
        ----
        Intended use is for combining fields from different simulations that were recorded
        using the same ``monitor``. The returned :class:`.DirectivityData` takes the ``monitor``
        from ``self``.
        """
        fields_dataset = self.fields_spherical + other.fields_spherical
        combined_data = DirectivityData.from_spherical_field_dataset(self.monitor, fields_dataset)
        return combined_data

    def normalize(self, source_spectrum_fn: Callable[[float], complex]) -> DirectivityData:
        """
        Return a copy of self after normalization is applied using the source
        spectrum function, for both field components and flux data.
        """

        fields_norm = {}
        for field_name, field_data in self.field_components.items():
            src_amps = source_spectrum_fn(field_data.f)
            fields_norm[field_name] = (field_data / src_amps).astype(field_data.dtype)

        # Normalize flux
        source_freq_amps = source_spectrum_fn(self.flux.f)
        source_power = abs(source_freq_amps) ** 2
        new_flux = (self.flux / source_power).astype(self.flux.dtype)

        return self.copy(deep=False, update=dict(fields_norm, flux=new_flux))

    @staticmethod
    def _check_valid_pol_basis(pol_basis: PolarizationBasis, tilt_angle: float) -> None:
        if pol_basis != "linear" and pol_basis != "circular":
            raise ValueError("'pol_basis' must be either 'linear' or 'circular'")
        if tilt_angle is not None and pol_basis == "circular":
            raise ValueError("'tilt_angle' is only defined for linear polarization.")

    def partial_radiation_intensity(
        self, pol_basis: PolarizationBasis = "linear", tilt_angle: Optional[float] = None
    ) -> xr.Dataset:
        """Partial radiation intensity in the frequency domain as a function of angles theta and phi.
        The partial radiation intensities are computed in the ``linear`` or ``circular`` polarization
        bases. If ``tilt_angle`` is not ``None``, the radiation intensity is computed in the linear
        polarization basis rotated by ``tilt_angle`` from the theta-axis. Radiation intensity is
        measured in units of Watts per unit solid angle.

        Parameters
        ----------
        pol_basis : PolarizationBasis
            The desired polarization basis used to express partial radiation intensity, either
            ``linear`` or ``circular``.
        tilt_angle : float
            The angle by which the co-polar vector is rotated from the theta-axis.
            At ``tilt_angle`` = 0, the co-polar vector coincides with the theta-axis and the cross-polar
            vector coincides with the phi-axis; while at ``tilt_angle = pi/2``, the co-polar vector
            coincides with the phi-axis.

        Returns
        -------
        xarray.Dataset
            Dataset containing the partial radiation intensities split into the two polarization states.
        """
        self._check_valid_pol_basis(pol_basis, tilt_angle)
        if pol_basis == "linear":
            if tilt_angle is not None:
                tilt_fields = self.fields_linear_polarization_tilted(tilt_angle)
                E1 = tilt_fields.Eco
                E2 = tilt_fields.Ecross
                H1 = tilt_fields.Hco
                H2 = tilt_fields.Hcross
                keys = ("Uco", "Ucross")
            else:
                E1 = self.Etheta
                E2 = self.Ephi
                H1 = self.Htheta
                H2 = self.Hphi
                keys = ("Utheta", "Uphi")
        else:
            E1 = self.fields_circular_polarization.Eright
            E2 = self.fields_circular_polarization.Eleft
            # needs extra -1 to counteract -1 in cross product below
            H1 = -1.0 * self.fields_circular_polarization.Hleft
            H2 = self.fields_circular_polarization.Hright
            keys = ("Uright", "Uleft")

        U_1 = (self.monitor.proj_distance**2) * 0.5 * np.real(E1 * np.conj(H2))
        U_2 = (self.monitor.proj_distance**2) * 0.5 * np.real(-E2 * np.conj(H1))

        data_arrays = (U_1, U_2)
        return _TracedDataset(dict(zip(keys, data_arrays)))

    @property
    def radiation_intensity(self) -> FieldProjectionAngleDataArray:
        """Radiation intensity in the frequency domain as a function of angles theta and phi.
        Radiation intensity is measured in units of Watts per unit solid angle.
        """
        # Calls partial radiation intensity using default linear polarization basis
        partial_U = self.partial_radiation_intensity()
        return partial_U.Utheta + partial_U.Uphi

    @property
    def radiated_power(self) -> FreqDataArray:
        """Total radiated power in the frequency domain with units of Watts."""
        # If this data was created using FieldProjectionAngleData, the sign
        # will already be correct. Also will be correct if monitor size is all nonzero.
        # TODO fix this sign issue in the backend if possible
        if (
            isinstance(self.monitor, FieldProjectionAngleMonitor)
            or self.monitor.size.count(0.0) == 0
        ):
            return FreqDataArray(self.flux.values, {"f": self.f})
        # The monitor could be planar and directed downward
        sign = 1.0 if self.monitor.normal_dir == "+" else -1.0
        return FreqDataArray(sign * self.flux.values, {"f": self.f})

    def partial_directivity(
        self, pol_basis: PolarizationBasis = "linear", tilt_angle: Optional[float] = None
    ) -> xr.Dataset:
        """Directivity in the frequency domain as a function of angles theta and phi.
        The partial directivities are computed in the ``linear`` or ``circular`` polarization
        bases. If ``tilt_angle`` is not ``None``, the radiation intensity is computed in the linear
        polarization basis rotated by ``tilt_angle`` from the theta-axis. Directivity is a dimensionless
        quantity defined as the ratio of the radiation intensity in a given direction to the average
        radiation intensity over all directions.

        Parameters
        ----------
        pol_basis : PolarizationBasis
            The desired polarization basis used to express partial directivity, either
            ``linear`` or ``circular``.
        tilt_angle : float
            The angle by which the co-polar vector is rotated from the theta-axis.
            At ``tilt_angle`` = 0, the co-polar vector coincides with the theta-axis and the cross-polar
            vector coincides with the phi-axis; while at ``tilt_angle = pi/2``, the co-polar vector
            coincides with the phi-axis.

        Returns
        -------
        ``xarray.Dataset``
            Dataset containing the partial directivities split into the two polarization states.
        """
        self._check_valid_pol_basis(pol_basis, tilt_angle)
        if pol_basis == "linear":
            if tilt_angle is None:
                rename_mapping = {"Utheta": "Dtheta", "Uphi": "Dphi"}
            else:
                rename_mapping = {"Uco": "Dco", "Ucross": "Dcross"}
        else:
            rename_mapping = {"Uright": "Dright", "Uleft": "Dleft"}
        # Average radiation intensity is total radiated power divided by 4 pi
        avg_radiation_intensity = self.radiated_power / (4 * np.pi)
        partial_U = self.partial_radiation_intensity(pol_basis=pol_basis, tilt_angle=tilt_angle)
        partial_D = partial_U / avg_radiation_intensity
        return _TracedDataset(partial_D.rename(rename_mapping))

    @property
    def directivity(self) -> FieldProjectionAngleDataArray:
        """Directivity in the frequency domain as a function of angles theta and phi.
        Directivity is a dimensionless quantity defined as the ratio of the radiation
        intensity in a given direction to the average radiation intensity over all directions.
        """
        # Calls partial directivity using default linear polarization basis
        partial_D = self.partial_directivity()
        return FieldProjectionAngleDataArray(partial_D.Dtheta + partial_D.Dphi)

    def calc_radiation_efficiency(self, power_in: FreqDataArray) -> FreqDataArray:
        """Calculate radiation efficiency as the ratio of radiated power to input power.

        Parameters
        ----------
        power_in : FreqDataArray
            Power supplied to the radiating element in the frequency domain, in units of Watts.

        Returns
        -------
        FreqDataArray
            Radiation efficiency (dimensionless) in the frequency domain, computed as
            radiated_power / power_in.
        """
        return FreqDataArray((self.radiated_power / power_in).values, {"f": self.f})

    def calc_partial_gain(
        self,
        power_in: FreqDataArray,
        pol_basis: PolarizationBasis = "linear",
        tilt_angle: Optional[float] = None,
    ) -> xr.Dataset:
        """The partial gain figures of merit for antennas. The partial gains are computed
        in the ``linear`` or ``circular`` polarization bases. If ``tilt_angle`` is not ``None``,
        the partial directivity is computed in the linear polarization basis rotated by ``tilt_angle``
        from the theta-axis. Gain is dimensionless.

        Parameters
        ----------
        power_in : FreqDataArray
            Power, in units of Watts, supplied to the radiating element in the frequency domain.

        pol_basis : PolarizationBasis
            The desired polarization basis used to express partial gain, either
            ``linear`` or ``circular``.

        tilt_angle : float
            The angle by which the co-polar vector is rotated from the theta-axis.
            At ``tilt_angle`` = 0, the co-polar vector coincides with the theta-axis and the cross-polar
            vector coincides with the phi-axis; while at ``tilt_angle = pi/2``, the co-polar vector
            coincides with the phi-axis.

        Returns
        -------
        ``xarray.Dataset``
            Dataset containing the partial gains split into the two polarization states.
        """
        self._check_valid_pol_basis(pol_basis, tilt_angle)
        radiation_efficiency = self.calc_radiation_efficiency(power_in)
        partial_D = self.partial_directivity(pol_basis=pol_basis, tilt_angle=tilt_angle)
        partial_G = radiation_efficiency * partial_D
        if pol_basis == "linear":
            if tilt_angle is None:
                rename_mapping = {"Dtheta": "Gtheta", "Dphi": "Gphi"}
            else:
                rename_mapping = {"Dco": "Gco", "Dcross": "Gcross"}
        else:
            rename_mapping = {"Dright": "Gright", "Dleft": "Gleft"}
        return partial_G.rename(rename_mapping)

    def calc_gain(self, power_in: FreqDataArray) -> FieldProjectionAngleDataArray:
        """The gain figure of merit for antennas. Gain is dimensionless.

        Parameters
        ----------
        power_in : FreqDataArray
            Power, in units of Watts, supplied to the radiating element in the frequency domain.
        """
        partial_G = self.calc_partial_gain(power_in)
        return FieldProjectionAngleDataArray(partial_G.Gtheta + partial_G.Gphi)

    @property
    def axial_ratio(self) -> FieldProjectionAngleDataArray:
        """Axial Ratio (AR) in the frequency domain as a function of angles theta and phi.
        AR is a dimensionless quantity defined as the ratio of the major axis to the minor
        axis of the polarization ellipse.

        Note
        ----
        The axial ratio computation is based on:

        Balanis, Constantine A., "Antenna Theory: Analysis and Design,"
        John Wiley & Sons, Chapter 2.12 (2016).
        """

        # Axial ratio calculations based on equations (2-65) to (2-67)
        # from Balanis, Constantine A., "Antenna Theory: Analysis and Design,"
        # John Wiley & Sons, 2016.
        #
        # The standard formula computes AR_denominator = (A + B) - |C| where
        # A = |Etheta|², B = |Ephi|², C = Etheta² + Ephi². For near-linear
        # polarization |C| ≈ A + B, causing catastrophic cancellation.
        #
        # Instead, we use the identity (A+B)² - |C|² = 4*(ad - bc)² where
        # a,b = Re,Im(Etheta) and c,d = Re,Im(Ephi). This "cross" term
        # avoids the subtraction entirely.
        cross = self.Etheta.real * self.Ephi.imag - self.Etheta.imag * self.Ephi.real

        AR_numerator = (
            np.abs(self.Etheta) ** 2
            + np.abs(self.Ephi) ** 2
            + np.abs(self.Etheta**2 + self.Ephi**2)
        )

        inds_zero = AR_numerator == 0
        axial_ratio_inverse = xr.zeros_like(AR_numerator)
        axial_ratio_inverse = axial_ratio_inverse.where(inds_zero, 2 * np.abs(cross) / AR_numerator)

        # Safety-net cap for the degenerate case of zero total field
        axial_ratio_inverse = axial_ratio_inverse.where(
            axial_ratio_inverse >= 1 / AXIAL_RATIO_CAP, 1 / AXIAL_RATIO_CAP
        )

        return 1 / axial_ratio_inverse

    @property
    def left_polarization(self) -> FieldProjectionAngleDataArray:
        """Electric far field for left-hand circular polarization
        (counterclockwise component) with an angle-based projection grid.
        """
        return self.fields_circular_polarization.Eleft

    @property
    def right_polarization(self) -> FieldProjectionAngleDataArray:
        """Electric far field for right-hand circular polarization
        (clockwise component) with an angle-based projection grid.
        """
        return self.fields_circular_polarization.Eright

    def fields_linear_polarization_tilted(self, tilt_angle: float) -> xr.Dataset:
        """Electric and magnetic fields in the linear polarization basis that is rotated
        at the pole of the radiation sphere by `tilt_angle`.

        Parameters
        ----------
        tilt_angle : float
            The angle by which the co-polar vector is rotated from the theta-axis.
            At ``tilt_angle`` = 0, the co-polar vector coincides with the theta-axis and the cross-polar
            vector coincides with the phi-axis; while at ``tilt_angle = pi/2``, the co-polar vector
            coincides with the phi-axis.

        Returns
        -------
        ``xarray.Dataset``
            Dataset containing (``Eco``, ``Ecross``, ``Hco``, ``Hcross``)
        """
        Eco = np.cos(tilt_angle) * self.Etheta + np.sin(tilt_angle) * self.Ephi
        Ecross = -np.sin(tilt_angle) * self.Etheta + np.cos(tilt_angle) * self.Ephi
        Hco = np.cos(tilt_angle) * self.Htheta + np.sin(tilt_angle) * self.Hphi
        Hcross = -np.sin(tilt_angle) * self.Htheta + np.cos(tilt_angle) * self.Hphi

        keys = ("Eco", "Ecross", "Hco", "Hcross")
        data_arrays = (Eco, Ecross, Hco, Hcross)
        return _TracedDataset(dict(zip(keys, data_arrays)))

    @property
    def fields_circular_polarization(self) -> xr.Dataset:
        """Electric and magnetic fields in the circular polarization basis.

        Note
        ----
        Uses IEEE handedness convention for polarization state, which means right-handed circularly
        polarization is associated with a clockwise rotation of the electric field vector from the
        point of the view of the source. However, we use the physics convention for time evolution
        of time-harmonic fields, which modifies the computation when compared to engineering references.

        Returns
        -------
        ``xarray.Dataset``
            xarray dataset containing (``Eleft``, ``Eright``, ``Hleft``, ``Hright``)
            in Spherical coordinates.
        """
        Eleft = (self.Etheta + 1j * self.Ephi) / np.sqrt(2.0)
        Eright = (self.Etheta - 1j * self.Ephi) / np.sqrt(2.0)
        Hleft = (self.Hphi - 1j * self.Htheta) / np.sqrt(2.0)
        Hright = (self.Hphi + 1j * self.Htheta) / np.sqrt(2.0)

        keys = ("Eleft", "Eright", "Hleft", "Hright")
        data_arrays = (Eleft, Eright, Hleft, Hright)
        return _TracedDataset(dict(zip(keys, data_arrays)))
