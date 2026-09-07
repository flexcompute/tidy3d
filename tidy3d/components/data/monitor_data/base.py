from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING, Any

import autograd.numpy as np
import xarray as xr
from pydantic import (
    Field,
    model_validator,
)

from tidy3d.components.autograd.source_factory import flip_direction as _flip_direction
from tidy3d.components.base import TYPE_TAG_STR
from tidy3d.components.base_sim.data.monitor_data import AbstractMonitorData
from tidy3d.components.data.data_array import DataArray
from tidy3d.components.data.dataset import AbstractFieldDataset
from tidy3d.components.grid.grid import (
    Coords,
    Grid,
)
from tidy3d.components.monitor import (
    AuxFieldTimeMonitor,
    FieldMonitor,
    FieldTimeMonitor,
    MediumMonitor,
    ModeMonitor,
    PermittivityMonitor,
)
from tidy3d.components.types import (
    Coordinate,
    Symmetry,
)
from tidy3d.components.types.monitor import MonitorType
from tidy3d.components.validators import required_if_symmetry_present
from tidy3d.exceptions import DataError
from tidy3d.log import log

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import SupportsComplex

    from tidy3d.compat import Self
    from tidy3d.components.data.data_array import FreqDataArray
    from tidy3d.components.data.dataset import Dataset
    from tidy3d.components.source.base import Source


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
        self, freq_array: FreqDataArray, method: str | None = None
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
    def flip_direction(direction: str | DataArray) -> str:
        """Flip the direction of a string ``('+', '-') -> ('-', '+')``."""
        return _flip_direction(direction)

    @staticmethod
    def get_amplitude(x: DataArray | SupportsComplex) -> complex:
        """Get the complex amplitude out of some data."""

        if isinstance(x, DataArray):
            x = x.values

        return complex(x)


class AbstractFieldData(MonitorData, AbstractFieldDataset, ABC):
    """Collection of scalar fields with some symmetry properties."""

    monitor: (
        FieldMonitor
        | FieldTimeMonitor
        | AuxFieldTimeMonitor
        | PermittivityMonitor
        | ModeMonitor
        | MediumMonitor
    ) = Field(discriminator=TYPE_TAG_STR)

    symmetry: tuple[Symmetry, Symmetry, Symmetry] = Field(
        default=(0, 0, 0),
        title="Symmetry",
        description="Symmetry eigenvalues of the original simulation in x, y, and z.",
    )

    symmetry_center: Coordinate | None = Field(
        default=None,
        title="Symmetry Center",
        description="Center of the symmetry planes of the original simulation in x, y, and z. "
        "Required only if any of the ``symmetry`` field are non-zero.",
    )
    grid_expanded: Grid | None = Field(
        default=None,
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

        update_dict: dict[str, DataArray | tuple[float, float, float] | None] = {}
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
                # then we need to interpolate, which is slower. Categorical components
                # (e.g. structure-ownership indices) must always snap to the nearest value,
                # never interpolate, since blending discrete indices is meaningless.
                use_sel = (
                    field_name in self.nearest_neighbor_components
                    or len(scalar_data.coords[dim_name]) == 1
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
