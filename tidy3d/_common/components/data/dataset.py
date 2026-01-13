"""Collections of DataArrays."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np
import xarray as xr
from pydantic import Field

from tidy3d._common.components.base import Tidy3dBaseModel
from tidy3d._common.components.data.data_array import (
    DataArray,
    TriangleMeshDataArray,
)
from tidy3d._common.exceptions import DataError
from tidy3d._common.log import log

if TYPE_CHECKING:
    from typing import Callable

    from tidy3d._common.components.data.data_array import ScalarFieldDataArray
    from tidy3d._common.components.types.base import ArrayLike, Axis

DEFAULT_MAX_SAMPLES_PER_STEP = 10_000
DEFAULT_MAX_CELLS_PER_STEP = 10_000
DEFAULT_TOLERANCE_CELL_FINDING = 1e-6


class Dataset(Tidy3dBaseModel, ABC):
    """Abstract base class for objects that store collections of `:class:`.DataArray`s."""

    @property
    def data_arrs(self) -> dict:
        """Returns a dictionary of all `:class:`.DataArray`s in the dataset."""
        data_arrs = {}
        for key in self.__class__.model_fields.keys():
            data = getattr(self, key)
            if isinstance(data, DataArray):
                data_arrs[key] = data
        return data_arrs


class TriangleMeshDataset(Dataset):
    """Dataset for storing triangular surface data."""

    surface_mesh: TriangleMeshDataArray = Field(
        title="Surface mesh data",
        description="Dataset containing the surface triangles and corresponding face indices "
        "for a surface mesh.",
    )


class AbstractFieldDataset(Dataset, ABC):
    """Collection of scalar fields with some symmetry properties."""

    @property
    @abstractmethod
    def field_components(self) -> dict[str, DataArray]:
        """Maps the field components to their associated data."""

    def apply_phase(self, phase: float) -> AbstractFieldDataset:
        """Create a copy where all elements are phase-shifted by a value (in radians)."""
        if phase == 0.0:
            return self
        phasor = np.exp(1j * phase)
        field_components_shifted = {}
        for fld_name, fld_cmp in self.field_components.items():
            fld_cmp_shifted = phasor * fld_cmp
            field_components_shifted[fld_name] = fld_cmp_shifted
        return self.updated_copy(**field_components_shifted)

    @property
    @abstractmethod
    def grid_locations(self) -> dict[str, str]:
        """Maps field components to the string key of their grid locations on the yee lattice."""

    @property
    @abstractmethod
    def symmetry_eigenvalues(self) -> dict[str, Callable[[Axis], float]]:
        """Maps field components to their (positive) symmetry eigenvalues."""

    def package_colocate_results(self, centered_fields: dict[str, ScalarFieldDataArray]) -> Any:
        """How to package the dictionary of fields computed via self.colocate()."""
        return xr.Dataset(centered_fields)

    def colocate(self, x: ArrayLike = None, y: ArrayLike = None, z: ArrayLike = None) -> xr.Dataset:
        """Colocate all of the data at a set of x, y, z coordinates.

        Parameters
        ----------
        x : Optional[array-like] = None
            x coordinates of locations.
            If not supplied, does not try to colocate on this dimension.
        y : Optional[array-like] = None
            y coordinates of locations.
            If not supplied, does not try to colocate on this dimension.
        z : Optional[array-like] = None
            z coordinates of locations.
            If not supplied, does not try to colocate on this dimension.

        Returns
        -------
        xr.Dataset
            Dataset containing all fields at the same spatial locations.
            For more details refer to `xarray's Documentation <https://tinyurl.com/cyca3krz>`_.

        Note
        ----
        For many operations (such as flux calculations and plotting),
        it is important that the fields are colocated at the same spatial locations.
        Be sure to apply this method to your field data in those cases.
        """

        if hasattr(self, "monitor") and self.monitor.colocate:
            with log as consolidated_logger:
                consolidated_logger.warning(
                    "Colocating data that has already been colocated during the solver "
                    "run. For most accurate results when colocating to custom coordinates set "
                    "'Monitor.colocate' to 'False' to use the raw data on the Yee grid "
                    "and avoid double interpolation. Note: the default value was changed to 'True' "
                    "in Tidy3D version 2.4.0."
                )

        # convert supplied coordinates to array and assign string mapping to them
        supplied_coord_map = {k: np.array(v) for k, v in zip("xyz", (x, y, z)) if v is not None}

        # dict of data arrays to combine in dataset and return
        centered_fields = {}

        # loop through field components
        for field_name, field_data in self.field_components.items():
            # loop through x, y, z dimensions and raise an error if only one element along dim
            for coord_name, coords_supplied in supplied_coord_map.items():
                coord_data = np.array(field_data.coords[coord_name])
                if coord_data.size == 1:
                    raise DataError(
                        f"colocate given {coord_name}={coords_supplied}, but "
                        f"data only has one coordinate at {coord_name}={coord_data[0]}. "
                        "Therefore, can't colocate along this dimension. "
                        f"supply {coord_name}=None to skip it."
                    )

            centered_fields[field_name] = field_data.interp(
                **supplied_coord_map, kwargs={"bounds_error": True}
            )

        # combine all centered fields in a dataset
        return self.package_colocate_results(centered_fields)
