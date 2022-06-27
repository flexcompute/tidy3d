""" Monitor Level Data, store the DataArrays associated with a single monitor."""
from abc import ABC, abstractmethod
from typing import Union, Dict, Tuple, Callable
from typing_extensions import Annotated

import pydantic as pd

from ..base import TYPE_TAG_STR
from ..types import Axis, Coordinate
from ..boundary import Symmetry
from ..grid import Grid
from ..validators import enforce_monitor_fields_present
from ..monitor import Monitor, MonitorType, FieldMonitor, FieldTimeMonitor, ModeFieldMonitor
from ..monitor import ModeMonitor, FluxMonitor, FluxTimeMonitor, PermittivityMonitor

from .base import Tidy3dData
from .data_array import ScalarFieldDataArray, ScalarFieldTimeDataArray, ScalarModeFieldDataArray
from .data_array import FluxTimeDataArray, FluxDataArray, ModeIndexDataArray, ModeAmpsDataArray
from .data_array import DataArray

# TODO: base class for field objects?
# TODO: saving and loading from hdf5 group or json file
# TODO: field data colocate
# TODO: mode data neff, keff properties
# TODO: docstring examples?
# TODO: ModeFieldData select by index -> FieldData
# TODO: equality checking two MonitorData
# TODO: normalization


class MonitorData(Tidy3dData, ABC):
    """Abstract base class of objects that store data pertaining to a single :class:`.monitor`."""

    monitor: MonitorType = pd.Field(
        ...,
        title="Monitor",
        description="Monitor associated with the data.",
        descriminator=TYPE_TAG_STR,
    )

    def apply_symmetry(
        self, symmetry: Symmetry, symmetry_center: Coordinate, grid_expanded: Grid
    ) -> "Self":
        """Return copy of self with symmetry applied."""
        return self.copy()

    def normalize(self, source_spectrum_fn) -> "Self":
        """Return copy of self after normalization is applied using source spectrum function."""
        return self.copy()


class AbstractFieldData(MonitorData, ABC):
    """Collection of scalar fields with some symmetry properties."""

    monitor: Union[FieldMonitor, FieldTimeMonitor, PermittivityMonitor, ModeFieldMonitor]

    @property
    @abstractmethod
    def field_components(self) -> Dict[str, Tuple[DataArray, str, Callable[[Axis], float]]]:
        """The components of the field in the :class:`.AbstractField`."""

    def apply_symmetry(
        self, symmetry: Symmetry, symmetry_center: Coordinate, grid_expanded: Grid
    ) -> "Self":
        """Create a copy of the :class:`.AbstractField` with symmetry applied

        Returns
        -------
        :class:`AbstractField`
            A new data object with the symmetry expanded fields.
        """

        new_fields = {}

        for field_name, (scalar_data, grid_key, eigenval_fn) in self.field_components.items():

            # get grid locations for this field component on the expanded grid
            grid_locations = grid_expanded[grid_key]

            for sym_dim, (sym_val, sym_center) in enumerate(zip(symmetry, symmetry_center)):

                # Continue if no symmetry along this dimension
                if sym_val == 0:
                    continue

                # Get coordinates for this field component on the expanded grid
                coords = grid_locations.to_list[sym_dim]

                # Get indexes of coords that lie on the left of the symmetry center
                flip_inds = np.where(coords < center)[0]

                # Get the symmetric coordinates on the right
                coords_interp = np.copy(coords)
                coords_interp[flip_inds] = 2 * center - coords[flip_inds]

                # Interpolate. There generally shouldn't be values out of bounds except potentially
                # when handling modes, in which case they should be at the boundary and close to 0.
                scalar_data = scalar_data.sel({dim_name: coords_interp}, method="nearest")
                scalar_data = scalar_data.assign_coords({dim_name: coords})

                # apply the symmetry eigenvalue to the flipped values
                sym_eig = eigenval_fn(sym_dim)
                scalar_data[{"xyz"[sym_dim]: flip_inds}] *= sym_val * sym_eig

            # assign the final scalar data to the new_fields
            new_fields[field_name] = scalar_data

        return self.copy(update=new_fields)


class ElectromagneticFieldData(AbstractFieldData, ABC):
    """Stores a collection of E and H fields with x, y, z components."""

    @property
    def field_components(self) -> Dict[str, Tuple[DataArray, str, Callable[[Axis], float]]]:
        """Maps field_name to (scalar data, grid key, function of dim giving symmetry eigenvalue."""
        return {
            "Ex": (self.Ex, "Ex", lambda dim: -1 if (dim == 0) else +1),
            "Ey": (self.Ey, "Ey", lambda dim: -1 if (dim == 1) else +1),
            "Ez": (self.Ez, "Ez", lambda dim: -1 if (dim == 2) else +1),
            "Hx": (self.Hx, "Hx", lambda dim: +1 if (dim == 0) else -1),
            "Hy": (self.Hy, "Hy", lambda dim: +1 if (dim == 1) else -1),
            "Hz": (self.Hz, "Hz", lambda dim: +1 if (dim == 2) else -1),
        }


class FieldData(ElectromagneticFieldData):
    """Data associated with a :class:`.FieldMonitor`: scalar components of E and H fields."""

    monitor: FieldMonitor

    Ex: ScalarFieldDataArray = pd.Field(
        None,
        title="Ex",
        description="Spatial distribution of the x-component of the electric field.",
    )
    Ey: ScalarFieldDataArray = pd.Field(
        None,
        title="Ey",
        description="Spatial distribution of the y-component of the electric field.",
    )
    Ez: ScalarFieldDataArray = pd.Field(
        None,
        title="Ez",
        description="Spatial distribution of the z-component of the electric field.",
    )
    Hx: ScalarFieldDataArray = pd.Field(
        None,
        title="Hx",
        description="Spatial distribution of the x-component of the magnetic field.",
    )
    Hy: ScalarFieldDataArray = pd.Field(
        None,
        title="Hy",
        description="Spatial distribution of the y-component of the magnetic field.",
    )
    Hz: ScalarFieldDataArray = pd.Field(
        None,
        title="Hz",
        description="Spatial distribution of the z-component of the magnetic field.",
    )

    _contains_monitor_fields = enforce_monitor_fields_present()

    def normalize(self, source_spectrum_fn) -> "FieldData":
        """Return copy of self after normalization is applied using source spectrum function."""
        src_amps = source_spectrum_fn(self.f)
        field_norm = {name: val / src_amps for name, (val, _, _) in self.field_components.items()}
        return self.copy(update=field_norm)


class FieldTimeData(ElectromagneticFieldData):
    """Data associated with a :class:`.FieldTimeMonitor`: scalar components of E and H fields."""

    monitor: FieldTimeMonitor

    Ex: ScalarFieldTimeDataArray = pd.Field(
        None,
        title="Ex",
        description="Spatial distribution of the x-component of the electric field.",
    )
    Ey: ScalarFieldTimeDataArray = pd.Field(
        None,
        title="Ey",
        description="Spatial distribution of the y-component of the electric field.",
    )
    Ez: ScalarFieldTimeDataArray = pd.Field(
        None,
        title="Ez",
        description="Spatial distribution of the z-component of the electric field.",
    )
    Hx: ScalarFieldTimeDataArray = pd.Field(
        None,
        title="Hx",
        description="Spatial distribution of the x-component of the magnetic field.",
    )
    Hy: ScalarFieldTimeDataArray = pd.Field(
        None,
        title="Hy",
        description="Spatial distribution of the y-component of the magnetic field.",
    )
    Hz: ScalarFieldTimeDataArray = pd.Field(
        None,
        title="Hz",
        description="Spatial distribution of the z-component of the magnetic field.",
    )

    _contains_monitor_fields = enforce_monitor_fields_present()


class ModeFieldData(ElectromagneticFieldData):
    """Data associated with a :class:`.ModeFieldMonitor`: scalar components of E and H fields."""

    monitor: ModeFieldMonitor

    Ex: ScalarModeFieldDataArray = pd.Field(
        ...,
        title="Ex",
        description="Spatial distribution of the x-component of the electric field of the mode.",
    )
    Ey: ScalarModeFieldDataArray = pd.Field(
        ...,
        title="Ey",
        description="Spatial distribution of the y-component of the electric field of the mode.",
    )
    Ez: ScalarModeFieldDataArray = pd.Field(
        ...,
        title="Ez",
        description="Spatial distribution of the z-component of the electric field of the mode.",
    )
    Hx: ScalarModeFieldDataArray = pd.Field(
        ...,
        title="Hx",
        description="Spatial distribution of the x-component of the magnetic field of the mode.",
    )
    Hy: ScalarModeFieldDataArray = pd.Field(
        ...,
        title="Hy",
        description="Spatial distribution of the y-component of the magnetic field of the mode.",
    )
    Hz: ScalarModeFieldDataArray = pd.Field(
        ...,
        title="Hz",
        description="Spatial distribution of the z-component of the magnetic field of the mode.",
    )


class PermittivityData(MonitorData):
    """Data for a :class:`.PermittivityMonitor`: diagonal components of the permittivity tensor."""

    monitor: PermittivityMonitor

    @property
    def field_components(self) -> Dict[str, Tuple[DataArray, str, Callable[[Axis], float]]]:
        """Maps field_name to (scalar data, grid key, function of dim giving symmetry eigenvalue."""
        return {
            "eps_xx": (self.eps_xx, "Ex", lambda dim: 1),
            "eps_yy": (self.eps_yy, "Ey", lambda dim: 1),
            "eps_zz": (self.eps_zz, "Ez", lambda dim: 1),
        }

    eps_xx: ScalarFieldDataArray = pd.Field(
        ...,
        title="Epsilon xx",
        description="Spatial distribution of the x-component of the electric field.",
    )
    eps_yy: ScalarFieldDataArray = pd.Field(
        ...,
        title="Epsilon yy",
        description="Spatial distribution of the y-component of the electric field.",
    )
    eps_zz: ScalarFieldDataArray = pd.Field(
        ...,
        title="Epsilon zz",
        description="Spatial distribution of the z-component of the electric field.",
    )


class ModeData(MonitorData):
    """Data associated with a :class:`.ModeMonitor`: modal amplitudes and propagation indices."""

    monitor: ModeMonitor
    amps: ModeAmpsDataArray = pd.Field(
        ..., title="Amplitudes", description="Complex-valued amplitudes associated with the mode."
    )
    n_complex: ModeIndexDataArray = pd.Field(
        ...,
        title="Amplitudes",
        description="Complex-valued effective propagation constants associated with the mode.",
    )

    def normalize(self, source_spectrum_fn) -> "ModeData":
        """Return copy of self after normalization is applied using source spectrum function."""
        source_freq_amps = source_spectrum_fn(self.f)[None, :, None]
        return self.copy(update={"amps": self.amps / source_freq_amps})


class FluxData(MonitorData):
    """Data associated with a :class:`.FluxMonitor`: flux data in the frequency-domain."""

    monitor: FluxMonitor
    flux: FluxDataArray

    def normalize(self, source_spectrum_fn) -> "Self":
        """Return copy of self after normalization is applied using source spectrum function."""
        source_power = abs(source_freq_amps) ** 2
        return self.copy(update={"flux": self.flux / source_power})


class FluxTimeData(MonitorData):
    """Data associated with a :class:`.FluxTimeMonitor`: flux data in the time-domain."""

    monitor: FluxTimeMonitor
    flux: FluxTimeDataArray


MonitorDataType = Annotated[
    Union[
        FieldData, FieldTimeData, PermittivityData, ModeFieldData, ModeData, FluxData, FluxTimeData
    ],
    pd.Field(discriminator=TYPE_TAG_STR),
]
