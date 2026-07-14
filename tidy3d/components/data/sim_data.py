"""Simulation Level Data"""

from __future__ import annotations

import json
import pathlib
import re
import types
from abc import ABC
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any, Union, get_args, get_origin

import h5py
import numpy as np
import xarray as xr
from pydantic import Field

from tidy3d.components.autograd.flux_monitor import is_flux_adjoint_helper_name
from tidy3d.components.autograd.utils import split_list
from tidy3d.components.base import (
    _LAZY_PROXY_UNHANDLED,
    Tidy3dBaseModel,
    _make_lazy_proxy,
    cached_property,
)
from tidy3d.components.base_sim.data.sim_data import AbstractSimulationData
from tidy3d.components.file_util import json_string_from_hdf5
from tidy3d.components.grid.grid_spec import GridSpec
from tidy3d.components.simulation import Simulation
from tidy3d.components.source.current import CustomCurrentSource
from tidy3d.components.source.time import GaussianPulse
from tidy3d.components.source.utils import GaussianBeamType, SourceType
from tidy3d.components.structure import Structure
from tidy3d.components.types.base import discriminated_union
from tidy3d.components.types.monitor_data import MonitorDataType, MonitorDataTypes
from tidy3d.components.viz import add_ax_if_none, equal_aspect
from tidy3d.exceptions import AdjointError, DataError, SetupError, Tidy3dKeyError
from tidy3d.log import log

from .data_array import FreqDataArray, TimeDataArray, _TracedDataset
from .monitor_data import (
    AbstractFieldData,
    FieldTimeData,
    PointCloudFieldData,
    PointCloudPermittivityData,
)
from .utils import static_dataarray_for_plot

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from os import PathLike

    from matplotlib.colors import Colormap
    from numpy.typing import NDArray

    from tidy3d.compat import Self
    from tidy3d.components.monitor import Monitor
    from tidy3d.components.types import Ax, Axis, ColormapType, FieldVal, PlotScale

    from .data_array import DataArray


def _flatten_monitor_annotation(annotation: Any) -> tuple[type, ...]:
    """Flatten monitor-field annotations into concrete monitor classes."""
    origin = get_origin(annotation)
    if origin in (Union, types.UnionType):
        return tuple(
            child for arg in get_args(annotation) for child in _flatten_monitor_annotation(arg)
        )
    if origin is Annotated:
        return _flatten_monitor_annotation(get_args(annotation)[0])
    return (annotation,)


DATA_TYPE_MAP = {}

# maps monitor type (string) to the class of the corresponding data
DATA_TYPE_NAME_MAP = {}
for data_type in MonitorDataTypes:
    for monitor_type in _flatten_monitor_annotation(data_type.model_fields["monitor"].annotation):
        DATA_TYPE_MAP[monitor_type] = data_type
        DATA_TYPE_NAME_MAP[monitor_type.__name__] = data_type

# residuals below this are considered good fits for broadband adjoint source creation
RESIDUAL_CUTOFF_ADJOINT = 1e-6

# for adjoint source, the minimum number of FWIDTH between the center frequency and zero
NUM_ADJOINT_FWIDTH_TO_ZERO = 3
# for broadband adjoint source, the minimum number of FWIDTH to reach the lowest frequency
# that is covered by the broadband pulse
NUM_ADJOINT_FWIDTH_TO_FMIN = 0.5
# If grouped Gaussian-like source center frequencies span more than this fraction of the
# grouped center frequency, use a small multi-frequency source approximation.
GAUSSIAN_WIDE_BANDWIDTH_THRESHOLD = 0.2


class _LazyMonitorDataMap(Mapping[str, MonitorDataType]):
    """Mapping that loads monitor data lazily by name."""

    def __init__(
        self, monitor_names: tuple[str, ...], loader: Callable[[str], MonitorDataType]
    ) -> None:
        self._monitor_names = monitor_names
        self._loader = loader

    def __getitem__(self, monitor_name: str) -> MonitorDataType:
        if monitor_name not in self._monitor_names:
            raise KeyError(monitor_name)
        return self._loader(monitor_name)

    def __contains__(self, monitor_name: object) -> bool:
        return monitor_name in self._monitor_names

    def __iter__(self) -> Iterator[str]:
        return iter(self._monitor_names)

    def __len__(self) -> int:
        return len(self._monitor_names)


class AdjointSourceInfo(Tidy3dBaseModel):
    """Stores information about the adjoint sources to pass to autograd pipeline."""

    sources: tuple[discriminated_union(SourceType), ...] = Field(
        title="Adjoint Sources",
        description="Set of processed sources to include in the adjoint simulation.",
    )

    post_norm: float | FreqDataArray = Field(
        title="Post Normalization Values",
        description="Factor to multiply the adjoint fields by after running "
        "given the adjoint source pipeline used.",
    )

    normalize_sim: bool = Field(
        title="Normalize Adjoint Simulation",
        description="Whether the adjoint simulation needs to be normalized "
        "given the adjoint source pipeline used.",
    )


@dataclass(frozen=True)
class _AdjointSimulationSetupResult:
    """Result of constructing adjoint simulations from traced monitor VJPs."""

    simulations: list[Simulation]
    all_sources_underflowed: bool = False


def _min_adjoint_source_amplitude(simulation: Simulation) -> float:
    """Minimum adjoint source amplitude expected to survive solver source serialization."""
    source_time_dtype = np.float64 if simulation.precision == "double" else np.float32
    return float(np.nextafter(source_time_dtype(0), source_time_dtype(1)))


def _current_dataset_magnitude(src: CustomCurrentSource) -> float:
    """Maximum absolute current amplitude stored in a custom current source dataset."""
    return float(
        max(
            np.max(np.abs(field_component.values))
            for field_component in src.current_dataset.field_components.values()
        )
    )


def _adjoint_source_dispatch_magnitude(src: SourceType) -> float:
    """Effective adjoint source magnitude after source-time and dataset amplitudes."""
    magnitude = abs(src.source_time.amplitude)
    if isinstance(src, CustomCurrentSource):
        magnitude *= _current_dataset_magnitude(src)
    return magnitude


def _is_dispatchable_adjoint_source(src: SourceType, min_amplitude: float) -> bool:
    """Whether an adjoint source magnitude is large enough to dispatch to the solver."""
    return _adjoint_source_dispatch_magnitude(src) >= min_amplitude


def _warn_skipped_adjoint_sources(num_skipped: int, min_amplitude: float) -> None:
    """Warn once if adjoint source contributions were skipped as numerically zero."""
    if not num_skipped:
        return

    log.warning(
        "Skipped %d adjoint source(s) whose effective magnitude underflows solver precision "
        "(%s). These contributions are treated as zero.",
        num_skipped,
        min_amplitude,
        log_once=True,
    )


def _filter_adjoint_sources(
    sources: list[SourceType],
    min_amplitude: float,
) -> tuple[list[SourceType], int]:
    """Drop raw adjoint sources whose effective magnitude is numerically zero."""
    filtered_sources = [
        source for source in sources if _is_dispatchable_adjoint_source(source, min_amplitude)
    ]
    return filtered_sources, len(sources) - len(filtered_sources)


def _filter_adjoint_source_infos(
    adjoint_source_infos: list[AdjointSourceInfo],
    min_amplitude: float,
) -> tuple[list[AdjointSourceInfo], int]:
    """Drop adjoint sources whose effective magnitude is numerically zero."""
    filtered_infos = []
    num_skipped = 0
    for adjoint_source_info in adjoint_source_infos:
        sources = tuple(
            src
            for src in adjoint_source_info.sources
            if _is_dispatchable_adjoint_source(src, min_amplitude)
        )
        num_skipped += len(adjoint_source_info.sources) - len(sources)
        if sources:
            filtered_infos.append(adjoint_source_info.updated_copy(sources=sources))

    return filtered_infos, num_skipped


@dataclass(frozen=True)
class AdjointSourceGroup:
    """Grouped adjoint sources that share a spatial port, with optional metadata."""

    sources: tuple[SourceType, ...]
    metadata: tuple[Any, ...] | None = None


class AbstractYeeGridSimulationData(AbstractSimulationData, ABC):
    """Data from an :class:`.AbstractYeeGridSimulation` involving
    electromagnetic fields on a Yee grid.

    Notes
    -----

        The ``SimulationData`` objects store a copy of the original :class:`.Simulation`:, so it can be recovered if the
        ``SimulationData`` is loaded in a new session and the :class:`.Simulation` is no longer in memory.

        More importantly, the ``SimulationData`` contains a reference to the data for each of the monitors within the
        original :class:`.Simulation`. This data can be accessed directly using the name given to the monitors initially.
    """

    @staticmethod
    def _raise_if_point_cloud_data(monitor_data: MonitorDataType, operation: str) -> None:
        """Reject structured field helpers for indexed point-cloud data."""
        if isinstance(monitor_data, PointCloudFieldData):
            data_kind = "fields"
            example_component = "Ex"
        elif isinstance(monitor_data, PointCloudPermittivityData):
            data_kind = "permittivity components"
            example_component = "eps_xx"
        else:
            return

        monitor_name = monitor_data.monitor.name
        raise DataError(
            f"'{operation}' is not supported for {type(monitor_data).__name__} because "
            f"point-cloud {data_kind} are indexed by 'index' rather than structured 'x', "
            "'y', and 'z' coordinates. Access point-cloud data directly, for example "
            f"sim_data['{monitor_name}'].{example_component}, and use "
            f"sim_data['{monitor_name}'].points for the corresponding coordinates."
        )

    def load_field_monitor(self, monitor_name: str) -> AbstractFieldData:
        """Load monitor and raise exception if not a field monitor."""
        mon_data = self[monitor_name]
        self._raise_if_point_cloud_data(mon_data, "load_field_monitor")
        if not isinstance(mon_data, AbstractFieldData):
            raise DataError(
                f"data for monitor '{monitor_name}' does not contain field data "
                f"as it is a '{type(mon_data)}'."
            )
        return mon_data

    def at_centers(self, field_monitor_name: str) -> xr.Dataset:
        """Return xarray.Dataset representation of field monitor data colocated at Yee cell centers.

        Parameters
        ----------
        field_monitor_name : str
            Name of field monitor used in the original :class:`.Simulation`.

        Returns
        -------
        xarray.Dataset
            Dataset containing all of the fields in the data interpolated to center locations on
            the Yee grid.
        """

        monitor_data = self.load_field_monitor(field_monitor_name)
        return monitor_data.at_coords(monitor_data.colocation_centers)

    def _at_boundaries(self, monitor_data: xr.Dataset) -> xr.Dataset:
        """Return xarray.Dataset representation of field monitor data colocated at Yee cell
        boundaries.

        Parameters
        ----------
        monitor_data : xr.Dataset
            Monitor data to be co-located.

        Returns
        -------
        xarray.Dataset
            Dataset containing all of the fields in the data interpolated to boundary locations on
            the Yee grid.
        """

        if monitor_data.monitor.colocate:
            # TODO: this still errors if monitor_data.colocate is allowed to be ``True`` in the
            # adjoint plugin, and the monitor data is tracked in a gradient computation. It seems
            # interpolating does something to the arrays that preserves the gradient chain.
            return monitor_data.package_colocate_results(monitor_data.field_components)

        # colocate to monitor grid boundaries
        return monitor_data.at_coords(monitor_data.colocation_boundaries)

    def at_boundaries(self, field_monitor_name: str) -> xr.Dataset:
        """Return xarray.Dataset representation of field monitor data colocated at Yee cell
        boundaries.

        Parameters
        ----------
        field_monitor_name : str
            Name of field monitor used in the original :class:`.Simulation`.

        Returns
        -------
        xarray.Dataset
            Dataset containing all of the fields in the data interpolated to boundary locations on
            the Yee grid.
        """

        # colocate to monitor grid boundaries
        return self._at_boundaries(self.load_field_monitor(field_monitor_name))

    def _get_poynting_vector(self, field_monitor_data: AbstractFieldData) -> xr.Dataset:
        """return ``xarray.Dataset`` of the Poynting vector at Yee cell centers.

        Calculated values represent the instantaneous Poynting vector for time-domain fields and the
        complex vector for frequency-domain: ``S = 1/2 E × conj(H)``.

        Only the available components are returned, e.g., if the indicated monitor doesn't include
        field component `"Ex"`, then `"Sy"` and `"Sz"` will not be calculated.

        Parameters
        ----------
        field_monitor_data: AbstractFieldData
            Field monitor data from which to extract Poynting vector.

        Returns
        -------
        xarray.DataArray
            DataArray containing the Poynting vector calculated based on the field components
            colocated at the center locations of the Yee grid.
        """
        field_dataset = self._at_boundaries(field_monitor_data)

        time_domain = isinstance(field_monitor_data, FieldTimeData)

        poynting_components = {}

        dims = "xyz"
        for axis, dim in enumerate(dims):
            dim_1 = dims[axis - 2]
            dim_2 = dims[axis - 1]

            required_components = [f + c for f in "EH" for c in (dim_1, dim_2)]
            if not all(field_cmp in field_dataset for field_cmp in required_components):
                continue

            e_1 = field_dataset.data_vars["E" + dim_1]
            e_2 = field_dataset.data_vars["E" + dim_2]
            h_1 = field_dataset.data_vars["H" + dim_1]
            h_2 = field_dataset.data_vars["H" + dim_2]
            poynting_components["S" + dim] = (
                e_1 * h_2 - e_2 * h_1
                if time_domain
                else 0.5 * (e_1 * h_2.conj() - e_2 * h_1.conj())
            )

            # 2D monitors have grid correction factors that can be different from 1. For Poynting,
            # it is always the product of a primal-located field and dual-located field, so the
            # total grid correction factor is the product of the two
            grid_correction = (
                field_monitor_data.grid_dual_correction * field_monitor_data.grid_primal_correction
            )
            poynting_components["S" + dim] *= grid_correction

        return _TracedDataset(poynting_components)

    def get_poynting_vector(self, field_monitor_name: str) -> xr.Dataset:
        """return ``xarray.Dataset`` of the Poynting vector at Yee cell centers.

        Calculated values represent the instantaneous Poynting vector for time-domain fields and the
        complex vector for frequency-domain: ``S = 1/2 E × conj(H)``.

        Only the available components are returned, e.g., if the indicated monitor doesn't include
        field component `"Ex"`, then `"Sy"` and `"Sz"` will not be calculated.

        Parameters
        ----------
        field_monitor_name : str
            Name of field monitor used in the original :class:`.Simulation`.

        Returns
        -------
        xarray.DataArray
            DataArray containing the Poynting vector calculated based on the field components
            colocated at the center locations of the Yee grid.
        """
        field_monitor_data = self.load_field_monitor(field_monitor_name)
        return self._get_poynting_vector(field_monitor_data=field_monitor_data)

    def _get_scalar_field(
        self,
        field_monitor_name: str,
        field_name: str,
        val: FieldVal,
        phase: float = 0.0,
    ) -> xr.DataArray:
        """return ``xarray.DataArray`` of the scalar field of a given monitor at Yee cell centers.

        Parameters
        ----------
        field_monitor_name : str
            Name of field monitor used in the original :class:`.Simulation`.
        field_name : str
            Name of the derived field component: one of `('E', 'H', 'S', 'Sx', 'Sy', 'Sz')`.
        val : Literal['real', 'imag', 'abs', 'abs^2', 'phase'] = 'real'
            Which part of the field to plot.
        phase : float = 0.0
            Optional phase to apply to result

        Returns
        -------
        xarray.DataArray
            DataArray containing the electric intensity of the field-like monitor.
            Data is interpolated to the center locations on Yee grid.
        """
        field_monitor_data = self.load_field_monitor(field_monitor_name)
        return self._get_scalar_field_from_data(
            field_monitor_data, field_name=field_name, val=val, phase=phase
        )

    def _get_scalar_field_from_data(
        self,
        field_monitor_data: AbstractFieldData,
        field_name: str,
        val: FieldVal,
        phase: float = 0.0,
    ) -> xr.DataArray:
        """return ``xarray.DataArray`` of the scalar field of a given monitor at Yee cell centers.

        Parameters
        ----------
        field_monitor_data : AbstractFieldData
            Field monitor data from which to extract scalar field.
        field_name : str
            Name of the derived field component: one of `('E', 'H', 'S', 'Sx', 'Sy', 'Sz')`.
        val : Literal['real', 'imag', 'abs', 'abs^2', 'phase'] = 'real'
            Which part of the field to plot.
        phase : float = 0.0
            Optional phase to apply to result

        Returns
        -------
        xarray.DataArray
            DataArray containing the electric intensity of the field-like monitor.
            Data is interpolated to the center locations on Yee grid.
        """

        if field_name[0] == "S":
            dataset = self._get_poynting_vector(field_monitor_data)
            if len(field_name) > 1:
                if field_name in dataset:
                    derived_data = dataset[field_name]
                    derived_data.name = field_name
                    return self._field_component_value(derived_data, val)
                raise Tidy3dKeyError(f"Poynting component {field_name} not available")
        else:
            dataset = self._at_boundaries(field_monitor_data)

        dataset = self.apply_phase(data=dataset, phase=phase)

        if field_name in ("E", "H", "S"):
            # Gather vector components
            required_components = [field_name + c for c in "xyz"]
            if not all(field_cmp in dataset for field_cmp in required_components):
                raise DataError(
                    f"Field monitor must contain '{field_name}x', '{field_name}y', and "
                    f"'{field_name}z' fields to compute '{field_name}'."
                )
            field_components = (dataset[c] for c in required_components)

            # Apply the requested transformation
            val = val.lower()
            if val in ("real", "re"):
                derived_data = sum(f.real**2 for f in field_components) ** 0.5
                derived_data.name = f"|Re{{{field_name}}}|"

            elif val in ("imag", "im"):
                derived_data = sum(f.imag**2 for f in field_components) ** 0.5
                derived_data.name = f"|Im{{{field_name}}}|"

            elif val == "abs":
                derived_data = sum(abs(f) ** 2 for f in field_components) ** 0.5
                derived_data.name = f"|{field_name}|"

            elif val == "abs^2":
                derived_data = sum(abs(f) ** 2 for f in field_components)
                if hasattr(derived_data, "name"):
                    derived_data.name = f"|{field_name}|²"

            elif val == "phase":
                raise Tidy3dKeyError(f"Phase is not defined for complex vector {field_name}")

            else:
                raise Tidy3dKeyError(
                    f"'val' of {val} not supported. "
                    "Must be one of 'real', 'imag', 'abs', 'abs^2', or 'phase'."
                )
            return derived_data

        raise Tidy3dKeyError(
            f"Derived field name must be one of 'E', 'H', 'S', 'Sx', 'Sy', or 'Sz', received "
            f"'{field_name}'."
        )

    def get_intensity(self, field_monitor_name: str) -> xr.DataArray:
        """return `xarray.DataArray` of the intensity of a field monitor at Yee cell centers.

        Parameters
        ----------
        field_monitor_name : str
            Name of field monitor used in the original :class:`.Simulation`.

        Returns
        -------
        xarray.DataArray
            DataArray containing the electric intensity of the field-like monitor.
            Data is interpolated to the center locations on Yee grid.
        """
        return self._get_scalar_field(
            field_monitor_name=field_monitor_name, field_name="E", val="abs^2"
        )

    @classmethod
    def _lazy_proxy_copy_state_keys(cls) -> tuple[str, ...]:
        """Preserve selected monitor names when copying a lazy proxy."""

        return ("monitor_names",)

    @classmethod
    def _lazy_proxy_supports_selective_loading(cls, lazy_state: dict[str, Any]) -> bool:
        """Whether the lazy proxy can resolve metadata and monitor data selectively."""

        suffix = pathlib.Path(lazy_state["_lazy_fname"]).suffix
        group_path = cls._construct_group_path(lazy_state["_lazy_group_path"])
        return suffix in {".hdf5", ".h5"} and group_path == "/"

    @classmethod
    def _lazy_proxy_monitor_selection(
        cls, lazy_state: dict[str, Any], model_dict: dict[str, Any]
    ) -> tuple[list[dict[str, Any]], tuple[int, ...]]:
        """Return raw monitor dictionaries and selected indices from lazy-file metadata."""

        monitor_dicts = model_dict["simulation"]["monitors"]
        requested_monitor_names = lazy_state.get("_lazy_monitor_names")
        if requested_monitor_names is None:
            selected_indices = tuple(range(len(monitor_dicts)))
        else:
            selected_indices = cls._selected_monitor_indices(monitor_dicts, requested_monitor_names)
        return monitor_dicts, selected_indices

    @classmethod
    def _lazy_proxy_ensure_monitor_data_map(cls, lazy_state: dict[str, Any]) -> None:
        """Build a lazy monitor-data map without validating simulation monitor metadata."""

        if lazy_state.get("_lazy_monitor_data_map") is not None:
            return

        model_dict = cls.dict_from_file(
            fname=lazy_state["_lazy_fname"],
            group_path=cls._construct_group_path(lazy_state["_lazy_group_path"]),
            load_data_arrays=False,
        )
        monitor_dicts, selected_indices = cls._lazy_proxy_monitor_selection(lazy_state, model_dict)
        selected_monitor_names = tuple(monitor_dicts[index]["name"] for index in selected_indices)
        lazy_state["_lazy_selected_monitor_names"] = selected_monitor_names
        lazy_state["_lazy_monitor_data"] = {}
        lazy_state["_lazy_monitor_data_map"] = _LazyMonitorDataMap(
            selected_monitor_names,
            lambda monitor_name: cls._lazy_proxy_load_monitor_data(lazy_state, monitor_name),
        )

    @classmethod
    def _lazy_proxy_ensure_simulation(cls, lazy_state: dict[str, Any]) -> None:
        """Load and validate simulation metadata for explicit simulation access."""

        if lazy_state.get("_lazy_simulation") is not None:
            return

        model_dict = cls.dict_from_file(
            fname=lazy_state["_lazy_fname"],
            group_path=cls._construct_group_path(lazy_state["_lazy_group_path"]),
            load_data_arrays=False,
        )
        monitor_dicts, selected_indices = cls._lazy_proxy_monitor_selection(lazy_state, model_dict)
        cls._load_selected_simulation_data_arrays(
            lazy_state["_lazy_fname"], model_dict, selected_indices
        )
        if lazy_state.get("_lazy_monitor_names") is not None:
            model_dict["simulation"]["monitors"] = [
                monitor_dicts[index] for index in selected_indices
            ]

        simulation_type = cls.model_fields["simulation"].annotation
        lazy_state["_lazy_simulation"] = simulation_type.model_validate(model_dict["simulation"])

    @classmethod
    def _lazy_proxy_ensure_metadata(cls, lazy_state: dict[str, Any]) -> None:
        """Load all lazy metadata needed by legacy proxy fallbacks."""

        cls._lazy_proxy_ensure_simulation(lazy_state)
        cls._lazy_proxy_ensure_monitor_data_map(lazy_state)

    @classmethod
    def _lazy_proxy_load_monitor_data(
        cls, lazy_state: dict[str, Any], monitor_name: str
    ) -> MonitorDataType:
        """Load and cache one monitor's data without materializing the full model."""

        cls._lazy_proxy_ensure_monitor_data_map(lazy_state)
        selected_monitor_names = lazy_state["_lazy_selected_monitor_names"]
        if monitor_name not in selected_monitor_names:
            raise KeyError(monitor_name)

        loaded_monitor_data = lazy_state["_lazy_monitor_data"]
        if monitor_name not in loaded_monitor_data:
            loaded_monitor_data[monitor_name] = cls.mnt_data_from_file(
                lazy_state["_lazy_fname"],
                mnt_name=monitor_name,
                **lazy_state["_lazy_parse_obj_kwargs"],
            )
        return loaded_monitor_data[monitor_name]

    @classmethod
    def _lazy_proxy_resolve_attr(cls, proxy: Any, name: str, lazy_state: dict[str, Any]) -> Any:
        """Resolve SimulationData metadata attributes lazily without full materialization."""

        if not cls._lazy_proxy_supports_selective_loading(lazy_state):
            return _LAZY_PROXY_UNHANDLED
        if name == "simulation":
            cls._lazy_proxy_ensure_simulation(lazy_state)
            return lazy_state["_lazy_simulation"]
        if name == "monitor_data":
            cls._lazy_proxy_ensure_monitor_data_map(lazy_state)
            return lazy_state["_lazy_monitor_data_map"]
        if name == "get_monitor_by_name":
            cls._lazy_proxy_ensure_simulation(lazy_state)
            return lazy_state["_lazy_simulation"].get_monitor_by_name
        return _LAZY_PROXY_UNHANDLED

    @classmethod
    def _lazy_proxy_materialize(cls, lazy_state: dict[str, Any]) -> Self:
        """Materialize the full selected SimulationData instance for a lazy proxy."""

        return cls.from_file(
            fname=lazy_state["_lazy_fname"],
            group_path=lazy_state["_lazy_group_path"],
            lazy=False,
            monitor_names=lazy_state.get("_lazy_monitor_names"),
            **lazy_state["_lazy_parse_obj_kwargs"],
        )

    @staticmethod
    def _selected_monitor_indices(
        monitor_dicts: list[dict[str, Any]], monitor_names: str | list[str] | tuple[str, ...]
    ) -> tuple[int, ...]:
        """Return monitor indices selected by name, preserving file order."""

        if isinstance(monitor_names, str):
            requested_names = (monitor_names,)
        else:
            requested_names = tuple(dict.fromkeys(monitor_names))

        requested_name_set = set(requested_names)
        selected_indices = tuple(
            index
            for index, monitor_dict in enumerate(monitor_dicts)
            if monitor_dict["name"] in requested_name_set
        )
        found_names = {monitor_dicts[index]["name"] for index in selected_indices}
        missing_names = [name for name in requested_names if name not in found_names]
        if missing_names:
            missing = ", ".join(repr(name) for name in missing_names)
            raise ValueError(f"Monitor name(s) not found in data file: {missing}.")

        return selected_indices

    @classmethod
    def _load_selected_simulation_data_arrays(
        cls, fname: PathLike, model_dict: dict[str, Any], selected_monitor_indices: tuple[int, ...]
    ) -> None:
        """Load simulation metadata arrays without loading monitor result data arrays."""

        selected_monitor_index_set = set(selected_monitor_indices)

        def should_load_path(path: str) -> bool:
            path_parts = path.strip("/").split("/")
            if not path_parts or path_parts[0] != "simulation":
                return False

            if len(path_parts) >= 3 and path_parts[1] == "monitors":
                try:
                    return int(path_parts[2]) in selected_monitor_index_set
                except ValueError:
                    return False

            return True

        cls._load_data_from_file(
            fname=fname,
            model_dict=model_dict,
            group_path="/",
            should_load_path=should_load_path,
        )

    @classmethod
    def mnt_data_from_file(
        cls, fname: PathLike, mnt_name: str, **model_validate_kwargs: Any
    ) -> MonitorDataType:
        """Loads data for a specific monitor from a .hdf5 file with data for a ``SimulationData``.

        Parameters
        ----------
        fname : PathLike
            Full path to an hdf5 file containing :class:`.SimulationData` data.
        mnt_name : str, optional
            ``.name`` of the monitor to load the data from.
        **model_validate_kwargs
            Keyword arguments passed to pydantic's ``model_validate`` method when loading model.

        Returns
        -------
        :class:`MonitorData`
            Monitor data corresponding to the ``mnt_name`` type.

        Example
        -------
        >>> field_data = your_simulation_data.from_file(fname='folder/data.hdf5', mnt_name="field") # doctest: +SKIP
        """

        if pathlib.Path(fname).suffix not in {".hdf5", ".h5"}:
            raise ValueError("'mnt_data_from_file' only works with '.hdf5' or '.h5' files.")

        # open file and ensure it has data
        with h5py.File(fname) as f_handle:
            if "data" not in f_handle:
                raise ValueError(f"could not find data in the supplied file {fname}")

            # get the monitor list from the json string
            json_string = json_string_from_hdf5(f_handle)
            json_dict = json.loads(json_string)
            monitor_list = json_dict["simulation"]["monitors"]

            monitor_index = cls._selected_monitor_indices(monitor_list, mnt_name)[0]
            monitor_index_str = str(monitor_index)
            if monitor_index_str not in f_handle["data"]:
                raise ValueError(f"No monitor with name '{mnt_name}' found in data file.")

            monitor_type_str = monitor_list[monitor_index]["type"]
            if monitor_type_str not in DATA_TYPE_NAME_MAP:
                raise ValueError(f"Could not find data type '{monitor_type_str}'.")
            monitor_data_type = DATA_TYPE_NAME_MAP[monitor_type_str]

            # load the monitor data from the file using the group_path
            group_path = f"data/{monitor_index_str}"
            return monitor_data_type.from_hdf5(
                f_handle, group_path=group_path, **model_validate_kwargs
            )

    @classmethod
    def from_file(
        cls,
        fname: PathLike,
        group_path: str | None = None,
        lazy: bool = False,
        on_load: Callable[[Any], None] | None = None,
        *,
        monitor_names: str | list[str] | tuple[str, ...] | None = None,
        **parse_obj_kwargs: Any,
    ) -> Self:
        """Load a SimulationData file, optionally materializing only selected monitors."""

        if monitor_names is None:
            return super().from_file(
                fname=fname,
                group_path=group_path,
                lazy=lazy,
                on_load=on_load,
                **parse_obj_kwargs,
            )

        if pathlib.Path(fname).suffix not in {".hdf5", ".h5"}:
            raise ValueError("'monitor_names' only works with '.hdf5' or '.h5' files.")

        group_path = cls._construct_group_path(group_path)
        if group_path != "/":
            raise ValueError("'monitor_names' can only be used when loading the full file.")

        if lazy:
            Proxy = _make_lazy_proxy(cls, on_load=on_load)
            return Proxy(fname, group_path, parse_obj_kwargs, monitor_names=monitor_names)

        model_dict = cls.dict_from_file(fname=fname, group_path=group_path, load_data_arrays=False)
        monitor_dicts = model_dict["simulation"]["monitors"]
        selected_indices = cls._selected_monitor_indices(monitor_dicts, monitor_names)

        selected_roots = {f"data/{index}" for index in selected_indices}
        selected_index_set = set(selected_indices)

        def should_load_path(subpath: str) -> bool:
            normalized_subpath = subpath.strip("/")
            path_parts = normalized_subpath.split("/")
            if len(path_parts) >= 3 and path_parts[:2] == ["simulation", "monitors"]:
                try:
                    return int(path_parts[2]) in selected_index_set
                except ValueError:
                    return False
            if not normalized_subpath.startswith("data/"):
                return True
            return any(
                normalized_subpath == selected_root
                or normalized_subpath.startswith(f"{selected_root}/")
                for selected_root in selected_roots
            )

        cls._load_data_from_file(
            fname=fname,
            model_dict=model_dict,
            group_path=group_path,
            should_load_path=should_load_path,
        )

        model_dict["simulation"]["monitors"] = [monitor_dicts[index] for index in selected_indices]
        model_dict["data"] = [model_dict["data"][index] for index in selected_indices]

        obj = cls._validate_model_dict(model_dict, **parse_obj_kwargs)
        if on_load is not None:
            on_load(obj)
        return obj

    @staticmethod
    def apply_phase(data: xr.DataArray | xr.Dataset, phase: float = 0.0) -> xr.DataArray:
        """Apply a phase to xarray data."""
        if phase != 0.0:
            if np.any(np.iscomplex(data.values)):
                data *= np.exp(1j * phase)
            else:
                log.warning(
                    f"Non-zero phase of {phase} specified but the data being plotted is "
                    "real-valued. The phase will be ignored in the plot."
                )
        return data

    def plot_field_monitor_data(
        self,
        field_monitor_data: AbstractFieldData,
        field_name: str,
        val: FieldVal = "real",
        scale: PlotScale = "lin",
        eps_alpha: float = 0.2,
        phase: float = 0.0,
        robust: bool = True,
        vmin: float | None = None,
        vmax: float | None = None,
        ax: Ax = None,
        shading: str = "flat",
        cmap: str | Colormap | None = None,
        **sel_kwargs: Any,
    ) -> Ax:
        """Plot the field data for a monitor with simulation plot overlaid.

        Parameters
        ----------
        field_monitor_data : AbstractFieldData
            Field monitor data to plot.
        field_name : str
            Name of ``field`` component to plot (eg. `'Ex'`).
            Also accepts ``'E'`` and ``'H'`` to plot the vector magnitudes of the electric and
            magnetic fields, and ``'S'`` for the Poynting vector.
        val : Literal['real', 'imag', 'abs', 'abs^2', 'phase'] = 'real'
            Which part of the field to plot.
        scale : Literal['lin', 'dB']
            Plot in linear or logarithmic (dB) scale.
        eps_alpha : float = 0.2
            Opacity of the structure permittivity.
            Must be between 0 and 1 (inclusive).
        phase : float = 0.0
            Optional phase (radians) to apply to the fields.
            Only has an effect on frequency-domain fields.
        robust : bool = True
            If True and vmin or vmax are absent, uses the 2nd and 98th percentiles of the data
            to compute the color limits. This helps in visualizing the field patterns especially
            in the presence of a source.
        vmin : float = None
            The lower bound of data range that the colormap covers. If ``None``, they are
            inferred from the data and other keyword arguments.
        vmax : float = None
            The upper bound of data range that the colormap covers. If ``None``, they are
            inferred from the data and other keyword arguments.
        ax : matplotlib.axes._subplots.Axes = None
            matplotlib axes to plot on, if not specified, one is created.
        shading: str = 'flat'
            Shading argument for Xarray plot method ('flat','nearest','goraud')
        cmap : Optional[Union[str, Colormap]] = None
            Colormap for visualizing the field values. ``None`` uses the default which infers it from the data.
        sel_kwargs : keyword arguments used to perform ``.sel()`` selection in the monitor data.
            These kwargs can select over the spatial dimensions (``x``, ``y``, ``z``),
            frequency or time dimensions (``f``, ``t``) or ``mode_index``, if applicable.
            For the plotting to work appropriately, the resulting data after selection must contain
            only two coordinates with len > 1.
            Furthermore, these should be spatial coordinates (``x``, ``y``, or ``z``).

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.
        """
        self._raise_if_point_cloud_data(field_monitor_data, "plot_field")

        # get the DataArray corresponding to the monitor_name and field_name
        if field_name in ("E", "H") or field_name[0] == "S":
            # Derived fields
            field_data = self._get_scalar_field_from_data(
                field_monitor_data, field_name, val, phase=phase
            )
        else:
            # Direct field component (e.g. Ex)
            if field_name not in field_monitor_data.field_components:
                raise DataError(f"field_name '{field_name}' not found in data.")
            field_component = field_monitor_data.field_components[field_name]
            field_component.name = field_name
            field_component = self.apply_phase(data=field_component, phase=phase)
            field_data = self._field_component_value(field_component, val)

        if scale == "dB":
            if val == "phase":
                log.warning("Plotting phase component in log scale masks the phase sign.")
            db_factor = {
                ("S", "real"): 10,
                ("S", "imag"): 10,
                ("S", "abs"): 10,
                ("S", "abs^2"): 5,
                ("S", "phase"): 1,
                ("E", "abs^2"): 10,
                ("H", "abs^2"): 10,
            }.get((field_name[0], val), 20)
            field_data = self._apply_log_scale(field_data, vmin=vmin, db_factor=db_factor)
            field_data.name += " (dB)"
            cmap_type = "sequential"
        elif scale == "lin":
            cmap_type = (
                "cyclic"
                if val == "phase"
                else (
                    "divergent"
                    if len(field_name) == 2 and val in ("real", "imag", "re", "im")
                    else "sequential"
                )
            )
        else:
            raise SetupError(f"The scale '{scale}' is not supported for plotting field data.")

        # interp out any monitor.size==0 dimensions
        monitor = field_monitor_data.monitor
        thin_dims = {
            "xyz"[dim]: monitor.center[dim]
            for dim in range(3)
            if monitor.size[dim] == 0 and "xyz"[dim] not in sel_kwargs
        }
        for axis, pos in thin_dims.items():
            if axis not in field_data.coords:
                continue
            if field_data.coords[axis].size <= 1:
                field_data = field_data.sel(**{axis: pos}, method="nearest")
            else:
                field_data = field_data.interp(**{axis: pos}, kwargs={"bounds_error": True})

        # select the extra coordinates out of the data from user-specified kwargs
        for coord_name, coord_val in sel_kwargs.items():
            interp_val = np.array(coord_val)
            if interp_val.size == 1:
                interp_val = interp_val.item()
            if (
                field_data.coords[coord_name].size <= 1
                or coord_name == "eme_port_index"
                or coord_name == "eme_cell_index"
                or coord_name == "sweep_index"
                or coord_name == "mode_index"
            ):
                field_data = field_data.sel(**{coord_name: interp_val}, method=None)
            else:
                field_data = field_data.interp(
                    **{coord_name: interp_val}, kwargs={"bounds_error": True}
                )

        # before dropping coordinates, check if a frequency can be derived from the data that can
        # be used to plot material permittivity
        if "f" in sel_kwargs:
            freq_eps_eval = sel_kwargs["f"]
        elif "f" in field_data.coords:
            freq_eps_eval = field_data.coords["f"].values[0]
        else:
            freq_eps_eval = None

        field_data = field_data.squeeze(drop=True)
        non_scalar_coords = {name: c for name, c in field_data.coords.items() if c.size > 1}

        # assert the data is valid for plotting
        if len(non_scalar_coords) != 2:
            raise DataError(
                f"Data after selection has {len(non_scalar_coords)} coordinates "
                f"({list(non_scalar_coords.keys())}), "
                "must be 2 spatial coordinates for plotting on plane. "
                "Please add keyword arguments to `plot_field()` to select out the other coords."
            )

        spatial_coords_in_data = {
            coord_name: (coord_name in non_scalar_coords) for coord_name in "xyz"
        }

        if sum(spatial_coords_in_data.values()) != 2:
            raise DataError(
                "All coordinates in the data after selection must be spatial (x, y, z), "
                f" given {non_scalar_coords.keys()}."
            )

        # get the spatial coordinate corresponding to the plane
        planar_coord = [name for name, c in spatial_coords_in_data.items() if c is False][0]
        axis = "xyz".index(planar_coord)
        if planar_coord in field_data.coords:
            position = float(field_data.coords[planar_coord])
        else:
            position = monitor.center[axis]

        return self.plot_scalar_array(
            field_data=field_data,
            axis=axis,
            position=position,
            freq=freq_eps_eval,
            eps_alpha=eps_alpha,
            robust=robust,
            vmin=vmin,
            vmax=vmax,
            cmap_type=cmap_type,
            ax=ax,
            shading=shading,
            cmap=cmap,
            infer_intervals=True if shading == "flat" else False,
        )

    def plot_field(
        self,
        field_monitor_name: str,
        field_name: str,
        val: FieldVal = "real",
        scale: PlotScale = "lin",
        eps_alpha: float = 0.2,
        phase: float = 0.0,
        robust: bool = True,
        vmin: float | None = None,
        vmax: float | None = None,
        ax: Ax = None,
        shading: str = "flat",
        cmap: str | Colormap | None = None,
        **sel_kwargs: Any,
    ) -> Ax:
        """Plot the field data for a monitor with simulation plot overlaid.

        Parameters
        ----------
        field_monitor_name : str
            Name of :class:`.FieldMonitor`, :class:`.FieldTimeData`, or
            :class:`~tidy3d.ModeSolverData`
            to plot.
        field_name : str
            Name of ``field`` component to plot (eg. `'Ex'`).
            Also accepts ``'E'`` and ``'H'`` to plot the vector magnitudes of the electric and
            magnetic fields, and ``'S'`` for the Poynting vector.
        val : Literal['real', 'imag', 'abs', 'abs^2', 'phase'] = 'real'
            Which part of the field to plot.
        scale : Literal['lin', 'dB']
            Plot in linear or logarithmic (dB) scale.
        eps_alpha : float = 0.2
            Opacity of the structure permittivity.
            Must be between 0 and 1 (inclusive).
        phase : float = 0.0
            Optional phase (radians) to apply to the fields.
            Only has an effect on frequency-domain fields.
        robust : bool = True
            If True and vmin or vmax are absent, uses the 2nd and 98th percentiles of the data
            to compute the color limits. This helps in visualizing the field patterns especially
            in the presence of a source.
        vmin : float = None
            The lower bound of data range that the colormap covers. If ``None``, they are
            inferred from the data and other keyword arguments.
        vmax : float = None
            The upper bound of data range that the colormap covers. If ``None``, they are
            inferred from the data and other keyword arguments.
        ax : matplotlib.axes._subplots.Axes = None
            matplotlib axes to plot on, if not specified, one is created.
        shading: str = 'flat'
            Shading argument for Xarray plot method ('flat','nearest','goraud')
        cmap : Optional[Union[str, Colormap]] = None
            Colormap for visualizing the field values. ``None`` uses the default which infers it from the data.
        sel_kwargs : keyword arguments used to perform ``.sel()`` selection in the monitor data.
            These kwargs can select over the spatial dimensions (``x``, ``y``, ``z``),
            frequency or time dimensions (``f``, ``t``) or ``mode_index``, if applicable.
            For the plotting to work appropriately, the resulting data after selection must contain
            only two coordinates with len > 1.
            Furthermore, these should be spatial coordinates (``x``, ``y``, or ``z``).

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.
        """

        field_monitor_data = self.load_field_monitor(field_monitor_name)

        return self.plot_field_monitor_data(
            field_monitor_data=field_monitor_data,
            field_name=field_name,
            val=val,
            scale=scale,
            eps_alpha=eps_alpha,
            phase=phase,
            robust=robust,
            vmin=vmin,
            vmax=vmax,
            ax=ax,
            shading=shading,
            cmap=cmap,
            **sel_kwargs,
        )

    @equal_aspect
    @add_ax_if_none
    def plot_scalar_array(
        self,
        field_data: xr.DataArray,
        axis: Axis,
        position: float,
        freq: float | None = None,
        eps_alpha: float = 0.2,
        robust: bool = True,
        vmin: float | None = None,
        vmax: float | None = None,
        cmap_type: ColormapType = "divergent",
        cmap: str | Colormap | None = None,
        ax: Ax = None,
        **kwargs: Any,
    ) -> Ax:
        """Plot the field data for a monitor with simulation plot overlaid.

        Parameters
        ----------
        field_data: xr.DataArray
            DataArray with the field data to plot.
            Must be a scalar field.
        axis: Axis
            Axis normal to the plotting plane.
        position: float
            Position along the axis.
        freq: float = None
            Frequency at which the permittivity is evaluated at (if dispersive).
            By default, chooses permittivity as frequency goes to infinity.
        eps_alpha : float = 0.2
            Opacity of the structure permittivity.
            Must be between 0 and 1 (inclusive).
        robust : bool = True
            If True and vmin or vmax are absent, uses the 2nd and 98th percentiles of the data
            to compute the color limits. This helps in visualizing the field patterns especially
            in the presence of a source.
        vmin : float = None
            The lower bound of data range that the colormap covers. If `None`, they are
            inferred from the data and other keyword arguments.
        vmax : float = None
            The upper bound of data range that the colormap covers. If `None`, they are
            inferred from the data and other keyword arguments.
        cmap_type : Literal["divergent", "sequential", "cyclic"] = "divergent"
            Type of color map to use for plotting.
        cmap : Optional[Union[str, Colormap]] = None
            Colormap for visualizing the field values. ``None`` uses the default which infers it from the data. Overrides inferred colormap from `cmap_type`.
        ax : matplotlib.axes._subplots.Axes = None
            matplotlib axes to plot on, if not specified, one is created.
        **kwargs : Extra arguments to ``DataArray.plot``.

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.
        """

        # select the cross section data
        interp_kwarg = {"xyz"[axis]: position}

        if cmap_type == "divergent":
            default_cmap = "RdBu"
            center = 0.0
            eps_reverse = False
        elif cmap_type == "sequential":
            default_cmap = "magma"
            center = False
            eps_reverse = True
        elif cmap_type == "cyclic":
            default_cmap = "twilight"
            vmin = -np.pi
            vmax = np.pi
            center = False
            eps_reverse = False
        else:
            default_cmap = None

        cmap_to_use = default_cmap if cmap is None else cmap

        # plot the field
        xy_coord_labels = list("xyz")
        xy_coord_labels.pop(axis)
        x_coord_label, y_coord_label = xy_coord_labels[0], xy_coord_labels[1]
        field_data_plot = static_dataarray_for_plot(field_data)
        field_data_plot.plot(
            ax=ax,
            x=x_coord_label,
            y=y_coord_label,
            cmap=cmap_to_use,
            vmin=vmin,
            vmax=vmax,
            robust=robust,
            center=center,
            cbar_kwargs={"label": field_data.name},
            **kwargs,
        )

        # plot the simulation epsilon
        ax = self.simulation.plot_structures_eps(
            freq=freq,
            cbar=False,
            alpha=eps_alpha,
            reverse=eps_reverse,
            ax=ax,
            **interp_kwarg,
        )

        # set the limits based on the xarray coordinates min and max
        x_coord_values = field_data_plot.coords[x_coord_label]
        y_coord_values = field_data_plot.coords[y_coord_label]
        ax.set_xlim(min(x_coord_values), max(x_coord_values))
        ax.set_ylim(min(y_coord_values), max(y_coord_values))

        return ax


class SimulationData(AbstractYeeGridSimulationData):
    """Stores data from a collection of :class:`.Monitor` objects in a :class:`.Simulation`.

    Notes
    -----

        The ``SimulationData`` objects store a copy of the original :class:`.Simulation`:, so it can be recovered if the
        ``SimulationData`` is loaded in a new session and the :class:`.Simulation` is no longer in memory.

        More importantly, the ``SimulationData`` contains a reference to the data for each of the monitors within the
        original :class:`.Simulation`. This data can be accessed directly using the name given to the monitors initially.

    Examples
    --------

    Standalone example:

    >>> import tidy3d as td
    >>> num_modes = 5
    >>> x = [-1,1,3]
    >>> y = [-2,0,2,4]
    >>> z = [-3,-1,1,3,5]
    >>> f = [2e14, 3e14]
    >>> coords = dict(x=x[:-1], y=y[:-1], z=z[:-1], f=f)
    >>> grid = td.Grid(boundaries=td.Coords(x=x, y=y, z=z))
    >>> scalar_field = td.ScalarFieldDataArray((1+1j) * np.random.random((2,3,4,2)), coords=coords)
    >>> field_monitor = td.FieldMonitor(
    ...     size=(2,4,6),
    ...     freqs=[2e14, 3e14],
    ...     name='field',
    ...     fields=['Ex'],
    ...     colocate=True,
    ... )
    >>> sim = td.Simulation(
    ...     size=(2, 4, 6),
    ...     grid_spec=td.GridSpec(wavelength=1.0),
    ...     monitors=[field_monitor],
    ...     run_time=2e-12,
    ...     sources=[
    ...         td.UniformCurrentSource(
    ...             size=(0, 0, 0),
    ...             center=(0, 0.5, 0),
    ...             polarization="Hx",
    ...             source_time=td.GaussianPulse(
    ...                 freq0=2e14,
    ...                 fwidth=4e13,
    ...             ),
    ...             current_amplitude_definition="total",
    ...         )
    ...     ],
    ... )
    >>> field_data = td.FieldData(monitor=field_monitor, Ex=scalar_field, grid_expanded=grid)
    >>> sim_data = td.SimulationData(simulation=sim, data=(field_data,))

    To save and load the :class:`SimulationData` object.

    .. code-block:: python

        sim_data.to_file(fname='path/to/file.hdf5') # Save a SimulationData object to a HDF5 file
        sim_data = SimulationData.from_file(fname='path/to/file.hdf5') # Load a SimulationData object from a HDF5 file.

    Optionally, the simulation data can be loaded in a lazy mode, which only holds a reference until a field is accessed
    or a method is applied. This is useful to save I/O operations and memory.

    .. code-block:: python

        sim_data = SimulationData.from_file(fname='path/to/file.hdf5', lazy=True) # Does not contain data until accessed.

    See Also
    --------

    **Notebooks:**
        * `Quickstart <../../notebooks/StartHere.html>`_: Usage in a basic simulation flow.
        * `Performing visualization of simulation data <../../notebooks/VizData.html>`_
        * `Advanced monitor data manipulation and visualization <../../notebooks/XarrayTutorial.html>`_

    """

    simulation: Simulation = Field(
        title="Simulation",
        description="Original :class:`.Simulation` associated with the data.",
    )

    data: tuple[discriminated_union(MonitorDataType), ...] = Field(
        title="Monitor Data",
        description="List of :class:`.MonitorData` instances "
        "associated with the monitors of the original :class:`.Simulation`.",
    )

    diverged: bool = Field(
        False,
        title="Diverged",
        description="A boolean flag denoting whether the simulation run diverged.",
    )

    @cached_property
    def field_decay(self) -> TimeDataArray:
        """Returns a TimeDataArray of field decay values over time steps."""
        log_str = self.log
        if log_str is None:
            raise DataError(
                "No log string in the SimulationData object, can't extract field decay."
            )

        matches = re.findall(r"- Time step\s+(\d+)\s+/.*?field decay:\s*([0-9.eE+-]+)", log_str)

        steps = [int(m[0]) for m in matches]
        decays = [float(m[1]) for m in matches]
        return TimeDataArray(decays, coords={"t": steps})

    @property
    def final_decay_value(self) -> float:
        """Returns value of the field decay at the final time step."""
        field_decay = self.field_decay
        if len(field_decay) == 0:
            log.warning("No field decay values found, using 1.0 as final decay value.")
            return 1.0
        return float(field_decay.values[-1])

    def source_spectrum(self, source_index: int) -> Callable:
        """Get a spectrum normalization function for a given source index."""

        if source_index is None or len(self.simulation.sources) == 0:
            return np.ones_like

        source = self.simulation.sources[source_index]
        source_time = source.source_time
        times = self.simulation.tmesh
        dt = self.simulation.dt

        # plug in mornitor_data frequency domain information
        def source_spectrum_fn(freqs: DataArray) -> NDArray:
            """Source amplitude as function of frequency."""
            spectrum = source_time.spectrum(times, freqs, dt)

            # Remove user defined amplitude and phase from the normalization
            # such that they would still have an effect on the output fields.
            # In other words, we are only normalizing out the arbitrary part of the spectrum
            # that depends on things like freq0, fwidth and offset.
            return spectrum / source_time.amplitude / np.exp(1j * source_time.phase)

        return source_spectrum_fn

    def renormalize(self, normalize_index: int) -> SimulationData:
        """Return a copy of the :class:`.SimulationData` with a different source used for the
        normalization."""

        num_sources = len(self.simulation.sources)
        if normalize_index == self.simulation.normalize_index or num_sources == 0:
            # already normalized to that index
            return self.copy()

        if normalize_index and (normalize_index < 0 or normalize_index >= num_sources):
            # normalize index out of bounds for source list
            raise DataError(
                f"normalize_index {normalize_index} out of bounds for list of sources "
                f"of length {num_sources}"
            )

        def source_spectrum_fn(freqs: DataArray) -> NDArray:
            """Normalization function that also removes previous normalization if needed."""
            new_spectrum_fn = self.source_spectrum(normalize_index)
            old_spectrum_fn = self.source_spectrum(self.simulation.normalize_index)
            return new_spectrum_fn(freqs) / old_spectrum_fn(freqs)

        # Make a new monitor_data dictionary with renormalized data
        data_normalized = tuple(mnt_data.normalize(source_spectrum_fn) for mnt_data in self.data)

        simulation = self.simulation.copy(deep=False, update={"normalize_index": normalize_index})

        return self.copy(deep=False, update={"simulation": simulation, "data": data_normalized})

    def _split_adjoint_data(self: SimulationData, num_mnts_original: int) -> tuple[list, list]:
        """Split data into original and adjoint sections by monitor names."""

        monitors_all = list(self.simulation.monitors)
        monitors_orig, monitors_adjoint = split_list(monitors_all, index=num_mnts_original)

        expected_original_names = [monitor.name for monitor in monitors_orig]
        expected_adjoint_names = [monitor.name for monitor in monitors_adjoint]
        expected_all_names = expected_original_names + expected_adjoint_names
        num_mnts_fld = sum(name.startswith("adjoint_fld_") for name in expected_adjoint_names)
        num_mnts_eps = sum(name.startswith("adjoint_eps_") for name in expected_adjoint_names)
        num_mnts_source_adj = sum(
            name.startswith("source_adjoint_") for name in expected_adjoint_names
        )
        num_mnts_flux_adj = sum(
            is_flux_adjoint_helper_name(name) for name in expected_adjoint_names
        )

        monitor_data_names = [mnt_data.monitor.name for mnt_data in self.data]

        # Use raw monitor_data lookup (not __getitem__) to avoid implicit symmetry expansion.
        monitor_data = self.monitor_data
        data_original = [
            monitor_data[name] for name in expected_original_names if name in monitor_data_names
        ]
        data_adjoint = [
            monitor_data[name] for name in expected_adjoint_names if name in monitor_data_names
        ]

        missing_original = [
            name for name in expected_original_names if name not in monitor_data_names
        ]
        missing_adjoint = [
            name for name in expected_adjoint_names if name not in monitor_data_names
        ]
        monitor_data_known_order = [
            name for name in monitor_data_names if name in expected_all_names
        ]
        expected_known_order = [name for name in expected_all_names if name in monitor_data_names]

        log.info(
            f" -> {num_mnts_original} monitors, {num_mnts_flux_adj} flux adjoint field monitors, "
            f"{num_mnts_fld} adjoint field monitors, "
            f"{num_mnts_source_adj} source adjoint monitors, {num_mnts_eps} adjoint eps monitors."
        )

        if missing_original or len(data_original) < len(expected_original_names):
            log.warning(
                "Combined SimulationData is missing expected original monitor data. "
                f"Expected {len(expected_original_names)} entries, got {len(data_original)}. "
                f"Missing names: {missing_original}."
            )

        if missing_adjoint or len(data_adjoint) < len(expected_adjoint_names):
            log.warning(
                "Combined SimulationData is missing expected adjoint monitor data. "
                f"Expected {len(expected_adjoint_names)} entries, got {len(data_adjoint)}. "
                f"Missing names: {missing_adjoint}."
            )

        if monitor_data_known_order != expected_known_order:
            log.warning(
                "Combined SimulationData monitor data order does not match combined simulation "
                f"monitor order. Expected order: {expected_known_order}, "
                f"got: {monitor_data_known_order}."
            )

        return data_original, data_adjoint

    def _split_original_fwd(self, num_mnts_original: int) -> tuple[SimulationData, SimulationData]:
        """Split this simulation data into original and fwd data from number of original mnts."""

        # split the data and monitors into the original ones & adjoint gradient ones (for 'fwd')
        data_original, data_fwd = self._split_adjoint_data(num_mnts_original=num_mnts_original)
        monitors_orig, monitors_fwd = split_list(self.simulation.monitors, index=num_mnts_original)

        # reconstruct the simulation data for the user, using original sim, and data for original mnts
        sim_original = self.simulation.updated_copy(monitors=monitors_orig)
        sim_data_original = self.updated_copy(
            simulation=sim_original,
            data=data_original,
            deep=False,
        )

        # construct the 'forward' simulation and its data, which is only used for for gradient calc.
        sim_fwd = self.simulation.updated_copy(monitors=monitors_fwd)
        sim_data_fwd = self.updated_copy(
            simulation=sim_fwd,
            data=data_fwd,
            deep=False,
        )

        return sim_data_original, sim_data_fwd

    def _make_adjoint_sims(
        self,
        data_vjp_paths: set[tuple],
        adjoint_monitors: list[Monitor],
    ) -> list[Simulation]:
        """Make the adjoint simulations from the original simulation and the VJP-containing data."""
        return self._make_adjoint_sims_with_result(
            data_vjp_paths=data_vjp_paths,
            adjoint_monitors=adjoint_monitors,
        ).simulations

    def _make_adjoint_sims_with_result(
        self,
        data_vjp_paths: set[tuple],
        adjoint_monitors: list[Monitor],
    ) -> _AdjointSimulationSetupResult:
        """Make adjoint simulations and report if all generated sources underflowed."""

        if not data_vjp_paths:
            return _AdjointSimulationSetupResult([])

        requested_monitor_names = {self.data[index].monitor.name for _, index, _ in data_vjp_paths}

        # generate the adjoint sources {mnt_name : list[Source]}
        sources_adj_dict = self._make_adjoint_sources(data_vjp_paths=data_vjp_paths)
        if not sources_adj_dict:
            return _AdjointSimulationSetupResult([])
        sources_generated_for_all_monitors = requested_monitor_names <= set(sources_adj_dict)

        adj_srcs = []
        for src_list in sources_adj_dict.values():
            adj_srcs += list(src_list)

        if not adj_srcs:
            return _AdjointSimulationSetupResult([])

        min_source_amplitude = _min_adjoint_source_amplitude(self.simulation)
        adj_srcs, num_skipped_before_grouping = _filter_adjoint_sources(
            adj_srcs, min_source_amplitude
        )
        if not adj_srcs:
            _warn_skipped_adjoint_sources(num_skipped_before_grouping, min_source_amplitude)
            return _AdjointSimulationSetupResult(
                [],
                all_sources_underflowed=bool(num_skipped_before_grouping)
                and sources_generated_for_all_monitors,
            )

        adjoint_source_infos = self._process_adjoint_sources(adj_srcs=adj_srcs)
        adjoint_source_infos, num_skipped_after_grouping = _filter_adjoint_source_infos(
            adjoint_source_infos, min_source_amplitude
        )
        num_skipped = num_skipped_before_grouping + num_skipped_after_grouping
        _warn_skipped_adjoint_sources(num_skipped, min_source_amplitude)

        if not adjoint_source_infos:
            return _AdjointSimulationSetupResult(
                [],
                all_sources_underflowed=bool(num_skipped) and sources_generated_for_all_monitors,
            )

        adj_sims = []
        for adjoint_source_info in adjoint_source_infos:
            adj_sims.append(
                make_adjoint_simulation(
                    simulation=self.simulation,
                    adjoint_source_info=adjoint_source_info,
                    adjoint_monitors=adjoint_monitors,
                )
            )

        log.info(f"Created {len(adj_sims)} adjoint simulations.")

        return _AdjointSimulationSetupResult(adj_sims)

    def _make_adjoint_sources(self, data_vjp_paths: set[tuple]) -> dict[str, list[SourceType]]:
        """Generate all of the non-zero sources for the adjoint simulation given the VJP data."""

        # map of index into 'self.data' to the list of datasets we need adjoint sources for
        adj_src_map = defaultdict(list)
        for _, index, dataset_name in data_vjp_paths:
            adj_src_map[index].append(dataset_name)

        # gather a dict of adjoint sources for every monitor data in the VJP that needs one
        sources_adj_all = defaultdict(list)
        for data_index, dataset_names in adj_src_map.items():
            mnt_data = self.data[data_index]
            sources_adj = mnt_data._make_adjoint_sources(
                dataset_names=dataset_names, fwidth=self._fwidth_adj
            )
            log.info(
                f"Created {len(sources_adj)} adjoint sources for monitor '{mnt_data.monitor.name}'."
            )
            if sources_adj:
                sources_adj_all[mnt_data.monitor.name] = sources_adj

        return sources_adj_all

    @property
    def _fwidth_adj(self) -> float:
        # fwidth of forward pass, try as default for adjoint
        normalize_index_fwd = self.simulation.normalize_index or 0
        return self.simulation.sources[normalize_index_fwd].source_time.fwidth

    @staticmethod
    def _adjoint_src_width_single(adj_srcs: list[SourceType]) -> list[SourceType]:
        """Ensure the adjoint source sufficiently decays before zero frequency."""
        adj_srcs_process_fwidth = []
        for adj_src in adj_srcs:
            source_time = adj_src.source_time
            freq0 = source_time._freq0

            fwidth = np.minimum(freq0 / NUM_ADJOINT_FWIDTH_TO_ZERO, source_time.fwidth)

            adj_srcs_process_fwidth.append(
                adj_src.updated_copy(source_time=source_time.updated_copy(fwidth=fwidth))
            )

        return adj_srcs_process_fwidth

    @staticmethod
    def _adjoint_port_group_hashes(
        adj_srcs: list[SourceType], *, adjust_fwidth: bool = True
    ) -> tuple[list[SourceType], list[str]]:
        """Return processed adjoint sources and their spatial-port grouping hashes."""

        processed_sources = (
            SimulationData._adjoint_src_width_single(adj_srcs) if adjust_fwidth else list(adj_srcs)
        )
        min_freq_tmp_src = np.maximum(
            0,
            np.min([src.source_time._freq0 - src.source_time.fwidth for src in processed_sources]),
        )
        max_freq_tmp_src = np.max(
            [src.source_time._freq0 + src.source_time.fwidth for src in processed_sources]
        )
        tmp_src_f0 = 0.5 * (min_freq_tmp_src + max_freq_tmp_src)
        tmp_src_fwidth = max_freq_tmp_src - min_freq_tmp_src
        tmp_src_time = GaussianPulse(freq0=tmp_src_f0, fwidth=tmp_src_fwidth)
        port_hashes = [
            src.updated_copy(source_time=tmp_src_time)._hash_self() for src in processed_sources
        ]
        return processed_sources, port_hashes

    @staticmethod
    def _group_adjoint_sources_by_port(
        adj_srcs: list[SourceType],
        metadata: list[Any] | None = None,
        *,
        adjust_fwidth: bool = True,
    ) -> list[AdjointSourceGroup]:
        """Group adjoint sources by spatial port while preserving optional per-source metadata."""

        if not adj_srcs:
            return []
        if metadata is not None and len(metadata) != len(adj_srcs):
            raise ValueError("'metadata' must have the same length as 'adj_srcs'.")

        processed_sources, port_hashes = SimulationData._adjoint_port_group_hashes(
            adj_srcs, adjust_fwidth=adjust_fwidth
        )
        grouped: dict[str, dict[str, Any]] = {}
        for index, (src, port_hash) in enumerate(zip(processed_sources, port_hashes)):
            group = grouped.setdefault(
                port_hash,
                {"sources": [], "metadata": [] if metadata is not None else None},
            )
            group["sources"].append(src)
            if metadata is not None:
                group["metadata"].append(metadata[index])

        return [
            AdjointSourceGroup(
                sources=tuple(group["sources"]),
                metadata=None if group["metadata"] is None else tuple(group["metadata"]),
            )
            for group in grouped.values()
        ]

    def _process_adjoint_sources(self, adj_srcs: list[SourceType]) -> list[AdjointSourceInfo]:
        """Compute list of final sources along with a post run normalization for adj fields."""
        port_groups = self._group_adjoint_sources_by_port(adj_srcs)
        adj_srcs_process_fwidth = [src for group in port_groups for src in group.sources]

        # Group sources by frequency or port, whichever gives fewer groups
        num_ports = len(port_groups)
        num_unique_freqs = len({src.source_time._freq0 for src in adj_srcs_process_fwidth})

        log.info(f"Found {num_ports} spatial ports and {num_unique_freqs} unique frequencies.")

        adjoint_infos = []
        if num_unique_freqs <= num_ports:
            log.info("Grouping adjoint sources by frequency.")
            unique_freqs = {src.source_time._freq0 for src in adj_srcs_process_fwidth}
            for freq0 in unique_freqs:
                group = [src for src in adj_srcs_process_fwidth if src.source_time._freq0 == freq0]
                post_norm = xr.DataArray(data=np.array([1 + 0j]), coords={"f": [freq0]})
                adjoint_infos.append(
                    AdjointSourceInfo(sources=group, post_norm=post_norm, normalize_sim=True)
                )
        else:
            log.info("Grouping adjoint sources by port.")

            #
            # warn if the forward simulation had symmetry and we are grouping by port, which
            # which means the individual adjoint simulations may not respect the original symmetry
            #
            if np.any(np.abs(self.simulation.symmetry) > 0) and (num_ports > 1):
                log.warning(
                    "The adjoint simulations for this problem are being broken into "
                    "multiple simulations that may not individually respect the symmetry of the "
                    "initial simulation. Gradients may be unreliable and it is recommended to "
                    "optimize this problem without utilizing symmetry."
                )

            for port_group in port_groups:
                processed_srcs, post_norm = self._process_adjoint_sources_broadband(
                    list(port_group.sources)
                )
                adjoint_infos.append(
                    AdjointSourceInfo(
                        sources=processed_srcs, post_norm=post_norm, normalize_sim=True
                    )
                )

        log.info(f"Created {len(adjoint_infos)} adjoint source groups.")
        return adjoint_infos

    def _process_adjoint_sources_broadband(
        self, adj_srcs: list[SourceType]
    ) -> tuple[list[SourceType], xr.DataArray]:
        """Process adjoint sources for the case of several sources at the same freq."""

        src_broadband = self._make_broadband_source(adj_srcs=adj_srcs)
        post_norm_amps = self._make_post_norm_amps(adj_srcs=adj_srcs)

        log.info(
            "Several adjoint sources, from one monitor. "
            "Only difference between them is the source time. "
            "Constructing broadband adjoint source and performing post-run normalization "
            f"of fields with {len(post_norm_amps)} frequencies."
        )

        return [src_broadband], post_norm_amps

    @staticmethod
    def _adjoint_src_width_broadband(adj_srcs: list[SourceType]) -> tuple[float, float]:
        """Find the adjoint source fwidth that sufficiently covers all adjoint frequencies."""

        adj_srcs_f0 = [adj_src.source_time._freq0 for adj_src in adj_srcs]
        middle_f0 = 0.5 * (np.max(adj_srcs_f0) + np.min(adj_srcs_f0))
        min_f0 = np.min(adj_srcs_f0)

        # width of source to sufficiently decay by zero frequency
        decay_by_f0_fwidth = middle_f0 / NUM_ADJOINT_FWIDTH_TO_ZERO
        # width of source to sufficiently cover all adjoint frequencies
        fwidth_to_min_f0 = (middle_f0 - min_f0) / NUM_ADJOINT_FWIDTH_TO_FMIN

        # log warning if the adjoint pulse width is not sufficiently decayed by zero frequency
        # which may cause some issues in the adjoint accuracy when using field sources
        if (fwidth_to_min_f0 > decay_by_f0_fwidth) and isinstance(adj_srcs[0], CustomCurrentSource):
            log.warning(
                "Adjoint source generated with a frequency spectrum that extends to or overlaps with 0 Hz. "
                "This can introduce errors into the gradient computation."
            )

        # Choose a wider pulse width in frequency especially when the min/max frequencies
        # for the broadband pulse might be very close together
        adj_src_fwidth = np.maximum(decay_by_f0_fwidth, fwidth_to_min_f0)

        return middle_f0, adj_src_fwidth

    def _make_broadband_source(self, adj_srcs: list[SourceType]) -> SourceType:
        """Make a broadband source for a set of adjoint sources."""

        adj_src_f0, adj_src_fwidth = self._adjoint_src_width_broadband(adj_srcs)

        source_index = self.simulation.normalize_index or 0

        src_time_base = self.simulation.sources[source_index].source_time.updated_copy(
            amplitude=1.0, phase=0.0
        )
        src_broadband = adj_srcs[0].updated_copy(
            source_time=src_time_base.updated_copy(freq0=adj_src_f0, fwidth=adj_src_fwidth)
        )

        # For grouped Gaussian-like sources, use a small multi-frequency approximation only
        # when the grouped source centers span more than the configured fractional threshold
        # of the broadband center frequency.
        if isinstance(src_broadband, get_args(GaussianBeamType)):
            num_freqs = 1
            if len(adj_srcs) > 1:
                src_freqs = np.array([src.source_time._freq0 for src in adj_srcs], dtype=float)
                freq_span = float(np.max(src_freqs) - np.min(src_freqs))
                if freq_span > GAUSSIAN_WIDE_BANDWIDTH_THRESHOLD * adj_src_f0:
                    num_freqs = 3
            src_broadband = src_broadband.updated_copy(num_freqs=num_freqs)

        return src_broadband

    @staticmethod
    def _make_post_norm_amps(adj_srcs: list[SourceType]) -> xr.DataArray:
        """Make a ``DataArray`` containing the complex amplitudes to multiply with adjoint field."""

        entries = []
        for src in adj_srcs:
            src_time = src.source_time
            amp_complex = src_time.amplitude * np.exp(1j * src_time.phase)
            entries.append((src_time._freq0, amp_complex))

        entries.sort(key=lambda entry: entry[0])

        freqs = []
        amps_complex = []
        for freq, amp_complex in entries:
            if freq in freqs:
                if not np.allclose(amp_complex, amps_complex[-1], rtol=1e-12, atol=0.0):
                    raise AdjointError(
                        "Adjoint source grouping produced conflicting post-normalization values "
                        f"for frequency {freq}. Each adjoint simulation must have a unique "
                        "post-normalization value per frequency."
                    )
                continue
            freqs.append(freq)
            amps_complex.append(amp_complex)

        return xr.DataArray(np.array(amps_complex), coords={"f": freqs})

    def _get_adjoint_data(self, structure_index: int, data_type: str) -> MonitorDataType:
        """Grab the field or permittivity data for a given structure index."""

        monitor_name = Structure._get_monitor_name(index=structure_index, data_type=data_type)
        return self[monitor_name]


def make_adjoint_simulation(
    simulation: Simulation,
    adjoint_source_info: AdjointSourceInfo,
    adjoint_monitors: list[Monitor],
) -> Simulation:
    """Construct a single adjoint simulation from processed adjoint source info."""

    sim_original = simulation

    # grab boundary conditions with flipped Bloch vectors (for adjoint)
    bc_adj = sim_original.boundary_spec.flipped_bloch_vecs

    # set the ADJ grid spec to use the same grid as sim_original for consistent meshing
    grid_spec_adj = GridSpec.from_grid(sim_original.grid)

    # only include monitors with the same freqs as the adjoint sources
    monitors = [m.updated_copy(freqs=adjoint_source_info.post_norm.f) for m in adjoint_monitors]

    # fields to update the 'fwd' simulation with to make it 'adj'
    sim_adj_update_dict = {
        "sources": adjoint_source_info.sources,
        "boundary_spec": bc_adj,
        "monitors": monitors,
        "post_norm": adjoint_source_info.post_norm,
        "grid_spec": grid_spec_adj,
    }

    if adjoint_source_info.normalize_sim:
        normalize_index_adj = 0
    else:
        normalize_index_adj = None

    sim_adj_update_dict["normalize_index"] = normalize_index_adj

    return sim_original.updated_copy(**sim_adj_update_dict)
