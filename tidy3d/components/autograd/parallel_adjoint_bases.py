from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from typing import Literal

    from tidy3d.components.autograd.types import AutogradFieldMap
    from tidy3d.components.data.data_array import DataArray
    from tidy3d.components.data.sim_data import SimulationData

# Threshold for cos(theta) to avoid unphysically large amplitudes near grazing angles
COS_THETA_THRESH = 1e-5


def _coord_index(coord_values: np.ndarray, target: object) -> int:
    values = np.asarray(coord_values)
    if values.size == 0:
        raise ValueError("No coordinate values available to index.")
    if values.dtype.kind in ("f", "c"):
        matches = np.where(np.isclose(values, float(target), rtol=1e-10, atol=0.0))[0]
    else:
        matches = np.where(values == target)[0]
    if matches.size == 0:
        raise ValueError(f"Could not find coordinate value {target!r} in {values}.")
    return int(matches[0])


def _index_for_dims(data_array: DataArray, coord_map: dict[str, object]) -> tuple[int, ...]:
    return tuple(
        _coord_index(data_array.coords[dim].values, coord_map[dim]) for dim in data_array.dims
    )


@dataclass(frozen=True)
class ModeAdjointBasis:
    monitor_index: int
    monitor_name: str
    freq: float
    direction: str
    mode_index: int
    data_path: tuple

    def _data_index_from_sim_data(self, sim_data_orig: SimulationData) -> tuple[int, ...]:
        mode_data = sim_data_orig.data[self.monitor_index]
        coord_map = {
            "f": float(self.freq),
            "direction": str(self.direction),
            "mode_index": int(self.mode_index),
        }
        return _index_for_dims(mode_data.amps, coord_map)

    def vjp_value(
        self, data_fields_vjp: AutogradFieldMap, sim_data_orig: SimulationData
    ) -> complex:
        vjp = data_fields_vjp.get(self.data_path)
        if vjp is None:
            return 0.0 + 0.0j
        data_index = self._data_index_from_sim_data(sim_data_orig)
        vjp_array = np.asarray(vjp)
        value = complex(vjp_array[data_index])
        return value

    def zero_vjp_entry(
        self, data_fields_vjp: AutogradFieldMap, sim_data_orig: SimulationData
    ) -> None:
        vjp = data_fields_vjp.get(self.data_path)
        if vjp is None:
            return
        vjp_array = np.asarray(vjp)
        vjp_array[self._data_index_from_sim_data(sim_data_orig)] = 0.0
        if vjp_array is not vjp:
            data_fields_vjp[self.data_path] = vjp_array


@dataclass(frozen=True)
class DiffractionAdjointBasis:
    monitor_index: int
    monitor_name: str
    freq: float
    order_x: int
    order_y: int
    polarization: Literal["s", "p"]
    data_path: tuple

    def _data_index_from_sim_data(self, sim_data_orig: SimulationData) -> tuple[int, ...]:
        diff_data = sim_data_orig.data[self.monitor_index]
        dataset_name = self.data_path[-1]
        field_data = getattr(diff_data, dataset_name)
        coord_map = {
            "orders_x": int(self.order_x),
            "orders_y": int(self.order_y),
            "f": float(self.freq),
        }
        return _index_for_dims(field_data, coord_map)

    def vjp_value(
        self, data_fields_vjp: AutogradFieldMap, sim_data_orig: SimulationData, norm: np.ndarray
    ) -> complex:
        vjp = data_fields_vjp.get(self.data_path)
        if vjp is None:
            return 0.0 + 0.0j
        try:
            data_index = self._data_index_from_sim_data(sim_data_orig)
        except ValueError:
            return 0.0 + 0.0j
        return complex(np.asarray(vjp)[data_index] * norm[data_index])

    def zero_vjp_entry(
        self, data_fields_vjp: AutogradFieldMap, sim_data_orig: SimulationData
    ) -> None:
        vjp = data_fields_vjp.get(self.data_path)
        if vjp is None:
            return
        try:
            data_index = self._data_index_from_sim_data(sim_data_orig)
        except ValueError:
            return
        vjp_array = np.asarray(vjp)
        vjp_array[data_index] = 0.0
        if vjp_array is not vjp:
            data_fields_vjp[self.data_path] = vjp_array


@dataclass(frozen=True)
class PointFieldAdjointBasis:
    monitor_index: int
    monitor_name: str
    freq: float
    component: Literal["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]
    data_path: tuple

    def _data_index_from_sim_data(self, sim_data_orig: SimulationData) -> tuple[int, ...]:
        field_data = sim_data_orig.data[self.monitor_index]
        field_component = field_data.field_components[self.component]
        coord_map = {"f": float(self.freq)}
        for dim in field_component.dims:
            if dim == "f":
                continue
            coord_map[dim] = field_component.coords[dim].values[0]
        return _index_for_dims(field_component, coord_map)

    def vjp_value(
        self, data_fields_vjp: AutogradFieldMap, sim_data_orig: SimulationData
    ) -> complex:
        vjp = data_fields_vjp.get(self.data_path)
        if vjp is None:
            return 0.0 + 0.0j
        data_index = self._data_index_from_sim_data(sim_data_orig)
        return complex(np.asarray(vjp)[data_index])

    def zero_vjp_entry(
        self, data_fields_vjp: AutogradFieldMap, sim_data_orig: SimulationData
    ) -> None:
        vjp = data_fields_vjp.get(self.data_path)
        if vjp is None:
            return
        vjp_array = np.asarray(vjp)
        vjp_array[self._data_index_from_sim_data(sim_data_orig)] = 0.0
        if vjp_array is not vjp:
            data_fields_vjp[self.data_path] = vjp_array


ParallelAdjointBasis = ModeAdjointBasis | DiffractionAdjointBasis | PointFieldAdjointBasis


def _build_mode_bases(
    freqs: list[float] | np.ndarray,
    directions: tuple[str, str] | np.ndarray,
    mode_indices: range | np.ndarray,
    monitor_name: str,
    monitor_index: int,
    data_path: tuple,
) -> list[ModeAdjointBasis]:
    bases: list[ModeAdjointBasis] = []
    for freq in freqs:
        for direction in directions:
            for mode_index in mode_indices:
                bases.append(
                    ModeAdjointBasis(
                        monitor_index=monitor_index,
                        monitor_name=monitor_name,
                        freq=float(freq),
                        direction=str(direction),
                        mode_index=int(mode_index),
                        data_path=data_path,
                    )
                )
    return bases


def _build_point_field_bases(
    component_freqs: list[tuple[str, list[float]]] | list[tuple[str, np.ndarray]],
    monitor_name: str,
    monitor_index: int,
    data_path_prefix: tuple,
) -> list[PointFieldAdjointBasis]:
    bases: list[PointFieldAdjointBasis] = []
    for component, freqs in component_freqs:
        if component not in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
            continue
        for freq in freqs:
            bases.append(
                PointFieldAdjointBasis(
                    monitor_index=monitor_index,
                    monitor_name=monitor_name,
                    freq=float(freq),
                    component=str(component),
                    data_path=(*data_path_prefix, component),
                )
            )
    return bases


def _build_diffraction_bases_for_freq(
    *,
    monitor_name: str,
    monitor_index: int,
    freq: float,
    orders_x: np.ndarray,
    orders_y: np.ndarray,
    pols: tuple[str, str] | np.ndarray,
    theta_for: object,
) -> list[DiffractionAdjointBasis]:
    bases: list[DiffractionAdjointBasis] = []
    for order_x in orders_x:
        for order_y in orders_y:
            angle_theta = float(theta_for(int(order_x), int(order_y)))
            if np.isnan(angle_theta) or np.cos(angle_theta) <= COS_THETA_THRESH:
                continue
            for pol in pols:
                pol_str = str(pol)
                if pol_str not in ("s", "p"):
                    continue
                dataset_name = "Ephi" if pol_str == "s" else "Etheta"
                bases.append(
                    DiffractionAdjointBasis(
                        monitor_index=monitor_index,
                        monitor_name=monitor_name,
                        freq=float(freq),
                        order_x=int(order_x),
                        order_y=int(order_y),
                        polarization=pol_str,
                        data_path=("data", monitor_index, dataset_name),
                    )
                )
    return bases
