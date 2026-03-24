from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Union, cast, get_args

import numpy as np

from tidy3d.components.autograd.source_factory import (
    diffraction_norm,
    diffraction_source_from_simulation,
    mode_source_from_monitor,
    point_current_source_from_simulation,
)
from tidy3d.components.diffraction import (
    DIFFRACTION_POLARIZATIONS,
    diffraction_angle_is_propagating,
)
from tidy3d.components.types.base import DiffractionPolarization, EMField
from tidy3d.log import log

if TYPE_CHECKING:
    from typing import Callable, Optional

    from tidy3d.components.autograd.types import AutogradFieldMap
    from tidy3d.components.data.data_array import DataArray
    from tidy3d.components.data.sim_data import SimulationData
    from tidy3d.components.monitor import DiffractionMonitor, FieldMonitor, ModeMonitor
    from tidy3d.components.simulation import Simulation
    from tidy3d.components.source.utils import SourceType

POINT_FIELD_COMPONENTS = cast(tuple[EMField, ...], get_args(EMField))
# Shared across basis types: float (freq/spatial), int (mode/order indices), str (direction).
CoordTarget = Union[float, int, str]


def _coord_index(coord_values: np.ndarray, target: CoordTarget) -> int:
    if coord_values.size == 0:
        raise ValueError("No coordinate values available to index.")
    if coord_values.dtype.kind in ("f", "c"):
        matches = np.where(np.isclose(coord_values, float(target), rtol=1e-10, atol=0.0))[0]
    else:
        matches = np.where(coord_values == target)[0]
    if matches.size == 0:
        raise ValueError(f"Could not find coordinate value {target!r} in {coord_values}.")
    return int(matches[0])


def _index_for_dims(data_array: DataArray, coord_map: dict[str, CoordTarget]) -> tuple[int, ...]:
    return tuple(
        _coord_index(data_array.coords[dim].values, coord_map[dim]) for dim in data_array.dims
    )


@dataclass(frozen=True)
class AbstractParallelAdjointBasis(ABC):
    monitor_index: int
    monitor_name: str
    freq: float
    data_path: tuple

    @abstractmethod
    def _data_index_from_sim_data(self, sim_data_orig: SimulationData) -> tuple[int, ...]:
        """Return the tuple index for this basis in the associated monitor data."""

    @abstractmethod
    def source_from_simulation(
        self,
        simulation: Simulation,
        coefficient: complex,
        fwidth: float,
    ) -> Optional[SourceType]:
        """Construct the canonical adjoint source for this basis."""

    def _data_index_or_none(self, sim_data_orig: SimulationData) -> Optional[tuple[int, ...]]:
        try:
            return self._data_index_from_sim_data(sim_data_orig)
        except ValueError as exc:
            log.warning(
                "Parallel adjoint basis index lookup failed; "
                "deferring this basis to sequential fallback. "
                f"basis_metadata={asdict(self)}; error={exc}"
            )
            return None

    @staticmethod
    def _vjp_value_for_index(
        data_fields_vjp: AutogradFieldMap,
        data_path: tuple,
        data_index: tuple[int, ...],
        norm: Optional[np.ndarray] = None,
    ) -> complex:
        vjp = data_fields_vjp.get(data_path)
        if vjp is None:
            return 0.0 + 0.0j

        value = np.asarray(vjp)[data_index]
        if norm is not None:
            value *= norm[data_index]
        return complex(value)

    @staticmethod
    def _zero_vjp_entry_for_index(
        data_fields_vjp: AutogradFieldMap,
        data_path: tuple,
        data_index: tuple[int, ...],
    ) -> None:
        vjp = data_fields_vjp.get(data_path)
        if vjp is None:
            return

        vjp_array = np.asarray(vjp)
        vjp_array[data_index] = 0.0
        if vjp_array is not vjp:
            data_fields_vjp[data_path] = vjp_array

    def vjp_value(
        self,
        data_fields_vjp: AutogradFieldMap,
        sim_data_orig: SimulationData,
    ) -> complex:
        data_index = self._data_index_or_none(sim_data_orig)
        if data_index is None:
            return 0.0 + 0.0j
        norm = self._vjp_norm(sim_data_orig)
        return self._vjp_value_for_index(data_fields_vjp, self.data_path, data_index, norm=norm)

    def _vjp_norm(
        self,
        sim_data_orig: SimulationData,
    ) -> Optional[np.ndarray]:
        return None

    def zero_vjp_entry(
        self,
        data_fields_vjp: AutogradFieldMap,
        sim_data_orig: SimulationData,
    ) -> None:
        data_index = self._data_index_or_none(sim_data_orig)
        if data_index is None:
            return
        self._zero_vjp_entry_for_index(data_fields_vjp, self.data_path, data_index)


@dataclass(frozen=True)
class ModeAdjointBasis(AbstractParallelAdjointBasis):
    direction: str
    mode_index: int

    def _data_index_from_sim_data(self, sim_data_orig: SimulationData) -> tuple[int, ...]:
        mode_data = sim_data_orig.data[self.monitor_index]
        coord_map = {
            "f": float(self.freq),
            "direction": str(self.direction),
            "mode_index": int(self.mode_index),
        }
        return _index_for_dims(mode_data.amps, coord_map)

    def source_from_simulation(
        self,
        simulation: Simulation,
        coefficient: complex,
        fwidth: float,
    ) -> Optional[SourceType]:
        monitor = cast("ModeMonitor", simulation.monitors[self.monitor_index])
        return mode_source_from_monitor(
            monitor=monitor,
            freq=self.freq,
            direction=self.direction,
            mode_index=self.mode_index,
            coefficient=coefficient,
            fwidth=fwidth,
        )


@dataclass(frozen=True)
class DiffractionAdjointBasis(AbstractParallelAdjointBasis):
    order_x: int
    order_y: int
    polarization: DiffractionPolarization

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

    def source_from_simulation(
        self,
        simulation: Simulation,
        coefficient: complex,
        fwidth: float,
    ) -> Optional[SourceType]:
        monitor = cast("DiffractionMonitor", simulation.monitors[self.monitor_index])
        return diffraction_source_from_simulation(
            simulation=simulation,
            monitor=monitor,
            freq=self.freq,
            order_x=self.order_x,
            order_y=self.order_y,
            polarization=self.polarization,
            coefficient=coefficient,
            fwidth=fwidth,
        )

    def _vjp_norm(
        self,
        sim_data_orig: SimulationData,
    ) -> Optional[np.ndarray]:
        diff_data = sim_data_orig.data[self.monitor_index]
        return diffraction_norm(diff_data)


@dataclass(frozen=True)
class PointFieldAdjointBasis(AbstractParallelAdjointBasis):
    component: EMField

    def _data_index_from_sim_data(self, sim_data_orig: SimulationData) -> tuple[int, ...]:
        field_data = sim_data_orig.data[self.monitor_index]
        field_component = field_data.field_components[self.component]
        coord_map = {"f": float(self.freq)}
        for dim in field_component.dims:
            if dim == "f":
                continue
            coord_map[dim] = field_component.coords[dim].values[0]
        return _index_for_dims(field_component, coord_map)

    def source_from_simulation(
        self,
        simulation: Simulation,
        coefficient: complex,
        fwidth: float,
    ) -> Optional[SourceType]:
        monitor = cast("FieldMonitor", simulation.monitors[self.monitor_index])
        return point_current_source_from_simulation(
            simulation=simulation,
            monitor=monitor,
            component=self.component,
            freq=self.freq,
            coefficient=coefficient,
            fwidth=fwidth,
        )


ParallelAdjointBasis = Union[ModeAdjointBasis, DiffractionAdjointBasis, PointFieldAdjointBasis]


def _build_mode_bases(
    freqs: Union[list[float], np.ndarray],
    directions: Union[tuple[str, ...], np.ndarray],
    mode_indices: Union[range, np.ndarray],
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
    component_freqs: dict[str, Union[list[float], np.ndarray]],
    monitor_name: str,
    monitor_index: int,
    data_path_prefix: tuple,
) -> list[PointFieldAdjointBasis]:
    bases: list[PointFieldAdjointBasis] = []
    for component, freqs in component_freqs.items():
        if component not in POINT_FIELD_COMPONENTS:
            raise ValueError(
                "Point-field parallel adjoint received unsupported field component "
                f"'{component}'. Expected one of {POINT_FIELD_COMPONENTS}."
            )
        for freq in freqs:
            bases.append(
                PointFieldAdjointBasis(
                    monitor_index=monitor_index,
                    monitor_name=monitor_name,
                    freq=float(freq),
                    component=cast(EMField, component),
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
    pols: Union[tuple[str, ...], np.ndarray],
    theta_for: Callable[[int, int], float],
) -> list[DiffractionAdjointBasis]:
    bases: list[DiffractionAdjointBasis] = []
    for order_x in orders_x:
        for order_y in orders_y:
            angle_theta = float(theta_for(int(order_x), int(order_y)))
            if not diffraction_angle_is_propagating(angle_theta):
                continue
            for pol in pols:
                pol_str = str(pol)
                if pol_str not in DIFFRACTION_POLARIZATIONS:
                    raise ValueError(
                        "Diffraction parallel adjoint received unsupported polarization "
                        f"'{pol_str}'. Expected one of {DIFFRACTION_POLARIZATIONS}."
                    )
                dataset_name = "Ephi" if pol_str == "s" else "Etheta"
                bases.append(
                    DiffractionAdjointBasis(
                        monitor_index=monitor_index,
                        monitor_name=monitor_name,
                        freq=float(freq),
                        order_x=int(order_x),
                        order_y=int(order_y),
                        polarization=cast(DiffractionPolarization, pol_str),
                        data_path=("data", monitor_index, dataset_name),
                    )
                )
    return bases
