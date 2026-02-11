from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from tidy3d.components.autograd.parallel_adjoint_bases import COS_THETA_THRESH
from tidy3d.components.data.data_array import DataArray, ScalarFieldDataArray
from tidy3d.components.data.dataset import FieldDataset
from tidy3d.components.data.monitor_data import DiffractionData
from tidy3d.components.data.sim_data import AdjointSourceInfo, SimulationData
from tidy3d.components.grid.grid import Coords
from tidy3d.components.source.current import CustomCurrentSource
from tidy3d.components.source.field import ModeSource, PlaneWave
from tidy3d.components.source.time import GaussianPulse
from tidy3d.constants import C_0, EPSILON_0, ETA_0

if TYPE_CHECKING:
    from typing import Literal

    from tidy3d import Source
    from tidy3d.components.monitor import DiffractionMonitor, FieldMonitor, ModeMonitor
    from tidy3d.components.simulation import Simulation


def flip_direction(direction: object) -> str:
    if hasattr(direction, "values"):
        direction = str(direction.values)
    if direction not in ("+", "-"):
        raise ValueError(f"Direction must be in {('+', '-')}, got '{direction}'.")
    return "-" if direction == "+" else "+"


def adjoint_fwidth_from_simulation(simulation: Simulation) -> float:
    normalize_index = simulation.normalize_index or 0
    return simulation.sources[normalize_index].source_time.fwidth


def _adjust_source_fwidth(source: Source) -> object:
    return SimulationData._adjoint_src_width_single([source])[0]


def adjoint_source_info_single(source: Source) -> AdjointSourceInfo:
    source = _adjust_source_fwidth(source)
    freq0 = source.source_time._freq0
    post_norm = DataArray(data=np.array([1 + 0j]), coords={"f": [freq0]})
    return AdjointSourceInfo(sources=(source,), post_norm=post_norm, normalize_sim=True)


def mode_source_from_monitor(
    monitor: ModeMonitor,
    freq: float,
    direction: str,
    mode_index: int,
    coefficient: complex,
    fwidth: float,
) -> ModeSource:
    k0 = 2 * np.pi * freq / C_0
    grad_const = k0 / 4 / ETA_0
    src_amp = 1j * grad_const * coefficient
    return ModeSource(
        source_time=GaussianPulse(
            amplitude=abs(src_amp),
            phase=np.angle(src_amp),
            freq0=freq,
            fwidth=fwidth,
        ),
        mode_spec=monitor.mode_spec,
        size=monitor.size,
        center=monitor.center,
        direction=flip_direction(direction),
        mode_index=mode_index,
    )


def point_current_source_from_simulation(
    simulation: Simulation,
    monitor: FieldMonitor,
    component: str,
    freq: float,
    coefficient: complex,
    fwidth: float,
) -> CustomCurrentSource | None:
    if not monitor.colocate:
        raise ValueError("Point-field adjoint sources require colocated field monitors.")

    grid = simulation.discretize_monitor(monitor)
    coords = {}
    spatial_coords = grid.boundaries
    spatial_coords_dict = spatial_coords.dict()
    for axis, dim in enumerate("xyz"):
        if monitor.size[axis] == 0:
            coords[dim] = np.array([monitor.center[axis]])
        else:
            coords[dim] = np.array(spatial_coords_dict[dim][:-1])
    values = (
        2
        * -1j
        * coefficient
        * np.ones(
            (len(coords["x"]), len(coords["y"]), len(coords["z"])),
            dtype=complex,
        )
    )

    if "H" in component:
        values *= -1

    grid_coords = Coords(**{key: coords[key] for key in "xyz"})
    size_element = grid_coords.cell_size_meshgrid
    for dim, key in enumerate("xyz"):
        coords[key] = np.array(coords[key]) - monitor.geometry.center[dim]

    coords["f"] = np.array([freq])
    values = np.expand_dims(values, axis=-1)
    size_element = np.reshape(size_element, values.shape)

    omega0 = 2 * np.pi * freq
    scaling_factor = 0.5 * omega0 * EPSILON_0 / size_element
    symmetry_factor = 1.0
    sym_center = simulation.center
    for dim, sym in enumerate(simulation.symmetry):
        if sym == 0:
            continue
        if np.isclose(monitor.center[dim], sym_center[dim]):
            continue
        symmetry_factor *= 2.0

    values *= scaling_factor * symmetry_factor
    values = np.nan_to_num(values, nan=0.0)

    if np.all(values == 0):
        return None

    dataset = FieldDataset(**{component: ScalarFieldDataArray(values, coords=coords)})
    return CustomCurrentSource(
        center=monitor.geometry.center,
        size=monitor.geometry.size,
        source_time=GaussianPulse(freq0=freq, fwidth=fwidth),
        current_dataset=dataset,
        interpolate=True,
    )


def diffraction_monitor_medium(simulation: Simulation, monitor: DiffractionMonitor) -> object:
    structures = [simulation.scene.background_structure, *list(simulation.structures or ())]
    mediums = simulation.scene.intersecting_media(monitor, structures)
    if len(mediums) != 1:
        raise ValueError("Diffraction monitor plane must be homogeneous to build adjoint sources.")
    return list(mediums)[0]


def bloch_vec_for_axis(simulation: Simulation, axis_name: str) -> float:
    boundary = simulation.boundary_spec[axis_name]
    plus = boundary.plus
    if hasattr(plus, "bloch_vec"):
        return float(plus.bloch_vec)
    return 0.0


def diffraction_order_range(
    size: float, bloch_vec: float, freq: float, medium: object
) -> np.ndarray:
    if size == 0:
        return np.array([0], dtype=int)
    eps = medium.eps_model(freq)
    index = np.real(np.sqrt(eps))
    limit = abs(index) * freq * size / C_0
    order_min = int(np.ceil(-limit - bloch_vec))
    order_max = int(np.floor(limit - bloch_vec))
    if order_max < order_min:
        return np.array([], dtype=int)
    return np.arange(order_min, order_max + 1, dtype=int)


def diffraction_source_from_simulation(
    simulation: Simulation,
    monitor: DiffractionMonitor,
    freq: float,
    order_x: int,
    order_y: int,
    polarization: Literal["s", "p"],
    coefficient: complex,
    fwidth: float,
) -> PlaneWave:
    medium = diffraction_monitor_medium(simulation, monitor)
    axis_names = ("x", "y", "z")
    normal_axis = monitor.normal_axis
    transverse_axes = [axis_names[i] for i in range(3) if i != normal_axis]
    axis_x, axis_y = transverse_axes

    size_x = simulation.size[axis_names.index(axis_x)]
    size_y = simulation.size[axis_names.index(axis_y)]
    bloch_vec_x = bloch_vec_for_axis(simulation, axis_x)
    bloch_vec_y = bloch_vec_for_axis(simulation, axis_y)

    ux = DiffractionData.reciprocal_coords(
        orders=np.array([order_x]),
        size=size_x,
        bloch_vec=bloch_vec_x,
        f=freq,
        medium=medium,
    )
    uy = DiffractionData.reciprocal_coords(
        orders=np.array([order_y]),
        size=size_y,
        bloch_vec=bloch_vec_y,
        f=freq,
        medium=medium,
    )
    theta_vals, phi_vals = DiffractionData.compute_angles((ux, uy))
    angle_theta = float(theta_vals[0, 0, 0])
    angle_phi = float(phi_vals[0, 0, 0])
    if np.isnan(angle_theta) or np.cos(angle_theta) <= COS_THETA_THRESH:
        raise ValueError("Adjoint source not available for evanescent diffraction order.")

    pol_angle = 0.0 if polarization == "p" else np.pi / 2
    bck_eps = medium.eps_model(freq)
    return _diffraction_plane_wave(
        monitor=monitor,
        freq=freq,
        angle_theta=angle_theta,
        angle_phi=angle_phi,
        pol_angle=pol_angle,
        coefficient=coefficient,
        fwidth=fwidth,
        bck_eps=bck_eps,
    )


def diffraction_source_from_data(
    diff_data: DiffractionData,
    freq: float,
    order_x: int,
    order_y: int,
    polarization: Literal["s", "p"],
    coefficient: complex,
    fwidth: float,
) -> PlaneWave | None:
    monitor = diff_data.monitor
    theta_data, phi_data = diff_data.angles
    angle_sel_kwargs = {"orders_x": int(order_x), "orders_y": int(order_y), "f": float(freq)}
    angle_theta = float(theta_data.sel(**angle_sel_kwargs))
    angle_phi = float(phi_data.sel(**angle_sel_kwargs))

    if np.isnan(angle_theta):
        return None

    pol_str = str(polarization)
    if pol_str not in ("p", "s"):
        raise ValueError(f"Something went wrong, given pol='{pol_str}' in adjoint source.")

    pol_angle = 0.0 if pol_str == "p" else np.pi / 2
    bck_eps = diff_data.medium.eps_model(freq)
    return _diffraction_plane_wave(
        monitor=monitor,
        freq=freq,
        angle_theta=angle_theta,
        angle_phi=angle_phi,
        pol_angle=pol_angle,
        coefficient=coefficient,
        fwidth=fwidth,
        bck_eps=bck_eps,
    )


def _diffraction_plane_wave(
    monitor: DiffractionMonitor,
    freq: float,
    angle_theta: float,
    angle_phi: float,
    pol_angle: float,
    coefficient: complex,
    fwidth: float,
    bck_eps: complex,
) -> PlaneWave:
    k0 = 2 * np.pi * freq / C_0
    grad_const = 0.5 * k0 / np.sqrt(bck_eps) * np.cos(angle_theta)
    normal_factor = 1.0 if monitor.normal_dir == "+" else -1.0
    src_amp = 1j * grad_const * coefficient * normal_factor
    src_angle_theta = normal_factor * angle_theta

    return PlaneWave(
        size=monitor.size,
        center=monitor.center,
        source_time=GaussianPulse(
            amplitude=abs(src_amp),
            phase=np.angle(src_amp),
            freq0=freq,
            fwidth=fwidth,
        ),
        direction=flip_direction(monitor.normal_dir),
        angle_theta=src_angle_theta,
        angle_phi=angle_phi,
        pol_angle=pol_angle,
    )


def diffraction_norm(diffraction_data: DiffractionData) -> np.ndarray:
    theta_data, _ = diffraction_data.angles
    cos_theta = np.cos(np.nan_to_num(theta_data))
    cos_theta[cos_theta <= COS_THETA_THRESH] = np.inf
    return 1.0 / np.sqrt(2.0 * np.asarray(diffraction_data.eta)) / np.sqrt(cos_theta)
