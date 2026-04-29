from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from tidy3d.components.data.data_array import DataArray, ScalarFieldDataArray
from tidy3d.components.data.dataset import FieldDataset
from tidy3d.components.diffraction import (
    DIFFRACTION_POLARIZATIONS,
    bloch_vec_at_freq,
    compute_angles,
    diffraction_amplitude_norm,
    diffraction_angle_is_propagating,
    diffraction_monitor_medium,
    domain_size_2d,
    reciprocal_coords,
    sim_bloch_vecs,
)
from tidy3d.components.grid.grid import Coords
from tidy3d.components.source.current import CustomCurrentSource
from tidy3d.components.source.field import (
    AstigmaticGaussianBeam,
    GaussianBeam,
    ModeSource,
    PlaneWave,
)
from tidy3d.components.source.time import GaussianPulse
from tidy3d.constants import C_0, EPSILON_0, ETA_0

if TYPE_CHECKING:
    from tidy3d import Source
    from tidy3d.components.data.monitor_data import DiffractionData
    from tidy3d.components.data.sim_data import AdjointSourceInfo
    from tidy3d.components.monitor import (
        AstigmaticGaussianOverlapMonitor,
        DiffractionMonitor,
        FieldMonitor,
        GaussianOverlapMonitor,
        ModeMonitor,
    )
    from tidy3d.components.simulation import Simulation
    from tidy3d.components.source.utils import GaussianBeamType
    from tidy3d.components.types.base import DiffractionPolarization


def flip_direction(direction: str | DataArray) -> str:
    """Flip the direction of a string ``('+', '-') -> ('-', '+')``."""
    if isinstance(direction, DataArray):
        direction = str(direction.values)

    if direction not in ("+", "-"):
        raise ValueError(f"Direction must be in {('+', '-')}, got '{direction}'.")
    return "-" if direction == "+" else "+"


def _adjust_source_fwidth(source: Source) -> Source:
    from tidy3d.components.data.sim_data import SimulationData

    return SimulationData._adjoint_src_width_single([source])[0]


def adjoint_source_info_single(source: Source, *, adjust_fwidth: bool = True) -> AdjointSourceInfo:
    from tidy3d.components.data.sim_data import AdjointSourceInfo

    if adjust_fwidth:
        source = _adjust_source_fwidth(source)
    freq0 = source.source_time._freq0
    post_norm = DataArray(data=np.array([1 + 0j]), coords={"f": [freq0]})
    return AdjointSourceInfo(sources=(source,), post_norm=post_norm, normalize_sim=True)


def mode_source_from_monitor(
    monitor: ModeMonitor,
    freq: float,
    direction: str | DataArray,
    mode_index: int,
    coefficient: complex,
    fwidth: float,
) -> ModeSource:
    k0 = 2 * np.pi * freq / C_0
    grad_const = k0 / 4 / ETA_0
    src_amp = 1j * grad_const * coefficient
    return ModeSource(
        source_time=_adjoint_source_time(freq=freq, src_amp=src_amp, fwidth=fwidth),
        mode_spec=monitor.mode_spec,
        size=monitor.size,
        center=monitor.center,
        direction=flip_direction(direction),
        mode_index=mode_index,
    )


def gaussian_source_from_monitor(
    monitor: GaussianOverlapMonitor | AstigmaticGaussianOverlapMonitor,
    freq: float,
    direction: str | DataArray,
    coefficient: complex,
    fwidth: float,
) -> GaussianBeamType:
    """Build a Gaussian-like adjoint source from overlap monitor metadata and coefficient."""
    from tidy3d.components.monitor import AstigmaticGaussianOverlapMonitor, GaussianOverlapMonitor

    k0 = 2 * np.pi * freq / C_0
    grad_const = k0 / 4 / ETA_0
    src_amp = 1j * grad_const * coefficient

    source_time = _adjoint_source_time(freq=freq, src_amp=src_amp, fwidth=fwidth)
    direction_flipped = flip_direction(direction)

    if isinstance(monitor, GaussianOverlapMonitor):
        return GaussianBeam(
            center=monitor.center,
            size=monitor.size,
            source_time=source_time,
            direction=direction_flipped,
            angle_theta=monitor.angle_theta,
            angle_phi=monitor.angle_phi,
            pol_angle=monitor.pol_angle,
            waist_radius=monitor.waist_radius,
            waist_distance=monitor.waist_distance,
            num_freqs=1,
        )

    if isinstance(monitor, AstigmaticGaussianOverlapMonitor):
        return AstigmaticGaussianBeam(
            center=monitor.center,
            size=monitor.size,
            source_time=source_time,
            direction=direction_flipped,
            angle_theta=monitor.angle_theta,
            angle_phi=monitor.angle_phi,
            pol_angle=monitor.pol_angle,
            waist_sizes=monitor.waist_sizes,
            waist_distances=monitor.waist_distances,
            num_freqs=1,
        )

    raise TypeError(
        "Expected GaussianOverlapMonitor or AstigmaticGaussianOverlapMonitor, "
        f"got '{type(monitor).__name__}'."
    )


def current_component_data_array(
    *,
    component: str,
    base_values: np.ndarray,
    spatial_coords: dict[str, np.ndarray],
    source_center: tuple[float, float, float],
    freq: float,
    symmetry_factor: float = 1.0,
) -> ScalarFieldDataArray | None:
    """Convert one field component to a scaled adjoint current data array."""
    values = 2 * -1j * np.asarray(base_values)
    if "H" in component:
        values *= -1

    coords = {key: np.asarray(spatial_coords[key]) for key in "xyz"}
    grid_coords = Coords(**coords)
    size_element = grid_coords.cell_size_meshgrid
    for dim, key in enumerate("xyz"):
        coords[key] = coords[key] - source_center[dim]

    coords["f"] = np.array([freq])
    values = np.expand_dims(values, axis=-1)
    size_element = np.reshape(size_element, values.shape)

    omega0 = 2 * np.pi * freq
    scaling_factor = 0.5 * omega0 * EPSILON_0 / size_element
    values *= scaling_factor * symmetry_factor
    values = np.nan_to_num(values, nan=0.0)
    if np.all(values == 0):
        return None
    return ScalarFieldDataArray(values, coords=coords)


def point_current_source_from_simulation(
    simulation: Simulation,
    monitor: FieldMonitor,
    component: str,
    freq: float,
    coefficient: complex,
    fwidth: float,
) -> CustomCurrentSource | None:
    grid = simulation.discretize_monitor(monitor)
    coords = {}
    spatial_coords = grid.boundaries
    spatial_coords_dict = spatial_coords.dict()
    for axis, dim in enumerate("xyz"):
        if monitor.size[axis] == 0:
            coords[dim] = np.array([monitor.center[axis]])
        else:
            coords[dim] = np.array(spatial_coords_dict[dim][:-1])

    base_values = coefficient * np.ones(
        (len(coords["x"]), len(coords["y"]), len(coords["z"])),
        dtype=complex,
    )
    symmetry_factor = 1.0
    sym_center = simulation.center
    for dim, sym in enumerate(simulation.symmetry):
        if sym == 0:
            continue
        if np.isclose(monitor.center[dim], sym_center[dim]):
            continue
        symmetry_factor *= 2.0

    source_data = current_component_data_array(
        component=component,
        base_values=base_values,
        spatial_coords=coords,
        source_center=monitor.geometry.center,
        freq=freq,
        symmetry_factor=symmetry_factor,
    )
    if source_data is None:
        return None

    dataset = FieldDataset(**{component: source_data})
    return CustomCurrentSource(
        center=monitor.geometry.center,
        size=monitor.geometry.size,
        source_time=GaussianPulse(freq0=freq, fwidth=fwidth),
        current_dataset=dataset,
        interpolate=True,
    )


def diffraction_source_from_simulation(
    simulation: Simulation,
    monitor: DiffractionMonitor,
    freq: float,
    order_x: int,
    order_y: int,
    polarization: DiffractionPolarization,
    coefficient: complex,
    fwidth: float,
) -> PlaneWave:
    medium = diffraction_monitor_medium(simulation, monitor)
    size_x, size_y = domain_size_2d(simulation, monitor)
    bloch_vec_x, bloch_vec_y = sim_bloch_vecs(simulation, monitor)
    freqs = np.asarray(monitor.freqs, dtype=float)
    freq_index = np.argmin(np.abs(freqs - freq))
    if not np.isclose(freqs[freq_index], freq, rtol=1e-10, atol=0.0):
        raise ValueError("Adjoint source frequency not found in diffraction monitor frequencies.")
    bloch_vec_x = bloch_vec_at_freq(bloch_vec_x, freq_index)
    bloch_vec_y = bloch_vec_at_freq(bloch_vec_y, freq_index)

    ux = reciprocal_coords(
        orders=np.array([order_x]),
        size=size_x,
        bloch_vec=bloch_vec_x,
        f=freq,
        medium=medium,
    )
    uy = reciprocal_coords(
        orders=np.array([order_y]),
        size=size_y,
        bloch_vec=bloch_vec_y,
        f=freq,
        medium=medium,
    )
    theta_vals, phi_vals = compute_angles((ux, uy))
    angle_theta = float(theta_vals[0, 0, 0])
    angle_phi = float(phi_vals[0, 0, 0])
    if not diffraction_angle_is_propagating(angle_theta):
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
    polarization: DiffractionPolarization,
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
    if pol_str not in DIFFRACTION_POLARIZATIONS:
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
        source_time=_adjoint_source_time(freq=freq, src_amp=src_amp, fwidth=fwidth),
        direction=flip_direction(monitor.normal_dir),
        angle_theta=src_angle_theta,
        angle_phi=angle_phi,
        pol_angle=pol_angle,
    )


def _adjoint_source_time(*, freq: float, src_amp: complex, fwidth: float) -> GaussianPulse:
    """Build a GaussianPulse from an adjoint source amplitude."""
    return GaussianPulse(
        amplitude=abs(src_amp),
        phase=np.angle(src_amp),
        freq0=freq,
        fwidth=fwidth,
    )


def diffraction_norm(diffraction_data: DiffractionData) -> np.ndarray:
    theta_data, _ = diffraction_data.angles
    return diffraction_amplitude_norm(theta_data.values, diffraction_data.eta)
