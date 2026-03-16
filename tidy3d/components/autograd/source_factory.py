from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from tidy3d.components.data.data_array import DataArray
from tidy3d.components.monitor import AstigmaticGaussianOverlapMonitor, GaussianOverlapMonitor
from tidy3d.components.source.field import (
    AstigmaticGaussianBeam,
    GaussianBeam,
    ModeSource,
    PlaneWave,
)
from tidy3d.components.source.time import GaussianPulse
from tidy3d.constants import C_0, ETA_0

if TYPE_CHECKING:
    from typing import Literal, Union

    from tidy3d.components.data.monitor_data import DiffractionData
    from tidy3d.components.monitor import DiffractionMonitor, ModeMonitor
    from tidy3d.components.source.utils import GaussianBeamType


def flip_direction(direction: Union[str, DataArray]) -> str:
    """Flip the direction of a string ``('+', '-') -> ('-', '+')``."""
    if isinstance(direction, DataArray):
        direction = str(direction.values)

    if direction not in ("+", "-"):
        raise ValueError(f"Direction must be in {('+', '-')}, got '{direction}'.")
    return "-" if direction == "+" else "+"


def mode_source_from_monitor(
    monitor: ModeMonitor,
    freq: float,
    direction: Union[str, DataArray],
    mode_index: int,
    coefficient: complex,
    fwidth: float,
) -> ModeSource:
    """Build a mode adjoint source from monitor metadata and a complex coefficient."""
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


def gaussian_source_from_monitor(
    monitor: Union[GaussianOverlapMonitor, AstigmaticGaussianOverlapMonitor],
    freq: float,
    direction: Union[str, DataArray],
    coefficient: complex,
    fwidth: float,
) -> GaussianBeamType:
    """Build a Gaussian-like adjoint source from overlap monitor metadata and coefficient."""
    k0 = 2 * np.pi * freq / C_0
    grad_const = k0 / 4 / ETA_0
    src_amp = 1j * grad_const * coefficient

    source_time = GaussianPulse(
        amplitude=abs(src_amp),
        phase=np.angle(src_amp),
        freq0=freq,
        fwidth=fwidth,
    )
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


def diffraction_source_from_data(
    diff_data: DiffractionData,
    freq: float,
    order_x: int,
    order_y: int,
    polarization: Literal["s", "p"],
    coefficient: complex,
    fwidth: float,
) -> PlaneWave | None:
    """Build a diffraction adjoint source from a single diffraction amplitude coordinate."""
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
