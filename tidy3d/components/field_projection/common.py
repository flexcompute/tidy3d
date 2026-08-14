"""Shared helpers for field projection implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, TypeVar

import autograd.numpy as anp
import numpy as np
from autograd.extend import defvjp, primitive
from rich.progress import track

from tidy3d.components.data.monitor_data import AbstractFieldProjectionData
from tidy3d.components.geometry.base import Geometry
from tidy3d.components.types import ArrayComplex4D, Axis
from tidy3d.constants import ETA_0
from tidy3d.exceptions import SetupError, format_chained_exception_message
from tidy3d.log import get_logging_console

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    import xarray as xr

    from tidy3d.components.data.data_array import (
        FieldProjectionAngleDataArray,
        FieldProjectionCartesianDataArray,
        FieldProjectionKSpaceDataArray,
    )
    from tidy3d.components.data.monitor_data import (
        FieldProjectionAngleData,
        FieldProjectionCartesianData,
        FieldProjectionKSpaceData,
    )
    from tidy3d.components.medium import MediumType
    from tidy3d.components.monitor import AbstractFieldProjectionMonitor, FieldProjectionSurface

# Default number of points per wavelength in the background medium to use for resampling fields.
PTS_PER_WVL = 10
APPROX_PROJECTION_BATCH_SIZE = 512
EXACT_PROJECTION_BATCH_SIZE = 32
PROJECTION_FREQ_CHUNK_SIZE = 8
FIELD_COMPONENT_NAMES = ("Er", "Etheta", "Ephi", "Hr", "Htheta", "Hphi")

# Numpy float array and related array types

ArrayLikeN2F = float | tuple[float, ...] | ArrayComplex4D
_TrackedItem = TypeVar("_TrackedItem")
_TangentialCurrentComponentName = Literal["electric_u", "electric_v", "magnetic_u", "magnetic_v"]


def _track_if_verbose(
    iterable: Iterable[_TrackedItem],
    *,
    verbose: bool,
    description: str,
    total: int | None = None,
) -> Iterable[_TrackedItem]:
    """Wrap an iterable in a progress tracker only when requested."""

    if not verbose:
        return iterable
    return track(
        iterable,
        description=description,
        total=total,
        console=get_logging_console(),
    )


@dataclass(frozen=True)
class _FarFieldIntegralSpec:
    """Static integration geometry for a separable far-field integral.

    Pure geometry: the differential areas are folded into the current values themselves
    (element currents), so the kernels only phase and sum.
    """

    idx_u: Axis
    idx_v: Axis
    is_2d: bool
    idx_integration_1d: Axis | None

    @property
    def line_axis(self) -> Axis:
        """Return the integration axis for 2D line-source projection."""

        if self.idx_integration_1d in (0, 1, 2):
            return self.idx_integration_1d
        raise ValueError("Expected 'idx_integration_1d' for 2D far-field projection.")

    @property
    def remaining_axis(self) -> Axis:
        """Return the non-integrated axis for 3D surface projection."""

        _, planar_axes = Geometry.pop_axis((0, 1, 2), axis=self.idx_u)
        if self.idx_v == planar_axes[0]:
            return planar_axes[1]
        if self.idx_v == planar_axes[1]:
            return planar_axes[0]
        raise ValueError(
            f"Expected integrated axes to be distinct, got {self.idx_u}, {self.idx_v}."
        )

    @property
    def integrated_axes(self) -> tuple[Axis, ...]:
        """Return the axes integrated by the projection kernel."""

        if self.is_2d:
            return (self.line_axis,)
        return Geometry.pop_axis((0, 1, 2), axis=self.remaining_axis)[1]


@dataclass(frozen=True)
class _FarFieldProjectionKernelSpec:
    """Static metadata for one prepared far-field projection kernel.

    The four tangential current components share one integration geometry and each index a
    shared source grid: ``pts_sets`` holds the distinct grids and ``component_pts_inds`` maps
    each component, in ``(electric_u, electric_v, magnetic_u, magnetic_v)`` order (u/v are the
    surface's in-plane axes; see :class:`_TangentialSurfaceCurrents`), to its grid, so the
    expensive per-grid phase factors are evaluated once per distinct grid:

    - colocated projection: one shared grid, ``component_pts_inds = (0, 0, 0, 0)``;
    - non-colocated projection: two staggered Yee grids — ``electric_u`` and ``magnetic_v``
      (built from ``H_v`` and ``E_u``) share the ``(dual_u, primal_v)`` grid while
      ``electric_v`` and ``magnetic_u`` (built from ``H_u`` and ``E_v``) share the
      ``(primal_u, dual_v)`` grid — giving ``component_pts_inds = (0, 1, 1, 0)``.
    """

    integral: _FarFieldIntegralSpec
    pts_sets: tuple[tuple[np.ndarray, np.ndarray, np.ndarray], ...]
    component_pts_inds: tuple[int, int, int, int]
    propagation_factor: complex
    eta: complex

    def component_pts(self, component_index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Source grid of one tangential component."""

        return self.pts_sets[self.component_pts_inds[component_index]]


@dataclass(frozen=True)
class _TangentialSurfaceCurrents:
    """Tangential electric and magnetic current components on a projection surface.

    ``u`` and ``v`` are the surface's two in-plane axes in ``pop_axis`` order (e.g. ``u=x``,
    ``v=y`` for a z-normal surface). ``electric_*`` are the equivalent electric current
    ``J = n x H`` components and ``magnetic_*`` the magnetic current ``M = -n x E`` components.
    The cross product swaps tangential components, so each current sits at the Yee position of
    the field it is built from: ``electric_u`` (from ``H_v``) and ``magnetic_v`` (from ``E_u``)
    share the ``(dual_u, primal_v)`` in-plane grid, while ``electric_v`` (from ``H_u``) and
    ``magnetic_u`` (from ``E_v``) share the ``(primal_u, dual_v)`` grid.
    """

    electric_u: np.ndarray
    electric_v: np.ndarray
    magnetic_u: np.ndarray
    magnetic_v: np.ndarray

    def as_tuple(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return current components in the canonical (kernel argument) order."""

        return (self.electric_u, self.electric_v, self.magnetic_u, self.magnetic_v)

    @staticmethod
    def component_index(component_name: _TangentialCurrentComponentName) -> int:
        """Position of one component in the :meth:`as_tuple` (kernel argument / VJP argnum) order."""

        return {"electric_u": 0, "electric_v": 1, "magnetic_u": 2, "magnetic_v": 3}[component_name]


@dataclass(frozen=True)
class _PairedAngleTrig:
    """Precomputed trigonometric values for paired spherical observation angles."""

    sin_theta: np.ndarray
    cos_theta: np.ndarray
    sin_phi: np.ndarray
    cos_phi: np.ndarray


if TYPE_CHECKING:
    _FarFieldsFromCurrentsPairs3DVJPMaker = Callable[
        [
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            _FarFieldProjectionKernelSpec,
            Axis,
            _PairedAngleTrig,
        ],
        Callable[[np.ndarray], np.ndarray],
    ]


@dataclass(frozen=True)
class _PreparedFarFieldProjection:
    """Prepared arrays reused across far-field evaluations at one frequency.

    ``pts_sets``/``component_pts_inds`` follow the :class:`_FarFieldProjectionKernelSpec`
    contract (distinct source grids + per-component grid index).
    """

    field_components: _TangentialSurfaceCurrents
    integral: _FarFieldIntegralSpec
    pts_sets: tuple[tuple[np.ndarray, np.ndarray, np.ndarray], ...]
    component_pts_inds: tuple[int, int, int, int]
    propagation_factor: complex
    eta: complex

    @property
    def kernel_spec(self) -> _FarFieldProjectionKernelSpec:
        """Return projection metadata without differentiable field component arrays."""

        return _FarFieldProjectionKernelSpec(
            integral=self.integral,
            pts_sets=self.pts_sets,
            component_pts_inds=self.component_pts_inds,
            propagation_factor=self.propagation_factor,
            eta=self.eta,
        )


def _prepare_far_field_projection(
    *,
    frequency: float,
    surface: FieldProjectionSurface,
    currents: xr.Dataset,
    medium: MediumType,
    is_2d_simulation: bool,
    simulation_size: tuple[float, float, float],
) -> _PreparedFarFieldProjection:
    """Prepare raw arrays reused across repeated approximate far-field evaluations.

    ``currents`` is either an :class:`xarray.Dataset` of colocated current components (all
    sharing one source grid) or, for non-colocated projection, a ``dict`` mapping each tangential
    component name to its own :class:`xarray.DataArray` carrying that component's native
    Yee-staggered positions and box-rule area weighting.
    """

    non_colocated = isinstance(currents, dict)

    def select_frequency(
        component: xr.Dataset | xr.DataArray,
    ) -> xr.Dataset | xr.DataArray:
        try:
            return component.sel(f=frequency)
        except Exception as e:
            raise SetupError(
                format_chained_exception_message(
                    f"Frequency {frequency} not found in fields for monitor "
                    f"'{surface.monitor.name}'",
                    e,
                )
            ) from e

    _, idx_uv = surface.monitor.pop_axis((0, 1, 2), axis=surface.axis)
    _, source_names = surface.monitor.pop_axis(("x", "y", "z"), axis=surface.axis)

    idx_integration_1d = None
    zero_dims = [dim for dim, size in enumerate(simulation_size) if size == 0]
    if is_2d_simulation:
        if len(zero_dims) != 1:
            raise ValueError("Expected exactly one dimension with size 0 for 2D simulation")

        zero_dim = zero_dims[0]
        _, plane_axes = Geometry.pop_axis((0, 1, 2), axis=zero_dim)
        if surface.axis == plane_axes[0]:
            idx_integration_1d = plane_axes[1]
        elif surface.axis == plane_axes[1]:
            idx_integration_1d = plane_axes[0]
        else:
            raise ValueError(
                f"Expected surface axis {surface.axis} to lie in the 2D plane {plane_axes}."
            )

    idx_u, idx_v = idx_uv
    cmp_1, cmp_2 = source_names

    propagation_factor = -1j * AbstractFieldProjectionData.wavenumber(
        medium=medium, frequency=frequency
    )

    component_keys = ("E" + cmp_1, "E" + cmp_2, "H" + cmp_1, "H" + cmp_2)

    # The differential areas are already folded into the current values for both schemes
    # (element currents), so the integration metadata is pure geometry shared by all four
    # components.
    integral = _FarFieldIntegralSpec(
        idx_u=idx_u,
        idx_v=idx_v,
        is_2d=is_2d_simulation,
        idx_integration_1d=idx_integration_1d,
    )

    if non_colocated:
        # Each component keeps its native Yee in-plane positions. Components on the same
        # staggered grid share one entry of ``pts_sets`` so their phase factors are evaluated
        # once per distinct grid. The encounter-order dedup over (electric_u, electric_v,
        # magnetic_u, magnetic_v) pairs them two-and-two (see _FarFieldProjectionKernelSpec),
        # yielding component_pts_inds == (0, 1, 1, 0).
        component_data = [select_frequency(currents[key]) for key in component_keys]
        pts_sets = []
        component_pts_inds = []
        seen_pts: dict[tuple[bytes, ...], int] = {}
        for component in component_data:
            pts = tuple(np.asarray(component.coords[name].values) for name in ("x", "y", "z"))
            key = tuple(p.tobytes() for p in pts)
            if key not in seen_pts:
                seen_pts[key] = len(pts_sets)
                pts_sets.append(pts)
            component_pts_inds.append(seen_pts[key])
        pts_sets = tuple(pts_sets)
        component_pts_inds = tuple(component_pts_inds)
    else:
        currents_f = select_frequency(currents)
        component_data = [currents_f[key] for key in component_keys]
        shared_pts = tuple(np.asarray(currents[name].values) for name in ("x", "y", "z"))
        # all four components share the single colocated grid and its phase evaluation
        pts_sets = (shared_pts,)
        component_pts_inds = (0, 0, 0, 0)

    component_arrays = []
    for component in component_data:
        component = component.transpose("x", "y", "z")
        component_arrays.append(anp.reshape(component.data, component.shape))
    field_components = _TangentialSurfaceCurrents(*component_arrays)

    return _PreparedFarFieldProjection(
        field_components=field_components,
        integral=integral,
        pts_sets=pts_sets,
        component_pts_inds=component_pts_inds,
        propagation_factor=propagation_factor,
        eta=ETA_0 / np.sqrt(medium.eps_model(frequency)),
    )


def _projection_data_from_fields(
    data_cls: (
        type[FieldProjectionAngleData]
        | type[FieldProjectionCartesianData]
        | type[FieldProjectionKSpaceData]
    ),
    field_array_cls: (
        type[FieldProjectionAngleDataArray]
        | type[FieldProjectionCartesianDataArray]
        | type[FieldProjectionKSpaceDataArray]
    ),
    *,
    monitor: AbstractFieldProjectionMonitor,
    projection_surfaces: list[FieldProjectionSurface],
    medium: MediumType,
    coords: dict[str, np.ndarray],
    fields: np.ndarray,
    is_2d_simulation: bool | None = None,
) -> FieldProjectionAngleData | FieldProjectionCartesianData | FieldProjectionKSpaceData:
    """Build a projection monitor data object from raw projected field arrays."""

    prototype = field_array_cls(fields[0], coords=coords)
    field_data = {FIELD_COMPONENT_NAMES[0]: prototype}
    for name, field in zip(FIELD_COMPONENT_NAMES[1:], fields[1:]):
        field_data[name] = prototype.copy(deep=False, data=field)

    kwargs = {
        "monitor": monitor,
        "projection_surfaces": projection_surfaces,
        "medium": medium,
        **field_data,
    }
    if is_2d_simulation is not None:
        kwargs["is_2d_simulation"] = is_2d_simulation
    return data_cls.model_construct(**kwargs)


def _normalize_far_field_phases(
    phases: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pad phase arrays with trailing singleton observation dims to share one code path."""

    obs_ndim = max(phase.ndim for phase in phases) - 1
    normalized = []
    for phase in phases:
        trailing_dims = obs_ndim - (phase.ndim - 1)
        normalized.append(phase.reshape(phase.shape + (1,) * trailing_dims))
    return tuple(normalized)


def _broadcast_phase_for_source_axis(
    phase: np.ndarray, *, source_axis: Axis, num_source_axes: int
) -> np.ndarray:
    """Reshape a phase array for multiplication against intermediate source axes."""

    return phase.reshape(
        tuple(phase.shape[0] if axis == source_axis else 1 for axis in range(num_source_axes))
        + phase.shape[1:]
    )


def _assemble_current_vectors(
    projected_components: list[np.ndarray], *, idx_u: Axis, idx_v: Axis, surface_axis: Axis
) -> tuple[np.ndarray, np.ndarray]:
    """Arrange tangential projected current components onto xyz-ordered vectors."""

    order = [idx_u, idx_v, surface_axis]
    zeros = anp.zeros_like(projected_components[0])
    electric = anp.array(
        [projected_components[order.index(i)] if i in order[:2] else zeros for i in range(3)]
    )
    magnetic = anp.array(
        [projected_components[order.index(i) + 2] if i in order[:2] else zeros for i in range(3)]
    )
    return electric, magnetic


def _spherical_far_fields_from_projected_components(
    projected_components: list[np.ndarray],
    *,
    idx_u: Axis,
    idx_v: Axis,
    surface_axis: Axis,
    sin_theta: np.ndarray,
    cos_theta: np.ndarray,
    sin_phi: np.ndarray,
    cos_phi: np.ndarray,
    eta: complex,
    paired_observations: bool,
) -> np.ndarray:
    """Convert projected tangential current components into spherical E/H fields."""

    electric, magnetic = _assemble_current_vectors(
        projected_components,
        idx_u=idx_u,
        idx_v=idx_v,
        surface_axis=surface_axis,
    )

    if paired_observations:
        cos_theta_cos_phi = cos_theta * cos_phi
        cos_theta_sin_phi = cos_theta * sin_phi
        sin_theta_term = sin_theta
        sin_phi_term = sin_phi
        cos_phi_term = cos_phi
    else:
        cos_theta_cos_phi = cos_theta[:, None] * cos_phi[None, :]
        cos_theta_sin_phi = cos_theta[:, None] * sin_phi[None, :]
        sin_theta_term = sin_theta[:, None]
        sin_phi_term = sin_phi[None, :]
        cos_phi_term = cos_phi[None, :]

    n_theta = (
        electric[0] * cos_theta_cos_phi
        + electric[1] * cos_theta_sin_phi
        - electric[2] * sin_theta_term
    )
    n_phi = -electric[0] * sin_phi_term + electric[1] * cos_phi_term
    l_theta = (
        magnetic[0] * cos_theta_cos_phi
        + magnetic[1] * cos_theta_sin_phi
        - magnetic[2] * sin_theta_term
    )
    l_phi = -magnetic[0] * sin_phi_term + magnetic[1] * cos_phi_term

    e_theta = -(l_phi + eta * n_theta)
    e_phi = l_theta - eta * n_phi
    e_r = anp.zeros_like(e_phi)
    h_theta = -e_phi / eta
    h_phi = e_theta / eta
    h_r = anp.zeros_like(h_phi)

    return anp.array([e_r, e_theta, e_phi, h_r, h_theta, h_phi])


def _frequency_chunk_slices(freqs: np.ndarray, freq_chunk_size: int | None) -> tuple[slice, ...]:
    """Return slices that partition frequencies into sequential chunks."""

    if freq_chunk_size is None:
        return (slice(0, len(freqs)),)
    if freq_chunk_size < 1:
        raise ValueError(f"Expected 'freq_chunk_size >= 1', got {freq_chunk_size}.")
    return tuple(
        slice(start, min(start + freq_chunk_size, len(freqs)))
        for start in range(0, len(freqs), freq_chunk_size)
    )


def _far_field_integral(
    currents: np.ndarray,
    phases: tuple[np.ndarray, np.ndarray, np.ndarray],
    spec: _FarFieldIntegralSpec,
) -> np.ndarray:
    """Evaluate the separable far-field surface/line integral.

    This helper computes the near-to-far integral using precomputed separable phase factors,
    with an implementation tailored for autograd. The currents carry their differential areas
    (element currents), so the integral is a pure phase-weighted sum.

    Parameters
    ----------
    currents : np.ndarray
        Complex area-weighted current values on the monitor grid with shape ``(nx, ny, nz)``.
    phases : tuple[np.ndarray, np.ndarray, np.ndarray]
        Phase factors along ``x``, ``y``, and ``z``. Observation dimensions may differ by
        trailing singleton axes, which are normalized internally.
    spec : _FarFieldIntegralSpec
        Static integration geometry.

    Returns
    -------
    np.ndarray
        Integrated values as an array with trailing axes ``(n_theta, n_phi)``.
    """
    phases = _normalize_far_field_phases(phases)
    integrated_axes = spec.integrated_axes
    first_axis = max(integrated_axes)
    result = anp.tensordot(currents, phases[first_axis], axes=((first_axis,), (0,)))
    remaining_axes = [axis for axis in range(3) if axis != first_axis]

    if spec.is_2d:
        for source_axis, axis in enumerate(remaining_axes):
            result = result * _broadcast_phase_for_source_axis(
                phases[axis], source_axis=source_axis, num_source_axes=len(remaining_axes)
            )
        return result

    second_axis = next(axis for axis in integrated_axes if axis != first_axis)
    source_axis = remaining_axes.index(second_axis)
    result = anp.sum(
        result
        * _broadcast_phase_for_source_axis(
            phases[second_axis], source_axis=source_axis, num_source_axes=len(remaining_axes)
        ),
        axis=source_axis,
    )
    return result * phases[spec.remaining_axis]


def _far_field_integral_pairs(
    currents: np.ndarray,
    phases: tuple[np.ndarray, np.ndarray, np.ndarray],
    spec: _FarFieldIntegralSpec,
) -> np.ndarray:
    """Evaluate the far-field integral for paired observation points."""

    phase_0, phase_1, phase_2 = phases
    if spec.is_2d:
        line_axis = spec.line_axis
        currents_phase = anp.tensordot(currents, phases[line_axis], axes=((line_axis,), (0,)))
        remaining_axes = [axis for axis in range(3) if axis != line_axis]
        for source_axis, axis in enumerate(remaining_axes):
            currents_phase = currents_phase * _broadcast_phase_for_source_axis(
                phases[axis], source_axis=source_axis, num_source_axes=len(remaining_axes)
            )
        return currents_phase

    output_axis = spec.remaining_axis

    if output_axis == 0:
        currents_phase = anp.tensordot(currents, phase_2, axes=((2,), (0,)))
        return anp.sum(currents_phase * phase_1[None, :, :], axis=1) * phase_0
    if output_axis == 1:
        currents_phase = anp.tensordot(currents, phase_2, axes=((2,), (0,)))
        return anp.sum(currents_phase * phase_0[:, None, :], axis=0) * phase_1

    currents_phase = anp.tensordot(currents, phase_1, axes=((1,), (0,)))
    return anp.sum(currents_phase * phase_0[:, None, :], axis=0) * phase_2


def _far_field_integral_pairs_3d_numpy(
    currents: np.ndarray,
    phase_0: np.ndarray,
    phase_1: np.ndarray,
    phase_2: np.ndarray,
    spec: _FarFieldIntegralSpec,
) -> np.ndarray:
    """Evaluate a 3D paired far-field integral using NumPy.

    The paired Cartesian/k-space projection path evaluates this integral once per
    field component, frequency, source surface, and observation batch. Letting
    autograd trace every weighting, contraction, and summation retains a large
    graph. The integral is linear in ``currents`` and all phase factors are static
    for the local projection, so the VJP can apply the adjoint integral directly.
    """

    output_axis = spec.remaining_axis

    if output_axis == 0:
        currents_phase = np.tensordot(currents, phase_2, axes=((2,), (0,)))
        return np.sum(currents_phase * phase_1[None, :, :], axis=1) * phase_0
    if output_axis == 1:
        currents_phase = np.tensordot(currents, phase_2, axes=((2,), (0,)))
        return np.sum(currents_phase * phase_0[:, None, :], axis=0) * phase_1

    currents_phase = np.tensordot(currents, phase_1, axes=((1,), (0,)))
    return np.sum(currents_phase * phase_0[:, None, :], axis=0) * phase_2


def _paired_weighted_phase_sum(
    phase_left: np.ndarray, phase_right: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """Contract paired phase products over observation points."""

    return anp.tensordot(phase_left, phase_right * weights[None, :], axes=((1,), (1,)))


def _far_field_integral_pairs_3d_currents_vjp(
    g: np.ndarray,
    phase_0: np.ndarray,
    phase_1: np.ndarray,
    phase_2: np.ndarray,
    spec: _FarFieldIntegralSpec,
) -> np.ndarray:
    """Apply the reverse paired far-field integral to an output cotangent."""

    output_axis = spec.remaining_axis

    if output_axis == 0:
        return anp.stack(
            [
                _paired_weighted_phase_sum(phase_1, phase_2, weighted_bar)
                for weighted_bar in g * phase_0
            ],
            axis=0,
        )

    if output_axis == 1:
        return anp.stack(
            [
                _paired_weighted_phase_sum(phase_0, phase_2, weighted_bar)
                for weighted_bar in g * phase_1
            ],
            axis=1,
        )

    return anp.stack(
        [
            _paired_weighted_phase_sum(phase_0, phase_1, weighted_bar)
            for weighted_bar in g * phase_2
        ],
        axis=2,
    )


def _paired_far_field_phases(
    pts_0: np.ndarray,
    pts_1: np.ndarray,
    pts_2: np.ndarray,
    propagation_factor: complex,
    sin_theta: np.ndarray,
    cos_theta: np.ndarray,
    sin_phi: np.ndarray,
    cos_phi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build separable paired-observation phase arrays."""

    phase_0 = np.exp((propagation_factor * pts_0)[:, None] * sin_theta[None, :] * cos_phi[None, :])
    phase_1 = np.exp((propagation_factor * pts_1)[:, None] * sin_theta[None, :] * sin_phi[None, :])
    phase_2 = np.exp((propagation_factor * pts_2)[:, None] * cos_theta[None, :])
    return phase_0, phase_1, phase_2


def _far_fields_from_currents_pairs_3d(
    field_components: _TangentialSurfaceCurrents,
    kernel_spec: _FarFieldProjectionKernelSpec,
    surface_axis: Axis,
    angle_trig: _PairedAngleTrig,
) -> np.ndarray:
    """Project 3D paired fields and convert to spherical E/H with one custom VJP."""

    return _far_fields_from_currents_pairs_3d_components(
        *field_components.as_tuple(),
        kernel_spec,
        surface_axis,
        angle_trig,
    )


@primitive
def _far_fields_from_currents_pairs_3d_components(
    electric_u: np.ndarray,
    electric_v: np.ndarray,
    magnetic_u: np.ndarray,
    magnetic_v: np.ndarray,
    kernel_spec: _FarFieldProjectionKernelSpec,
    surface_axis: Axis,
    angle_trig: _PairedAngleTrig,
) -> np.ndarray:
    """Project separate tangential current arrays with component-wise VJPs."""

    # one phase evaluation per distinct source grid, indexed per component
    phase_sets = tuple(
        _paired_far_field_phases(
            *pts,
            kernel_spec.propagation_factor,
            angle_trig.sin_theta,
            angle_trig.cos_theta,
            angle_trig.sin_phi,
            angle_trig.cos_phi,
        )
        for pts in kernel_spec.pts_sets
    )

    def project_component(
        component: np.ndarray,
        phases: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> np.ndarray:
        """Project one current component onto the paired observation angles."""

        phase_0, phase_1, phase_2 = phases
        return np.reshape(
            _far_field_integral_pairs_3d_numpy(
                component, phase_0, phase_1, phase_2, kernel_spec.integral
            ),
            angle_trig.sin_theta.shape,
        )

    projected_components = [
        project_component(component, phase_sets[pts_ind])
        for component, pts_ind in zip(
            (electric_u, electric_v, magnetic_u, magnetic_v),
            kernel_spec.component_pts_inds,
        )
    ]

    return _spherical_far_fields_from_projected_components(
        projected_components,
        idx_u=kernel_spec.integral.idx_u,
        idx_v=kernel_spec.integral.idx_v,
        surface_axis=surface_axis,
        sin_theta=angle_trig.sin_theta,
        cos_theta=angle_trig.cos_theta,
        sin_phi=angle_trig.sin_phi,
        cos_phi=angle_trig.cos_phi,
        eta=kernel_spec.eta,
        paired_observations=True,
    )


def _projected_axis_cotangent_pairs(
    *,
    axis: Axis,
    theta_bar: np.ndarray,
    phi_bar: np.ndarray,
    sin_theta: np.ndarray,
    cos_theta: np.ndarray,
    sin_phi: np.ndarray,
    cos_phi: np.ndarray,
) -> np.ndarray:
    """Return one xyz component of a projected vector-field cotangent."""

    if axis == 0:
        return theta_bar * cos_theta * cos_phi - phi_bar * sin_phi
    if axis == 1:
        return theta_bar * cos_theta * sin_phi + phi_bar * cos_phi
    return -theta_bar * sin_theta


def _projected_component_cotangent_pairs(
    g: np.ndarray,
    *,
    component_name: _TangentialCurrentComponentName,
    idx_u: Axis,
    idx_v: Axis,
    sin_theta: np.ndarray,
    cos_theta: np.ndarray,
    sin_phi: np.ndarray,
    cos_phi: np.ndarray,
    eta: complex,
) -> np.ndarray:
    """Map spherical E/H cotangents to one projected current-component cotangent."""

    e_theta_bar = g[1] + g[5] / eta
    e_phi_bar = g[2] - g[4] / eta

    match component_name:
        case "electric_u":
            axis = idx_u
            theta_bar = -eta * e_theta_bar
            phi_bar = -eta * e_phi_bar
        case "electric_v":
            axis = idx_v
            theta_bar = -eta * e_theta_bar
            phi_bar = -eta * e_phi_bar
        case "magnetic_u":
            axis = idx_u
            theta_bar = e_phi_bar
            phi_bar = -e_theta_bar
        case "magnetic_v":
            axis = idx_v
            theta_bar = e_phi_bar
            phi_bar = -e_theta_bar

    return _projected_axis_cotangent_pairs(
        axis=axis,
        theta_bar=theta_bar,
        phi_bar=phi_bar,
        sin_theta=sin_theta,
        cos_theta=cos_theta,
        sin_phi=sin_phi,
        cos_phi=cos_phi,
    )


def _far_fields_from_currents_pairs_3d_component_vjp(
    component_name: _TangentialCurrentComponentName,
) -> _FarFieldsFromCurrentsPairs3DVJPMaker:
    """Build a VJP maker for one tangential current component."""

    component_index = _TangentialSurfaceCurrents.component_index(component_name)

    def vjp_maker(
        _ans: np.ndarray,
        _electric_u: np.ndarray,
        _electric_v: np.ndarray,
        _magnetic_u: np.ndarray,
        _magnetic_v: np.ndarray,
        kernel_spec: _FarFieldProjectionKernelSpec,
        _surface_axis: Axis,
        angle_trig: _PairedAngleTrig,
    ) -> Callable[[np.ndarray], np.ndarray]:
        def vjp(g: np.ndarray) -> np.ndarray:
            spec = kernel_spec.integral
            pts_0, pts_1, pts_2 = kernel_spec.component_pts(component_index)
            projected_bar = _projected_component_cotangent_pairs(
                g,
                component_name=component_name,
                idx_u=spec.idx_u,
                idx_v=spec.idx_v,
                sin_theta=angle_trig.sin_theta,
                cos_theta=angle_trig.cos_theta,
                sin_phi=angle_trig.sin_phi,
                cos_phi=angle_trig.cos_phi,
                eta=kernel_spec.eta,
            )
            output_axis = spec.remaining_axis
            phase_0, phase_1, phase_2 = _paired_far_field_phases(
                pts_0,
                pts_1,
                pts_2,
                kernel_spec.propagation_factor,
                angle_trig.sin_theta,
                angle_trig.cos_theta,
                angle_trig.sin_phi,
                angle_trig.cos_phi,
            )
            output_size = (pts_0.size, pts_1.size, pts_2.size)[output_axis]
            g_projected = anp.reshape(projected_bar, (output_size, -1))

            return _far_field_integral_pairs_3d_currents_vjp(
                g_projected,
                phase_0,
                phase_1,
                phase_2,
                spec,
            )

        return vjp

    return vjp_maker


defvjp(
    _far_fields_from_currents_pairs_3d_components,
    _far_fields_from_currents_pairs_3d_component_vjp("electric_u"),
    _far_fields_from_currents_pairs_3d_component_vjp("electric_v"),
    _far_fields_from_currents_pairs_3d_component_vjp("magnetic_u"),
    _far_fields_from_currents_pairs_3d_component_vjp("magnetic_v"),
    argnums=(0, 1, 2, 3),
)
