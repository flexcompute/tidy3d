"""Near field to far field transformation plugin"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, TypeVar, Union

import autograd.numpy as anp
import numpy as np
import xarray as xr
from pydantic import Field, model_validator
from rich.progress import track

from tidy3d.constants import C_0, EPSILON_0, ETA_0, MICROMETER, MU_0
from tidy3d.exceptions import SetupError, format_chained_exception_message
from tidy3d.log import get_logging_console

from .autograd.functions import add_at, trapz
from .base import Tidy3dBaseModel, cached_property
from .data.data_array import (
    FieldProjectionAngleDataArray,
    FieldProjectionCartesianDataArray,
    FieldProjectionKSpaceDataArray,
)
from .data.monitor_data import (
    AbstractFieldProjectionData,
    FieldData,
    FieldProjectionAngleData,
    FieldProjectionCartesianData,
    FieldProjectionKSpaceData,
)
from .data.sim_data import SimulationData
from .geometry.base import Geometry
from .monitor import (
    FieldProjectionAngleMonitor,
    FieldProjectionCartesianMonitor,
    FieldProjectionSurface,
)
from .types import ArrayComplex4D, Axis, Coordinate
from .validators import validate_field_projection_monitors_2d

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from tidy3d.compat import Self

    from .medium import MediumType
    from .monitor import AbstractFieldProjectionMonitor, FieldMonitor, FieldProjectionKSpaceMonitor
    from .types import Direction

# Default number of points per wavelength in the background medium to use for resampling fields.
PTS_PER_WVL = 10
APPROX_PROJECTION_BATCH_SIZE = 512
APPROX_PROJECTION_FREQ_CHUNK_SIZE = 8
FIELD_COMPONENT_NAMES = ("Er", "Etheta", "Ephi", "Hr", "Htheta", "Hphi")
AXIS_WEIGHT_SHAPES = ((-1, 1, 1), (1, -1, 1), (1, 1, -1))

# Numpy float array and related array types

ArrayLikeN2F = Union[float, tuple[float, ...], ArrayComplex4D]
_TrackedItem = TypeVar("_TrackedItem")


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
    """Static metadata for a separable far-field integral."""

    weights: tuple[np.ndarray, np.ndarray, np.ndarray]
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
class _PreparedFarFieldProjection:
    """Prepared arrays reused across far-field evaluations at one frequency."""

    field_components: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    integral: _FarFieldIntegralSpec
    pts: tuple[np.ndarray, np.ndarray, np.ndarray]
    propagation_factor: complex
    eta: complex


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
    return data_cls(**kwargs)


def _trapz_weights_1d(points: np.ndarray) -> np.ndarray:
    """Trapezoidal integration weights for `trapz(y, x=points)`.

    Parameters
    ----------
    points : np.ndarray
        1D array of integration points.

    Returns
    -------
    np.ndarray
        Trapezoidal integration weights with shape ``(len(points),)``.
    """
    points = np.asarray(points)
    num_points = points.size
    if num_points <= 1:
        return np.ones((num_points,), dtype=float)

    deltas = np.diff(points)
    interior = (deltas[:-1] + deltas[1:]) / 2
    return np.concatenate(([deltas[0] / 2], interior, [deltas[-1] / 2]))


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


def _apply_axis_weights(
    currents: np.ndarray,
    weights: tuple[np.ndarray, np.ndarray, np.ndarray],
    axes: tuple[Axis, ...],
) -> np.ndarray:
    """Apply separable trapezoidal weights along the requested source axes."""

    weighted = currents
    for axis in axes:
        weighted = weighted * weights[axis].reshape(AXIS_WEIGHT_SHAPES[axis])
    return weighted


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

    This helper computes the near-to-far integral using precomputed separable phase factors
    and trapezoidal integration weights, with an implementation tailored for autograd.

    Parameters
    ----------
    currents : np.ndarray
        Complex surface current values on the monitor grid with shape ``(nx, ny, nz)``.
    phases : tuple[np.ndarray, np.ndarray, np.ndarray]
        Phase factors along ``x``, ``y``, and ``z``. Observation dimensions may differ by
        trailing singleton axes, which are normalized internally.
    spec : _FarFieldIntegralSpec
        Static integration metadata, including source-axis weights and dimensionality.

    Returns
    -------
    np.ndarray
        Integrated values as an array with trailing axes ``(n_theta, n_phi)``.
    """
    phases = _normalize_far_field_phases(phases)
    integrated_axes = spec.integrated_axes
    weighted_currents = _apply_axis_weights(currents, spec.weights, integrated_axes)
    first_axis = max(integrated_axes)
    result = anp.tensordot(weighted_currents, phases[first_axis], axes=((first_axis,), (0,)))
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
        weighted_currents = _apply_axis_weights(currents, spec.weights, (line_axis,))
        currents_phase = anp.tensordot(
            weighted_currents, phases[line_axis], axes=((line_axis,), (0,))
        )
        remaining_axes = [axis for axis in range(3) if axis != line_axis]
        for source_axis, axis in enumerate(remaining_axes):
            currents_phase = currents_phase * _broadcast_phase_for_source_axis(
                phases[axis], source_axis=source_axis, num_source_axes=len(remaining_axes)
            )
        return currents_phase

    output_axis = spec.remaining_axis
    integrated_axes = spec.integrated_axes
    weighted_currents = _apply_axis_weights(currents, spec.weights, integrated_axes)

    if output_axis == 0:
        currents_phase = anp.tensordot(weighted_currents, phase_2, axes=((2,), (0,)))
        return anp.sum(currents_phase * phase_1[None, :, :], axis=1) * phase_0
    if output_axis == 1:
        currents_phase = anp.tensordot(weighted_currents, phase_2, axes=((2,), (0,)))
        return anp.sum(currents_phase * phase_0[:, None, :], axis=0) * phase_1

    currents_phase = anp.tensordot(weighted_currents, phase_1, axes=((1,), (0,)))
    return anp.sum(currents_phase * phase_0[:, None, :], axis=0) * phase_2


class FieldProjector(Tidy3dBaseModel):
    """Projection of near fields to points on a given observation grid.

    Notes
    -----
    .. TODO make images to illustrate this

    See Also
    --------
    :class:`FieldProjectionAngleMonitor`
        :class:`Monitor` that samples electromagnetic near fields in the frequency domain
        and projects them at given observation angles.

    **Notebooks**:
        * `Performing near field to far field projections <../../notebooks/FieldProjections.html>`_
    """

    sim_data: SimulationData = Field(
        title="Simulation data",
        description="Container for simulation data containing the near field monitors.",
    )

    surfaces: tuple[FieldProjectionSurface, ...] = Field(
        title="Surface monitor with direction",
        description="tuple of each :class:`.FieldProjectionSurface` to use as source of "
        "near field.",
    )

    pts_per_wavelength: Optional[int] = Field(
        PTS_PER_WVL,
        title="Points per wavelength",
        description="Number of points per wavelength in the background medium with which "
        "to discretize the surface monitors for the projection. If ``None``, fields will "
        "will not resampled, but will still be colocated.",
    )

    origin: Optional[Coordinate] = Field(
        None,
        title="Local origin",
        description="Local origin used for defining observation points. If ``None``, uses the "
        "average of the centers of all surface monitors.",
        json_schema_extra={"units": MICROMETER},
    )

    @model_validator(mode="after")
    def _check_origin_set(self) -> Self:
        """Sets ``.origin`` as the average of centers of all surface monitors if not provided."""
        if self.origin is None:
            centers = np.array([surface.monitor.center for surface in self.surfaces])
            object.__setattr__(self, "origin", tuple(np.mean(centers, axis=0)))
        return self

    @cached_property
    def is_2d_simulation(self) -> bool:
        non_zero_dims = sum(1 for size in self.sim_data.simulation.size if size != 0)
        return non_zero_dims == 2

    @cached_property
    def medium(self) -> MediumType:
        """Medium into which fields are to be projected."""
        sim = self.sim_data.simulation
        monitor = self.surfaces[0].monitor
        return sim.monitor_medium(monitor)

    @cached_property
    def frequencies(self) -> list[float]:
        """Return the list of frequencies associated with the field monitors."""
        return self.surfaces[0].monitor.freqs

    @classmethod
    def from_near_field_monitors(
        cls,
        sim_data: SimulationData,
        near_monitors: list[FieldMonitor],
        normal_dirs: list[Direction],
        pts_per_wavelength: int = PTS_PER_WVL,
        origin: Coordinate = None,
    ) -> Self:
        """Constructs :class:`FieldProjection` from a list of surface monitors and their directions.

        Parameters
        ----------
        sim_data : :class:`.SimulationData`
            Container for simulation data containing the near field monitors.
        near_monitors : list[:class:`.FieldMonitor`]
            tuple of :class:`.FieldMonitor` objects on which near fields will be sampled.
        normal_dirs : list[:class:`.Direction`]
            tuple containing the :class:`.Direction` of the normal to each surface monitor
            w.r.t. to the positive x, y or z unit vectors. Must have the same length as monitors.
        pts_per_wavelength : int = 10
            Number of points per wavelength with which to discretize the
            surface monitors for the projection. If ``None``, fields will not be resampled.
        origin : :class:`.Coordinate`
            Local origin used for defining observation points. If ``None``, uses the
            average of the centers of all surface monitors.
        """

        if len(near_monitors) != len(normal_dirs):
            raise SetupError(
                f"Number of monitors ({len(near_monitors)}) does not equal "
                f"the number of directions ({len(normal_dirs)})."
            )

        surfaces = [
            FieldProjectionSurface(monitor=monitor, normal_dir=normal_dir)
            for monitor, normal_dir in zip(near_monitors, normal_dirs)
        ]

        return cls(
            sim_data=sim_data,
            surfaces=surfaces,
            pts_per_wavelength=pts_per_wavelength,
            origin=origin,
        )

    @cached_property
    def currents(self) -> dict[str, xr.Dataset]:
        """Sets the surface currents."""
        sim_data = self.sim_data
        surfaces = self.surfaces
        pts_per_wavelength = self.pts_per_wavelength
        medium = self.medium

        surface_currents = {}
        for surface in surfaces:
            current_data = self.compute_surface_currents(
                sim_data, surface, medium, pts_per_wavelength
            )

            # shift source coordinates relative to the local origin
            current_data = current_data.assign_coords(
                {
                    name: current_data.coords[name] - origin
                    for name, origin in zip(["x", "y", "z"], self.origin)
                }
            )

            surface_currents[surface.monitor.name] = current_data

        return surface_currents

    @staticmethod
    def compute_surface_currents(
        sim_data: SimulationData,
        surface: FieldProjectionSurface,
        medium: MediumType,
        pts_per_wavelength: int = PTS_PER_WVL,
    ) -> xr.Dataset:
        """Returns resampled surface current densities associated with the surface monitor.

        Parameters
        ----------
        sim_data : :class:`.SimulationData`
            Container for simulation data containing the near field monitors.
        surface: :class:`.FieldProjectionSurface`
            :class:`.FieldProjectionSurface` to use as source of near field.
        medium : :class:`.MediumType`
            Background medium through which to project fields.
        pts_per_wavelength : int = 10
            Number of points per wavelength with which to discretize the
            surface monitors for the projection. If ``None``, fields will not be
            resampled, but will still be colocated.

        Returns
        -------
        xarray.Dataset
            Colocated surface current densities for the given surface.
        """

        monitor_name = surface.monitor.name
        if monitor_name not in sim_data.monitor_data.keys():
            raise SetupError(f"No data for monitor named '{monitor_name}' found in sim_data.")

        field_data = sim_data[monitor_name]

        currents = FieldProjector._fields_to_currents(field_data, surface)
        currents = FieldProjector._resample_surface_currents(
            currents, sim_data, surface, medium, pts_per_wavelength
        )

        return currents

    @staticmethod
    def _fields_to_currents(field_data: FieldData, surface: FieldProjectionSurface) -> FieldData:
        """Returns surface current densities associated with a given :class:`.FieldData` object.

        Parameters
        ----------
        field_data : :class:`.FieldData`
            Container for field data associated with the given near field surface.
        surface: :class:`.FieldProjectionSurface`
            :class:`.FieldProjectionSurface` to use as source of near field.

        Returns
        -------
        :class:`.FieldData`
            Surface current densities for the given surface.
        """

        # figure out which field components are tangential or normal to the monitor
        _, (cmp_1, cmp_2) = surface.monitor.pop_axis(("x", "y", "z"), axis=surface.axis)

        signs = np.array([-1, 1])
        if surface.axis % 2 != 0:
            signs *= -1
        if surface.normal_dir == "-":
            signs *= -1

        E1 = "E" + cmp_1
        E2 = "E" + cmp_2
        H1 = "H" + cmp_1
        H2 = "H" + cmp_2

        surface_currents = {}

        surface_currents[E2] = field_data.field_components[H1] * signs[1]
        surface_currents[E1] = field_data.field_components[H2] * signs[0]

        surface_currents[H2] = field_data.field_components[E1] * signs[0]
        surface_currents[H1] = field_data.field_components[E2] * signs[1]

        new_monitor = surface.monitor.copy(update={"fields": (E1, E2, H1, H2)})

        return FieldData(
            monitor=new_monitor,
            symmetry=field_data.symmetry,
            symmetry_center=field_data.symmetry_center,
            grid_expanded=field_data.grid_expanded,
            **surface_currents,
        )

    @staticmethod
    def _resample_surface_currents(
        currents: FieldData,
        sim_data: SimulationData,
        surface: FieldProjectionSurface,
        medium: MediumType,
        pts_per_wavelength: int = PTS_PER_WVL,
    ) -> xr.Dataset:
        """Returns the surface current densities associated with the surface monitor.

        Parameters
        ----------
        currents : :class:`.FieldData`
            Surface currents defined on the original Yee grid.
        sim_data : :class:`.SimulationData`
            Container for simulation data containing the near field monitors.
        surface: :class:`.FieldProjectionSurface`
            :class:`.FieldProjectionSurface` to use as source of near field.
        medium : :class:`.MediumType`
            Background medium through which to project fields.
        pts_per_wavelength : int = 10
            Number of points per wavelength with which to discretize the
            surface monitors for the projection. If ``None``, fields will not be
            resampled, but will still be colocated.

        Returns
        -------
        xarray.Dataset
            Colocated surface current densities for the given surface.
        """

        # colocate surface currents on a regular grid of points on the monitor based on wavelength
        colocation_points = [None] * 3
        colocation_points[surface.axis] = surface.monitor.center[surface.axis]

        # use the highest frequency associated with the monitor to resample the surface currents
        frequency = max(surface.monitor.freqs)
        eps_complex = medium.eps_model(frequency)
        index_n, _ = medium.eps_complex_to_nk(eps_complex)
        wavelength = C_0 / frequency / index_n

        _, idx_uv = surface.monitor.pop_axis((0, 1, 2), axis=surface.axis)
        coord_list = sim_data.simulation.grid.boundaries.to_list
        for idx in idx_uv:
            coord_name = "xyz"[idx]

            # Skip resampling along dimensions where the current data has only one source
            # coordinate, such as the collapsed axis of a 2D simulation.
            if any(
                np.array(field_data.coords[coord_name]).size <= 1
                for field_data in currents.field_components.values()
                if field_data is not None
            ):
                continue

            # pick sample points on the monitor and handle the possibility of an "infinite" monitor
            # Fields within PML regions are included, to match the server-side computation.
            start = np.maximum(
                surface.monitor.center[idx] - surface.monitor.size[idx] / 2.0,
                coord_list[idx][0],
            )
            stop = np.minimum(
                surface.monitor.center[idx] + surface.monitor.size[idx] / 2.0,
                coord_list[idx][-1],
            )
            if pts_per_wavelength is None:
                points = sim_data.simulation.grid.boundaries.to_list[idx].copy()
                points[np.argwhere(points < start)] = start
                points[np.argwhere(points > stop)] = stop
                colocation_points[idx] = np.unique(points)
            else:
                size = stop - start
                num_pts = int(np.ceil(pts_per_wavelength * size / wavelength))
                points = np.linspace(start, stop, num_pts)
                colocation_points[idx] = points

        for idx, points in enumerate(colocation_points):
            if np.array(points).size <= 1:
                colocation_points[idx] = None

        currents = currents.colocate(*colocation_points)
        return currents

    @staticmethod
    def trapezoid(
        ary: NDArray,
        pts: Union[Iterable[NDArray], NDArray],
        axes: Union[Iterable[int], int] = 0,
    ) -> NDArray:
        """Trapezoidal integration in n dimensions.

        Parameters
        ----------
        ary : np.ndarray
            Array to integrate.
        pts : Iterable[np.ndarray]
            Iterable of points for each dimension.
        axes : Union[Iterable[int], int]
            Iterable of axes along which to integrate. If not an iterable, assume 1D integration.

        Returns
        -------
        np.ndarray
            Integrated array.
        """
        if not isinstance(axes, Iterable):
            axes = [axes]
            pts = [pts]

        for idx, (axis, pt) in enumerate(zip(axes, pts)):
            if ary.shape[axis - idx] > 1:
                ary = trapz(ary, pt, axis=axis - idx)
            else:  # array has only one element along axis
                ary = ary[(slice(None),) * (axis - idx) + (0,)]
        return ary

    def _far_fields_for_surface(
        self,
        frequency: float,
        theta: ArrayLikeN2F,
        phi: ArrayLikeN2F,
        surface: FieldProjectionSurface,
        currents: xr.Dataset,
        medium: MediumType,
    ) -> NDArray:
        """Compute far fields at an angle in spherical coordinates
        for a given set of surface currents and observation angles.

        Parameters
        ----------
        frequency : float
            Frequency to select from each :class:`.FieldMonitor` to use for projection.
            Must be a frequency stored in each :class:`FieldMonitor`.
        theta : Union[float, tuple[float, ...], np.ndarray]
            Polar angles (rad) downward from x=y=0 line relative to the local origin.
        phi : Union[float, tuple[float, ...], np.ndarray]
            Azimuthal (rad) angles from y=z=0 line relative to the local origin.
        surface: :class:`FieldProjectionSurface`
            :class:`FieldProjectionSurface` object to use as source of near field.
        currents : xarray.Dataset
            xarray Dataset containing surface currents associated with the surface monitor.
        medium : :class:`.MediumType`
            Background medium through which to project fields.

        Returns
        -------
        np.ndarray
            With leading dimension containing ``Er``, ``Etheta``, ``Ephi``, ``Hr``, ``Htheta``, ``Hphi``
            projected fields for each frequency.
        """
        prepared = self._prepare_far_field_projection(
            frequency=frequency,
            surface=surface,
            currents=currents,
            medium=medium,
        )
        return self._far_fields_from_prepared(
            theta=theta, phi=phi, prepared=prepared, surface_axis=surface.axis
        )

    def _prepare_far_field_projection(
        self,
        frequency: float,
        surface: FieldProjectionSurface,
        currents: xr.Dataset,
        medium: MediumType,
    ) -> _PreparedFarFieldProjection:
        """Prepare raw arrays needed for repeated far-field evaluations at one frequency."""
        try:
            currents_f = currents.sel(f=frequency)
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

        # integration dimension for 2d far field projection
        idx_integration_1d = None
        zero_dim = [dim for dim, size in enumerate(self.sim_data.simulation.size) if size == 0]
        if self.is_2d_simulation:
            # Ensure zero_dim has a single element since {zero_dim} expects a value
            if len(zero_dim) != 1:
                raise ValueError("Expected exactly one dimension with size 0 for 2D simulation")

            zero_dim = zero_dim[0]
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

        pts = tuple(currents[name].values for name in ("x", "y", "z"))
        weights = tuple(_trapz_weights_1d(pt) for pt in pts)

        E1 = "E" + cmp_1
        E2 = "E" + cmp_2
        H1 = "H" + cmp_1
        H2 = "H" + cmp_2

        field_components = tuple(
            anp.reshape(currents_f[field_component].data, currents_f[field_component].shape)
            for field_component in (E1, E2, H1, H2)
        )

        return _PreparedFarFieldProjection(
            field_components=field_components,
            integral=_FarFieldIntegralSpec(
                weights=weights,
                idx_u=idx_u,
                idx_v=idx_v,
                is_2d=self.is_2d_simulation,
                idx_integration_1d=idx_integration_1d,
            ),
            pts=pts,
            propagation_factor=propagation_factor,
            eta=ETA_0 / np.sqrt(medium.eps_model(frequency)),
        )

    def _far_fields_from_prepared(
        self,
        theta: ArrayLikeN2F,
        phi: ArrayLikeN2F,
        prepared: _PreparedFarFieldProjection,
        surface_axis: Axis,
    ) -> NDArray:
        """Evaluate far fields from pre-sliced current data."""
        theta = np.atleast_1d(theta)
        phi = np.atleast_1d(phi)

        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)

        pts = prepared.pts
        propagation_factor = prepared.propagation_factor
        phase_0 = np.exp(
            (propagation_factor * pts[0])[:, None, None]
            * sin_theta[None, :, None]
            * cos_phi[None, None, :]
        )
        phase_1 = np.exp(
            (propagation_factor * pts[1])[:, None, None]
            * sin_theta[None, :, None]
            * sin_phi[None, None, :]
        )
        phase_2 = np.exp((propagation_factor * pts[2])[:, None] * cos_theta[None, :])

        jm = []
        phases = (phase_0, phase_1, phase_2)
        for field_component in prepared.field_components:
            jm_i = _far_field_integral(
                field_component,
                phases,
                prepared.integral,
            )

            jm.append(anp.reshape(jm_i, (len(theta), len(phi))))

        J, M = _assemble_current_vectors(
            jm,
            idx_u=prepared.integral.idx_u,
            idx_v=prepared.integral.idx_v,
            surface_axis=surface_axis,
        )

        cos_theta_cos_phi = cos_theta[:, None] * cos_phi[None, :]
        cos_theta_sin_phi = cos_theta[:, None] * sin_phi[None, :]

        # Ntheta (8.33a)
        Ntheta = J[0] * cos_theta_cos_phi + J[1] * cos_theta_sin_phi - J[2] * sin_theta[:, None]

        # Nphi (8.33b)
        Nphi = -J[0] * sin_phi[None, :] + J[1] * cos_phi[None, :]

        # Ltheta  (8.34a)
        Ltheta = M[0] * cos_theta_cos_phi + M[1] * cos_theta_sin_phi - M[2] * sin_theta[:, None]

        # Lphi  (8.34b)
        Lphi = -M[0] * sin_phi[None, :] + M[1] * cos_phi[None, :]

        eta = prepared.eta

        Etheta = -(Lphi + eta * Ntheta)
        Ephi = Ltheta - eta * Nphi
        Er = anp.zeros_like(Ephi)
        Htheta = -Ephi / eta
        Hphi = Etheta / eta
        Hr = anp.zeros_like(Hphi)

        return anp.array([Er, Etheta, Ephi, Hr, Htheta, Hphi])

    def _far_fields_from_prepared_pairs(
        self,
        theta: ArrayLikeN2F,
        phi: ArrayLikeN2F,
        prepared: _PreparedFarFieldProjection,
        surface_axis: Axis,
    ) -> NDArray:
        """Evaluate far fields for paired observation angles."""
        theta = np.reshape(theta, (-1,))
        phi = np.reshape(phi, (-1,))

        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)

        pts = prepared.pts
        propagation_factor = prepared.propagation_factor
        phase_0 = np.exp(
            (propagation_factor * pts[0])[:, None] * sin_theta[None, :] * cos_phi[None, :]
        )
        phase_1 = np.exp(
            (propagation_factor * pts[1])[:, None] * sin_theta[None, :] * sin_phi[None, :]
        )
        phase_2 = np.exp((propagation_factor * pts[2])[:, None] * cos_theta[None, :])

        jm = []
        phases = (phase_0, phase_1, phase_2)
        for field_component in prepared.field_components:
            jm_i = _far_field_integral_pairs(field_component, phases, prepared.integral)
            jm.append(anp.reshape(jm_i, theta.shape))

        J, M = _assemble_current_vectors(
            jm,
            idx_u=prepared.integral.idx_u,
            idx_v=prepared.integral.idx_v,
            surface_axis=surface_axis,
        )

        cos_theta_cos_phi = cos_theta * cos_phi
        cos_theta_sin_phi = cos_theta * sin_phi

        Ntheta = J[0] * cos_theta_cos_phi + J[1] * cos_theta_sin_phi - J[2] * sin_theta
        Nphi = -J[0] * sin_phi + J[1] * cos_phi
        Ltheta = M[0] * cos_theta_cos_phi + M[1] * cos_theta_sin_phi - M[2] * sin_theta
        Lphi = -M[0] * sin_phi + M[1] * cos_phi

        eta = prepared.eta

        Etheta = -(Lphi + eta * Ntheta)
        Ephi = Ltheta - eta * Nphi
        Er = anp.zeros_like(Ephi)
        Htheta = -Ephi / eta
        Hphi = Etheta / eta
        Hr = anp.zeros_like(Hphi)

        return anp.array([Er, Etheta, Ephi, Hr, Htheta, Hphi])

    def _project_prepared_fields_pairs(
        self,
        theta: np.ndarray,
        phi: np.ndarray,
        prepared_surface_currents: list[
            tuple[FieldProjectionSurface, list[_PreparedFarFieldProjection]]
        ],
        phase_by_freq: np.ndarray,
    ) -> NDArray:
        """Project approximate fields for paired observation points."""
        num_points = theta.size
        phase_by_freq = anp.asarray(phase_by_freq)

        fields_by_freq = []
        for idx_f in range(phase_by_freq.shape[0]):
            fields_sum = anp.zeros((6, num_points), dtype=complex)
            for surface, prepared_by_freq in prepared_surface_currents:
                fields_surface = self._far_fields_from_prepared_pairs(
                    theta=theta,
                    phi=phi,
                    prepared=prepared_by_freq[idx_f],
                    surface_axis=surface.axis,
                )
                fields_sum = fields_sum + fields_surface * phase_by_freq[idx_f][None, :]
            fields_by_freq.append(fields_sum)

        return anp.moveaxis(anp.stack(fields_by_freq, axis=-1), 1, 0)

    @staticmethod
    def apply_window_to_currents(
        proj_monitor: AbstractFieldProjectionMonitor, currents: xr.Dataset
    ) -> xr.Dataset:
        """Apply windowing function to the surface currents."""
        if proj_monitor.size.count(0.0) == 0:
            return currents
        if proj_monitor.window_size == (0, 0):
            return currents

        pts = [currents[name].values for name in ["x", "y", "z"]]

        custom_bounds = [
            [pts[i][0] for i in range(3)],
            [pts[i][-1] for i in range(3)],
        ]

        window_size, window_minus, window_plus = proj_monitor.window_parameters(
            custom_bounds=custom_bounds
        )

        new_currents = currents.copy(deep=True)
        for dim, (dim_name, points) in enumerate(zip("xyz", pts)):
            window_fn = proj_monitor.window_function(
                points=points,
                window_size=window_size,
                window_minus=window_minus,
                window_plus=window_plus,
                dim=dim,
            )
            window_data = xr.DataArray(
                window_fn,
                dims=[dim_name],
                coords=[points],
            )
            new_currents *= window_data

        return new_currents

    def _windowed_surface_currents(
        self, proj_monitor: AbstractFieldProjectionMonitor
    ) -> list[tuple[FieldProjectionSurface, xr.Dataset]]:
        """Collect projection surfaces together with their windowed currents."""

        return [
            (
                surface,
                self.apply_window_to_currents(proj_monitor, self.currents[surface.monitor.name]),
            )
            for surface in self.surfaces
        ]

    def _prepare_far_field_surface_currents(
        self,
        surface_currents: list[tuple[FieldProjectionSurface, xr.Dataset]],
        freqs: np.ndarray,
        medium: MediumType,
    ) -> list[tuple[FieldProjectionSurface, list[_PreparedFarFieldProjection]]]:
        """Prepare reusable per-frequency far-field data for each surface."""

        return [
            (
                surface,
                [
                    self._prepare_far_field_projection(
                        frequency=frequency,
                        surface=surface,
                        currents=currents,
                        medium=medium,
                    )
                    for frequency in freqs
                ],
            )
            for surface, currents in surface_currents
        ]

    def _project_exact_fields_points(
        self,
        x: ArrayLikeN2F,
        y: ArrayLikeN2F,
        z: ArrayLikeN2F,
        surface_currents: list[tuple[FieldProjectionSurface, xr.Dataset]],
        medium: MediumType,
        *,
        verbose: bool,
    ) -> NDArray:
        """Project exact fields for a flat list of observation points."""

        x = np.reshape(x, (-1,))
        y = np.reshape(y, (-1,))
        z = np.reshape(z, (-1,))

        field_shape = (len(FIELD_COMPONENT_NAMES), len(self.frequencies))
        point_fields = []
        for x_obs, y_obs, z_obs in _track_if_verbose(
            zip(x, y, z),
            verbose=verbose,
            description="Computing projected fields",
            total=x.size,
        ):
            fields_sum = anp.zeros(field_shape, dtype=complex)
            for surface, currents in surface_currents:
                fields_surface = self._fields_for_surface_exact(
                    x=x_obs,
                    y=y_obs,
                    z=z_obs,
                    surface=surface,
                    currents=currents,
                    medium=medium,
                )
                fields_surface = anp.reshape(fields_surface, field_shape)
                fields_sum = fields_sum + fields_surface
            point_fields.append(fields_sum)

        return anp.stack(point_fields, axis=0)

    def project_fields(
        self,
        proj_monitor: AbstractFieldProjectionMonitor,
        verbose: bool = True,
        freq_chunk_size: int | None = APPROX_PROJECTION_FREQ_CHUNK_SIZE,
    ) -> AbstractFieldProjectionData:
        """Compute projected fields.

        Parameters
        ----------
        proj_monitor : :class:`~tidy3d.components.monitor.AbstractFieldProjectionMonitor`
            Instance of :class:`~tidy3d.components.monitor.AbstractFieldProjectionMonitor` defining
            the projection
            observation grid.
        verbose : bool = True
            Whether to display local progress bars while computing the projection.
        freq_chunk_size : int | None = 8
            Number of frequencies to prepare at once for approximate Cartesian and k-space
            projection. If ``None``, all frequencies are prepared together. Ignored for angular
            and exact projection paths.

        Returns
        -------
        :class:`.AbstractFieldProjectionData`
            Data structure with ``Er``, ``Etheta``, ``Ephi``, ``Hr``, ``Htheta``, ``Hphi``.
        """
        if freq_chunk_size is not None and freq_chunk_size < 1:
            raise ValueError(f"Expected 'freq_chunk_size >= 1', got {freq_chunk_size}.")
        validate_field_projection_monitors_2d((proj_monitor,), self.sim_data.simulation.size)
        if isinstance(proj_monitor, FieldProjectionAngleMonitor):
            return self._project_fields_angular(proj_monitor, verbose=verbose)
        if isinstance(proj_monitor, FieldProjectionCartesianMonitor):
            return self._project_fields_cartesian(
                proj_monitor, verbose=verbose, freq_chunk_size=freq_chunk_size
            )
        return self._project_fields_kspace(
            proj_monitor, verbose=verbose, freq_chunk_size=freq_chunk_size
        )

    def _project_fields_angular(
        self, monitor: FieldProjectionAngleMonitor, verbose: bool = True
    ) -> FieldProjectionAngleData:
        """Compute projected fields on an angle-based grid in spherical coordinates.

        Parameters
        ----------
        monitor : :class:`.FieldProjectionAngleMonitor`
            Instance of :class:`.FieldProjectionAngleMonitor` defining the projection
            observation grid.

        Returns
        -------
        :class:`.FieldProjectionAngleData`
            Data structure with ``Er``, ``Etheta``, ``Ephi``, ``Hr``, ``Htheta``, ``Hphi``.
        """
        freqs = np.atleast_1d(self.frequencies)
        theta = np.atleast_1d(monitor.theta)
        phi = np.atleast_1d(monitor.phi)

        # compute projected fields for the dataset associated with each monitor
        fields = np.zeros(
            (len(FIELD_COMPONENT_NAMES), 1, len(theta), len(phi), len(freqs)), dtype=complex
        )

        medium = monitor.medium if monitor.medium else self.medium
        k = AbstractFieldProjectionData.wavenumber(medium=medium, frequency=freqs)
        phase = np.atleast_1d(
            AbstractFieldProjectionData.propagation_factor(
                dist=monitor.proj_distance, k=k, is_2d_simulation=self.is_2d_simulation
            )
        )

        surface_currents = self._windowed_surface_currents(monitor)

        if monitor.far_field_approx:
            for surface, currents in surface_currents:
                for idx_f, frequency in enumerate(freqs):
                    _fields = self._far_fields_for_surface(
                        frequency=frequency,
                        theta=theta,
                        phi=phi,
                        surface=surface,
                        currents=currents,
                        medium=medium,
                    )
                    fields = add_at(fields, [..., idx_f], _fields[:, None] * phase[idx_f])
        else:
            theta_grid, phi_grid = np.meshgrid(theta, phi, indexing="ij")
            flat_theta = np.reshape(theta_grid, (-1,))
            flat_phi = np.reshape(phi_grid, (-1,))
            flat_x, flat_y, flat_z = monitor.sph_2_car(monitor.proj_distance, flat_theta, flat_phi)
            stacked_fields = self._project_exact_fields_points(
                x=flat_x,
                y=flat_y,
                z=flat_z,
                surface_currents=surface_currents,
                medium=medium,
                verbose=verbose,
            )
            stacked_fields = anp.reshape(
                stacked_fields,
                (len(theta), len(phi), len(FIELD_COMPONENT_NAMES), len(freqs)),
            )
            fields = anp.moveaxis(stacked_fields, 2, 0)[:, None, :, :, :]

        coords = {"r": np.atleast_1d(monitor.proj_distance), "theta": theta, "phi": phi, "f": freqs}
        return _projection_data_from_fields(
            FieldProjectionAngleData,
            FieldProjectionAngleDataArray,
            monitor=monitor,
            projection_surfaces=self.surfaces,
            medium=medium,
            coords=coords,
            fields=fields,
            is_2d_simulation=self.is_2d_simulation,
        )

    def _project_fields_cartesian(
        self,
        monitor: FieldProjectionCartesianMonitor,
        verbose: bool = True,
        freq_chunk_size: int | None = APPROX_PROJECTION_FREQ_CHUNK_SIZE,
    ) -> FieldProjectionCartesianData:
        """Compute projected fields on a Cartesian grid in spherical coordinates.

        Parameters
        ----------
        monitor : :class:`.FieldProjectionCartesianMonitor`
            Instance of :class:`.FieldProjectionCartesianMonitor` defining the projection
            observation grid.

        Returns
        -------
        :class:`.FieldProjectionCartesianData`
            Data structure with ``Er``, ``Etheta``, ``Ephi``, ``Hr``, ``Htheta``, ``Hphi``.
        """
        freqs = np.atleast_1d(self.frequencies)
        x, y, z = monitor.unpop_axis(
            monitor.proj_distance, (monitor.x, monitor.y), axis=monitor.proj_axis
        )
        x, y, z = list(map(np.atleast_1d, [x, y, z]))

        medium = monitor.medium if monitor.medium else self.medium
        wavenumber = AbstractFieldProjectionData.wavenumber(medium=medium, frequency=freqs)

        surface_currents = self._windowed_surface_currents(monitor)
        if monitor.far_field_approx:
            x_grid, y_grid, z_grid = np.meshgrid(x, y, z, indexing="ij")
            flat_x = np.reshape(x_grid, (-1,))
            flat_y = np.reshape(y_grid, (-1,))
            flat_z = np.reshape(z_grid, (-1,))
            r_obs, theta_obs, phi_obs = monitor.car_2_sph(flat_x, flat_y, flat_z)
            total_points = theta_obs.size
            num_batches = (
                total_points + APPROX_PROJECTION_BATCH_SIZE - 1
            ) // APPROX_PROJECTION_BATCH_SIZE
            freq_slices = _frequency_chunk_slices(freqs, freq_chunk_size)
            fields_by_freq_chunk = []
            for idx_chunk, freq_slice in enumerate(freq_slices, start=1):
                prepared_surface_currents = self._prepare_far_field_surface_currents(
                    surface_currents=surface_currents,
                    freqs=freqs[freq_slice],
                    medium=medium,
                )
                description = "Computing projected fields"
                if len(freq_slices) > 1:
                    description = (
                        f"Computing projected fields (freq chunk {idx_chunk}/{len(freq_slices)})"
                    )
                chunk_fields = []
                for start in _track_if_verbose(
                    range(0, total_points, APPROX_PROJECTION_BATCH_SIZE),
                    verbose=verbose,
                    description=description,
                    total=num_batches,
                ):
                    stop = min(start + APPROX_PROJECTION_BATCH_SIZE, total_points)
                    phase = AbstractFieldProjectionData.propagation_factor(
                        dist=r_obs[None, start:stop],
                        k=wavenumber[freq_slice, None],
                        is_2d_simulation=self.is_2d_simulation,
                    )
                    chunk_fields.append(
                        self._project_prepared_fields_pairs(
                            theta=theta_obs[start:stop],
                            phi=phi_obs[start:stop],
                            prepared_surface_currents=prepared_surface_currents,
                            phase_by_freq=phase,
                        )
                    )

                fields_by_freq_chunk.append(anp.concatenate(chunk_fields, axis=0))

            stacked_fields = anp.concatenate(fields_by_freq_chunk, axis=-1)
            stacked_fields = anp.reshape(
                stacked_fields, (len(x), len(y), len(z), len(FIELD_COMPONENT_NAMES), len(freqs))
            )
            fields = anp.moveaxis(stacked_fields, 3, 0)

            coords = {"x": x, "y": y, "z": z, "f": freqs}
            return _projection_data_from_fields(
                FieldProjectionCartesianData,
                FieldProjectionCartesianDataArray,
                monitor=monitor,
                projection_surfaces=self.surfaces,
                medium=medium,
                coords=coords,
                fields=fields,
            )

        x_grid, y_grid, z_grid = np.meshgrid(x, y, z, indexing="ij")
        stacked_fields = self._project_exact_fields_points(
            x=np.reshape(x_grid, (-1,)),
            y=np.reshape(y_grid, (-1,)),
            z=np.reshape(z_grid, (-1,)),
            surface_currents=surface_currents,
            medium=medium,
            verbose=verbose,
        )
        stacked_fields = anp.reshape(
            stacked_fields, (len(x), len(y), len(z), len(FIELD_COMPONENT_NAMES), len(freqs))
        )
        fields = anp.moveaxis(stacked_fields, 3, 0)

        coords = {"x": x, "y": y, "z": z, "f": freqs}
        return _projection_data_from_fields(
            FieldProjectionCartesianData,
            FieldProjectionCartesianDataArray,
            monitor=monitor,
            projection_surfaces=self.surfaces,
            medium=medium,
            coords=coords,
            fields=fields,
        )

    def _project_fields_kspace(
        self,
        monitor: FieldProjectionKSpaceMonitor,
        verbose: bool = True,
        freq_chunk_size: int | None = APPROX_PROJECTION_FREQ_CHUNK_SIZE,
    ) -> FieldProjectionKSpaceData:
        """Compute projected fields on a k-space grid in spherical coordinates.

        Parameters
        ----------
        monitor : :class:`.FieldProjectionKSpaceMonitor`
            Instance of :class:`.FieldProjectionKSpaceMonitor` defining the projection
            observation grid.

        Returns
        -------
        :class:`.FieldProjectionKSpaceData`
            Data structure with ``Er``, ``Etheta``, ``Ephi``, ``Hr``, ``Htheta``, ``Hphi``.
        """
        freqs = np.atleast_1d(self.frequencies)
        ux = np.atleast_1d(monitor.ux)
        uy = np.atleast_1d(monitor.uy)

        medium = monitor.medium if monitor.medium else self.medium
        k = AbstractFieldProjectionData.wavenumber(medium=medium, frequency=freqs)
        phase = np.atleast_1d(
            AbstractFieldProjectionData.propagation_factor(
                dist=monitor.proj_distance, k=k, is_2d_simulation=self.is_2d_simulation
            )
        )[:, None]

        surface_currents = self._windowed_surface_currents(monitor)
        if monitor.far_field_approx:
            ux_grid, uy_grid = np.meshgrid(ux, uy, indexing="ij")
            theta_obs, phi_obs = monitor.kspace_2_sph(
                np.reshape(ux_grid, (-1,)),
                np.reshape(uy_grid, (-1,)),
                monitor.proj_axis,
            )
            total_points = theta_obs.size
            num_batches = (
                total_points + APPROX_PROJECTION_BATCH_SIZE - 1
            ) // APPROX_PROJECTION_BATCH_SIZE
            freq_slices = _frequency_chunk_slices(freqs, freq_chunk_size)
            fields_by_freq_chunk = []
            for idx_chunk, freq_slice in enumerate(freq_slices, start=1):
                prepared_surface_currents = self._prepare_far_field_surface_currents(
                    surface_currents=surface_currents,
                    freqs=freqs[freq_slice],
                    medium=medium,
                )
                description = "Computing projected fields"
                if len(freq_slices) > 1:
                    description = (
                        f"Computing projected fields (freq chunk {idx_chunk}/{len(freq_slices)})"
                    )
                chunk_fields = []
                for start in _track_if_verbose(
                    range(0, total_points, APPROX_PROJECTION_BATCH_SIZE),
                    verbose=verbose,
                    description=description,
                    total=num_batches,
                ):
                    stop = min(start + APPROX_PROJECTION_BATCH_SIZE, total_points)
                    chunk_fields.append(
                        self._project_prepared_fields_pairs(
                            theta=theta_obs[start:stop],
                            phi=phi_obs[start:stop],
                            prepared_surface_currents=prepared_surface_currents,
                            phase_by_freq=phase[freq_slice],
                        )
                    )

                fields_by_freq_chunk.append(anp.concatenate(chunk_fields, axis=0))

            stacked_fields = anp.concatenate(fields_by_freq_chunk, axis=-1)
            stacked_fields = anp.reshape(
                stacked_fields, (len(ux), len(uy), len(FIELD_COMPONENT_NAMES), len(freqs))
            )
            fields = anp.moveaxis(stacked_fields, 2, 0)
            fields = fields[:, :, :, None, :]

            coords = {
                "ux": np.array(monitor.ux),
                "uy": np.array(monitor.uy),
                "r": np.atleast_1d(monitor.proj_distance),
                "f": freqs,
            }
            return _projection_data_from_fields(
                FieldProjectionKSpaceData,
                FieldProjectionKSpaceDataArray,
                monitor=monitor,
                projection_surfaces=self.surfaces,
                medium=medium,
                coords=coords,
                fields=fields,
            )

        ux_grid, uy_grid = np.meshgrid(ux, uy, indexing="ij")
        theta_obs, phi_obs = monitor.kspace_2_sph(
            np.reshape(ux_grid, (-1,)),
            np.reshape(uy_grid, (-1,)),
            monitor.proj_axis,
        )
        x_obs, y_obs, z_obs = monitor.sph_2_car(monitor.proj_distance, theta_obs, phi_obs)
        stacked_fields = self._project_exact_fields_points(
            x=x_obs,
            y=y_obs,
            z=z_obs,
            surface_currents=surface_currents,
            medium=medium,
            verbose=verbose,
        )
        stacked_fields = anp.reshape(
            stacked_fields, (len(ux), len(uy), len(FIELD_COMPONENT_NAMES), len(freqs))
        )
        fields = anp.moveaxis(stacked_fields, 2, 0)
        fields = fields[:, :, :, None, :]

        coords = {
            "ux": np.array(monitor.ux),
            "uy": np.array(monitor.uy),
            "r": np.atleast_1d(monitor.proj_distance),
            "f": freqs,
        }
        return _projection_data_from_fields(
            FieldProjectionKSpaceData,
            FieldProjectionKSpaceDataArray,
            monitor=monitor,
            projection_surfaces=self.surfaces,
            medium=medium,
            coords=coords,
            fields=fields,
        )

    """Exact projections"""

    def _fields_for_surface_exact(
        self,
        x: float,
        y: float,
        z: float,
        surface: FieldProjectionSurface,
        currents: xr.Dataset,
        medium: MediumType,
    ) -> NDArray:
        """Compute projected fields in spherical coordinates at a given projection point on a
        Cartesian grid for a given set of surface currents using the exact homogeneous medium
        Green's function without geometric approximations.

        Parameters
        ----------
        x : float
            Observation point x-coordinate (microns) relative to the local origin.
        y : float
            Observation point y-coordinate (microns) relative to the local origin.
        z : float
            Observation point z-coordinate (microns) relative to the local origin.
        surface: :class:`FieldProjectionSurface`
            :class:`FieldProjectionSurface` object to use as source of near field.
        currents : xarray.Dataset
            xarray Dataset containing surface currents associated with the surface monitor.
        medium : :class:`.MediumType`
            Background medium through which to project fields.

        Returns
        -------
        np.ndarray
            With leading dimension containing ``Er``, ``Etheta``, ``Ephi``, ``Hr``, ``Htheta``, ``Hphi``
            projected fields for each frequency.
        """
        freqs = anp.array(self.frequencies)
        i_omega = 1j * 2.0 * np.pi * freqs[None, None, None, :]
        wavenumber = AbstractFieldProjectionData.wavenumber(frequency=freqs, medium=medium)
        wavenumber = wavenumber[None, None, None, :]  # add space dimensions

        eps_complex = medium.eps_model(frequency=freqs)
        epsilon = EPSILON_0 * eps_complex[None, None, None, :]

        # source points
        pts = [currents[name].values for name in ["x", "y", "z"]]

        # transform the coordinate system so that the origin is at the source point
        # then the observation points in the new system are:
        x_new, y_new, z_new = (pt_obs - pt_src for pt_src, pt_obs in zip(pts, [x, y, z]))

        # tangential source components to use
        idx_w, idx_uv = surface.monitor.pop_axis((0, 1, 2), axis=surface.axis)
        _, source_names = surface.monitor.pop_axis(("x", "y", "z"), axis=surface.axis)

        idx_u, idx_v = idx_uv
        cmp_1, cmp_2 = source_names

        # set the surface current density Cartesian components
        J = [None] * 3
        M = [None] * 3
        J[idx_u] = currents[f"E{cmp_1}"].data
        J[idx_v] = currents[f"E{cmp_2}"].data
        J[idx_w] = anp.zeros(J[idx_u].shape)
        M[idx_u] = currents[f"H{cmp_1}"].data
        M[idx_v] = currents[f"H{cmp_2}"].data
        M[idx_w] = anp.zeros(M[idx_u].shape)

        # observation point in the new spherical system
        r, theta_obs, phi_obs = surface.monitor.car_2_sph(
            x_new[:, None, None, None], y_new[None, :, None, None], z_new[None, None, :, None]
        )

        # angle terms
        sin_theta = anp.sin(theta_obs)
        cos_theta = anp.cos(theta_obs)
        sin_phi = anp.sin(phi_obs)
        cos_phi = anp.cos(phi_obs)

        # Green's function and terms related to its derivatives
        ikr = 1j * wavenumber * r
        G = anp.exp(ikr) / (4.0 * np.pi * r)
        dG_dr = G * (ikr - 1.0) / r
        d2G_dr2 = dG_dr * (ikr - 1.0) / r + G / (r**2)

        # operations between unit vectors and currents
        def r_x_current(current: tuple[NDArray, ...]) -> tuple[NDArray, ...]:
            """Cross product between the r unit vector and the current."""
            return [
                sin_theta * sin_phi * current[2] - cos_theta * current[1],
                cos_theta * current[0] - sin_theta * cos_phi * current[2],
                sin_theta * cos_phi * current[1] - sin_theta * sin_phi * current[0],
            ]

        def r_dot_current(current: tuple[NDArray, ...]) -> NDArray:
            """Dot product between the r unit vector and the current."""
            return (
                sin_theta * cos_phi * current[0]
                + sin_theta * sin_phi * current[1]
                + cos_theta * current[2]
            )

        def r_dot_current_dtheta(current: tuple[NDArray, ...]) -> NDArray:
            """Theta derivative of the dot product between the r unit vector and the current."""
            return (
                cos_theta * cos_phi * current[0]
                + cos_theta * sin_phi * current[1]
                - sin_theta * current[2]
            )

        def r_dot_current_dphi_div_sin_theta(current: tuple[NDArray, ...]) -> NDArray:
            """Phi derivative of the dot product between the r unit vector and the current,
            analytically divided by sin theta."""
            return -sin_phi * current[0] + cos_phi * current[1]

        def grad_Gr_r_dot_current(current: tuple[NDArray, ...]) -> tuple[NDArray, ...]:
            """Gradient of the product of the gradient of the Green's function and the dot product
            between the r unit vector and the current."""
            temp = [
                d2G_dr2 * r_dot_current(current),
                dG_dr * r_dot_current_dtheta(current) / r,
                dG_dr * r_dot_current_dphi_div_sin_theta(current) / r,
            ]
            # convert to Cartesian coordinates
            return surface.monitor.sph_2_car_field(temp[0], temp[1], temp[2], theta_obs, phi_obs)

        def potential_terms(
            current: tuple[NDArray, ...], const: complex
        ) -> tuple[list[complex], list[complex], list[complex]]:
            """Assemble vector potential and its derivatives."""
            r_x_c = r_x_current(current)
            pot = [const * item * G for item in current]
            curl_pot = [const * item * dG_dr for item in r_x_c]
            grad_div_pot = grad_Gr_r_dot_current(current)
            grad_div_pot = [const * item for item in grad_div_pot]
            return pot, curl_pot, grad_div_pot

        # magnetic vector potential terms
        A, curl_A, grad_div_A = potential_terms(J, MU_0)

        # electric vector potential terms
        F, curl_F, grad_div_F = potential_terms(M, epsilon)

        # assemble the electric field components (Taflove 8.24, 8.27)
        e_x_integrand, e_y_integrand, e_z_integrand = (
            i_omega * (a + grad_div_a / (wavenumber**2)) - curl_f / epsilon
            for a, grad_div_a, curl_f in zip(A, grad_div_A, curl_F)
        )

        # assemble the magnetic field components (Taflove 8.25, 8.28)
        h_x_integrand, h_y_integrand, h_z_integrand = (
            i_omega * (f + grad_div_f / (wavenumber**2)) + curl_a / MU_0
            for f, grad_div_f, curl_a in zip(F, grad_div_F, curl_A)
        )

        # integrate over the surface
        e_x = self.trapezoid(e_x_integrand, (pts[idx_u], pts[idx_v]), (idx_u, idx_v))
        e_y = self.trapezoid(e_y_integrand, (pts[idx_u], pts[idx_v]), (idx_u, idx_v))
        e_z = self.trapezoid(e_z_integrand, (pts[idx_u], pts[idx_v]), (idx_u, idx_v))
        h_x = self.trapezoid(h_x_integrand, (pts[idx_u], pts[idx_v]), (idx_u, idx_v))
        h_y = self.trapezoid(h_y_integrand, (pts[idx_u], pts[idx_v]), (idx_u, idx_v))
        h_z = self.trapezoid(h_z_integrand, (pts[idx_u], pts[idx_v]), (idx_u, idx_v))

        # observation point in the original spherical system
        _, theta_obs, phi_obs = surface.monitor.car_2_sph(x, y, z)

        # convert fields to the original spherical system
        e_r, e_theta, e_phi = surface.monitor.car_2_sph_field(e_x, e_y, e_z, theta_obs, phi_obs)
        h_r, h_theta, h_phi = surface.monitor.car_2_sph_field(h_x, h_y, h_z, theta_obs, phi_obs)

        return anp.array([e_r, e_theta, e_phi, h_r, h_theta, h_phi])
