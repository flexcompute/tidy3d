"""Objects that define how data is recorded from simulation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from pydantic import (
    Field,
    NonNegativeFloat,
    PositiveFloat,
    PositiveInt,
    field_validator,
    model_validator,
)

from tidy3d.constants import HERTZ, MICROMETER, RADIAN, SECOND, inf
from tidy3d.exceptions import SetupError, ValidationError
from tidy3d.log import log

from .apodization import ApodizationSpec
from .autograd.parallel_adjoint_bases import (
    _build_diffraction_bases_for_freq,
    _build_mode_bases,
    _build_point_field_bases,
)
from .base import Tidy3dBaseModel, cached_property
from .base_sim.monitor import AbstractMonitor
from .diffraction import (
    DIFFRACTION_POLARIZATIONS,
    bloch_vec_at_freq,
    compute_angles,
    diffraction_monitor_medium,
    diffraction_orders,
    domain_size_2d,
    reciprocal_coords,
    sim_bloch_vecs,
)
from .medium import MediumType
from .microwave.base import MicrowaveBaseModel
from .mode_spec import ModeSpec
from .types import (
    AuxField,
    Axis,
    BoxSurface,
    Coordinate,
    Direction,
    EMField,
    EMSurfaceField,
    FreqArray,
    ObsGridArray,
)
from .validators import (
    assert_plane,
    assert_volumetric,
    validate_colocated_integration,
    validate_freqs_min,
    validate_freqs_not_empty,
    validate_freqs_num_not_too_many,
)
from .viz import ARROW_ALPHA, ARROW_COLOR_MONITOR

if TYPE_CHECKING:
    from pydantic import FieldValidationInfo

    from tidy3d.compat import Self

    from .autograd.parallel_adjoint_bases import DiffractionAdjointBasis, ParallelAdjointBasis
    from .simulation import Simulation
    from .types import ArrayFloat1D, Ax, Bound, FreqBound, Size

BYTES_REAL = 4
BYTES_COMPLEX = 8
WARN_NUM_FREQS = 2000
WARN_NUM_MODES = 100

# Field projection windowing factor that determines field decay at the edges of surface field
# projection monitors. A value of 15 leads to a decay of < 1e-3x in field amplitude.
# This number relates directly to the standard deviation of the Gaussian function which is used
# for windowing the monitor.
WINDOW_FACTOR = 15


def _diffraction_parallel_adjoint_bases(
    monitor: PlanarMonitor,
    simulation: Simulation,
    monitor_index: int,
) -> list[ParallelAdjointBasis]:
    """Shared helper for diffraction-style parallel adjoint bases."""
    if not isinstance(monitor, DiffractionMonitor):
        raise ValueError("Parallel adjoint diffraction bases require a DiffractionMonitor.")

    medium = diffraction_monitor_medium(simulation, monitor)
    orders_x, orders_y = diffraction_orders(simulation, monitor, medium)
    if orders_x.size == 0 or orders_y.size == 0:
        return []

    size_x, size_y = domain_size_2d(simulation, monitor)
    bloch_vec_x, bloch_vec_y = sim_bloch_vecs(simulation, monitor)

    bases: list[DiffractionAdjointBasis] = []
    freqs = [float(freq) for freq in monitor.freqs]
    for freq_index, freq in enumerate(freqs):
        ux = reciprocal_coords(
            orders=orders_x,
            size=size_x,
            bloch_vec=bloch_vec_at_freq(bloch_vec_x, freq_index),
            f=freq,
            medium=medium,
        )
        uy = reciprocal_coords(
            orders=orders_y,
            size=size_y,
            bloch_vec=bloch_vec_at_freq(bloch_vec_y, freq_index),
            f=freq,
            medium=medium,
        )
        theta_vals, _ = compute_angles((ux, uy))
        order_x_index = {int(val): idx for idx, val in enumerate(orders_x)}
        order_y_index = {int(val): idx for idx, val in enumerate(orders_y)}
        bases.extend(
            _build_diffraction_bases_for_freq(
                monitor_name=monitor.name,
                monitor_index=monitor_index,
                freq=freq,
                orders_x=orders_x,
                orders_y=orders_y,
                pols=DIFFRACTION_POLARIZATIONS,
                theta_for=lambda ox,
                oy,
                theta_vals=theta_vals,
                order_x_index=order_x_index,
                order_y_index=order_y_index: float(
                    np.asarray(theta_vals[order_x_index[ox], order_y_index[oy]]).item()
                ),
            )
        )
    return bases


class Monitor(AbstractMonitor):
    """Abstract base class for monitors.

    Notes
    -----

        **Practical Advice**

        **Choosing a Monitor Type**

        - ``FluxMonitor`` — total power flow through a surface.
          Output: ``sim_data["name"].flux`` (xarray DataArray indexed by frequency).
        - ``ModeMonitor`` — decompose fields into waveguide mode amplitudes.
          Output: ``sim_data["name"].amps.sel(direction="+", mode_index=0)``.
          Size should be 3-4x the waveguide width in each transverse dimension.
        - ``FieldMonitor`` — record E/H field components in the frequency domain.
          Output: ``sim_data["name"].Ex``, ``.Ey``, etc.
        - ``FieldTimeMonitor`` — record E/H fields vs time. Useful for animations
          and time-domain decay analysis (Q-factor extraction).
        - ``DiffractionMonitor`` — grating diffraction order efficiencies.
        - ``FieldProjectionAngleMonitor`` — far-field radiation pattern via
          near-to-far-field transformation.
    """

    interval_space: tuple[Literal[1], Literal[1], Literal[1]] = Field(
        (1, 1, 1),
        title="Spatial Interval",
        description="Number of grid step intervals between monitor recordings. If equal to 1, "
        "there will be no downsampling. If greater than 1, the step will be applied, but the "
        "first and last point of the monitor grid are always included. "
        "Not all monitors support values different from 1.",
    )

    colocate: Literal[True] = Field(
        True,
        title="Colocate Fields",
        description="Defines whether fields are colocated to grid cell boundaries (i.e. to the "
        "primal grid). Can be toggled for field recording monitors and is hard-coded for other "
        "monitors depending on their specific function.",
    )

    use_colocated_integration: Literal[True] = Field(
        True,
        title="Use Colocated Integration",
        description="Whether to use colocated fields for flux, dot products, and overlap "
        "integrals. Hard-coded to ``True`` for most monitor types. Can be toggled on field "
        "and overlap monitors.",
    )

    @property
    def _to_solver_monitor(self) -> Self:
        """Monitor definition that will be used to define the field recording during the time
        stepping."""
        return self

    @abstractmethod
    def storage_size(self, num_cells: int, tmesh: ArrayFloat1D) -> int:
        """Size of monitor storage given the number of points after discretization."""

    def _storage_size_solver(self, num_cells: int, tmesh: ArrayFloat1D) -> int:
        """Size of intermediate data recorded by the monitor during a solver run."""
        return self.storage_size(num_cells=num_cells, tmesh=tmesh)

    def supports_parallel_adjoint(self) -> bool:
        """Return ``True`` if this monitor can provide parallel adjoint bases."""
        return False

    def parallel_adjoint_bases(
        self, simulation: Simulation, monitor_index: int
    ) -> list[ParallelAdjointBasis]:
        """Return parallel adjoint bases for this monitor."""
        return []


class FreqMonitor(Monitor, ABC):
    """:class:`~tidy3d.Monitor` that records data in the frequency-domain."""

    freqs: FreqArray = Field(
        title="Frequencies",
        description="Array or list of frequencies stored by the field monitor.",
        json_schema_extra={"units": HERTZ},
    )

    apodization: ApodizationSpec = Field(
        default_factory=ApodizationSpec,
        title="Apodization Specification",
        description="Sets parameters of (optional) apodization. Apodization applies a windowing "
        "function to the Fourier transform of the time-domain fields into frequency-domain ones, "
        "and can be used to truncate the beginning and/or end of the time signal, for example "
        "to eliminate the source pulse when studying the eigenmodes of a system. Note: apodization "
        "affects the normalization of the frequency-domain fields.",
    )

    _freqs_not_empty = validate_freqs_not_empty()
    _freqs_lower_bound = validate_freqs_min()
    _warn_num_freqs = validate_freqs_num_not_too_many(WARN_NUM_FREQS)

    @cached_property
    def frequency_range(self) -> FreqBound:
        """Frequency range of the array ``self.freqs``.

        Returns
        -------
        tuple[float, float]
            Minimum and maximum frequencies of the frequency array.
        """
        return (min(self.freqs), max(self.freqs))


class TimeMonitor(Monitor, ABC):
    """:class:`~tidy3d.Monitor` that records data in the time-domain."""

    start: NonNegativeFloat = Field(
        0.0,
        title="Start Time",
        description="Time at which to start monitor recording.",
        json_schema_extra={"units": SECOND},
    )

    stop: NonNegativeFloat | None = Field(
        None,
        title="Stop Time",
        description="Time at which to stop monitor recording.  "
        "If not specified, record until end of simulation.",
        json_schema_extra={"units": SECOND},
    )

    interval: PositiveInt | None = Field(
        None,
        title="Time Interval",
        description="Sampling rate of the monitor: number of time steps between each measurement. "
        "Set ``interval`` to 1 for the highest possible resolution in time. "
        "Higher integer values downsample the data by measuring every ``interval`` time steps. "
        "This can be useful for reducing data storage as needed by the application.",
    )

    @model_validator(mode="after")
    def _warn_interval_default(self) -> Self:
        """If all defaults used for time sampler, warn and set ``interval=1`` internally."""
        val = self.interval

        if val is None:
            start = self.start
            stop = self.stop
            if start == 0.0 and stop is None:
                log.warning(
                    "The monitor 'interval' field was left as its default value, "
                    "which will set it to 1 internally. "
                    "A value of 1 means that the data will be sampled at every time step, "
                    "which may potentially produce more data than desired, "
                    "depending on the use case. "
                    "To reduce data storage, one may downsample the data by setting 'interval > 1'"
                    " or by choosing alternative 'start' and 'stop' values for the time sampling. "
                    "If you intended to use the highest resolution time sampling, "
                    "you may suppress this warning "
                    "by explicitly setting 'interval=1' in the monitor."
                )

            # set 'interval = 1' for backwards compatibility
            object.__setattr__(self, "interval", 1)

        return self

    @model_validator(mode="after")
    def stop_greater_than_start(self) -> Self:
        """Ensure sure stop is greater than or equal to start."""
        stop = self.stop
        start = self.start
        if stop and stop < start:
            raise SetupError("Monitor start time is greater than stop time.")
        return self

    def time_inds(self, tmesh: ArrayFloat1D) -> tuple[int, int]:
        """Compute the starting and stopping index of the monitor in a given discrete time mesh."""

        tmesh = np.array(tmesh)
        tind_beg, tind_end = (0, 0)

        if tmesh.size == 0:
            return (tind_beg, tind_end)

        # If monitor.stop is None, record until the end
        t_stop = self.stop
        if t_stop is None:
            tind_end = int(tmesh.size)
            t_stop = tmesh[-1]
        else:
            tend = np.nonzero(tmesh <= t_stop)[0]
            if tend.size > 0:
                tind_end = int(tend[-1] + 1)

        # Step to compare to in order to handle t_start = t_stop
        dt = 1e-20 if np.array(tmesh).size < 2 else tmesh[1] - tmesh[0]
        # If equal start and stopping time, record one time step
        if np.abs(self.start - t_stop) < dt and self.start <= tmesh[-1]:
            tind_beg = max(tind_end - 1, 0)
        else:
            tbeg = np.nonzero(tmesh[:tind_end] >= self.start)[0]
            tind_beg = tbeg[0] if tbeg.size > 0 else tind_end
        return (tind_beg, tind_end)

    def num_steps(self, tmesh: ArrayFloat1D) -> int:
        """Compute number of time steps for a time monitor."""

        tind_beg, tind_end = self.time_inds(tmesh)
        return int((tind_end - tind_beg) / self.interval)


class AbstractFieldMonitor(Monitor, ABC):
    """:class:`~tidy3d.Monitor` that records electromagnetic field data as a function of x,y,z."""

    fields: tuple[EMField, ...] = Field(
        ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"],
        title="Field Components",
        description="Collection of field components to store in the monitor.",
    )

    interval_space: tuple[PositiveInt, PositiveInt, PositiveInt] = Field(
        (1, 1, 1),
        title="Spatial Interval",
        description="Number of grid step intervals between monitor recordings. If equal to 1, "
        "there will be no downsampling. If greater than 1, the step will be applied, but the "
        "first and last point of the monitor grid are always included.",
    )

    colocate: bool = Field(
        True,
        title="Colocate Fields",
        description="Toggle whether fields should be colocated to grid cell boundaries (i.e. "
        "primal grid nodes).",
    )

    use_colocated_integration: bool = Field(
        True,
        title="Use Colocated Integration",
        description="Only takes effect when ``colocate=False``. If ``True``, flux, dot "
        "products, and overlap integrals still use fields interpolated to grid cell "
        "boundaries (colocated), even though the field data is stored at native Yee grid "
        "positions. Experimental feature that can give improved accuracy by avoiding "
        "interpolation of fields to Yee cell positions for integration.",
    )

    _colocated_integration_validator = validate_colocated_integration()

    def _storage_size_solver(self, num_cells: int, tmesh: ArrayFloat1D) -> int:
        """Size of intermediate data recorded by the monitor during a solver run."""
        final_data_size = self.storage_size(num_cells=num_cells, tmesh=tmesh)
        if len(self.fields) == 0:
            return 0

        # internally solver stores all E components if any one is requested, and same for H
        field_components_factor = 0
        if any(comp[0] == "E" for comp in self.fields):
            field_components_factor += 3
        if any(comp[0] == "H" for comp in self.fields):
            field_components_factor += 3

        # take out the stored field components factor and use the solver factor instead
        solver_data_size = final_data_size / len(self.fields) * field_components_factor
        return solver_data_size


class AbstractAuxFieldMonitor(Monitor, ABC):
    """:class:`.Monitor` that records auxiliary fields as a function of x,y,z.

    Notes
    -----
        Auxiliary fields are used in certain nonlinear material models.
        :class:`.TwoPhotonAbsorption` uses `Nfx`, `Nfy`, and `Nfz` for the
        free-carrier density.
    """

    fields: tuple[AuxField, ...] = Field(
        (),
        title="Aux Field Components",
        description="Collection of auxiliary field components to store in the monitor. "
        "Auxiliary fields which are not present in the simulation will be zero.",
    )

    interval_space: tuple[PositiveInt, PositiveInt, PositiveInt] = Field(
        (1, 1, 1),
        title="Spatial Interval",
        description="Number of grid step intervals between monitor recordings. If equal to 1, "
        "there will be no downsampling. If greater than 1, the step will be applied, but the "
        "first and last point of the monitor grid are always included.",
    )

    colocate: bool = Field(
        True,
        title="Colocate Fields",
        description="Toggle whether fields should be colocated to grid cell boundaries (i.e. "
        "primal grid nodes).",
    )

    def _storage_size_solver(self, num_cells: int, tmesh: ArrayFloat1D) -> int:
        """Size of intermediate data recorded by the monitor during a solver run."""
        final_data_size = self.storage_size(num_cells=num_cells, tmesh=tmesh)
        if len(self.fields) == 0:
            return 0

        # internally solver stores all aux field components
        field_components_factor = 3

        # take out the stored field components factor and use the solver factor instead
        solver_data_size = final_data_size / len(self.fields) * field_components_factor
        return solver_data_size


class PlanarMonitor(Monitor, ABC):
    """:class:`~tidy3d.Monitor` that has a planar geometry."""

    _plane_validator = assert_plane()

    @cached_property
    def normal_axis(self) -> Axis:
        """Axis normal to the monitor's plane."""
        return self.size.index(0.0)


class AbstractOverlapMonitor(PlanarMonitor, FreqMonitor):
    """:class:`~tidy3d.Monitor` that projects fields onto a specified basis and stores overlap amplitudes.

    This base is shared by ModeMonitor and Gaussian-overlap monitors.
    """

    store_fields_direction: Direction | None = Field(
        None,
        title="Store Fields",
        description="Propagation direction for the field profiles stored from overlap calculation.",
    )

    colocate: bool = Field(
        True,
        title="Colocate Fields",
        description="Toggle whether fields should be colocated to grid cell boundaries (i.e. primal grid nodes).",
    )

    use_colocated_integration: bool = Field(
        True,
        title="Use Colocated Integration",
        description="Only takes effect when ``colocate=False``. If ``True``, dot products "
        "and overlap integrals still use fields interpolated to grid cell boundaries "
        "(colocated), even though the field data is stored at native Yee grid positions. "
        "Experimental feature that can give improved accuracy by avoiding interpolation of "
        "fields to Yee cell positions for integration.",
    )

    _colocated_integration_validator = validate_colocated_integration()

    conjugated_dot_product: bool = Field(
        True,
        title="Conjugated Dot Product",
        description="Use conjugated or non-conjugated dot product for overlap/decomposition.",
    )

    _draw_overlap_arrows: bool = True
    """Whether to draw arrows in AbstractOverlapMonitor.plot(). Subclasses override to False."""

    def plot(
        self,
        x: float | None = None,
        y: float | None = None,
        z: float | None = None,
        ax: Ax = None,
        **patch_kwargs: dict[str, Any],
    ) -> Ax:
        """Plot this overlap monitor with an arrow indicating propagation direction."""
        ax = super().plot(x=x, y=y, z=z, ax=ax, **patch_kwargs)

        if not self._draw_overlap_arrows:
            return ax

        kwargs_alpha = patch_kwargs.get("alpha")
        arrow_alpha = ARROW_ALPHA if kwargs_alpha is None else kwargs_alpha

        ax = self._plot_arrow(
            x=x,
            y=y,
            z=z,
            ax=ax,
            direction=self._dir_arrow,
            bend_radius=None,
            bend_axis=None,
            color=ARROW_COLOR_MONITOR,
            alpha=arrow_alpha,
            both_dirs=True,
        )
        return ax

    @cached_property
    def _dir_arrow(self) -> tuple[float, float, float]:
        """Monitor direction normal vector in cartesian coordinates."""
        theta, phi = self._angles
        dx = np.cos(phi) * np.sin(theta)
        dy = np.sin(phi) * np.sin(theta)
        dz = np.cos(theta)
        return self.unpop_axis(dz, (dx, dy), axis=self.normal_axis)

    @property
    def _angles(self) -> tuple[float, float]:
        """Angle tuple (theta, phi) in radians. Children override to supply values."""
        return (0.0, 0.0)

    def _storage_size_solver(self, num_cells: int, tmesh: ArrayFloat1D) -> int:
        """Default size of intermediate data; store all fields on plane for overlap."""
        num_sample = len(getattr(self, "freqs", [0]))
        return BYTES_COMPLEX * num_cells * num_sample * 6


class AbstractModeMonitor(AbstractOverlapMonitor):
    """:class:`~tidy3d.Monitor` that records mode-related data."""

    _draw_overlap_arrows: bool = False  # AbstractModeMonitor.plot() draws its own arrows

    mode_spec: ModeSpec = Field(
        default_factory=ModeSpec,
        title="Mode Specification",
        description="Parameters to feed to mode solver which determine modes measured by monitor.",
    )

    store_fields_direction: Direction | None = Field(
        None,
        title="Store Fields",
        description="Propagation direction for the mode field profiles stored from mode solving.",
    )

    colocate: bool = Field(
        True,
        title="Colocate Fields",
        description="Toggle whether fields should be colocated to grid cell boundaries (i.e. "
        "primal grid nodes).",
    )

    conjugated_dot_product: bool = Field(
        True,
        title="Conjugated Dot Product",
        description="Use conjugated or non-conjugated dot product for mode decomposition.",
    )

    def plot(
        self,
        x: float | None = None,
        y: float | None = None,
        z: float | None = None,
        ax: Ax = None,
        **patch_kwargs: Any,
    ) -> Ax:
        """Plot this monitor."""
        # call the monitor.plot() function first
        ax = super().plot(x=x, y=y, z=z, ax=ax, **patch_kwargs)

        kwargs_alpha = patch_kwargs.get("alpha")
        arrow_alpha = ARROW_ALPHA if kwargs_alpha is None else kwargs_alpha

        bend_radius = self.mode_spec.bend_radius
        # Curvature has to be reversed because of ploting coordinates
        if (self.normal_axis, self._bend_axis) in [(1, 2), (2, 0), (2, 1)]:
            bend_radius = -bend_radius

        # and then add an arrow using the direction comuputed from `_dir_arrow`.
        ax = self._plot_arrow(
            x=x,
            y=y,
            z=z,
            ax=ax,
            direction=self._dir_arrow,
            bend_radius=bend_radius,
            bend_axis=self._bend_axis,
            color=ARROW_COLOR_MONITOR,
            alpha=arrow_alpha,
            both_dirs=True,
        )
        return ax

    @cached_property
    def _angles(self) -> tuple[float, float]:
        return (self.mode_spec.angle_theta, self.mode_spec.angle_phi)

    @cached_property
    def _bend_axis(self) -> Axis:
        if self.mode_spec.bend_radius is None:
            return None
        in_plane = [0, 0]
        in_plane[self.mode_spec.bend_axis] = 1
        direction = self.unpop_axis(0, in_plane, axis=self.normal_axis)
        return direction.index(1)

    @field_validator("mode_spec")
    @classmethod
    def _warn_num_modes(
        cls: type[ModeMonitor], val: ModeSpec, info: FieldValidationInfo
    ) -> ModeSpec:
        """Warn if number of modes is too large."""
        if isinstance(val.num_modes, int) and val.num_modes > WARN_NUM_MODES:
            log.warning(
                f"A large number ({val.num_modes}) of modes requested in monitor "
                f"'{info.field_name}'. This can lead to solver slow-down and increased cost. "
                "Consider decreasing the number of modes and using 'ModeSpec.target_neff' "
                "to target the modes of interest. This may become a hard limit in future "
                "Tidy3D versions.",
                custom_loc=["mode_spec", "num_modes"],
            )
        return val

    @property
    def _stored_freqs(self) -> list[float]:
        """Return actually stored frequencies of the data."""
        return self.mode_spec._sampling_freqs_mode_solver_data(freqs=self.freqs)

    def _storage_size_solver(self, num_cells: int, tmesh: ArrayFloat1D) -> int:
        """Size of intermediate data recorded by the monitor during a solver run."""
        # Need to store all fields on the mode surface
        bytes_single = (
            BYTES_COMPLEX * num_cells * len(self._stored_freqs) * self.mode_spec.num_modes * 6
        )
        if self.mode_spec.precision == "double":
            return 2 * bytes_single
        return bytes_single


class AbstractGaussianOverlapMonitor(AbstractOverlapMonitor):
    """:class:`~tidy3d.Monitor` that records amplitudes from decomposition onto a Gaussian-like beam.

    Common fields and behavior shared by GaussianOverlapMonitor and
    AstigmaticGaussianOverlapMonitor.
    """

    angle_theta: float = Field(
        0.0,
        title="Polar Angle",
        description="Polar angle of propagation direction.",
        json_schema_extra={"units": RADIAN},
    )

    angle_phi: float = Field(
        0.0,
        title="Azimuth Angle",
        description="Azimuth angle of propagation direction.",
        json_schema_extra={"units": RADIAN},
    )

    pol_angle: float = Field(
        0,
        title="Polarization Angle",
        description="Specifies the angle between the electric field polarization of the "
        "source and the plane defined by the injection axis and the propagation axis (rad). "
        "``pol_angle=0`` (default) specifies P polarization, "
        "while ``pol_angle=np.pi/2`` specifies S polarization. "
        "At normal incidence when S and P are undefined, ``pol_angle=0`` defines: "
        "- ``Ey`` polarization for propagation along ``x``."
        "- ``Ex`` polarization for propagation along ``y``."
        "- ``Ex`` polarization for propagation along ``z``.",
        json_schema_extra={"units": RADIAN},
    )

    @property
    def _angles(self) -> tuple[float, float]:
        return (self.angle_theta, self.angle_phi)

    def storage_size(self, num_cells: int, tmesh: ArrayFloat1D) -> int:
        """Size of monitor storage given the number of points after discretization."""
        # store complex amplitudes for +/- directions
        num_dirs = 2
        return BYTES_COMPLEX * len(self.freqs) * num_dirs


class GaussianOverlapMonitor(AbstractGaussianOverlapMonitor):
    """:class:`~tidy3d.Monitor` that records amplitudes from decomposition onto a Gaussian beam.

    Example
    -------
    >>> gauss = GaussianOverlapMonitor(
    ...     size=(0, 3, 3),
    ...     freqs=[2e14],
    ...     pol_angle=np.pi / 2,
    ...     waist_radius=1.0,
    ...     name="gaussian_monitor",
    ... )

    Notes
    --------
        If one wants the focus 'in front' of the monitor, a negative value of ``waist_distance``
        is needed. See also :class:`.GaussianBeam`.
    """

    waist_radius: PositiveFloat = Field(
        1.0,
        title="Waist Radius",
        description="Radius of the beam at the waist.",
        json_schema_extra={"units": MICROMETER},
    )

    waist_distance: float = Field(
        0.0,
        title="Waist Distance",
        description="Distance from the beam waist along the propagation direction. "
        "A positive value places the waist behind the monitor plane (toward the negative normal axis). "
        "A negative value places the waist in front of the monitor plane. "
        "For an angled beam, the distance is measured along the rotated propagation direction.",
        json_schema_extra={"units": MICROMETER},
    )


class AstigmaticGaussianOverlapMonitor(AbstractGaussianOverlapMonitor):
    """:class:`~tidy3d.Monitor` that records amplitudes from decomposition onto an astigmatic Gaussian beam.

    Notes
    -----
    The simple astigmatic Gaussian distribution allows
    both an elliptical intensity profile and different waist locations for the two principal axes
    of the ellipse. When equal waist sizes and equal waist distances are specified in the two
    directions, this monitor becomes equivalent to :class:`GaussianOverlapMonitor`.

    This class implements the simple astigmatic Gaussian beam described in [1]_.

    **References**:

    .. [1] Kochkina et al., Applied Optics, vol. 52, issue 24, 2013.

    Example
    -------
    >>> gauss = AstigmaticGaussianOverlapMonitor(
    ...     size=(0,3,3),
    ...     pol_angle=np.pi / 2,
    ...     waist_sizes=(1.0, 2.0),
    ...     waist_distances = (3.0, 4.0),
    ...     freqs=[2e14],
    ...     name="astigmatic_gaussian_monitor",
    ... )
    """

    waist_sizes: tuple[PositiveFloat, PositiveFloat] = Field(
        (1.0, 1.0),
        title="Waist sizes",
        description="Size of the beam at the waist in the local x and y directions.",
        json_schema_extra={"units": MICROMETER},
    )

    waist_distances: tuple[float, float] = Field(
        (0.0, 0.0),
        title="Waist distances",
        description="Distance to the beam waist along the propagation direction "
        "for the waist sizes in the local x and y directions. "
        "Positive values place the waist behind the monitor plane (toward the negative normal axis); "
        "negative values place the waist in front of the monitor plane.",
        json_schema_extra={"units": MICROMETER},
    )


class FieldMonitor(AbstractFieldMonitor, FreqMonitor):
    """:class:`~tidy3d.Monitor` that records electromagnetic fields in the frequency domain.

    Notes
    -----

        :class:`FieldMonitor` objects operate by running a discrete Fourier transform of the fields at a given set of
        frequencies to perform the calculation “in-place” with the time stepping. :class:`FieldMonitor`  objects are
        useful for investigating the steady-state field distribution in 2D and 3D regions of the simulation.


    Example
    -------
    >>> monitor = FieldMonitor(
    ...     center=(1,2,3),
    ...     size=(2,2,2),
    ...     fields=['Hx'],
    ...     freqs=[250e12, 300e12],
    ...     name='steady_state_monitor',
    ...     colocate=True)


    See Also
    --------

    **Notebooks**

    * `Quickstart <../../notebooks/StartHere.html>`_: Usage in a basic simulation flow.

    **Lectures**

    * `Introduction to FDTD Simulation <https://www.flexcompute.com/fdtd101/Lecture-1-Introduction-to-FDTD-Simulation/#presentation-slides>`_: Usage in a basic simulation flow.

    """

    def storage_size(self, num_cells: int, tmesh: ArrayFloat1D) -> int:
        """Size of monitor storage given the number of points after discretization."""
        # stores 1 complex number per grid cell, per frequency, per field
        return BYTES_COMPLEX * num_cells * len(self.freqs) * len(self.fields)

    def supports_parallel_adjoint(self) -> bool:
        """Return ``True`` when the field monitor is a single-point probe."""
        return len(self.zero_dims) == 3

    def parallel_adjoint_bases(
        self, simulation: Simulation, monitor_index: int
    ) -> list[ParallelAdjointBasis]:
        """Return parallel adjoint bases for single-point field monitors."""
        if not self.supports_parallel_adjoint():
            return []
        if not self.colocate:
            raise ValueError("Parallel adjoint point sources require colocated field monitors.")
        component_freqs = {component: list(self.freqs) for component in self.fields}
        return _build_point_field_bases(
            component_freqs=component_freqs,
            monitor_name=self.name,
            monitor_index=monitor_index,
            data_path_prefix=("data", monitor_index),
        )


class FieldTimeMonitor(AbstractFieldMonitor, TimeMonitor):
    """:class:`~tidy3d.Monitor` that records electromagnetic fields in the time domain.

    Notes
    -----

        :class:`FieldTimeMonitor` objects are best used to monitor the time dependence of the fields at a single
        point, but they can also be used to create “animations” of the field pattern evolution.

        To create an animation, we need to capture the frames at different time instances of the simulation. This can
        be done by using a :class:`FieldTimeMonitor`. Usually a FDTD simulation contains a large number of time steps
        and grid points. Recording the field at every time step and grid point will result in a large dataset. For
        the purpose of making animations, this is usually unnecessary.


    Example
    -------
    >>> monitor = FieldTimeMonitor(
    ...     center=(1,2,3),
    ...     size=(2,2,2),
    ...     fields=['Hx'],
    ...     start=1e-13,
    ...     stop=5e-13,
    ...     interval=2,
    ...     colocate=True,
    ...     name='movie_monitor')


    See Also
    --------

    **Notebooks**
        * `First walkthrough <../../notebooks/Simulation.html>`_: Usage in a basic simulation flow.
        * `Creating FDTD animations <../../notebooks/AnimationTutorial.html>`_.

    """

    def storage_size(self, num_cells: int, tmesh: ArrayFloat1D) -> int:
        """Size of monitor storage given the number of points after discretization."""
        # stores 1 real number per grid cell, per time step, per field
        num_steps = self.num_steps(tmesh)
        return BYTES_REAL * num_steps * num_cells * len(self.fields)


class AuxFieldTimeMonitor(AbstractAuxFieldMonitor, TimeMonitor):
    """:class:`.Monitor` that records auxiliary fields in the time domain.

    Notes
    -----
        Auxiliary fields are used in certain nonlinear material models.
        :class:`.TwoPhotonAbsorption` uses `Nfx`, `Nfy`, and `Nfz` for the
        free-carrier density.

    Example
    -------
    >>> monitor = AuxFieldTimeMonitor(
    ...     center=(1,2,3),
    ...     size=(0,0,0),
    ...     fields=['Nfx'],
    ...     start=1e-13,
    ...     stop=5e-13,
    ...     interval=2,
    ...     colocate=True,
    ...     name='aux_monitor')
    """

    def storage_size(self, num_cells: int, tmesh: ArrayFloat1D) -> int:
        """Size of monitor storage given the number of points after discretization."""
        # stores 1 real number per grid cell, per time step, per field
        num_steps = self.num_steps(tmesh)
        return BYTES_REAL * num_steps * num_cells * len(self.fields)


class AbstractMediumPropertyMonitor(FreqMonitor, ABC):
    """:class:`~tidy3d.Monitor` that records material properties in the frequency domain."""

    colocate: Literal[False] = Field(
        False,
        title="Colocate Fields",
        description="Colocation turned off, since colocated medium property values do not have a "
        "physical meaning - they do not correspond to the subpixel-averaged ones.",
    )

    interval_space: tuple[PositiveInt, PositiveInt, PositiveInt] = Field(
        (1, 1, 1),
        title="Spatial Interval",
        description="Number of grid step intervals between monitor recordings. If equal to 1, "
        "there will be no downsampling. If greater than 1, the step will be applied, but the "
        "first and last point of the monitor grid are always included.",
    )

    apodization: ApodizationSpec = Field(
        default_factory=ApodizationSpec,
        title="Apodization Specification",
        description="This field is ignored in this monitor.",
    )


class MediumMonitor(AbstractMediumPropertyMonitor):
    """:class:`~tidy3d.Monitor` that records the diagonal components of the complex-valued relative
    permittivity and permeability tensor in the frequency domain. The recorded data has the same shape as a
    :class:`.FieldMonitor` of the same geometry: the permittivity and permeability values are saved at the
    Yee grid locations, and can be interpolated to any point inside the monitor.

    Notes
    -----

        If 2D materials are present, then the permittivity values correspond to the
        volumetric equivalent of the 2D materials.

        .. TODO add links to relevant areas

    Example
    -------
    >>> monitor = MediumMonitor(
    ...     center=(1,2,3),
    ...     size=(2,2,2),
    ...     freqs=[250e12, 300e12],
    ...     name='medium_monitor')
    """

    def storage_size(self, num_cells: int, tmesh: ArrayFloat1D) -> int:
        """Size of monitor storage given the number of points after discretization."""
        # stores 6 complex number per grid cell, per frequency
        return BYTES_COMPLEX * num_cells * len(self.freqs) * 6


class PermittivityMonitor(AbstractMediumPropertyMonitor):
    """:class:`~tidy3d.Monitor` that records the diagonal components of the complex-valued relative
    permittivity tensor in the frequency domain. The recorded data has the same shape as a
    :class:`.FieldMonitor` of the same geometry: the permittivity values are saved at the
    Yee grid locations, and can be interpolated to any point inside the monitor.

    Notes
    -----

        If 2D materials are present, then the permittivity values correspond to the
        volumetric equivalent of the 2D materials.

        .. TODO add links to relevant areas

    Example
    -------
    >>> monitor = PermittivityMonitor(
    ...     center=(1,2,3),
    ...     size=(2,2,2),
    ...     freqs=[250e12, 300e12],
    ...     name='eps_monitor')
    """

    def storage_size(self, num_cells: int, tmesh: ArrayFloat1D) -> int:
        """Size of monitor storage given the number of points after discretization."""
        # stores 3 complex number per grid cell, per frequency
        return BYTES_COMPLEX * num_cells * len(self.freqs) * 3


class SurfaceIntegrationMonitor(Monitor, ABC):
    """Abstract class for monitors that perform surface integrals during the solver run, as in
    flux and near to far transformations."""

    normal_dir: Direction | None = Field(
        None,
        title="Normal Vector Orientation",
        description="Direction of the surface monitor's normal vector w.r.t. "
        "the positive x, y or z unit vectors. Must be one of ``'+'`` or ``'-'``. "
        "Applies to surface monitors only, and defaults to ``'+'`` if not provided.",
    )

    exclude_surfaces: tuple[BoxSurface, ...] | None = Field(
        None,
        title="Excluded Surfaces",
        description="Surfaces to exclude in the integration, if a volume monitor.",
    )

    @property
    def integration_surfaces(self) -> list[SurfaceIntegrationMonitor]:
        """Surfaces of the monitor where fields will be recorded for subsequent integration."""
        if self.size.count(0.0) == 0:
            return self.surfaces_with_exclusion(**self.model_dump())
        return [self]

    @model_validator(mode="after")
    def normal_dir_exists_for_surface(self) -> Self:
        """If the monitor is a surface, set default ``normal_dir`` if not provided.
        If the monitor is a box, warn that ``normal_dir`` is relevant only for surfaces."""
        if self.size.count(0.0) != 1:
            if self.normal_dir is not None:
                log.warning(
                    "The ``normal_dir`` field is relevant only for surface monitors "
                    f"and will be ignored for monitor {self.name}, which is a box."
                )
        else:
            if self.normal_dir is None:
                object.__setattr__(self, "normal_dir", "+")
        return self

    @model_validator(mode="after")
    def check_excluded_surfaces(self) -> Self:
        """Error if ``exclude_surfaces`` is provided for a surface monitor."""
        exclude_surfaces = self.exclude_surfaces
        if exclude_surfaces is None:
            return self
        name = self.name
        size = self.size
        if size.count(0.0) > 0:
            raise SetupError(
                f"Can't specify ``exclude_surfaces`` for surface monitor {name}; "
                "valid for box monitors only."
            )
        return self

    def _storage_size_solver(self, num_cells: int, tmesh: ArrayFloat1D) -> int:
        """Size of intermediate data recorded by the monitor during a solver run."""
        # Need to store all fields on the integration surface. Frequency-domain monitors store at
        # all frequencies, time domain at the current time step only.
        num_sample = len(getattr(self, "freqs", [0]))
        return BYTES_COMPLEX * num_cells * num_sample * 6


class AbstractFluxMonitor(SurfaceIntegrationMonitor, ABC):
    """:class:`~tidy3d.Monitor` that records flux during the solver run."""


class FluxMonitor(AbstractFluxMonitor, FreqMonitor):
    """:class:`~tidy3d.Monitor` that records power flux in the frequency domain.

    Notes
    -----

        If the monitor geometry is a 2D box, the total flux through this plane is returned, with a
        positive sign corresponding to power flow in the positive direction along the axis normal to
        the plane. If the geometry is a 3D box, the total power coming out of the box is returned by
        integrating the flux over all box surfaces (except the ones defined in ``exclude_surfaces``).

        **Practical Advice**

        If measured transmission exceeds 1.0 or is negative, verify that the monitor normal axis
        aligns with the expected power flow direction. A common mistake is placing a flux monitor
        with its normal pointing opposite to the propagation direction.

        **Extracting transmission**::

            sim_data = web.run(sim, task_name="my_sim")
            flux = sim_data["flux_monitor"].flux  # xarray DataArray indexed by frequency

            # If source injects 1W (ModeSource at center frequency), flux IS the transmission
            T = flux.values

            # For normalization against a reference simulation:
            # T = flux_device / flux_reference

    Example
    -------
    >>> monitor = FluxMonitor(
    ...     center=(1,2,3),
    ...     size=(2,2,0),
    ...     freqs=[200e12, 210e12],
    ...     name='flux_monitor')

    See Also
    --------

    **Notebooks**

    * `THz integrated demultiplexer/filter based on a ring resonator <../../notebooks/THzDemultiplexerFilter.html>`_
    """

    def storage_size(self, num_cells: int, tmesh: ArrayFloat1D) -> int:
        """Size of monitor storage given the number of points after discretization."""
        # stores 1 real number per frequency
        return BYTES_REAL * len(self.freqs)


class FluxTimeMonitor(AbstractFluxMonitor, TimeMonitor):
    """:class:`~tidy3d.Monitor` that records power flux in the time domain.

    Notes
    -----

        If the monitor geometry is a 2D box, the total flux through this plane is returned, with a
        positive sign corresponding to power flow in the positive direction along the axis normal to
        the plane. If the geometry is a 3D box, the total power coming out of the box is returned by
        integrating the flux over all box surfaces (except the ones defined in ``exclude_surfaces``).

    Example
    -------
    >>> monitor = FluxTimeMonitor(
    ...     center=(1,2,3),
    ...     size=(2,2,0),
    ...     start=1e-13,
    ...     stop=5e-13,
    ...     interval=2,
    ...     name='flux_vs_time')
    """

    def storage_size(self, num_cells: int, tmesh: ArrayFloat1D) -> int:
        """Size of monitor storage given the number of points after discretization."""
        # stores 1 real number per time step
        num_steps = self.num_steps(tmesh)
        return BYTES_REAL * num_steps


class ModeMonitor(AbstractModeMonitor):
    """:class:`~tidy3d.Monitor` that records amplitudes from modal decomposition of fields on plane.

    Notes
    ------

        The fields recorded by frequency monitors (and hence also mode monitors) are automatically
        normalized by the power amplitude spectrum of the source. For multiple sources, the user can
        select which source to use for the normalization too.

        We can also use the mode amplitudes recorded in the mode monitor to reveal the decomposition
        of the radiated power into forward- and backward-propagating modes, respectively.

        **Practical Advice**

        For reliable mode decomposition, place mode monitors in straight waveguide sections where the mode
        profile is well-defined. The monitor should be large enough to capture the full mode profile including
        evanescent tails — a typical sizing is 3-4x the waveguide width in each transverse dimension.

        **Extracting mode amplitudes**::

            amps = sim_data["mode_monitor"].amps

            # Forward-propagating power in fundamental mode
            T_mode0 = np.abs(amps.sel(direction="+", mode_index=0).values) ** 2

            # Backward-propagating (reflection)
            R_mode0 = np.abs(amps.sel(direction="-", mode_index=0).values) ** 2

        .. TODO give an example of how to extract the data from this mode.

        .. TODO add derivation in the notebook.

        .. TODO add link to method

        .. TODO add links to notebooks correspondingly

    Example
    -------
    >>> mode_spec = ModeSpec(num_modes=3)
    >>> monitor = ModeMonitor(
    ...     center=(1,2,3),
    ...     size=(2,2,0),
    ...     freqs=[200e12, 210e12],
    ...     mode_spec=mode_spec,
    ...     name='mode_monitor')

    See Also
    --------

    **Notebooks**:
        * `ModalSourcesMonitors <../../notebooks/ModalSourcesMonitors.html>`_
    """

    @property
    def _to_solver_monitor(self) -> Self:
        """Monitor definition that will be used to define the field recording during the time
        stepping."""
        return self.updated_copy(colocate=False)

    @property
    def _stored_freqs(self) -> list[float]:
        """Return actually stored frequencies of the data."""
        # always stored at original frequencies, no matter whether interp_spec is used
        return self.freqs

    def storage_size(self, num_cells: int, tmesh: int) -> int:
        """Size of monitor storage given the number of points after discretization."""
        amps_size = 3 * BYTES_COMPLEX * len(self._stored_freqs) * self.mode_spec.num_modes
        fields_size = 0
        if self.store_fields_direction is not None:
            fields_size = (
                6 * BYTES_COMPLEX * num_cells * len(self._stored_freqs) * self.mode_spec.num_modes
            )
            if self.mode_spec.precision == "double":
                fields_size *= 2
        return amps_size + fields_size

    def supports_parallel_adjoint(self) -> bool:
        """Return ``True`` for mode monitor amplitude adjoints."""
        return True

    def parallel_adjoint_bases(
        self, simulation: Simulation, monitor_index: int
    ) -> list[ParallelAdjointBasis]:
        """Return parallel adjoint bases for mode monitor amplitudes."""
        freqs = [float(freq) for freq in self._stored_freqs]
        directions = ("+", "-")
        mode_indices = range(self.mode_spec.num_modes)
        return _build_mode_bases(
            freqs=freqs,
            directions=directions,
            mode_indices=mode_indices,
            monitor_name=self.name,
            monitor_index=monitor_index,
            data_path=("data", monitor_index, "amps"),
        )


class ModeSolverMonitor(AbstractModeMonitor):
    """:class:`~tidy3d.Monitor` that stores the mode field profiles returned by the mode solver in the
    monitor plane.

    Example
    -------
    >>> mode_spec = ModeSpec(num_modes=3)
    >>> monitor = ModeSolverMonitor(
    ...     center=(1,2,3),
    ...     size=(2,2,0),
    ...     freqs=[200e12, 210e12],
    ...     mode_spec=mode_spec,
    ...     name='mode_monitor')
    """

    direction: Direction = Field(
        "+",
        title="Propagation Direction",
        description="Direction of waveguide mode propagation along the axis defined by its normal "
        "dimension.",
    )

    fields: tuple[EMField, ...] = Field(
        ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"),
        title="Field Components",
        description="Collection of field components to store in the monitor. Note that some "
        "methods like ``flux``, ``dot`` require all tangential field components, while others "
        "like ``mode_area`` require all E-field components.",
    )

    @property
    def _stored_freqs(self) -> list[float]:
        """Return actually stored frequencies of the data."""
        return self.mode_spec._sampling_freqs_mode_solver_data(freqs=self.freqs)

    @model_validator(mode="after")
    def set_store_fields(self) -> Self:
        """Ensure 'store_fields_direction' is compatible with 'direction'."""
        store_fields_direction = self.store_fields_direction
        direction = self.direction
        if store_fields_direction is None:
            object.__setattr__(self, "store_fields_direction", direction)
        elif store_fields_direction != direction:
            self._raise_validation_error_at_loc(
                ValidationError(
                    f"The values of 'direction' ({direction}) and 'store_fields_direction' "
                    f"({store_fields_direction}) must be equal."
                ),
                "store_fields_direction",
            )
        return self

    def storage_size(self, num_cells: int, tmesh: int) -> int:
        """Size of monitor storage given the number of points after discretization."""
        bytes_single = (
            6 * BYTES_COMPLEX * num_cells * len(self._stored_freqs) * self.mode_spec.num_modes
        )
        if self.mode_spec.precision == "double":
            return 2 * bytes_single
        return bytes_single


class FieldProjectionSurface(Tidy3dBaseModel):
    """Data structure to store surface monitors where near fields are recorded for field projections.

    Notes
    -----
    .. TODO add example and derivation, and more relevant links.

    See Also
    --------
    **Notebooks**:
        * `Performing near field to far field projections <../../notebooks/FieldProjections.html>`_
    """

    monitor: FieldMonitor = Field(
        title="Field Monitor",
        description=":class:`.FieldMonitor` on which near fields will be sampled and integrated.",
    )

    normal_dir: Direction = Field(
        title="Normal Vector Orientation",
        description=":class:`.Direction` of the surface monitor's normal vector w.r.t.\
 the positive x, y or z unit vectors. Must be one of '+' or '-'.",
    )

    @cached_property
    def axis(self) -> Axis:
        """Returns the :class:`.Axis` normal to this surface."""
        # assume that the monitor's axis is in the direction where the monitor is thinnest
        return self.monitor.size.index(0.0)

    @field_validator("monitor")
    @classmethod
    def is_plane(cls, val: FieldMonitor) -> FieldMonitor:
        """Ensures that the monitor is a plane, i.e., its ``size`` attribute has exactly 1 zero"""
        size = val.size
        if size.count(0.0) != 1:
            raise ValidationError(f"Monitor '{val.name}' must be planar, given size={size}")
        return val


class AbstractFieldProjectionMonitor(SurfaceIntegrationMonitor, FreqMonitor):
    """:class:`~tidy3d.Monitor` that samples electromagnetic near fields in the frequency domain
    and projects them to a given set of observation points.
    """

    custom_origin: Coordinate | None = Field(
        None,
        title="Local Origin",
        description="Local origin used for defining observation points. If ``None``, uses the "
        "monitor's center.",
        json_schema_extra={"units": MICROMETER},
    )

    far_field_approx: bool = Field(
        True,
        title="Far Field Approximation",
        description="Whether to enable the far field approximation when projecting fields. "
        "If ``True``, terms that decay as O(1/r^2) are ignored, as are the radial components "
        "of fields. Typically, this should be set to ``True`` only when the projection distance "
        "is much larger than the size of the device being modeled, and the projected points are "
        "in the far field of the device.",
    )

    interval_space: tuple[PositiveInt, PositiveInt, PositiveInt] = Field(
        (1, 1, 1),
        title="Spatial Interval",
        description="Number of grid step intervals at which near fields are recorded for "
        "projection to the far field, along each direction. If equal to 1, there will be no "
        "downsampling. If greater than 1, the step will be applied, but the first and last "
        "point of the monitor grid are always included. Using values greater than 1 can "
        "help speed up server-side far field projections with minimal accuracy loss, "
        "especially in cases where it is necessary for the grid resolution to be high for "
        "the FDTD simulation, but such a high resolution is unnecessary for the purpose of "
        "projecting the recorded near fields to the far field.",
    )

    window_size: tuple[NonNegativeFloat, NonNegativeFloat] = Field(
        (0, 0),
        title="Spatial filtering window size",
        description="Size of the transition region of the windowing function used to ensure that "
        "the recorded near fields decay to zero near the edges of the monitor. "
        "The two components refer to the two tangential directions associated with each surface. "
        "For surfaces with the normal along ``x``, the two components are (``y``, ``z``). "
        "For surfaces with the normal along ``y``, the two components are (``x``, ``z``). "
        "For surfaces with the normal along ``z``, the two components are (``x``, ``y``). "
        "Each value must be between 0 and 1, inclusive, and denotes the size of the transition "
        "region over which fields are scaled to less than a thousandth of the original amplitude, "
        "relative to half the size of the monitor in that direction. A value of 0 turns windowing "
        "off in that direction, while a value of 1 indicates that the window will be applied to "
        "the entire monitor in that direction. This field is applicable for surface monitors only, "
        "and otherwise must remain (0, 0).",
    )

    medium: MediumType | None = Field(
        None,
        title="Projection medium",
        description="Medium through which to project fields. Generally, the fields should be "
        "projected through the same medium as the one in which this monitor is placed, and "
        "this is the default behavior when ``medium=None``. A custom ``medium`` can be useful "
        "in some situations for advanced users, but we recommend trying to avoid using a "
        "non-default ``medium``.",
    )

    @model_validator(mode="after")
    def window_size_for_surface(self) -> Self:
        """Ensures that windowing is applied for surface monitors only."""
        val = self.window_size
        size = self.size
        name = self.name

        if size.count(0.0) != 1:
            if val != (0, 0):
                raise ValidationError(
                    f"A non-zero 'window_size' cannot be used for projection monitor '{name}'. "
                    "Windowing can be applied only for surface projection monitors."
                )
        return self

    @field_validator("window_size")
    @classmethod
    def window_size_leq_one(
        cls: type[AbstractFieldProjectionMonitor],
        val: tuple[float, float],
        info: FieldValidationInfo,
    ) -> tuple[float, float]:
        """Ensures that each component of the window size is less than or equal to 1."""
        if val[0] > 1 or val[1] > 1:
            raise ValidationError(
                f"Each component of 'window_size' for monitor '{info.field_name}' "
                "must be less than or equal to 1."
            )
        return val

    @property
    def projection_surfaces(self) -> tuple[FieldProjectionSurface, ...]:
        """Surfaces of the monitor where near fields will be recorded for subsequent projection."""
        surfaces = self.integration_surfaces
        return [
            FieldProjectionSurface(
                monitor=FieldMonitor(
                    center=surface.center,
                    size=surface.size,
                    freqs=self.freqs,
                    name=surface.name,
                    colocate=True,
                ),
                normal_dir=surface.normal_dir,
            )
            for surface in surfaces
        ]

    @property
    def local_origin(self) -> Coordinate:
        """Returns the local origin associated with this monitor."""
        if self.custom_origin is None:
            return self.center
        return self.custom_origin

    def window_parameters(self, custom_bounds: Bound = None) -> tuple[Size, Coordinate, Coordinate]:
        """Return the physical size of the window transition region based on the monitor's size
        and optional custom bounds (useful in case the monitor has infinite dimensions). The window
        size is returned in 3D. Also returns the coordinate where the transition region beings on
        the minus and plus side of the monitor."""

        window_size = [0, 0, 0]
        window_minus = [0, 0, 0]
        window_plus = [0, 0, 0]

        # windowing is for surface monitors only
        if self.size.count(0.0) != 1:
            return window_size, window_minus, window_plus

        _, plane_inds = self.pop_axis([0, 1, 2], axis=self.size.index(0.0))

        for i, ind in enumerate(plane_inds):
            if custom_bounds:
                size = min(self.size[ind], custom_bounds[1][ind] - custom_bounds[0][ind])
                bound_min = max(self.bounds[0][ind], custom_bounds[0][ind])
                bound_max = min(self.bounds[1][ind], custom_bounds[1][ind])
            else:
                size = self.size[ind]
                bound_min = self.bounds[0][ind]
                bound_max = self.bounds[1][ind]

            window_size[ind] = self.window_size[i] * size / 2
            window_minus[ind] = bound_min + window_size[ind]
            window_plus[ind] = bound_max - window_size[ind]

        return window_size, window_minus, window_plus

    @staticmethod
    def window_function(
        points: ArrayFloat1D,
        window_size: Size,
        window_minus: Coordinate,
        window_plus: Coordinate,
        dim: int,
    ) -> ArrayFloat1D:
        """Get the windowing function along a given direction for a given set of points."""
        rising_window = np.exp(
            -0.5
            * WINDOW_FACTOR
            * ((points[points < window_minus[dim]] - window_minus[dim]) / window_size[dim]) ** 2
        )
        falling_window = np.exp(
            -0.5
            * WINDOW_FACTOR
            * ((points[points > window_plus[dim]] - window_plus[dim]) / window_size[dim]) ** 2
        )
        window_fn = np.ones_like(points)
        window_fn[points < window_minus[dim]] = rising_window
        window_fn[points > window_plus[dim]] = falling_window
        return window_fn


class FieldProjectionAngleMonitor(AbstractFieldProjectionMonitor):
    """:class:`~tidy3d.Monitor` that samples electromagnetic near fields in the frequency domain
    and projects them at given observation angles.

    Notes
    -----

        .. TODO this needs an illustration

        **Parameters Caveats**

        The :attr:`center` and :attr:`size` parameters define
        where the monitor will be placed in order to record near fields, typically very close
        to the structure of interest. The near fields are then projected
        to far-field locations defined by :attr:`phi`, :attr:`theta`, and :attr:`proj_distance`, relative
        to the :attr:`custom_origin`.

        **Usage Caveats**

        The field projections make use of the analytical homogeneous medium Green’s function, which assumes that the
        fields are propagating in a homogeneous medium. Therefore, one should use :class:`PML` / :class:`Absorber` as
        boundary conditions in the part of the domain where fields are projected.

        .. TODO why not add equation here

        Server-side field projections will add to the monetary cost of the simulation. However, typically the far field
        projections have a very small computation cost compared to the FDTD simulation itself, so the increase in monetary
        cost should be negligibly small in most cases. For applications where the monitor is an open surface rather than a box that
        encloses the device, it is advisable to pick the size of the monitor such that the
        recorded near fields decay to negligible values near the edges of the monitor.

        .. TODO TYPO FIX o that the approximations are not used, and the projection is accurate even just a few wavelengths away from the near field locations.

        By default, if no :attr:`proj_distance` was provided, the fields are projected to a distance of 1m.

        **Server-side field projection Application**

        Provide the :class:`FieldProjectionAngleMonitor` monitor as an input to the
        :class:`.Simulation` object as one of its monitors. Now, we no longer need to provide a separate near-field
        :class:`FieldMonitor` - the near fields will automatically be recorded based on the size and location of the
        ``FieldProjectionAngleMonitor``. Note also that in some cases, the server-side computations may be slightly
        more accurate than client-side ones, because on the server, the near fields are not downsampled at all.

        We can re-project the already-computed far fields to a different distance away from the structure - we
        neither need to run another simulation nor re-run the :class:`FieldProjector`.

        **Far-Field Approximation Selection**

        .. TODO unsure if add on params?

        If the distance between the near and far field locations is
        much larger than the size of the device, one can typically set :attr:`far_field_approx` to
        ``True``, which will make use of the far-field approximation to speed up calculations.
        If the projection distance is comparable to the size of the device, we recommend setting
        :attr:`far_field_approx` to ``False``.

        .. image:: ../../notebooks/img/n2f_diagram.png

        .. TODO Fix that image so remove right irrelevant side

        When selected, it is assumed that:

        -   The fields are measured at a distance much greater than the size of our simulation in the transverse
            direction.
        -   The geometric approximations imply that any quantity whose magnitude drops off as
            :math:`\\frac{1}{r^2}` or faster is ignored.

        The advantages of these approximations are:

        *   The projections are computed relatively fast.
        *   The projections are cast in a simple mathematical form.
            which allows re-projecting the fields to different distance without the need to re-run a simulation or to
            re-run the :class:`FieldProjector`.

        In cases where we may want to project to intermediate distances where the far field approximation is no
        longer valid, simply include the class definition parameter :attr:`far_field_approx` to ``False`` in the
        ``FieldProjectionAngleMonitor`` instantiation. The resulting computations will be a bit slower,
        but the results will be significantly more accurate.

        .. TODO include here inherited methods.

    Example
    -------
    >>> monitor = FieldProjectionAngleMonitor(
    ...     center=(1,2,3),
    ...     size=(2,2,2),
    ...     freqs=[250e12, 300e12],
    ...     name='n2f_monitor',
    ...     custom_origin=(1,2,3),
    ...     phi=[0, np.pi/2],
    ...     theta=np.linspace(-np.pi/2, np.pi/2, 100),
    ...     far_field_approx=True,
    ...     )

    See Also
    --------

    **Notebooks**:

        * `Performing near field to far field projections <../../notebooks/FieldProjections.html>`_
        * `Field projection for a zone plate <../../notebooks/ZonePlateFieldProjection.html>`_: Realistic case study further demonstrating the accuracy of the field projections.
        * `Metalens in the visible frequency range <../../notebooks/Metalens.html>`_: Realistic case study further demonstrating the accuracy of the field projections.
        * `Multilevel blazed diffraction grating <../../notebooks/GratingEfficiency.html>`_: For far field projections in the context of perdiodic boundary conditions.
    """

    proj_distance: float = Field(
        1e6,
        title="Projection Distance",
        description="Radial distance of the projection points from ``local_origin``.",
        json_schema_extra={"units": MICROMETER},
    )

    theta: ObsGridArray = Field(
        title="Polar Angles",
        description="Polar angles with respect to the global z axis, relative to the location of "
        "``local_origin``, at which to project fields.",
        json_schema_extra={"units": RADIAN},
    )

    phi: ObsGridArray = Field(
        title="Azimuth Angles",
        description="Azimuth angles with respect to the global z axis, relative to the location of "
        "``local_origin``, at which to project fields.",
        json_schema_extra={"units": RADIAN},
    )

    def storage_size(self, num_cells: int, tmesh: ArrayFloat1D) -> int:
        """Size of monitor storage given the number of points after discretization."""
        # stores 1 complex number per pair of angles, per frequency,
        # for Er, Etheta, Ephi, Hr, Htheta, and Hphi (6 components)
        return BYTES_COMPLEX * len(self.theta) * len(self.phi) * len(self.freqs) * 6


class DirectivityMonitor(MicrowaveBaseModel, FieldProjectionAngleMonitor, FluxMonitor):
    """
    :class:`~tidy3d.Monitor` that records the radiation characteristics of antennas in the frequency domain
    at specified observation angles.

    Note
    ----
    For directivity, the computation is based on the ratio of the radiation
    intensity in a given direction to the average radiation intensity
    over all directions:

        Balanis, Constantine A., "Antenna Theory: Analysis and Design,"
        John Wiley & Sons, Chapter 2.6 (2016).

    For axial ratio, the computation is based on:

        Balanis, Constantine A., "Antenna Theory: Analysis and Design,"
        John Wiley & Sons, Chapter 2.12 (2016).

    Example
    -------
    >>> import numpy as np
    >>> monitor = DirectivityMonitor(  # doctest: +SKIP
    ...     center=(0, 0, 0),
    ...     size=(2, 2, 2),
    ...     freqs=[1e9, 2e9, 3e9],
    ...     name='directivity_monitor',
    ...     theta=np.linspace(0, np.pi, 10),
    ...     phi=np.linspace(0, 2*np.pi, 20),
    ... )
    """

    far_field_approx: Literal[True] = Field(
        True,
        title="Far Field Approximation",
        description="Directivity calculations require the far field approximation. "
        "This field is hard-coded to be ``True`` and cannot be changed.",
    )

    def storage_size(self, num_cells: int, tmesh: ArrayFloat1D) -> int:
        """Size of monitor storage given the number of points after discretization."""
        # stores 1 complex number per pair of angles, per frequency,
        # for Er, Etheta, Ephi, Hr, Htheta, and Hphi (6 components)
        # stores 1 real number per frequency for flux
        return BYTES_COMPLEX * len(self.theta) * len(self.phi) * len(
            self.freqs
        ) * 6 + BYTES_REAL * len(self.freqs)


class FieldProjectionCartesianMonitor(AbstractFieldProjectionMonitor):
    """:class:`~tidy3d.Monitor` that samples electromagnetic near fields in the frequency domain
    and projects them on a Cartesian observation plane.

    Notes
    -----

        **Parameters Caveats**

        The :attr:`center` and :attr:`size` fields define
        where the monitor will be placed in order to record near fields, typically very close
        to the structure of interest. The near fields are then projected
        to far-field locations defined by :attr:`x`, :attr:`y`, and :attr:`proj_distance`, relative
        to the :attr:`custom_origin`.

        Here, :attr:`x` and :attr:`y`, correspond to a local coordinate system
        where the local ``z`` axis is defined by :attr:`proj_axis`: which is the axis normal to this monitor.

        **Far-Field Approximation Selection**

        If the distance between the near and far field locations is much larger than the size of the
        device, one can typically set :attr:`far_field_approx` to ``True``, which will make use of the
        far-field approximation to speed up calculations. If the projection distance is comparable
        to the size of the device, we recommend setting :attr:`far_field_approx` to ``False``,
        so that the approximations are not used, and the projection is accurate even just a few
        wavelengths away from the near field locations.

        For applications where the monitor is an open surface rather than a box that
        encloses the device, it is advisable to pick the size of the monitor such that the
        recorded near fields decay to negligible values near the edges of the monitor.

        .. image:: ../../notebooks/img/n2f_diagram.png

        .. TODO unsure if add on params?

        When selected, it is assumed that:

        -   The fields are measured at a distance much greater than the size of our simulation in the transverse
            direction.
        -   The geometric approximations imply that any quantity whose magnitude drops off as
            :math:`\\frac{1}{r^2}` or faster is ignored.

        The advantages of these approximations are:

        *   The projections are computed relatively fast.
        *   The projections are cast in a simple mathematical form.
            which allows re-projecting the fields to different distance without the need to re-run a simulation or to
            re-run the :class:`FieldProjector`.


        In cases where we may want to project to intermediate distances where the far field approximation is no
        longer valid, simply include the class definition parameter ``far_field_approx=False`` in the
        ``FieldProjectionCartesianMonitor`` instantiation. The resulting computations will be a bit slower,
        but the results will be significantly more accurate.

        .. TODO include this example

        **Usage Caveats**

        .. TODO I believe a little illustration here would be handy.

        Since field projections rely on the surface equivalence principle, we have assumed that the tangential near
        fields recorded on the near field monitor serve as equivalent sources which generate the correct far fields.
        However, this requires that the field strength decays nearly to zero near the edges of the near-field
        monitor, which may not always be the case. For example, if we had used a larger aperture compared to the full
        simulation size in the transverse direction, we may expect a degradation in accuracy of the field
        projections. Despite this limitation, the field projections are still remarkably accurate in realistic
        scenarios. For realistic case studies further demonstrating the accuracy of the field projections,
        see our metalens and zone plate case studies.

        The field projections make use of the analytical homogeneous medium Green’s function, which assumes that the fields
        are propagating in a homogeneous medium. Therefore, one should use PMLs / absorbers as boundary conditions in the
        part of the domain where fields are projected. For far field projections in the context of perdiodic boundary
        conditions, see the diffraction efficiency example which demonstrates the use of a DiffractionMonitor.

        Server-side field projections will add to the monetary cost of the simulation. However, typically the far field
        projections have a very small computation cost compared to the FDTD simulation itself, so the increase in monetary
        cost should be negligibly small in most cases.

    Example
    -------
    >>> monitor = FieldProjectionCartesianMonitor(
    ...     center=(1,2,3),
    ...     size=(2,2,2),
    ...     freqs=[250e12, 300e12],
    ...     name='n2f_monitor',
    ...     custom_origin=(1,2,3),
    ...     x=[-1, 0, 1],
    ...     y=[-2, -1, 0, 1, 2],
    ...     proj_axis=2,
    ...     proj_distance=5,
    ...     far_field_approx=True,
    ...     )

    See Also
    --------

    **Notebooks**:
        * `Performing near field to far field projections <../../notebooks/FieldProjections.html>`_
        * `Field projection for a zone plate <../../notebooks/ZonePlateFieldProjection.html>`_
        * `Metalens in the visible frequency range <../../notebooks/Metalens.html>`_
        * `Multilevel blazed diffraction grating <../../notebooks/GratingEfficiency.html>`_
    """

    proj_axis: Axis = Field(
        title="Projection Plane Axis",
        description="Axis along which the observation plane is oriented.",
    )

    proj_distance: float = Field(
        1e6,
        title="Projection Distance",
        description="Signed distance of the projection plane along ``proj_axis``. "
        "from the plane containing ``local_origin``.",
        json_schema_extra={"units": MICROMETER},
    )

    x: ObsGridArray = Field(
        title="Local x Observation Coordinates",
        description="Local x observation coordinates w.r.t. ``local_origin`` and ``proj_axis``. "
        "When ``proj_axis`` is 0, this corresponds to the global y axis. "
        "When ``proj_axis`` is 1, this corresponds to the global x axis. "
        "When ``proj_axis`` is 2, this corresponds to the global x axis. ",
        json_schema_extra={"units": MICROMETER},
    )

    y: ObsGridArray = Field(
        title="Local y Observation Coordinates",
        description="Local y observation coordinates w.r.t. ``local_origin`` and ``proj_axis``. "
        "When ``proj_axis`` is 0, this corresponds to the global z axis. "
        "When ``proj_axis`` is 1, this corresponds to the global z axis. "
        "When ``proj_axis`` is 2, this corresponds to the global y axis. ",
        json_schema_extra={"units": MICROMETER},
    )

    def storage_size(self, num_cells: int, tmesh: ArrayFloat1D) -> int:
        """Size of monitor storage given the number of points after discretization."""
        # stores 1 complex number per pair of grid points, per frequency,
        # for Er, Etheta, Ephi, Hr, Htheta, and Hphi (6 components)
        return BYTES_COMPLEX * len(self.x) * len(self.y) * len(self.freqs) * 6


class FieldProjectionKSpaceMonitor(AbstractFieldProjectionMonitor):
    """:class:`~tidy3d.Monitor` that samples electromagnetic near fields in the frequency domain
    and projects them on an observation plane defined in k-space.

     Notes
     -----

        The :attr:`center` and :attr:`size`
        fields define where the monitor will be placed in order to record near fields, typically
        very close to the structure of interest. The near fields are then
        projected to far-field locations defined in k-space by ``ux``, ``uy``, and ``proj_distance``,
        relative to the ``custom_origin``. Here, ``ux`` and ``uy`` are associated with a local
        coordinate system where the local 'z' axis is defined by ``proj_axis``: which is the axis
        normal to this monitor. If the distance between the near and far field locations is much
        larger than the size of the device, one can typically set ``far_field_approx`` to ``True``,
        which will make use of the far-field approximation to speed up calculations. If the
        projection distance is comparable to the size of the device, we recommend setting
        ``far_field_approx`` to ``False``, so that the approximations are not used, and the
        projection is accurate even just a few wavelengths away from the near field locations.
        For applications where the monitor is an open surface rather than a box that
        encloses the device, it is advisable to pick the size of the monitor such that the
        recorded near fields decay to negligible values near the edges of the monitor.

        **Usage Caveats**

        .. TODO I believe a little illustration here would be handy.

        Since field projections rely on the surface equivalence principle, we have assumed that the tangential near
        fields recorded on the near field monitor serve as equivalent sources which generate the correct far fields.
        However, this requires that the field strength decays nearly to zero near the edges of the near-field
        monitor, which may not always be the case. For example, if we had used a larger aperture compared to the full
        simulation size in the transverse direction, we may expect a degradation in accuracy of the field
        projections. Despite this limitation, the field projections are still remarkably accurate in realistic
        scenarios. For realistic case studies further demonstrating the accuracy of the field projections,
        see our metalens and zone plate case studies.

        The field projections make use of the analytical homogeneous medium Green’s function, which assumes that the fields
        are propagating in a homogeneous medium. Therefore, one should use PMLs / absorbers as boundary conditions in the
        part of the domain where fields are projected. For far field projections in the context of perdiodic boundary
        conditions, see the diffraction efficiency example which demonstrates the use of a :class:`DiffractionMonitor`.

        Server-side field projections will add to the monetary cost of the simulation. However, typically the far field
        projections have a very small computation cost compared to the FDTD simulation itself, so the increase in monetary
        cost should be negligibly small in most cases.

    Example
    -------
    >>> monitor = FieldProjectionKSpaceMonitor(
    ...     center=(1,2,3),
    ...     size=(2,2,2),
    ...     freqs=[250e12, 300e12],
    ...     name='n2f_monitor',
    ...     custom_origin=(1,2,3),
    ...     proj_axis=2,
    ...     ux=[0.1,0.2],
    ...     uy=[0.3,0.4,0.5]
    ...     )

    See Also
    --------

    **Notebooks**:
        * `Performing near field to far field projections <../../notebooks/FieldProjections.html>`_
        * `Field projection for a zone plate <../../notebooks/ZonePlateFieldProjection.html>`_
        * `Metalens in the visible frequency range <../../notebooks/Metalens.html>`_
        * `Multilevel blazed diffraction grating <../../notebooks/GratingEfficiency.html>`_
    """

    proj_axis: Axis = Field(
        title="Projection Plane Axis",
        description="Axis along which the observation plane is oriented.",
    )

    proj_distance: float = Field(
        1e6,
        title="Projection Distance",
        description="Radial distance of the projection points from ``local_origin``.",
        json_schema_extra={"units": MICROMETER},
    )

    ux: ObsGridArray = Field(
        title="Normalized kx",
        description="Local x component of wave vectors on the observation plane, "
        "relative to ``local_origin`` and oriented with respect to ``proj_axis``, "
        "normalized by (2*pi/lambda) where lambda is the wavelength "
        "associated with the background medium. Must be in the range [-1, 1].",
    )

    uy: ObsGridArray = Field(
        title="Normalized ky",
        description="Local y component of wave vectors on the observation plane, "
        "relative to ``local_origin`` and oriented with respect to ``proj_axis``, "
        "normalized by (2*pi/lambda) where lambda is the wavelength "
        "associated with the background medium. Must be in the range [-1, 1].",
    )

    @model_validator(mode="after")
    def reciprocal_vector_range(self) -> Self:
        """Ensure that ux, uy are in [-1, 1]."""
        maxabs_ux = max(list(self.ux), key=abs)
        maxabs_uy = max(list(self.uy), key=abs)
        name = self.name
        if maxabs_ux > 1:
            self._raise_validation_error_at_loc(
                SetupError(f"Entries of 'ux' must lie in the range [-1, 1] for monitor {name}."),
                "ux",
            )
        if maxabs_uy > 1:
            self._raise_validation_error_at_loc(
                SetupError(f"Entries of 'uy' must lie in the range [-1, 1] for monitor {name}."),
                "uy",
            )
        return self

    def storage_size(self, num_cells: int, tmesh: ArrayFloat1D) -> int:
        """Size of monitor storage given the number of points after discretization."""
        # stores 1 complex number per pair of grid points, per frequency,
        # for Er, Etheta, Ephi, Hr, Htheta, and Hphi (6 components)
        return BYTES_COMPLEX * len(self.ux) * len(self.uy) * len(self.freqs) * 6


class DiffractionMonitor(PlanarMonitor, FreqMonitor):
    """:class:`~tidy3d.Monitor` that uses a 2D Fourier transform to compute the
    diffraction amplitudes and efficiency for allowed diffraction orders.

    Note
    ----

        The diffraction data are separated into S and P polarizations. At normal incidence when
        S and P are undefined, P(S) corresponds to ``Ey``(``Ez``) polarization for monitor normal
        to x, P(S) corresponds to ``Ex``(``Ez``) polarization for monitor normal to y, and P(S)
        corresponds to ``Ex``(``Ey``) polarization for monitor normal to z.

    Note
    ----

        The power amplitudes per polarization and diffraction order, and correspondingly the power
        per diffraction order, correspond to the power carried by each diffraction order in the
        monitor normal direction. They are not to be confused with power carried by plane waves
        in the propagation direction of each diffraction order, which can be obtained from the
        spherical-coordinate fields which are also stored. The power definition is such that the
        grating efficiency is the recorded power over the input source power, and the direct sum
        over the power in all orders should equal the total power flowing through the monitor.

    Example
    -------
    >>> monitor = DiffractionMonitor(
    ...     center=(1,2,3),
    ...     size=(inf,inf,0),
    ...     freqs=[250e12, 300e12],
    ...     name='diffraction_monitor',
    ...     normal_dir='+',
    ...     )

    See Also
    --------

    **Notebooks**
        * `Multilevel blazed diffraction grating <../../notebooks/GratingEfficiency.html>`_
    """

    normal_dir: Direction = Field(
        "+",
        title="Normal Vector Orientation",
        description="Direction of the surface monitor's normal vector w.r.t. "
        "the positive x, y or z unit vectors. Must be one of ``'+'`` or ``'-'``. "
        "Defaults to ``'+'`` if not provided.",
    )

    colocate: Literal[False] = Field(
        False,
        title="Colocate Fields",
        description="Defines whether fields are colocated to grid cell boundaries (i.e. to the "
        "primal grid). Can be toggled for field recording monitors and is hard-coded for other "
        "monitors depending on their specific function.",
    )

    @field_validator("size")
    @classmethod
    def diffraction_monitor_size(cls: type[DiffractionMonitor], val: Size) -> Size:
        """Ensure that the monitor is infinite in the transverse direction."""
        if val.count(inf) != 2:
            raise SetupError(
                "A 'DiffractionMonitor' must have a size of 'td.inf' along both "
                f"transverse directions, given size={val}."
            )
        return val

    def storage_size(self, num_cells: int, tmesh: ArrayFloat1D) -> int:
        """Size of monitor storage given the number of points after discretization."""
        # assumes 1 diffraction order per frequency; actual size will be larger
        return BYTES_COMPLEX * len(self.freqs)

    def supports_parallel_adjoint(self) -> bool:
        """Return ``True`` for diffraction monitor adjoints based on amplitude data."""
        return True

    def parallel_adjoint_bases(
        self, simulation: Simulation, monitor_index: int
    ) -> list[ParallelAdjointBasis]:
        """Return parallel adjoint bases for diffraction monitor amplitudes."""
        return _diffraction_parallel_adjoint_bases(self, simulation, monitor_index)

    def _storage_size_solver(self, num_cells: int, tmesh: ArrayFloat1D) -> int:
        """Size of intermediate data recorded by the monitor during a solver run."""
        return BYTES_COMPLEX * num_cells * len(self.freqs) * 6


class AbstractSurfaceMonitor(Monitor, ABC):
    """:class:`Monitor` that records electromagnetic field data as a function of x,y,z on PEC and lossy metal surfaces."""

    fields: tuple[EMSurfaceField, ...] = Field(
        ["E", "H"],
        title="Field Components",
        description="Collection of field components to store in the monitor.",
    )

    interval_space: tuple[Literal[1], Literal[1], Literal[1]] = Field(
        (1, 1, 1),
        title="Spatial Interval",
        description="Number of grid step intervals between monitor recordings. "
        "Only the value of 1 (no downsampling) is currently supported for surface monitors.",
    )

    colocate: Literal[True] = Field(
        True,
        title="Colocate Fields",
        description="For surface monitors fields are always colocated on surface.",
    )

    _check_volumetic = assert_volumetric()

    @model_validator(mode="after")
    def _warn_beta_stage(self) -> Self:
        """Warn that surface monitors are in beta stage."""

        log.warning(
            "Surface monitors are currently in beta stage. Please exercise caution when analyzing "
            "surface monitor data and verify results carefully. If you encounter any issues, "
            "please report them to our support team.",
            log_once=True,
        )
        return self


class SurfaceFieldMonitor(AbstractSurfaceMonitor, FreqMonitor):
    """:class:`Monitor` that records electromagnetic fields in the frequency domain on PEC and lossy metal surfaces.

    Notes
    -----

        :class:`SurfaceFieldMonitor` objects operate by running a discrete Fourier transform of the fields at a given set of
        frequencies to perform the calculation "in-place" with the time stepping. These monitors are designed
        to record fields on PEC (:class:`PECMedium`) and lossy metal (:class:`LossyMetalMedium`) surfaces,
        storing the normal E and tangential H fields.

    Example
    -------
    >>> import tidy3d as td
    >>> old_logging_level = td.config.logging_level
    >>> td.config.logging_level = "ERROR"
    >>> monitor = SurfaceFieldMonitor(
    ...     center=(1,2,3),
    ...     size=(2,2,2),
    ...     fields=['E', 'H'],
    ...     freqs=[250e12, 300e12],
    ...     name='surface_monitor')
    >>> td.config.logging_level = old_logging_level
    """

    def storage_size(self, num_cells: int, tmesh: ArrayFloat1D) -> int:
        """Size of monitor storage given the number of points after discretization.
        In general, this is severely overestimated for surface monitors.
        """

        # estimation based on triangulated surface when it crosses cells in xy plane
        num_tris = num_cells * 6
        num_points = num_cells * 4

        # storing 3 coordinate components per point
        storage = 3 * BYTES_REAL * num_points

        # storing 3 indices per triangle
        storage += 3 * BYTES_REAL * num_tris

        # EH field values + normal field
        storage += (
            BYTES_COMPLEX * num_points * len(self.freqs) * len(self.fields) * 3
            + 3 * num_points * BYTES_REAL
        )

        return storage

    def _storage_size_solver(self, num_cells: int, tmesh: ArrayFloat1D) -> int:
        """Size of intermediate data recorded by the monitor during a solver run."""

        # fields
        storage = BYTES_COMPLEX * num_cells * len(self.freqs) * len(self.fields) * 3

        # fields valid map
        storage += BYTES_REAL * num_cells * len(self.freqs) * len(self.fields) * 3

        # auxiliary variables (normals and locations)
        storage += BYTES_REAL * num_cells * 7 * 4

        return storage


class SurfaceFieldTimeMonitor(AbstractSurfaceMonitor, TimeMonitor):
    """:class:`Monitor` that records electromagnetic fields in the time domain on PEC and lossy metal surfaces.

    Notes
    -----

        :class:`SurfaceFieldTimeMonitor` objects are best used to monitor the time dependence of the fields
        on PEC (:class:`PECMedium`) and lossy metal (:class:`LossyMetalMedium`) surfaces. They can also be used to create
        “animations” of the field pattern evolution.

        To create an animation, we need to capture the frames at different time instances of the simulation. This can
        be done by using a :class:`SurfaceFieldTimeMonitor`. Usually a FDTD simulation contains a large number of time steps
        and grid points. Recording the field at every time step and grid point will result in a large dataset. For
        the purpose of making animations, this is usually unnecessary.


    Example
    -------
    >>> import tidy3d as td
    >>> old_logging_level = td.config.logging_level
    >>> td.config.logging_level = "ERROR"
    >>> monitor = SurfaceFieldTimeMonitor(
    ...     center=(1,2,3),
    ...     size=(2,2,2),
    ...     fields=['H'],
    ...     start=1e-13,
    ...     stop=5e-13,
    ...     interval=2,
    ...     name='movie_monitor')
    >>> td.config.logging_level = old_logging_level
    """

    def storage_size(self, num_cells: int, tmesh: ArrayFloat1D) -> int:
        """Size of monitor storage given the number of points after discretization.
        In general, this is severely overestimated for surface monitors.
        """
        num_steps = self.num_steps(tmesh)

        # estimation based on triangulated surface when it crosses cells in xy plane
        num_tris = num_cells * 6
        num_points = num_cells * 4

        # storing 3 coordinate components per point
        storage = 3 * BYTES_REAL * num_points

        # storing 3 indices per triangle
        storage += 3 * BYTES_REAL * num_tris

        # EH field values + normal field
        storage += (
            BYTES_COMPLEX * num_points * num_steps * len(self.fields) * 3
            + 3 * num_points * BYTES_REAL
        )

        return storage

    def _storage_size_solver(self, num_cells: int, tmesh: ArrayFloat1D) -> int:
        """Size of intermediate data recorded by the monitor during a solver run."""

        num_steps = self.num_steps(tmesh)

        # fields
        storage = BYTES_COMPLEX * num_cells * num_steps * len(self.fields) * 3

        # fields valid map
        storage += BYTES_REAL * num_cells * num_steps * len(self.fields) * 3

        # auxiliary variables (normals and locations)
        storage += BYTES_REAL * num_cells * 7 * 4

        return storage
