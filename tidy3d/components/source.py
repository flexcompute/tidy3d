"""Defines electric current sources for injecting light into simulation."""

from abc import ABC, abstractmethod
from typing import Union, Tuple, Optional

from typing_extensions import Literal
import pydantic
import numpy as np

from .base import Tidy3dBaseModel, cached_property

from .types import Direction, Polarization, Ax, FreqBound, ArrayFloat1D, Axis, PlotVal
from .validators import assert_plane, assert_volumetric, validate_name_str, get_value
from .validators import warn_if_dataset_none, assert_single_freq_in_range
from .data.dataset import FieldDataset
from .geometry import Box, Coordinate
from .mode import ModeSpec
from .viz import add_ax_if_none, PlotParams, plot_params_source
from .viz import ARROW_COLOR_SOURCE, ARROW_ALPHA, ARROW_COLOR_POLARIZATION
from ..constants import RADIAN, HERTZ, MICROMETER, GLANCING_CUTOFF
from ..constants import inf  # pylint:disable=unused-import
from ..exceptions import SetupError
from ..log import log


# in spectrum computation, discard amplitudes with relative magnitude smaller than cutoff
DFT_CUTOFF = 1e-8
# when checking if custom data spans the source plane, allow for a small tolerance
# due to numerical precision
DATA_SPAN_TOL = 1e-8
# width of Chebyshev grid used for broadband sources (in units of pulse width)
CHEB_GRID_WIDTH = 1.5


class SourceTime(ABC, Tidy3dBaseModel):
    """Base class describing the time dependence of a source."""

    amplitude: pydantic.NonNegativeFloat = pydantic.Field(
        1.0, title="Amplitude", description="Real-valued maximum amplitude of the time dependence."
    )

    phase: float = pydantic.Field(
        0.0, title="Phase", description="Phase shift of the time dependence.", units=RADIAN
    )

    @abstractmethod
    def amp_time(self, time: float) -> complex:
        """Complex-valued source amplitude as a function of time.

        Parameters
        ----------
        time : float
            Time in seconds.

        Returns
        -------
        complex
            Complex-valued source amplitude at that time..
        """

    def spectrum(
        self,
        times: ArrayFloat1D,
        freqs: ArrayFloat1D,
        dt: float,
        complex_fields: bool = False,
    ) -> complex:
        """Complex-valued source spectrum as a function of frequency

        Parameters
        ----------
        times : np.ndarray
            Times to use to evaluate spectrum Fourier transform.
            (Typically the simulation time mesh).
        freqs : np.ndarray
            Frequencies in Hz to evaluate spectrum at.
        dt : float or np.ndarray
            Time step to weight FT integral with.
            If array, use to weigh each of the time intervals in ``times``.
        complex_fields : bool
            Whether time domain fields are complex, e.g., for Bloch boundaries

        Returns
        -------
        np.ndarray
            Complex-valued array (of len(freqs)) containing spectrum at those frequencies.
        """

        times = np.array(times)
        freqs = np.array(freqs)
        time_amps = self.amp_time(times)

        if not complex_fields:
            time_amps = np.real(time_amps)

        # Cut to only relevant times
        count_times = np.where(np.abs(time_amps) / np.amax(np.abs(time_amps)) > DFT_CUTOFF)
        time_amps = time_amps[count_times]
        times_cut = times[count_times]

        # (Nf, Nt_cut) matrix that gives DFT when matrix multiplied with signal
        dft_matrix = np.exp(2j * np.pi * freqs[:, None] * times_cut) / np.sqrt(2 * np.pi)
        return dt * dft_matrix @ time_amps

    @add_ax_if_none
    def plot(self, times: ArrayFloat1D, val: PlotVal = "real", ax: Ax = None) -> Ax:
        """Plot the complex-valued amplitude of the source time-dependence.

        Parameters
        ----------
        times : np.ndarray
            Array of times (seconds) to plot source at.
            To see source time amplitude for a specific :class:`Simulation`,
            pass ``simulation.tmesh``.
        val : Literal['real', 'imag', 'abs'] = 'real'
            Which part of the spectrum to plot.
        ax : matplotlib.axes._subplots.Axes = None
            Matplotlib axes to plot on, if not specified, one is created.

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.
        """
        times = np.array(times)
        amp_complex = self.amp_time(times)

        if val == "real":
            ax.plot(times, amp_complex.real, color="blueviolet", label="real")
        elif val == "imag":
            ax.plot(times, amp_complex.imag, color="crimson", label="imag")
        elif val == "abs":
            ax.plot(times, np.abs(amp_complex), color="k", label="abs")
        else:
            raise ValueError(f"Plot 'val' option of '{val}' not recognized.")
        ax.set_xlabel("time (s)")
        ax.set_title("source amplitude")
        ax.legend()
        ax.set_aspect("auto")
        return ax

    # pylint:disable=too-many-arguments
    @add_ax_if_none
    def plot_spectrum(
        self,
        times: ArrayFloat1D,
        num_freqs: int = 101,
        val: PlotVal = "real",
        ax: Ax = None,
        complex_fields: bool = False,
    ) -> Ax:
        """Plot the complex-valued amplitude of the source time-dependence.

        Parameters
        ----------
        times : np.ndarray
            Array of evenly-spaced times (seconds) to evaluate source time-dependence at.
            The spectrum is computed from this value and the source time frequency content.
            To see source spectrum for a specific :class:`Simulation`,
            pass ``simulation.tmesh``.
        num_freqs : int = 101
            Number of frequencies to plot within the SourceTime.frequency_range.
        ax : matplotlib.axes._subplots.Axes = None
            Matplotlib axes to plot on, if not specified, one is created.
        complex_fields : bool
            Whether time domain fields are complex, e.g., for Bloch boundaries

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.
        """
        times = np.array(times)

        dts = np.diff(times)
        if not np.allclose(dts, dts[0] * np.ones_like(dts), atol=1e-17):
            raise SetupError("Supplied times not evenly spaced.")

        dt = np.mean(dts)

        fmin, fmax = self.frequency_range()
        freqs = np.linspace(fmin, fmax, num_freqs)

        spectrum = self.spectrum(times=times, dt=dt, freqs=freqs, complex_fields=complex_fields)

        if val == "real":
            ax.plot(freqs, spectrum.real, color="blueviolet", label="real")
        elif val == "imag":
            ax.plot(freqs, spectrum.imag, color="crimson", label="imag")
        elif val == "abs":
            ax.plot(freqs, np.abs(spectrum), color="k", label="abs")
        else:
            raise ValueError(f"Plot 'val' option of '{val}' not recognized.")
        ax.set_xlabel("frequency (Hz)")
        ax.set_title("source spectrum")
        ax.legend()
        ax.set_aspect("auto")
        return ax

    @abstractmethod
    def frequency_range(self, num_fwidth: float = 4.0) -> FreqBound:
        """Frequency range within plus/minus ``num_fwidth * fwidth`` of the central frequency."""


class Pulse(SourceTime, ABC):
    """A source time that ramps up with some ``fwidth`` and oscillates at ``freq0``."""

    freq0: pydantic.PositiveFloat = pydantic.Field(
        ..., title="Central Frequency", description="Central frequency of the pulse.", units=HERTZ
    )
    fwidth: pydantic.PositiveFloat = pydantic.Field(
        ...,
        title="",
        description="Standard deviation of the frequency content of the pulse.",
        units=HERTZ,
    )

    offset: float = pydantic.Field(
        5.0,
        title="Offset",
        description="Time delay of the maximum value of the "
        "pulse in units of 1 / (``2pi * fwidth``).",
        ge=2.5,
    )

    def frequency_range(self, num_fwidth: float = 4.0) -> FreqBound:
        """Frequency range within 5 standard deviations of the central frequency.

        Parameters
        ----------
        num_fwidth : float = 4.
            Frequency range defined as plus/minus ``num_fwidth * self.fwdith``.

        Returns
        -------
        Tuple[float, float]
            Minimum and maximum frequencies of the :class:`GaussianPulse` or :class:`ContinuousWave`
            power.
        """

        freq_width_range = num_fwidth * self.fwidth
        freq_min = max(0, self.freq0 - freq_width_range)
        freq_max = self.freq0 + freq_width_range
        return (freq_min, freq_max)


class GaussianPulse(Pulse):
    """Source time dependence that describes a Gaussian pulse.

    Example
    -------
    >>> pulse = GaussianPulse(freq0=200e12, fwidth=20e12)
    """

    def amp_time(self, time: float) -> complex:
        """Complex-valued source amplitude as a function of time."""

        twidth = 1.0 / (2 * np.pi * self.fwidth)
        omega0 = 2 * np.pi * self.freq0
        time_shifted = time - self.offset * twidth

        const = 1j + time_shifted / twidth**2 / omega0
        offset = np.exp(1j * self.phase)
        oscillation = np.exp(-1j * omega0 * time)
        amp = np.exp(-(time_shifted**2) / 2 / twidth**2) * self.amplitude

        return const * offset * oscillation * amp


class ContinuousWave(Pulse):
    """Source time dependence that ramps up to continuous oscillation
    and holds until end of simulation.

    Example
    -------
    >>> cw = ContinuousWave(freq0=200e12, fwidth=20e12)
    """

    def amp_time(self, time: float) -> complex:
        """Complex-valued source amplitude as a function of time."""

        twidth = 1.0 / (2 * np.pi * self.fwidth)
        omega0 = 2 * np.pi * self.freq0
        time_shifted = time - self.offset * twidth

        const = 1.0
        offset = np.exp(1j * self.phase)
        oscillation = np.exp(-1j * omega0 * time)
        amp = 1 / (1 + np.exp(-time_shifted / twidth)) * self.amplitude

        return const * offset * oscillation * amp


SourceTimeType = Union[GaussianPulse, ContinuousWave]

""" Source objects """


class Source(Box, ABC):
    """Abstract base class for all sources."""

    source_time: SourceTimeType = pydantic.Field(
        ..., title="Source Time", description="Specification of the source time-dependence."
    )

    name: str = pydantic.Field(None, title="Name", description="Optional name for the source.")

    @cached_property
    def plot_params(self) -> PlotParams:
        """Default parameters for plotting a Source object."""
        return plot_params_source

    _name_validator = validate_name_str()

    @cached_property
    def geometry(self) -> Box:
        """:class:`Box` representation of source."""

        return Box(center=self.center, size=self.size)

    @cached_property
    def _injection_axis(self):
        """Injection axis of the source."""
        return None

    @cached_property
    def _dir_vector(self) -> Tuple[float, float, float]:
        """Returns a vector indicating the source direction for arrow plotting, if not None."""
        return None

    @cached_property
    def _pol_vector(self) -> Tuple[float, float, float]:
        """Returns a vector indicating the source polarization for arrow plotting, if not None."""
        return None

    def plot(  #  pylint:disable=too-many-arguments
        self,
        x: float = None,
        y: float = None,
        z: float = None,
        ax: Ax = None,
        **patch_kwargs,
    ) -> Ax:

        kwargs_arrow_base = patch_kwargs.pop("arrow_base", None)

        # call the `Source.plot()` function first.
        ax = Box.plot(self, x=x, y=y, z=z, ax=ax, **patch_kwargs)

        kwargs_alpha = patch_kwargs.get("alpha")
        arrow_alpha = ARROW_ALPHA if kwargs_alpha is None else kwargs_alpha

        # then add the arrow based on the propagation direction
        if self._dir_vector is not None:
            bend_radius = None
            bend_axis = None
            if hasattr(self, "mode_spec"):
                bend_radius = self.mode_spec.bend_radius
                bend_axis = self._bend_axis

            ax = self._plot_arrow(
                x=x,
                y=y,
                z=z,
                ax=ax,
                direction=self._dir_vector,
                bend_radius=bend_radius,
                bend_axis=bend_axis,
                color=ARROW_COLOR_SOURCE,
                alpha=arrow_alpha,
                both_dirs=False,
                arrow_base=kwargs_arrow_base,
            )

        if self._pol_vector is not None:
            ax = self._plot_arrow(
                x=x,
                y=y,
                z=z,
                ax=ax,
                direction=self._pol_vector,
                color=ARROW_COLOR_POLARIZATION,
                alpha=arrow_alpha,
                both_dirs=False,
                arrow_base=kwargs_arrow_base,
            )

        return ax


""" Sources either: (1) implement current distributions or (2) generate fields."""


class CurrentSource(Source, ABC):
    """Source implements a current distribution directly."""

    polarization: Polarization = pydantic.Field(
        ...,
        title="Polarization",
        description="Specifies the direction and type of current component.",
    )

    @cached_property
    def _pol_vector(self) -> Tuple[float, float, float]:
        """Returns a vector indicating the source polarization for arrow plotting, if not None."""
        component = self.polarization[-1]  # 'x' 'y' or 'z'
        pol_axis = "xyz".index(component)
        pol_vec = [0, 0, 0]
        pol_vec[pol_axis] = 1
        return pol_vec


class ReverseInterpolatedSource(Source):
    """Abstract source that allows reverse-interpolation along zero-sized dimensions."""

    interpolate: bool = pydantic.Field(
        True,
        title="Enable Interpolation",
        description="Handles reverse-interpolation of zero-size dimensions of the source. "
        "If ``False``, the source data is snapped to the nearest Yee grid point. If ``True``, "
        "equivalent source data is applied on the surrounding Yee grid points to emulate "
        "placement at the specified location using linear interpolation.",
    )


class UniformCurrentSource(CurrentSource, ReverseInterpolatedSource):
    """Source in a rectangular volume with uniform time dependence. size=(0,0,0) gives point source.

    Example
    -------
    >>> pulse = GaussianPulse(freq0=200e12, fwidth=20e12)
    >>> pt_source = UniformCurrentSource(size=(0,0,0), source_time=pulse, polarization='Ex')
    """


class PointDipole(CurrentSource, ReverseInterpolatedSource):
    """Uniform current source with a zero size.

    Example
    -------
    >>> pulse = GaussianPulse(freq0=200e12, fwidth=20e12)
    >>> pt_dipole = PointDipole(center=(1,2,3), source_time=pulse, polarization='Ex')
    """

    size: Tuple[Literal[0], Literal[0], Literal[0]] = pydantic.Field(
        (0, 0, 0),
        title="Size",
        description="Size in x, y, and z directions, constrained to ``(0, 0, 0)``.",
        units=MICROMETER,
    )


class CustomCurrentSource(ReverseInterpolatedSource):
    """Implements a source corresponding to an input dataset containing ``E`` and ``H`` fields.
    Injects the specified components of the ``E`` and ``H`` dataset directly as ``J`` and ``M``
    current distributions in the FDTD solver.

    Note
    ----
        The coordinates of all provided fields are assumed to be relative to the source center.

    Example
    -------
    >>> from tidy3d import ScalarFieldDataArray
    >>> pulse = GaussianPulse(freq0=200e12, fwidth=20e12)
    >>> x = np.linspace(-1, 1, 101)
    >>> y = np.linspace(-1, 1, 101)
    >>> z = np.array([0])
    >>> f = [2e14]
    >>> coords = dict(x=x, y=y, z=z, f=f)
    >>> scalar_field = ScalarFieldDataArray(np.ones((101, 101, 1, 1)), coords=coords)
    >>> dataset = FieldDataset(Ex=scalar_field)
    >>> custom_source = CustomCurrentSource(
    ...     center=(1, 1, 1),
    ...     size=(2, 2, 0),
    ...     source_time=pulse,
    ...     current_dataset=dataset)

    """

    current_dataset: Optional[FieldDataset] = pydantic.Field(
        ...,
        title="Current Dataset",
        description=":class:`.FieldDataset` containing the desired frequency-domain "
        "electric and magnetic current patterns to inject.",
    )

    _current_dataset_none_warning = warn_if_dataset_none("current_dataset")
    _current_dataset_single_freq = assert_single_freq_in_range("current_dataset")


class FieldSource(Source, ABC):
    """A Source defined by the desired E and/or H fields."""


""" Field Sources can be defined either on a (1) surface or (2) volume. Defines injection_axis """


class PlanarSource(Source, ABC):
    """A source defined on a 2D plane."""

    _plane_validator = assert_plane()

    @cached_property
    def injection_axis(self):
        """Injection axis of the source."""
        return self._injection_axis

    @cached_property
    def _injection_axis(self):
        """Injection axis of the source."""
        return self.size.index(0.0)


class VolumeSource(Source, ABC):
    """A source defined in a 3D :class:`Box`."""

    _volume_validator = assert_volumetric()


""" Field Sources require more specification, for now, they all have a notion of a direction."""


class DirectionalSource(FieldSource, ABC):
    """A Field source that propagates in a given direction."""

    direction: Direction = pydantic.Field(
        ...,
        title="Direction",
        description="Specifies propagation in the positive or negative direction of the injection "
        "axis.",
    )

    @cached_property
    def _dir_vector(self) -> Tuple[float, float, float]:
        """Returns a vector indicating the source direction for arrow plotting, if not None."""
        if self._injection_axis is None:
            return None
        dir_vec = [0, 0, 0]
        dir_vec[int(self._injection_axis)] = 1 if self.direction == "+" else -1
        return dir_vec


class BroadbandSource(Source, ABC):
    """A source with frequency dependent field distributions."""

    num_freqs: int = pydantic.Field(
        1,
        title="Number of Frequency Points",
        description="Number of points used to approximate the frequency dependence of injected "
        "field. A Chebyshev interpolation is used, thus, only a small number of points, i.e., less "
        "than 20, is typically sufficient to obtain converged results.",
        ge=1,
        le=99,
    )

    @cached_property
    def frequency_grid(self) -> np.ndarray:
        """A Chebyshev grid used to approximate frequency dependence."""
        freq_min, freq_max = self.source_time.frequency_range(num_fwidth=CHEB_GRID_WIDTH)
        freq_avg = 0.5 * (freq_min + freq_max)
        freq_diff = 0.5 * (freq_max - freq_min)
        uni_points = (2 * np.arange(self.num_freqs) + 1) / (2 * self.num_freqs)
        cheb_points = np.cos(np.pi * np.flip(uni_points))
        return freq_avg + freq_diff * cheb_points

    @pydantic.validator("num_freqs", always=True, allow_reuse=True)
    def _warn_if_large_number_of_freqs(cls, val):
        """Warn if a large number of frequency points is requested."""

        if val is None:
            return val

        if val >= 20:
            log.warning(
                f"A large number ({val}) of frequency points is used in a broadband source. "
                "This can slow down simulation time and is only needed if the mode fields are "
                "expected to have a very sharp frequency dependence. 'num_freqs' < 20 is "
                "sufficient in most cases."
            )

        return val


""" Source current profiles determined by user-supplied data on a plane."""


class CustomFieldSource(FieldSource, PlanarSource):
    """Implements a source corresponding to an input dataset containing ``E`` and ``H`` fields,
    using the equivalence principle to define the actual injected currents. For the injection to
    work as expected (i.e. to reproduce the required ``E`` and ``H`` fields), the field data must
    decay by the edges of the source plane, or the source plane must span the entire simulation
    domain and the fields must match the simulation boundary conditions.
    The equivalent source currents are fully defined by the field components tangential to the
    source plane. For e.g. source normal along ``z``, the normal components (``Ez`` and ``Hz``)
    can be provided but will have no effect on the results, and at least one of the tangential
    components has to be in the dataset, i.e. at least one of ``Ex``, ``Ey``, ``Hx``, and ``Hy``.

    Note
    ----
        The coordinates of all provided fields are assumed to be relative to the source center.

    Note
    ----
        If only the ``E`` or only the ``H`` fields are provided, the source will not be directional,
        but will inject equal power in both directions instead.

    Example
    -------
    >>> from tidy3d import ScalarFieldDataArray
    >>> pulse = GaussianPulse(freq0=200e12, fwidth=20e12)
    >>> x = np.linspace(-1, 1, 101)
    >>> y = np.linspace(-1, 1, 101)
    >>> z = np.array([0])
    >>> f = [2e14]
    >>> coords = dict(x=x, y=y, z=z, f=f)
    >>> scalar_field = ScalarFieldDataArray(np.ones((101, 101, 1, 1)), coords=coords)
    >>> dataset = FieldDataset(Ex=scalar_field)
    >>> custom_source = CustomFieldSource(
    ...     center=(1, 1, 1),
    ...     size=(2, 2, 0),
    ...     source_time=pulse,
    ...     field_dataset=dataset)

    """

    field_dataset: Optional[FieldDataset] = pydantic.Field(
        ...,
        title="Field Dataset",
        description=":class:`.FieldDataset` containing the desired frequency-domain "
        "fields patterns to inject. At least one tangetial field component must be specified.",
    )

    _field_dataset_none_warning = warn_if_dataset_none("field_dataset")
    _field_dataset_single_freq = assert_single_freq_in_range("field_dataset")

    @pydantic.validator("field_dataset", always=True)
    def _tangential_component_defined(cls, val: FieldDataset, values: dict) -> FieldDataset:
        """Assert that at least one tangential field component is provided."""
        if val is None:
            return val
        size = get_value(key="size", values=values)
        normal_axis = size.index(0.0)
        _, (cmp1, cmp2) = cls.pop_axis("xyz", axis=normal_axis)
        for field in "EH":
            for cmp_name in (cmp1, cmp2):
                tangential_field = field + cmp_name
                if tangential_field in val.field_components:
                    return val
        raise SetupError("No tangential field found in the suppled 'field_dataset'.")


""" Source current profiles defined by (1) angle or (2) desired mode. Sets theta and phi angles."""


class AngledFieldSource(DirectionalSource, ABC):
    """A FieldSource defined with a an angled direction of propagation. The direction is defined by
    the polar and azimuth angles w.r.t. an injection axis, as well as forward ``+`` or
    backward ``-``. This base class only defines the ``direction`` and ``injection_axis``
    attributes, but it must be composed with a class that also defines ``angle_theta`` and
    ``angle_phi``."""

    angle_theta: float = pydantic.Field(
        0.0,
        title="Polar Angle",
        description="Polar angle of the propagation axis from the injection axis.",
        units=RADIAN,
    )

    angle_phi: float = pydantic.Field(
        0.0,
        title="Azimuth Angle",
        description="Azimuth angle of the propagation axis in the plane orthogonal to the "
        "injection axis.",
        units=RADIAN,
    )

    pol_angle: float = pydantic.Field(
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
        units=RADIAN,
    )

    @pydantic.validator("angle_theta", allow_reuse=True, always=True)
    def glancing_incidence(cls, val):
        """Warn if close to glancing incidence."""
        if np.abs(np.pi / 2 - val) < GLANCING_CUTOFF:
            log.warning(
                "Angled source propagation axis close to glancing angle. "
                "For best results, switch the injection axis."
            )
        return val

    # pylint: disable=no-member
    @cached_property
    def _dir_vector(self) -> Tuple[float, float, float]:
        """Source direction normal vector in cartesian coordinates."""
        radius = 1.0 if self.direction == "+" else -1.0
        dx = radius * np.cos(self.angle_phi) * np.sin(self.angle_theta)
        dy = radius * np.sin(self.angle_phi) * np.sin(self.angle_theta)
        dz = radius * np.cos(self.angle_theta)
        return self.unpop_axis(dz, (dx, dy), axis=self._injection_axis)

    @cached_property
    def _pol_vector(self) -> Tuple[float, float, float]:
        """Source polarization normal vector in cartesian coordinates."""
        normal_dir = [0.0, 0.0, 0.0]
        normal_dir[int(self._injection_axis)] = 1.0
        propagation_dir = list(self._dir_vector)
        if self.angle_theta == 0.0:
            pol_vector_p = np.array((0, 1, 0)) if self._injection_axis == 0 else np.array((1, 0, 0))
            pol_vector_p = self.rotate_points(pol_vector_p, normal_dir, angle=self.angle_phi)
        else:
            pol_vector_s = np.cross(normal_dir, propagation_dir)
            pol_vector_p = np.cross(propagation_dir, pol_vector_s)
            pol_vector_p = np.array(pol_vector_p) / np.linalg.norm(pol_vector_p)
        return self.rotate_points(pol_vector_p, propagation_dir, angle=self.pol_angle)


class ModeSource(DirectionalSource, PlanarSource, BroadbandSource):
    """Injects current source to excite modal profile on finite extent plane.

    Example
    -------
    >>> pulse = GaussianPulse(freq0=200e12, fwidth=20e12)
    >>> mode_spec = ModeSpec(target_neff=2.)
    >>> mode_source = ModeSource(
    ...     size=(10,10,0),
    ...     source_time=pulse,
    ...     mode_spec=mode_spec,
    ...     mode_index=1,
    ...     direction='-')
    """

    mode_spec: ModeSpec = pydantic.Field(
        ModeSpec(),
        title="Mode Specification",
        description="Parameters to feed to mode solver which determine modes measured by monitor.",
    )

    mode_index: pydantic.NonNegativeInt = pydantic.Field(
        0,
        title="Mode Index",
        description="Index into the collection of modes returned by mode solver. "
        " Specifies which mode to inject using this source. "
        "If larger than ``mode_spec.num_modes``, "
        "``num_modes`` in the solver will be set to ``mode_index + 1``.",
    )

    @cached_property
    def angle_theta(self):
        """Polar angle of propagation."""
        return self.mode_spec.angle_theta

    @cached_property
    def angle_phi(self):
        """Azimuth angle of propagation."""
        return self.mode_spec.angle_phi

    @cached_property
    def _dir_vector(self) -> Tuple[float, float, float]:
        """Source direction normal vector in cartesian coordinates."""
        radius = 1.0 if self.direction == "+" else -1.0
        dx = radius * np.cos(self.angle_phi) * np.sin(self.angle_theta)
        dy = radius * np.sin(self.angle_phi) * np.sin(self.angle_theta)
        dz = radius * np.cos(self.angle_theta)
        return self.unpop_axis(dz, (dx, dy), axis=self._injection_axis)

    @cached_property
    def _bend_axis(self) -> Axis:
        if self.mode_spec.bend_radius is None:
            return None
        in_plane = [0, 0]
        in_plane[self.mode_spec.bend_axis] = 1
        direction = self.unpop_axis(0, in_plane, axis=self.injection_axis)
        return direction.index(1)


""" Angled Field Sources one can use. """


class PlaneWave(AngledFieldSource, PlanarSource):
    """Uniform current distribution on an infinite extent plane. One element of size must be zero.

    Example
    -------
    >>> pulse = GaussianPulse(freq0=200e12, fwidth=20e12)
    >>> pw_source = PlaneWave(size=(inf,0,inf), source_time=pulse, pol_angle=0.1, direction='+')
    """


class GaussianBeam(AngledFieldSource, PlanarSource, BroadbandSource):
    """Guassian distribution on finite extent plane.

    Example
    -------
    >>> pulse = GaussianPulse(freq0=200e12, fwidth=20e12)
    >>> gauss = GaussianBeam(
    ...     size=(0,3,3),
    ...     source_time=pulse,
    ...     pol_angle=np.pi / 2,
    ...     direction='+',
    ...     waist_radius=1.0)
    """

    waist_radius: pydantic.PositiveFloat = pydantic.Field(
        1.0,
        title="Waist Radius",
        description="Radius of the beam at the waist.",
        units=MICROMETER,
    )

    waist_distance: float = pydantic.Field(
        0.0,
        title="Waist Distance",
        description="Distance from the beam waist along the propagation direction.",
        units=MICROMETER,
    )


class AstigmaticGaussianBeam(AngledFieldSource, PlanarSource, BroadbandSource):
    """This class implements the simple astigmatic Gaussian beam described in Kochkina et al.,
    Applied Optics, vol. 52, issue 24, 2013. The simple astigmatic Guassian distribution allows
    both an elliptical intensity profile and different waist locations for the two principal axes
    of the ellipse. When equal waist sizes and equal waist distances are specified in the two
    directions, this source becomes equivalent to :class:`GaussianBeam`.

    Example
    -------
    >>> pulse = GaussianPulse(freq0=200e12, fwidth=20e12)
    >>> gauss = AstigmaticGaussianBeam(
    ...     size=(0,3,3),
    ...     source_time=pulse,
    ...     pol_angle=np.pi / 2,
    ...     direction='+',
    ...     waist_sizes=(1.0, 2.0),
    ...     waist_distances = (3.0, 4.0))
    """

    waist_sizes: Tuple[pydantic.PositiveFloat, pydantic.PositiveFloat] = pydantic.Field(
        (1.0, 1.0),
        title="Waist sizes",
        description="Size of the beam at the waist in the local x and y directions.",
        units=MICROMETER,
    )

    waist_distances: Tuple[float, float] = pydantic.Field(
        (0.0, 0.0),
        title="Waist distances",
        description="Distance to the beam waist along the propagation direction "
        "for the waist sizes in the local x and y directions.",
        units=MICROMETER,
    )


class TFSF(AngledFieldSource, VolumeSource):
    """Total-field scattered-field (TFSF) source that can inject a plane wave in a finite region.
    The TFSF source injects 1 W / um^2 of power along the ``injection_axis``. Note that in the
    case of angled incidence, 1 W / um^2 is still injected along the source's ``injection_axis``,
    and not the propagation direction, unlike a ``PlaneWave`` source. This allows computing
    scattering and absorption cross sections without the need for additional normalization.
    """

    injection_axis: Axis = pydantic.Field(
        ...,
        title="Injection Axis",
        description="Specifies the injection axis. The plane of incidence is defined via this "
        "``injection_axis`` and the ``direction``. The popagation axis is defined with respect "
        "to the ``injection_axis`` by ``angle_theta`` and ``angle_phi``.",
    )

    @cached_property
    def _injection_axis(self):
        """Injection axis of the source."""
        return self.injection_axis

    @cached_property
    def injection_plane_center(self) -> Coordinate:
        """Center of the injection plane."""
        sign = 1 if self.direction == "-" else -1
        center = list(self.center)
        size = [0 if val == inf else val for val in self.size]
        center[self.injection_axis] += sign * size[self.injection_axis] / 2
        return tuple(center)

    def plot(  #  pylint:disable=too-many-arguments
        self,
        x: float = None,
        y: float = None,
        z: float = None,
        ax: Ax = None,
        **patch_kwargs,
    ) -> Ax:

        # call Source.plot but with the base of the arrow centered on the injection plane
        patch_kwargs["arrow_base"] = self.injection_plane_center
        ax = Source.plot(self, x=x, y=y, z=z, ax=ax, **patch_kwargs)
        return ax


# sources allowed in Simulation.sources
SourceType = Union[
    UniformCurrentSource,
    PointDipole,
    GaussianBeam,
    AstigmaticGaussianBeam,
    ModeSource,
    PlaneWave,
    CustomFieldSource,
    CustomCurrentSource,
    TFSF,
]
