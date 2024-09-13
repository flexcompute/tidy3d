"""Defines electric current sources for injecting light into simulation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union

import numpy as np
import pydantic.v1 as pydantic
from typing_extensions import Literal

from ..constants import GLANCING_CUTOFF, HERTZ, MICROMETER, RADIAN, inf
from ..exceptions import SetupError, ValidationError
from ..log import log
from .base import cached_property, skip_if_fields_missing
from .base_sim.source import AbstractSource
from .data.data_array import TimeDataArray
from .data.dataset import FieldDataset, ScalarFieldDataArray, TimeDataset
from .data.validators import validate_no_nans
from .geometry.base import Box
from .mode import ModeSpec
from .time import AbstractTimeDependence
from .types import (
    TYPE_TAG_STR,
    ArrayComplex1D,
    ArrayFloat1D,
    Ax,
    Axis,
    Coordinate,
    Direction,
    FreqBound,
    PlotVal,
    Polarization,
)
from .validators import (
    _assert_min_freq,
    assert_plane,
    assert_single_freq_in_range,
    assert_volumetric,
    warn_if_dataset_none,
)
from .viz import (
    ARROW_ALPHA,
    ARROW_COLOR_POLARIZATION,
    ARROW_COLOR_SOURCE,
    PlotParams,
    add_ax_if_none,
    plot_params_source,
)

# when checking if custom data spans the source plane, allow for a small tolerance
# due to numerical precision
DATA_SPAN_TOL = 1e-8
# width of Chebyshev grid used for broadband sources (in units of pulse width)
CHEB_GRID_WIDTH = 1.5
# Number of frequencies in a broadband source above which to issue a warning
WARN_NUM_FREQS = 20
# how many units of ``twidth`` from the ``offset`` until a gaussian pulse is considered "off"
END_TIME_FACTOR_GAUSSIAN = 10


class SourceTime(AbstractTimeDependence):
    """Base class describing the time dependence of a source."""

    @add_ax_if_none
    def plot_spectrum(
        self,
        times: ArrayFloat1D,
        num_freqs: int = 101,
        val: PlotVal = "real",
        ax: Ax = None,
    ) -> Ax:
        """Plot the complex-valued amplitude of the source time-dependence.
        Note: Only the real part of the time signal is used.

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

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.
        """

        fmin, fmax = self.frequency_range()
        return self.plot_spectrum_in_frequency_range(
            times, fmin, fmax, num_freqs=num_freqs, val=val, ax=ax
        )

    @abstractmethod
    def frequency_range(self, num_fwidth: float = 4.0) -> FreqBound:
        """Frequency range within plus/minus ``num_fwidth * fwidth`` of the central frequency."""

    @abstractmethod
    def end_time(self) -> float | None:
        """Time after which the source is effectively turned off / close to zero amplitude."""


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

    @property
    def twidth(self) -> float:
        """Width of pulse in seconds."""
        return 1.0 / (2 * np.pi * self.fwidth)

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

    remove_dc_component: bool = pydantic.Field(
        True,
        title="Remove DC Component",
        description="Whether to remove the DC component in the Gaussian pulse spectrum. "
        "If ``True``, the Gaussian pulse is modified at low frequencies to zero out the "
        "DC component, which is usually desirable so that the fields will decay. However, "
        "for broadband simulations, it may be better to have non-vanishing source power "
        "near zero frequency. Setting this to ``False`` results in an unmodified Gaussian "
        "pulse spectrum which can have a nonzero DC component.",
    )

    def amp_time(self, time: float) -> complex:
        """Complex-valued source amplitude as a function of time."""

        omega0 = 2 * np.pi * self.freq0
        time_shifted = time - self.offset * self.twidth

        offset = np.exp(1j * self.phase)
        oscillation = np.exp(-1j * omega0 * time)
        amp = np.exp(-(time_shifted**2) / 2 / self.twidth**2) * self.amplitude

        pulse_amp = offset * oscillation * amp

        # subtract out DC component
        if self.remove_dc_component:
            pulse_amp = pulse_amp * (1j + time_shifted / self.twidth**2 / omega0)
        else:
            # 1j to make it agree in large omega0 limit
            pulse_amp = pulse_amp * 1j

        return pulse_amp

    def end_time(self) -> float | None:
        """Time after which the source is effectively turned off / close to zero amplitude."""

        # TODO: decide if we should continue to return an end_time if the DC component remains
        # if not self.remove_dc_component:
        #     return None

        return self.offset * self.twidth + END_TIME_FACTOR_GAUSSIAN * self.twidth

    @property
    def amp_complex(self) -> complex:
        """Grab the complex amplitude from a ``GaussianPulse``."""
        phase = np.exp(1j * self.phase)
        return self.amplitude * phase

    @classmethod
    def from_amp_complex(cls, amp: complex, **kwargs) -> GaussianPulse:
        """Set the complex amplitude of a ``GaussianPulse``.

        Parameters
        ----------
        amp : complex
            Complex-valued amplitude to set in the returned ``GaussianPulse``.
        kwargs : dict
            Keyword arguments passed to ``GaussianPulse()``, excluding ``amplitude`` & ``phase``.
        """
        amplitude = abs(amp)
        phase = np.angle(amp)
        return cls(amplitude=amplitude, phase=phase, **kwargs)


class ContinuousWave(Pulse):
    """Source time dependence that ramps up to continuous oscillation
    and holds until end of simulation.

    Note
    ----
    Field decay will not occur, so the simulation will run for the full ``run_time``.
    Also, source normalization of frequency-domain monitors is not meaningful.

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

    def end_time(self) -> float | None:
        """Time after which the source is effectively turned off / close to zero amplitude."""
        return None


class CustomSourceTime(Pulse):
    """Custom source time dependence consisting of a real or complex envelope
    modulated at a central frequency, as shown below.

    Note
    ----
    .. math::

        amp\\_time(t) = amplitude \\cdot \\
                e^{i \\cdot phase - 2 \\pi i \\cdot freq0 \\cdot t} \\cdot \\
                envelope(t - offset / (2 \\pi \\cdot fwidth))

    Note
    ----
    Depending on the envelope, field decay may not occur.
    If field decay does not occur, then the simulation will run for the full ``run_time``.
    Also, if field decay does not occur, then source normalization of frequency-domain
    monitors is not meaningful.

    Note
    ----
    The source time dependence is linearly interpolated to the simulation time steps.
    The sampling rate should be sufficiently fast that this interpolation does not
    introduce artifacts. The source time dependence should also start at zero and ramp up smoothly.
    The first and last values of the envelope will be used for times that are out of range
    of the provided data.

    Example
    -------
    >>> cst = CustomSourceTime.from_values(freq0=1, fwidth=0.1,
    ...     values=np.linspace(0, 9, 10), dt=0.1)

    """

    offset: float = pydantic.Field(
        0.0,
        title="Offset",
        description="Time delay of the envelope in units of 1 / (``2pi * fwidth``).",
    )

    source_time_dataset: Optional[TimeDataset] = pydantic.Field(
        ...,
        title="Source time dataset",
        description="Dataset for storing the envelope of the custom source time. "
        "This envelope will be modulated by a complex exponential at frequency ``freq0``.",
    )

    _no_nans_dataset = validate_no_nans("source_time_dataset")
    _source_time_dataset_none_warning = warn_if_dataset_none("source_time_dataset")

    @pydantic.validator("source_time_dataset", always=True)
    def _more_than_one_time(cls, val):
        """Must have more than one time to interpolate."""
        if val is None:
            return val
        if val.values.size <= 1:
            raise ValidationError("'CustomSourceTime' must have more than one time coordinate.")
        return val

    @classmethod
    def from_values(
        cls, freq0: float, fwidth: float, values: ArrayComplex1D, dt: float
    ) -> CustomSourceTime:
        """Create a :class:`.CustomSourceTime` from a numpy array.

        Parameters
        ----------
        freq0 : float
            Central frequency of the source. The envelope provided will be modulated
            by a complex exponential at this frequency.
        fwidth : float
            Estimated frequency width of the source.
        values: ArrayComplex1D
            Complex values of the source envelope.
        dt: float
            Time step for the ``values`` array. This value should be sufficiently small
            that the interpolation to simulation time steps does not introduce artifacts.

        Returns
        -------
        CustomSourceTime
            :class:`.CustomSourceTime` with envelope given by ``values``, modulated by a complex
            exponential at frequency ``freq0``. The time coordinates are evenly spaced
            between ``0`` and ``dt * (N-1)`` with a step size of ``dt``, where ``N`` is the length of
            the values array.
        """

        times = np.arange(len(values)) * dt
        source_time_dataarray = TimeDataArray(values, coords=dict(t=times))
        source_time_dataset = TimeDataset(values=source_time_dataarray)
        return CustomSourceTime(
            freq0=freq0,
            fwidth=fwidth,
            source_time_dataset=source_time_dataset,
        )

    @property
    def data_times(self) -> ArrayFloat1D:
        """Times of envelope definition."""
        if self.source_time_dataset is None:
            return []
        data_times = self.source_time_dataset.values.coords["t"].values.squeeze()
        return data_times

    def _all_outside_range(self, run_time: float) -> bool:
        """Whether all times are outside range of definition."""

        # can't validate if data isn't loaded
        if self.source_time_dataset is None:
            return False

        # make time a numpy array for uniform handling
        data_times = self.data_times

        # shift time
        twidth = 1.0 / (2 * np.pi * self.fwidth)
        max_time_shifted = run_time - self.offset * twidth
        min_time_shifted = -self.offset * twidth

        return (max_time_shifted < min(data_times)) | (min_time_shifted > max(data_times))

    def amp_time(self, time: float) -> complex:
        """Complex-valued source amplitude as a function of time.

        Parameters
        ----------
        time : float
            Time in seconds.

        Returns
        -------
        complex
            Complex-valued source amplitude at that time.
        """

        if self.source_time_dataset is None:
            return None

        # make time a numpy array for uniform handling
        times = np.array([time] if isinstance(time, (int, float)) else time)
        data_times = self.data_times

        # shift time
        twidth = 1.0 / (2 * np.pi * self.fwidth)
        time_shifted = times - self.offset * twidth

        # mask times that are out of range
        mask = (time_shifted < min(data_times)) | (time_shifted > max(data_times))

        # get envelope
        envelope = np.zeros(len(time_shifted), dtype=complex)
        values = self.source_time_dataset.values
        envelope[mask] = values.sel(t=time_shifted[mask], method="nearest").to_numpy()
        if not all(mask):
            envelope[~mask] = values.interp(t=time_shifted[~mask]).to_numpy()

        # modulation, phase, amplitude
        omega0 = 2 * np.pi * self.freq0
        offset = np.exp(1j * self.phase)
        oscillation = np.exp(-1j * omega0 * times)
        amp = self.amplitude

        return offset * oscillation * amp * envelope

    def end_time(self) -> float | None:
        """Time after which the source is effectively turned off / close to zero amplitude."""

        if self.source_time_dataset is None:
            return None

        data_array = self.source_time_dataset.values

        t_coords = data_array.coords["t"]
        source_is_non_zero = ~np.isclose(abs(data_array), 0)
        t_non_zero = t_coords[source_is_non_zero]

        return np.max(t_non_zero)


SourceTimeType = Union[GaussianPulse, ContinuousWave, CustomSourceTime]

""" Source objects """


class Source(Box, AbstractSource, ABC):
    """Abstract base class for all sources."""

    source_time: SourceTimeType = pydantic.Field(
        ...,
        title="Source Time",
        description="Specification of the source time-dependence.",
        discriminator=TYPE_TAG_STR,
    )

    @cached_property
    def plot_params(self) -> PlotParams:
        """Default parameters for plotting a Source object."""
        return plot_params_source

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

    @pydantic.validator("source_time", always=True)
    def _freqs_lower_bound(cls, val):
        """Raise validation error if central frequency is too low."""
        _assert_min_freq(val.freq0, msg_start="'source_time.freq0'")
        return val

    def plot(
        self,
        x: float = None,
        y: float = None,
        z: float = None,
        ax: Ax = None,
        **patch_kwargs,
    ) -> Ax:
        """Plot this source."""

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

    confine_to_bounds: bool = pydantic.Field(
        False,
        title="Confine to Analytical Bounds",
        description="If ``True``, any source amplitudes which, after discretization, fall beyond "
        "the bounding box of the source are zeroed out, but only along directions where "
        "the source has a non-zero extent. The bounding box is inclusive. Should be set ```True`` "
        "when the current source is being used to excite a current in a conductive material.",
    )


class UniformCurrentSource(CurrentSource, ReverseInterpolatedSource):
    """Source in a rectangular volume with uniform time dependence.

    Notes
    -----

        Inputting the parameter ``size=(0,0,0)`` defines the equivalent of a point source.

    Example
    -------
    >>> pulse = GaussianPulse(freq0=200e12, fwidth=20e12)
    >>> pt_source = UniformCurrentSource(size=(0,0,0), source_time=pulse, polarization='Ex')
    """


class PointDipole(CurrentSource, ReverseInterpolatedSource):
    """Uniform current source with a zero size. The source corresponds to an infinitesimal antenna
    with a fixed current density, and is slightly different from a related definition that is used
    in some contexts, namely an oscillating electric or magnetic dipole. The two are related through
    a factor of ``omega ** 2`` in the power normalization, where ``omega`` is the angular frequency
    of the oscillation. This is discussed further in our
    `source normalization <../../faq/docs/faq/How-are-results-normalized.html>`_ FAQ page.

    ..
        TODO add image of how it looks like based on sim 1.

    Example
    -------
    >>> pulse = GaussianPulse(freq0=200e12, fwidth=20e12)
    >>> pt_dipole = PointDipole(center=(1,2,3), source_time=pulse, polarization='Ex')

    See Also
    --------

    **Notebooks**
        * `Particle swarm optimization of quantum emitter light extraction to free space <../../notebooks/BullseyeCavityPSO.html>`_
        * `Adjoint optimization of quantum emitter light extraction to an integrated waveguide <../../notebooks/AdjointPlugin12LightExtractor.html>`_
    """

    size: Tuple[Literal[0], Literal[0], Literal[0]] = pydantic.Field(
        (0, 0, 0),
        title="Size",
        description="Size in x, y, and z directions, constrained to ``(0, 0, 0)``.",
        units=MICROMETER,
    )


class CustomCurrentSource(ReverseInterpolatedSource):
    """Implements a source corresponding to an input dataset containing ``E`` and ``H`` fields.

    Notes
    -----

        Injects the specified components of the ``E`` and ``H`` dataset directly as ``J`` and ``M`` current
        distributions in the FDTD solver. The coordinates of all provided fields are assumed to be relative to the
        source center.

        The syntax is very similar to :class:`CustomFieldSource`, except instead of a ``field_dataset``, the source
        accepts a :attr:`current_dataset`. This dataset still contains :math:`E_{x,y,z}` and :math:`H_{x,y,
        z}` field components, which correspond to :math:`J` and :math:`M` components respectively. There are also
        fewer constraints on the data requirements for :class:`CustomCurrentSource`. It can be volumetric or planar
        without requiring tangential components. Finally, note that the dataset is still defined w.r.t. the source
        center, just as in the case of the :class:`CustomFieldSource`, and can then be placed anywhere in the simulation.

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

    See Also
    --------

    **Notebooks**
        * `Defining spatially-varying sources <../../notebooks/CustomFieldSource.html>`_
    """

    current_dataset: Optional[FieldDataset] = pydantic.Field(
        ...,
        title="Current Dataset",
        description=":class:`.FieldDataset` containing the desired frequency-domain "
        "electric and magnetic current patterns to inject.",
    )

    _no_nans_dataset = validate_no_nans("current_dataset")
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

        if val >= WARN_NUM_FREQS:
            log.warning(
                f"A large number ({val}) of frequency points is used in a broadband source. "
                "This can lead to solver slow-down and increased cost, and even introduce "
                "numerical noise. This may become a hard limit in future Tidy3D versions.",
                custom_loc=["num_freqs"],
            )

        return val


""" Source current profiles determined by user-supplied data on a plane."""


class CustomFieldSource(FieldSource, PlanarSource):
    """Implements a source corresponding to an input dataset containing ``E`` and ``H`` fields,
    using the equivalence principle to define the actual injected currents.

     Notes
     -----

        For the injection to work as expected (i.e. to reproduce the required ``E`` and ``H`` fields),
        the field data must decay by the edges of the source plane, or the source plane must span the entire
        simulation domain and the fields must match the simulation boundary conditions.

        The equivalent source currents are fully defined by the field components tangential to the
        source plane. For e.g. source normal along ``z``, the normal components (``Ez`` and ``Hz``)
        can be provided but will have no effect on the results, and at least one of the tangential
        components has to be in the dataset, i.e. at least one of ``Ex``, ``Ey``, ``Hx``, and ``Hy``.

        .. TODO add image here

        ..
            TODO is this generic? Only the field components tangential to the custom source plane are needed and used
            in the simulation. Due to the equivalence principle, these fully define the currents that need to be
            injected. This is not to say that the normal components of the data (:math:`E_x`, :math:`H_x` in our example)
            is lost or not injected. It is merely not needed as it can be uniquely obtained using the tangential components.

        ..
            TODO add example for this standalone
            Source data can be imported from file just as shown here, after the data is imported as a numpy array using
            standard numpy functions like loadtxt.

        If the data is not coming from a ``tidy3d`` simulation, the normalization is likely going to be arbitrary and
        the directionality of the source will likely not be perfect, even if both the ``E`` and ``H`` fields are
        provided. An empty normalizing run may be needed to accurately normalize results.

        To create this empty simulation it is recommended that users create a simulation with no structures but just a flux
        monitor (``tidy3D.FluxMonitor``) next to the custom source, ensuring that the flux monitor is at least one grid cell
        away from the source. Moreover, for accurate normalization, users must ensure that the same grid is used to run
        the original simulation as well as the empty simulation. The total flux calculated at the flux monitor of the empty
        simulation can then be used for proper normalization of results after ``tidy3d`` simulation.

        The coordinates of all provided fields are assumed to be relative to the source center.
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

    Creating an empty simulation with no structures with ``FluxMonitor`` for normalization but with the same grid as the
    original simulation.

    Example
    -------

    >>> sim_empty = sim.updated_copy(monitors = [Flux_monitor],  # doctest: +SKIP
    ...             structures = [],
    ...             grid_spec= sim.grid_spec.updated_copy(override_structures = sim.structures)
    ...             )

    See Also
    --------

    **Notebooks**
        * `Defining spatially-varying sources <../../notebooks/CustomFieldSource.html>`_
    """

    field_dataset: Optional[FieldDataset] = pydantic.Field(
        ...,
        title="Field Dataset",
        description=":class:`.FieldDataset` containing the desired frequency-domain "
        "fields patterns to inject. At least one tangential field component must be specified.",
    )

    _no_nans_dataset = validate_no_nans("field_dataset")
    _field_dataset_none_warning = warn_if_dataset_none("field_dataset")
    _field_dataset_single_freq = assert_single_freq_in_range("field_dataset")

    @pydantic.validator("field_dataset", always=True)
    @skip_if_fields_missing(["size"])
    def _tangential_component_defined(cls, val: FieldDataset, values: dict) -> FieldDataset:
        """Assert that at least one tangential field component is provided."""
        if val is None:
            return val
        size = values.get("size")
        normal_axis = size.index(0.0)
        _, (cmp1, cmp2) = cls.pop_axis("xyz", axis=normal_axis)
        for field in "EH":
            for cmp_name in (cmp1, cmp2):
                tangential_field = field + cmp_name
                if tangential_field in val.field_components:
                    return val
        raise SetupError("No tangential field found in the suppled 'field_dataset'.")

    @pydantic.validator("field_dataset", always=True)
    def _check_fields_interpolate(cls, val: FieldDataset) -> FieldDataset:
        """Checks whether the filds in 'field_dataset' can be interpolated."""
        if isinstance(val, FieldDataset):
            for name, data in val.field_components.items():
                if isinstance(data, ScalarFieldDataArray):
                    data._interp_validator(name)
        return val


""" Source current profiles defined by (1) angle or (2) desired mode. Sets theta and phi angles."""


class AngledFieldSource(DirectionalSource, ABC):
    """A FieldSource defined with an angled direction of propagation.

    Notes
    -----

        The direction is defined by
        the polar and azimuth angles w.r.t. an injection axis, as well as forward ``+`` or
        backward ``-``. This base class only defines the :attr:`direction` and :attr:`injection_axis`
        attributes, but it must be composed with a class that also defines :attr:`angle_theta` and
        :attr:`angle_phi`.

    """

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
                "For best results, switch the injection axis.",
                custom_loc=["angle_theta"],
            )
        return val

    @cached_property
    def _dir_vector(self) -> Tuple[float, float, float]:
        """Source direction normal vector in cartesian coordinates."""

        # Propagation vector assuming propagation along z
        radius = 1.0 if self.direction == "+" else -1.0
        dx = radius * np.cos(self.angle_phi) * np.sin(self.angle_theta)
        dy = radius * np.sin(self.angle_phi) * np.sin(self.angle_theta)
        dz = radius * np.cos(self.angle_theta)

        # Move to original injection axis
        return self.unpop_axis(dz, (dx, dy), axis=self._injection_axis)

    @cached_property
    def _pol_vector(self) -> Tuple[float, float, float]:
        """Source polarization normal vector in cartesian coordinates."""

        # Polarization vector assuming propagation along z
        pol_vector_z_normal = np.array([1.0, 0.0, 0.0])

        # Rotate polarization
        pol_vector_z_normal = self.rotate_points(
            pol_vector_z_normal, axis=[0, 0, 1], angle=self.pol_angle
        )

        # Rotate the fields back to the original propagation axes
        pol_vector_z_normal = self.rotate_points(
            pol_vector_z_normal, axis=[0, 1, 0], angle=self.angle_theta
        )
        pol_vector_z_normal = self.rotate_points(
            pol_vector_z_normal, axis=[0, 0, 1], angle=self.angle_phi
        )

        # Move to original injection axis
        pol_vector = self.unpop_axis(
            pol_vector_z_normal[2], pol_vector_z_normal[:2], axis=self._injection_axis
        )
        return pol_vector


class ModeSource(DirectionalSource, PlanarSource, BroadbandSource):
    """Injects current source to excite modal profile on finite extent plane.

    Notes
    -----

        Using this mode source, it is possible selectively excite one of the guided modes of a waveguide. This can be
        computed in our eigenmode solver :class:`tidy3d.plugins.mode.ModeSolver` and implement the mode simulation in
        FDTD.

        Mode sources are normalized to inject exactly 1W of power at the central frequency.

        The modal source allows you to do directional excitation. Illustrated
        by the image below, the field is perfectly launched to the right of the source and there's zero field to the
        left of the source. Now you can contrast the behavior of the modal source with that of a dipole source. If
        you just put a dipole into the waveguide, well, you see quite a bit different in the field distribution.
        First of all, the dipole source is not directional launching. It launches waves in both directions. The
        second is that the polarization of the dipole is set to selectively excite a TE mode. But it takes some
        propagation distance before the mode settles into a perfect TE mode profile. During this process,
        there is radiation into the substrate.

        .. image:: ../../_static/img/mode_vs_dipole_source.png

        .. TODO improve links to other APIs functionality here.

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

    See Also
    --------

    :class:`tidy3d.plugins.mode.ModeSolver`:
        Interface for solving electromagnetic eigenmodes in a 2D plane with translational invariance in the third dimension.

    **Notebooks:**
        * `Waveguide Y junction <../../notebooks/YJunction.html>`_
        * `90 degree optical hybrid <../../notebooks/90OpticalHybrid.html>`_

    **Lectures:**
        * `Prelude to Integrated Photonics Simulation: Mode Injection <https://www.flexcompute.com/fdtd101/Lecture-4-Prelude-to-Integrated-Photonics-Simulation-Mode-Injection/>`_
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

    See Also
    --------

    **Notebooks:**
        * `How to troubleshoot a diverged FDTD simulation <../../notebooks/DivergedFDTDSimulation.html>`_

    **Lectures:**
        * `Using FDTD to Compute a Transmission Spectrum <https://www.flexcompute.com/fdtd101/Lecture-2-Using-FDTD-to-Compute-a-Transmission-Spectrum/>`__
    """


class GaussianBeam(AngledFieldSource, PlanarSource, BroadbandSource):
    """Gaussian distribution on finite extent plane.

    Example
    -------
    >>> pulse = GaussianPulse(freq0=200e12, fwidth=20e12)
    >>> gauss = GaussianBeam(
    ...     size=(0,3,3),
    ...     source_time=pulse,
    ...     pol_angle=np.pi / 2,
    ...     direction='+',
    ...     waist_radius=1.0)

    Notes
    --------
    If one wants the focus 'in front' of the source, a negative value of ``beam_distance`` is needed.

    .. image:: ../../_static/img/beam_waist.png
        :width: 30%
        :align: center

    See Also
    --------

    **Notebooks**:
        * `Inverse taper edge coupler <../../notebooks/EdgeCoupler.html>`_
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
        description="Distance from the beam waist along the propagation direction. "
        "A positive value means the waist is positioned behind the source, considering the propagation direction. "
        "For example, for a beam propagating in the ``+`` direction, a positive value of ``beam_distance`` "
        "means the beam waist is positioned in the ``-`` direction (behind the source). "
        "A negative value means the beam waist is in the ``+`` direction (in front of the source). "
        "For an angled source, the distance is defined along the rotated propagation direction.",
        units=MICROMETER,
    )


class AstigmaticGaussianBeam(AngledFieldSource, PlanarSource, BroadbandSource):
    """The simple astigmatic Gaussian distribution allows
    both an elliptical intensity profile and different waist locations for the two principal axes
    of the ellipse. When equal waist sizes and equal waist distances are specified in the two
    directions, this source becomes equivalent to :class:`GaussianBeam`.

    Notes
    -----

        This class implements the simple astigmatic Gaussian beam described in _`[1]`.

        **References**:

        .. [1] Kochkina et al., Applied Optics, vol. 52, issue 24, 2013.

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
        "for the waist sizes in the local x and y directions. "
        "When ``direction`` is ``+`` and ``waist_distances`` are positive, the waist "
        "is on the ``-`` side (behind) the source plane. When ``direction`` is ``+`` and "
        "``waist_distances`` are negative, the waist is on the ``+`` side (in front) of "
        "the source plane.",
        units=MICROMETER,
    )


class TFSF(AngledFieldSource, VolumeSource):
    """Total-field scattered-field (TFSF) source that can inject a plane wave in a finite region.

    Notes
    -----

        The TFSF source injects :math:`1 W` of power per :math:`\\mu m^2` of source area along the :attr:`injection_axis`.
        Hence, the normalization for the incident field is :math:`|E_0|^2 = \\frac{2}{c\\epsilon_0}`, for any source size.
        Note that in the case of angled incidence, the same power is injected along the source's :attr:`injection_axis`,
        and not the propagation direction. This allows computing scattering and absorption cross-sections
        without the need for additional normalization.

        The TFSF source allows specifying a box region into which a plane wave is injected. Fields inside this region
        can be interpreted as the superposition of the incident field and the scattered field due to any scatterers
        present in the simulation domain. The fields at the edges of the TFSF box are modified at each time step such
        that the incident field is cancelled out, so that all fields outside the TFSF box are scattered fields only.
        This is useful in scenarios where one is interested in computing scattered fields only, for example when
        computing scattered cross-sections of various objects.

        It is important to note that when a non-uniform grid is used in the directions transverse to the
        :attr:`injection_axis` of the TFSF source, the suppression of the incident field outside the TFSF box may not be as
        close to zero as in the case of a uniform grid. Because of this, a warning may be issued when nonuniform grid
        TFSF setup is detected. In some cases, however, the accuracy may be only weakly affected, and the warnings
        can be ignored.

    See Also
    --------

    **Notebooks**:
        * `Defining a total-field scattered-field (TFSF) plane wave source <../../notebooks/TFSF.html>`_
        * `Nanoparticle Scattering <../../notebooks/PlasmonicNanoparticle.html>`_: To force a uniform grid in the TFSF region and avoid the warnings, a mesh override structure can be used as illustrated here.
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

    def plot(
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
