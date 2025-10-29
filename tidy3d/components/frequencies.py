"""Frequency utilities class ``FrequencyUtils`` and utility class ``FreqRange`` for frequency and wavelength handling."""

from __future__ import annotations

from typing import Any

import numpy as np
import pydantic as pd
import pydantic.v1 as pydantic
from numpy.typing import NDArray

from tidy3d.components.base import Tidy3dBaseModel
from tidy3d.components.source.time import GaussianPulse
from tidy3d.constants import C_0

O_BAND = (1.260, 1.360)
E_BAND = (1.360, 1.460)
S_BAND = (1.460, 1.530)
C_BAND = (1.530, 1.565)
L_BAND = (1.565, 1.625)
U_BAND = (1.625, 1.675)


class FrequencyUtils(Tidy3dBaseModel):
    """Utilities for classifying frequencies/wavelengths and generating samples for standard optical bands."""

    use_wavelength: bool = pd.Field(
        False,
        title="Use wavelength",
        description="Indicate whether to use wavelengths instead of frequencies for the return "
        "values of functions and parameters.",
    )

    def classification(self, value: float) -> tuple[str]:
        """Band classification for a given frequency/wavelength.

        Frequency values must be given in hertz (Hz). Wavelengths must be
        given in micrometers (μm).

        Parameters
        ----------
        value : float
            Value to classify.

        Returns
        -------
        tuple[str]
            String tuple with classification.
        """
        if self.use_wavelength:
            value = C_0 / value
        if value < 3:
            return ("near static",)
        if value < 300e6:
            if value < 30:
                return ("radio wave", "ELF")
            if value < 300:
                return ("radio wave", "SLF")
            if value < 3e3:
                return ("radio wave", "ULF")
            if value < 30e3:
                return ("radio wave", "VLF")
            if value < 300e3:
                return ("radio wave", "LF")
            if value < 3e6:
                return ("radio wave", "MF")
            if value < 30e6:
                return ("radio wave", "HF")
            return ("radio wave", "VHF")
        if value < 300e9:
            if value < 3e9:
                return ("microwave", "UHF")
            if value < 30e9:
                return ("microwave", "SHF")
            return ("microwave", "EHF")
        if value < 400e12:
            if value < 6e12:
                return ("infrared", "FIR")
            if value < 100e12:
                return ("infrared", "MIR")
            return ("infrared", "NIR")
        if value < 790e12:
            if value < 480e12:
                return ("visible", "red")
            if value < 510e12:
                return ("visible", "orange")
            if value < 530e12:
                return ("visible", "yellow")
            if value < 600e12:
                return ("visible", "green")
            if value < 620e12:
                return ("visible", "cyan")
            if value < 670e12:
                return ("visible", "blue")
            return ("visible", "violet")
        if value < 30e15:
            if value < 1e15:
                return ("ultraviolet", "NUV")
            if value < 1.5e15:
                return ("ultraviolet", "MUV")
            if value < 2.47e15:
                return ("ultraviolet", "FUV")
            return ("ultraviolet", "EUV")
        if value < 30e18:
            if value < 3e18:
                return ("X-ray", "soft X-ray")
            return ("X-ray", "hard X-ray")
        return ("γ-ray",)

    def o_band(self, n: int = 11) -> list[float]:
        """
        Optical O band frequencies/wavelengths sorted by wavelength.

        The returned samples are equally spaced in wavelength.

        Parameters
        ----------
        n : int
            Desired number of samples.

        Returns
        -------
        list[float]
            Samples list.
        """
        values = np.linspace(*O_BAND, n)
        if not self.use_wavelength:
            values = C_0 / values
        return values.tolist()

    def e_band(self, n: int = 11) -> list[float]:
        """
        Optical E band frequencies/wavelengths sorted by wavelength.

        The returned samples are equally spaced in wavelength.

        Parameters
        ----------
        n : int
            Desired number of samples.

        Returns
        -------
        list[float]
            Samples list.
        """
        values = np.linspace(*E_BAND, n)
        if not self.use_wavelength:
            values = C_0 / values
        return values.tolist()

    def s_band(self, n: int = 15) -> list[float]:
        """
        Optical S band frequencies/wavelengths sorted by wavelength.

        The returned samples are equally spaced in wavelength.

        Parameters
        ----------
        n : int
            Desired number of samples.

        Returns
        -------
        list[float]
            Samples list.
        """
        values = np.linspace(*S_BAND, n)
        if not self.use_wavelength:
            values = C_0 / values
        return values.tolist()

    def c_band(self, n: int = 8) -> list[float]:
        """
        Optical C band frequencies/wavelengths sorted by wavelength.

        The returned samples are equally spaced in wavelength.

        Parameters
        ----------
        n : int
            Desired number of samples.

        Returns
        -------
        list[float]
            Samples list.
        """
        values = np.linspace(*C_BAND, n)
        if not self.use_wavelength:
            values = C_0 / values
        return values.tolist()

    def l_band(self, n: int = 13) -> list[float]:
        """
        Optical L band frequencies/wavelengths sorted by wavelength.

        The returned samples are equally spaced in wavelength.

        Parameters
        ----------
        n : int
            Desired number of samples.

        Returns
        -------
        list[float]
            Samples list.
        """
        values = np.linspace(*L_BAND, n)
        if not self.use_wavelength:
            values = C_0 / values
        return values.tolist()

    def u_band(self, n: int = 11) -> list[float]:
        """
        Optical U band frequencies/wavelengths sorted by wavelength.

        The returned samples are equally spaced in wavelength.

        Parameters
        ----------
        n : int
            Desired number of samples.

        Returns
        -------
        list[float]
            Samples list.
        """
        values = np.linspace(*U_BAND, n)
        if not self.use_wavelength:
            values = C_0 / values
        return values.tolist()


frequencies = FrequencyUtils(use_wavelength=False)
wavelengths = FrequencyUtils(use_wavelength=True)

frequencies.__doc__ = (
    "Frequency utilities configured to interpret and return values in hertz (Hz). "
    "Use for RF, microwave, optical, and other band classifications in frequency units."
)

wavelengths.__doc__ = (
    "Frequency utilities configured to interpret and return values in micrometers (μm). "
    "Use for optical and photonic calculations where wavelength units are preferred."
)


class FreqRange(Tidy3dBaseModel):
    """
    Convenience class for handling frequency/wavelength conversion; it simplifies specification
    of frequency ranges and sample points for sources and monitors.

    Notes
    -----
        Depending on the context the user can define desired frequency range by specifying:
        - central frequency ``freq0`` and frequency bandwidth ``fwidth``;
        - frequency interval [``fmin``,``fmax``];
        - central wavelength ``wvl0`` and wavelength range ``wvl_width``;
        - wavelength interval [``wvl_min``, ``wvl_max``].

    Example
    -------
    >>> import tidy3d as td
    >>> freq0  = 1e12
    >>> fwidth = 1e11
    >>> freq_range = td.FreqRange(freq0=freq0, fwidth=fwidth)
    >>> central_freq = freq_range.freqs(num_points=1, spacing="uniform_freq")
    >>> freqs = freq_range.freqs(num_points=11, spacing="uniform_wvl")
    >>> source = freq_range.to_gaussian_pulse()
    """

    freq0: pydantic.PositiveFloat = pydantic.Field(
        ...,
        title="Central frequency",
        description="Real-valued positive central frequency.",
        units="Hz",
    )

    fwidth: pydantic.PositiveFloat = pydantic.Field(
        ...,
        title="Frequency bandwidth",
        description="Real-valued positive width of the frequency range (bandwidth).",
        units="Hz",
    )

    @pydantic.root_validator
    def check_half_fwidth_less_than_freq0(cls, values):
        freq0 = values.get("freq0")
        fwidth = values.get("fwidth")
        if freq0 is not None and fwidth is not None:
            if (fwidth / 2) >= freq0:
                raise ValueError(
                    "Frequency bandwidth `fwidth` must be strictly less than `2 * freq0`."
                )
        return values

    @property
    def fmin(self) -> float:
        """Infer lowest frequency ``fmin`` from central frequency ``freq0`` and bandwidth ``fwidth``."""
        return self.freq0 - self.fwidth / 2.0

    @property
    def fmax(self) -> float:
        """Infer highest frequency ``fmax`` from central frequency ``freq0`` and bandwidth ``fwidth``."""
        return self.freq0 + self.fwidth / 2.0

    @property
    def wvl0(self) -> float:
        """Get central wavelength from central frequency and bandwidth."""
        wvl_min = C_0 / self.fmax
        wvl_max = C_0 / self.fmin
        return 0.5 * (wvl_min + wvl_max)

    @classmethod
    def from_freq_interval(cls, fmin: float, fmax: float) -> FreqRange:
        """
        method ``from_freq_interval()`` creates instance of class ``FreqRange`` from frequency interval
        defined by arguments  ``fmin`` and ``fmax``.

        NB: central frequency never corresponds to central wavelength!
        ``freq0 = (fmin + fmax) / 2`` implies that ``wvl0 != (wvl_min + wvl_max) / 2`` and vise-versa.

        Parameters
        ----------
        fmin : float
            Lower bound of frequency of interest.
        fmax : float
            Upper bound of frequency of interest.

        Returns
        -------
        FreqRange
            An instance of ``FreqRange`` defined by frequency interval [``fmin``, ``fmax``].
        """

        # extract frequency-related info
        freq0 = 0.5 * (fmax + fmin)  # extract central freq
        fwidth = fmax - fmin  # extract bandwidth
        return cls(freq0=freq0, fwidth=fwidth)

    @classmethod
    def from_wvl(cls, wvl0: float, wvl_width: float) -> FreqRange:
        """
        method ``from_wvl()`` updated instance of class ``FreqRange`` by reassigning new
        frequency- and wavelength-related parameters.

        NB: central frequency never corresponds to central wavelength!
        ``wvl0 = (wvl_min + wvl_max) / 2`` implies that ``freq0 != (fmin + fmax) / 2`` and vise versa.

        Parameters
        ----------
        wvl0 : float
            Real-valued central wavelength.
        wvl_width : float
            Real-valued wavelength range.

        Returns
        -------
        FreqRange
            An instance of ``FreqRange`` defined by central wavelength ``wvl0`` and wavelength range ``wvl_width``.
        """

        # calculate lowest and highest frequencies
        fmin = C_0 / (wvl0 + wvl_width / 2.0)
        fmax = C_0 / (wvl0 - wvl_width / 2.0)

        return cls.from_freq_interval(fmin=fmin, fmax=fmax)

    @classmethod
    def from_wvl_interval(cls, wvl_min: float, wvl_max: float) -> FreqRange:
        """
        method ``from_wvl_interval()`` updated instance of class ``FreqRange`` by reassigning new
        frequency- and wavelength-related parameters.

        NB: central frequency never corresponds to central wavelength!
        ``wvl0 = (wvl_min + wvl_max) / 2`` implies that ``freq0 != (fmin + fmax) / 2``.

        Parameters
        ----------
        wvl_min : float
            The lowest wavelength of interest.
        wvl_max : float
            The longest wavelength of interest.

        Returns
        -------
        FreqRange
            An instance of ``FreqRange`` defined by the wavelength interval [``wvl_min``, ``wvl_max``].
        """

        # convert wavelength intervals to frequency interval
        fmax = C_0 / wvl_min
        fmin = C_0 / wvl_max

        return cls.from_freq_interval(fmin=fmin, fmax=fmax)

    def freqs(self, num_points: int, spacing: str = "uniform_freq") -> NDArray[np.float64]:
        """
        method ``freqs()`` returns a numpy array of ``num_point`` frequencies uniformly
        sampled from the specified frequency range;
        if ``num_points == 1`` method returns the central frequency ``freq0``.

        Parameters
        ----------
        num_points : int
            Number of frequency points in a frequency range of interest.
        spacing: str = "uniform_freq"
            Mode of frequency sampling.
            If spacing is set to ``uniform_freq``, frequencies are sampled uniformly over the frequency interval.
            If set to ``uniform_wvl``, frequencies correspond to uniformly sampled wavelengths over the wavelength interval.
            Frequencies are sorted in ascending order.

        Returns
        -------
        np.ndarray
            a numpy array of uniformly distributed frequency samples in a frequency range of interest.
        """
        if spacing == "uniform_wvl":
            return (
                np.array([C_0 / self.wvl0])
                if num_points == 1
                else np.sort(C_0 / self.wvls(num_points=num_points, spacing=spacing))
            )

        elif spacing == "uniform_freq":
            return (
                np.array([self.freq0])
                if num_points == 1
                else np.linspace(self.fmin, self.fmax, num_points)
            )

        else:
            raise ValueError(
                "Invalid `spacing` value in FreqRange.freqs(): expected 'uniform_freq' or 'uniform_wvl'. "
                f"Received: {spacing!r}. Please provide a valid spacing type."
            )

    def sweep_decade(self, num_points_per_decade: int) -> NDArray[np.float64]:
        """
        Generate frequencies with logarithmic spacing across decades.

        Notes
        -----
        This method creates a logarithmically spaced frequency sweep.
        It is analogous to the SPICE AC analysis command for a decade
        sweep:  ``.ac dec <num_points_per_decade> <fmin> <fmax>``

        Parameters
        ----------
        num_points_per_decade : int
            Number of frequency points per decade. Must be strictly positive.

        Returns
        -------
        NDArray[np.float64]
            Array of frequencies with logarithmic spacing across decades.

        Raises
        ------
        ValueError
            If ``num_points_per_decade`` is not positive, or if frequency range is invalid.

        Examples
        --------
        >>> import tidy3d as td
        >>> freq_range = td.FreqRange.from_freq_interval(1e3, 1e6)  # 1 kHz to 1 MHz
        >>> freqs = freq_range.sweep_decade(10)  # 10 points per decade
        """
        # Input validation
        if num_points_per_decade <= 0:
            raise ValueError(
                f"'num_points_per_decade' must be strictly positive, got {num_points_per_decade}."
            )

        # Calculate logarithmic range
        fstart = np.log10(self.fmin)
        fend = np.log10(self.fmax)
        num_decades = fend - fstart

        # Calculate total number of points
        # Add 1 to include the endpoint, ensuring we cover the full range
        num_total_points = int(np.round(num_decades * num_points_per_decade)) + 1

        # Generate logarithmically spaced frequencies
        return np.logspace(fstart, fend, num_total_points)

    def wvls(self, num_points: int, spacing: str = "uniform_wvl") -> NDArray[np.float64]:
        """
        method ``wvls()`` returns a numpy array of ``num_points`` wavelengths uniformly
        sampled from the range of wavelengths;
        if ``num_points == 1`` the method returns central wavelength ``wvl0``.

        Parameters
        ----------
        num_points : int
            Number of wavelength points in a range of wavelengths of interest.
        spacing: str = "uniform_wvl"
            Mode of wavelength sampling.
            If spacing is set to ``uniform_wvl``, wavelengths are sampled uniformly over the wavelength interval.
            If set to ``uniform_freq``, wavelengths correspond to uniformly sampled frequencies over the frequency interval.
            Wavelengths are sorted in ascending order.

        Returns
        -------
        np.ndarray
            a numpy array of uniformly distributed wavelength samples in ascending order.
        """
        if spacing == "uniform_freq":
            if num_points == 1:
                return np.array([C_0 / self.freq0])
            else:
                return np.sort(C_0 / self.freqs(num_points=num_points, spacing=spacing))

        elif spacing == "uniform_wvl":
            if num_points == 1:
                return np.array([self.wvl0])
            else:
                wvl_min = C_0 / self.fmax
                wvl_max = C_0 / self.fmin
                return np.linspace(wvl_min, wvl_max, num_points)

        else:
            raise ValueError(
                "Invalid `spacing` value in FreqRange.wvls(): expected 'uniform_freq' or 'uniform_wvl'. "
                f"Received: {spacing!r}. Please provide a valid spacing type."
            )

    def to_gaussian_pulse(self, **kwargs: Any) -> GaussianPulse:
        """
        to_gaussian_pulse(): Return a ``GaussianPulse`` instance based on this frequency range.

        This method constructs a ``GaussianPulse`` using
        ``GaussianPulse.from_frequency_range(fmin=self.fmin, fmax=self.fmax, **kwargs)``.

        If you prefer to define the pulse using ``freq0`` and ``fwidth`` directly,
        you should manually instantiate the ``GaussianPulse`` instead.

        Parameters
        ----------
        kwargs : dict
            Keyword arguments passed to ``GaussianPulse()``, excluding ``freq0`` & ``fwidth``.

        Returns
        -------
        GaussianPulse
            A ``GaussianPulse`` that maximizes its amplitude in the frequency range [``fmin``, ``fmax``].
        """

        duplicate_keys = {"fmin", "fmax"} & kwargs.keys()
        if duplicate_keys:
            is_plural = len(duplicate_keys) > 1
            keys_str = ", ".join(f"'{key}'" for key in sorted(duplicate_keys, reverse=True))
            raise ValueError(
                f"Keyword argument{'s' if is_plural else ''} {keys_str} "
                f"conflict{'' if is_plural else 's'} with values already set in the 'FreqRange' object. "
                f"Please exclude {'them' if is_plural else 'it'} from the 'to_gaussian_pulse()' call."
            )

        # create an instance of GaussianPulse class with defined frequency params
        return GaussianPulse.from_frequency_range(fmin=self.fmin, fmax=self.fmax, **kwargs)
