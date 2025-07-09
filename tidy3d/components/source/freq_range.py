"""Utility class ``FreqRange`` for frequency and wavelength handling."""

from __future__ import annotations

import numpy as np
import pydantic.v1 as pydantic
from numpy.typing import NDArray

from tidy3d import constants as td_const
from tidy3d.components.base import Tidy3dBaseModel
from tidy3d.components.source.time import GaussianPulse


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
    >>> central_freq = freq_range.freqs(num_points=1)
    >>> freqs = freq_range.freqs(num_points=11)
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

    @property
    def fmin(self) -> float:
        """Infer lowest frequency ``fmin`` from central frequency ``freq0`` and bandwidth ``fwidth``."""
        return self.freq0 - self.fwidth

    @property
    def fmax(self) -> float:
        """Infer highest frequency ``fmax`` from central frequency ``freq0`` and bandwidth ``fwidth``."""
        return self.freq0 + self.fwidth

    @property
    def lda0(self) -> float:
        """Get central wavelength from central frequency and bandwidth."""
        lmin = td_const.C_0 / (self.freq0 + self.fwidth)
        lmax = td_const.C_0 / (self.freq0 - self.fwidth)
        return 0.5 * (lmin + lmax)

    @classmethod
    def from_freq_interval(cls, fmin: float, fmax: float) -> FreqRange:
        """
        method ``from_freq_interval()`` creates instance of class ``FreqRange`` from frequency interval
        defined by arguments  ``fmin`` and ``fmax``.

        NB: central frequency never corresponds to central wavelength!
        ``freq0 = (fmin + fmax) / 2`` implies that ``lda0 != (lda_min + lda_max) / 2`` and vise-versa.

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
        fwidth = 0.5 * (fmax - fmin)  # extract bandwidth
        return cls(freq0=freq0, fwidth=fwidth)

    @classmethod
    def from_wavelength(cls, wvl0: float, wvl_width: float) -> FreqRange:
        """
        method ``from_wavelength()`` updated instance of class ``FreqRange`` by reassigning new
        frequency- and wavelength-related parameters.

        NB: central frequency never corresponds to central wavelength!
        ``lda0 = (lda_min + lda_max) / 2`` implies that ``freq0 != (fmin + fmax) / 2`` and vise versa.

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
        fmin = td_const.C_0 / (wvl0 + wvl_width)
        fmax = td_const.C_0 / (wvl0 - wvl_width)

        return cls.from_freq_interval(fmin=fmin, fmax=fmax)

    @classmethod
    def from_wvl_interval(cls, wvl_min: float, wvl_max: float) -> FreqRange:
        """
        method ``from_wvl_interval()`` updated instance of class ``FreqRange`` by reassigning new
        frequency- and wavelength-related parameters.

        NB: central frequency never corresponds to central wavelength!
        ``lda0 = (lda_min + lda_max) / 2`` implies that ``freq0 != (fmin + fmax) / 2``.

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
        fmax = td_const.C_0 / wvl_min
        fmin = td_const.C_0 / wvl_max

        return cls.from_freq_interval(fmin=fmin, fmax=fmax)

    def freqs(self, num_points: int) -> NDArray[np.float64]:
        """
        method ``freqs()`` returns a numpy array of ``num_point`` frequencies uniformly
        sampled from the specified frequency range;
        if ``num_points == 1`` method returns the central frequency ``freq0``.

        Parameters
        ----------
        num_points : int
            Number of frequency points in a frequency range of interest.

        Returns
        -------
        np.ndarray
            a numpy array of uniformly distributed frequency samples in a frequency range of interest.
        """

        if num_points == 1:  # return central frequency
            return np.array([self.freq0])
        else:
            # calculate frequency points and corresponding wavelengths
            return np.linspace(self.fmin, self.fmax, num_points)

    def ldas(self, num_points: int) -> NDArray[np.float64]:
        """
        method ``ldas()`` returns a numpy array of ``num_points`` wavelengths uniformly
        sampled from the range of wavelengths;
        if ``num_points == 1`` the method returns central wavelength ``lda0``.

        Parameters
        ----------
        num_points : int
            Number of wavelength points in a range of wavelengths of interest.

        Returns
        -------
        np.ndarray
            a numpy array of uniformly distributed wavelength samples in ascending order.
        """
        if num_points == 1:  # return central wavelength
            return np.array([self.lda0])
        else:
            # define shortest and longest wavelengths
            lmin = td_const.C_0 / self.fmax
            lmax = td_const.C_0 / self.fmin

            # generate array of wavelengths (in ascending order)
            return np.linspace(lmin, lmax, num_points)

    def to_gaussian_pulse(self, **kwargs) -> GaussianPulse:
        """
        method ``to_gaussian_pulse()`` returns instance of class ``GaussianPulse``
        with frequency-specific parameters defined in ``FreqRange``.

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
