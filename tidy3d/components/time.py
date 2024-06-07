"""Defines time dependence"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pydantic.v1 as pydantic

from ..constants import RADIAN
from ..exceptions import SetupError
from .base import Tidy3dBaseModel
from .types import ArrayFloat1D, Ax, PlotVal
from .viz import add_ax_if_none

# in spectrum computation, discard amplitudes with relative magnitude smaller than cutoff
DFT_CUTOFF = 1e-8


class AbstractTimeDependence(ABC, Tidy3dBaseModel):
    """Base class describing time dependence."""

    amplitude: pydantic.NonNegativeFloat = pydantic.Field(
        1.0, title="Amplitude", description="Real-valued maximum amplitude of the time dependence."
    )

    phase: float = pydantic.Field(
        0.0, title="Phase", description="Phase shift of the time dependence.", units=RADIAN
    )

    @abstractmethod
    def amp_time(self, time: float) -> complex:
        """Complex-valued amplitude as a function of time.

        Parameters
        ----------
        time : float
            Time in seconds.

        Returns
        -------
        complex
            Complex-valued amplitude at that time.
        """

    def spectrum(
        self,
        times: ArrayFloat1D,
        freqs: ArrayFloat1D,
        dt: float,
    ) -> complex:
        """Complex-valued spectrum as a function of frequency.
        Note: Only the real part of the time signal is used.

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

        Returns
        -------
        np.ndarray
            Complex-valued array (of len(freqs)) containing spectrum at those frequencies.
        """

        times = np.array(times)
        freqs = np.array(freqs)
        time_amps = np.real(self.amp_time(times))

        # if all time amplitudes are zero, just return (complex-valued) zeros for spectrum
        if np.all(np.equal(time_amps, 0.0)):
            return (0.0 + 0.0j) * np.zeros_like(freqs)

        # Cut to only relevant times
        relevant_time_inds = np.where(np.abs(time_amps) / np.amax(np.abs(time_amps)) > DFT_CUTOFF)
        # find first and last index where the filter is True
        start_ind = relevant_time_inds[0][0]
        stop_ind = relevant_time_inds[0][-1] + 1
        time_amps = time_amps[start_ind:stop_ind]
        times_cut = times[start_ind:stop_ind]
        if times_cut.size == 0:
            return (0.0 + 0.0j) * np.zeros_like(freqs)

        # only need to compute DTFT kernel for distinct dts
        # usually, there is only one dt, if times is simulation time mesh
        dts = np.diff(times_cut)
        dts_unique, kernel_indices = np.unique(dts, return_inverse=True)

        dft_kernels = [np.exp(2j * np.pi * freqs * curr_dt) for curr_dt in dts_unique]
        running_kernel = np.exp(2j * np.pi * freqs * times_cut[0])
        dft = np.zeros(len(freqs), dtype=complex)
        for amp, kernel_index in zip(time_amps, kernel_indices):
            dft += running_kernel * amp
            running_kernel *= dft_kernels[kernel_index]

        # kernel_indices was one index shorter than time_amps
        dft += running_kernel * time_amps[-1]

        return dt * dft / np.sqrt(2 * np.pi)

    @add_ax_if_none
    def plot_spectrum_in_frequency_range(
        self,
        times: ArrayFloat1D,
        fmin: float,
        fmax: float,
        num_freqs: int = 101,
        val: PlotVal = "real",
        ax: Ax = None,
    ) -> Ax:
        """Plot the complex-valued amplitude of the time-dependence.
        Note: Only the real part of the time signal is used.

        Parameters
        ----------
        times : np.ndarray
            Array of evenly-spaced times (seconds) to evaluate time-dependence at.
            The spectrum is computed from this value and the time frequency content.
            To see spectrum for a specific :class:`Simulation`,
            pass ``simulation.tmesh``.
        fmin : float
            Lower bound of frequency for the spectrum plot.
        fmax : float
            Upper bound of frequency for the spectrum plot.
        num_freqs : int = 101
            Number of frequencies to plot within the [fmin, fmax].
        ax : matplotlib.axes._subplots.Axes = None
            Matplotlib axes to plot on, if not specified, one is created.

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
        freqs = np.linspace(fmin, fmax, num_freqs)

        spectrum = self.spectrum(times=times, dt=dt, freqs=freqs)

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

    @add_ax_if_none
    def plot(self, times: ArrayFloat1D, val: PlotVal = "real", ax: Ax = None) -> Ax:
        """Plot the complex-valued amplitude of the time-dependence.

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
