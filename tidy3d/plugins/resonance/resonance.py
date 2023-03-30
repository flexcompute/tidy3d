"""Find resonances in time series data
"""

from typing import Tuple, List, Union
from functools import partial

import numpy as np
import scipy.linalg
import xarray as xr

from pydantic import Field, NonNegativeFloat, PositiveInt, validator

from ...log import log
from ...components.base import Tidy3dBaseModel
from ...components.types import ArrayFloat1D, ArrayComplex1D, ArrayComplex2D, ArrayComplex3D
from ...components.data.data_array import ScalarFieldTimeDataArray
from ...components.data.monitor_data import FieldTimeData
from ...constants import HERTZ
from ...exceptions import SetupError, ValidationError


INIT_NUM_FREQS = 200

TIME_STEP_RTOL = 1e-5

RCOND = 1e-4


# ResonanceData will be used internally
class ResonanceData(Tidy3dBaseModel):
    """Data class for storing objects computed while running the resonance finder."""

    eigvals: ArrayComplex1D = Field(..., title="Eigenvalues", description="Resonance eigenvalues.")
    complex_amplitudes: ArrayComplex1D = Field(
        None, title="Complex amplitudes", description="Complex resonance amplitudes"
    )
    errors: ArrayFloat1D = Field(
        None, title="Errors", description="Rough eigenvalue error estimate."
    )


class ResonanceFinder(Tidy3dBaseModel):
    """Tool that extracts resonance information from a time series of the form shown below.
    The resonance information consists of frequency :math:`f`, decay rate :math:`\\alpha`,
    Q factor :math:`Q = \\pi |f|/\\alpha`, amplitude :math:`a`, and phase :math:`\\phi`.

    Note
    ----
    .. math::

        f(t) = \\sum_k a_k e^{i \\phi_k} e^{-2 \\pi i f_k t - \\alpha_k t}

    Note
    ----
    We use the algorithm described in

    Vladimir A. Mandelshtam and Howard S. Taylor,
    "Harmonic inversion of time signals and its applications,"
    J. Chem. Phys. 107, 6756 (1997).

    Example
    -------
    >>> import numpy as np
    >>> from tidy3d.plugins.resonance import ResonanceFinder
    >>> t = np.linspace(0, 10000, 10000)
    >>> f1 = 2*np.pi*0.1 - 1j*0.002
    >>> f2 = 2*np.pi*0.2 - 1j*0.0005
    >>> sig = 2*np.exp(-1j*f1*t) + 3*1j*np.exp(-1j*f2*t)
    >>> resfinder = ResonanceFinder(freq_window=(0.05, 0.25))
    >>> resdata = resfinder.run_raw_signal(signal=sig, time_step=1)
    >>> resdata.to_dataframe()

    """

    freq_window: Tuple[float, float] = Field(
        ...,
        title="Window ``[fmin, fmax]``",
        description="Window ``[fmin, fmax]`` for the initial frequencies. "
        "The resonance finder is initialized with an even grid of frequencies between "
        "fmin and fmax. The resonance finder then iteratively optimizes and prunes these "
        "frequencies. Note that frequencies outside this window may be returned. "
        "A narrow frequency window that only contains a few resonances may give enhanced "
        "accuracy compared to a broad frequency window with many resonances.",
        units=HERTZ,
    )

    init_num_freqs: PositiveInt = Field(
        INIT_NUM_FREQS,
        title="Initial number of frequencies.",
        description="Number of frequencies with which the resonance finder is initialized. "
        "The resonance finder then iteratively optimizes and prunes these frequencies. "
        "The number of frequencies returned will be less than ``init_num_freqs``. "
        "Making this larger can find more resonances, while making it smaller can speed up "
        "the resonance finder.",
    )

    rcond: NonNegativeFloat = Field(
        RCOND,
        title="Cutoff for eigenvalues",
        description="Cutoff for eigenvalues, relative to the largest eigenvalue. "
        "The resonance finder solves a generalized eigenvalue problem of the form "
        ":math:`Ax = \\lambda B x`. If B has small eigenvalues, this is poorly conditioned, "
        "so we eliminate eigenvalues of B less than ``rcond`` times the largest eigenvalue. "
        "Making this closer to zero will typically return more resonances.",
    )

    @validator("freq_window", always=True)
    def _check_freq_window(cls, val):
        """Validate ``freq_window``"""
        if val[1] < val[0]:
            raise ValidationError(
                "The value of 'freq_window[1]' is less than the value of 'freq_window[0]'."
            )
        return val

    def run(self, signals: Union[FieldTimeData, Tuple[FieldTimeData, ...]]) -> xr.Dataset:
        """Finds resonances in a :class:`.FieldTimeData` or a Tuple of such.
        The time coordinates must be uniformly spaced, and the spacing must be the same
        across all supplied data. The resonance finder runs on the sum of the
        ``Ex``, ``Ey``, and ``Ez`` fields of all the supplied data,
        unless no electric fields are present at all, in which case it runs on the sum of the
        ``Hx``, ``Hy``, and ``Hz`` fields.
        Note that the signal should start after the sources have turned off.

        Parameters
        ----------
        signals: :class:`.FieldTimeData`
            Data to search for resonances

        Returns
        -------
        xr.Dataset
            Dataset containing the decay rate, Q, amplitude, phase, and estimation error
            of the resonances as a function of frequency. Modes with low Q, small
            amplitude, or high estimation error are likely to be spurious.
        """
        if isinstance(signals, FieldTimeData):
            signals = (signals,)
        if len(signals) == 0:
            raise SetupError("No 'FieldTimeData' supplied.")
        return self.run_scalar_field_time(self._aggregate_field_time(signals))

    def run_scalar_field_time(self, signal: ScalarFieldTimeDataArray) -> xr.Dataset:
        """Finds resonances in a :class:`.ScalarFieldTimeDataArray`.
        The time coordinates must be uniformly spaced to use the resonance finder;
        the time step is calculated automatically.
        Note that the signal should start after the sources have turned off.

        Parameters
        ----------
        signal : :class:`.ScalarFieldTimeDataArray`
            Time series to search for resonances

        Returns
        -------
        xr.Dataset
            Dataset containing the decay rate, Q, amplitude, phase, and estimation error
            of the resonances as a function of frequency. Modes with low Q, small
            amplitude, or high estimation error are likely to be spurious.
        """
        signal, dt = self._validate_scalar_field_time(signal)
        return self.run_raw_signal(signal, dt)

    def run_raw_signal(self, signal: List[complex], time_step: float) -> xr.Dataset:
        """Finds resonances in a time series.
        Note that the signal should start after the sources have turned off.

        Parameters
        ----------
        signal : List[complex]
            One-dimensional array holding the complex-valued time series data
            to search for resonances.
        time_step : float
            Time step / sampling rate of the data (in seconds).

        Returns
        -------
        xr.Dataset
            Dataset containing the decay rate, Q, amplitude, phase, and estimation error
            of the resonances as a function of frequency. Modes with low Q, small
            amplitude, or high estimation error are likely to be spurious.
        """
        signal = np.array(signal)
        if len(signal.shape) != 1:
            raise SetupError("The input signal should only have one dimension.")
        fmin, fmax = self.freq_window
        nfreqs = self.init_num_freqs
        log.info(f"\tRunning ResonanceFinder (nfreqs = {nfreqs})")
        omegas = np.linspace(
            time_step * fmin * 2 * np.pi,
            time_step * fmax * 2 * np.pi,
            nfreqs,
        )
        eigvals = np.exp(-1j * omegas)
        resdata = ResonanceData(eigvals=eigvals)
        resdata = self._iterate(signal, resdata)
        prev_num_eigvals = len(resdata.eigvals)
        for _ in range(1 + 2 * nfreqs):
            resdata = self._iterate(signal, resdata)
            new_num_eigvals = len(resdata.eigvals)
            if new_num_eigvals == prev_num_eigvals:
                log.info(f"\tCompleted ResonanceFinder (nfreqs = {new_num_eigvals})")
                break
            log.info(f"\tIterated ResonanceFinder (nfreqs = {new_num_eigvals})")
            prev_num_eigvals = new_num_eigvals

        resonances = self._get_resonance_info(data=resdata, time_step=time_step)

        return resonances.sortby("freq")

    def _validate_scalar_field_time(
        self, signal: ScalarFieldTimeDataArray
    ) -> Tuple[ArrayComplex1D, float]:
        """Validates a :class:`.ScalarFieldTimeDataArray` and returns the time step
        as well as underlying data array."""
        dts = np.diff(signal.t)
        dt = dts[0]
        if not np.allclose(dts, dt, rtol=1e-5, atol=0):
            raise SetupError(
                "The time coordinates in the supplied "
                "'ScalarFieldTimeDataArray' must be uniformly spaced to use "
                "the resonance finder."
            )
        for dim in signal.dims:
            if dim != "t" and len(signal.coords[dim]) > 1:
                raise SetupError(
                    "The length of each coordinate besides 't' of the supplied "
                    "'ScalarFieldTimeDataArray' must be equal to one."
                )
        return np.squeeze(signal.data), dt

    def _aggregate_field_time_comps(
        self, signals: Tuple[FieldTimeData, ...], comps
    ) -> ScalarFieldTimeDataArray:
        """Aggregates the given components from several :class:`.FieldTimeData`."""
        total_signal = None
        dt = -1
        coords = dict(x=[0], y=[0], z=[0], t=[0])

        for sig_field in signals:
            for comp in comps:
                field_comp = getattr(sig_field, comp, None)
                if not field_comp is None:
                    curr_signal, curr_dt = self._validate_scalar_field_time(field_comp)
                    if dt == -1:
                        dt = curr_dt
                        coords["t"] = field_comp.t
                    elif abs(curr_dt - dt) > TIME_STEP_RTOL:
                        raise SetupError(
                            "The time steps in the supplied 'FieldTimeData' must all be equal."
                        )
                    if total_signal is None:
                        total_signal = np.zeros(len(curr_signal), dtype=complex)
                    elif len(curr_signal) != len(total_signal):
                        raise SetupError(
                            "The length of the supplied 'FieldTimeData' must all be equal."
                        )
                    total_signal += curr_signal

        if total_signal is None:
            return None
        return ScalarFieldTimeDataArray(
            np.reshape(total_signal, (1, 1, 1, len(total_signal))), coords=coords
        )

    def _aggregate_field_time(
        self, signals: Union[FieldTimeData, Tuple[FieldTimeData, ...]]
    ) -> ScalarFieldTimeDataArray:
        """Aggregates several :class:`.FieldTimeData` into a single
        :class:`.ScalarFieldTimeDataArray`."""
        electric_components = ["Ex", "Ey", "Ez"]
        magnetic_components = ["Hx", "Hy", "Hz"]

        # first check electric components
        electric_data = self._aggregate_field_time_comps(signals, electric_components)
        if not electric_data is None:
            return electric_data
        # now check magnetic components
        magnetic_data = self._aggregate_field_time_comps(signals, magnetic_components)

        if magnetic_data is None:
            raise SetupError("No field components supplied.")
        return magnetic_data

    def _get_resonance_info(self, data: ResonanceData, time_step: float) -> xr.Dataset:
        """Extracts resonance information from the ResonanceData, and packages it as
        an xr.Dataset."""
        complex_omegas = 1j * np.log(data.eigvals)
        freqs = np.real(complex_omegas / (2 * np.pi)) / time_step
        decays = -np.imag(complex_omegas) / time_step
        q_factors = np.pi * np.abs(freqs) / decays
        amplitudes = np.abs(data.complex_amplitudes)
        phases = np.angle(data.complex_amplitudes)
        errors = np.array(data.errors)

        coords = {"freq": freqs}
        keys = ("decay", "Q", "amplitude", "phase", "error")
        vals = (decays, q_factors, amplitudes, phases, errors)

        data_arrays = tuple(map(partial(xr.DataArray, coords=coords), vals))
        return xr.Dataset(dict(zip(keys, data_arrays)))

    def _evaluate_matrices(self, signal: ArrayComplex1D, eigvals: ArrayComplex1D) -> ArrayComplex3D:
        """Compute the evolution matrices"""
        half_len = int(len(signal) / 2) - 2
        nfreqs = len(eigvals)
        zvals = eigvals / np.abs(eigvals)

        zinvl = np.exp(1j * np.outer(np.log(zvals) * 1j, np.arange(2 * half_len + 1)))

        prefactor = np.zeros((nfreqs, nfreqs), dtype=complex)

        prefactor[~np.eye(nfreqs, dtype=bool)] = np.reciprocal(
            np.subtract.outer(zvals, zvals)[~np.eye(nfreqs, dtype=bool)]
        )

        u_matrices = np.zeros((3, nfreqs, nfreqs), dtype=complex)
        for pval in range(3):

            u_matrices[pval, :, :] = prefactor * (
                np.outer(zvals, zinvl[:, : half_len + 1] @ signal[pval:][: half_len + 1])
                + np.outer(
                    zinvl[:, :half_len] @ signal[pval:][half_len + 1 : 2 * half_len + 1],
                    zinvl[:, half_len],
                )
            )
            u_matrices[pval, :, :] += u_matrices[pval, :, :].T

            for i in range(nfreqs):
                u_matrices[pval, i, i] = np.dot(
                    zinvl[i, : 2 * half_len + 1],
                    np.concatenate((np.arange(1, half_len + 2), np.arange(half_len, 0, -1)))
                    * signal[pval:][: 2 * half_len + 1],
                )

        return u_matrices

    def _gram_schmidt(self, a_matrix: ArrayComplex2D) -> ArrayComplex2D:
        """Perform the Gram-Schmidt process on the columns of a matrix."""
        new_a_matrix = np.zeros(a_matrix.shape, dtype=complex)
        for i in range(new_a_matrix.shape[1]):
            new_a_matrix[:, i] = a_matrix[:, i]
            for j in range(i):
                new_a_matrix[:, i] -= new_a_matrix[:, j] * np.dot(
                    new_a_matrix[:, j], a_matrix[:, i]
                )
            new_a_matrix[:, i] /= np.sqrt(np.dot(new_a_matrix[:, i], new_a_matrix[:, i]))
        return new_a_matrix

    def _solve_gen_eig_prob(
        self, a_matrix: ArrayComplex2D, b_matrix: ArrayComplex2D, rcond: float
    ) -> Tuple[ArrayComplex1D, ArrayComplex2D]:
        """Solve a generalized eigenvalue problem of the form

        .. math::
        Av = \\lambda B v

        We make B nonsingular by eliminating eigenvalues smaller
        than ```rcond`` times the largest eigenvalue. This method is used in
        Michael R. Wall and Daniel Neuhauser,
        "Extraction, through filter-diagonalization, of general quantum eigenvalues
        or classical normal mode frequencies of a short-time segment of a signal.
        I. Theory and application to a quantum-dynamics model,"
        J. Chem. Phys. 102, 8001 (1995).
        """
        eigvals_b, eigvecs_b = scipy.linalg.eig(b_matrix)
        large_inds = np.abs(eigvals_b) > rcond * np.amax(np.abs(eigvals_b))
        eigvals_b = eigvals_b[large_inds]
        eigvecs_b = eigvecs_b[:, large_inds]
        eigvecs_b = self._gram_schmidt(eigvecs_b)

        change_basis = eigvecs_b @ np.diag(1 / np.sqrt(eigvals_b))
        new_a_matrix = change_basis.T @ a_matrix @ change_basis
        eigvals, eigvecs_a = scipy.linalg.eig(new_a_matrix)
        eigvecs_a = self._gram_schmidt(eigvecs_a)

        eigvecs = change_basis @ eigvecs_a

        return eigvals, eigvecs

    def _find_amplitudes(
        self,
        signal: ArrayComplex1D,
        prev_eigvals: ArrayComplex1D,
        eigvals: ArrayComplex1D,
        eigvecs: ArrayComplex2D,
    ) -> ArrayComplex1D:
        """Compute the resonance amplitudes."""
        half_len = int(len(signal) / 2) - 2
        nfreqs = len(eigvals)
        zvals = prev_eigvals / np.abs(prev_eigvals)
        amplitudes = np.zeros(nfreqs, dtype=complex)

        zinvl = np.exp(1j * np.outer(np.log(zvals) * 1j, np.arange(half_len + 1)))
        factor = zinvl @ signal[: half_len + 1]

        for k in range(nfreqs):
            amplitudes[k] = np.square(eigvecs[:, k] @ factor)

        return amplitudes

    def _find_errors(
        self,
        eigvals: ArrayComplex1D,
        u_matrices: ArrayComplex3D,
        eigvecs: ArrayComplex2D,
    ) -> ArrayComplex1D:
        """Estimate the eigenvalue error."""
        nfreqs = len(eigvals)
        errors = np.zeros(nfreqs)
        for k in range(nfreqs):
            errors[k] = np.linalg.norm(
                (u_matrices[2] - (eigvals[k] ** 2) * u_matrices[0]) @ eigvecs[:, k]
            )
        return errors

    def _iterate(self, signal: ArrayComplex1D, prev_resdata: ResonanceData) -> ResonanceData:
        """Run a single iteration of the resonance finder."""
        prev_eigvals = prev_resdata.eigvals

        u_matrices = self._evaluate_matrices(signal, prev_eigvals)

        eigvals, eigvecs = self._solve_gen_eig_prob(u_matrices[1], u_matrices[0], self.rcond)

        errors = self._find_errors(eigvals, u_matrices, eigvecs)

        amplitudes = self._find_amplitudes(signal, prev_eigvals, eigvals, eigvecs)

        return ResonanceData(eigvals=eigvals, errors=errors, complex_amplitudes=amplitudes)
