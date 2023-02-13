# pylint: disable=invalid-name, too-many-lines
"""Defines properties of the medium / materials"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple, Union, Callable, Optional, Dict, List
import functools

import pydantic as pd
import numpy as np
import xarray as xr

from .base import Tidy3dBaseModel, cached_property
from .grid.grid import Coords, Grid
from .types import PoleAndResidue, Ax, FreqBound, TYPE_TAG_STR, InterpMethod, Numpy, Bound
from .data.dataset import PermittivityDataset
from .data.data_array import ScalarFieldDataArray
from .viz import add_ax_if_none
from .validators import validate_name_str
from ..constants import C_0, pec_val, EPSILON_0
from ..constants import HERTZ, CONDUCTIVITY, PERMITTIVITY, RADPERSEC, MICROMETER, SECOND
from ..log import log, ValidationError, SetupError

# evaluate frequency as this number (Hz) if inf
FREQ_EVAL_INF = 1e50

# extrapolation option in custom medium
FILL_VALUE = "extrapolate"


def ensure_freq_in_range(eps_model: Callable[[float], complex]) -> Callable[[float], complex]:
    """Decorate ``eps_model`` to log warning if frequency supplied is out of bounds."""

    @functools.wraps(eps_model)
    def _eps_model(self, frequency: float) -> complex:
        """New eps_model function."""

        # evaluate infs and None as FREQ_EVAL_INF
        is_inf_scalar = isinstance(frequency, float) and np.isinf(frequency)
        if frequency is None or is_inf_scalar:
            frequency = FREQ_EVAL_INF

        if isinstance(frequency, np.ndarray):
            frequency[np.where(np.isinf(frequency))] = FREQ_EVAL_INF

        # if frequency range not present just return original function
        if self.frequency_range is None:
            return eps_model(self, frequency)

        fmin, fmax = self.frequency_range
        if np.any(frequency < fmin) or np.any(frequency > fmax):
            log.warning(
                "frequency passed to `Medium.eps_model()`"
                f"is outside of `Medium.frequency_range` = {self.frequency_range}"
            )
        return eps_model(self, frequency)

    return _eps_model


""" Medium Definitions """


class AbstractMedium(ABC, Tidy3dBaseModel):
    """A medium within which electromagnetic waves propagate."""

    name: str = pd.Field(None, title="Name", description="Optional unique name for medium.")

    frequency_range: FreqBound = pd.Field(
        None,
        title="Frequency Range",
        description="Optional range of validity for the medium.",
        units=(HERTZ, HERTZ),
    )

    _name_validator = validate_name_str()

    @abstractmethod
    def eps_model(self, frequency: float) -> complex:
        """Complex-valued permittivity as a function of frequency.

        Parameters
        ----------
        frequency : float
            Frequency to evaluate permittivity at (Hz).

        Returns
        -------
        complex
            Complex-valued relative permittivity evaluated at ``frequency``.
        """

    def nk_model(self, frequency: float) -> Tuple[float, float]:
        """Real and imaginary parts of the refactive index as a function of frequency.

        Parameters
        ----------
        frequency : float
            Frequency to evaluate permittivity at (Hz).
        Returns
        -------
        Tuple[float, float]
            Real part (n) and imaginary part (k) of refractive index of medium.
        """
        eps_complex = self.eps_model(frequency=frequency)
        return self.eps_complex_to_nk(eps_complex)

    @ensure_freq_in_range
    def eps_diagonal(self, frequency: float) -> Tuple[complex, complex, complex]:
        """Main diagonal of the complex-valued permittivity tensor as a function of frequency.

        Parameters
        ----------
        frequency : float
            Frequency to evaluate permittivity at (Hz).

        Returns
        -------
        complex
            The diagonal elements of the relative permittivity tensor evaluated at ``frequency``.
        """

        # This only needs to be overwritten for anisotropic materials
        eps = self.eps_model(frequency)
        return (eps, eps, eps)

    @add_ax_if_none
    def plot(self, freqs: float, ax: Ax = None) -> Ax:  # pylint: disable=invalid-name
        """Plot n, k of a :class:`Medium` as a function of frequency.

        Parameters
        ----------
        freqs: float
            Frequencies (Hz) to evaluate the medium properties at.
        ax : matplotlib.axes._subplots.Axes = None
            Matplotlib axes to plot on, if not specified, one is created.

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.
        """

        freqs = np.array(freqs)
        eps_complex = self.eps_model(freqs)
        n, k = AbstractMedium.eps_complex_to_nk(eps_complex)

        freqs_thz = freqs / 1e12
        ax.plot(freqs_thz, n, label="n")
        ax.plot(freqs_thz, k, label="k")
        ax.set_xlabel("frequency (THz)")
        ax.set_title("medium dispersion")
        ax.legend()
        ax.set_aspect("auto")
        return ax

    """ Conversion helper functions """

    @staticmethod
    def nk_to_eps_complex(n: float, k: float = 0.0) -> complex:
        """Convert n, k to complex permittivity.

        Parameters
        ----------
        n : float
            Real part of refractive index.
        k : float = 0.0
            Imaginary part of refrative index.

        Returns
        -------
        complex
            Complex-valued relative permittivty.
        """
        eps_real = n**2 - k**2
        eps_imag = 2 * n * k
        return eps_real + 1j * eps_imag

    @staticmethod
    def eps_complex_to_nk(eps_c: complex) -> Tuple[float, float]:
        """Convert complex permittivity to n, k values.

        Parameters
        ----------
        eps_c : complex
            Complex-valued relative permittivity.

        Returns
        -------
        Tuple[float, float]
            Real and imaginary parts of refractive index (n & k).
        """
        ref_index = np.sqrt(eps_c)
        return ref_index.real, ref_index.imag

    @staticmethod
    def nk_to_eps_sigma(n: float, k: float, freq: float) -> Tuple[float, float]:
        """Convert ``n``, ``k`` at frequency ``freq`` to permittivity and conductivity values.

        Parameters
        ----------
        n : float
            Real part of refractive index.
        k : float = 0.0
            Imaginary part of refrative index.
        frequency : float
            Frequency to evaluate permittivity at (Hz).

        Returns
        -------
        Tuple[float, float]
            Real part of relative permittivity & electric conductivity.
        """
        eps_complex = AbstractMedium.nk_to_eps_complex(n, k)
        eps_real, eps_imag = eps_complex.real, eps_complex.imag
        omega = 2 * np.pi * freq
        sigma = omega * eps_imag * EPSILON_0
        return eps_real, sigma

    @staticmethod
    def eps_sigma_to_eps_complex(eps_real: float, sigma: float, freq: float) -> complex:
        """convert permittivity and conductivity to complex permittivity at freq

        Parameters
        ----------
        eps_real : float
            Real-valued relative permittivity.
        sigma : float
            Conductivity.
        freq : float
            Frequency to evaluate permittivity at (Hz).
            If not supplied, returns real part of permittivity (limit as frequency -> infinity.)

        Returns
        -------
        complex
            Complex-valued relative permittivity.
        """
        if freq is None:
            return eps_real
        omega = 2 * np.pi * freq

        return eps_real + 1j * sigma / omega / EPSILON_0

    @staticmethod
    def eps_complex_to_eps_sigma(eps_complex: complex, freq: float) -> Tuple[float, float]:
        """Convert complex permittivity at frequency ``freq``
        to permittivity and conductivity values.

        Parameters
        ----------
        eps_complex : complex
            Complex-valued relative permittivity.
        freq : float
            Frequency to evaluate permittivity at (Hz).

        Returns
        -------
        Tuple[float, float]
            Real part of relative permittivity & electric conductivity.
        """
        eps_real, eps_imag = eps_complex.real, eps_complex.imag  # pylint:disable=no-member
        omega = 2 * np.pi * freq
        sigma = omega * eps_imag * EPSILON_0
        return eps_real, sigma


""" Dispersionless Medium """

# PEC keyword
class PECMedium(AbstractMedium):
    """Perfect electrical conductor class.

    Note
    ----
    To avoid confusion from duplicate PECs, should import ``tidy3d.PEC`` instance directly.
    """

    def eps_model(self, frequency: float) -> complex:

        # return something like frequency with value of pec_val + 0j
        return 0j * frequency + pec_val


# PEC builtin instance
PEC = PECMedium(name="PEC")


class Medium(AbstractMedium):
    """Dispersionless medium.

    Example
    -------
    >>> dielectric = Medium(permittivity=4.0, name='my_medium')
    >>> eps = dielectric.eps_model(200e12)
    """

    permittivity: float = pd.Field(
        1.0, ge=1.0, title="Permittivity", description="Relative permittivity.", units=PERMITTIVITY
    )

    conductivity: float = pd.Field(
        0.0,
        ge=0.0,
        title="Conductivity",
        description="Electric conductivity.  Defined such that the imaginary part of the complex "
        "permittivity at angular frequency omega is given by conductivity/omega.",
        units=CONDUCTIVITY,
    )

    @ensure_freq_in_range
    def eps_model(self, frequency: float) -> complex:
        """Complex-valued permittivity as a function of frequency."""

        return AbstractMedium.eps_sigma_to_eps_complex(
            self.permittivity, self.conductivity, frequency
        )

    @classmethod
    def from_nk(cls, n: float, k: float, freq: float, **kwargs):
        """Convert ``n`` and ``k`` values at frequency ``freq`` to :class:`Medium`.

        Parameters
        ----------
        n : float
            Real part of refractive index.
        k : float = 0
            Imaginary part of refrative index.
        freq : float
            Frequency to evaluate permittivity at (Hz).

        Returns
        -------
        :class:`Medium`
            medium containing the corresponding ``permittivity`` and ``conductivity``.
        """
        eps, sigma = AbstractMedium.nk_to_eps_sigma(n, k, freq)
        return cls(permittivity=eps, conductivity=sigma, **kwargs)


class CustomMedium(AbstractMedium):
    """:class:`.Medium` with user-supplied permittivity distribution.

    Example
    -------
    >>> Nx, Ny, Nz = 10, 9, 8
    >>> X = np.linspace(-1, 1, Nx)
    >>> Y = np.linspace(-1, 1, Ny)
    >>> Z = np.linspace(-1, 1, Nz)
    >>> freqs = [2e14]
    >>> data = np.ones((Nx, Ny, Nz, 1))
    >>> eps_diagonal_data = ScalarFieldDataArray(data, coords=dict(x=X, y=Y, z=Z, f=freqs))
    >>> eps_components = {f"eps_{d}{d}": eps_diagonal_data for d in "xyz"}
    >>> eps_dataset = PermittivityDataset(**eps_components)
    >>> dielectric = CustomMedium(eps_dataset=eps_dataset, name='my_medium')
    >>> eps = dielectric.eps_model(200e12)
    """

    eps_dataset: PermittivityDataset = pd.Field(
        ...,
        title="Permittivity Dataset",
        description="User-supplied dataset containing complex-valued permittivity "
        "as a function of space. Permittivity distribution over the Yee-grid will be "
        "interpolated based on ``interp_method``.",
    )

    interp_method: InterpMethod = pd.Field(
        "nearest",
        title="Interpolation method",
        description="Interpolation method to obtain permittivity values "
        "that are not supplied at the Yee grids; For grids outside the range "
        "of the supplied data, extrapolation will be applied. When the extrapolated "
        "value is smaller (greater) than the minimal (maximal) of the supplied data, "
        "the extrapolated value will take the minimal (maximal) of the supplied data.",
    )

    @pd.validator("eps_dataset", always=True)
    def _single_frequency(cls, val):
        """Assert only one frequency supplied."""
        for name, eps_dataset_component in val.field_components.items():
            freqs = eps_dataset_component.f
            if len(freqs) != 1:
                raise SetupError(
                    f"'eps_dataset.{name}' must have a single frequency, "
                    f"but it contains {len(freqs)} frequencies."
                )
        return val

    @pd.validator("eps_dataset", always=True)
    def _eps_inf_greater_no_less_than_one_sigma_positive(cls, val):
        """Assert any eps_inf must be >=1"""

        for comp in ["eps_xx", "eps_yy", "eps_zz"]:
            eps_inf, sigma = CustomMedium.eps_complex_to_eps_sigma(
                val.field_components[comp], val.field_components[comp].f
            )
            if np.any(eps_inf.values < 1):
                raise SetupError(
                    "Permittivity at infinite frequency at any spatial point "
                    "must be no less than one."
                )
            if np.any(sigma.values < 0):
                raise SetupError(
                    "Negative imaginary part of refrative index leads to a gain medium, "
                    "which is not supported."
                )
        return val

    @ensure_freq_in_range
    def eps_dataset_freq(self, frequency: float) -> PermittivityDataset:
        """Permittivity dataset at ``frequency``. The dispersion comes
        from DC conductivity that results in nonzero Im[eps].

        Parameters
        ----------
        frequency : float
            Frequency to evaluate permittivity at (Hz).

        Returns
        -------
        :class:`.PermittivityDataset`
            The permittivity evaluated at ``frequency``.
        """

        new_field_components = {}
        for name, eps_dataset_component in self.eps_dataset.field_components.items():
            freq = eps_dataset_component.coords["f"][0]
            eps_freq = (
                eps_dataset_component.real + 1j * eps_dataset_component.imag * freq / frequency
            )
            eps_freq = eps_freq.assign_coords({"f": [frequency]})
            new_field_components.update({name: eps_freq})
        return PermittivityDataset(**new_field_components)

    def eps_diagonal_on_grid(
        self,
        frequency: float,
        coords: Coords,
    ) -> Tuple[Numpy, Numpy, Numpy]:
        """Spatial profile of main diagonal of the complex-valued permittivity
        at ``frequency`` interpolated at the supplied coordinates.

        Parameters
        ----------
        frequency : float
            Frequency to evaluate permittivity at (Hz).
        coords : :class:`.Coords`
            The grid point coordinates over which interpolation is performed.

        Returns
        -------
        Tuple[Numpy, Numpy, Numpy]
            The complex-valued permittivity tensor at ``frequency`` interpolated
            at the supplied coordinate.
        """

        eps_freq = self.eps_dataset_freq(frequency)
        interp_shape = [len(coord_comp) for coord_comp in coords.to_list]
        eps_list = [
            np.array(
                self._interp(eps_freq.field_components[comp], coords, self.interp_method)
            ).reshape(interp_shape)
            for comp in ["eps_xx", "eps_yy", "eps_zz"]
        ]
        return tuple(eps_list)

    @ensure_freq_in_range
    def eps_diagonal(self, frequency: float) -> Tuple[complex, complex, complex]:
        """Main diagonal of the complex-valued permittivity tensor
        at ``frequency``. Spatially, we take max{||eps||}, so that autoMesh generation
        works appropriately.
        """
        eps_freq = self.eps_dataset_freq(frequency)
        eps_np_list = [
            np.array(sclr_fld).ravel() for _, sclr_fld in eps_freq.field_components.items()
        ]
        eps_list = [eps_comp[np.argmax(np.abs(eps_comp))] for eps_comp in eps_np_list]
        return tuple(eps_list)

    @ensure_freq_in_range
    def eps_model(self, frequency: float) -> complex:
        """Spatial and poloarizaiton average of complex-valued permittivity
        as a function of frequency.
        """
        eps_freq = self.eps_dataset_freq(frequency)
        eps_array_avgs = [np.mean(eps_array) for _, eps_array in eps_freq.field_components.items()]
        return np.mean(eps_array_avgs)

    @classmethod
    def from_eps_raw(
        cls, eps: ScalarFieldDataArray, interp_method: InterpMethod = "nearest"
    ) -> CustomMedium:
        """Construct a :class:`.CustomMedium` from datasets containing raw permittivity values.

        Parameters
        ----------
        eps : :class:`.ScalarFieldDataArray`
            Dataset containing complex-valued permittivity as a function of space.
        interp_method : :class:`.InterpMethod`, optional
                Interpolation method to obtain permittivity values that are not supplied
                at the Yee grids.

        Returns
        -------
        :class:`.CustomMedium`
            Medium containing the spatially varying permittivity data.
        """
        field_components = {field_name: eps.copy() for field_name in ("eps_xx", "eps_yy", "eps_zz")}
        eps_dataset = PermittivityDataset(**field_components)
        return cls(eps_dataset=eps_dataset, interp_method=interp_method)

    @classmethod
    def from_nk(
        cls,
        n: ScalarFieldDataArray,
        k: Optional[ScalarFieldDataArray] = None,
        interp_method: InterpMethod = "nearest",
    ) -> CustomMedium:
        """Construct a :class:`.CustomMedium` from datasets containing n and k values.

        Parameters
        ----------
        n : :class:`.ScalarFieldDataArray`
            Real part of refractive index.
        k : :class:`.ScalarFieldDataArray` = None
            Imaginary part of refrative index.
        interp_method : :class:`.InterpMethod`, optional
                Interpolation method to obtain permittivity values that are not supplied
                at the Yee grids.

        Returns
        -------
        :class:`.CustomMedium`
            Medium containing the spatially varying permittivity data.
        """
        if k is None:
            k = xr.zeros_like(n)

        if n.coords != k.coords:
            raise SetupError("`n` and `k` must have same coordinates.")

        eps_values = Medium.nk_to_eps_complex(n=n.data, k=k.data)
        coords = {k: np.array(v) for k, v in n.coords.items()}
        eps_scalar_field_data = ScalarFieldDataArray(eps_values, coords=coords)
        return cls.from_eps_raw(eps=eps_scalar_field_data, interp_method=interp_method)

    @staticmethod
    def _interp(
        scalar_dataset: ScalarFieldDataArray,
        coord_interp: Coords,
        interp_method: InterpMethod,
    ) -> ScalarFieldDataArray:
        """
        Enhance xarray's ``.interp`` in two ways:
            1) Check if the coordinate of the supplied data are in monotically increasing order.
            If they are, apply the faster ``assume_sorted=True``.

            2) For axes of single entry, instead of error, apply ``isel()`` along the axis.

            3) When linear interp is applied, in the extrapolated region, filter values smaller
            or larger than the original data's min(max) will be replaced with the original min(max).

        Parameters
        ----------
        scalar_dataset : :class:`.ScalarFieldDataArray`
            Supplied scalar dataset.
        coord_interp : :class:`.Coords`
            The grid point coordinates over which interpolation is performed.
        interp_method : :class:`.InterpMethod`
            Interpolation method.

        Returns
        -------
        :class:`.ScalarFieldDataArray`
            The interpolated scalar dataset.
        """

        # check in x/y/z axes, which of them are supplied with a single entry.
        all_coords = "xyz"
        is_single_entry = [scalar_dataset.sizes[ax] == 1 for ax in all_coords]
        interp_ax = [
            ax for (ax, single_entry) in zip(all_coords, is_single_entry) if not single_entry
        ]
        isel_ax = [ax for ax in all_coords if ax not in interp_ax]

        # apply isel for the axis containing single entry
        if len(isel_ax) > 0:
            scalar_dataset = scalar_dataset.isel(
                {ax: [0] * len(coord_interp.to_dict[ax]) for ax in isel_ax}
            )
            scalar_dataset = scalar_dataset.assign_coords(
                {ax: coord_interp.to_dict[ax] for ax in isel_ax}
            )
            if len(interp_ax) == 0:
                return scalar_dataset

        # Apply interp for the rest
        #   first check if it's sorted
        is_sorted = all((np.all(np.diff(scalar_dataset.coords[f]) > 0) for f in interp_ax))
        interp_param = dict(
            kwargs={"fill_value": FILL_VALUE},
            assume_sorted=is_sorted,
            method=interp_method,
        )
        #   interpolation
        interp_dataset = scalar_dataset.interp(
            {ax: coord_interp.to_dict[ax] for ax in interp_ax},
            **interp_param,
        )

        # filter any values larger/smaller than the original data's max/min.
        max_val = max(scalar_dataset.values.ravel())
        min_val = min(scalar_dataset.values.ravel())
        interp_dataset = interp_dataset.where(interp_dataset >= min_val, min_val)
        interp_dataset = interp_dataset.where(interp_dataset <= max_val, max_val)
        return interp_dataset

    def grids(self, bounds: Bound) -> Dict[str, Grid]:
        """Make a :class:`.Grid` corresponding to the data in each ``eps_ii`` component.
        The min and max coordinates along each dimension are bounded by ``bounds``."""

        rmin, rmax = bounds
        pt_mins = dict(zip("xyz", rmin))
        pt_maxs = dict(zip("xyz", rmax))

        def make_grid(scalar_field: ScalarFieldDataArray) -> Grid:
            """Make a grid for a single dataset."""

            def make_bound_coords(coords: np.ndarray, pt_min: float, pt_max: float) -> List[float]:
                """Convert user supplied coords into boundary coords to use in :class:`.Grid`."""

                # get coordinates of the bondaries halfway between user-supplied data
                coord_bounds = (coords[1:] + coords[:1]) / 2.0

                # res-set coord boundaries that lie outside geometry bounds to the boundary (0 vol.)
                coord_bounds[coord_bounds <= pt_min] = pt_min
                coord_bounds[coord_bounds >= pt_max] = pt_max

                # add the geometry bounds in explicitly
                return [pt_min] + coord_bounds.tolist() + [pt_max]

            # grab user supplied data long this dimension
            coords = {key: np.array(val) for key, val in scalar_field.coords.items()}
            spatial_coords = {key: coords[key] for key in "xyz"}

            # convert each spatial coord to boundary coords
            bound_coords = {}
            for key, coords in spatial_coords.items():
                pt_min = pt_mins[key]
                pt_max = pt_maxs[key]
                bound_coords[key] = make_bound_coords(coords=coords, pt_min=pt_min, pt_max=pt_max)

            # construct grid
            boundaries = Coords(**bound_coords)
            return Grid(boundaries=boundaries)

        grids = {}
        for field_name in ("eps_xx", "eps_yy", "eps_zz"):

            # grab user supplied data long this dimension
            scalar_field = self.eps_dataset.field_components[field_name]

            # feed it to make_grid
            grids[field_name] = make_grid(scalar_field)

        return grids


""" Dispersive Media """


class DispersiveMedium(AbstractMedium, ABC):
    """A Medium with dispersion (propagation characteristics depend on frequency)"""

    @cached_property
    @abstractmethod
    def pole_residue(self):
        """Representation of Medium as a pole-residue model."""

    @staticmethod
    def tuple_to_complex(value: Tuple[float, float]) -> complex:
        """Convert a tuple of real and imaginary parts to complex number."""

        val_r, val_i = value
        return val_r + 1j * val_i

    @staticmethod
    def complex_to_tuple(value: complex) -> Tuple[float, float]:
        """Convert a complex number to a tuple of real and imaginary parts."""

        return (value.real, value.imag)


class PoleResidue(DispersiveMedium):
    """A dispersive medium described by the pole-residue pair model.
    The frequency-dependence of the complex-valued permittivity is described by:

    Note
    ----
    .. math::

        \\epsilon(\\omega) = \\epsilon_\\infty - \\sum_i
        \\left[\\frac{c_i}{j \\omega + a_i} +
        \\frac{c_i^*}{j \\omega + a_i^*}\\right]

    Example
    -------
    >>> pole_res = PoleResidue(eps_inf=2.0, poles=[((1+2j), (3+4j)), ((5+6j), (7+8j))])
    >>> eps = pole_res.eps_model(200e12)
    """

    eps_inf: float = pd.Field(
        1.0,
        title="Epsilon at Infinity",
        description="Relative permittivity at infinite frequency (:math:`\\epsilon_\\infty`).",
        units=PERMITTIVITY,
    )

    poles: Tuple[PoleAndResidue, ...] = pd.Field(
        (),
        title="Poles",
        description="Tuple of complex-valued (:math:`a_i, c_i`) poles for the model.",
        units=(RADPERSEC, RADPERSEC),
    )

    @ensure_freq_in_range
    def eps_model(self, frequency: float) -> complex:
        """Complex-valued permittivity as a function of frequency."""

        omega = 2 * np.pi * frequency
        eps = self.eps_inf + np.zeros_like(frequency) + 0.0j
        for (a, c) in self.poles:
            a_cc = np.conj(a)
            c_cc = np.conj(c)
            eps -= c / (1j * omega + a)
            eps -= c_cc / (1j * omega + a_cc)
        return eps

    @cached_property
    def pole_residue(self):
        """Representation of Medium as a pole-residue model."""

        return PoleResidue(
            eps_inf=self.eps_inf,
            poles=self.poles,
            frequency_range=self.frequency_range,
            name=self.name,
        )

    def __str__(self):
        """string representation"""
        return (
            f"td.PoleResidue("
            f"\n\teps_inf={self.eps_inf}, "
            f"\n\tpoles={self.poles}, "
            f"\n\tfrequency_range={self.frequency_range})"
        )


class Sellmeier(DispersiveMedium):
    """A dispersive medium described by the Sellmeier model.
    The frequency-dependence of the refractive index is described by:

    Note
    ----
    .. math::

        n(\\lambda)^2 = 1 + \\sum_i \\frac{B_i \\lambda^2}{\\lambda^2 - C_i}

    Example
    -------
    >>> sellmeier_medium = Sellmeier(coeffs=[(1,2), (3,4)])
    >>> eps = sellmeier_medium.eps_model(200e12)
    """

    coeffs: Tuple[Tuple[float, pd.PositiveFloat], ...] = pd.Field(
        title="Coefficients",
        description="List of Sellmeier (:math:`B_i, C_i`) coefficients.",
        units=(None, MICROMETER + "^2"),
    )

    def _n_model(self, frequency: float) -> complex:
        """Complex-valued refractive index as a function of frequency."""

        wvl = C_0 / frequency
        wvl2 = wvl**2
        n_squared = 1.0
        for (B, C) in self.coeffs:
            n_squared += B * wvl2 / (wvl2 - C)
        return np.sqrt(n_squared)

    @ensure_freq_in_range
    def eps_model(self, frequency: float) -> complex:
        """Complex-valued permittivity as a function of frequency."""

        n = self._n_model(frequency)
        return AbstractMedium.nk_to_eps_complex(n)

    @cached_property
    def pole_residue(self):
        """Representation of Medium as a pole-residue model."""

        poles = []
        for (B, C) in self.coeffs:
            beta = 2 * np.pi * C_0 / np.sqrt(C)
            alpha = -0.5 * beta * B
            a = 1j * beta
            c = 1j * alpha
            poles.append((a, c))

        return PoleResidue(
            eps_inf=1,
            poles=poles,
            frequency_range=self.frequency_range,
            name=self.name,
        )

    @classmethod
    def from_dispersion(cls, n: float, freq: float, dn_dwvl: float = 0, **kwargs):
        """Convert ``n`` and wavelength dispersion ``dn_dwvl`` values at frequency ``freq`` to
        a single-pole :class:`Sellmeier` medium.

        Parameters
        ----------
        n : float
            Real part of refractive index. Must be larger than or equal to one.
        dn_dwvl : float = 0
            Derivative of the refractive index with wavelength (1/um). Must be negative.
        frequency : float
            Frequency to evaluate permittivity at (Hz).

        Returns
        -------
        :class:`Medium`
            medium containing the corresponding ``permittivity`` and ``conductivity``.
        """

        if dn_dwvl >= 0:
            raise ValidationError("Dispersion ``dn_dwvl`` must be smaller than zero.")
        if n < 1:
            raise ValidationError("Refractive index ``n`` cannot be smaller than one.")

        wvl = C_0 / freq
        nsqm1 = n**2 - 1
        c_coeff = -(wvl**3) * n * dn_dwvl / (nsqm1 - wvl * n * dn_dwvl)
        b_coeff = (wvl**2 - c_coeff) / wvl**2 * nsqm1
        coeffs = [(b_coeff, c_coeff)]

        return cls(coeffs=coeffs, **kwargs)


class Lorentz(DispersiveMedium):
    """A dispersive medium described by the Lorentz model.
    The frequency-dependence of the complex-valued permittivity is described by:

    Note
    ----
    .. math::

        \\epsilon(f) = \\epsilon_\\infty + \\sum_i
        \\frac{\\Delta\\epsilon_i f_i^2}{f_i^2 - 2jf\\delta_i - f^2}

    Example
    -------
    >>> lorentz_medium = Lorentz(eps_inf=2.0, coeffs=[(1,2,3), (4,5,6)])
    >>> eps = lorentz_medium.eps_model(200e12)
    """

    eps_inf: float = pd.Field(
        1.0,
        title="Epsilon at Infinity",
        description="Relative permittivity at infinite frequency (:math:`\\epsilon_\\infty`).",
        units=PERMITTIVITY,
    )

    coeffs: Tuple[Tuple[float, float, float], ...] = pd.Field(
        ...,
        title="Coefficients",
        description="List of (:math:`\\Delta\\epsilon_i, f_i, \\delta_i`) values for model.",
        units=(PERMITTIVITY, HERTZ, HERTZ),
    )

    @ensure_freq_in_range
    def eps_model(self, frequency: float) -> complex:
        """Complex-valued permittivity as a function of frequency."""

        eps = self.eps_inf + 0.0j
        for (de, f, delta) in self.coeffs:
            eps += (de * f**2) / (f**2 - 2j * frequency * delta - frequency**2)
        return eps

    @cached_property
    def pole_residue(self):
        """Representation of Medium as a pole-residue model."""

        poles = []
        for (de, f, delta) in self.coeffs:

            w = 2 * np.pi * f
            d = 2 * np.pi * delta

            if d > w:
                r = np.sqrt(d * d - w * w) + 0j
                a0 = -d + r
                c0 = de * w**2 / 4 / r
                a1 = -d - r
                c1 = -c0
                poles.extend(((a0, c0), (a1, c1)))
            else:
                r = np.sqrt(w * w - d * d)
                a = -d - 1j * r
                c = 1j * de * w**2 / 2 / r
                poles.append((a, c))

        return PoleResidue(
            eps_inf=self.eps_inf,
            poles=poles,
            frequency_range=self.frequency_range,
            name=self.name,
        )


class Drude(DispersiveMedium):
    """A dispersive medium described by the Drude model.
    The frequency-dependence of the complex-valued permittivity is described by:

    Note
    ----
    .. math::

        \\epsilon(f) = \\epsilon_\\infty - \\sum_i
        \\frac{ f_i^2}{f^2 + jf\\delta_i}

    Example
    -------
    >>> drude_medium = Drude(eps_inf=2.0, coeffs=[(1,2), (3,4)])
    >>> eps = drude_medium.eps_model(200e12)
    """

    eps_inf: float = pd.Field(
        1.0,
        title="Epsilon at Infinity",
        description="Relative permittivity at infinite frequency (:math:`\\epsilon_\\infty`).",
        units=PERMITTIVITY,
    )

    coeffs: Tuple[Tuple[float, pd.PositiveFloat], ...] = pd.Field(
        ...,
        title="Coefficients",
        description="List of (:math:`f_i, \\delta_i`) values for model.",
        units=(HERTZ, HERTZ),
    )

    @ensure_freq_in_range
    def eps_model(self, frequency: float) -> complex:
        """Complex-valued permittivity as a function of frequency."""

        eps = self.eps_inf + 0.0j
        for (f, delta) in self.coeffs:
            eps -= (f**2) / (frequency**2 + 1j * frequency * delta)
        return eps

    @cached_property
    def pole_residue(self):
        """Representation of Medium as a pole-residue model."""

        poles = []
        a0 = 0j

        for (f, delta) in self.coeffs:

            w = 2 * np.pi * f
            d = 2 * np.pi * delta

            c0 = (w**2) / 2 / d + 0j
            c1 = -c0
            a1 = -d + 0j

            poles.extend(((a0, c0), (a1, c1)))
        return PoleResidue(
            eps_inf=self.eps_inf,
            poles=poles,
            frequency_range=self.frequency_range,
            name=self.name,
        )


class Debye(DispersiveMedium):
    """A dispersive medium described by the Debye model.
    The frequency-dependence of the complex-valued permittivity is described by:

    Note
    ----
    .. math::

        \\epsilon(f) = \\epsilon_\\infty + \\sum_i
        \\frac{\\Delta\\epsilon_i}{1 - jf\\tau_i}

    Example
    -------
    >>> debye_medium = Debye(eps_inf=2.0, coeffs=[(1,2),(3,4)])
    >>> eps = debye_medium.eps_model(200e12)
    """

    eps_inf: float = pd.Field(
        1.0,
        title="Epsilon at Infinity",
        description="Relative permittivity at infinite frequency (:math:`\\epsilon_\\infty`).",
        units=PERMITTIVITY,
    )

    coeffs: Tuple[Tuple[float, pd.PositiveFloat], ...] = pd.Field(
        ...,
        title="Coefficients",
        description="List of (:math:`\\Delta\\epsilon_i, \\tau_i`) values for model.",
        units=(PERMITTIVITY, SECOND),
    )

    @ensure_freq_in_range
    def eps_model(self, frequency: float) -> complex:
        """Complex-valued permittivity as a function of frequency."""

        eps = self.eps_inf + 0.0j
        for (de, tau) in self.coeffs:
            eps += de / (1 - 1j * frequency * tau)
        return eps

    @cached_property
    def pole_residue(self):
        """Representation of Medium as a pole-residue model."""

        poles = []
        for (de, tau) in self.coeffs:
            a = -2 * np.pi / tau + 0j
            c = -0.5 * de * a

            poles.append((a, c))

        return PoleResidue(
            eps_inf=self.eps_inf,
            poles=poles,
            frequency_range=self.frequency_range,
            name=self.name,
        )


IsotropicMediumType = Union[Medium, PoleResidue, Sellmeier, Lorentz, Debye, Drude]


class AnisotropicMedium(AbstractMedium):
    """Diagonally anisotropic medium.

    Note
    ----
    Only diagonal anisotropy is currently supported.

    Example
    -------
    >>> medium_xx = Medium(permittivity=4.0)
    >>> medium_yy = Medium(permittivity=4.1)
    >>> medium_zz = Medium(permittivity=3.9)
    >>> anisotropic_dielectric = AnisotropicMedium(xx=medium_xx, yy=medium_yy, zz=medium_zz)
    """

    xx: IsotropicMediumType = pd.Field(
        ...,
        title="XX Component",
        description="Medium describing the xx-component of the diagonal permittivity tensor.",
        discriminator=TYPE_TAG_STR,
    )

    yy: IsotropicMediumType = pd.Field(
        ...,
        title="YY Component",
        description="Medium describing the yy-component of the diagonal permittivity tensor.",
        discriminator=TYPE_TAG_STR,
    )

    zz: IsotropicMediumType = pd.Field(
        ...,
        title="ZZ Component",
        description="Medium describing the zz-component of the diagonal permittivity tensor.",
        discriminator=TYPE_TAG_STR,
    )

    @cached_property
    def components(self) -> Dict[str, Medium]:
        """Dictionary of diagonal medium components."""
        return dict(xx=self.xx, yy=self.yy, zz=self.zz)

    @ensure_freq_in_range
    def eps_model(self, frequency: float) -> complex:
        """Complex-valued permittivity as a function of frequency."""

        eps_xx = self.xx.eps_model(frequency)
        eps_yy = self.yy.eps_model(frequency)
        eps_zz = self.zz.eps_model(frequency)
        return np.mean((eps_xx, eps_yy, eps_zz))

    @ensure_freq_in_range
    def eps_diagonal(self, frequency: float) -> Tuple[complex, complex, complex]:
        """Main diagonal of the complex-valued permittivity tensor as a function of frequency."""

        eps_xx = self.xx.eps_model(frequency)
        eps_yy = self.yy.eps_model(frequency)
        eps_zz = self.zz.eps_model(frequency)
        return (eps_xx, eps_yy, eps_zz)

    @add_ax_if_none
    def plot(self, freqs: float, ax: Ax = None) -> Ax:
        """Plot n, k of a :class:`Medium` as a function of frequency."""

        freqs = np.array(freqs)
        freqs_thz = freqs / 1e12

        for label, medium_component in zip(("xx", "yy", "zz"), (self.xx, self.yy, self.zz)):

            eps_complex = medium_component.eps_model(freqs)
            n, k = AbstractMedium.eps_complex_to_nk(eps_complex)
            ax.plot(freqs_thz, n, label=f"n, eps_{label}")
            ax.plot(freqs_thz, k, label=f"k, eps_{label}")

        ax.set_xlabel("frequency (THz)")
        ax.set_title("medium dispersion")
        ax.legend()
        ax.set_aspect("auto")
        return ax


# types of mediums that can be used in Simulation and Structures

MediumType = Union[
    Medium,
    CustomMedium,
    AnisotropicMedium,
    PECMedium,
    PoleResidue,
    Sellmeier,
    Lorentz,
    Debye,
    Drude,
]
