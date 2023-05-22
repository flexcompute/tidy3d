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
from .types import PoleAndResidue, Ax, FreqBound, TYPE_TAG_STR, InterpMethod, Bound, ArrayComplex4D
from .types import Axis, TensorReal
from .data.dataset import PermittivityDataset
from .data.data_array import ScalarFieldDataArray
from .viz import add_ax_if_none
from .geometry import Geometry
from .validators import validate_name_str
from ..constants import C_0, pec_val, EPSILON_0, LARGE_NUMBER, fp_eps
from ..constants import HERTZ, CONDUCTIVITY, PERMITTIVITY, RADPERSEC, MICROMETER, SECOND
from ..exceptions import ValidationError, SetupError
from ..log import log
from .transformation import RotationType


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
        # don't warn for evaluating infinite frequency
        if is_inf_scalar:
            return eps_model(self, frequency)
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

    def eps_comp(self, row: Axis, col: Axis, frequency: float) -> complex:
        """Single component of the complex-valued permittivity tensor as a function of frequency.

        Parameters
        ----------
        row : int
            Component's row in the permittivity tensor (0, 1, or 2 for x, y, or z respectively).
        col : int
            Component's column in the permittivity tensor (0, 1, or 2 for x, y, or z respectively).
        frequency : float
            Frequency to evaluate permittivity at (Hz).

        Returns
        -------
        complex
           Element of the relative permittivity tensor evaluated at ``frequency``.
        """

        # This only needs to be overwritten for anisotropic materials
        if row == col:
            return self.eps_model(frequency)
        return 0j

    @cached_property
    @abstractmethod
    def n_cfl(self):
        """To ensure a stable FDTD simulation, it is essential to select an appropriate
        time step size in accordance with the CFL condition. The maximal time step
        size is inversely proportional to the speed of light in the medium, and thus
        proportional to the index of refraction. However, for dispersive medium,
        anisotropic medium, and other more complicated media, there are complications in
        deciding on the choice of the index of refraction.

        This property computes the index of refraction related to CFL condition, so that
        the FDTD with this medium is stable when the time step size that doesn't take
        material factor into account is multiplied by ``n_cfl``.
        """

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

    @ensure_freq_in_range
    def sigma_model(self, freq: float) -> complex:
        """Complex-valued conductivity as a function of frequency.

        Parameters
        ----------
        freq: float
            Frequency to evaluate conductivity at (Hz).

        Returns
        -------
        complex
            Complex conductivity at this frequency.
        """
        omega = freq * 2 * np.pi
        eps_complex = self.eps_model(freq)
        eps_inf = self.eps_model(np.inf)
        sigma = (eps_inf - eps_complex) * 1j * omega * EPSILON_0
        return sigma


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

    @cached_property
    def n_cfl(self):
        """This property computes the index of refraction related to CFL condition, so that
        the FDTD with this medium is stable when the time step size that doesn't take
        material factor into account is multiplied by ``n_cfl``.
        """
        return 1.0


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

    @cached_property
    def n_cfl(self):
        """This property computes the index of refraction related to CFL condition, so that
        the FDTD with this medium is stable when the time step size that doesn't take
        material factor into account is multiplied by ``n_cfl``.

        For dispersiveless medium, it equals ``sqrt(permittivity)``.
        """
        return np.sqrt(self.permittivity)

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

    @cached_property
    def n_cfl(self):
        """This property computes the index of refraction related to CFL condition, so that
        the FDTD with this medium is stable when the time step size that doesn't take
        material factor into account is multiplied by ``n_cfl```.

        For dispersiveless custom medium, it equals ``min[sqrt(eps_inf)]``, where ``min``
        is performed over all components and spatial points.
        """
        eps_array_min = [
            float(np.min(eps_array.real))
            for _, eps_array in self.eps_dataset.field_components.items()
        ]
        return np.sqrt(min(eps_array_min))

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
    ) -> Tuple[ArrayComplex4D, ArrayComplex4D, ArrayComplex4D]:
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
        Tuple[np.ndarray, np.ndarray, np.ndarray]
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

    def eps_comp_on_grid(
        self,
        row: Axis,
        col: Axis,
        frequency: float,
        coords: Coords,
    ) -> ArrayComplex4D:
        """Spatial profile of a single component of the complex-valued permittivity tensor at
        ``frequency`` interpolated at the supplied coordinates.

        Parameters
        ----------
        row : int
            Component's row in the permittivity tensor (0, 1, or 2 for x, y, or z respectively).
        col : int
            Component's column in the permittivity tensor (0, 1, or 2 for x, y, or z respectively).
        frequency : float
            Frequency to evaluate permittivity at (Hz).
        coords : :class:`.Coords`
            The grid point coordinates over which interpolation is performed.

        Returns
        -------
        np.ndarray
            Single component of the complex-valued permittivity tensor at ``frequency`` interpolated
            at the supplied coordinates.
        """

        if row == col:
            eps_freq = self.eps_dataset_freq(frequency)
            interp_shape = [len(coord_comp) for coord_comp in coords.to_list]
            comp = ["eps_xx", "eps_yy", "eps_zz"][row]
            eps = self._interp(eps_freq.field_components[comp], coords, self.interp_method)
            eps = np.array(eps).reshape(interp_shape)
            return eps
        return 0

    @ensure_freq_in_range
    def eps_diagonal(self, frequency: float) -> Tuple[complex, complex, complex]:
        """Main diagonal of the complex-valued permittivity tensor
        at ``frequency``. Spatially, we take max{|eps|}, so that autoMesh generation
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
        """Spatial and polarizaiton average of complex-valued permittivity
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

    @cached_property
    def n_cfl(self):
        """This property computes the index of refraction related to CFL condition, so that
        the FDTD with this medium is stable when the time step size that doesn't take
        material factor into account is multiplied by ``n_cfl``.

        For PoleResidue model, it equals ``sqrt(eps_inf)``
        [https://ieeexplore.ieee.org/document/9082879].
        """
        return np.sqrt(self.pole_residue.eps_inf)

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

    eps_inf: pd.PositiveFloat = pd.Field(
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

    @classmethod
    def from_medium(cls, medium: Medium) -> "PoleResidue":
        """Convert a :class:`.Medium` to a pole residue model.

        Parameters
        ----------
        medium: :class:`.Medium`
            The medium with permittivity and conductivity to convert.

        Returns
        -------
        :class:`.PoleResidue`
            The pole residue equivalent.
        """
        poles = [(0, medium.conductivity / (2 * EPSILON_0))]
        return PoleResidue(
            eps_inf=medium.permittivity, poles=poles, frequency_range=medium.frequency_range
        )

    def to_medium(self) -> Medium:
        """Convert to a :class:`.Medium`.
        Requires the pole residue model to only have a pole at 0 frequency,
        corresponding to a constant conductivity term.

        Returns
        -------
        :class:`.Medium`
            The non-dispersive equivalent with constant permittivity and conductivity.
        """
        res = 0
        for (a, c) in self.poles:
            if abs(a) > fp_eps:
                raise ValidationError("Cannot convert dispersive 'PoleResidue' to 'Medium'.")
            res += (c + np.conj(c)) / 2
        sigma = res * 2 * EPSILON_0
        return Medium(
            permittivity=self.eps_inf,
            conductivity=np.real(sigma),
            frequency_range=self.frequency_range,
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

        wvl = C_0 / np.array(frequency)
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
        freq : float
            Frequency at which ``n`` and ``dn_dwvl`` are sampled.

        Returns
        -------
        :class:`Sellmeier`
            Single-pole Sellmeier medium with the prvoided refractive index and index dispersion
            valuesat at the prvoided frequency.
        """

        if dn_dwvl >= 0:
            raise ValidationError("Dispersion ``dn_dwvl`` must be smaller than zero.")
        if n < 1:
            raise ValidationError("Refractive index ``n`` cannot be smaller than one.")

        wvl = C_0 / np.array(freq)
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

    eps_inf: pd.PositiveFloat = pd.Field(
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

    eps_inf: pd.PositiveFloat = pd.Field(
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

    eps_inf: pd.PositiveFloat = pd.Field(
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

    @cached_property
    def n_cfl(self):
        """This property computes the index of refraction related to CFL condition, so that
        the FDTD with this medium is stable when the time step size that doesn't take
        material factor into account is multiplied by ``n_cfl``.

        For this medium, it takes the minimal of ``n_clf`` in all components.
        """
        return min((mat_component.n_cfl for mat_component in self.components.values()))

    @ensure_freq_in_range
    def eps_model(self, frequency: float) -> complex:
        """Complex-valued permittivity as a function of frequency."""

        return np.mean(self.eps_diagonal(frequency), axis=0)

    @ensure_freq_in_range
    def eps_diagonal(self, frequency: float) -> Tuple[complex, complex, complex]:
        """Main diagonal of the complex-valued permittivity tensor as a function of frequency."""

        eps_xx = self.xx.eps_model(frequency)
        eps_yy = self.yy.eps_model(frequency)
        eps_zz = self.zz.eps_model(frequency)
        return (eps_xx, eps_yy, eps_zz)

    def eps_comp(self, row: Axis, col: Axis, frequency: float) -> complex:
        """Single component the complex-valued permittivity tensor as a function of frequency.

        Parameters
        ----------
        row : int
            Component's row in the permittivity tensor (0, 1, or 2 for x, y, or z respectively).
        col : int
            Component's column in the permittivity tensor (0, 1, or 2 for x, y, or z respectively).
        frequency : float
            Frequency to evaluate permittivity at (Hz).

        Returns
        -------
        complex
           Element of the relative permittivity tensor evaluated at ``frequency``.
        """

        if row != col:
            return 0j
        cmp = "xyz"[row]
        field_name = cmp + cmp
        return self.components[field_name].eps_model(frequency)

    @add_ax_if_none
    def plot(self, freqs: float, ax: Ax = None) -> Ax:
        """Plot n, k of a :class:`.Medium` as a function of frequency."""

        freqs = np.array(freqs)
        freqs_thz = freqs / 1e12

        for label, medium_component in self.elements.items():

            eps_complex = medium_component.eps_model(freqs)
            n, k = AbstractMedium.eps_complex_to_nk(eps_complex)
            ax.plot(freqs_thz, n, label=f"n, eps_{label}")
            ax.plot(freqs_thz, k, label=f"k, eps_{label}")

        ax.set_xlabel("frequency (THz)")
        ax.set_title("medium dispersion")
        ax.legend()
        ax.set_aspect("auto")
        return ax

    @property
    def elements(self) -> Dict[str, IsotropicMediumType]:
        """The diagonal elements of the medium as a dictionary."""
        return dict(xx=self.xx, yy=self.yy, zz=self.zz)


class FullyAnisotropicMedium(AbstractMedium):
    """Fully anisotropic medium including all 9 components of the permittivity and conductivity
    tensors. Provided permittivity tensor and the symmetric part of the conductivity tensor must
    have coinciding main directions. A non-symmetric conductivity tensor can be used to model
    magneto-optic effects. Note that dispersive properties and subpixel averaging are currently not
    supported for fully anisotropic materials.

    Note
    ----
    Simulations involving fully anisotropic materials are computationally more intensive, thus,
    they take longer time to complete. This increase strongly depends on the filling fraction of
    the simulation domain by fully anisotropic materials, varying approximately in the range from
    1.5 to 5. Cost of running a simulation is adjusted correspondingly.

    Example
    -------
    >>> perm = [[2, 0, 0], [0, 1, 0], [0, 0, 3]]
    >>> cond = [[0.1, 0, 0], [0, 0, 0], [0, 0, 0]]
    >>> anisotropic_dielectric = FullyAnisotropicMedium(permittivity=perm, conductivity=cond)
    """

    permittivity: TensorReal = pd.Field(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        title="Permittivity",
        description="Relative permittivity tensor.",
        units=PERMITTIVITY,
    )

    conductivity: TensorReal = pd.Field(
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        title="Conductivity",
        description="Electric conductivity tensor. Defined such that the imaginary part "
        "of the complex permittivity at angular frequency omega is given by conductivity/omega.",
        units=CONDUCTIVITY,
    )

    @pd.validator("permittivity", always=True)
    def permittivity_spd_and_ge_one(cls, val):
        """Check that provided permittivity tensor is symmetric positive definite
        with eigenvalues >= 1.
        """

        if not np.allclose(val, np.transpose(val), atol=fp_eps):
            raise ValidationError("Provided permittivity tensor is not symmetric.")

        if np.any(np.linalg.eigvals(val) < 1 - fp_eps):
            raise ValidationError("Main diagonal of provided permittivity tensor is not >= 1.")

        return val

    @pd.validator("conductivity", always=True)
    def conductivity_ge_zero_and_commutes(cls, val, values):
        """Check that the symmetric part of conductivity tensor commutes with permittivity tensor
        (that is, simultaneously diagonalizable) with eigenvalues >= 0.
        """

        perm = values.get("permittivity")
        cond_sym = 0.5 * (val + val.T)
        comm_diff = np.abs(np.matmul(perm, cond_sym) - np.matmul(cond_sym, perm))

        if not np.allclose(comm_diff, 0, atol=fp_eps):
            raise ValidationError(
                "Main directions of conductivity and permittivity tensor do not coincide."
            )

        if np.any(np.linalg.eigvals(cond_sym) < -fp_eps):
            raise ValidationError("Main diagonal of provided conductivity tensor is not >= 0.")

        return val

    @classmethod
    def from_diagonal(cls, xx: Medium, yy: Medium, zz: Medium, rotation: RotationType):
        """Construct a fully anisotropic medium by rotating a diagonally ansisotropic medium.

        Parameters
        ----------
        xx : :class:`.Medium`
            Medium describing the xx-component of the diagonal permittivity tensor.
        yy : :class:`.Medium`
            Medium describing the yy-component of the diagonal permittivity tensor.
        zz : :class:`.Medium`
            Medium describing the zz-component of the diagonal permittivity tensor.
        rotation : :class:`.RotationType`
                Rotation applied to diagonal permittivity tensor.

        Returns
        -------
        :class:`FullyAnisotropicMedium`
            Resulting fully anisotropic medium.
        """

        permittivity_diag = np.diag([comp.permittivity for comp in [xx, yy, zz]]).tolist()
        conductivity_diag = np.diag([comp.conductivity for comp in [xx, yy, zz]]).tolist()

        permittivity = rotation.rotate_tensor(permittivity_diag)
        conductivity = rotation.rotate_tensor(conductivity_diag)

        return cls(permittivity=permittivity, conductivity=conductivity)

    @cached_property
    def _to_diagonal(self) -> AnisotropicMedium:
        """Construct a diagonally anisotropic medium from main components.

        Returns
        -------
        :class:`AnisotropicMedium`
            Resulting diagonally anisotropic medium.
        """

        perm, cond, _ = self.eps_sigma_diag

        return AnisotropicMedium(
            xx=Medium(permittivity=perm[0], conductivity=cond[0]),
            yy=Medium(permittivity=perm[1], conductivity=cond[1]),
            zz=Medium(permittivity=perm[2], conductivity=cond[2]),
        )

    @cached_property
    def eps_sigma_diag(
        self,
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float], TensorReal]:
        """Main components of permittivity and conductivity tensors and their directions."""

        perm_diag, vecs = np.linalg.eig(self.permittivity)
        cond_diag = np.diag(np.matmul(np.transpose(vecs), np.matmul(self.conductivity, vecs)))

        return (perm_diag, cond_diag, vecs)

    @ensure_freq_in_range
    def eps_model(self, frequency: float) -> complex:
        """Complex-valued permittivity as a function of frequency."""
        perm_diag, cond_diag, _ = self.eps_sigma_diag

        eps_diag = AbstractMedium.eps_sigma_to_eps_complex(perm_diag, cond_diag, frequency)
        return np.mean(eps_diag)

    @ensure_freq_in_range
    def eps_diagonal(self, frequency: float) -> Tuple[complex, complex, complex]:
        """Main diagonal of the complex-valued permittivity tensor as a function of frequency."""

        perm_diag, cond_diag, _ = self.eps_sigma_diag

        if not np.isscalar(frequency):
            perm_diag = perm_diag[:, None]
            cond_diag = cond_diag[:, None]
        return AbstractMedium.eps_sigma_to_eps_complex(perm_diag, cond_diag, frequency)

    def eps_comp(self, row: Axis, col: Axis, frequency: float) -> complex:
        """Single component the complex-valued permittivity tensor as a function of frequency.

        Parameters
        ----------
        row : int
            Component's row in the permittivity tensor (0, 1, or 2 for x, y, or z respectively).
        col : int
            Component's column in the permittivity tensor (0, 1, or 2 for x, y, or z respectively).
        frequency : float
            Frequency to evaluate permittivity at (Hz).

        Returns
        -------
        complex
           Element of the relative permittivity tensor evaluated at ``frequency``.
        """

        eps = self.permittivity[row][col]
        sig = self.conductivity[row][col]
        return AbstractMedium.eps_sigma_to_eps_complex(eps, sig, frequency)

    @cached_property
    def n_cfl(self):
        """This property computes the index of refraction related to CFL condition, so that
        the FDTD with this medium is stable when the time step size that doesn't take
        material factor into account is multiplied by ``n_cfl``.

        For this medium, it take the minimal of ``sqrt(permittivity)`` for main directions.
        """

        perm_diag, _, _ = self.eps_sigma_diag
        return min(np.sqrt(perm_diag))

    @add_ax_if_none
    def plot(self, freqs: float, ax: Ax = None) -> Ax:
        """Plot n, k of a :class:`FullyAnisotropicMedium` as a function of frequency."""

        diagonal_medium = self._to_diagonal
        ax = diagonal_medium.plot(freqs=freqs, ax=ax)
        _, _, directions = self.eps_sigma_diag

        # rename components from xx, yy, zz to 1, 2, 3 to avoid misleading
        # and add their directions
        for label, n_line, k_line, direction in zip(
            ("1", "2", "3"), ax.lines[-6::2], ax.lines[-5::2], directions.T
        ):
            direction_str = f"({direction[0]:.2f}, {direction[1]:.2f}, {direction[2]:.2f})"
            k_line.set_label(f"k, eps_{label} {direction_str}")
            n_line.set_label(f"n, eps_{label} {direction_str}")

        ax.legend()
        return ax


# types of mediums that can be used in Simulation and Structures

MediumType3D = Union[
    Medium,
    CustomMedium,
    AnisotropicMedium,
    PECMedium,
    PoleResidue,
    Sellmeier,
    Lorentz,
    Debye,
    Drude,
    FullyAnisotropicMedium,
]


class Medium2D(AbstractMedium):
    """2D diagonally anisotropic medium.

    Note
    ----
    Only diagonal anisotropy is currently supported.

    Example
    -------
    >>> drude_medium = Drude(eps_inf=2.0, coeffs=[(1,2), (3,4)])
    >>> medium2d = Medium2D(ss=drude_medium, tt=drude_medium)

    """

    ss: IsotropicMediumType = pd.Field(
        ...,
        title="SS Component",
        description="Medium describing the ss-component of the diagonal permittivity tensor. "
        "The ss-component refers to the in-plane dimension of the medium that is the first "
        "component in order of 'x', 'y', 'z'. "
        "If the 2D material is normal to the y-axis, for example, then this determines the "
        "xx-component of the corresponding 3D medium.",
        discriminator=TYPE_TAG_STR,
    )

    tt: IsotropicMediumType = pd.Field(
        ...,
        title="TT Component",
        description="Medium describing the tt-component of the diagonal permittivity tensor. "
        "The tt-component refers to the in-plane dimension of the medium that is the second "
        "component in order of 'x', 'y', 'z'. "
        "If the 2D material is normal to the y-axis, for example, then this determines the "
        "zz-component of the corresponding 3D medium.",
        discriminator=TYPE_TAG_STR,
    )

    @classmethod
    def _weighted_avg(cls, meds: List[IsotropicMediumType], weights: List[float]) -> PoleResidue:
        """Average ``meds`` with weights ``weights``."""
        eps_inf = 1
        poles = []
        for (med, weight) in zip(meds, weights):
            if isinstance(med, DispersiveMedium):
                pole_res = med.pole_residue
                eps_inf += weight * (med.pole_residue.eps_inf - 1)
            elif isinstance(med, Medium):
                pole_res = PoleResidue.from_medium(med)
                eps_inf += weight * (med.eps_model(np.inf) - 1)
            elif isinstance(med, PECMedium):
                pole_res = PoleResidue.from_medium(Medium(conductivity=LARGE_NUMBER))
            else:
                raise ValidationError("Invalid medium type for the components of 'Medium2D'.")
            poles += [(a, weight * c) for (a, c) in pole_res.poles]
        return PoleResidue(eps_inf=np.real(eps_inf), poles=poles)

    def volumetric_equivalent(
        self,
        axis: Axis,
        adjacent_media: Tuple[MediumType3D, MediumType3D],
        adjacent_dls: Tuple[float, float],
    ) -> AnisotropicMedium:
        """Produces a 3D volumetric equivalent medium. The new medium has thickness equal to
        the average of the ``dls`` in the ``axis`` direction.
        The ss and tt components of the 2D material are mapped in order onto the xx, yy, and
        zz components of the 3D material, excluding the ``axis`` component. The conductivity
        and residues (in the case of a dispersive 2D material) are rescaled by ``1/dl``.
        The neighboring media ``neighbors`` enter in as a background for the resulting
        volumetric equivalent.


        Parameters
        ----------
        axis : Axis
            Index (0, 1, or 2 for x, y, or z respectively) of the normal direction to the
            2D material.
        adjacent_media : Tuple[MediumType3D, MediumType3D]
            The neighboring media on either side of the 2D material.
            The first element is directly on the - side of the 2D material in the supplied axis,
            and the second element is directly on the + side.
        adjacent_dls : Tuple[float, float]
            Each dl represents twice the thickness of the desired volumetric model on the
            respective side of the 2D material.

        Returns
        -------
        :class:`.AnisotropicMedium`
            The 3D material corresponding to this 2D material.
        """

        def get_component(med: MediumType3D, comp: Axis) -> IsotropicMediumType:
            """Extract the ``comp`` component of ``med``."""
            if isinstance(med, AnisotropicMedium):
                dim = "xyz"[comp]
                element_name = dim + dim
                return med.elements[element_name]
            return med

        def get_background(comp: Axis) -> PoleResidue:
            """Get the background medium appropriate for the ``comp`` component."""
            meds = [get_component(med=med, comp=comp) for med in adjacent_media]
            # the Yee site for the E field in the normal direction is fully contained
            # in the medium on the + side
            if comp == axis:
                return meds[1]
            weights = np.array(adjacent_dls) / np.sum(adjacent_dls)
            return self._weighted_avg(meds, weights)

        dl = (adjacent_dls[0] + adjacent_dls[1]) / 2
        media_bg = [get_background(comp=i) for i in range(3)]

        # perform weighted average of planar media transverse dimensions with the
        # respective background media
        media_fg_plane = list(self.elements.values())
        _, media_bg_plane = Geometry.pop_axis(media_bg, axis=axis)
        media_fg_weighted = [
            self._weighted_avg([media_bg, media_fg], [1, 1 / dl])
            for media_bg, media_fg in zip(media_bg_plane, media_fg_plane)
        ]

        # combine the two weighted, planar media with the background medium and put in the xyz basis
        media_3d = Geometry.unpop_axis(
            ax_coord=media_bg[axis], plane_coords=media_fg_weighted, axis=axis
        )
        media_3d_kwargs = {dim + dim: medium for dim, medium in zip("xyz", media_3d)}
        return AnisotropicMedium(**media_3d_kwargs, frequency_range=self.frequency_range)

    def to_anisotropic_medium(self, axis: Axis, thickness: float) -> AnisotropicMedium:
        """Generate a 3D :class:`.AnisotropicMedium` equivalent of a given thickness.

        Parameters
        ----------
        axis: Axis
            The normal axis to the 2D medium.
        thickness: float
            The thickness of the desired 3D medium.

        Returns
        -------
        :class:`.AnisotropicMedium`
            The 3D equivalent of this 2D medium.
        """
        media = list(self.elements.values())
        print(media)
        media_weighted = [self._weighted_avg([medium], [1 / thickness]) for medium in media]
        media_3d = Geometry.unpop_axis(ax_coord=Medium(), plane_coords=media_weighted, axis=axis)
        media_3d_kwargs = {dim + dim: medium for dim, medium in zip("xyz", media_3d)}
        return AnisotropicMedium(**media_3d_kwargs, frequency_range=self.frequency_range)

    def to_pole_residue(self, thickness: float) -> PoleResidue:
        """Generate a :class:`.PoleResidue` equivalent of a given thickness.
        The 2D medium to be isotropic in-plane (otherwise the components are averaged).

        Parameters
        ----------
        thickness: float
            The thickness of the desired 3D medium.

        Returns
        -------
        :class:`.PoleResidue`
            The 3D equivalent pole residue model of this 2D medium.
        """
        return self._weighted_avg(
            [self.ss, self.tt], [1 / (2 * thickness), 1 / (2 * thickness)]
        ).updated_copy(frequency_range=self.frequency_range)

    def to_medium(self, thickness: float) -> Medium:
        """Generate a :class:`.Medium` equivalent of a given thickness.
        The 2D medium should be isotropic in-plane (otherwise the components are averaged)
        and non-dispersive besides a constant conductivity.

        Parameters
        ----------
        thickness: float
            The thickness of the desired 3D medium.

        Returns
        -------
        :class:`.Medium`
            The 3D equivalent of this 2D medium.
        """
        return self.to_pole_residue(thickness=thickness).to_medium()

    @classmethod
    def from_medium(cls, medium: Medium, thickness: float) -> Medium2D:
        """Generate a :class:`.Medium2D` equivalent of a :class:`.Medium`
        with a given thickness.

        Parameters
        ----------
        medium: :class:`.Medium`
            The 3D medium to convert.
        thickness : float
            The thickness of the 3D material.

        Returns
        -------
        :class:`.Medium2D`
            The 2D equivalent of the given 3D medium.
        """
        med = cls._weighted_avg([medium], [thickness])
        return Medium2D(ss=med, tt=med, frequency_range=medium.frequency_range)

    @classmethod
    def from_dispersive_medium(cls, medium: DispersiveMedium, thickness: float) -> Medium2D:
        """Generate a :class:`.Medium2D` equivalent of a :class:`.DispersiveMedium`
        with a given thickness.

        Parameters
        ----------
        medium: :class:`.DispersiveMedium`
            The 3D dispersive medium to convert.
        thickness : float
            The thickness of the 3D material.

        Returns
        -------
        :class:`.Medium2D`
            The 2D equivalent of the given 3D medium.
        """
        med = cls._weighted_avg([medium], [thickness])
        return Medium2D(ss=med, tt=med, frequency_range=medium.frequency_range)

    @classmethod
    def from_anisotropic_medium(
        cls, medium: AnisotropicMedium, axis: Axis, thickness: float
    ) -> Medium2D:
        """Generate a :class:`.Medium2D` equivalent of a :class:`.AnisotropicMedium`
        with given normal axis and thickness. The ``ss`` and ``tt`` components of the resulting
        2D medium correspond to the first of the ``xx``, ``yy``, and ``zz`` components of
        the 3D medium, with the ``axis`` component removed.

        Parameters
        ----------
        medium: :class:`.AnisotropicMedium`
            The 3D anisotropic medium to convert.
        axis: :class:`.Axis`
            The normal axis to the 2D material.
        thickness : float
            The thickness of the 3D material.

        Returns
        -------
        :class:`.Medium2D`
            The 2D equivalent of the given 3D medium.
        """
        media = list(medium.elements.values())
        _, media_plane = Geometry.pop_axis(media, axis=axis)
        media_plane_scaled = []
        for _, med in enumerate(media_plane):
            media_plane_scaled.append(cls._weighted_avg([med], [thickness]))
        media_kwargs = {dim + dim: medium for dim, medium in zip("st", media_plane_scaled)}
        return Medium2D(**media_kwargs, frequency_range=medium.frequency_range)

    @ensure_freq_in_range
    def eps_model(self, frequency: float) -> complex:
        """Complex-valued permittivity as a function of frequency."""
        return np.mean(self.eps_diagonal(frequency=frequency), axis=0)

    @ensure_freq_in_range
    def eps_diagonal(self, frequency: float) -> Tuple[complex, complex]:
        """Main diagonal of the complex-valued permittivity tensor as a function of frequency."""
        log.warning("Evaluating permittivity of a 'Medium2D' is unphysical.")

        eps_ss = self.ss.eps_model(frequency)
        eps_tt = self.tt.eps_model(frequency)
        return (eps_ss, eps_tt)

    @add_ax_if_none
    def plot(self, freqs: float, ax: Ax = None) -> Ax:
        """Plot n, k of a :class:`.Medium` as a function of frequency."""
        log.warning(
            "The refractive index of a 'Medium2D' is unphysical. "
            "Use 'Medium2D.plot_sigma' instead to plot surface conductivty, or call "
            "'Medium2D.to_anisotropic_medium' or 'Medium2D.to_pole_residue' first "
            "to obtain the physical refractive index."
        )

        freqs = np.array(freqs)
        freqs_thz = freqs / 1e12

        for label, medium_component in self.elements.items():

            eps_complex = medium_component.eps_model(freqs)
            n, k = AbstractMedium.eps_complex_to_nk(eps_complex)
            ax.plot(freqs_thz, n, label=f"n, eps_{label}")
            ax.plot(freqs_thz, k, label=f"k, eps_{label}")

        ax.set_xlabel("frequency (THz)")
        ax.set_title("medium dispersion")
        ax.legend()
        ax.set_aspect("auto")
        return ax

    @add_ax_if_none
    def plot_sigma(self, freqs: float, ax: Ax = None) -> Ax:
        """Plot the surface conductivity of the 2D material."""
        freqs = np.array(freqs)
        freqs_thz = freqs / 1e12

        for label, medium_component in self.elements.items():
            sigma = medium_component.sigma_model(freqs)
            ax.plot(freqs_thz, np.real(sigma) * 1e6, label=f"Re($\\sigma$) ($\\mu$S), eps_{label}")
            ax.plot(freqs_thz, np.imag(sigma) * 1e6, label=f"Im($\\sigma$) ($\\mu$S), eps_{label}")

        ax.set_xlabel("frequency (THz)")
        ax.set_title("surface conductivity")
        ax.legend()
        ax.set_aspect("auto")
        return ax

    @ensure_freq_in_range
    def sigma_model(self, freq: float) -> complex:
        """Complex-valued conductivity as a function of frequency.

        Parameters
        ----------
        freq: float
            Frequency to evaluate conductivity at (Hz).

        Returns
        -------
        complex
            Complex conductivity at this frequency.
        """
        return np.mean([self.ss.sigma_model(freq), self.tt.sigma_model(freq)], axis=0)

    @property
    def elements(self) -> Dict[str, IsotropicMediumType]:
        """The diagonal elements of the 2D medium as a dictionary."""
        return dict(ss=self.ss, tt=self.tt)

    @cached_property
    def n_cfl(self):
        """This property computes the index of refraction related to CFL condition, so that
        the FDTD with this medium is stable when the time step size that doesn't take
        material factor into account is multiplied by ``n_cfl``.
        """
        return 1.0


# types of mediums that can be used in Simulation and Structures

MediumType = Union[MediumType3D, Medium2D]
