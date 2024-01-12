"""Defines perturbations to properties of the medium / materials"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Union, Tuple, List
import functools

import pydantic as pd
import numpy as np

# import xarray as xr

# import matplotlib.pyplot as plt

from .data.data_array import SpatialDataArray, HeatDataArray, ChargeDataArray
from .base import Tidy3dBaseModel, cached_property
from ..constants import KELVIN, CMCUBE, PERCMCUBE, inf
from ..log import log
from ..components.types import Ax, ArrayLike, Complex, FieldVal, InterpMethod, TYPE_TAG_STR
from ..components.viz import add_ax_if_none

""" Generic perturbation classes """


class AbstractPerturbation(ABC, Tidy3dBaseModel):
    """Abstract class for a generic perturbation."""

    @cached_property
    @abstractmethod
    def perturbation_range(self) -> Union[Tuple[float, float], Tuple[Complex, Complex]]:
        """Perturbation range."""

    @cached_property
    @abstractmethod
    def is_complex(self) -> bool:
        """Whether perturbation is complex valued."""

    @staticmethod
    def _linear_range(interval: Tuple[float, float], ref: float, coeff: Union[float, Complex]):
        """Find value range for a linear perturbation."""
        if coeff in (0, 0j):  # to avoid 0*inf
            return np.array([0, 0])
        return tuple(np.sort(coeff * (np.array(interval) - ref)))

    @staticmethod
    def _get_val(
        field: Union[ArrayLike[float], ArrayLike[Complex], SpatialDataArray], val: FieldVal
    ) -> Union[ArrayLike[float], ArrayLike[Complex], SpatialDataArray]:
        """Get specified value from a field."""

        if val == "real":
            return np.real(field)

        if val == "imag":
            return np.imag(field)

        if val == "abs":
            return np.abs(field)

        if val == "abs^2":
            return np.abs(field) ** 2

        if val == "phase":
            return np.arctan2(np.real(field), np.imag(field))

        raise ValueError(
            "Unknown 'val' key. Argument 'val' can take values 'real', 'imag', 'abs', "
            "'abs^2', or 'phase'."
        )

    @staticmethod
    def _array_type(value: Union[ArrayLike[float], ArrayLike[Complex], SpatialDataArray]) -> str:
        """Check whether variable is scalar, array, or spatial array."""
        if isinstance(value, SpatialDataArray):
            return "spatial"
        if np.ndim(value) == 0:
            return "scalar"
        return "array"


""" Elementary heat perturbation classes """


def ensure_temp_in_range(
    sample: Callable[
        Union[ArrayLike[float], SpatialDataArray],
        Union[ArrayLike[float], ArrayLike[Complex], SpatialDataArray],
    ]
) -> Callable[
    Union[ArrayLike[float], SpatialDataArray],
    Union[ArrayLike[float], ArrayLike[Complex], SpatialDataArray],
]:
    """Decorate ``sample`` to log warning if temperature supplied is out of bounds."""

    @functools.wraps(sample)
    def _sample(
        self, temperature: Union[ArrayLike[float], SpatialDataArray]
    ) -> Union[ArrayLike[float], ArrayLike[Complex], SpatialDataArray]:
        """New sample function."""

        if np.iscomplexobj(temperature):
            raise ValueError("Cannot pass complex 'temperature' to 'sample()'")

        temp_min, temp_max = self.temperature_range
        temperature_numpy = np.array(temperature)
        if np.any(temperature_numpy < temp_min) or np.any(temperature_numpy > temp_max):
            log.warning(
                "Temperature passed to 'HeatPerturbation.sample()'"
                f"is outside of 'HeatPerturbation.temperature_range' = {self.temperature_range}"
            )
        return sample(self, temperature)

    return _sample


class HeatPerturbation(AbstractPerturbation):
    """Abstract class for heat perturbation."""

    temperature_range: Tuple[pd.NonNegativeFloat, pd.NonNegativeFloat] = pd.Field(
        (0, inf),
        title="Temperature range",
        description="Temparature range in which perturbation model is valid.",
        units=KELVIN,
    )

    @abstractmethod
    def sample(
        self, temperature: Union[ArrayLike[float], SpatialDataArray]
    ) -> Union[ArrayLike[float], ArrayLike[Complex], SpatialDataArray]:
        """Sample perturbation.

        Parameters
        ----------
        temperature : Union[ArrayLike[float], SpatialDataArray]
            Temperature sample point(s).

        Returns
        -------
        Union[ArrayLike[float], ArrayLike[Complex], SpatialDataArray]
            Sampled perturbation value(s).
        """

    @add_ax_if_none
    def plot(
        self,
        temperature: ArrayLike[float],
        val: FieldVal = "real",
        ax: Ax = None,
    ) -> Ax:
        """Plot perturbation using provided temperature sample points.

        Parameters
        ----------
        temperature : ArrayLike[float]
            Array of temperature sample points.
        val : Literal['real', 'imag', 'abs', 'abs^2', 'phase'] = 'real'
            Which part of the field to plot.
        ax : matplotlib.axes._subplots.Axes = None
            Matplotlib axes to plot on, if not specified, one is created.

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.
        """

        temperature_numpy = np.array(temperature)

        values = self.sample(temperature_numpy)
        values = self._get_val(values, val)

        ax.plot(temperature_numpy, values)
        ax.set_xlabel("temperature (K)")
        ax.set_ylabel(f"{val}(perturbation value)")
        ax.set_title("temperature dependence")
        ax.set_aspect("auto")

        return ax


class LinearHeatPerturbation(HeatPerturbation):
    """Specifies parameter's perturbation due to thermal effects as a linear function of
    temperature:

    Note
    ----
    .. math::

        \\Delta X (T) = \\text{coeff} \\times (T - \\text{temperature\\_ref}),

    where ``coeff`` is the parameter's sensitivity (thermo-optic coefficient) to temperature and
    ``temperature_ref`` is the reference temperature point. A temperature range in which such
    a model is deemed accurate may be provided as a field ``temperature_range``
    (default: ``[0, inf]``). Wherever is applied, Tidy3D will check that the parameter's value
    does not go out of its physical bounds within ``temperature_range`` due to perturbations and
    raise a warning if this check fails. A warning is also issued if the perturbation model is
    evaluated outside of ``temperature_range``.

    Example
    -------
    >>> heat_perturb = LinearHeatPerturbation(
    ...     temperature_ref=300,
    ...     coeff=0.0001,
    ...     temperature_range=[200, 500],
    ... )
    """

    temperature_ref: pd.NonNegativeFloat = pd.Field(
        ...,
        title="Reference temperature",
        description="Temperature at which perturbation is zero.",
        units=KELVIN,
    )

    coeff: Union[float, Complex] = pd.Field(
        ...,
        title="Thermo-optic Coefficient",
        description="Sensitivity (derivative) of perturbation with respect to temperature.",
        units=f"1/{KELVIN}",
    )

    @cached_property
    def perturbation_range(self) -> Union[Tuple[float, float], Tuple[Complex, Complex]]:
        """Range of possible perturbation values in the provided ``temperature_range``."""
        return self._linear_range(self.temperature_range, self.temperature_ref, self.coeff)

    @ensure_temp_in_range
    def sample(
        self, temperature: Union[ArrayLike[float], SpatialDataArray]
    ) -> Union[ArrayLike[float], ArrayLike[Complex], SpatialDataArray]:
        """Sample perturbation at temperature points.

        Parameters
        ----------
        temperature : Union[ArrayLike[float], SpatialDataArray]
            Temperature sample point(s).

        Returns
        -------
        Union[ArrayLike[float], ArrayLike[Complex], SpatialDataArray]
            Sampled perturbation value(s).
        """

        # convert to numpy if not spatial data array
        t_vals = np.array(temperature) if self._array_type(temperature) == "array" else temperature

        return self.coeff * (t_vals - self.temperature_ref)

    @cached_property
    def is_complex(self) -> bool:
        """Whether perturbation is complex valued."""
        return np.iscomplex(self.coeff)


class CustomHeatPerturbation(HeatPerturbation):
    """Specifies parameter's perturbation due to thermal effects as a custom function of
    temperature defined as an array of perturbation values at sample temperature points. The linear
    interpolation is used to calculate perturbation values between sample temperature points. For
    temperature values outside of the provided sample region the perturbation value is extrapolated
    as a constant.
    The temperature range, ``temperature_range``, in which the perturbation model is assumed to be
    accurate is calculated automatically as the minimal and maximal sample temperature points.
    Wherever is applied, Tidy3D will check that the parameter's value
    does not go out of its physical bounds within ``temperature_range`` due to perturbations and
    raise a warning if this check fails. A warning is also issued if the perturbation model is
    evaluated outside of ``temperature_range``.

    Example
    -------
    >>> from tidy3d import HeatDataArray
    >>> perturbation_data = HeatDataArray([0.001, 0.002, 0.004], coords=dict(T=[250, 300, 350]))
    >>> heat_perturb = CustomHeatPerturbation(
    ...     perturbation_values=perturbation_data
    ... )
    """

    perturbation_values: HeatDataArray = pd.Field(
        ...,
        title="Perturbation Values",
        description="Sampled perturbation values.",
    )

    temperature_range: Tuple[pd.NonNegativeFloat, pd.NonNegativeFloat] = pd.Field(
        None,
        title="Temperature range",
        description="Temparature range in which perturbation model is valid. For "
        ":class:`.CustomHeatPerturbation` this field is computed automatically based on "
        "temperature sample points provided in ``perturbation_values``.",
        units=KELVIN,
    )

    interp_method: InterpMethod = pd.Field(
        "linear",
        title="Interpolation method",
        description="Interpolation method to obtain perturbation values between sample points.",
    )

    @cached_property
    def perturbation_range(self) -> Union[Tuple[float, float], Tuple[Complex, Complex]]:
        """Range of possible parameter perturbation values."""
        return np.min(self.perturbation_values).item(), np.max(self.perturbation_values).item()

    @pd.root_validator(skip_on_failure=True)
    def compute_temperature_range(cls, values):
        """Compute and set temperature range based on provided ``perturbation_values``."""
        if values["temperature_range"] is not None:
            log.warning(
                "Temperature range for 'CustomHeatPerturbation' is calculated automatically "
                "based on provided 'perturbation_values'. Provided 'temperature_range' will be "
                "overwritten."
            )

        perturbation_values = values["perturbation_values"]

        # .item() to convert to a scalar
        temperature_range = (
            np.min(perturbation_values.coords["T"]).item(),
            np.max(perturbation_values.coords["T"]).item(),
        )

        values.update({"temperature_range": temperature_range})

        return values

    @ensure_temp_in_range
    def sample(
        self, temperature: Union[ArrayLike[float], SpatialDataArray]
    ) -> Union[ArrayLike[float], ArrayLike[Complex], SpatialDataArray]:
        """Sample perturbation at provided temperature points.

        Parameters
        ----------
        temperature : Union[ArrayLike[float], SpatialDataArray]
            Temperature sample point(s).

        Returns
        -------
        Union[ArrayLike[float], ArrayLike[Complex], SpatialDataArray]
            Sampled perturbation value(s).
        """

        t_range = self.temperature_range
        temperature_clip = np.clip(temperature, t_range[0], t_range[1])
        data = self.perturbation_values.interp(T=temperature_clip, method=self.interp_method)
        # preserve input type
        if isinstance(temperature, SpatialDataArray):
            return SpatialDataArray(data.drop_vars("T"))
        if np.ndim(temperature) == 0:
            return data.item()
        return data.data

    @cached_property
    def is_complex(self) -> bool:
        """Whether perturbation is complex valued."""
        return np.iscomplexobj(self.perturbation_values)


HeatPerturbationType = Union[LinearHeatPerturbation, CustomHeatPerturbation]


""" Elementary charge perturbation classes """


def ensure_charge_in_range(
    sample: Callable[
        [Union[ArrayLike[float], SpatialDataArray], Union[ArrayLike[float], SpatialDataArray]],
        Union[ArrayLike[float], ArrayLike[Complex], SpatialDataArray],
    ]
) -> Callable[
    [Union[ArrayLike[float], SpatialDataArray], Union[ArrayLike[float], SpatialDataArray]],
    Union[ArrayLike[float], ArrayLike[Complex], SpatialDataArray],
]:
    """Decorate ``sample`` to log warning if charge supplied is out of bounds."""

    @functools.wraps(sample)
    def _sample(
        self,
        electron_density: Union[ArrayLike[float], SpatialDataArray],
        hole_density: Union[ArrayLike[float], SpatialDataArray],
    ) -> Union[ArrayLike[float], ArrayLike[Complex], SpatialDataArray]:
        """New sample function."""

        # disable complex input
        if np.iscomplexobj(electron_density):
            raise ValueError("Cannot pass complex 'electron_density' to 'sample()'")

        if np.iscomplexobj(hole_density):
            raise ValueError("Cannot pass complex 'hole_density' to 'sample()'")

        # check ranges
        e_min, e_max = self.electron_range

        electron_numpy = np.array(electron_density)
        if np.any(electron_numpy < e_min) or np.any(electron_numpy > e_max):
            log.warning(
                "Electron density values passed to 'ChargePerturbation.sample()'"
                f"is outside of 'ChargePerturbation.electron_range' = {self.electron_range}"
            )

        h_min, h_max = self.hole_range

        hole_numpy = np.array(hole_density)
        if np.any(hole_numpy < h_min) or np.any(hole_numpy > h_max):
            log.warning(
                "Hole density values passed to 'ChargePerturbation.sample()'"
                f"is outside of 'ChargePerturbation.hole_range' = {self.hole_range}"
            )

        return sample(self, electron_density, hole_density)

    return _sample


class ChargePerturbation(AbstractPerturbation):
    """Abstract class for charge perturbation."""

    electron_range: Tuple[pd.NonNegativeFloat, pd.NonNegativeFloat] = pd.Field(
        (0, inf),
        title="Electron Density Range",
        description="Range of electrons densities in which perturbation model is valid.",
    )

    hole_range: Tuple[pd.NonNegativeFloat, pd.NonNegativeFloat] = pd.Field(
        (0, inf),
        title="Hole Density Range",
        description="Range of holes densities in which perturbation model is valid.",
    )

    @abstractmethod
    def sample(
        self,
        electron_density: Union[ArrayLike[float], SpatialDataArray],
        hole_density: Union[ArrayLike[float], SpatialDataArray],
    ) -> Union[ArrayLike[float], ArrayLike[Complex], SpatialDataArray]:
        """Sample perturbation.

        Parameters
        ----------
        electron_density : Union[ArrayLike[float], SpatialDataArray]
            Electron density sample point(s).
        hole_density : Union[ArrayLike[float], SpatialDataArray]
            Hole density sample point(s).

        Note
        ----
        Cannot provide a :class:`.SpatialDataArray` for one argument and a regular array
        (``list``, ``tuple``, ``numpy.nd_array``) for the other. Additionally, if both arguments are
        regular arrays they must be one-dimensional arrays.

        Returns
        -------
        Union[ArrayLike[float], ArrayLike[Complex], SpatialDataArray]
            Sampled perturbation value(s).
        """

    @add_ax_if_none
    def plot(
        self,
        electron_density: ArrayLike[float],
        hole_density: ArrayLike[float],
        val: FieldVal = "real",
        ax: Ax = None,
    ) -> Ax:
        """Plot perturbation using provided electron and hole density sample points.

        Parameters
        ----------
        electron_density : Union[ArrayLike[float], SpatialDataArray]
            Array of electron density sample points.
        hole_density : Union[ArrayLike[float], SpatialDataArray]
            Array of hole density sample points.
        val : Literal['real', 'imag', 'abs', 'abs^2', 'phase'] = 'real'
            Which part of the field to plot.
        ax : matplotlib.axes._subplots.Axes = None
            Matplotlib axes to plot on, if not specified, one is created.

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.
        """

        values = self.sample(electron_density, hole_density)
        values = self._get_val(values, val)

        if np.ndim(electron_density) == 0:
            ax.plot(hole_density, values, label=f"electron density = {electron_density} 1/cm^3")
            ax.set_ylabel(f"{val}(perturbation value)")
            ax.set_xlabel("hole density (1/cm^3)")
            ax.set_title(f"charge dependence of {val}(perturbation value)")
            ax.set_aspect("auto")
            ax.legend()

        elif np.ndim(hole_density) == 0:
            ax.plot(electron_density, values, label=f"hole density = {hole_density} 1/cm^3")
            ax.set_ylabel(f"{val}(perturbation value)")
            ax.set_xlabel("electron density (1/cm^3)")
            ax.set_title(f"charge dependence of {val}(perturbation value)")
            ax.set_aspect("auto")
            ax.legend()

        else:
            e_mesh, h_mesh = np.meshgrid(electron_density, hole_density, indexing="ij")
            pc = ax.pcolormesh(e_mesh, h_mesh, values, shading="gouraud")
            plt.colorbar(pc, ax=ax)
            ax.set_xlabel("electron density (1/cm^3)")
            ax.set_ylabel("hole density (1/cm^3)")

        ax.set_title(f"charge dependence of {val}(perturbation value)")
        ax.set_aspect("auto")

        return ax

    @staticmethod
    def _get_eh_types(electron_density, hole_density):
        """Get types of provided arguments and check that no mixing between spatial and regular
        arrays.
        """
        e_type = AbstractPerturbation._array_type(electron_density)
        h_type = AbstractPerturbation._array_type(hole_density)

        one_array = e_type == "array" or h_type == "array"
        one_spatial = e_type == "spatial" or h_type == "spatial"

        if one_array and one_spatial:
            raise ValueError(
                "Cannot mix 'SpatialDataArray' and regular python arrays for 'electron_density'"
                "'hole_density'."
            )

        if e_type == "array" and h_type == "array" and (np.ndim(e_type) > 1 or np.ndim(h_type) > 1):
            raise ValueError(
                "Cannot mix multidimensional arrays for 'electron_density' and 'hole_density'."
            )

        return e_type, h_type


class LinearChargePerturbation(ChargePerturbation):
    """Specifies parameter's perturbation due to free carrier effects as a linear function of
    electron and hole densities:

    Note
    ----
    .. math::

        \\Delta X (T) = \\text{electron\\_coeff} \\times (N_e - \\text{electron\\_ref})
        + \\text{hole\\_coeff} \\times (N_h - \\text{hole\\_ref}),

    where ``electron_coeff`` and ``hole_coeff`` are the parameter's sensitivities to electron and
    hole densities, while ``electron_ref`` and ``hole_ref`` are reference electron and hole density
    values. Ranges of electron and hole densities in which such
    a model is deemed accurate may be provided as fields ``electron_range`` and ``hole_range``
    (default: ``[0, inf]`` each). Wherever is applied, Tidy3D will check that the parameter's value
    does not go out of its physical bounds within ``electron_range`` x ``hole_range`` due to
    perturbations and raise a warning if this check fails. A warning is also issued if
    the perturbation model is evaluated outside of ``electron_range`` x ``hole_range``.

    Example
    -------
    >>> charge_perturb = LinearChargePerturbation(
    ...     electron_ref=0,
    ...     electron_coeff=0.0001,
    ...     electron_range=[0, 1e19],
    ...     hole_ref=0,
    ...     hole_coeff=0.0002,
    ...     hole_range=[0, 2e19],
    ... )
    """

    electron_ref: pd.NonNegativeFloat = pd.Field(
        ...,
        title="Reference Electron Density",
        description="Electron density value at which there is no perturbation due to electrons's "
        "presence.",
        units=PERCMCUBE,
    )

    hole_ref: pd.NonNegativeFloat = pd.Field(
        ...,
        title="Reference Hole Density",
        description="Hole density value at which there is no perturbation due to holes' presence.",
        units=PERCMCUBE,
    )

    electron_coeff: float = pd.Field(
        ...,
        title="Sensitivity to Electron Density",
        description="Sensitivity (derivative) of perturbation with respect to electron density.",
        units=CMCUBE,
    )

    hole_coeff: float = pd.Field(
        ...,
        title="Sensitivity to Hole Density",
        description="Sensitivity (derivative) of perturbation with respect to hole density.",
        units=CMCUBE,
    )

    @cached_property
    def perturbation_range(self) -> Union[Tuple[float, float], Tuple[Complex, Complex]]:
        """Range of possible perturbation values within provided ``electron_range`` and
        ``hole_range``.
        """

        range_from_e = self._linear_range(
            self.electron_range, self.electron_ref, self.electron_coeff
        )
        range_from_h = self._linear_range(self.hole_range, self.hole_ref, self.hole_coeff)

        return tuple(np.array(range_from_e) + np.array(range_from_h))

    @ensure_charge_in_range
    def sample(
        self,
        electron_density: Union[ArrayLike[float], SpatialDataArray],
        hole_density: Union[ArrayLike[float], SpatialDataArray],
    ) -> Union[ArrayLike[float], ArrayLike[Complex], SpatialDataArray]:
        """Sample perturbation at electron and hole density points.

        Parameters
        ----------
        electron_density : Union[ArrayLike[float], SpatialDataArray]
            Electron density sample point(s).
        hole_density : Union[ArrayLike[float], SpatialDataArray]
            Hole density sample point(s).

        Note
        ----
        Cannot provide a :class:`.SpatialDataArray` for one argument and a regular array
        (``list``, ``tuple``, ``numpy.nd_array``) for the other. Additionally, if both arguments are
        regular arrays they must be one-dimensional arrays.

        Returns
        -------
        Union[ArrayLike[float], ArrayLike[Complex], SpatialDataArray]
            Sampled perturbation value(s).
        """
        e_type, h_type = self._get_eh_types(electron_density, hole_density)

        if e_type == "array" and h_type == "array":
            e_mesh, h_mesh = np.meshgrid(electron_density, hole_density, indexing="ij")

            return self.electron_coeff * (e_mesh - self.electron_ref) + self.hole_coeff * (
                h_mesh - self.hole_ref
            )

        e_vals = np.array(electron_density) if e_type == "array" else electron_density
        h_vals = np.array(hole_density) if h_type == "array" else hole_density

        return self.electron_coeff * (e_vals - self.electron_ref) + self.hole_coeff * (
            h_vals - self.hole_ref
        )

    @cached_property
    def is_complex(self) -> bool:
        """Whether perturbation is complex valued."""
        return np.iscomplex(self.electron_coeff) or np.iscomplex(self.hole_coeff)


class CustomChargePerturbation(ChargePerturbation):
    """Specifies parameter's perturbation due to free carrier effects as a custom function of
    electron and hole densities defined as a two-dimensional array of perturbation values at sample
    electron and hole density points. The linear interpolation is used to calculate perturbation
    values between sample points. For electron and hole density values outside of the provided
    sample region the perturbation value is extrapolated as a constant.
    The electron and hole density ranges, ``electron_range`` and ``hole_range``, in which
    the perturbation model is assumed to be accurate is calculated automatically as the minimal and
    maximal density values provided in ``perturbation_values``. Wherever is applied, Tidy3D will
    check that the parameter's value does not go out of its physical bounds within
    ``electron_range`` x ``hole_range`` due to perturbations and raise a warning if this check
    fails. A warning is also issued if the perturbation model is evaluated outside of
    ``electron_range`` x ``hole_range``.

    Example
    -------
    >>> from tidy3d import ChargeDataArray
    >>> perturbation_data = ChargeDataArray(
    ...     [[0.001, 0.002, 0.004], [0.003, 0.002, 0.001]],
    ...     coords=dict(n=[2e15, 2e19], p=[1e16, 1e17, 1e18]),
    ... )
    >>> charge_perturb = CustomChargePerturbation(
    ...     perturbation_values=perturbation_data,
    ... )
    """

    perturbation_values: ChargeDataArray = pd.Field(
        ...,
        title="Petrubation Values",
        description="2D array (vs electron and hole densities) of sampled perturbation values.",
    )

    electron_range: Tuple[pd.NonNegativeFloat, pd.NonNegativeFloat] = pd.Field(
        None,
        title="Electron Density Range",
        description="Range of electrons densities in which perturbation model is valid. For "
        ":class:`.CustomChargePerturbation` this field is computed automatically based on "
        "provided ``perturbation_values``",
    )

    hole_range: Tuple[pd.NonNegativeFloat, pd.NonNegativeFloat] = pd.Field(
        None,
        title="Hole Density Range",
        description="Range of holes densities in which perturbation model is valid. For "
        ":class:`.CustomChargePerturbation` this field is computed automatically based on "
        "provided ``perturbation_values``",
    )

    interp_method: InterpMethod = pd.Field(
        "linear",
        title="Interpolation method",
        description="Interpolation method to obtain perturbation values between sample points.",
    )

    @cached_property
    def perturbation_range(self) -> Union[Tuple[float, float], Tuple[complex, complex]]:
        """Range of possible parameter perturbation values."""
        return np.min(self.perturbation_values).item(), np.max(self.perturbation_values).item()

    @pd.root_validator(skip_on_failure=True)
    def compute_eh_ranges(cls, values):
        """Compute and set electron and hole density ranges based on provided
        ``perturbation_values``.
        """
        if values["electron_range"] is not None:
            log.warning(
                "Electron density range for 'CustomChargePerturbation' is calculated automatically "
                "based on provided 'perturbation_values'. Provided 'electron_range' will be "
                "overwritten."
            )

        if values["hole_range"] is not None:
            log.warning(
                "Hole density range for 'CustomChargePerturbation' is calculated automatically "
                "based on provided 'perturbation_values'. Provided 'hole_range' will be "
                "overwritten."
            )

        perturbation_values = values["perturbation_values"]

        electron_range = (
            np.min(perturbation_values.coords["n"]).item(),
            np.max(perturbation_values.coords["n"]).item(),
        )

        hole_range = (
            np.min(perturbation_values.coords["p"]).item(),
            np.max(perturbation_values.coords["p"]).item(),
        )

        values.update({"electron_range": electron_range, "hole_range": hole_range})

        return values

    @ensure_charge_in_range
    def sample(
        self,
        electron_density: Union[ArrayLike[float], SpatialDataArray],
        hole_density: Union[ArrayLike[float], SpatialDataArray],
    ) -> Union[ArrayLike[float], ArrayLike[Complex], SpatialDataArray]:
        """Sample perturbation at electron and hole density points.

        Parameters
        ----------
        electron_density : Union[ArrayLike[float], SpatialDataArray]
            Electron density sample point(s).
        hole_density : Union[ArrayLike[float], SpatialDataArray]
            Hole density sample point(s).

        Note
        ----
        Cannot provide a :class:`.SpatialDataArray` for one argument and a regular array
        (``list``, ``tuple``, ``numpy.nd_array``) for the other. Additionally, if both arguments are
        regular arrays they must be one-dimensional arrays.

        Returns
        -------
        Union[ArrayLike[float], ArrayLike[Complex], SpatialDataArray]
            Sampled perturbation value(s).
        """
        e_type, h_type = self._get_eh_types(electron_density, hole_density)

        e_clip = np.clip(electron_density, self.electron_range[0], self.electron_range[1])
        h_clip = np.clip(hole_density, self.hole_range[0], self.hole_range[1])

        data = self.perturbation_values.interp(n=e_clip, p=h_clip, method=self.interp_method)

        if e_type == "scalar" and h_type == "scalar":
            return data.item()
        if e_type == "spatial" or h_type == "spatial":
            return SpatialDataArray(data.drop_vars(["n", "p"]))
        return data.data

    @cached_property
    def is_complex(self) -> bool:
        """Whether perturbation is complex valued."""
        return np.iscomplexobj(self.perturbation_values)


ChargePerturbationType = Union[LinearChargePerturbation, CustomChargePerturbation]

PerturbationType = Union[HeatPerturbationType, ChargePerturbationType]


class ParameterPerturbation(Tidy3dBaseModel):
    """Stores information about parameter perturbations due to different physical effect. If both
    heat and charge perturbation models are included their effects are superimposed.

    Example
    -------
    >>> from tidy3d import LinearChargePerturbation, CustomHeatPerturbation, HeatDataArray
    >>>
    >>> perturbation_data = HeatDataArray([0.001, 0.002, 0.004], coords=dict(T=[250, 300, 350]))
    >>> heat_perturb = CustomHeatPerturbation(
    ...     perturbation_values=perturbation_data
    ... )
    >>> charge_perturb = LinearChargePerturbation(
    ...     electron_ref=0,
    ...     electron_coeff=0.0001,
    ...     electron_range=[0, 1e19],
    ...     hole_ref=0,
    ...     hole_coeff=0.0002,
    ...     hole_range=[0, 2e19],
    ... )
    >>> param_perturb = ParameterPerturbation(heat=heat_perturb, charge=charge_perturb)
    """

    heat: HeatPerturbationType = pd.Field(
        None,
        title="Heat Perturbation",
        description="Heat perturbation to apply.",
        discriminator=TYPE_TAG_STR,
    )

    charge: ChargePerturbationType = pd.Field(
        None,
        title="Charge Perturbation",
        description="Charge perturbation to apply.",
        discriminator=TYPE_TAG_STR,
    )

    @cached_property
    def perturbation_list(self) -> List[PerturbationType]:
        """Provided perturbations as a list."""
        perturb_list = []
        for p in [self.heat, self.charge]:
            if p is not None:
                perturb_list.append(p)
        return perturb_list

    @cached_property
    def perturbation_range(self) -> Union[Tuple[float, float], Tuple[Complex, Complex]]:
        """Range of possible parameter perturbation values due to both heat and charge effects."""
        prange = np.zeros(2)

        for p in self.perturbation_list:
            prange = prange + p.perturbation_range

        return tuple(prange)

    @staticmethod
    def _zeros_like(
        T: SpatialDataArray = None,
        n: SpatialDataArray = None,
        p: SpatialDataArray = None,
    ):
        """Check that fields have the same coordinates and return an array field with zeros."""
        template = None
        for field in [T, n, p]:
            if field is not None:
                if template is not None and field.coords != template.coords:
                    raise ValueError(
                        "'temperature', 'electron_density', and 'hole_density' must have the same "
                        "coordinates if provided."
                    )
                template = field

        if template is None:
            raise ValueError(
                "At least one of 'temperature', 'electron_density', or 'hole_density' must be "
                "provided."
            )

        return xr.zeros_like(template)

    def apply_data(
        self,
        temperature: SpatialDataArray = None,
        electron_density: SpatialDataArray = None,
        hole_density: SpatialDataArray = None,
    ) -> SpatialDataArray:
        """Sample perturbations on provided heat and/or charge data. At least one of
        ``temperature``, ``electron_density``, and ``hole_density`` must be not ``None``.
        All provided fields must have identical coords.

        Parameters
        ----------
        temperature : SpatialDataArray = None
            Temperature field data.
        electron_density : SpatialDataArray = None
            Electron density field data.
        hole_density : SpatialDataArray = None
            Hole density field data.

        Returns
        -------
        SpatialDataArray
            Sampled perturbation field.
        """

        result = self._zeros_like(temperature, electron_density, hole_density)

        if temperature is not None and self.heat is not None:
            result = result + self.heat.sample(temperature)

        if (electron_density is not None or hole_density is not None) and self.charge is not None:
            if electron_density is None:
                electron_density = 0

            if hole_density is None:
                hole_density = 0

            result = result + self.charge.sample(electron_density, hole_density)

        return result

    @cached_property
    def is_complex(self) -> bool:
        """Whether perturbation is complex valued."""

        return np.any([p.is_complex for p in self.perturbation_list])
