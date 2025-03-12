"""Defines perturbations to properties of the medium / materials"""

from __future__ import annotations

import functools
from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Tuple, Union

try:
    import matplotlib.pyplot as plt
except ImportError:
    pass
import numpy as np
import pydantic.v1 as pd
import xarray as xr

from ..components.data.validators import validate_no_nans
from ..components.types import TYPE_TAG_STR, ArrayLike, Ax, Complex, FieldVal, InterpMethod
from ..components.viz import add_ax_if_none
from ..constants import C_0, CMCUBE, EPSILON_0, HERTZ, KELVIN, PERCMCUBE, inf
from ..exceptions import DataError
from ..log import log
from .base import Tidy3dBaseModel, cached_property
from .data.data_array import ChargeDataArray, HeatDataArray, IndexedDataArray, SpatialDataArray
from .data.unstructured.base import UnstructuredGridDataset
from .data.utils import (
    CustomSpatialDataType,
    _check_same_coordinates,
    _get_numpy_array,
    _zeros_like,
)

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
        field: Union[ArrayLike[float], ArrayLike[Complex], CustomSpatialDataType], val: FieldVal
    ) -> Union[ArrayLike[float], ArrayLike[Complex], CustomSpatialDataType]:
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


""" Elementary heat perturbation classes """


def ensure_temp_in_range(
    sample: Callable[
        Union[ArrayLike[float], CustomSpatialDataType],
        Union[ArrayLike[float], ArrayLike[Complex], CustomSpatialDataType],
    ],
) -> Callable[
    Union[ArrayLike[float], CustomSpatialDataType],
    Union[ArrayLike[float], ArrayLike[Complex], CustomSpatialDataType],
]:
    """Decorate ``sample`` to log warning if temperature supplied is out of bounds."""

    @functools.wraps(sample)
    def _sample(
        self, temperature: Union[ArrayLike[float], CustomSpatialDataType]
    ) -> Union[ArrayLike[float], ArrayLike[Complex], CustomSpatialDataType]:
        """New sample function."""

        if np.iscomplexobj(temperature):
            raise DataError("Cannot pass complex 'temperature' to 'sample()'")

        temp_min, temp_max = self.temperature_range
        temperature_numpy = _get_numpy_array(temperature)
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
        description="Temperature range in which perturbation model is valid.",
        units=KELVIN,
    )

    @abstractmethod
    def sample(
        self, temperature: Union[ArrayLike[float], CustomSpatialDataType]
    ) -> Union[ArrayLike[float], ArrayLike[Complex], CustomSpatialDataType]:
        """Sample perturbation.

        Parameters
        ----------
        temperature : Union[
                ArrayLike[float],
                :class:`.SpatialDataArray`,
                :class:`.TriangularGridDataset`,
                :class:`.TetrahedralGridDataset`,
            ]
            Temperature sample point(s).

        Returns
        -------
        Union[
            ArrayLike[float],
            ArrayLike[complex],
            :class:`.SpatialDataArray`,
            :class:`.TriangularGridDataset`,
            :class:`.TetrahedralGridDataset`,
        ]
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
    temperature.

    Notes
    -----

        .. math::

            \\Delta X (T) = \\text{coeff} \\times (T - \\text{temperature\\_ref}),

        where ``coeff`` is the parameter's sensitivity (thermo-optic coefficient) to temperature and
        ``temperature_ref`` is the reference temperature point. A temperature range in which such
        a model is deemed accurate may be provided as a field ``temperature_range``
        (default: ``[0, inf]``). Wherever is applied, Tidy3D will check that the parameter's value
        does not go out of its physical bounds within ``temperature_range`` due to perturbations and
        raise a warning if this check fails. A warning is also issued if the perturbation model is
        evaluated outside of ``temperature_range``.

        .. TODO link to relevant example new

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
        self, temperature: Union[ArrayLike[float], CustomSpatialDataType]
    ) -> Union[ArrayLike[float], ArrayLike[Complex], CustomSpatialDataType]:
        """Sample perturbation at temperature points.

        Parameters
        ----------
        temperature : Union[
            ArrayLike[float],
            :class:`.SpatialDataArray`,
            :class:`.TriangularGridDataset`,
            :class:`.TetrahedralGridDataset`,
        ]
            Temperature sample point(s).

        Returns
        -------
        Union[
            ArrayLike[float],
            ArrayLike[complex],
            :class:`.SpatialDataArray`,
            :class:`.TriangularGridDataset`,
            :class:`.TetrahedralGridDataset`,
        ]
            Sampled perturbation value(s).
        """

        temp_vals = temperature
        if isinstance(temperature, (list, tuple)):
            temp_vals = np.array(temperature)

        return self.coeff * (temp_vals - self.temperature_ref)

    @cached_property
    def is_complex(self) -> bool:
        """Whether perturbation is complex valued."""
        return np.iscomplex(self.coeff)


class CustomHeatPerturbation(HeatPerturbation):
    """Specifies parameter's perturbation due to thermal effects as a custom function of
    temperature defined as an array of perturbation values at sample temperature points.

     Notes
     -----

         The linear
        interpolation is used to calculate perturbation values between sample temperature points. For
        temperature values outside of the provided sample region the perturbation value is extrapolated
        as a constant.
        The temperature range, ``temperature_range``, in which the perturbation model is assumed to be
        accurate is calculated automatically as the minimal and maximal sample temperature points.
        Wherever is applied, Tidy3D will check that the parameter's value
        does not go out of its physical bounds within ``temperature_range`` due to perturbations and
        raise a warning if this check fails. A warning is also issued if the perturbation model is
        evaluated outside of ``temperature_range``.

        .. TODO link to relevant example new

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
        description="Temperature range in which perturbation model is valid. For "
        ":class:`.CustomHeatPerturbation` this field is computed automatically based on "
        "temperature sample points provided in ``perturbation_values``.",
        units=KELVIN,
    )

    interp_method: InterpMethod = pd.Field(
        "linear",
        title="Interpolation method",
        description="Interpolation method to obtain perturbation values between sample points.",
    )

    _no_nans = validate_no_nans("perturbation_values")

    @cached_property
    def perturbation_range(self) -> Union[Tuple[float, float], Tuple[Complex, Complex]]:
        """Range of possible parameter perturbation values."""
        return np.min(self.perturbation_values).item(), np.max(self.perturbation_values).item()

    @pd.root_validator(skip_on_failure=True)
    def compute_temperature_range(cls, values):
        """Compute and set temperature range based on provided ``perturbation_values``."""

        perturbation_values = values["perturbation_values"]

        # .item() to convert to a scalar
        temperature_range = (
            np.min(perturbation_values.coords["T"]).item(),
            np.max(perturbation_values.coords["T"]).item(),
        )

        if (
            values["temperature_range"] is not None
            and values["temperature_range"] != temperature_range
        ):
            log.warning(
                "Temperature range for 'CustomHeatPerturbation' is calculated automatically "
                "based on provided 'perturbation_values'. Provided 'temperature_range' will be "
                "overwritten."
            )

        values.update({"temperature_range": temperature_range})

        return values

    @ensure_temp_in_range
    def sample(
        self, temperature: Union[ArrayLike[float], CustomSpatialDataType]
    ) -> Union[ArrayLike[float], ArrayLike[Complex], CustomSpatialDataType]:
        """Sample perturbation at provided temperature points.

        Parameters
        ----------
        temperature : Union[
                ArrayLike[float],
                :class:`.SpatialDataArray`,
                :class:`.TriangularGridDataset`,
                :class:`.TetrahedralGridDataset`,
            ]
            Temperature sample point(s).

        Returns
        -------
        Union[
            ArrayLike[float],
            ArrayLike[complex],
            :class:`.SpatialDataArray`,
            :class:`.TriangularGridDataset`,
            :class:`.TetrahedralGridDataset`,
        ]
            Sampled perturbation value(s).
        """

        t_range = self.temperature_range
        temp_clip = np.clip(_get_numpy_array(temperature), t_range[0], t_range[1])
        sampled = self.perturbation_values.interp(
            T=temp_clip.ravel(), method=self.interp_method
        ).values
        sampled = np.reshape(sampled, np.shape(temp_clip))

        # preserve input type
        if isinstance(temperature, SpatialDataArray):
            return SpatialDataArray(sampled, coords=temperature.coords)
        if isinstance(temperature, UnstructuredGridDataset):
            return temperature.updated_copy(
                values=IndexedDataArray(sampled, coords=temperature.values.coords)
            )
        if np.ndim(temperature) == 0:
            return sampled.item()
        return sampled

    @cached_property
    def is_complex(self) -> bool:
        """Whether perturbation is complex valued."""
        return np.iscomplexobj(self.perturbation_values)


HeatPerturbationType = Union[LinearHeatPerturbation, CustomHeatPerturbation]


""" Elementary charge perturbation classes """


def ensure_charge_in_range(
    sample: Callable[
        [
            Union[ArrayLike[float], CustomSpatialDataType],
            Union[ArrayLike[float], CustomSpatialDataType],
        ],
        Union[ArrayLike[float], ArrayLike[Complex], CustomSpatialDataType],
    ],
) -> Callable[
    [
        Union[ArrayLike[float], CustomSpatialDataType],
        Union[ArrayLike[float], CustomSpatialDataType],
    ],
    Union[ArrayLike[float], ArrayLike[Complex], CustomSpatialDataType],
]:
    """Decorate ``sample`` to log warning if charge supplied is out of bounds."""

    @functools.wraps(sample)
    def _sample(
        self,
        electron_density: Union[ArrayLike[float], CustomSpatialDataType],
        hole_density: Union[ArrayLike[float], CustomSpatialDataType],
    ) -> Union[ArrayLike[float], ArrayLike[Complex], CustomSpatialDataType]:
        """New sample function."""

        # disable complex input
        if np.iscomplexobj(electron_density):
            raise DataError("Cannot pass complex 'electron_density' to 'sample()'")

        if np.iscomplexobj(hole_density):
            raise DataError("Cannot pass complex 'hole_density' to 'sample()'")

        # check ranges
        e_min, e_max = self.electron_range

        electron_numpy = _get_numpy_array(electron_density)
        if np.any(electron_numpy < e_min) or np.any(electron_numpy > e_max):
            log.warning(
                "Electron density values passed to 'ChargePerturbation.sample()'"
                f"is outside of 'ChargePerturbation.electron_range' = {self.electron_range}"
            )

        h_min, h_max = self.hole_range

        hole_numpy = _get_numpy_array(hole_density)
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
        electron_density: Union[ArrayLike[float], CustomSpatialDataType],
        hole_density: Union[ArrayLike[float], CustomSpatialDataType],
    ) -> Union[ArrayLike[float], ArrayLike[Complex], CustomSpatialDataType]:
        """Sample perturbation.

        Parameters
        ----------
        electron_density : Union[
                ArrayLike[float],
                :class:`.SpatialDataArray`,
                :class:`.TriangularGridDataset`,
                :class:`.TetrahedralGridDataset`,
            ]
            Electron density sample point(s).
        hole_density : Union[
                ArrayLike[float],
                :class:`.SpatialDataArray`,
                :class:`.TriangularGridDataset`,
                :class:`.TetrahedralGridDataset`,
            ]
            Hole density sample point(s).

        Note
        ----
        Provided ``electron_density`` and ``hole_density`` must be of the same type and match
        shapes/coordinates, unless one of them is a scalar.

        Returns
        -------
        Union[
            ArrayLike[float],
            ArrayLike[complex],
            :class:`.SpatialDataArray`,
            :class:`.TriangularGridDataset`,
            :class:`.TetrahedralGridDataset`,
        ]
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
        electron_density : Union[ArrayLike[float], CustomSpatialDataType]
            Array of electron density sample points.
        hole_density : Union[ArrayLike[float], CustomSpatialDataType]
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


class LinearChargePerturbation(ChargePerturbation):
    """Specifies parameter's perturbation due to free carrier effects as a linear function of
    electron and hole densities:

    Notes
    -----

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

        .. TODO add example here and links

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
        electron_density: Union[ArrayLike[float], CustomSpatialDataType],
        hole_density: Union[ArrayLike[float], CustomSpatialDataType],
    ) -> Union[ArrayLike[float], ArrayLike[Complex], CustomSpatialDataType]:
        """Sample perturbation at electron and hole density points.

        Parameters
        ----------
        electron_density : Union[
                ArrayLike[float],
                :class:`.SpatialDataArray`,
                :class:`.TriangularGridDataset`,
                :class:`.TetrahedralGridDataset`,
            ]
            Electron density sample point(s).
        hole_density : Union[
                ArrayLike[float],
                :class:`.SpatialDataArray`,
                :class:`.TriangularGridDataset`,
                :class:`.TetrahedralGridDataset`,
            ]
            Hole density sample point(s).

        Note
        ----
        Provided ``electron_density`` and ``hole_density`` must be of the same type and match
        shapes/coordinates, unless one of them is a scalar or both are 1d arrays, in which case
        values are broadcasted.

        Returns
        -------
        Union[
            ArrayLike[float],
            ArrayLike[complex],
            :class:`.SpatialDataArray`,
            :class:`.TriangularGridDataset`,
            :class:`.TetrahedralGridDataset`,
        ]
            Sampled perturbation value(s).
        """
        inputs = [electron_density, hole_density]

        no_scalars = all(np.ndim(_get_numpy_array(arr)) > 0 for arr in inputs)
        both_1d = all(
            isinstance(arr, (list, tuple, np.ndarray)) and np.ndim(arr) == 1 for arr in inputs
        )

        # we allow combining a scalar with any other type
        # or 2 1d arrays (broadcasting)
        # otherwise we require match in shape/coords
        if (
            no_scalars
            and not both_1d
            and not _check_same_coordinates(electron_density, hole_density)
        ):
            raise DataError(
                "Provided electron and hole density data must be of the same type and shape."
            )

        e_vals = electron_density
        h_vals = hole_density

        # convert python arrays into numpy
        if isinstance(electron_density, (list, tuple)):
            e_vals = np.array(electron_density)

        if isinstance(hole_density, (list, tuple)):
            h_vals = np.array(hole_density)

        # broadcast if both are 1d arrays
        if both_1d:
            e_vals, h_vals = np.meshgrid(e_vals, h_vals, indexing="ij")

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
    electron and hole density points.

    Notes
    -----

        The linear interpolation is used to calculate perturbation
        values between sample points. For electron and hole density values outside of the provided
        sample region the perturbation value is extrapolated as a constant.
        The electron and hole density ranges, ``electron_range`` and ``hole_range``, in which
        the perturbation model is assumed to be accurate is calculated automatically as the minimal and
        maximal density values provided in ``perturbation_values``. Wherever is applied, Tidy3D will
        check that the parameter's value does not go out of its physical bounds within
        ``electron_range`` x ``hole_range`` due to perturbations and raise a warning if this check
        fails. A warning is also issued if the perturbation model is evaluated outside of
        ``electron_range`` x ``hole_range``.

        .. TODO add example here and links

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

    _no_nans = validate_no_nans("perturbation_values")

    @cached_property
    def perturbation_range(self) -> Union[Tuple[float, float], Tuple[complex, complex]]:
        """Range of possible parameter perturbation values."""
        return np.min(self.perturbation_values).item(), np.max(self.perturbation_values).item()

    @pd.root_validator(skip_on_failure=True)
    def compute_eh_ranges(cls, values):
        """Compute and set electron and hole density ranges based on provided
        ``perturbation_values``.
        """

        perturbation_values = values["perturbation_values"]

        electron_range = (
            np.min(perturbation_values.coords["n"]).item(),
            np.max(perturbation_values.coords["n"]).item(),
        )

        hole_range = (
            np.min(perturbation_values.coords["p"]).item(),
            np.max(perturbation_values.coords["p"]).item(),
        )

        if values["electron_range"] is not None and electron_range != values["electron_range"]:
            log.warning(
                "Electron density range for 'CustomChargePerturbation' is calculated automatically "
                "based on provided 'perturbation_values'. Provided 'electron_range' will be "
                "overwritten."
            )

        if values["hole_range"] is not None and hole_range != values["hole_range"]:
            log.warning(
                "Hole density range for 'CustomChargePerturbation' is calculated automatically "
                "based on provided 'perturbation_values'. Provided 'hole_range' will be "
                "overwritten."
            )

        values.update({"electron_range": electron_range, "hole_range": hole_range})

        return values

    @ensure_charge_in_range
    def sample(
        self,
        electron_density: Union[ArrayLike[float], CustomSpatialDataType],
        hole_density: Union[ArrayLike[float], CustomSpatialDataType],
    ) -> Union[ArrayLike[float], ArrayLike[Complex], CustomSpatialDataType]:
        """Sample perturbation at electron and hole density points.

        Parameters
        ----------
        electron_density : Union[
                ArrayLike[float],
                :class:`.SpatialDataArray`,
                :class:`.TriangularGridDataset`,
                :class:`.TetrahedralGridDataset`,
            ]
            Electron density sample point(s).
        hole_density : Union[
                ArrayLike[float],
                :class:`.SpatialDataArray`,
                :class:`.TriangularGridDataset`,
                :class:`.TetrahedralGridDataset`,
            ]
            Hole density sample point(s).

        Note
        ----
        Provided ``electron_density`` and ``hole_density`` must be of the same type and match
        shapes/coordinates, unless one of them is a scalar or both are 1d arrays, in which case
        values are broadcasted.

        Returns
        -------
        Union[
            ArrayLike[float],
            ArrayLike[complex],
            :class:`.SpatialDataArray`,
            :class:`.TriangularGridDataset`,
            :class:`.TetrahedralGridDataset`,
        ]
            Sampled perturbation value(s).
        """
        inputs = [electron_density, hole_density]

        no_scalars = all(np.ndim(_get_numpy_array(arr)) > 0 for arr in inputs)
        both_1d = all(
            isinstance(arr, (list, tuple, np.ndarray)) and np.ndim(_get_numpy_array(arr)) == 1
            for arr in inputs
        )

        # we allow combining a scalar with any other type
        # or 2 1d arrays (broadcasting)
        # otherwise we require match in shape/coords
        if (
            no_scalars
            and not both_1d
            and not _check_same_coordinates(electron_density, hole_density)
        ):
            raise DataError(
                "Provided electron and hole density data must be of the same type and shape."
            )

        # clip to allowed values
        # (this also implicitly convert python arrays into numpy
        e_vals = np.core.umath.clip(
            electron_density, self.electron_range[0], self.electron_range[1]
        )
        h_vals = np.core.umath.clip(hole_density, self.hole_range[0], self.hole_range[1])

        # we cannot pass UnstructuredGridDataset directly into xarray interp
        # thus we need to explicitly grad the underlying xarray
        if isinstance(e_vals, UnstructuredGridDataset):
            e_vals = e_vals.values
        if isinstance(h_vals, UnstructuredGridDataset):
            h_vals = h_vals.values

        # note that the dimensionality of this operation differs depending on whether xarrays
        # or simple unlabeled arrays are provided:
        # - for unlabeled arrays, values are broadcasted
        # - for xarrays, values are considered pairwise based on xarrays' coords
        sampled = self.perturbation_values.interp(n=e_vals, p=h_vals, method=self.interp_method)

        # grab the result without any labels
        sampled = sampled.values

        # preserve input type
        for arr in inputs:
            if isinstance(arr, SpatialDataArray):
                return SpatialDataArray(sampled, coords=arr.coords)

            if isinstance(arr, UnstructuredGridDataset):
                return arr.updated_copy(values=IndexedDataArray(sampled, coords=arr.values.coords))

        if all(np.ndim(_get_numpy_array(arr)) == 0 for arr in inputs):
            return sampled.item()

        return sampled

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

    @pd.root_validator(skip_on_failure=True)
    def _check_not_empty(cls, values):
        """Check that perturbation model is not empty."""

        heat = values.get("heat")
        charge = values.get("charge")

        if heat is None and charge is None:
            raise DataError(
                "Perturbation models 'heat' and 'charge' in 'ParameterPerturbation' cannot be "
                "simultaneously 'None'."
            )

        return values

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
        T: CustomSpatialDataType = None,
        n: CustomSpatialDataType = None,
        p: CustomSpatialDataType = None,
    ):
        """Check that fields have the same coordinates and return an array field with zeros."""
        template = None
        for field in [T, n, p]:
            if field is not None:
                if template is not None and not _check_same_coordinates(field, template):
                    raise DataError(
                        "'temperature', 'electron_density', and 'hole_density' must have the same "
                        "coordinates if provided."
                    )
                template = field

        if template is None:
            raise DataError(
                "At least one of 'temperature', 'electron_density', or 'hole_density' must be "
                "provided."
            )

        return _zeros_like(template)

    def apply_data(
        self,
        temperature: CustomSpatialDataType = None,
        electron_density: CustomSpatialDataType = None,
        hole_density: CustomSpatialDataType = None,
    ) -> CustomSpatialDataType:
        """Sample perturbations on provided heat and/or charge data. At least one of
        ``temperature``, ``electron_density``, and ``hole_density`` must be not ``None``.
        All provided fields must have identical coords.

        Parameters
        ----------
        temperature : Union[
                :class:`.SpatialDataArray`,
                :class:`.TriangularGridDataset`,
                :class:`.TetrahedralGridDataset`,
            ] = None
            Temperature field data.
        electron_density : Union[
                :class:`.SpatialDataArray`,
                :class:`.TriangularGridDataset`,
                :class:`.TetrahedralGridDataset`,
            ] = None
            Electron density field data.
        hole_density : Union[
                :class:`.SpatialDataArray`,
                :class:`.TriangularGridDataset`,
                :class:`.TetrahedralGridDataset`,
            ] = None
            Hole density field data.

        Returns
        -------
        Union[
            :class:`.SpatialDataArray`,
            :class:`.TriangularGridDataset`,
            :class:`.TetrahedralGridDataset`,
        ] = None
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


class PermittivityPerturbation(Tidy3dBaseModel):
    """A general medium perturbation model which is defined through perturbation to
    permittivity and conductivity.

    Example
    -------
    >>> from tidy3d import LinearChargePerturbation, LinearHeatPerturbation, PermittivityPerturbation, C_0
    >>>
    >>> heat_perturb = LinearHeatPerturbation(
    ...     temperature_ref=300,
    ...     coeff=0.001,
    ... )
    >>> charge_perturb = LinearChargePerturbation(
    ...     electron_ref=0,
    ...     electron_coeff=0.0001,
    ...     hole_ref=0,
    ...     hole_coeff=0.0002,
    ... )
    >>> delta_eps = ParameterPerturbation(heat=heat_perturb)
    >>> delta_sigma = ParameterPerturbation(charge=charge_perturb)
    >>> permittivity_pb = PermittivityPerturbation(delta_eps=delta_eps, delta_sigma=delta_sigma)
    """

    delta_eps: Optional[ParameterPerturbation] = pd.Field(
        None,
        title="Permittivity Perturbation",
        description="Perturbation model for permittivity.",
    )

    delta_sigma: Optional[ParameterPerturbation] = pd.Field(
        None,
        title="Conductivity Perturbation",
        description="Perturbation model for conductivity.",
    )

    @pd.root_validator(skip_on_failure=True)
    def _check_not_complex(cls, values):
        """Check that perturbation values are not complex."""

        delta_eps = values.get("delta_eps")
        delta_sigma = values.get("delta_sigma")

        delta_eps_complex = False if delta_eps is None else delta_eps.is_complex
        delta_sigma_complex = False if delta_sigma is None else delta_sigma.is_complex

        if delta_eps_complex or delta_sigma_complex:
            raise DataError(
                "Perturbation models 'delta_eps' and 'delta_sigma' in 'PermittivityPerturbation' cannot be "
                "complex-valued."
            )

        return values

    @pd.root_validator(skip_on_failure=True)
    def _check_not_empty(cls, values):
        """Check that perturbation model is not empty."""

        delta_eps = values.get("delta_eps")
        delta_sigma = values.get("delta_sigma")

        if delta_eps is None and delta_sigma is None:
            raise DataError(
                "Perturbation models 'delta_eps' and 'delta_sigma' in 'PermittivityPerturbation' cannot be "
                "simultaneously 'None'."
            )

        return values

    def _delta_eps_delta_sigma_ranges(self):
        """Perturbation range of permittivity."""

        delta_eps_range = (0, 0) if self.delta_eps is None else self.delta_eps.perturbation_range
        delta_sigma_range = (
            (0, 0) if self.delta_sigma is None else self.delta_sigma.perturbation_range
        )
        return delta_eps_range, delta_sigma_range

    def _sample_delta_eps_delta_sigma(
        self,
        temperature: CustomSpatialDataType = None,
        electron_density: CustomSpatialDataType = None,
        hole_density: CustomSpatialDataType = None,
    ) -> CustomSpatialDataType:
        """Compute effictive pertubation to eps and sigma."""

        delta_eps_sampled = None
        if self.delta_eps is not None:
            delta_eps_sampled = self.delta_eps.apply_data(
                temperature, electron_density, hole_density
            )

        delta_sigma_sampled = None
        if self.delta_sigma is not None:
            delta_sigma_sampled = self.delta_sigma.apply_data(
                temperature, electron_density, hole_density
            )

        return delta_eps_sampled, delta_sigma_sampled


class AbstractDeltaModel(Tidy3dBaseModel):
    """Abstract class for the definition of delta models"""

    @abstractmethod
    def delta_k(self) -> ChargePerturbationType:
        """Return the perturbation range of the model."""

    @abstractmethod
    def delta_n(self) -> HeatPerturbationType:
        """Return the perturbation range of the model."""


class NedeljkovicSorefMashanovich(AbstractDeltaModel):
    """Nedeljkovic-Soref-Mashanovich model for the perturbation of the refractive index and
    extinction coefficient due to free carriers.

    M. Nedeljkovic, R. Soref and G. Z. Mashanovich, "Free-Carrier Electrorefraction and Electroabsorption
    Modulation Predictions for Silicon Over the 1–14- μm Infrared Wavelength Range," in IEEE Photonics
    Journal, vol. 3, no. 6, pp. 1171-1180, Dec. 2011, doi: 10.1109/JPHOT.2011.2171930

    Example
    -------
    """

    perturb_coeffs = xr.Dataset(
        {
            "a": (
                "wvl",
                [
                    3.48e-22,
                    8.88e-21,
                    3.22e-20,
                    1.67e-20,
                    6.29e-21,
                    3.10e-21,
                    7.45e-22,
                    2.16e-22,
                    9.28e-23,
                    4.58e-23,
                    3.26e-23,
                    2.70e-23,
                    2.25e-23,
                    1.36e-23,
                    1.85e-23,
                    3.05e-23,
                    4.08e-23,
                    4.14e-23,
                    3.81e-23,
                    4.23e-23,
                    5.81e-23,
                    8.20e-23,
                    1.13e-22,
                    1.22e-22,
                    1.09e-22,
                    1.20e-22,
                    1.62e-22,
                ],
            ),
            "b": (
                "wvl",
                [
                    1.229,
                    1.167,
                    1.149,
                    1.169,
                    1.193,
                    1.210,
                    1.245,
                    1.277,
                    1.299,
                    1.319,
                    1.330,
                    1.338,
                    1.345,
                    1.359,
                    1.354,
                    1.345,
                    1.340,
                    1.341,
                    1.344,
                    1.344,
                    1.338,
                    1.331,
                    1.325,
                    1.324,
                    1.328,
                    1.327,
                    1.321,
                ],
            ),
            "c": (
                "wvl",
                [
                    1.02e-19,
                    5.84e-20,
                    6.21e-20,
                    8.08e-20,
                    3.40e-20,
                    6.05e-20,
                    5.43e-20,
                    5.58e-20,
                    6.65e-20,
                    8.53e-20,
                    1.53e-19,
                    1.22e-19,
                    1.29e-19,
                    9.99e-20,
                    1.32e-19,
                    1.57e-18,
                    1.45e-18,
                    1.70e-18,
                    1.25e-18,
                    8.14e-19,
                    1.55e-18,
                    4.81e-18,
                    4.72e-18,
                    2.09e-18,
                    1.16e-18,
                    2.01e-18,
                    7.52e-18,
                ],
            ),
            "d": (
                "wvl",
                [
                    1.089,
                    1.109,
                    1.119,
                    1.123,
                    1.151,
                    1.145,
                    1.153,
                    1.158,
                    1.160,
                    1.159,
                    1.149,
                    1.158,
                    1.160,
                    1.170,
                    1.167,
                    1.111,
                    1.115,
                    1.115,
                    1.125,
                    1.137,
                    1.124,
                    1.100,
                    1.102,
                    1.124,
                    1.140,
                    1.130,
                    1.101,
                ],
            ),
            "p": (
                "wvl",
                [
                    2.98e-22,
                    5.40e-22,
                    1.91e-21,
                    5.70e-21,
                    6.57e-21,
                    6.95e-21,
                    7.25e-21,
                    1.19e-20,
                    2.46e-20,
                    3.64e-20,
                    4.96e-20,
                    5.91e-20,
                    5.52e-20,
                    3.19e-20,
                    3.56e-20,
                    8.65e-20,
                    2.09e-19,
                    2.07e-19,
                    3.01e-19,
                    5.07e-19,
                    1.51e-19,
                    2.19e-19,
                    3.04e-19,
                    4.44e-19,
                    6.96e-19,
                    1.05e-18,
                    1.45e-18,
                ],
            ),
            "q": (
                "wvl",
                [
                    1.016,
                    1.011,
                    0.992,
                    0.976,
                    0.981,
                    0.986,
                    0.991,
                    0.985,
                    0.973,
                    0.968,
                    0.965,
                    0.964,
                    0.969,
                    0.984,
                    0.984,
                    0.966,
                    0.948,
                    0.951,
                    0.944,
                    0.934,
                    0.965,
                    0.958,
                    0.953,
                    0.945,
                    0.936,
                    0.928,
                    0.922,
                ],
            ),
            "r": (
                "wvl",
                [
                    1.25e-18,
                    1.53e-18,
                    2.28e-18,
                    5.19e-18,
                    3.62e-18,
                    9.28e-18,
                    9.99e-18,
                    1.29e-17,
                    2.03e-17,
                    3.31e-17,
                    6.92e-17,
                    8.23e-17,
                    1.15e-16,
                    4.81e-16,
                    7.44e-16,
                    7.11e-16,
                    5.29e-16,
                    9.72e-16,
                    1.22e-15,
                    1.16e-15,
                    3.16e-15,
                    1.51e-14,
                    2.71e-14,
                    2.65e-14,
                    2.94e-14,
                    6.85e-14,
                    2.60e-13,
                ],
            ),
            "s": (
                "wvl",
                [
                    0.835,
                    0.838,
                    0.841,
                    0.832,
                    0.849,
                    0.834,
                    0.839,
                    0.838,
                    0.833,
                    0.826,
                    0.812,
                    0.812,
                    0.807,
                    0.776,
                    0.769,
                    0.774,
                    0.783,
                    0.772,
                    0.769,
                    0.772,
                    0.750,
                    0.716,
                    0.704,
                    0.706,
                    0.705,
                    0.686,
                    0.656,
                ],
            ),
        },
        coords={"wvl": np.array([1.3, 1.55] + list(np.arange(2, 14.5, 0.5)))},
    )

    ref_freq: pd.NonNegativeFloat = pd.Field(
        ...,
        title="Reference Frequency",
        description="Reference frequency to evaluate perturbation at (Hz).",
        units=HERTZ,
    )

    electrons_grid: np.ndarray = pd.Field(
        np.concatenate(([0], np.logspace(-6, 22, num=200))),
        title="Electron concentration grid.",
        descriptio="The model will be evaluated at these concentration values. Since "
        "the data at these locations will later be interpolated to determine perturbations "
        "one should provide representative values. Usually, it is convenient to provide "
        "evenly spaced values in logarithmic scale to cover the whole range of concentrations, "
        "i.e., `np.concatenate(([0], np.logspace(-6, 22, num=200)))`.",
    )

    holes_grid: np.ndarray = pd.Field(
        np.concatenate(([0], np.logspace(-6, 22, num=200))),
        title="Hole concentration grid.",
        descriptio="The model will be evaluated at these concentration values. Since "
        "the data at these locations will later be interpolated to determine perturbations "
        "one should provide representative values. Usually, it is convenient to provide "
        "evenly spaced values in logarithmic scale to cover the whole range of concentrations, "
        "i.e., `np.concatenate(([0], np.logspace(-6, 22, num=200)))`.",
    )

    @pd.root_validator(skip_on_failure=True)
    def _check_freq_in_range(cls, values):
        """Check that the given frequency is within validity range.
        If not, issue a warning.
        """

        freq = values.get("ref_freq")
        wavelengths = list(values.get("perturb_coeffs").coords["wvl"])

        freq_range = (C_0 / np.max(wavelengths), C_0 / np.min(wavelengths))

        if freq < freq_range[0] or freq > freq_range[1]:
            log.warning(
                f"Given frequency {freq} Hz is outside of the validity range ({freq_range[0]} - "
                f"{freq_range[1]} Hz) of the Nedeljkovic-Soref-Mashanovich model."
            )

        return values

    @cached_property
    def ref_wavelength(self) -> float:
        """Reference wavelength (um) to evaluate perturbation at."""
        return C_0 / self.ref_freq

    @cached_property
    def _coeffs_at_ref_freq(self) -> xr.Dataset:
        """Coefficients at reference frequency."""
        return self.perturb_coeffs.interp(wvl=self.ref_wavelength)

    def delta_k(self) -> ChargePerturbationType:
        """Return the perturbation range of the model."""

        # create free-carrier mesh
        # NOTE: this range (0, 1e22) should work in most cases. If need be we can add a new field
        # to the model to allow for custom ranges.
        Ne_range = self.electrons_grid
        Nh_range = self.holes_grid

        Ne_mesh, Nh_mesh = np.meshgrid(Ne_range, Nh_range, indexing="ij")

        # get parameters a, b, c, d
        ke_coeff = self._coeffs_at_ref_freq["a"].item()
        ke_pow = self._coeffs_at_ref_freq["b"].item()

        kh_coeff = self._coeffs_at_ref_freq["c"].item()
        kh_pow = self._coeffs_at_ref_freq["d"].item()

        dk_mesh = ke_coeff * Ne_mesh**ke_pow + kh_coeff * Nh_mesh**kh_pow

        # NOTE: since the formula gives us the absorption coefficient, we need to multiply by a factor
        # to get the extinction coefficient (k)
        # additionally, this dk is in cm^-1, so we need toconvert to um^-1
        k_factor = self.ref_wavelength * 1e-4 / 4 / np.pi
        dk_mesh = dk_mesh * k_factor

        # convert t ChargeDataArray
        dk_data = ChargeDataArray(dk_mesh, coords=dict(n=Ne_range, p=Nh_range))

        # create CustomChargePerturbation
        k_si_charge = CustomChargePerturbation(perturbation_values=dk_data)

        return ParameterPerturbation(charge=k_si_charge)

    def delta_n(self) -> ChargePerturbationType:
        """Return the perturbation range of the model."""

        # create free-carrier mesh
        # NOTE: this range (0, 1e22) should work in most cases. If need be we can add a new field
        # to the model to allow for custom ranges.
        Ne_range = self.electrons_grid
        Nh_range = self.holes_grid

        Ne_mesh, Nh_mesh = np.meshgrid(Ne_range, Nh_range, indexing="ij")

        # get parameters p, q, r, s
        ne_coeff = self._coeffs_at_ref_freq["p"].item()
        ne_pow = self._coeffs_at_ref_freq["q"].item()

        nh_coeff = self._coeffs_at_ref_freq["r"].item()
        nh_pow = self._coeffs_at_ref_freq["s"].item()

        dn_mesh = -ne_coeff * Ne_mesh**ne_pow - nh_coeff * Nh_mesh**nh_pow

        # create ChargeDataArray
        dn_data = ChargeDataArray(dn_mesh, coords=dict(n=Ne_range, p=Nh_range))

        # create CustomChargePerturbation
        n_si_charge = CustomChargePerturbation(perturbation_values=dn_data)

        return ParameterPerturbation(charge=n_si_charge)


class IndexPerturbation(Tidy3dBaseModel):
    """A general medium perturbation model which is defined through perturbation to
    refractive index, n and k.

    Example
    -------
    >>> from tidy3d import LinearChargePerturbation, LinearHeatPerturbation, IndexPerturbation, C_0
    >>>
    >>> heat_perturb = LinearHeatPerturbation(
    ...     temperature_ref=300,
    ...     coeff=0.001,
    ... )
    >>> charge_perturb = LinearChargePerturbation(
    ...     electron_ref=0,
    ...     electron_coeff=0.0001,
    ...     hole_ref=0,
    ...     hole_coeff=0.0002,
    ... )
    >>> dn_pb = ParameterPerturbation(heat=heat_perturb)
    >>> dk_pb = ParameterPerturbation(charge=charge_perturb)
    >>> index_pb = IndexPerturbation(delta_n=dn_pb, delta_k=dk_pb, freq=C_0)
    """

    delta_n: Optional[ParameterPerturbation] = pd.Field(
        None,
        title="Refractive Index Perturbation",
        description="Perturbation of the real part of refractive index.",
    )

    delta_k: Optional[ParameterPerturbation] = pd.Field(
        None,
        title="Exctinction Coefficient Perturbation",
        description="Perturbation of the imaginary part of refractive index.",
    )

    freq: pd.NonNegativeFloat = pd.Field(
        ...,
        title="Frequency",
        description="Frequency to evaluate permittivity at (Hz).",
        units=HERTZ,
    )

    @pd.root_validator(skip_on_failure=True)
    def _check_not_complex(cls, values):
        """Check that perturbation values are not complex."""

        dn = values.get("delta_n")
        dk = values.get("delta_k")

        dn_complex = False if dn is None else dn.is_complex
        dk_complex = False if dk is None else dk.is_complex

        if dn_complex or dk_complex:
            raise DataError(
                "Perturbation models 'dn' and 'dk' in 'IndexPerturbation' cannot be "
                "complex-valued."
            )

        return values

    @pd.root_validator(skip_on_failure=True)
    def _check_not_empty(cls, values):
        """Check that perturbation model is not empty."""

        dn = values.get("delta_n")
        dk = values.get("delta_k")

        if dn is None and dk is None:
            raise DataError(
                "Perturbation models 'dn' and 'dk' in 'IndexPerturbation' cannot be "
                "simultaneously 'None'."
            )

        return values

    def _delta_eps_delta_sigma_ranges(self, n: float, k: float):
        """Perturbation range of permittivity."""
        omega0 = 2 * np.pi * self.freq

        dn_range = [0] if self.delta_n is None else self.delta_n.perturbation_range
        dk_range = [0] if self.delta_k is None else self.delta_k.perturbation_range

        dn_grid, dk_grid = np.meshgrid(dn_range, dk_range)

        # deal with possible 0 * inf
        dk_dn = np.zeros_like(dn_grid)
        inds = np.logical_and(dn_grid != 0, dk_grid != 0)
        dk_dn[inds] = dn_grid[inds] * dk_grid[inds]
        k_dn = 0 if k == 0 else k * dn_grid

        # ignore potential inf - inf
        with np.errstate(invalid="ignore"):
            delta_eps = (2 * n + dn_grid) * dn_grid - (2 * n + dk_grid) * dk_grid
            delta_sigma = 2 * omega0 * (k_dn + n * dk_grid + dk_dn) * EPSILON_0

        if np.any(np.isnan(delta_eps)):
            delta_eps_range = (-inf, inf)
        else:
            delta_eps_range = (np.min(delta_eps), np.max(delta_eps))

        if np.any(np.isnan(delta_sigma)):
            delta_sigma_range = (-inf, inf)
        else:
            delta_sigma_range = (np.min(delta_sigma), np.max(delta_sigma))

        return delta_eps_range, delta_sigma_range

    def _sample_delta_eps_delta_sigma(
        self,
        n: float,
        k: float,
        temperature: CustomSpatialDataType = None,
        electron_density: CustomSpatialDataType = None,
        hole_density: CustomSpatialDataType = None,
    ) -> CustomSpatialDataType:
        """Compute effictive pertubation to eps and sigma."""

        # delta_eps = 2 * n * dn + dn ** 2 - 2 * k * dk - dk ** 2
        # delta_sigma = 2 * omega * (k * dn + n * dk + dk * dn)
        dn_sampled = (
            None
            if self.delta_n is None
            else self.delta_n.apply_data(temperature, electron_density, hole_density)
        )
        dk_sampled = (
            None
            if self.delta_k is None
            else self.delta_k.apply_data(temperature, electron_density, hole_density)
        )

        omega0 = 2 * np.pi * self.freq

        delta_eps = None
        delta_sigma = None
        if dn_sampled is not None:
            delta_eps = 2 * n * dn_sampled + dn_sampled**2
            if k != 0:
                delta_sigma = 2 * omega0 * k * dn_sampled

        if dk_sampled is not None:
            delta_eps = 0 if delta_eps is None else delta_eps
            delta_eps = delta_eps - 2 * k * dk_sampled - dk_sampled**2

            delta_sigma = 0 if delta_sigma is None else delta_sigma
            delta_sigma = delta_sigma + 2 * omega0 * n * dk_sampled

            if dn_sampled is not None:
                delta_sigma = delta_sigma + 2 * omega0 * dk_sampled * dn_sampled

        if delta_sigma is not None:
            delta_sigma = delta_sigma * EPSILON_0

        return delta_eps, delta_sigma

    def from_perturbation_delta_model(cls, deltas_model: AbstractDeltaModel) -> IndexPerturbation:
        """Create an IndexPerturbation from a DeltaPerturbationModel."""
        return IndexPerturbation(delta_n=deltas_model.delta_n, delta_k=deltas_model.delta_k)
