"""Shared abstractions and helpers for electromagnetic medium models."""

from __future__ import annotations

import functools
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, ClassVar, get_args

import autograd.numpy as np
import numpy as npo
from numpy.typing import NDArray
from pydantic import Field, field_validator, model_validator

from tidy3d.components.autograd.derivative_utils import (
    integrate_within_bounds,
)
from tidy3d.components.autograd.path_utils import (
    format_traced_path,
    format_traced_paths,
    raise_unsupported_traced_path,
    traced_paths,
    validate_traced_path,
)
from tidy3d.components.autograd.types import (
    PathType,
)
from tidy3d.components.base import Tidy3dBaseModel, cached_property
from tidy3d.components.data.utils import UnstructuredGridDatasetType
from tidy3d.components.material.tcad.heat import ThermalSpecType
from tidy3d.components.nonlinear import (
    NonlinearModel,
    NonlinearSpec,
    NonlinearSpecType,
    NonlinearSusceptibility,
)
from tidy3d.components.time_modulation import ModulationSpec
from tidy3d.components.types import TYPE_TAG_STR, FreqBound, InterpMethod
from tidy3d.components.validators import (
    validate_name_str,
)
from tidy3d.components.viz import VisualizationSpec, add_ax_if_none
from tidy3d.constants import (
    EPSILON_0,
    HBAR,
    HERTZ,
    fp_eps,
)
from tidy3d.exceptions import AdjointError, ValidationError
from tidy3d.log import log

if TYPE_CHECKING:
    from typing import NoReturn

    import xarray as xr
    from numpy.typing import ArrayLike
    from pydantic import PositiveInt

    from tidy3d.compat import Self
    from tidy3d.components.autograd.derivative_utils import DerivativeInfo
    from tidy3d.components.autograd.path_utils import AutogradRoute
    from tidy3d.components.autograd.types import AutogradFieldMap
    from tidy3d.components.data.dataset import ElectromagneticFieldDataset
    from tidy3d.components.types import (
        Ax,
        Axis,
        Bound,
        PermittivityComponent,
    )


ArrayFloat = NDArray[npo.floating]
ArrayComplex = NDArray[np.complexfloating]
ArrayGeneric = NDArray[Any]
FrequencyArray = Sequence[float] | ArrayFloat
WeightFunction = Callable[[float], ArrayComplex]
ComplexArrayOrScalar = complex | ArrayGeneric

_EPS_SIGMA_TRACED_PATHS = traced_paths("permittivity", "conductivity")


def _validate_traced_custom_data_path(
    medium_name: str,
    field_path: tuple[Any, ...],
    *,
    scalar_data: Mapping[str, Any] | None = None,
    indexed_data: Mapping[str, Sequence[Sequence[Any]]] | None = None,
) -> None:
    """Reject unstructured custom data at a validated traced medium path."""
    scalar_data = scalar_data or {}
    indexed_data = indexed_data or {}

    if len(field_path) >= 1 and field_path[0] in scalar_data:
        custom_data_path = field_path[:1]
        spatial_data = scalar_data[field_path[0]]
    elif len(field_path) >= 3 and field_path[0] in indexed_data:
        data_values = indexed_data[field_path[0]]
        component_values = data_values[field_path[1]]
        custom_data_path = field_path[:3]
        spatial_data = component_values[field_path[2]]
    else:
        return

    if isinstance(spatial_data, UnstructuredGridDatasetType):
        parameter = format_traced_path(custom_data_path)
        raise AdjointError(
            f"Automatic differentiation with respect to medium parameter '{parameter}' is not "
            f"supported for medium type '{medium_name}' when the traced custom data is "
            "unstructured. Use structured SpatialDataArray data or provide a custom_vjp."
        )


# evaluate frequency as this number (Hz) if inf
FREQ_EVAL_INF = 1e50

# extrapolation option in custom medium
FILL_VALUE = "extrapolate"

ALLOWED_INTERP_METHODS = get_args(InterpMethod)


def _normalize_frequency_input(
    frequency: float | FrequencyArray | None,
) -> float | ArrayFloat:
    """Normalize frequency inputs to a scalar float or float array."""
    if frequency is None:
        return FREQ_EVAL_INF

    if np.isscalar(frequency):
        frequency = float(frequency)
        return FREQ_EVAL_INF if np.isinf(frequency) else frequency

    frequency = np.array(frequency, dtype=float, copy=True)
    frequency[np.isinf(frequency)] = FREQ_EVAL_INF
    return frequency


def _constant_over_frequency(
    value: complex, frequency: float | FrequencyArray | None
) -> complex | ArrayComplex:
    """Return a scalar constant or broadcast it over the supplied frequency shape."""
    frequency = _normalize_frequency_input(frequency)
    if np.isscalar(frequency):
        return complex(value)
    # Constant media should still return one value per frequency sample when the
    # caller passes a vectorized frequency input.
    return np.full(np.shape(frequency), value, dtype=complex)


def ensure_freq_in_range(
    eps_model: Callable[[AbstractMedium, float], complex],
) -> Callable[[AbstractMedium, float], complex]:
    """Decorate ``eps_model`` to log warning if frequency supplied is out of bounds."""

    @functools.wraps(eps_model)
    def _eps_model(self: AbstractMedium, frequency: float) -> complex:
        """New eps_model function."""
        # evaluate infs and None as FREQ_EVAL_INF
        is_inf_scalar = frequency is None or (np.isscalar(frequency) and np.isinf(frequency))
        frequency = _normalize_frequency_input(frequency)

        # if frequency range not present just return original function
        if self.frequency_range is None:
            return eps_model(self, frequency)

        fmin, fmax = self.frequency_range
        # don't warn for evaluating infinite frequency
        if is_inf_scalar:
            return eps_model(self, frequency)

        outside_lower = np.zeros_like(frequency, dtype=bool)
        outside_upper = np.zeros_like(frequency, dtype=bool)

        if fmin > 0:
            outside_lower = frequency / fmin < 1 - fp_eps
        elif fmin == 0:
            outside_lower = frequency < 0

        if fmax > 0:
            outside_upper = frequency / fmax > 1 + fp_eps

        if np.any(outside_lower | outside_upper):
            log.warning(
                "frequency passed to 'Medium.eps_model()'"
                f"is outside of 'Medium.frequency_range' = {self.frequency_range}",
                capture=False,
            )
        return eps_model(self, frequency)

    return _eps_model


""" Medium Definitions """


class AbstractMedium(ABC, Tidy3dBaseModel):
    """A medium within which electromagnetic waves propagate."""

    _traced_supported_paths: ClassVar[tuple[PathType, ...]] = ()

    name: str | None = Field(
        default=None, title="Name", description="Optional unique name for medium."
    )

    frequency_range: FreqBound | None = Field(
        default=None,
        title="Frequency Range",
        description="Optional range of validity for the medium.",
        json_schema_extra={"units": (HERTZ, HERTZ)},
    )

    allow_gain: bool = Field(
        default=False,
        title="Allow gain medium",
        description="Allow the medium to be active. Caution: "
        "simulations with a gain medium are unstable, and are likely to diverge."
        "Simulations where ``allow_gain`` is set to ``True`` will still be charged even if "
        "diverged. Monitor data up to the divergence point will still be returned and can be "
        "useful in some cases.",
    )

    nonlinear_spec: NonlinearSpecType | None = Field(
        default=None,
        title="Nonlinear Spec",
        description="Nonlinear spec applied on top of the base medium properties.",
    )

    modulation_spec: ModulationSpec | None = Field(
        default=None,
        title="Modulation Spec",
        description="Modulation spec applied on top of the base medium properties.",
    )

    viz_spec: VisualizationSpec | None = Field(
        default=None,
        title="Visualization Specification",
        description="Plotting specification for visualizing medium.",
    )

    heat_spec: ThermalSpecType | None = Field(
        default=None,
        title="Heat Specification",
        description="DEPRECATED: Use :class:`~tidy3d.MultiPhysicsMedium`. Specification of the medium heat properties. They are "
        "used for solving the heat equation via the :class:`~tidy3d.HeatSimulation` interface. Such simulations can be"
        "used for investigating the influence of heat propagation on the properties of optical systems. "
        "Once the temperature distribution in the system is found using :class:`~tidy3d.HeatSimulation` object, "
        "``Simulation.perturbed_mediums_copy()`` can be used to convert mediums with perturbation "
        "models defined into spatially dependent custom mediums. "
        "Otherwise, the ``heat_spec`` does not directly affect the running of an optical "
        "``Simulation``.",
        discriminator=TYPE_TAG_STR,
    )

    @model_validator(mode="after")
    def _run_after_validators(self) -> Self:
        """Run post-init validations in an explicit, dependency-aware order."""
        self._validate_nonlinear_spec()
        self._check_either_modulation_or_nonlinear_spec()
        self._validate_modulation_spec_after()
        return self

    @field_validator("nonlinear_spec", mode="before")
    @classmethod
    def _add_nonlinear_spec_type_to_legacy_mapping(cls, val: Any) -> Any:
        """Add a discriminator to legacy raw dict nonlinear_spec inputs."""
        if not isinstance(val, Mapping) or not val or TYPE_TAG_STR in val:
            return val

        spec_type = (
            "NonlinearSpec" if "models" in val or "num_iters" in val else "NonlinearSusceptibility"
        )
        return {TYPE_TAG_STR: spec_type, **val}

    def _validate_nonlinear_spec(self) -> Self:
        """Check compatibility with nonlinear_spec."""
        if self.__class__.__name__ == "AnisotropicMedium" and any(
            comp.nonlinear_spec is not None for comp in [self.xx, self.yy, self.zz]
        ):
            raise ValidationError(
                "Nonlinearities are not currently supported for the components "
                "of an anisotropic medium."
            )
        if self.__class__.__name__ == "Medium2D" and any(
            comp.nonlinear_spec is not None for comp in [self.ss, self.tt]
        ):
            raise ValidationError(
                "Nonlinearities are not currently supported for the components of a 2D medium."
            )

        if self.nonlinear_spec is None:
            return self
        if isinstance(self.nonlinear_spec, NonlinearModel):
            log.warning(
                "The API for 'nonlinear_spec' has changed. "
                "The old usage 'nonlinear_spec=model' is deprecated and will be removed "
                "in a future release. The new usage is "
                r"'nonlinear_spec=NonlinearSpec(models=\[model])'."
            )
        for model in self._nonlinear_models:
            model._validate_medium_type(self)
            model._validate_medium(self)
            if (
                isinstance(self.nonlinear_spec, NonlinearSpec)
                and isinstance(model, NonlinearSusceptibility)
                and model.numiters is not None
            ):
                raise ValidationError(
                    "'NonlinearSusceptibility.numiters' is deprecated. "
                    "Please use 'NonlinearSpec.num_iters' instead."
                )
        return self

    def _check_either_modulation_or_nonlinear_spec(self) -> Self:
        """Check compatibility with modulation_spec."""
        val = self.modulation_spec
        nonlinear_spec = self.nonlinear_spec
        if val is not None and nonlinear_spec is not None:
            raise ValidationError(
                f"For medium class {self.type}, 'modulation_spec' of class {type(val).__name__} and "
                f"'nonlinear_spec' of class {type(nonlinear_spec).__name__} are "
                "not simultaneously supported."
            )
        return self

    _name_validator = validate_name_str()

    def _validate_modulation_spec_after(self) -> Self:
        """Check compatibility with nonlinear_spec."""
        if self.__class__.__name__ == "Medium2D" and any(
            comp.modulation_spec is not None for comp in [self.ss, self.tt]
        ):
            raise ValidationError(
                "Time modulation is not currently supported for the components of a 2D medium."
            )
        return self

    @property
    def charge(self) -> None:
        return None

    @property
    def electrical(self) -> None:
        return None

    @property
    def heat(self) -> ThermalSpecType | None:
        return self.heat_spec

    @property
    def optical(self) -> None:
        return None

    @cached_property
    def _nonlinear_models(self) -> list:
        """The nonlinear models in the nonlinear_spec."""
        if self.nonlinear_spec is None:
            return []
        if isinstance(self.nonlinear_spec, NonlinearModel):
            return [self.nonlinear_spec]
        if self.nonlinear_spec.models is None:
            return []
        return list(self.nonlinear_spec.models)

    @cached_property
    def _nonlinear_num_iters(self) -> PositiveInt:
        """The num_iters of the nonlinear_spec."""
        if self.nonlinear_spec is None:
            return 0
        if isinstance(self.nonlinear_spec, NonlinearModel):
            if self.nonlinear_spec.numiters is None:
                return 1  # old default value for backwards compatibility
            return self.nonlinear_spec.numiters
        return self.nonlinear_spec.num_iters

    @cached_property
    def is_spatially_uniform(self) -> bool:
        """Whether the medium is spatially uniform."""
        return True

    @cached_property
    def is_time_modulated(self) -> bool:
        """Whether any component of the medium is time modulated."""
        return self.modulation_spec is not None and self.modulation_spec.applied_modulation

    @cached_property
    def is_nonlinear(self) -> bool:
        """Whether the medium is nonlinear."""
        return self.nonlinear_spec is not None

    @cached_property
    def is_custom(self) -> bool:
        """Whether the medium is custom."""
        return False

    @cached_property
    def is_fully_anisotropic(self) -> bool:
        """Whether the medium is fully anisotropic."""
        return False

    @cached_property
    def _incompatible_material_types(self) -> list[str]:
        """A list of material properties present which may lead to incompatibilities."""
        properties = [
            self.is_time_modulated,
            self.is_nonlinear,
            self.is_custom,
            self.is_fully_anisotropic,
        ]
        names = ["time modulated", "nonlinear", "custom", "fully anisotropic"]
        types = [name for name, prop in zip(names, properties) if prop]
        return types

    @cached_property
    def _has_incompatibilities(self) -> bool:
        """Whether the medium has incompatibilities. Certain medium types are incompatible
        with certain others, and such pairs are not allowed to intersect in a simulation."""
        return len(self._incompatible_material_types) > 0

    def _compatible_with(self, other: AbstractMedium) -> bool:
        """Whether these two media are compatible if in structures that intersect."""
        if not (self._has_incompatibilities and other._has_incompatibilities):
            return True
        for med1, med2 in [(self, other), (other, self)]:
            if med1.is_custom:
                # custom and fully_anisotropic is OK
                if med2.is_nonlinear or med2.is_time_modulated:
                    return False
            if med1.is_fully_anisotropic:
                if med2.is_nonlinear or med2.is_time_modulated:
                    return False
            if med1.is_nonlinear:
                if med2.is_time_modulated:
                    return False
        return True

    @abstractmethod
    def eps_model(self, frequency: float) -> complex:
        # TODO this should be moved out of here into FDTD Simulation Mediums?
        """Complex-valued permittivity as a function of frequency.

        Parameters
        ----------
        frequency : float or ArrayLike
            Frequency or frequencies to evaluate permittivity at (Hz).

        Returns
        -------
        complex
            Complex-valued relative permittivity evaluated at ``frequency``.
        """

    def nk_model(self, frequency: float) -> tuple[float, float]:
        """Real and imaginary parts of the refactive index as a function of frequency.

        Parameters
        ----------
        frequency : float
            Frequency to evaluate permittivity at (Hz).

        Returns
        -------
        tuple[float, float]
            Real part (n) and imaginary part (k) of refractive index of medium.
        """
        eps_complex = self.eps_model(frequency=frequency)
        return self.eps_complex_to_nk(eps_complex)

    def loss_tangent_model(self, frequency: float) -> tuple[float, float]:
        """Permittivity and loss tangent as a function of frequency.

        Parameters
        ----------
        frequency : float
            Frequency to evaluate permittivity at (Hz).

        Returns
        -------
        tuple[float, float]
            Real part of permittivity and loss tangent.
        """
        eps_complex = self.eps_model(frequency=frequency)
        return self.eps_complex_to_eps_loss_tangent(eps_complex)

    @ensure_freq_in_range
    def eps_diagonal(self, frequency: float) -> tuple[complex, complex, complex]:
        """Main diagonal of the complex-valued permittivity tensor as a function of frequency.

        Parameters
        ----------
        frequency : float
            Frequency to evaluate permittivity at (Hz).

        Returns
        -------
        tuple[complex, complex, complex]
            The diagonal elements of the relative permittivity tensor evaluated at ``frequency``.
        """

        # This only needs to be overwritten for anisotropic materials
        eps = self.eps_model(frequency)
        return (eps, eps, eps)

    def eps_diagonal_numerical(self, frequency: float) -> tuple[complex, complex, complex]:
        """Main diagonal of the complex-valued permittivity tensor for numerical considerations
        such as meshing and runtime estimation.

        Parameters
        ----------
        frequency : float
            Frequency to evaluate permittivity at (Hz).

        Returns
        -------
        tuple[complex, complex, complex]
            The diagonal elements of relative permittivity tensor relevant for numerical
            considerations evaluated at ``frequency``.
        """

        if self.is_pec:
            # also 1 for lossy metal and Medium2D, but let's handle them in the subclass.
            return (1.0 + 0j,) * 3

        return self.eps_diagonal(frequency)

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

    def _eps_plot(
        self, frequency: float, eps_component: PermittivityComponent | None = None
    ) -> float:
        """Returns real part of epsilon for plotting. A specific component of the epsilon tensor can
        be selected for anisotropic medium.

        Parameters
        ----------
        frequency : float
            Frequency to evaluate permittivity at.
        eps_component : PermittivityComponent
            Component of the permittivity tensor to plot
            e.g. ``"xx"``, ``"yy"``, ``"zz"``, ``"xy"``, ``"yz"``, ...
            Defaults to ``None``, which returns the average of the diagonal values.

        Returns
        -------
        float
            Element ``eps_component`` of the relative permittivity tensor evaluated at ``frequency``.
        """
        # Assumes the material is isotropic
        # Will need to be overridden for anisotropic materials
        return self.eps_model(frequency).real

    @cached_property
    @abstractmethod
    def n_cfl(self) -> float:
        # TODO this should be moved out of here into FDTD Simulation Mediums?
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
    def plot(self, freqs: float, ax: Ax = None) -> Ax:
        """Plot n, k of a :class:`.Medium` as a function of frequency.

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
        eps_complex = np.array([self.eps_model(freq) for freq in freqs])
        n, k = AbstractMedium.eps_complex_to_nk(eps_complex)

        freqs_thz = freqs / 1e12
        ax.plot(freqs_thz, n, label="n")
        ax.plot(freqs_thz, k, label="k")
        ax.set_xlabel("frequency (THz)")
        ax.set_title("medium dispersion")
        ax.legend()
        ax.set_aspect("auto")
        return ax

    def background_index_from_freqs(self, freqs: ArrayLike) -> NDArray:
        """Complex refractive index sampled at the provided frequencies."""
        freqs_arr = np.asarray(freqs, dtype=float)
        background_n = np.zeros(freqs_arr.size, dtype=complex)
        for freq_id, freq in enumerate(freqs_arr):
            eps = self.eps_model(float(freq))
            n_val, k_val = self.eps_complex_to_nk(eps)
            background_n[freq_id] = np.squeeze(n_val) + 1j * np.squeeze(k_val)
        return background_n

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
            Complex-valued relative permittivity.
        """
        eps_real = n**2 - k**2
        eps_imag = 2 * n * k
        return eps_real + 1j * eps_imag

    @staticmethod
    def eps_complex_to_nk(eps_c: complex) -> tuple[float, float]:
        """Convert complex permittivity to n, k values.

        Parameters
        ----------
        eps_c : complex
            Complex-valued relative permittivity.

        Returns
        -------
        tuple[float, float]
            Real and imaginary parts of refractive index (n & k).
        """
        eps_c = np.array(eps_c)
        ref_index = np.sqrt(eps_c)
        return np.real(ref_index), np.imag(ref_index)

    @staticmethod
    def nk_to_eps_sigma(n: float, k: float, freq: float) -> tuple[float, float]:
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
        tuple[float, float]
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
        freq = _normalize_frequency_input(freq)
        omega = 2 * np.pi * freq
        return AbstractMedium._eps_sigma_to_eps_complex_from_omega(eps_real, sigma, omega)

    @staticmethod
    def _eps_sigma_to_eps_complex_from_omega(
        eps_real: float | ArrayGeneric | xr.DataArray,
        sigma: float | ArrayGeneric | xr.DataArray,
        omega: float | ArrayGeneric | xr.DataArray,
    ) -> complex | ArrayGeneric | xr.DataArray:
        """Convert permittivity and conductivity to complex permittivity from angular frequency."""
        return eps_real + 1j * sigma / omega / EPSILON_0

    @staticmethod
    def eps_complex_to_eps_sigma(eps_complex: complex, freq: float) -> tuple[float, float]:
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
        tuple[float, float]
            Real part of relative permittivity & electric conductivity.
        """
        eps_real, eps_imag = eps_complex.real, eps_complex.imag
        omega = 2 * np.pi * freq
        sigma = omega * eps_imag * EPSILON_0
        return eps_real, sigma

    @staticmethod
    def eps_complex_to_eps_loss_tangent(eps_complex: complex) -> tuple[float, float]:
        """Convert complex permittivity to permittivity and loss tangent.

        Parameters
        ----------
        eps_complex : complex
            Complex-valued relative permittivity.

        Returns
        -------
        tuple[float, float]
            Real part of relative permittivity & loss tangent
        """
        eps_real, eps_imag = eps_complex.real, eps_complex.imag
        return eps_real, eps_imag / eps_real

    @staticmethod
    def eps_loss_tangent_to_eps_complex(eps_real: float, loss_tangent: float) -> complex:
        """Convert permittivity and loss tangent to complex permittivity.

        Parameters
        ----------
        eps_real : float
            Real part of relative permittivity
        loss_tangent : float
            Loss tangent

        Returns
        -------
        eps_complex : complex
            Complex-valued relative permittivity.
        """
        return eps_real * (1 + 1j * loss_tangent)

    @staticmethod
    def eV_to_angular_freq(f_eV: float) -> float:
        """Convert frequency in unit of eV to rad/s.

        Parameters
        ----------
        f_eV : float
            Frequency in unit of eV
        """
        return f_eV / HBAR

    @staticmethod
    def angular_freq_to_eV(f_rad: float) -> float:
        """Convert frequency in unit of rad/s to eV.

        Parameters
        ----------
        f_rad : float
            Frequency in unit of rad/s
        """
        return f_rad * HBAR

    @staticmethod
    def angular_freq_to_Hz(f_rad: float) -> float:
        """Convert frequency in unit of rad/s to Hz.

        Parameters
        ----------
        f_rad : float
            Frequency in unit of rad/s
        """
        return f_rad / 2 / np.pi

    @staticmethod
    def Hz_to_angular_freq(f_hz: float) -> float:
        """Convert frequency in unit of Hz to rad/s.

        Parameters
        ----------
        f_hz : float
            Frequency in unit of Hz
        """
        return f_hz * 2 * np.pi

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

    @cached_property
    def is_pec(self) -> bool:
        """Whether the medium is a PEC."""
        return False

    @cached_property
    def is_pec_like(self) -> bool:
        """Whether the medium is treated as a PEC medium in surface monitors."""
        return self.is_pec

    @cached_property
    def is_pmc(self) -> bool:
        """Whether the medium is a PMC."""
        return False

    def sel_inside(self, bounds: Bound) -> AbstractMedium:
        """Return a new medium that contains the minimal amount data necessary to cover
        a spatial region defined by ``bounds``.


        Parameters
        ----------
        bounds : tuple[float, float, float], tuple[float, float float]
            Min and max bounds packaged as ``(minx, miny, minz), (maxx, maxy, maxz)``.

        Returns
        -------
        AbstractMedium
            Medium with reduced data.
        """

        if self.modulation_spec is not None:
            modulation_reduced = self.modulation_spec.sel_inside(bounds)
            return self.updated_copy(modulation_spec=modulation_reduced)

        return self

    """ Autograd code """

    @classmethod
    def _traced_autograd_supported_parameters(cls) -> tuple[str, ...]:
        """Return user-facing supported parameter names for setup validation."""
        return format_traced_paths(cls._traced_supported_paths)

    def _raise_unsupported_traced_path(
        self,
        field_path: tuple[Any, ...],
        *,
        supported_parameters: tuple[str, ...] | None = None,
    ) -> NoReturn:
        """Raise a user-facing validation error for an unsupported medium trace."""
        raise_unsupported_traced_path(
            parameter_kind="medium",
            owner_kind="medium type",
            owner_name=type(self).__name__,
            field_path=field_path,
            supported_parameters=(
                type(self)._traced_autograd_supported_parameters()
                if supported_parameters is None
                else supported_parameters
            ),
        )

    def _resolve_autograd_route(self, field_path: tuple[Any, ...]) -> AutogradRoute:
        """Resolve and validate one traced medium path for adjoint routing."""
        return validate_traced_path(
            parameter_kind="medium",
            owner_kind="medium type",
            owner_name=type(self).__name__,
            field_path=field_path,
            supported_paths=self._traced_supported_paths,
            supported_parameters=type(self)._traced_autograd_supported_parameters(),
        )

    def _compute_derivatives(self, derivative_info: DerivativeInfo) -> AutogradFieldMap:
        """Compute the adjoint derivatives for this object."""
        raise NotImplementedError(f"Can't compute derivative for 'Medium': '{type(self)}'.")

    def _derivative_eps_sigma_volume(
        self, E_der_map: ElectromagneticFieldDataset, bounds: Bound
    ) -> dict[str, xr.DataArray]:
        """Get the derivative w.r.t permittivity and conductivity in the volume."""

        vjp_eps_complex = self._derivative_eps_complex_volume(E_der_map=E_der_map, bounds=bounds)

        values = vjp_eps_complex.values

        # compute directly with frequency dimension
        freqs = vjp_eps_complex.coords["f"].values
        omegas = 2 * np.pi * freqs
        eps_vjp = np.real(values)
        sigma_vjp = -np.imag(values) / omegas / EPSILON_0

        eps_vjp = np.sum(eps_vjp)
        sigma_vjp = np.sum(sigma_vjp)

        return {"permittivity": eps_vjp, "conductivity": sigma_vjp}

    def _derivative_eps_complex_volume(
        self, E_der_map: ElectromagneticFieldDataset, bounds: Bound
    ) -> xr.DataArray:
        """Get the derivative w.r.t complex-valued permittivity in the volume."""
        vjp_value = None
        for field_name in ("Ex", "Ey", "Ez"):
            fld = E_der_map[field_name]
            vjp_value_fld = integrate_within_bounds(
                arr=fld,
                dims=("x", "y", "z"),
                bounds=bounds,
            )
            if vjp_value is None:
                vjp_value = vjp_value_fld
            else:
                vjp_value += vjp_value_fld

        return vjp_value

    def __repr__(self) -> str:
        """If the medium has a name, use it as the representation. Otherwise, use the default representation."""
        if self.name:
            return self.name
        return super().__repr__()
