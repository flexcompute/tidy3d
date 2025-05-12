from __future__ import annotations

import functools
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union

import autograd.numpy as np
import pydantic.v1 as pd
import xarray as xr
from numpy.typing import NDArray

from tidy3d.components.autograd.derivative_utils import DerivativeInfo, integrate_within_bounds
from tidy3d.components.autograd.types import AutogradFieldMap
from tidy3d.components.base import Tidy3dBaseModel, cached_property, skip_if_fields_missing
from tidy3d.components.data.data_array import DATA_ARRAY_MAP
from tidy3d.components.data.dataset import ElectromagneticFieldDataset, PermittivityDataset
from tidy3d.components.data.unstructured.base import UnstructuredGridDataset
from tidy3d.components.data.utils import (
    CustomSpatialDataType,
    _get_numpy_array,
)
from tidy3d.components.grid.grid import Coords
from tidy3d.components.material.tcad.heat import ThermalSpecType
from tidy3d.components.time_modulation import ModulationSpec
from tidy3d.components.types import (
    TYPE_TAG_STR,
    ArrayComplex3D,
    Ax,
    Axis,
    Bound,
    FreqBound,
    InterpMethod,
    PermittivityComponent,
)
from tidy3d.components.validators import validate_name_str
from tidy3d.components.viz import VisualizationSpec, add_ax_if_none
from tidy3d.constants import EPSILON_0, HBAR, HERTZ
from tidy3d.exceptions import ValidationError
from tidy3d.log import log

from .nonlinear import NonlinearModel, NonlinearSpec, NonlinearSusceptibility
from .utils import ensure_freq_in_range


class AbstractMedium(ABC, Tidy3dBaseModel):
    """A medium within which electromagnetic waves propagate."""

    name: str = pd.Field(None, title="Name", description="Optional unique name for medium.")

    frequency_range: FreqBound = pd.Field(
        None,
        title="Frequency Range",
        description="Optional range of validity for the medium.",
        units=(HERTZ, HERTZ),
    )

    allow_gain: bool = pd.Field(
        False,
        title="Allow gain medium",
        description="Allow the medium to be active. Caution: "
        "simulations with a gain medium are unstable, and are likely to diverge."
        "Simulations where 'allow_gain' is set to 'True' will still be charged even if "
        "diverged. Monitor data up to the divergence point will still be returned and can be "
        "useful in some cases.",
    )

    nonlinear_spec: Union[NonlinearSpec, NonlinearSusceptibility] = pd.Field(
        None,
        title="Nonlinear Spec",
        description="Nonlinear spec applied on top of the base medium properties.",
    )

    modulation_spec: ModulationSpec = pd.Field(
        None,
        title="Modulation Spec",
        description="Modulation spec applied on top of the base medium properties.",
    )

    viz_spec: Optional[VisualizationSpec] = pd.Field(
        None,
        title="Visualization Specification",
        description="Plotting specification for visualizing medium.",
    )

    @cached_property
    def _nonlinear_models(self) -> List:
        """The nonlinear models in the nonlinear_spec."""
        if self.nonlinear_spec is None:
            return []
        if isinstance(self.nonlinear_spec, NonlinearModel):
            return [self.nonlinear_spec]
        if self.nonlinear_spec.models is None:
            return []
        return list(self.nonlinear_spec.models)

    @cached_property
    def _nonlinear_num_iters(self) -> pd.PositiveInt:
        """The num_iters of the nonlinear_spec."""
        if self.nonlinear_spec is None:
            return 0
        if isinstance(self.nonlinear_spec, NonlinearModel):
            if self.nonlinear_spec.numiters is None:
                return 1  # old default value for backwards compatibility
            return self.nonlinear_spec.numiters
        return self.nonlinear_spec.num_iters

    def _post_init_validators(self) -> None:
        """Call validators taking ``self`` that get run after init."""
        self._validate_nonlinear_spec()
        self._validate_modulation_spec_post_init()

    def _validate_nonlinear_spec(self):
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
                "Nonlinearities are not currently supported for the components " "of a 2D medium."
            )

        if self.nonlinear_spec is None:
            return
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

    def _validate_modulation_spec_post_init(self):
        """Check compatibility with nonlinear_spec."""
        if self.__class__.__name__ == "Medium2D" and any(
            comp.modulation_spec is not None for comp in [self.ss, self.tt]
        ):
            raise ValidationError(
                "Time modulation is not currently supported for the components " "of a 2D medium."
            )

    heat_spec: Optional[ThermalSpecType] = pd.Field(
        None,
        title="Heat Specification",
        description="DEPRECATED: Use `td.MultiPhysicsMedium`. Specification of the medium heat properties. They are "
        "used for solving the heat equation via the ``HeatSimulation`` interface. Such simulations can be"
        "used for investigating the influence of heat propagation on the properties of optical systems. "
        "Once the temperature distribution in the system is found using ``HeatSimulation`` object, "
        "``Simulation.perturbed_mediums_copy()`` can be used to convert mediums with perturbation "
        "models defined into spatially dependent custom mediums. "
        "Otherwise, the ``heat_spec`` does not directly affect the running of an optical "
        "``Simulation``.",
        discriminator=TYPE_TAG_STR,
    )

    @property
    def charge(self):
        return None

    @property
    def electrical(self):
        return None

    @property
    def heat(self):
        return self.heat_spec

    @property
    def optical(self):
        return None

    @pd.validator("modulation_spec", always=True)
    @skip_if_fields_missing(["nonlinear_spec"])
    def _validate_modulation_spec(cls, val, values):
        """Check compatibility with modulation_spec."""
        nonlinear_spec = values.get("nonlinear_spec")
        if val is not None and nonlinear_spec is not None:
            raise ValidationError(
                f"For medium class {cls}, 'modulation_spec' of class {type(val)} and "
                f"'nonlinear_spec' of class {type(nonlinear_spec)} are "
                "not simultaneously supported."
            )
        return val

    _name_validator = validate_name_str()

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
        return isinstance(self, AbstractCustomMedium)

    @cached_property
    def is_fully_anisotropic(self) -> bool:
        """Whether the medium is fully anisotropic."""
        from .anisotropic import FullyAnisotropicMedium

        return isinstance(self, FullyAnisotropicMedium)

    @cached_property
    def _incompatible_material_types(self) -> List[str]:
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

    def loss_tangent_model(self, frequency: float) -> Tuple[float, float]:
        """Permittivity and loss tangent as a function of frequency.

        Parameters
        ----------
        frequency : float
            Frequency to evaluate permittivity at (Hz).

        Returns
        -------
        Tuple[float, float]
            Real part of permittivity and loss tangent.
        """
        eps_complex = self.eps_model(frequency=frequency)
        return self.eps_complex_to_eps_loss_tangent(eps_complex)

    @ensure_freq_in_range
    def eps_diagonal(self, frequency: float) -> Tuple[complex, complex, complex]:
        """Main diagonal of the complex-valued permittivity tensor as a function of frequency.

        Parameters
        ----------
        frequency : float
            Frequency to evaluate permittivity at (Hz).

        Returns
        -------
        Tuple[complex, complex, complex]
            The diagonal elements of the relative permittivity tensor evaluated at ``frequency``.
        """

        # This only needs to be overwritten for anisotropic materials
        eps = self.eps_model(frequency)
        return (eps, eps, eps)

    def eps_diagonal_numerical(self, frequency: float) -> Tuple[complex, complex, complex]:
        """Main diagonal of the complex-valued permittivity tensor for numerical considerations
        such as meshing and runtime estimation.

        Parameters
        ----------
        frequency : float
            Frequency to evaluate permittivity at (Hz).

        Returns
        -------
        Tuple[complex, complex, complex]
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
        self, frequency: float, eps_component: Optional[PermittivityComponent] = None
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
    def n_cfl(self):
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
        eps_c = np.array(eps_c)
        ref_index = np.sqrt(eps_c)
        return np.real(ref_index), np.imag(ref_index)

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
        eps_real, eps_imag = eps_complex.real, eps_complex.imag
        omega = 2 * np.pi * freq
        sigma = omega * eps_imag * EPSILON_0
        return eps_real, sigma

    @staticmethod
    def eps_complex_to_eps_loss_tangent(eps_complex: complex) -> Tuple[float, float]:
        """Convert complex permittivity to permittivity and loss tangent.

        Parameters
        ----------
        eps_complex : complex
            Complex-valued relative permittivity.

        Returns
        -------
        Tuple[float, float]
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
    def eV_to_angular_freq(f_eV: float):
        """Convert frequency in unit of eV to rad/s.

        Parameters
        ----------
        f_eV : float
            Frequency in unit of eV
        """
        return f_eV / HBAR

    @staticmethod
    def angular_freq_to_eV(f_rad: float):
        """Convert frequency in unit of rad/s to eV.

        Parameters
        ----------
        f_rad : float
            Frequency in unit of rad/s
        """
        return f_rad * HBAR

    @staticmethod
    def angular_freq_to_Hz(f_rad: float):
        """Convert frequency in unit of rad/s to Hz.

        Parameters
        ----------
        f_rad : float
            Frequency in unit of rad/s
        """
        return f_rad / 2 / np.pi

    @staticmethod
    def Hz_to_angular_freq(f_hz: float):
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
    def is_pec(self):
        """Whether the medium is a PEC."""
        return False

    def sel_inside(self, bounds: Bound) -> AbstractMedium:
        """Return a new medium that contains the minimal amount data necessary to cover
        a spatial region defined by ``bounds``.


        Parameters
        ----------
        bounds : Tuple[float, float, float], Tuple[float, float float]
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

    def compute_derivatives(self, derivative_info: DerivativeInfo) -> AutogradFieldMap:
        """Compute the adjoint derivatives for this object."""
        raise NotImplementedError(f"Can't compute derivative for 'Medium': '{type(self)}'.")

    def derivative_eps_sigma_volume(
        self, E_der_map: ElectromagneticFieldDataset, bounds: Bound, freqs: NDArray
    ) -> dict[str, xr.DataArray]:
        """Get the derivative w.r.t permittivity and conductivity in the volume."""

        vjp_eps_complex = self.derivative_eps_complex_volume(
            E_der_map=E_der_map, bounds=bounds, freqs=freqs
        )

        values = vjp_eps_complex.values

        eps_vjp, sigma_vjp = self.eps_complex_to_eps_sigma(eps_complex=values, freq=freqs)

        eps_vjp = np.sum(eps_vjp)
        sigma_vjp = np.sum(sigma_vjp)

        return dict(permittivity=eps_vjp, conductivity=sigma_vjp)

    def derivative_eps_complex_volume(
        self, E_der_map: ElectromagneticFieldDataset, bounds: Bound, freqs: NDArray
    ) -> xr.DataArray:
        """Get the derivative w.r.t complex-valued permittivity in the volume."""

        vjp_value = 0.0
        for field_name in ("Ex", "Ey", "Ez"):
            fld = E_der_map[field_name].sel(f=freqs)
            vjp_value_fld = integrate_within_bounds(
                arr=fld,
                dims=("x", "y", "z"),
                bounds=bounds,
            )
            vjp_value += vjp_value_fld

        return vjp_value.sum("f")

    def __repr__(self):
        """If the medium has a name, use it as the representation. Otherwise, use the default representation."""
        if self.name:
            return self.name
        else:
            return super().__repr__()


class AbstractCustomMedium(AbstractMedium, ABC):
    """A spatially varying medium."""

    interp_method: InterpMethod = pd.Field(
        "nearest",
        title="Interpolation method",
        description="Interpolation method to obtain permittivity values "
        "that are not supplied at the Yee grids; For grids outside the range "
        "of the supplied data, extrapolation will be applied. When the extrapolated "
        "value is smaller (greater) than the minimal (maximal) of the supplied data, "
        "the extrapolated value will take the minimal (maximal) of the supplied data.",
    )

    subpixel: bool = pd.Field(
        False,
        title="Subpixel averaging",
        description="If ``True``, apply the subpixel averaging method specified by "
        "``Simulation``'s field ``subpixel`` for this type of material on the "
        "interface of the structure, including exterior boundary and "
        "intersection interfaces with other structures.",
    )

    @cached_property
    @abstractmethod
    def is_isotropic(self) -> bool:
        """The medium is isotropic or anisotropic."""

    def _interp_method(self, comp: Axis) -> InterpMethod:
        """Interpolation method applied to comp."""
        return self.interp_method

    @abstractmethod
    def eps_dataarray_freq(
        self, frequency: float
    ) -> Tuple[CustomSpatialDataType, CustomSpatialDataType, CustomSpatialDataType]:
        """Permittivity array at ``frequency``.

        Parameters
        ----------
        frequency : float
            Frequency to evaluate permittivity at (Hz).

        Returns
        -------
        Tuple[
            Union[
                :class:`.SpatialDataArray`,
                :class:`.TriangularGridDataset`,
                :class:`.TetrahedralGridDataset`
            ],
            Union[
                :class:`.SpatialDataArray`,
                :class:`.TriangularGridDataset`,
                :class:`.TetrahedralGridDataset`
            ],
            Union[
                :class:`.SpatialDataArray`,
                :class:`.TriangularGridDataset`,
                :class:`.TetrahedralGridDataset`
            ],
        ]
            The permittivity evaluated at ``frequency``.
        """

    def eps_diagonal_on_grid(
        self,
        frequency: float,
        coords: Coords,
    ) -> Tuple[ArrayComplex3D, ArrayComplex3D, ArrayComplex3D]:
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
        Tuple[ArrayComplex3D, ArrayComplex3D, ArrayComplex3D]
            The complex-valued permittivity tensor at ``frequency`` interpolated
            at the supplied coordinate.
        """
        eps_spatial = self.eps_dataarray_freq(frequency)
        if self.is_isotropic:
            eps_interp = _get_numpy_array(
                coords.spatial_interp(eps_spatial[0], self._interp_method(0))
            )
            return (eps_interp, eps_interp, eps_interp)
        return tuple(
            _get_numpy_array(coords.spatial_interp(eps_comp, self._interp_method(comp)))
            for comp, eps_comp in enumerate(eps_spatial)
        )

    def eps_comp_on_grid(
        self,
        row: Axis,
        col: Axis,
        frequency: float,
        coords: Coords,
    ) -> ArrayComplex3D:
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
        ArrayComplex3D
            Single component of the complex-valued permittivity tensor at ``frequency`` interpolated
            at the supplied coordinates.
        """

        if row == col:
            return self.eps_diagonal_on_grid(frequency, coords)[row]
        return 0j

    @ensure_freq_in_range
    def eps_model(self, frequency: float) -> complex:
        """Complex-valued spatially averaged permittivity as a function of frequency."""
        if self.is_isotropic:
            return np.mean(_get_numpy_array(self.eps_dataarray_freq(frequency)[0]))
        return np.mean(
            [np.mean(_get_numpy_array(eps_comp)) for eps_comp in self.eps_dataarray_freq(frequency)]
        )

    @ensure_freq_in_range
    def eps_diagonal(self, frequency: float) -> Tuple[complex, complex, complex]:
        """Main diagonal of the complex-valued permittivity tensor
        at ``frequency``. Spatially, we take max{||eps||}, so that autoMesh generation
        works appropriately.
        """
        eps_spatial = self.eps_dataarray_freq(frequency)
        if self.is_isotropic:
            eps_comp = _get_numpy_array(eps_spatial[0]).ravel()
            eps = eps_comp[np.argmax(np.abs(eps_comp))]
            return (eps, eps, eps)
        eps_spatial_array = (_get_numpy_array(eps_comp).ravel() for eps_comp in eps_spatial)
        return tuple(eps_comp[np.argmax(np.abs(eps_comp))] for eps_comp in eps_spatial_array)

    def _get_real_vals(self, x: np.ndarray) -> np.ndarray:
        """Grab the real part of the values in array.
        Used for _eps_bounds()
        """
        return _get_numpy_array(np.real(x)).ravel()

    def _eps_bounds(
        self, frequency: float = None, eps_component: Optional[PermittivityComponent] = None
    ) -> Tuple[float, float]:
        """Returns permittivity bounds for setting the color bounds when plotting.

        Parameters
        ----------
        frequency : float = None
            Frequency to evaluate the relative permittivity of all mediums.
            If not specified, evaluates at infinite frequency.
        eps_component : Optional[PermittivityComponent] = None
            Component of the permittivity tensor to plot for anisotropic materials,
            e.g. ``"xx"``, ``"yy"``, ``"zz"``, ``"xy"``, ``"yz"``, ...
            Defaults to ``None``, which returns the average of the diagonal values.

        Returns
        -------
        Tuple[float, float]
            The min and max values of the permittivity for the selected component and evaluated at ``frequency``.
        """
        eps_dataarray = self.eps_dataarray_freq(frequency)
        all_eps = np.concatenate(self._get_real_vals(eps_comp) for eps_comp in eps_dataarray)
        return (np.min(all_eps), np.max(all_eps))

    @staticmethod
    def _validate_isreal_dataarray(dataarray: CustomSpatialDataType) -> bool:
        """Validate that the dataarray is real"""
        return np.all(np.isreal(_get_numpy_array(dataarray)))

    @staticmethod
    def _validate_isreal_dataarray_tuple(
        dataarray_tuple: Tuple[CustomSpatialDataType, ...],
    ) -> bool:
        """Validate that the dataarray is real"""
        return np.all([AbstractCustomMedium._validate_isreal_dataarray(f) for f in dataarray_tuple])

    @abstractmethod
    def _sel_custom_data_inside(self, bounds: Bound):
        """Return a new medium that contains the minimal amount custom data necessary to cover
        a spatial region defined by ``bounds``."""

    def sel_inside(self, bounds: Bound) -> AbstractCustomMedium:
        """Return a new medium that contains the minimal amount data necessary to cover
        a spatial region defined by ``bounds``.


        Parameters
        ----------
        bounds : Tuple[float, float, float], Tuple[float, float float]
            Min and max bounds packaged as ``(minx, miny, minz), (maxx, maxy, maxz)``.

        Returns
        -------
        AbstractMedium
            Medium with reduced data.
        """

        self_mod_data_reduced = super().sel_inside(bounds)

        return self_mod_data_reduced._sel_custom_data_inside(bounds)

    @staticmethod
    def _not_loaded(field):
        """Check whether data was not loaded."""
        if isinstance(field, str) and field in DATA_ARRAY_MAP:
            return True
        # attempting to construct an UnstructuredGridDataset from a dict
        elif isinstance(field, dict) and field.get("type") in (
            "TriangularGridDataset",
            "TetrahedralGridDataset",
        ):
            return any(
                isinstance(subfield, str) and subfield in DATA_ARRAY_MAP
                for subfield in [field["points"], field["cells"], field["values"]]
            )
        # attempting to pass an UnstructuredGridDataset with zero points
        elif isinstance(field, UnstructuredGridDataset):
            return any(len(subfield) == 0 for subfield in [field.points, field.cells, field.values])

    def _derivative_field_cmp(
        self,
        E_der_map: ElectromagneticFieldDataset,
        eps_data: PermittivityDataset,
        dim: str,
        freqs: NDArray,
    ) -> np.ndarray:
        coords_interp = {key: val for key, val in eps_data.coords.items() if len(val) > 1}
        dims_sum = {dim for dim in eps_data.coords.keys() if dim not in coords_interp}

        # compute sizes along each of the interpolation dimensions
        sizes_list = []
        for _, coords in coords_interp.items():
            num_coords = len(coords)
            coords = np.array(coords)

            # compute distances between midpoints for all internal coords
            mid_points = (coords[1:] + coords[:-1]) / 2.0
            dists = np.diff(mid_points)
            sizes = np.zeros(num_coords)
            sizes[1:-1] = dists

            # estimate the sizes on the edges using 2 x the midpoint distance
            sizes[0] = 2 * abs(mid_points[0] - coords[0])
            sizes[-1] = 2 * abs(coords[-1] - mid_points[-1])

            sizes_list.append(sizes)

        # turn this into a volume element, should be re-sizeable to the gradient shape
        if sizes_list:
            d_vol = functools.reduce(np.outer, sizes_list)
        else:
            # if sizes_list is empty, then reduce() fails
            d_vol = np.array(1.0)

        # TODO: probably this could be more robust. eg if the DataArray has weird edge cases
        E_der_dim = E_der_map[f"E{dim}"].sel(f=freqs)
        E_der_dim_interp = (
            E_der_dim.interp(**coords_interp, assume_sorted=True).fillna(0.0).sum(dims_sum).sum("f")
        )
        vjp_array = np.array(E_der_dim_interp.values).astype(complex)
        vjp_array = vjp_array.reshape(eps_data.shape)

        # multiply by volume elements (if possible, being defensive here..)
        try:
            vjp_array *= d_vol.reshape(vjp_array.shape)
        except ValueError:
            log.warning(
                "Skipping volume element normalization of 'CustomMedium' gradients. "
                f"Could not reshape the volume elements of shape {d_vol.shape} "
                f"to the shape of the gradient {vjp_array.shape}. "
                "If you encounter this warning, gradient direction will be accurate but the norm "
                "will be inaccurate. Please raise an issue on the tidy3d front end with this "
                "message and some information about your simulation setup and we will investigate. "
            )
        return vjp_array
