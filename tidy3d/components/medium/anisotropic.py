"""Uniform diagonal and fully anisotropic medium models."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import autograd.numpy as np
from pydantic import (
    Field,
    field_validator,
    model_validator,
)

from tidy3d.components.autograd.path_utils import resolve_delegated_autograd_route
from tidy3d.components.base import cached_property
from tidy3d.components.types import TYPE_TAG_STR, TensorReal
from tidy3d.components.viz import add_ax_if_none
from tidy3d.constants import (
    CONDUCTIVITY,
    PERMITTIVITY,
    fp_eps,
)
from tidy3d.exceptions import ValidationError
from tidy3d.log import log

if TYPE_CHECKING:
    from pydantic import FieldValidationInfo

    from tidy3d.compat import Self
    from tidy3d.components.autograd.derivative_utils import DerivativeInfo
    from tidy3d.components.autograd.path_utils import AutogradRoute
    from tidy3d.components.autograd.types import AutogradFieldMap, TracedFloat
    from tidy3d.components.time_modulation import ModulationSpec
    from tidy3d.components.transformation import RotationType
    from tidy3d.components.types import (
        Ax,
        Axis,
        Bound,
        PermittivityComponent,
    )

    from .medium_types import IsotropicUniformMediumType


from .base import (
    AbstractMedium,
    ensure_freq_in_range,
)
from .isotropic import Medium, PECMedium, PMCMedium
from .lossy_metal import LossyMetalMedium


class AnisotropicMedium(AbstractMedium):
    """Diagonally anisotropic medium.

    Notes
    -----

        Only diagonal anisotropy is currently supported.

    Example
    -------
    >>> medium_xx = Medium(permittivity=4.0)
    >>> medium_yy = Medium(permittivity=4.1)
    >>> medium_zz = Medium(permittivity=3.9)
    >>> anisotropic_dielectric = AnisotropicMedium(xx=medium_xx, yy=medium_yy, zz=medium_zz)

    See Also
    --------

    :class:`CustomAnisotropicMedium`
        Diagonally anisotropic medium with spatially varying permittivity in each component.

    :class:`FullyAnisotropicMedium`
        Fully anisotropic medium including all 9 components of the permittivity and conductivity tensors.

    **Notebooks**
        * `Broadband polarizer assisted by anisotropic metamaterial <../../notebooks/SWGBroadbandPolarizer.html>`_
        * `Thin film lithium niobate adiabatic waveguide coupler <../../notebooks/AdiabaticCouplerLN.html>`_
    """

    xx: IsotropicUniformMediumType = Field(
        title="XX Component",
        description="Medium describing the xx-component of the diagonal permittivity tensor.",
        discriminator=TYPE_TAG_STR,
    )

    yy: IsotropicUniformMediumType = Field(
        title="YY Component",
        description="Medium describing the yy-component of the diagonal permittivity tensor.",
        discriminator=TYPE_TAG_STR,
    )

    zz: IsotropicUniformMediumType = Field(
        title="ZZ Component",
        description="Medium describing the zz-component of the diagonal permittivity tensor.",
        discriminator=TYPE_TAG_STR,
    )

    allow_gain: bool | None = Field(
        None,
        title="Allow gain medium",
        description="This field is ignored. Please set ``allow_gain`` in each component",
    )

    @field_validator("modulation_spec")
    @classmethod
    def _validate_modulation_spec(cls, val: ModulationSpec | None) -> ModulationSpec | None:
        """Check compatibility with modulation_spec."""
        if val is not None:
            raise ValidationError(
                f"A 'modulation_spec' of class {type(val)} is not "
                f"currently supported for medium class {cls.__name__}. "
                "Please add modulation to each component."
            )
        return val

    @field_validator("xx", "yy", "zz")
    @classmethod
    def _no_surface_lossy_metal_component(
        cls, val: IsotropicUniformMediumType, info: FieldValidationInfo
    ) -> IsotropicUniformMediumType:
        """A non-penetrable lossy metal has no anisotropic-component formulation."""
        if isinstance(val, LossyMetalMedium) and not val.penetrable:
            raise ValidationError(
                f"The '{info.field_name}' component is a non-penetrable 'LossyMetalMedium', "
                "which is not supported as a component of an 'AnisotropicMedium'. Set "
                "'penetrable=True' to use it as a regular conductive medium."
            )
        return val

    @model_validator(mode="after")
    def _run_after_validators(self) -> Self:
        """Run post-init validations in an explicit, dependency-aware order."""
        super()._run_after_validators()
        self._ignored_fields()
        return self

    def _ignored_fields(self) -> Self:
        """The field is ignored."""
        if self.xx is not None and self.allow_gain is not None:
            log.warning(
                "The field 'allow_gain' is ignored. Please set 'allow_gain' in each component."
            )
        return self

    @cached_property
    def components(self) -> dict[str, Medium]:
        """Dictionary of diagonal medium components."""
        return {"xx": self.xx, "yy": self.yy, "zz": self.zz}

    @cached_property
    def is_time_modulated(self) -> bool:
        """Whether any component of the medium is time modulated."""
        return any(mat.is_time_modulated for mat in self.components.values())

    @cached_property
    def n_cfl(self) -> float:
        """This property computes the index of refraction related to CFL condition, so that
        the FDTD with this medium is stable when the time step size that doesn't take
        material factor into account is multiplied by ``n_cfl``.

        For this medium, it takes the minimal of ``n_clf`` in all components.
        """
        return min(mat_component.n_cfl for mat_component in self.components.values())

    @ensure_freq_in_range
    def eps_model(self, frequency: float) -> complex:
        """Complex-valued permittivity as a function of frequency."""
        eps_diag = self.eps_diagonal(frequency)
        return (eps_diag[0] + eps_diag[1] + eps_diag[2]) / 3

    @ensure_freq_in_range
    def eps_diagonal(self, frequency: float) -> tuple[complex, complex, complex]:
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

    def _eps_plot(
        self, frequency: float, eps_component: PermittivityComponent | None = None
    ) -> float:
        """Returns real part of epsilon for plotting. A specific component of the epsilon tensor can
        be selected for anisotropic medium.

        Parameters
        ----------
        frequency : float
        eps_component : PermittivityComponent

        Returns
        -------
        float
            Element ``eps_component`` of the relative permittivity tensor evaluated at ``frequency``.
        """
        if eps_component is None:
            # return the average of the diag
            return self.eps_model(frequency).real
        if eps_component in ["xx", "yy", "zz"]:
            # return the requested diagonal component
            comp2indx = {"x": 0, "y": 1, "z": 2}
            return self.eps_comp(
                row=comp2indx[eps_component[0]],
                col=comp2indx[eps_component[1]],
                frequency=frequency,
            ).real
        raise ValueError(
            f"Plotting component '{eps_component}' of a diagonally-anisotropic permittivity tensor is not supported."
        )

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
    def elements(self) -> dict[str, IsotropicUniformMediumType]:
        """The diagonal elements of the medium as a dictionary."""
        return {"xx": self.xx, "yy": self.yy, "zz": self.zz}

    @cached_property
    def is_pec(self) -> bool:
        """Whether the medium is a PEC."""
        return any(self.is_comp_pec(i) for i in range(3))

    @cached_property
    def is_pmc(self) -> bool:
        """Whether the medium is a PMC."""
        return any(self.is_comp_pmc(i) for i in range(3))

    def is_comp_pec(self, comp: Axis) -> bool:
        """Whether the medium is a PEC."""
        return isinstance(self.components[["xx", "yy", "zz"][comp]], PECMedium)

    def is_comp_pmc(self, comp: Axis) -> bool:
        """Whether the medium is a PMC."""
        return isinstance(self.components[["xx", "yy", "zz"][comp]], PMCMedium)

    def sel_inside(self, bounds: Bound) -> Self:
        """Return a new medium that contains the minimal amount data necessary to cover
        a spatial region defined by ``bounds``.


        Parameters
        ----------
        bounds : tuple[float, float, float], tuple[float, float float]
            Min and max bounds packaged as ``(minx, miny, minz), (maxx, maxy, maxz)``.

        Returns
        -------
        AnisotropicMedium
            AnisotropicMedium with reduced data.
        """

        new_comps = [comp.sel_inside(bounds) for comp in [self.xx, self.yy, self.zz]]

        return self.updated_copy(**dict(zip(["xx", "yy", "zz"], new_comps)))

    # --- shared autograd helpers ---
    @staticmethod
    def _component_derivative_info(
        derivative_info: DerivativeInfo, component: str
    ) -> DerivativeInfo | None:
        """Build ``DerivativeInfo`` filtered to a single anisotropic component."""

        component_paths = [
            tuple(path[1:]) for path in derivative_info.paths if path and path[0] == component
        ]
        if not component_paths:
            return None

        axis = component[0]  # f.e. xx -> x
        projected_E = derivative_info.project_der_map_to_axis(axis, "E")
        projected_D = derivative_info.project_der_map_to_axis(axis, "D")
        return derivative_info.updated_copy(
            paths=component_paths, E_der_map=projected_E, D_der_map=projected_D
        )

    def _resolve_autograd_route(self, field_path: tuple[Any, ...]) -> AutogradRoute:
        """Resolve and validate one traced AnisotropicMedium path for adjoint routing."""
        components = self.components
        return resolve_delegated_autograd_route(
            parameter_kind="medium",
            owner_kind="medium type",
            owner_name=type(self).__name__,
            field_path=field_path,
            delegates=components,
            supported_parameters=tuple(f"{component}.<parameter>" for component in components),
        )

    def _compute_derivatives(self, derivative_info: DerivativeInfo) -> AutogradFieldMap:
        """Delegate derivatives for each diagonal component of an anisotropic medium."""

        vjps: AutogradFieldMap = {}
        components = self.components
        for comp_name, component in components.items():
            comp_info = self._component_derivative_info(
                derivative_info=derivative_info, component=comp_name
            )
            if comp_info is None:
                continue
            comp_vjps = component._compute_derivatives(comp_info)
            for sub_path, value in comp_vjps.items():
                vjps[(comp_name, *sub_path)] = value

        return vjps


class AnisotropicMediumFromMedium2D(AnisotropicMedium):
    """The same as ``AnisotropicMedium``, but converted from Medium2D.
    (This class is for internal use only)
    """


class FullyAnisotropicMedium(AbstractMedium):
    """Fully anisotropic medium including all 9 components of the permittivity and conductivity
    tensors.

    Notes
    -----

        Provided permittivity tensor and the symmetric part of the conductivity tensor must
        have coinciding main directions. A non-symmetric conductivity tensor can be used to model
        magneto-optic effects. Note that dispersive properties and subpixel averaging are currently not
        supported for fully anisotropic materials.

    Note
    ----

        Simulations involving fully anisotropic materials are computationally more intensive, thus,
        they take longer time to complete. This increase strongly depends on the filling fraction of
        the simulation domain by fully anisotropic materials, varying approximately in the range from
        1.5 to 5. The cost of running a simulation is adjusted correspondingly.

    Example
    -------
    >>> perm = [[2, 0, 0], [0, 1, 0], [0, 0, 3]]
    >>> cond = [[0.1, 0, 0], [0, 0, 0], [0, 0, 0]]
    >>> anisotropic_dielectric = FullyAnisotropicMedium(permittivity=perm, conductivity=cond)

    See Also
    --------

    :class:`CustomAnisotropicMedium`
        Diagonally anisotropic medium with spatially varying permittivity in each component.

    :class:`AnisotropicMedium`
        Diagonally anisotropic medium.

    **Notebooks**
        * `Broadband polarizer assisted by anisotropic metamaterial <../../notebooks/SWGBroadbandPolarizer.html>`_
        * `Thin film lithium niobate adiabatic waveguide coupler <../../notebooks/AdiabaticCouplerLN.html>`_
        * `Defining fully anisotropic materials <../../notebooks/FullyAnisotropic.html>`_
    """

    @cached_property
    def is_fully_anisotropic(self) -> bool:
        """Whether the medium is fully anisotropic."""
        return True

    permittivity: TensorReal = Field(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        title="Permittivity",
        description="Relative permittivity tensor.",
        json_schema_extra={"units": PERMITTIVITY},
    )

    conductivity: TensorReal = Field(
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        title="Conductivity",
        description="Electric conductivity tensor. Defined such that the imaginary part "
        "of the complex permittivity at angular frequency omega is given by conductivity/omega.",
        json_schema_extra={"units": CONDUCTIVITY},
    )

    @field_validator("modulation_spec")
    @classmethod
    def _validate_modulation_spec(cls, val: ModulationSpec | None) -> ModulationSpec | None:
        """Check compatibility with modulation_spec."""
        if val is not None:
            raise ValidationError(
                f"A 'modulation_spec' of class {type(val)} is not "
                f"currently supported for medium class {cls.__name__}."
            )
        return val

    @field_validator("permittivity")
    @classmethod
    def permittivity_spd_and_ge_one(cls, val: TracedFloat) -> TracedFloat:
        """Check that provided permittivity tensor is symmetric positive definite
        with eigenvalues >= 1.
        """

        if not np.allclose(val, np.transpose(val), atol=fp_eps):
            raise ValidationError("Provided permittivity tensor is not symmetric.")

        if np.any(np.linalg.eigvals(val) < 1 - fp_eps):
            raise ValidationError("Main diagonal of provided permittivity tensor is not >= 1.")

        return val

    @model_validator(mode="after")
    def _run_after_validators(self) -> Self:
        """Run post-init validations in an explicit, dependency-aware order."""
        super()._run_after_validators()
        self._conductivity_commutes()
        self._passivity_validation()
        return self

    def _conductivity_commutes(self) -> Self:
        """Check that the symmetric part of conductivity tensor commutes with permittivity tensor
        (that is, simultaneously diagonalizable).
        """

        val = self.conductivity
        perm = self.permittivity
        cond_sym = 0.5 * (val + val.T)
        comm_diff = np.abs(np.matmul(perm, cond_sym) - np.matmul(cond_sym, perm))

        if not np.allclose(comm_diff, 0, atol=fp_eps):
            self._raise_validation_error_at_loc(
                ValidationError(
                    "Main directions of conductivity and permittivity tensor do not coincide."
                ),
                "conductivity",
            )

        return self

    def _passivity_validation(self) -> Self:
        """Assert passive medium if ``allow_gain`` is False."""
        val = self.conductivity
        if self.allow_gain:
            return self

        cond_sym = 0.5 * (val + val.T)
        if np.any(np.linalg.eigvals(cond_sym) < -fp_eps):
            self._raise_validation_error_at_loc(
                ValidationError(
                    "For passive medium, main diagonal of provided conductivity tensor "
                    "must be non-negative. "
                    "To simulate a gain medium, please set 'allow_gain=True'. "
                    "Caution: simulations with a gain medium are unstable, and are likely to diverge."
                ),
                "conductivity",
            )
        return self

    @classmethod
    def from_diagonal(cls, xx: Medium, yy: Medium, zz: Medium, rotation: RotationType) -> Self:
        """Construct a fully anisotropic medium by rotating a diagonally anisotropic medium.

        Parameters
        ----------
        xx : :class:`.Medium`
            Medium describing the xx-component of the diagonal permittivity tensor.
        yy : :class:`.Medium`
            Medium describing the yy-component of the diagonal permittivity tensor.
        zz : :class:`.Medium`
            Medium describing the zz-component of the diagonal permittivity tensor.
        rotation : Union[:class:`.RotationAroundAxis`]
                Rotation applied to diagonal permittivity tensor.

        Returns
        -------
        :class:`FullyAnisotropicMedium`
            Resulting fully anisotropic medium.
        """

        if any(comp.nonlinear_spec is not None for comp in [xx, yy, zz]):
            raise ValidationError(
                "Nonlinearities are not currently supported for the components "
                "of a fully anisotropic medium."
            )

        if any(comp.modulation_spec is not None for comp in [xx, yy, zz]):
            raise ValidationError(
                "Modulation is not currently supported for the components "
                "of a fully anisotropic medium."
            )

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
    ) -> tuple[tuple[float, float, float], tuple[float, float, float], TensorReal]:
        """Main components of permittivity and conductivity tensors and their directions."""

        perm_diag, vecs = np.linalg.eig(self.permittivity)
        cond_diag = np.diag(np.matmul(np.transpose(vecs), np.matmul(self.conductivity, vecs)))

        return (perm_diag, cond_diag, vecs)

    @ensure_freq_in_range
    def eps_model(self, frequency: float) -> complex:
        """Complex-valued permittivity as a function of frequency."""
        perm_diag, cond_diag, _ = self.eps_sigma_diag

        if not np.isscalar(frequency):
            perm_diag = perm_diag[:, None]
            cond_diag = cond_diag[:, None]
        eps_diag = AbstractMedium.eps_sigma_to_eps_complex(perm_diag, cond_diag, frequency)
        return np.mean(eps_diag, axis=0)

    @ensure_freq_in_range
    def eps_diagonal(self, frequency: float) -> tuple[complex, complex, complex]:
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

    def _eps_plot(
        self, frequency: float, eps_component: PermittivityComponent | None = None
    ) -> float:
        """Returns real part of epsilon for plotting. A specific component of the epsilon tensor can
        be selected for anisotropic medium.

        Parameters
        ----------
        frequency : float
        eps_component : PermittivityComponent

        Returns
        -------
        float
            Element ``eps_component`` of the relative permittivity tensor evaluated at ``frequency``.
        """
        if eps_component is None:
            # return the average of the diag
            return self.eps_model(frequency).real

        # return the requested component
        comp2indx = {"x": 0, "y": 1, "z": 2}
        return self.eps_comp(
            row=comp2indx[eps_component[0]], col=comp2indx[eps_component[1]], frequency=frequency
        ).real

    @cached_property
    def n_cfl(self) -> float:
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
