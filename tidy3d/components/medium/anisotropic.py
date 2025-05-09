from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import autograd.numpy as np
import pydantic.v1 as pd

from tidy3d.components.base import cached_property, skip_if_fields_missing
from tidy3d.components.data.utils import CustomSpatialDataType
from tidy3d.components.transformation import RotationType
from tidy3d.components.types import (
    TYPE_TAG_STR,
    Ax,
    Axis,
    Bound,
    InterpMethod,
    PermittivityComponent,
    TensorReal,
)
from tidy3d.components.viz import add_ax_if_none
from tidy3d.constants import CONDUCTIVITY, PERMITTIVITY, fp_eps
from tidy3d.exceptions import SetupError, ValidationError
from tidy3d.log import log

from .base import AbstractCustomMedium, AbstractMedium
from .dispersionless import CustomMedium, Medium, PECMedium
from .utils import ensure_freq_in_range


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

    xx: IsotropicUniformMediumType = pd.Field(
        ...,
        title="XX Component",
        description="Medium describing the xx-component of the diagonal permittivity tensor.",
        discriminator=TYPE_TAG_STR,
    )

    yy: IsotropicUniformMediumType = pd.Field(
        ...,
        title="YY Component",
        description="Medium describing the yy-component of the diagonal permittivity tensor.",
        discriminator=TYPE_TAG_STR,
    )

    zz: IsotropicUniformMediumType = pd.Field(
        ...,
        title="ZZ Component",
        description="Medium describing the zz-component of the diagonal permittivity tensor.",
        discriminator=TYPE_TAG_STR,
    )

    allow_gain: bool = pd.Field(
        None,
        title="Allow gain medium",
        description="This field is ignored. Please set ``allow_gain`` in each component",
    )

    @pd.validator("modulation_spec", always=True)
    def _validate_modulation_spec(cls, val):
        """Check compatibility with modulation_spec."""
        if val is not None:
            raise ValidationError(
                f"A 'modulation_spec' of class {type(val)} is not "
                f"currently supported for medium class {cls}. "
                "Please add modulation to each component."
            )
        return val

    @pd.root_validator(pre=True)
    def _ignored_fields(cls, values):
        """The field is ignored."""
        if values.get("xx") is not None and values.get("allow_gain") is not None:
            log.warning(
                "The field 'allow_gain' is ignored. Please set 'allow_gain' in each component."
            )
        return values

    @cached_property
    def components(self) -> Dict[str, Medium]:
        """Dictionary of diagonal medium components."""
        return dict(xx=self.xx, yy=self.yy, zz=self.zz)

    @cached_property
    def is_time_modulated(self) -> bool:
        """Whether any component of the medium is time modulated."""
        return any(mat.is_time_modulated for mat in self.components.values())

    @cached_property
    def n_cfl(self):
        """This property computes the index of refraction related to CFL condition, so that
        the FDTD with this medium is stable when the time step size that doesn't take
        material factor into account is multiplied by ``n_cfl``.

        For this medium, it takes the minimal of ``n_clf`` in all components.
        """
        return min(mat_component.n_cfl for mat_component in self.components.values())

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

    def _eps_plot(
        self, frequency: float, eps_component: Optional[PermittivityComponent] = None
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
        elif eps_component in ["xx", "yy", "zz"]:
            # return the requested diagonal component
            comp2indx = {"x": 0, "y": 1, "z": 2}
            return self.eps_comp(
                row=comp2indx[eps_component[0]],
                col=comp2indx[eps_component[1]],
                frequency=frequency,
            ).real
        else:
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
    def elements(self) -> Dict[str, IsotropicUniformMediumType]:
        """The diagonal elements of the medium as a dictionary."""
        return dict(xx=self.xx, yy=self.yy, zz=self.zz)

    @cached_property
    def is_pec(self):
        """Whether the medium is a PEC."""
        return any(self.is_comp_pec(i) for i in range(3))

    def is_comp_pec(self, comp: Axis):
        """Whether the medium is a PEC."""
        return isinstance(self.components[["xx", "yy", "zz"][comp]], PECMedium)

    def sel_inside(self, bounds: Bound):
        """Return a new medium that contains the minimal amount data necessary to cover
        a spatial region defined by ``bounds``.


        Parameters
        ----------
        bounds : Tuple[float, float, float], Tuple[float, float float]
            Min and max bounds packaged as ``(minx, miny, minz), (maxx, maxy, maxz)``.

        Returns
        -------
        AnisotropicMedium
            AnisotropicMedium with reduced data.
        """

        new_comps = [comp.sel_inside(bounds) for comp in [self.xx, self.yy, self.zz]]

        return self.updated_copy(**dict(zip(["xx", "yy", "zz"], new_comps)))


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

    @pd.validator("modulation_spec", always=True)
    def _validate_modulation_spec(cls, val):
        """Check compatibility with modulation_spec."""
        if val is not None:
            raise ValidationError(
                f"A 'modulation_spec' of class {type(val)} is not "
                f"currently supported for medium class {cls}."
            )
        return val

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
    @skip_if_fields_missing(["permittivity"])
    def conductivity_commutes(cls, val, values):
        """Check that the symmetric part of conductivity tensor commutes with permittivity tensor
        (that is, simultaneously diagonalizable).
        """

        perm = values.get("permittivity")
        cond_sym = 0.5 * (val + val.T)
        comm_diff = np.abs(np.matmul(perm, cond_sym) - np.matmul(cond_sym, perm))

        if not np.allclose(comm_diff, 0, atol=fp_eps):
            raise ValidationError(
                "Main directions of conductivity and permittivity tensor do not coincide."
            )

        return val

    @pd.validator("conductivity", always=True)
    @skip_if_fields_missing(["allow_gain"])
    def _passivity_validation(cls, val, values):
        """Assert passive medium if ``allow_gain`` is False."""
        if values.get("allow_gain"):
            return val

        cond_sym = 0.5 * (val + val.T)
        if np.any(np.linalg.eigvals(cond_sym) < -fp_eps):
            raise ValidationError(
                "For passive medium, main diagonal of provided conductivity tensor "
                "must be non-negative. "
                "To simulate a gain medium, please set 'allow_gain=True'. "
                "Caution: simulations with a gain medium are unstable, and are likely to diverge."
            )
        return val

    @classmethod
    def from_diagonal(cls, xx: Medium, yy: Medium, zz: Medium, rotation: RotationType):
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
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float], TensorReal]:
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

    def _eps_plot(
        self, frequency: float, eps_component: Optional[PermittivityComponent] = None
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


class CustomAnisotropicMedium(AbstractCustomMedium, AnisotropicMedium):
    """Diagonally anisotropic medium with spatially varying permittivity in each component.

    Note
    ----
        Only diagonal anisotropy is currently supported.

    Example
    -------
    >>> Nx, Ny, Nz = 10, 9, 8
    >>> x = np.linspace(-1, 1, Nx)
    >>> y = np.linspace(-1, 1, Ny)
    >>> z = np.linspace(-1, 1, Nz)
    >>> coords = dict(x=x, y=y, z=z)
    >>> permittivity= SpatialDataArray(np.ones((Nx, Ny, Nz)), coords=coords)
    >>> conductivity= SpatialDataArray(np.ones((Nx, Ny, Nz)), coords=coords)
    >>> medium_xx = CustomMedium(permittivity=permittivity, conductivity=conductivity)
    >>> medium_yy = CustomMedium(permittivity=permittivity, conductivity=conductivity)
    >>> d_epsilon = SpatialDataArray(np.random.random((Nx, Ny, Nz)), coords=coords)
    >>> f = SpatialDataArray(1+np.random.random((Nx, Ny, Nz)), coords=coords)
    >>> delta = SpatialDataArray(np.random.random((Nx, Ny, Nz)), coords=coords)
    >>> medium_zz = CustomLorentz(eps_inf=permittivity, coeffs=[(d_epsilon,f,delta),])
    >>> anisotropic_dielectric = CustomAnisotropicMedium(xx=medium_xx, yy=medium_yy, zz=medium_zz)

    See Also
    --------

    :class:`AnisotropicMedium`
        Diagonally anisotropic medium.

    **Notebooks**
        * `Broadband polarizer assisted by anisotropic metamaterial <../../notebooks/SWGBroadbandPolarizer.html>`_
        * `Thin film lithium niobate adiabatic waveguide coupler <../../notebooks/AdiabaticCouplerLN.html>`_
        * `Defining fully anisotropic materials <../../notebooks/FullyAnisotropic.html>`_
    """

    xx: Union[IsotropicCustomMediumType, CustomMedium] = pd.Field(
        ...,
        title="XX Component",
        description="Medium describing the xx-component of the diagonal permittivity tensor.",
        discriminator=TYPE_TAG_STR,
    )

    yy: Union[IsotropicCustomMediumType, CustomMedium] = pd.Field(
        ...,
        title="YY Component",
        description="Medium describing the yy-component of the diagonal permittivity tensor.",
        discriminator=TYPE_TAG_STR,
    )

    zz: Union[IsotropicCustomMediumType, CustomMedium] = pd.Field(
        ...,
        title="ZZ Component",
        description="Medium describing the zz-component of the diagonal permittivity tensor.",
        discriminator=TYPE_TAG_STR,
    )

    interp_method: Optional[InterpMethod] = pd.Field(
        None,
        title="Interpolation method",
        description="When the value is 'None', each component will follow its own "
        "interpolation method. When the value is other than 'None', the interpolation "
        "method specified by this field will override the one in each component.",
    )

    allow_gain: bool = pd.Field(
        None,
        title="Allow gain medium",
        description="This field is ignored. Please set ``allow_gain`` in each component",
    )

    subpixel: bool = pd.Field(
        None,
        title="Subpixel averaging",
        description="This field is ignored. Please set ``subpixel`` in each component",
    )

    @pd.validator("xx", always=True)
    def _isotropic_xx(cls, val):
        """If it's `CustomMedium`, make sure it's isotropic."""
        if isinstance(val, CustomMedium) and not val.is_isotropic:
            raise SetupError("The xx-component medium type is not isotropic.")
        return val

    @pd.validator("yy", always=True)
    def _isotropic_yy(cls, val):
        """If it's `CustomMedium`, make sure it's isotropic."""
        if isinstance(val, CustomMedium) and not val.is_isotropic:
            raise SetupError("The yy-component medium type is not isotropic.")
        return val

    @pd.validator("zz", always=True)
    def _isotropic_zz(cls, val):
        """If it's `CustomMedium`, make sure it's isotropic."""
        if isinstance(val, CustomMedium) and not val.is_isotropic:
            raise SetupError("The zz-component medium type is not isotropic.")
        return val

    @pd.root_validator(pre=True)
    def _ignored_fields(cls, values):
        """The field is ignored."""
        if values.get("xx") is not None:
            if values.get("allow_gain") is not None:
                log.warning(
                    "The field 'allow_gain' is ignored. Please set 'allow_gain' in each component."
                )
            if values.get("subpixel") is not None:
                log.warning(
                    "The field 'subpixel' is ignored. Please set 'subpixel' in each component."
                )
        return values

    @cached_property
    def is_spatially_uniform(self) -> bool:
        """Whether the medium is spatially uniform."""
        return any(comp.is_spatially_uniform for comp in self.components.values())

    @cached_property
    def n_cfl(self):
        """This property computes the index of refraction related to CFL condition, so that
        the FDTD with this medium is stable when the time step size that doesn't take
        material factor into account is multiplied by ``n_cfl``.

        For this medium, it takes the minimal of ``n_clf`` in all components.
        """
        return min(mat_component.n_cfl for mat_component in self.components.values())

    @cached_property
    def is_isotropic(self):
        """Whether the medium is isotropic."""
        return False

    def _interp_method(self, comp: Axis) -> InterpMethod:
        """Interpolation method applied to comp."""
        # override `interp_method` in components if self.interp_method is not None
        if self.interp_method is not None:
            return self.interp_method
        # use component's interp_method
        comp_map = ["xx", "yy", "zz"]
        return self.components[comp_map[comp]].interp_method

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
                :class:`.TetrahedralGridDataset`,
            ],
            Union[
                :class:`.SpatialDataArray`,
                :class:`.TriangularGridDataset`,
                :class:`.TetrahedralGridDataset`,
            ],
            Union[
                :class:`.SpatialDataArray`,
                :class:`.TriangularGridDataset`,
                :class:`.TetrahedralGridDataset`,
            ],
        ]
            The permittivity evaluated at ``frequency``.
        """
        return tuple(
            mat_component.eps_dataarray_freq(frequency)[ind]
            for ind, mat_component in enumerate(self.components.values())
        )

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
        comps = ["xx", "yy", "zz"]
        if eps_component in comps:
            # Return the bounds of a specific component
            eps_dataarray = self.eps_dataarray_freq(frequency)
            eps = self._get_real_vals(eps_dataarray[comps.index(eps_component)])
            return (np.min(eps), np.max(eps))
        elif eps_component is None:
            # Returns the bounds across all components
            return super()._eps_bounds(frequency=frequency)
        else:
            raise ValueError(
                f"Plotting component '{eps_component}' of a diagonally-anisotropic permittivity tensor is not supported."
            )

    def _sel_custom_data_inside(self, bounds: Bound):
        return self


class CustomAnisotropicMediumInternal(CustomAnisotropicMedium):
    """Diagonally anisotropic medium with spatially varying permittivity in each component.

    Notes
    -----

        Only diagonal anisotropy is currently supported.

    Example
    -------
    >>> Nx, Ny, Nz = 10, 9, 8
    >>> X = np.linspace(-1, 1, Nx)
    >>> Y = np.linspace(-1, 1, Ny)
    >>> Z = np.linspace(-1, 1, Nz)
    >>> coords = dict(x=X, y=Y, z=Z)
    >>> permittivity= SpatialDataArray(np.ones((Nx, Ny, Nz)), coords=coords)
    >>> conductivity= SpatialDataArray(np.ones((Nx, Ny, Nz)), coords=coords)
    >>> medium_xx = CustomMedium(permittivity=permittivity, conductivity=conductivity)
    >>> medium_yy = CustomMedium(permittivity=permittivity, conductivity=conductivity)
    >>> d_epsilon = SpatialDataArray(np.random.random((Nx, Ny, Nz)), coords=coords)
    >>> f = SpatialDataArray(1+np.random.random((Nx, Ny, Nz)), coords=coords)
    >>> delta = SpatialDataArray(np.random.random((Nx, Ny, Nz)), coords=coords)
    >>> medium_zz = CustomLorentz(eps_inf=permittivity, coeffs=[(d_epsilon,f,delta),])
    >>> anisotropic_dielectric = CustomAnisotropicMedium(xx=medium_xx, yy=medium_yy, zz=medium_zz)
    """

    xx: Union[IsotropicCustomMediumInternalType, CustomMedium] = pd.Field(
        ...,
        title="XX Component",
        description="Medium describing the xx-component of the diagonal permittivity tensor.",
        discriminator=TYPE_TAG_STR,
    )

    yy: Union[IsotropicCustomMediumInternalType, CustomMedium] = pd.Field(
        ...,
        title="YY Component",
        description="Medium describing the yy-component of the diagonal permittivity tensor.",
        discriminator=TYPE_TAG_STR,
    )

    zz: Union[IsotropicCustomMediumInternalType, CustomMedium] = pd.Field(
        ...,
        title="ZZ Component",
        description="Medium describing the zz-component of the diagonal permittivity tensor.",
        discriminator=TYPE_TAG_STR,
    )
