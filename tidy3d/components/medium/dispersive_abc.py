from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Tuple

import autograd.numpy as np
import pydantic.v1 as pd

from tidy3d.components.base import cached_property, skip_if_fields_missing
from tidy3d.components.data.data_array import SpatialDataArray
from tidy3d.components.data.utils import _get_numpy_array
from tidy3d.exceptions import ValidationError
from tidy3d.log import log

from .base import AbstractCustomMedium, AbstractMedium


class DispersiveMedium(AbstractMedium, ABC):
    """
    A Medium with dispersion: field propagation characteristics depend on frequency.

    Notes
    -----

        In dispersive mediums, the displacement field :math:`D(t)` depends on the previous electric field :math:`E(
        t')` and time-dependent permittivity :math:`\\epsilon` changes.

        .. math::

            D(t) = \\int \\epsilon(t - t') E(t') \\delta t'

        Dispersive mediums can be defined in three ways:

        - Imported from our `material library <../material_library.html>`_.
        - Defined directly by specifying the parameters in the `various supplied dispersive models <../mediums.html>`_.
        - Fitted to optical n-k data using the `dispersion fitting tool plugin <../plugins/dispersion.html>`_.

        It is important to keep in mind that dispersive materials are inevitably slower to simulate than their
        dispersion-less counterparts, with complexity increasing with the number of poles included in the dispersion
        model. For simulations with a narrow range of frequencies of interest, it may sometimes be faster to define
        the material through its real and imaginary refractive index at the center frequency.


    See Also
    --------

    :class:`CustomPoleResidue`:
        A spatially varying dispersive medium described by the pole-residue pair model.

    **Notebooks**
        * `Fitting dispersive material models <../../notebooks/Fitting.html>`_

    **Lectures**
        * `Modeling dispersive material in FDTD <https://www.flexcompute.com/fdtd101/Lecture-5-Modeling-dispersive-material-in-FDTD/>`_
    """

    @staticmethod
    def _permittivity_modulation_validation():
        """Assert modulated permittivity cannot be <= 0 at any time."""

        @pd.validator("eps_inf", allow_reuse=True, always=True)
        @skip_if_fields_missing(["modulation_spec"])
        def _validate_permittivity_modulation(cls, val, values):
            """Assert modulated permittivity cannot be <= 0."""
            modulation = values.get("modulation_spec")
            if modulation is None or modulation.permittivity is None:
                return val

            min_eps_inf = np.min(_get_numpy_array(val))
            if min_eps_inf - modulation.permittivity.max_modulation <= 0:
                raise ValidationError(
                    "The minimum permittivity value with modulation applied was found to be negative."
                )
            return val

        return _validate_permittivity_modulation

    @staticmethod
    def _conductivity_modulation_validation():
        """Assert passive medium at any time if not ``allow_gain``."""

        @pd.validator("modulation_spec", allow_reuse=True, always=True)
        @skip_if_fields_missing(["allow_gain"])
        def _validate_conductivity_modulation(cls, val, values):
            """With conductivity modulation, the medium can exhibit gain during the cycle.
            So `allow_gain` must be True when the conductivity is modulated.
            """
            if val is None or val.conductivity is None:
                return val

            if not values.get("allow_gain"):
                raise ValidationError(
                    "For passive medium, 'conductivity' must be non-negative at any time. "
                    "With conductivity modulation, this medium can sometimes be active. "
                    "Please set 'allow_gain=True'. "
                    "Caution: simulations with a gain medium are unstable, and are likely to diverge."
                )
            return val

        return _validate_conductivity_modulation

    @abstractmethod
    def _pole_residue_dict(self) -> Dict:
        """Dict representation of Medium as a pole-residue model."""

    @cached_property
    def pole_residue(self):
        """Representation of Medium as a pole-residue model."""
        from .dispersive import PoleResidue

        return PoleResidue(**self._pole_residue_dict(), allow_gain=self.allow_gain)

    @cached_property
    def n_cfl(self):
        """This property computes the index of refraction related to CFL condition, so that
        the FDTD with this medium is stable when the time step size that doesn't take
        material factor into account is multiplied by ``n_cfl``.

        For PoleResidue model, it equals ``sqrt(eps_inf)``
        [https://ieeexplore.ieee.org/document/9082879].
        """
        permittivity = self.pole_residue.eps_inf
        if self.modulation_spec is not None and self.modulation_spec.permittivity is not None:
            permittivity -= self.modulation_spec.permittivity.max_modulation
        n, _ = self.eps_complex_to_nk(permittivity)
        return n

    @staticmethod
    def tuple_to_complex(value: Tuple[float, float]) -> complex:
        """Convert a tuple of real and imaginary parts to complex number."""

        val_r, val_i = value
        return val_r + 1j * val_i

    @staticmethod
    def complex_to_tuple(value: complex) -> Tuple[float, float]:
        """Convert a complex number to a tuple of real and imaginary parts."""

        return (value.real, value.imag)


class CustomDispersiveMedium(AbstractCustomMedium, DispersiveMedium, ABC):
    """A spatially varying dispersive medium."""

    @cached_property
    def n_cfl(self):
        """This property computes the index of refraction related to CFL condition, so that
        the FDTD with this medium is stable when the time step size that doesn't take
        material factor into account is multiplied by ``n_cfl``.

        For PoleResidue model, it equals ``sqrt(eps_inf)``
        [https://ieeexplore.ieee.org/document/9082879].
        """
        permittivity = np.min(_get_numpy_array(self.pole_residue.eps_inf))
        if self.modulation_spec is not None and self.modulation_spec.permittivity is not None:
            permittivity -= self.modulation_spec.permittivity.max_modulation
        n, _ = self.eps_complex_to_nk(permittivity)
        return n

    @cached_property
    def is_isotropic(self):
        """Whether the medium is isotropic."""
        return True

    @cached_property
    def pole_residue(self):
        """Representation of Medium as a pole-residue model."""
        from .dispersive import CustomPoleResidue

        return CustomPoleResidue(
            **self._pole_residue_dict(),
            interp_method=self.interp_method,
            allow_gain=self.allow_gain,
            subpixel=self.subpixel,
        )

    @staticmethod
    def _warn_if_data_none(nested_tuple_field: str):
        """Warn if any of `eps_inf` and nested_tuple_field are not loaded,
        and return a vacuum with eps_inf = 1.
        """

        @pd.root_validator(pre=True, allow_reuse=True)
        def _warn_if_none(cls, values):
            """Warn if any of `eps_inf` and nested_tuple_field are not load."""
            eps_inf = values.get("eps_inf")
            coeffs = values.get(nested_tuple_field)
            fail_load = False

            if AbstractCustomMedium._not_loaded(eps_inf):
                log.warning("Loading 'eps_inf' without data; constructing a vacuum medium instead.")
                fail_load = True
            for coeff in coeffs:
                if fail_load:
                    break
                for coeff_i in coeff:
                    if AbstractCustomMedium._not_loaded(coeff_i):
                        log.warning(
                            f"Loading '{nested_tuple_field}' without data; "
                            "constructing a vacuum medium instead."
                        )
                        fail_load = True
                        break

            if fail_load and eps_inf is None:
                return {nested_tuple_field: ()}
            if fail_load:
                eps_inf = SpatialDataArray(np.ones((1, 1, 1)), coords=dict(x=[0], y=[0], z=[0]))
                return {"eps_inf": eps_inf, nested_tuple_field: ()}
            return values

        return _warn_if_none
