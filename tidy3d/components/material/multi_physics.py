from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import Field

from tidy3d.components.autograd.path_utils import format_traced_path
from tidy3d.components.base import Tidy3dBaseModel
from tidy3d.components.material.solver_types import (
    ChargeMediumType,
    HeatMediumType,
    OpticalMediumType,
)
from tidy3d.components.types.base import TYPE_TAG_STR
from tidy3d.exceptions import AdjointError

if TYPE_CHECKING:
    from tidy3d.components.autograd.path_utils import AutogradRoute


class MultiPhysicsMedium(Tidy3dBaseModel):
    """
    Contains multiple multi-physical properties as defined for each solver medium.

    Examples
    --------
    For *silica* (:math:`SiO_2`):

    >>> import tidy3d as td
    >>> SiO2 = td.MultiPhysicsMedium(
    ...   optical=td.Medium(permittivity=3.9),
    ...   charge=td.ChargeInsulatorMedium(permittivity=3.9), # redefining permittivity
    ...   name="SiO2",
    ... )

    For a silicon ``MultiPhysicsMedium`` composed of an optical model
    from the material library and custom charge :class:`SemiconductorMedium`:

    >>> import tidy3d as td
    >>> default_multiphysics_Si = td.MultiPhysicsMedium(
    ...     optical=td.material_library['cSi']['Palik_LowLoss'],
    ...     charge=td.SemiconductorMedium(
    ...         N_c=td.ConstantEffectiveDOS(N=2.86e19),
    ...         N_v=td.ConstantEffectiveDOS(N=3.1e19),
    ...         E_g=td.ConstantEnergyBandGap(eg=1.11),
    ...         mobility_n=td.CaugheyThomasMobility(
    ...             mu_min=52.2,
    ...             mu=1471.0,
    ...             ref_N=9.68e16,
    ...             exp_N=0.68,
    ...             exp_1=-0.57,
    ...             exp_2=-2.33,
    ...             exp_3=2.4,
    ...             exp_4=-0.146,
    ...         ),
    ...         mobility_p=td.CaugheyThomasMobility(
    ...             mu_min=44.9,
    ...             mu=470.5,
    ...             ref_N=2.23e17,
    ...             exp_N=0.719,
    ...             exp_1=-0.57,
    ...             exp_2=-2.33,
    ...             exp_3=2.4,
    ...             exp_4=-0.146,
    ...         ),
    ...         R=[
    ...             td.ShockleyReedHallRecombination(
    ...                 tau_n=3.3e-6,
    ...                 tau_p=4e-6
    ...             ),
    ...             td.RadiativeRecombination(
    ...                 r_const=1.6e-14
    ...             ),
    ...             td.AugerRecombination(
    ...                 c_n=2.8e-31,
    ...                 c_p=9.9e-32
    ...             ),
    ...         ],
    ...         delta_E_g=td.SlotboomBandGapNarrowing(
    ...             v1=6.92 * 1e-3,
    ...             n2=1.3e17,
    ...             c2=0.5,
    ...             min_N=1e15,
    ...         ),
    ...         N_a=[td.ConstantDoping(concentration=1e15)],
    ...         N_d=[td.ConstantDoping(concentration=1e15)]
    ...     )
    ... )
    """

    name: str | None = Field(None, title="Name", description="Medium name")

    optical: OpticalMediumType | None = Field(
        None,
        title="Optical properties",
        description="Specifies optical properties.",
        discriminator=TYPE_TAG_STR,
    )

    # electrical: Optional[ElectricalMediumType] = Field(
    #     None,
    #     title="Electrical properties",
    #     description="Specifies electrical properties for RF simulations. This is currently not in use.",
    # )

    heat: HeatMediumType | None = Field(
        None,
        title="Heat properties",
        description="Specifies properties for Heat simulations.",
        discriminator=TYPE_TAG_STR,
    )

    charge: ChargeMediumType | None = Field(
        None,
        title="Charge properties",
        description="Specifies properties for Charge simulations.",
        discriminator=TYPE_TAG_STR,
    )

    def _resolve_autograd_route(self, field_path: tuple[Any, ...]) -> AutogradRoute:
        """Reject native autograd routes through multiphysics media."""
        parameter = format_traced_path(tuple(field_path))
        raise AdjointError(
            f"Automatic differentiation with respect to medium parameter '{parameter}' is not "
            f"supported for medium type '{type(self).__name__}'. Use an optical medium directly "
            "or provide a custom_vjp."
        )

    def __getattr__(self, name: str) -> Any:
        """
        Delegate attribute lookup to inner media or fail fast.

        Parameters
        ----------
        name : str
            The attribute that could not be found on the ``MultiPhysicsMedium`` itself.

        Returns
        -------
        Any
            * The attribute value obtained from a delegated sub-medium when
            ``name`` is listed in ``DELEGATED_ATTRIBUTES``.
            * ``None`` when ``name`` is explicitly ignored (e.g. ``"__deepcopy__"``).

        Raises
        ------
        ValueError
            If ``name`` is neither ignored nor in the delegation map, signalling that
            the caller may have intended to access ``optical``, ``heat``, or
            ``charge`` directly.

        Notes
        -----
        Only the attributes enumerated in the local ``DELEGATED_ATTRIBUTES`` dict are
        forwarded.
        Extend that mapping as additional cross-medium shim behaviour becomes
        necessary.
        """
        # first check whether the attribute is already present
        try:
            return super().__getattr__(name)
        except AttributeError:
            pass

        IGNORED_ATTRIBUTES = ["__deepcopy__"]
        if name in IGNORED_ATTRIBUTES:
            return None

        DELEGATED_ATTRIBUTES = {
            "is_pmc": self.optical,
            "_eps_plot": self.optical,
            "viz_spec": self.optical,
            "eps_diagonal_numerical": self.optical,
            "eps_complex_to_nk": self.optical,
            "nonlinear_spec": self.optical,
            "is_pec": self.optical,
            "is_pec_like": self.optical,
            "is_time_modulated": self.optical,
            "is_nonlinear": self.optical,
            "is_fully_anisotropic": self.optical,
            "is_custom": self.optical,
            "is_isotropic": self.optical,
            "is_spatially_uniform": self.optical,
            "_incompatible_material_types": self.optical,
            "frequency_range": self.optical,
            "eps_model": self.optical,
            "n_cfl": self.optical,
            "allow_gain": self.optical,
        }

        if name == "_has_incompatibilities":
            return (self.optical and self.optical._has_incompatibilities) or (
                self.charge and self.charge._has_incompatibilities
            )

        if name in DELEGATED_ATTRIBUTES:
            sub = DELEGATED_ATTRIBUTES[name]
            if sub is None:
                raise AttributeError(
                    f"Requested attribute {name!r}, but the optical medium is 'None' "
                    " on this 'MultiPhysicsMedium' instance."
                )
            return getattr(sub, name)

        raise AttributeError(
            f"MultiPhysicsMedium has no attribute called {name}. "
            "Did you mean to access the attribute of one of the optical, heat or charge media?"
        )

    @property
    def heat_spec(self) -> HeatMediumType | None:
        if self.heat is not None:
            return self.heat

        if self.optical is not None:
            return self.optical.heat_spec
        return None
