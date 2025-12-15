"""Defines heat material specifications"""

from __future__ import annotations

from typing import TYPE_CHECKING, Union

import numpy as np
import pydantic.v1 as pd

from tidy3d.components.data.data_array import SpatialDataArray
from tidy3d.components.medium import AbstractMedium
from tidy3d.components.tcad.doping import ConstantDoping, DopingBoxType
from tidy3d.components.tcad.types import (
    BandGapNarrowingModelType,
    ConstantEffectiveDOS,
    ConstantEnergyBandGap,
    EffectiveDOSModelType,
    EnergyBandGapModelType,
    MobilityModelType,
    RecombinationModelType,
)
from tidy3d.constants import CONDUCTIVITY, ELECTRON_VOLT, PERCMCUBE, PERMITTIVITY
from tidy3d.log import log

if TYPE_CHECKING:
    from tidy3d.components.tcad.grid import DopingGradientRefinementSpec, GridRefinementRegion
    from tidy3d.components.types import Bound


class AbstractChargeMedium(AbstractMedium):
    """Abstract class for Charge specifications
    Currently, permittivity is treated as a constant."""

    permittivity: float = pd.Field(
        1.0, ge=1.0, title="Permittivity", description="Relative permittivity.", units=PERMITTIVITY
    )

    @property
    def charge(self):
        """
        This means that a charge medium has been defined inherently within this solver medium.
        This provides interconnection with the :class:`MultiPhysicsMedium` higher-dimensional classes.
        """
        return self

    def eps_model(self, frequency: float) -> complex:
        return self.permittivity

    def n_cfl(self) -> None:
        return None


class ChargeInsulatorMedium(AbstractChargeMedium):
    """
    Insulating medium. Conduction simulations will not solve for electric
    potential in a structure that has a medium with this ``charge``.

    Example
    -------
    >>> import tidy3d as td
    >>> solid = td.ChargeInsulatorMedium()
    >>> solid2 = td.ChargeInsulatorMedium(permittivity=1.1)

    Note
    ----
        A relative permittivity :math:`\\varepsilon` will be assumed 1 if no value is specified.
    """


class ChargeConductorMedium(AbstractChargeMedium):
    """Conductor medium for conduction simulations.

    Example
    -------
    >>> import tidy3d as td
    >>> solid = td.ChargeConductorMedium(conductivity=3)

    Note
    ----
        A relative permittivity will be assumed 1 if no value is specified.
    """

    conductivity: pd.PositiveFloat = pd.Field(
        ...,
        title="Electric conductivity",
        description="Electric conductivity of material.",
        units=CONDUCTIVITY,
    )


class SemiconductorMedium(AbstractChargeMedium):
    """
    This class is used to define semiconductors.

    Notes
    -----
    Semiconductors are associated with ``Charge`` simulations. During these simulations
    the Drift-Diffusion (DD) equations will be solved in semiconductors. In what follows, a
    description of the assumptions taken and its limitations is put forward.

    The iso-thermal DD equations are summarized here

    .. math::

        \\begin{equation}
                - \\nabla \\cdot \\left( \\varepsilon_0 \\varepsilon_r \\nabla \\psi \\right) = q
            \\left( p - n + N_d^+ - N_a^- \\right)
        \\end{equation}

    .. math::

        \\begin{equation}
            q \\frac{\\partial n}{\\partial t} = \\nabla \\cdot \\mathbf{J_n} - qR
        \\end{equation}

    .. math::

        \\begin{equation}
            q \\frac{\\partial p}{\\partial t} = -\\nabla \\cdot \\mathbf{J_p} - qR
        \\end{equation}

    As well as iso-thermal, the system is considered to be at :math:`T=300`. This restriction will
    be removed in future releases.

    The above system requires the definition of the flux functions (free carrier current density), :math:`\\mathbf{J_n}` and
    :math:`\\mathbf{J_p}`. We consider the usual form

    .. math::

        \\begin{equation}
             \\mathbf{J_n} = q \\mu_n \\mathbf{F_{n}} + q D_n \\nabla n
        \\end{equation}


    .. math::

        \\begin{equation}
             \\mathbf{J_p} = q \\mu_p \\mathbf{F_{p}} - q D_p \\nabla p
        \\end{equation}


    where we simplify the effective field defined in [1]_ to

    .. math::

        \\begin{equation}
            \\mathbf{F_{n,p}} = \\nabla \\psi
        \\end{equation}

    i.e., we are not considering the effect of band-gap narrowing and degeneracy on the effective
    electric field :math:`\\mathbf{F_{n,p}}`. This is a good approximation for non-degenerate semiconductors.

    Let's explore how material properties are defined as class parameters or other classes.

     .. list-table::
       :widths: 25 25 75
       :header-rows: 1

       * - Symbol
         - Parameter Name
         - Description
       * - :math:`N_a`
         - ``N_a``
         - Ionized acceptors density
       * - :math:`N_d`
         - ``N_d``
         - Ionized donors density
       * - :math:`N_c`
         - ``N_c``
         - Effective density of states in the conduction band.
       * - :math:`N_v`
         - ``N_v``
         - Effective density of states in valence band.
       * - :math:`R`
         - ``R``
         - Generation-Recombination term.
       * - :math:`E_g`
         - ``E_g``
         - Bandgap Energy.
       * - :math:`\\Delta E_g`
         - ``delta_E_g``
         - Bandgap Narrowing.
       * - :math:`\\sigma`
         - ``conductivity``
         - Electrical conductivity.
       * - :math:`\\varepsilon_r`
         - ``permittivity``
         - Relative permittivity.
       * - :math:`q`
         - ``tidy3d.constants.Q_e``
         - Fundamental electron charge.

    Example
    -------
        >>> import tidy3d as td
        >>> default_Si = td.SemiconductorMedium(
        ...     N_c=td.ConstantEffectiveDOS(N=2.86e19),
        ...     N_v=td.ConstantEffectiveDOS(N=3.1e19),
        ...     E_g=td.ConstantEnergyBandGap(eg=1.11),
        ...     mobility_n=td.CaugheyThomasMobility(
        ...         mu_min=52.2,
        ...         mu=1471.0,
        ...         ref_N=9.68e16,
        ...         exp_N=0.68,
        ...         exp_1=-0.57,
        ...         exp_2=-2.33,
        ...         exp_3=2.4,
        ...         exp_4=-0.146,
        ...     ),
        ...     mobility_p=td.CaugheyThomasMobility(
        ...         mu_min=44.9,
        ...         mu=470.5,
        ...         ref_N=2.23e17,
        ...         exp_N=0.719,
        ...         exp_1=-0.57,
        ...         exp_2=-2.33,
        ...         exp_3=2.4,
        ...         exp_4=-0.146,
        ...     ),
        ...     R=([
        ...         td.ShockleyReedHallRecombination(
        ...             tau_n=3.3e-6,
        ...             tau_p=4e-6
        ...         ),
        ...         td.RadiativeRecombination(
        ...             r_const=1.6e-14
        ...         ),
        ...         td.AugerRecombination(
        ...             c_n=2.8e-31,
        ...             c_p=9.9e-32
        ...         ),
        ...     ]),
        ...     delta_E_g=td.SlotboomBandGapNarrowing(
        ...         v1=6.92 * 1e-3,
        ...         n2=1.3e17,
        ...         c2=0.5,
        ...         min_N=1e15,
        ...     ),
        ...     N_a=[td.ConstantDoping(concentration=1e15)],
        ...     N_d=[td.ConstantDoping(concentration=1e15)]
        ... )


    Warning
    -------
        Current limitations of the formulation include:

        - Boltzmann statistics are supported
        - Iso-thermal equations with :math:`T=300K`
        - Steady state only
        - Dopants are considered to be fully ionized

    Note
    ----
        - Both :math:`N_a` and :math:`N_d` can be either a positive number or an ``xarray.DataArray``.
        - Default values for parameters and models are those appropriate for Silicon.
        - The current implementation is a good approximation for non-degenerate semiconductors.


    .. [1] Schroeder, D., T. Ostermann, and O. Kalz. "Comparison of transport models far the simulation of degenerate semiconductors." Semiconductor science and technology 9.4 (1994): 364.

    """

    N_c: Union[EffectiveDOSModelType, pd.PositiveFloat] = pd.Field(
        ...,
        title="Effective density of electron states",
        description=":math:`N_c` Effective density of states in the conduction band.",
        units=PERCMCUBE,
    )

    N_v: Union[EffectiveDOSModelType, pd.PositiveFloat] = pd.Field(
        ...,
        title="Effective density of hole states",
        description=":math:`N_v` Effective density of states in the valence band.",
        units=PERCMCUBE,
    )

    E_g: Union[EnergyBandGapModelType, pd.PositiveFloat] = pd.Field(
        ...,
        title="Band-gap energy",
        description=":math:`E_g` Band-gap energy",
        units=ELECTRON_VOLT,
    )

    mobility_n: MobilityModelType = pd.Field(
        ...,
        title="Mobility model for electrons",
        description="Mobility model for electrons",
    )

    mobility_p: MobilityModelType = pd.Field(
        ...,
        title="Mobility model for holes",
        description="Mobility model for holes",
    )

    R: tuple[RecombinationModelType, ...] = pd.Field(
        [],
        title="Generation-Recombination models",
        description="Array containing the R models to be applied to the material.",
    )

    delta_E_g: BandGapNarrowingModelType = pd.Field(
        None,
        title="Bandgap narrowing model.",
        description=":math:`\\Delta E_g` Bandgap narrowing model.",
        units=ELECTRON_VOLT,
    )

    N_a: Union[pd.NonNegativeFloat, SpatialDataArray, tuple[DopingBoxType, ...]] = pd.Field(
        (),
        title="Doping: Acceptor concentration",
        description="Concentration of acceptor impurities, which create mobile holes, resulting in p-type material. "
        "Can be specified as a single float for uniform doping, a :class:`SpatialDataArray` for a custom profile, "
        "or a tuple of geometric shapes to define specific doped regions.",
        units=PERCMCUBE,
    )

    N_d: Union[pd.NonNegativeFloat, SpatialDataArray, tuple[DopingBoxType, ...]] = pd.Field(
        (),
        title="Doping: Donor concentration",
        description="Concentration of donor impurities, which create mobile electrons, resulting in n-type material. "
        "Can be specified as a single float for uniform doping, a :class:`SpatialDataArray` for a custom profile, "
        "or a tuple of geometric shapes to define specific doped regions.",
        units=PERCMCUBE,
    )

    # DEPRECATION VALIDATORS
    @pd.validator("N_c", always=True)
    def check_nc_uses_model(cls, val, values):
        """Issue deprecation warning if float is provided"""
        if isinstance(val, (float, int)):
            log.warning(
                "Passing a float to 'N_c' is deprecated and will be removed in future versions. "
                "Please use 'ConstantEffectiveDOS' instead."
            )
            return ConstantEffectiveDOS(N=val)
        return val

    @pd.validator("N_v", always=True)
    def check_nv_uses_model(cls, val, values):
        """Issue deprecation warning if float is provided"""
        if isinstance(val, (float, int)):
            log.warning(
                "Passing a float to 'N_v' is deprecated and will be removed in future versions. "
                "Please use 'ConstantEffectiveDOS' instead."
            )
            return ConstantEffectiveDOS(N=val)
        return val

    @pd.validator("E_g", always=True)
    def check_eg_uses_model(cls, val, values):
        """Issue deprecation warning if float is provided"""
        if isinstance(val, (float, int)):
            log.warning(
                "Passing a float to 'E_g' is deprecated and will be removed in future versions. "
                "Please use 'ConstantEnergyBandGap' instead."
            )
            return ConstantEnergyBandGap(eg=val)
        return val

    @pd.validator("N_d", always=True)
    def check_nd_uses_model(cls, val, values):
        """Issue deprecation warning if float is provided"""
        if isinstance(val, (float, int)):
            log.warning(
                "Passing a float to 'N_d' is deprecated and will be removed in future versions. "
                f"Please use a list of 'DopingBoxType' instead, e.g., [ConstantDoping(concentration={val})]."
            )
            return (ConstantDoping(concentration=val),)
        return val

    @pd.validator("N_a", always=True)
    def check_na_uses_model(cls, val, values):
        """Issue deprecation warning if float is provided"""
        if isinstance(val, (float, int)):
            log.warning(
                "Passing a float to 'N_a' is deprecated and will be removed in future versions. "
                f"Please use a list of 'DopingBoxType' instead, e.g., [ConstantDoping(concentration={val})]."
            )
            return (ConstantDoping(concentration=val),)
        return val

    def compute_doping_refinement_regions(
        self,
        bounds: Bound,
        spec: DopingGradientRefinementSpec,
        normal_axis: int,
    ) -> list[GridRefinementRegion]:
        """Compute mesh refinement regions based on doping gradients.

        This method samples the net doping (|N_d - N_a|) on a 2D grid within
        the given bounds, computes the gradient magnitude, and generates
        :class:`GridRefinementRegion` objects in areas of rapid doping variation.

        Parameters
        ----------
        bounds : Bound
            Tuple of (min_corner, max_corner) defining the sampling region.
        spec : DopingGradientRefinementSpec
            Specification controlling refinement parameters.
        normal_axis : int
            The axis normal to the 2D plane (0=x, 1=y, 2=z).

        Returns
        -------
        list[GridRefinementRegion]
            List of refinement regions to be added to the mesh specification.
        """
        from tidy3d.components.tcad.grid import GridRefinementRegion

        refinement_regions = []

        # Determine the in-plane axes
        axes = [0, 1, 2]
        axes.remove(normal_axis)
        ax1, ax2 = axes

        # Extract bounds
        min_corner, max_corner = bounds
        normal_pos = (min_corner[normal_axis] + max_corner[normal_axis]) / 2

        # Create sampling grid
        num_samples = spec.num_samples
        coord_names = ["x", "y", "z"]

        # Create 1D coordinate arrays for the 2D plane
        coords_1 = np.linspace(min_corner[ax1], max_corner[ax1], num_samples)
        coords_2 = np.linspace(min_corner[ax2], max_corner[ax2], num_samples)

        # Build coords dict for sampling
        coords = {coord_names[normal_axis]: np.array([normal_pos])}
        coords[coord_names[ax1]] = coords_1
        coords[coord_names[ax2]] = coords_2

        # Compute net doping and gradient for N_d
        net_doping = np.zeros((num_samples, num_samples, 1))
        grad_magnitude = np.zeros((num_samples, num_samples, 1))

        # Process donor doping (N_d)
        if isinstance(self.N_d, tuple):
            for doping_box in self.N_d:
                doping_contrib = doping_box._get_contrib(coords, meshgrid=True)
                grad_contrib = doping_box.gradient_magnitude(coords, meshgrid=True)
                if doping_contrib.ndim == 2:
                    doping_contrib = doping_contrib[:, :, np.newaxis]
                    grad_contrib = grad_contrib[:, :, np.newaxis]
                net_doping = net_doping + doping_contrib
                grad_magnitude = np.maximum(grad_magnitude, grad_contrib)

        # Process acceptor doping (N_a) - subtract from net doping
        if isinstance(self.N_a, tuple):
            for doping_box in self.N_a:
                doping_contrib = doping_box._get_contrib(coords, meshgrid=True)
                grad_contrib = doping_box.gradient_magnitude(coords, meshgrid=True)
                if doping_contrib.ndim == 2:
                    doping_contrib = doping_contrib[:, :, np.newaxis]
                    grad_contrib = grad_contrib[:, :, np.newaxis]
                net_doping = net_doping - doping_contrib
                grad_magnitude = np.maximum(grad_magnitude, grad_contrib)

        # Squeeze to 2D
        net_doping = np.abs(net_doping.squeeze())
        grad_magnitude = grad_magnitude.squeeze()

        # Compute local mesh size: dl = N / (|grad_N| * k)
        # Avoid division by zero
        eps = 1e-10
        local_dl = np.where(
            grad_magnitude > eps,
            net_doping / (grad_magnitude * spec.resolution_factor),
            spec.max_dl,
        )

        # Clamp to [min_dl, max_dl]
        local_dl = np.clip(local_dl, spec.min_dl, spec.max_dl)

        # Find regions where mesh needs to be finer than max_dl
        # Group contiguous fine regions into refinement boxes
        needs_refinement = local_dl < spec.max_dl * 0.9  # 90% threshold

        if not np.any(needs_refinement):
            return refinement_regions

        # Simple approach: create refinement regions around each connected component
        # For efficiency, we use a grid-based clustering approach
        from scipy import ndimage

        labeled, num_features = ndimage.label(needs_refinement)

        for label_id in range(1, num_features + 1):
            mask = labeled == label_id

            # Find bounding box of this region
            rows, cols = np.where(mask)
            if len(rows) == 0:
                continue

            row_min, row_max = rows.min(), rows.max()
            col_min, col_max = cols.min(), cols.max()

            # Get the minimum dl in this region
            dl_internal = local_dl[mask].min()

            # Compute region bounds in physical coordinates
            region_min_1 = coords_1[row_min]
            region_max_1 = coords_1[row_max]
            region_min_2 = coords_2[col_min]
            region_max_2 = coords_2[col_max]

            # Build center and size for the refinement region
            center = [0.0, 0.0, 0.0]
            size = [0.0, 0.0, 0.0]

            center[normal_axis] = normal_pos
            size[normal_axis] = 0.0  # 2D refinement

            center[ax1] = (region_min_1 + region_max_1) / 2
            size[ax1] = region_max_1 - region_min_1

            center[ax2] = (region_min_2 + region_max_2) / 2
            size[ax2] = region_max_2 - region_min_2

            # Add some padding
            padding = dl_internal * 2
            size[ax1] = size[ax1] + 2 * padding
            size[ax2] = size[ax2] + 2 * padding

            transition_thickness = dl_internal * spec.transition_factor

            refinement_regions.append(
                GridRefinementRegion(
                    center=tuple(center),
                    size=tuple(size),
                    dl_internal=dl_internal,
                    transition_thickness=transition_thickness,
                )
            )

        return refinement_regions
