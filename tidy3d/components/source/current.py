"""Defines electric current sources for injecting light into simulation."""

from __future__ import annotations

from abc import ABC
from math import cos, isclose, sin
from typing import Optional

import pydantic.v1 as pydantic
from typing_extensions import Literal

from tidy3d.components.autograd import AutogradFieldMap
from tidy3d.components.autograd.derivative_utils import DerivativeInfo
from tidy3d.components.base import cached_property
from tidy3d.components.data.dataset import FieldDataset
from tidy3d.components.data.validators import validate_can_interpolate, validate_no_nans
from tidy3d.components.types import Polarization
from tidy3d.components.validators import assert_single_freq_in_range, warn_if_dataset_none
from tidy3d.constants import MICROMETER

from .base import Source
from .time import SourceTimeType


class CurrentSource(Source, ABC):
    """Source implements a current distribution directly."""

    polarization: Polarization = pydantic.Field(
        ...,
        title="Polarization",
        description="Specifies the direction and type of current component.",
    )

    @cached_property
    def _pol_vector(self) -> tuple[float, float, float]:
        """Returns a vector indicating the source polarization for arrow plotting, if not None."""
        component = self.polarization[-1]  # 'x' 'y' or 'z'
        pol_axis = "xyz".index(component)
        pol_vec = [0, 0, 0]
        pol_vec[pol_axis] = 1
        return pol_vec


class ReverseInterpolatedSource(Source):
    """Abstract source that allows reverse-interpolation along zero-sized dimensions."""

    interpolate: bool = pydantic.Field(
        True,
        title="Enable Interpolation",
        description="Handles reverse-interpolation of zero-size dimensions of the source. "
        "If ``False``, the source data is snapped to the nearest Yee grid point. If ``True``, "
        "equivalent source data is applied on the surrounding Yee grid points to emulate "
        "placement at the specified location using linear interpolation.",
    )

    confine_to_bounds: bool = pydantic.Field(
        False,
        title="Confine to Analytical Bounds",
        description="If ``True``, any source amplitudes which, after discretization, fall beyond "
        "the bounding box of the source are zeroed out, but only along directions where "
        "the source has a non-zero extent. The bounding box is inclusive. Should be set ```True`` "
        "when the current source is being used to excite a current in a conductive material.",
    )


class UniformCurrentSource(CurrentSource, ReverseInterpolatedSource):
    """Source in a rectangular volume with uniform time dependence.

    Notes
    -----

        Inputting the parameter ``size=(0,0,0)`` defines the equivalent of a point source.

    Example
    -------
    >>> from tidy3d import GaussianPulse
    >>> pulse = GaussianPulse(freq0=200e12, fwidth=20e12)
    >>> pt_source = UniformCurrentSource(size=(0,0,0), source_time=pulse, polarization='Ex')
    """


class PointDipole(CurrentSource, ReverseInterpolatedSource):
    """Uniform current source with a zero size. The source corresponds to an infinitesimal antenna
    with a fixed current density, and is slightly different from a related definition that is used
    in some contexts, namely an oscillating electric or magnetic dipole. The two are related through
    a factor of ``omega ** 2`` in the power normalization, where ``omega`` is the angular frequency
    of the oscillation. This is discussed further in our
    `source normalization <../../faq/docs/faq/How-are-results-normalized.html>`_ FAQ page.

    ..
        TODO add image of how it looks like based on sim 1.

    Example
    -------
    >>> from tidy3d import GaussianPulse
    >>> pulse = GaussianPulse(freq0=200e12, fwidth=20e12)
    >>> pt_dipole = PointDipole(center=(1,2,3), source_time=pulse, polarization='Ex')

    See Also
    --------

    **Notebooks**
        * `Particle swarm optimization of quantum emitter light extraction to free space <../../notebooks/BullseyeCavityPSO.html>`_
        * `Adjoint optimization of quantum emitter light extraction to an integrated waveguide <../../notebooks/AdjointPlugin12LightExtractor.html>`_
    """

    size: tuple[Literal[0], Literal[0], Literal[0]] = pydantic.Field(
        (0, 0, 0),
        title="Size",
        description="Size in x, y, and z directions, constrained to ``(0, 0, 0)``.",
        units=MICROMETER,
    )

    @classmethod
    def sources_from_angles(
        cls,
        source_time: SourceTimeType,
        angle_theta: float,
        angle_phi: float,
        component: Literal["electric", "magnetic"] = "electric",
        **kwargs,
    ) -> list[PointDipole]:
        """Returns a list of `PointDipole` objects used to emulate a single dipole polarized in an arbitrary direction. The direction is specificed using a polar and azimuthal angle.

        Parameters
        ----------
        source_time: :class:`.SourceTime`
            Specification of the source time-dependence.
        angle_theta : float
            Polar angle w.r.t. the z-axis.
        angle_phi : float
            Azimuth angle around the z-axis.
        component : Literal["electric", "magnetic"] = "electric"
            The type of polarization.
        kwargs : dict
            Keyword arguments passed to ``PointDipole()``, excluding ``source_time`` and ``polarization``

        Returns
        -------
        list[PointDipole]
            A list of ``PointDipole`` objects that emulate a single dipole with an arbitrary direction of polarization.
        """
        if not (component == "electric" or component == "magnetic"):
            raise ValueError('Component must be "electric" or "magnetic"')

        dipoles: list[PointDipole] = []
        polarizations = ["Ex", "Ey", "Ez"] if component == "electric" else ["Hx", "Hy", "Hz"]

        multipliers = [
            sin(angle_theta) * cos(angle_phi),
            sin(angle_theta) * sin(angle_phi),
            cos(angle_theta),
        ]

        for polarization, mult in zip(polarizations, multipliers):
            if not isclose(mult, 0.0, rel_tol=0.0, abs_tol=1e-9):
                modulated_source_time = source_time.updated_copy(
                    amplitude=source_time.amplitude * mult
                )
                dipoles.append(
                    cls(
                        source_time=modulated_source_time,
                        polarization=polarization,
                        **kwargs,
                    )
                )

        return dipoles


class CustomCurrentSource(ReverseInterpolatedSource):
    """Implements a source corresponding to an input dataset containing ``E`` and ``H`` fields.

    Notes
    -----

        Injects the specified components of the ``E`` and ``H`` dataset directly as ``J`` and ``M`` current
        distributions in the FDTD solver. The coordinates of all provided fields are assumed to be relative to the
        source center.

        The syntax is very similar to :class:`CustomFieldSource`, except instead of a ``field_dataset``, the source
        accepts a :attr:`current_dataset`. This dataset still contains :math:`E_{x,y,z}` and :math:`H_{x,y,
        z}` field components, which correspond to :math:`J` and :math:`M` components respectively. There are also
        fewer constraints on the data requirements for :class:`CustomCurrentSource`. It can be volumetric or planar
        without requiring tangential components. Finally, note that the dataset is still defined w.r.t. the source
        center, just as in the case of the :class:`CustomFieldSource`, and can then be placed anywhere in the simulation.

    Example
    -------
    >>> from tidy3d import ScalarFieldDataArray, GaussianPulse
    >>> import numpy as np
    >>> pulse = GaussianPulse(freq0=200e12, fwidth=20e12)
    >>> x = np.linspace(-1, 1, 101)
    >>> y = np.linspace(-1, 1, 101)
    >>> z = np.array([0])
    >>> f = [2e14]
    >>> coords = dict(x=x, y=y, z=z, f=f)
    >>> scalar_field = ScalarFieldDataArray(np.ones((101, 101, 1, 1)), coords=coords)
    >>> dataset = FieldDataset(Ex=scalar_field)
    >>> custom_source = CustomCurrentSource(
    ...     center=(1, 1, 1),
    ...     size=(2, 2, 0),
    ...     source_time=pulse,
    ...     current_dataset=dataset)

    See Also
    --------

    **Notebooks**
        * `Defining spatially-varying sources <../../notebooks/CustomFieldSource.html>`_
    """

    current_dataset: Optional[FieldDataset] = pydantic.Field(
        ...,
        title="Current Dataset",
        description=":class:`.FieldDataset` containing the desired frequency-domain "
        "electric and magnetic current patterns to inject.",
    )

    _no_nans_dataset = validate_no_nans("current_dataset")
    _current_dataset_none_warning = warn_if_dataset_none("current_dataset")
    _current_dataset_single_freq = assert_single_freq_in_range("current_dataset")
    _can_interpolate = validate_can_interpolate("current_dataset")

    def _compute_derivatives(self, derivative_info: DerivativeInfo) -> AutogradFieldMap:
        """Compute derivatives with respect to CustomCurrentSource parameters.

        The VJP rule for CustomCurrentSource is similar to CustomMedium.permittivity,
        but we only use E_adj (ignore E_fwd) and compute derivatives with respect to
        the current_dataset field components (Ex, Ey, Ez, Hx, Hy, Hz).

        Parameters
        ----------
        derivative_info : DerivativeInfo
            Information needed for derivative computation.

        Returns
        -------
        AutogradFieldMap
            Dictionary mapping parameter paths to their gradients.
        """
        # Import here to avoid circular imports
        from tidy3d.components.autograd.derivative_utils import integrate_within_bounds

        # Get the source bounds for integration
        source_bounds = derivative_info.bounds

        # Compute derivatives with respect to each field component in current_dataset
        derivative_map = {}

        # For CustomCurrentSource, we compute derivatives with respect to the field components
        # in current_dataset (Ex, Ey, Ez, Hx, Hy, Hz)
        for field_path in derivative_info.paths:
            field_name = field_path[-1]  # e.g., 'Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz'

            # Get the corresponding adjoint field component
            if field_name in derivative_info.E_adj:
                adjoint_field = derivative_info.E_adj[field_name]

                # Integrate the adjoint field within the source bounds
                # This gives us the gradient with respect to the current_dataset field component
                vjp_value = integrate_within_bounds(
                    arr=adjoint_field,
                    dims=("x", "y", "z"),
                    bounds=source_bounds,
                )

                # Sum over frequency dimension to get scalar gradient
                vjp_scalar = vjp_value.sum().values

                # Store the gradient for this field component
                derivative_map[tuple(field_path)] = vjp_scalar
            else:
                # If the field component is not in E_adj, set gradient to 0
                derivative_map[tuple(field_path)] = 0.0

        # For now, return placeholder gradients with expected structure
        # This is a placeholder for future implementation
        import tidy3d as td

        td.log.debug("CustomCurrentSource gradient computation not yet implemented")

        return derivative_map
