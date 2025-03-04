"""Defines electric current sources for injecting light into simulation."""

from __future__ import annotations

from abc import ABC
from typing import Optional, Tuple

import pydantic.v1 as pydantic
from typing_extensions import Literal

from ...constants import MICROMETER
from ..base import cached_property
from ..data.dataset import FieldDataset
from ..data.validators import validate_can_interpolate, validate_no_nans
from ..types import Polarization
from ..validators import assert_single_freq_in_range, warn_if_dataset_none
from .base import Source


class CurrentSource(Source, ABC):
    """Source implements a current distribution directly."""

    polarization: Polarization = pydantic.Field(
        ...,
        title="Polarization",
        description="Specifies the direction and type of current component.",
    )

    @cached_property
    def _pol_vector(self) -> Tuple[float, float, float]:
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

    size: Tuple[Literal[0], Literal[0], Literal[0]] = pydantic.Field(
        (0, 0, 0),
        title="Size",
        description="Size in x, y, and z directions, constrained to ``(0, 0, 0)``.",
        units=MICROMETER,
    )


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
