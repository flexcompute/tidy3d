"""Defines electric current sources for injecting light into simulation."""

from __future__ import annotations

from abc import ABC
from math import cos, isclose, sin
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import numpy as np
from pydantic import Field, model_validator

from tidy3d.components.autograd.derivative_utils import (
    transpose_interp_field_to_dataset,
)
from tidy3d.components.base import cached_property
from tidy3d.components.data.dataset import FieldDataset
from tidy3d.components.data.validators import validate_can_interpolate, validate_no_nans
from tidy3d.components.types import Polarization
from tidy3d.components.validators import assert_single_freq_in_range, warn_if_dataset_none
from tidy3d.constants import MICROMETER
from tidy3d.log import log

from .adjoint_helpers import (
    accumulate_center_vjp,
    assign_center_path_derivatives,
    parse_source_field_component,
    split_source_paths,
)
from .base import Source

if TYPE_CHECKING:
    from tidy3d.compat import Self
    from tidy3d.components.autograd import AutogradFieldMap
    from tidy3d.components.autograd.derivative_utils import DerivativeInfo
    from tidy3d.components.data.data_array import ScalarFieldDataArray
    from tidy3d.components.types.time import SourceTimeType


def _get_direct_source_adjoint_and_sign(
    field_name: str,
    *,
    e_adj: dict[str, Any],
    h_adj: dict[str, Any],
) -> tuple[Any, float]:
    """Get adjoint field/sign for direct ``CustomCurrentSource`` gradients."""
    field_type, _ = parse_source_field_component(field_name)
    if field_type == "H":
        return h_adj[field_name], -1.0
    return e_adj[field_name], 1.0


class CurrentSource(Source, ABC):
    """Source implements a current distribution directly."""

    polarization: Polarization = Field(
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
        return tuple(pol_vec)


class ReverseInterpolatedSource(Source):
    """Abstract source that allows reverse-interpolation along zero-sized dimensions."""

    interpolate: bool = Field(
        True,
        title="Enable Interpolation",
        description="Handles reverse-interpolation of zero-size dimensions of the source. "
        "If ``False``, the source data is snapped to the nearest Yee grid point. If ``True``, "
        "equivalent source data is applied on the surrounding Yee grid points to emulate "
        "placement at the specified location using linear interpolation.",
    )

    confine_to_bounds: bool = Field(
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
    >>> pt_source = UniformCurrentSource(
    ...     size=(0,0,0), source_time=pulse, polarization='Ex', current_amplitude_definition='total',
    ... )
    """

    current_amplitude_definition: Literal["density", "total"] = Field(
        "density",
        title="Current Amplitude Definition",
        description="Defines how the ``source_time`` amplitude is interpreted. "
        "If ``'total'``, the ``source_time`` parameter is interpreted as the total "
        "current in Amperes (A) / Volts (V) when an electric / magnetic current polarization "
        "is chosen. The solver automatically scales the current density by the source "
        "cross-sectional area to ensure the integrated current equals the specified amplitude, "
        "regardless of mesh resolution. If ``'density'`` (default), ``source_time`` represents "
        "the current density (e.g., A/m²), meaning the total injected current will scale with "
        "the source geometry size.",
    )

    @model_validator(mode="after")
    def _warn_current_amplitude_definition_default_change(self) -> Self:
        """Warn that the default of 'current_amplitude_definition' will change from 'density' to 'total'."""
        if "current_amplitude_definition" not in self.model_fields_set:
            log.warning(
                "The default value of 'current_amplitude_definition' for 'UniformCurrentSource' "
                "will change from 'density' to 'total' in a future release. To avoid this warning "
                "and ensure consistent behavior, please explicitly set "
                "current_amplitude_definition='density' or current_amplitude_definition='total' "
                "when creating the source."
            )
        return self


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

    size: tuple[Literal[0], Literal[0], Literal[0]] = Field(
        (0, 0, 0),
        title="Size",
        description="Size in x, y, and z directions, constrained to ``(0, 0, 0)``.",
        json_schema_extra={"units": MICROMETER},
    )

    @classmethod
    def sources_from_angles(
        cls,
        source_time: SourceTimeType,
        angle_theta: float,
        angle_phi: float,
        component: Literal["electric", "magnetic"] = "electric",
        **kwargs: Any,
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
        In other words, the dataset is interpreted in a local coordinate frame centered at
        :attr:`center`; when injecting/interpolating, the simulation-space coordinates are
        ``dataset_coords + center``. This means the same dataset can be translated in space by
        changing :attr:`center` without modifying dataset coordinates.

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

    _supported_traced_source_fields: ClassVar[tuple[str, ...]] = ("current_dataset", "center")
    _traced_source_dataset_key: ClassVar[str] = "current_dataset"

    current_dataset: FieldDataset | None = Field(
        title="Current Dataset",
        description=":class:`.FieldDataset` containing the desired frequency-domain "
        "electric and magnetic current patterns to inject.",
    )

    _no_nans_dataset = validate_no_nans("current_dataset")
    _current_dataset_none_warning = warn_if_dataset_none("current_dataset")
    _current_dataset_single_freq = assert_single_freq_in_range("current_dataset")
    _can_interpolate = validate_can_interpolate("current_dataset")

    def _confine_mask(self, field_data: ScalarFieldDataArray) -> np.ndarray:
        """Mask selecting dataset points inside source bounds on nonzero-size axes."""
        mask = np.ones(field_data.shape, dtype=float)
        for dim in "xyz":
            if dim not in field_data.coords:
                continue
            axis = "xyz".index(dim)
            half_size = 0.5 * float(self.size[axis])
            if half_size <= 0.0:
                continue
            coords = np.asarray(field_data.coords[dim].data, dtype=float)
            inside = np.abs(coords) <= (half_size + 1e-12)
            reshape = [1] * mask.ndim
            reshape[field_data.dims.index(dim)] = coords.size
            mask *= inside.reshape(reshape)
        return mask

    def _adjoint_interp_methods(self) -> dict[str, str]:
        """Interpolation mode per axis for source VJP projection."""
        if self.interpolate:
            return dict.fromkeys("xyz", "linear")

        axis_methods = {}
        for axis, dim in enumerate("xyz"):
            axis_methods[dim] = "nearest" if np.isclose(self.size[axis], 0.0) else "linear"
        return axis_methods

    def _compute_dataset_derivatives(
        self,
        dataset_paths: list[tuple],
        *,
        center: tuple[float, float, float],
        interp_methods: dict[str, str],
        e_adj: dict[str, Any],
        h_adj: dict[str, Any],
    ) -> AutogradFieldMap:
        """Compute derivatives for traced ``current_dataset`` paths."""
        derivative_map: AutogradFieldMap = {}
        for field_path in dataset_paths:
            field_path = tuple(field_path)
            field_name = field_path[1]
            field_data = getattr(self.current_dataset, field_name)

            adjoint_field, component_sign = _get_direct_source_adjoint_and_sign(
                field_name,
                e_adj=e_adj,
                h_adj=h_adj,
            )

            adjoint_on_dataset = transpose_interp_field_to_dataset(
                adjoint_field,
                field_data,
                center=center,
                method=interp_methods,
            )
            if self.confine_to_bounds:
                adjoint_on_dataset = adjoint_on_dataset * self._confine_mask(field_data)

            # Keep source gradients stable against simulation grid-refinement changes.
            vjp_field = component_sign * adjoint_on_dataset
            derivative_map[field_path] = vjp_field.transpose(*field_data.dims).values

        return derivative_map

    def _compute_center_derivatives(
        self,
        center_paths: list[tuple],
        *,
        center: tuple[float, float, float],
        bounds: Any,
        e_adj: dict[str, Any],
        h_adj: dict[str, Any],
    ) -> AutogradFieldMap:
        """Compute derivatives for traced ``center`` paths."""
        derivative_map: AutogradFieldMap = {}
        if not center_paths:
            return derivative_map

        def _get_adjoint_and_sign(field_name: str) -> tuple[Any, float]:
            return _get_direct_source_adjoint_and_sign(
                field_name,
                e_adj=e_adj,
                h_adj=h_adj,
            )

        vjp_center = accumulate_center_vjp(
            field_components=self.current_dataset.field_components,
            center=center,
            bounds=bounds,
            source_size=tuple(self.size),
            get_adjoint_and_sign=_get_adjoint_and_sign,
        )

        assign_center_path_derivatives(
            derivative_map,
            center_paths,
            vjp_center=vjp_center,
        )
        return derivative_map

    def _compute_derivatives(self, derivative_info: DerivativeInfo) -> AutogradFieldMap:
        """Compute derivatives with respect to CustomCurrentSource parameters."""
        derivative_map = {}
        center = tuple(self.center)
        interp_methods = self._adjoint_interp_methods()
        h_adj = derivative_info.H_adj or {}
        e_adj = derivative_info.E_adj or {}

        dataset_paths, center_paths = split_source_paths(derivative_info.paths)

        derivative_map.update(
            self._compute_dataset_derivatives(
                dataset_paths,
                center=center,
                interp_methods=interp_methods,
                e_adj=e_adj,
                h_adj=h_adj,
            )
        )
        derivative_map.update(
            self._compute_center_derivatives(
                center_paths,
                center=center,
                bounds=derivative_info.bounds,
                e_adj=e_adj,
                h_adj=h_adj,
            )
        )

        return derivative_map
