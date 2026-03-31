"""Defines electric field sources for injecting light into simulation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Literal, Optional, Union

import numpy as np
from pydantic import Field, NonNegativeInt, field_validator, model_validator

from tidy3d.components.autograd import TracedFloat, TracedPositiveFloat
from tidy3d.components.autograd.derivative_utils import (
    transpose_interp_field_to_dataset,
)
from tidy3d.components.autograd.utils import get_static, negate_vjp_map
from tidy3d.components.base import Tidy3dBaseModel, cached_property
from tidy3d.components.data.dataset import FieldDataset
from tidy3d.components.data.validators import validate_can_interpolate, validate_no_nans
from tidy3d.components.mode_spec import ModeSpec
from tidy3d.components.source.frame import PECFrame
from tidy3d.components.types import TYPE_TAG_STR, Axis, Direction
from tidy3d.components.types.mode_spec import ModeSpecType
from tidy3d.components.validators import (
    assert_plane,
    assert_single_freq_in_range,
    assert_volumetric,
    warn_backward_waist_distance,
    warn_if_dataset_none,
)
from tidy3d.constants import GLANCING_CUTOFF, MICROMETER, RADIAN, inf
from tidy3d.exceptions import AdjointError, SetupError
from tidy3d.log import log

from .adjoint_helpers import (
    accumulate_center_vjp,
    assign_center_path_derivatives,
    parse_source_field_component,
    split_source_paths,
    validate_no_collapsed_bounds_for_requested_center_axes,
    validate_no_zero_dim_center_paths,
)
from .base import Source

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from tidy3d.compat import Self
    from tidy3d.components.autograd import AutogradFieldMap, PathType
    from tidy3d.components.autograd.derivative_utils import DerivativeInfo
    from tidy3d.components.beam import BeamGrid, BeamPose
    from tidy3d.components.data.data_array import ScalarFieldDataArray
    from tidy3d.components.types import Ax, Coordinate

# width of Chebyshev grid used for broadband sources (in units of pulse width)
CHEB_GRID_WIDTH = 1.5
# For broadband plane waves with constan in-plane k, the Chebyshev grid is truncated at
# ``CRITICAL_FREQUENCY_FACTOR * f_crit``, where ``f_crit`` is the critical frequency
# (oblique propagation).
CRITICAL_FREQUENCY_FACTOR = 1.15


def _get_curl_coupled_source_adjoint_and_sign(
    *,
    field_name: str,
    injection_axis: int,
    e_adj: dict[str, Any],
    h_adj: dict[str, Any],
    source_name: str,
) -> tuple[Any, float]:
    """Get adjoint field/sign for curl-coupled ``CustomFieldSource`` gradients."""
    field_type, component_axis = parse_source_field_component(field_name, source_name=source_name)
    if component_axis == injection_axis:
        raise ValueError(
            f"Field component '{field_name}' is normal to injection axis '{'xyz'[injection_axis]}'."
        )

    n_vec = np.eye(3, dtype=float)[injection_axis]
    e_vec = np.eye(3, dtype=float)[component_axis]
    cross = np.cross(n_vec, e_vec)
    target_axis = int(np.flatnonzero(cross)[0])
    component_sign = float(cross[target_axis])

    target_component = f"{'H' if field_type == 'E' else 'E'}{'xyz'[target_axis]}"
    if field_type == "E":
        return h_adj[target_component], component_sign
    return e_adj[target_component], component_sign


class FieldSource(Source, ABC):
    """A Source defined by the desired E and/or H fields."""


""" Field Sources can be defined either on a (1) surface or (2) volume. Defines injection_axis """


class PlanarSource(Source, ABC):
    """A source defined on a 2D plane."""

    _plane_validator = assert_plane()
    use_colocated_integration: bool = Field(
        True,
        title="Use Colocated Integration",
        description="If ``True`` (default), source power normalization uses fields "
        "interpolated to grid cell boundaries (colocated). If ``False``, uses fields at "
        "native Yee grid positions (non-colocated). Should match the "
        "``use_colocated_integration`` setting on monitors for consistent power normalization. "
        "Experimental feature that can give improved accuracy by avoiding interpolation of "
        "fields to Yee cell positions for integration.",
    )

    @cached_property
    def injection_axis(self) -> Axis:
        """Injection axis of the source."""
        return self._injection_axis

    @cached_property
    def _injection_axis(self) -> Axis:
        """Injection axis of the source."""
        return self.size.index(0.0)


class VolumeSource(Source, ABC):
    """A source defined in a 3D :class:`~tidy3d.Box`."""

    _volume_validator = assert_volumetric()


""" Field Sources require more specification, for now, they all have a notion of a direction."""


class DirectionalSource(FieldSource, ABC):
    """A Field source that propagates in a given direction."""

    direction: Direction = Field(
        title="Direction",
        description="Specifies propagation in the positive or negative direction of the injection "
        "axis.",
    )

    @cached_property
    def _dir_vector(self) -> Optional[tuple[float, float, float]]:
        """Returns a vector indicating the source direction for arrow plotting, if not None."""
        if self._injection_axis is None:
            return None
        dir_vec = [0, 0, 0]
        dir_vec[int(self._injection_axis)] = 1 if self.direction == "+" else -1
        return tuple(dir_vec)


class BroadbandSource(Source, ABC):
    """A source with frequency dependent field distributions."""

    num_freqs: int = Field(
        1,
        title="Number of Frequency Points",
        description="Number of points used to approximate the frequency dependence of the injected "
        "field. For 'chebyshev', a Chebyshev interpolation is used with 'num_freqs' terms "
        "(max 20). For 'pole_residue', the mode solver samples at 'num_freqs' uniform "
        "frequencies and fits ceil((num_freqs - 1) / 3) poles; higher values provide denser "
        "sampling and more poles for the fit (max 50).",
        ge=1,
        le=50,
    )

    broadband_method: Literal["chebyshev", "pole_residue"] = Field(
        "chebyshev",
        title="Broadband Method",
        description="Method for representing the frequency dependence of the injected field. "
        "'chebyshev' uses Chebyshev polynomial interpolation (default). "
        "'pole_residue' uses a pole-residue (vector fitting) decomposition with "
        "auxiliary differential equation (ADE) time stepping. The pole-residue method "
        "can be more accurate for highly dispersive modes and uses fewer broadband terms "
        "than frequency samples.",
    )

    @model_validator(mode="after")
    def _validate_num_freqs_limit(self) -> Self:
        """Enforce method-specific bounds on num_freqs."""
        if self.broadband_method == "chebyshev" and self.num_freqs > 20:
            raise SetupError(
                f"For broadband_method='chebyshev', 'num_freqs' must be <= 20, got {self.num_freqs}."
            )
        if self.broadband_method == "pole_residue" and self.num_freqs < 3:
            raise SetupError(
                f"For broadband_method='pole_residue', 'num_freqs' must be >= 3, got {self.num_freqs}. "
                "The vector fitting algorithm requires more frequency samples than fit coefficients."
            )
        return self

    @cached_property
    def frequency_grid(self) -> NDArray:
        """Frequency grid used to approximate frequency dependence.

        For Chebyshev: returns ``num_freqs`` Chebyshev-spaced points.
        For pole_residue: returns ``num_freqs`` uniformly spaced points.
        """
        if self.num_freqs == 1:
            return np.array([self.source_time._freq0])
        freq_min, freq_max = self.source_time.frequency_range_sigma(sigma=CHEB_GRID_WIDTH)
        if self.broadband_method == "pole_residue":
            return np.linspace(freq_min, freq_max, self.num_freqs)
        return self._chebyshev_freq_grid(freq_min, freq_max)

    def _chebyshev_freq_grid(self, freq_min: float, freq_max: float) -> NDArray:
        """A Chebyshev grid based on a minimum and maximum frequency."""
        freq_avg = 0.5 * (freq_min + freq_max)
        freq_diff = 0.5 * (freq_max - freq_min)
        uni_points = (2 * np.arange(self.num_freqs) + 1) / (2 * self.num_freqs)
        cheb_points = np.cos(np.pi * np.flip(uni_points))
        return freq_avg + freq_diff * cheb_points


""" Source current profiles determined by user-supplied data on a plane."""


class CustomFieldSource(FieldSource, PlanarSource):
    """Implements a source corresponding to an input dataset containing ``E`` and ``H`` fields,
    using the equivalence principle to define the actual injected currents.

     Notes
     -----

        For the injection to work as expected (i.e. to reproduce the required ``E`` and ``H`` fields),
        the field data must decay by the edges of the source plane, or the source plane must span the entire
        simulation domain and the fields must match the simulation boundary conditions.

        The equivalent source currents are fully defined by the field components tangential to the
        source plane. For e.g. source normal along ``z``, the normal components (``Ez`` and ``Hz``)
        can be provided but will have no effect on the results, and at least one of the tangential
        components has to be in the dataset, i.e. at least one of ``Ex``, ``Ey``, ``Hx``, and ``Hy``.

        .. TODO add image here

        ..
            TODO is this generic? Only the field components tangential to the custom source plane are needed and used
            in the simulation. Due to the equivalence principle, these fully define the currents that need to be
            injected. This is not to say that the normal components of the data (:math:`E_x`, :math:`H_x` in our example)
            is lost or not injected. It is merely not needed as it can be uniquely obtained using the tangential components.

        ..
            TODO add example for this standalone
            Source data can be imported from file just as shown here, after the data is imported as a numpy array using
            standard numpy functions like loadtxt.

        If the data is not coming from a ``tidy3d`` simulation, the normalization is likely going to be arbitrary and
        the directionality of the source will likely not be perfect, even if both the ``E`` and ``H`` fields are
        provided. An empty normalizing run may be needed to accurately normalize results.

        To create this empty simulation it is recommended that users create a simulation with no structures but just a flux
        monitor (``tidy3D.FluxMonitor``) next to the custom source, ensuring that the flux monitor is at least one grid cell
        away from the source. Moreover, for accurate normalization, users must ensure that the same grid is used to run
        the original simulation as well as the empty simulation. The total flux calculated at the flux monitor of the empty
        simulation can then be used for proper normalization of results after ``tidy3d`` simulation.

        The coordinates of all provided fields are assumed to be relative to the source center.
        In other words, the dataset is interpreted in a local coordinate frame centered at
        :attr:`center`; when injecting/interpolating, the simulation-space coordinates are
        ``dataset_coords + center``. This means the same dataset can be translated in space by
        changing :attr:`center` without modifying dataset coordinates.
        If only the ``E`` or only the ``H`` fields are provided, the source will not be directional,
        but will inject equal power in both directions instead.

    Example
    -------
    >>> from tidy3d import ScalarFieldDataArray, GaussianPulse
    >>> import tidy3d as td
    >>> pulse = GaussianPulse(freq0=200e12, fwidth=20e12)
    >>> x = np.linspace(-1, 1, 101)
    >>> y = np.linspace(-1, 1, 101)
    >>> z = np.array([0])
    >>> f = [2e14]
    >>> coords = dict(x=x, y=y, z=z, f=f)
    >>> scalar_field = ScalarFieldDataArray(np.ones((101, 101, 1, 1)), coords=coords)
    >>> dataset = FieldDataset(Ex=scalar_field)
    >>> custom_source = CustomFieldSource(
    ...     center=(1, 1, 1),
    ...     size=(2, 2, 0),
    ...     source_time=pulse,
    ...     field_dataset=dataset)

    Creating an empty simulation with no structures with ``FluxMonitor`` for normalization but with the same grid as the
    original simulation.

    Example
    -------
    >>> Flux_monitor = td.FluxMonitor(
    ...     center=(0, 0, 0),
    ...     size=(3, 3, 3),
    ...     freqs=f,
    ...     name="Flux",
    ...     )
    >>> sim = td.Simulation(
    ...       center=[0,0,0],
    ...       size=(4, 4, 4),
    ...       structures=[],
    ...       sources=[custom_source],
    ...       monitors=[],
    ...       run_time = 1e-6,
    ...       shutoff=1e-6,
    ...    )
    >>> sim_empty = sim.updated_copy(monitors = [Flux_monitor],  # doctest: +SKIP
    ...             structures = [],
    ...             grid_spec= sim.grid_spec.updated_copy(override_structures = sim.structures)
    ...             )

    See Also
    --------

    **Notebooks**
        * `Defining spatially-varying sources <../../notebooks/CustomFieldSource.html>`_
    """

    field_dataset: Optional[FieldDataset] = Field(
        None,
        title="Field Dataset",
        description=":class:`.FieldDataset` containing the desired frequency-domain "
        "fields patterns to inject. At least one tangential field component must be specified.",
    )

    _no_nans_dataset = validate_no_nans("field_dataset")
    _field_dataset_none_warning = warn_if_dataset_none("field_dataset")
    _field_dataset_single_freq = assert_single_freq_in_range("field_dataset")
    _can_interpolate = validate_can_interpolate("field_dataset")

    @model_validator(mode="after")
    def _tangential_component_defined(self) -> FieldDataset:
        """Assert that at least one tangential field component is provided."""
        val = self.field_dataset
        if val is None:
            return self
        normal_axis = self.size.index(0.0)
        _, (cmp1, cmp2) = self.pop_axis("xyz", axis=normal_axis)
        for field in "EH":
            for cmp_name in (cmp1, cmp2):
                tangential_field = field + cmp_name
                if tangential_field in val.field_components:
                    return self
        raise SetupError("No tangential field found in the suppled 'field_dataset'.")

    def _compute_dataset_derivatives(
        self,
        dataset_paths: list[tuple],
        *,
        center: tuple[float, float, float],
        e_adj: dict[str, Any],
        h_adj: dict[str, Any],
    ) -> AutogradFieldMap:
        """Compute derivatives for traced ``field_dataset`` paths."""
        derivative_map: AutogradFieldMap = {}
        for field_path in dataset_paths:
            field_path = tuple(field_path)
            if len(field_path) < 2:
                raise ValueError(
                    "Field source derivative paths must include dataset component names, "
                    f"got '{field_path}'."
                )

            field_name = field_path[1]
            _, component_axis = parse_source_field_component(
                field_name, source_name=type(self).__name__
            )
            field_data = getattr(self.field_dataset, field_name, None)
            if field_data is None:
                raise ValueError(f"Cannot find field '{field_name}' in field dataset.")

            if component_axis == self.injection_axis:
                derivative_map[field_path] = np.zeros_like(field_data.data)
                continue

            adjoint_field, component_sign = _get_curl_coupled_source_adjoint_and_sign(
                field_name=field_name,
                injection_axis=self.injection_axis,
                e_adj=e_adj,
                h_adj=h_adj,
                source_name=type(self).__name__,
            )

            adjoint_on_dataset = transpose_interp_field_to_dataset(
                adjoint_field, field_data, center=center
            )

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

        center_field_components = {}
        for field_name, field_data in self.field_dataset.field_components.items():
            _, component_axis = parse_source_field_component(
                field_name, source_name=type(self).__name__
            )
            if component_axis != self.injection_axis:
                center_field_components[field_name] = field_data

        def _get_adjoint_and_sign(field_name: str) -> tuple[Any, float]:
            return _get_curl_coupled_source_adjoint_and_sign(
                field_name=field_name,
                injection_axis=self.injection_axis,
                e_adj=e_adj,
                h_adj=h_adj,
                source_name=type(self).__name__,
            )

        vjp_center = accumulate_center_vjp(
            field_components=center_field_components,
            center=center,
            bounds=bounds,
            source_size=tuple(self.size),
            get_adjoint_and_sign=_get_adjoint_and_sign,
        )

        validate_no_collapsed_bounds_for_requested_center_axes(
            center_paths,
            bounds=bounds,
        )
        assign_center_path_derivatives(
            derivative_map,
            center_paths,
            vjp_center=vjp_center,
        )
        return derivative_map

    def _compute_derivatives(self, derivative_info: DerivativeInfo) -> AutogradFieldMap:
        """Compute derivatives with respect to CustomFieldSource parameters."""
        derivative_map = {}
        center = tuple(self.center)
        e_adj = derivative_info.E_adj or {}
        h_adj = derivative_info.H_adj or {}

        supported_roots = ("field_dataset", "center")
        for field_path in derivative_info.paths:
            self._validate_traced_source_path(
                tuple(field_path),
                dataset_key="field_dataset",
                supported_roots=supported_roots,
            )

        dataset_paths, center_paths = split_source_paths(
            derivative_info.paths,
            primary_roots={"field_dataset"},
        )
        validate_no_zero_dim_center_paths(
            center_paths,
            source_size=tuple(self.size),
            source_name=type(self).__name__,
        )

        derivative_map.update(
            self._compute_dataset_derivatives(
                dataset_paths,
                center=center,
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


""" Source current profiles defined by (1) angle or (2) desired mode. Sets theta and phi angles."""


class AngledFieldSource(DirectionalSource, ABC):
    """A FieldSource defined with an angled direction of propagation.

    Notes
    -----

        The direction is defined by
        the polar and azimuth angles w.r.t. an injection axis, as well as forward ``+`` or
        backward ``-``. This base class only defines the :attr:`direction` and :attr:`injection_axis`
        attributes, but it must be composed with a class that also defines :attr:`angle_theta` and
        :attr:`angle_phi`.

    """

    angle_theta: float = Field(
        0.0,
        title="Polar Angle",
        description="Polar angle of the propagation axis from the injection axis.",
        json_schema_extra={"units": RADIAN},
    )

    angle_phi: float = Field(
        0.0,
        title="Azimuth Angle",
        description="Azimuth angle of the propagation axis in the plane orthogonal to the "
        "injection axis.",
        json_schema_extra={"units": RADIAN},
    )

    pol_angle: float = Field(
        0,
        title="Polarization Angle",
        description="Specifies the angle between the electric field polarization of the "
        "source and the plane defined by the injection axis and the propagation axis (rad). "
        "``pol_angle=0`` (default) specifies P polarization, "
        "while ``pol_angle=np.pi/2`` specifies S polarization. "
        "At normal incidence when S and P are undefined, ``pol_angle=0`` defines: "
        "- ``Ey`` polarization for propagation along ``x``."
        "- ``Ex`` polarization for propagation along ``y``."
        "- ``Ex`` polarization for propagation along ``z``.",
        json_schema_extra={"units": RADIAN},
    )

    @field_validator("angle_theta")
    @classmethod
    def glancing_incidence(cls, val: float) -> float:
        """Warn if close to glancing incidence."""
        static_val = get_static(val)
        if np.abs(np.pi / 2 - static_val) < GLANCING_CUTOFF:
            log.warning(
                "Angled source propagation axis close to glancing angle. "
                "For best results, switch the injection axis.",
                custom_loc=["angle_theta"],
            )
        return val

    @cached_property
    def _dir_vector(self) -> tuple[float, float, float]:
        """Source direction normal vector in cartesian coordinates."""

        # Propagation vector assuming propagation along z
        radius = 1.0 if self.direction == "+" else -1.0
        dx = radius * np.cos(self.angle_phi) * np.sin(self.angle_theta)
        dy = radius * np.sin(self.angle_phi) * np.sin(self.angle_theta)
        dz = radius * np.cos(self.angle_theta)

        # Move to original injection axis
        return self.unpop_axis(dz, (dx, dy), axis=self._injection_axis)

    @cached_property
    def _pol_vector(self) -> tuple[float, float, float]:
        """Source polarization normal vector in cartesian coordinates."""

        # Polarization vector assuming propagation along z
        pol_vector_z_normal = np.array([1.0, 0.0, 0.0])

        # Rotate polarization
        pol_vector_z_normal = self.rotate_points(
            pol_vector_z_normal, axis=[0, 0, 1], angle=self.pol_angle
        )

        # Rotate the fields back to the original propagation axes
        pol_vector_z_normal = self.rotate_points(
            pol_vector_z_normal, axis=[0, 1, 0], angle=self.angle_theta
        )
        pol_vector_z_normal = self.rotate_points(
            pol_vector_z_normal, axis=[0, 0, 1], angle=self.angle_phi
        )

        # Move to original injection axis
        pol_vector = self.unpop_axis(
            pol_vector_z_normal[2], pol_vector_z_normal[:2], axis=self._injection_axis
        )
        return pol_vector


class AbstractModeSource(DirectionalSource, PlanarSource, BroadbandSource):
    """Abstract base class for mode-based sources.

    Provides common functionality for sources that inject electromagnetic modes,
    including angle-dependent propagation direction and optional PEC frames.
    """

    mode_spec: ModeSpecType = Field(
        default_factory=ModeSpec,
        title="Mode Specification",
        description="Parameters to feed to mode solver which determine modes measured by monitor.",
        discriminator=TYPE_TAG_STR,
    )

    frame: Optional[PECFrame] = Field(
        None,
        title="Source Frame",
        description="Add a thin frame around the source during the FDTD run to improve "
        "the injection quality. The frame is positioned along the primal grid lines "
        "so that it aligns with the boundaries of the mode solver used to obtain the source profile.",
    )

    @cached_property
    def angle_theta(self) -> float:
        """Polar angle of propagation."""
        return self.mode_spec.angle_theta

    @cached_property
    def angle_phi(self) -> float:
        """Azimuth angle of propagation."""
        return self.mode_spec.angle_phi

    @cached_property
    def _dir_vector(self) -> tuple[float, float, float]:
        """Source direction normal vector in cartesian coordinates."""
        radius = 1.0 if self.direction == "+" else -1.0
        dx = radius * np.cos(self.angle_phi) * np.sin(self.angle_theta)
        dy = radius * np.sin(self.angle_phi) * np.sin(self.angle_theta)
        dz = radius * np.cos(self.angle_theta)
        return self.unpop_axis(dz, (dx, dy), axis=self._injection_axis)

    @cached_property
    def _bend_axis(self) -> Optional[Axis]:
        """Bend axis for curved sources."""
        if self.mode_spec.bend_radius is None:
            return None
        in_plane = [0, 0]
        in_plane[self.mode_spec.bend_axis] = 1
        direction = self.unpop_axis(0, in_plane, axis=self.injection_axis)
        return direction.index(1)


class ModeSource(AbstractModeSource):
    """Injects current source to excite modal profile on finite extent plane.

    Notes
    -----

        Using this mode source, it is possible selectively excite one of the guided modes of a waveguide. This can be
        computed in our eigenmode solver :class:`tidy3d.plugins.mode.ModeSolver` and implement the mode simulation in
        FDTD.

        Mode sources are normalized to inject exactly 1W of power at the central frequency.

        The modal source allows you to do directional excitation. Illustrated
        by the image below, the field is perfectly launched to the right of the source and there's zero field to the
        left of the source. Now you can contrast the behavior of the modal source with that of a dipole source. If
        you just put a dipole into the waveguide, well, you see quite a bit different in the field distribution.
        First of all, the dipole source is not directional launching. It launches waves in both directions. The
        second is that the polarization of the dipole is set to selectively excite a TE mode. But it takes some
        propagation distance before the mode settles into a perfect TE mode profile. During this process,
        there is radiation into the substrate.

        .. image:: /_static/img/mode_vs_dipole_source.png

        **Practical Advice**

        Place mode sources in waveguide sections with a uniform cross-section. The mode profile is
        computed assuming translational invariance (or constant bend radius) along the injection axis.
        For bent waveguides, set ``bend_radius`` and ``bend_axis`` accordingly. Avoid placing mode
        sources at tapers, junctions, or other locations where the cross-section is changing.

        .. TODO improve links to other APIs functionality here.

    Example
    -------
    >>> from tidy3d import GaussianPulse
    >>> pulse = GaussianPulse(freq0=200e12, fwidth=20e12)
    >>> mode_spec = ModeSpec(target_neff=2.)
    >>> mode_source = ModeSource(
    ...     size=(10,10,0),
    ...     source_time=pulse,
    ...     mode_spec=mode_spec,
    ...     mode_index=1,
    ...     direction='-')

    See Also
    --------

    :class:`tidy3d.plugins.mode.ModeSolver`:
        Interface for solving electromagnetic eigenmodes in a 2D plane with translational invariance in the third dimension.

    **Notebooks:**
        * `Waveguide Y junction <../../notebooks/YJunction.html>`_
        * `90 degree optical hybrid <../../notebooks/90OpticalHybrid.html>`_

    **Lectures:**
        * `Prelude to Integrated Photonics Simulation: Mode Injection <https://www.flexcompute.com/fdtd101/Lecture-4-Prelude-to-Integrated-Photonics-Simulation-Mode-Injection/>`_
    """

    mode_index: NonNegativeInt = Field(
        0,
        title="Mode Index",
        description="Index into the collection of modes returned by mode solver. "
        " Specifies which mode to inject using this source. "
        "If larger than ``mode_spec.num_modes``, "
        "``num_modes`` in the solver will be set to ``mode_index + 1``.",
    )


""" Angled Field Sources one can use. """


class AbstractAngularSpec(Tidy3dBaseModel, ABC):
    """Abstract base for defining angular variability specifications for plane waves."""


class FixedInPlaneKSpec(AbstractAngularSpec):
    """Plane wave is injected such that its in-plane wavevector is constant. That is,
    the injected field satisfies Bloch boundary conditions and its propagation direction is
    frequency dependent.
    """


class FixedAngleSpec(AbstractAngularSpec):
    """Plane wave is injected such that its propagation direction is frequency independent.
    When using this option boundary conditions in tangential directions must be set to periodic.
    """


class PlaneWave(AngledFieldSource, PlanarSource, BroadbandSource):
    """Uniform current distribution on an infinite extent plane. One element of size must be zero.

    Notes
    -----

        For oblique incidence, there are two possible settings: fixed in-plane k-vector and fixed-angle mode.
        The first requires Bloch periodic boundary conditions, and the incidence angle is exact only at the central wavelength.
        The latter requires periodic boundary conditions and maintains a constant propagation angle over a broadband spectrum.
        For more information and important notes, see this example: `Broadband PlaneWave With Constant Oblique Incident Angle <https://docs.simulation.cloud/projects/tidy3d/en/latest/notebooks/BroadbandPlaneWaveWithConstantObliqueIncidentAngle.html>`_.

    Example
    -------
    >>> from tidy3d import GaussianPulse
    >>> pulse = GaussianPulse(freq0=200e12, fwidth=20e12)
    >>> pw_source = PlaneWave(size=(inf,0,inf), source_time=pulse, pol_angle=0.1, direction='+')

    See Also
    --------

    **Notebooks:**
        * `How to troubleshoot a diverged FDTD simulation <../../notebooks/DivergedFDTDSimulation.html>`_

    **Lectures:**
        * `Using FDTD to Compute a Transmission Spectrum <https://www.flexcompute.com/fdtd101/Lecture-2-Using-FDTD-to-Compute-a-Transmission-Spectrum/>`__
    """

    angular_spec: Union[FixedInPlaneKSpec, FixedAngleSpec] = Field(
        default_factory=FixedInPlaneKSpec,
        title="Angular Dependence Specification",
        description="Specification of plane wave propagation direction dependence on wavelength.",
        discriminator=TYPE_TAG_STR,
    )

    num_freqs: int = Field(
        3,
        title="Number of Frequency Points",
        description="Number of points used to approximate the frequency dependence of the injected "
        "field. Default is 3, which should cover even very broadband plane waves. For simulations "
        "which are not very broadband and the source is very large (e.g. metalens simulations), "
        "decreasing the value to 1 may lead to a speed up in the preprocessing.",
        ge=1,
        le=20,
    )

    broadband_method: Literal["chebyshev"] = Field(
        "chebyshev",
        title="Broadband Method",
        description="PlaneWave only supports the Chebyshev broadband method.",
    )

    @cached_property
    def _is_fixed_angle(self) -> bool:
        """Whether the plane wave is at a fixed non-zero angle."""
        return isinstance(self.angular_spec, FixedAngleSpec) and self.angle_theta != 0.0

    @cached_property
    def frequency_grid(self) -> NDArray:
        """A Chebyshev grid used to approximate frequency dependence."""
        if self.num_freqs == 1:
            return np.array([self.source_time._freq0])
        freq_min, freq_max = self.source_time.frequency_range_sigma(sigma=CHEB_GRID_WIDTH)
        if not self._is_fixed_angle:
            # For frequency-dependent angles (constat in-plane k), truncate minimum frequency at
            # the critical frequency of glancing incidence
            f_crit = self.source_time._freq0 * np.sin(self.angle_theta)
            freq_min = max(freq_min, f_crit * CRITICAL_FREQUENCY_FACTOR)
        return self._chebyshev_freq_grid(freq_min, freq_max)

    @model_validator(mode="after")
    def _validate_source_frequency_range(self) -> Self:
        """Error if a broadband plane wave with constant in-plane k is defined such that
        the source frequency range is entirely below ``f_crit * CRITICAL_FREQUENCY_FACTOR."""
        if self._is_fixed_angle or self.num_freqs == 1:
            return self
        freq_min, freq_max = self.source_time.frequency_range_sigma(sigma=CHEB_GRID_WIDTH)
        f_crit = self.source_time._freq0 * np.sin(self.angle_theta)
        if f_crit * CRITICAL_FREQUENCY_FACTOR > freq_max:
            raise SetupError(
                "Broadband plane wave source defined with a bandwidth too close to the critical "
                "frequency of oblique incidence. Increase the source bandwidth, or disable the "
                "broadband handling by setting 'num_freqs' to 1."
            )
        return self


def _source_remap_background_index(derivative_info: DerivativeInfo, freqs: np.ndarray) -> NDArray:
    """Background index for Gaussian-like source remap."""
    if derivative_info.source_background_index is None:
        raise AdjointError(
            "Gaussian-like source derivative remap requires 'source_background_index' "
            "in DerivativeInfo, but it was not provided."
        )
    sampled = derivative_info.source_background_index.sel(f=freqs, method="nearest")
    return np.asarray(sampled.values, dtype=complex).reshape(-1)


def _slice_field_map_at_frequency(
    field_map: Optional[dict[str, ScalarFieldDataArray]], freq: float
) -> dict[str, ScalarFieldDataArray]:
    """Slice a field-component map to a single frequency."""
    if not field_map:
        return {}
    return {name: data_array.sel(f=[freq]) for name, data_array in field_map.items()}


def _slice_derivative_info_at_frequency(
    derivative_info: DerivativeInfo, freq: float
) -> DerivativeInfo:
    """Create a per-frequency ``DerivativeInfo`` view for Gaussian-like remap."""
    source_background_index = derivative_info.source_background_index
    if source_background_index is None:
        raise AdjointError(
            "Gaussian-like source derivative remap requires 'source_background_index' "
            "in DerivativeInfo, but it was not provided."
        )
    return derivative_info.updated_copy(
        E_adj=_slice_field_map_at_frequency(derivative_info.E_adj, freq),
        H_adj=_slice_field_map_at_frequency(derivative_info.H_adj, freq),
        frequencies=np.asarray([freq], dtype=float),
        source_background_index=source_background_index.sel(f=[freq], method="nearest"),
    )


def _accumulate_derivative_value(
    derivative_map: AutogradFieldMap,
    field_path: PathType,
    value: Any,
) -> None:
    """Accumulate one per-frequency derivative contribution into ``derivative_map``."""
    if field_path not in derivative_map:
        derivative_map[field_path] = value
        return
    prev_value = derivative_map[field_path]
    if isinstance(prev_value, tuple) or isinstance(value, tuple):
        summed = np.asarray(prev_value, dtype=float) + np.asarray(value, dtype=float)
        derivative_map[field_path] = tuple(summed.tolist())
        return
    derivative_map[field_path] = prev_value + value


def _build_gaussian_like_field_dataset(
    source: AbstractGaussianBeam, derivative_info: DerivativeInfo
) -> FieldDataset:
    """Build analytic beam fields as a ``FieldDataset`` for source VJP chaining."""
    from tidy3d.components.data.data_array import ScalarFieldDataArray

    source_center = np.asarray(source.center, dtype=float)
    freqs = np.asarray(derivative_info.frequencies, dtype=float).reshape(-1)
    if freqs.size != 1:
        raise ValueError(
            "Gaussian-like remap dataset construction expects a single frequency slice, got "
            f"{freqs.size} frequencies."
        )
    background_n = _source_remap_background_index(derivative_info, freqs)
    field_arrays = {}
    for component in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        field_map = derivative_info.E_adj if component.startswith("E") else derivative_info.H_adj
        component_data = field_map[component]
        # Keep mock dataset coordinates source-relative to match ``CustomFieldSource`` convention.
        # In source VJP interpolation these coordinates are shifted back to simulation coordinates
        # via ``+ center`` (see ``transpose_interp_field_to_dataset``), so this is not a double shift.
        coords = {
            dim: np.asarray(component_data.coords[dim].data, dtype=float) - source_center[axis]
            for axis, dim in enumerate("xyz")
        }
        coords["f"] = freqs
        component_field = _compute_component_field_on_coords(
            source=source,
            component=component,
            coords=coords,
            background_n=background_n,
            center=(0.0, 0.0, 0.0),
            normalize=True,
        )
        field_arrays[component] = ScalarFieldDataArray(component_field, coords=coords)

    return FieldDataset(**field_arrays)


def _compute_dataset_vjp_weights(
    source: AbstractGaussianBeam, derivative_info: DerivativeInfo, field_dataset: FieldDataset
) -> dict[tuple[str, str], np.ndarray]:
    """Compute complex source dataset cotangents by reusing CustomFieldSource VJP."""
    mock_source = CustomFieldSource(
        center=tuple(float(val) for val in source.center),
        size=tuple(float(val) for val in source.size),
        source_time=source.source_time,
        field_dataset=field_dataset,
    )
    dataset_paths = [("field_dataset", component) for component in field_dataset.field_components]
    derivative_info_dataset = derivative_info.updated_copy(paths=dataset_paths)
    dataset_weights = mock_source._compute_derivatives(derivative_info_dataset)
    if source.direction == "-":
        # Backward-propagating Gaussian-like beams require an extra global sign
        # in the dataset cotangent remap to align with source-direction convention.
        dataset_weights = negate_vjp_map(dataset_weights)
    return dataset_weights


def _autograd_param_vjp(
    source: AbstractGaussianBeam,
    derivative_info: DerivativeInfo,
    path: PathType,
    dataset_weights: dict[tuple[str, str], np.ndarray],
    field_dataset: FieldDataset,
) -> float:
    """Map dataset VJP weights to one source parameter via autograd VJP."""
    import autograd.numpy as anp
    from autograd import grad

    value0 = source.get_param(path)
    base_center = tuple(float(val) for val in source.center)
    weights = {
        key: anp.asarray(val) for key, val in dataset_weights.items() if key[0] == "field_dataset"
    }

    component_grids = {
        component: {
            dim: np.asarray(data_array.coords[dim].data, dtype=float)
            for dim in ("x", "y", "z", "f")
        }
        for component, data_array in field_dataset.field_components.items()
    }
    component_background_n = {
        component: _source_remap_background_index(derivative_info, grid["f"])
        for component, grid in component_grids.items()
    }

    def _helper_component_field(
        params: dict[str, Any], *, normalize: bool, component: str
    ) -> np.ndarray:
        grid = component_grids[component]
        shift_x = params["center"][0] - base_center[0]
        shift_y = params["center"][1] - base_center[1]
        shift_z = params["center"][2] - base_center[2]
        return _compute_component_field_on_coords(
            source=source,
            component=component,
            coords=grid,
            background_n=component_background_n[component],
            # Dataset coordinates are source-relative to the base source center.
            # Center shifts are represented as offsets in this relative frame.
            center=(shift_x, shift_y, shift_z),
            normalize=normalize,
            params=params,
        )

    def objective(param_value: Any) -> Any:
        params = source.get_param_state(path, param_value)
        total = 0.0 + 0.0j
        for component in field_dataset.field_components:
            weight = weights.get(("field_dataset", component))
            if weight is None:
                continue
            component_field = _helper_component_field(params, normalize=True, component=component)
            total += anp.sum(weight * component_field)
        return anp.real(total)

    return float(grad(objective)(value0))


def _make_beam_pose(
    source: AbstractGaussianBeam,
    *,
    center: tuple[float, float, float],
    params: Optional[dict[str, Any]] = None,
) -> BeamPose:
    """Construct a ``BeamPose`` from source defaults with optional traced overrides."""
    from tidy3d.components import beam as beam_module

    params = {} if params is None else params
    return beam_module.BeamPose(
        injection_axis=int(source.injection_axis),
        direction=params.get("direction", source.direction),
        angle_theta=params.get("angle_theta", source.angle_theta),
        angle_phi=params.get("angle_phi", source.angle_phi),
        pol_angle=params.get("pol_angle", source.pol_angle),
        center=center,
    )


def _compute_component_field_on_coords(
    source: AbstractGaussianBeam,
    *,
    component: str,
    coords: dict[str, np.ndarray],
    background_n: np.ndarray,
    center: tuple[float, float, float],
    normalize: bool,
    params: Optional[dict[str, Any]] = None,
) -> np.ndarray:
    """Compute one beam field component on explicit coordinate arrays."""
    from tidy3d.components import beam as beam_module

    pose = _make_beam_pose(source, center=center, params=params)
    grid = beam_module.BeamGrid(
        freqs=np.asarray(coords["f"], dtype=float),
        background_n=background_n,
        x=np.asarray(coords["x"], dtype=float),
        y=np.asarray(coords["y"], dtype=float),
        z=np.asarray(coords["z"], dtype=float),
    )
    params = {} if params is None else params
    beam_param_overrides = {
        key: params[key]
        for key in ("waist_radius", "waist_distance", "waist_sizes", "waist_distances")
        if key in params and params[key] is not None
    }
    fields = source.compute_beam_fields_on_grid(
        pose=pose,
        grid=grid,
        normalize=normalize,
        params=beam_param_overrides,
    )
    return fields[component]


def _compute_gaussian_like_derivatives(
    source: AbstractGaussianBeam, derivative_info: DerivativeInfo
) -> AutogradFieldMap:
    """Compute Gaussian-like source derivatives through dataset-remap chain rule.

    Chain rule:
    1) analytic Gaussian-like source -> mock ``FieldDataset``
    2) reuse ``CustomFieldSource`` VJP for dataset cotangents
    3) map dataset cotangents back to source parameters
    """
    derivative_map: AutogradFieldMap = {}
    validated_paths = []

    for field_path in derivative_info.paths:
        field_path = tuple(field_path)
        field_path = source.validate_traced_path(field_path)
        validated_paths.append(field_path)

    param_paths, center_paths = split_source_paths(validated_paths, primary_roots=None)

    freqs = np.asarray(derivative_info.frequencies, dtype=float).reshape(-1)
    for freq in freqs:
        derivative_info_freq = _slice_derivative_info_at_frequency(derivative_info, float(freq))
        field_dataset = _build_gaussian_like_field_dataset(
            source, derivative_info=derivative_info_freq
        )
        dataset_weights = _compute_dataset_vjp_weights(source, derivative_info_freq, field_dataset)

        if center_paths:
            mock_source = CustomFieldSource(
                center=tuple(float(val) for val in source.center),
                size=tuple(float(val) for val in source.size),
                source_time=source.source_time,
                field_dataset=field_dataset,
            )
            derivative_info_center = derivative_info_freq.updated_copy(
                paths=center_paths,
                bounds=derivative_info_freq.bounds,
                bounds_intersect=derivative_info_freq.bounds_intersect,
            )
            center_derivatives = mock_source._compute_derivatives(derivative_info_center)
            if source.direction == "-":
                center_derivatives = negate_vjp_map(center_derivatives)
            for field_path, value in center_derivatives.items():
                _accumulate_derivative_value(derivative_map, field_path, value)

        for field_path in param_paths:
            value = _autograd_param_vjp(
                source,
                derivative_info_freq,
                field_path,
                dataset_weights=dataset_weights,
                field_dataset=field_dataset,
            )
            _accumulate_derivative_value(derivative_map, field_path, value)

    return derivative_map


class AbstractGaussianBeam(AngledFieldSource, PlanarSource, BroadbandSource, ABC):
    """Shared base class for Gaussian-like planar field sources."""

    angle_theta: TracedFloat = Field(
        0.0,
        title="Polar Angle",
        description="Polar angle of the propagation axis from the injection axis.",
        json_schema_extra={"units": RADIAN},
    )

    angle_phi: TracedFloat = Field(
        0.0,
        title="Azimuth Angle",
        description="Azimuth angle of the propagation axis in the plane orthogonal to the "
        "injection axis.",
        json_schema_extra={"units": RADIAN},
    )

    pol_angle: TracedFloat = Field(
        0.0,
        title="Polarization Angle",
        description="Specifies the angle between the electric field polarization of the "
        "source and the plane defined by the injection axis and the propagation axis (rad). "
        "``pol_angle=0`` (default) specifies P polarization, "
        "while ``pol_angle=np.pi/2`` specifies S polarization. "
        "At normal incidence when S and P are undefined, ``pol_angle=0`` defines: "
        "- ``Ey`` polarization for propagation along ``x``."
        "- ``Ex`` polarization for propagation along ``y``."
        "- ``Ex`` polarization for propagation along ``z``.",
        json_schema_extra={"units": RADIAN},
    )

    num_freqs: int = Field(
        1,
        title="Number of Frequency Points",
        description="Number of points used to approximate the frequency dependence of the injected "
        "field. For broadband, angled Gaussian beams it is advisable to check the beam propagation "
        "in an empty simulation to ensure there are no injection artifacts when 'num_freqs' > 1. "
        "Note that larger values of 'num_freqs' could spread out the source time signal and "
        "introduce numerical noise, or prevent timely field decay.",
        ge=1,
        le=50,
    )

    def _compute_derivatives(self, derivative_info: DerivativeInfo) -> AutogradFieldMap:
        """Compute derivatives for Gaussian-like beam source parameters."""
        return _compute_gaussian_like_derivatives(self, derivative_info)

    @property
    def _common_scalar_roots(self) -> set[str]:
        return {"angle_theta", "angle_phi", "pol_angle"}

    @property
    @abstractmethod
    def _beam_scalar_roots(self) -> set[str]:
        """Traced scalar roots specific to this Gaussian-like source."""

    @property
    @abstractmethod
    def _beam_tuple_roots(self) -> set[str]:
        """Traced tuple-valued roots specific to this Gaussian-like source."""

    @abstractmethod
    def _beam_param_state(self) -> dict[str, Any]:
        """Return source-parameter state used by autograd remap objective."""

    @abstractmethod
    def compute_beam_fields_on_grid(
        self,
        *,
        pose: BeamPose,
        grid: BeamGrid,
        normalize: bool,
        params: Optional[dict[str, Any]] = None,
    ) -> dict[str, np.ndarray]:
        """Compute Gaussian-like fields on a component grid."""

    def get_param(self, path: PathType) -> float:
        """Extract scalar parameter value from a traced source path."""
        root = path[0]
        if root == "center":
            return float(self.center[path[1]])
        if root in self._beam_tuple_roots:
            return float(getattr(self, root)[path[1]])
        return float(getattr(self, root))

    def get_param_state(self, path: PathType, value: Any) -> dict[str, Any]:
        """Return parameter state dictionary after overriding one traced parameter."""
        params = {
            "center": tuple(self.center),
            "angle_theta": self.angle_theta,
            "angle_phi": self.angle_phi,
            "pol_angle": self.pol_angle,
            "direction": self.direction,
            **self._beam_param_state(),
        }
        root = path[0]
        if root == "center":
            center = list(params["center"])
            center[path[1]] = value
            params["center"] = tuple(center)
        elif root in self._beam_tuple_roots:
            values = list(params[root])
            values[path[1]] = value
            params[root] = tuple(values)
        else:
            params[root] = value
        return params

    def validate_traced_path(self, field_path: PathType) -> PathType:
        """Validate traced Gaussian-like source paths."""
        if not field_path:
            raise ValueError(f"Empty traced source path encountered in '{type(self).__name__}'.")

        root = field_path[0]
        scalar_allowed = self._common_scalar_roots | self._beam_scalar_roots
        if root in scalar_allowed or root in self._beam_tuple_roots:
            return field_path

        if root == "center":
            try:
                validate_no_zero_dim_center_paths(
                    [field_path],
                    source_size=tuple(self.size),
                    source_name=type(self).__name__,
                )
            except AdjointError as err:
                raise ValueError(str(err)) from err
            return field_path

        raise ValueError(
            f"Unsupported traced source path '{field_path}' for '{type(self).__name__}'. "
            "Supported parameters are beam waists, beam distances, angle_theta, angle_phi, pol_angle, "
            "and lateral center components."
        )


class GaussianBeam(AbstractGaussianBeam):
    """Gaussian distribution on finite extent plane.

    Example
    -------
    >>> from tidy3d import GaussianPulse
    >>> pulse = GaussianPulse(freq0=200e12, fwidth=20e12)
    >>> gauss = GaussianBeam(
    ...     size=(0,3,3),
    ...     source_time=pulse,
    ...     pol_angle=np.pi / 2,
    ...     direction='+',
    ...     waist_radius=1.0)



    See Also
    --------

    **Notebooks**:
        * `Inverse taper edge coupler <../../notebooks/EdgeCoupler.html>`_
    """

    waist_radius: TracedPositiveFloat = Field(
        1.0,
        title="Waist Radius",
        description="Radius of the beam at the waist.",
        json_schema_extra={"units": MICROMETER},
    )

    waist_distance: TracedFloat = Field(
        0.0,
        title="Waist Distance",
        description="Distance from the beam waist along the propagation direction. "
        "A positive value places the waist behind the source plane (toward the negative normal axis). "
        "A negative value places the waist in front of the source plane (toward the positive normal axis). "
        "This definition is independent of the ``direction`` parameter, ensuring consistent waist "
        "positioning for both forward- and backward-propagating beams. "
        "For an angled source, the distance is measured along the rotated propagation direction.",
        json_schema_extra={"units": MICROMETER},
    )

    _backward_waist_warning = warn_backward_waist_distance("waist_distance")

    @property
    def _beam_scalar_roots(self) -> set[str]:
        return {"waist_radius", "waist_distance"}

    @property
    def _beam_tuple_roots(self) -> set[str]:
        return set()

    def _beam_param_state(self) -> dict[str, Any]:
        return {
            "waist_radius": self.waist_radius,
            "waist_distance": self.waist_distance,
        }

    def compute_beam_fields_on_grid(
        self,
        *,
        pose: BeamPose,
        grid: BeamGrid,
        normalize: bool,
        params: Optional[dict[str, Any]] = None,
    ) -> dict[str, np.ndarray]:
        """Compute Gaussian beam fields from compact pose/grid context."""
        from tidy3d.components import beam as beam_module

        params = {} if params is None else params
        return beam_module.gaussian_like_fields_on_grid_from_context(
            pose=pose,
            grid=grid,
            waist_radius=params.get("waist_radius", self.waist_radius),
            waist_distance=params.get("waist_distance", self.waist_distance),
            normalize=normalize,
        )


class AstigmaticGaussianBeam(AbstractGaussianBeam):
    """The simple astigmatic Gaussian distribution allows
    both an elliptical intensity profile and different waist locations for the two principal axes
    of the ellipse. When equal waist sizes and equal waist distances are specified in the two
    directions, this source becomes equivalent to :class:`GaussianBeam`.

    Notes
    -----

        This class implements the simple astigmatic Gaussian beam described in _`[1]`.

        **References**:

        .. [1] Kochkina et al., Applied Optics, vol. 52, issue 24, 2013.

    Example
    -------
    >>> from tidy3d import GaussianPulse
    >>> pulse = GaussianPulse(freq0=200e12, fwidth=20e12)
    >>> gauss = AstigmaticGaussianBeam(
    ...     size=(0,3,3),
    ...     source_time=pulse,
    ...     pol_angle=np.pi / 2,
    ...     direction='+',
    ...     waist_sizes=(1.0, 2.0),
    ...     waist_distances = (3.0, 4.0))
    """

    waist_sizes: tuple[TracedPositiveFloat, TracedPositiveFloat] = Field(
        (1.0, 1.0),
        title="Waist sizes",
        description="Size of the beam at the waist in the local x and y directions.",
        json_schema_extra={"units": MICROMETER},
    )

    waist_distances: tuple[TracedFloat, TracedFloat] = Field(
        (0.0, 0.0),
        title="Waist distances",
        description="Distance to the beam waist along the propagation direction "
        "for the waist sizes in the local x and y directions. "
        "Positive values place the waist behind the source plane (toward the negative normal axis); "
        "negative values place the waist in front of the source plane. "
        "This definition is independent of the ``direction`` parameter.",
        json_schema_extra={"units": MICROMETER},
    )

    backward_waist_warning = warn_backward_waist_distance("waist_distances")

    @property
    def _beam_scalar_roots(self) -> set[str]:
        return set()

    @property
    def _beam_tuple_roots(self) -> set[str]:
        return {"waist_sizes", "waist_distances"}

    def _beam_param_state(self) -> dict[str, Any]:
        return {
            "waist_sizes": tuple(self.waist_sizes),
            "waist_distances": tuple(self.waist_distances),
        }

    def compute_beam_fields_on_grid(
        self,
        *,
        pose: BeamPose,
        grid: BeamGrid,
        normalize: bool,
        params: Optional[dict[str, Any]] = None,
    ) -> dict[str, np.ndarray]:
        """Compute astigmatic Gaussian beam fields from compact pose/grid context."""
        from tidy3d.components import beam as beam_module

        params = {} if params is None else params
        return beam_module.astigmatic_gaussian_like_fields_on_grid_from_context(
            pose=pose,
            grid=grid,
            waist_sizes=tuple(params.get("waist_sizes", self.waist_sizes)),
            waist_distances=tuple(params.get("waist_distances", self.waist_distances)),
            normalize=normalize,
        )


class TFSF(AngledFieldSource, VolumeSource, BroadbandSource):
    """Total-field scattered-field (TFSF) source that can inject a plane wave in a finite region.

    Notes
    -----

        The TFSF source injects :math:`1 W` of power per :math:`\\mu m^2` of source area along the :attr:`injection_axis`.
        Hence, the normalization for the incident field is :math:`|E_0|^2 = \\frac{2}{c\\epsilon_0}`, for any source size.
        Note that in the case of angled incidence, the same power is injected along the source's :attr:`injection_axis`,
        and not the propagation direction. This allows computing scattering and absorption cross-sections
        without the need for additional normalization.

        The TFSF source allows specifying a box region into which a plane wave is injected. Fields inside this region
        can be interpreted as the superposition of the incident field and the scattered field due to any scatterers
        present in the simulation domain. The fields at the edges of the TFSF box are modified at each time step such
        that the incident field is cancelled out, so that all fields outside the TFSF box are scattered fields only.
        This is useful in scenarios where one is interested in computing scattered fields only, for example when
        computing scattered cross-sections of various objects.

        It is important to note that when a non-uniform grid is used in the directions transverse to the
        :attr:`injection_axis` of the TFSF source, the suppression of the incident field outside the TFSF box may not be as
        close to zero as in the case of a uniform grid. Because of this, a warning may be issued when nonuniform grid
        TFSF setup is detected. In some cases, however, the accuracy may be only weakly affected, and the warnings
        can be ignored.

    See Also
    --------

    **Notebooks**:
        * `Defining a total-field scattered-field (TFSF) plane wave source <../../notebooks/TFSF.html>`_
        * `Nanoparticle Scattering <../../notebooks/PlasmonicNanoparticle.html>`_: To force a uniform grid in the TFSF region and avoid the warnings, a mesh override structure can be used as illustrated here.
    """

    num_freqs: int = Field(
        1,
        title="Number of Frequency Points",
        description="Number of points used to approximate the frequency dependence of the injected "
        "field. A Chebyshev interpolation is used, thus, only a small number of points is "
        "typically sufficient to obtain converged results.",
        ge=1,
        le=20,
    )

    broadband_method: Literal["chebyshev"] = Field(
        "chebyshev",
        title="Broadband Method",
        description="TFSF only supports the Chebyshev broadband method.",
    )

    injection_axis: Axis = Field(
        title="Injection Axis",
        description="Specifies the injection axis. The plane of incidence is defined via this "
        "``injection_axis`` and the ``direction``. The popagation axis is defined with respect "
        "to the ``injection_axis`` by ``angle_theta`` and ``angle_phi``.",
    )

    use_colocated_integration: bool = Field(
        True,
        title="Use Colocated Integration",
        description="If ``True`` (default), source power normalization uses fields "
        "interpolated to grid cell boundaries (colocated). If ``False``, uses fields at "
        "native Yee grid positions (non-colocated). Should match the "
        "``use_colocated_integration`` setting on monitors for consistent power normalization. "
        "Experimental feature that can give improved accuracy by avoiding interpolation of "
        "fields to Yee cell positions for integration.",
    )

    @cached_property
    def _injection_axis(self) -> Axis:
        """Injection axis of the source."""
        return self.injection_axis

    @cached_property
    def injection_plane_center(self) -> Coordinate:
        """Center of the injection plane."""
        sign = 1 if self.direction == "-" else -1
        center = list(self.center)
        size = [0 if val == inf else val for val in self.size]
        center[self.injection_axis] += sign * size[self.injection_axis] / 2
        return tuple(center)

    def plot(
        self,
        x: Optional[float] = None,
        y: Optional[float] = None,
        z: Optional[float] = None,
        ax: Ax = None,
        **patch_kwargs: Any,
    ) -> Ax:
        # call Source.plot but with the base of the arrow centered on the injection plane
        patch_kwargs["arrow_base"] = self.injection_plane_center
        ax = Source.plot(self, x=x, y=y, z=z, ax=ax, **patch_kwargs)
        return ax
