"""Defines electric field sources for injecting light into simulation."""

from __future__ import annotations

from abc import ABC
from typing import Optional, Tuple, Union

import numpy as np
import pydantic.v1 as pydantic

from ...constants import GLANCING_CUTOFF, MICROMETER, RADIAN, inf
from ...exceptions import SetupError
from ...log import log
from ..base import Tidy3dBaseModel, cached_property, skip_if_fields_missing
from ..data.dataset import FieldDataset
from ..data.validators import validate_can_interpolate, validate_no_nans
from ..mode_spec import ModeSpec
from ..types import (
    TYPE_TAG_STR,
    Ax,
    Axis,
    Coordinate,
    Direction,
)
from ..validators import (
    assert_plane,
    assert_single_freq_in_range,
    assert_volumetric,
    warn_if_dataset_none,
)
from .base import Source

# width of Chebyshev grid used for broadband sources (in units of pulse width)
CHEB_GRID_WIDTH = 1.5
# Number of frequencies in a broadband source above which to issue a warning
WARN_NUM_FREQS = 20
# For broadband plane waves with constan in-plane k, the Chebyshev grid is truncated at
# ``CRITICAL_FREQUENCY_FACTOR * f_crit``, where ``f_crit`` is the critical frequency
# (oblique propagation).
CRITICAL_FREQUENCY_FACTOR = 1.15


class FieldSource(Source, ABC):
    """A Source defined by the desired E and/or H fields."""


""" Field Sources can be defined either on a (1) surface or (2) volume. Defines injection_axis """


class PlanarSource(Source, ABC):
    """A source defined on a 2D plane."""

    _plane_validator = assert_plane()

    @cached_property
    def injection_axis(self):
        """Injection axis of the source."""
        return self._injection_axis

    @cached_property
    def _injection_axis(self):
        """Injection axis of the source."""
        return self.size.index(0.0)


class VolumeSource(Source, ABC):
    """A source defined in a 3D :class:`Box`."""

    _volume_validator = assert_volumetric()


""" Field Sources require more specification, for now, they all have a notion of a direction."""


class DirectionalSource(FieldSource, ABC):
    """A Field source that propagates in a given direction."""

    direction: Direction = pydantic.Field(
        ...,
        title="Direction",
        description="Specifies propagation in the positive or negative direction of the injection "
        "axis.",
    )

    @cached_property
    def _dir_vector(self) -> Tuple[float, float, float]:
        """Returns a vector indicating the source direction for arrow plotting, if not None."""
        if self._injection_axis is None:
            return None
        dir_vec = [0, 0, 0]
        dir_vec[int(self._injection_axis)] = 1 if self.direction == "+" else -1
        return dir_vec


class BroadbandSource(Source, ABC):
    """A source with frequency dependent field distributions."""

    # Default as for analytic beam sources; overwrriten for ModeSource below
    num_freqs: int = pydantic.Field(
        3,
        title="Number of Frequency Points",
        description="Number of points used to approximate the frequency dependence of the injected "
        "field. Default is 3, which should cover even very broadband sources. For simulations "
        "which are not very broadband and the source is very large (e.g. metalens simulations), "
        "decreasing the value to 1 may lead to a speed up in the preprocessing.",
        ge=1,
        le=20,
    )

    @cached_property
    def frequency_grid(self) -> np.ndarray:
        """A Chebyshev grid used to approximate frequency dependence."""
        freq_min, freq_max = self.source_time.frequency_range(num_fwidth=CHEB_GRID_WIDTH)
        return self._chebyshev_freq_grid(freq_min, freq_max)

    def _chebyshev_freq_grid(self, freq_min, freq_max):
        """A Chebyshev grid based on a minimum and maximum frequency."""
        freq_avg = 0.5 * (freq_min + freq_max)
        freq_diff = 0.5 * (freq_max - freq_min)
        uni_points = (2 * np.arange(self.num_freqs) + 1) / (2 * self.num_freqs)
        cheb_points = np.cos(np.pi * np.flip(uni_points))
        return freq_avg + freq_diff * cheb_points

    @pydantic.validator("num_freqs", always=True, allow_reuse=True)
    def _warn_if_large_number_of_freqs(cls, val):
        """Warn if a large number of frequency points is requested."""

        if val is None:
            return val

        if val >= WARN_NUM_FREQS:
            log.warning(
                f"A large number ({val}) of frequency points is used in a broadband source. "
                "This can lead to solver slow-down and increased cost, and even introduce "
                "numerical noise. This may become a hard limit in future Tidy3D versions.",
                custom_loc=["num_freqs"],
            )

        return val


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

    field_dataset: Optional[FieldDataset] = pydantic.Field(
        ...,
        title="Field Dataset",
        description=":class:`.FieldDataset` containing the desired frequency-domain "
        "fields patterns to inject. At least one tangential field component must be specified.",
    )

    _no_nans_dataset = validate_no_nans("field_dataset")
    _field_dataset_none_warning = warn_if_dataset_none("field_dataset")
    _field_dataset_single_freq = assert_single_freq_in_range("field_dataset")
    _can_interpolate = validate_can_interpolate("field_dataset")

    @pydantic.validator("field_dataset", always=True)
    @skip_if_fields_missing(["size"])
    def _tangential_component_defined(cls, val: FieldDataset, values: dict) -> FieldDataset:
        """Assert that at least one tangential field component is provided."""
        if val is None:
            return val
        size = values.get("size")
        normal_axis = size.index(0.0)
        _, (cmp1, cmp2) = cls.pop_axis("xyz", axis=normal_axis)
        for field in "EH":
            for cmp_name in (cmp1, cmp2):
                tangential_field = field + cmp_name
                if tangential_field in val.field_components:
                    return val
        raise SetupError("No tangential field found in the suppled 'field_dataset'.")


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

    angle_theta: float = pydantic.Field(
        0.0,
        title="Polar Angle",
        description="Polar angle of the propagation axis from the injection axis.",
        units=RADIAN,
    )

    angle_phi: float = pydantic.Field(
        0.0,
        title="Azimuth Angle",
        description="Azimuth angle of the propagation axis in the plane orthogonal to the "
        "injection axis.",
        units=RADIAN,
    )

    pol_angle: float = pydantic.Field(
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
        units=RADIAN,
    )

    @pydantic.validator("angle_theta", allow_reuse=True, always=True)
    def glancing_incidence(cls, val):
        """Warn if close to glancing incidence."""
        if np.abs(np.pi / 2 - val) < GLANCING_CUTOFF:
            log.warning(
                "Angled source propagation axis close to glancing angle. "
                "For best results, switch the injection axis.",
                custom_loc=["angle_theta"],
            )
        return val

    @cached_property
    def _dir_vector(self) -> Tuple[float, float, float]:
        """Source direction normal vector in cartesian coordinates."""

        # Propagation vector assuming propagation along z
        radius = 1.0 if self.direction == "+" else -1.0
        dx = radius * np.cos(self.angle_phi) * np.sin(self.angle_theta)
        dy = radius * np.sin(self.angle_phi) * np.sin(self.angle_theta)
        dz = radius * np.cos(self.angle_theta)

        # Move to original injection axis
        return self.unpop_axis(dz, (dx, dy), axis=self._injection_axis)

    @cached_property
    def _pol_vector(self) -> Tuple[float, float, float]:
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


class ModeSource(DirectionalSource, PlanarSource, BroadbandSource):
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

        .. image:: ../../_static/img/mode_vs_dipole_source.png

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

    mode_spec: ModeSpec = pydantic.Field(
        ModeSpec(),
        title="Mode Specification",
        description="Parameters to feed to mode solver which determine modes measured by monitor.",
    )

    mode_index: pydantic.NonNegativeInt = pydantic.Field(
        0,
        title="Mode Index",
        description="Index into the collection of modes returned by mode solver. "
        " Specifies which mode to inject using this source. "
        "If larger than ``mode_spec.num_modes``, "
        "``num_modes`` in the solver will be set to ``mode_index + 1``.",
    )

    num_freqs: int = pydantic.Field(
        1,
        title="Number of Frequency Points",
        description="Number of points used to approximate the frequency dependence of injected "
        "field. A Chebyshev interpolation is used, thus, only a small number of points, i.e., less "
        "than 20, is typically sufficient to obtain converged results.",
        ge=1,
        le=99,
    )

    @cached_property
    def angle_theta(self):
        """Polar angle of propagation."""
        return self.mode_spec.angle_theta

    @cached_property
    def angle_phi(self):
        """Azimuth angle of propagation."""
        return self.mode_spec.angle_phi

    @cached_property
    def _dir_vector(self) -> Tuple[float, float, float]:
        """Source direction normal vector in cartesian coordinates."""
        radius = 1.0 if self.direction == "+" else -1.0
        dx = radius * np.cos(self.angle_phi) * np.sin(self.angle_theta)
        dy = radius * np.sin(self.angle_phi) * np.sin(self.angle_theta)
        dz = radius * np.cos(self.angle_theta)
        return self.unpop_axis(dz, (dx, dy), axis=self._injection_axis)

    @cached_property
    def _bend_axis(self) -> Axis:
        if self.mode_spec.bend_radius is None:
            return None
        in_plane = [0, 0]
        in_plane[self.mode_spec.bend_axis] = 1
        direction = self.unpop_axis(0, in_plane, axis=self.injection_axis)
        return direction.index(1)


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

    angular_spec: Union[FixedInPlaneKSpec, FixedAngleSpec] = pydantic.Field(
        FixedInPlaneKSpec(),
        title="Angular Dependence Specification",
        description="Specification of plane wave propagation direction dependence on wavelength.",
        discriminator=TYPE_TAG_STR,
    )

    @cached_property
    def _is_fixed_angle(self) -> bool:
        """Whether the plane wave is at a fixed non-zero angle."""
        return isinstance(self.angular_spec, FixedAngleSpec) and self.angle_theta != 0.0

    @cached_property
    def frequency_grid(self) -> np.ndarray:
        """A Chebyshev grid used to approximate frequency dependence."""
        freq_min, freq_max = self.source_time.frequency_range(num_fwidth=CHEB_GRID_WIDTH)
        if not self._is_fixed_angle:
            # For frequency-dependent angles (constat in-plane k), truncate minimum frequency at
            # the critical frequency of glancing incidence
            f_crit = self.source_time.freq0 * np.sin(self.angle_theta)
            freq_min = max(freq_min, f_crit * CRITICAL_FREQUENCY_FACTOR)
        return self._chebyshev_freq_grid(freq_min, freq_max)

    def _post_init_validators(self) -> None:
        """Error if a broadband plane wave with constant in-plane k is defined such that
        the source frequency range is entirely below ``f_crit * CRITICAL_FREQUENCY_FACTOR."""
        if self._is_fixed_angle or self.num_freqs == 1:
            return
        freq_min, freq_max = self.source_time.frequency_range(num_fwidth=CHEB_GRID_WIDTH)
        f_crit = self.source_time.freq0 * np.sin(self.angle_theta)
        if f_crit * CRITICAL_FREQUENCY_FACTOR > freq_max:
            raise SetupError(
                "Broadband plane wave source defined with a bandwidth too close to the critical "
                "frequency of oblique incidence. Increase the source bandwidth, or disable the "
                "broadband handling by setting 'num_freqs' to 1."
            )


class GaussianBeam(AngledFieldSource, PlanarSource, BroadbandSource):
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

    Notes
    --------
    If one wants the focus 'in front' of the source, a negative value of ``beam_distance`` is needed.

    .. image:: ../../_static/img/beam_waist.png
        :width: 30%
        :align: center

    See Also
    --------

    **Notebooks**:
        * `Inverse taper edge coupler <../../notebooks/EdgeCoupler.html>`_
    """

    waist_radius: pydantic.PositiveFloat = pydantic.Field(
        1.0,
        title="Waist Radius",
        description="Radius of the beam at the waist.",
        units=MICROMETER,
    )

    waist_distance: float = pydantic.Field(
        0.0,
        title="Waist Distance",
        description="Distance from the beam waist along the propagation direction. "
        "A positive value means the waist is positioned behind the source, considering the propagation direction. "
        "For example, for a beam propagating in the ``+`` direction, a positive value of ``beam_distance`` "
        "means the beam waist is positioned in the ``-`` direction (behind the source). "
        "A negative value means the beam waist is in the ``+`` direction (in front of the source). "
        "For an angled source, the distance is defined along the rotated propagation direction.",
        units=MICROMETER,
    )


class AstigmaticGaussianBeam(AngledFieldSource, PlanarSource, BroadbandSource):
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

    waist_sizes: Tuple[pydantic.PositiveFloat, pydantic.PositiveFloat] = pydantic.Field(
        (1.0, 1.0),
        title="Waist sizes",
        description="Size of the beam at the waist in the local x and y directions.",
        units=MICROMETER,
    )

    waist_distances: Tuple[float, float] = pydantic.Field(
        (0.0, 0.0),
        title="Waist distances",
        description="Distance to the beam waist along the propagation direction "
        "for the waist sizes in the local x and y directions. "
        "When ``direction`` is ``+`` and ``waist_distances`` are positive, the waist "
        "is on the ``-`` side (behind) the source plane. When ``direction`` is ``+`` and "
        "``waist_distances`` are negative, the waist is on the ``+`` side (in front) of "
        "the source plane.",
        units=MICROMETER,
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

    injection_axis: Axis = pydantic.Field(
        ...,
        title="Injection Axis",
        description="Specifies the injection axis. The plane of incidence is defined via this "
        "``injection_axis`` and the ``direction``. The popagation axis is defined with respect "
        "to the ``injection_axis`` by ``angle_theta`` and ``angle_phi``.",
    )

    @cached_property
    def _injection_axis(self):
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
        x: float = None,
        y: float = None,
        z: float = None,
        ax: Ax = None,
        **patch_kwargs,
    ) -> Ax:
        # call Source.plot but with the base of the arrow centered on the injection plane
        patch_kwargs["arrow_base"] = self.injection_plane_center
        ax = Source.plot(self, x=x, y=y, z=z, ax=ax, **patch_kwargs)
        return ax
