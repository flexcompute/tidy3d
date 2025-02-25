"""Classes for creating data based on analytic beams like plane wave, Gaussian beam, and
astigmatic Gaussian beam."""

from abc import abstractmethod
from typing import Optional, Tuple, Union

import autograd.numpy as np
import pydantic.v1 as pd

from ..constants import C_0, ETA_0, HERTZ, MICROMETER, RADIAN
from .base import cached_property
from .data.data_array import ScalarFieldDataArray
from .data.monitor_data import FieldData
from .geometry.base import Box
from .grid.grid import Coords, Grid
from .medium import Medium, MediumType
from .monitor import FieldMonitor
from .source.field import FixedAngleSpec, FixedInPlaneKSpec
from .types import TYPE_TAG_STR, Direction, FreqArray, Literal, Numpy
from .validators import assert_plane

DEFAULT_RESOLUTION = 200


class BeamProfile(Box):
    """Base class for handling analytic beams."""

    resolution: float = pd.Field(
        DEFAULT_RESOLUTION,
        title="Sampling resolution",
        description="Sampling resolution in the tangential directions of the beam (defines a "
        "number of equally spaced points).",
        units=MICROMETER,
    )

    freqs: FreqArray = pd.Field(
        ...,
        title="Frequencies",
        description="List of frequencies at which the beam is sampled.",
        units=HERTZ,
    )

    background_medium: MediumType = pd.Field(
        Medium(),
        title="Background Medium",
        description="Background medium in which the beam is embedded.",
    )

    angle_theta: float = pd.Field(
        0.0,
        title="Polar Angle",
        description="Polar angle of the propagation axis from the normal axis.",
        units=RADIAN,
    )

    angle_phi: float = pd.Field(
        0.0,
        title="Azimuth Angle",
        description="Azimuth angle of the propagation axis in the plane orthogonal to the "
        "normal axis.",
        units=RADIAN,
    )

    pol_angle: float = pd.Field(
        0.0,
        title="Polarization Angle",
        description="Specifies the angle between the electric field polarization of the "
        "beam and the plane defined by the normal axis and the propagation axis (rad). "
        "``pol_angle=0`` (default) specifies P polarization, "
        "while ``pol_angle=np.pi/2`` specifies S polarization. "
        "At normal incidence when S and P are undefined, ``pol_angle=0`` defines: "
        "- ``Ey`` polarization for propagation along ``x``."
        "- ``Ex`` polarization for propagation along ``y``."
        "- ``Ex`` polarization for propagation along ``z``.",
        units=RADIAN,
    )

    direction: Direction = pd.Field(
        "+",
        title="Direction",
        description="Specifies propagation in the positive or negative direction of the normal "
        "axis.",
    )

    _plane_validator = assert_plane()

    @property
    def grid(self) -> Grid:
        """Return a Grid object on which the beam will be sampled."""
        dim_n, dim_tan = self.pop_axis("xyz", self.size.index(0.0))
        bounds_n, bounds_tan = self.pop_axis(np.array(self.bounds).T, self.size.index(0.0))
        boundaries = {
            dim_n: bounds_n,
            dim_tan[0]: np.linspace(bounds_tan[0][0], bounds_tan[0][1], int(self.resolution)),
            dim_tan[1]: np.linspace(bounds_tan[1][0], bounds_tan[1][1], int(self.resolution)),
        }
        return Grid(boundaries=boundaries)

    @property
    def monitor(self) -> FieldMonitor:
        """``FieldMonitor`` with the same center, size, and frequencies as the beam, to be used
        in the ``FieldData`` created by the ``field_data`` method."""
        return FieldMonitor(
            size=self.size,
            center=self.center,
            freqs=self.freqs,
            name="<<BEAM_DATA>>",
        )

    @cached_property
    def field_data(self) -> FieldData:
        """Compute a FieldData for the spatial E and H field amplitudes of an analytically defined
        beam."""

        background_n = np.zeros(len(self.freqs), dtype=complex)
        for freq_id, freq in enumerate(self.freqs):
            eps = self.background_medium.eps_model(freq)
            nk_medium = self.background_medium.eps_complex_to_nk(eps)
            background_n[freq_id] = np.squeeze(nk_medium[0]) + 1j * np.squeeze(nk_medium[1])

        data_dict = self._field_data_on_grid(grid=self.grid, background_n=background_n)
        data_raw = FieldData(monitor=self.monitor, grid_expanded=self.grid, **data_dict)

        # Normalize by the flux
        fields_norm = {}
        flux = np.abs(data_raw.flux)
        for field_name, field_data in data_raw.field_components.items():
            fields_norm[field_name] = (field_data / np.sqrt(flux)).astype(field_data.dtype)

        return data_raw.updated_copy(**fields_norm)

    def _field_data_on_grid(self, grid: Grid, background_n: Numpy, colocate=True) -> dict:
        """Compute the field data for each field component on a grid for the beam.
        A dictionary of the scalar field data arrays is returned, not yet packaged as ``FieldData``.
        """

        field_components = ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]
        if colocate:
            # Just sample at the grid boundaries for each field component
            # We skip the last boundary because that's how things also work in the solver (no
            # field defined at the last boundary)
            coords_dict = {key: val[:-1] for key, val in grid.boundaries.to_dict.items()}
            grid_dict = {component: Coords(**coords_dict) for component in field_components}
        else:
            # Yee grid for each field component
            grid_dict = grid.yee.grid_dict

        # Compute each field component over the beam grid
        scalar_fields = {}
        for comp, field in enumerate(field_components):
            x, y, z = grid_dict[field].to_list
            # Center component grid at beam center
            x_c, y_c, z_c = (coords - cent for coords, cent in zip((x, y, z), self.center))
            # Stack E and H fields
            field_vals = self.analytic_beam(x_c, y_c, z_c, background_n, field=field[0])
            # Get the current field component
            field_vals = field_vals[comp % 3]
            # Make the ScalarFieldDataArray for the current component
            coords = dict(x=x, y=y, z=z, f=np.array(self.freqs))
            field_data = ScalarFieldDataArray(field_vals, coords=coords)
            scalar_fields[field] = field_data

        return scalar_fields

    @abstractmethod
    def scalar_field(self, points: Numpy, background_n: float) -> Numpy:
        """Scalar field corresponding to the analytic beam in coordinate system such that the
        propagation direction is z and the ``E``-field is entirely ``x``-polarized. The field is
        computed on an unstructured array ``points`` of shape ``(3, ...)``."""
        pass

    def analytic_beam_z_normal(
        self, points: Numpy, background_n: float, field: Literal["E", "H"]
    ) -> Numpy:
        """Analytic beam with all the beam parameters but assuming ``z`` as the normal axis."""

        # Add a frequency dimension to points
        points = np.repeat(points[:, :, np.newaxis], len(self.freqs), axis=2)

        # Rotate points to axes where propagation is along z
        points_prop_z = self._rotate_points_z(points, background_n)

        # Reflection at the z = 0 plane for negative direction
        if self.direction == "-":
            points_prop_z[2, ...] *= -1

        # Get fields polarized along x and propagating along z
        scalar_field = self.scalar_field(points_prop_z, background_n)

        # Set the correct field component based on the scalar field values
        field_vals = np.zeros(points.shape, dtype=np.complex128)
        if field == "E":
            field_vals[0, ...] = scalar_field
        else:
            field_vals[1, ...] = scalar_field / ETA_0 * background_n

        # Rotate polarization
        field_vals = self.rotate_points(field_vals, [0, 0, 1], self.pol_angle)

        # Reflection of the field components at the z = 0 plane for negative direction
        if self.direction == "-":
            if field == "E":
                field_vals[2, :] *= -1
            else:
                field_vals[:2, :] *= -1

        # Rotate the fields back to the original propagation axes
        field_vals = self._inverse_rotate_field_vals_z(field_vals, background_n)

        return field_vals

    def analytic_beam(
        self,
        x: Numpy,
        y: Numpy,
        z: Numpy,
        background_n: float,
        field: Literal["E", "H"],
    ) -> Numpy:
        """Sample the analytic beam fields on a cartesian grid of points in x, y, z."""

        # Make a meshgrid
        (x_mesh, y_mesh, z_mesh) = np.meshgrid(x, y, z, indexing="ij")
        (Nx, Ny, Nz) = x_mesh.shape

        # Move to coordinates where the normal axis is z
        (z_new, (x_new, y_new)) = self.pop_axis((x_mesh, y_mesh, z_mesh), axis=self.size.index(0.0))
        points = np.stack([coords.ravel() for coords in (x_new, y_new, z_new)], axis=0)

        # Sample the plane wave fields
        f_x, f_y, f_z = self.analytic_beam_z_normal(points, background_n, field)

        # Move back to original coordinates
        field_vals = np.stack(self.unpop_axis(f_z, (f_x, f_y), axis=self.size.index(0.0)), axis=0)

        # H field gets a -1 factor if a reflection was involved
        if self.size.index(0.0) == 1 and field == "H":
            field_vals *= -1

        # Reshape to (3, Nx, Ny, Nz, num_freqs)
        return np.reshape(field_vals, (3, Nx, Ny, Nz, len(self.freqs)))

    def _rotate_points_z(self, points: Numpy, background_n: Numpy) -> Numpy:
        """Rotate points to new coordinates where z is the propagation axis."""
        points_prop_z = self.rotate_points(points, [0, 0, 1], -self.angle_phi)
        points_prop_z = self.rotate_points(points_prop_z, [0, 1, 0], -self.angle_theta)
        return points_prop_z

    def _inverse_rotate_field_vals_z(self, field_vals: Numpy, background_n: Numpy) -> Numpy:
        """Rotate field values from coordinates where z is the propagation axis to angled
        coordinates."""
        field_vals = self.rotate_points(field_vals, [0, 1, 0], self.angle_theta)
        field_vals = self.rotate_points(field_vals, [0, 0, 1], self.angle_phi)
        return field_vals


class PlaneWaveBeamProfile(BeamProfile):
    """Component for constructing plane wave beam data. The normal direction is implicitly
    defined by the ``size`` parameter.

    See also :class:`.PlaneWave`.
    """

    angular_spec: Union[FixedInPlaneKSpec, FixedAngleSpec] = pd.Field(
        FixedAngleSpec(),
        title="Angular Dependence Specification",
        description="Specification of plane wave propagation direction dependence on wavelength.",
        discriminator=TYPE_TAG_STR,
    )

    as_fixed_angle_source: bool = pd.Field(
        False,
        title="Fixed Angle Flag",
        description="Fixed angle flag. Only used internally when computing source beams for "
        "injection in an FDTD simulation with fixed angle boudnaries. Use ``angular_spec`` to "
        "switch between waves with fixed angle and fixed in-plane k.",
    )

    angle_theta_frequency: Optional[float] = pd.Field(
        None,
        title="Frequency at Which Angle Theta is Defined",
        description="Frequency for which ``angle_theta`` is set. This only has an effect for "
        "fixed in-plane wave-vector beams. If not supplied, the average of the beam ``freqs`` is "
        "used.",
    )

    @property
    def _angle_theta_frequency(self):
        if not self.angle_theta_frequency:
            return np.mean(self.freqs)
        return self.angle_theta_frequency

    def in_plane_k(self, background_n: float):
        """In-plane wave vector. Only the real part is taken so the beam has no in-plane decay."""
        k0 = 2 * np.pi * self._angle_theta_frequency / C_0 * background_n
        k_in_plane = k0.real * np.sin(self.angle_theta)
        return [k_in_plane * np.cos(self.angle_phi), k_in_plane * np.sin(self.angle_phi)]

    def scalar_field(self, points: Numpy, background_n: float) -> Numpy:
        """Scalar field for plane wave.
        Scalar field corresponding to the analytic beam in coordinate system such that the
        propagation direction is z and the ``E``-field is entirely ``x``-polarized. The field is
        computed on an unstructured array ``points`` of shape ``(3, N_points, N_freqs)``.
        For the special case of fixed in-plane k, the propagation axis is different at every
        frequency, and the points a frquency-dependent rotation has been applied to the
        ``points`` in ``self._rotate_points_z``.
        """

        kz = 2 * np.pi * np.array(self.freqs) / C_0 * background_n
        if self.as_fixed_angle_source:
            kz *= np.cos(self.angle_theta)
        return np.exp(1j * points[2] * kz)

    def _angle_theta_actual(self, background_n: Numpy) -> Numpy:
        """Compute the frequency-dependent actual propagation angle theta."""
        k0 = 2 * np.pi * np.array(self.freqs) / C_0 * background_n
        kx, ky = self.in_plane_k(background_n)
        k_perp = np.sqrt(kx**2 + ky**2)
        return np.real(np.arcsin(k_perp / k0))

    def _rotate_points_z(self, points: Numpy, background_n: Numpy) -> Numpy:
        """Rotate points to new coordinates where z is the propagation axis."""
        if self.as_fixed_angle_source:
            # For fixed-angle, we do not rotate the points
            return points
        elif isinstance(self.angular_spec, FixedInPlaneKSpec):
            # For fixed in-plane k, the rotation is angle-dependent
            points = self.rotate_points(points, [0, 0, 1], -self.angle_phi)
            angle_theta_actual = self._angle_theta_actual(background_n=background_n)
            for ind, theta_actual in enumerate(angle_theta_actual):
                points[:, :, ind] = self.rotate_points(points[:, :, ind], [0, 1, 0], -theta_actual)
            return points
        return super()._rotate_points_z(points, background_n)

    def _inverse_rotate_field_vals_z(self, field_vals: Numpy, background_n: Numpy) -> Numpy:
        """Rotate field values from coordinates where z is the propagation axis to angled
        coordinates. Special handling is needed if fixed in-plane k wave."""
        if isinstance(self.angular_spec, FixedInPlaneKSpec):
            # For fixed in-plane k, the rotation is angle-dependent
            angle_theta_actual = self._angle_theta_actual(background_n=background_n)
            for ind, theta_actual in enumerate(angle_theta_actual):
                field_vals[:, :, ind] = self.rotate_points(
                    field_vals[:, :, ind], [0, 1, 0], theta_actual
                )
            field_vals = self.rotate_points(field_vals, [0, 0, 1], self.angle_phi)
            return field_vals
        return super()._inverse_rotate_field_vals_z(field_vals, background_n)


class GaussianBeamProfile(BeamProfile):
    """Component for constructing Gaussian beam data. The normal direction is implicitly
    defined by the ``size`` parameter.

    See also :class:`.GaussianBeam`.
    """

    waist_radius: pd.PositiveFloat = pd.Field(
        1.0,
        title="Waist Radius",
        description="Radius of the beam at the waist.",
        units=MICROMETER,
    )

    waist_distance: float = pd.Field(
        0.0,
        title="Waist Distance",
        description="Distance from the beam waist along the propagation direction. "
        "A positive value means the waist is positioned behind the beam, considering the propagation direction. "
        "For example, for a beam propagating in the ``+`` direction, a positive value of ``beam_distance`` "
        "means the beam waist is positioned in the ``-`` direction (behind the beam). "
        "A negative value means the beam waist is in the ``+`` direction (in front of the beam). "
        "For an angled beam, the distance is defined along the rotated propagation direction.",
        units=MICROMETER,
    )

    def beam_params(self, z: Numpy, k0: Numpy) -> Tuple[Numpy, Numpy, Numpy]:
        """Compute the parameters needed to evaluate a Gaussian beam at z.

        Parameters
        ----------
        z : Numpy
            Axial distance from the beam center.
        k0 : Numpy
            Wave vector magnitude.
        """

        w_0, z_0 = self.waist_radius, self.waist_distance
        z_r = w_0**2 * k0 / 2  # shape k0
        w_z = w_0 * np.sqrt(1 + ((z + z_0) / z_r) ** 2)  # shape (Np, Nk0)
        # inv_r_z shape (Np, Nk0)
        inv_r_z = (z + z_0) / ((z + z_0) ** 2 + z_r**2)
        # we choose gauge such that psi_g = 0 at z = 0 (beam plane)
        # this is needed for a proper interpolation between different frequencies
        # psi_g shape (Np, Nk0)
        psi_g = np.arctan((z + z_0) / z_r) - np.arctan(z_0 / z_r)
        return w_z, inv_r_z, psi_g

    def scalar_field(self, points: Numpy, background_n: float) -> Numpy:
        """Scalar field for Gaussian beam.
        Scalar field corresponding to the analytic beam in coordinate system such that the
        propagation direction is z and the ``E``-field is entirely ``x``-polarized. The field is
        computed on an unstructured array ``points`` of shape ``(3, ...)``.
        """
        k0 = 2 * np.pi * np.array(self.freqs) / C_0 * background_n
        x, y, z = points
        w_0 = self.waist_radius
        w_z, inv_r_z, psi_g = self.beam_params(z, k0)
        r_2 = x**2 + y**2
        scalar_gaussian = w_0 / w_z
        scalar_gaussian *= np.exp(-r_2 / w_z**2)
        scalar_gaussian *= np.exp(1j * (z * k0 + r_2 * k0 / 2 * inv_r_z - psi_g))

        return scalar_gaussian


class AstigmaticGaussianBeamProfile(BeamProfile):
    """Component for constructing astigmatic Gaussian beam data. The normal direction is implicitly
    defined by the ``size`` parameter.

    See also :class:`.AstigmaticGaussianBeam`.
    """

    waist_sizes: Tuple[pd.PositiveFloat, pd.PositiveFloat] = pd.Field(
        (1.0, 1.0),
        title="Waist sizes",
        description="Size of the beam at the waist in the local x and y directions.",
        units=MICROMETER,
    )

    waist_distances: Tuple[float, float] = pd.Field(
        (0.0, 0.0),
        title="Waist distances",
        description="Distance to the beam waist along the propagation direction "
        "for the waist sizes in the local x and y directions. "
        "When ``direction`` is ``+`` and ``waist_distances`` are positive, the waist "
        "is on the ``-`` side (behind) the beam plane. When ``direction`` is ``+`` and "
        "``waist_distances`` are negative, the waist is on the ``+`` side (in front) of "
        "the beam plane.",
        units=MICROMETER,
    )

    def beam_params(self, z: Numpy, k0: Numpy) -> Tuple[Numpy, Numpy, Numpy, Numpy]:
        """Compute the parameters needed to evaluate an astigmatic Gaussian beam at z.

        Parameters
        ----------
        z : Numpy
            Axial distance from the beam center.
        k0 : Numpy
            Wave vector magnitude.
        """

        w_xy, z_xy = self.waist_sizes, self.waist_distances  # shape (2, )
        z_r = [w**2 * k0 / 2 for w in w_xy]  # shape (2, Nk0)
        w_z, w_0, inv_r_z, psi_g = [], [], [], []  # final shape (2, Np, Nk0) after loop below
        for w, z_i, z_ri in zip(w_xy, z_xy, z_r):
            w_z.append(w * np.sqrt(1 + ((z + z_i) / z_ri) ** 2))
            w_0.append(w * np.sqrt(1 + (z_i / z_ri) ** 2))
            inv_r_z.append((z + z_i) / ((z + z_i) ** 2 + z_ri**2))
            # we choose gauge such that psi_g = 0 at z = 0 (beam plane)
            # this is needed for a proper interpolation between different frequencies
            # psi_g shape (Np, Nk0)
            psi_g_1 = np.arctan((z + z_i) / z_ri) / 2
            psi_g_2 = np.arctan(z_i / z_ri) / 2
            psi_g.append(psi_g_1 - psi_g_2)

        return w_0, w_z, inv_r_z, psi_g

    def scalar_field(self, points: Numpy, background_n: float) -> Numpy:
        """
        Scalar field for astigmatic Gaussian beam.
        Scalar field corresponding to the analytic beam in coordinate system such that the
        propagation direction is z and the ``E``-field is entirely ``x``-polarized. The field is
        computed on an unstructured array ``points`` of shape ``(3, ...)``.
        """
        k0 = 2 * np.pi * np.array(self.freqs) / C_0 * background_n
        x, y, z = points
        w_0, w_z, inv_r_z, psi_g = self.beam_params(z, k0)
        x_2 = x**2
        y_2 = y**2

        q_term1_inv = 1j * x_2 * k0 / 2 * inv_r_z[0] - x_2 / w_z[0] ** 2
        q_term2_inv = 1j * y_2 * k0 / 2 * inv_r_z[1] - y_2 / w_z[1] ** 2
        angle_term = q_term1_inv + q_term2_inv + 1j * (z * k0 - psi_g[0] - psi_g[1])
        scalar_gaussian = np.exp(angle_term)

        ampl = np.sqrt(w_0[0] * w_0[1] / w_z[0] / w_z[1])
        return scalar_gaussian * ampl
