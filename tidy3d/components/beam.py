"""Classes for creating data based on analytic beams like plane wave, Gaussian beam, and
astigmatic Gaussian beam."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Union

import autograd.numpy as np
from pydantic import Field, PositiveFloat

from tidy3d.constants import C_0, ETA_0, HERTZ, MICROMETER, RADIAN

from .base import cached_property
from .data.data_array import ScalarFieldDataArray
from .data.monitor_data import FieldData
from .geometry.base import Box
from .grid.grid import Coords, Grid, _compute_1d_cell_sizes
from .medium import Medium, MediumType
from .monitor import FieldMonitor
from .source.field import FixedAngleSpec, FixedInPlaneKSpec
from .transformation import rotate_points_around_axis
from .types import TYPE_TAG_STR, Direction, FreqArray
from .validators import assert_plane, warn_backward_waist_distance

if TYPE_CHECKING:
    from typing import Callable, Literal

    from numpy.typing import NDArray

DEFAULT_RESOLUTION = 200


@dataclass(frozen=True)
class BeamPose:
    """Pose and orientation parameters shared by Gaussian-like beam builders."""

    injection_axis: int
    direction: Direction
    angle_theta: float
    angle_phi: float
    pol_angle: float
    center: tuple[float, float, float]


@dataclass(frozen=True)
class BeamGrid:
    """Sampling grid and medium parameters for Gaussian-like beam builders."""

    x: NDArray
    y: NDArray
    z: NDArray
    freqs: NDArray
    background_n: NDArray


class BeamProfile(Box):
    """Base class for handling analytic beams."""

    resolution: float = Field(
        DEFAULT_RESOLUTION,
        title="Sampling resolution",
        description="Sampling resolution in the tangential directions of the beam (defines a "
        "number of equally spaced points).",
        json_schema_extra={"units": MICROMETER},
    )

    freqs: FreqArray = Field(
        title="Frequencies",
        description="List of frequencies at which the beam is sampled.",
        json_schema_extra={"units": HERTZ},
    )

    background_medium: MediumType = Field(
        default_factory=Medium,
        title="Background Medium",
        description="Background medium in which the beam is embedded.",
    )

    angle_theta: float = Field(
        0.0,
        title="Polar Angle",
        description="Polar angle of the propagation axis from the normal axis.",
        json_schema_extra={"units": RADIAN},
    )

    angle_phi: float = Field(
        0.0,
        title="Azimuth Angle",
        description="Azimuth angle of the propagation axis in the plane orthogonal to the "
        "normal axis.",
        json_schema_extra={"units": RADIAN},
    )

    pol_angle: float = Field(
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
        json_schema_extra={"units": RADIAN},
    )

    direction: Direction = Field(
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
        background_n = self.background_medium.background_index_from_freqs(self.freqs)

        data_dict = self._field_data_on_grid(grid=self.grid, background_n=background_n)
        data_raw = FieldData(monitor=self.monitor, grid_expanded=self.grid, **data_dict)

        # Normalize by the flux
        fields_norm = {}
        flux = np.abs(data_raw.flux)
        for field_name, field_data in data_raw.field_components.items():
            fields_norm[field_name] = (field_data / np.sqrt(flux)).astype(field_data.dtype)

        return data_raw.updated_copy(**fields_norm)

    def _field_data_on_grid(
        self,
        grid: Grid,
        background_n: NDArray,
        colocate: bool = True,
        field_components: Optional[tuple[str, ...]] = None,
    ) -> dict[str, ScalarFieldDataArray]:
        """Compute the field data for each field component on a grid for the beam.
        A dictionary of the scalar field data arrays is returned, not yet packaged as ``FieldData``.
        """

        all_field_components = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
        if field_components is None:
            field_components = all_field_components
        else:
            invalid_components = [
                field for field in field_components if field not in all_field_components
            ]
            if invalid_components:
                invalid_components_str = ", ".join(invalid_components)
                raise ValueError(f"Invalid field components requested: {invalid_components_str}")
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
        for field in field_components:
            x, y, z = grid_dict[field].to_list
            fields_dict = self._beam_fields_on_component_grid(
                x=x, y=y, z=z, background_n=background_n
            )
            field_vals = fields_dict[field]
            # Make the ScalarFieldDataArray for the current component
            coords = {"x": x, "y": y, "z": z, "f": np.array(self.freqs)}
            field_data = ScalarFieldDataArray(field_vals, coords=coords)
            scalar_fields[field] = field_data

        return scalar_fields

    def _beam_fields_on_component_grid(
        self, *, x: NDArray, y: NDArray, z: NDArray, background_n: NDArray
    ) -> dict[str, NDArray]:
        """Compute all six field components on one component grid."""
        # Center component grid at beam center
        x_c, y_c, z_c = (coords - cent for coords, cent in zip((x, y, z), self.center))
        # Compute vector E and H fields once each and split into components.
        e_vals = self.analytic_beam(x_c, y_c, z_c, background_n, field="E")
        h_vals = self.analytic_beam(x_c, y_c, z_c, background_n, field="H")
        return {
            "Ex": e_vals[0],
            "Ey": e_vals[1],
            "Ez": e_vals[2],
            "Hx": h_vals[0],
            "Hy": h_vals[1],
            "Hz": h_vals[2],
        }

    @abstractmethod
    def scalar_field(self, points: NDArray, background_n: float) -> NDArray:
        """Scalar field corresponding to the analytic beam in coordinate system such that the
        propagation direction is z and the ``E``-field is entirely ``x``-polarized. The field is
        computed on an unstructured array ``points`` of shape ``(3, ...)``."""

    def analytic_beam_z_normal(
        self, points: NDArray, background_n: float, field: Literal["E", "H"]
    ) -> NDArray:
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

        e_vals, h_vals = _gaussian_like_local_eh_from_scalar(scalar_field, background_n)
        e_vals, h_vals = _gaussian_like_apply_pol_and_direction(
            e_vals=e_vals,
            h_vals=h_vals,
            pol_angle=self.pol_angle,
            direction=self.direction,
            rotate_fn=lambda vals, axis, angle: self.rotate_points(vals, list(axis), angle),
        )
        field_vals = e_vals if field == "E" else h_vals
        # Rotate the fields back to the original propagation axes
        field_vals = self._inverse_rotate_field_vals_z(field_vals, background_n)

        return field_vals

    def analytic_beam(
        self,
        x: NDArray,
        y: NDArray,
        z: NDArray,
        background_n: float,
        field: Literal["E", "H"],
    ) -> NDArray:
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

    def _rotate_points_z(self, points: NDArray, background_n: NDArray) -> NDArray:
        """Rotate points to new coordinates where z is the propagation axis."""
        points_prop_z = self.rotate_points(points, [0, 0, 1], -self.angle_phi)
        points_prop_z = self.rotate_points(points_prop_z, [0, 1, 0], -self.angle_theta)
        return points_prop_z

    def _inverse_rotate_field_vals_z(self, field_vals: NDArray, background_n: NDArray) -> NDArray:
        """Rotate field values from coordinates where z is the propagation axis to angled
        coordinates."""
        field_vals = self.rotate_points(field_vals, [0, 1, 0], self.angle_theta)
        field_vals = self.rotate_points(field_vals, [0, 0, 1], self.angle_phi)
        return field_vals


def _gaussian_like_local_eh_from_scalar(
    scalar_field: NDArray, background_n: NDArray
) -> tuple[NDArray, NDArray]:
    """Construct local (x-polarized) E/H vectors from scalar beam field."""
    background_n = np.array(background_n)
    zeros = np.zeros_like(scalar_field)
    e_vals = np.stack((scalar_field, zeros, zeros), axis=0)
    h_vals = np.stack((zeros, scalar_field / ETA_0 * background_n, zeros), axis=0)
    return e_vals, h_vals


def _gaussian_like_apply_pol_and_direction(
    *,
    e_vals: NDArray,
    h_vals: NDArray,
    pol_angle: float,
    direction: Direction,
    rotate_fn: Any,
) -> tuple[NDArray, NDArray]:
    """Apply polarization rotation and backward-direction parity in local frame."""
    e_vals = rotate_fn(e_vals, (0.0, 0.0, 1.0), pol_angle)
    h_vals = rotate_fn(h_vals, (0.0, 0.0, 1.0), pol_angle)
    if direction == "-":
        e_vals = np.stack((e_vals[0], e_vals[1], -e_vals[2]), axis=0)
        h_vals = np.stack((-h_vals[0], -h_vals[1], h_vals[2]), axis=0)
    return e_vals, h_vals


def _gaussian_scalar_field(
    *,
    points_prop_z: NDArray,
    freqs: NDArray,
    background_n: NDArray,
    direction: Direction,
    waist_radius: float,
    waist_distance: float,
) -> NDArray:
    """Scalar field in propagation-aligned coordinates for a Gaussian beam."""
    x_p, y_p, z_p = points_prop_z
    k0 = 2.0 * np.pi * np.array(freqs) / C_0 * np.array(background_n)
    z_0 = waist_distance
    if direction == "-":
        z_0 = -z_0
    z_r = waist_radius**2 * k0 / 2.0
    w_z = waist_radius * np.sqrt(1.0 + ((z_p + z_0) / z_r) ** 2)
    inv_r_z = (z_p + z_0) / ((z_p + z_0) ** 2 + z_r**2)
    psi_g = np.arctan((z_p + z_0) / z_r) - np.arctan(z_0 / z_r)
    r2 = x_p**2 + y_p**2
    scalar = (waist_radius / w_z) * np.exp(-r2 / w_z**2)
    scalar *= np.exp(1j * (z_p * k0 + r2 * k0 / 2.0 * inv_r_z - psi_g))
    return scalar


def _astigmatic_gaussian_scalar_field(
    *,
    points_prop_z: NDArray,
    freqs: NDArray,
    background_n: NDArray,
    direction: Direction,
    waist_sizes: tuple[float, float],
    waist_distances: tuple[float, float],
) -> NDArray:
    """Scalar field in propagation-aligned coordinates for an astigmatic Gaussian beam."""
    x_p, y_p, z_p = points_prop_z
    k0 = 2.0 * np.pi * np.array(freqs) / C_0 * np.array(background_n)
    z_xy = waist_distances
    if direction == "-":
        z_xy = tuple(-z_i for z_i in z_xy)
    z_r = [w**2 * k0 / 2.0 for w in waist_sizes]
    w_0 = []
    w_z = []
    inv_r_z = []
    psi_g = []
    for w, z_i, z_ri in zip(waist_sizes, z_xy, z_r):
        w_z.append(w * np.sqrt(1.0 + ((z_p + z_i) / z_ri) ** 2))
        w_0.append(w * np.sqrt(1.0 + (z_i / z_ri) ** 2))
        inv_r_z.append((z_p + z_i) / ((z_p + z_i) ** 2 + z_ri**2))
        psi_g.append(np.arctan((z_p + z_i) / z_ri) / 2.0 - np.arctan(z_i / z_ri) / 2.0)
    x2 = x_p**2
    y2 = y_p**2
    q_term1_inv = 1j * x2 * k0 / 2.0 * inv_r_z[0] - x2 / w_z[0] ** 2
    q_term2_inv = 1j * y2 * k0 / 2.0 * inv_r_z[1] - y2 / w_z[1] ** 2
    angle_term = q_term1_inv + q_term2_inv + 1j * (z_p * k0 - psi_g[0] - psi_g[1])
    scalar = np.exp(angle_term)
    ampl = np.sqrt(w_0[0] * w_0[1] / w_z[0] / w_z[1])
    return scalar * ampl


def _gaussian_like_prepare_points(
    *,
    pose: BeamPose,
    grid: BeamGrid,
) -> tuple[NDArray, NDArray]:
    """Prepare source-relative, propagation-aligned sample points for Gaussian-like beams."""
    x = np.array(grid.x)
    y = np.array(grid.y)
    z = np.array(grid.z)
    freqs = np.array(grid.freqs)

    x_c = x - pose.center[0]
    y_c = y - pose.center[1]
    z_c = z - pose.center[2]
    nx, ny, nz = x_c.size, y_c.size, z_c.size
    mesh_shape = (nx, ny, nz)
    x_mesh = np.broadcast_to(x_c.reshape((nx, 1, 1)), mesh_shape)
    y_mesh = np.broadcast_to(y_c.reshape((1, ny, 1)), mesh_shape)
    z_mesh = np.broadcast_to(z_c.reshape((1, 1, nz)), mesh_shape)

    axis_to_coords = {
        0: (x_mesh, y_mesh, z_mesh),
        1: (y_mesh, x_mesh, z_mesh),
        2: (z_mesh, x_mesh, y_mesh),
    }
    z_new, x_new, y_new = axis_to_coords[pose.injection_axis]

    points = np.stack((x_new.reshape(-1), y_new.reshape(-1), z_new.reshape(-1)), axis=0)
    points = np.repeat(points[:, :, np.newaxis], freqs.size, axis=2)

    points_prop_z = rotate_points_around_axis(points, axis=(0.0, 0.0, 1.0), angle=-pose.angle_phi)
    points_prop_z = rotate_points_around_axis(
        points_prop_z, axis=(0.0, 1.0, 0.0), angle=-pose.angle_theta
    )
    if pose.direction == "-":
        points_prop_z = np.stack((points_prop_z[0], points_prop_z[1], -points_prop_z[2]), axis=0)
    return freqs, points_prop_z


def _gaussian_like_fields_from_scalar(
    *,
    scalar_field: NDArray,
    pose: BeamPose,
    grid: BeamGrid,
    normalize: bool,
) -> dict[str, NDArray]:
    """Build E/H components from a scalar beam field on one component grid."""
    x = np.array(grid.x)
    y = np.array(grid.y)
    z = np.array(grid.z)
    nx, ny, nz = x.size, y.size, z.size
    e_vals, h_vals = _gaussian_like_local_eh_from_scalar(scalar_field, grid.background_n)
    e_vals, h_vals = _gaussian_like_apply_pol_and_direction(
        e_vals=e_vals,
        h_vals=h_vals,
        pol_angle=pose.pol_angle,
        direction=pose.direction,
        rotate_fn=lambda vals, axis, angle: rotate_points_around_axis(vals, axis=axis, angle=angle),
    )

    e_vals = rotate_points_around_axis(e_vals, axis=(0.0, 1.0, 0.0), angle=pose.angle_theta)
    e_vals = rotate_points_around_axis(e_vals, axis=(0.0, 0.0, 1.0), angle=pose.angle_phi)
    h_vals = rotate_points_around_axis(h_vals, axis=(0.0, 1.0, 0.0), angle=pose.angle_theta)
    h_vals = rotate_points_around_axis(h_vals, axis=(0.0, 0.0, 1.0), angle=pose.angle_phi)

    perm = ((2, 0, 1), (0, 2, 1), (0, 1, 2))[pose.injection_axis]
    e_xyz = tuple(e_vals[idx] for idx in perm)
    h_xyz = tuple(h_vals[idx] for idx in perm)

    e_xyz = np.stack(e_xyz, axis=0)
    h_xyz = np.stack(h_xyz, axis=0)
    if pose.injection_axis == 1:
        h_xyz = -h_xyz

    e_xyz = np.reshape(e_xyz, (3, nx, ny, nz, grid.freqs.size))
    h_xyz = np.reshape(h_xyz, (3, nx, ny, nz, grid.freqs.size))

    fields = {f"E{axis}": e_xyz[i] for i, axis in enumerate("xyz")}
    fields.update({f"H{axis}": h_xyz[i] for i, axis in enumerate("xyz")})

    if not normalize:
        return fields

    tangential_axes = [idx for idx in range(3) if idx != pose.injection_axis]
    ax1, ax2 = tangential_axes

    def _normal_plane(values: NDArray) -> NDArray:
        if pose.injection_axis == 0:
            return values[0, :, :, :]
        if pose.injection_axis == 1:
            return values[:, 0, :, :]
        return values[:, :, 0, :]

    e1 = _normal_plane(e_xyz[ax1])
    e2 = _normal_plane(e_xyz[ax2])
    h1 = _normal_plane(h_xyz[ax1])
    h2 = _normal_plane(h_xyz[ax2])

    complex_poynting = 0.5 * (e1 * np.conj(h2) - e2 * np.conj(h1))
    c1 = _compute_1d_cell_sizes(np.asarray((x, y, z)[ax1]))
    c2 = _compute_1d_cell_sizes(np.asarray((x, y, z)[ax2]))
    d_area = c1[:, np.newaxis] * c2[np.newaxis, :]
    flux = np.real(np.sum(complex_poynting * d_area[:, :, np.newaxis], axis=(0, 1)))
    flux_safe = np.maximum(np.abs(flux), 1e-30)
    norm = np.sqrt(flux_safe).reshape((1, 1, 1, -1))

    return {name: arr / norm for name, arr in fields.items()}


def _compute_vector_beam_fields(
    *,
    pose: BeamPose,
    grid: BeamGrid,
    normalize: bool,
    scalar_func: Callable[..., NDArray],
) -> dict[str, NDArray]:
    """Shared Gaussian-like pipeline from scalar builder to vector field components."""
    freqs, points_prop_z = _gaussian_like_prepare_points(pose=pose, grid=grid)
    scalar_field = scalar_func(
        points_prop_z=points_prop_z,
        freqs=freqs,
        background_n=grid.background_n,
        direction=pose.direction,
    )
    return _gaussian_like_fields_from_scalar(
        scalar_field=scalar_field,
        pose=pose,
        grid=grid,
        normalize=normalize,
    )


def gaussian_like_fields_on_grid_from_context(
    *,
    pose: BeamPose,
    grid: BeamGrid,
    waist_radius: float,
    waist_distance: float,
    normalize: bool = False,
) -> dict[str, NDArray]:
    """Compute Gaussian-beam fields on a grid from compact context objects."""
    return _compute_vector_beam_fields(
        pose=pose,
        grid=grid,
        normalize=normalize,
        scalar_func=lambda **kwargs: _gaussian_scalar_field(
            **kwargs,
            waist_radius=waist_radius,
            waist_distance=waist_distance,
        ),
    )


def astigmatic_gaussian_like_fields_on_grid_from_context(
    *,
    pose: BeamPose,
    grid: BeamGrid,
    waist_sizes: tuple[float, float],
    waist_distances: tuple[float, float],
    normalize: bool = False,
) -> dict[str, NDArray]:
    """Compute astigmatic Gaussian-beam fields on a grid from compact context objects."""
    return _compute_vector_beam_fields(
        pose=pose,
        grid=grid,
        normalize=normalize,
        scalar_func=lambda **kwargs: _astigmatic_gaussian_scalar_field(
            **kwargs,
            waist_sizes=waist_sizes,
            waist_distances=waist_distances,
        ),
    )


class PlaneWaveBeamProfile(BeamProfile):
    """Component for constructing plane wave beam data. The normal direction is implicitly
    defined by the ``size`` parameter.

    See Also
    --------
    :class:`.PlaneWave`
    """

    angular_spec: Union[FixedInPlaneKSpec, FixedAngleSpec] = Field(
        FixedAngleSpec(),
        title="Angular Dependence Specification",
        description="Specification of plane wave propagation direction dependence on wavelength.",
        discriminator=TYPE_TAG_STR,
    )

    as_fixed_angle_source: bool = Field(
        False,
        title="Fixed Angle Flag",
        description="Fixed angle flag. Only used internally when computing source beams for "
        "injection in an FDTD simulation with fixed angle boudnaries. Use ``angular_spec`` to "
        "switch between waves with fixed angle and fixed in-plane k.",
    )

    angle_theta_frequency: Optional[float] = Field(
        None,
        title="Frequency at Which Angle Theta is Defined",
        description="Frequency for which ``angle_theta`` is set. This only has an effect for "
        "fixed in-plane wave-vector beams. If not supplied, the average of the beam ``freqs`` is "
        "used.",
    )

    @property
    def _angle_theta_frequency(self) -> float:
        if not self.angle_theta_frequency:
            return np.mean(self.freqs)
        return self.angle_theta_frequency

    def in_plane_k(self, background_n: float) -> list[float]:
        """In-plane wave vector. Only the real part is taken so the beam has no in-plane decay."""
        k0 = 2 * np.pi * self._angle_theta_frequency / C_0 * background_n
        k_in_plane = k0.real * np.sin(self.angle_theta)
        return [k_in_plane * np.cos(self.angle_phi), k_in_plane * np.sin(self.angle_phi)]

    def scalar_field(self, points: NDArray, background_n: float) -> NDArray:
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

    def _angle_theta_actual(self, background_n: NDArray) -> NDArray:
        """Compute the frequency-dependent actual propagation angle theta."""
        k0 = 2 * np.pi * np.array(self.freqs) / C_0 * background_n
        kx, ky = self.in_plane_k(background_n)
        k_perp = np.sqrt(kx**2 + ky**2)
        return np.real(np.arcsin(k_perp / k0)) * np.sign(self.angle_theta)

    def _rotate_points_z(self, points: NDArray, background_n: NDArray) -> NDArray:
        """Rotate points to new coordinates where z is the propagation axis."""
        if self.as_fixed_angle_source:
            # For fixed-angle, we do not rotate the points
            return points
        if isinstance(self.angular_spec, FixedInPlaneKSpec):
            # For fixed in-plane k, the rotation is angle-dependent
            points = self.rotate_points(points, [0, 0, 1], -self.angle_phi)
            angle_theta_actual = self._angle_theta_actual(background_n=background_n)
            for ind, theta_actual in enumerate(angle_theta_actual):
                points[:, :, ind] = self.rotate_points(points[:, :, ind], [0, 1, 0], -theta_actual)
            return points
        return super()._rotate_points_z(points, background_n)

    def _inverse_rotate_field_vals_z(self, field_vals: NDArray, background_n: NDArray) -> NDArray:
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


class AbstractGaussianBeamProfile(BeamProfile, ABC):
    """Shared template for Gaussian-like beam profile field construction."""

    @abstractmethod
    def _fields_on_grid_from_context(
        self, *, pose: BeamPose, grid: BeamGrid, normalize: bool
    ) -> dict[str, NDArray]:
        """Child hook to build E/H fields from BeamPose and BeamGrid."""

    def _beam_pose(self, *, center: tuple[float, float, float]) -> BeamPose:
        return BeamPose(
            injection_axis=self.size.index(0.0),
            direction=self.direction,
            angle_theta=self.angle_theta,
            angle_phi=self.angle_phi,
            pol_angle=self.pol_angle,
            center=center,
        )

    def _beam_grid(self, *, x: NDArray, y: NDArray, z: NDArray, background_n: NDArray) -> BeamGrid:
        return BeamGrid(x=x, y=y, z=z, freqs=np.array(self.freqs), background_n=background_n)

    def _beam_fields_on_component_grid(
        self, *, x: NDArray, y: NDArray, z: NDArray, background_n: NDArray
    ) -> dict[str, NDArray]:
        """Compute Gaussian-like beam fields on one component grid."""
        pose = self._beam_pose(center=self.center)
        grid = self._beam_grid(x=x, y=y, z=z, background_n=background_n)
        return self._fields_on_grid_from_context(pose=pose, grid=grid, normalize=False)

    def analytic_beam(
        self,
        x: NDArray,
        y: NDArray,
        z: NDArray,
        background_n: float,
        field: Literal["E", "H"],
    ) -> NDArray:
        """Sample Gaussian-like beam vector fields on a Cartesian grid."""
        pose = self._beam_pose(center=(0.0, 0.0, 0.0))
        grid = self._beam_grid(x=x, y=y, z=z, background_n=background_n)
        fields = self._fields_on_grid_from_context(pose=pose, grid=grid, normalize=False)
        if field == "E":
            return np.stack((fields["Ex"], fields["Ey"], fields["Ez"]), axis=0)
        return np.stack((fields["Hx"], fields["Hy"], fields["Hz"]), axis=0)


class GaussianBeamProfile(AbstractGaussianBeamProfile):
    """Component for constructing Gaussian beam data. The normal direction is implicitly
    defined by the ``size`` parameter.

    See Also
    --------
    :class:`.GaussianBeam`
    """

    waist_radius: PositiveFloat = Field(
        1.0,
        title="Waist Radius",
        description="Radius of the beam at the waist.",
        json_schema_extra={"units": MICROMETER},
    )

    waist_distance: float = Field(
        0.0,
        title="Waist Distance",
        description="Distance from the beam waist along the propagation direction. "
        "A positive value places the waist behind the beam plane (toward the negative normal axis). "
        "A negative value places the waist in front of the beam plane (toward the positive normal axis). "
        "This definition is independent of the ``direction`` parameter, ensuring consistent waist "
        "positioning for both forward- and backward-propagating beams. "
        "For an angled beam, the distance is measured along the rotated propagation direction.",
        json_schema_extra={"units": MICROMETER},
    )
    _backward_waist_warning = warn_backward_waist_distance("waist_distance")

    def _fields_on_grid_from_context(
        self, *, pose: BeamPose, grid: BeamGrid, normalize: bool
    ) -> dict[str, NDArray]:
        """Compute Gaussian-beam fields from compact context."""
        return gaussian_like_fields_on_grid_from_context(
            pose=pose,
            grid=grid,
            waist_radius=self.waist_radius,
            waist_distance=self.waist_distance,
            normalize=normalize,
        )

    def scalar_field(self, points: NDArray, background_n: float) -> NDArray:
        """Scalar field for Gaussian beam.
        Scalar field corresponding to the analytic beam in coordinate system such that the
        propagation direction is z and the ``E``-field is entirely ``x``-polarized. The field is
        computed on an unstructured array ``points`` of shape ``(3, ...)``.
        """
        return _gaussian_scalar_field(
            points_prop_z=points,
            freqs=np.array(self.freqs),
            background_n=np.array(background_n),
            direction=self.direction,
            waist_radius=self.waist_radius,
            waist_distance=self.waist_distance,
        )


class AstigmaticGaussianBeamProfile(AbstractGaussianBeamProfile):
    """Component for constructing astigmatic Gaussian beam data. The normal direction is implicitly
    defined by the ``size`` parameter.

    See Also
    --------
    :class:`.AstigmaticGaussianBeam`
    """

    waist_sizes: tuple[PositiveFloat, PositiveFloat] = Field(
        (1.0, 1.0),
        title="Waist sizes",
        description="Size of the beam at the waist in the local x and y directions.",
        json_schema_extra={"units": MICROMETER},
    )

    waist_distances: tuple[float, float] = Field(
        (0.0, 0.0),
        title="Waist distances",
        description="Distance to the beam waist along the propagation direction "
        "for the waist sizes in the local x and y directions. "
        "Positive values place the waist behind the beam plane (toward the negative normal axis); "
        "negative values place the waist in front of the beam plane. "
        "This definition is independent of the ``direction`` parameter.",
        json_schema_extra={"units": MICROMETER},
    )
    _backward_waist_warning = warn_backward_waist_distance("waist_distances")

    def _fields_on_grid_from_context(
        self, *, pose: BeamPose, grid: BeamGrid, normalize: bool
    ) -> dict[str, NDArray]:
        """Compute astigmatic Gaussian-beam fields from compact context."""
        return astigmatic_gaussian_like_fields_on_grid_from_context(
            pose=pose,
            grid=grid,
            waist_sizes=self.waist_sizes,
            waist_distances=self.waist_distances,
            normalize=normalize,
        )

    def scalar_field(self, points: NDArray, background_n: float) -> NDArray:
        """
        Scalar field for astigmatic Gaussian beam.
        Scalar field corresponding to the analytic beam in coordinate system such that the
        propagation direction is z and the ``E``-field is entirely ``x``-polarized. The field is
        computed on an unstructured array ``points`` of shape ``(3, ...)``.
        """
        return _astigmatic_gaussian_scalar_field(
            points_prop_z=points,
            freqs=np.array(self.freqs),
            background_n=np.array(background_n),
            direction=self.direction,
            waist_sizes=self.waist_sizes,
            waist_distances=self.waist_distances,
        )
