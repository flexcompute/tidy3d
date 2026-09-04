"""Geometry transformations and coordinate conversions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import autograd.numpy as np

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from tidy3d.components.types import Axis, Coordinate

    from .array import GeometryArray
    from .core import Geometry


def translated(
    self,  # pyrefly: ignore[implicit-any-parameter]
    x: float,
    y: float,
    z: float,
) -> Geometry:
    """Return a translated copy of this geometry.

    Parameters
    ----------
    x : float
        Translation along x.
    y : float
        Translation along y.
    z : float
        Translation along z.

    Returns
    -------
    :class:`~tidy3d.Geometry`
        Translated copy of this geometry.
    """
    from .transformed import Transformed

    return Transformed(geometry=self, transform=Transformed.translation(x, y, z))


def scaled(
    self,  # pyrefly: ignore[implicit-any-parameter]
    x: float = 1.0,
    y: float = 1.0,
    z: float = 1.0,
) -> Geometry:
    """Return a scaled copy of this geometry.

    Parameters
    ----------
    x : float = 1.0
        Scaling factor along x.
    y : float = 1.0
        Scaling factor along y.
    z : float = 1.0
        Scaling factor along z.

    Returns
    -------
    :class:`~tidy3d.Geometry`
        Scaled copy of this geometry.
    """
    from .transformed import Transformed

    return Transformed(geometry=self, transform=Transformed.scaling(x, y, z))


def rotated(
    self,  # pyrefly: ignore[implicit-any-parameter]
    angle: float,
    axis: Axis | Coordinate,
) -> Geometry:
    """Return a rotated copy of this geometry.

    Parameters
    ----------
    angle : float
        Rotation angle (in radians).
    axis : Union[int, tuple[float, float, float]]
        Axis of rotation: 0, 1, or 2 for x, y, and z, respectively, or a 3D vector.

    Returns
    -------
    :class:`~tidy3d.Geometry`
        Rotated copy of this geometry.
    """
    from .transformed import Transformed

    return Transformed(geometry=self, transform=Transformed.rotation(angle, axis))


def reflected(
    self,  # pyrefly: ignore[implicit-any-parameter]
    normal: Coordinate,
) -> Geometry:
    """Return a reflected copy of this geometry.

    Parameters
    ----------
    normal : tuple[float, float, float]
        The 3D normal vector of the plane of reflection. The plane is assumed
            to pass through the origin (0,0,0).

    Returns
    -------
    :class:`~tidy3d.Geometry`
        Reflected copy of this geometry.
    """
    from .transformed import Transformed

    return Transformed(geometry=self, transform=Transformed.reflection(normal))


def array(
    self,  # pyrefly: ignore[implicit-any-parameter]
    offsets: ArrayLike | None = None,
    transforms: ArrayLike | None = None,
) -> GeometryArray:
    """Return an array of copies of this geometry with optional offsets and/or linear transforms.

    This method creates a :class:`GeometryArray` containing multiple copies of this
    geometry. When both ``offsets`` and ``transforms`` are provided, transforms are
    applied to each copy before the translation given by offsets is applied.

    Parameters
    ----------
    offsets : Optional[ArrayLike] = None
        Optional array of offset vectors with shape (N, 3) where N is the number of
        geometries. Each row specifies the (x, y, z) translation for one geometry
        (after any transform is applied). If not provided, no additional translation
        is applied beyond any transforms.
    transforms : Optional[ArrayLike] = None
        Optional array of 4x4 linear-only transform matrices with shape (N, 4, 4).
        Each transform must be a valid homogeneous linear transform
        (rotation/reflection/scale/shear) with no translation component.
        Typical transforms can be created using ``Transformed.rotation``,
        ``Transformed.reflection``, or ``Transformed.scaling``.

    Returns
    -------
    :class:`GeometryArray`
        Array containing N copies of this geometry.

    Notes
    -----
    - ``offsets`` represent all per-instance translation.
    - ``transforms`` represent linear transforms only and must not contain translation.
    - If both ``offsets`` and ``transforms`` are ``None``, the array contains a single
      instance of the base geometry.
    - If both are provided, they must have the same length and transforms are applied
      before the translation given by offsets.
    - Adjoint/autodiff is not currently supported for ``GeometryArray``.

    Example
    -------
    >>> import tidy3d as td
    >>> import numpy as np
    >>> box = td.Box(size=(1, 1, 1))
    >>> # Create a 2x2 grid of boxes using offsets
    >>> offsets = [[0, 0, 0], [2, 0, 0], [0, 2, 0], [2, 2, 0]]
    >>> array = box.array(offsets=offsets)
    >>> # Create array using linear transforms only (rotation around z-axis)
    >>> transforms = [np.eye(4), td.Transformed.rotation(np.pi/4, 2)]
    >>> array = box.array(transforms=transforms)
    >>> # Both None gives single instance of base geometry
    >>> array = box.array()
    """
    from .array import GeometryArray

    return GeometryArray(geometry=self, offsets=offsets, transforms=transforms)


""" Field and coordinate transformations """


def car_2_sph(x: float, y: float, z: float) -> tuple[float, float, float]:
    """Convert Cartesian to spherical coordinates.

    Parameters
    ----------
    x : float
        x coordinate relative to ``local_origin``.
    y : float
        y coordinate relative to ``local_origin``.
    z : float
        z coordinate relative to ``local_origin``.

    Returns
    -------
    tuple[float, float, float]
        r, theta, and phi coordinates relative to ``local_origin``.
    """
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    return r, theta, phi


def sph_2_car(r: float, theta: float, phi: float) -> tuple[float, float, float]:
    """Convert spherical to Cartesian coordinates.

    Parameters
    ----------
    r : float
        radius.
    theta : float
        polar angle (rad) downward from x=y=0 line.
    phi : float
        azimuthal (rad) angle from y=z=0 line.

    Returns
    -------
    tuple[float, float, float]
        x, y, and z coordinates relative to ``local_origin``.
    """
    r_sin_theta = r * np.sin(theta)
    x = r_sin_theta * np.cos(phi)
    y = r_sin_theta * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z


def sph_2_car_field(
    f_r: float, f_theta: float, f_phi: float, theta: float, phi: float
) -> tuple[complex, complex, complex]:
    """Convert vector field components in spherical coordinates to cartesian.

    Parameters
    ----------
    f_r : float
        radial component of the vector field.
    f_theta : float
        polar angle component of the vector fielf.
    f_phi : float
        azimuthal angle component of the vector field.
    theta : float
        polar angle (rad) of location of the vector field.
    phi : float
        azimuthal angle (rad) of location of the vector field.

    Returns
    -------
    tuple[float, float, float]
        x, y, and z components of the vector field in cartesian coordinates.
    """
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    f_x = f_r * sin_theta * cos_phi + f_theta * cos_theta * cos_phi - f_phi * sin_phi
    f_y = f_r * sin_theta * sin_phi + f_theta * cos_theta * sin_phi + f_phi * cos_phi
    f_z = f_r * cos_theta - f_theta * sin_theta
    return f_x, f_y, f_z


def car_2_sph_field(
    f_x: float, f_y: float, f_z: float, theta: float, phi: float
) -> tuple[complex, complex, complex]:
    """Convert vector field components in cartesian coordinates to spherical.

    Parameters
    ----------
    f_x : float
        x component of the vector field.
    f_y : float
        y component of the vector fielf.
    f_z : float
        z component of the vector field.
    theta : float
        polar angle (rad) of location of the vector field.
    phi : float
        azimuthal angle (rad) of location of the vector field.

    Returns
    -------
    tuple[float, float, float]
        radial (s), elevation (theta), and azimuthal (phi) components
        of the vector field in spherical coordinates.
    """
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    f_r = f_x * sin_theta * cos_phi + f_y * sin_theta * sin_phi + f_z * cos_theta
    f_theta = f_x * cos_theta * cos_phi + f_y * cos_theta * sin_phi - f_z * sin_theta
    f_phi = -f_x * sin_phi + f_y * cos_phi
    return f_r, f_theta, f_phi


def kspace_2_sph(ux: float, uy: float, axis: Axis) -> tuple[float, float]:
    """Convert normalized k-space coordinates to angles.

    Parameters
    ----------
    ux : float
        normalized kx coordinate.
    uy : float
        normalized ky coordinate.
    axis : int
        axis along which the observation plane is oriented.

    Returns
    -------
    tuple[float, float]
        theta and phi coordinates relative to ``local_origin``.
    """
    phi_local = np.arctan2(uy, ux)
    with np.errstate(invalid="ignore"):
        theta_local = np.arcsin(np.sqrt(ux**2 + uy**2))
    # Spherical coordinates rotation matrix reference:
    # https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula#Matrix_notation
    if axis == 2:
        return theta_local, phi_local

    x = np.cos(theta_local)
    y = np.sin(theta_local) * np.cos(phi_local)
    z = np.sin(theta_local) * np.sin(phi_local)

    if axis == 1:
        x, y, z = y, x, z

    theta = np.arccos(z)
    phi = np.arctan2(y, x)
    return theta, phi
