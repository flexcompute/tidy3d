"""Semiconductor doping definitions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import xarray as xr
from flexcompute.core._migration.em.numerical.raw import doping as doping_numerics
from pydantic import Field, NonNegativeFloat, PositiveFloat, model_validator

from tidy3d.components.autograd import TracedSize
from tidy3d.components.base import cached_property
from tidy3d.components.data.data_array import SpatialDataArray
from tidy3d.components.geometry.base import Box
from tidy3d.constants import MICROMETER, PERCMCUBE, inf
from tidy3d.log import log

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

    from tidy3d.compat import Self

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

    from tidy3d.compat import Self


class AbstractDopingBox(Box):
    """Derived class from Box to deal with dopings"""

    # Override size so that we can set default values
    size: TracedSize = Field(
        default=(inf, inf, inf),
        title="Size",
        description="Size in x, y, and z directions.",
        json_schema_extra={"units": MICROMETER},
    )

    @staticmethod
    def _complete_coords(
        coords: dict[str, ArrayLike], meshgrid: bool
    ) -> dict[str, NDArray[np.floating]]:
        """Return x, y, z coordinate arrays, filling omitted axes at zero."""

        coords_full = {axis: np.asarray(coords[axis]) for axis in "xyz" if axis in coords}
        if len(coords_full) == 3 and meshgrid:
            return coords_full

        if meshgrid:
            fill_value = np.array([0.0])
        else:
            fill_shape = next(iter(coords_full.values())).shape if coords_full else (1,)
            fill_value = np.zeros(fill_shape)

        coords_full = {axis: coords_full.get(axis, fill_value) for axis in "xyz"}
        if not meshgrid:
            x, y, z = np.broadcast_arrays(coords_full["x"], coords_full["y"], coords_full["z"])
            coords_full = {"x": x, "y": y, "z": z}

        return coords_full

    def _coords_for_inside(
        self, coords_full: dict[str, NDArray[np.floating]]
    ) -> dict[str, NDArray[np.floating]]:
        """Nudge exact finite-boundary coordinates inward before geometry inside checks."""

        coords_inside = {}
        for axis_index, axis in enumerate("xyz"):
            coord = coords_full[axis]
            lower = self.bounds[0][axis_index]
            upper = self.bounds[1][axis_index]
            center = self.center[axis_index]

            coord_inside = coord
            if np.isfinite(lower):
                coord_inside = np.where(
                    coord_inside == lower, np.nextafter(lower, center), coord_inside
                )
            if np.isfinite(upper):
                coord_inside = np.where(
                    coord_inside == upper, np.nextafter(upper, center), coord_inside
                )

            coords_inside[axis] = coord_inside

        return coords_inside

    def _get_inside_and_coords(
        self, coords: dict[str, ArrayLike], meshgrid: bool = True
    ) -> tuple[
        dict[str, NDArray[np.floating]],
        NDArray[np.bool_],
        NDArray[np.floating],
        NDArray[np.floating],
        NDArray[np.floating],
    ]:
        """Returns completed coordinates and locations inside the doping box."""

        coords_full = self._complete_coords(coords=coords, meshgrid=meshgrid)
        x = coords_full["x"]
        y = coords_full["y"]
        z = coords_full["z"]
        coords_inside = self._coords_for_inside(coords_full)
        x_inside = coords_inside["x"]
        y_inside = coords_inside["y"]
        z_inside = coords_inside["z"]

        if meshgrid:
            indices_in_box = self.inside_meshgrid(x_inside, y_inside, z_inside)
            X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
        else:
            X, Y, Z = x, y, z
            indices_in_box = self.inside(x_inside, y_inside, z_inside)

        return coords_full, indices_in_box, X, Y, Z


class ConstantDoping(AbstractDopingBox):
    """
    Sets constant doping :math:`N` in the specified box with a :py:attr:`~.Box.size` and :py:attr:`concentration`. For translationally invariant behavior in one dimension, the box must have infinite size in the
    homogenous (invariant) direction.

    Example
    -------
    >>> import tidy3d as td
    >>> box_coords = [
    ...     [-1, -1, -1],
    ...     [1, 1, 1]
    ... ]
    >>> constant_box1 = td.ConstantDoping(center=(0, 0, 0), size=(2, 2, 2), concentration=1e18)
    >>> constant_box2 = td.ConstantDoping.from_bounds(rmin=box_coords[0], rmax=box_coords[1], concentration=1e18)
    """

    concentration: NonNegativeFloat = Field(
        default=0,
        title="Doping concentration density.",
        description="Doping concentration density.",
        json_schema_extra={"units": PERCMCUBE},
    )

    def _get_contrib(self, coords: dict, meshgrid: bool = True) -> NDArray:
        """Returns the contribution to the doping a the locations specified in coords"""

        _, indices_in_box, X, _, _ = self._get_inside_and_coords(coords=coords, meshgrid=meshgrid)

        contrib = np.zeros(X.shape)
        contrib[indices_in_box] = self.concentration

        return contrib.squeeze()


class GaussianDoping(AbstractDopingBox):
    """Sets a gaussian doping in the specified box. For translationally invariant behavior in one dimension, the box must have infinite size in the
    homogenous (invariant) direction.

    Notes
    -----
    The concentration is a product of one factor per axis,
    :math:`N = N_{\\text{max}} f_x f_y f_z`. Along an axis the factor is 1 more than
    :math:`\\text{width}` away from both faces of the box, and within :math:`\\text{width}` of a
    face other than the ``source`` face it is

    .. math::

        f = \\exp \\left[
        - \\ln \\left( \\frac{N_{\\text{max}}}{N_{\\text{ref}}} \\right)
        \\left( \\frac{\\text{width} - d}{\\text{width}} \\right)^2
        \\right],

    where :math:`d` is the distance to that face. On the ``source`` face the factor stays 1. A face
    therefore has concentration :math:`N_{\\text{ref}}`, an edge where two such faces meet
    :math:`N_{\\text{ref}}^2 / N_{\\text{max}}`, and a corner of three
    :math:`N_{\\text{ref}}^3 / N_{\\text{max}}^2`. Along an axis where the box is thinner than
    :math:`2\\,\\text{width}`, the lower face (``xmin``, ``ymin`` or ``zmin``) sets the factor where
    the two transitions overlap.

    Example
    -------
    >>> import tidy3d as td
    >>> box_coords = [
    ...     [-1, -1, -1],
    ...     [1, 1, 1]
    ... ]
    >>> gaussian_box1 = td.GaussianDoping(
    ...     center=(0, 0, 0),
    ...     size=(2, 2, 2),
    ...     ref_con=1e15,
    ...     concentration=1e18,
    ...     width=0.1,
    ...     source="xmin"
    ... )
    >>> gaussian_box2 = td.GaussianDoping.from_bounds(
    ...     rmin=box_coords[0],
    ...     rmax=box_coords[1],
    ...     ref_con=1e15,
    ...     concentration=1e18,
    ...     width=0.1,
    ...     source="xmin"
    ... )
    """

    ref_con: PositiveFloat = Field(
        title="Reference concentration.",
        description="Concentration on a face of the box other than the ``source`` face. Where two "
        "or three such faces meet, at an edge or a corner, the concentration falls to "
        "``ref_con**2 / concentration`` or ``ref_con**3 / concentration**2``.",
        json_schema_extra={"units": PERCMCUBE},
    )

    concentration: PositiveFloat = Field(
        title="Concentration",
        description="The concentration at the center of the box.",
        json_schema_extra={"units": PERCMCUBE},
    )

    width: PositiveFloat = Field(
        title="Width of the gaussian.",
        description="Width of the gaussian. The concentration will transition from "
        "``concentration`` at the center of the box to ``ref_con`` at a face "
        "of the box in a distance equal to ``width``. ",
        json_schema_extra={"units": MICROMETER},
    )

    source: Literal["xmin", "xmax", "ymin", "ymax", "zmin", "zmax"] = Field(
        default="xmin",
        title="Source face",
        description="Specifies the side of the box acting as the source, i.e., "
        "the face specified does not have a gaussian evolution normal to it, instead "
        "the concentration is constant from this face. Accepted values for ``source`` "
        "are [``xmin``, ``xmax``, ``ymin``, ``ymax``, ``zmin``, ``zmax``]",
    )

    # Stays a client check because it is a warning, which the construction path does not
    # re-emit. The construction path runs before this validator, so a box that fails a core
    # rule raises without logging this warning.
    @model_validator(mode="after")
    def _validate_width_vs_size(self: Self) -> Self:
        """Warn if box size may be too small for the specified transition width."""
        val = self.width
        size = self.size
        for i, s in enumerate(size):
            if not np.isinf(s) and s < 2 * val:
                dim_name = ["x", "y", "z"][i]
                log.warning(
                    f"Box size in '{dim_name}' direction ({s} μm) is less than "
                    f"'2*width' ({2 * val} μm) for 'GaussianDoping'. "
                    "This may result in unexpected behavior in the transition region."
                )
        return self

    @cached_property
    def sigma(self) -> float:
        """The sigma parameter of the pseudo-gaussian"""
        return doping_numerics.gaussian_doping_sigma(
            width=self.width, ref_con=self.ref_con, concentration=self.concentration
        )

    def _get_contrib(self, coords: dict[str, ArrayLike], meshgrid: bool = True) -> NDArray:
        """Returns the contribution to the doping a the locations specified in coords"""

        _, indices_in_box, X, Y, Z = self._get_inside_and_coords(coords=coords, meshgrid=meshgrid)
        total_contrib = doping_numerics.gaussian_doping_concentration(
            X,
            Y,
            Z,
            inside=indices_in_box,
            bounds=self.bounds,
            width=self.width,
            ref_con=self.ref_con,
            concentration=self.concentration,
            source=self.source,
        )
        return total_contrib.squeeze()


class CustomDoping(AbstractDopingBox):
    """Sets a custom doping profile in the specified box.

    :class:`.CustomDoping` wraps a :class:`.SpatialDataArray` as an additive doping box so
    arbitrary spatial doping profiles can be summed with other doping boxes in ``N_a`` and
    ``N_d``.

    Example
    -------
    >>> import tidy3d as td
    >>> import numpy as np
    >>> box_coords = [
    ...     [-1, -1, -1],
    ...     [1, 1, 1]
    ... ]
    >>> x = np.linspace(-1, 1, 5)
    >>> y = np.linspace(-1, 1, 5)
    >>> z = np.linspace(-1, 1, 5)
    >>> data = np.random.rand(5, 5, 5)*1e18
    >>> concentration = td.SpatialDataArray(
    ...     data=data,
    ...     coords={'x': x, 'y': y, 'z': z},
    ... )
    >>> custom_box1 = td.CustomDoping(
    ...     center=(0, 0, 0),
    ...     size=(2, 2, 2),
    ...     concentration=concentration
    ... )
    >>> custom_box2 = td.CustomDoping.from_bounds(
    ...     rmin=box_coords[0],
    ...     rmax=box_coords[1],
    ...     concentration=concentration
    ... )
    """

    concentration: SpatialDataArray = Field(
        title="Doping concentration data array.",
        description="Doping concentration data array.",
        json_schema_extra={"units": PERCMCUBE},
    )

    def _get_contrib(self, coords: dict, meshgrid: bool = True) -> NDArray:
        """Returns the contribution to the doping a the locations specified in coords"""

        coords_full, indices_in_box, X, _Y, _Z = self._get_inside_and_coords(
            coords=coords, meshgrid=meshgrid
        )

        contrib = np.zeros(X.shape)
        # interpolate
        if meshgrid:
            interp_result = self.concentration.interp(coords_full)
            contrib[indices_in_box] = interp_result.values[indices_in_box]
        else:
            # X, Y, Z are 1D arrays of coordinates
            # interp_result = self.concentration.interp(coords_full)
            # contrib = np.zeros(X.shape)
            # contrib[indices_in_box] = interp_result.values[indices_in_box]

            interp_coords = coords_full
            interp_da = {
                name: xr.DataArray(data, dims="new_dim") for name, data in interp_coords.items()
            }
            interp_res = self.concentration.interp(
                **interp_da, kwargs={"fill_value": 0, "bounds_error": False}
            )
            contrib[indices_in_box] = interp_res.values[indices_in_box]

        return contrib.squeeze()


DopingBoxType = ConstantDoping | GaussianDoping | CustomDoping
