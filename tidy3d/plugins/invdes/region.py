# container for everything defining the inverse design
import abc
import typing

import jax.numpy as jnp
import numpy as np
import jax
import pydantic.v1 as pd

import tidy3d as td
from tidy3d.components.types import annotate_type, Size, Coordinate
import tidy3d.plugins.adjoint as tda

from .transformation import TransformationType
from .base import InvdesBaseModel
from .penalty import PenaltyType, ErosionDilationPenalty, RadiusPenalty, Penalty


class DesignRegion(InvdesBaseModel, abc.ABC):
    size: Size = pd.Field(
        ...,
        title="Size",
        description="Size in x, y, and z directions.",
        units=td.constants.MICROMETER,
    )

    center: Coordinate = pd.Field(
        ...,
        title="Center",
        description="Center of object in x, y, and z.",
        units=td.constants.MICROMETER,
    )

    params_shape: typing.Tuple[pd.PositiveInt, pd.PositiveInt, pd.PositiveInt] = pd.Field(
        ...,
        title="Parameters Shape",
        description="Shape of the parameters array in (x, y, z) directions.",
    )

    eps_bounds: typing.Tuple[float, float] = pd.Field(
        ...,
        title="",
        description="",
    )

    # TODO: support symmetry
    # symmetry: typing.Tuple[Symmetry, Symmetry, Symmetry] = pd.Field(
    #     (0, 0, 0),
    #     title="Symmetry",
    #     description="Symmetry of the design region. If specified, values on the '-' side of the "
    #     "central axes will be filled in using the values on the '+' side of the axis.",
    # )

    transformations: typing.Tuple[annotate_type(TransformationType), ...] = pd.Field(
        (),
        title="Transformations",
        description="Transformations that get applied from first to last on the parameter array."
        "The end result of the transformations should be the material density of the design region "
        ". With floating point values between (0, 1), where 0 corresponds to the minimum relative "
        "permittivity and 1 corresponds to the maximum relative permittivity. "
        "Set 'eps_bounds' to determine what permittivity values are evaluated given this density.",
    )

    # TODO: instead of penalty_weights, add `weight` to the `tdi.Penalty` instead.
    penalties: typing.Tuple[annotate_type(PenaltyType), ...] = pd.Field(
        (),
        title="Penalties",
        description="Set of penalties that get evaluated on the material density. Note that the "
        "penalties are applied after 'transformations' are applied. To set the weights of the "
        "penalties, set 'penalty_weights'. ",
    )

    penalty_weights: typing.Tuple[float, ...] = pd.Field(
        None,
        title="Penalty Weights",
        description="Sets the weight of each penalty in '.penalties'. If not 'None'.",
    )

    @property
    def geometry(self) -> td.Box:
        """``Box`` corresponding to this design region."""
        return td.Box(center=self.center, size=self.size)

    def material_density(self, data: jnp.ndarray) -> jnp.ndarray:
        """Evaluate the transformations on a parameter array to give the material density (0,1)."""
        for transformation in self.transformations:
            data = transformation.evaluate(data)
        return data

    @property
    def _num_penalties(self) -> int:
        """How many penalties are present."""
        return len(self.penalties)

    @property
    def _penalty_weights(self) -> jnp.ndarray:
        """Penalty weights as an array."""

        if not self._num_penalties:
            raise ValueError("Can't get penalty weights because `penalties` are not defined.")

        if self.penalty_weights is None:
            return jnp.ones(self._num_penalties)

        return jnp.array(self.penalty_weights)

    def penalty_value(self, data: jnp.ndarray) -> jnp.ndarray:
        """Evaluate the transformations on a dataset."""

        if not self.penalties:
            return 0.0

        # sum the penalty values scaled by their weights (optional)
        material_density = self.material_density(data)
        penalty_values = [penalty.evaluate(material_density) for penalty in self.penalties]
        penalties_weighted = self._penalty_weights * jnp.array(penalty_values)
        return jnp.sum(penalties_weighted)

    @abc.abstractmethod
    def to_jax_structure(self) -> tda.JaxStructure:
        """Convert this ``DesignRegion`` into a custom ``JaxStructure``. Implement in subclass."""


class TopologyDesignRegion(DesignRegion):
    """Design region as a pixellated permittivity grid."""

    transformations: typing.Tuple[annotate_type(TransformationType), ...] = ()
    penalties: typing.Tuple[annotate_type(typing.Union[ErosionDilationPenalty, Penalty]), ...] = ()

    @property
    def step_sizes(self) -> typing.Tuple[float, float, float]:
        """Step sizes along x, y, z."""
        bounds = np.array(self.geometry.bounds)
        return tuple((bounds[1] - bounds[0]).tolist())

    @property
    def coords(self) -> typing.Dict[str, typing.List[float]]:
        """Coordinates for the custom medium corresponding to this design region."""

        rmin, rmax = self.geometry.bounds

        coords = dict()

        for center, coord_key, ptmin, ptmax, num_pts in zip(
            self.center, "xyz", rmin, rmax, self.params_shape
        ):
            size = ptmax - ptmin
            if np.isinf(size):
                coord_vals = num_pts * [center]
            else:
                step_size = size / num_pts
                coord_vals = np.linspace(
                    ptmin + step_size / 2, ptmax - step_size / 2, num_pts
                ).tolist()
            coords[coord_key] = coord_vals

        coords["f"] = [td.C_0]  # TODO: is this a safe choice?
        return coords

    def eps_values(self, params: jnp.ndarray) -> jnp.ndarray:
        """Values for the custom medium permittivity."""
        material_density = self.material_density(params)
        eps_min, eps_max = self.eps_bounds
        arr_3d = eps_min + material_density * (eps_max - eps_min)
        arr_3d = jax.lax.stop_gradient(arr_3d)
        arr_3d = arr_3d.reshape(params.shape)
        return jnp.expand_dims(arr_3d, axis=-1)

    def to_jax_structure(self, params: jnp.ndarray) -> tda.JaxStructureStaticGeometry:
        """Convert this ``DesignRegion`` into a custom ``JaxStructure``."""
        coords = self.coords
        eps_values = self.eps_values(params)
        data_array = tda.JaxDataArray(values=eps_values, coords=coords)
        field_components = {f"eps_{dim}{dim}": data_array for dim in "xyz"}
        eps_dataset = tda.JaxPermittivityDataset(**field_components)
        medium = tda.JaxCustomMedium(eps_dataset=eps_dataset)
        return tda.JaxStructureStaticGeometry(geometry=self.geometry, medium=medium)


class ShapeDesignRegion(DesignRegion):
    transformations: typing.Literal[()] = ()
    penalties: typing.Tuple[annotate_type(typing.Union[RadiusPenalty, Penalty]), ...] = ()


class LevelSetDesignRegion(DesignRegion):
    """Implement later"""

    pass


DesignRegionType = typing.Union[TopologyDesignRegion]
