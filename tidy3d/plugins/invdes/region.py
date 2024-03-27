# container for specification fully defining the inverse design problem

import abc
import typing

import jax.numpy as jnp
import numpy as np
import pydantic.v1 as pd

import tidy3d as td
from tidy3d.components.types import Coordinate
import tidy3d.plugins.adjoint as tda

from .base import InvdesBaseModel
from .transformation import FilterProject, TransformationType
from .penalty import ErosionDilationPenalty, PenaltyType

# TODO: support auto handling of symmetry in parameters


class DesignRegion(InvdesBaseModel, abc.ABC):
    """Base class for design regions in the ``invdes`` plugin."""

    size: typing.Tuple[pd.PositiveFloat, pd.PositiveFloat, pd.PositiveFloat] = pd.Field(
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

    eps_bounds: typing.Tuple[float, float] = pd.Field(
        ...,
        title="Relative Permittivity Bounds",
        description="Minimum and maximum relative permittivity expressed to the design region.",
    )

    transformations: typing.Tuple[TransformationType, ...] = pd.Field(
        (),
        title="Transformations",
        description="Transformations that get applied from first to last on the parameter array."
        "The end result of the transformations should be the material density of the design region "
        ". With floating point values between (0, 1), 0 corresponds to the minimum relative "
        "permittivity and 1 corresponds to the maximum relative permittivity. "
        "Specific permittivity values given the density array are determined by ``eps_bounds``.",
    )

    penalties: typing.Tuple[PenaltyType, ...] = pd.Field(
        (),
        title="Penalties",
        description="Set of penalties that get evaluated on the material density. Note that the "
        "penalties are applied after ``transformations`` are applied. Penalty weights can be set "
        "inside of the penalties directly through the ``.weight`` field.",
    )

    @property
    def geometry(self) -> td.Box:
        """``Box`` corresponding to this design region."""
        return td.Box(center=self.center, size=self.size)

    def material_density(self, params: jnp.ndarray) -> jnp.ndarray:
        """Evaluate the transformations on a parameter array to give the material density (0,1)."""
        for transformation in self.transformations:
            params = self.evaluate_transformation(transformation=transformation, params=params)
        return params

    def penalty_value(self, data: jnp.ndarray) -> jnp.ndarray:
        """Evaluate the transformations on a dataset."""

        if not self.penalties:
            return 0.0

        # sum the penalty values scaled by their weights (optional)
        material_density = self.material_density(data)
        penalty_values = [
            self.evaluate_penalty(penalty=penalty, material_density=material_density)
            for penalty in self.penalties
        ]
        return jnp.sum(jnp.array(penalty_values))

    @abc.abstractmethod
    def evaluate_transformation(self, transformation: TransformationType) -> float:
        """How this design region evaluates a transformation given some passed information."""

    @abc.abstractmethod
    def evaluate_penalty(self, penalty: PenaltyType) -> float:
        """How this design region evaluates a penalty given some passed information."""

    @abc.abstractmethod
    def to_jax_structure(self, *args, **kwargs) -> tda.JaxStructure:
        """Convert this ``DesignRegion`` into a custom ``JaxStructure``. Implement in subclass."""

    def to_structure(self, *args, **kwargs) -> td.Structure:
        """Convert this ``DesignRegion`` into a ``Structure``. Implement in subclass."""
        return self.to_jax_structure(*args, **kwargs).to_structure()


class TopologyDesignRegion(DesignRegion):
    """Design region as a pixellated permittivity grid."""

    pixel_size: pd.PositiveFloat
    transformations: typing.Tuple[FilterProject, ...] = ()
    penalties: typing.Tuple[ErosionDilationPenalty, ...] = ()

    @staticmethod
    def _check_params(params: jnp.ndarray):
        """Ensure ``params`` are between 0 and 1."""
        if np.any(params < 0) or np.any(params > 1):
            raise ValueError(
                "Parameters in the 'invdes' plugin's topology optimization feature "
                "are restricted to be between 0 and 1."
            )

    @property
    def params_shape(self) -> typing.Tuple[int, int, int]:
        """Shape of the parameters array in (x, y, z), given the ``pixel_size`` and bounds."""
        # rmin, rmax = np.array(self.geometry.bounds)
        # lengths = rmax - rmin
        side_lengths = np.array(self.size)
        num_pixels = np.ceil(side_lengths / self.pixel_size)
        # TODO: if the structure is infinite but the simulation is finite, need reduced bounds
        num_pixels[np.isinf(num_pixels)] = 1
        return tuple(int(n) for n in num_pixels)

    @property
    def params_random(self) -> np.ndarray:
        """Convenience for generating random parameters between (0,1) with correct shape."""
        return np.random.random(self.params_shape)

    @property
    def params_zeros(self):
        """Convenience for generating random parameters of all 0 values with correct shape."""
        return np.zeros(self.params_shape)

    @property
    def params_ones(self):
        """Convenience for generating random parameters of all 1 values with correct shape."""
        return np.ones(self.params_shape)

    @property
    def coords(self) -> typing.Dict[str, typing.List[float]]:
        """Coordinates for the custom medium corresponding to this design region."""

        lengths = np.array(self.size)

        rmin, rmax = self.geometry.bounds
        params_shape = self.params_shape

        coords = dict()
        for dim, ptmin, ptmax, length, num_pts in zip("xyz", rmin, rmax, lengths, params_shape):
            step_size = length / num_pts
            if np.isinf(length):
                coord_vals = [self.center["xyz".index(dim)]]
            else:
                coord_vals = np.linspace(ptmin + step_size / 2, ptmax - step_size / 2, num_pts)
                coord_vals = coord_vals.tolist()
            coords[dim] = coord_vals

        coords["f"] = [td.C_0]  # TODO: is this a safe choice?
        return coords

    def eps_values(self, params: jnp.ndarray) -> jnp.ndarray:
        """Values for the custom medium permittivity."""

        self._check_params(params)

        material_density = self.material_density(params)
        eps_min, eps_max = self.eps_bounds
        arr_3d = eps_min + material_density * (eps_max - eps_min)
        arr_3d = arr_3d.reshape(params.shape)
        return jnp.expand_dims(arr_3d, axis=-1)

    def to_jax_structure(self, params: jnp.ndarray) -> tda.JaxStructureStaticGeometry:
        """Convert this ``DesignRegion`` into a custom ``JaxStructure``."""
        self._check_params(params)

        coords = self.coords
        eps_values = self.eps_values(params)
        data_array = tda.JaxDataArray(values=eps_values, coords=coords)
        field_components = {f"eps_{dim}{dim}": data_array for dim in "xyz"}
        eps_dataset = tda.JaxPermittivityDataset(**field_components)
        medium = tda.JaxCustomMedium(eps_dataset=eps_dataset)
        return tda.JaxStructureStaticGeometry(geometry=self.geometry, medium=medium)

    def to_mesh_override_structure(self) -> td.MeshOverrideStructure:
        """Generate mesh override structure for this ``DesignRegion`` using ``pixel_size`` step."""
        return td.MeshOverrideStructure(
            geometry=self.geometry,
            dl=3 * [self.pixel_size],
            enforce=True,
        )

    def evaluate_transformation(
        self, transformation: TransformationType, params: jnp.ndarray
    ) -> jnp.ndarray:
        """Evaluate a transformation, passing in design_region_dl."""
        self._check_params(params)
        return transformation.evaluate(spatial_data=params, design_region_dl=self.pixel_size)

    def evaluate_penalty(
        self, penalty: ErosionDilationPenalty, material_density: jnp.ndarray
    ) -> float:
        """Evaluate an erosion-dilation penalty, passing in pixel_size."""
        return penalty.evaluate(x=material_density, pixel_size=self.pixel_size)


DesignRegionType = typing.Union[TopologyDesignRegion]
