# container for specification fully defining the inverse design problem
from __future__ import annotations

import abc
import warnings
from typing import TYPE_CHECKING, Any, Literal

import autograd.numpy as anp
import numpy as np
from autograd import elementwise_grad, grad
from pydantic import Field, PositiveFloat, field_validator, model_validator

import tidy3d as td
from tidy3d.components.types import TYPE_TAG_STR, Coordinate, Size
from tidy3d.exceptions import ValidationError

from .base import InvdesBaseModel
from .initialization import InitializationSpecType, UniformInitializationSpec
from .penalty import PenaltyType
from .transformation import TransformationType

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from tidy3d.compat import Self
    from tidy3d.plugins.autograd.invdes.symmetries import MirrorSymmetry


class DesignRegion(InvdesBaseModel, abc.ABC):
    """Base class for design regions in the ``invdes`` plugin."""

    size: Size = Field(
        title="Size",
        description="Size in x, y, and z directions.",
        json_schema_extra={"units": td.constants.MICROMETER},
    )

    center: Coordinate = Field(
        title="Center",
        description="Center of object in x, y, and z.",
        json_schema_extra={"units": td.constants.MICROMETER},
    )

    eps_bounds: tuple[float, float] = Field(
        title="Relative Permittivity Bounds",
        description="Minimum and maximum relative permittivity expressed to the design region.",
    )

    transformations: tuple[TransformationType, ...] = Field(
        (),
        title="Transformations",
        description="Transformations that get applied from first to last on the parameter array."
        "The end result of the transformations should be the material density of the design region "
        ". With floating point values between (0, 1), 0 corresponds to the minimum relative "
        "permittivity and 1 corresponds to the maximum relative permittivity. "
        "Specific permittivity values given the density array are determined by ``eps_bounds``.",
    )

    penalties: tuple[PenaltyType, ...] = Field(
        (),
        title="Penalties",
        description="Set of penalties that get evaluated on the material density. Note that the "
        "penalties are applied after ``transformations`` are applied. Penalty weights can be set "
        "inside of the penalties directly through the ``.weight`` field.",
    )

    initialization_spec: InitializationSpecType = Field(
        default_factory=lambda: UniformInitializationSpec(value=0.5),
        title="Initialization Specification",
        description="Specification of how to initialize the parameters in the design region.",
        discriminator=TYPE_TAG_STR,
    )

    @field_validator("eps_bounds")
    @classmethod
    def _validate_ge_one(cls, v: tuple[float, float]) -> tuple[float, float]:
        if any(vi < 1 for vi in v):
            raise ValueError("Each value in 'eps_bounds' must be '>=1.0'.")
        return v

    @model_validator(mode="after")
    def _validate_eps_bounds(self) -> Self:
        if self.eps_bounds[1] < self.eps_bounds[0]:
            raise ValidationError(
                f"Maximum relative permittivity ({self.eps_bounds[1]}) must be "
                f"greater than minimum relative permittivity ({self.eps_bounds[0]})."
            )
        return self

    @property
    def geometry(self) -> td.Box:
        """``Box`` corresponding to this design region."""
        return td.Box(center=self.center, size=self.size)

    def material_density(
        self, params: anp.ndarray, symmetry: MirrorSymmetry | None = None
    ) -> anp.ndarray:
        """Evaluate transformations on parameters to give the material density (0, 1)."""
        for transformation in self.transformations:
            params = self.evaluate_transformation(
                transformation=transformation,
                params=params,
                symmetry=symmetry,
            )
        return params

    def penalty_value(
        self, data: anp.ndarray, symmetry: MirrorSymmetry | None = None
    ) -> anp.ndarray:
        """Evaluate the transformations on a dataset."""

        if not self.penalties:
            return 0.0

        # sum the penalty values scaled by their weights (optional)
        material_density = self.material_density(data, symmetry=symmetry)
        penalty_values = [
            self.evaluate_penalty(
                penalty=penalty,
                material_density=material_density,
                symmetry=symmetry,
            )
            for penalty in self.penalties
        ]
        return anp.sum(anp.array(penalty_values))

    @abc.abstractmethod
    def evaluate_transformation(
        self,
        transformation: None,
        params: anp.ndarray,
        symmetry: MirrorSymmetry | None = None,
    ) -> anp.ndarray:
        """How this design region evaluates a transformation given some passed information."""

    @abc.abstractmethod
    def evaluate_penalty(
        self,
        penalty: None,
        material_density: anp.ndarray,
        symmetry: MirrorSymmetry | None = None,
    ) -> float:
        """How this design region evaluates a penalty given some passed information."""

    @abc.abstractmethod
    def to_structure(self, *args: Any, **kwargs: Any) -> td.Structure:
        """Convert this ``DesignRegion`` into a ``Structure`` with tracers. Implement in subs."""

    @property
    def initial_parameters(self) -> np.ndarray:
        """Generate initial parameters based on the initialization specification."""
        params0 = self.initialization_spec.create_parameters(self.params_shape)
        self._check_params(params0)
        return params0


class TopologyDesignRegion(DesignRegion):
    """Design region as a pixellated permittivity grid."""

    pixel_size: PositiveFloat = Field(
        title="Pixel Size",
        description="Pixel size of the design region in x, y, z. For now, we only support the same "
        "pixel size in all 3 dimensions. If ``TopologyDesignRegion.override_structure_dl`` is left "
        "``None``, the ``pixel_size`` will determine the FDTD mesh size in the design region. "
        "Therefore, if your pixel size is large compared to the FDTD grid size, we recommend "
        "setting the ``override_structure_dl`` directly to "
        "a value on the same order as the grid size.",
    )

    uniform: tuple[bool, bool, bool] = Field(
        (False, False, True),
        title="Uniform",
        description="Axes along which the design should be uniform. By default, the structure "
        "is assumed to be uniform, i.e. invariant, in the z direction.",
    )

    transformations: tuple[TransformationType, ...] = Field(
        (),
        title="Transformations",
        description="Transformations that get applied from first to last on the parameter array."
        "The end result of the transformations should be the material density of the design region "
        ". With floating point values between (0, 1), 0 corresponds to the minimum relative "
        "permittivity and 1 corresponds to the maximum relative permittivity. "
        "Specific permittivity values given the density array are determined by ``eps_bounds``.",
    )
    penalties: tuple[PenaltyType, ...] = Field(
        (),
        title="Penalties",
        description="Set of penalties that get evaluated on the material density. Note that the "
        "penalties are applied after ``transformations`` are applied. Penalty weights can be set "
        "inside of the penalties directly through the ``.weight`` field.",
    )

    override_structure_dl: PositiveFloat | Literal[False] | None = Field(
        None,
        title="Design Region Override Structure",
        description="Defines grid size when adding an ``override_structure`` to the "
        "``JaxSimulation.grid_spec`` corresponding to this design region. "
        "If left ``None``, ``invdes`` will mesh the simulation with the same resolution as the "
        "``pixel_size``. "
        "This is advised if the pixel size is relatively close to the FDTD grid size. "
        "Supplying ``False`` will completely leave out the override structure.",
    )

    priority: int | None = Field(
        None,
        title="Priority",
        description="Priority of the structure applied in structure overlapping region. "
        "The material property in the overlapping region is dictated by the structure "
        "of higher priority. For structures of equal priority, "
        "the structure added later to the structure list takes precedence. When `priority` is None, "
        "the value is automatically assigned based on `structure_priority_mode` in the `Simulation`.",
    )

    @model_validator(mode="after")
    def _run_validation_checks(self) -> Self:
        """Run topology-specific validation checks after model initialization."""
        self._validate_eps_values()
        self._validate_penalty_value()
        self._validate_gradients()
        return self

    def _validate_eps_values(self) -> None:
        """Validate the epsilon values by evaluating the transformations."""
        try:
            x = self.initial_parameters
            self.eps_values(x)
        except Exception as e:
            raise ValidationError(f"Could not evaluate transformations: {e!s}") from e

    def _validate_penalty_value(self) -> None:
        """Validate the penalty values by evaluating the penalties."""
        try:
            x = self.initial_parameters
            self.penalty_value(x)
        except Exception as e:
            raise ValidationError(f"Could not evaluate penalties: {e!s}") from e

    def _validate_gradients(self) -> None:
        """Validate the gradients of the penalties and transformations."""
        x = self.initial_parameters

        penalty_independent = False
        if self.penalties:
            with warnings.catch_warnings(record=True) as w:
                penalty_grad = grad(self.penalty_value)(x)
                penalty_independent = any("independent" in str(warn.message).lower() for warn in w)
            if np.any(np.isnan(penalty_grad) | np.isinf(penalty_grad)):
                raise ValidationError("Penalty gradients contain 'NaN' or 'Inf' values.")

        eps_independent = False
        if self.transformations:
            with warnings.catch_warnings(record=True) as w:
                eps_grad = elementwise_grad(self.eps_values)(x)
                eps_independent = any("independent" in str(warn.message).lower() for warn in w)
            if np.any(np.isnan(eps_grad) | np.isinf(eps_grad)):
                raise ValidationError("Transformation gradients contain 'NaN' or 'Inf' values.")

        if penalty_independent and eps_independent:
            raise ValidationError(
                "Both penalty and transformation gradients appear to be independent of the input parameters. "
                "This indicates that the optimization will not function correctly. "
                "Please double-check the definitions of both the penalties and transformations."
            )
        if penalty_independent:
            td.log.warning(
                "Penalty gradient seems independent of input, meaning that it "
                "will not contribute to the objective gradient during optimization. "
                "This is likely not correct - double-check the penalties."
            )
        elif eps_independent:
            td.log.warning(
                "Transformation gradient seems independent of input, meaning that it "
                "will not contribute to the objective gradient during optimization. "
                "This is likely not correct - double-check the transformations."
            )

    @staticmethod
    def _check_params(params: anp.ndarray = None) -> None:
        """Ensure ``params`` are between 0 and 1."""
        if params is None:
            return
        if np.any((params < 0) | (params > 1)):
            raise ValueError(
                "Parameters in the 'invdes' plugin's topology optimization feature "
                "are restricted to be between 0 and 1."
            )

    @property
    def params_shape(self) -> tuple[int, int, int]:
        """Shape of the parameters array in (x, y, z), given the ``pixel_size`` and bounds."""
        side_lengths = np.array(self.size)
        num_pixels = np.ceil(side_lengths / self.pixel_size)
        # TODO: if the structure is infinite but the simulation is finite, need reduced bounds
        num_pixels[np.logical_or(np.isinf(num_pixels), self.uniform)] = 1
        return tuple(int(n) for n in num_pixels)

    def _warn_deprecate_params(self) -> None:
        td.log.warning(
            "Parameter initialization via design region methods is deprecated and will be "
            "removed in the future. Please specify this through the design region's "
            "'initialization_spec' instead."
        )

    def params_uniform(self, value: float) -> NDArray[np.floating]:
        """Make an array of parameters with all the same value."""
        self._warn_deprecate_params()
        return value * np.ones(self.params_shape)

    @property
    def params_random(self) -> NDArray[np.floating]:
        """Convenience for generating random parameters between (0,1) with correct shape."""
        self._warn_deprecate_params()
        return np.random.random(self.params_shape)

    @property
    def params_zeros(self) -> NDArray[np.floating]:
        """Convenience for generating random parameters of all 0 values with correct shape."""
        self._warn_deprecate_params()
        return self.params_uniform(0.0)

    @property
    def params_half(self) -> NDArray[np.floating]:
        """Convenience for generating random parameters of all 0.5 values with correct shape."""
        self._warn_deprecate_params()
        return self.params_uniform(0.5)

    @property
    def params_ones(self) -> NDArray[np.floating]:
        """Convenience for generating random parameters of all 1 values with correct shape."""
        self._warn_deprecate_params()
        return self.params_uniform(1.0)

    @property
    def coords(self) -> dict[str, list[float]]:
        """Coordinates for the custom medium corresponding to this design region."""

        lengths = np.array(self.size)

        rmin, rmax = self.geometry.bounds
        params_shape = self.params_shape

        coords = {}
        for dim, ptmin, ptmax, length, num_pts in zip("xyz", rmin, rmax, lengths, params_shape):
            step_size = length / num_pts
            if np.isinf(length):
                coord_vals = [self.center["xyz".index(dim)]]
            else:
                coord_vals = np.linspace(ptmin + step_size / 2, ptmax - step_size / 2, num_pts)
                coord_vals = coord_vals.tolist()
            coords[dim] = coord_vals

        return coords

    def eps_values(
        self, params: anp.ndarray, symmetry: MirrorSymmetry | None = None
    ) -> anp.ndarray:
        """Values for the custom medium permittivity."""

        self._check_params(params)

        material_density = self.material_density(params, symmetry=symmetry)
        eps_min, eps_max = self.eps_bounds
        eps_arr = eps_min + material_density * (eps_max - eps_min)
        return eps_arr.reshape(params.shape)

    def to_structure(
        self, params: anp.ndarray, symmetry: MirrorSymmetry | None = None
    ) -> td.Structure:
        """Convert this ``DesignRegion`` into a custom ``Structure``."""
        self._check_params(params)

        coords = self.coords
        eps_values = self.eps_values(params, symmetry=symmetry)
        eps_data_array = td.SpatialDataArray(eps_values, coords=coords)
        medium = td.CustomMedium(permittivity=eps_data_array)
        return td.Structure(geometry=self.geometry, medium=medium, priority=self.priority)

    @property
    def _override_structure_dl(self) -> float:
        """Override structure step size along all three dimensions."""
        if self.override_structure_dl is None:
            return self.pixel_size
        if self.override_structure_dl is False:
            return None
        return self.override_structure_dl

    @property
    def mesh_override_structure(self) -> td.MeshOverrideStructure:
        """Generate mesh override structure for this ``DesignRegion`` using ``pixel_size`` step."""

        dl = self._override_structure_dl

        if not dl:
            return None

        return td.MeshOverrideStructure(
            geometry=self.geometry,
            dl=(dl, dl, dl),
            enforce=True,
        )

    def evaluate_transformation(
        self,
        transformation: TransformationType,
        params: anp.ndarray,
        symmetry: MirrorSymmetry | None = None,
    ) -> anp.ndarray:
        """Evaluate a transformation, passing in design_region_dl."""
        self._check_params(params)
        kwargs = {"spatial_data": params, "design_region_dl": self.pixel_size}
        if symmetry is not None:
            kwargs["symmetry"] = symmetry
        return transformation.evaluate(**kwargs)

    def evaluate_penalty(
        self,
        penalty: PenaltyType,
        material_density: anp.ndarray,
        symmetry: MirrorSymmetry | None = None,
    ) -> float:
        """Evaluate an erosion-dilation penalty, passing in pixel_size."""
        kwargs = {"x": material_density, "pixel_size": self.pixel_size}
        if symmetry is not None:
            kwargs["symmetry"] = symmetry
        return penalty.evaluate(**kwargs)


DesignRegionType = TopologyDesignRegion
