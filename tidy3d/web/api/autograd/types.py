from __future__ import annotations

import inspect
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, NamedTuple, Protocol

import numpy as np

from tidy3d.components.autograd.derivative_utils import DerivativeInfo
from tidy3d.components.autograd.types import PathType
from tidy3d.components.types import ArrayLike
from tidy3d.exceptions import AdjointError, format_chained_exception_message

if TYPE_CHECKING:
    from tidy3d.components.autograd import AutogradFieldMap
    from tidy3d.components.autograd.path_utils import AutogradRoute
    from tidy3d.components.geometry.utils import GeometryType
    from tidy3d.components.medium import MediumType
    from tidy3d.components.simulation import Simulation
    from tidy3d.components.structure import Structure


class DerivativeHelper(Protocol):
    """Callable used to query structure-native derivatives."""

    def __call__(
        self,
        derivative_info: DerivativeInfo,
        *,
        derivative_view: DerivativeView | None = None,
    ) -> AutogradFieldMap: ...


@dataclass(frozen=True)
class DerivativeView:
    """Derivative dispatch view for geometry and medium targets.

    Non-None entries in this view always override the temporary target structure
    geometry/medium used inside ``derivative_helper``. Derivatives are still
    computed and returned only for the paths requested in ``DerivativeInfo``.
    It does not imply re-running forward/adjoint simulations. The permittivity
    dataset (``DerivativeInfo.eps_data``) and forward/adjoint field data used for
    derivative computations remain from the original structure simulation context.
    """

    geometry: GeometryType | None = None
    """Optional geometry override for the helper target structure."""

    medium: MediumType | None = None
    """Optional medium override for the helper target structure."""


NumericalComputeDerivatives = (
    Callable[[ArrayLike, DerivativeInfo], dict[PathType, Any]]
    | Callable[[ArrayLike, DerivativeInfo, DerivativeHelper], dict[PathType, Any]]
)


@dataclass
class NumericalStructureConfig:
    """Configuration for numerical structure insertion and custom numerical gradients.

    Example
    -------
    .. code-block:: python

        import numpy as np
        import xarray as xr
        import tidy3d as td
        from tidy3d.web.api.autograd.types import DerivativeView, NumericalStructureConfig

        def create_cylinder(parameters):
            radius, length = parameters
            geometry = td.Cylinder(center=(0, 0, 0), radius=radius, length=length, axis=2)
            return td.Structure(geometry=geometry, medium=td.Medium(permittivity=2.25))

        def cylinder_vjp(parameters, derivative_info, derivative_helper):
            step = 1e-3
            params_np = np.asarray(parameters, dtype=float)
            helper_info = derivative_info.updated_copy(
                paths=[("medium", "permittivity")],
                deep=False,
            )
            vjps = {}
            for path in derivative_info.paths:
                param_idx = path[0]
                params_up = params_np.copy()
                params_down = params_np.copy()
                params_up[param_idx] += step
                params_down[param_idx] -= step

                cylinder_up = create_cylinder(params_up)
                cylinder_down = create_cylinder(params_down)
                eps_up = derivative_info.updated_epsilon(cylinder_up.geometry)
                eps_down = derivative_info.updated_epsilon(cylinder_down.geometry)
                delta_eps = (eps_up - eps_down) / (2 * step)

                dJ_deps = derivative_helper(
                    helper_info,
                    derivative_view=DerivativeView(
                        # Use medium-only view so derivative_helper keeps original bounds.
                        medium=td.CustomMedium(
                            permittivity=xr.ones_like(delta_eps.isel(f=0, drop=True))
                        ),
                    ),
                )[("medium", "permittivity")]
                vjps[path] = float(np.real(np.sum(delta_eps.sum("f").data * dJ_deps)))
            return vjps

        numerical_structure = NumericalStructureConfig(
            create=create_cylinder,
            compute_derivatives=cylinder_vjp,
            parameters=np.array([0.5, 0.2]),
        )
    """

    create: Callable[[ArrayLike], Structure]
    """Function that creates the structure from static ``parameters``."""

    compute_derivatives: NumericalComputeDerivatives
    """Function that computes numerical gradients for ``("numerical", index, param_i)`` paths.
    Signatures:
    - ``compute_derivatives(parameters, derivative_info) -> dict[path, gradient]``
    - ``compute_derivatives(parameters, derivative_info, derivative_helper) -> dict[path, gradient]``
    The optional ``derivative_helper`` callback supports
    ``derivative_helper(derivative_info, derivative_view=...)`` to evaluate
    native derivatives using an explicit geometry/medium derivative view.
    """

    parameters: ArrayLike
    """1D parameter vector consumed by ``create`` and ``compute_derivatives``."""
    _uses_derivative_helper: bool = field(init=False, default=False, repr=False)

    def __post_init__(self) -> None:
        self._validate_callables()
        self._validate_create_signature()
        self._validate_compute_derivatives_signature()
        self._validate_parameters()

    def _validate_callables(self) -> None:
        if not callable(self.create):
            raise AdjointError("NumericalStructureConfig.create must be callable.")
        if not callable(self.compute_derivatives):
            raise AdjointError("NumericalStructureConfig.compute_derivatives must be callable.")

    def _validate_create_signature(self) -> None:
        create_sig = inspect.signature(self.create)
        create_arg_names = list(create_sig.parameters.keys())
        if len(create_arg_names) != 1:
            raise AdjointError(
                "NumericalStructureConfig.create should accept one argument "
                "(the parameters vector used for structure creation), and it currently "
                f"accepts {len(create_arg_names)} arguments."
            )

    def _validate_compute_derivatives_signature(self) -> None:
        vjp_sig = inspect.signature(self.compute_derivatives)
        vjp_arg_names = list(vjp_sig.parameters.keys())
        num_args = len(vjp_arg_names)
        if num_args not in (2, 3):
            raise AdjointError(
                "NumericalStructureConfig.compute_derivatives should accept either two "
                "arguments (parameters, derivative_info) or three arguments "
                "(parameters, derivative_info, derivative_helper), and it currently accepts "
                f"{num_args} arguments. The parameters were the values used for "
                "structure creation, derivative_info contains the chunked field/path context, "
                "and derivative_helper (when accepted) allows querying structure-native "
                "derivatives for a provided DerivativeInfo."
            )
        if vjp_arg_names[1] != "derivative_info":
            raise AdjointError(
                "NumericalStructureConfig.compute_derivatives second argument name is "
                f"{vjp_arg_names[1]} but it should be derivative_info."
            )
        if num_args == 3 and vjp_arg_names[2] != "derivative_helper":
            raise AdjointError(
                "NumericalStructureConfig.compute_derivatives third argument name is "
                f"{vjp_arg_names[2]} but it should be derivative_helper."
            )
        self._uses_derivative_helper = num_args == 3

    def _validate_parameters(self) -> None:
        try:
            array_params = np.asarray(self.parameters)
        except Exception as exc:
            raise AdjointError(
                format_chained_exception_message(
                    "NumericalStructureConfig.parameters must be array-like (e.g., list, "
                    "tuple, numpy array, or compatible autograd array-like)",
                    exc,
                )
            ) from exc

        if array_params.ndim != 1:
            raise AdjointError("Parameters for each numerical structure must be 1D array-like.")


@dataclass
class CustomVJPConfig:
    """Configuration for overriding gradients on existing traced structure paths.

    Example
    -------
    .. code-block:: python

        import tidy3d as td
        from tidy3d.web.api.autograd.types import CustomVJPConfig

        def polyslab_vjp(polyslab, derivative_info):
            # Return gradient values keyed by derivative paths in derivative_info.paths.
            return {path: 0.0 for path in derivative_info.paths}

        custom_vjp = CustomVJPConfig(
            structure=1,
            compute_derivatives=polyslab_vjp,
            path_key=("geometry", "vertices"),
        )
    """

    structure: int | type[GeometryType] | type[MediumType]
    """Target existing traced structure(s) in ``("structures", ...)`` namespace.
    Can be an index or a geometry/medium type (expanded to matching indices).
    """

    compute_derivatives: Callable[[GeometryType | MediumType, DerivativeInfo], dict[PathType, Any]]
    """Function for computing the targeted vjp value. The function should accept the geometry or medium in the
    structure depending on if this is a geometry or medium path (see path_key) as the first argument. The second
    argument should accept a DerivativeInfo object that contains important for computing the gradient. The function
    should return a dict object that maps the path to the computed gradient value.
    """

    path_key: tuple[str, ...] | None = None
    """Path key corresponding to the vjp. For example, this could be ('geometry', 'radius') if you are targeting
    the radius parameter in the given structure geometry. It can also target the medium by specifying medium first
    (i.e. - ('medium', 'permittivity') will target the permittivity variable in the structure's medium). If not
    specified or set to None, the supplied function applies for all possible vjp paths.
    """

    def __post_init__(self) -> None:
        self._validate_callable()
        self._validate_compute_derivatives_signature()

    def _validate_callable(self) -> None:
        if not callable(self.compute_derivatives):
            raise AdjointError("CustomVJPConfig.compute_derivatives must be callable.")

    def _validate_compute_derivatives_signature(self) -> None:
        vjp_sig = inspect.signature(self.compute_derivatives)
        vjp_arg_names = list(vjp_sig.parameters.keys())
        if len(vjp_arg_names) != 2:
            raise AdjointError(
                "CustomVJPConfig compute_derivatives function should accept two arguments "
                "(target, derivative_info), and it currently accepts "
                f"{len(vjp_arg_names)} arguments. The target is the geometry or medium "
                "instance selected by path_key, and derivative_info contains the field "
                "data and path metadata needed to compute the VJP."
            )
        if vjp_arg_names[1] != "derivative_info":
            raise AdjointError(
                "CustomVJPConfig compute_derivatives function second argument name is "
                f"{vjp_arg_names[1]} but it should be derivative_info."
            )


CustomVJPSpec = (
    CustomVJPConfig
    | dict[str, CustomVJPConfig]
    | Sequence[CustomVJPConfig]
    | dict[str, Sequence[CustomVJPConfig]]
    | Sequence[Sequence[CustomVJPConfig]]
)


class SetupRunResult(NamedTuple):
    sim_fields: AutogradFieldMap
    simulation: Simulation
    numerical_structure_map: dict[int, NumericalStructureConfig]
    autograd_routes: tuple[AutogradRoute, ...] = ()

    @property
    def needs_autograd(self) -> bool:
        """Whether this prepared simulation has validated fields needing autograd."""
        return bool(self.sim_fields)
