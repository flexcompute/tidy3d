from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, Protocol

import numpy as np

from tidy3d.components.autograd.derivative_utils import DerivativeInfo
from tidy3d.components.autograd.types import PathType
from tidy3d.components.types import ArrayLike
from tidy3d.exceptions import AdjointError, format_chained_exception_message

if TYPE_CHECKING:
    from tidy3d.components.autograd import AutogradFieldMap
    from tidy3d.components.geometry.utils import GeometryType
    from tidy3d.components.medium import MediumType
    from tidy3d.components.simulation import Simulation
    from tidy3d.components.structure import Structure


@dataclass(frozen=True)
class DerivativeView:
    """Derivative dispatch view for geometry and medium targets.

    Non-None entries override the target structure geometry or medium for one
    predeclared numerical derivative query. The view is resolved and compiled
    before simulation and does not imply a separate forward or adjoint run.
    """

    geometry: GeometryType | None = None
    """Optional geometry override for the query target structure."""

    medium: MediumType | None = None
    """Optional medium override for the query target structure."""


@dataclass(frozen=True)
class NumericalDerivativeQueryContext:
    """Read-only setup context for resolving one numerical derivative query."""

    simulation: Simulation
    structure_index: int

    @property
    def structure(self) -> Structure:
        """Return the inserted numerical structure targeted by the query."""

        return self.simulation.structures[self.structure_index]

    @property
    def permittivity_template(self) -> Any:
        """Return the static epsilon-evaluation grid used by ``updated_epsilon``."""

        frequencies = tuple(self.simulation._freqs_adjoint)
        if not frequencies:
            raise AdjointError(
                "Numerical derivative queries require at least one adjoint frequency."
            )
        simulation_plane = self.simulation if self.simulation.size.count(0.0) == 1 else None
        monitor_box = self.structure._adjoint_monitor_box(
            grid=self.simulation.grid,
            plane=simulation_plane,
        )
        return self.simulation.epsilon(
            box=monitor_box,
            coord_key="centers",
            freq=frequencies[0],
        )


class DerivativeViewFactory(Protocol):
    """Setup-time callable that constructs one concrete derivative view."""

    def __call__(self, context: NumericalDerivativeQueryContext) -> DerivativeView: ...


@dataclass(frozen=True)
class NumericalDerivativeQuery:
    """Predeclared structure-native derivative data required by a numerical callback.

    ``paths`` are structure-local (starting with ``geometry`` or ``medium``) and are evaluated
    on the inserted structure, or on its ``derivative_view`` when one is given. ``summed``
    results are summed over each callback chunk's frequencies; ``resolved`` results keep a
    trailing frequency axis and currently require ``CustomMedium`` paths.
    """

    paths: tuple[PathType, ...]
    derivative_view: DerivativeView | DerivativeViewFactory | None = None
    frequency_mode: Literal["summed", "resolved"] = "summed"

    def __post_init__(self) -> None:
        """Normalize and validate the query's structure-local derivative paths."""

        paths = tuple(tuple(path) for path in self.paths)
        if not paths:
            raise AdjointError("NumericalDerivativeQuery.paths must not be empty.")
        if len(set(paths)) != len(paths):
            raise AdjointError("NumericalDerivativeQuery.paths must not contain duplicates.")
        for path in paths:
            if not path or path[0] not in ("geometry", "medium"):
                raise AdjointError(
                    "NumericalDerivativeQuery paths must start with 'geometry' or 'medium'; "
                    f"received {path!r}."
                )
        if self.frequency_mode not in ("summed", "resolved"):
            raise AdjointError(
                "NumericalDerivativeQuery.frequency_mode must be 'summed' or 'resolved'; "
                f"received {self.frequency_mode!r}."
            )
        object.__setattr__(self, "paths", paths)

    def view_for(self, simulation: Simulation, structure_index: int) -> DerivativeView | None:
        """The concrete view for the inserted structure ``structure_index`` of ``simulation``."""

        return self.resolve_view(NumericalDerivativeQueryContext(simulation, structure_index))

    def resolve_view(self, context: NumericalDerivativeQueryContext) -> DerivativeView | None:
        """Resolve the optional setup-time view factory to a concrete view."""

        view_or_factory = self.derivative_view
        if view_or_factory is None or isinstance(view_or_factory, DerivativeView):
            return view_or_factory
        if not callable(view_or_factory):
            raise AdjointError(
                "NumericalDerivativeQuery.derivative_view must be a DerivativeView, "
                "a setup-time view factory, or None."
            )
        view = view_or_factory(context)
        if not isinstance(view, DerivativeView):
            raise AdjointError(
                "A numerical derivative view factory must return a DerivativeView, "
                f"not {type(view).__name__}."
            )
        return view


NumericalComputeDerivatives = (
    Callable[[ArrayLike, DerivativeInfo], dict[PathType, Any]]
    | Callable[
        [ArrayLike, DerivativeInfo, Mapping[str, dict[PathType, Any]]],
        dict[PathType, Any],
    ]
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
        from tidy3d.web.api.autograd.types import (
            DerivativeView,
            NumericalDerivativeQuery,
            NumericalStructureConfig,
        )

        def create_cylinder(parameters):
            radius, length = parameters
            geometry = td.Cylinder(center=(0, 0, 0), radius=radius, length=length, axis=2)
            return td.Structure(geometry=geometry, medium=td.Medium(permittivity=2.25))

        def cylinder_vjp(parameters, derivative_info, query_results):
            step = 1e-3
            params_np = np.asarray(parameters, dtype=float)
            dJ_deps = query_results["permittivity_sensitivity"][("medium", "permittivity")]
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

                delta_eps = delta_eps.transpose("x", "y", "z", "f")
                vjps[path] = float(np.real(np.sum(delta_eps.data * dJ_deps)))
            return vjps

        numerical_structure = NumericalStructureConfig(
            create=create_cylinder,
            compute_derivatives=cylinder_vjp,
            parameters=np.array([0.5, 0.2]),
            derivative_queries={
                "permittivity_sensitivity": NumericalDerivativeQuery(
                    paths=(("medium", "permittivity"),),
                    derivative_view=lambda context: DerivativeView(
                        medium=td.CustomMedium(
                            permittivity=xr.ones_like(context.permittivity_template)
                        )
                    ),
                    frequency_mode="resolved",
                )
            },
        )
    """

    create: Callable[[ArrayLike], Structure]
    """Function that creates the structure from static ``parameters``."""

    compute_derivatives: NumericalComputeDerivatives
    """Function that computes numerical gradients for ``("numerical", index, param_i)`` paths.
    Signatures:
    - ``compute_derivatives(parameters, derivative_info) -> dict[path, gradient]``
    - ``compute_derivatives(parameters, derivative_info, query_results) -> dict[path, gradient]``
    ``query_results`` contains only the structure-native derivatives declared in
    ``derivative_queries``.
    """

    parameters: ArrayLike
    """1D parameter vector consumed by ``create`` and ``compute_derivatives``."""

    derivative_queries: Mapping[str, NumericalDerivativeQuery] = field(default_factory=dict)
    """Structure-native derivatives the callback needs, declared before the run by key.

    The callback then takes ``query_results`` as its third argument: for each key, the
    query's derivatives by structure-local path, for the callback's frequency chunk.
    """

    _accepts_query_results: bool = field(init=False, default=False, repr=False)

    def __post_init__(self) -> None:
        self._validate_callables()
        self._validate_create_signature()
        self._validate_compute_derivatives_signature()
        self._validate_parameters()
        self._validate_derivative_queries()

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
                "(parameters, derivative_info, query_results), and it currently accepts "
                f"{num_args} arguments. The parameters were the values used for "
                "structure creation, derivative_info contains the chunked field/path context, "
                "and query_results contains the predeclared structure-native derivatives."
            )
        if vjp_arg_names[1] != "derivative_info":
            raise AdjointError(
                "NumericalStructureConfig.compute_derivatives second argument name is "
                f"{vjp_arg_names[1]} but it should be derivative_info."
            )
        if num_args == 3 and vjp_arg_names[2] != "query_results":
            raise AdjointError(
                "NumericalStructureConfig.compute_derivatives third argument name is "
                f"{vjp_arg_names[2]} but it should be query_results."
            )
        self._accepts_query_results = num_args == 3

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

    def _validate_derivative_queries(self) -> None:
        """Validate and snapshot the user-facing query mapping."""

        if not isinstance(self.derivative_queries, Mapping):
            raise AdjointError("NumericalStructureConfig.derivative_queries must be a mapping.")
        queries = dict(self.derivative_queries)
        for key, query in queries.items():
            if not isinstance(key, str) or not key:
                raise AdjointError(
                    "NumericalStructureConfig derivative query keys must be nonempty strings."
                )
            if not isinstance(query, NumericalDerivativeQuery):
                raise AdjointError(
                    "NumericalStructureConfig derivative query values must be "
                    "NumericalDerivativeQuery instances."
                )
        if queries and not self._accepts_query_results:
            raise AdjointError(
                "NumericalStructureConfig.compute_derivatives must accept query_results when "
                "derivative_queries are declared."
            )
        self.derivative_queries = queries


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

    Geometry sampling metadata on that ``DerivativeInfo`` (``adaptive_vjp_spacing()``,
    ``derivative_utils.discretization_wavelength()``, and the clip-context material length scale) is resolved
    per structure from the simulation definition, and only for structures with a traced
    geometry path. A callback on a structure whose traced paths are all under ``medium``
    therefore raises ``AdjointError`` from those accessors; every other derivative input
    (fields, permittivity, bounds) is available on any path.
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

    @property
    def needs_autograd(self) -> bool:
        """Whether this prepared simulation has validated fields needing autograd."""
        return bool(self.sim_fields)
