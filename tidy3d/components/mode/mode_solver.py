"""Solve for modes in a 2D cross-sectional plane in a simulation, assuming translational
invariance along a given propagation axis.
"""

from __future__ import annotations

from functools import wraps
from math import isclose
from typing import Dict, List, Tuple, Union

import numpy as np
import pydantic.v1 as pydantic
import xarray as xr
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

from ...constants import C_0
from ...exceptions import SetupError, ValidationError
from ...log import log
from ..base import Tidy3dBaseModel, cached_property, skip_if_fields_missing
from ..boundary import PML, Absorber, Boundary, BoundarySpec, PECBoundary, StablePML
from ..data.data_array import (
    FreqModeDataArray,
    ModeIndexDataArray,
    ScalarModeFieldCylindricalDataArray,
    ScalarModeFieldDataArray,
)
from ..data.monitor_data import ModeSolverData
from ..data.sim_data import SimulationData
from ..eme.data.sim_data import EMESimulationData
from ..eme.simulation import EMESimulation
from ..geometry.base import Box
from ..grid.grid import Coords, Grid
from ..medium import FullyAnisotropicMedium, LossyMetalMedium
from ..mode_spec import ModeSpec
from ..monitor import ModeMonitor, ModeSolverMonitor
from ..scene import Scene
from ..simulation import Simulation
from ..source.field import ModeSource
from ..source.time import SourceTime
from ..structure import Structure
from ..subpixel_spec import SurfaceImpedance
from ..types import (
    TYPE_TAG_STR,
    ArrayComplex3D,
    ArrayComplex4D,
    ArrayFloat1D,
    ArrayFloat2D,
    Ax,
    Axis,
    Axis2D,
    Direction,
    EMField,
    EpsSpecType,
    FreqArray,
    Literal,
    PlotScale,
    Symmetry,
)
from ..validators import (
    validate_freqs_min,
    validate_freqs_not_empty,
    validate_mode_plane_radius,
)
from ..viz import make_ax, plot_params_pml

# Importing the local solver may not work if e.g. scipy is not installed
IMPORT_ERROR_MSG = """Could not import local solver, 'ModeSolver' objects can still be constructed
but will have to be run through the server.
"""
try:
    from .solver import compute_modes

    LOCAL_SOLVER_IMPORTED = True
except ImportError:
    log.warning(IMPORT_ERROR_MSG)
    LOCAL_SOLVER_IMPORTED = False

FIELD = Tuple[ArrayComplex3D, ArrayComplex3D, ArrayComplex3D]
MODE_MONITOR_NAME = "<<<MODE_SOLVER_MONITOR>>>"

# Warning for field intensity at edges over total field intensity larger than this value
FIELD_DECAY_CUTOFF = 1e-2

# Maximum allowed size of the field data produced by the mode solver
MAX_MODES_DATA_SIZE_GB = 20

MODE_SIMULATION_TYPE = Union[Simulation, EMESimulation]
MODE_SIMULATION_DATA_TYPE = Union[SimulationData, EMESimulationData]
MODE_PLANE_TYPE = Union[Box, ModeSource, ModeMonitor, ModeSolverMonitor]

# When using ``angle_rotation`` without a bend, use a very large effective radius
EFFECTIVE_RADIUS_FACTOR = 10_000

# Log a warning when the PML covers more than this portion of the mode plane in any axis
WARN_THICK_PML_PERCENT = 50


def require_fdtd_simulation(fn):
    """Decorate a function to check that ``simulation`` is an FDTD ``Simulation``."""

    @wraps(fn)
    def _fn(self, **kwargs):
        """New decorated function."""
        if not isinstance(self.simulation, Simulation):
            raise SetupError(
                f"The function '{fn.__name__}' is only supported "
                "for 'simulation' of type FDTD 'Simulation'."
            )
        return fn(self, **kwargs)

    return _fn


class ModeSolver(Tidy3dBaseModel):
    """
    Interface for solving electromagnetic eigenmodes in a 2D plane with translational
    invariance in the third dimension.

    See Also
    --------

    :class:`ModeSource`:
        Injects current source to excite modal profile on finite extent plane.

    **Notebooks:**
        * `Waveguide Y junction <../../notebooks/YJunction.html>`_
        * `Photonic crystal waveguide polarization filter <../../../notebooks/PhotonicCrystalWaveguidePolarizationFilter.html>`_

    **Lectures:**
        * `Prelude to Integrated Photonics Simulation: Mode Injection <https://www.flexcompute.com/fdtd101/Lecture-4-Prelude-to-Integrated-Photonics-Simulation-Mode-Injection/>`_
    """

    simulation: MODE_SIMULATION_TYPE = pydantic.Field(
        ...,
        title="Simulation",
        description="Simulation or EMESimulation defining all structures and mediums.",
        discriminator="type",
    )

    plane: MODE_PLANE_TYPE = pydantic.Field(
        ...,
        title="Plane",
        description="Cross-sectional plane in which the mode will be computed.",
        discriminator=TYPE_TAG_STR,
    )

    mode_spec: ModeSpec = pydantic.Field(
        ...,
        title="Mode specification",
        description="Container with specifications about the modes to be solved for.",
    )

    freqs: FreqArray = pydantic.Field(
        ..., title="Frequencies", description="A list of frequencies at which to solve."
    )

    direction: Direction = pydantic.Field(
        "+",
        title="Propagation direction",
        description="Direction of waveguide mode propagation along the axis defined by its normal "
        "dimension.",
    )

    colocate: bool = pydantic.Field(
        True,
        title="Colocate fields",
        description="Toggle whether fields should be colocated to grid cell boundaries (i.e. "
        "primal grid nodes). Default is ``True``.",
    )

    fields: Tuple[EMField, ...] = pydantic.Field(
        ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"],
        title="Field Components",
        description="Collection of field components to store in the monitor. Note that some "
        "methods like ``flux``, ``dot`` require all tangential field components, while others "
        "like ``mode_area`` require all E-field components.",
    )

    @pydantic.validator("simulation", pre=True, always=True)
    def _convert_to_simulation(cls, val):
        """Convert to regular Simulation if e.g. JaxSimulation given."""
        if hasattr(val, "to_simulation"):
            val = val.to_simulation()[0]
            log.warning(
                "'JaxSimulation' is no longer directly supported in 'ModeSolver', "
                "converting to static simulation."
            )
        return val

    @pydantic.validator("plane", always=True)
    def is_plane(cls, val):
        """Raise validation error if not planar."""
        if val.size.count(0.0) != 1:
            raise ValidationError(f"ModeSolver plane must be planar, given size={val}")
        return val

    _freqs_not_empty = validate_freqs_not_empty()
    _freqs_lower_bound = validate_freqs_min()

    @pydantic.validator("plane", always=True)
    @skip_if_fields_missing(["simulation"])
    def plane_in_sim_bounds(cls, val, values):
        """Check that the plane is at least partially inside the simulation bounds."""
        sim_center = values.get("simulation").center
        sim_size = values.get("simulation").size
        sim_box = Box(size=sim_size, center=sim_center)

        if not sim_box.intersects(val):
            raise SetupError("'ModeSolver.plane' must intersect 'ModeSolver.simulation'.")
        return val

    def _post_init_validators(self) -> None:
        validate_mode_plane_radius(
            mode_spec=self.mode_spec, plane=self.plane, msg_prefix="Mode solver"
        )
        self._warn_thick_pml(simulation=self.simulation, plane=self.plane, mode_spec=self.mode_spec)

    @classmethod
    def _warn_thick_pml(
        cls, simulation: Simulation, plane: Box, mode_spec: ModeSpec, warn_str: str = "'ModeSolver'"
    ):
        """Warn if the pml covers a significant portion of the mode plane."""
        coord_0, coord_1 = cls._plane_grid(
            simulation=simulation,
            plane=plane,
        )
        num_cells = [len(coord_0), len(coord_1)]
        effective_num_pml = cls._effective_num_pml(
            simulation=simulation, plane=plane, mode_spec=mode_spec
        )
        for i in (0, 1):
            if 2 * effective_num_pml[i] > (WARN_THICK_PML_PERCENT / 100) * num_cells[i]:
                log.warning(
                    f"{warn_str}:  "
                    f"The mode solver pml in tangential axis '{i}' "
                    f"covers more than '{WARN_THICK_PML_PERCENT}%' of the "
                    "mode plane cells. Consider using a larger mode plane "
                    "or smaller 'num_pml'."
                )

    @cached_property
    def normal_axis(self) -> Axis:
        """Axis normal to the mode plane."""
        return self.plane.size.index(0.0)

    @staticmethod
    def plane_center_tangential(plane) -> tuple[float, float]:
        """Mode lane center in the tangential axes."""
        _, plane_center = plane.pop_axis(plane.center, plane.size.index(0.0))
        return plane_center

    @cached_property
    def normal_axis_2d(self) -> Axis2D:
        """Axis normal to the mode plane in a 2D plane that is normal to the bend_axis_3d."""
        _, idx_plane = self.plane.pop_axis((0, 1, 2), axis=self.bend_axis_3d)

        return idx_plane.index(self.normal_axis)

    @staticmethod
    def _solver_symmetry(simulation: Simulation, plane: Box) -> Tuple[Symmetry, Symmetry]:
        """Get symmetry for solver for propagation along self.normal axis."""
        normal_axis = plane.size.index(0.0)
        mode_symmetry = list(simulation.symmetry)
        for dim in range(3):
            if simulation.center[dim] != plane.center[dim]:
                mode_symmetry[dim] = 0
        _, solver_sym = plane.pop_axis(mode_symmetry, axis=normal_axis)
        return solver_sym

    @cached_property
    def solver_symmetry(self) -> Tuple[Symmetry, Symmetry]:
        """Get symmetry for solver for propagation along self.normal axis."""
        return self._solver_symmetry(simulation=self.simulation, plane=self.plane)

    @classmethod
    def _get_solver_grid(
        cls,
        simulation: Simulation,
        plane: Box,
        keep_additional_layers: bool = False,
        truncate_symmetry: bool = True,
    ) -> Grid:
        """Grid for the mode solver, not snapped to plane or simulation zero dims, and optionally
        corrected for symmetries.

        Parameters
        ----------
        keep_additional_layers : bool = False
            Do not discard layers of cells in front and behind the main layer of cells. Together they
            represent the region where custom medium data is needed for proper subpixel.
        truncate_symmetry : bool = True
            Truncate to symmetry quadrant if symmetry present.

        Returns
        -------
        :class:`.Grid`
            The resulting grid.
        """

        span_inds = simulation._discretize_inds_monitor(plane, colocate=False)
        normal_axis = plane.size.index(0.0)
        solver_symmetry = cls._solver_symmetry(simulation=simulation, plane=plane)

        # Remove extension along monitor normal
        if not keep_additional_layers:
            span_inds[normal_axis, 0] += 1
            span_inds[normal_axis, 1] -= 1

        # Do not extend if simulation has a single pixel along a dimension
        for dim, num_cells in enumerate(simulation.grid.num_cells):
            if num_cells <= 1:
                span_inds[dim] = [0, 1]

        # Truncate to symmetry quadrant if symmetry present
        if truncate_symmetry:
            _, plane_inds = Box.pop_axis([0, 1, 2], normal_axis)
            for dim, sym in enumerate(solver_symmetry):
                if sym != 0:
                    span_inds[plane_inds[dim], 0] += np.diff(span_inds[plane_inds[dim]]) // 2

        return simulation._subgrid(span_inds=span_inds)

    @cached_property
    def _solver_grid(self) -> Grid:
        """Grid for the mode solver, not snapped to plane or simulation zero dims, and also with
        a small correction for symmetries. We don't do the snapping yet because 0-sized cells are
        currently confusing to the subpixel averaging. The final data coordinates along the
        plane normal dimension and dimensions where the simulation domain is 2D will be correctly
        set after the solve."""

        return self._get_solver_grid(
            simulation=self.simulation,
            plane=self.plane,
            keep_additional_layers=False,
            truncate_symmetry=True,
        )

    @cached_property
    def _num_cells_freqs_modes(self) -> Tuple[int, int, int]:
        """Get the number of spatial points, number of freqs, and number of modes requested."""
        num_cells = np.prod(self._solver_grid.num_cells)
        num_modes = self.mode_spec.num_modes
        num_freqs = len(self.freqs)
        return num_cells, num_freqs, num_modes

    def solve(self) -> ModeSolverData:
        """:class:`.ModeSolverData` containing the field and effective index data.

        Returns
        -------
        ModeSolverData
            :class:`.ModeSolverData` object containing the effective index and mode fields.
        """
        log.warning(
            "Use the remote mode solver with subpixel averaging for better accuracy through "
            "'tidy3d.web.run(...)' or the deprecated 'tidy3d.plugins.mode.web.run(...)'.",
            log_once=True,
        )
        return self.data

    def _freqs_for_group_index(self) -> FreqArray:
        """Get frequencies used to compute group index."""
        f_step = self.mode_spec.group_index_step
        fractional_steps = (1 - f_step, 1, 1 + f_step)
        return np.outer(self.freqs, fractional_steps).flatten()

    def _remove_freqs_for_group_index(self) -> FreqArray:
        """Remove frequencies used to compute group index.

        Returns
        -------
        FreqArray
            Filtered frequency array with only original values.
        """
        return np.array(self.freqs[1 : len(self.freqs) : 3])

    def _get_data_with_group_index(self) -> ModeSolverData:
        """:class:`.ModeSolverData` with fields, effective and group indices on unexpanded grid.

        Returns
        -------
        ModeSolverData
            :class:`.ModeSolverData` object containing the effective and group indices, and mode
            fields.
        """

        # create a copy with the required frequencies for numerical differentiation
        mode_spec = self.mode_spec.copy(update={"group_index_step": False})
        mode_solver = self.copy(
            update={"freqs": self._freqs_for_group_index(), "mode_spec": mode_spec}
        )

        return mode_solver.data_raw._group_index_post_process(self.mode_spec.group_index_step)

    @cached_property
    def grid_snapped(self) -> Grid:
        """The solver grid snapped to the plane normal and to simulation 0-sized dims if any."""
        return self._grid_snapped(simulation=self.simulation, plane=self.plane)

    @classmethod
    def _grid_snapped(cls, simulation: Simulation, plane: Box) -> Grid:
        """The solver grid snapped to the plane normal and to simulation 0-sized dims if any."""
        solver_grid = cls._get_solver_grid(
            simulation=simulation,
            plane=plane,
            keep_additional_layers=False,
            truncate_symmetry=True,
        )
        # snap to plane center along normal direction
        grid_snapped = solver_grid.snap_to_box_zero_dim(plane)
        # snap to simulation center if simulation is 0D along a tangential dimension
        normal_axis = plane.size.index(0.0)
        return simulation._snap_zero_dim(grid_snapped, skip_axis=normal_axis)

    @cached_property
    def data_raw(self) -> ModeSolverData:
        """:class:`.ModeSolverData` containing the field and effective index on unexpanded grid.

        Returns
        -------
        ModeSolverData
            :class:`.ModeSolverData` object containing the effective index and mode fields.
        """

        if self.mode_spec.group_index_step > 0:
            return self._get_data_with_group_index()

        if self.mode_spec.angle_rotation and np.abs(self.mode_spec.angle_theta) > 0:
            return self.rotated_mode_solver_data

        # Compute data on the Yee grid
        mode_solver_data = self._data_on_yee_grid()

        # Colocate to grid boundaries if requested
        if self.colocate:
            mode_solver_data = self._colocate_data(mode_solver_data=mode_solver_data)

        # normalize modes
        self._normalize_modes(mode_solver_data=mode_solver_data)

        # filter polarization if requested
        if self.mode_spec.filter_pol is not None:
            self._filter_polarization(mode_solver_data=mode_solver_data)

        # sort modes if requested
        if self.mode_spec.track_freq and len(self.freqs) > 1:
            mode_solver_data = mode_solver_data.overlap_sort(self.mode_spec.track_freq)

        self._field_decay_warning(mode_solver_data.symmetry_expanded)

        mode_solver_data = self._filter_components(mode_solver_data)
        return mode_solver_data

    @cached_property
    def bend_axis_3d(self) -> Axis:
        """Converts the 2D bend axis into its corresponding 3D axis for a bend structure.
        For a straight waveguide, the rotated axis is equivalent to the bend axis
        and can be determined using angle_phi."""
        _, idx_plane = self.plane.pop_axis((0, 1, 2), axis=self.normal_axis)

        if self.mode_spec.bend_axis is not None:
            return idx_plane[self.mode_spec.bend_axis]

        rotation_axis_index = int(abs(np.cos(self.mode_spec.angle_phi)))
        return idx_plane[rotation_axis_index]

    @cached_property
    def rotated_mode_solver_data(self) -> ModeSolverData:
        # Create a mode solver with rotated geometries for a reference solution with 0-degree angle
        solver_ref = self.rotated_structures_copy
        # Need to store all fields for the rotation
        solver_ref = solver_ref.updated_copy(
            fields=["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"], validate=False
        )
        solver_ref_data = solver_ref.data_raw

        # The reference data should always be colocated
        n_complex = solver_ref_data.n_complex
        if not solver_ref.colocate:
            solver_ref_data = solver_ref._colocate_data(mode_solver_data=solver_ref_data)

        # Transform the colocated mode solution from Cartesian to cylindrical coordinates
        # if a bend structure is simulated.
        solver_ref_data_cylindrical = self._car_2_cyn(mode_solver_data=solver_ref_data)
        # # if self.mode_spec.bend_radius is None, use this instead
        #     solver_ref_data_straight = self._ref_data_straight(mode_solver_data=solver_ref_data)

        try:
            solver = self.reduced_simulation_copy
        except Exception as e:
            solver = self
            log.warning(
                "Mode solver reduced_simulation_copy failed. "
                "Falling back to non-reduced simulation, which may be slower. "
                f"Exception: {str(e)}"
            )

        # Compute the mode solution by rotating the reference data to the monitor plane
        rotated_mode_fields = self._mode_rotation(
            solver_ref_data_cylindrical=solver_ref_data_cylindrical,
            solver=solver,
        )
        # # if self.mode_spec.bend_radius is None, use this instead
        #     rotated_mode_fields = self._mode_rotation_straight(
        #         solver_ref_data=solver_ref_data_straight,
        #         solver=solver,
        #     )

        # TODO: At a later time, we should ensure that `eps_spec` is automatically returned to
        # to compute the backward propagation mode solution using a mode solver
        # with direction "-".
        eps_spec = []
        for _ in self.freqs:
            eps_spec.append("tensorial_complex")
        # finite grid corrections
        grid_factors = solver._grid_correction(
            simulation=solver.simulation,
            plane=solver.plane,
            mode_spec=solver.mode_spec,
            n_complex=n_complex,
            direction=solver.direction,
        )

        # Make mode solver data on the Yee grid
        mode_solver_monitor = solver.to_mode_solver_monitor(name=MODE_MONITOR_NAME)
        grid_expanded = solver.simulation.discretize_monitor(mode_solver_monitor)
        rotated_mode_data = ModeSolverData(
            monitor=mode_solver_monitor,
            symmetry=solver.simulation.symmetry,
            symmetry_center=solver.simulation.center,
            grid_expanded=grid_expanded,
            grid_primal_correction=grid_factors[0],
            grid_dual_correction=grid_factors[1],
            eps_spec=eps_spec,
            **rotated_mode_fields,
        )

        self._normalize_modes(mode_solver_data=rotated_mode_data)
        rotated_mode_data = self._filter_components(rotated_mode_data)

        return rotated_mode_data

    @cached_property
    def rotated_structures_copy(self):
        """Create a copy of the original ModeSolver with rotated structures
        to the simulation and updates the ModeSpec to disable bend correction
        and reset angles to normal."""

        rotated_structures = self._rotate_structures()
        rotated_simulation = self.simulation.updated_copy(structures=rotated_structures)
        rotated_mode_spec = self.mode_spec.updated_copy(
            angle_rotation=False, angle_theta=0, angle_phi=0
        )

        return self.updated_copy(simulation=rotated_simulation, mode_spec=rotated_mode_spec)

    def _rotate_structures(self) -> List[Structure]:
        """Rotate the structures intersecting with modal plane by angle theta
        if bend_correction is enabeled for bend simulations."""

        _, (idx_u, idx_v) = self.plane.pop_axis((0, 1, 2), axis=self.bend_axis_3d)

        mnt_center = self.plane.center
        angle_theta = self.mode_spec.angle_theta
        angle_phi = self.mode_spec.angle_phi

        theta_map = {
            (0, 2): -angle_theta * np.cos(angle_phi),
            (0, 1): angle_theta * np.sin(angle_phi),
            (1, 2): angle_theta * np.cos(angle_phi),
            (1, 0): -angle_theta * np.sin(angle_phi),
            (2, 1): -angle_theta * np.cos(angle_phi),
            (2, 0): angle_theta * np.sin(angle_phi),
        }
        theta = theta_map.get((self.normal_axis, self.bend_axis_3d), 0)

        # Get the translation values
        translate_coords = [0, 0, 0]
        translate_coords[idx_u] = mnt_center[idx_u]
        translate_coords[idx_v] = mnt_center[idx_v]

        reduced_sim_solver = self.reduced_simulation_copy
        rotated_structures = []
        for structure in Scene.intersecting_structures(
            self.plane, reduced_sim_solver.simulation.structures
        ):
            # Rotate and apply translations
            geometry = structure.geometry
            geometry = (
                geometry.translated(
                    x=-translate_coords[0], y=-translate_coords[1], z=-translate_coords[2]
                )
                .rotated(theta, axis=self.bend_axis_3d)
                .translated(x=translate_coords[0], y=translate_coords[1], z=translate_coords[2])
            )

            rotated_structures.append(structure.updated_copy(geometry=geometry))

        return rotated_structures

    @cached_property
    def rotated_bend_center(self) -> List:
        """Calculate the center at the rotated bend such that the modal plane is normal
        to the azimuthal direction of the bend."""
        rotated_bend_center = list(self.plane.center)
        idx_rotate = 3 - self.normal_axis - self.bend_axis_3d
        rotated_bend_center[idx_rotate] -= self._bend_radius

        return rotated_bend_center

    # # Leaving for future reference if needed
    # def _ref_data_straight(
    #     self, mode_solver_data: ModeSolverData
    # ) -> Dict[Union[ScalarModeFieldDataArray, ModeIndexDataArray]]:
    #     """Convert reference data to be centered at the monitor center."""

    #     # Reference solution stored
    #     lateral_axis = 3 - self.normal_axis - self.bend_axis_3d
    #     axes = ("x", "y", "z")
    #     normal_dim = axes[self.normal_axis]
    #     lateral_dim = axes[lateral_axis]

    #     solver_data_straight = {}
    #     for name, field in mode_solver_data.field_components.items():
    #         solver_data_straight[name] = field.copy()
    #         for dim, dim_name in enumerate("xyz"):
    #             # Only shift coordinates for normal_dim and lateral_dim
    #             if dim_name in (normal_dim, lateral_dim):
    #                 coords_shift = field.coords[dim_name] - self.plane.center[dim]
    #                 solver_data_straight[name].coords[dim_name] = coords_shift

    #     solver_data_straight["n_complex"] = mode_solver_data.n_complex

    #     return solver_data_straight

    def _car_2_cyn(
        self, mode_solver_data: ModeSolverData
    ) -> Dict[Union[ScalarModeFieldCylindricalDataArray, ModeIndexDataArray]]:
        """Convert cartesian fields to cylindrical fields centered at the
        rotated bend center."""

        # Extract coordinates from one of the six field components as they are colocated
        pts = [mode_solver_data.Ex[name].values.copy() for name in ["x", "y", "z"]]
        f, mode_index = mode_solver_data.Ex.f, mode_solver_data.Ex.mode_index

        lateral_axis = 3 - self.normal_axis - self.bend_axis_3d

        idx_w, idx_uv = self.plane.pop_axis((0, 1, 2), axis=self.bend_axis_3d)
        idx_u, idx_v = idx_uv

        pts[lateral_axis] -= self.rotated_bend_center[lateral_axis]
        rho = np.sort(np.abs(pts[lateral_axis]))

        theta = np.atleast_1d(
            np.arctan2(
                self.plane.center[idx_v] - self.rotated_bend_center[idx_v],
                self.plane.center[idx_u] - self.rotated_bend_center[idx_u],
            )
        )
        axial = pts[idx_w]

        # Initialize output data arrays for cylindrical fields
        field_data_cylindrical = {
            field_name_cylindrical: np.zeros(
                (len(rho), len(theta), len(axial), len(f), len(mode_index)), dtype=complex
            )
            for field_name_cylindrical in ("Er", "Etheta", "Eaxial", "Hr", "Htheta", "Haxial")
        }

        cos_theta, sin_theta = np.cos(theta), np.sin(theta)

        axes = ("x", "y", "z")
        axial_name, plane_names = self.plane.pop_axis(axes, axis=self.bend_axis_3d)
        cmp_1, cmp_2 = plane_names
        plane_name_normal = axes[self.normal_axis]
        plane_name_lateral = axes[lateral_axis]

        # Determine which coordinate transformation to use based on the lateral axis
        if cmp_1 == plane_name_lateral:
            lateral_coord_value = rho * cos_theta + self.rotated_bend_center[idx_u]
        elif cmp_2 == plane_name_lateral:
            lateral_coord_value = rho * sin_theta + self.rotated_bend_center[idx_v]

        # Transform fields from cartesian to cylindrical
        fields_interp = {
            field_name: getattr(mode_solver_data, field_name)
            .sel({plane_name_normal: self.plane.center[self.normal_axis]}, method="nearest")
            .interp(
                {plane_name_lateral: lateral_coord_value},
                method="linear",
                kwargs={"fill_value": "extrapolate"},
            )
            .transpose(plane_name_lateral, ...)
            .values
            for field_name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
        }

        for field_type in ["E", "H"]:
            field_data_cylindrical[field_type + "r"][:, 0] = (
                fields_interp[field_type + cmp_1] * cos_theta
                + fields_interp[field_type + cmp_2] * sin_theta
            )
            field_data_cylindrical[field_type + "theta"][:, 0] = (
                -fields_interp[field_type + cmp_1] * sin_theta
                + fields_interp[field_type + cmp_2] * cos_theta
            )
            field_data_cylindrical[field_type + "axial"][:, 0] = fields_interp[
                field_type + axial_name
            ]

        coords = {
            "rho": rho,
            "theta": theta,
            "axial": axial,
            "f": f,
            "mode_index": mode_index,
        }

        solver_data_cylindrical = {
            name: ScalarModeFieldCylindricalDataArray(field_data_cylindrical[name], coords=coords)
            for name in ("Er", "Etheta", "Eaxial", "Hr", "Htheta", "Haxial")
        }
        solver_data_cylindrical["n_complex"] = mode_solver_data.n_complex

        return solver_data_cylindrical

    # # Leaving for future reference if needed
    # def _mode_rotation_straight(
    #     self,
    #     solver_ref_data: Dict[Union[ModeSolverData]],
    #     solver: ModeSolver,
    # ) -> ModeSolverData:
    #     """Rotate the mode solver solution from the reference plane
    #     to the desired monitor plane."""

    #     rotated_data_arrays = {}
    #     for field_name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
    #         if self.colocate is True:
    #             # Get colocation coordinates in the solver plane
    #             normal_dim, _ = self.plane.pop_axis("xyz", self.normal_axis)
    #             colocate_coords = self._get_colocation_coordinates()
    #             colocate_coords[normal_dim] = np.atleast_1d(self.plane.center[self.normal_axis])
    #             x = colocate_coords["x"]
    #             y = colocate_coords["y"]
    #             z = colocate_coords["z"]
    #             xyz_coords = [x.copy(), y.copy(), z.copy()]
    #         else:
    #             # Extract coordinate values from the corresponding field component
    #             xyz_coords = solver.grid_snapped[field_name].to_list
    #             x, y, z = (coord.copy() for coord in xyz_coords)

    #         lateral_axis = 3 - self.normal_axis - self.bend_axis_3d
    #         axes = ("x", "y", "z")
    #         normal_dim = axes[self.normal_axis]
    #         lateral_dim = axes[lateral_axis]
    #         axial_dim = axes[self.bend_axis_3d]

    #         pts = [coord.copy() for coord in xyz_coords]
    #         pts[self.normal_axis] -= self.plane.center[self.normal_axis]
    #         pts[lateral_axis] -= self.plane.center[lateral_axis]
    #         axial = pts[self.bend_axis_3d]

    #         f = np.atleast_1d(self.freqs)
    #         mode_index = np.arange(self.mode_spec.num_modes)

    #         k0 = 2 * np.pi * f / C_0
    #         n_eff = solver_ref_data["n_complex"].values
    #         beta = k0[:, None] * n_eff
    #         beta = beta.reshape(1, 1, 1, len(f), len(mode_index))

    #         # Interpolation coords for amplitude and phase of local fields
    #         amp_interp_coors = pts[lateral_axis] * np.cos(self.mode_spec.angle_theta)
    #         phase_interp_coors = pts[lateral_axis] * np.sin(self.mode_spec.angle_theta)

    #         # Initialize output arrays
    #         shape = (len(x), len(y), len(z), len(f), len(mode_index))
    #         phase_fields = np.zeros(shape, dtype=complex)
    #         rotated_field_cmp = np.zeros(shape, dtype=complex)

    #         phase_shape = [1] * 5
    #         phase_shape[lateral_axis] = shape[lateral_axis]
    #         phase_interp_coors = phase_interp_coors.reshape(phase_shape)

    #         sign = 1 if self.normal_axis_2d == 0 else 1
    #         if self.direction == "+":
    #             phase_fields[...] = np.exp(sign * 1j * phase_interp_coors * beta)
    #         else:
    #             phase_fields[...] = np.exp(sign * -1j * phase_interp_coors * beta)

    #         # Interpolate field components
    #         local_fields = {}
    #         for field in ["E", "H"]:
    #             for comp in ["x", "y", "z"]:
    #                 data = solver_ref_data[f"{field}{comp}"].interp(
    #                     {axial_dim: axial, lateral_dim: amp_interp_coors},
    #                     method="linear",
    #                     kwargs={"fill_value": "extrapolate"},
    #                 )

    #                 local_fields[f"{field}{comp}"] = phase_fields * data.values

    #         if field_name == f"E{lateral_dim}":
    #             rotated_field_cmp = local_fields[f"E{lateral_dim}"] * np.cos(
    #                 self.mode_spec.angle_theta
    #             )
    #             +local_fields[f"E{normal_dim}"] * np.sin(self.mode_spec.angle_theta)
    #         elif field_name == f"E{normal_dim}":
    #             rotated_field_cmp = local_fields[f"E{normal_dim}"] * np.sin(
    #                 self.mode_spec.angle_theta
    #             )
    #             -local_fields[f"E{lateral_dim}"] * np.sin(self.mode_spec.angle_theta)
    #         elif field_name == f"E{axial_dim}":
    #             rotated_field_cmp = local_fields[f"E{axial_dim}"]
    #         if field_name == f"H{lateral_dim}":
    #             rotated_field_cmp = local_fields[f"H{lateral_dim}"] * np.cos(
    #                 self.mode_spec.angle_theta
    #             )
    #             +local_fields[f"H{normal_dim}"] * np.sin(self.mode_spec.angle_theta)
    #         elif field_name == f"H{normal_dim}":
    #             rotated_field_cmp = local_fields[f"H{normal_dim}"] * np.sin(
    #                 self.mode_spec.angle_theta
    #             )
    #             -local_fields[f"H{lateral_dim}"] * np.sin(self.mode_spec.angle_theta)
    #         elif field_name == f"H{axial_dim}":
    #             rotated_field_cmp = local_fields[f"H{axial_dim}"]

    #         coords = {"x": x, "y": y, "z": z, "f": f, "mode_index": mode_index}
    #         rotated_data_arrays[field_name] = ScalarModeFieldDataArray(
    #             rotated_field_cmp, coords=coords
    #         )

    #     rotated_data_arrays["n_complex"] = solver_ref_data["n_complex"]

    #     return rotated_data_arrays

    def _mode_rotation(
        self,
        solver_ref_data_cylindrical: Dict[
            Union[ScalarModeFieldCylindricalDataArray, ModeIndexDataArray]
        ],
        solver: ModeSolver,
    ) -> ModeSolverData:
        """Rotate the mode solver solution from the reference plane in cylindrical coordinates
        to the desired monitor plane."""
        rotated_data_arrays = {}
        for field_name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
            if self.colocate is True:
                # Get colocation coordinates in the solver plane
                normal_dim, _ = self.plane.pop_axis("xyz", self.normal_axis)
                colocate_coords = self._get_colocation_coordinates()
                colocate_coords[normal_dim] = np.atleast_1d(self.plane.center[self.normal_axis])
                x = colocate_coords["x"]
                y = colocate_coords["y"]
                z = colocate_coords["z"]
                xyz_coords = [x.copy(), y.copy(), z.copy()]
            else:
                # Extract coordinate values from one of the six field components
                xyz_coords = solver.grid_snapped[field_name].to_list
                x, y, z = (coord.copy() for coord in xyz_coords)

            f = np.atleast_1d(self.freqs)
            mode_index = np.arange(self.mode_spec.num_modes)

            # Initialize output arrays
            shape = (x.size, y.size, z.size, len(f), mode_index.size)
            rotated_field_cmp = np.zeros(shape, dtype=complex)

            idx_w, idx_uv = self.plane.pop_axis((0, 1, 2), axis=self.bend_axis_3d)
            idx_u, idx_v = idx_uv

            pts = [coord.copy() for coord in xyz_coords]

            pts[idx_u] -= self.bend_center[idx_u]
            pts[idx_v] -= self.bend_center[idx_v]

            rho = np.sqrt(pts[idx_u] ** 2 + pts[idx_v] ** 2)
            theta = np.arctan2(pts[idx_v], pts[idx_u])
            axial = pts[idx_w]

            theta_rel = theta - self.theta_reference

            cos_theta = pts[idx_u] / rho
            sin_theta = pts[idx_v] / rho

            cmp_normal, source_names = self.plane.pop_axis(("x", "y", "z"), axis=self.bend_axis_3d)
            cmp_1, cmp_2 = source_names

            k0 = 2 * np.pi * f / C_0
            n_eff = solver_ref_data_cylindrical["n_complex"].values
            beta = k0[:, None] * n_eff * np.abs(self._bend_radius)

            lateral_axis = 3 - self.normal_axis - self.bend_axis_3d

            # Interpolate field components
            fields = {}
            for field in ["E", "H"]:
                for comp in ["r", "theta", "axial"]:
                    data = (
                        solver_ref_data_cylindrical[f"{field}{comp}"]
                        .isel(theta=0)
                        .interp(
                            rho=rho,
                            axial=axial,
                            method="linear",
                            kwargs={"fill_value": "extrapolate"},
                        )
                    )

                    if lateral_axis > self.bend_axis_3d:
                        data = data.transpose("axial", ...)

                    fields[f"{field}{comp}"] = data.values

            # Determine the phase factor based on normal_axis_2d
            sign = -1 if self.normal_axis_2d == 0 else 1
            if (self.direction == "+" and self._bend_radius >= 0) or (
                self.direction == "-" and self._bend_radius < 0
            ):
                phase = np.exp(sign * 1j * theta_rel[:, None, None] * beta)
            else:
                phase = np.exp(sign * -1j * theta_rel[:, None, None] * beta)

            # Set fixed index to normal_axis
            idx = [slice(None)] * 3
            idx[self.normal_axis] = 0
            idx_x, idx_y, idx_z = idx

            # Assign rotated fields
            if lateral_axis > self.bend_axis_3d:
                phase_expansion = phase[None, :]
                cos_theta_expansion = cos_theta[None, :, None, None]
                sin_theta_expansion = sin_theta[None, :, None, None]
            else:
                phase_expansion = phase[:, None]
                cos_theta_expansion = cos_theta[:, None, None, None]
                sin_theta_expansion = sin_theta[:, None, None, None]

            if field_name == f"E{cmp_1}":
                rotated_field_cmp[idx_x, idx_y, idx_z, :, :] = phase_expansion * (
                    fields["Er"] * cos_theta_expansion - fields["Etheta"] * sin_theta_expansion
                )
            elif field_name == f"E{cmp_2}":
                rotated_field_cmp[idx_x, idx_y, idx_z, :, :] = phase_expansion * (
                    fields["Er"] * sin_theta_expansion + fields["Etheta"] * cos_theta_expansion
                )
            elif field_name == f"E{cmp_normal}":
                rotated_field_cmp[idx_x, idx_y, idx_z, :, :] = phase_expansion * fields["Eaxial"]
            elif field_name == f"H{cmp_1}":
                rotated_field_cmp[idx_x, idx_y, idx_z, :, :] = phase_expansion * (
                    fields["Hr"] * cos_theta_expansion - fields["Htheta"] * sin_theta_expansion
                )
            elif field_name == f"H{cmp_2}":
                rotated_field_cmp[idx_x, idx_y, idx_z, :, :] = phase_expansion * (
                    fields["Hr"] * sin_theta_expansion + fields["Htheta"] * cos_theta_expansion
                )
            elif field_name == f"H{cmp_normal}":
                rotated_field_cmp[idx_x, idx_y, idx_z, :, :] = phase_expansion * fields["Haxial"]

            coords = {"x": x, "y": y, "z": z, "f": f, "mode_index": mode_index}
            rotated_data_arrays[field_name] = ScalarModeFieldDataArray(
                rotated_field_cmp, coords=coords
            )

        rotated_data_arrays["n_complex"] = solver_ref_data_cylindrical["n_complex"]

        return rotated_data_arrays

    @cached_property
    def theta_reference(self) -> float:
        """Computes the azimutal angle of the reference modal plane."""
        _, local_coords = self.plane.pop_axis((0, 1, 2), axis=self.bend_axis_3d)
        local_coord_x, local_coord_y = local_coords
        theta_ref = np.arctan2(
            self.plane.center[local_coord_y] - self.bend_center[local_coord_y],
            self.plane.center[local_coord_x] - self.bend_center[local_coord_x],
        )

        return theta_ref

    @cached_property
    def _bend_radius(self):
        """A bend_radius to use when ``angle_rotation`` is on. When there is no bend defined, we
        use an effectively very large radius, much larger than the mode plane. This is only used
        for the rotation of the fields - the reference modes are still computed without any
        bend applied."""
        if self.mode_spec.bend_radius is not None:
            return self.mode_spec.bend_radius
        mode_plane_bnds = self.plane.bounds_intersection(self.plane.bounds, self.simulation.bounds)
        largest_dim = np.amax(np.array(mode_plane_bnds[1]) - np.array(mode_plane_bnds[0]))
        return EFFECTIVE_RADIUS_FACTOR * largest_dim

    @cached_property
    def bend_center(self) -> List:
        """Computes the bend center based on plane center, angle_theta and angle_phi."""
        _, id_bend_uv = self.plane.pop_axis((0, 1, 2), axis=self.bend_axis_3d)

        id_bend_u, id_bend_v = id_bend_uv

        theta = self.mode_spec.angle_theta
        phi = self.mode_spec.angle_phi
        bend_radius = self._bend_radius
        bend_center = list(self.plane.center)

        angle_map = {
            (0, 2): lambda: theta * np.cos(phi),
            (0, 1): lambda: theta * np.sin(phi),
            (1, 2): lambda: theta * np.cos(phi),
            (1, 0): lambda: theta * np.sin(phi),
            (2, 0): lambda: theta * np.sin(phi),
            (2, 1): lambda: theta * np.cos(phi),
        }

        update_map = {
            (0, 2): lambda angle: (bend_radius * np.sin(angle), -bend_radius * np.cos(angle)),
            (0, 1): lambda angle: (bend_radius * np.sin(angle), -bend_radius * np.cos(angle)),
            (1, 2): lambda angle: (-bend_radius * np.cos(angle), bend_radius * np.sin(angle)),
            (1, 0): lambda angle: (bend_radius * np.sin(angle), -bend_radius * np.cos(angle)),
            (2, 0): lambda angle: (-bend_radius * np.cos(angle), bend_radius * np.sin(angle)),
            (2, 1): lambda angle: (-bend_radius * np.cos(angle), bend_radius * np.sin(angle)),
        }

        if (self.normal_axis, self.bend_axis_3d) in angle_map:
            angle = angle_map[(self.normal_axis, self.bend_axis_3d)]()
            delta_u, delta_v = update_map[(self.normal_axis, self.bend_axis_3d)](angle)
            bend_center[id_bend_u] = self.plane.center[id_bend_u] + delta_u
            bend_center[id_bend_v] = self.plane.center[id_bend_v] + delta_v

        return bend_center

    def _data_on_yee_grid(self) -> ModeSolverData:
        """Solve for all modes, and construct data with fields on the Yee grid."""

        # we try to do reduced simulation copy for efficiency
        # it should never fail -- if it does, this is likely due to an oversight
        # in the Simulation.subsection method. but falling back to non-reduced
        # simulation prevents unneeded errors in this case
        try:
            solver = self.reduced_simulation_copy
        except Exception as e:
            solver = self
            log.warning(
                "Mode solver reduced_simulation_copy failed. "
                "Falling back to non-reduced simulation, which may be slower. "
                f"Exception: {str(e)}"
            )

        _, _solver_coords = solver.plane.pop_axis(
            solver._solver_grid.boundaries.to_list, axis=solver.normal_axis
        )

        # Compute and store the modes at all frequencies
        n_complex, fields, eps_spec = solver._solve_all_freqs(
            coords=_solver_coords, symmetry=solver.solver_symmetry
        )

        # start a dictionary storing the data arrays for the ModeSolverData
        index_data = ModeIndexDataArray(
            np.stack(n_complex, axis=0),
            coords=dict(
                f=list(solver.freqs),
                mode_index=np.arange(solver.mode_spec.num_modes),
            ),
        )
        data_dict = {"n_complex": index_data}

        # Construct the field data on Yee grid
        for field_name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
            xyz_coords = solver.grid_snapped[field_name].to_list
            scalar_field_data = ScalarModeFieldDataArray(
                np.stack([field_freq[field_name] for field_freq in fields], axis=-2),
                coords=dict(
                    x=xyz_coords[0],
                    y=xyz_coords[1],
                    z=xyz_coords[2],
                    f=list(solver.freqs),
                    mode_index=np.arange(solver.mode_spec.num_modes),
                ),
            )
            data_dict[field_name] = scalar_field_data

        # finite grid corrections
        grid_factors = solver._grid_correction(
            simulation=solver.simulation,
            plane=solver.plane,
            mode_spec=solver.mode_spec,
            n_complex=index_data,
            direction=solver.direction,
        )

        # make mode solver data on the Yee grid
        mode_solver_monitor = solver.to_mode_solver_monitor(name=MODE_MONITOR_NAME, colocate=False)
        grid_expanded = solver.simulation.discretize_monitor(mode_solver_monitor)
        mode_solver_data = ModeSolverData(
            monitor=mode_solver_monitor,
            symmetry=solver.simulation.symmetry,
            symmetry_center=solver.simulation.center,
            grid_expanded=grid_expanded,
            grid_primal_correction=grid_factors[0],
            grid_dual_correction=grid_factors[1],
            eps_spec=eps_spec,
            **data_dict,
        )

        return mode_solver_data

    def _data_on_yee_grid_relative(self, basis: ModeSolverData) -> ModeSolverData:
        """Solve for all modes, and construct data with fields on the Yee grid."""
        if basis.monitor.colocate:
            raise ValidationError("Relative mode solver 'basis' must have 'colocate=False'.")
        _, _solver_coords = self.plane.pop_axis(
            self._solver_grid.boundaries.to_list, axis=self.normal_axis
        )

        basis_fields = []
        for freq_ind in range(len(basis.n_complex.f)):
            basis_fields_freq = {}
            for field_name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
                basis_fields_freq[field_name] = (
                    basis.field_components[field_name].isel(f=freq_ind).to_numpy()
                )
            basis_fields.append(basis_fields_freq)

        # Compute and store the modes at all frequencies
        n_complex, fields, eps_spec = self._solve_all_freqs_relative(
            coords=_solver_coords, symmetry=self.solver_symmetry, basis_fields=basis_fields
        )

        # start a dictionary storing the data arrays for the ModeSolverData
        index_data = ModeIndexDataArray(
            np.stack(n_complex, axis=0),
            coords=dict(
                f=list(self.freqs),
                mode_index=np.arange(self.mode_spec.num_modes),
            ),
        )
        data_dict = {"n_complex": index_data}

        # Construct the field data on Yee grid
        for field_name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
            xyz_coords = self.grid_snapped[field_name].to_list
            scalar_field_data = ScalarModeFieldDataArray(
                np.stack([field_freq[field_name] for field_freq in fields], axis=-2),
                coords=dict(
                    x=xyz_coords[0],
                    y=xyz_coords[1],
                    z=xyz_coords[2],
                    f=list(self.freqs),
                    mode_index=np.arange(self.mode_spec.num_modes),
                ),
            )
            data_dict[field_name] = scalar_field_data

        # finite grid corrections
        grid_factors = self._grid_correction(
            simulation=self.simulation,
            plane=self.plane,
            mode_spec=self.mode_spec,
            n_complex=index_data,
            direction=self.direction,
        )

        # make mode solver data on the Yee grid
        mode_solver_monitor = self.to_mode_solver_monitor(name=MODE_MONITOR_NAME, colocate=False)
        grid_expanded = self.simulation.discretize_monitor(mode_solver_monitor)
        mode_solver_data = ModeSolverData(
            monitor=mode_solver_monitor,
            symmetry=self.simulation.symmetry,
            symmetry_center=self.simulation.center,
            grid_expanded=grid_expanded,
            grid_primal_correction=grid_factors[0],
            grid_dual_correction=grid_factors[1],
            eps_spec=eps_spec,
            **data_dict,
        )

        return mode_solver_data

    def _get_colocation_coordinates(self) -> Dict[str, ArrayFloat1D]:
        """Get colocation coordinates in the solver plane.

        Returns:
            colocate_coords (dict): Dictionary containing the colocation coordinates for each dimension.
        """
        # Get colocation coordinates in the solver plane
        _, plane_dims = self.plane.pop_axis("xyz", self.normal_axis)
        colocate_coords = {}

        for dim, sym in zip(plane_dims, self.solver_symmetry):
            coords = self.grid_snapped.boundaries.to_dict[dim]
            if len(coords) > 2:
                if sym == 0:
                    colocate_coords[dim] = coords[1:-1]
                else:
                    colocate_coords[dim] = coords[:-1]

        return colocate_coords

    def _colocate_data(self, mode_solver_data: ModeSolverData) -> ModeSolverData:
        """Colocate data to Yee grid boundaries."""

        colocate_coords = self._get_colocation_coordinates()

        # Colocate input data to new coordinates
        data_dict_colocated = {}
        for key, field in mode_solver_data.symmetry_expanded.field_components.items():
            data_dict_colocated[key] = field.interp(**colocate_coords).astype(field.dtype)

        # Update data
        mode_solver_monitor = self.to_mode_solver_monitor(name=MODE_MONITOR_NAME)
        grid_expanded = self.simulation.discretize_monitor(mode_solver_monitor)
        data_dict_colocated.update({"monitor": mode_solver_monitor, "grid_expanded": grid_expanded})
        mode_solver_data = mode_solver_data._updated(update=data_dict_colocated)

        return mode_solver_data

    def _normalize_modes(self, mode_solver_data: ModeSolverData):
        """Normalize modes. Note: this modifies ``mode_solver_data`` in-place."""
        scaling = np.sqrt(np.abs(mode_solver_data.flux))
        for field in mode_solver_data.field_components.values():
            field /= scaling

    def _filter_components(self, mode_solver_data: ModeSolverData):
        skip_components = {
            comp: None
            for comp in mode_solver_data.field_components.keys()
            if comp not in self.fields
        }
        return mode_solver_data.updated_copy(**skip_components, validate=False)

    def _filter_polarization(self, mode_solver_data: ModeSolverData):
        """Filter polarization. Note: this modifies ``mode_solver_data`` in-place."""
        pol_frac = mode_solver_data.pol_fraction
        for ifreq in range(len(self.freqs)):
            te_frac = pol_frac.te.isel(f=ifreq)
            if self.mode_spec.filter_pol == "te":
                sort_inds = np.concatenate(
                    (
                        np.where(te_frac >= 0.5)[0],
                        np.where(te_frac < 0.5)[0],
                        np.where(np.isnan(te_frac))[0],
                    )
                )
            elif self.mode_spec.filter_pol == "tm":
                sort_inds = np.concatenate(
                    (
                        np.where(te_frac <= 0.5)[0],
                        np.where(te_frac > 0.5)[0],
                        np.where(np.isnan(te_frac))[0],
                    )
                )
            for data in list(mode_solver_data.field_components.values()) + [
                mode_solver_data.n_complex,
                mode_solver_data.grid_primal_correction,
                mode_solver_data.grid_dual_correction,
            ]:
                data.values[..., ifreq, :] = data.values[..., ifreq, sort_inds]

    @cached_property
    def data(self) -> ModeSolverData:
        """:class:`.ModeSolverData` containing the field and effective index data.

        Returns
        -------
        ModeSolverData
            :class:`.ModeSolverData` object containing the effective index and mode fields.
        """
        mode_solver_data = self.data_raw
        return mode_solver_data.symmetry_expanded_copy

    @cached_property
    def sim_data(self) -> MODE_SIMULATION_DATA_TYPE:
        """:class:`.SimulationData` object containing the :class:`.ModeSolverData` for this object.

        Returns
        -------
        SimulationData
            :class:`.SimulationData` object containing the effective index and mode fields.
        """
        monitor_data = self.data
        new_monitors = list(self.simulation.monitors) + [monitor_data.monitor]
        new_simulation = self.simulation.copy(update=dict(monitors=new_monitors))
        if isinstance(new_simulation, Simulation):
            return SimulationData(simulation=new_simulation, data=(monitor_data,))
        elif isinstance(new_simulation, EMESimulation):
            return EMESimulationData(
                simulation=new_simulation, data=(monitor_data,), smatrix=None, port_modes=None
            )
        else:
            raise SetupError(
                "The 'simulation' provided does not correspond to any known "
                "'AbstractSimulationData' type."
            )

    def _get_epsilon(self, freq: float) -> ArrayComplex4D:
        """Compute the epsilon tensor in the plane. Order of components is xx, xy, xz, yx, etc."""
        eps_keys = ["Ex", "Exy", "Exz", "Eyx", "Ey", "Eyz", "Ezx", "Ezy", "Ez"]
        eps_tensor = [
            self.simulation.epsilon_on_grid(self._solver_grid, key, freq) for key in eps_keys
        ]
        return np.stack(eps_tensor, axis=0)

    @staticmethod
    def _tensorial_material_profile_modal_plane_tranform(
        mat_data: ArrayComplex4D, normal_axis: Axis
    ) -> ArrayComplex4D:
        """For tensorial material response function such as epsilon and mu, pick and tranform it to
        modal plane with normal axis rotated to z.
        """
        # get rid of normal axis
        mat_tensor = np.take(mat_data, indices=[0], axis=1 + normal_axis)
        mat_tensor = np.squeeze(mat_tensor, axis=1 + normal_axis)

        # convert to into 3-by-3 representation for easier axis swap
        flat_shape = np.shape(mat_tensor)  # 9 components flat
        tensor_shape = [3, 3] + list(flat_shape[1:])  # 3-by-3 matrix
        mat_tensor = mat_tensor.reshape(tensor_shape)

        # swap axes to plane coordinates (normal_axis goes to z)
        if normal_axis == 0:
            # swap x and y
            mat_tensor[[0, 1], :, ...] = mat_tensor[[1, 0], :, ...]
            mat_tensor[:, [0, 1], ...] = mat_tensor[:, [1, 0], ...]
        if normal_axis <= 1:
            # swap x (normal_axis==0) or y (normal_axis==1) and z
            mat_tensor[[1, 2], :, ...] = mat_tensor[[2, 1], :, ...]
            mat_tensor[:, [1, 2], ...] = mat_tensor[:, [2, 1], ...]

        # back to "flat" representation
        mat_tensor = mat_tensor.reshape(flat_shape)

        # construct to feed to mode solver
        return mat_tensor

    @staticmethod
    def _diagonal_material_profile_modal_plane_tranform(
        mat_data: ArrayComplex4D, normal_axis: Axis
    ) -> ArrayComplex3D:
        """For diagonal material response function such as epsilon and mu, pick and tranform it to
        modal plane with normal axis rotated to z.
        """
        # get rid of normal axis
        mat_tensor = np.take(mat_data, indices=[0], axis=1 + normal_axis)
        mat_tensor = np.squeeze(mat_tensor, axis=1 + normal_axis)

        # swap axes to plane coordinates (normal_axis goes to z)
        if normal_axis == 0:
            # swap x and y
            mat_tensor[[0, 1], :, ...] = mat_tensor[[1, 0], :, ...]
        if normal_axis <= 1:
            # swap x (normal_axis==0) or y (normal_axis==1) and z
            mat_tensor[[1, 2], :, ...] = mat_tensor[[2, 1], :, ...]

        # construct to feed to mode solver
        return mat_tensor

    def _solver_eps(self, freq: float) -> ArrayComplex4D:
        """Diagonal permittivity in the shape needed by solver, with normal axis rotated to z."""

        # Get diagonal epsilon components in the plane
        eps_tensor = self._get_epsilon(freq)
        # tranformation
        return self._tensorial_material_profile_modal_plane_tranform(eps_tensor, self.normal_axis)

    def _solve_all_freqs(
        self,
        coords: Tuple[ArrayFloat1D, ArrayFloat1D],
        symmetry: Tuple[Symmetry, Symmetry],
    ) -> Tuple[List[float], List[Dict[str, ArrayComplex4D]], List[EpsSpecType]]:
        """Call the mode solver at all requested frequencies."""

        fields = []
        n_complex = []
        eps_spec = []
        for freq in self.freqs:
            n_freq, fields_freq, eps_spec_freq = self._solve_single_freq(
                freq=freq, coords=coords, symmetry=symmetry
            )
            fields.append(fields_freq)
            n_complex.append(n_freq)
            eps_spec.append(eps_spec_freq)
        return n_complex, fields, eps_spec

    def _solve_all_freqs_relative(
        self,
        coords: Tuple[ArrayFloat1D, ArrayFloat1D],
        symmetry: Tuple[Symmetry, Symmetry],
        basis_fields: List[Dict[str, ArrayComplex4D]],
    ) -> Tuple[List[float], List[Dict[str, ArrayComplex4D]], List[EpsSpecType]]:
        """Call the mode solver at all requested frequencies."""

        fields = []
        n_complex = []
        eps_spec = []
        for freq, basis_fields_freq in zip(self.freqs, basis_fields):
            n_freq, fields_freq, eps_spec_freq = self._solve_single_freq_relative(
                freq=freq, coords=coords, symmetry=symmetry, basis_fields=basis_fields_freq
            )
            fields.append(fields_freq)
            n_complex.append(n_freq)
            eps_spec.append(eps_spec_freq)

        return n_complex, fields, eps_spec

    @staticmethod
    def _postprocess_solver_fields(solver_fields, normal_axis, plane, mode_spec, coords):
        """Postprocess `solver_fields` from `compute_modes` to proper coordinate"""
        fields = {key: [] for key in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")}
        diff_coords = (np.diff(coords[0]), np.diff(coords[1]))

        for mode_index in range(mode_spec.num_modes):
            # Get E and H fields at the current mode_index
            ((Ex, Ey, Ez), (Hx, Hy, Hz)) = ModeSolver._process_fields(
                solver_fields, mode_index, normal_axis, plane, diff_coords
            )

            # Note: back in original coordinates
            fields_mode = {"Ex": Ex, "Ey": Ey, "Ez": Ez, "Hx": Hx, "Hy": Hy, "Hz": Hz}
            for field_name, field in fields_mode.items():
                fields[field_name].append(field)

        for field_name, field in fields.items():
            fields[field_name] = np.stack(field, axis=-1)
        return fields

    def _solve_single_freq(
        self,
        freq: float,
        coords: Tuple[ArrayFloat1D, ArrayFloat1D],
        symmetry: Tuple[Symmetry, Symmetry],
    ) -> Tuple[float, Dict[str, ArrayComplex4D], EpsSpecType]:
        """Call the mode solver at a single frequency.

        The fields are rotated from propagation coordinates back to global coordinates.
        """

        if not LOCAL_SOLVER_IMPORTED:
            raise ImportError(IMPORT_ERROR_MSG)

        solver_fields, n_complex, eps_spec = compute_modes(
            eps_cross=self._solver_eps(freq),
            coords=coords,
            freq=freq,
            mode_spec=self.mode_spec,
            symmetry=symmetry,
            direction=self.direction,
            precision=self._precision,
            plane_center=self.plane_center_tangential(self.plane),
        )

        fields = self._postprocess_solver_fields(
            solver_fields, self.normal_axis, self.plane, self.mode_spec, coords
        )
        return n_complex, fields, eps_spec

    def _rotate_field_coords_inverse(self, field: FIELD) -> FIELD:
        """Move the propagation axis to the z axis in the array."""
        f_x, f_y, f_z = np.moveaxis(field, source=1 + self.normal_axis, destination=3)
        f_n, f_ts = self.plane.pop_axis((f_x, f_y, f_z), axis=self.normal_axis)
        return np.stack(self.plane.unpop_axis(f_n, f_ts, axis=2), axis=0)

    def _postprocess_solver_fields_inverse(self, fields):
        """Convert ``fields`` to ``solver_fields``. Doesn't change gauge."""
        E = [fields[key] for key in ("Ex", "Ey", "Ez")]
        H = [fields[key] for key in ("Hx", "Hy", "Hz")]

        (Ex, Ey, Ez) = self._rotate_field_coords_inverse(E)
        (Hx, Hy, Hz) = self._rotate_field_coords_inverse(H)

        # apply -1 to H fields if a reflection was involved in the rotation
        if self.normal_axis == 1:
            Hx *= -1
            Hy *= -1
            Hz *= -1

        solver_fields = np.stack((Ex, Ey, Ez, Hx, Hy, Hz), axis=0)
        return solver_fields

    def _solve_single_freq_relative(
        self,
        freq: float,
        coords: Tuple[ArrayFloat1D, ArrayFloat1D],
        symmetry: Tuple[Symmetry, Symmetry],
        basis_fields: Dict[str, ArrayComplex4D],
    ) -> Tuple[float, Dict[str, ArrayComplex4D], EpsSpecType]:
        """Call the mode solver at a single frequency.
        Modes are computed as linear combinations of ``basis_fields``.
        """

        if not LOCAL_SOLVER_IMPORTED:
            raise ImportError(IMPORT_ERROR_MSG)

        solver_basis_fields = self._postprocess_solver_fields_inverse(basis_fields)

        solver_fields, n_complex, eps_spec = compute_modes(
            eps_cross=self._solver_eps(freq),
            coords=coords,
            freq=freq,
            mode_spec=self.mode_spec,
            symmetry=symmetry,
            direction=self.direction,
            solver_basis_fields=solver_basis_fields,
            precision=self._precision,
            plane_center=self.plane_center_tangential(self.plane),
        )

        fields = self._postprocess_solver_fields(
            solver_fields, self.normal_axis, self.plane, self.mode_spec, coords
        )
        return n_complex, fields, eps_spec

    @staticmethod
    def _rotate_field_coords(field: FIELD, normal_axis: Axis, plane: MODE_PLANE_TYPE) -> FIELD:
        """Move the propagation axis=z to the proper order in the array."""
        f_x, f_y, f_z = np.moveaxis(field, source=3, destination=1 + normal_axis)
        return np.stack(plane.unpop_axis(f_z, (f_x, f_y), axis=normal_axis), axis=0)

    @staticmethod
    def _weighted_coord_max(
        array: ArrayFloat2D, u: ArrayFloat1D, v: ArrayFloat1D
    ) -> Tuple[int, int]:
        """2D argmax for an array weighted in both directions."""
        if not np.all(np.isfinite(array)):  # make sure the array is valid
            return 0, 0

        m = array * u.reshape(-1, 1)
        i = np.arange(array.shape[0])
        i = (m * i.reshape(-1, 1)).sum() / m.sum()
        i = int(0.5 + i) if np.isfinite(i) else 0  # in case m.sum() ~ 0

        m = array * v
        j = np.arange(array.shape[1])
        j = (m * j).sum() / m.sum()
        j = int(0.5 + j) if np.isfinite(j) else 0

        return i, j

    @staticmethod
    def _inverted_gauge(e_field: FIELD, diff_coords: Tuple[ArrayFloat1D, ArrayFloat1D]) -> bool:
        """Check if the lower xy region of the mode has a negative sign."""
        dx, dy = diff_coords
        e_x, e_y = e_field[:2, :, :, 0]
        e_x = e_x.real
        e_y = e_y.real
        e = e_x if np.abs(e_x).max() > np.abs(e_y).max() else e_y
        abs_e = np.abs(e)
        e_2 = abs_e**2
        i, j = e.shape
        while i > 0 and j > 0:
            if (e[:i, :j] > 0).all():
                return False
            elif (e[:i, :j] < 0).all():
                return True
            else:
                threshold = abs_e[:i, :j].max() * 0.5
                i, j = ModeSolver._weighted_coord_max(e_2[:i, :j], dx[:i], dy[:j])
                if abs(e[i, j]) >= threshold:
                    return e[i, j] < 0
                # Do not close the window for 1D mode solvers
                if e.shape[0] == 1:
                    i = 1
                elif e.shape[1] == 1:
                    j = 1
        return False

    @staticmethod
    def _process_fields(
        mode_fields: ArrayComplex4D,
        mode_index: pydantic.NonNegativeInt,
        normal_axis: Axis,
        plane: MODE_PLANE_TYPE,
        diff_coords: Tuple[ArrayFloat1D, ArrayFloat1D],
    ) -> Tuple[FIELD, FIELD]:
        """Transform solver fields to simulation axes and set gauge."""

        # Separate E and H fields (in solver coordinates)
        E, H = mode_fields[..., mode_index]

        # Set gauge to highest-amplitude in-plane E being real and positive
        ind_max = np.argmax(np.abs(E[:2]))
        phi = np.angle(E[:2].ravel()[ind_max])
        E *= np.exp(-1j * phi)
        H *= np.exp(-1j * phi)

        # High-order modes with opposite-sign lobes will show inconsistent sign, heavily dependent
        # on the exact local grid to choose the previous gauge. This method flips the gauge sign
        # when necessary to make it more consistent.
        if ModeSolver._inverted_gauge(E, diff_coords):
            E *= -1
            H *= -1

        # Rotate back to original coordinates
        (Ex, Ey, Ez) = ModeSolver._rotate_field_coords(E, normal_axis, plane)
        (Hx, Hy, Hz) = ModeSolver._rotate_field_coords(H, normal_axis, plane)

        # apply -1 to H fields if a reflection was involved in the rotation
        if normal_axis == 1:
            Hx *= -1
            Hy *= -1
            Hz *= -1

        return ((Ex, Ey, Ez), (Hx, Hy, Hz))

    def _field_decay_warning(self, field_data: ModeSolverData):
        """Warn if any of the modes do not decay at the edges."""
        _, plane_dims = self.plane.pop_axis(["x", "y", "z"], axis=self.normal_axis)
        field_sizes = field_data.Ex.sizes
        for freq_index in range(field_sizes["f"]):
            for mode_index in range(field_sizes["mode_index"]):
                e_edge, e_norm = 0, 0
                # Sum up the total field intensity
                for E in (field_data.Ex, field_data.Ey, field_data.Ez):
                    e_norm += np.sum(np.abs(E[{"f": freq_index, "mode_index": mode_index}]) ** 2)
                # Sum up the field intensity at the edges
                if field_sizes[plane_dims[0]] > 1:
                    for E in (field_data.Ex, field_data.Ey, field_data.Ez):
                        isel = {plane_dims[0]: [0, -1], "f": freq_index, "mode_index": mode_index}
                        e_edge += np.sum(np.abs(E[isel]) ** 2)
                if field_sizes[plane_dims[1]] > 1:
                    for E in (field_data.Ex, field_data.Ey, field_data.Ez):
                        isel = {plane_dims[1]: [0, -1], "f": freq_index, "mode_index": mode_index}
                        e_edge += np.sum(np.abs(E[isel]) ** 2)
                # Warn if needed
                if e_edge / e_norm > FIELD_DECAY_CUTOFF:
                    log.warning(
                        f"Mode field at frequency index {freq_index}, mode index {mode_index} does "
                        "not decay at the plane boundaries."
                    )

    @staticmethod
    def _grid_correction(
        simulation: MODE_SIMULATION_TYPE,
        plane: Box,
        mode_spec: ModeSpec,
        n_complex: ModeIndexDataArray,
        direction: Direction,
    ) -> [FreqModeDataArray, FreqModeDataArray]:
        """Correct the fields due to propagation on the grid.

        Return a copy of the :class:`.ModeSolverData` with the fields renormalized to account
        for propagation on a finite grid along the propagation direction. The fields are assumed to
        have ``E exp(1j k r)`` dependence on the finite grid and are then resampled using linear
        interpolation to the exact position of the mode plane. This is needed to correctly compute
        overlap with fields that come from a :class:`.FieldMonitor` placed in the same grid.

        Parameters
        ----------
        grid : :class:`.Grid`
            Numerical grid on which the modes are assumed to propagate.

        Returns
        -------
        :class:`.ModeSolverData`
            Copy of the data with renormalized fields.
        """
        normal_axis = plane.size.index(0.0)
        normal_pos = plane.center[normal_axis]
        normal_dim = "xyz"[normal_axis]

        # Primal and dual grid along the normal direction,
        # i.e. locations of the tangential E-field and H-field components, respectively
        grid = simulation.grid
        normal_primal = grid.boundaries.to_list[normal_axis]
        normal_primal = xr.DataArray(normal_primal, coords={normal_dim: normal_primal})
        normal_dual = grid.centers.to_list[normal_axis]
        normal_dual = xr.DataArray(normal_dual, coords={normal_dim: normal_dual})

        # Propagation phase at the primal and dual locations. The k-vector is along the propagation
        # direction, so angle_theta has to be taken into account. The distance along the propagation
        # direction is the distance along the normal direction over cosine(theta).
        cos_theta = np.cos(mode_spec.angle_theta)
        k_vec = cos_theta * 2 * np.pi * n_complex * n_complex.f / C_0
        if direction == "-":
            k_vec *= -1
        phase_primal = np.exp(1j * k_vec * (normal_primal - normal_pos))
        phase_dual = np.exp(1j * k_vec * (normal_dual - normal_pos))

        # Fields are modified by a linear interpolation to the exact monitor position
        if normal_primal.size > 1:
            phase_primal = phase_primal.interp(**{normal_dim: normal_pos})
        else:
            phase_primal = phase_primal.squeeze(dim=normal_dim)
        if normal_dual.size > 1:
            phase_dual = phase_dual.interp(**{normal_dim: normal_pos})
        else:
            phase_dual = phase_dual.squeeze(dim=normal_dim)

        return FreqModeDataArray(phase_primal), FreqModeDataArray(phase_dual)

    @property
    def _is_tensorial(self) -> bool:
        """Whether the mode computation should be fully tensorial. This is either due to fully
        anisotropic media, or due to an angled waveguide, in which case the transformed eps and mu
        become tensorial. A separate check is done inside the solver, which looks at the actual
        eps and mu and uses a tolerance to determine whether to invoke the tensorial solver, so
        the actual behavior may differ from what's predicted by this property."""
        return abs(self.mode_spec.angle_theta) > 0 or self._has_fully_anisotropic_media

    @cached_property
    def _intersecting_media(self) -> List:
        """List of media (including simulation background) intersecting the mode plane."""
        total_structures = [self.simulation.scene.background_structure]
        total_structures += list(self.simulation.structures)
        return self.simulation.scene.intersecting_media(self.plane, total_structures)

    @cached_property
    def _has_fully_anisotropic_media(self) -> bool:
        """Check if there are any fully anisotropic media in the plane of the mode."""
        if np.any(
            [isinstance(mat, FullyAnisotropicMedium) for mat in self.simulation.scene.mediums]
        ):
            for int_mat in self._intersecting_media:
                if isinstance(int_mat, FullyAnisotropicMedium):
                    return True
        return False

    @cached_property
    def _has_complex_eps(self) -> bool:
        """Check if there are media with a complex-valued epsilon in the plane of the mode.
        A separate check is done inside the solver, which looks at the actual
        eps and mu and uses a tolerance to determine whether to use real or complex fields, so
        the actual behavior may differ from what's predicted by this property."""
        check_freqs = np.unique([np.amin(self.freqs), np.amax(self.freqs), np.mean(self.freqs)])
        for int_mat in self._intersecting_media:
            for freq in check_freqs:
                max_imag_eps = np.amax(np.abs(np.imag(int_mat.eps_model(freq))))
                if not isclose(max_imag_eps, 0):
                    return False
        return True

    @cached_property
    def _contain_good_conductor(self) -> bool:
        """Whether modal plane might contain structures made of good conductors with large permittivity
        or permeability values.
        """
        sim = self.reduced_simulation_copy.simulation
        apply_sibc = isinstance(sim._subpixel.lossy_metal, SurfaceImpedance)
        for medium in sim.scene.mediums:
            if medium.is_pec:
                return True
            if apply_sibc and isinstance(medium, LossyMetalMedium):
                return True
        return False

    @cached_property
    def _precision(self) -> Literal["single", "double"]:
        """single or double precision applied in mode solver."""
        precision = self.mode_spec.precision
        if precision == "auto":
            if self._contain_good_conductor:
                return "double"
            return "single"
        return precision

    def to_source(
        self,
        source_time: SourceTime,
        direction: Direction = None,
        mode_index: pydantic.NonNegativeInt = 0,
        num_freqs: pydantic.PositiveInt = 1,
        **kwargs,
    ) -> ModeSource:
        """Creates :class:`.ModeSource` from a :class:`ModeSolver` instance plus additional
        specifications.

        Parameters
        ----------
        source_time: :class:`.SourceTime`
            Specification of the source time-dependence.
        direction : Direction = None
            Whether source will inject in ``"+"`` or ``"-"`` direction relative to plane normal.
            If not specified, uses the direction from the mode solver.
        mode_index : int = 0
            Index into the list of modes returned by mode solver to use in source.

        Returns
        -------
        :class:`.ModeSource`
            Mode source with specifications taken from the ModeSolver instance and the method
            inputs.
        """

        if direction is None:
            direction = self.direction

        return ModeSource(
            center=self.plane.center,
            size=self.plane.size,
            source_time=source_time,
            mode_spec=self.mode_spec,
            mode_index=mode_index,
            direction=direction,
            num_freqs=num_freqs,
            **kwargs,
        )

    def to_monitor(self, freqs: List[float] = None, name: str = None) -> ModeMonitor:
        """Creates :class:`ModeMonitor` from a :class:`ModeSolver` instance plus additional
        specifications.

        Parameters
        ----------
        freqs : List[float]
            Frequencies to include in Monitor (Hz).
            If not specified, passes ``self.freqs``.
        name : str
            Required name of monitor.

        Returns
        -------
        :class:`.ModeMonitor`
            Mode monitor with specifications taken from the ModeSolver instance and the method
            inputs.
        """

        if freqs is None:
            freqs = self.freqs

        if name is None:
            raise ValueError(
                "A 'name' must be passed to 'ModeSolver.to_monitor'. "
                "The default value of 'None' is for backwards compatibility and is not accepted."
            )

        return ModeMonitor(
            center=self.plane.center,
            size=self.plane.size,
            freqs=freqs,
            mode_spec=self.mode_spec,
            name=name,
        )

    def to_mode_solver_monitor(self, name: str, colocate: bool = None) -> ModeSolverMonitor:
        """Creates :class:`ModeSolverMonitor` from a :class:`ModeSolver` instance.

        Parameters
        ----------
        name : str
            Name of the monitor.
        colocate : bool
            Whether to colocate fields or compute on the Yee grid. If not provided, the value
            set in the :class:`ModeSolver` instance is used.

        Returns
        -------
        :class:`.ModeSolverMonitor`
            Mode monitor with specifications taken from the ModeSolver instance and ``name``.
        """

        if colocate is None:
            colocate = self.colocate

        return ModeSolverMonitor(
            size=self.plane.size,
            center=self.plane.center,
            mode_spec=self.mode_spec,
            freqs=self.freqs,
            direction=self.direction,
            colocate=colocate,
            name=name,
        )

    @require_fdtd_simulation
    def sim_with_source(
        self,
        source_time: SourceTime,
        direction: Direction = None,
        mode_index: pydantic.NonNegativeInt = 0,
    ) -> Simulation:
        """Creates :class:`Simulation` from a :class:`ModeSolver`. Creates a copy of
        the ModeSolver's original simulation with a ModeSource added corresponding to
        the ModeSolver parameters.

        Parameters
        ----------
        source_time: :class:`.SourceTime`
            Specification of the source time-dependence.
        direction : Direction = None
            Whether source will inject in ``"+"`` or ``"-"`` direction relative to plane normal.
            If not specified, uses the direction from the mode solver.
        mode_index : int = 0
            Index into the list of modes returned by mode solver to use in source.

        Returns
        -------
        :class:`.Simulation`
            Copy of the simulation with a :class:`.ModeSource` with specifications taken
            from the ModeSolver instance and the method inputs.
        """

        mode_source = self.to_source(
            mode_index=mode_index, direction=direction, source_time=source_time
        )
        new_sources = list(self.simulation.sources) + [mode_source]
        new_sim = self.simulation.updated_copy(sources=new_sources)
        return new_sim

    @require_fdtd_simulation
    def sim_with_monitor(
        self,
        freqs: List[float] = None,
        name: str = None,
    ) -> Simulation:
        """Creates :class:`.Simulation` from a :class:`ModeSolver`. Creates a copy of
        the ModeSolver's original simulation with a mode monitor added corresponding to
        the ModeSolver parameters.

        Parameters
        ----------
        freqs : List[float] = None
            Frequencies to include in Monitor (Hz).
            If not specified, uses the frequencies from the mode solver.
        name : str
            Required name of monitor.

        Returns
        -------
        :class:`.Simulation`
            Copy of the simulation with a :class:`.ModeMonitor` with specifications taken
            from the ModeSolver instance and the method inputs.
        """

        mode_monitor = self.to_monitor(freqs=freqs, name=name)
        new_monitors = list(self.simulation.monitors) + [mode_monitor]
        new_sim = self.simulation.updated_copy(monitors=new_monitors)
        return new_sim

    def sim_with_mode_solver_monitor(
        self,
        name: str,
    ) -> Simulation:
        """Creates :class:`Simulation` from a :class:`ModeSolver`. Creates a
        copy of the ModeSolver's original simulation with a mode solver monitor
        added corresponding to the ModeSolver parameters.

        Parameters
        ----------
        name : str
            Name of the monitor.

        Returns
        -------
        :class:`.Simulation`
            Copy of the simulation with a :class:`.ModeSolverMonitor` with specifications taken
            from the ModeSolver instance and ``name``.
        """
        mode_solver_monitor = self.to_mode_solver_monitor(name=name)
        new_monitors = list(self.simulation.monitors) + [mode_solver_monitor]
        new_sim = self.simulation.updated_copy(monitors=new_monitors)
        return new_sim

    def plot_field(
        self,
        field_name: str,
        val: Literal["real", "imag", "abs"] = "real",
        scale: PlotScale = "lin",
        eps_alpha: float = 0.2,
        robust: bool = True,
        vmin: float = None,
        vmax: float = None,
        ax: Ax = None,
        **sel_kwargs,
    ) -> Ax:
        """Plot the field for a :class:`.ModeSolverData` with :class:`.Simulation` plot overlaid.

        Parameters
        ----------
        field_name : str
            Name of ``field`` component to plot (eg. ``'Ex'``).
            Also accepts ``'E'`` and ``'H'`` to plot the vector magnitudes of the electric and
            magnetic fields, and ``'S'`` for the Poynting vector.
        val : Literal['real', 'imag', 'abs', 'abs^2', 'dB'] = 'real'
            Which part of the field to plot.
        eps_alpha : float = 0.2
            Opacity of the structure permittivity.
            Must be between 0 and 1 (inclusive).
        robust : bool = True
            If True and vmin or vmax are absent, uses the 2nd and 98th percentiles of the data
            to compute the color limits. This helps in visualizing the field patterns especially
            in the presence of a source.
        vmin : float = None
            The lower bound of data range that the colormap covers. If ``None``, they are
            inferred from the data and other keyword arguments.
        vmax : float = None
            The upper bound of data range that the colormap covers. If ``None``, they are
            inferred from the data and other keyword arguments.
        ax : matplotlib.axes._subplots.Axes = None
            matplotlib axes to plot on, if not specified, one is created.
        sel_kwargs : keyword arguments used to perform ``.sel()`` selection in the monitor data.
            These kwargs can select over the spatial dimensions (``x``, ``y``, ``z``),
            frequency or time dimensions (``f``, ``t``) or `mode_index`, if applicable.
            For the plotting to work appropriately, the resulting data after selection must contain
            only two coordinates with len > 1.
            Furthermore, these should be spatial coordinates (``x``, ``y``, or ``z``).

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.
        """

        sim_data = self.sim_data
        return sim_data.plot_field(
            field_monitor_name=self.data.monitor.name,
            field_name=field_name,
            val=val,
            scale=scale,
            eps_alpha=eps_alpha,
            robust=robust,
            vmin=vmin,
            vmax=vmax,
            ax=ax,
            **sel_kwargs,
        )

    def plot(
        self,
        ax: Ax = None,
        **patch_kwargs,
    ) -> Ax:
        """Plot the mode plane simulation's components.

        Parameters
        ----------
        ax : matplotlib.axes._subplots.Axes = None
            Matplotlib axes to plot on, if not specified, one is created.

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.

        See Also
        ---------

        **Notebooks**
            * `Visualizing geometries in Tidy3D: Plotting Materials <../../notebooks/VizSimulation.html#Plotting-Materials>`_

        """
        # Get the mode plane normal axis, center, and limits.
        a_center, h_lim, v_lim, _ = self._center_and_lims(
            simulation=self.simulation, plane=self.plane
        )

        ax = self.simulation.plot(
            x=a_center[0],
            y=a_center[1],
            z=a_center[2],
            hlim=h_lim,
            vlim=v_lim,
            source_alpha=0,
            monitor_alpha=0,
            lumped_element_alpha=0,
            ax=ax,
            **patch_kwargs,
        )

        return self.plot_pml(ax=ax)

    def plot_eps(
        self,
        freq: float = None,
        alpha: float = None,
        ax: Ax = None,
    ) -> Ax:
        """Plot the mode plane simulation's components.
        The permittivity is plotted in grayscale based on its value at the specified frequency.

        Parameters
        ----------
        freq : float = None
            Frequency to evaluate the relative permittivity of all mediums.
            If not specified, evaluates at infinite frequency.
        alpha : float = None
            Opacity of the structures being plotted.
            Defaults to the structure default alpha.
        ax : matplotlib.axes._subplots.Axes = None
            Matplotlib axes to plot on, if not specified, one is created.

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.

        See Also
        ---------

        **Notebooks**
            * `Visualizing geometries in Tidy3D: Plotting Permittivity <../../notebooks/VizSimulation.html#Plotting-Permittivity>`_
        """

        # Get the mode plane normal axis, center, and limits.
        a_center, h_lim, v_lim, _ = self._center_and_lims(
            simulation=self.simulation, plane=self.plane
        )

        # Plot at central mode frequency if freq is not provided.
        f = freq if freq is not None else self.freqs[len(self.freqs) // 2]

        return self.simulation.plot_eps(
            x=a_center[0],
            y=a_center[1],
            z=a_center[2],
            freq=f,
            alpha=alpha,
            hlim=h_lim,
            vlim=v_lim,
            source_alpha=0,
            monitor_alpha=0,
            lumped_element_alpha=0,
            ax=ax,
        )

    def plot_structures_eps(
        self,
        freq: float = None,
        alpha: float = None,
        cbar: bool = True,
        reverse: bool = False,
        ax: Ax = None,
    ) -> Ax:
        """Plot the mode plane simulation's components.
        The permittivity is plotted in grayscale based on its value at the specified frequency.

        Parameters
        ----------
        freq : float = None
            Frequency to evaluate the relative permittivity of all mediums.
            If not specified, evaluates at infinite frequency.
        alpha : float = None
            Opacity of the structures being plotted.
            Defaults to the structure default alpha.
        cbar : bool = True
            Whether to plot a colorbar for the relative permittivity.
        reverse : bool = False
            If ``False``, the highest permittivity is plotted in black.
            If ``True``, it is plotteed in white (suitable for black backgrounds).
        ax : matplotlib.axes._subplots.Axes = None
            Matplotlib axes to plot on, if not specified, one is created.

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.

        See Also
        ---------

        **Notebooks**
            * `Visualizing geometries in Tidy3D: Plotting Permittivity <../../notebooks/VizSimulation.html#Plotting-Permittivity>`_
        """

        # Get the mode plane normal axis, center, and limits.
        a_center, h_lim, v_lim, _ = self._center_and_lims(
            simulation=self.simulation, plane=self.plane
        )

        # Plot at central mode frequency if freq is not provided.
        f = freq if freq is not None else self.freqs[len(self.freqs) // 2]

        return self.simulation.plot_structures_eps(
            x=a_center[0],
            y=a_center[1],
            z=a_center[2],
            freq=f,
            alpha=alpha,
            cbar=cbar,
            reverse=reverse,
            hlim=h_lim,
            vlim=v_lim,
            ax=ax,
        )

    def plot_grid(
        self,
        ax: Ax = None,
        **kwargs,
    ) -> Ax:
        """Plot the mode plane cell boundaries as lines.

        Parameters
        ----------
        ax : matplotlib.axes._subplots.Axes = None
            Matplotlib axes to plot on, if not specified, one is created.
        **kwargs
            Optional keyword arguments passed to the matplotlib ``LineCollection``.
            For details on accepted values, refer to
            `Matplotlib's documentation <https://tinyurl.com/2p97z4cn>`_.

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.
        """

        # Get the mode plane normal axis, center, and limits.
        a_center, h_lim, v_lim, _ = self._center_and_lims(
            simulation=self.simulation, plane=self.plane
        )

        return self.simulation.plot_grid(
            x=a_center[0], y=a_center[1], z=a_center[2], hlim=h_lim, vlim=v_lim, ax=ax, **kwargs
        )

    @classmethod
    def _plane_grid(cls, simulation: Simulation, plane: Box) -> Tuple[Coords, Coords]:
        """Plane grid for mode solver."""
        # Get the mode plane normal axis, center, and limits.
        _, _, _, t_axes = cls._center_and_lims(simulation=simulation, plane=plane)

        grid_snapped = cls._grid_snapped(simulation=simulation, plane=plane)

        # Mode plane grid.
        plane_grid = grid_snapped.boundaries.to_list
        coord_0 = plane_grid[t_axes[0]]
        coord_1 = plane_grid[t_axes[1]]
        return coord_0, coord_1

    @classmethod
    def _effective_num_pml(
        cls, simulation: Simulation, plane: Box, mode_spec: ModeSpec
    ) -> Tuple[pydantic.NonNegativeFloat, pydantic.NonNegativeFloat]:
        """Number of cells of the mode solver pml."""
        coord_0, coord_1 = cls._plane_grid(simulation=simulation, plane=plane)

        # Number of PML layers in ModeSpec.
        num_pml_0 = mode_spec.num_pml[0]
        num_pml_1 = mode_spec.num_pml[1]
        num_pml_0 = min(num_pml_0, len(coord_0) - 1)
        num_pml_1 = min(num_pml_1, len(coord_1) - 1)
        return (num_pml_0, num_pml_1)

    @classmethod
    def _pml_thickness(
        cls, simulation: Simulation, plane: Box, mode_spec: ModeSpec
    ) -> Tuple[
        Tuple[pydantic.NonNegativeFloat, pydantic.NonNegativeFloat],
        Tuple[pydantic.NonNegativeFloat, pydantic.NonNegativeFloat],
    ]:
        """Thickness of the mode solver pml in the form
        ((plus0, minus0), (plus1, minus1))
        where 0 and 1 are the lexicographically-ordered tangential axes
        to the mode plane.
        """
        # Get the mode plane normal axis, center, and limits.
        solver_symmetry = cls._solver_symmetry(simulation=simulation, plane=plane)
        coord_0, coord_1 = cls._plane_grid(simulation=simulation, plane=plane)

        # Number of PML layers in ModeSpec.
        num_pml_0, num_pml_1 = cls._effective_num_pml(
            simulation=simulation, plane=plane, mode_spec=mode_spec
        )

        # Calculate PML thickness.
        pml_thick_0_plus = 0
        pml_thick_0_minus = 0
        if num_pml_0 > 0:
            pml_thick_0_plus = coord_0[-1] - coord_0[-num_pml_0 - 1]
            pml_thick_0_minus = coord_0[num_pml_0] - coord_0[0]
            if solver_symmetry[0] != 0:
                pml_thick_0_minus = pml_thick_0_plus

        pml_thick_1_plus = 0
        pml_thick_1_minus = 0
        if num_pml_1 > 0:
            pml_thick_1_plus = coord_1[-1] - coord_1[-num_pml_1 - 1]
            pml_thick_1_minus = coord_1[num_pml_1] - coord_1[0]
            if solver_symmetry[1] != 0:
                pml_thick_1_minus = pml_thick_1_plus

        return ((pml_thick_0_plus, pml_thick_0_minus), (pml_thick_1_plus, pml_thick_1_minus))

    @classmethod
    def _mode_plane_size(
        cls, simulation: Simulation, plane: Box
    ) -> Tuple[pydantic.NonNegativeFloat, pydantic.NonNegativeFloat]:
        """The size of the mode plane intersected with the simulation."""
        _, h_lim, v_lim, _ = cls._center_and_lims(simulation=simulation, plane=plane)
        return h_lim[1] - h_lim[0], v_lim[1] - v_lim[0]

    @classmethod
    def _mode_plane_size_no_pml(
        cls, simulation: Simulation, plane: Box, mode_spec: ModeSpec
    ) -> Tuple[pydantic.NonNegativeFloat, pydantic.NonNegativeFloat]:
        """The size of the remaining portion of the mode plane, after the pml
        has been removed."""
        size = cls._mode_plane_size(simulation=simulation, plane=plane)
        pml_thickness = cls._pml_thickness(simulation=simulation, plane=plane, mode_spec=mode_spec)
        size0 = size[0] - pml_thickness[0][0] - pml_thickness[0][1]
        size1 = size[1] - pml_thickness[1][0] - pml_thickness[1][1]
        return size0, size1

    def plot_pml(
        self,
        ax: Ax = None,
    ) -> Ax:
        """Plot the mode plane absorbing boundaries.

        Parameters
        ----------
        ax : matplotlib.axes._subplots.Axes = None
            Matplotlib axes to plot on, if not specified, one is created.

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.
        """
        return self._plot_pml(
            simulation=self.simulation, plane=self.plane, mode_spec=self.mode_spec, ax=ax
        )

    @classmethod
    def _plot_pml(
        cls, simulation: Simulation, plane: Box, mode_spec: ModeSpec, ax: Ax = None
    ) -> Ax:
        """Plot the mode plane absorbing boundaries."""
        # Get the mode plane normal axis, center, and limits.
        _, h_lim, v_lim, _ = cls._center_and_lims(simulation=simulation, plane=plane)

        # Create ax if ax=None.
        if not ax:
            ax = make_ax()
            ax.set_xlim(h_lim)
            ax.set_ylim(v_lim)

        # Number of PML layers in ModeSpec.
        num_pml_0 = mode_spec.num_pml[0]
        num_pml_1 = mode_spec.num_pml[1]

        ((pml_thick_0_plus, pml_thick_0_minus), (pml_thick_1_plus, pml_thick_1_minus)) = (
            cls._pml_thickness(simulation=simulation, plane=plane, mode_spec=mode_spec)
        )

        # Mode Plane width and height
        mp_w, mp_h = cls._mode_plane_size(simulation=simulation, plane=plane)

        # Plot the absorbing layers.
        if num_pml_0 > 0 or num_pml_1 > 0:
            pml_rect = []
            if pml_thick_0_minus > 0:
                pml_rect.append(Rectangle((h_lim[0], v_lim[0]), pml_thick_0_minus, mp_h))
            if pml_thick_0_plus > 0:
                pml_rect.append(
                    Rectangle((h_lim[1] - pml_thick_0_plus, v_lim[0]), pml_thick_0_plus, mp_h)
                )
            if pml_thick_1_minus > 0:
                pml_rect.append(Rectangle((h_lim[0], v_lim[0]), mp_w, pml_thick_1_minus))
            if pml_thick_1_plus > 0:
                pml_rect.append(
                    Rectangle((h_lim[0], v_lim[1] - pml_thick_1_plus), mp_w, pml_thick_1_plus)
                )

            pc = PatchCollection(
                pml_rect,
                alpha=plot_params_pml.alpha,
                facecolor=plot_params_pml.facecolor,
                edgecolor=plot_params_pml.edgecolor,
                hatch=plot_params_pml.hatch,
                zorder=plot_params_pml.zorder,
            )
            ax.add_collection(pc)

        return ax

    @staticmethod
    def _center_and_lims(simulation: Simulation, plane: Box) -> Tuple[List, List, List, List]:
        """Get the mode plane center and limits."""
        normal_axis = plane.size.index(0.0)

        n_axis, t_axes = plane.pop_axis([0, 1, 2], normal_axis)
        a_center = [None, None, None]
        a_center[n_axis] = plane.center[n_axis]

        _, (h_min_s, v_min_s) = Box.pop_axis(simulation.bounds[0], axis=n_axis)
        _, (h_max_s, v_max_s) = Box.pop_axis(simulation.bounds[1], axis=n_axis)

        h_min = plane.center[t_axes[0]] - plane.size[t_axes[0]] / 2
        h_max = plane.center[t_axes[0]] + plane.size[t_axes[0]] / 2
        v_min = plane.center[t_axes[1]] - plane.size[t_axes[1]] / 2
        v_max = plane.center[t_axes[1]] + plane.size[t_axes[1]] / 2

        h_lim = [
            h_min if abs(h_min) < abs(h_min_s) else h_min_s,
            h_max if abs(h_max) < abs(h_max_s) else h_max_s,
        ]
        v_lim = [
            v_min if abs(v_min) < abs(v_min_s) else v_min_s,
            v_max if abs(v_max) < abs(v_max_s) else v_max_s,
        ]

        return a_center, h_lim, v_lim, t_axes

    def _validate_modes_size(self):
        """Make sure that the total size of the modes fields is not too large."""
        monitor = self.to_mode_solver_monitor(name=MODE_MONITOR_NAME)
        num_cells = self.simulation._monitor_num_cells(monitor)
        # size in GB
        total_size = monitor._storage_size_solver(num_cells=num_cells, tmesh=[]) / 1e9
        if total_size > MAX_MODES_DATA_SIZE_GB:
            raise SetupError(
                f"Mode solver has {total_size:.2f}GB of estimated storage, "
                f"a maximum of {MAX_MODES_DATA_SIZE_GB:.2f}GB is allowed. Consider making the "
                "mode plane smaller, or decreasing the resolution or number of requested "
                "frequencies or modes."
            )

    def validate_pre_upload(self, source_required: bool = True):
        """Validate the fully initialized mode solver is ok for upload to our servers."""
        self._validate_modes_size()

    @cached_property
    def reduced_simulation_copy(self):
        """Strip objects not used by the mode solver from simulation object.
        This might significantly reduce upload time in the presence of custom mediums.
        """

        # for now, we handle EME simulation by converting to FDTD simulation
        # because we can't take planar subsection of an EME simulation.
        # eventually, we will convert to ModeSimulation
        if isinstance(self.simulation, EMESimulation):
            return self.as_fdtd_mode_solver.reduced_simulation_copy

        # we preserve extra cells along the normal direction to ensure there is enough data for
        # subpixel
        extended_grid = self._get_solver_grid(
            simulation=self.simulation,
            plane=self.plane,
            keep_additional_layers=True,
            truncate_symmetry=False,
        )
        grids_1d = extended_grid.boundaries
        new_sim_box = Box.from_bounds(
            rmin=(grids_1d.x[0], grids_1d.y[0], grids_1d.z[0]),
            rmax=(grids_1d.x[-1], grids_1d.y[-1], grids_1d.z[-1]),
        )

        # remove PML, Absorers, etc, to avoid unnecessary cells
        bspec = self.simulation.boundary_spec

        new_bspec_dict = {}
        for axis in "xyz":
            bcomp = bspec[axis]
            for bside, sign in zip([bcomp.plus, bcomp.minus], "+-"):
                if isinstance(bside, (PML, StablePML, Absorber)):
                    new_bspec_dict[axis + sign] = PECBoundary()
                else:
                    new_bspec_dict[axis + sign] = bside

        new_bspec = BoundarySpec(
            x=Boundary(plus=new_bspec_dict["x+"], minus=new_bspec_dict["x-"]),
            y=Boundary(plus=new_bspec_dict["y+"], minus=new_bspec_dict["y-"]),
            z=Boundary(plus=new_bspec_dict["z+"], minus=new_bspec_dict["z-"]),
        )

        # extract sub-simulation removing everything irrelevant
        new_sim = self.simulation.subsection(
            region=new_sim_box,
            monitors=[],
            sources=[],
            grid_spec="identical",
            boundary_spec=new_bspec,
            remove_outside_custom_mediums=True,
            remove_outside_structures=True,
            include_pml_cells=True,
            validate_geometries=False,
            deep_copy=False,
        )
        # Let's only validate mode solver where geometry validation is skipped: geometry replaced by its bounding
        # box
        structures = [
            strc.updated_copy(geometry=strc.geometry.bounding_box, deep=False)
            for strc in new_sim.structures
        ]
        # skip validation as it's validated already in subsection
        aux_new_sim = new_sim.updated_copy(structures=structures, deep=False, validate=False)
        # validate mode solver here where geometry is replaced by its bounding box
        new_mode = self.updated_copy(simulation=aux_new_sim, deep=False)
        # return full mode solver and skip validation
        return new_mode.updated_copy(simulation=new_sim, deep=False, validate=False)

    def to_fdtd_mode_solver(self) -> ModeSolver:
        """Construct a new :class:`.ModeSolver` by converting ``simulation``
        from a :class:`.EMESimulation` to an FDTD :class:`.Simulation`.
        Only used as a workaround until :class:`.EMESimulation` is natively supported in the
        :class:`.ModeSolver` webapi."""
        if not isinstance(self.simulation, EMESimulation):
            raise ValidationError(
                "The method 'to_fdtd_mode_solver' is only needed "
                "when the 'simulation' is an 'EMESimulation'."
            )
        fdtd_sim = self.simulation._as_fdtd_sim
        return self.updated_copy(simulation=fdtd_sim)

    @cached_property
    def as_fdtd_mode_solver(self) -> ModeSolver:
        """Construct a new :class:`.ModeSolver` by converting ``simulation``
        from a :class:`.EMESimulation` to an FDTD :class:`.Simulation`.
        Only used as a workaround until :class:`.EMESimulation` is natively supported in the
        :class:`.ModeSolver` webapi."""
        return self.to_fdtd_mode_solver()

    def _patch_data(self, data: ModeSolverData):
        """
        Patch the :class:`.ModeSolver` with the provided data so that
        it will be used everywhere instead of locally-computed data.
        This function is available as a workaround while we transition
        to the new :class:`.ModeSimulation` interface. It is used
        in the webapi to make functions like ``plot_field`` operate
        on the data from the remote mode solver rather than data from
        the local mode solver. It can also be used manually if needed.

        Parameters
        ----------
        data : :class:`.ModeSolverData`
            The mode solver data to be used, typically the result
            of a remote mode solver run.
        """
        self._cached_properties["data_raw"] = data
        self._cached_properties.pop("data", None)
        self._cached_properties.pop("sim_data", None)
