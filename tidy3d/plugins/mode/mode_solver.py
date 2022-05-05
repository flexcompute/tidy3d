"""Solve for modes in a 2D cross-sectional plane in a simulation, assuming translational
invariance along a given propagation axis.
"""

from typing import List, Tuple, Union, Dict
import logging

import h5py
import numpy as np
import pydantic

from ...components.base import Tidy3dBaseModel
from ...components import Box
from ...components import Simulation
from ...components import ModeSpec
from ...components import ModeMonitor
from ...components.source import ModeSource, SourceTime
from ...components.types import Direction, Array, Ax, Literal, ArrayLike, Axis, Symmetry
from ...components.data import Tidy3dData, ModeIndexData, ModeFieldData, ScalarModeFieldData
from ...components.data import AbstractSimulationData
from ...log import ValidationError, DataError

from .solver import compute_modes

FIELD = Tuple[Array[complex], Array[complex], Array[complex]]

# Warning for field intensity at edges over total field intensity larger than this value
FIELD_DECAY_CUTOFF = 1e-2


class ModeSolverData(AbstractSimulationData):
    """Holds data associated with :class:`.ModeSolver`.

    Parameters
    ----------
    plane : :class:`.Box`
        Cross-sectional plane in which the modes were be computed.
    mode_spec : :class:`.ModeSpec`
        Container with specifications about the modes.
    data_dict : Dict[str, Union[ModeFieldData, ModeIndexData]]
        Mapping of "n_complex" to :class:`.ModeIndexData`, and "fields" to :class:`.ModeFieldData`.
    """

    plane: Box
    mode_spec: ModeSpec
    data_dict: Dict[str, Union[ModeFieldData, ModeIndexData]]

    @property
    def fields(self):
        """Get field data."""
        return self.data_dict.get("fields")

    @property
    def n_complex(self):
        """Get complex effective indexes."""
        scalar_data = self.data_dict.get("n_complex")
        if scalar_data:
            return scalar_data.data
        return None

    @property
    def n_eff(self):
        """Get real part of effective index."""
        scalar_data = self.data_dict.get("n_complex")
        if scalar_data:
            return scalar_data.n_eff
        return None

    @property
    def k_eff(self):
        """Get imaginary part of effective index."""
        scalar_data = self.data_dict.get("n_complex")
        if scalar_data:
            return scalar_data.k_eff
        return None

    def add_to_handle(self, handle: Union[h5py.File, h5py.Group]) -> None:
        """Export to an hdf5 handle, which can be a file or a group.

        Parameters
        ----------
        handle : Union[hdf5.File, hdf5.Group]
            Handle to write the ModeSolverData to.
        """

        # save pydantic models as string
        json_dict = {
            "simulation": self.simulation,
            "plane": self.plane,
            "mode_spec": self.mode_spec,
        }
        for name, obj in json_dict.items():
            Tidy3dData.save_string(handle, name, obj.json())

        # make groups for mode fields and index data
        for name, data in self.data_dict.items():
            data_grp = handle.create_group(name)
            data.add_to_group(data_grp)

    @classmethod
    def load_from_handle(cls, handle: Union[h5py.File, h5py.Group]) -> "ModeSolverData":
        """Load from an hdf5 handle, which can be a file or a group.

        Parameters
        ----------
        handle : Union[hdf5.File, hdf5.Group]
            Handle to load the ModeSolverData from.
        """

        # construct pydantic models from string
        json_dict = {
            "simulation": Simulation,
            "plane": Box,
            "mode_spec": ModeSpec,
        }
        obj_dict = {}
        for name, obj in json_dict.items():
            json_string = Tidy3dData.load_string(handle, name)
            obj_dict[name] = obj.parse_raw(json_string)

        # load fields and effective index data
        data_dict = {
            "fields": ModeFieldData.load_from_group(handle["fields"]),
            "n_complex": ModeIndexData.load_from_group(handle["n_complex"]),
        }
        return cls(data_dict=data_dict, **obj_dict)

    def to_file(self, fname: str) -> None:
        """Export :class:`.ModeSolverData` to single hdf5 file.

        Parameters
        ----------
        fname : str
            Path to .hdf5 data file (including filename).
        """

        with h5py.File(fname, "a") as f_handle:
            self.add_to_handle(f_handle)

    @classmethod
    def from_file(cls, fname: str, **kwargs) -> "ModeSolverData":  # pylint:disable=unused-argument
        """Load :class:`.ModeSolverData` from .hdf5 file.

        Parameters
        ----------
        fname : str
            Path to .hdf5 data file (including filename).

        Returns
        -------
        :class:`.ModeSolverData`
            A :class:`.ModeSolverData` instance.
        """

        # read from file at fname
        with h5py.File(fname, "r") as f_handle:
            mode_data = cls.load_from_handle(f_handle)

        return mode_data

    # pylint:disable=too-many-arguments, too-many-locals, too-many-branches, too-many-statements
    def plot_field(
        self,
        field_name: str,
        val: Literal["real", "imag", "abs"] = "real",
        freq: float = None,
        mode_index: int = None,
        eps_alpha: float = 0.2,
        robust: bool = True,
        ax: Ax = None,
        **patch_kwargs,
    ) -> Ax:
        """Plot the field data for a monitor with simulation plot overlayed.

        Parameters
        ----------
        field_name : str
            Name of `field` to plot (eg. 'Ex').
            Also accepts `'int'` to plot intensity.
        val : Literal['real', 'imag', 'abs'] = 'real'
            Which part of the field to plot.
            If ``field_name='int'``, this has no effect.
        freq: float = None
            Specifies the frequency (Hz) to plot.
            Also sets the frequency at which the permittivity is evaluated at (if dispersive).
        mode_index: int = None
            Specifies which mode index to plot.
        eps_alpha : float = 0.2
            Opacity of the structure permittivity.
            Must be between 0 and 1 (inclusive).
        robust : bool = True
            If specified, uses the 2nd and 98th percentiles of the data to compute the color limits.
            This helps in visualizing the field patterns especially in the presence of a source.
        ax : matplotlib.axes._subplots.Axes = None
            matplotlib axes to plot on, if not specified, one is created.
        **patch_kwargs
            Optional keyword arguments passed to ``add_artist(patch, **patch_kwargs)``.

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.
        """

        if mode_index >= self.mode_spec.num_modes:
            raise DataError("``mode_index`` larger than ``mode_spec.num_modes``.")
        mode_fields = self.fields.sel_mode_index(mode_index=mode_index)

        # get the field data component
        if field_name == "int":
            xr_data = 0.0
            for field in ("Ex", "Ey", "Ez"):
                mode_fields = mode_fields[field]
                xr_data += abs(mode_fields) ** 2
            val = "abs"
        else:
            xr_data = mode_fields.data_dict.get(field_name).data

        field_data = xr_data.sel(f=freq, method="nearest")

        axis = self.plane.size.index(0.0)
        position = self.plane.center[axis]

        ax = self.plot_field_array(
            field_data=field_data,
            axis=axis,
            position=position,
            val=val,
            freq=freq,
            eps_alpha=eps_alpha,
            robust=robust,
            ax=ax,
            **patch_kwargs,
        )

        n_eff = self.n_eff.isel(mode_index=mode_index).sel(f=freq, method="nearest")
        title = f"f={float(field_data.f):1.2e}, n_eff={float(n_eff):1.4f}"
        ax.set_title(title)

        return ax


class ModeSolver(Tidy3dBaseModel):
    """Interface for solving electromagnetic eigenmodes in a 2D plane with translational
    invariance in the third dimension.
    """

    simulation: Simulation = pydantic.Field(
        ..., title="Simulation", description="Simulation defining all structures and mediums."
    )

    plane: Box = pydantic.Field(
        ..., title="Plane", description="Cross-sectional plane in which the mode will be computed."
    )

    mode_spec: ModeSpec = pydantic.Field(
        ...,
        title="Mode specification",
        description="Container with specifications about the modes to be solved for.",
    )

    freqs: Union[List[float], ArrayLike] = pydantic.Field(
        ..., title="Frequencies", description="A list of frequencies at which to solve."
    )

    @pydantic.validator("plane", always=True)
    def is_plane(cls, val):
        """Raise validation error if not planar."""
        if val.size.count(0.0) != 1:
            raise ValidationError(f"ModeSolver plane must be planar, given size={val}")
        return val

    @pydantic.validator("freqs", always=True)
    def freqs_not_empty(cls, val):
        """Raise validation error if ``freqs`` is an empty list."""
        if len(val) == 0:
            raise ValidationError("ModeSolver 'freqs' must be a non-empty list.")
        return val

    @property
    def normal_axis(self) -> Axis:
        """Axis normal to the mode plane."""
        return self.plane.size.index(0.0)

    @property
    def _plane_sym(self) -> Box:
        """Potentially smaller plane if symmetries present in the simulation."""
        return self.simulation.min_sym_box(self.plane)

    # pylint:disable=too-many-locals
    def solve(self) -> ModeSolverData:
        """Finds the modal profile and effective index of the modes.

        Returns
        -------
        ModeSolverData
            :class:`.ModeSolverData` object containing the effective index and mode fields for all
            modes.
        """

        normal_axis = self.normal_axis

        # get the in-plane grid coordinates on which eps and the mode fields live
        plane_grid = self.simulation.discretize(self.plane)

        # restrict to a smaller plane if symmetries present in the simulation
        plane_grid_sym = self.simulation.discretize(self._plane_sym)

        # Coords and symmetry arguments to the solver (restricted to in-plane)
        _, solver_coords = self.plane.pop_axis(plane_grid_sym.boundaries.to_list, axis=normal_axis)
        mode_symmetry = list(self.simulation.symmetry)
        for dim in range(3):
            if self.simulation.center[dim] != self.plane.center[dim]:
                mode_symmetry[dim] = 0
        _, solver_symmetry = self.plane.pop_axis(mode_symmetry, axis=normal_axis)

        # Compute and store the modes at all frequencies
        n_complex, fields = self._solve_all_freqs(coords=solver_coords, symmetry=solver_symmetry)

        # Generate the dictionary of ScalarModeFieldData for every field
        data_dict = {}
        for field_name in fields[0].keys():
            xyz_coords = plane_grid_sym[field_name].to_list
            xyz_coords[normal_axis] = [self.plane.center[normal_axis]]
            data_dict[field_name] = ScalarModeFieldData(
                x=xyz_coords[0],
                y=xyz_coords[1],
                z=xyz_coords[2],
                f=self.freqs,
                mode_index=np.arange(self.mode_spec.num_modes),
                values=np.stack([field_freq[field_name] for field_freq in fields], axis=-2),
            )
        field_data = ModeFieldData(
            data_dict=data_dict,
            expanded_grid=plane_grid.yee.grid_dict,
            symmetry_center=self.simulation.center,
            symmetry=self.simulation.symmetry,
        )
        field_data = field_data.expand_syms
        self._field_decay_warning(field_data)
        index_data = ModeIndexData(
            f=self.freqs,
            mode_index=np.arange(self.mode_spec.num_modes),
            values=np.stack(n_complex, axis=0),
        )
        mode_data = ModeSolverData(
            simulation=self.simulation,
            plane=self.plane,
            mode_spec=self.mode_spec,
            data_dict={"fields": field_data, "n_complex": index_data},
        )

        return mode_data

    def _get_epsilon(self, plane: Box, freq: float) -> Array[complex]:
        """Compute the diagonal components of the epsilon tensor in the plane."""

        eps_xx = self.simulation.epsilon(plane, "Ex", freq)
        eps_yy = self.simulation.epsilon(plane, "Ey", freq)
        eps_zz = self.simulation.epsilon(plane, "Ez", freq)

        return np.stack((eps_xx, eps_yy, eps_zz), axis=0)

    def _solver_eps(self, freq: float) -> Array[complex]:
        """Get the diagonal permittivity in the shape needed to be supplied to the sovler, with the
        normal axis rotated to z."""

        # Get diagonal epsilon components in the plane
        (eps_xx, eps_yy, eps_zz) = self._get_epsilon(self._plane_sym, freq)

        # get rid of normal axis
        eps_xx = np.squeeze(eps_xx, axis=self.normal_axis)
        eps_yy = np.squeeze(eps_yy, axis=self.normal_axis)
        eps_zz = np.squeeze(eps_zz, axis=self.normal_axis)

        # swap axes to plane coordinates (normal_axis goes to z)
        eps_sim_ax = (eps_xx, eps_yy, eps_zz)
        eps_zz, (eps_xx, eps_yy) = self.plane.pop_axis(eps_sim_ax, axis=self.normal_axis)

        # construct eps to feed to mode solver
        return np.stack((eps_xx, eps_yy, eps_zz), axis=0)

    def _solve_all_freqs(
        self, coords: Tuple[Array, Array], symmetry: Tuple[Symmetry, Symmetry]
    ) -> Tuple[List[float], List[Dict[str, Array[complex]]]]:
        """Call the mode solver at all requested frequencies."""

        fields = []
        n_complex = []
        for freq in self.freqs:
            n_freq, fields_freq = self._solve_single_freq(
                freq=freq, coords=coords, symmetry=symmetry
            )
            fields.append(fields_freq)
            n_complex.append(n_freq)

        return n_complex, fields

    def _solve_single_freq(
        self, freq: float, coords: Tuple[Array, Array], symmetry: Tuple[Symmetry, Symmetry]
    ) -> Tuple[float, Dict[str, Array[complex]]]:
        """Call the mode solver at a single frequency.
        The fields are rotated from propagation coordinates back to global coordinates.
        """

        solver_fields, n_complex = compute_modes(
            eps_cross=self._solver_eps(freq),
            coords=coords,
            freq=freq,
            mode_spec=self.mode_spec,
            symmetry=symmetry,
        )

        fields = {"Ex": [], "Ey": [], "Ez": [], "Hx": [], "Hy": [], "Hz": []}
        for mode_index in range(self.mode_spec.num_modes):
            # Get E and H fields at the current mode_index
            ((Ex, Ey, Ez), (Hx, Hy, Hz)) = self._process_fields(solver_fields, mode_index)

            # Note: back in original coordinates
            fields_mode = {"Ex": Ex, "Ey": Ey, "Ez": Ez, "Hx": Hx, "Hy": Hy, "Hz": Hz}
            for field_name, field in fields_mode.items():
                fields[field_name].append(field)

        for field_name, field in fields.items():
            fields[field_name] = np.stack(field, axis=-1)

        return n_complex, fields

    def _rotate_field_coords(self, field: FIELD) -> FIELD:
        """Move the propagation axis=z to the proper order in the array."""
        f_x, f_y, f_z = np.moveaxis(field, source=3, destination=1 + self.normal_axis)
        f_rot = np.stack(self.plane.unpop_axis(f_z, (f_x, f_y), axis=self.normal_axis), axis=0)
        return f_rot

    def _process_fields(self, mode_fields: Array[complex], mode_index: int) -> Tuple[FIELD, FIELD]:
        """Transform solver fields to simulation axes, set gauge, and check decay at boundaries."""

        # Separate E and H fields (in solver coordinates)
        E, H = mode_fields[..., mode_index]

        # Set gauge to highest-amplitude in-plane E being real and positive
        ind_max = np.argmax(np.abs(E[:2]))
        phi = np.angle(E[:2].ravel()[ind_max])
        E *= np.exp(-1j * phi)
        H *= np.exp(-1j * phi)

        # Rotate back to original coordinates
        (Ex, Ey, Ez) = self._rotate_field_coords(E)
        (Hx, Hy, Hz) = self._rotate_field_coords(H)

        # apply -1 to H fields if a reflection was involved in the rotation
        if self.normal_axis == 1:
            Hx *= -1
            Hy *= -1
            Hz *= -1

        return ((Ex, Ey, Ez), (Hx, Hy, Hz))

    def _field_decay_warning(self, field_data: ModeFieldData):
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
                    logging.warning(
                        f"Mode field at frequency index {freq_index}, mode index {mode_index} does "
                        "not decay at the plane boundaries."
                    )

    def to_source(
        self,
        source_time: SourceTime,
        direction: Direction,
        mode_index: int = 0,
    ) -> ModeSource:
        """Creates :class:`.ModeSource` from a :class:`.ModeSolver` instance plus additional
        specifications.

        Parameters
        ----------
        source_time: :class:`.SourceTime`
            Specification of the source time-dependence.
        direction : Direction
            Whether source will inject in ``"+"`` or ``"-"`` direction relative to plane normal.
        mode_index : int = 0
            Index into the list of modes returned by mode solver to use in source.

        Returns
        -------
        :class:`.ModeSource`
            Mode source with specifications taken from the ModeSolver instance and the method
            inputs.
        """

        return ModeSource(
            center=self.plane.center,
            size=self.plane.size,
            source_time=source_time,
            mode_spec=self.mode_spec,
            mode_index=mode_index,
            direction=direction,
        )

    def to_monitor(self, freqs: List[float], name: str) -> ModeMonitor:
        """Creates :class:`ModeMonitor` from a :class:`.ModeSolver` instance plus additional
        specifications.

        Parameters
        ----------
        freqs : List[float]
            Frequencies to include in Monitor (Hz).
        name : str
            Required name of monitor.

        Returns
        -------
        :class:`.ModeMonitor`
            Mode monitor with specifications taken from the ModeSolver instance and the method
            inputs.
        """

        return ModeMonitor(
            center=self.plane.center,
            size=self.plane.size,
            freqs=freqs,
            mode_spec=self.mode_spec,
            name=name,
        )
