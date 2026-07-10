from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any

import numpy as np

import tidy3d as td
from tidy3d import ModeIndexDataArray
from tidy3d.components.data.data_array import (
    CurrentFreqModeDataArray,
    ImpedanceFreqModeDataArray,
    VoltageFreqModeDataArray,
)
from tidy3d.components.data.point_cloud import POINT_CLOUD_PERMITTIVITY_COMPONENTS
from tidy3d.components.microwave.data.dataset import TransmissionLineDataset
from tidy3d.components.types import TYPE_TAG_STR

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from tidy3d.components.types.monitor_data import MonitorDataTypes


def get_spatial_coords_dict(
    simulation: td.Simulation, monitor: td.Monitor, field_name: str
) -> dict[str, Any]:
    """Return monitor-data spatial coordinates for a monitor field component."""

    grid = simulation.discretize_monitor(monitor)
    spatial_coords = grid.boundaries if monitor.colocate else grid[field_name]
    spatial_coords_dict = spatial_coords.model_dump()

    coords = {}
    for axis, dim in enumerate("xyz"):
        if monitor.size[axis] == 0:
            coords[dim] = [monitor.center[axis]]
        elif monitor.colocate:
            coords[dim] = spatial_coords_dict[dim][:-1]
        else:
            coords[dim] = spatial_coords_dict[dim]

    return coords


class SyntheticMonitorDataFactory:
    """Generate synthetic monitor data that mirrors ``tests.utils.run_emulated``."""

    def __init__(
        self,
        simulation: td.Simulation,
        *,
        x0: complex | float = 1.0,
        data_gen_fn: Callable[[tuple[int, ...]], np.ndarray] = np.random.random,
    ) -> None:
        self.simulation = simulation
        self.x0 = x0
        self.data_gen_fn = data_gen_fn
        self.is_adjoint_sim = isinstance(simulation.post_norm, td.FreqDataArray)

    def _norm_factor(self, freqs: np.ndarray) -> np.ndarray | None:
        if not self.simulation.sources:
            return None

        normalize_index = self.simulation.normalize_index
        if normalize_index is None:
            return None
        if normalize_index < 0 or normalize_index >= len(self.simulation.sources):
            return None

        source_time = self.simulation.sources[normalize_index].source_time
        if not hasattr(source_time, "amp_freq"):
            return None

        norm = np.array([source_time.amp_freq(freq) for freq in freqs], dtype=complex)
        phase_factor = np.exp(1j * source_time.phase)
        denom = source_time.amplitude * phase_factor
        if np.abs(denom) < np.finfo(float).eps:
            denom = np.finfo(float).eps * phase_factor
        norm /= denom
        return norm

    @staticmethod
    def _stabilize_norm(norm: np.ndarray) -> np.ndarray:
        """Clamp very small normalization values to avoid infs in synthetic data."""

        eps = np.finfo(float).eps
        norm = np.array(norm, copy=True)
        small = np.abs(norm) < eps
        if np.any(small):
            norm[small] = eps
        return norm

    def _make_data(
        self,
        coords: dict,
        data_array_type: type,
        *,
        is_complex: bool = False,
    ) -> td.components.data.data_array.DataArray:
        """Make a synthetic DataArray with the supplied coordinates and type."""

        from scipy.ndimage import gaussian_filter

        data_shape = tuple(len(coords[key]) for key in data_array_type._dims)
        data = np.zeros(data_shape, dtype=complex if is_complex else float)

        for source in self.simulation.sources:
            source_time = source.source_time
            source_scale = source_time.amplitude * np.exp(1j * source_time.phase)
            if isinstance(source, td.CustomCurrentSource) and source.current_dataset is not None:
                components = list(source.current_dataset.field_components.values())
                if components:
                    dataset_scale = np.sum([np.mean(comp.values) for comp in components])
                    if dataset_scale != 0:
                        source_scale *= dataset_scale
            if not is_complex:
                source_scale = abs(source_scale)

            source_time_norm = source_time.updated_copy(amplitude=1.0, phase=0.0)
            if isinstance(source, td.CustomCurrentSource) and source.current_dataset is not None:
                src_hash = (
                    f"{type(source).__name__}:{source.center}:{source.size}:"
                    f"{source_time_norm._hash_self()}"
                )
                seed = int(hashlib.md5(src_hash.encode("utf-8")).hexdigest()[:8], 16)
            else:
                source_norm = source.updated_copy(source_time=source_time_norm)
                seed = int(source_norm._hash_self()[:8], 16)

            np.random.seed(seed)
            contrib = self.data_gen_fn(data_shape)
            contrib = (1 + 0.5j) * contrib if is_complex else contrib
            contrib = gaussian_filter(contrib, sigma=1.0)
            data += contrib * source_scale

        if "f" in data_array_type._dims and not self.is_adjoint_sim:
            freqs = np.array(coords["f"], dtype=float)
            norm = self._norm_factor(freqs)
            if norm is not None:
                if not is_complex:
                    norm = np.abs(norm)
                norm = self._stabilize_norm(norm)
                shape = [1] * len(data_shape)
                shape[data_array_type._dims.index("f")] = len(freqs)
                data = data / norm.reshape(shape)

        data_scale = 1e-6 / max(1.0, np.sqrt(float(np.prod(data_shape))))
        return data_array_type(self.x0 * data * data_scale, coords=coords)

    def make_field_data(self, monitor: td.FieldMonitor) -> td.FieldData:
        field_cmps = {}
        grid = self.simulation.discretize_monitor(monitor)
        for field_name in monitor.fields:
            coords = get_spatial_coords_dict(self.simulation, monitor, field_name)
            coords["f"] = list(monitor.freqs)
            field_cmps[field_name] = self._make_data(
                coords=coords, data_array_type=td.ScalarFieldDataArray, is_complex=True
            )

        return td.FieldData(
            monitor=monitor,
            symmetry=(0, 0, 0),
            symmetry_center=self.simulation.center,
            grid_expanded=grid,
            **field_cmps,
        )

    def make_point_cloud_field_data(
        self, monitor: td.PointCloudFieldMonitor
    ) -> td.PointCloudFieldData:
        field_cmps = {}
        coords = {"index": np.asarray(monitor.points.coords["index"]), "f": list(monitor.freqs)}
        for field_name in monitor.fields:
            field_cmps[field_name] = self._make_data(
                coords=coords, data_array_type=td.IndexedFreqDataArray, is_complex=True
            )

        return td.PointCloudFieldData(monitor=monitor, points=monitor.points, **field_cmps)

    def make_point_cloud_permittivity_data(
        self, monitor: td.PointCloudPermittivityMonitor
    ) -> td.PointCloudPermittivityData:
        field_cmps = {}
        coords = {"index": np.asarray(monitor.points.coords["index"]), "f": list(monitor.freqs)}
        for component_name in POINT_CLOUD_PERMITTIVITY_COMPONENTS:
            values = self._make_data(
                coords=coords,
                data_array_type=td.IndexedFreqDataArray,
                is_complex=True,
            )
            field_cmps[component_name] = td.IndexedFreqDataArray(
                1.5**2 + np.abs(values.values),
                coords=coords,
            )

        return td.PointCloudPermittivityData(monitor=monitor, points=monitor.points, **field_cmps)

    def make_field_time_data(self, monitor: td.FieldTimeMonitor) -> td.FieldTimeData:
        field_cmps = {}
        grid = self.simulation.discretize_monitor(monitor)
        tmesh = self.simulation.tmesh
        for field_name in monitor.fields:
            coords = get_spatial_coords_dict(self.simulation, monitor, field_name)
            idx_begin, idx_end = monitor.time_inds(tmesh)
            coords["t"] = tmesh[idx_begin:idx_end]
            field_cmps[field_name] = self._make_data(
                coords=coords, data_array_type=td.ScalarFieldTimeDataArray, is_complex=False
            )

        return td.FieldTimeData(
            monitor=monitor,
            symmetry=(0, 0, 0),
            symmetry_center=self.simulation.center,
            grid_expanded=grid,
            **field_cmps,
        )

    def make_mode_solver_data(self, monitor: td.ModeSolverMonitor) -> td.ModeSolverData:
        field_cmps = {}
        grid = self.simulation.discretize_monitor(monitor)
        index_coords = {
            "f": list(monitor.freqs),
            "mode_index": np.arange(monitor.mode_spec.num_modes),
        }
        index_shape = (len(index_coords["f"]), len(index_coords["mode_index"]))
        index_data = ModeIndexDataArray(
            (1 + 1j) * self.data_gen_fn(index_shape), coords=index_coords
        )

        for field_name in ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]:
            coords = get_spatial_coords_dict(self.simulation, monitor, field_name)
            coords["f"] = list(monitor.freqs)
            coords["mode_index"] = index_coords["mode_index"]
            field_cmps[field_name] = self._make_data(
                coords=coords, data_array_type=td.ScalarModeFieldDataArray, is_complex=True
            )

        return td.ModeSolverData(
            monitor=monitor,
            symmetry=(0, 0, 0),
            symmetry_center=self.simulation.center,
            grid_expanded=grid,
            n_complex=index_data,
            **field_cmps,
        )

    def make_microwave_mode_solver_data(
        self, monitor: td.MicrowaveModeSolverMonitor
    ) -> td.MicrowaveModeSolverData:
        field_cmps = {}
        grid = self.simulation.discretize_monitor(monitor)
        index_coords = {
            "f": list(monitor.freqs),
            "mode_index": np.arange(monitor.mode_spec.num_modes),
        }
        index_shape = (len(index_coords["f"]), len(index_coords["mode_index"]))
        index_data = ModeIndexDataArray(
            (1 + 1j) * self.data_gen_fn(index_shape), coords=index_coords
        )

        for field_name in ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]:
            coords = get_spatial_coords_dict(self.simulation, monitor, field_name)
            coords["f"] = list(monitor.freqs)
            coords["mode_index"] = index_coords["mode_index"]
            field_cmps[field_name] = self._make_data(
                coords=coords, data_array_type=td.ScalarModeFieldDataArray, is_complex=True
            )

        impedance_specs: tuple = monitor.mode_spec._impedance_specs_as_tuple
        if len(impedance_specs) == 1:
            impedance_specs = impedance_specs * monitor.mode_spec.num_modes
        used_mode_inds = [
            mode_index for mode_index, spec in enumerate(impedance_specs) if spec is not None
        ]

        index_coords = {"f": list(monitor.freqs), "mode_index": np.array(used_mode_inds)}
        index_shape = (len(index_coords["f"]), len(index_coords["mode_index"]))
        z0_data = ImpedanceFreqModeDataArray(
            (1000 + 1j) * self.data_gen_fn(index_shape), coords=index_coords
        )
        v_data = VoltageFreqModeDataArray(
            (1000 + 1j) * self.data_gen_fn(index_shape), coords=index_coords
        )
        i_data = CurrentFreqModeDataArray(
            (1000 + 1j) * self.data_gen_fn(index_shape), coords=index_coords
        )
        tl_data = TransmissionLineDataset(Z0=z0_data, voltage_coeffs=v_data, current_coeffs=i_data)

        return td.MicrowaveModeSolverData(
            monitor=monitor,
            symmetry=(0, 0, 0),
            symmetry_center=self.simulation.center,
            grid_expanded=grid,
            n_complex=index_data,
            transmission_line_data=tl_data,
            **field_cmps,
        )

    def make_eps_data(self, monitor: td.PermittivityMonitor) -> td.PermittivityData:
        field_mnt = td.FieldMonitor(**monitor.model_dump(exclude={TYPE_TAG_STR, "fields"}))
        field_data = self.make_field_data(field_mnt)

        def permittivity_from_field(
            field_component: td.ScalarFieldDataArray,
        ) -> td.ScalarFieldDataArray:
            intensity = np.abs(field_component) ** 2
            intensity_min = intensity.min()
            intensity_max = intensity.max()
            intensity_range = intensity_max - intensity_min
            if float(intensity_range) == 0.0:
                intensity_norm = td.ScalarFieldDataArray(
                    np.zeros_like(intensity.values), coords=intensity.coords
                )
            else:
                intensity_norm = (intensity - intensity_min) / intensity_range
            return 1.5**2 + intensity_norm * (3.5**2 - 1.5**2)

        return td.PermittivityData(
            monitor=monitor,
            eps_xx=permittivity_from_field(field_data.Ex),
            eps_yy=permittivity_from_field(field_data.Ey),
            eps_zz=permittivity_from_field(field_data.Ez),
            grid_expanded=self.simulation.discretize_monitor(monitor),
        )

    def make_medium_data(self, monitor: td.MediumMonitor) -> td.MediumData:
        field_mnt = td.FieldMonitor(**monitor.model_dump(exclude={TYPE_TAG_STR, "fields"}))
        field_data = self.make_field_data(field_mnt)
        return td.MediumData(
            monitor=monitor,
            eps_xx=field_data.Ex,
            eps_yy=field_data.Ey,
            eps_zz=field_data.Ez,
            mu_xx=field_data.Hx,
            mu_yy=field_data.Hy,
            mu_zz=field_data.Hz,
            grid_expanded=self.simulation.discretize_monitor(monitor),
        )

    def make_diff_data(self, monitor: td.DiffractionMonitor) -> td.DiffractionData:
        axis_names = ("x", "y", "z")
        normal_axis = monitor.normal_axis
        axis_x, axis_y = [axis_names[i] for i in range(3) if i != normal_axis]
        size_x = self.simulation.size[axis_names.index(axis_x)]
        size_y = self.simulation.size[axis_names.index(axis_y)]
        freqs = list(monitor.freqs)
        orders_x = np.array([0], dtype=int) if size_x == 0 else np.arange(-1, 2, dtype=int)
        orders_y = np.array([0], dtype=int) if size_y == 0 else np.arange(-2, 3, dtype=int)

        coords = {"orders_x": orders_x, "orders_y": orders_y, "f": freqs}
        values = self.data_gen_fn((len(orders_x), len(orders_y), len(freqs)))
        data = td.DiffractionDataArray(values, coords=coords)
        field_data = dict.fromkeys(("Er", "Etheta", "Ephi", "Hr", "Htheta", "Hphi"), data)
        return td.DiffractionData(
            monitor=monitor,
            sim_size=(size_x, size_y),
            bloch_vecs=(0.0, 0.0),
            medium=self.simulation.medium,
            **field_data,
        )

    def make_mode_data(self, monitor: td.ModeMonitor) -> td.ModeData:
        index_coords = {
            "f": list(monitor.freqs),
            "mode_index": np.arange(monitor.mode_spec.num_modes),
        }
        n_complex = self._make_data(
            coords=index_coords, data_array_type=td.ModeIndexDataArray, is_complex=True
        )
        coords_amps = {"direction": ["+", "-"]}
        coords_amps.update(index_coords)
        amps = self._make_data(
            coords=coords_amps, data_array_type=td.ModeAmpsDataArray, is_complex=True
        )

        field_cmps = {}
        if monitor.store_fields_direction is not None:
            for field_name in ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]:
                coords = get_spatial_coords_dict(self.simulation, monitor, field_name)
                coords["f"] = list(monitor.freqs)
                coords["mode_index"] = index_coords["mode_index"]
                field_cmps[field_name] = self._make_data(
                    coords=coords, data_array_type=td.ScalarModeFieldDataArray, is_complex=True
                )

        return td.ModeData(
            monitor=monitor,
            n_complex=n_complex,
            amps=amps,
            grid_expanded=self.simulation.discretize_monitor(monitor),
            **field_cmps,
        )

    def make_microwave_mode_data(self, monitor: td.MicrowaveModeMonitor) -> td.MicrowaveModeData:
        index_coords = {
            "f": list(monitor.freqs),
            "mode_index": np.arange(monitor.mode_spec.num_modes),
        }
        n_complex = self._make_data(
            coords=index_coords, data_array_type=td.ModeIndexDataArray, is_complex=True
        )
        coords_amps = {"direction": ["+", "-"]}
        coords_amps.update(index_coords)
        amps = self._make_data(
            coords=coords_amps, data_array_type=td.ModeAmpsDataArray, is_complex=True
        )

        field_cmps = {}
        if monitor.store_fields_direction is not None:
            for field_name in ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]:
                coords = get_spatial_coords_dict(self.simulation, monitor, field_name)
                coords["f"] = list(monitor.freqs)
                coords["mode_index"] = index_coords["mode_index"]
                field_cmps[field_name] = self._make_data(
                    coords=coords, data_array_type=td.ScalarModeFieldDataArray, is_complex=True
                )

        impedance_specs: tuple = monitor.mode_spec._impedance_specs_as_tuple
        if len(impedance_specs) == 1:
            impedance_specs = impedance_specs * monitor.mode_spec.num_modes
        used_mode_inds = [
            mode_index for mode_index, spec in enumerate(impedance_specs) if spec is not None
        ]

        index_coords = {"f": list(monitor.freqs), "mode_index": np.array(used_mode_inds)}
        index_shape = (len(index_coords["f"]), len(index_coords["mode_index"]))
        z0_data = ImpedanceFreqModeDataArray(
            (1000 + 1j) * self.data_gen_fn(index_shape), coords=index_coords
        )
        v_data = VoltageFreqModeDataArray(
            (1000 + 1j) * self.data_gen_fn(index_shape), coords=index_coords
        )
        i_data = CurrentFreqModeDataArray(
            (1000 + 1j) * self.data_gen_fn(index_shape), coords=index_coords
        )
        tl_data = TransmissionLineDataset(Z0=z0_data, voltage_coeffs=v_data, current_coeffs=i_data)

        return td.MicrowaveModeData(
            monitor=monitor,
            n_complex=n_complex,
            amps=amps,
            grid_expanded=self.simulation.discretize_monitor(monitor),
            transmission_line_data=tl_data,
            **field_cmps,
        )

    def make_gaussian_overlap_data(
        self,
        monitor: td.GaussianOverlapMonitor
        | td.AstigmaticGaussianOverlapMonitor
        | td.ThinLensOverlapMonitor,
    ) -> td.FieldOverlapData:
        grid = self.simulation.discretize_monitor(monitor)
        coords_amps = {"direction": ["+", "-"], "f": list(monitor.freqs), "mode_index": [0]}
        amps = self._make_data(
            coords=coords_amps, data_array_type=td.ModeAmpsDataArray, is_complex=True
        )

        field_cmps = {}
        if monitor.store_fields_direction is not None:
            for field_name in ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]:
                coords = get_spatial_coords_dict(self.simulation, monitor, field_name)
                coords["f"] = list(monitor.freqs)
                field_cmps[field_name] = self._make_data(
                    coords=coords, data_array_type=td.ScalarFieldDataArray, is_complex=True
                )

        return td.FieldOverlapData(
            monitor=monitor,
            amps=amps,
            symmetry=(0, 0, 0),
            symmetry_center=self.simulation.center,
            grid_expanded=grid,
            **field_cmps,
        )

    def make_flux_data(self, monitor: td.FluxMonitor) -> td.FluxData:
        coords = {"f": list(monitor.freqs)}
        flux = self._make_data(coords=coords, data_array_type=td.FluxDataArray, is_complex=False)
        return td.FluxData(monitor=monitor, flux=flux)

    def make_directivity_data(self, monitor: td.DirectivityMonitor) -> td.DirectivityData:
        freqs = list(monitor.freqs)
        radii = np.atleast_1d(monitor.proj_distance)
        theta = list(monitor.theta)
        phi = list(monitor.phi)
        flux = self._make_data(
            coords={"f": freqs}, data_array_type=td.FluxDataArray, is_complex=False
        )
        coords = {"r": radii, "theta": theta, "phi": phi, "f": freqs}
        scalar_field = self._make_data(
            coords=coords, data_array_type=td.FieldProjectionAngleDataArray, is_complex=True
        )
        return td.DirectivityData(
            monitor=monitor,
            flux=flux,
            Er=scalar_field,
            Etheta=scalar_field,
            Ephi=scalar_field,
            Hr=scalar_field,
            Htheta=scalar_field,
            Hphi=scalar_field,
            projection_surfaces=monitor.projection_surfaces,
        )

    def make_field_projection_angle_data(
        self, monitor: td.FieldProjectionAngleMonitor
    ) -> td.FieldProjectionAngleData:
        freqs = list(monitor.freqs)
        radii = np.atleast_1d(getattr(monitor, "proj_distance", 1.0))
        coords = {"r": radii, "theta": list(monitor.theta), "phi": list(monitor.phi), "f": freqs}
        scalar_field = self._make_data(
            coords=coords, data_array_type=td.FieldProjectionAngleDataArray, is_complex=True
        )
        return td.FieldProjectionAngleData(
            monitor=monitor,
            Er=scalar_field,
            Etheta=scalar_field,
            Ephi=scalar_field,
            Hr=scalar_field,
            Htheta=scalar_field,
            Hphi=scalar_field,
            projection_surfaces=monitor.projection_surfaces,
        )

    def make_field_projection_cartesian_data(
        self, monitor: td.FieldProjectionCartesianMonitor
    ) -> td.FieldProjectionCartesianData:
        freqs = list(monitor.freqs)
        proj_distance = getattr(monitor, "proj_distance", 1.0)
        x_plane = list(monitor.x)
        y_plane = list(monitor.y)

        if monitor.proj_axis == 0:
            coords = {"x": np.atleast_1d(proj_distance), "y": x_plane, "z": y_plane, "f": freqs}
        elif monitor.proj_axis == 1:
            coords = {"x": x_plane, "y": np.atleast_1d(proj_distance), "z": y_plane, "f": freqs}
        else:
            coords = {"x": x_plane, "y": y_plane, "z": np.atleast_1d(proj_distance), "f": freqs}

        scalar_field = self._make_data(
            coords=coords, data_array_type=td.FieldProjectionCartesianDataArray, is_complex=True
        )
        return td.FieldProjectionCartesianData(
            monitor=monitor,
            Er=scalar_field,
            Etheta=scalar_field,
            Ephi=scalar_field,
            Hr=scalar_field,
            Htheta=scalar_field,
            Hphi=scalar_field,
            projection_surfaces=monitor.projection_surfaces,
        )

    def make_field_projection_kspace_data(
        self, monitor: td.FieldProjectionKSpaceMonitor
    ) -> td.FieldProjectionKSpaceData:
        freqs = list(monitor.freqs)
        radii = np.atleast_1d(getattr(monitor, "proj_distance", 1.0))
        coords = {"ux": list(monitor.ux), "uy": list(monitor.uy), "r": radii, "f": freqs}
        scalar_field = self._make_data(
            coords=coords, data_array_type=td.FieldProjectionKSpaceDataArray, is_complex=True
        )
        return td.FieldProjectionKSpaceData(
            monitor=monitor,
            Er=scalar_field,
            Etheta=scalar_field,
            Ephi=scalar_field,
            Hr=scalar_field,
            Htheta=scalar_field,
            Hphi=scalar_field,
            projection_surfaces=monitor.projection_surfaces,
        )

    def make_aux_field_time_data(self, monitor: td.AuxFieldTimeMonitor) -> td.AuxFieldTimeData:
        field_cmps = {}
        grid = self.simulation.discretize_monitor(monitor)
        tmesh = self.simulation.tmesh
        for field_name in monitor.fields:
            coords = get_spatial_coords_dict(self.simulation, monitor, field_name)
            idx_begin, idx_end = monitor.time_inds(tmesh)
            coords["t"] = tmesh[idx_begin:idx_end]
            field_cmps[field_name] = self._make_data(
                coords=coords, data_array_type=td.ScalarFieldTimeDataArray, is_complex=False
            )

        return td.AuxFieldTimeData(
            monitor=monitor,
            symmetry=(0, 0, 0),
            symmetry_center=self.simulation.center,
            grid_expanded=grid,
            **field_cmps,
        )

    def make_flux_time_data(self, monitor: td.FluxTimeMonitor) -> td.FluxTimeData:
        flux = self._make_data(
            coords={"t": [0, 1, 2]}, data_array_type=td.FluxTimeDataArray, is_complex=False
        )
        return td.FluxTimeData(monitor=monitor, flux=flux)

    def make_monitor_data(self, monitor: td.Monitor) -> MonitorDataTypes:
        monitor_maker_map = {
            td.FieldMonitor: self.make_field_data,
            td.PointCloudFieldMonitor: self.make_point_cloud_field_data,
            td.PointCloudPermittivityMonitor: self.make_point_cloud_permittivity_data,
            td.FieldTimeMonitor: self.make_field_time_data,
            td.ModeSolverMonitor: self.make_mode_solver_data,
            td.MicrowaveModeSolverMonitor: self.make_microwave_mode_solver_data,
            td.ModeMonitor: self.make_mode_data,
            td.MicrowaveModeMonitor: self.make_microwave_mode_data,
            td.PermittivityMonitor: self.make_eps_data,
            td.MediumMonitor: self.make_medium_data,
            td.DiffractionMonitor: self.make_diff_data,
            td.FluxMonitor: self.make_flux_data,
            td.DirectivityMonitor: self.make_directivity_data,
            td.FieldProjectionAngleMonitor: self.make_field_projection_angle_data,
            td.FieldProjectionCartesianMonitor: self.make_field_projection_cartesian_data,
            td.FieldProjectionKSpaceMonitor: self.make_field_projection_kspace_data,
            td.AuxFieldTimeMonitor: self.make_aux_field_time_data,
            td.FluxTimeMonitor: self.make_flux_time_data,
            td.GaussianOverlapMonitor: self.make_gaussian_overlap_data,
            td.AstigmaticGaussianOverlapMonitor: self.make_gaussian_overlap_data,
            td.ThinLensOverlapMonitor: self.make_gaussian_overlap_data,
        }
        return monitor_maker_map[type(monitor)](monitor)

    def make_simulation_data(self, path: str | Path | None = None) -> td.SimulationData:
        data = tuple(self.make_monitor_data(monitor) for monitor in self.simulation.monitors)
        sim_data = td.SimulationData(simulation=self.simulation, data=data)
        if path is not None:
            sim_data.to_file(str(path))
        return sim_data


def make_monitor_data(
    simulation: td.Simulation,
    monitor: td.Monitor,
    *,
    x0: complex | float = 1.0,
    data_gen_fn: Callable[[tuple[int, ...]], np.ndarray] = np.random.random,
) -> MonitorDataTypes:
    """Generate synthetic data for a single monitor."""

    return SyntheticMonitorDataFactory(
        simulation, x0=x0, data_gen_fn=data_gen_fn
    ).make_monitor_data(monitor)


def make_simulation_data(
    simulation: td.Simulation,
    path: str | Path | None = None,
    *,
    x0: complex | float = 1.0,
    data_gen_fn: Callable[[tuple[int, ...]], np.ndarray] = np.random.random,
) -> td.SimulationData:
    """Generate synthetic ``SimulationData`` for a simulation."""

    return SyntheticMonitorDataFactory(
        simulation, x0=x0, data_gen_fn=data_gen_fn
    ).make_simulation_data(path=path)


__all__ = [
    "SyntheticMonitorDataFactory",
    "get_spatial_coords_dict",
    "make_monitor_data",
    "make_simulation_data",
]
