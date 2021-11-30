# pylint: disable=too-many-locals
""" Tools to switch between new and old Tidy3D formats """
from typing import Dict, Tuple, List, Union
import numpy as np
import h5py

from tidy3d import Simulation, SimulationData, DATA_TYPE_MAP
from tidy3d import Box, Sphere, Cylinder, PolySlab
from tidy3d import Medium, AnisotropicMedium
from tidy3d.components.medium import DispersiveMedium, PECMedium
from tidy3d import VolumeSource, ModeSource, PlaneWave, GaussianBeam
from tidy3d import GaussianPulse
from tidy3d import PML, Absorber, StablePML
from tidy3d import FieldMonitor, FieldTimeMonitor, FluxMonitor, FluxTimeMonitor
from tidy3d.components.monitor import AbstractFieldMonitor, AbstractFluxMonitor, ModeMonitor
from tidy3d.components.monitor import FreqMonitor, TimeMonitor
from tidy3d.components.types import Numpy
from tidy3d.components.data import FieldData, FieldTimeData

# maps monitor name to dictionary mapping data label to data value
MonitorDataDict = Dict[str, Union[Numpy, Dict[str, Numpy]]]
SolverDataDict = Dict[str, MonitorDataDict]


def old_json_parameters(sim: Simulation) -> Dict:
    """Convert simulation parameters to a dict."""

    cent = sim.center
    size = sim.size

    pml_layers = []
    for pml in sim.pml_layers:
        profile = "standard"
        if not pml:
            pml_layers.append({"profile": "standard", "Nlayers": 0})
            continue
        if isinstance(pml, StablePML):
            profile = "standard"
        elif isinstance(pml, Absorber):
            profile = "absorber"
        elif isinstance(pml, PML):
            profile = "standard"
        pml_layers.append({"profile": profile, "Nlayers": pml.num_layers})

    sizes = sim.grid.sizes
    mesh_step_x = np.mean(sizes.x)
    mesh_step_y = np.mean(sizes.y)
    mesh_step_z = np.mean(sizes.z)

    parameters = {
        "unit_length": "um",
        "unit_frequency": "THz",
        "unit_time": "ps",
        "x_cent": cent[0],
        "y_cent": cent[1],
        "z_cent": cent[2],
        "x_span": size[0],
        "y_span": size[1],
        "z_span": size[2],
        "mesh_step": (mesh_step_x, mesh_step_y, mesh_step_z),
        "symmetries": sim.symmetry,
        "pml_layers": pml_layers,
        "run_time": sim.run_time * 1e12,
        "courant": sim.courant,
        "shutoff": sim.shutoff,
        "subpixel": sim.subpixel,
        "time_steps": 100,
        "nodes": 100,
        "compute_weight": 1.0,
    }

    """ TODO: Support nonuniform coordinates """
    # if sim.coords is not None:
    #     parameters.update({"coords": [c.tolist() for c in sim.coords]})

    return parameters


def old_json_structures(sim: Simulation) -> Tuple[List[Dict], List[Dict]]:
    """Convert all Structure objects to a list of text-defined geometries, and all the corresponding
    Medium objects to a list of text-defined materials.
    """

    medium_list = []
    medium_map = sim.medium_map
    for (medium, imed) in medium_map.items():
        """TODO: support custom material names (currently no names at all?)"""
        med = {"name": f"mat_{imed}"}
        if isinstance(medium, Medium):
            """TODO: support diagonal anisotropy in non-dispersive media"""
            med.update(
                {
                    "type": "Medium",
                    "permittivity": [medium.permittivity] * 3,
                    "conductivity": [medium.conductivity] * 3,
                    "poles": [],
                }
            )

        elif isinstance(medium, AnisotropicMedium):
            """TODO: support diagonal anisotropy in non-dispersive media"""
            med.update(
                {
                    "type": "Medium",
                    "permittivity": [
                        medium.xx.permittivity,
                        medium.yy.permittivity,
                        medium.zz.permittivity,
                    ],
                    "conductivity": [
                        medium.xx.conductivity,
                        medium.yy.conductivity,
                        medium.zz.conductivity,
                    ],
                    "poles": [],
                }
            )
        elif isinstance(medium, DispersiveMedium):
            poles = []
            for (a, c) in medium.pole_residue.poles:
                poles.append([a.real, a.imag, c.real, c.imag])
            med.update(
                {
                    "type": "PoleResidue",
                    "permittivity": [medium.pole_residue.eps_inf] * 3,
                    "conductivity": [0, 0, 0],
                    "poles": poles,
                }
            )
        elif isinstance(medium, PECMedium):
            med.update({"type": "PEC"})
        medium_list.append(med)

    struct_list = []
    for structure in sim.structures:
        struct = {"name": structure.medium.name, "mat_index": medium_map[structure.medium]}
        geom = structure.geometry
        if isinstance(geom, Box):
            cent, size = geom.center, geom.size
            struct.update(
                {
                    "type": "Box",
                    "x_cent": cent[0],
                    "y_cent": cent[1],
                    "z_cent": cent[2],
                    "x_span": size[0],
                    "y_span": size[1],
                    "z_span": size[2],
                }
            )
            struct_list.append(struct)
        elif isinstance(geom, Sphere):
            struct.update(
                {
                    "type": "Sphere",
                    "x_cent": geom.center[0],
                    "y_cent": geom.center[1],
                    "z_cent": geom.center[2],
                    "radius": geom.radius,
                }
            )
            struct_list.append(struct)
        elif isinstance(geom, Cylinder):
            struct.update(
                {
                    "type": "Cylinder",
                    "x_cent": geom.center[0],
                    "y_cent": geom.center[1],
                    "z_cent": geom.center[2],
                    "axis": ["x", "y", "z"][geom.axis],
                    "radius": geom.radius,
                    "height": geom.length,
                }
            )
            struct_list.append(struct)
        elif isinstance(geom, PolySlab):
            struct.update(
                {
                    "type": "PolySlab",
                    "vertices": geom.vertices,
                    "z_cent": (geom.slab_bounds[0] + geom.slab_bounds[1]) / 2,
                    "z_size": geom.slab_bounds[1] - geom.slab_bounds[0],
                    # "slant_angle": geom.sidewall_angle_rad,
                    # "dilation": geom.dilation,
                }
            )
            struct_list.append(struct)

        """ TODO: Support GdsSlab """
        # if isinstance(geom, GdsSlab):
        #     for ip, poly_slab in enumerate(geom.poly_slabs):
        #         poly = struct.copy()
        #         if poly["name"] is None:
        #             poly["name"] = "struct%04d_poly%04d" % (istruct, ip)
        #         else:
        #             poly["name"] += "_poly%04d" % ip
        #         poly.update(
        #             {
        #                 "type": "PolySlab",
        #                 "vertices": poly_slab.vertices.tolist(),
        #                 "z_cent": float(poly_slab.z_cent),
        #                 "z_size": float(poly_slab.z_size),
        #             }
        #         )
        #         struct_list.append(poly)

    return medium_list, struct_list


def old_json_sources(sim: Simulation) -> List[Dict]:
    """Export all sources in the Simulation."""

    src_list = []
    for source in sim.sources:
        # Get source_time
        name = source.name

        if isinstance(source.source_time, GaussianPulse):
            src_time = {
                "type": "GaussianPulse",
                "frequency": source.source_time.freq0 * 1e-12,
                "fwidth": source.source_time.fwidth * 1e-12,
                "offset": source.source_time.offset,
                "phase": source.source_time.phase,
            }

        src = {}

        if isinstance(source, VolumeSource):
            """TODO: Is polarization the right word if we're talking about J and M?
            Check Lumerical notation."""
            component = "E" if source.polarization[0] == "J" else "H"
            component += source.polarization[1]
            if all(s == 0 for s in source.size):
                src = {
                    "name": name,
                    "type": "PointDipole",
                    "source_time": src_time,
                    "center": source.center,
                    "component": component,
                    "amplitude": source.source_time.amplitude,
                }
            else:
                src = {
                    "name": name,
                    "type": "VolumeSource",
                    "source_time": src_time,
                    "center": source.center,
                    "size": source.size,
                    "component": component,
                    "amplitude": source.source_time.amplitude,
                }
        elif isinstance(source, ModeSource):
            mode_ind = source.mode.mode_index
            num_modes = source.mode.num_modes if source.mode.num_modes else mode_ind + 1
            direction = "forward" if source.direction == "+" else "backward"
            src = {
                "name": name,
                "type": "ModeSource",
                "source_time": src_time,
                "center": list(source.center),
                "size": list(source.size),
                "direction": direction,
                "amplitude": source.source_time.amplitude,
                "mode_ind": mode_ind,
                "target_neff": source.mode.target_neff,
                "Nmodes": num_modes,
            }
        elif isinstance(source, PlaneWave):
            normal_index = [s == 0 for s in source.size].index(True)
            injection_axis = source.direction + "xyz"[normal_index]
            position = source.center[normal_index]
            src = {
                "name": name,
                "type": "PlaneWave",
                "source_time": src_time,
                "injection_axis": injection_axis,
                "position": position,
                "polarization": source.polarization[1],
                "amplitude": source.source_time.amplitude,
            }
        elif isinstance(source, GaussianBeam):
            normal_index = [s == 0 for s in source.size].index(True)
            injection_axis = source.direction + "xyz"[normal_index]
            direction = "forward" if source.direction == "+" else "backward"
            src = {
                "name": name,
                "type": "GaussianBeam",
                "source_time": src_time,
                "center": list(source.center),
                "normal": "xyz"[normal_index],
                "direction": direction,
                "angle_theta": float(source.angle_theta),
                "angle_phi": float(source.angle_phi),
                "waist_radius": float(source.waist_radius),
                "waist_distance": float(source.waist_distance),
                "pol_angle": float(source.pol_angle),
                "amplitude": source.source_time.amplitude,
            }

        if src:
            src_list.append(src)

    return src_list


def old_json_monitors(sim: Simulation) -> Dict:
    """Export all monitors in the Simulation.

    TODO: All monitors in the old solver interpolate spatially to the center of the Yee cells by
    default. FreqMonitors can be asked to return the fields at the Yee grid locations instead.
    Going forward we should instead always return the raw data from the solver, at sufficiently many
    locations to be able to do any interpolation (user-side) within the monitor.
    """

    mnt_list = []
    for monitor in sim.monitors:
        mnt = {
            "name": monitor.name,
            "x_cent": monitor.center[0],
            "y_cent": monitor.center[1],
            "z_cent": monitor.center[2],
            "x_span": monitor.size[0],
            "y_span": monitor.size[1],
            "z_span": monitor.size[2],
        }
        """ TODO: Time monitors in the revamp work with a TimeSampler that's a sequence of ints,
        presumably the indexes of the discrete time points at which to record. This is currently
        not supported by the solver, the most freedom there is to record every N steps. Also from a
        user perspective, the user will rarely want to specifically give the indexes of the
        time points at which to sample. Much more common would be to specify things in units of
        simulation time. We need some convenience function(s).

        For that, and more generally, it seems that currently the Simulation doesn't have a
        TimeGrid or somethng like it? This is also needed e.g. to plot the source dependence, etc.
        """
        if isinstance(monitor, FreqMonitor):
            mnt.update({"frequency": [f * 1e-12 for f in monitor.freqs]})
        elif isinstance(monitor, TimeMonitor):
            # handle case where stop is None
            stop = monitor.stop if monitor.stop else sim.run_time
            # handle case where stop > sim.run_time
            stop = min(stop, sim.run_time)

            mnt.update(
                {
                    "t_start": monitor.start,
                    "t_stop": stop,
                    "t_step": sim.dt * monitor.interval,
                }
            )

        if isinstance(monitor, AbstractFieldMonitor):
            store = []
            if np.any([field[0] == "E" for field in monitor.fields]):
                store.append("E")
            if np.any([field[0] == "H" for field in monitor.fields]):
                store.append("H")

            if isinstance(monitor, FieldMonitor):
                mnt.update(
                    {
                        "type": "FrequencyMonitor",
                        "store": store,
                        "interpolate": False,
                    }
                )
            elif isinstance(monitor, FieldTimeMonitor):
                mnt.update(
                    {
                        "type": "TimeMonitor",
                        "store": store,
                    }
                )
        elif isinstance(monitor, AbstractFluxMonitor):
            if isinstance(monitor, FluxTimeMonitor):
                mnt.update(
                    {
                        "type": "TimeMonitor",
                        "store": ["flux"],
                    }
                )
            elif isinstance(monitor, FluxMonitor):
                mnt.update(
                    {
                        "type": "FrequencyMonitor",
                        "store": ["flux"],
                        "interpolate": True,
                    }
                )
        elif isinstance(monitor, ModeMonitor):
            num_modes = max([m.mode_index for m in monitor.modes]) + 1
            mnt.update(
                {
                    "type": "ModeMonitor",
                    "Nmodes": num_modes,
                    "target_neff": None,
                    "store": ["mode_amps"],
                }
            )

        """ TODO: Support PermittivityMonitor """
        #

        mnt_list.append(mnt)

    return mnt_list


def export_old_json(sim: Simulation) -> Dict:
    """Write a Simulation in the old Tidy3D json format."""

    sim_dict = {}
    sim_dict["parameters"] = old_json_parameters(sim)
    medium_list, struct_list = old_json_structures(sim)
    sim_dict["materials"] = medium_list
    sim_dict["structures"] = struct_list
    sim_dict["sources"] = old_json_sources(sim)
    sim_dict["monitors"] = old_json_monitors(sim)

    return sim_dict


""" Discretize monitors with values """


def discretize_monitor(values, mon: AbstractFieldMonitor):
    """Discretizes spatial extent of a monitor"""
    Nx, Ny, Nz = values.shape[:3]
    center = mon.center
    size = mon.size
    x, y, z = make_coordinates_3d(center, size, (Nx, Ny, Nz))
    return x, y, z


def make_coordinates_1d(cmin: float, cmax: float, num: int):
    """returns an endpoint-inclusive array of points between cmin and cmax with spacing dl"""
    return np.linspace(cmin, cmax, num)


def make_coordinates_3d(center, size, shape):
    """gets the x,y,z coordinates of a box with center, size and discretization of grid_size"""
    rmin = np.array([c - s / 2.0 for (c, s) in zip(center, size)])
    rmax = np.array([c + s / 2.0 for (c, s) in zip(center, size)])
    x, y, z = [make_coordinates_1d(cmin, cmax, num) for (cmin, cmax, num) in zip(rmin, rmax, shape)]
    return x, y, z


def load_solver_results(
    simulation: Simulation,
    solver_data_dict: SolverDataDict,
    log_string: str = None,
) -> SimulationData:
    """load the solver_data_dict and simulation into SimulationData"""

    # constuct monitor_data dictionary
    monitor_data = {}
    for monitor in simulation.monitors:
        name = monitor.name
        monitor_data_dict = solver_data_dict[name]
        monitor_data_type = DATA_TYPE_MAP[monitor.data_type]
        if monitor.type in ("FieldMonitor", "FieldTimeMonitor"):
            field_data = {}
            for field_name, data_dict in monitor_data_dict.items():
                field_data[field_name] = monitor_data_type(**data_dict)
            field_data_type = FieldData if monitor.type == "FieldMonitor" else FieldTimeData
            monitor_data[name] = field_data_type(data_dict=field_data)
        else:
            monitor_data[name] = monitor_data_type(**monitor_data_dict)
    return SimulationData(simulation=simulation, monitor_data=monitor_data, log_string=log_string)


def load_old_monitor_data(simulation: Simulation, data_file: str) -> SolverDataDict:
    """Load a monitor data file from the old Tidy3D format as a dict that can
    be passed to SimulationData.
    """

    data_dict = {}
    with h5py.File(data_file, "r") as f_handle:
        for monitor in simulation.monitors:
            name = monitor.name
            data_group = f_handle[name]

            if isinstance(monitor, FreqMonitor):
                sampler_values = np.array(monitor.freqs)
                sampler_label = "f"

            elif isinstance(monitor, TimeMonitor):
                sampler_values = np.array(data_group["tmesh"]).ravel()
                sampler_label = "t"

            if isinstance(monitor, AbstractFieldMonitor):
                data_dict[name] = {}
                for field_name in monitor.fields:
                    comp = ["x", "y", "z"].index(field_name[1])
                    field_vals = np.array(f_handle[name][field_name[0]][comp, ...])
                    # x, y, z = discretize_monitor(field_vals, monitor)
                    subgrid = simulation.discretize(monitor.geometry)
                    yee_locs = subgrid[field_name]
                    data_dict[name][field_name] = {
                        "x": yee_locs.x,
                        "y": yee_locs.y,
                        "z": yee_locs.z,
                        "values": field_vals,
                        sampler_label: sampler_values,
                    }

            elif isinstance(monitor, AbstractFluxMonitor):
                values = np.array(data_group["flux"]).ravel()
                data_dict[name] = {"values": values, sampler_label: sampler_values}

            elif isinstance(monitor, ModeMonitor):
                values = np.array(data_group["mode_amps"])
                values = np.swapaxes(values, 1, 2)  # put frequency at last axis
                mode_index = np.arange(values.shape[1])
                data_dict[name] = {
                    "values": values,
                    "direction": ["+", "-"],
                    "mode_index": mode_index,
                    sampler_label: sampler_values,
                }

    return data_dict
