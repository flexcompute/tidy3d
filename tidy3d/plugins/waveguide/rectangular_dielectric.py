"""Rectangular dielectric waveguide utilities."""

from typing import List, Any

import numpy
import pydantic

from ...components.base import Tidy3dBaseModel, cached_property
from ...components.boundary import BoundarySpec, Periodic
from ...components.data.data_array import ModeIndexDataArray, FreqModeDataArray
from ...components.geometry import Box, PolySlab
from ...components.grid.grid_spec import GridSpec
from ...components.medium import Medium, MediumType
from ...components.mode import ModeSpec
from ...components.simulation import Simulation

from ...components.source import ModeSource, GaussianPulse
from ...components.structure import Structure
from ...components.types import ArrayFloat1D, Ax, Axis, Coordinate, Literal, Size1D, Union
from ...components.types import TYPE_TAG_STR
from ...constants import C_0, inf, MICROMETER, RADIAN
from ...exceptions import Tidy3dError, ValidationError
from ...log import log

from ..mode.mode_solver import ModeSolver


class RectangularDielectric(Tidy3dBaseModel):
    """General rectangular dielectric waveguide

    Supports:
    - Strip and rib geometries
    - Angled sidewalls
    - Modes in waveguide bends
    - Surface and sidewall loss models
    - Coupled waveguides
    """

    wavelength: Union[float, ArrayFloat1D] = pydantic.Field(
        ...,
        title="Wavelength",
        description="Wavelength(s) at which to calculate modes (in μm).",
        units=MICROMETER,
    )

    core_width: Union[Size1D, ArrayFloat1D] = pydantic.Field(
        ...,
        title="Core width",
        description="Core width at the top of the waveguide.  If set to an array, defines "
        "the widths of adjacent waveguides.",
        units=MICROMETER,
    )

    core_thickness: Size1D = pydantic.Field(
        ...,
        title="Core Thickness",
        description="Thickness of the core layer.",
        units=MICROMETER,
    )

    core_medium: MediumType = pydantic.Field(
        ...,
        title="Core Medium",
        description="Medium associated with the core layer.",
        discriminator=TYPE_TAG_STR,
    )

    clad_medium: MediumType = pydantic.Field(
        ...,
        title="Clad Medium",
        description="Medium associated with the upper cladding layer.",
        discriminator=TYPE_TAG_STR,
    )

    box_medium: MediumType = pydantic.Field(
        None,
        title="Box Medium",
        description="Medium associated with the lower cladding layer.",
        discriminator=TYPE_TAG_STR,
    )

    slab_thickness: Size1D = pydantic.Field(
        0.0,
        title="Slab Thickness",
        description="Thickness of the slab for rib geometry.",
        units=MICROMETER,
    )

    clad_thickness: Size1D = pydantic.Field(
        None,
        title="Clad Thickness",
        description="Domain size above the core layer.",
        units=MICROMETER,
    )

    box_thickness: Size1D = pydantic.Field(
        None,
        title="Box Thickness",
        description="Domain size below the core layer.",
        units=MICROMETER,
    )

    side_margin: Size1D = pydantic.Field(
        None,
        title="Side Margin",
        description="Domain size to the sides of the waveguide core.",
        units=MICROMETER,
    )

    sidewall_angle: float = pydantic.Field(
        0.0,
        title="Sidewall Angle",
        description="Angle of the core sidewalls measured from the vertical direction (in "
        "radians).  Positive (negative) values create waveguides with bases wider (narrower) "
        "than their tops.",
        units=RADIAN,
    )

    gap: Union[float, ArrayFloat1D] = pydantic.Field(
        0.0,
        title="Gap",
        description="Distance between adjacent waveguides, measured at the top core edges.  "
        "An array can be used to define one gap per pair of adjacent waveguides.",
        units=MICROMETER,
    )

    sidewall_thickness: Size1D = pydantic.Field(
        0.0,
        title="Sidewall Thickness",
        description="Sidewall layer thickness (within core).",
        units=MICROMETER,
    )

    sidewall_medium: MediumType = pydantic.Field(
        None,
        title="Sidewall medium",
        description="Medium associated with the sidewall layer to model sidewall losses.",
        discriminator=TYPE_TAG_STR,
    )

    surface_thickness: Size1D = pydantic.Field(
        0.0,
        title="Surface Thickness",
        description="Thickness of the surface layers defined on the top of the waveguide and  "
        "slab regions (if any).",
        units=MICROMETER,
    )

    surface_medium: MediumType = pydantic.Field(
        None,
        title="Surface Medium",
        description="Medium associated with the surface layer to model surface losses.",
        discriminator=TYPE_TAG_STR,
    )

    origin: Coordinate = pydantic.Field(
        (0, 0, 0),
        title="Origin",
        description="Center of the waveguide geometry.  This coordinate represents the base "
        "of the waveguides (substrate surface) in the normal axis, and center of the geometry "
        "in the remaining axes.",
        units=MICROMETER,
    )

    length: Size1D = pydantic.Field(
        1e30,
        title="Length",
        description="Length of the waveguides in the propagation direction",
        units=MICROMETER,
    )

    propagation_axis: Axis = pydantic.Field(
        0,
        title="Propagation Axis",
        description="Axis of propagation of the waveguide",
    )

    normal_axis: Axis = pydantic.Field(
        2,
        title="Normal Axis",
        description="Axis normal to the substrate surface",
    )

    mode_spec: ModeSpec = pydantic.Field(
        ModeSpec(num_modes=2),
        title="Mode Specification",
        description=":class:`ModeSpec` defining waveguide mode properties.",
    )

    grid_resolution: int = pydantic.Field(
        15,
        title="Grid Resolution",
        description="Solver grid resolution per wavelength.",
    )

    max_grid_scaling: float = pydantic.Field(
        1.2,
        title="Maximal Grid Scaling",
        description="Maximal size increase between adjacent grid boundaries.",
    )

    @pydantic.validator("wavelength", "core_width", "gap", always=True)
    def _set_array(cls, val):
        """Ensure values are not negative and convert to numpy arrays."""
        result = numpy.array(val, ndmin=1)
        if any(result < 0):
            raise ValidationError("Values may not be negative.")
        return result

    @pydantic.validator("box_medium", always=True)
    def _set_box_medium(cls, val, values):
        """Set BOX medium same as cladding as default value."""
        return values["clad_medium"] if val is None else val

    @pydantic.validator("clad_thickness", always=True)
    def _set_clad_thickness(cls, val, values):
        """Set default cladding thickness based on the max wavelength in the cladding medium."""
        if val is None:
            wavelength = values["wavelength"]
            medium = values["clad_medium"]
            n = numpy.array([medium.nk_model(f)[0] for f in C_0 / wavelength])
            lda = wavelength / n
            return 1.5 * lda.max()
        return val

    @pydantic.validator("box_thickness", always=True)
    def _set_box_thickness(cls, val, values):
        """Set default BOX thickness based on the max wavelength in the BOX medium."""
        if val is None:
            wavelength = values["wavelength"]
            medium = values["box_medium"]
            n = numpy.array([medium.nk_model(f)[0] for f in C_0 / wavelength])
            lda = wavelength / n
            return 1.5 * lda.max()
        return val

    @pydantic.validator("side_margin", always=True)
    def _set_side_margin(cls, val, values):
        """Set default side margin based on BOX and cladding thicknesses."""
        return max(values["clad_thickness"], values["box_thickness"]) if val is None else val

    @pydantic.validator("gap", always=True)
    def _validate_gaps(cls, val, values):
        """Ensure the number of gaps is compatible with the number of cores supplied."""
        if val.size == 1 and values["core_width"].size != 2:
            # If a single value is defined, use it for all gaps
            return numpy.array([val[0]] * (values["core_width"].size - 1))
        if val.size != values["core_width"].size - 1:
            raise ValidationError("Number of gaps must be 1 less than number of core widths.")
        return val

    @pydantic.root_validator
    def _ensure_consistency(cls, values):
        """Ensure consistency in setting surface/sidewall models and propagation/normal axes."""
        sidewall_thickness = values["sidewall_thickness"]
        sidewall_medium = values["sidewall_medium"]
        surface_thickness = values["surface_thickness"]
        surface_medium = values["surface_medium"]
        propagation_axis = values["propagation_axis"]
        normal_axis = values["normal_axis"]

        if sidewall_thickness > 0 and sidewall_medium is None:
            raise ValidationError(
                "Sidewall medium must be provided when sidewall thickness is greater than 0."
            )

        if sidewall_thickness == 0 and sidewall_medium is not None:
            log.warning("Sidewall medium not used because sidewall thickness is zero.")

        if surface_thickness > 0 and surface_medium is None:
            raise ValidationError(
                "Surface medium must be provided when surface thickness is greater than 0."
            )

        if surface_thickness == 0 and surface_medium is not None:
            log.warning("Surface medium not used because surface thickness is zero.")

        if propagation_axis == normal_axis:
            raise ValidationError("Propagation and normal axes must be different.")

        return values

    @cached_property
    def lateral_axis(self) -> Axis:
        """Lateral direction axis."""
        return 3 - self.propagation_axis - self.normal_axis

    def _swap_axis(
        self, lateral_coord: Any, normal_coord: Any, propagation_coord: Any
    ) -> List[Any]:
        """Swap the model coordinates to desired axes."""
        result = [None, None, None]
        result[self.lateral_axis] = lateral_coord
        result[self.propagation_axis] = propagation_coord
        result[self.normal_axis] = normal_coord
        return result

    def _translate(
        self, lateral_coord: float, normal_coord: float, propagation_coord: float
    ) -> List[float]:
        """Swap the model coordinates to desired axes and translate to origin."""
        coordinates = self._swap_axis(lateral_coord, normal_coord, propagation_coord)
        result = [a + b for a, b in zip(self.origin, coordinates)]
        return result

    def _transform_in_plane(self, lateral_coord: float, propagation_coord: float) -> List[float]:
        """Swap the model coordinates to desired axes in the substrate plane."""
        result = self._translate(lateral_coord, 0, propagation_coord)
        _, result = Box.pop_axis(result, self.normal_axis)
        return result

    @cached_property
    def height(self) -> Size1D:
        """Domain height (size in the normal direction)."""
        return self.box_thickness + self.core_thickness + self.clad_thickness

    @cached_property
    def width(self) -> Size1D:
        """Domain width (size in the lateral direction)."""
        w = self.core_width.sum() + self.gap.sum() + 2 * self.side_margin
        if self.sidewall_angle > 0:
            w += 2 * self.core_thickness * numpy.tan(self.sidewall_angle)
        return w

    @property
    def _core_starts(self) -> List[float]:
        """Starting positions of each waveguide (x is the position in the lateral direction)."""
        core_x = [-0.5 * (self.core_width.sum() + self.gap.sum())]
        core_x.extend(core_x[0] + numpy.cumsum(self.core_width[:-1]) + numpy.cumsum(self.gap))
        return core_x

    # pylint:disable=too-many-locals
    @property
    def _override_structures(self) -> List[Structure]:
        """Build override structures to define the simulation grid."""

        # Grid resolution factor applied to the materials (increase for waveguide corners
        # and decrase for evanescent tail regions).
        scale_factor = 1.5

        freqs = C_0 / self.wavelength
        nk_core = numpy.array([self.core_medium.nk_model(f) for f in freqs])
        nk_clad = numpy.array([self.clad_medium.nk_model(f) for f in freqs])
        nk_box = numpy.array([self.box_medium.nk_model(f) for f in freqs])
        lda_core = self.wavelength / nk_core[:, 0]
        lda_clad = self.wavelength / nk_clad[:, 0]
        lda_box = self.wavelength / nk_box[:, 0]

        core_x = self._core_starts

        i = numpy.argmin(lda_core)
        hi_index = Medium.from_nk(n=nk_core[i, 0] * scale_factor, k=0, freq=freqs[i])

        i = numpy.argmin(lda_box)
        lo_index_box = Medium.from_nk(n=max(1.0, nk_box[i, 0] / scale_factor), k=0, freq=freqs[i])

        i = numpy.argmin(lda_clad)
        lo_index_clad = Medium.from_nk(n=max(1.0, nk_clad[i, 0] / scale_factor), k=0, freq=freqs[i])

        # Gather all waveguide edge intervals into `hi_res` list
        corner_margin = max(lda_box.max(), lda_clad.max()) / self.grid_resolution
        if self.sidewall_angle > 0:
            dx = (self.core_thickness - self.slab_thickness) * numpy.tan(self.sidewall_angle)
            hi_res = [
                pair
                for x, w in zip(core_x, self.core_width)
                for pair in [
                    [x - dx - corner_margin, x + corner_margin],
                    [x + w - corner_margin, x + w + dx + corner_margin],
                ]
            ]
        elif self.sidewall_angle < 0:
            dx = (self.core_thickness - self.slab_thickness) * numpy.tan(self.sidewall_angle)
            hi_res = [
                pair
                for x, w in zip(core_x, self.core_width)
                for pair in [
                    [x - corner_margin, x - dx + corner_margin],
                    [x + w + dx - corner_margin, x + w + corner_margin],
                ]
            ]
        else:
            hi_res = [
                pair
                for x, w in zip(core_x, self.core_width)
                for pair in [
                    [x - corner_margin, x + corner_margin],
                    [x + w - corner_margin, x + w + corner_margin],
                ]
            ]

        # The gaps between waveguides can be small enough to merge adjacent high
        # resolution intervals (specially with angled sidewalls), so we merge
        # intervals that overlap
        i = 0
        while i < len(hi_res) - 1:
            if hi_res[i][1] >= hi_res[i + 1][0]:
                hi_res[i][1] = hi_res.pop(i + 1)[1]
            else:
                i += 1

        # Create override structures to improve the mode solver grid.  We want
        # high resolution around all core edges, but not along the whole slab
        # in case of rib geometry.  Start with the 4 corners:
        override_structures = [
            Structure(
                geometry=Box(
                    center=self._translate(0.5 * (a + b), y, 0),
                    size=self._swap_axis(b - a, 2 * corner_margin, inf),
                ),
                medium=hi_index,
            )
            for (a, b) in hi_res
            for y in (self.slab_thickness, self.core_thickness)
        ]
        # Low resolution on the sides:
        override_structures.extend(
            Structure(
                geometry=Box(
                    center=self._translate(0.5 * (a + b), 0, 0),
                    size=self._swap_axis(b - a, inf, inf),
                ),
                medium=lo_index_clad,
            )
            for (a, b) in ((-self.width, hi_res[0][0]), (hi_res[-1][1], self.width))
        )
        # Low resolution above and below:
        override_structures.extend(
            Structure(
                geometry=Box(
                    center=self._translate(0, 0.5 * (a + b), 0),
                    size=self._swap_axis(inf, b - a, inf),
                ),
                medium=lo_index,
            )
            for (a, b, lo_index) in (
                (-2 * self.box_thickness, -corner_margin, lo_index_box),
                (
                    self.core_thickness + corner_margin,
                    self.core_thickness + 2 * self.clad_thickness,
                    lo_index_clad,
                ),
            )
        )

        return override_structures

    @cached_property
    def grid_spec(self) -> GridSpec:
        """Waveguide grid specification with overriding geometry."""
        grid_spec = GridSpec.auto(
            min_steps_per_wvl=self.grid_resolution,
            wavelength=self.wavelength.min(),
            override_structures=self._override_structures,
            max_scale=self.max_grid_scaling,
        )
        return grid_spec

    @cached_property
    def structures(self) -> List[Structure]:
        """Waveguide structures for simulation, including the core(s), slabs (if any), and bottom
        cladding, if different from the top. For bend modes, the structure is a 270 degree bend
        regardless of :attr:`length`."""

        # Create a local copy of these values, as they will be modified
        # according to the desired geometry
        core_w = numpy.array(self.core_width, copy=True)
        core_t = self.core_thickness
        slab_t = self.slab_thickness

        core_x = self._core_starts

        if self.mode_spec.bend_radius is None or self.mode_spec.bend_radius == 0.0:
            half_length = 0.5 * self.length

            def polyslab_vertices(x, w):
                return (
                    self._transform_in_plane(x, -half_length),
                    self._transform_in_plane(x + w, -half_length),
                    self._transform_in_plane(x + w, half_length),
                    self._transform_in_plane(x, half_length),
                )

        else:
            if (self.normal_axis > self.lateral_axis) != (self.mode_spec.bend_axis == 1):
                raise Tidy3dError(
                    "Waveguide band axis must be the substrate normal "
                    f"(mode_spec.bend_axis = {1 - self.mode_spec.bend_axis})"
                )

            bend_radius = self.mode_spec.bend_radius

            # 10 nm resolution (at center)
            num_points = 1 + int(0.5 + 1.5 * numpy.pi * abs(bend_radius) / 0.01)

            angles = numpy.linspace(-0.75 * numpy.pi, 0.75 * numpy.pi, num_points)
            if bend_radius < 0:
                angles = -angles
            sin = numpy.sin(angles)
            cos = numpy.cos(angles)

            def polyslab_vertices(x, w):
                r_in = bend_radius + x
                v_in = numpy.vstack((-bend_radius + r_in * cos, r_in * sin)).T
                r_out = r_in + w
                v_out = numpy.vstack((-bend_radius + r_out * cos, r_out * sin)).T
                return [self._transform_in_plane(*v) for v in list(v_out) + list(v_in[::-1])]

        # Create the actual waveguide geometry
        structures = []

        normal_origin = self.origin[self.normal_axis]

        # Surface and sidewall loss regions are created first, so that the core
        # can be applied on top.
        if self.surface_thickness > 0:
            structures.extend(
                Structure(
                    geometry=PolySlab(
                        vertices=polyslab_vertices(x, w),
                        slab_bounds=(
                            normal_origin + core_t - self.surface_thickness,
                            normal_origin + core_t,
                        ),
                        sidewall_angle=self.sidewall_angle,
                        reference_plane="top",
                        axis=self.normal_axis,
                    ),
                    medium=self.surface_medium,
                )
                for x, w in zip(core_x, core_w)
            )

            # Add loss region over slab surface, if rib geometry
            if slab_t > 0:
                structures.append(
                    Structure(
                        geometry=Box(
                            center=self._translate(0, 0.5 * slab_t, 0),
                            size=self._swap_axis(inf, slab_t, self.length),
                        ),
                        medium=self.surface_medium,
                    )
                )

            # Correct core geometry to leave the lossy regions with their
            # specified thickness
            dx = self.surface_thickness * numpy.tan(self.sidewall_angle)
            core_x -= dx
            core_w += 2 * dx
            core_t -= self.surface_thickness
            slab_t = max(0, slab_t - self.surface_thickness)

        if self.sidewall_thickness > 0:
            structures.extend(
                Structure(
                    geometry=PolySlab(
                        vertices=polyslab_vertices(x, w),
                        slab_bounds=(normal_origin, normal_origin + core_t),
                        sidewall_angle=self.sidewall_angle,
                        reference_plane="top",
                        axis=self.normal_axis,
                    ),
                    medium=self.sidewall_medium,
                )
                for x, w in zip(core_x, core_w)
            )

            # Core position offset and width reduction to accommodate lossy
            # regions
            dx = self.sidewall_thickness / numpy.cos(self.sidewall_angle)
            core_x += dx
            core_w -= 2 * dx

        # Waveguide cores
        structures.extend(
            Structure(
                geometry=PolySlab(
                    vertices=polyslab_vertices(x, w),
                    slab_bounds=(normal_origin, normal_origin + core_t),
                    sidewall_angle=self.sidewall_angle,
                    reference_plane="top",
                    axis=self.normal_axis,
                ),
                medium=self.core_medium,
            )
            for x, w in zip(core_x, core_w)
        )

        # Slab for rib geometry
        if slab_t > 0:
            structures.append(
                Structure(
                    geometry=Box(
                        center=self._translate(0, 0.5 * slab_t, 0),
                        size=self._swap_axis(inf, slab_t, self.length),
                    ),
                    medium=self.core_medium,
                )
            )

        # Lower cladding
        if self.box_medium != self.clad_medium:
            structures.append(
                Structure(
                    geometry=Box(
                        center=self._translate(0, -self.box_thickness, 0),
                        size=self._swap_axis(inf, 2 * self.box_thickness, self.length),
                    ),
                    medium=self.box_medium,
                )
            )

        return structures

    @cached_property
    def mode_solver(self) -> ModeSolver:
        """Create a mode solver based on this waveguide structure

        Returns
        -------
        :class:`ModeSolver`

        Example
        -------
        >>> wg = waveguide.RectangularDielectric(
        ...     wavelength=1.55,
        ...     core_width=0.5,
        ...     core_thickness=0.22,
        ...     core_medium=Medium(permittivity=3.48**2),
        ...     clad_medium=Medium(permittivity=1.45**2),
        ...     num_modes=2,
        ... )
        >>> mode_data = wg.mode_solver.solve()
        >>> mode_data.n_eff.values
        array([[2.4536054 1.7850305]], dtype=float32)

        """
        freqs = C_0 / self.wavelength
        f_max = freqs.max()
        f_min = freqs.min()
        freq0 = 0.5 * (f_max + f_min)
        fwidth = max(0.1 * freq0, f_max - f_min)

        plane = Box(
            center=self._translate(0, 0.5 * self.height - self.box_thickness, 0),
            size=self._swap_axis(self.width, self.height, 0),
        )

        # Source used only to silence warnings
        mode_source = ModeSource(
            center=plane.center,
            size=plane.size,
            source_time=GaussianPulse(freq0=freq0, fwidth=fwidth),
            direction="+",
            mode_spec=self.mode_spec,
        )

        simulation = Simulation(
            center=plane.center,
            size=plane.size,
            medium=self.clad_medium,
            structures=self.structures,
            boundary_spec=BoundarySpec.all_sides(Periodic()),
            grid_spec=self.grid_spec,
            sources=[mode_source],
            run_time=1e-12,
        )

        mode_solver = ModeSolver(
            simulation=simulation,
            plane=plane,
            mode_spec=self.mode_spec,
            freqs=freqs,
        )

        return mode_solver

    @property
    def n_eff(self) -> ModeIndexDataArray:
        """Calculate the effective index."""
        return self.mode_solver.data.n_eff

    @property
    def n_complex(self) -> ModeIndexDataArray:
        """Calculate the complex effective index."""
        return self.mode_solver.data.n_complex

    @property
    def n_group(self) -> ModeIndexDataArray:
        r"""Calculate the group index."""
        return self.mode_solver.data.n_group

    @property
    def mode_area(self) -> FreqModeDataArray:
        """Calculate the effective mode area."""
        return self.mode_solver.data.mode_area

    # plot wrappers

    # pylint:disable=too-many-arguments
    def plot(
        self,
        x: float = None,
        y: float = None,
        z: float = None,
        ax: Ax = None,
        source_alpha: float = None,
        monitor_alpha: float = None,
        **patch_kwargs,
    ) -> Ax:
        """Plot each of simulation's components on a plane defined by one nonzero x,y,z coordinate.

        Parameters
        ----------
        x : float = None
            position of plane in x direction, only one of x, y, z must be specified to define plane.
        y : float = None
            position of plane in y direction, only one of x, y, z must be specified to define plane.
        z : float = None
            position of plane in z direction, only one of x, y, z must be specified to define plane.
        source_alpha : float = None
            Opacity of the sources. If ``None``, uses Tidy3d default.
        monitor_alpha : float = None
            Opacity of the monitors. If ``None``, uses Tidy3d default.
        ax : matplotlib.axes._subplots.Axes = None
            Matplotlib axes to plot on, if not specified, one is created.

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.
        """
        return self.mode_solver.simulation.plot(
            x=x,
            y=y,
            z=z,
            ax=ax,
            source_alpha=source_alpha,
            monitor_alpha=monitor_alpha,
            **patch_kwargs,
        )

    # pylint:disable=too-many-arguments
    def plot_eps(
        self,
        x: float = None,
        y: float = None,
        z: float = None,
        freq: float = None,
        alpha: float = None,
        source_alpha: float = None,
        monitor_alpha: float = None,
        ax: Ax = None,
    ) -> Ax:
        """Plot each of simulation's components on a plane defined by one nonzero x,y,z coordinate.
        The permittivity is plotted in grayscale based on its value at the specified frequency.

        Parameters
        ----------
        x : float = None
            position of plane in x direction, only one of x, y, z must be specified to define plane.
        y : float = None
            position of plane in y direction, only one of x, y, z must be specified to define plane.
        z : float = None
            position of plane in z direction, only one of x, y, z must be specified to define plane.
        freq : float = None
            Frequency to evaluate the relative permittivity of all mediums.
            If not specified, evaluates at infinite frequency.
        alpha : float = None
            Opacity of the structures being plotted.
            Defaults to the structure default alpha.
        source_alpha : float = None
            Opacity of the sources. If ``None``, uses Tidy3d default.
        monitor_alpha : float = None
            Opacity of the monitors. If ``None``, uses Tidy3d default.
        ax : matplotlib.axes._subplots.Axes = None
            Matplotlib axes to plot on, if not specified, one is created.

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.
        """
        return self.mode_solver.simulation.plot_eps(
            x=x,
            y=y,
            z=z,
            freq=freq,
            alpha=alpha,
            source_alpha=source_alpha,
            monitor_alpha=monitor_alpha,
            ax=ax,
        )

    def plot_structures(
        self, x: float = None, y: float = None, z: float = None, ax: Ax = None
    ) -> Ax:
        """Plot each of simulation's structures on a plane defined by one nonzero x,y,z coordinate.

        Parameters
        ----------
        x : float = None
            position of plane in x direction, only one of x, y, z must be specified to define plane.
        y : float = None
            position of plane in y direction, only one of x, y, z must be specified to define plane.
        z : float = None
            position of plane in z direction, only one of x, y, z must be specified to define plane.
        ax : matplotlib.axes._subplots.Axes = None
            Matplotlib axes to plot on, if not specified, one is created.

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.
        """
        return self.mode_solver.simulation.plot_structures(
            x=x,
            y=y,
            z=z,
            ax=ax,
        )

    # pylint:disable=too-many-arguments
    def plot_structures_eps(
        self,
        x: float = None,
        y: float = None,
        z: float = None,
        freq: float = None,
        alpha: float = None,
        cbar: bool = True,
        reverse: bool = False,
        ax: Ax = None,
    ) -> Ax:
        """Plot each of simulation's structures on a plane defined by one nonzero x,y,z coordinate.
        The permittivity is plotted in grayscale based on its value at the specified frequency.

        Parameters
        ----------
        x : float = None
            position of plane in x direction, only one of x, y, z must be specified to define plane.
        y : float = None
            position of plane in y direction, only one of x, y, z must be specified to define plane.
        z : float = None
            position of plane in z direction, only one of x, y, z must be specified to define plane.
        freq : float = None
            Frequency to evaluate the relative permittivity of all mediums.
            If not specified, evaluates at infinite frequency.
        reverse : bool = False
            If ``False``, the highest permittivity is plotted in black.
            If ``True``, it is plotteed in white (suitable for black backgrounds).
        cbar : bool = True
            Whether to plot a colorbar for the relative permittivity.
        alpha : float = None
            Opacity of the structures being plotted.
            Defaults to the structure default alpha.
        ax : matplotlib.axes._subplots.Axes = None
            Matplotlib axes to plot on, if not specified, one is created.

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.
        """
        return self.mode_solver.simulation.plot_structures_eps(
            x=x,
            y=y,
            z=z,
            freq=freq,
            alpha=alpha,
            cbar=cbar,
            reverse=reverse,
            ax=ax,
        )

    def plot_grid(
        self,
        x: float = None,
        y: float = None,
        z: float = None,
        ax: Ax = None,
        **kwargs,
    ) -> Ax:
        """Plot the cell boundaries as lines on a plane defined by one nonzero x,y,z coordinate.

        Parameters
        ----------
        x : float = None
            position of plane in x direction, only one of x, y, z must be specified to define plane.
        y : float = None
            position of plane in y direction, only one of x, y, z must be specified to define plane.
        z : float = None
            position of plane in z direction, only one of x, y, z must be specified to define plane.
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
        return self.mode_solver.simulation.plot_grid(
            x=x,
            y=y,
            z=z,
            ax=ax,
            **kwargs,
        )

    # pylint:disable=too-many-arguments
    def plot_field(
        self,
        field_name: str,
        val: Literal["real", "imag", "abs"] = "real",
        eps_alpha: float = 0.2,
        robust: bool = True,
        vmin: float = None,
        vmax: float = None,
        ax: Ax = None,
        **sel_kwargs,
    ) -> Ax:
        """Plot the field for a :class:`.ModeSolverData` with :class:`.Simulation` plot overlayed.

        Parameters
        ----------
        field_name : str
            Name of `field` component to plot (eg. `'Ex'`).
            Also accepts `'E'` and `'H'` to plot the vector magnitudes of the electric and
            magnetic fields, and `'S'` for the Poynting vector.
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
        return self.mode_solver.plot_field(
            field_name=field_name,
            val=val,
            eps_alpha=eps_alpha,
            robust=robust,
            vmin=vmin,
            vmax=vmax,
            ax=ax,
            **sel_kwargs,
        )
