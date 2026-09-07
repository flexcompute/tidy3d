from __future__ import annotations

from typing import TYPE_CHECKING

import autograd.numpy as np
from pydantic import Field

from tidy3d.components.data.data_array import ScalarFieldDataArray
from tidy3d.components.data.dataset import (
    FIELD_STRUCTURE_E_COMPONENTS,
    FieldStructureDataset,
    MediumDataset,
    PermittivityDataset,
)
from tidy3d.components.data.em_fields import frequency_normalized_field_components
from tidy3d.components.data.monitor_data.base import AbstractFieldData

if TYPE_CHECKING:
    from collections.abc import Callable

    import xarray as xr

    from tidy3d.components.source.base import Source
    from tidy3d.components.types import ArrayFloat1D
from tidy3d.components.monitor import (
    FieldStructureMonitor,
    MediumMonitor,
    PermittivityMonitor,
)
from tidy3d.constants import EPSILON_0
from tidy3d.exceptions import (
    AdjointError,
    DataError,
)


class PermittivityData(PermittivityDataset, AbstractFieldData):
    """Data for a :class:`.PermittivityMonitor`: diagonal components of the permittivity tensor.

    Notes
    -----

        The data is stored as a `DataArray <https://docs.xarray.dev/en/stable/generated/xarray.DataArray.html>`_
        object using the `xarray <https://docs.xarray.dev/en/stable/index.html>`_ package.

    Example
    -------
    >>> from tidy3d import Grid, ScalarFieldDataArray
    >>> from tidy3d.components.grid.grid import Coords
    >>> x = [-1,1,3]
    >>> y = [-2,0,2,4]
    >>> z = [-3,-1,1,3,5]
    >>> f = [2e14, 3e14]
    >>> coords = dict(x=x[:-1], y=y[:-1], z=z[:-1], f=f)
    >>> grid = Grid(boundaries=Coords(x=x, y=y, z=z))
    >>> sclr_fld = ScalarFieldDataArray((1+1j) * np.random.random((2,3,4,2)), coords=coords)
    >>> monitor = PermittivityMonitor(size=(2,4,6), freqs=[2e14, 3e14], name='eps')
    >>> data = PermittivityData(
    ...     monitor=monitor, eps_xx=sclr_fld, eps_yy=sclr_fld, eps_zz=sclr_fld, grid_expanded=grid
    ... )
    """

    monitor: PermittivityMonitor = Field(
        title="Monitor",
        description="Permittivity monitor associated with the data.",
    )


class MediumData(MediumDataset, AbstractFieldData):
    """Data for a :class:`.MediumMonitor`: diagonal components of the permittivity and permeability tensor.

    Notes
    -----

        The data is stored as a `DataArray <https://docs.xarray.dev/en/stable/generated/xarray.DataArray.html>`_
        object using the `xarray <https://docs.xarray.dev/en/stable/index.html>`_ package.

    Example
    -------
    >>> from tidy3d import Grid, ScalarFieldDataArray
    >>> from tidy3d.components.grid.grid import Coords
    >>> x = [-1,1,3]
    >>> y = [-2,0,2,4]
    >>> z = [-3,-1,1,3,5]
    >>> f = [2e14, 3e14]
    >>> coords = dict(x=x[:-1], y=y[:-1], z=z[:-1], f=f)
    >>> grid = Grid(boundaries=Coords(x=x, y=y, z=z))
    >>> sclr_fld = ScalarFieldDataArray((1+1j) * np.random.random((2,3,4,2)), coords=coords)
    >>> monitor = MediumMonitor(size=(2,4,6), freqs=[2e14, 3e14], name='medium')
    >>> data = MediumData(
    ...     monitor=monitor, eps_xx=sclr_fld, eps_yy=sclr_fld, eps_zz=sclr_fld, mu_xx=sclr_fld, mu_yy=sclr_fld, mu_zz=sclr_fld, grid_expanded=grid
    ... )
    """

    monitor: MediumMonitor = Field(
        title="Monitor", description="Medium property monitor associated with the data."
    )


class FieldStructureData(FieldStructureDataset, AbstractFieldData):
    """Data for a :class:`.FieldStructureMonitor`: matched electric field, permittivity, and
    per-Yee structure-ownership components on their native Yee grids, plus the derived
    displacement field and absorbed power density.

    Notes
    -----

        The stored primitives (``Ex``/``Ey``/``Ez``, ``eps_xx``/``eps_yy``/``eps_zz``,
        ``structure_index_x``/``structure_index_y``/``structure_index_z``) are recorded on their
        native Yee grids. The derived displacement field components ``Dx``/``Dy``/``Dz`` and the
        ``absorbed_power_density`` are computed on demand from the raw ``self`` arrays; access the
        symmetry-expanded quantities via ``self.symmetry_expanded``.

        Following :class:`.PointCloudFieldMonitor`, the displacement components are stored in
        **electric-field units**, i.e. scaled only by the *relative* permittivity
        (``D / epsilon_0 = eps_rel * E``).

    Example
    -------
    >>> from tidy3d import Grid, ScalarFieldDataArray, SpatialDataArray
    >>> from tidy3d.components.grid.grid import Coords
    >>> x = [-1,1,3]
    >>> y = [-2,0,2,4]
    >>> z = [-3,-1,1,3,5]
    >>> f = [2e14, 3e14]
    >>> coords = dict(x=x[:-1], y=y[:-1], z=z[:-1], f=f)
    >>> scoords = dict(x=x[:-1], y=y[:-1], z=z[:-1])
    >>> grid = Grid(boundaries=Coords(x=x, y=y, z=z))
    >>> fld = ScalarFieldDataArray((1+1j) * np.random.random((2,3,4,2)), coords=coords)
    >>> idx = SpatialDataArray(np.zeros((2,3,4)), coords=scoords)
    >>> monitor = FieldStructureMonitor(size=(2,4,6), freqs=[2e14, 3e14], name='field_structure')
    >>> data = FieldStructureData(
    ...     monitor=monitor, Ex=fld, Ey=fld, Ez=fld,
    ...     eps_xx=fld, eps_yy=fld, eps_zz=fld,
    ...     structure_index_x=idx, structure_index_y=idx, structure_index_z=idx,
    ...     grid_expanded=grid,
    ... )
    """

    monitor: FieldStructureMonitor = Field(
        title="Monitor",
        description="Field–medium monitor associated with the data.",
    )

    def _displacement_component(self, e_name: str, eps_name: str) -> ScalarFieldDataArray:
        """Single displacement component ``D_i / epsilon_0 = eps_ii * E_i`` (in E-field units).

        ``eps_ii`` and ``E_i`` share the native Yee grid and frequencies, so they align directly.
        Computed from the raw ``self`` arrays; use ``self.symmetry_expanded`` for the full domain.
        """
        displacement = getattr(self, eps_name) * getattr(self, e_name)
        return ScalarFieldDataArray(displacement.data, coords=displacement.coords)

    def _displacement_components(self) -> dict[str, ScalarFieldDataArray]:
        """All three displacement components ``D_i = eps_ii * E_i`` (see :meth:`_displacement_component`)."""
        return {
            d_name: self._displacement_component(e_name, eps_name)
            for e_name, eps_name, d_name in zip(
                FIELD_STRUCTURE_E_COMPONENTS, ("eps_xx", "eps_yy", "eps_zz"), ("Dx", "Dy", "Dz")
            )
        }

    @property
    def Dx(self) -> ScalarFieldDataArray:
        """x-component of ``D / epsilon_0 = eps_xx * Ex`` on the ``Ex`` Yee grid."""
        return self._displacement_component("Ex", "eps_xx")

    @property
    def Dy(self) -> ScalarFieldDataArray:
        """y-component of ``D / epsilon_0 = eps_yy * Ey`` on the ``Ey`` Yee grid."""
        return self._displacement_component("Ey", "eps_yy")

    @property
    def Dz(self) -> ScalarFieldDataArray:
        """z-component of ``D / epsilon_0 = eps_zz * Ez`` on the ``Ez`` Yee grid."""
        return self._displacement_component("Ez", "eps_zz")

    def colocate_displacement(
        self, x: ArrayFloat1D = None, y: ArrayFloat1D = None, z: ArrayFloat1D = None
    ) -> xr.Dataset:
        """Colocate the displacement components ``Dx``/``Dy``/``Dz`` to supplied coordinates.

        Analogous to :meth:`.FieldData.colocate`, but for the derived displacement field. Returns
        an :class:`xarray.Dataset` of the colocated ``Dx``/``Dy``/``Dz`` components (in E-field
        units). Symmetry is not expanded here; call on ``self.symmetry_expanded`` for the full
        domain.

        Parameters
        ----------
        x, y, z : Optional[array-like] = None
            Coordinates to colocate to along each dimension; ``None`` skips that dimension.

        Returns
        -------
        xarray.Dataset
            Dataset containing the colocated ``Dx``/``Dy``/``Dz`` components.
        """
        supplied_coord_map = {k: np.array(v) for k, v in zip("xyz", (x, y, z)) if v is not None}
        centered: dict[str, ScalarFieldDataArray] = {}
        for name, data in self._displacement_components().items():
            for coord_name, coords_supplied in supplied_coord_map.items():
                if np.array(data.coords[coord_name]).size == 1:
                    raise DataError(
                        f"colocate_displacement given {coord_name}={coords_supplied}, but "
                        f"'{name}' has a single coordinate at "
                        f"{coord_name}={np.array(data.coords[coord_name])[0]}. "
                        f"Supply {coord_name}=None to skip this dimension."
                    )
            centered[name] = data.interp(**supplied_coord_map, kwargs={"bounds_error": True})
        # preserve traced arrays on item access so an (unsupported) objective built from the
        # result still registers a VJP path and reaches the explicit AdjointError below
        return self.package_colocate_results(centered)

    def _make_adjoint_sources(self, dataset_names: list[str], fwidth: float) -> list[Source]:
        """Objectives through this diagnostic monitor's data are unsupported: fail loudly.

        The inherited default returns no sources, which would silently produce zero-looking
        gradients instead of an error.
        """
        raise AdjointError(
            f"Objective functions depending on data of 'FieldStructureMonitor' "
            f"'{self.monitor.name}' are not supported: it is a diagnostic monitor with no "
            "adjoint implementation. Base the objective on a supported monitor type instead "
            "(e.g. 'FieldMonitor'). The 'FieldStructureMonitor' can still be carried in the "
            "simulation as long as the objective does not depend on its data."
        )

    @property
    def per_component_absorbed_power(self) -> dict[str, xr.DataArray]:
        """Per-component absorbed power ``p_i = 1/2 w eps0 Im(eps_ii) |E_i|^2`` on native grids.

        Computed from the raw ``self`` arrays (ungated, full absorption), each component kept on its
        own ``E_i`` Yee grid rather than colocated. Keys are ``"x"/"y"/"z"``. Colocating and summing
        these across components (as :attr:`absorbed_power_density` does) blends the staggered grids;
        keeping them separate preserves the exact per-component values (e.g. to feed a downstream
        solver per component and avoid interface smearing).
        """
        per_axis: dict[str, xr.DataArray] = {}
        for axis, e_name, eps_name in zip(
            "xyz", FIELD_STRUCTURE_E_COMPONENTS, ("eps_xx", "eps_yy", "eps_zz")
        ):
            e_field = getattr(self, e_name)
            eps = getattr(self, eps_name)
            # 1/2 w eps0 = pi f eps0; eps and E share the same 'f' coordinate. ``|E|**2`` avoids the
            # full-volume complex temporary that ``E * conj(E)`` would allocate.
            prefactor = np.pi * EPSILON_0 * e_field.coords["f"]
            per_axis[axis] = prefactor * eps.imag * np.abs(e_field) ** 2
        return per_axis

    def _colocate_axes(self, per_axis: dict[str, xr.DataArray]) -> dict[str, xr.DataArray]:
        """Colocate each per-axis native scalar onto the common colocation grid.

        Reuses :attr:`colocation_boundaries` (the same targets as field-monitor colocation) and
        applies the monitor's ``interval_space`` downsampling, so recorded data is colocated at its
        own resolution rather than interpolated back onto the full-resolution grid. Dimension-
        agnostic (no plane/normal assumption): each dimension with more than one sample is
        interpolated onto its (possibly downsampled) boundary coordinates, while a singleton
        (plane-normal) dimension is aligned to the monitor-plane center so the components share it.
        Returns the colocated components keyed as the input.
        """
        boundaries = self.colocation_boundaries.to_dict
        # Match the recorded (possibly ``interval_space``-downsampled) resolution rather than the
        # full grid; ``downsample`` is a no-op when ``interval_space`` is 1.
        targets = {
            dim: self.monitor.downsample(np.asarray(boundaries[dim]), axis=axis)
            for axis, dim in enumerate("xyz")
        }

        colocated: dict[str, xr.DataArray] = {}
        for name, data in per_axis.items():
            interp_coords = {}
            align_coords = {}
            for axis, dim in enumerate("xyz"):
                if dim not in data.dims:
                    continue
                if data.sizes[dim] > 1:
                    # in-plane: interpolate the staggered component onto the boundary coordinates
                    interp_coords[dim] = targets[dim]
                else:
                    # plane-normal singleton: share the monitor-plane center across components
                    align_coords[dim] = [self.monitor.center[axis]]
            if interp_coords:
                data = data.interp(**interp_coords, kwargs={"fill_value": 0})
            if align_coords:
                data = data.assign_coords(align_coords)
            colocated[name] = data
        return colocated

    def _colocate_and_sum_axes(self, per_axis: dict[str, xr.DataArray]) -> xr.DataArray:
        """Colocate the per-axis native scalars onto the common colocation grid and sum them.

        Thin wrapper over :meth:`_colocate_axes` (which handles the shared, ``interval_space``-aware
        colocation) that sums the colocated components.
        """
        total: xr.DataArray | None = None
        for data in self._colocate_axes(per_axis).values():
            total = data if total is None else total + data
        return total

    @property
    def absorbed_power_density(self) -> ScalarFieldDataArray:
        """Absorbed power density ``P_abs = 1/2 w eps0 sum_i Im(eps_ii) |E_i|^2`` [W/um^3].

        Purely electromagnetic (no band-gap gate): the per-component absorption is formed on each
        native Yee grid, then colocated onto the common grid-boundary coordinates and summed.
        Computed from the raw ``self`` arrays; use ``self.symmetry_expanded`` for the full domain.

        The electric field components are source-normalized (like all frequency-domain field data),
        so ``P_abs`` is per unit source power; scale to a physical input power via
        :meth:`.SimulationData.optical_generation`'s ``power_scale``.
        """
        total = self._colocate_and_sum_axes(self.per_component_absorbed_power)
        ordered = total.real.transpose(*[dim for dim in ("x", "y", "z", "f") if dim in total.dims])
        return ScalarFieldDataArray(ordered.data, coords=ordered.coords)

    def normalize(self, source_spectrum_fn: Callable[[float], complex]) -> FieldStructureData:
        """Return copy of self with only the electric field components source-normalized.

        ``Ex``/``Ey``/``Ez`` are divided by the source spectrum; the permittivity (``eps_*``) and
        structure-ownership (``structure_index_*``) components are intrinsic material properties and
        left untouched.
        """
        e_components = {
            name: field
            for name, field in self.field_components.items()
            if name in FIELD_STRUCTURE_E_COMPONENTS
        }
        if not e_components:
            return self.copy()
        return self.copy(
            deep=False,
            update=frequency_normalized_field_components(e_components, source_spectrum_fn),
        )
