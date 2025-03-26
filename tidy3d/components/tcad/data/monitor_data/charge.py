"""Monitor level data, store the DataArrays associated with a single heat-charge monitor."""

from __future__ import annotations

from typing import Dict, Union

import pydantic.v1 as pd

from tidy3d.components.base import skip_if_fields_missing
from tidy3d.components.data.data_array import (
    DataArray,
    IndexedVoltageDataArray,
    SpatialDataArray,
    SteadyVoltageDataArray,
)
from tidy3d.components.data.utils import TetrahedralGridDataset, TriangularGridDataset
from tidy3d.components.tcad.data.monitor_data.abstract import HeatChargeMonitorData
from tidy3d.components.tcad.monitors.charge import (
    SteadyCapacitanceMonitor,
    SteadyEnergyBandMonitor,
    SteadyFreeCarrierMonitor,
    SteadyPotentialMonitor,
)
from tidy3d.components.types import TYPE_TAG_STR, Ax, Axis, annotate_type
from tidy3d.components.viz import add_ax_if_none
from tidy3d.exceptions import DataError
from tidy3d.log import log

FieldDataset = Union[
    SpatialDataArray, annotate_type(Union[TriangularGridDataset, TetrahedralGridDataset])
]

UnstructuredFieldType = Union[TriangularGridDataset, TetrahedralGridDataset]


class SteadyPotentialData(HeatChargeMonitorData):
    """Stores electric potential :math:`\\psi` from a charge simulation."""

    monitor: SteadyPotentialMonitor = pd.Field(
        ...,
        title="Electric potential monitor",
        description="Electric potential monitor associated with a `charge` simulation.",
    )

    potential: FieldDataset = pd.Field(
        None,
        title="Electric potential series",
        description="Contains the electric potential series.",
    )

    @property
    def field_components(self) -> Dict[str, DataArray]:
        """Maps the field components to their associated data."""
        return dict(potential=self.potential)

    @pd.validator("potential", always=True)
    @skip_if_fields_missing(["monitor"])
    def warn_no_data(cls, val, values):
        """Warn if no data provided."""

        mnt = values.get("monitor")

        if val is None:
            log.warning(
                f"No data is available for monitor '{mnt.name}'. This is typically caused by "
                "monitor not intersecting any solid medium."
            )

        return val

    @property
    def symmetry_expanded_copy(self) -> SteadyPotentialData:
        """Return copy of self with symmetry applied."""

        new_potential = self._symmetry_expanded_copy(property=self.potential)
        return self.updated_copy(potential=new_potential, symmetry=(0, 0, 0))

    def field_name(self, val: str) -> str:
        """Gets the name of the fields to be plot."""
        if val == "abs^2":
            return "|V|²"
        else:
            return "V"


class SteadyFreeCarrierData(HeatChargeMonitorData):
    """
    Stores free-carrier concentration in charge simulations.

    Notes
    -----

        This data contains the carrier concentrations: the amount of electrons and holes per unit volume as defined in the
        ``monitor``.
    """

    monitor: SteadyFreeCarrierMonitor = pd.Field(
        ...,
        title="Free carrier monitor",
        description="Free carrier data associated with a Charge simulation.",
    )

    electrons: UnstructuredFieldType = pd.Field(
        None,
        title="Electrons series",
        description=r"Contains the computed electrons concentration $n$.",
        discriminator=TYPE_TAG_STR,
    )
    # n = electrons

    holes: UnstructuredFieldType = pd.Field(
        None,
        title="Holes series",
        description=r"Contains the computed holes concentration $p$.",
        discriminator=TYPE_TAG_STR,
    )
    # p = holes

    @property
    def field_components(self) -> Dict[str, DataArray]:
        """Maps the field components to their associated data."""
        return dict(electrons=self.electrons, holes=self.holes)

    @pd.root_validator(skip_on_failure=True)
    def check_correct_data_type(cls, values):
        """Issue error if incorrect data type is used"""

        mnt = values.get("monitor")
        field_data = {field: values.get(field) for field in ["electrons", "holes"]}

        for field, data in field_data.items():
            if isinstance(data, TetrahedralGridDataset) or isinstance(data, TriangularGridDataset):
                if not isinstance(data.values, IndexedVoltageDataArray):
                    raise ValueError(
                        f"In the data associated with monitor {mnt}, the field {field} does not contain "
                        "data associated to any voltage value."
                    )

        return values

    @pd.root_validator(skip_on_failure=True)
    def warn_no_data(cls, values):
        """Warn if no data provided."""

        mnt = values.get("monitor")
        electrons = values.get("electrons")
        holes = values.get("holes")

        if electrons is None or holes is None:
            log.warning(
                f"No data is available for monitor '{mnt.name}'. This is typically caused by "
                "monitor not intersecting any solid medium."
            )

        return values

    @property
    def symmetry_expanded_copy(self) -> SteadyFreeCarrierData:
        """Return copy of self with symmetry applied."""

        new_electrons = self._symmetry_expanded_copy(property=self.electrons)
        new_holes = self._symmetry_expanded_copy(property=self.holes)

        return self.updated_copy(
            electrons=new_electrons,
            holes=new_holes,
            symmetry=(0, 0, 0),
        )

    def field_name(self, val: str = "") -> str:
        """Gets the name of the fields to be plot."""
        if val == "abs^2":
            return "Electrons², Holes²"
        else:
            return "Electrons, Holes"


class SteadyEnergyBandData(HeatChargeMonitorData):
    """
    Stores energy bands in charge simulations.

    Notes
    -----

        This data contains the energy bands data:
        Ec -> Energy of the bottom of the conduction band, [eV]
        Ev -> Energy of the top of the valence band, [eV]
        Ei -> Intrinsic Fermi level, [eV]
        Efn -> Quasi-Fermi level for electrons, [eV]
        Efp -> Quasi-Fermi level for holes, [eV]
        as defined in the  ``monitor``.
    """

    monitor: SteadyEnergyBandMonitor = pd.Field(
        ...,
        title="Energy band monitor",
        description="Energy bands data associated with a Charge simulation.",
    )

    Ec: UnstructuredFieldType = pd.Field(
        None,
        title="Conduction band series",
        description=r"Contains the computed energy of the bottom of the conduction band $Ec$.",
        discriminator=TYPE_TAG_STR,
    )

    Ev: UnstructuredFieldType = pd.Field(
        None,
        title="Valence band series",
        description=r"Contains the computed energy of the top of the valence band $Ec$.",
        discriminator=TYPE_TAG_STR,
    )

    Ei: UnstructuredFieldType = pd.Field(
        None,
        title="Intrinsic Fermi level series",
        description=r"Contains the computed intrinsic Fermi level for the material $Ei$.",
        discriminator=TYPE_TAG_STR,
    )

    Efn: UnstructuredFieldType = pd.Field(
        None,
        title="Electron's quasi-Fermi level series",
        description=r"Contains the computed quasi-Fermi level for electrons $Efn$.",
        discriminator=TYPE_TAG_STR,
    )

    Efp: UnstructuredFieldType = pd.Field(
        None,
        title="Hole's quasi-Fermi level series",
        description=r"Contains the computed quasi-Fermi level for holes $Efp$.",
        discriminator=TYPE_TAG_STR,
    )

    @property
    def field_components(self) -> Dict[str, DataArray]:
        """Maps the field components to their associated data."""
        return dict(Ec=self.Ec, Ev=self.Ev, Ei=self.Ei, Efn=self.Efn, Efp=self.Efp)

    @pd.root_validator(skip_on_failure=True)
    def check_correct_data_type(cls, values):
        """Issue error if incorrect data type is used"""

        mnt = values.get("monitor")
        field_data = {field: values.get(field) for field in ["Ec", "Ev", "Ei", "Efn", "Efp"]}

        for field, data in field_data.items():
            if isinstance(data, TetrahedralGridDataset) or isinstance(data, TriangularGridDataset):
                if not isinstance(data.values, IndexedVoltageDataArray):
                    raise ValueError(
                        f"In the data associated with monitor {mnt}, the field {field} does not contain "
                        "data associated to any voltage value."
                    )

        return values

    @pd.root_validator(skip_on_failure=True)
    def warn_no_data(cls, values):
        """Warn if no data provided."""

        mnt = values.get("monitor")
        fields = ["Ec", "Ev", "Ei", "Efn", "Efp"]
        for field_name in fields:
            field_data = values.get(field_name)

            if field_data is None:
                log.warning(
                    f"No data is available for monitor '{mnt.name}'. This is typically caused by "
                    "monitor not intersecting any solid medium."
                )

        return values

    @property
    def symmetry_expanded_copy(self) -> SteadyEnergyBandData:
        """Return copy of self with symmetry applied."""

        new_Ec = self._symmetry_expanded_copy(property=self.Ec)
        new_Ev = self._symmetry_expanded_copy(property=self.Ev)
        new_Ei = self._symmetry_expanded_copy(property=self.Ei)
        new_Efn = self._symmetry_expanded_copy(property=self.Efn)
        new_Efp = self._symmetry_expanded_copy(property=self.Efp)

        return self.updated_copy(
            Ec=new_Ec,
            Ev=new_Ev,
            Ei=new_Ei,
            Efn=new_Efn,
            Efp=new_Efp,
            symmetry=(0, 0, 0),
        )

    def field_name(self, val: str = "") -> str:
        """Gets the name of the fields to be plot."""
        if val == "abs^2":
            return "|Ec|², |Ev|², |Ei|², |Efn|², |Efp|²"
        else:
            return "Ec, Ev, Ei, Efn, Efp"

    @add_ax_if_none
    def plot_slice(self, axis: Axis, pos: float, voltage: float, ax: Ax = None) -> Ax:
        """Plot the data field and/or the unstructured grid.

        Parameters
        ----------
        axis : Axis
            The normal direction of the slicing plane.
        pos : float
            Position of the slicing plane along its normal direction.
        voltage : float
            Applied bias voltage
        ax : matplotlib.axes._subplots.Axes = None
            matplotlib axes to plot on, if not specified, one is created.
        """

        if not isinstance(self.Ec, TriangularGridDataset):
            raise DataError(
                "Bandgap monitor slice plot can be done only for a 2D unstructured dataset."
            )

        if axis == self.Ec.normal_axis:
            raise DataError(
                f"Triangular grid (normal: {self.Ec.normal_axis}) cannot be sliced by a parallel "
                "plane."
            )

        Ec_slice = self.Ec.sel(voltage=voltage).plane_slice(axis=axis, pos=pos)
        Ev_slice = self.Ev.sel(voltage=voltage).plane_slice(axis=axis, pos=pos)
        Ei_slice = self.Ei.sel(voltage=voltage).plane_slice(axis=axis, pos=pos)
        Efn_slice = self.Efn.sel(voltage=voltage).plane_slice(axis=axis, pos=pos)
        Efp_slice = self.Efp.sel(voltage=voltage).plane_slice(axis=axis, pos=pos)

        Ec_slice.plot(ax=ax, label="Ec")
        Ev_slice.plot(ax=ax, label="Ev")
        Ei_slice.plot(ax=ax, label="Ei")
        Efn_slice.plot(ax=ax, label="Efn")
        Efp_slice.plot(ax=ax, label="Efp")
        ax.legend()

        return ax

    def plane_slice(self, axis: Axis, pos: float) -> SteadyEnergyBandData:
        """Slice data with a plane and return the resulting :class:`.SteadyEnergyBandData`.

        Parameters
        ----------
        axis : Axis
            The normal direction of the slicing plane.
        pos : float
            Position of the slicing plane along its normal direction.

        Returns
        -------
        SteadyEnergyBandData
            The resulting slice.
        """

        if not isinstance(self.Ec, TetrahedralGridDataset):
            raise DataError(
                "Bandgap monitor plane slice can be done only for a 3D unstructured dataset."
            )

        Ec_slice = self.Ec.plane_slice(axis=axis, pos=pos)
        Ev_slice = self.Ev.plane_slice(axis=axis, pos=pos)
        Ei_slice = self.Ei.plane_slice(axis=axis, pos=pos)
        Efn_slice = self.Efn.plane_slice(axis=axis, pos=pos)
        Efp_slice = self.Efp.plane_slice(axis=axis, pos=pos)

        return SteadyEnergyBandData(
            Ec=Ec_slice,
            Ev=Ev_slice,
            Ei=Ei_slice,
            Efn=Efn_slice,
            Efp=Efp_slice,
            monitor=self.monitor,
            symmetry=self.symmetry,
            symmetry_center=self.symmetry_center,
        )


class SteadyCapacitanceData(HeatChargeMonitorData):
    """
    Class that stores capacitance data from a Charge simulation.

    Notes
    -----
        The small signal-capacitance of electrons :math:`C_n` and holes  :math:`C_p`  is computed from the charge due to
         electrons :math:`Q_n` and holes :math:`Q_p` at an applied voltage :math:`V` at a voltage difference
        :math:`\\Delta V` between two simulations.

        .. math::

            C_{n,p} = \\frac{Q_{n,p}(V + \\Delta V) - Q_{n,p}(V)}{\\Delta V}


    This is only computed when a voltage source with more than two sources is included within the simulation and determines the :math:`\\Delta V`.
    """

    monitor: SteadyCapacitanceMonitor = pd.Field(
        ...,
        title="Capacitance monitor",
        description="Capacitance data associated with a Charge simulation.",
    )

    hole_capacitance: SteadyVoltageDataArray = pd.Field(
        None,
        title="Hole capacitance",
        description=r"Small signal capacitance ($\frac{dQ_p}{dV}$) associated to the monitor.",
    )
    # C_p = hole_capacitance

    electron_capacitance: SteadyVoltageDataArray = pd.Field(
        None,
        title="Electron capacitance",
        description=r"Small signal capacitance ($\frac{dQn}{dV}$) associated to the monitor.",
    )
    # C_n = electron_capacitance

    @pd.validator("hole_capacitance", always=True)
    @skip_if_fields_missing(["monitor"])
    def warn_no_data(cls, val, values):
        """Warn if no data provided."""

        mnt = values.get("monitor")

        if val is None:
            log.warning(
                f"No data is available for monitor '{mnt.name}'. This is typically caused by "
                "monitor not intersecting any solid medium."
            )

        return val

    def field_name(self, val: str) -> str:
        """Gets the name of the fields to be plot."""
        return ""

    @property
    def symmetry_expanded_copy(self) -> SteadyCapacitanceData:
        """Return copy of self with symmetry applied."""
        return self
