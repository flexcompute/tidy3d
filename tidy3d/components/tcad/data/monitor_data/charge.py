"""Monitor level data, store the DataArrays associated with a single heat-charge monitor."""

from __future__ import annotations

from typing import Dict, Union

import numpy as np
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
from tidy3d.components.types import TYPE_TAG_STR, Ax, annotate_type
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
        """Gets the name of the fields to be plotted."""
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
        """Gets the name of the fields to be plotted."""
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
        """Gets the name of the fields to be plotted."""
        if val == "abs^2":
            return "|Ec|², |Ev|², |Ei|², |Efn|², |Efp|²"
        else:
            return "Ec, Ev, Ei, Efn, Efp"

    @add_ax_if_none
    def plot(self, ax: Ax = None, **sel_kwargs) -> Ax:
        """Plot the 1D cross-section of the energy bandgap diagram.

        Parameters
        ----------
        ax : matplotlib.axes._subplots.Axes = None
            matplotlib axes to plot on, if not specified, one is created.
        sel_kwargs : keyword arguments used to perform ``.sel()`` selection in the monitor data.
            These kwargs can select over the spatial dimensions (``x``, ``y``, or ``z``)
            and the bias voltage (``voltage``).
            For the plotting to work appropriately, the resulting data after selection must contain
            only one coordinate with len > 1.
            Furthermore, these should be spatial coordinates (``x``, ``y``, or ``z``).
        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.
        """

        selection_data = dict()

        if ("voltage" not in sel_kwargs) and (self.Ec.values.coords.sizes["voltage"] > 1):
            raise DataError(
                "'voltage' is not selected for the plot with multiple voltage data points."
            )

        selection_data = {coord: sel_kwargs[coord] for coord in "xyz" if coord in sel_kwargs.keys()}
        N_coords = len(selection_data.keys())

        if "voltage" in sel_kwargs:
            selection_data["voltage"] = sel_kwargs["voltage"]

        if isinstance(self.Ec, TetrahedralGridDataset):
            if N_coords != 2:
                raise DataError(
                    "2 spatial coordinate values have to be defined to plot the 1D cross-section figure for a 3D dataset."
                )

        elif isinstance(self.Ec, TriangularGridDataset):
            if N_coords != 1:
                raise DataError(
                    "1 spatial coordinate value has to be defined to plot the 1D cross-section figure for a 2D dataset."
                )

            for index, coord_name in enumerate(["x", "y", "z"]):
                if coord_name in selection_data:
                    axis = index
                    continue

            if axis == self.Ec.normal_axis:
                raise DataError(
                    f"Triangular grid (normal: {self.Ec.normal_axis}) cannot be sliced by a parallel plane."
                )

        Ec_data = self.Ec
        Ev_data = self.Ev
        Ei_data = self.Ei
        Efn_data = self.Efn
        Efp_data = self.Efp

        for coord_name, coord_val in selection_data.items():
            Ec_data = Ec_data.sel(**{coord_name: coord_val}, method="nearest")
            Ev_data = Ev_data.sel(**{coord_name: coord_val}, method="nearest")
            Ei_data = Ei_data.sel(**{coord_name: coord_val}, method="nearest")
            Efn_data = Efn_data.sel(**{coord_name: coord_val}, method="nearest")
            Efp_data = Efp_data.sel(**{coord_name: coord_val}, method="nearest")

        Ec_data.plot(ax=ax, label="Ec")
        Ev_data.plot(ax=ax, label="Ev")
        Ei_data.plot(ax=ax, label="Ei")
        Efn_data.plot(ax=ax, label="Efn")
        Efp_data.plot(ax=ax, label="Efp")
        ax.legend()

        return ax


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
        """Gets the name of the fields to be plotted."""
        return ""

    @property
    def symmetry_expanded_copy(self) -> SteadyCapacitanceData:
        """Return copy of self with symmetry applied."""
        num_symmetries = np.sum(np.array([1 if d > 0 else 0 for d in self.symmetry]))
        scaling_factor = np.power(2, num_symmetries)

        if self.hole_capacitance is None:
            new_hole_capacitance = None
        else:
            new_values = self.hole_capacitance.values * scaling_factor
            new_hole_capacitance = SteadyVoltageDataArray(
                data=new_values, coords=self.hole_capacitance.coords
            )

        if self.electron_capacitance is None:
            new_electron_capacitance = None
        else:
            new_values = self.electron_capacitance.values * scaling_factor
            new_electron_capacitance = SteadyVoltageDataArray(
                data=new_values, coords=self.electron_capacitance.coords
            )

        return self.updated_copy(
            hole_capacitance=new_hole_capacitance,
            electron_capacitance=new_electron_capacitance,
            symmetry=(0, 0, 0),
        )
