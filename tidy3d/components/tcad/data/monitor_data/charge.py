"""Monitor level data, store the DataArrays associated with a single heat-charge monitor."""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
from pydantic import Field, model_validator

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
from tidy3d.components.types import Ax, discriminated_union
from tidy3d.components.viz import add_ax_if_none
from tidy3d.exceptions import DataError
from tidy3d.log import log

FieldDataset = Union[
    SpatialDataArray, discriminated_union(Union[TriangularGridDataset, TetrahedralGridDataset])
]

UnstructuredFieldType = discriminated_union(Union[TriangularGridDataset, TetrahedralGridDataset])


class SteadyPotentialData(HeatChargeMonitorData):
    """Stores electric potential :math:`\\psi` from a charge simulation."""

    monitor: SteadyPotentialMonitor = Field(
        title="Electric potential monitor",
        description="Electric potential monitor associated with a `charge` simulation.",
    )

    potential: Optional[FieldDataset] = Field(
        None,
        title="Electric potential series",
        description="Contains the electric potential series.",
    )

    @property
    def field_components(self) -> dict[str, DataArray]:
        """Maps the field components to their associated data."""
        return {"potential": self.potential}

    @model_validator(mode="after")
    def warn_no_data(self):
        """Warn if no data provided."""
        if self.potential is None:
            log.warning(
                f"No data is available for monitor '{self.monitor.name}'. This is "
                "typically caused by monitor not intersecting any solid medium."
            )

        return self

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

    monitor: SteadyFreeCarrierMonitor = Field(
        title="Free carrier monitor",
        description="Free carrier data associated with a Charge simulation.",
    )

    electrons: Optional[UnstructuredFieldType] = Field(
        None,
        title="Electrons series",
        description=r"Contains the computed electrons concentration $n$.",
    )
    # n = electrons

    holes: Optional[UnstructuredFieldType] = Field(
        None,
        title="Holes series",
        description=r"Contains the computed holes concentration $p$.",
    )
    # p = holes

    @property
    def field_components(self) -> dict[str, DataArray]:
        """Maps the field components to their associated data."""
        return {"electrons": self.electrons, "holes": self.holes}

    @model_validator(mode="after")
    def check_correct_data_type(self):
        """Issue error if incorrect data type is used"""
        field_data = {field: getattr(self, field) for field in ["electrons", "holes"]}
        for field, data in field_data.items():
            if isinstance(data, TetrahedralGridDataset) or isinstance(data, TriangularGridDataset):
                if not isinstance(data.values, IndexedVoltageDataArray):
                    raise ValueError(
                        f"In the data associated with monitor {self.monitor}, the "
                        f"field {field} does not contain data associated to any voltage value."
                    )
        return self

    @model_validator(mode="after")
    def warn_no_data(self):
        """Warn if no data provided."""

        if self.electrons is None or self.holes is None:
            log.warning(
                f"No data is available for monitor '{self.monitor.name}'. This is "
                "typically caused by monitor not intersecting any solid medium."
            )
        return self

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

    monitor: SteadyEnergyBandMonitor = Field(
        title="Energy band monitor",
        description="Energy bands data associated with a Charge simulation.",
    )

    Ec: Optional[UnstructuredFieldType] = Field(
        None,
        title="Conduction band series",
        description=r"Contains the computed energy of the bottom of the conduction band $Ec$.",
    )

    Ev: Optional[UnstructuredFieldType] = Field(
        None,
        title="Valence band series",
        description=r"Contains the computed energy of the top of the valence band $Ec$.",
    )

    Ei: Optional[UnstructuredFieldType] = Field(
        None,
        title="Intrinsic Fermi level series",
        description=r"Contains the computed intrinsic Fermi level for the material $Ei$.",
    )

    Efn: Optional[UnstructuredFieldType] = Field(
        None,
        title="Electron's quasi-Fermi level series",
        description=r"Contains the computed quasi-Fermi level for electrons $Efn$.",
    )

    Efp: Optional[UnstructuredFieldType] = Field(
        None,
        title="Hole's quasi-Fermi level series",
        description=r"Contains the computed quasi-Fermi level for holes $Efp$.",
    )

    @property
    def field_components(self) -> dict[str, DataArray]:
        """Maps the field components to their associated data."""
        return {"Ec": self.Ec, "Ev": self.Ev, "Ei": self.Ei, "Efn": self.Efn, "Efp": self.Efp}

    @model_validator(mode="after")
    def check_correct_data_type(self):
        """Issue error if incorrect data type is used"""

        field_data = {field: getattr(self, field) for field in ["Ec", "Ev", "Ei", "Efn", "Efp"]}

        for field, data in field_data.items():
            if isinstance(data, TetrahedralGridDataset) or isinstance(data, TriangularGridDataset):
                if not isinstance(data.values, IndexedVoltageDataArray):
                    raise ValueError(
                        f"In the data associated with monitor {self.monitor}, the "
                        f"field {field} does not contain data associated to any voltage value."
                    )

        return self

    @model_validator(mode="after")
    def warn_no_data(self):
        """Warn if no data provided."""

        fields = ["Ec", "Ev", "Ei", "Efn", "Efp"]
        for field_name in fields:
            field_data = getattr(self, field_name)

            if field_data is None:
                log.warning(
                    f"No data is available for monitor '{self.monitor.name}'. This "
                    "is typically caused by monitor not intersecting any solid medium."
                )

        return self

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

        selection_data = {}

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

    monitor: SteadyCapacitanceMonitor = Field(
        title="Capacitance monitor",
        description="Capacitance data associated with a Charge simulation.",
    )

    hole_capacitance: Optional[SteadyVoltageDataArray] = Field(
        None,
        title="Hole capacitance",
        description=r"Small signal capacitance ($\frac{dQ_p}{dV}$) associated to the monitor.",
    )
    # C_p = hole_capacitance

    electron_capacitance: Optional[SteadyVoltageDataArray] = Field(
        None,
        title="Electron capacitance",
        description=r"Small signal capacitance ($\frac{dQn}{dV}$) associated to the monitor.",
    )
    # C_n = electron_capacitance

    @model_validator(mode="after")
    def warn_no_data(self):
        """Warn if no data provided."""

        if self.hole_capacitance is None:
            log.warning(
                f"No data is available for monitor '{self.monitor.name}'. This is "
                "typically caused by monitor not intersecting any solid medium."
            )

        return self

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
