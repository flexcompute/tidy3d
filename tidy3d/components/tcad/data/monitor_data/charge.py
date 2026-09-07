"""Monitor level data, store the DataArrays associated with a single heat-charge monitor."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from pydantic import Field, model_validator

from tidy3d.components.data.data_array import (
    IndexedFieldVoltageDataArray,
    IndexedVoltageDataArray,
    PointDataArray,
    SpatialDataArray,
    SteadyVoltageDataArray,
)
from tidy3d.components.data.utils import TetrahedralGridDataset, TriangularGridDataset
from tidy3d.components.tcad.data.monitor_data.abstract import HeatChargeMonitorData
from tidy3d.components.tcad.monitors.charge import (
    SelfHeatingMonitor,
    SteadyCapacitanceMonitor,
    SteadyChargeResidualMonitor,
    SteadyCurrentDensityMonitor,
    SteadyElectricFieldMonitor,
    SteadyEnergyBandMonitor,
    SteadyFreeCarrierMonitor,
    SteadyGenerationRecombinationMonitor,
    SteadyPotentialMonitor,
)
from tidy3d.components.types import TYPE_TAG_STR
from tidy3d.components.types.base import discriminated_union
from tidy3d.components.viz import add_ax_if_none
from tidy3d.constants import VOLUMETRIC_HEAT_RATE
from tidy3d.exceptions import DataError

if TYPE_CHECKING:
    from tidy3d.compat import Self
    from tidy3d.components.types import Ax

FieldDataset = (
    discriminated_union(TriangularGridDataset | TetrahedralGridDataset) | SpatialDataArray
)

UnstructuredFieldType = discriminated_union(TriangularGridDataset | TetrahedralGridDataset)


class SteadyPotentialData(HeatChargeMonitorData):
    """Stores electric potential :math:`\\psi` from a charge simulation."""

    monitor: SteadyPotentialMonitor = Field(
        title="Electric potential monitor",
        description="Electric potential monitor associated with a `charge` simulation.",
    )

    potential: FieldDataset | None = Field(
        default=None,
        title="Electric potential series",
        description="Contains the electric potential series.",
        json_schema_extra={"units": "V"},
    )

    @property
    def field_components(self) -> dict[str, FieldDataset | None]:
        """Maps the field components to their associated data."""
        return {"potential": self.potential}


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

    electrons: UnstructuredFieldType | None = Field(
        default=None,
        title="Electrons series",
        description=r"Contains the computed electrons concentration :math:`n`.",
        json_schema_extra={"units": "1/cm^3"},
    )
    # n = electrons

    holes: UnstructuredFieldType | None = Field(
        default=None,
        title="Holes series",
        description=r"Contains the computed holes concentration :math:`p`.",
        json_schema_extra={"units": "1/cm^3"},
    )
    # p = holes

    @property
    def field_components(self) -> dict[str, UnstructuredFieldType | None]:
        """Maps the field components to their associated data."""
        return {"electrons": self.electrons, "holes": self.holes}

    @model_validator(mode="after")
    def check_correct_data_type(self) -> Self:
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


class SteadyEnergyBandData(HeatChargeMonitorData):
    """
    Stores energy bands in charge simulations.

    Notes
    -----

    This data contains the energy bands data [eV]:

     .. list-table::
       :widths: 25 25 75
       :header-rows: 1

       * - Symbol
         - Parameter Name
         - Description
       * - :math:`E_c`
         - ``Ec``
         - Energy of the bottom of the conduction band
       * - :math:`E_v`
         - ``Ev``
         - Energy of the top of the valence band
       * - :math:`E_i`
         - ``Ei``
         - Intrinsic Fermi level
       * - :math:`E_{fn}`
         - ``Efn``
         - Quasi-Fermi level for electrons
       * - :math:`E_{fp}`
         - ``Efp``
         - Quasi-Fermi level for holes

    as defined in the  ``monitor``.
    """

    monitor: SteadyEnergyBandMonitor = Field(
        title="Energy band monitor",
        description="Energy bands data associated with a Charge simulation.",
    )

    Ec: UnstructuredFieldType | None = Field(
        default=None,
        title="Conduction band series",
        description="Contains the computed energy of the bottom of the conduction band :math:`E_c`.",
        json_schema_extra={"units": "eV"},
    )

    Ev: UnstructuredFieldType | None = Field(
        default=None,
        title="Valence band series",
        description="Contains the computed energy of the top of the valence band :math:`E_v`.",
        json_schema_extra={"units": "eV"},
    )

    Ei: UnstructuredFieldType | None = Field(
        default=None,
        title="Intrinsic Fermi level series",
        description="Contains the computed intrinsic Fermi level for the material :math:`E_i`.",
        json_schema_extra={"units": "eV"},
    )

    Efn: UnstructuredFieldType | None = Field(
        default=None,
        title="Electron's quasi-Fermi level series",
        description="Contains the computed quasi-Fermi level for electrons :math:`E_{fn}`.",
        json_schema_extra={"units": "eV"},
    )

    Efp: UnstructuredFieldType | None = Field(
        default=None,
        title="Hole's quasi-Fermi level series",
        description="Contains the computed quasi-Fermi level for holes :math:`E_{fp}`.",
        json_schema_extra={"units": "eV"},
    )

    @property
    def field_components(self) -> dict[str, UnstructuredFieldType | None]:
        """Maps the field components to their associated data."""
        return {"Ec": self.Ec, "Ev": self.Ev, "Ei": self.Ei, "Efn": self.Efn, "Efp": self.Efp}

    @model_validator(mode="after")
    def check_correct_data_type(self) -> Self:
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

    @add_ax_if_none
    def plot(self, ax: Ax = None, **sel_kwargs: Any) -> Ax:
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

    The small signal-capacitance of electrons  :math:`C_n`  and holes  :math:`C_p`  is computed from the charge due to electrons :math:`Q_n` and holes :math:`Q_p` at an applied voltage :math:`V` at a voltage difference
    :math:`\\Delta V` between two simulations.

    .. math::

        C_{n,p} = \\frac{Q_{n,p}(V + \\Delta V) - Q_{n,p}(V)}{\\Delta V}


    This is only computed when a voltage source with more than two sources is included within the simulation and determines the :math:`\\Delta V`.
    """

    monitor: SteadyCapacitanceMonitor = Field(
        title="Capacitance monitor",
        description="Capacitance data associated with a Charge simulation.",
    )

    hole_capacitance: SteadyVoltageDataArray | None = Field(
        default=None,
        title="Hole capacitance",
        description="Small signal capacitance :math:`(\\frac{dQ_p}{dV})` associated to the monitor. "
        "Units: fF (3D) or fF/μm (2D, per unit length).",
        json_schema_extra={"units": "fF"},
    )
    # C_p = hole_capacitance

    electron_capacitance: SteadyVoltageDataArray | None = Field(
        default=None,
        title="Electron capacitance",
        description="Small signal capacitance :math:`(\\frac{dQn}{dV})` associated to the monitor. "
        "Units: fF (3D) or fF/μm (2D, per unit length).",
        json_schema_extra={"units": "fF"},
    )
    # C_n = electron_capacitance

    @property
    def field_components(self) -> dict[str, UnstructuredFieldType]:
        """Maps the field components to their associated data."""
        return {}

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
            deep=False,
            validate=False,
        )


class SteadyElectricFieldData(HeatChargeMonitorData):
    """
    Stores electric field :math:`\\vec{E}` from a Charge/Conduction simulation.

    Notes
    -----
        The electric field is computed as the negative gradient of the electric potential :math:`\\vec{E} = -\\nabla \\psi`.
        It is given in units of :math:`V/\\mu m` (Volts per micrometer).
    """

    monitor: SteadyElectricFieldMonitor = Field(
        title="Electric field monitor",
        description="Electric field data associated with a Charge/Conduction simulation.",
    )

    E: UnstructuredFieldType | None = Field(
        default=None,
        title="Electric field",
        description="Contains the computed electric field.",
        json_schema_extra={"units": ":math:`V/\\mu m`"},
    )

    @property
    def field_components(self) -> dict[str, UnstructuredFieldType]:
        """Maps the field components to their associated data."""
        return {"E": self.E}

    @model_validator(mode="after")
    def check_correct_data_type(self) -> Self:
        """Issue error if incorrect data type is used"""

        if isinstance(self.E, TetrahedralGridDataset) or isinstance(self.E, TriangularGridDataset):
            if not isinstance(self.E.values, (IndexedFieldVoltageDataArray, PointDataArray)):
                raise ValueError(
                    f"The data associated with monitor {self.monitor.name} must contain a field. This can be "
                    "defined with 'IndexedFieldVoltageDataArray' or 'PointDataArray'."
                )

        return self


class SteadyChargeResidualData(HeatChargeMonitorData):
    """Stores per-node residual snapshots from a Charge simulation.

    Notes
    -----
        ``residual_potential``, ``residual_electrons``, ``residual_holes`` (and
        ``residual_temperature`` when the thermal solver is active) are the
        per-node signed residual components of each governing equation.  They are
        dimensionless and on the same scale as the simulation's convergence
        tolerance, so the nodes with the largest magnitude (approaching or
        exceeding that tolerance) are where the solution least satisfies the
        equations (the least-converged regions).
    """

    monitor: SteadyChargeResidualMonitor = Field(
        title="Residual monitor",
        description="Per-node residual monitor associated with a Charge simulation.",
    )

    residual_potential: UnstructuredFieldType = Field(
        title="Potential residual",
        description="Signed residual of the Poisson equation :math:`R_\\psi`.",
        json_schema_extra={"units": "dimensionless"},
    )

    residual_electrons: UnstructuredFieldType = Field(
        title="Electron continuity residual",
        description="Signed residual of the electron continuity (carrier conservation) "
        "equation :math:`R_n`.",
        json_schema_extra={"units": "dimensionless"},
    )

    residual_holes: UnstructuredFieldType = Field(
        title="Hole continuity residual",
        description="Signed residual of the hole continuity (carrier conservation) "
        "equation :math:`R_p`.",
        json_schema_extra={"units": "dimensionless"},
    )

    residual_temperature: UnstructuredFieldType | None = Field(
        default=None,
        title="Thermal residual",
        description="Signed residual of the heat equation :math:`R_T`. "
        "Present only when the thermal solver is active.",
        json_schema_extra={"units": "dimensionless"},
    )

    @property
    def field_components(self) -> dict[str, UnstructuredFieldType | None]:
        """Maps the field components to their associated data."""
        components: dict[str, UnstructuredFieldType | None] = {
            "residual_potential": self.residual_potential,
            "residual_electrons": self.residual_electrons,
            "residual_holes": self.residual_holes,
        }
        if self.residual_temperature is not None:
            components["residual_temperature"] = self.residual_temperature
        return components


class SelfHeatingData(HeatChargeMonitorData):
    """
    Stores the volumetric self-heating rate :math:`q` from a Charge/Conduction simulation.

    Notes
    -----
        ``heat_rate`` is the *total* volumetric heat generation rate in
        :math:`W/\\mu m^3`: Joule heating :math:`\\vec{J} \\cdot \\vec{E}` for a
        ``Conduction`` simulation, and Joule plus recombination heating for a ``Charge``
        simulation. Positive values generate heat, matching the sign convention of
        :class:`HeatSource`, so the field can be handed straight to a ``Heat``
        simulation through ``to_spatial_data_array``.
    """

    monitor: SelfHeatingMonitor = Field(
        title="Self-heating monitor",
        description="Self-heating monitor associated with a Charge/Conduction simulation.",
    )

    heat_rate: FieldDataset | None = Field(
        default=None,
        title="Volumetric heat rate",
        description="Contains the computed total volumetric heat generation rate.",
        json_schema_extra={"units": VOLUMETRIC_HEAT_RATE},
    )

    @property
    def field_components(self) -> dict[str, FieldDataset | None]:
        """Maps the field components to their associated data."""
        return {"heat_rate": self.heat_rate}

    def total_power(self, voltage: float | None = None) -> float:
        """Integrate the heat rate over the recorded region.

        Returns watts in 3D, and watts per micron of depth in 2D, where the mesh cells are
        areas. Useful either side of ``to_spatial_data_array``: the resampling and the
        second interpolation onto the consuming simulation's mesh both cost accuracy, and
        comparing the total before and after is the cheapest way to see how much.

        Parameters
        ----------
        voltage : float = None
            Bias point to integrate. Required when the data holds more than one bias.

        Returns
        -------
        float
            The integrated heat generation rate.
        """
        data = self._resolve_field(field="heat_rate")
        if isinstance(data, SpatialDataArray):
            raise DataError(
                f"The data for monitor '{self.monitor.name}' is already Cartesian; integrate "
                "it directly with xarray rather than through 'total_power'."
            )

        data = self._select_voltage(data=data, voltage=voltage)
        cell_values = np.asarray(data.get_cell_values(), dtype=float)
        cell_volumes = np.asarray(data.get_cell_volumes(), dtype=float)
        return float((cell_values * cell_volumes).sum())


class SteadyCurrentDensityData(HeatChargeMonitorData):
    """
    Stores current density :math:`\\vec{J}` from a Charge/Conduction simulation.
    Units: :math:`A/\\mu m^2` (3D) or :math:`A/\\mu m` (2D, per unit length).
    """

    monitor: SteadyCurrentDensityMonitor = Field(
        title="Current density monitor",
        description="Current density data associated with a Charge/Conduction simulation.",
    )

    J: UnstructuredFieldType | None = Field(
        default=None,
        title="Current density",
        description="Contains the computed current density.",
        discriminator=TYPE_TAG_STR,
        json_schema_extra={"units": ":math:`A/\\mu m^2`"},
    )

    @property
    def field_components(self) -> dict[str, UnstructuredFieldType]:
        """Maps the field components to their associated data."""
        return {"J": self.J}

    @model_validator(mode="after")
    def check_correct_data_type(self) -> Self:
        """Issue error if incorrect data type is used"""

        mnt = self.monitor
        J = self.J

        if isinstance(J, TetrahedralGridDataset) or isinstance(J, TriangularGridDataset):
            AcceptedTypes = (IndexedFieldVoltageDataArray, PointDataArray)
            if not isinstance(J.values, AcceptedTypes):
                raise ValueError(
                    f"In the data associated with monitor {mnt}, must contain a field. This can be "
                    "defined with IndexedFieldVoltageDataArray or PointDataArray."
                )

        return self


class SteadyGenerationRecombinationData(HeatChargeMonitorData):
    """Stores per-node generation-recombination rates from a Charge simulation.

    Notes
    -----
        ``net_recombination`` is the net generation-recombination rate entering
        the carrier continuity equations, :math:`U = R - G`, over every
        generation and recombination mechanism active in the simulation's media.
        Recombination is positive (removes carriers) and generation is negative
        (adds carriers). The remaining fields hold the individual mechanism
        contributions that make it up; of those, only ``impact_ionization`` is
        reported (when an impact-ionization model is active), and the rest stay
        ``None``. Rates are in :math:`cm^{-3}\\,s^{-1}`.
    """

    monitor: SteadyGenerationRecombinationMonitor = Field(
        title="Generation-recombination monitor",
        description="Generation-recombination rate monitor associated with a Charge simulation.",
    )

    net_recombination: UnstructuredFieldType | None = Field(
        default=None,
        title="Net generation-recombination rate",
        description="Net generation-recombination source term :math:`U = R - G` entering the "
        "continuity equations, including every generation-recombination mechanism active "
        "in the simulation's media (recombination positive, generation negative).",
        json_schema_extra={"units": "1/(cm^3 s)"},
    )

    impact_ionization: UnstructuredFieldType | None = Field(
        default=None,
        title="Impact-ionization generation rate",
        description="Signed contribution of impact-ionization (avalanche) generation, "
        ":math:`-G_\\mathrm{ii}` (negative, since generation adds carriers). Present only "
        "when an impact-ionization model is active.",
        json_schema_extra={"units": "1/(cm^3 s)"},
    )

    shockley_reed_hall: UnstructuredFieldType | None = Field(
        default=None,
        title="Shockley-Reed-Hall recombination rate",
        description="Signed contribution of Shockley-Reed-Hall recombination, "
        ":math:`R_\\mathrm{SRH}` (positive). Not currently reported; stays ``None``.",
        json_schema_extra={"units": "1/(cm^3 s)"},
    )

    auger: UnstructuredFieldType | None = Field(
        default=None,
        title="Auger recombination rate",
        description="Signed contribution of Auger recombination, "
        ":math:`R_\\mathrm{Auger}` (positive). Not currently reported; stays ``None``.",
        json_schema_extra={"units": "1/(cm^3 s)"},
    )

    radiative: UnstructuredFieldType | None = Field(
        default=None,
        title="Radiative recombination rate",
        description="Signed contribution of radiative recombination, "
        ":math:`R_\\mathrm{rad}` (positive). Not currently reported; stays ``None``.",
        json_schema_extra={"units": "1/(cm^3 s)"},
    )

    band_to_band_tunneling: UnstructuredFieldType | None = Field(
        default=None,
        title="Band-to-band tunneling generation rate",
        description="Signed contribution of band-to-band tunneling generation, "
        ":math:`-G_\\mathrm{btbt}` (negative, since generation adds carriers). "
        "Not currently reported; stays ``None``.",
        json_schema_extra={"units": "1/(cm^3 s)"},
    )

    distributed_generation: UnstructuredFieldType | None = Field(
        default=None,
        title="Distributed carrier generation rate",
        description="Signed contribution of distributed carrier generation, "
        ":math:`-G_\\mathrm{dist}` (negative, since generation adds carriers). "
        "Not currently reported; stays ``None``.",
        json_schema_extra={"units": "1/(cm^3 s)"},
    )

    @property
    def field_components(self) -> dict[str, UnstructuredFieldType | None]:
        """Maps the field components to their associated data."""
        # ``net_recombination`` stays present even when ``None`` so missing data
        # is reported like other monitors; per-mechanism entries appear only
        # when populated.
        components: dict[str, UnstructuredFieldType | None] = {
            "net_recombination": self.net_recombination
        }
        optional = {
            "impact_ionization": self.impact_ionization,
            "shockley_reed_hall": self.shockley_reed_hall,
            "auger": self.auger,
            "radiative": self.radiative,
            "band_to_band_tunneling": self.band_to_band_tunneling,
            "distributed_generation": self.distributed_generation,
        }
        components.update({name: data for name, data in optional.items() if data is not None})
        return components

    @model_validator(mode="after")
    def check_correct_data_type(self) -> Self:
        """Issue error if incorrect data type is used"""
        fields = [
            "net_recombination",
            "impact_ionization",
            "shockley_reed_hall",
            "auger",
            "radiative",
            "band_to_band_tunneling",
            "distributed_generation",
        ]
        field_data = {field: getattr(self, field) for field in fields}
        for field, data in field_data.items():
            if isinstance(data, TetrahedralGridDataset) or isinstance(data, TriangularGridDataset):
                if not isinstance(data.values, IndexedVoltageDataArray):
                    raise ValueError(
                        f"In the data associated with monitor {self.monitor}, the "
                        f"field {field} does not contain data associated to any voltage value."
                    )
        return self
