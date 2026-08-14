.. currentmodule:: tidy3d

Charge-Heat Coupled Sources
-----------------------------

.. autosummary::
   :toctree: ../_autosummary/
   :template: module.rst

   HeatFromElectricSource

.. _charge-self-heating:

Exporting self-heating to a heat simulation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

An *isothermal* charge simulation solves at a fixed lattice temperature, but it still
computes the heat its own operation generates: Joule heating plus the heat released by
recombination. Record that field with a :class:`SelfHeatingMonitor`, pick the bias point
you care about, and feed it to a separate heat simulation as a :class:`HeatSource`::

    charge_data = web.run(charge_sim, task_name="charge")
    q = charge_data["self_heat"].to_spatial_data_array(
        bounds=device_bounds, resolution=0.01, voltage=1.0
    )
    heat_sim = td.HeatChargeSimulation(
        ...,
        sources=[td.HeatSource(rate=q, structures=["device"])],
        monitors=[temperature_monitor],
    )

The handoff is explicit rather than automatic because a charge simulation solves at a
list of bias points while a heat simulation solves once — only you can say which bias the
temperature should correspond to. Consequently, a :class:`HeatFromElectricSource` is
rejected in an *isothermal* charge simulation; it drives a heat solve from a
``Conduction`` simulation only, where a single voltage is the only supported case anyway.
On a non-isothermal charge simulation the source is ignored with a warning rather than
rejected, since that analysis already couples heat self-consistently.

A ``Conduction`` simulation exports its Joule heating through the same monitor, if you
want the field itself rather than the coupled temperature.

Either way the field is resampled from the electrical solver's unstructured grid onto a
Cartesian grid, and the heat solver then interpolates it onto the heat mesh. Detail lost
in the first step cannot be recovered in the second, so choose a resolution fine enough
to resolve the heating profile — its peaks sit in junctions and at contacts, not in the
bulk.

This is a **one-way** handoff: the charge solve does not see the resulting temperature
rise. For a self-consistent solution use a non-isothermal analysis spec
(:class:`SteadyChargeDCAnalysis`), which couples both physics inside the solver.
