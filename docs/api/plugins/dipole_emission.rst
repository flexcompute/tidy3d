.. currentmodule:: tidy3d

Dipole Emission Study Plugin
============================

The dipole emission plugin evaluates radiative emission from classical electric
dipoles embedded in a passive optical structure. It reports angular radiation
intensity per squared dipole moment for x-, y-, and z-oriented dipoles into a
selected collection half-space. By default, the result is summed over all
sampled dipole positions using ``position_weights``. ``position_weights`` may
represent emitter-density quadrature weights, and may provide one weight per
position or one weight per position and dipole axis.

The main result is ``radiation_intensity`` with dimensions
``("dipole_axis", "polarization", "angle", "f")``:

* ``radiation_intensity`` is stored and includes the study ``source_time`` spectrum.
* ``radiation_intensity_transfer(bulk_refractive_index)`` is a derived method that
  normalizes by the total power the same dipoles would emit in a uniform medium at
  the given refractive index, yielding the cavity transfer in ``1/sr``. Integrating
  it over the study's collection solid angle gives the half-space radiative
  contribution; for a bulk emitter a single study integrates to ``1/2`` (the full
  radiative Purcell factor requires both collection half-spaces).

Use ``store_position_indexes`` to additionally retain radiation intensity at
selected individual positions. The stored optional array has dimensions
``("index", "dipole_axis", "polarization", "angle", "f")``, with a matching
derived transfer method.

Both outputs are normalized per squared electric dipole moment, with dipole
moment expressed in ``C*um``. The emitted field is linear in the dipole moment
``d``, so the radiated power scales as ``|d|**2``. Multiplying
``radiation_intensity`` by the physical ``|d|**2`` gives the corresponding
angular power density in ``W/sr``. The transfer methods are normalized ratios:
the ``source_time`` spectrum and the dipole-moment scaling cancel between the
radiated and bulk emitted powers.
If the dipole strength varies with position or orientation, include those
relative ``|d|**2`` factors in ``position_weights``.

Defining a Study
----------------

Use ``base_sim`` for the source-free optical environment and
``analysis_region`` for the region containing the emitters and the part of the
structure whose scattering response contributes to the collected radiation. The
``normal_axis`` and ``direction`` of the analysis region select the collection
side. For example, ``normal_axis=2`` and ``direction="+"`` describes emission
collected toward ``+z``. ``base_sim`` must be three-dimensional; the radiation
intensity is a per-solid-angle quantity, so 2D simulations (a zero-size
dimension) are rejected.

Far-field directions are supplied as a :class:`SphericalAngleDataArray`, with
``theta`` and ``phi`` in radians. ``theta`` is measured away from the selected
collection normal and must satisfy ``abs(theta) < pi/2``. The ``p``
polarization lies in the plane formed by the observation direction and
collection normal; ``s`` is orthogonal to that plane. On the result, the
per-direction angles are read from the ``DipoleEmissionStudyData.theta`` and
``.phi`` properties; the data arrays carry only an integer ``angle`` index.

.. code-block:: python

   import tidy3d as td
   from tidy3d.plugins import dipole_emission

   # Define base_sim as a source-free td.Simulation containing the optical structure.

   positions = td.PointDataArray(
       [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]
   )

   angles = td.SphericalAngleDataArray(
       [[0.0, 0.0], [0.3, 0.5]]
   )

   region = dipole_emission.EmissionAnalysisRegion(
       center=(0, 0, 0),
       size=(2, 2, 2),
       normal_axis=2,
       direction="+",
       dl=0.02,
   )

   study = dipole_emission.DipoleEmissionStudy(
       base_sim=base_sim,
       analysis_region=region,
       positions=positions,
       position_weights=[1.0, 1.0],
       store_position_indexes=(0,),
       source_time=td.GaussianPulse(freq0=freq0, fwidth=fwidth),
       freqs=[freq0],
       angles=angles,
       polarizations=("p", "s"),
   )

   data = study.run(folder_name="dipole_emission")

Notes
-----

Dipole-emission submissions require EM Enterprise entitlement in cloud
environments. The entitlement check is performed by the server-side submission
and estimate flow, so constructing a simulation or calling
``Simulation.validate_pre_upload()`` may still succeed locally before the server
rejects an account that does not have access.

Ordinary monitors in ``base_sim`` are copied into every angle and polarization
task as diagnostics. Their data is available only from the returned batch data,
for example with ``study.run(..., return_batch_data=True)``.

The study targets radiative emission into a continuum of far-field directions.
Non-radiative channels, ohmic absorption, surface-plasmon coupling, and trapped
guided-mode power are not included in the returned radiation intensity.

The v1 workflow is intended for open finite-domain collection geometries. Base
simulations whose boundary conditions are incompatible with the generated
reciprocal probes are rejected during ordinary Tidy3D validation.

API Reference
-------------

.. autosummary::
   :toctree: ../_autosummary/
   :template: module.rst

   plugins.dipole_emission.EmissionAnalysisRegion
   plugins.dipole_emission.DipoleEmissionStudy
   plugins.dipole_emission.DipoleEmissionStudyData
   plugins.dipole_emission.DipoleEmissionStudyDataArray
   plugins.dipole_emission.DipoleEmissionStudyPositionDataArray
   SphericalAngleDataArray
