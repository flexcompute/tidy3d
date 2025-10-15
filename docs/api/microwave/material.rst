.. _rf_material_models:

RF Materials Models
-------------------

.. autosummary::
   :toctree: ../_autosummary/
   :template: module.rst

   tidy3d.PECMedium
   tidy3d.PMCMedium
   tidy3d.rf.LossyMetalMedium
   tidy3d.rf.SurfaceImpedanceFitterParam
   tidy3d.rf.HammerstadSurfaceRoughness
   tidy3d.rf.HuraySurfaceRoughness

The :class:`~tidy3d.PECMedium` and :class:`~tidy3d.rf.LossyMetalMedium` classes can be used to model metallic materials.

.. code-block:: python

   # lossless metal
   my_pec = PECMedium()

   # lossy metal (conductivity in S/um)
   my_lossy_metal = LossyMetalMedium(conductivity=58, freq_range=(1e9, 10e9))

Note that the unit of ``conductivity`` is ``S/um`` and the unit of ``freq_range`` is ``Hz``. The :class:`~tidy3d.rf.LossyMetalMedium` class implements the surface impedance boundary condition (SIBC). It can accept surface roughness specifications using the Hammerstad or Huray models. Please refer to their respective documentation pages for details. Edge singularity correction is also available but turned off by default at this time.

.. note::

   When modeling lossy metals, always be sure to check the skin depth --- if the skin depth is significant compared to the geometry size, then :class:`~tidy3d.rf.LossyMetalMedium` may be not accurate. In that case, use a regular dispersive medium instead.


.. autosummary::
   :toctree: ../_autosummary/
   :template: module.rst

   tidy3d.Medium
   tidy3d.plugins.dispersion.FastDispersionFitter

To model lossless dielectrics, use the regular :class:`~tidy3d.Medium`.

.. code-block:: python

   # lossless dielectric
   my_lossless_dielectric = Medium(permittivity=2.2)

To model a lossy dielectric with constant loss tangent, use the ``constant_loss_tangent_model()`` method of the :class:`tidy3d.plugins.dispersion.FastDispersionFitter` utility class.

.. code-block:: python

   # lossy dielectric (constant loss tangent)
   my_lossy_dielectric = FastDispersionFitter.constant_loss_tangent_model(
       eps_real=4.4,
       loss_tangent=0.002,
       frequency_range=(1e9, 5e9)
   )

More advanced material models, including frequency dependence and anisotropy, are also available in Tidy3D.


.. seealso::

   For a more comprehensive discussion of the different EM mediums available in Tidy3D, please refer to the EM Mediums page:

   + `EM Mediums <../mediums.html>`_

~~~~