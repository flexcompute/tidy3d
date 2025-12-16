.. currentmodule:: tidy3d

Sources
=======

Overview
--------

Sources in Tidy3D provide the necessary excitation to investigate the EM behavior of the structures under simulation. The type of source used in a simulation tends to be very application-specific. For instance, a ``PlaneWave`` source may be used for a unit cell simulation in a metalens; whereas a ``ModeSource`` would be more appropriate for a waveguide crossing problem.

A generic source can be thought of as composed of two parts: a spatial profile and a temporal profile. The former dictates the field distribution in space, whereas the latter determines the frequency band covered by the source, and by extension, the overall simulation.

The following section describes how to specify source time-dependence:

* `Source Time Dependence`_

The following sections describe the available spatial source distributions in Tidy3D:

* `Dipole and Uniform Current`_
* `Plane Wave`_
* `Gaussian Beam`_
* `Mode Source`_
* `Total-Field/Scattered-Field (TFSF)`_
* `User-defined`_

~~~~

Source Time Dependence
----------------------

.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   tidy3d.GaussianPulse
   tidy3d.ContinuousWave
   tidy3d.BroadbandPulse
   tidy3d.SourceTime
   tidy3d.CustomSourceTime

Each source requires the ``source_time`` parameter to be defined, which provides the time-dependence of the source.

.. code-block:: python

   # frequency information
   my_center_frequency = 200e12  # center frequency in Hz
   my_bandwidth = 20e12  # bandwidth
   
   # my source time
   my_source_time = GaussianPulse(
       freq0=my_center_frequency,
       fwidth=my_bandwidth
   )

   # a point dipole source
   my_dipole_source = PointDipole(
       center=(0,0,0),
       source_time=my_source_time,
       polarization='Ex'
   )

In the example above, we defined a ``PointDipole`` source with a modulated Gaussian pulse time-dependence. This is well-suited for simulations with a specified center frequency and bandwidth, and thus is by far the most common type of time-dependence used.

For niche applications, the user may wish to define a ``ContinuousWave`` excitation or even a ``CustomSourceTime`` function. Please refer to their respective documentation page for more details.

~~~~

Dipole and Uniform Current
--------------------------

.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   tidy3d.PointDipole
   tidy3d.UniformCurrentSource

The ``PointDipole`` represents an infinitesimally small isotropic radiator.

.. code-block:: python

   my_point_dipole = PointDipole(
       center=(0,0,0),
       source_time=my_source_time,
       polarization='Ex'
   )

The ``UniformCurrentSource`` represents a 1D line, 2D sheet, or 3D box of uniform current.

.. code-block:: python

   # define a 2D sheet of current polarized in the z-direction
   my_uniform_current_source = UniformCurrentSource(
       center=(0,0,0),
       size=(1,0,1),
       source_time=my_source_time,
       polarization='Ez'
   )

.. seealso::

   Example applications:

   * `Band structure calculation of a photonic crystal slab <../notebooks/Bandstructure.html>`_
   * `Nanobeam cavity <../notebooks/NanobeamCavity.html>`_

~~~~

Plane Wave 
----------

.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   tidy3d.PlaneWave
   tidy3d.FixedInPlaneKSpec
   tidy3d.FixedAngleSpec

The ``PlaneWave`` class represents an incident plane wave of a certain polarization and orientation. This is typically used in conjunction with periodic boundary conditions, e.g. in a unit cell simulation.

.. code-block:: python

   # define a x-polarized plane wave travelling at a 45 degree angle
   # from the positive z-axis
   my_plane_wave_source = PlaneWave(
       center=(0,0,-1),
       size=(td.inf, td.inf, 0),   # extend source to simulation edge
       source_time=my_source_time,
       direction='+',
       angle_theta=np.pi/4,
       pol_angle=0,
   )

.. seealso::

   For more detailed explanation and examples, please see the following learning center resources:

   + `Broadband plane wave with a constant oblique incident angle <../notebooks/BroadbandPlaneWaveWithConstantObliqueIncidentAngle.html>`_

   Example applications:

   * `Dielectric metasurface absorber <../notebooks/DielectricMetasurfaceAbsorber.html>`_
   * `Mid-IR metalens based on silicon nanopillars <../notebooks/MidIRMetalens.html>`_


~~~~

Gaussian Beam
-------------

.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   tidy3d.GaussianBeam
   tidy3d.AstigmaticGaussianBeam

The ``GaussianBeam`` and ``AstigmaticGaussianBeam`` classes implement the paraxial form of the Gaussian beam.

.. code-block:: python

   # define a y-polarized gaussian beam propagating in the +x direction
   # with waist radius = 3 um
   my_gaussian_beam = GaussianBeam(
       size=(0,10,10),   # size of the source plane
       direction='+',
       source_time=my_source_time,
       pol_angle=0,
       waist_radius=3,
       waist_distance=5,
   )

.. note::

   The paraxial approximation to the Gaussian beam is only appropriate when the ``waist_radius`` is larger than the source wavelength.
   

.. seealso::

   Example applications:

   * `Inverse taper edge coupler <../notebooks/EdgeCoupler.html>`_
   * `Uniform grating coupler <../notebooks/GratingCoupler.html>`_

~~~~

.. include:: /api/mode/sources.rst

~~~~

Total-Field/Scattered-Field (TFSF)
----------------------------------

.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   tidy3d.TFSF

The ``TFSF`` class represents an incident plane wave source that is confined to a specific TFSF region. This type of source is typically used for scattering simulations. Within the TFSF region, the overall field comprises of the incident plane wave and the scattered field. At the edges of the TFSF region, the incident plane wave field is deliberately cancelled out, such that outside the TFSF region, the total field only contains the scattered field.

.. seealso::

   For more detailed explanation and examples, please see the following learning center resource:

   + `Defining a total-field scattered-field (TFSF) plane wave source <../notebooks/TFSF.html>`_

   Example applications:

   + `Scattering of a plasmonic nanoparticle <../notebooks/PlasmonicNanoparticle.html>`_
   + `Multipole expansion for electromagnetic scattering <../notebooks/MultipoleExpansion.html>`_

~~~~

User-defined
------------

.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   tidy3d.CustomFieldSource
   tidy3d.CustomCurrentSource

It is possible to define an arbitrary field source in Tidy3D using ``CustomFieldSource`` or ``CustomCurrentSource``. Possible usage scenarios include defining a field profile that is not currently available in Tidy3D, or injecting a source that is based on the output of a previous simulation. 

Please see the learning center article linked below for more details. 

.. seealso::

   For more detailed explanation and examples, please see the following learning center resource:

   + `Defining spatially-varying sources <../notebooks/CustomFieldSource.html>`_

~~~~
