.. currentmodule:: tidy3d

Sources
=======

Overview
--------

Sources in Tidy3D provide the necessary excitation to investigate the EM behaviour of the structures under simulation. The type of source used in a simulation tends to very application-specific. For instance, a ``PlaneWave`` source may be used for a unit cell simulation in a metalens; whereas a ``ModeSource`` would be more appropriate for a waveguide crossing problem.

The following sections on this page describe the respective source types available in Tidy3D:

* `Dipole and Uniform Current`_
* `Plane Wave`_
* `Gaussian Beam`_
* `Mode Source`_
* `Total-Field/Scattered-Field (TFSF)`_
* `User-defined`_

In addition to the spatial distribution of the source, it is also important to define the source time-dependence. This is covered in the following section:

* `Source Time Dependence`_

~~~~

Source Time Dependence
----------------------

.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   tidy3d.GaussianPulse
   tidy3d.ContinuousWave
   tidy3d.CustomSourceTime

Each source requires the ``source_time`` parameter to be defined, which provides the time-dependence of the source.

.. code-block:: python

   # frequency information
   my_center_frequency = 200e12  # center frequency in Hz
   my_bandwidth = 20e12  # bandwidth
   
   # my source time
   my_source_time = GaussianPulse(freq0=my_center_frequency, fwidth=my_bandwidth)

   # a point dipole source
   my_dipole_source = PointDipole(source_time=my_source_time)

In the example above, we defined a ``PointSource`` with a modulated Gaussian pulse time-dependence. This is well-suited for simulations with a specific center frequency and bandwidth, and thus is by far the most common type of time-dependence used.

For specific applications, the user may wish to define a ``ContinuousWave`` excitation or even a ``CustomSourceTime`` function. Please refer to their respective documentation page for more details.

~~~~

Dipole and Uniform Current
--------------------------

.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   tidy3d.PointDipole
   tidy3d.UniformCurrentSource

~~~~

Plane Wave 
----------

.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   tidy3d.PlaneWave
   tidy3d.FixedInPlaneK
   tidy3d.FixedAngle


.. seealso::

   For more detailed explanation and examples, please see the following learning center resources:

   + `Broadband plane wave with a constant oblique incident angle <../notebooks/BroadbandPlaneWaveWithConstantObliqueIncidentAngle.html>`_


~~~~

Gaussian Beam
-------------

.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   tidy3d.GaussianBeam
   tidy3d.AstigmaticGaussianBeam



~~~~

Mode Source
-----------

.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   tidy3d.ModeSource
   tidy3d.ModeSpec


.. seealso::

   For more detailed explanation and examples, please see the following learning center resources:

   + `Defining mode sources and monitors <../notebooks/ModalSourcesMonitors.html>`_
   + `Injecting modes in bent and angled waveguides <../notebooks/ModeBentAngled.html>`_

   For a short introduction to the use of mode sources, see the following FDTD101 lecture:

   + `Mode injection <https://www.flexcompute.com/fdtd101/Lecture-4-Prelude-to-Integrated-Photonics-Simulation-Mode-Injection/>`_

~~~~

Total-Field/Scattered-Field (TFSF)
----------------------------------

.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   tidy3d.TFSF

.. seealso::

   For more detailed explanation and examples, please see the following learning center resource:

   + `Defining a total-field scattered-field (TFSF) plane wave source <../notebooks/TFSF.html>`_

~~~~

User-defined
------------

.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   tidy3d.CustomFieldSource
   tidy3d.CustomCurrentSource


.. seealso::

   For more detailed explanation and examples, please see the following learning center resource:

   + `Defining spatially-varying sources <../notebooks/CustomFieldSource.html>`_

~~~~

