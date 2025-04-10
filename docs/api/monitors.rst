
.. currentmodule:: tidy3d

Monitors
========

Overview
--------

The full space-time distribution of the EM field is typically too large to efficiently store on disk or send over networks. Instead, we use monitors to record specific subsets of the field distribution, or derived quantities, that are relevant to our simulation goals.

The types of monitors in Tidy3D include:

* `Field`_: Records the EM field components in a spatial region (from 0D up to 3D) at specified time or frequency points
* `Flux`_: Records EM power flow across a 2D surface or 3D bounding box at specified time or frequency points
* `Mode`_: Records mode coefficient(s) of the field across a 2D plane
* `Diffraction`_: Records diffraction coefficient(s) in a periodic simulation
* `Far-field`_: Various monitors for calculating far-field projection and radiation characteristics
* `Permittivity`_: Records material properties within a given region

.. seealso::

   To learn more about accessing and plotting monitor data, please refer to the following learning center articles:

   + `Performing visualization of simulation data <../notebooks/VizData.html>`_
   + `Advanced monitor data (xarray) manipulation and visualization <../notebooks/XarrayTutorial.html>`_
   + `Creating FDTD animations <../notebooks/AnimationTutorial.html>`_

~~~~

Field 
-----

.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   tidy3d.FieldMonitor
   tidy3d.FieldTimeMonitor

The ``FieldMonitor`` records the EM field components within a spatial region at specified frequency point(s). The ``FieldTimeMonitor`` does the same, except at specified time intervals instead of frequency.

.. code-block:: python

   # define a 2D field monitor at frequency f0
   my_field_monitor = FieldMonitor(
       center=(0,0,0),
       size=(10,10,0),
       name='My field monitor',
       freqs=[f0],
   )

   # define a 1D field-time monitor starting at 1ps
   my_fieldtime_monitor = FieldTimeMonitor(
       center=(0,0,0),
       size=(10,0,0),
       name='My fieldtime monitor',
       start=1e-12,
       interval=50,   # number of solver time steps between each measurement
   )

.. note::

   The amount of data generated can be very large when recording 2D or 3D field information across a large number of time or frequency points. To save space, consider using the ``interval_space`` parameter to downsample the grid resolution, or the ``interval``, ``freqs`` parameters to reduce the number of time/frequency points.

~~~~

Flux
----

.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   tidy3d.FluxMonitor
   tidy3d.FluxTimeMonitor

The ``FluxMonitor`` records EM power flux through a 2D surface or 3D bounding box at specified frequency point(s). The ``FluxTimeMonitor`` does the same, except at specified time intervals instead of frequency.

.. code-block:: python

   # define 2D flux monitor at frequency f0
   # to record power flux in the +z direction
   my_flux_monitor = FluxMonitor(
       center=(0,0,0),
       size=(10,10,0),
       name='My flux monitor',
       normal_dir='+',
       freqs=[f0],
   )

   # define a 3D flux-time monitor to record outgoing power vs time
   my_fluxtime_monitor = FluxTimeMonitor(
       center=(0,0,0),
       size=(10,10,10),
       name='My flux-time monitor',
       interval=50,    # number of solver times between each measurement
   )

.. seealso::

   Example applications:

   + `Scattering of a plasmonic nanoparticle <../notebooks/PlasmonicNanoparticle.html>`_
   + `Plasmonic Yagi-Uda nanoantenna <../notebooks/PlasmonicYagiUdaNanoantenna.html>`_

~~~~

Mode
----

.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   tidy3d.ModeSpec
   tidy3d.ModeMonitor
   tidy3d.ModeSolverMonitor

The ``ModeMonitor`` records the mode coefficients of the incident field at specified frequency point(s).

Behind the scene, a mode solver simulation is first performed to determine the eigenmodes in the ``ModeMonitor`` plane. A ``ModeSpec`` instance is necessary to provide the settings for this calculation. Then, the resulting modes are used to calculate the mode coefficients. The user does not need to explicitly perform the mode solver simulation, as it is automatically performed whenever a ``ModeSource`` or ``ModeMonitor`` is present in the simulation.

.. code-block:: python

   # define a mode spec
   # search for 3 modes near effective index of 2.2
   my_mode_spec = ModeSpec(num_modes=3, target_neff=2.2)

   # define a mode monitor at 20 freq points between 240 and 300 THz
   my_mode_monitor = ModeMonitor(
       center=(10,0,0),
       size=(0,20,20),
       name='My mode monitor',
       freqs=np.linspace(240e12, 300e12, 20),
       mode_spec=my_mode_spec,
   )
   
.. seealso::

   For more details and examples, please refer to the following learning center article:

   + `Defining mode sources and monitors <../notebooks/ModalSourcesMonitors>`_

   Example applications:

   + `Waveguide Y-junction <../notebooks/YJunction.html>`_
   + `Broadband directional coupler <../notebooks/BroadbandDirectionalCoupler.html>`_

~~~~

Diffraction
-----------

.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   tidy3d.DiffractionMonitor

The ``DiffractionMonitor`` records the diffraction coefficients of the allowed diffraction orders in a periodic simulation.

.. code-block::

   # define a diffraction monitor at 20 freq points between 250 and 300 THz
   # for outgoing fields in the +z direction
   my_diffraction_monitor = DiffractionMonitor(
       center=(0,0,10),
       size=(td.inf, td.inf, 0),    # extend monitor to simulation edges
       freqs=np.linspace(250e12, 300e12, 20),
       name='My diffraction monitor`,
       normal_dir='+',
   )

For more detailed examples, please refer to the demo models linked below. 

.. seealso::

   Example applications:

   + `Multilevel blazed diffraction grating <../notebooks/GratingEfficiency.html>`_
   + `Mid-IR metalens based on silicon nanopillars <../notebooks/MidIRMetalens.html>`_

~~~~

Far-field
---------

.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   tidy3d.FieldProjectionCartesianMonitor
   tidy3d.FieldProjectionAngleMonitor
   tidy3d.FieldProjectionKSpaceMonitor
   tidy3d.DirectivityMonitor

.. seealso::

   For more details and examples, please refer to the following learning center article:

   + `Performing near field to far field projections <../notebooks/FieldProjections.html>`_

~~~~

Permittivity
------------

.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   tidy3d.PermittivityMonitor

~~~~

Apodization
-----------

.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   tidy3d.ApodizationSpec

~~~~
