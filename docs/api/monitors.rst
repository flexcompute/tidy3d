
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
       interval=50,    # number of solver time steps between each measurement
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

Behind the scenes, a mode solver simulation is first performed to determine the eigenmodes in the ``ModeMonitor`` plane. A ``ModeSpec`` instance is necessary to provide the settings for this calculation. Then, the resulting modes are used to calculate the mode coefficients. The user does not need to explicitly perform the mode solver simulation, as it is automatically performed whenever a ``ModeSource`` or ``ModeMonitor`` is present in the simulation.

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

   + `Defining mode sources and monitors <../notebooks/ModalSourcesMonitors.html>`_

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

.. code-block:: python

   # define a diffraction monitor at 20 freq points between 250 and 300 THz
   # for outgoing fields in the +z direction
   my_diffraction_monitor = DiffractionMonitor(
       center=(0,0,10),
       size=(td.inf, td.inf, 0),    # extend monitor to simulation edges
       freqs=np.linspace(250e12, 300e12, 20),
       name='My diffraction monitor',
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
   tidy3d.AuxFieldTimeMonitor

The far-field monitor records the near-field within the simulation domain in order to project it to some far away location. This can be a very efficient way to simulate the scattering or radiative response of devices such as lenses and antenna.

.. code-block:: python

   # define a far-field monitor
   my_far_field_monitor = FieldProjectionCartesianMonitor(
       center=(0,0,10),
       size=(td.inf, td.inf, 0),
       name="My far field",
       freqs=[f0],
       far_field_approx=True,
       proj_axis=2,
       proj_distance=20,
       x=np.linspace(-10,10,101),
       y=np.linspace(-10,10,101),
   )

The ``far_field_approx`` parameter should be set to ``True`` (default) when the projection plane is far from the simulation domain. For intermediate distances, the user can set it to ``False`` for increased accuracy at the expense of slightly greater computational cost. 

When including far-field projection monitors in the ``Simulation`` object, the far-field calculation is performed server-side. The computation is much faster and slightly more accurate than if performed locally. However, it will slightly increase the cost of the simulation.

.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   tidy3d.FieldProjector
   tidy3d.FieldProjectionSurface

The user also has the option of performing the far-field calculation on their local machine using ``FieldProjector`` object. In that case, they should first set up one or more ``FieldMonitor`` in the simulation. After the simulation is completed, the ``FieldProjector`` object operates on the recorded field data.

.. code-block:: python

   # define a far-field projector from previously defined field monitors
   my_far_field_projector = FieldProjector.from_near_field_monitors(
       sim_data = simulation_data,    # previously computed simulation data
       near_monitors = [monitor1],    # list of previously defined near field monitors
       normal_dirs=['+'],             # projection direction relative to monitor
       pts_per_wavelength=10          # controls sampling rate (reduce to speed up computation)
   )

   # project field based on previously defined far-field monitor
   projected_field = my_far_field_projector.project_fields(my_far_field_monitor)

Please see the learning center article below for detailed explanations on additional settings and usage scenarios. 

.. seealso::

   For more details and examples, please refer to the following learning center article:

   + `Performing near field to far field projections <../notebooks/FieldProjections.html>`_

   Example applications:

   + `Mid-IR metalens based on silicon nanopillars <../notebooks/MidIRMetalens.html>`_
   + `Spherical Fresnel lens <../notebooks/FresnelLens.html>`_
   + `Directivity and S-parameters computation of patch antenna <../notebooks/AntennaCharacteristics.html>`_

~~~~

Permittivity
------------

.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   tidy3d.PermittivityMonitor

The ``PermittivityMonitor`` is used to record local relative permittivity data in the region of interest. This data can be useful for post-simulation calculations that require permittivity values, such as mode volume and absorption density.

.. code-block:: python

   my_permittivity_monitor = PermittivityMonitor(
       center=(0,0,0),
       size=(8,8,8),
       freqs=[250e12, 300e12],
       name='my permittivity monitor',
   )

.. seealso::

   Example applications:

   + `Bistability in photonic crystal microcavities <../notebooks/BistablePCCavity.html>`_

~~~~

Apodization
-----------

.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   tidy3d.ApodizationSpec

The ``ApodizationSpec`` is used to specify apodization specifications for frequency-domain monitors. Typically, the default Tidy3D settings are acceptable and it is not necessary to define a custom instance. Please refer to the documentation page for more details. 

~~~~
