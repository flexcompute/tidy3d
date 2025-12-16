Mode Source
-----------

.. autosummary::
   :toctree: _autosummary/
   :template: module.rst

   tidy3d.ModeSource
   tidy3d.ModeSpec
   tidy3d.PECFrame

The ``ModeSource`` class represents a propagating mode in a given structure cross section.

Behind the scenes, the desired mode is initially obtained by running the mode solver on the ``ModeSource`` plane. The ``ModeSpec`` class provides the settings for the mode solver. Once the desired mode is found, it is injected into the 3D simulation along the plane normal. The user does not need to invoke the mode solver explicitly; it is performed automatically whenever a ``ModeSource`` or ``ModeMonitor`` is included in a Tidy3D simulation.

.. code-block:: python

   # custom mode specification tells the mode solver to search for
   # two modes around effective index = 1.5
   my_mode_spec = ModeSpec(num_modes=2, target_neff=1.5)

   # custom mode source injects the first mode along the +x direction
   my_mode_source = ModeSource(
       size=(0,10,10),
       source_time=my_source_time,
       direction='+',
       mode_spec=my_mode_spec,
       mode_index=0,
   )
       
.. seealso::

   For more detailed explanation and examples, please see the following learning center resources:

   + `Defining mode sources and monitors <../notebooks/ModalSourcesMonitors.html>`_
   + `Injecting modes in bent and angled waveguides <../notebooks/ModeBentAngled.html>`_

   For a short introduction to the use of mode sources, see the following FDTD101 lecture:

   + `Mode injection <https://www.flexcompute.com/fdtd101/Lecture-4-Prelude-to-Integrated-Photonics-Simulation-Mode-Injection/>`_

