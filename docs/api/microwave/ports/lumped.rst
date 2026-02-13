.. _lumped_port:

Lumped Port & Elements
----------------------

.. autosummary::
   :toctree: ../_autosummary/
   :template: module.rst

   tidy3d.rf.LumpedPort
   tidy3d.rf.CoaxialLumpedPort

The :class:`~tidy3d.rf.LumpedPort` feature represents a planar, uniform current excitation with a fixed impedance termination.

.. code-block:: python

   # Define a lumped port
   my_port_1 = LumpedPort(
       name="My Port 1",
       center=(0,0,0),
       size=(0, port_width, port_height),
       voltage_axis=2,   # z-axis aligned excitation
       impedance=50,   # port impedance
   )

The :class:`~tidy3d.rf.LumpedPort` must be planar (exactly one zero-size dimension). Only axis-aligned planes are supported at this time. If you need a narrow port, provide a small but finite width along the lateral axis. Only real ``impedance`` values are supported at this time.

.. note::

   Lumped ports and elements are fundamentally approximations and thus should only be used when the port/element size is much smaller than the wavelength of interest (typically ``lambda/10``). For more accurate results, especially when the port is adjacent to an intentional waveguide or transmission line, consider using the :class:`~tidy3d.rf.WavePort` excitation instead.

The ``CoaxialLumpedPort`` represents an analytical coaxial field source.

.. code-block:: python

   # Define coaxial lumped port
   my_coaxial_port_1 = CoaxialLumpedPort(
       name="My Coaxial Port 1",
       center=(0,0,0),
       inner_diameter=1000,   # inner diameter in um
       outer_diameter=2000,   # outer diameter in um
       normal_axis=0,   # normal axis to port plane
       direction="+",   # direction of signal along normal axis
       impedance=50,   # port impedance
   )

.. note::

   Because the ``CoaxialLumpedPort`` injects an analytical field source, the structure connected to this port must match the physical port dimensions. Any deviation will result in signal reflection and potential inaccuracies. One common source of this issue is in imported geometries with faceted cylinders.

.. autosummary::
   :toctree: ../_autosummary/
   :template: module.rst

   tidy3d.rf.LumpedResistor
   tidy3d.rf.CoaxialLumpedResistor
   tidy3d.rf.LinearLumpedElement
   tidy3d.rf.RLCNetwork
   tidy3d.rf.AdmittanceNetwork

For a simple resistive lumped element, use ``LumpedResistor``.

.. code-block:: python

   my_resistor = LumpedResistor(
       name="My resistor",
       center=(0,0,0),
       size=(0, element_width, element_height),
       voltage_axis=2,   # z-axis aligned
       resistance=50,   # real-valued impedance
   )

For more complicated RLC networks, use the general ``LinearLumpedElement`` class.

.. code-block:: python

   my_lumped_element = LinearLumpedElement(
       name="My lumped element",
       center=(0,0,0),
       size=(0, element_width, element_height),
       voltage_axis=2,   # z-axis aligned
       network=RLCNetwork(resistance=50, inductance=1e-9)  # RLC network
   )

All lumped elements should be added to the ``lumped_elements`` field of the base :class:`~tidy3d.Simulation` instance.

.. code-block:: python

   my_simulation = Simulation(
       lumped_elements=[my_resistor, my_lumped_element],
       ...,
   )


.. seealso::

   For more in-depth discussion and examples, please see the following learning center article:

   + `Using lumped elements in Tidy3D simulations <../../notebooks/LinearLumpedElements.html>`_

   Example applications:

   + `Hybrid microstrip/co-planar waveguide bandpass filter <../../notebooks/HybridMicrostripCPWBandpassFilter.html>`_
   + `Designing a power divider (part 3) <../../notebooks/WPDHarmonicSuppression3.html>`_


~~~~
