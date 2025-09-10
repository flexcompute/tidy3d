.. _wave_port:

Wave Port
---------

.. autosummary::
   :toctree: ../_autosummary/
   :template: module.rst

   tidy3d.plugins.smatrix.WavePort
   tidy3d.ModeSpec

The :class:`.WavePort` represents a modal source port. The port mode is first calculated in the 2D mode solver, then injected into the 3D simulation. The :class:`.WavePort` is also automatically terminated with a modal absorbing boundary :class:`.ModeABCBoundary` that perfectly absorbs the outgoing mode. Any non-matching modes are subject to PEC reflection.

.. code-block:: python

   my_wave_port_1 = WavePort(
       center=(0,0,0),
       size=(port_width, port_height, 0),
       name='My Wave Port 1',
       direction='+',  # direction of signal
       mode_spec=ModeSpec(target_neff=1.5),  # specification for mode solver
       current_integral=my_current_integral,  # current integration curve for port impedance calculation
   )

Most fields are self explanatory. Some additional notes:

* ``mode_spec`` is used to specify the effective index search value for the mode solver
* ``current_integral`` and/or ``voltage_integral`` are used to specify the integration paths for port impedance calculation. If only one of the two is specified, then the port power is also used (automatically determined).

If it is desired to only solve for the 2D port mode, one can use the ``to_mode_solver()`` convenience method to generate a :class:`.ModeSolver` simulation object.

.. code-block:: python

   # Define a mode solver from the wave port
   my_mode_solver = my_wave_port_1.to_mode_solver(
       simulation=base_sim,   # base Simulation object
       freqs=my_frequencies,   # frequencies for 2D mode solver
   )

   # Execute mode solver
   my_mode_data = web.run(my_mode_solver, task_name='mode solver')


.. autosummary::
   :toctree: ../_autosummary/
   :template: module.rst

   tidy3d.plugins.microwave.VoltageIntegralAxisAligned
   tidy3d.plugins.microwave.CurrentIntegralAxisAligned
   tidy3d.plugins.microwave.CustomVoltageIntegral2D
   tidy3d.plugins.microwave.CustomCurrentIntegral2D
   tidy3d.plugins.microwave.AxisAlignedPathIntegral
   tidy3d.plugins.microwave.CustomPathIntegral2D
   tidy3d.plugins.microwave.ImpedanceCalculator

The classes above are used to define the voltage/current integration paths for impedance calculation.

.. code-block:: python

   # Define voltage integration line
   my_voltage_integral = VoltageIntegralAxisAligned(
       center=(0,0,0),  # center of integration line
       size=(5, 0, 0),  # length of integration line
       sign='+',  # sign of integral
   )

   # Define current integration loop
   my_current_integral = CurrentIntegralAxisAligned(
       center=(0,0,0),  # center of integration loop
       size=(20, 20, 0),  # size of integration loop
       sign='+', # sign of integral (should match wave port direction)
   )

In addition to being used in the :class:`.WavePort` definition, the current/voltage integration objects can also be applied to arbitrary EM field data (2D and 3D). This is most commonly used in conjunction with the ``ImpedanceCalculator`` to calculate the line impedance of a 2D mode.

.. code-block:: python

   # Define impedance calculator
   my_Z_calculator = ImpedanceCalculator(
       voltage_integral = my_voltage_integral,
       current_integral = my_current_integral,
   )

   # Calculate impedance of 2D mode
   Z_mode = my_Z_calculator.compute_impedance(my_mode_data)

As before, only one of the two integration paths (voltage or current) are strictly necessary. This determines the convention used to calculate the impedance (PI, PV, or VI).

.. seealso::

   For more information, please see the following articles:

   + `Computing the characteristic impedance of transmission lines <../../notebooks/CharacteristicImpedanceCalculator.html>`_

   Example applications:

   + `Differential stripline benchmark <../../notebooks/DifferentialStripline.html>`_


~~~~
