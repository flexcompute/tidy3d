.. _impedance_calculator:

Impedance Calculator
--------------------

.. autosummary::
   :toctree: ../_autosummary/
   :template: module.rst

   tidy3d.ImpedanceCalculator

The :class:`~tidy3d.ImpedanceCalculator` computes characteristic impedance from electromagnetic field data using voltage and current path integrals. It supports three calculation methods depending on which integrals are provided:

* **V and I method**: :math:`Z_0 = V / I` (when both voltage and current integrals are provided)
* **P and V method**: :math:`Z_0 = |V|^2 / (2P^*)` (when only voltage integral is provided)
* **P and I method**: :math:`Z_0 = 2P / |I|^2` (when only current integral is provided)

where :math:`P = \frac{V I^*}{2}` is the complex power flow through the cross-section.

**Basic Usage**

.. code-block:: python

   import tidy3d as td

   # Define voltage integration path
   voltage_integral = td.AxisAlignedVoltageIntegral(
       center=(0, 0, 0),
       size=(0, 0, 2),  # Vertical line
       sign="+",
       extrapolate_to_endpoints=True,
       snap_path_to_grid=True
   )

   # Define current integration contour
   current_integral = td.AxisAlignedCurrentIntegral(
       center=(0, 0, 0),
       size=(4, 2, 0),  # Rectangular loop
       sign="+",
       snap_contour_to_grid=True
   )

   # Create impedance calculator
   # Note: The impedance calculator can also accept "None" for either the "voltage_integral" or the "current_integral",
   # which determines the method for computing the impedance. This alternative method is detailed below.
   Z_calculator = td.ImpedanceCalculator(
       voltage_integral=voltage_integral,
       current_integral=current_integral
   )

   # Compute impedance from mode data
   mode_data = # ... obtain from ModeSimulation or ModeSolver
   impedance = Z_calculator.compute_impedance(mode_data)

**Using with Mode Solver Data**

The impedance calculator is commonly used with mode solver results to determine the characteristic impedance of transmission line modes:

.. code-block:: python

   import tidy3d.web as web

   # Run mode simulation
   mode_sim = td.ModeSimulation(
       size=(10, 10, 0),
       grid_spec=td.GridSpec.auto(wavelength=0.3),
       structures=[...],
       monitors=[mode_monitor],
       freqs=[1e9, 2e9, 3e9]
   )

   mode_sim_data = web.run(mode_sim, task_name='mode_solver')
   # Calculate impedance for each mode
   Z0 = Z_calculator.compute_impedance(mode_sim_data.modes)

**Obtaining Voltage and Current**

You can also retrieve the voltage and current values along with impedance:

.. code-block:: python

   # Get impedance, voltage, and current
   Z, V, I = Z_calculator.compute_impedance(
       mode_data,
       return_voltage_and_current=True
   )

   print(f"Impedance: {Z} Ω")
   print(f"Voltage: {V} V")
   print(f"Current: {I} A")

**Single Integral Calculation**

When only voltage or current integral is specified, the complex power flow is automatically used:

.. code-block:: python

   # Calculator with only voltage integral
   Z_calc_V = td.ImpedanceCalculator(
       voltage_integral=voltage_integral,
       current_integral=None
   )

   # Computes: Z = V^2 / (2*P)
   Z_from_V = Z_calc_V.compute_impedance(mode_data)

   # Calculator with only current integral
   Z_calc_I = td.ImpedanceCalculator(
       voltage_integral=None,
       current_integral=current_integral
   )

   # Computes: Z = 2*P / I^2
   Z_from_I = Z_calc_I.compute_impedance(mode_data)

.. note::

   For detailed information on path integral classes (voltage integrals, current integrals, composite integrals, custom paths, etc.), see :ref:`path_integrals`.

**Field Data Compatibility**

The impedance calculator and path integral classes work with various types of field data:

* :class:`~tidy3d.ModeSolverData`: Mode field profiles from 2D mode solver
* :class:`~tidy3d.FieldData`: Frequency-domain field data from monitors
* :class:`~tidy3d.FieldTimeData`: Time-domain field data from monitors
* :class:`~tidy3d.rf.MicrowaveModeSolverData`: Microwave mode solver data (includes pre-computed integrals)

.. code-block:: python

   # Works with different data types
   Z_from_mode = Z_calculator.compute_impedance(mode_solver_data)
   Z_from_monitor = Z_calculator.compute_impedance(field_monitor_data)
   Z_from_time = Z_calculator.compute_impedance(field_time_data)

**Phase Convention**

.. note::

   Tidy3D uses the physics phase convention :math:`e^{-i\omega t}`. Some RF simulation software and textbooks use the electrical engineering convention :math:`e^{i\omega t}`. This affects calculated S-parameters and impedance values.

   To convert between conventions, use complex conjugation:

   .. code-block:: python

      import numpy as np

      # Convert from physics to engineering convention
      Z_engineering = np.conjugate(Z_physics)

      # Convert from engineering to physics convention
      Z_physics = np.conjugate(Z_engineering)

.. seealso::

   Related documentation:

   + :ref:`path_integrals`
   + :ref:`microwave_mode_solver`

   Tutorials and examples:

   + `Computing the characteristic impedance of transmission lines <../../notebooks/CharacteristicImpedanceCalculator.html>`_

~~~~
