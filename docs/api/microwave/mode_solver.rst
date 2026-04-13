.. currentmodule:: tidy3d

.. _microwave_mode_solver:

RF Mode Analysis
----------------

.. autosummary::
   :toctree: ../_autosummary/
   :template: module.rst

   rf.MicrowaveModeSpec
   rf.AutoImpedanceSpec
   rf.CustomImpedanceSpec
   rf.MicrowaveModeMonitor
   rf.MicrowaveModeSolverMonitor
   ModeSimulation

The :class:`~tidy3d.rf.MicrowaveModeSpec` extends the standard :class:`~tidy3d.ModeSpec` to include automatic characteristic impedance calculation for transmission line modes. This is particularly useful for microwave and RF applications where the characteristic impedance :math:`Z_0` is a critical parameter.

Unlike the standard :class:`~tidy3d.ModeSpec`, :class:`~tidy3d.rf.MicrowaveModeSpec` includes an ``impedance_specs`` field that defines how voltage and current path integrals are computed for each mode. These integrals are then used to calculate the characteristic impedance of the transmission line.

.. code-block:: python

   # Create a microwave mode specification with automatic impedance calculation
   mode_spec = MicrowaveModeSpec(
       num_modes=2,
       target_neff=1.5,
       impedance_specs=AutoImpedanceSpec()  # Automatic path integral setup
   )

   # Use in a microwave mode monitor
   monitor = MicrowaveModeMonitor(
       center=(0, 0, 0),
       size=(2, 2, 0),
       freqs=[1e9, 2e9, 3e9],
       mode_spec=mode_spec,
       name='mode_monitor'
   )

**Automatic Impedance Calculation**

The :class:`~tidy3d.rf.AutoImpedanceSpec` automatically determines appropriate voltage and current integration paths based on the mode field distribution. This is the recommended approach for most use cases, as it eliminates the need to manually define integration paths.

.. code-block:: python

   # Using automatic impedance specification (recommended)
   mode_spec_auto = MicrowaveModeSpec(
       num_modes=1,
       impedance_specs=AutoImpedanceSpec()
   )

**Custom Impedance Calculation**

For more control over the impedance calculation, you can specify custom voltage and current integration paths using :class:`~tidy3d.rf.CustomImpedanceSpec`. This allows you to define exactly where and how the voltage and current are computed.

.. code-block:: python

   # Define custom voltage path (line integral)
   voltage_spec = AxisAlignedVoltageIntegralSpec(
       center=(0, 0, 0),
       size=(0, 0, 1),  # Vertical line
       sign="+"
   )

   # Define custom current path (contour integral)
   current_spec = AxisAlignedCurrentIntegralSpec(
       center=(0, 0, 0),
       size=(2, 1, 0),  # Rectangular loop
       sign="+"
   )

   # Create custom impedance specification
   custom_impedance = CustomImpedanceSpec(
       voltage_spec=voltage_spec,
       current_spec=current_spec
   )

   # Use in mode specification
   mode_spec_custom = MicrowaveModeSpec(
       num_modes=1,
       impedance_specs=custom_impedance
   )

**Multiple Modes**

When solving for multiple modes, you can provide different impedance specifications for each mode:

.. code-block:: python

   # Different impedance specs for each mode
   mode_spec_multi = MicrowaveModeSpec(
       num_modes=2,
       impedance_specs=(
           AutoImpedanceSpec(),  # Auto for first mode
           custom_impedance,         # Custom for second mode
       )
   )

   # Or use a single spec for all modes (will be duplicated)
   mode_spec_shared = MicrowaveModeSpec(
       num_modes=2,
       impedance_specs=AutoImpedanceSpec()  # Applied to both modes
   )

**Mode Solver Monitors**

There are two types of monitors for microwave mode analysis:

* :class:`~tidy3d.rf.MicrowaveModeMonitor`: Records mode amplitudes and transmission line parameters (voltage, current, impedance) on a monitor plane during a full 3D FDTD simulation.
* :class:`~tidy3d.rf.MicrowaveModeSolverMonitor`: Stores the complete 2D mode field profiles along with transmission line parameters from a standalone mode solver calculation.

.. code-block:: python

   # Monitor for mode amplitudes in full simulation
   mode_monitor = MicrowaveModeMonitor(
       center=(0, 0, 0),
       size=(2, 2, 0),
       freqs=[1e9, 2e9],
       mode_spec=mode_spec,
       name='mode_monitor'
   )

   # Monitor for mode field profiles (mode solver only)
   mode_solver_monitor = MicrowaveModeSolverMonitor(
       center=(0, 0, 0),
       size=(2, 2, 0),
       freqs=[1e9, 2e9],
       mode_spec=mode_spec,
       name='mode_solver_monitor'
   )

**Mode Simulation**

For standalone mode solving (without a full 3D FDTD simulation), use :class:`~tidy3d.ModeSimulation`:

.. code-block:: python

   import tidy3d.web as web

   # Create a mode simulation for transmission line analysis
   mode_sim = ModeSimulation(
       size=(10, 10, 0),
       grid_spec=GridSpec.auto(wavelength=0.3),
       structures=[...],  # Your transmission line structures
       mode_spec=mode_spec,  # MicrowaveModeSpec defining impedance calculation
       freqs=[1e9, 2e9, 3e9],
   )

   # Run the mode simulation
   mode_sim_data = web.run(mode_sim, task_name='mode_analysis')

   # Access impedance results from the modes
   Z0 = mode_sim_data.modes.transmission_line_data.Z0

.. seealso::

   For more information on path integral specifications, see:

   + :ref:`path_integrals`
   + :ref:`impedance_calculator`
   + :ref:`wave_port`

   For practical examples:

   + `Computing the characteristic impedance of transmission lines <../../notebooks/CharacteristicImpedanceCalculator.html>`_
   + `Differential stripline benchmark <../../notebooks/DifferentialStripline.html>`_
   + `Edge-mounted SMA to co-planar waveguide transition <../../notebooks/SMAEdgeMount.html>`_

~~~~
