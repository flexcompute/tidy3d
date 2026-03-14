.. _wave_port:

Wave Port
---------

.. autosummary::
   :toctree: ../_autosummary/
   :template: module.rst

   tidy3d.rf.WavePort
   tidy3d.rf.TerminalWavePort

The :class:`~tidy3d.rf.WavePort` represents a modal source port for RF and microwave simulations. The port mode is first calculated in the 2D mode solver with automatic characteristic impedance calculation, then injected into the 3D simulation. The :class:`~tidy3d.rf.WavePort` is also automatically terminated with a modal absorbing boundary :class:`~tidy3d.ModeABCBoundary` that perfectly absorbs the outgoing mode. Any non-matching modes are subject to PEC reflection.

**Basic Usage**

.. code-block:: python

   my_wave_port_1 = WavePort(
       center=(0,0,0),
       size=(port_width, port_height, 0),
       name='My Wave Port 1',
       direction='+',  # direction of signal
       mode_spec=MicrowaveModeSpec(
           num_modes=1,
           target_neff=1.5,
           impedance_specs=AutoImpedanceSpec()  # automatic impedance calculation
       ),
   )

Key parameters:

* ``mode_spec`` uses :class:`~tidy3d.rf.MicrowaveModeSpec` to specify mode solver settings and impedance calculation
* ``impedance_specs`` within :class:`~tidy3d.rf.MicrowaveModeSpec` defines how voltage, current, and characteristic impedance are computed. Use :class:`~tidy3d.rf.AutoImpedanceSpec` for automatic calculation (recommended) or :class:`~tidy3d.rf.CustomImpedanceSpec` for manual control.


**Multimode WavePort Support**

WavePorts can support multiple modes simultaneously. This is useful for multimode waveguides and transmission lines.

.. code-block:: python

   # Create a WavePort that solves for 3 modes
   multimode_port = WavePort(
       center=(0, 0, 0),
       size=(4, 4, 0),
       direction='+',
       mode_spec=MicrowaveModeSpec(
           num_modes=3,  # solve for 3 modes
           impedance_specs=AutoImpedanceSpec()  # applied to all modes
       ),
       name='multimode_port'
   )

When creating sources from a multimode port, specify which mode to excite:

.. code-block:: python

   source_time = GaussianPulse(freq0=10e9, fwidth=1e9)

   # Create sources for different modes
   source_mode0 = multimode_port.to_source(source_time, mode_index=0)
   source_mode1 = multimode_port.to_source(source_time, mode_index=1)
   source_mode2 = multimode_port.to_source(source_time, mode_index=2)

You can also specify different impedance calculations for each mode:

.. code-block:: python

   mode_spec = MicrowaveModeSpec(
       num_modes=2,
       impedance_specs=(
           CustomImpedanceSpec(...),  # custom for mode 0
           AutoImpedanceSpec(),       # auto for mode 1
       )
   )

**Mode Solver**

If you need to solve for the 2D port mode without running a full 3D simulation, use the ``to_mode_solver()`` convenience method:

.. code-block:: python

   import tidy3d.web as web

   # Define a mode solver from the wave port
   my_mode_solver = my_wave_port_1.to_mode_solver(
       simulation=base_sim,   # base Simulation object
       freqs=my_frequencies,   # frequencies for 2D mode solver
   )

   # Execute mode solver
   my_mode_data = web.run(my_mode_solver, task_name='mode solver')

The resulting ``my_mode_data`` will be :class:`~tidy3d.rf.MicrowaveModeSolverData` containing mode fields and transmission line parameters (characteristic impedance, voltage/current coefficients).

**Accessing Transmission Line Data**

After running a simulation with :class:`~tidy3d.rf.WavePort`, you can access the transmission line characteristics from the mode data:

.. code-block:: python

   # Get mode data from TerminalComponentModeler results
   mode_data = tcm_data.data['port1']['mode_monitor']

   # Access characteristic impedance for each mode
   Z0_mode0 = mode_data.transmission_line_data.Z0.sel(mode_index=0)

   # Get voltage and current coefficients
   voltage_coeff = mode_data.transmission_line_data.voltage_coeffs.sel(mode_index=0)
   current_coeff = mode_data.transmission_line_data.current_coeffs.sel(mode_index=0)

Alternatively, use the port's ``get_port_impedance()`` method:

.. code-block:: python

   # Get impedance directly from port
   Z0 = my_wave_port_1.get_port_impedance(mode_data, mode_index=0)

.. seealso::

   **Related Documentation:**

   + :ref:`microwave_mode_solver` - MicrowaveModeSpec and transmission line mode analysis
   + :ref:`microwave_migration` - Migration guide for API changes

   **Tutorials and Examples:**

   + `Differential stripline benchmark <../../notebooks/DifferentialStripline.html>`_
   + `Coplanar waveguide RF photonics <../../notebooks/CPWRFPhotonics1.html>`_
   + `Through silicon via <../../notebooks/ThroughSiliconVia.html>`_

**Terminal Wave Port**

The :class:`~tidy3d.rf.TerminalWavePort` is a terminal-driven wave port that supports
single-ended and differential terminal excitations.

.. code-block:: python

   my_terminal_wave_port = TerminalWavePort(
       center=(0, 0, 0),
       size=(4, 4, 0),
       name='My Terminal Wave Port',
       direction='+',
       differential_pairs=(("T0", "T1"),),
   )

~~~~
