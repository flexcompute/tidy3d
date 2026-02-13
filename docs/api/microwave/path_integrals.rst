.. _path_integrals:

Path Integrals
--------------

Path integrals compute voltage and current from electromagnetic field data by integrating electric and magnetic fields along specified paths. These are essential for characterizing transmission lines and computing characteristic impedance.

.. note::

   For information on how path integrals are used to calculate characteristic impedance in transmission line mode analysis, see :ref:`microwave_mode_solver`. For the corresponding specification classes (``*Spec``) used in :class:`~tidy3d.rf.MicrowaveModeSpec`, see the end of this page.

Voltage Path Integrals
~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: ../_autosummary/
   :template: module.rst

   tidy3d.rf.AxisAlignedVoltageIntegral
   tidy3d.rf.Custom2DVoltageIntegral

Voltage path integrals compute voltage by integrating the electric field :math:`\mathbf{E}` along a line path:

.. math::

   V = -\int_{\text{path}} \mathbf{E} \cdot d\mathbf{l}

**AxisAlignedVoltageIntegral**

Integrates along an axis-aligned path, suitable for transmission lines with dominant electric field directions.

.. code-block:: python

   import tidy3d.web as web

   # Create voltage path integral
   voltage_integral = AxisAlignedVoltageIntegral(
       center=(0, 0, 0),
       size=(0, 0, 2.0),  # Line from (0,0,-1) to (0,0,1)
       sign="+",
       extrapolate_to_endpoints=True,  # Extrapolate field to path endpoints
       snap_path_to_grid=True          # Snap path to simulation grid
   )

   # Compute voltage from mode data
   mode_data = web.run(mode_solver, task_name='mode_solver')
   voltage = voltage_integral.compute_voltage(mode_data)

   print(f"Voltage: {voltage.values} V")

The ``sign`` parameter determines integration direction: ``"+"`` integrates toward positive axis, ``"-"`` toward negative.

**Custom2DVoltageIntegral**

Integrates along an arbitrary 2D path for non-standard geometries.

.. code-block:: python

   import numpy as np

   # Define custom path vertices in 2D plane
   vertices = np.array([
       [0, 0],    # Start point
       [0, 1],    # Intermediate point
       [1, 1],    # End point
   ])

   # Create custom voltage integral
   custom_voltage = Custom2DVoltageIntegral(
       axis=2,         # Normal axis (z)
       position=0.0,   # Position along z-axis
       vertices=vertices
   )

   # Compute voltage
   voltage = custom_voltage.compute_voltage(mode_data)

The path follows the vertices in order, integrating :math:`\mathbf{E} \cdot d\mathbf{l}` along each segment.

Current Path Integrals
~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: ../_autosummary/
   :template: module.rst

   tidy3d.rf.AxisAlignedCurrentIntegral
   tidy3d.rf.Custom2DCurrentIntegral
   tidy3d.rf.CompositeCurrentIntegral

Current path integrals compute current using Ampère's circuital law by integrating the magnetic field :math:`\mathbf{H}` around a closed contour:

.. math::

   I = \oint_{\text{contour}} \mathbf{H} \cdot d\mathbf{l}

**AxisAlignedCurrentIntegral**

Integrates around an axis-aligned rectangular loop, the most common approach for transmission lines.

.. code-block:: python

   # Create current path integral
   current_integral = AxisAlignedCurrentIntegral(
       center=(0, 0, 0),
       size=(3, 2, 0),  # Loop dimensions in xy-plane
       sign="+",
       snap_contour_to_grid=True  # Snap contour to grid for accuracy
   )

   # Compute current from mode data
   current = current_integral.compute_current(mode_data)

   print(f"Current: {current.values} A")

The rectangular loop is automatically constructed as four line segments. The ``sign`` parameter determines circulation direction and should match the power flow direction.

**Custom2DCurrentIntegral**

Integrates around an arbitrary closed 2D contour for non-standard geometries.

.. code-block:: python

   # Define closed contour (counterclockwise for positive current)
   vertices = np.array([
       [0, 0],
       [2, 0],
       [2, 1],
       [0, 1],
       [0, 0],  # Close the loop
   ])

   # Create custom current integral
   custom_current = Custom2DCurrentIntegral(
       axis=2,         # Normal axis
       position=0.0,   # Position along normal axis
       vertices=vertices
   )

   # Compute current
   current = custom_current.compute_current(mode_data)

For positive current in the positive ``axis`` direction, order vertices counterclockwise when viewed from the positive axis.

**CompositeCurrentIntegral**

Combines multiple current paths for complex geometries like differential lines.

.. code-block:: python

   # Define two separate current paths
   path_1 = AxisAlignedCurrentIntegral(
       center=(-2, 0, 0),
       size=(1, 1, 0),
       sign="+"
   )

   path_2 = AxisAlignedCurrentIntegral(
       center=(2, 0, 0),
       size=(1, 1, 0),
       sign="+"
   )

   # Combine paths
   composite_current = CompositeCurrentIntegral(
       path_specs=(path_1, path_2),
       sum_spec="sum"  # "sum" adds currents, "split" keeps them separate
   )

   # Compute combined current
   current = composite_current.compute_current(mode_data)

The ``sum_spec`` parameter controls combination:

* ``"sum"``: Adds all currents together
* ``"split"``: Returns maximum of phase-separated contributions (useful for identifying dominant current directions)

Usage with ImpedanceCalculator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Path integrals are commonly used with :class:`~tidy3d.ImpedanceCalculator` to compute characteristic impedance:

.. code-block:: python

   # Create impedance calculator from voltage and current integrals
   Z_calculator = ImpedanceCalculator(
       voltage_integral=voltage_integral,
       current_integral=current_integral
   )

   # Compute impedance: Z = V / I
   Z0 = Z_calculator.compute_impedance(mode_data)

   # Or get voltage, current, and impedance together
   Z, V, I = Z_calculator.compute_impedance(
       mode_data,
       return_voltage_and_current=True
   )

See :ref:`impedance_calculator` for more details.

Additional Information
~~~~~~~~~~~~~~~~~~~~~~

**Best Practices**

* For voltage integrals, the path should follow electric field lines between conductors
* For current integrals, the contour should enclose the current-carrying region
* Use ``snap_path_to_grid=True`` and ``snap_contour_to_grid=True`` for improved accuracy
* Use ``extrapolate_to_endpoints=True`` for voltage paths to better capture fields at conductor boundaries
* Ensure the ``sign`` parameter matches the desired power flow direction

**Path Integral Specification Classes**

The classes documented above (``*Integral``) are **execution classes** that perform actual computations on field data. For use in :class:`~tidy3d.rf.MicrowaveModeSpec` impedance specifications, corresponding **specification classes** (``*Spec``) exist:

.. autosummary::
   :toctree: ../_autosummary/
   :template: module.rst

   tidy3d.rf.AxisAlignedVoltageIntegralSpec
   tidy3d.rf.Custom2DVoltageIntegralSpec
   tidy3d.rf.AxisAlignedCurrentIntegralSpec
   tidy3d.rf.Custom2DCurrentIntegralSpec
   tidy3d.rf.CompositeCurrentIntegralSpec

These specification classes have the same parameters but are used for configuration (in :class:`~tidy3d.rf.CustomImpedanceSpec`) rather than direct computation. See :ref:`microwave_mode_solver` for their usage in mode analysis.

.. seealso::

   **Related documentation:**

   + :ref:`microwave_mode_solver` - Using path integrals for characteristic impedance calculation
   + :ref:`impedance_calculator` - ImpedanceCalculator class for post-processing

   **Practical examples:**

   + `Computing the characteristic impedance of transmission lines <../../notebooks/CharacteristicImpedanceCalculator.html>`_
   + `Differential stripline benchmark <../../notebooks/DifferentialStripline.html>`_

~~~~
