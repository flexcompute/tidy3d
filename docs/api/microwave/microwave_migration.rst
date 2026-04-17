.. _microwave_migration:

v2.10 RF Refactor Migration Guide
---------------------------------

In version ``v2.10.0``, the microwave and RF simulation capabilities underwent significant refactoring to improve consistency, clarity, and functionality. This guide covers three major sets of breaking changes:

1. **Path Integral Class Renames** - Classes were renamed for consistency
2. **WavePort API Changes** - WavePort was refactored to support multiple modes with cleaner impedance specification
3. **Component Modeler Refactor** - ComponentModeler classes were refactored for improved web support

This guide helps you update your scripts to work with v2.10+.

1. Path Integral Class Renames
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Path integral classes were renamed for improved consistency and clarity. Additionally, these classes (and the impedance calculator) were refactored out of the plugin and re-exported at the top-level package for convenience.

**Key changes:**

*   **Renamed Classes**: Path integral classes have been renamed to follow a consistent naming pattern.
*   **New Import Path (simplified)**: The path integral classes and impedance calculator are now exported at the 'tidy3d.rf' sub-package . Prefer importing directly from ``tidy3d.rf`` (e.g., ``from tidy3d.rf import AxisAlignedVoltageIntegral, ImpedanceCalculator``). Existing plugin imports continue to work for backwards compatibility where applicable.

Class Name Changes
^^^^^^^^^^^^^^^^^^

The following classes have been renamed for consistency:

**Voltage Integrals:**

*   ``VoltageIntegralAxisAligned`` → ``AxisAlignedVoltageIntegral``
*   ``CustomVoltageIntegral2D`` → ``Custom2DVoltageIntegral``

**Current Integrals:**

*   ``CurrentIntegralAxisAligned`` → ``AxisAlignedCurrentIntegral``
*   ``CustomCurrentIntegral2D`` → ``Custom2DCurrentIntegral``

**Path Integrals:**

*   ``CustomPathIntegral2D`` → ``Custom2DPathIntegral``

Migration Examples
^^^^^^^^^^^^^^^^^^

**Before (v2.9.x):**

.. code-block:: python

    from tidy3d.plugins.microwave import (
        VoltageIntegralAxisAligned,
        CurrentIntegralAxisAligned,
    )

    voltage_path = VoltageIntegralAxisAligned(
        center=(0, 0, 0),
        size=(0, 0, 1),
        sign="+",
    )

    current_path = CurrentIntegralAxisAligned(
        center=(0, 0, 0),
        size=(2, 1, 0),
        sign="+",
    )

**After (v2.10+):**

.. code-block:: python

    from tidy3d.rf import (
        AxisAlignedVoltageIntegral,
        AxisAlignedCurrentIntegral,
    )

    voltage_path = AxisAlignedVoltageIntegral(
        center=(0, 0, 0),
        size=(0, 0, 1),
        sign="+",
    )

    current_path = AxisAlignedCurrentIntegral(
        center=(0, 0, 0),
        size=(2, 1, 0),
        sign="+",
    )

Custom 2D Path Integrals
""""""""""""""""""""""""

**Before:**

.. code-block:: python

    from tidy3d.plugins.microwave import (
        CustomVoltageIntegral2D,
        CustomCurrentIntegral2D,
    )

    vertices = [[0, 0], [1, 0], [1, 1], [0, 1]]

    voltage_path = CustomVoltageIntegral2D(
        axis=2,
        position=0.0,
        vertices=vertices,
    )

    current_path = CustomCurrentIntegral2D(
        axis=2,
        position=0.0,
        vertices=vertices,
    )

**After:**

.. code-block:: python

    from tidy3d.rf import (
        Custom2DVoltageIntegral,
        Custom2DCurrentIntegral,
    )

    vertices = [[0, 0], [1, 0], [1, 1], [0, 1]]

    voltage_path = Custom2DVoltageIntegral(
        axis=2,
        position=0.0,
        vertices=vertices,
    )

    current_path = Custom2DCurrentIntegral(
        axis=2,
        position=0.0,
        vertices=vertices,
    )

Summary
^^^^^^^

All functionality remains the same—only class names and preferred import paths have changed. Update your imports to the top level (``from tidy3d.rf import ...``) and class names according to the table above, and your code will work with v2.10.

2. WavePort API Changes for Multimodal Support
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``WavePort`` class was refactored to support multiple modes and to provide cleaner integration with the microwave mode solver. This required several breaking changes to the API.

Overview of Breaking Changes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The WavePort refactor introduces the following breaking changes:

*   **Removed fields**: ``voltage_integral`` and ``current_integral`` fields are removed from ``WavePort``
*   **Impedance specification moved**: Voltage and current path integrals are now specified via ``MicrowaveModeSpec.impedance_specs`` instead of directly on the port
*   **ModeSpec type changed**: ``mode_spec`` field now requires ``MicrowaveModeSpec`` instead of generic ``ModeSpec``
*   **Method renamed**: ``compute_port_impedance()`` renamed to ``get_port_impedance()`` with new signature
*   **Return shapes changed**: ``compute_voltage()`` and ``compute_current()`` now return data with a ``mode_index`` dimension
*   **Deprecated field**: ``mode_index`` field is deprecated (still works but triggers warning)

Removed Fields
^^^^^^^^^^^^^^

The following fields have been **removed** from ``WavePort``:

*   ``voltage_integral: Optional[VoltageIntegralType]`` - Path integral for voltage calculation
*   ``current_integral: Optional[CurrentIntegralType]`` - Path integral for current calculation

Deprecated Fields
^^^^^^^^^^^^^^^^^

*   ``mode_index: Optional[int]`` - **Deprecated** field that previously specified which single mode to use (default was 0 in v2.9). Still works for backward compatibility but triggers a deprecation warning. Will be removed in a future version. **Migration**: For backwards compatible code, omit this field entirely. For forward compatible code, use ``mode_selection`` instead if you need to select specific modes.

New MicrowaveModeSpec Integration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``mode_spec`` field now requires a ``MicrowaveModeSpec`` (instead of the generic ``ModeSpec``). This new class includes:

*   ``impedance_specs``: Defines how to compute voltage, current, and characteristic impedance for each mode

Migration Examples
^^^^^^^^^^^^^^^^^^

Single-Mode WavePort
""""""""""""""""""""

**Before (v2.9.x):**

.. code-block:: python

    # Define path integrals
    voltage_path = AxisAlignedVoltageIntegralSpec(
        center=(0, 0, 0),
        size=(1.0, 0, 0),
        sign="+",
    )

    current_path = Custom2DCurrentIntegralSpec.from_circular_path(
        center=(0, 0, 0),
        radius=0.5,
        num_points=21,
        normal_axis=2,
        clockwise=False
    )

    # Create WavePort - path integrals attached directly to port
    port = WavePort(
        center=(0, 0, -5),
        size=(2, 2, 0),
        direction="+",
        mode_spec=ModeSpec(num_modes=1),  # Generic ModeSpec
        mode_index=0,  # Which mode to excite
        voltage_integral=voltage_path,  # Attached to port
        current_integral=current_path,  # Attached to port
        name="port1"
    )

**After (v2.10+):**

.. code-block:: python

    # Define path integrals (same as before)
    voltage_path = AxisAlignedVoltageIntegralSpec(
        center=(0, 0, 0),
        size=(1.0, 0, 0),
        sign="+",
    )

    current_path = Custom2DCurrentIntegralSpec.from_circular_path(
        center=(0, 0, 0),
        radius=0.5,
        num_points=21,
        normal_axis=2,
        clockwise=False
    )

    # Create WavePort - path integrals now in MicrowaveModeSpec
    port = WavePort(
        center=(0, 0, -5),
        size=(2, 2, 0),
        direction="+",
        mode_spec=MicrowaveModeSpec(  # Use MicrowaveModeSpec
            num_modes=1,
            impedance_specs=CustomImpedanceSpec(
                voltage_spec=voltage_path,  # Moved to impedance_specs
                current_spec=current_path
            )
        ),
        name="port1"
        # Note: mode_index, voltage_integral, current_integral removed
    )

    # Mode selection now happens at source creation
    source_time = GaussianPulse(freq0=10e9, fwidth=1e9)
    source = port.to_source(source_time, mode_index=0)  # Mode selected here

Multi-Mode WavePort (New Feature!)
"""""""""""""""""""""""""""""""""""

The new API enables WavePorts to support multiple modes simultaneously:

.. code-block:: python

    # Create a 3-mode WavePort
    port = WavePort(
        center=(0, 0, -5),
        size=(4, 4, 0),
        direction="+",
        mode_spec=MicrowaveModeSpec(
            num_modes=3,  # Solve for 3 modes
            impedance_specs=AutoImpedanceSpec()  # Auto-compute impedance for all modes
        ),
        name="multimode_port"
    )

    # Create sources for different modes
    source_time = GaussianPulse(freq0=10e9, fwidth=1e9)
    source_mode0 = port.to_source(source_time, mode_index=0)  # Excite mode 0
    source_mode1 = port.to_source(source_time, mode_index=1)  # Excite mode 1
    source_mode2 = port.to_source(source_time, mode_index=2)  # Excite mode 2

Per-Mode Impedance Specifications
""""""""""""""""""""""""""""""""""

For advanced use cases, you can specify different impedance calculation methods for each mode:

.. code-block:: python

    # Define custom specs for mode 0, auto for modes 1 and 2
    port = WavePort(
        center=(0, 0, -5),
        size=(4, 4, 0),
        direction="+",
        mode_spec=MicrowaveModeSpec(
            num_modes=3,
            impedance_specs=(
                CustomImpedanceSpec(
                    voltage_spec=custom_voltage_path,
                    current_spec=custom_current_path
                ),  # Mode 0 uses custom specs
                AutoImpedanceSpec(),  # Mode 1 uses auto
                AutoImpedanceSpec(),  # Mode 2 uses auto
            )
        ),
        name="mixed_impedance_port"
    )

Method Changes
^^^^^^^^^^^^^^

``compute_port_impedance()`` → ``get_port_impedance()``
""""""""""""""""""""""""""""""""""""""""""""""""""""""""

The method for retrieving port impedance has been renamed and now requires a ``mode_index`` parameter:

**Before (v2.9.x):**

.. code-block:: python

    # Get impedance - implicitly used port.mode_index
    Z0 = port.compute_port_impedance(sim_data)

**After (v2.10+):**

.. code-block:: python

    # Get impedance for mode 0
    Z0_mode0 = port.get_port_impedance(sim_data, mode_index=0)

    # Get impedance for mode 1 (multimodal port)
    Z0_mode1 = port.get_port_impedance(sim_data, mode_index=1)

Voltage and Current Computation Shape Changes
""""""""""""""""""""""""""""""""""""""""""""""

For multimodal ports, ``compute_voltage()`` and ``compute_current()`` now return data for **all modes** with a ``mode_index`` dimension:

**Before (v2.9.x):**

.. code-block:: python

    # Returns shape: (n_freqs,) - single mode only
    voltage = port.compute_voltage(sim_data)
    current = port.compute_current(sim_data)

**After (v2.10+):**

.. code-block:: python

    # For single-mode port: Returns shape: (n_freqs, 1)
    # For multi-mode port: Returns shape: (n_freqs, n_modes)
    voltage = port.compute_voltage(sim_data)
    current = port.compute_current(sim_data)

    # Select specific mode using xarray selection
    voltage_mode0 = voltage.sel(mode_index=0)
    voltage_mode1 = voltage.sel(mode_index=1)

Using AutoImpedanceSpec for Simplified Setup
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For most cases, you can use ``AutoImpedanceSpec`` which automatically computes voltage, current, and impedance from the electromagnetic fields:

.. code-block:: python

    # Simplest form - let Tidy3D auto-compute everything
    port = WavePort(
        center=(0, 0, -5),
        size=(2, 2, 0),
        direction="+",
        mode_spec=MicrowaveModeSpec(
            num_modes=2,
            impedance_specs=AutoImpedanceSpec()  # Works for all modes
        ),
        name="simple_port"
    )

Breaking Changes Summary
^^^^^^^^^^^^^^^^^^^^^^^^

The following table summarizes all breaking changes to the ``WavePort`` API:

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Change
     - Old (v2.9.x)
     - New (v2.10+)
   * - Port field: mode_index
     - Required ``int`` (default: 0)
     - **Deprecated** ``Optional[int]``. Omit this field.
   * - Port field: mode_selection
     - Did not exist
     - **NEW** ``Optional[tuple[int, ...]]`` to select specific modes (default: ``None`` = all modes)
   * - Port field: voltage_integral
     - ``voltage_integral=path``
     - Removed. Use ``mode_spec.impedance_specs``
   * - Port field: current_integral
     - ``current_integral=path``
     - Removed. Use ``mode_spec.impedance_specs``
   * - mode_spec type
     - ``ModeSpec``
     - ``MicrowaveModeSpec``
   * - Impedance method
     - ``compute_port_impedance(sim_data)``
     - ``get_port_impedance(sim_data, mode_index)``
   * - Monitor type
     - Returns ``ModeMonitor``
     - Returns ``MicrowaveModeMonitor``
   * - Voltage/current shape
     - ``(n_freqs,)``
     - ``(n_freqs, n_modes)`` - added mode_index dimension
   * - Multimodal support
     - Not supported
     - Fully supported via ``mode_spec.num_modes``

Benefits of the New API
^^^^^^^^^^^^^^^^^^^^^^^^

The refactored API provides several advantages:

*   **Multimodal ports**: Support for multiple modes enables more accurate modeling of multimode waveguides and transmission lines
*   **Cleaner separation of concerns**: Mode selection happens at excitation time, not port definition time
*   **Type safety**: ``MicrowaveModeSpec`` and ``MicrowaveModeMonitor`` make RF-specific behavior explicit
*   **Flexibility**: Per-mode impedance specifications allow fine-grained control
*   **Consistency**: Aligns with the general pattern of ``ModeSource`` where ``mode_index`` is a source parameter

Backward Compatibility
^^^^^^^^^^^^^^^^^^^^^^

There is **no backward compatibility** for WavePort instantiation with the old field names (``voltage_integral``, ``current_integral``). Attempting to use these fields will result in a Pydantic validation error.

The ``mode_index`` field is **deprecated** but still functional for backward compatibility:

*   In v2.9: Required ``int`` field (default: 0) specifying which single mode to use
*   In v2.10+: Optional ``int`` field (still accepts single integer only) but triggers a deprecation warning
*   **Migration**: Simply omit the ``mode_index`` field from your WavePort definitions

The new ``mode_selection`` field replaces ``mode_index`` for selecting modes:

*   ``mode_selection: Optional[tuple[int, ...]]`` - New field that accepts a tuple of mode indices (e.g., ``(0, 2)`` to use modes 0 and 2)
*   When ``None`` (default), all modes from ``mode_spec.num_modes`` are used
*   **Note**: ``mode_index`` accepts only a single ``int``, while ``mode_selection`` accepts a tuple for multiple modes

3. Component Modeler Refactor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``smatrix`` plugin classes (``ComponentModeler``, ``TerminalComponentModeler``, etc.) were refactored to improve web and GUI support for RF capabilities. This section helps you update your scripts to the new, more robust API.

.. note::

   This content is also available as a standalone guide: :ref:`smatrix_migration`

.. include:: /api/plugins/smatrix_migration.rst
   :start-line: 12
