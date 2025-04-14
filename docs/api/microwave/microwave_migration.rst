.. _microwave_migration:

v2.10 Refactor Migration
-------------------------

In version ``v2.10.0``, microwave plugin path integral classes were renamed for improved consistency and clarity. Additionally, these classes (and the impedance calculator) were refactored out of the plugin and moved into ``tidy3d.components.microwave`` and re-exported at the top-level package for convenience. This guide helps you update your scripts to use the new class names and imports.

Key Changes
~~~~~~~~~~~

*   **Renamed Classes**: Path integral classes have been renamed to follow a consistent naming pattern.
*   **New Import Path (simplified)**: The path integral classes and impedance calculator are now exported at the top level. Prefer importing directly from ``tidy3d`` (e.g., ``from tidy3d import AxisAlignedVoltageIntegral, ImpedanceCalculator``). Existing plugin imports continue to work for backwards compatibility where applicable.

Class Name Changes
~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~

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

    from tidy3d import (
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
~~~~~~~~~~~~~~~~~~~~~~~~~

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

    from tidy3d import (
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
~~~~~~~

All functionality remains the same—only class names and preferred import paths have changed. Update your imports to the top level (``from tidy3d import ...``) and class names according to the table above, and your code will work with v2.10. For impedance calculations, import ``ImpedanceCalculator`` directly via ``from tidy3d import ImpedanceCalculator``.
