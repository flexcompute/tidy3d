.. currentmodule:: tidy3d

Simulation
==========

At the heart of Tidy3D is the :class:`Simulation` class,
which represents a complete electromagnetic simulation using the Finite-Difference Time-Domain (FDTD) method.
The :class:`Simulation` object encapsulates all the information needed to run an electromagnetic simulation, 
including the geometry, materials, sources, monitors, and simulation parameters.

Working alongside the :class:`Simulation` class is the :class:`RunTimeSpec` class,
which provides a way to automatically determine an appropriate simulation duration based on the physical characteristics of your system.


Basic Usage
-----------

Creating a simulation involves defining the simulation domain, adding structures, sources, and monitors, and specifying simulation parameters:

.. code-block:: python

    import tidy3d as td
    
    # Create a basic simulation
    sim = td.Simulation(
        size=(3, 3, 3),                # Size of the simulation domain
        grid_spec=td.GridSpec.uniform(dl=0.1),  # Grid specification
        run_time=1e-12,                # Simulation duration
        medium=td.Medium(permittivity=1.0)      # Background medium
    )

All Tidy3D components apart from data structures are fully immutable, including the :class:`Simulation` object.
This means that once created, a simulation object cannot be modified directly. Instead, you can create updated copies:

.. code-block:: python

    # Add a structure to the simulation
    box = td.Structure(
        geometry=td.Box(size=(1, 1, 1), center=(0, 0, 0)),
        medium=td.Medium(permittivity=4.0)
    )
    sim = sim.updated_copy(structures=[box])


Simulation Duration
-------------------

The duration of a simulation is a critical parameter that affects both accuracy and computational cost. Tidy3D offers two approaches to specify this duration:

1. **Direct specification**: Provide a specific time value in seconds.

   .. code-block:: python
   
       sim = td.Simulation(
           # ... other parameters ...
           run_time=1e-12  # 1 picosecond
       )

2. **Automatic calculation** using :class:`RunTimeSpec`: This approach calculates an appropriate run time based on the physical characteristics of your system, such as the quality factor of resonant structures and the duration of sources.

   .. code-block:: python
   
       run_time_spec = td.RunTimeSpec(
           quality_factor=1000,  # Expected quality factor
           source_factor=3       # Buffer factor for source contribution
       )
       
       sim = td.Simulation(
           # ... other parameters ...
           run_time=run_time_spec
       )

The :class:`RunTimeSpec` approach is particularly useful for resonant systems where the appropriate simulation duration may not be immediately obvious.


References
----------

For more details, see:

.. autosummary::
    :toctree: _autosummary/
    :template: module.rst

    tidy3d.Simulation
    tidy3d.RunTimeSpec
