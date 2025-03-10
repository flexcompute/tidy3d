***************
API |:computer:|
***************

This guide will help you understand how to create, configure, 
and work with simulations in Tidy3D **using the Python API** rather than the web interface.
This is the recommended way to work with Tidy3D as it gives you finer control
and allows you to fully automate your simulations and analysis of the results.

Please refer to the :doc:`/install` page for detailed instructions. Make sure you have:

1. **Installed Tidy3D** on your system.
2. **Configured your API key** to authenticate with the Tidy3D servers.

That done, you can start using Tidy3D. Here is a simple example of a dipole simulation:

.. code-block:: python

    import matplotlib.pylab as plt
    import tidy3d as td
    import tidy3d.web as web

    # simulation parameters
    side_length = 9.0           # side length of the simulation box (µm)
    grid_size = 50e-3           # grid size (µm)
    lambda0 = 1.0               # wavelength (µm)
    freq0 = td.C_0 / lambda0    # central frequency (Hz)

    # apply a PML in all directions
    boundary_spec = td.BoundarySpec.all_sides(boundary=td.PML())

    # define dipole source at origin pointing in z
    dipole_source = td.PointDipole(
        center=(0, 0, 0), 
        polarization='Ez', 
        source_time=td.GaussianPulse(freq0=freq0, fwidth=freq0/50)
    )

    # define monitor to measure fields in xz plane at central frequency
    monitor = td.FieldMonitor(
        center=(0, 0, 0), 
        size=(td.inf, 0, td.inf),
        freqs=[freq0], 
        name='freq_domain'
    )

    # define simulation
    simulation = td.Simulation(
        size=(side_length, side_length, side_length),
        grid_spec=td.GridSpec.uniform(dl=grid_size),
        structures=[],
        sources=[dipole_source],
        monitors=[monitor],
        run_time=200 / freq0,
        boundary_spec=boundary_spec
    )

    # run simulation
    sim_data = web.run(simulation, task_name='intro_dipole', path='data/data_dipole.hdf5')

    # check that frequency domain fields look good
    ax = sim_data.plot_field('freq_domain', 'Ez', y=0, f=freq0)
    plt.show()

The resulting figure of the :math:`E_z` field component in the :math:`x-z`-plane is shown here:

.. image:: /_static/img/intro_dipole_ez.png
   :align: right
   :width: 400px
   :alt: Electric field pattern from a dipole source simulated with Tidy3D.

We import libraries (``tidy3d`` as ``td`` for simulation, ``tidy3d.web`` for API access), 
define simulation parameters (domain size, grid resolution, wavelength/frequency), 
apply boundary conditions (PML), add a dipole source, configure a field monitor to record fields, 
define the complete simulation, run it in the cloud, and then view the results.

You can find many :doc:`/lectures/index` with examples of how to use the Python API.
**But reading the following sections will help you understand the details of how to define
and refine each of these elements to create any Tidy3D simulation you need.**

.. toctree::
    :hidden:

    simulation
    boundary_conditions
    geometry
    mediums
    material_library
    rf_material_library
    structures
    sources
    analytic_beams
    monitors
    mode
    field_projector
    lumped_elements
    discretization
    subpixel_averaging
    output_data
    scene
    logging
    submit_simulations
    heat/index
    charge/index
    eme/index
    microwave/index
    plugins/index
    spice
    constants
    abstract_base
    abstract_models

.. include:: /api/simulation.rst
.. include:: /api/boundary_conditions.rst
.. include:: /api/geometry.rst
.. include:: /api/mediums.rst
.. include:: /api/structures.rst
.. include:: /api/sources.rst
.. include:: /api/analytic_beams.rst
.. include:: /api/monitors.rst
.. include:: /api/mode.rst
.. include:: /api/field_projector.rst
.. include:: /api/lumped_elements.rst
.. include:: /api/discretization.rst
.. include:: /api/subpixel_averaging.rst
.. include:: /api/output_data.rst
.. include:: /api/scene.rst
.. include:: /api/logging.rst
.. include:: /api/submit_simulations.rst
.. include:: /api/heat/index.rst
.. include:: /api/charge/index.rst
.. include:: /api/eme/index.rst
.. include:: /api/microwave/index.rst
.. include:: /api/plugins/index.rst
.. include:: /api/constants.rst
.. include:: /api/abstract_base.rst
.. include:: /api/abstract_models.rst