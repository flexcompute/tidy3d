.. currentmodule:: tidy3d

.. _howdoi:

How do I ...
============


Work with Tidy3d Components
---------------------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - How do I...
     - Solution
   * - Get help related to any Tidy3d object?
     - ``obj.help()``.
   * - Save and load any Tidy3d object?
     - If the object ``obj`` is an instance of ``ObjClass``, save and load with ``obj.to_file(path='path/to/file.json')`` and ``obj = ObjClass.from_file(path='path/to/file.hdf5')``, respectively.
   * - Get all data in a Tidy3d object as a dictionary.
     - ``obj.dict()``.

Plot Tidy3D Components
----------------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - How do I...
     - Solution
   * - Plot an object with a spatial definition?  (:class:`Simulation`, :class:`Structure`, etc.)
     - ``obj.plot(x=0)`` will plot the object on the ``x=0`` plane with ``y`` and ``z`` alternatively accepted to specify other planar axes.  If ``ax`` argument is supplied, will plot to an existing axis, ie. ``obj.plot(y=0, ax=ax)``.
   * - Change the plot characteristics (facecolor, edgecolor, etc).
     - Plotting keyword arguments can be supplied to ``plot()``, for example ``obj.plot(x=0, edgecolor='blue', fill=False)``.  These keyword arguments correspond to those fed to `Matplotlib Patches <https://tinyurl.com/2nf5c2fk>`__.

Create Geometries
-----------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - How do I...
     - Solution
   * - Load a structure from GDS format?
     - From a cell in the  ``gdspy`` package, the ``td.PolySlab.from_gds(gds_cell, ...)`` method will load the geometry into a ``td.PolySlab``.
   * - Create a complex geometry, such as a ring?
     - While many complex geometries can be created using ``td.PolySlab``, simple geometries, such as rings, can be constructed by overlaying two structures with ``td.Cylinder()`` geomety with the inner cylinder defined with medium of air.  Note that structures later in the ``structures`` list will override previous structures.
       
Materials
---------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - How do I...
     - Solution
   * - Create a lossy material?
     - ``td.Medium(permittivity=2.0, conductivity=1.0)``
   * - Create a material from n, k values at a given frequency?
     - ``td.Medium.from_nk(n=2.0, k=1.0, freq=200e12)``
   * - Create an anisotropic material?
     - ``td.AnisotropicMedium(xx=medium_xx, yy=medium_yy, zz=medium_zz)`` for three ``td.Medium`` objects defining the diagonal elements of the permittivity tensor.
   * - Create a dispersive material from model parameters?
     - Call one of the dispersive models with your parameters, for example ``debye_medium = Debye(eps_inf=2.0, coeffs=[(1,2),(3,4)])``.
   * - Load a commonly-used dispersive material?
     - Import one of several material models from our ``td.material_library``.

Work with Simulation Data
-------------------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - How do I...
     - Solution
   * - Access the original :class:`Simulation` that created the data?
     - ``sim_data.simulation`` returns a copy of the original :class:`Simulation`.
   * - Print the log file of the task?
     - ``print(sim_data.log)``.
   * - Save and load the :class:`SimulationData` object?
     - ``sim_data.to_file(path='path/to/file.hdf5')`` to save and ``sim_data = SimulationData.from_file(path='path/to/file.hdf5')`` to load.
   * - Access the data for a specific :class:`Monitor`?
     - ``sim_data[monitor_name]``.
   * - Interpolate the electromagnetic field data at the Yee cell centers?
     - ``sim_data.at_centers(monitor_name)`` if ``monitor_name`` corresponds to a :class:`FieldMonitor` or :class:`FieldTimeMonitor`.

Work with Monitor Data
----------------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - How do I...
     - Solution
   * - Get my monitor's data from a :class:`SimulationData` object?
     - ``mon_data = sim_data[monitor_name]`` where ``monitor_name`` was the ``.name`` of the original :class:`Monitor`.
   * - Select the data at a certain coordinate value (ie. ``x=0.0``, ``f=200e12``?
     - ``mon_data.sel(z=0, f=200e12)``.
   * - Get the data at certin index into a coordinate (ie ``t=3`` for the 4th measured time step)?
     - ``mon_data.isel(t=3)``.
   * - Interpolate the data at various coordinate values, eg. field data at the origin (``x=y=z=0``)?
     - ``mon_data.interp(x=0, y=0, z=0)`` performs linear interpolation of the data, where the values must be within the range recorded by the original monitor specification.
   * - Get the real part, imagninary part, or absolute value of complex-valued data.
     - ``mon_data.real``, ``mon_data.imag``, ``abs(mon_data)``, respectively.
   * - Get the raw data values as a ``numpy`` array?
     - ``mon_data.values``.
   * - Get a specific field component (eg. ``Hy``) for a :class:`FieldMonitor` or :class:`FieldTimeMonitor`?
     - ``component_data = sim_data[monitor_name].Hy``. 

Plot Data
---------

.. list-table::
   :header-rows: 1
   :widths: 40 60


   * - How do I...
     - Solution
   * - Plot the :class:`MonitorData` as a function of one of its coordinates.
     - ``mon_data.plot()`` if the data is already 1D. To select x axis explicitly or plot all the data on same plot, ``mon_data.plot.line(x='f', ax=ax)``.  Note, for all plotting, if ``ax`` not supplied, will be created.
   * - Plot the :class:`MonitorData` as a function of two of its coordinates?
     - ``mon_data.plot()`` if data is 2D, otherwise one can use ``mon_data.plot.pcolormesh(x='x', y='y')`` to specify the ``x`` and ``y`` coordinates explicitly.
   * - Plot the simulation structure on top of my field plot?
     - ``sim_data.plot_field(monitor_name, component, z=3, val='real', **kwargs)`` to plot the real part of the fields on the ``z=3`` plane.

Submit Jobs to Server
---------------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - How do I...
     - Solution
   * - Submit my simulation to run on Flexcompute's servers?
     - ``sim_data = web.run(simulation, task_name='my_task', path='out/data.hdf5')`` or ``job = web.Job(simulation, task_name='my_task'); job.run(path='out/data.hdf5')``.
   * - Upload a job to the web without running it so I can inspect it first?
     - Once you've created a :class:`Job`, you can upload it to our servers with ``job.upload()`` and it will not run until you excplicitly tell it to with ``job.start()``.
   * - Monitor the progress of a simulation?
     - ``web.monitor(task_id)``, ``job.monitor()``, or ``batch.monitor()`` will display the progress of your simulation(s).
   * - Load the results of a simulation?
     - ``sim_data = job.load(path)`` will download the results to ``path`` and load them as :class:`SimulationData` object.
   * - See information about my :class:`Job`, such as how many credits it will take?
     - After uploading your job with ``job.upload()`` you can get a host of information about it through ``task_info = job.get_info()``.
   * - Submit multiple simulations?
     - The :class:`Batch` interface was created to manage multiple :class:`Job` instances and gives a similar interface with large number of jobs in mind.
   * - Loop through :class:`Batch` data without loading all of the data into memory?
     - ``for task_name, sim_data in batch.items():`` will ``yield`` the :class:`SimulationData` for each :class:`Job` in the batch one by one, so you can perform your postprocessing in the loop body without loading each of the simulations' data into memory at once.
   * - Save or load a :class:`Job` or :class:`Batch` so I can work with it later?
     - Like most other tidy3d objects, :class:`Job` and :class:`Batch` instances have ``.to_file(path)`` and ``.from_file()`` methods that will export and load thier metadata as .json files.  This is especially useful for loading batches for analysis long after they have run.

Extensions
----------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - How do I...
     - Solution
   * - Create a material from optical n,k data?
     - Refer to the tutorial on the ``DispersionFitter`` plugin.
   * - Specify the modes for a :class:`ModeMonitor` or :class:`SourceMonitor`?
     - Refer to the tutorial on the ``ModeSolver`` plugin.
   * - Projet electromagnetic near field data to the far field?
     - Refer to the tutorial on the ``Near2Far`` plugin.

