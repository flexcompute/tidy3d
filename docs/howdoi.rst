.. currentmodule:: tidy3d

.. _howdoi:

How do I ... |:eyes:|
========================


Work with the Tidy3d Package
----------------------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - How do I...
     - Solution
   * - See the version of tidy3d I am using?
     - In python, after importing `tidy3d`, run `print(tidy3d.__version__)`. Or, do this through the command line via `python -c "import tidy3d; print(tidy3d.__version__)"`.

Work with Tidy3d Components
---------------------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - How do I...
     - Solution
   * - Get help related to a Tidy3d object (``obj``)?
     - ``obj.help()`` or ``help(Obj)`` if ``Obj`` is the class name (eg. ``help(td.Box)``).
   * - Save and load any Tidy3d object?
     - If the object ``obj`` is an instance of ``ObjClass``, save and load with ``obj.to_file(fname='path/to/file.json')`` and ``obj = ObjClass.from_file(fname='path/to/file.json')``, respectively.
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
     - ``obj.plot(x=0)`` will plot the object on the ``x=0`` plane, but ``y`` and ``z`` are alternatively accepted to specify other planar axes.  If ``ax`` argument will plot to an existing axis, ie. ``obj.plot(y=0, ax=ax)``.
   * - Change the object plotting characteristics (facecolor, edgecolor, etc).
     - Plotting keyword arguments can be supplied to ``plot()``, for example ``obj.plot(x=0, edgecolor='blue', fill=False)``.  These keyword arguments correspond to those fed to `Matplotlib Patches <https://tinyurl.com/2nf5c2fk>`__.
   * - Change the global plot characteristics (title, size, etc).
     - The plotting function return a matplotlib ``Axes``, which can be manipulated, for example ``ax = obj.plot(x=0);  ax.set_title('my_title')``.

Create Geometries
-----------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - How do I...
     - Solution
   * - Load a structure from GDS format?
     - From a cell in the ``gdstk`` package, the ``ps = td.PolySlab.from_gds(gds_cell, ...)`` method will load the geometry into a :class:`PolySlab`.
   * - Create a complex geometry, such as a ring?
     - While many complex geometries can be created by supplying the vertices to :class:`PolySlab`, simple geometries, such as rings, can be constructed by overlaying two structures with :class:`Cylinder` geometry with the inner cylinder defined with medium of air.  Note that structures later in the ``structures`` list will override previous structures, which can be leveraged to make more complex geometries.

Materials
---------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - How do I...
     - Solution
   * - Create a lossy material (with a conductivity)?
     - ``td.Medium(permittivity=2.0, conductivity=1.0)``
   * - Create a material from n, k values at a given frequency?
     - ``td.Medium.from_nk(n=2.0, k=1.0, freq=200e12)``
   * - Create an anisotropic material?
     - ``td.AnisotropicMedium(xx=medium_xx, yy=medium_yy, zz=medium_zz)`` for three ``td.Medium`` objects defining the diagonal elements of the permittivity tensor.
   * - Create a dispersive material from model parameters?
     - Call one of the dispersive models with your parameters, for example ``debye_medium = td.Debye(eps_inf=2.0, coeffs=[(1,2),(3,4)])``.
   * - Create an active material?
     - Tidy3D by default will not allow medium specifications that give rise to active materials to avoid potential divergence. However, you can override this by setting the field ``allow_gain=True`` in any medium, e.g. ``td.Medium(permittivity=2.0, conductivity=-1.0, allow_gain=True)``
   * - Create a spatially varying material?
     - Call one of the "Custom-" materials, e.g. ``CustomMedium`` for a spatially varying non-dispersive medium.
   * - Load a commonly-used dispersive material?
     - Import one of several material models from our ``td.material_library``.

Work with Simulation Data
-------------------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - How do I...
     - Solution
   * - Load the data from a simulation task id `task_id` into the python client?
     - Using the web API ``import tidy3d.web as web``, you can load into a :class:`SimulationData` object through ``sim_data = web.load(task_id, path='path/to/file.hdf5')``.
   * - Access the original :class:`Simulation` that created the data?
     - ``sim_data.simulation`` returns a copy of the original :class:`Simulation`.
   * - Print the log file of the task?
     - ``print(sim_data.log)``.
   * - Save and load the :class:`SimulationData` object?
     - ``sim_data.to_file(fname='path/to/file.hdf5')`` to save and ``sim_data = SimulationData.from_file(fname='path/to/file.hdf5')`` to load.
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
     - ``mon_data = sim_data[monitor_name]`` where ``monitor_name`` was the ``.name`` of the original monitor.
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
   * - Get a specific field component (eg. ``'Hy'``) for a :class:`FieldMonitor` or :class:`FieldTimeMonitor`?
     - ``component_data = sim_data[monitor_name].Hy`` or, to access by string: ``component_data = sim_data[monitor_name].data_dict['Hy']``.

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
     - Once you've created a :class:`tidy3d.web.container.Job`, you can upload it to our servers with ``job.upload()`` and it will not run until you explicitly tell it to with ``job.start()``.
   * - Monitor the progress of a simulation?
     - ``web.monitor(task_id)``, ``job.monitor()``, or ``batch.monitor()`` will display the progress of your simulation(s).
   * - Load the results of a simulation?
     - ``sim_data = job.load(path)`` will download the results to ``path`` and load them as :class:`SimulationData` object.
   * - See information about my :class:`tidy3d.web.container.Job`, such as how many credits it will take?
     - After uploading your job with ``job.upload()`` you can get a host of information about it through ``task_info = job.get_info()``.
   * - Submit multiple simulations?
     - The :class:`tidy3d.web.container.Batch` interface was created to manage multiple :class:`tidy3d.web.container.Job` instances and gives a similar interface with large number of jobs in mind.
   * - Loop through :class:`tidy3d.web.container.BatchData` without loading all of the data into memory?
     - ``for task_name, sim_data in batch_data.items():`` will give access to a :class:`SimulationData` instance for each :class:`tidy3d.web.container.Job` in the batch one by one, so you can perform your postprocessing in the loop body without loading each of the simulations' data into memory at once.
   * - Save or load a :class:`tidy3d.web.container.Job` or :class:`tidy3d.web.container.Batch` so I can work with it later?
     - Like most other tidy3d objects, :class:`tidy3d.web.container.Job` and :class:`tidy3d.web.container.Batch` instances have ``.to_file(path)`` and ``.from_file(path)`` methods that will export and load their metadata as .json files.  This is especially useful for loading batches for analysis long after they have run.

Extensions
----------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - How do I...
     - Solution
   * - Create a material from optical n,k data?
     - Refer to the `Dispersion tutorial <notebooks/Fitting.html>`_ on the :class:`tidy3d.plugins.DispersionFitter` plugin.
   * - Specify the modes for a :class:`ModeMonitor` or :class:`ModeSource`?
     - Refer to the `Mode Solver tutorial <notebooks/ModeSolver.html>`_ on the :class:`tidy3d.plugins.ModeSolver` plugin.
   * - Project electromagnetic near field data to the far field?
     - Refer to the `FieldProjections tutorial <notebooks/FieldProjections.html>`_.
   * - Compute scattering matrix parameters for modeling my device?
     - Refer to the `S Matrix tutorial <notebooks/SMatrix.html>`_ on the :class:`tidy3d.plugins.ComponentModeler` plugin.
