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
   * - Get help related to any Tidy3d component?
     - ``obj.help()`` will print a useful message.  ``obj.help(methods=True)`` will list out more information about the object's methods.
   * - Save and load any Tidy3d object?
     - If the object ``obj`` is an instance of ``ObjClass``, you can export and load to & from json through ``obj.export(path='path/to/file.json')`` and ``obj = ObjClass.load(path='path/to/file.hdf5')``.


Work with Simulation Data
-------------------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - How do I...
     - Solution
   * - Get the data for a specific :class:`Monitor`?
     - ``sim_data[monitor_name]`` or ``sim_data.monitor_data[monitor_name].data`` returns the xarray data object and ``sim_data.monitor_data[monitor_name]`` returns the :class:`MonitorData` container.
   * - Save and load the :class:`SimulationData` object?
     - ``sim_data.export(path='path/to/file.hdf5')`` to save and ``sim_data = SimulationData.load(path='path/to/file.hdf5')`` to load.
   * - Access the original :class:`Simulation` that created the data?
     - ``sim_data.simulation`` returns a copy of the original :class:`Simulation`.

Work with Monitor Data
----------------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - How do I...
     - Solution
   * - Get my monitor's data from a :class:`SimulationData` object?
     - ``mon_data = sim_data[monitor_name]`` where ``monitor_name`` was the key of your :class:`Monitor` in the original :attr:`Simulation.monitors` dictionary.
   * - Get the data at a certain coordinate value (ie. ``x=0.0``, ``f=200e12``?
     - For data with coordinate ``name`` equal to ``value``, use ``mon_data.sel(name: value)``, eg. ``mon_data.sel(z=0, f=200e12)``.  If the data is not stored at that exact value, an error will be raised, use ``mon_data.interp`` instead.
   * - Get the data at certin index into a coordinate (ie ``t=0`` for "first time step".)
     - For data at index ``index`` into coordinate ``name``, use ``mon_data.isel(name: index)``, eg. ``mon_data.isel(t=0)``.  Note, like ``.sel`` you can pass multiple ``name: index`` pairs.
   * - Interpolate the data at various coordinate values, eg. field data at ``x=y=z=0``?
     - ``mon_data.interp(x=0, y=0, z=0)`` performs linear interpolation of the data, where the values must be within the range recorded by the original monitor specification.
   * - Access a specific ``field`` component from a :class:`FieldMonitor` or :class:`FieldTimeMonitor`?
     - If eg. ``'Ey'`` is in the :attr:`Monitor.fields` list, you can grab it through ``field_mon.Ey`` or ``field_mon['Ey']``.
   * - Get the real part, imagninary part, or absolute value of complex-valued data.
     - ``mon_data.real``, ``mon_data.imag``, ``np.abs(mon_data)``, respectively.
   * - Get the original :class:`Monitor` that created this :class:`MonitorData`?
     - ``mon_data.monitor`` returns a copy of the :class:`Monitor` and ``mon_data.monitor_name`` gives its name in the :attr:`Simulation.monitors` dictionary.
   * - Access the xarray representation of the data from a raw :class:`MonitorData` instance?
     - if ``isinstance(mon_data, MonitorData)``, then ``mon_data.data`` provides the interface for all of the functionality described above.  (Eg. ``mon_data.data.isel(x=0)``).
   * Get the raw data values as a ``numpy`` array?
     - ``mon_data.values``


Plot Tidy3D Components
----------------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - How do I...
     - Solution
   * - Plot an object with some spatial definition?  (:class:`Simulation`, :class:`Structure`, etc.)
     - ``obj.plot(x=0)`` will plot the object on the ``x=0`` plane with ``y`` and ``z`` also accepted to specify other planar axes.  If ``ax`` argument is supplied, will plot to an existing axis, ie. ``obj.plot(y=0, ax=ax)``.
   * - Change the plot characteristics (facecolor, edgecolor, etc).
     - Plotting keyword arguments can be supplied to ``plot()``, for example ``obj.plot(x=0, edgecolor='blue', fill=False)``.  These keyword arguments correspond to those fed to `Matplotlib Patches <https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.Patch.html#matplotlib.patches.Patch>`__.


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
     - ``sim_data.simulation.plot_structures_eps(z=0.0, ax=ax, alpha=.5, cbar=False, lw=0)`` will plot the ``z=0`` cross-section, where ``ax`` is the axis where your fields are plotted on.
