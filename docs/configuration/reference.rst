Configuration Reference
=======================

All configuration for ``tidy3d`` lives under a single object:

.. code-block:: python

   from tidy3d import config

The tables below list the built-in sections and every option they expose.

How to read this page
---------------------

- Environment variable overrides always take precedence. Follow the pattern
  ``TIDY3D_<SECTION>__<FIELD>`` (nesting continues with additional ``__``
  segments such as ``TIDY3D_PLUGINS__SAMPLE__ENABLED``).
- The *Persisted* column marks fields written to disk when you call
  ``config.save()``. Unmarked fields remain in-memory unless you pass
  ``include_defaults=True`` or store them in profiles or environment variables.
- Descriptions call out notable allowed values, defaults, and persistence
  behavior. Literal values appear in quotes and tuples use the ``(x, y, z)``
  notation seen throughout the API.


Logging (``config.logging``)
----------------------------

Controls the verbosity and suppression behavior of the global logger.

.. list-table::
   :header-rows: 1
   :widths: 22 18 10 50

   * - Option
     - Default
     - Persisted
     - Description
   * - ``level``
     - ``"WARNING"``
     - Yes
     - Lowest logging level that will be emitted. Accepts ``"DEBUG"``, ``"SUPPORT"``, ``"USER"``,
       ``"INFO"``, ``"WARNING"``, ``"ERROR"``, and ``"CRITICAL"``.
   * - ``suppression``
     - ``True``
     - No
     - Suppress repeated log messages when ``True`` so only the first occurrence of identical
       messages is shown.


Simulation (``config.simulation``)
----------------------------------

Optional overrides that tweak solver behavior at runtime.

.. list-table::
   :header-rows: 1
   :widths: 22 18 10 50

   * - Option
     - Default
     - Persisted
     - Description
   * - ``use_local_subpixel``
     - ``None``
     - No
     - Set to ``True`` to force local subpixel averaging, ``False`` to disable it, or leave ``None`` to keep the solver default.


Microwave (``config.microwave``)
--------------------------------

Options that apply to the microwave solver add-on.

.. list-table::
   :header-rows: 1
   :widths: 22 18 10 50

   * - Option
     - Default
     - Persisted
     - Description
   * - ``suppress_rf_license_warning``
     - ``False``
     - No
     - Skip the warning about RF license availability when set to ``True``.


Adjoint (``config.adjoint``)
----------------------------

Parameters for adjoint behavior, including local execution settings and numerical tolerances.
These overrides apply only when ``local_gradient`` is ``True``; otherwise the service uses its
remote defaults and emits a warning reminding you to enable local gradients.

.. list-table::
   :header-rows: 1
   :widths: 24 18 10 48

   * - Option
     - Default
     - Persisted
     - Description
   * - ``min_wvl_fraction``
     - ``0.05``
     - No
     - Minimum fraction of the smallest wavelength used when discretizing cylinders for autograd derivatives.
   * - ``points_per_wavelength``
     - ``10``
     - No
     - Number of material sample points per wavelength for cylinder discretization (must be positive).
   * - ``default_wavelength_fraction``
     - ``0.1``
     - No
     - Fallback fraction of the minimum wavelength when adaptive spacing is needed (must be ``>= 0``).
   * - ``minimum_spacing_fraction``
     - ``0.01``
     - No
     - Smallest normalized spacing allowed when constructing adaptive finite-difference stencils (must be ``>= 0``).
   * - ``local_gradient``
     - ``False``
     - Yes
     - Enable local gradient evaluation. Remote gradients ignore other adjoint overrides unless this is ``True``.
   * - ``local_adjoint_dir``
     - ``"adjoint_data"``
     - Yes
     - Directory (relative to the working directory) where intermediate gradient artifacts are stored when ``local_gradient`` is enabled.
   * - ``gradient_precision``
     - ``"single"``
     - No
     - Floating-point precision used for gradient calculations. Accepts ``"single"`` or ``"double"``.
   * - ``monitor_interval_poly``
     - ``(1, 1, 1)``
     - No
     - Cell spacing between samples for polynomial autograd monitors.
   * - ``monitor_interval_custom``
     - ``(1, 1, 1)``
     - No
     - Cell spacing between samples for custom autograd monitors.
   * - ``quadrature_sample_fraction``
     - ``0.4``
     - No
     - Fraction of uniform samples reused when building Gauss quadrature nodes (between ``0`` and ``1``).
   * - ``gauss_quadrature_order``
     - ``7``
     - No
     - Maximum Gauss–Legendre order used in composite quadrature rules (must be positive).
   * - ``edge_clip_tolerance``
     - ``1e-9``
     - No
     - Padding tolerance used when clipping polygon edges during surface integrations (must be ``>= 0``).
   * - ``solver_freq_chunk_size``
     - ``None``
     - No
     - Upper bound on the number of frequencies processed per chunk during gradient evaluation. Set to a positive integer to enable chunking or leave ``None`` to disable it.
   * - ``max_traced_structures``
     - ``500``
     - No
     - Maximum number of structures whose fields may be traced in an adjoint run.
   * - ``max_adjoint_per_fwd``
     - ``10``
     - No
     - Maximum number of adjoint simulations dispatched per forward solve.


Web (``config.web``)
--------------------

Settings for the cloud API client and related environment overrides.

.. list-table::
   :header-rows: 1
   :widths: 24 18 10 48

   * - Option
     - Default
     - Persisted
     - Description
   * - ``apikey``
     - ``None``
     - Yes
     - API key used for authentication. The value is masked when serialized. Also accepts ``SIMCLOUD_APIKEY`` as a shortcut environment variable.
   * - ``ssl_verify``
     - ``True``
     - No
     - Verify SSL certificates for API requests.
   * - ``enable_caching``
     - ``True``
     - Yes
     - Allow the web service to return cached simulation results when available.
   * - ``api_endpoint``
     - ``"https://tidy3d-api.simulation.cloud"``
     - No
     - Base URL for API calls. Must be an HTTP or HTTPS URL.
   * - ``website_endpoint``
     - ``"https://tidy3d.simulation.cloud"``
     - No
     - Base URL for the Tidy3D website. Must be an HTTP or HTTPS URL.
   * - ``s3_region``
     - ``"us-gov-west-1"``
     - No
     - AWS region used by the platform’s S3 storage.
   * - ``timeout``
     - ``120``
     - No
     - HTTP request timeout in seconds (between ``0`` and ``300``).
   * - ``ssl_version``
     - ``None``
     - No
     - Explicit TLS version to enforce (for example ``ssl.TLSVersion.TLSv1_2``). Leave ``None`` to let ``requests`` negotiate the version.
   * - ``env_vars``
     - ``{}``
     - No
     - Additional environment variables exported before API calls. Useful for proxy or credential helpers.


Local Cache (``config.local_cache``)
------------------------------------

Controls the optional on-disk cache for simulation artifacts.

.. list-table::
   :header-rows: 1
   :widths: 24 18 10 48

   * - Option
     - Default
     - Persisted
     - Description
   * - ``enabled``
     - ``False``
     - Yes
     - Turn the local cache on or off. When enabled, results are reused if the inputs match.
   * - ``directory``
     - ``<base>/cache/simulations``
     - Yes
     - Directory where cached artifacts are stored. The path is expanded, resolved, and created if missing. ``<base>`` comes from ``TIDY3D_BASE_DIR`` when set, otherwise ``XDG_CACHE_HOME`` on Unix or ``~/.cache`` elsewhere (resolving to ``~/.cache/tidy3d/simulations`` by default).
   * - ``max_size_gb``
     - ``10.0``
     - Yes
     - Maximum cache size in gigabytes. ``0`` disables the size limit.
   * - ``max_entries``
     - ``0``
     - Yes
     - Maximum number of cached simulations retained. ``0`` means no limit and eviction falls back to size constraints.


Plugins (``config.plugins``)
----------------------------

Container that holds plugin-defined sections. After a plugin calls
``@register_plugin("name")``, its configuration becomes available under
``config.plugins.<name>`` and follows the same persistence and environment variable rules described above (for example ``TIDY3D_PLUGINS__NAME__FIELD``).
