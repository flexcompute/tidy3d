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
   * - ``warn_once``
     - ``False``
     - No
     - Show each unique warning message at most once per process when ``True``.


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
     - Controls whether local subpixel averaging is used in ``Simulation.epsilon``,
       ``ModeSimulation.run_local``, and ``ModeSolver.solve``. Requires the ``tidy3d-extras``
       package (``pip install "tidy3d[extras]"``). Set to ``True`` to force local subpixel
       averaging (raises an error if ``tidy3d-extras`` is not installed), ``False`` to disable
       it and use permittivity staircasing, or leave ``None`` to use subpixel averaging when
       available and silently fall back to staircasing otherwise.
       See :doc:`../extras/index` for more details.


Validation (``config.validation``)
----------------------------------

Controls how much validation runs when models are built or loaded.

.. list-table::
   :header-rows: 1
   :widths: 22 18 10 50

   * - Option
     - Default
     - Persisted
     - Description
   * - ``mode``
     - ``"full"``
     - No
     - ``"full"`` runs tidy3d's validators on construction and parsing. ``"fast"`` skips the
       check validators and keeps pydantic's type checking, field constraints, coercion, every
       validator that normalises or derives a value, and serialization, so already validated
       data loads several times faster and dumps identically. Use it only for data that was
       already validated, such as files accepted by the server; ``validate_pre_upload`` is
       unaffected. ``from_file(..., validate=...)`` overrides it for one call, context-locally.
       Never written to disk, not even with a named profile: it applies to the current process
       only.

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

Parameters for adjoint behavior, including local execution settings, source planning, and
numerical tolerances. Local-gradient overrides apply only when ``local_gradient`` is ``True``;
field-source PCA options instead apply during client-side adjoint source planning.

Field-source compression is experimental and disabled by default (the mode is ``None``). Set
``config.adjoint.field_source_reduction_mode`` to ``"pca"``, ``"temporal"``, or ``"auto"`` to opt
in. All three share the same spatial decomposition and differ only in how the retained modes are
injected: ``"pca"`` runs one adjoint simulation per mode, ``"temporal"`` synthesizes one multi-tone
waveform per mode so every batched mode is injected in one adjoint simulation, and ``"auto"`` takes the
PCA path while its plan needs at most
``config.adjoint.field_source_reduction_max_separate_sims`` simulations — short independent runs
parallelize well and skip synthesis risk — and otherwise takes the temporal path, reverting to the
PCA plan if the synthesis is not feasible.

Every mode is best-effort. Whenever a reduction does not apply, misses its accuracy gates, or would
not actually use fewer simulations than standard port-versus-frequency grouping, the sources revert
to that standard grouping, so enabling a mode never costs more adjoint simulations than leaving it
off. Only the combined temporal simulation is given an explicit run time, since its synthesized
pulse can outlast the forward run; every separate simulation inherits the forward ``run_time`` and is
ended by the solver's own shutoff once its fields decay. Temporal synthesis additionally requires a readable
field-decay record showing the forward simulation decayed below a positive shutoff threshold, since it
sizes the synthesized pulses from the duration the forward run actually took; without that it is
skipped with a warning. The default coverage of ``0.999``
allows an approximate source reconstruction; set it to ``1.0`` to retain the full numerical rank
of each compatible source-profile matrix. Coverage applies per current type: electric and magnetic
current blocks are truncated independently, each in its own physical units, so no material
sampling is involved and dispersive or lossy backgrounds are supported. The retained energy is
measured over all sources decomposed together rather than per monitor or per frequency, so a
monitor carrying only a small share of that energy may be represented coarsely, or dropped, even
though the requested coverage is met. The retained fraction describes the injected source profiles
rather than the gradient: the discarded component still contributes through the simulated field
response, which can amplify it, so coverage constrains gradient error indirectly rather than
bounding it, and coverage is an energy measure, so ``0.999`` can discard up to roughly 3% of the
source amplitude. If a weak monitor's own gradient matters — suppressing leakage into a dim
port, for instance — use ``1.0`` so that every source and frequency is represented exactly. Compression is attempted
only for point, line, and planar ``FieldData`` current-source batches whose support dimension,
frequency tuple, and field-component tuple all match exactly; sources that split components across
monitors on one support are merged beforehand. Simulations with symmetry use the standard adjoint
grouping path. Oversized or non-reducing blocks also fall back to standard port-versus-frequency
grouping.

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
     - ``0.001``
     - No
     - Fraction of the shortest free-space adjoint wavelength used as a lower bound for adaptive shape-gradient surface-sampling spacing (must be ``>= 0``). The spacing is also never finer than the smallest grid step near the traced structure, so the larger of the two bounds applies.
   * - ``boundary_snapping_fraction``
     - ``1.0``
     - No
     - Fraction of minimum local grid size used to snap coordinates outside of boundaries for shape gradients (must be ``>= 0.5``).
   * - ``pec_detection_threshold``
     - ``-100.0``
     - No
     - Real permittivity threshold below which a material is treated as PEC in shape-gradient boundary integration (must be ``<= 0``).
   * - ``local_gradient``
     - ``False``
     - Yes
     - Enable local gradient evaluation. Remote gradients ignore local-only execution and numerical overrides, while client-side source-planning settings such as field-source PCA still apply.
   * - ``local_adjoint_dir``
     - ``"adjoint_data"``
     - Yes
     - Directory (relative to the working directory) where intermediate gradient artifacts are stored when ``local_gradient`` is enabled.
   * - ``parallel_run``
     - ``False``
     - Yes
     - Launch canonical adjoint simulations for supported monitors in parallel with the forward solve when local gradients are enabled. If any unsupported monitors are present, parallel adjoint is disabled and the sequential adjoint pipeline is used.
   * - ``parallel_adjoint_mode_direction_policy``
     - ``"assume_outgoing"``
     - Yes
     - Policy for selecting mode directions when launching parallel adjoint simulations. Accepts ``"assume_outgoing"`` or ``"run_both_directions"``.
   * - ``field_source_reduction_mode``
     - ``None``
     - No
     - Strategy for compressing compatible ``FieldData``-derived adjoint current sources before adjoint simulations are launched. ``None`` applies no reduction and uses standard adjoint source grouping; ``"pca"`` compresses sources into principal components, running one adjoint simulation per component; ``"temporal"`` combines those components into a single adjoint simulation with synthesized multi-tone source waveforms; ``"auto"`` takes the PCA path while its plan needs at most ``field_source_reduction_max_separate_sims`` simulations and otherwise takes the temporal path. Every mode reverts to standard grouping when its reduction does not apply, misses its accuracy gates, or would not use fewer simulations. The temporal path additionally requires a forward simulation that kept a positive ``shutoff`` and decayed below it, since it sizes its waveforms from the duration that run actually took.
   * - ``field_source_reduction_max_separate_sims``
     - ``2``
     - No
     - In ``"auto"`` mode, the largest number of planned adjoint simulations for which separate PCA component simulations are preferred over the single combined temporal simulation. ``0`` always attempts the combined simulation first.
   * - ``field_source_pca_min_energy_coverage``
     - ``0.999``
     - No
     - Fraction of weighted source-profile energy retained per current type by the field-source PCA decomposition, shared by the ``"pca"``, ``"temporal"``, and ``"auto"`` modes. Must be between ``0`` and ``1``; ``1.0`` retains the full numerical rank.
   * - ``field_source_pca_max_matrix_entries``
     - ``20_000_000``
     - Yes
     - Maximum positive number of dense complex entries in one PCA profile matrix. Oversized blocks use standard port-versus-frequency grouping. This bounds stored entries only: peak memory during the decomposition is several times larger, and decomposition time also grows with the frequency count, so equally sized blocks can take very different times.
   * - ``field_source_temporal_pulse_scale``
     - ``1.0``
     - No
     - Positive scale factor setting where the synthesized pulse duration search starts, as a multiple of the duration the forward simulation actually ran. The synthesis escalates only when the spectral targets cannot be met accurately at that length, and never past the bound set by ``field_source_temporal_max_run_time_ratio``.
   * - ``field_source_temporal_max_run_time_ratio``
     - ``2.0``
     - No
     - Upper bound on the combined temporal adjoint simulation's run time, as a multiple of the duration the forward simulation actually ran. The adjoint decays in the same structure, so it receives that measured duration as a ring-down margin after its pulse ends, and the pulse itself is capped at this ratio minus one times the measured duration. Must exceed ``1.0``, since the margin alone accounts for one whole multiple. Plans that cannot synthesize a valid pulse within the budget fall back rather than exceed it.
   * - ``field_source_temporal_max_sources``
     - ``64``
     - No
     - Maximum positive number of source objects allowed in the combined temporal adjoint simulation, counting one per retained spatial mode per support. This bounds what the reduction will plan, so that it does not collapse into one large, poorly conditioned simulation; it is not a constraint on what a simulation may contain. Larger plans fall back to the other reduction paths with a warning naming this setting.
   * - ``field_source_temporal_max_waveform_entries``
     - ``20_000_000``
     - Yes
     - Maximum positive number of dense complex values a temporal plan may allocate and retain while it is built. The count grows with the number of time samples times the number of adjoint frequencies, with the square of the frequency count, and with the number of synthesized waveforms. Each value is a complex128, so the default is roughly 320 MB. This estimates peak usage rather than bounding it exactly. Time samples are already bounded by ``field_source_temporal_max_run_time_ratio``, so this guards the remaining axis: an objective spanning very many adjoint frequencies. Larger plans fall back before anything is allocated, with a warning naming this setting. Note that it counts values, not work: the cost of planning grows with the cube of the frequency count, faster than the budget itself.
   * - ``field_source_temporal_spectrum_rtol``
     - ``0.001``
     - No
     - Maximum relative error allowed between each synthesized waveform's spectrum and its complex adjoint targets, greater than ``0`` and at most ``0.1``. Like the coverage setting, this constrains the injected sources rather than the gradient: the spectral mismatch still reaches the gradient through the simulated field response, which can amplify it.
   * - ``gradient_precision``
     - ``"single"``
     - No
     - Floating-point precision used for gradient calculations. Accepts ``"single"`` or ``"double"``.
   * - ``monitor_interval_poly``
     - ``(1, 1, 1)``
     - No
     - Cell spacing between samples for the volumetric adjoint monitors of geometry paths handled volumetrically (numerical structures and custom-vjp-owned paths). Standard shape gradients record surface point clouds whose density is set by ``default_wavelength_fraction`` and ignore this setting.
   * - ``monitor_interval_custom``
     - ``(1, 1, 1)``
     - No
     - Cell spacing between samples for the volumetric adjoint monitors of structures with traced medium paths. Shape-gradient surface point clouds are unaffected.
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
   * - ``memory_allotment_fraction``
     - ``0.75``
     - Yes
     - Fraction of reported available RAM reserved for adjoint postprocessing when auto-selecting frequency chunk sizes and TriangleMesh surface-gradient batch sizes (between ``0`` and ``1``).
   * - ``max_traced_structures``
     - ``500``
     - No
     - Maximum number of structures whose fields may be traced in an adjoint run (must be positive).
   * - ``max_adjoint_per_fwd``
     - ``10``
     - No
     - Maximum number of adjoint simulations dispatched per forward solve (must be positive).


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
   * - ``default_num_workers``
     - ``10``
     - No
     - Default worker count for configurable ``Batch`` thread pools when ``num_workers`` is not provided (must be positive). Upload/start uses a fixed concurrency of ``64`` workers.
   * - ``ssl_version``
     - ``None``
     - No
     - Explicit TLS version to enforce. Accepts ``"TLSv1"``, ``"TLSv1_1"``, ``"TLSv1_2"``, or ``"TLSv1_3"``. Leave ``None`` to let ``requests`` negotiate the version.
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
     - ``True``
     - Yes
     - Turn the local cache on or off. When enabled, results are reused if the inputs match.
   * - ``directory``
     - Platform-dependent
     - Yes
     - Directory where cached artifacts are stored. The path is expanded, resolved, and created if missing. Uses ``<TIDY3D_BASE_DIR>/cache/simulations`` when ``TIDY3D_BASE_DIR`` is set, otherwise ``<XDG_CACHE_HOME>/tidy3d/simulations`` when ``XDG_CACHE_HOME`` is set, and otherwise ``~/.cache/tidy3d/simulations``.
   * - ``max_size_gb``
     - ``10.0``
     - Yes
     - Maximum cache size in gigabytes. ``0`` disables the size limit.
   * - ``max_entries``
     - ``0``
     - Yes
     - Maximum number of cached simulations retained. ``0`` means no limit and eviction falls back to size constraints.


Batch Data Cache (``config.batch_data_cache``)
----------------------------------------------

Controls the optional in-memory cache for loaded batch task data.

.. list-table::
   :header-rows: 1
   :widths: 24 18 10 48

   * - Option
     - Default
     - Persisted
     - Description
   * - ``enabled``
     - ``True``
     - No
     - Cache batch results in memory when files are below the size threshold.
   * - ``max_total_size_gb``
     - ``1.0``
     - No
     - Cache batch task data only when the combined size of all task data files is at or below this threshold. ``0`` disables the cache.


vGPU (``config.vgpu``)
----------------------

Defaults used for virtual GPU cloud runs.

.. list-table::
   :header-rows: 1
   :widths: 24 18 10 48

   * - Option
     - Default
     - Persisted
     - Description
   * - ``priority``
     - ``None``
     - No
     - Default queue priority for vGPU runs. When set, must be between ``1`` and ``10``.
   * - ``vgpu_allocation``
     - ``None``
     - No
     - Default virtual GPU allocation for vGPU runs. When set, must be a positive whole number supported by the license.
   * - ``ignore_memory_limit``
     - ``None``
     - No
     - Default flag to allow vGPU runs above the estimated memory limit.


Plugins (``config.plugins``)
----------------------------

Container that holds plugin-defined sections. After a plugin calls
``@register_plugin("name")``, its configuration becomes available under
``config.plugins.<name>`` and follows the same persistence and environment variable rules described above (for example ``TIDY3D_PLUGINS__NAME__FIELD``).
