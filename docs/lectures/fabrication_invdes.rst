Fabrication-aware inverse design
================================

.. meta::
   :description: Fabrication-aware inverse design seminar notebooks and walkthrough
   :keywords: tidy3d, inverse design, fabrication-aware, adjoint, grating coupler

The October 9, 2025 seminar walks through a complete dual-layer grating coupler workflow: start from a uniform baseline, pull a strong seed design with Bayesian optimization, switch to adjoint gradients for per-tooth control, study fabrication sensitivities, and close the loop with measurement-driven calibration. Everything runs inside Tidy3D, so you can rerun the exact same jobs or adapt the utilities to your own device stack.

Seminar recording: `YouTube link <https://www.youtube.com/watch?v=OpVBJmomzoo>`_

Notebook lineup
----------------
* :doc:`Setup Guide: Building the Simulation <../notebooks/2025-10-09-invdes-seminar/00_setup_guide>` - builds the nominal SiN stack, launches the reference simulation, and visualizes the initial geometry so the later notebooks can reuse the cached job ID.
* :doc:`Bayesian Optimization: Finding a Strong Baseline <../notebooks/2025-10-09-invdes-seminar/01_bayes>` - uses a five-parameter Bayesian search to quickly find a good uniform grating. This provides a practical baseline before investing in gradients.
* :doc:`Adjoint Optimization: High-Dimensional Refinement <../notebooks/2025-10-09-invdes-seminar/02_adjoint>` - expands to per-tooth parameters and applies Adam with adjoint sensitivities to apodize the grating and boost efficiency.
* :doc:`Fabrication Sensitivity Analysis: Is Our Design Robust? <../notebooks/2025-10-09-invdes-seminar/03_sensitivity>` - sweeps :math:`\pm 20` nm etch bias, runs Monte Carlo samples, and logs adjoint-derived sensitivity units (:math:`\Delta` objective / :math:`\Delta` parameter) so readers understand what the gradients mean physically.
* :doc:`Robust Adjoint Optimization for Manufacturability <../notebooks/2025-10-09-invdes-seminar/04_adjoint_robust>` - penalizes variance across nominal/over/under corners, illustrating a fabrication-aware adjoint loop that matches what we demoed live.
* :doc:`Monte Carlo View: Nominal vs Robust Grating <../notebooks/2025-10-09-invdes-seminar/05_robust_comparison>` - reruns the Monte Carlo campaign for both nominal and robust devices to quantify yield improvements.
* :doc:`Measurement Calibration: Bridging Simulation and Fabrication <../notebooks/2025-10-09-invdes-seminar/06_measurement_calibration>` - demonstrates gradient-based calibration of tooth widths against (synthetic) spectra, using adjoint sensitivities to recover the as-fabricated geometry from optical measurements.

Getting the code
----------------
The notebooks are available in the `Tidy3D notebooks repository <https://github.com/flexcompute/tidy3d-notebooks/tree/develop/2025-10-09-invdes-seminar>`_. You will need the ``.ipynb`` files as well as the helper scripts `setup.py <https://github.com/flexcompute/tidy3d-notebooks/blob/develop/2025-10-09-invdes-seminar/setup.py>`_ and `optim.py <https://github.com/flexcompute/tidy3d-notebooks/blob/develop/2025-10-09-invdes-seminar/optim.py>`_ to run the examples.

How to run the series
---------------------
1. Install ``tidy3d`` and ``bayesian-optimization`` (``pip install tidy3d bayesian-optimization``) and configure your API key.
2. Execute the notebooks in order; each step writes results into ``results/`` and later notebooks assume those JSON files exist.

Supporting assets
-----------------
* `setup.py <https://github.com/flexcompute/tidy3d-notebooks/blob/develop/2025-10-09-invdes-seminar/setup.py>`_ - shared simulation builders, fabrication constraints, and helper functions.
* `optim.py <https://github.com/flexcompute/tidy3d-notebooks/blob/develop/2025-10-09-invdes-seminar/optim.py>`_ - a lightweight, autograd-friendly Adam implementation with parameter clipping.
* ``results/`` - JSON checkpoints (Bayes best point, adjoint refinements, robust design) consumed by subsequent notebooks.




