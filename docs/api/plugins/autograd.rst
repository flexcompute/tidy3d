.. currentmodule:: tidy3d

Automatic Differentiation with Autograd
---------------------------------------

.. toctree::

    ./../../../tidy3d/plugins/autograd/README

Differential Operators
~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: ../_autosummary/
   :template: module.rst

    plugins.autograd.differential_operators.grad
    plugins.autograd.differential_operators.value_and_grad

Optimizers
~~~~~~~~~~

.. autosummary::
   :toctree: ../_autosummary/
   :template: module.rst

    plugins.autograd.optimizers.Adam
    plugins.autograd.optimizers.adam
    plugins.autograd.optimizers.apply_updates
    plugins.autograd.optimizers.optimize

Functions
~~~~~~~~~

.. autosummary::
   :toctree: ../_autosummary/
   :template: module.rst

    plugins.autograd.functions.add_at
    plugins.autograd.functions.convolve
    plugins.autograd.functions.grey_closing
    plugins.autograd.functions.grey_dilation
    plugins.autograd.functions.grey_erosion
    plugins.autograd.functions.grey_opening
    plugins.autograd.functions.interpn
    plugins.autograd.functions.least_squares
    plugins.autograd.functions.morphological_gradient
    plugins.autograd.functions.morphological_gradient_external
    plugins.autograd.functions.morphological_gradient_internal
    plugins.autograd.functions.pad
    plugins.autograd.functions.rescale
    plugins.autograd.functions.smooth_max
    plugins.autograd.functions.smooth_min
    plugins.autograd.functions.threshold
    plugins.autograd.functions.trapz

Utilities
~~~~~~~~~

.. autosummary::
   :toctree: ../_autosummary/
   :template: module.rst

    plugins.autograd.utilities.chain
    plugins.autograd.utilities.get_kernel_size_px
    plugins.autograd.utilities.make_kernel
    plugins.autograd.utilities.scalar_objective

Primitives
~~~~~~~~~~

.. autosummary::
   :toctree: ../_autosummary/
   :template: module.rst

    plugins.autograd.primitives.gaussian_filter
    plugins.autograd.primitives.interpolate_spline

Inverse Design
~~~~~~~~~~~~~~

.. autosummary::
   :toctree: ../_autosummary/
   :template: module.rst

    plugins.autograd.invdes.CircularFilter
    plugins.autograd.invdes.ConicFilter
    plugins.autograd.invdes.ErosionDilationPenalty
    plugins.autograd.invdes.FilterAndProject
    plugins.autograd.invdes.GaussianFilter
    plugins.autograd.invdes.grey_indicator
    plugins.autograd.invdes.initialize_params_from_simulation
    plugins.autograd.invdes.make_circular_filter
    plugins.autograd.invdes.make_conic_filter
    plugins.autograd.invdes.make_curvature_penalty
    plugins.autograd.invdes.make_erosion_dilation_penalty
    plugins.autograd.invdes.make_filter
    plugins.autograd.invdes.make_filter_and_project
    plugins.autograd.invdes.make_gaussian_filter
    plugins.autograd.invdes.ramp_projection
    plugins.autograd.invdes.smoothed_projection
    plugins.autograd.invdes.symmetrize_diagonal
    plugins.autograd.invdes.symmetrize_mirror
    plugins.autograd.invdes.symmetrize_rotation
    plugins.autograd.invdes.tanh_projection
