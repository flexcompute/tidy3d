.. currentmodule:: tidy3d

Automatic Differentiation with Autograd 
---------------------------------------

.. toctree::

    ./../../../tidy3d/plugins/autograd/README

.. autosummary::
   :toctree: ../_autosummary/
   :template: module.rst

    tidy3d.plugins.autograd.functions.threshold
    tidy3d.plugins.autograd.functions.rescale
    tidy3d.plugins.autograd.functions.morphological_gradient_external
    tidy3d.plugins.autograd.functions.morphological_gradient_internal
    tidy3d.plugins.autograd.functions.morphological_gradient
    tidy3d.plugins.autograd.functions.grey_closing
    tidy3d.plugins.autograd.functions.grey_opening
    tidy3d.plugins.autograd.functions.grey_erosion
    tidy3d.plugins.autograd.functions.grey_dilation
    tidy3d.plugins.autograd.functions.pad
    tidy3d.plugins.autograd.functions.convolve

    tidy3d.plugins.autograd.utilities.chain
    tidy3d.plugins.autograd.utilities.make_kernel
    tidy3d.plugins.autograd.utilities.get_kernel_size_px

    tidy3d.plugins.autograd.primitives.gaussian_filter

    tidy3d.plugins.autograd.invdes.grey_indicator
    tidy3d.plugins.autograd.invdes.make_circular_filter
    tidy3d.plugins.autograd.invdes.make_conic_filter
    tidy3d.plugins.autograd.invdes.make_curvature_penalty
    tidy3d.plugins.autograd.invdes.make_erosion_dilation_penalty
    tidy3d.plugins.autograd.invdes.make_filter
    tidy3d.plugins.autograd.invdes.make_filter_and_project
    tidy3d.plugins.autograd.invdes.ramp_projection
    tidy3d.plugins.autograd.invdes.tanh_projection

    tidy3d.plugins.autograd.types.PaddingType
    tidy3d.plugins.autograd.types.KernelType

