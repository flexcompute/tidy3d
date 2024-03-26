# LEGACY: where we define transformations applied to parameters from design region

import typing


import tidy3d.plugins.adjoint.utils.filter as tda_filter

# make these classes importable through `tidy3d.plugins.invdes`
CircularFilter = tda_filter.CircularFilter
ConicFilter = tda_filter.ConicFilter
BinaryProjector = tda_filter.BinaryProjector


# define the allowable transformation types for the `DesignRegion.transformations` field
TransformationType = typing.Union[CircularFilter, ConicFilter, BinaryProjector]
