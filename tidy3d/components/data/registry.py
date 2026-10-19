"""Registry mapping ``DataArray`` subclass names to the classes themselves.

The registry lives here, below :mod:`tidy3d.components.base`, because
``Tidy3dBaseModel`` needs the name lookup to rehydrate HDF5-backed fields but
must not depend on the data layer that defines the classes. The dict starts
empty; :mod:`tidy3d.components.data.data_array.registry` fills it with the
built-in array types, and plugins that add their own array types extend the
same dict.

The values are ``DataArray`` subclasses. They are typed as plain ``type`` so
that this module, which the whole client sits on top of, needs no import of the
data layer even for type checking.
"""

from __future__ import annotations

DATA_ARRAY_MAP: dict[str, type] = {}

__all__ = ["DATA_ARRAY_MAP"]
