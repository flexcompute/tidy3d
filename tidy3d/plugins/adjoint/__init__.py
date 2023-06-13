"""Imports for adjoint plugin."""

# import the jax version of tidy3d components
try:
    from .components.geometry import JaxBox, JaxPolySlab, JaxGeometryGroup
    from .components.medium import JaxMedium, JaxAnisotropicMedium, JaxCustomMedium
    from .components.structure import JaxStructure
    from .components.simulation import JaxSimulation
    from .components.data.sim_data import JaxSimulationData
    from .components.data.monitor_data import JaxModeData
    from .components.data.dataset import JaxPermittivityDataset
    from .components.data.data_array import JaxDataArray
except ImportError as e:
    raise ImportError(
        "The 'jax' package is required for adjoint plugin and not installed. "
        "To get the appropriate packages, install tidy3d using '[jax]' option, for example: "
        "$pip install 'tidy3d[jax]'."
    ) from e

try:
    from .web import run, run_async
except ImportError:
    pass
