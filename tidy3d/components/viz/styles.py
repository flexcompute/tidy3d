import matplotlib.pyplot as plt
from matplotlib import cycler

from ...log import log
from .color_paletes import CATEGORICAL_PALETTES_HEX

# Tidy3D default plotting style parameters
_TIDY3D_STYLE_PARAMS = {
    "axes.prop_cycle": cycler(color=CATEGORICAL_PALETTES_HEX[""]),
    "axes.grid": True,
    "grid.linestyle": ":",
}

# Store original parameters before applying Tidy3D style
_ORIGINAL_PARAMS = {}

try:
    # Store only the parameters that Tidy3D style will modify
    for key in _TIDY3D_STYLE_PARAMS:
        if key in plt.rcParams:
            _ORIGINAL_PARAMS[key] = plt.rcParams[key]
    # Apply the Tidy3D style automatically on import
    plt.rcParams.update(_TIDY3D_STYLE_PARAMS)
except Exception as e:
    log.error(f"Failed to apply Tidy3D plotting style on import. Error: {e}")
    _ORIGINAL_PARAMS = {}  # Clear original params if application failed


def reset_previous_style():
    """
    Resets matplotlib rcParams to the values they had before the Tidy3D
    style was automatically applied on import.
    """
    if not _ORIGINAL_PARAMS:
        log.warning("No previous Matplotlib style state found to reset to.")
        return

    try:
        plt.rcParams.update(_ORIGINAL_PARAMS)
    except Exception as e:
        log.error(f"Failed to reset previous Matplotlib style. Error: {e}")
