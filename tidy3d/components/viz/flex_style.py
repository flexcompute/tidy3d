from ...log import log
from .flex_color_paletes import CATEGORICAL_PALETTES_HEX

_ORIGINAL_PARAMS = {}


def apply_tidy3d_params():
    """
    Applies a set of defaults to the matplotlib params that are following the tidy3d color palettes and design.
    """
    global _ORIGINAL_PARAMS
    try:
        import matplotlib.pyplot as plt
        from matplotlib import cycler

        _TIDY3D_STYLE_PARAMS = {
            "axes.prop_cycle": cycler(color=CATEGORICAL_PALETTES_HEX["flex_distinct"]),
            "axes.grid": True,
            "grid.linestyle": ":",
            "axes.edgecolor": "#ECEBEA",
        }

        try:
            for key in _TIDY3D_STYLE_PARAMS:
                if key in plt.rcParams:
                    _ORIGINAL_PARAMS[key] = plt.rcParams[key]
            plt.rcParams.update(_TIDY3D_STYLE_PARAMS)
        except Exception as e:
            log.error(f"Failed to apply Tidy3D plotting style on import. Error: {e}")
            _ORIGINAL_PARAMS = {}
    except ImportError:
        pass


def reset_previous_style():
    """
    Resets matplotlib rcParams to the values they had before the Tidy3D
    style was automatically applied on import.
    """
    if not _ORIGINAL_PARAMS:
        log.warning("No previous Matplotlib style state found to reset to.")
        return

    try:
        import matplotlib.pyplot as plt

        plt.rcParams.update(_ORIGINAL_PARAMS)
    except ImportError:
        log.error("Matplotlib is not installed on your system. Failed to reset to default styles.")
    except Exception as e:
        log.error(f"Failed to reset previous Matplotlib style. Error: {e}")
