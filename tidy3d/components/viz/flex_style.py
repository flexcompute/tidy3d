from __future__ import annotations

from tidy3d.log import log

_ORIGINAL_PARAMS = None


def apply_tidy3d_params():
    """
    Applies a set of defaults to the matplotlib params that are following the tidy3d color palettes and design.
    """
    global _ORIGINAL_PARAMS
    try:
        import matplotlib as mpl
        import matplotlib.pyplot as plt

        _ORIGINAL_PARAMS = mpl.rcParams.copy()

        try:
            plt.style.use("tidy3d.style")
        except Exception as e:
            log.error(f"Failed to apply Tidy3D plotting style on import. Error: {e}")
            _ORIGINAL_PARAMS = {}
    except ImportError:
        pass


def restore_matplotlib_rcparams():
    """
    Resets matplotlib rcParams to the values they had before the Tidy3D
    style was automatically applied on import.
    """
    global _ORIGINAL_PARAMS
    try:
        import matplotlib.pyplot as plt
        from matplotlib import style

        if not _ORIGINAL_PARAMS:
            style.use("default")
            return

        plt.rcParams.update(_ORIGINAL_PARAMS)
    except ImportError:
        log.error("Matplotlib is not installed on your system. Failed to reset to default styles.")
    except Exception as e:
        log.error(f"Failed to reset previous Matplotlib style. Error: {e}")
