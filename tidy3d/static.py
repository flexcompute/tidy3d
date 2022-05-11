"""Tools for dealing with static Simulation objects."""
import functools

from .log import Tidy3dError, log


class StaticException(Tidy3dError):
    """An exception when the tidy3d object has changed."""


class MakeStatic:
    """Context manager that stores a hash and checks if the object has changed upon teardown."""

    def __init__(self, simulation: "Simulation"):
        """When context manager is initalized, return a copy of the simulation."""
        if simulation.type != "Simulation":
            raise ValueError("simulation must be a ``Simulation`` type.")
        self.simulation = simulation.copy(deep=True)
        self.original_hash = self.simulation._freeze()

    def __enter__(self):
        """Freeze the simulation when entering context."""
        log.debug("-> entering static context")
        return self.simulation

    def __exit__(self, *args):
        """Unfeeze the simulation when leaving context, check the hashes equal."""
        log.debug("<- done with static context")
        final_hash = self.simulation._unfreeze()
        if final_hash != self.original_hash:
            raise StaticException("Simulation has changed in static context.")


def make_static(function):
    """Decorates a function to make the first argument (Simulation) static during the call."""

    @functools.wraps(function)
    def static_function(*args, **kwargs):

        # the first argument is assumed to be the simulation, note this can be customized later.
        sim_original, *args = args

        # call the original function within the static context manager
        with MakeStatic(sim_original) as sim_static:
            new_args = sim_static, *args
            ret_value = function(*new_args, **kwargs)

        return ret_value

    return static_function
