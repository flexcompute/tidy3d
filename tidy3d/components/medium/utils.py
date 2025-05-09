import functools
from typing import Callable

import numpy as np

from tidy3d.constants import fp_eps
from tidy3d.log import log

from .constants import FREQ_EVAL_INF


def ensure_freq_in_range(eps_model: Callable[[float], complex]) -> Callable[[float], complex]:
    """Decorate ``eps_model`` to log warning if frequency supplied is out of bounds."""

    @functools.wraps(eps_model)
    def _eps_model(self, frequency: float) -> complex:
        """New eps_model function."""
        # evaluate infs and None as FREQ_EVAL_INF
        is_inf_scalar = isinstance(frequency, float) and np.isinf(frequency)
        if frequency is None or is_inf_scalar:
            frequency = FREQ_EVAL_INF

        if isinstance(frequency, np.ndarray):
            frequency = frequency.astype(float)
            frequency[np.where(np.isinf(frequency))] = FREQ_EVAL_INF

        # if frequency range not present just return original function
        if self.frequency_range is None:
            return eps_model(self, frequency)

        fmin, fmax = self.frequency_range
        # don't warn for evaluating infinite frequency
        if is_inf_scalar:
            return eps_model(self, frequency)
        if np.any(frequency < fmin * (1 - fp_eps)) or np.any(frequency > fmax * (1 + fp_eps)):
            log.warning(
                "frequency passed to 'Medium.eps_model()'"
                f"is outside of 'Medium.frequency_range' = {self.frequency_range}",
                capture=False,
            )
        return eps_model(self, frequency)

    return _eps_model
