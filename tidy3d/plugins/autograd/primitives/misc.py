from __future__ import annotations

import autograd.numpy as anp
from autograd.extend import defjvp, defvjp, primitive
from flexcompute.core._migration.em.numerical.raw.spatial import gaussian_filter

__all__ = ["gaussian_filter"]

anp.unwrap = primitive(anp.unwrap)
defjvp(anp.unwrap, lambda g, ans, x, *args, **kwargs: g)
defvjp(anp.unwrap, lambda ans, x, *args, **kwargs: lambda g: g)
