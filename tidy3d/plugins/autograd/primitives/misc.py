from __future__ import annotations

import autograd.numpy as np
import scipy.ndimage
from autograd.extend import defjvp, defvjp, primitive

gaussian_filter = primitive(scipy.ndimage.gaussian_filter)
defvjp(
    gaussian_filter,
    lambda ans, x, *args, **kwargs: lambda g: gaussian_filter(g, *args, **kwargs),
)

np.unwrap = primitive(np.unwrap)
defjvp(np.unwrap, lambda g, ans, x, *args, **kwargs: g)
defvjp(np.unwrap, lambda ans, x, *args, **kwargs: lambda g: g)
