"""Print one JSON line of ``import tidy3d`` costs; run in a fresh interpreter.

``seconds`` is the tidy3d import time, ``ratio`` divides it by the numpy + pydantic +
xarray import time in the same process, and ``built`` counts models whose pydantic
schema was built during import. ``tests/test_package/test_import_budget.py`` budgets these.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator

t0 = time.perf_counter()
import numpy  # noqa: E402, F401
import pydantic  # noqa: E402, F401
import xarray  # noqa: E402, F401

t1 = time.perf_counter()
import tidy3d as td  # noqa: E402

t2 = time.perf_counter()


def subclasses(cls: type) -> Iterator[type]:
    for sub in cls.__subclasses__():
        yield sub
        yield from subclasses(sub)


built = {c for c in subclasses(td.Tidy3dBaseModel) if c.__pydantic_complete__}
print(json.dumps({"seconds": t2 - t1, "built": len(built), "ratio": (t2 - t1) / (t1 - t0)}))
