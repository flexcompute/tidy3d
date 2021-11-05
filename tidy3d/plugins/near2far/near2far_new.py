"""Near field to far field transformation plugin."""
from typing import Dict, Tuple

import numpy as np
import pydantic as pd
import xarray as xr

from ...constants import C_0, ETA_0
from ...components.data import CollectionData, FieldData
from ...components import FieldMonitor
from ...log import SetupError


class Near2Far:
    """tool for projecting from near field data to far field data"""

    def __init__(self, near_field_data: Dict[str, FieldData], frequency: float):
        """create a near2far object"""

        for name, field_data in near_field_data.items():
            pass
            # make sure near_field_data is planar.
            # make sure near_field_data has planar E, H components.
            # make sure frequency is contained in near_field_data.
            # construct the J, H data on the surface at frequency

    def _compute_radiation_vectors(
        self, theta: float, phi: float, axis: int
    ) -> Tuple[CollectionData, CollectionData]:
        """computes N and L vectors for theta and phi values using the near field J, M."""

    def project_fields(self, x: float, y: float, z: float) -> FieldData:
        """Project to x,y,z,frequency points described by far field monitor."""

    def scattered_power(self, phis: float, thetas: float, axis: int) -> CollectionData:
        """Get power scattered to various angles."""

        # for name, field_data in near_field_data.items():
        #     try:
        #         field_data.sel(f=val)
        #     except Exception as e:
        #         raise SetupError(f"frequency {frequency} not found in near field data named {name}")
