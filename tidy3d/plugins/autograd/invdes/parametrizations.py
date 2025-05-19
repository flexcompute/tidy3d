from __future__ import annotations

from typing import Callable, Optional, Union

import pydantic.v1 as pd
from numpy.typing import NDArray

from tidy3d.components.base import Tidy3dBaseModel
from tidy3d.plugins.autograd.constants import BETA_DEFAULT, ETA_DEFAULT
from tidy3d.plugins.autograd.types import KernelType, PaddingType

from .filters import make_filter
from .projections import tanh_projection


class FilterAndProject(Tidy3dBaseModel):
    """A class that combines filtering and projection operations."""

    radius: Union[float, tuple[float, ...]] = pd.Field(
        ..., title="Radius", description="The radius of the kernel."
    )
    dl: Union[float, tuple[float, ...]] = pd.Field(
        ..., title="Grid Spacing", description="The grid spacing."
    )
    size_px: Union[int, tuple[int, ...]] = pd.Field(
        None, title="Size in Pixels", description="The size of the kernel in pixels."
    )
    beta: pd.NonNegativeFloat = pd.Field(
        BETA_DEFAULT, title="Beta", description="The beta parameter for the tanh projection."
    )
    eta: pd.NonNegativeFloat = pd.Field(
        ETA_DEFAULT, title="Eta", description="The eta parameter for the tanh projection."
    )
    filter_type: KernelType = pd.Field(
        "conic", title="Filter Type", description="The type of filter to create."
    )
    padding: PaddingType = pd.Field(
        "reflect", title="Padding", description="The padding mode to use."
    )

    def __call__(
        self, array: NDArray, beta: Optional[float] = None, eta: Optional[float] = None
    ) -> NDArray:
        """Apply the filter and projection to an input array.

        Parameters
        ----------
        array : np.ndarray
            The input array to filter and project.
        beta : float = None
            The beta parameter for the tanh projection. If None, uses the instance's beta.
        eta : float = None
            The eta parameter for the tanh projection. If None, uses the instance's eta.

        Returns
        -------
        np.ndarray
            The filtered and projected array.
        """
        filter_instance = make_filter(
            radius=self.radius,
            dl=self.dl,
            size_px=self.size_px,
            filter_type=self.filter_type,
            padding=self.padding,
        )
        filtered = filter_instance(array)
        beta = beta if beta is not None else self.beta
        eta = eta if eta is not None else self.eta
        projected = tanh_projection(filtered, beta, eta)
        return projected


def make_filter_and_project(
    radius: Optional[Union[float, tuple[float, ...]]] = None,
    dl: Optional[Union[float, tuple[float, ...]]] = None,
    *,
    size_px: Optional[Union[int, tuple[int, ...]]] = None,
    beta: float = BETA_DEFAULT,
    eta: float = ETA_DEFAULT,
    filter_type: KernelType = "conic",
    padding: PaddingType = "reflect",
) -> Callable:
    """Create a function that filters and projects an array.

    See Also
    --------
    :func:`~parametrizations.FilterAndProject`.
    """
    return FilterAndProject(
        radius=radius,
        dl=dl,
        size_px=size_px,
        beta=beta,
        eta=eta,
        filter_type=filter_type,
        padding=padding,
    )
