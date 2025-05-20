"""Tool for finding and characterizing lobes in antenna radiation patterns."""

from math import isclose, isnan
from typing import Optional

import numpy as np
import pydantic.v1 as pd
from pandas import DataFrame
from scipy.signal import find_peaks, peak_widths

from ...components.base import Tidy3dBaseModel, cached_property, skip_if_fields_missing
from ...components.types import ArrayFloat1D, ArrayLike, Ax
from ...constants import fp_eps
from ...exceptions import ValidationError
from ...log import log
from .viz import plot_params_lobe_FNBW, plot_params_lobe_peak, plot_params_lobe_width

# The minimum plateau size for peak finding, which is set to 0 to ensure that all peaks are found.
# A value must be provided to retrieve additional information from `find_peaks`.
MIN_PLATEAU_SIZE = 0
DEFAULT_MIN_LOBE_REL_HEIGHT = 1e-3
DEFAULT_NULL_THRESHOLD = 1e-3


class LobeMeasurer(Tidy3dBaseModel):
    """
    Tool for detecting and analyzing lobes in antenna radiation patterns,
    along with their characteristics such as direction and beamwidth.

    Example
    -------
    >>> theta = np.linspace(0, 2 * np.pi, 301, endpoint=False)
    >>> Urad = np.cos(theta) ** 2 * np.cos(3 * theta) ** 2
    >>> lobe_measurer = LobeMeasurer(
    ...     angle=theta,
    ...     radiation_pattern=Urad) # doctest: +SKIP
    >>> lobe_measures = lobe_measurer.lobe_measures # doctest: +SKIP
    """

    angle: ArrayFloat1D = pd.Field(
        ...,
        title="Angle",
        description="A 1-dimensional array of angles in radians. The angles should be "
        "in the range [0, 2π] and should be sorted in ascending order.",
    )

    radiation_pattern: ArrayFloat1D = pd.Field(
        ...,
        title="Radiation Pattern",
        description="A 1-dimensional array of real values representing the radiation pattern "
        "of the antenna measured on a linear scale.",
    )

    apply_cyclic_extension: bool = pd.Field(
        True,
        title="Apply Cyclic Extension",
        description="To enable accurate peak finding near boundaries of the ``angle`` array, "
        "we need to extend the signal using its periodicity. If lobes near the boundaries are not "
        "of interest, this can be set to ``False``.",
    )

    width_measure: float = pd.Field(
        0.5,
        gt=0.0,
        le=1.0,
        title="Beamwidth Measure",
        description="Relative magnitude of the lobes at which the beamwidth is measured. "
        "Default value of ``0.5`` corresponds with the half-power beamwidth.",
    )

    min_lobe_height: float = pd.Field(
        DEFAULT_MIN_LOBE_REL_HEIGHT,
        gt=0.0,
        le=1.0,
        title="Minimum Lobe Height",
        description="Only lobes in the radiation pattern with heights above this value are found. "
        "Lobe heights are measured relative to the maximum value in ``radiation_pattern``.",
    )

    null_threshold: float = pd.Field(
        DEFAULT_NULL_THRESHOLD,
        gt=0.0,
        le=1.0,
        title="Null Threshold",
        description="The threshold for detecting nulls, "
        "which is relative to the maximum value in the ``radiation_pattern``.",
    )

    @pd.validator("angle", always=True)
    def _sorted_angle(cls, val):
        """Ensure the angle array is sorted."""
        if not np.all(np.diff(val) >= 0):
            raise ValidationError("The angle array must be sorted in ascending order.")
        return val

    @pd.validator("radiation_pattern", always=True)
    def _nonnegative_radiation_pattern(cls, val):
        """Ensure the radiation pattern is nonnegative."""
        if not np.all(val >= 0):
            raise ValidationError("Radiation pattern must be nonnegative.")
        return val

    @pd.validator("apply_cyclic_extension", always=True)
    @skip_if_fields_missing(["angle"])
    def _cyclic_extension_valid(cls, val, values):
        if val:
            angle = values.get("angle")
            if np.any(angle < 0) or np.any(angle > 2 * np.pi):
                raise ValidationError(
                    "When using cyclic extension, the angle array must be in the range [0, 2π]."
                )
        return val

    @cached_property
    def lobe_measures(self) -> DataFrame:
        """
        The lobe measures as a pandas ``pandas.DataFrame`` with the following columns:

        - ``direction``: The angular position of the lobe peak.
        - ``magnitude``: The height of the lobe peak.
        - ``beamwidth``: The width of the lobe at the specified beamwidth measure.
        - ``beamwidth magnitude``: The magnitude at which the beamwidth is measured.
        - ``beamwidth bounds``: The bounds of the lobe at the the beamwidth measure.
        - ``FNBW``: The first null beam width (FNBW) of the lobe.
        - ``FNBW bounds``: The locations of the nulls on either side of the lobe.

        Returns
        -------
        DataFrame
            A DataFrame containing all lobe measures, where rows indicate the lobe index.
        """
        if self.apply_cyclic_extension:
            angle, signal = self.cyclic_extension(self.angle, self.radiation_pattern)
        else:
            angle, signal = self.angle, self.radiation_pattern

        peaks, peak_props = find_peaks(
            signal,
            plateau_size=MIN_PLATEAU_SIZE,
            height=self.min_lobe_height,
        )
        if len(peaks) == 0:
            return DataFrame()
        # Find the locations of nulls in the signal
        max_mag = np.max(signal)
        null_height = self.null_threshold * max_mag
        _, null_peak_props = find_peaks(
            -signal,
            plateau_size=MIN_PLATEAU_SIZE,
            height=-null_height,
        )

        # Extract and post process the peak properties from Scipy
        peak_locations = self._extract_peak_locations(angle, peak_props)
        peak_heights = self._extract_peak_heights(peak_props)
        widths, beamwidth_height, beamwidth_min, beamwidth_max = self._calc_peak_widths(
            angle, signal, peaks
        )
        null_locations = self._extract_peak_locations(angle, null_peak_props)
        FNBWs, FNBW_min, FNBW_max = self._first_null_beam_widths(peak_locations, null_locations)

        width_bounds = list(zip(beamwidth_min, beamwidth_max))
        FNBW_bounds = list(zip(FNBW_min, FNBW_max))
        data = {
            "direction": peak_locations,
            "magnitude": peak_heights,
            "beamwidth": widths,
            "beamwidth magnitude": beamwidth_height,
            "beamwidth bounds": width_bounds,
            "FNBW": FNBWs,
            "FNBW bounds": FNBW_bounds,
        }

        dataframe = DataFrame(data)
        if self.apply_cyclic_extension:
            dataframe = self._filter_out_of_bounds_lobes(dataframe)
        return dataframe

    @staticmethod
    def cyclic_extension(angle: ArrayLike, signal: ArrayLike) -> tuple[ArrayLike, ArrayLike]:
        """
        This helper function extends the given `signal` array by leveraging its periodic nature.
        It ensures that the peak finding algorithm in Scipy can reliably detect peaks near the
        minimum and maximum values of the ``angle`` array. The returned arrays are extended to
        the range [-π, 3π).

        Parameters
        ----------
        angle : ArrayLike
            The array of angles, expected to be in the range [0, 2π].
        signal : ArrayLike
            The array of signal values corresponding to the angle array.

        Returns
        -------
        tuple[ArrayLike, ArrayLike]
            A tuple containing the extended `angle` and `signal` arrays.
        """
        # Ensure the last sample is not duplicated if it is at 2π
        if isclose(angle[-1], 2 * np.pi, rel_tol=fp_eps):
            angle = angle[:-1]
            signal = signal[:-1]

        first_locs = np.array(np.logical_and(angle >= 0, angle < np.pi)).nonzero()
        second_locs = np.array(np.logical_and(angle >= np.pi, angle < 2 * np.pi)).nonzero()
        angle = np.concatenate(
            [angle[second_locs] - 2 * np.pi, angle, angle[first_locs] + 2 * np.pi]
        )
        signal = np.concatenate([signal[second_locs], signal, signal[first_locs]])
        return angle, signal

    @staticmethod
    def _extract_peak_locations(angle: ArrayLike, peak_props: dict) -> ArrayLike:
        """Get the peak positions in terms of the angular coordinates."""
        left_locs = peak_props["left_edges"]
        right_locs = peak_props["right_edges"]
        left_x = angle[left_locs]
        right_x = angle[right_locs]
        peaks = (left_x + right_x) / 2
        return peaks

    @staticmethod
    def _extract_peak_heights(peak_props: dict) -> ArrayLike:
        """Get the peak heights."""
        return peak_props["peak_heights"]

    def _calc_peak_widths(
        self, angle: ArrayLike, signal: ArrayLike, peaks: ArrayLike
    ) -> tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
        """Get the peak widths in terms of the angular coordinates."""
        rel_height = 1.0 - self.width_measure
        last_element = len(signal) - 1
        left_ips = np.zeros_like(peaks)
        right_ips = last_element * np.ones_like(peaks)
        dummy_prominence_data = (signal[peaks], left_ips, right_ips)
        widths, width_heights, left_ips, right_ips = peak_widths(
            signal, peaks, rel_height, dummy_prominence_data
        )
        left_x = np.interp(left_ips, np.arange(len(angle)), angle)
        right_x = np.interp(right_ips, np.arange(len(angle)), angle)
        widths = right_x - left_x
        # Check that widths were properly found and are not simply the bounds
        improper_width_mask = np.logical_or(left_ips == 0, right_ips == last_element)
        widths[improper_width_mask] = np.nan
        width_heights[improper_width_mask] = np.nan
        left_x[improper_width_mask] = np.nan
        right_x[improper_width_mask] = np.nan
        return widths, width_heights, left_x, right_x

    @staticmethod
    def _first_null_beam_widths(
        peaks: ArrayLike, nulls: ArrayLike
    ) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
        """Calculate the first null beam width for each lobe. If a beam null is not found,
        the value for the width is set to ``nan``."""
        insertion_indices = np.searchsorted(nulls, peaks)

        left_nulls = np.full_like(peaks, np.nan)
        right_nulls = np.full_like(peaks, np.nan)
        left_mask = (insertion_indices > 0) & (insertion_indices < len(nulls))
        right_mask = (insertion_indices >= 0) & (insertion_indices < len(nulls))
        left_nulls[left_mask] = nulls[insertion_indices[left_mask] - 1]
        right_nulls[right_mask] = nulls[insertion_indices[right_mask]]

        peak_FNBW = right_nulls - left_nulls
        return peak_FNBW, left_nulls, right_nulls

    @staticmethod
    def _filter_out_of_bounds_lobes(dataframe: DataFrame) -> DataFrame:
        """Filter out lobes that are out of bounds of the original signal."""
        filtered_dataframe = dataframe[
            dataframe["direction"].between(0, 2 * np.pi, inclusive="left")
        ]
        return filtered_dataframe.reset_index(drop=True)

    @property
    def main_lobe(self) -> DataFrame:
        """Properties of the main lobe."""
        if self.lobe_measures.empty:
            return DataFrame()
        main_lobe_idx = self.lobe_measures["magnitude"].idxmax()
        return self.lobe_measures.iloc[main_lobe_idx]

    @property
    def side_lobes(self) -> DataFrame:
        """Properties of all side lobes."""
        if self.lobe_measures.empty:
            return DataFrame()
        main_lobe_idx = self.lobe_measures["magnitude"].idxmax()
        side_lobes = self.lobe_measures.drop(main_lobe_idx)
        return side_lobes

    @property
    def sidelobe_level(self) -> Optional[float]:
        """The sidelobe level returned on a linear scale."""
        if self.side_lobes.empty:
            return None
        main_lobe_level = self.main_lobe["magnitude"]
        side_lobes = self.side_lobes
        max_side_lobe_idx = side_lobes["magnitude"].idxmax()
        side_lobe_level = side_lobes.at[max_side_lobe_idx, "magnitude"]
        return side_lobe_level / main_lobe_level

    def plot(
        self,
        lobe_index: int,
        ax: Ax,
        include_beamwidth: bool = True,
        include_FNWB: bool = True,
    ) -> Ax:
        """Annotate an existing ``matplotlib`` axis with lobe locations and widths.

        Parameters
        ----------
        lobe_index : int
            Index of the lobe to plot from the :attr:`lobe_measures`.
        ax : Ax
            ``matplotlib`` axis on which to plot the lobe measure.
        include_beamwidth : bool, optional
            If True, plot the beamwidth markers and fill the beamwidth region.
        include_FNWB : bool, optional
            If True, plot the First Null Beam Width markers and fill the FNBW region.

        Returns
        -------
        Ax
            The matplotlib axis with the plotted lobe measures.
        """

        lobe = self.lobe_measures.iloc[lobe_index]
        direction = lobe["direction"]
        ax.axvline(direction, **plot_params_lobe_peak.to_kwargs())

        # Additional annotations of the plot
        if include_beamwidth and not isnan(lobe["beamwidth magnitude"]):
            width_bounds = lobe["beamwidth bounds"]
            ax.axvline(width_bounds[0], **plot_params_lobe_width.to_kwargs())
            ax.axvline(width_bounds[1], **plot_params_lobe_width.to_kwargs())

        if include_FNWB and not isnan(lobe["FNBW"]):
            FNBW_bounds = lobe["FNBW bounds"]
            ax.axvline(FNBW_bounds[0], **plot_params_lobe_FNBW.to_kwargs())
            ax.axvline(FNBW_bounds[1], **plot_params_lobe_FNBW.to_kwargs())

        return ax

    @pd.root_validator(pre=False)
    def _warn_rf_license(cls, values):
        log.warning(
            "ℹ️ ⚠️ RF simulations are subject to new license requirements in the future. You have instantiated at least one RF-specific component.",
            log_once=True,
        )
        return values
