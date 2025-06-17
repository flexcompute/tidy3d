"""Methods for loading and parsing KLayout DRC results."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Union

import pydantic.v1 as pd

from tidy3d.components.base import Tidy3dBaseModel, cached_property
from tidy3d.components.types import Coordinate2D
from tidy3d.exceptions import FileError

# Types for DRC markers
DRCEdge = tuple[Coordinate2D, Coordinate2D]
DRCEdgePair = tuple[DRCEdge, DRCEdge]
DRCPolygon = tuple[Coordinate2D, ...]
DRCMultiPolygon = tuple[DRCPolygon, ...]


def parse_edge(value: str) -> EdgeMarker:
    """
    Extract coordinates from edge format: ``(x1,y1;x2,y2)``.

    Parameters
    ----------
    value : str
        The edge value string from DRC result database, with format ``(x1,y1;x2,y2)``.

    Returns
    -------
    :class:`.EdgeMarker`
        :class:`.EdgeMarker` containing start and end points of the edge.

    Raises
    ------
    ValueError
        If the edge format is invalid.
    """
    # Extract coordinates from edge format: (x1,y1;x2,y2)
    pattern = r"\(([\d.-]+),([\d.-]+);([\d.-]+),([\d.-]+)\)"
    match = re.match(pattern, value)
    if match:
        coords = [float(x) for x in match.groups()]
        return EdgeMarker(edge=((coords[0], coords[1]), (coords[2], coords[3])))
    raise ValueError(f"Invalid edge format: '{value}'.")


def parse_edge_pair(value: str) -> EdgePairMarker:
    """
    Extract coordinates from edge-pair format: ``(x1,y1;x2,y2)|(x3,y3;x4,y4)``.

    Parameters
    ----------
    value : str
        The edge-pair value string from DRC result database, with format ``(x1,y1;x2,y2)|(x3,y3;x4,y4)``.

    Returns
    -------
    :class:`.EdgePairMarker`
        :class:`.EdgePairMarker` containing both edges' coordinates.

    Raises
    ------
    ValueError
        If the edge-pair format is invalid.
    """
    # Extract coordinates from edge-pair format: (x1,y1;x2,y2)|(x3,y3;x4,y4)
    pattern = (
        r"\(([\d.-]+),([\d.-]+);([\d.-]+),([\d.-]+)\)\|\(([\d.-]+),([\d.-]+);([\d.-]+),([\d.-]+)\)"
    )
    match = re.match(pattern, value)
    if match:
        coords = [float(x) for x in match.groups()]
        return EdgePairMarker(
            edge_pair=(
                ((coords[0], coords[1]), (coords[2], coords[3])),
                ((coords[4], coords[5]), (coords[6], coords[7])),
            )
        )
    raise ValueError(f"Invalid edge-pair format: '{value}'.")


def parse_polygon_coordinates(coords_str: str) -> DRCPolygon:
    """Parse coordinates for a single polygon into a tuple of (x, y) pairs.

    Parameters
    ----------
    coords_str : str
        The string of coordinates for a single polygon, with format ``x1,y1;x2,y2;...``.

    Returns
    -------
    DRCPolygon
        A tuple of (x, y) pairs for the polygon.

    Raises
    ------
    ValueError
        If the coordinate format is invalid.
    """
    coords_pattern = r"([\d.-]+),([\d.-]+)"  # match the coordinates

    # Split by semicolon to validate each coordinate pair
    coord_pairs = coords_str.split(";")

    # Each coordinate pair should match the pattern exactly
    coords = []
    for pair in coord_pairs:
        match = re.match(coords_pattern, pair)
        if not match:
            raise ValueError(f"Invalid coordinate pair in polygon: '{pair}'.")
        coords.append((float(match.group(1)), float(match.group(2))))

    return tuple(coords)


def parse_polygons(value: str) -> MultiPolygonMarker:
    """
    Extract coordinates from polygon format: ``(x1,y1;x2,y2;...)`` including multiple polygons separated by ``/``.

    Parameters
    ----------
    value : str
        The polygon value string from DRC result database, with format ``(x1,y1;x2,y2;...)``
        or multiple polygons separated by ``/`` like ``(x1,y1;.../x3,y3;...)``.

    Returns
    -------
    :class:`.MultiPolygonMarker`
        :class:`.MultiPolygonMarker` containing one or more polygon shapes.

    Raises
    ------
    ValueError
        If the polygon format is invalid or contains incomplete coordinate pairs.
    """
    # Extract the full content inside outer parentheses
    outer_pattern = r"\((.*)\)"
    match = re.match(outer_pattern, value)
    if not match:
        raise ValueError(f"Invalid polygon format: '{value}'.")

    coords_content = match.group(1)

    # Parse multiple polygons separated by '/'
    polygon_parts = coords_content.split("/")
    polygons = []
    for part in polygon_parts:
        polygons.append(parse_polygon_coordinates(part.strip()))

    return MultiPolygonMarker(polygons=tuple(polygons))


def parse_violation_value(value: str) -> Union[EdgeMarker, EdgePairMarker, MultiPolygonMarker]:
    """
    Parse a violation value based on its type (edge, edge-pair, or polygon).

    Parameters
    ----------
    value : str
        The value string from DRC result database.

    Returns
    -------
    Union[:class:`.EdgeMarker`, :class:`.EdgePairMarker`, :class:`.MultiPolygonMarker`]
        The parsed violation marker.

    Raises
    ------
    ValueError
        If the violation marker type is invalid.
    """
    if value.startswith("edge: "):
        return parse_edge(value=value.replace("edge: ", ""))
    elif value.startswith("edge-pair: "):
        return parse_edge_pair(value=value.replace("edge-pair: ", ""))
    elif value.startswith("polygon: "):
        return parse_polygons(value=value.replace("polygon: ", ""))
    raise ValueError(
        f"Invalid marker type (should start with 'edge:', 'edge-pair:', or 'polygon:'): '{value}'."
    )


class EdgeMarker(Tidy3dBaseModel):
    """A class for storing KLayout DRC edge marker results."""

    edge: DRCEdge = pd.Field(
        title="DRC Edge Marker",
        description="The edge marker of the DRC violation. The format is ((x1, y1), (x2, y2)).",
    )


class EdgePairMarker(Tidy3dBaseModel):
    """A class for storing KLayout DRC edge pair marker results."""

    edge_pair: DRCEdgePair = pd.Field(
        title="DRC Edge Pair Marker",
        description="The edge pair marker of the DRC violation. The format is (edge1, edge2), where an edge has format ((x1, y1), (x2, y2)).",
    )


class MultiPolygonMarker(Tidy3dBaseModel):
    """A class for storing KLayout DRC multi-polygon marker results."""

    polygons: DRCMultiPolygon = pd.Field(
        title="DRC Multi-Polygon Marker",
        description="The multi-polygon marker of the DRC violation. The format is (polygon1, polygon2, ...), where each polygon has format ((x1, y1), (x2, y2), ...).",
    )


DRCMarker = Union[EdgeMarker, EdgePairMarker, MultiPolygonMarker]


class DRCViolation(Tidy3dBaseModel):
    """A class for storing KLayout DRC violation results for a single category."""

    category: str = pd.Field(
        title="DRC Violation Category", description="The category of the DRC violation."
    )
    markers: tuple[DRCMarker, ...] = pd.Field(
        title="DRC Markers", description="Tuple of DRC markers in this category."
    )

    @cached_property
    def count(self) -> int:
        """The number of DRC markers in this category."""
        return len(self.markers)

    def __str__(self) -> str:
        """Get a nice string summary of the number of markers in this category."""
        return f"{self.category}: {self.count}"


class DRCResults(Tidy3dBaseModel):
    """A class for loading and storing KLayout DRC results."""

    violations_by_category: dict[str, DRCViolation] = pd.Field(
        title="DRC Violations", description="Dictionary of DRC violations by category."
    )

    @cached_property
    def is_clean(self) -> bool:
        """Whether the DRC is clean (no violations)."""
        return all(v.count == 0 for v in self.violations_by_category.values())

    @cached_property
    def violation_counts(self) -> dict[str, int]:
        """Counts violations by category.

        Returns
        -------
        dict[str, int]
            A dictionary of violation counts for each category.
        """
        return {
            category: violation.count for category, violation in self.violations_by_category.items()
        }

    @cached_property
    def categories(self) -> tuple[str, ...]:
        """A tuple of all DRC categories."""
        return tuple(self.violations_by_category.keys())

    def __getitem__(self, category: str) -> DRCViolation:
        """Get DRC violation result by category.

        Parameters
        ----------
        category : str
            The category of the DRC violation.

        Returns
        -------
        :class:`.DRCViolation`
            The DRC violation result for the given category.
        """
        return self.violations_by_category[category]

    def __str__(self) -> str:
        """Get a nice string representation of the DRC results."""
        summary = "DRC results summary\n"
        summary += "--------------------------------\n"
        summary += f"Total violations: {sum(violation.count for violation in self.violations_by_category.values())}\n\n"
        summary += "Violations by category:\n"
        for violation in self.violations_by_category.values():
            summary += violation.__str__() + "\n"
        return summary

    @classmethod
    def load(cls, resultsfile: Union[str, Path]) -> DRCResults:
        """Create a :class:`.DRCResults` instance from a results file.

        Parameters
        ----------
        resultsfile : Union[str, Path]
            Path to the KLayout DRC results file.

        Returns
        -------
        :class:`.DRCResults`
            A :class:`.DRCResults` object containing the DRC results.

        Raises
        ------
        FileNotFoundError
            If the DRC result file is not found.
        ET.ParseError
            If the DRC result file is not a valid XML file.

        Example
        -------
        >>> from tidy3d.plugins.klayout.drc import DRCResults
        >>> results = DRCResults.load(resultsfile="drc_results.lyrdb") # doctest: +SKIP
        >>> print(results) # doctest: +SKIP
        """
        return cls(violations_by_category=violations_from_file(resultsfile=resultsfile))


def violations_from_file(resultsfile: Union[str, Path]) -> dict[str, DRCViolation]:
    """Loads a KLayout DRC results file and returns the results as a dictionary of :class:`.DRCViolation` objects.

    Parameters
    ----------
    resultsfile : Union[str, Path]
        Path to the KLayout DRC results file.

    Returns
    -------
    dict[str, :class:`.DRCViolation`]
        A dictionary of :class:`.DRCViolation` objects for each category.

    Raises
    ------
    FileNotFoundError
        If the DRC result file is not found.
    ET.ParseError
        If the DRC result file is not a valid XML file.
    """
    # Parse the results file
    try:
        xmltree = ET.parse(resultsfile)
    except FileNotFoundError as err:
        raise FileError(f"DRC result file not found: '{resultsfile}'.") from err
    except ET.ParseError as err:
        raise ET.ParseError(f"Invalid XML format in DRC result file: '{resultsfile}'.") from err

    # Initialize violations dict with all the categories
    violations = {}
    for category in xmltree.getroot().findall(".//categories/category/name"):
        violations[category.text] = DRCViolation(category=category.text, markers=())

    # Parse markers
    for item in xmltree.getroot().findall(".//item"):
        category = item.find("category").text
        value = item.find("values/value").text
        marker = parse_violation_value(value)
        violations[category] = DRCViolation(
            category=category,
            markers=(*violations[category].markers, marker),
        )
    return violations
