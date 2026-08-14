"""Methods for integrating KLayout's DRC with Tidy3D."""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from subprocess import run
from typing import Any

from pydantic import Field, FilePath, field_validator

from tidy3d.components.base import Tidy3dBaseModel
from tidy3d.components.geometry.base import Geometry
from tidy3d.components.simulation import Simulation
from tidy3d.components.structure import Structure
from tidy3d.exceptions import ValidationError, format_chained_exception_message
from tidy3d.log import get_logging_console
from tidy3d.plugins.klayout.drc.defaults import (
    DEFAULT_GDSFILE,
    DEFAULT_RESULTSFILE,
    DEFAULT_VERBOSE,
)
from tidy3d.plugins.klayout.drc.results import DRCResults, _validate_result_limits
from tidy3d.plugins.klayout.util import check_installation

SUPPORTED_DRC_SUFFIXES: frozenset[str] = frozenset({".drc", ".lydrc"})


def _validate_drc_args(value: Any) -> dict[str, str]:
    """Coerce DRC arguments to strings and reject reserved keys."""
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValidationError("drc_args must be a mapping of keys to values.")
    try:
        drc_args = {str(key): str(item) for key, item in value.items()}
    except Exception as error:
        raise ValidationError(
            format_chained_exception_message(
                "Could not coerce keys and values of drc_args to strings",
                error,
            )
        ) from error

    reserved_keys = {"gdsfile", "resultsfile"}
    conflicts = reserved_keys.intersection(drc_args)
    if conflicts:
        conflict_str = ", ".join(sorted(conflicts))
        raise ValidationError(
            f"Invalid DRC argument key(s) {conflict_str}: these names are reserved and "
            "automatically managed by Tidy3D."
        )
    return drc_args


def _validate_drc_runset_format(drc_runset: Path) -> None:
    """Validate the source and report declarations required by the DRC runner."""
    with drc_runset.open("r") as file:
        content = file.read()
    if not re.search(r"source\(\s*\$gdsfile\s*\)", content):
        raise ValidationError(
            "DRC runset is not formatted correctly. The GDS source must be loaded with "
            "'source($gdsfile)'. Please refer to the documentation at "
            "'tidy3d/plugins/klayout/drc/README.md' for more details."
        )
    if not re.search(r"""report\(['"](.*?)['"],\s*\$resultsfile\)""", content):
        raise ValidationError(
            "DRC runset is not formatted correctly. The report must be defined as "
            "'report(\"<your report name>\", $resultsfile)'. Please refer to the documentation at "
            "'tidy3d/plugins/klayout/drc/README.md' for more details."
        )


class DRCConfig(Tidy3dBaseModel):
    """Configuration for KLayout DRC."""

    gdsfile: FilePath = Field(
        title="GDS File",
        description="The path to the GDS file to write the Tidy3D object to.",
    )
    drc_runset: FilePath = Field(
        title="DRC Runset file",
        description="Path to the KLayout DRC runset file.",
    )
    resultsfile: Path = Field(
        title="DRC Results File",
        description="Path to the KLayout DRC results file.",
    )
    verbose: bool = Field(
        title="Verbose",
        description="Whether to print logging.",
    )
    drc_args: dict[str, str] = Field(
        default_factory=dict,
        title="DRC File Arguments",
        description="Optional key/value pairs forwarded to KLayout as -rd <key>=<value> definitions.",
    )

    @field_validator("gdsfile")
    @classmethod
    def _validate_gdsfile_filetype(cls, v: FilePath) -> FilePath:
        """Check GDS filetype is ``.gds``."""
        if v.suffix != ".gds":
            raise ValidationError(f"GDS file '{v}' must end with '.gds'.")
        return v

    @field_validator("drc_runset")
    @classmethod
    def _validate_drc_runset_filetype(cls, v: FilePath) -> FilePath:
        """Check DRC runset filetype is ``.drc`` or ``.lydrc``."""
        if v.suffix not in SUPPORTED_DRC_SUFFIXES:
            raise ValidationError(
                f"DRC runset file '{v}' must end with one of {', '.join(SUPPORTED_DRC_SUFFIXES)}."
            )
        return v

    @field_validator("drc_runset")
    @classmethod
    def _validate_drc_runset_format(cls, v: FilePath) -> FilePath:
        """Check if the DRC runset file is formatted correctly.
        The checks are:
        1. The GDS source must be loaded with 'source($gdsfile)'.
        2. The report must be defined as 'report("<your string>", $resultsfile)'.
        """
        _validate_drc_runset_format(Path(v))
        return v

    @field_validator("drc_args", mode="before")
    @classmethod
    def _validate_drc_args_field(cls, value: Any) -> dict[str, str]:
        """Apply the shared DRC argument validation contract."""
        return _validate_drc_args(value)


class DRCRunner(Tidy3dBaseModel):
    """A class for running KLayout DRC. Can be used to run DRC on a Tidy3D object or a GDS file.

    Parameters
    ----------
    drc_runset : Path
        The path to the KLayout DRC runset file.
    verbose : bool
        Whether to print logging. Default is ``True``.

    Example
    -------
    >>> # Running DRC on a GDS file:
    >>> from tidy3d.plugins.klayout.drc import DRCRunner
    >>> runner = DRCRunner(drc_runset="my_drc_runset.drc", verbose=True) # doctest: +SKIP
    >>> results = runner.run(source="my_layout.gds", resultsfile="drc_results.lyrdb") # doctest: +SKIP
    >>> print(results) # doctest: +SKIP
    >>> # Running DRC on a Tidy3D object:
    >>> import tidy3d as td
    >>> from tidy3d.plugins.klayout.drc import DRCRunner
    >>> vertices = [(-2, 0), (-1, 1), (0, 0.5), (1, 1), (2, 0), (0, -1)]
    >>> geom = td.PolySlab(vertices=vertices, slab_bounds=(0, 0.22), axis=2)
    >>> runner = DRCRunner(drc_runset="my_drc_runset.drc", verbose=True) # doctest: +SKIP
    >>> results = runner.run(source=geom, td_object_gds_savefile="geom.gds", resultsfile="drc_results.lyrdb", z=0.1, gds_layer=0, gds_dtype=0) # doctest: +SKIP
    >>> print(results) # doctest: +SKIP
    """

    drc_runset: FilePath = Field(
        title="DRC Runset file",
        description="Path to the KLayout DRC runset file.",
    )
    verbose: bool = Field(
        default=DEFAULT_VERBOSE,
        title="Verbose",
        description="Whether to print logging.",
    )

    def run(
        self,
        source: Geometry | Structure | Simulation | Path,
        td_object_gds_savefile: Path = DEFAULT_GDSFILE,
        resultsfile: Path = DEFAULT_RESULTSFILE,
        drc_args: dict[str, str] | None = None,
        max_results: int | None = None,
        max_results_per_cell: int | None = None,
        **to_gds_file_kwargs: Any,
    ) -> DRCResults:
        """Runs KLayout's DRC on a GDS file or a Tidy3D object. The Tidy3D object can be a
        :class:`~tidy3d.Geometry`, :class:`~tidy3d.Structure`, or
        :class:`~tidy3d.Simulation`.

        Parameters
        ----------
        source : Union[:class:`~tidy3d.Geometry`, :class:`~tidy3d.Structure`, :class:`~tidy3d.Simulation`, Path]
            The :class:`~tidy3d.Geometry`, :class:`~tidy3d.Structure`,
            :class:`~tidy3d.Simulation`, or GDS file to run DRC on.
        td_object_gds_savefile : Path
            The path to save the Tidy3D object to. Defaults to ``"layout.gds"``.
        resultsfile : Path
            The path to save the KLayout DRC results file to. Defaults to ``"drc_results.lyrdb"``.
        drc_args : Optional[dict[str, str]] = None
            Additional key/value pairs passed through to KLayout as ``-rd key=value`` CLI arguments.
        max_results : Optional[int]
            Maximum number of markers to load from the results file. ``None`` (default) loads all
            markers.
        max_results_per_cell : Optional[int]
            Maximum number of markers to load for each cell. Cannot be combined
            with ``max_results``.
        **to_gds_file_kwargs
            Additional keyword arguments to pass to the Tidy3D object-specific ``to_gds_file()`` method.

        Returns
        -------
        :class:`.DRCResults`
            The DRC results object containing violations and status.

        Example
        -------
        Running DRC on a GDS file:
        >>> from tidy3d.plugins.klayout.drc import DRCRunner
        >>> runner = DRCRunner(drc_runset="my_drc_runset.drc", verbose=True) # doctest: +SKIP
        >>> results = runner.run(source="my_layout.gds") # doctest: +SKIP
        >>> print(results) # doctest: +SKIP

        Running DRC on a Tidy3D object:
        >>> import tidy3d as td
        >>> from tidy3d.plugins.klayout.drc import DRCRunner
        >>> vertices = [(-2, 0), (-1, 1), (0, 0.5), (1, 1), (2, 0), (0, -1)]
        >>> geom = td.PolySlab(vertices=vertices, slab_bounds=(0, 0.22), axis=2)
        >>> runner = DRCRunner(drc_runset="my_drc_runset.drc", verbose=True) # doctest: +SKIP
        >>> results = runner.run(source=geom, z=0.1, gds_layer=0, gds_dtype=0) # doctest: +SKIP
        >>> print(results) # doctest: +SKIP
        """
        _validate_result_limits(
            max_results=max_results,
            max_results_per_cell=max_results_per_cell,
        )

        if isinstance(source, (Geometry, Structure, Simulation)):
            gdsfile = td_object_gds_savefile
            if self.verbose:
                console = get_logging_console()
                console.log(f"Writing Tidy3D object to GDS file '{gdsfile}'.")
            source.to_gds_file(fname=gdsfile, **to_gds_file_kwargs)
        else:
            gdsfile = source

        config = DRCConfig(
            gdsfile=gdsfile,
            drc_runset=self.drc_runset,
            resultsfile=resultsfile,
            verbose=self.verbose,
            drc_args={} if drc_args is None else drc_args,
        )
        return run_drc_on_gds(
            config=config,
            max_results=max_results,
            max_results_per_cell=max_results_per_cell,
        )


def run_drc_on_gds(
    config: DRCConfig,
    max_results: int | None = None,
    max_results_per_cell: int | None = None,
) -> DRCResults:
    """Runs KLayout's DRC on a GDS file.

    Parameters
    ----------
    config : :class:`.DRCConfig`
        The configuration for the DRC run.
    max_results : Optional[int]
        Maximum number of markers to load from the results file. ``None`` (default) loads all
        markers.
    max_results_per_cell : Optional[int]
        Maximum number of markers to load for each cell. Cannot be combined
        with ``max_results``.

    Returns
    -------
    :class:`.DRCResults`
        The DRC results object containing violations and status.

    Example
    -------
    >>> from tidy3d.plugins.klayout.drc import run_drc_on_gds, DRCConfig
    >>> config = DRCConfig(gdsfile="geom.gds", drc_runset="my_drc_runset.drc", resultsfile="drc_results.lyrdb", verbose=True) # doctest: +SKIP
    >>> results = run_drc_on_gds(config) # doctest: +SKIP
    >>> print(results) # doctest: +SKIP
    """
    _validate_result_limits(
        max_results=max_results,
        max_results_per_cell=max_results_per_cell,
    )
    klayout_cmd = check_installation(raise_error=True)

    if config.verbose:
        console = get_logging_console()
        console.log(
            f"Running KLayout DRC on GDS file '{config.gdsfile}' with runset '{config.drc_runset}' and saving results to '{config.resultsfile}'..."
        )
    # run klayout DRC as a subprocess
    cmd = [
        klayout_cmd,
        "-b",
        "-r",
        config.drc_runset,
        "-rd",
        f"gdsfile={config.gdsfile}",
        "-rd",
        f"resultsfile={config.resultsfile}",
    ]

    for key, value in config.drc_args.items():
        cmd.extend(["-rd", f"{key}={value}"])

    output = run(cmd, capture_output=True)

    if output.returncode != 0:
        msg = output.stderr.decode(errors="replace")
        raise RuntimeError(f"KLayout DRC failed with error message: '{msg}'.")
    if config.verbose:
        console.log("KLayout DRC completed successfully.")

    return DRCResults.load(
        resultsfile=config.resultsfile,
        max_results=max_results,
        max_results_per_cell=max_results_per_cell,
    )
