"""Running KLayout LVS from Tidy3D."""

from __future__ import annotations

import os
import re
import secrets
from collections.abc import Mapping
from pathlib import Path
from subprocess import PIPE, STDOUT, run
from typing import TYPE_CHECKING, Any

from pydantic import Field, FilePath, field_validator, model_validator

from tidy3d.components.base import Tidy3dBaseModel
from tidy3d.exceptions import ValidationError, format_chained_exception_message
from tidy3d.log import get_logging_console, log
from tidy3d.plugins.klayout.lvs.defaults import (
    DEFAULT_RESULTSFILE,
    DEFAULT_VERBOSE,
    FILE_VARIABLES,
)
from tidy3d.plugins.klayout.lvs.results import load_results
from tidy3d.plugins.klayout.util import check_installation

if TYPE_CHECKING:
    from tidy3d.plugins.klayout.lvs.results import LVSResults

SUPPORTED_LVS_SUFFIXES: frozenset[str] = frozenset({".lvs", ".lylvs"})
SUPPORTED_LAYOUT_SUFFIXES: frozenset[str] = frozenset({".gds", ".oas"})
_OUTPUT_TAIL_LINES = 50
_VARIABLE_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _runset_variable_names(renames: Mapping[str, str]) -> dict[str, str]:
    """Return the runset's name for each file variable."""
    return {variable: renames.get(variable, variable) for variable in FILE_VARIABLES}


def _validate_lvs_runset(value: Path) -> Path:
    """Check the runset's suffix and make its path absolute."""
    if value.suffix not in SUPPORTED_LVS_SUFFIXES:
        raise ValidationError(f"LVS runset '{value}' must end in '.lvs' or '.lylvs'.")
    return value.absolute()


def _validate_runset_variables(value: Any) -> dict[str, str]:
    """Check that ``runset_variables`` gives each file variable its own valid name."""
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValidationError(
            "'runset_variables' must be a mapping from file variables to runset variable names."
        )
    unknown = [repr(key) for key in value if key not in FILE_VARIABLES]
    if unknown:
        raise ValidationError(
            f"Unknown file variable(s) in 'runset_variables': {', '.join(unknown)}. The keys must "
            "be 'gdsfile', 'schematicfile' or 'resultsfile'."
        )
    renames = {}
    for variable, name in value.items():
        # KLayout treats '-rd $input=...' like '-rd input=...'.
        bare_name = name.removeprefix("$") if isinstance(name, str) else name
        if not isinstance(bare_name, str) or not _VARIABLE_NAME.fullmatch(bare_name):
            raise ValidationError(
                f"Invalid runset variable name {name!r} for '{variable}': use letters, digits and "
                "underscores, and don't start with a digit."
            )
        renames[variable] = bare_name
    owners: dict[str, str] = {}
    for variable, name in _runset_variable_names(renames).items():
        if name in owners:
            raise ValidationError(
                f"'{owners[name]}' and '{variable}' can't share the runset variable '{name}'. "
                "Give each file variable its own name in 'runset_variables'."
            )
        owners[name] = variable
    return renames


def _validate_lvs_args(value: Any) -> dict[str, str]:
    """Coerce LVS arguments to strings."""
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValidationError("'lvs_args' must be a mapping of keys to values.")
    try:
        return {str(key): str(item) for key, item in value.items()}
    except Exception as error:
        raise ValidationError(
            format_chained_exception_message(
                "Could not coerce keys and values of 'lvs_args' to strings", error
            )
        ) from error


class LVSConfig(Tidy3dBaseModel):
    """Configuration for a KLayout LVS run.

    Every path is made absolute, because KLayout resolves relative paths against the layout's
    folder.
    """

    gdsfile: FilePath = Field(
        title="Layout file",
        description="Path to the layout, a '.gds' or '.oas' file.",
    )
    lvs_runset: FilePath = Field(
        title="LVS runset",
        description="Path to the KLayout LVS runset, a '.lvs' or '.lylvs' file.",
    )
    schematicfile: FilePath = Field(
        title="Schematic file",
        description="Path to the schematic netlist, which the runset reads.",
    )
    resultsfile: Path = Field(
        default=DEFAULT_RESULTSFILE,
        title="Results file",
        description="Path to write the LVS results file to. Its folder must exist, and it can't "
        "be one of the input files.",
    )
    runset_variables: dict[str, str] = Field(
        default_factory=dict,
        title="Runset variables",
        description="The runset's names for the file variables 'gdsfile', 'schematicfile' and "
        "'resultsfile', where they differ, such as ``{'gdsfile': 'input'}``.",
    )
    lvs_args: dict[str, str] = Field(
        default_factory=dict,
        title="LVS arguments",
        description="Other key/value pairs forwarded to KLayout as -rd <key>=<value> definitions.",
    )
    verbose: bool = Field(
        default=DEFAULT_VERBOSE,
        title="Verbose",
        description="Whether to log the start and the verdict of the run.",
    )

    @field_validator("gdsfile")
    @classmethod
    def _validate_gdsfile(cls, value: Path) -> Path:
        """Check that the layout is a GDS or OASIS file, and make its path absolute."""
        if value.suffix not in SUPPORTED_LAYOUT_SUFFIXES:
            raise ValidationError(f"Layout file '{value}' must end in '.gds' or '.oas'.")
        return value.absolute()

    @field_validator("lvs_runset")
    @classmethod
    def _validate_lvs_runset_field(cls, value: Path) -> Path:
        """Check the runset's suffix and make its path absolute."""
        return _validate_lvs_runset(value)

    @field_validator("schematicfile")
    @classmethod
    def _validate_schematicfile(cls, value: Path) -> Path:
        """Make the schematic's path absolute."""
        return value.absolute()

    @field_validator("resultsfile")
    @classmethod
    def _validate_resultsfile(cls, value: Path) -> Path:
        """Check that the results file can be written, and make its path absolute."""
        path = value.absolute()
        if not path.parent.is_dir():
            raise ValidationError(f"The folder of results file '{path}' doesn't exist.")
        if path.is_dir():
            raise ValidationError(f"Results file '{path}' is a folder.")
        return path

    @field_validator("runset_variables", mode="before")
    @classmethod
    def _validate_runset_variables_field(cls, value: Any) -> dict[str, str]:
        """Check the runset variable names."""
        return _validate_runset_variables(value)

    @field_validator("lvs_args", mode="before")
    @classmethod
    def _validate_lvs_args_field(cls, value: Any) -> dict[str, str]:
        """Coerce LVS arguments to strings."""
        return _validate_lvs_args(value)

    @model_validator(mode="after")
    def _validate_lvs_args_keys(self) -> LVSConfig:
        """Reject ``lvs_args`` keys that would redefine a file variable."""
        names = set(_runset_variable_names(self.runset_variables).values())
        conflicts = [repr(key) for key in self.lvs_args if key.removeprefix("$") in names]
        if conflicts:
            self._raise_validation_error_at_loc(
                f"Invalid 'lvs_args' key(s) {', '.join(conflicts)}: the runner sets these runset "
                "variables to the file paths.",
                "lvs_args",
            )
        return self

    @model_validator(mode="after")
    def _validate_resultsfile_not_input(self) -> LVSConfig:
        """Reject a results path that names an input file, which the run would replace."""
        if not self.resultsfile.exists():
            return self
        inputs = (
            ("layout", self.gdsfile),
            ("schematic", self.schematicfile),
            ("LVS runset", self.lvs_runset),
        )
        for label, path in inputs:
            # samefile also catches links and other spellings of the same path.
            if self.resultsfile.samefile(path):
                self._raise_validation_error_at_loc(
                    f"Results file '{self.resultsfile}' is the same file as the {label} "
                    f"'{path}', which the run would replace with the results. Choose another "
                    "results path.",
                    "resultsfile",
                )
        return self


class LVSRunner(Tidy3dBaseModel):
    """Runs KLayout LVS, which extracts a netlist from a layout and compares it with a
    schematic netlist.

    KLayout executes the runset in a separate process. The runset receives the layout,
    schematic and results paths in the variables ``$gdsfile``, ``$schematicfile`` and
    ``$resultsfile``; use ``runset_variables`` for a runset that uses other names. KLayout
    writes the results file only if the runset calls ``report_lvs`` and runs ``compare``.

    Parameters
    ----------
    lvs_runset : Path
        Path to the KLayout LVS runset, a ``.lvs`` or ``.lylvs`` file.
    runset_variables : Optional[dict[str, str]]
        The runset's names for the file variables ``'gdsfile'``, ``'schematicfile'`` and
        ``'resultsfile'``, where they differ. A leading ``$`` is ignored.
    verbose : bool
        Whether to log the start and the verdict of each run. Default is ``True``.

    Example
    -------
    >>> from tidy3d.plugins.klayout.lvs import LVSRunner
    >>> runner = LVSRunner(lvs_runset="photonic.lvs")  # doctest: +SKIP
    >>> results = runner.run(gdsfile="chip.gds", schematicfile="chip.cir")  # doctest: +SKIP
    >>> print(results)  # doctest: +SKIP

    A runset that reads the file paths from ``$input``, ``$schematic`` and ``$report``:

    >>> runner = LVSRunner(
    ...     lvs_runset="pdk.lvs",
    ...     runset_variables={"gdsfile": "input", "schematicfile": "schematic", "resultsfile": "report"},
    ... )  # doctest: +SKIP
    """

    lvs_runset: FilePath = Field(
        title="LVS runset",
        description="Path to the KLayout LVS runset, a '.lvs' or '.lylvs' file.",
    )
    runset_variables: dict[str, str] = Field(
        default_factory=dict,
        title="Runset variables",
        description="The runset's names for the file variables 'gdsfile', 'schematicfile' and "
        "'resultsfile', where they differ, such as ``{'gdsfile': 'input'}``.",
    )
    verbose: bool = Field(
        default=DEFAULT_VERBOSE,
        title="Verbose",
        description="Whether to log the start and the verdict of each run.",
    )

    @field_validator("lvs_runset")
    @classmethod
    def _validate_lvs_runset_field(cls, value: Path) -> Path:
        """Check the runset's suffix and make its path absolute."""
        return _validate_lvs_runset(value)

    @field_validator("runset_variables", mode="before")
    @classmethod
    def _validate_runset_variables_field(cls, value: Any) -> dict[str, str]:
        """Check the runset variable names."""
        return _validate_runset_variables(value)

    def run(
        self,
        gdsfile: str | Path,
        schematicfile: str | Path,
        resultsfile: str | Path = DEFAULT_RESULTSFILE,
        lvs_args: dict[str, Any] | None = None,
        load_shapes: bool = True,
    ) -> LVSResults:
        """Runs KLayout LVS on a layout and loads the results.

        Parameters
        ----------
        gdsfile : Union[str, Path]
            Path to the layout, a ``.gds`` or ``.oas`` file.
        schematicfile : Union[str, Path]
            Path to the schematic netlist, which the runset reads.
        resultsfile : Union[str, Path]
            Path to write the results file to. Defaults to ``"lvs_results.lvsdb"``. A path ending
            in ``.gz`` gets a gzip-compressed file. An existing file is replaced only once KLayout
            has written the new one.
        lvs_args : Optional[dict[str, Any]]
            Other definitions passed to KLayout as ``-rd key=value``. They can't reuse the name of
            a file variable.
        load_shapes : bool
            Whether to load the shapes and labels of nets and device abstracts.

        Returns
        -------
        :class:`.LVSResults`
            The results, with KLayout's output and exit code.

        Raises
        ------
        pydantic.ValidationError
            If an input file is missing or has the wrong suffix, the results file's folder is
            missing, the results file is one of the input files, or ``lvs_args`` reuses a file
            variable.
        RuntimeError
            If KLayout isn't found or writes no results file.
        :class:`.FileError`
            If the results file can't be parsed.

        Notes
        -----
        The run logs a warning if the runset never mentions one of the file variables, which
        suggests that ``runset_variables`` is missing or names the wrong variable. It also warns if
        KLayout exits with a non-zero code but writes a results file, as some runsets do when the
        netlists don't match.

        Example
        -------
        >>> from tidy3d.plugins.klayout.lvs import LVSRunner
        >>> runner = LVSRunner(lvs_runset="photonic.lvs")  # doctest: +SKIP
        >>> results = runner.run(
        ...     gdsfile="chip.gds", schematicfile="chip.cir", resultsfile="chip.lvsdb"
        ... )  # doctest: +SKIP
        >>> print(results.verdict)  # doctest: +SKIP
        """
        config = LVSConfig(
            gdsfile=gdsfile,
            lvs_runset=self.lvs_runset,
            schematicfile=schematicfile,
            resultsfile=resultsfile,
            runset_variables=self.runset_variables,
            lvs_args=lvs_args,
            verbose=self.verbose,
        )
        return run_lvs_on_gds(config, load_shapes=load_shapes)


def run_lvs_on_gds(config: LVSConfig, load_shapes: bool = True) -> LVSResults:
    """Runs KLayout LVS as configured and loads the results. It warns as :meth:`.LVSRunner.run`
    does.

    Parameters
    ----------
    config : :class:`.LVSConfig`
        The configuration of the run.
    load_shapes : bool
        Whether to load the shapes and labels of nets and device abstracts.

    Returns
    -------
    :class:`.LVSResults`
        The results, with KLayout's output and exit code.

    Raises
    ------
    RuntimeError
        If KLayout isn't found or writes no results file.
    :class:`.FileError`
        If the results file can't be parsed.

    Example
    -------
    >>> from tidy3d.plugins.klayout.lvs import LVSConfig, run_lvs_on_gds
    >>> config = LVSConfig(
    ...     gdsfile="chip.gds", lvs_runset="photonic.lvs", schematicfile="chip.cir"
    ... )  # doctest: +SKIP
    >>> results = run_lvs_on_gds(config)  # doctest: +SKIP
    """
    klayout_cmd = check_installation(raise_error=True)
    names = _runset_variable_names(config.runset_variables)
    _warn_unmentioned_variables(config.lvs_runset, names)

    resultsfile = config.resultsfile
    # KLayout writes here first, so a failed run leaves an existing results file untouched. The
    # name ends with the results file's name, so a '.gz' file is still compressed.
    temporary = resultsfile.with_name(f".tmp-{secrets.token_hex(8)}-{resultsfile.name}")
    cmd = [
        klayout_cmd,
        "-b",
        "-r",
        str(config.lvs_runset),
        "-rd",
        f"{names['gdsfile']}={config.gdsfile}",
        "-rd",
        f"{names['schematicfile']}={config.schematicfile}",
        "-rd",
        f"{names['resultsfile']}={temporary}",
    ]
    for key, value in config.lvs_args.items():
        cmd.extend(["-rd", f"{key}={value}"])

    if config.verbose:
        # Paths can contain square brackets, which Rich would read as markup.
        get_logging_console().log(
            f"Running KLayout LVS on layout '{config.gdsfile}' and schematic "
            f"'{config.schematicfile}' with runset '{config.lvs_runset}', saving results to "
            f"'{resultsfile}'...",
            markup=False,
        )
    try:
        process = run(cmd, stdout=PIPE, stderr=STDOUT)
        output = process.stdout.decode(errors="replace")
        if not temporary.exists():
            raise RuntimeError(
                _no_results_message(process.returncode, output, names["resultsfile"])
            )
        os.replace(temporary, resultsfile)
    finally:
        temporary.unlink(missing_ok=True)

    if process.returncode != 0:
        log.warning(
            f"KLayout LVS exited with code {process.returncode} but wrote a results file. Some "
            "runsets exit with a non-zero code when the netlists don't match. KLayout's output "
            "is in the results' 'log'."
        )
    results = load_results(
        resultsfile,
        load_shapes=load_shapes,
        output=output,
        exit_code=process.returncode,
    )
    if config.verbose:
        get_logging_console().log(f"KLayout LVS finished with verdict '{results.verdict}'.")
    return results


def _warn_unmentioned_variables(lvs_runset: Path, names: Mapping[str, str]) -> None:
    """Warn if the runset never mentions a file variable, which suggests a wrong name.

    It warns rather than raises, because a runset can read the variable in a file it pulls in
    with ``# %include``, which this check doesn't read.
    """
    text = lvs_runset.read_text(encoding="utf-8", errors="replace")
    missing = [
        f"${name}" for name in names.values() if not re.search(rf"\${re.escape(name)}(?!\w)", text)
    ]
    if missing:
        log.warning(
            f"LVS runset '{lvs_runset}' never mentions {', '.join(missing)}. The runner passes the "
            f"layout, schematic and results paths in ${names['gdsfile']}, "
            f"${names['schematicfile']} and ${names['resultsfile']}; if the runset uses other "
            "variable names, set them with 'runset_variables'."
        )


def _no_results_message(exit_code: int, output: str, results_variable: str) -> str:
    """Explain a run that wrote no results file, with the end of KLayout's output."""
    lines = output.splitlines()
    if len(lines) > _OUTPUT_TAIL_LINES:
        heading = f"KLayout's output, last {_OUTPUT_TAIL_LINES} of {len(lines)} lines:"
    else:
        heading = "KLayout's output:"
    shown = "\n".join(lines[-_OUTPUT_TAIL_LINES:]) if lines else "(none)"
    return (
        f"KLayout LVS exited with code {exit_code} and wrote no results file. KLayout writes it "
        f"only when the runset calls 'report_lvs' with ${results_variable} and runs 'compare' "
        "without an error. If the runset uses another name for the results variable, set it with "
        f"'runset_variables'.\n{heading}\n{shown}"
    )
