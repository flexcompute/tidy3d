"""Validate FAQ sources and generate the Sphinx publication tree."""

from __future__ import annotations

import argparse
import ast
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlsplit

import yaml

FRONT_MATTER_RE = re.compile(r"\A---[ \t]*\n(.*?)\n---[ \t]*(?:\n|\Z)", re.DOTALL)
JEKYLL_HIGHLIGHT_RE = re.compile(
    r"{%\s*highlight\s+([A-Za-z0-9_+-]+)\s*%}(.*?){%\s*endhighlight\s*%}",
    re.DOTALL,
)
MARKDOWN_FENCE_RE = re.compile(
    r"(?P<opening>^[ \t]{0,3}(?P<fence>`{3,}|~{3,})[ \t]*"
    r"(?P<language>[A-Za-z0-9_+-]+)[^\n]*\n)"
    r"(?P<code>.*?)"
    r"(?P<closing>^[ \t]{0,3}(?P=fence)[ \t]*(?:\n|\Z))",
    re.DOTALL | re.MULTILINE,
)
MARKDOWN_FENCED_BLOCK_RE = re.compile(
    r"^[ \t]{0,3}(?P<fence>`{3,}|~{3,})[^\n]*\n"
    r".*?"
    r"^[ \t]{0,3}(?P=fence)[ \t]*(?:\n|\Z)",
    re.DOTALL | re.MULTILINE,
)
JEKYLL_ROLE_RE = re.compile(
    r"`(?P<text>[^`\n]+)`{:\s*[^}\n]*?\brole=(?P<quote>[\"'])"
    r"(?P<role>[A-Za-z][A-Za-z0-9:._-]*)(?P=quote)[^}\n]*}"
)
JEKYLL_ATTRIBUTE_RE = re.compile(r"{:\s*.*?}", re.DOTALL)
JEKYLL_INCLUDE_RE = re.compile(r"{%\s*include\s+.*?%}")
MARKDOWN_IMAGE_RE = re.compile(r"!\[[^\]]*]\(\s*<?([^)\s>]+)")
PLACEHOLDER_LINK_RE = re.compile(r"]\(\s*(?:none|null)?\s*\)", re.IGNORECASE)
ROOT_RELATIVE_LINK_RE = re.compile(
    r"""(?:(?<!!)\[[^\]]*]\(\s*|href=["'])/(?!/)""",
    re.IGNORECASE,
)
NON_LATEST_TIDY3D_DOCS_RE = re.compile(
    r"https://docs\.flexcompute\.com/projects/tidy3d/en/(?:stable|v\d+(?:\.\d+)+)/",
    re.IGNORECASE,
)
INTERNAL_API_LINK_RE = re.compile(
    r"https://docs\.flexcompute\.com/projects/tidy3d/en/latest/"
    r"(?P<path>api/[A-Za-z0-9_./-]+\.html)",
    re.IGNORECASE,
)
RST_CURRENT_MODULE_RE = re.compile(r"^\s*\.\.\s+currentmodule::\s*(?P<module>\S+)\s*$")
RST_AUTOSUMMARY_RE = re.compile(r"^(?P<indent>\s*)\.\.\s+autosummary::\s*$")
RST_TOCTREE_OPTION_RE = re.compile(r"^:toctree:\s*(?P<path>\S+)\s*$")
RST_EXPLICIT_TARGET_RE = re.compile(r".*<(?P<target>[^>]+)>\s*$")
HTML_IMAGE_RE = re.compile(r"""<img\b[^>]*\bsrc=["']([^"']+)["']""", re.IGNORECASE)
HTML_IMAGE_TAG_RE = re.compile(
    r"""<img\b(?=[^>]*\bsrc=(?P<quote>["'])(?P<target>[^"']+)(?P=quote))[^>]*>""",
    re.IGNORECASE,
)
BLOCK_HTML_TAG_RE = re.compile(
    r"(?:</?(?:p|div)(?:\s[^>]*)?>[ \t]*)+",
    re.IGNORECASE,
)
INLINE_HTML_TAG_RE = re.compile(r"</?span(?:\s[^>]*)?>", re.IGNORECASE)
INITIAL_H1_RE = re.compile(r"\A[ \t]*#\s+[^\n]+[ \t]*(?:\n+|\Z)")
TRAILING_TRANSITION_RE = re.compile(r"(?:\A|\n)[ \t]*---[ \t]*(?:\n[ \t]*)?\Z")
RST_INCLUDE_RE = re.compile(
    r"^\s*\.\.\s+(?:include|literalinclude)::\s+(?P<target>\S+)\s*$",
    re.MULTILINE,
)
RST_IMAGE_RE = re.compile(
    r"^\s*\.\.\s+(?:figure|image)::\s+(?P<target>\S+)\s*$",
    re.MULTILINE,
)
CATEGORY_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._()-]*\Z")
FAQ_FILENAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\.md\Z")
GENERATED_MARKER = ".tidy3d-faq-generated"
LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1\n"
ALIAS_MANIFEST = "faq_aliases.json"
FAQ_LFS_INCLUDE = "**/docs/faq/_faqs/img/**"
FAQ_FRONT_MATTER_FIELDS = frozenset({"title", "date", "enabled", "category", "version"})
COMMON_PYTHON_IMPORTS = {
    "anp": "import autograd.numpy as anp",
    "gdstk": "import gdstk",
    "go": "import plotly.graph_objects as go",
    "np": "import numpy as np",
    "pd": "import pandas as pd",
    "plt": "import matplotlib.pyplot as plt",
    "td": "import tidy3d as td",
    "tidy3d": "import tidy3d",
    "trimesh": "import trimesh",
    "web": "from tidy3d import web",
    "xr": "import xarray as xr",
}
TIDY3D_BARE_SYMBOLS = {
    "AnisotropicMedium",
    "DiffractionMonitor",
    "DistanceUnstructuredGrid",
    "FieldMonitor",
    "FieldProjectionAngleMonitor",
    "FieldProjectionCartesianMonitor",
    "FieldProjectionKSpaceMonitor",
    "FieldTimeMonitor",
    "FluxMonitor",
    "FluxTimeMonitor",
    "FullyAnisotropicMedium",
    "Medium",
    "ModeMonitor",
    "ModeSolverMonitor",
    "ModeSpec",
    "PermittivityMonitor",
    "SimulationData",
    "UniformUnstructuredGrid",
    "inf",
}


class FaqValidationError(ValueError):
    """Raised when FAQ authoring sources are inconsistent."""


@dataclass(frozen=True)
class Faq:
    source_path: Path
    manifest_path: str
    title: str
    date: str
    category: str
    version: str | None
    body: str

    @property
    def output_name(self) -> str:
        return self.source_path.name


@dataclass(frozen=True)
class Category:
    identifier: str
    title: str
    faqs: tuple[Faq, ...]


@dataclass(frozen=True)
class FaqAlias:
    output_name: str
    target: Faq


def _single_line(value: str, field: str, manifest_path: str) -> str:
    normalized = value.strip()
    if not normalized or any(character in normalized for character in "\r\n"):
        raise FaqValidationError(
            f"{manifest_path}: front matter field {field!r} must be a non-empty single line"
        )
    return normalized


def _format_date(value: Any, manifest_path: str) -> str:
    if isinstance(value, dt.datetime):
        return value.isoformat(sep=" ")
    if isinstance(value, dt.date):
        return value.isoformat()
    if isinstance(value, str):
        return _single_line(value, "date", manifest_path)
    raise FaqValidationError(
        f"{manifest_path}: front matter field 'date' must be a date or non-empty string"
    )


def _format_version(value: Any, manifest_path: str) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise FaqValidationError(f"{manifest_path}: front matter field 'version' must be a scalar")
    return _single_line(str(value), "version", manifest_path)


def _declared_internal_api_routes(docs_root: Path) -> frozenset[str]:
    """Derive published API routes from committed RST and autosummary declarations."""

    resolved_docs_root = docs_root.resolve()
    api_root = resolved_docs_root / "api"
    routes: set[str] = set()
    for source_path in sorted(api_root.rglob("*.rst")):
        relative_source = source_path.relative_to(api_root)
        if "_autosummary" in relative_source.parts:
            continue
        routes.add(source_path.with_suffix(".html").relative_to(resolved_docs_root).as_posix())

        current_module: str | None = None
        lines = source_path.read_text(encoding="utf-8").splitlines()
        line_index = 0
        while line_index < len(lines):
            current_module_match = RST_CURRENT_MODULE_RE.match(lines[line_index])
            if current_module_match is not None:
                current_module = current_module_match.group("module")

            autosummary_match = RST_AUTOSUMMARY_RE.match(lines[line_index])
            if autosummary_match is None:
                line_index += 1
                continue

            directive_indent = len(autosummary_match.group("indent"))
            toctree_path: str | None = None
            entries: list[str] = []
            line_index += 1
            while line_index < len(lines):
                line = lines[line_index]
                if not line.strip():
                    line_index += 1
                    continue
                indentation = len(line) - len(line.lstrip())
                if indentation <= directive_indent:
                    break

                stripped = line.strip()
                toctree_match = RST_TOCTREE_OPTION_RE.match(stripped)
                if toctree_match is not None:
                    toctree_path = toctree_match.group("path")
                elif not stripped.startswith((":", "..")):
                    entry = stripped.lstrip("~")
                    explicit_target_match = RST_EXPLICIT_TARGET_RE.match(entry)
                    if explicit_target_match is not None:
                        entry = explicit_target_match.group("target")
                    if current_module is not None and not entry.startswith("tidy3d."):
                        entry = f"{current_module}.{entry}"
                    entries.append(entry)
                line_index += 1

            if toctree_path is not None:
                output_root = (source_path.parent / toctree_path).resolve()
                for entry in entries:
                    route = (output_root / f"{entry}.html").relative_to(resolved_docs_root)
                    routes.add(route.as_posix())

    return frozenset(routes)


def _parse_faq(
    source_path: Path,
    manifest_path: str,
    internal_api_routes: frozenset[str] | None = None,
) -> tuple[dict[str, Any], str]:
    content = source_path.read_text(encoding="utf-8")
    match = FRONT_MATTER_RE.match(content)
    if match is None:
        raise FaqValidationError(f"{manifest_path}: missing YAML front matter")

    metadata = yaml.safe_load(match.group(1))
    if not isinstance(metadata, dict):
        raise FaqValidationError(f"{manifest_path}: front matter must be a mapping")
    unsupported_fields = sorted(
        (str(field) for field in metadata if field not in FAQ_FRONT_MATTER_FIELDS),
        key=str.casefold,
    )
    if unsupported_fields:
        raise FaqValidationError(
            f"{manifest_path}: unsupported front matter fields: {', '.join(unsupported_fields)}"
        )
    body = content[match.end() :]
    if PLACEHOLDER_LINK_RE.search(body):
        raise FaqValidationError(
            f"{manifest_path}: links may not have empty or placeholder targets"
        )
    if ROOT_RELATIVE_LINK_RE.search(body):
        raise FaqValidationError(f"{manifest_path}: links may not use site-root-relative targets")
    if NON_LATEST_TIDY3D_DOCS_RE.search(body):
        raise FaqValidationError(
            f"{manifest_path}: internal Tidy3D documentation links must use the latest route"
        )
    if TRAILING_TRANSITION_RE.search(body):
        raise FaqValidationError(
            f"{manifest_path}: FAQ bodies may not end with a Markdown transition"
        )
    if internal_api_routes is not None:
        for link_match in INTERNAL_API_LINK_RE.finditer(body):
            if link_match.group("path") not in internal_api_routes:
                raise FaqValidationError(
                    f"{manifest_path}: internal API link is not declared by the Sphinx sources: "
                    f"{link_match.group(0)}"
                )
    code_blocks = [
        (code_match.start(), code_match.group(1), code_match.group(2))
        for code_match in JEKYLL_HIGHLIGHT_RE.finditer(body)
    ]
    code_blocks.extend(
        (
            code_match.start(),
            code_match.group("language"),
            code_match.group("code"),
        )
        for code_match in MARKDOWN_FENCE_RE.finditer(body)
    )
    for index, (_, language, raw_code) in enumerate(sorted(code_blocks), start=1):
        code = raw_code.strip()
        code_lines = code.splitlines()
        if code_lines and code_lines[0].strip().casefold() == language.casefold():
            raise FaqValidationError(
                f"{manifest_path}: code block {index} repeats its {language!r} language marker"
            )
        if language.casefold() not in {"py", "python"}:
            continue
        try:
            tree = ast.parse(code, filename=manifest_path)
        except SyntaxError as exc:
            raise FaqValidationError(
                f"{manifest_path}: Python code block {index} is invalid: "
                f"{exc.msg} at line {exc.lineno}"
            ) from exc
        loaded = {
            node.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
        }
        missing_imports = sorted((loaded & TIDY3D_BARE_SYMBOLS) - _bound_python_names(tree))
        if missing_imports:
            raise FaqValidationError(
                f"{manifest_path}: Python code block {index} must import or qualify "
                f"Tidy3D symbols: {', '.join(missing_imports)}"
            )
    return metadata, body


def _bound_python_names(tree: ast.AST) -> set[str]:
    """Return names bound anywhere in a Python example."""

    bound = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    }
    bound.update(node.arg for node in ast.walk(tree) if isinstance(node, ast.arg))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            bound.update(alias.asname or alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            bound.update(alias.asname or alias.name for alias in node.names)
        elif isinstance(node, (ast.AsyncFunctionDef, ast.ClassDef, ast.FunctionDef)):
            bound.add(node.name)
        elif isinstance(node, ast.ExceptHandler) and node.name is not None:
            bound.add(node.name)
    return bound


def _add_missing_common_imports(language: str, raw_code: str) -> str:
    """Make conventional module aliases explicit in copyable Python examples."""

    code = raw_code.strip()
    if language.casefold() not in {"py", "python"}:
        return code
    tree = ast.parse(code)
    loaded = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }
    bound = _bound_python_names(tree)
    imports = [
        statement
        for name, statement in COMMON_PYTHON_IMPORTS.items()
        if name in loaded and name not in bound
    ]
    if not imports:
        return code
    future_imports = [
        node
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module == "__future__"
    ]
    if future_imports:
        last_future_line = future_imports[-1].end_lineno or future_imports[-1].lineno
        lines = code.splitlines()
        suffix = lines[last_future_line:]
        while suffix and not suffix[0].strip():
            suffix.pop(0)
        return "\n".join(
            [
                *lines[:last_future_line],
                *imports,
                *([""] if suffix else []),
                *suffix,
            ]
        )
    return "\n".join([*imports, "", code])


def _require_string(metadata: dict[str, Any], field: str, manifest_path: str) -> str:
    value = metadata.get(field)
    if not isinstance(value, str):
        raise FaqValidationError(
            f"{manifest_path}: front matter field {field!r} must be a non-empty string"
        )
    return _single_line(value, field, manifest_path)


def _validate_manifest_path(value: Any) -> str:
    if not isinstance(value, str):
        raise FaqValidationError("FAQ manifest paths must be strings")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or len(path.parts) != 2
        or path.parts[0] != "_faqs"
        or path.suffix != ".md"
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise FaqValidationError(
            f"{value!r}: FAQ manifest paths must have the form '_faqs/<slug>.md'"
        )
    if path.name != path.name.strip():
        raise FaqValidationError(f"{value!r}: FAQ filenames may not have surrounding whitespace")
    if not FAQ_FILENAME_RE.fullmatch(path.name):
        raise FaqValidationError(
            f"{value!r}: FAQ filenames may contain only letters, digits, dots, underscores, and hyphens"
        )
    return path.as_posix()


def _validate_category_id(value: Any) -> str:
    identifier = _require_string({"id": value}, "id", "faq_categories.json")
    if not CATEGORY_ID_RE.fullmatch(identifier) or identifier.casefold() == "index":
        raise FaqValidationError(
            f"{identifier!r}: FAQ category ids must be safe, non-reserved filenames"
        )
    return identifier


def _validate_output_filename(value: Any, context: str) -> str:
    if not isinstance(value, str) or not FAQ_FILENAME_RE.fullmatch(value):
        raise FaqValidationError(
            f"{context}: FAQ filenames may contain only letters, digits, dots, "
            "underscores, and hyphens"
        )
    return value


def _local_asset_path(faq_path: Path, target: str, manifest_path: str) -> Path | None:
    parsed = urlsplit(target)
    if parsed.scheme or parsed.netloc or target.startswith(("#", "data:")):
        return None

    if parsed.path.startswith("/"):
        raise FaqValidationError(
            f"{manifest_path}: local images must use a relative './img/' reference"
        )
    clean_target = parsed.path.removeprefix("/").removeprefix("./")
    relative_path = PurePosixPath(clean_target)
    if (
        not relative_path.parts
        or relative_path.parts[0] != "img"
        or any(part in {"", ".", ".."} for part in relative_path.parts)
    ):
        raise FaqValidationError(
            f"{manifest_path}: local images must be stored under and referenced from './img/'"
        )
    return faq_path.parent.joinpath(*relative_path.parts)


def _validate_local_assets(faq_path: Path, body: str, manifest_path: str) -> set[Path]:
    assets: set[Path] = set()
    targets = MARKDOWN_IMAGE_RE.findall(body) + HTML_IMAGE_RE.findall(body)
    for target in targets:
        asset = _local_asset_path(faq_path, target, manifest_path)
        if asset is None:
            continue
        _validate_asset_file(asset, f"{manifest_path}: referenced image {target!r}")
        assets.add(asset)
    return assets


def _validate_asset_file(asset: Path, context: str) -> None:
    if asset.is_symlink():
        raise FaqValidationError(f"{context} may not be a symlink")
    if not asset.is_file():
        raise FaqValidationError(f"{context} does not exist or is not a regular file")
    if _is_lfs_pointer(asset):
        raise FaqValidationError(f"{context} is an unhydrated Git LFS pointer")


def _is_lfs_pointer(path: Path) -> bool:
    with path.open("rb") as asset_file:
        return asset_file.read(len(LFS_POINTER_PREFIX)) == LFS_POINTER_PREFIX


def _unhydrated_lfs_assets(source_root: Path) -> tuple[Path, ...]:
    image_root = source_root / "_faqs" / "img"
    if image_root.is_symlink():
        raise FaqValidationError("_faqs/img may not be a symlink")
    if not image_root.exists():
        return ()
    if not image_root.is_dir():
        raise FaqValidationError("_faqs/img must be a regular directory")
    return tuple(
        path
        for path in sorted(image_root.rglob("*"))
        if path.is_file() and not path.is_symlink() and _is_lfs_pointer(path)
    )


def _hydrate_lfs_assets_for_sphinx(source_root: Path) -> None:
    """Hydrate FAQ image pointers when Sphinx is invoked without the docs CLI."""

    if not _unhydrated_lfs_assets(source_root):
        return
    command = ["git", "lfs", "pull", f"--include={FAQ_LFS_INCLUDE}"]
    package_root = source_root.parent.parent
    try:
        subprocess.run(command, cwd=package_root, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise FaqValidationError(
            "Unable to hydrate FAQ images for the Sphinx build. "
            f"Run `{' '.join(command)}` from a Git checkout with Git LFS access."
        ) from exc
    remaining = _unhydrated_lfs_assets(source_root)
    if remaining:
        relative_paths = ", ".join(path.relative_to(source_root).as_posix() for path in remaining)
        raise FaqValidationError(
            "Git LFS completed without hydrating FAQ images: "
            f"{relative_paths}. Verify repository LFS access."
        )


def _validate_image_tree(source_root: Path) -> set[Path]:
    image_root = source_root / "_faqs" / "img"
    if image_root.is_symlink():
        raise FaqValidationError("_faqs/img may not be a symlink")
    if not image_root.exists():
        return set()
    if not image_root.is_dir():
        raise FaqValidationError("_faqs/img must be a regular directory")
    assets: set[Path] = set()
    for asset in sorted(image_root.rglob("*")):
        relative_path = asset.relative_to(source_root).as_posix()
        if asset.is_symlink():
            raise FaqValidationError(f"{relative_path} may not be a symlink")
        if asset.is_dir():
            continue
        _validate_asset_file(asset, relative_path)
        assets.add(asset)
    return assets


def _load_and_validate_aliases(
    source_root: Path, categories: tuple[Category, ...]
) -> tuple[FaqAlias, ...]:
    alias_file = source_root / ALIAS_MANIFEST
    if alias_file.is_symlink():
        raise FaqValidationError(f"{ALIAS_MANIFEST} may not be a symlink")
    if not alias_file.is_file():
        raise FaqValidationError(f"missing FAQ alias manifest: {alias_file}")
    raw_aliases = json.loads(alias_file.read_text(encoding="utf-8"))
    if not isinstance(raw_aliases, dict):
        raise FaqValidationError(f"{ALIAS_MANIFEST} must contain a mapping")

    faqs_by_path = {faq.manifest_path: faq for category in categories for faq in category.faqs}
    canonical_output_names = {faq.output_name for faq in faqs_by_path.values()}
    seen_aliases: set[str] = set()
    aliases: list[FaqAlias] = []
    for raw_output_name, raw_target in raw_aliases.items():
        output_name = _validate_output_filename(raw_output_name, ALIAS_MANIFEST)
        target_path = _validate_manifest_path(raw_target)
        target = faqs_by_path.get(target_path)
        if target is None:
            raise FaqValidationError(
                f"{ALIAS_MANIFEST}: alias {output_name!r} targets an FAQ that is not published"
            )
        if output_name in canonical_output_names or output_name in seen_aliases:
            raise FaqValidationError(
                f"{ALIAS_MANIFEST}: alias output {output_name!r} is duplicated"
            )
        if output_name.casefold() != target.output_name.casefold():
            raise FaqValidationError(
                f"{ALIAS_MANIFEST}: alias {output_name!r} may differ from its target only by case"
            )
        seen_aliases.add(output_name)
        aliases.append(FaqAlias(output_name=output_name, target=target))
    return tuple(sorted(aliases, key=lambda alias: alias.output_name))


def load_and_validate(source_root: Path) -> tuple[Category, ...]:
    """Load and validate all FAQ authoring sources."""

    source_root = source_root.resolve()
    docs_root: Path | None = source_root.parent
    if not (docs_root / "api").is_dir():
        docs_root = None
    internal_api_routes = (
        _declared_internal_api_routes(docs_root) if docs_root is not None else None
    )
    manifest_file = source_root / "faq_categories.json"
    raw_faq_root = source_root / "_faqs"
    if manifest_file.is_symlink():
        raise FaqValidationError("faq_categories.json may not be a symlink")
    if not manifest_file.is_file():
        raise FaqValidationError(f"missing FAQ manifest: {manifest_file}")
    if raw_faq_root.is_symlink():
        raise FaqValidationError("_faqs may not be a symlink")
    if not raw_faq_root.is_dir():
        raise FaqValidationError(f"missing FAQ source directory: {raw_faq_root}")
    available_assets = _validate_image_tree(source_root)

    raw_categories = json.loads(manifest_file.read_text(encoding="utf-8"))
    if not isinstance(raw_categories, list):
        raise FaqValidationError("faq_categories.json must contain a list")

    seen_category_ids: set[str] = set()
    seen_category_titles: set[str] = set()
    seen_manifest_paths: set[str] = set()
    seen_output_names: set[str] = set()
    categories: list[Category] = []

    for raw_category in raw_categories:
        if not isinstance(raw_category, dict):
            raise FaqValidationError("each FAQ category must be a mapping")
        identifier = _validate_category_id(raw_category.get("id"))
        title = _require_string(raw_category, "category", "faq_categories.json")
        identifier_key = identifier.casefold()
        if identifier_key in seen_category_ids:
            raise FaqValidationError(f"duplicate FAQ category id: {identifier!r}")
        if title in seen_category_titles:
            raise FaqValidationError(f"duplicate FAQ category title: {title!r}")
        seen_category_ids.add(identifier_key)
        seen_category_titles.add(title)

        raw_paths = raw_category.get("faqs")
        if not isinstance(raw_paths, list):
            raise FaqValidationError(f"{identifier}: 'faqs' must be a list")

        faqs: list[Faq] = []
        for raw_path in raw_paths:
            manifest_path = _validate_manifest_path(raw_path)
            if manifest_path in seen_manifest_paths:
                raise FaqValidationError(f"duplicate FAQ manifest entry: {manifest_path}")
            seen_manifest_paths.add(manifest_path)

            source_path = source_root.joinpath(*PurePosixPath(manifest_path).parts)
            if source_path.is_symlink():
                raise FaqValidationError(f"{manifest_path}: FAQ sources may not be symlinks")
            if not source_path.is_file():
                raise FaqValidationError(f"{manifest_path}: source file does not exist")
            metadata, body = _parse_faq(source_path, manifest_path, internal_api_routes)
            if metadata.get("enabled") is not True:
                raise FaqValidationError(f"{manifest_path}: listed FAQs must have enabled: true")
            faq_category = _require_string(metadata, "category", manifest_path)
            if faq_category != title:
                raise FaqValidationError(
                    f"{manifest_path}: category {faq_category!r} does not match {title!r}"
                )
            output_key = source_path.name.casefold()
            if output_key in seen_output_names:
                raise FaqValidationError(
                    f"{manifest_path}: output filename collides case-insensitively"
                )
            seen_output_names.add(output_key)
            _validate_local_assets(source_path, body, manifest_path)

            faqs.append(
                Faq(
                    source_path=source_path,
                    manifest_path=manifest_path,
                    title=_require_string(metadata, "title", manifest_path),
                    date=_format_date(metadata.get("date"), manifest_path),
                    category=faq_category,
                    version=_format_version(metadata.get("version"), manifest_path),
                    body=body,
                )
            )
        categories.append(Category(identifier=identifier, title=title, faqs=tuple(faqs)))

    enabled_sources: set[str] = set()
    referenced_assets: set[Path] = set()
    for source_path in sorted(raw_faq_root.rglob("*.md")):
        relative_path = source_path.relative_to(source_root).as_posix()
        _validate_manifest_path(relative_path)
        if source_path.is_symlink():
            raise FaqValidationError(f"{relative_path}: FAQ sources may not be symlinks")
        metadata, body = _parse_faq(source_path, relative_path, internal_api_routes)
        _require_string(metadata, "title", relative_path)
        _format_date(metadata.get("date"), relative_path)
        _format_version(metadata.get("version"), relative_path)
        faq_category = _require_string(metadata, "category", relative_path)
        if faq_category not in seen_category_titles:
            raise FaqValidationError(
                f"{relative_path}: category {faq_category!r} is not declared in the manifest"
            )
        referenced_assets.update(_validate_local_assets(source_path, body, relative_path))
        enabled = metadata.get("enabled")
        if not isinstance(enabled, bool):
            raise FaqValidationError(
                f"{relative_path}: front matter field 'enabled' must be a boolean"
            )
        if enabled:
            enabled_sources.add(relative_path)

    unlisted = sorted(enabled_sources - seen_manifest_paths)
    disabled_but_listed = sorted(seen_manifest_paths - enabled_sources)
    if unlisted:
        raise FaqValidationError("enabled FAQs missing from manifest: " + ", ".join(unlisted))
    if disabled_but_listed:
        raise FaqValidationError(
            "disabled FAQs present in manifest: " + ", ".join(disabled_but_listed)
        )
    unused_assets = sorted(
        asset.relative_to(source_root).as_posix() for asset in available_assets - referenced_assets
    )
    if unused_assets:
        raise FaqValidationError("unused FAQ media: " + ", ".join(unused_assets))
    validated_categories = tuple(categories)
    _load_and_validate_aliases(source_root, validated_categories)
    return validated_categories


def _render_faq(faq: Faq) -> str:
    def render_jekyll_role(match: re.Match[str]) -> str:
        role = {"class": "py:class"}.get(match.group("role"), match.group("role"))
        return f"{{{role}}}`{match.group('text')}`"

    def render_markdown_code_block(match: re.Match[str]) -> str:
        code = _add_missing_common_imports(match.group("language"), match.group("code"))
        return f"{match.group('opening')}{code}\n{match.group('closing')}"

    def render_code_block(match: re.Match[str]) -> str:
        language = match.group(1)
        code = _add_missing_common_imports(language, match.group(2))
        return f"\n\n```{language}\n{code}\n```\n\n"

    def render_html_image(match: re.Match[str]) -> str:
        target = match.group("target")
        if _local_asset_path(faq.source_path, target, faq.manifest_path) is None:
            return match.group(0)
        image_name = PurePosixPath(urlsplit(target).path).stem
        alt_text = image_name.replace("-", " ").replace("_", " ")
        return f"![{alt_text}]({target})"

    def render_prose(text: str) -> str:
        text = JEKYLL_INCLUDE_RE.sub("", text)
        text = JEKYLL_ROLE_RE.sub(render_jekyll_role, text)
        text = JEKYLL_ATTRIBUTE_RE.sub("", text)
        text = text.replace("&nbsp;", " ").replace("\N{NO-BREAK SPACE}", " ")
        text = HTML_IMAGE_TAG_RE.sub(render_html_image, text)
        text = BLOCK_HTML_TAG_RE.sub("\n\n", text)
        return INLINE_HTML_TAG_RE.sub("", text)

    def transform_outside_fences(text: str, transform: Any) -> str:
        rendered_segments: list[str] = []
        previous_end = 0
        for match in MARKDOWN_FENCED_BLOCK_RE.finditer(text):
            rendered_segments.append(transform(text[previous_end : match.start()]))
            rendered_segments.append(match.group(0))
            previous_end = match.end()
        rendered_segments.append(transform(text[previous_end:]))
        return "".join(rendered_segments)

    body = MARKDOWN_FENCE_RE.sub(render_markdown_code_block, faq.body)
    body = transform_outside_fences(
        body,
        lambda prose: JEKYLL_HIGHLIGHT_RE.sub(render_code_block, prose),
    )
    body = transform_outside_fences(body, render_prose).strip()
    body = INITIAL_H1_RE.sub("", body, count=1).strip()

    headers = ["Date", "Category"]
    values = [faq.date, faq.category]
    if faq.version is not None:
        headers.append("Version")
        values.append(faq.version)
    header_row = "| " + " | ".join(headers) + " |"
    separator_row = "|" + "|".join("-" * (len(header) + 2) for header in headers) + "|"
    value_row = "| " + " | ".join(values) + " |"
    return f"# {faq.title}\n\n{header_row}\n{separator_row}\n{value_row}\n\n{body}\n"


def _render_alias(alias: FaqAlias) -> str:
    target_route = f"{alias.target.source_path.stem}.html"
    return (
        "---\n"
        "orphan: true\n"
        "---\n\n"
        "# FAQ page moved\n\n"
        "```{raw} html\n"
        f'<meta http-equiv="refresh" content="0; url={target_route}">\n'
        f'<p>This FAQ has moved to <a href="{target_route}">{target_route}</a>.</p>\n'
        "```\n"
    )


def _aliases_canonical_file(alias_path: Path, canonical_path: Path) -> bool:
    """Return whether a case-only alias resolves to its canonical file."""

    return alias_path.exists() and alias_path.samefile(canonical_path)


def _write_generated_tree(
    source_root: Path,
    categories: tuple[Category, ...],
    aliases: tuple[FaqAlias, ...],
    output_root: Path,
) -> None:
    faq_output = output_root / "faq"
    faq_output.mkdir(parents=True)
    image_root = source_root / "_faqs" / "img"
    if image_root.is_dir():
        shutil.copytree(image_root, faq_output / "img")

    manifest: list[dict[str, str]] = []
    for category in categories:
        category_lines = [
            category.title,
            "=" * len(category.title),
            "",
            ".. toctree::",
            "   :maxdepth: 2",
            "",
        ]
        for faq in category.faqs:
            (faq_output / faq.output_name).write_text(_render_faq(faq), encoding="utf-8")
            category_lines.append(f"   faq/{faq.output_name}")
            manifest.append(
                {
                    "category": category.title,
                    "route": f"faq/{faq.source_path.stem}.html",
                    "source": faq.manifest_path,
                    "title": faq.title,
                }
            )
        category_lines.append("")
        (output_root / f"{category.identifier}.rst").write_text(
            "\n".join(category_lines), encoding="utf-8"
        )

    alias_manifest: dict[str, str] = {}
    for alias in aliases:
        alias_path = faq_output / alias.output_name
        canonical_path = faq_output / alias.target.output_name
        alias_manifest[f"faq/{Path(alias.output_name).stem}.html"] = (
            f"faq/{alias.target.source_path.stem}.html"
        )
        if _aliases_canonical_file(alias_path, canonical_path):
            # Case-insensitive filesystems cannot represent both routes. Keep
            # the canonical page intact; Linux publication builds emit both.
            continue
        if alias_path.exists():
            raise FaqValidationError(f"alias output path already exists: {alias_path}")
        alias_path.write_text(
            _render_alias(alias),
            encoding="utf-8",
        )

    index_lines = [
        "FAQ |:mag_right:|",
        "=================",
        "",
        ".. toctree::",
        "   :maxdepth: 2",
        "",
        *(f"   {category.identifier}" for category in categories),
        "",
    ]
    (output_root / "index.rst").write_text("\n".join(index_lines), encoding="utf-8")
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_root / "aliases.json").write_text(
        json.dumps(alias_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_root / GENERATED_MARKER).write_text(
        "Generated by docs/generate_faq_docs.py. Do not edit.\n",
        encoding="utf-8",
    )


def _safe_output_root(source_root: Path, output_root: Path) -> Path:
    if output_root.is_symlink():
        raise ValueError("output directory may not be a symlink")

    resolved_output = output_root.resolve()
    default_output = source_root / "docs"
    if (
        resolved_output == source_root
        or resolved_output in source_root.parents
        or (source_root in resolved_output.parents and resolved_output != default_output)
    ):
        raise ValueError(
            "output directory must be the default generated directory or outside the FAQ source"
        )
    if resolved_output.exists():
        if not resolved_output.is_dir():
            raise ValueError("existing output path must be a directory")
        if not (resolved_output / GENERATED_MARKER).is_file():
            raise ValueError(
                "refusing to replace an output directory without the FAQ generator marker"
            )
    return resolved_output


def generate(source_root: Path, output_root: Path) -> tuple[Category, ...]:
    """Validate sources and atomically replace the generated publication tree."""

    source_root = source_root.resolve()
    output_root = _safe_output_root(source_root, output_root)

    categories = load_and_validate(source_root)
    aliases = _load_and_validate_aliases(source_root, categories)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(tempfile.mkdtemp(prefix=f".{output_root.name}-", dir=output_root.parent))
    previous_root: Path | None = None
    try:
        _write_generated_tree(source_root, categories, aliases, temporary_root)
        if output_root.exists():
            previous_root = Path(
                tempfile.mkdtemp(prefix=f".{output_root.name}-previous-", dir=output_root.parent)
            )
            previous_root.rmdir()
            output_root.replace(previous_root)
        try:
            temporary_root.replace(output_root)
        except BaseException:
            if previous_root is not None:
                previous_root.replace(output_root)
                previous_root = None
            raise
        if previous_root is not None:
            shutil.rmtree(previous_root)
            previous_root = None
    finally:
        if temporary_root.exists():
            shutil.rmtree(temporary_root)
    return categories


def _stage_source_entry(source: Path, destination: Path) -> None:
    """Copy one source entry into the isolated Sphinx source tree."""

    if source.is_symlink():
        raise FaqValidationError(f"Sphinx source entries may not be symlinks: {source}")
    if source.is_dir():
        shutil.copytree(source, destination)
    else:
        shutil.copy2(source, destination)


def _first_source_tree_symlink(source_root: Path) -> Path | None:
    """Return the first symlink without traversing linked directories."""

    if source_root.is_symlink():
        return source_root
    for root, directory_names, file_names in os.walk(source_root, followlinks=False):
        root_path = Path(root)
        directory_names.sort()
        for name in sorted([*directory_names, *file_names]):
            path = root_path / name
            if path.is_symlink():
                return path
    return None


def _external_sphinx_include_paths(docs_root: Path) -> tuple[Path, ...]:
    """Return package-relative includes and their local image dependencies."""

    package_root = docs_root.parent
    external_paths: set[Path] = set()
    for source in sorted(docs_root.rglob("*.rst")):
        content = source.read_text(encoding="utf-8")
        for match in RST_INCLUDE_RE.finditer(content):
            raw_target = match.group("target")
            if raw_target.startswith("/"):
                target = docs_root / raw_target.removeprefix("/")
            else:
                target = source.parent / raw_target
            resolved_target = target.resolve()
            if resolved_target == docs_root or docs_root in resolved_target.parents:
                continue
            if resolved_target == package_root or package_root not in resolved_target.parents:
                raise FaqValidationError(
                    f"{source.relative_to(docs_root)}: include target escapes the package root: "
                    f"{raw_target}"
                )
            if target.is_symlink() or not resolved_target.is_file():
                raise FaqValidationError(
                    f"{source.relative_to(docs_root)}: include target is not a regular file: "
                    f"{raw_target}"
                )
            external_paths.add(resolved_target.relative_to(package_root))

    for relative_path in tuple(external_paths):
        if relative_path.suffix.casefold() not in {".html", ".md", ".rst"}:
            continue
        included_source = package_root / relative_path
        content = included_source.read_text(encoding="utf-8")
        image_targets = (
            MARKDOWN_IMAGE_RE.findall(content)
            + HTML_IMAGE_RE.findall(content)
            + RST_IMAGE_RE.findall(content)
        )
        for raw_target in image_targets:
            parsed_target = urlsplit(raw_target)
            if (
                parsed_target.scheme
                or parsed_target.netloc
                or not parsed_target.path
                or raw_target.startswith(("#", "data:"))
            ):
                continue
            image = included_source.parent / parsed_target.path
            resolved_image = image.resolve()
            if resolved_image == package_root or package_root not in resolved_image.parents:
                raise FaqValidationError(
                    f"{relative_path}: image target escapes the package root: {raw_target}"
                )
            if image.is_symlink() or not resolved_image.is_file():
                raise FaqValidationError(
                    f"{relative_path}: image target is not a regular file: {raw_target}"
                )
            external_paths.add(resolved_image.relative_to(package_root))
    return tuple(sorted(external_paths))


def generate_for_sphinx(
    docs_root: Path,
    staging_package_root: Path,
) -> tuple[Category, ...]:
    """Stage a package-shaped Sphinx source tree with generated FAQs."""

    source_symlink = _first_source_tree_symlink(docs_root)
    if source_symlink is not None:
        relative_path = source_symlink.relative_to(docs_root).as_posix()
        raise FaqValidationError(f"Sphinx source tree may not contain symlinks: {relative_path}")
    resolved_docs_root = docs_root.resolve()
    source_package_root = resolved_docs_root.parent
    resolved_staging_package_root = staging_package_root.resolve()
    if (
        resolved_staging_package_root == source_package_root
        or source_package_root in resolved_staging_package_root.parents
    ):
        raise ValueError("the staged Sphinx package tree must be outside the source checkout")
    source_root = resolved_docs_root / "faq"
    if not source_root.is_dir():
        raise FaqValidationError(f"missing FAQ source directory: {source_root}")

    _hydrate_lfs_assets_for_sphinx(source_root)
    external_include_paths = _external_sphinx_include_paths(resolved_docs_root)
    resolved_staging_package_root.mkdir(parents=True)
    resolved_staging_docs_root = resolved_staging_package_root / resolved_docs_root.name
    try:
        for relative_path in external_include_paths:
            source = source_package_root / relative_path
            destination = resolved_staging_package_root / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            _stage_source_entry(source, destination)

        resolved_staging_docs_root.mkdir()
        for source in sorted(resolved_docs_root.iterdir()):
            if source.name == "faq":
                continue
            _stage_source_entry(source, resolved_staging_docs_root / source.name)

        staged_faq_root = resolved_staging_docs_root / "faq"
        staged_faq_root.mkdir()
        for source in sorted(source_root.iterdir()):
            if source.name == "docs":
                continue
            _stage_source_entry(source, staged_faq_root / source.name)
        return generate(source_root, staged_faq_root / "docs")
    except BaseException:
        shutil.rmtree(resolved_staging_package_root)
        raise


def _parser() -> argparse.ArgumentParser:
    default_source = Path(__file__).resolve().parent / "faq"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=default_source)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate authoring sources without writing generated documentation",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.check:
        categories = load_and_validate(args.source_root)
    else:
        output_root = args.output_dir or args.source_root / "docs"
        categories = generate(args.source_root, output_root)
    faq_count = sum(len(category.faqs) for category in categories)
    print(f"Validated {faq_count} FAQs in {len(categories)} categories.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
