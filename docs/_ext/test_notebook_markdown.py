"""Tests for the nbsphinx notebook Markdown compatibility transform."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

from notebook_markdown import normalize_markdown_for_nbsphinx, normalize_notebook_source


def test_normalizes_code_links_and_math_nested_in_strong_emphasis():
    source = "Use [`ModeSpec`](https://example.com/mode) with **global $y$** and **$N = 1$**."

    assert normalize_markdown_for_nbsphinx(source) == (
        "Use [ModeSpec](https://example.com/mode) with **global** $y$ and $N = 1$."
    )


def test_preserves_strong_text_around_multiple_math_spans():
    assert normalize_markdown_for_nbsphinx("**A $x$ B $y$ C**") == ("**A** $x$ **B** $y$ **C**")


def test_does_not_normalize_fenced_examples():
    source = (
        "```markdown\n"
        "[`ModeSpec`](https://example.com/mode) and **global $y$**\n"
        "```\n"
        "[`ModeSpec`](https://example.com/mode)\n"
    )

    assert normalize_markdown_for_nbsphinx(source) == (
        "```markdown\n"
        "[`ModeSpec`](https://example.com/mode) and **global $y$**\n"
        "```\n"
        "[ModeSpec](https://example.com/mode)\n"
    )


def test_does_not_normalize_inline_code_examples():
    source = (
        "Literal ``[`ModeSpec`](https://example.com/mode)`` and `**global $y$**`.\n"
        "Use [`ModeSpec`](https://example.com/mode) with **global $y$**.\n"
    )

    assert normalize_markdown_for_nbsphinx(source) == (
        "Literal ``[`ModeSpec`](https://example.com/mode)`` and `**global $y$**`.\n"
        "Use [ModeSpec](https://example.com/mode) with **global** $y$.\n"
    )


def test_does_not_normalize_blockquoted_or_nested_fences():
    source = (
        "> > ```markdown\n"
        "> > [`ModeSpec`](https://example.com/mode) and **global $y$**\n"
        "> > ```\n"
        "- ~~~markdown\n"
        "  [`ModeSpec`](https://example.com/mode) and **global $y$**\n"
        "  ~~~\n"
        "[`ModeSpec`](https://example.com/mode) and **global $y$**\n"
    )

    assert normalize_markdown_for_nbsphinx(source) == (
        "> > ```markdown\n"
        "> > [`ModeSpec`](https://example.com/mode) and **global $y$**\n"
        "> > ```\n"
        "- ~~~markdown\n"
        "  [`ModeSpec`](https://example.com/mode) and **global $y$**\n"
        "  ~~~\n"
        "[ModeSpec](https://example.com/mode) and **global** $y$\n"
    )


def test_does_not_normalize_indented_code_blocks():
    source = (
        "    [`ModeSpec`](https://example.com/mode) and **global $y$**\n"
        ">     [`ModeSpec`](https://example.com/mode) and **global $y$**\n"
        "[`ModeSpec`](https://example.com/mode) and **global $y$**\n"
    )

    assert normalize_markdown_for_nbsphinx(source) == (
        "    [`ModeSpec`](https://example.com/mode) and **global $y$**\n"
        ">     [`ModeSpec`](https://example.com/mode) and **global $y$**\n"
        "[ModeSpec](https://example.com/mode) and **global** $y$\n"
    )


def test_normalizes_only_markdown_cells_in_notebook_json():
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["Use [`ModeSpec`](https://example.com).\n", "**global $y$**"],
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": "print('**global $y$**')",
                "outputs": [],
                "execution_count": None,
            },
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    source = [json.dumps(notebook)]

    normalize_notebook_source(None, "notebooks/example", source)

    normalized = json.loads(source[0])
    assert normalized["cells"][0]["source"] == [
        "Use [ModeSpec](https://example.com).\n",
        "**global** $y$",
    ]
    assert normalized["cells"][1]["source"] == "print('**global $y$**')"


def test_removes_release_artifact_output_with_only_an_orphan_closing_tag():
    notebook = {
        "cells": [
            {
                "cell_type": "code",
                "metadata": {},
                "source": "df.head(10)",
                "outputs": [
                    {
                        "output_type": "display_data",
                        "metadata": {},
                        "data": {"text/html": ["\n", "</div>"]},
                    }
                ],
                "execution_count": 6,
            }
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    source = [json.dumps(notebook)]

    normalize_notebook_source(None, "notebooks/AnisotropicBendsEME", source)

    normalized = json.loads(source[0])
    assert normalized["cells"][0]["outputs"] == []


def test_preserves_fallback_mime_data_when_removing_orphan_html():
    notebook = {
        "cells": [
            {
                "cell_type": "code",
                "metadata": {},
                "source": "df.head(10)",
                "outputs": [
                    {
                        "output_type": "display_data",
                        "metadata": {},
                        "data": {
                            "text/html": "</div>",
                            "text/plain": ["   value\n", "0      1"],
                        },
                    }
                ],
                "execution_count": 6,
            }
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    source = [json.dumps(notebook)]

    normalize_notebook_source(None, "notebooks/example", source)

    data = json.loads(source[0])["cells"][0]["outputs"][0]["data"]
    assert data == {"text/plain": ["   value\n", "0      1"]}


def test_preserves_balanced_html_output():
    notebook = {
        "cells": [
            {
                "cell_type": "code",
                "metadata": {},
                "source": "df.head(10)",
                "outputs": [
                    {
                        "output_type": "display_data",
                        "metadata": {},
                        "data": {"text/html": ["<div>", "<table></table>", "</div>"]},
                    }
                ],
                "execution_count": 6,
            }
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    original = json.dumps(notebook)
    source = [original]

    normalize_notebook_source(None, "notebooks/example", source)

    assert source == [original]


def test_ignores_non_notebook_sources():
    source = ["Use [`ModeSpec`](https://example.com)."]

    normalize_notebook_source(None, "guide", source)

    assert source == ["Use [`ModeSpec`](https://example.com)."]


def test_setup_registers_source_read_handler():
    app = Mock()

    from notebook_markdown import setup

    assert setup(app) == {"parallel_read_safe": True}
    app.connect.assert_called_once_with("source-read", normalize_notebook_source)


def test_nbsphinx_build_does_not_leak_intermediate_rst(tmp_path: Path):
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "html"
    source_dir.mkdir()
    extension_dir = Path(__file__).resolve().parent
    (source_dir / "conf.py").write_text(
        "import sys\n"
        f"sys.path.insert(0, {str(extension_dir)!r})\n"
        "extensions = ['nbsphinx', 'notebook_markdown']\n"
        "master_doc = 'index'\n"
        "nbsphinx_execute = 'never'\n",
        encoding="utf-8",
    )
    (source_dir / "index.rst").write_text(
        "Notebook\n========\n\n.. toctree::\n\n   example\n",
        encoding="utf-8",
    )
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "id": "example-markdown",
                "metadata": {},
                "source": (
                    "# Example\n\nUse [`ModeSpec`](https://example.com/mode) with **global $y$** "
                    "and **$N = 1$**."
                ),
            },
            {
                "cell_type": "code",
                "id": "orphan-html-output",
                "metadata": {},
                "source": "display('table')",
                "outputs": [
                    {
                        "output_type": "display_data",
                        "metadata": {},
                        "data": {"text/html": ["\n", "</orphanclosingtag>"]},
                    }
                ],
                "execution_count": 1,
            },
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (source_dir / "example.ipynb").write_text(json.dumps(notebook), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, "-m", "sphinx", "-W", "-b", "html", source_dir, output_dir],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    html = (output_dir / "example.html").read_text(encoding="utf-8")
    assert 'href="https://example.com/mode">ModeSpec</a>' in html
    assert "<strong>global</strong>" in html
    assert ":math:`" not in html
    assert "&gt;`__" not in html
    assert "</orphanclosingtag>" not in html
