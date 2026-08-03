from __future__ import annotations

import ast
import io
import json
import runpy
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import generate_faq_docs
import pytest
from sphinx.application import Sphinx

FAQ_ROOT = Path(__file__).parent / "faq"


def _skip_if_faq_lfs_assets_are_unavailable() -> None:
    image_root = FAQ_ROOT / "_faqs" / "img"
    for path in image_root.rglob("*"):
        if not path.is_file():
            continue
        with path.open("rb") as file:
            if file.read(len(generate_faq_docs.LFS_POINTER_PREFIX)) == (
                generate_faq_docs.LFS_POINTER_PREFIX
            ):
                pytest.skip("FAQ corpus tests require hydrated Git LFS image objects")


def _snapshot(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _write_fixture(root: Path, *, image_target: str = "./img/example.png") -> None:
    raw_root = root / "_faqs"
    image_root = raw_root / "img"
    image_root.mkdir(parents=True)
    (image_root / "example.png").write_bytes(b"image")
    (raw_root / "example.md").write_text(
        "---\n"
        "title: Example\n"
        "date: 2026-07-29\n"
        "enabled: true\n"
        "category: General\n"
        "---\n"
        f"![Example]({image_target})\n",
        encoding="utf-8",
    )
    (root / "faq_categories.json").write_text(
        json.dumps([{"category": "General", "id": "general", "faqs": ["_faqs/example.md"]}]),
        encoding="utf-8",
    )
    (root / generate_faq_docs.ALIAS_MANIFEST).write_text("{}\n", encoding="utf-8")


def test_corpus_tests_skip_when_lfs_assets_are_unavailable(monkeypatch, tmp_path: Path) -> None:
    image_root = tmp_path / "_faqs" / "img"
    image_root.mkdir(parents=True)
    (image_root / "example.png").write_bytes(
        generate_faq_docs.LFS_POINTER_PREFIX
        + b"oid sha256:0000000000000000000000000000000000000000000000000000000000000000\n"
        + b"size 1\n"
    )
    monkeypatch.setattr(sys.modules[__name__], "FAQ_ROOT", tmp_path)

    with pytest.raises(pytest.skip.Exception, match="hydrated Git LFS"):
        _skip_if_faq_lfs_assets_are_unavailable()


def test_imported_faq_sources_are_complete_and_valid() -> None:
    _skip_if_faq_lfs_assets_are_unavailable()
    categories = generate_faq_docs.load_and_validate(FAQ_ROOT)

    assert len(categories) == 22
    assert sum(len(category.faqs) for category in categories) == 255
    assert len({faq.manifest_path for category in categories for faq in category.faqs}) == 255


def test_parallel_checkpoint_faq_distinguishes_web_run_from_direct_batch() -> None:
    source = (
        FAQ_ROOT
        / "_faqs"
        / "how-do-i-save-or-load-a-tidy3d-parallel-job-so-i-can-work-with-it-later.md"
    ).read_text(encoding="utf-8")

    assert "its direct return value preserves that structure" in source
    assert "checkpoint records the submitted tasks" in source
    assert "does not recreate the caller's original keys, indices, or nesting" in source
    assert "construct directly from a nested simulation container behaves differently" in source
    assert "round trip preserves that `Batch` container structure" in source
    assert "internal flat" not in source
    assert "container-aware" not in source
    assert "hash-based" not in source
    assert 'batch.load(path_dir="path")' in source


def test_abort_parallel_faq_warns_that_batch_delete_is_destructive() -> None:
    source = (FAQ_ROOT / "_faqs" / "how-do-I-abort-multiple-simulations-job.md").read_text(
        encoding="utf-8"
    )

    assert "for batch_name, job in batch.jobs.items()" in source
    assert "print(batch_name, job.status, job.task_ids)" in source
    assert "web.abort(task_id)" in source
    assert "use the server task IDs printed from `batch.jobs`" in source
    assert "works for both single-step simulations and multi-step workflows" in source
    assert "construct and save a `web.Batch` directly" in source
    assert "task_name_1" not in source
    assert "`Batch.delete()` is not a cancellation-only operation" in source
    assert "permanently deletes every server-side task" in source
    assert "including completed results and their data" in source


def test_job_faqs_do_not_assume_every_workflow_has_one_task_id() -> None:
    job_faqs = (
        "how-do-I-abort-multiple-simulations-job.md",
        "how-can-i-optimize-the-simulation-cost.md",
        "how-do-i-estimate-how-many-credits-my-simulation-will-take.md",
        "how-do-i-see-the-cost-of-my-simulation.md",
        "how-do-i-upload-a-job-to-the-web-without-running-it-so-i-can-inspect-it-first.md",
        "web.Job.md",
        "web.estimate_cost.md",
    )
    sources = {name: (FAQ_ROOT / "_faqs" / name).read_text(encoding="utf-8") for name in job_faqs}

    assert all(
        "job.task_id" not in source.replace("job.task_ids", "")
        for name, source in sources.items()
        if name != "web.Job.md"
    )
    assert "Most simulations have one server task and expose `job.task_id`" in sources["web.Job.md"]
    assert "job.task_ids" in sources["how-do-I-abort-multiple-simulations-job.md"]
    assert "job.task_ids" in sources["web.Job.md"]
    assert (
        "job.task_ids"
        in sources[
            "how-do-i-upload-a-job-to-the-web-without-running-it-so-i-can-inspect-it-first.md"
        ]
    )
    assert "job.estimate_cost()" in sources["web.estimate_cost.md"]
    assert "job.real_cost()" in sources["how-do-i-see-the-cost-of-my-simulation.md"]
    assert "job.step()" in sources["web.Job.md"]


def test_web_run_async_faq_supports_parallel_vgpu_without_payment_override() -> None:
    source = (FAQ_ROOT / "_faqs" / "web.run_async.md").read_text(encoding="utf-8")

    assert "Ordinary batches return" in source
    assert "traced FDTD parameters trigger automatic differentiation" in source
    assert "returns a dictionary mapping task names to `SimulationData`" in source
    assert "cannot be mixed with other workflow types" in source
    assert "do not require switching to the FlexCredit pool" in source
    assert "tidy3d.config.vgpu.vgpu_allocation = 4" in source
    assert "tidy3d.config.vgpu.priority = 5" in source
    assert "pay_type" not in source


def test_batch_submission_faqs_document_the_conditional_run_async_return() -> None:
    for faq_name in (
        "how-do-i-run-a-batch-simulation.md",
        "how-do-i-submit-batch-simulations.md",
    ):
        source = (FAQ_ROOT / "_faqs" / faq_name).read_text(encoding="utf-8")

        assert "For the ordinary, untraced simulations above" in source
        assert "dict[str, SimulationData]" in source
        assert "traced FDTD tasks cannot be mixed with other workflow types" in source
        assert "does not provide <code>BatchData</code>-specific helpers" in source


def test_web_batch_faq_describes_default_eager_results() -> None:
    source = (FAQ_ROOT / "_faqs" / "web.Batch.md").read_text(encoding="utf-8")

    assert "results already downloaded by Batch.run()" in source
    assert "lazy-loaded from disk" not in source
    assert "downloads on demand" not in source


def test_bloch_boundary_faq_balances_method_code_markup() -> None:
    source = (FAQ_ROOT / "_faqs" / "how-do-i-set-the-bloch-boundary-condition.md").read_text(
        encoding="utf-8"
    )

    assert "<code>.bloch_from_source()</code>" in source
    assert "method .bloch_from_source()</code>" not in source


def test_public_faqs_do_not_reference_private_module_paths() -> None:
    private_paths = ("tidy3d.web.api.", "tidy3d.components.")
    violations = {
        path.name: private_path
        for path in sorted((FAQ_ROOT / "_faqs").glob("*.md"))
        for private_path in private_paths
        if private_path in path.read_text(encoding="utf-8")
    }

    assert violations == {}


def test_public_faq_links_do_not_include_hubspot_tracking_parameters() -> None:
    tracking_parameters = ("__hstc", "__hssc", "__hsfp")
    violations = {
        path.name: parameter
        for path in sorted((FAQ_ROOT / "_faqs").glob("*.md"))
        for parameter in tracking_parameters
        if parameter in path.read_text(encoding="utf-8")
    }

    assert violations == {}


def test_public_run_async_autosummary_target_has_reference_documentation() -> None:
    source_path = (
        Path(__file__).parent.parent / "tidy3d" / "web" / "api" / "autograd" / "autograd.py"
    )
    module = ast.parse(source_path.read_text(encoding="utf-8"))
    run_async = next(
        node
        for node in module.body
        if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)) and node.name == "run_async"
    )
    docstring = ast.get_docstring(run_async)
    simulations_annotation = ast.unparse(run_async.args.args[0].annotation)
    return_annotation = ast.unparse(run_async.returns)

    assert docstring is not None
    assert "Parameters\n----------" in docstring
    assert "Returns\n-------" in docstring
    assert "Notes\n-----" in docstring
    assert "Examples\n--------" in docstring
    assert "simulations" in docstring
    assert "path_dir" in docstring
    assert "traced FDTD" in docstring
    assert "cannot be mixed with" in docstring
    assert "run_async_custom" not in docstring
    assert "config.run" not in docstring
    assert "WorkflowOperationType" not in docstring
    assert "WorkflowOperationType" not in simulations_annotation
    assert "tuple[" in simulations_annotation
    assert ", ...]" in simulations_annotation
    assert "ModeSolver" in simulations_annotation
    assert "ModalComponentModeler" in simulations_annotation
    assert "BatchData" in return_annotation
    assert "dict[str, SimulationData]" in return_annotation
    assert "traced FDTD parameters returns a dictionary" in docstring
    assert "Traced FDTD batches always return an" in docstring
    assert "eagerly materialized dictionary" in docstring


def test_public_batch_autosummary_stubs_have_one_canonical_owner() -> None:
    api_root = Path(__file__).parent / "api"
    simulation_source = (api_root / "simulation.rst").read_text(encoding="utf-8")
    output_source = (api_root / "output_data.rst").read_text(encoding="utf-8")
    submission_source = (api_root / "submit_simulations.rst").read_text(encoding="utf-8")

    assert ".. autosummary::\n\n   web.Batch\n" in simulation_source
    assert ".. autosummary::\n\n   web.BatchData\n" in output_source
    assert (
        ".. autosummary::\n"
        "   :toctree: _autosummary/\n"
        "   :template: module.rst\n\n"
        "   web.Job\n"
        "   web.Batch\n"
        "   web.BatchData\n"
    ) in submission_source


def test_run_async_custom_documents_its_workflow_batch_contract() -> None:
    source_path = (
        Path(__file__).parent.parent / "tidy3d" / "web" / "api" / "autograd" / "autograd.py"
    )
    module = ast.parse(source_path.read_text(encoding="utf-8"))
    run_async_custom = next(
        node
        for node in module.body
        if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef))
        and node.name == "run_async_custom"
    )
    docstring = ast.get_docstring(run_async_custom)
    simulations_annotation = ast.unparse(run_async_custom.args.args[0].annotation)
    return_annotation = ast.unparse(run_async_custom.returns)

    assert docstring is not None
    assert "ModeSolver" in simulations_annotation
    assert "ModalComponentModeler" in simulations_annotation
    assert "WorkflowOperationType" not in simulations_annotation
    assert "tuple[" in simulations_annotation
    assert ", ...]" in simulations_annotation
    assert "BatchData" in return_annotation
    assert "dict[str, SimulationData]" in return_annotation
    assert "Ordinary workflow batches return" in docstring
    assert "traced FDTD parameters returns a dictionary" in docstring
    assert "cannot be mixed with" in docstring
    assert "config.run" not in docstring
    assert "TODO" not in docstring


def test_parameter_sweep_cost_faq_accepts_nested_batch_input() -> None:
    source = (FAQ_ROOT / "_faqs" / "how-can-i-estimate-the-cost-of-a-parameter-sweep.md").read_text(
        encoding="utf-8"
    )

    assert "web.Batch(simulations=sims).estimate_cost()" in source
    assert "first flatten" not in source
    assert "flatten_sims" not in source


def test_web_run_faq_matches_the_public_container_wrapper() -> None:
    source = (FAQ_ROOT / "_faqs" / "web.run.md").read_text(encoding="utf-8")

    assert "dictionary, list, tuple, or nested combination" in source
    assert "returned data preserves the input container structure" in source


def test_eme_cloud_parallel_faq_uses_the_explicit_batch_api() -> None:
    source = (FAQ_ROOT / "_faqs" / "how-do-i-run-eme-locally.md").read_text(encoding="utf-8")

    assert "web.run_async(mode_sims)" in source
    assert 'web.run(mode_sims, task_name="eme_local_modes")' not in source


def test_web_delete_old_faq_matches_the_public_signature() -> None:
    source = (FAQ_ROOT / "_faqs" / "web.delete_old.md").read_text(encoding="utf-8")

    assert "**days_old** *(int)*" in source
    assert '**folder_name** *(str, default="default")*' in source
    assert "**folder**" not in source


def test_mode_sorting_faq_explains_cross_frequency_tracking() -> None:
    source = (FAQ_ROOT / "_faqs" / "how-are-the-output-modes-sorted.md").read_text(encoding="utf-8")

    assert 'ModeSortSpec.track_freq</code> defaults to <code>"central"' in source
    assert "order is exact at the central frequency" in source
    assert "overlap-based tracking can change the order at the other frequencies" in source
    assert "within-group ordering is exact only at the frequency selected" in source
    assert "from tidy3d import web" in source
    assert 'web.run(mode_solver, task_name="mode_sorting")' in source
    assert "tidy3d.plugins.mode.web" not in source


def test_uniform_grid_faq_does_not_reference_an_undefined_structure() -> None:
    source = (FAQ_ROOT / "_faqs" / "how-can-i-create-a-uniform-grid.md").read_text(encoding="utf-8")

    assert "tidy3d.GridSpec.uniform(dl=0.02)" in source
    assert "structures=[structure]" not in source


@pytest.mark.parametrize(
    "faq_name",
    [
        "how-do-i-inject-a-specific-optical-mode-in-a-waveguide.md",
        "how-do-i-set-a-modesource.md",
    ],
)
def test_modesource_faqs_require_verifying_the_selected_mode(faq_name: str) -> None:
    source = (FAQ_ROOT / "_faqs" / faq_name).read_text(encoding="utf-8")

    assert "selects the second returned mode" in source
    assert "does not by itself identify that mode as TE1" in source
    assert "inspect the modal fields and polarization fractions" in source
    assert "to inject the first-order TE mode" not in source


def test_generation_is_deterministic(tmp_path: Path) -> None:
    _skip_if_faq_lfs_assets_are_unavailable()
    first = tmp_path / "first"
    second = tmp_path / "second"

    generate_faq_docs.generate(FAQ_ROOT, first)
    generate_faq_docs.generate(FAQ_ROOT, second)
    generate_faq_docs.generate(FAQ_ROOT, first)

    assert _snapshot(first) == _snapshot(second)
    manifest = json.loads((first / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest) == 255
    assert (first / "index.rst").is_file()
    assert (first / generate_faq_docs.GENERATED_MARKER).is_file()
    aliases = json.loads((first / "aliases.json").read_text(encoding="utf-8"))
    assert len(aliases) == 11
    assert aliases["faq/How-are-results-normalized.html"] == ("faq/how-are-results-normalized.html")
    assert (first / "faq" / "How-are-results-normalized.md").is_file()
    assert not any(
        item["source"].endswith(
            "how-do-i-load-the-data-from-a-simulation-task-id-task-id-into-the-python-client.md"
        )
        for item in manifest
    )


def test_generate_for_sphinx_creates_required_faq_tree(tmp_path: Path) -> None:
    docs_root = tmp_path / "source" / "docs"
    _write_fixture(docs_root / "faq")
    staging_package_root = tmp_path / "staged"
    staged_docs_root = staging_package_root / "docs"
    source_snapshot = _snapshot(docs_root)

    categories = generate_faq_docs.generate_for_sphinx(docs_root, staging_package_root)

    assert len(categories) == 1
    assert (staged_docs_root / "faq" / "docs" / "index.rst").is_file()
    assert (staged_docs_root / "faq" / "docs" / "faq" / "example.md").is_file()
    assert not (docs_root / "faq" / "docs").exists()
    assert _snapshot(docs_root) == source_snapshot


def test_generate_for_sphinx_isolates_sources_from_sphinx_writes(tmp_path: Path) -> None:
    docs_root = tmp_path / "source" / "docs"
    _write_fixture(docs_root / "faq")
    (docs_root / "index.rst").write_text("Staged documentation.\n", encoding="utf-8")
    staging_package_root = tmp_path / "staged"
    staged_docs_root = staging_package_root / "docs"
    source_snapshot = _snapshot(docs_root)

    generate_faq_docs.generate_for_sphinx(docs_root, staging_package_root)

    assert (staged_docs_root / "index.rst").read_text(encoding="utf-8") == (
        "Staged documentation.\n"
    )
    assert (staged_docs_root / "faq" / "_faqs" / "example.md").is_file()
    assert (staged_docs_root / "faq" / "docs" / "faq" / "example.md").is_file()
    assert not (staged_docs_root / "index.rst").is_symlink()
    assert not (staged_docs_root / "faq").is_symlink()

    (staged_docs_root / "index.rst").write_text("Sphinx modified this.\n", encoding="utf-8")
    autosummary_root = staged_docs_root / "api" / "_autosummary"
    autosummary_root.mkdir(parents=True)
    (autosummary_root / "generated.rst").write_text("Generated by Sphinx.\n", encoding="utf-8")

    assert _snapshot(docs_root) == source_snapshot
    assert not (docs_root / "api").exists()


def test_generate_for_sphinx_preserves_external_include_layout(tmp_path: Path) -> None:
    package_root = tmp_path / "source"
    docs_root = package_root / "docs"
    _write_fixture(docs_root / "faq")
    (package_root / "CHANGELOG.md").write_text("Included changelog.\n", encoding="utf-8")
    plugin_readme = package_root / "tidy3d" / "plugins" / "example" / "README.md"
    plugin_readme.parent.mkdir(parents=True)
    plugin_readme.write_text(
        "Included plugin documentation.\n\n.. image:: ../../../notebooks/img/plugin.png\n",
        encoding="utf-8",
    )
    plugin_image = package_root / "notebooks" / "img" / "plugin.png"
    plugin_image.parent.mkdir(parents=True)
    plugin_image.write_bytes(b"\x89PNG\r\n\x1a\n")
    (docs_root / "changelog.rst").write_text(
        ".. include:: ../CHANGELOG.md\n",
        encoding="utf-8",
    )
    plugin_docs = docs_root / "api" / "plugins"
    plugin_docs.mkdir(parents=True)
    (plugin_docs / "example.rst").write_text(
        ".. include:: ../../../tidy3d/plugins/example/README.md\n",
        encoding="utf-8",
    )
    staging_package_root = tmp_path / "staged"

    generate_faq_docs.generate_for_sphinx(docs_root, staging_package_root)

    assert (staging_package_root / "CHANGELOG.md").read_text(encoding="utf-8") == (
        "Included changelog.\n"
    )
    assert (
        (staging_package_root / "tidy3d" / "plugins" / "example" / "README.md")
        .read_text(encoding="utf-8")
        .startswith("Included plugin documentation.\n")
    )
    assert (staging_package_root / "notebooks" / "img" / "plugin.png").read_bytes() == (
        b"\x89PNG\r\n\x1a\n"
    )


def test_generate_for_sphinx_rejects_include_outside_package(tmp_path: Path) -> None:
    package_root = tmp_path / "source"
    docs_root = package_root / "docs"
    _write_fixture(docs_root / "faq")
    (tmp_path / "outside.md").write_text("Outside package.\n", encoding="utf-8")
    (docs_root / "unsafe.rst").write_text(
        ".. include:: ../../outside.md\n",
        encoding="utf-8",
    )
    staging_package_root = tmp_path / "staged"

    with pytest.raises(generate_faq_docs.FaqValidationError, match="escapes the package root"):
        generate_faq_docs.generate_for_sphinx(docs_root, staging_package_root)

    assert not staging_package_root.exists()


def test_generate_for_sphinx_rejects_a_symlinked_image_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    docs_root = tmp_path / "source" / "docs"
    _write_fixture(docs_root / "faq")
    image_root = docs_root / "faq" / "_faqs" / "img"
    shutil.rmtree(image_root)
    outside_image_root = tmp_path / "outside-images"
    outside_image_root.mkdir()
    (outside_image_root / "example.png").write_bytes(generate_faq_docs.LFS_POINTER_PREFIX)
    image_root.symlink_to(outside_image_root, target_is_directory=True)

    def reject_lfs_pull(*_args, **_kwargs):
        raise AssertionError("a symlinked image root must be rejected before Git LFS runs")

    monkeypatch.setattr(subprocess, "run", reject_lfs_pull)

    with pytest.raises(
        generate_faq_docs.FaqValidationError,
        match=r"Sphinx source tree.*faq/_faqs/img",
    ):
        generate_faq_docs.generate_for_sphinx(docs_root, tmp_path / "staged")

    assert image_root.is_symlink()
    assert not (tmp_path / "staged").exists()


@pytest.mark.parametrize("nested_directory", [False, True])
def test_generate_for_sphinx_rejects_symlinked_source_entries(
    tmp_path: Path, nested_directory: bool
) -> None:
    docs_root = tmp_path / "source" / "docs"
    _write_fixture(docs_root / "faq")
    if nested_directory:
        outside = tmp_path / "outside-docs"
        outside.mkdir()
        (outside / "private.rst").write_text("Private\n", encoding="utf-8")
        api_root = docs_root / "api"
        api_root.mkdir()
        source_symlink = api_root / "linked-docs"
        source_symlink.symlink_to(outside, target_is_directory=True)
    else:
        outside = tmp_path / "outside.rst"
        outside.write_text("Private\n", encoding="utf-8")
        source_symlink = docs_root / "linked.rst"
        source_symlink.symlink_to(outside)

    staging_root = tmp_path / "staged"
    with pytest.raises(
        generate_faq_docs.FaqValidationError,
        match=r"Sphinx source tree may not contain symlinks",
    ):
        generate_faq_docs.generate_for_sphinx(docs_root, staging_root)

    assert source_symlink.is_symlink()
    assert not staging_root.exists()


def test_sphinx_configuration_registers_faq_staging(monkeypatch) -> None:
    fake_tidy3d = ModuleType("tidy3d")
    fake_tidy3d.__version__ = "test"
    monkeypatch.setitem(sys.modules, "tidy3d", fake_tidy3d)
    monkeypatch.setenv("TIDY3D_DOCS_VERSION", "latest")
    config = runpy.run_path(Path(__file__).parent / "conf.py")
    connections = []

    class FakeSphinxApp:
        def connect(self, event, callback, **kwargs):
            connections.append((event, callback, kwargs))

    config["setup"](FakeSphinxApp())

    assert "faq/README.md" in config["exclude_patterns"]
    assert ("config-inited", config["_stage_faq_sources"], {"priority": 100}) in connections
    assert ("build-finished", config["_write_legacy_api_doc_redirects"], {}) in connections
    assert ("build-finished", config["_cleanup_staged_faq_sources"], {}) in connections


def test_sphinx_configuration_writes_legacy_api_doc_redirects(monkeypatch, tmp_path: Path) -> None:
    fake_tidy3d = ModuleType("tidy3d")
    fake_tidy3d.__version__ = "test"
    monkeypatch.setitem(sys.modules, "tidy3d", fake_tidy3d)
    monkeypatch.setenv("TIDY3D_DOCS_VERSION", "latest")
    config = runpy.run_path(Path(__file__).parent / "conf.py")

    class FakeBuilder:
        format = "html"

        @staticmethod
        def get_outfilename(docname):
            return tmp_path / f"{docname}.html"

    class FakeSphinxApp:
        builder = FakeBuilder()

    assert config["LEGACY_API_DOC_REDIRECTS"] == {
        "api/_autosummary/tidy3d.web.api.asynchronous.run_async": (
            "api/_autosummary/tidy3d.web.run_async"
        ),
        "api/_autosummary/tidy3d.web.api.container.Batch": "api/_autosummary/tidy3d.web.Batch",
        "api/_autosummary/tidy3d.web.api.container.BatchData": (
            "api/_autosummary/tidy3d.web.BatchData"
        ),
        "api/_autosummary/tidy3d.web.api.container.Job": "api/_autosummary/tidy3d.web.Job",
    }
    config["_write_legacy_api_doc_redirects"](FakeSphinxApp(), None)

    for legacy_docname, canonical_docname in config["LEGACY_API_DOC_REDIRECTS"].items():
        redirect_path = tmp_path / f"{legacy_docname}.html"
        expected_target = f"{Path(canonical_docname).name}.html"
        redirect = redirect_path.read_text(encoding="utf-8")
        assert f'content="0; url={expected_target}"' in redirect
        assert f'href="{expected_target}"' in redirect


def test_sphinx_doc_targets_follow_the_active_source_tree(monkeypatch, tmp_path: Path) -> None:
    fake_tidy3d = ModuleType("tidy3d")
    fake_tidy3d.__version__ = "test"
    monkeypatch.setitem(sys.modules, "tidy3d", fake_tidy3d)
    monkeypatch.setenv("TIDY3D_DOCS_VERSION", "latest")
    config = runpy.run_path(Path(__file__).parent / "conf.py")
    staged_docs_root = tmp_path / "staged-docs"
    autosummary_root = staged_docs_root / "api" / "_autosummary"
    autosummary_root.mkdir(parents=True)
    (autosummary_root / "tidy3d.Box.rst").write_text("Box\n", encoding="utf-8")

    assert config["_get_doc_targets"](staged_docs_root) == {"tidy3d.Box"}

    observed_roots = []

    def observe_doc_targets(source_root):
        observed_roots.append(Path(source_root))
        return {"tidy3d.Box"}

    process_docstring = config["autodoc_process_docstring"]
    monkeypatch.setitem(process_docstring.__globals__, "_get_doc_targets", observe_doc_targets)
    monkeypatch.setitem(
        process_docstring.__globals__,
        "_get_alias_type_docs",
        lambda: {"Alias": "Alias"},
    )
    monkeypatch.setitem(process_docstring.__globals__, "_get_tidy3d_class_map", dict)

    class DocumentedObject:
        __module__ = "tidy3d"

    class FakeSphinxApp:
        srcdir = staged_docs_root

    process_docstring(
        FakeSphinxApp(),
        "class",
        "tidy3d.DocumentedObject",
        DocumentedObject,
        None,
        ["Documentation."],
    )

    assert observed_roots == [staged_docs_root]


def test_render_adds_missing_common_imports_to_copyable_python(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    faq_file = tmp_path / "_faqs" / "example.md"
    faq_file.write_text(
        faq_file.read_text(encoding="utf-8")
        + "\n{% highlight python %}\n"
        + "tidy3d.Box(size=np.ones(3))\n"
        + "plt.plot([0, 1])\n"
        + "{% endhighlight %}\n",
        encoding="utf-8",
    )

    categories = generate_faq_docs.load_and_validate(tmp_path)
    rendered = generate_faq_docs._render_faq(categories[0].faqs[0])

    assert (
        "```python\n"
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "import tidy3d\n\n"
        "tidy3d.Box(size=np.ones(3))"
    ) in rendered


def test_render_preserves_existing_common_imports(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    faq_file = tmp_path / "_faqs" / "example.md"
    faq_file.write_text(
        faq_file.read_text(encoding="utf-8")
        + "\n{% highlight python %}\n"
        + "import tidy3d\n"
        + "tidy3d.Box(size=(1, 1, 1))\n"
        + "{% endhighlight %}\n",
        encoding="utf-8",
    )

    categories = generate_faq_docs.load_and_validate(tmp_path)
    rendered = generate_faq_docs._render_faq(categories[0].faqs[0])

    assert rendered.count("import tidy3d") == 1


def test_render_adds_missing_imports_after_future_imports() -> None:
    rendered = generate_faq_docs._add_missing_common_imports(
        "python",
        "from __future__ import annotations\n\nvalues: np.ndarray = np.ones(3)",
    )

    assert rendered.startswith(
        "from __future__ import annotations\nimport numpy as np\n\nvalues: np.ndarray"
    )
    compile(rendered, "<faq-example>", "exec")


def test_render_adds_missing_third_party_imports(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    faq_file = tmp_path / "_faqs" / "example.md"
    faq_file.write_text(
        faq_file.read_text(encoding="utf-8")
        + "\n{% highlight python %}\n"
        + "points = anp.array([[0, 0]])\n"
        + 'cell = gdstk.Cell("TOP")\n'
        + "mesh = trimesh.creation.box()\n"
        + "{% endhighlight %}\n",
        encoding="utf-8",
    )

    categories = generate_faq_docs.load_and_validate(tmp_path)
    rendered = generate_faq_docs._render_faq(categories[0].faqs[0])

    assert ("```python\nimport autograd.numpy as anp\nimport gdstk\nimport trimesh\n\n") in rendered


def test_render_adds_missing_common_imports_to_markdown_fence(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    faq_file = tmp_path / "_faqs" / "example.md"
    faq_file.write_text(
        faq_file.read_text(encoding="utf-8")
        + "\n```python\n"
        + "tidy3d.Box(size=(1, 1, 1))\n"
        + "```\n",
        encoding="utf-8",
    )

    categories = generate_faq_docs.load_and_validate(tmp_path)
    rendered = generate_faq_docs._render_faq(categories[0].faqs[0])

    assert "```python\nimport tidy3d\n\ntidy3d.Box" in rendered


def test_render_only_normalizes_jekyll_and_html_markup_outside_fences(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    faq_file = tmp_path / "_faqs" / "example.md"
    faq_file.write_text(
        faq_file.read_text(encoding="utf-8")
        + "\n<span>Outside prose</span>{: .note}\n"
        + "\n```python\n"
        + 'formatted = f"{7:02d}"\n'
        + 'inline_html = "<span>keep</span>"\n'
        + 'block_html = "<div>keep</div>"\n'
        + 'jekyll = "{% include keep.html %}"\n'
        + 'highlight = "{% highlight python %}keep{% endhighlight %}"\n'
        + "```\n",
        encoding="utf-8",
    )

    categories = generate_faq_docs.load_and_validate(tmp_path)
    rendered = generate_faq_docs._render_faq(categories[0].faqs[0])

    assert "Outside prose" in rendered
    assert "<span>Outside prose</span>" not in rendered
    assert "{: .note}" not in rendered
    assert 'formatted = f"{7:02d}"' in rendered
    assert 'inline_html = "<span>keep</span>"' in rendered
    assert 'block_html = "<div>keep</div>"' in rendered
    assert 'jekyll = "{% include keep.html %}"' in rendered
    assert 'highlight = "{% highlight python %}keep{% endhighlight %}"' in rendered


def test_render_converts_local_html_images_to_markdown(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    faq_file = tmp_path / "_faqs" / "example.md"
    faq_file.write_text(
        faq_file.read_text(encoding="utf-8").replace(
            "![Example](./img/example.png)",
            '<div><img width="640" src="./img/example.png" /></div>',
        ),
        encoding="utf-8",
    )

    categories = generate_faq_docs.load_and_validate(tmp_path)
    rendered = generate_faq_docs._render_faq(categories[0].faqs[0])

    assert "![example](./img/example.png)" in rendered
    assert "<img" not in rendered


def test_render_preserves_jekyll_semantic_roles(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    faq_file = tmp_path / "_faqs" / "example.md"
    faq_file.write_text(
        faq_file.read_text(encoding="utf-8")
        + '\nUse `Simulation.shutoff`{: .interpreted-text role="py:obj"} and '
        + '`TFSFSource`{: .interpreted-text role="class"}.\n',
        encoding="utf-8",
    )

    categories = generate_faq_docs.load_and_validate(tmp_path)
    rendered = generate_faq_docs._render_faq(categories[0].faqs[0])

    assert "{py:obj}`Simulation.shutoff`" in rendered
    assert "{py:class}`TFSFSource`" in rendered
    assert "{: .interpreted-text" not in rendered


def test_validation_rejects_an_unlisted_enabled_faq(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    (tmp_path / "_faqs" / "orphan.md").write_text(
        "---\ntitle: Orphan\ndate: 2026-07-29\nenabled: true\ncategory: General\n---\n",
        encoding="utf-8",
    )

    with pytest.raises(
        generate_faq_docs.FaqValidationError,
        match="enabled FAQs missing from manifest",
    ):
        generate_faq_docs.load_and_validate(tmp_path)


def test_validation_rejects_a_missing_local_image(tmp_path: Path) -> None:
    _write_fixture(tmp_path, image_target="./img/missing.png")

    with pytest.raises(generate_faq_docs.FaqValidationError, match="does not exist"):
        generate_faq_docs.load_and_validate(tmp_path)


def test_validation_rejects_a_trailing_markdown_transition(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    faq_file = tmp_path / "_faqs" / "example.md"
    faq_file.write_text(
        faq_file.read_text(encoding="utf-8") + "\n---\n",
        encoding="utf-8",
    )

    with pytest.raises(generate_faq_docs.FaqValidationError, match="may not end"):
        generate_faq_docs.load_and_validate(tmp_path)


def test_validation_rejects_an_absolute_local_image_path(tmp_path: Path) -> None:
    _write_fixture(tmp_path, image_target="/img/example.png")

    with pytest.raises(generate_faq_docs.FaqValidationError, match=r"relative '\./img/'"):
        generate_faq_docs.load_and_validate(tmp_path)


def test_validation_rejects_an_unsafe_category_id(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    manifest_file = tmp_path / "faq_categories.json"
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    manifest[0]["id"] = "../outside"
    manifest_file.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(generate_faq_docs.FaqValidationError, match="safe, non-reserved"):
        generate_faq_docs.load_and_validate(tmp_path)


def test_validation_rejects_an_alias_for_a_different_route(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    (tmp_path / generate_faq_docs.ALIAS_MANIFEST).write_text(
        json.dumps({"legacy.md": "_faqs/example.md"}),
        encoding="utf-8",
    )

    with pytest.raises(generate_faq_docs.FaqValidationError, match="only by case"):
        generate_faq_docs.load_and_validate(tmp_path)


def test_validation_rejects_invalid_python_examples(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    faq_file = tmp_path / "_faqs" / "example.md"
    faq_file.write_text(
        faq_file.read_text(encoding="utf-8")
        + "\n{% highlight python %}\nfield_data..interp(x=0)\n{% endhighlight %}\n",
        encoding="utf-8",
    )

    with pytest.raises(generate_faq_docs.FaqValidationError, match="Python code block"):
        generate_faq_docs.load_and_validate(tmp_path)


def test_validation_rejects_invalid_fenced_python_examples(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    faq_file = tmp_path / "_faqs" / "example.md"
    faq_file.write_text(
        faq_file.read_text(encoding="utf-8") + "\n```python\nfield_data..interp(x=0)\n```\n",
        encoding="utf-8",
    )

    with pytest.raises(generate_faq_docs.FaqValidationError, match="Python code block"):
        generate_faq_docs.load_and_validate(tmp_path)


def test_validation_rejects_unimported_bare_tidy3d_symbols(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    faq_file = tmp_path / "_faqs" / "example.md"
    faq_file.write_text(
        faq_file.read_text(encoding="utf-8")
        + "\n{% highlight python %}\n"
        + "monitor = FieldMonitor(size=(1, 1, 1), freqs=[2e14])\n"
        + "{% endhighlight %}\n",
        encoding="utf-8",
    )

    with pytest.raises(generate_faq_docs.FaqValidationError, match="must import or qualify"):
        generate_faq_docs.load_and_validate(tmp_path)


def test_validation_rejects_repeated_code_block_language_marker(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    faq_file = tmp_path / "_faqs" / "example.md"
    faq_file.write_text(
        faq_file.read_text(encoding="utf-8")
        + "\n{% highlight python %}\npython\nprint('example')\n{% endhighlight %}\n",
        encoding="utf-8",
    )

    with pytest.raises(generate_faq_docs.FaqValidationError, match="repeats its"):
        generate_faq_docs.load_and_validate(tmp_path)


def test_validation_rejects_placeholder_links(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    faq_file = tmp_path / "_faqs" / "example.md"
    faq_file.write_text(
        faq_file.read_text(encoding="utf-8") + "\n[Example](None)\n",
        encoding="utf-8",
    )

    with pytest.raises(generate_faq_docs.FaqValidationError, match="placeholder targets"):
        generate_faq_docs.load_and_validate(tmp_path)


def test_validation_rejects_root_relative_links(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    faq_file = tmp_path / "_faqs" / "example.md"
    faq_file.write_text(
        faq_file.read_text(encoding="utf-8") + "\n[Example](/tidy3d/example/)\n",
        encoding="utf-8",
    )

    with pytest.raises(generate_faq_docs.FaqValidationError, match="site-root-relative"):
        generate_faq_docs.load_and_validate(tmp_path)


@pytest.mark.parametrize("docs_version", ["stable", "v2.9.0"])
def test_validation_rejects_non_latest_internal_docs_links(
    tmp_path: Path, docs_version: str
) -> None:
    _write_fixture(tmp_path)
    faq_file = tmp_path / "_faqs" / "example.md"
    faq_file.write_text(
        faq_file.read_text(encoding="utf-8")
        + f"\n[Example](https://docs.flexcompute.com/projects/tidy3d/en/{docs_version}/api/)\n",
        encoding="utf-8",
    )

    with pytest.raises(generate_faq_docs.FaqValidationError, match="must use the latest"):
        generate_faq_docs.load_and_validate(tmp_path)


def test_validation_rejects_internal_api_links_not_declared_by_sphinx(tmp_path: Path) -> None:
    docs_root = tmp_path / "docs"
    faq_root = docs_root / "faq"
    _write_fixture(faq_root)
    api_index = docs_root / "api" / "index.rst"
    api_index.parent.mkdir()
    api_index.write_text(
        ".. currentmodule:: tidy3d\n\n.. autosummary::\n   :toctree: _autosummary/\n\n   Box\n",
        encoding="utf-8",
    )
    faq_file = faq_root / "_faqs" / "example.md"
    faq_file.write_text(
        faq_file.read_text(encoding="utf-8")
        + "\n[Missing](https://docs.flexcompute.com/projects/tidy3d/en/latest/"
        + "api/_autosummary/tidy3d.Missing.html)\n",
        encoding="utf-8",
    )

    with pytest.raises(generate_faq_docs.FaqValidationError, match="declared by the Sphinx"):
        generate_faq_docs.load_and_validate(faq_root)


def test_validation_uses_committed_autosummary_declarations(tmp_path: Path) -> None:
    docs_root = tmp_path / "docs"
    faq_root = docs_root / "faq"
    _write_fixture(faq_root)
    api_source = docs_root / "api" / "mode" / "sources.rst"
    api_source.parent.mkdir(parents=True)
    api_source.write_text(
        ".. currentmodule:: tidy3d\n\n"
        ".. autosummary::\n"
        "   :toctree: _autosummary/\n\n"
        "   ModeSource\n",
        encoding="utf-8",
    )
    faq_file = faq_root / "_faqs" / "example.md"
    faq_file.write_text(
        faq_file.read_text(encoding="utf-8")
        + "\n[ModeSource](https://docs.flexcompute.com/projects/tidy3d/en/latest/"
        + "api/mode/_autosummary/tidy3d.ModeSource.html)\n",
        encoding="utf-8",
    )

    generate_faq_docs.load_and_validate(faq_root)
    assert not (docs_root / "api" / "mode" / "_autosummary").exists()


def test_validation_accepts_internal_api_links_to_committed_sphinx_sources(
    tmp_path: Path,
) -> None:
    docs_root = tmp_path / "docs"
    faq_root = docs_root / "faq"
    _write_fixture(faq_root)
    api_source = docs_root / "api" / "index.rst"
    api_source.parent.mkdir(parents=True)
    api_source.write_text("Box\n===\n", encoding="utf-8")
    faq_file = faq_root / "_faqs" / "example.md"
    faq_file.write_text(
        faq_file.read_text(encoding="utf-8")
        + "\n[API](https://docs.flexcompute.com/projects/tidy3d/en/latest/api/index.html)\n",
        encoding="utf-8",
    )

    generate_faq_docs.load_and_validate(faq_root)


def test_generation_preserves_html_block_boundaries(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    faq_file = tmp_path / "_faqs" / "example.md"
    faq_file.write_text(
        faq_file.read_text(encoding="utf-8")
        + "\n<div>See a variety of</div><div>examples here.</div>\n",
        encoding="utf-8",
    )

    output_root = tmp_path / "docs"
    generate_faq_docs.generate(tmp_path, output_root)

    rendered = (output_root / "faq" / "example.md").read_text(encoding="utf-8")
    assert "variety of\n\nexamples" in rendered
    assert "variety ofexamples" not in rendered


def test_generation_drops_an_initial_source_heading(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    faq_file = tmp_path / "_faqs" / "example.md"
    faq_file.write_text(
        faq_file.read_text(encoding="utf-8").replace(
            "---\n![Example]",
            "---\n# Example\n\n![Example]",
        ),
        encoding="utf-8",
    )

    output_root = tmp_path / "docs"
    generate_faq_docs.generate(tmp_path, output_root)

    rendered = (output_root / "faq" / "example.md").read_text(encoding="utf-8")
    assert rendered.count("# Example\n") == 1


def test_generation_preserves_canonical_page_on_case_insensitive_filesystem(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root = tmp_path / "source"
    output_root = tmp_path / "published"
    _write_fixture(source_root)
    (source_root / generate_faq_docs.ALIAS_MANIFEST).write_text(
        json.dumps({"Example.md": "_faqs/example.md"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        generate_faq_docs,
        "_aliases_canonical_file",
        lambda alias_path, canonical_path: (
            alias_path.name.casefold() == canonical_path.name.casefold()
        ),
    )

    generate_faq_docs.generate(source_root, output_root)

    canonical = (output_root / "faq" / "example.md").read_text(encoding="utf-8")
    assert canonical.startswith("# Example\n")
    assert "FAQ page moved" not in canonical
    aliases = json.loads((output_root / "aliases.json").read_text(encoding="utf-8"))
    assert aliases == {"faq/Example.html": "faq/example.html"}


def test_validation_checks_disabled_faq_metadata(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    (tmp_path / "_faqs" / "disabled.md").write_text(
        "---\ndate: 2026-07-29\nenabled: false\ncategory: General\n---\n",
        encoding="utf-8",
    )

    with pytest.raises(generate_faq_docs.FaqValidationError, match="field 'title'"):
        generate_faq_docs.load_and_validate(tmp_path)


def test_validation_rejects_legacy_cms_front_matter(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    faq_file = tmp_path / "_faqs" / "example.md"
    faq_file.write_text(
        faq_file.read_text(encoding="utf-8").replace(
            "title: Example\n",
            "_schema: default\ntitle: Example\n_inputs:\n  title:\n    type: text\n",
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        generate_faq_docs.FaqValidationError,
        match=r"unsupported front matter fields: _inputs, _schema",
    ):
        generate_faq_docs.load_and_validate(tmp_path)


def test_validation_rejects_non_scalar_version_metadata(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    faq_file = tmp_path / "_faqs" / "example.md"
    faq_file.write_text(
        faq_file.read_text(encoding="utf-8").replace(
            "enabled: true\n", "enabled: true\nversion: [2, 10]\n"
        ),
        encoding="utf-8",
    )

    with pytest.raises(generate_faq_docs.FaqValidationError, match=r"version.*scalar"):
        generate_faq_docs.load_and_validate(tmp_path)


def test_validation_rejects_an_unhydrated_lfs_image(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    (tmp_path / "_faqs" / "img" / "example.png").write_bytes(
        generate_faq_docs.LFS_POINTER_PREFIX
        + b"oid sha256:0000000000000000000000000000000000000000000000000000000000000000\n"
        + b"size 1\n"
    )

    with pytest.raises(generate_faq_docs.FaqValidationError, match="Git LFS pointer"):
        generate_faq_docs.load_and_validate(tmp_path)


def test_real_sphinx_build_hydrates_and_generates_faqs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source_docs_root = Path(__file__).parent
    docs_root = tmp_path / "docs"
    docs_root.mkdir()
    shutil.copy2(source_docs_root / "conf.py", docs_root / "conf.py")
    shutil.copy2(
        source_docs_root / "generate_faq_docs.py",
        docs_root / "generate_faq_docs.py",
    )
    (tmp_path / "CHANGELOG.md").write_text("Included changelog.\n", encoding="utf-8")
    plugin_readme = tmp_path / "tidy3d" / "plugins" / "example" / "README.md"
    plugin_readme.parent.mkdir(parents=True)
    plugin_readme.write_text(
        "Included plugin documentation.\n\n.. image:: ../../../notebooks/img/plugin.png\n",
        encoding="utf-8",
    )
    plugin_image = tmp_path / "notebooks" / "img" / "plugin.png"
    plugin_image.parent.mkdir(parents=True)
    plugin_image.write_bytes(b"\x89PNG\r\n\x1a\n")
    (docs_root / "changelog.rst").write_text(
        "Changelog\n=========\n\n.. include:: ../CHANGELOG.md\n",
        encoding="utf-8",
    )
    plugin_docs = docs_root / "api" / "plugins"
    plugin_docs.mkdir(parents=True)
    (plugin_docs / "example.rst").write_text(
        "Example plugin\n==============\n\n"
        ".. include:: ../../../tidy3d/plugins/example/README.md\n",
        encoding="utf-8",
    )
    (docs_root / "index.rst").write_text(
        "Test documentation\n"
        "==================\n\n"
        ".. toctree::\n"
        "   :maxdepth: 1\n\n"
        "   changelog\n"
        "   api/plugins/example\n"
        "   faq/docs/index\n",
        encoding="utf-8",
    )
    _write_fixture(docs_root / "faq")
    faq_file = docs_root / "faq" / "_faqs" / "example.md"
    faq_file.write_text(
        faq_file.read_text(encoding="utf-8").replace(
            "![Example](./img/example.png)",
            '<div><img width="640" src="./img/example.png" /></div>',
        ),
        encoding="utf-8",
    )
    image = docs_root / "faq" / "_faqs" / "img" / "example.png"
    image.write_bytes(
        generate_faq_docs.LFS_POINTER_PREFIX
        + b"oid sha256:0000000000000000000000000000000000000000000000000000000000000000\n"
        + b"size 1\n"
    )
    commands = []

    def hydrate_image(command, *, cwd, check):
        commands.append((command, Path(cwd), check))
        image.write_bytes(b"\x89PNG\r\n\x1a\n")
        return subprocess.CompletedProcess(command, 0)

    fake_tidy3d = ModuleType("tidy3d")
    fake_tidy3d.__file__ = str(tmp_path / "tidy3d" / "__init__.py")
    fake_tidy3d.__version__ = "test"
    fake_docs_version = ModuleType("docs_version")
    fake_docs_version.is_public_docs_version_tag = lambda value: False
    fake_docs_version.normalize_docs_version = lambda value: value
    monkeypatch.setattr(subprocess, "run", hydrate_image)
    monkeypatch.setitem(sys.modules, "tidy3d", fake_tidy3d)
    monkeypatch.setitem(sys.modules, "docs_version", fake_docs_version)
    monkeypatch.delitem(sys.modules, "generate_faq_docs")
    monkeypatch.setenv("TIDY3D_DOCS_VERSION", "latest")

    try:
        app = Sphinx(
            srcdir=str(docs_root),
            confdir=str(docs_root),
            outdir=str(tmp_path / "output"),
            doctreedir=str(tmp_path / "doctrees"),
            buildername="html",
            confoverrides={
                "extensions": ["myst_parser", "sphinx.ext.autodoc"],
                "source_suffix": {
                    ".rst": "restructuredtext",
                    ".md": "markdown",
                },
                "exclude_patterns": [],
                "html_extra_path": [],
                "html_static_path": [],
                "html_theme": "basic",
            },
            status=io.StringIO(),
            warning=io.StringIO(),
            freshenv=True,
        )
        staged_docs_root = Path(app.srcdir)
        assert isinstance(app._tidy3d_staged_sources, Path)
        app.build(force_all=True)
    finally:
        while str(docs_root) in sys.path:
            sys.path.remove(str(docs_root))

    assert commands == [
        (
            [
                "git",
                "lfs",
                "pull",
                "--include=**/docs/faq/_faqs/img/**",
            ],
            tmp_path,
            True,
        )
    ]
    assert app.statuscode == 0
    assert "faq/docs/faq/example" in app.env.found_docs
    assert "Included changelog." in (tmp_path / "output" / "changelog.html").read_text(
        encoding="utf-8"
    )
    assert "Included plugin documentation." in (
        tmp_path / "output" / "api" / "plugins" / "example.html"
    ).read_text(encoding="utf-8")
    assert (tmp_path / "output" / "_images" / "plugin.png").is_file()
    assert (tmp_path / "output" / "_images" / "example.png").is_file()
    faq_html = (tmp_path / "output" / "faq" / "docs" / "faq" / "example.html").read_text(
        encoding="utf-8"
    )
    assert "./img/example.png" not in faq_html
    assert not (docs_root / "faq" / "docs").exists()
    assert not staged_docs_root.exists()
    assert not hasattr(app, "_tidy3d_staged_sources")
    expected_redirects = {
        "tidy3d.web.api.asynchronous.run_async.html": "tidy3d.web.run_async.html",
        "tidy3d.web.api.container.Batch.html": "tidy3d.web.Batch.html",
        "tidy3d.web.api.container.BatchData.html": "tidy3d.web.BatchData.html",
        "tidy3d.web.api.container.Job.html": "tidy3d.web.Job.html",
    }
    redirect_root = tmp_path / "output" / "api" / "_autosummary"
    for legacy_name, canonical_name in expected_redirects.items():
        redirect = (redirect_root / legacy_name).read_text(encoding="utf-8")
        assert f'content="0; url={canonical_name}"' in redirect
        assert f'href="{canonical_name}"' in redirect


def test_validation_rejects_an_unreferenced_image_symlink(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"private")
    (tmp_path / "_faqs" / "img" / "leak.png").symlink_to(outside)

    with pytest.raises(generate_faq_docs.FaqValidationError, match="may not be a symlink"):
        generate_faq_docs.load_and_validate(tmp_path)


def test_validation_rejects_unused_media(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    (tmp_path / "_faqs" / "img" / "unused.png").write_bytes(b"unused")

    with pytest.raises(generate_faq_docs.FaqValidationError, match="unused FAQ media"):
        generate_faq_docs.load_and_validate(tmp_path)


def test_generation_rejects_an_output_inside_raw_sources(tmp_path: Path) -> None:
    _write_fixture(tmp_path)

    with pytest.raises(ValueError, match="default generated directory"):
        generate_faq_docs.generate(tmp_path, tmp_path / "_faqs" / "generated")


def test_generation_preserves_an_unmarked_output_directory(tmp_path: Path) -> None:
    _write_fixture(tmp_path / "source")
    output_root = tmp_path / "existing"
    output_root.mkdir()
    sentinel = output_root / "keep.txt"
    sentinel.write_text("keep\n", encoding="utf-8")

    with pytest.raises(ValueError, match="without the FAQ generator marker"):
        generate_faq_docs.generate(tmp_path / "source", output_root)

    assert sentinel.read_text(encoding="utf-8") == "keep\n"


def test_generation_restores_previous_output_after_install_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root = tmp_path / "source"
    output_root = tmp_path / "published"
    _write_fixture(source_root)
    generate_faq_docs.generate(source_root, output_root)
    sentinel = output_root / "keep.txt"
    sentinel.write_text("previous\n", encoding="utf-8")
    real_replace = Path.replace

    def fail_new_tree_install(path: Path, target: Path) -> Path:
        if (
            path.parent == output_root.parent
            and path.name.startswith(f".{output_root.name}-")
            and not path.name.startswith(f".{output_root.name}-previous-")
            and Path(target) == output_root
        ):
            raise OSError("simulated install failure")
        return real_replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_new_tree_install)

    with pytest.raises(OSError, match="simulated install failure"):
        generate_faq_docs.generate(source_root, output_root)

    assert sentinel.read_text(encoding="utf-8") == "previous\n"
