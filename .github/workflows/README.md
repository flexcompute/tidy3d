# GitHub Actions Workflows

This directory houses the manual CI/CD workflows that drive the Tidy3D Python client release, test, maintenance, and documentation automation.

## Release Workflows

All release workflows in this repository now rely on `workflow_dispatch` (manual) events or explicit `workflow_call`s. Nothing runs automatically after merging to a branch, which prevents unintentional releases.

### `tidy3d-python-client-release.yml`

The orchestrator for the entire release pipeline. It sequences:

1. **Scope detection** (`determine-workflow-scope`) – figures out which stages need to run, how `release_type` should map to deployments, whether to push docs to `latest`, and if submodule tests must be enforced.
2. **Tagging** – delegates to `tidy3d-python-client-create-tag.yml` when tagging is enabled.
3. **Testing** – reuses `tidy3d-python-client-tests.yml` with knobs for local, remote, CLI, and submodule suites. The workflow consumes the `workflow_success` output from the tests job before proceeding.
4. **Docs sync & GitHub release** – mirrors the release tag to ReadTheDocs (`tidy3d-docs-sync-readthedocs-repo.yml`) and creates a GitHub release when the deployment stage is active.
5. **Package deployment** – invokes `tidy3d-python-client-deploy.yml` with the resolved TestPyPI/PyPI targets.

**Trigger**
- Manual run through the GitHub Actions UI (`workflow_dispatch` inputs).
- `workflow_call` so other workflows/scripts can orchestrate releases with pre-filled inputs.

**Key inputs**
- `release_tag` *(required)* – tag to create and test (e.g., `v2.10.0`, `v2.10.0rc1`).
- `release_type` – controls defaults for downstream deployment jobs. Options:
  - `draft`
  - `testpypi`
  - `pypi`
- `workflow_control` – allows resuming or skipping stages:
  - `start-tag` (default), `start-tests`, `start-deploy`
  - `only-tag`, `only-tests`, `only-tag-tests`, `only-tag-deploy`
- Test toggles:
  - `client_tests`
  - `cli_tests`
  - `submodule_tests` (auto-enabled for non-RC `pypi` releases even if left `false`)

When invoked via `workflow_call`, two optional overrides are also honored:
- `deploy_testpypi`
- `deploy_pypi`

If those overrides are omitted, deployment targets are inferred from `release_type`:

| `release_type` | Automatic deployments | Notes |
| --- | --- | --- |
| `draft` | none | Runs tagging/tests/sync but does not publish packages. |
| `testpypi` | TestPyPI | Requires version parity with `pyproject.toml`. Good for validating artifacts. |
| `pypi` | TestPyPI + PyPI | Enforces semver tag format, auto-runs submodule tests when the tag is non-RC, and mirrors docs to the `latest` ref. |

**Testing stage**
- Uses the unified `tidy3d-python-client-tests.yml` workflow instead of the retired release-specific test workflow.
- Automatically passes `release_tag` so tests run against the tagged commit.
- `compile-tests-results` blocks deployment until every requested suite reports success through the tests workflow’s `workflow_success` output.

**Deployment stage**
- Always creates a GitHub release and syncs docs when `run_deploy` is `true`.
- ReadTheDocs sync pushes the tag to the mirror repository and targets `latest` automatically for non-RC `pypi` releases (semver tags only).
- Package publication happens through `tidy3d-python-client-deploy.yml`; deployment targets can still be narrowed by re-running the orchestrator with `only-tag-deploy` or `start-deploy`.

**Outputs**
- `workflow_success` – `true` only when deployments (or the chosen stages) complete successfully. Use this signal when chaining workflows.

### Supporting release workflows

#### `tidy3d-python-client-create-tag.yml`
- Manual or called workflow that (re)creates tags.
- For `release_type: testpypi` and `release_type: pypi`, validates that `pyproject.toml` matches the tag (minus the `v` prefix) before pushing.
- Retags automatically by deleting the old tag locally and on origin.
- Output: `tag_created`.

#### `tidy3d-python-client-deploy.yml`
- Manual deployment entry point (can also be called from the release orchestrator).
- Requires selecting at least one of `deploy_testpypi` or `deploy_pypi`.
- Builds the distribution with Poetry, uploads artifacts, and publishes via Twine.
- Emits a short deployment summary and fails if any requested target fails.

## Test Workflows

### `tidy3d-python-client-tests.yml`

Primary CI workflow; it runs on PRs (`latest`, `develop`, `pre/*`), merge queue (`merge_group`), manual dispatch, and `workflow_call`. Highlights:
- **Code quality**: `ruff format`, `ruff check`, `mypy`, `zizmor`, schema regeneration, commit/branch linting.
- **Local tests**: Self-hosted Slurm runners on Python 3.10 and 3.13 (coverage enforced, diff-coverage comments for 3.13).
- **Remote tests**: GitHub-hosted matrix across Windows, Linux, and macOS for Python 3.10–3.13.
- **Optional suites**: CLI tests, version consistency checks, and submodule validation (non-RC release tags only) can be toggled via inputs.
- **Dynamic scope**: Determines which jobs to run based on the event (draft PRs, approvals, merge queue, manual overrides).
- **Outputs**: `workflow_success` summarizes whether every required job succeeded; the release workflow uses this to decide if deployment can continue.

> The previous `tidy3d-python-client-release-tests.yml` workflow has been removed. Release-specific suites now live entirely inside this unified workflow.

### `tidy3d-python-client-develop-cli.yml`

Reusable workflow that runs the develop-CLI integration tests. It is usually invoked by the main tests workflow when `cli_tests` is requested but can also be triggered directly.

## Maintenance Workflows

### `tidy3d-python-client-daily.yml`

Scheduled at 05:00 UTC and also manually runnable. It fans out to:
- `tidy3d-python-client-update-lockfile.yml` – keeps dependencies fresh.
- The submodule smoke-test workflow – ensures docs/notebooks submodules stay aligned (same helper the release tests call).

### `tidy3d-python-client-update-lockfile.yml`

Manual or called workflow that updates `poetry.lock`, authenticates against AWS CodeArtifact, and opens a PR on `develop` with the refreshed lockfile (`daily-chore/update-poetry-lock`). Requires `AWS_CODEARTIFACT_ACCESS_KEY` and `AWS_CODEARTIFACT_ACCESS_SECRET`.

## Documentation Workflows

### `tidy3d-docs-sync-readthedocs-repo.yml`

Mirrors a source ref (branch or tag) to the ReadTheDocs mirror repository.
- Inputs: `source_ref` (defaults to the triggering ref) and optional `target_ref`.
- Outputs: `workflow_success`, `synced_ref`.
- Used automatically by the release workflow and can also be run manually when docs need to be re-synced without a full release.

## Best Practices

### For releases

1. **Dry-run first** – kick off the release workflow with `release_type: draft` to verify tagging and tests without publishing packages.
2. **Use `testpypi` before `pypi`** – it enforces version parity and helps catch packaging issues before production uploads.
3. **Respect semver tags** – `release_type: pypi` will fail early if the tag is not `v{major}.{minor}.{patch}[rc{num}]`.
4. **Leverage `workflow_control`** – resume from `start-tests` or `start-deploy` instead of repeating earlier successful stages.
5. **Watch `workflow-validation`** – that job in the tests workflow aggregates lint, schema, CLI, and test failures.
6. **Let submodule tests run for stable releases** – they are auto-enabled for non-RC PyPI releases; only disable when you have a compelling reason.

### Version validation

For `release_type: testpypi` or `release_type: pypi`, the tagging workflow enforces version alignment:

```bash
# pyproject.toml must contain:
version = "2.10.0"

# And the release workflow must be invoked with:
release_tag: v2.10.0
```

Additionally, `release_type: pypi` enables strict tag-format validation inside the orchestrator before anything runs.

### Recommended release flow

1. **Draft dry run**

   ```yaml
   release_tag: v2.10.0rc1
   release_type: draft
   workflow_control: start-tag
   ```

2. **TestPyPI publishing (after the draft run passes)**

   ```yaml
   release_tag: v2.10.0rc1
   release_type: testpypi
   workflow_control: start-deploy  # reuse prior tag/tests
   ```

3. **Stable PyPI release**

   ```yaml
   release_tag: v2.10.0
   release_type: pypi
   workflow_control: start-deploy
   # Submodule tests run automatically when the tag is non-RC.
   ```

Re-running with `only-tag` or `only-tag-deploy` is helpful when you must recreate a tag or redo deployments without re-running every test.

### Troubleshooting

- **Version mismatch (`create-tag`)**
  ```
  Version mismatch!
   pyproject.toml: 2.9.0
   Release tag:    2.10.0
  ```
  Update `pyproject.toml` (and `tidy3d/version.py`) so the version matches `release_tag` minus the `v`.

- **Invalid tag format (`release_type: pypi`)**
  ```
  Invalid tag format: v2.10
    Expected format: v{major}.{minor}.{patch}[rc{num}]
  ```
  Use `v2.10.0`, `v2.10.1rc1`, etc.

- **Tag already exists**
  The tagging workflow deletes and recreates the tag automatically. No manual cleanup is needed.

- **Tests blocking deployment**
  Inspect the `workflow-validation` job inside `tidy3d-python-client-tests`. After fixing the issue, rerun the release workflow with `workflow_control: start-tests` or `start-deploy` as appropriate.

- **Manual deployment run fails immediately**
  `tidy3d-python-client-deploy.yml` requires at least one of `deploy_testpypi` or `deploy_pypi` to be set to `true`; otherwise it aborts during input validation.

## Workflow outputs

- `tidy3d-python-client-release.yml`: `workflow_success`
- `tidy3d-python-client-tests.yml`: `workflow_success`
- `tidy3d-python-client-create-tag.yml`: `tag_created`
- `tidy3d-docs-sync-readthedocs-repo.yml`: `workflow_success`, `synced_ref`

Use these outputs when chaining workflows or when external automation needs to know whether a stage succeeded.

## AWS CodeArtifact Integration

Private dependencies are sourced through AWS CodeArtifact:
- Configured inside `tidy3d-python-client-update-lockfile.yml`.
- Credentials come from `AWS_CODEARTIFACT_ACCESS_KEY` and `AWS_CODEARTIFACT_ACCESS_SECRET`.
- The workflow injects a temporary auth token into Poetry before running `poetry update --lock`.

## Related documentation

- Release workflow details: `docs/development/release/version.rst`
- Development guidelines: `AGENTS.md`
- General repository info: `README.md`
