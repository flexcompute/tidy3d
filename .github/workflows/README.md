# GitHub Actions Workflows

This directory houses the manual CI/CD workflows that drive the Tidy3D Python client release, test, and maintenance automation.

## Release Workflows

All release workflows in this repository now rely on `workflow_dispatch` (manual) events or explicit `workflow_call`s. Nothing runs automatically after merging to a branch, which prevents unintentional releases.

### `tidy3d-python-client-release.yml`

The orchestrator for the entire release pipeline. It sequences:

1. **Scope detection** (`determine-workflow-scope`) – figures out which stages need to run, how `release_type` should map to deployments, and if submodule tests must be enforced.
2. **Tagging** – delegates to `tidy3d-python-client-create-tag.yml` when tagging is enabled.
3. **Testing** – reuses `tidy3d-python-client-tests.yml` with knobs for local, remote, CLI, and submodule suites. The workflow consumes the `workflow_success` output from the tests job before proceeding.
4. **GitHub release** – creates a GitHub release when the deployment stage is active.
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
  - `extras_integration_tests` (runs with `test_type='full'` covering 10 platform/Python combinations)

When invoked via `workflow_call`, two optional overrides are also honored:
- `deploy_testpypi`
- `deploy_pypi`

If those overrides are omitted, deployment targets are inferred from `release_type`:

| `release_type` | Automatic deployments | Notes |
| --- | --- | --- |
| `draft` | none | Runs tagging/tests but does not publish packages. |
| `testpypi` | TestPyPI | Requires version parity with `pyproject.toml`. Good for validating artifacts. |
| `pypi` | TestPyPI + PyPI | Enforces semver tag format and auto-runs submodule tests when the tag is non-RC. |

**Testing stage**
- Uses the unified `tidy3d-python-client-tests.yml` workflow instead of the retired release-specific test workflow.
- Automatically passes `release_tag` so tests run against the tagged commit.
- `compile-tests-results` blocks deployment until every requested suite reports success through the tests workflow’s `workflow_success` output.

**Deployment stage**
- Creates a GitHub release when the PyPI deployment path succeeds.
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
- **Local tests**: Self-hosted Slurm runners on Python 3.10 and 3.13 (coverage enforced, diff-coverage comments for 3.13).
- **Remote tests**: GitHub-hosted matrix across Windows, Linux, and macOS for Python 3.10–3.13.
- **Optional suites**: CLI tests, version consistency checks, submodule validation (non-RC release tags only), and `tidy3d-extras` integration tests can be toggled via inputs.
- **Extras integration tests**: When enabled on merge_group, runs basic smoke tests (4 configurations). When called from release workflow, runs full tests (10 configurations covering all architectures and Python 3.10/3.13).
- **Test type control**: `test_type` input ("basic" or "full") can override automatic selection for extras integration tests.
- **Test selection control**: `test_selection` input (`testmon` or `full`) controls whether local/remote suites run with pytest-testmon (`--testmon --testmon-forceselect`) or full (`--no-testmon`) execution.
- **Default policy**: PR and manual runs default to `test_selection: testmon`; merge queue (`merge_group`) forces `full` for safety.
- **Testmon cache strategy**: local/remote testmon caches are shared by runner + Python and anchored to the default branch (`develop`) SHA, with dependency hash in the primary key. Restore keys degrade from exact branch+dependency to broader runner+Python prefixes. PR runs restore from shared caches and do not write new entries.
- **Cache refresh path**: successful merge queue runs (`merge_group`) execute full coverage in testmon collection mode (`--testmon-noselect`) and write refreshed shared caches; no additional post-merge push run is required.
- **Core cache telemetry**: local and remote jobs emit lightweight telemetry-only steps for cache outcome (`telemetry-cache-exact-hit`, `telemetry-cache-fallback-hit`, `telemetry-cache-miss`) and selection mode (`telemetry-selection-*`) so CI analytics can be derived from the jobs API without log scraping.
- **Dynamic scope**: Determines which jobs to run based on the event (draft PRs, approvals, merge queue, manual overrides).
- **Outputs**: `workflow_success` summarizes whether every required job succeeded; the release workflow uses this to decide if deployment can continue.

Manual full-suite safety run: trigger `tidy3d-python-client-tests.yml` with `workflow_dispatch` and set `test_selection=full`.

> The previous `tidy3d-python-client-release-tests.yml` workflow has been removed. Release-specific suites now live entirely inside this unified workflow.

### `tidy3d-python-client-develop-cli.yml`

Reusable workflow that runs the develop-CLI integration tests. It is usually invoked by the main tests workflow when `cli_tests` is requested but can also be triggered directly.

### `tidy3d-extras-python-client-tests-integration.yml`

Dedicated integration test workflow for `tidy3d-extras` package. Tests the optional extras functionality with two test modes: basic (smoke tests) and full (comprehensive coverage).

**Test Modes:**
- **Basic tests** (default for PRs): 4 test configurations for fast smoke testing
  - Linux x86_64 (`ubuntu-latest`) - Python 3.10
  - macOS arm64 (`macos-latest`) - Python 3.10
  - Windows x64 (`windows-latest`) - Python 3.10
  - Windows x64 (`windows-latest`) - Python 3.13
- **Full tests** (default for releases): 10 test configurations covering all architectures and Python versions
  - Linux x86_64 (`ubuntu-latest`) - Python 3.10, 3.13
  - Linux aarch64 (`linux-arm64`) - Python 3.10, 3.13
  - macOS x86_64 (`macos-15-intel`) - Python 3.10, 3.13
  - macOS arm64 (`macos-latest`) - Python 3.10, 3.13
  - Windows x64 (`windows-latest`) - Python 3.10, 3.13

**Key Features:**
- **Test type control**: `test_type` input ("basic" or "full") determines scope
- **Automatic selection**: Basic for merge_group (PR merges), full for release workflow
- **Architecture coverage**: Full mode tests all runner architectures where wheels are built (x86_64, aarch64, arm64)
- **Python version coverage**: Full mode tests both minimum supported (3.10) and latest (3.13) Python versions
- **AWS CodeArtifact integration**: Authenticates with CodeArtifact to access private dependencies
- **Comprehensive test coverage**: Includes doctests, extras license verification, and full test suite with coverage reporting
- **Release tag support**: Can test against a specific release tag via the `release_tag` input
- **Invocation**: Called from `tidy3d-python-client-tests.yml` when `extras_integration_tests` is enabled, or run manually via `workflow_dispatch`
- **Outputs**: `workflow_success` indicates whether all integration tests passed

The workflow ensures that the `tidy3d-extras` package installs and functions correctly across all supported platforms and architectures before releases.

## Maintenance Workflows

### `tidy3d-python-client-daily.yml`

Scheduled at 05:00 UTC and also manually runnable. It fans out to:
- `tidy3d-python-client-update-lockfile.yml` – keeps dependencies fresh.
- `tidy3d-python-client-release.yml` – runs a daily draft release (`daily-0.0.0`) with client and CLI tests enabled to catch breaking changes early. This validates that the package can be built and tested against the latest develop branch without actually publishing artifacts.
  - The release workflow explicitly calls the tests workflow with `test_selection: full`, so daily release validation keeps full non-testmon coverage.

### `tidy3d-python-client-update-lockfile.yml`

Manual or called workflow that updates `poetry.lock`, authenticates against AWS CodeArtifact, and opens a PR with the refreshed lockfile. Requires `AWS_CODEARTIFACT_ACCESS_KEY` and `AWS_CODEARTIFACT_ACCESS_SECRET`.

**Key inputs:**
- `source_branch` – branch to checkout and update lockfile for (defaults to `develop`). Useful for updating lockfiles on feature branches or release branches.
- `run_workflow` – boolean to enable/disable the workflow execution.

The workflow creates a PR with branch name `chore/update-poetry-lock-{source_branch}` targeting the specified source branch.

### `tidy3d-python-client-build-changelog-pr.yml`

Manual workflow that builds `CHANGELOG.md` from Towncrier fragments and opens a PR.

**Key inputs:**
- `source_branch` – branch to checkout and build changelog from (defaults to `develop`).
- `target_branch` – branch to open the PR against (defaults to `develop`).
- `release_version` – optional override for the release version. If omitted, it is derived from `pyproject.toml` by stripping `.devN`.
- `release_date` – optional override in `YYYY-MM-DD`. If omitted, UTC `today` is used.
- `previous_version` – optional override for the compare-link previous version. If omitted, the workflow uses the latest reachable stable `vX.Y.Z` tag, and falls back to the latest stable heading in `CHANGELOG.md` when no tag is available.
- `run_workflow` – boolean guard to enable/disable execution.

The workflow:
1. Installs Poetry dependencies (`--extras dev`).
2. Runs `towncrier build --yes`.
3. Runs `scripts/changelog_refs.py` to update compare reference links.
4. Opens a PR with the generated changelog updates.

If no fragments are present in `changelog.d/`, the workflow exits without opening a PR.

## Best Practices

### For releases

1. **Dry-run first** – kick off the release workflow with `release_type: draft` to verify tagging and tests without publishing packages.
2. **Use `testpypi` before `pypi`** – it enforces version parity and helps catch packaging issues before production uploads.
3. **Respect semver tags** – `release_type: pypi` will fail early if the tag is not `v{major}.{minor}.{patch}[rc{num}]`.
4. **Leverage `workflow_control`** – resume from `start-tests` or `start-deploy` instead of repeating earlier successful stages.
5. **Watch `workflow-validation`** – that job in the tests workflow aggregates CLI and test failures.
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
    Expected format: v{major}.{minor}.{patch}[rc{num}|.dev{num}]
  ```
  Use `v2.10.0`, `v2.10.1rc1`, `v2.11.0.dev0`, etc.

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

Use these outputs when chaining workflows or when external automation needs to know whether a stage succeeded.

## AWS CodeArtifact Integration

Private dependencies are sourced through AWS CodeArtifact:
- Configured inside `tidy3d-python-client-update-lockfile.yml`.
- Credentials come from `AWS_CODEARTIFACT_ACCESS_KEY` and `AWS_CODEARTIFACT_ACCESS_SECRET`.
- The workflow injects a temporary auth token into Poetry before running `poetry update --lock`.

## Related documentation

- Release workflow details: `docs/development/release/version.rst`
- Development guidelines: `AGENTS.md`
- Docker development environment: `docs/development/docker.rst` – comprehensive guide for setting up and using the Docker-based development environment
- General repository info: `README.md`
