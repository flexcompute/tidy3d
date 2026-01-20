# GitHub Actions Workflows

This directory houses the manual CI/CD workflows that drive the Tidy3D Python client release, test, maintenance, and documentation automation.

## Deploy Workflow

### `public_tidy3d-python-client-deploy.yml`
- Manual deployment entry point (can also be called from the release orchestrator).
- Requires selecting at least one of `deploy_testpypi` or `deploy_pypi`.
- Builds the distribution with Poetry, uploads artifacts, and publishes via Twine.
- Emits a short deployment summary and fails if any requested target fails.

## Test Workflows

### `public_tidy3d-python-client-tests.yml`

Primary CI workflow; it runs on PRs (`latest`, `develop`, `pre/*`), merge queue (`merge_group`), manual dispatch, and `workflow_call`. Highlights:
- **Code quality**: `ruff format`, `ruff check`, `mypy`, `zizmor`, schema regeneration, commit/branch linting, and changelog policy enforcement (no direct `CHANGELOG.md` edits on regular PR branches).
- **Local tests**: Self-hosted Slurm runners on Python 3.10 and 3.13 (coverage enforced, diff-coverage comments for 3.13).
- **Remote tests**: GitHub-hosted matrix across Windows, Linux, and macOS for Python 3.10–3.13.
- **Optional suites**: CLI tests, version consistency checks, and `tidy3d-extras` integration tests can be toggled via inputs.
- **Extras integration tests**: When enabled on merge_group, runs basic smoke tests (4 configurations). When called from release workflow, runs full tests (10 configurations covering all architectures and Python 3.10/3.13).
- **Test type control**: `test_type` input ("basic" or "full") can override automatic selection for extras integration tests.
- **Test selection control**: `test_selection` input (`testmon` or `full`) controls whether local/remote suites run with pytest-testmon (`--testmon --testmon-forceselect`) or full (`--no-testmon`) execution.
- **Default policy**: PR and manual runs default to `test_selection: testmon`; merge queue (`merge_group`) forces `full` for safety.
- **Testmon cache strategy**: local/remote testmon caches are shared by runner + Python and anchored to the default branch (`develop`) SHA, with dependency hash in the primary key. Restore keys degrade from exact branch+dependency to broader runner+Python prefixes. PR runs restore from shared caches and do not write new entries.
- **Cache refresh path**: successful merge queue runs (`merge_group`) execute full coverage in testmon collection mode (`--testmon-noselect`) and write refreshed shared caches; no additional post-merge push run is required.
- **Core cache telemetry**: local and remote jobs emit lightweight telemetry-only steps for cache outcome (`telemetry-cache-exact-hit`, `telemetry-cache-fallback-hit`, `telemetry-cache-miss`) and selection mode (`telemetry-selection-*`) so CI analytics can be derived from the jobs API without log scraping.
- **Dynamic scope**: Determines which jobs to run based on the event (draft PRs, approvals, merge queue, manual overrides).
- **Outputs**: `workflow_success` summarizes whether every required job succeeded; the release workflow uses this to decide if deployment can continue.

Manual full-suite safety run: trigger `public_tidy3d-python-client-tests.yml` with `workflow_dispatch` and set `test_selection=full`.

### `public_tidy3d-python-client-develop-cli.yml`

Reusable workflow that runs the develop-CLI integration tests. It is usually invoked by the main tests workflow when `cli_tests` is requested but can also be triggered directly.

### `public_tidy3d-python-client-tests-extras-integration.yml`

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
- **Invocation**: Called from `public_tidy3d-python-client-tests.yml` when `extras_integration_tests` is enabled, or run manually via `workflow_dispatch`
- **Outputs**: `workflow_success` indicates whether all integration tests passed

The workflow ensures that the `tidy3d-extras` package installs and functions correctly across all supported platforms and architectures before releases.

## Maintenance Workflows

### `public_tidy3d-python-client-daily.yml`

Scheduled at 05:00 UTC and also manually runnable. It fans out to:
- `public_tidy3d-python-client-update-lockfile.yml` – keeps dependencies fresh.

### `public_tidy3d-python-client-update-lockfile.yml`

Manual or called workflow that updates `poetry.lock`, authenticates against AWS CodeArtifact, and opens a PR with the refreshed lockfile. Requires `AWS_CODEARTIFACT_ACCESS_KEY` and `AWS_CODEARTIFACT_ACCESS_SECRET`.

**Key inputs:**
- `source_branch` – branch to checkout and update lockfile for (defaults to `develop`). Useful for updating lockfiles on feature branches or release branches.
- `run_workflow` – boolean to enable/disable the workflow execution.

The workflow creates a PR with branch name `chore/update-poetry-lock-{source_branch}` targeting the specified source branch.

### `public_tidy3d-python-client-build-changelog-pr.yml`

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

- **Watch `workflow-validation`** – that job in the tests workflow aggregates lint, schema, CLI, and test failures.

### Troubleshooting

- **Tests blocking deployment**
  Inspect the `workflow-validation` job inside `public_tidy3d-python-client-tests`. After fixing the issue, rerun the tests.

- **Manual deployment run fails immediately**
  `public_tidy3d-python-client-deploy.yml` requires at least one of `deploy_testpypi` or `deploy_pypi` to be set to `true`; otherwise it aborts during input validation.

## Workflow outputs

- `public_tidy3d-python-client-tests.yml`: `workflow_success`

Use these outputs when chaining workflows or when external automation needs to know whether a stage succeeded.

## AWS CodeArtifact Integration

Private dependencies are sourced through AWS CodeArtifact:
- Configured inside `public_tidy3d-python-client-update-lockfile.yml`.
- Credentials come from `AWS_CODEARTIFACT_ACCESS_KEY` and `AWS_CODEARTIFACT_ACCESS_SECRET`.
- The workflow injects a temporary auth token into Poetry before running `poetry update --lock`.

## Related documentation

- Release workflow details: `docs/development/release/version.rst`
- Development guidelines: `AGENTS.md`
- Docker development environment: `docs/development/docker.rst` – comprehensive guide for setting up and using the Docker-based development environment
- General repository info: `README.md`
