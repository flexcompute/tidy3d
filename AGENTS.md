# Repository Guidelines

## Project Structure & Documentation
- See the root `README.md` → “Project Overview” for a map of directories, and consult subdirectory READMEs for local nuances; follow those layouts when adding code.
- Keep all README files, `docs/`, and this AGENTS.md in sync whenever tooling or behavior changes.
- Prefer extending existing utilities in `scripts/` before adding new helper modules.

## Workflow & Tooling
- Make sure your local environment is bootstrapped with `poetry install --extras dev` and `poetry run pre-commit install`; once that’s done on a machine, you shouldn’t need to repeat it unless dependencies change.
- Prefix every repo command with `poetry run` to match CI.
- Re-run `poetry run pytest` locally as part of your development loop; `pyproject.toml` already wires markers, doctests, coverage, and env vars.
- The pre-commit hooks you enabled during onboarding run automatically; still run `poetry run pre-commit run --all-files` before opening a PR or when new hooks land so your tree matches `.pre-commit-config.yaml` and the checks in `.github/workflows/tidy3d-python-client-tests.yml` (covers `ruff format`, `ruff check`, doc hooks).
- When editing YAML, Python, or docs, match the surrounding indentation exactly; never re-indent or reformat lines you didn’t otherwise modify.

### Do / Don't
- **Do** run `poetry run pre-commit run --all-files` before opening a PR; **don't** skip it even if individual hooks passed earlier.
- **Do** stick to `poetry run …` commands; **don't** invoke tools outside Poetry, since that drifts from CI environments.
- **Do** reuse `scripts/` utilities; **don't** add new helper modules without checking for an existing script first.

## Coding Style & Naming
- Prefer top-level imports.
- Favor descriptive `snake_case` functions and `PascalCase` classes tied to the physics domain.
- Most simulation and monitor classes subclass `Tidy3dBaseModel`; reserve `pydantic.BaseModel` for lightweight helpers (see `tidy3d/updater.py`), and rely on `.updated_copy(...)` plus shared validators in `tidy3d/components/validators.py`.
- Public APIs covered in `docs/` should use the existing Numpy-inspired block pattern rendered by our Sphinx Book Theme; lean on current API examples or the theme reference at https://sphinx-book-theme.readthedocs.io/en/stable/reference/api-numpy.html. Internal helpers may keep short docstrings but match local tone.
- Prefer `tidy3d.log.log` and `tidy3d.exceptions` for contributor-facing messages; touch stdlib `logging` only when muting external libraries such as `pyroots`.
- Reuse types from `tidy3d/components/types`, domain constants from `tidy3d/constants`, and runtime defaults from `tidy3d/config`.
- For the `Medium` and `Simulation` class families, centralize `@model_validator(mode="after")` logic in `_run_after_validators()` with a short docstring, and call dependent checks in explicit order instead of relying on decorator registration.
- For those same families, prefix validator helpers with `_` (e.g., `_check_*`, `_validate_*`) and use `call_wrapped_validator(...)` for validator factories so ordering stays explicit.

## Testing Guidelines
- Mirror the source tree with `test_<feature>.py`, add a short module docstring, and import `tidy3d as td`; keep single-use fixtures local but upstream broadly useful helpers into `tests/conftest.py`.
- Lean on `tests/conftest.py` for RNG seeding, matplotlib cleanup, logger reset, and autograd helpers, and use pytest’s `tmp_path` for artifacts.
- `poetry run pytest` is the canonical entry point; the `pyproject` config already selects markers, doctests, coverage, xdist, and env vars, so reproduce CI locally before opening a PR.
- Prefer pytest primitives (`pytest.raises`, `pytest.approx`, `@pytest.mark.parametrize`) for coverage, and add doctest snippets for new public APIs to keep docs and runtime behavior aligned.
- Mock all external/web APIs in tests (e.g., `web.run`, HTTP clients) and assert the contract instead of hitting live services; CI must stay hermetic.
- Numerical tests require explicit maintainer confirmation. They run real simulations and are excluded by default (`-m 'not numerical'`). When approved, run selectively via `poetry run pytest -m numerical path/to/test.py -k specific_case`.
- Scope tests by path: `poetry run pytest -q tests/test_web/` or `poetry run pytest -q tidy3d/components/geometry/`.
- Scope tests by keyword: `poetry run pytest -q -k "feature_name or module_name"`.
- Keep repo defaults (incl. xdist); for speed disable coverage and mute warnings: `poetry run pytest -q --no-cov -W ignore tests/... --maxfail=1`.
- Doctest a single module quickly: `poetry run pytest -q --no-cov -W ignore tidy3d/components/foo.py`.

## Schema Assets
- Files under `schemas/` are generated artifacts; never edit them manually.
- Run `poetry run python scripts/regenerate_schema.py` after model/serialization changes, and commit the output alongside the code.
- CI reruns the script and fails the PR whenever checked-in schemas drift from regenerated results.

## Commit & Pull Request Guidelines
- Follow Conventional Commits per `.commitlintrc.json`.
- Branch names must use an allowed prefix (`chore`, `hotfix`, `daily-chore`) or include a Jira key to satisfy CI.
- PRs should link issues, summarize behavior changes, list the `poetry run …` checks you executed, and call out docs/schema updates.
- For user-facing changes (new features, bug fixes, breaking changes), add a changelog fragment under `changelog.d/` using the pattern `<PR_NUMBER>.<type>.md` (for example `1234.added.md`) instead of editing `CHANGELOG.md` directly; CI rejects direct `CHANGELOG.md` edits on regular PR branches.
- Release managers can use the GitHub Actions workflow `public/tidy3d/python-client-build-changelog-pr` to generate `CHANGELOG.md` from fragments and open a PR (defaults source/target to `develop`).

_Reminder: update this AGENTS.md whenever workflow, tooling, or review expectations change so agents stay in sync with the repo._
