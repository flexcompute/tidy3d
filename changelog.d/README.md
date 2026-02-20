# Changelog Fragments

Use Towncrier fragments for user-visible changes instead of editing `CHANGELOG.md` directly.
`CHANGELOG.md` is release-generated; regular PRs should add fragment files in this directory.
CI rejects direct `CHANGELOG.md` edits in regular PRs (auto-generated `chore/build-changelog-*` branches are exempt).

## File naming

Create one file per change using:

- Single entry for a PR/type: `<PR_NUMBER>.<type>.md`
- Multiple entries for the same PR/type: `<PR_NUMBER>.<N>.<type>.md` where `N` starts at `1`

Examples:

- `1234.added.md`
- `1235.1.added.md`
- `1235.2.added.md`
- `1236.changed.md`
- `1237.fixed.md`
- `1238.removed.md`
- `1239.breaking.md`
- `1240.planned_deprecation.md`

## Allowed fragment types

- `added`
- `changed`
- `fixed`
- `removed`
- `breaking`
- `planned_deprecation`

## Content format

- Write plain text only (no leading `-` bullet).
- Keep it to one short sentence per fragment file.
- Focus on the user-visible impact.

Examples:

- `added`: `Added GeometryArray for efficiently representing repeated geometry instances.`
- `changed`: `Improved local cache performance for repeated result loads.`
- `fixed`: `Fixed race conditions when reading the local configuration directory in parallel jobs.`
- `removed`: `Removed the deprecated legacy material alias from the public API.`
- `breaking`: `ModeSortSpec.sort_key is now required; update any code relying on None defaults.`
- `planned_deprecation`: `CurrentIntegralAxisAligned is deprecated and will be removed in a future release; use AxisAlignedCurrentIntegral.`
