from __future__ import annotations

import re

_PUBLIC_DOCS_VERSION_TAG_RE = re.compile(r"^v(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(rc[0-9]+)?$")


def _strip_docs_version_ref(version_value: str) -> str:
    version_value = version_value.strip()
    for prefix in ("refs/tags/", "refs/heads/"):
        if version_value.startswith(prefix):
            return version_value[len(prefix) :]
    if version_value.startswith("refs/remotes/"):
        remote_ref = version_value[len("refs/remotes/") :]
        if "/" in remote_ref:
            return remote_ref.split("/", 1)[1]
        return remote_ref
    for prefix in ("origin/", "upstream/"):
        if version_value.startswith(prefix):
            return version_value[len(prefix) :]
    return version_value


def is_public_docs_version_tag(version_value: str) -> bool:
    return bool(_PUBLIC_DOCS_VERSION_TAG_RE.match(version_value))


def normalize_docs_version(version_value: str) -> str:
    version_value = _strip_docs_version_ref(version_value)
    if is_public_docs_version_tag(version_value):
        return version_value
    version_alias = version_value.lower()
    if version_alias in {"develop", "stable"}:
        return "stable"
    if version_alias in {"latest", "main", "master"}:
        return "latest"
    return "latest"
