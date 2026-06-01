from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from docs_version import normalize_docs_version


def test_normalize_docs_version_accepts_public_release_tags():
    assert normalize_docs_version("v2.9.0") == "v2.9.0"
    assert normalize_docs_version("refs/tags/v2.9.0rc1") == "v2.9.0rc1"


def test_normalize_docs_version_maps_branch_style_values_to_public_slugs():
    assert normalize_docs_version("develop") == "stable"
    assert normalize_docs_version("refs/heads/develop") == "stable"
    assert normalize_docs_version("origin/main") == "latest"
    assert normalize_docs_version("feature/docs-preview") == "latest"
