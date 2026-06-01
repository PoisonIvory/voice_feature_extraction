"""Provenance helpers for snapshot manifest metadata."""

from __future__ import annotations

import platform
import subprocess
from pathlib import Path
from typing import Any


def detect_source_commit(repo_dir: Path) -> str | None:
    """Return short git commit hash when available."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_dir,
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        return None
    value = result.stdout.strip()
    return value or None


def build_provenance(
    *,
    pipeline_version: str | None,
    opensmile_version: str | None,
    source_commit: str | None = None,
    python_version: str | None = None,
    repo_dir: Path | None = None,
) -> dict[str, Any]:
    """Build provenance payload for manifest metadata."""
    resolved_source_commit = source_commit
    if resolved_source_commit is None and repo_dir is not None:
        resolved_source_commit = detect_source_commit(repo_dir)

    return {
        "source_commit": resolved_source_commit,
        "pipeline_version": pipeline_version,
        "opensmile_version": opensmile_version,
        "python_version": python_version or platform.python_version(),
    }

