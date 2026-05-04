"""Shared project data paths used by the CLI and app."""

from __future__ import annotations

from pathlib import Path

PROJECT_DATA_DIR = Path("data")
RAW_DIR = PROJECT_DATA_DIR / "raw"
PROCESSED_DIR = PROJECT_DATA_DIR / "processed"
ARTIFACT_DIR = PROJECT_DATA_DIR / "artifacts"
EXPORT_DIR = PROJECT_DATA_DIR / "exports"


def ensure_data_dirs() -> None:
    """Create all runtime data directories if they do not already exist."""
    for path in [RAW_DIR, PROCESSED_DIR, ARTIFACT_DIR, EXPORT_DIR]:
        path.mkdir(parents=True, exist_ok=True)
