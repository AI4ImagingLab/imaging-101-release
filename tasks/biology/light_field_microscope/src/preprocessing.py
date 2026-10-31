"""
Preprocessing Helpers for the Light-Field Microscope Task.

The current benchmark uses a compact metadata file plus saved reference
artifacts. This module intentionally stays small: it only exposes the metadata
loader needed by `main.py`, tests, and the notebook.
"""

from __future__ import annotations

import json
from pathlib import Path


def load_metadata(path: str | Path = "data/meta_data.json") -> dict:
    """Load the task metadata JSON file."""
    metadata_path = Path(path)
    with metadata_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
