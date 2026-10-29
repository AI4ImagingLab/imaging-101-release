"""Isolated temporary files, without dependence on a stale global pytest root."""
from pathlib import Path
import tempfile
import pytest
@pytest.fixture
def tmp_path():
    with tempfile.TemporaryDirectory(prefix='imaging101_intrinsic_') as directory:
        yield Path(directory)
