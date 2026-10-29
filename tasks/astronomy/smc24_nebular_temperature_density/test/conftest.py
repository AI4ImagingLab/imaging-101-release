"""Keep contract-test temporary files in the task-owned output directory."""
from pathlib import Path
import tempfile
import pytest

@pytest.fixture
def tmp_path():
    output=Path(__file__).resolve().parents[1]/'output'
    output.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='contract-',dir=output) as folder:
        yield Path(folder)
