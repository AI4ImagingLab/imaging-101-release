"""Use an owned temporary directory, avoiding inherited pytest-root ACLs."""
from pathlib import Path
import tempfile
import pytest
@pytest.fixture
def tmp_path():
    with tempfile.TemporaryDirectory(prefix='indentation-judge-') as directory:
        yield Path(directory)
