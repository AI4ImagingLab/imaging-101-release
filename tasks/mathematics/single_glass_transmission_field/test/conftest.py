from pathlib import Path
import tempfile
import pytest
@pytest.fixture
def tmp_path():
    with tempfile.TemporaryDirectory(prefix='imaging101_reflection_') as d:yield Path(d)
