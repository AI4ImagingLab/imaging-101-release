import sys
import tempfile,pytest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

@pytest.fixture
def tmp_path():
    with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent,prefix='contract-') as directory:yield Path(directory)
