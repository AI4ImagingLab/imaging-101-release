from pathlib import Path
import sys,pytest,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
@pytest.fixture(scope='session')
def root():return ROOT
@pytest.fixture(scope='session')
def oracle():
    with np.load(ROOT/'test/fixtures/upstream_states.npz',allow_pickle=False) as f:return {k:f[k] for k in f.files}
