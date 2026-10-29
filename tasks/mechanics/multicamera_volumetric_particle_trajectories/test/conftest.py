from pathlib import Path
import sys,importlib.util
import pytest,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
@pytest.fixture(scope='session')
def data():
    from src.preprocessing import load_data
    return load_data(ROOT/'data/raw_data.npz')
@pytest.fixture(scope='session')
def stages():
    with np.load(ROOT/'test/fixtures/qualified_upstream_stages.npz',allow_pickle=False)as z:return {k:z[k] for k in z.files}
@pytest.fixture(scope='session')
def judge():
    spec=importlib.util.spec_from_file_location('standalone_judge',ROOT/'evaluation/judge.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
