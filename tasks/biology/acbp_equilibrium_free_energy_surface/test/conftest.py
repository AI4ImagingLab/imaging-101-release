from pathlib import Path
import sys,importlib.util,pytest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
@pytest.fixture(scope='session')
def root():return ROOT
@pytest.fixture(scope='session')
def judge():
 spec=importlib.util.spec_from_file_location('standalone_judge',ROOT/'evaluation/judge.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
@pytest.fixture(scope='session')
def fresh(tmp_path_factory):
 from src.main import run
 p=tmp_path_factory.mktemp('fresh')/'prediction.npz';return p,run(output_path=p)
