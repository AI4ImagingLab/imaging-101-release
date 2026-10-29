from pathlib import Path
import sys,pytest,numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

@pytest.fixture(scope='session')
def task_root():return ROOT

@pytest.fixture(scope='session')
def fresh():
    from src.preprocessing import load_observations
    from src.solvers import reconstruct
    image,airlight=load_observations(ROOT/'data/raw_data.npz')
    rgb,t,stages=reconstruct(image,airlight)
    return image,airlight,rgb,t,stages
