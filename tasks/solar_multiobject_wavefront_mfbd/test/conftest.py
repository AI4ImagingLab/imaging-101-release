from pathlib import Path
import sys
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
torch.set_num_threads(1)


@pytest.fixture(scope='session')
def oracle():
    with np.load(ROOT / 'test/fixtures/upstream_stages.npz', allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


@pytest.fixture(scope='session')
def optics():
    from src.preprocessing import load_data
    from src.physics_model import build_optics
    _, mixing = load_data(ROOT / 'data/raw_data.npz')
    return build_optics(mixing)
