from pathlib import Path
import numpy as np
from src.visualization import agreement
from src.physics_model import RADIUS,POISSON
from src.preprocessing import load_data

def test_metrics_and_constants():
    m=agreement([0,6,8],[0,3,4]);assert m['ncc']==1
    np.testing.assert_allclose(m['nrmse'],np.sqrt(25/3)/4)
    root=Path(__file__).resolve().parents[2];text=(root/'README.md').read_text(encoding='utf8')
    assert '18.64e-6' in text and RADIUS==18.64e-6
    assert '| Poisson ratio nu | 0.5 |' in text and POISSON==.5
    data=load_data(root/'data/raw_data.npz')
    assert data['offsets'].shape==(455,) and data['offsets'][-1]==6355919
    assert data['force'].dtype==np.float64
    assert np.all(np.diff(data['offsets'])>0)
    assert np.all(np.isin(data['segment'],[0,1]))
