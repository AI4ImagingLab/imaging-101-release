from pathlib import Path
import numpy as np
from src.preprocessing import load_observations,GROUPS,PIXEL_WIDTH_UM
from src.physics_model import TOP,BOTTOM,HILL_SLOPE,LOG_IC50,EPS

ROOT=Path(__file__).resolve().parents[2]


def test_public_calibration_and_physical_metadata_agree():
    ratio,low,high=load_observations(ROOT/'data/raw_data.npz')
    assert len(GROUPS)==43 and [GROUPS.count(x) for x in ['control','DCCD','FC','IAA']]==[12,12,8,11]
    assert ratio.dtype==low.dtype==high.dtype==np.float32
    for array in [ratio,low,high]:assert not np.isinf(array).any() and np.isnan(array).any()
    text=(ROOT/'README.md').read_text(encoding='utf-8')
    assert '0.18037518037518038' in text and PIXEL_WIDTH_UM==0.18037518037518038
    assert '1.1171,-0.0295,1.2133,6.2110' in text and (TOP,BOTTOM,HILL_SLOPE,LOG_IC50)==(1.1171,-.0295,1.2133,6.2110)
    assert 'epsilon is 0.0001' in text and EPS==.0001


def test_main_import_does_not_change_backend():
    import matplotlib
    before=matplotlib.get_backend()
    import src.main
    assert matplotlib.get_backend()==before
