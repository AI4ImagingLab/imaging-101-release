from pathlib import Path
import numpy as np
import pytest
from src.physics_model import buxton_signal, forward_volume, LABEL_DURATION, POST_LABEL_DELAY
from src.preprocessing import load_observations
ROOT=Path(__file__).resolve().parents[2]

def test_zero_flow_no_signal_and_causality():
    assert np.all(buxton_signal(LABEL_DURATION,POST_LABEL_DELAY,1.,0.,500.)==0)
    assert np.all(buxton_signal(LABEL_DURATION,POST_LABEL_DELAY,1.,1e-5,5000.)==0)

def test_bolus_continuity_at_arrival_and_end():
    att=600.;tau=300.;eps=1e-6
    y=buxton_signal([tau]*3,[att-tau-eps,att-tau,att-tau+eps],8e7,1e-5,att)
    assert y[0]==y[1]==0 and 0<y[2]<.01
    y=buxton_signal([tau]*3,[att-eps,att,att+eps],8e7,1e-5,att)
    np.testing.assert_allclose(y,y[1],rtol=1e-8)

def test_forward_volume_matches_point_model():
    d=load_observations(ROOT/'data/raw_data.npz');flow=np.ones((1,5,35,35))*60;att=np.ones_like(flow)*1000
    y=forward_volume(flow,att,d['m0'])
    assert y.shape==(1,7,5,35,35)
    np.testing.assert_allclose(y[0,:,2,17,17],buxton_signal(LABEL_DURATION,POST_LABEL_DELAY,float(d['m0'][0,2,17,17]),1e-5,1000),rtol=1e-13)

def test_declared_readme_constants():
    text=(ROOT/'README.md').read_text(encoding='utf8')
    assert '0.98' in text and '1650' in text and '0.85' in text and '13.56' in text
    assert '100, 100, 150, 150, 400, 800, 1800' in text
    assert '170, 270, 370, 520, 670, 1070, 1870' in text

def test_input_rejects_invalid_mask_and_shapes(tmp_path):
    d=load_observations(ROOT/'data/raw_data.npz');d['mask']=d['mask'].copy();d['mask'][0,0,0,0]=2
    np.savez(tmp_path/'bad.npz',**d)
    with pytest.raises(ValueError):load_observations(tmp_path/'bad.npz')


def test_visualization_metrics_and_backend_match_judge_known_case():
    from src.visualization import metric_pair
    from src import main
    import matplotlib
    before=matplotlib.get_backend()
    import importlib
    importlib.reload(main)
    assert matplotlib.get_backend()==before
    n,e=metric_pair(np.array([2.,4.]),np.array([1.,3.]))
    assert n==pytest.approx(14/np.sqrt(200)) and e==pytest.approx(.5)
