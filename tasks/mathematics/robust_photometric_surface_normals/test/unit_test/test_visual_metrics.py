from pathlib import Path
import importlib.util
import numpy as np
from src.visualization import metrics

T=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('independent_normal_judge',T/'evaluation/judge.py')
judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)

def test_all_reported_metrics_against_known_orientation_and_background_case():
    truth=np.array([[1.,0,0],[0,1.,0],[0,0,0]])
    estimate=np.array([[0.,1,0],[1.,0,0],[.2,0,0]])
    support=np.array([True,True,False])
    actual=metrics(estimate,truth,support)
    assert actual['ncc']==0 and abs(actual['nrmse']-np.sqrt(2/3))<1e-15
    assert actual['mean_angular_error_deg']==90 and actual['unit_norm_max_error']==.2
    assert actual==judge.normal_metrics(estimate,truth,support)

def test_public_metadata_and_output_conventions_match_source():
    text=(T/'README.md').read_text(encoding='utf8')
    for literal in ['(1, 256, 256, 3)','(1, 50, 256, 256)','float64','50','256','Cartesian','zero']:
        assert literal in text
