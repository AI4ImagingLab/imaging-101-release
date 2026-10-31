"""Fresh full real acquisition, not an archived reconstruction comparison only."""
from pathlib import Path
import importlib.util
import numpy as np
from src.main import run
from src.visualization import map_metrics

TASK=Path(__file__).resolve().parents[2]

def test_fresh_complete_pipeline_and_independent_judge(tmp_path):
    prediction=tmp_path/'fresh/reconstruction.npz'
    result=run(TASK/'data/raw_data.npz',prediction)
    with np.load(TASK/'evaluation/baseline_reference.npz',allow_pickle=False) as z:
        ncc,error=map_metrics(result['reconstruction'],z['reconstruction'])
    assert ncc.min()>=.999999999999
    assert error.max()<=2.9645042444036815e-8
    with np.load(prediction,allow_pickle=False) as z:
        assert z.files==['reconstruction']
        np.testing.assert_array_equal(z['reconstruction'],result['reconstruction'])
    spec=importlib.util.spec_from_file_location('pipeline_independent_judge',TASK/'evaluation/judge.py')
    judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)
    report,code=judge.evaluate(prediction)
    assert code==0,report
    assert report['metrics']['forward_weighted_rms_TECu']<=1.5272988162761219
