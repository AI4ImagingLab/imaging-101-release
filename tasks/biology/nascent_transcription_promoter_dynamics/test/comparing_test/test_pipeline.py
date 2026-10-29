"""Full fresh bound-observation reconstruction, distinct from frozen stage tests."""
from pathlib import Path
import importlib.util
import numpy as np
from src.main import reconstruct
T=Path(__file__).resolve().parents[2]

def test_complete_bound_pipeline_and_judge(tmp_path):
    output=reconstruct(T/'data/raw_data.npz')
    prediction=tmp_path/'complete_prediction.npz';np.savez_compressed(prediction,**output)
    spec=importlib.util.spec_from_file_location('standalone_judge',T/'evaluation/judge.py');judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)
    report=judge.evaluate(prediction)
    assert report['status']=='pass',report
