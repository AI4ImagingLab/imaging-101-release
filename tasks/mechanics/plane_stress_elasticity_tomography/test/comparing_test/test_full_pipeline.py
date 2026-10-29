import importlib.util
import json
from pathlib import Path
import numpy as np
from src.main import run

ROOT=Path(__file__).resolve().parents[2]

def test_fresh_complete_pipeline_numeric_parity_and_independent_judge(tmp_path):
    destination=tmp_path/'nested/reconstruction.npz'
    fresh=run(ROOT/'data/raw_data.npz',destination)
    tolerance=json.loads((ROOT/'test/fixtures/stage_tolerances.json').read_text())
    with np.load(ROOT/'notebooks/reference_outputs/qualified_tv.npz',allow_pickle=False) as reference:
        field=reference['reconstruction'][0]
        assert np.linalg.norm(fresh.reconstruction-field)/np.linalg.norm(field)<tolerance['frozen_full_field_relative_boundary']
        cost=reference['objective_history'][-1]
        assert abs(fresh.objective_history[-1]-cost)/abs(cost)<tolerance['frozen_full_cost_relative_boundary']
    spec=importlib.util.spec_from_file_location('full_pipeline_judge',ROOT/'evaluation/judge.py')
    judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)
    report,code=judge.evaluate(destination)
    assert code==0 and report['passed']
