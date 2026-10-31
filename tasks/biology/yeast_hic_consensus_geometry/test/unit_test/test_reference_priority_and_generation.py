from pathlib import Path
import importlib.util,json
import numpy as np
from src.generate_data import retain_contacts

ROOT=Path(__file__).resolve().parents[2]

def test_retention_endpoints_and_exact_seed_case():
    a=np.array([[0,10,2],[0,0,4],[0,0,0]])
    assert np.array_equal(retain_contacts(a,1),a)
    assert not retain_contacts(a,0).any()
    # RandomState0's deterministic independent count-retention fixture.
    assert np.array_equal(retain_contacts(a,.5),np.array([[0,6,1],[0,0,2],[0,0,0]]))
    assert np.array_equal(a,np.array([[0,10,2],[0,0,4],[0,0,0]]))

def test_actual_truth_precedence_and_invalid_truth_no_fallback(tmp_path,monkeypatch):
    spec=importlib.util.spec_from_file_location('priority_hic_judge',ROOT/'evaluation/judge.py');judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)
    with np.load(ROOT/'data/raw_data.npz') as f:support=f['bead_support'][0]
    t=np.arange(350,dtype=float)/350;coordinates=np.column_stack([t,t*t,t*t*t])[None];coordinates[0,~support]=0
    truth=tmp_path/'truth.npz';baseline=tmp_path/'baseline.npz';prediction=tmp_path/'prediction.npz'
    np.savez_compressed(truth,reconstruction=coordinates,visibility_bias=np.ones(350));np.savez_compressed(baseline,reconstruction=coordinates*.5,visibility_bias=np.ones(350));np.savez_compressed(prediction,reconstruction=coordinates)
    configuration=json.loads((ROOT/'evaluation/metrics.json').read_text());configuration['relative_poisson_excess_boundary']=1e100
    config=tmp_path/'metrics.json';config.write_text(json.dumps(configuration))
    monkeypatch.setattr(judge,'TRUTH',truth);monkeypatch.setattr(judge,'REFERENCE',baseline);monkeypatch.setattr(judge,'METRICS',config)
    report,code=judge.evaluate(prediction);assert code==0 and report['metrics']['relative_poisson_excess']==0
    np.savez_compressed(truth,wrong=coordinates)
    report,code=judge.evaluate(prediction);assert code==2 and report['metrics']=={} and report['errors']
