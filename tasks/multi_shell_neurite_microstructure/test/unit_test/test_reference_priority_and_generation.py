from pathlib import Path
import importlib.util,json
import numpy as np
from numpy.testing import assert_allclose
from src.generate_data import add_gaussian_noise

ROOT=Path(__file__).resolve().parents[2]

def test_exact_known_noise_preserves_b0_and_signs():
    y=np.array([[10.,1.,2.],[20.,1.,2.]])
    got=add_gaussian_noise(y,[0,200,400],.1,seed=42)
    expected=np.array([[10.,1.4967141530112327,1.8617356988288154],[20.,2.295377076201385,5.046059712816051]])
    assert_allclose(got,expected,rtol=0,atol=1e-14)
    assert np.array_equal(y,np.array([[10.,1.,2.],[20.,1.,2.]]))
    assert add_gaussian_noise(y,[0,200,400],100,seed=42)[0,2]<0
    assert np.array_equal(add_gaussian_noise(y,[0,200,400],0),y)

def test_actual_truth_precedence_and_invalid_truth_no_fallback(tmp_path,monkeypatch):
    spec=importlib.util.spec_from_file_location('priority_judge',ROOT/'evaluation/judge.py');judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)
    shape=(1,1,44427);r=np.arange(1,44428,dtype=float).reshape(shape)
    truth={key:2*r for key in judge.KEYS};baseline={key:r for key in judge.KEYS}
    for name,arrays in [('truth.npz',truth),('baseline.npz',baseline),('prediction.npz',truth)]:np.savez_compressed(tmp_path/name,**arrays)
    np.savez_compressed(tmp_path/'raw.npz',observation_mask=np.ones(shape,dtype=bool))
    config={'fields':{key:dict(ncc_boundary=.999999,nrmse_boundary=1e-8) for key in judge.KEYS}};(tmp_path/'metrics.json').write_text(json.dumps(config))
    for name,value in [('TRUTH',tmp_path/'truth.npz'),('REFERENCE',tmp_path/'baseline.npz'),('CONFIG',tmp_path/'metrics.json'),('RAW',tmp_path/'raw.npz'),('SHAPE',shape)]:monkeypatch.setattr(judge,name,value)
    report,code=judge.evaluate(tmp_path/'prediction.npz');assert code==0 and report['metrics'][judge.KEYS[0]+'.nrmse']==0
    np.savez_compressed(tmp_path/'truth.npz',wrong=r)
    report,code=judge.evaluate(tmp_path/'prediction.npz');assert code==2 and report['metrics']=={} and report['errors']
