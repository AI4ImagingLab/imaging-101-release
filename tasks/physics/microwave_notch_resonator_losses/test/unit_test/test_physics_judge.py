import importlib.util
from pathlib import Path
import numpy as np
import pytest
from src.physics_model import forward,dissipation_quality
from src.solvers import _S21_notch

TASK=Path(__file__).parents[2]
spec=importlib.util.spec_from_file_location('judge',TASK/'evaluation/judge.py')
judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)

def test_forward_limits_and_independent_convention():
    f=np.array([9.,10.,11.]);q=100.
    np.testing.assert_allclose(forward(f,10.,q,200.,0.),_S21_notch(f,fr=10.,Ql=q,Qc=200.,phi=0.),rtol=1e-14)
    assert forward(np.array([10.]),10.,100.,200.,0.)[0]==.5
    np.testing.assert_allclose(forward(f,10.,q,np.inf,0.),1.)
    qc,qi=dissipation_quality(100.,200.,0.)
    assert qc==200. and qi==200.

def test_known_metrics():
    ref={'resonance_frequency':np.ones((1,1)),'quality_factors':np.ones((1,3)),'reconstruction':np.ones((1,1001),complex)}
    pred={k:2*v for k,v in ref.items()}
    assert all(v==1. for v in judge.score(pred,ref,ref['reconstruction']).values())

def test_degenerate_reference_rejected(tmp_path,monkeypatch):
    import shutil
    assets=tmp_path/'task/evaluation';assets.mkdir(parents=True)
    (assets.parent/'data').mkdir()
    shutil.copyfile(TASK/'evaluation/metrics.json',assets/'metrics.json')
    shutil.copyfile(TASK/'data/raw_data.npz',assets.parent/'data/raw_data.npz')
    reference=judge.validate(TASK/'evaluation/baseline_reference.npz')
    reference['quality_factors'][0,0]=0
    np.savez(assets/'baseline_reference.npz',**reference)
    monkeypatch.setattr(judge,'HERE',assets)
    report,code=judge.evaluate(TASK/'evaluation/baseline_reference.npz')
    assert code==2 and report['status']=='error' and report['metrics']=={}
    assert 'reference quality' in report['errors'][0]

@pytest.mark.parametrize('case',['missing','key','shape','dtype','nan','inf','complex_real'])
def test_bad_inputs(tmp_path,case):
    path=tmp_path/'bad.npz'
    data=judge.validate(TASK/'evaluation/baseline_reference.npz')
    if case=='missing':pass
    else:
        if case=='key':data.pop('quality_factors')
        elif case=='shape':data['quality_factors']=np.ones((3,))
        elif case=='dtype':data['quality_factors']=data['quality_factors'].astype(np.float32)
        elif case=='complex_real':data['reconstruction']=data['reconstruction'].real
        else:data['quality_factors'][0,0]=np.nan if case=='nan' else np.inf
        np.savez(path,**data)
    report,code=judge.evaluate(path)
    assert code==2 and report['metrics']=={} and report['errors']

def test_poor_prediction(tmp_path):
    data=judge.validate(TASK/'evaluation/baseline_reference.npz')
    np.savez(tmp_path/'zero.npz',**{k:np.zeros_like(v) for k,v in data.items()})
    report,code=judge.evaluate(tmp_path/'zero.npz')
    assert code==1 and report['status']=='fail'
