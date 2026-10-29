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

def test_known_constant_signal_accepts_nonunique_better_parameters(tmp_path,monkeypatch):
    spec=importlib.util.spec_from_file_location('objective_judge',ROOT/'evaluation/judge.py');judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)
    shape=(1,1,1,1);one=np.ones(shape)
    baseline=dict(intra_fraction=one*.5,intrinsic_diffusivity=one*.001,extra_transverse_diffusivity=one*.0005,extra_mean_diffusivity=one*(1-.5*2/3)*.001)
    np.savez(tmp_path/'baseline.npz',**baseline)
    np.savez(tmp_path/'raw.npz',observation_mask=np.ones(shape,dtype=bool),diffusion_signal=np.full(shape+(3,),100.),bvalues=np.array([[0.,200.,400.]]))
    for name,value in [('REFERENCE',tmp_path/'baseline.npz'),('RAW',tmp_path/'raw.npz'),('SHAPE',shape)]:monkeypatch.setattr(judge,name,value)
    for fraction in (0.,.4,1.):
        fields={key:one*0 for key in judge.KEYS};fields['intra_fraction']=one*fraction
        np.savez(tmp_path/'prediction.npz',**fields)
        report,code=judge.evaluate(tmp_path/'prediction.npz')
        assert code==0,report
        assert report['metrics']['maximum_normalized_mse_excess']<0
