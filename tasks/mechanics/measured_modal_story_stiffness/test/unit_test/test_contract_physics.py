from pathlib import Path
import importlib.util,shutil
import numpy as np
import pytest
from src.physics_model import stiffness_matrix,influence_matrices,eigenmodes,residual_jacobian
from src.preprocessing import MASS,NOMINAL_STIFFNESS,load_data,normalize_modes
from src.visualization import relative_error,stiffness_figure
ROOT=Path(__file__).resolve().parents[2]
sp=importlib.util.spec_from_file_location('judge',ROOT/'evaluation/judge.py');judge=importlib.util.module_from_spec(sp);sp.loader.exec_module(judge)
def good():
    with np.load(ROOT/'evaluation/baseline_reference.npz') as z:return {k:z[k] for k in judge.SHAPES}
def test_spring_energy_and_eigen_equilibrium():
    alpha=np.array([-.2,.3,.1,-.4]);u=np.array([1.,2.,-.3,.7]);K=stiffness_matrix(alpha)
    assert u@K@u==pytest.approx(np.sum(10*(1+alpha)*np.diff(np.r_[0,u])**2))
    lam,v=eigenmodes(alpha,np.array([2,0]));np.testing.assert_allclose(K@v,MASS*v*lam,atol=1e-12);assert NOMINAL_STIFFNESS==10 and MASS==12.060/386.088
@pytest.mark.parametrize('form',[1,2,3])
def test_analytic_jacobian(form):
    f,v=load_data(ROOT/'data/raw_data.npz');x=np.array([-.15,.2,-.1,.4,1.1,-.7])[:6 if form==3 else 4];r,J=residual_jacobian(x,f,v,form);h=1e-6
    numeric=np.column_stack([(residual_jacobian(x+np.eye(len(x))[i]*h,f,v,form)[0]-residual_jacobian(x-np.eye(len(x))[i]*h,f,v,form)[0])/(2*h) for i in range(len(x))]);np.testing.assert_allclose(J,numeric,rtol=2e-6,atol=2e-8)
def test_metrics_and_normalization():
    assert relative_error([2.,4.],np.array([1.,2.]))==1
    f,v=load_data(ROOT/'data/raw_data.npz');y,q=normalize_modes(v);np.testing.assert_array_equal(y[q,np.arange(2)],np.ones(2))
    assert np.max(abs(y))==1
@pytest.mark.parametrize('bad',['missing','shape','int','float16','complex','nan','inf'])
def test_invalid(tmp_path,bad):
    d=good()
    if bad=='missing':del d['reconstruction']
    elif bad=='shape':d['reconstruction']=d['reconstruction'][0]
    elif bad in ('int','float16','complex'):d['reconstruction']=d['reconstruction'].astype(bad)
    else:d['reconstruction'][0,0]=float(bad)
    p=tmp_path/'bad.npz';np.savez(p,**d);r=judge.evaluate(p);assert r['status']=='error' and r['metrics']=={}
def test_error_poor_and_truth_precedence(tmp_path):
    assert judge.evaluate(tmp_path/'missing')['status']=='error'
    d=good();p=tmp_path/'p.npz';np.savez(p,**d);r=judge.evaluate(p);assert r['passed'];assert r['metrics']['modal_residual_rms']==pytest.approx(np.sqrt(.17206511765637536/8))
    assert judge.evaluate(p,tmp_path)['status']=='error'
    for name in ('metrics.json','baseline_reference.npz'):shutil.copyfile(ROOT/'evaluation'/name,tmp_path/name)
    d['reconstruction']*=1.5;np.savez(tmp_path/'ground_truth.npz',**d)
    r=judge.evaluate(p,tmp_path,ROOT/'data/raw_data.npz');assert r['status']=='fail' and r['metrics']['parameter_relative_error']==pytest.approx(1/3)
    d=good();d['unmeasured_modes']*=0;np.savez(p,**d);assert judge.evaluate(p)['status']=='fail'
