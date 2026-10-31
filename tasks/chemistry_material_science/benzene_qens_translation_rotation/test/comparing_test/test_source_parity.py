from pathlib import Path
import numpy as np
from src.preprocessing import load_raw,prepare
from src.physics_model import predict,weighted_sse,LOWER,UPPER
from src.solvers import profile

ROOT=Path(__file__).resolve().parents[2];FIX=ROOT/'test/fixtures'
def data():return prepare(load_raw(ROOT/'data/raw_data.npz'))

def test_original_preprocessing_all_selected_groups():
    d=data()
    with np.load(FIX/'source_selected_rebin.npz') as f:
        for n in d:
            np.testing.assert_allclose(d[n]['rebinned_signal'],f[n+'_signal'],rtol=0,atol=1e-11)
            np.testing.assert_allclose(d[n]['rebinned_variance'],f[n+'_variance'],rtol=0,atol=1e-12)

def test_original_full_physical_forward_and_likelihood():
    d=data()
    with np.load(FIX/'qens_source_physical_forward_cases.npz') as f:
        np.testing.assert_array_equal(np.column_stack([LOWER,UPPER]),f['bounds_physical'])
        for n in d:
            for a,b in [('energy','physical_energy'),('q','q'),('y','observation'),('sigma','sigma'),('response','resolution')]:np.testing.assert_allclose(d[n][a],f[n+'_'+b],rtol=2e-11,atol=1e-11)
        for i,p in enumerate(f['parameters_physical']):
            curves=predict(p,d)
            for n in d:np.testing.assert_allclose(curves[n],f['predictions_'+n+'_'+str(i)],rtol=2e-11,atol=2e-10)
            np.testing.assert_allclose(-weighted_sse(curves,d)/2,f['log_likelihood'][i],rtol=2e-11)

def test_independent_bvls_profile_objective():
    with np.load(FIX/'qens_independent_profile_oracle.npz') as f:
        for i,g in enumerate(f['global_parameters']):
            objective,p,curves=profile(g,data())
            np.testing.assert_allclose(objective,f['weighted_sse_per_energy'][i].sum(),rtol=1e-8)
            assert np.all(p>=LOWER) and np.all(p<=UPPER)

def test_source_global_finite_difference_derivatives():
    d=data()
    with np.load(FIX/'qens_source_v2_global_derivative_cases.npz') as f:
        for i,pnative in enumerate(f['parameters']):
            p=pnative.copy();p[:2]/=1.5;p[[3,16]]*=2.25
            for j in range(3):
                step=f['steps'][j]/(1.5 if j<2 else 1.)
                a=p.copy();b=p.copy();a[j]+=step;b[j]-=step
                fa=predict(a,d);fb=predict(b,d)
                for n in d:
                    actual=(fa[n]-fb[n])/(2*step);expected=f['prediction_derivatives_'+n][i,:,:,j]*(1.5 if j<2 else 1.)
                    assert np.linalg.norm(actual-expected)/np.linalg.norm(expected)<2e-6
