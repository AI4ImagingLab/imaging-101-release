from pathlib import Path
import numpy as np
import pytest
from src.preprocessing import load_data
from src.solvers import prepare_inverse,prior_terms,Objective,_unique_minimum

ROOT=Path(__file__).resolve().parents[2]

@pytest.mark.parametrize('prior',['tv','smooth','tikhonov'])
def test_prior_gradient_and_Hessian_by_independent_differences(prior):
    c=prepare_inverse(load_data(ROOT/'data/raw_data.npz'))
    E=200+20*np.sin(np.arange(800)*.1);d=np.cos(np.arange(800)*.13);d/=np.linalg.norm(d);h=1e-3
    value,g,H=prior_terms(E,c,prior)
    plus=prior_terms(E+h*d,c,prior);minus=prior_terms(E-h*d,c,prior)
    assert abs((plus[0]-minus[0])/(2*h)-g@d)<1e-6
    actual=(plus[1]-minus[1])/(2*h);expected=H@d
    assert np.linalg.norm(actual-expected)/max(1e-12,np.linalg.norm(expected))<1e-6

def test_barrier_fixed_coefficients_cost_gradient_hessian_and_projection_boundary():
    c=prepare_inverse(load_data(ROOT/'data/raw_data.npz'));o=Objective(c)
    E=np.full(800,200.);E[0]=.09;E[1]=301.;d=np.zeros(800);d[0]=1.;d[1]=.2;h=1e-6
    value,g,diag=o.barrier(E)
    plus=o.barrier(E+h*d);minus=o.barrier(E-h*d)
    assert abs((plus[0]-minus[0])/(2*h)-g@d)/abs(g@d)<1e-8
    np.testing.assert_allclose((plus[1]-minus[1])/(2*h),diag*d,rtol=1e-8,atol=1e-9)
    assert o.barrier(np.full(800,.1))[0]==0
    assert o.barrier(np.full(800,300.))[0]==0

def test_source_exact_tie_is_explicit_error():
    with pytest.raises(FloatingPointError):_unique_minimum(np.ones(100))
