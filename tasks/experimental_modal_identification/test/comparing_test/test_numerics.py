
from pathlib import Path
import numpy as np
from src.preprocessing import load_data
from src.solvers import reconstruct
ROOT=Path(__file__).resolve().parents[2]
def test_full_upstream_parity():
    result=reconstruct(load_data(ROOT/'data/raw_data.npz'))
    with np.load(ROOT/'evaluation/baseline_reference.npz',allow_pickle=False) as f:
        for k in f.files:
            assert result[k].shape==f[k].shape
            assert np.linalg.norm(result[k]-f[k])/np.linalg.norm(f[k])<1e-5,k

def test_polynomial_stage_upstream():
    from src.solvers import prepare,identify_poles
    d=load_data(ROOT/'data/raw_data.npz');f,h,dt=prepare(d['frequency'],d['accelerance']);p=np.concatenate(identify_poles(f,h,dt));ref=np.load(ROOT/'test/fixtures/upstream_stages.npz')
    # Polynomial root ordering is not physical: nearest roots are compared within each order.
    start=0
    for order in range(2,121,2):
        x=p[start:start+order];y=ref['output_poles'][start:start+order];assert max(min(abs(z-y)) for z in x)<2e-4;start+=order
    np.testing.assert_array_equal(f,ref['output_frequency'])
    result=reconstruct(d)
    for key,stage in [('natural_frequency','output_natural_frequency'),('damping_ratio','output_damping'),('reconstruction','output_modal_constants'),('lower_residual','output_lower_residual'),('upper_residual','output_upper_residual')]:
        assert np.linalg.norm(result[key][0]-ref[stage])/np.linalg.norm(ref[stage])<1e-5
def test_independent_pole_residue_formula():
    from src.physics_model import forward,pole_properties
    p=np.array([-2+30j]);freq=np.array([0.,1.,5.]);a=np.array([[1.2+.3j]]);lr=np.array([.4+.1j]);ur=np.array([.01+.02j]);w=2*np.pi*freq
    expected=-w*w*(a[0,0]/(1j*w-p[0])+a[0,0].conjugate()/(1j*w-p[0].conjugate()))+lr[0]-w*w*ur[0]
    np.testing.assert_allclose(forward(p,a,lr,ur,freq)[0],expected,rtol=1e-13,atol=1e-13)
    fn,xi=pole_properties(p);np.testing.assert_allclose(fn,np.sqrt(904)/(2*np.pi));np.testing.assert_allclose(xi,2/np.sqrt(904))
