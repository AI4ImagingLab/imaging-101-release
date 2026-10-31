from pathlib import Path
import sys,numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.main import run
from src.physics_model import kernel,relaxation_times,curvature_matrix,predict,objective
from src.solvers import compare_lcurve

def test_stages_and_final():
    a=np.load(ROOT/'test/fixtures/upstream.npz');t,g,r,s,b,f,v,p,y,res,pen=run(ROOT)
    np.testing.assert_array_equal(t,a['input_time']);np.testing.assert_array_equal(g,a['input_g2']);np.testing.assert_allclose(s,a['output_s'],rtol=3e-15)
    np.testing.assert_allclose(b,a['output_beta'],rtol=3e-14);np.testing.assert_allclose(f,a['output_g1'],rtol=3e-14,equal_nan=True)
    np.testing.assert_allclose(p,a['output_spectrum'],rtol=2e-6,atol=1e-9);np.testing.assert_allclose(y,a['output_prediction'],rtol=1e-10,atol=1e-12)
    for j,end in enumerate(v):np.testing.assert_allclose(objective(p[j],kernel(t[:end],s),f[:end,j]),res[j]**2,rtol=1e-11)

def test_complete_lcurve_parity():
    t,g,r,s,b,f,v,*_=run(ROOT);a=np.load(ROOT/'notebooks/reference_outputs/upstream_lcurve.npz');alphas,res,pen,selected,p,y=compare_lcurve(t,g,s,b,f,v)
    np.testing.assert_array_equal(alphas,a['input_alpha']);np.testing.assert_array_equal(selected,a['output_alpha']);np.testing.assert_allclose(res,a['output_residual_norm'],rtol=2e-7,atol=1e-10);np.testing.assert_allclose(pen,a['output_penalty_norm'],rtol=2e-7,atol=1e-9)
    np.testing.assert_allclose(p,a['output_spectrum'],rtol=2e-6,atol=1e-8)

def test_independent_physics():
    r=np.array([1.,2.,4.]);s=relaxation_times(r);np.testing.assert_allclose(s/s[0],[1,2,4],rtol=2e-15)
    t=np.array([0.,s[0],2*s[0]]);k=kernel(t,s);np.testing.assert_allclose(k[:,0],np.exp(-np.array([0,1,2])))
    np.testing.assert_array_equal(k[0],np.ones(3));assert np.all(np.diff(k,axis=0)<0)
    np.testing.assert_allclose(curvature_matrix(8)@np.arange(8.),0,atol=1e-15)
    u=np.array([.2,.3,.5]);z=np.array([.7,-.2,.1]);np.testing.assert_allclose((k@u)@z,u@(k.T@z),rtol=2e-15)
    np.testing.assert_allclose(predict(np.array([[1.,0.,0.]]),t,s,np.array([.1]))[:,0],1+.1*np.exp(-2*t/s[0]))

def test_public_constants():
    text=(ROOT/'README.md').read_text(encoding='utf-8')
    for expected in ['293','8.9e-4','1.33','817','150','1.380649e-23','(1, 2, 373)']:assert expected in text
