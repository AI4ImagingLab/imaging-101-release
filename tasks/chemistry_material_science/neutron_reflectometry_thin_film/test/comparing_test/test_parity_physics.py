from pathlib import Path
import numpy as np
from src.physics_model import forward,profile,residuals,abeles,slabs,INITIAL,BOUNDS,Z
from src.preprocessing import load_observations
F=Path(__file__).resolve().parents[1]/'fixtures/upstream.npz'

def test_public_data_exact():
    with np.load(F,allow_pickle=False) as f:
        for a,k in zip(load_observations(),('input_q','input_reflectivity','input_error','input_dq')): np.testing.assert_array_equal(a,f[k])

def test_forward_and_residual_upstream():
    q,r,e,dq=load_observations()
    with np.load(F,allow_pickle=False) as f:
        np.testing.assert_array_equal(INITIAL,f['param_initial'])
        np.testing.assert_allclose(forward(INITIAL,q,dq),f['output_initial_model'],rtol=2e-12,atol=1e-15)
        np.testing.assert_allclose(residuals(INITIAL,q,r,e,dq),f['output_initial_residual'],rtol=2e-11,atol=1e-10)
        np.testing.assert_allclose(forward(f['output_parameters'],q,dq),f['output_prediction'],rtol=2e-12,atol=1e-15)
        np.testing.assert_allclose(profile(f['output_parameters']),f['output_profile'],rtol=2e-13,atol=1e-13)
        np.testing.assert_array_equal(slabs(f['output_parameters']),f['output_slabs'][:,:4])
        np.testing.assert_array_equal(Z,f['input_z'])

def test_single_interface_fresnel_and_total_reflection():
    rho=6.36e-6;q=np.linspace(.025,.25,71)
    k=q/2;kb=np.sqrt(k*k-4*np.pi*rho)
    expected=((k-kb)/(k+kb))**2
    np.testing.assert_allclose(abeles(q,np.array([[0,0,0,0],[0,6.36,0,0]])),expected,rtol=2e-12)
    np.testing.assert_allclose(abeles(np.array([.001,.005]),np.array([[0,0,0,0],[0,6.36,0,0]])),1.,atol=2e-12)

def test_zero_contrast_and_layer_split_invariance():
    q=np.linspace(.01,.3,100)
    np.testing.assert_allclose(abeles(q,np.array([[0,2.,0,0],[100,2.,0,0],[0,2.,0,0]])),0.,atol=1e-27)
    a=np.array([[0,0.,0,0],[100,3,0,0],[0,6,0,0]])
    b=np.array([[0,0.,0,0],[40,3,0,0],[60,3,0,0],[0,6,0,0]])
    np.testing.assert_allclose(abeles(q,a),abeles(q,b),rtol=2e-12,atol=1e-14)

def test_full_pipeline_against_upstream():
    # Active reconstruction, never read output/reconstruction.npz as the test result.
    from src.solvers import reconstruct
    x,info=reconstruct(*load_observations())
    with np.load(F,allow_pickle=False) as f:
        nrmse=np.sqrt(np.mean((x['reconstruction'][0]-f['output_profile'])**2))/np.ptp(f['output_profile'])
        pe=np.max(np.abs(x['parameters'][0]-f['output_parameters'])/np.ptp(BOUNDS,axis=1))
        assert nrmse<2e-5 and pe<2e-4
    assert info['optimizer_success']
