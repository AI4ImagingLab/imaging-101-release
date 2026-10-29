from pathlib import Path
import numpy as np
from src.physics_model import phase_differences,encode_phasors
from src.solvers import recover_velocity,reconstruct

ROOT=Path(__file__).resolve().parents[2]
def fixture():
    with np.load(ROOT/'test/fixtures/upstream_stages.npz',allow_pickle=False) as z:return {k:z[k] for k in z.files}

def test_original_phase_and_complete_selected_inverse_stages():
    f=fixture();phase=phase_differences(f['encoding_images'])
    np.testing.assert_allclose(phase,f['phase_differences'],rtol=0,atol=2e-6)
    velocity,diagnostics=recover_velocity(f['encoding_images'])
    np.testing.assert_allclose(velocity,f['velocity'],rtol=0,atol=5e-8)
    np.testing.assert_array_equal(diagnostics['mask'],f['mask'])
    np.testing.assert_allclose(diagnostics['reference_magnitude'],f['reference_magnitude'],rtol=0,atol=0)

def test_native_pipeline_all_phases_directions_voxels():
    with np.load(ROOT/'data/raw_data.npz',allow_pickle=False) as z:data=z['encoding_images']
    out,diagnostics=reconstruct(data)
    with np.load(ROOT/'evaluation/baseline_reference.npz',allow_pickle=False) as z:reference=z['reconstruction']
    assert out['reconstruction'].shape==(1,27,3,54,67,100)
    assert out['reconstruction'].dtype==np.float32
    np.testing.assert_allclose(out['reconstruction'],reference,rtol=0,atol=5e-8)

def test_analytic_known_velocity_and_venc_sign():
    t=np.arange(9);v=np.zeros((2,3,4,9,3));v[...,0]=.02*np.sin(2*np.pi*t/9)
    v[...,1]=.01*np.cos(2*np.pi*t/9);v[...,2]=.003
    common=np.exp(1j*.7)*np.ones(v.shape[:-1]);image=encode_phasors(v)*common[...,None]
    estimate,diag=recover_velocity(image,mask_fraction=.1)
    target=v-v.mean(axis=3,keepdims=True)
    np.testing.assert_allclose(estimate,target,atol=1e-16)

def test_common_phase_invariance_and_time_constant_gauge():
    f=fixture();images=f['encoding_images'].astype(np.complex128)
    a,da=recover_velocity(images);b,db=recover_velocity(images*np.exp(1j*1.73))
    np.testing.assert_allclose(a,b,atol=2e-16);np.testing.assert_array_equal(da['mask'],db['mask'])

def test_dynamic_mask_zeroing_and_principal_interval():
    v=np.zeros((1,1,1,4,3));v[...,0]=[.01,.02,.03,.04]
    images=encode_phasors(v);images[:,:,:,0,:]*=.01
    out,diag=recover_velocity(images)
    assert not diag['mask'][0,0,0,0];assert np.all(out[0,0,0,0]==0)
    expected=np.array([.01,.02,.03,.04])-.025
    np.testing.assert_allclose(out[0,0,0,1:,0],expected[1:],atol=1e-16)
    assert abs(out[...,0].mean())>0

def test_alias_period_and_phase_branch_are_explicit():
    v=np.zeros((1,1,1,3,3));v[...,0]=[.17,.0,-.17]
    p=phase_differences(encode_phasors(v))
    np.testing.assert_allclose(p[0,0,0,:,0]/np.pi*.16,[-.15,0,.15],atol=1e-16)
