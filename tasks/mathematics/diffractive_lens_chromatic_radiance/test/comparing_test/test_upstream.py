from pathlib import Path
import numpy as np
from src.physics_model import gradient,positions,patch_columns,overlap
F=Path(__file__).resolve().parents[1]/'fixtures'
def test_native_mex_primitives():
    z=np.load(F/'primitives.npz',allow_pickle=False);a=z['input_primitive']
    p=patch_columns(a,4,3);np.testing.assert_allclose(p,z['output_patches'],atol=1e-12)
    np.testing.assert_allclose(overlap(p,a.shape,4,3),z['output_overlap'],atol=1e-12)
    gx,gy=gradient(a);np.testing.assert_allclose(gx,z['output_gx'],atol=1e-12);np.testing.assert_allclose(gy,z['output_gy'],atol=1e-12)

def test_gradient_constant():
    gx,gy=gradient(np.ones((7,9)));assert not gx.any() and not gy.any()

def test_roundtrip_overlap():
    a=np.arange(143.).reshape(11,13);p=patch_columns(a,4,3)
    np.testing.assert_allclose(overlap(p,a.shape,4,3)/overlap(np.ones_like(p),a.shape,4,3),a)

def test_full_original_stage_parity():
    from src.solvers import reconstruct_channel
    T=F.parents[1];image=np.load(T/'data/raw_data.npz',allow_pickle=False)['measurements'][0]
    z=np.load(F/'stages.npz',allow_pickle=False)
    for c in [1,3]:
        out,psf,diag=reconstruct_channel(image[:,:,1],image[:,:,c-1],capture=True)
        np.testing.assert_allclose(diag['initial'],z[f'c{c}_initial'],rtol=1e-9,atol=1e-10)
        for i,row in enumerate(diag['iteration']):
            for k in ['psf_ft_first','blur_base','A_first','b_first','coefficients','deblur']:
                atol=1e-9 if k in ('A_first','b_first') else 1e-8
                np.testing.assert_allclose(row[k],z[f'c{c}_i{i}_{k}'],rtol=0 if k in ('A_first','b_first') else 1e-7,atol=atol,err_msg=f'channel{c},iteration{i},{k}')

def test_patch_adjoint_and_psf_normal():
    from src.physics_model import psf_prior,estimate_psf
    rng=np.random.default_rng(15);a=rng.normal(size=(11,13));b=rng.normal(size=patch_columns(a,4,3).shape)
    np.testing.assert_allclose(np.sum(patch_columns(a,4,3)*b),np.sum(a*overlap(b,a.shape,4,3)),atol=1e-12)
    x=rng.normal(size=(8,8));y=rng.normal(size=(8,8));p=psf_prior(8,.3,.3);h=estimate_psf(x,y,p);fx=np.fft.fft2(x)
    np.testing.assert_allclose((abs(fx)**2+p)*h,np.conj(fx)*np.fft.fft2(y),atol=1e-12)
    assert p.min()>=.3
