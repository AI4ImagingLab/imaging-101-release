from pathlib import Path
import numpy as np
from src.preprocessing import load_data, prepare
from src.physics_model import transfer_function, laplacian_kernel, apply_blur, laplacian
from src.solvers import reconstruct

ROOT=Path(__file__).resolve().parents[2]

def test_frozen_upstream_stages():
    image, psf=load_data(ROOT/'data/raw_data.npz')
    p,k=prepare(image,psf)
    with np.load(ROOT/'test/fixtures/upstream_stages.npz',allow_pickle=False) as f:
        np.testing.assert_array_equal(p,f['padded'])
        np.testing.assert_allclose(k,f['kernel'],atol=1e-8,rtol=0)
        np.testing.assert_allclose(transfer_function(k),f['otf'],atol=3e-7,rtol=0)
        np.testing.assert_array_equal(laplacian_kernel(p.shape),f['laplacian'])
        np.testing.assert_allclose(reconstruct(image,psf),f['reconstruction'],rtol=2e-5,atol=.02)
        np.testing.assert_allclose(reconstruct(image,psf),f['double_reconstruction'],rtol=2e-5,atol=.02)

def test_independent_adjoint_and_flux():
    rng=np.random.default_rng(28)
    x,y=rng.normal(size=(24,24)),rng.normal(size=(24,24))
    psf=np.array([[0,0,0],[.05,.5,.25],[0,.2,0]])
    _,kernel=prepare(x,psf,pad=0);h=transfer_function(kernel)
    np.testing.assert_allclose(np.vdot(apply_blur(x,h),y),np.vdot(x,apply_blur(y,h,True)),atol=1e-12)
    np.testing.assert_allclose(apply_blur(x,h).sum(),x.sum(),atol=1e-12)
    np.testing.assert_allclose(apply_blur(np.ones_like(x),h),1,atol=1e-14)

def test_regularized_normal_equation():
    image,psf=load_data(ROOT/'data/raw_data.npz');p,k=prepare(image,psf)
    h=transfer_function(k);x=reconstruct(image,psf,return_full=True)
    lhs=apply_blur(apply_blur(x,h)-p,h,True)+.005*laplacian(laplacian(x))
    assert np.linalg.norm(lhs)/np.linalg.norm(p)<1e-13
    # DC response is one and the Laplacian annihilates constants.
    np.testing.assert_allclose(x.sum(),p.sum(),rtol=1e-14)

def test_pipeline(tmp_path):
    import subprocess,sys
    subprocess.run([sys.executable,'-m','src.main'],cwd=ROOT,check=True)
    subprocess.run([sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(ROOT/'output/reconstruction.npz'),'--report',str(tmp_path/'report.json')],cwd=tmp_path,check=True)
