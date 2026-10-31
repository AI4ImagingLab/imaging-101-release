from pathlib import Path
import numpy as np
from src.preprocessing import load_observations
from src.physics_model import dilation,erosion
from src.solvers import reconstruct,blind_comparison
ROOT=Path(__file__).resolve().parents[2]

def test_original_stage_parity():
    y,p=load_observations(ROOT/'data/raw_data.npz')
    with np.load(ROOT/'test/fixtures/upstream_stages.npz') as f:
        np.testing.assert_array_equal(erosion(y,p),f['output_erosion'])
        np.testing.assert_array_equal(dilation(reconstruct(y,p),p),f['output_constrained_forward'])
        with np.load(ROOT/'evaluation/ground_truth.npz') as z:
            np.testing.assert_array_equal(dilation(z['reconstruction'][0],p),f['output_truth_forward'])

def test_published_blind_comparison_parity():
    import torch
    torch.set_num_threads(1)
    y,p=load_observations(ROOT/'data/raw_data.npz');surface,probe,loss=blind_comparison(y)
    with np.load(ROOT/'notebooks/reference_outputs/upstream_methods.npz') as f:
        np.testing.assert_allclose(surface,f['blind_surfaces'],rtol=2e-6,atol=2e-5)
        np.testing.assert_allclose(probe,f['blind_tip'],rtol=2e-6,atol=2e-5)
        np.testing.assert_allclose(loss,f['blind_loss'],rtol=2e-6,atol=1e-3)

def test_acquisition_noise_matches_independent_truth():
    y,p=load_observations(ROOT/'data/raw_data.npz')
    with np.load(ROOT/'evaluation/ground_truth.npz') as z:s=z['reconstruction'][0]
    noise=y-dilation(s,p)
    assert abs(noise.mean())<.06
    assert 2.93<noise.std()<3.07
    assert abs(np.mean(noise**2)/9-1)<.03

def test_nonnegative_projection_and_forward_noise():
    y,p=load_observations(ROOT/'data/raw_data.npz');x=reconstruct(y,p)
    assert x.min()==0
    assert 2.9<np.sqrt(np.mean((dilation(x,p)-y)**2))<3.2

def test_generic_order_adjoint_and_opening_identity():
    rng=np.random.default_rng(73);s=rng.uniform(0,5,(9,11));p=np.array([[-4.,-2.,-5.],[-1.,0.,-3.],[-7.,-2.,-6.]])
    image=dilation(s,p);upper=erosion(image,p)
    assert np.all(upper>=s-1e-12)
    np.testing.assert_allclose(dilation(upper,p),image,rtol=0,atol=1e-12)
    y=image+rng.uniform(0,2,image.shape)
    assert np.all(s<=erosion(y,p)+1e-12)
    assert np.all(dilation(erosion(y,p),p)<=y+1e-12)

def test_known_analytic_asymmetric_contact():
    p=np.array([[-100.,-100.,-100.],[-2.,0.,-100.],[-100.,-100.,-100.]])
    s=np.zeros((5,5));s[2,2]=7
    d=dilation(s,p)
    assert d[2,2]==7 and d[2,3]==5 and d[2,1]==0
    assert erosion(d,p)[2,2]==7
