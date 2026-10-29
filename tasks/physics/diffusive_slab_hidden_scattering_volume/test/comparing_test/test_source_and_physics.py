"""Frozen original stages and independent physical controls, selected before extraction."""
from pathlib import Path
import numpy as np
import pytest
import torch
from src.preprocessing import preprocess, load_observations
from src.physics_model import diffusion_kernel, transfer_function, frequency_mapping, convolve, paired_wiener
from src.solvers import migrate, reconstruct

ROOT=Path(__file__).resolve().parents[2]
torch.set_num_threads(1)

@pytest.fixture(scope='module')
def upstream():
    with np.load(ROOT/'test/fixtures/upstream_stages.npz',allow_pickle=False) as f:
        yield {k:f[k] for k in f.files}

def rel(a,b):
    return np.linalg.norm((np.asarray(a)-b).ravel())/np.linalg.norm(b.ravel())

def test_original_preprocessing(upstream):
    b=preprocess(load_observations(ROOT/'data/raw_data.npz'))
    np.testing.assert_array_equal(b,upstream['output_preprocessed'])

def test_original_kernel(upstream):
    np.testing.assert_allclose(diffusion_kernel(),upstream['output_kernel'],rtol=1e-12,atol=1e-20)

def test_original_transfer(upstream):
    np.testing.assert_allclose(transfer_function().numpy(),upstream['output_transfer'],rtol=0,atol=5e-7)

def test_original_frequency_mapping(upstream):
    np.testing.assert_array_equal(frequency_mapping().numpy()[::8,::4,::4],upstream['output_frequency_mapping'])

def test_original_deconvolution(upstream):
    b=torch.from_numpy(upstream['output_preprocessed'])
    padded=torch.nn.functional.pad(b,(0,32,0,32,0,128))
    assert rel(paired_wiener(padded).numpy(),upstream['output_deconvolution'])<.00035

def test_original_migration_only(upstream):
    b=torch.from_numpy(upstream['output_preprocessed'])
    field=migrate(torch.nn.functional.pad(b,(0,32,0,32,0,128))).numpy()[:128,:32,:32]
    assert rel(field,upstream['output_migration_only'])<1e-6

def test_original_full_pipeline(upstream):
    fresh=reconstruct(load_observations(ROOT/'data/raw_data.npz'))
    assert rel(fresh['reconstruction'],upstream['output_combined'])<.003
    assert np.all(fresh['reconstruction'][-15:]==0)

@pytest.mark.parametrize('index',[0,1,2])
def test_independent_known_geometry(index):
    target_depth=[.3,.5,.75][index]
    with np.load(ROOT/'test/fixtures/physical_point_controls.npz',allow_pickle=False) as f:
        observations=f[f'control_{index}_observations']
    # Independent FFT-generated analytic controls contain only sub-ULP negative
    # roundoff (~1e-16 L1), not signed measured observations. Assert this bound
    # before removing it for the strict nonnegative count-input contract.
    assert abs(observations[observations<0]).sum()/abs(observations).sum()<1e-12
    field=reconstruct(np.maximum(observations,0))['reconstruction']
    iz,ix,iy=np.unravel_index(np.argmax(field),field.shape)
    assert abs(iz/127.-target_depth)<=.09
    assert np.hypot(-.35+ix*.7/31,-.35+iy*.7/31)<=.15

@pytest.mark.parametrize('index,target',[(0,(.12,-.16,.5)),(1,(-.15,.10,.65))])
def test_independent_offaxis_coordinate_order(index,target):
    with np.load(ROOT/'test/fixtures/offaxis_controls.npz',allow_pickle=False) as f:
        observations=f[f'control_{index}_observations']
        original=f[f'control_{index}_reconstruction']
    assert abs(observations[observations<0]).sum()/abs(observations).sum()<1e-12
    field=reconstruct(np.maximum(observations,0))['reconstruction']
    iz,ix,iy=np.unravel_index(np.argmax(field),field.shape)
    assert abs(iz/127.-target[2])<=.09
    assert np.hypot(-.35+ix*.7/31-target[0],-.35+iy*.7/31-target[1])<=.15
    assert rel(field,original)<.003

def test_diffusion_flux_and_symmetry():
    k=diffusion_kernel()
    assert k.shape==(256,64,64) and np.isfinite(k).all() and np.min(k)>=0
    assert abs(float(k.sum())-1)<1e-12

def test_forward_adjoint():
    g=torch.Generator().manual_seed(43)
    a=torch.randn((256,64,64),generator=g)
    b=torch.randn((256,64,64),generator=g)
    lhs=torch.sum(convolve(a)*b).double().item()
    rhs=torch.sum(a*convolve(b,adjoint=True)).double().item()
    assert abs(lhs-rhs)/max(abs(lhs),abs(rhs),1)<2e-5

def test_constant_forward_flux():
    y=convolve(torch.ones((256,64,64)))
    torch.testing.assert_close(y,torch.ones_like(y),rtol=2e-6,atol=2e-6)
