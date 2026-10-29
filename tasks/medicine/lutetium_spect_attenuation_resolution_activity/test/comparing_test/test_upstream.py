"""Captured exact-input source fixtures frozen before local extraction."""
from pathlib import Path
import numpy as np
import pytest
import torch
from src.physics_model import Acquisition,DELTA
from src.preprocessing import load_data,scatter_estimate
from src.solvers import reconstruct

ROOT=Path(__file__).resolve().parents[2]
torch.set_num_threads(1)


@pytest.fixture(scope='module')
def setup():
    d=load_data(ROOT/'data/raw_data.npz');device='cuda' if torch.cuda.is_available() else 'cpu'
    return d,Acquisition(d['attenuation'],d['angles'],d['radii'],device)


def test_stage_fp_bp_and_norm(setup):
    d,H=setup;z=np.load(ROOT/'test/fixtures/upstream_stages.npz')
    for fun,inp,out in [(H.forward,'input_object','output_forward'),(H.backward,'input_projections','output_backward')]:
        a=fun(torch.tensor(z[inp],device=H.device)).cpu().numpy();r=z[out]
        assert np.linalg.norm((a-r).astype(float))/np.linalg.norm(r.astype(float))<1e-6
    norm=H.backward(torch.ones((12,128,128),device=H.device),0).cpu().numpy()
    assert np.max(abs(norm-z['output_norm_subset0']))<2e-4


def test_tew_and_first_update(setup):
    d,H=setup;z=np.load(ROOT/'test/fixtures/upstream_stages.npz');g=torch.tensor(d['energy_windows'][0],device=H.device,dtype=torch.float32)
    s=scatter_estimate(d['energy_windows'],H.device);one=torch.ones(H.shape,device=H.device)
    norm=H.backward(torch.ones_like(g[::8]),0)
    ratio=g[::8]/(H.forward(one,0)+s[::8]+DELTA)
    grad=H.backward(ratio,0)-norm
    update=one+one*(1/(norm+DELTA))*grad;update[update<=0]=0
    for a,key in [(ratio,'output_ratio_subset0'),(grad,'output_gradient_subset0'),(update,'output_update0')]:
        ref=z[key];got=a.cpu().numpy()
        assert np.linalg.norm((got-ref).astype(float))/np.linalg.norm(ref.astype(float))<1e-6


def test_full_pipeline_all_32_states(setup):
    d,H=setup;ref=np.load(ROOT/'notebooks/reference_outputs/upstream_measurements.npz')
    q,states=reconstruct(H,d['energy_windows'][0],scatter_estimate(d['energy_windows'],H.device),capture=True)
    r=ref['reconstruction'][0];a=q.cpu().numpy()
    assert np.sqrt(np.mean((a.astype(float)-r)**2))/np.ptp(r)<1e-6
    assert np.max(abs(a-r))/np.ptp(r)<5e-6
    assert len(states)==32
    assert max(np.sqrt(np.mean((x.astype(float)-y)**2))/np.ptp(y) for x,y in zip(states,ref['updates']))<1e-6
    pred=(H.forward(q)+scatter_estimate(d['energy_windows'],H.device)).cpu().numpy().astype(float)
    assert np.linalg.norm(pred-d['energy_windows'][0])/np.linalg.norm(d['energy_windows'][0].astype(float))<.22


def test_inexact_continuum_bp_disclosed_and_small_weak_gap(setup):
    _,H=setup;z=np.load(ROOT/'test/fixtures/upstream_stages.npz')
    x=torch.tensor(z['input_object'],device=H.device);y=torch.tensor(z['input_projections'],device=H.device)
    hx=H.forward(x);by=H.backward(y)
    a=torch.sum(hx.double()*y.double());b=torch.sum(x.double()*by.double())
    scale=torch.linalg.vector_norm(hx.double())*torch.linalg.vector_norm(y.double())+torch.linalg.vector_norm(x.double())*torch.linalg.vector_norm(by.double())
    assert float(abs(a-b)/scale)<2e-5


def test_full_physics_forward_linearity(setup):
    _,H=setup;z=np.load(ROOT/'test/fixtures/upstream_stages.npz')
    x=torch.tensor(z['input_object'],device=H.device)
    first=H.forward(x,0).double();scaled=H.forward(1.7*x,0).double()
    assert float(torch.linalg.vector_norm(scaled-1.7*first)/torch.linalg.vector_norm(first))<1e-6
