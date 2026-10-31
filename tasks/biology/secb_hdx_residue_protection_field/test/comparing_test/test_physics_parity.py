from pathlib import Path
import numpy as np
import torch
from src.preprocessing import load_data
from src.physics_model import forward,losses,tensors
from src.initialization import initial_energy
from src.solvers import reconstruct
ROOT=Path(__file__).resolve().parents[2]

def setup():
 raw,y=load_data(ROOT/'data/raw_data.npz');return raw,y,np.load(ROOT/'test/fixtures/upstream.npz')

def test_control_and_limits():
 raw,y,f=setup();np.testing.assert_array_equal(y,f['input_corrected_observed']);x=raw['membership'];k=raw['intrinsic_rates'];np.testing.assert_array_equal(.9*(x@(k>0)),raw['capacity']);np.testing.assert_array_equal(.9*(x@raw['exchanges']),raw['capacity']);assert np.count_nonzero(raw['exchanges'])==123
 inputs=tensors(raw,y);g=torch.zeros(146,dtype=torch.float64);times=torch.tensor([[0.,1e300]],dtype=torch.float64);a=forward(g,inputs[0],inputs[1],times).numpy();np.testing.assert_array_equal(a[:,0],0);np.testing.assert_allclose(a[:,1],raw['capacity'],atol=1e-14)

def test_original_initial_forward_gradient():
 raw,y,f=setup();g=torch.tensor(f['input_initial'],requires_grad=True);m,t,p=losses(g,*tensors(raw,y));(m+t).backward();np.testing.assert_allclose(p.detach(),f['output_initial_curve'],rtol=1e-12,atol=1e-13);np.testing.assert_allclose(g.grad,f['output_initial_gradient'],rtol=1e-10,atol=1e-13);np.testing.assert_allclose([m.item(),t.item()],[f['output_initial_mse'],f['output_initial_tv']],rtol=1e-12)

def test_finite_difference_gradient_and_native_scaling():
 raw,y,f=setup();g=torch.tensor(f['input_initial'],requires_grad=True);inputs=tensors(raw,y);m,t,p=losses(g,*inputs);(m+t).backward(retain_graph=True);gradient=g.grad.numpy().copy()
 for i in [1,22,75,132]:
  h=.01;lo=g.detach().clone();hi=lo.clone();lo[i]-=h;hi[i]+=h
  ml,tl,_=losses(lo,*inputs);mh,th,_=losses(hi,*inputs);np.testing.assert_allclose(((mh+th-ml-tl)/(2*h)).item(),gradient[i],rtol=3e-6,atol=1e-10)
 native=((p/.9-inputs[-1]/.9)**2).mean()+t/.81
 np.testing.assert_allclose((.81*native).item(),(m+t).item(),rtol=1e-14)
 nativegrad=torch.autograd.grad(native,g)[0].numpy();np.testing.assert_allclose(.81*nativegrad,gradient,rtol=1e-12,atol=1e-15)


def test_first_nesterov_step():
 raw,y,f=setup();energy,_,history,_=reconstruct(raw,y,f['input_initial'],max_epochs=1);np.testing.assert_allclose(energy,f['input_initial']-15000*f['output_initial_gradient'],rtol=1e-13,atol=1e-10);assert history.shape==(1,2)

def test_public_only_initialization():
 raw,y,f=setup();g=initial_energy(raw);np.testing.assert_allclose(g,f['input_initial'],rtol=3e-8,atol=.00025)

def test_complete_fresh_output_parity():
 raw,y,f=setup()
 with np.load(ROOT/'output/reconstruction.npz') as a:
  np.testing.assert_allclose(a['reconstruction'][0],f['output_auxiliary'],rtol=2e-6,atol=.015);np.testing.assert_allclose(a['uptake'][0],f['output_uptake'],rtol=2e-6,atol=2e-6)
