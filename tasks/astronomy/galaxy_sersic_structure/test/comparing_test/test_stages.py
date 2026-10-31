from pathlib import Path
import numpy as np,torch
from src.preprocessing import load_data
from src.physics_model import PixelModel,objective
ROOT=Path(__file__).resolve().parents[2]
def test_original_pixel_quadrature_psf_likelihood_gradient():
    a=load_data(ROOT/'data/raw_data.npz')
    with np.load(ROOT/'test/fixtures/upstream_stages.npz',allow_pickle=False)as f:
        params=torch.tensor(f['fixed_parameters'],requires_grad=True)
        for i,band in enumerate('grz'):
            model=PixelModel(a['psfs'][i]);np.testing.assert_allclose(model.psf_fft.numpy(),f['PSF_fft_'+band],rtol=2e-5,atol=2e-7)
            fixed=params.detach().clone();fixed[-1]=0
            np.testing.assert_allclose(model.intrinsic(fixed).numpy(),f['stage_fixed_intrinsic_'+band],rtol=2e-5,atol=2e-7)
            np.testing.assert_allclose(model.observed(fixed).numpy(),f['stage_fixed_observed_'+band],rtol=2e-5,atol=2e-7)
            value=objective(params,model,torch.tensor(a['images'][i]),torch.tensor((1/np.sqrt(a['inverse_variance'][i])).astype(np.float32)),torch.tensor(~a['mask'][0]));grad=torch.autograd.grad(value,params)[0]
            np.testing.assert_allclose(value.detach().numpy(),f['objective_'+band],rtol=2e-5,atol=.01)
            np.testing.assert_allclose(grad.numpy(),f['gradient_'+band],rtol=5e-4,atol=.01)
