from pathlib import Path
import numpy as np,torch
from src.solvers import INITIAL_LATENT,physical_parameters
from src.preprocessing import load_data
from src.physics_model import PixelModel,objective
ROOT=Path(__file__).resolve().parents[2]

def test_original_prior_initialization_and_warm_adam_update():
    data=load_data(ROOT/'data/raw_data.npz')
    with np.load(ROOT/'test/fixtures/upstream_initialization.npz',allow_pickle=False)as f:
        np.testing.assert_array_equal(INITIAL_LATENT,f['latent'])
        for i in range(3):
            z=torch.nn.Parameter(torch.tensor(INITIAL_LATENT[i]));p=physical_parameters(z)
            np.testing.assert_allclose(p.detach().numpy(),f['physical'][i],rtol=1e-6,atol=1e-6)
            model=PixelModel(data['psfs'][i]);optimizer=torch.optim.Adam([z],lr=.05)
            loss=objective(p,model,torch.tensor(data['images'][i]),torch.tensor(1/np.sqrt(data['inverse_variance'][i])),torch.tensor(~data['mask'][0]));loss.backward();optimizer.step()
            np.testing.assert_allclose(z.detach().numpy(),f['warm_latent'][i],rtol=2e-5,atol=2e-6)
