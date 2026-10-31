from pathlib import Path
import numpy as np
from src.preprocessing import load_data,prepare
from src.physics_model import *
from src.solvers import reconstruct,reconstruct_frame

ROOT=Path(__file__).resolve().parents[2]

def test_upstream_stages():
    a,y,p,m=prepare(load_data(ROOT/'data/raw_data.npz'))
    with np.load(ROOT/'test/fixtures/upstream_stages.npz',allow_pickle=False) as f:
        np.testing.assert_array_equal(y,f['output_preprocessed'])
        np.testing.assert_array_equal(a,f['output_geometry'])
        np.testing.assert_array_equal(p,f['output_equilibria'])
        for i in range(190):
            np.testing.assert_allclose(diffusion_tensor(p[i]),f['output_coefficients'][i].squeeze(),rtol=0,atol=1e-10)
        lap=diffusion_matrix(diffusion_tensor(p[0]),gradient_matrix(p[0].shape))
        x=f['output_x0'];scale=y[0].max()
        np.testing.assert_allclose((lap@x.ravel()).reshape(x.shape),f['output_g_gradient'],rtol=1e-9,atol=1e-7)
        fg=(a.T@(a@x.ravel()-y[0]/scale)/.025**2).reshape(x.shape)
        np.testing.assert_allclose(fg,f['output_f_gradient'],rtol=1e-9,atol=1e-7)
        loss,_=loss_gradient(x,a,y[0]/scale,lap)
        np.testing.assert_allclose(loss,f['output_f_loss'].item()+10*f['output_g_loss'].item(),rtol=1e-10)
        x3=reconstruct_frame(a,y[0],p[0],m,max_iterations=3,tolerance=0)/scale
        np.testing.assert_allclose(x3,f['output_three_updates'],rtol=1e-9,atol=1e-10)

def test_full_native_pipeline():
    fresh=reconstruct(load_data(ROOT/'data/raw_data.npz'))
    with np.load(ROOT/'evaluation/baseline_reference.npz',allow_pickle=False) as f:
        r=f['reconstruction'][0]
    assert fresh.shape==(190,120,41)
    assert np.linalg.norm(fresh-r)/np.linalg.norm(r)<1e-7
