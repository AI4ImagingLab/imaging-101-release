from pathlib import Path
import numpy as np
import pytest
from src.physics_model import KERNELS,covariance_operators
from src.solvers import factor_observations,negative_log_likelihood,reconstruct
ROOT=Path(__file__).resolve().parents[2]
@pytest.mark.parametrize('kernel',KERNELS)
def test_frozen_original_stages(kernel):
    with np.load(ROOT/'test/fixtures/upstream_stages.npz',allow_pickle=False) as f:
        z,y,e=f['data'].T;theta=f[kernel+'_theta']
        k,g,l,a=factor_observations(z,y,e,theta,kernel)
        for value,key in [(k,'K'),(l,'L'),(a,'alpha')]:np.testing.assert_allclose(value,f[kernel+'_'+key],rtol=1e-10,atol=1e-8)
        nll,gradient=negative_log_likelihood(theta,z,y,e,kernel)
        np.testing.assert_allclose(nll,f[kernel+'_nll'],rtol=1e-10,atol=1e-8)
        np.testing.assert_allclose(gradient,f[kernel+'_nll_gradient'],rtol=1e-10,atol=1e-8)
        kk,gg,d,m=covariance_operators(f['kernel_points'],f['kernel_points'],theta,kernel)
        value=np.stack([kk,gg[0],gg[1],d,m],axis=-1)
        np.testing.assert_allclose(value,f[kernel+'_kernel_derivatives'],rtol=1e-10,atol=1e-8)
def test_complete_pipeline():
    from evaluation.judge import evaluate
    result,_=reconstruct(ROOT/'data/raw_data.npz')
    report=evaluate(result)
    assert report['passed'],report
