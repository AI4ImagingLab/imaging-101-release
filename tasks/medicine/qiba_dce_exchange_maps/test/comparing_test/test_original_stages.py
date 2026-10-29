from pathlib import Path
import numpy as np
import pytest
from scipy.linalg.blas import dsyrk
from src.preprocessing import signal_to_concentration
from src.physics_model import conv,julia_square_sum,prescott_transpose_product
from src.solvers import fit

ROOT=Path(__file__).resolve().parents[2]

@pytest.fixture(scope='module')
def source():
    with np.load(ROOT/'test/fixtures/original_stages.npz',allow_pickle=False) as f:
        return {k:f[k] for k in f.files}

def test_source_preprocessing(source):
    r1,ct=signal_to_concentration(source['input_signal_curves'],np.ones(32))
    np.testing.assert_allclose(r1,source['output_r1_curves'],rtol=1e-10,atol=1e-11)
    np.testing.assert_allclose(ct,source['input_concentration'],rtol=1e-10,atol=1e-11)

@pytest.mark.parametrize('steps',[1,2,5,10])
def test_frozen_early_updates(source,steps):
    rates,_,_=fit(source['input_concentration'],source['input_time_minutes'],source['input_plasma'],max_iter=steps)
    np.testing.assert_allclose(rates,source[f'output_update_{steps}'],rtol=1e-6,atol=1e-8)

def test_original_forward_stages(source):
    t=source['input_physics_time']
    np.testing.assert_allclose(conv(np.ones(41),np.array([.4]),t)[0],source['output_constant_convolution'],rtol=1e-10,atol=1e-11)
    np.testing.assert_allclose(conv(t,np.array([.4]),t)[0],source['output_linear_convolution'],rtol=1e-10,atol=1e-11)

def test_source_jacobian(source):
    p=source['output_raw_rates'][source['input_curve_indices'][0]]
    def residual(x):return x[0]*conv(source['input_plasma'],x[1:2],source['input_time_minutes'])[0]-source['input_concentration'][0]
    j=np.zeros((1321,2))
    for c in range(2):
        h=6.0554544523933395e-6*max(1.,abs(p[c]));up=p.copy();down=p.copy();up[c]+=h;down[c]-=h
        j[:,c]=(residual(up)-residual(down))/(h+h)
    np.testing.assert_allclose(j,source['output_first_jacobian'],rtol=1e-5,atol=1e-7)

def test_independently_captured_native_arithmetic():
    with np.load(ROOT/'test/fixtures/native_case2716_trace.npz',allow_pickle=False) as f:
        for i in range(1,6):
            j=f[f'iteration{i}_j'];r=f[f'iteration{i}_fcur'];lam=f[f'iteration{i}_scalars'][0]
            dtd=julia_square_sum(j.T)
            np.testing.assert_array_equal(dtd,f[f'iteration{i}_dtd'])
            rhs=-prescott_transpose_product(j.T[None],r[None])[0]
            np.testing.assert_array_equal(rhs,f[f'iteration{i}_rhs'])
            jj=dsyrk(1.,j,trans=1);jj[1,0]=jj[0,1];jj[np.diag_indices(2)]+=lam*np.maximum(dtd,1e-6)
            np.testing.assert_array_equal(jj,f[f'iteration{i}_jj'])
