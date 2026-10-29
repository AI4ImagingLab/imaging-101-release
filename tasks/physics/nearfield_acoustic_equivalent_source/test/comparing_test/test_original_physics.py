from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
import pytest
from src.preprocessing import load_pressure
from src.physics_model import propagator,forward,adjoint,coordinates,SOURCE_SHAPE,MEASUREMENT_SHAPE
from src.solvers import solve,difference,difference_adjoint,data_gradient,project_dual
ROOT=Path(__file__).resolve().parents[2]
torch.set_num_threads(1)
@pytest.fixture(scope='module')
def pressure():return load_pressure(ROOT/'data/raw_data.npz')
def test_public_observation_and_original_forward(pressure):
    with np.load(ROOT/'test/fixtures/upstream_methods.npz',allow_pickle=False) as z:
        np.testing.assert_array_equal(pressure,z['pressure']);np.testing.assert_array_equal(propagator(),z['G'])
def test_original_first_101_updates(pressure):
    q,record=solve(pressure,iterations=101,return_history=True)
    with np.load(ROOT/'test/fixtures/upstream_stages.npz',allow_pickle=False) as z:
        for n in [1,2,10,100,101]:
            for key in ['q','u']:np.testing.assert_allclose(record['history'][n][key],z[f'iteration{n}_{key}'],rtol=2e-5,atol=2e-6)
        np.testing.assert_array_equal(difference(torch.from_numpy(z['case_x'])).numpy(),z['case_diff'])
        np.testing.assert_array_equal(difference_adjoint(torch.from_numpy(z['case_y'])).numpy(),z['case_adjoint'])
        assert record['lambdaa']==float(z['lambdaa']);assert record['sigma']==float(z['sigma'])
@pytest.mark.parametrize('method',['LN','TV','TGV'])
def test_complete_original_pipeline(pressure,method):
    actual,_=solve(pressure,method=method)
    with np.load(ROOT/'test/fixtures/upstream_methods.npz',allow_pickle=False) as z:
        ref=z[method][0,0]
        assert np.linalg.norm(actual-ref)/np.linalg.norm(ref)<2e-5
@pytest.mark.parametrize('order',[1,2])
def test_independent_difference_adjoint(order):
    torch.manual_seed(122);x=torch.randn(1,1,13,17,dtype=torch.complex128);y=torch.randn(1,1,13,17,2,dtype=torch.complex128)
    np.testing.assert_allclose(torch.vdot(difference(x,order).flatten(),y.flatten()),torch.vdot(x.flatten(),difference_adjoint(y,order).flatten()),rtol=1e-13,atol=1e-11)
def test_independent_gradient_directional_derivative():
    torch.manual_seed(123);G=torch.randn(1,1,5,12,dtype=torch.complex128);q=torch.randn(1,1,3,4,dtype=torch.complex128);p=torch.randn(1,1,1,5,dtype=torch.complex128);v=torch.randn_like(q)
    objective=lambda x:torch.linalg.norm(G@x.reshape(1,1,12,1)-p.reshape(1,1,5,1))**2
    finite=(objective(q+1e-6*v)-objective(q-1e-6*v))/2e-6
    analytic=torch.vdot(data_gradient(q,p,G).flatten(),v.flatten()).real
    np.testing.assert_allclose(finite,analytic,rtol=2e-8,atol=2e-8)
def test_forward_hermitian_and_radiating_monopole():
    rng=np.random.default_rng(44);q=rng.normal(size=SOURCE_SHAPE)+1j*rng.normal(size=SOURCE_SHAPE);p=rng.normal(size=MEASUREMENT_SHAPE)+1j*rng.normal(size=MEASUREMENT_SHAPE)
    np.testing.assert_allclose(np.vdot(forward(q),p),np.vdot(q,adjoint(p)),rtol=1e-12)
    q=np.zeros(SOURCE_SHAPE,dtype=complex);q[0,0]=1
    radius=np.linalg.norm(coordinates(MEASUREMENT_SHAPE,z_m=.025),axis=1)
    actual=forward(q).ravel();expected=(1j*2*np.pi*1150*1.2)/(4*np.pi*radius)*np.exp(-1j*2*np.pi*1150/343*radius)
    np.testing.assert_allclose(actual,expected,rtol=1e-7)
def test_zero_boundary_and_second_order_affine_interior():
    y,x=torch.meshgrid(torch.arange(17.),torch.arange(19.),indexing='ij');p=(2*x+3*y)[None,None]
    d=difference(p,2);assert not d[..., :-2,:-2,:].any();assert d.abs().max()>0
def test_dual_complex_ball_projection():
    torch.manual_seed(9);u=torch.randn(2,3,2,dtype=torch.complex128)*10;v=project_dual(u,2.)
    assert torch.linalg.norm(v,dim=-1).max()<=2.000000000000001
    np.testing.assert_allclose(project_dual(v,2.),v,atol=1e-15)
def test_independent_helmholtz_green_identity():
    # Away from its singularity, outgoing exp(-ikr)/r obeys Laplacian g+k²g=0.
    k=2*np.pi*1150/343;x=np.array([.17,.23,.11]);step=1e-5
    g=lambda r:np.exp(-1j*k*np.linalg.norm(r))/np.linalg.norm(r)
    lap=sum((g(x+np.eye(3)[i]*step)-2*g(x)+g(x-np.eye(3)[i]*step))/step**2 for i in range(3))
    assert abs(lap+k*k*g(x))/abs(k*k*g(x))<1e-6
def test_invalid_parameters(pressure):
    for method in ['VN','',None]:
        with pytest.raises(ValueError):solve(pressure,method=method)
    for count in [0,-1,True,1.5]:
        with pytest.raises(ValueError):solve(pressure,iterations=count)
