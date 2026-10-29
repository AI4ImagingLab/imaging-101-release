from pathlib import Path
import numpy as np
from scipy import sparse
from src.preprocessing import load_observations,normalize_clock,rotational_displacement
from src.physics_model import accumulate_events,normalized_motion,directional_matrix,derivative_matrices
from src.solvers import reconstruct
ROOT=Path(__file__).resolve().parents[2]
def fixture():return np.load(ROOT/'test/fixtures/upstream_stages.npz',allow_pickle=False)
def test_upstream_all_acquisition_stages():
    q=fixture();e,w,K=load_observations(ROOT/'data/raw_data.npz');tau,dt=normalize_clock(e);d=rotational_displacement(w,K,dt);iwe=accumulate_events(tau,d);ux,uy,m,mask=normalized_motion(d)
    for x,k in [(d[None],'displacement'),(iwe,'iwe'),(iwe*mask/m,'niwe'),(ux,'unit_flowx'),(uy,'unit_flowy'),(m,'flow_magnitude')]:np.testing.assert_allclose(x,q[k],rtol=0,atol=2e-10)
    A=directional_matrix(uy,ux);original=sparse.csr_matrix((q['Adata'],q['Aindices'],q['Aindptr']),shape=A.shape)
    difference=A-original
    assert difference.nnz==0 or np.max(abs(difference.data))<=2e-10
def test_upstream_derivatives_and_laplacian():
    q=fixture();Dy,Dx,L=derivative_matrices((180,240));x=q['derivative_input'].ravel()
    for D,k in [(Dy,'derivative_y'),(Dx,'derivative_x'),(L,'laplacian')]:np.testing.assert_allclose(D@x,q[k],rtol=0,atol=2e-14)
def test_original_first_bregman_outer_update():
    q=fixture();x,state=reconstruct(*load_observations(ROOT/'data/raw_data.npz'),outer_iterations=1)
    np.testing.assert_allclose(state['raw_log_image'].ravel(),q['one_outer'],rtol=0,atol=2e-7)
def test_full_original_TV_and_Laplacian():
    q=fixture()
    for method,key in [('TV','l1'),('Laplacian','l2')]:
        x,_=reconstruct(*load_observations(ROOT/'data/raw_data.npz'),method=method);np.testing.assert_allclose(x,q[key],rtol=0,atol=2e-7)
def test_seconds_to_milliseconds_invariance_full_image():
    e,w,K=load_observations(ROOT/'data/raw_data.npz');x,a=reconstruct(e,w,K);e[:,0]*=1000.;y,b=reconstruct(e,w/1000.,K)
    for k in ['iwe','niwe','displacement']:np.testing.assert_allclose(a[k],b[k],rtol=0,atol=2e-10)
    np.testing.assert_allclose(x,y,rtol=0,atol=2e-10)
def test_rotational_pinhole_center_independent_formula():
    K=np.array([[100.,0.,120.],[0.,100.,90.],[0.,0.,1.]])
    d=rotational_displacement(np.array([0.,1.,0.]),K,.1)
    np.testing.assert_allclose(d[:,90,120],[-10.,0.],atol=0,rtol=0)
    np.testing.assert_allclose(d[0,90,130],-10.1,atol=2e-15,rtol=0)
def test_bilinear_event_conservation_and_integer_impulse():
    e=np.array([[0.,2.,3.,1.],[1.,4.,5.,0.]])
    z=accumulate_events(e,np.zeros((2,8,9)));assert z[3,2]==1 and z[5,4]==-1 and z.sum()==0
    z=accumulate_events(e,np.full((2,8,9),.25));assert abs(z.sum())<1e-15
    assert z[3,2]==.75**2
def test_fractional_coordinates_use_two_dimensional_flow_index():
    e=np.array([[0.,2.2,3.5,1.],[1.,7.,6.,0.]])
    d=np.zeros((2,8,9));d[0]=np.arange(9)[None,:]
    z=accumulate_events(e,d)
    np.testing.assert_allclose(z[3:5,4:6],[[.4,.1],[.4,.1]],rtol=0,atol=1e-15)
    assert abs(z.sum())<1e-15
def test_directional_difference_affine_sign_constant_and_adjoint():
    h,w=9,11;y,x=np.mgrid[:h,:w];ux=np.ones((h,w));uy=np.zeros_like(ux);ux[[0,-1]]=0;ux[:,[0,-1]]=0
    A=directional_matrix(uy,ux);np.testing.assert_allclose((A@x.ravel()).reshape(h,w)[1:-1,1:-1],-1)
    np.testing.assert_allclose(A@np.ones(h*w),0,atol=1e-15)
    for D in [A,*derivative_matrices((h,w))]:
        a=np.sin(np.arange(h*w));b=np.cos(np.arange(h*w));assert abs(np.dot(D@a,b)-np.dot(a,D.T@b))<1e-12
def test_motion_compensation_sharpens_actual_measurements():
    e,w,K=load_observations(ROOT/'data/raw_data.npz');tau,dt=normalize_clock(e);d=rotational_displacement(w,K,dt)
    corrected=accumulate_events(tau,d);raw=accumulate_events(tau,np.zeros_like(d))
    assert np.mean(corrected**2)>1.5*np.mean(raw**2)
