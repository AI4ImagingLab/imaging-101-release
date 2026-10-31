import numpy as np
from src.physics_model import green_matrices,incident_field,prepare_fft,fft_A,fft_AH,total_field,scattered_field,cg_fft
from src.solvers import linear_kernel,tikhonov
def test_fft_dense_matrix_equivalence_and_complex_adjoint():
    rng=np.random.default_rng(9);n=6;sources=3;c=rng.normal(size=(n,n))*.1+1j*rng.normal(size=(n,n))*.02
    e=rng.normal(size=(n*n,sources))+1j*rng.normal(size=(n*n,sources));v=rng.normal(size=e.shape)+1j*rng.normal(size=e.shape)
    _,gd=green_matrices(np.arange(0,2*np.pi,2*np.pi/5),n);a=np.eye(n*n)-gd.conj()*c.ravel()[None,:];cache=prepare_fft(n,sources)
    np.testing.assert_allclose(fft_A(e,c,cache),a@e,rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(fft_AH(e,c,cache),a.conj().T@e,rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(np.vdot(v,fft_A(e,c,cache)),np.vdot(fft_AH(v,c,cache),e),rtol=1e-12,atol=1e-12)
def test_zero_contrast_zero_scattering_and_incident_phase():
    phi=np.array([0,np.pi/2]);field=incident_field(phi,n=6);total,errors=total_field(np.zeros((6,6)),field)
    np.testing.assert_allclose(total,field,atol=1e-8)
    gs,_=green_matrices(np.array([0,1]),n=6);np.testing.assert_array_equal(scattered_field(np.zeros((6,6)),total,gs),0)
    from src.physics_model import coordinates,WAVENUMBER
    x,y=coordinates(6);np.testing.assert_allclose(field[:,0],np.exp(-1j*WAVENUMBER*x.ravel()),atol=1e-14)
def test_weak_scattering_limit_and_state_equation():
    n=6;phi=np.array([0,1,2]);ei=incident_field(phi,n);gs,gd=green_matrices(np.array([0,1]),n);shape=np.zeros((n,n));shape[2:4,2:4]=1
    discrepancies=[]
    for strength in [1e-3,1e-4]:
        chi=shape*strength;et,errors=total_field(chi,ei)
        true=scattered_field(chi,et,gs);born=scattered_field(chi,ei,gs)
        discrepancies.append(np.linalg.norm(true-born)/np.linalg.norm(born))
        assert np.linalg.norm(et-gd@(chi.ravel()[:,None]*et)-ei)/np.linalg.norm(ei)<2e-8
    assert discrepancies[1]<.11*discrepancies[0]
def test_linear_kernel_flattening_and_tikhonov_stationarity():
    rng=np.random.default_rng(10);g=rng.normal(size=(3,5))+1j*rng.normal(size=(3,5));e=rng.normal(size=(5,2))+1j*rng.normal(size=(5,2));x=rng.normal(size=5)
    k=linear_kernel(g,e);np.testing.assert_allclose(k@x,(g@(x[:,None]*e)).ravel(),atol=1e-14)
    y=rng.normal(size=(3,2));chi=tikhonov(k,y,.03)
    np.testing.assert_allclose(k.conj().T@(k@chi-y.ravel())+.03*chi,0,atol=1e-12)
