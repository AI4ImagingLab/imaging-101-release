import numpy as np
from scipy.special import eval_hermite,gammaln
from src import physics_model as pm

def test_canonical_Hermite_normalization():
    q=np.linspace(-12,12,1201);a=pm.hermite_amplitudes(q,np.zeros_like(q))
    np.testing.assert_allclose(np.trapezoid(abs(a)**2,q,axis=1),1,rtol=0,atol=1e-10)

def test_Born_known_vacuum_and_single_photon():
    q=np.array([-1.3,-.2,.4,2.1]);theta=np.array([.2,.7,1.6,2.3]);a=pm.hermite_amplitudes(q,theta)
    vacuum=np.zeros((31,31),complex);vacuum[0,0]=1
    one=np.zeros_like(vacuum);one[1,1]=1
    np.testing.assert_allclose(pm.born_probabilities(vacuum,a),np.exp(-q*q)/np.sqrt(np.pi),rtol=1e-13)
    np.testing.assert_allclose(pm.born_probabilities(one,a),2*q*q*np.exp(-q*q)/np.sqrt(np.pi),rtol=1e-13)

def test_Wigner_canonical_negative_origin_and_coherent_phase_sign():
    q=np.linspace(-4,4,161);one=np.zeros((31,31),complex);one[1,1]=1
    W=pm.wigner_function(one,q,q)
    assert abs(W[80,80]+1/np.pi)<1e-14
    alpha=.7+.4j;n=np.arange(31)
    ket=np.exp(-abs(alpha)**2/2)*alpha**n*np.exp(-gammaln(n+1)/2)
    rho=np.outer(ket,ket.conj());rho/=np.trace(rho)
    W=pm.wigner_function(rho,q,q);row,col=np.unravel_index(np.argmax(W),W.shape)
    assert abs(q[col]-np.sqrt(2)*alpha.real)<.026 and abs(q[row]-np.sqrt(2)*alpha.imag)<.026

def test_density_update_preserves_Hermitian_PSD_trace():
    q=np.linspace(-3,3,199);theta=np.linspace(0,2*np.pi,199)
    a=pm.hermite_amplitudes(q,theta);rho=np.eye(31)/31
    for _ in range(3):
        rho,_,_=pm.density_update(rho,a)
        assert np.max(abs(rho-rho.conj().T))<1e-14
        assert abs(np.trace(rho)-1)<1e-14
        assert np.linalg.eigvalsh(rho).min()>-1e-14

def test_displacement_zero_and_adjoint():
    np.testing.assert_array_equal(pm.displacement((8,8),0),np.eye(8))
    D=pm.displacement((8,8),.3+.2j)
    np.testing.assert_allclose(D.conj().T,pm.displacement((8,8),-.3-.2j),atol=1e-14)

def test_quadrature_periodicity_and_phase_shift():
    q=np.array([-.3,.8]);theta=np.array([.2,1.1]);rho=np.diag(np.arange(1,32));rho=rho/np.trace(rho)
    a=pm.hermite_amplitudes(q,theta);b=pm.hermite_amplitudes(-q,theta+np.pi)
    np.testing.assert_allclose(a,b,rtol=1e-12,atol=1e-12)
