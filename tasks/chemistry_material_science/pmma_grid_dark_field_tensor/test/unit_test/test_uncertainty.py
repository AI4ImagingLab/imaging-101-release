import numpy as np
from src.physics_model import moments_from_coefficients,mode_ratios,covariance_eigenvalues
from src.visualization import conditional_eigenvalue_uncertainty

def test_conditional_uncertainty_against_amplitude_finite_differences():
    truth=np.array([[[4.]],[[2.]],[[.6]]]);ratios=mode_ratios(truth,np.array([[.7]]))[:,0,0]
    coefficients=np.ones((2,13,13,5));coefficients[1]*=ratios
    c,_,_=moments_from_coefficients(*coefficients)
    pcov=np.broadcast_to(np.eye(5)*1e-6,(2,13,13,5,5))
    actual=conditional_eigenvalue_uncertainty(coefficients,pcov,c)[0,0]
    expected=0.
    for channel in range(2):
        derivative=[]
        for i in range(5):
            plus=coefficients.copy();minus=coefficients.copy();plus[channel,:,:,i]+=1e-6;minus[channel,:,:,i]-=1e-6
            cp,_,_=moments_from_coefficients(*plus);cm,_,_=moments_from_coefficients(*minus)
            derivative.append((covariance_eigenvalues(cp)[0,0,0]-covariance_eigenvalues(cm)[0,0,0])/2e-6)
        expected+=np.linalg.norm(derivative)*1e-3
    np.testing.assert_allclose(actual,expected,rtol=1e-7)
