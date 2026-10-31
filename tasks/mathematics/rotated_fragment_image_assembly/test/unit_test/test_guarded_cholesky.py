"""Independent small analytic/guard tests for the exact candidate file."""
import numpy as np
import pytest
from src.preprocessing import mahalanobis_squared_spd


def test_diagonal_metric_exact_and_zero():
    c=np.diag([1.,4.,16.]);x=np.array([[1.,2.,4.],[-2.,0.,8.],[0.,0.,0.]])
    np.testing.assert_array_equal(mahalanobis_squared_spd(c,x),[3.,8.,0.])


def test_nondiagonal_metric_known_whitening_and_broadcast():
    l=np.array([[2.,0.,0.],[1.,3.,0.],[-1.,2.,4.]])
    z=np.array([[1.,2.,3.],[0.,-1.,4.],[-2.,3.,1.]])
    x=z@l.T;c=l@l.T
    expected=np.sum(z*z,axis=1)
    np.testing.assert_allclose(mahalanobis_squared_spd(c,x),expected,rtol=1e-15)
    np.testing.assert_allclose(mahalanobis_squared_spd(np.stack([c,c]),np.stack([x,x])),np.stack([expected,expected]),rtol=1e-15)
    for scale in [1e-6,1e6]:
        np.testing.assert_allclose(mahalanobis_squared_spd(c*scale**2,x*scale),expected,rtol=1e-14)


def test_rotated_spd_color_basis_invariance():
    rotation=np.array([[.6,-.8,0.],[.8,.6,0.],[0.,0.,1.]])
    diagonal=np.diag([1.,4.,16.]);x=np.array([[1.,2.,4.],[-2.,0.,8.]])
    covariance=rotation@diagonal@rotation.T
    rotated_vectors=x@rotation.T
    np.testing.assert_allclose(mahalanobis_squared_spd(covariance,rotated_vectors),[3.,8.],rtol=1e-14)


@pytest.mark.parametrize('cov,x',[
    (np.diag([0.,1.,1.]),np.ones((2,3))),
    (np.diag([1.,-1.,1.]),np.ones((2,3))),
    (np.diag([1.,1.,0.]),np.ones((2,3))),
    (np.full((3,3),np.nan),np.ones((2,3))),
    (np.eye(3),np.full((2,3),np.inf)),
    (np.array([[2.,1.,0.],[0.,2.,0.],[0.,0.,2.]]),np.ones((2,3))),
    (np.eye(3).astype(complex)+1j,np.ones((2,3))),
    (np.eye(4),np.ones((2,3))),
    (np.eye(3),np.ones((2,4))),
])
def test_invalid_covariance_or_vectors_rejected(cov,x):
    with pytest.raises((ValueError,ArithmeticError,FloatingPointError)):
        mahalanobis_squared_spd(cov,x)
