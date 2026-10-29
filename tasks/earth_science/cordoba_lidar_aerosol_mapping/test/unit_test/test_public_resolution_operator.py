"""Mathematical output-resolution operator versus the local direct evaluation."""
import numpy as np
from src.solvers import source_smooth

def test_asymmetric_resolution_has_declared_triangular_form():
    n=30;lower=np.zeros((n,n));upper=np.zeros((n,n))
    for i in range(n):
        neighbours=np.arange(max(0,i-5),min(n,i+6));weight=1/len(neighbours)
        lower[i,neighbours[neighbours<i]]=weight;upper[i,neighbours[neighbours>=i]]=weight
    values=np.stack([np.arange(n)**2,np.ones(n)]).astype(float)
    expected=np.linalg.solve(np.eye(n)-lower,upper@values.T).T
    np.testing.assert_allclose(source_smooth(values),expected,rtol=1e-14,atol=1e-12)
    np.testing.assert_allclose(expected[1],1,atol=1e-14)
