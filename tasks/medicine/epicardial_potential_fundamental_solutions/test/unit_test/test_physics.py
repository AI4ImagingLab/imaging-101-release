import numpy as np
from src.physics_model import kernel_matrices,vertex_normals
from src.solvers import ridge_coefficients
def test_kernel_normal_derivative_and_harmonicity():
    x=np.array([[1.,2.,3.]]);s=np.array([[0.,0.,0.],[4.,-2.,1.]])
    normal=np.array([[0.,0.,1.]])
    k,d=kernel_matrices(x,s,normal);eps=1e-4
    kp=kernel_matrices(x+eps*normal,s)[0];km=kernel_matrices(x-eps*normal,s)[0]
    np.testing.assert_allclose((kp-km)/(2*eps),d,rtol=1e-8,atol=1e-10)
    lap=0
    for axis in np.eye(3):lap=lap+(kernel_matrices(x+eps*axis,s)[0]-2*k+kernel_matrices(x-eps*axis,s)[0])/eps**2
    np.testing.assert_allclose(lap,0,atol=1e-8)
def test_svd_ridge_normal_equation_and_adjoint():
    a=np.array([[1.,2.],[3.,5.],[2.,-1.]])
    u,s,vt=np.linalg.svd(a,full_matrices=False);b=np.array([2.,1.,3.]);l=.2
    coef=ridge_coefficients(u,s,vt,b,l)
    np.testing.assert_allclose((a.T@a+l*l*np.eye(2))@coef,a.T@b,rtol=1e-12)
    x=np.array([1.,-2.]);y=np.array([3.,4.,-5.])
    np.testing.assert_allclose(np.dot(a@x,y),np.dot(x,a.T@y),rtol=1e-14)
