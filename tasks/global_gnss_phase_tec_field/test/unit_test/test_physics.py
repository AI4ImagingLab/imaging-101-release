import numpy as np
from src.physics_model import real_harmonics, slant_factor, design_matrix,temporal_precision,chunk_bounds
from src.solvers import lemke

def test_known_harmonics_and_mapping():
    x=real_harmonics(np.array([0.,np.pi/2]),np.array([np.pi/2,np.pi/2]),degree=1)
    np.testing.assert_allclose(x[:,0],1/np.sqrt(4*np.pi),atol=1e-15)
    np.testing.assert_allclose(x[:,1],np.sqrt(3/(4*np.pi))*np.array([0,1]),atol=1e-15)
    np.testing.assert_allclose(x[:,3],np.sqrt(3/(4*np.pi))*np.array([1,0]),atol=1e-15)
    assert slant_factor(np.array([np.pi/2]))[0] == 1
    assert slant_factor(np.array([np.deg2rad(10)]))[0] > 2

def test_continuous_time_field_and_reference_cancellation():
    times=np.array([0.,900.,1800.,3599.999,3600.,3600.001,85800.])
    n=len(times)
    d=dict(time=times,time_ref=np.zeros(n),mlt1=np.zeros(n),mcolat1=np.full(n,np.pi/2),el=np.full(n,np.pi/2),mlt1_ref=np.zeros(n),mcolat1_ref=np.full(n,np.pi/2),el_ref=np.full(n,np.pi/2))
    A,_=design_matrix(d,degree=0)
    np.testing.assert_allclose(A@np.ones(25),0,atol=1e-15)
    np.testing.assert_allclose(A@np.arange(25),times/3600/np.sqrt(4*np.pi),atol=2e-15)

def test_complementarity_against_independent_active_set():
    M=np.array([[2.,-1.,0.],[-1.,2.,-1.],[0.,-1.,2.]])
    q=np.array([-1.,-.2,.5]);z=lemke(M,q)
    np.testing.assert_allclose(z,[11/15,7/15,0],atol=1e-15)
    assert min(z)>=0 and min(M@z+q)>=-1e-14
    assert abs(z@(M@z+q))<1e-14
    np.testing.assert_array_equal(lemke(np.eye(3),np.ones(3)),np.zeros(3))

def test_sphere_orthonormality_independent_quadrature():
    x,w=np.polynomial.legendre.leggauss(20)
    lon=np.arange(40)*2*np.pi/40
    ll,cc=np.meshgrid(lon,np.arccos(x));basis=real_harmonics(ll.ravel(),cc.ravel(),degree=6)
    quadrature=np.repeat(w,40)*(2*np.pi/40)
    np.testing.assert_allclose(basis.T@(quadrature[:,None]*basis),np.eye(49),rtol=0,atol=2e-14)

def test_temporal_precision_energy_and_open_endpoints():
    c=np.arange(12,dtype=float);R=temporal_precision(degree=1,intervals=2,sigma_rel=.5)
    np.testing.assert_array_equal(R,R.T)
    assert c@R@c==np.sum(np.diff(c.reshape(3,4),axis=0)**2)/.5**2
    np.testing.assert_array_equal(R[:4,-4:],np.zeros((4,4)))
    assert np.linalg.eigvalsh(R).min()>-1e-14

def test_original_chunk_partition_covers_every_ray():
    b=chunk_bounds(504358)
    assert len(b)==168 and b[0]==(0,3002) and b[-1]==(501334,504358)
    assert all(a[1]==c[0] for a,c in zip(b,b[1:]))

def test_forward_adjoint_and_objective_directional_derivative():
    n=9;t=np.arange(n)*600.;z=np.zeros(n)
    d=dict(time=t,time_ref=z,mlt1=t/86400*2*np.pi,mcolat1=np.ones(n),el=np.ones(n),mlt1_ref=z,mcolat1_ref=np.ones(n),el_ref=np.ones(n))
    A,w=design_matrix(d,degree=2);rng=np.random.default_rng(914)
    c=rng.normal(size=225);direction=rng.normal(size=225);y=rng.normal(size=n)
    np.testing.assert_allclose((A@direction)@y,direction@(A.T@y),rtol=1e-14,atol=1e-14)
    def loss(v):return np.dot(w,(A@v-y)**2)
    eps=1e-5;fd=(loss(c+eps*direction)-loss(c-eps*direction))/(2*eps)
    analytic=2*direction@(A.T@(w*(A@c-y)))
    np.testing.assert_allclose(fd,analytic,rtol=2e-9,atol=1e-7)
