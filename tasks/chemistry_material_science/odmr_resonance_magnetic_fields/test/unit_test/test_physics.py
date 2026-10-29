from pathlib import Path
from fractions import Fraction
import numpy as np
from src.physics_model import model_jac,physical_fields,fma
from src.solvers import gauss_jordan,fit
T=Path(__file__).resolve().parents[2]
def test_independent_jacobian():
 with np.load(T/'test/fixtures/independent_physics.npz',allow_pickle=False) as z:
  _,j=model_jac(z['frequency_ghz'],z['source_parameters'])
  assert np.linalg.norm(j-z['complex_step_jacobian'])/np.linalg.norm(j)<1e-7
def test_known_spin_one_signs():
 gamma=28.024e-6;remanence=7.;odd=-900.
 signed=np.array([remanence+odd,remanence-odd]);split=2*gamma*signed*np.array([-1,1])
 p=np.zeros((2,2,1,6),dtype='float32');p[:,0,0,0]=2.87-split/2;p[:,1,0,0]=2.87+split/2
 fields=physical_fields(p,1,1)[0,:,0,0]
 np.testing.assert_allclose(fields[:2],[remanence,odd],atol=.015)
 np.testing.assert_allclose(fields[2:],[2.87,2.87],atol=3e-7)
def test_linear_system_backward_error():
 rng=np.random.default_rng(27);a=rng.normal(size=(32,6,6)).astype('float32')
 a=a@a.swapaxes(1,2)+np.eye(6,dtype='float32')*3;b=rng.normal(size=(32,6)).astype('float32')
 x=gauss_jordan(a.copy(),b.copy())
 assert np.max(np.linalg.norm((a@x[...,None])[...,0]-b,axis=1)/(np.linalg.norm(a,axis=(1,2))*np.linalg.norm(x,axis=1)+np.linalg.norm(b,axis=1)))<1e-6
def test_lorentzian_limits_and_symmetry():
 x=np.linspace(2.82,2.92,101);p=np.array([[2.87,.001,.01,.01,.01,0.]])
 y,j=model_jac(x,p)
 np.testing.assert_allclose(y,y[:,::-1],atol=1e-13)
 assert np.max(y)<1 and np.min(y)>.96
 p[:,2:5]=0;np.testing.assert_array_equal(model_jac(x,p)[0],np.ones((1,101)))
def test_constraints_and_objective():
 x=np.linspace(2.837,2.850,51,dtype='float32');p=np.array([[2.844,.001,.01,.012,.014,0]],dtype='float32')
 y=model_jac(x,p)[0];guess=p.copy();guess[:,0]+=.001
 q,sse,_=fit(x,y,guess,10000)
 assert sse[0]<np.sum((model_jac(x,guess)[0]-y)**2)
 assert .0001<=q[0,1]<=.005 and np.all(q[0,2:5]>=.003)
def test_double_rounding_caveat():
 a=np.float32(1+2**-23);b=np.float32((1-2**-23)*2**-24);c=np.float32(1+2**-23)
 exact=Fraction(float(a))*Fraction(float(b))+Fraction(float(c))
 approx=fma(np.array(a),np.array(b),np.array(c));neighbors=[np.nextafter(approx,np.float32(-np.inf)),approx,np.nextafter(approx,np.float32(np.inf))]
 correct=min(neighbors,key=lambda v:(abs(Fraction(float(v))-exact),int(v.view('uint32')&1)))
 assert approx!=correct
