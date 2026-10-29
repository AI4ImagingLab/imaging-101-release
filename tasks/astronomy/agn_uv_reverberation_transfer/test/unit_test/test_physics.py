import numpy as np
from scipy.integrate import quad
from src.physics_model import covariance,unpack
from src.solvers import INITIALS,likelihood
from src.preprocessing import load_data,prepare
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def test_continuum_covariance():
 p=INITIALS[0];t=np.array([0.,10.,100.]);K=covariance(t,np.ones(3,int),t,np.ones(3,int),p)
 np.testing.assert_allclose(K,100*np.exp(-abs(t[:,None]-t[None,:])/100))
def test_top_hat_integral():
 p=INITIALS[0];sigma,tau,lags,widths,scales=unpack(p)
 for i,j,t1,t2 in [(1,2,5.,18.),(2,2,5.,8.),(2,3,7.,12.)]:
  def inner(u):
   if j==1:return np.exp(-abs(t1-u-t2)/tau)
   lo,hi=lags[j-1]+np.array([-.5,.5])*widths[j-1];point=t2-t1+u
   return quad(lambda v:np.exp(-abs(t1-u-t2+v)/tau),lo,hi,points=[point] if lo<point<hi else None,epsabs=1e-9)[0]/widths[j-1]
  if i==1:value=inner(0)
  else:
   lo,hi=lags[i-1]+np.array([-.5,.5])*widths[i-1];value=quad(inner,lo,hi,epsabs=1e-9)[0]/widths[i-1]
  expected=sigma**2*scales[i-1]*scales[j-1]*value
  np.testing.assert_allclose(covariance([t1],[i],[t2],[j],p)[0,0],expected,rtol=1e-8)
def test_symmetry_positive_noise():
 d=prepare(load_data(ROOT/'data/raw_data.npz'));K=covariance(d['time'],d['channel'],d['time'],d['channel'],INITIALS[0]);np.testing.assert_allclose(K,K.T,atol=1e-10);assert np.linalg.eigvalsh(K+np.diag(d['error']**2)).min()>0
 assert np.isfinite(likelihood(INITIALS[0],d))
def test_constant_offsets_marginalized():
 obs=load_data(ROOT/'data/raw_data.npz');shifted=obs.copy();shifted[:,:,1]+=np.array([10,20,30])[:,None]
 np.testing.assert_allclose(likelihood(INITIALS[0],prepare(obs)),likelihood(INITIALS[0],prepare(shifted)),atol=1e-9)
def test_illegal_scales():
 d=prepare(load_data(ROOT/'data/raw_data.npz'));p=INITIALS[0].copy();p[4]=-1;assert likelihood(p,d)<-1e200
