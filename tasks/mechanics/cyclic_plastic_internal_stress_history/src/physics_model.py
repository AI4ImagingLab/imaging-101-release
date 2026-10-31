"""Uniaxial four-component MAFTr closed forms; see NOTICE/source provenance."""
import numpy as np

def internal_states(parameters,plastic_strain):
 q=np.asarray(parameters,dtype=float);eps=np.asarray(plastic_strain,dtype=float)
 k0,k1,k2=q[:3];c=q[3:7];a=q[7:]*np.sqrt(2/3);bar=a[4]
 D=np.r_[0.,np.sign(np.diff(eps))];acc=np.r_[0.,np.cumsum(abs(np.diff(eps)))];kiso=D/k2*(1-(1-k0*k2)*np.exp(-np.sqrt(1.5)*k1*k2*acc))
 ends=np.r_[np.flatnonzero(D[:-1]*D[1:]==-1),len(eps)-1]
 alpha=np.zeros((4,len(eps)));ref=np.zeros(4);epsref=eps[0];lo=0
 for hi in ends:
  sl=slice(lo,hi+1);direction=D[sl];delta=abs(eps[sl]-epsref)
  alpha[:3,sl]=direction[None,:]*a[:3,None]-(direction[None,:]*a[:3,None]-ref[:3,None])*np.exp(-c[:3,None]*delta[None,:])
  g=np.full(len(delta),ref[3]);remaining=delta.copy();negative=direction*ref[3]<-bar;tn=np.zeros(len(delta))
  if np.any(negative):
   tn[negative]=-np.log(a[3]/(a[3]-bar-direction[negative]*ref[3]))/c[3]
   step=np.minimum(remaining[negative],tn[negative]);g[negative]=direction[negative]*(a[3]-bar)-(direction[negative]*(a[3]-bar)-ref[3])*np.exp(-c[3]*step)
   remaining[negative]=np.maximum(remaining[negative]-tn[negative],0.)
  linear=(remaining>0)&((tn>0)|(direction*g<=bar))
  if np.any(linear):
   tl=(bar-direction[linear]*g[linear])/(c[3]*a[3]);g[linear]+=direction[linear]*c[3]*a[3]*np.minimum(remaining[linear],tl);remaining[linear]=np.maximum(remaining[linear]-tl,0.)
  positive=remaining>0
  g[positive]=direction[positive]*(a[3]+bar)-(direction[positive]*(a[3]+bar)-g[positive])*np.exp(-c[3]*remaining[positive])
  alpha[3,sl]=g;ref=alpha[:,hi].copy();epsref=eps[hi];lo=hi+1
 return alpha.T,kiso,np.sqrt(1.5)*alpha.sum(0)+kiso
