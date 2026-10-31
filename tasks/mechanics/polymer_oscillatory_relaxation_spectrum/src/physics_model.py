"""Generalized Maxwell kernel; Sachin Shanbhag, MIT, see task LICENSE."""
import numpy as np
def get_kern_mat(s,w):
    hs=np.r_[.5*np.log(s[1]/s[0]),.5*(np.log(s[2:])-np.log(s[:-2])),.5*np.log(s[-1]/s[-2])]
    ws=np.asarray(w)[:,None]*s[None,:]
    return np.vstack([ws**2/(1+ws**2),ws/(1+ws**2)])*hs[None,:]
def kernel_prestore(H,kernMat,G0=0.):
    g=kernMat@np.exp(H);g[:len(g)//2]+=G0;return g
def kernel_D(H,kernMat):
    return kernMat*np.exp(H)[None,:]
