"""Explicit truncated spectral operators used by the empirical peptide-mixture inverse.
Adapted from SSCalcPy, CC-BY-NC-SA4.0; see NOTICE.
"""
import numpy as np
def spectral_svd(A):
    u,s,vh=np.linalg.svd(A,full_matrices=True)
    S=np.zeros((u.shape[0],vh.shape[0]));S[:len(s),:len(s)]=np.diag(s)
    return u,S,vh
def structure_estimate(F,q,decomposition,rank):
    u,S,vh=decomposition
    return F @ vh.T[:,:rank] @ np.linalg.pinv(S[:rank,:rank]) @ u[:,:rank].T @ q
def spectral_refit(decomposition,rank):
    u,S,vh=decomposition
    return (u[:,:rank] @ S[:rank,:rank] @ vh[:rank,:])[:,0]
def rmsd(a,b):return float(np.sqrt(np.mean((a-b)**2)))
