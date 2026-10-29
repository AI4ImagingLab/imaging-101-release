"""Local covariance and spatial acquisition mathematics.

MIT-derived mathematical conventions: Casey W. Stark, dachshund commit
6a2aeed196fd08767791b1b3c5a6df8e546d261d. See NOTICE.
"""
import numpy as np
from scipy.spatial import cKDTree

SIGNAL_VARIANCE = 0.05
CORRELATION_LENGTHS = (2.5, 2.5, 2.0)
TABLE_SIZE = 200
TABLE_MAX = 8.0

def kernel_from_q(q, variance=SIGNAL_VARIANCE):
    """Author's linearly interpolated exp(-q), with the original cutoff at q=8."""
    q = np.asarray(q,dtype=np.float64)
    if np.any(q < 0) or not np.isfinite(q).all():
        raise ValueError('Squared separation must be finite and nonnegative')
    step = TABLE_MAX/(TABLE_SIZE-1)
    table = np.exp(-step*np.arange(TABLE_SIZE))
    xx = q*(1.0/step)
    ix = xx.astype(np.int64)
    valid = ix <= TABLE_SIZE-2
    safe = np.minimum(ix,TABLE_SIZE-2)
    value = table[safe]*(safe+1-xx)+table[safe+1]*(xx-safe)
    return variance*np.where(valid,value,0.0)

def pairwise_covariance(a,b, lengths=CORRELATION_LENGTHS, variance=SIGNAL_VARIANCE):
    """Dense small-case covariance; positions have shape (n,3), units Mpc/h."""
    delta = np.asarray(a)[:,None,:]-np.asarray(b)[None,:,:]
    q = (delta[...,0]**2+delta[...,1]**2)/(2*lengths[0]**2)
    q += delta[...,2]**2/(2*lengths[2]**2)
    return kernel_from_q(q,variance)

def sparse_covariance(a,b=None,lengths=CORRELATION_LENGTHS,variance=SIGNAL_VARIANCE,
                      tree_b=None):
    """Exact sparse evaluation of the selected finite-support lookup kernel.

    The general spatial tree only finds potentially nonzero pairs. Covariances
    are evaluated from physical coordinate differences in the source convention.
    """
    a=np.asarray(a,dtype=np.float64)
    b=a if b is None else np.asarray(b,dtype=np.float64)
    scale=np.asarray(lengths)
    ta=cKDTree(a/scale)
    tb=cKDTree(b/scale) if tree_b is None else tree_b
    coo=ta.sparse_distance_matrix(tb,4.0,output_type='coo_matrix')
    for start in range(0,len(coo.data),500000):
        end=min(start+500000,len(coo.data))
        delta=a[coo.row[start:end]]-b[coo.col[start:end]]
        q=(delta[:,0]**2+delta[:,1]**2)/(2*lengths[0]**2)
        q+=delta[:,2]**2/(2*lengths[2]**2)
        coo.data[start:end]=kernel_from_q(q,variance)
    result=coo.tocsr()
    result.eliminate_zeros()
    result.sort_indices()
    return result

def whitened_product(covariance, weights, vector):
    """(I + W C W) vector, without materializing the whitened matrix."""
    return vector+weights*(covariance@(weights*vector))

def map_from_coefficients(positions,coefficients,coordinates,chunk_size=1024,
                          lengths=CORRELATION_LENGTHS,variance=SIGNAL_VARIANCE):
    """Evaluate C(map,data) alpha on every requested point, with bounded memory."""
    tree=cKDTree(np.asarray(positions)/np.asarray(lengths))
    result=np.empty(len(coordinates),dtype=np.float64)
    for start in range(0,len(coordinates),chunk_size):
        end=min(start+chunk_size,len(coordinates))
        cross=sparse_covariance(coordinates[start:end],positions,lengths,variance,tree)
        result[start:end]=cross@coefficients
    return result
