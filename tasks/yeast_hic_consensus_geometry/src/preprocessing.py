"""Native Hi-C counts and explicit iterative visibility correction.

Adapted from iced ICE_normalization, BSD-3-Clause, The iced developers.
The normalization uses observations only; it never sees inferred coordinates.
"""
from pathlib import Path
import numpy as np
from scipy import sparse

N_BEADS=350

def load_data(path):
    """Load the four batch-first public arrays without pickle or modification."""
    with np.load(Path(path),allow_pickle=False) as f:
        out={k:f[k].copy() for k in ['contact_counts','bead_support','chromosome_index','locus_bounds']}
    if out['contact_counts'].shape!=(1,N_BEADS,N_BEADS) or out['contact_counts'].dtype!=np.int64:
        raise ValueError('contact_counts must be int64 (1,350,350)')
    if out['bead_support'].shape!=(1,N_BEADS) or out['bead_support'].dtype!=np.bool_:
        raise ValueError('bead_support must be bool (1,350)')
    if out['chromosome_index'].shape!=(1,N_BEADS) or out['locus_bounds'].shape!=(1,N_BEADS,2):
        raise ValueError('Incorrect locus metadata shapes')
    c=out['contact_counts'][0]
    if np.any(c<0) or np.any(np.tril(c)!=0):
        raise ValueError('Counts must be nonnegative and strictly upper triangular')
    if not np.array_equal(out['bead_support'][0],c.sum(0)+c.sum(1)>0):
        raise ValueError('Support disagrees with observed contacts')
    return out

def build_pairs(counts):
    """Return row/column indices and float64 counts for every positive pair."""
    c=np.asarray(counts)
    if c.ndim!=2 or c.shape[0]!=c.shape[1] or not np.all(np.isfinite(c)) or np.any(c<0):
        raise ValueError('Expected a finite nonnegative square count matrix')
    row,col=np.nonzero(c)
    if np.any(row>=col):
        raise ValueError('Only unique upper-triangle pairs may be stored')
    return row,col,c[row,col].astype(np.float64)

def ice_normalize(counts,max_iter=3000,eps=1e-4):
    """Equalize locus visibility while preserving total counts.

    Return normalized upper triangle, dimensionless bias vector, and iteration
    diagnostics. Unobserved loci receive the neutral bias1; they never occur in
    the likelihood. Source NaN at these loci is represented by public support.
    The arithmetic and rescaling order follow the original sparse ICE routine.
    """
    row,col,values=build_pairs(counts)
    n=np.asarray(counts).shape[0]
    x=sparse.coo_matrix((values.copy(),(row,col)),shape=(n,n))
    total=x.sum();bias=np.ones((n,1));old=None;history=[]
    if total<=0:raise ValueError('No observed contacts')
    for iteration in range(max_iter):
        marginal=x.sum(axis=0)+x.sum(axis=1).T-x.diagonal()
        change=np.asarray(marginal).reshape((n,1))
        change/=change[change!=0].mean()
        change[change==0]=1
        bias*=change
        x.data/=change[x.row,0]
        x.data/=change[x.col,0]
        bias*=np.sqrt(x.sum()/total)
        x*=total/x.sum()
        delta=float(np.abs(old-bias).sum()) if old is not None else None
        history.append((iteration+1,delta))
        if old is not None and delta<eps:break
        old=bias.copy()
    absent=np.asarray(x.sum(0)+x.sum(1).T).ravel()==0
    bias[absent]=1
    return x.toarray(),bias.ravel(),history

def aggregate_contacts(counts,chromosomes,factor):
    """Sum unique contacts within fixed consecutive bins of each chromosome.

    Keep every source bin, preserve chromosome boundaries, and discard only
    intra-new-bin contacts (which carry no distinct-bead spatial constraint).
    Return coarse upper triangle, coarse chromosome labels, native-to-coarse
    mapping and the number of discarded within-bin contacts.
    """
    if not isinstance(factor,int) or factor<1:raise ValueError('factor must be a positive integer')
    counts=np.asarray(counts);chromosomes=np.asarray(chromosomes);mapping=np.empty(len(chromosomes),dtype=np.int64);coarse=[]
    for chromosome in np.unique(chromosomes):
        idx=np.flatnonzero(chromosomes==chromosome)
        for first in range(0,len(idx),factor):
            mapping[idx[first:first+factor]]=len(coarse);coarse.append(chromosome)
    row,col,n=build_pairs(counts);i=mapping[row];j=mapping[col];keep=i!=j
    out=np.zeros((len(coarse),len(coarse)),dtype=np.int64)
    np.add.at(out,(i[keep],j[keep]),n[keep].astype(np.int64))
    return out,np.asarray(coarse,dtype=np.int16),mapping,int(n[~keep].sum())
