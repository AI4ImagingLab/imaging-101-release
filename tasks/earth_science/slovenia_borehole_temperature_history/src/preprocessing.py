"""Measured-profile handling; CIBOR conventions, see NOTICE."""
from pathlib import Path
import numpy as np
from scipy.stats import t as student_t

def ordered_product(a,b):
    """Columnwise accumulation preserving the author regression roundoff."""
    vector=b.ndim==1
    b=b[:,None] if vector else b
    out=np.zeros((a.shape[0],b.shape[1]))
    for k in range(a.shape[1]):out+=a[:,k,None]*b[k,None,:]
    return out[:,0] if vector else out

def regression_inverse(matrix):
    """Original scaled-pivot LU factorization and two triangular solves."""
    a=matrix.copy();n=len(a);scale=1/np.max(abs(a),axis=1);indices=[]
    for j in range(n):
        for i in range(j):
            for k in range(i):a[i,j]-=a[i,k]*a[k,j]
        maximum=-1.;pivot=j
        for i in range(j,n):
            for k in range(j):a[i,j]-=a[i,k]*a[k,j]
            candidate=scale[i]*abs(a[i,j])
            if candidate>=maximum:maximum=candidate;pivot=i
        if pivot!=j:
            a[[j,pivot]]=a[[pivot,j]];scale[pivot]=scale[j]
        indices.append(pivot)
        if a[j,j]==0:a[j,j]=1e-20
        if j!=n-1:a[j+1:,j]*=1/a[j,j]
    result=np.empty_like(a)
    for column in range(n):
        b=np.eye(n)[:,column].copy();first=-1
        for i in range(n):
            ip=indices[i];value=b[ip];b[ip]=b[i]
            if first>=0:
                for j in range(first,i):value-=a[i,j]*b[j]
            elif value!=0:first=i
            b[i]=value
        for i in range(n-1,-1,-1):
            value=b[i]
            for j in range(i+1,n):value-=a[i,j]*b[j]
            b[i]=value/a[i,i]
        result[:,column]=b
    return result

def load_data(path='data/raw_data.npz'):
    with np.load(Path(path),allow_pickle=False) as a:
        return {k:a[k][0] for k in a.files}

def profiles(data):
    boundaries=np.r_[0,np.cumsum(data['counts'])]
    return [(data['depth'][a:b],data['temperature'][a:b]) for a,b in zip(boundaries[:-1],boundaries[1:])]

def detrend(depth,temperature):
    """Return three author perturbation anomalies and intercept/slope/half-widths.

    The original regression returns a Student-t interval half-width, then its
    anomaly routine multiplies that width by two. These are perturbation bands,
    not calibrated 95 percent confidence intervals.
    """
    selected=(depth>=200.)&(depth<=300.)
    z,y=depth[selected],temperature[selected]
    design=np.column_stack([np.ones(len(z)),z])
    inverse=regression_inverse(ordered_product(design.T,design))
    coefficients=ordered_product(ordered_product(inverse,design.T),y)
    projector=ordered_product(ordered_product(design,inverse),design.T)
    sse=ordered_product(y[None,:],ordered_product(np.eye(len(z))-projector,y))[0]
    sigma=np.sqrt(abs(sse)/(len(z)-2))
    # Exact first 1e-5 grid point accepted by original q_tstudent search.
    probability=(1+float(np.float32(.95)))/2
    quantile=np.ceil(student_t.ppf(probability-1e-5,len(z)-2)/1e-5)*1e-5
    widths=quantile*sigma*np.sqrt(np.diag(inverse))
    intercept,slope=coefficients
    central=temperature-intercept-slope*depth
    sign=1 if slope>=0 else -1
    perturbation=-sign*2*widths[0]+2*widths[1]*depth
    return np.column_stack([central+perturbation,central,central-perturbation]),np.r_[coefficients,widths]
