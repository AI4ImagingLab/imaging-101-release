"""Input-only reference construction and gene selection.
Adapted from YMa-lab/CARDspa R/CARD.prop.R and R/utilities.R.
Copyright CARD contributors; GPL-3.0. Exact provenance in test/validation.md.
"""
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist


def reference_basis(counts, types, samples, n_types=20):
    """Mean library sizes times sample-averaged cell-type gene frequencies."""
    counts=np.asarray(counts,dtype=np.float64)
    totals=counts.sum(axis=0)
    active=counts.sum(axis=1)>0
    counts=counts[active]
    basis=np.zeros((counts.shape[0],n_types),dtype=np.float64)
    for k in range(n_types):
        frequencies=[];library_sizes=[]
        for sample in np.unique(samples):
            idx=(types==k)&(samples==sample)
            if not np.any(idx):continue
            mean=counts[:,idx].mean(axis=1)
            summed=mean*np.count_nonzero(idx)
            frequencies.append(summed/summed.sum())
            library_sizes.append(totals[idx].sum()/np.count_nonzero(idx))
        if not frequencies:raise ValueError("Every declared cell type needs reference cells")
        basis[:,k]=np.mean(frequencies,axis=0)*np.mean(library_sizes)
    return basis,active


def informative_genes(basis,sc_counts,types,common):
    """Natural-log fold change >1.25; remove top 1% SC dispersion genes."""
    chosen=np.zeros(basis.shape[0],dtype=bool)
    for k in range(basis.shape[1]):
        other=np.mean(np.delete(basis,k,axis=1),axis=1)
        fc=np.log(basis[:,k]+1e-6)-np.log(other+1e-6)
        chosen|=(fc>1.25)&(basis[:,k]>0)
    idx=np.flatnonzero(chosen&common)
    counts=sc_counts[idx]
    dispersion=[]
    with np.errstate(invalid="ignore",divide="ignore"):
        for k in range(basis.shape[1]):
            z=counts[:,types==k]
            if z.shape[1]>1:dispersion.append(np.var(z,axis=1,ddof=1)/np.mean(z,axis=1))
        ds=np.stack(dispersion,axis=1)
        n=np.sum(np.isfinite(ds),axis=1)
        mean=np.nansum(ds,axis=1)/n
    cutoff=np.nanquantile(mean,.99,method="linear")
    return idx[mean<cutoff]


def prepare_data(path):
    """Return normalized full-support data, basis, kernel, identifiers and scales."""
    with np.load(Path(path),allow_pickle=False) as z:
        C=z["spatial_counts"][0].astype(np.float64)
        S=z["single_cell_counts"][0].astype(np.float64)
        types=z["cell_type_index"][0];samples=z["cell_sample_index"][0]
        xy=z["spatial_coordinates"][0].copy()
        gn=z["spatial_gene_names"][0];sn=z["single_cell_gene_names"][0]
    if C.shape!=(25753,428) or S.shape!=(19736,1926):raise ValueError("Unexpected native observations")
    if np.any(C<0) or np.any(S<0) or not np.all(np.isfinite(xy)):raise ValueError("Invalid observations")
    # Exact author QC is defined on observed counts, before any inverse fit.
    qc=(C>0).sum(axis=1)>5
    qc_counts=C[qc]
    if not np.all((qc_counts.sum(axis=0)>=100)&(qc_counts.sum(axis=0)<=1e6)):
        raise ValueError("Native binding requires all 428 source-supported spots")
    B,active=reference_basis(S,types,samples)
    sn=sn[active];S=S[active]
    common=np.isin(sn,gn[qc])&np.array(["mt-" not in name for name in sn])
    selected=informative_genes(B,S,types,common)
    # Source C.UTF-8 R collation is case-insensitive for these ASCII gene names.
    # Match both acquisition modalities after preserving that order.
    selected=selected[np.argsort(np.char.lower(sn[selected]),kind="stable")]
    genes=sn[selected];lookup={name:i for i,name in enumerate(gn)}
    Csel=C[[lookup[name] for name in genes]]
    if not np.all(Csel.sum(axis=0)>0):raise ValueError("Unexpected zero informative-gene spot")
    X=Csel/Csel.sum(axis=0)
    B=B[selected]
    mean_X=float(X.mean());mean_B=float(B.mean())
    norm=xy-xy.min(axis=0)
    norm/=norm.max()
    W=np.exp(-cdist(norm,norm,"sqeuclidean")/(2*.1**2))
    np.fill_diagonal(W,0)
    return {"X":X*.1/mean_X,"basis":B*.1/mean_B,"kernel":W,
            "coordinates":xy,"genes":genes,"selected_counts":Csel,
            "mean_X":mean_X,"mean_B":mean_B,"reference_basis":B}
