"""Geometry-only P1 interpolation and strict existing-observation interface."""
from pathlib import Path
import numpy as np
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree

SHAPES={'fluorescence':(1,56),'excitation':(1,56),'links':(1,56,2),'nodes':(1,5707,3),'tetrahedra':(1,29448,4),'boundary_vertices':(1,5707),'source_positions':(1,8,3),'detector_positions':(1,8,3),'coarse_nodes':(1,3323,3),'coarse_tetrahedra':(1,17350,4)}
def load_raw(path):
    with np.load(path,allow_pickle=False) as f:
        if set(f.files)!=set(SHAPES):raise ValueError('Unexpected observation keys')
        raw={k:f[k] for k in f.files}
    for k,a in raw.items():
        kind=np.bool_ if k=='boundary_vertices' else np.int64 if k in ('links','tetrahedra','coarse_tetrahedra') else np.float64
        if a.shape!=SHAPES[k] or a.dtype!=kind or not np.isfinite(a).all():raise ValueError('Invalid observation '+k)
    if np.any(raw['fluorescence']<=0) or np.any(raw['excitation']<=0):raise ValueError('Nonpositive intensity')
    for key,n in [('links',8),('tetrahedra',5707),('coarse_tetrahedra',3323)]:
        if np.any(raw[key]<0) or np.any(raw[key]>=n):raise ValueError('Invalid index '+key)
    return {k:a[0] for k,a in raw.items()}

def barycentric_locations(nodes,elements,points,tolerance=1e-10):
    """Locate points in fixed tetrahedra; -1 denotes outside, no resampling."""
    tet=nodes[elements];origin=tet[:,0];inverse=np.linalg.inv(np.transpose(tet[:,1:]-tet[:,:1],(0,2,1)))
    lo=tet.min(axis=1)-tolerance;hi=tet.max(axis=1)+tolerance
    centers=tet.mean(axis=1);radius=np.linalg.norm(tet-centers[:,None],axis=2).max()
    neighborhoods=cKDTree(centers).query_ball_point(points,radius+tolerance)
    indices=np.full(len(points),-1,dtype=np.int64);weights=np.zeros((len(points),4))
    for j,(point,near) in enumerate(zip(points,neighborhoods)):
        candidates=np.asarray(sorted(near),dtype=int)
        candidates=candidates[np.all((point>=lo[candidates])&(point<=hi[candidates]),axis=1)]
        if not len(candidates):continue
        xyz=np.einsum('eij,ej->ei',inverse[candidates],point-origin[candidates])
        bary=np.column_stack((1-xyz.sum(axis=1),xyz));ok=np.flatnonzero(np.min(bary,axis=1)>=-tolerance)
        if len(ok):i=ok[0];indices[j]=candidates[i];weights[j]=bary[i]
    return indices,weights

def interpolation_matrix(nodes,elements,points,nearest_inside_fallback=False):
    """P1 interpolation; optional author's nearest *fine-node* inside fallback."""
    indices,weights=barycentric_locations(nodes,elements,points)
    outside=indices<0
    if outside.any():
        if not nearest_inside_fallback:raise ValueError('Point outside mesh')
        valid=np.flatnonzero(~outside)
        if not len(valid):raise ValueError('No inside interpolation point')
        distances,nearest=cKDTree(points[valid]).query(points[outside])
        chosen=valid[nearest];indices[outside]=indices[chosen];weights[outside]=weights[chosen]
    vertices=elements[indices]
    matrix=coo_matrix((weights.ravel(),(np.repeat(np.arange(len(points)),4),vertices.ravel())),shape=(len(points),len(nodes))).tocsr()
    return matrix,indices,weights,outside

def geometry_operators(raw):
    nodes=raw['nodes'];elements=raw['tetrahedra']
    source,*_=interpolation_matrix(nodes,elements,raw['source_positions'])
    adjoint,ind,weights,_=interpolation_matrix(nodes,elements,raw['detector_positions'])
    filtered=weights.copy();boundary=raw['boundary_vertices'][elements[ind]]
    # Preserve source's sequential removal/renormalization convention.
    for i in range(8):
        for j in range(4):
            if not boundary[i,j]:filtered[i,j]=0;filtered[i]/=filtered[i].sum()
    detector=coo_matrix((filtered.ravel(),(np.repeat(np.arange(8),4),elements[ind].ravel())),shape=(8,len(nodes))).tocsr()
    fine_to_coarse,*_=interpolation_matrix(nodes,elements,raw['coarse_nodes'])
    coarse_to_fine,*_=interpolation_matrix(raw['coarse_nodes'],raw['coarse_tetrahedra'],nodes,True)
    return source.T.tocsc(),detector,adjoint.T.tocsc(),fine_to_coarse,coarse_to_fine
