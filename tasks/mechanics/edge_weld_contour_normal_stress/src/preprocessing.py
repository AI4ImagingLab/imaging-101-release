"""Published averaged CMM observations, fixed mask and cubic surface fitting."""
from pathlib import Path
import numpy as np
from scipy.interpolate import bisplrep,bisplev

SHAPES={'surface_samples':(1,2010,3),'fit_mask':(1,2010),'outline':(1,128,3),'mesh_nodes':(1,34826,3),'mesh_elements':(1,30370,8),'surface_node_indices':(1,3166),'rigid_anchor_indices':(1,2)}
E_MPA=195600.
POISSON=.294
FRAME_OFFSET_MM=1.

def load_raw(path):
    with np.load(path,allow_pickle=False) as z:
        if set(z.files)!=set(SHAPES):raise ValueError('Require exactly declared observation/calibration keys')
        raw={k:z[k] for k in z.files}
    for key,a in raw.items():
        dtype=np.bool_ if key=='fit_mask' else np.int64 if key in ('mesh_elements','surface_node_indices','rigid_anchor_indices') else np.float64
        if a.shape!=SHAPES[key] or a.dtype!=dtype or not np.isfinite(a).all():raise ValueError('Invalid '+key)
    for key in ('mesh_elements','surface_node_indices','rigid_anchor_indices'):
        if np.any(raw[key]<0) or np.any(raw[key]>=34826):raise ValueError('Invalid mesh index')
    if len(np.unique(raw['surface_node_indices']))!=3166 or len(np.unique(raw['rigid_anchor_indices']))!=2:raise ValueError('Repeated calibration index')
    return {k:v[0] for k,v in raw.items()}

def fit_surface(raw,knot_spacing=(3.,7.)):
    """Original explicit knot construction, with general FITPACK least squares."""
    sx,sy=knot_spacing
    if not np.isfinite([sx,sy]).all() or min(sx,sy)<=0:raise ValueError('Positive finite knot spacing required')
    lo=raw['outline'].min(axis=0);hi=raw['outline'].max(axis=0)
    tx=np.linspace(lo[0],hi[0],int((hi[0]-lo[0])/sx));ty=np.linspace(lo[1],hi[1],int((hi[1]-lo[1])/sy))
    if len(tx)<3 or len(ty)<3:raise ValueError('Insufficient knots')
    tx=np.insert(tx,0,[lo[0]]*3);tx=np.insert(tx,-1,[hi[0]]*3)
    ty=np.insert(ty,0,[lo[1]]*3);ty=np.insert(ty,-1,[hi[1]]*3)
    samples=raw['surface_samples'][raw['fit_mask']]
    return bisplrep(samples[:,0],samples[:,1],samples[:,2],kx=3,ky=3,tx=tx,ty=ty,task=-1)

def boundary_displacements(raw,tck):
    """Preserve native VTK float32 evaluation storage and six-decimal BC text."""
    query=raw['mesh_nodes'][raw['surface_node_indices'],:2]-FRAME_OFFSET_MM
    height=np.array([bisplev(x,y,tck) for x,y in query],dtype=np.float32)
    return np.array([float('%6.6f'%v) for v in height],dtype=np.float64)

def serialized_geometry(raw):
    """Exact original six-decimal millimetre FE geometry protocol."""
    return np.char.mod('%.6f',raw['mesh_nodes']).astype(np.float64)
