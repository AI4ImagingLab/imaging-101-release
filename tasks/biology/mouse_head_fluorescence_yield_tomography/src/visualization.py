"""Scientific slices/metrics and traceable figure export; no backend mutation."""
from pathlib import Path
import hashlib,json,os
import numpy as np
from .preprocessing import barycentric_locations

def metrics(x,y):
    norm=np.linalg.norm(x)
    return {'ncc':float(x@y/norm/np.linalg.norm(y)) if norm else 0.,'dr_nrmse':float(np.sqrt(np.mean((x-y)**2))/np.ptp(y)),'max_abs_mm_inverse':float(np.max(abs(x-y)))}

def slice_field(nodes,elements,values,axis,coordinate,resolution=42):
    """Display-only P1 sample of actual tissue; known exterior remains NaN."""
    other=[j for j in range(3) if j!=axis];a=np.linspace(nodes[:,other[0]].min(),nodes[:,other[0]].max(),resolution);b=np.linspace(nodes[:,other[1]].min(),nodes[:,other[1]].max(),resolution);A,B=np.meshgrid(a,b);points=np.zeros((A.size,3));points[:,axis]=coordinate;points[:,other[0]]=A.ravel();points[:,other[1]]=B.ravel();ind,w=barycentric_locations(nodes,elements,points);inside=ind>=0;out=np.full(A.size,np.nan);out[inside]=(values[elements[ind[inside]]]*w[inside]).sum(axis=1);return A,B,out.reshape(A.shape)

def export_figure(fig,label,numeric):
    """Optionally save paired PNG/SVG/PDF + numerical NPZ/JSON privately."""
    directory=os.environ.get('IMAGING_FIGURE_DIR')
    if not directory:return
    root=Path(directory);root.mkdir(parents=True,exist_ok=True);files={}
    for ext in ['png','svg','pdf']:
        p=root/(label+'.'+ext);fig.savefig(p,dpi=170,bbox_inches='tight');files[ext]={'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
    p=root/(label+'.npz');np.savez_compressed(p,**numeric);files['numeric_npz']={'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
    (root/(label+'.json')).write_text(json.dumps({'label':label,'files':files,'numeric_arrays':{k:{'shape':list(np.asarray(a).shape),'dtype':str(np.asarray(a).dtype)} for k,a in numeric.items()}},indent=2),encoding='utf8')
