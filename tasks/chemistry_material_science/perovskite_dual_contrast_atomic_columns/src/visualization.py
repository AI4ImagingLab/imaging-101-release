"""Scientific diagnostics and traceable figure export; no backend selection."""
from pathlib import Path
import hashlib,json
import numpy as np
from .solvers import translation_vectors,atomic_planes
def ellipse_properties(columns):
    a=np.asarray(columns)
    cov=np.stack((a[:,2],a[:,3],a[:,3],a[:,4]),axis=-1).reshape(-1,2,2)
    values,vectors=np.linalg.eigh(cov)
    return np.sqrt(values[:,1]/values[:,0]),vectors[:,:,1]
def plane_spacings(positions,axis_index=2):
    vectors=translation_vectors(positions);v=np.array(vectors[axis_index]);normal=np.array([v[1],-v[0]])/np.linalg.norm(v)
    planes=atomic_planes(positions,v)
    coordinate=np.array([np.mean(positions[ids]@normal) for ids in planes])
    coordinate=np.sort(coordinate)
    return (coordinate[:-1]+coordinate[1:])*.5,np.diff(coordinate),v
def distance_asymmetries(positions):
    vectors=translation_vectors(positions);records=[]
    for v in vectors:
        xy=[];difference=[]
        for ids in atomic_planes(positions,v):
            for prev,mid,nxt in zip(ids[:-2],ids[1:-1],ids[2:]):
                xy.append(positions[mid]);difference.append(np.linalg.norm(positions[mid]-positions[prev])-np.linalg.norm(positions[nxt]-positions[mid]))
        records.append((np.array(v),np.asarray(xy),np.asarray(difference)))
    return records
def export_figure(fig,directory,stem,numeric,caption):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    files={}
    for extension in ['png','svg','pdf']:
        p=directory/f'{stem}.{extension}';fig.savefig(p,dpi=170,bbox_inches='tight')
        files[extension]=dict(file=p.name,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    p=directory/f'{stem}.npz';np.savez_compressed(p,**numeric)
    files['npz']=dict(file=p.name,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    record=dict(caption=caption,files=files,numeric_keys={k:dict(shape=list(np.asarray(v).shape),dtype=str(np.asarray(v).dtype)) for k,v in numeric.items()})
    (directory/f'{stem}.json').write_text(json.dumps(record,indent=2),encoding='utf8')
    return record
