"""Native-axis scientific displays and paired numeric figure provenance."""
from pathlib import Path
import hashlib,json
import numpy as np
import matplotlib.pyplot as plt

def draw_slice(ax,field,slice_index,mask=None,title='',vmin=None,vmax=None,cmap='viridis',norm=None):
    """Native x horizontal/y vertical, no anatomical reorientation or resampling."""
    image=np.asarray(field)[:,:,slice_index].T
    if mask is not None:image=np.ma.array(image,mask=~np.asarray(mask)[:,:,slice_index].T)
    artist=ax.imshow(image,origin='lower',interpolation='nearest',vmin=vmin,vmax=vmax,cmap=cmap,norm=norm)
    ax.set(title=title,xlabel='Native x index',ylabel='Native y index');return artist

def export_figure(fig,name,sources,metadata,figure_dir=None):
    if figure_dir is None:return None
    folder=Path(figure_dir);folder.mkdir(parents=True,exist_ok=True);arrays={key:np.asarray(value) for key,value in sources.items()}
    if any(a.dtype.hasobject for a in arrays.values()):raise ValueError('Numeric source cannot contain Python objects')
    numeric=folder/(name+'_sources.npz');np.savez_compressed(numeric,**arrays);files=[]
    for extension in ['png','svg','pdf']:
        path=folder/(name+'.'+extension);fig.savefig(path,dpi=170,bbox_inches='tight',facecolor='white');files.append(dict(name=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    record=dict(metadata,figure=name,numeric_source=numeric.name,numeric_sha256=hashlib.sha256(numeric.read_bytes()).hexdigest(),arrays={key:dict(shape=list(value.shape),dtype=str(value.dtype)) for key,value in arrays.items()},visuals=files)
    (folder/(name+'_provenance.json')).write_text(json.dumps(record,indent=2,allow_nan=False),encoding='utf-8');return record
