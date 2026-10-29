"""Maintainer scientific plots and numerical exports; no backend selection."""
from pathlib import Path
import json,hashlib
import numpy as np

def export_figure(fig,directory,name,numeric,caption):
    """Export PNG/SVG/PDF and the exact plotted arrays, with content hashes."""
    destination=Path(directory);destination.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(destination/(name+'.npz'),**numeric)
    for extension in ('png','svg','pdf'):fig.savefig(destination/(name+'.'+extension),dpi=170,bbox_inches='tight')
    record=dict(name=name,caption=caption,source='Fresh local active computation unless explicitly labelled qualified upstream',artifacts={})
    for extension in ('png','svg','pdf','npz'):
        p=destination/(name+'.'+extension);record['artifacts'][extension]=dict(file=p.name,sha256=hashlib.file_digest(p.open('rb'),'sha256').hexdigest())
    (destination/(name+'.json')).write_text(json.dumps(record,indent=2),encoding='utf8')
    return record

def image_panel(ax,image,title,unit='1',signed=False,limit=None):
    """Draw every selected slice sample with explicit physical image coordinates."""
    values=np.asarray(image)
    if signed:
        maximum=float(np.max(np.abs(values))) if limit is None else limit
        kwargs=dict(cmap='RdBu_r',vmin=-maximum,vmax=maximum)
    else:kwargs=dict(cmap='magma',vmin=None,vmax=limit)
    artist=ax.imshow(values,origin='lower',extent=[0,128*1.796875,0,128*1.796875],**kwargs)
    ax.set(xlabel='Column / mm',ylabel='Row / mm',title=title)
    ax.figure.colorbar(artist,ax=ax,label=unit,shrink=.8)
    return artist
