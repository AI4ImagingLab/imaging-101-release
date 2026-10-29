"""Coordinate-aware review plots and traceable numeric figure exports."""
from pathlib import Path
import json
import hashlib
import numpy as np
import matplotlib.pyplot as plt

COLORS=['#2978b5','#e18727','#429963','#bd5394','#8e76bf']

def draw_structure(ax,coordinates,support,chromosomes,title):
    """Plot measured-support beads; lines join only consecutive supported bins."""
    x=np.asarray(coordinates);support=np.asarray(support,dtype=bool)
    for chromosome in np.unique(chromosomes):
        indices=np.flatnonzero((chromosomes==chromosome)&support)
        ax.scatter(*x[indices].T,s=7,color=COLORS[int(chromosome)%len(COLORS)],alpha=.85,label=f'Chr{int(chromosome)+1}')
        for i in indices:
            if i+1<len(x) and support[i+1] and chromosomes[i+1]==chromosome:
                ax.plot(*x[[i,i+1]].T,color=COLORS[int(chromosome)%len(COLORS)],lw=.65,alpha=.7)
    points=x[support];middle=(points.min(0)+points.max(0))/2;radius=max(float(np.ptp(points,axis=0).max())/2,1e-6)
    for setter,c in zip([ax.set_xlim,ax.set_ylim,ax.set_zlim],middle):setter(c-radius,c+radius)
    ax.set_box_aspect((1,1,1));ax.set_xlabel('x [model units]');ax.set_ylabel('y [model units]');ax.set_zlabel('z [model units]');ax.set_title(title,pad=12)
    return ax

def export_figure(fig,name,sources,metadata,figure_dir=None):
    """Optionally save paired PNG/SVG/PDF, numeric NPZ and JSON provenance.

    The notebook always computes and displays figures; exporting is enabled by
    an explicit maintainer environment path and never needed by the solver.
    """
    # Identify chromosome colors once for every geometry figure, including
    # on-screen notebook displays when file export is disabled.
    if fig.axes:
        handles,labels=fig.axes[0].get_legend_handles_labels()
        if labels and all(label.startswith('Chr') for label in labels):
            fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,-.14),ncol=len(labels),frameon=False)
    if figure_dir is None:return None
    directory=Path(figure_dir);directory.mkdir(parents=True,exist_ok=True)
    arrays={k:np.asarray(v) for k,v in sources.items()}
    if any(a.dtype.hasobject for a in arrays.values()):raise ValueError('Numeric figure sources cannot have object dtype')
    np.savez_compressed(directory/(name+'_sources.npz'),**arrays)
    files=[]
    for extension in ['png','svg','pdf']:
        path=directory/(name+'.'+extension)
        fig.savefig(path,dpi=180,bbox_inches='tight',facecolor='white')
        files.append(dict(name=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    numeric=directory/(name+'_sources.npz')
    record=dict(metadata,figure=name,numeric_source=numeric.name,numeric_sha256=hashlib.sha256(numeric.read_bytes()).hexdigest(),arrays={k:dict(shape=list(a.shape),dtype=str(a.dtype)) for k,a in arrays.items()},visuals=files)
    (directory/(name+'_provenance.json')).write_text(json.dumps(record,indent=2,allow_nan=False),encoding='utf-8')
    return record
