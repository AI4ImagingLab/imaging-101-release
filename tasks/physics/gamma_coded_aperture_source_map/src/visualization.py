"""Scientific diagnostics; importing this module never selects a backend."""
import numpy as np
import matplotlib.pyplot as plt
def metrics(x,reference):
    """Uncentered cosine NCC and dynamic-range NRMSE using float64 accumulation."""
    a=np.asarray(x,dtype=np.float64).ravel();b=np.asarray(reference,dtype=np.float64).ravel()
    den=np.linalg.norm(a)*np.linalg.norm(b);span=np.ptp(b)
    if span<=0 or np.linalg.norm(b)==0:raise ValueError('Nondegenerate reference required')
    return {'ncc':float(np.dot(a,b)/den) if den>0 else 0.,'nrmse':float(np.sqrt(np.mean((a-b)**2))/span)}
def contrast_to_noise(a):
    """Diagnostic peak3x3 mean relative to outer detector-response pixels, signed map."""
    a=np.asarray(a);r,c=np.unravel_index(np.argmax(a),a.shape);rr,cc=np.indices(a.shape);signal=(abs(rr-r)<=1)&(abs(cc-c)<=1);background=(abs(rr-r)>5)|(abs(cc-c)>5)
    return float((a[signal].mean()-a[background].mean())/a[background].std())
def image_panels(arrays,titles,cmap='magma',symmetric=False):
    """Return a labelled figure; caller owns saving and closing."""
    fig,axes=plt.subplots(1,len(arrays),figsize=(4*len(arrays),3.7),layout='constrained')
    for ax,a,title in zip(np.atleast_1d(axes),arrays,titles):
        limit=np.max(np.abs(a));kw={'vmin':-limit,'vmax':limit} if symmetric else {}
        im=ax.imshow(a,cmap=cmap,**kw);ax.set(title=title,xlabel='column / pixel',ylabel='row / pixel');fig.colorbar(im,ax=ax,shrink=.8)
    return fig
