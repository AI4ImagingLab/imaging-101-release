"""Maintainer plots; display conversion never changes submitted arrays."""
import numpy as np
import matplotlib.pyplot as plt
def rgb_preview(u):
    """Return the 427-row physical RGB preview on [0,1]; input is centered YCbCr."""
    y=u[0,:427]+128.;cb=u[1,:427];cr=u[2,:427]
    return np.clip(np.stack([y+1.402*cr,y-.344136*cb-.714136*cr,y+1.772*cb],axis=-1)/255.,0,1)
def comparison_figure(midpoint,fresh,upstream,tv2):
    """Compare measured midpoint, fresh TV, original TV and diagnostic original TV2."""
    fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    for ax,u,title in zip(axes.ravel(),[midpoint,fresh,upstream,tv2],[
        'Coefficient midpoint / ordinary inverse DCT',
        'Fresh local TV reconstruction — 2000 updates',
        'Independently executed original TV — 2000 updates',
        'Original TV2 — 50 updates (boundary-gradient defect)']):
        ax.imshow(rgb_preview(u));ax.set_title(title,fontsize=11);ax.set_axis_off()
    return fig
