"""Review plots; importing this module does not change the plotting backend."""
import numpy as np
import matplotlib.pyplot as plt

def metrics(estimate: np.ndarray, reference: np.ndarray) -> dict:
    """Standard uncentered NCC and dynamic-range NRMSE on real channel arrays."""
    a=np.asarray(estimate,dtype=np.float64).ravel()
    b=np.asarray(reference,dtype=np.float64).ravel()
    norm=np.linalg.norm(a)*np.linalg.norm(b)
    if norm==0 or np.ptp(b)==0:
        raise ValueError('Undefined metric for zero-norm or constant arrays')
    return dict(ncc_vs_ref=float(np.dot(a,b)/norm),
                nrmse_vs_ref=float(np.sqrt(np.mean((a-b)**2))/np.ptp(b)))

def comparison_plot(holograms: np.ndarray, fresh: np.ndarray,
                    upstream: np.ndarray, uncompensated: np.ndarray):
    """Show acquisition, fresh/reference phase, carrier comparison and errors."""
    fig,axes=plt.subplots(2,3,figsize=(14,8),constrained_layout=True)
    panels=[(holograms[0],'Measured first phase-shift frame [DN]','gray',None),
            (np.angle(uncompensated),'Fresh quadrature: before tilt compensation [rad]','twilight',(-np.pi,np.pi)),
            (np.angle(fresh),'Fresh SOSR: compensated wrapped phase [rad]','twilight',(-np.pi,np.pi)),
            (np.angle(upstream),'Reproduced upstream SOSR wrapped phase [rad]','twilight',(-np.pi,np.pi)),
            (np.abs(fresh),'Fresh interferometric amplitude [DN]','magma',None),
            (np.abs(fresh-upstream),'Fresh minus upstream complex error [DN]','inferno',(0,max(float(np.max(np.abs(fresh-upstream))),1e-12)))]
    for ax,(im,title,cmap,limits) in zip(axes.ravel(),panels):
        kw={} if limits is None else dict(vmin=limits[0],vmax=limits[1])
        artist=ax.imshow(im,cmap=cmap,origin='upper',**kw)
        ax.set_title(title,fontsize=10); ax.set_xlabel('Column [pixel]'); ax.set_ylabel('Row [pixel]')
        fig.colorbar(artist,ax=ax,shrink=.8)
    return fig
