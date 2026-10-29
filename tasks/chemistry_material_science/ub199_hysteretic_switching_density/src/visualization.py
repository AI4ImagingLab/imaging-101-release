"""Review plots and independently tested standard metrics; no backend changes."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from .solvers import vari_weights
from .physics_model import design_matrix


def metrics(estimate, reference):
    """Cosine NCC and reference-dynamic-range normalized RMS error."""
    x=np.asarray(estimate).ravel();r=np.asarray(reference).ravel()
    scale=np.ptp(r)
    if scale <= 0 or np.linalg.norm(r) == 0:
        raise ValueError('Reference must have nonzero norm and dynamic range')
    denom=np.linalg.norm(x)*np.linalg.norm(r)
    return dict(ncc=float(x@r/denom) if denom else 0.,
                nrmse=float(np.sqrt(np.mean((x-r)**2))/scale))


def local_consistency(state, result):
    """Moment predictions and rectangle-integral residual for local quadratics.

    No unique globally integrated moment follows from a mixed derivative without
    marginal boundary functions. This diagnostic checks the local acquisition
    model and its derivative/integral identity rather than inventing such data.
    """
    h=state['H'];r=state['Hr'];hc=(h-r)/2;hb=(h+r)/2
    a=design_matrix(h,r);m=state['DM'];dh=state['dH']
    prediction=[];integral_errors=[];s=result['selected_smoothing']
    for j in result['domain_indices']:
        w,use=vari_weights(s,s,0.,s,s,0.,hc,hb,dh,hc[j],hb[j])
        aw=a[use,:6]*np.sqrt(w[:,None]);b=m[use]*np.sqrt(w)
        c=np.linalg.lstsq(aw,b,rcond=0)[0]
        prediction.append(a[j,:6]@c)
        # The signed mixed difference equals the integrated -2*rho exactly
        # for a quadratic surface, including arbitrary marginal functions.
        corners=design_matrix(np.array([h[j]+dh,h[j]+dh,h[j],h[j]]),
                              np.array([r[j]+dh,r[j],r[j]+dh,r[j]]))[:,:6]@c
        measured=corners@[1.,-1.,-1.,1.]
        integral_errors.append(measured-c[5]*dh**2)
    return dict(moment_prediction=np.array(prediction),
                moment_residual=m[result['domain_indices']]-np.array(prediction),
                local_integral_error=np.array(integral_errors))


def density_plot(ax, state, result, title='Fresh signed switching density'):
    """Plot native measured positions, without filling missing observations."""
    j=result['domain_indices'];hc=(state['H'][j]-state['Hr'][j])/2
    hb=(state['H'][j]+state['Hr'][j])/2
    im=ax.scatter(hc*1000,hb*1000,c=result['reconstruction']*1000,
                  s=7,cmap='RdBu_r',norm=TwoSlopeNorm(vmin=-1.5,vcenter=0.,vmax=9.),rasterized=True)
    ax.set(xlabel='Coercivity Bc (mT)',ylabel='Interaction field Bu (mT)',title=title)
    ax.set_xlim(0,120);ax.set_ylim(-170,50)
    return im
