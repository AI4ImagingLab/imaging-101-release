"""Scientific radar review and explicit terrain display; no backend changes."""
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator

def terrain_display(image,profile_x,station_x,surface_elevation,depth_step):
    """Source-compatible post-migration elevation display, not wave migration.

    PCHIP the independent terrain survey to traces, round relative surface
    heights to depth samples, vertically shift each trace, and fill only the
    geometrically unobserved display support with NaN. Required solver output
    remains the finite planar wavefield without display padding or NaNs.
    """
    surface=PchipInterpolator(station_x,surface_elevation)(profile_x)
    shifts=np.round((surface-surface.min())/depth_step).astype(int)
    maximum=int(shifts.max());offset=maximum-shifts
    displayed=np.full((image.shape[0]+maximum,image.shape[1]),np.nan)
    for i,start in enumerate(offset):displayed[start:start+len(image),i]=image[:,i]
    return displayed,float(surface.max()),float(surface.min())

def review(raw,time_ns,x,prepared,fresh,upstream,low_velocity,high_velocity,bandlimited,depth_step,station_x,elevation):
    """Return figures for acquisition, velocity, oracle, bandwidth and terrain."""
    fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained')
    limit=np.percentile(abs(prepared),99)
    items=(raw,prepared,fresh,upstream,fresh-upstream,bandlimited-fresh)
    titles=('Measured complete radargram','Time-zero/background prepared','Fresh Stolt migration','Independent upstream migration','Fresh minus upstream','Bandlimited minus source boundary')
    for i,(ax,a,title) in enumerate(zip(axes.ravel(),items,titles)):
        if i==0:extent=(x[0],x[-1],time_ns[-1],time_ns[0]);ylabel='Recorded time [ns]'
        elif i==1:extent=(x[0],x[-1],(len(a)-1)*(time_ns[1]-time_ns[0]),0);ylabel='Corrected time [ns]'
        else:extent=(x[0],x[-1],(len(a)-1)*depth_step,0);ylabel='Nominal depth [m]'
        scale=limit if i<4 else max(float(abs(a).max()),1e-15)
        im=ax.imshow(a,aspect='auto',extent=extent,cmap='RdBu_r',vmin=-scale,vmax=scale)
        ax.set(title=title,xlabel='Along-profile distance [m]',ylabel=ylabel);fig.colorbar(im,ax=ax,label='Instrument amplitude')
    fig2,axes2=plt.subplots(1,3,figsize=(15,4),layout='constrained')
    # Compare the same nominal velocity for the vertical plotting grid so that
    # changes in focusing can be distinguished from coordinate rescaling.
    for ax,a,label in zip(axes2,(low_velocity,fresh,high_velocity),('Fresh 0.12 m/ns','Fresh 0.14 m/ns','Fresh 0.16 m/ns')):
        im=ax.imshow(a,aspect='auto',extent=(x[0],x[-1],(len(a)-1)*depth_step,0),cmap='RdBu_r',vmin=-limit,vmax=limit)
        ax.set(title=label,xlabel='Along-profile distance [m]',ylabel='Nominal 0.14 m/ns depth grid [m]');fig2.colorbar(im,ax=ax,label='Instrument amplitude')
    terrain,max_elev,min_elev=terrain_display(fresh,x,station_x,elevation,depth_step)
    fig3,ax3=plt.subplots(figsize=(12,4),layout='constrained')
    im=ax3.imshow(terrain,aspect='auto',extent=(x[0],x[-1],max_elev-(len(terrain)-1)*depth_step,max_elev),cmap='RdBu_r',vmin=-limit,vmax=limit)
    ax3.plot(station_x,elevation,'k.',ms=2,label='Independent terrain survey')
    ax3.set(xlim=(x[0],x[-1]),ylim=(max_elev-(len(terrain)-1)*depth_step,max_elev+.1),title='Fresh migrated image displayed in surveyed elevation coordinates',xlabel='Along-profile distance [m]',ylabel='Survey elevation [m]')
    ax3.legend();fig3.colorbar(im,ax=ax3,label='Instrument amplitude')
    return fig,fig2,fig3
