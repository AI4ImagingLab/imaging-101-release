"""Maintainer figures for the real observations and fresh inverse solution."""
import numpy as np
import matplotlib.pyplot as plt

def plot_comparison(stations,data,grid,fresh,reference,station_prediction):
    """Return a four-panel Figure; coordinates are converted from m to km."""
    fig,axes=plt.subplots(1,4,figsize=(17,4),constrained_layout=True)
    im=axes[0].scatter(stations[:,0]/1000,stations[:,1]/1000,c=data,s=12,cmap='viridis')
    axes[0].set_title('Real station disturbance');fig.colorbar(im,ax=axes[0],label='mGal')
    extent=[grid[0,0,0]/1000,grid[0,-1,0]/1000,grid[0,0,1]/1000,grid[-1,0,1]/1000]
    for ax,array,title in zip(axes[1:3],[fresh,reference],['Fresh field at 3 km','Upstream field at 3 km']):
        im=ax.imshow(array,origin='lower',extent=extent,aspect='auto',cmap='viridis',vmin=reference.min(),vmax=reference.max());ax.set_title(title);fig.colorbar(im,ax=ax,label='mGal')
    axes[3].scatter(data,station_prediction,s=8);low=min(data.min(),station_prediction.min());high=max(data.max(),station_prediction.max());axes[3].plot([low,high],[low,high],'k--');axes[3].set_xlabel('Observed (mGal)');axes[3].set_ylabel('Fresh station fit (mGal)');axes[3].set_title('Data fit, not physical truth')
    for ax in axes[:3]:ax.set_xlabel('Local easting (km)');ax.set_ylabel('Local northing (km)')
    return fig
