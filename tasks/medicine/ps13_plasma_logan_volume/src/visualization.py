"""Actual PET comparisons and quantitative plot export."""
from pathlib import Path
import os
import numpy as np
import matplotlib.pyplot as plt

def metrics(a,b):
    a=np.asarray(a,np.float64).ravel();b=np.asarray(b,np.float64).ravel();span=np.ptp(b);na=np.linalg.norm(a);nb=np.linalg.norm(b)
    if span<=0 or nb<=0:raise ValueError("Degenerate reference")
    return {"ncc":float(np.dot(a,b)/(na*nb)) if na else 0.,"nrmse":float(np.sqrt(np.mean((a-b)**2))/span)}

def save(fig,name):
    directory=os.environ.get("IMAGING_FIGURE_DIR")
    if directory:
        root=Path(directory);root.mkdir(parents=True,exist_ok=True)
        for extension in ("png","svg","pdf"):fig.savefig(root/(name+"."+extension),dpi=170,bbox_inches="tight")
    return fig

def maps(arrays,titles,name,units,cmap="viridis",shared=False,symmetric=False):
    fig,axes=plt.subplots(1,len(arrays),figsize=(4.2*len(arrays),4),layout="constrained",squeeze=False)
    lo=min(float(np.min(a)) for a in arrays);hi=max(float(np.max(a)) for a in arrays)
    for i,(ax,a,title) in enumerate(zip(axes[0],arrays,titles)):
        low,high=(lo,hi) if shared else (float(np.min(a)),float(np.max(a)))
        signed=symmetric[i] if isinstance(symmetric,(list,tuple)) else symmetric
        if signed:high=max(abs(low),abs(high),1e-14);low=-high
        selected=cmap[i] if isinstance(cmap,(list,tuple)) else cmap
        im=ax.imshow(a,origin="lower",cmap=selected,vmin=low,vmax=high,rasterized=True)
        ax.set(title=title,xlabel="Native column x",ylabel="Native row y")
        fig.colorbar(im,ax=ax,label=units[i] if isinstance(units,(list,tuple)) else units,shrink=.8)
    return save(fig,name)

def arterial(data,plasma,start,duration):
    t=data["blood_time"][0]/60;p=data["plasma_activity"][0]/1000;fraction=data["parent_fraction"][0];mid=start+.5*duration
    fig,axes=plt.subplots(1,3,figsize=(14,4),layout="constrained")
    for ax,limit in [(axes[0],(0,90)),(axes[1],(0,6))]:
        ax.plot(t,p,"o-",label="Measured plasma");ax.plot(t,p*fraction,"s--",label="Measured parent plasma");ax.plot(mid,plasma/1000,".",label="Frame input")
        ax.set(xlim=limit,xlabel="Minutes after injection",ylabel="kBq/mL",title="Arterial input" if limit[1]==90 else "Early bolus sampling");ax.legend(fontsize=8);ax.grid(alpha=.2)
    axes[2].plot(t,fraction,"o-");axes[2].set(xlabel="Minutes after injection",ylabel="Parent tracer fraction",ylim=(0,1.05),title="Measured metabolite correction");axes[2].grid(alpha=.2)
    return save(fig,"02_arterial_calibration")

def transformed_fits(x,y,w,volume,offset,pixel_indices):
    fig,axes=plt.subplots(2,2,figsize=(10,8),layout="constrained")
    for ax,index in zip(axes.ravel(),pixel_indices):
        xx=x[index,:,1];yy=y[index];order=np.argsort(xx);ax.scatter(xx,yy,c=np.arange(len(xx)),cmap="viridis",s=24,label="Measured late frames")
        ax.plot(xx[order],offset[index]+volume[index]*xx[order],color="tab:red",label=f"Fresh slope {volume[index]:.3f}")
        ax.set(title=f"Observation-selected voxel {index}",xlabel="Integrated plasma / tissue (min)",ylabel="Integrated tissue / tissue (min)");ax.legend(fontsize=8);ax.grid(alpha=.2)
    return save(fig,"03_late_time_model_fits")

def activity_volume(mean_activity,volume,mask):
    fig,axes=plt.subplots(1,3,figsize=(14,4),layout="constrained")
    for ax,a,title,unit in [(axes[0],mean_activity/1000,"Late observed activity","kBq/mL"),(axes[1],volume,"Fresh distribution volume","mL plasma / mL tissue")]:
        im=ax.imshow(a,origin="lower",cmap="viridis",rasterized=True);ax.set(title=title,xlabel="Native column x",ylabel="Native row y");fig.colorbar(im,ax=ax,label=unit,shrink=.8)
    axes[2].scatter(mean_activity[mask]/1000,volume[mask],s=3,alpha=.25,rasterized=True)
    axes[2].set(xlabel="Late activity (kBq/mL)",ylabel="Distribution volume (mL/mL)",title="Activity and kinetic contrast differ");axes[2].grid(alpha=.2)
    return save(fig,"05_activity_and_kinetic_contrast")

def residual_figure(residual,y,w,mask,volume):
    local=np.sqrt(np.sum(w*residual**2,axis=1)/np.sum(w*y**2,axis=1));image=np.zeros(mask.shape);image[mask]=local
    fig,axes=plt.subplots(1,3,figsize=(14,4),layout="constrained")
    im=axes[0].imshow(image[0],origin="lower",cmap="magma",rasterized=True);axes[0].set(title="Late-model relative residual",xlabel="Native column x",ylabel="Native row y");fig.colorbar(im,ax=axes[0],label="relative weighted RMS",shrink=.8)
    axes[1].hist(local,bins=60,color="#4477aa");axes[1].set(title="All valid voxels",xlabel="Relative weighted RMS",ylabel="Voxels");axes[1].set_yscale("log")
    axes[2].hist(volume[mask],bins=60,color="#228833");axes[2].axvline(0,color="tab:red",ls="--");negative=int(np.count_nonzero(volume[mask]<0));axes[2].set(title=f"Unclipped estimates: {negative} below zero",xlabel="Distribution volume (mL/mL)",ylabel="Voxels")
    return save(fig,"06_model_residuals_and_negative_estimates"),local

def late_sensitivity(maps_by_start,rows):
    fig,axes=plt.subplots(2,3,figsize=(13,8),layout="constrained")
    low=min(float(a.min()) for a in maps_by_start.values());high=max(float(a.max()) for a in maps_by_start.values())
    for ax,(start,a) in zip(axes.ravel(),maps_by_start.items()):
        im=ax.imshow(a[0],origin="lower",cmap="viridis",vmin=low,vmax=high,rasterized=True);ax.set(title=f"Fresh late start {start} min",xlabel="Native column x",ylabel="Native row y");fig.colorbar(im,ax=ax,label="mL/mL",shrink=.8)
    axes[1,1].plot([r["start"] for r in rows],[r["median"] for r in rows],"o-");axes[1,1].set(title="Actual late-window sensitivity",xlabel="Late start (min)",ylabel="Median V (mL/mL)");axes[1,1].grid(alpha=.2)
    axes[1,2].plot([r["start"] for r in rows],[r["nrmse_vs_default"] for r in rows],"o-");axes[1,2].set(title="Change from selected default",xlabel="Late start (min)",ylabel="Dynamic-range NRMSE");axes[1,2].grid(alpha=.2)
    return save(fig,"07_late_window_sensitivity")

def assumption_sensitivity(rows):
    fig,axes=plt.subplots(1,3,figsize=(14,4.4),layout="constrained");labels=[r["label"] for r in rows];x=np.arange(len(rows))
    for ax,key,title in zip(axes,["median","nrmse_vs_default","seconds"],["Actual volume estimates","Change from selected default","Measured reconstruction runtime"]):
        ax.bar(x,[r[key] for r in rows],color="#4477aa");ax.set_xticks(x,labels,rotation=25,ha="right");ax.set(title=title,ylabel={"median":"Median V (mL/mL)","nrmse_vs_default":"Dynamic-range NRMSE","seconds":"seconds"}[key]);ax.grid(axis="y",alpha=.2)
    return save(fig,"08_input_and_quadrature_sensitivity")
