"""Measured scientific plots, with optional maintainer exports."""
from pathlib import Path
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

def metrics(a,b):
    a=np.asarray(a,np.float64).ravel();b=np.asarray(b,np.float64).ravel()
    span=np.ptp(b);na=np.linalg.norm(a);nb=np.linalg.norm(b)
    if span<=0 or nb<=0:raise ValueError("Degenerate reference")
    return {"ncc":float(np.dot(a,b)/(na*nb)) if na else 0.,"nrmse":float(np.sqrt(np.mean((a-b)**2))/span)}

def save(fig,name):
    directory=os.environ.get("IMAGING_FIGURE_DIR")
    if directory:
        root=Path(directory);root.mkdir(parents=True,exist_ok=True)
        for extension in ("png","svg","pdf"):fig.savefig(root/(name+"."+extension),dpi=170,bbox_inches="tight")
    return fig

def images(arrays,titles,name,unit="Pa",cmap="viridis",symmetric=False,shared=True):
    fig,axes=plt.subplots(1,len(arrays),figsize=(4.3*len(arrays),4),layout="constrained",squeeze=False)
    global_low=min(float(np.min(a)) for a in arrays);global_high=max(float(np.max(a)) for a in arrays)
    for index,(ax,a,title) in enumerate(zip(axes[0],arrays,titles)):
        lo,hi=(global_low,global_high) if shared else (float(np.min(a)),float(np.max(a)))
        signed=symmetric[index] if isinstance(symmetric,(list,tuple)) else symmetric
        if signed:hi=max(abs(lo),abs(hi));lo=-hi
        selected_cmap=cmap[index] if isinstance(cmap,(list,tuple)) else cmap
        im=ax.imshow(a,origin="lower",extent=(-3,3,-3,3),cmap=selected_cmap,vmin=lo,vmax=hi,rasterized=True)
        ax.set(title=title,xlabel="x / L0",ylabel="y / L0");fig.colorbar(im,ax=ax,label=unit,shrink=.8)
    return save(fig,name)

def line_profiles(q,truth,upstream,fresh):
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout="constrained")
    for ax,axis,title in zip(axes,[0,1],["Horizontal centerline","Vertical centerline"]):
        for p,label,style in [(truth,"Analytic truth","-"),(upstream,"Original CPU","--"),(fresh,"Fresh local",":")]:
            line=p[p.shape[0]//2,:] if axis==0 else p[:,p.shape[1]//2]
            ax.plot(q,line*1e6,style,label=label)
        ax.set(title=title,xlabel="coordinate / L0",ylabel="Pressure (microPa)");ax.legend();ax.grid(alpha=.2)
    return save(fig,"05_pressure_profiles")

def residuals(history,rx,ry):
    fig,axes=plt.subplots(1,3,figsize=(14,4),layout="constrained")
    axes[0].semilogy(history[:,0],history[:,1]);axes[0].axhline(1e-4,color="tab:red",ls="--",label="relative criterion")
    axes[0].set(xlabel="CG update",ylabel="Implicit relative residual",title="Full native reconstruction");axes[0].legend()
    axes[0].annotate(f"Absolute criterion met\nfinal norm = {history[-1,2]:.3e}",xy=(history[-1,0],history[-1,1]),xytext=(.4,.42),textcoords="axes fraction",arrowprops={"arrowstyle":"->","color":"0.3"},bbox={"facecolor":"white","edgecolor":"0.8","alpha":.9},fontsize=9)
    axes[1].hist(rx.ravel(),bins=80,histtype="step",label="x edges");axes[1].hist(ry.ravel(),bins=80,histtype="step",label="y edges")
    axes[1].set(xlabel="Normalized pressure increment residual",ylabel="Edges",title="Measurement consistency");axes[1].set_yscale("log");axes[1].legend()
    m=np.hypot(rx[:-1],ry[:,:-1]);im=axes[2].imshow(m,origin="lower",extent=(-3,3,-3,3),rasterized=True)
    axes[2].set(title="Local edge inconsistency",xlabel="x / L0",ylabel="y / L0");fig.colorbar(im,ax=axes[2],label="dimensionless")
    return save(fig,"06_solver_and_observation_residuals")

def sensitivity(rows):
    fig,axes=plt.subplots(1,3,figsize=(14,4),layout="constrained")
    labels=[r["label"] for r in rows];x=np.arange(len(rows))
    for ax,key,title in zip(axes,["nrmse","iterations","seconds"],["Actual pressure error","Actual solver work","Measured CPU runtime"]):
        ax.bar(x,[r[key] for r in rows],color=["#4477aa","#228833","#cc6677"][:len(rows)])
        ax.set_xticks(x,labels,rotation=15);ax.set(title=title,ylabel={"nrmse":"Dynamic-range NRMSE","iterations":"CG updates","seconds":"seconds"}[key]);ax.grid(axis="y",alpha=.2)
    return save(fig,"07_stopping_tolerance_sensitivity")

def grid_noise(grid_rows,noise_rows,noisy_error):
    fig,axes=plt.subplots(1,3,figsize=(14,4),layout="constrained")
    axes[0].loglog([r["n"] for r in grid_rows],[r["derivative_error"] for r in grid_rows],"o-",label="Acquisition gradient error")
    axes[0].loglog([r["n"] for r in grid_rows],[r["nrmse"] for r in grid_rows],"s--",label="Pressure NRMSE, fixed stop")
    axes[0].set(title="Sampling and stopping effects",xlabel="Grid points per axis",ylabel="Relative error");axes[0].set_xticks([129,257,1001],["129","257","1001"]);axes[0].xaxis.set_minor_formatter(NullFormatter());axes[0].legend(fontsize=8)
    axes[1].plot([r["noise"]*100 for r in noise_rows],[r["nrmse"] for r in noise_rows],"o-")
    axes[1].set(title="Actual noisy reconstructions, 129²",xlabel="Gradient noise / gradient RMS (%)",ylabel="Dynamic-range pressure NRMSE");axes[1].grid(alpha=.2)
    lim=float(np.max(abs(noisy_error)));im=axes[2].imshow(noisy_error*1e6,origin="lower",extent=(-3,3,-3,3),cmap="RdBu_r",vmin=-lim*1e6,vmax=lim*1e6)
    axes[2].set(title="5% noise: pressure error",xlabel="x / L0",ylabel="y / L0");fig.colorbar(im,ax=axes[2],label="microPa")
    return save(fig,"08_grid_and_noise_experiments")
