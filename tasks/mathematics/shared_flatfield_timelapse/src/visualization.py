"""Scientific comparisons from measured and genuinely computed arrays."""
import numpy as np
import matplotlib.pyplot as plt
from .preprocessing import upsample_field
from .physics_model import quality

def native_summary(images,flatfield,indices=(0,12,50,99)):
    """Gain-only correction, matching author's final is_timelapse=False."""
    field=upsample_field(flatfield,images.shape[-2:])
    mean=[];std=[];views=[]
    for i,image in enumerate(images):
        corrected=image.astype(np.float32)/field
        mean.append(float(corrected.mean()));std.append(float(corrected.std()))
        if i in indices:views.append(corrected.copy())
    return dict(indices=np.asarray(indices),corrected=np.asarray(views),
                mean_by_frame=np.asarray(mean),std_by_frame=np.asarray(std))

def image_grid(images,indices):
    fig,axes=plt.subplots(1,4,figsize=(14,4),layout="constrained")
    for ax,i in zip(axes,indices):
        handle=ax.imshow(images[i],cmap="gray",vmin=0,vmax=255)
        ax.set_title(f"Original t{5100+i:05d}");ax.set_xlabel("Sensor column");ax.set_ylabel("Sensor row")
    fig.colorbar(handle,ax=axes,label="Original DN",shrink=.7)
    return fig

def search_comparison(fresh,source):
    fig,axes=plt.subplots(1,2,figsize=(13,4),layout="constrained")
    for sl,ax,title in [(slice(0,6),axes[0],"Complete coarse search"),(slice(6,None),axes[1],"Complete additional fine search")]:
        x=np.arange(len(fresh[sl]))
        ax.plot(x,[p["cost"] for p in source[sl]],"o-",label="Reproduced original Torch")
        ax.plot(x,[p["cost"] for p in fresh[sl]],"x--",label="Fresh local NumPy")
        ax.set_xticks(x,[str(p["smoothness_flatfield"]) for p in fresh[sl]])
        ax.set_title(title);ax.set_xlabel("Illumination penalty λ");ax.set_ylabel("Entropy + Fourier cost");ax.grid(alpha=.2);ax.legend()
    return fig

def search_fields(fields,search):
    fig,axes=plt.subplots(3,5,figsize=(14,8),layout="constrained")
    for i,ax in enumerate(axes.flat):
        if i>=len(fields):ax.axis("off");continue
        im=ax.imshow(fields[i],cmap="viridis",vmin=.30,vmax=1.55)
        ax.set_title(f'Fresh λ={search[i]["smoothness_flatfield"]:g}');ax.set_xticks([]);ax.set_yticks([])
    fig.colorbar(im,ax=list(axes.flat),label="Mean-one sensor gain",shrink=.7)
    return fig

def recovered_fields(fresh,source):
    s=fresh["flatfield"];ref=source["reconstruction"][0];b=fresh["baseline"];br=source["baseline"][0]
    fig,axes=plt.subplots(2,3,figsize=(14,8),layout="constrained")
    for ax,a,title in [(axes[0,0],ref,"Reproduced original flatfield"),(axes[0,1],s,"Fresh local flatfield")]:
        im=ax.imshow(a,cmap="viridis",vmin=.30,vmax=1.55);ax.set_title(title)
    fig.colorbar(im,ax=list(axes[0,:2]),label="Dimensionless gain",shrink=.8)
    im=axes[0,2].imshow(s-ref,cmap="coolwarm");axes[0,2].set_title("Fresh − original");fig.colorbar(im,ax=axes[0,2],label="Gain difference")
    axes[1,0].plot(br,label="Original");axes[1,0].plot(b,"--",label="Fresh");axes[1,0].set_ylabel("Temporal baseline / DN");axes[1,0].legend()
    axes[1,1].plot(b-br);axes[1,1].set_ylabel("Baseline difference / DN")
    axes[1,2].plot([h["iterations"] for h in fresh["history"]],"o-");axes[1,2].set_ylabel("Inner updates");axes[1,2].set_xlabel("Reweighting iteration")
    for ax in axes[1,:2]:ax.set_xlabel("Chronological frame index");ax.grid(alpha=.2)
    return fig

def correction_examples(fresh,source):
    fig,axes=plt.subplots(3,4,figsize=(14,7),layout="constrained")
    for j,i in enumerate(fresh["indices"]):
        for row,values,title in [(0,source["corrected"][j],"Original gain correction"),(1,fresh["corrected"][j],"Fresh gain correction")]:
            im=axes[row,j].imshow(values,cmap="gray",vmin=70,vmax=180)
            axes[row,j].set_title(f"{title}, t{5100+i:05d}");axes[row,j].set_xticks([]);axes[row,j].set_yticks([])
        diff=fresh["corrected"][j]-source["corrected"][j]
        dm=axes[2,j].imshow(diff,cmap="coolwarm");axes[2,j].set_title("Fresh − original / DN")
        fig.colorbar(dm,ax=axes[2,j],shrink=.6)
    fig.colorbar(im,ax=list(axes[:2].flat),label="Gain-corrected DN",shrink=.6)
    return fig

def temporal_comparison(images,fresh,source):
    fig,axes=plt.subplots(1,2,figsize=(13,4),layout="constrained")
    for ax,name,label in [(axes[0],"mean_by_frame","Spatial mean / DN"),(axes[1],"std_by_frame","Spatial standard deviation / DN")]:
        # std(uint8_stack) otherwise allocates an unnecessary full float64
        # centered stack. Preserve identical statistics one frame at a time.
        raw=np.asarray([float(np.mean(frame)) if name=="mean_by_frame" else float(np.std(frame)) for frame in images])
        ax.plot(raw,label="Original uncorrected observations")
        ax.plot(source[name],label="Reproduced original gain correction")
        ax.plot(fresh[name],"--",label="Fresh local gain correction")
        ax.set_xlabel("Chronological frame index");ax.set_ylabel(label);ax.grid(alpha=.2);ax.legend(fontsize=8)
    return fig

def quality_controls(coarse,fresh):
    s=fresh["flatfield"];b=fresh["baseline"]
    pairs=[("Fresh",s,b),("Unit gain",np.ones_like(s),coarse.mean(axis=(1,2))),
           ("Constant b",s,np.full_like(b,b.mean())),("Reversed b",s,b[::-1])]
    labels=[];scores=[]
    for label,field,baseline in pairs:labels.append(label);scores.append(quality(coarse,field,baseline))
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout="constrained")
    for ax,key,bound in [(axes[0],"objective_counts",10.09925304298181),(axes[1],"mean_abs_residual_counts",1.6162489661058033)]:
        ax.bar(labels,[p[key] for p in scores],color=["#238b8b","#b8a98c","#d5916b","#9584b5"])
        ax.axhline(bound,color="black",ls="--",label="Public boundary");ax.set_ylabel(key+" / DN");ax.legend()
    return fig,scores
