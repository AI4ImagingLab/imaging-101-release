"""Review plots and independently implemented relative-field error measurements."""
import numpy as np
import matplotlib.pyplot as plt
def field_metrics(estimate,reference,mask):
    x=estimate[mask];y=reference[mask];a=max(0.,np.sum(x*y)/np.sum(x*x));error=0.;energy=0.
    for i in range(0,len(mask)-19,10):
        for j in range(0,mask.shape[1]-19,10):
            m=mask[i:i+20,j:j+20];xx=estimate[i:i+20,j:j+20];yy=reference[i:i+20,j:j+20];xxnorm=np.sum(xx*xx*m)
            local_a=np.sum(xx*yy*m)/xxnorm if xxnorm>1e-5 else 0.
            error+=np.sum(m*(local_a*xx-yy)**2);energy+=np.sum(m*yy*yy)
    return {'ncc':float(np.sum(x*y)/np.sqrt(np.sum(x*x)*np.sum(y*y))),'nrmse':float(np.sqrt(np.mean((a*x-y)**2))/np.ptp(y)),'source_lmse':float(error/energy),'positive_scale':float(a)}
def image_grid(images,titles,columns=3,cmap='gray',limits=None,figsize=None):
    count=len(images);rows=(count+columns-1)//columns
    fig,axes=plt.subplots(rows,columns,figsize=figsize or (4*columns,3.7*rows),squeeze=False,constrained_layout=True)
    for i,ax in enumerate(axes.ravel()):
        if i>=count:ax.set_visible(False);continue
        lim={} if limits is None else {'vmin':limits[0],'vmax':limits[1]}
        panel_cmap=cmap[i] if isinstance(cmap,(tuple,list)) else cmap
        if limits is None and panel_cmap=='coolwarm':
            extent=float(np.nanmax(abs(images[i])));lim={'vmin':-extent,'vmax':extent}
        im=ax.imshow(images[i],cmap=panel_cmap,**lim);ax.set_title(titles[i],fontsize=10);ax.set_axis_off()
        if np.ndim(images[i])==2:fig.colorbar(im,ax=ax,shrink=.76,pad=.02)
    return fig
def scientific_style():
    plt.rcParams.update({'figure.dpi':115,'savefig.dpi':150,'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':False,'axes.titleweight':'semibold','font.family':'DejaVu Sans'})
