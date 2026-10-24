"""Display helpers; unclipped arrays remain separate from RGB display mapping."""
import numpy as np
import matplotlib.pyplot as plt
def image_grid(arrays,titles,columns=3,signed=False):
    rows=(len(arrays)+columns-1)//columns;fig,axes=plt.subplots(rows,columns,figsize=(4.7*columns,3.6*rows),squeeze=False,constrained_layout=True)
    for i,ax in enumerate(axes.ravel()):
        if i>=len(arrays):ax.set_visible(False);continue
        a=np.asarray(arrays[i]);kw={}
        if a.ndim==3:
            count=int(np.count_nonzero((a<0)|(a>1)));a=np.clip(a,0,1)
            if count:ax.text(.5,-.025,f'Display only: {count} channel samples outside[0,1]',transform=ax.transAxes,ha='center',fontsize=8)
        else:
            panel_signed=(i in signed) if isinstance(signed,(tuple,list,set)) else signed
            kw['cmap']='coolwarm' if panel_signed else 'gray'
            if panel_signed:extent=float(np.max(abs(a)));kw.update(vmin=-extent,vmax=extent)
        im=ax.imshow(a,**kw);ax.set_title(titles[i],fontsize=10);ax.axis('off')
        if a.ndim==2:fig.colorbar(im,ax=ax,shrink=.8)
    return fig
def scientific_style():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'figure.dpi':115,'savefig.dpi':150,'axes.spines.top':False,'axes.spines.right':False})
