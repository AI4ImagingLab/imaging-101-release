import numpy as np
import matplotlib.pyplot as plt
CONDITIONS=[500,1000,1500,2000,2500,2800]

def spectra_figure(observed,predicted,source):
    fig,axs=plt.subplots(2,3,figsize=(13,7),layout='constrained');t=(np.arange(1000)-33)*.05
    for i,ax in enumerate(axs.flat):
        ax.semilogy(t,observed[i],'.',ms=2,color='gray',label='Real counts')
        ax.semilogy(t,source[i],'--',lw=1.5,label='Independent original C++')
        ax.semilogy(t,predicted[i],lw=1,label='Fresh local LM');ax.axvspan(0,(947-33)*.05,color='green',alpha=.04)
        ax.set(xlabel='ROI-relative time (ns)',ylabel='Counts / 50 ps',title=f'{CONDITIONS[i]}k nominal cumulative counts',ylim=(1,max(observed[i])*1.4))
    axs[0,0].legend(fontsize=7);return fig

def parameter_figure(parameters,source):
    fig,axs=plt.subplots(2,3,figsize=(13,7),layout='constrained');p=np.asarray(parameters).copy();s=np.asarray(source).copy()
    for v in [p,s]:
        for i in range(6):
            g=v[i,6:12].reshape(2,3);v[i,6:12]=g[np.argsort(g[:,0])].ravel()
    for j,color in zip([0,2,4],['C0','C1','C2']):
        axs[0,0].plot(CONDITIONS,p[:,j]*50,'o-',color=color,label=f'Fresh lifetime {j//2+1}');axs[0,0].plot(CONDITIONS,s[:,j]*50,'x--',color=color)
        axs[0,1].plot(CONDITIONS,p[:,j+1],'o-',color=color,label=f'Component {j//2+1}')
    axs[0,0].set(yscale='log',ylabel='Lifetime (ps)',title='Original source crosses / fresh circles');axs[0,0].legend(fontsize=7)
    axs[0,1].set(ylabel='Dimensionless coefficient',title='Fitted material/source intensity coefficients');axs[0,1].legend(fontsize=7)
    for col,label in [(6,'Narrow'),(9,'Broad')]:
        axs[0,2].plot(CONDITIONS,p[:,col]*50,'o-',label=label);axs[0,2].plot(CONDITIONS,s[:,col]*50,'x--')
        axs[1,0].plot(CONDITIONS,p[:,col+2],'o-',label=label)
        axs[1,1].plot(CONDITIONS,p[:,col+1]*50,'o-',label=label)
    axs[0,2].set(ylabel='FWHM (ps)',title='Jointly inferred response widths');axs[0,2].legend(fontsize=7)
    axs[1,0].set(ylabel='Response coefficient',title='Gaussian response weights');axs[1,0].legend(fontsize=7)
    axs[1,1].set(ylabel='ROI-relative Gaussian mean (ps)',title='Joint timing offsets, no alignment');axs[1,1].legend(fontsize=7)
    axs[1,2].plot(CONDITIONS,p[:,12],'o-',label='Fresh');axs[1,2].plot(CONDITIONS,s[:,12],'x--',label='Original');axs[1,2].set(ylabel='Counts / 50 ps',title='Fitted random background');axs[1,2].legend(fontsize=7)
    for ax in axs.flat:ax.set_xlabel('Nominal cumulative counts (thousands)')
    return fig

def residual_figure(observed,predicted):
    fig,axs=plt.subplots(2,3,figsize=(13,6),layout='constrained');idx=np.arange(33,947);t=(idx-33)*.05
    for i,ax in enumerate(axs.flat):
        r=(observed[i,idx]-predicted[i,idx])/np.sqrt(observed[i,idx]+1);ax.plot(t,r,lw=.5);ax.axhline(0,color='black',lw=.7);ax.set(xlabel='ROI-relative time (ns)',ylabel='Weighted residual',title=f'All914selected bins: {CONDITIONS[i]}k')
    return fig

def prompt_figure(prompt,parameters):
    fig,ax=plt.subplots(figsize=(9,4),layout='constrained');pt=(np.arange(len(prompt))-np.argmax(prompt))*5
    ax.plot(pt,prompt/prompt.max(),'.',ms=3,color='black',label='Independent Co-60 prompt, native5ps')
    t=np.linspace(-600,600,1201)
    for i in [0,5]:
        p=parameters[i];mean=np.sum(p[[7,10]]*p[[8,11]])*50;g=np.zeros_like(t)
        for j in [6,9]:
            sigma=p[j]*50/(2*np.sqrt(2*np.log(2)));m=p[j+1]*50-mean;g+=p[j+2]*np.exp(-.5*((t-m)/sigma)**2)/sigma
        ax.plot(t-t[np.argmax(g)],g/g.max(),label=f'Fresh sample response, {CONDITIONS[i]}k')
    ax.set(xlim=(-600,600),xlabel='Time relative to each displayed peak (ps)',ylabel='Peak-normalized shape',title='Independent prompt vs joint sample response: diagnostic alignment only');ax.legend(fontsize=8);return fig
