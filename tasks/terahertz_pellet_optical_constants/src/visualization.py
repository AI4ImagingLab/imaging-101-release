import numpy as np
import matplotlib.pyplot as plt
def metrics(a,b):
    a=np.asarray(a).ravel();b=np.asarray(b).ravel()
    return dict(ncc=float(a@b/np.linalg.norm(a)/np.linalg.norm(b)),nrmse=float(np.sqrt(np.mean((a-b)**2))/np.ptp(b)))
def plot_optical(frequency_hz,fresh,original):
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for row,key in enumerate(('real_index','absorption_coefficient')):
        for channel,name in enumerate(('Polyethylene reference','5 wt% lactose in polyethylene')):
            a=axes[row,channel];a.plot(frequency_hz*1e-12,original[key][0,channel],label='Independent original',color='.65',lw=3);a.plot(frequency_hz*1e-12,fresh[key][0,channel],label='Fresh local',color='#0072B2',lw=1)
            a.set(xlabel='Frequency (THz)',ylabel='Refractive index'if row==0 else 'Apparent absorption (cm⁻¹)',title=name);a.legend()
            if row:a.axhline(0,color='.3',lw=.7)
    fig.suptitle('Measured pellet optical constants — Cambridge author data, CC BY 4.0')
    return fig
