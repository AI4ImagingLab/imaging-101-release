import numpy as np
import matplotlib.pyplot as plt
def show_fit(data,result,title):
 alpha=result['reconstruction'][0];k=result['isotropic_stress'][0];pred=np.sqrt(1.5)*alpha.sum(1)+k;y=data['measured_stress'];eps=data['plastic_strain'];fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
 axes[0].plot(y,label='measured',lw=1);axes[0].plot(pred,label='model prediction',lw=1);axes[0].set_xlabel('ordered loading sample');axes[0].set_ylabel('stress [MPa]');axes[0].legend()
 axes[1].plot(pred-y,lw=.8);axes[1].set_xlabel('ordered loading sample');axes[1].set_ylabel('model minus measured [MPa]')
 axes[2].plot(eps,y,label='measured',lw=.8);axes[2].plot(eps,pred,label='model',lw=.8);axes[2].set_xlabel('plastic axial strain [1]');axes[2].set_ylabel('stress [MPa]');axes[2].legend();fig.suptitle(title);return fig
