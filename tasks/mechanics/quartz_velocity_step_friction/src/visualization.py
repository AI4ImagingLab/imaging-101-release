import numpy as np
import matplotlib.pyplot as plt
from .physics_model import loadpoint,loadpoint_velocity

def review_figures(data,fresh,upstream):
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    t=data['time'];x=data['displacement']
    axes[0,0].plot(t,x,'.',ms=2,label='Real loading coordinate')
    axes[0,0].plot(t,1000*loadpoint(t),label='Independent smooth loading model')
    axes[0,0].set(xlabel='Window-relative acquisition time [s]',ylabel='Loadpoint displacement [mm]',title='No friction parameters used to locate the velocity step');axes[0,0].legend(fontsize=8)
    axes[0,1].plot(t,1e6*loadpoint_velocity(t));axes[0,1].set(xlabel='Acquisition time [s]',ylabel='Prescribed loading speed [micrometre/s]',title='Positive loading step; measurement jitter is not slider velocity')
    axes[1,0].plot(x,data['friction'],'.',ms=3,color='.55',label='Real detrended observation')
    axes[1,0].plot(x,upstream['friction'][0],lw=2,label='Original-method run')
    axes[1,0].plot(x,fresh['friction'][0],'--',label='Fresh cleaned')
    axes[1,0].set(xlabel='Measured loadpoint displacement [mm]',ylabel='Friction coefficient',title='Full selected914-point inverse fit');axes[1,0].legend(fontsize=8)
    axes[1,1].plot(x,1000*(data['friction']-fresh['friction'][0]),lw=.8)
    axes[1,1].axhline(0,color='k',lw=.7);axes[1,1].set(xlabel='Measured loadpoint displacement [mm]',ylabel='Measured minus fresh [1e-3]',title='Structured residuals limit physical interpretation')
    for ax in axes.flat:ax.grid(alpha=.2)
    fig.tight_layout()
    second,axes=plt.subplots(1,2,figsize=(12,4))
    axes[0].plot(x,upstream['state'][0],label='Original inferred state',lw=2)
    axes[0].plot(x,fresh['state'][0],'--',label='Fresh inferred state');axes[0].set(xlabel='Measured loadpoint displacement [mm]',ylabel='Contact state theta [s]',title='Internal memory is inferred, not measured truth')
    axes[1].plot(x,1e6*upstream['slider_velocity'][0],label='Original inferred slider speed',lw=2)
    axes[1].plot(x,1e6*fresh['slider_velocity'][0],'--',label='Fresh inferred slider speed');axes[1].set(xlabel='Measured loadpoint displacement [mm]',ylabel='Slider velocity [micrometre/s]',title='Machine compliance separates slider and loading speeds')
    for ax in axes:ax.legend(fontsize=8);ax.grid(alpha=.2)
    second.tight_layout()
    third,axes=plt.subplots(1,2,figsize=(10,4))
    positions=np.arange(2)
    axes[0].bar(positions-.16,upstream['reconstruction'][0,:2],.32,label='Original run')
    axes[0].bar(positions+.16,fresh['reconstruction'][0,:2],.32,label='Fresh cleaned');axes[0].set(xticks=positions,xticklabels=['a','b'],ylabel='Dimensionless rate/state coefficient');axes[0].legend(fontsize=8)
    axes[1].bar([0,1],[upstream['reconstruction'][0,2]*1e6,fresh['reconstruction'][0,2]*1e6]);axes[1].set(xticks=[0,1],xticklabels=['Original run','Fresh cleaned'],ylabel='Characteristic slip Dc [micrometre]')
    third.suptitle('Parameters: independent original method versus fresh code, not true values');third.tight_layout()
    return [fig,second,third]
