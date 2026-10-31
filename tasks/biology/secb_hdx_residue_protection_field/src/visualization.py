import matplotlib.pyplot as plt
import numpy as np

def comparison(raw,corrected,energy,uptake,history,upstream,halfinit):
    fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained');r=raw['residue_number'];mask=raw['exchanges']
    axs[0,0].plot(r,energy/1000,label='Fresh local standard');axs[0,0].plot(r,upstream/1000,'--',label='Independent corrected-source V2');axs[0,0].plot(r,halfinit/1000,':',label='V1 uncorrected half-time init (counterfactual)');axs[0,0].scatter(r[~mask],energy[~mask]/1000,marker='x',c='gray',label='Auxiliary TV support (not physical)');axs[0,0].set(xlabel='Protein residue number',ylabel='Conditional energy / auxiliary field (kJ/mol)',title='SecB residue field and declared auxiliary sites');axs[0,0].legend(fontsize=7)
    for color,j in enumerate([0,15,30,45,62]):
        axs[0,1].semilogx(raw['time_s'][1:],corrected[j,1:],'o',ms=3,color=f'C{color}');axs[0,1].semilogx(raw['time_s'][1:],uptake[j,1:],color=f'C{color}',label=f'Peptide {j+1}')
    axs[0,1].set(xlabel='Exposure (s)',ylabel='Corrected deuterium uptake (Da)',title='Measured and fresh modeled representative peptides');axs[0,1].legend(fontsize=7)
    im=axs[1,0].imshow((uptake-corrected),aspect='auto',cmap='coolwarm');axs[1,0].set(xlabel='Native exposure index',ylabel='Peptide index',title='All63 by7 fresh model residuals (Da)');fig.colorbar(im,ax=axs[1,0])
    axs[1,1].semilogy(history[:,0],label='Measured-data mean square');axs[1,1].semilogy(history[:,1],label='Full-domain TV');axs[1,1].semilogy(history.sum(axis=1),label='Total');axs[1,1].set(xlabel='Actual SGD step',ylabel='Objective terms',title='Fresh standard optimization history');axs[1,1].legend(fontsize=7)
    return fig


def correction_limits(raw,corrected,energy):
    from scipy.special import expit
    rate=raw['intrinsic_rates']*expit(-energy/(8.31446261815324*303.15));time=np.logspace(-2,12,300)
    v=.9*(raw['membership']@(-np.expm1(-rate[:,None]*time[None,:])))
    fig,axs=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for j in [0,15,30,45,62]:
        axs[0].semilogx(time,v[j]/raw['capacity'][j],label=f'Peptide {j+1}')
    axs[0].axhline(1,color='black',ls='--',label='Finite-label capacity')
    axs[0].set(xlabel='Hypothetical exposure (s)',ylabel='Predicted uptake / calibrated capacity',title='Independent finite-label limit; extrapolation, not new data');axs[0].legend(fontsize=7)
    axs[1].plot(raw['capacity'],.9*(raw['membership']@(raw['intrinsic_rates']>0)),'o',label='Corrected V2 limit')
    axs[1].plot(raw['capacity'],raw['membership']@(raw['intrinsic_rates']>0),'x',label='V1 missing-fraction limit')
    axs[1].plot(raw['capacity'],raw['capacity'],'k--',label='Independent capacity')
    axs[1].set(xlabel='Calibrated capacity (Da)',ylabel='Infinite-time modeled uptake (Da)',title='All63 peptides: V1 overshoots by 1/0.9');axs[1].legend(fontsize=7)
    return fig
